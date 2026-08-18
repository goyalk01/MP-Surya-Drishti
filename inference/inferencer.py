"""
Standalone inference module for the segmentation framework.

Supports both single-pass whole-image inference and native-resolution sliding-window
tiled inference with Gaussian/Uniform blending, batched tile forwarding, and optional
Test-Time Augmentation (TTA).

The ``SegmentationResult`` dataclass and ``prediction_report.json`` provide
standard contracts consumed directly by downstream modules.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast

from models.registry import ensure_models_registered, load_model_from_checkpoint
from postprocessing.area_estimator import AreaEstimator
from postprocessing.mask_cleaner import MaskCleaner
from postprocessing.polygon_extractor import PolygonExtractor
from preprocessing.augmentation import AugmentationPipeline
from preprocessing.tiled_dataset import generate_tile_coordinates
from utils.checkpoint_manager import find_best_available_checkpoint
from utils.device_utils import get_device

logger = logging.getLogger(__name__)


def generate_gaussian_weight_map(
    tile_size: int = 512,
    sigma: Optional[float] = None,
    min_weight: float = 0.1,
) -> np.ndarray:
    """
    Generate a 2D Gaussian weight window for seamless tile blending.

    Weights are peak at center (1.0) and smoothly decay toward borders,
    bounded away from zero by min_weight to prevent edge pixel starvation.

    Args:
        tile_size: Square tile dimension.
        sigma: Standard deviation of Gaussian distribution. Defaults to tile_size / 4.
        min_weight: Minimum weight floor at tile borders.

    Returns:
        2D numpy array of shape (tile_size, tile_size) with float32 weights.
    """
    if sigma is None:
        sigma = tile_size / 4.0

    y, x = np.ogrid[
        -(tile_size - 1) / 2 : (tile_size - 1) / 2 : complex(0, tile_size),
        -(tile_size - 1) / 2 : (tile_size - 1) / 2 : complex(0, tile_size),
    ]
    gaussian = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))

    # Normalize to [min_weight, 1.0]
    g_min = gaussian.min()
    g_max = gaussian.max()
    if g_max > g_min:
        gaussian = min_weight + (1.0 - min_weight) * (gaussian - g_min) / (g_max - g_min)
    else:
        gaussian = np.ones((tile_size, tile_size), dtype=np.float32)

    return gaussian.astype(np.float32)


@dataclass
class SegmentationResult:
    """
    Structured output from segmentation inference.

    Attributes:
        original_image: Input image as RGB numpy array (H, W, 3).
        binary_mask: Binary segmentation mask (H, W) with values {0, 1}.
        overlay_image: Original with colored overlay (H, W, 3).
        polygon: List of polygon dicts in GeoJSON-style format.
        roof_area_pixels: Total detected area in pixels.
        roof_area_percent: Detected area as percentage of total image.
        usable_area_percent: Usable solar area percentage after deductions.
        confidence: Mean model confidence score on detected rooftop pixels.
        model: Model architecture name (e.g. "SegFormer").
        version: Framework schema version (e.g. "v1").
        rooftop_area_m2_estimate: Estimated area in m² (or 0.0 if unscaled).
        is_estimated: True if m² calculation is an unscaled estimate.
        processing_time_ms: Total inference time in milliseconds.
        image_path: Path to the original input image.
        metadata: Additional metadata dictionary for extensibility.
    """

    original_image: np.ndarray
    binary_mask: np.ndarray
    overlay_image: np.ndarray
    polygon: list[dict[str, Any]]
    roof_area_pixels: int
    roof_area_percent: float
    usable_area_percent: float
    confidence: float
    model: str
    version: str = "v1"
    rooftop_area_m2_estimate: float = 0.0
    is_estimated: bool = True
    processing_time_ms: float = 0.0
    image_path: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_prediction_report(self) -> dict[str, Any]:
        """Generate a clean prediction report for downstream Solar Analytics."""
        return {
            "roof_area_pixels": self.roof_area_pixels,
            "roof_area_percent": round(self.roof_area_percent, 2),
            "usable_area_percent": round(self.usable_area_percent, 2),
            "confidence": round(self.confidence, 4),
            "model": self.model,
            "version": self.version,
            "rooftop_area_m2_estimate": self.rooftop_area_m2_estimate,
            "is_estimated": self.is_estimated,
            "processing_time_ms": round(self.processing_time_ms, 2),
            "polygons_found": len(self.polygon),
            "image_path": self.image_path,
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert full metadata to dictionary format."""
        report = self.to_prediction_report()
        report["polygon"] = self.polygon
        report["mask_shape"] = list(self.binary_mask.shape)
        report["metadata"] = self.metadata
        return report


class SegmentationInferencer:
    """
    Model-agnostic segmentation inference engine with native-resolution tiling support.

    Loads ANY registered model from a checkpoint and runs single-pass or sliding-window
    tiled inference on individual images.

    Args:
        checkpoint_path: Path to saved model checkpoint (auto-selects best if None).
        image_size: Input tile/image size for model (default 512).
        tile_stride: Step size between tiles for tiled inference (default 256).
        tile_batch_size: Number of tiles to batch per forward pass (default 8).
        blend_mode: Blending window type ('gaussian' or 'uniform').
        confidence_threshold: Threshold for binary mask generation (default 0.5).
        gsd: Ground Sampling Distance in metres per pixel (default 1.0).
        device: Inference device (auto-detected if None).
        apply_cleaner: Whether to apply morphological MaskCleaner postprocessing (default False).
        cleaner_min_area: Minimum area for MaskCleaner connected components (default 50).
    """

    def __init__(
        self,
        checkpoint_path: Optional[str | Path] = None,
        image_size: int = 512,
        tile_stride: int = 256,
        tile_batch_size: int = 8,
        blend_mode: str = "gaussian",
        confidence_threshold: float = 0.5,
        gsd: Optional[float] = 1.0,
        device: Optional[torch.device] = None,
        apply_cleaner: bool = False,
        cleaner_min_area: int = 50,
    ) -> None:
        self.image_size = image_size
        self.tile_stride = tile_stride
        self.tile_batch_size = tile_batch_size
        self.blend_mode = blend_mode.lower()
        self.confidence_threshold = confidence_threshold
        self.gsd = gsd
        self.device = device or get_device()
        self.use_amp = self.device.type == "cuda"
        self.apply_cleaner = apply_cleaner
        self.cleaner_min_area = cleaner_min_area

        ensure_models_registered()

        # Resolve checkpoint path
        if checkpoint_path is None:
            checkpoint_path = find_best_available_checkpoint()
        self.checkpoint_path = Path(checkpoint_path)

        # Load model strictly from checkpoint
        self.model_instance = load_model_from_checkpoint(
            self.checkpoint_path, device=self.device
        )
        self.model_instance.eval()
        self.model_instance.confidence_threshold = confidence_threshold

        self.model_type = getattr(self.model_instance, "model_type", "SegFormer").capitalize()

        # Normalization and postprocessing pipelines
        self.aug = AugmentationPipeline(image_size=image_size)
        self.mask_cleaner = MaskCleaner(min_region_area=cleaner_min_area)
        self.polygon_extractor = PolygonExtractor()
        self.area_estimator = AreaEstimator(gsd=gsd)

        # Precompute Gaussian and Uniform weight maps
        self.gaussian_weight = generate_gaussian_weight_map(tile_size=image_size)
        self.uniform_weight = np.ones((image_size, image_size), dtype=np.float32)

        logger.info(
            "SegmentationInferencer ready (model=%s, checkpoint=%s, device=%s, image_size=%d, amp=%s)",
            self.model_type,
            self.checkpoint_path.name,
            self.device,
            image_size,
            self.use_amp,
        )

    def _predict_batch_with_tta(
        self,
        batch_tensors: torch.Tensor,
        tta_enabled: bool = False,
    ) -> torch.Tensor:
        """
        Run forward pass on a batch of tile tensors with optional TTA.

        Args:
            batch_tensors: Tensor of shape (B, 3, H, W) on device.
            tta_enabled: If True, averages predictions over isometric transforms.

        Returns:
            Probability tensor of shape (B, num_classes, H, W).
        """
        self.model_instance.eval()
        with torch.no_grad():
            with autocast(device_type="cuda", enabled=self.use_amp):
                # 1. Base forward pass
                out_base = self.model_instance.forward(batch_tensors)
                prob_base = F.softmax(out_base["upsampled_logits"], dim=1)

                if not tta_enabled:
                    return prob_base

                # 2. Horizontal Flip
                hflip_tensors = torch.flip(batch_tensors, dims=[3])
                out_hflip = self.model_instance.forward(hflip_tensors)
                prob_hflip = torch.flip(
                    F.softmax(out_hflip["upsampled_logits"], dim=1), dims=[3]
                )

                # 3. Vertical Flip
                vflip_tensors = torch.flip(batch_tensors, dims=[2])
                out_vflip = self.model_instance.forward(vflip_tensors)
                prob_vflip = torch.flip(
                    F.softmax(out_vflip["upsampled_logits"], dim=1), dims=[2]
                )

                # 4. Rotation 90 degrees
                rot90_tensors = torch.rot90(batch_tensors, k=1, dims=[2, 3])
                out_rot90 = self.model_instance.forward(rot90_tensors)
                prob_rot90 = torch.rot90(
                    F.softmax(out_rot90["upsampled_logits"], dim=1), k=-1, dims=[2, 3]
                )

                # Average across 4 augmentations
                prob_avg = (prob_base + prob_hflip + prob_vflip + prob_rot90) / 4.0
                return prob_avg

    def _run_tiled_inference(
        self,
        original_image: np.ndarray,
        tile_size: int,
        stride: int,
        batch_size: int,
        blend_mode: str,
        tta_enabled: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Execute sliding-window inference with probability accumulation and blending.

        Returns:
            Tuple of (binary_mask (H, W), confidence_map (H, W)) at original resolution.
        """
        h, w = original_image.shape[:2]
        num_classes = getattr(self.model_instance, "num_labels", 2)

        # Coordinate grid for full image coverage
        coords = generate_tile_coordinates(
            image_width=w,
            image_height=h,
            tile_size=tile_size,
            stride=stride,
        )

        # Select blend weight map
        if blend_mode == "uniform":
            weight_window = self.uniform_weight
        else:
            weight_window = self.gaussian_weight

        # Accumulation buffers: (num_classes, H, W) and (H, W)
        prob_accumulator = np.zeros((num_classes, h, w), dtype=np.float32)
        weight_accumulator = np.zeros((h, w), dtype=np.float32)

        val_transform = self.aug.get_val_transform()

        # Process in batches
        for i in range(0, len(coords), batch_size):
            batch_coords = coords[i : i + batch_size]
            batch_tiles = []

            for x1, y1, x2, y2 in batch_coords:
                crop = original_image[y1:y2, x1:x2]
                # Pad if tile is smaller than tile_size (small image edge case)
                if crop.shape[0] < tile_size or crop.shape[1] < tile_size:
                    pad_h = max(0, tile_size - crop.shape[0])
                    pad_w = max(0, tile_size - crop.shape[1])
                    crop = cv2.copyMakeBorder(
                        crop, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT
                    )

                augmented = val_transform(image=crop)
                t = torch.from_numpy(augmented["image"]).permute(2, 0, 1).float() / 255.0
                batch_tiles.append(t)

            batch_tensor = torch.stack(batch_tiles, dim=0).to(self.device)

            # Predict batch with optional TTA
            batch_probs = self._predict_batch_with_tta(
                batch_tensor, tta_enabled=tta_enabled
            )
            batch_probs_np = batch_probs.cpu().numpy()

            # Accumulate each tile into full-resolution map
            for b_idx, (x1, y1, x2, y2) in enumerate(batch_coords):
                tile_p = batch_probs_np[b_idx]  # (num_classes, tile_size, tile_size)
                crop_h = y2 - y1
                crop_w = x2 - x1

                tile_p_valid = tile_p[:, :crop_h, :crop_w]
                weight_valid = weight_window[:crop_h, :crop_w]

                prob_accumulator[:, y1:y2, x1:x2] += tile_p_valid * weight_valid[np.newaxis, :, :]
                weight_accumulator[y1:y2, x1:x2] += weight_valid

        # Normalize by total accumulated weights
        weight_safe = np.maximum(weight_accumulator, 1e-7)
        normalized_probs = prob_accumulator / weight_safe[np.newaxis, :, :]

        foreground_prob = normalized_probs[1]
        binary_mask = (foreground_prob >= self.confidence_threshold).astype(np.uint8)

        return binary_mask, foreground_prob

    def run(
        self,
        image_path: str | Path,
        tiled: bool = True,
        tta: bool = False,
        stride: Optional[int] = None,
        batch_size: Optional[int] = None,
        blend_mode: Optional[str] = None,
        apply_cleaner: Optional[bool] = None,
    ) -> SegmentationResult:
        """
        Run full segmentation pipeline on a single image.

        Args:
            image_path: Path to the input image file.
            tiled: If True, uses native-resolution sliding-window tiled inference.
                   If False, uses legacy single-pass whole-image resize.
            tta: Whether to apply Test-Time Augmentation (flips + rotations).
            stride: Tile step size (defaults to instance setting, e.g. 256).
            batch_size: Tile batch size (defaults to instance setting, e.g. 8).
            blend_mode: 'gaussian' or 'uniform' blending.
            apply_cleaner: Override for applying morphological MaskCleaner.

        Returns:
            SegmentationResult object.
        """
        start_time = time.time()
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # 1. Load image and convert to RGB
        original = cv2.imread(str(image_path))
        if original is None:
            raise RuntimeError(f"Could not read image file: {image_path}")
        original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        original_h, original_w = original.shape[:2]

        stride = stride or self.tile_stride
        batch_size = batch_size or self.tile_batch_size
        blend_mode = (blend_mode or self.blend_mode).lower()
        use_cleaner = self.apply_cleaner if apply_cleaner is None else apply_cleaner

        if tiled:
            # Native-Resolution Sliding-Window Tiled Inference
            binary_mask_raw, confidence_map = self._run_tiled_inference(
                original_image=original,
                tile_size=self.image_size,
                stride=stride,
                batch_size=batch_size,
                blend_mode=blend_mode,
                tta_enabled=tta,
            )
            strategy = "tiled_native_resolution"
        else:
            # Legacy Single-Pass 512x512 Resize Inference
            resized = cv2.resize(
                original,
                (self.image_size, self.image_size),
                interpolation=cv2.INTER_LINEAR,
            )
            augmented = self.aug.get_val_transform()(image=resized)
            image_tensor = torch.from_numpy(augmented["image"]).permute(2, 0, 1).float() / 255.0
            image_tensor = image_tensor.unsqueeze(0).to(self.device)

            self.model_instance.eval()
            with torch.no_grad():
                with autocast(device_type="cuda", enabled=self.use_amp):
                    prediction = self.model_instance.predict(image_tensor)

            pred_mask_512 = prediction["binary_mask"].squeeze(0).cpu().numpy().astype(np.uint8)
            conf_512 = prediction["confidence_map"].squeeze(0).cpu().numpy()

            binary_mask_raw = cv2.resize(
                pred_mask_512, (original_w, original_h), interpolation=cv2.INTER_NEAREST
            )
            confidence_map = cv2.resize(
                conf_512, (original_w, original_h), interpolation=cv2.INTER_LINEAR
            )
            strategy = "single_pass_resize"

        # 4. Postprocessing (Primary: raw mask; Optional: MaskCleaner diagnostic)
        if use_cleaner:
            final_mask = self.mask_cleaner.clean(binary_mask_raw)
        else:
            final_mask = binary_mask_raw

        polygons = self.polygon_extractor.extract(final_mask)
        area_metrics = self.area_estimator.estimate(final_mask)
        overlay = self._create_overlay(original, final_mask)

        # Calculate mean confidence score on detected rooftop pixels
        rooftop_pixels = final_mask > 0
        confidence = (
            float(confidence_map[rooftop_pixels].mean())
            if rooftop_pixels.any()
            else 0.0
        )

        processing_time = (time.time() - start_time) * 1000  # ms

        result = SegmentationResult(
            original_image=original,
            binary_mask=final_mask,
            overlay_image=overlay,
            polygon=polygons,
            roof_area_pixels=area_metrics["roof_area_pixels"],
            roof_area_percent=area_metrics["roof_area_percent"],
            usable_area_percent=area_metrics["usable_area_percent"],
            confidence=confidence,
            model=self.model_type,
            version="v1",
            rooftop_area_m2_estimate=area_metrics["rooftop_area_m2_estimate"],
            is_estimated=area_metrics["is_estimated"],
            processing_time_ms=processing_time,
            image_path=str(image_path),
            metadata={
                "checkpoint": str(self.checkpoint_path),
                "strategy": strategy,
                "tiled": tiled,
                "tile_size": self.image_size,
                "stride": stride if tiled else None,
                "blend_mode": blend_mode if tiled else None,
                "tta": tta,
                "apply_cleaner": use_cleaner,
                "input_resolution": f"{original_w}x{original_h}",
            },
        )

        logger.info(
            "Inference complete (%s): %s (roof_pixels=%d, roof_pct=%.1f%%, conf=%.3f, time=%.1fms)",
            strategy,
            image_path.name,
            result.roof_area_pixels,
            result.roof_area_percent,
            result.confidence,
            result.processing_time_ms,
        )

        return result

    def _create_overlay(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        color: tuple[int, int, int] = (0, 255, 0),
        alpha: float = 0.45,
    ) -> np.ndarray:
        """Create a colored overlay of the mask on the original image."""
        overlay = image.copy()
        color_mask = np.zeros_like(image)
        color_mask[mask > 0] = color

        mask_region = mask > 0
        overlay[mask_region] = (
            (1 - alpha) * image[mask_region] + alpha * color_mask[mask_region]
        ).astype(np.uint8)

        return overlay


# Backward-compatible alias
RooftopInferencer = SegmentationInferencer
