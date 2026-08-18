"""
Standalone inference module for the segmentation framework.

Completely independent from the training pipeline. Loads ANY registered
model from a checkpoint (auto-detecting the model type) and runs
segmentation on individual images, returning structured results.

The ``SegmentationResult`` dataclass and ``prediction_report.json``
provide standard contracts consumed directly by downstream modules.
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
from torch.amp import autocast

from models.registry import ensure_models_registered, load_model_from_checkpoint
from postprocessing.area_estimator import AreaEstimator
from postprocessing.mask_cleaner import MaskCleaner
from postprocessing.polygon_extractor import PolygonExtractor
from preprocessing.augmentation import AugmentationPipeline
from preprocessing.normalizer import ImageNormalizer
from utils.checkpoint_manager import find_best_available_checkpoint
from utils.device_utils import get_device

logger = logging.getLogger(__name__)


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
    Model-agnostic segmentation inference engine.

    Loads ANY registered model from a checkpoint and runs segmentation
    on individual images. The model type is auto-detected from the checkpoint file.

    Args:
        checkpoint_path: Path to the saved model checkpoint (auto-selects best if None).
        image_size: Input image size for the model.
        confidence_threshold: Threshold for binary mask generation.
        gsd: Ground Sampling Distance in metres per pixel (optional).
        device: Inference device (auto-detected if None).
        apply_cleaner: Whether to apply morphological MaskCleaner postprocessing (default False).
    """

    def __init__(
        self,
        checkpoint_path: Optional[str | Path] = None,
        image_size: int = 512,
        confidence_threshold: float = 0.5,
        gsd: Optional[float] = 1.0,
        device: Optional[torch.device] = None,
        apply_cleaner: bool = False,
    ) -> None:
        self.image_size = image_size
        self.confidence_threshold = confidence_threshold
        self.gsd = gsd
        self.device = device or get_device()
        self.use_amp = self.device.type == "cuda"
        self.apply_cleaner = apply_cleaner

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
        self.mask_cleaner = MaskCleaner(min_region_area=50)
        self.polygon_extractor = PolygonExtractor()
        self.area_estimator = AreaEstimator(gsd=gsd)

        logger.info(
            "SegmentationInferencer ready (model=%s, checkpoint=%s, device=%s, image_size=%d, amp=%s)",
            self.model_type,
            self.checkpoint_path.name,
            self.device,
            image_size,
            self.use_amp,
        )

    def run(self, image_path: str | Path) -> SegmentationResult:
        """
        Run full segmentation pipeline on a single image.

        Args:
            image_path: Path to the input image file.

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

        # 2. Resize & Normalize
        resized = cv2.resize(
            original,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_LINEAR,
        )
        augmented = self.aug.get_val_transform()(image=resized)
        image_tensor = torch.from_numpy(augmented["image"]).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0).to(self.device)

        # 3. Model Inference (eval mode, no_grad, AMP if CUDA)
        self.model_instance.eval()
        with torch.no_grad():
            with autocast(device_type="cuda", enabled=self.use_amp):
                prediction = self.model_instance.predict(image_tensor)

        binary_mask = prediction["binary_mask"].squeeze(0).cpu().numpy().astype(np.uint8)
        confidence_map = prediction["confidence_map"].squeeze(0).cpu().numpy()

        # 4. Postprocessing (Primary: raw reconstruction; Optional: cleaned)
        if self.apply_cleaner:
            processed_mask_512 = self.mask_cleaner.clean(binary_mask)
        else:
            processed_mask_512 = binary_mask

        final_mask_original = cv2.resize(
            processed_mask_512,
            (original_w, original_h),
            interpolation=cv2.INTER_NEAREST,
        )

        polygons = self.polygon_extractor.extract(final_mask_original)
        area_metrics = self.area_estimator.estimate(final_mask_original)
        overlay = self._create_overlay(original, final_mask_original)

        # Calculate mean confidence score on rooftop pixels
        rooftop_pixels = binary_mask > 0
        confidence = (
            float(confidence_map[rooftop_pixels].mean())
            if rooftop_pixels.any()
            else 0.0
        )

        processing_time = (time.time() - start_time) * 1000  # ms

        result = SegmentationResult(
            original_image=original,
            binary_mask=final_mask_original,
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
                "apply_cleaner": self.apply_cleaner,
                "input_resolution": f"{original_w}x{original_h}",
                "model_resolution": f"{self.image_size}x{self.image_size}",
            },
        )

        logger.info(
            "Inference complete: %s (roof_pixels=%d, roof_pct=%.1f%%, usable_pct=%.1f%%, conf=%.3f, time=%.1fms)",
            image_path.name,
            result.roof_area_pixels,
            result.roof_area_percent,
            result.usable_area_percent,
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
