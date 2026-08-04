"""
Standalone inference module for the segmentation framework.

Completely independent from the training pipeline. Loads ANY registered
model from a checkpoint (auto-detecting the model type) and runs
segmentation on individual images, returning structured results.

The ``SegmentationResult`` dataclass and ``prediction_report.json``
provide standard contracts consumed directly by downstream modules
(shadow detection, panel placement, solar analytics, FastAPI endpoints).
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

from models.registry import ensure_models_registered, load_model_from_checkpoint
from postprocessing.area_estimator import AreaEstimator
from postprocessing.mask_cleaner import MaskCleaner
from postprocessing.polygon_extractor import PolygonExtractor
from preprocessing.augmentation import AugmentationPipeline
from utils.device_utils import get_device

logger = logging.getLogger(__name__)


@dataclass
class SegmentationResult:
    """
    Structured output from segmentation inference.

    This dataclass is the standard contract between the segmentation
    framework and all downstream consumers (solar analytics, API, UI).

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
        """
        Generate a clean prediction report for downstream Solar Analytics.

        Returns:
            JSON-serializable prediction report matching platform spec.
        """
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
        checkpoint_path: Path to the saved model checkpoint.
        image_size: Input image size for the model.
        confidence_threshold: Threshold for binary mask generation.
        gsd: Ground Sampling Distance in metres per pixel (optional).
        device: Inference device (auto-detected if None).
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        image_size: int = 512,
        confidence_threshold: float = 0.5,
        gsd: Optional[float] = 1.0,
        device: Optional[torch.device] = None,
    ) -> None:
        self.image_size = image_size
        self.confidence_threshold = confidence_threshold
        self.gsd = gsd
        self.device = device or get_device()

        ensure_models_registered()

        # Load model from checkpoint (auto-detects type)
        self.model_instance = load_model_from_checkpoint(
            checkpoint_path, device=self.device
        )
        self.model_instance.eval()
        self.model_instance.confidence_threshold = confidence_threshold

        self.model_type = getattr(self.model_instance, "model_type", "SegFormer").capitalize()

        # Pipeline components
        self.augmentation = AugmentationPipeline(image_size=image_size)
        self.transform = self.augmentation.get_inference_transform()
        self.mask_cleaner = MaskCleaner()
        self.polygon_extractor = PolygonExtractor()
        self.area_estimator = AreaEstimator(gsd=gsd)

        logger.info(
            "SegmentationInferencer ready (model=%s, device=%s, image_size=%d)",
            self.model_type,
            self.device,
            image_size,
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

        # 1. Load image
        original = cv2.imread(str(image_path))
        if original is None:
            raise RuntimeError(f"Could not read image: {image_path}")
        original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        original_h, original_w = original.shape[:2]

        # 2. Preprocess
        resized = cv2.resize(
            original,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_LINEAR,
        )
        transformed = self.transform(image=resized)
        image_normalized = transformed["image"]

        image_tensor = (
            torch.from_numpy(image_normalized)
            .permute(2, 0, 1)
            .float()
            .unsqueeze(0)
            .to(self.device)
        )

        # 3. Model Inference
        with torch.no_grad():
            prediction = self.model_instance.predict(image_tensor)

        binary_mask = prediction["binary_mask"].squeeze(0).cpu().numpy()
        confidence_map = prediction["confidence_map"].squeeze(0).cpu().numpy()

        # 4. Postprocessing
        cleaned_mask = self.mask_cleaner.clean(binary_mask.astype(np.uint8))
        cleaned_mask_original = cv2.resize(
            cleaned_mask,
            (original_w, original_h),
            interpolation=cv2.INTER_NEAREST,
        )

        polygons = self.polygon_extractor.extract(cleaned_mask_original)
        area_metrics = self.area_estimator.estimate(cleaned_mask_original)
        overlay = self._create_overlay(original, cleaned_mask_original)

        # Calculate mean confidence on detected pixels
        rooftop_pixels = cleaned_mask > 0
        confidence = (
            float(confidence_map[rooftop_pixels].mean())
            if rooftop_pixels.any()
            else 0.0
        )

        processing_time = (time.time() - start_time) * 1000  # ms

        result = SegmentationResult(
            original_image=original,
            binary_mask=cleaned_mask_original,
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
                "image_size": self.image_size,
                "original_size": [original_h, original_w],
                "gsd": self.gsd,
                "threshold": self.confidence_threshold,
            },
        )

        logger.info(
            "Inference complete [%s]: %s — %d px (%.1f%%), conf=%.3f, time=%.0f ms",
            self.model_type,
            image_path.name,
            area_metrics["roof_area_pixels"],
            area_metrics["roof_area_percent"],
            confidence,
            processing_time,
        )

        return result

    def _create_overlay(
        self,
        original: np.ndarray,
        mask: np.ndarray,
        color: tuple[int, int, int] = (0, 255, 0),
        alpha: float = 0.4,
    ) -> np.ndarray:
        """Create a colored overlay of the mask on the original image."""
        overlay = original.copy()
        color_mask = np.zeros_like(original)
        color_mask[mask > 0] = color

        mask_region = mask > 0
        overlay[mask_region] = (
            (1 - alpha) * original[mask_region] + alpha * color_mask[mask_region]
        ).astype(np.uint8)

        return overlay


# Backward-compatible alias
RooftopInferencer = SegmentationInferencer
