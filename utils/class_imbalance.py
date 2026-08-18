"""
Class distribution and imbalance measurement utility for aerial segmentation.

Measures exact background and rooftop pixel counts strictly from the training
partition (preventing validation/test leakage) to compute data-driven loss weights.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from preprocessing.dataset_loader import MassachusettsDataset

logger = logging.getLogger(__name__)


def compute_training_class_imbalance(
    root_dir: str | Path = "datasets/massachusetts",
    train_images_dir: str = "train",
    train_masks_dir: str = "train_labels",
    mask_building_value: int = 255,
    extensions: Optional[list[str]] = None,
) -> dict[str, Any]:
    """
    Compute ground-truth class distribution strictly across the training split.

    Args:
        root_dir: Root dataset directory.
        train_images_dir: Training images subdirectory.
        train_masks_dir: Training masks subdirectory.
        mask_building_value: Raw pixel value for building foreground.
        extensions: Allowed file extensions.

    Returns:
        Dictionary containing:
            - total_images: Number of training images analyzed.
            - total_pixels: Total pixel count across training set.
            - background_pixels: Total background pixel count.
            - rooftop_pixels: Total rooftop pixel count.
            - rooftop_percentage: Exact rooftop class percentage.
            - background_percentage: Exact background class percentage.
            - pos_weight: Recommended positive class weight for CE loss (bg / roof).
            - focal_alpha: Recommended alpha balancing parameter for Focal loss.
    """
    _, mask_paths = MassachusettsDataset.discover_pairs(
        root_dir=root_dir,
        images_dir=train_images_dir,
        masks_dir=train_masks_dir,
        extensions=extensions,
    )

    if not mask_paths:
        raise RuntimeError(
            f"No training masks found in {root_dir}/{train_masks_dir}"
        )

    total_bg_pixels = 0
    total_roof_pixels = 0

    threshold = mask_building_value // 2

    for msk_path in mask_paths:
        mask = cv2.imread(str(msk_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            logger.warning("Could not read mask for imbalance calculation: %s", msk_path)
            continue

        binary = (mask >= threshold).astype(np.uint8)
        roof_cnt = int(binary.sum())
        bg_cnt = int(binary.size - roof_cnt)

        total_roof_pixels += roof_cnt
        total_bg_pixels += bg_cnt

    total_pixels = total_bg_pixels + total_roof_pixels
    if total_pixels == 0:
        raise ValueError("Zero total pixels found in training masks.")

    rooftop_pct = (total_roof_pixels / total_pixels) * 100.0
    background_pct = (total_bg_pixels / total_pixels) * 100.0

    # pos_weight for BCE / CE: ratio of background to foreground
    pos_weight = total_bg_pixels / max(total_roof_pixels, 1)

    # focal_alpha for foreground in Focal Loss: (1 - rooftop_fraction) or ratio
    focal_alpha = total_bg_pixels / total_pixels

    result = {
        "total_images": len(mask_paths),
        "total_pixels": total_pixels,
        "background_pixels": total_bg_pixels,
        "rooftop_pixels": total_roof_pixels,
        "rooftop_percentage": round(rooftop_pct, 2),
        "background_percentage": round(background_pct, 2),
        "pos_weight": round(pos_weight, 4),
        "focal_alpha": round(focal_alpha, 4),
    }

    logger.info(
        "Training Class Distribution (%d images): Background=%.2f%%, Rooftop=%.2f%% | "
        "Recommended pos_weight=%.2f, focal_alpha=%.4f",
        result["total_images"],
        result["background_percentage"],
        result["rooftop_percentage"],
        result["pos_weight"],
        result["focal_alpha"],
    )

    return result
