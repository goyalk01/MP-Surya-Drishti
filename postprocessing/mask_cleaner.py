"""
Mask cleaning and refinement for segmentation outputs.

Applies morphological operations to remove noise, fill holes, and
filter out small isolated regions from raw segmentation masks.

Designed to be reusable by any segmentation module in the platform
(rooftop segmentation, shadow detection, panel health).
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class MaskCleaner:
    """
    Clean and refine binary segmentation masks using morphological operations.

    The cleaning pipeline:
    1. Morphological opening — removes small noise (false positives).
    2. Morphological closing — fills small holes in detected regions.
    3. Connected component filtering — removes tiny isolated regions
       below a minimum area threshold.

    Args:
        opening_kernel_size: Kernel size for morphological opening.
        closing_kernel_size: Kernel size for morphological closing.
        min_region_area: Minimum area (in pixels) for a connected component
            to be kept. Smaller regions are removed.
        opening_iterations: Number of opening iterations.
        closing_iterations: Number of closing iterations.
    """

    def __init__(
        self,
        opening_kernel_size: int = 5,
        closing_kernel_size: int = 5,
        min_region_area: int = 500,
        opening_iterations: int = 1,
        closing_iterations: int = 2,
    ) -> None:
        self.opening_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (opening_kernel_size, opening_kernel_size)
        )
        self.closing_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (closing_kernel_size, closing_kernel_size)
        )
        self.min_region_area = min_region_area
        self.opening_iterations = opening_iterations
        self.closing_iterations = closing_iterations

        logger.debug(
            "MaskCleaner initialized (open_k=%d, close_k=%d, min_area=%d)",
            opening_kernel_size,
            closing_kernel_size,
            min_region_area,
        )

    def clean(self, mask: np.ndarray) -> np.ndarray:
        """
        Apply the full cleaning pipeline to a binary mask.

        Args:
            mask: Binary mask (H, W) with values {0, 1} or {0, 255}.

        Returns:
            Cleaned binary mask (H, W) with values {0, 1}.
        """
        # Ensure binary 0/1
        if mask.max() > 1:
            mask = (mask > 127).astype(np.uint8)
        else:
            mask = mask.astype(np.uint8)

        # Step 1: Morphological opening — remove noise
        cleaned = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            self.opening_kernel,
            iterations=self.opening_iterations,
        )

        # Step 2: Morphological closing — fill holes
        cleaned = cv2.morphologyEx(
            cleaned,
            cv2.MORPH_CLOSE,
            self.closing_kernel,
            iterations=self.closing_iterations,
        )

        # Step 3: Remove small connected components
        cleaned = self._filter_small_regions(cleaned)

        pixels_removed = int(np.int64(mask.sum()) - np.int64(cleaned.sum()))
        logger.debug(
            "Mask cleaned: %d pixels removed (%.1f%%)",
            pixels_removed,
            100 * pixels_removed / max(int(mask.sum()), 1),
        )

        return cleaned

    def _filter_small_regions(self, mask: np.ndarray) -> np.ndarray:
        """
        Remove connected components smaller than the minimum area.

        Args:
            mask: Binary mask (H, W).

        Returns:
            Filtered mask with small regions removed.
        """
        # Find connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )

        # Create output mask
        filtered = np.zeros_like(mask)

        for label_id in range(1, num_labels):  # Skip background (0)
            area = stats[label_id, cv2.CC_STAT_AREA]
            if area >= self.min_region_area:
                filtered[labels == label_id] = 1

        return filtered

    def smooth_edges(
        self,
        mask: np.ndarray,
        blur_size: int = 5,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """
        Smooth mask edges using Gaussian blur and re-thresholding.

        This is optional post-processing that produces smoother polygon
        boundaries. Useful for visualization and report generation.

        Args:
            mask: Binary mask (H, W).
            blur_size: Gaussian blur kernel size.
            threshold: Re-thresholding value after blur.

        Returns:
            Smoothed binary mask.
        """
        # Convert to float for blurring
        mask_float = mask.astype(np.float32)
        blurred = cv2.GaussianBlur(
            mask_float, (blur_size, blur_size), sigmaX=0
        )
        smoothed = (blurred > threshold).astype(np.uint8)

        return smoothed
