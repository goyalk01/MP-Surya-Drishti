"""
Rooftop area estimation from segmentation masks.

Converts pixel-level rooftop detection into real-world area estimates.

Core Metrics (Scale-Agnostic):
- Roof area in pixels (exact model detection count)
- Roof coverage percentage (relative to total image area)
- Usable area percentage (adjusted for setbacks & obstacles)

Estimated Metrics (Scale-Dependent):
- Roof area in square metres (calculated ONLY if reliable GSD metadata is available)
- Clearly labels area_m2 as an estimate (`is_estimated: True/False`)
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


class AreaEstimator:
    """
    Estimate rooftop area from binary segmentation masks.

    Args:
        gsd: Ground Sampling Distance in metres per pixel (optional).
            Massachusetts Buildings Dataset: ~1.0 m/px.
            If GSD is 0.0 or None, area_m2 is reported as estimated fallback.
    """

    def __init__(self, gsd: Optional[float] = 1.0) -> None:
        self.gsd = gsd if (gsd is not None and gsd > 0) else None
        self.has_reliable_scale = self.gsd is not None and self.gsd > 0

        if self.has_reliable_scale:
            logger.info("AreaEstimator initialized (GSD=%.2f m/px)", self.gsd)
        else:
            logger.info(
                "AreaEstimator initialized without GSD scale metadata. "
                "Calculations will be returned in pixels and percentages."
            )

    def estimate(
        self,
        mask: np.ndarray,
        obstacle_fraction: float = 0.0,
        setback_fraction: float = 0.05,
    ) -> dict[str, Any]:
        """
        Estimate rooftop area metrics from a binary mask.

        Args:
            mask: Binary mask (H, W) with values {0, 1}.
            obstacle_fraction: Estimated fraction of roof occupied by obstacles.
            setback_fraction: Perimeter safety setback fraction (default 5%).

        Returns:
            Dictionary containing:
                - roof_area_pixels: Total detected roof pixels.
                - roof_area_percent: Rooftop coverage percentage of the image.
                - usable_area_percent: Usable solar area percentage after deductions.
                - total_pixels: Image resolution in total pixels.
                - gsd: GSD used for m² calculation (or None).
                - is_estimated: True if m² calculation is an approximation/estimate.
                - rooftop_area_m2_estimate: Estimated area in m² (or None/0.0).
                - usable_area_m2_estimate: Estimated usable solar area in m².
        """
        roof_area_pixels = int(np.sum(mask > 0))
        total_pixels = mask.shape[0] * mask.shape[1]

        roof_area_percent = (
            (roof_area_pixels / total_pixels) * 100.0 if total_pixels > 0 else 0.0
        )

        # Usable fraction calculation
        usable_fraction = max(0.0, 1.0 - obstacle_fraction - setback_fraction)
        usable_area_percent = roof_area_percent * usable_fraction

        # Square metre estimation (if scale exists)
        if self.has_reliable_scale and self.gsd is not None:
            rooftop_area_m2 = round(roof_area_pixels * (self.gsd ** 2), 2)
            usable_area_m2 = round(rooftop_area_m2 * usable_fraction, 2)
            is_estimated = False
        else:
            rooftop_area_m2 = 0.0
            usable_area_m2 = 0.0
            is_estimated = True

        result = {
            "roof_area_pixels": roof_area_pixels,
            "roof_area_percent": round(roof_area_percent, 2),
            "usable_area_percent": round(usable_area_percent, 2),
            "total_pixels": total_pixels,
            "gsd": self.gsd,
            "is_estimated": is_estimated,
            "rooftop_area_m2_estimate": rooftop_area_m2,
            "usable_area_m2_estimate": usable_area_m2,
        }

        logger.debug(
            "Area estimate: %d px (%.1f%% of image, usable: %.1f%%)",
            roof_area_pixels,
            roof_area_percent,
            usable_area_percent,
        )

        return result
