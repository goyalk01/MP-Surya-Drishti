"""
Polygon extraction from binary segmentation masks.

Converts binary masks into GeoJSON-compatible polygon representations
using OpenCV contour detection and Douglas-Peucker simplification.

The output format is designed to be directly consumable by:
- Leaflet / MapLibre (frontend map visualization)
- Shadow Detection module (Phase 2)
- Panel Placement Optimization module (Phase 2)
- Report generation (PDF export)
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class PolygonExtractor:
    """
    Extract polygon boundaries from binary segmentation masks.

    Uses OpenCV contour detection with Douglas-Peucker polygon
    simplification to produce clean, simplified polygon coordinates.

    Args:
        epsilon_factor: Simplification tolerance as a fraction of
            contour perimeter. Higher values = more simplification.
            Default 0.005 (0.5% of perimeter).
        min_contour_area: Minimum contour area in pixels to be
            included. Filters out tiny contours.
    """

    def __init__(
        self,
        epsilon_factor: float = 0.005,
        min_contour_area: int = 200,
    ) -> None:
        self.epsilon_factor = epsilon_factor
        self.min_contour_area = min_contour_area

        logger.debug(
            "PolygonExtractor initialized (epsilon=%.4f, min_area=%d)",
            epsilon_factor,
            min_contour_area,
        )

    def extract(
        self,
        mask: np.ndarray,
    ) -> list[dict]:
        """
        Extract polygons from a binary mask.

        Args:
            mask: Binary mask (H, W) with values {0, 1} or {0, 255}.

        Returns:
            List of GeoJSON-style polygon dictionaries. Each dict has:
                - type: "Polygon"
                - coordinates: List of [x, y] coordinate pairs.
                - area_px: Area of this polygon in pixels.
                - bbox: Bounding box [x, y, width, height].
                - perimeter: Perimeter in pixels.
        """
        # Ensure uint8 with values 0/255 for contour detection
        if mask.max() <= 1:
            mask_255 = (mask * 255).astype(np.uint8)
        else:
            mask_255 = mask.astype(np.uint8)

        # Find contours (external only — no nested contours)
        contours, hierarchy = cv2.findContours(
            mask_255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        polygons = []

        for contour in contours:
            area = cv2.contourArea(contour)

            # Filter small contours
            if area < self.min_contour_area:
                continue

            # Simplify polygon using Douglas-Peucker
            perimeter = cv2.arcLength(contour, closed=True)
            epsilon = self.epsilon_factor * perimeter
            simplified = cv2.approxPolyDP(contour, epsilon, closed=True)

            # Convert to coordinate list: [(x, y), ...]
            coordinates = simplified.squeeze().tolist()

            # Handle edge case: single point or line
            if not isinstance(coordinates[0], list):
                continue  # Skip degenerate polygons

            # Close the polygon (first point = last point)
            if coordinates[0] != coordinates[-1]:
                coordinates.append(coordinates[0])

            # Bounding box
            x, y, w, h = cv2.boundingRect(contour)

            polygon_dict = {
                "type": "Polygon",
                "coordinates": coordinates,
                "area_px": int(area),
                "bbox": [int(x), int(y), int(w), int(h)],
                "perimeter": float(perimeter),
                "num_vertices": len(coordinates),
            }

            polygons.append(polygon_dict)

        # Sort by area (largest first)
        polygons.sort(key=lambda p: p["area_px"], reverse=True)

        logger.info(
            "Extracted %d polygons from mask (total area: %d px)",
            len(polygons),
            sum(p["area_px"] for p in polygons),
        )

        return polygons

    def extract_largest(
        self,
        mask: np.ndarray,
    ) -> Optional[dict]:
        """
        Extract only the largest polygon from a mask.

        Useful when you expect a single primary rooftop.

        Args:
            mask: Binary mask (H, W).

        Returns:
            The largest polygon dict, or None if no polygons found.
        """
        polygons = self.extract(mask)
        return polygons[0] if polygons else None

    @staticmethod
    def polygon_to_mask(
        polygon: dict,
        image_shape: tuple[int, int],
    ) -> np.ndarray:
        """
        Convert a polygon back to a binary mask.

        Useful for downstream modules that need a mask from polygon data.

        Args:
            polygon: GeoJSON-style polygon dict with 'coordinates'.
            image_shape: Output mask shape (H, W).

        Returns:
            Binary mask (H, W) with the polygon filled.
        """
        mask = np.zeros(image_shape, dtype=np.uint8)
        coords = np.array(polygon["coordinates"], dtype=np.int32)
        cv2.fillPoly(mask, [coords], color=1)
        return mask
