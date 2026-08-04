"""
Unit tests for postprocessing modules (mask cleaner, polygon extractor, area estimator).
"""

import numpy as np
import pytest

from postprocessing.area_estimator import AreaEstimator
from postprocessing.mask_cleaner import MaskCleaner
from postprocessing.polygon_extractor import PolygonExtractor


class TestMaskCleaner:
    """Tests for the MaskCleaner class."""

    def test_removes_small_noise(self):
        """Small isolated pixels should be removed."""
        cleaner = MaskCleaner(min_region_area=50)

        mask = np.zeros((100, 100), dtype=np.uint8)
        # Large region (should be kept)
        mask[20:60, 20:60] = 1
        # Tiny noise (should be removed)
        mask[80, 80] = 1
        mask[90, 90] = 1

        cleaned = cleaner.clean(mask)

        # Large region preserved
        assert cleaned[40, 40] == 1
        # Noise removed
        assert cleaned[80, 80] == 0
        assert cleaned[90, 90] == 0

    def test_preserves_large_regions(self):
        """Large regions should not be affected by cleaning."""
        cleaner = MaskCleaner(min_region_area=10)

        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:90, 10:90] = 1

        cleaned = cleaner.clean(mask)

        # Most of the large region should be preserved
        assert cleaned[50, 50] == 1

    def test_handles_empty_mask(self):
        """Should handle an all-zero mask without error."""
        cleaner = MaskCleaner()
        mask = np.zeros((100, 100), dtype=np.uint8)
        cleaned = cleaner.clean(mask)
        assert cleaned.sum() == 0


class TestPolygonExtractor:
    """Tests for the PolygonExtractor class."""

    def test_extracts_single_polygon(self):
        """Should extract one polygon from a single rectangular region."""
        extractor = PolygonExtractor(min_contour_area=10)

        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:80, 20:80] = 1

        polygons = extractor.extract(mask)

        assert len(polygons) == 1
        assert polygons[0]["type"] == "Polygon"
        assert polygons[0]["area_px"] > 0
        assert len(polygons[0]["coordinates"]) >= 4  # At least a quad

    def test_extracts_multiple_polygons(self):
        """Should extract separate polygons for disjoint regions."""
        extractor = PolygonExtractor(min_contour_area=10)

        mask = np.zeros((200, 200), dtype=np.uint8)
        mask[10:40, 10:40] = 1   # Region 1
        mask[100:180, 100:180] = 1  # Region 2

        polygons = extractor.extract(mask)

        assert len(polygons) == 2

    def test_empty_mask_returns_empty(self):
        """Should return empty list for an all-zero mask."""
        extractor = PolygonExtractor()
        mask = np.zeros((100, 100), dtype=np.uint8)
        polygons = extractor.extract(mask)
        assert len(polygons) == 0

    def test_polygon_to_mask_roundtrip(self):
        """Converting polygon back to mask should approximate the original."""
        extractor = PolygonExtractor(min_contour_area=10, epsilon_factor=0.001)

        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:80, 20:80] = 1

        polygons = extractor.extract(mask)
        reconstructed = PolygonExtractor.polygon_to_mask(
            polygons[0], (100, 100)
        )

        # Should have substantial overlap
        overlap = np.logical_and(mask, reconstructed).sum()
        original_area = mask.sum()
        assert overlap / original_area > 0.8  # At least 80% overlap


class TestAreaEstimator:
    """Tests for the AreaEstimator class."""

    def test_area_in_m2(self):
        """Should correctly convert pixels to m² using GSD."""
        estimator = AreaEstimator(gsd=1.0)

        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[0:10, 0:10] = 1  # 100 pixels

        result = estimator.estimate(mask)

        assert result["area_px"] == 100
        assert result["area_m2"] == 100.0  # 100 × 1.0² = 100
        assert abs(result["area_pct"] - 1.0) < 0.01  # 100/10000 = 1%

    def test_area_unknown_gsd(self):
        """Should report area_m2=0 when GSD is unknown."""
        estimator = AreaEstimator(gsd=0.0)

        mask = np.ones((50, 50), dtype=np.uint8)
        result = estimator.estimate(mask)

        assert result["area_m2"] == 0.0
        assert result["area_px"] == 2500

    def test_usable_area_adjustment(self):
        """Usable area should deduct obstacles, shadows, and setbacks."""
        estimator = AreaEstimator(gsd=1.0)

        result = estimator.usable_area_adjustment(
            area_m2=100.0,
            obstacle_fraction=0.10,
            shadow_fraction=0.15,
            setback_fraction=0.05,
        )

        # 100 - 10 - 15 - 5 = 70
        assert abs(result["usable_area_m2"] - 70.0) < 0.01
