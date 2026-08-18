"""
Unit tests for native-resolution sliding-window tiled inference, Gaussian blending,
batched tile forwarding, TTA, and single-pass backward compatibility.
"""

from pathlib import Path
import cv2
import numpy as np
import pytest
import torch

from inference.inferencer import (
    SegmentationInferencer,
    SegmentationResult,
    generate_gaussian_weight_map,
)


class TestGaussianBlendingWeightMap:
    """Tests for 2D Gaussian weight map generation and properties."""

    def test_gaussian_weight_map_properties(self):
        """Verify peak at center, minimum at borders, and symmetry."""
        tile_size = 512
        min_weight = 0.15
        weight_map = generate_gaussian_weight_map(
            tile_size=tile_size, min_weight=min_weight
        )

        assert weight_map.shape == (tile_size, tile_size)
        assert not np.isnan(weight_map).any()
        assert not np.isinf(weight_map).any()

        # Center should be close to 1.0 (the maximum)
        center_val = weight_map[tile_size // 2, tile_size // 2]
        assert np.isclose(center_val, 1.0, atol=1e-3)

        # Corners should be close to min_weight (the minimum)
        corner_val = weight_map[0, 0]
        assert np.isclose(corner_val, min_weight, atol=1e-3)
        assert np.all(weight_map >= min_weight)
        assert np.all(weight_map <= 1.0)

        # Symmetry
        assert np.allclose(weight_map, weight_map.T)
        assert np.allclose(weight_map, np.fliplr(weight_map))
        assert np.allclose(weight_map, np.flipud(weight_map))


class TestTiledInferenceEngine:
    """Tests for sliding-window inference, batching, TTA, and shape preservation."""

    def test_tiled_inference_shape_preservation(self, tmp_path):
        """Ensure full-resolution 1500x1500 image produces exact 1500x1500 mask."""
        ckpt_path = Path("outputs/experiments/exp_002/checkpoints/best_loss.pth")
        if not ckpt_path.exists():
            pytest.skip("Checkpoint best_loss.pth not available for inference test")

        img_path = tmp_path / "test_1500.png"
        dummy_img = np.random.randint(0, 256, (1500, 1500, 3), dtype=np.uint8)
        cv2.imwrite(str(img_path), dummy_img)

        inferencer = SegmentationInferencer(
            checkpoint_path=ckpt_path,
            image_size=512,
            tile_stride=256,
            tile_batch_size=8,
            blend_mode="gaussian",
        )

        # 1. Run Tiled Inference
        res_tiled = inferencer.run(img_path, tiled=True, tta=False)

        assert isinstance(res_tiled, SegmentationResult)
        assert res_tiled.binary_mask.shape == (1500, 1500)
        assert res_tiled.overlay_image.shape == (1500, 1500, 3)
        assert res_tiled.metadata["strategy"] == "tiled_native_resolution"
        assert res_tiled.metadata["tiled"] is True

        # 2. Run Single-Pass Inference (Backward Compatibility)
        res_single = inferencer.run(img_path, tiled=False)
        assert res_single.binary_mask.shape == (1500, 1500)
        assert res_single.metadata["strategy"] == "single_pass_resize"
        assert res_single.metadata["tiled"] is False

    def test_tiled_inference_with_tta(self, tmp_path):
        """Ensure TTA inference executes cleanly and returns valid probability/mask bounds."""
        ckpt_path = Path("outputs/experiments/exp_002/checkpoints/best_loss.pth")
        if not ckpt_path.exists():
            pytest.skip("Checkpoint best_loss.pth not available for inference test")

        img_path = tmp_path / "test_800.png"
        dummy_img = np.random.randint(0, 256, (800, 800, 3), dtype=np.uint8)
        cv2.imwrite(str(img_path), dummy_img)

        inferencer = SegmentationInferencer(
            checkpoint_path=ckpt_path,
            image_size=512,
            tile_stride=256,
            tile_batch_size=4,
        )

        res_tta = inferencer.run(img_path, tiled=True, tta=True)
        assert res_tta.binary_mask.shape == (800, 800)
        assert res_tta.metadata["tta"] is True
        assert set(np.unique(res_tta.binary_mask)).issubset({0, 1})

    def test_inferencer_cleaner_configurability(self, tmp_path):
        """Verify that apply_cleaner flag and cleaner_min_area are respected."""
        ckpt_path = Path("outputs/experiments/exp_002/checkpoints/best_loss.pth")
        if not ckpt_path.exists():
            pytest.skip("Checkpoint best_loss.pth not available for inference test")

        img_path = tmp_path / "test_cleaner.png"
        dummy_img = np.zeros((600, 600, 3), dtype=np.uint8)
        cv2.imwrite(str(img_path), dummy_img)

        inferencer = SegmentationInferencer(
            checkpoint_path=ckpt_path,
            apply_cleaner=False,
            cleaner_min_area=10,
        )
        assert inferencer.mask_cleaner.min_region_area == 10

        res_raw = inferencer.run(img_path, apply_cleaner=False)
        assert res_raw.metadata["apply_cleaner"] is False

        res_cleaned = inferencer.run(img_path, apply_cleaner=True)
        assert res_cleaned.metadata["apply_cleaner"] is True
