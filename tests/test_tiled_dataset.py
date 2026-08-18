"""
Unit tests for native-resolution patch dataset, tile coordinate generation,
split isolation, and class imbalance measurement.
"""

from pathlib import Path
import cv2
import numpy as np
import pytest
import torch

from preprocessing.tiled_dataset import (
    TiledMassachusettsDataset,
    generate_tile_coordinates,
)
from utils.class_imbalance import compute_training_class_imbalance


class TestTiledDatasetAndCoordinates:
    """Tests for tile grid generation and 100% spatial coverage."""

    def test_tile_coordinates_coverage_standard_1500(self):
        """Verify 1500x1500 image produces expected tiles with full coverage."""
        w, h = 1500, 1500
        tile_size = 512
        stride = 256

        coords = generate_tile_coordinates(
            image_width=w,
            image_height=h,
            tile_size=tile_size,
            stride=stride,
        )

        assert len(coords) > 0

        # Build a coverage mask to verify every pixel is covered
        coverage = np.zeros((h, w), dtype=np.int32)
        for x1, y1, x2, y2 in coords:
            assert x2 - x1 == tile_size
            assert y2 - y1 == tile_size
            assert 0 <= x1 < x2 <= w
            assert 0 <= y1 < y2 <= h
            coverage[y1:y2, x1:x2] += 1

        # Every pixel must be covered by at least 1 tile
        assert np.all(coverage >= 1), "Some pixels were missed in tile coverage"

    def test_tile_coordinates_exact_multiple_and_odd_sizes(self):
        """Verify coverage on exact multiple, smaller than tile, and odd dimensions."""
        test_sizes = [
            (512, 512),
            (1024, 768),
            (300, 400),   # Smaller than tile_size
            (1337, 891),  # Odd prime-like dimensions
        ]

        tile_size = 512
        stride = 256

        for w, h in test_sizes:
            coords = generate_tile_coordinates(w, h, tile_size=tile_size, stride=stride)
            assert len(coords) > 0
            eff_w = max(w, tile_size)
            eff_h = max(h, tile_size)
            coverage = np.zeros((eff_h, eff_w), dtype=np.int32)
            for x1, y1, x2, y2 in coords:
                coverage[y1:y2, x1:x2] += 1
            assert np.all(coverage[:h, :w] >= 1), f"Missed coverage for size ({w}, {h})"

    def test_tiled_dataset_item_shapes_and_values(self, tmp_path):
        """Test dataset returns properly shaped tensors with binary mask labels."""
        img_dir = tmp_path / "images"
        msk_dir = tmp_path / "masks"
        img_dir.mkdir()
        msk_dir.mkdir()

        # Create two 1000x1000 test images and masks
        for i in range(2):
            img_arr = np.random.randint(0, 256, (1000, 1000, 3), dtype=np.uint8)
            # Mask with raw values 0 and 255
            msk_arr = np.zeros((1000, 1000), dtype=np.uint8)
            msk_arr[200:600, 200:600] = 255

            cv2.imwrite(str(img_dir / f"test_{i}.png"), img_arr)
            cv2.imwrite(str(msk_dir / f"test_{i}.png"), msk_arr)

        img_paths = sorted(list(img_dir.glob("*.png")))
        msk_paths = sorted(list(msk_dir.glob("*.png")))

        dataset = TiledMassachusettsDataset(
            image_paths=img_paths,
            mask_paths=msk_paths,
            tile_size=512,
            stride=256,
            mask_building_value=255,
        )

        assert len(dataset) > 0

        sample = dataset[0]
        assert "pixel_values" in sample
        assert "labels" in sample
        assert "tile_coords" in sample
        assert "image_path" in sample

        # Check tensor shapes
        assert sample["pixel_values"].shape == (3, 512, 512)
        assert sample["labels"].shape == (512, 512)

        # Labels must be strictly binary {0, 1}
        unique_labels = torch.unique(sample["labels"]).tolist()
        for val in unique_labels:
            assert val in [0, 1], f"Unexpected label value: {val}"

    def test_tiled_dataset_split_isolation(self, tmp_path):
        """Verify patches strictly preserve source image split boundaries (no leakage)."""
        train_img_dir = tmp_path / "train"
        val_img_dir = tmp_path / "val"
        train_img_dir.mkdir()
        val_img_dir.mkdir()

        # Create 2 train files, 1 val file
        for i in range(2):
            img = np.zeros((600, 600, 3), dtype=np.uint8)
            cv2.imwrite(str(train_img_dir / f"train_{i}.png"), img)

        val_img = np.zeros((600, 600, 3), dtype=np.uint8)
        cv2.imwrite(str(val_img_dir / "val_0.png"), val_img)

        train_imgs = list(train_img_dir.glob("*.png"))
        val_imgs = list(val_img_dir.glob("*.png"))

        train_dataset = TiledMassachusettsDataset(
            image_paths=train_imgs,
            mask_paths=train_imgs,
            tile_size=512,
            stride=256,
        )
        val_dataset = TiledMassachusettsDataset(
            image_paths=val_imgs,
            mask_paths=val_imgs,
            tile_size=512,
            stride=256,
        )

        train_sources = {Path(s["image_path"]).name for s in [train_dataset[i] for i in range(len(train_dataset))]}
        val_sources = {Path(s["image_path"]).name for s in [val_dataset[i] for i in range(len(val_dataset))]}

        # Intersecting sources must be strictly empty (zero leakage)
        assert train_sources.isdisjoint(val_sources)
        assert "val_0.png" not in train_sources
        assert "train_0.png" in train_sources


class TestClassImbalanceUtility:
    """Tests for measuring ground-truth class distribution strictly on training data."""

    def test_compute_training_class_imbalance(self, tmp_path):
        """Verify accurate calculation of background/rooftop pixel distributions."""
        dataset_root = tmp_path / "mock_massachusetts"
        train_imgs = dataset_root / "train"
        train_msks = dataset_root / "train_labels"
        train_imgs.mkdir(parents=True)
        train_msks.mkdir(parents=True)

        # Create mock 100x100 mask with 20% foreground (2000 pixels)
        msk_1 = np.zeros((100, 100), dtype=np.uint8)
        msk_1[:20, :] = 255  # 2000 pixels
        img_1 = np.zeros((100, 100, 3), dtype=np.uint8)

        cv2.imwrite(str(train_imgs / "img_1.png"), img_1)
        cv2.imwrite(str(train_msks / "img_1.png"), msk_1)

        stats = compute_training_class_imbalance(
            root_dir=dataset_root,
            train_images_dir="train",
            train_masks_dir="train_labels",
            mask_building_value=255,
        )

        assert stats["total_images"] == 1
        assert stats["total_pixels"] == 10000
        assert stats["rooftop_pixels"] == 2000
        assert stats["background_pixels"] == 8000
        assert stats["rooftop_percentage"] == 20.0
        assert stats["background_percentage"] == 80.0
        assert stats["pos_weight"] == 4.0  # 8000 / 2000
