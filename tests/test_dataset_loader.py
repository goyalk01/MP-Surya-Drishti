"""
Unit tests for the dataset loader module supporting official dataset splits.
"""

import pytest

pytest.importorskip("cv2")
from preprocessing.dataset_loader import MassachusettsDataset


class TestMassachusettsDataset:
    """Tests for the MassachusettsDataset class."""

    def test_mismatched_lengths_raises(self):
        """Dataset raises ValueError when image and mask lists differ."""
        with pytest.raises(ValueError, match="does not match"):
            MassachusettsDataset(
                image_paths=["a.png", "b.png"],
                mask_paths=["a.png"],
            )

    def test_discover_pairs_missing_dir(self, tmp_path):
        """discover_pairs raises FileNotFoundError for missing directory."""
        with pytest.raises(FileNotFoundError):
            MassachusettsDataset.discover_pairs(
                root_dir=tmp_path,
                images_dir="nonexistent_train",
                masks_dir="nonexistent_labels",
            )

    def test_discover_pairs_official_split_names(self, tmp_path):
        """discover_pairs correctly matches image-mask pairs in official split folders."""
        train_dir = tmp_path / "train"
        train_labels_dir = tmp_path / "train_labels"
        train_dir.mkdir()
        train_labels_dir.mkdir()

        for name in ["img1", "img2", "img3"]:
            (train_dir / f"{name}.png").write_bytes(b"fake_image")
            (train_labels_dir / f"{name}.png").write_bytes(b"fake_mask")

        (train_dir / "unmatched.png").write_bytes(b"fake_image")

        image_paths, mask_paths = MassachusettsDataset.discover_pairs(
            root_dir=tmp_path,
            images_dir="train",
            masks_dir="train_labels",
        )

        assert len(image_paths) == 3
        assert len(mask_paths) == 3
        assert all(ip.stem == mp.stem for ip, mp in zip(image_paths, mask_paths))

    def test_discover_all_splits(self, tmp_path):
        """discover_all_splits discovers train, val, and test partitions."""
        for split, label_split in [("train", "train_labels"), ("val", "val_labels"), ("test", "test_labels")]:
            img_d = tmp_path / split
            msk_d = tmp_path / label_split
            img_d.mkdir()
            msk_d.mkdir()
            (img_d / "sample.png").write_bytes(b"fake")
            (msk_d / "sample.png").write_bytes(b"fake")

        splits = MassachusettsDataset.discover_all_splits(root_dir=tmp_path)

        assert "train" in splits
        assert "val" in splits
        assert "test" in splits
        assert len(splits["train"][0]) == 1
        assert len(splits["val"][0]) == 1
        assert len(splits["test"][0]) == 1

    def test_validate_all_exist(self, tmp_path):
        """validate() returns is_valid=True when all files exist."""
        img_dir = tmp_path / "train"
        msk_dir = tmp_path / "train_labels"
        img_dir.mkdir()
        msk_dir.mkdir()

        img_path = img_dir / "test.png"
        msk_path = msk_dir / "test.png"
        img_path.write_bytes(b"fake")
        msk_path.write_bytes(b"fake")

        ds = MassachusettsDataset(
            image_paths=[img_path],
            mask_paths=[msk_path],
        )

        result = ds.validate()
        assert result["is_valid"] is True
        assert result["total_pairs"] == 1

    def test_validate_missing_files(self):
        """validate() detects missing files."""
        ds = MassachusettsDataset(
            image_paths=["nonexistent.png"],
            mask_paths=["also_missing.png"],
        )

        result = ds.validate()
        assert result["is_valid"] is False
        assert len(result["missing_images"]) == 1
        assert len(result["missing_masks"]) == 1
