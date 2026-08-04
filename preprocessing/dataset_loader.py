"""
Dataset loader for the Massachusetts Buildings Dataset.

Provides a PyTorch Dataset class that loads aerial image and mask pairs
from filesystem split directories, validates mask values, and applies
preprocessing transforms for segmentation training.

Matches the official Massachusetts Buildings Dataset layout:
    datasets/massachusetts/
        ├── train/ & train_labels/
        ├── val/ & val_labels/
        └── test/ & test_labels/
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class MassachusettsDataset(Dataset):
    """
    PyTorch Dataset for the Massachusetts Buildings Dataset.

    Loads aerial images and their corresponding binary rooftop masks.
    Supports configurable transforms for data augmentation and normalization.

    Args:
        image_paths: List of absolute or relative paths to image files.
        mask_paths: List of absolute or relative paths to corresponding mask files.
        image_size: Target size to resize images and masks to (height, width).
        transform: Optional Albumentations transform pipeline.
        normalizer: Optional image normalizer.
        mask_building_value: Pixel value representing buildings in raw masks.
            Defaults to 255 (white).

    Raises:
        ValueError: If image_paths and mask_paths have different lengths.
    """

    def __init__(
        self,
        image_paths: list[str | Path],
        mask_paths: list[str | Path],
        image_size: int = 512,
        transform: Optional[Callable] = None,
        normalizer: Optional[Any] = None,
        mask_building_value: int = 255,
    ) -> None:
        if len(image_paths) != len(mask_paths):
            raise ValueError(
                f"Number of images ({len(image_paths)}) does not match "
                f"number of masks ({len(mask_paths)})"
            )

        self.image_paths = [Path(p) for p in image_paths]
        self.mask_paths = [Path(p) for p in mask_paths]
        self.image_size = image_size
        self.transform = transform
        self.normalizer = normalizer
        self.mask_building_value = mask_building_value

        logger.info(
            "Initialized MassachusettsDataset with %d image-mask pairs "
            "(image_size=%d)",
            len(self.image_paths),
            self.image_size,
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """
        Load and preprocess a single image-mask pair.

        Returns:
            Dictionary containing:
                - pixel_values: Normalized image tensor (C, H, W).
                - labels: Binary mask tensor (H, W) with values {0, 1}.
                - image_path: Original image file path (str).
        """
        image = cv2.imread(str(self.image_paths[idx]))
        if image is None:
            raise FileNotFoundError(
                f"Could not load image: {self.image_paths[idx]}"
            )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(self.mask_paths[idx]), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(
                f"Could not load mask: {self.mask_paths[idx]}"
            )

        image = cv2.resize(
            image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR
        )
        mask = cv2.resize(
            mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST
        )

        mask = (mask >= self.mask_building_value // 2).astype(np.uint8)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        if self.normalizer is not None:
            image = self.normalizer.normalize(image)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        mask = torch.from_numpy(mask).long()

        return {
            "pixel_values": image,
            "labels": mask,
            "image_path": str(self.image_paths[idx]),
        }

    def validate(self) -> dict[str, Any]:
        """
        Run validation checks on the dataset instance.

        Returns:
            Dictionary with validation results.
        """
        missing_images = [str(p) for p in self.image_paths if not p.exists()]
        missing_masks = [str(p) for p in self.mask_paths if not p.exists()]

        result = {
            "total_pairs": len(self.image_paths),
            "missing_images": missing_images,
            "missing_masks": missing_masks,
            "is_valid": len(missing_images) == 0 and len(missing_masks) == 0,
        }

        if result["is_valid"]:
            logger.info("Dataset validation passed: %d pairs", result["total_pairs"])
        else:
            logger.warning(
                "Dataset validation failed: %d missing images, %d missing masks",
                len(missing_images),
                len(missing_masks),
            )

        return result

    @staticmethod
    def discover_pairs(
        root_dir: str | Path,
        images_dir: str = "train",
        masks_dir: str = "train_labels",
        extensions: Optional[list[str]] = None,
    ) -> tuple[list[Path], list[Path]]:
        """
        Discover image-mask pairs from specified image and mask directories.

        Matches images to masks by filename (stem). Only includes pairs
        where both the image and mask file exist.

        Args:
            root_dir: Root dataset directory.
            images_dir: Subdirectory or path for images (e.g. "train", "val", "test").
            masks_dir: Subdirectory or path for masks (e.g. "train_labels", "val_labels").
            extensions: Allowed file extensions.

        Returns:
            Tuple of (image_paths, mask_paths), sorted by filename stem.
        """
        if extensions is None:
            extensions = [".tiff", ".tif", ".png", ".jpg", ".jpeg"]

        root = Path(root_dir)
        img_dir = root / images_dir if not Path(images_dir).is_absolute() else Path(images_dir)
        msk_dir = root / masks_dir if not Path(masks_dir).is_absolute() else Path(masks_dir)

        if not img_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {img_dir}")
        if not msk_dir.exists():
            raise FileNotFoundError(f"Masks directory not found: {msk_dir}")

        mask_lookup: dict[str, Path] = {}
        for msk_file in msk_dir.iterdir():
            if msk_file.suffix.lower() in extensions:
                mask_lookup[msk_file.stem] = msk_file

        image_paths: list[Path] = []
        mask_paths: list[Path] = []

        for img_file in sorted(img_dir.iterdir()):
            if img_file.suffix.lower() not in extensions:
                continue
            if img_file.stem in mask_lookup:
                image_paths.append(img_file)
                mask_paths.append(mask_lookup[img_file.stem])
            else:
                logger.warning(
                    "No matching mask found for image in %s: %s", img_dir.name, img_file.name
                )

        logger.info(
            "Discovered %d image-mask pairs in '%s' ↔ '%s'",
            len(image_paths),
            img_dir.name,
            msk_dir.name,
        )

        return image_paths, mask_paths

    @classmethod
    def discover_all_splits(
        cls,
        root_dir: str | Path,
        train_images_dir: str = "train",
        train_masks_dir: str = "train_labels",
        val_images_dir: str = "val",
        val_masks_dir: str = "val_labels",
        test_images_dir: str = "test",
        test_masks_dir: str = "test_labels",
        extensions: Optional[list[str]] = None,
    ) -> dict[str, tuple[list[Path], list[Path]]]:
        """
        Discover pairs for all official partitions: train, val, and test.

        Returns:
            Dictionary mapping split names ("train", "val", "test") to
            (image_paths, mask_paths) tuples.
        """
        splits = {}
        split_configs = [
            ("train", train_images_dir, train_masks_dir),
            ("val", val_images_dir, val_masks_dir),
            ("test", test_images_dir, test_masks_dir),
        ]

        for split_name, img_dir, msk_dir in split_configs:
            images, masks = cls.discover_pairs(
                root_dir=root_dir,
                images_dir=img_dir,
                masks_dir=msk_dir,
                extensions=extensions,
            )
            splits[split_name] = (images, masks)

        return splits
