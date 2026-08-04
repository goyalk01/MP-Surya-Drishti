"""
Dataset splitter for creating reproducible train/val/test splits.

Splits a list of image-mask pairs into training, validation, and test
sets with configurable ratios and random seed. Saves a manifest file
for reproducibility.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class DatasetSplitter:
    """
    Split image-mask file lists into train/val/test sets.

    Args:
        train_ratio: Fraction of data for training (default 0.8).
        val_ratio: Fraction of data for validation (default 0.1).
        test_ratio: Fraction of data for testing (default 0.1).
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
    ) -> None:
        total = train_ratio + val_ratio + test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Split ratios must sum to 1.0, got {total:.4f} "
                f"(train={train_ratio}, val={val_ratio}, test={test_ratio})"
            )

        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed

        logger.info(
            "DatasetSplitter initialized (train=%.0f%%, val=%.0f%%, test=%.0f%%, seed=%d)",
            train_ratio * 100,
            val_ratio * 100,
            test_ratio * 100,
            seed,
        )

    def split(
        self,
        image_paths: list[Path],
        mask_paths: list[Path],
    ) -> dict[str, dict[str, list[Path]]]:
        """
        Split image-mask pairs into train/val/test sets.

        Args:
            image_paths: List of image file paths.
            mask_paths: List of corresponding mask file paths.

        Returns:
            Dictionary with 'train', 'val', 'test' keys, each containing
            'images' and 'masks' lists of Path objects.
        """
        n = len(image_paths)
        if n != len(mask_paths):
            raise ValueError(
                f"Image count ({n}) != mask count ({len(mask_paths)})"
            )

        # Shuffle indices with fixed seed
        rng = np.random.RandomState(self.seed)
        indices = rng.permutation(n)

        # Calculate split boundaries
        n_train = int(n * self.train_ratio)
        n_val = int(n * self.val_ratio)
        # Test gets the remainder to avoid off-by-one
        train_idx = indices[:n_train]
        val_idx = indices[n_train : n_train + n_val]
        test_idx = indices[n_train + n_val :]

        splits = {
            "train": {
                "images": [image_paths[i] for i in train_idx],
                "masks": [mask_paths[i] for i in train_idx],
            },
            "val": {
                "images": [image_paths[i] for i in val_idx],
                "masks": [mask_paths[i] for i in val_idx],
            },
            "test": {
                "images": [image_paths[i] for i in test_idx],
                "masks": [mask_paths[i] for i in test_idx],
            },
        }

        logger.info(
            "Split %d samples → train=%d, val=%d, test=%d",
            n,
            len(train_idx),
            len(val_idx),
            len(test_idx),
        )

        return splits

    def save_manifest(
        self,
        splits: dict[str, dict[str, list[Path]]],
        manifest_path: str | Path,
    ) -> None:
        """
        Save the split manifest to a JSON file for reproducibility.

        Args:
            splits: Output of ``split()`` method.
            manifest_path: Path to save the manifest JSON.
        """
        manifest_path = Path(manifest_path)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)

        manifest = {
            "seed": self.seed,
            "ratios": {
                "train": self.train_ratio,
                "val": self.val_ratio,
                "test": self.test_ratio,
            },
            "splits": {},
        }

        for split_name, data in splits.items():
            manifest["splits"][split_name] = {
                "count": len(data["images"]),
                "images": [str(p) for p in data["images"]],
                "masks": [str(p) for p in data["masks"]],
            }

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        logger.info("Saved split manifest to %s", manifest_path)

    def load_manifest(
        self,
        manifest_path: str | Path,
    ) -> dict[str, dict[str, list[Path]]]:
        """
        Load a previously saved split manifest.

        Args:
            manifest_path: Path to the manifest JSON file.

        Returns:
            Dictionary with the same structure as ``split()`` output.
        """
        manifest_path = Path(manifest_path)
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        splits = {}
        for split_name, data in manifest["splits"].items():
            splits[split_name] = {
                "images": [Path(p) for p in data["images"]],
                "masks": [Path(p) for p in data["masks"]],
            }

        logger.info(
            "Loaded split manifest from %s (train=%d, val=%d, test=%d)",
            manifest_path,
            len(splits["train"]["images"]),
            len(splits["val"]["images"]),
            len(splits["test"]["images"]),
        )

        return splits

    @staticmethod
    def create_dataloaders(
        train_dataset,
        val_dataset,
        test_dataset=None,
        batch_size: int = 8,
        num_workers: int = 2,
        pin_memory: bool = True,
    ) -> dict[str, DataLoader]:
        """
        Create PyTorch DataLoader objects for each split.

        Args:
            train_dataset: Training dataset instance.
            val_dataset: Validation dataset instance.
            test_dataset: Optional test dataset instance.
            batch_size: Batch size for all loaders.
            num_workers: Number of worker processes for data loading.
            pin_memory: Pin memory for faster GPU transfer.

        Returns:
            Dictionary with 'train', 'val', and optionally 'test' DataLoaders.
        """
        loaders = {
            "train": DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=True,
            ),
            "val": DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=False,
            ),
        }

        if test_dataset is not None:
            loaders["test"] = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=False,
            )

        logger.info(
            "Created DataLoaders (batch_size=%d, num_workers=%d)",
            batch_size,
            num_workers,
        )

        return loaders
