"""
Dataset splitter and DataLoader factory for the framework.

Provides utilities for split manifests and high-performance PyTorch DataLoaders.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class DatasetSplitter:
    """
    Split image-mask file lists into train/val/test sets and construct DataLoaders.
    """

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
        Create high-performance PyTorch DataLoader objects for each split.

        Configures pin_memory, persistent_workers, and prefetch_factor for maximum
        GPU utilization and data throughput.
        """
        is_cuda = torch.cuda.is_available()
        use_pin = pin_memory and is_cuda
        use_persistent = num_workers > 0

        loader_kwargs: dict[str, Any] = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": use_pin,
        }

        if use_persistent:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = 2

        loaders = {
            "train": DataLoader(
                train_dataset,
                shuffle=True,
                drop_last=True,
                **loader_kwargs,
            ),
            "val": DataLoader(
                val_dataset,
                shuffle=False,
                drop_last=False,
                **loader_kwargs,
            ),
        }

        if test_dataset is not None:
            loaders["test"] = DataLoader(
                test_dataset,
                shuffle=False,
                drop_last=False,
                **loader_kwargs,
            )

        logger.info(
            "Created DataLoaders (batch_size=%d, num_workers=%d, pin_memory=%s, persistent_workers=%s)",
            batch_size,
            num_workers,
            use_pin,
            use_persistent,
        )

        return loaders
