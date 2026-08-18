"""
Native-resolution tiled dataset loader for the Massachusetts Buildings Dataset.

Crops native-resolution patches (default 512x512 with stride 256) directly from
full-resolution aerial images and ground-truth masks. Guarantees 100% spatial
coverage by aligning boundary tiles to image edges without artificial padding.

Prevents data leakage by extracting patches strictly within pre-defined dataset splits.
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


def generate_tile_coordinates(
    image_width: int,
    image_height: int,
    tile_size: int = 512,
    stride: int = 256,
) -> list[tuple[int, int, int, int]]:
    """
    Generate bounding box coordinates for sliding-window tile extraction.

    Guarantees 100% pixel coverage across the entire image by anchoring
    the final step in each dimension to the image boundary.

    Args:
        image_width: Width of the source image in pixels.
        image_height: Height of the source image in pixels.
        tile_size: Height and width of each tile.
        stride: Step size between adjacent tiles.

    Returns:
        List of (x1, y1, x2, y2) tuples defining tile bounding boxes.
    """
    if image_width <= tile_size:
        x_coords = [0]
    else:
        x_coords = list(range(0, image_width - tile_size + 1, stride))
        if x_coords[-1] + tile_size < image_width:
            x_coords.append(image_width - tile_size)

    if image_height <= tile_size:
        y_coords = [0]
    else:
        y_coords = list(range(0, image_height - tile_size + 1, stride))
        if y_coords[-1] + tile_size < image_height:
            y_coords.append(image_height - tile_size)

    # Remove duplicates while preserving order
    x_coords = sorted(list(set(x_coords)))
    y_coords = sorted(list(set(y_coords)))

    coordinates = []
    for y in y_coords:
        for x in x_coords:
            coordinates.append((x, y, x + tile_size, y + tile_size))

    return coordinates


class TiledMassachusettsDataset(Dataset):
    """
    Native-resolution patch dataset for aerial rooftop segmentation.

    Extracts fixed-size crops directly from full-resolution source images
    and masks without pre-resizing, preserving sharp edges and fine structures.

    Args:
        image_paths: List of file paths to full-resolution source images.
        mask_paths: List of file paths to corresponding full-resolution masks.
        tile_size: Dimension of square patches (default 512).
        stride: Sliding-window step size (default 256).
        transform: Optional Albumentations augmentation pipeline.
        normalizer: Optional image normalizer.
        mask_building_value: Pixel value for buildings in raw masks (default 255).
    """

    def __init__(
        self,
        image_paths: list[str | Path],
        mask_paths: list[str | Path],
        tile_size: int = 512,
        stride: int = 256,
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
        self.tile_size = tile_size
        self.stride = stride
        self.transform = transform
        self.normalizer = normalizer
        self.mask_building_value = mask_building_value

        # Build tile index mapping: tile_index -> (image_idx, (x1, y1, x2, y2))
        self.tile_index: list[tuple[int, tuple[int, int, int, int]]] = []
        self._build_tile_index()

        logger.info(
            "Initialized TiledMassachusettsDataset with %d source images → %d total patches "
            "(tile_size=%d, stride=%d)",
            len(self.image_paths),
            len(self.tile_index),
            self.tile_size,
            self.stride,
        )

    def _build_tile_index(self) -> None:
        """Inspect source image dimensions and construct the tile index table."""
        for img_idx, img_path in enumerate(self.image_paths):
            # Fast header inspection using cv2
            img = cv2.imread(str(img_path))
            if img is None:
                raise FileNotFoundError(f"Could not load image: {img_path}")
            h, w = img.shape[:2]

            coords = generate_tile_coordinates(
                image_width=w,
                image_height=h,
                tile_size=self.tile_size,
                stride=self.stride,
            )

            for bbox in coords:
                self.tile_index.append((img_idx, bbox))

    def __len__(self) -> int:
        return len(self.tile_index)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """
        Extract and preprocess a single native-resolution tile.

        Returns:
            Dictionary containing:
                - pixel_values: Normalized image tensor (C, H, W).
                - labels: Binary mask tensor (H, W) with values {0, 1}.
                - image_path: Original image file path (str).
                - tile_coords: (x1, y1, x2, y2) tuple.
                - image_idx: Index of source image.
        """
        img_idx, (x1, y1, x2, y2) = self.tile_index[idx]
        img_path = self.image_paths[img_idx]
        mask_path = self.mask_paths[img_idx]

        # Load full-resolution image and mask
        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"Could not load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Could not load mask: {mask_path}")

        # Crop patch directly at native resolution
        img_h, img_w = image.shape[:2]

        # Handle edge case where image is smaller than tile_size
        if img_w < self.tile_size or img_h < self.tile_size:
            pad_w = max(0, self.tile_size - img_w)
            pad_h = max(0, self.tile_size - img_h)
            image = cv2.copyMakeBorder(
                image, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT
            )
            mask = cv2.copyMakeBorder(
                mask, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0
            )

        tile_img = image[y1:y2, x1:x2]
        tile_mask = mask[y1:y2, x1:x2]

        # Reuse exact Massachusetts thresholding logic: 255 -> 1, 0 -> 0
        tile_mask = (tile_mask >= self.mask_building_value // 2).astype(np.uint8)

        # Apply augmentation transforms jointly
        if self.transform is not None:
            augmented = self.transform(image=tile_img, mask=tile_mask)
            tile_img = augmented["image"]
            tile_mask = augmented["mask"]

        # Normalize image
        if self.normalizer is not None:
            tile_img = self.normalizer.normalize(tile_img)
        else:
            tile_img = torch.from_numpy(tile_img).permute(2, 0, 1).float() / 255.0

        tile_mask = torch.from_numpy(tile_mask).long()

        return {
            "pixel_values": tile_img,
            "labels": tile_mask,
            "image_path": str(img_path),
            "tile_coords": (x1, y1, x2, y2),
            "image_idx": img_idx,
        }
