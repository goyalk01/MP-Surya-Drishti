"""
Image and file I/O utilities.

Provides consistent image loading/saving and JSON I/O functions
used across all modules in the segmentation pipeline.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def load_image(
    path: str | Path,
    color_mode: str = "rgb",
    target_size: Optional[tuple[int, int]] = None,
) -> np.ndarray:
    """
    Load an image from disk.

    Args:
        path: Path to the image file.
        color_mode: "rgb" (default), "bgr", or "gray".
        target_size: Optional (width, height) to resize to.

    Returns:
        Image as numpy array.

    Raises:
        FileNotFoundError: If the file does not exist.
        RuntimeError: If the image cannot be read.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    if color_mode == "gray":
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    else:
        image = cv2.imread(str(path))

    if image is None:
        raise RuntimeError(f"Could not read image: {path}")

    if color_mode == "rgb" and len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if target_size is not None:
        interpolation = (
            cv2.INTER_NEAREST if color_mode == "gray" else cv2.INTER_LINEAR
        )
        image = cv2.resize(image, target_size, interpolation=interpolation)

    return image


def save_image(
    image: np.ndarray,
    path: str | Path,
    create_dirs: bool = True,
) -> Path:
    """
    Save an image to disk.

    Args:
        image: Image as numpy array (RGB or grayscale).
        path: Output file path.
        create_dirs: Create parent directories if needed.

    Returns:
        Path to the saved file.
    """
    path = Path(path)

    if create_dirs:
        path.parent.mkdir(parents=True, exist_ok=True)

    # Convert RGB to BGR for OpenCV
    if len(image.shape) == 3 and image.shape[2] == 3:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        image_bgr = image

    cv2.imwrite(str(path), image_bgr)
    logger.debug("Saved image: %s", path)

    return path


def save_json(
    data: Any,
    path: str | Path,
    indent: int = 2,
) -> Path:
    """
    Save data to a JSON file.

    Args:
        data: JSON-serializable data.
        path: Output file path.
        indent: JSON indentation level.

    Returns:
        Path to the saved file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, default=str)

    logger.debug("Saved JSON: %s", path)
    return path


def load_json(path: str | Path) -> Any:
    """
    Load data from a JSON file.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed JSON data.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data
