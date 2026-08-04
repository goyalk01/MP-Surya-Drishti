"""
Image normalizer for the segmentation framework.

Provides a model-agnostic normalization interface. Attempts to load
a model-specific HuggingFace image processor, falling back to standard
ImageNet normalization if unavailable.

This normalizer works with any model in the framework — it does not
import or depend on any specific model class.
"""

from __future__ import annotations

import logging
from typing import Optional, Union

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

# Known HuggingFace processor mappings for auto-detection
# Add new model processors here when integrating new architectures
_PROCESSOR_MAP = {
    "segformer": "transformers.SegformerImageProcessor",
    "deeplabv3plus": None,  # Uses standard ImageNet normalization
    "mask2former": "transformers.Mask2FormerImageProcessor",
    "unet": None,  # Uses standard ImageNet normalization
}


class ImageNormalizer:
    """
    Model-agnostic image normalizer for the segmentation framework.

    Automatically detects and loads the correct HuggingFace image
    processor based on the backbone identifier. Falls back to standard
    ImageNet normalization for models without a dedicated processor.

    Args:
        backbone: HuggingFace model identifier (e.g., ``nvidia/mit-b2``)
            or a descriptive string. Used to auto-detect the processor.
        image_size: Target image size for the processor.
        model_type: Optional model type identifier (e.g., "segformer").
            Helps select the correct processor when the backbone string
            is ambiguous.
        do_reduce_labels: Whether to reduce label indices by 1. Set False
            for binary segmentation (0=background, 1=foreground).
        mean: Custom normalization mean. Defaults to ImageNet values.
        std: Custom normalization std. Defaults to ImageNet values.
    """

    # ImageNet defaults — used by most pretrained backbones
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        backbone: str = "nvidia/mit-b2",
        image_size: int = 512,
        model_type: Optional[str] = None,
        do_reduce_labels: bool = False,
        mean: Optional[tuple[float, ...]] = None,
        std: Optional[tuple[float, ...]] = None,
    ) -> None:
        self.backbone = backbone
        self.image_size = image_size
        self.model_type = model_type
        self.mean = mean or self.IMAGENET_MEAN
        self.std = std or self.IMAGENET_STD
        self.processor = None

        # Try to load a HuggingFace processor
        self.processor = self._auto_detect_processor(
            backbone, model_type, image_size, do_reduce_labels
        )

        if self.processor is not None:
            logger.info(
                "Loaded image processor for '%s' (size=%d)",
                backbone,
                image_size,
            )
        else:
            logger.info(
                "Using standard ImageNet normalization for '%s' (size=%d)",
                backbone,
                image_size,
            )

    def _auto_detect_processor(
        self,
        backbone: str,
        model_type: Optional[str],
        image_size: int,
        do_reduce_labels: bool,
    ):
        """
        Attempt to auto-detect and load the correct HuggingFace processor.

        Tries model-specific processors first, then falls back to the
        generic AutoImageProcessor.
        """
        # Strategy 1: Try SegformerImageProcessor if model_type is segformer
        # or backbone looks like a SegFormer identifier
        if model_type == "segformer" or "mit-b" in backbone.lower() or "segformer" in backbone.lower():
            try:
                from transformers import SegformerImageProcessor
                return SegformerImageProcessor.from_pretrained(
                    backbone,
                    do_reduce_labels=do_reduce_labels,
                    size={"height": image_size, "width": image_size},
                    do_resize=True,
                )
            except Exception:
                pass  # Fall through to next strategy

        # Strategy 2: Try AutoImageProcessor (works for many HF models)
        try:
            from transformers import AutoImageProcessor
            return AutoImageProcessor.from_pretrained(
                backbone,
                size={"height": image_size, "width": image_size},
                do_resize=True,
            )
        except Exception:
            pass

        # Strategy 3: Return None — will use manual normalization
        return None

    def normalize(
        self,
        image: Union[np.ndarray, "Image.Image"],
        return_tensors: str = "pt",
    ) -> torch.Tensor:
        """
        Normalize an image for model input.

        Args:
            image: Input image as numpy array (H, W, C) in RGB uint8
                or a PIL Image.
            return_tensors: Tensor format. Defaults to ``"pt"`` (PyTorch).

        Returns:
            Normalized image tensor of shape (C, H, W).
        """
        if self.processor is not None:
            if isinstance(image, np.ndarray):
                pil_image = Image.fromarray(image)
            else:
                pil_image = image

            encoding = self.processor(
                images=pil_image,
                return_tensors=return_tensors,
            )
            # Shape: (1, C, H, W) → (C, H, W)
            return encoding["pixel_values"].squeeze(0)

        return self._manual_normalize(image)

    def normalize_batch(
        self,
        images: list[Union[np.ndarray, "Image.Image"]],
        return_tensors: str = "pt",
    ) -> torch.Tensor:
        """
        Normalize a batch of images.

        Args:
            images: List of input images.
            return_tensors: Tensor format.

        Returns:
            Batch tensor of shape (B, C, H, W).
        """
        if self.processor is not None:
            pil_images = []
            for img in images:
                if isinstance(img, np.ndarray):
                    pil_images.append(Image.fromarray(img))
                else:
                    pil_images.append(img)

            encoding = self.processor(
                images=pil_images,
                return_tensors=return_tensors,
            )
            return encoding["pixel_values"]

        return torch.stack([self._manual_normalize(img) for img in images])

    def _manual_normalize(
        self, image: Union[np.ndarray, "Image.Image"]
    ) -> torch.Tensor:
        """
        Manual normalization using configurable mean/std.

        Args:
            image: Input image as numpy array (H, W, C) or PIL Image.

        Returns:
            Normalized tensor of shape (C, H, W).
        """
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Ensure float32 and scale to [0, 1]
        image = image.astype(np.float32) / 255.0

        # Apply normalization
        mean = np.array(self.mean, dtype=np.float32)
        std = np.array(self.std, dtype=np.float32)
        image = (image - mean) / std

        # (H, W, C) → (C, H, W)
        tensor = torch.from_numpy(image).permute(2, 0, 1).float()

        return tensor
