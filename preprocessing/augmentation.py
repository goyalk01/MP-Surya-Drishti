"""
Data augmentation pipeline for aerial image segmentation.

Uses Albumentations for joint image-mask transforms to ensure
spatial augmentations are applied consistently to both.

The pipeline is configurable for train (heavy augmentation) and
val/test (normalization only) modes.
"""

from __future__ import annotations

import logging
from typing import Optional

import albumentations as A
from albumentations.pytorch import ToTensorV2

logger = logging.getLogger(__name__)


class AugmentationPipeline:
    """
    Configurable augmentation pipeline for segmentation tasks.

    Provides separate transform chains for training and evaluation.
    Training augmentations increase data diversity while evaluation
    transforms only resize and normalize.

    Args:
        image_size: Target image size (applies to both height and width).
        mean: Normalization mean per channel. Defaults to ImageNet values.
        std: Normalization std per channel. Defaults to ImageNet values.
    """

    # ImageNet normalization — consistent with SegFormer pretrained weights
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        image_size: int = 512,
        mean: Optional[tuple[float, ...]] = None,
        std: Optional[tuple[float, ...]] = None,
    ) -> None:
        self.image_size = image_size
        self.mean = mean or self.IMAGENET_MEAN
        self.std = std or self.IMAGENET_STD

        logger.info(
            "Initialized AugmentationPipeline (image_size=%d)", self.image_size
        )

    def get_train_transform(self) -> A.Compose:
        """
        Build training augmentation pipeline.

        Includes spatial transforms, color jitter, and noise to improve
        model generalization on diverse aerial imagery.

        Returns:
            Albumentations Compose pipeline for training.
        """
        return A.Compose(
            [
                # Spatial transforms
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.15,
                    rotate_limit=30,
                    border_mode=0,  # cv2.BORDER_CONSTANT
                    p=0.5,
                ),

                # Color and brightness augmentations
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.4,
                ),
                A.HueSaturationValue(
                    hue_shift_limit=15,
                    sat_shift_limit=25,
                    val_shift_limit=15,
                    p=0.3,
                ),

                # Noise and blur (simulates image quality variation)
                A.GaussianBlur(blur_limit=(3, 5), p=0.2),
                A.GaussNoise(p=0.2),

                # Geometric distortion (handles lens effects)
                A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.2),

                # Normalize using ImageNet stats (matches SegFormer pretrained)
                A.Normalize(mean=self.mean, std=self.std),
            ]
        )

    def get_val_transform(self) -> A.Compose:
        """
        Build validation/test transform pipeline.

        Only applies normalization — no augmentation, to ensure
        consistent evaluation metrics.

        Returns:
            Albumentations Compose pipeline for validation/testing.
        """
        return A.Compose(
            [
                A.Normalize(mean=self.mean, std=self.std),
            ]
        )

    def get_inference_transform(self) -> A.Compose:
        """
        Build inference transform pipeline.

        Same as validation — normalize only. Kept as a separate method
        for semantic clarity and future customization.

        Returns:
            Albumentations Compose pipeline for inference.
        """
        return A.Compose(
            [
                A.Normalize(mean=self.mean, std=self.std),
            ]
        )

    def denormalize(self, image_tensor) -> "np.ndarray":
        """
        Reverse normalization to recover a displayable image.

        Useful for visualization after augmentation.

        Args:
            image_tensor: Normalized image as numpy array (H, W, C) or
                tensor (C, H, W).

        Returns:
            Denormalized image as uint8 numpy array (H, W, C).
        """
        import numpy as np

        if hasattr(image_tensor, "numpy"):
            image = image_tensor.numpy()
        else:
            image = image_tensor

        # Handle (C, H, W) → (H, W, C)
        if image.ndim == 3 and image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))

        mean = np.array(self.mean)
        std = np.array(self.std)

        image = (image * std + mean) * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)

        return image
