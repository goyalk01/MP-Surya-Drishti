"""
Loss functions for semantic segmentation.

Provides a combined BCE + Dice loss that handles both per-pixel
classification accuracy (BCE/CE) and region-level overlap quality (Dice).
This combination is the standard approach for binary and multi-class
segmentation with class imbalance.

These loss functions are model-agnostic — they operate on logits
and target masks from any segmentation model in the framework.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class DiceLoss(nn.Module):
    """
    Soft Dice loss for semantic segmentation.

    Computes the Dice coefficient between predicted probabilities and
    ground truth masks, returning 1 - Dice as the loss.

    Works with any number of classes.

    Args:
        smooth: Smoothing factor to prevent division by zero and
            stabilize gradients. Default 1.0.
    """

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Dice loss.

        Args:
            logits: Raw model output (B, num_classes, H, W).
            targets: Ground truth masks (B, H, W) with class indices.

        Returns:
            Scalar Dice loss value.
        """
        num_classes = logits.shape[1]

        # Convert logits to probabilities
        probs = F.softmax(logits, dim=1)

        # One-hot encode targets: (B, H, W) → (B, num_classes, H, W)
        targets_one_hot = F.one_hot(targets.long(), num_classes=num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()

        # Flatten spatial dimensions for dot product
        probs_flat = probs.contiguous().view(probs.shape[0], num_classes, -1)
        targets_flat = targets_one_hot.contiguous().view(
            targets_one_hot.shape[0], num_classes, -1
        )

        # Compute Dice per class
        intersection = (probs_flat * targets_flat).sum(dim=2)
        cardinality = probs_flat.sum(dim=2) + targets_flat.sum(dim=2)

        dice_per_class = (2.0 * intersection + self.smooth) / (
            cardinality + self.smooth
        )

        # Average across classes and batch
        dice_loss = 1.0 - dice_per_class.mean()

        return dice_loss


class CombinedSegmentationLoss(nn.Module):
    """
    Combined Cross-Entropy + Dice loss for semantic segmentation.

    CE handles per-pixel classification, while Dice handles region-level
    overlap. The combination addresses class imbalance common in aerial
    imagery where target regions may be a small fraction of the image.

    This loss is model-agnostic — it operates on upsampled logits
    from any segmentation model that follows the framework interface.

    Args:
        bce_weight: Weight for the cross-entropy component.
        dice_weight: Weight for the Dice component.
        pos_weight: Positive class weight for CE to handle imbalance.
            Values > 1 increase recall for the foreground class.
        dice_smooth: Smoothing factor for Dice loss.
    """

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        pos_weight: Optional[float] = None,
        dice_smooth: float = 1.0,
    ) -> None:
        super().__init__()

        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

        # Standard cross-entropy loss for multi-class (works for 2-class)
        if pos_weight is not None:
            weight = torch.tensor([1.0, pos_weight])
            self.ce_loss = nn.CrossEntropyLoss(weight=weight)
        else:
            self.ce_loss = nn.CrossEntropyLoss()

        self.dice_loss = DiceLoss(smooth=dice_smooth)

        logger.info(
            "CombinedSegmentationLoss initialized (ce_weight=%.2f, "
            "dice_weight=%.2f, pos_weight=%s)",
            bce_weight,
            dice_weight,
            pos_weight,
        )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Compute combined loss.

        Args:
            logits: Raw model output (B, num_classes, H, W).
            targets: Ground truth masks (B, H, W) with class indices.

        Returns:
            Dictionary containing:
                - loss: Combined weighted loss (scalar).
                - ce_loss: Cross-entropy component (scalar).
                - dice_loss: Dice component (scalar).
        """
        # Move CE weight to the same device as logits
        if hasattr(self.ce_loss, "weight") and self.ce_loss.weight is not None:
            self.ce_loss.weight = self.ce_loss.weight.to(logits.device)

        ce = self.ce_loss(logits, targets.long())
        dice = self.dice_loss(logits, targets)

        combined = self.bce_weight * ce + self.dice_weight * dice

        return {
            "loss": combined,
            "ce_loss": ce,
            "dice_loss": dice,
        }
