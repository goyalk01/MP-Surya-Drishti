"""
Loss functions for semantic segmentation.

Provides:
- DiceLoss: Soft Dice loss for region overlap.
- FocalLoss: Modulated focal cross-entropy for boundary and hard-example mining.
- CombinedSegmentationLoss: Standard Cross-Entropy + Dice loss.
- FocalDiceLoss: Selectable Focal + Soft Dice loss for aerial class imbalance.
- get_loss_function: Model-agnostic loss factory.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class DiceLoss(nn.Module):
    """
    Soft Dice loss for semantic segmentation.

    Computes the Dice coefficient between predicted probabilities and
    ground truth masks, returning 1 - Dice as the loss.

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

        if targets.dim() == 2:
            targets = targets.unsqueeze(1)  # (B, 1, W) or (B, H, 1)

        # One-hot encode targets: (B, H, W) → (B, num_classes, H, W)
        targets_clamped = torch.clamp(targets.long(), 0, num_classes - 1)
        targets_one_hot = F.one_hot(targets_clamped, num_classes=num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()

        # Convert logits to probabilities
        probs = F.softmax(logits, dim=1)

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


class FocalLoss(nn.Module):
    """
    Focal Loss for dense classification with severe class imbalance.

    FL(p_t) = - alpha_t * (1 - p_t)^gamma * log(p_t)

    Down-weights easy background examples and focuses training gradients
    on hard boundary pixels and minority foreground classes.

    Args:
        alpha: Weighting factor for classes. Can be scalar (for binary) or list/Tensor.
        gamma: Focusing parameter for modulating factor (default 2.0).
        ignore_index: Class index to ignore during loss computation.
        reduction: 'mean', 'sum', or 'none'.
    """

    def __init__(
        self,
        alpha: Optional[float | list[float] | torch.Tensor] = 0.25,
        gamma: float = 2.0,
        ignore_index: int = 255,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

        if isinstance(alpha, (int, float)):
            self.alpha = torch.tensor([1.0 - float(alpha), float(alpha)], dtype=torch.float32)
        elif isinstance(alpha, (list, tuple)):
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        elif isinstance(alpha, torch.Tensor):
            self.alpha = alpha
        else:
            self.alpha = None

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute multi-class Focal loss.

        Args:
            logits: Raw model logits (B, num_classes, H, W).
            targets: Ground truth class indices (B, H, W).

        Returns:
            Scalar loss tensor.
        """
        num_classes = logits.shape[1]
        valid_mask = targets != self.ignore_index
        if not valid_mask.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        # Compute unreduced cross entropy
        ce_loss = F.cross_entropy(
            logits, targets.long(), ignore_index=self.ignore_index, reduction="none"
        )

        # Compute p_t = exp(-CE) with numerical clamping
        p_t = torch.exp(-ce_loss)
        p_t = torch.clamp(p_t, min=1e-7, max=1.0 - 1e-7)

        # Modulating factor (1 - p_t)^gamma
        modulating_factor = torch.pow(1.0 - p_t, self.gamma)

        focal_loss = modulating_factor * ce_loss

        if self.alpha is not None:
            if self.alpha.device != logits.device:
                self.alpha = self.alpha.to(logits.device)
            targets_clamped = torch.clamp(targets.long(), 0, num_classes - 1)
            alpha_t = self.alpha[targets_clamped]
            focal_loss = alpha_t * focal_loss

        focal_loss = focal_loss[valid_mask]

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class CombinedSegmentationLoss(nn.Module):
    """
    Combined Cross-Entropy + Dice loss for semantic segmentation.

    Args:
        bce_weight: Weight for cross-entropy component.
        dice_weight: Weight for Dice component.
        pos_weight: Positive class weight for CE.
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

        if pos_weight is not None:
            weight = torch.tensor([1.0, float(pos_weight)])
            self.ce_loss = nn.CrossEntropyLoss(weight=weight)
        else:
            self.ce_loss = nn.CrossEntropyLoss()

        self.dice_loss = DiceLoss(smooth=dice_smooth)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute combined CE + Dice loss."""
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


class FocalDiceLoss(nn.Module):
    """
    Combined Focal + Soft Dice loss for hard example and boundary optimization.

    Args:
        focal_weight: Weight for Focal loss component (default 0.5).
        dice_weight: Weight for Dice loss component (default 0.5).
        focal_gamma: Focusing exponent (default 2.0).
        focal_alpha: Balancing parameter for foreground/background (default 0.25).
        dice_smooth: Smoothing factor for Dice loss (default 1.0).
        ignore_index: Ignored class index (default 255).
    """

    def __init__(
        self,
        focal_weight: float = 0.5,
        dice_weight: float = 0.5,
        focal_gamma: float = 2.0,
        focal_alpha: Optional[float | list[float]] = 0.25,
        dice_smooth: float = 1.0,
        ignore_index: int = 255,
    ) -> None:
        super().__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight

        self.focal_loss = FocalLoss(
            alpha=focal_alpha,
            gamma=focal_gamma,
            ignore_index=ignore_index,
        )
        self.dice_loss = DiceLoss(smooth=dice_smooth)

        logger.info(
            "FocalDiceLoss initialized (focal_w=%.2f, dice_w=%.2f, gamma=%.2f, alpha=%s)",
            focal_weight,
            dice_weight,
            focal_gamma,
            focal_alpha,
        )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute combined Focal + Dice loss."""
        focal = self.focal_loss(logits, targets)
        dice = self.dice_loss(logits, targets)
        combined = self.focal_weight * focal + self.dice_weight * dice

        return {
            "loss": combined,
            "focal_loss": focal,
            "dice_loss": dice,
            "ce_loss": focal,  # alias for logger compatibility
        }


def get_loss_function(config: dict[str, Any]) -> nn.Module:
    """
    Factory function to create the configured segmentation loss module.

    Args:
        config: Training config or loss sub-dictionary.

    Returns:
        Configured nn.Module loss instance.
    """
    loss_cfg = config.get("loss", config) if isinstance(config, dict) else {}
    loss_type = loss_cfg.get("name", loss_cfg.get("type", "ce_dice")).lower()

    if loss_type in ["focal_dice", "focal"]:
        return FocalDiceLoss(
            focal_weight=loss_cfg.get("focal_weight", 0.5),
            dice_weight=loss_cfg.get("dice_weight", 0.5),
            focal_gamma=loss_cfg.get("focal_gamma", 2.0),
            focal_alpha=loss_cfg.get("focal_alpha", 0.25),
            dice_smooth=loss_cfg.get("dice_smooth", 1.0),
        )
    elif loss_type in ["ce_dice", "bce_dice"]:
        return CombinedSegmentationLoss(
            bce_weight=loss_cfg.get("bce_weight", loss_cfg.get("ce_weight", 0.5)),
            dice_weight=loss_cfg.get("dice_weight", 0.5),
            pos_weight=loss_cfg.get("pos_weight", 2.0),
            dice_smooth=loss_cfg.get("dice_smooth", 1.0),
        )
    else:
        logger.warning(
            "Unknown loss type '%s', falling back to CombinedSegmentationLoss (ce_dice)",
            loss_type,
        )
        return CombinedSegmentationLoss()
