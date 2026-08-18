"""
Segmentation metrics for evaluating rooftop detection quality.

Provides IoU (Intersection over Union), Dice coefficient, and
pixel accuracy — the standard metrics for semantic segmentation.

Supports both batch-level accumulation and per-image computation.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


class SegmentationMetrics:
    """
    Accumulate and compute segmentation metrics over batches.

    Maintains a running confusion matrix that can be updated batch-by-batch
    and computes final metrics with ``compute()``.

    Args:
        num_classes: Number of segmentation classes (default 2).
        ignore_index: Label value to exclude from metrics (default 255).
    """

    def __init__(
        self,
        num_classes: int = 2,
        ignore_index: int = 255,
    ) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index

        # Confusion matrix: shape (num_classes, num_classes)
        # Rows = ground truth, Columns = predictions
        self.confusion_matrix = np.zeros(
            (num_classes, num_classes), dtype=np.int64
        )

    def reset(self) -> None:
        """Reset the confusion matrix for a new evaluation run."""
        self.confusion_matrix = np.zeros(
            (self.num_classes, self.num_classes), dtype=np.int64
        )

    def update(
        self,
        predictions: torch.Tensor | np.ndarray,
        targets: torch.Tensor | np.ndarray,
    ) -> None:
        """
        Update the confusion matrix with a batch or single pair of predictions.

        Args:
            predictions: Predicted labels (B, H, W) or (H, W) as Tensor or ndarray.
            targets: Ground truth labels (B, H, W) or (H, W) as Tensor or ndarray.
        """
        if isinstance(predictions, torch.Tensor):
            preds = predictions.detach().cpu().numpy().flatten()
        else:
            preds = np.asarray(predictions).flatten()

        if isinstance(targets, torch.Tensor):
            tgts = targets.detach().cpu().numpy().flatten()
        else:
            tgts = np.asarray(targets).flatten()

        # Filter out ignored pixels and out-of-bound indices
        valid = (
            (tgts != self.ignore_index)
            & (tgts >= 0)
            & (tgts < self.num_classes)
            & (preds >= 0)
            & (preds < self.num_classes)
        )
        preds = preds[valid].astype(np.int64)
        tgts = tgts[valid].astype(np.int64)

        if len(tgts) > 0:
            cm_update = np.bincount(
                self.num_classes * tgts + preds,
                minlength=self.num_classes**2,
            ).reshape(self.num_classes, self.num_classes)
            self.confusion_matrix += cm_update

    def compute(self) -> dict[str, float]:
        """
        Compute all metrics from the accumulated confusion matrix.

        Returns:
            Dictionary with:
                - iou: Mean IoU across classes.
                - iou_per_class: IoU for each class.
                - dice: Mean Dice coefficient across classes.
                - dice_per_class: Dice for each class.
                - pixel_accuracy: Overall pixel accuracy.
                - rooftop_iou: IoU specifically for the rooftop class (class 1).
                - rooftop_dice: Dice specifically for the rooftop class.
        """
        cm = self.confusion_matrix

        # Per-class IoU
        intersection = np.diag(cm)
        union = cm.sum(axis=1) + cm.sum(axis=0) - intersection

        # Avoid division by zero
        valid = union > 0
        iou_per_class = np.zeros(self.num_classes, dtype=np.float64)
        iou_per_class[valid] = intersection[valid] / union[valid]

        # Per-class Dice
        dice_per_class = np.zeros(self.num_classes, dtype=np.float64)
        sum_preds_and_tgts = cm.sum(axis=1) + cm.sum(axis=0)
        valid_dice = sum_preds_and_tgts > 0
        dice_per_class[valid_dice] = (
            2.0 * intersection[valid_dice] / sum_preds_and_tgts[valid_dice]
        )

        # Mean metrics (across valid classes only)
        mean_iou = float(iou_per_class[valid].mean()) if valid.any() else 0.0
        mean_dice = float(dice_per_class[valid_dice].mean()) if valid_dice.any() else 0.0

        # Pixel accuracy
        total_correct = intersection.sum()
        total_pixels = cm.sum()
        pixel_accuracy = (
            float(total_correct / total_pixels) if total_pixels > 0 else 0.0
        )

        # Rooftop-specific metrics (class index 1)
        rooftop_iou = float(iou_per_class[1]) if self.num_classes > 1 else 0.0
        rooftop_dice = float(dice_per_class[1]) if self.num_classes > 1 else 0.0

        return {
            "iou": mean_iou,
            "iou_per_class": iou_per_class.tolist(),
            "dice": mean_dice,
            "dice_per_class": dice_per_class.tolist(),
            "pixel_accuracy": pixel_accuracy,
            "rooftop_iou": rooftop_iou,
            "rooftop_dice": rooftop_dice,
        }

    @staticmethod
    def compute_single(
        prediction: torch.Tensor | np.ndarray,
        target: torch.Tensor | np.ndarray,
    ) -> dict[str, float]:
        """
        Compute metrics for a single image pair (no accumulation).

        Args:
            prediction: Predicted binary mask (H, W).
            target: Ground truth binary mask (H, W).

        Returns:
            Dictionary with iou, dice, and pixel_accuracy.
        """
        if isinstance(prediction, torch.Tensor):
            pred = prediction.cpu().numpy().astype(bool)
        else:
            pred = prediction.astype(bool)

        if isinstance(target, torch.Tensor):
            tgt = target.cpu().numpy().astype(bool)
        else:
            tgt = target.astype(bool)

        intersection = np.logical_and(pred, tgt).sum()
        union = np.logical_or(pred, tgt).sum()

        iou = float(intersection / union) if union > 0 else 0.0
        dice = (
            float(2.0 * intersection / (pred.sum() + tgt.sum()))
            if (pred.sum() + tgt.sum()) > 0
            else 0.0
        )
        total = pred.size
        correct = np.sum(pred == tgt)
        pixel_accuracy = float(correct / total) if total > 0 else 0.0

        return {
            "iou": iou,
            "dice": dice,
            "pixel_accuracy": pixel_accuracy,
        }
