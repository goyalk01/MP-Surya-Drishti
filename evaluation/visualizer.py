"""
Visualization utilities for segmentation results.

Provides plotting functions for:
- Side-by-side comparison (original, ground truth, prediction)
- Colored overlay on original image
- Training curves (loss, IoU, Dice over epochs)
- Epoch sample snapshots
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

# Use non-interactive backend for server/Colab compatibility
plt.switch_backend("agg")


class SegmentationVisualizer:
    """
    Generate segmentation visualizations and training plots.

    All plots are saved to disk rather than displayed interactively,
    making them compatible with headless environments like Colab.

    Args:
        output_dir: Directory to save visualization outputs.
        overlay_color: RGB color for the rooftop overlay (default: green).
        overlay_alpha: Transparency of the overlay (0.0–1.0).
    """

    def __init__(
        self,
        output_dir: str | Path = "outputs/visualizations",
        overlay_color: tuple[int, int, int] = (0, 255, 0),
        overlay_alpha: float = 0.4,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.overlay_color = overlay_color
        self.overlay_alpha = overlay_alpha

        logger.info("SegmentationVisualizer output dir: %s", self.output_dir)

    def plot_comparison(
        self,
        original: np.ndarray,
        ground_truth: np.ndarray,
        prediction: np.ndarray,
        title: str = "Segmentation Comparison",
        filename: Optional[str] = None,
        metrics: Optional[dict[str, float]] = None,
    ) -> Path:
        """
        Generate a 3-panel comparison: Original | Ground Truth | Prediction.

        Args:
            original: Original image (H, W, 3) in RGB uint8.
            ground_truth: Ground truth mask (H, W) binary.
            prediction: Predicted mask (H, W) binary.
            title: Plot title.
            filename: Output filename (auto-generated if None).
            metrics: Optional metrics dict to display as text.

        Returns:
            Path to the saved figure.
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        axes[0].imshow(original)
        axes[0].set_title("Original Image", fontsize=14)
        axes[0].axis("off")

        axes[1].imshow(ground_truth, cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("Ground Truth", fontsize=14)
        axes[1].axis("off")

        axes[2].imshow(prediction, cmap="gray", vmin=0, vmax=1)
        axes[2].set_title("Prediction", fontsize=14)
        axes[2].axis("off")

        # Add metrics text if provided
        if metrics:
            metric_text = " | ".join(
                f"{k}: {v:.4f}" for k, v in metrics.items()
            )
            fig.suptitle(f"{title}\n{metric_text}", fontsize=16, y=1.02)
        else:
            fig.suptitle(title, fontsize=16, y=1.02)

        plt.tight_layout()

        if filename is None:
            filename = "comparison.png"
        save_path = self.output_dir / filename
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        logger.info("Saved comparison plot: %s", save_path)
        return save_path

    def plot_overlay(
        self,
        original: np.ndarray,
        mask: np.ndarray,
        title: str = "Rooftop Overlay",
        filename: Optional[str] = None,
    ) -> tuple[np.ndarray, Path]:
        """
        Create and save a colored overlay of the mask on the original image.

        Args:
            original: Original image (H, W, 3) in RGB uint8.
            mask: Binary mask (H, W).
            title: Plot title.
            filename: Output filename.

        Returns:
            Tuple of (overlay_image, save_path).
        """
        overlay = self.create_overlay(original, mask)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        axes[0].imshow(original)
        axes[0].set_title("Original", fontsize=14)
        axes[0].axis("off")

        axes[1].imshow(overlay)
        axes[1].set_title("Rooftop Overlay", fontsize=14)
        axes[1].axis("off")

        fig.suptitle(title, fontsize=16)
        plt.tight_layout()

        if filename is None:
            filename = "overlay.png"
        save_path = self.output_dir / filename
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        logger.info("Saved overlay plot: %s", save_path)
        return overlay, save_path

    def create_overlay(
        self,
        original: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """
        Create a colored overlay image without plotting.

        Args:
            original: Original image (H, W, 3) in RGB uint8.
            mask: Binary mask (H, W).

        Returns:
            Overlay image (H, W, 3) in RGB uint8.
        """
        overlay = original.copy()

        # Create colored mask
        color_mask = np.zeros_like(original)
        color_mask[mask > 0] = self.overlay_color

        # Blend
        mask_region = mask > 0
        overlay[mask_region] = (
            (1 - self.overlay_alpha) * original[mask_region]
            + self.overlay_alpha * color_mask[mask_region]
        ).astype(np.uint8)

        return overlay

    def plot_training_curves(
        self,
        history: list[dict[str, float]],
        filename: str = "training_curves.png",
    ) -> Path:
        """
        Plot training and validation curves over epochs.

        Generates a 2×2 grid: Loss, IoU, Dice, Learning Rate.

        Args:
            history: List of per-epoch metric dictionaries.
            filename: Output filename.

        Returns:
            Path to the saved figure.
        """
        epochs = [h.get("epoch", i + 1) for i, h in enumerate(history)]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Loss
        train_loss = [h.get("train_loss", 0) for h in history]
        val_loss = [h.get("val_loss", 0) for h in history]
        axes[0, 0].plot(epochs, train_loss, "b-", label="Train Loss", linewidth=2)
        axes[0, 0].plot(epochs, val_loss, "r-", label="Val Loss", linewidth=2)
        axes[0, 0].set_title("Loss", fontsize=14)
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # IoU
        val_iou = [h.get("val_iou", 0) for h in history]
        axes[0, 1].plot(epochs, val_iou, "g-", label="Val IoU", linewidth=2)
        axes[0, 1].set_title("IoU", fontsize=14)
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Dice
        val_dice = [h.get("val_dice", 0) for h in history]
        axes[1, 0].plot(epochs, val_dice, "m-", label="Val Dice", linewidth=2)
        axes[1, 0].set_title("Dice Score", fontsize=14)
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Learning Rate
        lr = [h.get("lr", 0) for h in history]
        axes[1, 1].plot(epochs, lr, "k-", label="Learning Rate", linewidth=2)
        axes[1, 1].set_title("Learning Rate", fontsize=14)
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_yscale("log")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        fig.suptitle("Training Progress", fontsize=16)
        plt.tight_layout()

        save_path = self.output_dir / filename
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        logger.info("Saved training curves: %s", save_path)
        return save_path

    def save_epoch_sample(
        self,
        epoch: int,
        original: np.ndarray,
        ground_truth: np.ndarray,
        prediction: np.ndarray,
        metrics: Optional[dict[str, float]] = None,
    ) -> Path:
        """
        Save a visual sample for a specific training epoch.

        Args:
            epoch: Current epoch number.
            original: Original image (H, W, 3).
            ground_truth: Ground truth mask (H, W).
            prediction: Predicted mask (H, W).
            metrics: Optional metrics for this sample.

        Returns:
            Path to the saved figure.
        """
        filename = f"epoch_{epoch:03d}_sample.png"
        return self.plot_comparison(
            original=original,
            ground_truth=ground_truth,
            prediction=prediction,
            title=f"Epoch {epoch}",
            filename=filename,
            metrics=metrics,
        )
