"""
Abstract base class for all segmentation models in the MP Surya-Drishti framework.

Every segmentation model (SegFormer, DeepLabV3+, Mask2Former, U-Net, etc.)
must extend ``BaseSegmentationModel`` and implement the required interface.

This ensures that the training pipeline, inference engine, evaluation module,
and all downstream consumers (shadow detection, panel placement, API, dashboard)
work with ANY model implementation without code changes.

CONTRACT:
    forward()  → returns {"upsampled_logits": Tensor(B, C, H, W), "loss": optional}
    predict()  → returns {"binary_mask": Tensor(B, H, W), "confidence_map": Tensor(B, H, W), "probabilities": Tensor(B, C, H, W)}
    save_checkpoint() / load_checkpoint() / from_checkpoint() → checkpoint persistence
    get_param_count()  → parameter counts
    model_type (property) → string identifier used in registry and checkpoints
"""

from __future__ import annotations

import logging
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class BaseSegmentationModel(ABC, nn.Module):
    """
    Abstract base class that defines the interface for all segmentation models.

    Subclasses must implement:
        - ``model_type`` property
        - ``forward()``
        - ``_get_state_dict()`` (model weights to save)
        - ``_load_state_dict_from_checkpoint()`` (restore model weights)
        - ``from_checkpoint()`` class method

    Args:
        backbone: Model backbone identifier (e.g., ``nvidia/mit-b2``).
        num_labels: Number of segmentation classes.
        id2label: Mapping from class index to label name.
        label2id: Mapping from label name to class index.
        confidence_threshold: Threshold for binary mask generation.
        image_size: Expected input image size.
    """

    def __init__(
        self,
        backbone: str,
        num_labels: int = 2,
        id2label: Optional[dict[int, str]] = None,
        label2id: Optional[dict[str, int]] = None,
        confidence_threshold: float = 0.5,
        image_size: int = 512,
    ) -> None:
        super().__init__()

        self.backbone_name = backbone
        self.num_labels = num_labels
        self.confidence_threshold = confidence_threshold
        self.image_size = image_size

        if id2label is None:
            id2label = {0: "background", 1: "rooftop"}
        if label2id is None:
            label2id = {"background": 0, "rooftop": 1}

        self.id2label = id2label
        self.label2id = label2id

    # ----------------------------------------------------------------
    # ABSTRACT INTERFACE — must be implemented by every model
    # ----------------------------------------------------------------

    @property
    @abstractmethod
    def model_type(self) -> str:
        """Unique string identifier for this model type."""
        ...

    @abstractmethod
    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass through the model.

        Must return dict with at least "upsampled_logits" (B, num_labels, H, W).
        """
        ...

    @abstractmethod
    def _get_state_dict(self) -> dict[str, Any]:
        """Return the model's state dictionary for checkpointing."""
        ...

    @abstractmethod
    def _load_state_dict_from_checkpoint(
        self, state_dict: dict[str, Any]
    ) -> None:
        """Load model weights strictly from a checkpoint state dictionary."""
        ...

    @classmethod
    @abstractmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: Optional[torch.device] = None,
    ) -> BaseSegmentationModel:
        """Create a model instance and restore weights from a saved checkpoint."""
        ...

    # ----------------------------------------------------------------
    # SHARED IMPLEMENTATIONS — inherited by all models
    # ----------------------------------------------------------------

    def predict(
        self,
        pixel_values: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Generate predictions (binary masks and confidence maps).

        Args:
            pixel_values: Input tensor (B, C, H, W).

        Returns:
            Dictionary containing:
                - binary_mask: (B, H, W) with values {0, 1}
                - confidence_map: (B, H, W) rooftop class probability
                - probabilities: (B, num_labels, H, W) full class probs
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(pixel_values)
            upsampled_logits = outputs["upsampled_logits"]

            probabilities = F.softmax(upsampled_logits, dim=1)
            foreground_prob = probabilities[:, 1, :, :]
            binary_mask = (foreground_prob >= self.confidence_threshold).long()

        return {
            "binary_mask": binary_mask,
            "confidence_map": foreground_prob,
            "probabilities": probabilities,
        }

    def get_confidence(
        self,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-pixel confidence scores from logits."""
        probabilities = F.softmax(logits, dim=1)
        confidence, _ = torch.max(probabilities, dim=1)
        return confidence

    def save_checkpoint(
        self,
        path: str | Path,
        epoch: int,
        optimizer_state: Optional[dict] = None,
        scheduler_state: Optional[dict] = None,
        scaler_state: Optional[dict] = None,
        metrics: Optional[dict[str, float]] = None,
        config: Optional[dict[str, Any]] = None,
        extra: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Save a full training checkpoint containing all parameters and state objects.

        Args:
            path: File path for the checkpoint.
            epoch: Current training epoch.
            optimizer_state: Optimizer state dict.
            scheduler_state: LR scheduler state dict.
            scaler_state: AMP GradScaler state dict.
            metrics: Validation metrics at this checkpoint.
            config: Full experiment configuration dictionary.
            extra: Additional metadata (e.g., best_iou, best_loss).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        random_state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }

        checkpoint = {
            "model_type": self.model_type,
            "epoch": epoch,
            "model_state_dict": self._get_state_dict(),
            "backbone": self.backbone_name,
            "num_labels": self.num_labels,
            "id2label": self.id2label,
            "label2id": self.label2id,
            "confidence_threshold": self.confidence_threshold,
            "image_size": self.image_size,
            "random_state": random_state,
        }

        if optimizer_state is not None:
            checkpoint["optimizer_state_dict"] = optimizer_state
        if scheduler_state is not None:
            checkpoint["scheduler_state_dict"] = scheduler_state
        if scaler_state is not None:
            checkpoint["scaler_state_dict"] = scaler_state
        if metrics is not None:
            checkpoint["metrics"] = metrics
        if config is not None:
            checkpoint["config"] = config
        if extra is not None:
            checkpoint["extra"] = extra

        torch.save(checkpoint, path)
        logger.info(
            "Saved %s checkpoint to %s (epoch=%d)",
            self.model_type,
            path,
            epoch,
        )

    def load_checkpoint(
        self,
        path: str | Path,
        device: Optional[torch.device] = None,
        restore_random_state: bool = True,
    ) -> dict[str, Any]:
        """
        Load a training checkpoint into the current model instance.

        Args:
            path: Path to the checkpoint file.
            device: Device to map tensors to.
            restore_random_state: Whether to restore RNG states for reproducible training resume.

        Returns:
            Full checkpoint dictionary.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        map_location = device if device else "cpu"
        checkpoint = torch.load(
            path, map_location=map_location, weights_only=False
        )

        self._load_state_dict_from_checkpoint(checkpoint["model_state_dict"])

        if restore_random_state and "random_state" in checkpoint:
            rng_data = checkpoint["random_state"]
            if "python" in rng_data:
                random.setstate(rng_data["python"])
            if "numpy" in rng_data:
                np.random.set_state(rng_data["numpy"])
            if "torch" in rng_data:
                torch.set_rng_state(rng_data["torch"])
            if "cuda" in rng_data and rng_data["cuda"] is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng_data["cuda"])

        logger.info(
            "Loaded %s checkpoint from %s (epoch=%d)",
            self.model_type,
            path,
            checkpoint.get("epoch", -1),
        )

        return checkpoint

    def get_param_count(self) -> dict[str, int]:
        """Get the number of model parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        return {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable,
        }
