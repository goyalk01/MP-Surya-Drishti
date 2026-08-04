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
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

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
        - ``_build_model()`` (called during __init__)
        - ``_get_state_dict()`` (model weights to save)
        - ``_load_state_dict()`` (restore model weights)
        - ``from_checkpoint()`` class method

    The base class provides shared implementations for:
        - ``predict()`` — generic argmax-based prediction from logits
        - ``get_confidence()`` — softmax confidence extraction
        - ``save_checkpoint()`` / ``load_checkpoint()`` — persistence
        - ``get_param_count()`` — parameter counting

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
        """
        Unique string identifier for this model type.

        Used in the model registry, checkpoint files, and config.
        Examples: "segformer", "deeplabv3plus", "mask2former", "unet"
        """
        ...

    @abstractmethod
    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass through the model.

        MUST return a dictionary containing at minimum:
            - "upsampled_logits": Tensor of shape (B, num_labels, H, W)
              where H, W match the input spatial dimensions.

        OPTIONALLY may also include:
            - "loss": Scalar loss tensor (if labels provided and the
              model computes loss internally).
            - "logits": Raw logits before upsampling.

        Args:
            pixel_values: Input images (B, C, H, W).
            labels: Optional ground truth masks (B, H, W).

        Returns:
            Dictionary with at least "upsampled_logits".
        """
        ...

    @abstractmethod
    def _get_state_dict(self) -> dict[str, Any]:
        """
        Return the model's state dictionary for checkpointing.

        This allows models that wrap inner modules (e.g., HuggingFace
        models) to return the correct state dict.
        """
        ...

    @abstractmethod
    def _load_state_dict_from_checkpoint(
        self, state_dict: dict[str, Any]
    ) -> None:
        """
        Load model weights from a checkpoint state dictionary.

        Args:
            state_dict: The model state dict from a saved checkpoint.
        """
        ...

    @classmethod
    @abstractmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: Optional[torch.device] = None,
    ) -> "BaseSegmentationModel":
        """
        Create a model instance from a saved checkpoint.

        This is the standard way to load any model for inference.
        The checkpoint must contain "model_type" to identify which
        class to instantiate.

        Args:
            checkpoint_path: Path to the checkpoint file.
            device: Device to place the model on.

        Returns:
            Initialized model with loaded weights.
        """
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

        This is a generic implementation that works for any model
        that returns "upsampled_logits" from forward().

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

            # Softmax to get class probabilities
            probabilities = F.softmax(upsampled_logits, dim=1)

            # Foreground class probability (class index 1)
            foreground_prob = probabilities[:, 1, :, :]

            # Binary mask using threshold
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
        """
        Compute per-pixel confidence scores from logits.

        Args:
            logits: Raw logits tensor (B, num_labels, H, W).

        Returns:
            Confidence map (B, H, W) with values in [0, 1].
        """
        probabilities = F.softmax(logits, dim=1)
        confidence, _ = torch.max(probabilities, dim=1)
        return confidence

    def save_checkpoint(
        self,
        path: str | Path,
        epoch: int,
        optimizer_state: Optional[dict] = None,
        scheduler_state: Optional[dict] = None,
        metrics: Optional[dict[str, float]] = None,
        extra: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Save a full training checkpoint.

        The checkpoint always includes "model_type" so that
        ``load_model_from_checkpoint()`` in the registry can
        identify which class to instantiate.

        Args:
            path: File path for the checkpoint.
            epoch: Current training epoch.
            optimizer_state: Optimizer state dict.
            scheduler_state: LR scheduler state dict.
            metrics: Validation metrics at this checkpoint.
            extra: Any additional metadata to save.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

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
        }

        if optimizer_state is not None:
            checkpoint["optimizer_state_dict"] = optimizer_state
        if scheduler_state is not None:
            checkpoint["scheduler_state_dict"] = scheduler_state
        if metrics is not None:
            checkpoint["metrics"] = metrics
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
    ) -> dict[str, Any]:
        """
        Load a training checkpoint into the current model instance.

        Args:
            path: Path to the checkpoint file.
            device: Device to map tensors to.

        Returns:
            Full checkpoint dictionary (includes optimizer/scheduler
            states and metrics for training resume).
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        map_location = device if device else "cpu"
        checkpoint = torch.load(
            path, map_location=map_location, weights_only=False
        )

        self._load_state_dict_from_checkpoint(checkpoint["model_state_dict"])
        logger.info(
            "Loaded %s checkpoint from %s (epoch=%d)",
            self.model_type,
            path,
            checkpoint.get("epoch", -1),
        )

        return checkpoint

    def get_param_count(self) -> dict[str, int]:
        """
        Get the number of model parameters.

        Returns:
            Dictionary with 'total', 'trainable', and 'frozen' counts.
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        return {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable,
        }
