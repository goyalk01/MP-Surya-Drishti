"""
SegFormer model implementation for the rooftop segmentation framework.

This is the FIRST registered model in the MP Surya-Drishti CV framework.
It extends ``BaseSegmentationModel`` and registers itself via the
``@register_model("segformer")`` decorator.

To add a new model (e.g., DeepLabV3+), create a similar file that:
    1. Extends ``BaseSegmentationModel``
    2. Uses ``@register_model("deeplabv3plus")``
    3. Implements all abstract methods
    4. Gets imported in ``models/registry.py::ensure_models_registered()``

That's it — the entire pipeline (train, infer, evaluate) works automatically.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F

from models.base_model import BaseSegmentationModel
from models.registry import register_model

logger = logging.getLogger(__name__)


@register_model("segformer")
class SegFormerModel(BaseSegmentationModel):
    """
    SegFormer implementation for semantic segmentation.

    Wraps HuggingFace ``SegformerForSemanticSegmentation`` behind the
    framework's standard interface. Uses a pretrained Mix Transformer
    (MiT) backbone with a lightweight MLP decoder head.

    Args:
        backbone: HuggingFace model identifier (e.g., ``nvidia/mit-b2``).
        num_labels: Number of output classes (default 2: background + rooftop).
        id2label: Mapping from class index to label name.
        label2id: Mapping from label name to class index.
        confidence_threshold: Threshold for binary mask generation.
        image_size: Expected input image size (for upsampling logits).
    """

    @property
    def model_type(self) -> str:
        return "segformer"

    def __init__(
        self,
        backbone: str = "nvidia/mit-b2",
        num_labels: int = 2,
        id2label: Optional[dict[int, str]] = None,
        label2id: Optional[dict[str, int]] = None,
        confidence_threshold: float = 0.5,
        image_size: int = 512,
    ) -> None:
        super().__init__(
            backbone=backbone,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            confidence_threshold=confidence_threshold,
            image_size=image_size,
        )

        # Load SegFormer from HuggingFace
        from transformers import SegformerForSemanticSegmentation

        self.model = SegformerForSemanticSegmentation.from_pretrained(
            backbone,
            num_labels=num_labels,
            id2label=self.id2label,
            label2id=self.label2id,
            ignore_mismatched_sizes=True,
        )

        logger.info(
            "Loaded SegFormer backbone '%s' with %d labels (head reinitialized)",
            backbone,
            num_labels,
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass through SegFormer.

        Args:
            pixel_values: Input tensor of shape (B, C, H, W).
            labels: Optional ground truth masks of shape (B, H, W) with
                class indices.

        Returns:
            Dictionary containing:
                - logits: Raw model output (B, num_labels, H/4, W/4).
                - upsampled_logits: Logits interpolated to input size
                    (B, num_labels, H, W).
                - loss: Cross-entropy loss (only if labels provided).
        """
        outputs = self.model(
            pixel_values=pixel_values,
            labels=labels,
        )

        logits = outputs.logits  # (B, num_labels, H/4, W/4)

        # Upsample logits to original input resolution
        upsampled_logits = F.interpolate(
            logits,
            size=(pixel_values.shape[2], pixel_values.shape[3]),
            mode="bilinear",
            align_corners=False,
        )

        result = {
            "logits": logits,
            "upsampled_logits": upsampled_logits,
        }

        if outputs.loss is not None:
            result["loss"] = outputs.loss

        return result

    def _get_state_dict(self) -> dict[str, Any]:
        """Return the inner HuggingFace model's state dict."""
        return self.model.state_dict()

    def _load_state_dict_from_checkpoint(
        self, state_dict: dict[str, Any]
    ) -> None:
        """Load weights into the inner HuggingFace model."""
        self.model.load_state_dict(state_dict)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: Optional[torch.device] = None,
    ) -> "SegFormerModel":
        """
        Create a SegFormerModel instance from a saved checkpoint.

        Args:
            checkpoint_path: Path to the checkpoint file.
            device: Device to place the model on.

        Returns:
            Initialized SegFormerModel with loaded weights.
        """
        map_location = device if device else "cpu"
        checkpoint = torch.load(
            checkpoint_path, map_location=map_location, weights_only=False
        )

        model = cls(
            backbone=checkpoint["backbone"],
            num_labels=checkpoint["num_labels"],
            id2label=checkpoint.get("id2label", {0: "background", 1: "rooftop"}),
            label2id=checkpoint.get("label2id", {"background": 0, "rooftop": 1}),
            confidence_threshold=checkpoint.get("confidence_threshold", 0.5),
            image_size=checkpoint.get("image_size", 512),
        )

        model.model.load_state_dict(checkpoint["model_state_dict"])

        if device:
            model = model.to(device)

        logger.info(
            "Loaded SegFormerModel from checkpoint '%s' (epoch=%d)",
            checkpoint_path,
            checkpoint.get("epoch", -1),
        )

        return model


# Keep backward-compatible alias
RooftopSegFormer = SegFormerModel
