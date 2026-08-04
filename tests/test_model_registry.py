"""
Unit tests for the framework's model registry and BaseSegmentationModel interface.
"""

from typing import Any, Optional
import numpy as np
import pytest
import torch

from models.base_model import BaseSegmentationModel
from models.registry import (
    create_model,
    ensure_models_registered,
    list_registered_models,
    register_model,
)


@register_model("mock_model")
class DummySegmentationModel(BaseSegmentationModel):
    """Dummy model for framework registry testing."""

    @property
    def model_type(self) -> str:
        return "mock_model"

    def __init__(
        self,
        backbone: str = "dummy_backbone",
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
        self.conv = torch.nn.Conv2d(3, num_labels, kernel_size=1)

    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        logits = self.conv(pixel_values)
        res = {"upsampled_logits": logits}
        if labels is not None:
            res["loss"] = torch.tensor(0.5, requires_grad=True)
        return res

    def _get_state_dict(self) -> dict[str, Any]:
        return self.state_dict()

    def _load_state_dict_from_checkpoint(self, state_dict: dict[str, Any]) -> None:
        self.load_state_dict(state_dict)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: Optional[torch.device] = None,
    ) -> "DummySegmentationModel":
        model = cls()
        if device:
            model = model.to(device)
        return model


class TestModelRegistry:
    """Tests for registry functions and BaseSegmentationModel methods."""

    def test_registered_model_in_list(self):
        """Registry lists the mock model."""
        registered = list_registered_models()
        assert "mock_model" in registered

    def test_create_model_from_config(self):
        """create_model instantiates correct model with parameters."""
        config = {
            "model_type": "mock_model",
            "backbone": "test_backbone",
            "num_labels": 2,
            "confidence_threshold": 0.6,
            "image_size": 256,
        }
        model = create_model(config)
        assert isinstance(model, DummySegmentationModel)
        assert model.model_type == "mock_model"
        assert model.confidence_threshold == 0.6
        assert model.image_size == 256

    def test_create_model_missing_type_raises(self):
        """create_model raises ValueError when model_type is missing."""
        with pytest.raises(ValueError, match="must contain 'model_type'"):
            create_model({"backbone": "test"})

    def test_create_model_unregistered_type_raises(self):
        """create_model raises ValueError for unknown model_type."""
        with pytest.raises(ValueError, match="Unknown model_type"):
            create_model({"model_type": "non_existent_model"})

    def test_generic_predict_output_structure(self):
        """BaseSegmentationModel.predict produces correct dict keys and shapes."""
        model = DummySegmentationModel(image_size=128)
        dummy_input = torch.randn(2, 3, 128, 128)
        prediction = model.predict(dummy_input)

        assert "binary_mask" in prediction
        assert "confidence_map" in prediction
        assert "probabilities" in prediction
        assert prediction["binary_mask"].shape == (2, 128, 128)
        assert prediction["confidence_map"].shape == (2, 128, 128)
        assert prediction["probabilities"].shape == (2, 2, 128, 128)

    def test_checkpoint_save_includes_model_type(self, tmp_path):
        """save_checkpoint saves model_type in the metadata dictionary."""
        model = DummySegmentationModel()
        ckpt_path = tmp_path / "test_ckpt.pth"
        model.save_checkpoint(ckpt_path, epoch=1)

        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        assert checkpoint["model_type"] == "mock_model"
        assert checkpoint["epoch"] == 1
