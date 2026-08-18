"""
Unit tests for checkpoint backward compatibility and strict model loading.
"""

from pathlib import Path
import pytest
import torch

from models.registry import ensure_models_registered, load_model_from_checkpoint
from utils.checkpoint_manager import find_best_available_checkpoint


class TestCheckpointCompatibility:
    """Tests for checkpoint schema backward compatibility and strict loading."""

    def test_legacy_checkpoint_loads_strictly(self):
        """Verify that existing trained checkpoints load with strict=True without error."""
        ensure_models_registered()

        ckpt_path = Path("outputs/experiments/exp_002/checkpoints/best_loss.pth")
        if not ckpt_path.exists():
            pytest.skip("Legacy checkpoint not found for testing")

        device = torch.device("cpu")
        model = load_model_from_checkpoint(ckpt_path, device=device)

        assert model is not None
        assert getattr(model, "model_type", None) == "segformer"
        assert getattr(model, "num_labels", None) == 2
        assert getattr(model, "image_size", None) == 512

    def test_checkpoint_manager_finds_best_checkpoint(self):
        """Verify find_best_available_checkpoint accurately discovers best_loss.pth."""
        ckpt = find_best_available_checkpoint()
        assert ckpt.exists()
        assert ckpt.suffix == ".pth"
