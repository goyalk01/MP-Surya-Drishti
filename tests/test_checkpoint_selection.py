"""
Unit tests for intelligent checkpoint discovery and validation-metric selection.
"""

from pathlib import Path
import pytest
import torch

from utils.checkpoint_manager import find_best_available_checkpoint, CheckpointManager


class TestCheckpointSelection:
    """Tests for auto-checkpoint selection based strictly on validation metrics."""

    def test_find_best_available_checkpoint_prefers_higher_val_iou(self, tmp_path):
        """Auto-selection chooses the checkpoint with highest stored validation score."""
        ckpt_a = tmp_path / "model_a.pth"
        ckpt_b = tmp_path / "model_b.pth"

        # Model A: val rooftop_iou = 0.30
        torch.save(
            {"epoch": 10, "metrics": {"rooftop_iou": 0.30, "val_iou": 0.50}},
            ckpt_a,
        )
        # Model B: val rooftop_iou = 0.45
        torch.save(
            {"epoch": 15, "metrics": {"rooftop_iou": 0.45, "val_iou": 0.60}},
            ckpt_b,
        )

        best = find_best_available_checkpoint(search_dirs=[tmp_path])
        assert best == ckpt_b

    def test_find_best_available_checkpoint_fallback_hierarchy(self, tmp_path):
        """Fallback hierarchy prefers best_loss > best_iou > latest if scores are tied."""
        p_loss = tmp_path / "best_loss.pth"
        p_iou = tmp_path / "best_iou.pth"
        p_latest = tmp_path / "latest.pth"

        for p in [p_loss, p_iou, p_latest]:
            torch.save({"epoch": 5, "metrics": {}}, p)

        best = find_best_available_checkpoint(search_dirs=[tmp_path])
        assert best == p_loss

    def test_find_best_available_checkpoint_existing_experiments(self):
        """In the existing exp_002, best_loss.pth is selected due to higher stored val score."""
        exp_ckpt_dir = Path("outputs/experiments/exp_002/checkpoints")
        if exp_ckpt_dir.exists():
            best = find_best_available_checkpoint(search_dirs=[exp_ckpt_dir])
            assert best.name == "best_loss.pth"

    def test_latest_and_best_loss_tensor_equivalence(self):
        """Verify that latest.pth and best_loss.pth contain identical weights in exp_002."""
        p_loss = Path("outputs/experiments/exp_002/checkpoints/best_loss.pth")
        p_latest = Path("outputs/experiments/exp_002/checkpoints/latest.pth")

        if p_loss.exists() and p_latest.exists():
            ckpt_loss = torch.load(p_loss, map_location="cpu", weights_only=False)
            ckpt_latest = torch.load(p_latest, map_location="cpu", weights_only=False)

            sd_loss = ckpt_loss["model_state_dict"]
            sd_latest = ckpt_latest["model_state_dict"]

            assert set(sd_loss.keys()) == set(sd_latest.keys())
            for k in sd_loss:
                assert torch.equal(sd_loss[k], sd_latest[k]), f"Tensor mismatch in {k}"
            assert ckpt_loss.get("epoch") == ckpt_latest.get("epoch")
