"""
Unit tests for segmentation loss functions: DiceLoss, FocalLoss, FocalDiceLoss,
and get_loss_function factory.
"""

import pytest
import torch

from training.losses import (
    DiceLoss,
    FocalLoss,
    FocalDiceLoss,
    CombinedSegmentationLoss,
    get_loss_function,
)


class TestSegmentationLosses:
    """Tests for loss numerical stability, gradient flow, and loss formulation."""

    def test_dice_loss_bounds(self):
        """Test DiceLoss outputs 0.0 on identical prediction and >0.0 on mismatch."""
        loss_fn = DiceLoss(smooth=1.0)

        # Perfect prediction on 10x10 mask
        logits_perfect = torch.zeros(1, 2, 10, 10)
        logits_perfect[:, 1, :, :] = 10.0
        logits_perfect[:, 0, :, :] = -10.0
        targets = torch.ones(1, 10, 10, dtype=torch.long)
        loss_perf = loss_fn(logits_perfect, targets)
        assert loss_perf.item() < 0.05

        # Completely wrong prediction on 10x10 mask
        logits_wrong = torch.zeros(1, 2, 10, 10)
        logits_wrong[:, 0, :, :] = 10.0
        logits_wrong[:, 1, :, :] = -10.0
        loss_wrong = loss_fn(logits_wrong, targets)
        assert loss_wrong.item() > 0.90

    def test_focal_loss_numerical_stability_and_gradients(self):
        """Test FocalLoss computes finite values without NaN under extreme logits."""
        loss_fn = FocalLoss(gamma=2.0, alpha=0.25)

        # Extreme logits to test overflow / underflow
        logits = torch.randn(4, 2, 64, 64, requires_grad=True)
        targets = torch.randint(0, 2, (4, 64, 64), dtype=torch.long)

        loss = loss_fn(logits, targets)

        assert not torch.isnan(loss).any()
        assert not torch.isinf(loss).any()
        assert loss.item() >= 0.0

        # Test gradient propagation
        loss.backward()
        assert logits.grad is not None
        assert not torch.isnan(logits.grad).any()

    def test_focal_loss_ignore_index(self):
        """Test that pixels marked with ignore_index are omitted from loss computation."""
        loss_fn = FocalLoss(gamma=2.0, alpha=0.25, ignore_index=255)

        logits = torch.randn(2, 2, 32, 32)
        # All pixels ignored
        targets = torch.full((2, 32, 32), 255, dtype=torch.long)

        loss = loss_fn(logits, targets)
        assert loss.item() == 0.0

    def test_focal_dice_loss_contract(self):
        """Test FocalDiceLoss returns dict with loss, focal_loss, and dice_loss."""
        loss_fn = FocalDiceLoss(
            focal_weight=0.5,
            dice_weight=0.5,
            focal_gamma=2.0,
            focal_alpha=0.25,
        )

        logits = torch.randn(2, 2, 32, 32, requires_grad=True)
        targets = torch.randint(0, 2, (2, 32, 32), dtype=torch.long)

        out = loss_fn(logits, targets)

        assert isinstance(out, dict)
        assert "loss" in out
        assert "focal_loss" in out
        assert "dice_loss" in out

        # Combined loss = 0.5 * focal + 0.5 * dice
        expected_combined = 0.5 * out["focal_loss"] + 0.5 * out["dice_loss"]
        assert torch.isclose(out["loss"], expected_combined)

        out["loss"].backward()
        assert logits.grad is not None

    def test_get_loss_function_factory(self):
        """Test get_loss_function returns correct loss module based on configuration."""
        cfg_focal = {"loss": {"type": "focal_dice", "focal_gamma": 2.5}}
        fn_focal = get_loss_function(cfg_focal)
        assert isinstance(fn_focal, FocalDiceLoss)
        assert fn_focal.focal_loss.gamma == 2.5

        cfg_ce = {"loss": {"name": "ce_dice", "bce_weight": 0.7}}
        fn_ce = get_loss_function(cfg_ce)
        assert isinstance(fn_ce, CombinedSegmentationLoss)
        assert fn_ce.bce_weight == 0.7
