"""
Unit tests for the segmentation metrics module.
"""

import numpy as np
import pytest
import torch

from evaluation.metrics import SegmentationMetrics


class TestSegmentationMetrics:
    """Tests for the SegmentationMetrics class."""

    def test_perfect_prediction_iou(self):
        """IoU should be 1.0 for identical prediction and target."""
        metrics = SegmentationMetrics(num_classes=2)

        pred = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]])
        target = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]])

        metrics.update(pred, target)
        result = metrics.compute()

        assert abs(result["iou"] - 1.0) < 1e-6
        assert abs(result["dice"] - 1.0) < 1e-6
        assert abs(result["pixel_accuracy"] - 1.0) < 1e-6

    def test_zero_overlap_iou(self):
        """IoU should be 0.0 when prediction and target don't overlap."""
        metrics = SegmentationMetrics(num_classes=2)

        pred = torch.tensor([[1, 1, 0, 0], [1, 1, 0, 0]])
        target = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]])

        metrics.update(pred, target)
        result = metrics.compute()

        assert result["rooftop_iou"] == 0.0

    def test_partial_overlap(self):
        """IoU should be between 0 and 1 for partial overlap."""
        metrics = SegmentationMetrics(num_classes=2)

        pred = torch.tensor([[0, 1, 1, 0], [0, 1, 1, 0]])
        target = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]])

        metrics.update(pred, target)
        result = metrics.compute()

        assert 0.0 < result["iou"] < 1.0

    def test_reset_clears_state(self):
        """reset() should zero the confusion matrix."""
        metrics = SegmentationMetrics(num_classes=2)

        pred = torch.tensor([[1, 1], [1, 1]])
        target = torch.tensor([[1, 1], [1, 1]])

        metrics.update(pred, target)
        metrics.reset()

        assert metrics.confusion_matrix.sum() == 0

    def test_compute_single(self):
        """compute_single should work for a single image pair."""
        pred = np.array([[0, 1, 1], [0, 1, 1]], dtype=np.uint8)
        target = np.array([[0, 1, 1], [0, 1, 1]], dtype=np.uint8)

        result = SegmentationMetrics.compute_single(pred, target)

        assert abs(result["iou"] - 1.0) < 1e-6
        assert abs(result["dice"] - 1.0) < 1e-6

    def test_accumulation_across_batches(self):
        """Metrics should accumulate correctly across multiple updates."""
        metrics = SegmentationMetrics(num_classes=2)

        # Batch 1: perfect
        pred1 = torch.tensor([[1, 1], [0, 0]])
        target1 = torch.tensor([[1, 1], [0, 0]])

        # Batch 2: also perfect
        pred2 = torch.tensor([[0, 0], [1, 1]])
        target2 = torch.tensor([[0, 0], [1, 1]])

        metrics.update(pred1, target1)
        metrics.update(pred2, target2)
        result = metrics.compute()

        assert abs(result["iou"] - 1.0) < 1e-6
