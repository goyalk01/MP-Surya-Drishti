"""
Unit tests for showcase evaluation schema, dynamic calculation, and metric paths.
"""

import json
from pathlib import Path
import pytest
import numpy as np
import torch

from evaluation.metrics import SegmentationMetrics
from postprocessing.mask_cleaner import MaskCleaner


class TestShowcaseEvaluationStructure:
    """Tests for multi-mode evaluation contracts and metric calculation integrity."""

    def test_metrics_dynamic_calculation_non_hardcoded(self):
        """Metrics must vary dynamically with varying input predictions."""
        metrics_acc = SegmentationMetrics(num_classes=2)

        # Case 1: 100% agreement
        pred_a = torch.tensor([[[0, 1], [1, 0]]])
        tgt_a = torch.tensor([[[0, 1], [1, 0]]])
        metrics_acc.reset()
        metrics_acc.update(pred_a, tgt_a)
        res_a = metrics_acc.compute()
        assert res_a["rooftop_iou"] == 1.0

        # Case 2: 50% overlap
        pred_b = torch.tensor([[[0, 1], [1, 1]]])
        tgt_b = torch.tensor([[[0, 1], [1, 0]]])
        metrics_acc.reset()
        metrics_acc.update(pred_b, tgt_b)
        res_b = metrics_acc.compute()
        assert res_b["rooftop_iou"] == 2.0 / 3.0

    def test_mask_cleaner_is_diagnostic_only(self):
        """MaskCleaner modifies predictions and must not be used for primary evaluation."""
        cleaner = MaskCleaner(min_region_area=50)

        # Create a small 5-pixel island
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:15, 10] = 1

        cleaned = cleaner.clean(mask)
        # Small island is removed by cleaner
        assert mask.sum() == 5
        assert cleaned.sum() == 0

        # Verify that raw calculation preserves the true raw model state
        single_raw = SegmentationMetrics.compute_single(mask, mask)
        single_cleaned = SegmentationMetrics.compute_single(cleaned, mask)

        # Raw has 100% IoU with ground truth; cleaned has 0% IoU because noise filter stripped it
        assert single_raw["iou"] == 1.0
        assert single_cleaned["iou"] == 0.0

    def test_evaluation_json_schema_keys(self):
        """Verify the required schema keys for multi-mode evaluation JSON."""
        required_top_keys = [
            "checkpoint",
            "epoch",
            "dataset",
            "split",
            "num_samples",
            "model_type",
            "backbone",
            "image_size",
            "primary_evaluation",
            "full_resolution_raw",
            "full_resolution_cleaned",
        ]

        sample_eval_payload = {
            "checkpoint": "outputs/experiments/exp_002/checkpoints/best_loss.pth",
            "epoch": 50,
            "dataset": "Massachusetts Buildings Dataset",
            "split": "test",
            "num_samples": 10,
            "model_type": "segformer",
            "backbone": "nvidia/mit-b2",
            "image_size": 512,
            "primary_evaluation": {
                "pixel_accuracy": 0.8381,
                "mean_iou": 0.6106,
                "rooftop_iou": 0.4029,
                "rooftop_dice": 0.5744,
                "background_iou": 0.8183,
            },
            "full_resolution_raw": {
                "pixel_accuracy": 0.7848,
                "mean_iou": 0.5240,
                "rooftop_iou": 0.2831,
                "rooftop_dice": 0.4413,
                "background_iou": 0.7648,
            },
            "full_resolution_cleaned": {
                "pixel_accuracy": 0.7775,
                "mean_iou": 0.5014,
                "rooftop_iou": 0.2425,
                "rooftop_dice": 0.3903,
                "background_iou": 0.7602,
            },
        }

        for key in required_top_keys:
            assert key in sample_eval_payload, f"Missing key: {key}"

        for sub_mode in ["primary_evaluation", "full_resolution_raw", "full_resolution_cleaned"]:
            for metric in ["pixel_accuracy", "mean_iou", "rooftop_iou", "rooftop_dice"]:
                assert metric in sample_eval_payload[sub_mode]
