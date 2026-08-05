"""
Unit tests for forward pass execution and inference pipeline.
"""

import pytest

pytest.importorskip("cv2")
pytest.importorskip("transformers")

import numpy as np
import torch

from models.segformer_model import SegFormerModel
from inference.inferencer import SegmentationInferencer, SegmentationResult


class TestForwardAndInference:
    """Tests for model forward pass and inferencer run execution."""

    def test_single_forward_pass_output_shape(self):
        """Single forward pass outputs correctly shaped logits."""
        model = SegFormerModel(backbone="nvidia/mit-b2", num_labels=2, image_size=128)
        model.eval()

        dummy_batch = torch.randn(1, 3, 128, 128)
        outputs = model(dummy_batch)

        assert "upsampled_logits" in outputs
        assert outputs["upsampled_logits"].shape == (1, 2, 128, 128)

    def test_predict_dictionary_contract(self):
        """Predict method returns binary mask and confidence maps."""
        model = SegFormerModel(backbone="nvidia/mit-b2", num_labels=2, image_size=128)
        dummy_batch = torch.randn(1, 3, 128, 128)

        preds = model.predict(dummy_batch)

        assert "binary_mask" in preds
        assert "confidence_map" in preds
        assert "probabilities" in preds

        assert preds["binary_mask"].shape == (1, 128, 128)
        assert preds["confidence_map"].shape == (1, 128, 128)

    def test_prediction_report_serialization(self):
        """SegmentationResult converts to prediction_report.json schema correctly."""
        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
        dummy_mask = np.zeros((100, 100), dtype=np.uint8)
        dummy_mask[20:60, 20:60] = 1

        result = SegmentationResult(
            original_image=dummy_img,
            binary_mask=dummy_mask,
            overlay_image=dummy_img,
            polygon=[],
            roof_area_pixels=1600,
            roof_area_percent=16.0,
            usable_area_percent=14.4,
            confidence=0.95,
            model="SegFormer",
            version="v1",
            rooftop_area_m2_estimate=1600.0,
            is_estimated=False,
            processing_time_ms=50.0,
            image_path="test.jpg",
        )

        report = result.to_prediction_report()

        assert report["roof_area_pixels"] == 1600
        assert report["roof_area_percent"] == 16.0
        assert report["usable_area_percent"] == 14.4
        assert report["confidence"] == 0.95
        assert report["model"] == "SegFormer"
        assert report["version"] == "v1"
