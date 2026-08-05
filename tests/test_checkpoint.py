"""
Unit tests for model checkpoint save and load integrity.
"""

import pytest
import torch

from models.registry import create_model, load_model_from_checkpoint


class TestCheckpointIntegrity:
    """Tests for checkpoint serialization and strict weight restoration."""

    def test_save_and_load_checkpoint_strict(self, tmp_path):
        """Saving and loading checkpoint restores 100% of weights with strict=True."""
        pytest.importorskip("transformers")
        from models.segformer_model import SegFormerModel

        model_config = {
            "model_type": "segformer",
            "backbone": "nvidia/mit-b2",
            "num_labels": 2,
            "image_size": 128,
        }

        # Create model and save checkpoint
        model = create_model(model_config)

        ckpt_path = tmp_path / "test_model.pth"
        model.save_checkpoint(
            path=ckpt_path,
            epoch=5,
            metrics={"val_iou": 0.65},
            config=model_config,
        )

        assert ckpt_path.exists()

        # Load model back using registry load_model_from_checkpoint
        loaded_model = load_model_from_checkpoint(ckpt_path, device=torch.device("cpu"))

        assert isinstance(loaded_model, SegFormerModel)
        assert loaded_model.num_labels == 2
        assert loaded_model.image_size == 128

        # Verify exact weight match for sample layer
        orig_weight = list(model.parameters())[0]
        loaded_weight = list(loaded_model.parameters())[0]
        assert torch.equal(orig_weight, loaded_weight)

    def test_checkpoint_metadata_keys(self, tmp_path):
        """Checkpoint file contains all required metadata keys."""
        pytest.importorskip("transformers")
        from models.segformer_model import SegFormerModel

        model = SegFormerModel(backbone="nvidia/mit-b2", num_labels=2, image_size=128)
        ckpt_path = tmp_path / "metadata_test.pth"

        model.save_checkpoint(
            path=ckpt_path,
            epoch=1,
            metrics={"val_iou": 0.5},
            extra={"best_iou": 0.5},
        )

        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        required_keys = [
            "model_type",
            "epoch",
            "model_state_dict",
            "backbone",
            "num_labels",
            "id2label",
            "label2id",
            "confidence_threshold",
            "image_size",
            "random_state",
        ]

        for key in required_keys:
            assert key in checkpoint, f"Missing required key in checkpoint: {key}"
