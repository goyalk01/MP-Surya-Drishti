"""
End-to-End Smoke Test for Native-Resolution Tiling, Focal-Dice Loss,
Checkpointing, and Sliding-Window Inference.

Proves complete pipeline correctness end-to-end on synthetic data.
"""

from pathlib import Path
import cv2
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from inference.inferencer import SegmentationInferencer
from models.registry import create_model, ensure_models_registered, load_model_from_checkpoint
from preprocessing.augmentation import AugmentationPipeline
from preprocessing.tiled_dataset import TiledMassachusettsDataset
from training.trainer import SegmentationTrainer


class TestEndToEndPipelineSmoke:
    """Fast verification that training, loss, checkpointing, and inference work seamlessly."""

    @pytest.fixture
    def synthetic_dataset(self, tmp_path):
        """Create a compact synthetic dataset with 600x600 images and masks for fast CPU testing."""
        dataset_dir = tmp_path / "mock_massachusetts"
        train_img_dir = dataset_dir / "train"
        train_msk_dir = dataset_dir / "train_labels"
        val_img_dir = dataset_dir / "val"
        val_msk_dir = dataset_dir / "val_labels"

        for d in [train_img_dir, train_msk_dir, val_img_dir, val_msk_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Generate 2 train images and 1 val image (600x600 with rooftop rectangles)
        train_imgs, train_msks = [], []
        for i in range(2):
            img = np.random.randint(40, 200, (600, 600, 3), dtype=np.uint8)
            mask = np.zeros((600, 600), dtype=np.uint8)
            # Add rooftop rectangles
            cv2.rectangle(img, (100, 100), (300, 300), (220, 180, 150), -1)
            cv2.rectangle(mask, (100, 100), (300, 300), 255, -1)
            cv2.rectangle(img, (350, 350), (550, 550), (200, 160, 140), -1)
            cv2.rectangle(mask, (350, 350), (550, 550), 255, -1)

            ip = train_img_dir / f"train_scene_{i}.png"
            mp = train_msk_dir / f"train_scene_{i}.png"
            cv2.imwrite(str(ip), img)
            cv2.imwrite(str(mp), mask)
            train_imgs.append(ip)
            train_msks.append(mp)

        # 1 val image
        val_img = np.random.randint(40, 200, (600, 600, 3), dtype=np.uint8)
        val_mask = np.zeros((600, 600), dtype=np.uint8)
        cv2.rectangle(val_img, (150, 150), (450, 450), (220, 180, 150), -1)
        cv2.rectangle(val_mask, (150, 150), (450, 450), 255, -1)

        val_ip = val_img_dir / "val_scene_0.png"
        val_mp = val_msk_dir / "val_scene_0.png"
        cv2.imwrite(str(val_ip), val_img)
        cv2.imwrite(str(val_mp), val_mask)

        return {
            "train_imgs": train_imgs,
            "train_msks": train_msks,
            "val_imgs": [val_ip],
            "val_msks": [val_mp],
            "sample_img": val_ip,
        }

    def test_full_pipeline_smoke_run(self, synthetic_dataset, tmp_path):
        """Execute 1-epoch tiled training, verify checkpoint schema, and run sliding-window inference."""
        ensure_models_registered()

        tile_size = 512
        stride = 512

        aug = AugmentationPipeline(image_size=tile_size)
        train_ds = TiledMassachusettsDataset(
            image_paths=synthetic_dataset["train_imgs"],
            mask_paths=synthetic_dataset["train_msks"],
            tile_size=tile_size,
            stride=stride,
            transform=aug.get_train_transform(),
        )
        val_ds = TiledMassachusettsDataset(
            image_paths=synthetic_dataset["val_imgs"],
            mask_paths=synthetic_dataset["val_msks"],
            tile_size=tile_size,
            stride=stride,
            transform=aug.get_val_transform(),
        )

        assert len(train_ds) > 0
        assert len(val_ds) > 0

        train_loader = DataLoader(train_ds, batch_size=4, shuffle=False)
        val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)

        model_cfg = {
            "model_type": "segformer",
            "backbone": "nvidia/mit-b2",
            "num_labels": 2,
            "image_size": tile_size,
        }
        model = create_model(model_cfg)

        ckpt_dir = tmp_path / "exp_smoke" / "checkpoints"
        logs_dir = tmp_path / "exp_smoke" / "logs"

        training_cfg = {
            "training": {
                "strategy": "tiled",
                "epochs": 1,
                "batch_size": 4,
                "loss": {
                    "name": "focal_dice",
                    "focal_weight": 0.5,
                    "dice_weight": 0.5,
                    "focal_gamma": 2.0,
                    "focal_alpha": 0.25,
                },
                "tiling": {
                    "tile_size": tile_size,
                    "stride": stride,
                },
                "optimizer": {"learning_rate": 6e-5},
                "scheduler": {"name": "cosine_with_warmup", "warmup_steps": 5},
                "mixed_precision": False,
                "gradient_clip_max_norm": 1.0,
                "early_stopping": {"enabled": False},
            }
        }

        trainer = SegmentationTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=training_cfg,
            checkpoint_dir=ckpt_dir,
            log_dir=logs_dir,
            output_dir=tmp_path / "exp_smoke",
        )

        train_result = trainer.train()

        assert "history" in train_result
        assert len(train_result["history"]) == 1
        history_epoch = train_result["history"][0]
        assert not np.isnan(history_epoch["train_loss"])
        assert not np.isinf(history_epoch["train_loss"])
        assert not np.isnan(history_epoch["val_loss"])

        # Check saved checkpoint exists
        latest_ckpt = ckpt_dir / "latest.pth"
        assert latest_ckpt.exists()

        ckpt_data = torch.load(latest_ckpt, map_location="cpu", weights_only=False)
        assert ckpt_data["epoch"] == 1
        assert ckpt_data.get("training_strategy") == "tiled"
        assert ckpt_data.get("tile_size") == tile_size
        assert ckpt_data.get("loss_type") == "focal_dice"

        # Verify strict reloading from saved checkpoint
        reloaded_model = load_model_from_checkpoint(latest_ckpt, device=torch.device("cpu"))
        assert reloaded_model is not None

        # Verify Sliding-Window Tiled Inference on 600x600 image (reconstructed seamlessly)
        inferencer = SegmentationInferencer(
            checkpoint_path=latest_ckpt,
            image_size=tile_size,
            tile_stride=256,
            blend_mode="gaussian",
        )

        res = inferencer.run(synthetic_dataset["sample_img"], tiled=True, tta=False)

        assert res.binary_mask.shape == (600, 600)
        assert res.overlay_image.shape == (600, 600, 3)
        assert 0.0 <= res.confidence <= 1.0
        assert res.metadata["strategy"] == "tiled_native_resolution"
        assert res.metadata["tiled"] is True
