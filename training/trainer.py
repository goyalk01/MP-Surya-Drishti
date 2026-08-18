"""
Generic training loop for the segmentation framework.

Supports:
- TensorBoard logging (Loss, IoU, Dice, LR, GPU Memory, Epoch Time, Val Time)
- Smart checkpointing (best_iou.pth, best_loss.pth, latest.pth)
- Experiment versioning (exp_001, exp_002, ...)
- Mixed precision training (AMP)
- Gradient clipping & early stopping
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from evaluation.metrics import SegmentationMetrics
from training.losses import get_loss_function
from training.scheduler import get_scheduler
from utils.device_utils import get_device

logger = logging.getLogger(__name__)


class SegmentationTrainer:
    """
    Model-agnostic training pipeline for semantic segmentation.

    Args:
        model: Any model extending BaseSegmentationModel.
        train_loader: DataLoader for training set.
        val_loader: DataLoader for validation set.
        config: Training configuration dictionary.
        checkpoint_dir: Directory where checkpoints are saved.
        log_dir: Directory for TensorBoard log files.
        output_dir: Experiment output directory.
        device: Training device (auto-detected if None).
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict[str, Any],
        checkpoint_dir: str | Path = "outputs/checkpoints",
        log_dir: str | Path = "outputs/logs",
        output_dir: str | Path = "outputs",
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        self.output_dir = Path(output_dir)
        self.device = device or get_device()

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Move model to device
        self.model = self.model.to(self.device)

        model_name = getattr(model, "model_type", model.__class__.__name__)
        num_labels = getattr(model, "num_labels", 2)

        # Training hyperparams
        train_cfg = config.get("training", config)
        self.epochs = train_cfg.get("epochs", 50)
        self.log_interval = train_cfg.get("log_interval", 10)
        self.val_interval = train_cfg.get("val_interval", 1)

        # Strategy & Tiling metadata
        self.training_strategy = train_cfg.get("strategy", "tiled")
        tiling_cfg = train_cfg.get("tiling", {})
        self.tile_size = tiling_cfg.get("tile_size", getattr(model, "image_size", 512))
        self.tile_stride = tiling_cfg.get("stride", 256)
        loss_cfg = train_cfg.get("loss", {})
        self.loss_type = loss_cfg.get("name", loss_cfg.get("type", "focal_dice"))
        self.tta_enabled = False

        # Optimizer
        opt_cfg = train_cfg.get("optimizer", {})
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=opt_cfg.get("learning_rate", 6e-5),
            weight_decay=opt_cfg.get("weight_decay", 0.01),
            betas=tuple(opt_cfg.get("betas", [0.9, 0.999])),
            eps=opt_cfg.get("eps", 1e-8),
        )

        # Loss function
        self.criterion = get_loss_function(train_cfg)

        # LR Scheduler
        sched_cfg = train_cfg.get("scheduler", {})
        num_training_steps = len(train_loader) * self.epochs
        self.scheduler = get_scheduler(
            self.optimizer, sched_cfg, num_training_steps
        )
        self.scheduler_is_plateau = sched_cfg.get("name") == "plateau"

        # Mixed precision & gradient clipping
        self.use_amp = train_cfg.get("mixed_precision", True) and self.device.type == "cuda"
        self.scaler = GradScaler(enabled=self.use_amp)
        self.gradient_clip = train_cfg.get("gradient_clip_max_norm", 1.0)

        # Early stopping & Metrics
        es_cfg = train_cfg.get("early_stopping", {})
        self.early_stopping_enabled = es_cfg.get("enabled", True)
        self.early_stopping_patience = es_cfg.get("patience", 10)
        self.early_stopping_min_delta = es_cfg.get("min_delta", 0.001)

        self.metrics = SegmentationMetrics(num_classes=num_labels)

        # State tracking
        self.start_epoch = 0
        self.best_iou = 0.0
        self.best_loss = float("inf")
        self.patience_counter = 0
        self.history: list[dict[str, Any]] = []

        # TensorBoard SummaryWriter
        self.writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=str(self.log_dir))
            logger.info("TensorBoard SummaryWriter initialized at %s", self.log_dir)
        except Exception as e:
            logger.warning("Could not initialize TensorBoard SummaryWriter: %s", e)

        logger.info(
            "SegmentationTrainer ready: model=%s, epochs=%d, device=%s",
            model_name,
            self.epochs,
            self.device,
        )

    def train(self, resume_from: Optional[str | Path] = None) -> dict[str, Any]:
        """Run the training loop with TensorBoard logging and smart checkpointing."""
        if resume_from is not None:
            self._resume_checkpoint(resume_from)

        logger.info("Starting training (epochs %d to %d)", self.start_epoch + 1, self.epochs)

        for epoch in range(self.start_epoch, self.epochs):
            epoch_start = time.time()

            # 1. Train epoch
            train_metrics = self._train_epoch(epoch)
            train_time = time.time() - epoch_start

            # 2. Validate epoch
            val_start = time.time()
            val_metrics = {}
            if (epoch + 1) % self.val_interval == 0:
                val_metrics = self._validate_epoch(epoch)
            val_time = time.time() - val_start

            epoch_total_time = time.time() - epoch_start
            current_lr = self.optimizer.param_groups[0]["lr"]

            # GPU Memory usage (MB)
            gpu_memory_mb = 0.0
            if torch.cuda.is_available():
                gpu_memory_mb = torch.cuda.max_memory_allocated(self.device) / (1024 * 1024)

            # Combined metrics dict
            epoch_data = {
                "epoch": epoch + 1,
                "train_loss": train_metrics.get("loss", 0.0),
                "train_ce_loss": train_metrics.get("ce_loss", 0.0),
                "train_dice_loss": train_metrics.get("dice_loss", 0.0),
                "val_loss": val_metrics.get("loss", 0.0),
                "val_iou": val_metrics.get("iou", 0.0),
                "val_dice": val_metrics.get("dice", 0.0),
                "val_pixel_accuracy": val_metrics.get("pixel_accuracy", 0.0),
                "lr": current_lr,
                "gpu_memory_mb": round(gpu_memory_mb, 1),
                "epoch_time_s": round(epoch_total_time, 2),
                "val_time_s": round(val_time, 2),
            }
            self.history.append(epoch_data)

            # ---- TensorBoard Logging ----
            if self.writer is not None:
                step = epoch + 1
                self.writer.add_scalar("Loss/Train", epoch_data["train_loss"], step)
                self.writer.add_scalar("Loss/Val", epoch_data["val_loss"], step)
                self.writer.add_scalar("Metrics/Val_IoU", epoch_data["val_iou"], step)
                self.writer.add_scalar("Metrics/Val_Dice", epoch_data["val_dice"], step)
                self.writer.add_scalar("Metrics/Val_Accuracy", epoch_data["val_pixel_accuracy"], step)
                self.writer.add_scalar("System/LearningRate", current_lr, step)
                self.writer.add_scalar("System/GPUMemory_MB", gpu_memory_mb, step)
                self.writer.add_scalar("System/EpochTime_s", epoch_total_time, step)
                self.writer.add_scalar("System/ValTime_s", val_time, step)

            # Log progress
            logger.info(
                "Epoch %d/%d — train_loss=%.4f | val_loss=%.4f | val_iou=%.4f | "
                "val_dice=%.4f | lr=%.2e | gpu_mem=%.1fMB | time=%.1fs",
                epoch + 1,
                self.epochs,
                epoch_data["train_loss"],
                epoch_data["val_loss"],
                epoch_data["val_iou"],
                epoch_data["val_dice"],
                current_lr,
                gpu_memory_mb,
                epoch_total_time,
            )

            # Scheduler step
            if self.scheduler_is_plateau and val_metrics:
                self.scheduler.step(val_metrics.get("iou", 0.0))

            # ---- Smart Checkpointing ----
            if val_metrics:
                self._handle_smart_checkpointing(epoch, val_metrics)

            # Always save latest.pth
            self._save_checkpoint(epoch, epoch_data, filename="latest.pth")

            # Early stopping check
            val_iou = val_metrics.get("iou", 0.0)
            if self.early_stopping_enabled and val_metrics:
                if self._check_early_stopping(val_iou):
                    logger.info("Early stopping triggered at epoch %d", epoch + 1)
                    break

        if self.writer is not None:
            self.writer.close()

        return {
            "history": self.history,
            "best_iou": self.best_iou,
            "best_loss": self.best_loss,
            "total_epochs": len(self.history),
        }

    def _train_epoch(self, epoch: int) -> dict[str, float]:
        """Run single training epoch."""
        self.model.train()
        total_loss, total_ce, total_dice, count = 0.0, 0.0, 0.0, 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.epochs} [Train]", leave=False)

        for batch in pbar:
            pixel_values = batch["pixel_values"].to(self.device)
            labels = batch["labels"].to(self.device)

            self.optimizer.zero_grad()

            with autocast(enabled=self.use_amp):
                outputs = self.model(pixel_values)
                loss_dict = self.criterion(outputs["upsampled_logits"], labels)
                loss = loss_dict["loss"]

            self.scaler.scale(loss).backward()

            if self.gradient_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            if not self.scheduler_is_plateau:
                self.scheduler.step()

            total_loss += loss.item()
            total_ce += loss_dict["ce_loss"].item()
            total_dice += loss_dict["dice_loss"].item()
            count += 1

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        return {
            "loss": total_loss / max(count, 1),
            "ce_loss": total_ce / max(count, 1),
            "dice_loss": total_dice / max(count, 1),
        }

    @torch.no_grad()
    def _validate_epoch(self, epoch: int) -> dict[str, float]:
        """Run validation loop."""
        self.model.eval()
        self.metrics.reset()
        total_loss, count = 0.0, 0

        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch + 1}/{self.epochs} [Val]", leave=False)

        for batch in pbar:
            pixel_values = batch["pixel_values"].to(self.device)
            labels = batch["labels"].to(self.device)

            with autocast(enabled=self.use_amp):
                outputs = self.model(pixel_values)
                loss_dict = self.criterion(outputs["upsampled_logits"], labels)

            total_loss += loss_dict["loss"].item()
            count += 1

            preds = torch.argmax(outputs["upsampled_logits"], dim=1)
            self.metrics.update(preds, labels)

        results = self.metrics.compute()
        results["loss"] = total_loss / max(count, 1)
        return results

    def _handle_smart_checkpointing(self, epoch: int, val_metrics: dict[str, float]) -> None:
        """Save best_iou.pth and best_loss.pth when metrics improve."""
        val_iou = val_metrics.get("iou", 0.0)
        val_loss = val_metrics.get("loss", float("inf"))

        # Best IoU Checkpoint
        if val_iou > self.best_iou + self.early_stopping_min_delta:
            self.best_iou = val_iou
            self.patience_counter = 0
            self._save_checkpoint(epoch, val_metrics, filename="best_iou.pth")
            logger.info("✓ Saved best_iou.pth (IoU: %.4f)", self.best_iou)

        # Best Loss Checkpoint
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self._save_checkpoint(epoch, val_metrics, filename="best_loss.pth")
            logger.info("✓ Saved best_loss.pth (Loss: %.4f)", self.best_loss)

    def _save_checkpoint(
        self,
        epoch: int,
        metrics: dict[str, Any],
        filename: str = "latest.pth",
    ) -> None:
        """Save checkpoint using the model's persistence method."""
        path = self.checkpoint_dir / filename

        if hasattr(self.model, "save_checkpoint"):
            self.model.save_checkpoint(
                path=path,
                epoch=epoch + 1,
                optimizer_state=self.optimizer.state_dict(),
                scheduler_state=self.scheduler.state_dict() if hasattr(self.scheduler, "state_dict") else None,
                metrics=metrics,
                extra={
                    "best_iou": self.best_iou,
                    "best_loss": self.best_loss,
                    "training_strategy": self.training_strategy,
                    "tile_size": self.tile_size,
                    "tile_stride": self.tile_stride,
                    "loss_type": self.loss_type,
                    "tta_enabled": self.tta_enabled,
                },
            )
        else:
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "metrics": metrics,
                    "training_strategy": self.training_strategy,
                    "tile_size": self.tile_size,
                    "tile_stride": self.tile_stride,
                    "loss_type": self.loss_type,
                    "tta_enabled": self.tta_enabled,
                },
                path,
            )

    def _resume_checkpoint(self, checkpoint_path: str | Path) -> None:
        """Resume training state from checkpoint."""
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        logger.info("Resuming from %s", path)
        if hasattr(self.model, "load_checkpoint"):
            checkpoint = self.model.load_checkpoint(path, device=self.device)
        else:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint["model_state_dict"])

        self.start_epoch = checkpoint.get("epoch", 0)
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        extra = checkpoint.get("extra", {})
        self.best_iou = extra.get("best_iou", 0.0)
        self.best_loss = extra.get("best_loss", float("inf"))

    def _check_early_stopping(self, current_iou: float) -> bool:
        if current_iou <= self.best_iou + self.early_stopping_min_delta:
            self.patience_counter += 1
            if self.patience_counter >= self.early_stopping_patience:
                return True
        return False
