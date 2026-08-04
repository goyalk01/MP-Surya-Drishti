"""
Experiment versioning and management system for MP Surya-Drishti.

Automatically manages experiment directory structures:
    outputs/experiments/exp_001/
    outputs/experiments/exp_002/
    ...

Each experiment directory contains:
    ├── config.yaml          ← Snapshot of merged model, dataset, & training configs
    ├── metrics.json         ← Full training/validation metrics history
    ├── checkpoints/         ← best_iou.pth, best_loss.pth, latest.pth
    ├── logs/                ← TensorBoard events + training.log
    └── plots/               ← Loss/IoU curves & validation prediction overlays
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import yaml

from utils.io_utils import save_json

logger = logging.getLogger(__name__)


class ExperimentManager:
    """
    Manage experiment directory creation, versioning, and artifact tracking.

    Args:
        base_dir: Root directory for experiments (e.g. "outputs/experiments").
        exp_name: Optional custom experiment name (e.g. "exp_001" or "segformer_baseline").
            If None, auto-increments to the next "exp_XXX" index.
    """

    def __init__(
        self,
        base_dir: str | Path = "outputs/experiments",
        exp_name: Optional[str] = None,
    ) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        if exp_name is None:
            self.exp_name = self._get_next_experiment_name()
        else:
            self.exp_name = exp_name

        self.exp_dir = self.base_dir / self.exp_name
        self.checkpoints_dir = self.exp_dir / "checkpoints"
        self.logs_dir = self.exp_dir / "logs"
        self.plots_dir = self.exp_dir / "plots"

        # Create subdirectories
        self._create_directories()

        logger.info("Initialized Experiment: %s at %s", self.exp_name, self.exp_dir)

    def _get_next_experiment_name(self) -> str:
        """Find existing exp_XXX folders and return the next formatted index."""
        existing = list(self.base_dir.glob("exp_*"))
        indices = []
        for path in existing:
            if path.is_dir():
                parts = path.name.split("_")
                if len(parts) >= 2 and parts[1].isdigit():
                    indices.append(int(parts[1]))

        next_idx = max(indices) + 1 if indices else 1
        return f"exp_{next_idx:03d}"

    def _create_directories(self) -> None:
        """Create all subdirectories for this experiment."""
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

    def save_config_snapshot(
        self,
        model_config: dict[str, Any],
        dataset_config: dict[str, Any],
        training_config: dict[str, Any],
    ) -> Path:
        """
        Save a merged snapshot of all configurations used for this experiment.

        Args:
            model_config: Model parameters dict.
            dataset_config: Dataset parameters dict.
            training_config: Training parameters dict.

        Returns:
            Path to saved config.yaml.
        """
        merged_config = {
            "experiment_name": self.exp_name,
            "model": model_config.get("model", model_config),
            "dataset": dataset_config.get("dataset", dataset_config),
            "training": training_config.get("training", training_config),
        }

        config_path = self.exp_dir / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(merged_config, f, default_flow_style=False, sort_keys=False)

        logger.info("Saved experiment config snapshot: %s", config_path)
        return config_path

    def save_metrics_history(self, history: list[dict[str, Any]]) -> Path:
        """
        Save metrics history to metrics.json inside the experiment directory.

        Args:
            history: List of metric dicts per epoch.

        Returns:
            Path to saved metrics.json.
        """
        metrics_path = self.exp_dir / "metrics.json"
        return save_json(history, metrics_path, indent=2)
