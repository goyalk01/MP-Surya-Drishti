"""
Checkpoint management utilities for MP Surya-Drishti.

Provides checkpoint listing, metadata inspection, automated best checkpoint
selection based on validation metrics, and cleanup functionality.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)


def find_best_available_checkpoint(
    search_dirs: Optional[list[str | Path]] = None,
    preferred_metric: str = "rooftop_iou",
) -> Path:
    """
    Find and select the best available trained checkpoint across experiment directories.

    Evaluates stored VALIDATION metrics inside checkpoint metadata (e.g. validation
    rooftop_iou or val_iou). Never uses test data for checkpoint selection.

    Selection hierarchy:
      1. Candidate with the highest stored validation rooftop_iou (or val_iou).
      2. If metrics are equal/unavailable, priority: best_loss.pth > best_iou.pth > latest.pth.
      3. Most recently modified candidate.

    Args:
        search_dirs: Optional list of directories to search. If None, searches standard
                     experiment and model checkpoint locations.
        preferred_metric: Metric key inside checkpoint['metrics'] to maximize (default: 'rooftop_iou').

    Returns:
        Path to the best available checkpoint.

    Raises:
        FileNotFoundError: If no checkpoint (.pth) is found.
    """
    if search_dirs is None:
        search_dirs = [
            Path("outputs/experiments"),
            Path("models/checkpoints"),
            Path("outputs/checkpoints"),
        ]

    candidate_paths: list[Path] = []

    for s_dir in search_dirs:
        p = Path(s_dir)
        if not p.exists():
            continue
        if p.is_file() and p.suffix == ".pth":
            candidate_paths.append(p)
        elif p.is_dir():
            candidate_paths.extend(p.glob("**/*.pth"))

    # Filter out non-checkpoints or empty files if any
    valid_candidates = [p for p in candidate_paths if p.exists() and p.stat().st_size > 1024]

    if not valid_candidates:
        raise FileNotFoundError(
            "No valid model checkpoint (*.pth) found. Please run training first."
        )

    # Inspect metadata of each candidate
    evaluated_candidates: list[dict[str, Any]] = []

    for ckpt_path in valid_candidates:
        info: dict[str, Any] = {
            "path": ckpt_path,
            "filename": ckpt_path.name,
            "val_score": -1.0,
            "epoch": -1,
            "mtime": ckpt_path.stat().st_mtime,
        }

        try:
            ckpt_meta = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            metrics = ckpt_meta.get("metrics", {})
            epoch = ckpt_meta.get("epoch", -1)
            info["epoch"] = epoch

            # Extract validation metrics consistently (never test metrics)
            val_iou = metrics.get("val_iou", metrics.get("iou", -1.0))
            roof_iou = metrics.get("rooftop_iou", metrics.get("val_rooftop_iou"))
            if roof_iou is None:
                iou_per_class = metrics.get("iou_per_class")
                if isinstance(iou_per_class, list) and len(iou_per_class) > 1:
                    roof_iou = iou_per_class[1]
            if roof_iou is None and preferred_metric != "rooftop_iou":
                roof_iou = metrics.get(preferred_metric, -1.0)

            info["val_iou"] = float(val_iou) if isinstance(val_iou, (int, float)) else -1.0
            info["roof_iou"] = float(roof_iou) if isinstance(roof_iou, (int, float)) else -1.0

        except Exception as err:
            logger.debug("Could not read metadata from %s: %s", ckpt_path, err)

        evaluated_candidates.append(info)

    # Sort candidates by:
    # 1. Validation rooftop IoU (or preferred metric)
    # 2. Validation Mean IoU
    # 3. Filename priority (best_loss > best_iou > latest > others)
    # 4. Modification time (newest first)
    def filename_priority(name: str) -> int:
        if name == "best_loss.pth":
            return 3
        if name == "best_iou.pth":
            return 2
        if name == "latest.pth":
            return 1
        return 0

    evaluated_candidates.sort(
        key=lambda x: (
            x["roof_iou"],
            x["val_iou"],
            filename_priority(x["filename"]),
            x["mtime"],
        ),
        reverse=True,
    )

    best_candidate = evaluated_candidates[0]
    best_path = best_candidate["path"]

    logger.info(
        "Auto-selected best checkpoint: %s (epoch=%s, val_%s=%.4f)",
        best_path,
        best_candidate["epoch"],
        preferred_metric,
        best_candidate["val_score"],
    )

    return best_path


class CheckpointManager:
    """
    Manage model checkpoints: list, compare, auto-select, and clean up.

    Args:
        checkpoint_dir: Directory where checkpoints are stored.
        keep_top_k: Maximum number of checkpoints to keep.
            Older checkpoints beyond this count are deleted.
    """

    def __init__(
        self,
        checkpoint_dir: str | Path = "models/checkpoints",
        keep_top_k: int = 3,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_top_k = keep_top_k

    def list_checkpoints(self) -> list[dict[str, Any]]:
        """
        List all checkpoints with metadata.

        Returns:
            List of dicts with checkpoint info, sorted by modification time.
        """
        checkpoints = []

        for ckpt_path in self.checkpoint_dir.glob("*.pth"):
            info = {
                "path": str(ckpt_path),
                "filename": ckpt_path.name,
                "size_mb": ckpt_path.stat().st_size / (1024 * 1024),
                "modified": datetime.fromtimestamp(
                    ckpt_path.stat().st_mtime
                ).isoformat(),
            }

            try:
                checkpoint = torch.load(
                    ckpt_path, map_location="cpu", weights_only=False
                )
                info["epoch"] = checkpoint.get("epoch", -1)
                info["metrics"] = checkpoint.get("metrics", {})
            except Exception:
                info["epoch"] = -1
                info["metrics"] = {}

            checkpoints.append(info)

        checkpoints.sort(key=lambda x: x["modified"], reverse=True)
        return checkpoints

    def get_best_checkpoint(self) -> Optional[Path]:
        """
        Get the path to the best checkpoint based on validation metrics.

        Returns:
            Path to best checkpoint file, or None if none found.
        """
        try:
            return find_best_available_checkpoint(search_dirs=[self.checkpoint_dir])
        except FileNotFoundError:
            return None

    def cleanup(self) -> int:
        """
        Remove old checkpoints, keeping only the top K plus best checkpoints.

        Returns:
            Number of checkpoints deleted.
        """
        checkpoints = list(self.checkpoint_dir.glob("*.pth"))

        protected = {"best_checkpoint.pth", "best_loss.pth", "best_iou.pth", "latest.pth"}
        removable = [p for p in checkpoints if p.name not in protected]

        removable.sort(key=lambda p: p.stat().st_mtime)
        to_remove = removable[: max(0, len(removable) - self.keep_top_k)]

        for path in to_remove:
            path.unlink()
            logger.info("Deleted old checkpoint: %s", path.name)

        return len(to_remove)
