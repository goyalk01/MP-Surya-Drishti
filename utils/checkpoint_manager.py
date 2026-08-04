"""
Checkpoint management utilities.

Provides checkpoint listing, cleanup, and metadata tracking
beyond the basic save/load in the model class.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manage model checkpoints: list, compare, and clean up.

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

            # Try to read epoch and metrics without loading full weights
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

        # Sort by modification time (newest first)
        checkpoints.sort(key=lambda x: x["modified"], reverse=True)

        return checkpoints

    def get_best_checkpoint(self) -> Optional[Path]:
        """
        Get the path to the best checkpoint.

        Returns:
            Path to best_checkpoint.pth, or None if not found.
        """
        best_path = self.checkpoint_dir / "best_checkpoint.pth"
        if best_path.exists():
            return best_path

        # Fallback: find any checkpoint
        checkpoints = list(self.checkpoint_dir.glob("*.pth"))
        if checkpoints:
            return max(checkpoints, key=lambda p: p.stat().st_mtime)

        return None

    def cleanup(self) -> int:
        """
        Remove old checkpoints, keeping only the top K plus the best.

        Returns:
            Number of checkpoints deleted.
        """
        checkpoints = list(self.checkpoint_dir.glob("*.pth"))

        # Always keep best_checkpoint.pth and last_checkpoint.pth
        protected = {"best_checkpoint.pth", "last_checkpoint.pth"}
        removable = [
            p for p in checkpoints if p.name not in protected
        ]

        # Sort by modification time (oldest first)
        removable.sort(key=lambda p: p.stat().st_mtime)

        # Remove oldest, keeping top K
        to_remove = removable[: max(0, len(removable) - self.keep_top_k)]

        for path in to_remove:
            path.unlink()
            logger.info("Deleted old checkpoint: %s", path.name)

        return len(to_remove)
