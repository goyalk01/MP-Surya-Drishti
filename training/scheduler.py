"""
Learning rate scheduler factory.

Provides configurable scheduler creation from training config,
supporting cosine annealing, step decay, plateau, and warmup variants.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch.optim as optim
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LambdaLR,
    ReduceLROnPlateau,
    StepLR,
)

logger = logging.getLogger(__name__)


def get_scheduler(
    optimizer: optim.Optimizer,
    scheduler_config: dict[str, Any],
    num_training_steps: int = 0,
) -> Any:
    """
    Create a learning rate scheduler from configuration.

    Supported schedulers:
        - ``cosine``: Cosine annealing to zero.
        - ``cosine_with_warmup``: Linear warmup then cosine decay.
        - ``step``: Step decay by gamma every step_size epochs.
        - ``plateau``: Reduce on plateau (monitors validation metric).

    Args:
        optimizer: The optimizer to schedule.
        scheduler_config: Dictionary from ``training_config.yaml`` scheduler section.
        num_training_steps: Total number of training steps (for warmup calculation).

    Returns:
        PyTorch LR scheduler instance.

    Raises:
        ValueError: If the scheduler name is not recognized.
    """
    name = scheduler_config.get("name", "cosine_with_warmup")
    warmup_steps = scheduler_config.get("warmup_steps", 500)

    if name == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=num_training_steps,
            eta_min=1e-7,
        )
        logger.info(
            "Created CosineAnnealingLR scheduler (T_max=%d)", num_training_steps
        )

    elif name == "cosine_with_warmup":
        scheduler = _get_cosine_with_warmup(
            optimizer,
            warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
        )
        logger.info(
            "Created CosineWithWarmup scheduler (warmup=%d, total=%d)",
            warmup_steps,
            num_training_steps,
        )

    elif name == "step":
        step_size = scheduler_config.get("step_size", 10)
        gamma = scheduler_config.get("gamma", 0.5)
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        logger.info(
            "Created StepLR scheduler (step_size=%d, gamma=%.2f)",
            step_size,
            gamma,
        )

    elif name == "plateau":
        patience = scheduler_config.get("patience", 5)
        factor = scheduler_config.get("factor", 0.5)
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="max",  # Maximize IoU
            patience=patience,
            factor=factor,
            min_lr=1e-7,
            verbose=True,
        )
        logger.info(
            "Created ReduceLROnPlateau scheduler (patience=%d, factor=%.2f)",
            patience,
            factor,
        )

    else:
        raise ValueError(
            f"Unknown scheduler: '{name}'. "
            f"Supported: cosine, cosine_with_warmup, step, plateau"
        )

    return scheduler


def _get_cosine_with_warmup(
    optimizer: optim.Optimizer,
    warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
) -> LambdaLR:
    """
    Create a cosine annealing scheduler with linear warmup.

    During warmup, the learning rate increases linearly from 0 to the
    initial LR. After warmup, it follows a cosine curve down to near zero.

    Args:
        optimizer: The optimizer to schedule.
        warmup_steps: Number of warmup steps.
        num_training_steps: Total number of training steps.
        num_cycles: Number of cosine cycles (default 0.5 = half cosine).

    Returns:
        LambdaLR scheduler with warmup + cosine decay.
    """

    def lr_lambda(current_step: int) -> float:
        # Warmup phase: linear increase
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))

        # Cosine decay phase
        progress = float(current_step - warmup_steps) / float(
            max(1, num_training_steps - warmup_steps)
        )
        return max(
            0.0,
            0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress)),
        )

    return LambdaLR(optimizer, lr_lambda)
