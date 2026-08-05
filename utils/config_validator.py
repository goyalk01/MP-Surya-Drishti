"""
Configuration validator for MP Surya-Drishti.

Validates YAML configuration files for model, dataset, and training parameters.
Raises descriptive errors with clear troubleshooting instructions.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def validate_model_config(config: dict[str, Any]) -> dict[str, Any]:
    """
    Validate the model configuration dictionary.

    Args:
        config: Model config dictionary (from model_config.yaml).

    Returns:
        Validated model dictionary.

    Raises:
        ValueError: If mandatory parameters are missing or invalid.
    """
    model_cfg = config.get("model", config)

    if not isinstance(model_cfg, dict):
        raise ValueError("Model configuration must contain a 'model' key or dictionary.")

    model_type = model_cfg.get("model_type")
    if not model_type or not isinstance(model_type, str):
        raise ValueError(
            "Invalid model_config: 'model_type' string is required (e.g. 'segformer')."
        )

    backbone = model_cfg.get("backbone")
    if not backbone or not isinstance(backbone, str):
        raise ValueError(
            "Invalid model_config: 'backbone' string is required (e.g. 'nvidia/mit-b2')."
        )

    num_labels = model_cfg.get("num_labels", 2)
    if not isinstance(num_labels, int) or num_labels < 2:
        raise ValueError(
            f"Invalid model_config: 'num_labels' must be an integer >= 2, got {num_labels}."
        )

    image_size = model_cfg.get("image_size", 512)
    if not isinstance(image_size, int) or image_size <= 0:
        raise ValueError(
            f"Invalid model_config: 'image_size' must be a positive integer, got {image_size}."
        )

    confidence_threshold = model_cfg.get("confidence_threshold", 0.5)
    if not isinstance(confidence_threshold, (int, float)) or not (0.0 <= confidence_threshold <= 1.0):
        raise ValueError(
            f"Invalid model_config: 'confidence_threshold' must be between 0.0 and 1.0, got {confidence_threshold}."
        )

    logger.debug("Model configuration validation passed.")
    return model_cfg


def validate_dataset_config(config: dict[str, Any]) -> dict[str, Any]:
    """
    Validate the dataset configuration dictionary.

    Args:
        config: Dataset config dictionary (from dataset_config.yaml).

    Returns:
        Validated dataset dictionary.

    Raises:
        ValueError: If mandatory parameters are missing or invalid.
    """
    dataset_cfg = config.get("dataset", config)

    if not isinstance(dataset_cfg, dict):
        raise ValueError("Dataset configuration must contain a 'dataset' key or dictionary.")

    root_dir = dataset_cfg.get("root_dir")
    if not root_dir or not isinstance(root_dir, str):
        raise ValueError("Invalid dataset_config: 'root_dir' string is required.")

    logger.debug("Dataset configuration validation passed.")
    return dataset_cfg


def validate_training_config(config: dict[str, Any]) -> dict[str, Any]:
    """
    Validate the training configuration dictionary.

    Args:
        config: Training config dictionary (from training_config.yaml).

    Returns:
        Validated training dictionary.

    Raises:
        ValueError: If mandatory parameters are missing or invalid.
    """
    training_cfg = config.get("training", config)

    if not isinstance(training_cfg, dict):
        raise ValueError("Training configuration must contain a 'training' key or dictionary.")

    epochs = training_cfg.get("epochs", 50)
    if not isinstance(epochs, int) or epochs <= 0:
        raise ValueError(f"Invalid training_config: 'epochs' must be a positive integer, got {epochs}.")

    batch_size = training_cfg.get("batch_size", 8)
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError(f"Invalid training_config: 'batch_size' must be a positive integer, got {batch_size}.")

    opt_cfg = training_cfg.get("optimizer", {})
    lr = opt_cfg.get("learning_rate", 6e-5)
    if not isinstance(lr, (int, float)) or lr <= 0:
        raise ValueError(f"Invalid training_config: 'learning_rate' must be > 0, got {lr}.")

    logger.debug("Training configuration validation passed.")
    return training_cfg
