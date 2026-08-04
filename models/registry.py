"""
Model registry and factory for the segmentation framework.

Provides a decorator-based registration system so new models can be
added with zero changes to the training, inference, or evaluation code.

To register a new model:
    1. Create a class extending ``BaseSegmentationModel``.
    2. Decorate it with ``@register_model("your_model_name")``.
    3. Import the module (so the decorator executes).

That's it. The entire pipeline (train, infer, evaluate, API) will
work with the new model via config: ``model_type: your_model_name``.

Usage:
    from models.registry import create_model, load_model_from_checkpoint

    # Create a new model from config
    model = create_model(model_config)

    # Load any model from a checkpoint (auto-detects model_type)
    model = load_model_from_checkpoint("checkpoints/best.pth", device)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, Type

import torch

logger = logging.getLogger(__name__)

# Global model registry: maps model_type string → model class
_MODEL_REGISTRY: dict[str, Type] = {}


def register_model(model_type: str):
    """
    Decorator to register a segmentation model class in the global registry.

    Args:
        model_type: Unique identifier for this model (e.g., "segformer",
            "deeplabv3plus", "unet", "mask2former").

    Returns:
        Decorator function.

    Example::

        @register_model("segformer")
        class SegFormerModel(BaseSegmentationModel):
            ...
    """

    def decorator(cls):
        if model_type in _MODEL_REGISTRY:
            logger.warning(
                "Model type '%s' is already registered (overwriting with %s)",
                model_type,
                cls.__name__,
            )
        _MODEL_REGISTRY[model_type] = cls
        logger.debug("Registered model: '%s' → %s", model_type, cls.__name__)
        return cls

    return decorator


def create_model(model_config: dict[str, Any]) -> Any:
    """
    Create a model instance from a configuration dictionary.

    Looks up the model class from the registry using ``model_type``
    and passes all relevant config fields to the constructor.

    Args:
        model_config: Dictionary from ``model_config.yaml["model"]``.
            Must contain "model_type" key.

    Returns:
        Initialized model instance.

    Raises:
        ValueError: If model_type is not registered.
    """
    model_type = model_config.get("model_type")
    if model_type is None:
        raise ValueError(
            "model_config must contain 'model_type'. "
            f"Available types: {list_registered_models()}"
        )

    if model_type not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model_type: '{model_type}'. "
            f"Available types: {list_registered_models()}"
        )

    cls = _MODEL_REGISTRY[model_type]

    # Build constructor kwargs from config
    kwargs = {
        "backbone": model_config.get("backbone"),
        "num_labels": model_config.get("num_labels", 2),
        "id2label": {
            int(k): v for k, v in model_config.get("id2label", {}).items()
        }
        or None,
        "label2id": model_config.get("label2id") or None,
        "confidence_threshold": model_config.get("confidence_threshold", 0.5),
        "image_size": model_config.get("image_size", 512),
    }

    # Remove None values so class defaults apply
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    model = cls(**kwargs)

    logger.info(
        "Created model '%s' (%s) with backbone='%s', num_labels=%d",
        model_type,
        cls.__name__,
        model_config.get("backbone", "N/A"),
        model_config.get("num_labels", 2),
    )

    return model


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    device: Optional[torch.device] = None,
) -> Any:
    """
    Load any registered model from a checkpoint file.

    The checkpoint must contain a "model_type" field (saved automatically
    by ``BaseSegmentationModel.save_checkpoint()``). This function looks
    up the correct class and calls its ``from_checkpoint()`` method.

    Args:
        checkpoint_path: Path to the checkpoint .pth file.
        device: Device to place the model on.

    Returns:
        Loaded model instance, ready for inference or training resume.

    Raises:
        FileNotFoundError: If checkpoint file doesn't exist.
        ValueError: If model_type is missing or unregistered.
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    # Peek at checkpoint to read model_type
    map_location = device if device else "cpu"
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)

    model_type = checkpoint.get("model_type")
    if model_type is None:
        raise ValueError(
            f"Checkpoint at '{path}' does not contain 'model_type'. "
            "It may have been saved by an older version of the framework."
        )

    if model_type not in _MODEL_REGISTRY:
        raise ValueError(
            f"Checkpoint model_type '{model_type}' is not registered. "
            f"Available types: {list_registered_models()}. "
            f"Make sure the model module is imported."
        )

    cls = _MODEL_REGISTRY[model_type]
    model = cls.from_checkpoint(checkpoint_path, device=device)

    logger.info(
        "Loaded model '%s' from checkpoint '%s'",
        model_type,
        path,
    )

    return model


def list_registered_models() -> list[str]:
    """
    List all registered model types.

    Returns:
        List of model type identifier strings.
    """
    return list(_MODEL_REGISTRY.keys())


def get_model_class(model_type: str) -> Optional[Type]:
    """
    Get the class for a registered model type.

    Args:
        model_type: Model identifier string.

    Returns:
        Model class, or None if not registered.
    """
    return _MODEL_REGISTRY.get(model_type)


def ensure_models_registered() -> None:
    """
    Import all model modules to trigger registration decorators.

    Call this at application startup to ensure all models are available
    in the registry. New model files added to ``models/`` should be
    imported here.
    """
    # Import each model module — the @register_model decorators
    # will execute and populate the registry.
    try:
        import models.segformer_model  # noqa: F401
    except ImportError as e:
        logger.warning("Could not import segformer_model: %s", e)

    # ---- Add future model imports below ----
    # try:
    #     import models.deeplabv3plus_model  # noqa: F401
    # except ImportError as e:
    #     logger.warning("Could not import deeplabv3plus_model: %s", e)
    #
    # try:
    #     import models.mask2former_model  # noqa: F401
    # except ImportError as e:
    #     logger.warning("Could not import mask2former_model: %s", e)
    #
    # try:
    #     import models.unet_model  # noqa: F401
    # except ImportError as e:
    #     logger.warning("Could not import unet_model: %s", e)

    logger.info("Model registry: %s", list_registered_models())
