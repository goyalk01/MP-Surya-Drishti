"""
Models module for MP Surya-Drishti Segmentation Framework.

Provides the abstract base class, model registry, and all registered
model implementations. New models are auto-registered via decorator.
"""

from models.base_model import BaseSegmentationModel
from models.registry import (
    create_model,
    ensure_models_registered,
    list_registered_models,
    load_model_from_checkpoint,
    register_model,
)

# Import model implementations to trigger registration
from models.segformer_model import SegFormerModel

# Backward compatibility alias
from models.segformer_model import RooftopSegFormer

__all__ = [
    "BaseSegmentationModel",
    "register_model",
    "create_model",
    "load_model_from_checkpoint",
    "list_registered_models",
    "ensure_models_registered",
    "SegFormerModel",
    "RooftopSegFormer",
]
