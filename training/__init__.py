"""
Training module for MP Surya-Drishti Rooftop Segmentation.

Provides the training loop, loss functions, and learning rate schedulers.
"""

from training.losses import CombinedSegmentationLoss, DiceLoss
from training.scheduler import get_scheduler
from training.trainer import SegmentationTrainer

__all__ = [
    "CombinedSegmentationLoss",
    "DiceLoss",
    "get_scheduler",
    "SegmentationTrainer",
]
