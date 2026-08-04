"""
Inference module for MP Surya-Drishti Segmentation Framework.

Provides standalone, model-agnostic inference independent of training.
"""

from inference.inferencer import SegmentationInferencer, SegmentationResult

# Backward-compatible alias
from inference.inferencer import RooftopInferencer

__all__ = [
    "SegmentationInferencer",
    "SegmentationResult",
    "RooftopInferencer",
]
