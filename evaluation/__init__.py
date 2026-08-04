"""
Evaluation module for MP Surya-Drishti Rooftop Segmentation.

Provides segmentation metrics and visualization utilities.
"""

from evaluation.metrics import SegmentationMetrics
from evaluation.visualizer import SegmentationVisualizer

__all__ = ["SegmentationMetrics", "SegmentationVisualizer"]
