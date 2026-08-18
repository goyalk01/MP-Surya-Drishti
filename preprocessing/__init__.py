"""
Preprocessing module for MP Surya-Drishti Rooftop Segmentation.

Provides dataset loading, augmentation, normalization, and splitting
utilities for aerial image segmentation tasks.
"""

from preprocessing.dataset_loader import MassachusettsDataset
from preprocessing.tiled_dataset import TiledMassachusettsDataset, generate_tile_coordinates
from preprocessing.augmentation import AugmentationPipeline
from preprocessing.normalizer import ImageNormalizer
from preprocessing.splitter import DatasetSplitter

__all__ = [
    "MassachusettsDataset",
    "TiledMassachusettsDataset",
    "generate_tile_coordinates",
    "AugmentationPipeline",
    "ImageNormalizer",
    "DatasetSplitter",
]
