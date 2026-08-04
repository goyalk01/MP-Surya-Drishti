# Changelog

All notable changes to **MP Surya-Drishti** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-08-05

### Added
- **Base Segmentation Architecture**: Abstract `BaseSegmentationModel` interface and dynamic decorator-based `@register_model` factory.
- **SegFormer Integration**: Pretrained HuggingFace `SegFormer` (`nvidia/mit-b2`) as the baseline model plugin.
- **Official Dataset Partition Support**: Support for official Massachusetts Buildings Dataset layout (`train/train_labels`, `val/val_labels`, `test/test_labels`).
- **TensorBoard Integration**: Real-time logging of Loss, IoU, Dice, Learning Rate, GPU Memory (MB), and Epoch/Val timing.
- **Smart Checkpointing**: Automated saving of `best_iou.pth`, `best_loss.pth`, and `latest.pth`.
- **Experiment Versioning**: Auto-incrementing experiment directories (`outputs/experiments/exp_001/`, `exp_002/`).
- **Prediction Reports**: Standardized `prediction_report.json` schema for Solar Analytics ingestion.
- **Postprocessing Engine**: Morphological mask cleaning, Douglas-Peucker polygon extraction, and scale-aware area estimation.
- **CLI Suite**: Unified `main.py` entry point supporting `train`, `infer`, `evaluate`, and `verify-dataset`.
