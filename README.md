<div align="center">

# ☀️ MP Surya-Drishti

### AI-Assisted Community Solar Advisory System for Rooftop Feasibility & Panel Maintenance

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace Transformers](https://img.shields.io/badge/%F9%9F%A4%97-Transformers-yellow.svg)](https://huggingface.co/docs/transformers/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.9%2B-5C3EE8.svg?logo=opencv&logoColor=white)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Project Status: Production-Ready](https://img.shields.io/badge/Status-Production--Ready-success.svg)]()
[![EPICS](https://img.shields.io/badge/EPICS-VIT%20Bhopal-orange.svg)](https://vitbhopal.ac.in/)

</div>

---

## 📌 Project Overview

### Problem Statement
Rooftop solar adoption across communities and residential households is often hindered by high upfront survey costs, lack of preliminary rooftop feasibility insights, complex obstacle/shadow assessments, and uncertainty regarding energy generation ROI. Manual site surveys by certified installers are time-consuming and expensive for early-stage decision making.

### Significance & Objective
Accelerating solar adoption requires democratic access to fast, automated, and intelligent rooftop feasibility advisory tools. By providing computer-vision-driven rooftop segmentation, shadow analytics, usable area estimation, and generation forecasts, **MP Surya-Drishti** empowers households, local institutions, and DISCOM planners with instant data-driven insights.

### Beneficiaries
* **Households & Community Members**: Instant feasibility advisory, usable roof area estimation, and projected financial ROI before hiring installers.
* **Solar Installers & Surveyors**: Automated preliminary site analysis to streamline physical surveys.
* **Urban & Energy Planners**: Community-scale rooftop solar potential mapping.

> ⚠️ **Disclaimer**: *MP Surya-Drishti is an AI advisory tool designed for preliminary rooftop solar assessment. It is not intended to replace certified solar installers, structural engineering inspections, or DISCOM utility surveys.*

---

## ✨ Features & Architecture

### Current Module — Phase 1: Rooftop Segmentation Framework ✅
* **Pluggable Computer Vision Framework**: Generic `BaseSegmentationModel` interface allowing seamless integration of SegFormer, DeepLabV3+, Mask2Former, or U-Net backbones.
* **Official Dataset Pipeline**: Native support for the official Massachusetts Buildings Dataset partition structure (`train/train_labels`, `val/val_labels`, `test/test_labels`).
* **TensorBoard & Experiment Versioning**: Real-time logging of Loss, IoU, Dice, Learning Rate, GPU Memory usage (MB), and Epoch/Validation timing under versioned folders (`outputs/experiments/exp_001/`, `exp_002/`).
* **Smart Checkpointing**: Automated saving of `best_iou.pth`, `best_loss.pth`, and `latest.pth` with strict weight loading (`strict=True`).
* **Postprocessing & Polygon Extraction**: Morphological mask cleaning, Douglas-Peucker contour simplification (GeoJSON polygons), and scale-aware rooftop area estimation.
* **Standardized Prediction Reports**: Generates `prediction_report.json` designed for direct downstream integration.

---

## 📁 Repository Structure

```
rooftop_segmentation/
├── configs/                    → YAML configurations (model_type, dataset splits, training hyperparams)
├── datasets/                   → Dataset storage directory + README
│   └── massachusetts/          → Official dataset layout (train, train_labels, val, val_labels, test, test_labels)
├── models/                     → Pluggable Model Architecture Core
│   ├── base_model.py           → Abstract BaseSegmentationModel contract
│   ├── registry.py             → Model registry (@register_model) & factory (create_model)
│   ├── segformer_model.py      → SegFormerModel (nvidia/mit-b2 baseline)
│   └── MODEL_TEMPLATE.py       → Developer guide for adding new vision backbones
├── preprocessing/              → Dataset loader, Albumentations transforms, normalizer
├── training/                   → Loss functions (CE+Dice), LR schedulers, generic trainer
├── evaluation/                 → Metrics (IoU, Dice, Accuracy), visualizer
├── inference/                  → SegmentationInferencer & SegmentationResult engine
├── postprocessing/             → Mask cleaner, polygon extractor, area estimator
├── utils/                      → Logger, device utils, checkpoint & experiment managers, config validator
├── tests/                      → PyTest unit test suite (checkpointing, forward pass, metrics, postprocessing)
├── docs/                       → Architecture guides & technical documentation
├── assets/                     → Repository visual assets & diagrams
├── examples/                   → Python usage examples & scripts (infer_sample.py)
├── main.py                     → Unified Framework CLI (train | infer | evaluate | verify-dataset)
├── requirements.txt            → Production Python dependencies
├── CONTRIBUTING.md             → Contribution guidelines
├── CODE_OF_CONDUCT.md          → Contributor code of conduct
├── SECURITY.md                 → Vulnerability reporting policy
├── CHANGELOG.md                → Semantic version history
├── ROADMAP.md                  → Multi-phase project roadmap
├── LICENSE                     → MIT License
└── README.md                   → GitHub project documentation
```

---

## 🛠️ Technology Stack

| Category | Technologies |
|----------|-------------|
| **Core Language** | Python 3.10+ |
| **Deep Learning Framework** | PyTorch 2.1+, TorchVision, TorchMetrics |
| **Model Architectures** | HuggingFace Transformers (SegFormer `mit-b2`), YOLOv8 (Planned) |
| **Computer Vision & Image Processing** | OpenCV, Albumentations, Pillow |
| **Scientific Computing & Geometry** | NumPy, SciPy, Pandas, Shapely |
| **Visualization & Experiment Tracking** | Matplotlib, TensorBoard |
| **Execution Environments** | Google Colab (T4 GPU), Local PyTorch CUDA |
| **Version Control & Governance** | Git, GitHub Actions |

---

## 💾 Checkpoint Format & State Preservation

Every checkpoint (`best_iou.pth`, `best_loss.pth`, `latest.pth`) contains complete, self-contained state metadata:

```python
{
    "model_type": "segformer",
    "epoch": 25,
    "model_state_dict": dict,      # 100% trained encoder + decode head weights
    "optimizer_state_dict": dict,  # AdamW optimizer state
    "scheduler_state_dict": dict,  # LR scheduler state
    "scaler_state_dict": dict,     # AMP GradScaler state (if CUDA)
    "random_state": dict,          # Python, NumPy, PyTorch, CUDA RNG states
    "backbone": "nvidia/mit-b2",
    "num_labels": 2,
    "id2label": {0: "background", 1: "rooftop"},
    "label2id": {"background": 0, "rooftop": 1},
    "confidence_threshold": 0.5,
    "image_size": 512,
    "metrics": dict,               # Val IoU, Dice, Accuracy
    "config": dict,                # Full experiment configuration snapshot
}
```

---

## 🚀 Installation & Workflow

### 1. Clone & Setup Environment
```bash
git clone https://github.com/goyalk01/MP-Surya-Drishti.git
cd MP-Surya-Drishti

python -m venv venv
# On Linux/macOS: source venv/bin/activate
# On Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Verify Dataset Setup
Ensure the Massachusetts Buildings Dataset is placed in `datasets/massachusetts/`:
```bash
python main.py verify-dataset
```

### 3. Training
Start fine-tuning SegFormer on the dataset partitions. Outputs are saved under `outputs/experiments/exp_001/`:
```bash
python main.py train
```
*Monitor real-time training & metrics in TensorBoard:*
```bash
tensorboard --logdir outputs/experiments
```

### 4. Evaluation
Evaluate the trained weights on the official test set:
```bash
python main.py evaluate --checkpoint outputs/experiments/exp_001/checkpoints/best_iou.pth
```

### 5. Single Image Inference
Run inference to produce segmented overlays and `prediction_report.json`:
```bash
python main.py infer --image path/to/sample.jpg --checkpoint outputs/experiments/exp_001/checkpoints/best_iou.pth
```

---

## ❓ Common Warnings & Troubleshooting

### 1. Pretrained Weight Loading Logs During Training
- **Log Message**: `Some weights of SegformerForSemanticSegmentation were not initialized...`
- **Explanation**: This is expected when initializing a fresh model for transfer learning. HuggingFace loads encoder weights from `nvidia/mit-b2` and re-initializes the classification head for 2 classes.
- **Checkpoint Restoration**: When loading saved checkpoints (`best_iou.pth`), the framework instantiates the model architecture directly from configuration and restores weights with `strict=True`. **Zero warnings** occur during inference or evaluation checkpoint loading.

### 2. Missing Trained Checkpoint Error
- **Error**: `FileNotFoundError: No trained checkpoint ('best_iou.pth') found...`
- **Resolution**: Run `python main.py train` to generate experiment checkpoints before evaluating or inferring.

---

## 📜 License & Authors

Distributed under the **MIT License**. Developed by the **MP Surya-Drishti EPICS Team** at **VIT Bhopal University**.
