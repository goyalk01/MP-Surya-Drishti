<div align="center">

# ☀️ MP Surya-Drishti

### AI-Assisted Community Solar Advisory System for Rooftop Feasibility & Panel Maintenance

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace Transformers](https://img.shields.io/badge/%F9%9F%A4%97-Transformers-yellow.svg)](https://huggingface.co/docs/transformers/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.9%2B-5C3EE8.svg?logo=opencv&logoColor=white)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Project Status: Active](https://img.shields.io/badge/Status-Active%20Development-success.svg)]()
[![EPICS](https://img.shields.io/badge/EPICS-VIT%20Bhopal-orange.svg)](https://vitbhopal.ac.in/)

</div>

---

## 📌 Project Overview

### What Problem Are We Solving?
Rooftop solar adoption across communities and residential households is often hindered by high upfront survey costs, lack of accurate preliminary rooftop feasibility insights, complex obstacle/shadow assessments, and uncertainty regarding energy generation ROI. Manual site surveys by certified installers are time-consuming and expensive for early-stage decision making.

### Why Is This Important?
Accelerating solar adoption requires democratic access to fast, automated, and intelligent rooftop feasibility advisory tools. By providing computer-vision-driven rooftop segmentation, shadow analytics, usable area estimation, and generation forecasts, **MP Surya-Drishti** empowers households, local institutions, and DISCOM planners with instant data-driven insights.

### Who Benefits?
* **Households & Community Members**: Instant feasibility advisory, usable roof area estimation, and projected financial ROI before hiring installers.
* **Solar Installers & Surveyors**: Automated preliminary site analysis to streamline physical surveys.
* **Urban & Energy Planners**: Community-scale rooftop solar potential mapping.

> ⚠️ **Disclaimer**: *MP Surya-Drishti is an AI advisory tool designed for preliminary rooftop solar assessment. It is not intended to replace certified solar installers, structural engineering inspections, or DISCOM utility surveys.*

---

## ✨ Features & Platform Architecture

### Current Module — Phase 1: Rooftop Segmentation Framework ✅
* **Pluggable Computer Vision Framework**: Generic `BaseSegmentationModel` interface allowing seamless integration of SegFormer, DeepLabV3+, Mask2Former, or U-Net backbones.
* **Official Dataset Pipeline**: Native support for the official Massachusetts Buildings Dataset partition structure (`train/train_labels`, `val/val_labels`, `test/test_labels`).
* **TensorBoard & Experiment Versioning**: Real-time logging of Loss, IoU, Dice, Learning Rate, GPU Memory usage (MB), and Epoch/Validation timing under versioned folders (`outputs/experiments/exp_001/`, `exp_002/`).
* **Smart Checkpointing**: Automated saving of `best_iou.pth`, `best_loss.pth`, and `latest.pth`.
* **Postprocessing & Polygon Extraction**: Morphological mask cleaning, Douglas-Peucker contour simplification (GeoJSON polygons), and scale-aware rooftop area estimation.
* **Standardized Prediction Reports**: Generates `prediction_report.json` designed for direct downstream integration.

### Upcoming Platform Modules 🔜
* 🌤️ **Shadow Detection & Sun-Tracking Engine**: Solar azimuth/elevation ray tracing.
* 📦 **Obstacle Detection**: HVAC unit, water tank, and vegetation masking via vision models.
* ⚡ **Solar Generation Prediction**: Weather-aware (GHI, DNI, temp) PV yield modeling.
* 🎯 **Intelligent Panel Placement**: Automated string & tilt layout optimization.
* 🔍 **Panel Health & Defect Monitoring**: Infrared & RGB anomaly classification (hotspots, cracks, dust).
* 🌐 **Interactive Web Dashboard**: Next.js + Leaflet/MapLibre map UI with automated PDF report generation.

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
├── evaluation/                 → Metrics (IoU, Dice, Accuracy), visualization plots
├── inference/                  → SegmentationInferencer & SegmentationResult engine
├── postprocessing/             → Mask cleaner, polygon extractor, area estimator
├── utils/                      → Logger, device utils, checkpoint & experiment managers, I/O
├── tests/                      → PyTest unit test suite
├── docs/                       → Architecture guides & technical documentation
├── assets/                     → Repository visual assets & diagrams
├── examples/                   → Python usage examples & scripts
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
| **Deep Learning Framework** | PyTorch, TorchVision, TorchMetrics |
| **Model Architectures** | HuggingFace Transformers (SegFormer `mit-b2`), YOLOv8 (Planned) |
| **Computer Vision & Image Processing** | OpenCV, Albumentations, Pillow |
| **Scientific Computing & Geometry** | NumPy, SciPy, Pandas, Shapely |
| **Visualization & Experiment Tracking** | Matplotlib, TensorBoard |
| **Execution Environments** | Google Colab (T4 GPU), Local PyTorch CUDA |
| **Version Control & Governance** | Git, GitHub Actions |

---

## 🚀 Installation & Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/goyalk01/MP-Surya-Drishti.git
cd MP-Surya-Drishti
```

### 2. Create Virtual Environment
```bash
# Linux / macOS
python -m venv venv
source venv/bin/activate

# Windows (PowerShell)
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 📊 Dataset Setup

This repository uses the **Massachusetts Buildings Dataset** as its default benchmark dataset.

### Directory Structure
Extract the dataset into `datasets/massachusetts/` adhering to the official split layout:
```
datasets/massachusetts/
├── train/              ← Aerial images (.tiff / .png)
├── train_labels/       ← Binary building masks (.tiff / .png)
├── val/                ← Validation images
├── val_labels/         ← Validation building masks
├── test/               ← Test images
└── test_labels/        ← Test building masks
```

### Download Link & Instructions
1. Download from Kaggle: [Massachusetts Buildings Dataset](https://www.kaggle.com/datasets/balraj98/massachusetts-buildings-dataset)
2. Extract the archive into `datasets/massachusetts/`.
3. Verify file pairing across all splits:
   ```bash
   python main.py verify-dataset
   ```

---

## 🏋️ Training & CLI Suite

The framework provides a model-agnostic CLI via `main.py`:

### 1. Verify Dataset Integrity
Verifies image-mask pairing across official `train`, `val`, and `test` splits:
```bash
python main.py verify-dataset
```

### 2. Fine-Tune Model
Trains the active model specified in `configs/model_config.yaml` (default: `segformer`), creating a new experiment folder `outputs/experiments/exp_001/`:
```bash
python main.py train
```
*Launch TensorBoard to monitor training in real time:*
```bash
tensorboard --logdir outputs/experiments
```

### 3. Run Inference on an Image
Runs segmentation, cleans masks, extracts polygons, and outputs `prediction_report.json`:
```bash
python main.py infer --image path/to/aerial_image.jpg --checkpoint outputs/experiments/exp_001/checkpoints/best_iou.pth
```

### 4. Evaluate Model on Test Set
Computes Mean IoU, Rooftop IoU, Dice coefficient, and Pixel Accuracy on the official test set:
```bash
python main.py evaluate --checkpoint outputs/experiments/exp_001/checkpoints/best_iou.pth
```

---

## 🗺️ Multi-Phase Roadmap

- [x] **Phase 1: Rooftop Segmentation Framework**
  - [x] Pluggable model architecture (`BaseSegmentationModel` + Registry)
  - [x] SegFormer model integration (`nvidia/mit-b2`)
  - [x] Official Massachusetts dataset discovery pipeline
  - [x] Postprocessing (morphological cleaning, GeoJSON polygons, area estimation)
  - [x] TensorBoard logging, smart checkpointing (`best_iou.pth`), experiment versioning
  - [x] Standardized `prediction_report.json` output contract
- [ ] **Phase 2: Post-Installation Shadow & Obstacle Analytics**
- [ ] **Phase 3: Energy Generation & Financial Prediction Engine**
- [ ] **Phase 4: Panel Defect & Health Monitoring**
- [ ] **Phase 5: Production FastAPI & Interactive Next.js Web Dashboard**

*For detailed milestone breakdowns, see [ROADMAP.md](ROADMAP.md).*

---

## 🤝 Contributing

We welcome contributions! Please review our [CONTRIBUTING.md](CONTRIBUTING.md) and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) before submitting Pull Requests.

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 👥 Authors & Team

**MP Surya-Drishti Development Team**
* **EPICS Engineering Team** — VIT Bhopal University
* **Project Maintainers**: EPICS Project Contributors

---

## 💖 Acknowledgements

* [Massachusetts Buildings Dataset](https://www.kaggle.com/datasets/balraj98/massachusetts-buildings-dataset) by Volodymyr Mnih.
* [HuggingFace Transformers](https://huggingface.co/docs/transformers/) for SegFormer model architecture implementations.
* [PyTorch Ecosystem](https://pytorch.org/) for core deep learning libraries.
* **VIT Bhopal University** & **EPICS Program** for guidance and institutional support.
