# Massachusetts Buildings Dataset

Place the official dataset files in this directory structure:

```
datasets/massachusetts/
├── train/              ← Training aerial images (.tiff / .png)
├── train_labels/       ← Training rooftop binary masks (.tiff / .png)
├── val/                ← Validation aerial images
├── val_labels/         ← Validation rooftop binary masks
├── test/               ← Test aerial images
└── test_labels/        ← Test rooftop binary masks
```

## Download Instructions

Download the dataset from Kaggle:
https://www.kaggle.com/datasets/balraj98/massachusetts-buildings-dataset

Extract the subfolders directly into `datasets/massachusetts/`.

Each aerial image must have a corresponding mask in the matching labels folder with the **same filename stem** (e.g. `22828930_15.tiff` in `train/` ↔ `22828930_15.tiff` in `train_labels/`).

## Dataset Verification

Run the verification command to check file pairing across all splits:

```bash
python main.py verify-dataset
```
