"""
MP Surya-Drishti — Model Test Evaluation & Showcase Report Generator
====================================================================

This script evaluates trained segmentation models on the official test set and produces:
  1. Primary Model Performance: Native model resolution (512x512)
  2. Full-Resolution Raw Diagnostic: 1500x1500 raw reconstruction without cleaner
  3. Full-Resolution Postprocessed Diagnostic: 1500x1500 with MaskCleaner
  4. Per-image side-by-side comparison images, raw masks, and overlays
  5. Multi-evaluation pipeline comparison charts and confusion matrices
  6. Standardized JSON metrics artifacts

Usage:
    python test_evaluation_showcase.py
    python test_evaluation_showcase.py --checkpoint outputs/experiments/exp_002/checkpoints/best_loss.pth
    python test_evaluation_showcase.py --checkpoint outputs/experiments/exp_002/checkpoints/best_iou.pth
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import cv2
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast

# ── Framework Imports ──────────────────────────────────────────────
from models.registry import ensure_models_registered, load_model_from_checkpoint
from preprocessing.normalizer import ImageNormalizer
from preprocessing.dataset_loader import MassachusettsDataset
from preprocessing.augmentation import AugmentationPipeline
from evaluation.metrics import SegmentationMetrics
from postprocessing.mask_cleaner import MaskCleaner
from postprocessing.polygon_extractor import PolygonExtractor
from postprocessing.area_estimator import AreaEstimator
from utils.checkpoint_manager import find_best_available_checkpoint
from utils.device_utils import get_device

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_evaluation_showcase")


# ══════════════════════════════════════════════════════════════════
#  Image & Mask Helpers
# ══════════════════════════════════════════════════════════════════

def load_image_rgb(path: Path) -> np.ndarray:
    """Load an image file as RGB numpy array."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_mask_binary(path: Path) -> np.ndarray:
    """Load a mask file and binarize it ({0, 1})."""
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {path}")
    return (mask > 127).astype(np.uint8)


def create_colored_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.45,
) -> np.ndarray:
    """Create a colored overlay of the mask on the original image."""
    overlay = image.copy()
    color_layer = np.zeros_like(image)
    color_layer[mask > 0] = color
    region = mask > 0
    overlay[region] = (
        (1 - alpha) * image[region] + alpha * color_layer[region]
    ).astype(np.uint8)
    return overlay


# ══════════════════════════════════════════════════════════════════
#  Visualization Generators
# ══════════════════════════════════════════════════════════════════

def save_comparison_figure(
    original: np.ndarray,
    ground_truth: np.ndarray,
    prediction_raw: np.ndarray,
    overlay: np.ndarray,
    primary_metrics: dict[str, float],
    fullres_metrics: dict[str, float],
    save_path: Path,
    image_name: str,
) -> None:
    """Save a professional 4-panel comparison figure with raw model prediction."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    axes[0, 0].imshow(original)
    axes[0, 0].set_title("Original Satellite Image (1500×1500)", fontsize=13, fontweight="bold")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(ground_truth, cmap="gray", vmin=0, vmax=1)
    axes[0, 1].set_title("Ground Truth Rooftop Mask", fontsize=13, fontweight="bold")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(prediction_raw, cmap="gray", vmin=0, vmax=1)
    axes[1, 0].set_title("Predicted Rooftop Mask (Raw Model Output)", fontsize=13, fontweight="bold")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title("Raw Prediction Overlay", fontsize=13, fontweight="bold")
    axes[1, 1].axis("off")

    metric_text = (
        f"Primary (512×512) — IoU: {primary_metrics['iou']:.4f} | Dice: {primary_metrics['dice']:.4f} | Acc: {primary_metrics['pixel_accuracy']:.4f}\n"
        f"Full-Res (1500×1500 Raw) — IoU: {fullres_metrics['iou']:.4f} | Dice: {fullres_metrics['dice']:.4f} | Acc: {fullres_metrics['pixel_accuracy']:.4f}"
    )
    fig.suptitle(
        f"MP Surya-Drishti — Rooftop Segmentation\nSample: {image_name}\n{metric_text}",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_diagnostic_comparison_figure(
    original: np.ndarray,
    ground_truth: np.ndarray,
    prediction_raw: np.ndarray,
    prediction_cleaned: np.ndarray,
    raw_iou: float,
    cleaned_iou: float,
    save_path: Path,
    image_name: str,
) -> None:
    """Save a 4-panel diagnostic comparison contrasting Raw vs MaskCleaner."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    axes[0, 0].imshow(original)
    axes[0, 0].set_title("Original Satellite Image", fontsize=13, fontweight="bold")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(ground_truth, cmap="gray", vmin=0, vmax=1)
    axes[0, 1].set_title("Ground Truth Mask", fontsize=13, fontweight="bold")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(prediction_raw, cmap="gray", vmin=0, vmax=1)
    axes[1, 0].set_title(f"Raw Model Prediction (IoU: {raw_iou:.4f})", fontsize=13, fontweight="bold")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(prediction_cleaned, cmap="gray", vmin=0, vmax=1)
    axes[1, 1].set_title(f"Postprocessed Diagnostic — MaskCleaner (IoU: {cleaned_iou:.4f})", fontsize=13, fontweight="bold")
    axes[1, 1].axis("off")

    fig.suptitle(
        f"MP Surya-Drishti — Diagnostic Comparison: Raw vs MaskCleaner\nSample: {image_name}",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_metrics_bar_chart(
    primary_metrics: dict[str, Any],
    save_path: Path,
    model_name: str = "SegFormer-B2",
) -> None:
    """Save a bar chart of primary model metrics (512x512)."""
    metric_names = [
        "Pixel Accuracy",
        "Mean IoU",
        "Rooftop IoU",
        "Rooftop Dice",
        "Background IoU",
    ]
    metric_values = [
        primary_metrics["pixel_accuracy"],
        primary_metrics["mean_iou"],
        primary_metrics["rooftop_iou"],
        primary_metrics["rooftop_dice"],
        primary_metrics.get("background_iou", primary_metrics.get("iou_per_class", [0.0, 0.0])[0]),
    ]

    colors = ["#9C27B0", "#2196F3", "#FF9800", "#4CAF50", "#607D8B"]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(metric_names, metric_values, color=colors, width=0.55, edgecolor="white", linewidth=1.5)

    for bar, val in zip(bars, metric_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.015,
            f"{val * 100:.2f}%",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_ylim(0, 1.10)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(
        f"MP Surya-Drishti — Primary Model Performance ({model_name} @ 512×512)",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_pipeline_comparison_chart(
    primary_metrics: dict[str, Any],
    fullres_raw_metrics: dict[str, Any],
    fullres_clean_metrics: dict[str, Any],
    save_path: Path,
) -> None:
    """Save a multi-bar chart comparing the 3 evaluation modes."""
    modes = [
        "Primary (512×512)",
        "1500×1500 Raw",
        "1500×1500 + Cleaner",
    ]

    roof_iou = [
        primary_metrics["rooftop_iou"] * 100,
        fullres_raw_metrics["rooftop_iou"] * 100,
        fullres_clean_metrics["rooftop_iou"] * 100,
    ]
    roof_dice = [
        primary_metrics["rooftop_dice"] * 100,
        fullres_raw_metrics["rooftop_dice"] * 100,
        fullres_clean_metrics["rooftop_dice"] * 100,
    ]
    mean_iou = [
        primary_metrics["mean_iou"] * 100,
        fullres_raw_metrics["mean_iou"] * 100,
        fullres_clean_metrics["mean_iou"] * 100,
    ]
    accuracy = [
        primary_metrics["pixel_accuracy"] * 100,
        fullres_raw_metrics["pixel_accuracy"] * 100,
        fullres_clean_metrics["pixel_accuracy"] * 100,
    ]

    x = np.arange(len(modes))
    width = 0.20

    fig, ax = plt.subplots(figsize=(12, 6))

    b1 = ax.bar(x - 1.5 * width, roof_iou, width, label="Rooftop IoU", color="#FF9800")
    b2 = ax.bar(x - 0.5 * width, roof_dice, width, label="Rooftop Dice", color="#4CAF50")
    b3 = ax.bar(x + 0.5 * width, mean_iou, width, label="Mean IoU", color="#2196F3")
    b4 = ax.bar(x + 1.5 * width, accuracy, width, label="Pixel Accuracy", color="#9C27B0")

    for bar_group in [b1, b2, b3, b4]:
        for bar in bar_group:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 1.0,
                f"{h:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    ax.set_ylabel("Percentage (%)", fontsize=12)
    ax.set_title(
        "MP Surya-Drishti — Evaluation Mode Comparison\n(Primary vs Full-Resolution Diagnostics)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(modes, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_per_image_iou_chart(
    per_image_results: list[dict[str, Any]],
    save_path: Path,
) -> None:
    """Save a horizontal bar chart showing per-image Primary Rooftop IoU scores."""
    names = [r["image_name"] for r in per_image_results]
    ious = [r["primary_metrics"]["rooftop_iou"] * 100 for r in per_image_results]

    sorted_pairs = sorted(zip(names, ious), key=lambda x: x[1], reverse=True)
    names, ious = zip(*sorted_pairs)

    fig, ax = plt.subplots(figsize=(10, max(6, len(names) * 0.5)))

    colors = ["#4CAF50" if v >= 40.0 else "#FF9800" if v >= 25.0 else "#F44336" for v in ious]
    bars = ax.barh(range(len(names)), ious, color=colors, height=0.55)

    for i, (bar, val) in enumerate(zip(bars, ious)):
        ax.text(val + 0.8, i, f"{val:.2f}%", va="center", fontsize=10, fontweight="bold")

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlim(0, 105)
    ax.set_xlabel("Primary Rooftop IoU (%)", fontsize=12)
    ax.set_title(
        "MP Surya-Drishti — Per-Image Primary Rooftop IoU (512×512)",
        fontsize=14,
        fontweight="bold",
    )
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_confusion_matrix_chart(
    confusion_matrix: np.ndarray,
    save_path: Path,
    title_suffix: str = "Primary (512×512)",
) -> None:
    """Save a normalized confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(7, 6))

    cm_normalized = confusion_matrix.astype(float)
    row_sums = cm_normalized.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_percent = cm_normalized / row_sums * 100

    im = ax.imshow(cm_percent, cmap="Blues", vmin=0, vmax=100)
    plt.colorbar(im, label="Percentage (%)")

    labels = ["Background", "Rooftop"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_yticklabels(labels, fontsize=12)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Ground Truth", fontsize=12)

    for i in range(2):
        for j in range(2):
            count = int(confusion_matrix[i, j])
            pct = cm_percent[i, j]
            ax.text(
                j, i,
                f"{count:,}\n({pct:.1f}%)",
                ha="center", va="center",
                fontsize=11,
                fontweight="bold",
                color="white" if pct > 50 else "black",
            )

    ax.set_title(
        f"Confusion Matrix — {title_suffix}",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════
#  Main Evaluation Execution Engine
# ══════════════════════════════════════════════════════════════════

def run_evaluation_showcase(
    checkpoint_path: Optional[str | Path] = None,
    dataset_root: str | Path = "datasets/massachusetts",
    output_base_dir: str | Path = "outputs",
    image_size: int = 512,
) -> dict[str, Any]:
    """
    Execute complete test evaluation and showcase report generation.

    Args:
        checkpoint_path: Explicit checkpoint path. If None, auto-selects best via validation metrics.
        dataset_root: Root path to Massachusetts dataset.
        output_base_dir: Base output directory.
        image_size: Input resolution (default 512).

    Returns:
        Dictionary containing all dynamically computed metrics and report paths.
    """
    dataset_root = Path(dataset_root)
    test_images_dir = dataset_root / "test"
    test_masks_dir = dataset_root / "test_labels"

    if not test_images_dir.exists():
        raise FileNotFoundError(f"Test images directory not found: {test_images_dir}")

    # 1. Resolve Checkpoint
    if checkpoint_path is None:
        resolved_checkpoint = find_best_available_checkpoint()
    else:
        resolved_checkpoint = Path(checkpoint_path)
        if not resolved_checkpoint.exists():
            raise FileNotFoundError(f"Specified checkpoint not found: {resolved_checkpoint}")

    # 2. Output directory setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_base_dir) / f"test_evaluation_outputs_{timestamp}"
    images_dir = output_dir / "images"
    charts_dir = output_dir / "charts"
    images_dir.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Evaluation output directory: %s", output_dir)
    logger.info("Evaluating checkpoint: %s", resolved_checkpoint)

    # 3. Read Checkpoint Metadata
    device = get_device()
    ckpt_raw = torch.load(resolved_checkpoint, map_location="cpu", weights_only=False)
    epoch = ckpt_raw.get("epoch", -1)
    model_type = ckpt_raw.get("model_type", "segformer")
    backbone = ckpt_raw.get("backbone", "nvidia/mit-b2")

    checkpoint_info = {
        "checkpoint_path": str(resolved_checkpoint),
        "model_type": model_type,
        "backbone": backbone,
        "num_labels": ckpt_raw.get("num_labels", 2),
        "image_size": ckpt_raw.get("image_size", image_size),
        "epoch": epoch,
        "id2label": ckpt_raw.get("id2label", {0: "background", 1: "rooftop"}),
        "label2id": ckpt_raw.get("label2id", {"background": 0, "rooftop": 1}),
        "confidence_threshold": ckpt_raw.get("confidence_threshold", 0.5),
        "stored_val_metrics": ckpt_raw.get("metrics", {}),
        "has_optimizer_state": "optimizer_state_dict" in ckpt_raw,
        "has_scheduler_state": "scheduler_state_dict" in ckpt_raw,
        "has_scaler_state": "scaler_state_dict" in ckpt_raw,
        "has_random_state": "random_state" in ckpt_raw,
        "total_tensors": len(ckpt_raw.get("model_state_dict", {})),
    }

    with open(output_dir / "checkpoint_info.json", "w", encoding="utf-8") as f:
        json.dump(checkpoint_info, f, indent=2, default=str)

    # 4. Load Model
    ensure_models_registered()
    model = load_model_from_checkpoint(resolved_checkpoint, device=device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())

    # 5. Discover test samples
    img_paths, mask_paths = MassachusettsDataset.discover_pairs(
        root_dir=dataset_root,
        images_dir="test",
        masks_dir="test_labels",
    )

    if not img_paths:
        raise RuntimeError(f"No test image-mask pairs discovered in {dataset_root}")

    # 6. Prepare Evaluators
    aug = AugmentationPipeline(image_size=image_size)
    val_transform = aug.get_val_transform()
    mask_cleaner = MaskCleaner(min_region_area=50)
    polygon_extractor = PolygonExtractor()
    area_estimator = AreaEstimator(gsd=1.0)
    use_amp = device.type == "cuda"

    # Metric Accumulators
    primary_acc = SegmentationMetrics(num_classes=2)
    fullres_raw_acc = SegmentationMetrics(num_classes=2)
    fullres_clean_acc = SegmentationMetrics(num_classes=2)

    per_image_results: list[dict[str, Any]] = []
    total_inference_time_ms = 0.0

    print()
    print("=" * 65)
    print("  MP SURYA-DRISHTI — SEGFORMER-B2 EVALUATION & SHOWCASE")
    print("=" * 65)
    print(f"  Checkpoint : {resolved_checkpoint.name}")
    print(f"  Path       : {resolved_checkpoint}")
    print(f"  Epoch      : {epoch}")
    print(f"  Model      : {model_type} ({backbone})")
    print(f"  Parameters : {total_params:,}")
    print(f"  Samples    : {len(img_paths)} test images")
    print(f"  Device     : {device}")
    print("-" * 65)
    print("  RUNNING MULTI-MODE EVALUATION ON TEST SAMPLES...")
    print("-" * 65)

    for idx, (img_path, msk_path) in enumerate(zip(img_paths, mask_paths), 1):
        image_name = img_path.stem

        # Load original image and ground truth
        original = load_image_rgb(img_path)
        orig_h, orig_w = original.shape[:2]
        gt_mask_1500 = load_mask_binary(msk_path)

        # Create 512x512 ground truth for Primary Evaluation
        gt_mask_512 = cv2.resize(gt_mask_1500, (image_size, image_size), interpolation=cv2.INTER_NEAREST)

        # Preprocess Image using exact validation transform matching training
        resized_img = cv2.resize(original, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
        augmented = val_transform(image=resized_img, mask=gt_mask_512)
        img_tensor = torch.from_numpy(augmented["image"]).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(device)

        # Model Inference
        start_t = time.time()
        with torch.no_grad():
            with autocast(device_type="cuda", enabled=use_amp):
                prediction = model.predict(img_tensor)
        inference_ms = (time.time() - start_t) * 1000
        total_inference_time_ms += inference_ms

        # Raw 512x512 Prediction
        pred_mask_512 = prediction["binary_mask"].squeeze(0).cpu().numpy().astype(np.uint8)
        conf_map_512 = prediction["confidence_map"].squeeze(0).cpu().numpy()

        # Mode A: Primary Evaluation (512x512 native)
        primary_single = SegmentationMetrics.compute_single(pred_mask_512, gt_mask_512)
        primary_acc.update(
            torch.from_numpy(pred_mask_512).unsqueeze(0),
            torch.from_numpy(gt_mask_512).unsqueeze(0),
        )

        # Mode B: Full-Resolution Raw Reconstruction (1500x1500 raw nearest neighbor upsampling)
        pred_mask_1500_raw = cv2.resize(pred_mask_512, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        fullres_raw_single = SegmentationMetrics.compute_single(pred_mask_1500_raw, gt_mask_1500)
        fullres_raw_acc.update(
            torch.from_numpy(pred_mask_1500_raw).unsqueeze(0),
            torch.from_numpy(gt_mask_1500).unsqueeze(0),
        )

        # Mode C: Full-Resolution Postprocessed Diagnostic (MaskCleaner)
        pred_mask_512_cleaned = mask_cleaner.clean(pred_mask_512)
        pred_mask_1500_cleaned = cv2.resize(pred_mask_512_cleaned, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        fullres_clean_single = SegmentationMetrics.compute_single(pred_mask_1500_cleaned, gt_mask_1500)
        fullres_clean_acc.update(
            torch.from_numpy(pred_mask_1500_cleaned).unsqueeze(0),
            torch.from_numpy(gt_mask_1500).unsqueeze(0),
        )

        # Confidence & Geometry Extraction (on Raw Prediction)
        roof_pixels = pred_mask_512 > 0
        mean_conf = float(conf_map_512[roof_pixels].mean()) if roof_pixels.any() else 0.0
        area_info = area_estimator.estimate(pred_mask_1500_raw)
        polygons = polygon_extractor.extract(pred_mask_1500_raw)
        raw_overlay = create_colored_overlay(original, pred_mask_1500_raw)

        # Save Visual Artifacts
        prefix = f"{idx:03d}_{image_name}"

        # 4-Panel Primary Comparison Figure
        save_comparison_figure(
            original=original,
            ground_truth=gt_mask_1500,
            prediction_raw=pred_mask_1500_raw,
            overlay=raw_overlay,
            primary_metrics=primary_single,
            fullres_metrics=fullres_raw_single,
            save_path=images_dir / f"{prefix}_comparison.png",
            image_name=image_name,
        )

        # Diagnostic Comparison (Raw vs Cleaner)
        save_diagnostic_comparison_figure(
            original=original,
            ground_truth=gt_mask_1500,
            prediction_raw=pred_mask_1500_raw,
            prediction_cleaned=pred_mask_1500_cleaned,
            raw_iou=fullres_raw_single["iou"],
            cleaned_iou=fullres_clean_single["iou"],
            save_path=images_dir / f"{prefix}_diagnostic_cleaner.png",
            image_name=image_name,
        )

        # Raw Mask & Overlay Images
        cv2.imwrite(str(images_dir / f"{prefix}_raw_mask.png"), pred_mask_1500_raw * 255)
        cv2.imwrite(str(images_dir / f"{prefix}_cleaned_mask_diagnostic.png"), pred_mask_1500_cleaned * 255)
        cv2.imwrite(str(images_dir / f"{prefix}_overlay.png"), cv2.cvtColor(raw_overlay, cv2.COLOR_RGB2BGR))

        entry = {
            "index": idx,
            "image_name": image_name,
            "primary_metrics": {
                "rooftop_iou": round(primary_single["iou"], 4),
                "rooftop_dice": round(primary_single["dice"], 4),
                "pixel_accuracy": round(primary_single["pixel_accuracy"], 4),
            },
            "full_resolution_raw_metrics": {
                "rooftop_iou": round(fullres_raw_single["iou"], 4),
                "rooftop_dice": round(fullres_raw_single["dice"], 4),
                "pixel_accuracy": round(fullres_raw_single["pixel_accuracy"], 4),
            },
            "full_resolution_cleaned_metrics": {
                "rooftop_iou": round(fullres_clean_single["iou"], 4),
                "rooftop_dice": round(fullres_clean_single["dice"], 4),
                "pixel_accuracy": round(fullres_clean_single["pixel_accuracy"], 4),
            },
            "confidence": round(mean_conf, 4),
            "roof_area_pixels": area_info["roof_area_pixels"],
            "roof_area_percent": round(area_info["roof_area_percent"], 2),
            "polygons_count": len(polygons),
            "inference_time_ms": round(inference_ms, 1),
        }
        per_image_results.append(entry)

        print(
            f"  [{idx:02d}/{len(img_paths)}] {image_name:18s} | "
            f"Primary IoU: {primary_single['iou']*100:5.2f}% | "
            f"Full-Res Raw: {fullres_raw_single['iou']*100:5.2f}% | "
            f"Cleaner: {fullres_clean_single['iou']*100:5.2f}% | "
            f"Time: {inference_ms:4.0f}ms"
        )

    # 7. Compute Global Aggregate Metrics
    res_primary = primary_acc.compute()
    res_fullres_raw = fullres_raw_acc.compute()
    res_fullres_clean = fullres_clean_acc.compute()

    primary_payload = {
        "pixel_accuracy": round(res_primary["pixel_accuracy"], 4),
        "mean_iou": round(res_primary["iou"], 4),
        "rooftop_iou": round(res_primary["rooftop_iou"], 4),
        "rooftop_dice": round(res_primary["rooftop_dice"], 4),
        "background_iou": round(res_primary["iou_per_class"][0], 4) if len(res_primary["iou_per_class"]) > 0 else 0.0,
    }

    fullres_raw_payload = {
        "pixel_accuracy": round(res_fullres_raw["pixel_accuracy"], 4),
        "mean_iou": round(res_fullres_raw["iou"], 4),
        "rooftop_iou": round(res_fullres_raw["rooftop_iou"], 4),
        "rooftop_dice": round(res_fullres_raw["rooftop_dice"], 4),
        "background_iou": round(res_fullres_raw["iou_per_class"][0], 4) if len(res_fullres_raw["iou_per_class"]) > 0 else 0.0,
    }

    fullres_clean_payload = {
        "pixel_accuracy": round(res_fullres_clean["pixel_accuracy"], 4),
        "mean_iou": round(res_fullres_clean["iou"], 4),
        "rooftop_iou": round(res_fullres_clean["rooftop_iou"], 4),
        "rooftop_dice": round(res_fullres_clean["rooftop_dice"], 4),
        "background_iou": round(res_fullres_clean["iou_per_class"][0], 4) if len(res_fullres_clean["iou_per_class"]) > 0 else 0.0,
    }

    # 8. Save JSON Outputs
    with open(output_dir / "per_image_metrics.json", "w", encoding="utf-8") as f:
        json.dump(per_image_results, f, indent=2)

    eval_results_json = {
        "checkpoint": str(resolved_checkpoint),
        "epoch": epoch,
        "dataset": "Massachusetts Buildings Dataset",
        "split": "test",
        "num_samples": len(img_paths),
        "model_type": model_type,
        "backbone": backbone,
        "image_size": image_size,
        "primary_evaluation": primary_payload,
        "full_resolution_raw": fullres_raw_payload,
        "full_resolution_cleaned": fullres_clean_payload,
        "total_inference_time_ms": round(total_inference_time_ms, 1),
        "avg_inference_time_ms": round(total_inference_time_ms / len(img_paths), 1),
    }

    with open(output_dir / "test_evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(eval_results_json, f, indent=2)

    # 9. Generate Charts
    logger.info("Generating charts...")
    save_metrics_bar_chart(primary_payload, charts_dir / "metrics_bar_chart.png", model_name="SegFormer-B2")
    save_pipeline_comparison_chart(primary_payload, fullres_raw_payload, fullres_clean_payload, charts_dir / "pipeline_comparison_chart.png")
    save_per_image_iou_chart(per_image_results, charts_dir / "per_image_iou_chart.png")
    save_confusion_matrix_chart(primary_acc.confusion_matrix, charts_dir / "confusion_matrix.png", title_suffix="Primary (512×512)")

    # 10. Generate Summary Report Text File
    report_lines = [
        "=" * 65,
        "  MP SURYA-DRISHTI — SEGFORMER-B2 EVALUATION REPORT",
        "  Best Verified Baseline Performance & Diagnostics",
        "=" * 65,
        "",
        f"  Report Date : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"  Checkpoint  : {resolved_checkpoint.name}",
        f"  Epoch       : {epoch}",
        f"  Model       : {model_type} ({backbone})",
        f"  Parameters  : {total_params:,}",
        f"  Dataset     : Massachusetts Buildings Dataset",
        f"  Split       : test",
        f"  Samples     : {len(img_paths)} satellite images",
        f"  Resolution  : {image_size}×{image_size}",
        "",
        "=" * 65,
        "  PERFORMANCE CARD (PRIMARY BASELINE)",
        "=" * 65,
        f"  MODEL             : SegFormer-B2 ({backbone})",
        f"  CHECKPOINT        : {resolved_checkpoint.name}",
        f"  EPOCH             : {epoch}",
        f"  TEST IMAGES       : {len(img_paths)}",
        f"  PIXEL ACCURACY    : {primary_payload['pixel_accuracy'] * 100:.2f}%",
        f"  MEAN IoU          : {primary_payload['mean_iou'] * 100:.2f}%",
        f"  ROOFTOP IoU       : {primary_payload['rooftop_iou'] * 100:.2f}%",
        f"  ROOFTOP DICE      : {primary_payload['rooftop_dice'] * 100:.2f}%",
        f"  BACKGROUND IoU    : {primary_payload['background_iou'] * 100:.2f}%",
        "",
        "=" * 65,
        "  PIPELINE COMPARISON TABLE",
        "=" * 65,
        f"  {'Evaluation Mode':<25s} | {'Pixel Acc':>9s} | {'Mean IoU':>8s} | {'Roof IoU':>8s} | {'Roof Dice':>9s} |",
        "  " + "-" * 63,
        f"  {'512×512 Primary':<25s} | {primary_payload['pixel_accuracy']*100:8.2f}% | {primary_payload['mean_iou']*100:7.2f}% | {primary_payload['rooftop_iou']*100:7.2f}% | {primary_payload['rooftop_dice']*100:8.2f}% |",
        f"  {'1500×1500 Raw':<25s} | {fullres_raw_payload['pixel_accuracy']*100:8.2f}% | {fullres_raw_payload['mean_iou']*100:7.2f}% | {fullres_raw_payload['rooftop_iou']*100:7.2f}% | {fullres_raw_payload['rooftop_dice']*100:8.2f}% |",
        f"  {'1500×1500 + Cleaner':<25s} | {fullres_clean_payload['pixel_accuracy']*100:8.2f}% | {fullres_clean_payload['mean_iou']*100:7.2f}% | {fullres_clean_payload['rooftop_iou']*100:7.2f}% | {fullres_clean_payload['rooftop_dice']*100:8.2f}% |",
        "  " + "-" * 63,
        "",
        "  NOTE ON POSTPROCESSING:",
        "  MaskCleaner currently filters small rooftop clusters as noise,",
        "  which reduces measured rooftop IoU on this test set. Therefore,",
        "  the raw model output is presented as the primary baseline.",
        "",
        "=" * 65,
        "  PER-IMAGE BREAKDOWN",
        "=" * 65,
        f"  {'#':>3s} | {'Image Name':<18s} | {'Primary IoU':>11s} | {'Raw 1500 IoU':>12s} | {'Cleaned IoU':>11s} |",
        "  " + "-" * 63,
    ]

    for r in per_image_results:
        report_lines.append(
            f"  {r['index']:3d} | {r['image_name']:<18s} | {r['primary_metrics']['rooftop_iou']*100:10.2f}% | {r['full_resolution_raw_metrics']['rooftop_iou']*100:11.2f}% | {r['full_resolution_cleaned_metrics']['rooftop_iou']*100:10.2f}% |"
        )

    report_lines.extend([
        "  " + "-" * 63,
        "",
        "=" * 65,
        "  GENERATED OUTPUT ARTIFACTS",
        "=" * 65,
        f"  Output Directory : {output_dir}",
        f"  Results JSON     : test_evaluation_results.json",
        f"  Per-Image JSON   : per_image_metrics.json",
        f"  Checkpoint Info  : checkpoint_info.json",
        f"  Summary Report   : summary_report.txt",
        f"  Charts Folder    : charts/",
        f"  Images Folder    : images/ ({len(img_paths)*3} files)",
        "=" * 65,
    ])

    report_text = "\n".join(report_lines)
    with open(output_dir / "summary_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    # 11. Terminal Output
    print()
    print("=" * 65)
    print("MP SURYA-DRISHTI — SEGFORMER-B2 EVALUATION")
    print("=" * 65)
    print(f"Checkpoint : {resolved_checkpoint.name}")
    print(f"Epoch      : {epoch}")
    print(f"Dataset    : Massachusetts Buildings")
    print(f"Split      : test")
    print(f"Samples    : {len(img_paths)}")
    print(f"Resolution : {image_size}×{image_size}")
    print()
    print("PRIMARY MODEL PERFORMANCE")
    print("-" * 65)
    print(f"Pixel Accuracy : {primary_payload['pixel_accuracy'] * 100:.2f}%")
    print(f"Mean IoU       : {primary_payload['mean_iou'] * 100:.2f}%")
    print(f"Rooftop IoU    : {primary_payload['rooftop_iou'] * 100:.2f}%")
    print(f"Rooftop Dice   : {primary_payload['rooftop_dice'] * 100:.2f}%")
    print()
    print("FULL-RESOLUTION DIAGNOSTICS")
    print("-" * 65)
    print(f"Raw 1500x1500 Rooftop IoU     : {fullres_raw_payload['rooftop_iou'] * 100:.2f}%")
    print(f"MaskCleaner Rooftop IoU       : {fullres_clean_payload['rooftop_iou'] * 100:.2f}%")
    print()
    print("NOTE:")
    print("MaskCleaner currently reduces measured rooftop IoU on this")
    print("test set and is therefore NOT used for the primary result.")
    print("=" * 65)
    print()
    print(f"[OK] Outputs saved to: {output_dir}")
    print()

    return eval_results_json


def main() -> None:
    """CLI Entry Point."""
    parser = argparse.ArgumentParser(
        description="Evaluate segmentation models and generate showcase artifacts.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to checkpoint (auto-selects best available via validation metrics if omitted)",
    )
    parser.add_argument(
        "--dataset-root",
        default="datasets/massachusetts",
        help="Path to dataset root directory",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Base outputs directory",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=512,
        help="Model input image size (default 512)",
    )

    args = parser.parse_args()

    run_evaluation_showcase(
        checkpoint_path=args.checkpoint,
        dataset_root=args.dataset_root,
        output_base_dir=args.output_dir,
        image_size=args.image_size,
    )


if __name__ == "__main__":
    main()
