"""
MP Surya-Drishti — Model Test Evaluation & Output Generator
============================================================

This script produces a complete, self-contained test evaluation report folder with:
  1. Per-image inference results (mask, overlay, side-by-side comparison)
  2. Per-image metrics (IoU, Dice, Pixel Accuracy)
  3. Aggregate evaluation metrics across the full test set
  4. Checkpoint metadata & model architecture summary
  5. Summary charts and evaluation report

Run:
    python test_evaluation_showcase.py

Output:
    outputs/test_evaluation_outputs_YYYYMMDD_HHMMSS/
        ├── checkpoint_info.json
        ├── test_evaluation_results.json
        ├── per_image_metrics.json
        ├── summary_report.txt
        ├── charts/
        │   ├── metrics_bar_chart.png
        │   ├── per_image_iou_chart.png
        │   └── confusion_matrix.png
        └── images/
            ├── 001_<name>_comparison.png
            ├── 001_<name>_overlay.png
            ├── 001_<name>_mask.png
            ├── ...
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

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
from evaluation.metrics import SegmentationMetrics
from postprocessing.mask_cleaner import MaskCleaner
from postprocessing.polygon_extractor import PolygonExtractor
from postprocessing.area_estimator import AreaEstimator
from utils.device_utils import get_device

# ── Configuration ──────────────────────────────────────────────────
CHECKPOINT_PATH = Path("outputs/experiments/exp_002/checkpoints/best_iou.pth")
TEST_IMAGES_DIR = Path("datasets/massachusetts/test")
TEST_MASKS_DIR = Path("datasets/massachusetts/test_labels")
IMAGE_SIZE = 512
OUTPUT_BASE = Path("outputs")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_evaluation_showcase")


# ══════════════════════════════════════════════════════════════════
#  Helper Functions
# ══════════════════════════════════════════════════════════════════

def load_image_rgb(path: Path) -> np.ndarray:
    """Load an image file as RGB numpy array."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_mask_binary(path: Path) -> np.ndarray:
    """Load a mask file and binarize it (0 or 1)."""
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {path}")
    return (mask > 127).astype(np.uint8)


def create_colored_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple = (0, 255, 0),
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


def save_comparison_figure(
    original: np.ndarray,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    overlay: np.ndarray,
    metrics: dict,
    save_path: Path,
    image_name: str,
) -> None:
    """Save a professional 4-panel comparison figure."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    axes[0, 0].imshow(original)
    axes[0, 0].set_title("Original Satellite Image", fontsize=13, fontweight="bold")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(ground_truth, cmap="gray", vmin=0, vmax=1)
    axes[0, 1].set_title("Ground Truth Mask", fontsize=13, fontweight="bold")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(prediction, cmap="gray", vmin=0, vmax=1)
    axes[1, 0].set_title("Model Prediction", fontsize=13, fontweight="bold")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title("Prediction Overlay", fontsize=13, fontweight="bold")
    axes[1, 1].axis("off")

    metric_text = (
        f"IoU: {metrics['iou']:.4f}  |  "
        f"Dice: {metrics['dice']:.4f}  |  "
        f"Pixel Accuracy: {metrics['pixel_accuracy']:.4f}"
    )
    fig.suptitle(
        f"MP Surya-Drishti — Rooftop Segmentation\n{image_name}\n{metric_text}",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_metrics_bar_chart(
    aggregate_metrics: dict,
    save_path: Path,
) -> None:
    """Save a professional bar chart of aggregate metrics."""
    metric_names = [
        "Mean IoU",
        "Rooftop IoU",
        "Mean Dice",
        "Rooftop Dice",
        "Pixel Accuracy",
    ]
    metric_values = [
        aggregate_metrics["iou"],
        aggregate_metrics["rooftop_iou"],
        aggregate_metrics["dice"],
        aggregate_metrics["rooftop_dice"],
        aggregate_metrics["pixel_accuracy"],
    ]

    colors = ["#2196F3", "#FF9800", "#4CAF50", "#F44336", "#9C27B0"]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(metric_names, metric_values, color=colors, width=0.6, edgecolor="white", linewidth=1.5)

    for bar, val in zip(bars, metric_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(
        "MP Surya-Drishti — Aggregate Test Set Metrics (SegFormer-B2)",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_per_image_iou_chart(
    per_image_results: list[dict],
    save_path: Path,
) -> None:
    """Save a horizontal bar chart showing per-image IoU scores."""
    names = [r["image_name"] for r in per_image_results]
    ious = [r["metrics"]["iou"] for r in per_image_results]

    # Sort by IoU descending
    sorted_pairs = sorted(zip(names, ious), key=lambda x: x[1], reverse=True)
    names, ious = zip(*sorted_pairs)

    fig, ax = plt.subplots(figsize=(10, max(6, len(names) * 0.5)))

    colors = ["#4CAF50" if v >= 0.5 else "#FF9800" if v >= 0.3 else "#F44336" for v in ious]
    bars = ax.barh(range(len(names)), ious, color=colors, height=0.6)

    for i, (bar, val) in enumerate(zip(bars, ious)):
        ax.text(val + 0.01, i, f"{val:.4f}", va="center", fontsize=10, fontweight="bold")

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("IoU Score", fontsize=12)
    ax.set_title(
        "MP Surya-Drishti — Per-Image Rooftop IoU Scores",
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
) -> None:
    """Save a confusion matrix heatmap."""
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
        "Confusion Matrix — Test Set",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════
#  Main Evaluation Pipeline
# ══════════════════════════════════════════════════════════════════

def main() -> None:
    """Run the complete test evaluation pipeline."""
    print()
    print("=" * 70)
    print("  MP SURYA-DRISHTI — TEST EVALUATION & SHOWCASE REPORT")
    print("  AI-Assisted Community Solar Advisory System")
    print("  Phase 1: Rooftop Segmentation (SegFormer-B2)")
    print("=" * 70)
    print()

    # ── Validate prerequisites ─────────────────────────────────
    if not CHECKPOINT_PATH.exists():
        logger.error("Checkpoint not found: %s", CHECKPOINT_PATH)
        logger.error("Please ensure trained checkpoints exist in outputs/experiments/")
        sys.exit(1)

    if not TEST_IMAGES_DIR.exists():
        logger.error("Test images not found: %s", TEST_IMAGES_DIR)
        sys.exit(1)

    # ── Create output directory ────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_BASE / f"test_evaluation_outputs_{timestamp}"
    images_dir = output_dir / "images"
    charts_dir = output_dir / "charts"
    images_dir.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Output directory: %s", output_dir)

    # ── Load checkpoint metadata ───────────────────────────────
    logger.info("Loading checkpoint: %s", CHECKPOINT_PATH)
    device = get_device()

    checkpoint_raw = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    checkpoint_info = {
        "checkpoint_path": str(CHECKPOINT_PATH),
        "model_type": checkpoint_raw.get("model_type", "segformer"),
        "backbone": checkpoint_raw.get("backbone", "nvidia/mit-b2"),
        "num_labels": checkpoint_raw.get("num_labels", 2),
        "image_size": checkpoint_raw.get("image_size", IMAGE_SIZE),
        "epoch": checkpoint_raw.get("epoch", "unknown"),
        "id2label": checkpoint_raw.get("id2label", {0: "background", 1: "rooftop"}),
        "label2id": checkpoint_raw.get("label2id", {"background": 0, "rooftop": 1}),
        "confidence_threshold": checkpoint_raw.get("confidence_threshold", 0.5),
        "training_metrics": checkpoint_raw.get("metrics", {}),
        "has_optimizer_state": "optimizer_state_dict" in checkpoint_raw,
        "has_scheduler_state": "scheduler_state_dict" in checkpoint_raw,
        "has_scaler_state": "scaler_state_dict" in checkpoint_raw,
        "has_random_state": "random_state" in checkpoint_raw,
        "has_config": "config" in checkpoint_raw,
        "total_parameters": len(checkpoint_raw.get("model_state_dict", {})),
    }

    with open(output_dir / "checkpoint_info.json", "w", encoding="utf-8") as f:
        json.dump(checkpoint_info, f, indent=2, default=str)

    print(f"  Checkpoint:   {CHECKPOINT_PATH}")
    print(f"  Model:        {checkpoint_info['model_type']} ({checkpoint_info['backbone']})")
    print(f"  Epoch:        {checkpoint_info['epoch']}")
    print(f"  Device:       {device}")
    print(f"  Image Size:   {IMAGE_SIZE}×{IMAGE_SIZE}")
    print()

    # ── Load Model ─────────────────────────────────────────────
    logger.info("Loading model from checkpoint (strict=True, pretrained=False)...")
    ensure_models_registered()
    model = load_model_from_checkpoint(CHECKPOINT_PATH, device=device)
    model.eval()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    checkpoint_info["total_param_count"] = total_params
    checkpoint_info["trainable_param_count"] = trainable_params

    print(f"  Total Parameters:     {total_params:,}")
    print(f"  Trainable Parameters: {trainable_params:,}")
    print()

    # ── Setup Preprocessing & Postprocessing ───────────────────
    normalizer = ImageNormalizer(
        backbone=checkpoint_info["backbone"],
        image_size=IMAGE_SIZE,
        model_type=checkpoint_info["model_type"],
    )
    mask_cleaner = MaskCleaner()
    polygon_extractor = PolygonExtractor()
    area_estimator = AreaEstimator(gsd=1.0)
    metrics_accumulator = SegmentationMetrics(num_classes=2)
    use_amp = device.type == "cuda"

    # ── Discover test image-mask pairs ─────────────────────────
    test_images = sorted(TEST_IMAGES_DIR.glob("*.png"))
    if not test_images:
        test_images = sorted(TEST_IMAGES_DIR.glob("*.tif"))
    if not test_images:
        test_images = sorted(TEST_IMAGES_DIR.glob("*.jpg"))

    logger.info("Found %d test images", len(test_images))
    print(f"  Test Images Found: {len(test_images)}")
    print()
    print("-" * 70)
    print("  RUNNING INFERENCE ON ALL TEST IMAGES...")
    print("-" * 70)
    print()

    # ── Per-Image Inference Loop ───────────────────────────────
    per_image_results = []
    total_inference_time = 0.0

    for idx, image_path in enumerate(test_images, 1):
        mask_path = TEST_MASKS_DIR / image_path.name
        image_name = image_path.stem

        logger.info("[%d/%d] Processing: %s", idx, len(test_images), image_name)

        # Load original image and ground truth
        original = load_image_rgb(image_path)
        original_h, original_w = original.shape[:2]

        has_gt = mask_path.exists()
        if has_gt:
            gt_mask = load_mask_binary(mask_path)
        else:
            gt_mask = np.zeros((original_h, original_w), dtype=np.uint8)
            logger.warning("  No ground truth mask found for: %s", image_name)

        # Preprocess
        resized = cv2.resize(original, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
        image_tensor = normalizer.normalize(resized).unsqueeze(0).to(device)

        # Inference
        start = time.time()
        with torch.no_grad():
            with autocast(device_type="cuda", enabled=use_amp):
                prediction = model.predict(image_tensor)
        inference_ms = (time.time() - start) * 1000
        total_inference_time += inference_ms

        # Extract prediction mask
        pred_mask = prediction["binary_mask"].squeeze(0).cpu().numpy().astype(np.uint8)
        confidence_map = prediction["confidence_map"].squeeze(0).cpu().numpy()

        # Postprocess: clean and resize to original resolution
        pred_mask_cleaned = mask_cleaner.clean(pred_mask)
        pred_mask_original = cv2.resize(
            pred_mask_cleaned, (original_w, original_h), interpolation=cv2.INTER_NEAREST
        )

        # Compute per-image metrics
        if has_gt:
            img_metrics = SegmentationMetrics.compute_single(pred_mask_original, gt_mask)
            # Also accumulate for aggregate metrics
            gt_tensor = torch.from_numpy(gt_mask).unsqueeze(0).long()
            pred_tensor = torch.from_numpy(pred_mask_original).unsqueeze(0).long()
            metrics_accumulator.update(pred_tensor, gt_tensor)
        else:
            img_metrics = {"iou": 0.0, "dice": 0.0, "pixel_accuracy": 0.0}

        # Confidence score
        rooftop_pixels = pred_mask > 0
        confidence = float(confidence_map[rooftop_pixels].mean()) if rooftop_pixels.any() else 0.0

        # Area estimation
        area_result = area_estimator.estimate(pred_mask_original)

        # Polygons
        polygons = polygon_extractor.extract(pred_mask_original)

        # Create overlay
        overlay = create_colored_overlay(original, pred_mask_original)

        # ── Save images ────────────────────────────────────────
        prefix = f"{idx:03d}_{image_name}"

        # Save comparison figure (4-panel)
        save_comparison_figure(
            original=original,
            ground_truth=gt_mask if has_gt else np.zeros_like(pred_mask_original),
            prediction=pred_mask_original,
            overlay=overlay,
            metrics=img_metrics,
            save_path=images_dir / f"{prefix}_comparison.png",
            image_name=image_name,
        )

        # Save individual outputs
        cv2.imwrite(
            str(images_dir / f"{prefix}_mask.png"),
            pred_mask_original * 255,
        )
        cv2.imwrite(
            str(images_dir / f"{prefix}_overlay.png"),
            cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
        )

        # Store result
        result_entry = {
            "index": idx,
            "image_name": image_name,
            "image_path": str(image_path),
            "has_ground_truth": has_gt,
            "metrics": img_metrics,
            "confidence": round(confidence, 4),
            "roof_area_pixels": area_result["roof_area_pixels"],
            "roof_area_percent": round(area_result["roof_area_percent"], 2),
            "usable_area_percent": round(area_result["usable_area_percent"], 2),
            "polygons_found": len(polygons),
            "inference_time_ms": round(inference_ms, 1),
        }
        per_image_results.append(result_entry)

        print(
            f"  [{idx:02d}/{len(test_images)}] {image_name:20s}  "
            f"IoU={img_metrics['iou']:.4f}  "
            f"Dice={img_metrics['dice']:.4f}  "
            f"Acc={img_metrics['pixel_accuracy']:.4f}  "
            f"Conf={confidence:.3f}  "
            f"Time={inference_ms:.0f}ms"
        )

    # ── Aggregate Metrics ──────────────────────────────────────
    print()
    print("-" * 70)
    print("  COMPUTING AGGREGATE METRICS...")
    print("-" * 70)

    aggregate_metrics = metrics_accumulator.compute()
    confusion_matrix = metrics_accumulator.confusion_matrix.copy()

    # Average per-image metrics
    avg_iou = np.mean([r["metrics"]["iou"] for r in per_image_results])
    avg_dice = np.mean([r["metrics"]["dice"] for r in per_image_results])
    avg_acc = np.mean([r["metrics"]["pixel_accuracy"] for r in per_image_results])
    avg_conf = np.mean([r["confidence"] for r in per_image_results])
    avg_time = total_inference_time / len(per_image_results) if per_image_results else 0

    # ── Save JSON Results ──────────────────────────────────────
    with open(output_dir / "per_image_metrics.json", "w", encoding="utf-8") as f:
        json.dump(per_image_results, f, indent=2)

    eval_results = {
        "model": checkpoint_info["model_type"],
        "backbone": checkpoint_info["backbone"],
        "checkpoint": str(CHECKPOINT_PATH),
        "training_epoch": checkpoint_info["epoch"],
        "test_images_count": len(test_images),
        "aggregate_metrics": {
            "mean_iou": round(aggregate_metrics["iou"], 4),
            "rooftop_iou": round(aggregate_metrics["rooftop_iou"], 4),
            "background_iou": round(aggregate_metrics["iou_per_class"][0], 4),
            "mean_dice": round(aggregate_metrics["dice"], 4),
            "rooftop_dice": round(aggregate_metrics["rooftop_dice"], 4),
            "pixel_accuracy": round(aggregate_metrics["pixel_accuracy"], 4),
        },
        "per_image_averages": {
            "avg_iou": round(float(avg_iou), 4),
            "avg_dice": round(float(avg_dice), 4),
            "avg_pixel_accuracy": round(float(avg_acc), 4),
            "avg_confidence": round(float(avg_conf), 4),
            "avg_inference_time_ms": round(avg_time, 1),
        },
        "confusion_matrix": confusion_matrix.tolist(),
        "total_inference_time_ms": round(total_inference_time, 1),
    }

    with open(output_dir / "test_evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)

    # ── Generate Charts ────────────────────────────────────────
    logger.info("Generating charts...")

    save_metrics_bar_chart(aggregate_metrics, charts_dir / "metrics_bar_chart.png")
    save_per_image_iou_chart(per_image_results, charts_dir / "per_image_iou_chart.png")
    save_confusion_matrix_chart(confusion_matrix, charts_dir / "confusion_matrix.png")

    # ── Generate Summary Report ────────────────────────────────
    report_lines = [
        "=" * 70,
        "  MP SURYA-DRISHTI — TEST EVALUATION & SHOWCASE REPORT",
        "  AI-Assisted Community Solar Advisory System",
        "  Phase 1: Rooftop Segmentation using SegFormer-B2",
        "=" * 70,
        "",
        f"  Report Generated:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"  Checkpoint:        {CHECKPOINT_PATH}",
        f"  Training Epoch:    {checkpoint_info['epoch']}",
        f"  Model:             {checkpoint_info['model_type']} ({checkpoint_info['backbone']})",
        f"  Parameters:        {total_params:,} ({trainable_params:,} trainable)",
        f"  Device:            {device}",
        f"  Image Size:        {IMAGE_SIZE}×{IMAGE_SIZE}",
        f"  Test Images:       {len(test_images)}",
        "",
        "-" * 70,
        "  AGGREGATE METRICS (Accumulated Confusion Matrix)",
        "-" * 70,
        f"  Mean IoU:          {aggregate_metrics['iou']:.4f}",
        f"  Rooftop IoU:       {aggregate_metrics['rooftop_iou']:.4f}",
        f"  Background IoU:    {aggregate_metrics['iou_per_class'][0]:.4f}",
        f"  Mean Dice:         {aggregate_metrics['dice']:.4f}",
        f"  Rooftop Dice:      {aggregate_metrics['rooftop_dice']:.4f}",
        f"  Pixel Accuracy:    {aggregate_metrics['pixel_accuracy']:.4f}",
        "",
        "-" * 70,
        "  PER-IMAGE AVERAGES",
        "-" * 70,
        f"  Average IoU:              {avg_iou:.4f}",
        f"  Average Dice:             {avg_dice:.4f}",
        f"  Average Pixel Accuracy:   {avg_acc:.4f}",
        f"  Average Confidence:       {avg_conf:.4f}",
        f"  Average Inference Time:   {avg_time:.1f} ms",
        f"  Total Inference Time:     {total_inference_time:.1f} ms",
        "",
        "-" * 70,
        "  CHECKPOINT CONTENTS",
        "-" * 70,
        f"  model_state_dict:    YES ({checkpoint_info['total_parameters']} tensors)",
        f"  optimizer_state_dict: {'YES' if checkpoint_info['has_optimizer_state'] else 'NO'}",
        f"  scheduler_state_dict: {'YES' if checkpoint_info['has_scheduler_state'] else 'NO'}",
        f"  scaler_state_dict:   {'YES' if checkpoint_info['has_scaler_state'] else 'NO'}",
        f"  random_state:        {'YES' if checkpoint_info['has_random_state'] else 'NO'}",
        f"  config:              {'YES' if checkpoint_info['has_config'] else 'NO'}",
        "",
        "-" * 70,
        "  PER-IMAGE RESULTS",
        "-" * 70,
    ]

    header = f"  {'#':>3s}  {'Image':20s}  {'IoU':>8s}  {'Dice':>8s}  {'Accuracy':>8s}  {'Conf':>6s}  {'Time':>7s}"
    report_lines.append(header)
    report_lines.append("  " + "-" * len(header.strip()))

    for r in per_image_results:
        line = (
            f"  {r['index']:3d}  {r['image_name']:20s}  "
            f"{r['metrics']['iou']:8.4f}  "
            f"{r['metrics']['dice']:8.4f}  "
            f"{r['metrics']['pixel_accuracy']:8.4f}  "
            f"{r['confidence']:6.3f}  "
            f"{r['inference_time_ms']:5.0f}ms"
        )
        report_lines.append(line)

    report_lines.extend([
        "",
        "-" * 70,
        "  OUTPUT FILES GENERATED",
        "-" * 70,
        f"  Outputs Folder:               {output_dir}",
        f"  Checkpoint Info:              checkpoint_info.json",
        f"  Evaluation Results:           test_evaluation_results.json",
        f"  Per-Image Metrics:            per_image_metrics.json",
        f"  Summary Report:               summary_report.txt",
        f"  Metrics Bar Chart:            charts/metrics_bar_chart.png",
        f"  Per-Image IoU Chart:          charts/per_image_iou_chart.png",
        f"  Confusion Matrix:             charts/confusion_matrix.png",
        f"  Comparison Images:            images/*_comparison.png ({len(test_images)} files)",
        f"  Prediction Masks:             images/*_mask.png ({len(test_images)} files)",
        f"  Overlay Images:               images/*_overlay.png ({len(test_images)} files)",
        "",
        "=" * 70,
        "  END OF REPORT",
        "=" * 70,
    ])

    report_text = "\n".join(report_lines)

    with open(output_dir / "summary_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    # ── Print Final Summary ────────────────────────────────────
    print()
    print("=" * 70)
    print("  EVALUATION RESULTS SUMMARY")
    print("=" * 70)
    print(f"  Mean IoU:          {aggregate_metrics['iou']:.4f}")
    print(f"  Rooftop IoU:       {aggregate_metrics['rooftop_iou']:.4f}")
    print(f"  Mean Dice:         {aggregate_metrics['dice']:.4f}")
    print(f"  Rooftop Dice:      {aggregate_metrics['rooftop_dice']:.4f}")
    print(f"  Pixel Accuracy:    {aggregate_metrics['pixel_accuracy']:.4f}")
    print(f"  Avg Confidence:    {avg_conf:.4f}")
    print(f"  Avg Inference:     {avg_time:.1f} ms/image")
    print("=" * 70)
    print()
    print(f"  [OK] All outputs saved to: {output_dir}")
    print(f"  [OK] Total files generated: {3 * len(test_images) + 6}")
    print()


if __name__ == "__main__":
    main()
