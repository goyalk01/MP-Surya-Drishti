"""
MP Surya-Drishti — Model Test Evaluation & Scientific Showcase Report Generator
================================================================================

Evaluates trained segmentation models on the official Massachusetts test set and produces:
  1. Primary Model Performance: Native model resolution (512x512 single pass)
  2. Native-Resolution Tiled Inference: Sliding-window (512x512 tiles, stride 256, Gaussian blend)
  3. Native-Resolution TTA: Sliding-window + Test-Time Augmentation (flips + rot90)
  4. Diagnostic Postprocessing: Sliding-window + MaskCleaner (min_region_area=10)
  5. 5-Panel Visual Debug Figures: Original, Ground Truth, Single-Pass, Tiled, Error Map
  6. Tile Grid Overlay Visualization: Illustrating native 512x512 sliding window over 1500x1500
  7. Multi-pipeline comparison charts, per-image IoU distributions, and confusion matrices
  8. Standardized JSON metrics artifacts

Usage:
    python test_evaluation_showcase.py
    python test_evaluation_showcase.py --checkpoint outputs/experiments/exp_002/checkpoints/best_loss.pth
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

from evaluation.metrics import SegmentationMetrics
from inference.inferencer import SegmentationInferencer, generate_gaussian_weight_map
from models.registry import ensure_models_registered, load_model_from_checkpoint
from postprocessing.area_estimator import AreaEstimator
from postprocessing.mask_cleaner import MaskCleaner
from postprocessing.polygon_extractor import PolygonExtractor
from preprocessing.augmentation import AugmentationPipeline
from preprocessing.dataset_loader import MassachusettsDataset
from preprocessing.tiled_dataset import generate_tile_coordinates
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


def compute_error_difference_map(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
) -> np.ndarray:
    """
    Generate an RGB error difference map.
    - True Positive (TP): Green (0, 220, 0)
    - False Positive (FP): Red (230, 40, 40)
    - False Negative (FN): Blue (40, 100, 230)
    - True Negative (TN): Dark Gray (30, 30, 30)
    """
    h, w = ground_truth.shape[:2]
    diff = np.full((h, w, 3), 30, dtype=np.uint8)

    tp = (prediction == 1) & (ground_truth == 1)
    fp = (prediction == 1) & (ground_truth == 0)
    fn = (prediction == 0) & (ground_truth == 1)

    diff[tp] = [0, 220, 0]      # Green = Correct Detection
    diff[fp] = [230, 40, 40]    # Red = False Alarm
    diff[fn] = [40, 100, 230]   # Blue = Missed Rooftop

    return diff


# ══════════════════════════════════════════════════════════════════
#  Visualization Generators
# ══════════════════════════════════════════════════════════════════

def save_5panel_debug_figure(
    original: np.ndarray,
    ground_truth: np.ndarray,
    pred_single: np.ndarray,
    pred_tiled: np.ndarray,
    error_map: np.ndarray,
    single_iou: float,
    tiled_iou: float,
    save_path: Path,
    image_name: str,
) -> None:
    """Save a comprehensive 5-panel debug comparison figure."""
    fig, axes = plt.subplots(1, 5, figsize=(25, 6))

    axes[0].imshow(original)
    axes[0].set_title("Original Satellite (1500×1500)", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(ground_truth, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Ground Truth Mask", fontsize=11, fontweight="bold")
    axes[1].axis("off")

    axes[2].imshow(pred_single, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title(f"Single-Pass 512 (IoU: {single_iou*100:.1f}%)", fontsize=11, fontweight="bold")
    axes[2].axis("off")

    axes[3].imshow(pred_tiled, cmap="gray", vmin=0, vmax=1)
    axes[3].set_title(f"Tiled Native-Res (IoU: {tiled_iou*100:.1f}%)", fontsize=11, fontweight="bold")
    axes[3].axis("off")

    axes[4].imshow(error_map)
    axes[4].set_title("Tiled Error Map\n(TP:Grn, FP:Red, FN:Blu)", fontsize=11, fontweight="bold")
    axes[4].axis("off")

    plt.suptitle(
        f"MP Surya-Drishti Diagnostic — {image_name}",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_tile_grid_visualization(
    image: np.ndarray,
    coords: list[tuple[int, int, int, int]],
    save_path: Path,
) -> None:
    """Save satellite image with overlaid tile grid illustrating native tiling."""
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)

    for i, (x1, y1, x2, y2) in enumerate(coords):
        rect = plt.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=False,
            edgecolor="#00FFCC",
            linewidth=1.2,
            linestyle="--",
            alpha=0.8,
        )
        ax.add_patch(rect)
        ax.text(
            x1 + 15,
            y1 + 35,
            f"T{i+1}",
            color="#FFFFFF",
            fontsize=9,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#000000", alpha=0.6),
        )

    ax.set_title(
        f"Sliding-Window Native Tiling Grid ({len(coords)} tiles, 512×512, stride 256)",
        fontsize=13,
        fontweight="bold",
    )
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_pipeline_comparison_chart(
    mode_a: dict[str, float],
    mode_b: dict[str, float],
    mode_c: dict[str, float],
    mode_d: dict[str, float],
    save_path: Path,
) -> None:
    """Save 4-mode comparative bar chart across pipelines."""
    labels = ["Pixel Acc", "Mean IoU", "Rooftop IoU", "Rooftop Dice"]
    metrics_a = [mode_a["pixel_accuracy"], mode_a["mean_iou"], mode_a["rooftop_iou"], mode_a["rooftop_dice"]]
    metrics_b = [mode_b["pixel_accuracy"], mode_b["mean_iou"], mode_b["rooftop_iou"], mode_b["rooftop_dice"]]
    metrics_c = [mode_c["pixel_accuracy"], mode_c["mean_iou"], mode_c["rooftop_iou"], mode_c["rooftop_dice"]]
    metrics_d = [mode_d["pixel_accuracy"], mode_d["mean_iou"], mode_d["rooftop_iou"], mode_d["rooftop_dice"]]

    x = np.arange(len(labels))
    width = 0.20

    fig, ax = plt.subplots(figsize=(13, 7))
    r1 = ax.bar(x - 1.5 * width, [v * 100 for v in metrics_a], width, label="A: 512×512 Baseline", color="#4A90E2", alpha=0.9)
    r2 = ax.bar(x - 0.5 * width, [v * 100 for v in metrics_b], width, label="B: Tiled Native (512, s256)", color="#2ECC71", alpha=0.9)
    r3 = ax.bar(x + 0.5 * width, [v * 100 for v in metrics_c], width, label="C: Tiled + TTA", color="#F39C12", alpha=0.9)
    r4 = ax.bar(x + 1.5 * width, [v * 100 for v in metrics_d], width, label="D: Tiled + Cleaner", color="#9B59B6", alpha=0.9)

    ax.set_ylabel("Score (%)", fontsize=12, fontweight="bold")
    ax.set_title("MP Surya-Drishti — Multi-Mode Evaluation Matrix", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_ylim(0, 105)

    for rects in [r1, r2, r3, r4]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.1f}%",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_per_image_iou_chart(
    per_image_results: list[dict[str, Any]],
    save_path: Path,
) -> None:
    """Save bar chart showing per-image rooftop IoU."""
    names = [r["image_name"].replace(".png", "") for r in per_image_results]
    iou_a = [r["single_metrics"]["rooftop_iou"] * 100 for r in per_image_results]
    iou_b = [r["tiled_metrics"]["rooftop_iou"] * 100 for r in per_image_results]

    x = np.arange(len(names))
    width = 0.4

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width/2, iou_a, width, label="Single-Pass Baseline", color="#4A90E2", alpha=0.85)
    ax.bar(x + width/2, iou_b, width, label="Tiled Native Resolution", color="#2ECC71", alpha=0.85)

    ax.set_ylabel("Rooftop IoU (%)", fontsize=11, fontweight="bold")
    ax.set_title("Per-Image Rooftop IoU Comparison on Test Set", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_metrics_bar_chart(
    metrics: dict[str, float],
    save_path: Path,
    model_name: str = "SegFormer-B2",
) -> None:
    """Save single-model primary metrics bar chart."""
    labels = ["Pixel Acc", "Mean IoU", "Rooftop IoU", "Rooftop Dice", "Background IoU"]
    keys = ["pixel_accuracy", "mean_iou", "rooftop_iou", "rooftop_dice", "background_iou"]
    values = [metrics.get(k, 0.0) * 100 for k in keys]
    colors = ["#2ECC71", "#3498DB", "#E74C3C", "#9B59B6", "#1ABC9C"]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, values, color=colors, width=0.55, edgecolor="#333333", linewidth=1.2)
    ax.set_ylabel("Score (%)", fontsize=12, fontweight="bold")
    ax.set_title(f"Primary Benchmark Performance — {model_name}", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.2f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_confusion_matrix_chart(
    cm: np.ndarray,
    save_path: Path,
    title_suffix: str = "Primary",
) -> None:
    """Save normalized confusion matrix heatmap."""
    cm_norm = cm.astype(np.float64) / np.maximum(cm.sum(axis=1, keepdims=True), 1e-7)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    classes = ["Background", "Rooftop"]
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        title=f"Normalized Confusion Matrix ({title_suffix})",
        ylabel="True Label",
        xlabel="Predicted Label",
    )

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm_norm[i, j] * 100
            raw = cm[i, j]
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(
                j, i, f"{val:.1f}%\n({raw:,})",
                ha="center", va="center", color=color, fontsize=10, fontweight="bold",
            )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════
#  Evaluation Showcase Runner
# ══════════════════════════════════════════════════════════════════

def run_evaluation_showcase(
    checkpoint_path: Optional[str | Path] = None,
    dataset_root: str | Path = "datasets/massachusetts",
    output_base_dir: str | Path = "outputs",
    image_size: int = 512,
    tile_stride: int = 256,
    eval_tta: bool = False,
) -> dict[str, Any]:
    """Execute evaluation across the 4 experimental modes."""
    device = get_device()
    output_dir = Path(output_base_dir)
    images_dir = output_dir / "images"
    charts_dir = output_dir / "charts"

    images_dir.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)

    # 1. Resolve Checkpoint
    if checkpoint_path is None:
        resolved_checkpoint = find_best_available_checkpoint()
    else:
        resolved_checkpoint = Path(checkpoint_path)

    logger.info("Evaluating Checkpoint: %s on device: %s", resolved_checkpoint, device)

    # 2. Discover Test Set
    img_paths, mask_paths = MassachusettsDataset.discover_pairs(
        root_dir=dataset_root,
        images_dir="test",
        masks_dir="test_labels",
    )
    logger.info("Discovered %d test image/mask pairs in %s", len(img_paths), dataset_root)

    # 3. Instantiate Inferencer
    inferencer = SegmentationInferencer(
        checkpoint_path=resolved_checkpoint,
        image_size=image_size,
        tile_stride=tile_stride,
        tile_batch_size=8,
        blend_mode="gaussian",
        device=device,
    )

    # 4. Initialize Metric Accumulators for Modes A, B, C, D
    acc_mode_a = SegmentationMetrics(num_classes=2)  # Single-Pass 512
    acc_mode_b = SegmentationMetrics(num_classes=2)  # Tiled Native
    acc_mode_c = SegmentationMetrics(num_classes=2)  # Tiled + TTA
    acc_mode_d = SegmentationMetrics(num_classes=2)  # Tiled + Cleaner

    per_image_results = []

    # 5. Execute Evaluations
    for idx, (img_p, msk_p) in enumerate(zip(img_paths, mask_paths)):
        img_name = img_p.name
        img_rgb = load_image_rgb(img_p)
        gt_binary = load_mask_binary(msk_p)

        # Mode A: Single-Pass Resize 512
        res_a = inferencer.run(img_p, tiled=False)
        gt_512 = cv2.resize(gt_binary, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
        pred_512 = cv2.resize(res_a.binary_mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
        acc_mode_a.update(pred_512, gt_512)

        # Mode B: Tiled Native Resolution (512x512, stride 256, Gaussian blend)
        res_b = inferencer.run(img_p, tiled=True, tta=False, apply_cleaner=False)
        acc_mode_b.update(res_b.binary_mask, gt_binary)

        # Mode C: Tiled + TTA (if enabled via --tta)
        if eval_tta:
            res_c = inferencer.run(img_p, tiled=True, tta=True, apply_cleaner=False)
            acc_mode_c.update(res_c.binary_mask, gt_binary)
        else:
            acc_mode_c.update(res_b.binary_mask, gt_binary)

        # Mode D: Tiled + Cleaner Diagnostic (applied directly on Mode B prediction)
        mask_d = inferencer.mask_cleaner.clean(res_b.binary_mask)
        acc_mode_d.update(mask_d, gt_binary)

        # Compute per-image metrics
        m_a = SegmentationMetrics(num_classes=2)
        m_a.update(pred_512, gt_512)
        r_a = m_a.compute()

        m_b = SegmentationMetrics(num_classes=2)
        m_b.update(res_b.binary_mask, gt_binary)
        r_b = m_b.compute()

        # Generate error map and debug visual figure
        err_map = compute_error_difference_map(gt_binary, res_b.binary_mask)
        fig_path = images_dir / f"debug_{img_name}"
        save_5panel_debug_figure(
            original=img_rgb,
            ground_truth=gt_binary,
            pred_single=res_a.binary_mask,
            pred_tiled=res_b.binary_mask,
            error_map=err_map,
            single_iou=r_a["rooftop_iou"],
            tiled_iou=r_b["rooftop_iou"],
            save_path=fig_path,
            image_name=img_name,
        )

        per_image_results.append({
            "index": idx + 1,
            "image_name": img_name,
            "single_metrics": {
                "pixel_accuracy": round(r_a["pixel_accuracy"], 4),
                "mean_iou": round(r_a["iou"], 4),
                "rooftop_iou": round(r_a["rooftop_iou"], 4),
                "rooftop_dice": round(r_a["rooftop_dice"], 4),
            },
            "tiled_metrics": {
                "pixel_accuracy": round(r_b["pixel_accuracy"], 4),
                "mean_iou": round(r_b["iou"], 4),
                "rooftop_iou": round(r_b["rooftop_iou"], 4),
                "rooftop_dice": round(r_b["rooftop_dice"], 4),
            },
        })

    # 6. Compute Aggregate Metrics
    mode_a_res = acc_mode_a.compute()
    mode_b_res = acc_mode_b.compute()
    mode_c_res = acc_mode_c.compute()
    mode_d_res = acc_mode_d.compute()

    payload_a = {
        "pixel_accuracy": round(mode_a_res["pixel_accuracy"], 4),
        "mean_iou": round(mode_a_res["iou"], 4),
        "rooftop_iou": round(mode_a_res["rooftop_iou"], 4),
        "rooftop_dice": round(mode_a_res["rooftop_dice"], 4),
        "background_iou": round(mode_a_res["iou_per_class"][0], 4) if len(mode_a_res["iou_per_class"]) > 0 else 0.0,
    }
    payload_b = {
        "pixel_accuracy": round(mode_b_res["pixel_accuracy"], 4),
        "mean_iou": round(mode_b_res["iou"], 4),
        "rooftop_iou": round(mode_b_res["rooftop_iou"], 4),
        "rooftop_dice": round(mode_b_res["rooftop_dice"], 4),
        "background_iou": round(mode_b_res["iou_per_class"][0], 4) if len(mode_b_res["iou_per_class"]) > 0 else 0.0,
    }
    payload_c = {
        "pixel_accuracy": round(mode_c_res["pixel_accuracy"], 4),
        "mean_iou": round(mode_c_res["iou"], 4),
        "rooftop_iou": round(mode_c_res["rooftop_iou"], 4),
        "rooftop_dice": round(mode_c_res["rooftop_dice"], 4),
        "background_iou": round(mode_c_res["iou_per_class"][0], 4) if len(mode_c_res["iou_per_class"]) > 0 else 0.0,
    }
    payload_d = {
        "pixel_accuracy": round(mode_d_res["pixel_accuracy"], 4),
        "mean_iou": round(mode_d_res["iou"], 4),
        "rooftop_iou": round(mode_d_res["rooftop_iou"], 4),
        "rooftop_dice": round(mode_d_res["rooftop_dice"], 4),
        "background_iou": round(mode_d_res["iou_per_class"][0], 4) if len(mode_d_res["iou_per_class"]) > 0 else 0.0,
    }

    # 7. Generate Tile Grid Visualization on First Image
    first_img = load_image_rgb(img_paths[0])
    tile_coords = generate_tile_coordinates(
        image_width=first_img.shape[1],
        image_height=first_img.shape[0],
        tile_size=image_size,
        stride=tile_stride,
    )
    save_tile_grid_visualization(first_img, tile_coords, charts_dir / "tile_grid_visualization.png")

    # 8. Generate Benchmark Charts
    save_metrics_bar_chart(payload_a, charts_dir / "metrics_bar_chart.png")
    save_pipeline_comparison_chart(payload_a, payload_b, payload_c, payload_d, charts_dir / "pipeline_comparison_chart.png")
    save_per_image_iou_chart(per_image_results, charts_dir / "per_image_iou_chart.png")
    save_confusion_matrix_chart(acc_mode_a.confusion_matrix, charts_dir / "confusion_matrix.png", title_suffix="Mode A: 512×512")

    # 9. Save JSON Artifacts
    full_report = {
        "report_generated_at": datetime.now().isoformat(),
        "checkpoint": str(resolved_checkpoint),
        "dataset": "Massachusetts Buildings Dataset",
        "split": "test",
        "num_samples": len(img_paths),
        "evaluation_modes": {
            "mode_a_single_pass_512": payload_a,
            "mode_b_tiled_native_resolution": payload_b,
            "mode_c_tiled_tta": payload_c,
            "mode_d_tiled_cleaner": payload_d,
        },
        "per_image_results": per_image_results,
    }

    with open(output_dir / "test_evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    with open(output_dir / "per_image_metrics.json", "w", encoding="utf-8") as f:
        json.dump(per_image_results, f, indent=2)

    # 10. Write Text Summary Report
    report_lines = [
        "=" * 68,
        "  MP SURYA-DRISHTI — TEST EVALUATION & BENCHMARK REPORT",
        "=" * 68,
        f"  Report Date : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"  Checkpoint  : {resolved_checkpoint.name}",
        f"  Test Images : {len(img_paths)} satellite scenes (1500×1500 px)",
        "",
        "=" * 68,
        "  EVALUATION MATRIX COMPARISON",
        "=" * 68,
        f"  {'Evaluation Mode':<30s} | {'Pixel Acc':>9s} | {'Mean IoU':>8s} | {'Roof IoU':>8s} | {'Roof Dice':>9s} |",
        "  " + "-" * 66,
        f"  {'Mode A: 512×512 Baseline':<30s} | {payload_a['pixel_accuracy']*100:8.2f}% | {payload_a['mean_iou']*100:7.2f}% | {payload_a['rooftop_iou']*100:7.2f}% | {payload_a['rooftop_dice']*100:8.2f}% |",
        f"  {'Mode B: Tiled Native (s256)':<30s} | {payload_b['pixel_accuracy']*100:8.2f}% | {payload_b['mean_iou']*100:7.2f}% | {payload_b['rooftop_iou']*100:7.2f}% | {payload_b['rooftop_dice']*100:8.2f}% |",
        f"  {'Mode C: Tiled + TTA':<30s} | {payload_c['pixel_accuracy']*100:8.2f}% | {payload_c['mean_iou']*100:7.2f}% | {payload_c['rooftop_iou']*100:7.2f}% | {payload_c['rooftop_dice']*100:8.2f}% |",
        f"  {'Mode D: Tiled + Cleaner':<30s} | {payload_d['pixel_accuracy']*100:8.2f}% | {payload_d['mean_iou']*100:7.2f}% | {payload_d['rooftop_iou']*100:7.2f}% | {payload_d['rooftop_dice']*100:8.2f}% |",
        "  " + "-" * 66,
        "",
        "=" * 68,
        "  GENERATED OUTPUT ARTIFACTS",
        "=" * 68,
        f"  Output Directory : {output_dir}",
        f"  Results JSON     : test_evaluation_results.json",
        f"  Per-Image JSON   : per_image_metrics.json",
        f"  Charts Folder    : charts/",
        f"  Debug Figures    : images/ (5-panel comparisons)",
        "=" * 68,
    ]

    with open(output_dir / "summary_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print("\n" + "\n".join(report_lines) + "\n")
    return full_report


def main() -> None:
    """CLI Entry Point."""
    parser = argparse.ArgumentParser(description="Run evaluation showcase.")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--dataset-root", default="datasets/massachusetts")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--tta", action="store_true", default=False, help="Enable Test-Time Augmentation")

    args = parser.parse_args()

    run_evaluation_showcase(
        checkpoint_path=args.checkpoint,
        dataset_root=args.dataset_root,
        output_base_dir=args.output_dir,
        image_size=args.image_size,
        tile_stride=args.stride,
        eval_tta=args.tta,
    )


if __name__ == "__main__":
    main()
