"""
MP Surya-Drishti — Segmentation Framework CLI Entry Point.

Model-agnostic entry point that uses the model registry to create
and load any registered segmentation model.

Commands:
    verify-dataset     Verify official dataset splits and pairing integrity
    measure-imbalance  Measure exact ground-truth class distribution on training split
    train              Train model with native-resolution tiling or whole-image strategy
    infer              Run inference with optional sliding-window tiling, Gaussian blend, and TTA
    evaluate           Evaluate model performance across single-pass and tiled modes
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from utils.checkpoint_manager import find_best_available_checkpoint
from utils.config_validator import (
    validate_dataset_config,
    validate_model_config,
    validate_training_config,
)
from utils.logger import setup_logger


def load_config(config_path: str) -> dict:
    """Load and parse a YAML configuration file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def cmd_train(args: argparse.Namespace) -> None:
    """Execute training pipeline using official dataset splits and configurable strategy."""
    from models.registry import create_model, ensure_models_registered
    from preprocessing.augmentation import AugmentationPipeline
    from preprocessing.dataset_loader import MassachusettsDataset
    from preprocessing.splitter import DatasetSplitter
    from preprocessing.tiled_dataset import TiledMassachusettsDataset
    from training.trainer import SegmentationTrainer
    from utils.device_utils import print_device_info
    from utils.experiment_manager import ExperimentManager

    # Load & Validate Configs
    model_config = load_config(args.model_config)
    dataset_config = load_config(args.dataset_config)
    training_config = load_config(args.training_config)

    model_cfg = validate_model_config(model_config)
    dataset_cfg = validate_dataset_config(dataset_config)
    training_cfg = validate_training_config(training_config)

    # CLI Overrides
    if args.loss is not None:
        training_config["training"]["loss"]["name"] = args.loss
        training_cfg["loss"]["name"] = args.loss
    if args.epochs is not None:
        training_config["training"]["epochs"] = args.epochs
        training_cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        training_config["training"]["batch_size"] = args.batch_size
        training_cfg["batch_size"] = args.batch_size

    use_tiling = args.tiled if args.tiled is not None else (training_cfg.get("strategy") == "tiled")
    tile_size = args.tile_size or training_cfg.get("tiling", {}).get("tile_size", model_cfg.get("image_size", 512))
    tile_stride = args.stride or training_cfg.get("tiling", {}).get("stride", 256)

    training_config["training"]["strategy"] = "tiled" if use_tiling else "full_image"
    training_config["training"]["tiling"] = {"tile_size": tile_size, "stride": tile_stride}

    # Initialize experiment manager
    exp_name = args.exp_name or ("exp_003" if use_tiling else None)
    exp_manager = ExperimentManager(
        base_dir=args.experiment_dir,
        exp_name=exp_name,
    )

    logger = setup_logger(
        log_level="DEBUG" if args.verbose else "INFO",
        log_file=exp_manager.logs_dir / "training.log",
    )

    logger.info("=" * 60)
    logger.info("MP Surya-Drishti — Segmentation Framework Training")
    logger.info(
        "Experiment: %s | Strategy: %s | Loss: %s | Output: %s",
        exp_manager.exp_name,
        "Native-Resolution Tiling" if use_tiling else "Whole-Image Resize",
        training_config["training"]["loss"]["name"],
        exp_manager.exp_dir,
    )
    logger.info("=" * 60)

    print_device_info()

    # Save experiment config snapshot
    exp_manager.save_config_snapshot(model_config, dataset_config, training_config)

    ensure_models_registered()

    # Discover official splits: train, val
    logger.info("Loading official dataset splits from: %s", dataset_cfg["root_dir"])

    train_img_paths, train_mask_paths = MassachusettsDataset.discover_pairs(
        root_dir=dataset_cfg["root_dir"],
        images_dir=dataset_cfg.get("train_images_dir", "train"),
        masks_dir=dataset_cfg.get("train_masks_dir", "train_labels"),
        extensions=dataset_cfg.get("image_extensions"),
    )

    val_img_paths, val_mask_paths = MassachusettsDataset.discover_pairs(
        root_dir=dataset_cfg["root_dir"],
        images_dir=dataset_cfg.get("val_images_dir", "val"),
        masks_dir=dataset_cfg.get("val_masks_dir", "val_labels"),
        extensions=dataset_cfg.get("image_extensions"),
    )

    augmentation = AugmentationPipeline(image_size=tile_size)
    train_transform = augmentation.get_train_transform()
    val_transform = augmentation.get_val_transform()

    if use_tiling:
        logger.info("Building Native-Resolution Tiled Datasets (tile_size=%d, stride=%d)", tile_size, tile_stride)
        train_dataset = TiledMassachusettsDataset(
            image_paths=train_img_paths,
            mask_paths=train_mask_paths,
            tile_size=tile_size,
            stride=tile_stride,
            transform=train_transform,
            mask_building_value=dataset_cfg.get("mask_building_value", 255),
        )
        val_dataset = TiledMassachusettsDataset(
            image_paths=val_img_paths,
            mask_paths=val_mask_paths,
            tile_size=tile_size,
            stride=tile_stride,
            transform=val_transform,
            mask_building_value=dataset_cfg.get("mask_building_value", 255),
        )
    else:
        logger.info("Building Standard Resized Massachusetts Datasets (image_size=%d)", tile_size)
        train_dataset = MassachusettsDataset(
            image_paths=train_img_paths,
            mask_paths=train_mask_paths,
            image_size=tile_size,
            transform=train_transform,
            mask_building_value=dataset_cfg.get("mask_building_value", 255),
        )
        val_dataset = MassachusettsDataset(
            image_paths=val_img_paths,
            mask_paths=val_mask_paths,
            image_size=tile_size,
            transform=val_transform,
            mask_building_value=dataset_cfg.get("mask_building_value", 255),
        )

    loaders = DatasetSplitter.create_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=training_cfg.get("batch_size", 8),
        num_workers=training_cfg.get("num_workers", 2),
        pin_memory=training_cfg.get("pin_memory", True),
    )

    # Create Model & Trainer
    model = create_model(model_cfg)

    trainer = SegmentationTrainer(
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        config=training_config,
        checkpoint_dir=exp_manager.checkpoints_dir,
        log_dir=exp_manager.logs_dir,
        output_dir=exp_manager.exp_dir,
    )

    result = trainer.train(resume_from=training_cfg.get("resume_from"))

    # Save metrics history
    exp_manager.save_metrics_history(result["history"])

    # Generate training curves plot in plots_dir
    try:
        from evaluation.visualizer import SegmentationVisualizer
        visualizer = SegmentationVisualizer(output_dir=exp_manager.plots_dir)
        visualizer.plot_training_history(result["history"], filename="training_curves.png")
    except Exception as e:
        logger.warning("Could not save training plots: %s", e)

    logger.info("=" * 60)
    logger.info(
        "Training complete! Best IoU: %.4f | Best Loss: %.4f | Artifacts saved to: %s",
        result["best_iou"],
        result["best_loss"],
        exp_manager.exp_dir,
    )
    logger.info("=" * 60)


def cmd_infer(args: argparse.Namespace) -> None:
    """Run inference on a single image with optional sliding-window tiling and TTA."""
    from evaluation.visualizer import SegmentationVisualizer
    from inference.inferencer import SegmentationInferencer
    from utils.io_utils import save_image, save_json

    logger = setup_logger(log_level="DEBUG" if args.verbose else "INFO")

    logger.info("=" * 60)
    logger.info("MP Surya-Drishti — Segmentation Framework Inference")
    logger.info("=" * 60)

    dataset_config = load_config(args.dataset_config)
    dataset_cfg = validate_dataset_config(dataset_config)
    gsd = dataset_cfg.get("gsd_metres_per_pixel", 1.0)

    model_config = load_config(args.model_config)
    model_cfg = validate_model_config(model_config)
    image_size = args.tile_size or model_cfg.get("image_size", 512)

    # Determine checkpoint path
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else find_best_available_checkpoint()
    if not checkpoint_path.exists():
        checkpoint_path = find_best_available_checkpoint()
    logger.info("Using checkpoint: %s", checkpoint_path)

    inferencer = SegmentationInferencer(
        checkpoint_path=checkpoint_path,
        image_size=image_size,
        tile_stride=args.stride or 256,
        tile_batch_size=args.batch_size or 8,
        blend_mode=args.blend_mode or "gaussian",
        gsd=gsd,
        apply_cleaner=args.cleaner,
    )

    use_tiled = not args.no_tiled

    result = inferencer.run(
        image_path=args.image,
        tiled=use_tiled,
        tta=args.tta,
        stride=args.stride,
        batch_size=args.batch_size,
        blend_mode=args.blend_mode,
        apply_cleaner=args.cleaner,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_image(result.binary_mask * 255, output_dir / "mask.png")
    save_image(result.overlay_image, output_dir / "overlay.png")
    save_json(result.to_dict(), output_dir / "prediction_report.json")

    # Generate visual overlay plot
    vis = SegmentationVisualizer(output_dir=output_dir)
    vis.plot_overlay(result.original_image, result.binary_mask, filename="visualization.png")

    logger.info("Outputs saved to: %s", output_dir)


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Evaluate model performance on the official test set using trained weights."""
    from evaluation.metrics import SegmentationMetrics
    from inference.inferencer import SegmentationInferencer
    from preprocessing.augmentation import AugmentationPipeline
    from preprocessing.dataset_loader import MassachusettsDataset
    from utils.device_utils import get_device
    from utils.io_utils import save_json

    logger = setup_logger(log_level="DEBUG" if args.verbose else "INFO")

    logger.info("=" * 60)
    logger.info("MP Surya-Drishti — Segmentation Framework Evaluation")
    logger.info("=" * 60)

    model_config = load_config(args.model_config)
    dataset_config = load_config(args.dataset_config)

    model_cfg = validate_model_config(model_config)
    dataset_cfg = validate_dataset_config(dataset_config)

    # Determine checkpoint path
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else find_best_available_checkpoint()
    if not checkpoint_path.exists():
        checkpoint_path = find_best_available_checkpoint()
    logger.info("Selected checkpoint: %s", checkpoint_path)

    test_img_paths, test_mask_paths = MassachusettsDataset.discover_pairs(
        root_dir=dataset_cfg["root_dir"],
        images_dir=dataset_cfg.get("test_images_dir", "test"),
        masks_dir=dataset_cfg.get("test_masks_dir", "test_labels"),
        extensions=dataset_cfg.get("image_extensions"),
    )

    use_tiled = args.tiled
    inferencer = SegmentationInferencer(
        checkpoint_path=checkpoint_path,
        image_size=args.tile_size or model_cfg.get("image_size", 512),
        tile_stride=args.stride or 256,
        tile_batch_size=args.batch_size or 8,
        blend_mode=args.blend_mode or "gaussian",
        apply_cleaner=args.cleaner,
    )

    metrics = SegmentationMetrics(num_classes=2)

    for img_p, msk_p in zip(test_img_paths, test_mask_paths):
        res = inferencer.run(
            img_p,
            tiled=use_tiled,
            tta=args.tta,
            stride=args.stride,
            batch_size=args.batch_size,
            blend_mode=args.blend_mode,
            apply_cleaner=args.cleaner,
        )
        gt_mask = cv2.imread(str(msk_p), cv2.IMREAD_GRAYSCALE)
        gt_binary = (gt_mask >= dataset_cfg.get("mask_building_value", 255) // 2).astype(np.uint8)

        # Compare at result mask resolution
        if res.binary_mask.shape != gt_binary.shape:
            gt_eval = cv2.resize(gt_binary, (res.binary_mask.shape[1], res.binary_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
        else:
            gt_eval = gt_binary

        metrics.update(res.binary_mask, gt_eval)

    results = metrics.compute()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read checkpoint metadata
    import torch
    ckpt_raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    epoch = ckpt_raw.get("epoch", -1)

    eval_payload = {
        "checkpoint": str(checkpoint_path),
        "epoch": epoch,
        "dataset": "Massachusetts Buildings Dataset",
        "split": "test",
        "num_samples": len(test_img_paths),
        "strategy": "tiled_native_resolution" if use_tiled else "single_pass_resize",
        "tta": args.tta,
        "cleaner": args.cleaner,
        "primary_evaluation": {
            "pixel_accuracy": round(results["pixel_accuracy"], 4),
            "mean_iou": round(results["iou"], 4),
            "rooftop_iou": round(results["rooftop_iou"], 4),
            "rooftop_dice": round(results["rooftop_dice"], 4),
            "background_iou": round(results["iou_per_class"][0], 4) if len(results["iou_per_class"]) > 0 else 0.0,
        },
    }

    save_json(eval_payload, output_dir / "evaluation_results.json")

    print("\n" + "=" * 60)
    print("  MP SURYA-DRISHTI — EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Checkpoint Loaded : {checkpoint_path}")
    print(f"  Epoch             : {epoch}")
    print(f"  Test Samples      : {len(test_img_paths)}")
    print(f"  Evaluation Mode   : {'Native Tiled (1500x1500)' if use_tiled else 'Single Pass (512x512)'}")
    print(f"  TTA Enabled       : {args.tta}")
    print(f"  Cleaner Applied   : {args.cleaner}")
    print("-" * 60)
    print(f"  Pixel Accuracy    : {results['pixel_accuracy'] * 100:.2f}%")
    print(f"  Mean IoU          : {results['iou'] * 100:.2f}%")
    print(f"  Rooftop IoU       : {results['rooftop_iou'] * 100:.2f}%")
    print(f"  Rooftop Dice      : {results['rooftop_dice'] * 100:.2f}%")
    print("-" * 60)
    print(f"  Results saved to  : {output_dir / 'evaluation_results.json'}")
    print("=" * 60 + "\n")


def cmd_measure_imbalance(args: argparse.Namespace) -> None:
    """Measure class distribution on training split."""
    from utils.class_imbalance import compute_training_class_imbalance

    dataset_config = load_config(args.dataset_config)
    dataset_cfg = validate_dataset_config(dataset_config)

    stats = compute_training_class_imbalance(
        root_dir=dataset_cfg["root_dir"],
        train_images_dir=dataset_cfg.get("train_images_dir", "train"),
        train_masks_dir=dataset_cfg.get("train_masks_dir", "train_labels"),
        mask_building_value=dataset_cfg.get("mask_building_value", 255),
    )

    print("\n" + "=" * 60)
    print("  TRAINING SET CLASS DISTRIBUTION REPORT")
    print("=" * 60)
    print(f"  Images Analyzed    : {stats['total_images']}")
    print(f"  Total Pixels       : {stats['total_pixels']:,}")
    print(f"  Background Pixels  : {stats['background_pixels']:,} ({stats['background_percentage']}%)")
    print(f"  Rooftop Pixels     : {stats['rooftop_pixels']:,} ({stats['rooftop_percentage']}%)")
    print("-" * 60)
    print(f"  Recommended pos_weight  : {stats['pos_weight']}")
    print(f"  Recommended focal_alpha : {stats['focal_alpha']}")
    print("=" * 60 + "\n")


def cmd_verify_dataset(args: argparse.Namespace) -> None:
    """Verify official dataset splits and pairing integrity."""
    from preprocessing.dataset_loader import MassachusettsDataset

    logger = setup_logger(log_level="INFO")

    dataset_config = load_config(args.dataset_config)
    dataset_cfg = validate_dataset_config(dataset_config)

    logger.info("Verifying dataset splits at: %s", dataset_cfg["root_dir"])

    split_definitions = [
        ("Training", dataset_cfg.get("train_images_dir", "train"), dataset_cfg.get("train_masks_dir", "train_labels")),
        ("Validation", dataset_cfg.get("val_images_dir", "val"), dataset_cfg.get("val_masks_dir", "val_labels")),
        ("Test", dataset_cfg.get("test_images_dir", "test"), dataset_cfg.get("test_masks_dir", "test_labels")),
    ]

    all_valid = True
    split_summaries = []

    for name, img_dir, msk_dir in split_definitions:
        try:
            img_paths, msk_paths = MassachusettsDataset.discover_pairs(
                root_dir=dataset_cfg["root_dir"],
                images_dir=img_dir,
                masks_dir=msk_dir,
                extensions=dataset_cfg.get("image_extensions"),
            )
            dataset_inst = MassachusettsDataset(image_paths=img_paths, mask_paths=msk_paths)
            val_res = dataset_inst.validate()
            is_valid = val_res["is_valid"]
            if not is_valid:
                all_valid = False

            split_summaries.append({
                "name": name,
                "img_count": len(img_paths),
                "msk_count": len(msk_paths),
                "status": "✓ PASSED" if is_valid else "✗ FAILED",
            })
        except FileNotFoundError as e:
            all_valid = False
            split_summaries.append({
                "name": name,
                "img_count": 0,
                "msk_count": 0,
                "status": f"✗ NOT FOUND ({e.args[0]})",
            })

    print("\n" + "=" * 55)
    print("  MASSACHUSETTS DATASET VERIFICATION REPORT")
    print("=" * 55)

    for summary in split_summaries:
        print(f"  {summary['name']} Images:     {summary['img_count']}")
        print(f"  {summary['name']} Masks:      {summary['msk_count']}")
        print(f"  {summary['name']} Pairing:    {summary['status']}")
        print("-" * 55)

    overall_status = "✓ VALID DATASET" if all_valid else "✗ INVALID DATASET"
    print(f"  OVERALL STATUS:       {overall_status}")
    print("=" * 55 + "\n")

    if not all_valid:
        sys.exit(1)


def main() -> None:
    """CLI Entry Point."""
    parser = argparse.ArgumentParser(
        prog="MP Surya-Drishti — Segmentation Framework",
        description="Model-agnostic segmentation framework for solar advisory.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging.")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train
    train_parser = subparsers.add_parser("train", help="Train model")
    train_parser.add_argument("--model-config", default="configs/model_config.yaml")
    train_parser.add_argument("--dataset-config", default="configs/dataset_config.yaml")
    train_parser.add_argument("--training-config", default="configs/training_config.yaml")
    train_parser.add_argument("--experiment-dir", default="outputs/experiments")
    train_parser.add_argument("--exp-name", default=None, help="Custom exp name (e.g. exp_003)")
    train_parser.add_argument("--tiled", action="store_true", default=None, help="Enable native-resolution patch training")
    train_parser.add_argument("--no-tiled", dest="tiled", action="store_false", help="Disable tiling (whole image resize)")
    train_parser.add_argument("--tile-size", type=int, default=None, help="Patch size (default 512)")
    train_parser.add_argument("--stride", type=int, default=None, help="Tile stride (default 256)")
    train_parser.add_argument("--loss", choices=["focal_dice", "ce_dice", "bce_dice"], default=None, help="Loss function")
    train_parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    train_parser.add_argument("--batch-size", type=int, default=None, help="Batch size")

    # Infer
    infer_parser = subparsers.add_parser("infer", help="Run inference")
    infer_parser.add_argument("--image", required=True, help="Input image path")
    infer_parser.add_argument("--checkpoint", default=None, help="Path to checkpoint (auto-selects best if omitted)")
    infer_parser.add_argument("--output-dir", default="outputs/predictions")
    infer_parser.add_argument("--model-config", default="configs/model_config.yaml")
    infer_parser.add_argument("--dataset-config", default="configs/dataset_config.yaml")
    infer_parser.add_argument("--tiled", action="store_true", default=True, help="Use native-resolution sliding-window inference")
    infer_parser.add_argument("--no-tiled", dest="tiled", action="store_false", help="Use single-pass resize inference")
    infer_parser.add_argument("--tta", action="store_true", default=False, help="Enable Test-Time Augmentation")
    infer_parser.add_argument("--cleaner", action="store_true", default=False, help="Apply MaskCleaner postprocessing")
    infer_parser.add_argument("--tile-size", type=int, default=512, help="Tile size (default 512)")
    infer_parser.add_argument("--stride", type=int, default=256, help="Tile stride (default 256)")
    infer_parser.add_argument("--batch-size", type=int, default=8, help="Batch size for tiles")
    infer_parser.add_argument("--blend-mode", choices=["gaussian", "uniform"], default="gaussian")

    # Evaluate
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate model")
    eval_parser.add_argument("--checkpoint", default=None, help="Path to checkpoint (auto-selects best if omitted)")
    eval_parser.add_argument("--output-dir", default="outputs/reports")
    eval_parser.add_argument("--model-config", default="configs/model_config.yaml")
    eval_parser.add_argument("--dataset-config", default="configs/dataset_config.yaml")
    eval_parser.add_argument("--tiled", action="store_true", default=False, help="Evaluate using native-resolution sliding window")
    eval_parser.add_argument("--tta", action="store_true", default=False, help="Enable Test-Time Augmentation")
    eval_parser.add_argument("--cleaner", action="store_true", default=False, help="Apply MaskCleaner diagnostic")
    eval_parser.add_argument("--tile-size", type=int, default=512)
    eval_parser.add_argument("--stride", type=int, default=256)
    eval_parser.add_argument("--batch-size", type=int, default=8)
    eval_parser.add_argument("--blend-mode", choices=["gaussian", "uniform"], default="gaussian")

    # Measure Imbalance
    imb_parser = subparsers.add_parser("measure-imbalance", help="Measure training set class distribution")
    imb_parser.add_argument("--dataset-config", default="configs/dataset_config.yaml")

    # Verify
    verify_parser = subparsers.add_parser("verify-dataset", help="Verify dataset")
    verify_parser.add_argument("--dataset-config", default="configs/dataset_config.yaml")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    commands = {
        "train": cmd_train,
        "infer": cmd_infer,
        "evaluate": cmd_evaluate,
        "measure-imbalance": cmd_measure_imbalance,
        "verify-dataset": cmd_verify_dataset,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
