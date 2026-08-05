"""
MP Surya-Drishti — Segmentation Framework CLI Entry Point.

Model-agnostic entry point that uses the model registry to create
and load any registered segmentation model.

Commands:
    verify-dataset  Verify official dataset splits and pairing integrity
    train           Train model on official partitions with TensorBoard & experiment tracking
    infer           Run inference on a single image and generate prediction_report.json
    evaluate        Evaluate model performance on the official test set using trained weights
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

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


def find_default_checkpoint() -> Path:
    """
    Find the latest trained best_iou.pth checkpoint across experiment directories.

    Returns:
        Path to best_iou.pth file.

    Raises:
        FileNotFoundError: If no trained checkpoint is found.
    """
    candidates = [
        Path("models/checkpoints/best_iou.pth"),
        Path("outputs/checkpoints/best_iou.pth"),
    ]

    # Look in experiments
    exp_dir = Path("outputs/experiments")
    if exp_dir.exists():
        for exp in sorted(exp_dir.glob("exp_*"), reverse=True):
            ckpt = exp / "checkpoints" / "best_iou.pth"
            if ckpt.exists():
                return ckpt

    for cand in candidates:
        if cand.exists():
            return cand

    raise FileNotFoundError(
        "No trained checkpoint ('best_iou.pth') found. "
        "Please run training first using: python main.py train"
    )


def cmd_train(args: argparse.Namespace) -> None:
    """Execute training pipeline using official dataset splits."""
    from models.registry import create_model, ensure_models_registered
    from preprocessing.augmentation import AugmentationPipeline
    from preprocessing.dataset_loader import MassachusettsDataset
    from preprocessing.splitter import DatasetSplitter
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

    # Initialize experiment manager
    exp_manager = ExperimentManager(
        base_dir=args.experiment_dir,
        exp_name=args.exp_name,
    )

    logger = setup_logger(
        log_level="DEBUG" if args.verbose else "INFO",
        log_file=exp_manager.logs_dir / "training.log",
    )

    logger.info("=" * 60)
    logger.info("MP Surya-Drishti — Segmentation Framework Training")
    logger.info("Experiment: %s | Output Path: %s", exp_manager.exp_name, exp_manager.exp_dir)
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

    augmentation = AugmentationPipeline(image_size=model_cfg["image_size"])
    train_transform = augmentation.get_train_transform()
    val_transform = augmentation.get_val_transform()

    train_dataset = MassachusettsDataset(
        image_paths=train_img_paths,
        mask_paths=train_mask_paths,
        image_size=model_cfg["image_size"],
        transform=train_transform,
        mask_building_value=dataset_cfg.get("mask_building_value", 255),
    )

    val_dataset = MassachusettsDataset(
        image_paths=val_img_paths,
        mask_paths=val_mask_paths,
        image_size=model_cfg["image_size"],
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
    """Run inference on a single image and generate prediction_report.json."""
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
    image_size = model_cfg.get("image_size", 512)

    # Determine checkpoint path
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        checkpoint_path = find_default_checkpoint()
        logger.info("Using auto-detected checkpoint: %s", checkpoint_path)

    inferencer = SegmentationInferencer(
        checkpoint_path=checkpoint_path,
        image_size=image_size,
        gsd=gsd,
    )

    result = inferencer.run(args.image)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_image(result.binary_mask * 255, output_dir / "mask.png")
    save_image(result.overlay_image, output_dir / "overlay.png")
    save_json(result.to_dict(), output_dir / "result.json")

    report = result.to_prediction_report()
    save_json(report, output_dir / "prediction_report.json")

    visualizer = SegmentationVisualizer(output_dir=output_dir)
    visualizer.plot_overlay(
        result.original_image,
        result.binary_mask,
        filename="visualization.png",
    )

    print("\n" + "=" * 50)
    print("  PREDICTION REPORT")
    print("=" * 50)
    print(f"  Model:                 {report['model']}")
    print(f"  Version:               {report['version']}")
    print(f"  Roof Area (Pixels):    {report['roof_area_pixels']} px")
    print(f"  Roof Area (%):         {report['roof_area_percent']}%")
    print(f"  Usable Area (%):       {report['usable_area_percent']}%")
    print(f"  Confidence:            {report['confidence']}")
    print(f"  Area m² (Estimated):   {report['rooftop_area_m2_estimate']} m²")
    print(f"  Is Estimated:          {report['is_estimated']}")
    print(f"  Polygons Found:        {report['polygons_found']}")
    print(f"  Processing Time:       {report['processing_time_ms']} ms")
    print(f"  Report saved to:       {output_dir / 'prediction_report.json'}")
    print("=" * 50 + "\n")


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Evaluate model on the official test set using trained weights."""
    import torch

    from evaluation.metrics import SegmentationMetrics
    from models.registry import ensure_models_registered, load_model_from_checkpoint
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

    ensure_models_registered()

    # Determine checkpoint path
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        checkpoint_path = find_default_checkpoint()
        logger.info("Using auto-detected checkpoint: %s", checkpoint_path)

    device = get_device()
    logger.info("Loading trained weights from checkpoint: %s", checkpoint_path)
    model = load_model_from_checkpoint(checkpoint_path, device=device)
    model.eval()

    test_img_paths, test_mask_paths = MassachusettsDataset.discover_pairs(
        root_dir=dataset_cfg["root_dir"],
        images_dir=dataset_cfg.get("test_images_dir", "test"),
        masks_dir=dataset_cfg.get("test_masks_dir", "test_labels"),
        extensions=dataset_cfg.get("image_extensions"),
    )

    augmentation = AugmentationPipeline(image_size=model_cfg["image_size"])
    test_dataset = MassachusettsDataset(
        image_paths=test_img_paths,
        mask_paths=test_mask_paths,
        image_size=model_cfg["image_size"],
        transform=augmentation.get_val_transform(),
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=4, shuffle=False, num_workers=2
    )

    metrics = SegmentationMetrics(num_classes=getattr(model, "num_labels", 2))

    with torch.no_grad():
        for batch in test_loader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            prediction = model.predict(pixel_values)
            metrics.update(prediction["binary_mask"], labels)

    results = metrics.compute()
    output_dir = Path(args.output_dir)
    save_json(results, output_dir / "evaluation_results.json")

    print("\n" + "=" * 50)
    print("  EVALUATION RESULTS")
    print("=" * 50)
    print(f"  Checkpoint Loaded: {checkpoint_path}")
    print(f"  Test Samples:      {len(test_dataset)}")
    print(f"  Mean IoU:          {results['iou']:.4f}")
    print(f"  Rooftop IoU:       {results['rooftop_iou']:.4f}")
    print(f"  Mean Dice:         {results['dice']:.4f}")
    print(f"  Rooftop Dice:      {results['rooftop_dice']:.4f}")
    print(f"  Pixel Accuracy:    {results['pixel_accuracy']:.4f}")
    print(f"  Results saved:     {output_dir / 'evaluation_results.json'}")
    print("=" * 50 + "\n")


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
    train_parser.add_argument("--exp-name", default=None, help="Custom exp name (e.g. exp_001)")

    # Infer
    infer_parser = subparsers.add_parser("infer", help="Run inference")
    infer_parser.add_argument("--image", required=True, help="Input image path")
    infer_parser.add_argument("--checkpoint", default="models/checkpoints/best_iou.pth")
    infer_parser.add_argument("--output-dir", default="outputs/predictions")
    infer_parser.add_argument("--model-config", default="configs/model_config.yaml")
    infer_parser.add_argument("--dataset-config", default="configs/dataset_config.yaml")

    # Evaluate
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate model")
    eval_parser.add_argument("--checkpoint", default="models/checkpoints/best_iou.pth")
    eval_parser.add_argument("--output-dir", default="outputs/reports")
    eval_parser.add_argument("--model-config", default="configs/model_config.yaml")
    eval_parser.add_argument("--dataset-config", default="configs/dataset_config.yaml")

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
        "verify-dataset": cmd_verify_dataset,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
