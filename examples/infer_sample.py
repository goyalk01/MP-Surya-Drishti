"""
Example script: Running MP Surya-Drishti inference programmatically in Python.

Usage:
    python examples/infer_sample.py --image path/to/sample.jpg --checkpoint models/checkpoints/best_iou.pth
"""

from __future__ import annotations

import argparse
from pathlib import Path

from inference.inferencer import SegmentationInferencer


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample programatic inference script.")
    parser.add_argument("--image", required=True, help="Path to input image.")
    parser.add_argument("--checkpoint", default="models/checkpoints/best_iou.pth", help="Path to checkpoint.")
    args = parser.parse_args()

    # Initialize inferencer engine
    inferencer = SegmentationInferencer(
        checkpoint_path=args.checkpoint,
        image_size=512,
        gsd=1.0,
    )

    # Run inference
    result = inferencer.run(args.image)

    # Generate prediction report
    report = result.to_prediction_report()

    print("\n--- PREDICTION REPORT ---")
    for key, value in report.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
