"""
Device detection and GPU optimization utilities for PyTorch.

Auto-detects the best available compute device (CUDA, MPS, CPU),
enables CUDNN benchmarking for optimal convolution throughput,
and logs diagnostic hardware info.
"""

from __future__ import annotations

import logging
import platform

import torch

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """
    Auto-detect the best available compute device and configure hardware flags.

    Priority: CUDA > MPS (Apple Silicon) > CPU.

    Returns:
        torch.device for the best available backend.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        logger.info("Using device: CUDA (%s) | CUDNN benchmark enabled", torch.cuda.get_device_name(0))
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using device: MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        logger.info("Using device: CPU")

    return device


def print_device_info() -> dict[str, str]:
    """
    Print and return detailed device information.

    Returns:
        Dictionary with device details.
    """
    info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_available": str(torch.cuda.is_available()),
    }

    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda or "unknown"
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_total"] = f"{torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB"
        info["gpu_count"] = str(torch.cuda.device_count())

    if hasattr(torch.backends, "mps"):
        info["mps_available"] = str(torch.backends.mps.is_available())

    for key, value in info.items():
        logger.info("  %s: %s", key, value)

    return info
