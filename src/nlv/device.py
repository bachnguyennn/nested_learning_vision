"""
device.py — resolve training device (ported from nested_learning).
"""
from __future__ import annotations
import torch


def resolve_device(device_str: str) -> torch.device:
    normalized = str(device_str).strip().lower()
    if normalized.startswith("cuda"):
        if not torch.cuda.is_available():
            return torch.device("cpu")
        parts = normalized.split(":")
        idx = int(parts[1]) if len(parts) > 1 else 0
        idx = min(idx, max(torch.cuda.device_count() - 1, 0))
        return torch.device(f"cuda:{idx}")
    if normalized.startswith("mps"):
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            return torch.device("cpu")
        return torch.device("mps")
    return torch.device(device_str)


def auto_device() -> torch.device:
    """Pick the best available device automatically."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
