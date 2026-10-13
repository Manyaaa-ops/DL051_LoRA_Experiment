"""Shared utilities: seeding, device selection, memory measurement, CSV I/O."""

from __future__ import annotations

import csv
import json
import os
import random
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml

__all__ = [
    "set_seed",
    "resolve_device",
    "PeakMemoryMeter",
    "append_csv_row",
    "read_csv",
    "save_json",
    "load_config",
    "Timer",
]


def set_seed(seed: int = 42) -> None:
    """Seed python, numpy and torch (CPU + CUDA/MPS) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str = "auto") -> torch.device:
    """Pick the compute device.

    ``auto`` prefers CUDA (the assignment's target is a Colab T4), then
    Apple-silicon MPS, then CPU. Every recorded metric states the device it
    was measured on, so cross-device comparisons are never mixed silently.
    """
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


class PeakMemoryMeter:
    """Peak GPU memory via PyTorch allocator statistics.

    On CUDA: exact *allocated CUDA tensor memory* peak via
    ``torch.cuda.max_memory_allocated`` (NOT total device/process memory).
    On Apple silicon: this torch build exposes no peak-stats API for MPS, so
    the meter SAMPLES ``torch.mps.current_allocated_memory()`` on every
    ``sample()`` call (once per training batch) and keeps the maximum.
    On CPU: returns None so that no GPU metric is ever invented for CPU runs.
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.backend = device.type if device.type in ("cuda", "mps") else None
        self.peak_mib: Optional[float] = None
        self._mps_peak = 0

    def reset(self) -> None:
        if self.backend == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        elif self.backend == "mps":
            torch.mps.synchronize()
            self._mps_peak = 0
        self.peak_mib = None

    def sample(self) -> None:
        """Track peak allocated memory (needed for the MPS sampled path)."""
        if self.backend == "mps":
            self._mps_peak = max(self._mps_peak, torch.mps.current_allocated_memory())

    def read(self) -> Optional[float]:
        if self.backend == "cuda":
            torch.cuda.synchronize()
            self.peak_mib = torch.cuda.max_memory_allocated() / (1024**2)
        elif self.backend == "mps":
            torch.mps.synchronize()
            self.sample()
            self.peak_mib = self._mps_peak / (1024**2)
        else:
            return None
        return self.peak_mib


class Timer:
    """Wall-clock timer (seconds)."""

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.time() - self.start


def append_csv_row(path: str, row: Dict[str, Any]) -> None:
    """Append one row to a CSV file, writing the header on first use."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    new_file = not os.path.exists(path)
    fieldnames = list(row.keys())
    if not new_file:
        with open(path, newline="") as f:
            existing = next(csv.reader(f), None)
        if existing:
            # keep the existing header; write missing fields as empty
            for k in row:
                if k not in existing:
                    existing.append(k)
            fieldnames = existing
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def save_json(obj: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=str)


def load_config(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)
