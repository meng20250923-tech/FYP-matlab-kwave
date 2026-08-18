"""Shared input and naming helpers for reconstruction commands."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

from pat_fno.data.mnist import ROOT


def reconstruction_tag(condition: str, split: str, keep_fraction: float, seed: int) -> str:
    """Return the deterministic filename tag for one reconstruction run."""
    return f"{condition}_{split}_keep{keep_fraction:.2f}_seed{seed}"


def subsampled_path(args: argparse.Namespace, condition: str) -> Path:
    """Return the saved incomplete-measurement path for one condition."""
    return (
        ROOT
        / "results"
        / "reconstruction"
        / args.dataset
        / "subsampled"
        / f"{reconstruction_tag(condition, args.split, args.keep_fraction, args.seed)}.h5"
    )


def load_subsampled(
    path: Path,
    max_samples: int,
    *,
    include_mask: bool = False,
) -> dict[str, np.ndarray]:
    """Load a bounded number of samples from an incomplete-data archive."""
    if not path.exists():
        raise FileNotFoundError(f"Missing subsampled data: {path}")
    if max_samples <= 0:
        raise ValueError("--max-samples must be positive.")
    names = ["observed_data", "p0", "label", "source_index"]
    if include_mask:
        names.append("mask")
    with h5py.File(path, "r") as handle:
        count = min(max_samples, len(handle["p0"]))
        return {name: handle[name][:count] for name in names}
