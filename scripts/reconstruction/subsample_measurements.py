"""Create reproducible randomly subsampled PAT measurements from saved k-Wave data."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from pat_fno.data.mnist import CONDITIONS, ROOT, conditions, load_arrays


def parse_args() -> argparse.Namespace:
    """Parse command-line options for measurement subsampling."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="mnist_medium_v1")
    parser.add_argument("--condition", choices=("all", *CONDITIONS), default="all")
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--keep-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument(
        "--keep-fractions",
        type=float,
        nargs="+",
        default=None,
        help="Generate several retention fractions in one run.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Generate several independent mask seeds in one run.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_requests(args: argparse.Namespace) -> tuple[list[float], list[int]]:
    """Resolve singular or plural retention fractions and random seeds."""
    keep_fractions = args.keep_fractions or [args.keep_fraction]
    seeds = args.seeds or [args.seed]
    if any(not 0.0 < value <= 1.0 for value in keep_fractions):
        raise ValueError("All retention fractions must be in (0, 1].")
    return keep_fractions, seeds


def output_path(
    output_root: Path,
    condition: str,
    split: str,
    keep_fraction: float,
    seed: int,
) -> Path:
    """Return the stable HDF5 path for one masked measurement set."""
    return output_root / f"{condition}_{split}_keep{keep_fraction:.2f}_seed{seed}.h5"


def write_subsampled_measurements(
    path: Path,
    arrays: dict[str, np.ndarray],
    full_data: np.ndarray,
    uniforms: np.ndarray,
    keep_fraction: float,
    seed: int,
    dataset: str,
    condition: str,
    split: str,
) -> float:
    """Write one requested retention level and return its realised fraction."""
    mask = uniforms < keep_fraction
    observed_data = np.where(mask, full_data, 0.0).astype(np.float32)
    with h5py.File(path, "w") as handle:
        handle.create_dataset(
            "observed_data",
            data=observed_data,
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )
        handle.create_dataset(
            "mask", data=mask, compression="gzip", compression_opts=4, shuffle=True
        )
        handle.create_dataset("p0", data=arrays["p0"], compression="gzip", compression_opts=4)
        handle.create_dataset("label", data=arrays["label"])
        handle.create_dataset("source_index", data=arrays["source_index"])
        handle.attrs["dataset"] = dataset
        handle.attrs["condition"] = condition
        handle.attrs["split"] = split
        handle.attrs["keep_fraction_requested"] = keep_fraction
        handle.attrs["keep_fraction_actual"] = float(mask.mean())
        handle.attrs["seed"] = seed
        handle.attrs["source_measurement"] = "kwave_forward"
        handle.attrs["nested_mask_design"] = True
    return float(mask.mean())


def generate_seed_outputs(
    args: argparse.Namespace,
    output_root: Path,
    condition: str,
    arrays: dict[str, np.ndarray],
    full_data: np.ndarray,
    keep_fractions: list[float],
    seed: int,
) -> None:
    """Generate nested retention masks and files for one random seed."""
    # Draw one uniform array per seed so masks are nested across fractions.
    # Reinitialising for each condition preserves matched masks for equal shapes.
    uniforms = np.random.default_rng(seed).random(full_data.shape)
    for keep_fraction in keep_fractions:
        path = output_path(output_root, condition, args.split, keep_fraction, seed)
        if path.exists() and not args.overwrite:
            print(f"Skip existing file: {path}")
            continue
        actual_fraction = write_subsampled_measurements(
            path,
            arrays,
            full_data,
            uniforms,
            keep_fraction,
            seed,
            args.dataset,
            condition,
            args.split,
        )
        print(f"{condition}: saved {path.name}; kept {actual_fraction:.2%}")


def generate_condition_outputs(
    args: argparse.Namespace,
    dataset_root: Path,
    output_root: Path,
    condition: str,
    keep_fractions: list[float],
    seeds: list[int],
) -> None:
    """Load one condition and generate every requested seed and retention."""
    arrays: dict[str, Any] = load_arrays(dataset_root, condition, args.split)
    full_data = arrays["kwave_forward"].astype(np.float32, copy=False)
    for seed in seeds:
        generate_seed_outputs(
            args,
            output_root,
            condition,
            arrays,
            full_data,
            keep_fractions,
            seed,
        )


def main() -> None:
    """Create reproducible masked measurements for all requested settings."""
    args = parse_args()
    keep_fractions, seeds = resolve_requests(args)

    dataset_root = ROOT / "datasets" / args.dataset
    output_root = ROOT / "results" / "reconstruction" / args.dataset / "subsampled"
    output_root.mkdir(parents=True, exist_ok=True)

    for condition in conditions(args.condition):
        generate_condition_outputs(
            args,
            dataset_root,
            output_root,
            condition,
            keep_fractions,
            seeds,
        )


if __name__ == "__main__":
    main()
