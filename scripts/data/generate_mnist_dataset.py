"""Generate smoke, medium, or large MNIST PAT datasets with one CLI.

Existing shards are skipped, so interrupted runs are safely resumable. The
three scales share the same deterministic MNIST split and numerical operators.
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
from pathlib import Path
from typing import Any

import numpy as np

from pat_fno.config.mnist_pat import SHARD_SIZE, SPLITS
from pat_fno.data.mnist import (
    CONDITIONS,
    ROOT,
    build_setting,
    conditions,
    generate_sample,
    load_mnist_splits,
    save_json,
    to_pressure,
    write_shard,
)

SCALE_CONFIGS = {
    "smoke": {
        "dataset": "mnist_smoke",
        "splits": {"train": 8, "validation": 4, "test": 4},
        "shard_size": 8,
    },
    "medium": {
        "dataset": "mnist_medium_v1",
        "splits": dict(SPLITS),
        "shard_size": SHARD_SIZE,
    },
    "large": {
        "dataset": "mnist_large_v1",
        "splits": {"train": 50_000, "validation": 5_000, "test": 10_000},
        "shard_size": 250,
    },
}

_SETTING = None


def _worker_init(condition: str) -> None:
    global _SETTING
    logging.getLogger().setLevel(logging.ERROR)
    _SETTING = build_setting(condition)


def _worker(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    p0 = to_pressure(image)
    raw, scaled, kwave = generate_sample(p0, _SETTING)
    return p0, raw, scaled, kwave


def parse_args() -> argparse.Namespace:
    """Parse command-line options for deterministic dataset generation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", choices=tuple(SCALE_CONFIGS), required=True)
    parser.add_argument("--dataset", help="Override the default dataset directory name.")
    parser.add_argument("--condition", choices=("all", *CONDITIONS), default="all")
    parser.add_argument("--split", choices=("all", "train", "validation", "test"), default="all")
    parser.add_argument("--workers", type=int, default=min(6, max(1, (os.cpu_count() or 2) - 1)))
    parser.add_argument("--shard-size", type=int)
    parser.add_argument("--maxtasksperchild", type=int, default=50)
    return parser.parse_args()


def _dataset_root(scale: str, dataset: str) -> Path:
    """Return the output directory for one dataset scale."""
    if scale == "smoke":
        return ROOT / "results" / "mnist_smoke" / "dataset"
    return ROOT / "datasets" / dataset


def initialise_manifest(
    root: Path,
    dataset: str,
    scale: str,
    limits: dict[str, int],
) -> tuple[Path, dict[str, Any]]:
    """Load an existing manifest and update its invariant dataset metadata."""
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest.update(
        {
            "dataset": dataset,
            "scale": scale,
            "limits": limits,
            "grid": 64,
            "cfl": 1.0,
            "generator": "parallel",
        }
    )
    manifest.setdefault("conditions", {})
    return manifest_path, manifest


def shard_metadata(condition: str, split: str, setting: Any) -> dict[str, Any]:
    """Build the physical metadata stored with one completed shard."""
    return {
        "condition": condition,
        "split": split,
        "theta_deg": float(np.rad2deg(setting.computation.theta_max)),
        "boundary": setting.kwaveBoundary,
        "Nt": setting.Nt,
        "dt": setting.dt,
        "sound_speed": setting.soundSpeed,
    }


def generate_split_shards(
    pool: Any,
    output: Path,
    condition: str,
    split: str,
    split_arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
    shard_size: int,
    workers: int,
    setting: Any,
) -> None:
    """Generate all ordered shards for one condition and dataset split."""
    images, labels, indices = split_arrays
    for start in range(0, len(images), shard_size):
        stop = min(start + shard_size, len(images))
        path = output / f"{split}_{start:05d}_{stop:05d}.h5"
        if path.exists():
            print(f"skip completed shard {condition}/{path.name}", flush=True)
            continue
        print(
            f"{condition} {split}: samples {start}:{stop} with {workers} workers",
            flush=True,
        )
        result = list(pool.imap(_worker, images[start:stop], chunksize=1))
        p0, fourier_raw, data_fft, data_kwave = (
            np.stack(values).astype(np.float32, copy=False) for values in zip(*result, strict=False)
        )
        write_shard(
            path,
            p0,
            fourier_raw,
            data_fft,
            data_kwave,
            labels[start:stop],
            indices[start:stop],
            shard_metadata(condition, split, setting),
        )
        print(f"saved {path}", flush=True)


def generate_condition(
    context: Any,
    args: argparse.Namespace,
    root: Path,
    raw_splits: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    chosen_splits: tuple[str, ...],
    shard_size: int,
    condition: str,
    manifest: dict[str, Any],
    manifest_path: Path,
) -> None:
    """Generate every requested split for one acquisition condition."""
    setting = build_setting(condition)
    output = root / condition
    output.mkdir(parents=True, exist_ok=True)
    with context.Pool(
        args.workers,
        initializer=_worker_init,
        initargs=(condition,),
        maxtasksperchild=args.maxtasksperchild,
    ) as pool:
        for split in chosen_splits:
            generate_split_shards(
                pool,
                output,
                condition,
                split,
                raw_splits[split],
                shard_size,
                args.workers,
                setting,
            )

    manifest["conditions"][condition] = {
        "boundary": setting.kwaveBoundary,
        "theta_deg": float(np.rad2deg(setting.computation.theta_max)),
        "Nt": setting.Nt,
        "dt": setting.dt,
        "complete": args.split == "all",
    }
    save_json(manifest_path, manifest)


def finalise_manifest(
    manifest: dict[str, Any], manifest_path: Path, all_splits_requested: bool
) -> None:
    """Record whether every condition and split has completed."""
    manifest["complete"] = (
        all_splits_requested
        and set(manifest["conditions"]) == set(CONDITIONS)
        and all(spec.get("complete", False) for spec in manifest["conditions"].values())
    )
    save_json(manifest_path, manifest)


def main() -> None:
    """Generate the requested MNIST PAT dataset shards."""
    args = parse_args()
    config = SCALE_CONFIGS[args.scale]
    dataset = args.dataset or config["dataset"]
    limits = config["splits"]
    shard_size = args.shard_size or config["shard_size"]
    root = _dataset_root(args.scale, dataset)
    raw_splits = load_mnist_splits(ROOT / "data" / "raw_mnist", limits)
    chosen_splits = tuple(limits) if args.split == "all" else (args.split,)
    context = mp.get_context("spawn")
    manifest_path, manifest = initialise_manifest(root, dataset, args.scale, limits)

    for condition in conditions(args.condition):
        generate_condition(
            context,
            args,
            root,
            raw_splits,
            chosen_splits,
            shard_size,
            condition,
            manifest,
            manifest_path,
        )

    finalise_manifest(manifest, manifest_path, args.split == "all")
    print(f"Dataset manifest: {manifest_path}")


if __name__ == "__main__":
    main()
