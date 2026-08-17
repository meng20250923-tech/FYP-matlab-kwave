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

import numpy as np

from pat_fno.data.mnist import (
    CONDITIONS,
    ROOT,
    SHARD_SIZE,
    SPLITS,
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", choices=tuple(SCALE_CONFIGS), required=True)
    parser.add_argument("--dataset", help="Override the default dataset directory name.")
    parser.add_argument("--condition", choices=("all", *CONDITIONS), default="all")
    parser.add_argument("--split", choices=("all", "train", "validation", "test"), default="all")
    parser.add_argument("--workers", type=int, default=min(6, max(1, (os.cpu_count() or 2) - 1)))
    parser.add_argument("--shard-size", type=int)
    parser.add_argument("--maxtasksperchild", type=int, default=50)
    return parser.parse_args()


def _dataset_root(scale: str, dataset: str):
    if scale == "smoke":
        return ROOT / "results" / "mnist_smoke" / "dataset"
    return ROOT / "datasets" / dataset


def main() -> None:
    args = parse_args()
    config = SCALE_CONFIGS[args.scale]
    dataset = args.dataset or config["dataset"]
    limits = config["splits"]
    shard_size = args.shard_size or config["shard_size"]
    root = _dataset_root(args.scale, dataset)
    raw_splits = load_mnist_splits(ROOT / "data" / "raw_mnist", limits)
    chosen_splits = tuple(limits) if args.split == "all" else (args.split,)
    context = mp.get_context("spawn")

    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest.update(
        {
            "dataset": dataset,
            "scale": args.scale,
            "limits": limits,
            "grid": 64,
            "cfl": 1.0,
            "generator": "parallel",
        }
    )
    manifest.setdefault("conditions", {})

    for condition in conditions(args.condition):
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
                images, labels, indices = raw_splits[split]
                for start in range(0, len(images), shard_size):
                    stop = min(start + shard_size, len(images))
                    path = output / f"{split}_{start:05d}_{stop:05d}.h5"
                    if path.exists():
                        print(f"skip completed shard {condition}/{path.name}", flush=True)
                        continue
                    print(
                        f"{condition} {split}: samples {start}:{stop} with {args.workers} workers",
                        flush=True,
                    )
                    result = list(pool.imap(_worker, images[start:stop], chunksize=1))
                    p0, fourier_raw, data_fft, data_kwave = (
                        np.stack(values).astype(np.float32, copy=False)
                        for values in zip(*result, strict=False)
                    )
                    write_shard(
                        path,
                        p0,
                        fourier_raw,
                        data_fft,
                        data_kwave,
                        labels[start:stop],
                        indices[start:stop],
                        {
                            "condition": condition,
                            "split": split,
                            "theta_deg": float(np.rad2deg(setting.computation.theta_max)),
                            "boundary": setting.kwaveBoundary,
                            "Nt": setting.Nt,
                            "dt": setting.dt,
                            "sound_speed": setting.soundSpeed,
                        },
                    )
                    print(f"saved {path}", flush=True)

        manifest["conditions"][condition] = {
            "boundary": setting.kwaveBoundary,
            "theta_deg": float(np.rad2deg(setting.computation.theta_max)),
            "Nt": setting.Nt,
            "dt": setting.dt,
            "complete": args.split == "all",
        }
        save_json(manifest_path, manifest)

    manifest["complete"] = (
        args.split == "all"
        and set(manifest["conditions"]) == set(CONDITIONS)
        and all(spec.get("complete", False) for spec in manifest["conditions"].values())
    )
    save_json(manifest_path, manifest)
    print(f"Dataset manifest: {manifest_path}")


if __name__ == "__main__":
    main()
