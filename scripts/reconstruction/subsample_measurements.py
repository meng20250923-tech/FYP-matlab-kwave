"""Create reproducible randomly subsampled PAT measurements from saved k-Wave data."""

from __future__ import annotations

import argparse

import h5py
import numpy as np

from pat_fno.data.mnist import CONDITIONS, ROOT, conditions, load_arrays


def parse_args() -> argparse.Namespace:
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


def main() -> None:
    args = parse_args()
    keep_fractions = args.keep_fractions or [args.keep_fraction]
    seeds = args.seeds or [args.seed]
    if any(not 0.0 < value <= 1.0 for value in keep_fractions):
        raise ValueError("All retention fractions must be in (0, 1].")

    dataset_root = ROOT / "datasets" / args.dataset
    output_root = ROOT / "results" / "reconstruction" / args.dataset / "subsampled"
    output_root.mkdir(parents=True, exist_ok=True)

    for condition in conditions(args.condition):
        arrays = load_arrays(dataset_root, condition, args.split)
        full_data = arrays["kwave_forward"].astype(np.float32, copy=False)
        for seed in seeds:
            # Draw one uniform array per seed so masks are nested across fractions.
            # Reinitialising for each condition preserves matched masks for equal shapes.
            uniforms = np.random.default_rng(seed).random(full_data.shape)
            for keep_fraction in keep_fractions:
                output_path = output_root / (
                    f"{condition}_{args.split}_keep{keep_fraction:.2f}_seed{seed}.h5"
                )
                if output_path.exists() and not args.overwrite:
                    print(f"Skip existing file: {output_path}")
                    continue
                mask = uniforms < keep_fraction
                observed_data = np.where(mask, full_data, 0.0).astype(np.float32)
                with h5py.File(output_path, "w") as handle:
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
                    handle.create_dataset(
                        "p0", data=arrays["p0"], compression="gzip", compression_opts=4
                    )
                    handle.create_dataset("label", data=arrays["label"])
                    handle.create_dataset("source_index", data=arrays["source_index"])
                    handle.attrs["dataset"] = args.dataset
                    handle.attrs["condition"] = condition
                    handle.attrs["split"] = args.split
                    handle.attrs["keep_fraction_requested"] = keep_fraction
                    handle.attrs["keep_fraction_actual"] = float(mask.mean())
                    handle.attrs["seed"] = seed
                    handle.attrs["source_measurement"] = "kwave_forward"
                    handle.attrs["nested_mask_design"] = True
                print(f"{condition}: saved {output_path.name}; kept {mask.mean():.2%}")


if __name__ == "__main__":
    main()
