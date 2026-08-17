"""Evaluate Fourier and learned forward operators and write one comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from pat_fno.data.mnist import (
    CONDITIONS,
    ROOT,
    batch_metrics,
    conditions,
    load_arrays,
    save_json,
)

MODELS = (
    ("fno_only", "FNO-only"),
    ("fourier_to_fno", "Fourier-to-FNO"),
    ("fno_to_fourier", "FNO-to-Fourier"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", choices=("medium", "large"), default="medium")
    parser.add_argument("--dataset")
    parser.add_argument("--condition", choices=("all", *CONDITIONS), default="all")
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help="Reuse an existing Fourier baseline instead of loading the dataset.",
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def _evaluate_fourier(dataset: str, scale: str, selected: str, split: str) -> dict:
    dataset_root = ROOT / "datasets" / dataset
    baseline_root = ROOT / "results" / f"mnist_{scale}" / "baselines"
    baseline_root.mkdir(parents=True, exist_ok=True)
    summary = {}
    for condition in conditions(selected):
        arrays = load_arrays(dataset_root, condition, split)
        metrics = batch_metrics(arrays["data_fft"], arrays["kwave_forward"])
        summary[condition] = metrics
        sensor = arrays["data_fft"].shape[1] // 2
        figure, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
        for axis, value, title in zip(
            axes[0],
            (arrays["p0"][0], arrays["data_fft"][0], arrays["kwave_forward"][0]),
            ("MNIST p0", "c x Fourier", "k-Wave target"),
            strict=False,
        ):
            axis.imshow(value, aspect="auto")
            axis.set_title(title)
        axes[1, 0].imshow(arrays["kwave_forward"][0] - arrays["data_fft"][0], aspect="auto")
        axes[1, 0].set_title("k-Wave - c x Fourier")
        axes[1, 1].plot(arrays["data_fft"][0, sensor], label="c x Fourier")
        axes[1, 1].plot(arrays["kwave_forward"][0, sensor], label="k-Wave")
        axes[1, 1].legend()
        axes[1, 1].set_title("Middle sensor trace")
        axes[1, 2].axis("off")
        axes[1, 2].text(
            0.05,
            0.8,
            "\n".join(f"{key}: {value:.4g}" for key, value in metrics.items()),
            fontsize=12,
        )
        figure.savefig(baseline_root / f"baseline_{condition}_{split}.png", dpi=180)
        plt.close(figure)
        print(
            f"{condition}: relL2={metrics['rel_l2_mean']:.4f}, "
            f"corr={metrics['centered_corr_mean']:.4f}"
        )
    save_json(baseline_root / f"baseline_metrics_{split}.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    dataset = args.dataset or f"mnist_{args.scale}_v1"
    baseline_root = ROOT / "results" / f"mnist_{args.scale}" / "baselines"
    baseline_path = baseline_root / f"baseline_metrics_{args.split}.json"
    baseline = (
        _read_json(baseline_path) or {}
        if args.metrics_only
        else _evaluate_fourier(dataset, args.scale, args.condition, args.split)
    )

    model_root = ROOT / "results" / "mnist_medium" / dataset
    output = ROOT / "results" / f"mnist_{args.scale}" / dataset
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for condition in conditions(args.condition):
        if condition in baseline:
            rows.append(
                {"condition": condition, "model": "c x Fourier baseline", **baseline[condition]}
            )
        for key, label in MODELS:
            result = _read_json(model_root / condition / key / "metrics.json")
            if result:
                rows.append({"condition": condition, "model": label, **result[args.split]})

    save_json(
        output / "comparison.json",
        {
            "dataset": dataset,
            "scale": args.scale,
            "split": args.split,
            "rows": rows,
        },
    )
    if not rows:
        print("No completed metrics found.")
        return

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for condition in conditions(args.condition):
        group = [row for row in rows if row["condition"] == condition]
        x = list(range(len(group)))
        labels = [row["model"] for row in group]
        axes[0].plot(x, [row["rel_l2_mean"] for row in group], marker="o", label=condition)
        axes[1].plot(x, [row["centered_corr_mean"] for row in group], marker="o", label=condition)
        for axis in axes:
            axis.set_xticks(x, labels, rotation=22, ha="right")
            axis.grid(alpha=0.3)
    axes[0].set_ylabel("mean per-sample relative L2")
    axes[1].set_ylabel("mean centered correlation")
    axes[0].legend()
    axes[1].legend()
    figure.savefig(output / "comparison.png", dpi=180)
    plt.close(figure)
    print(f"Saved forward comparison in {output}")


if __name__ == "__main__":
    main()
