"""Analyse the additional experiments explicitly requested by the supervisor.

Subcommands:
  sample-efficiency  Summarise training-size runs and draw learning curves.
  reconstruction    Summarise retention robustness, uncertainty, convergence,
                    and recorded Lipschitz/step-size estimates.

This script is read-only with respect to experiment outputs. It writes derived
tables and figures under results/evaluation/<dataset>/required_experiments.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pat_fno.data.mnist import CONDITIONS, ROOT, centered_corr, rel_l2
from scripts.reconstruction.common import reconstruction_tag

LEARNED = (
    ("fno_only", "FNO-only"),
    ("fourier_to_fno", "Fourier-to-FNO"),
    ("fno_to_fourier", "FNO-to-Fourier"),
)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = []
    for row in rows:
        keys.extend(key for key in row if key not in keys)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, draws: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan"), float("nan")
    if draws <= 0:
        return float("nan"), float("nan")
    means = np.empty(draws)
    for index in range(draws):
        means[index] = rng.choice(values, size=len(values), replace=True).mean()
    return tuple(float(value) for value in np.quantile(means, [0.025, 0.975]))


def sample_efficiency(args: argparse.Namespace) -> None:
    source = ROOT / "results" / "sample_efficiency" / args.dataset
    output = ROOT / "results" / "evaluation" / args.dataset / "required_experiments"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for condition in CONDITIONS:
        for key, label in LEARNED:
            model_root = source / condition / key
            if not model_root.exists():
                continue
            for run in sorted(model_root.glob("n*_seed*")):
                metrics_path = run / "metrics.json"
                history_path = run / "history.json"
                if not metrics_path.exists() or not history_path.exists():
                    continue
                metrics = json.loads(metrics_path.read_text())
                history = json.loads(history_path.read_text())
                arguments = history["arguments"]
                rows.append(
                    {
                        "condition": condition,
                        "model": label,
                        "train_samples": arguments.get("train_samples"),
                        "seed": arguments["seed"],
                        **metrics["test"],
                    }
                )
    if not rows:
        raise FileNotFoundError(f"No completed sample-efficiency runs under {source}")
    rows.sort(key=lambda row: (row["condition"], row["model"], row["train_samples"], row["seed"]))
    write_csv(output / "sample_efficiency_runs.csv", rows)

    aggregate = []
    for condition in CONDITIONS:
        for _, label in LEARNED:
            sizes = sorted(
                {
                    row["train_samples"]
                    for row in rows
                    if row["condition"] == condition and row["model"] == label
                }
            )
            for size in sizes:
                group = [
                    row
                    for row in rows
                    if row["condition"] == condition
                    and row["model"] == label
                    and row["train_samples"] == size
                ]
                values = np.asarray([row["rel_l2_mean"] for row in group])
                aggregate.append(
                    {
                        "condition": condition,
                        "model": label,
                        "train_samples": size,
                        "runs": len(group),
                        "rel_l2_mean": float(values.mean()),
                        "rel_l2_std_across_seeds": float(values.std(ddof=1))
                        if len(values) > 1
                        else float("nan"),
                    }
                )
    write_csv(output / "sample_efficiency_summary.csv", aggregate)

    figure, axes = plt.subplots(
        1, len(CONDITIONS), figsize=(11, 4.2), sharey=True, constrained_layout=True
    )
    for axis, condition in zip(np.atleast_1d(axes), CONDITIONS, strict=False):
        for _, label in LEARNED:
            group = [
                row for row in aggregate if row["condition"] == condition and row["model"] == label
            ]
            if not group:
                continue
            x = np.asarray([row["train_samples"] for row in group])
            y = np.asarray([row["rel_l2_mean"] for row in group])
            error = np.nan_to_num([row["rel_l2_std_across_seeds"] for row in group])
            axis.errorbar(x, y, yerr=error, marker="o", capsize=3, label=label)
        axis.set_xscale("log")
        axis.set(
            title=condition.replace("_", " "), xlabel="training samples", ylabel="test relative L2"
        )
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    figure.savefig(output / "sample_efficiency.png", dpi=200)
    plt.close(figure)
    print(f"Saved sample-efficiency analysis in {output}")


def per_sample_metrics(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path)
    prediction, target = data["reconstruction"], data["p0"]
    relative = np.asarray([rel_l2(a, b) for a, b in zip(prediction, target, strict=False)])
    correlation = np.asarray(
        [centered_corr(a, b) for a, b in zip(prediction, target, strict=False)]
    )
    mse = np.mean((prediction.astype(np.float64) - target.astype(np.float64)) ** 2, axis=(1, 2))
    return {"relative_l2": relative, "correlation": correlation, "mse": mse}


def fit_curve(values: np.ndarray) -> dict[str, float]:
    y = np.asarray(values, dtype=float)[1:]
    x = np.arange(1, len(values), dtype=float)
    valid = np.isfinite(y) & (y > 0)
    x, y = x[valid], y[valid]
    if len(x) < 3:
        return {
            name: float("nan")
            for name in ("semilog_slope", "semilog_r2", "loglog_slope", "loglog_r2")
        }

    def fit(x_value, y_value):
        slope, intercept = np.polyfit(x_value, y_value, 1)
        prediction = slope * x_value + intercept
        total = np.sum((y_value - y_value.mean()) ** 2)
        r2 = 1.0 - np.sum((y_value - prediction) ** 2) / total if total > 0 else float("nan")
        return float(slope), float(r2)

    semilog = fit(x, np.log(y))
    loglog = fit(np.log(x), np.log(y))
    return {
        "semilog_slope": semilog[0],
        "semilog_r2": semilog[1],
        "loglog_slope": loglog[0],
        "loglog_r2": loglog[1],
    }


def method_paths(root: Path, condition: str, tag: str, itr_step: str) -> dict[str, Path]:
    return {
        "Fourier inverse": root / "fourier" / f"{tag}.npz",
        "Time reversal": root / "time_reversal" / f"{tag}.npz",
        "Iterated time reversal": root / "iterated_time_reversal" / f"{tag}_step{itr_step}.npz",
        "Gradient descent (1/L)": root / "gradient_descent" / f"{tag}.npz",
        "FNO-only": root / "learned_operator" / "medium" / "fno_only" / f"{tag}.npz",
        "Fourier-to-FNO": root / "learned_operator" / "medium" / "fourier_to_fno" / f"{tag}.npz",
        "FNO-to-Fourier": root / "learned_operator" / "medium" / "fno_to_fourier" / f"{tag}.npz",
    }


def reconstruction(args: argparse.Namespace) -> None:
    root = ROOT / "results" / "reconstruction" / args.dataset
    output = ROOT / "results" / "evaluation" / args.dataset / "required_experiments"
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.bootstrap_seed)
    rows, fits, lipschitz = [], [], []
    itr_steps = {
        "periodic_theta89": args.periodic_itr_step,
        "pml_outside_theta45": args.pml_itr_step,
    }
    full_retention_itr_steps = {
        "periodic_theta89": args.periodic_full_retention_itr_step,
        "pml_outside_theta45": args.pml_full_retention_itr_step,
    }
    retention_itr_steps = None
    if args.periodic_itr_steps is not None or args.pml_itr_steps is not None:
        if args.periodic_itr_steps is None or args.pml_itr_steps is None:
            raise ValueError("Provide both --periodic-itr-steps and --pml-itr-steps.")
        expected = len(args.keep_fractions)
        if len(args.periodic_itr_steps) != expected or len(args.pml_itr_steps) != expected:
            raise ValueError(
                "Each retention-specific ITR step list must contain one value "
                f"for every --keep-fractions entry ({expected} values expected)."
            )
        retention_itr_steps = {
            "periodic_theta89": args.periodic_itr_steps,
            "pml_outside_theta45": args.pml_itr_steps,
        }

    for fraction_index, fraction in enumerate(args.keep_fractions):
        for seed in args.seeds:
            for condition in CONDITIONS:
                tag = reconstruction_tag(condition, args.split, fraction, seed)
                if retention_itr_steps is not None:
                    itr_step = retention_itr_steps[condition][fraction_index]
                else:
                    itr_step = (
                        full_retention_itr_steps[condition]
                        if np.isclose(fraction, 1.0)
                        else itr_steps[condition]
                    )
                paths = method_paths(root, condition, tag, itr_step)
                for method, path in paths.items():
                    if not path.exists():
                        if not args.allow_missing:
                            raise FileNotFoundError(path)
                        continue
                    values = per_sample_metrics(path)
                    row = {
                        "condition": condition,
                        "keep_fraction": fraction,
                        "seed": seed,
                        "method": method,
                        "num_samples": len(values["relative_l2"]),
                    }
                    for metric, samples in values.items():
                        low, high = bootstrap_ci(samples, rng, args.bootstrap_draws)
                        row[f"{metric}_mean"] = float(np.nanmean(samples))
                        row[f"{metric}_ci95_low"] = low
                        row[f"{metric}_ci95_high"] = high
                    rows.append(row)

                gd_path, itr_path = paths["Gradient descent (1/L)"], paths["Iterated time reversal"]
                if gd_path.exists() and itr_path.exists():
                    series = (("GD", np.load(gd_path)), ("Iterated TR", np.load(itr_path)))
                    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
                    for method, data in series:
                        for metric_key, metric_label, style in (
                            ("residual_history", "residual", "-"),
                            ("error_history", "rel-L2", "--"),
                        ):
                            histories = data[metric_key]
                            mean = histories.mean(axis=0)
                            low, high = np.quantile(histories, [0.025, 0.975], axis=0)
                            x = np.arange(len(mean))
                            fits.append(
                                {
                                    "condition": condition,
                                    "keep_fraction": fraction,
                                    "seed": seed,
                                    "method": method,
                                    "metric": metric_label,
                                    **fit_curve(mean),
                                }
                            )
                            for axis, scale in zip(axes, ("semilogy", "loglog"), strict=False):
                                start = 0 if scale == "semilogy" else 1
                                getattr(axis, scale)(
                                    x[start:], mean[start:], style, label=f"{method} {metric_label}"
                                )
                                axis.fill_between(x[start:], low[start:], high[start:], alpha=0.10)
                    for axis, title in zip(
                        axes, ("Semilog convergence", "Loglog convergence"), strict=False
                    ):
                        axis.set(
                            title=title, xlabel="iteration", ylabel="mean with 95% sample interval"
                        )
                        axis.grid(alpha=0.3)
                        axis.legend(fontsize=8)
                    figure.savefig(output / f"{tag}_convergence_with_intervals.png", dpi=200)
                    plt.close(figure)

                    gd = np.load(gd_path)
                    if "lipschitz" in gd and "step_size" in gd:
                        lipschitz.append(
                            {
                                "condition": condition,
                                "keep_fraction": fraction,
                                "seed": seed,
                                "num_samples": len(gd["lipschitz"]),
                                "lipschitz_mean": float(gd["lipschitz"].mean()),
                                "lipschitz_std": float(gd["lipschitz"].std(ddof=1)),
                                "lipschitz_min": float(gd["lipschitz"].min()),
                                "lipschitz_max": float(gd["lipschitz"].max()),
                                "step_size_mean": float(gd["step_size"].mean()),
                                "step_size_std": float(gd["step_size"].std(ddof=1)),
                            }
                        )

    if not rows:
        raise FileNotFoundError("No reconstruction outputs matched the requested fractions/seeds.")
    write_csv(output / "reconstruction_with_uncertainty.csv", rows)
    write_csv(output / "convergence_fits.csv", fits)
    write_csv(output / "lipschitz_step_size_summary.csv", lipschitz)

    figure, axes = plt.subplots(
        1, len(CONDITIONS), figsize=(12, 4.5), sharey=True, constrained_layout=True
    )
    for axis, condition in zip(np.atleast_1d(axes), CONDITIONS, strict=False):
        for method in sorted({row["method"] for row in rows}):
            points = []
            for fraction in sorted(args.keep_fractions):
                group = [
                    row["relative_l2_mean"]
                    for row in rows
                    if row["condition"] == condition
                    and row["method"] == method
                    and row["keep_fraction"] == fraction
                ]
                if group:
                    points.append(
                        (
                            fraction,
                            float(np.mean(group)),
                            float(np.std(group, ddof=1)) if len(group) > 1 else 0.0,
                        )
                    )
            if points:
                x, y, error = map(np.asarray, zip(*points, strict=False))
                axis.errorbar(x, y, yerr=error, marker="o", capsize=3, label=method)
        axis.set(
            title=condition.replace("_", " "),
            xlabel="measurement retention",
            ylabel="mean reconstruction relative L2",
        )
        axis.grid(alpha=0.3)
        axis.legend(fontsize=7)
    figure.savefig(output / "retention_robustness.png", dpi=200)
    plt.close(figure)
    print(f"Saved reconstruction analysis in {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="mnist_medium_v1")
    subparsers = parser.add_subparsers(dest="command", required=True)
    sample = subparsers.add_parser("sample-efficiency")
    sample.set_defaults(function=sample_efficiency)
    recon = subparsers.add_parser("reconstruction")
    recon.add_argument("--split", default="test")
    recon.add_argument("--keep-fractions", type=float, nargs="+", default=(0.10, 0.25, 0.50, 1.00))
    recon.add_argument("--seeds", type=int, nargs="+", default=(20260802,))
    recon.add_argument("--periodic-itr-step", default="1.5")
    recon.add_argument("--pml-itr-step", default="2")
    recon.add_argument("--periodic-full-retention-itr-step", default="0.75")
    recon.add_argument("--pml-full-retention-itr-step", default="1.75")
    recon.add_argument(
        "--periodic-itr-steps",
        nargs="+",
        help="One ITR step per --keep-fractions entry for the periodic condition.",
    )
    recon.add_argument(
        "--pml-itr-steps",
        nargs="+",
        help="One ITR step per --keep-fractions entry for the PML condition.",
    )
    recon.add_argument("--bootstrap-draws", type=int, default=2000)
    recon.add_argument("--bootstrap-seed", type=int, default=20260805)
    recon.add_argument("--allow-missing", action="store_true")
    recon.set_defaults(function=reconstruction)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
