"""Generate a structured, publication-ready figure set from saved experiments.

The script is read-only with respect to experiment data.  Every output file is a
single plot or a single image (no subplot panels), so figures can be placed and
captioned independently in the dissertation.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from pat_fno.data.mnist import ROOT

CONDITIONS = ("periodic_theta89", "pml_outside_theta45")
RETENTION_FRACTIONS = (0.10, 0.25, 0.50, 1.00)
CONDITION_LABELS = {
    "periodic_theta89": r"Periodic $89^\circ$",
    "pml_outside_theta45": r"PML $45^\circ$",
}
METHOD_COLORS = {
    "Fourier inverse": "#6B7280",
    "Time reversal": "#8B5CF6",
    "Iterated time reversal": "#DC2626",
    "Gradient descent (1/L)": "#2563EB",
    "FNO-only": "#2563EB",
    "Fourier-to-FNO": "#059669",
    "FNO-to-Fourier": "#E69F00",
    "GD": "#2563EB",
    "Iterated TR": "#DC2626",
}
METHOD_PATHS = {
    "Fourier inverse": ("fourier", None),
    "Time reversal": ("time_reversal", None),
    "Iterated time reversal": ("iterated_time_reversal", "step"),
    "Gradient descent (1/L)": ("gradient_descent", None),
    "FNO-only": ("learned_operator/medium/fno_only", None),
    "Fourier-to-FNO": ("learned_operator/medium/fourier_to_fno", None),
    "FNO-to-Fourier": ("learned_operator/medium/fno_to_fourier", None),
}
ITR_STEPS = {
    ("periodic_theta89", 0.10): "2",
    ("periodic_theta89", 0.25): "1.5",
    ("periodic_theta89", 0.50): "1.5",
    ("periodic_theta89", 1.00): "0.75",
    ("pml_outside_theta45", 0.10): "2.5",
    ("pml_outside_theta45", 0.25): "2",
    ("pml_outside_theta45", 0.50): "2.5",
    ("pml_outside_theta45", 1.00): "1.75",
}

ITR_VALIDATION_STEPS = {
    ("periodic_theta89", 0.10): (0.5, 1.0, 1.5, 2.0, 2.5),
    ("periodic_theta89", 0.25): (0.5, 1.0, 1.5, 2.0, 2.5),
    ("periodic_theta89", 0.50): (0.5, 1.0, 1.5, 2.0, 2.5),
    ("periodic_theta89", 1.00): (0.1, 0.25, 0.5, 0.75, 1.0),
    ("pml_outside_theta45", 0.10): (0.5, 1.0, 1.5, 2.0, 2.5),
    ("pml_outside_theta45", 0.25): (0.5, 1.0, 1.5, 2.0, 2.5),
    ("pml_outside_theta45", 0.50): (0.5, 1.0, 1.5, 2.0, 2.5),
    ("pml_outside_theta45", 1.00): (
        0.1,
        0.25,
        0.5,
        0.75,
        1.0,
        1.25,
        1.5,
        1.75,
        2.0,
        2.25,
        2.5,
    ),
}

LEARNED_FORWARD_SCENARIOS = (
    ("fno_only", "FNO-only"),
    ("fourier_to_fno", "Fourier-to-FNO"),
    ("fno_to_fourier", "FNO-to-Fourier"),
)


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV file into an ordered list of row dictionaries."""
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def setup_style() -> None:
    """Apply the common publication style to subsequent figures."""
    plt.rcParams.update(
        {
            "figure.figsize": (7.0, 4.6),
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 11,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "lines.linewidth": 2.0,
            "lines.markersize": 6,
        }
    )


def save_figure(figure: plt.Figure, path: Path) -> None:
    """Save one untitled figure in both PNG and editable SVG formats."""
    path.parent.mkdir(parents=True, exist_ok=True)
    for axis in figure.axes:
        axis.set_title("")
    if getattr(figure, "_suptitle", None) is not None:
        figure._suptitle.remove()
    base = path.parent / path.stem
    figure.savefig(Path(f"{base}.png"), bbox_inches="tight", facecolor="white")
    figure.savefig(Path(f"{base}.svg"), bbox_inches="tight", facecolor="white")
    plt.close(figure)


def line_figure(
    x: np.ndarray,
    series: list[tuple[str, np.ndarray]],
    xlabel: str,
    ylabel: str,
    title: str,
    path: Path,
    *,
    xscale: str = "linear",
    yscale: str = "linear",
    intervals: list[tuple[np.ndarray, np.ndarray]] | None = None,
) -> None:
    figure, axis = plt.subplots()
    markers = ("o", "s", "^", "D", "v", "P", "X")
    for index, (label, values) in enumerate(series):
        color = METHOD_COLORS.get(label)
        axis.plot(x, values, marker=markers[index % len(markers)], label=label, color=color)
        if intervals is not None:
            low, high = intervals[index]
            axis.fill_between(x, low, high, color=color, alpha=0.12, linewidth=0)
    axis.set(xlabel=xlabel, ylabel=ylabel, title=title)
    axis.set_xscale(xscale)
    axis.set_yscale(yscale)
    axis.legend(frameon=False)
    save_figure(figure, path)


def learned_prediction_path(condition: str, scenario: str) -> Path:
    """Return the saved large-test prediction archive for one learned model."""
    return (
        ROOT / "results/mnist_medium/mnist_large_v1" / condition / scenario / "test_predictions.npz"
    )


def load_prediction_sample(condition: str, scenario: str, sample_index: int) -> np.ndarray:
    """Load one learned pressure-field prediction without retaining the archive."""
    with np.load(learned_prediction_path(condition, scenario)) as saved:
        return np.asarray(saved["prediction"][sample_index])


def load_forward_sample(
    condition: str,
    sample_index: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Load analytical, reference, and learned fields for one large-test sample."""
    analytical, target = _large_forward_arrays(condition)
    learned = {
        scenario: load_prediction_sample(condition, scenario, sample_index)
        for scenario, _ in LEARNED_FORWARD_SCENARIOS
    }
    return (
        np.asarray(analytical[sample_index]),
        np.asarray(target[sample_index]),
        learned,
    )


def plot_sample_efficiency(output: Path) -> None:
    data = read_csv(
        ROOT
        / "results/evaluation/mnist_large_v1/required_experiments/sample_efficiency_summary.csv"
    )
    for condition in CONDITIONS:
        selected = [r for r in data if r["condition"] == condition]
        sizes = np.asarray(sorted({int(r["train_samples"]) for r in selected}))
        figure, axis = plt.subplots(figsize=(7.5, 4.8))
        for model, color, marker in (
            ("FNO-only", "#2563eb", "o"),
            ("Fourier-to-FNO", "#dc2626", "s"),
            ("FNO-to-Fourier", "#059669", "^"),
        ):
            lookup = {int(r["train_samples"]): r for r in selected if r["model"] == model}
            mean = np.asarray([float(lookup[n]["rel_l2_mean"]) for n in sizes])
            std = np.asarray([float(lookup[n]["rel_l2_std_across_seeds"]) for n in sizes])
            axis.errorbar(
                sizes,
                mean,
                yerr=std,
                color=color,
                marker=marker,
                linewidth=1.8,
                capsize=4,
                label=model,
            )
        axis.axhline(
            0.22,
            color="#6b7280",
            linestyle="--",
            linewidth=1.2,
            label=r"empirical threshold $0.22$",
        )
        axis.set(
            xlabel="Number of training samples",
            ylabel=r"Test relative $L_2$ error (mean $\pm$ SD)",
            title=f"Sample efficiency: {CONDITION_LABELS[condition]}",
        )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
        save_figure(
            figure,
            output / "02_sample_efficiency" / f"{condition}_sample_efficiency.png",
        )


def plot_training_curves(output: Path) -> None:
    """Save one five-epoch training-history plot per acquisition condition."""
    root = ROOT / "results/mnist_medium/mnist_medium_v1"
    scenarios = (
        ("fno_only", "FNO-only", "#2563EB"),
        ("fourier_to_fno", "Fourier-to-FNO", "#DC2626"),
        ("fno_to_fourier", "FNO-to-Fourier", "#059669"),
    )
    for condition in CONDITIONS:
        figure, axis = plt.subplots(figsize=(7.2, 4.8))
        for scenario, label, color in scenarios:
            history_path = root / condition / scenario / "history.json"
            history = json.loads(history_path.read_text(encoding="utf-8"))["history"]
            epochs = np.asarray([row["epoch"] for row in history])
            train = np.asarray([row["train_normalized_mse"] for row in history])
            validation = np.asarray([row["normalized_mse"] for row in history])
            axis.plot(epochs, train, linestyle="--", color=color, alpha=0.65)
            axis.plot(epochs, validation, marker="o", color=color, label=label)
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Normalised MSE")
        axis.set_yscale("log")
        axis.set_xticks(np.arange(1, 6))
        axis.legend(frameon=False)
        save_figure(
            figure,
            output / "01_forward_accuracy" / f"{condition}_training_curves.png",
        )


def plot_runtime(output: Path) -> None:
    methods = ("Fourier", "fno_only", "fourier_to_fno", "fno_to_fourier", "k-Wave")
    labels = ("Fourier", "FNO-only", "Fourier-to-FNO", "FNO-to-Fourier", "k-Wave")
    modes = (
        ("CPU, batch 1", "cpu", "#111827", "o"),
        ("RTX 4090, batch 1", "cuda_batch1", "#2563eb", "s"),
        ("RTX 4090, batch 64", "cuda_batch64", "#dc2626", "^"),
    )
    for condition in CONDITIONS:
        figure, axis = plt.subplots(figsize=(7.5, 4.8))
        x = np.arange(len(methods))
        for label, suffix, color, marker in modes:
            lookup = {
                r["method"]: r
                for r in read_csv(
                    ROOT / f"results/evaluation/mnist_large_v1/runtime/{condition}_{suffix}.csv"
                )
            }
            values = np.asarray(
                [
                    float(lookup[m]["mean_ms_per_sample"])
                    if not (suffix.startswith("cuda") and m == "k-Wave")
                    else np.nan
                    for m in methods
                ]
            )
            axis.plot(x, values, color=color, marker=marker, linewidth=1.8, label=label)
        axis.set_xticks(x, labels, rotation=18, ha="right")
        axis.set_yscale("log")
        axis.set_ylabel("Runtime per sample (ms, log scale)")
        axis.set_title(f"Forward runtime: {CONDITION_LABELS[condition]}")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
        save_figure(figure, output / "03_runtime" / f"{condition}_runtime_log.png")


def plot_reconstruction_metrics(output: Path) -> None:
    """Plot retention robustness in separate classical and learned panels."""
    rows = read_csv(
        ROOT
        / "results/evaluation/mnist_medium_v1"
        / "required_experiments/reconstruction_with_uncertainty.csv"
    )
    fractions = np.asarray(RETENTION_FRACTIONS)
    groups = {
        "classical": (
            "Fourier inverse",
            "Time reversal",
            "Iterated time reversal",
            "Gradient descent (1/L)",
        ),
        "learned": ("FNO-only", "Fourier-to-FNO", "FNO-to-Fourier"),
    }
    all_low = np.asarray([float(row["relative_l2_ci95_low"]) for row in rows])
    all_high = np.asarray([float(row["relative_l2_ci95_high"]) for row in rows])
    padding = 0.04 * (all_high.max() - all_low.min())
    common_ylim = (max(0.0, all_low.min() - padding), all_high.max() + padding)

    for condition in CONDITIONS:
        selected = [row for row in rows if row["condition"] == condition]
        for group_name, methods in groups.items():
            figure, axis = plt.subplots(figsize=(6.2, 4.5))
            for index, method in enumerate(methods):
                lookup = {
                    float(row["keep_fraction"]): row for row in selected if row["method"] == method
                }
                mean = np.asarray([float(lookup[x]["relative_l2_mean"]) for x in fractions])
                low = np.asarray([float(lookup[x]["relative_l2_ci95_low"]) for x in fractions])
                high = np.asarray([float(lookup[x]["relative_l2_ci95_high"]) for x in fractions])
                color = METHOD_COLORS[method]
                marker = ("o", "s", "^", "D")[index]
                axis.plot(
                    fractions * 100,
                    mean,
                    color=color,
                    marker=marker,
                    linewidth=1.8,
                    markersize=5.5,
                    label=method,
                )
                axis.fill_between(
                    fractions * 100,
                    low,
                    high,
                    color=color,
                    alpha=0.10,
                    linewidth=0,
                )
            axis.set_xlabel("Retained measurements (%)")
            axis.set_ylabel(r"Mean relative $L_2$ error")
            axis.set_xticks(fractions * 100)
            axis.set_ylim(*common_ylim)
            axis.legend(frameon=False, fontsize=8.5)
            save_figure(
                figure,
                output
                / "04_reconstruction_metrics"
                / f"{condition}_{group_name}_relative_l2_versus_retention.png",
            )


def reconstruction_path(condition: str, fraction: float, method: str) -> Path:
    directory, special = METHOD_PATHS[method]
    tag = f"{condition}_test_keep{fraction:.2f}_seed20260802"
    if special == "step":
        tag += f"_step{ITR_STEPS[(condition, fraction)]}"
    return ROOT / "results/reconstruction/mnist_medium_v1" / directory / f"{tag}.npz"


def plot_itr_step_selection(output: Path) -> None:
    """Save one validation step-size curve per condition and retention."""
    root = ROOT / "results/reconstruction/mnist_medium_v1/iterated_time_reversal"
    directory = output / "04_reconstruction_metrics" / "itr_step_selection"
    for condition in CONDITIONS:
        color = "#2563EB" if condition == "periodic_theta89" else "#DC2626"
        for fraction in RETENTION_FRACTIONS:
            steps = np.asarray(ITR_VALIDATION_STEPS[(condition, fraction)])
            errors = []
            for step in steps:
                tag = (
                    f"{condition}_validation_keep{fraction:.2f}_seed20260802_"
                    f"step{step:g}_metrics.json"
                )
                metrics = json.loads((root / tag).read_text(encoding="utf-8"))
                errors.append(float(metrics["final_relative_l2_mean"]))
            errors = np.asarray(errors)
            selected = float(ITR_STEPS[(condition, fraction)])
            selected_index = int(np.flatnonzero(np.isclose(steps, selected))[0])
            figure, axis = plt.subplots(figsize=(5.6, 4.2))
            axis.plot(steps, errors, color=color, marker="o", linewidth=2.0)
            axis.scatter(
                steps[selected_index],
                errors[selected_index],
                marker="*",
                s=180,
                color=color,
                edgecolor="black",
                linewidth=0.8,
                zorder=4,
                label=rf"selected $\alpha={selected:g}$",
            )
            axis.set_xlabel(r"Step size $\alpha$")
            axis.set_ylabel(r"Validation relative $L_2$ error")
            axis.set_yscale("log")
            axis.set_xticks(steps)
            axis.legend(frameon=False)
            save_figure(figure, directory / f"{condition}_keep{fraction:.2f}_step_selection.png")


def plot_convergence_panels(output: Path) -> None:
    """Plot four retention levels in one publication-ready row."""
    fractions = RETENTION_FRACTIONS
    specs = (
        ("relative_l2", r"Mean relative $L_2$ error"),
        ("residual", "Mean measurement residual"),
    )
    for condition in CONDITIONS:
        for metric, ylabel in specs:
            histories = {}
            for fraction in fractions:
                histories[fraction] = {}
                for method, label in (
                    ("Gradient descent (1/L)", "GD"),
                    ("Iterated time reversal", "Iterated TR"),
                ):
                    with np.load(reconstruction_path(condition, fraction, method)) as saved:
                        key = "error_history" if metric == "relative_l2" else "residual_history"
                        values = np.asarray(saved[key], dtype=float)
                        histories[fraction][label] = (
                            np.mean(values, axis=0),
                            np.percentile(values, 25, axis=0),
                            np.percentile(values, 75, axis=0),
                        )

            for scale in ("semilog", "loglog"):
                figure, axes = plt.subplots(
                    1, 4, figsize=(12.0, 3.15), sharey=True, constrained_layout=True
                )
                for axis, fraction in zip(axes, fractions, strict=False):
                    for label, (mean, low, high) in histories[fraction].items():
                        iterations = np.arange(len(mean))
                        valid = iterations > 0 if scale == "loglog" else iterations >= 0
                        x = iterations[valid]
                        axis.plot(x, mean[valid], color=METHOD_COLORS[label], label=label)
                        axis.fill_between(
                            x,
                            low[valid],
                            high[valid],
                            color=METHOD_COLORS[label],
                            alpha=0.13,
                            linewidth=0,
                        )
                    axis.set_title(f"{fraction:.0%} retained")
                    axis.set_xlabel("Iteration")
                    axis.set_yscale("log")
                    if scale == "loglog":
                        axis.set_xscale("log")
                    axis.grid(True, which="both", alpha=0.20)
                axes[0].set_ylabel(ylabel)
                handles, labels = axes[0].get_legend_handles_labels()
                figure.legend(
                    handles,
                    labels,
                    loc="upper center",
                    ncol=2,
                    frameon=False,
                    bbox_to_anchor=(0.5, 1.08),
                )
                name = f"{condition}_{metric}_{scale}_all_retention.png"
                save_figure(
                    figure,
                    output / "05_convergence" / "retention_panels" / name,
                )


def plot_convergence(output: Path) -> None:
    for condition in CONDITIONS:
        for fraction in RETENTION_FRACTIONS:
            histories: dict[str, dict[str, np.ndarray]] = {}
            for method, label in (
                ("Gradient descent (1/L)", "GD"),
                ("Iterated time reversal", "Iterated TR"),
            ):
                with np.load(reconstruction_path(condition, fraction, method)) as saved:
                    histories[label] = {
                        "residual": np.mean(saved["residual_history"], axis=0),
                        "relative_l2": np.mean(saved["error_history"], axis=0),
                    }
            for metric, ylabel in (
                ("residual", "Mean data residual"),
                ("relative_l2", r"Mean relative $L_2$ error"),
            ):
                length = min(len(item[metric]) for item in histories.values())
                iterations = np.arange(1, length)
                series = [(label, values[metric][1:length]) for label, values in histories.items()]
                base = f"{condition}_keep{fraction:.2f}_{metric}"
                title = (
                    f"GD and iterated TR: {CONDITION_LABELS[condition]}, {fraction:.0%} retained"
                )
                line_figure(
                    iterations,
                    series,
                    "Iteration",
                    ylabel,
                    title,
                    output / "05_convergence" / "semilog" / f"{base}_semilog.png",
                    yscale="log",
                )
                line_figure(
                    iterations,
                    series,
                    "Iteration (log scale)",
                    ylabel,
                    title,
                    output / "05_convergence" / "loglog" / f"{base}_loglog.png",
                    xscale="log",
                    yscale="log",
                )


def _large_forward_arrays(condition: str) -> tuple[np.ndarray, np.ndarray]:
    analytical, target = [], []
    root = ROOT / "datasets/mnist_large_v1" / condition
    for shard in sorted(root.glob("test_*.h5")):
        with h5py.File(shard, "r") as handle:
            analytical.append(handle["data_fft"][...])
            target.append(handle["kwave_forward"][...])
    return np.concatenate(analytical), np.concatenate(target)


def _sample_relative_l2(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    difference = (prediction - target).reshape(len(target), -1)
    denominator = np.linalg.norm(target.reshape(len(target), -1), axis=1)
    return np.linalg.norm(difference, axis=1) / np.maximum(denominator, np.finfo(float).eps)


def _large_initial_pressure(sample_index: int) -> np.ndarray:
    """Load one initial-pressure image without concatenating the large split."""
    remaining = sample_index
    root = ROOT / "datasets/mnist_large_v1/periodic_theta89"
    for shard in sorted(root.glob("test_*.h5")):
        with h5py.File(shard, "r") as handle:
            count = int(handle["p0"].shape[0])
            if remaining < count:
                return np.asarray(handle["p0"][remaining])
            remaining -= count
    raise IndexError(f"Large-test sample index out of range: {sample_index}")


def plot_forward_error_ecdf(output: Path) -> None:
    labels = ("Fourier", "FNO-only", "Fourier-to-FNO", "FNO-to-Fourier")
    scenarios = (None, *(scenario for scenario, _ in LEARNED_FORWARD_SCENARIOS))
    colors = ("#6B7280", "#2563EB", "#059669", "#0891B2")
    for condition in CONDITIONS:
        analytical, target = _large_forward_arrays(condition)
        distributions = []
        for scenario in scenarios:
            if scenario is None:
                prediction = analytical
                reference = target
            else:
                with np.load(learned_prediction_path(condition, scenario)) as saved:
                    prediction = saved["prediction"]
                    reference = saved["target"]
            distributions.append(_sample_relative_l2(prediction, reference))
        figure, axis = plt.subplots(figsize=(7.2, 4.8))
        for label, values, color in zip(labels, distributions, colors, strict=False):
            ordered = np.sort(values)
            cumulative = np.arange(1, len(ordered) + 1) / len(ordered)
            axis.plot(ordered, cumulative, label=label, color=color)
        axis.set_xlabel(r"Per-sample relative $L_2$ error")
        axis.set_ylabel("Empirical cumulative probability")
        axis.set_title(f"Distribution of forward errors: {CONDITION_LABELS[condition]}")
        axis.set_xlim(left=0)
        axis.set_ylim(0, 1.01)
        axis.legend(frameon=False, loc="lower right")
        save_figure(
            figure,
            output / "01_forward_accuracy" / f"{condition}_forward_error_ecdf.png",
        )


def plot_forward_prediction_examples(output: Path, sample_index: int) -> None:
    """Save each representative large-test pressure field independently."""
    methods = (
        ("fourier", None),
        ("fno_only", "fno_only"),
        ("fourier_to_fno", "fourier_to_fno"),
        ("fno_to_fourier", "fno_to_fourier"),
        ("kwave_reference", "reference"),
    )
    directory = output / "01_forward_accuracy" / "forward_prediction_examples"
    initial_pressure = _large_initial_pressure(sample_index)
    figure, axis = plt.subplots(figsize=(4.0, 3.5))
    axis.imshow(initial_pressure, cmap="viridis", vmin=0.0, vmax=1.0, origin="lower")
    axis.set_xlabel("$x$ index")
    axis.set_ylabel("$y$ index")
    save_figure(figure, directory / "initial_pressure.png")
    for condition in CONDITIONS:
        analytical, target, learned = load_forward_sample(condition, sample_index)
        fields: dict[str, np.ndarray] = {
            "fourier": analytical,
            "kwave_reference": target,
        }
        for name, scenario in methods:
            if scenario in (None, "reference"):
                continue
            fields[name] = learned[scenario]
        lower = min(float(field.min()) for field in fields.values())
        upper = max(float(field.max()) for field in fields.values())
        for name, _ in methods:
            figure, axis = plt.subplots(figsize=(4.0, 3.5))
            axis.imshow(
                fields[name],
                cmap="viridis",
                vmin=lower,
                vmax=upper,
                aspect="auto",
                origin="upper",
            )
            axis.set_xlabel("Time index")
            axis.set_ylabel("Sensor index")
            save_figure(figure, directory / f"{condition}_{name}.png")


def plot_forward_error_maps(output: Path, sample_index: int) -> None:
    """Save large-test absolute forward-error maps as independent panels."""
    directory = output / "01_forward_accuracy" / "forward_error_maps"
    directory.mkdir(parents=True, exist_ok=True)
    methods = (
        ("fourier", None),
        ("fno_only", "fno_only"),
        ("fourier_to_fno", "fourier_to_fno"),
        ("fno_to_fourier", "fno_to_fourier"),
    )
    errors: dict[tuple[str, str], np.ndarray] = {}
    for condition in CONDITIONS:
        analytical, reference, learned = load_forward_sample(condition, sample_index)
        errors[(condition, "fourier")] = np.abs(analytical - reference)
        for name, scenario in methods:
            if scenario is None:
                continue
            errors[(condition, name)] = np.abs(learned[scenario] - reference)

    vmax = float(np.percentile(np.concatenate([error.ravel() for error in errors.values()]), 99))
    colorbar_figure, colorbar_axis = plt.subplots(figsize=(0.8, 3.5))
    scalar_mappable = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=0.0, vmax=vmax), cmap="magma")
    colorbar_figure.colorbar(
        scalar_mappable,
        cax=colorbar_axis,
        orientation="vertical",
        label="Absolute pressure error",
    )
    colorbar_figure.savefig(
        directory / "absolute_pressure_error_colorbar.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.03,
    )
    colorbar_figure.savefig(
        directory / "absolute_pressure_error_colorbar.svg",
        bbox_inches="tight",
        pad_inches=0.03,
    )
    plt.close(colorbar_figure)
    for condition in CONDITIONS:
        for name, _ in methods:
            figure, axis = plt.subplots(figsize=(4.0, 3.5))
            axis.imshow(
                errors[(condition, name)],
                cmap="magma",
                vmin=0.0,
                vmax=vmax,
                aspect="auto",
                origin="lower",
            )
            axis.set_xlabel("Time index")
            axis.set_ylabel("Sensor index")
            save_figure(figure, directory / f"{condition}_{name}_absolute_error.png")


def plot_forward_sensor_traces(output: Path, sample_index: int) -> None:
    """Save central-sensor large-test pressure traces independently."""
    directory = output / "01_forward_accuracy" / "forward_sensor_traces"
    methods = (
        ("Fourier", None, "#7F7F7F", "--", 2.0),
        ("FNO-only", "fno_only", "#1F77B4", "-", 2.0),
        ("Fourier-to-FNO", "fourier_to_fno", "#D62728", "-", 2.0),
        ("FNO-to-Fourier", "fno_to_fourier", "#2CA02C", "-", 2.0),
        ("k-Wave reference", "reference", "#000000", "-", 2.4),
    )
    for condition in CONDITIONS:
        analytical, target, learned = load_forward_sample(condition, sample_index)
        sensor_index = int(target.shape[0] // 2)
        traces: dict[str, np.ndarray] = {
            "Fourier": np.asarray(analytical[sensor_index]),
            "k-Wave reference": np.asarray(target[sensor_index]),
        }
        for label, scenario, _, _, _ in methods:
            if scenario in (None, "reference"):
                continue
            traces[label] = np.asarray(learned[scenario][sensor_index])

        figure, axis = plt.subplots(figsize=(7.2, 4.6))
        for label, _, color, linestyle, linewidth in methods:
            axis.plot(
                np.arange(traces[label].size),
                traces[label],
                label=label,
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
            )
        axis.set_xlabel("Time index")
        axis.set_ylabel("Pressure amplitude")
        axis.legend(frameon=False, ncol=2, loc="upper right")
        save_figure(figure, directory / f"{condition}_central_sensor_trace.png")


def plot_reconstruction_keep025_comparison(output: Path) -> None:
    rows = read_csv(
        ROOT
        / "results/evaluation/mnist_medium_v1"
        / "required_experiments/reconstruction_with_uncertainty.csv"
    )
    methods = list(METHOD_PATHS)
    short_labels = (
        "Fourier",
        "TR",
        "Iterated TR",
        "GD (1/L)",
        "FNO-only",
        "Fourier-to-FNO",
        "FNO-to-Fourier",
    )
    figure, axis = plt.subplots(figsize=(8.6, 4.9))
    x = np.arange(len(methods))
    for condition, marker, color in (
        ("periodic_theta89", "o", "#2563EB"),
        ("pml_outside_theta45", "s", "#DC2626"),
    ):
        values = []
        for method in methods:
            row = next(
                item
                for item in rows
                if item["condition"] == condition
                and np.isclose(float(item["keep_fraction"]), 0.25)
                and item["method"] == method
            )
            values.append(float(row["relative_l2_mean"]))
        axis.plot(x, values, marker=marker, color=color, label=CONDITION_LABELS[condition])
        for position, value in zip(x, values, strict=False):
            axis.annotate(
                f"{value:.2f}",
                (position, value),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color=color,
            )
    axis.set_xticks(x, short_labels, rotation=18, ha="right")
    axis.set_ylabel(r"Mean reconstruction relative $L_2$")
    axis.set_title("Reconstruction comparison at 25% measurement retention")
    axis.legend(frameon=False)
    save_figure(figure, output / "04_reconstruction_metrics" / "keep0.25_method_comparison.png")


def plot_acquisition_geometry(output: Path) -> None:
    """Draw the mirrored computational domain and detector interface."""
    from matplotlib.patches import Arc, Rectangle

    specs = (
        ("periodic_theta89", 89, "Periodic boundary", False),
        ("pml_outside_theta45", 45, "Exterior PML", True),
    )
    for condition, angle, boundary_label, has_pml in specs:
        figure, axis = plt.subplots(figsize=(5.2, 4.4))
        axis.set_aspect("equal")
        axis.set_xlim(-1.35, 1.35)
        axis.set_ylim(-1.25, 1.25)
        if has_pml:
            axis.add_patch(
                Rectangle(
                    (-1.16, -1.08),
                    2.32,
                    2.16,
                    facecolor="#E5E7EB",
                    edgecolor="#6B7280",
                    linewidth=1.2,
                    hatch="///",
                    label="Exterior PML",
                )
            )
        axis.add_patch(
            Rectangle(
                (-1.0, -1.0),
                2.0,
                2.0,
                facecolor="#F8FAFC",
                edgecolor="#111827",
                linewidth=1.5,
            )
        )
        axis.add_patch(
            Rectangle(
                (-0.78, -0.82),
                1.56,
                0.65,
                facecolor="#DBEAFE",
                edgecolor="#2563EB",
                linewidth=1.2,
            )
        )
        axis.text(0, -0.50, r"Source domain $p_0$", ha="center", va="center")
        axis.add_patch(
            Rectangle(
                (-0.78, 0.17),
                1.56,
                0.65,
                facecolor="#ECFDF5",
                edgecolor="#059669",
                linewidth=1.2,
                linestyle="--",
            )
        )
        axis.text(0, 0.50, "Mirrored source", ha="center", va="center")
        axis.plot([-1, 1], [0, 0], color="#DC2626", linewidth=3)
        sensor_x = np.linspace(-0.92, 0.92, 16)
        axis.scatter(sensor_x, np.zeros_like(sensor_x), s=18, color="#DC2626", zorder=5)
        axis.text(
            0,
            0.07,
            "64-point detector line",
            ha="center",
            va="bottom",
            color="#991B1B",
            fontsize=9,
        )
        axis.add_patch(
            Arc(
                (0, 0),
                1.10,
                1.10,
                theta1=-angle,
                theta2=0,
                color="#D97706",
                linewidth=2,
            )
        )
        radius = 0.55
        axis.plot(
            [0, radius * np.cos(np.deg2rad(angle))],
            [0, -radius * np.sin(np.deg2rad(angle))],
            color="#D97706",
            linewidth=1.5,
        )
        axis.plot([0, radius], [0, 0], color="#D97706", linewidth=1.5)
        axis.text(
            0.42,
            -0.22 if angle > 60 else -0.13,
            rf"$\theta_{{\max}}={angle}^\circ$",
            color="#92400E",
            fontsize=10,
        )
        axis.text(0, 1.14, boundary_label, ha="center", va="center", fontsize=10)
        axis.set_title(f"Acquisition geometry: {CONDITION_LABELS[condition]}")
        axis.axis("off")
        save_figure(figure, output / "00_method_diagrams" / f"{condition}_geometry.png")


def plot_fno_block(output: Path) -> None:
    """Draw the implemented FNO layer as a publication-ready block diagram."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    figure, axis = plt.subplots(figsize=(11.2, 3.1))
    axis.set_xlim(0, 12)
    axis.set_ylim(0, 4)
    axis.axis("off")

    def box(x, y, w, h, label, color):
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.04",
            facecolor=color,
            edgecolor="#374151",
            linewidth=1.2,
        )
        axis.add_patch(patch)
        axis.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=10)

    def arrow(x1, y1, x2, y2):
        axis.add_patch(
            FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                arrowstyle="-|>",
                mutation_scale=12,
                color="#374151",
                linewidth=1.2,
            )
        )

    box(0.15, 1.45, 1.15, 1.0, "Input\nchannels", "#F3F4F6")
    box(1.65, 1.45, 1.10, 1.0, r"$1\times1$ lifting", "#DBEAFE")
    arrow(1.30, 1.95, 1.65, 1.95)
    box(3.25, 2.55, 1.05, 0.85, "2-D FFT", "#FEF3C7")
    box(4.70, 2.55, 1.45, 0.85, r"Retain modes\nand multiply $R_\ell$", "#FDE68A")
    box(6.55, 2.55, 1.15, 0.85, "Inverse FFT", "#FEF3C7")
    box(4.70, 0.55, 1.45, 0.85, r"Local map $W_\ell$", "#EDE9FE")
    arrow(2.75, 1.95, 3.25, 2.95)
    arrow(2.75, 1.95, 4.70, 0.98)
    arrow(4.30, 2.98, 4.70, 2.98)
    arrow(6.15, 2.98, 6.55, 2.98)
    box(8.15, 1.45, 0.85, 1.0, "Sum", "#DCFCE7")
    arrow(7.70, 2.98, 8.15, 2.12)
    arrow(6.15, 0.98, 8.15, 1.78)
    box(9.40, 1.45, 0.95, 1.0, "GELU", "#D1FAE5")
    arrow(9.00, 1.95, 9.40, 1.95)
    box(10.75, 1.45, 1.05, 1.0, r"$1\times1$ projection", "#DBEAFE")
    arrow(10.35, 1.95, 10.75, 1.95)
    axis.text(
        5.45,
        3.68,
        r"Spectral branch: $\mathcal{F}^{-1}(R_\ell\mathcal{F}(v_\ell))$",
        ha="center",
        fontsize=11,
    )
    axis.text(5.45, 0.20, r"Pointwise branch: $W_\ell v_\ell$", ha="center", fontsize=11)
    axis.set_title("Implemented Fourier neural operator layer")
    save_figure(figure, output / "00_method_diagrams" / "fno_spectral_layer.png")


def plot_reconstruction_keep025_correlation(output: Path) -> None:
    rows = read_csv(
        ROOT
        / "results/evaluation/mnist_medium_v1"
        / "required_experiments/reconstruction_with_uncertainty.csv"
    )
    methods = list(METHOD_PATHS)
    short = (
        "Fourier",
        "TR",
        "Iterated TR",
        "GD (1/L)",
        "FNO-only",
        "Fourier-to-FNO",
        "FNO-to-Fourier",
    )
    figure, axis = plt.subplots(figsize=(8.6, 4.9))
    x = np.arange(len(methods))
    for condition, marker, color in (
        ("periodic_theta89", "o", "#2563EB"),
        ("pml_outside_theta45", "s", "#DC2626"),
    ):
        values = []
        for method in methods:
            row = next(
                item
                for item in rows
                if item["condition"] == condition
                and np.isclose(float(item["keep_fraction"]), 0.25)
                and item["method"] == method
            )
            values.append(float(row["correlation_mean"]))
        axis.plot(x, values, marker=marker, color=color, label=CONDITION_LABELS[condition])
        for position, value in zip(x, values, strict=False):
            axis.annotate(
                f"{value:.2f}",
                (position, value),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color=color,
            )
    axis.set_xticks(x, short, rotation=18, ha="right")
    axis.set_ylabel("Mean reconstruction correlation")
    axis.set_title("Structural agreement at 25% measurement retention")
    axis.set_ylim(0, 1)
    axis.legend(frameon=False)
    save_figure(
        figure,
        output / "04_reconstruction_metrics" / "keep0.25_correlation_comparison.png",
    )


def _load_reconstruction_sample(condition: str, fraction: float, sample_index: int):
    arrays = {}
    truth = None
    for method in METHOD_PATHS:
        with np.load(reconstruction_path(condition, fraction, method)) as saved:
            arrays[method] = np.asarray(saved["reconstruction"][sample_index])
            if truth is None:
                truth = np.asarray(saved["p0"][sample_index])
    return truth, arrays


def plot_complete_reconstruction_montages(output: Path, sample_index: int) -> None:
    """Save every reconstruction as an independent, untitled image."""
    order = ("Ground truth", *METHOD_PATHS.keys())
    short = {
        "Ground truth": "Ground truth",
        "Fourier inverse": "Fourier inverse",
        "Time reversal": "Time reversal",
        "Iterated time reversal": "Iterated TR",
        "Gradient descent (1/L)": "Gradient descent",
        "FNO-only": "FNO-only",
        "Fourier-to-FNO": "Fourier-to-FNO",
        "FNO-to-Fourier": "FNO-to-Fourier",
    }
    for condition in CONDITIONS:
        truth, arrays = _load_reconstruction_sample(condition, 0.25, sample_index)
        images = {"Ground truth": truth, **arrays}
        directory = output / "06_reconstruction_examples" / condition / "keep0.25" / "individual"
        for key in order:
            figure, axis = plt.subplots(figsize=(3.2, 3.2))
            axis.imshow(images[key], cmap="gray", vmin=0, vmax=1, origin="lower")
            axis.axis("off")
            filename = short[key].lower().replace(" ", "_").replace("-", "_")
            save_figure(figure, directory / f"{filename}.png")


def plot_retention_montages(output: Path, sample_index: int) -> None:
    """Save one reconstruction per method, condition, and retention level."""
    fractions = RETENTION_FRACTIONS
    methods = (
        ("Gradient descent (1/L)", "Gradient descent"),
        ("Iterated time reversal", "Iterated TR"),
    )
    for condition in CONDITIONS:
        for method, label in methods:
            for fraction in fractions:
                with np.load(reconstruction_path(condition, fraction, method)) as saved:
                    image = np.asarray(saved["reconstruction"][sample_index])
                figure, axis = plt.subplots(figsize=(3.2, 3.2))
                axis.imshow(image, cmap="gray", vmin=0, vmax=1, origin="lower")
                axis.axis("off")
                method_name = label.lower().replace(" ", "_")
                save_figure(
                    figure,
                    output
                    / "07_retention_examples"
                    / condition
                    / f"{method_name}_keep{fraction:.2f}.png",
                )


def write_readme(output: Path) -> None:
    """Document the generated figure-directory structure."""
    text = """# Dissertation figure set

All figures in this directory are generated by
`python -m scripts.evaluation.generate_thesis_figures` from saved result files.
No checkpoint or experiment result is modified. Each PNG is a publication-ready
plot, diagram, or comparison montage.

- `01_forward_accuracy`: pure Fourier and three learned forward operators.
- `02_sample_efficiency`: five training-set sizes for FNO-only, Fourier-to-FNO,
  and FNO-to-Fourier, reported as mean plus or minus one standard deviation over
  three independent training seeds.
- `03_runtime`: per-sample runtime on a logarithmic vertical scale.
- `04_reconstruction_metrics`: retention robustness with bootstrap 95% intervals.
- `05_convergence/semilog`: convergence against a linear iteration axis.
- `05_convergence/loglog`: convergence with logarithmic iteration and error axes.
- `06_reconstruction_examples`: complete ground-truth and seven-method comparison montages.

Runtime plots distinguish hardware-controlled CPU latency from practical RTX
4090 latency and throughput. Sample-efficiency error bars quantify variation
across three independent training seeds at every tested training-set size.
"""
    (output / "README.md").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse command-line options for structured figure generation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/evaluation/thesis_figures",
        help="New figure-set directory.",
    )
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove only the selected output directory before regeneration.",
    )
    return parser.parse_args()


def generate_figure_set(output: Path, sample_index: int) -> None:
    """Generate the complete structured thesis figure set."""
    setup_style()
    # Method diagrams in 00_method_diagrams are curated separately.
    plot_forward_error_ecdf(output)
    plot_forward_prediction_examples(output, sample_index)
    plot_forward_error_maps(output, sample_index)
    plot_forward_sensor_traces(output, sample_index)
    plot_training_curves(output)
    plot_sample_efficiency(output)
    plot_runtime(output)
    plot_reconstruction_metrics(output)
    plot_reconstruction_keep025_comparison(output)
    plot_reconstruction_keep025_correlation(output)
    plot_convergence(output)
    # Legacy multi-panel convergence files are preserved but not regenerated.
    plot_complete_reconstruction_montages(output, sample_index)
    plot_retention_montages(output, sample_index)
    write_readme(output)


def prepare_output_directory(output: Path, clean: bool) -> None:
    """Create the output directory, optionally removing its existing contents."""
    if clean and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)


def main() -> None:
    """Generate all configured figures from saved experiment results."""
    args = parse_args()
    prepare_output_directory(args.output, args.clean)
    generate_figure_set(args.output, args.sample_index)
    print(f"Saved structured thesis figures in {args.output}")


if __name__ == "__main__":
    main()
