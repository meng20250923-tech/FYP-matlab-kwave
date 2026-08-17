"""Run the repository's fixed small-scale PAT reproducibility example."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from pat_fno.config import load_experiment_config
from pat_fno.data.mnist import batch_metrics, build_setting
from pat_fno.models import TinyFNO2d
from pat_fno.operators.fourier import numpy_forward_2d, numpy_inverse_2d

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
REQUIRED_SAMPLE_FIELDS = {
    "initial_pressure",
    "kwave_target",
    "fourier_prediction",
    "mask_25",
    "observed_pressure",
    "label",
    "source_index",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the reproducibility example."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="smoke/reproducibility.yaml",
        help="Configuration path relative to the repository configs directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional summary path overriding the configured output.",
    )
    return parser.parse_args()


def _repository_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def _load_samples(path: Path, expected_count: int) -> dict[str, np.ndarray]:
    """Load and validate the fixed sample input archive."""
    with np.load(path) as archive:
        fields = set(archive.files)
        if fields != REQUIRED_SAMPLE_FIELDS:
            raise ValueError(
                f"Unexpected sample fields: missing={sorted(REQUIRED_SAMPLE_FIELDS - fields)}, "
                f"unknown={sorted(fields - REQUIRED_SAMPLE_FIELDS)}."
            )
        samples = {name: archive[name] for name in archive.files}

    expected_shapes = {
        "initial_pressure": (expected_count, 64, 64),
        "kwave_target": (expected_count, 64, 91),
        "fourier_prediction": (expected_count, 64, 91),
        "mask_25": (expected_count, 64, 91),
        "observed_pressure": (expected_count, 64, 91),
        "label": (expected_count,),
        "source_index": (expected_count,),
    }
    for name, shape in expected_shapes.items():
        if samples[name].shape != shape:
            raise ValueError(f"{name} has shape {samples[name].shape}; expected {shape}.")
        if not np.isfinite(samples[name]).all():
            raise ValueError(f"{name} contains a non-finite value.")
    return samples


def _fourier_predictions(initial_pressure: np.ndarray, condition: str) -> np.ndarray:
    """Evaluate the NumPy Fourier forward operator for the fixed batch."""
    setting = build_setting(condition)
    setting.computation.interpolationMethodF = "cubic"
    return np.stack(
        [setting.soundSpeed * numpy_forward_2d(image, setting) for image in initial_pressure]
    ).astype(np.float32)


def _fno_predictions(initial_pressure: np.ndarray, checkpoint_path: Path) -> np.ndarray:
    """Load the supplied FNO-only checkpoint and evaluate it on CPU."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    arguments = checkpoint["arguments"]
    if arguments["scenario"] != "fno_only":
        raise ValueError("The reproducibility checkpoint must contain an FNO-only model.")
    model = TinyFNO2d(
        arguments["modes"],
        arguments["modes"],
        arguments["width"],
        arguments["layers"],
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()
    normalization = checkpoint["normalization"]
    pressure = torch.from_numpy(initial_pressure.astype(np.float32, copy=False))
    source = F.interpolate(pressure[:, None], size=(64, 91), mode="bilinear", align_corners=False)
    source = (source - normalization["p0_mean"]) / normalization["p0_std"]
    with torch.no_grad():
        prediction = model(source)[:, 0]
        prediction = prediction * normalization["data_std"] + normalization["data_mean"]
    return prediction.numpy()


def _fourier_reconstructions(observed_pressure: np.ndarray, condition: str) -> np.ndarray:
    """Apply the analytical Fourier inverse to the masked measurements."""
    setting = build_setting(condition)
    setting.computation.interpolationMethodI = "cubic"
    return np.stack(
        [
            numpy_inverse_2d(measurement / setting.soundSpeed, setting)
            for measurement in observed_pressure
        ]
    ).astype(np.float32)


def _assert_close(name: str, actual: float, expected: float, tolerance: float) -> None:
    """Require one scalar result to match its fixed reference value."""
    if not np.isclose(actual, expected, atol=tolerance, rtol=0.0):
        raise AssertionError(
            f"{name}={actual:.10g} does not match expected {expected:.10g} "
            f"within absolute tolerance {tolerance:g}."
        )


def _timed_call(operation, *args) -> tuple[Any, float]:
    """Evaluate one operation and return its result and elapsed seconds."""
    start = time.perf_counter()
    result = operation(*args)
    return result, time.perf_counter() - start


def _assert_array_difference(
    message: str,
    actual: np.ndarray,
    expected: np.ndarray,
    tolerance: float,
    *,
    report_difference: bool = True,
) -> float:
    """Validate two arrays and return their maximum absolute difference."""
    maximum_difference = float(np.max(np.abs(actual - expected)))
    if maximum_difference > tolerance:
        detail = f" {maximum_difference:.6g}." if report_difference else ""
        raise AssertionError(f"{message}{detail}")
    return maximum_difference


def _evaluate_operators(
    samples: dict[str, np.ndarray],
    config: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Evaluate the fixed forward and inverse operators with timings."""
    fourier, fourier_seconds = _timed_call(
        _fourier_predictions,
        samples["initial_pressure"],
        config["condition"],
    )
    fno, fno_seconds = _timed_call(
        _fno_predictions,
        samples["initial_pressure"],
        _repository_path(config["checkpoint"]),
    )
    reconstruction, inverse_seconds = _timed_call(
        _fourier_reconstructions,
        samples["observed_pressure"],
        config["condition"],
    )
    predictions = {
        "fourier": fourier,
        "fno": fno,
        "reconstruction": reconstruction,
    }
    timings = {
        "fourier_forward_seconds": fourier_seconds,
        "fno_forward_seconds": fno_seconds,
        "fourier_inverse_seconds": inverse_seconds,
    }
    return predictions, timings


def _compute_metrics(
    samples: dict[str, np.ndarray],
    predictions: dict[str, np.ndarray],
) -> dict[str, dict[str, float]]:
    """Compute the three metric groups reported by the example."""
    return {
        "fourier_forward": batch_metrics(predictions["fourier"], samples["kwave_target"]),
        "fno_only_forward": batch_metrics(predictions["fno"], samples["kwave_target"]),
        "fourier_inverse": batch_metrics(
            predictions["reconstruction"],
            samples["initial_pressure"],
        ),
    }


def _validate_metrics(
    metrics: dict[str, dict[str, float]],
    expected: dict[str, Any],
    actual_retention: float,
    tolerance: float,
) -> None:
    """Compare every reported metric with the versioned expected results."""
    for group, values in metrics.items():
        for metric, value in values.items():
            _assert_close(f"{group}.{metric}", value, expected[group][metric], tolerance)
    _assert_close("actual_retention", actual_retention, expected["actual_retention"], tolerance)


def _build_summary(
    config: dict[str, Any],
    samples: dict[str, np.ndarray],
    reconstruction: np.ndarray,
    metrics: dict[str, dict[str, float]],
    timings: dict[str, float],
    actual_retention: float,
    maximum_difference: float,
) -> dict[str, Any]:
    """Construct the stable machine-readable verification summary."""
    return {
        "status": "PASS",
        "condition": config["condition"],
        "sample_count": config["sample_count"],
        "source_indices": samples["source_index"].astype(int).tolist(),
        "requested_retention": config["retention"],
        "actual_retention": actual_retention,
        "maximum_fourier_reference_difference": maximum_difference,
        "metrics": metrics,
        "timings": timings,
        "shapes": {
            "initial_pressure": list(samples["initial_pressure"].shape),
            "pressure_data": list(samples["kwave_target"].shape),
            "reconstruction": list(reconstruction.shape),
        },
    }


def run_check(config_path: str | Path, output_override: Path | None = None) -> dict[str, Any]:
    """Execute the reproducibility workflow and return its summary."""
    config = load_experiment_config(config_path)
    samples = _load_samples(_repository_path(config["sample_data"]), config["sample_count"])
    expected = json.loads(_repository_path(config["expected_results"]).read_text(encoding="utf-8"))
    predictions, timings = _evaluate_operators(samples, config)
    maximum_difference = _assert_array_difference(
        "Recomputed Fourier predictions differ from the supplied reference by",
        predictions["fourier"],
        samples["fourier_prediction"],
        config["tolerances"]["array_atol"],
    )
    metrics = _compute_metrics(samples, predictions)
    actual_retention = float(samples["mask_25"].mean())
    _assert_array_difference(
        "Observed pressure is inconsistent with the supplied mask.",
        samples["observed_pressure"],
        samples["mask_25"] * samples["kwave_target"],
        config["tolerances"]["array_atol"],
        report_difference=False,
    )
    _validate_metrics(metrics, expected, actual_retention, config["tolerances"]["metric_atol"])
    summary = _build_summary(
        config,
        samples,
        predictions["reconstruction"],
        metrics,
        timings,
        actual_retention,
        maximum_difference,
    )
    output_path = output_override or _repository_path(config["output"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    """Run the configured example and print a concise verification report."""
    args = parse_args()
    summary = run_check(args.config, args.output)
    print("Sample data validation: PASS")
    print("Fourier forward reproduction: PASS")
    print("FNO-only checkpoint evaluation: PASS")
    print("Measurement masking: PASS")
    print("Fourier inverse reconstruction: PASS")
    output_path = args.output or _repository_path(load_experiment_config(args.config)["output"])
    print(f"Summary: {output_path}")
    print(f"Verified samples: {summary['sample_count']}")
    print("Overall reproducibility check: PASS")


if __name__ == "__main__":
    main()
