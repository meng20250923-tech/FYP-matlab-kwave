"""Regression tests for forward-operator runtime benchmarking."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.evaluation import benchmark_forward_operators as benchmark


class ScalingModel(torch.nn.Module):
    """Apply a fixed scale while recording the model input."""

    def __init__(self):
        super().__init__()
        self.inputs = []

    def forward(self, values):
        self.inputs.append(values.clone())
        return values * 0.5


def make_arguments(**overrides):
    """Construct a complete benchmark argument namespace."""
    values = {
        "dataset": "example",
        "condition": "periodic_theta89",
        "split": "test",
        "checkpoint_root": None,
        "device": "cpu",
        "batch_size": 2,
        "max_samples": 3,
        "warmup": 2,
        "repeats": 3,
        "include_kwave": True,
        "overwrite": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_time_callable_preserves_warmup_repeat_and_sync_order(monkeypatch):
    """Warm-up calls precede one initial and one per-repeat synchronization."""
    events = []
    clock = iter((0.0, 0.25, 1.0, 1.25, 2.0, 2.25))
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(
        benchmark,
        "synchronize",
        lambda device: events.append(("synchronize", device.type)),
    )

    durations = benchmark.time_callable(
        lambda: events.append(("call", None)),
        warmup=2,
        repeats=3,
        device=torch.device("cpu"),
    )

    assert durations == [0.25, 0.25, 0.25]
    assert events == [
        ("call", None),
        ("call", None),
        ("synchronize", "cpu"),
        ("call", None),
        ("synchronize", "cpu"),
        ("call", None),
        ("synchronize", "cpu"),
        ("call", None),
        ("synchronize", "cpu"),
    ]


@pytest.mark.parametrize(
    ("scenario", "expected"),
    (
        ("fno_only", torch.full((2, 2, 3), 0.9)),
        ("fourier_to_fno", torch.full((2, 2, 3), 1.2)),
        ("fno_to_fourier", torch.full((2, 2, 3), 2.18)),
    ),
)
def test_learned_pipeline_preserves_scenario_composition(scenario, expected, monkeypatch):
    """Each learned scenario must retain its preprocessing and operator order."""
    p0 = torch.ones((2, 2, 3))
    model = ScalingModel()
    setting = SimpleNamespace()
    normalization = {"p0_mean": 0.1, "p0_std": 0.5, "data_mean": 0.2, "data_std": 0.75}
    monkeypatch.setattr(benchmark, "_resize_p0", lambda values, _shape: values)
    monkeypatch.setattr(benchmark, "fourier", lambda values, _setting: values * 2.0)

    pipeline = benchmark.learned_pipeline(
        scenario,
        p0,
        setting,
        (2, 3),
        normalization,
        model,
    )

    assert torch.allclose(pipeline(), expected)
    assert len(model.inputs) == 1


def test_add_comparisons_preserves_output_field_order():
    """CSV field order must retain speed-up before acquisition condition."""
    rows = [
        {"method": "Fourier", "mean_ms_per_sample": 2.0},
        {"method": "k-Wave", "mean_ms_per_sample": 10.0},
    ]

    benchmark.add_comparisons(rows, "periodic_theta89")

    assert rows[0]["speedup_vs_kwave"] == 5.0
    assert rows[1]["speedup_vs_kwave"] == 1.0
    assert list(rows[0])[-2:] == ["speedup_vs_kwave", "condition"]
    assert rows[0]["condition"] == "periodic_theta89"


def test_benchmark_condition_preserves_operator_order(tmp_path, monkeypatch):
    """One condition must benchmark Fourier, learned models, then k-Wave."""
    events = []
    arrays = {
        "p0": np.zeros((5, 3, 4), dtype=np.float64),
        "kwave_forward": np.zeros((5, 3, 4), dtype=np.float32),
    }
    args = make_arguments(batch_size=3, max_samples=2)
    monkeypatch.setattr(benchmark, "ROOT", tmp_path)
    monkeypatch.setattr(benchmark, "load_arrays", lambda *_args: arrays)
    monkeypatch.setattr(benchmark, "build_setting", lambda condition: condition)
    monkeypatch.setattr(
        benchmark,
        "benchmark_fourier",
        lambda p0, setting, current_args, device, count: (
            events.append(("Fourier", tuple(p0.shape), setting, count))
            or {"method": "Fourier", "mean_ms_per_sample": 2.0}
        ),
    )
    monkeypatch.setattr(
        benchmark,
        "benchmark_learned_operators",
        lambda *_args: (
            events.append(("learned",)) or [{"method": "fno_only", "mean_ms_per_sample": 1.0}]
        ),
    )
    monkeypatch.setattr(
        benchmark,
        "benchmark_kwave",
        lambda p0, setting, current_args, count: (
            events.append(("k-Wave", p0.dtype, tuple(p0.shape), count))
            or {"method": "k-Wave", "mean_ms_per_sample": 4.0}
        ),
    )
    monkeypatch.setattr(
        benchmark,
        "write_benchmark",
        lambda output_root, output_json, condition, device, metadata, rows: events.append(
            ("write", [row["method"] for row in rows], metadata["batch_size"])
        ),
    )

    benchmark.benchmark_condition(
        args,
        "periodic_theta89",
        torch.device("cpu"),
        tmp_path / "checkpoints",
        tmp_path / "runtime",
    )

    assert events == [
        ("Fourier", (2, 3, 4), "periodic_theta89", 2),
        ("learned",),
        ("k-Wave", np.dtype("float32"), (2, 3, 4), 2),
        ("write", ["Fourier", "fno_only", "k-Wave"], 2),
    ]
