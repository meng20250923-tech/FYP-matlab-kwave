"""Tests for required-experiment aggregation and orchestration helpers."""

from types import SimpleNamespace

import numpy as np
import pytest

from scripts.evaluation import analyze_required_experiments as analysis


def reconstruction_arguments(**overrides):
    """Return the minimal argument namespace used by reconstruction helpers."""
    values = {
        "keep_fractions": [0.10, 0.25, 0.50, 1.00],
        "periodic_itr_step": "1.5",
        "pml_itr_step": "2",
        "periodic_full_retention_itr_step": "0.75",
        "pml_full_retention_itr_step": "1.75",
        "periodic_itr_steps": ["2", "1.5", "1.5", "0.75"],
        "pml_itr_steps": ["2.5", "2", "2.5", "1.75"],
        "allow_missing": False,
        "bootstrap_draws": 10,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_retention_specific_itr_steps_are_selected_by_condition() -> None:
    """Each condition and retention must use its independently selected step."""
    arguments = reconstruction_arguments()
    steps = analysis.retention_specific_itr_steps(arguments)

    assert analysis.select_itr_step(arguments, "periodic_theta89", 0.10, 0, steps) == "2"
    assert analysis.select_itr_step(arguments, "periodic_theta89", 1.00, 3, steps) == "0.75"
    assert analysis.select_itr_step(arguments, "pml_outside_theta45", 0.50, 2, steps) == "2.5"


@pytest.mark.parametrize(
    ("periodic_steps", "pml_steps"),
    ((None, ["2.5", "2", "2.5", "1.75"]), (["2", "1.5"], ["2.5", "2"])),
)
def test_invalid_retention_specific_itr_steps_are_rejected(
    periodic_steps: list[str] | None,
    pml_steps: list[str] | None,
) -> None:
    """Both step lists must be present and match the retention count."""
    arguments = reconstruction_arguments(
        periodic_itr_steps=periodic_steps,
        pml_itr_steps=pml_steps,
    )
    with pytest.raises(ValueError):
        analysis.retention_specific_itr_steps(arguments)


def test_sample_efficiency_aggregation_preserves_seed_statistics() -> None:
    """Aggregation must retain the existing mean and sample-SD definitions."""
    rows = [
        {
            "condition": "periodic_theta89",
            "model": "FNO-only",
            "train_samples": 1000,
            "seed": 1,
            "rel_l2_mean": 0.2,
        },
        {
            "condition": "periodic_theta89",
            "model": "FNO-only",
            "train_samples": 1000,
            "seed": 2,
            "rel_l2_mean": 0.4,
        },
    ]

    aggregate = analysis.aggregate_sample_efficiency(rows)

    assert aggregate == [
        {
            "condition": "periodic_theta89",
            "model": "FNO-only",
            "train_samples": 1000,
            "runs": 2,
            "rel_l2_mean": pytest.approx(0.3),
            "rel_l2_std_across_seeds": pytest.approx(np.std([0.2, 0.4], ddof=1)),
        }
    ]


def test_reconstruction_bootstrap_call_order_is_stable(tmp_path, monkeypatch) -> None:
    """Bootstrap calls must remain method-major and metric-minor."""
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    first.touch()
    second.touch()
    markers = {first: 1.0, second: 2.0}
    calls = []

    def fake_metrics(path):
        marker = markers[path]
        return {
            "relative_l2": np.asarray([marker * 10]),
            "correlation": np.asarray([marker * 20]),
            "mse": np.asarray([marker * 30]),
        }

    def fake_bootstrap(values, rng, draws):
        del rng, draws
        calls.append(float(values[0]))
        return float(values[0]), float(values[0])

    monkeypatch.setattr(analysis, "per_sample_metrics", fake_metrics)
    monkeypatch.setattr(analysis, "bootstrap_ci", fake_bootstrap)
    rows = []

    analysis.append_reconstruction_metrics(
        {"first": first, "second": second},
        "periodic_theta89",
        0.25,
        20260802,
        reconstruction_arguments(),
        np.random.default_rng(20260805),
        rows,
    )

    assert calls == [10.0, 20.0, 30.0, 20.0, 40.0, 60.0]
    assert [row["method"] for row in rows] == ["first", "second"]
