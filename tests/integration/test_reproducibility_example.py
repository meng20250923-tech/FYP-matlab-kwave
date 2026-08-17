"""Integration test for the fixed repository reproducibility example."""

from pathlib import Path

import numpy as np
import pytest

from scripts.smoke.run_reproducibility_check import (
    _assert_array_difference,
    _build_summary,
    run_check,
)


@pytest.mark.slow
def test_reproducibility_example(tmp_path: Path):
    summary = run_check("smoke/reproducibility.yaml", tmp_path / "summary.json")
    assert summary["status"] == "PASS"
    assert summary["sample_count"] == 4
    assert summary["shapes"] == {
        "initial_pressure": [4, 64, 64],
        "pressure_data": [4, 64, 91],
        "reconstruction": [4, 64, 64],
    }


def test_array_validation_returns_exact_maximum() -> None:
    """Array validation reports the maximum absolute discrepancy."""
    actual = np.array([0.0, 1.5, -2.0])
    expected = np.array([0.0, 1.0, -1.75])

    difference = _assert_array_difference("difference", actual, expected, 0.5)

    assert difference == 0.5
    with pytest.raises(AssertionError, match="difference 0.5"):
        _assert_array_difference("difference", actual, expected, 0.49)


def test_summary_schema_is_stable() -> None:
    """Summary construction retains the documented output fields and shapes."""
    samples = {
        "source_index": np.array([3, 7]),
        "initial_pressure": np.zeros((2, 4, 4)),
        "kwave_target": np.zeros((2, 4, 5)),
    }
    summary = _build_summary(
        {"condition": "periodic_theta89", "sample_count": 2, "retention": 0.25},
        samples,
        np.zeros((2, 4, 4)),
        {"forward": {"mse": 0.0}},
        {"forward_seconds": 0.1},
        0.25,
        0.0,
    )

    assert list(summary) == [
        "status",
        "condition",
        "sample_count",
        "source_indices",
        "requested_retention",
        "actual_retention",
        "maximum_fourier_reference_difference",
        "metrics",
        "timings",
        "shapes",
    ]
    assert summary["source_indices"] == [3, 7]
    assert summary["shapes"]["pressure_data"] == [2, 4, 5]
