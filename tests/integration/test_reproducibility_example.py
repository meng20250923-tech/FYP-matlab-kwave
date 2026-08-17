"""Integration test for the fixed repository reproducibility example."""

from pathlib import Path

import pytest

from scripts.smoke.run_reproducibility_check import run_check


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
