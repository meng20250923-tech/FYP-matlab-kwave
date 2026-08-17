"""Unit tests for the Python k-Wave operator wrappers."""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pytest

from pat_fno.operators.kwave import (
    kwave_adjoint_2d,
    kwave_forward_2d,
    kwave_inverse_2d,
)

forward_module = importlib.import_module("pat_fno.operators.kwave.forward")
inverse_module = importlib.import_module("pat_fno.operators.kwave.inverse")
adjoint_module = importlib.import_module("pat_fno.operators.kwave.adjoint")

CALLS: list[dict[str, Any]] = []


class FakeGrid:
    """Minimal grid used to test wrapper behaviour."""

    def __init__(
        self,
        nx: int,
        dx: float,
        ny: int,
        dy: float,
    ) -> None:
        self.nx = nx
        self.dx = dx
        self.ny = ny
        self.dy = dy
        self.nt = 0
        self.dt = 0.0

    def setTime(self, nt: int, dt: float) -> None:
        """Store the temporal grid."""
        self.nt = nt
        self.dt = dt


def fake_solver(
    grid: FakeGrid,
    medium: dict[str, np.ndarray],
    source: dict[str, Any],
    sensor: dict[str, Any],
    **options: Any,
) -> dict[str, np.ndarray]:
    """Record one solver call and return deterministic arrays."""
    CALLS.append(
        {
            "grid": grid,
            "medium": medium,
            "source": {
                key: np.array(value, copy=True) if isinstance(value, np.ndarray) else value
                for key, value in source.items()
            },
            "sensor": sensor,
            "options": options,
        }
    )

    if "p0" in source and "time_reversal_boundary_data" not in sensor:
        return {"p": np.zeros((grid.nt, grid.ny))}

    return {"p_final": np.zeros((grid.nx, grid.ny))}


def fake_backend() -> tuple[type[FakeGrid], Any]:
    """Return the deterministic test backend."""
    return FakeGrid, fake_solver


@pytest.fixture(autouse=True)
def replace_kwave_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the external solver for every wrapper test."""
    CALLS.clear()
    for module in (
        forward_module,
        inverse_module,
        adjoint_module,
    ):
        monkeypatch.setattr(module, "require_kwave", fake_backend)


def make_setting(boundary: str = "pml") -> dict[str, object]:
    """Return a compact k-Wave configuration."""
    return {
        "Nx": 6,
        "Ny": 5,
        "Nt": 8,
        "dx": 1e-4,
        "dy": 1e-4,
        "dt": 4e-8,
        "soundSpeed": 1500.0,
        "mediumDensity": 1000.0,
        "kwaveBoundary": boundary,
        "pmlSize": 7,
    }


@pytest.mark.parametrize(
    ("boundary", "expected_pml_size"),
    [("periodic", 0), ("pml", 7)],
)
def test_forward_configures_boundary_and_output_shape(
    boundary: str,
    expected_pml_size: int,
) -> None:
    """Forward propagation should configure the requested boundary."""
    output = kwave_forward_2d(
        np.ones((6, 5)),
        make_setting(boundary),
    )

    assert output.shape == (5, 8)
    assert CALLS[-1]["options"]["PMLSize"] == expected_pml_size


def test_inverse_and_adjoint_return_image_fields() -> None:
    """Inverse and adjoint wrappers should return image-shaped arrays."""
    data = np.ones((5, 8))
    setting = make_setting()

    inverse = kwave_inverse_2d(data, setting)
    adjoint = kwave_adjoint_2d(data, setting)

    assert inverse.shape == (6, 5)
    assert adjoint.shape == (6, 5)


def test_adjoint_applies_first_order_source_scaling() -> None:
    """The adjoint should scale its source before solver execution."""
    data = np.arange(40, dtype=float).reshape(5, 8)
    setting = make_setting()

    kwave_adjoint_2d(data, setting)
    source = CALLS[-1]["source"]["p"]

    reversed_data = np.fliplr(data)
    expected = np.pad(reversed_data, ((0, 0), (0, 1)))
    expected += np.pad(reversed_data, ((0, 0), (1, 0)))
    expected[:, -2] += expected[:, -1]
    expected = expected[:, :-1]
    expected *= (
        setting["mediumDensity"] * setting["soundSpeed"] * setting["dx"] / (4 * setting["dt"])
    )

    np.testing.assert_array_equal(source, expected)


def test_forward_rejects_incorrect_image_shape() -> None:
    """Forward propagation should reject an incompatible image grid."""
    with pytest.raises(ValueError, match="p0 size"):
        kwave_forward_2d(
            np.zeros((5, 5)),
            make_setting(),
        )


def test_inverse_rejects_incorrect_data_shape() -> None:
    """Time reversal should reject incompatible sensor data."""
    with pytest.raises(ValueError, match="Ny x Nt"):
        kwave_inverse_2d(
            np.zeros((5, 7)),
            make_setting(),
        )


def test_unknown_boundary_is_rejected() -> None:
    """An unknown boundary name should raise an informative error."""
    with pytest.raises(ValueError, match="kwaveBoundary"):
        kwave_forward_2d(
            np.zeros((6, 5)),
            make_setting("unknown"),
        )
