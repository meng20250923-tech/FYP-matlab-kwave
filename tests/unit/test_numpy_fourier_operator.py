"""Unit tests for the NumPy Fourier PAT operators."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pat_fno.operators.fourier import (
    numpy_adjoint_2d,
    numpy_forward_2d,
    numpy_inverse_2d,
)


def make_setting(
    forward_method: str = "linear",
    reconstruction_method: str = "linear",
) -> dict[str, object]:
    """Return a small dictionary-style PAT configuration."""
    return {
        "Nx": 12,
        "Ny": 12,
        "Nt": 14,
        "dx": 1e-4,
        "dy": 1e-4,
        "dt": 4e-8,
        "soundSpeed": 1500.0,
        "computation": {
            "theta_max": math.radians(89.0),
            "interpolationMethodF": forward_method,
            "interpolationMethodI": reconstruction_method,
            "interpolationMethodA": reconstruction_method,
        },
    }


@pytest.mark.parametrize(
    "method",
    ["nearest", "linear", "cubic", "trig"],
)
def test_forward_interpolation_methods_return_finite_data(
    method: str,
) -> None:
    """Each supported interpolation method should produce sensor data."""
    rng = np.random.default_rng(20260817)
    pressure = rng.standard_normal((12, 12))

    output = numpy_forward_2d(
        pressure,
        make_setting(forward_method=method),
    )

    assert output.shape == (12, 14)
    assert np.isfinite(output).all()


def test_forward_rejects_unknown_interpolation_method() -> None:
    """An unsupported interpolation method should raise an error."""
    pressure = np.zeros((12, 12))

    with pytest.raises(ValueError, match="Unsupported SciPy"):
        numpy_forward_2d(
            pressure,
            make_setting(forward_method="unknown"),
        )


def test_inverse_and_reference_adjoint_return_finite_images() -> None:
    """Inverse and reference-adjoint operators should reconstruct images."""
    rng = np.random.default_rng(20260817)
    pressure = rng.standard_normal((12, 12))
    setting = make_setting()

    data = numpy_forward_2d(pressure, setting)
    inverse = numpy_inverse_2d(data, setting)
    adjoint = numpy_adjoint_2d(data, setting)

    assert inverse.shape == pressure.shape
    assert adjoint.shape == pressure.shape
    assert np.isfinite(inverse).all()
    assert np.isfinite(adjoint).all()


def test_dictionary_and_attribute_settings_are_equivalent() -> None:
    """Dictionary and MATLAB-style attribute settings should agree."""

    class Namespace:
        """Minimal attribute container used by the compatibility test."""

    computation = Namespace()
    computation.theta_max = math.radians(89.0)
    computation.interpolationMethodF = "linear"
    computation.interpolationMethodI = "linear"
    computation.interpolationMethodA = "linear"

    attribute_setting = Namespace()
    attribute_setting.Nx = 12
    attribute_setting.Ny = 12
    attribute_setting.Nt = 14
    attribute_setting.dx = 1e-4
    attribute_setting.dy = 1e-4
    attribute_setting.dt = 4e-8
    attribute_setting.soundSpeed = 1500.0
    attribute_setting.computation = computation

    pressure = np.arange(144, dtype=float).reshape(12, 12)
    dictionary_output = numpy_forward_2d(
        pressure,
        make_setting(),
    )
    attribute_output = numpy_forward_2d(
        pressure,
        attribute_setting,
    )

    np.testing.assert_array_equal(
        dictionary_output,
        attribute_output,
    )
