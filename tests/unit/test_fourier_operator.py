"""Unit tests for the differentiable Fourier PAT operators."""

from __future__ import annotations

import math

import pytest
import torch

from pat_fno.operators.fourier import (
    create_k_vector,
    fpat_adjoint_2d,
    fpat_forward_2d,
    fpat_forward_2d_batched,
    fpat_inverse_2d,
)

THETA_MAX = math.radians(89.0)
SOUND_SPEED = 1500.0
SPACING = (1e-4, 1e-4)
TIME_STEP = 4e-8
TIME_SAMPLES = 12


def test_k_vector_has_expected_shape_and_zero_frequency() -> None:
    """The wavenumber grid should contain one centred zero frequency."""
    vector = create_k_vector(16, SPACING[0])

    assert vector.shape == (16,)
    assert vector.dtype == torch.float64
    assert vector[8].item() == pytest.approx(0.0)
    assert torch.count_nonzero(vector == 0.0).item() == 1


def test_single_forward_output_shape_and_finiteness() -> None:
    """A single image should map to a finite sensor-time field."""
    pressure = torch.randn(16, 16, dtype=torch.float64)

    output = fpat_forward_2d(
        pressure,
        THETA_MAX,
        SOUND_SPEED,
        TIME_SAMPLES,
        SPACING,
        TIME_STEP,
    )

    assert output.shape == (16, TIME_SAMPLES)
    assert torch.isfinite(output).all()


def test_batched_forward_matches_stacked_single_calls() -> None:
    """The vectorised forward operator should match individual calls."""
    torch.manual_seed(20260817)
    pressure = torch.randn(2, 16, 16, dtype=torch.float64)

    batched = fpat_forward_2d_batched(
        pressure,
        THETA_MAX,
        SOUND_SPEED,
        TIME_SAMPLES,
        SPACING,
        TIME_STEP,
    )
    stacked = torch.stack(
        [
            fpat_forward_2d(
                item,
                THETA_MAX,
                SOUND_SPEED,
                TIME_SAMPLES,
                SPACING,
                TIME_STEP,
            )
            for item in pressure
        ]
    )

    torch.testing.assert_close(
        batched,
        stacked,
        rtol=1e-12,
        atol=1e-12,
    )


def test_forward_supports_automatic_differentiation() -> None:
    """The forward operator should provide a finite image-space gradient."""
    pressure = torch.randn(
        12,
        12,
        dtype=torch.float64,
        requires_grad=True,
    )

    output = fpat_forward_2d(
        pressure,
        THETA_MAX,
        SOUND_SPEED,
        10,
        SPACING,
        TIME_STEP,
    )
    loss = output.square().mean()
    loss.backward()

    assert pressure.grad is not None
    assert pressure.grad.shape == pressure.shape
    assert torch.isfinite(pressure.grad).all()


def test_inverse_and_paper_adjoint_return_image_fields() -> None:
    """Inverse and paper-adjoint operators should return finite images."""
    data = torch.randn(12, 10, dtype=torch.float64)

    inverse = fpat_inverse_2d(
        data,
        THETA_MAX,
        SOUND_SPEED,
        (12, 12),
        SPACING,
        TIME_STEP,
    )
    adjoint = fpat_adjoint_2d(
        data,
        THETA_MAX,
        SOUND_SPEED,
        (12, 12),
        SPACING,
        TIME_STEP,
    )

    assert inverse.shape == (12, 12)
    assert adjoint.shape == (12, 12)
    assert torch.isfinite(inverse).all()
    assert torch.isfinite(adjoint).all()


def test_batched_forward_rejects_unbatched_input() -> None:
    """The batched interface should reject a tensor without a batch axis."""
    pressure = torch.randn(16, 16)

    with pytest.raises(ValueError, match="batch, Nx, Ny"):
        fpat_forward_2d_batched(
            pressure,
            THETA_MAX,
            SOUND_SPEED,
            TIME_SAMPLES,
            SPACING,
            TIME_STEP,
        )
