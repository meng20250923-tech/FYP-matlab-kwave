"""Unit tests for the Fourier Neural Operator implementation."""

import pytest
import torch

from pat_fno.models import TinyFNO2d


def test_thesis_model_parameter_count() -> None:
    """The thesis configuration should contain 100,337 parameters."""
    model = TinyFNO2d(modes1=8, modes2=8, width=16, layers=3)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    assert parameter_count == 100_337


def test_fno_preserves_spatial_shape() -> None:
    """The model should produce one output channel on the input grid."""
    model = TinyFNO2d(modes1=4, modes2=4, width=8, layers=2)
    values = torch.randn(2, 1, 16, 20)

    with torch.no_grad():
        output = model(values)

    assert output.shape == (2, 1, 16, 20)
    assert torch.isfinite(output).all()


def test_fno_rejects_incorrect_channel_count() -> None:
    """The model should reject inputs that are not single-channel fields."""
    model = TinyFNO2d(modes1=4, modes2=4, width=8, layers=2)
    values = torch.randn(2, 2, 16, 20)

    with pytest.raises(ValueError, match="one input channel"):
        model(values)
