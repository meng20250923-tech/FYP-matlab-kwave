"""Compact two-dimensional Fourier neural operator used in the PAT experiments."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    """Apply a learned channel map to retained two-dimensional Fourier modes.

    Args:
        in_channels: Number of input feature channels.
        out_channels: Number of output feature channels.
        modes1: Maximum number of retained modes along the first spatial axis.
        modes2: Maximum number of retained modes along the second spatial axis.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes1: int,
        modes2: int,
    ) -> None:
        """Initialise the positive- and negative-frequency channel weights."""
        super().__init__()

        if min(in_channels, out_channels, modes1, modes2) <= 0:
            raise ValueError("Channel counts and retained mode counts must be positive.")

        self.modes1 = modes1
        self.modes2 = modes2

        scale = 1.0 / (in_channels * out_channels)
        weight_shape = (in_channels, out_channels, modes1, modes2)
        self.weights_pos = nn.Parameter(scale * torch.randn(*weight_shape, dtype=torch.cfloat))
        self.weights_neg = nn.Parameter(scale * torch.randn(*weight_shape, dtype=torch.cfloat))

    @staticmethod
    def _mul(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Multiply retained Fourier coefficients by learned channel weights."""
        return torch.einsum("bixy,ioxy->boxy", values, weights)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Transform a feature field while preserving its spatial resolution."""
        if values.ndim != 4:
            raise ValueError(
                "SpectralConv2d expects a tensor with shape (batch, channels, height, width)."
            )
        if values.shape[1] != self.weights_pos.shape[0]:
            raise ValueError(
                f"Expected {self.weights_pos.shape[0]} channels, received {values.shape[1]}."
            )

        batch, _, height, width = values.shape
        transformed = torch.fft.rfft2(values)

        output = torch.zeros(
            batch,
            self.weights_pos.shape[1],
            height,
            width // 2 + 1,
            dtype=torch.cfloat,
            device=values.device,
        )

        retained_modes1 = min(self.modes1, height)
        retained_modes2 = min(self.modes2, width // 2 + 1)

        positive_values = transformed[:, :, :retained_modes1, :retained_modes2]
        negative_values = transformed[:, :, -retained_modes1:, :retained_modes2]
        positive_weights = self.weights_pos[:, :, :retained_modes1, :retained_modes2]
        negative_weights = self.weights_neg[:, :, :retained_modes1, :retained_modes2]

        output[:, :, :retained_modes1, :retained_modes2] = self._mul(
            positive_values,
            positive_weights,
        )
        output[:, :, -retained_modes1:, :retained_modes2] = self._mul(
            negative_values,
            negative_weights,
        )

        return torch.fft.irfft2(output, s=(height, width))


class TinyFNO2d(nn.Module):
    """Map a scalar two-dimensional field to another scalar field using an FNO.

    Two normalised coordinate channels are concatenated internally with the
    scalar input. The resulting three-channel tensor is lifted to the latent
    width, transformed by repeated spectral layers, and projected back to one
    output channel.

    Args:
        modes1: Retained Fourier modes along the first spatial axis.
        modes2: Retained Fourier modes along the second spatial axis.
        width: Number of latent feature channels.
        layers: Number of spectral layers.
    """

    def __init__(
        self,
        modes1: int = 12,
        modes2: int = 12,
        width: int = 24,
        layers: int = 4,
    ) -> None:
        """Initialise the lifting, spectral, skip, and projection layers."""
        super().__init__()

        if min(modes1, modes2, width, layers) <= 0:
            raise ValueError("Modes, width, and number of layers must be positive.")

        self.lift = nn.Conv2d(3, width, 1)
        self.spectral = nn.ModuleList(
            [SpectralConv2d(width, width, modes1, modes2) for _ in range(layers)]
        )
        self.skip = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(layers)])
        self.project = nn.Sequential(
            nn.Conv2d(width, 64, 1),
            nn.GELU(),
            nn.Conv2d(64, 1, 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Predict a scalar output field from a scalar input field."""
        if values.ndim != 4:
            raise ValueError("TinyFNO2d expects a tensor with shape (batch, 1, height, width).")
        if values.shape[1] != 1:
            raise ValueError(f"TinyFNO2d expects one input channel, received {values.shape[1]}.")

        batch, _, height, width = values.shape
        coordinate_y = (
            torch.linspace(0, 1, height, device=values.device)
            .view(1, 1, height, 1)
            .expand(batch, 1, height, width)
        )
        coordinate_x = (
            torch.linspace(0, 1, width, device=values.device)
            .view(1, 1, 1, width)
            .expand(batch, 1, height, width)
        )

        state = self.lift(torch.cat((values, coordinate_y, coordinate_x), dim=1))

        for spectral_layer, skip_layer in zip(
            self.spectral,
            self.skip,
            strict=True,
        ):
            state = F.gelu(spectral_layer(state) + skip_layer(state))

        return self.project(state)
