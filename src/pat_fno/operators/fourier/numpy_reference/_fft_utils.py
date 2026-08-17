"""Shared numerical helpers for the Python Fourier PAT operators."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def get_setting(setting: Any, name: str, default: Any = None) -> Any:
    """Read a (possibly dotted) field from a dict or MATLAB-like object."""
    value = setting
    for part in name.split("."):
        if isinstance(value, dict):
            if part not in value:
                if default is not None:
                    return default
                raise KeyError(f"Missing setting.{name}")
            value = value[part]
        else:
            if not hasattr(value, part):
                if default is not None:
                    return default
                raise AttributeError(f"Missing setting.{name}")
            value = getattr(value, part)
    return value


def kgrid(nx: int, dx: float, ny: int, dy: float) -> tuple[np.ndarray, np.ndarray]:
    """Return k-Wave-compatible, FFT-shifted ``kx`` and ``ky`` grids."""
    kx_1d = np.fft.fftshift(2.0 * np.pi * np.fft.fftfreq(nx, d=dx))
    ky_1d = np.fft.fftshift(2.0 * np.pi * np.fft.fftfreq(ny, d=dy))
    return np.meshgrid(kx_1d, ky_1d, indexing="ij")


def unitary_fft2(values: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(values))) / math.sqrt(values.size)


def unitary_ifft2(values: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(values))) * math.sqrt(values.size)
