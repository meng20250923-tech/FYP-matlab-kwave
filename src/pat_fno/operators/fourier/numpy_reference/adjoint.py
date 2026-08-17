"""Limited-angle 2-D Fourier PAT adjoint operator (MATLAB translation)."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import griddata

from ._utils import get_setting, kgrid, unitary_fft2, unitary_ifft2


def numpy_adjoint_2d(f: np.ndarray, setting: object) -> np.ndarray:
    """Evaluate the NumPy reference Fourier PAT adjoint.

    Args:
        f: Sensor-time pressure data with shape ``(Ny, Nt)``.
        setting: Fourier interpolation, grid, and acquisition settings.

    Returns:
        Adjoint pressure field with shape ``(Nx, Ny)``.
    """
    method = get_setting(setting, "computation.interpolationMethodA")
    theta_max, c = get_setting(setting, "computation.theta_max"), get_setting(setting, "soundSpeed")
    nx, ny = get_setting(setting, "Nx"), get_setting(setting, "Ny")
    dx, dy, dt = get_setting(setting, "dx"), get_setting(setting, "dy"), get_setting(setting, "dt")
    fmask = np.concatenate((np.flipud(f.T), f.T), axis=0) / np.sqrt(2.0)
    data_kx, data_ky = kgrid(fmask.shape[0], c * dt, fmask.shape[1], dy)
    w = c * data_kx
    p_wky = unitary_fft2(fmask)
    p_wky[np.abs(w) < np.abs(c * data_ky)] = 0
    ky_max = np.abs((w / c) * np.sin(theta_max))
    factor = c * np.ones_like(w)
    factor[np.abs(data_ky) > ky_max] = 0
    radicand = (w / c) ** 2 - data_ky**2
    kx_new = np.sign(w) * np.sqrt(np.maximum(radicand, 0.0))
    p0_kx, p0_ky = kgrid(2 * nx, dx, ny, dy)
    recovered = griddata(
        (kx_new.ravel(), data_ky.ravel()),
        (p_wky * factor).ravel(),
        (p0_kx, p0_ky),
        method=method,
        fill_value=0.0,
    )
    recovered *= np.sqrt(p_wky.size / recovered.size)
    p_xy = np.real(unitary_ifft2(recovered))
    return p_xy[nx:, :]
