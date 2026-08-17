"""Limited-angle 2-D Fourier PAT forward operator (MATLAB translation)."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import griddata

from ._utils import get_setting, kgrid, unitary_fft2
from .interpolation import triginterp


def numpy_forward_2d(p: np.ndarray, setting: object) -> np.ndarray:
    method = get_setting(setting, "computation.interpolationMethodF")
    theta_max = get_setting(setting, "computation.theta_max")
    c = get_setting(setting, "soundSpeed")
    nt, ny = get_setting(setting, "Nt"), get_setting(setting, "Ny")
    nx, dx, dy = get_setting(setting, "Nx"), get_setting(setting, "dx"), get_setting(setting, "dy")
    dt = get_setting(setting, "dt")

    data_kx, data_ky = kgrid(2 * nt, c * dt, ny, dy)
    w = c * data_kx
    p0_mask = np.concatenate((np.flipud(p), p), axis=0)
    p0_kx, p0_ky = kgrid(2 * nx, dx, ny, dy)
    p_kxky = unitary_fft2(p0_mask)
    radicand = (w / c) ** 2 - data_ky**2
    kx_new = np.sign(w) * np.sqrt(np.maximum(radicand, 0.0))

    if method == "trig":
        p_wky = np.empty_like(kx_new, dtype=complex)
        for col in range(kx_new.shape[1]):
            p_wky[:, col] = triginterp(kx_new[:, col], p0_kx[:, col], p_kxky[:, col])
    else:
        valid = {"nearest", "linear", "cubic"}
        if method not in valid:
            raise ValueError(f"Unsupported SciPy interpolation method: {method}")
        p_wky = griddata(
            (p0_ky.ravel(), p0_kx.ravel()), p_kxky.ravel(),
            (data_ky, kx_new), method=method, fill_value=np.nan,
        )

    p_wky *= np.sqrt(p_kxky.size / p_wky.size)
    p_wky = np.nan_to_num(p_wky)
    p_wky *= kx_new != 0
    dc_source = (p0_kx == 0) & (p0_ky == 0)
    dc_target = (data_ky == 0) & (w == 0)
    if np.any(dc_source) and np.any(dc_target):
        p_wky[dc_target] = p_kxky[dc_source][0]

    with np.errstate(divide="ignore", invalid="ignore"):
        weight = w / (c**2 * kx_new)
    weight = np.nan_to_num(weight)
    weight[(data_ky == 0) & (w == 0)] = 1.0 / c
    ky_max = np.abs((w / c) * np.sin(theta_max))
    weight[np.abs(data_ky) > ky_max] = 0
    frec = np.real(np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(weight * p_wky))) * np.sqrt(p_wky.size))
    return (np.sqrt(2.0) * frec[frec.shape[0] // 2 :, :]).T
