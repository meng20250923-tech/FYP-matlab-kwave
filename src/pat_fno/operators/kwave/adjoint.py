"""Python translation of the mirrored-grid k-Wave adjoint PAT operator."""

from __future__ import annotations

import numpy as np

from ._backend import get_setting, require_kwave


def kwave_adjoint_2d(data: np.ndarray, setting: object) -> np.ndarray:
    nx, ny, nt = (get_setting(setting, key) for key in ("Nx", "Ny", "Nt"))
    if data.shape != (ny, nt):
        raise ValueError("data must have size Ny x Nt.")
    kWaveGrid, kspaceFirstOrder2D = require_kwave()
    dx, dy, dt = (get_setting(setting, key) for key in ("dx", "dy", "dt"))
    c, rho = get_setting(setting, "soundSpeed"), get_setting(setting, "mediumDensity", 1000)
    grid = kWaveGrid(2 * nx, dx, ny, dy)
    grid.setTime(nt, dt)
    mask = np.zeros((2 * nx, ny), dtype=bool)
    mask[nx, :] = True
    reversed_data = np.fliplr(data)
    source_data = np.pad(reversed_data, ((0, 0), (0, 1))) + np.pad(reversed_data, ((0, 0), (1, 0)))
    source_data[:, -2] += source_data[:, -1]
    source_data = source_data[:, :-1] * (rho * c * dx / (4 * dt))
    medium = {"sound_speed": c * np.ones((2 * nx, ny)), "density": rho * np.ones((2 * nx, ny))}
    source = {"p": source_data, "p_mask": mask, "p_mode": "additive"}
    options = {"PMLInside": False, "PlotSim": False, "DataCast": "single"}
    boundary = get_setting(setting, "kwaveBoundary", "pml")
    if boundary == "periodic":
        options.update(PMLSize=0, PMLAlpha=0)
    elif boundary == "pml":
        pml_size = get_setting(setting, "pmlSize")
        if pml_size is not None:
            options["PMLSize"] = pml_size
    else:
        raise ValueError("setting.kwaveBoundary must be 'pml' or 'periodic'.")
    result = kspaceFirstOrder2D(grid, medium, source, {"mask": mask, "record": ["p_final"]}, **options)
    full = np.asarray(result["p_final"] if isinstance(result, dict) else result.p_final, dtype=float) / (rho * c**2)
    if full.shape == (ny, 2 * nx):
        full = full.T
    if full.shape == (ny, nx):
        return full.T
    if full.shape == (nx, ny):
        return full
    return full[nx:, :] + np.flipud(full[:nx, :])
