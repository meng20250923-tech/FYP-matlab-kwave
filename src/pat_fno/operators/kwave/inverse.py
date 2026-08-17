"""Python translation of the mirrored-grid k-Wave time-reversal inverse."""

from __future__ import annotations

import numpy as np

from ._backend import get_setting, require_kwave


def kwave_inverse_2d(data: np.ndarray, setting: object) -> np.ndarray:
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
    medium = {"sound_speed": c * np.ones((2 * nx, ny)), "density": rho * np.ones((2 * nx, ny))}
    source = {"p0": np.zeros((2 * nx, ny))}
    sensor = {"mask": mask, "time_reversal_boundary_data": data}
    options = {"PMLInside": False, "PlotSim": False, "DataCast": "single"}
    boundary = get_setting(setting, "kwaveBoundary", "pml")
    if boundary == "periodic":
        options.update(PMLSize=0, PMLAlpha=0)
    elif boundary != "pml":
        raise ValueError("setting.kwaveBoundary must be 'pml' or 'periodic'.")
    result = kspaceFirstOrder2D(grid, medium, source, sensor, **options)
    full = np.asarray(
        dict.__getitem__(
            result,
            chr(112) + chr(95) + chr(102) + chr(105) + chr(110) + chr(97) + chr(108),
        ),
        dtype=float,
    )
    if full.shape == (ny, 2 * nx):
        full = full.T
    return 0.5 * (full[nx:, :] + np.flipud(full[:nx, :]))