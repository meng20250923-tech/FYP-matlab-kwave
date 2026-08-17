"""Python translation of the mirrored-grid k-Wave forward PAT operator."""

from __future__ import annotations

import numpy as np

from ._kwave_backend_v2 import get_setting, require_kwave


def kSpaceForwardKWave2D(p0: np.ndarray, setting: object) -> np.ndarray:
    nx, ny = get_setting(setting, "Nx"), get_setting(setting, "Ny")
    if p0.shape != (nx, ny):
        raise ValueError("p0 size must be setting.Nx x setting.Ny.")
    kWaveGrid, kspaceFirstOrder2D = require_kwave()
    dx, dy, nt, dt = (get_setting(setting, key) for key in ("dx", "dy", "Nt", "dt"))
    c, rho = get_setting(setting, "soundSpeed"), get_setting(setting, "mediumDensity", 1000)
    grid = kWaveGrid(2 * nx, dx, ny, dy)
    grid.setTime(nt, dt)
    medium = {"sound_speed": c * np.ones((2 * nx, ny)), "density": rho * np.ones((2 * nx, ny))}
    source = {"p0": np.vstack((np.flipud(p0), p0))}
    mask = np.zeros((2 * nx, ny), dtype=bool)
    mask[nx, :] = True
    sensor = {"mask": mask}
    boundary = get_setting(setting, "kwaveBoundary", "pml")
    options = {"PMLInside": False, "PlotSim": False, "DataCast": "single"}
    if boundary == "periodic":
        options.update(PMLSize=0, PMLAlpha=0)
    elif boundary == "pml":
        pml_size = get_setting(setting, "pmlSize")
        if pml_size is not None:
            options["PMLSize"] = pml_size
    else:
        raise ValueError("setting.kwaveBoundary must be 'pml' or 'periodic'.")
    return np.asarray(dict.__getitem__(kspaceFirstOrder2D(grid, medium, source, sensor, **options), chr(112)), dtype=float).T
