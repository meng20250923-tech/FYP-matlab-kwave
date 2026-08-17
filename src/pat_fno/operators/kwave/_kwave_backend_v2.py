"""Python k-Wave compatibility layer with collision-free, single-threaded calls.

Dataset generation uses process-level parallelism.  The upstream solver otherwise
chooses all host CPUs *inside every worker*, so the explicit one-thread default is
essential to avoid oversubscription.  Individual callers may pass ``NumThreads``.
"""

from __future__ import annotations

import os
import tempfile
import uuid
from typing import Any

import numpy as np


def get_setting(setting: Any, name: str, default: Any = None) -> Any:
    return setting.get(name, default) if isinstance(setting, dict) else getattr(setting, name, default)


def require_kwave() -> tuple[type, Any]:
    try:
        from kwave.kgrid import kWaveGrid
        from kwave.kmedium import kWaveMedium
        from kwave.ksensor import kSensor
        from kwave.ksource import kSource
        from kwave.kspaceFirstOrder2D import kspaceFirstOrder2D as native_solver
        from kwave.options.simulation_execution_options import SimulationExecutionOptions
        from kwave.options.simulation_options import SimulationOptions
    except ImportError as exc:
        raise ImportError("Python k-Wave is required. Activate the environment that contains the 'kwave' package before running KWaveOperator2D.") from exc

    class LegacyGrid:
        def __init__(self, nx: int, dx: float, ny: int, dy: float):
            self.native = kWaveGrid(N=(nx, ny), spacing=(dx, dy)); self.dx = dx; self.dy = dy

        def setTime(self, nt: int, dt: float) -> None:
            self.dt = dt; self.native.setTime(nt, dt)

    def legacy_solver(grid: LegacyGrid, medium: dict, source: dict, sensor: dict, **options: Any) -> Any:
        pml_inside = bool(options.get("PMLInside", False)); pml_size = options.get("PMLSize", 20)
        if np.isscalar(pml_size): pml_size = [int(pml_size), int(pml_size)]
        if options.get("PMLSize", None) == 0: pml_size = [0, 0]
        scratch = options.get("TempPath", tempfile.gettempdir()); os.makedirs(scratch, exist_ok=True)
        token = f"{os.getpid()}_{uuid.uuid4().hex}"
        simulation_options = SimulationOptions(
            data_cast=options.get("DataCast", "single"), pml_inside=pml_inside, pml_size=pml_size, smooth_p0=False,
            save_to_disk=True, data_path=scratch, input_filename=f"kwave_input_{token}.h5", output_filename=f"kwave_output_{token}.h5",
        )
        execution_options = SimulationExecutionOptions(
            is_gpu_simulation=False, delete_data=True, verbose_level=0, show_sim_log=False,
            num_threads=int(options.get("NumThreads", 1)),
        )
        native_medium = kWaveMedium(sound_speed=medium["sound_speed"], density=medium.get("density", 1))
        native_source = kSource()
        for field, value in source.items(): setattr(native_source, field, value)
        native_sensor = kSensor(mask=np.asarray(sensor["mask"]), record=sensor.get("record", ["p"]))
        if "time_reversal_boundary_data" in sensor: native_sensor.time_reversal_boundary_data = sensor["time_reversal_boundary_data"]
        try:
            return native_solver(kgrid=grid.native, medium=native_medium, source=native_source, sensor=native_sensor,
                                 simulation_options=simulation_options, execution_options=execution_options)
        finally:
            for filename in (simulation_options.input_filename, simulation_options.output_filename):
                path = os.path.join(scratch, filename)
                if os.path.exists(path): os.remove(path)

    return LegacyGrid, legacy_solver
