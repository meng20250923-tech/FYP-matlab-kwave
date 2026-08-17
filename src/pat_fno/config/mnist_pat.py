"""Frozen physical and dataset settings for the Test4 MNIST PAT experiments."""

from __future__ import annotations

GRID_SIZE = 64
DX = DY = 1e-4
SOUND_SPEED = 1500.0
MEDIUM_DENSITY = 1000.0
CFL = 1.0
SEED = 20260728
SHARD_SIZE = 250
SPLITS = {"train": 5000, "validation": 1000, "test": 1000}
CONDITIONS = {
    "periodic_theta89": {"boundary": "periodic", "theta_deg": 89.0},
    "pml_outside_theta45": {"boundary": "pml", "theta_deg": 45.0},
}
