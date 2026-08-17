"""Physical and dataset settings for the MNIST PAT experiments."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetScale:
    """Configuration of one generated MNIST PAT dataset."""

    dataset: str
    train_samples: int
    validation_samples: int
    test_samples: int
    shard_size: int

    @property
    def splits(self) -> dict[str, int]:
        """Return sample counts indexed by split name."""
        return {
            "train": self.train_samples,
            "validation": self.validation_samples,
            "test": self.test_samples,
        }


GRID_SIZE = 64
DX = 1e-4
DY = 1e-4
SOUND_SPEED = 1500.0
MEDIUM_DENSITY = 1000.0
CFL = 1.0
SEED = 20260728

DATASET_SCALES = {
    "smoke": DatasetScale(
        dataset="mnist_smoke",
        train_samples=8,
        validation_samples=4,
        test_samples=4,
        shard_size=8,
    ),
    "medium": DatasetScale(
        dataset="mnist_medium_v1",
        train_samples=5_000,
        validation_samples=1_000,
        test_samples=1_000,
        shard_size=250,
    ),
    "large": DatasetScale(
        dataset="mnist_large_v1",
        train_samples=50_000,
        validation_samples=5_000,
        test_samples=10_000,
        shard_size=250,
    ),
}

CONDITIONS = {
    "periodic_theta89": {
        "boundary": "periodic",
        "theta_deg": 89.0,
    },
    "pml_outside_theta45": {
        "boundary": "pml",
        "theta_deg": 45.0,
    },
}


def get_dataset_scale(name: str) -> DatasetScale:
    """Return a named dataset configuration.

    Args:
        name: Dataset scale, one of ``smoke``, ``medium``, or ``large``.

    Returns:
        The requested immutable dataset configuration.

    Raises:
        ValueError: If the scale name is unknown.
    """
    try:
        return DATASET_SCALES[name]
    except KeyError as error:
        choices = ", ".join(DATASET_SCALES)
        raise ValueError(f"Unknown dataset scale {name!r}; choose from {choices}.") from error


# Compatibility aliases for commands written before explicit scales were added.
_DEFAULT_SCALE = DATASET_SCALES["medium"]
SHARD_SIZE = _DEFAULT_SCALE.shard_size
SPLITS = _DEFAULT_SCALE.splits
