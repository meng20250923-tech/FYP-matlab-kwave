"""Unit tests for the MNIST PAT experiment configuration."""

import pytest

from pat_fno.config.mnist_pat import (
    CFL,
    CONDITIONS,
    DATASET_SCALES,
    DX,
    DY,
    GRID_SIZE,
    MEDIUM_DENSITY,
    SHARD_SIZE,
    SOUND_SPEED,
    SPLITS,
    get_dataset_scale,
)


def test_physical_configuration_matches_experiments() -> None:
    """Physical constants should match the completed experiments."""
    assert GRID_SIZE == 64
    assert DX == pytest.approx(1e-4)
    assert DY == pytest.approx(1e-4)
    assert SOUND_SPEED == pytest.approx(1500.0)
    assert MEDIUM_DENSITY == pytest.approx(1000.0)
    assert CFL == pytest.approx(1.0)


def test_dataset_scales_match_completed_datasets() -> None:
    """Medium and large split sizes should match the saved datasets."""
    assert DATASET_SCALES["medium"].splits == {
        "train": 5_000,
        "validation": 1_000,
        "test": 1_000,
    }
    assert DATASET_SCALES["large"].splits == {
        "train": 50_000,
        "validation": 5_000,
        "test": 10_000,
    }


def test_legacy_aliases_retain_medium_configuration() -> None:
    """Earlier commands should continue to receive medium defaults."""
    assert SPLITS == {
        "train": 5_000,
        "validation": 1_000,
        "test": 1_000,
    }
    assert SHARD_SIZE == 250


def test_acquisition_conditions_match_experiments() -> None:
    """Both reported acquisition conditions should remain available."""
    assert CONDITIONS["periodic_theta89"] == {
        "boundary": "periodic",
        "theta_deg": 89.0,
    }
    assert CONDITIONS["pml_outside_theta45"] == {
        "boundary": "pml",
        "theta_deg": 45.0,
    }


def test_unknown_dataset_scale_is_rejected() -> None:
    """Unknown scale names should produce an informative error."""
    with pytest.raises(ValueError, match="Unknown dataset scale"):
        get_dataset_scale("unknown")
