"""Tests for MNIST preprocessing and dataset shard I/O."""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from pat_fno.data.mnist import (
    build_setting,
    conditions,
    load_arrays,
    to_pressure,
    write_shard,
)


def test_build_setting_matches_experimental_geometry() -> None:
    """Physical settings should reproduce the completed experiments."""
    periodic = build_setting("periodic_theta89")
    pml = build_setting("pml_outside_theta45")

    assert (periodic.Nx, periodic.Ny) == (64, 64)
    assert periodic.Nt == 91
    assert periodic.kwaveBoundary == "periodic"
    assert pml.kwaveBoundary == "pml"
    assert np.rad2deg(periodic.computation.theta_max) == pytest.approx(89.0)
    assert np.rad2deg(pml.computation.theta_max) == pytest.approx(45.0)


def test_condition_selection() -> None:
    """Condition selection should preserve the configured ordering."""
    assert conditions("all") == [
        "periodic_theta89",
        "pml_outside_theta45",
    ]
    assert conditions("periodic_theta89") == ["periodic_theta89"]


def test_to_pressure_resizes_and_normalises_image() -> None:
    """MNIST images should become finite normalised pressure fields."""
    image = np.arange(28 * 28, dtype=np.uint8).reshape(28, 28)

    pressure = to_pressure(image)

    assert pressure.shape == (64, 64)
    assert pressure.dtype == np.float32
    assert np.isfinite(pressure).all()
    assert pressure.min() >= 0.0
    assert pressure.max() == pytest.approx(1.0)


def test_to_pressure_preserves_zero_image() -> None:
    """A zero image should remain zero after pressure preprocessing."""
    pressure = to_pressure(np.zeros((28, 28), dtype=np.uint8))

    assert pressure.shape == (64, 64)
    assert np.count_nonzero(pressure) == 0


def test_shard_round_trip(tmp_path) -> None:
    """A completed HDF5 shard should round-trip without numerical change."""
    condition = "periodic_theta89"
    condition_root = tmp_path / condition
    condition_root.mkdir()

    rng = np.random.default_rng(20260728)
    p0 = rng.random((3, 64, 64), dtype=np.float32)
    raw = rng.standard_normal((3, 64, 91)).astype(np.float32)
    scaled = rng.standard_normal((3, 64, 91)).astype(np.float32)
    kwave = rng.standard_normal((3, 64, 91)).astype(np.float32)
    labels = np.array([1, 4, 8], dtype=np.int64)
    indices = np.array([11, 22, 33], dtype=np.int64)

    path = condition_root / "test_00000_00003.h5"
    write_shard(
        path,
        p0,
        raw,
        scaled,
        kwave,
        labels,
        indices,
        {"condition": condition, "split": "test"},
    )

    arrays = load_arrays(tmp_path, condition, "test")

    np.testing.assert_array_equal(arrays["p0"], p0)
    np.testing.assert_array_equal(arrays["fourier_raw"], raw)
    np.testing.assert_array_equal(arrays["data_fft"], scaled)
    np.testing.assert_array_equal(arrays["kwave_forward"], kwave)
    np.testing.assert_array_equal(arrays["label"], labels)
    np.testing.assert_array_equal(arrays["source_index"], indices)
    assert not path.with_suffix(".partial").exists()


def test_incomplete_shard_is_ignored(tmp_path) -> None:
    """Incomplete shards should not contribute to loaded arrays."""
    condition = "periodic_theta89"
    condition_root = tmp_path / condition
    condition_root.mkdir()

    complete_path = condition_root / "test_00000_00001.h5"
    values = np.ones((1, 64, 64), dtype=np.float32)
    data = np.ones((1, 64, 91), dtype=np.float32)

    write_shard(
        complete_path,
        values,
        data,
        data,
        data,
        np.array([8]),
        np.array([10]),
        {"condition": condition, "split": "test"},
    )

    incomplete_path = condition_root / "test_00001_00002.h5"
    with h5py.File(incomplete_path, "w") as handle:
        handle.attrs["complete"] = False

    arrays = load_arrays(tmp_path, condition, "test")

    assert arrays["p0"].shape[0] == 1
    assert arrays["source_index"].tolist() == [10]


def test_load_arrays_rejects_missing_shards(tmp_path) -> None:
    """A missing split should produce an informative error."""
    with pytest.raises(FileNotFoundError, match="No periodic_theta89/test shards"):
        load_arrays(tmp_path, "periodic_theta89", "test")
