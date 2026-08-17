"""Tests for deterministic measurement subsampling."""

from __future__ import annotations

import argparse

import h5py
import numpy as np
import pytest

from scripts.reconstruction import subsample_measurements as subsampling


def make_args(**overrides: object) -> argparse.Namespace:
    """Return a complete argument namespace with optional overrides."""
    values = {
        "dataset": "example_dataset",
        "condition": "periodic_theta89",
        "split": "test",
        "keep_fraction": 0.25,
        "seed": 17,
        "keep_fractions": None,
        "seeds": None,
        "overwrite": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_resolve_requests_preserves_plural_order() -> None:
    """Explicit fractions and seeds retain their requested order."""
    args = make_args(keep_fractions=[0.5, 0.1, 1.0], seeds=[9, 3])

    fractions, seeds = subsampling.resolve_requests(args)

    assert fractions == [0.5, 0.1, 1.0]
    assert seeds == [9, 3]


@pytest.mark.parametrize("fraction", [0.0, -0.1, 1.01])
def test_resolve_requests_rejects_invalid_fraction(fraction: float) -> None:
    """Retention fractions must lie in the documented interval."""
    with pytest.raises(ValueError, match="retention fractions"):
        subsampling.resolve_requests(make_args(keep_fraction=fraction))


def test_write_subsampled_measurements_preserves_schema(tmp_path) -> None:
    """One output file contains the expected arrays, dtypes, and metadata."""
    full_data = np.arange(12, dtype=np.float64).reshape(2, 2, 3)
    uniforms = np.array(
        [
            [[0.05, 0.30, 0.10], [0.90, 0.20, 0.70]],
            [[0.24, 0.25, 0.26], [0.00, 0.99, 0.15]],
        ]
    )
    arrays = {
        "p0": np.arange(8, dtype=np.float32).reshape(2, 2, 2),
        "label": np.array([3, 8]),
        "source_index": np.array([11, 17]),
    }
    path = tmp_path / "masked.h5"

    actual = subsampling.write_subsampled_measurements(
        path,
        arrays,
        full_data,
        uniforms,
        0.25,
        17,
        "example_dataset",
        "periodic_theta89",
        "test",
    )

    expected_mask = uniforms < 0.25
    expected_data = np.where(expected_mask, full_data, 0.0).astype(np.float32)
    assert actual == float(expected_mask.mean())
    with h5py.File(path, "r") as handle:
        np.testing.assert_array_equal(handle["mask"], expected_mask)
        np.testing.assert_array_equal(handle["observed_data"], expected_data)
        np.testing.assert_array_equal(handle["p0"], arrays["p0"])
        np.testing.assert_array_equal(handle["label"], arrays["label"])
        np.testing.assert_array_equal(handle["source_index"], arrays["source_index"])
        assert handle["observed_data"].dtype == np.dtype(np.float32)
        assert handle.attrs["dataset"] == "example_dataset"
        assert handle.attrs["condition"] == "periodic_theta89"
        assert handle.attrs["split"] == "test"
        assert handle.attrs["keep_fraction_requested"] == 0.25
        assert handle.attrs["keep_fraction_actual"] == actual
        assert handle.attrs["seed"] == 17
        assert handle.attrs["source_measurement"] == "kwave_forward"
        assert bool(handle.attrs["nested_mask_design"])


def test_generate_seed_outputs_draws_once_and_reuses_uniforms(monkeypatch, tmp_path) -> None:
    """Every fraction for one seed uses the same uniform random array."""
    full_data = np.zeros((2, 3, 4), dtype=np.float32)
    expected_uniforms = np.linspace(0.0, 1.0, full_data.size).reshape(full_data.shape)
    random_calls: list[tuple[int, tuple[int, ...]]] = []
    writes: list[tuple[float, np.ndarray]] = []

    class FakeGenerator:
        def random(self, shape: tuple[int, ...]) -> np.ndarray:
            random_calls.append((23, shape))
            return expected_uniforms

    monkeypatch.setattr(
        subsampling.np.random,
        "default_rng",
        lambda seed: FakeGenerator(),
    )

    def record_write(
        path,
        arrays,
        data,
        uniforms,
        keep_fraction,
        seed,
        dataset,
        condition,
        split,
    ) -> float:
        writes.append((keep_fraction, uniforms))
        return float((uniforms < keep_fraction).mean())

    monkeypatch.setattr(subsampling, "write_subsampled_measurements", record_write)
    args = make_args(overwrite=True)

    subsampling.generate_seed_outputs(
        args,
        tmp_path,
        "periodic_theta89",
        {},
        full_data,
        [0.10, 0.25, 1.00],
        23,
    )

    assert random_calls == [(23, full_data.shape)]
    assert [fraction for fraction, _ in writes] == [0.10, 0.25, 1.00]
    assert all(uniforms is expected_uniforms for _, uniforms in writes)


def test_generate_seed_outputs_skips_existing_file_after_random_draw(monkeypatch, tmp_path) -> None:
    """Existing outputs are skipped without changing the RNG draw location."""
    args = make_args(overwrite=False)
    path = subsampling.output_path(tmp_path, args.condition, args.split, 0.25, 17)
    path.touch()
    draws: list[tuple[int, ...]] = []

    class FakeGenerator:
        def random(self, shape: tuple[int, ...]) -> np.ndarray:
            draws.append(shape)
            return np.zeros(shape)

    monkeypatch.setattr(subsampling.np.random, "default_rng", lambda seed: FakeGenerator())
    monkeypatch.setattr(
        subsampling,
        "write_subsampled_measurements",
        lambda *args, **kwargs: pytest.fail("Existing output was overwritten."),
    )

    subsampling.generate_seed_outputs(
        args,
        tmp_path,
        args.condition,
        {},
        np.zeros((2, 3), dtype=np.float32),
        [0.25],
        17,
    )

    assert draws == [(2, 3)]
