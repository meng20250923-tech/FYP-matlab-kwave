"""Tests for forward-operator training utilities."""

from __future__ import annotations

import random
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

from pat_fno.data.mnist import load_arrays, write_shard
from scripts.train import train_forward_operator as training


def make_arrays() -> dict[str, np.ndarray]:
    """Create a small paired forward-operator dataset."""
    p0 = np.arange(2 * 4 * 4, dtype=np.float32).reshape(2, 4, 4)
    fourier = np.arange(
        2 * 3 * 5,
        dtype=np.float32,
    ).reshape(2, 3, 5)
    kwave = fourier + 2.0

    return {
        "p0": p0,
        "data_fft": fourier,
        "kwave_forward": kwave,
        "label": np.array([3, 8], dtype=np.int64),
        "source_index": np.array([10, 20], dtype=np.int64),
    }


def test_cpu_device_selection() -> None:
    """An explicit CPU request should return a CPU device."""
    assert training._device("cpu") == torch.device("cpu")


def test_seed_reproduces_python_numpy_and_torch_sequences() -> None:
    """The training seed should control every random-number source."""
    training._seed(1234)
    first = (
        random.random(),
        np.random.random(),
        torch.rand(1).item(),
    )

    training._seed(1234)
    second = (
        random.random(),
        np.random.random(),
        torch.rand(1).item(),
    )

    assert first == second


def test_forward_dataset_fno_only_resizes_image_input() -> None:
    """FNO-only should receive a resized normalised pressure image."""
    arrays = make_arrays()
    dataset = training.ForwardDataset(
        arrays,
        scenario="fno_only",
        p0_mean=2.0,
        p0_std=4.0,
        data_mean=3.0,
        data_std=5.0,
    )

    source, target, p0 = dataset[0]

    expected_source = training._resize_p0(
        torch.from_numpy(arrays["p0"][:1]),
        (3, 5),
    )[0]
    expected_source = (expected_source - 2.0) / 4.0

    torch.testing.assert_close(source, expected_source)
    torch.testing.assert_close(
        target,
        (torch.from_numpy(arrays["kwave_forward"][0]) - 3.0) / 5.0,
    )
    torch.testing.assert_close(
        p0,
        torch.from_numpy(arrays["p0"][0]),
    )


def test_forward_dataset_hybrid_sources() -> None:
    """Hybrid scenarios should select their intended source domains."""
    arrays = make_arrays()

    fourier_to_fno = training.ForwardDataset(
        arrays,
        scenario="fourier_to_fno",
        p0_mean=2.0,
        p0_std=4.0,
        data_mean=3.0,
        data_std=5.0,
    )
    fno_to_fourier = training.ForwardDataset(
        arrays,
        scenario="fno_to_fourier",
        p0_mean=2.0,
        p0_std=4.0,
        data_mean=3.0,
        data_std=5.0,
    )

    fourier_source, _, _ = fourier_to_fno[0]
    image_source, _, _ = fno_to_fourier[0]

    torch.testing.assert_close(
        fourier_source,
        (torch.from_numpy(arrays["data_fft"][0]) - 3.0) / 5.0,
    )
    torch.testing.assert_close(
        image_source,
        (torch.from_numpy(arrays["p0"][0]) - 2.0) / 4.0,
    )


def test_compact_loader_matches_shared_loader(tmp_path) -> None:
    """The compact loader should match common fields from the full loader."""
    condition = "periodic_theta89"
    condition_root = tmp_path / condition
    condition_root.mkdir()

    arrays = make_arrays()
    raw = arrays["data_fft"] / 1500.0

    write_shard(
        condition_root / "train_00000_00002.h5",
        arrays["p0"],
        raw,
        arrays["data_fft"],
        arrays["kwave_forward"],
        arrays["label"],
        arrays["source_index"],
        {"condition": condition, "split": "train"},
    )

    full = load_arrays(tmp_path, condition, "train")
    compact = training.load_arrays_compact(
        tmp_path,
        condition,
        "train",
    )

    assert set(compact) == {
        "p0",
        "data_fft",
        "kwave_forward",
        "label",
        "source_index",
    }
    for name in compact:
        np.testing.assert_array_equal(compact[name], full[name])


class IdentityModel(torch.nn.Module):
    """Return the supplied single-channel field unchanged."""

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values


def test_evaluate_restores_physical_units() -> None:
    """Evaluation should undo target normalisation before metrics."""
    arrays = make_arrays()
    arrays["data_fft"] = arrays["kwave_forward"].copy()

    dataset = training.ForwardDataset(
        arrays,
        scenario="fourier_to_fno",
        p0_mean=0.0,
        p0_std=1.0,
        data_mean=3.0,
        data_std=5.0,
    )
    loader = DataLoader(dataset, batch_size=2, shuffle=False)

    result = training._evaluate(
        IdentityModel(),
        loader,
        scenario="fourier_to_fno",
        setting=SimpleNamespace(),
        data_mean=3.0,
        data_std=5.0,
        device=torch.device("cpu"),
        return_arrays=True,
    )

    assert result["normalized_mse"] == 0.0
    assert result["rel_l2_mean"] == 0.0
    assert result["mse"] == 0.0
    np.testing.assert_array_equal(
        result["prediction"],
        arrays["kwave_forward"],
    )
    np.testing.assert_array_equal(
        result["target"],
        arrays["kwave_forward"],
    )


def test_fno_to_fourier_prediction_is_normalised(monkeypatch) -> None:
    """The Fourier branch should be normalised in the data domain."""
    physical = torch.full((2, 3, 5), 13.0)

    def fake_forward(model, p0, source, setting):
        return physical

    monkeypatch.setattr(
        training,
        "_FORWARD_FOURIER",
        fake_forward,
    )

    prediction = training._model_prediction(
        IdentityModel(),
        scenario="fno_to_fourier",
        source=torch.zeros((2, 4, 4)),
        p0=torch.zeros((2, 4, 4)),
        setting=SimpleNamespace(),
        data_mean=3.0,
        data_std=5.0,
    )

    torch.testing.assert_close(
        prediction,
        torch.full((2, 3, 5), 2.0),
    )
