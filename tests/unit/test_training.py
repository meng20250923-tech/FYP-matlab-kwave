"""Tests for forward-operator training utilities."""

from __future__ import annotations

import random
from pathlib import Path
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


def test_training_subset_matches_sorted_shuffled_prefix() -> None:
    """Sample-efficiency subsets preserve the established deterministic selection."""
    arrays = make_arrays()
    seed = 42
    expected_indices = np.sort(np.random.default_rng(seed).permutation(2)[:1])

    selected = training._select_training_subset(arrays, 1, seed)

    for name, values in arrays.items():
        np.testing.assert_array_equal(selected[name], values[expected_indices])
    assert training._select_training_subset(arrays, None, seed) is arrays


def test_training_subset_rejects_invalid_size() -> None:
    """Requested subsets must be nonempty and no larger than the split."""
    arrays = make_arrays()
    for sample_count in (0, 3):
        with np.testing.assert_raises_regex(ValueError, "train-samples"):
            training._select_training_subset(arrays, sample_count, 42)


def test_normalization_matches_direct_statistics() -> None:
    """Training normalization retains the original epsilon convention."""
    arrays = make_arrays()

    normalization = training._normalization(arrays)

    assert normalization == {
        "p0_mean": float(arrays["p0"].mean()),
        "p0_std": float(arrays["p0"].std() + 1e-6),
        "data_mean": float(arrays["kwave_forward"].mean()),
        "data_std": float(arrays["kwave_forward"].std() + 1e-6),
    }


def test_result_root_preserves_experiment_layout(monkeypatch, tmp_path) -> None:
    """Default, sample-efficiency, and explicit outputs retain stable locations."""
    monkeypatch.setattr(training, "ROOT", tmp_path)
    base = {
        "condition": "periodic_theta89",
        "scenario": "fno_only",
        "seed": 17,
        "train_samples": None,
        "output_root": None,
    }
    args = SimpleNamespace(**base)
    assert training._result_root(args, "mnist_medium_v1") == (
        tmp_path / "results/mnist_medium/mnist_medium_v1/periodic_theta89/fno_only"
    )

    args.train_samples = 1000
    assert training._result_root(args, "mnist_large_v1") == (
        tmp_path
        / "results/sample_efficiency/mnist_large_v1/periodic_theta89/fno_only"
        / "n1000_seed17"
    )

    args.output_root = Path("/tmp/explicit_training_output")
    assert training._result_root(args, "mnist_large_v1") == args.output_root


def test_build_loaders_preserves_evaluation_order() -> None:
    """Validation and test loaders retain input order without shuffling."""
    arrays = make_arrays()
    normalization = training._normalization(arrays)

    _, validation_loader, test_loader = training._build_loaders(
        arrays,
        arrays,
        arrays,
        "fourier_to_fno",
        normalization,
        batch_size=1,
        seed=123,
    )

    validation_p0 = torch.cat([batch[2] for batch in validation_loader]).numpy()
    test_p0 = torch.cat([batch[2] for batch in test_loader]).numpy()
    np.testing.assert_array_equal(validation_p0, arrays["p0"])
    np.testing.assert_array_equal(test_p0, arrays["p0"])


def test_save_test_outputs_preserves_archive_fields(tmp_path) -> None:
    """The test archive retains predictions, targets, and sample identifiers."""
    arrays = make_arrays()
    prediction = arrays["kwave_forward"] + 1.0
    target = arrays["kwave_forward"]

    training._save_test_outputs(tmp_path, arrays, prediction, target)

    with np.load(tmp_path / "test_predictions.npz") as saved:
        assert saved.files == ["prediction", "target", "p0", "label", "source_index"]
        np.testing.assert_array_equal(saved["prediction"], prediction)
        np.testing.assert_array_equal(saved["target"], target)
        np.testing.assert_array_equal(saved["p0"], arrays["p0"])
        np.testing.assert_array_equal(saved["label"], arrays["label"])
        np.testing.assert_array_equal(saved["source_index"], arrays["source_index"])
