"""Unit tests for reconstruction helpers and learned-operator paths."""

from __future__ import annotations

from argparse import Namespace
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

from pat_fno.models import TinyFNO2d
from scripts.reconstruction import run_reconstruction as reconstruction
from scripts.reconstruction.common import (
    load_subsampled,
    reconstruction_tag,
    subsampled_path,
)


def test_reconstruction_tag_encodes_experiment() -> None:
    """The result tag records condition, split, retention, and seed."""
    tag = reconstruction_tag(
        "periodic_theta89",
        "test",
        0.25,
        20260802,
    )

    assert tag == "periodic_theta89_test_keep0.25_seed20260802"


def test_subsampled_path_uses_expected_layout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Subsampled measurements follow the established result layout."""
    monkeypatch.setattr(
        "scripts.reconstruction.common.ROOT",
        tmp_path,
    )
    arguments = Namespace(
        dataset="mnist_medium_v1",
        split="test",
        keep_fraction=0.5,
        seed=17,
    )

    path = subsampled_path(arguments, "pml_outside_theta45")

    assert path == (
        tmp_path
        / "results"
        / "reconstruction"
        / "mnist_medium_v1"
        / "subsampled"
        / "pml_outside_theta45_test_keep0.50_seed17.h5"
    )


def test_load_subsampled_reads_requested_prefix(tmp_path) -> None:
    """Only the requested number of samples is loaded."""
    path = tmp_path / "subsampled.h5"
    arrays = {
        "observed_data": np.arange(60).reshape(3, 4, 5),
        "p0": np.arange(48).reshape(3, 4, 4),
        "label": np.array([1, 2, 3]),
        "source_index": np.array([10, 11, 12]),
        "mask": np.ones((3, 4, 5), dtype=bool),
    }

    with h5py.File(path, "w") as handle:
        for name, values in arrays.items():
            handle.create_dataset(name, data=values)

    loaded = load_subsampled(path, 2, include_mask=True)

    assert set(loaded) == set(arrays)
    assert all(values.shape[0] == 2 for values in loaded.values())
    np.testing.assert_array_equal(
        loaded["source_index"],
        np.array([10, 11]),
    )


def test_load_subsampled_validates_inputs(tmp_path) -> None:
    """Missing files and non-positive sample counts are rejected."""
    with pytest.raises(FileNotFoundError):
        load_subsampled(tmp_path / "missing.h5", 1)

    path = tmp_path / "empty.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("p0", data=np.zeros((1, 2, 2)))

    with pytest.raises(ValueError, match="max-samples"):
        load_subsampled(path, 0)


def test_method_registry_covers_all_reconstruction_families() -> None:
    """The CLI exposes every implemented reconstruction family."""
    assert reconstruction.METHODS == (
        "fourier",
        "time_reversal",
        "iterated_time_reversal",
        "gradient_descent",
        "learned",
        "adjoint",
    )


def test_lipschitz_estimate_for_identity_operator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Power iteration returns one for an identity normal operator."""
    monkeypatch.setattr(
        reconstruction,
        "kwave_forward_2d",
        lambda image, setting: image,
    )
    monkeypatch.setattr(
        reconstruction,
        "kwave_adjoint_2d",
        lambda data, setting: data,
    )
    setting = SimpleNamespace(Nx=4, Ny=4)
    mask = np.ones((4, 4), dtype=np.float64)

    value = reconstruction.estimate_lipschitz(
        mask,
        setting,
        iterations=4,
    )

    assert value == pytest.approx(1.0)


def test_load_learned_model_preserves_checkpoint(tmp_path) -> None:
    """A saved thesis checkpoint loads strictly into the migrated FNO."""
    torch.manual_seed(4)
    source_model = TinyFNO2d(
        modes1=8,
        modes2=8,
        width=16,
        layers=3,
    )
    normalization = {
        "p0_mean": 0.2,
        "p0_std": 0.4,
        "data_mean": -0.1,
        "data_std": 0.7,
    }
    checkpoint_path = tmp_path / "best.pt"
    torch.save(
        {
            "arguments": {
                "modes": 8,
                "width": 16,
                "layers": 3,
            },
            "model": source_model.state_dict(),
            "normalization": normalization,
        },
        checkpoint_path,
    )

    loaded_model, loaded_normalization = reconstruction._load_learned_model(
        checkpoint_path,
        torch.device("cpu"),
    )

    assert loaded_normalization == normalization
    assert loaded_model.state_dict().keys() == source_model.state_dict().keys()
    assert not loaded_model.training
    assert all(not parameter.requires_grad for parameter in loaded_model.parameters())


class IdentityModel(torch.nn.Module):
    """Return the supplied single-channel field unchanged."""

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values


@pytest.mark.parametrize(
    ("scenario", "expected_shape"),
    [
        ("fno_only", (6, 8)),
        ("fourier_to_fno", (6, 8)),
        ("fno_to_fourier", (6, 8)),
    ],
)
def test_learned_prediction_paths_preserve_output_domain(
    monkeypatch: pytest.MonkeyPatch,
    scenario: str,
    expected_shape: tuple[int, int],
) -> None:
    """Every learned scenario returns one detector-time pressure field."""
    monkeypatch.setattr(
        reconstruction,
        "fpat_forward_2d",
        lambda image, theta, sound_speed, nt, spacing, dt: torch.nn.functional.interpolate(
            image[None, None],
            size=(6, nt),
            mode="bilinear",
            align_corners=False,
        )[0, 0],
    )
    setting = SimpleNamespace(
        Ny=6,
        Nt=8,
        soundSpeed=1.0,
        dx=1.0,
        dy=1.0,
        dt=0.1,
        computation=SimpleNamespace(theta_max=np.pi / 2),
    )
    normalization = {
        "p0_mean": 0.0,
        "p0_std": 1.0,
        "data_mean": 0.0,
        "data_std": 1.0,
    }
    image = torch.linspace(0, 1, 36).reshape(6, 6)

    output = reconstruction._learned_prediction(
        IdentityModel(),
        scenario,
        image,
        setting,
        normalization,
    )

    assert output.shape == expected_shape
    assert torch.isfinite(output).all()


def test_iterated_time_reversal_rejects_non_positive_step() -> None:
    """ITR requires a strictly positive update step."""
    arguments = Namespace(step_size=0.0)

    with pytest.raises(ValueError, match="step-size"):
        reconstruction.run_iterative(
            arguments,
            "periodic_theta89",
            gradient_descent=False,
        )


def test_iterative_method_configuration_preserves_result_naming() -> None:
    """Iterative method configuration retains the established output suffixes."""
    arguments = Namespace(step_size=1.5)

    assert reconstruction._iterative_method_configuration(arguments, True) == (
        "gradient_descent",
        "",
    )
    assert reconstruction._iterative_method_configuration(arguments, False) == (
        "iterated_time_reversal",
        "_step1.5",
    )


@pytest.mark.parametrize("gradient_descent", [False, True])
def test_iterative_sample_retains_update_order(
    monkeypatch: pytest.MonkeyPatch,
    gradient_descent: bool,
) -> None:
    """The extracted sample loop follows the established reconstruction updates."""
    monkeypatch.setattr(reconstruction, "kwave_forward_2d", lambda image, setting: image)
    monkeypatch.setattr(reconstruction, "kwave_adjoint_2d", lambda residual, setting: residual)
    monkeypatch.setattr(reconstruction, "kwave_inverse_2d", lambda residual, setting: residual)
    monkeypatch.setattr(reconstruction, "estimate_lipschitz", lambda mask, setting, count: 2.0)
    target = np.ones((2, 2), dtype=np.float64)
    observed = np.ones_like(target)
    mask = np.ones_like(target)

    image, residuals, errors, lipschitz, effective_step = reconstruction._run_iterative_sample(
        target=target,
        observed=observed,
        mask=mask,
        setting=object(),
        iterations=2,
        gradient_descent=gradient_descent,
        step_size=0.25,
        power_iterations=4,
        progress_prefix="test",
    )

    expected = np.full((2, 2), 0.4375 if not gradient_descent else 0.75)
    np.testing.assert_array_equal(image, expected)
    np.testing.assert_allclose(
        residuals,
        [1.0, 0.75 if not gradient_descent else 0.5, 0.5625 if not gradient_descent else 0.25],
    )
    np.testing.assert_allclose(errors, residuals)
    if gradient_descent:
        assert lipschitz == 2.0
        assert effective_step == 0.5
    else:
        assert lipschitz is None
        assert effective_step is None


def test_learned_sample_optimisation_is_deterministic() -> None:
    """The extracted learned loop retains Adam updates and image projection."""
    truth = torch.ones((2, 2), dtype=torch.float32)
    observation = torch.full((2, 2), 0.6, dtype=torch.float32)
    mask = torch.ones_like(observation)
    setting = SimpleNamespace(Ny=2, Nt=2)
    normalization = {
        "p0_mean": 0.0,
        "p0_std": 1.0,
        "data_mean": 0.0,
        "data_std": 1.0,
    }

    image, history = reconstruction._optimise_learned_sample(
        model=IdentityModel(),
        scenario="fno_only",
        observation=observation,
        mask=mask,
        truth=truth,
        setting=setting,
        normalization=normalization,
        iterations=2,
        learning_rate=0.1,
    )

    np.testing.assert_allclose(history, [1.0, 0.9, 0.8008828], rtol=1e-6)
    np.testing.assert_allclose(image.numpy(), np.full((2, 2), 0.1991172), rtol=1e-6)
    assert not image.requires_grad
