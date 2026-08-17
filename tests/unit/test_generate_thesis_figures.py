"""Tests for structured thesis-figure generation."""

from __future__ import annotations

import numpy as np

from scripts.evaluation import generate_thesis_figures as figures


def test_learned_prediction_path_uses_stable_result_layout(monkeypatch, tmp_path) -> None:
    """Learned predictions resolve under the configured project root."""
    monkeypatch.setattr(figures, "ROOT", tmp_path)

    path = figures.learned_prediction_path("periodic_theta89", "fourier_to_fno")

    assert path == (
        tmp_path
        / "results/mnist_medium/mnist_large_v1"
        / "periodic_theta89/fourier_to_fno/test_predictions.npz"
    )


def test_load_prediction_sample_returns_requested_item(monkeypatch, tmp_path) -> None:
    """Loading one prediction preserves its saved values and shape."""
    archive = tmp_path / "predictions.npz"
    predictions = np.arange(60, dtype=np.float32).reshape(3, 4, 5)
    np.savez(archive, prediction=predictions)
    monkeypatch.setattr(figures, "learned_prediction_path", lambda condition, scenario: archive)

    loaded = figures.load_prediction_sample("periodic_theta89", "fno_only", 1)

    np.testing.assert_array_equal(loaded, predictions[1])


def test_load_forward_sample_preserves_scenario_order(monkeypatch) -> None:
    """One forward sample includes every learned scenario in declared order."""
    analytical = np.arange(24).reshape(3, 2, 4)
    target = analytical + 100
    calls: list[tuple[str, str, int]] = []
    monkeypatch.setattr(
        figures,
        "_large_forward_arrays",
        lambda condition: (analytical, target),
    )

    def load(condition: str, scenario: str, sample_index: int) -> np.ndarray:
        calls.append((condition, scenario, sample_index))
        return np.full((2, 4), len(calls), dtype=np.float32)

    monkeypatch.setattr(figures, "load_prediction_sample", load)

    loaded_analytical, loaded_target, learned = figures.load_forward_sample(
        "pml_outside_theta45",
        2,
    )

    np.testing.assert_array_equal(loaded_analytical, analytical[2])
    np.testing.assert_array_equal(loaded_target, target[2])
    assert list(learned) == [scenario for scenario, _ in figures.LEARNED_FORWARD_SCENARIOS]
    assert calls == [
        ("pml_outside_theta45", scenario, 2) for scenario, _ in figures.LEARNED_FORWARD_SCENARIOS
    ]


def test_prepare_output_directory_respects_clean_flag(tmp_path) -> None:
    """Cleaning removes stale outputs while ordinary preparation preserves them."""
    output = tmp_path / "figures"
    output.mkdir()
    stale = output / "stale.txt"
    stale.write_text("old", encoding="utf-8")

    figures.prepare_output_directory(output, clean=False)
    assert stale.exists()

    figures.prepare_output_directory(output, clean=True)
    assert output.is_dir()
    assert not stale.exists()


def test_generate_figure_set_preserves_generation_order(monkeypatch, tmp_path) -> None:
    """The orchestrator calls every active generator in its established order."""
    calls: list[tuple[str, tuple[object, ...]]] = []
    names = (
        "setup_style",
        "plot_forward_error_ecdf",
        "plot_forward_prediction_examples",
        "plot_forward_error_maps",
        "plot_forward_sensor_traces",
        "plot_training_curves",
        "plot_sample_efficiency",
        "plot_runtime",
        "plot_reconstruction_metrics",
        "plot_reconstruction_keep025_comparison",
        "plot_reconstruction_keep025_correlation",
        "plot_convergence",
        "plot_complete_reconstruction_montages",
        "plot_retention_montages",
        "write_readme",
    )

    for name in names:
        monkeypatch.setattr(
            figures,
            name,
            lambda *args, _name=name: calls.append((_name, args)),
        )

    figures.generate_figure_set(tmp_path, sample_index=13)

    assert [name for name, _ in calls] == list(names)
    sample_generators = {
        "plot_forward_prediction_examples",
        "plot_forward_error_maps",
        "plot_forward_sensor_traces",
        "plot_complete_reconstruction_montages",
        "plot_retention_montages",
    }
    for name, arguments in calls:
        if name == "setup_style":
            assert arguments == ()
        elif name in sample_generators:
            assert arguments == (tmp_path, 13)
        else:
            assert arguments == (tmp_path,)
