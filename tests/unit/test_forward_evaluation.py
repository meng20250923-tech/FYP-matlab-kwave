"""Regression tests for forward-model evaluation orchestration."""

from __future__ import annotations

import json
from types import SimpleNamespace

from scripts.evaluation import evaluate_forward as evaluation


def make_arguments(**overrides):
    """Construct a complete evaluation argument namespace."""
    values = {
        "scale": "medium",
        "dataset": "mnist_medium_v1",
        "condition": "all",
        "split": "test",
        "metrics_only": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_load_baseline_reuses_metrics_without_evaluation(tmp_path, monkeypatch):
    """Metrics-only mode must read the baseline and avoid dataset evaluation."""
    baseline = {"periodic_theta89": {"rel_l2_mean": 0.25}}
    path = tmp_path / "results" / "mnist_medium" / "baselines"
    path.mkdir(parents=True)
    (path / "baseline_metrics_test.json").write_text(json.dumps(baseline))
    monkeypatch.setattr(evaluation, "ROOT", tmp_path)
    monkeypatch.setattr(
        evaluation,
        "_evaluate_fourier",
        lambda *_args: (_ for _ in ()).throw(AssertionError("unexpected evaluation")),
    )

    result = evaluation.load_baseline(
        make_arguments(metrics_only=True),
        "mnist_medium_v1",
    )

    assert result == baseline


def test_collect_comparison_rows_preserves_condition_and_model_order(tmp_path, monkeypatch):
    """Baseline and learned rows must follow the established stable ordering."""
    condition_order = ["periodic_theta89", "pml_outside_theta45"]
    monkeypatch.setattr(evaluation, "ROOT", tmp_path)
    monkeypatch.setattr(evaluation, "conditions", lambda _selected: condition_order)
    baseline = {
        condition: {"rel_l2_mean": 0.4 + index} for index, condition in enumerate(condition_order)
    }
    for condition_index, condition in enumerate(condition_order):
        for scenario_index, (scenario, _label) in enumerate(evaluation.MODELS):
            path = (
                tmp_path
                / "results"
                / "mnist_medium"
                / "mnist_medium_v1"
                / condition
                / scenario
                / "metrics.json"
            )
            path.parent.mkdir(parents=True)
            path.write_text(
                json.dumps(
                    {
                        "test": {
                            "rel_l2_mean": condition_index + scenario_index / 10,
                        }
                    }
                )
            )

    rows = evaluation.collect_comparison_rows(
        make_arguments(),
        "mnist_medium_v1",
        baseline,
    )

    assert [(row["condition"], row["model"]) for row in rows] == [
        ("periodic_theta89", "c x Fourier baseline"),
        ("periodic_theta89", "FNO-only"),
        ("periodic_theta89", "Fourier-to-FNO"),
        ("periodic_theta89", "FNO-to-Fourier"),
        ("pml_outside_theta45", "c x Fourier baseline"),
        ("pml_outside_theta45", "FNO-only"),
        ("pml_outside_theta45", "Fourier-to-FNO"),
        ("pml_outside_theta45", "FNO-to-Fourier"),
    ]


def test_save_comparison_preserves_json_field_order(tmp_path):
    """Comparison JSON metadata must retain its documented field order."""
    rows = [{"condition": "periodic_theta89", "model": "FNO-only"}]

    evaluation.save_comparison(
        tmp_path,
        "mnist_medium_v1",
        "medium",
        "test",
        rows,
    )

    content = json.loads((tmp_path / "comparison.json").read_text())
    assert list(content) == ["dataset", "scale", "split", "rows"]
    assert content["rows"] == rows


def test_main_writes_empty_comparison_without_plotting(tmp_path, monkeypatch, capsys):
    """An empty metric collection must still be saved before returning."""
    events = []
    args = make_arguments(dataset=None)
    monkeypatch.setattr(evaluation, "ROOT", tmp_path)
    monkeypatch.setattr(evaluation, "parse_args", lambda: args)
    monkeypatch.setattr(
        evaluation,
        "load_baseline",
        lambda current_args, dataset: events.append(("baseline", dataset)) or {},
    )
    monkeypatch.setattr(
        evaluation,
        "collect_comparison_rows",
        lambda current_args, dataset, baseline: events.append(("rows", dataset, baseline)) or [],
    )
    monkeypatch.setattr(
        evaluation,
        "save_comparison",
        lambda output, dataset, scale, split, rows: events.append(
            ("save", output, dataset, scale, split, rows)
        ),
    )
    monkeypatch.setattr(
        evaluation,
        "plot_comparison",
        lambda *_args: (_ for _ in ()).throw(AssertionError("unexpected plot")),
    )

    evaluation.main()

    expected_output = tmp_path / "results" / "mnist_medium" / "mnist_medium_v1"
    assert events == [
        ("baseline", "mnist_medium_v1"),
        ("rows", "mnist_medium_v1", {}),
        ("save", expected_output, "mnist_medium_v1", "medium", "test", []),
    ]
    assert capsys.readouterr().out.strip() == "No completed metrics found."
