"""Load and validate versioned YAML experiment configurations."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

CONFIG_ROOT = Path(__file__).resolve().parents[3] / "configs"


class ConfigError(ValueError):
    """Indicate an invalid or unsupported experiment configuration."""


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{context} must be a mapping.")
    return value


def _keys(value: Any, expected: set[str], context: str) -> Mapping[str, Any]:
    mapping = _mapping(value, context)
    missing = expected - set(mapping)
    unknown = set(mapping) - expected
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing {sorted(missing)}")
        if unknown:
            details.append(f"unknown {sorted(unknown)}")
        raise ConfigError(f"Invalid {context}: {', '.join(details)}.")
    return mapping


def _positive(value: Any, context: str) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        raise ConfigError(f"{context} must be positive.")


def _fractions(values: Any, context: str) -> None:
    if not isinstance(values, list) or not values:
        raise ConfigError(f"{context} must be a non-empty list.")
    if any(not isinstance(value, (int, float)) or not 0 < value <= 1 for value in values):
        raise ConfigError(f"{context} values must lie in (0, 1].")
    if len(set(values)) != len(values):
        raise ConfigError(f"{context} must not contain duplicates.")


def _validate_data(config: Mapping[str, Any]) -> None:
    _keys(config, {"study", "seed", "physics", "conditions", "datasets"}, "data config")
    physics = _keys(
        config["physics"],
        {"grid_size", "dx", "dy", "sound_speed", "medium_density", "cfl"},
        "physics",
    )
    for name, value in physics.items():
        _positive(value, f"physics.{name}")
    conditions = _keys(
        config["conditions"],
        {"periodic_theta89", "pml_outside_theta45"},
        "conditions",
    )
    for name, condition in conditions.items():
        values = _keys(condition, {"boundary", "theta_deg"}, f"conditions.{name}")
        if values["boundary"] not in {"periodic", "pml"}:
            raise ConfigError(f"conditions.{name}.boundary is unsupported.")
        _positive(values["theta_deg"], f"conditions.{name}.theta_deg")
    datasets = _keys(config["datasets"], {"medium", "large"}, "datasets")
    fields = {"name", "train_samples", "validation_samples", "test_samples", "shard_size"}
    for name, dataset in datasets.items():
        values = _keys(dataset, fields, f"datasets.{name}")
        for field in fields - {"name"}:
            _positive(values[field], f"datasets.{name}.{field}")


def _validate_model(model: Any, context: str) -> None:
    values = _keys(model, {"modes", "width", "layers"}, context)
    for name, value in values.items():
        _positive(value, f"{context}.{name}")


def _validate_optimizer(optimizer: Any, context: str) -> None:
    values = _keys(optimizer, {"name", "learning_rate", "weight_decay"}, context)
    _positive(values["learning_rate"], f"{context}.learning_rate")
    if not isinstance(values["weight_decay"], (int, float)) or values["weight_decay"] < 0:
        raise ConfigError(f"{context}.weight_decay must be non-negative.")


def _validate_forward(config: Mapping[str, Any]) -> None:
    _keys(
        config,
        {"study", "seed", "scenarios", "model", "optimizer", "dataset_runs", "overrides"},
        "forward config",
    )
    if set(config["scenarios"]) != {"fno_only", "fourier_to_fno", "fno_to_fourier"}:
        raise ConfigError("forward scenarios must contain the three learned operators.")
    _validate_model(config["model"], "model")
    _validate_optimizer(config["optimizer"], "optimizer")
    runs = _keys(config["dataset_runs"], {"medium", "large"}, "dataset_runs")
    for name, run in runs.items():
        values = _keys(run, {"dataset", "epochs", "batch_size"}, f"dataset_runs.{name}")
        _positive(values["epochs"], f"dataset_runs.{name}.epochs")
        _positive(values["batch_size"], f"dataset_runs.{name}.batch_size")
    for name, override in _mapping(config["overrides"], "overrides").items():
        values = _keys(override, {"epochs"}, f"overrides.{name}")
        _positive(values["epochs"], f"overrides.{name}.epochs")


def _validate_sample_efficiency(config: Mapping[str, Any]) -> None:
    _keys(
        config,
        {
            "study",
            "dataset",
            "conditions",
            "scenarios",
            "train_samples",
            "seeds",
            "epochs",
            "batch_size",
            "model",
            "optimizer",
        },
        "sample-efficiency config",
    )
    if sorted(config["train_samples"]) != [1_000, 5_000, 10_000, 25_000, 50_000]:
        raise ConfigError("sample-efficiency sizes do not match the completed study.")
    if sorted(config["seeds"]) != [20260728, 20260729, 20260730]:
        raise ConfigError("sample-efficiency seeds do not match the completed study.")
    _positive(config["epochs"], "epochs")
    _positive(config["batch_size"], "batch_size")
    _validate_model(config["model"], "model")
    _validate_optimizer(config["optimizer"], "optimizer")


def _validate_reconstruction(config: Mapping[str, Any]) -> None:
    _keys(
        config,
        {
            "study",
            "dataset",
            "split",
            "conditions",
            "retentions",
            "mask_seed",
            "test_samples",
            "bootstrap",
            "gradient_descent",
            "iterated_time_reversal",
            "learned_optimization",
        },
        "reconstruction config",
    )
    _fractions(config["retentions"], "retentions")
    bootstrap = _keys(config["bootstrap"], {"draws", "seed"}, "bootstrap")
    _positive(bootstrap["draws"], "bootstrap.draws")
    gd = _keys(
        config["gradient_descent"],
        {"iterations", "detailed_retention", "detailed_iterations", "power_iterations", "step"},
        "gradient_descent",
    )
    for field in {"iterations", "detailed_iterations", "power_iterations"}:
        _positive(gd[field], f"gradient_descent.{field}")
    itr = _keys(
        config["iterated_time_reversal"],
        {
            "iterations",
            "detailed_retention",
            "detailed_iterations",
            "validation_samples",
            "candidate_steps",
            "selected_steps",
        },
        "iterated_time_reversal",
    )
    for field in {"iterations", "detailed_iterations"}:
        _positive(itr[field], f"iterated_time_reversal.{field}")
    retention_keys = {f"{value:.2f}" for value in config["retentions"]}
    if set(itr["validation_samples"]) != retention_keys:
        raise ConfigError("ITR validation sample counts must cover every retention.")
    condition_keys = set(config["conditions"])
    for field in {"candidate_steps", "selected_steps"}:
        values = _keys(itr[field], condition_keys, f"iterated_time_reversal.{field}")
        for condition, entries in values.items():
            if set(entries) != retention_keys:
                raise ConfigError(f"ITR {field}.{condition} must cover every retention.")
            for retention, entry in entries.items():
                candidates = entry if field == "candidate_steps" else [entry]
                if any(not isinstance(value, (int, float)) or value <= 0 for value in candidates):
                    raise ConfigError(f"ITR {field}.{condition}.{retention} must be positive.")
    learned = _keys(
        config["learned_optimization"],
        {"updates", "optimizer", "learning_rate", "image_range"},
        "learned_optimization",
    )
    _positive(learned["updates"], "learned_optimization.updates")
    _positive(learned["learning_rate"], "learned_optimization.learning_rate")
    if learned["image_range"] != [0.0, 1.0]:
        raise ConfigError("learned_optimization.image_range must be [0.0, 1.0].")


def _validate_runtime(config: Mapping[str, Any]) -> None:
    _keys(config, {"study", "dataset", "split", "conditions", "modes"}, "runtime config")
    modes = _keys(config["modes"], {"cpu_batch1", "rtx4090_batch1", "rtx4090_batch64"}, "modes")
    fields = {"device", "batch_size", "warmup", "repeats", "include_kwave"}
    for name, mode in modes.items():
        values = _keys(mode, fields, f"modes.{name}")
        for field in {"batch_size", "repeats"}:
            _positive(values[field], f"modes.{name}.{field}")
        if not isinstance(values["warmup"], int) or values["warmup"] < 0:
            raise ConfigError(f"modes.{name}.warmup must be a non-negative integer.")


def _validate_reproducibility(config: Mapping[str, Any]) -> None:
    _keys(
        config,
        {
            "study",
            "condition",
            "sample_data",
            "checkpoint",
            "expected_results",
            "output",
            "sample_count",
            "retention",
            "tolerances",
        },
        "reproducibility config",
    )
    if config["condition"] not in {"periodic_theta89", "pml_outside_theta45"}:
        raise ConfigError("reproducibility condition is unsupported.")
    _positive(config["sample_count"], "sample_count")
    _fractions([config["retention"]], "retention")
    tolerances = _keys(
        config["tolerances"],
        {"array_atol", "metric_atol"},
        "tolerances",
    )
    for name, value in tolerances.items():
        _positive(value, f"tolerances.{name}")
    for name in {"sample_data", "checkpoint", "expected_results", "output"}:
        if not isinstance(config[name], str) or not config[name].strip():
            raise ConfigError(f"{name} must be a non-empty path string.")


VALIDATORS = {
    "data_generation": _validate_data,
    "forward_training": _validate_forward,
    "sample_efficiency": _validate_sample_efficiency,
    "reconstruction_retention": _validate_reconstruction,
    "runtime_benchmark": _validate_runtime,
    "reproducibility_example": _validate_reproducibility,
}


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    """Load one YAML configuration and reject unsupported fields or values."""
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = CONFIG_ROOT / config_path
    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ConfigError(f"Unable to load configuration {config_path}: {error}") from error
    config = dict(_mapping(data, "configuration root"))
    study = config.get("study")
    try:
        validator = VALIDATORS[study]
    except (KeyError, TypeError) as error:
        raise ConfigError(f"Unsupported configuration study {study!r}.") from error
    validator(config)
    return config
