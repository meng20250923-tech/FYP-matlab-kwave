"""Tests for versioned experiment configuration files."""

from copy import deepcopy

import pytest
import yaml

from pat_fno.config import ConfigError, load_experiment_config
from pat_fno.config.mnist_pat import CONDITIONS, DATASET_SCALES, SEED

CONFIGS = (
    "data/mnist_pat.yaml",
    "training/forward_operators.yaml",
    "training/sample_efficiency.yaml",
    "reconstruction/retention_study.yaml",
    "evaluation/runtime.yaml",
)


@pytest.mark.parametrize("path", CONFIGS)
def test_completed_experiment_configs_are_valid(path):
    assert load_experiment_config(path)["study"]


def test_data_config_matches_python_constants():
    config = load_experiment_config("data/mnist_pat.yaml")
    assert config["seed"] == SEED
    assert config["conditions"] == CONDITIONS
    for scale in ("medium", "large"):
        dataset = config["datasets"][scale]
        expected = DATASET_SCALES[scale]
        assert dataset["name"] == expected.dataset
        assert dataset["train_samples"] == expected.train_samples
        assert dataset["validation_samples"] == expected.validation_samples
        assert dataset["test_samples"] == expected.test_samples
        assert dataset["shard_size"] == expected.shard_size


def test_reconstruction_config_records_final_itr_steps():
    config = load_experiment_config("reconstruction/retention_study.yaml")
    assert config["iterated_time_reversal"]["selected_steps"] == {
        "periodic_theta89": {"0.10": 2.0, "0.25": 1.5, "0.50": 1.5, "1.00": 0.75},
        "pml_outside_theta45": {"0.10": 2.5, "0.25": 2.0, "0.50": 2.5, "1.00": 1.75},
    }


def test_unknown_top_level_field_is_rejected(tmp_path):
    config = deepcopy(load_experiment_config("evaluation/runtime.yaml"))
    config["unexpected"] = True
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ConfigError, match="unknown"):
        load_experiment_config(path)


def test_invalid_retention_is_rejected(tmp_path):
    config = deepcopy(load_experiment_config("reconstruction/retention_study.yaml"))
    config["retentions"][0] = 0.0
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ConfigError, match="values must lie"):
        load_experiment_config(path)
