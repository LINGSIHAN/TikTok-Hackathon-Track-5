import json
from pathlib import Path

import pytest

from src.training.config import ConfigError, config_from_mapping, load_config


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def valid_config():
    return {
        "seed": 42,
        "data": {
            "manifest_path": "data/processed/manifest.csv",
            "train_split": "train",
            "val_split": "val",
            "test_split": "test",
            "image_size": 224,
            "batch_size": 8,
            "num_workers": 0,
        },
        "model": {
            "pretrained": False,
            "freeze_backbone": True,
            "unfreeze_last_blocks": 0,
        },
        "training": {
            "epochs": 2,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "patience": 1,
            "mixed_precision": False,
        },
        "robustness": {"enabled": True, "clean_probability": 0.35},
        "output": {
            "checkpoint_path": "artifacts/checkpoints/model.safetensors",
            "metadata_path": "artifacts/metrics/training_metadata.json",
            "history_path": "artifacts/metrics/history.json",
        },
    }


def test_load_json_config(tmp_path):
    path = tmp_path / "experiment.json"
    path.write_text(json.dumps(valid_config()), encoding="utf-8")

    config = load_config(path)

    assert config.seed == 42
    assert config.robustness.enabled is True
    assert config.data.batch_size == 8
    assert config.initialization is None
    assert "initialization" not in config.to_dict()


def test_optional_warm_start_configuration_is_parsed():
    raw = valid_config()
    raw["model"]["pretrained"] = False
    raw["initialization"] = {
        "checkpoint_path": "artifacts/checkpoints/model.safetensors",
        "expected_sha256": "a" * 64,
        "freeze_frozen_batchnorm": True,
    }
    raw["output"]["checkpoint_path"] = (
        "artifacts/checkpoints/model_v2.safetensors"
    )

    config = config_from_mapping(raw)

    assert config.initialization is not None
    assert config.initialization.expected_sha256 == "a" * 64
    assert config.initialization.freeze_frozen_batchnorm is True
    assert config.to_dict()["initialization"] == raw["initialization"]


def _yaml_sections(path):
    sections = {"root": []}
    current = "root"
    for line in path.read_text(encoding="utf-8").splitlines():
        if line and not line.startswith(" ") and line.endswith(":"):
            current = line[:-1]
            sections[current] = []
        elif line.strip():
            sections[current].append(line.strip())
    return sections


def test_clean_and_robust_experiments_are_controlled_comparisons():
    clean = _yaml_sections(REPOSITORY_ROOT / "configs/train_clean.yaml")
    robust = _yaml_sections(REPOSITORY_ROOT / "configs/train_robust.yaml")

    assert set(clean) == set(robust)
    for section in ("root", "data", "model", "training"):
        assert clean[section] == robust[section]
    assert clean["robustness"] == ["enabled: false", "clean_probability: 1.0"]
    assert robust["robustness"] == ["enabled: true", "clean_probability: 0.35"]
    assert clean["output"] != robust["output"]


def test_unknown_key_is_rejected():
    raw = valid_config()
    raw["training"]["typo"] = 1

    with pytest.raises(ConfigError, match="Unknown key"):
        config_from_mapping(raw)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"expected_sha256": "not-a-hash"}, "expected_sha256"),
        ({"freeze_frozen_batchnorm": 1}, "freeze_frozen_batchnorm"),
        ({"checkpoint_path": "model.pt"}, "safetensors"),
    ],
)
def test_invalid_warm_start_values_are_rejected(change, message):
    raw = valid_config()
    raw["model"]["pretrained"] = False
    raw["initialization"] = {
        "checkpoint_path": "artifacts/checkpoints/model.safetensors",
        "expected_sha256": "a" * 64,
        "freeze_frozen_batchnorm": True,
        **change,
    }
    raw["output"]["checkpoint_path"] = (
        "artifacts/checkpoints/model_v2.safetensors"
    )

    with pytest.raises(ConfigError, match=message):
        config_from_mapping(raw)


def test_warm_start_rejects_pretrained_and_output_overwrite():
    raw = valid_config()
    raw["model"]["pretrained"] = True
    raw["initialization"] = {
        "checkpoint_path": "artifacts/checkpoints/parent.safetensors",
        "expected_sha256": "a" * 64,
        "freeze_frozen_batchnorm": True,
    }
    raw["output"]["checkpoint_path"] = (
        "artifacts/checkpoints/model_v2.safetensors"
    )

    with pytest.raises(ConfigError, match="pretrained"):
        config_from_mapping(raw)

    raw["model"]["pretrained"] = False
    raw["initialization"]["checkpoint_path"] = raw["output"]["checkpoint_path"]
    with pytest.raises(ConfigError, match="must be different"):
        config_from_mapping(raw)


def test_output_artifact_paths_must_be_distinct():
    raw = valid_config()
    raw["output"]["history_path"] = raw["output"]["metadata_path"]

    with pytest.raises(ConfigError, match="output paths must be distinct"):
        config_from_mapping(raw)


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("data", "batch_size", 0, "batch_size"),
        ("training", "learning_rate", 0.0, "learning_rate"),
        ("robustness", "clean_probability", 1.1, "clean_probability"),
        ("output", "checkpoint_path", "model.pt", "safetensors"),
    ],
)
def test_invalid_values_are_rejected(section, key, value, message):
    raw = valid_config()
    raw[section][key] = value

    with pytest.raises(ConfigError, match=message):
        config_from_mapping(raw)
