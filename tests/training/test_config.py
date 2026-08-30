import json

import pytest

from src.training.config import ConfigError, config_from_mapping, load_config


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


def test_unknown_key_is_rejected():
    raw = valid_config()
    raw["training"]["typo"] = 1

    with pytest.raises(ConfigError, match="Unknown key"):
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
