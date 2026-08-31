from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
from safetensors import safe_open
from torch.utils.data import DataLoader, TensorDataset

from src.data.preprocessing import (
    PREPROCESSING_CONTRACT_ID,
    PREPROCESSING_METADATA_KEY,
)
from src.models import checkpoints as checkpoint_module
from src.models.checkpoints import save_checkpoint
from src.models import efficientnet as efficientnet_module
from src.training import train as train_module
from src.training.config import config_from_mapping


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _warm_start_config(tmp_path: Path, parent: Path, expected_sha256: str):
    return config_from_mapping(
        {
            "seed": 42,
            "data": {
                "manifest_path": "unused.csv",
                "train_split": "train",
                "val_split": "val",
                "test_split": "test",
                "image_size": 224,
                "batch_size": 4,
                "num_workers": 0,
            },
            "model": {
                "pretrained": False,
                "freeze_backbone": True,
                "unfreeze_last_blocks": 1,
            },
            "training": {
                "epochs": 1,
                "learning_rate": 3e-5,
                "weight_decay": 1e-4,
                "patience": 1,
                "mixed_precision": False,
            },
            "robustness": {"enabled": True, "clean_probability": 0.35},
            "output": {
                "checkpoint_path": str(tmp_path / "model_v2.safetensors"),
                "metadata_path": str(tmp_path / "model_v2_metadata.json"),
                "history_path": str(tmp_path / "model_v2_history.json"),
            },
            "initialization": {
                "checkpoint_path": str(parent),
                "expected_sha256": expected_sha256,
                "freeze_frozen_batchnorm": True,
            },
        }
    )


def _parent_metadata(**overrides: str) -> dict[str, str]:
    metadata = {
        "architecture": "efficientnet_b0_binary",
        "image_size": "224",
        PREPROCESSING_METADATA_KEY: PREPROCESSING_CONTRACT_ID,
    }
    metadata.update(overrides)
    return metadata


def test_initialization_verifies_hash_and_loads_parent_weights(tmp_path) -> None:
    parent_model = torch.nn.Linear(1, 1)
    with torch.no_grad():
        parent_model.weight.fill_(2.0)
        parent_model.bias.fill_(-1.0)
    parent = tmp_path / "parent.safetensors"
    save_checkpoint(parent_model, parent, metadata=_parent_metadata())
    destination = torch.nn.Linear(1, 1)
    config = _warm_start_config(tmp_path, parent, _sha256(parent))

    actual_sha256 = train_module._initialize_model_from_checkpoint(
        destination,
        config,
        device=torch.device("cpu"),
    )

    assert actual_sha256 == _sha256(parent)
    assert torch.equal(destination.weight, parent_model.weight)
    assert torch.equal(destination.bias, parent_model.bias)


def test_initialization_rejects_changed_checkpoint(tmp_path) -> None:
    parent = tmp_path / "parent.safetensors"
    save_checkpoint(
        torch.nn.Linear(1, 1),
        parent,
        metadata=_parent_metadata(),
    )
    config = _warm_start_config(tmp_path, parent, "0" * 64)

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        train_module._initialize_model_from_checkpoint(
            torch.nn.Linear(1, 1),
            config,
            device=torch.device("cpu"),
        )


@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        (_parent_metadata(architecture="other_model"), "architecture"),
        (_parent_metadata(image_size="256"), "image_size"),
        (
            {"architecture": "efficientnet_b0_binary", "image_size": "224"},
            "preprocessing contract",
        ),
    ],
)
def test_initialization_rejects_incompatible_metadata(
    tmp_path,
    metadata,
    message,
) -> None:
    parent = tmp_path / "parent.safetensors"
    save_checkpoint(torch.nn.Linear(1, 1), parent, metadata=metadata)
    config = _warm_start_config(tmp_path, parent, _sha256(parent))

    with pytest.raises(ValueError, match=message):
        train_module._initialize_model_from_checkpoint(
            torch.nn.Linear(1, 1),
            config,
            device=torch.device("cpu"),
        )


def test_training_loads_before_optimizer_and_records_parent_lineage(
    tmp_path,
    monkeypatch,
) -> None:
    parent_model = torch.nn.Linear(1, 1)
    parent = tmp_path / "parent.safetensors"
    save_checkpoint(parent_model, parent, metadata=_parent_metadata())
    parent_sha256 = _sha256(parent)
    config = _warm_start_config(tmp_path, parent, parent_sha256)
    loader = DataLoader(
        TensorDataset(
            torch.tensor([[0.0], [1.0], [2.0], [3.0]]),
            torch.tensor([0, 0, 1, 1]),
        ),
        batch_size=4,
    )
    events: list[str] = []
    original_load_checkpoint = checkpoint_module.load_checkpoint
    original_adamw = torch.optim.AdamW

    def tracked_load_checkpoint(*args, **kwargs):
        events.append("load")
        return original_load_checkpoint(*args, **kwargs)

    def tracked_adamw(*args, **kwargs):
        events.append("optimizer")
        return original_adamw(*args, **kwargs)

    monkeypatch.setattr(checkpoint_module, "load_checkpoint", tracked_load_checkpoint)
    monkeypatch.setattr(torch.optim, "AdamW", tracked_adamw)
    monkeypatch.setattr(
        efficientnet_module,
        "build_model",
        lambda **_kwargs: torch.nn.Linear(1, 1),
    )
    monkeypatch.setattr(
        train_module,
        "create_dataloaders",
        lambda _config: (loader, loader),
    )

    metadata = train_module.train_from_config(config, device_name="cpu")

    assert events[:2] == ["load", "optimizer"]
    assert metadata["parent_checkpoint_sha256"] == parent_sha256
    assert metadata["checkpoint_sha256"] == _sha256(
        Path(config.output.checkpoint_path)
    )
    with safe_open(
        config.output.checkpoint_path, framework="pt", device="cpu"
    ) as checkpoint:
        output_metadata = checkpoint.metadata() or {}
    assert output_metadata["parent_checkpoint_sha256"] == parent_sha256
