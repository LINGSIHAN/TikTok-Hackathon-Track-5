from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from src.models.checkpoints import load_checkpoint, save_checkpoint


def test_safetensors_checkpoint_round_trip(tmp_path) -> None:
    source = torch.nn.Linear(3, 1)
    destination = torch.nn.Linear(3, 1)
    checkpoint = tmp_path / "tiny.safetensors"

    save_checkpoint(source, checkpoint, metadata={"experiment": "unit-test"})
    metadata = load_checkpoint(destination, checkpoint)

    assert metadata["architecture"] == "efficientnet_b0_binary"
    assert metadata["experiment"] == "unit-test"
    for source_parameter, destination_parameter in zip(
        source.parameters(), destination.parameters(), strict=True
    ):
        assert torch.equal(source_parameter, destination_parameter)


def test_checkpoint_helpers_reject_pickle_extension(tmp_path) -> None:
    model = torch.nn.Linear(1, 1)

    with pytest.raises(ValueError, match="safetensors"):
        save_checkpoint(model, tmp_path / "model.pt")
