from __future__ import annotations

from types import SimpleNamespace

import pytest
from PIL import Image

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from src.training import train as train_module
from src.data.preprocessing import (
    PREPROCESSING_CONTRACT_ID,
    PREPROCESSING_METADATA_KEY,
)
from src.transforms import robustness


def test_training_applies_robustness_before_shared_preprocess(monkeypatch) -> None:
    events = []
    expected = torch.zeros(3, 16, 16)

    def fake_build_preprocess(image_size):
        assert image_size == 16

        def preprocess(image):
            events.append(("preprocess", image.getpixel((0, 0))))
            return expected

        return preprocess

    def fake_robustness(image, seed, clean_probability):
        events.append(
            (
                "robustness",
                clean_probability,
                image.mode,
                image.getpixel((0, 0)),
            )
        )
        transformed = image.copy()
        transformed.putpixel((0, 0), (0, 255, 0))
        return transformed, {"transform": "test", "seed": seed}

    monkeypatch.setattr(train_module, "build_image_preprocess", fake_build_preprocess)
    monkeypatch.setattr(robustness, "sample_training_transform", fake_robustness)
    transform = train_module.TrainingImageTransform(
        16,
        robust=True,
        clean_probability=0.35,
    )

    result = transform(
        Image.new("RGBA", (40, 20), color=(0, 0, 255, 0))
    )

    assert result is expected
    assert events == [
        ("robustness", 0.35, "RGB", (255, 255, 255)),
        ("preprocess", (0, 255, 0)),
    ]


def test_training_checkpoint_metadata_embeds_preprocessing_contract() -> None:
    config = SimpleNamespace(
        seed=42,
        data=SimpleNamespace(image_size=224),
        robustness=SimpleNamespace(enabled=True),
    )
    early_stopping = SimpleNamespace(best_epoch=3, best_loss=0.125)

    metadata = train_module._build_checkpoint_metadata(config, early_stopping)

    assert metadata[PREPROCESSING_METADATA_KEY] == PREPROCESSING_CONTRACT_ID
    assert metadata["image_size"] == "224"
