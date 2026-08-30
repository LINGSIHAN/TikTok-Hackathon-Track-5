from __future__ import annotations

import math

import pytest
from PIL import Image

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from src.inference import predictor as predictor_module
from src.inference.predictor import Predictor


class ConstantLogitModel(torch.nn.Module):
    def __init__(self, logit: float = 0.0) -> None:
        super().__init__()
        self.register_buffer("logit", torch.tensor(logit))

    def forward(self, batch):
        return self.logit.expand(batch.shape[0], 1)


def zero_preprocess(_image: Image.Image):
    return torch.zeros(3, 8, 8)


def test_predict_pil_returns_sigmoid_probability_and_sets_eval_mode() -> None:
    model = ConstantLogitModel(logit=2.0)
    model.train()
    predictor = Predictor(model=model, preprocess=zero_preprocess)

    probability = predictor.predict_pil(Image.new("L", (10, 10), color=128))

    assert probability == pytest.approx(1 / (1 + math.exp(-2.0)))
    assert model.training is False


def test_predict_pil_rejects_non_pil_input() -> None:
    predictor = Predictor(model=ConstantLogitModel(), preprocess=zero_preprocess)

    with pytest.raises(TypeError, match="PIL"):
        predictor.predict_pil("not an image")


def test_stress_test_uses_every_grid_entry_deterministically() -> None:
    calls = []

    def fake_transform(image, name, severity, seed):
        calls.append((name, severity, seed))
        return image.copy()

    predictor = Predictor(
        model=ConstantLogitModel(),
        preprocess=zero_preprocess,
        transform_grid={"jpeg": [90, 50], "crop": [0.8]},
        apply_transform_fn=fake_transform,
    )

    results = predictor.stress_test(Image.new("RGB", (8, 8)))

    assert calls == [("jpeg", 90, 42), ("jpeg", 50, 42), ("crop", 0.8, 42)]
    assert results == [
        {"transform": "jpeg", "severity": "90", "pred": 0.5},
        {"transform": "jpeg", "severity": "50", "pred": 0.5},
        {"transform": "crop", "severity": "0.8", "pred": 0.5},
    ]


def test_from_checkpoint_builds_without_pretrained_download(
    monkeypatch, tmp_path
) -> None:
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.touch()
    model = ConstantLogitModel()
    observed = {}

    def fake_build_model(*, pretrained):
        observed["pretrained"] = pretrained
        return model

    def fake_load_checkpoint(received_model, path, device):
        observed.update(model=received_model, path=path, device=device)
        return {}

    monkeypatch.setattr(predictor_module, "build_model", fake_build_model)
    monkeypatch.setattr(predictor_module, "load_checkpoint", fake_load_checkpoint)

    result = Predictor.from_checkpoint(checkpoint, device="cpu")

    assert isinstance(result, Predictor)
    assert observed == {
        "pretrained": False,
        "model": model,
        "path": checkpoint,
        "device": "cpu",
    }
