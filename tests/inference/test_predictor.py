from __future__ import annotations

import math

import pytest
from PIL import Image

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from src.data.preprocessing import (
    PREPROCESSING_CONTRACT_ID,
    PREPROCESSING_METADATA_KEY,
)
from src.inference import predictor as predictor_module
from src.inference.predictor import (
    Predictor,
    build_inference_preprocess,
    prepare_stress_image,
)


class ConstantLogitModel(torch.nn.Module):
    def __init__(self, logit: float = 0.0) -> None:
        super().__init__()
        self.register_buffer("logit", torch.tensor(logit))

    def forward(self, batch):
        return self.logit.expand(batch.shape[0], 1)


def zero_preprocess(_image: Image.Image):
    return torch.zeros(3, 8, 8)


def test_inference_uses_shared_preprocess_builder(monkeypatch) -> None:
    sentinel = object()
    observed = []

    def fake_build(image_size):
        observed.append(image_size)
        return sentinel

    monkeypatch.setattr(predictor_module, "build_image_preprocess", fake_build)

    assert build_inference_preprocess(96) is sentinel
    assert observed == [96]


def test_predict_pil_returns_sigmoid_probability_and_sets_eval_mode() -> None:
    model = ConstantLogitModel(logit=2.0)
    model.train()
    predictor = Predictor(model=model, preprocess=zero_preprocess)

    probability = predictor.predict_pil(Image.new("L", (10, 10), color=128))

    assert probability == pytest.approx(1 / (1 + math.exp(-2.0)))
    assert model.training is False


def test_predict_pil_normalizes_transparency_before_preprocessing() -> None:
    observed = []

    def inspect_preprocess(image: Image.Image):
        observed.append((image.mode, image.getpixel((0, 0))))
        return torch.zeros(3, 8, 8)

    predictor = Predictor(
        model=ConstantLogitModel(),
        preprocess=inspect_preprocess,
    )

    predictor.predict_pil(Image.new("RGBA", (2, 2), color=(255, 0, 0, 0)))

    assert observed == [("RGB", (255, 255, 255))]


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


def test_stress_test_normalizes_before_applying_transform() -> None:
    observed = []

    def fake_transform(image, name, severity, seed):
        observed.append((image.mode, image.getpixel((0, 0))))
        return image.copy()

    predictor = Predictor(
        model=ConstantLogitModel(),
        preprocess=zero_preprocess,
        transform_grid={"jpeg": [90]},
        apply_transform_fn=fake_transform,
    )

    predictor.stress_test(
        Image.new("RGBA", (2, 2), color=(255, 0, 0, 0))
    )

    assert observed == [("RGB", (255, 255, 255))]


def test_prepare_stress_image_bounds_edge_without_changing_aspect_ratio() -> None:
    working = prepare_stress_image(Image.new("RGB", (4000, 2000)), max_edge=1000)

    assert working.size == (1000, 500)


def test_stress_test_generates_transforms_lazily() -> None:
    live_transforms = 0
    maximum_live_transforms = 0

    class TrackedImage:
        pass

    def fake_transform(image, name, severity, seed):
        nonlocal live_transforms, maximum_live_transforms
        del image, name, severity, seed
        live_transforms += 1
        maximum_live_transforms = max(maximum_live_transforms, live_transforms)
        return TrackedImage()

    predictor = Predictor(
        model=ConstantLogitModel(),
        preprocess=zero_preprocess,
        transform_grid={"jpeg": [90, 50, 30]},
        apply_transform_fn=fake_transform,
    )

    def consume_one_at_a_time(images, batch_size=8):
        nonlocal live_transforms
        probabilities = []
        for _image in images:
            probabilities.append(0.5)
            live_transforms -= 1
        return probabilities

    predictor.predict_many_pil = consume_one_at_a_time
    predictor.stress_test(Image.new("RGB", (8, 8)))

    assert maximum_live_transforms == 1


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
        return {
            PREPROCESSING_METADATA_KEY: PREPROCESSING_CONTRACT_ID,
            "image_size": "96",
        }

    def fake_build_preprocess(image_size):
        observed["image_size"] = image_size
        return zero_preprocess

    monkeypatch.setattr(predictor_module, "build_model", fake_build_model)
    monkeypatch.setattr(predictor_module, "load_checkpoint", fake_load_checkpoint)
    monkeypatch.setattr(
        predictor_module,
        "build_inference_preprocess",
        fake_build_preprocess,
    )

    result = Predictor.from_checkpoint(checkpoint, device="cpu")

    assert isinstance(result, Predictor)
    assert observed == {
        "pretrained": False,
        "model": model,
        "path": checkpoint,
        "device": "cpu",
        "image_size": 96,
    }


@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        ({"image_size": "224"}, "legacy checkpoints must be retrained"),
        (
            {
                PREPROCESSING_METADATA_KEY: "legacy-square-resize-v0",
                "image_size": "224",
            },
            "does not match",
        ),
    ],
)
def test_from_checkpoint_rejects_incompatible_preprocessing_metadata(
    monkeypatch,
    tmp_path,
    metadata,
    message,
) -> None:
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.touch()

    monkeypatch.setattr(
        predictor_module,
        "build_model",
        lambda *, pretrained: ConstantLogitModel(),
    )
    monkeypatch.setattr(
        predictor_module,
        "load_checkpoint",
        lambda model, path, device: metadata,
    )

    with pytest.raises(ValueError, match=message):
        Predictor.from_checkpoint(checkpoint, device="cpu")
