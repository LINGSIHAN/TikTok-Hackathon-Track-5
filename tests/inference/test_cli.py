from __future__ import annotations

import io
import json

import pytest
from PIL import Image

pytest.importorskip("torch")
pytest.importorskip("torchvision")

from src.inference import cli


class FakePredictor:
    def predict_pil(self, image: Image.Image) -> float:
        return image.getpixel((0, 0))[0] / 255.0


def test_predict_directory_is_sorted_and_skips_corrupt_images(tmp_path) -> None:
    input_dir = tmp_path / "images"
    nested = input_dir / "nested"
    nested.mkdir(parents=True)
    Image.new("RGB", (2, 2), color=(255, 0, 0)).save(input_dir / "B.png")
    Image.new("RGB", (2, 2), color=(64, 0, 0)).save(nested / "a.jpg")
    (input_dir / "corrupt.png").write_bytes(b"not an image")
    (input_dir / "ignore.txt").write_text("ignored", encoding="utf-8")
    warnings = io.StringIO()

    results = cli.predict_directory(input_dir, FakePredictor(), warnings)

    assert results == [
        {"image_path": "B.png", "pred": 1.0},
        {"image_path": "nested/a.jpg", "pred": pytest.approx(64 / 255)},
    ]
    assert "skipping corrupt image 'corrupt.png'" in warnings.getvalue()


def test_predict_directory_composites_transparency_over_white(tmp_path) -> None:
    input_dir = tmp_path / "images"
    input_dir.mkdir()
    Image.new("RGBA", (2, 2), color=(0, 0, 255, 0)).save(
        input_dir / "transparent.png"
    )

    results = cli.predict_directory(input_dir, FakePredictor())

    assert results == [{"image_path": "transparent.png", "pred": 1.0}]


def test_write_predictions_has_exact_schema(tmp_path) -> None:
    output_path = tmp_path / "nested" / "predictions.json"
    records = [{"image_path": "image.png", "pred": 0.25}]

    cli.write_predictions(output_path, records)

    decoded = json.loads(output_path.read_text(encoding="utf-8"))
    assert decoded == records
    assert set(decoded[0]) == {"image_path", "pred"}


def test_main_uses_checkpoint_and_writes_json(monkeypatch, tmp_path) -> None:
    input_dir = tmp_path / "images"
    input_dir.mkdir()
    Image.new("RGB", (1, 1), color=(128, 0, 0)).save(input_dir / "one.png")
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.touch()
    output = tmp_path / "predictions.json"
    observed = {}

    def fake_from_checkpoint(path, device):
        observed.update(path=path, device=device)
        return FakePredictor()

    monkeypatch.setattr(
        cli.Predictor, "from_checkpoint", staticmethod(fake_from_checkpoint)
    )

    exit_code = cli.main(
        [
            "--input",
            str(input_dir),
            "--output",
            str(output),
            "--checkpoint",
            str(checkpoint),
            "--device",
            "cpu",
        ]
    )

    assert exit_code == 0
    assert observed == {"path": checkpoint, "device": "cpu"}
    assert json.loads(output.read_text(encoding="utf-8"))[0]["image_path"] == "one.png"
