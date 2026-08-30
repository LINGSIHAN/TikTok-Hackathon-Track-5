from __future__ import annotations

import io

import pytest
from PIL import Image

from scripts.prepare_sid_subset import _open_example_image
from src.data.preprocessing import normalize_pil_image


def test_normalize_applies_exif_orientation() -> None:
    image = Image.new("RGB", (3, 2), color=(12, 34, 56))
    image.getexif()[274] = 6

    normalized = normalize_pil_image(image)

    assert normalized.mode == "RGB"
    assert normalized.size == (2, 3)
    assert normalized.getexif().get(274) is None


@pytest.mark.parametrize("mode", ["RGBA", "RGBa", "LA", "La"])
def test_normalize_composites_alpha_modes_over_white(mode: str) -> None:
    image = Image.new(mode, (1, 1))

    normalized = normalize_pil_image(image)

    assert normalized.mode == "RGB"
    assert normalized.getpixel((0, 0)) == (255, 255, 255)


def test_normalize_rejects_animated_images() -> None:
    payload = io.BytesIO()
    frames = [
        Image.new("RGB", (2, 2), color=(255, 0, 0)),
        Image.new("RGB", (2, 2), color=(0, 255, 0)),
    ]
    frames[0].save(payload, format="GIF", save_all=True, append_images=frames[1:])
    payload.seek(0)

    with Image.open(payload) as animated:
        with pytest.raises(ValueError, match="animated images"):
            normalize_pil_image(animated)


def test_dataset_preparation_uses_the_shared_alpha_policy() -> None:
    source = Image.new("RGBA", (2, 1), color=(255, 0, 0, 0))
    source.putpixel((1, 0), (0, 255, 0, 128))

    normalized = _open_example_image(source)

    assert normalized.getpixel((0, 0)) == (255, 255, 255)
    assert normalized.getpixel((1, 0)) == (127, 255, 127)
