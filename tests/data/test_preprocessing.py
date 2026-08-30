from __future__ import annotations

import pytest
from PIL import Image, ImageDraw

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from src.data.preprocessing import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    PREPROCESSING_CONTRACT_ID,
    PREPROCESSING_METADATA_KEY,
    build_image_preprocess,
    normalize_pil_image,
    resolve_checkpoint_image_size,
)


def test_normalize_pil_image_applies_exif_orientation() -> None:
    image = Image.new("RGB", (3, 2), color=(12, 34, 56))
    image.getexif()[274] = 6

    normalized = normalize_pil_image(image)

    assert normalized.mode == "RGB"
    assert normalized.size == (2, 3)
    assert normalized.getexif().get(274) is None


def test_normalize_pil_image_composites_alpha_over_white() -> None:
    image = Image.new("RGBA", (2, 1), color=(255, 0, 0, 0))
    image.putpixel((1, 0), (0, 255, 0, 128))

    normalized = normalize_pil_image(image)

    assert normalized.mode == "RGB"
    assert normalized.getpixel((0, 0)) == (255, 255, 255)
    assert normalized.getpixel((1, 0)) == (127, 255, 127)


def test_current_checkpoint_preprocessing_contract_is_accepted() -> None:
    metadata = {
        PREPROCESSING_METADATA_KEY: PREPROCESSING_CONTRACT_ID,
        "image_size": "224",
    }

    assert resolve_checkpoint_image_size(metadata) == 224
    assert resolve_checkpoint_image_size(metadata, requested_image_size=224) == 224


def test_legacy_checkpoint_without_preprocessing_contract_is_rejected() -> None:
    with pytest.raises(ValueError, match="legacy checkpoints must be retrained"):
        resolve_checkpoint_image_size({"image_size": "224"})


def test_mismatched_checkpoint_preprocessing_contract_is_rejected() -> None:
    metadata = {
        PREPROCESSING_METADATA_KEY: "legacy-square-resize-v0",
        "image_size": "224",
    }

    with pytest.raises(ValueError, match="does not match"):
        resolve_checkpoint_image_size(metadata)


def test_requested_image_size_must_match_checkpoint() -> None:
    metadata = {
        PREPROCESSING_METADATA_KEY: PREPROCESSING_CONTRACT_ID,
        "image_size": "224",
    }

    with pytest.raises(ValueError, match="requested image_size 256"):
        resolve_checkpoint_image_size(metadata, requested_image_size=256)


def test_preprocess_uses_explicit_short_edge_resize_and_center_crop() -> None:
    pipeline = build_image_preprocess(32)

    assert isinstance(pipeline.transforms[0], transforms.Resize)
    assert pipeline.transforms[0].size == 32
    assert pipeline.transforms[0].interpolation == InterpolationMode.BILINEAR
    assert pipeline.transforms[0].antialias is True
    assert isinstance(pipeline.transforms[1], transforms.CenterCrop)
    assert pipeline.transforms[1].size == (32, 32)


@pytest.mark.parametrize("source_size", [(120, 60), (60, 120)])
def test_preprocess_preserves_geometry_before_center_crop(source_size) -> None:
    """The centered half should fill the crop for either source orientation."""

    width, height = source_size
    image = Image.new("RGB", source_size, color=(255, 0, 0))
    draw = ImageDraw.Draw(image)
    if width > height:
        draw.rectangle(
            (width // 4, 0, 3 * width // 4 - 1, height - 1),
            fill=(0, 255, 0),
        )
    else:
        draw.rectangle(
            (0, height // 4, width - 1, 3 * height // 4 - 1),
            fill=(0, 255, 0),
        )

    tensor = build_image_preprocess(32)(image)
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    restored = tensor * std + mean

    assert tensor.shape == (3, 32, 32)
    assert restored[1].mean().item() > 0.45
    assert restored[0].mean().item() < 0.08
    assert restored[2].mean().item() < 0.08


@pytest.mark.parametrize("image_size", [0, -1])
def test_preprocess_rejects_non_positive_size(image_size: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        build_image_preprocess(image_size)


@pytest.mark.parametrize("image_size", [True, 32.0, "32"])
def test_preprocess_rejects_non_integer_size(image_size) -> None:
    with pytest.raises(TypeError, match="integer"):
        build_image_preprocess(image_size)
