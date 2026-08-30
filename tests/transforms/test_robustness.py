from __future__ import annotations

import hashlib

import numpy as np
import pytest
from PIL import Image

from src.transforms.robustness import TRANSFORM_GRID, apply_transform, sample_training_transform


@pytest.fixture()
def sample_image() -> Image.Image:
    gradient = np.zeros((37, 53, 3), dtype=np.uint8)
    gradient[..., 0] = np.arange(53, dtype=np.uint8)
    gradient[..., 1] = np.arange(37, dtype=np.uint8)[:, None]
    gradient[..., 2] = 180
    return Image.fromarray(gradient, mode="RGB")


@pytest.mark.parametrize(
    ("transform_name", "severity"),
    [
        (transform_name, severity)
        for transform_name, severities in TRANSFORM_GRID.items()
        for severity in severities
    ],
)
def test_every_transform_preserves_rgb_dimensions_and_pixel_range(
    sample_image: Image.Image, transform_name: str, severity: float | int
) -> None:
    output = apply_transform(sample_image, transform_name, severity, seed=123)

    assert output.mode == "RGB"
    assert output.size == sample_image.size
    pixels = np.asarray(output)
    assert pixels.dtype == np.uint8
    assert pixels.min() >= 0
    assert pixels.max() <= 255


@pytest.mark.parametrize(
    ("transform_name", "severity"),
    [(name, values[0]) for name, values in TRANSFORM_GRID.items()],
)
def test_transforms_are_deterministic(
    sample_image: Image.Image, transform_name: str, severity: float | int
) -> None:
    first = apply_transform(sample_image, transform_name, severity, seed=91)
    second = apply_transform(sample_image, transform_name, severity, seed=91)

    assert hashlib.sha256(first.tobytes()).digest() == hashlib.sha256(second.tobytes()).digest()


def test_noise_changes_when_seed_changes(sample_image: Image.Image) -> None:
    first = apply_transform(sample_image, "gaussian_noise", 0.10, seed=1)
    second = apply_transform(sample_image, "gaussian_noise", 0.10, seed=2)

    assert first.tobytes() != second.tobytes()


def test_non_rgb_inputs_are_converted_to_rgb() -> None:
    grayscale = Image.new("L", (20, 10), color=127)
    output = apply_transform(grayscale, "jpeg", 70)

    assert output.mode == "RGB"
    assert output.size == grayscale.size


@pytest.mark.parametrize(
    ("name", "severity", "match"),
    [
        ("not-a-transform", 1, "unknown transform"),
        ("jpeg", 12, "invalid severity"),
    ],
)
def test_invalid_transform_requests_fail_clearly(
    sample_image: Image.Image, name: str, severity: float | int, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        apply_transform(sample_image, name, severity)


def test_training_sampler_is_reproducible(sample_image: Image.Image) -> None:
    first_image, first_metadata = sample_training_transform(sample_image, seed=812)
    second_image, second_metadata = sample_training_transform(sample_image, seed=812)

    assert first_metadata == second_metadata
    assert first_image.tobytes() == second_image.tobytes()
    assert first_image.size == sample_image.size
    assert first_image.mode == "RGB"


def test_training_sampler_can_be_forced_clean(sample_image: Image.Image) -> None:
    output, metadata = sample_training_transform(sample_image, seed=7, clean_probability=1.0)

    assert metadata == {"transform": "clean", "severity": None, "seed": 7}
    assert output.tobytes() == sample_image.tobytes()


@pytest.mark.parametrize("probability", [-0.1, 1.1])
def test_training_sampler_rejects_invalid_clean_probability(
    sample_image: Image.Image, probability: float
) -> None:
    with pytest.raises(ValueError, match="clean_probability"):
        sample_training_transform(sample_image, seed=1, clean_probability=probability)
