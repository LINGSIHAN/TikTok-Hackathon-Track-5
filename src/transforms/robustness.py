"""Deterministic, image-size-preserving robustness transformations."""

from __future__ import annotations

from io import BytesIO
from numbers import Real
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


TRANSFORM_GRID: dict[str, tuple[float | int, ...]] = {
    "jpeg": (90, 70, 50, 30),
    "gaussian_blur": (0.5, 1.0, 2.0),
    "resize": (0.5, 0.25),
    "gaussian_noise": (0.02, 0.05, 0.10),
    "brightness": (-0.2, 0.2),
    "contrast": (-0.2, 0.2),
    "saturation": (-0.2, 0.2),
    "center_crop": (0.8,),
}

_UPSAMPLE_RESAMPLE = Image.Resampling.LANCZOS
_DOWNSAMPLE_RESAMPLE = Image.Resampling.LANCZOS


def apply_transform(
    image: Image.Image,
    transform_name: str,
    severity: float | int,
    seed: int = 42,
) -> Image.Image:
    """Apply one deterministic evaluation transformation to an RGB PIL image.

    Args:
        image: Input image. Non-RGB inputs are converted to RGB.
        transform_name: A key in :data:`TRANSFORM_GRID`.
        severity: One of the published severities for ``transform_name``.
        seed: Seed used by stochastic transformations, currently Gaussian noise.

    Returns:
        An RGB image with exactly the same dimensions as the input.
    """
    _validate_request(image, transform_name, severity, seed)
    rgb_image = image.convert("RGB")
    original_size = rgb_image.size

    if transform_name == "jpeg":
        output = _jpeg_compress(rgb_image, int(severity))
    elif transform_name == "gaussian_blur":
        output = rgb_image.filter(ImageFilter.GaussianBlur(radius=float(severity)))
    elif transform_name == "resize":
        output = _resize_and_restore(rgb_image, float(severity))
    elif transform_name == "gaussian_noise":
        output = _add_gaussian_noise(rgb_image, float(severity), seed)
    elif transform_name == "brightness":
        output = ImageEnhance.Brightness(rgb_image).enhance(1.0 + float(severity))
    elif transform_name == "contrast":
        output = ImageEnhance.Contrast(rgb_image).enhance(1.0 + float(severity))
    elif transform_name == "saturation":
        output = ImageEnhance.Color(rgb_image).enhance(1.0 + float(severity))
    else:  # center_crop; all names were validated above.
        output = _center_crop_and_restore(rgb_image, float(severity))

    if output.size != original_size:
        output = output.resize(original_size, _UPSAMPLE_RESAMPLE)
    return output.convert("RGB")


def sample_training_transform(
    image: Image.Image,
    seed: int,
    clean_probability: float = 0.35,
) -> tuple[Image.Image, dict[str, Any]]:
    """Sample a reproducible clean or transformed training view.

    ``clean_probability`` defaults to the agreed 35% clean / 65% transformed
    mix. The returned metadata is suitable for experiment logging.
    """
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL.Image.Image")
    if not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if not isinstance(clean_probability, Real) or not 0.0 <= clean_probability <= 1.0:
        raise ValueError("clean_probability must be between 0.0 and 1.0")

    rng = np.random.default_rng(seed)
    if rng.random() < clean_probability:
        return image.convert("RGB").copy(), {
            "transform": "clean",
            "severity": None,
            "seed": seed,
        }

    transform_names = tuple(TRANSFORM_GRID)
    transform_name = transform_names[int(rng.integers(len(transform_names)))]
    severities = TRANSFORM_GRID[transform_name]
    severity = severities[int(rng.integers(len(severities)))]
    transform_seed = int(rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))
    return apply_transform(image, transform_name, severity, seed=transform_seed), {
        "transform": transform_name,
        "severity": severity,
        "seed": transform_seed,
    }


def _validate_request(
    image: Image.Image,
    transform_name: str,
    severity: float | int,
    seed: int,
) -> None:
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL.Image.Image")
    if transform_name not in TRANSFORM_GRID:
        available = ", ".join(TRANSFORM_GRID)
        raise ValueError(f"unknown transform '{transform_name}'; expected one of: {available}")
    if severity not in TRANSFORM_GRID[transform_name]:
        allowed = ", ".join(str(value) for value in TRANSFORM_GRID[transform_name])
        raise ValueError(
            f"invalid severity {severity!r} for '{transform_name}'; expected one of: {allowed}"
        )
    if not isinstance(seed, int):
        raise TypeError("seed must be an integer")


def _jpeg_compress(image: Image.Image, quality: int) -> Image.Image:
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=quality, optimize=False, progressive=False)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return decoded.convert("RGB").copy()


def _resize_and_restore(image: Image.Image, scale: float) -> Image.Image:
    width, height = image.size
    reduced_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    reduced = image.resize(reduced_size, _DOWNSAMPLE_RESAMPLE)
    return reduced.resize((width, height), _UPSAMPLE_RESAMPLE)


def _add_gaussian_noise(image: Image.Image, sigma: float, seed: int) -> Image.Image:
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    noise = np.random.default_rng(seed).normal(0.0, sigma, size=pixels.shape)
    noisy = np.clip(pixels + noise, 0.0, 1.0)
    return Image.fromarray(np.rint(noisy * 255.0).astype(np.uint8), mode="RGB")


def _center_crop_and_restore(image: Image.Image, fraction: float) -> Image.Image:
    width, height = image.size
    crop_width = max(1, round(width * fraction))
    crop_height = max(1, round(height * fraction))
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    crop = image.crop((left, top, left + crop_width, top + crop_height))
    return crop.resize((width, height), _UPSAMPLE_RESAMPLE)
