"""Shared geometry-safe image preprocessing for every model entrypoint.

Robustness transformations operate on the source image before this pipeline.
The shorter edge is then resized while preserving aspect ratio, followed by a
center crop to the model's square input size.  Keeping this in one module
prevents training, evaluation, and inference from drifting apart.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from PIL import Image, ImageOps

if TYPE_CHECKING:
    from torchvision.transforms import Compose


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
PREPROCESSING_METADATA_KEY = "preprocessing_contract"
PREPROCESSING_CONTRACT_ID = (
    "pil-exif-white-alpha-short-edge-bilinear-center-crop-imagenet-v1"
)


def normalize_pil_image(image: Image.Image) -> Image.Image:
    """Return an orientation-correct, white-composited RGB copy of ``image``."""

    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL.Image.Image")
    if getattr(image, "is_animated", False):
        raise ValueError("animated images are not supported")

    oriented = ImageOps.exif_transpose(image)
    has_alpha = any(band.casefold() == "a" for band in oriented.getbands())
    if has_alpha or "transparency" in oriented.info:
        if oriented.mode == "La":
            oriented = oriented.convert("LA")
        foreground = oriented.convert("RGBA")
        background = Image.new("RGBA", foreground.size, (255, 255, 255, 255))
        return Image.alpha_composite(background, foreground).convert("RGB")
    return oriented.convert("RGB")


def _validate_image_size(image_size: int) -> int:
    if isinstance(image_size, bool) or not isinstance(image_size, int):
        raise TypeError("image_size must be an integer")
    if image_size <= 0:
        raise ValueError("image_size must be positive")
    return image_size


def resolve_checkpoint_image_size(
    metadata: Mapping[str, object],
    *,
    requested_image_size: int | None = None,
) -> int:
    """Validate preprocessing lineage and return the checkpoint input size.

    Checkpoints without the current contract are rejected deliberately: model
    weights trained with a different geometry or alpha policy cannot be paired
    safely with this preprocessing implementation.
    """

    if not isinstance(metadata, Mapping):
        raise TypeError("checkpoint metadata must be a mapping")

    contract = metadata.get(PREPROCESSING_METADATA_KEY)
    if contract is None:
        raise ValueError(
            "checkpoint metadata is missing the preprocessing contract; "
            "legacy checkpoints must be retrained with the current pipeline"
        )
    if contract != PREPROCESSING_CONTRACT_ID:
        raise ValueError(
            f"checkpoint preprocessing contract {contract!r} does not match "
            f"the required contract {PREPROCESSING_CONTRACT_ID!r}; retrain or "
            "select a compatible checkpoint"
        )

    raw_image_size = metadata.get("image_size")
    if isinstance(raw_image_size, bool):
        raise ValueError("checkpoint metadata contains an invalid image_size")
    try:
        checkpoint_image_size = int(raw_image_size)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise ValueError(
            "checkpoint metadata contains an invalid image_size"
        ) from error
    if checkpoint_image_size <= 0:
        raise ValueError("checkpoint metadata contains an invalid image_size")

    if requested_image_size is not None:
        requested = _validate_image_size(requested_image_size)
        if requested != checkpoint_image_size:
            raise ValueError(
                f"requested image_size {requested} does not match checkpoint "
                f"image_size {checkpoint_image_size}"
            )
    return checkpoint_image_size


def build_image_preprocess(image_size: int) -> "Compose":
    """Build the canonical aspect-preserving model preprocessing pipeline."""

    image_size = _validate_image_size(image_size)

    from torchvision import transforms
    from torchvision.transforms import InterpolationMode

    return transforms.Compose(
        [
            transforms.Resize(
                image_size,
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            ),
            transforms.CenterCrop((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
