"""Single-image and robustness inference for the trained detector."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping
from importlib import import_module
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import Tensor, nn
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from src.models.checkpoints import load_checkpoint
from src.models.efficientnet import build_model


DEFAULT_CHECKPOINT_PATH = Path("artifacts/checkpoints/model.safetensors")
DEFAULT_IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

Preprocess = Callable[[Image.Image], Tensor]
ApplyTransform = Callable[[Image.Image, str, Any, int], Image.Image]


def build_inference_preprocess(
    image_size: int = DEFAULT_IMAGE_SIZE,
) -> transforms.Compose:
    """Return preprocessing aligned with the repository's training pipeline."""

    if image_size <= 0:
        raise ValueError("image_size must be positive")

    return transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size), interpolation=InterpolationMode.BILINEAR
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


class Predictor:
    """CPU/GPU-safe inference facade shared by the CLI and Streamlit UI."""

    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        preprocess: Preprocess | None = None,
        transform_grid: Mapping[str, Iterable[Any]] | None = None,
        apply_transform_fn: ApplyTransform | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()
        self.preprocess = preprocess or build_inference_preprocess()
        self._transform_grid = transform_grid
        self._apply_transform = apply_transform_fn

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path = DEFAULT_CHECKPOINT_PATH,
        device: str = "cpu",
    ) -> "Predictor":
        """Construct the fixed detector architecture and load its weights."""

        checkpoint_path = Path(path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

        model = build_model(pretrained=False)
        metadata = load_checkpoint(model, checkpoint_path, device=device)
        try:
            image_size = int(metadata.get("image_size", DEFAULT_IMAGE_SIZE))
        except (TypeError, ValueError) as error:
            raise ValueError("checkpoint metadata contains an invalid image_size") from error
        return cls(
            model=model,
            device=device,
            preprocess=build_inference_preprocess(image_size),
        )

    def predict_pil(self, image: Image.Image) -> float:
        """Return the probability that ``image`` is AI-generated."""

        if not isinstance(image, Image.Image):
            raise TypeError("image must be a PIL.Image.Image")

        return self.predict_many_pil([image], batch_size=1)[0]

    def predict_many_pil(
        self,
        images: Iterable[Image.Image],
        batch_size: int = 8,
    ) -> list[float]:
        """Predict several PIL images in bounded batches for CPU deployment."""

        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise TypeError("batch_size must be an integer")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        tensors: list[Tensor] = []
        for image in images:
            if not isinstance(image, Image.Image):
                raise TypeError("every image must be a PIL.Image.Image")
            tensor = self.preprocess(image.convert("RGB"))
            if not isinstance(tensor, Tensor):
                raise TypeError("preprocess must return a torch.Tensor")
            if tensor.ndim != 3:
                raise ValueError(
                    "preprocess must return a CHW tensor, "
                    f"received shape {tuple(tensor.shape)}"
                )
            tensors.append(tensor)

        probabilities: list[float] = []
        for offset in range(0, len(tensors), batch_size):
            batch_tensors = tensors[offset : offset + batch_size]
            batch = torch.stack(batch_tensors).to(self.device)
            with torch.inference_mode():
                logits = self.model(batch)

            if not isinstance(logits, Tensor):
                raise TypeError("model must return a torch.Tensor of logits")
            if logits.numel() != len(batch_tensors):
                raise ValueError(
                    "model must return exactly one logit per input image; "
                    f"received shape {tuple(logits.shape)} for "
                    f"batch size {len(batch_tensors)}"
                )

            batch_probabilities = torch.sigmoid(logits.reshape(-1)).tolist()
            if not all(math.isfinite(value) for value in batch_probabilities):
                raise RuntimeError("model returned a non-finite probability")
            probabilities.extend(float(value) for value in batch_probabilities)

        return probabilities

    def stress_test(self, image: Image.Image) -> list[dict[str, str | float]]:
        """Predict every transform/severity pair in ``TRANSFORM_GRID``."""

        if not isinstance(image, Image.Image):
            raise TypeError("image must be a PIL.Image.Image")

        transform_grid, apply_transform = self._resolve_transform_suite()
        descriptors: list[tuple[str, Any]] = []
        transformed_images: list[Image.Image] = []

        for transform_name, severities in transform_grid.items():
            for severity in severities:
                descriptors.append((str(transform_name), severity))
                transformed_images.append(
                    apply_transform(
                        image.convert("RGB"), transform_name, severity, 42
                    )
                )

        probabilities = self.predict_many_pil(transformed_images)
        return [
            {
                "transform": transform_name,
                "severity": str(severity),
                "pred": probability,
            }
            for (transform_name, severity), probability in zip(
                descriptors, probabilities, strict=True
            )
        ]

    def _resolve_transform_suite(
        self,
    ) -> tuple[Mapping[str, Iterable[Any]], ApplyTransform]:
        if self._transform_grid is not None and self._apply_transform is not None:
            return self._transform_grid, self._apply_transform

        robustness = import_module("src.transforms.robustness")
        transform_grid = (
            self._transform_grid
            if self._transform_grid is not None
            else robustness.TRANSFORM_GRID
        )
        apply_transform = (
            self._apply_transform
            if self._apply_transform is not None
            else robustness.apply_transform
        )
        return transform_grid, apply_transform
