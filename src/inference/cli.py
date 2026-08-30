"""Directory-to-JSON command line inference."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO

from PIL import Image, UnidentifiedImageError

from src.data.preprocessing import normalize_pil_image

from .predictor import DEFAULT_CHECKPOINT_PATH, Predictor


SUPPORTED_EXTENSIONS = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


def iter_image_paths(input_dir: str | Path) -> list[Path]:
    """Return supported files recursively in deterministic relative-path order."""

    root = Path(input_dir)
    if not root.is_dir():
        raise NotADirectoryError(f"input directory not found: {root}")

    paths = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(
        paths,
        key=lambda path: (
            path.relative_to(root).as_posix().casefold(),
            path.relative_to(root).as_posix(),
        ),
    )


def predict_directory(
    input_dir: str | Path,
    predictor: Predictor,
    warning_stream: TextIO | None = None,
) -> list[dict[str, str | float]]:
    """Predict valid images and warn while skipping corrupt image files.

    ``image_path`` values are POSIX-style paths relative to ``input_dir`` so
    output remains portable across Kaggle, Windows, and Streamlit deployments.
    """

    root = Path(input_dir)
    warnings = warning_stream if warning_stream is not None else sys.stderr
    predictions: list[dict[str, str | float]] = []

    for image_path in iter_image_paths(root):
        relative_path = image_path.relative_to(root).as_posix()
        try:
            with Image.open(image_path) as opened:
                opened.load()
                image = normalize_pil_image(opened)
        except (UnidentifiedImageError, OSError, ValueError) as error:
            print(
                f"warning: skipping corrupt image '{relative_path}': {error}",
                file=warnings,
            )
            continue

        predictions.append(
            {
                "image_path": relative_path,
                "pred": float(predictor.predict_pil(image)),
            }
        )

    return predictions


def write_predictions(
    output_path: str | Path,
    predictions: Sequence[dict[str, str | float]],
) -> Path:
    """Write prediction records as strict, human-readable JSON."""

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(list(predictions), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score every supported image in a directory."
    )
    parser.add_argument("--input", required=True, type=Path, help="image directory")
    parser.add_argument(
        "--output", required=True, type=Path, help="destination JSON file"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT_PATH,
        help=f"safetensors weights (default: {DEFAULT_CHECKPOINT_PATH})",
    )
    parser.add_argument("--device", default="cpu", help="torch device (default: cpu)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.input.is_dir():
        print(f"error: input directory not found: {args.input}", file=sys.stderr)
        return 2

    predictor = Predictor.from_checkpoint(args.checkpoint, device=args.device)
    predictions = predict_directory(args.input, predictor)
    write_predictions(args.output, predictions)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
