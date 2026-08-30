"""Render the deterministic evaluation grid for one input image."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from src.transforms.robustness import TRANSFORM_GRID, apply_transform  # noqa: E402


def build_contact_sheet(image: Image.Image, seed: int = 42, cell_width: int = 240) -> Image.Image:
    """Return a labelled contact sheet containing every evaluation severity."""
    rgb_image = image.convert("RGB")
    cell_height = max(1, round(rgb_image.height * cell_width / rgb_image.width))
    entries: list[tuple[str, Image.Image]] = [("clean", rgb_image)]
    for transform_name, severities in TRANSFORM_GRID.items():
        for severity in severities:
            label = f"{transform_name}: {severity}"
            entries.append((label, apply_transform(rgb_image, transform_name, severity, seed)))

    columns = 4
    label_height = 28
    rows = math.ceil(len(entries) / columns)
    sheet = Image.new("RGB", (columns * cell_width, rows * (cell_height + label_height)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (label, transformed) in enumerate(entries):
        row, column = divmod(index, columns)
        x = column * cell_width
        y = row * (cell_height + label_height)
        preview = transformed.resize((cell_width, cell_height), Image.Resampling.LANCZOS)
        sheet.paste(preview, (x, y))
        draw.text((x + 6, y + cell_height + 6), label, fill="black")
    return sheet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Input JPG/PNG image")
    parser.add_argument("--output", required=True, type=Path, help="PNG contact-sheet path")
    parser.add_argument("--seed", default=42, type=int, help="Transformation seed")
    args = parser.parse_args()

    with Image.open(args.input) as image:
        sheet = build_contact_sheet(image, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output, format="PNG")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
