"""Validated manifest-backed image dataset."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

from src.data.preprocessing import normalize_pil_image


REQUIRED_COLUMNS = frozenset({"path", "label", "split"})
ALLOWED_SPLITS = frozenset({"train", "val", "test"})


class ImageManifestDataset:
    """Map-style dataset yielding ``(RGB image or tensor, binary label)``.

    A plain map-style object is sufficient for ``torch.utils.data.DataLoader``;
    avoiding a top-level torch import also keeps manifest validation lightweight.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        split: str | None = None,
        transform: Callable[[Image.Image], Any] | None = None,
        root_dir: str | Path | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        self.transform = transform
        self.root_dir = (
            Path(root_dir).resolve()
            if root_dir is not None
            else self.manifest_path.parent.parent.parent.resolve()
        )

        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")
        if split is not None and split not in ALLOWED_SPLITS:
            allowed = ", ".join(sorted(ALLOWED_SPLITS))
            raise ValueError(f"Unknown split '{split}'; expected one of: {allowed}")

        frame = pd.read_csv(self.manifest_path)
        missing = REQUIRED_COLUMNS - set(frame.columns)
        if missing:
            raise ValueError(
                f"Manifest is missing required column(s): {', '.join(sorted(missing))}"
            )
        if frame.empty:
            raise ValueError("Manifest contains no rows")

        frame = frame.copy()
        if frame["path"].isna().any() or (
            frame["path"].astype(str).str.strip() == ""
        ).any():
            raise ValueError("Manifest paths must be non-empty strings")
        frame["path"] = frame["path"].astype(str)

        numeric_labels = pd.to_numeric(frame["label"], errors="coerce")
        if numeric_labels.isna().any() or (numeric_labels % 1 != 0).any():
            raise ValueError("Manifest labels must be integers 0 or 1")
        frame["label"] = numeric_labels.astype(int)
        if not set(frame["label"].unique()).issubset({0, 1}):
            raise ValueError("Manifest labels must contain only 0 and 1")

        if frame["split"].isna().any():
            raise ValueError("Manifest split values must not be empty")
        frame["split"] = frame["split"].astype(str).str.strip()
        unknown_splits = set(frame["split"].unique()) - ALLOWED_SPLITS
        if unknown_splits:
            raise ValueError(
                "Manifest contains unknown split(s): "
                + ", ".join(sorted(unknown_splits))
            )
        if frame["path"].duplicated().any():
            raise ValueError("Manifest contains duplicate image paths")
        if "sha256" in frame:
            if frame["sha256"].isna().any() or (
                frame["sha256"].astype(str).str.strip() == ""
            ).any():
                raise ValueError("Manifest image hashes must not be empty")
            if frame["sha256"].duplicated().any():
                raise ValueError("Manifest contains duplicate image hashes")
        if "source_id" in frame:
            if frame["source_id"].isna().any() or (
                frame["source_id"].astype(str).str.strip() == ""
            ).any():
                raise ValueError("Manifest source_id values must not be empty")
            source_split_counts = frame.groupby("source_id")["split"].nunique()
            if (source_split_counts > 1).any():
                raise ValueError("A source_id appears in more than one split")

        if split is not None:
            frame = frame.loc[frame["split"] == split].reset_index(drop=True)
            if frame.empty:
                raise ValueError(f"Manifest contains no rows for split '{split}'")

        self.split = split
        self.df = frame
        self._paths = [self._safe_path(value) for value in frame["path"]]

    def _safe_path(self, manifest_value: str) -> Path:
        relative = Path(manifest_value)
        if relative.is_absolute():
            raise ValueError("Manifest image paths must be relative to root_dir")
        resolved = (self.root_dir / relative).resolve()
        try:
            resolved.relative_to(self.root_dir)
        except ValueError as error:
            raise ValueError(
                f"Manifest path escapes root_dir: {manifest_value}"
            ) from error
        return resolved

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[Any, int]:
        image_path = self._paths[idx]
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")

        try:
            with Image.open(image_path) as opened:
                opened.load()
                image = normalize_pil_image(opened)
        except OSError as error:
            raise OSError(f"Image could not be decoded: {image_path}") from error

        label = int(self.df.iloc[idx]["label"])
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def compute_sha256(file_path: str | Path) -> str:
    """Return a chunked SHA-256 digest for a file."""

    hasher = hashlib.sha256()
    with Path(file_path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()
