import hashlib
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset


class ImageManifestDataset(Dataset):
    """Generic dataset reader that parses manifest.csv and yields PIL images/tensors."""

    def __init__(
        self,
        manifest_path: Union[str, Path],
        split: Optional[str] = None,
        transform: Optional[Callable] = None,
        root_dir: Optional[Union[str, Path]] = None,
    ):
        self.manifest_path = Path(manifest_path)
        self.split = split
        self.transform = transform
        self.root_dir = Path(root_dir) if root_dir else self.manifest_path.parent.parent.parent

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found at {self.manifest_path}")

        df = pd.read_csv(self.manifest_path)

        if self.split:
            df = df[df["split"] == self.split].reset_index(drop=True)

        self.df = df

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[Union[Image.Image, torch.Tensor], int]:
        row = self.df.iloc[idx]
        img_path = self.root_dir / row["path"]

        if not img_path.exists():
            raise FileNotFoundError(f"Image not found at {img_path}")

        image = Image.open(img_path).convert("RGB")
        label = int(row["label"])

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def compute_sha256(file_path: Path) -> str:
    """Computes SHA256 hash for deduplication."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()