from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from src.data.dataset import ImageManifestDataset, compute_sha256


def _write_image(path: Path, color: tuple[int, int, int], mode: str = "RGB") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new(mode, (12, 10), color=color if mode == "RGB" else color[0])
    image.save(path)


def _write_manifest(root: Path, rows: list[dict]) -> Path:
    manifest = root / "data" / "processed" / "manifest.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(manifest, index=False)
    return manifest


def _valid_rows() -> list[dict]:
    return [
        {
            "path": "data/raw/authentic/a.png",
            "label": 0,
            "split": "train",
            "source_id": "source-a",
            "sha256": "hash-a",
        },
        {
            "path": "data/raw/generated/b.png",
            "label": 1,
            "split": "val",
            "source_id": "source-b",
            "sha256": "hash-b",
        },
    ]


def test_dataset_preserves_order_converts_rgb_and_applies_transform(tmp_path):
    _write_image(tmp_path / "data/raw/authentic/a.png", (32, 0, 0), mode="L")
    _write_image(tmp_path / "data/raw/generated/b.png", (0, 64, 0))
    manifest = _write_manifest(tmp_path, _valid_rows())

    dataset = ImageManifestDataset(
        manifest,
        split="train",
        transform=lambda image: (image.mode, image.size),
    )

    assert len(dataset) == 1
    transformed, label = dataset[0]
    assert transformed == ("RGB", (12, 10))
    assert label == 0
    assert list(dataset.df["path"]) == ["data/raw/authentic/a.png"]


def test_dataset_composites_transparent_images_over_white(tmp_path):
    image_path = tmp_path / "data/raw/authentic/a.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGBA", (2, 2), color=(0, 0, 255, 0)).save(image_path)
    manifest = _write_manifest(tmp_path, [_valid_rows()[0]])

    image, label = ImageManifestDataset(manifest, split="train")[0]

    assert image.mode == "RGB"
    assert image.getpixel((0, 0)) == (255, 255, 255)
    assert label == 0


def test_compute_sha256_matches_known_content(tmp_path):
    path = tmp_path / "sample.bin"
    path.write_bytes(b"abc")

    assert (
        compute_sha256(path)
        == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


@pytest.mark.parametrize("missing_column", ["path", "label", "split"])
def test_required_columns_are_validated(tmp_path, missing_column):
    row = _valid_rows()[0]
    row.pop(missing_column)
    manifest = _write_manifest(tmp_path, [row])

    with pytest.raises(ValueError, match="missing required"):
        ImageManifestDataset(manifest)


@pytest.mark.parametrize("label", [-1, 2, 0.5, "unknown"])
def test_binary_labels_are_validated(tmp_path, label):
    row = _valid_rows()[0]
    row["label"] = label
    manifest = _write_manifest(tmp_path, [row])

    with pytest.raises(ValueError, match="labels"):
        ImageManifestDataset(manifest)


def test_unknown_and_empty_splits_fail_clearly(tmp_path):
    manifest = _write_manifest(tmp_path, [_valid_rows()[0]])

    with pytest.raises(ValueError, match="Unknown split"):
        ImageManifestDataset(manifest, split="validation")
    with pytest.raises(ValueError, match="no rows for split 'test'"):
        ImageManifestDataset(manifest, split="test")


def test_path_traversal_is_rejected(tmp_path):
    row = _valid_rows()[0]
    row["path"] = "../outside.png"
    manifest = _write_manifest(tmp_path, [row])

    with pytest.raises(ValueError, match="escapes root_dir"):
        ImageManifestDataset(manifest)


def test_source_groups_cannot_cross_splits(tmp_path):
    rows = _valid_rows()
    rows[1]["source_id"] = rows[0]["source_id"]
    manifest = _write_manifest(tmp_path, rows)

    with pytest.raises(ValueError, match="source_id"):
        ImageManifestDataset(manifest)


def test_missing_and_corrupt_images_fail_at_access(tmp_path):
    manifest = _write_manifest(tmp_path, [_valid_rows()[0]])
    dataset = ImageManifestDataset(manifest, split="train")
    with pytest.raises(FileNotFoundError, match="Image not found"):
        dataset[0]

    image_path = tmp_path / "data/raw/authentic/a.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_text("not an image", encoding="utf-8")
    with pytest.raises(OSError, match="could not be decoded"):
        dataset[0]
