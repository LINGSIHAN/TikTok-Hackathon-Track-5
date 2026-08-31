from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

from scripts.prepare_genimage_v2 import (
    DATASET_ID,
    DATASET_VERSION,
    DEFAULT_GENERATED_PER_GENERATOR,
    DEFAULT_REAL_TOTAL,
    EXPECTED_FILE_COUNT,
    EXPECTED_GENERATOR_IMAGES,
    EXPECTED_NATURE_IMAGES,
    EXPECTED_TOTAL_BYTES,
    GENERATOR_DIRECTORIES,
    GENIMAGE_MANIFEST_RELATIVE,
    GENIMAGE_SUMMARY_RELATIVE,
    IMAGE_ROOT_RELATIVE,
    METADATA_FILENAME,
    METADATA_SHA256,
    NATURE_DIRECTORY,
    SIDManifestSpec,
    SelectionSpec,
    TRAINING_MANIFEST_RELATIVE,
    InventorySpec,
    file_sha256,
    prepare_genimage_v2,
    validate_inventory,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SMALL_GENERATOR_COUNT = 12
SMALL_NATURE_COUNT = 80
SMALL_GENERATOR_SELECTION = 10
SMALL_REAL_SELECTION = 70
SMALL_SID_COUNTS = {
    ("train", 0): 8,
    ("train", 1): 8,
    ("val", 0): 1,
    ("val", 1): 1,
    ("test", 0): 1,
    ("test", 1): 1,
}
SMALL_SID_SPEC = SIDManifestSpec(
    total=20,
    split_label_counts=SMALL_SID_COUNTS,
    require_summary=False,
)
SMALL_SELECTION_SPEC = SelectionSpec(
    generated_per_generator=SMALL_GENERATOR_SELECTION,
    real_total=SMALL_REAL_SELECTION,
)


def _image(index: int) -> Image.Image:
    """Return a tiny but JPEG-distinct deterministic image."""

    image = Image.new(
        "RGB",
        (12, 12),
        ((index * 29) % 256, (index * 71) % 256, (index * 113) % 256),
    )
    pixels = image.load()
    for bit in range(8):
        color = (255, 255, 255) if index & (1 << bit) else (0, 0, 0)
        x = 1 + (bit % 4) * 3
        y = 1 + (bit // 4) * 5
        for dx in range(2):
            for dy in range(2):
                pixels[x + dx, y + dy] = color
    return image


def _save_source(path: Path, index: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _image(index).save(path, format="JPEG", quality=96, subsampling=0)


def _build_genimage_tree(root: Path) -> tuple[Path, InventorySpec]:
    dataset_root = root / "attached" / "dataset"
    index = 0
    for generator in GENERATOR_DIRECTORIES:
        for item in range(SMALL_GENERATOR_COUNT):
            filename = f"{item % 10}_{generator}_{item:03d}.jpg"
            _save_source(dataset_root / generator / filename, index)
            index += 1

    nature_rows = []
    for item in range(SMALL_NATURE_COUNT):
        filename = f"nature-{item:03d}.jpg"
        _save_source(dataset_root / NATURE_DIRECTORY / filename, index)
        nature_rows.append(
            {
                "filename": filename,
                "class": str(item % 10),
                "width": 12,
                "height": 12,
                "quality": 96,
            }
        )
        index += 1
    metadata_path = dataset_root / METADATA_FILENAME
    with metadata_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("filename", "class", "width", "height", "quality"),
        )
        writer.writeheader()
        writer.writerows(nature_rows)
    return dataset_root.parent, _inventory_spec(dataset_root)


def _inventory_spec(dataset_root: Path) -> InventorySpec:
    files = sorted(path for path in dataset_root.rglob("*") if path.is_file())
    return InventorySpec(
        generator_images=SMALL_GENERATOR_COUNT,
        nature_images=SMALL_NATURE_COUNT,
        file_count=len(files),
        total_bytes=sum(path.stat().st_size for path in files),
        metadata_sha256=file_sha256(dataset_root / METADATA_FILENAME),
    )


def _normalized_bytes(image: Image.Image) -> bytes:
    from io import BytesIO

    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=95, subsampling=0)
    return buffer.getvalue()


def _build_sid_manifest(output_root: Path) -> Path:
    rows = []
    item = 0
    for split, per_label in (("train", 8), ("val", 1), ("test", 1)):
        for label in (0, 1):
            for _ in range(per_label):
                image = _image(10_000 + item)
                payload = _normalized_bytes(image)
                digest = hashlib.sha256(payload).hexdigest()
                label_directory = "authentic" if label == 0 else "generated"
                relative = Path("data/raw") / label_directory / f"{digest}.jpg"
                destination = output_root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(payload)
                rows.append(
                    {
                        "path": relative.as_posix(),
                        "label": label,
                        "split": split,
                        "dataset": "saberzl/SID_Set",
                        "source_split": "train",
                        "source_id": f"sid-{item:03d}",
                        "width": 12,
                        "height": 12,
                        "sha256": digest,
                    }
                )
                item += 1
    manifest_path = output_root / "data/processed/manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return manifest_path


def _prepare(input_root: Path, output_root: Path, inventory_spec: InventorySpec):
    sid_manifest = _build_sid_manifest(output_root)
    return prepare_genimage_v2(
        input_root=input_root,
        sid_manifest=sid_manifest,
        output_root=output_root,
        license_confirmed=True,
        inventory_spec=inventory_spec,
        selection_spec=SMALL_SELECTION_SPEC,
        sid_spec=SMALL_SID_SPEC,
    )


def test_pinned_public_inventory_constants() -> None:
    assert DATASET_ID == "cartografia/unbiased-tiny-genimage"
    assert DATASET_VERSION == 1
    assert GENERATOR_DIRECTORIES == (
        "ADM",
        "BigGAN",
        "Midjourney",
        "VQDM",
        "glide",
        "stable_diffusion_v_1_5",
        "wukong",
    )
    assert EXPECTED_GENERATOR_IMAGES == 2_500
    assert EXPECTED_NATURE_IMAGES == 5_828
    assert EXPECTED_FILE_COUNT == 23_329
    assert EXPECTED_TOTAL_BYTES == 2_528_629_592
    assert METADATA_SHA256 == (
        "5f9a46e53e624339f6db8cc4d4a4fe5e54a0371e4b07a7da278300f6ed699e91"
    )
    assert DEFAULT_GENERATED_PER_GENERATOR == 800
    assert DEFAULT_REAL_TOTAL == 5_600


def test_prepare_script_direct_help_entrypoint() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/prepare_genimage_v2.py", "--help"],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--input-root" in completed.stdout
    assert "--confirm-license" in completed.stdout


def test_preparation_is_exact_balanced_deterministic_and_idempotent(tmp_path: Path) -> None:
    input_root, inventory_spec = _build_genimage_tree(tmp_path / "source")
    output_root = tmp_path / "project"
    first, first_combined, first_summary = _prepare(
        input_root, output_root, inventory_spec
    )
    second, second_combined, second_summary = _prepare(
        input_root, output_root, inventory_spec
    )

    assert first.to_csv(index=False) == second.to_csv(index=False)
    assert first_combined.to_csv(index=False) == second_combined.to_csv(index=False)
    assert first_summary["selected_digest"] == second_summary["selected_digest"]
    assert len(first) == 140
    assert first.groupby("label").size().to_dict() == {0: 70, 1: 70}
    assert first.groupby(["split", "label"]).size().to_dict() == {
        ("test", 0): 7,
        ("test", 1): 7,
        ("train", 0): 56,
        ("train", 1): 56,
        ("val", 0): 7,
        ("val", 1): 7,
    }
    assert first.loc[first["label"] == 0].groupby("source_class").size().to_dict() == {
        str(value): 7 for value in range(10)
    }
    assert first_summary["generator_counts"] == {
        **{name: 10 for name in GENERATOR_DIRECTORIES},
        NATURE_DIRECTORY: 70,
    }
    assert first_summary["split_counts"] == {
        "train": {"0": 56, "1": 56},
        "val": {"0": 7, "1": 7},
        "test": {"0": 7, "1": 7},
    }
    assert first_summary["combined_counts"] == {
        "total": 156,
        "train": 128,
        "val": 14,
        "test": 14,
        "sid_train_rows": 16,
        "genimage_rows": 140,
    }
    assert set(first_combined.loc[first_combined["dataset"] == "saberzl/SID_Set", "split"]) == {
        "train"
    }
    assert all(value.startswith("unbiased-tiny-genimage:v1:") for value in first["source_id"])
    assert all(
        value.startswith("sid-set:")
        for value in first_combined.loc[
            first_combined["dataset"] == "saberzl/SID_Set", "source_id"
        ]
    )
    assert not first_combined["sha256"].duplicated().any()

    assert (output_root / GENIMAGE_MANIFEST_RELATIVE).is_file()
    assert (output_root / TRAINING_MANIFEST_RELATIVE).is_file()
    summary_path = output_root / GENIMAGE_SUMMARY_RELATIVE
    assert summary_path.is_file()
    assert json.loads(summary_path.read_text(encoding="utf-8"))["wildfake_used"] is False
    first_image = output_root / Path(first.iloc[0]["path"])
    assert IMAGE_ROOT_RELATIVE.as_posix() in first.iloc[0]["path"]
    with Image.open(first_image) as opened:
        assert opened.format == "JPEG"
        assert opened.mode == "RGB"


def test_same_label_sid_duplicate_is_removed_and_quota_refilled(tmp_path: Path) -> None:
    input_root, _ = _build_genimage_tree(tmp_path / "source")
    dataset_root = input_root / "dataset"
    duplicate_path = sorted((dataset_root / NATURE_DIRECTORY).iterdir())[0]
    _image(10_000).save(duplicate_path, format="PNG")
    inventory_spec = _inventory_spec(dataset_root)

    frame, _, summary = _prepare(input_root, tmp_path / "project", inventory_spec)

    assert len(frame) == 140
    assert summary["deduplication"]["duplicate_of_sid"] == 1
    assert int((frame["generator"] == NATURE_DIRECTORY).sum()) == 70


def test_cross_dataset_conflicting_label_duplicate_is_rejected(tmp_path: Path) -> None:
    input_root, _ = _build_genimage_tree(tmp_path / "source")
    dataset_root = input_root / "dataset"
    generated_path = sorted((dataset_root / GENERATOR_DIRECTORIES[0]).iterdir())[0]
    _image(10_000).save(generated_path, format="PNG")
    inventory_spec = _inventory_spec(dataset_root)
    output_root = tmp_path / "project"
    sid_manifest = _build_sid_manifest(output_root)

    with pytest.raises(ValueError, match="conflicting labels"):
        prepare_genimage_v2(
            input_root=input_root,
            sid_manifest=sid_manifest,
            output_root=output_root,
            license_confirmed=True,
            inventory_spec=inventory_spec,
            selection_spec=SMALL_SELECTION_SPEC,
            sid_spec=SMALL_SID_SPEC,
        )


def test_corrupt_source_image_is_rejected_before_outputs(tmp_path: Path) -> None:
    input_root, _ = _build_genimage_tree(tmp_path / "source")
    dataset_root = input_root / "dataset"
    corrupt = sorted((dataset_root / GENERATOR_DIRECTORIES[0]).iterdir())[0]
    corrupt.write_bytes(b"not an image")
    inventory_spec = _inventory_spec(dataset_root)
    output_root = tmp_path / "project"
    sid_manifest = _build_sid_manifest(output_root)

    with pytest.raises(RuntimeError, match="could not be decoded"):
        prepare_genimage_v2(
            input_root=input_root,
            sid_manifest=sid_manifest,
            output_root=output_root,
            license_confirmed=True,
            inventory_spec=inventory_spec,
            selection_spec=SMALL_SELECTION_SPEC,
            sid_spec=SMALL_SID_SPEC,
        )
    assert not (output_root / GENIMAGE_MANIFEST_RELATIVE).exists()


def test_wrong_metadata_hash_and_unexpected_inventory_are_rejected(tmp_path: Path) -> None:
    input_root, inventory_spec = _build_genimage_tree(tmp_path / "source")
    wrong_hash_spec = InventorySpec(
        generator_images=inventory_spec.generator_images,
        nature_images=inventory_spec.nature_images,
        file_count=inventory_spec.file_count,
        total_bytes=inventory_spec.total_bytes,
        metadata_sha256="0" * 64,
    )
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        validate_inventory(input_root, spec=wrong_hash_spec)

    extra = input_root / "dataset" / GENERATOR_DIRECTORIES[0] / "extra.jpg"
    _save_source(extra, 999_999)
    with pytest.raises(RuntimeError, match="generator image counts"):
        validate_inventory(input_root, spec=inventory_spec)


def test_symlinked_input_is_rejected(tmp_path: Path) -> None:
    input_root, inventory_spec = _build_genimage_tree(tmp_path / "source")
    dataset_root = input_root / "dataset"
    source = sorted((dataset_root / GENERATOR_DIRECTORIES[0]).iterdir())[0]
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(source.read_bytes())
    source.unlink()
    try:
        source.symlink_to(outside)
    except OSError:
        pytest.skip("symlinks are unavailable in this environment")

    with pytest.raises(RuntimeError, match="Symlinks"):
        validate_inventory(input_root, spec=inventory_spec)


def test_license_confirmation_is_required_before_reading_inputs(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Licence confirmation"):
        prepare_genimage_v2(
            input_root=tmp_path / "missing",
            sid_manifest=tmp_path / "missing.csv",
            output_root=tmp_path,
            license_confirmed=False,
        )
