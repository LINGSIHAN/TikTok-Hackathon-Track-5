"""Prepare the pinned Unbiased Tiny GenImage subset for v2 training.

The Kaggle input is treated as immutable.  Every source image is decoded and
normalized with the same RGB/JPEG contract used by SID_Set before selection,
deduplication, and split assignment.  WildFake is intentionally not accepted
as an input to this workflow.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT_TEXT = str(REPOSITORY_ROOT)
if REPOSITORY_ROOT_TEXT not in sys.path:
    sys.path.insert(0, REPOSITORY_ROOT_TEXT)

import pandas as pd
from PIL import Image

from src.data.preprocessing import normalize_pil_image


DATASET_ID = "cartografia/unbiased-tiny-genimage"
DATASET_VERSION = 1
DATASET_URL = "https://www.kaggle.com/datasets/cartografia/unbiased-tiny-genimage"
DATASET_LICENSE = "CC BY-NC-SA 4.0"
LICENSE_CONFIRMATION_DATE = "2026-08-31"
GENERATOR_DIRECTORIES = (
    "ADM",
    "BigGAN",
    "Midjourney",
    "VQDM",
    "glide",
    "stable_diffusion_v_1_5",
    "wukong",
)
NATURE_DIRECTORY = "Nature"
METADATA_FILENAME = "nature_metadata.csv"
METADATA_SHA256 = (
    "5f9a46e53e624339f6db8cc4d4a4fe5e54a0371e4b07a7da278300f6ed699e91"
)
EXPECTED_FILE_COUNT = 23_329
EXPECTED_TOTAL_BYTES = 2_528_629_592
EXPECTED_GENERATOR_IMAGES = 2_500
EXPECTED_NATURE_IMAGES = 5_828
DEFAULT_GENERATED_PER_GENERATOR = 800
DEFAULT_REAL_TOTAL = 5_600
DEFAULT_SEED = 42
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"})
SPLIT_ORDER = ("train", "val", "test")
SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}
MANIFEST_COLUMNS = (
    "path",
    "label",
    "split",
    "dataset",
    "source_split",
    "source_id",
    "width",
    "height",
    "sha256",
    "generator",
    "source_class",
    "source_path",
)
GENIMAGE_MANIFEST_RELATIVE = Path("data/processed/genimage_v2_manifest.csv")
GENIMAGE_SUMMARY_RELATIVE = Path("data/processed/genimage_v2_manifest_summary.json")
TRAINING_MANIFEST_RELATIVE = Path("data/processed/train_v2_manifest.csv")
IMAGE_ROOT_RELATIVE = Path("data/external/genimage_v2")


@dataclass(frozen=True)
class InventorySpec:
    """Immutable source-inventory expectations.

    Tests use a smaller spec; the CLI always uses ``DEFAULT_INVENTORY_SPEC``.
    """

    generator_images: int
    nature_images: int
    file_count: int
    total_bytes: int
    metadata_sha256: str


@dataclass(frozen=True)
class SelectionSpec:
    generated_per_generator: int
    real_total: int


@dataclass(frozen=True)
class SIDManifestSpec:
    total: int
    split_label_counts: Mapping[tuple[str, int], int]
    require_summary: bool = True


DEFAULT_INVENTORY_SPEC = InventorySpec(
    generator_images=EXPECTED_GENERATOR_IMAGES,
    nature_images=EXPECTED_NATURE_IMAGES,
    file_count=EXPECTED_FILE_COUNT,
    total_bytes=EXPECTED_TOTAL_BYTES,
    metadata_sha256=METADATA_SHA256,
)
DEFAULT_SELECTION_SPEC = SelectionSpec(
    generated_per_generator=DEFAULT_GENERATED_PER_GENERATOR,
    real_total=DEFAULT_REAL_TOTAL,
)
DEFAULT_SID_SPEC = SIDManifestSpec(
    total=6_000,
    split_label_counts={
        ("train", 0): 2_400,
        ("train", 1): 2_400,
        ("val", 0): 300,
        ("val", 1): 300,
        ("test", 0): 300,
        ("test", 1): 300,
    },
)


def file_sha256(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _seeded_key(seed: int, namespace: str, value: str) -> str:
    payload = f"{seed}:{namespace}:{value}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _split_targets(count: int) -> dict[str, int]:
    train = int(count * SPLIT_RATIOS["train"])
    validation = int(count * SPLIT_RATIOS["val"])
    return {"train": train, "val": validation, "test": count - train - validation}


def _validate_specs(
    inventory_spec: InventorySpec,
    selection_spec: SelectionSpec,
    seed: int,
) -> None:
    integer_values = {
        "generator_images": inventory_spec.generator_images,
        "nature_images": inventory_spec.nature_images,
        "file_count": inventory_spec.file_count,
        "total_bytes": inventory_spec.total_bytes,
        "generated_per_generator": selection_spec.generated_per_generator,
        "real_total": selection_spec.real_total,
        "seed": seed,
    }
    for name, value in integer_values.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if selection_spec.generated_per_generator <= 0 or selection_spec.real_total <= 0:
        raise ValueError("selection quotas must be positive")
    if selection_spec.generated_per_generator > inventory_spec.generator_images:
        raise ValueError("generated selection quota exceeds each generator inventory")
    if selection_spec.real_total > inventory_spec.nature_images:
        raise ValueError("real selection quota exceeds the Nature inventory")
    digest = inventory_spec.metadata_sha256
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError("metadata_sha256 must be a lowercase SHA-256 digest")


def _is_dataset_root(path: Path) -> bool:
    return (
        (path / METADATA_FILENAME).is_file()
        and (path / NATURE_DIRECTORY).is_dir()
        and all((path / name).is_dir() for name in GENERATOR_DIRECTORIES)
    )


def locate_dataset_root(input_root: str | Path) -> Path:
    """Locate the single dataset root below a Kaggle input mount."""

    supplied = Path(input_root)
    if not supplied.exists():
        raise FileNotFoundError(
            f"GenImage input was not found: {supplied}. In Kaggle, attach "
            f"{DATASET_ID} version {DATASET_VERSION} with Add Input."
        )
    supplied = supplied.resolve()
    candidates: list[Path] = []
    if _is_dataset_root(supplied):
        candidates.append(supplied)
    for metadata_path in supplied.rglob(METADATA_FILENAME):
        candidate = metadata_path.parent.resolve()
        if candidate != supplied and _is_dataset_root(candidate):
            candidates.append(candidate)
    candidates = sorted(set(candidates), key=lambda item: item.as_posix())
    if not candidates:
        raise RuntimeError(
            f"Could not find {METADATA_FILENAME}, {NATURE_DIRECTORY}, and all "
            "seven generator directories beneath the input root"
        )
    if len(candidates) != 1:
        raise RuntimeError("Multiple possible GenImage dataset roots were found")
    try:
        candidates[0].relative_to(supplied)
    except ValueError as error:
        raise RuntimeError("The discovered dataset root escapes the input root") from error
    return candidates[0]


def _safe_files(directory: Path, dataset_root: Path) -> list[Path]:
    if directory.is_symlink():
        raise RuntimeError(f"Symlinked source directories are not allowed: {directory.name}")
    files: list[Path] = []
    for entry in directory.rglob("*"):
        if entry.is_symlink():
            raise RuntimeError(
                "Symlinks are not allowed in the source dataset: "
                + entry.relative_to(dataset_root).as_posix()
            )
        if entry.is_dir():
            raise RuntimeError(
                "Unexpected nested source directory: "
                + entry.relative_to(dataset_root).as_posix()
            )
        if entry.is_file():
            resolved = entry.resolve()
            try:
                resolved.relative_to(dataset_root)
            except ValueError as error:
                raise RuntimeError("A source file escapes the dataset root") from error
            if resolved.suffix.casefold() not in IMAGE_SUFFIXES:
                raise RuntimeError(
                    "Unexpected non-image source file: "
                    + resolved.relative_to(dataset_root).as_posix()
                )
            files.append(resolved)
    return sorted(files, key=lambda item: item.relative_to(dataset_root).as_posix())


def validate_inventory(
    input_root: str | Path,
    *,
    spec: InventorySpec = DEFAULT_INVENTORY_SPEC,
) -> tuple[Path, dict[str, list[Path]], dict[str, Any]]:
    """Require the exact pinned Kaggle version-1 inventory."""

    root = locate_dataset_root(input_root)
    metadata_path = root / METADATA_FILENAME
    if metadata_path.is_symlink():
        raise RuntimeError("The metadata file must not be a symlink")

    top_level_expected = {
        METADATA_FILENAME,
        NATURE_DIRECTORY,
        *GENERATOR_DIRECTORIES,
    }
    top_level_actual = {entry.name for entry in root.iterdir()}
    if top_level_actual != top_level_expected:
        missing = sorted(top_level_expected - top_level_actual)
        unexpected = sorted(top_level_actual - top_level_expected)
        raise RuntimeError(
            f"Unexpected dataset top-level inventory; missing={missing}, "
            f"unexpected={unexpected}"
        )

    sources = {
        name: _safe_files(root / name, root)
        for name in (*GENERATOR_DIRECTORIES, NATURE_DIRECTORY)
    }
    generator_counts = {name: len(sources[name]) for name in GENERATOR_DIRECTORIES}
    wrong_generators = {
        name: count
        for name, count in generator_counts.items()
        if count != spec.generator_images
    }
    if wrong_generators:
        raise RuntimeError(
            f"Unexpected generator image counts: {wrong_generators}; expected "
            f"{spec.generator_images} in every generator directory"
        )
    if len(sources[NATURE_DIRECTORY]) != spec.nature_images:
        raise RuntimeError(
            f"Unexpected Nature image count: {len(sources[NATURE_DIRECTORY])}; "
            f"expected {spec.nature_images}"
        )

    all_files = [metadata_path, *[path for paths in sources.values() for path in paths]]
    file_count = len(all_files)
    total_bytes = sum(path.stat().st_size for path in all_files)
    if file_count != spec.file_count:
        raise RuntimeError(
            f"Unexpected dataset file count: {file_count}; expected {spec.file_count}"
        )
    if total_bytes != spec.total_bytes:
        raise RuntimeError(
            f"Unexpected dataset byte count: {total_bytes}; expected {spec.total_bytes}"
        )
    metadata_digest = file_sha256(metadata_path)
    if metadata_digest != spec.metadata_sha256:
        raise RuntimeError(
            f"{METADATA_FILENAME} SHA-256 mismatch: {metadata_digest}; expected "
            f"{spec.metadata_sha256}"
        )

    inventory_lines = [
        f"{path.relative_to(root).as_posix()}\t{path.stat().st_size}\n"
        for path in sorted(all_files, key=lambda item: item.relative_to(root).as_posix())
    ]
    inventory_digest = hashlib.sha256("".join(inventory_lines).encode("utf-8")).hexdigest()
    inventory = {
        "file_count": file_count,
        "total_bytes": total_bytes,
        "metadata_file": METADATA_FILENAME,
        "metadata_sha256": metadata_digest,
        "inventory_digest": inventory_digest,
        "generator_image_counts": generator_counts,
        "nature_image_count": len(sources[NATURE_DIRECTORY]),
    }
    return root, sources, inventory


def read_nature_metadata(
    metadata_path: str | Path,
    nature_files: Sequence[Path],
) -> dict[str, dict[str, Any]]:
    """Validate metadata and return rows keyed by exact source filename."""

    expected_columns = {"filename", "class", "width", "height", "quality"}
    with Path(metadata_path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or set(reader.fieldnames) != expected_columns:
            raise RuntimeError(
                f"{METADATA_FILENAME} must have exactly these columns: "
                + ", ".join(sorted(expected_columns))
            )
        rows = list(reader)

    by_name: dict[str, dict[str, Any]] = {}
    for row_number, row in enumerate(rows, start=2):
        filename = str(row.get("filename", "")).strip()
        source_class = str(row.get("class", "")).strip()
        if not filename or Path(filename).name != filename:
            raise RuntimeError(f"Unsafe or empty metadata filename at row {row_number}")
        if filename in by_name:
            raise RuntimeError(f"Duplicate metadata filename: {filename}")
        if not source_class:
            raise RuntimeError(f"Empty metadata class at row {row_number}")
        try:
            width = int(str(row.get("width", "")))
            height = int(str(row.get("height", "")))
            quality = int(str(row.get("quality", "")))
        except ValueError as error:
            raise RuntimeError(f"Invalid numeric metadata at row {row_number}") from error
        if width <= 0 or height <= 0 or quality != 96:
            raise RuntimeError(f"Invalid dimensions or JPEG quality at row {row_number}")
        by_name[filename] = {
            "class": source_class,
            "width": width,
            "height": height,
            "quality": quality,
        }

    file_names = [path.name for path in nature_files]
    if len(file_names) != len(set(file_names)):
        raise RuntimeError("Nature filenames must be unique even if nested")
    if set(by_name) != set(file_names):
        missing = sorted(set(file_names) - set(by_name))[:5]
        unknown = sorted(set(by_name) - set(file_names))[:5]
        raise RuntimeError(
            f"Nature metadata does not exactly cover the images; "
            f"missing={missing}, unknown={unknown}"
        )
    return by_name


def _normalized_source(path: Path) -> tuple[bytes, str, int, int, tuple[int, int]]:
    try:
        with Image.open(path) as opened:
            opened.load()
            source_size = opened.size
            image = normalize_pil_image(opened)
    except (OSError, TypeError, ValueError) as error:
        raise RuntimeError(f"Source image could not be decoded: {path.name}") from error
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95, subsampling=0)
    payload = buffer.getvalue()
    return payload, hashlib.sha256(payload).hexdigest(), image.width, image.height, source_size


def _valid_sha256(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def read_sid_manifest(
    manifest_path: str | Path,
    *,
    output_root: str | Path,
    spec: SIDManifestSpec = DEFAULT_SID_SPEC,
) -> tuple[list[dict[str, Any]], dict[str, int], str]:
    """Validate the frozen SID manifest and all referenced normalized files."""

    path = Path(manifest_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"SID manifest not found: {path}. Recreate the pinned SID_Set subset first."
        )
    required = {
        "path",
        "label",
        "split",
        "dataset",
        "source_split",
        "source_id",
        "width",
        "height",
        "sha256",
    }
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise RuntimeError("SID manifest schema is incomplete")
        rows = list(reader)
    if len(rows) != spec.total:
        raise RuntimeError(f"SID manifest has {len(rows)} rows; expected {spec.total}")

    root = Path(output_root).resolve()
    counts: Counter[tuple[str, int]] = Counter()
    digest_labels: dict[str, int] = {}
    validated: list[dict[str, Any]] = []
    for row_number, raw in enumerate(rows, start=2):
        try:
            label = int(raw["label"])
            width = int(raw["width"])
            height = int(raw["height"])
        except (TypeError, ValueError) as error:
            raise RuntimeError(f"Invalid SID values at manifest row {row_number}") from error
        split = raw["split"].strip()
        digest = raw["sha256"].strip()
        if label not in (0, 1) or split not in SPLIT_ORDER or not _valid_sha256(digest):
            raise RuntimeError(f"Invalid SID label, split, or hash at row {row_number}")
        if raw["dataset"] != "saberzl/SID_Set" or raw["source_split"] != "train":
            raise RuntimeError(f"Unexpected SID provenance at manifest row {row_number}")
        if not raw["source_id"].strip():
            raise RuntimeError(f"Empty SID source_id at manifest row {row_number}")
        previous_label = digest_labels.setdefault(digest, label)
        if previous_label != label:
            raise RuntimeError("SID manifest contains a conflicting-label duplicate")
        if len(digest_labels) != len(validated) + 1:
            raise RuntimeError("SID manifest contains duplicate image hashes")

        relative = Path(raw["path"])
        if relative.is_absolute():
            raise RuntimeError("SID manifest image paths must be relative")
        image_path = (root / relative).resolve()
        try:
            image_path.relative_to(root)
        except ValueError as error:
            raise RuntimeError("A SID manifest path escapes the output root") from error
        if not image_path.is_file() or file_sha256(image_path) != digest:
            raise RuntimeError(f"SID image is missing or changed: {raw['path']}")
        try:
            with Image.open(image_path) as opened:
                opened.load()
                normalized = normalize_pil_image(opened)
                if normalized.size != (width, height):
                    raise RuntimeError(
                        f"SID dimensions disagree with the manifest: {raw['path']}"
                    )
        except OSError as error:
            raise RuntimeError(f"SID image is unreadable: {raw['path']}") from error

        counts[(split, label)] += 1
        validated.append(
            {
                **{column: raw.get(column, "") for column in MANIFEST_COLUMNS},
                "label": label,
                "width": width,
                "height": height,
                "source_id": "sid-set:" + raw["source_id"].strip(),
                "generator": "SID_Set",
                "source_class": "",
                "source_path": raw["path"],
            }
        )
    if dict(counts) != dict(spec.split_label_counts):
        raise RuntimeError(f"Unexpected SID split/class counts: {dict(counts)}")

    if spec.require_summary:
        summary_path = path.with_name("manifest_summary.json")
        if not summary_path.is_file():
            raise RuntimeError("SID manifest_summary.json is required for pinned provenance")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        expected = {
            "dataset": "saberzl/SID_Set",
            "dataset_revision": "dc03ead57929879319ce30a82bfcfb8d317b10bd",
            "source_split": "train",
            "seed": 42,
            "total": 6_000,
        }
        if any(summary.get(key) != value for key, value in expected.items()):
            raise RuntimeError("SID summary does not match the pinned 6,000-image subset")
    return validated, digest_labels, file_sha256(path)


def scan_genimage_candidates(
    dataset_root: Path,
    sources: Mapping[str, Sequence[Path]],
    nature_metadata: Mapping[str, Mapping[str, Any]],
    sid_digest_labels: Mapping[str, int],
) -> tuple[list[dict[str, Any]], Counter[str]]:
    """Decode all source images and remove normalized duplicates."""

    records: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    seen: dict[str, tuple[int, str]] = {}
    for generator in (*GENERATOR_DIRECTORIES, NATURE_DIRECTORY):
        label = 0 if generator == NATURE_DIRECTORY else 1
        for source_path in sources[generator]:
            _, digest, width, height, source_size = _normalized_source(source_path)
            relative = source_path.relative_to(dataset_root).as_posix()
            if generator == NATURE_DIRECTORY:
                metadata = nature_metadata[source_path.name]
                if source_size != (int(metadata["width"]), int(metadata["height"])):
                    raise RuntimeError(
                        f"Nature dimensions disagree with metadata: {relative}"
                    )
                source_class = str(metadata["class"])
            else:
                source_class = source_path.stem.split("_", 1)[0]

            sid_label = sid_digest_labels.get(digest)
            if sid_label is not None:
                if sid_label != label:
                    raise ValueError(
                        "GenImage and SID contain the same normalized image with "
                        f"conflicting labels (sha256={digest})"
                    )
                skipped["duplicate_of_sid"] += 1
                continue

            prior = seen.get(digest)
            if prior is not None:
                if prior[0] != label:
                    raise ValueError(
                        "GenImage contains the same normalized image with conflicting "
                        f"labels (sha256={digest}, paths={prior[1]!r} and {relative!r})"
                    )
                skipped["duplicate_within_genimage"] += 1
                continue
            seen[digest] = (label, relative)
            records.append(
                {
                    "source_file": source_path,
                    "label": label,
                    "generator": generator,
                    "source_class": source_class,
                    "source_path": relative,
                    "source_id": (
                        f"unbiased-tiny-genimage:v{DATASET_VERSION}:"
                        f"{generator}:{relative}"
                    ),
                    "width": width,
                    "height": height,
                    "sha256": digest,
                }
            )
    return records, skipped


def _balanced_real_order(records: Sequence[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record["source_class"])].append(record)
    for source_class, values in groups.items():
        values.sort(
            key=lambda item: _seeded_key(seed, f"real-class:{source_class}", item["source_id"])
        )
    class_order = sorted(
        groups,
        key=lambda value: _seeded_key(seed, "real-class-order", value),
    )
    ordered: list[dict[str, Any]] = []
    depth = 0
    while True:
        added = False
        for source_class in class_order:
            values = groups[source_class]
            if depth < len(values):
                ordered.append(values[depth])
                added = True
        if not added:
            return ordered
        depth += 1


def select_and_split(
    candidates: Sequence[dict[str, Any]],
    *,
    seed: int,
    spec: SelectionSpec = DEFAULT_SELECTION_SPEC,
) -> list[dict[str, Any]]:
    """Select exact class/generator quotas and assign fixed 80/10/10 splits."""

    selected: list[dict[str, Any]] = []
    by_generator: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in candidates:
        by_generator[str(record["generator"])].append(record)

    for generator in GENERATOR_DIRECTORIES:
        ordered = sorted(
            by_generator[generator],
            key=lambda item: _seeded_key(seed, f"select:{generator}", item["source_id"]),
        )
        if len(ordered) < spec.generated_per_generator:
            raise RuntimeError(
                f"Only {len(ordered)} unique {generator} images remain after "
                f"deduplication; need {spec.generated_per_generator}"
            )
        chosen = ordered[: spec.generated_per_generator]
        split_order = sorted(
            chosen,
            key=lambda item: _seeded_key(seed, f"split:{generator}", item["source_id"]),
        )
        targets = _split_targets(len(chosen))
        cursor = 0
        for split in SPLIT_ORDER:
            for record in split_order[cursor : cursor + targets[split]]:
                selected.append({**record, "split": split})
            cursor += targets[split]

    real_order = _balanced_real_order(by_generator[NATURE_DIRECTORY], seed)
    if len(real_order) < spec.real_total:
        raise RuntimeError(
            f"Only {len(real_order)} unique Nature images remain after deduplication; "
            f"need {spec.real_total}"
        )
    chosen_real = real_order[: spec.real_total]
    real_split_order = sorted(
        chosen_real,
        key=lambda item: _seeded_key(seed, "split:Nature", item["source_id"]),
    )
    targets = _split_targets(len(chosen_real))
    cursor = 0
    for split in SPLIT_ORDER:
        for record in real_split_order[cursor : cursor + targets[split]]:
            selected.append({**record, "split": split})
        cursor += targets[split]
    return selected


def _safe_destination(root: Path, relative: Path) -> Path:
    destination = (root / relative).resolve()
    try:
        destination.relative_to(root)
    except ValueError as error:
        raise RuntimeError("An output path escapes the output root") from error
    return destination


def materialize_selected(
    records: Sequence[dict[str, Any]], output_root: str | Path
) -> list[dict[str, Any]]:
    """Write normalized selected images atomically and return manifest rows."""

    root = Path(output_root).resolve()
    manifest_rows: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        payload, digest, width, height, _ = _normalized_source(record["source_file"])
        if digest != record["sha256"] or (width, height) != (
            record["width"],
            record["height"],
        ):
            raise RuntimeError("A source image changed while the subset was prepared")
        label_directory = "authentic" if record["label"] == 0 else "generated"
        relative = IMAGE_ROOT_RELATIVE / label_directory / f"{digest}.jpg"
        destination = _safe_destination(root, relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_file() and file_sha256(destination) == digest:
            try:
                with Image.open(destination) as opened:
                    opened.load()
                    if opened.format != "JPEG" or opened.mode != "RGB":
                        raise RuntimeError("Existing normalized output has the wrong format")
            except OSError as error:
                raise RuntimeError("Existing normalized output is unreadable") from error
        else:
            temporary = destination.with_name(f".{destination.name}.part")
            try:
                temporary.write_bytes(payload)
                if file_sha256(temporary) != digest:
                    raise RuntimeError("Normalized output hash verification failed")
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)
        manifest_rows.append(
            {
                "path": relative.as_posix(),
                "label": int(record["label"]),
                "split": record["split"],
                "dataset": DATASET_ID,
                "source_split": f"v{DATASET_VERSION}",
                "source_id": record["source_id"],
                "width": int(record["width"]),
                "height": int(record["height"]),
                "sha256": digest,
                "generator": record["generator"],
                "source_class": record["source_class"],
                "source_path": record["source_path"],
            }
        )
        if index % 500 == 0 or index == len(records):
            print(f"Normalized {index}/{len(records)} selected GenImage images.")
    return manifest_rows


def _sorted_frame(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    split_rank = {name: index for index, name in enumerate(SPLIT_ORDER)}
    frame["_split_rank"] = frame["split"].map(split_rank)
    frame = frame.sort_values(
        ["_split_rank", "label", "dataset", "generator", "source_id", "sha256"],
        kind="stable",
    ).drop(columns="_split_rank")
    return frame.reset_index(drop=True)


def _write_frame_atomic(frame: pd.DataFrame, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        frame.to_csv(temporary, index=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return file_sha256(path)


def _selected_digest(frame: pd.DataFrame) -> str:
    columns = ["sha256", "label", "split", "dataset", "source_id", "generator"]
    canonical = frame.sort_values(columns, kind="stable")[columns]
    lines = [
        "\t".join(map(str, row)) + "\n"
        for row in canonical.itertuples(index=False, name=None)
    ]
    return hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()


def _split_counts(frame: pd.DataFrame) -> dict[str, dict[str, int]]:
    return {
        split: {
            str(label): int(((frame["split"] == split) & (frame["label"] == label)).sum())
            for label in (0, 1)
        }
        for split in SPLIT_ORDER
    }


def prepare_genimage_v2(
    *,
    input_root: str | Path,
    sid_manifest: str | Path,
    output_root: str | Path,
    license_confirmed: bool,
    seed: int = DEFAULT_SEED,
    inventory_spec: InventorySpec = DEFAULT_INVENTORY_SPEC,
    selection_spec: SelectionSpec = DEFAULT_SELECTION_SPEC,
    sid_spec: SIDManifestSpec = DEFAULT_SID_SPEC,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Prepare GenImage, combine only SID train rows, and write audit artifacts."""

    if not license_confirmed:
        raise ValueError(
            "Licence confirmation is required. Review the GenImage/ImageNet-derived "
            "terms and rerun with --confirm-license."
        )
    _validate_specs(inventory_spec, selection_spec, seed)
    root = Path(output_root).resolve()
    dataset_root, sources, inventory = validate_inventory(input_root, spec=inventory_spec)
    nature_metadata = read_nature_metadata(
        dataset_root / METADATA_FILENAME,
        sources[NATURE_DIRECTORY],
    )
    sid_rows, sid_digest_labels, sid_manifest_sha256 = read_sid_manifest(
        sid_manifest,
        output_root=root,
        spec=sid_spec,
    )
    candidates, skipped = scan_genimage_candidates(
        dataset_root,
        sources,
        nature_metadata,
        sid_digest_labels,
    )
    selected = select_and_split(candidates, seed=seed, spec=selection_spec)
    rows = materialize_selected(selected, root)
    genimage_frame = _sorted_frame(rows)

    expected_total = (
        len(GENERATOR_DIRECTORIES) * selection_spec.generated_per_generator
        + selection_spec.real_total
    )
    if len(genimage_frame) != expected_total or genimage_frame["sha256"].duplicated().any():
        raise RuntimeError("Prepared GenImage manifest failed total/uniqueness validation")
    expected_class_counts = {
        0: selection_spec.real_total,
        1: len(GENERATOR_DIRECTORIES) * selection_spec.generated_per_generator,
    }
    if genimage_frame.groupby("label").size().to_dict() != expected_class_counts:
        raise RuntimeError("Prepared GenImage manifest is not class balanced")

    sid_train = [row for row in sid_rows if row["split"] == "train"]
    combined_frame = _sorted_frame([*rows, *sid_train])
    if combined_frame["sha256"].duplicated().any():
        raise RuntimeError("The combined v2 manifest contains duplicate image hashes")

    genimage_manifest_path = root / GENIMAGE_MANIFEST_RELATIVE
    training_manifest_path = root / TRAINING_MANIFEST_RELATIVE
    manifest_sha256 = _write_frame_atomic(genimage_frame, genimage_manifest_path)
    combined_manifest_sha256 = _write_frame_atomic(combined_frame, training_manifest_path)

    split_counts = _split_counts(genimage_frame)
    generator_counts = {
        name: int((genimage_frame["generator"] == name).sum())
        for name in (*GENERATOR_DIRECTORIES, NATURE_DIRECTORY)
    }
    generator_split_counts = {
        name: {
            split: int(
                ((genimage_frame["generator"] == name) & (genimage_frame["split"] == split)).sum()
            )
            for split in SPLIT_ORDER
        }
        for name in (*GENERATOR_DIRECTORIES, NATURE_DIRECTORY)
    }
    selected_digest = _selected_digest(genimage_frame)
    summary: dict[str, Any] = {
        "schema_version": 1,
        "dataset": DATASET_ID,
        "dataset_id": DATASET_ID,
        "dataset_url": DATASET_URL,
        "dataset_version": DATASET_VERSION,
        "license": DATASET_LICENSE,
        "license_confirmation": {
            "confirmed": True,
            "confirmed_on": LICENSE_CONFIRMATION_DATE,
            "scope": "hackathon non-commercial training",
        },
        "seed": seed,
        "total": len(genimage_frame),
        "class_counts": {
            "0_real": expected_class_counts[0],
            "1_generated": expected_class_counts[1],
        },
        "split_counts": split_counts,
        "generator_counts": generator_counts,
        "generator_split_counts": generator_split_counts,
        "real_source_class_count": len(
            set(
                genimage_frame.loc[
                    genimage_frame["label"] == 0, "source_class"
                ]
            )
        ),
        "inventory": inventory,
        "deduplication": dict(sorted(skipped.items())),
        "image_normalization": "RGB JPEG, quality 95, 4:4:4 subsampling",
        "sha256_definition": "SHA-256 of the stored normalized JPEG bytes",
        "selected_digest_definition": (
            "SHA-256 of sorted tab-separated sha256,label,split,dataset,source_id,generator rows"
        ),
        "selected_digest": selected_digest,
        "dataset_digest": selected_digest,
        "manifest_sha256": manifest_sha256,
        "combined_manifest_sha256": combined_manifest_sha256,
        "sid_manifest_sha256": sid_manifest_sha256,
        "combined_counts": {
            "total": len(combined_frame),
            "train": int((combined_frame["split"] == "train").sum()),
            "val": int((combined_frame["split"] == "val").sum()),
            "test": int((combined_frame["split"] == "test").sum()),
            "sid_train_rows": len(sid_train),
            "genimage_rows": len(genimage_frame),
        },
        "manifests": {
            "genimage": GENIMAGE_MANIFEST_RELATIVE.as_posix(),
            "training": TRAINING_MANIFEST_RELATIVE.as_posix(),
        },
        "pillow_version": package_version("Pillow"),
        "wildfake_used": False,
    }
    summary_path = root / GENIMAGE_SUMMARY_RELATIVE
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_summary = summary_path.with_name(f".{summary_path.name}.tmp")
    try:
        temporary_summary.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary_summary.replace(summary_path)
    finally:
        temporary_summary.unlink(missing_ok=True)
    print(f"GenImage manifest: {genimage_manifest_path}")
    print(f"Combined training manifest: {training_manifest_path}")
    print(f"Selected digest: {summary['selected_digest']}")
    return genimage_frame, combined_frame, summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate Kaggle Unbiased Tiny GenImage v1 and prepare the fixed "
            "11,200-image v2 subset."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("/kaggle/input/unbiased-tiny-genimage"),
        help="Attached Kaggle dataset directory.",
    )
    parser.add_argument(
        "--sid-manifest",
        type=Path,
        default=None,
        help="Pinned SID manifest (default: OUTPUT_ROOT/data/processed/manifest.csv).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPOSITORY_ROOT,
        help="Repository root receiving normalized images and manifests.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--confirm-license",
        action="store_true",
        help="Confirm that the GenImage/ImageNet-derived licence permits this run.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_root = args.output_root.resolve()
    sid_manifest = (
        args.sid_manifest
        if args.sid_manifest is not None
        else output_root / "data/processed/manifest.csv"
    )
    prepare_genimage_v2(
        input_root=args.input_root,
        sid_manifest=sid_manifest,
        output_root=output_root,
        license_confirmed=args.confirm_license,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    raise SystemExit(main())
