"""Stream a small, balanced SID_Set subset and build a safe manifest.

Only labels 0 (real) and 1 (fully synthetic) are retained. Images are decoded
and re-encoded identically as high-quality RGB JPEGs so file-format shortcuts
do not distinguish the two classes. Duplicates in the normalized training
representation are removed before source-grouped, class-balanced split assignment.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, ImageOps


DEFAULT_DATASET = "saberzl/SID_Set"
DEFAULT_TOTAL = 6_000
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
)
SPLIT_ORDER = ("train", "val", "test")
SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}
LABEL_DIRECTORIES = {0: "authentic", 1: "generated"}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stream a balanced real/synthetic subset of SID_Set without "
            "downloading the full 140 GB dataset."
        )
    )
    parser.add_argument(
        "--total",
        type=int,
        default=DEFAULT_TOTAL,
        help="Even number of images to retain across labels 0 and 1.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed.")
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help="Hugging Face dataset id (default: saberzl/SID_Set).",
    )
    parser.add_argument(
        "--source-split",
        default="train",
        help="Upstream split to stream; the default keeps SID validation unused.",
    )
    parser.add_argument(
        "--shuffle-buffer",
        type=int,
        default=512,
        help="Streaming shuffle buffer; this controls RAM, not download size.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Repository root receiving data/ (default: this repository).",
    )
    return parser.parse_args(argv)


def _validate_options(total: int, seed: int, shuffle_buffer: int = 1) -> None:
    if (
        isinstance(total, bool)
        or not isinstance(total, int)
        or total < 20
        or total % 2
    ):
        raise ValueError("total must be an even integer of at least 20")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if (
        isinstance(shuffle_buffer, bool)
        or not isinstance(shuffle_buffer, int)
        or shuffle_buffer <= 0
    ):
        raise ValueError("shuffle_buffer must be greater than zero")


def _open_example_image(value: Any) -> Image.Image:
    """Decode a Hugging Face image value into an owned RGB image."""

    if isinstance(value, Image.Image):
        value.load()
        return ImageOps.exif_transpose(value).convert("RGB")

    if isinstance(value, Mapping):
        raw_bytes = value.get("bytes")
        raw_path = value.get("path")
        if raw_bytes is not None:
            source: Any = io.BytesIO(raw_bytes)
        elif raw_path:
            source = raw_path
        else:
            raise ValueError("image mapping contains neither bytes nor path")
    elif isinstance(value, (str, Path)):
        source = value
    else:
        raise TypeError("image must be a PIL image, path, or bytes/path mapping")

    with Image.open(source) as opened:
        opened.load()
        return ImageOps.exif_transpose(opened).convert("RGB")


def _normalized_jpeg(image: Image.Image) -> tuple[bytes, str]:
    """Encode the exact training representation and return bytes plus SHA-256."""

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95, subsampling=0)
    payload = buffer.getvalue()
    return payload, hashlib.sha256(payload).hexdigest()


def _class_targets(count: int) -> dict[str, int]:
    train = int(count * SPLIT_RATIOS["train"])
    val = int(count * SPLIT_RATIOS["val"])
    return {"train": train, "val": val, "test": count - train - val}


def _seeded_group_key(source_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{source_id}".encode("utf-8")).hexdigest()


def _assign_grouped_stratified_splits(
    records: list[dict[str, Any]], seed: int
) -> None:
    """Assign each source group to one split while balancing both labels."""

    groups: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        groups[str(record["source_id"])].append(index)

    label_totals = Counter(int(record["label"]) for record in records)
    targets = {
        split: {label: _class_targets(label_totals[label])[split] for label in (0, 1)}
        for split in SPLIT_ORDER
    }
    current = {split: {0: 0, 1: 0} for split in SPLIT_ORDER}
    ordered_groups = sorted(
        groups.items(),
        key=lambda item: (-len(item[1]), _seeded_group_key(item[0], seed)),
    )

    for _source_id, indices in ordered_groups:
        group_counts = Counter(int(records[index]["label"]) for index in indices)

        def candidate_score(split: str) -> tuple[float, float, int]:
            overflow = sum(
                max(
                    current[split][label]
                    + group_counts[label]
                    - targets[split][label],
                    0,
                )
                for label in (0, 1)
            )
            remaining_fraction = sum(
                group_counts[label]
                * (targets[split][label] - current[split][label])
                / max(targets[split][label], 1)
                for label in (0, 1)
            )
            return (float(overflow), -remaining_fraction, SPLIT_ORDER.index(split))

        selected_split = min(SPLIT_ORDER, key=candidate_score)
        for index in indices:
            records[index]["split"] = selected_split
        for label in (0, 1):
            current[selected_split][label] += group_counts[label]


def _validate_manifest(frame: pd.DataFrame, expected_total: int) -> None:
    if len(frame) != expected_total:
        raise RuntimeError(
            f"prepared {len(frame)} images but expected exactly {expected_total}"
        )
    if set(frame["label"].unique()) != {0, 1}:
        raise RuntimeError("manifest must contain only labels 0 and 1")
    if frame.groupby("label").size().to_dict() != {
        0: expected_total // 2,
        1: expected_total // 2,
    }:
        raise RuntimeError("manifest is not balanced between labels 0 and 1")
    if set(frame["split"].unique()) != set(SPLIT_ORDER):
        raise RuntimeError("manifest must contain train, val, and test splits")
    split_label_counts = frame.groupby(["split", "label"]).size()
    expected_pairs = {
        (split, label) for split in SPLIT_ORDER for label in (0, 1)
    }
    if not expected_pairs.issubset(set(split_label_counts.index)):
        raise RuntimeError("every split must contain both labels")
    if frame["sha256"].duplicated().any():
        raise RuntimeError("duplicate normalized images remain in the manifest")
    if frame.groupby("source_id")["split"].nunique().max() != 1:
        raise RuntimeError("a source_id appears in more than one split")
    if frame.groupby("sha256")["split"].nunique().max() != 1:
        raise RuntimeError("an image hash appears in more than one split")


def prepare_examples(
    examples: Iterable[Mapping[str, Any]],
    *,
    total: int = DEFAULT_TOTAL,
    seed: int = 42,
    output_root: str | Path,
    dataset_name: str = DEFAULT_DATASET,
    source_split: str = "train",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Materialize a balanced subset from a stream of SID-like examples."""

    _validate_options(total, seed)
    root = Path(output_root).resolve()
    raw_root = root / "data" / "raw"
    processed_root = root / "data" / "processed"
    for directory in LABEL_DIRECTORIES.values():
        (raw_root / directory).mkdir(parents=True, exist_ok=True)
    processed_root.mkdir(parents=True, exist_ok=True)

    quota = total // 2
    selected = Counter({0: 0, 1: 0})
    skipped = Counter()
    seen_hash_labels: dict[str, int] = {}
    records: list[dict[str, Any]] = []

    for example in examples:
        raw_label = example.get("label")
        if isinstance(raw_label, bool):
            skipped["invalid_label"] += 1
            continue
        try:
            numeric_label = float(raw_label)
            if not math.isfinite(numeric_label) or not numeric_label.is_integer():
                raise ValueError
            label = int(numeric_label)
        except (TypeError, ValueError, OverflowError):
            skipped["invalid_label"] += 1
            continue
        if label not in LABEL_DIRECTORIES:
            skipped["excluded_label"] += 1
            continue
        if selected[label] >= quota:
            skipped["class_quota_full"] += 1
            if all(selected[value] >= quota for value in LABEL_DIRECTORIES):
                break
            continue

        try:
            image = _open_example_image(example.get("image"))
            normalized_bytes, digest = _normalized_jpeg(image)
        except (OSError, TypeError, ValueError) as error:
            skipped[f"unreadable_{type(error).__name__}"] += 1
            continue

        previous_label = seen_hash_labels.get(digest)
        if previous_label is not None:
            if previous_label != label:
                raise ValueError(
                    "the same normalized image was observed with conflicting labels "
                    f"{previous_label} and {label} (sha256={digest})"
                )
            skipped["duplicate_normalized_image"] += 1
            continue

        source_value = example.get("img_id") or example.get("source_id")
        source_id = str(source_value).strip() if source_value is not None else ""
        if not source_id:
            source_id = digest

        destination = raw_root / LABEL_DIRECTORIES[label] / f"{digest}.jpg"
        destination.write_bytes(normalized_bytes)
        relative_path = destination.relative_to(root).as_posix()
        seen_hash_labels[digest] = label
        selected[label] += 1
        records.append(
            {
                "path": relative_path,
                "label": label,
                "split": "",
                "dataset": dataset_name,
                "source_split": source_split,
                "source_id": source_id,
                "width": image.width,
                "height": image.height,
                "sha256": digest,
            }
        )
        if len(records) % 250 == 0 or len(records) == total:
            print(
                f"Selected {len(records)}/{total} images "
                f"(real={selected[0]}, synthetic={selected[1]})."
            )
        if all(selected[value] >= quota for value in LABEL_DIRECTORIES):
            break

    if selected != Counter({0: quota, 1: quota}):
        raise RuntimeError(
            "The source stream ended before a balanced subset could be built: "
            f"real={selected[0]}/{quota}, synthetic={selected[1]}/{quota}."
        )

    _assign_grouped_stratified_splits(records, seed)
    frame = pd.DataFrame(records, columns=MANIFEST_COLUMNS)
    split_rank = {name: index for index, name in enumerate(SPLIT_ORDER)}
    frame["_split_rank"] = frame["split"].map(split_rank)
    frame = frame.sort_values(
        ["_split_rank", "label", "source_id", "sha256"], kind="stable"
    ).drop(columns="_split_rank")
    frame = frame.reset_index(drop=True)
    _validate_manifest(frame, total)

    manifest_path = processed_root / "manifest.csv"
    temporary_manifest = manifest_path.with_suffix(".csv.tmp")
    frame.to_csv(temporary_manifest, index=False)
    temporary_manifest.replace(manifest_path)

    split_counts = {
        split: {
            str(label): int(
                ((frame["split"] == split) & (frame["label"] == label)).sum()
            )
            for label in (0, 1)
        }
        for split in SPLIT_ORDER
    }
    summary: dict[str, Any] = {
        "schema_version": 1,
        "dataset": dataset_name,
        "dataset_url": f"https://huggingface.co/datasets/{dataset_name}",
        "source_split": source_split,
        "license": "CC BY 4.0 (per the SID_Set dataset card)",
        "seed": seed,
        "total": total,
        "class_counts": {"0_real": selected[0], "1_full_synthetic": selected[1]},
        "split_counts": split_counts,
        "skipped": dict(sorted(skipped.items())),
        "image_normalization": "RGB JPEG, quality 95, 4:4:4 subsampling",
        "sha256_definition": "SHA-256 of the stored normalized JPEG bytes",
    }
    summary_path = processed_root / "manifest_summary.json"
    temporary_summary = summary_path.with_suffix(".json.tmp")
    temporary_summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary_summary.replace(summary_path)
    print(f"Manifest written to {manifest_path}.")
    print(json.dumps(summary["split_counts"], indent=2, sort_keys=True))
    return frame, summary


def stream_sid_examples(
    *,
    dataset_name: str,
    source_split: str,
    seed: int,
    shuffle_buffer: int,
) -> Iterable[Mapping[str, Any]]:
    """Return a shuffled Hugging Face stream projected to required columns."""

    try:
        from datasets import Image as HuggingFaceImage
        from datasets import load_dataset
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Hugging Face datasets is required; install requirements-train.txt"
        ) from error

    dataset = load_dataset(
        dataset_name,
        split=source_split,
        streaming=True,
    )
    required = {"image", "label"}
    available = set(dataset.column_names or [])
    missing = required - available
    if missing:
        raise RuntimeError(
            f"Dataset is missing required column(s): {', '.join(sorted(missing))}"
        )
    removable = [
        name
        for name in available
        if name not in {"image", "label", "img_id", "source_id"}
    ]
    if removable:
        dataset = dataset.remove_columns(removable)
    dataset = dataset.cast_column("image", HuggingFaceImage(decode=False))
    return dataset.shuffle(seed=seed, buffer_size=shuffle_buffer)


def process_and_build_manifest(
    *,
    total: int = DEFAULT_TOTAL,
    seed: int = 42,
    dataset_name: str = DEFAULT_DATASET,
    source_split: str = "train",
    shuffle_buffer: int = 512,
    output_root: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Stream SID_Set and write the repository's training manifest."""

    _validate_options(total, seed, shuffle_buffer)
    root = (
        Path(output_root)
        if output_root is not None
        else Path(__file__).resolve().parent.parent
    )
    examples = stream_sid_examples(
        dataset_name=dataset_name,
        source_split=source_split,
        seed=seed,
        shuffle_buffer=shuffle_buffer,
    )
    return prepare_examples(
        examples,
        total=total,
        seed=seed,
        output_root=root,
        dataset_name=dataset_name,
        source_split=source_split,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    process_and_build_manifest(
        total=args.total,
        seed=args.seed,
        dataset_name=args.dataset,
        source_split=args.source_split,
        shuffle_buffer=args.shuffle_buffer,
        output_root=args.output_root,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    raise SystemExit(main())
