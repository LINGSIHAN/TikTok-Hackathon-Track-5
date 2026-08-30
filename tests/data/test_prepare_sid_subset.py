from pathlib import Path

import pytest
from PIL import Image

from scripts.prepare_sid_subset import prepare_examples


def _balanced_examples(group_count: int = 30):
    examples = [{"image": Image.new("RGB", (8, 8), "white"), "label": 2}]
    for group in range(group_count):
        source_id = f"source-{group:03d}"
        examples.extend(
            [
                {
                    "image": Image.new(
                        "RGB", (8, 8), ((group * 7) % 256, 30, 60)
                    ),
                    "label": 0,
                    "img_id": source_id,
                },
                {
                    "image": Image.new(
                        "RGB", (8, 8), (200, (group * 5) % 256, 150)
                    ),
                    "label": 1,
                    "img_id": source_id,
                },
            ]
        )
    return examples


def _source_split_map(frame):
    return frame.groupby("source_id")["split"].first().to_dict()


def test_prepare_examples_is_balanced_grouped_and_deterministic(tmp_path):
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first, summary = prepare_examples(
        _balanced_examples(), total=40, seed=42, output_root=first_root
    )
    second, _ = prepare_examples(
        _balanced_examples(), total=40, seed=42, output_root=second_root
    )

    assert first.to_csv(index=False) == second.to_csv(index=False)
    assert first.groupby("label").size().to_dict() == {0: 20, 1: 20}
    assert first.groupby(["split", "label"]).size().to_dict() == {
        ("test", 0): 2,
        ("test", 1): 2,
        ("train", 0): 16,
        ("train", 1): 16,
        ("val", 0): 2,
        ("val", 1): 2,
    }
    assert first.groupby("source_id")["split"].nunique().max() == 1
    assert not first["sha256"].duplicated().any()
    assert summary["class_counts"] == {
        "0_real": 20,
        "1_full_synthetic": 20,
    }
    assert summary["skipped"]["excluded_label"] == 1
    assert (first_root / "data/processed/manifest.csv").is_file()
    assert (first_root / "data/processed/manifest_summary.json").is_file()
    assert all((first_root / Path(value)).is_file() for value in first["path"])


def test_seed_changes_group_assignment_without_changing_content(tmp_path):
    first, _ = prepare_examples(
        _balanced_examples(), total=40, seed=1, output_root=tmp_path / "one"
    )
    second, _ = prepare_examples(
        _balanced_examples(), total=40, seed=2, output_root=tmp_path / "two"
    )

    assert set(first["sha256"]) == set(second["sha256"])
    assert _source_split_map(first) != _source_split_map(second)


def test_images_that_collapse_to_same_normalized_jpeg_are_deduplicated(tmp_path):
    # At JPEG quality 95 these two flat colors encode to identical bytes.
    near_duplicates = [
        {
            "image": Image.new("RGB", (8, 8), (0, 0, 0)),
            "label": 0,
            "img_id": "near-a",
        },
        {
            "image": Image.new("RGB", (8, 8), (1, 0, 0)),
            "label": 0,
            "img_id": "near-b",
        },
    ]

    frame, summary = prepare_examples(
        near_duplicates + _balanced_examples(),
        total=20,
        seed=42,
        output_root=tmp_path,
    )

    assert len(frame) == 20
    assert not frame["sha256"].duplicated().any()
    assert summary["skipped"]["duplicate_normalized_image"] == 1


def test_conflicting_exact_duplicate_labels_are_rejected(tmp_path):
    pixels = Image.new("RGB", (8, 8), "purple")
    examples = [
        {"image": pixels.copy(), "label": 0, "img_id": "real"},
        {"image": pixels.copy(), "label": 1, "img_id": "synthetic"},
    ]

    with pytest.raises(ValueError, match="conflicting labels"):
        prepare_examples(examples, total=20, seed=42, output_root=tmp_path)


def test_insufficient_source_fails_instead_of_writing_empty_success(tmp_path):
    examples = [
        {
            "image": Image.new("RGB", (8, 8), (index, 0, 0)),
            "label": 0,
            "img_id": f"real-{index}",
        }
        for index in range(10)
    ]

    with pytest.raises(RuntimeError, match="stream ended"):
        prepare_examples(examples, total=20, seed=42, output_root=tmp_path)


@pytest.mark.parametrize("total", [0, 19, 21])
def test_total_must_be_even_and_large_enough(tmp_path, total):
    with pytest.raises(ValueError, match="even integer"):
        prepare_examples([], total=total, seed=42, output_root=tmp_path)
