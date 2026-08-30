import json
from pathlib import Path

import pytest
from PIL import Image

import scripts.evaluate_wildfake_demo as wildfake_eval
from scripts.evaluate_wildfake_demo import (
    ValidatedSample,
    build_public_summary,
    confusion_summary,
    validate_dataset,
    verify_download_manifest,
    verify_frozen_checkpoint,
    write_evaluation_manifest,
    write_public_report,
    resolve_external_device,
    save_clean_public_figure,
)


def _image(path, color, image_format="PNG"):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (7, 5), color).save(path, format=image_format)


def _small_counts(monkeypatch):
    monkeypatch.setattr(wildfake_eval, "EXPECTED_COUNTS", {"real": 2, "generated": 3})
    monkeypatch.setattr(wildfake_eval, "EXPECTED_TOTAL", 5)


def _valid_fixture(monkeypatch, tmp_path):
    _small_counts(monkeypatch)
    root = tmp_path / "wildfake"
    _image(root / "real/r0.png", (1, 2, 3))
    _image(root / "real/r1.png", (4, 5, 6))
    _image(root / "generated/g0.png", (7, 8, 9))
    _image(root / "generated/g1.png", (10, 11, 12))
    _image(root / "generated/g2.png", (13, 14, 15))
    return root


def test_dataset_validation_exact_counts_and_deterministic_manifest(
    monkeypatch, tmp_path
):
    root = _valid_fixture(monkeypatch, tmp_path)

    samples, digest = validate_dataset(root)
    first = write_evaluation_manifest(tmp_path / "manifest.csv", samples)
    first_content = (tmp_path / "manifest.csv").read_bytes()
    second = write_evaluation_manifest(tmp_path / "manifest.csv", list(reversed(samples)))

    assert len(samples) == 5
    assert len(digest) == 64
    assert first == second
    assert (tmp_path / "manifest.csv").read_bytes() == first_content
    assert all(sample.path.startswith(("real/", "generated/")) for sample in samples)


def test_dataset_validation_rejects_wrong_count(monkeypatch, tmp_path):
    root = _valid_fixture(monkeypatch, tmp_path)
    (root / "real/r1.png").unlink()

    with pytest.raises(RuntimeError, match="expected exactly 2"):
        validate_dataset(root)


def test_dataset_validation_records_same_label_duplicate_hash(monkeypatch, tmp_path):
    root = _valid_fixture(monkeypatch, tmp_path)
    (root / "generated/g2.png").write_bytes((root / "generated/g1.png").read_bytes())
    duplicate_groups = []

    samples, _ = validate_dataset(root, duplicate_groups=duplicate_groups)

    assert len(samples) == 5
    assert len(duplicate_groups) == 1
    assert duplicate_groups[0]["label"] == 1
    assert duplicate_groups[0]["paths"] == ["generated/g1.png", "generated/g2.png"]


def test_dataset_validation_rejects_conflicting_labels(monkeypatch, tmp_path):
    root = _valid_fixture(monkeypatch, tmp_path)
    (root / "generated/g2.png").write_bytes((root / "real/r0.png").read_bytes())

    with pytest.raises(RuntimeError, match="Conflicting labels"):
        validate_dataset(root)


def test_dataset_validation_rejects_corrupt_image(monkeypatch, tmp_path):
    root = _valid_fixture(monkeypatch, tmp_path)
    (root / "generated/g2.png").write_bytes(b"broken")

    with pytest.raises(RuntimeError, match="Unreadable image"):
        validate_dataset(root)


def test_download_manifest_must_match_locked_revision_counts_and_digest(
    monkeypatch, tmp_path
):
    root = _valid_fixture(monkeypatch, tmp_path)
    _, digest = validate_dataset(root)
    payload = {
        "revision": wildfake_eval.WILDFAKE_REVISION,
        "dataset_digest": digest,
        "total_count": 5,
        "class_counts": {"real": 2, "generated": 3},
    }
    (root / "download_manifest.json").write_text(json.dumps(payload), encoding="utf-8")

    assert verify_download_manifest(root, digest) == payload
    payload["dataset_digest"] = "0" * 64
    (root / "download_manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="aggregate download digest"):
        verify_download_manifest(root, digest)


def test_frozen_checkpoint_enforcement(tmp_path):
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"locked checkpoint")
    import hashlib

    expected = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    context = tmp_path / "run_context.json"
    context.write_text(json.dumps({"checkpoint_sha256": expected}), encoding="utf-8")
    assert verify_frozen_checkpoint(checkpoint, context) == expected

    checkpoint.write_bytes(b"changed checkpoint")
    with pytest.raises(RuntimeError, match="Frozen checkpoint SHA-256 mismatch"):
        verify_frozen_checkpoint(checkpoint, context)


def test_explicit_external_device_is_preserved():
    assert resolve_external_device("cpu") == "cpu"


def test_confusion_summary_is_class_normalized_for_imbalanced_data(monkeypatch):
    monkeypatch.setattr(wildfake_eval, "EXPECTED_COUNTS", {"real": 2, "generated": 3})
    rows = [
        {"transform": "clean", "label": "0", "pred": "0.1"},
        {"transform": "clean", "label": "0", "pred": "0.8"},
        {"transform": "clean", "label": "1", "pred": "0.9"},
        {"transform": "clean", "label": "1", "pred": "0.4"},
        {"transform": "clean", "label": "1", "pred": "0.7"},
        {"transform": "jpeg", "label": "1", "pred": "0.1"},
    ]

    result = confusion_summary(rows, threshold=0.5)

    assert result["raw"] == {
        "true_negative": 1,
        "false_positive": 1,
        "false_negative": 1,
        "true_positive": 2,
    }
    assert result["class_normalized"]["real"] == {
        "predicted_real": 0.5,
        "predicted_generated": 0.5,
    }
    assert result["class_normalized"]["generated"]["predicted_real"] == pytest.approx(1 / 3)


def _evaluation_fixture():
    metrics = {
        "roc_auc": 0.8,
        "average_precision": 0.9,
        "balanced_accuracy": 0.7,
        "f1": 0.75,
        "false_positive_rate": 0.2,
        "false_negative_rate": 0.4,
        "brier_score": 0.15,
    }
    return {
        "scenarios": [
            {"transform": "clean", "severity": None, "num_samples": 5, "metrics": metrics}
        ],
        "summary": {
            "clean": metrics,
            "mean_transformed": {},
            "worst_case": None,
            "clean_to_worst_roc_auc_drop": None,
        },
    }


def test_public_outputs_are_aggregate_only_and_path_free(monkeypatch, tmp_path):
    monkeypatch.setattr(wildfake_eval, "EXPECTED_COUNTS", {"real": 2, "generated": 3})
    monkeypatch.setattr(wildfake_eval, "EXPECTED_TOTAL", 5)
    confusion = {
        "raw": {
            "true_negative": 1,
            "false_positive": 1,
            "false_negative": 1,
            "true_positive": 2,
        },
        "class_normalized": {
            "real": {"predicted_real": 0.5, "predicted_generated": 0.5},
            "generated": {"predicted_real": 1 / 3, "predicted_generated": 2 / 3},
        },
    }
    summary = build_public_summary(
        evaluation=_evaluation_fixture(),
        checkpoint_sha256="a" * 64,
        digest="b" * 64,
        manifest_sha256="c" * 64,
        confusion=confusion,
        duplicate_groups=[],
        mode="clean",
    )
    report = tmp_path / "report.md"
    write_public_report(report, summary)
    serialized = json.dumps(summary)
    report_text = report.read_text(encoding="utf-8")

    assert "image_path" not in serialized
    assert str(tmp_path) not in serialized
    assert str(tmp_path) not in report_text
    assert "never used for training" in report_text
    assert "13,841" not in report_text
    assert "5 total" in report_text

    figure = tmp_path / "summary.png"
    save_clean_public_figure(summary, figure)
    assert figure.is_file()
    assert figure.stat().st_size > 1_000
