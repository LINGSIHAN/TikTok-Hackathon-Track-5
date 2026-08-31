from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from scripts import run_genimage_v2_calibration_kaggle as runner


def test_stage_candidate_copies_only_allowlisted_verified_members(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = b"candidate-v2"
    checkpoint_path = tmp_path / "checkpoint.bin"
    checkpoint_path.write_bytes(checkpoint)
    checkpoint_hash = runner.training_runner.file_sha256(checkpoint_path)
    sid_predictions = b"image_path,label,transform,severity,pred\na,0,clean,,0.1\n"
    gen_predictions = b"image_path,label,transform,severity,pred\nb,1,clean,,0.9\n"
    sid_path = tmp_path / "sid.csv"
    gen_path = tmp_path / "gen.csv"
    sid_path.write_bytes(sid_predictions)
    gen_path.write_bytes(gen_predictions)
    evaluation_context = {
        "audit_artifacts": {
            "v2_sid": {"predictions_sha256": runner.training_runner.file_sha256(sid_path)},
            "v2_genimage": {"predictions_sha256": runner.training_runner.file_sha256(gen_path)},
        }
    }
    export_context = {
        "v2_checkpoint_sha256": checkpoint_hash,
        "wildfake_used": False,
        "sid_manifest_sha256": runner.training_runner.EXPECTED_SID_MANIFEST_SHA256,
        "genimage_manifest_sha256": runner.EXPECTED_GENIMAGE_MANIFEST_SHA256,
        "combined_manifest_sha256": runner.EXPECTED_COMBINED_MANIFEST_SHA256,
        "genimage_selected_digest": runner.EXPECTED_GENIMAGE_SELECTED_DIGEST,
    }
    archive = tmp_path / "candidate.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(runner.MEMBERS["checkpoint"], checkpoint)
        bundle.writestr(
            runner.MEMBERS["metadata"],
            json.dumps(
                {
                    "checkpoint_sha256": checkpoint_hash,
                    "parent_checkpoint_sha256": runner.training_runner.V1_SHA256,
                    "dataset_lineage": {
                        "sid_manifest_sha256": runner.training_runner.EXPECTED_SID_MANIFEST_SHA256,
                        "genimage_manifest_sha256": runner.EXPECTED_GENIMAGE_MANIFEST_SHA256,
                        "combined_manifest_sha256": runner.EXPECTED_COMBINED_MANIFEST_SHA256,
                        "genimage_selected_digest": runner.EXPECTED_GENIMAGE_SELECTED_DIGEST,
                    },
                }
            ),
        )
        bundle.writestr(runner.MEMBERS["sid_test"], sid_predictions)
        bundle.writestr(runner.MEMBERS["genimage_test"], gen_predictions)
        bundle.writestr(runner.MEMBERS["evaluation_context"], json.dumps(evaluation_context))
        bundle.writestr(runner.MEMBERS["export_context"], json.dumps(export_context))
        bundle.writestr("ignored/untrusted.txt", "not extracted")
    monkeypatch.setattr(runner, "EXPECTED_V2_SHA256", checkpoint_hash)
    monkeypatch.setattr(
        runner,
        "EXPECTED_SID_TEST_PREDICTIONS_SHA256",
        runner.training_runner.file_sha256(sid_path),
    )
    monkeypatch.setattr(
        runner,
        "EXPECTED_GENIMAGE_TEST_PREDICTIONS_SHA256",
        runner.training_runner.file_sha256(gen_path),
    )

    staged = runner.stage_candidate(archive, tmp_path / "staged")

    assert set(staged) == set(runner.MEMBERS)
    assert staged["checkpoint"].read_bytes() == checkpoint
    assert not (tmp_path / "staged/untrusted.txt").exists()


def test_stage_candidate_rejects_duplicate_required_member(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "duplicate.zip"
    with pytest.warns(UserWarning, match="Duplicate name"):
        with zipfile.ZipFile(archive, "w") as bundle:
            bundle.writestr(runner.MEMBERS["checkpoint"], b"one")
            bundle.writestr(runner.MEMBERS["checkpoint"], b"two")

    with pytest.raises(RuntimeError, match="exactly one"):
        runner.stage_candidate(archive, tmp_path / "staged")


def test_safe_work_dir_is_scoped(tmp_path: Path) -> None:
    assert runner._safe_work_dir(tmp_path) == (
        tmp_path.resolve() / "genimage_v2_calibration_work"
    )


def test_calibration_runner_contains_no_training_or_wildfake_path() -> None:
    source = Path(runner.__file__).read_text(encoding="utf-8")

    assert '"src.training.train"' not in source
    assert "wildfake_demo" not in source.casefold()


def test_calibration_notebook_calls_only_calibration_runner() -> None:
    notebook_path = runner.REPOSITORY_ROOT / "notebooks/calibrate_genimage_v2_kaggle.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = "\n".join(
        line
        for cell in notebook["cells"]
        for line in cell.get("source", [])
    )

    assert "run_genimage_v2_calibration_kaggle.py" in source
    assert "run_genimage_v2_kaggle.py" not in source
    assert "src.training.train" not in source
