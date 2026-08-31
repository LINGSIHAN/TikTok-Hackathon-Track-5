"""Run validation-only v2 threshold calibration on Kaggle without retraining."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts import run_genimage_v2_kaggle as training_runner  # noqa: E402


EXPECTED_V2_SHA256 = "b45022d9dab2a02300934c239eee24dd40ef8e402f24c1f27fc2d63a46117c12"
EXPECTED_SID_TEST_PREDICTIONS_SHA256 = (
    "c5aea34b33d15871a6bca2a7f617a198f1617389dfc7e1cf11ecb94002d99283"
)
EXPECTED_GENIMAGE_TEST_PREDICTIONS_SHA256 = (
    "225c780a4c03a1f57e067cbafe4df19583a60991a75edf5c1fbfc1d2c5f0b1ea"
)
EXPECTED_GENIMAGE_MANIFEST_SHA256 = (
    "3be38cf60de6da1fc523fa21796087d8c334cf33283bb9a3907d0f83e981453d"
)
EXPECTED_COMBINED_MANIFEST_SHA256 = (
    "6e65b23145c0dbe0033cdc7b83c34af6773e2013cf8c9c4c644e3feecc78a803"
)
EXPECTED_GENIMAGE_SELECTED_DIGEST = (
    "a037ddc229d579a11f8fc21b6f09ff6ae414febb5b89f9e627fbf8ca49833c5b"
)
MEMBERS = {
    "checkpoint": "local_audit/checkpoints/model_v2.safetensors",
    "metadata": "local_audit/checkpoints/model_v2_metadata.json",
    "sid_test": "local_audit/evaluations/v2_sid/predictions.csv",
    "genimage_test": "local_audit/evaluations/v2_genimage/predictions.csv",
    "evaluation_context": "local_audit/evaluations/execution_metadata.json",
    "export_context": "local_audit/execution_metadata.json",
}
MAX_MEMBER_BYTES = {
    "checkpoint": 64 * 1024 * 1024,
    "metadata": 2 * 1024 * 1024,
    "sid_test": 32 * 1024 * 1024,
    "genimage_test": 64 * 1024 * 1024,
    "evaluation_context": 2 * 1024 * 1024,
    "export_context": 2 * 1024 * 1024,
}


def _safe_work_dir(output_root: Path) -> Path:
    root = output_root.resolve()
    if root == Path(root.anchor):
        raise RuntimeError("Refusing to use a filesystem root as output-root")
    work = (root / "genimage_v2_calibration_work").resolve()
    if work.parent != root:
        raise RuntimeError("Calibration work directory escaped output-root")
    return work


def _copy_stream(source: Any, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("wb") as handle:
        shutil.copyfileobj(source, handle)
    temporary.replace(destination)


def stage_candidate(candidate_input: Path, destination: Path) -> dict[str, Path]:
    """Copy only allowlisted candidate artifacts from a ZIP or extracted tree."""

    candidate_input = candidate_input.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    staged = {key: destination / member for key, member in MEMBERS.items()}
    if candidate_input.is_file():
        if not zipfile.is_zipfile(candidate_input):
            raise RuntimeError("Candidate input must be the original v2 ZIP or its extracted directory")
        with zipfile.ZipFile(candidate_input) as archive:
            infos_by_name: dict[str, list[zipfile.ZipInfo]] = {}
            for info in archive.infolist():
                normalized = info.filename.replace("\\", "/").lstrip("./")
                infos_by_name.setdefault(normalized, []).append(info)
            for key, member in MEMBERS.items():
                infos = infos_by_name.get(member, [])
                if len(infos) != 1 or infos[0].is_dir():
                    raise RuntimeError(f"Candidate ZIP must contain exactly one {member}")
                info = infos[0]
                if info.file_size <= 0 or info.file_size > MAX_MEMBER_BYTES[key]:
                    raise RuntimeError(f"Candidate ZIP member has an unsafe size: {member}")
                with archive.open(info, "r") as source:
                    _copy_stream(source, staged[key])
    elif candidate_input.is_dir():
        for key, member in MEMBERS.items():
            direct = candidate_input / member
            matches = [direct] if direct.is_file() else [
                path
                for path in candidate_input.rglob(Path(member).name)
                if path.is_file() and path.as_posix().endswith(member)
            ]
            if len(matches) != 1:
                raise RuntimeError(
                    f"Candidate directory must contain exactly one extracted {member}"
                )
            if matches[0].stat().st_size <= 0 or matches[0].stat().st_size > MAX_MEMBER_BYTES[key]:
                raise RuntimeError(f"Candidate artifact has an unsafe size: {matches[0]}")
            staged[key].parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(matches[0], staged[key])
    else:
        raise RuntimeError(f"Candidate input does not exist: {candidate_input}")
    validate_staged_candidate(staged)
    return staged


def validate_staged_candidate(staged: dict[str, Path]) -> None:
    checkpoint_hash = training_runner.file_sha256(staged["checkpoint"])
    if checkpoint_hash != EXPECTED_V2_SHA256:
        raise RuntimeError(
            f"Unexpected v2 checkpoint SHA-256: {checkpoint_hash}; expected {EXPECTED_V2_SHA256}"
        )
    metadata = json.loads(staged["metadata"].read_text(encoding="utf-8"))
    if metadata.get("checkpoint_sha256") != checkpoint_hash:
        raise RuntimeError("Candidate metadata does not match the v2 checkpoint")
    if metadata.get("parent_checkpoint_sha256") != training_runner.V1_SHA256:
        raise RuntimeError("Candidate metadata does not identify the frozen v1 parent")
    lineage = metadata.get("dataset_lineage", {})
    expected_lineage = {
        "sid_manifest_sha256": training_runner.EXPECTED_SID_MANIFEST_SHA256,
        "genimage_manifest_sha256": EXPECTED_GENIMAGE_MANIFEST_SHA256,
        "combined_manifest_sha256": EXPECTED_COMBINED_MANIFEST_SHA256,
        "genimage_selected_digest": EXPECTED_GENIMAGE_SELECTED_DIGEST,
    }
    if any(lineage.get(key) != value for key, value in expected_lineage.items()):
        raise RuntimeError("Candidate metadata has unexpected dataset lineage")
    context = json.loads(staged["evaluation_context"].read_text(encoding="utf-8"))
    audit = context.get("audit_artifacts", {})
    expected_prediction_hashes = {
        "sid_test": EXPECTED_SID_TEST_PREDICTIONS_SHA256,
        "genimage_test": EXPECTED_GENIMAGE_TEST_PREDICTIONS_SHA256,
    }
    for key, expected in expected_prediction_hashes.items():
        actual = training_runner.file_sha256(staged[key])
        audit_name = "v2_sid" if key == "sid_test" else "v2_genimage"
        if audit.get(audit_name, {}).get("predictions_sha256") != expected:
            raise RuntimeError(f"Candidate audit context has an unexpected {key} hash")
        if expected != actual:
            raise RuntimeError(f"Candidate {key} predictions do not match their audit hash")
    export_context = json.loads(staged["export_context"].read_text(encoding="utf-8"))
    if export_context.get("v2_checkpoint_sha256") != checkpoint_hash:
        raise RuntimeError("Candidate export context does not match the checkpoint")
    expected_export_lineage = {
        "sid_manifest_sha256": training_runner.EXPECTED_SID_MANIFEST_SHA256,
        "genimage_manifest_sha256": EXPECTED_GENIMAGE_MANIFEST_SHA256,
        "combined_manifest_sha256": EXPECTED_COMBINED_MANIFEST_SHA256,
        "genimage_selected_digest": EXPECTED_GENIMAGE_SELECTED_DIGEST,
    }
    if any(
        export_context.get(key) != value
        for key, value in expected_export_lineage.items()
    ):
        raise RuntimeError("Candidate export context has unexpected dataset lineage")
    if export_context.get("wildfake_used") is not False:
        raise RuntimeError("Candidate export does not preserve the WildFake holdout")


def ensure_prepared_data(input_root: Path) -> None:
    try:
        training_runner.validate_sid_subset(REPOSITORY_ROOT)
        training_runner.validate_prepared_v2_data(REPOSITORY_ROOT)
        print("Reusing the validated SID and GenImage prepared images.", flush=True)
        return
    except (FileNotFoundError, RuntimeError) as error:
        print("Prepared images need deterministic recreation:", error, flush=True)
    training_runner.ensure_sid_subset(REPOSITORY_ROOT)
    training_runner.run(
        sys.executable,
        "scripts/prepare_genimage_v2.py",
        "--input-root",
        input_root,
        "--sid-manifest",
        training_runner.SID_MANIFEST,
        "--output-root",
        REPOSITORY_ROOT,
        "--seed",
        "42",
        "--confirm-license",
    )
    training_runner.validate_prepared_v2_data(REPOSITORY_ROOT)


def validate_calibration_manifests() -> None:
    sid_hash = training_runner.file_sha256(
        REPOSITORY_ROOT / training_runner.SID_MANIFEST
    )
    genimage_hash = training_runner.file_sha256(
        REPOSITORY_ROOT / training_runner.GENIMAGE_MANIFEST
    )
    if sid_hash != training_runner.EXPECTED_SID_MANIFEST_SHA256:
        raise RuntimeError("Calibration SID manifest differs from the reviewed v2 run")
    if genimage_hash != EXPECTED_GENIMAGE_MANIFEST_SHA256:
        raise RuntimeError("Calibration GenImage manifest differs from the reviewed v2 run")
    combined_hash = training_runner.file_sha256(
        REPOSITORY_ROOT / training_runner.COMBINED_MANIFEST
    )
    if combined_hash != EXPECTED_COMBINED_MANIFEST_SHA256:
        raise RuntimeError("Calibration combined manifest differs from the reviewed v2 run")
    summary = json.loads(
        (REPOSITORY_ROOT / training_runner.GENIMAGE_SUMMARY).read_text(
            encoding="utf-8"
        )
    )
    if summary.get("selected_digest") != EXPECTED_GENIMAGE_SELECTED_DIGEST:
        raise RuntimeError("Calibration GenImage selection digest differs from v2 lineage")


def _evaluate_validation(
    *, manifest: Path, checkpoint: Path, output_dir: Path, device: str = "cuda"
) -> None:
    training_runner.run(
        sys.executable,
        "-m",
        "src.evaluation.evaluate",
        "--manifest",
        manifest,
        "--checkpoint",
        checkpoint,
        "--split",
        "val",
        "--output-dir",
        output_dir,
        "--root-dir",
        REPOSITORY_ROOT,
        "--mode",
        "clean",
        "--batch-size",
        "64",
        "--num-workers",
        "2",
        "--device",
        device,
        "--image-size",
        "224",
        "--threshold",
        "0.5",
        "--seed",
        "42",
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def package_calibration_export(
    *, work_dir: Path, output_root: Path, staged: dict[str, Path], repository: dict[str, str]
) -> Path:
    export_dir = (output_root.resolve() / "genimage_v2_calibration_export").resolve()
    if export_dir.parent != output_root.resolve():
        raise RuntimeError("Calibration export escaped output-root")
    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True)
    files = {
        work_dir / "public/threshold_calibration.json": Path("public/threshold_calibration.json"),
        work_dir / "public/threshold-calibration-report.md": Path("public/threshold-calibration-report.md"),
        work_dir / "validation/sid/metrics.json": Path("local_audit/validation/sid/metrics.json"),
        work_dir / "validation/sid/predictions.csv": Path("local_audit/validation/sid/predictions.csv"),
        work_dir / "validation/genimage/metrics.json": Path("local_audit/validation/genimage/metrics.json"),
        work_dir / "validation/genimage/predictions.csv": Path("local_audit/validation/genimage/predictions.csv"),
        staged["checkpoint"]: Path("local_audit/checkpoints/model_v2.safetensors"),
        staged["metadata"]: Path("local_audit/checkpoints/model_v2_metadata.json"),
        staged["sid_test"]: Path("local_audit/source/v2_sid_test_predictions.csv"),
        staged["genimage_test"]: Path(
            "local_audit/source/v2_genimage_test_predictions.csv"
        ),
        staged["evaluation_context"]: Path("local_audit/source/evaluation_metadata.json"),
        staged["export_context"]: Path("local_audit/source/export_metadata.json"),
    }
    for source, relative in files.items():
        if not source.is_file() or source.stat().st_size == 0:
            raise RuntimeError(f"Required calibration artifact is missing: {source}")
        destination = export_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    calibration = json.loads(
        (work_dir / "public/threshold_calibration.json").read_text(encoding="utf-8")
    )
    context = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository": repository,
        "candidate_checkpoint_sha256": training_runner.file_sha256(staged["checkpoint"]),
        "calibration_summary_sha256": training_runner.file_sha256(
            work_dir / "public/threshold_calibration.json"
        ),
        "selection_status": calibration["selection"]["status"],
        "promotion_recommendation": calibration["promotion"]["recommendation"],
        "evidence_status": "exploratory_rescore_of_previously_observed_test_predictions",
        "wildfake_used": False,
        "retraining_performed": False,
    }
    _write_json(export_dir / "local_audit/execution_metadata.json", context)
    (export_dir / "README.txt").write_text(
        "RealityCheck GenImage v2 validation-only calibration export\n\n"
        "The numeric threshold was selected on clean validation predictions "
        "only. The earlier 0.50 test review motivated this calibration, so the "
        "saved test re-score is exploratory rather than a fresh holdout. No "
        "model retraining was performed.\n",
        encoding="utf-8",
    )
    archive = Path(
        shutil.make_archive(
            str(output_root.resolve() / "genimage_v2_calibration_export"),
            "zip",
            export_dir,
        )
    )
    print("Validated calibration export ready:", archive, flush=True)
    return archive


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calibrate the existing GenImage v2 candidate on validation data only."
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--candidate-input", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("/kaggle/working"))
    parser.add_argument("--license-confirmed", action="store_true")
    parser.add_argument("--skip-tests", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.license_confirmed:
        raise RuntimeError("Calibration is blocked until --license-confirmed is supplied")
    input_root = args.input_root.resolve()
    if not input_root.is_dir():
        raise RuntimeError(
            "GenImage input is missing. Attach cartografia/unbiased-tiny-genimage version 1."
        )
    os.chdir(REPOSITORY_ROOT)
    print("[Phase 1/5] Verifying GPU, repository, dataset, and candidate ZIP...", flush=True)
    repository = training_runner.repository_context(REPOSITORY_ROOT)
    training_runner.validate_cuda()
    training_runner.validate_genimage_attachment(input_root)
    if training_runner.file_sha256(REPOSITORY_ROOT / training_runner.V1_CHECKPOINT) != training_runner.V1_SHA256:
        raise RuntimeError("Frozen v1 checkpoint hash mismatch")
    work_dir = _safe_work_dir(args.output_root)
    if work_dir.exists():
        shutil.rmtree(work_dir)
    staged = stage_candidate(args.candidate_input, work_dir / "candidate")
    print("[Phase 2/5] Recreating the locked validation images (this can be quiet for several minutes)...", flush=True)
    ensure_prepared_data(input_root)
    validate_calibration_manifests()
    if not args.skip_tests:
        print("[Phase 3/5] Running calibration safety tests...", flush=True)
        training_runner.run(
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/evaluation/test_calibration.py",
            "tests/scripts/test_calibrate_genimage_v2_threshold.py",
            "tests/scripts/test_run_genimage_v2_calibration_kaggle.py",
        )
    else:
        print("[Phase 3/5] Safety tests explicitly skipped.", flush=True)
    print("[Phase 4/5] Running SID validation inference...", flush=True)
    _evaluate_validation(
        manifest=REPOSITORY_ROOT / training_runner.SID_MANIFEST,
        checkpoint=staged["checkpoint"],
        output_dir=work_dir / "validation/sid",
    )
    print("[Phase 4/5] Running GenImage validation inference...", flush=True)
    _evaluate_validation(
        manifest=REPOSITORY_ROOT / training_runner.GENIMAGE_MANIFEST,
        checkpoint=staged["checkpoint"],
        output_dir=work_dir / "validation/genimage",
    )
    print("[Phase 5/5] Locking the threshold, re-scoring saved tests, and packaging...", flush=True)
    training_runner.run(
        sys.executable,
        "scripts/calibrate_genimage_v2_threshold.py",
        "--sid-validation",
        work_dir / "validation/sid/predictions.csv",
        "--genimage-validation",
        work_dir / "validation/genimage/predictions.csv",
        "--sid-test",
        staged["sid_test"],
        "--genimage-test",
        staged["genimage_test"],
        "--output-json",
        work_dir / "public/threshold_calibration.json",
        "--output-report",
        work_dir / "public/threshold-calibration-report.md",
    )
    package_calibration_export(
        work_dir=work_dir,
        output_root=args.output_root,
        staged=staged,
        repository=repository,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    raise SystemExit(main())
