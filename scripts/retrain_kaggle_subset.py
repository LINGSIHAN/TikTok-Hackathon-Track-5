"""Retrain and package both models from an existing prepared Kaggle subset.

This recovery entrypoint is intentionally separate from data acquisition. It
expects the validated 6,000-image SID_Set subset produced by the main notebook,
runs the controlled clean/robust experiments in fresh subprocesses, validates
their artifacts, smoke-tests the selected checkpoint, and creates the same
``hackathon_export.zip`` bundle as the notebook.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT_TEXT = str(REPOSITORY_ROOT)
if REPOSITORY_ROOT_TEXT not in sys.path:
    sys.path.insert(0, REPOSITORY_ROOT_TEXT)

from scripts.prepare_sid_subset import DEFAULT_DATASET_REVISION

EXPECTED_TOTAL = 6_000
EXPECTED_SHUFFLE_BUFFER = 512
EXPECTED_COUNTS = {
    ("train", "0"): 2_400,
    ("train", "1"): 2_400,
    ("val", "0"): 300,
    ("val", "1"): 300,
    ("test", "0"): 300,
    ("test", "1"): 300,
}
EXPECTED_SCENARIOS = 20
EXPECTED_REPOSITORY = "https://github.com/LINGSIHAN/TikTok-Hackathon-Track-5.git"
EXPECTED_BRANCH = "master"
ALLOWED_GENERATED_UNTRACKED = frozenset(
    {
        "data/processed/manifest.csv",
        "data/processed/manifest_summary.json",
        "artifacts/checkpoints/model.safetensors",
        "artifacts/checkpoints/model_metadata.json",
        "artifacts/metrics/training_history.json",
        "artifacts/metrics/metrics.json",
        "artifacts/metrics/predictions.csv",
        "artifacts/metrics/robustness.png",
        "artifacts/metrics/run_context.json",
        "artifacts/metrics/pip_freeze.txt",
        "artifacts/metrics/clean_baseline/metrics.json",
        "artifacts/metrics/clean_baseline/predictions.csv",
        "artifacts/metrics/clean_baseline/robustness.png",
    }
)


def run(*args: object) -> None:
    """Run one visible, fail-fast subprocess command."""

    command = [str(arg) for arg in args]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def validate_cuda() -> dict[str, Any]:
    """Run a real CUDA operation before the expensive recovery work."""

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU is available; choose Kaggle GPU T4 x2")
    gpu_name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    required_arch = f"sm_{capability[0]}{capability[1]}"
    if required_arch not in torch.cuda.get_arch_list():
        raise RuntimeError(
            f"PyTorch {torch.__version__} has no {required_arch} kernels for "
            f"{gpu_name}; choose Kaggle GPU T4 x2"
        )
    probe = torch.arange(16, dtype=torch.float32, device="cuda").reshape(4, 4)
    probe_checksum = float((probe @ probe.T).sum().item())
    torch.cuda.synchronize()
    print("CUDA tensor preflight passed on", gpu_name, "checksum:", probe_checksum)
    return {
        "python": sys.version.split()[0],
        "pytorch": torch.__version__,
        "torchvision": version("torchvision"),
        "pillow": version("Pillow"),
        "numpy": version("numpy"),
        "pandas": version("pandas"),
        "datasets": version("datasets"),
        "huggingface_hub": version("huggingface_hub"),
        "safetensors": version("safetensors"),
        "cuda_runtime": torch.version.cuda,
        "gpu": gpu_name,
        "gpu_capability": list(capability),
        "required_cuda_arch": required_arch,
        "compiled_cuda_arches": torch.cuda.get_arch_list(),
        "cuda_probe_checksum": probe_checksum,
    }


def validate_worktree_clean(root: Path) -> None:
    """Reject code changes while allowing only known generated run outputs."""

    tracked_changes = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        text=True,
        cwd=root,
    ).strip()
    if tracked_changes:
        raise RuntimeError("Tracked repository changes would make the run non-reproducible")
    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard"],
        text=True,
        cwd=root,
    ).splitlines()
    unexpected_untracked = sorted(set(untracked) - ALLOWED_GENERATED_UNTRACKED)
    if unexpected_untracked:
        raise RuntimeError(
            "Unexpected untracked files would make the run non-reproducible: "
            + ", ".join(unexpected_untracked)
        )


def validate_repository(root: Path) -> dict[str, str]:
    """Require a clean checkout at the freshly fetched remote master commit."""

    origin = subprocess.check_output(
        ["git", "remote", "get-url", "origin"], text=True, cwd=root
    ).strip()
    normalized_origin = origin.rstrip("/").removesuffix(".git")
    normalized_expected = EXPECTED_REPOSITORY.rstrip("/").removesuffix(".git")
    if normalized_origin != normalized_expected:
        raise RuntimeError(f"Unexpected Git origin: {origin}")

    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], text=True, cwd=root
    ).strip()
    if branch != EXPECTED_BRANCH:
        raise RuntimeError(
            f"Expected branch {EXPECTED_BRANCH!r}, found {branch or 'detached HEAD'!r}"
        )

    validate_worktree_clean(root)

    run("git", "fetch", "--depth", "1", "origin", EXPECTED_BRANCH)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=root).strip()
    fetched = subprocess.check_output(
        ["git", "rev-parse", "FETCH_HEAD"], text=True, cwd=root
    ).strip()
    if head != fetched:
        raise RuntimeError(
            "Local HEAD is not the fetched origin/master commit; run "
            "`git pull --ff-only origin master` and retry"
        )
    return {"repository": origin, "branch": branch, "commit": head}


def validate_prepared_subset(root: Path) -> list[dict[str, str]]:
    """Fail unless the exact expected prepared subset is present and intact."""

    manifest_path = root / "data/processed/manifest.csv"
    summary_path = root / "data/processed/manifest_summary.json"
    with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "path",
            "label",
            "split",
            "source_id",
            "sha256",
            "dataset",
            "source_split",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise RuntimeError("Prepared manifest schema is incomplete")
        rows = list(reader)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    recorded_revision = summary.get("dataset_revision")
    if recorded_revision not in (None, DEFAULT_DATASET_REVISION):
        raise RuntimeError("Prepared subset uses an unexpected SID_Set revision")
    recorded_shuffle_buffer = summary.get("shuffle_buffer")
    if recorded_shuffle_buffer not in (None, EXPECTED_SHUFFLE_BUFFER):
        raise RuntimeError("Prepared subset uses an unexpected shuffle buffer")

    if len(rows) != EXPECTED_TOTAL or summary.get("total") != EXPECTED_TOTAL:
        raise RuntimeError("Prepared subset must contain exactly 6,000 rows")
    expected_summary = {
        "schema_version": 1,
        "dataset": "saberzl/SID_Set",
        "source_split": "train",
        "seed": 42,
        "sha256_definition": "SHA-256 of the stored normalized JPEG bytes",
    }
    if any(summary.get(key) != value for key, value in expected_summary.items()):
        raise RuntimeError("Prepared subset provenance does not match SID_Set seed 42")

    counts = Counter((row["split"], row["label"]) for row in rows)
    if dict(counts) != EXPECTED_COUNTS:
        raise RuntimeError(f"Unexpected split/class counts: {dict(counts)}")
    expected_split_counts = {
        split: {label: EXPECTED_COUNTS[(split, label)] for label in ("0", "1")}
        for split in ("train", "val", "test")
    }
    if summary.get("split_counts") != expected_split_counts or summary.get(
        "class_counts"
    ) != {"0_real": 3000, "1_full_synthetic": 3000}:
        raise RuntimeError("Prepared manifest counts do not reconcile to its summary")

    seen_hashes: set[str] = set()
    source_splits: dict[str, str] = {}
    root_resolved = root.resolve()
    for index, row in enumerate(rows, start=2):
        if row["dataset"] != "saberzl/SID_Set" or row["source_split"] != "train":
            raise RuntimeError(
                f"Prepared image provenance mismatch at manifest row {index}"
            )

        digest = row["sha256"]
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise RuntimeError(f"Invalid SHA-256 at manifest row {index}")
        if digest in seen_hashes:
            raise RuntimeError(f"Duplicate SHA-256 at manifest row {index}")
        seen_hashes.add(digest)

        source_id = row["source_id"]
        if not source_id:
            raise RuntimeError(f"Empty source_id at manifest row {index}")
        previous = source_splits.setdefault(source_id, row["split"])
        if previous != row["split"]:
            raise RuntimeError(f"source_id crosses splits: {source_id}")

        expected_relative = (
            Path("data/raw")
            / ("authentic" if row["label"] == "0" else "generated")
            / f"{digest}.jpg"
        )
        relative_path = Path(row["path"])
        if relative_path != expected_relative:
            raise RuntimeError(
                f"Prepared image path disagrees with its label/hash: {row['path']}"
            )
        image_path = (root / relative_path).resolve()
        try:
            image_path.relative_to(root_resolved)
        except ValueError as error:
            raise RuntimeError(f"Manifest path escapes the repository: {row['path']}") from error
        if not image_path.is_file() or image_path.stat().st_size == 0:
            raise RuntimeError(f"Prepared image is missing or empty: {row['path']}")
        if file_sha256(image_path) != digest:
            raise RuntimeError(f"Prepared image hash mismatch: {row['path']}")

    print(f"Validated {len(rows)} prepared images and manifest rows.")
    return rows


def validate_metrics(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    scenarios = payload.get("scenarios")
    if not isinstance(scenarios, list) or len(scenarios) != EXPECTED_SCENARIOS:
        raise RuntimeError(f"{path} does not contain 20 evaluation scenarios")
    for scenario in scenarios:
        if scenario.get("num_samples") != 600:
            raise RuntimeError(f"Unexpected scenario sample count in {path}")
        for name, value in scenario.get("metrics", {}).items():
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise RuntimeError(f"Non-finite metric {name} in {path}")
    return payload


def require_artifacts(root: Path) -> None:
    required = [
        root / "artifacts/checkpoints/model.safetensors",
        root / "artifacts/checkpoints/model_metadata.json",
        root / "artifacts/metrics/training_history.json",
        root / "artifacts/metrics/metrics.json",
        root / "artifacts/metrics/predictions.csv",
        root / "artifacts/metrics/robustness.png",
        root / "artifacts/runs/clean/model.safetensors",
        root / "artifacts/runs/clean/model_metadata.json",
        root / "artifacts/runs/clean/history.json",
        root / "artifacts/metrics/clean_baseline/metrics.json",
        root / "artifacts/metrics/clean_baseline/predictions.csv",
        root / "artifacts/metrics/clean_baseline/robustness.png",
    ]
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise RuntimeError("Missing or empty required artifacts: " + ", ".join(missing))


def smoke_checkpoint(root: Path, rows: list[dict[str, str]]) -> float:
    from PIL import Image

    from src.inference.predictor import Predictor

    test_row = next(row for row in rows if row["split"] == "test")
    predictor = Predictor.from_checkpoint(
        root / "artifacts/checkpoints/model.safetensors",
        device="cuda",
    )
    with Image.open(root / test_row["path"]) as image:
        probability = predictor.predict_pil(image)
    if not 0.0 <= probability <= 1.0:
        raise RuntimeError("Checkpoint smoke test returned an invalid probability")
    print("Checkpoint smoke probability:", probability)
    return probability


def package_export(
    root: Path,
    output_root: Path,
    repository_info: dict[str, str],
    cuda_info: dict[str, Any],
) -> Path:
    from src.data.preprocessing import PREPROCESSING_CONTRACT_ID

    output_root = output_root.resolve()
    if output_root == Path(output_root.anchor):
        raise RuntimeError("Refusing to use a filesystem root as the output directory")
    output_root.mkdir(parents=True, exist_ok=True)
    export_dir = (output_root / "export").resolve()
    if export_dir != output_root / "export":
        raise RuntimeError("Resolved export directory does not match the requested output root")
    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir()

    relative_files = [
        "artifacts/checkpoints/model.safetensors",
        "artifacts/checkpoints/model_metadata.json",
        "artifacts/runs/clean/model.safetensors",
        "artifacts/runs/clean/model_metadata.json",
        "artifacts/runs/clean/history.json",
        "artifacts/metrics/training_history.json",
        "artifacts/metrics/metrics.json",
        "artifacts/metrics/predictions.csv",
        "artifacts/metrics/robustness.png",
        "artifacts/metrics/clean_baseline/metrics.json",
        "artifacts/metrics/clean_baseline/predictions.csv",
        "artifacts/metrics/clean_baseline/robustness.png",
        "data/processed/manifest.csv",
        "data/processed/manifest_summary.json",
        "configs/train_clean.yaml",
        "configs/train_robust.yaml",
        "requirements.txt",
        "requirements-train.txt",
    ]
    copies = [(root / relative, export_dir / relative) for relative in relative_files]
    for source, destination in copies:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(source, destination)

    manifest_path = root / "data/processed/manifest.csv"
    checkpoint_path = root / "artifacts/checkpoints/model.safetensors"
    clean_checkpoint_path = root / "artifacts/runs/clean/model.safetensors"
    manifest_summary = json.loads(
        (root / "data/processed/manifest_summary.json").read_text(encoding="utf-8")
    )
    dataset_revision = manifest_summary.get("dataset_revision")
    run_context = {
        **repository_info,
        "subset_size": EXPECTED_TOTAL,
        "seed": 42,
        "shuffle_buffer": manifest_summary.get("shuffle_buffer"),
        "dataset_revision": dataset_revision,
        "dataset_revision_status": (
            "pinned" if dataset_revision == DEFAULT_DATASET_REVISION else "legacy_unrecorded"
        ),
        **cuda_info,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "preprocessing_contract": PREPROCESSING_CONTRACT_ID,
        "manifest_sha256": file_sha256(manifest_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "clean_checkpoint_sha256": file_sha256(clean_checkpoint_path),
    }
    context_path = export_dir / "artifacts/metrics/run_context.json"
    context_path.write_text(json.dumps(run_context, indent=2) + "\n", encoding="utf-8")
    freeze_path = export_dir / "artifacts/metrics/pip_freeze.txt"
    freeze_path.write_text(
        subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"], text=True
        ),
        encoding="utf-8",
    )

    archive_base = output_root / "hackathon_export"
    archive = Path(shutil.make_archive(str(archive_base), "zip", export_dir))
    print("Validated export ready:", archive)
    return archive


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Retrain both models from an existing prepared Kaggle subset."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/kaggle/working"),
        help="Directory receiving export/ and hackathon_export.zip.",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip pytest only when it has already passed in this same environment.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    os.chdir(REPOSITORY_ROOT)
    cuda_info = validate_cuda()
    repository_info = validate_repository(REPOSITORY_ROOT)
    rows = validate_prepared_subset(REPOSITORY_ROOT)
    if not args.skip_tests:
        run(sys.executable, "-m", "pytest", "-q")

    run(
        sys.executable,
        "-m",
        "src.training.train",
        "--config",
        "configs/train_clean.yaml",
        "--device",
        "cuda",
    )
    run(
        sys.executable,
        "-m",
        "src.evaluation.evaluate",
        "--manifest",
        "data/processed/manifest.csv",
        "--checkpoint",
        "artifacts/runs/clean/model.safetensors",
        "--split",
        "test",
        "--output-dir",
        "artifacts/metrics/clean_baseline",
        "--device",
        "cuda",
    )
    run(
        sys.executable,
        "-m",
        "src.training.train",
        "--config",
        "configs/train_robust.yaml",
        "--device",
        "cuda",
    )
    run(
        sys.executable,
        "-m",
        "src.evaluation.evaluate",
        "--manifest",
        "data/processed/manifest.csv",
        "--checkpoint",
        "artifacts/checkpoints/model.safetensors",
        "--split",
        "test",
        "--output-dir",
        "artifacts/metrics",
        "--device",
        "cuda",
    )

    require_artifacts(REPOSITORY_ROOT)
    clean = validate_metrics(REPOSITORY_ROOT / "artifacts/metrics/clean_baseline/metrics.json")
    robust = validate_metrics(REPOSITORY_ROOT / "artifacts/metrics/metrics.json")
    smoke_checkpoint(REPOSITORY_ROOT, rows)
    print(
        "Clean / robust mean transformed ROC-AUC:",
        clean["summary"]["mean_transformed"]["roc_auc"],
        "/",
        robust["summary"]["mean_transformed"]["roc_auc"],
    )
    package_export(REPOSITORY_ROOT, args.output_root, repository_info, cuda_info)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
