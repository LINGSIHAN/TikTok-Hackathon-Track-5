"""Evaluate a checkpoint on clean images and the complete transform grid."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from PIL import Image
from torch.utils.data import DataLoader

from src.data.preprocessing import (
    build_image_preprocess,
    normalize_pil_image,
    resolve_checkpoint_image_size,
)
from src.evaluation.metrics import compute_binary_metrics
from src.evaluation.plotting import save_robustness_plot
from src.training.engine import extract_logits, resolve_device, seed_worker, set_global_seed


@dataclass(frozen=True)
class Scenario:
    transform: str
    severity: Any | None


class EvaluationImageTransform:
    """Apply one deterministic test transform, then model preprocessing."""

    def __init__(
        self,
        image_size: int,
        scenario: Scenario,
        *,
        seed: int = 42,
    ) -> None:
        self.scenario = scenario
        self.seed = seed
        self.preprocess = build_image_preprocess(image_size)

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = normalize_pil_image(image)
        if self.scenario.transform != "clean":
            from src.transforms.robustness import apply_transform

            digest = hashlib.sha256(image.tobytes()).digest()
            image_seed = self.seed ^ int.from_bytes(digest[:4], "little")
            image = apply_transform(
                image,
                self.scenario.transform,
                self.scenario.severity,
                seed=image_seed,
            )
        return self.preprocess(image)


def build_scenarios(transform_grid: Mapping[str, Sequence[Any]]) -> list[Scenario]:
    """Expand the transform grid in stable insertion order, prefixed by clean."""

    scenarios = [Scenario("clean", None)]
    for transform_name, severities in transform_grid.items():
        for severity in severities:
            scenarios.append(Scenario(str(transform_name), severity))
    return scenarios


def _manifest_paths(manifest_path: str | Path, split: str) -> list[str]:
    with Path(manifest_path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "path" not in reader.fieldnames:
            raise ValueError("Manifest must contain a 'path' column")
        if "split" not in reader.fieldnames:
            raise ValueError("Manifest must contain a 'split' column")
        return [row["path"] for row in reader if row["split"] == split]


def _unpack_evaluation_batch(
    batch: Any,
) -> tuple[torch.Tensor, torch.Tensor, Sequence[str] | None]:
    paths: Sequence[str] | None = None
    if isinstance(batch, Mapping):
        images = batch.get("image", batch.get("images"))
        labels = batch.get("label", batch.get("labels"))
        candidate_paths = batch.get("path", batch.get("paths"))
        if candidate_paths is not None:
            paths = [str(value) for value in candidate_paths]
    elif isinstance(batch, (tuple, list)) and len(batch) >= 2:
        images, labels = batch[0], batch[1]
        if len(batch) >= 3:
            paths = [str(value) for value in batch[2]]
    else:
        raise TypeError("Unsupported evaluation batch structure")
    if not isinstance(images, torch.Tensor) or not isinstance(labels, torch.Tensor):
        raise TypeError("Evaluation images and labels must be torch tensors")
    return images, labels, paths


def predict_loader(
    model: torch.nn.Module,
    loader: Iterable[Any],
    *,
    device: torch.device,
    fallback_paths: Sequence[str] | None = None,
) -> tuple[list[int], list[float], list[str]]:
    """Return labels, AIGC probabilities, and paths for one scenario."""

    model.eval()
    labels: list[int] = []
    probabilities: list[float] = []
    paths: list[str] = []
    offset = 0
    with torch.inference_mode():
        for batch in loader:
            images, batch_labels, batch_paths = _unpack_evaluation_batch(batch)
            images = images.to(device, non_blocking=device.type == "cuda")
            logits = extract_logits(model(images))
            batch_probabilities = torch.sigmoid(logits).cpu().numpy()
            label_values = batch_labels.reshape(-1).cpu().to(torch.int64).numpy()
            batch_size = int(label_values.shape[0])
            if batch_probabilities.shape[0] != batch_size:
                raise ValueError("Model output and labels have different batch sizes")
            if batch_paths is None:
                if fallback_paths is None:
                    batch_paths = [str(index) for index in range(offset, offset + batch_size)]
                else:
                    batch_paths = fallback_paths[offset : offset + batch_size]
            if len(batch_paths) != batch_size:
                raise ValueError("Number of image paths does not match the batch size")

            labels.extend(int(value) for value in label_values)
            probabilities.extend(float(value) for value in batch_probabilities)
            paths.extend(batch_paths)
            offset += batch_size
    if not labels:
        raise ValueError("Cannot evaluate an empty data loader")
    if fallback_paths is not None and offset != len(fallback_paths):
        raise ValueError("Manifest order/length does not match the dataset")
    return labels, probabilities, paths


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    temporary_path.replace(path)


def _write_predictions(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["image_path", "label", "transform", "severity", "pred"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_scenarios(
    scenario_results: Sequence[Mapping[str, Any]],
    *,
    primary_metric: str = "roc_auc",
) -> dict[str, Any]:
    """Summarize clean, mean transformed, and worst transformed performance."""

    if not scenario_results or scenario_results[0]["transform"] != "clean":
        raise ValueError("The first scenario result must be the clean baseline")
    transformed = list(scenario_results[1:])
    clean_metrics = dict(scenario_results[0]["metrics"])
    mean_transformed: dict[str, float] = {}
    if transformed:
        for metric in clean_metrics:
            values = [float(result["metrics"][metric]) for result in transformed]
            finite = [value for value in values if math.isfinite(value)]
            mean_transformed[metric] = (
                sum(finite) / len(finite) if finite else float("nan")
            )

    finite_results = [
        result
        for result in transformed
        if math.isfinite(float(result["metrics"][primary_metric]))
    ]
    worst = (
        min(finite_results, key=lambda result: float(result["metrics"][primary_metric]))
        if finite_results
        else None
    )
    clean_primary = float(clean_metrics[primary_metric])
    worst_value = (
        float(worst["metrics"][primary_metric]) if worst is not None else float("nan")
    )
    return {
        "clean": clean_metrics,
        "mean_transformed": mean_transformed,
        "worst_case": (
            {
                "metric": primary_metric,
                "transform": worst["transform"],
                "severity": worst["severity"],
                "value": worst_value,
            }
            if worst is not None
            else None
        ),
        f"clean_to_worst_{primary_metric}_drop": (
            clean_primary - worst_value
            if math.isfinite(clean_primary) and math.isfinite(worst_value)
            else float("nan")
        ),
    }


def evaluate_checkpoint(
    *,
    manifest_path: str | Path,
    checkpoint_path: str | Path,
    split: str,
    output_dir: str | Path,
    batch_size: int = 32,
    num_workers: int = 0,
    device_name: str = "auto",
    image_size: int = 224,
    threshold: float = 0.5,
    seed: int = 42,
) -> dict[str, Any]:
    """Run clean plus complete robustness-grid evaluation and write artifacts."""

    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    if image_size <= 0:
        raise ValueError("image_size must be greater than zero")

    from src.data.dataset import ImageManifestDataset
    from src.models.checkpoints import load_checkpoint
    from src.models.efficientnet import build_model
    from src.transforms.robustness import TRANSFORM_GRID

    set_global_seed(seed)
    device = resolve_device(device_name)
    model = build_model(pretrained=False, freeze_backbone=False, unfreeze_last_blocks=0)
    checkpoint_metadata = load_checkpoint(model, checkpoint_path, device=str(device))
    preprocessing_size = resolve_checkpoint_image_size(
        checkpoint_metadata,
        requested_image_size=image_size,
    )
    model.to(device)

    manifest_paths = _manifest_paths(manifest_path, split)
    if not manifest_paths:
        raise ValueError(f"Manifest contains no samples for split '{split}'")
    output_path = Path(output_dir)
    generator = torch.Generator().manual_seed(seed)
    scenarios = build_scenarios(TRANSFORM_GRID)
    scenario_results: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []

    for scenario in scenarios:
        dataset = ImageManifestDataset(
            manifest_path,
            split,
            transform=EvaluationImageTransform(
                preprocessing_size,
                scenario,
                seed=seed,
            ),
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            worker_init_fn=seed_worker,
            generator=generator,
            pin_memory=device.type == "cuda",
            persistent_workers=num_workers > 0,
        )
        labels, probabilities, paths = predict_loader(
            model,
            loader,
            device=device,
            fallback_paths=manifest_paths,
        )
        metrics = compute_binary_metrics(labels, probabilities, threshold=threshold)
        scenario_result = {
            "transform": scenario.transform,
            "severity": scenario.severity,
            "num_samples": len(labels),
            "metrics": metrics,
        }
        scenario_results.append(scenario_result)
        for image_path, label, probability in zip(paths, labels, probabilities):
            prediction_rows.append(
                {
                    "image_path": image_path,
                    "label": label,
                    "transform": scenario.transform,
                    "severity": "" if scenario.severity is None else scenario.severity,
                    "pred": probability,
                }
            )

    artifact = {
        "schema_version": 1,
        "split": split,
        "threshold": threshold,
        "seed": seed,
        "checkpoint": str(checkpoint_path),
        "checkpoint_metadata": checkpoint_metadata,
        "scenarios": scenario_results,
        "summary": summarize_scenarios(scenario_results),
    }
    _write_predictions(output_path / "predictions.csv", prediction_rows)
    _write_json(output_path / "metrics.json", artifact)
    save_robustness_plot(scenario_results, output_path / "robustness.png")
    return _json_safe(artifact)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate clean and transformed AIGC detector performance."
    )
    parser.add_argument("--manifest", required=True, help="Dataset manifest CSV")
    parser.add_argument("--checkpoint", required=True, help="Model .safetensors file")
    parser.add_argument("--split", default="test", help="Manifest split to evaluate")
    parser.add_argument("--output-dir", required=True, help="Artifact output directory")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = evaluate_checkpoint(
        manifest_path=args.manifest,
        checkpoint_path=args.checkpoint,
        split=args.split,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device_name=args.device,
        image_size=args.image_size,
        threshold=args.threshold,
        seed=args.seed,
    )
    print(
        f"Evaluated {len(result['scenarios'])} scenarios; artifacts written to "
        f"{args.output_dir}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    raise SystemExit(main())
