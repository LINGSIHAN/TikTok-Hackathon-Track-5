import csv
import json

import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

import src.models.checkpoints as checkpoints_module
import src.models.efficientnet as efficientnet_module
from src.data.preprocessing import (
    PREPROCESSING_CONTRACT_ID,
    PREPROCESSING_METADATA_KEY,
)
from src.evaluation import evaluate as evaluate_module
from src.evaluation.evaluate import (
    EvaluationImageTransform,
    Scenario,
    _manifest_paths,
    _write_json,
    _write_predictions,
    build_scenarios,
    evaluate_checkpoint,
    predict_loader,
    summarize_scenarios,
)
from src.transforms import robustness


class IdentityLogitModel(torch.nn.Module):
    def forward(self, inputs):
        return inputs[:, :1]


def _patch_evaluation_checkpoint(monkeypatch, metadata) -> None:
    monkeypatch.setattr(
        efficientnet_module,
        "build_model",
        lambda **kwargs: IdentityLogitModel(),
    )
    monkeypatch.setattr(
        checkpoints_module,
        "load_checkpoint",
        lambda model, path, device: metadata,
    )


def test_evaluator_accepts_current_preprocessing_metadata(monkeypatch, tmp_path):
    metadata = {
        PREPROCESSING_METADATA_KEY: PREPROCESSING_CONTRACT_ID,
        "image_size": "224",
    }
    _patch_evaluation_checkpoint(monkeypatch, metadata)
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("path,label,split\n", encoding="utf-8")

    with pytest.raises(ValueError, match="no samples"):
        evaluate_checkpoint(
            manifest_path=manifest,
            checkpoint_path=tmp_path / "model.safetensors",
            split="test",
            output_dir=tmp_path / "output",
            device_name="cpu",
            image_size=224,
        )


@pytest.mark.parametrize(
    ("metadata", "image_size", "message"),
    [
        ({"image_size": "224"}, 224, "legacy checkpoints must be retrained"),
        (
            {
                PREPROCESSING_METADATA_KEY: "legacy-square-resize-v0",
                "image_size": "224",
            },
            224,
            "does not match",
        ),
        (
            {
                PREPROCESSING_METADATA_KEY: PREPROCESSING_CONTRACT_ID,
                "image_size": "224",
            },
            256,
            "requested image_size 256",
        ),
    ],
)
def test_evaluator_rejects_incompatible_preprocessing_metadata(
    monkeypatch,
    tmp_path,
    metadata,
    image_size,
    message,
):
    _patch_evaluation_checkpoint(monkeypatch, metadata)

    with pytest.raises(ValueError, match=message):
        evaluate_checkpoint(
            manifest_path=tmp_path / "not-reached.csv",
            checkpoint_path=tmp_path / "model.safetensors",
            split="test",
            output_dir=tmp_path / "output",
            device_name="cpu",
            image_size=image_size,
        )


def test_evaluation_applies_scenario_before_shared_preprocess(monkeypatch):
    events = []
    expected = torch.zeros(3, 12, 12)

    def fake_build_preprocess(image_size):
        assert image_size == 12

        def preprocess(image):
            events.append(("preprocess", image.getpixel((0, 0))))
            return expected

        return preprocess

    def fake_apply_transform(image, name, severity, seed):
        events.append(
            (
                "robustness",
                name,
                severity,
                isinstance(seed, int),
                image.mode,
                image.getpixel((0, 0)),
            )
        )
        transformed = image.copy()
        transformed.putpixel((0, 0), (0, 255, 0))
        return transformed

    monkeypatch.setattr(
        evaluate_module,
        "build_image_preprocess",
        fake_build_preprocess,
    )
    monkeypatch.setattr(robustness, "apply_transform", fake_apply_transform)
    transform = EvaluationImageTransform(12, Scenario("jpeg", 90), seed=7)

    result = transform(
        Image.new("RGBA", (40, 20), color=(0, 0, 255, 0))
    )

    assert result is expected
    assert events == [
        ("robustness", "jpeg", 90, True, "RGB", (255, 255, 255)),
        ("preprocess", (0, 255, 0)),
    ]


def test_build_scenarios_is_clean_first_and_stable():
    scenarios = build_scenarios({"jpeg": [90, 50], "blur": [0.5]})

    assert scenarios == [
        Scenario("clean", None),
        Scenario("jpeg", 90),
        Scenario("jpeg", 50),
        Scenario("blur", 0.5),
    ]


def test_predict_loader_uses_manifest_fallback_paths():
    loader = DataLoader(
        TensorDataset(torch.tensor([[-2.0], [2.0]]), torch.tensor([0, 1])),
        batch_size=2,
    )

    labels, probabilities, paths = predict_loader(
        IdentityLogitModel(),
        loader,
        device=torch.device("cpu"),
        fallback_paths=["real.jpg", "fake.jpg"],
    )

    assert labels == [0, 1]
    assert probabilities == pytest.approx([0.1192029, 0.8807971])
    assert paths == ["real.jpg", "fake.jpg"]


def test_summary_reports_worst_transformation_and_drop():
    results = [
        {
            "transform": "clean",
            "severity": None,
            "metrics": {"roc_auc": 0.9, "brier_score": 0.1},
        },
        {
            "transform": "jpeg",
            "severity": 30,
            "metrics": {"roc_auc": 0.7, "brier_score": 0.2},
        },
        {
            "transform": "blur",
            "severity": 2.0,
            "metrics": {"roc_auc": 0.6, "brier_score": 0.3},
        },
    ]

    summary = summarize_scenarios(results)

    assert summary["mean_transformed"]["roc_auc"] == pytest.approx(0.65)
    assert summary["worst_case"]["transform"] == "blur"
    assert summary["clean_to_worst_roc_auc_drop"] == pytest.approx(0.3)


def test_manifest_filter_and_artifact_writers(tmp_path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "path,label,split\ntrain.jpg,0,train\ntest.jpg,1,test\n",
        encoding="utf-8",
    )
    assert _manifest_paths(manifest, "test") == ["test.jpg"]

    predictions_path = tmp_path / "predictions.csv"
    _write_predictions(
        predictions_path,
        [
            {
                "image_path": "test.jpg",
                "label": 1,
                "transform": "clean",
                "severity": "",
                "pred": 0.8,
            }
        ],
    )
    with predictions_path.open(encoding="utf-8", newline="") as handle:
        assert list(csv.DictReader(handle))[0]["image_path"] == "test.jpg"

    metrics_path = tmp_path / "metrics.json"
    _write_json(metrics_path, {"undefined": float("nan")})
    assert json.loads(metrics_path.read_text(encoding="utf-8")) == {"undefined": None}


def test_evaluator_clean_mode_resolves_images_from_external_root(
    monkeypatch, tmp_path
):
    metadata = {
        PREPROCESSING_METADATA_KEY: PREPROCESSING_CONTRACT_ID,
        "image_size": "224",
    }
    _patch_evaluation_checkpoint(monkeypatch, metadata)
    monkeypatch.setattr(
        evaluate_module,
        "build_image_preprocess",
        lambda image_size: (
            lambda image: torch.tensor(
                [-2.0 if image.getpixel((0, 0))[0] < 128 else 2.0]
            )
        ),
    )
    image_root = tmp_path / "external"
    image_root.mkdir()
    Image.new("RGB", (4, 4), "black").save(image_root / "real.png")
    Image.new("RGB", (4, 4), "white").save(image_root / "generated.png")
    manifest = tmp_path / "audit" / "manifest.csv"
    manifest.parent.mkdir()
    manifest.write_text(
        "path,label,split\nreal.png,0,test\ngenerated.png,1,test\n",
        encoding="utf-8",
    )

    result = evaluate_checkpoint(
        manifest_path=manifest,
        checkpoint_path=tmp_path / "model.safetensors",
        split="test",
        output_dir=tmp_path / "output",
        root_dir=image_root,
        scenario_mode="clean",
        device_name="cpu",
        image_size=224,
    )

    assert result["scenario_mode"] == "clean"
    assert len(result["scenarios"]) == 1
    assert result["scenarios"][0]["transform"] == "clean"
    assert result["scenarios"][0]["metrics"]["roc_auc"] == pytest.approx(1.0)

    monkeypatch.setattr(robustness, "TRANSFORM_GRID", {"jpeg": (90,)})
    full_result = evaluate_checkpoint(
        manifest_path=manifest,
        checkpoint_path=tmp_path / "model.safetensors",
        split="test",
        output_dir=tmp_path / "full-output",
        root_dir=image_root,
        scenario_mode="full",
        device_name="cpu",
        image_size=224,
    )
    assert [scenario["transform"] for scenario in full_result["scenarios"]] == [
        "clean",
        "jpeg",
    ]


def test_evaluator_rejects_unknown_scenario_mode(tmp_path):
    with pytest.raises(ValueError, match="scenario_mode"):
        evaluate_checkpoint(
            manifest_path=tmp_path / "manifest.csv",
            checkpoint_path=tmp_path / "model.safetensors",
            split="test",
            output_dir=tmp_path / "output",
            scenario_mode="tuned",
        )
