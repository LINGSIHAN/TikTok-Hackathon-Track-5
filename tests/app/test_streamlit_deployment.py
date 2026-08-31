from __future__ import annotations

import hashlib
import json
from pathlib import Path

from app import streamlit_app


EXPECTED_V1_SHA256 = (
    "806fbabc5ecae8394369d08738cbf0c993568137d323a8133167e4557d04eed2"
)
EXPECTED_V2_SHA256 = (
    "b45022d9dab2a02300934c239eee24dd40ef8e402f24c1f27fc2d63a46117c12"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_streamlit_defaults_to_verified_v2_and_preserves_v1(
    monkeypatch,
) -> None:
    monkeypatch.delenv("AIGC_CHECKPOINT_PATH", raising=False)

    deployed = streamlit_app._checkpoint_path()
    v1 = streamlit_app.REPOSITORY_ROOT / "artifacts/checkpoints/model.safetensors"
    metadata_path = deployed.with_name("model_v2_metadata.json")

    assert deployed == (
        streamlit_app.REPOSITORY_ROOT
        / "artifacts/checkpoints/model_v2.safetensors"
    )
    assert deployed.is_file()
    assert v1.is_file()
    assert _sha256(deployed) == EXPECTED_V2_SHA256
    assert _sha256(v1) == EXPECTED_V1_SHA256

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["checkpoint_sha256"] == EXPECTED_V2_SHA256
    assert metadata["parent_checkpoint_sha256"] == EXPECTED_V1_SHA256


def test_streamlit_checkpoint_environment_override_is_preserved(
    monkeypatch, tmp_path: Path
) -> None:
    custom = tmp_path / "custom.safetensors"
    monkeypatch.setenv("AIGC_CHECKPOINT_PATH", str(custom))

    assert streamlit_app._checkpoint_path() == custom
