from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from src.models.efficientnet import build_model, count_parameters


def test_build_model_emits_one_logit_without_downloading_weights() -> None:
    model = build_model(pretrained=False)

    with torch.inference_mode():
        output = model(torch.zeros(2, 3, 224, 224))

    assert output.shape == (2, 1)
    assert count_parameters(model) < 2_000_000_000


def test_freeze_backbone_keeps_classifier_trainable() -> None:
    model = build_model(pretrained=False, freeze_backbone=True)

    assert not any(parameter.requires_grad for parameter in model.features.parameters())
    assert all(parameter.requires_grad for parameter in model.classifier.parameters())
    assert count_parameters(model, trainable_only=True) < count_parameters(model)


def test_unfreeze_last_feature_block() -> None:
    model = build_model(
        pretrained=False, freeze_backbone=True, unfreeze_last_blocks=1
    )
    feature_blocks = list(model.features.children())

    assert any(parameter.requires_grad for parameter in feature_blocks[-1].parameters())
    assert not any(
        parameter.requires_grad
        for block in feature_blocks[:-1]
        for parameter in block.parameters()
    )


@pytest.mark.parametrize("value", [-1, 10_000])
def test_invalid_unfreeze_block_count(value: int) -> None:
    with pytest.raises(ValueError):
        build_model(pretrained=False, unfreeze_last_blocks=value)
