"""EfficientNet-B0 binary classifier used by the training and inference code."""

from __future__ import annotations

from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


def build_model(
    pretrained: bool = False,
    freeze_backbone: bool = False,
    unfreeze_last_blocks: int = 0,
) -> nn.Module:
    """Build an EfficientNet-B0 that emits one AIGC logit per image.

    Args:
        pretrained: Load ImageNet-1K weights when true. This can trigger a
            one-time torchvision download if the weights are not cached.
        freeze_backbone: Freeze all feature-extractor parameters. The binary
            classifier head always remains trainable.
        unfreeze_last_blocks: If the backbone is frozen, make this many of the
            final top-level EfficientNet feature blocks trainable again.

    Returns:
        A torchvision EfficientNet whose output has shape ``[batch, 1]``.
    """

    if isinstance(unfreeze_last_blocks, bool) or not isinstance(
        unfreeze_last_blocks, int
    ):
        raise TypeError("unfreeze_last_blocks must be an integer")
    if unfreeze_last_blocks < 0:
        raise ValueError("unfreeze_last_blocks must be non-negative")

    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = efficientnet_b0(weights=weights)

    final_layer = model.classifier[-1]
    if not isinstance(final_layer, nn.Linear):
        raise RuntimeError("unexpected EfficientNet-B0 classifier structure")
    model.classifier[-1] = nn.Linear(final_layer.in_features, 1)

    feature_blocks = list(model.features.children())
    if unfreeze_last_blocks > len(feature_blocks):
        raise ValueError(
            "unfreeze_last_blocks cannot exceed the number of feature blocks "
            f"({len(feature_blocks)})"
        )

    if freeze_backbone:
        for parameter in model.features.parameters():
            parameter.requires_grad = False
        if unfreeze_last_blocks:
            for block in feature_blocks[-unfreeze_last_blocks:]:
                for parameter in block.parameters():
                    parameter.requires_grad = True

    # Be explicit so rebuilding a head never inherits a frozen state.
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True

    return model


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """Return the number of scalar parameters in ``model``.

    Set ``trainable_only`` to count only parameters whose ``requires_grad``
    flag is true.
    """

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if not trainable_only or parameter.requires_grad
    )
