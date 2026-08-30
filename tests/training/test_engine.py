import math

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.training.engine import EarlyStopping, extract_logits, run_epoch, set_global_seed


def test_run_epoch_evaluates_dummy_binary_model():
    inputs = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
    labels = torch.tensor([0, 0, 1, 1])
    loader = DataLoader(TensorDataset(inputs, labels), batch_size=2)
    model = torch.nn.Linear(1, 1)
    with torch.no_grad():
        model.weight.fill_(2.0)
        model.bias.fill_(-3.0)

    result = run_epoch(model, loader, device=torch.device("cpu"))

    assert len(result.labels) == 4
    assert len(result.probabilities) == 4
    assert math.isfinite(result.loss)
    assert result.metrics["balanced_accuracy"] == pytest.approx(1.0)


def test_run_epoch_updates_weights_when_optimizer_is_supplied():
    set_global_seed(7)
    inputs = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
    labels = torch.tensor([0, 0, 1, 1])
    loader = DataLoader(TensorDataset(inputs, labels), batch_size=4)
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    before = model.weight.detach().clone()

    run_epoch(model, loader, device=torch.device("cpu"), optimizer=optimizer)

    assert not torch.equal(before, model.weight.detach())


def test_early_stopping_restores_best_weights():
    model = torch.nn.Linear(1, 1, bias=False)
    stopper = EarlyStopping(patience=2)
    with torch.no_grad():
        model.weight.fill_(1.0)
    assert stopper.update(0.4, model, epoch=1) is False
    with torch.no_grad():
        model.weight.fill_(9.0)
    assert stopper.update(0.5, model, epoch=2) is False
    assert stopper.update(0.6, model, epoch=3) is True

    stopper.restore(model)

    assert model.weight.item() == pytest.approx(1.0)
    assert stopper.best_epoch == 1


def test_extract_logits_validates_shape():
    assert extract_logits({"logits": torch.ones(2, 1)}).shape == (2,)
    with pytest.raises(ValueError, match="one logit"):
        extract_logits(torch.ones(2, 2))
