import math

import pytest
import torch

from model.multimodal_late_fusion_wsi_gene import MultimodalLateFusion
from utils.loss_utils import LossClass, survival_outputs


def test_event_and_censor_likelihood():
    # h=(.2,.5): event at interval 1 -> .8*.5; censored at interval 1 -> .8*.5.
    logits = torch.logit(torch.tensor([[0.2, 0.5], [0.2, 0.5]]))
    loss = LossClass()(logits, torch.tensor([1, 1]), torch.tensor([0., 1.]))
    torch.testing.assert_close(loss, torch.tensor(-math.log(0.4)))
    first_event = LossClass()(logits[:1], torch.tensor([0]), torch.tensor([0.]))
    torch.testing.assert_close(first_event, torch.tensor(-math.log(0.2)))


def test_censoring_changes_likelihood():
    logits = torch.logit(torch.tensor([[0.2, 0.1]]))
    labels = torch.tensor([1])
    event = LossClass()(logits, labels, torch.tensor([0.]))
    censored = LossClass()(logits, labels, torch.tensor([1.]))
    torch.testing.assert_close(event, torch.tensor(-math.log(0.08)))
    torch.testing.assert_close(censored, torch.tensor(-math.log(0.72)))


def test_extreme_logits_stay_finite_and_backpropagate():
    logits = torch.tensor([[1000., -1000.], [-1000., 1000.]], requires_grad=True)
    loss = LossClass()(logits, torch.tensor([1, 0]), torch.tensor([0., 1.]))
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()


def test_survival_outputs_conserve_probability():
    hazards, survival, mass = survival_outputs(torch.randn(4, 5))
    assert (survival[:, 1:] <= survival[:, :-1]).all()
    torch.testing.assert_close(mass.sum(1) + survival[:, -1], torch.ones(4))


def test_late_fusion_survival_backward():
    model = MultimodalLateFusion(10, 8, 4, hidden_dim=16, num_blocks=2)
    genes = torch.randn(2, 10, requires_grad=True)
    patches = torch.randn(2, 5, 8, requires_grad=True)
    loss = LossClass()(model(genes, patches), torch.tensor([0, 3]), torch.tensor([0., 1.]))
    loss.backward()
    assert torch.isfinite(loss)
    assert genes.grad.abs().sum() > 0 and patches.grad.abs().sum() > 0


def test_invalid_censoring_rejected():
    with pytest.raises(ValueError, match="Censoring"):
        LossClass()(torch.zeros(1, 3), torch.tensor([1]), torch.tensor([float('nan')]))
