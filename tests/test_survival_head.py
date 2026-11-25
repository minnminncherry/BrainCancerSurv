import torch

from model.out_survival_head import SurvivalPredictionHead
from model.gene_resnet_mlp_model import ResMLPGenomics
from model.out_mutation_head import MutationPredictionHead
from utils.loss_utils import LossClass


def test_head_preserves_linear_weights_and_output():
    old = torch.nn.Linear(8, 4)
    head = SurvivalPredictionHead(8, 4)
    head.load_state_dict(old.state_dict(), strict=True)
    features = torch.randn(3, 8)
    torch.testing.assert_close(head(features), old(features))
    assert set(head.state_dict()) == {'weight', 'bias'}


def test_resmlp_features_feed_both_heads():
    encoder = ResMLPGenomics(10, hidden_dim=16, num_blocks=1, dropout=0)
    survival_head = SurvivalPredictionHead(16, 4)
    mutation_head = MutationPredictionHead(16, 3, hidden_dim=8, dropout=0)
    features = encoder.forward_features(torch.randn(2, 10))
    survival_logits = survival_head(features)
    mutation_logits = mutation_head(features)
    assert features.shape == (2, 16)
    assert survival_logits.shape == (2, 4)
    assert mutation_logits.shape == (2, 3)
    # The existing standalone ResMLP forward remains supported.
    assert encoder(torch.randn(2, 10)).shape == (2, 4)
    loss = LossClass()(survival_logits, torch.tensor([0, 2]), torch.tensor([0., 1.]))
    loss += torch.nn.functional.binary_cross_entropy_with_logits(mutation_logits, torch.zeros(2, 3))
    loss.backward()
    assert encoder.input_projection[1].weight.grad.abs().sum() > 0
