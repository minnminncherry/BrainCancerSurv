"""Small CPU checks for CI; no patient data, checkpoints, or database required."""
import pytest
import torch

from model.model_MIL_wsi import MILWSI
from model.late_fusion import MultimodalLateFusion
from model.cross_attn import MultimodalCrossAttention
from model.model_ViT import ViTMRI


def test_mil_ignores_padded_patches():
    torch.manual_seed(42)
    model = MILWSI(input_dim=8, hidden_dim=16, dropout=0).eval()
    patches = torch.randn(2, 3, 8)
    padded = torch.cat([patches, torch.zeros(2, 2, 8)], dim=1)
    with torch.no_grad():
        logits, attention = model(padded, return_attention=True)
        torch.testing.assert_close(logits, model(patches))
    assert logits.shape == (2, 4)
    assert (attention[:, 3:] == 0).all()


@pytest.mark.parametrize('model_class', [MultimodalLateFusion, MultimodalCrossAttention])
def test_fusion_backward(model_class):
    model = model_class(10, 8, 4, hidden_dim=16, num_blocks=1, dropout=0)
    genes = torch.randn(2, 10, requires_grad=True)
    patches = torch.randn(2, 3, 8, requires_grad=True)
    logits = model(genes, patches)
    assert logits.shape == (2, 4)
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 3])).backward()
    for data in (genes, patches):
        assert data.grad is not None
        assert torch.isfinite(data.grad).all()
        assert data.grad.abs().sum() > 0


def test_mri_transformer_volume_backward():
    model = ViTMRI(hidden_dim=16, num_layers=1, patch_size=8, dropout=0)
    volume = torch.randn(2, 1, 9, 10, 11, requires_grad=True)
    logits = model(volume)
    assert logits.shape == (2, 4)
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()
    assert volume.grad is not None and torch.isfinite(volume.grad).all()
    assert model.transformer.layers[0].self_attn.in_proj_weight.grad is not None
