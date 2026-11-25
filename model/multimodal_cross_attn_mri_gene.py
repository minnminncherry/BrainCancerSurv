"""Genomic-guided cross-attention over spatial 3D MRI tokens."""

import torch
import torch.nn as nn

from model.gene_resnet_mlp_model import ResMLPGenomics
from model.mri_ViT_model import ViTMRI
from model.out_survival_head import SurvivalPredictionHead


class MultimodalMRICrossAttention(nn.Module):
    """Inputs: genes [B, G], MRI [B, 1, D, H, W]. Output: logits [B, C]."""

    def __init__(self, genomic_input_dim, n_classes=4, hidden_dim=128,
                 dropout=0.3, num_blocks=2, num_heads=4):
        super().__init__()
        if num_heads < 1 or hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by a positive num_heads.")
        self.genomic_encoder = ResMLPGenomics(
            genomic_input_dim, hidden_dim=hidden_dim,
            dropout=dropout, num_blocks=num_blocks,
        )
        self.genomic_encoder.to_logits = nn.Identity()
        self.mri_encoder = ViTMRI(hidden_dim=128, dropout=dropout)
        self.mri_encoder.classifier = nn.Identity()
        self.mri_projection = nn.Linear(128, hidden_dim)
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True,
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
        )
        self.classifier = SurvivalPredictionHead(hidden_dim, n_classes)

    def forward(self, genomic_data, mri_data, return_attention=False):
        genes = self.genomic_encoder.forward_features(genomic_data)
        tokens = self.mri_projection(self.mri_encoder.forward_tokens(mri_data))
        if genes.size(0) != tokens.size(0):
            raise ValueError("Genomic and MRI inputs must have the same batch size.")
        attended, weights = self.cross_attention(
            query=genes.unsqueeze(1), key=tokens, value=tokens,
            need_weights=return_attention,
        )
        fused = self.fusion(torch.cat([genes, attended.squeeze(1)], dim=-1))
        logits = self.classifier(fused)
        if return_attention:
            p = self.mri_encoder.patch_size
            grid = tuple((size + p - 1) // p for size in mri_data.shape[-3:])
            return logits, weights.squeeze(1).reshape(mri_data.size(0), *grid)
        return logits
