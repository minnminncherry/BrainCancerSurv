"""Late fusion of gene expression and 3D MRI embeddings."""

import torch
import torch.nn as nn

from model.gene_resnet_mlp_model import ResMLPGenomics
from model.mri_ViT_model import ViTMRI
from model.out_survival_head import SurvivalPredictionHead


class MultimodalMRILateFusion(nn.Module):
    """Inputs: genes [B, G], MRI [B, 1, D, H, W]. Output: logits [B, C]."""

    def __init__(self, genomic_input_dim, n_classes=4, hidden_dim=128,
                 dropout=0.3, num_blocks=2):
        super().__init__()
        self.genomic_encoder = ResMLPGenomics(
            genomic_input_dim, hidden_dim=hidden_dim,
            dropout=dropout, num_blocks=num_blocks,
        )
        self.genomic_encoder.to_logits = nn.Identity()
        self.mri_encoder = ViTMRI(hidden_dim=128, dropout=dropout)
        self.mri_encoder.classifier = nn.Identity()
        self.mri_projection = nn.Linear(128, hidden_dim)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
        )
        self.classifier = SurvivalPredictionHead(hidden_dim, n_classes)

    def forward_features(self, genomic_data, mri_data):
        genes = self.genomic_encoder.forward_features(genomic_data)
        mri = self.mri_projection(self.mri_encoder.forward_features(mri_data))
        if genes.size(0) != mri.size(0):
            raise ValueError("Genomic and MRI inputs must have the same batch size.")
        return self.fusion(torch.cat([genes, mri], dim=-1))

    def forward(self, genomic_data, mri_data):
        return self.classifier(self.forward_features(genomic_data, mri_data))
