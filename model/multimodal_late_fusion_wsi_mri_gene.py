"""Joint genomic, WSI, and MRI survival prediction."""
import torch
import torch.nn as nn

from model.multimodal_late_fusion_mri_gene import MultimodalMRILateFusion
from model.wsi_MIL_model import MILWSI


class MultimodalWSIMRILateFusion(MultimodalMRILateFusion):
    """ResMLP + WSI MIL + MRI ViT -> concatenate -> survival logits."""

    def __init__(self, genomic_input_dim, wsi_input_dim=1024, n_classes=4,
                 hidden_dim=32, dropout=0.3, num_blocks=1):
        super().__init__(genomic_input_dim, n_classes, hidden_dim, dropout, num_blocks)
        self.wsi_encoder = MILWSI(wsi_input_dim, n_classes, hidden_dim, dropout)
        self.wsi_encoder.classifier = nn.Identity()
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
        )

    def forward_features(self, genomic_data, wsi_data, mri_data):
        genes = self.genomic_encoder.forward_features(genomic_data)
        wsi = self.wsi_encoder(wsi_data, return_features=True)
        mri = self.mri_projection(self.mri_encoder.forward_features(mri_data))
        if not genes.size(0) == wsi.size(0) == mri.size(0):
            raise ValueError("All three modalities must have the same batch size.")
        return self.fusion(torch.cat([genes, wsi, mri], dim=-1))

    def forward(self, genomic_data, wsi_data, mri_data):
        return self.classifier(self.forward_features(genomic_data, wsi_data, mri_data))
