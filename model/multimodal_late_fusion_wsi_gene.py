import torch
import torch.nn as nn
from model.out_survival_head import SurvivalPredictionHead
from model.gene_resnet_mlp_model import ResMLPGenomics
from model.wsi_MIL_model import MILWSI

class MultimodalLateFusion(nn.Module):
    """ResMLP + attention MIL embeddings, fused into interval hazard/class logits."""
    def __init__(self, genomic_input_dim, wsi_input_dim, n_classes, hidden_dim=256, dropout=0.2, num_blocks=5):
        super(MultimodalLateFusion, self).__init__()
        self.genomic_fc = ResMLPGenomics(
            input_dim=genomic_input_dim,
            n_classes=hidden_dim,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            dropout=dropout,
        )
        self.genomic_fc.to_logits = nn.LayerNorm(hidden_dim)
        self.wsi_fc = MILWSI(
            input_dim=wsi_input_dim,
            n_classes=n_classes,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        # Both encoders train jointly; the unused MIL classifier is removed.
        self.wsi_fc.classifier = nn.Identity()
        self.wsi_norm = nn.LayerNorm(hidden_dim)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = SurvivalPredictionHead(hidden_dim, n_classes)

    def forward(self, genomic_data, wsi_data):
        genomic_features = self.genomic_fc(genomic_data)
        wsi_features = self.wsi_norm(self.wsi_fc(wsi_data, return_features=True))
        combined_features = torch.cat((genomic_features, wsi_features), dim=1)
        fused_embedding = self.fusion(combined_features)
        output = self.classifier(fused_embedding)
        return output
