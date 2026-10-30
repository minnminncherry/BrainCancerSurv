import torch
import torch.nn as nn
from model.model_resnet_mlp_gene import ResMLPGenomics
from model.model_MIL_wsi import MILWSI

class MultimodalLateFusion(nn.Module):
    def __init__(self, genomic_input_dim, wsi_input_dim, n_classes, hidden_dim=256, dropout=0.2, num_blocks=5):
        super(MultimodalLateFusion, self).__init__()
        self.genomic_fc = ResMLPGenomics(
            input_dim=genomic_input_dim,
            n_classes=hidden_dim,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            dropout=dropout,
        )
        self.wsi_fc = MILWSI(
            input_dim=wsi_input_dim,
            n_classes=n_classes,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        # Fusion uses MIL embeddings; only the fused classifier is trained.
        self.wsi_fc.classifier = nn.Identity()
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_dim, n_classes)

    def forward(self, genomic_data, wsi_data):
        genomic_features = self.genomic_fc(genomic_data)
        wsi_features = self.wsi_fc(wsi_data, return_features=True)
        combined_features = torch.cat((genomic_features, wsi_features), dim=1)
        fused_embedding = self.fusion(combined_features)
        output = self.classifier(fused_embedding)
        return output
