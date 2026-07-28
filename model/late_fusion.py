import torch
import torch.nn as nn
from model.model_resnet_mlp_gene import ResMLPGenomics

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
        self.wsi_fc = nn.Sequential(
            nn.Linear(wsi_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_dim * 2, n_classes)

    def forward(self, genomic_data, wsi_data):
        if wsi_data.dim() == 3:
            wsi_data = wsi_data.mean(dim=1)
        genomic_features = self.genomic_fc(genomic_data)
        wsi_features = self.wsi_fc(wsi_data)
        combined_features = torch.cat((genomic_features, wsi_features), dim=1)
        output = self.classifier(combined_features)
        return output
