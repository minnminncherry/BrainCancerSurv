import torch
import torch.nn as nn

from model.model_resnet_mlp_gene import ResMLPGenomics
from model.model_MIL_wsi import MILWSI


class MultimodalCrossAttention(nn.Module):
    """Use a ResMLP genomic query to attend to MIL-weighted WSI patch tokens.

    Inputs: genomic_data [B, genes], wsi_data [B, patches, features].
    Output: survival-class logits [B, n_classes]. All WSI rows attend, including zeros.
    """

    def __init__(self, genomic_input_dim, wsi_input_dim, n_classes,
                 hidden_dim=256, dropout=0.2, num_blocks=5, num_heads=4):
        super().__init__()
        if num_heads < 1 or hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by a positive num_heads.")
        self.wsi_input_dim = wsi_input_dim
        self.genomic_encoder = ResMLPGenomics(
            input_dim=genomic_input_dim, n_classes=hidden_dim,
            hidden_dim=hidden_dim, num_blocks=num_blocks, dropout=dropout,
        )
        self.wsi_encoder = MILWSI(
            input_dim=wsi_input_dim, n_classes=n_classes,
            hidden_dim=hidden_dim, dropout=dropout,
        )
        self.wsi_encoder.classifier = nn.Identity()
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True,
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2), nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, n_classes),
        )

    def forward(self, genomic_data, wsi_data, return_attention=False):
        if genomic_data.dim() == 1:
            genomic_data = genomic_data.unsqueeze(0)
        if wsi_data.dim() == 2:
            wsi_data = wsi_data.unsqueeze(0)
        if wsi_data.dim() != 3 or wsi_data.size(-1) != self.wsi_input_dim:
            raise ValueError("Expected WSI features [batch, patches, wsi_input_dim].")
        if wsi_data.size(0) != genomic_data.size(0):
            raise ValueError("Genomic and WSI inputs must have the same batch size.")
        if wsi_data.size(1) == 0:
            raise ValueError("Each patient needs at least one WSI patch.")

        genomic_features = self.genomic_encoder(genomic_data)
        patch_features = self.wsi_encoder(
            wsi_data, return_tokens=True, mask_padding=False,
        )
        query = genomic_features.unsqueeze(1)
        attended, weights = self.cross_attention(
            query=query, key=patch_features, value=patch_features,
            need_weights=return_attention,
        )
        fused = self.feed_forward(attended)
        combined = torch.cat([genomic_features, fused.squeeze(1)], dim=-1)
        logits = self.classifier(combined)
        if return_attention:
            return logits, weights.squeeze(1)  # Head-averaged patch weights [B, patches].
        return logits
