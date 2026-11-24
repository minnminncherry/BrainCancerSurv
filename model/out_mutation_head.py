"""Multi-label mutation prediction from expression or learned genomic embeddings."""
import torch.nn as nn


class MutationPredictionHead(nn.Module):
    """Return [B, n_targets] logits for BCEWithLogitsLoss; apply sigmoid at inference."""

    def __init__(self, input_dim, n_targets, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_targets),
        )

    def forward(self, features):
        return self.network(features.float())
