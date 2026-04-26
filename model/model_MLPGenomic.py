import torch.nn as nn

class MLPGenomics(nn.Module):
    def __init__(self, input_dim, output_dim, projection_dim=128, dropout=0.1):
        super(MLPGenomics, self).__init__()
        hidden_dim = max(32, projection_dim // 2)

        # Backbone that maps high-dimensional genomic vectors to a compact embedding.
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, projection_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Classification head: projection -> output logits (n_classes).
        self.to_logits = nn.Linear(projection_dim, output_dim)
    
    def forward(self, x, return_projection=False):
        projection = self.net(x)
        logits = self.to_logits(projection)
        if return_projection:
            return logits, projection
        return logits