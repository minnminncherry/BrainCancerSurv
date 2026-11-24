import torch
import torch.nn as nn
import torch.nn.functional as F

class TRANSMILWSI(nn.Module):

    def __init__(self, input_dim=512, n_classes=4, hidden_dim=256, dropout=0.25):
        super().__init__()
        self.input_dim = int(input_dim)
        self.n_classes = int(n_classes)

        self.patch_encoder = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.classifier = nn.Linear(hidden_dim, self.n_classes)

    def forward(self, x, return_attention=False):
        if not torch.is_tensor(x):
            raise TypeError("TRANSMILWSI input must be a torch.Tensor.")

        x = x.float()
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if x.dim() != 3:
            raise ValueError("Expected input shape [num_patches, input_dim] or [batch, num_patches, input_dim].")

        valid_patches = x.abs().sum(dim=-1) > 0
        patch_features = self.patch_encoder(x)
        attention_scores = self.attention(patch_features).squeeze(-1)
        attention_scores = attention_scores.masked_fill(~valid_patches, -1e9)
        attention_weights = F.softmax(attention_scores, dim=1)

        bag_features = torch.sum(patch_features * attention_weights.unsqueeze(-1), dim=1)
        logits = self.classifier(bag_features)

        if return_attention:
            return logits, attention_weights
        return logits