"""Small 3D CNN for MRI survival prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.out_survival_head import SurvivalPredictionHead


class MRIConv3d(nn.Conv3d):
    """Use native PyTorch convolution to bypass failing cuDNN engines."""

    def forward(self, x):
        with torch.backends.cudnn.flags(enabled=False):
            return super().forward(x.contiguous())


class CNN3DMRI(nn.Module):
    """Encode MRI volumes [B, 1, D, H, W] and return survival logits."""

    def __init__(self, n_classes=4, hidden_dim=128, dropout=0.3, n_mutation_targets=0):
        super().__init__()
        self.n_classes = n_classes
        layers = []
        in_channels = 1
        for channels in (16, 32, 64):
            layers.extend([
                MRIConv3d(in_channels, channels, kernel_size=3,
                          stride=2, padding=1, bias=False),
                nn.GroupNorm(4, channels),
                nn.ReLU(inplace=True),
            ])
            in_channels = channels

        self.encoder = nn.Sequential(*layers, nn.AdaptiveAvgPool3d(1), nn.Flatten(1))
        self.embedding = nn.Sequential(nn.Linear(64, hidden_dim), nn.ReLU())
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), SurvivalPredictionHead(hidden_dim, n_classes),
        )
        self.mutation_head = nn.Linear(hidden_dim, n_mutation_targets) if n_mutation_targets else None

    def forward_features(self, x):
        """Return a patient embedding [B, hidden_dim]."""
        if x.ndim != 5 or x.shape[1] != 1 or min(x.shape[2:]) < 1:
            raise ValueError("Expected MRI volumes [batch, 1, depth, height, width].")
        return self.embedding(self.encoder(x.float()))

    def forward(self, x):
        # Keep raw logits; the survival loss converts them into hazards.
        features = self.forward_features(x)
        logits = self.classifier(features)
        if self.mutation_head is not None:
            return {"survival_logits": logits,
                    "mri_mutation_logits": self.mutation_head(features),
                    "mri_embedding": features}
        return logits

    def forward_tokens(self, x):
        """Return 64 spatial MRI tokens [B, 64, hidden_dim] for attention."""
        if x.ndim != 5 or x.shape[1] != 1 or min(x.shape[2:]) < 1:
            raise ValueError("Expected MRI volumes [batch, 1, depth, height, width].")
        features = self.encoder[:-2](x.float())
        tokens = F.adaptive_avg_pool3d(features, (4, 4, 4)).flatten(2).transpose(1, 2)
        return self.embedding(tokens)
