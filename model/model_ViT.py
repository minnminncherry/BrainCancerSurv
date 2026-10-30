import torch
import torch.nn as nn
import torch.nn.functional as F


class ViTMRI(nn.Module):
    """3D patch embeddings, spatial positions, and Transformer survival classification."""

    def __init__(self, input_dim=1024, n_classes=4, hidden_dim=128,
                 dropout=0.3, patch_size=32, num_layers=2, num_heads=4):
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        if patch_size < 1 or num_layers < 1:
            raise ValueError("patch_size and num_layers must be positive.")
        self.input_dim = int(input_dim)
        self.n_classes = int(n_classes)
        self.patch_size = int(patch_size)
        self.volume_projection = nn.Linear(self.patch_size ** 3, hidden_dim)
        self.feature_projection = nn.Linear(self.input_dim, hidden_dim)
        self.position_embedding = nn.Parameter(torch.empty(1, hidden_dim, 7, 7, 7))
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        self.embedding_dropout = nn.Dropout(dropout)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=num_heads,
                dim_feedforward=hidden_dim * 4, dropout=dropout,
                activation="gelu", batch_first=True, norm_first=True,
            ),
            num_layers=num_layers, norm=nn.LayerNorm(hidden_dim),
            enable_nested_tensor=False,
        )
        # Initialize cloned encoder layers independently.
        for layer in self.transformer.layers:
            nn.init.xavier_uniform_(layer.self_attn.in_proj_weight)
            nn.init.xavier_uniform_(layer.self_attn.out_proj.weight)
            nn.init.xavier_uniform_(layer.linear1.weight)
            nn.init.xavier_uniform_(layer.linear2.weight)
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, self.n_classes))

    def forward(self, x, return_attention=False):
        """Return logits; optional attention contains final pooling weights, not self-attention."""
        if not torch.is_tensor(x):
            raise TypeError("ViTMRI input must be a torch.Tensor.")
        x = x.float()
        if x.dim() == 5:
            if x.size(1) != 1 or min(x.shape[-3:]) < 1:
                raise ValueError("Expected nonempty MRI volumes [batch, 1, depth, height, width].")
            p = self.patch_size
            padding = [(-size) % p for size in x.shape[-3:]]
            x = F.pad(x, (0, padding[2], 0, padding[1], 0, padding[0]))
            grid = tuple(size // p for size in x.shape[-3:])
            valid = F.max_pool3d((x != 0).float(), p, p).flatten(1).bool()
            # Linear patch projection avoids cuDNN convolution engine dependence.
            patches = x.unfold(2, p, p).unfold(3, p, p).unfold(4, p, p)
            patches = patches.permute(0, 2, 3, 4, 1, 5, 6, 7).reshape(x.size(0), -1, p ** 3)
            tokens = self.volume_projection(patches)
            positions = F.interpolate(
                self.position_embedding, size=grid, mode="trilinear", align_corners=False
            ).flatten(2).transpose(1, 2)
        else:
            if x.dim() == 2:
                x = x.unsqueeze(0)
            if x.dim() != 3 or x.size(-1) != self.input_dim or x.size(1) == 0:
                raise ValueError(
                    "Expected MRI volumes [batch, 1, D, H, W] or "
                    f"patch features [batch, num_patches, {self.input_dim}]."
                )
            valid = x.abs().sum(dim=-1) > 0
            tokens = self.feature_projection(x)
            positions = F.interpolate(
                self.position_embedding.flatten(2), size=x.size(1),
                mode="linear", align_corners=False,
            ).transpose(1, 2)

        # Ensure attention has an unmasked key even for an entirely blank sample.
        valid = valid.clone()
        valid[~valid.any(dim=1), 0] = True
        tokens = self.embedding_dropout(tokens + positions)
        tokens = self.transformer(tokens, src_key_padding_mask=~valid)
        scores = self.attention(tokens).squeeze(-1).masked_fill(~valid, float("-inf"))
        weights = scores.softmax(dim=1)
        features = (tokens * weights.unsqueeze(-1)).sum(dim=1)
        logits = self.classifier(features)
        return (logits, weights) if return_attention else logits
