"""ResMLP-guided MRI/WSI branches with mutation-informed survival prediction."""
import torch
import torch.nn as nn

from model.gene_resnet_mlp_model import ResMLPGenomics
from model.mri_ViT_model import ViTMRI
from model.wsi_MIL_model import MILWSI
from model.out_mutation_head import MutationPredictionHead
from model.out_survival_head import SurvivalPredictionHead


class MutationGuidedSurvival(nn.Module):
    """Return separate mutation logits, [B, embedding_dim] embeddings, and survival logits."""

    def __init__(self, genomic_input_dim, n_targets, wsi_input_dim=1024,
                 n_classes=4, embedding_dim=64, genomic_hidden_dim=64,
                 num_blocks=1, dropout=0.3, fusion="cross_attention",
                 num_heads=4, mri_hidden_dim=128, mri_num_layers=2, mri_patch_size=32):
        super().__init__()
        if embedding_dim < 1 or n_targets < 1:
            raise ValueError("Embedding size and mutation target count must be positive.")
        if fusion not in {"late_fusion", "cross_attention"}:
            raise ValueError("Choose late_fusion or cross_attention.")
        if num_heads < 1 or embedding_dim % num_heads:
            raise ValueError("embedding_dim must be divisible by num_heads.")
        self.embedding_dim, self.fusion_type = embedding_dim, fusion
        self.genomic_encoder = ResMLPGenomics(
            genomic_input_dim, hidden_dim=genomic_hidden_dim,
            num_blocks=num_blocks, dropout=dropout,
        )
        self.genomic_encoder.to_logits = nn.Identity()
        self.genomic_projection = nn.Linear(genomic_hidden_dim, embedding_dim)
        self.mri_encoder = ViTMRI(
            hidden_dim=mri_hidden_dim, num_layers=mri_num_layers,
            patch_size=mri_patch_size, num_heads=num_heads, dropout=dropout,
        )
        self.mri_encoder.classifier = nn.Identity()
        self.mri_projection = nn.Linear(mri_hidden_dim, embedding_dim)
        self.wsi_encoder = MILWSI(wsi_input_dim, n_classes, embedding_dim, dropout)
        self.wsi_encoder.classifier = nn.Identity()
        if fusion == "cross_attention":
            self.mri_attention = nn.MultiheadAttention(embedding_dim, num_heads, dropout, batch_first=True)
            self.wsi_attention = nn.MultiheadAttention(embedding_dim, num_heads, dropout, batch_first=True)
        self.mri_fusion = self._embedding_head(embedding_dim * 2, embedding_dim, dropout)
        self.wsi_fusion = self._embedding_head(embedding_dim * 2, embedding_dim, dropout)
        self.mri_embedding_head = self._embedding_head(embedding_dim, embedding_dim, dropout)
        self.wsi_embedding_head = self._embedding_head(embedding_dim, embedding_dim, dropout)
        self.mri_mutation_head = MutationPredictionHead(embedding_dim, n_targets, embedding_dim, dropout)
        self.wsi_mutation_head = MutationPredictionHead(embedding_dim, n_targets, embedding_dim, dropout)
        self.survival_head = nn.Sequential(
            nn.Linear(2 * embedding_dim + 2 * n_targets, embedding_dim),
            nn.ReLU(), nn.Dropout(dropout), SurvivalPredictionHead(embedding_dim, n_classes),
        )

    @staticmethod
    def _embedding_head(input_dim, output_dim, dropout):
        return nn.Sequential(nn.Linear(input_dim, output_dim), nn.ReLU(), nn.Dropout(dropout))

    def forward(self, genomic_data, wsi_data, mri_data):
        genes = self.genomic_projection(self.genomic_encoder.forward_features(genomic_data))
        if not genes.size(0) == wsi_data.size(0) == mri_data.size(0):
            raise ValueError("All modalities must have the same patient batch size.")
        if self.fusion_type == "cross_attention":
            wsi_tokens = self.wsi_encoder(wsi_data, return_tokens=True, mask_padding=False)
            mri_tokens = self.mri_projection(self.mri_encoder.forward_tokens(mri_data))
            query = genes.unsqueeze(1)
            wsi = self.wsi_attention(query, wsi_tokens, wsi_tokens, need_weights=False)[0].squeeze(1)
            mri = self.mri_attention(query, mri_tokens, mri_tokens, need_weights=False)[0].squeeze(1)
        else:
            wsi = self.wsi_encoder(wsi_data, return_features=True)
            mri = self.mri_projection(self.mri_encoder.forward_features(mri_data))
        mri_features = self.mri_fusion(torch.cat([genes, mri], dim=-1))
        wsi_features = self.wsi_fusion(torch.cat([genes, wsi], dim=-1))
        mri_embedding = self.mri_embedding_head(mri_features)
        wsi_embedding = self.wsi_embedding_head(wsi_features)
        mri_logits = self.mri_mutation_head(mri_features)
        wsi_logits = self.wsi_mutation_head(wsi_features)
        # Predictions, never the true mutation labels, inform the survival head.
        survival_features = torch.cat([
            mri_embedding, wsi_embedding, mri_logits.sigmoid(), wsi_logits.sigmoid(),
        ], dim=-1)
        return {
            "survival_logits": self.survival_head(survival_features),
            "mri_mutation_logits": mri_logits, "wsi_mutation_logits": wsi_logits,
            "mri_embedding": mri_embedding, "wsi_embedding": wsi_embedding,
        }
