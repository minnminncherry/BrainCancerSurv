"""Use a shared genomic query to attend separately to WSI and MRI."""
import torch
import torch.nn as nn

from model.multimodal_late_fusion_wsi_mri_gene import MultimodalWSIMRILateFusion


class MultimodalWSIMRICrossAttention(MultimodalWSIMRILateFusion):
    def __init__(self, genomic_input_dim, wsi_input_dim=1024, n_classes=4,
                 hidden_dim=32, dropout=0.3, num_blocks=1, num_heads=4):
        if num_heads < 1 or hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by a positive num_heads.")
        super().__init__(genomic_input_dim, wsi_input_dim, n_classes,
                         hidden_dim, dropout, num_blocks)
        self.wsi_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout, batch_first=True)
        self.mri_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout, batch_first=True)

    def _attend(self, genomic_data, wsi_data, mri_data, return_attention=False):
        genes = self.genomic_encoder.forward_features(genomic_data)
        wsi = self.wsi_encoder(wsi_data, return_tokens=True, mask_padding=False)
        mri = self.mri_projection(self.mri_encoder.forward_tokens(mri_data))
        if not genes.size(0) == wsi.size(0) == mri.size(0):
            raise ValueError("All three modalities must have the same batch size.")
        query = genes.unsqueeze(1)
        wsi_output, wsi_weights = self.wsi_attention(query, wsi, wsi, need_weights=return_attention)
        mri_output, mri_weights = self.mri_attention(query, mri, mri, need_weights=return_attention)
        fused = self.fusion(torch.cat([genes, wsi_output.squeeze(1), mri_output.squeeze(1)], dim=-1))
        return fused, wsi_weights, mri_weights

    def forward_features(self, genomic_data, wsi_data, mri_data):
        return self._attend(genomic_data, wsi_data, mri_data)[0]

    def forward(self, genomic_data, wsi_data, mri_data, return_attention=False):
        fused, wsi_weights, mri_weights = self._attend(
            genomic_data, wsi_data, mri_data, return_attention,
        )
        logits = self.classifier(fused)
        if return_attention:
            p = self.mri_encoder.patch_size
            grid = tuple((size + p - 1) // p for size in mri_data.shape[-3:])
            return logits, {"wsi": wsi_weights.squeeze(1),
                            "mri": mri_weights.squeeze(1).reshape(mri_data.size(0), *grid)}
        return logits
