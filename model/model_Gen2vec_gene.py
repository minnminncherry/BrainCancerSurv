import torch
import torch.nn as nn


class Gen2VecGenomics(nn.Module):
    """
    Hybrid Gen2Vec model for tabular genomic expression data.
    It combines attention-pooled gene embeddings with a direct MLP branch.
    """

    def __init__(
        self,
        input_dim,
        n_classes=4,
        embedding_dim=32,
        hidden_dim=256,
        dropout=0.2,
        use_zero_mask=False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.use_zero_mask = use_zero_mask

        self.gene_embeddings = nn.Parameter(torch.empty(input_dim, embedding_dim))
        self.value_embeddings = nn.Parameter(torch.empty(input_dim, embedding_dim))

        self.token_projection = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.mlp_branch = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.gene_embeddings)
        nn.init.xavier_uniform_(self.value_embeddings)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        x = x.float()
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.size(1) != self.input_dim:
            raise ValueError(f"Expected {self.input_dim} genomic features, got {x.size(1)}.")

        gene_tokens = (
            self.gene_embeddings.unsqueeze(0)
            + x.unsqueeze(-1) * self.value_embeddings.unsqueeze(0)
        )
        gene_features = self.token_projection(gene_tokens)
        attention_logits = self.attention(gene_features)
        if self.use_zero_mask:
            attention_logits = attention_logits.masked_fill(x.unsqueeze(-1) == 0, -1e9)
        attention_weights = torch.softmax(attention_logits, dim=1)
        gen2vec_embedding = torch.sum(attention_weights * gene_features, dim=1)
        mlp_embedding = self.mlp_branch(x)
        patient_embedding = torch.cat([gen2vec_embedding, mlp_embedding], dim=1)

        logits = self.classifier(patient_embedding)
        return logits

    def captum(self, omics):
        logits = self.forward(omics)
        hazards = torch.sigmoid(logits)
        survival = torch.cumprod(1 - hazards, dim=1)
        risk = -torch.sum(survival, dim=1)
        return risk
