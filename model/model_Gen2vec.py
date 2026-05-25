import torch
import torch.nn as nn


class Gen2VecGenomics(nn.Module):
    """
    Gen2Vec-style model for tabular genomic expression data.

    Each gene has a trainable embedding vector. The expression value for that
    gene scales its embedding, then attention pooling combines all gene vectors
    into one patient-level representation for classification.
    """

    def __init__(
        self,
        input_dim,
        n_classes=4,
        embedding_dim=64,
        hidden_dim=128,
        dropout=0.25,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.n_classes = n_classes

        self.gene_embeddings = nn.Parameter(torch.empty(input_dim, embedding_dim))
        self.value_projection = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.gene_embeddings)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def _get_omics_tensor(self, x=None, **kwargs):
        if x is not None:
            data_omics = x
        elif "data_omics" in kwargs:
            data_omics = kwargs["data_omics"]
        else:
            raise ValueError("Expected an input tensor or keyword argument `data_omics`.")

        if not torch.is_tensor(data_omics):
            raise TypeError("Omics input must be a torch.Tensor.")

        data_omics = data_omics.float()
        if data_omics.dim() == 1:
            data_omics = data_omics.unsqueeze(0)
        if data_omics.size(1) != self.input_dim:
            raise ValueError(
                f"Expected {self.input_dim} genomic features, got {data_omics.size(1)}."
            )
        return data_omics

    def forward(self, x=None, **kwargs):
        data_omics = self._get_omics_tensor(x=x, **kwargs)

        gene_tokens = data_omics.unsqueeze(-1) * self.gene_embeddings.unsqueeze(0)
        gene_features = self.value_projection(gene_tokens)
        attention_logits = self.attention(gene_features)
        attention_weights = torch.softmax(attention_logits, dim=1)
        patient_embedding = torch.sum(attention_weights * gene_features, dim=1)

        return self.classifier(patient_embedding)

    def captum(self, omics):
        logits = self.forward(omics)
        hazards = torch.sigmoid(logits)
        survival = torch.cumprod(1 - hazards, dim=1)
        risk = -torch.sum(survival, dim=1)
        return risk
