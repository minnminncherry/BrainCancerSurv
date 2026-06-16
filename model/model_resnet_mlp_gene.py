import torch
import torch.nn as nn


class Dense(nn.Linear):
    """Dense/fully connected layer. In PyTorch, this is implemented by nn.Linear."""


class ResidualMLPBlock(nn.Module):
    """
    Residual block for tabular genomic features.
    The input and output dimensions are the same, so the skip connection is direct.
    """

    def __init__(self, hidden_dim, dropout=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(x + self.block(x))


class ResMLPGenomics(nn.Module):

    """
    Residual MLP model for genomic survival-bin prediction.
    Input: normalized gene expression tensor [batch, input_dim].
    Output: class logits [batch, n_classes].
    """

    def __init__(
        self,
        input_dim,
        n_classes=4,
        hidden_dim=256,
        num_blocks=5,
        dropout=0.2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.n_classes = n_classes

        self.residual_blocks = nn.Sequential(
            *[ResidualMLPBlock(hidden_dim, dropout=dropout) for _ in range(num_blocks)]
        )
        self.input_projection = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.to_logits = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, n_classes),
        )

        self._init_weights()

    def _init_weights(self):
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
            raise ValueError(f"Expected {self.input_dim} genomic features, got {data_omics.size(1)}.")
        return data_omics

    def forward(self, x=None, **kwargs):
        data_omics = self._get_omics_tensor(x=x, **kwargs)
        features = self.input_projection(data_omics)
        features = self.residual_blocks(features)
        logits = self.to_logits(features)
        return logits

    def captum(self, omics):
        logits = self.forward(omics)
        hazards = torch.sigmoid(logits)
        survival = torch.cumprod(1 - hazards, dim=1)
        risk = -torch.sum(survival, dim=1)
        return risk
 
