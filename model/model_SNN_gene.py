import math

import torch
import torch.nn as nn


class SNNBlock(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.25):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.SELU(inplace=True),
            nn.AlphaDropout(dropout),
        )

    def forward(self, x):
        return self.block(x)


class SNNGenomics(nn.Module):
    """
    Self-normalizing neural network for tabular genomic features.
    It returns class logits with the same shape as MLPGenomics: [batch, n_classes].
    """

    def __init__(
        self,
        input_dim,
        n_classes=4,
        hidden_dim=256,
        dropout=0.25,
    ):
        super().__init__()
        self.n_classes = n_classes

        self.net = nn.Sequential(
            SNNBlock(input_dim, hidden_dim, dropout=dropout),
            SNNBlock(hidden_dim, hidden_dim, dropout=dropout),
        )
        self.to_logits = nn.Linear(hidden_dim, n_classes)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                std = 1.0 / math.sqrt(module.weight.size(1))
                nn.init.normal_(module.weight, mean=0.0, std=std)
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
        return data_omics

    def forward(self, x=None, **kwargs):
        data_omics = self._get_omics_tensor(x=x, **kwargs)
        features = self.net(data_omics)
        return self.to_logits(features)

    def captum(self, omics):
        logits = self.forward(omics)
        hazards = torch.sigmoid(logits)
        survival = torch.cumprod(1 - hazards, dim=1)
        risk = -torch.sum(survival, dim=1)
        return risk
