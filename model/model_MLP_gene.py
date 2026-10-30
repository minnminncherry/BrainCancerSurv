import torch
import torch.nn as nn


class MLPGenomics(nn.Module):
    def __init__(
        self,
        input_dim,
        n_classes=4,
        projection_dim=256,
        dropout=0.1,
        num_layers=2,
    ):
        super(MLPGenomics, self).__init__()
        self.projection_dim = projection_dim
        self.n_classes = n_classes

        hidden_dim = projection_dim // 2
        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        ]
        for _ in range(max(int(num_layers) - 1, 0)):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
        self.net = nn.Sequential(*layers)

        self.to_logits = nn.Sequential(
            nn.Linear(hidden_dim, n_classes)
        )

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
        data = self.net(data_omics)
        logits = self.to_logits(data)
        return logits

    def captum(self, omics):
        data_omics = self._get_omics_tensor(x=omics)
        data = self.net(data_omics)
        logits = self.to_logits(data)

        hazards = torch.sigmoid(logits)
        survival = torch.cumprod(1 - hazards, dim=1)
        risk = -torch.sum(survival, dim=1)
        return risk
