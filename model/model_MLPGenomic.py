import torch
import torch.nn as nn


class MLPGenomics(nn.Module):
    def __init__(
        self,
        input_dim,
        n_classes=4,
        projection_dim=512,
        dropout=0.1,
    ):
        super(MLPGenomics, self).__init__()
        self.projection_dim = projection_dim


        self.net = nn.Sequential(
            nn.Linear(input_dim, projection_dim//2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(projection_dim//2, projection_dim//2), nn.ReLU(), nn.Dropout(dropout)
        ) 

        self.to_logits = nn.Sequential(
            nn.Linear(projection_dim //2, n_classes)
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

        return data_omics.float().squeeze()

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


# class MLPGenomics(MLPOmics):
#     def __init__(self, input_dim, output_dim, projection_dim=512, dropout=0.1):
#         super().__init__(
#             input_dim=input_dim,
#             n_classes=output_dim,
#             projection_dim=projection_dim,
#             dropout=dropout,
#         )
