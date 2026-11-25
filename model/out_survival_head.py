"""Shared survival prediction layer, separate from feature encoders."""
import torch.nn as nn


class SurvivalPredictionHead(nn.Linear):
    """Map embeddings [B, input_dim] to interval logits [B, n_classes].

    Return raw logits: NLLSurvLoss handles hazards; cross-entropy also accepts
    logits. Inheriting Linear preserves the existing weight/bias checkpoint keys.
    """

    def __init__(self, input_dim, n_classes=4):
        super().__init__(input_dim, n_classes)
