"""Discrete-time survival likelihood with 1=censored and 0=observed event."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LossClass(nn.Module):
    """Accept hazard logits [B, intervals], time-bin indices [B], and censoring [B].

    Events contribute S(before bin) * hazard(bin). Censored observations
    contribute survival through their bin. alpha=0 is the ordinary likelihood;
    alpha>0 downweights censored examples, matching event-focused weighting.
    """
    def __init__(self, alpha=0.0):
        super().__init__()
        if not 0 <= alpha < 1:
            raise ValueError("alpha must be in [0, 1).")
        self.alpha = alpha

    def nullLoss(self, logits, Y, c):
        """Negative log-likelihood using time bins Y and censoring c."""
        if logits.ndim != 2 or logits.size(0) == 0:
            raise ValueError("Expected nonempty hazard logits [batch, intervals].")
        labels = Y.reshape(-1).to(logits.device)
        censor = c.reshape(-1).to(device=logits.device, dtype=logits.dtype)
        if labels.numel() != logits.size(0) or censor.numel() != logits.size(0):
            raise ValueError("One time bin and censoring label are required per patient.")
        if not torch.isfinite(labels).all() or not (labels == labels.long()).all():
            raise ValueError("Survival time bins must be finite integers.")
        labels = labels.long()
        if (labels < 0).any() or (labels >= logits.size(1)).any():
            raise ValueError("Time bins must index the model's hazard intervals.")
        if not ((censor == 0) | (censor == 1)).all():
            raise ValueError("Censoring must be 0 (event) or 1 (censored).")
        log_survival = F.logsigmoid(-logits).cumsum(dim=1)
        log_before = F.pad(log_survival, (1, 0), value=0.0)[:, :-1]
        index = labels.unsqueeze(1)
        event_log_prob = (log_before + F.logsigmoid(logits)).gather(1, index).squeeze(1)
        censored_log_prob = log_survival.gather(1, index).squeeze(1)
        return (-((1 - censor) * event_log_prob + (1 - self.alpha) * censor * censored_log_prob)).mean()

    def forward(self, logits, Y, c):
        return self.nullLoss(logits, Y, c)


def survival_outputs(logits):
    """Hazards, survival through intervals, and event probability mass per interval."""
    hazards = logits.sigmoid()
    log_survival = F.logsigmoid(-logits).cumsum(dim=1)
    survival = log_survival.exp()
    before = F.pad(log_survival, (1, 0), value=0.0)[:, :-1].exp()
    return hazards, survival, before * hazards
