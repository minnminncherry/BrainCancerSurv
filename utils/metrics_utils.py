"""Metrics for right-censored survival outcomes."""
import warnings

import numpy as np


def concordance_index(event_times, censorships, risks, tied_tol=1e-8):
    """Harrell's C-index, with the scikit-survival tied-time convention.

    censorship=0 denotes an observed event; 1 denotes censoring.
    Higher risk predicts shorter survival. A death is comparable with later
    observations and with censoring at the same time. Equal-time deaths are
    excluded. Risk ties (within tied_tol) receive half credit.
    No comparable pairs yields NaN rather than a misleading zero.
    """
    times = np.asarray(event_times, dtype=float)
    censor = np.asarray(censorships, dtype=float)
    scores = np.asarray(risks, dtype=float)
    if any(values.ndim != 1 for values in (times, censor, scores)):
        raise ValueError("Times, censorships, and risks must be one-dimensional.")
    if not len(times) == len(censor) == len(scores):
        raise ValueError("Times, censorships, and risks must have equal lengths.")
    if not all(np.isfinite(values).all() for values in (times, censor, scores)):
        raise ValueError("Survival metrics require finite times, censoring labels, and risks.")
    if not np.isin(censor, [0, 1]).all():
        raise ValueError("Censorship must be 0 (event) or 1 (censored).")
    if not np.isfinite(tied_tol) or tied_tol < 0:
        raise ValueError("tied_tol must be finite and nonnegative.")

    credit, comparable = 0.0, 0
    for event in np.flatnonzero(censor == 0):
        eligible = (times > times[event]) | ((times == times[event]) & (censor == 1))
        differences = scores[event] - scores[eligible]
        credit += np.count_nonzero(differences > tied_tol)
        credit += 0.5 * np.count_nonzero(np.abs(differences) <= tied_tol)
        comparable += differences.size
    if comparable == 0:
        warnings.warn("Harrell's C-index is undefined: no comparable patient pairs.", RuntimeWarning, stacklevel=2)
        return float("nan")
    return float(credit / comparable)
