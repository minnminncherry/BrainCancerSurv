from __future__ import annotations

from typing import Tuple

import numpy as np


def fold_start_end(n_samples: int, n_folds: int, fold_index: int) -> Tuple[int, int]:
    """
    Return the [start, end) slice for a contiguous K-fold split.

    Fold sizes are balanced by distributing the remainder across the first folds,
    like `np.array_split(np.arange(n_samples), n_folds)`.
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be > 0, got {n_samples}")
    if n_folds <= 1:
        raise ValueError(f"n_folds must be > 1, got {n_folds}")
    if not (0 <= fold_index < n_folds):
        raise ValueError(f"fold_index must be in [0, {n_folds - 1}], got {fold_index}")

    base = n_samples // n_folds
    remainder = n_samples % n_folds

    # First `remainder` folds get one extra sample.
    start = fold_index * base + min(fold_index, remainder)
    end = start + base + (1 if fold_index < remainder else 0)
    return start, end


def fold_arange(n_samples: int, n_folds: int, fold_index: int, *, dtype=np.int64) -> np.ndarray:
    """
    Return the indices for one fold as `np.arange(start, end)`.
    """
    start, end = fold_start_end(n_samples=n_samples, n_folds=n_folds, fold_index=fold_index)
    return np.arange(start, end, dtype=dtype)

