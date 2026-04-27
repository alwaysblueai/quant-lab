from __future__ import annotations

import math

import numpy as np
import pandas as pd


def purged_kfold_split(
    dates: np.ndarray,
    n_splits: int = 5,
    label_horizon: int = 1,
    embargo_pct: float = 0.01,
) -> list[dict[str, np.ndarray]]:
    """Build Purged K-Fold train/test masks on a unique, sorted date axis."""
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    if label_horizon < 0:
        raise ValueError("label_horizon must be >= 0")
    if embargo_pct < 0.0 or embargo_pct >= 1.0:
        raise ValueError("embargo_pct must be in [0, 1)")

    date_axis = np.asarray(pd.to_datetime(np.asarray(dates)))
    if date_axis.ndim != 1:
        raise ValueError("dates must be a 1-D array")
    if pd.isna(date_axis).any():
        raise ValueError("dates contains NaT values")
    if len(date_axis) != len(np.unique(date_axis)):
        raise ValueError("dates must be unique for purged_kfold_split")
    if len(date_axis) > 1 and not np.all(date_axis[:-1] <= date_axis[1:]):
        raise ValueError("dates must be sorted in chronological order")

    n_dates = len(date_axis)
    if n_splits > n_dates:
        raise ValueError("n_splits cannot exceed the number of dates")

    fold_sizes = np.full(n_splits, n_dates // n_splits, dtype=int)
    fold_sizes[: n_dates % n_splits] += 1
    embargo_days = int(math.ceil(n_dates * float(embargo_pct)))

    splits: list[dict[str, np.ndarray]] = []
    all_idx = np.arange(n_dates, dtype=int)
    cursor = 0
    for fold_size in fold_sizes:
        test_start = cursor
        test_end_exclusive = cursor + int(fold_size)
        test_end_inclusive = test_end_exclusive - 1
        cursor = test_end_exclusive

        test_mask = np.zeros(n_dates, dtype=bool)
        test_mask[test_start:test_end_exclusive] = True

        train_mask = ~test_mask

        # Purge train points whose label window overlaps the test window.
        overlap = (all_idx <= test_end_inclusive) & ((all_idx + int(label_horizon)) >= test_start)
        train_mask[overlap] = False

        # Embargo the immediate post-test period.
        embargo_start = test_end_exclusive
        embargo_end = min(n_dates, embargo_start + embargo_days)
        if embargo_end > embargo_start:
            train_mask[embargo_start:embargo_end] = False

        splits.append({"train": train_mask, "test": test_mask})

    return splits
