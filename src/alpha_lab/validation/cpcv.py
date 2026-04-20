from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd


def cpcv_split(
    dates: np.ndarray,
    n_splits: int = 6,
    n_test_splits: int = 2,
    label_horizon: int = 1,
    embargo_pct: float = 0.01,
) -> list[dict[str, np.ndarray]]:
    """Combinatorial Purged Cross-Validation split masks."""
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    if n_test_splits < 1 or n_test_splits >= n_splits:
        raise ValueError("n_test_splits must be in [1, n_splits)")
    if label_horizon < 0:
        raise ValueError("label_horizon must be >= 0")
    if embargo_pct < 0.0 or embargo_pct >= 1.0:
        raise ValueError("embargo_pct must be in [0, 1)")

    date_axis = np.asarray(pd.to_datetime(np.asarray(dates)))
    if date_axis.ndim != 1:
        raise ValueError("dates must be 1-D")
    if pd.isna(date_axis).any():
        raise ValueError("dates contains NaT")
    if len(date_axis) != len(np.unique(date_axis)):
        raise ValueError("dates must be unique")
    if len(date_axis) > 1 and not np.all(date_axis[:-1] <= date_axis[1:]):
        raise ValueError("dates must be sorted")
    n_dates = len(date_axis)
    if n_splits > n_dates:
        raise ValueError("n_splits cannot exceed number of dates")

    fold_sizes = np.full(n_splits, n_dates // n_splits, dtype=int)
    fold_sizes[: n_dates % n_splits] += 1
    fold_ranges: list[tuple[int, int]] = []
    cursor = 0
    for size in fold_sizes:
        start = cursor
        end = cursor + int(size)  # exclusive
        fold_ranges.append((start, end))
        cursor = end

    embargo_days = int(math.ceil(float(n_dates) * float(embargo_pct)))
    all_idx = np.arange(n_dates, dtype=int)
    splits: list[dict[str, np.ndarray]] = []
    for test_combo in itertools.combinations(range(n_splits), int(n_test_splits)):
        test_mask = np.zeros(n_dates, dtype=bool)
        for fold_idx in test_combo:
            start, end = fold_ranges[fold_idx]
            test_mask[start:end] = True

        train_mask = ~test_mask
        test_idx = np.flatnonzero(test_mask)
        if test_idx.size == 0:
            continue

        # Purge overlapping label windows.
        test_start = int(test_idx.min())
        test_end = int(test_idx.max())
        overlap = (all_idx <= test_end) & ((all_idx + int(label_horizon)) >= test_start)
        train_mask[overlap] = False

        # Embargo periods immediately after each contiguous test block.
        block_starts = [test_idx[0]]
        block_ends: list[int] = []
        for i in range(1, len(test_idx)):
            if test_idx[i] != test_idx[i - 1] + 1:
                block_ends.append(int(test_idx[i - 1]))
                block_starts.append(int(test_idx[i]))
        block_ends.append(int(test_idx[-1]))

        for end in block_ends:
            embargo_start = end + 1
            embargo_end = min(n_dates, embargo_start + embargo_days)
            if embargo_end > embargo_start:
                train_mask[embargo_start:embargo_end] = False

        splits.append({"train": train_mask, "test": test_mask})

    return splits


def compute_pbo(
    in_sample_scores: pd.DataFrame,
    out_of_sample_scores: pd.DataFrame,
) -> float:
    """Estimate Probability of Backtest Overfitting (PBO).

    ``in_sample_scores`` and ``out_of_sample_scores`` must share shape
    ``[n_splits, n_strategies]``.
    """
    if in_sample_scores.shape != out_of_sample_scores.shape:
        raise ValueError("in_sample_scores and out_of_sample_scores must have same shape")
    if in_sample_scores.empty:
        return float("nan")

    is_values = in_sample_scores.to_numpy(dtype=float)
    oos_values = out_of_sample_scores.to_numpy(dtype=float)
    n_splits, n_strategies = is_values.shape
    if n_strategies < 2:
        return float("nan")

    logit_vals: list[float] = []
    for i in range(n_splits):
        row_is = is_values[i, :]
        row_oos = oos_values[i, :]
        if not np.all(np.isfinite(row_is)) or not np.all(np.isfinite(row_oos)):
            continue

        best_idx = int(np.argmax(row_is))
        ranks = pd.Series(row_oos).rank(method="average", ascending=True).to_numpy(dtype=float)
        percentile = (ranks[best_idx] - 0.5) / float(n_strategies)
        percentile = min(max(percentile, 1e-6), 1.0 - 1e-6)
        logit_vals.append(float(np.log(percentile / (1.0 - percentile))))

    if not logit_vals:
        return float("nan")
    arr = np.asarray(logit_vals, dtype=float)
    return float(np.mean(arr < 0.0))
