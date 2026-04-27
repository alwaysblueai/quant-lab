from __future__ import annotations

import numpy as np
import pandas as pd


def winsorize_series(s: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    if s.empty:
        return s.copy()
    lo = s.quantile(lower)
    hi = s.quantile(upper)
    return s.clip(lower=lo, upper=hi)


def zscore_series(s: pd.Series) -> pd.Series:
    """Z-score standardize a numeric series using **population** standard deviation (ddof=0).

    Design note — ddof=0 vs ddof=1
    --------------------------------
    This function intentionally uses ``ddof=0`` (population std) rather than
    ``ddof=1`` (sample std).  The purpose is cross-sectional normalization:
    we want every date's factor distribution to have unit variance **within
    that cross-section**, not to estimate a population parameter.  With a
    small cross-section (e.g. 10 assets), ``ddof=1`` inflates the denominator
    and systematically compresses z-scores toward zero.

    In contrast, :func:`~alpha_lab.evaluation.compute_ic_summary` uses
    ``ddof=1`` because it estimates the *time-series* dispersion of IC values
    across dates — a true sample-statistics use case.  The two choices are
    deliberate and should not be unified.

    Returns a zero-filled series when std is 0 or NaN (constant input).
    """
    if s.empty:
        return s.copy()
    std = s.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(np.zeros(len(s)), index=s.index, dtype=float)
    return (s - s.mean()) / std
