"""Tail-risk metrics for long-short return series.

Computes max drawdown, drawdown duration, VaR/CVaR, and max consecutive
loss streaks from a per-date long-short return series.  All functions
operate on a ``pd.DataFrame`` with columns ``[date, long_short_return]``
— the same schema produced by :func:`~alpha_lab.quantile.long_short_return`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TailRiskSummary:
    """Scalar tail-risk metrics for a long-short return series."""

    max_drawdown: float
    """Maximum peak-to-trough drawdown of cumulative L/S return.
    Expressed as a positive fraction (e.g. 0.15 means 15% drawdown).
    NaN when the series is empty or all-NaN."""

    max_drawdown_duration: int
    """Number of dates from the drawdown peak to the drawdown trough.
    0 when there is no drawdown; -1 when the series is empty."""

    max_consecutive_loss_days: int
    """Longest streak of consecutive dates with ``long_short_return <= 0``.
    0 when there are no losses; -1 when the series is empty."""

    var_5: float
    """5th-percentile (historical) Value-at-Risk of daily L/S returns.
    Expressed as a return (typically negative).  NaN when empty."""

    cvar_5: float
    """Conditional VaR (Expected Shortfall) at the 5% level.
    Mean of all L/S returns at or below the 5th percentile.
    NaN when empty."""

    calmar_ratio: float
    """Mean L/S return divided by max drawdown.  NaN when max_drawdown
    is zero or the series is empty."""

    n_loss_dates: int
    """Count of dates with ``long_short_return < 0``."""

    n_total_dates: int
    """Total finite L/S return observations."""


def compute_tail_risk(
    long_short_df: pd.DataFrame,
) -> TailRiskSummary:
    """Compute tail-risk summary from a long-short return DataFrame.

    Parameters
    ----------
    long_short_df:
        DataFrame with columns ``[date, long_short_return]``.
        NaN returns are dropped before computation.

    Returns
    -------
    TailRiskSummary
    """
    if long_short_df.empty or "long_short_return" not in long_short_df.columns:
        return _empty_summary()

    df = long_short_df[["date", "long_short_return"]].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["long_short_return"])
    df = df.sort_values("date", kind="mergesort").reset_index(drop=True)

    if df.empty:
        return _empty_summary()

    returns = df["long_short_return"].to_numpy(dtype=float)
    n_total = len(returns)

    # --- Max drawdown via cumulative return ---
    cum = np.cumsum(returns)
    running_max = np.maximum.accumulate(cum)
    drawdowns = running_max - cum  # always >= 0
    max_dd = float(np.max(drawdowns))

    # Duration: dates from peak to trough of worst drawdown
    if max_dd > 0:
        trough_idx = int(np.argmax(drawdowns))
        # Peak is the last date where running_max == running_max[trough_idx]
        peak_val = running_max[trough_idx]
        peak_candidates = np.where(cum[: trough_idx + 1] >= peak_val - 1e-15)[0]
        peak_idx = int(peak_candidates[-1]) if len(peak_candidates) > 0 else 0
        dd_duration = trough_idx - peak_idx
    else:
        dd_duration = 0

    # --- Max consecutive loss days ---
    is_loss = returns <= 0
    max_consec = _max_consecutive_true(is_loss)

    # --- VaR and CVaR at 5% ---
    var_5 = float(np.percentile(returns, 5))
    mask_below = returns <= var_5
    cvar_5 = float(np.mean(returns[mask_below])) if mask_below.any() else var_5

    # --- Calmar ratio ---
    mean_ret = float(np.mean(returns))
    calmar = mean_ret / max_dd if max_dd > 0 else float("nan")

    n_loss = int(np.sum(returns < 0))

    return TailRiskSummary(
        max_drawdown=max_dd,
        max_drawdown_duration=dd_duration,
        max_consecutive_loss_days=max_consec,
        var_5=var_5,
        cvar_5=cvar_5,
        calmar_ratio=calmar,
        n_loss_dates=n_loss,
        n_total_dates=n_total,
    )


def _max_consecutive_true(arr: np.ndarray) -> int:
    """Return the length of the longest consecutive True run."""
    if len(arr) == 0:
        return 0
    max_run = 0
    current = 0
    for val in arr:
        if val:
            current += 1
            if current > max_run:
                max_run = current
        else:
            current = 0
    return max_run


def _empty_summary() -> TailRiskSummary:
    return TailRiskSummary(
        max_drawdown=float("nan"),
        max_drawdown_duration=-1,
        max_consecutive_loss_days=-1,
        var_5=float("nan"),
        cvar_5=float("nan"),
        calmar_ratio=float("nan"),
        n_loss_dates=0,
        n_total_dates=0,
    )
