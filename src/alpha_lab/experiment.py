from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from alpha_lab.evaluation import compute_ic, compute_rank_ic
from alpha_lab.interfaces import validate_factor_output
from alpha_lab.labels import forward_return
from alpha_lab.quantile import long_short_return, quantile_returns
from alpha_lab.splits import time_split


@dataclass(frozen=True)
class ExperimentSummary:
    """Scalar summary of one experiment (or one split thereof).

    All metrics are computed over the evaluation period.  Fields are NaN
    when the evaluation period contains insufficient data.
    """

    mean_ic: float
    """Cross-sectional Pearson IC averaged across evaluation dates."""

    mean_rank_ic: float
    """Cross-sectional Spearman RankIC averaged across evaluation dates."""

    ic_ir: float
    """Information ratio: mean_ic / std_ic (ddof=1).  NaN when fewer than
    two non-NaN IC observations are available or std_ic == 0."""

    mean_long_short_return: float
    """Average top-minus-bottom quantile return across evaluation dates."""

    long_short_hit_rate: float
    """Fraction of evaluation dates on which the long-short return was > 0."""

    n_dates: int
    """Number of distinct evaluation dates that produced a finite IC value."""


@dataclass
class ExperimentResult:
    """Full output of one :func:`run_factor_experiment` call.

    ``factor_df`` and ``label_df`` always cover the **full sample** so the
    caller can inspect the complete time-series.  All other DataFrames and
    ``summary`` are restricted to the **evaluation period** (test split when
    a split is requested, full sample otherwise).
    """

    factor_df: pd.DataFrame
    """Canonical long-form factor output for the full sample."""

    label_df: pd.DataFrame
    """Canonical long-form forward-return labels for the full sample."""

    ic_df: pd.DataFrame
    """Pearson IC by date over the evaluation period."""

    rank_ic_df: pd.DataFrame
    """Spearman RankIC by date over the evaluation period."""

    quantile_returns_df: pd.DataFrame
    """Mean return per quantile bucket and date over the evaluation period."""

    long_short_df: pd.DataFrame
    """Long-short (top minus bottom quantile) return by date."""

    summary: ExperimentSummary
    """Scalar summary metrics for the evaluation period."""


def run_factor_experiment(
    prices: pd.DataFrame,
    factor_fn: Callable[[pd.DataFrame], pd.DataFrame],
    *,
    horizon: int = 1,
    n_quantiles: int = 5,
    train_end: str | pd.Timestamp | None = None,
    test_start: str | pd.Timestamp | None = None,
    val_start: str | pd.Timestamp | None = None,
) -> ExperimentResult:
    """Run a factor experiment end-to-end.

    **Timestamp discipline**

    - Factor values are computed by ``factor_fn`` and must only use
      information available at or before their observation date.  The runner
      validates the canonical schema but cannot verify the internals of
      ``factor_fn`` — that responsibility rests with the factor author.
    - Labels are ``forward_return(prices, horizon=horizon)``.  The label
      stored at date *t* equals ``close[t + horizon] / close[t] - 1``,
      where ``t + horizon`` is measured in per-asset row count.  The label
      value uses strictly future prices; the label is *stored* at *t* so it
      can be merged with the factor on ``(date, asset)`` without lookahead.
    - Evaluation metrics are computed on the **test period** when a split is
      provided, or on the full sample otherwise.  Training-period dates are
      never included in IC or quantile metrics.

    Parameters
    ----------
    prices:
        Long-form price panel with columns ``[date, asset, close]``.
    factor_fn:
        Callable that accepts a price panel and returns a canonical
        ``[date, asset, factor, value]`` DataFrame.  Must produce exactly
        one factor name.
    horizon:
        Forward-return look-ahead in per-asset rows.
    n_quantiles:
        Number of quantile buckets for :func:`~alpha_lab.quantile.quantile_returns`.
    train_end:
        Last date (inclusive) of the training period.  Provide together
        with ``test_start`` to restrict evaluation to the test period.
    test_start:
        First date (inclusive) of the evaluation period.
    val_start:
        Optional start of a validation window between ``train_end`` and
        ``test_start``; passed through to :func:`~alpha_lab.splits.time_split`.

    Returns
    -------
    ExperimentResult
    """
    # --- Step 0: validate split arguments -----------------------------------
    # Require both or neither.  A lone train_end / test_start, or a val_start
    # without a complete split, silently evaluates on the full sample — that is
    # a dangerous misuse path and must be an explicit error.
    if (train_end is None) != (test_start is None):
        raise ValueError(
            "train_end and test_start must both be provided or both be omitted; "
            f"got train_end={train_end!r}, test_start={test_start!r}."
        )
    if val_start is not None and train_end is None:
        raise ValueError(
            "val_start requires both train_end and test_start to be specified."
        )

    # --- Step 1: factor values (full sample) --------------------------------
    factor_df = factor_fn(prices)
    validate_factor_output(factor_df)

    # --- Step 2: forward-return labels (full sample) ------------------------
    # Labels at date t: close[t+horizon]/close[t] - 1 (strictly future prices).
    # Stored at t so they merge with factor on (date, asset) without lookahead.
    label_df = forward_return(prices, horizon=horizon)

    # --- Step 3: resolve evaluation period ----------------------------------
    if train_end is not None and test_start is not None:
        # time_split is date-comparison-based and is panel-safe: all rows
        # sharing a date receive the same mask value.
        masks = time_split(
            factor_df["date"],
            train_end=train_end,
            test_start=test_start,
            val_start=val_start,
        )
        eval_factor = factor_df[masks["test"]].reset_index(drop=True)
        # Filter labels by the exact test-period dates (not by positional mask,
        # since label_df may have a different row structure than factor_df).
        eval_date_index = pd.DatetimeIndex(eval_factor["date"].unique())
        eval_label = label_df[label_df["date"].isin(eval_date_index)].reset_index(
            drop=True
        )
    else:
        eval_factor = factor_df.copy()
        eval_label = label_df.copy()

    # --- Step 4: IC / RankIC -----------------------------------------------
    ic_df = compute_ic(eval_factor, eval_label)
    rank_ic_df = compute_rank_ic(eval_factor, eval_label)

    # --- Step 5: quantile returns and long-short ----------------------------
    qr_df = quantile_returns(eval_factor, eval_label, n_quantiles=n_quantiles)
    ls_df = long_short_return(qr_df)

    # --- Step 6: summary ----------------------------------------------------
    summary = _summarise(ic_df, rank_ic_df, ls_df)

    return ExperimentResult(
        factor_df=factor_df,
        label_df=label_df,
        ic_df=ic_df,
        rank_ic_df=rank_ic_df,
        quantile_returns_df=qr_df,
        long_short_df=ls_df,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _summarise(
    ic_df: pd.DataFrame,
    rank_ic_df: pd.DataFrame,
    ls_df: pd.DataFrame,
) -> ExperimentSummary:
    """Compute scalar summary metrics from per-date evaluation DataFrames."""
    ic_vals = ic_df["ic"].dropna() if not ic_df.empty else pd.Series(dtype=float)
    rank_ic_vals = (
        rank_ic_df["rank_ic"].dropna()
        if not rank_ic_df.empty
        else pd.Series(dtype=float)
    )
    ls_vals = (
        ls_df["long_short_return"].dropna()
        if not ls_df.empty
        else pd.Series(dtype=float)
    )

    mean_ic = float(ic_vals.mean()) if len(ic_vals) > 0 else float("nan")
    mean_rank_ic = float(rank_ic_vals.mean()) if len(rank_ic_vals) > 0 else float("nan")

    # ic_ir = mean_ic / std_ic  (ddof=1; NaN when std unavailable or zero)
    ic_std = float(ic_vals.std(ddof=1)) if len(ic_vals) > 1 else float("nan")
    if np.isnan(ic_std) or ic_std == 0.0:
        ic_ir: float = float("nan")
    else:
        ic_ir = mean_ic / ic_std

    mean_ls = float(ls_vals.mean()) if len(ls_vals) > 0 else float("nan")
    hit_rate = float((ls_vals > 0).mean()) if len(ls_vals) > 0 else float("nan")
    # n_dates: dates with a finite IC value (matches the denominator of mean_ic and ic_ir).
    n_dates = (
        int(ic_df.loc[ic_df["ic"].notna(), "date"].nunique())
        if not ic_df.empty
        else 0
    )

    return ExperimentSummary(
        mean_ic=mean_ic,
        mean_rank_ic=mean_rank_ic,
        ic_ir=ic_ir,
        mean_long_short_return=mean_ls,
        long_short_hit_rate=hit_rate,
        n_dates=n_dates,
    )
