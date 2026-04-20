"""Regime-conditional factor evaluation.

Classifies each evaluation date into market regimes (bull/bear direction,
high/low volatility) using equal-weight cross-sectional market returns, then
computes per-regime IC and long-short return statistics.  The output
summarises whether a factor's predictive power is regime-dependent — a
critical robustness diagnostic for promotion decisions.

All regime labels are assigned using **past information only**: the direction
regime at date *t* is determined by the sign of the market return at *t*
(realised that day, not forward-looking), and the volatility regime is
determined by comparing a trailing rolling standard deviation to its
expanding median up to *t*.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from alpha_lab.research_evaluation_config import (
    DEFAULT_RESEARCH_EVALUATION_CONFIG,
    RegimeAnalysisConfig,
)

DEFAULT_REGIME_CONFIG = DEFAULT_RESEARCH_EVALUATION_CONFIG.regime_analysis


@dataclass(frozen=True)
class RegimeStats:
    """Per-regime IC and long-short return statistics."""

    regime_label: str
    n_dates: int
    mean_ic: float
    mean_rank_ic: float
    mean_long_short_return: float
    long_short_hit_rate: float
    ic_positive_rate: float


@dataclass(frozen=True)
class RegimeConditionalSummary:
    """Full regime-conditional evaluation output."""

    direction_regimes: tuple[RegimeStats, ...]
    """Bull / Bear regime statistics."""

    volatility_regimes: tuple[RegimeStats, ...]
    """High-vol / Low-vol regime statistics."""

    regime_flags: tuple[str, ...]
    """Diagnostic flags indicating regime-dependent weakness."""


def classify_market_regimes(
    ic_df: pd.DataFrame,
    long_short_df: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    vol_window: int = 20,
) -> pd.DataFrame:
    """Assign direction and volatility regime labels to evaluation dates.

    Parameters
    ----------
    ic_df:
        IC DataFrame with a ``date`` column (used for the set of eval dates).
    long_short_df:
        Long-short return DataFrame with a ``date`` column.
    prices:
        Long-form price panel ``[date, asset, close]``.
    vol_window:
        Rolling window for volatility estimation (in trading days).

    Returns
    -------
    DataFrame with columns ``[date, market_return, direction_regime,
    rolling_vol, vol_regime]``.
    """
    eval_dates = set()
    if not ic_df.empty and "date" in ic_df.columns:
        eval_dates |= set(pd.to_datetime(ic_df["date"]).unique())
    if not long_short_df.empty and "date" in long_short_df.columns:
        eval_dates |= set(pd.to_datetime(long_short_df["date"]).unique())

    if not eval_dates:
        return _empty_regime_frame()

    # Equal-weight cross-sectional market return per date
    pdf = prices[["date", "asset", "close"]].copy()
    pdf["date"] = pd.to_datetime(pdf["date"], errors="coerce")
    pdf = pdf.sort_values(["asset", "date"], kind="mergesort").reset_index(drop=True)
    pdf["asset_return"] = pdf.groupby("asset")["close"].pct_change()

    mkt = (
        pdf.dropna(subset=["asset_return"])
        .groupby("date")["asset_return"]
        .mean()
        .rename("market_return")
        .reset_index()
    )
    mkt["date"] = pd.to_datetime(mkt["date"])
    mkt = mkt.sort_values("date", kind="mergesort").reset_index(drop=True)

    # Direction regime: sign of market return on that date
    mkt["direction_regime"] = np.where(mkt["market_return"] >= 0, "bull", "bear")

    # Volatility regime: trailing rolling std vs expanding median
    mkt["rolling_vol"] = (
        mkt["market_return"]
        .rolling(window=vol_window, min_periods=max(1, vol_window // 2))
        .std(ddof=1)
    )
    expanding_median = mkt["rolling_vol"].expanding(min_periods=1).median()
    mkt["vol_regime"] = np.where(mkt["rolling_vol"] >= expanding_median, "high_vol", "low_vol")

    # Filter to evaluation dates
    result = mkt[mkt["date"].isin(eval_dates)].reset_index(drop=True)
    return result


def compute_regime_conditional(
    ic_df: pd.DataFrame,
    rank_ic_df: pd.DataFrame,
    long_short_df: pd.DataFrame,
    regime_df: pd.DataFrame,
    *,
    config: RegimeAnalysisConfig = DEFAULT_REGIME_CONFIG,
) -> RegimeConditionalSummary:
    """Compute per-regime IC and L/S statistics with diagnostic flags.

    Parameters
    ----------
    ic_df:
        Pearson IC by date ``[date, ic]``.
    rank_ic_df:
        Spearman RankIC by date ``[date, rank_ic]``.
    long_short_df:
        Long-short return by date ``[date, long_short_return]``.
    regime_df:
        Regime classification from :func:`classify_market_regimes`.
    config:
        Regime analysis thresholds.

    Returns
    -------
    RegimeConditionalSummary
    """
    if regime_df.empty:
        return RegimeConditionalSummary(
            direction_regimes=(),
            volatility_regimes=(),
            regime_flags=(),
        )

    merged = _merge_regime_data(ic_df, rank_ic_df, long_short_df, regime_df)
    if merged.empty:
        return RegimeConditionalSummary(
            direction_regimes=(),
            volatility_regimes=(),
            regime_flags=(),
        )

    direction_stats = _compute_regime_group_stats(
        merged, regime_col="direction_regime", config=config
    )
    volatility_stats = _compute_regime_group_stats(merged, regime_col="vol_regime", config=config)

    flags = _collect_regime_flags(direction_stats, volatility_stats, config=config)

    return RegimeConditionalSummary(
        direction_regimes=tuple(direction_stats),
        volatility_regimes=tuple(volatility_stats),
        regime_flags=tuple(flags),
    )


def _merge_regime_data(
    ic_df: pd.DataFrame,
    rank_ic_df: pd.DataFrame,
    long_short_df: pd.DataFrame,
    regime_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge IC, RankIC, L/S return, and regime labels on date."""
    base = regime_df[["date", "direction_regime", "vol_regime"]].copy()
    base["date"] = pd.to_datetime(base["date"])

    if not ic_df.empty and "ic" in ic_df.columns:
        ic = ic_df[["date", "ic"]].copy()
        ic["date"] = pd.to_datetime(ic["date"])
        base = base.merge(ic, on="date", how="left")

    if not rank_ic_df.empty and "rank_ic" in rank_ic_df.columns:
        ric = rank_ic_df[["date", "rank_ic"]].copy()
        ric["date"] = pd.to_datetime(ric["date"])
        base = base.merge(ric, on="date", how="left")

    if not long_short_df.empty and "long_short_return" in long_short_df.columns:
        ls = long_short_df[["date", "long_short_return"]].copy()
        ls["date"] = pd.to_datetime(ls["date"])
        base = base.merge(ls, on="date", how="left")

    return base


def _compute_regime_group_stats(
    merged: pd.DataFrame,
    *,
    regime_col: str,
    config: RegimeAnalysisConfig,
) -> list[RegimeStats]:
    """Compute per-regime statistics for a given regime column."""
    if regime_col not in merged.columns:
        return []

    stats: list[RegimeStats] = []
    for label, group in merged.groupby(regime_col, sort=True):
        n = len(group)
        if n < config.min_regime_dates:
            continue

        ic_vals = group["ic"].dropna() if "ic" in group.columns else pd.Series(dtype=float)
        ric_vals = (
            group["rank_ic"].dropna() if "rank_ic" in group.columns else pd.Series(dtype=float)
        )
        ls_vals = (
            group["long_short_return"].dropna()
            if "long_short_return" in group.columns
            else pd.Series(dtype=float)
        )

        mean_ic = float(ic_vals.mean()) if len(ic_vals) > 0 else float("nan")
        mean_rank_ic = float(ric_vals.mean()) if len(ric_vals) > 0 else float("nan")
        mean_ls = float(ls_vals.mean()) if len(ls_vals) > 0 else float("nan")
        ls_hit = float((ls_vals > 0).mean()) if len(ls_vals) > 0 else float("nan")
        ic_pos = float((ic_vals > 0).mean()) if len(ic_vals) > 0 else float("nan")

        stats.append(
            RegimeStats(
                regime_label=str(label),
                n_dates=n,
                mean_ic=mean_ic,
                mean_rank_ic=mean_rank_ic,
                mean_long_short_return=mean_ls,
                long_short_hit_rate=ls_hit,
                ic_positive_rate=ic_pos,
            )
        )
    return stats


def _collect_regime_flags(
    direction_stats: list[RegimeStats],
    volatility_stats: list[RegimeStats],
    *,
    config: RegimeAnalysisConfig,
) -> list[str]:
    """Flag regime-dependent weaknesses."""
    flags: list[str] = []

    for rs in direction_stats:
        if np.isfinite(rs.mean_ic) and rs.mean_ic <= config.regime_negative_ic_flag_threshold:
            flags.append(f"regime_{rs.regime_label}_negative_ic")
        if (
            np.isfinite(rs.mean_long_short_return)
            and rs.mean_long_short_return <= config.regime_negative_ls_flag_threshold
        ):
            flags.append(f"regime_{rs.regime_label}_negative_ls")

    for rs in volatility_stats:
        if np.isfinite(rs.mean_ic) and rs.mean_ic <= config.regime_negative_ic_flag_threshold:
            flags.append(f"regime_{rs.regime_label}_negative_ic")
        if (
            np.isfinite(rs.mean_long_short_return)
            and rs.mean_long_short_return <= config.regime_negative_ls_flag_threshold
        ):
            flags.append(f"regime_{rs.regime_label}_negative_ls")

    if flags:
        flags.append("regime_conditional_weakness")

    return flags


def _empty_regime_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "date",
            "market_return",
            "direction_regime",
            "rolling_vol",
            "vol_regime",
        ]
    )
