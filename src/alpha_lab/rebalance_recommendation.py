from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from alpha_lab.factor_report import estimate_half_life


@dataclass(frozen=True)
class RebalanceRecommendation:
    """Research-side rebalance cadence recommendation."""

    cadence_label: str
    cadence_periods: int
    rationale: str
    warnings: tuple[str, ...]
    half_life_periods: float | None
    turnover_context: float | None


def recommend_rebalance_cadence(
    *,
    half_life_periods: float | None,
    turnover_context: float | None = None,
    min_periods: int = 1,
    max_periods: int = 63,
) -> RebalanceRecommendation:
    """Map decay half-life + turnover context to a practical cadence."""
    if min_periods <= 0:
        raise ValueError("min_periods must be > 0")
    if max_periods < min_periods:
        raise ValueError("max_periods must be >= min_periods")
    if turnover_context is not None and turnover_context < 0:
        raise ValueError("turnover_context must be >= 0")

    warnings: list[str] = []
    if half_life_periods is None or not np.isfinite(half_life_periods):
        cadence = 5
        rationale = "half_life unavailable; fallback to weekly review cadence"
    else:
        hl = float(half_life_periods)
        cadence = int(round(max(min_periods, min(max_periods, hl / 2.0))))
        rationale = (
            "cadence set to about half of estimated signal half-life "
            "to balance freshness and turnover"
        )
        if hl < 3:
            warnings.append("signal_half_life_is_very_short")
        if hl > 40:
            warnings.append("signal_half_life_is_slow_moving")

    if turnover_context is not None:
        if turnover_context > 4.0 and cadence <= 3:
            warnings.append("high_turnover_and_fast_cadence_execution_risk")
        if turnover_context > 2.0 and cadence <= 5:
            warnings.append("turnover_is_high_consider_trade_buffer")

    label = _cadence_label(cadence)
    return RebalanceRecommendation(
        cadence_label=label,
        cadence_periods=int(cadence),
        rationale=rationale,
        warnings=tuple(warnings),
        half_life_periods=half_life_periods,
        turnover_context=turnover_context,
    )


def recommend_rebalance_from_decay_profile(
    decay_profile: pd.DataFrame,
    *,
    turnover_context: float | None = None,
    min_periods: int = 1,
    max_periods: int = 63,
) -> RebalanceRecommendation:
    """Estimate half-life from decay profile then recommend cadence."""
    hl = estimate_half_life(decay_profile)
    return recommend_rebalance_cadence(
        half_life_periods=hl if np.isfinite(hl) else None,
        turnover_context=turnover_context,
        min_periods=min_periods,
        max_periods=max_periods,
    )


def _cadence_label(periods: int) -> str:
    if periods <= 2:
        return "daily_or_2day"
    if periods <= 7:
        return "weekly"
    if periods <= 15:
        return "biweekly"
    if periods <= 31:
        return "monthly"
    return "slow_monthly_plus"
