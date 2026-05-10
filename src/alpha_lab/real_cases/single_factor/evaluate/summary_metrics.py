from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from alpha_lab.costs import cost_adjusted_long_short
from alpha_lab.research_evaluation_config import (
    ResearchEvaluationConfig,
)

# Cross-module imports (auto-added by split)
from ._utils import _parse_optional_float


def _build_decay_horizons(target_horizon: int) -> tuple[int, ...]:
    base_horizons = {1, 2, 3, 5, 10, 20}
    if target_horizon > 0:
        base_horizons.add(int(target_horizon))
    return tuple(sorted(base_horizons))


def _build_autocorr_lags() -> tuple[int, ...]:
    return (1, 2, 3, 5, 10)


def _resolve_rebalance_step(rebalance_frequency: str) -> int:
    text = rebalance_frequency.strip().upper()
    if not text:
        return 1
    try:
        explicit = int(text)
    except ValueError:
        explicit = 0
    if explicit > 0:
        return explicit
    mapping = {
        "D": 1,
        "DAILY": 1,
        "W": 5,
        "WEEKLY": 5,
        "M": 21,
        "MONTHLY": 21,
        "Q": 63,
        "QUARTERLY": 63,
    }
    return mapping.get(text, 1)


def _compute_ic_decay_rebalance_consistency(
    *,
    rebalance_step: int,
    half_life_horizon: object,
    thresholds: ResearchEvaluationConfig,
) -> tuple[float, bool]:
    half_life = _parse_optional_float(half_life_horizon)
    if half_life is None or half_life <= 0.0:
        return float("nan"), False
    ratio = float(rebalance_step / half_life)
    return ratio, bool(ratio > thresholds.factor_verdict.ic_decay_warn_rebalance_ratio)


def _compute_rank_ic_ir(rank_ic_df: pd.DataFrame) -> float:
    """Compute mean/std IR over daily RankIC time series."""
    if rank_ic_df.empty or "rank_ic" not in rank_ic_df.columns:
        return float("nan")
    series = pd.to_numeric(rank_ic_df["rank_ic"], errors="coerce").dropna()
    if len(series) <= 1:
        return float("nan")
    std = float(series.std(ddof=1))
    if not np.isfinite(std) or std <= 0.0:
        return float("nan")
    return float(series.mean() / std)


def _compute_cost_adjusted_long_short_ir(
    *,
    long_short_df: pd.DataFrame,
    long_short_turnover_df: pd.DataFrame,
    cost_rate: float,
) -> float:
    """Compute IR on net long-short returns after linear transaction costs."""
    adjusted = cost_adjusted_long_short(
        long_short_df=long_short_df,
        long_short_turnover_df=long_short_turnover_df,
        cost_rate=cost_rate,
    )
    if adjusted.empty or "adjusted_return" not in adjusted.columns:
        return float("nan")
    series = pd.to_numeric(adjusted["adjusted_return"], errors="coerce").dropna()
    if len(series) <= 1:
        return float("nan")
    std = float(series.std(ddof=1))
    if not np.isfinite(std) or std <= 0.0:
        return float("nan")
    return float(series.mean() / std)


def _build_group_monotonicity_summary(
    *,
    group_returns: pd.DataFrame,
    n_quantiles: int,
) -> dict[str, object]:
    """Return Q-top minus Q-bottom spread and monotonic share summary."""
    if group_returns.empty:
        return {
            "qtop_qbottom_spread_mean": float("nan"),
            "monotonic_share": float("nan"),
            "summary": "unavailable",
        }
    required = {"date", "group", "group_return"}
    if not required.issubset(group_returns.columns):
        return {
            "qtop_qbottom_spread_mean": float("nan"),
            "monotonic_share": float("nan"),
            "summary": "unavailable",
        }
    frame = group_returns.loc[:, ["date", "group", "group_return"]].copy()
    frame["group"] = pd.to_numeric(frame["group"], errors="coerce")
    frame["group_return"] = pd.to_numeric(frame["group_return"], errors="coerce")
    frame = frame.dropna(subset=["group", "group_return"])
    if frame.empty:
        return {
            "qtop_qbottom_spread_mean": float("nan"),
            "monotonic_share": float("nan"),
            "summary": "unavailable",
        }
    pivot = frame.pivot_table(
        index="date",
        columns="group",
        values="group_return",
        aggfunc="mean",
    )
    quantile_cols = sorted(
        [int(col) for col in pivot.columns if np.isfinite(float(col))],
    )
    if len(quantile_cols) < 2:
        return {
            "qtop_qbottom_spread_mean": float("nan"),
            "monotonic_share": float("nan"),
            "summary": "unavailable",
        }
    ordered = pivot.loc[:, quantile_cols]
    monotonic_mask = ordered.diff(axis=1).iloc[:, 1:].ge(0.0).all(axis=1)
    monotonic_share = float(monotonic_mask.mean()) if len(monotonic_mask) > 0 else float("nan")
    top_group = max(quantile_cols)
    bottom_group = min(quantile_cols)
    spread_series = ordered[top_group] - ordered[bottom_group]
    spread_mean = float(spread_series.mean()) if len(spread_series) > 0 else float("nan")
    if np.isfinite(spread_mean) and np.isfinite(monotonic_share):
        summary = (
            f"Q{top_group}-Q{bottom_group}={spread_mean:.6f}; monotonic_share={monotonic_share:.1%}"
        )
    else:
        summary = "unavailable"
    if top_group != int(n_quantiles):
        summary = f"{summary}; observed_top_group=Q{top_group}"
    return {
        "qtop_qbottom_spread_mean": spread_mean,
        "monotonic_share": monotonic_share,
        "summary": summary,
    }


def _build_ic_decay_half_life_summary(decay_summary: Mapping[str, object]) -> str:
    horizon_raw = _parse_optional_float(decay_summary.get("ic_half_life_horizon"))
    status = str(decay_summary.get("ic_half_life_status") or "unavailable")
    if horizon_raw is None or not np.isfinite(horizon_raw):
        return f"status={status}"
    return f"half_life={horizon_raw:.2f}; status={status}"


def _compute_ic_decay_retention_ratio(
    decay_df: pd.DataFrame,
    *,
    horizon_short: int,
    horizon_long: int,
) -> float:
    """Compute IC retention ratio ``IC(horizon_long) / IC(horizon_short)``."""
    if decay_df.empty:
        return float("nan")
    if "horizon" not in decay_df.columns or "mean_ic" not in decay_df.columns:
        return float("nan")
    working = decay_df.loc[:, ["horizon", "mean_ic"]].copy()
    working["horizon"] = pd.to_numeric(working["horizon"], errors="coerce")
    working["mean_ic"] = pd.to_numeric(working["mean_ic"], errors="coerce")
    working = working.dropna(subset=["horizon", "mean_ic"])
    if working.empty:
        return float("nan")
    short_values = working.loc[
        working["horizon"] == float(int(horizon_short)),
        "mean_ic",
    ]
    long_values = working.loc[
        working["horizon"] == float(int(horizon_long)),
        "mean_ic",
    ]
    if len(short_values) == 0 or len(long_values) == 0:
        return float("nan")
    short_ic = float(short_values.mean())
    long_ic = float(long_values.mean())
    if not np.isfinite(short_ic) or abs(short_ic) <= 1e-12:
        return float("nan")
    if not np.isfinite(long_ic):
        return float("nan")
    return float(long_ic / short_ic)
