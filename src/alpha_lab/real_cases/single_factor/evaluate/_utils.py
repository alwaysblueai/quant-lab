from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from alpha_lab.evaluation import (
    compute_ic,
    compute_ic_summary,
)
from alpha_lab.labels import forward_return
from alpha_lab.quantile import long_short_return, quantile_returns

_DUAL_SCOPE_ROW_METRIC_KEYS: tuple[str, ...] = (
    "mean_ic",
    "mean_ic_ci_lower",
    "mean_ic_ci_upper",
    "mean_rank_ic",
    "mean_rank_ic_ci_lower",
    "mean_rank_ic_ci_upper",
    "mean_mutual_information",
    "mutual_information_ir",
    "mutual_information_positive_rate",
    "mutual_information_valid_ratio",
    "ic_ir",
    "ic_t_stat",
    "ic_p_value",
    "ic_positive_rate",
    "rank_ic_positive_rate",
    "ic_valid_ratio",
    "rank_ic_valid_ratio",
    "mean_long_short_return",
    "mean_long_short_return_ci_lower",
    "mean_long_short_return_ci_upper",
    "long_short_ir",
    "long_short_hit_rate",
    "long_short_return_per_turnover",
    "subperiod_ic_positive_share",
    "subperiod_long_short_positive_share",
    "subperiod_ic_min_mean",
    "subperiod_long_short_min_mean",
    "rolling_ic_positive_share",
    "rolling_rank_ic_positive_share",
    "rolling_long_short_positive_share",
    "rolling_ic_min_mean",
    "rolling_rank_ic_min_mean",
    "rolling_long_short_min_mean",
    "n_dates_used",
    "mean_long_short_turnover",
    "mean_eval_assets_per_date",
    "min_eval_assets_per_date",
    "eval_coverage_ratio_mean",
    "eval_coverage_ratio_min",
    "mean_cost_adjusted_long_short_return",
)


def _safe_ratio(numerator: float | None, denominator: float | None) -> float:
    if numerator is None or denominator is None:
        return float("nan")
    if not np.isfinite(numerator) or not np.isfinite(denominator):
        return float("nan")
    if abs(denominator) <= 1e-12:
        return float("nan")
    return float(numerator / denominator)


def _finite_series_mean(values: pd.Series) -> float:
    arr = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(arr.mean()) if not arr.empty else float("nan")


def _finite_or_nan(value: float | None) -> float:
    if value is None or not np.isfinite(value):
        return float("nan")
    return float(value)


def _parse_flags(value: object) -> list[str]:
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    return [token.strip() for token in text.split(";") if token.strip()]


def _parse_optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        numeric = float(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            numeric = float(text)
        except ValueError:
            return None
    else:
        return None
    if not np.isfinite(numeric):
        return None
    return int(numeric)


def _parse_optional_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        numeric = float(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            numeric = float(text)
        except ValueError:
            return None
    else:
        return None
    if not np.isfinite(numeric):
        return None
    return numeric


@dataclass(frozen=True)
class _LightweightVariantSummary:
    """Minimal variant summary for sensitivity diagnostics.

    These diagnostics only need the scalar edge metrics used in gating and
    side-by-side comparisons, so they should not pay the cost of a full
    ``run_factor_experiment`` invocation.
    """

    mean_ic: float
    mean_long_short_return: float
    long_short_ir: float


def _resolve_variant_label_df(
    *,
    prices: pd.DataFrame,
    horizon: int,
    label_fn: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    label_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if label_df is not None:
        return label_df
    if label_fn is not None:
        return label_fn(prices)
    return forward_return(prices, horizon=horizon)


def _evaluate_variant_lightweight(
    *,
    factor_df: pd.DataFrame,
    label_df: pd.DataFrame,
    n_quantiles: int,
) -> _LightweightVariantSummary:
    """Compute only the scalar metrics needed by variant diagnostics."""
    if factor_df.empty or label_df.empty:
        return _LightweightVariantSummary(
            mean_ic=float("nan"),
            mean_long_short_return=float("nan"),
            long_short_ir=float("nan"),
        )

    merged_eval = (
        factor_df[["date", "asset", "value"]]
        .rename(columns={"value": "value_factor"})
        .merge(
            label_df[["date", "asset", "value"]].rename(columns={"value": "value_label"}),
            on=["date", "asset"],
            how="inner",
            validate="one_to_one",
        )
    )
    ic_df = compute_ic(factor_df, label_df, merged_pairs=merged_eval)
    ic_summary = compute_ic_summary(pd.to_numeric(ic_df["ic"], errors="coerce"))

    quantile_df = quantile_returns(
        factor_df,
        label_df,
        n_quantiles=n_quantiles,
        merged_pairs=merged_eval,
    )
    long_short_df = long_short_return(quantile_df)
    long_short_values = pd.to_numeric(
        long_short_df["long_short_return"],
        errors="coerce",
    ).dropna()
    mean_long_short_return = (
        float(long_short_values.mean()) if len(long_short_values) > 0 else float("nan")
    )
    long_short_std = (
        float(long_short_values.std(ddof=1)) if len(long_short_values) > 1 else float("nan")
    )
    if not np.isfinite(long_short_std) or long_short_std <= 0.0:
        long_short_ir = float("nan")
    else:
        long_short_ir = mean_long_short_return / long_short_std

    return _LightweightVariantSummary(
        mean_ic=float(ic_summary["mean_ic"]),
        mean_long_short_return=mean_long_short_return,
        long_short_ir=long_short_ir,
    )


def _metric_label(value: str) -> str:
    label = "".join(ch.lower() if ch.isalnum() else "_" for ch in value.strip())
    return "_".join(part for part in label.split("_") if part) or "unknown"
