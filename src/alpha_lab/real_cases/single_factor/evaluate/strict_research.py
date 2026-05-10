from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_lab.experiment import ExperimentResult
from alpha_lab.splits import TimeSeriesSplitContract

# Cross-module imports (auto-added by split)
from ._utils import (
    _DUAL_SCOPE_ROW_METRIC_KEYS,
    _finite_series_mean,
    _metric_label,
    _parse_optional_float,
    _safe_ratio,
)
from .summary_metrics import _compute_cost_adjusted_long_short_ir, _compute_rank_ic_ir


def _merge_dual_scope_report_metrics(
    metrics: dict[str, object],
    *,
    oos_result: ExperimentResult,
    full_result: ExperimentResult | None,
    is_result: ExperimentResult | None,
    oos_row: pd.Series,
    full_row: pd.Series | None,
    is_row: pd.Series | None,
    cost_rate: float,
) -> None:
    """Expose full-sample report metrics while preserving OOS gate metrics."""

    metrics["split_semantics"] = "factor_time_series_holdout"
    metrics["split_semantics_label"] = "Alpha-Lab: IS/OOS = 因子时序样本内/外"
    metrics["metric_scope"] = "oos" if full_result is not None else "full_sample"
    metrics["primary_metric_scope"] = metrics["metric_scope"]
    if full_result is None or full_row is None:
        metrics["report_metric_scope"] = "full_sample"
        metrics["report_timeseries_scope"] = "full_sample"
        return

    metrics["report_metric_scope"] = "full_sample_with_oos_parentheses"
    metrics["report_timeseries_scope"] = "full_path_split_by_phase"
    metrics["report_split_phase_column"] = "split_phase"
    for key in _DUAL_SCOPE_ROW_METRIC_KEYS:
        _copy_scoped_row_metric(metrics, key=key, scope="full", row=full_row)
        _copy_scoped_row_metric(metrics, key=key, scope="oos", row=oos_row)
        if is_row is not None:
            _copy_scoped_row_metric(metrics, key=key, scope="is", row=is_row)

    metrics["rank_ic_ir_full"] = _compute_rank_ic_ir(full_result.rank_ic_df)
    metrics["rank_ic_ir_oos"] = _compute_rank_ic_ir(oos_result.rank_ic_df)
    if is_result is not None:
        metrics["rank_ic_ir_is"] = _compute_rank_ic_ir(is_result.rank_ic_df)

    metrics["ls_max_drawdown_full"] = float(full_result.summary.ls_max_drawdown)
    metrics["ls_max_drawdown_oos"] = float(oos_result.summary.ls_max_drawdown)
    metrics["max_drawdown_full"] = float(full_result.summary.ls_max_drawdown)
    metrics["max_drawdown_oos"] = float(oos_result.summary.ls_max_drawdown)
    metrics["ls_max_drawdown_duration_full"] = int(full_result.summary.ls_max_drawdown_duration)
    metrics["ls_max_drawdown_duration_oos"] = int(oos_result.summary.ls_max_drawdown_duration)
    metrics["ls_max_consecutive_loss_days_full"] = int(
        full_result.summary.ls_max_consecutive_loss_days
    )
    metrics["ls_max_consecutive_loss_days_oos"] = int(
        oos_result.summary.ls_max_consecutive_loss_days
    )
    metrics["ls_var_5_full"] = float(full_result.summary.ls_var_5)
    metrics["ls_var_5_oos"] = float(oos_result.summary.ls_var_5)
    metrics["ls_cvar_5_full"] = float(full_result.summary.ls_cvar_5)
    metrics["ls_cvar_5_oos"] = float(oos_result.summary.ls_cvar_5)
    metrics["ls_calmar_ratio_full"] = float(full_result.summary.ls_calmar_ratio)
    metrics["ls_calmar_ratio_oos"] = float(oos_result.summary.ls_calmar_ratio)
    if is_result is not None:
        metrics["ls_max_drawdown_is"] = float(is_result.summary.ls_max_drawdown)
        metrics["max_drawdown_is"] = float(is_result.summary.ls_max_drawdown)
        metrics["ls_max_drawdown_duration_is"] = int(is_result.summary.ls_max_drawdown_duration)
        metrics["ls_max_consecutive_loss_days_is"] = int(
            is_result.summary.ls_max_consecutive_loss_days
        )
        metrics["ls_var_5_is"] = float(is_result.summary.ls_var_5)
        metrics["ls_cvar_5_is"] = float(is_result.summary.ls_cvar_5)
        metrics["ls_calmar_ratio_is"] = float(is_result.summary.ls_calmar_ratio)

    cost_ir_full = _compute_cost_adjusted_long_short_ir(
        long_short_df=full_result.long_short_df,
        long_short_turnover_df=full_result.long_short_turnover_df,
        cost_rate=cost_rate,
    )
    cost_ir_oos = _compute_cost_adjusted_long_short_ir(
        long_short_df=oos_result.long_short_df,
        long_short_turnover_df=oos_result.long_short_turnover_df,
        cost_rate=cost_rate,
    )
    metrics["cost_aware_long_short_ir_full"] = (
        cost_ir_full if np.isfinite(cost_ir_full) else float(full_row["long_short_ir"])
    )
    metrics["cost_aware_long_short_ir_oos"] = (
        cost_ir_oos if np.isfinite(cost_ir_oos) else float(oos_row["long_short_ir"])
    )
    if is_result is not None and is_row is not None:
        cost_ir_is = _compute_cost_adjusted_long_short_ir(
            long_short_df=is_result.long_short_df,
            long_short_turnover_df=is_result.long_short_turnover_df,
            cost_rate=cost_rate,
        )
        metrics["cost_aware_long_short_ir_is"] = (
            cost_ir_is if np.isfinite(cost_ir_is) else float(is_row["long_short_ir"])
        )

    for key in (
        "mean_ic",
        "mean_rank_ic",
        "ic_ir",
        "rank_ic_ir",
        "mean_long_short_return",
        "long_short_ir",
        "mean_cost_adjusted_long_short_return",
        "cost_aware_long_short_ir",
        "max_drawdown",
    ):
        _copy_oos_decay_ratio(metrics, key=key)


def _copy_scoped_row_metric(
    metrics: dict[str, object],
    *,
    key: str,
    scope: str,
    row: pd.Series,
) -> None:
    if key not in row:
        return
    value = row[key]
    if isinstance(value, np.generic):
        value = value.item()
    metrics[f"{key}_{scope}"] = value


def _copy_oos_decay_ratio(metrics: dict[str, object], *, key: str) -> None:
    is_value = _parse_optional_float(metrics.get(f"{key}_is"))
    oos_value = _parse_optional_float(metrics.get(f"{key}_oos"))
    metrics[f"{key}_oos_decay_ratio"] = _safe_ratio(oos_value, is_value)


def _merge_strict_research_evidence_metrics(
    metrics: dict[str, object],
    *,
    full_result: ExperimentResult | None,
    contract: TimeSeriesSplitContract | None,
    enabled: bool,
) -> None:
    if not enabled:
        return

    metrics["strict_research_evidence"] = "enabled"
    metrics["strict_research_evidence_items"] = [
        "bootstrap_uncertainty",
        "post_split_gap_scan",
        "subsample_stability",
        "regime_segment",
    ]
    if full_result is None:
        metrics["strict_research_evidence"] = "missing_full_path"
        return

    rank_ic = _dated_numeric_series(full_result.rank_ic_df, "rank_ic")
    long_short = _dated_numeric_series(full_result.long_short_df, "long_short_return")
    _merge_strict_bootstrap_ir_metrics(
        metrics,
        rank_ic=rank_ic,
        long_short=long_short,
    )
    _merge_strict_subsample_metrics(metrics, rank_ic=rank_ic, long_short=long_short)
    if contract is not None:
        _merge_strict_gap_scan_metrics(
            metrics,
            rank_ic=rank_ic,
            contract=contract,
            gaps=(5, 10, 20),
        )
    _merge_strict_regime_metrics(metrics, result=full_result)


def _dated_numeric_series(frame: pd.DataFrame, value_col: str) -> pd.DataFrame:
    if frame.empty or "date" not in frame.columns or value_col not in frame.columns:
        return pd.DataFrame(columns=["date", value_col])
    out = frame[["date", value_col]].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out[value_col] = pd.to_numeric(out[value_col], errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["date", value_col])
    return out.sort_values("date", kind="mergesort").reset_index(drop=True)


def _merge_strict_subsample_metrics(
    metrics: dict[str, object],
    *,
    rank_ic: pd.DataFrame,
    long_short: pd.DataFrame,
) -> None:
    _merge_ordered_half_metric(
        metrics,
        prefix="strict_subsample_rank_ic",
        frame=rank_ic,
        value_col="rank_ic",
    )
    _merge_ordered_half_metric(
        metrics,
        prefix="strict_subsample_long_short_return",
        frame=long_short,
        value_col="long_short_return",
    )
    _merge_odd_even_year_metric(
        metrics,
        prefix="strict_subsample_rank_ic",
        frame=rank_ic,
        value_col="rank_ic",
    )
    _merge_odd_even_year_metric(
        metrics,
        prefix="strict_subsample_long_short_return",
        frame=long_short,
        value_col="long_short_return",
    )


def _merge_strict_bootstrap_ir_metrics(
    metrics: dict[str, object],
    *,
    rank_ic: pd.DataFrame,
    long_short: pd.DataFrame,
) -> None:
    rank_lower, rank_upper = _bootstrap_ir_ci(rank_ic["rank_ic"] if "rank_ic" in rank_ic else [])
    ls_lower, ls_upper = _bootstrap_ir_ci(
        long_short["long_short_return"] if "long_short_return" in long_short else []
    )
    metrics["strict_bootstrap_rank_ic_ir_ci_lower"] = rank_lower
    metrics["strict_bootstrap_rank_ic_ir_ci_upper"] = rank_upper
    metrics["strict_bootstrap_long_short_ir_ci_lower"] = ls_lower
    metrics["strict_bootstrap_long_short_ir_ci_upper"] = ls_upper
    metrics["strict_bootstrap_ir_confidence_level"] = 0.99
    metrics["strict_bootstrap_ir_resamples"] = 800


def _bootstrap_ir_ci(
    values: pd.Series | list[float],
    *,
    n_resamples: int = 800,
) -> tuple[float, float]:
    arr = pd.to_numeric(pd.Series(values), errors="coerce")
    arr = arr.replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    if arr.size < 3:
        return float("nan"), float("nan")
    rng = np.random.default_rng(20260429)
    samples = rng.choice(arr, size=(int(n_resamples), arr.size), replace=True)
    means = samples.mean(axis=1)
    stds = samples.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ir_values = means / stds
    ir_values = ir_values[np.isfinite(ir_values)]
    if ir_values.size == 0:
        return float("nan"), float("nan")
    lower, upper = np.quantile(ir_values, [0.005, 0.995])
    return float(lower), float(upper)


def _merge_ordered_half_metric(
    metrics: dict[str, object],
    *,
    prefix: str,
    frame: pd.DataFrame,
    value_col: str,
) -> None:
    if frame.empty:
        metrics[f"{prefix}_first_half_mean"] = float("nan")
        metrics[f"{prefix}_second_half_mean"] = float("nan")
        metrics[f"{prefix}_second_vs_first_ratio"] = float("nan")
        return
    ordered_dates = pd.Index(frame["date"].drop_duplicates().sort_values())
    split_at = max(1, int(len(ordered_dates) // 2))
    first_dates = set(ordered_dates[:split_at])
    first = frame.loc[frame["date"].isin(first_dates), value_col]
    second = frame.loc[~frame["date"].isin(first_dates), value_col]
    first_mean = _finite_series_mean(first)
    second_mean = _finite_series_mean(second)
    metrics[f"{prefix}_first_half_mean"] = first_mean
    metrics[f"{prefix}_second_half_mean"] = second_mean
    metrics[f"{prefix}_second_vs_first_ratio"] = _safe_ratio(second_mean, first_mean)


def _merge_odd_even_year_metric(
    metrics: dict[str, object],
    *,
    prefix: str,
    frame: pd.DataFrame,
    value_col: str,
) -> None:
    if frame.empty:
        metrics[f"{prefix}_odd_year_mean"] = float("nan")
        metrics[f"{prefix}_even_year_mean"] = float("nan")
        metrics[f"{prefix}_odd_even_spread"] = float("nan")
        return
    years = frame["date"].dt.year
    odd = _finite_series_mean(frame.loc[(years % 2) == 1, value_col])
    even = _finite_series_mean(frame.loc[(years % 2) == 0, value_col])
    metrics[f"{prefix}_odd_year_mean"] = odd
    metrics[f"{prefix}_even_year_mean"] = even
    metrics[f"{prefix}_odd_even_spread"] = (
        float(odd - even) if np.isfinite(odd) and np.isfinite(even) else float("nan")
    )


def _merge_strict_gap_scan_metrics(
    metrics: dict[str, object],
    *,
    rank_ic: pd.DataFrame,
    contract: TimeSeriesSplitContract,
    gaps: tuple[int, ...],
) -> None:
    if rank_ic.empty:
        return
    oos_start = pd.Timestamp(contract.oos_start)
    is_mask = rank_ic["date"] <= pd.Timestamp(contract.is_end)
    is_mean = _finite_series_mean(rank_ic.loc[is_mask, "rank_ic"])
    oos_rank_ic = rank_ic.loc[rank_ic["date"] >= oos_start].reset_index(drop=True)
    for gap in gaps:
        suffix = f"gap_{int(gap)}"
        sliced = oos_rank_ic.iloc[int(gap) :] if int(gap) > 0 else oos_rank_ic
        value = _finite_series_mean(sliced["rank_ic"])
        metrics[f"strict_post_split_rank_ic_{suffix}_mean"] = value
        metrics[f"strict_post_split_rank_ic_{suffix}_n_dates"] = int(
            sliced["rank_ic"].dropna().shape[0]
        )
        metrics[f"strict_post_split_rank_ic_{suffix}_retention_vs_is"] = _safe_ratio(
            value,
            is_mean,
        )


def _merge_strict_regime_metrics(
    metrics: dict[str, object],
    *,
    result: ExperimentResult,
) -> None:
    summary = result.regime_summary
    if summary is None:
        metrics["strict_regime_flags"] = []
        return
    metrics["strict_regime_flags"] = list(summary.regime_flags)
    for prefix, items in (
        ("direction", summary.direction_regimes),
        ("volatility", summary.volatility_regimes),
    ):
        for item in items:
            label = _metric_label(str(item.regime_label))
            base = f"strict_regime_{prefix}_{label}"
            metrics[f"{base}_n_dates"] = int(item.n_dates)
            metrics[f"{base}_mean_ic"] = float(item.mean_ic)
            metrics[f"{base}_mean_rank_ic"] = float(item.mean_rank_ic)
            metrics[f"{base}_mean_long_short_return"] = float(item.mean_long_short_return)
            metrics[f"{base}_long_short_hit_rate"] = float(item.long_short_hit_rate)
