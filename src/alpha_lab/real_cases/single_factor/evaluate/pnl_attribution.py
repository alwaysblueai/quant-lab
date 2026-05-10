from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from alpha_lab.experiment import ExperimentResult
from alpha_lab.research_evaluation_config import (
    ResearchEvaluationConfig,
)
from alpha_lab.splits import TimeSeriesSplitContract

# Cross-module imports (auto-added by split)
from ._utils import _evaluate_variant_lightweight, _resolve_variant_label_df


def _merge_daily_pnl_attribution_metrics(
    metrics: dict[str, object],
    *,
    result: ExperimentResult,
    cost_rate: float,
) -> pd.DataFrame:
    """Decompose the headline long-short return stream into long-leg, short-leg,
    and transaction-cost drag on a per-date basis.

    Headline ``long_short_ir`` answers "how profitable is the long-short
    portfolio net of costs" — it does not answer "which leg carries the
    weight" or "how much of the gross Sharpe is eaten by turnover."  The
    summary statistics here make both visible.  This is the minimum needed
    to tell a pure-short signal apart from a balanced long-short signal, and
    to tell a high-turnover signal apart from a signal that actually holds
    its positions.
    """
    metrics["daily_pnl_n_dates"] = 0
    metrics["daily_pnl_long_leg_mean"] = float("nan")
    metrics["daily_pnl_short_leg_mean"] = float("nan")
    metrics["daily_pnl_gross_mean"] = float("nan")
    metrics["daily_pnl_cost_drag_mean"] = float("nan")
    metrics["daily_pnl_net_mean"] = float("nan")
    metrics["daily_pnl_long_contribution_ratio"] = float("nan")
    metrics["daily_pnl_cost_drag_share"] = float("nan")
    metrics["daily_pnl_worst_day_net"] = float("nan")
    metrics["daily_pnl_best_day_net"] = float("nan")
    empty_frame = pd.DataFrame(
        columns=["date", "long_leg", "short_leg", "gross", "cost_drag", "net"]
    )

    qr = result.quantile_returns_df
    if qr is None or qr.empty:
        return empty_frame

    per_bucket = (
        qr[["date", "quantile", "mean_return"]].dropna(subset=["quantile", "mean_return"]).copy()
    )
    if per_bucket.empty:
        return empty_frame
    per_bucket["quantile"] = pd.to_numeric(per_bucket["quantile"], errors="coerce")
    per_bucket = per_bucket.dropna(subset=["quantile"])
    if per_bucket.empty:
        return empty_frame
    per_bucket["quantile"] = per_bucket["quantile"].astype(int)
    per_bucket = (
        per_bucket.groupby(["date", "quantile"], sort=True, as_index=False)["mean_return"]
        .mean()
        .sort_values(["date", "quantile"], kind="mergesort")
    )

    grouped = per_bucket.groupby("date", sort=True, group_keys=False)
    long_leg = (
        grouped.tail(1)
        .set_index("date")[["quantile", "mean_return"]]
        .rename(columns={"quantile": "q_max", "mean_return": "long_leg"})
    )
    short_leg = (
        grouped.head(1)
        .set_index("date")[["quantile", "mean_return"]]
        .rename(columns={"quantile": "q_min", "mean_return": "short_leg"})
    )
    n_q = per_bucket.groupby("date", sort=True)["quantile"].nunique()
    merged = (
        long_leg.join(short_leg)
        .join(n_q.rename("n_q"))
        .reset_index()
        .sort_values("date", kind="mergesort")
        .reset_index(drop=True)
    )
    merged = merged.loc[merged["n_q"] >= 2, ["date", "long_leg", "short_leg"]]
    if merged.empty:
        return empty_frame
    merged["gross"] = merged["long_leg"] - merged["short_leg"]

    turnover = result.long_short_turnover_df
    if cost_rate > 0.0 and turnover is not None and not turnover.empty:
        cost_series = turnover[["date", "long_short_turnover"]].rename(
            columns={"long_short_turnover": "turnover"}
        )
        merged = merged.merge(cost_series, on="date", how="left", validate="one_to_one")
        merged["cost_drag"] = merged["turnover"] * float(cost_rate)
    else:
        merged["cost_drag"] = 0.0
    merged["net"] = merged["gross"] - merged["cost_drag"]

    n_rows = int(len(merged))
    if n_rows == 0:
        return empty_frame

    gross_mean = float(merged["gross"].mean())
    long_mean = float(merged["long_leg"].mean())
    short_mean = float(merged["short_leg"].mean())
    cost_mean = float(merged["cost_drag"].mean())
    net_mean = float(merged["net"].mean())

    metrics["daily_pnl_n_dates"] = n_rows
    metrics["daily_pnl_long_leg_mean"] = long_mean
    metrics["daily_pnl_short_leg_mean"] = short_mean
    metrics["daily_pnl_gross_mean"] = gross_mean
    metrics["daily_pnl_cost_drag_mean"] = cost_mean
    metrics["daily_pnl_net_mean"] = net_mean
    if np.isfinite(gross_mean) and abs(gross_mean) > 0.0:
        metrics["daily_pnl_long_contribution_ratio"] = long_mean / gross_mean
        metrics["daily_pnl_cost_drag_share"] = cost_mean / abs(gross_mean)
    metrics["daily_pnl_worst_day_net"] = float(merged["net"].min())
    metrics["daily_pnl_best_day_net"] = float(merged["net"].max())
    return merged[["date", "long_leg", "short_leg", "gross", "cost_drag", "net"]].reset_index(
        drop=True
    )


def _merge_signal_lag_sensitivity_metrics(
    metrics: dict[str, object],
    *,
    prices: pd.DataFrame,
    factor_df: pd.DataFrame,
    horizon: int,
    n_quantiles: int,
    evaluation_config: ResearchEvaluationConfig,
    label_df: pd.DataFrame | None = None,
    base_result: ExperimentResult | None = None,
    label_fn: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    split_contract: TimeSeriesSplitContract | None = None,
    lags: tuple[int, ...] = (0, 1, 2, 3),
    enabled: bool = True,
) -> pd.DataFrame:
    """Measure IC/IR when the signal is executed ``k`` days late.

    A short-horizon price-volume factor that looks great at lag 0 but
    collapses at lag 1 or 2 is exploiting information that the trader cannot
    realistically act on (e.g. same-bar close or open).  The per-lag fields
    here quantify how durable the signal is under realistic execution
    latency.
    """
    for lag in lags:
        metrics[f"lag_sensitivity_mean_ic_lag_{lag}"] = float("nan")
        metrics[f"lag_sensitivity_long_short_ir_lag_{lag}"] = float("nan")
    metrics["lag_sensitivity_lags"] = tuple(int(lag) for lag in lags)
    metrics["lag_sensitivity_ic_decay_lag_1"] = float("nan")
    rows: list[dict[str, object]] = []
    if not enabled:
        return pd.DataFrame(columns=["lag", "mean_ic", "long_short_ir"])

    if factor_df.empty:
        return pd.DataFrame(columns=["lag", "mean_ic", "long_short_ir"])

    resolved_label_df = _resolve_variant_label_df(
        prices=prices,
        horizon=horizon,
        label_fn=label_fn,
        label_df=label_df,
    )
    if split_contract is not None:
        oos_start = split_contract.oos_start
        factor_df = factor_df[pd.to_datetime(factor_df["date"]) >= oos_start].reset_index(drop=True)
        resolved_label_df = resolved_label_df[
            pd.to_datetime(resolved_label_df["date"]) >= oos_start
        ].reset_index(drop=True)
        if factor_df.empty or resolved_label_df.empty:
            return pd.DataFrame(columns=["lag", "mean_ic", "long_short_ir"])
    sorted_factor = factor_df.sort_values(["asset", "date"], kind="mergesort")
    for lag in lags:
        if lag == 0:
            if base_result is not None:
                mic = float(base_result.summary.mean_ic)
                ir = float(base_result.summary.long_short_ir)
                metrics[f"lag_sensitivity_mean_ic_lag_{lag}"] = mic
                metrics[f"lag_sensitivity_long_short_ir_lag_{lag}"] = ir
                rows.append({"lag": int(lag), "mean_ic": mic, "long_short_ir": ir})
                continue
            shifted = factor_df
        else:
            shifted = sorted_factor.copy()
            shifted["value"] = shifted.groupby("asset", sort=False)["value"].shift(int(lag))
            shifted = shifted.dropna(subset=["value"])
            if shifted.empty:
                continue
        try:
            variant = _evaluate_variant_lightweight(
                factor_df=shifted,
                label_df=resolved_label_df,
                n_quantiles=n_quantiles,
            )
        except Exception:
            continue
        mic = float(variant.mean_ic)
        ir = float(variant.long_short_ir)
        metrics[f"lag_sensitivity_mean_ic_lag_{lag}"] = mic
        metrics[f"lag_sensitivity_long_short_ir_lag_{lag}"] = ir
        rows.append({"lag": int(lag), "mean_ic": mic, "long_short_ir": ir})

    ic0 = metrics.get("lag_sensitivity_mean_ic_lag_0")
    ic1 = metrics.get("lag_sensitivity_mean_ic_lag_1")
    if (
        isinstance(ic0, float)
        and isinstance(ic1, float)
        and np.isfinite(ic0)
        and np.isfinite(ic1)
        and abs(ic0) > 0.0
    ):
        metrics["lag_sensitivity_ic_decay_lag_1"] = ic1 / ic0
    if rows:
        return pd.DataFrame(rows, columns=["lag", "mean_ic", "long_short_ir"])
    return pd.DataFrame(columns=["lag", "mean_ic", "long_short_ir"])
