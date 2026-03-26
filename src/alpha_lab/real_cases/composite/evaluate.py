from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from alpha_lab.costs import cost_adjusted_long_short
from alpha_lab.experiment import ExperimentResult, run_factor_experiment
from alpha_lab.reporting import summarise_experiment_result

from .spec import CompositeCaseSpec


@dataclass(frozen=True)
class CompositeEvaluationResult:
    """Evaluation outputs and summary metrics for one composite run."""

    experiment_result: ExperimentResult
    metrics: dict[str, object]
    ic_timeseries: pd.DataFrame
    group_returns: pd.DataFrame
    turnover: pd.DataFrame
    exposure_summary: pd.DataFrame


def evaluate_composite_case(
    *,
    prices: pd.DataFrame,
    composite_factor: pd.DataFrame,
    spec: CompositeCaseSpec,
    coverage_by_date: pd.DataFrame,
    exposure_summary: pd.DataFrame | None,
) -> CompositeEvaluationResult:
    """Evaluate the composite factor using the canonical experiment pipeline."""

    result = run_factor_experiment(
        prices,
        lambda _prices: composite_factor.copy(),
        horizon=spec.target.horizon,
        n_quantiles=spec.n_quantiles,
    )

    cost_rate = spec.transaction_cost.one_way_rate
    summary_df = summarise_experiment_result(
        result,
        cost_rate=cost_rate if cost_rate > 0 else None,
    )
    row = summary_df.iloc[0]

    ic_timeseries = result.ic_df[["date", "ic"]].merge(
        result.rank_ic_df[["date", "rank_ic"]],
        on="date",
        how="outer",
        sort=True,
    )

    group_returns = result.quantile_returns_df.rename(
        columns={"quantile": "group", "mean_return": "group_return"}
    )
    turnover = result.long_short_turnover_df.rename(
        columns={"long_short_turnover": "turnover"}
    )

    exposure_df = (
        exposure_summary.copy()
        if exposure_summary is not None
        else pd.DataFrame(
            columns=[
                "exposure",
                "mean_abs_corr_before",
                "mean_abs_corr_after",
                "corr_reduction",
                "n_dates_used",
            ]
        )
    )

    if coverage_by_date.empty:
        coverage_mean = float("nan")
        coverage_min = float("nan")
    else:
        coverage_mean = float(coverage_by_date["composite_coverage"].mean())
        coverage_min = float(coverage_by_date["composite_coverage"].min())

    cost_adjusted_mean = float(row["mean_cost_adjusted_long_short_return"])
    if cost_rate > 0:
        # Keep this explicit for auditability and future extension to timeseries export.
        _ = cost_adjusted_long_short(
            result.long_short_df,
            result.long_short_turnover_df,
            cost_rate=cost_rate,
        )

    metrics: dict[str, object] = {
        "case_name": spec.name,
        "target_kind": spec.target.kind,
        "target_horizon": spec.target.horizon,
        "rebalance_frequency": spec.rebalance_frequency,
        "n_quantiles": spec.n_quantiles,
        "n_components": len(spec.components),
        "mean_ic": float(row["mean_ic"]),
        "mean_rank_ic": float(row["mean_rank_ic"]),
        "ic_ir": float(row["ic_ir"]),
        "mean_long_short_return": float(row["mean_long_short_return"]),
        "long_short_hit_rate": float(row["long_short_hit_rate"]),
        "mean_long_short_turnover": float(row["mean_long_short_turnover"]),
        "mean_cost_adjusted_long_short_return": cost_adjusted_mean,
        "transaction_cost_one_way_rate": cost_rate,
        "n_dates_used": int(row["n_dates_used"]),
        "coverage_mean": coverage_mean,
        "coverage_min": coverage_min,
        "missingness_mean": (
            float(1.0 - coverage_mean) if np.isfinite(coverage_mean) else float("nan")
        ),
    }

    return CompositeEvaluationResult(
        experiment_result=result,
        metrics=metrics,
        ic_timeseries=ic_timeseries.sort_values("date", kind="mergesort").reset_index(
            drop=True
        ),
        group_returns=group_returns.sort_values(
            ["date", "group"],
            kind="mergesort",
        ).reset_index(drop=True),
        turnover=turnover.sort_values("date", kind="mergesort").reset_index(drop=True),
        exposure_summary=exposure_df,
    )
