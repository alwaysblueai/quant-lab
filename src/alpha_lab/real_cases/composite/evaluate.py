from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from alpha_lab.costs import cost_adjusted_long_short
from alpha_lab.experiment import ExperimentResult, run_factor_experiment
from alpha_lab.reporting import summarise_experiment_result
from alpha_lab.reporting.campaign_triage import build_campaign_triage
from alpha_lab.reporting.factor_verdict import build_factor_verdict
from alpha_lab.reporting.level2_promotion import build_level2_promotion
from alpha_lab.reporting.neutralization_comparison import (
    NeutralizationComparison,
    build_raw_vs_neutralized_comparison,
)
from alpha_lab.research_evaluation_config import (
    DEFAULT_RESEARCH_EVALUATION_CONFIG,
    NeutralizationComparisonConfig,
    ResearchEvaluationConfig,
    research_evaluation_audit_snapshot,
)

from .spec import CompositeCaseSpec


@dataclass(frozen=True)
class CompositeEvaluationResult:
    """Evaluation outputs and summary metrics for one composite run."""

    experiment_result: ExperimentResult
    metrics: dict[str, object]
    ic_timeseries: pd.DataFrame
    rolling_stability: pd.DataFrame
    group_returns: pd.DataFrame
    turnover: pd.DataFrame
    exposure_summary: pd.DataFrame


def evaluate_composite_case(
    *,
    prices: pd.DataFrame,
    composite_factor: pd.DataFrame,
    raw_composite_factor: pd.DataFrame | None,
    spec: CompositeCaseSpec,
    coverage_by_date: pd.DataFrame,
    exposure_summary: pd.DataFrame | None,
    evaluation_config: ResearchEvaluationConfig = DEFAULT_RESEARCH_EVALUATION_CONFIG,
) -> CompositeEvaluationResult:
    """Evaluate the composite factor using the canonical experiment pipeline."""

    result = run_factor_experiment(
        prices,
        lambda _prices: composite_factor.copy(),
        horizon=spec.target.horizon,
        n_quantiles=spec.n_quantiles,
        rolling_stability_thresholds=evaluation_config.rolling_stability,
    )

    cost_rate = spec.transaction_cost.one_way_rate
    summary_df = summarise_experiment_result(
        result,
        cost_rate=cost_rate if cost_rate > 0 else None,
        evaluation_config=evaluation_config,
    )
    row = summary_df.iloc[0]

    raw_row: pd.Series | None = None
    if spec.neutralization.enabled and raw_composite_factor is not None:
        raw_result = run_factor_experiment(
            prices,
            lambda _prices: raw_composite_factor.copy(),
            horizon=spec.target.horizon,
            n_quantiles=spec.n_quantiles,
            rolling_stability_thresholds=evaluation_config.rolling_stability,
        )
        raw_summary_df = summarise_experiment_result(
            raw_result,
            cost_rate=cost_rate if cost_rate > 0 else None,
            evaluation_config=evaluation_config,
        )
        raw_row = raw_summary_df.iloc[0]

    ic_timeseries = result.ic_df[["date", "ic"]].merge(
        result.rank_ic_df[["date", "rank_ic"]],
        on="date",
        how="outer",
        sort=True,
    )
    rolling_stability = result.rolling_stability_df.copy()

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

    if exposure_df.empty:
        neutralization_mean_corr_reduction = float("nan")
        neutralization_min_corr_reduction = float("nan")
        neutralization_exposure_count = 0
    else:
        corr_reduction = pd.to_numeric(
            exposure_df["corr_reduction"],
            errors="coerce",
        ).dropna()
        neutralization_mean_corr_reduction = (
            float(corr_reduction.mean())
            if len(corr_reduction) > 0
            else float("nan")
        )
        neutralization_min_corr_reduction = (
            float(corr_reduction.min())
            if len(corr_reduction) > 0
            else float("nan")
        )
        neutralization_exposure_count = int(exposure_df["exposure"].nunique())

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
        "mean_ic_ci_lower": float(row["mean_ic_ci_lower"]),
        "mean_ic_ci_upper": float(row["mean_ic_ci_upper"]),
        "mean_rank_ic": float(row["mean_rank_ic"]),
        "mean_rank_ic_ci_lower": float(row["mean_rank_ic_ci_lower"]),
        "mean_rank_ic_ci_upper": float(row["mean_rank_ic_ci_upper"]),
        "ic_ir": float(row["ic_ir"]),
        "ic_positive_rate": float(row["ic_positive_rate"]),
        "rank_ic_positive_rate": float(row["rank_ic_positive_rate"]),
        "ic_valid_ratio": float(row["ic_valid_ratio"]),
        "rank_ic_valid_ratio": float(row["rank_ic_valid_ratio"]),
        "mean_long_short_return": float(row["mean_long_short_return"]),
        "mean_long_short_return_ci_lower": float(row["mean_long_short_return_ci_lower"]),
        "mean_long_short_return_ci_upper": float(row["mean_long_short_return_ci_upper"]),
        "long_short_ir": float(row["long_short_ir"]),
        "long_short_hit_rate": float(row["long_short_hit_rate"]),
        "long_short_return_per_turnover": float(row["long_short_return_per_turnover"]),
        "subperiod_ic_positive_share": float(row["subperiod_ic_positive_share"]),
        "subperiod_long_short_positive_share": float(
            row["subperiod_long_short_positive_share"]
        ),
        "subperiod_ic_min_mean": float(row["subperiod_ic_min_mean"]),
        "subperiod_long_short_min_mean": float(row["subperiod_long_short_min_mean"]),
        "rolling_window_size": int(row["rolling_window_size"]),
        "rolling_ic_positive_share": float(row["rolling_ic_positive_share"]),
        "rolling_rank_ic_positive_share": float(row["rolling_rank_ic_positive_share"]),
        "rolling_long_short_positive_share": float(row["rolling_long_short_positive_share"]),
        "rolling_ic_min_mean": float(row["rolling_ic_min_mean"]),
        "rolling_rank_ic_min_mean": float(row["rolling_rank_ic_min_mean"]),
        "rolling_long_short_min_mean": float(row["rolling_long_short_min_mean"]),
        "mean_long_short_turnover": float(row["mean_long_short_turnover"]),
        "mean_cost_adjusted_long_short_return": cost_adjusted_mean,
        "transaction_cost_one_way_rate": cost_rate,
        "n_dates_used": int(row["n_dates_used"]),
        "mean_eval_assets_per_date": float(row["mean_eval_assets_per_date"]),
        "min_eval_assets_per_date": float(row["min_eval_assets_per_date"]),
        "eval_coverage_ratio_mean": float(row["eval_coverage_ratio_mean"]),
        "eval_coverage_ratio_min": float(row["eval_coverage_ratio_min"]),
        "uncertainty_flags": _parse_flags(row["uncertainty_flags"]),
        "uncertainty_method": str(row["uncertainty_method"]),
        "uncertainty_confidence_level": float(row["uncertainty_confidence_level"]),
        "uncertainty_bootstrap_resamples": _parse_optional_int(
            row["uncertainty_bootstrap_resamples"]
        ),
        "uncertainty_bootstrap_block_length": _parse_optional_int(
            row.get("uncertainty_bootstrap_block_length")
        ),
        "rolling_instability_flags": _parse_flags(row["rolling_instability_flags"]),
        "instability_flags": _parse_flags(row["instability_flags"]),
        "coverage_mean": coverage_mean,
        "coverage_min": coverage_min,
        "missingness_mean": (
            float(1.0 - coverage_mean) if np.isfinite(coverage_mean) else float("nan")
        ),
        "neutralization_enabled": bool(spec.neutralization.enabled),
        "neutralization_exposure_count": neutralization_exposure_count,
        "neutralization_mean_corr_reduction": neutralization_mean_corr_reduction,
        "neutralization_min_corr_reduction": neutralization_min_corr_reduction,
        "research_evaluation_profile": evaluation_config.profile_name,
        "research_evaluation_snapshot": research_evaluation_audit_snapshot(
            evaluation_config
        ),
    }
    if raw_row is not None:
        comparison = _build_neutralization_comparison(
            raw_row=raw_row,
            neutralized_row=row,
            neutralization_mean_corr_reduction=neutralization_mean_corr_reduction,
            thresholds=evaluation_config.neutralization_comparison,
        )
        _merge_neutralization_comparison_metrics(metrics, comparison=comparison)

    verdict = build_factor_verdict(
        metrics,
        thresholds=evaluation_config.factor_verdict,
    )
    metrics["factor_verdict"] = verdict.label
    metrics["factor_verdict_reasons"] = list(verdict.reasons)
    metrics.update(
        build_campaign_triage(
            metrics,
            thresholds=evaluation_config.campaign_triage,
        ).to_dict()
    )
    metrics.update(
        build_level2_promotion(
            metrics,
            thresholds=evaluation_config.level2_promotion,
        ).to_dict()
    )

    return CompositeEvaluationResult(
        experiment_result=result,
        metrics=metrics,
        ic_timeseries=ic_timeseries.sort_values("date", kind="mergesort").reset_index(
            drop=True
        ),
        rolling_stability=rolling_stability.sort_values(
            "date",
            kind="mergesort",
        ).reset_index(drop=True),
        group_returns=group_returns.sort_values(
            ["date", "group"],
            kind="mergesort",
        ).reset_index(drop=True),
        turnover=turnover.sort_values("date", kind="mergesort").reset_index(drop=True),
        exposure_summary=exposure_df,
    )


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


def _build_neutralization_comparison(
    *,
    raw_row: pd.Series,
    neutralized_row: pd.Series,
    neutralization_mean_corr_reduction: float,
    thresholds: NeutralizationComparisonConfig,
) -> NeutralizationComparison:
    raw_metrics = _comparison_metrics_from_summary_row(raw_row)
    neutralized_metrics = _comparison_metrics_from_summary_row(neutralized_row)
    return build_raw_vs_neutralized_comparison(
        raw_metrics,
        neutralized_metrics,
        neutralization_mean_corr_reduction=(
            neutralization_mean_corr_reduction
            if np.isfinite(neutralization_mean_corr_reduction)
            else None
        ),
        thresholds=thresholds,
    )


def _comparison_metrics_from_summary_row(row: pd.Series) -> dict[str, object]:
    return {
        "mean_ic": float(row["mean_ic"]),
        "mean_rank_ic": float(row["mean_rank_ic"]),
        "mean_long_short_return": float(row["mean_long_short_return"]),
        "ic_ir": float(row["ic_ir"]),
        "ic_valid_ratio": float(row["ic_valid_ratio"]),
        "rank_ic_valid_ratio": float(row["rank_ic_valid_ratio"]),
        "eval_coverage_ratio_mean": float(row["eval_coverage_ratio_mean"]),
        "eval_coverage_ratio_min": float(row["eval_coverage_ratio_min"]),
        "rolling_ic_positive_share": float(row["rolling_ic_positive_share"]),
        "rolling_rank_ic_positive_share": float(row["rolling_rank_ic_positive_share"]),
        "rolling_long_short_positive_share": float(row["rolling_long_short_positive_share"]),
        "rolling_ic_min_mean": float(row["rolling_ic_min_mean"]),
        "rolling_rank_ic_min_mean": float(row["rolling_rank_ic_min_mean"]),
        "rolling_long_short_min_mean": float(row["rolling_long_short_min_mean"]),
        "mean_ic_ci_lower": float(row["mean_ic_ci_lower"]),
        "mean_ic_ci_upper": float(row["mean_ic_ci_upper"]),
        "mean_rank_ic_ci_lower": float(row["mean_rank_ic_ci_lower"]),
        "mean_rank_ic_ci_upper": float(row["mean_rank_ic_ci_upper"]),
        "mean_long_short_return_ci_lower": float(row["mean_long_short_return_ci_lower"]),
        "mean_long_short_return_ci_upper": float(row["mean_long_short_return_ci_upper"]),
        "uncertainty_flags": _parse_flags(row["uncertainty_flags"]),
        "rolling_instability_flags": _parse_flags(row["rolling_instability_flags"]),
    }


def _merge_neutralization_comparison_metrics(
    metrics: dict[str, object],
    *,
    comparison: NeutralizationComparison,
) -> None:
    payload = comparison.to_dict()
    delta = comparison.delta
    raw = comparison.raw
    metrics["neutralization_comparison"] = payload
    metrics["neutralization_comparison_flags"] = list(comparison.interpretation_flags)
    metrics["neutralization_comparison_reasons"] = list(comparison.interpretation_reasons)
    metrics["neutralization_raw_mean_ic"] = raw.get("mean_ic")
    metrics["neutralization_raw_mean_rank_ic"] = raw.get("mean_rank_ic")
    metrics["neutralization_raw_mean_long_short_return"] = raw.get("mean_long_short_return")
    metrics["neutralization_raw_ic_ir"] = raw.get("ic_ir")
    metrics["neutralization_mean_ic_delta"] = delta.get("mean_ic_delta")
    metrics["neutralization_mean_rank_ic_delta"] = delta.get("mean_rank_ic_delta")
    metrics["neutralization_mean_long_short_return_delta"] = delta.get(
        "mean_long_short_return_delta"
    )
    metrics["neutralization_ic_ir_delta"] = delta.get("ic_ir_delta")
    metrics["neutralization_valid_ratio_min_delta"] = delta.get("valid_ratio_min_delta")
    metrics["neutralization_eval_coverage_ratio_mean_delta"] = delta.get(
        "eval_coverage_ratio_mean_delta"
    )
    metrics["neutralization_uncertainty_overlap_zero_count_delta"] = delta.get(
        "uncertainty_overlap_zero_count_delta"
    )
    metrics["neutralization_rolling_positive_share_min_delta"] = delta.get(
        "rolling_positive_share_min_delta"
    )
    metrics["neutralization_rolling_worst_mean_min_delta"] = delta.get(
        "rolling_worst_mean_min_delta"
    )
