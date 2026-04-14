from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from alpha_lab.costs import cost_adjusted_long_short
from alpha_lab.data_quality.corporate_actions import detect_unadjusted_splits
from alpha_lab.data_quality.outlier_detection import detect_stale_prices, filter_zero_volume
from alpha_lab.decay import compute_factor_autocorrelation, compute_ic_decay
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
from alpha_lab.validation.deflated_sharpe import deflated_sharpe_ratio

from .spec import SingleFactorCaseSpec


@dataclass(frozen=True)
class SingleFactorEvaluationResult:
    """Evaluation outputs and summary metrics for one single-factor run."""

    experiment_result: ExperimentResult
    metrics: dict[str, object]
    ic_timeseries: pd.DataFrame
    ic_decay: pd.DataFrame
    factor_autocorrelation: pd.DataFrame
    rolling_stability: pd.DataFrame
    group_returns: pd.DataFrame
    turnover: pd.DataFrame
    coverage: pd.DataFrame
    neutralization_summary: pd.DataFrame


def evaluate_single_factor_case(
    *,
    prices: pd.DataFrame,
    factor_df: pd.DataFrame,
    raw_factor_df: pd.DataFrame | None,
    spec: SingleFactorCaseSpec,
    coverage_by_date: pd.DataFrame,
    neutralization_summary: pd.DataFrame | None,
    evaluation_config: ResearchEvaluationConfig = DEFAULT_RESEARCH_EVALUATION_CONFIG,
) -> SingleFactorEvaluationResult:
    """Evaluate the single factor using the canonical experiment pipeline."""

    result = run_factor_experiment(
        prices,
        lambda _prices: factor_df.copy(),
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
    if spec.neutralization.enabled and raw_factor_df is not None:
        raw_result = run_factor_experiment(
            prices,
            lambda _prices: raw_factor_df.copy(),
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
    ic_decay = compute_ic_decay(
        factor_df=factor_df,
        prices_df=prices,
        horizons=_build_decay_horizons(spec.target.horizon),
    )
    factor_autocorrelation = compute_factor_autocorrelation(
        factor_df=factor_df,
        lags=_build_autocorr_lags(),
    )
    rolling_stability = result.rolling_stability_df.copy()

    group_returns = result.quantile_returns_df.rename(
        columns={"quantile": "group", "mean_return": "group_return"}
    )
    turnover = result.long_short_turnover_df.rename(
        columns={"long_short_turnover": "turnover"}
    )

    neutral_df = (
        neutralization_summary.copy()
        if neutralization_summary is not None
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
        coverage_mean = float(coverage_by_date["coverage"].mean())
        coverage_min = float(coverage_by_date["coverage"].min())

    if neutral_df.empty:
        neutralization_mean_corr_reduction = float("nan")
        neutralization_min_corr_reduction = float("nan")
        neutralization_exposure_count = 0
    else:
        corr_reduction = pd.to_numeric(
            neutral_df["corr_reduction"],
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
        neutralization_exposure_count = int(neutral_df["exposure"].nunique())

    cost_adjusted_mean = float(row["mean_cost_adjusted_long_short_return"])
    if cost_rate > 0:
        _ = cost_adjusted_long_short(
            result.long_short_df,
            result.long_short_turnover_df,
            cost_rate=cost_rate,
        )
    data_quality_summary = _build_data_quality_summary(
        prices=prices,
        integrity_checks=result.integrity_checks,
    )
    dsr_pvalue = _parse_optional_float(row.get("dsr_pvalue"))
    if dsr_pvalue is None:
        long_short_ir = _parse_optional_float(row.get("long_short_ir"))
        n_dates_used = _parse_optional_int(row.get("n_dates_used"))
        if (
            long_short_ir is not None
            and n_dates_used is not None
            and n_dates_used >= 2
        ):
            dsr_pvalue = deflated_sharpe_ratio(
                observed_sr=long_short_ir,
                n_trials=1,
                n_obs=n_dates_used,
            )
    if dsr_pvalue is None:
        dsr_pvalue = float("nan")

    metrics: dict[str, object] = {
        "case_name": spec.name,
        "factor_name": spec.factor_name,
        "target_kind": spec.target.kind,
        "target_horizon": spec.target.horizon,
        "direction": spec.direction,
        "rebalance_frequency": spec.rebalance_frequency,
        "n_quantiles": spec.n_quantiles,
        "mean_ic": float(row["mean_ic"]),
        "mean_ic_ci_lower": float(row["mean_ic_ci_lower"]),
        "mean_ic_ci_upper": float(row["mean_ic_ci_upper"]),
        "mean_rank_ic": float(row["mean_rank_ic"]),
        "mean_rank_ic_ci_lower": float(row["mean_rank_ic_ci_lower"]),
        "mean_rank_ic_ci_upper": float(row["mean_rank_ic_ci_upper"]),
        "ic_ir": float(row["ic_ir"]),
        "ic_t_stat": float(row["ic_t_stat"]),
        "ic_p_value": float(row["ic_p_value"]),
        "dsr_pvalue": float(dsr_pvalue),
        "ic_positive_rate": float(row["ic_positive_rate"]),
        "rank_ic_positive_rate": float(row["rank_ic_positive_rate"]),
        "ic_valid_ratio": float(row["ic_valid_ratio"]),
        "rank_ic_valid_ratio": float(row["rank_ic_valid_ratio"]),
        "split_description": str(row["split_description"]),
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
        "neutralization_comparison": {
            "raw": {},
            "neutralized": {},
            "delta": {},
            "interpretation_flags": [],
            "interpretation_reasons": [],
        },
        "neutralization_comparison_flags": [],
        "neutralization_comparison_reasons": [],
        "neutralization_raw_mean_ic": float("nan"),
        "neutralization_raw_mean_rank_ic": float("nan"),
        "neutralization_raw_mean_long_short_return": float("nan"),
        "neutralization_raw_ic_ir": float("nan"),
        "neutralization_mean_ic_delta": float("nan"),
        "neutralization_mean_rank_ic_delta": float("nan"),
        "neutralization_mean_long_short_return_delta": float("nan"),
        "neutralization_ic_ir_delta": float("nan"),
        "neutralization_valid_ratio_min_delta": float("nan"),
        "neutralization_eval_coverage_ratio_mean_delta": float("nan"),
        "neutralization_uncertainty_overlap_zero_count_delta": float("nan"),
        "neutralization_rolling_positive_share_min_delta": float("nan"),
        "neutralization_rolling_worst_mean_min_delta": float("nan"),
        "research_evaluation_profile": evaluation_config.profile_name,
        "research_evaluation_snapshot": research_evaluation_audit_snapshot(
            evaluation_config
        ),
        **data_quality_summary,
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

    return SingleFactorEvaluationResult(
        experiment_result=result,
        metrics=metrics,
        ic_timeseries=ic_timeseries.sort_values("date", kind="mergesort").reset_index(
            drop=True
        ),
        ic_decay=ic_decay.sort_values("horizon", kind="mergesort").reset_index(drop=True),
        factor_autocorrelation=factor_autocorrelation.sort_values(
            "lag",
            kind="mergesort",
        ).reset_index(drop=True),
        rolling_stability=rolling_stability.sort_values(
            "date",
            kind="mergesort",
        ).reset_index(drop=True),
        group_returns=group_returns.sort_values(
            ["date", "group"],
            kind="mergesort",
        ).reset_index(drop=True),
        turnover=turnover.sort_values("date", kind="mergesort").reset_index(drop=True),
        coverage=coverage_by_date.sort_values("date", kind="mergesort").reset_index(drop=True),
        neutralization_summary=neutral_df,
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


def _build_decay_horizons(target_horizon: int) -> tuple[int, ...]:
    base_horizons = {1, 2, 3, 5, 10, 20}
    if target_horizon > 0:
        base_horizons.add(int(target_horizon))
    return tuple(sorted(base_horizons))


def _build_autocorr_lags() -> tuple[int, ...]:
    return (1, 2, 3, 5, 10)


def _build_data_quality_summary(
    *,
    prices: pd.DataFrame,
    integrity_checks: Iterable[object],
) -> dict[str, object]:
    suspended_rows = _count_suspended_rows(prices)
    stale_rows = _count_stale_rows(prices)
    suspected_split_rows = _count_suspected_split_rows(prices)
    warn_count, fail_count, hard_fail_count = _integrity_status_counts(integrity_checks)

    status = "pass"
    if fail_count > 0 or (suspected_split_rows is not None and suspected_split_rows > 0):
        status = "fail"
    elif (
        warn_count > 0
        or (suspended_rows is not None and suspended_rows > 0)
        or (stale_rows is not None and stale_rows > 0)
    ):
        status = "warn"

    return {
        "data_quality_status": status,
        "data_quality_suspended_rows": suspended_rows,
        "data_quality_stale_rows": stale_rows,
        "data_quality_suspected_split_rows": suspected_split_rows,
        "data_quality_integrity_warn_count": warn_count,
        "data_quality_integrity_fail_count": fail_count,
        "data_quality_hard_fail_count": hard_fail_count,
    }


def _count_suspended_rows(prices: pd.DataFrame) -> int | None:
    required = {"date", "asset", "volume"}
    if not required.issubset(set(prices.columns)):
        return None
    flagged = filter_zero_volume(prices, action="flag")
    if "is_suspended" not in flagged.columns:
        return None
    return int(flagged["is_suspended"].fillna(False).astype(bool).sum())


def _count_stale_rows(prices: pd.DataFrame) -> int | None:
    required = {"date", "asset", "close"}
    if not required.issubset(set(prices.columns)):
        return None
    flagged = detect_stale_prices(prices, max_identical_days=5)
    if "is_stale_price" not in flagged.columns:
        return None
    return int(flagged["is_stale_price"].fillna(False).astype(bool).sum())


def _count_suspected_split_rows(prices: pd.DataFrame) -> int | None:
    required = {"date", "asset", "close"}
    if not required.issubset(set(prices.columns)):
        return None
    flagged = detect_unadjusted_splits(prices, threshold=0.45)
    if "suspected_split" not in flagged.columns:
        return None
    return int(flagged["suspected_split"].fillna(False).astype(bool).sum())


def _integrity_status_counts(
    checks: Iterable[object],
) -> tuple[int, int, int]:
    warn_count = 0
    fail_count = 0
    hard_fail_count = 0
    for check in checks:
        status = str(getattr(check, "status", "")).strip().lower()
        severity = str(getattr(check, "severity", "")).strip().lower()
        if status == "warn":
            warn_count += 1
        if status == "fail":
            fail_count += 1
            if severity == "error":
                hard_fail_count += 1
    return warn_count, fail_count, hard_fail_count


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
