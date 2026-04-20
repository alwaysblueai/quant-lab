from __future__ import annotations

import concurrent.futures
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import TypeVar

import numpy as np
import pandas as pd

from alpha_lab.costs import cost_adjusted_long_short
from alpha_lab.data_quality.corporate_actions import detect_unadjusted_splits
from alpha_lab.data_quality.outlier_detection import detect_stale_prices, filter_zero_volume
from alpha_lab.data_quality.tradability import (
    apply_untradable_mask_to_labels,
    detect_limit_moves,
    summarise_tradability,
)
from alpha_lab.decay import (
    compute_factor_autocorrelation,
    compute_ic_decay,
    estimate_ic_half_life,
)
from alpha_lab.evaluation import compute_ic, compute_ic_summary
from alpha_lab.experiment import ExperimentResult, run_factor_experiment
from alpha_lab.grouped_evaluation import (
    conditional_ic_by_cross_section_size,
    conditional_ic_by_factor_magnitude,
)
from alpha_lab.labels import forward_return
from alpha_lab.marginal_contribution import compute_marginal_contribution
from alpha_lab.quantile import long_short_return, quantile_returns
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
from alpha_lab.validation.haircut_sharpe import haircut_sharpe_ratio

from .spec import SingleFactorCaseSpec

T = TypeVar("T")


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
    capacity_estimation: pd.DataFrame
    conditional_ic_by_magnitude: pd.DataFrame
    conditional_ic_by_cross_section_size: pd.DataFrame
    neutralization_summary: pd.DataFrame
    lag_sensitivity: pd.DataFrame
    random_baseline_null: pd.DataFrame
    daily_pnl_attribution: pd.DataFrame


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


def evaluate_single_factor_case(
    *,
    prices: pd.DataFrame,
    factor_df: pd.DataFrame,
    raw_factor_df: pd.DataFrame | None,
    spec: SingleFactorCaseSpec,
    coverage_by_date: pd.DataFrame,
    neutralization_summary: pd.DataFrame | None,
    precomputed_forward_labels: Mapping[int, pd.DataFrame] | None = None,
    evaluation_config: ResearchEvaluationConfig = DEFAULT_RESEARCH_EVALUATION_CONFIG,
    progress_callback: Callable[[str, int], None] | None = None,
) -> SingleFactorEvaluationResult:
    """Evaluate the single factor using the canonical experiment pipeline."""
    diagnostics_cfg = evaluation_config.single_factor_diagnostics

    def _emit_progress(message: str, percent: int) -> None:
        if progress_callback is not None:
            progress_callback(message, percent)

    def _run_with_stage_heartbeat(
        *,
        stage_message: str,
        stage_percent: int,
        fn: Callable[[], T],
        heartbeat_seconds: int = 20,
    ) -> T:
        _emit_progress(stage_message, stage_percent)
        if progress_callback is None or heartbeat_seconds <= 0:
            return fn()
        started = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(fn)
            while True:
                try:
                    return future.result(timeout=heartbeat_seconds)
                except concurrent.futures.TimeoutError:
                    elapsed_seconds = max(1, int(time.monotonic() - started))
                    _emit_progress(
                        f"{stage_message}（已运行 {elapsed_seconds}s）",
                        stage_percent,
                    )

    close_label_fn = _build_cached_close_label_fn(
        precomputed_forward_labels,
        horizon=spec.target.horizon,
    )

    result = _run_with_stage_heartbeat(
        stage_message="运行核心回测",
        stage_percent=5,
        fn=lambda: run_factor_experiment(
            prices,
            lambda _prices: factor_df.copy(),
            horizon=spec.target.horizon,
            n_quantiles=spec.n_quantiles,
            rolling_stability_thresholds=evaluation_config.rolling_stability,
            label_fn=close_label_fn,
        ),
    )

    _emit_progress("汇总核心指标", 18)
    cost_rate = spec.transaction_cost.one_way_rate
    summary_df = summarise_experiment_result(
        result,
        cost_rate=cost_rate if cost_rate > 0 else None,
        evaluation_config=evaluation_config,
    )
    row = summary_df.iloc[0]

    raw_row: pd.Series | None = None
    if (
        diagnostics_cfg.run_neutralization_raw_comparison
        and spec.neutralization.enabled
        and raw_factor_df is not None
    ):
        raw_result = _run_with_stage_heartbeat(
            stage_message="运行原始因子对照回测",
            stage_percent=12,
            fn=lambda: run_factor_experiment(
                prices,
                lambda _prices: raw_factor_df.copy(),
                horizon=spec.target.horizon,
                n_quantiles=spec.n_quantiles,
                rolling_stability_thresholds=evaluation_config.rolling_stability,
                label_fn=close_label_fn,
            ),
        )
        raw_summary_df = summarise_experiment_result(
            raw_result,
            cost_rate=cost_rate if cost_rate > 0 else None,
            evaluation_config=evaluation_config,
        )
        raw_row = raw_summary_df.iloc[0]

    ic_timeseries = (
        result.ic_df[["date", "ic"]]
        .merge(
            result.rank_ic_df[["date", "rank_ic"]],
            on="date",
            how="outer",
            sort=True,
        )
        .merge(
            result.mutual_information_df[["date", "mutual_information"]],
            on="date",
            how="outer",
            sort=True,
        )
    )
    _emit_progress("生成稳定性与衰减诊断", 32)
    decay_horizons = _build_decay_horizons(spec.target.horizon)
    decay_horizon_set = set(decay_horizons)
    decay_label_cache = (
        {
            int(h): labels
            for h, labels in precomputed_forward_labels.items()
            if int(h) in decay_horizon_set
        }
        if precomputed_forward_labels is not None
        else None
    )
    ic_decay = (
        compute_ic_decay(
            factor_df=factor_df,
            prices_df=prices,
            horizons=decay_horizons,
            precomputed_labels_by_horizon=decay_label_cache,
        )
        if diagnostics_cfg.compute_ic_decay
        else _empty_ic_decay_frame()
    )
    decay_summary = estimate_ic_half_life(ic_decay)
    decay_retention_5_over_1 = _compute_ic_decay_retention_ratio(
        ic_decay,
        horizon_short=1,
        horizon_long=5,
    )
    factor_autocorrelation = (
        compute_factor_autocorrelation(
            factor_df=factor_df,
            lags=_build_autocorr_lags(),
        )
        if diagnostics_cfg.compute_factor_autocorrelation
        else _empty_factor_autocorrelation_frame()
    )
    if diagnostics_cfg.compute_conditional_ic:
        conditional_by_magnitude = conditional_ic_by_factor_magnitude(
            factor_df=factor_df,
            labels_df=result.label_df,
        )
        conditional_by_cross_section = conditional_ic_by_cross_section_size(
            factor_df=factor_df,
            labels_df=result.label_df,
        )
    else:
        conditional_by_magnitude = _empty_conditional_ic_by_magnitude_frame()
        conditional_by_cross_section = _empty_conditional_ic_by_cross_section_size_frame()
    rolling_stability = result.rolling_stability_df.copy()

    group_returns = result.quantile_returns_df.rename(
        columns={"quantile": "group", "mean_return": "group_return"}
    )
    turnover = result.long_short_turnover_df.rename(columns={"long_short_turnover": "turnover"})
    capacity_spec = getattr(spec, "capacity", None)
    capacity_enabled = bool(
        getattr(capacity_spec, "enabled", False) if capacity_spec is not None else False
    )
    capacity_participation_rate = float(
        getattr(capacity_spec, "participation_rate", 0.05) if capacity_spec is not None else 0.05
    )
    capacity_adv_lookback = int(
        getattr(capacity_spec, "adv_lookback", 20) if capacity_spec is not None else 20
    )
    rebalance_step = _resolve_rebalance_step(spec.rebalance_frequency)
    decay_ratio, decay_mismatch_flag = _compute_ic_decay_rebalance_consistency(
        rebalance_step=rebalance_step,
        half_life_horizon=decay_summary.get("ic_half_life_horizon"),
        thresholds=evaluation_config,
    )
    _emit_progress("整理覆盖率与容量摘要", 46)
    capacity_estimation = (
        _build_capacity_estimation(
            prices=prices,
            labels_df=result.label_df,
            quantile_assignments_df=result.quantile_assignments_df,
            long_short_df=result.long_short_df,
            mean_long_short_turnover=float(row["mean_long_short_turnover"]),
            n_quantiles=spec.n_quantiles,
            rebalance_step=rebalance_step,
            enabled=capacity_enabled,
            participation_rate=capacity_participation_rate,
            adv_lookback=capacity_adv_lookback,
        )
        if diagnostics_cfg.compute_capacity_estimation
        else _empty_capacity_estimation_frame(
            enabled=capacity_enabled,
            participation_rate=capacity_participation_rate,
            adv_lookback=capacity_adv_lookback,
        )
    )
    capacity_summary = (
        capacity_estimation.iloc[0].to_dict()
        if not capacity_estimation.empty
        else _empty_capacity_summary()
    )
    conditional_summary = _build_conditional_ic_summary(
        conditional_by_magnitude=conditional_by_magnitude,
        conditional_by_cross_section=conditional_by_cross_section,
    )
    rank_ic_ir = _compute_rank_ic_ir(result.rank_ic_df)
    group_monotonicity = _build_group_monotonicity_summary(
        group_returns=group_returns,
        n_quantiles=spec.n_quantiles,
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
            float(corr_reduction.mean()) if len(corr_reduction) > 0 else float("nan")
        )
        neutralization_min_corr_reduction = (
            float(corr_reduction.min()) if len(corr_reduction) > 0 else float("nan")
        )
        neutralization_exposure_count = int(neutral_df["exposure"].nunique())

    cost_adjusted_mean = float(row["mean_cost_adjusted_long_short_return"])
    cost_aware_long_short_ir = _compute_cost_adjusted_long_short_ir(
        long_short_df=result.long_short_df,
        long_short_turnover_df=result.long_short_turnover_df,
        cost_rate=cost_rate,
    )
    if not np.isfinite(cost_aware_long_short_ir):
        cost_aware_long_short_ir = float(row["long_short_ir"])
    data_quality_summary = _build_data_quality_summary(
        prices=prices,
        integrity_checks=result.integrity_checks,
    )
    dsr_pvalue = _parse_optional_float(row.get("dsr_pvalue"))
    if dsr_pvalue is None:
        long_short_ir = _parse_optional_float(row.get("long_short_ir"))
        n_dates_used = _parse_optional_int(row.get("n_dates_used"))
        if long_short_ir is not None and n_dates_used is not None and n_dates_used >= 2:
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
        "rebalance_step_dates": rebalance_step,
        "n_quantiles": spec.n_quantiles,
        "mean_ic": float(row["mean_ic"]),
        "mean_ic_ci_lower": float(row["mean_ic_ci_lower"]),
        "mean_ic_ci_upper": float(row["mean_ic_ci_upper"]),
        "mean_rank_ic": float(row["mean_rank_ic"]),
        "rank_ic_ir": rank_ic_ir,
        "mean_mutual_information": float(row["mean_mutual_information"]),
        "mean_rank_ic_ci_lower": float(row["mean_rank_ic_ci_lower"]),
        "mean_rank_ic_ci_upper": float(row["mean_rank_ic_ci_upper"]),
        "mutual_information_ir": float(row["mutual_information_ir"]),
        "mutual_information_positive_rate": float(row["mutual_information_positive_rate"]),
        "mutual_information_valid_ratio": float(row["mutual_information_valid_ratio"]),
        "ic_ir": float(row["ic_ir"]),
        "ic_t_stat": float(row["ic_t_stat"]),
        "ic_p_value": float(row["ic_p_value"]),
        "dsr_pvalue": float(dsr_pvalue),
        "ic_positive_rate": float(row["ic_positive_rate"]),
        "rank_ic_positive_rate": float(row["rank_ic_positive_rate"]),
        "ic_valid_ratio": float(row["ic_valid_ratio"]),
        "rank_ic_valid_ratio": float(row["rank_ic_valid_ratio"]),
        "split_description": str(row["split_description"]),
        "ic_decay_half_life_summary": _build_ic_decay_half_life_summary(decay_summary),
        "ic_decay_retention_5_over_1": decay_retention_5_over_1,
        "mean_long_short_return": float(row["mean_long_short_return"]),
        "mean_long_short_return_ci_lower": float(row["mean_long_short_return_ci_lower"]),
        "mean_long_short_return_ci_upper": float(row["mean_long_short_return_ci_upper"]),
        "long_short_ir": float(row["long_short_ir"]),
        "long_short_hit_rate": float(row["long_short_hit_rate"]),
        "long_short_return_per_turnover": float(row["long_short_return_per_turnover"]),
        "subperiod_ic_positive_share": float(row["subperiod_ic_positive_share"]),
        "subperiod_long_short_positive_share": float(row["subperiod_long_short_positive_share"]),
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
        "cost_aware_long_short_ir": cost_aware_long_short_ir,
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
        "ls_max_drawdown": float(result.summary.ls_max_drawdown),
        "max_drawdown": float(result.summary.ls_max_drawdown),
        "ls_max_drawdown_duration": int(result.summary.ls_max_drawdown_duration),
        "ls_max_consecutive_loss_days": int(result.summary.ls_max_consecutive_loss_days),
        "ls_var_5": float(result.summary.ls_var_5),
        "ls_cvar_5": float(result.summary.ls_cvar_5),
        "ls_calmar_ratio": float(result.summary.ls_calmar_ratio),
        "regime_flags": list(result.summary.regime_flags),
        "ic_half_life_horizon": _parse_optional_float(decay_summary.get("ic_half_life_horizon")),
        "ic_half_life_status": str(decay_summary.get("ic_half_life_status") or "unavailable"),
        "ic_half_life_not_reached": bool(decay_summary.get("ic_half_life_not_reached")),
        "ic_decay_rebalance_ratio": decay_ratio,
        "ic_decay_mismatch_flag": decay_mismatch_flag,
        "coverage_mean": coverage_mean,
        "coverage_min": coverage_min,
        "coverage_summary": _build_coverage_summary(
            n_dates_used=int(row["n_dates_used"]),
            mean_eval_assets_per_date=float(row["mean_eval_assets_per_date"]),
            eval_coverage_ratio_mean=float(row["eval_coverage_ratio_mean"]),
        ),
        "group_monotonicity_qtop_qbottom": group_monotonicity["qtop_qbottom_spread_mean"],
        "group_monotonicity_share": group_monotonicity["monotonic_share"],
        "group_monotonicity_summary": group_monotonicity["summary"],
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
        "research_evaluation_snapshot": research_evaluation_audit_snapshot(evaluation_config),
        "single_factor_diagnostics_mode": "streamlined"
        if _skipped_single_factor_diagnostics(diagnostics_cfg)
        else "full",
        "single_factor_skipped_diagnostics": _skipped_single_factor_diagnostics(diagnostics_cfg),
        **capacity_summary,
        **conditional_summary,
        **data_quality_summary,
    }
    _emit_progress("运行附加稳健性诊断", 62)
    _merge_marginal_contribution_metrics(
        metrics,
        factor_df=factor_df,
        label_df=result.label_df,
        enabled=diagnostics_cfg.run_marginal_contribution,
    )
    _merge_tradability_metrics(
        metrics,
        prices=prices,
        factor_df=factor_df,
        label_df=result.label_df,
        horizon=spec.target.horizon,
        n_quantiles=spec.n_quantiles,
        evaluation_config=evaluation_config,
        cost_rate=cost_rate,
        enabled=diagnostics_cfg.run_tradability_checks,
    )
    _merge_execution_price_sensitivity_metrics(
        metrics,
        prices=prices,
        factor_df=factor_df,
        horizon=spec.target.horizon,
        n_quantiles=spec.n_quantiles,
        evaluation_config=evaluation_config,
        cost_rate=cost_rate,
        enabled=diagnostics_cfg.run_execution_price_sensitivity,
    )
    _merge_haircut_sharpe_metrics(metrics)
    _merge_param_sensitivity_metrics(
        metrics,
        prices=prices,
        factor_df=factor_df,
        horizon=spec.target.horizon,
        base_n_quantiles=spec.n_quantiles,
        evaluation_config=evaluation_config,
        label_df=result.label_df,
        label_fn=close_label_fn,
        enabled=diagnostics_cfg.run_param_sensitivity,
    )
    _merge_baseline_factor_comparison_metrics(
        metrics,
        prices=prices,
        factor_df=factor_df,
        horizon=spec.target.horizon,
        n_quantiles=spec.n_quantiles,
        evaluation_config=evaluation_config,
        label_df=result.label_df,
        label_fn=close_label_fn,
        enabled=diagnostics_cfg.run_baseline_comparison,
    )
    random_baseline_null_df = _merge_random_factor_baseline_metrics(
        metrics,
        factor_df=factor_df,
        label_df=result.label_df,
        enabled=diagnostics_cfg.run_random_baseline,
    )
    daily_pnl_attribution_df = _merge_daily_pnl_attribution_metrics(
        metrics,
        result=result,
        cost_rate=cost_rate,
    )
    lag_sensitivity_df = _merge_signal_lag_sensitivity_metrics(
        metrics,
        prices=prices,
        factor_df=factor_df,
        horizon=spec.target.horizon,
        n_quantiles=spec.n_quantiles,
        evaluation_config=evaluation_config,
        label_df=result.label_df,
        base_result=result,
        label_fn=close_label_fn,
        enabled=diagnostics_cfg.run_lag_sensitivity,
    )
    _emit_progress("汇总结论与分层判定", 86)
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
        ic_timeseries=ic_timeseries.sort_values("date", kind="mergesort").reset_index(drop=True),
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
        capacity_estimation=capacity_estimation,
        conditional_ic_by_magnitude=conditional_by_magnitude,
        conditional_ic_by_cross_section_size=conditional_by_cross_section,
        neutralization_summary=neutral_df,
        lag_sensitivity=lag_sensitivity_df,
        random_baseline_null=random_baseline_null_df,
        daily_pnl_attribution=daily_pnl_attribution_df,
    )


def _build_cached_close_label_fn(
    precomputed_forward_labels: Mapping[int, pd.DataFrame] | None,
    *,
    horizon: int,
) -> Callable[[pd.DataFrame], pd.DataFrame] | None:
    if precomputed_forward_labels is None:
        return None
    cached = precomputed_forward_labels.get(int(horizon))
    if cached is None:
        return None

    def _label_fn(_prices: pd.DataFrame) -> pd.DataFrame:
        return cached.copy()

    return _label_fn


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


def _skipped_single_factor_diagnostics(diagnostics_cfg: object) -> list[str]:
    mapping = {
        "neutralization_raw_comparison": getattr(
            diagnostics_cfg, "run_neutralization_raw_comparison", True
        ),
        "ic_decay": getattr(diagnostics_cfg, "compute_ic_decay", True),
        "factor_autocorrelation": getattr(diagnostics_cfg, "compute_factor_autocorrelation", True),
        "capacity_estimation": getattr(diagnostics_cfg, "compute_capacity_estimation", True),
        "conditional_ic": getattr(diagnostics_cfg, "compute_conditional_ic", True),
        "marginal_contribution": getattr(diagnostics_cfg, "run_marginal_contribution", True),
        "tradability_checks": getattr(diagnostics_cfg, "run_tradability_checks", True),
        "execution_price_sensitivity": getattr(
            diagnostics_cfg, "run_execution_price_sensitivity", True
        ),
        "parameter_sensitivity": getattr(diagnostics_cfg, "run_param_sensitivity", True),
        "baseline_factor_comparison": getattr(diagnostics_cfg, "run_baseline_comparison", True),
        "random_baseline": getattr(diagnostics_cfg, "run_random_baseline", True),
        "lag_sensitivity": getattr(diagnostics_cfg, "run_lag_sensitivity", True),
    }
    return [name for name, enabled in mapping.items() if not enabled]


def _empty_ic_decay_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "horizon",
            "mean_ic",
            "mean_rank_ic",
            "ic_ir",
            "t_stat",
            "p_value",
            "n_dates",
        ]
    )


def _empty_factor_autocorrelation_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["lag", "mean_autocorr", "std_autocorr", "n_dates"])


def _empty_conditional_ic_by_magnitude_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "magnitude_quintile",
            "mean_ic",
            "mean_rank_ic",
            "ic_positive_rate",
            "rank_ic_positive_rate",
            "n_dates_used",
            "mean_assets_per_date",
        ]
    )


def _empty_conditional_ic_by_cross_section_size_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "cross_section_bucket",
            "median_valid_assets_threshold",
            "mean_valid_assets",
            "mean_ic",
            "mean_rank_ic",
            "ic_positive_rate",
            "rank_ic_positive_rate",
            "n_dates_used",
        ]
    )


def _empty_capacity_estimation_frame(
    *,
    enabled: bool,
    participation_rate: float,
    adv_lookback: int,
) -> pd.DataFrame:
    summary = _empty_capacity_summary()
    summary["capacity_enabled"] = bool(enabled)
    summary["capacity_status"] = "skipped"
    summary["capacity_notes"] = "capacity diagnostics skipped by evaluation profile"
    summary["capacity_participation_rate"] = float(participation_rate)
    summary["capacity_adv_lookback"] = int(adv_lookback)
    return pd.DataFrame([summary])


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


def _build_capacity_estimation(
    *,
    prices: pd.DataFrame,
    labels_df: pd.DataFrame,
    quantile_assignments_df: pd.DataFrame,
    long_short_df: pd.DataFrame,
    mean_long_short_turnover: float,
    n_quantiles: int,
    rebalance_step: int,
    enabled: bool,
    participation_rate: float,
    adv_lookback: int,
) -> pd.DataFrame:
    summary = _empty_capacity_summary()
    summary["capacity_enabled"] = bool(enabled)
    summary["capacity_participation_rate"] = float(participation_rate)
    summary["capacity_adv_lookback"] = int(adv_lookback)
    if not enabled:
        summary["capacity_status"] = "disabled"
        summary["capacity_notes"] = "capacity estimation disabled by case spec"
        return pd.DataFrame([summary])

    cap_col = _resolve_market_cap_col(prices)
    has_amount = "amount" in prices.columns
    if cap_col is None and not has_amount:
        summary["capacity_status"] = "unavailable"
        summary["capacity_notes"] = "missing explicit market-cap column and amount column"
        return pd.DataFrame([summary])

    summary["equal_weight_mean_long_short_return"] = _finite_or_nan(
        _parse_optional_float(
            long_short_df["long_short_return"].mean() if not long_short_df.empty else float("nan")
        )
    )

    if cap_col is not None:
        weighted_mean = _compute_market_cap_weighted_long_short_return(
            prices=prices,
            labels_df=labels_df,
            quantile_assignments_df=quantile_assignments_df,
            n_quantiles=n_quantiles,
            cap_col=cap_col,
        )
        summary["capacity_market_cap_column"] = cap_col
        summary["market_cap_weighted_mean_long_short_return"] = weighted_mean
        equal_weight_mean = _parse_optional_float(
            summary.get("equal_weight_mean_long_short_return")
        )
        if np.isfinite(weighted_mean) and equal_weight_mean is not None:
            summary["market_cap_vs_equal_weight_return_delta"] = float(
                weighted_mean - equal_weight_mean
            )

    if has_amount:
        mean_traded_adv = _compute_mean_traded_adv(
            prices=prices,
            quantile_assignments_df=quantile_assignments_df,
            n_quantiles=n_quantiles,
            rebalance_step=rebalance_step,
            adv_lookback=adv_lookback,
        )
        summary["mean_traded_adv"] = mean_traded_adv

    turnover = _parse_optional_float(mean_long_short_turnover)
    traded_adv = _parse_optional_float(summary.get("mean_traded_adv"))
    if traded_adv is not None and turnover is not None and turnover > 0.0:
        summary["estimated_capacity_upper_bound"] = float(
            traded_adv * float(participation_rate) / turnover
        )

    unavailable_parts: list[str] = []
    if cap_col is None:
        unavailable_parts.append("missing explicit market-cap column")
    if not has_amount:
        unavailable_parts.append("missing amount column for ADV")
    if _parse_optional_float(summary.get("estimated_capacity_upper_bound")) is None:
        if traded_adv is None:
            unavailable_parts.append("mean traded ADV unavailable")
        if turnover is None or turnover <= 0.0:
            unavailable_parts.append("turnover unavailable for capacity upper bound")

    if unavailable_parts:
        if cap_col is not None or has_amount:
            summary["capacity_status"] = "partial"
        else:
            summary["capacity_status"] = "unavailable"
        summary["capacity_notes"] = "; ".join(unavailable_parts)
    else:
        summary["capacity_status"] = "available"
        summary["capacity_notes"] = "capacity diagnostics available"

    return pd.DataFrame([summary])


def _compute_market_cap_weighted_long_short_return(
    *,
    prices: pd.DataFrame,
    labels_df: pd.DataFrame,
    quantile_assignments_df: pd.DataFrame,
    n_quantiles: int,
    cap_col: str,
) -> float:
    working = quantile_assignments_df.merge(
        labels_df[["date", "asset", "value"]].rename(columns={"value": "label_value"}),
        on=["date", "asset"],
        how="inner",
        validate="one_to_one",
    ).merge(
        prices[["date", "asset", cap_col]].rename(columns={cap_col: "market_cap_value"}),
        on=["date", "asset"],
        how="left",
        validate="many_to_one",
    )
    if working.empty:
        return float("nan")

    working["label_value"] = pd.to_numeric(working["label_value"], errors="coerce")
    working["market_cap_value"] = pd.to_numeric(working["market_cap_value"], errors="coerce")
    working = working.dropna(subset=["label_value", "market_cap_value"]).copy()
    working = working.loc[working["market_cap_value"] > 0.0].copy()
    if working.empty:
        return float("nan")

    rows: list[float] = []
    for _, block in working.groupby("date", sort=True):
        long_block = block.loc[block["quantile"] == n_quantiles]
        short_block = block.loc[block["quantile"] == 1]
        if long_block.empty or short_block.empty:
            continue
        long_weights = long_block["market_cap_value"] / long_block["market_cap_value"].sum()
        short_weights = short_block["market_cap_value"] / short_block["market_cap_value"].sum()
        rows.append(
            float((long_block["label_value"] * long_weights).sum())
            - float((short_block["label_value"] * short_weights).sum())
        )
    return float(np.mean(rows)) if rows else float("nan")


def _compute_mean_traded_adv(
    *,
    prices: pd.DataFrame,
    quantile_assignments_df: pd.DataFrame,
    n_quantiles: int,
    rebalance_step: int,
    adv_lookback: int,
) -> float:
    if "amount" not in prices.columns or quantile_assignments_df.empty:
        return float("nan")

    adv_frame = prices[["date", "asset", "amount"]].copy()
    adv_frame["date"] = pd.to_datetime(adv_frame["date"], errors="coerce")
    adv_frame["amount"] = pd.to_numeric(adv_frame["amount"], errors="coerce")
    adv_frame = adv_frame.dropna(subset=["date", "asset", "amount"]).sort_values(
        ["asset", "date"],
        kind="mergesort",
    )
    if adv_frame.empty:
        return float("nan")

    min_periods = min(int(adv_lookback), max(3, int(adv_lookback) // 2))
    adv_frame["adv"] = adv_frame.groupby("asset", sort=False)["amount"].transform(
        lambda series: series.rolling(adv_lookback, min_periods=min_periods).mean()
    )
    adv_lookup = adv_frame[["date", "asset", "adv"]]

    assignments = quantile_assignments_df.copy()
    assignments["date"] = pd.to_datetime(assignments["date"], errors="coerce")
    assignments = assignments.sort_values(["date", "asset"], kind="mergesort")
    active_dates = (
        assignments["date"].drop_duplicates().sort_values().iloc[:: max(1, rebalance_step)]
    )
    if len(active_dates) < 2:
        return float("nan")

    traded_adv_values: list[float] = []
    for prev_date, curr_date in zip(active_dates[:-1], active_dates[1:], strict=False):
        prev_block = assignments.loc[assignments["date"] == prev_date]
        curr_block = assignments.loc[assignments["date"] == curr_date]
        traded_assets = _traded_assets_between_assignments(
            prev_block=prev_block,
            curr_block=curr_block,
            n_quantiles=n_quantiles,
        )
        if not traded_assets:
            continue
        adv_rows = adv_lookup.loc[
            (adv_lookup["date"] == curr_date) & (adv_lookup["asset"].isin(traded_assets)),
            "adv",
        ].dropna()
        if len(adv_rows) == 0:
            continue
        traded_adv_values.append(float(adv_rows.mean()))
    return float(np.mean(traded_adv_values)) if traded_adv_values else float("nan")


def _traded_assets_between_assignments(
    *,
    prev_block: pd.DataFrame,
    curr_block: pd.DataFrame,
    n_quantiles: int,
) -> set[str]:
    prev_long = set(prev_block.loc[prev_block["quantile"] == n_quantiles, "asset"].astype(str))
    prev_short = set(prev_block.loc[prev_block["quantile"] == 1, "asset"].astype(str))
    curr_long = set(curr_block.loc[curr_block["quantile"] == n_quantiles, "asset"].astype(str))
    curr_short = set(curr_block.loc[curr_block["quantile"] == 1, "asset"].astype(str))
    return (prev_long ^ curr_long) | (prev_short ^ curr_short)


def _build_conditional_ic_summary(
    *,
    conditional_by_magnitude: pd.DataFrame,
    conditional_by_cross_section: pd.DataFrame,
) -> dict[str, object]:
    q1_mean_ic = _conditional_metric_value(
        conditional_by_magnitude,
        group_col="magnitude_quintile",
        group_value="Q1",
        value_col="mean_ic",
    )
    q5_mean_ic = _conditional_metric_value(
        conditional_by_magnitude,
        group_col="magnitude_quintile",
        group_value="Q5",
        value_col="mean_ic",
    )
    small_mean_ic = _conditional_metric_value(
        conditional_by_cross_section,
        group_col="cross_section_bucket",
        group_value="small_cross_section",
        value_col="mean_ic",
    )
    large_mean_ic = _conditional_metric_value(
        conditional_by_cross_section,
        group_col="cross_section_bucket",
        group_value="large_cross_section",
        value_col="mean_ic",
    )
    return {
        "conditional_ic_q1_mean_ic": q1_mean_ic,
        "conditional_ic_q5_mean_ic": q5_mean_ic,
        "conditional_ic_extreme_minus_base_ic": (
            float(q5_mean_ic - q1_mean_ic)
            if np.isfinite(q5_mean_ic) and np.isfinite(q1_mean_ic)
            else float("nan")
        ),
        "conditional_ic_small_cross_section_mean_ic": small_mean_ic,
        "conditional_ic_large_cross_section_mean_ic": large_mean_ic,
    }


def _conditional_metric_value(
    frame: pd.DataFrame,
    *,
    group_col: str,
    group_value: str,
    value_col: str,
) -> float:
    if frame.empty or group_col not in frame.columns or value_col not in frame.columns:
        return float("nan")
    values = pd.to_numeric(
        frame.loc[frame[group_col] == group_value, value_col],
        errors="coerce",
    ).dropna()
    return float(values.iloc[0]) if len(values) > 0 else float("nan")


def _resolve_market_cap_col(prices: pd.DataFrame) -> str | None:
    for col in ("market_cap", "circ_mv", "total_mv", "value"):
        if col in prices.columns:
            return col
    return None


def _empty_capacity_summary() -> dict[str, object]:
    return {
        "capacity_enabled": False,
        "capacity_status": "unavailable",
        "capacity_notes": "capacity diagnostics unavailable",
        "capacity_market_cap_column": "",
        "capacity_participation_rate": float("nan"),
        "capacity_adv_lookback": float("nan"),
        "equal_weight_mean_long_short_return": float("nan"),
        "market_cap_weighted_mean_long_short_return": float("nan"),
        "market_cap_vs_equal_weight_return_delta": float("nan"),
        "mean_traded_adv": float("nan"),
        "estimated_capacity_upper_bound": float("nan"),
    }


def _finite_or_nan(value: float | None) -> float:
    if value is None or not np.isfinite(value):
        return float("nan")
    return float(value)


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


def _merge_tradability_metrics(
    metrics: dict[str, object],
    *,
    prices: pd.DataFrame,
    factor_df: pd.DataFrame,
    label_df: pd.DataFrame,
    horizon: int,
    n_quantiles: int,
    evaluation_config: ResearchEvaluationConfig,
    cost_rate: float,
    enabled: bool = True,
) -> None:
    """Flag rows whose next-day execution is blocked (limit-up/down, suspended)
    and re-run the evaluation with those labels blanked to quantify the
    tradability leakage.

    The delta fields surface how much of the headline IC / long-short return is
    attributable to days that could never have been traded in practice — the
    single largest silent leakage source for short-horizon price-volume
    factors.
    """
    counts = _empty_tradability_summary()
    metrics.update(counts)
    metrics["tradability_filtered_mean_ic"] = float("nan")
    metrics["tradability_filtered_mean_long_short_return"] = float("nan")
    metrics["tradability_ic_delta"] = float("nan")
    metrics["tradability_long_short_return_delta"] = float("nan")
    metrics["tradability_filter_applied"] = False
    if not enabled:
        return
    try:
        flagged = detect_limit_moves(prices)
        counts.update(summarise_tradability(flagged))
        metrics.update(counts)
    except Exception:
        return

    untradable_rows = _parse_optional_int(counts.get("tradability_untradable_rows"))
    if untradable_rows is None or untradable_rows <= 0:
        return

    try:
        masked_labels = apply_untradable_mask_to_labels(
            label_df,
            prices,
            direction="both",
        )
    except Exception:
        return

    try:
        filtered_summary = _evaluate_variant_lightweight(
            factor_df=factor_df,
            label_df=masked_labels[["date", "asset", "factor", "value"]],
            n_quantiles=n_quantiles,
        )
    except Exception:
        return

    filtered_mean_ic = float(filtered_summary.mean_ic)
    filtered_mean_ls = float(filtered_summary.mean_long_short_return)
    base_mean_ic = _finite_or_nan(_parse_optional_float(metrics.get("mean_ic")))
    base_mean_ls = _finite_or_nan(_parse_optional_float(metrics.get("mean_long_short_return")))

    metrics["tradability_filter_applied"] = True
    metrics["tradability_filtered_mean_ic"] = filtered_mean_ic
    metrics["tradability_filtered_mean_long_short_return"] = filtered_mean_ls
    if np.isfinite(filtered_mean_ic) and np.isfinite(base_mean_ic):
        metrics["tradability_ic_delta"] = filtered_mean_ic - base_mean_ic
    if np.isfinite(filtered_mean_ls) and np.isfinite(base_mean_ls):
        metrics["tradability_long_short_return_delta"] = filtered_mean_ls - base_mean_ls


def _empty_tradability_summary() -> dict[str, object]:
    return {
        "tradability_total_rows": 0,
        "tradability_limit_up_rows": 0,
        "tradability_limit_down_rows": 0,
        "tradability_untradable_rows": 0,
        "tradability_untradable_rate": float("nan"),
    }


def _merge_execution_price_sensitivity_metrics(
    metrics: dict[str, object],
    *,
    prices: pd.DataFrame,
    factor_df: pd.DataFrame,
    horizon: int,
    n_quantiles: int,
    evaluation_config: ResearchEvaluationConfig,
    cost_rate: float,
    enabled: bool = True,
) -> None:
    """Re-run the evaluation with next-open execution prices and record the
    delta vs the default close-to-close convention.

    Price-volume factors that look strong at close-to-close but disappear
    under next-open execution are capturing the signal-day close itself —
    a bar the real-world trader cannot transact on.  The delta fields here
    make that failure mode visible without requiring the caller to re-run
    the pipeline manually.
    """
    metrics["next_open_execution_available"] = False
    metrics["next_open_mean_ic"] = float("nan")
    metrics["next_open_mean_long_short_return"] = float("nan")
    metrics["next_open_long_short_ir"] = float("nan")
    metrics["next_open_mean_ic_delta"] = float("nan")
    metrics["next_open_mean_long_short_return_delta"] = float("nan")
    if not enabled:
        return

    if "open" not in prices.columns:
        return

    try:
        next_open_summary = _evaluate_variant_lightweight(
            factor_df=factor_df,
            label_df=forward_return(
                prices,
                horizon=horizon,
                execution_price_mode="next_open",
            ),
            n_quantiles=n_quantiles,
        )
    except Exception:
        return

    next_open_mean_ic = float(next_open_summary.mean_ic)
    next_open_mean_ls = float(next_open_summary.mean_long_short_return)
    base_mean_ic = _finite_or_nan(_parse_optional_float(metrics.get("mean_ic")))
    base_mean_ls = _finite_or_nan(_parse_optional_float(metrics.get("mean_long_short_return")))

    metrics["next_open_execution_available"] = True
    metrics["next_open_mean_ic"] = next_open_mean_ic
    metrics["next_open_mean_long_short_return"] = next_open_mean_ls
    metrics["next_open_long_short_ir"] = float(next_open_summary.long_short_ir)
    if np.isfinite(next_open_mean_ic) and np.isfinite(base_mean_ic):
        metrics["next_open_mean_ic_delta"] = next_open_mean_ic - base_mean_ic
    if np.isfinite(next_open_mean_ls) and np.isfinite(base_mean_ls):
        metrics["next_open_mean_long_short_return_delta"] = next_open_mean_ls - base_mean_ls


def _merge_haircut_sharpe_metrics(metrics: dict[str, object]) -> None:
    """Add a multiple-testing-adjusted Sharpe ratio to the metrics dict.

    ``n_trials`` is conservatively set to 1 — the default single-factor case
    assumes no grid search.  Callers running a parameter sweep should
    overwrite these fields with a higher ``n_trials`` after collecting
    variant results.
    """
    observed_sharpe = _parse_optional_float(metrics.get("long_short_ir"))
    n_obs = _parse_optional_int(metrics.get("n_dates_used"))

    metrics["haircut_sharpe_n_trials_assumed"] = 1
    if observed_sharpe is None or n_obs is None or n_obs < 2:
        metrics["haircut_sharpe_observed"] = (
            float(observed_sharpe) if observed_sharpe is not None else float("nan")
        )
        metrics["haircut_sharpe_expected_max"] = float("nan")
        metrics["haircut_sharpe_adjusted"] = float("nan")
        metrics["haircut_sharpe_ratio"] = float("nan")
        return

    result = haircut_sharpe_ratio(
        observed_sharpe=float(observed_sharpe),
        n_trials=1,
        n_obs=int(n_obs),
    )
    metrics["haircut_sharpe_observed"] = result.observed_sharpe
    metrics["haircut_sharpe_expected_max"] = result.expected_max_sharpe
    metrics["haircut_sharpe_adjusted"] = result.haircut_sharpe
    metrics["haircut_sharpe_ratio"] = result.haircut_ratio


def _merge_marginal_contribution_metrics(
    metrics: dict[str, object],
    *,
    factor_df: pd.DataFrame,
    label_df: pd.DataFrame,
    enabled: bool = True,
) -> None:
    """Run independent Fama-MacBeth regression (no existing factors) and
    write scalar fields into ``metrics``.

    In single-factor scenarios the spanning test is not applicable (no
    existing factors to control for), so those fields remain ``None``.
    The Fama-MacBeth regression is still informative: it cross-validates
    the IC evidence via an explicit cross-sectional predictive-power
    coefficient.
    """
    metrics["fama_macbeth_mean_coefficient"] = None
    metrics["fama_macbeth_t_statistic"] = None
    metrics["fama_macbeth_p_value"] = None
    metrics["fama_macbeth_n_dates"] = None
    metrics["spanning_is_spanned"] = None
    metrics["spanning_r_squared_increment"] = None
    metrics["marginal_flags"] = []
    if not enabled:
        return

    summary = compute_marginal_contribution(
        candidate_factor_df=factor_df,
        label_df=label_df,
        existing_factor_dfs=None,
    )
    fm = summary.fama_macbeth
    metrics["fama_macbeth_mean_coefficient"] = (
        float(fm.mean_coefficient) if fm is not None else None
    )
    metrics["fama_macbeth_t_statistic"] = float(fm.t_statistic) if fm is not None else None
    metrics["fama_macbeth_p_value"] = float(fm.p_value) if fm is not None else None
    metrics["fama_macbeth_n_dates"] = int(fm.n_dates) if fm is not None else None
    metrics["spanning_is_spanned"] = None
    metrics["spanning_r_squared_increment"] = None
    metrics["marginal_flags"] = list(summary.marginal_flags)


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


def _merge_param_sensitivity_metrics(
    metrics: dict[str, object],
    *,
    prices: pd.DataFrame,
    factor_df: pd.DataFrame,
    horizon: int,
    base_n_quantiles: int,
    evaluation_config: ResearchEvaluationConfig,
    label_df: pd.DataFrame | None = None,
    label_fn: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    enabled: bool = True,
) -> None:
    """Re-run the factor at alternative ``n_quantiles`` settings and record
    the IC/IR spread across variants.

    A factor whose headline IC collapses when you move from 5 to 3 or 10
    buckets is capturing a single quirky slice of the distribution rather
    than a monotone cross-sectional signal.  The spread fields surface that
    instability without requiring the caller to re-run the pipeline by hand.
    """
    defaults = {
        "param_sensitivity_n_variants": 0,
        "param_sensitivity_n_quantiles_grid": (),
        "param_sensitivity_mean_ic_min": float("nan"),
        "param_sensitivity_mean_ic_max": float("nan"),
        "param_sensitivity_mean_ic_std": float("nan"),
        "param_sensitivity_mean_ic_range": float("nan"),
        "param_sensitivity_long_short_ir_min": float("nan"),
        "param_sensitivity_long_short_ir_max": float("nan"),
        "param_sensitivity_long_short_ir_std": float("nan"),
        "param_sensitivity_long_short_ir_range": float("nan"),
    }
    metrics.update(defaults)
    if not enabled:
        return

    candidate_grid = tuple(q for q in (3, 5, 10) if q != base_n_quantiles and q >= 2)
    if not candidate_grid:
        return

    resolved_label_df = _resolve_variant_label_df(
        prices=prices,
        horizon=horizon,
        label_fn=label_fn,
        label_df=label_df,
    )

    ic_values: list[float] = []
    ir_values: list[float] = []
    used_grid: list[int] = []
    for q in candidate_grid:
        try:
            variant = _evaluate_variant_lightweight(
                factor_df=factor_df,
                label_df=resolved_label_df,
                n_quantiles=q,
            )
        except Exception:
            continue
        mic = float(variant.mean_ic)
        ir = float(variant.long_short_ir)
        if np.isfinite(mic):
            ic_values.append(mic)
        if np.isfinite(ir):
            ir_values.append(ir)
        used_grid.append(int(q))

    if not used_grid:
        return

    metrics["param_sensitivity_n_variants"] = len(used_grid)
    metrics["param_sensitivity_n_quantiles_grid"] = tuple(used_grid)
    if ic_values:
        arr = np.asarray(ic_values, dtype=float)
        metrics["param_sensitivity_mean_ic_min"] = float(arr.min())
        metrics["param_sensitivity_mean_ic_max"] = float(arr.max())
        metrics["param_sensitivity_mean_ic_std"] = float(arr.std(ddof=1)) if arr.size >= 2 else 0.0
        metrics["param_sensitivity_mean_ic_range"] = float(arr.max() - arr.min())
    if ir_values:
        arr = np.asarray(ir_values, dtype=float)
        metrics["param_sensitivity_long_short_ir_min"] = float(arr.min())
        metrics["param_sensitivity_long_short_ir_max"] = float(arr.max())
        metrics["param_sensitivity_long_short_ir_std"] = (
            float(arr.std(ddof=1)) if arr.size >= 2 else 0.0
        )
        metrics["param_sensitivity_long_short_ir_range"] = float(arr.max() - arr.min())


def _merge_baseline_factor_comparison_metrics(
    metrics: dict[str, object],
    *,
    prices: pd.DataFrame,
    factor_df: pd.DataFrame,
    horizon: int,
    n_quantiles: int,
    evaluation_config: ResearchEvaluationConfig,
    label_df: pd.DataFrame | None = None,
    label_fn: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    enabled: bool = True,
) -> None:
    """Compare the factor against cheap price-volume baselines (20-day
    momentum and 5-day reversal).

    If a researcher's "new" factor cannot outperform these two lines of
    pandas on the same universe, the headline IC is almost certainly a
    transformation of the same effect.  The comparison delta makes that
    obvious.
    """
    metrics["baseline_momentum_mean_ic"] = float("nan")
    metrics["baseline_momentum_long_short_ir"] = float("nan")
    metrics["baseline_momentum_factor_rank_corr"] = float("nan")
    metrics["baseline_reversal_mean_ic"] = float("nan")
    metrics["baseline_reversal_long_short_ir"] = float("nan")
    metrics["baseline_reversal_factor_rank_corr"] = float("nan")
    metrics["baseline_best_mean_ic"] = float("nan")
    metrics["baseline_factor_mean_ic_advantage"] = float("nan")
    if not enabled:
        return

    if "close" not in prices.columns:
        return

    try:
        mom_df = _build_price_baseline_factor(prices, lookback=20, reversal=False)
        rev_df = _build_price_baseline_factor(prices, lookback=5, reversal=True)
    except Exception:
        return

    resolved_label_df = _resolve_variant_label_df(
        prices=prices,
        horizon=horizon,
        label_fn=label_fn,
        label_df=label_df,
    )
    base_mean_ic = _finite_or_nan(_parse_optional_float(metrics.get("mean_ic")))
    best = float("nan")

    for name, base_df in (("momentum", mom_df), ("reversal", rev_df)):
        if base_df.empty:
            continue
        try:
            result = _evaluate_variant_lightweight(
                factor_df=base_df,
                label_df=resolved_label_df,
                n_quantiles=n_quantiles,
            )
        except Exception:
            continue
        mic = float(result.mean_ic)
        ir = float(result.long_short_ir)
        metrics[f"baseline_{name}_mean_ic"] = mic
        metrics[f"baseline_{name}_long_short_ir"] = ir
        try:
            rank_corr = _cross_sectional_rank_corr(factor_df, base_df)
        except Exception:
            rank_corr = float("nan")
        metrics[f"baseline_{name}_factor_rank_corr"] = float(rank_corr)
        if np.isfinite(mic):
            if not np.isfinite(best) or mic > best:
                best = mic

    if np.isfinite(best):
        metrics["baseline_best_mean_ic"] = best
        if np.isfinite(base_mean_ic):
            metrics["baseline_factor_mean_ic_advantage"] = base_mean_ic - best


def _build_price_baseline_factor(
    prices: pd.DataFrame,
    *,
    lookback: int,
    reversal: bool,
) -> pd.DataFrame:
    """Return a canonical long-form factor dataframe for a price baseline."""
    if lookback < 1:
        raise ValueError("lookback must be >= 1")
    ret_col = f"ret_{lookback}d"
    if ret_col in prices.columns:
        wide = prices[["date", "asset", ret_col]].copy()
        wide["_ret"] = pd.to_numeric(wide[ret_col], errors="coerce")
    else:
        wide = prices[["date", "asset", "close"]].copy()
        wide["close"] = pd.to_numeric(wide["close"], errors="coerce")
        wide = wide.dropna(subset=["close"])
        wide = wide.sort_values(["asset", "date"], kind="mergesort")
        grouped = wide.groupby("asset", sort=False, group_keys=False)
        wide["_ret"] = grouped["close"].pct_change(lookback)
    if reversal:
        wide["_ret"] = -wide["_ret"]
    factor_name = f"{'reversal' if reversal else 'momentum'}_{lookback}d"
    out = wide[["date", "asset", "_ret"]].rename(columns={"_ret": "value"}).dropna()
    out["factor"] = factor_name
    return out[["date", "asset", "factor", "value"]].reset_index(drop=True)


def _cross_sectional_rank_corr(
    left_factor: pd.DataFrame,
    right_factor: pd.DataFrame,
) -> float:
    """Mean per-date Spearman correlation between two factor series."""
    left = left_factor[["date", "asset", "value"]].rename(columns={"value": "_l"})
    right = right_factor[["date", "asset", "value"]].rename(columns={"value": "_r"})
    merged = left.merge(right, on=["date", "asset"], how="inner").dropna()
    if merged.empty:
        return float("nan")
    per_date = merged.groupby("date", sort=False)[["_l", "_r"]].apply(
        lambda g: g["_l"].corr(g["_r"], method="spearman")
    )
    per_date = per_date.replace([np.inf, -np.inf], np.nan).dropna()
    if per_date.empty:
        return float("nan")
    return float(per_date.mean())


def _merge_random_factor_baseline_metrics(
    metrics: dict[str, object],
    *,
    factor_df: pd.DataFrame,
    label_df: pd.DataFrame,
    n_permutations: int = 50,
    seed: int = 20260418,
    enabled: bool = True,
) -> pd.DataFrame:
    """Estimate a null distribution for mean rank-IC by shuffling the factor
    values within each date, and report a p-value for the observed value.

    This is the cheapest possible "is the observed IC distinguishable from
    noise on this universe?" check.  It does not replace Deflated Sharpe or
    Haircut Sharpe — those account for multiple testing — but it does catch
    the case where a factor's IC is numerically positive but inside the
    cross-sectional noise floor of this particular universe.
    """
    metrics["random_baseline_n_permutations"] = 0
    metrics["random_baseline_mean_ic_mean"] = float("nan")
    metrics["random_baseline_mean_ic_std"] = float("nan")
    metrics["random_baseline_mean_ic_p95"] = float("nan")
    metrics["random_baseline_mean_ic_p99"] = float("nan")
    metrics["random_baseline_p_value"] = float("nan")
    metrics["random_baseline_observed_z_score"] = float("nan")
    empty_null = pd.DataFrame(columns=["permutation", "mean_ic"])
    if not enabled:
        return empty_null

    if factor_df.empty or label_df.empty:
        return empty_null

    left = factor_df[["date", "asset", "value"]].rename(columns={"value": "_f"})
    right = label_df[["date", "asset", "value"]].rename(columns={"value": "_y"})
    merged = left.merge(right, on=["date", "asset"], how="inner").dropna()
    if merged.empty:
        return empty_null

    # Per-date observed rank-IC using Spearman.
    def _date_ic(frame: pd.DataFrame) -> float:
        if len(frame) < 3:
            return np.nan
        return float(frame["_f"].corr(frame["_y"], method="spearman"))

    observed_per_date = (
        merged.groupby("date", sort=False)
        .apply(_date_ic)
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if observed_per_date.empty:
        return empty_null
    observed_mean = float(observed_per_date.mean())

    rng = np.random.default_rng(seed)
    # Group labels (rank) and factor (rank) per date once, then permute.
    groups = list(merged.groupby("date", sort=False))
    date_ranks: list[tuple[np.ndarray, np.ndarray]] = []
    for _, frame in groups:
        if len(frame) < 3:
            continue
        f_rank = pd.Series(frame["_f"]).rank(method="average").to_numpy(dtype=float)
        y_rank = pd.Series(frame["_y"]).rank(method="average").to_numpy(dtype=float)
        date_ranks.append((f_rank, y_rank))
    if not date_ranks:
        return empty_null

    null_means: list[float] = []
    for _ in range(int(n_permutations)):
        per_date: list[float] = []
        for f_rank, y_rank in date_ranks:
            shuffled = rng.permutation(f_rank)
            # Pearson on ranks ≡ Spearman rho.
            f_mean = shuffled.mean()
            y_mean = y_rank.mean()
            num = float(((shuffled - f_mean) * (y_rank - y_mean)).sum())
            denom = float(
                np.sqrt(((shuffled - f_mean) ** 2).sum() * ((y_rank - y_mean) ** 2).sum())
            )
            if denom > 0.0:
                per_date.append(num / denom)
        if per_date:
            null_means.append(float(np.mean(per_date)))

    if not null_means:
        return empty_null

    arr = np.asarray(null_means, dtype=float)
    metrics["random_baseline_n_permutations"] = int(arr.size)
    metrics["random_baseline_mean_ic_mean"] = float(arr.mean())
    metrics["random_baseline_mean_ic_std"] = float(arr.std(ddof=1)) if arr.size >= 2 else 0.0
    metrics["random_baseline_mean_ic_p95"] = float(np.quantile(arr, 0.95))
    metrics["random_baseline_mean_ic_p99"] = float(np.quantile(arr, 0.99))
    tail = (
        float((arr >= observed_mean).sum())
        if observed_mean >= 0
        else float((arr <= observed_mean).sum())
    )
    metrics["random_baseline_p_value"] = (tail + 1.0) / (arr.size + 1.0)
    std = metrics["random_baseline_mean_ic_std"]
    if isinstance(std, float) and std > 0.0:
        metrics["random_baseline_observed_z_score"] = float((observed_mean - arr.mean()) / std)
    return pd.DataFrame(
        {
            "permutation": np.arange(1, arr.size + 1, dtype=int),
            "mean_ic": arr,
        }
    )


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
    top = int(qr["quantile"].max())
    bottom = int(qr["quantile"].min())
    if top == bottom:
        return empty_frame

    long_leg = qr[qr["quantile"] == top][["date", "mean_return"]].rename(
        columns={"mean_return": "long_leg"}
    )
    short_leg = qr[qr["quantile"] == bottom][["date", "mean_return"]].rename(
        columns={"mean_return": "short_leg"}
    )
    merged = long_leg.merge(short_leg, on="date", how="inner")
    if merged.empty:
        return empty_frame
    merged["gross"] = merged["long_leg"] - merged["short_leg"]

    turnover = result.long_short_turnover_df
    if cost_rate > 0.0 and turnover is not None and not turnover.empty:
        cost_series = turnover[["date", "long_short_turnover"]].rename(
            columns={"long_short_turnover": "turnover"}
        )
        merged = merged.merge(cost_series, on="date", how="left")
        merged["cost_drag"] = merged["turnover"].fillna(0.0) * float(cost_rate)
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


def _build_coverage_summary(
    *,
    n_dates_used: int,
    mean_eval_assets_per_date: float,
    eval_coverage_ratio_mean: float,
) -> str:
    if not np.isfinite(mean_eval_assets_per_date) or not np.isfinite(eval_coverage_ratio_mean):
        return f"n_dates={n_dates_used}"
    return (
        f"n_dates={n_dates_used}; avg_assets={mean_eval_assets_per_date:.1f}; "
        f"coverage={eval_coverage_ratio_mean:.1%}"
    )
