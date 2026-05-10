from __future__ import annotations

import concurrent.futures
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TypeVar

import numpy as np
import pandas as pd

from alpha_lab.decay import (
    compute_factor_autocorrelation,
    compute_ic_decay,
    estimate_ic_half_life,
)
from alpha_lab.experiment import ExperimentResult, run_factor_experiment
from alpha_lab.frame_utils import readonly_shallow_copy
from alpha_lab.grouped_evaluation import (
    conditional_ic_by_cross_section_size,
    conditional_ic_by_factor_magnitude,
)
from alpha_lab.reporting import summarise_experiment_result
from alpha_lab.reporting.campaign_triage import build_campaign_triage
from alpha_lab.reporting.factor_verdict import build_factor_verdict
from alpha_lab.reporting.level2_promotion import build_level2_promotion
from alpha_lab.research_evaluation_config import (
    DEFAULT_RESEARCH_EVALUATION_CONFIG,
    ResearchEvaluationConfig,
    research_evaluation_audit_snapshot,
)
from alpha_lab.splits import TimeSeriesSplitContract
from alpha_lab.validation.deflated_sharpe import deflated_sharpe_ratio

from ..spec import SingleFactorCaseSpec

# Cross-module imports (auto-added by split)
from ._utils import _parse_flags, _parse_optional_float, _parse_optional_int
from .capacity import (
    _build_capacity_estimation,
    _build_conditional_ic_summary,
    _empty_capacity_summary,
)
from .comparisons import (
    _build_neutralization_comparison,
    _merge_baseline_factor_comparison_metrics,
    _merge_execution_price_sensitivity_metrics,
    _merge_haircut_sharpe_metrics,
    _merge_marginal_contribution_metrics,
    _merge_neutralization_comparison_metrics,
    _merge_param_sensitivity_metrics,
    _merge_random_factor_baseline_metrics,
    _merge_tradability_metrics,
)
from .coverage import (
    _annotate_coverage_warmup,
    _build_coverage_summary,
    _build_effective_coverage_by_date,
    _count_coverage_break_days,
    _coverage_decision_frame,
    _coverage_warmup_summary,
    _merge_supplied_coverage_details,
    _summarise_effective_coverage,
    _with_split_phase,
)
from .data_quality import _build_data_quality_summary
from .diagnostics import (
    _empty_capacity_estimation_frame,
    _empty_conditional_ic_by_cross_section_size_frame,
    _empty_conditional_ic_by_magnitude_frame,
    _empty_factor_autocorrelation_frame,
    _empty_ic_decay_frame,
)
from .pnl_attribution import (
    _merge_daily_pnl_attribution_metrics,
    _merge_signal_lag_sensitivity_metrics,
)
from .strict_research import (
    _merge_dual_scope_report_metrics,
    _merge_strict_research_evidence_metrics,
)
from .summary_metrics import (
    _build_autocorr_lags,
    _build_decay_horizons,
    _build_group_monotonicity_summary,
    _build_ic_decay_half_life_summary,
    _compute_cost_adjusted_long_short_ir,
    _compute_ic_decay_rebalance_consistency,
    _compute_ic_decay_retention_ratio,
    _compute_rank_ic_ir,
    _resolve_rebalance_step,
)

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
    stage_timings: Mapping[str, float] = field(default_factory=dict)


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
    split_contract: TimeSeriesSplitContract | None = None,
) -> SingleFactorEvaluationResult:
    """Evaluate the single factor using the canonical experiment pipeline."""
    evaluate_started = time.perf_counter()
    diagnostics_cfg = evaluation_config.single_factor_diagnostics

    stage_timings: dict[str, float] = {}

    class _StageTimer:
        def __init__(self, name: str) -> None:
            self._name = name
            self._start = 0.0

        def __enter__(self) -> None:
            self._start = time.perf_counter()

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            traceback: object | None,
        ) -> None:
            del exc_type, exc, traceback
            stage_timings[self._name] = stage_timings.get(self._name, 0.0) + (
                time.perf_counter() - self._start
            )

    def _stage(name: str) -> _StageTimer:
        return _StageTimer(name)

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
    core_close_label_fn = _build_cached_close_label_fn(
        precomputed_forward_labels,
        horizon=spec.target.horizon,
        copy=False,
    )
    core_factor_df = readonly_shallow_copy(factor_df)
    raw_result: ExperimentResult | None = None

    with _stage("core_backtest"):
        result = _run_with_stage_heartbeat(
            stage_message="运行核心回测",
            stage_percent=5,
            fn=lambda: run_factor_experiment(
                prices,
                lambda _prices: core_factor_df,
                horizon=spec.target.horizon,
                n_quantiles=spec.n_quantiles,
                rolling_stability_thresholds=evaluation_config.rolling_stability,
                label_fn=core_close_label_fn,
                precomputed_forward_labels=precomputed_forward_labels,
                split_contract=split_contract,
            ),
        )

    _emit_progress("汇总核心指标", 18)
    cost_rate = spec.transaction_cost.one_way_rate
    with _stage("core_summary"):
        summary_df = summarise_experiment_result(
            result,
            cost_rate=cost_rate if cost_rate > 0 else None,
            evaluation_config=evaluation_config,
        )
        row = summary_df.iloc[0]

    report_result = result
    report_row: pd.Series | None = row
    is_result: ExperimentResult | None = None
    is_row: pd.Series | None = None
    if split_contract is not None:
        with _stage("core_backtest_full_report_path"):
            report_result = _run_with_stage_heartbeat(
                stage_message="生成双口径报告路径",
                stage_percent=22,
                fn=lambda: run_factor_experiment(
                    prices,
                    lambda _prices: core_factor_df,
                    horizon=spec.target.horizon,
                    n_quantiles=spec.n_quantiles,
                    rolling_stability_thresholds=evaluation_config.rolling_stability,
                    label_fn=core_close_label_fn,
                    precomputed_forward_labels=precomputed_forward_labels,
                    split_contract=None,
                ),
            )
        with _stage("core_summary_full_report_path"):
            report_summary_df = summarise_experiment_result(
                report_result,
                cost_rate=cost_rate if cost_rate > 0 else None,
                evaluation_config=evaluation_config,
            )
            report_row = report_summary_df.iloc[0]
        with _stage("core_backtest_is_report_path"):
            is_end = pd.Timestamp(split_contract.is_end)
            is_factor_df = core_factor_df.loc[
                pd.to_datetime(core_factor_df["date"], errors="coerce") <= is_end
            ]
            is_result = _run_with_stage_heartbeat(
                stage_message="生成 IS 指标口径",
                stage_percent=24,
                fn=lambda: run_factor_experiment(
                    prices,
                    lambda _prices: is_factor_df,
                    horizon=spec.target.horizon,
                    n_quantiles=spec.n_quantiles,
                    rolling_stability_thresholds=evaluation_config.rolling_stability,
                    label_fn=core_close_label_fn,
                    precomputed_forward_labels=precomputed_forward_labels,
                    split_contract=None,
                ),
            )
        with _stage("core_summary_is_report_path"):
            is_summary_df = summarise_experiment_result(
                is_result,
                cost_rate=cost_rate if cost_rate > 0 else None,
                evaluation_config=evaluation_config,
            )
            is_row = is_summary_df.iloc[0]

    raw_row: pd.Series | None = None
    if (
        diagnostics_cfg.run_neutralization_raw_comparison
        and spec.neutralization.enabled
        and raw_factor_df is not None
    ):
        with _stage("neutralization_raw_backtest"):
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
                    precomputed_forward_labels=precomputed_forward_labels,
                    split_contract=split_contract,
                ),
            )
        with _stage("neutralization_raw_summary"):
            raw_summary_df = summarise_experiment_result(
                raw_result,
                cost_rate=cost_rate if cost_rate > 0 else None,
                evaluation_config=evaluation_config,
            )
            raw_row = raw_summary_df.iloc[0]

    with _stage("ic_timeseries"):
        ic_timeseries = _build_ic_timeseries_frame(report_result)
    diagnostic_dates = _select_diagnostic_dates(
        factor_df,
        max_dates=diagnostics_cfg.diagnostic_max_dates,
        mode=diagnostics_cfg.diagnostic_sample_mode,
    )
    diagnostic_factor_df = _sample_diagnostic_frame(factor_df, diagnostic_dates=diagnostic_dates)
    diagnostic_label_df = _sample_diagnostic_frame(
        result.label_df,
        diagnostic_dates=diagnostic_dates,
    )
    diagnostic_sampled_dates = (
        int(len(pd.Index(diagnostic_dates).drop_duplicates()))
        if diagnostic_dates is not None
        else 0
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
    diagnostic_decay_label_cache = (
        {
            int(h): _sample_diagnostic_frame(labels, diagnostic_dates=diagnostic_dates)
            for h, labels in decay_label_cache.items()
        }
        if decay_label_cache is not None and diagnostic_dates is not None
        else decay_label_cache
    )
    decay_factor_df = (
        diagnostic_factor_df if diagnostic_decay_label_cache is not None else factor_df
    )
    with _stage("ic_decay"):
        ic_decay = (
            compute_ic_decay(
                factor_df=decay_factor_df,
                prices_df=prices,
                horizons=decay_horizons,
                precomputed_labels_by_horizon=diagnostic_decay_label_cache,
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

    with _stage("factor_autocorrelation"):
        factor_autocorrelation = (
            compute_factor_autocorrelation(
                factor_df=diagnostic_factor_df,
                lags=_build_autocorr_lags(),
            )
            if diagnostics_cfg.compute_factor_autocorrelation
            else _empty_factor_autocorrelation_frame()
        )

    with _stage("conditional_ic"):
        if diagnostics_cfg.compute_conditional_ic:
            conditional_by_magnitude = conditional_ic_by_factor_magnitude(
                factor_df=diagnostic_factor_df,
                labels_df=diagnostic_label_df,
            )
            conditional_by_cross_section = conditional_ic_by_cross_section_size(
                factor_df=diagnostic_factor_df,
                labels_df=diagnostic_label_df,
            )
        else:
            conditional_by_magnitude = _empty_conditional_ic_by_magnitude_frame()
            conditional_by_cross_section = _empty_conditional_ic_by_cross_section_size_frame()
        conditional_summary = _build_conditional_ic_summary(
            conditional_by_magnitude=conditional_by_magnitude,
            conditional_by_cross_section=conditional_by_cross_section,
        )
    rolling_stability = report_result.rolling_stability_df.copy()

    group_returns = report_result.quantile_returns_df.rename(
        columns={"quantile": "group", "mean_return": "group_return"}
    )
    turnover_source = (
        pd.concat(
            [is_result.long_short_turnover_df, result.long_short_turnover_df],
            ignore_index=True,
        )
        if split_contract is not None and is_result is not None
        else report_result.long_short_turnover_df
    )
    turnover = turnover_source.rename(columns={"long_short_turnover": "turnover"})
    oos_group_returns = result.quantile_returns_df.rename(
        columns={"quantile": "group", "mean_return": "group_return"}
    )
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
    with _stage("capacity_estimation"):
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
    rank_ic_ir = _compute_rank_ic_ir(result.rank_ic_df)
    group_monotonicity = _build_group_monotonicity_summary(
        group_returns=oos_group_returns,
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

    supplied_coverage_by_date = coverage_by_date
    coverage_by_date = _build_effective_coverage_by_date(
        prices=prices,
        factor_df=report_result.factor_df,
        label_df=report_result.label_df,
    )
    coverage_by_date = _merge_supplied_coverage_details(
        coverage_by_date,
        supplied_coverage_by_date,
    )
    coverage_by_date = _annotate_coverage_warmup(coverage_by_date)
    coverage_for_decision = _coverage_decision_frame(coverage_by_date)
    coverage_stats = _summarise_effective_coverage(coverage_for_decision)
    coverage_raw_stats = _summarise_effective_coverage(coverage_by_date)
    coverage_warmup_summary = _coverage_warmup_summary(coverage_by_date)
    coverage_mean = float(coverage_stats["mean_asset_coverage"])
    coverage_min = float(coverage_stats["min_asset_coverage"])
    coverage_break_days = (
        0
        if coverage_for_decision.empty
        else _count_coverage_break_days(
            coverage_for_decision,
            threshold=evaluation_config.campaign_triage.min_coverage_min_warn,
            drop_threshold=0.20,
        )
    )
    raw_coverage_break_days = (
        0
        if coverage_by_date.empty
        else _count_coverage_break_days(
            coverage_by_date,
            threshold=evaluation_config.campaign_triage.min_coverage_min_warn,
            drop_threshold=0.20,
        )
    )

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
    with _stage("data_quality_summary"):
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
        "date_coverage": float(coverage_stats["date_coverage"]),
        "n_coverage_dates": int(coverage_stats["n_dates"]),
        "n_valid_coverage_dates": int(coverage_stats["n_valid_dates"]),
        "avg_asset_coverage": float(coverage_stats["mean_asset_coverage"]),
        "mean_asset_coverage": float(coverage_stats["mean_asset_coverage"]),
        "median_asset_coverage": float(coverage_stats["median_asset_coverage"]),
        "min_asset_coverage": coverage_min,
        "max_asset_coverage": float(coverage_stats["max_asset_coverage"]),
        "overall_sample_coverage": float(coverage_stats["overall_sample_coverage"]),
        "sample_coverage": float(coverage_stats["overall_sample_coverage"]),
        "avg_assets": float(coverage_stats["avg_assets"]),
        "avg_valid_score_assets": float(coverage_stats["avg_valid_score_assets"]),
        "avg_valid_forward_return_assets": float(coverage_stats["avg_valid_forward_return_assets"]),
        "coverage_break_days": coverage_break_days,
        "coverage_break_days_raw": raw_coverage_break_days,
        "coverage_break_days_ex_warmup": coverage_break_days,
        "coverage_warmup_excluded_days": int(
            coverage_warmup_summary["coverage_warmup_excluded_days"]
        ),
        "coverage_warmup_start": coverage_warmup_summary["coverage_warmup_start"],
        "coverage_warmup_end": coverage_warmup_summary["coverage_warmup_end"],
        "coverage_warmup_policy": (
            "leading dates before first positive scored asset count are excluded "
            "from promotion coverage stats"
        ),
        "coverage_mean_raw": float(coverage_raw_stats["mean_asset_coverage"]),
        "coverage_min_raw": float(coverage_raw_stats["min_asset_coverage"]),
        "coverage_summary": _build_coverage_summary(
            coverage_stats=coverage_stats,
            warmup_summary=coverage_warmup_summary,
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
        "diagnostic_max_dates": diagnostics_cfg.diagnostic_max_dates,
        "diagnostic_sample_mode": diagnostics_cfg.diagnostic_sample_mode,
        "diagnostic_sampled_dates": diagnostic_sampled_dates,
        "single_factor_diagnostics_mode": "streamlined"
        if _skipped_single_factor_diagnostics(diagnostics_cfg)
        else "full",
        "single_factor_skipped_diagnostics": _skipped_single_factor_diagnostics(diagnostics_cfg),
        **capacity_summary,
        **conditional_summary,
        **data_quality_summary,
    }
    _merge_dual_scope_report_metrics(
        metrics,
        oos_result=result,
        full_result=report_result if report_result is not result else None,
        is_result=is_result,
        oos_row=row,
        full_row=report_row if report_result is not result else None,
        is_row=is_row,
        cost_rate=cost_rate,
    )
    _merge_strict_research_evidence_metrics(
        metrics,
        full_result=report_result if report_result is not result else None,
        contract=split_contract,
        enabled=evaluation_config.profile_name == "stricter_research",
    )
    if result.split_contract is not None:
        metrics.update(_split_contract_metrics(result.split_contract))
    _emit_progress("运行附加稳健性诊断", 62)
    with _stage("marginal_contribution"):
        _merge_marginal_contribution_metrics(
            metrics,
            factor_df=factor_df,
            label_df=result.label_df,
            enabled=diagnostics_cfg.run_marginal_contribution,
        )
    with _stage("tradability"):
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
    with _stage("execution_price_sensitivity"):
        _merge_execution_price_sensitivity_metrics(
            metrics,
            prices=prices,
            factor_df=factor_df,
            horizon=spec.target.horizon,
            n_quantiles=spec.n_quantiles,
            evaluation_config=evaluation_config,
            cost_rate=cost_rate,
            split_contract=split_contract,
            enabled=diagnostics_cfg.run_execution_price_sensitivity,
        )
    with _stage("haircut_sharpe"):
        _merge_haircut_sharpe_metrics(metrics)
    with _stage("param_sensitivity"):
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
    with _stage("baseline_factor_comparison"):
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
    with _stage("random_baseline_null"):
        random_baseline_null_df = _merge_random_factor_baseline_metrics(
            metrics,
            factor_df=factor_df,
            label_df=result.label_df,
            n_permutations=20 if evaluation_config.profile_name == "exploratory_screening" else 50,
            enabled=diagnostics_cfg.run_random_baseline,
        )
    with _stage("daily_pnl_attribution"):
        daily_pnl_attribution_df = _merge_daily_pnl_attribution_metrics(
            metrics,
            result=result,
            cost_rate=cost_rate,
        )
    with _stage("lag_sensitivity"):
        lag_sensitivity_df = _merge_signal_lag_sensitivity_metrics(
            metrics,
            prices=prices,
            factor_df=diagnostic_factor_df,
            horizon=spec.target.horizon,
            n_quantiles=spec.n_quantiles,
            evaluation_config=evaluation_config,
            label_df=diagnostic_label_df,
            base_result=result if diagnostic_dates is None else None,
            label_fn=close_label_fn,
            split_contract=split_contract,
            enabled=diagnostics_cfg.run_lag_sensitivity,
        )
    _emit_progress("汇总结论与分层判定", 86)
    with _stage("neutralization_comparison"):
        if raw_row is not None:
            comparison = _build_neutralization_comparison(
                raw_row=raw_row,
                neutralized_row=row,
                neutralization_mean_corr_reduction=neutralization_mean_corr_reduction,
                thresholds=evaluation_config.neutralization_comparison,
            )
            _merge_neutralization_comparison_metrics(metrics, comparison=comparison)

    with _stage("factor_verdict"):
        verdict = build_factor_verdict(
            metrics,
            thresholds=evaluation_config.factor_verdict,
        )
        metrics["factor_verdict"] = verdict.label
        metrics["factor_verdict_reasons"] = list(verdict.reasons)
    with _stage("campaign_triage"):
        metrics.update(
            build_campaign_triage(
                metrics,
                thresholds=evaluation_config.campaign_triage,
            ).to_dict()
        )
    with _stage("level2_promotion"):
        metrics.update(
            build_level2_promotion(
                metrics,
                thresholds=evaluation_config.level2_promotion,
            ).to_dict()
        )

    with _stage("result_packaging"):
        packaged_ic_timeseries = (
            _with_split_phase(
                ic_timeseries,
                split_contract,
                drop_embargo=True,
            )
            .sort_values(
                "date",
                kind="mergesort",
            )
            .reset_index(drop=True)
        )
        packaged_ic_decay = ic_decay.sort_values("horizon", kind="mergesort").reset_index(drop=True)
        packaged_factor_autocorrelation = factor_autocorrelation.sort_values(
            "lag",
            kind="mergesort",
        ).reset_index(drop=True)
        packaged_rolling_stability = (
            _with_split_phase(
                rolling_stability,
                split_contract,
                drop_embargo=True,
            )
            .sort_values(
                "date",
                kind="mergesort",
            )
            .reset_index(drop=True)
        )
        packaged_group_returns = (
            _with_split_phase(
                group_returns,
                split_contract,
                drop_embargo=True,
            )
            .sort_values(
                ["date", "group"],
                kind="mergesort",
            )
            .reset_index(drop=True)
        )
        packaged_turnover = (
            _with_split_phase(
                turnover,
                split_contract,
                drop_embargo=True,
            )
            .sort_values("date", kind="mergesort")
            .reset_index(drop=True)
        )
        packaged_coverage = (
            _with_split_phase(
                coverage_by_date,
                split_contract,
                drop_embargo=False,
            )
            .sort_values(
                "date",
                kind="mergesort",
            )
            .reset_index(drop=True)
        )

    final_stage_timings = dict(stage_timings)
    evaluate_total = time.perf_counter() - evaluate_started
    named_children_total = sum(final_stage_timings.values())
    _merge_experiment_stage_timings(
        final_stage_timings,
        prefix="core_backtest",
        experiment_stage_timings=result.stage_timings,
    )
    if raw_result is not None:
        _merge_experiment_stage_timings(
            final_stage_timings,
            prefix="neutralization_raw_backtest",
            experiment_stage_timings=raw_result.stage_timings,
        )
    final_stage_timings["evaluate_total"] = evaluate_total
    final_stage_timings["evaluate_named_children_total"] = named_children_total
    final_stage_timings["evaluate_other"] = max(evaluate_total - named_children_total, 0.0)

    return SingleFactorEvaluationResult(
        experiment_result=result,
        metrics=metrics,
        ic_timeseries=packaged_ic_timeseries,
        ic_decay=packaged_ic_decay,
        factor_autocorrelation=packaged_factor_autocorrelation,
        rolling_stability=packaged_rolling_stability,
        group_returns=packaged_group_returns,
        turnover=packaged_turnover,
        coverage=packaged_coverage,
        capacity_estimation=capacity_estimation,
        conditional_ic_by_magnitude=conditional_by_magnitude,
        conditional_ic_by_cross_section_size=conditional_by_cross_section,
        neutralization_summary=neutral_df,
        lag_sensitivity=lag_sensitivity_df,
        random_baseline_null=random_baseline_null_df,
        daily_pnl_attribution=daily_pnl_attribution_df,
        stage_timings=final_stage_timings,
    )


def _build_ic_timeseries_frame(result: ExperimentResult) -> pd.DataFrame:
    """Build the report IC timeseries from one experiment result."""

    ic_df = (
        result.ic_df[["date", "ic"]].copy()
        if {"date", "ic"}.issubset(result.ic_df.columns)
        else pd.DataFrame(columns=["date", "ic"])
    )
    rank_ic_df = (
        result.rank_ic_df[["date", "rank_ic"]].copy()
        if {"date", "rank_ic"}.issubset(result.rank_ic_df.columns)
        else pd.DataFrame(columns=["date", "rank_ic"])
    )
    mi_df = (
        result.mutual_information_df[["date", "mutual_information"]].copy()
        if {"date", "mutual_information"}.issubset(result.mutual_information_df.columns)
        else pd.DataFrame(columns=["date", "mutual_information"])
    )
    return (
        ic_df.merge(rank_ic_df, on="date", how="outer", sort=True)
        .merge(mi_df, on="date", how="outer", sort=True)
        .reset_index(drop=True)
    )


def _build_cached_close_label_fn(
    precomputed_forward_labels: Mapping[int, pd.DataFrame] | None,
    *,
    horizon: int,
    copy: bool = True,
) -> Callable[[pd.DataFrame], pd.DataFrame] | None:
    if precomputed_forward_labels is None:
        return None
    cached = precomputed_forward_labels.get(int(horizon))
    if cached is None:
        return None

    def _label_fn(_prices: pd.DataFrame) -> pd.DataFrame:
        if copy:
            return cached.copy()
        return readonly_shallow_copy(cached)

    return _label_fn


def _split_contract_metrics(contract: TimeSeriesSplitContract) -> dict[str, object]:
    metadata = contract.to_metadata()
    return {
        "split_contract": metadata,
        "split_policy": metadata["policy"],
        "split_source": metadata["source"],
        "is_start": metadata["is_start"],
        "is_end": metadata["is_end"],
        "oos_start": metadata["oos_start"],
        "oos_end": metadata["oos_end"],
        "split_embargo_days": metadata["embargo_days"],
        "split_min_oos_dates": metadata["min_oos_dates"],
        "split_min_is_dates": metadata["min_is_dates"],
        "split_n_dates": metadata["n_dates"],
        "split_n_is_dates": metadata["n_is_dates"],
        "split_n_oos_dates": metadata["n_oos_dates"],
        "split_target_horizon": metadata["target_horizon"],
        "split_rebalance_step": metadata["rebalance_step"],
    }


def _merge_experiment_stage_timings(
    target: dict[str, float],
    *,
    prefix: str,
    experiment_stage_timings: Mapping[str, float],
) -> None:
    for name, seconds in experiment_stage_timings.items():
        target[f"{prefix}.{name}"] = float(seconds)


def _select_diagnostic_dates(
    factor_df: pd.DataFrame,
    *,
    max_dates: int | None,
    mode: str,
) -> pd.DatetimeIndex | None:
    if max_dates is None or max_dates <= 0 or factor_df.empty or "date" not in factor_df.columns:
        return None
    dates = pd.DatetimeIndex(pd.to_datetime(factor_df["date"], errors="coerce").dropna().unique())
    dates = dates.sort_values()
    if len(dates) <= max_dates:
        return None
    if mode == "even":
        positions = np.linspace(0, len(dates) - 1, num=int(max_dates), dtype=int)
        return pd.DatetimeIndex(dates.take(np.unique(positions))).sort_values()
    return pd.DatetimeIndex(dates[-int(max_dates) :]).sort_values()


def _sample_diagnostic_frame(
    frame: pd.DataFrame,
    *,
    diagnostic_dates: pd.DatetimeIndex | None,
) -> pd.DataFrame:
    if diagnostic_dates is None or frame.empty or "date" not in frame.columns:
        return frame
    sampled = frame[pd.to_datetime(frame["date"], errors="coerce").isin(diagnostic_dates)].copy()
    return sampled.reset_index(drop=True)


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
