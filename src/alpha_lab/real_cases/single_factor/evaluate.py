from __future__ import annotations

import concurrent.futures
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import TypeVar

import numpy as np
import pandas as pd

from alpha_lab.baseline_factor_suite import (
    baseline_required_columns_available,
    iter_baseline_factor_specs,
)
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
from alpha_lab.evaluation import (
    compute_ic,
    compute_ic_summary,
    compute_mean_rank_ic_permutation_null,
)
from alpha_lab.experiment import ExperimentResult, run_factor_experiment
from alpha_lab.factor_recipe import build_factor_from_recipe_mapping
from alpha_lab.frame_utils import readonly_shallow_copy
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
from alpha_lab.splits import TimeSeriesSplitContract
from alpha_lab.validation.deflated_sharpe import deflated_sharpe_ratio
from alpha_lab.validation.haircut_sharpe import haircut_sharpe_ratio

from .spec import SingleFactorCaseSpec

T = TypeVar("T")

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
    turnover = turnover_source.rename(
        columns={"long_short_turnover": "turnover"}
    )
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
        "avg_valid_forward_return_assets": float(
            coverage_stats["avg_valid_forward_return_assets"]
        ),
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
            n_permutations=20
            if evaluation_config.profile_name == "exploratory_screening"
            else 50,
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
        packaged_ic_timeseries = _with_split_phase(
            ic_timeseries,
            split_contract,
            drop_embargo=True,
        ).sort_values(
            "date",
            kind="mergesort",
        ).reset_index(drop=True)
        packaged_ic_decay = ic_decay.sort_values("horizon", kind="mergesort").reset_index(
            drop=True
        )
        packaged_factor_autocorrelation = factor_autocorrelation.sort_values(
            "lag",
            kind="mergesort",
        ).reset_index(drop=True)
        packaged_rolling_stability = _with_split_phase(
            rolling_stability,
            split_contract,
            drop_embargo=True,
        ).sort_values(
            "date",
            kind="mergesort",
        ).reset_index(drop=True)
        packaged_group_returns = _with_split_phase(
            group_returns,
            split_contract,
            drop_embargo=True,
        ).sort_values(
            ["date", "group"],
            kind="mergesort",
        ).reset_index(drop=True)
        packaged_turnover = _with_split_phase(
            turnover,
            split_contract,
            drop_embargo=True,
        ).sort_values("date", kind="mergesort").reset_index(drop=True)
        packaged_coverage = _with_split_phase(
            coverage_by_date,
            split_contract,
            drop_embargo=False,
        ).sort_values(
            "date",
            kind="mergesort",
        ).reset_index(drop=True)

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


def _with_split_phase(
    frame: pd.DataFrame,
    contract: TimeSeriesSplitContract | None,
    *,
    drop_embargo: bool,
) -> pd.DataFrame:
    """Annotate report rows with IS/OOS phase from the strict split contract."""

    if contract is None or frame.empty or "date" not in frame.columns:
        return frame.copy()

    out = frame.copy()
    dates = pd.to_datetime(out["date"], errors="coerce")
    is_end = pd.Timestamp(contract.is_end)
    oos_start = pd.Timestamp(contract.oos_start)
    out["split_phase"] = np.select(
        [dates <= is_end, dates >= oos_start],
        ["IS", "OOS"],
        default="EMBARGO",
    )
    if drop_embargo:
        out = out.loc[out["split_phase"] != "EMBARGO"]
    return out.reset_index(drop=True)


def _build_effective_coverage_by_date(
    *,
    prices: pd.DataFrame,
    factor_df: pd.DataFrame,
    label_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute coverage against the eligible universe, not the scored subset."""

    columns = [
        "date",
        "eligible_count",
        "valid_score_count",
        "valid_forward_return_count",
        "valid_sample_count",
        "asset_coverage",
        "forward_return_coverage",
        "sample_coverage",
        "coverage",
        "missingness",
        "n_assets",
        "n_non_null",
        "missing_score_count",
        "missing_forward_return_count",
        "invalid_sample_count",
        "universe_count",
        "feature_row_count",
        "complete_feature_count",
        "feature_nan_row_count",
        "label_available_count",
        "scored_count",
        "scored_evaluable_count",
        "missing_feature_count",
        "missing_label_count",
        "filtered_count",
        "score_coverage",
        "universe_coverage",
    ]
    if prices.empty or not {"date", "asset"}.issubset(prices.columns):
        return pd.DataFrame(columns=columns)

    eligible = prices[["date", "asset"]].copy()
    eligible["date"] = pd.to_datetime(eligible["date"], errors="coerce")
    eligible = eligible.dropna(subset=["date", "asset"]).drop_duplicates()
    if eligible.empty:
        return pd.DataFrame(columns=columns)
    eligible["date"] = eligible["date"].dt.normalize()
    eligible["asset"] = eligible["asset"].astype(str)

    def _valid_flags(frame: pd.DataFrame, flag_col: str) -> pd.DataFrame:
        if frame.empty or not {"date", "asset", "value"}.issubset(frame.columns):
            return pd.DataFrame(columns=["date", "asset", flag_col])
        out = frame[["date", "asset", "value"]].copy()
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out = out.dropna(subset=["date", "asset"])
        if out.empty:
            return pd.DataFrame(columns=["date", "asset", flag_col])
        out["date"] = out["date"].dt.normalize()
        out["asset"] = out["asset"].astype(str)
        out[flag_col] = pd.to_numeric(out["value"], errors="coerce").notna()
        return (
            out.groupby(["date", "asset"], as_index=False)[flag_col]
            .max()
            .reset_index(drop=True)
        )

    score_flags = _valid_flags(factor_df, "has_valid_score")
    label_flags = _valid_flags(label_df, "has_valid_forward_return")
    merged = eligible.merge(score_flags, on=["date", "asset"], how="left")
    merged = merged.merge(label_flags, on=["date", "asset"], how="left")
    merged["has_valid_score"] = merged["has_valid_score"].fillna(False).astype(bool)
    merged["has_valid_forward_return"] = (
        merged["has_valid_forward_return"].fillna(False).astype(bool)
    )
    merged["has_valid_sample"] = (
        merged["has_valid_score"] & merged["has_valid_forward_return"]
    )

    summary = merged.groupby("date", sort=True).agg(
        eligible_count=("asset", "nunique"),
        valid_score_count=("has_valid_score", "sum"),
        valid_forward_return_count=("has_valid_forward_return", "sum"),
        valid_sample_count=("has_valid_sample", "sum"),
    )
    count_cols = [
        "eligible_count",
        "valid_score_count",
        "valid_forward_return_count",
        "valid_sample_count",
    ]
    for col in count_cols:
        summary[col] = summary[col].astype(int)
    denominator = summary["eligible_count"].replace(0, pd.NA)
    summary["asset_coverage"] = summary["valid_score_count"] / denominator
    summary["forward_return_coverage"] = (
        summary["valid_forward_return_count"] / denominator
    )
    summary["sample_coverage"] = summary["valid_sample_count"] / denominator
    summary["coverage"] = summary["asset_coverage"]
    summary["missingness"] = 1.0 - summary["asset_coverage"]
    summary["n_assets"] = summary["eligible_count"]
    summary["n_non_null"] = summary["valid_score_count"]
    summary["missing_score_count"] = (
        summary["eligible_count"] - summary["valid_score_count"]
    )
    summary["missing_forward_return_count"] = (
        summary["eligible_count"] - summary["valid_forward_return_count"]
    )
    summary["invalid_sample_count"] = (
        summary["eligible_count"] - summary["valid_sample_count"]
    )
    summary["universe_count"] = summary["eligible_count"]
    summary["feature_row_count"] = summary["eligible_count"]
    summary["complete_feature_count"] = summary["eligible_count"]
    summary["feature_nan_row_count"] = 0
    summary["label_available_count"] = summary["valid_forward_return_count"]
    summary["scored_count"] = summary["valid_score_count"]
    summary["scored_evaluable_count"] = summary["valid_sample_count"]
    summary["missing_feature_count"] = 0
    summary["missing_label_count"] = summary["missing_forward_return_count"]
    summary["filtered_count"] = summary["missing_forward_return_count"]
    summary["score_coverage"] = summary["asset_coverage"]
    summary["universe_coverage"] = summary["asset_coverage"]
    return summary.reset_index()[columns]


def _merge_supplied_coverage_details(
    effective: pd.DataFrame,
    supplied: pd.DataFrame | None,
) -> pd.DataFrame:
    if effective.empty or supplied is None or supplied.empty or "date" not in supplied.columns:
        return effective
    detail_columns = [
        "universe_count",
        "feature_row_count",
        "complete_feature_count",
        "feature_nan_row_count",
        "label_available_count",
        "scored_count",
        "scored_evaluable_count",
        "missing_feature_count",
        "missing_label_count",
        "filtered_count",
        "score_coverage",
        "universe_coverage",
    ]
    available = [column for column in detail_columns if column in supplied.columns]
    if not available:
        return effective

    left = effective.copy()
    left["date"] = pd.to_datetime(left["date"], errors="coerce")
    right = supplied.loc[:, ["date", *available]].copy()
    right["date"] = pd.to_datetime(right["date"], errors="coerce")
    right = right.dropna(subset=["date"]).drop_duplicates(subset=["date"], keep="last")
    merged = left.merge(
        right,
        on="date",
        how="left",
        suffixes=("", "_supplied"),
        validate="one_to_one",
    )
    for column in available:
        supplied_column = f"{column}_supplied"
        if supplied_column not in merged.columns:
            continue
        supplied_values = pd.to_numeric(merged[supplied_column], errors="coerce")
        base_values = pd.to_numeric(merged[column], errors="coerce")
        merged[column] = supplied_values.combine_first(base_values)
        merged = merged.drop(columns=[supplied_column])
    return merged


def _annotate_coverage_warmup(coverage_by_date: pd.DataFrame) -> pd.DataFrame:
    """Mark leading no-score dates as warmup-only diagnostics.

    Model-factor runs can legitimately emit zero scored assets before the first
    walk-forward model is fitted. Those leading dates should stay visible in
    coverage.csv, but they should not drive promotion coverage blockers.
    """

    out = coverage_by_date.copy()
    if out.empty:
        out["is_warmup"] = pd.Series(dtype=bool)
        out["coverage_eval_included"] = pd.Series(dtype=bool)
        return out
    if "date" in out.columns:
        out["_coverage_date_sort"] = pd.to_datetime(out["date"], errors="coerce")
        out = out.sort_values("_coverage_date_sort", kind="mergesort").drop(
            columns=["_coverage_date_sort"]
        )

    score_count = pd.Series(0.0, index=out.index, dtype=float)
    for column in ("valid_score_count", "scored_count", "n_non_null"):
        if column in out.columns:
            values = pd.to_numeric(out[column], errors="coerce")
            score_count = values.combine_first(score_count)
            break

    has_score = score_count.fillna(0.0) > 0
    if bool(has_score.any()):
        first_scored_position = int(np.flatnonzero(has_score.to_numpy())[0])
        warmup_mask = pd.Series(False, index=out.index)
        if first_scored_position > 0:
            warmup_mask.iloc[:first_scored_position] = True
    else:
        warmup_mask = pd.Series(False, index=out.index)

    out["is_warmup"] = warmup_mask.astype(bool)
    out["coverage_eval_included"] = ~out["is_warmup"]
    return out


def _coverage_decision_frame(coverage_by_date: pd.DataFrame) -> pd.DataFrame:
    if coverage_by_date.empty or "coverage_eval_included" not in coverage_by_date.columns:
        return coverage_by_date
    included = coverage_by_date["coverage_eval_included"].fillna(True).astype(bool)
    return coverage_by_date.loc[included].reset_index(drop=True)


def _coverage_warmup_summary(coverage_by_date: pd.DataFrame) -> dict[str, object]:
    if coverage_by_date.empty or "is_warmup" not in coverage_by_date.columns:
        return {
            "coverage_warmup_excluded_days": 0,
            "coverage_warmup_start": None,
            "coverage_warmup_end": None,
        }
    warmup = coverage_by_date.loc[coverage_by_date["is_warmup"].fillna(False).astype(bool)]
    if warmup.empty:
        return {
            "coverage_warmup_excluded_days": 0,
            "coverage_warmup_start": None,
            "coverage_warmup_end": None,
        }
    dates = (
        pd.to_datetime(warmup["date"], errors="coerce")
        if "date" in warmup.columns
        else pd.Series(dtype="datetime64[ns]")
    )
    dates = dates.dropna()
    return {
        "coverage_warmup_excluded_days": int(len(warmup)),
        "coverage_warmup_start": dates.min().strftime("%Y-%m-%d") if not dates.empty else None,
        "coverage_warmup_end": dates.max().strftime("%Y-%m-%d") if not dates.empty else None,
    }


def _summarise_effective_coverage(coverage_by_date: pd.DataFrame) -> dict[str, float | int]:
    if coverage_by_date.empty:
        return {
            "n_dates": 0,
            "n_valid_dates": 0,
            "date_coverage": float("nan"),
            "mean_asset_coverage": float("nan"),
            "median_asset_coverage": float("nan"),
            "min_asset_coverage": float("nan"),
            "max_asset_coverage": float("nan"),
            "overall_sample_coverage": float("nan"),
            "avg_assets": float("nan"),
            "avg_valid_score_assets": float("nan"),
            "avg_valid_forward_return_assets": float("nan"),
    }

    def _numeric(col: str) -> pd.Series:
        if col not in coverage_by_date.columns:
            return pd.Series(dtype=float)
        return pd.to_numeric(coverage_by_date[col], errors="coerce")

    eligible = _numeric("eligible_count")
    valid_scores = _numeric("valid_score_count")
    valid_labels = _numeric("valid_forward_return_count")
    valid_samples = _numeric("valid_sample_count")
    asset_coverage = _numeric("asset_coverage").replace([np.inf, -np.inf], np.nan)
    n_dates = (
        int(coverage_by_date["date"].nunique())
        if "date" in coverage_by_date.columns
        else int(len(coverage_by_date))
    )
    n_valid_dates = int((valid_samples.fillna(0) > 0).sum())
    total_eligible = float(eligible.sum())
    total_valid_samples = float(valid_samples.sum())
    return {
        "n_dates": n_dates,
        "n_valid_dates": n_valid_dates,
        "date_coverage": float(n_valid_dates / n_dates) if n_dates else float("nan"),
        "mean_asset_coverage": float(asset_coverage.mean()),
        "median_asset_coverage": float(asset_coverage.median()),
        "min_asset_coverage": float(asset_coverage.min()),
        "max_asset_coverage": float(asset_coverage.max()),
        "overall_sample_coverage": (
            float(total_valid_samples / total_eligible)
            if total_eligible > 0
            else float("nan")
        ),
        "avg_assets": float(eligible.mean()),
        "avg_valid_score_assets": float(valid_scores.mean()),
        "avg_valid_forward_return_assets": float(valid_labels.mean()),
    }


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
    metrics["ls_max_drawdown_duration_full"] = int(
        full_result.summary.ls_max_drawdown_duration
    )
    metrics["ls_max_drawdown_duration_oos"] = int(
        oos_result.summary.ls_max_drawdown_duration
    )
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
        metrics["ls_max_drawdown_duration_is"] = int(
            is_result.summary.ls_max_drawdown_duration
        )
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


def _safe_ratio(numerator: float | None, denominator: float | None) -> float:
    if numerator is None or denominator is None:
        return float("nan")
    if not np.isfinite(numerator) or not np.isfinite(denominator):
        return float("nan")
    if abs(denominator) <= 1e-12:
        return float("nan")
    return float(numerator / denominator)


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


def _finite_series_mean(values: pd.Series) -> float:
    arr = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(arr.mean()) if not arr.empty else float("nan")


def _count_coverage_break_days(
    coverage_by_date: pd.DataFrame,
    *,
    threshold: float,
    drop_threshold: float,
) -> int:
    if coverage_by_date.empty or "coverage" not in coverage_by_date.columns:
        return 0
    coverage = pd.to_numeric(coverage_by_date["coverage"], errors="coerce").replace(
        [np.inf, -np.inf],
        np.nan,
    )
    low = coverage < float(threshold)
    abrupt_drop = coverage.diff() <= -abs(float(drop_threshold))
    return int((low | abrupt_drop).fillna(False).sum())


def _metric_label(value: str) -> str:
    label = "".join(ch.lower() if ch.isalnum() else "_" for ch in value.strip())
    return "_".join(part for part in label.split("_") if part) or "unknown"


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
    split_contract: TimeSeriesSplitContract | None = None,
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
        next_open_label_df = forward_return(
            prices,
            horizon=horizon,
            execution_price_mode="next_open",
        )
        if split_contract is not None:
            oos_start = pd.Timestamp(split_contract.oos_start)
            factor_dates = pd.to_datetime(factor_df["date"], errors="coerce")
            label_dates = pd.to_datetime(next_open_label_df["date"], errors="coerce")
            factor_for_variant = factor_df.loc[factor_dates >= oos_start].reset_index(drop=True)
            label_for_variant = next_open_label_df.loc[label_dates >= oos_start].reset_index(
                drop=True
            )
        else:
            factor_for_variant = factor_df
            label_for_variant = next_open_label_df
        next_open_summary = _evaluate_variant_lightweight(
            factor_df=factor_for_variant,
            label_df=label_for_variant,
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
    """Compare the factor against the default research baseline suite."""
    metrics["baseline_momentum_mean_ic"] = float("nan")
    metrics["baseline_momentum_long_short_ir"] = float("nan")
    metrics["baseline_momentum_factor_rank_corr"] = float("nan")
    metrics["baseline_reversal_mean_ic"] = float("nan")
    metrics["baseline_reversal_long_short_ir"] = float("nan")
    metrics["baseline_reversal_factor_rank_corr"] = float("nan")
    metrics["baseline_best_mean_ic"] = float("nan")
    metrics["baseline_suite_best_name"] = ""
    metrics["baseline_suite_best_family"] = ""
    metrics["baseline_suite_best_mean_ic"] = float("nan")
    metrics["baseline_suite_best_long_short_ir"] = float("nan")
    metrics["baseline_suite_best_factor_rank_corr"] = float("nan")
    metrics["baseline_suite_count"] = 0
    metrics["baseline_suite_evaluated_count"] = 0
    metrics["baseline_suite_skipped_count"] = 0
    metrics["baseline_suite_evaluated_names"] = ()
    metrics["baseline_suite_skipped_names"] = ()
    metrics["baseline_factor_mean_ic_advantage"] = float("nan")
    if not enabled:
        return

    if "close" not in prices.columns:
        return

    resolved_label_df = _resolve_variant_label_df(
        prices=prices,
        horizon=horizon,
        label_fn=label_fn,
        label_df=label_df,
    )
    base_mean_ic = _finite_or_nan(_parse_optional_float(metrics.get("mean_ic")))
    best = float("nan")
    best_name = ""
    best_family = ""
    best_ir = float("nan")
    best_rank_corr = float("nan")
    evaluated_names: list[str] = []
    skipped_names: list[str] = []
    specs = iter_baseline_factor_specs()
    metrics["baseline_suite_count"] = len(specs)

    for spec in specs:
        if not baseline_required_columns_available(spec, prices.columns):
            skipped_names.append(spec.name)
            continue
        try:
            base_df = build_factor_from_recipe_mapping(
                prices=prices,
                recipe=spec.recipe,
                factor_name=spec.name,
            )
            base_df["date"] = pd.to_datetime(base_df["date"], errors="coerce")
        except Exception:
            skipped_names.append(spec.name)
            continue
        if base_df.empty:
            skipped_names.append(spec.name)
            continue
        try:
            result = _evaluate_variant_lightweight(
                factor_df=base_df,
                label_df=resolved_label_df,
                n_quantiles=n_quantiles,
            )
        except Exception:
            skipped_names.append(spec.name)
            continue
        evaluated_names.append(spec.name)
        mic = float(result.mean_ic)
        ir = float(result.long_short_ir)
        metrics[f"baseline_{spec.name}_mean_ic"] = mic
        metrics[f"baseline_{spec.name}_long_short_ir"] = ir
        try:
            rank_corr = _cross_sectional_rank_corr(factor_df, base_df)
        except Exception:
            rank_corr = float("nan")
        metrics[f"baseline_{spec.name}_factor_rank_corr"] = float(rank_corr)
        if spec.name == "mom_20d":
            metrics["baseline_momentum_mean_ic"] = mic
            metrics["baseline_momentum_long_short_ir"] = ir
            metrics["baseline_momentum_factor_rank_corr"] = float(rank_corr)
        elif spec.name == "rev_5d":
            metrics["baseline_reversal_mean_ic"] = mic
            metrics["baseline_reversal_long_short_ir"] = ir
            metrics["baseline_reversal_factor_rank_corr"] = float(rank_corr)
        if np.isfinite(mic):
            if not np.isfinite(best) or mic > best:
                best = mic
                best_name = spec.name
                best_family = spec.family
                best_ir = ir
                best_rank_corr = float(rank_corr)

    metrics["baseline_suite_evaluated_count"] = len(evaluated_names)
    metrics["baseline_suite_skipped_count"] = len(skipped_names)
    metrics["baseline_suite_evaluated_names"] = tuple(evaluated_names)
    metrics["baseline_suite_skipped_names"] = tuple(skipped_names)
    if np.isfinite(best):
        metrics["baseline_best_mean_ic"] = best
        metrics["baseline_suite_best_name"] = best_name
        metrics["baseline_suite_best_family"] = best_family
        metrics["baseline_suite_best_mean_ic"] = best
        metrics["baseline_suite_best_long_short_ir"] = best_ir
        metrics["baseline_suite_best_factor_rank_corr"] = best_rank_corr
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
    metrics["random_baseline_observed_mean_rank_ic"] = float("nan")
    metrics["random_baseline_observed_z_score"] = float("nan")
    empty_null = pd.DataFrame(columns=["permutation", "mean_ic"])
    if not enabled:
        return empty_null

    if factor_df.empty or label_df.empty:
        return empty_null

    observed_mean, arr = compute_mean_rank_ic_permutation_null(
        factor_df,
        label_df,
        n_permutations=int(n_permutations),
        seed=int(seed),
        min_assets_per_date=3,
    )
    if not np.isfinite(observed_mean) or arr.size == 0:
        return empty_null

    metrics["random_baseline_n_permutations"] = int(arr.size)
    metrics["random_baseline_mean_ic_mean"] = float(arr.mean())
    metrics["random_baseline_mean_ic_std"] = float(arr.std(ddof=1)) if arr.size >= 2 else 0.0
    metrics["random_baseline_mean_ic_p95"] = float(np.quantile(arr, 0.95))
    metrics["random_baseline_mean_ic_p99"] = float(np.quantile(arr, 0.99))
    metrics["random_baseline_observed_mean_rank_ic"] = float(observed_mean)
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

    per_bucket = (
        qr[["date", "quantile", "mean_return"]]
        .dropna(subset=["quantile", "mean_return"])
        .copy()
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
        factor_df = factor_df[pd.to_datetime(factor_df["date"]) >= oos_start].reset_index(
            drop=True
        )
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
    coverage_stats: Mapping[str, float | int],
    warmup_summary: Mapping[str, object] | None = None,
) -> str:
    def _as_float(key: str) -> float:
        value = coverage_stats.get(key)
        if value is None:
            return float("nan")
        return float(value)

    n_dates = int(coverage_stats.get("n_dates") or 0)
    avg_assets = _as_float("avg_assets")
    avg_asset_coverage = _as_float("mean_asset_coverage")
    overall_sample_coverage = _as_float("overall_sample_coverage")
    if not np.isfinite(avg_assets) or not np.isfinite(avg_asset_coverage):
        return f"n_dates={n_dates}"
    summary = (
        f"n_dates={n_dates}; avg_assets={avg_assets:.1f}; "
        f"asset_coverage={avg_asset_coverage:.1%}; "
        f"sample_coverage={overall_sample_coverage:.1%}"
    )
    warmup_days = int((warmup_summary or {}).get("coverage_warmup_excluded_days") or 0)
    if warmup_days > 0:
        summary = f"{summary}; warmup_excluded_days={warmup_days}"
    return summary
