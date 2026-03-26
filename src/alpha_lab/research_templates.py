from __future__ import annotations

import logging
import numbers
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

from alpha_lab.alpha_pool_diagnostics import AlphaPoolDiagnostics, alpha_pool_diagnostics
from alpha_lab.alpha_registry import (
    AlphaRegistryEntry,
    alpha_entry_from_experiment,
    upsert_alpha_registry_entry,
)
from alpha_lab.capacity_diagnostics import CapacityDiagnosticsResult, run_capacity_diagnostics
from alpha_lab.composite_signals import CompositeSignalResult, compose_signals
from alpha_lab.experiment import ExperimentResult, run_factor_experiment
from alpha_lab.experiment_metadata import ExperimentMetadata
from alpha_lab.exposure_audit import ExposureAuditResult, run_exposure_audit
from alpha_lab.factor_report import FactorReport
from alpha_lab.factor_selection import FactorSelectionReport, screen_factors
from alpha_lab.handoff import HandoffExportResult, export_handoff_artifact
from alpha_lab.interfaces import validate_factor_output
from alpha_lab.labels import (
    LabelResult,
    rankpct_label,
    regression_forward_label,
    trend_scanning_labels,
    triple_barrier_labels,
)
from alpha_lab.multiple_testing import adjust_pvalues
from alpha_lab.neutralization import NeutralizationResult, neutralize_signal
from alpha_lab.purged_validation import PurgedFold, purged_fold_summary, purged_kfold_split
from alpha_lab.quantile import long_short_return, quantile_returns
from alpha_lab.research_costs import ResearchCostResult, layered_research_costs
from alpha_lab.research_universe import (
    ResearchUniverseResult,
    ResearchUniverseRules,
    construct_research_universe,
)
from alpha_lab.signal_transforms import (
    apply_min_coverage_gate,
    rank_cross_section,
    winsorize_cross_section,
    zscore_cross_section,
)
from alpha_lab.timing import DelaySpec
from alpha_lab.trial_log import append_trial_log, trial_row_from_result
from alpha_lab.walk_forward import WalkForwardResult, run_walk_forward_experiment

logger = logging.getLogger(__name__)

PromotionVerdict = Literal[
    "reject",
    "needs_review",
    "candidate_for_registry",
    "candidate_for_external_backtest",
]
LabelMethod = Literal[
    "forward_return",
    "rankpct",
    "triple_barrier",
    "trend_scanning",
]
ValidationMode = Literal["single_split", "purged_kfold", "walk_forward"]


@dataclass(frozen=True)
class PromotionDecision:
    """Machine-readable promotion recommendation."""

    verdict: PromotionVerdict
    reasons: tuple[str, ...]
    blocking_issues: tuple[str, ...]
    warnings: tuple[str, ...]
    metrics: dict[str, float]


@dataclass(frozen=True)
class SignalPreprocessSpec:
    """Cross-sectional preprocessing controls for workflow templates."""

    apply_winsorize: bool = True
    winsorize_lower: float = 0.01
    winsorize_upper: float = 0.99
    apply_zscore: bool = True
    apply_rank: bool = False
    min_group_size: int = 3
    min_coverage: float | None = None

    def __post_init__(self) -> None:
        if self.winsorize_lower < 0 or self.winsorize_upper > 1:
            raise ValueError("winsorize bounds must be within [0, 1]")
        if self.winsorize_lower >= self.winsorize_upper:
            raise ValueError("winsorize_lower must be < winsorize_upper")
        if self.min_group_size <= 0:
            raise ValueError("min_group_size must be > 0")
        if self.min_coverage is not None and (
            self.min_coverage <= 0 or self.min_coverage > 1
        ):
            raise ValueError("min_coverage must be in (0, 1] when provided")


@dataclass(frozen=True)
class NeutralizationSpec:
    """Residual neutralization configuration."""

    size_col: str | None = None
    industry_col: str | None = None
    beta_col: str | None = None
    min_obs: int = 20
    ridge: float = 1e-8

    def __post_init__(self) -> None:
        if self.size_col is None and self.industry_col is None and self.beta_col is None:
            raise ValueError(
                "at least one of size_col/industry_col/beta_col must be provided"
            )
        if self.min_obs <= 0:
            raise ValueError("min_obs must be > 0")
        if self.ridge < 0:
            raise ValueError("ridge must be >= 0")


@dataclass(frozen=True)
class SingleFactorDecisionThresholds:
    """Promotion thresholds for single-factor workflow."""

    min_rank_ic_registry: float = 0.01
    min_ic_ir_registry: float = 0.10
    min_rank_ic_external: float = 0.02
    min_ic_ir_external: float = 0.20
    min_coverage: float = 0.60
    min_tradable_ratio: float = 0.50
    max_adjusted_pvalue_external: float = 0.10


@dataclass(frozen=True)
class CompositeDecisionThresholds:
    """Promotion thresholds for composite-signal workflow."""

    min_selected_factors: int = 2
    min_rank_ic_registry: float = 0.01
    min_ic_ir_registry: float = 0.10
    min_rank_ic_external: float = 0.02
    min_ic_ir_external: float = 0.20
    min_effective_breadth_registry: float = 1.20
    min_effective_breadth_external: float = 1.60
    max_avg_abs_corr_external: float = 0.70
    max_adv_flag_ratio_external: float = 0.15
    max_mean_cost_bps_external: float = 25.0
    max_abs_industry_exposure_external: float = 0.25
    max_abs_style_exposure_external: float = 0.30

    def __post_init__(self) -> None:
        if self.min_selected_factors < 2:
            raise ValueError("min_selected_factors must be >= 2")


@dataclass(frozen=True)
class SingleFactorWorkflowSpec:
    """Configuration for the canonical single-factor workflow."""

    experiment_name: str
    factor_fn: Callable[[pd.DataFrame], pd.DataFrame]
    horizon: int = 5
    n_quantiles: int = 5
    delay_spec: DelaySpec | None = None
    universe_rules: ResearchUniverseRules | None = None
    preprocess: SignalPreprocessSpec = field(default_factory=SignalPreprocessSpec)
    neutralization: NeutralizationSpec | None = None
    label_method: LabelMethod = "forward_return"
    label_kwargs: dict[str, object] = field(default_factory=dict)
    screening_min_abs_monotonicity: float = 0.10
    screening_max_pairwise_corr: float = 0.95
    screening_max_vif: float = 20.0
    validation_mode: ValidationMode = "purged_kfold"
    train_end: str | pd.Timestamp | None = None
    test_start: str | pd.Timestamp | None = None
    purged_n_splits: int = 5
    purged_embargo_periods: int = 0
    walk_forward_train_size: int = 252
    walk_forward_test_size: int = 63
    walk_forward_step: int = 63
    walk_forward_val_size: int = 0
    metadata: ExperimentMetadata | None = None
    hypothesis: str | None = None
    research_question: str | None = None
    factor_spec: str | None = None
    dataset_id: str | None = None
    dataset_hash: str | None = None
    trial_id: str | None = None
    trial_count: int | None = None
    assumptions: tuple[str, ...] = ()
    caveats: tuple[str, ...] = ()
    decision_thresholds: SingleFactorDecisionThresholds = field(
        default_factory=SingleFactorDecisionThresholds
    )
    append_trial_log: bool = False
    trial_log_path: str | Path | None = None
    update_registry: bool = False
    registry_path: str | Path | None = None
    registry_alpha_id: str | None = None
    registry_taxonomy: str | None = None
    registry_tags: tuple[str, ...] = ()
    registry_notes: str | None = None
    export_handoff: bool = False
    handoff_output_dir: str | Path | None = None
    handoff_artifact_name: str | None = None
    handoff_include_label_snapshot: bool = False
    handoff_overwrite: bool = True

    def __post_init__(self) -> None:
        if not self.experiment_name.strip():
            raise ValueError("experiment_name must be non-empty")
        if self.horizon <= 0:
            raise ValueError("horizon must be > 0")
        if self.n_quantiles < 2:
            raise ValueError("n_quantiles must be >= 2")
        if self.validation_mode == "single_split":
            if (self.train_end is None) != (self.test_start is None):
                raise ValueError(
                    "single_split requires train_end and test_start together"
                )
        if self.validation_mode == "purged_kfold":
            if self.purged_n_splits < 2:
                raise ValueError("purged_n_splits must be >= 2")
        if self.validation_mode == "walk_forward":
            if self.walk_forward_train_size <= 0:
                raise ValueError("walk_forward_train_size must be > 0")
            if self.walk_forward_test_size <= 0:
                raise ValueError("walk_forward_test_size must be > 0")
            if self.walk_forward_step <= 0:
                raise ValueError("walk_forward_step must be > 0")
        if self.export_handoff and self.handoff_output_dir is None:
            raise ValueError("handoff_output_dir is required when export_handoff=True")


@dataclass(frozen=True)
class CompositeWorkflowSpec:
    """Configuration for the canonical multi-factor/composite workflow."""

    experiment_name: str
    horizon: int = 5
    n_quantiles: int = 5
    delay_spec: DelaySpec | None = None
    universe_rules: ResearchUniverseRules | None = None
    preprocess: SignalPreprocessSpec = field(default_factory=SignalPreprocessSpec)
    neutralization: NeutralizationSpec | None = None
    label_method: LabelMethod = "forward_return"
    label_kwargs: dict[str, object] = field(default_factory=dict)
    screening_min_coverage: float = 0.6
    screening_min_abs_monotonicity: float = 0.1
    screening_max_pairwise_corr: float = 0.9
    screening_max_vif: float = 10.0
    composite_method: Literal["equal", "ic", "icir"] = "icir"
    composite_lookback: int = 63
    composite_min_history: int = 10
    composite_factor_name: str = "composite_signal"
    portfolio_top_k: int = 20
    portfolio_bottom_k: int = 20
    portfolio_weighting_method: Literal["equal", "rank", "score"] = "equal"
    portfolio_value: float = 10_000_000.0
    adv_window: int = 20
    capacity_max_adv_participation: float = 0.05
    capacity_concentration_weight_threshold: float = 0.05
    cost_flat_fee_bps: float = 1.0
    cost_spread_bps: float = 5.0
    cost_impact_eta: float = 0.1
    metadata: ExperimentMetadata | None = None
    hypothesis: str | None = None
    research_question: str | None = None
    factor_spec: str | None = None
    dataset_id: str | None = None
    dataset_hash: str | None = None
    trial_id: str | None = None
    trial_count: int | None = None
    assumptions: tuple[str, ...] = ()
    caveats: tuple[str, ...] = ()
    decision_thresholds: CompositeDecisionThresholds = field(
        default_factory=CompositeDecisionThresholds
    )
    append_trial_log: bool = False
    trial_log_path: str | Path | None = None
    update_registry: bool = False
    registry_path: str | Path | None = None
    registry_alpha_id: str | None = None
    registry_taxonomy: str | None = None
    registry_tags: tuple[str, ...] = ()
    registry_notes: str | None = None
    export_handoff: bool = False
    handoff_output_dir: str | Path | None = None
    handoff_artifact_name: str | None = None
    handoff_include_label_snapshot: bool = False
    handoff_overwrite: bool = True

    def __post_init__(self) -> None:
        if not self.experiment_name.strip():
            raise ValueError("experiment_name must be non-empty")
        if self.horizon <= 0:
            raise ValueError("horizon must be > 0")
        if self.n_quantiles < 2:
            raise ValueError("n_quantiles must be >= 2")
        if self.portfolio_top_k <= 0:
            raise ValueError("portfolio_top_k must be > 0")
        if self.portfolio_bottom_k <= 0:
            raise ValueError("portfolio_bottom_k must be > 0")
        if self.portfolio_value <= 0:
            raise ValueError("portfolio_value must be > 0")
        if self.adv_window <= 1:
            raise ValueError("adv_window must be > 1")
        if self.export_handoff and self.handoff_output_dir is None:
            raise ValueError("handoff_output_dir is required when export_handoff=True")


@dataclass(frozen=True)
class SingleFactorWorkflowResult:
    """Outputs of the single-factor end-to-end workflow."""

    universe_result: ResearchUniverseResult
    factor_raw: pd.DataFrame
    factor_processed: pd.DataFrame
    neutralization_result: NeutralizationResult | None
    label_result: LabelResult
    screening_labels: pd.DataFrame
    experiment_result: ExperimentResult
    factor_report: FactorReport | None
    selection_report: FactorSelectionReport
    validation_summary: pd.DataFrame | None
    purged_folds: tuple[PurgedFold, ...] | None
    walk_forward_result: WalkForwardResult | None
    decision: PromotionDecision
    trial_log_row: pd.DataFrame | None
    registry_entry: AlphaRegistryEntry | None
    handoff_export: HandoffExportResult | None
    multiple_testing_pvalue: float | None
    multiple_testing_adjusted_pvalue: float | None


@dataclass(frozen=True)
class CompositeWorkflowResult:
    """Outputs of the composite-signal end-to-end workflow."""

    universe_result: ResearchUniverseResult
    candidate_signals: pd.DataFrame
    processed_signals: pd.DataFrame
    neutralization_diagnostics: pd.DataFrame
    label_result: LabelResult
    screening_labels: pd.DataFrame
    selection_report: FactorSelectionReport
    selected_signals: pd.DataFrame
    composite_signal_result: CompositeSignalResult
    composite_experiment: ExperimentResult
    alpha_return_panel: pd.DataFrame
    alpha_pool_diagnostics: AlphaPoolDiagnostics
    portfolio_weights: pd.DataFrame
    trade_plan: pd.DataFrame
    exposure_audit: ExposureAuditResult | None
    capacity_diagnostics: CapacityDiagnosticsResult | None
    cost_diagnostics: ResearchCostResult | None
    decision: PromotionDecision
    trial_log_row: pd.DataFrame | None
    registry_entry: AlphaRegistryEntry | None
    handoff_export: HandoffExportResult | None


def run_single_factor_research_workflow(
    prices: pd.DataFrame,
    *,
    spec: SingleFactorWorkflowSpec,
    asset_metadata: pd.DataFrame | None = None,
    market_state: pd.DataFrame | None = None,
    neutralization_exposures: pd.DataFrame | None = None,
) -> SingleFactorWorkflowResult:
    """Run the canonical single-factor workflow using existing building blocks."""
    _log_panel_shape(step="single_factor.prices_input", frame=prices)
    _assert_panel_cardinality(step="single_factor.prices_input", frame=prices)

    universe_result = construct_research_universe(
        prices,
        asset_metadata=asset_metadata,
        market_state=market_state,
        rules=spec.universe_rules,
    )

    factor_raw = spec.factor_fn(prices)
    validate_factor_output(factor_raw)
    _log_panel_shape(step="single_factor.factor_raw", frame=factor_raw)
    _assert_panel_cardinality(step="single_factor.factor_raw", frame=factor_raw)

    factor_processed, neutral_result = _prepare_signal(
        factor_raw,
        universe_result=universe_result,
        preprocess=spec.preprocess,
        neutralization=spec.neutralization,
        neutralization_exposures=neutralization_exposures,
    )
    _log_panel_shape(step="single_factor.factor_processed", frame=factor_processed)
    _assert_panel_cardinality(step="single_factor.factor_processed", frame=factor_processed)

    prepared_factor_fn = _build_prepared_factor_fn(
        spec.factor_fn,
        universe_result=universe_result,
        preprocess=spec.preprocess,
        neutralization=spec.neutralization,
        neutralization_exposures=neutralization_exposures,
    )

    label_result = _build_label_result(
        prices,
        method=spec.label_method,
        horizon=spec.horizon,
        label_kwargs=spec.label_kwargs,
    )
    _log_panel_shape(step="single_factor.label_result.labels", frame=label_result.labels)
    _assert_panel_cardinality(step="single_factor.label_result.labels", frame=label_result.labels)

    screening_labels = _labels_to_canonical(label_result)
    _log_panel_shape(step="single_factor.screening_labels", frame=screening_labels)
    _assert_panel_cardinality(step="single_factor.screening_labels", frame=screening_labels)

    delay_spec = spec.delay_spec or DelaySpec.for_horizon(spec.horizon)
    experiment_metadata = _resolve_metadata(
        base=spec.metadata,
        hypothesis=spec.hypothesis,
        research_question=spec.research_question,
        factor_spec=spec.factor_spec,
        dataset_id=spec.dataset_id,
        dataset_hash=spec.dataset_hash,
        trial_id=spec.trial_id,
        trial_count=spec.trial_count,
        assumptions=spec.assumptions,
        caveats=spec.caveats,
    )

    experiment_result = run_factor_experiment(
        prices,
        prepared_factor_fn,
        horizon=spec.horizon,
        n_quantiles=spec.n_quantiles,
        train_end=spec.train_end if spec.validation_mode == "single_split" else None,
        test_start=spec.test_start if spec.validation_mode == "single_split" else None,
        delay_spec=delay_spec,
        metadata=experiment_metadata,
        generate_factor_report=True,
    )

    selection_report = screen_factors(
        experiment_result.factor_df,
        screening_labels,
        n_quantiles=spec.n_quantiles,
        min_coverage=spec.decision_thresholds.min_coverage,
        min_abs_monotonicity=spec.screening_min_abs_monotonicity,
        max_pairwise_corr=spec.screening_max_pairwise_corr,
        max_vif=spec.screening_max_vif,
    )

    purged_folds: tuple[PurgedFold, ...] | None = None
    walk_forward_result: WalkForwardResult | None = None
    validation_summary: pd.DataFrame | None = None

    if spec.validation_mode == "purged_kfold":
        interval_samples = _interval_samples_from_labels(
            label_result,
            fallback_horizon=spec.horizon,
        )
        _log_panel_shape(step="single_factor.interval_samples", frame=interval_samples)
        _assert_panel_cardinality(step="single_factor.interval_samples", frame=interval_samples)
        folds = purged_kfold_split(
            interval_samples,
            n_splits=spec.purged_n_splits,
            decision_col="date",
            start_col="event_start",
            end_col="event_end",
            embargo_periods=spec.purged_embargo_periods,
        )
        purged_folds = tuple(folds)
        validation_summary = purged_fold_summary(folds)
    elif spec.validation_mode == "walk_forward":
        wf = run_walk_forward_experiment(
            prices,
            prepared_factor_fn,
            train_size=spec.walk_forward_train_size,
            test_size=spec.walk_forward_test_size,
            step=spec.walk_forward_step,
            horizon=spec.horizon,
            n_quantiles=spec.n_quantiles,
            val_size=spec.walk_forward_val_size,
            purge_periods=delay_spec.purge_periods,
            embargo_periods=delay_spec.embargo_periods,
        )
        walk_forward_result = wf
        validation_summary = wf.fold_summary_df.copy()

    pvalue_raw, pvalue_adj = _multiple_testing_pvalues(
        factor_report=experiment_result.factor_report,
        trial_count=(
            experiment_result.metadata.trial_count
            if experiment_result.metadata is not None
            else None
        ),
    )

    decision = _single_factor_decision(
        experiment=experiment_result,
        selection=selection_report,
        universe_result=universe_result,
        thresholds=spec.decision_thresholds,
        adjusted_pvalue=pvalue_adj,
        validation_summary=validation_summary,
    )
    experiment_result.metadata = _attach_decision_to_metadata(
        experiment_result.metadata,
        decision=decision,
    )

    trial_row: pd.DataFrame | None = None
    if spec.append_trial_log:
        trial_row = trial_row_from_result(
            experiment_result,
            experiment_name=spec.experiment_name,
        )
        if spec.trial_log_path is None:
            append_trial_log(trial_row)
        else:
            append_trial_log(trial_row, path=spec.trial_log_path)

    registry_entry: AlphaRegistryEntry | None = None
    if spec.update_registry:
        registry_entry = alpha_entry_from_experiment(
            experiment_result,
            alpha_id=spec.registry_alpha_id or spec.experiment_name,
            lifecycle_stage=_registry_stage_from_decision(decision.verdict),
            taxonomy=spec.registry_taxonomy,
            tags=spec.registry_tags,
            notes=spec.registry_notes,
        )
        if spec.registry_path is None:
            upsert_alpha_registry_entry(registry_entry)
        else:
            upsert_alpha_registry_entry(registry_entry, path=spec.registry_path)

    handoff_export: HandoffExportResult | None = None
    if spec.export_handoff:
        exclusion_df = (
            universe_result.exclusion_reasons
            if not universe_result.exclusion_reasons.empty
            else None
        )
        handoff_export = export_handoff_artifact(
            experiment_result,
            output_dir=Path(spec.handoff_output_dir) if spec.handoff_output_dir else Path("."),
            artifact_name=spec.handoff_artifact_name or spec.experiment_name,
            experiment_id=spec.experiment_name,
            universe_df=universe_result.universe,
            tradability_df=universe_result.tradability,
            exclusion_reasons_df=exclusion_df,
            include_label_snapshot=spec.handoff_include_label_snapshot,
            overwrite=spec.handoff_overwrite,
        )

    return SingleFactorWorkflowResult(
        universe_result=universe_result,
        factor_raw=factor_raw,
        factor_processed=factor_processed,
        neutralization_result=neutral_result,
        label_result=label_result,
        screening_labels=screening_labels,
        experiment_result=experiment_result,
        factor_report=experiment_result.factor_report,
        selection_report=selection_report,
        validation_summary=validation_summary,
        purged_folds=purged_folds,
        walk_forward_result=walk_forward_result,
        decision=decision,
        trial_log_row=trial_row,
        registry_entry=registry_entry,
        handoff_export=handoff_export,
        multiple_testing_pvalue=pvalue_raw,
        multiple_testing_adjusted_pvalue=pvalue_adj,
    )


def run_composite_signal_research_workflow(
    prices: pd.DataFrame,
    *,
    spec: CompositeWorkflowSpec,
    factor_fns: Mapping[str, Callable[[pd.DataFrame], pd.DataFrame]] | None = None,
    candidate_signals: pd.DataFrame | None = None,
    asset_metadata: pd.DataFrame | None = None,
    market_state: pd.DataFrame | None = None,
    neutralization_exposures: pd.DataFrame | None = None,
    exposure_data: pd.DataFrame | None = None,
) -> CompositeWorkflowResult:
    """Run the canonical multi-factor/composite workflow."""
    if (factor_fns is None) == (candidate_signals is None):
        raise ValueError(
            "provide exactly one of factor_fns or candidate_signals"
        )

    universe_result = construct_research_universe(
        prices,
        asset_metadata=asset_metadata,
        market_state=market_state,
        rules=spec.universe_rules,
    )

    if factor_fns is not None:
        candidate = _generate_candidate_signals(prices, factor_fns=factor_fns)
    else:
        assert candidate_signals is not None
        validate_factor_output(candidate_signals)
        candidate = candidate_signals.copy()

    processed, neutral_diag = _prepare_multi_factor_signals(
        candidate,
        universe_result=universe_result,
        preprocess=spec.preprocess,
        neutralization=spec.neutralization,
        neutralization_exposures=neutralization_exposures,
    )

    label_result = _build_label_result(
        prices,
        method=spec.label_method,
        horizon=spec.horizon,
        label_kwargs=spec.label_kwargs,
    )
    screening_labels = _labels_to_canonical(label_result)

    selection_report = screen_factors(
        processed,
        screening_labels,
        n_quantiles=spec.n_quantiles,
        min_coverage=spec.screening_min_coverage,
        min_abs_monotonicity=spec.screening_min_abs_monotonicity,
        max_pairwise_corr=spec.screening_max_pairwise_corr,
        max_vif=spec.screening_max_vif,
    )
    selected_signals = _select_signals_from_screening(
        processed,
        selection=selection_report,
    )

    composite_signal_result = compose_signals(
        selected_signals,
        method=spec.composite_method,
        labels=screening_labels if spec.composite_method in {"ic", "icir"} else None,
        lookback=spec.composite_lookback,
        min_history=spec.composite_min_history,
        output_factor=spec.composite_factor_name,
    )

    delay_spec = spec.delay_spec or DelaySpec.for_horizon(spec.horizon)
    experiment_metadata = _resolve_metadata(
        base=spec.metadata,
        hypothesis=spec.hypothesis,
        research_question=spec.research_question,
        factor_spec=spec.factor_spec,
        dataset_id=spec.dataset_id,
        dataset_hash=spec.dataset_hash,
        trial_id=spec.trial_id,
        trial_count=spec.trial_count,
        assumptions=spec.assumptions,
        caveats=spec.caveats,
    )
    composite_experiment = run_factor_experiment(
        prices,
        lambda _prices: composite_signal_result.composite.copy(),
        horizon=spec.horizon,
        n_quantiles=spec.n_quantiles,
        delay_spec=delay_spec,
        metadata=experiment_metadata,
        generate_factor_report=True,
    )

    alpha_return_panel = _alpha_return_panel(
        selected_signals,
        labels=screening_labels,
        n_quantiles=spec.n_quantiles,
    )
    pool_diag = alpha_pool_diagnostics(alpha_return_panel)

    portfolio_weights = _composite_portfolio_weights(
        composite_signal_result.composite,
        top_k=spec.portfolio_top_k,
        bottom_k=spec.portfolio_bottom_k,
        weighting_method=spec.portfolio_weighting_method,
    )
    trade_plan = _trade_plan_from_weights(
        portfolio_weights,
        prices=prices,
        portfolio_value=spec.portfolio_value,
        adv_window=spec.adv_window,
    )

    exposure_audit_result: ExposureAuditResult | None
    if exposure_data is None:
        exposure_audit_result = None
    else:
        exposure_audit_result = run_exposure_audit(
            portfolio_weights,
            exposure_data,
        )

    capacity_diag: CapacityDiagnosticsResult | None = None
    cost_diag: ResearchCostResult | None = None
    if {"adv_dollar", "daily_volatility"}.issubset(trade_plan.columns):
        diag_input = trade_plan.dropna(subset=["adv_dollar", "daily_volatility"]).copy()
        if not diag_input.empty:
            capacity_diag = run_capacity_diagnostics(
                diag_input,
                portfolio_value=spec.portfolio_value,
                max_adv_participation=spec.capacity_max_adv_participation,
                concentration_weight_threshold=spec.capacity_concentration_weight_threshold,
            )
            cost_diag = layered_research_costs(
                diag_input,
                flat_fee_bps=spec.cost_flat_fee_bps,
                spread_bps=spec.cost_spread_bps,
                impact_eta=spec.cost_impact_eta,
            )

    decision = _composite_decision(
        experiment=composite_experiment,
        selection=selection_report,
        pool_diag=pool_diag,
        capacity_diag=capacity_diag,
        exposure_diag=exposure_audit_result,
        cost_diag=cost_diag,
        thresholds=spec.decision_thresholds,
    )
    composite_experiment.metadata = _attach_decision_to_metadata(
        composite_experiment.metadata,
        decision=decision,
    )

    trial_row: pd.DataFrame | None = None
    if spec.append_trial_log:
        trial_row = trial_row_from_result(
            composite_experiment,
            experiment_name=spec.experiment_name,
        )
        if spec.trial_log_path is None:
            append_trial_log(trial_row)
        else:
            append_trial_log(trial_row, path=spec.trial_log_path)

    registry_entry: AlphaRegistryEntry | None = None
    if spec.update_registry:
        registry_entry = alpha_entry_from_experiment(
            composite_experiment,
            alpha_id=spec.registry_alpha_id or spec.experiment_name,
            lifecycle_stage=_registry_stage_from_decision(decision.verdict),
            taxonomy=spec.registry_taxonomy,
            tags=spec.registry_tags,
            notes=spec.registry_notes,
        )
        if spec.registry_path is None:
            upsert_alpha_registry_entry(registry_entry)
        else:
            upsert_alpha_registry_entry(registry_entry, path=spec.registry_path)

    handoff_export: HandoffExportResult | None = None
    if spec.export_handoff:
        exclusion_df = (
            universe_result.exclusion_reasons
            if not universe_result.exclusion_reasons.empty
            else None
        )
        handoff_export = export_handoff_artifact(
            composite_experiment,
            output_dir=Path(spec.handoff_output_dir) if spec.handoff_output_dir else Path("."),
            artifact_name=spec.handoff_artifact_name or spec.experiment_name,
            experiment_id=spec.experiment_name,
            universe_df=universe_result.universe,
            tradability_df=universe_result.tradability,
            exclusion_reasons_df=exclusion_df,
            include_label_snapshot=spec.handoff_include_label_snapshot,
            overwrite=spec.handoff_overwrite,
        )

    return CompositeWorkflowResult(
        universe_result=universe_result,
        candidate_signals=candidate,
        processed_signals=processed,
        neutralization_diagnostics=neutral_diag,
        label_result=label_result,
        screening_labels=screening_labels,
        selection_report=selection_report,
        selected_signals=selected_signals,
        composite_signal_result=composite_signal_result,
        composite_experiment=composite_experiment,
        alpha_return_panel=alpha_return_panel,
        alpha_pool_diagnostics=pool_diag,
        portfolio_weights=portfolio_weights,
        trade_plan=trade_plan,
        exposure_audit=exposure_audit_result,
        capacity_diagnostics=capacity_diag,
        cost_diagnostics=cost_diag,
        decision=decision,
        trial_log_row=trial_row,
        registry_entry=registry_entry,
        handoff_export=handoff_export,
    )


def _generate_candidate_signals(
    prices: pd.DataFrame,
    *,
    factor_fns: Mapping[str, Callable[[pd.DataFrame], pd.DataFrame]],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for _, fn in factor_fns.items():
        out = fn(prices)
        validate_factor_output(out)
        frames.append(out[["date", "asset", "factor", "value"]].copy())
    if not frames:
        raise ValueError("factor_fns produced no candidate signals")
    merged = pd.concat(frames, ignore_index=True)
    dupes = merged.duplicated(subset=["date", "asset", "factor"])
    if dupes.any():
        raise ValueError("candidate signals contain duplicate (date, asset, factor) rows")
    return merged.sort_values(["date", "asset", "factor"], kind="mergesort").reset_index(drop=True)


def _build_prepared_factor_fn(
    base_factor_fn: Callable[[pd.DataFrame], pd.DataFrame],
    *,
    universe_result: ResearchUniverseResult,
    preprocess: SignalPreprocessSpec,
    neutralization: NeutralizationSpec | None,
    neutralization_exposures: pd.DataFrame | None,
) -> Callable[[pd.DataFrame], pd.DataFrame]:
    def _prepared(prices: pd.DataFrame) -> pd.DataFrame:
        raw = base_factor_fn(prices)
        validate_factor_output(raw)
        prepared, _neutral = _prepare_signal(
            raw,
            universe_result=universe_result,
            preprocess=preprocess,
            neutralization=neutralization,
            neutralization_exposures=neutralization_exposures,
        )
        return prepared

    return _prepared


def _prepare_signal(
    factor_df: pd.DataFrame,
    *,
    universe_result: ResearchUniverseResult,
    preprocess: SignalPreprocessSpec,
    neutralization: NeutralizationSpec | None,
    neutralization_exposures: pd.DataFrame | None,
) -> tuple[pd.DataFrame, NeutralizationResult | None]:
    _log_panel_shape(step="single_factor.neutralization.input", frame=factor_df)
    out = _apply_universe_mask(
        factor_df,
        universe_df=universe_result.universe,
        tradability_df=universe_result.tradability,
    )
    _log_panel_shape(step="single_factor.neutralization.after_universe_mask", frame=out)
    out = _apply_preprocess(out, preprocess=preprocess)
    _log_panel_shape(step="single_factor.neutralization.after_preprocess", frame=out)
    if neutralization is None:
        logger.info("step=single_factor.neutralization.skip reason=no_neutralization_spec")
        return out, None
    merged = _merge_neutralization_exposures(
        out,
        neutralization_exposures=neutralization_exposures,
        spec=neutralization,
    )
    _log_panel_shape(step="single_factor.neutralization.before_fit", frame=merged)
    merged, size_col, industry_col, beta_col = _neutralization_input_columns(
        merged,
        size_col=neutralization.size_col,
        industry_col=neutralization.industry_col,
        beta_col=neutralization.beta_col,
    )
    neutral = neutralize_signal(
        merged,
        value_col="value",
        by="date",
        size_col=size_col,
        industry_col=industry_col,
        beta_col=beta_col,
        min_obs=neutralization.min_obs,
        ridge=neutralization.ridge,
        output_col="value_neutralized",
    )
    norm = neutral.data.copy()
    norm["value"] = norm["value_neutralized"]
    norm = norm.drop(columns=["value_neutralized"], errors="ignore")
    out_norm = norm[["date", "asset", "factor", "value"]]
    _log_panel_shape(step="single_factor.neutralization.after_fit", frame=out_norm)
    return out_norm, neutral


def _prepare_multi_factor_signals(
    candidate: pd.DataFrame,
    *,
    universe_result: ResearchUniverseResult,
    preprocess: SignalPreprocessSpec,
    neutralization: NeutralizationSpec | None,
    neutralization_exposures: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = _apply_universe_mask(
        candidate,
        universe_df=universe_result.universe,
        tradability_df=universe_result.tradability,
    )
    out = _apply_preprocess(out, preprocess=preprocess)
    if neutralization is None:
        return out, pd.DataFrame(columns=["factor", "exposure", "corr_reduction"])

    frames: list[pd.DataFrame] = []
    diagnostics: list[pd.DataFrame] = []
    for factor_name, group in out.groupby("factor", sort=True):
        merged = _merge_neutralization_exposures(
            group,
            neutralization_exposures=neutralization_exposures,
            spec=neutralization,
        )
        merged, size_col, industry_col, beta_col = _neutralization_input_columns(
            merged,
            size_col=neutralization.size_col,
            industry_col=neutralization.industry_col,
            beta_col=neutralization.beta_col,
        )
        neutral = neutralize_signal(
            merged,
            value_col="value",
            by="date",
            size_col=size_col,
            industry_col=industry_col,
            beta_col=beta_col,
            min_obs=neutralization.min_obs,
            ridge=neutralization.ridge,
            output_col="value_neutralized",
        )
        frame = neutral.data.copy()
        frame["value"] = frame["value_neutralized"]
        frame = frame.drop(columns=["value_neutralized"], errors="ignore")
        frames.append(frame[["date", "asset", "factor", "value"]])
        diag = neutral.diagnostics.copy()
        diag["factor"] = str(factor_name)
        diagnostics.append(diag)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["date", "asset", "factor"], kind="mergesort").reset_index(
        drop=True
    )
    diag_df = (
        pd.concat(diagnostics, ignore_index=True)
        if diagnostics
        else pd.DataFrame(columns=["factor", "exposure", "corr_reduction"])
    )
    return combined, diag_df


def _apply_universe_mask(
    factor_df: pd.DataFrame,
    *,
    universe_df: pd.DataFrame,
    tradability_df: pd.DataFrame,
) -> pd.DataFrame:
    out = factor_df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    if out["date"].isna().any():
        raise ValueError("factor_df contains invalid date values")
    universe = universe_df.copy()
    tradability = tradability_df.copy()
    universe["date"] = pd.to_datetime(universe["date"], errors="coerce")
    tradability["date"] = pd.to_datetime(tradability["date"], errors="coerce")
    merged = out.merge(
        universe[["date", "asset", "in_universe"]],
        on=["date", "asset"],
        how="left",
    ).merge(
        tradability[["date", "asset", "is_tradable"]],
        on=["date", "asset"],
        how="left",
    )
    tradable = merged["in_universe"].fillna(False) & merged["is_tradable"].fillna(False)
    merged["value"] = merged["value"].where(tradable, np.nan)
    return merged[["date", "asset", "factor", "value"]].sort_values(
        ["date", "asset", "factor"],
        kind="mergesort",
    ).reset_index(drop=True)


def _apply_preprocess(
    factor_df: pd.DataFrame,
    *,
    preprocess: SignalPreprocessSpec,
) -> pd.DataFrame:
    out = factor_df.copy()
    if preprocess.apply_winsorize:
        out = winsorize_cross_section(
            out,
            value_col="value",
            by="date",
            lower=preprocess.winsorize_lower,
            upper=preprocess.winsorize_upper,
            min_group_size=preprocess.min_group_size,
        )
    if preprocess.apply_zscore:
        out = zscore_cross_section(
            out,
            value_col="value",
            by="date",
            min_group_size=preprocess.min_group_size,
        )
    if preprocess.apply_rank:
        out = rank_cross_section(
            out,
            value_col="value",
            by="date",
            pct=True,
            min_group_size=max(2, preprocess.min_group_size),
        )
    if preprocess.min_coverage is not None:
        out = apply_min_coverage_gate(
            out,
            value_col="value",
            by="date",
            min_coverage=preprocess.min_coverage,
        )
    return out


def _merge_neutralization_exposures(
    factor_df: pd.DataFrame,
    *,
    neutralization_exposures: pd.DataFrame | None,
    spec: NeutralizationSpec,
) -> pd.DataFrame:
    if neutralization_exposures is None:
        raise ValueError("neutralization_exposures is required when neutralization is enabled")
    required = {"date", "asset"}
    needed: list[str] = [
        col
        for col in (spec.size_col, spec.industry_col, spec.beta_col)
        if col is not None
    ]
    required.update(needed)
    missing = required - set(neutralization_exposures.columns)
    if missing:
        raise ValueError(
            "neutralization_exposures missing required columns: "
            f"{sorted(missing)}"
        )
    exposure = neutralization_exposures.copy()
    exposure["date"] = pd.to_datetime(exposure["date"], errors="coerce")
    if exposure["date"].isna().any():
        raise ValueError("neutralization_exposures contains invalid date values")
    merged = factor_df.merge(
        exposure[list(required)],
        on=["date", "asset"],
        how="left",
        validate="many_to_one",
    )
    return merged


def _neutralization_input_columns(
    df: pd.DataFrame,
    *,
    size_col: str | None,
    industry_col: str | None,
    beta_col: str | None,
) -> tuple[pd.DataFrame, str | None, str | None, str | None]:
    out = df.copy()
    resolved_size = size_col
    resolved_industry = industry_col
    resolved_beta = beta_col

    if resolved_size == "size_exposure":
        out = out.rename(columns={"size_exposure": "__size_input"})
        resolved_size = "__size_input"
    if resolved_beta == "beta_exposure":
        out = out.rename(columns={"beta_exposure": "__beta_input"})
        resolved_beta = "__beta_input"
    if resolved_industry is not None and resolved_industry.startswith("industry"):
        out = out.rename(columns={resolved_industry: "__industry_input"})
        resolved_industry = "__industry_input"

    return out, resolved_size, resolved_industry, resolved_beta


def _build_label_result(
    prices: pd.DataFrame,
    *,
    method: LabelMethod,
    horizon: int,
    label_kwargs: Mapping[str, object],
) -> LabelResult:
    kwargs = dict(label_kwargs)
    logger.info(
        "step=single_factor.label_build.start method=%s base_horizon=%d n_price_rows=%d",
        method,
        int(horizon),
        int(len(prices)),
    )
    if method == "forward_return":
        resolved_h = _as_int(kwargs.pop("horizon", horizon), field_name="horizon")
        label_name = kwargs.pop("label_name", None)
        result = regression_forward_label(
            prices,
            horizon=resolved_h,
            label_name=label_name if isinstance(label_name, str) else None,
        )
        _log_panel_shape(step="single_factor.label_build.end", frame=result.labels)
        return result
    if method == "rankpct":
        resolved_h = _as_int(kwargs.pop("horizon", horizon), field_name="horizon")
        label_name = kwargs.pop("label_name", None)
        result = rankpct_label(
            prices,
            horizon=resolved_h,
            label_name=label_name if isinstance(label_name, str) else None,
        )
        _log_panel_shape(step="single_factor.label_build.end", frame=result.labels)
        return result
    if method == "triple_barrier":
        resolved_h = _as_int(kwargs.pop("horizon", horizon), field_name="horizon")
        pt_mult = _as_float(kwargs.pop("pt_mult", 1.0), field_name="pt_mult")
        sl_mult = _as_float(kwargs.pop("sl_mult", 1.0), field_name="sl_mult")
        volatility_lookback = _as_int(
            kwargs.pop("volatility_lookback", 20),
            field_name="volatility_lookback",
        )
        label_name = kwargs.pop("label_name", "triple_barrier")
        if not isinstance(label_name, str):
            raise ValueError("triple_barrier label_name must be a string")
        result = triple_barrier_labels(
            prices,
            horizon=resolved_h,
            pt_mult=pt_mult,
            sl_mult=sl_mult,
            volatility_lookback=volatility_lookback,
            label_name=label_name,
        )
        _log_panel_shape(step="single_factor.label_build.end", frame=result.labels)
        return result
    if method == "trend_scanning":
        min_horizon = _as_int(kwargs.pop("min_horizon", 2), field_name="min_horizon")
        max_horizon = _as_int(
            kwargs.pop("max_horizon", max(horizon, min_horizon)),
            field_name="max_horizon",
        )
        label_name = kwargs.pop("label_name", "trend_scan")
        if not isinstance(label_name, str):
            raise ValueError("trend_scanning label_name must be a string")
        result = trend_scanning_labels(
            prices,
            min_horizon=min_horizon,
            max_horizon=max_horizon,
            label_name=label_name,
        )
        _log_panel_shape(step="single_factor.label_build.end", frame=result.labels)
        return result
    raise ValueError(f"unsupported label method {method!r}")


def _labels_to_canonical(label_result: LabelResult) -> pd.DataFrame:
    labels = label_result.labels.copy()
    labels["date"] = pd.to_datetime(labels["date"], errors="coerce")
    if labels["date"].isna().any():
        raise ValueError("label_result contains invalid date values")
    out = labels[["date", "asset", "label_name", "label_value"]].rename(
        columns={"label_name": "factor", "label_value": "value"}
    )
    out = out.sort_values(["date", "asset", "factor"], kind="mergesort").reset_index(drop=True)
    return out


def _interval_samples_from_labels(
    label_result: LabelResult,
    *,
    fallback_horizon: int,
) -> pd.DataFrame:
    labels = label_result.labels.copy()
    labels["date"] = pd.to_datetime(labels["date"], errors="coerce")
    start_col = labels["event_start"] if "event_start" in labels.columns else labels["date"]
    end_col = (
        labels["event_end"]
        if "event_end" in labels.columns
        else pd.Series(pd.NaT, index=labels.index, dtype="datetime64[ns]")
    )
    realized_col = (
        labels["realized_horizon"]
        if "realized_horizon" in labels.columns
        else pd.Series(np.nan, index=labels.index, dtype=float)
    )
    start = pd.to_datetime(start_col, errors="coerce")
    end = pd.to_datetime(end_col, errors="coerce")
    realized = pd.to_numeric(realized_col, errors="coerce")

    start = start.fillna(labels["date"])
    fallback_days = realized.fillna(float(fallback_horizon)).clip(lower=0.0)
    end = end.fillna(labels["date"] + pd.to_timedelta(fallback_days, unit="D"))
    end = end.where(end >= start, start)
    out = pd.DataFrame(
        {
            "date": labels["date"],
            "asset": labels["asset"],
            "event_start": start,
            "event_end": end,
        }
    )
    return out.sort_values(["date", "asset"], kind="mergesort").reset_index(drop=True)


def _resolve_metadata(
    *,
    base: ExperimentMetadata | None,
    hypothesis: str | None,
    research_question: str | None,
    factor_spec: str | None,
    dataset_id: str | None,
    dataset_hash: str | None,
    trial_id: str | None,
    trial_count: int | None,
    assumptions: tuple[str, ...],
    caveats: tuple[str, ...],
) -> ExperimentMetadata:
    if base is None:
        return ExperimentMetadata(
            hypothesis=hypothesis,
            research_question=research_question,
            factor_spec=factor_spec,
            dataset_id=dataset_id,
            dataset_hash=dataset_hash,
            trial_id=trial_id,
            trial_count=trial_count,
            assumptions=assumptions,
            caveats=caveats,
        )
    return replace(
        base,
        hypothesis=hypothesis or base.hypothesis,
        research_question=research_question or base.research_question,
        factor_spec=factor_spec or base.factor_spec,
        dataset_id=dataset_id or base.dataset_id,
        dataset_hash=dataset_hash or base.dataset_hash,
        trial_id=trial_id or base.trial_id,
        trial_count=trial_count if trial_count is not None else base.trial_count,
        assumptions=assumptions or base.assumptions,
        caveats=caveats or base.caveats,
    )


def _multiple_testing_pvalues(
    *,
    factor_report: FactorReport | None,
    trial_count: int | None,
) -> tuple[float | None, float | None]:
    if factor_report is None:
        return None, None
    row = factor_report.ic_summary_df[
        factor_report.ic_summary_df["metric"].astype(str) == "rank_ic"
    ]
    if row.empty:
        return None, None
    t_stat = pd.to_numeric(row["t_stat"].iloc[0], errors="coerce")
    n_obs = pd.to_numeric(row["n_obs"].iloc[0], errors="coerce")
    if not np.isfinite(t_stat) or not np.isfinite(n_obs):
        return None, None
    if float(n_obs) <= 1:
        return None, None
    p_raw = float(2.0 * stats.t.sf(abs(float(t_stat)), df=float(n_obs) - 1.0))
    n_tests = float(trial_count) if trial_count is not None else 1.0
    mt = adjust_pvalues([p_raw], n_tests=n_tests, method="bonferroni")
    return p_raw, float(mt.adjusted_pvalues[0])


def _single_factor_decision(
    *,
    experiment: ExperimentResult,
    selection: FactorSelectionReport,
    universe_result: ResearchUniverseResult,
    thresholds: SingleFactorDecisionThresholds,
    adjusted_pvalue: float | None,
    validation_summary: pd.DataFrame | None,
) -> PromotionDecision:
    summary = experiment.summary
    decisions = selection.decisions
    factor_name = (
        str(experiment.factor_df["factor"].iloc[0]) if not experiment.factor_df.empty else "unknown"
    )
    sel_row = decisions[decisions["factor"].astype(str) == factor_name]
    sel_decision = (
        str(sel_row["decision"].iloc[0])
        if not sel_row.empty
        else "needs_review"
    )

    coverage = float("nan")
    if experiment.factor_report is not None and not experiment.factor_report.coverage_df.empty:
        coverage = float(
            pd.to_numeric(
                experiment.factor_report.coverage_df["coverage_overlap_vs_factor"],
                errors="coerce",
            ).mean()
        )

    tradable_ratio = float("nan")
    if not universe_result.diagnostics.empty:
        tradable_ratio = float(
            pd.to_numeric(universe_result.diagnostics["tradable_ratio"], errors="coerce").mean()
        )

    blockers: list[str] = []
    warnings: list[str] = []
    reasons: list[str] = []

    if (
        not np.isfinite(summary.mean_rank_ic)
        or summary.mean_rank_ic < thresholds.min_rank_ic_registry
    ):
        blockers.append("mean_rank_ic_below_registry_threshold")
    if not np.isfinite(summary.ic_ir) or summary.ic_ir < thresholds.min_ic_ir_registry:
        blockers.append("ic_ir_below_registry_threshold")
    if not np.isfinite(coverage) or coverage < thresholds.min_coverage:
        blockers.append("coverage_below_threshold")
    if not np.isfinite(tradable_ratio) or tradable_ratio < thresholds.min_tradable_ratio:
        blockers.append("tradable_ratio_below_threshold")
    if sel_decision != "candidate_factor":
        blockers.append("factor_screening_not_candidate")

    if validation_summary is not None and not validation_summary.empty:
        if "n_train" in validation_summary.columns:
            if (pd.to_numeric(validation_summary["n_train"], errors="coerce") <= 0).any():
                blockers.append("validation_fold_without_train_samples")
        if "n_test" in validation_summary.columns:
            if (pd.to_numeric(validation_summary["n_test"], errors="coerce") <= 0).any():
                blockers.append("validation_fold_without_test_samples")

    if adjusted_pvalue is not None and adjusted_pvalue > thresholds.max_adjusted_pvalue_external:
        warnings.append("multiple_testing_adjusted_pvalue_above_external_threshold")

    metrics = {
        "mean_rank_ic": float(summary.mean_rank_ic),
        "ic_ir": float(summary.ic_ir),
        "mean_coverage": float(coverage),
        "mean_tradable_ratio": float(tradable_ratio),
        "adjusted_pvalue": float(adjusted_pvalue) if adjusted_pvalue is not None else np.nan,
    }

    if blockers:
        reasons.append("one_or_more_hard_gates_failed")
        return PromotionDecision(
            verdict="reject",
            reasons=tuple(reasons),
            blocking_issues=tuple(sorted(set(blockers))),
            warnings=tuple(sorted(set(warnings))),
            metrics=metrics,
        )

    external_ready = (
        summary.mean_rank_ic >= thresholds.min_rank_ic_external
        and summary.ic_ir >= thresholds.min_ic_ir_external
        and (
            adjusted_pvalue is None
            or adjusted_pvalue <= thresholds.max_adjusted_pvalue_external
        )
    )
    if external_ready and not warnings:
        reasons.append("all_external_gates_passed")
        verdict: PromotionVerdict = "candidate_for_external_backtest"
    elif warnings:
        reasons.append("soft_gates_require_manual_review")
        verdict = "needs_review"
    else:
        reasons.append("registry_gates_passed")
        verdict = "candidate_for_registry"

    return PromotionDecision(
        verdict=verdict,
        reasons=tuple(reasons),
        blocking_issues=tuple(sorted(set(blockers))),
        warnings=tuple(sorted(set(warnings))),
        metrics=metrics,
    )


def _select_signals_from_screening(
    candidate: pd.DataFrame,
    *,
    selection: FactorSelectionReport,
) -> pd.DataFrame:
    decisions = selection.decisions.copy()
    chosen = decisions[decisions["decision"].astype(str) == "candidate_factor"]
    if len(chosen) >= 2:
        factors = set(chosen["factor"].astype(str))
        out = candidate[candidate["factor"].astype(str).isin(factors)].copy()
        return out.sort_values(["date", "asset", "factor"], kind="mergesort").reset_index(drop=True)

    ranked = selection.summary.sort_values(
        ["rank_ic_mean", "coverage"],
        ascending=[False, False],
        kind="mergesort",
    )
    top = ranked["factor"].astype(str).head(2).tolist()
    out = candidate[candidate["factor"].astype(str).isin(top)].copy()
    if out["factor"].nunique() < 2:
        raise ValueError("composite workflow requires at least 2 factors after screening")
    return out.sort_values(["date", "asset", "factor"], kind="mergesort").reset_index(drop=True)


def _alpha_return_panel(
    signals: pd.DataFrame,
    *,
    labels: pd.DataFrame,
    n_quantiles: int,
) -> pd.DataFrame:
    series_list: list[pd.Series] = []
    for factor_name, group in signals.groupby("factor", sort=True):
        qret = quantile_returns(group, labels, n_quantiles=n_quantiles)
        ls = long_short_return(qret)
        if ls.empty:
            continue
        s = ls.set_index("date")["long_short_return"].rename(str(factor_name))
        series_list.append(s)
    if not series_list:
        raise ValueError("unable to build alpha return panel from candidate signals")
    panel = pd.concat(series_list, axis=1, sort=False).sort_index()
    panel = panel.dropna(axis=0, how="all").fillna(0.0)
    return panel


def _composite_portfolio_weights(
    composite_signal: pd.DataFrame,
    *,
    top_k: int,
    bottom_k: int,
    weighting_method: Literal["equal", "rank", "score"],
) -> pd.DataFrame:
    from alpha_lab.portfolio_research import portfolio_weights

    return portfolio_weights(
        composite_signal,
        method=weighting_method,
        top_k=top_k,
        bottom_k=bottom_k,
    )


def _trade_plan_from_weights(
    weights: pd.DataFrame,
    *,
    prices: pd.DataFrame,
    portfolio_value: float,
    adv_window: int,
) -> pd.DataFrame:
    if weights.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "asset",
                "trade_weight",
                "trade_dollar",
                "target_weight",
                "adv_dollar",
                "daily_volatility",
            ]
        )

    w = weights.copy()
    w["date"] = pd.to_datetime(w["date"], errors="coerce")
    pivot = (
        w.pivot(index="date", columns="asset", values="weight")
        .sort_index()
        .fillna(0.0)
    )
    diff = pivot.diff().fillna(pivot)
    long_trade = diff.stack().rename("trade_weight").reset_index()
    long_trade["trade_dollar"] = long_trade["trade_weight"] * float(portfolio_value)
    target = pivot.stack().rename("target_weight").reset_index()
    trade = long_trade.merge(target, on=["date", "asset"], how="left")

    price = prices.copy()
    price["date"] = pd.to_datetime(price["date"], errors="coerce")
    price = price.sort_values(["asset", "date"], kind="mergesort").reset_index(drop=True)
    price["ret"] = price.groupby("asset", sort=False)["close"].pct_change(fill_method=None)
    if "daily_volatility" not in price.columns:
        price["daily_volatility"] = (
            price.groupby("asset", sort=False)["ret"]
            .rolling(adv_window, min_periods=2)
            .std(ddof=0)
            .reset_index(level=0, drop=True)
        )
    if "adv_dollar" not in price.columns:
        if "dollar_volume" in price.columns:
            price["adv_dollar"] = pd.to_numeric(price["dollar_volume"], errors="coerce")
        elif "volume" in price.columns:
            px = pd.to_numeric(price["close"], errors="coerce")
            vol = pd.to_numeric(price["volume"], errors="coerce")
            price["dollar_volume"] = px * vol
            price["adv_dollar"] = (
                price.groupby("asset", sort=False)["dollar_volume"]
                .rolling(adv_window, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )
        else:
            price["adv_dollar"] = np.nan

    merged = trade.merge(
        price[["date", "asset", "adv_dollar", "daily_volatility"]],
        on=["date", "asset"],
        how="left",
    )
    return merged.sort_values(["date", "asset"], kind="mergesort").reset_index(drop=True)


def _composite_decision(
    *,
    experiment: ExperimentResult,
    selection: FactorSelectionReport,
    pool_diag: AlphaPoolDiagnostics,
    capacity_diag: CapacityDiagnosticsResult | None,
    exposure_diag: ExposureAuditResult | None,
    cost_diag: ResearchCostResult | None,
    thresholds: CompositeDecisionThresholds,
) -> PromotionDecision:
    summary = experiment.summary
    n_selected = int(
        (selection.decisions["decision"].astype(str) == "candidate_factor").sum()
    )
    breadth = float(pool_diag.breadth_summary["effective_breadth"].iloc[0])
    avg_abs_corr = float(pool_diag.breadth_summary["avg_abs_corr"].iloc[0])

    adv_flag_ratio = np.nan
    if capacity_diag is not None and not capacity_diag.warnings.empty:
        adv_flag_ratio = float(capacity_diag.warnings["adv_limit_flag_ratio"].iloc[0])

    mean_cost_bps = np.nan
    if cost_diag is not None and not cost_diag.summary.empty:
        mean_cost_bps = float(cost_diag.summary["mean_cost_bps"].iloc[0])

    max_abs_industry = np.nan
    max_abs_style = np.nan
    if exposure_diag is not None and not exposure_diag.summary.empty:
        max_abs_industry = float(exposure_diag.summary["max_abs_industry_exposure"].iloc[0])
        max_abs_style = float(exposure_diag.summary["max_abs_style_exposure"].iloc[0])

    blockers: list[str] = []
    warnings: list[str] = []
    reasons: list[str] = []

    if n_selected < thresholds.min_selected_factors:
        blockers.append("insufficient_selected_factors")
    if (
        not np.isfinite(summary.mean_rank_ic)
        or summary.mean_rank_ic < thresholds.min_rank_ic_registry
    ):
        blockers.append("composite_rank_ic_below_registry_threshold")
    if not np.isfinite(summary.ic_ir) or summary.ic_ir < thresholds.min_ic_ir_registry:
        blockers.append("composite_ic_ir_below_registry_threshold")
    if not np.isfinite(breadth) or breadth < thresholds.min_effective_breadth_registry:
        blockers.append("effective_breadth_below_registry_threshold")

    if capacity_diag is None:
        warnings.append("capacity_diagnostics_missing")
    if cost_diag is None:
        warnings.append("cost_diagnostics_missing")
    if exposure_diag is None:
        warnings.append("exposure_diagnostics_missing")

    metrics = {
        "mean_rank_ic": float(summary.mean_rank_ic),
        "ic_ir": float(summary.ic_ir),
        "n_selected_factors": float(n_selected),
        "effective_breadth": float(breadth),
        "avg_abs_corr": float(avg_abs_corr),
        "adv_flag_ratio": float(adv_flag_ratio) if np.isfinite(adv_flag_ratio) else np.nan,
        "mean_cost_bps": float(mean_cost_bps) if np.isfinite(mean_cost_bps) else np.nan,
        "max_abs_industry_exposure": (
            float(max_abs_industry) if np.isfinite(max_abs_industry) else np.nan
        ),
        "max_abs_style_exposure": float(max_abs_style) if np.isfinite(max_abs_style) else np.nan,
    }

    if blockers:
        reasons.append("one_or_more_hard_gates_failed")
        return PromotionDecision(
            verdict="reject",
            reasons=tuple(reasons),
            blocking_issues=tuple(sorted(set(blockers))),
            warnings=tuple(sorted(set(warnings))),
            metrics=metrics,
        )

    external_ready = (
        summary.mean_rank_ic >= thresholds.min_rank_ic_external
        and summary.ic_ir >= thresholds.min_ic_ir_external
        and breadth >= thresholds.min_effective_breadth_external
        and avg_abs_corr <= thresholds.max_avg_abs_corr_external
        and (
            not np.isfinite(adv_flag_ratio)
            or adv_flag_ratio <= thresholds.max_adv_flag_ratio_external
        )
        and (
            not np.isfinite(mean_cost_bps)
            or mean_cost_bps <= thresholds.max_mean_cost_bps_external
        )
        and (
            not np.isfinite(max_abs_industry)
            or max_abs_industry <= thresholds.max_abs_industry_exposure_external
        )
        and (
            not np.isfinite(max_abs_style)
            or max_abs_style <= thresholds.max_abs_style_exposure_external
        )
    )

    if external_ready and not warnings:
        reasons.append("all_external_gates_passed")
        verdict: PromotionVerdict = "candidate_for_external_backtest"
    elif warnings:
        reasons.append("soft_gates_require_manual_review")
        verdict = "needs_review"
    else:
        reasons.append("registry_gates_passed")
        verdict = "candidate_for_registry"

    return PromotionDecision(
        verdict=verdict,
        reasons=tuple(reasons),
        blocking_issues=tuple(sorted(set(blockers))),
        warnings=tuple(sorted(set(warnings))),
        metrics=metrics,
    )


def _attach_decision_to_metadata(
    metadata: ExperimentMetadata | None,
    *,
    decision: PromotionDecision,
) -> ExperimentMetadata | None:
    if metadata is None:
        return None
    warning_tuple = tuple(
        list(metadata.warnings) + list(decision.warnings) + list(decision.blocking_issues)
    )
    return replace(
        metadata,
        verdict=decision.verdict,
        warnings=warning_tuple,
    )


def _registry_stage_from_decision(verdict: PromotionVerdict) -> str:
    if verdict == "candidate_for_external_backtest":
        return "approved_for_external_backtest"
    if verdict == "candidate_for_registry":
        return "candidate"
    if verdict == "needs_review":
        return "discovery"
    return "discovery"


def _as_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must not be boolean")
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        if not np.isfinite(float(value)):
            raise ValueError(f"{field_name} must be finite")
        return int(float(value))
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError as exc:
            raise ValueError(f"{field_name} must be an integer-like value") from exc
    raise ValueError(f"{field_name} must be an integer-like value")


def _as_float(value: object, *, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must not be boolean")
    if isinstance(value, numbers.Real):
        out = float(value)
        if not np.isfinite(out):
            raise ValueError(f"{field_name} must be finite")
        return out
    if isinstance(value, str):
        try:
            out = float(value)
        except ValueError as exc:
            raise ValueError(f"{field_name} must be a float-like value") from exc
        if not np.isfinite(out):
            raise ValueError(f"{field_name} must be finite")
        return out
    raise ValueError(f"{field_name} must be a float-like value")


def _log_panel_shape(*, step: str, frame: pd.DataFrame) -> None:
    n_rows = int(len(frame))
    n_dates = int(frame["date"].nunique()) if "date" in frame.columns else -1
    n_assets = int(frame["asset"].nunique()) if "asset" in frame.columns else -1
    logger.info(
        "step=%s n_rows=%d n_dates=%s n_assets=%s n_cols=%d",
        step,
        n_rows,
        "NA" if n_dates < 0 else str(n_dates),
        "NA" if n_assets < 0 else str(n_assets),
        int(frame.shape[1]),
    )


def _assert_panel_cardinality(*, step: str, frame: pd.DataFrame) -> None:
    if "date" in frame.columns:
        n_dates = int(frame["date"].nunique())
        if n_dates > 5_000:
            raise ValueError(
                f"{step}: unique dates={n_dates} exceeds safety limit 5000."
            )
    if "asset" in frame.columns:
        n_assets = int(frame["asset"].nunique())
        if n_assets > 10_000:
            raise ValueError(
                f"{step}: unique assets={n_assets} exceeds safety limit 10000."
            )
