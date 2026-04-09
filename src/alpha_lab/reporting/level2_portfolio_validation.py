"""Standardized Level 2 portfolio-validation workflow for promoted cases."""

from __future__ import annotations

import datetime
import json
import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict, cast

import numpy as np
import pandas as pd

from alpha_lab.artifact_contracts import (
    validate_portfolio_validation_metrics_payload,
    validate_portfolio_validation_package_payload,
    validate_portfolio_validation_summary_payload,
)
from alpha_lab.experiment import ExperimentResult
from alpha_lab.key_metrics_contracts import project_portfolio_validation_metrics
from alpha_lab.portfolio_research import (
    portfolio_cost_adjusted_returns,
    portfolio_turnover,
    portfolio_weights,
    simulate_portfolio_returns,
)
from alpha_lab.research_evaluation_config import (
    DEFAULT_RESEARCH_EVALUATION_CONFIG,
    Level2PortfolioValidationConfig,
)

LEVEL2_PORTFOLIO_VALIDATION_SCHEMA_VERSION = "1.0.0"
LEVEL2_PORTFOLIO_VALIDATION_PACKAGE_TYPE = "alpha_lab_level2_portfolio_validation_package"
LEVEL2_PORTFOLIO_VALIDATION_RECOMMENDATION_TAXONOMY: tuple[str, ...] = (
    "Credible at portfolio level",
    "Needs portfolio refinement",
    "Not evaluated (not promoted)",
)
LEVEL2_PORTFOLIO_ROBUSTNESS_TAXONOMY: tuple[str, ...] = (
    "Robust at portfolio level",
    "Credible but sensitive",
    "Fragile at portfolio level",
    "Inconclusive portfolio evidence",
)

DEFAULT_LEVEL2_PORTFOLIO_VALIDATION_CONFIG = (
    DEFAULT_RESEARCH_EVALUATION_CONFIG.level2_portfolio_validation
)


class ScenarioMetricsRow(TypedDict):
    weighting_method: str
    holding_period: int
    rebalance_step: int
    leg_size_k: int
    n_return_dates: int
    mean_portfolio_return: float | None
    portfolio_ir: float | None
    portfolio_hit_rate: float | None
    mean_turnover: float | None
    max_abs_weight_mean: float | None
    top5_abs_weight_share_mean: float | None
    effective_names_mean: float | None
    gross_exposure_mean: float | None
    net_exposure_mean: float | None
    mean_cost_adjusted_return_by_cost_rate: dict[str, float | None]


class HoldingSensitivityRow(TypedDict):
    holding_period: int | None
    mean_portfolio_return: float | None
    mean_turnover: float | None
    mean_cost_adjusted_return_review_rate: float | None


class WeightingSensitivityRow(TypedDict):
    weighting_method: str
    mean_portfolio_return: float | None
    mean_turnover: float | None
    mean_cost_adjusted_return_review_rate: float | None


class BenchmarkRelativeEvaluationPayload(TypedDict, total=False):
    status: str
    note: str
    assessment: str
    interpretation: str
    benchmark_name: str | None
    benchmark_excess_return: float | None
    benchmark_active_return: float | None
    benchmark_relative_return: float | None
    benchmark_long_short_excess_return: float | None
    benchmark_information_ratio: float | None
    benchmark_tracking_error: float | None
    benchmark_relative_max_drawdown: float | None
    benchmark_return_estimate_from_excess: float | None
    risk_flags: list[str]


class SummaryBenchmarkSlice(TypedDict):
    benchmark_relative_status: str | None
    benchmark_relative_assessment: str | None
    benchmark_relative_excess_return: float | None
    benchmark_relative_tracking_error: float | None
    benchmark_relative_risks: list[str]


class ConcentrationExposureDiagnosticsPayload(TypedDict, total=False):
    max_abs_weight_mean: float | None
    top5_abs_weight_share_mean: float | None
    effective_names_mean: float | None
    gross_exposure_mean: float | None
    net_exposure_mean: float | None


class PortfolioRobustnessSummaryPayload(TypedDict):
    taxonomy_label: str
    support_reasons: list[str]
    fragility_reasons: list[str]
    scenario_sensitivity_notes: list[str]
    benchmark_relative_support_note: str
    cost_sensitivity_note: str
    concentration_turnover_risk_note: str


class Level2PortfolioValidationSummaryPayload(TypedDict, total=False):
    validation_status: str
    promotion_decision: str
    recommendation: str
    remains_credible_at_portfolio_level: bool | None
    base_weighting_method: str | None
    base_holding_period: int | None
    rebalance_step_assumption: int
    base_mean_portfolio_return: float | None
    base_mean_turnover: float | None
    base_cost_adjusted_return_review_rate: float | None
    major_risks: list[str]
    major_caveats: list[str]
    benchmark_relative_status: str | None
    benchmark_relative_assessment: str | None
    benchmark_relative_excess_return: float | None
    benchmark_relative_tracking_error: float | None
    benchmark_relative_risks: list[str]
    portfolio_robustness_summary: PortfolioRobustnessSummaryPayload


class Level2PortfolioValidationMetricsPayload(TypedDict, total=False):
    protocol_settings: dict[str, object]
    scenario_metrics: list[ScenarioMetricsRow]
    holding_period_sensitivity: list[HoldingSensitivityRow]
    weighting_sensitivity: list[WeightingSensitivityRow]
    turnover_summary: Mapping[str, object]
    transaction_cost_sensitivity: Mapping[str, object]
    concentration_exposure_diagnostics: ConcentrationExposureDiagnosticsPayload
    benchmark_relative_evaluation: BenchmarkRelativeEvaluationPayload


class Level2PortfolioValidationPackagePayload(TypedDict, total=False):
    schema_version: str
    package_type: str
    created_at_utc: str
    input_case_identity: Mapping[str, object]
    promotion_decision_context: Mapping[str, object]
    portfolio_validation_settings: Mapping[str, object]
    key_portfolio_results: Mapping[str, object]
    major_risks: list[str]
    major_caveats: list[str]
    portfolio_robustness_summary: PortfolioRobustnessSummaryPayload
    recommendation: dict[str, object]


class Level2PortfolioValidationBundlePayload(TypedDict):
    portfolio_validation_summary: Level2PortfolioValidationSummaryPayload
    portfolio_validation_metrics: Level2PortfolioValidationMetricsPayload
    portfolio_validation_package: Level2PortfolioValidationPackagePayload


@dataclass(frozen=True)
class Level2PortfolioValidationBundle:
    """Structured Level 2 portfolio-validation payload."""

    summary: Level2PortfolioValidationSummaryPayload
    metrics: Level2PortfolioValidationMetricsPayload
    package: Level2PortfolioValidationPackagePayload

    def to_dict(self) -> Level2PortfolioValidationBundlePayload:
        return {
            "portfolio_validation_summary": self.summary,
            "portfolio_validation_metrics": self.metrics,
            "portfolio_validation_package": self.package,
        }


def build_level2_portfolio_validation_bundle(
    *,
    key_metrics: Mapping[str, object],
    case_context: Mapping[str, object] | None = None,
    promotion_decision: Mapping[str, object] | None = None,
    experiment_result: ExperimentResult | None = None,
    config: Level2PortfolioValidationConfig = DEFAULT_LEVEL2_PORTFOLIO_VALIDATION_CONFIG,
) -> Level2PortfolioValidationBundle:
    """Build a compact, auditable Level 2 portfolio-validation bundle."""

    context = dict(case_context) if isinstance(case_context, Mapping) else {}
    promotion = dict(promotion_decision) if isinstance(promotion_decision, Mapping) else {}
    portfolio_metrics_contract = project_portfolio_validation_metrics(key_metrics)

    promotion_label = (
        _safe_str(promotion.get("verdict"))
        or portfolio_metrics_contract["promotion_decision"]
        or "N/A"
    )
    promotion_reasons = _to_text_list(promotion.get("reasons")) or _to_text_list(
        portfolio_metrics_contract["promotion_reasons"]
    )
    promotion_blockers = _to_text_list(promotion.get("blockers")) or _to_text_list(
        portfolio_metrics_contract["promotion_blockers"]
    )
    promoted = promotion_label == "Promote to Level 2"

    methods = _normalize_methods(config.weighting_methods, default=config.default_weighting_method)
    holding_grid = _normalize_holding_grid(
        config.holding_period_grid,
        default=config.default_holding_period,
    )
    cost_grid = _normalize_cost_grid(
        config.transaction_cost_grid,
        review_cost=config.review_cost_rate,
    )
    rebalance_input = (
        _safe_str(context.get("rebalance_frequency"))
        or portfolio_metrics_contract["rebalance_frequency"]
        or "unspecified"
    )
    rebalance_step, rebalance_source = _resolve_rebalance_step(rebalance_input)

    protocol_settings = {
        "portfolio_construction_assumptions": {
            "long_leg": "top-k cross-sectional signal bucket",
            "short_leg": "bottom-k cross-sectional signal bucket",
            "leg_size_policy": "quantile-implied k from assignment density",
            "net_exposure_target": 0.0,
            "gross_exposure_target": 2.0,
        },
        "weighting_scheme": {
            "default": config.default_weighting_method,
            "sensitivity_methods": list(methods),
        },
        "rebalance_assumption": {
            "input_frequency": rebalance_input,
            "rebalance_step": rebalance_step,
            "source": rebalance_source,
        },
        "holding_period_sensitivity": list(holding_grid),
        "transaction_cost_sensitivity": list(cost_grid),
        "benchmark_relative_policy": (
            "benchmark-relative evaluation is included only when benchmark metrics "
            "are already present in case evidence"
        ),
    }

    case_identity = {
        "case_name": _safe_str(context.get("case_name")) or portfolio_metrics_contract["case_name"],
        "case_id": _safe_str(context.get("case_id")),
        "case_output_dir": _safe_str(context.get("case_output_dir")),
        "package_type": _safe_str(context.get("package_type")),
        "experiment_name": _safe_str(context.get("experiment_name")),
    }
    promotion_context = {
        "decision": promotion_label,
        "reasons": promotion_reasons,
        "blockers": promotion_blockers,
        "source": _safe_str(promotion.get("source")) or "level2_promotion_gate",
    }
    summary: Level2PortfolioValidationSummaryPayload
    metrics: Level2PortfolioValidationMetricsPayload
    package: Level2PortfolioValidationPackagePayload

    if not promoted and not config.run_for_non_promoted_cases:
        warnings.warn(
            "Level 2 portfolio validation suppressed: factor is not promoted and "
            "run_for_non_promoted_cases=False. Set run_for_non_promoted_cases=True "
            "to run portfolio validation for non-promoted cases.",
            stacklevel=2,
        )
        benchmark_eval = _benchmark_relative_summary(key_metrics, config=config)
        risks = list(promotion_blockers) if promotion_blockers else ["not promoted to Level 2"]
        caveats = ["Portfolio validation skipped because promotion gate was not passed."]
        robustness_summary = _inconclusive_portfolio_robustness_summary(
            reason="Portfolio validation was skipped because promotion gate was not passed.",
            fragility_reasons=risks,
            benchmark_eval=benchmark_eval,
        )
        summary = {
            "validation_status": "skipped_not_promoted",
            "promotion_decision": promotion_label,
            "recommendation": LEVEL2_PORTFOLIO_VALIDATION_RECOMMENDATION_TAXONOMY[2],
            "remains_credible_at_portfolio_level": None,
            "major_risks": risks,
            "major_caveats": caveats,
            "portfolio_robustness_summary": robustness_summary,
            **_summary_benchmark_slice(benchmark_eval),
        }
        metrics = {
            "protocol_settings": protocol_settings,
            "scenario_metrics": [],
            "holding_period_sensitivity": [],
            "weighting_sensitivity": [],
            "turnover_summary": {},
            "transaction_cost_sensitivity": {},
            "concentration_exposure_diagnostics": {},
            "benchmark_relative_evaluation": benchmark_eval,
        }
        package = {
            "schema_version": LEVEL2_PORTFOLIO_VALIDATION_SCHEMA_VERSION,
            "package_type": LEVEL2_PORTFOLIO_VALIDATION_PACKAGE_TYPE,
            "created_at_utc": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
            "input_case_identity": case_identity,
            "promotion_decision_context": promotion_context,
            "portfolio_validation_settings": protocol_settings,
            "key_portfolio_results": {},
            "major_risks": risks,
            "major_caveats": caveats,
            "portfolio_robustness_summary": robustness_summary,
            "recommendation": {
                "label": LEVEL2_PORTFOLIO_VALIDATION_RECOMMENDATION_TAXONOMY[2],
                "remains_credible_at_portfolio_level": None,
                "rationale": ["promotion gate not passed"],
            },
        }
        return Level2PortfolioValidationBundle(
            summary=cast(Level2PortfolioValidationSummaryPayload, _jsonable(summary)),
            metrics=cast(Level2PortfolioValidationMetricsPayload, _jsonable(metrics)),
            package=cast(Level2PortfolioValidationPackagePayload, _jsonable(package)),
        )

    if experiment_result is None:
        benchmark_eval = _benchmark_relative_summary(key_metrics, config=config)
        risks = ["portfolio-validation inputs are unavailable"]
        caveats = [
            "No in-memory experiment outputs were provided to compute standardized "
            "portfolio sensitivity checks."
        ]
        recommendation = LEVEL2_PORTFOLIO_VALIDATION_RECOMMENDATION_TAXONOMY[1]
        robustness_summary = _inconclusive_portfolio_robustness_summary(
            reason=(
                "Portfolio scenario evidence is unavailable because experiment outputs "
                "were not provided."
            ),
            fragility_reasons=risks,
            benchmark_eval=benchmark_eval,
        )
        summary = {
            "validation_status": "blocked_missing_inputs",
            "promotion_decision": promotion_label,
            "recommendation": recommendation,
            "remains_credible_at_portfolio_level": False,
            "major_risks": risks,
            "major_caveats": caveats,
            "portfolio_robustness_summary": robustness_summary,
            **_summary_benchmark_slice(benchmark_eval),
        }
        metrics = {
            "protocol_settings": protocol_settings,
            "scenario_metrics": [],
            "holding_period_sensitivity": [],
            "weighting_sensitivity": [],
            "turnover_summary": {},
            "transaction_cost_sensitivity": {},
            "concentration_exposure_diagnostics": {},
            "benchmark_relative_evaluation": benchmark_eval,
        }
        package = {
            "schema_version": LEVEL2_PORTFOLIO_VALIDATION_SCHEMA_VERSION,
            "package_type": LEVEL2_PORTFOLIO_VALIDATION_PACKAGE_TYPE,
            "created_at_utc": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
            "input_case_identity": case_identity,
            "promotion_decision_context": promotion_context,
            "portfolio_validation_settings": protocol_settings,
            "key_portfolio_results": {},
            "major_risks": risks,
            "major_caveats": caveats,
            "portfolio_robustness_summary": robustness_summary,
            "recommendation": {
                "label": recommendation,
                "remains_credible_at_portfolio_level": False,
                "rationale": ["portfolio-validation inputs are unavailable"],
            },
        }
        return Level2PortfolioValidationBundle(
            summary=cast(Level2PortfolioValidationSummaryPayload, _jsonable(summary)),
            metrics=cast(Level2PortfolioValidationMetricsPayload, _jsonable(metrics)),
            package=cast(Level2PortfolioValidationPackagePayload, _jsonable(package)),
        )

    eval_factor, eval_returns = _prepare_eval_inputs(experiment_result)
    if eval_factor.empty or eval_returns.empty:
        benchmark_eval = _benchmark_relative_summary(key_metrics, config=config)
        risks = ["portfolio-validation inputs are empty after temporal alignment"]
        caveats = [
            "Unable to compute portfolio scenarios because no aligned factor/return "
            "rows remained in the evaluation period."
        ]
        recommendation = LEVEL2_PORTFOLIO_VALIDATION_RECOMMENDATION_TAXONOMY[1]
        robustness_summary = _inconclusive_portfolio_robustness_summary(
            reason=(
                "Portfolio scenario evidence is unavailable because aligned factor/return "
                "rows were empty in the evaluation window."
            ),
            fragility_reasons=risks,
            benchmark_eval=benchmark_eval,
        )
        summary = {
            "validation_status": "blocked_empty_inputs",
            "promotion_decision": promotion_label,
            "recommendation": recommendation,
            "remains_credible_at_portfolio_level": False,
            "major_risks": risks,
            "major_caveats": caveats,
            "portfolio_robustness_summary": robustness_summary,
            **_summary_benchmark_slice(benchmark_eval),
        }
        metrics = {
            "protocol_settings": protocol_settings,
            "scenario_metrics": [],
            "holding_period_sensitivity": [],
            "weighting_sensitivity": [],
            "turnover_summary": {},
            "transaction_cost_sensitivity": {},
            "concentration_exposure_diagnostics": {},
            "benchmark_relative_evaluation": benchmark_eval,
        }
        package = {
            "schema_version": LEVEL2_PORTFOLIO_VALIDATION_SCHEMA_VERSION,
            "package_type": LEVEL2_PORTFOLIO_VALIDATION_PACKAGE_TYPE,
            "created_at_utc": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
            "input_case_identity": case_identity,
            "promotion_decision_context": promotion_context,
            "portfolio_validation_settings": protocol_settings,
            "key_portfolio_results": {},
            "major_risks": risks,
            "major_caveats": caveats,
            "portfolio_robustness_summary": robustness_summary,
            "recommendation": {
                "label": recommendation,
                "remains_credible_at_portfolio_level": False,
                "rationale": ["no aligned factor/label inputs for portfolio validation"],
            },
        }
        return Level2PortfolioValidationBundle(
            summary=cast(Level2PortfolioValidationSummaryPayload, _jsonable(summary)),
            metrics=cast(Level2PortfolioValidationMetricsPayload, _jsonable(metrics)),
            package=cast(Level2PortfolioValidationPackagePayload, _jsonable(package)),
        )

    leg_k = _determine_leg_size(experiment_result, key_metrics)
    scenarios = _run_scenarios(
        eval_factor=eval_factor,
        eval_returns=eval_returns,
        methods=methods,
        holding_grid=holding_grid,
        cost_grid=cost_grid,
        rebalance_step=rebalance_step,
        leg_k=leg_k,
    )
    benchmark_eval = _benchmark_relative_summary(key_metrics, config=config)

    if not scenarios:
        risks = ["portfolio scenarios failed to produce finite outputs"]
        caveats = [
            "Portfolio scenario engine returned no finite scenario metrics for the "
            "configured protocol grid."
        ]
        recommendation = LEVEL2_PORTFOLIO_VALIDATION_RECOMMENDATION_TAXONOMY[1]
        robustness_summary = _inconclusive_portfolio_robustness_summary(
            reason=(
                "Portfolio scenario evidence is unavailable because no finite "
                "scenario rows were produced."
            ),
            fragility_reasons=risks,
            benchmark_eval=benchmark_eval,
        )
        summary = {
            "validation_status": "blocked_no_scenarios",
            "promotion_decision": promotion_label,
            "recommendation": recommendation,
            "remains_credible_at_portfolio_level": False,
            "major_risks": risks,
            "major_caveats": caveats,
            "portfolio_robustness_summary": robustness_summary,
            **_summary_benchmark_slice(benchmark_eval),
        }
        metrics = {
            "protocol_settings": protocol_settings,
            "scenario_metrics": [],
            "holding_period_sensitivity": [],
            "weighting_sensitivity": [],
            "turnover_summary": {},
            "transaction_cost_sensitivity": {},
            "concentration_exposure_diagnostics": {},
            "benchmark_relative_evaluation": benchmark_eval,
        }
        package = {
            "schema_version": LEVEL2_PORTFOLIO_VALIDATION_SCHEMA_VERSION,
            "package_type": LEVEL2_PORTFOLIO_VALIDATION_PACKAGE_TYPE,
            "created_at_utc": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
            "input_case_identity": case_identity,
            "promotion_decision_context": promotion_context,
            "portfolio_validation_settings": protocol_settings,
            "key_portfolio_results": {},
            "major_risks": risks,
            "major_caveats": caveats,
            "portfolio_robustness_summary": robustness_summary,
            "recommendation": {
                "label": recommendation,
                "remains_credible_at_portfolio_level": False,
                "rationale": ["no finite scenario metrics were generated"],
            },
        }
        return Level2PortfolioValidationBundle(
            summary=cast(Level2PortfolioValidationSummaryPayload, _jsonable(summary)),
            metrics=cast(Level2PortfolioValidationMetricsPayload, _jsonable(metrics)),
            package=cast(Level2PortfolioValidationPackagePayload, _jsonable(package)),
        )

    base = _select_base_scenario(
        scenarios,
        default_method=config.default_weighting_method,
        default_holding=config.default_holding_period,
    )
    holding_sensitivity = _holding_sensitivity_rows(
        scenarios,
        weighting_method=str(base["weighting_method"]),
        review_cost_rate=config.review_cost_rate,
    )
    weighting_sensitivity = _weighting_sensitivity_rows(
        scenarios,
        holding_period=int(base["holding_period"]),
        method_order=methods,
        review_cost_rate=config.review_cost_rate,
    )

    cost_by_rate = _coerce_mapping(base.get("mean_cost_adjusted_return_by_cost_rate"))
    review_cost_key = _rate_key(config.review_cost_rate)
    review_cost_adjusted = _to_float(cost_by_rate.get(review_cost_key))
    turnover_values = [_to_float(row.get("mean_turnover")) for row in scenarios]
    turnover_finite = [x for x in turnover_values if x is not None]
    base_mean_return = _to_float(base.get("mean_portfolio_return"))
    base_mean_turnover = _to_float(base.get("mean_turnover"))
    base_max_abs_weight = _to_float(base.get("max_abs_weight_mean"))
    base_effective_names = _to_float(base.get("effective_names_mean"))
    benchmark_eval = _benchmark_relative_summary(
        key_metrics,
        base_mean_return=base_mean_return,
        config=config,
    )
    benchmark_risks = _to_text_list(benchmark_eval.get("risk_flags"))

    risks = []
    if base_mean_turnover is not None and base_mean_turnover > config.max_mean_turnover_warn:
        risks.append("high portfolio turnover under baseline assumptions")
    if (
        review_cost_adjusted is not None
        and review_cost_adjusted <= config.min_cost_adjusted_return_warn
    ):
        risks.append("baseline return does not survive simple transaction-cost stress")
    if (
        base_max_abs_weight is not None
        and base_max_abs_weight > config.max_single_name_weight_warn
    ):
        risks.append("portfolio concentration is high in single names")
    if base_effective_names is not None and base_effective_names < config.min_effective_names_warn:
        risks.append("effective diversification is low")
    if base_mean_return is not None and base_mean_return <= 0.0:
        risks.append("baseline portfolio return is non-positive")
    hold_returns = [_to_float(row.get("mean_portfolio_return")) for row in holding_sensitivity]
    finite_hold_returns = [x for x in hold_returns if x is not None]
    if (
        finite_hold_returns
        and min(finite_hold_returns) < config.sensitivity_sign_flip_pivot_return
        < max(finite_hold_returns)
    ):
        risks.append("holding-period sensitivity shows sign instability")
    for benchmark_risk in benchmark_risks:
        _append_unique(risks, benchmark_risk)
    benchmark_excess_weak = "excess return is weak relative to benchmark" in benchmark_risks
    standalone_does_not_survive_relative = False
    if (
        base_mean_return is not None
        and base_mean_return > 0.0
        and benchmark_excess_weak
    ):
        standalone_does_not_survive_relative = True
        _append_unique(
            risks,
            "apparent standalone strength does not survive relative comparison",
        )
    if promotion_blockers:
        risks.append("promotion context still carries unresolved blockers")

    caveats = [
        "Research-grade portfolio approximation only; no execution replay or fill simulation.",
    ]
    if benchmark_eval.get("status") == "not_available":
        caveats.append(
            "Benchmark-relative diagnostics are unavailable in current case artifacts."
        )
    if rebalance_source == "fallback_default":
        caveats.append(
            "Rebalance step used a conservative default because frequency token was unrecognized."
        )

    if risks:
        recommendation = LEVEL2_PORTFOLIO_VALIDATION_RECOMMENDATION_TAXONOMY[1]
        credible = False
        rationale_candidates: list[str] = []
        if benchmark_excess_weak:
            rationale_candidates.append("excess return is weak relative to benchmark")
        if "benchmark-relative risk is elevated" in benchmark_risks:
            rationale_candidates.append("benchmark-relative risk is elevated")
        if standalone_does_not_survive_relative:
            rationale_candidates.append(
                "apparent standalone strength does not survive relative comparison"
            )
        rationale_candidates.extend(risks)
        rationale = _dedupe_text(rationale_candidates)[:3]
    else:
        recommendation = LEVEL2_PORTFOLIO_VALIDATION_RECOMMENDATION_TAXONOMY[0]
        credible = True
        rationale = [
            (
                "baseline turnover, concentration, and transaction-cost checks remain "
                "within protocol limits"
            )
        ]
        if benchmark_eval.get("status") == "available":
            rationale.append("remains credible relative to benchmark")

    concentration_summary: ConcentrationExposureDiagnosticsPayload = {
        "max_abs_weight_mean": base.get("max_abs_weight_mean"),
        "top5_abs_weight_share_mean": base.get("top5_abs_weight_share_mean"),
        "effective_names_mean": base.get("effective_names_mean"),
        "gross_exposure_mean": base.get("gross_exposure_mean"),
        "net_exposure_mean": base.get("net_exposure_mean"),
    }
    turnover_summary: dict[str, float | None] = {
        "base_mean_turnover": base.get("mean_turnover"),
        "scenario_mean_turnover_min": min(turnover_finite) if turnover_finite else None,
        "scenario_mean_turnover_max": max(turnover_finite) if turnover_finite else None,
    }
    transaction_cost_summary: dict[str, object] = {
        "review_cost_rate": config.review_cost_rate,
        "baseline_by_cost_rate": cost_by_rate,
    }
    robustness_summary = _build_portfolio_robustness_summary(
        validation_status="completed",
        recommendation=recommendation,
        risks=risks,
        base_mean_return=base_mean_return,
        base_mean_turnover=base_mean_turnover,
        review_cost_adjusted=review_cost_adjusted,
        holding_sensitivity=holding_sensitivity,
        weighting_sensitivity=weighting_sensitivity,
        cost_by_rate=cost_by_rate,
        concentration_summary=concentration_summary,
        benchmark_eval=benchmark_eval,
        config=config,
    )

    summary = {
        "validation_status": "completed",
        "promotion_decision": promotion_label,
        "recommendation": recommendation,
        "remains_credible_at_portfolio_level": credible,
        "base_weighting_method": base.get("weighting_method"),
        "base_holding_period": base.get("holding_period"),
        "rebalance_step_assumption": rebalance_step,
        "base_mean_portfolio_return": base.get("mean_portfolio_return"),
        "base_mean_turnover": base.get("mean_turnover"),
        "base_cost_adjusted_return_review_rate": review_cost_adjusted,
        "major_risks": risks,
        "major_caveats": caveats,
        "portfolio_robustness_summary": robustness_summary,
        **_summary_benchmark_slice(benchmark_eval),
    }
    metrics = {
        "protocol_settings": protocol_settings,
        "scenario_metrics": scenarios,
        "holding_period_sensitivity": holding_sensitivity,
        "weighting_sensitivity": weighting_sensitivity,
        "turnover_summary": turnover_summary,
        "transaction_cost_sensitivity": transaction_cost_summary,
        "concentration_exposure_diagnostics": concentration_summary,
        "benchmark_relative_evaluation": benchmark_eval,
    }
    package = {
        "schema_version": LEVEL2_PORTFOLIO_VALIDATION_SCHEMA_VERSION,
        "package_type": LEVEL2_PORTFOLIO_VALIDATION_PACKAGE_TYPE,
        "created_at_utc": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "input_case_identity": case_identity,
        "promotion_decision_context": promotion_context,
        "portfolio_validation_settings": protocol_settings,
        "key_portfolio_results": {
            "baseline_scenario": base,
            "holding_period_sensitivity": holding_sensitivity,
            "weighting_sensitivity": weighting_sensitivity,
            "turnover_summary": turnover_summary,
            "transaction_cost_sensitivity": transaction_cost_summary,
            "concentration_exposure_diagnostics": concentration_summary,
            "benchmark_relative_evaluation": benchmark_eval,
            "portfolio_robustness_summary": robustness_summary,
        },
        "major_risks": risks,
        "major_caveats": caveats,
        "portfolio_robustness_summary": robustness_summary,
        "recommendation": {
            "label": recommendation,
            "remains_credible_at_portfolio_level": credible,
            "rationale": rationale,
        },
    }

    summary_payload = cast(Level2PortfolioValidationSummaryPayload, _jsonable(summary))
    metrics_payload = cast(Level2PortfolioValidationMetricsPayload, _jsonable(metrics))
    package_payload = cast(Level2PortfolioValidationPackagePayload, _jsonable(package))

    validate_portfolio_validation_summary_payload(summary_payload)
    validate_portfolio_validation_metrics_payload(metrics_payload)
    validate_portfolio_validation_package_payload(package_payload)

    return Level2PortfolioValidationBundle(
        summary=summary_payload,
        metrics=metrics_payload,
        package=package_payload,
    )


def export_level2_portfolio_validation_bundle(
    bundle: Level2PortfolioValidationBundle,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write Level 2 portfolio-validation bundle artifacts."""

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / "portfolio_validation_summary.json"
    metrics_path = out_dir / "portfolio_validation_metrics.json"
    package_path = out_dir / "portfolio_validation_package.json"
    markdown_path = out_dir / "portfolio_validation_package.md"

    validate_portfolio_validation_summary_payload(bundle.summary, source=summary_path)
    validate_portfolio_validation_metrics_payload(bundle.metrics, source=metrics_path)
    validate_portfolio_validation_package_payload(bundle.package, source=package_path)

    summary_path.write_text(
        json.dumps(bundle.summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    metrics_path.write_text(
        json.dumps(bundle.metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    package_path.write_text(
        json.dumps(bundle.package, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    markdown_path.write_text(_portfolio_validation_markdown(bundle), encoding="utf-8")

    return {
        "summary": summary_path,
        "metrics": metrics_path,
        "package_json": package_path,
        "package_markdown": markdown_path,
    }


def _portfolio_validation_markdown(bundle: Level2PortfolioValidationBundle) -> str:
    summary = bundle.summary
    metrics = bundle.metrics
    package = bundle.package
    baseline = _coerce_mapping(
        _coerce_mapping(package.get("key_portfolio_results")).get("baseline_scenario")
    )
    concentration = _coerce_mapping(metrics.get("concentration_exposure_diagnostics"))
    turnover = _coerce_mapping(metrics.get("turnover_summary"))
    transaction_cost = _coerce_mapping(metrics.get("transaction_cost_sensitivity"))
    benchmark = _coerce_mapping(metrics.get("benchmark_relative_evaluation"))
    robustness = _coerce_mapping(summary.get("portfolio_robustness_summary"))

    lines = [
        "# Level 2 Portfolio Validation Package",
        "",
        f"- Validation status: `{_safe_str(summary.get('validation_status')) or 'N/A'}`",
        f"- Promotion decision: `{_safe_str(summary.get('promotion_decision')) or 'N/A'}`",
        f"- Recommendation: `{_safe_str(summary.get('recommendation')) or 'N/A'}`",
    ]
    if robustness:
        lines.extend(
            [
                "",
                "## Portfolio Robustness Summary",
                "",
                (
                    "- Taxonomy label: "
                    f"`{_safe_str(robustness.get('taxonomy_label')) or 'N/A'}`"
                ),
                (
                    "- Benchmark-relative support: "
                    f"{_safe_str(robustness.get('benchmark_relative_support_note')) or 'N/A'}"
                ),
                (
                    "- Cost sensitivity: "
                    f"{_safe_str(robustness.get('cost_sensitivity_note')) or 'N/A'}"
                ),
                (
                    "- Concentration/turnover: "
                    f"{_safe_str(robustness.get('concentration_turnover_risk_note')) or 'N/A'}"
                ),
            ]
        )
        support_reasons = _to_text_list(robustness.get("support_reasons"))
        fragility_reasons = _to_text_list(robustness.get("fragility_reasons"))
        sensitivity_notes = _to_text_list(robustness.get("scenario_sensitivity_notes"))
        if support_reasons:
            for reason in support_reasons:
                lines.append(f"- Support reason: {reason}")
        else:
            lines.append("- Support reason: none")
        if fragility_reasons:
            for reason in fragility_reasons:
                lines.append(f"- Fragility reason: {reason}")
        else:
            lines.append("- Fragility reason: none")
        if sensitivity_notes:
            for sensitivity_note in sensitivity_notes:
                lines.append(f"- Scenario sensitivity: {sensitivity_note}")

    lines.extend(
        [
            "",
            "## Baseline Scenario",
            "",
        (
            "- Weighting / holding / rebalance-step: "
            f"`{_safe_str(baseline.get('weighting_method')) or 'N/A'}` / "
            f"`{_format_value(baseline.get('holding_period'))}` / "
            f"`{_format_value(baseline.get('rebalance_step'))}`"
        ),
        (
            "- Mean return / IR / turnover: "
            f"{_format_value(baseline.get('mean_portfolio_return'))} / "
            f"{_format_value(baseline.get('portfolio_ir'))} / "
            f"{_format_value(baseline.get('mean_turnover'))}"
        ),
        "",
        "## Concentration & Exposure",
        "",
        (
            "- Max abs weight / top-5 abs share / effective names: "
            f"{_format_value(concentration.get('max_abs_weight_mean'))} / "
            f"{_format_value(concentration.get('top5_abs_weight_share_mean'))} / "
            f"{_format_value(concentration.get('effective_names_mean'))}"
        ),
        (
            "- Gross / net exposure: "
            f"{_format_value(concentration.get('gross_exposure_mean'))} / "
            f"{_format_value(concentration.get('net_exposure_mean'))}"
        ),
        "",
        "## Turnover & Cost Sensitivity",
        "",
        (
            "- Turnover range (scenario mean min -> max): "
            f"{_format_value(turnover.get('scenario_mean_turnover_min'))} -> "
            f"{_format_value(turnover.get('scenario_mean_turnover_max'))}"
        ),
        (
            "- Review cost rate: "
            f"{_format_value(transaction_cost.get('review_cost_rate'))}"
        ),
    ]
    )

    cost_rows = _coerce_mapping(transaction_cost.get("baseline_by_cost_rate"))
    if cost_rows:
        for rate, value in sorted(cost_rows.items(), key=lambda row: row[0]):
            lines.append(f"- Cost {rate}: adjusted mean return={_format_value(value)}")
    else:
        lines.append("- Cost sensitivity: N/A")

    lines.extend(["", "## Benchmark-Relative Evaluation", ""])
    lines.append(f"- Status: `{_safe_str(benchmark.get('status')) or 'N/A'}`")
    assessment = _safe_str(benchmark.get("assessment"))
    if assessment is not None:
        lines.append(f"- Assessment: `{assessment}`")
    benchmark_fields = [
        ("benchmark_name", "Benchmark"),
        ("benchmark_excess_return", "Excess return"),
        ("benchmark_active_return", "Active return"),
        ("benchmark_relative_return", "Relative return"),
        ("benchmark_long_short_excess_return", "Long-short excess return"),
        ("benchmark_information_ratio", "Information ratio"),
        ("benchmark_tracking_error", "Tracking error"),
        ("benchmark_return_estimate_from_excess", "Estimated benchmark return"),
        ("benchmark_relative_max_drawdown", "Relative drawdown gap"),
    ]
    for key, label in benchmark_fields:
        if key in benchmark:
            lines.append(f"- {label}: {_format_value(benchmark.get(key))}")
    benchmark_risks = _to_text_list(benchmark.get("risk_flags"))
    if benchmark_risks:
        for risk in benchmark_risks:
            lines.append(f"- Relative risk: {risk}")
    elif benchmark.get("status") == "available":
        lines.append("- Relative risk: none")
    benchmark_note = _safe_str(benchmark.get("note"))
    if benchmark_note:
        lines.append(f"- Note: {benchmark_note}")

    risks = _to_text_list(summary.get("major_risks"))
    caveats = _to_text_list(summary.get("major_caveats"))
    lines.extend(["", "## Risks & Caveats", ""])
    if risks:
        for risk in risks:
            lines.append(f"- Risk: {risk}")
    else:
        lines.append("- Risk: none")
    if caveats:
        for caveat in caveats:
            lines.append(f"- Caveat: {caveat}")
    else:
        lines.append("- Caveat: none")
    lines.append("")
    return "\n".join(lines)


def _prepare_eval_inputs(result: ExperimentResult) -> tuple[pd.DataFrame, pd.DataFrame]:
    factor = result.factor_df[["date", "asset", "factor", "value"]].copy()
    returns = result.label_df[["date", "asset", "value"]].copy()
    factor["date"] = pd.to_datetime(factor["date"], errors="coerce")
    returns["date"] = pd.to_datetime(returns["date"], errors="coerce")
    factor = factor.dropna(subset=["date", "asset", "value"]).copy()
    returns = returns.dropna(subset=["date", "asset", "value"]).copy()

    if not result.long_short_df.empty and "date" in result.long_short_df.columns:
        eval_dates = pd.to_datetime(result.long_short_df["date"], errors="coerce").dropna()
        if len(eval_dates) > 0:
            unique_dates = set(pd.DatetimeIndex(eval_dates).tolist())
            factor = factor[factor["date"].isin(unique_dates)].copy()
            returns = returns[returns["date"].isin(unique_dates)].copy()

    return (
        factor.sort_values(["date", "asset"], kind="mergesort").reset_index(drop=True),
        returns.sort_values(["date", "asset"], kind="mergesort").reset_index(drop=True),
    )


def _determine_leg_size(result: ExperimentResult, key_metrics: Mapping[str, object]) -> int:
    assignments = result.quantile_assignments_df
    if not assignments.empty and {"date", "asset", "quantile"}.issubset(assignments.columns):
        frame = assignments[["date", "asset", "quantile"]].copy()
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame["quantile"] = pd.to_numeric(frame["quantile"], errors="coerce")
        frame = frame.dropna(subset=["date", "asset", "quantile"])
        if not frame.empty:
            qmax = int(frame["quantile"].max())
            top_counts = frame[frame["quantile"] == qmax].groupby("date")["asset"].nunique()
            bottom_counts = frame[frame["quantile"] == 1].groupby("date")["asset"].nunique()
            joined = pd.concat([top_counts.rename("top"), bottom_counts.rename("bottom")], axis=1)
            joined = joined.dropna(how="any")
            if not joined.empty:
                median_leg = float(joined[["top", "bottom"]].min(axis=1).median())
                if math.isfinite(median_leg) and median_leg >= 1:
                    return int(max(1, math.floor(median_leg)))

    metrics_contract = project_portfolio_validation_metrics(key_metrics)
    mean_assets = metrics_contract["mean_eval_assets_per_date"]
    n_quantiles = metrics_contract["n_quantiles"]
    if (
        mean_assets is not None
        and n_quantiles is not None
        and n_quantiles >= 2
        and mean_assets >= 2
    ):
        return int(max(1, math.floor(mean_assets / n_quantiles)))
    return 5


def _run_scenarios(
    *,
    eval_factor: pd.DataFrame,
    eval_returns: pd.DataFrame,
    methods: tuple[str, ...],
    holding_grid: tuple[int, ...],
    cost_grid: tuple[float, ...],
    rebalance_step: int,
    leg_k: int,
) -> list[ScenarioMetricsRow]:
    scenarios: list[ScenarioMetricsRow] = []
    for method in methods:
        for holding_period in holding_grid:
            weights = portfolio_weights(
                eval_factor,
                method=method,
                top_k=leg_k,
                bottom_k=leg_k,
            )
            if weights.empty:
                continue
            portfolio_return = simulate_portfolio_returns(
                weights,
                eval_returns,
                holding_period=holding_period,
                rebalance_frequency=rebalance_step,
            )
            if portfolio_return.empty:
                continue

            turnover = portfolio_turnover(weights)
            active_turnover = _active_turnover(turnover, rebalance_step=rebalance_step)

            ret_values = pd.to_numeric(
                portfolio_return["portfolio_return"],
                errors="coerce",
            ).dropna()
            mean_return = float(ret_values.mean()) if len(ret_values) > 0 else float("nan")
            ir = _mean_div_std(ret_values)
            hit_rate = float((ret_values > 0).mean()) if len(ret_values) > 0 else float("nan")

            turnover_values = (
                pd.to_numeric(active_turnover["portfolio_turnover"], errors="coerce").dropna()
                if not active_turnover.empty
                else pd.Series(dtype=float)
            )
            mean_turnover = (
                float(turnover_values.mean()) if len(turnover_values) > 0 else float("nan")
            )

            concentration = _concentration_metrics(weights)

            cost_adjusted_by_rate: dict[str, float | None] = {}
            for rate in cost_grid:
                cost_adjusted = portfolio_cost_adjusted_returns(
                    portfolio_return,
                    active_turnover,
                    cost_rate=rate,
                )
                adjusted = pd.to_numeric(
                    cost_adjusted["adjusted_return"],
                    errors="coerce",
                ).dropna()
                cost_adjusted_by_rate[_rate_key(rate)] = (
                    float(adjusted.mean()) if len(adjusted) > 0 else None
                )

            scenarios.append(
                {
                    "weighting_method": method,
                    "holding_period": holding_period,
                    "rebalance_step": rebalance_step,
                    "leg_size_k": leg_k,
                    "n_return_dates": int(len(ret_values)),
                    "mean_portfolio_return": _finite_or_none(mean_return),
                    "portfolio_ir": _finite_or_none(ir),
                    "portfolio_hit_rate": _finite_or_none(hit_rate),
                    "mean_turnover": _finite_or_none(mean_turnover),
                    "max_abs_weight_mean": concentration["max_abs_weight_mean"],
                    "top5_abs_weight_share_mean": concentration["top5_abs_weight_share_mean"],
                    "effective_names_mean": concentration["effective_names_mean"],
                    "gross_exposure_mean": concentration["gross_exposure_mean"],
                    "net_exposure_mean": concentration["net_exposure_mean"],
                    "mean_cost_adjusted_return_by_cost_rate": cost_adjusted_by_rate,
                }
            )
    return scenarios


def _select_base_scenario(
    scenarios: Sequence[ScenarioMetricsRow],
    *,
    default_method: str,
    default_holding: int,
) -> ScenarioMetricsRow:
    for row in scenarios:
        if (
            _safe_str(row.get("weighting_method")) == default_method
            and _to_int(row.get("holding_period")) == default_holding
        ):
            return row
    return scenarios[0]


def _holding_sensitivity_rows(
    scenarios: Sequence[ScenarioMetricsRow],
    *,
    weighting_method: str,
    review_cost_rate: float,
) -> list[HoldingSensitivityRow]:
    out: list[HoldingSensitivityRow] = []
    for row in scenarios:
        if _safe_str(row.get("weighting_method")) != weighting_method:
            continue
        cost_map = _coerce_mapping(row.get("mean_cost_adjusted_return_by_cost_rate"))
        out.append(
            {
                "holding_period": _to_int(row.get("holding_period")),
                "mean_portfolio_return": _to_float(row.get("mean_portfolio_return")),
                "mean_turnover": _to_float(row.get("mean_turnover")),
                "mean_cost_adjusted_return_review_rate": _to_float(
                    cost_map.get(
                    _rate_key(review_cost_rate)
                    )
                ),
            }
        )
    out.sort(key=lambda row: row["holding_period"] if isinstance(row["holding_period"], int) else 0)
    return out


def _weighting_sensitivity_rows(
    scenarios: Sequence[ScenarioMetricsRow],
    *,
    holding_period: int,
    method_order: Sequence[str],
    review_cost_rate: float,
) -> list[WeightingSensitivityRow]:
    order = {name: idx for idx, name in enumerate(method_order)}
    out: list[WeightingSensitivityRow] = []
    for row in scenarios:
        if _to_int(row.get("holding_period")) != holding_period:
            continue
        method = _safe_str(row.get("weighting_method")) or "unknown"
        cost_map = _coerce_mapping(row.get("mean_cost_adjusted_return_by_cost_rate"))
        out.append(
            {
                "weighting_method": method,
                "mean_portfolio_return": _to_float(row.get("mean_portfolio_return")),
                "mean_turnover": _to_float(row.get("mean_turnover")),
                "mean_cost_adjusted_return_review_rate": _to_float(
                    cost_map.get(
                    _rate_key(review_cost_rate)
                    )
                ),
            }
        )
    out.sort(key=lambda row: order.get(_safe_str(row.get("weighting_method")) or "", 10**6))
    return out


def _benchmark_relative_summary(
    metrics: Mapping[str, object],
    *,
    base_mean_return: float | None = None,
    config: Level2PortfolioValidationConfig = DEFAULT_LEVEL2_PORTFOLIO_VALIDATION_CONFIG,
) -> BenchmarkRelativeEvaluationPayload:
    metrics_contract = project_portfolio_validation_metrics(metrics)
    benchmark_name = metrics_contract["benchmark_name"]
    benchmark_excess_return = _first_non_none_float(
        metrics_contract["benchmark_excess_return"],
        metrics_contract["portfolio_validation_benchmark_excess_return"],
        _first_metric_float(
            metrics,
            (
                "benchmark_relative_return",
                "benchmark_active_return",
            ),
        ),
    )
    benchmark_active_return = _first_non_none_float(
        metrics_contract["benchmark_active_return"],
        _first_metric_float(
            metrics,
            (
                "benchmark_relative_return",
                "benchmark_excess_return",
            ),
        ),
    )
    benchmark_relative_return = metrics_contract["benchmark_relative_return"]
    benchmark_long_short_excess = _first_non_none_float(
        metrics_contract["benchmark_long_short_excess_return"],
        _first_metric_float(
            metrics,
            (
                "benchmark_excess_long_short_return",
                "benchmark_relative_long_short_return",
            ),
        ),
    )
    benchmark_information_ratio = _first_non_none_float(
        metrics_contract["benchmark_information_ratio"],
        _first_metric_float(
            metrics,
            (
                "benchmark_active_information_ratio",
                "benchmark_relative_information_ratio",
            ),
        ),
    )
    benchmark_tracking_error = _first_non_none_float(
        metrics_contract["benchmark_tracking_error"],
        metrics_contract["portfolio_validation_benchmark_tracking_error"],
        _first_metric_float(
            metrics,
            (
                "benchmark_active_risk",
                "benchmark_relative_risk",
            ),
        ),
    )
    benchmark_relative_drawdown = _first_non_none_float(
        metrics_contract["benchmark_relative_max_drawdown"],
        _first_metric_float(
            metrics,
            (
                "benchmark_relative_drawdown",
                "benchmark_active_drawdown",
                "benchmark_excess_drawdown",
            ),
        ),
    )
    portfolio_max_drawdown = _first_non_none_float(
        metrics_contract["portfolio_max_drawdown"],
        _first_metric_float(
            metrics,
            ("base_portfolio_max_drawdown",),
        ),
    )
    benchmark_max_drawdown = _first_non_none_float(
        metrics_contract["benchmark_max_drawdown"],
        _first_metric_float(
            metrics,
            ("benchmark_drawdown",),
        ),
    )
    if (
        benchmark_relative_drawdown is None
        and portfolio_max_drawdown is not None
        and benchmark_max_drawdown is not None
    ):
        benchmark_relative_drawdown = (
            abs(portfolio_max_drawdown) - abs(benchmark_max_drawdown)
        )

    has_relative_signal = any(
        value is not None
        for value in (
            benchmark_excess_return,
            benchmark_active_return,
            benchmark_relative_return,
            benchmark_long_short_excess,
            benchmark_information_ratio,
            benchmark_tracking_error,
            benchmark_relative_drawdown,
        )
    )
    if not has_relative_signal:
        note = "benchmark-relative metrics are not present in case evidence"
        if benchmark_name is not None:
            note = (
                f"benchmark `{benchmark_name}` is named but benchmark-relative metrics "
                "are not present in case evidence"
            )
        payload: BenchmarkRelativeEvaluationPayload = {"status": "not_available", "note": note}
        if benchmark_name is not None:
            payload["benchmark_name"] = benchmark_name
        return payload

    benchmark_return_estimate = (
        base_mean_return - benchmark_excess_return
        if base_mean_return is not None and benchmark_excess_return is not None
        else None
    )

    risk_flags: list[str] = []
    if (
        benchmark_excess_return is not None
        and benchmark_excess_return <= config.min_benchmark_excess_return_warn
    ):
        risk_flags.append("excess return is weak relative to benchmark")
    if (
        benchmark_information_ratio is not None
        and benchmark_information_ratio <= config.min_benchmark_information_ratio_warn
    ):
        risk_flags.append("benchmark-relative risk-adjusted return is weak")
    if (
        benchmark_tracking_error is not None
        and benchmark_tracking_error >= config.max_benchmark_tracking_error_warn
    ):
        risk_flags.append("benchmark-relative risk is elevated")
    if (
        benchmark_relative_drawdown is not None
        and benchmark_relative_drawdown >= config.max_benchmark_relative_drawdown_warn
    ):
        risk_flags.append("benchmark-relative drawdown is elevated")

    return {
        "status": "available",
        "assessment": (
            "supports_standalone_strength" if not risk_flags else "relative_risk_elevated"
        ),
        "interpretation": (
            "remains credible relative to benchmark"
            if not risk_flags
            else "benchmark-relative risk is elevated"
        ),
        "benchmark_name": benchmark_name,
        "benchmark_excess_return": benchmark_excess_return,
        "benchmark_active_return": benchmark_active_return,
        "benchmark_relative_return": benchmark_relative_return,
        "benchmark_long_short_excess_return": benchmark_long_short_excess,
        "benchmark_information_ratio": benchmark_information_ratio,
        "benchmark_tracking_error": benchmark_tracking_error,
        "benchmark_relative_max_drawdown": benchmark_relative_drawdown,
        "benchmark_return_estimate_from_excess": benchmark_return_estimate,
        "risk_flags": risk_flags,
    }


def _summary_benchmark_slice(
    benchmark_eval: BenchmarkRelativeEvaluationPayload,
) -> SummaryBenchmarkSlice:
    return {
        "benchmark_relative_status": benchmark_eval.get("status"),
        "benchmark_relative_assessment": benchmark_eval.get("assessment"),
        "benchmark_relative_excess_return": benchmark_eval.get("benchmark_excess_return"),
        "benchmark_relative_tracking_error": benchmark_eval.get("benchmark_tracking_error"),
        "benchmark_relative_risks": benchmark_eval.get("risk_flags", []),
    }


def _inconclusive_portfolio_robustness_summary(
    *,
    reason: str,
    fragility_reasons: Sequence[str] | None = None,
    benchmark_eval: BenchmarkRelativeEvaluationPayload | None = None,
) -> PortfolioRobustnessSummaryPayload:
    benchmark_note, _, _, _ = _benchmark_support_assessment(
        benchmark_eval or cast(BenchmarkRelativeEvaluationPayload, {})
    )
    fragilities = _dedupe_text(list(fragility_reasons or []))
    if not fragilities:
        fragilities = [reason]
    return {
        "taxonomy_label": LEVEL2_PORTFOLIO_ROBUSTNESS_TAXONOMY[3],
        "support_reasons": [],
        "fragility_reasons": fragilities,
        "scenario_sensitivity_notes": [
            (
                "Sensitivity diagnostics are inconclusive because portfolio "
                "validation did not complete."
            )
        ],
        "benchmark_relative_support_note": benchmark_note,
        "cost_sensitivity_note": (
            "Cost sensitivity is inconclusive because portfolio validation did "
            "not complete."
        ),
        "concentration_turnover_risk_note": (
            "Concentration/turnover diagnostics are inconclusive because "
            "portfolio validation did not complete."
        ),
    }


def _build_portfolio_robustness_summary(
    *,
    validation_status: str,
    recommendation: str,
    risks: Sequence[str],
    base_mean_return: float | None,
    base_mean_turnover: float | None,
    review_cost_adjusted: float | None,
    holding_sensitivity: Sequence[HoldingSensitivityRow],
    weighting_sensitivity: Sequence[WeightingSensitivityRow],
    cost_by_rate: Mapping[str, object],
    concentration_summary: ConcentrationExposureDiagnosticsPayload,
    benchmark_eval: BenchmarkRelativeEvaluationPayload,
    config: Level2PortfolioValidationConfig,
) -> PortfolioRobustnessSummaryPayload:
    if validation_status != "completed":
        return _inconclusive_portfolio_robustness_summary(
            reason="Portfolio validation status is not completed.",
            fragility_reasons=risks,
            benchmark_eval=benchmark_eval,
        )

    support_reasons: list[str] = []
    fragility_reasons: list[str] = _dedupe_text(risks)

    hold_note, hold_sign_instability, hold_material_sensitivity, hold_stable = (
        _scenario_sensitivity_assessment(
            holding_sensitivity,
            dimension="holding period",
            base_mean_return=base_mean_return,
            config=config,
        )
    )
    weighting_note, weight_sign_instability, weight_material_sensitivity, weight_stable = (
        _scenario_sensitivity_assessment(
            weighting_sensitivity,
            dimension="weighting method",
            base_mean_return=base_mean_return,
            config=config,
        )
    )
    scenario_notes = [hold_note, weighting_note]
    if hold_sign_instability:
        _append_unique(
            fragility_reasons,
            "holding-period sensitivity shows sign instability",
        )
    if weight_sign_instability:
        _append_unique(
            fragility_reasons,
            "weighting-method sensitivity shows sign instability",
        )
    if hold_stable:
        _append_unique(
            support_reasons,
            "holding-period sensitivity is stable across tested scenarios",
        )
    if weight_stable:
        _append_unique(
            support_reasons,
            "weighting-method sensitivity is stable across tested scenarios",
        )

    benchmark_note, benchmark_fragile, benchmark_supportive, benchmark_risks = (
        _benchmark_support_assessment(benchmark_eval)
    )
    for risk in benchmark_risks:
        _append_unique(fragility_reasons, risk)
    if benchmark_supportive:
        _append_unique(
            support_reasons,
            "benchmark-relative evidence supports standalone portfolio strength",
        )

    cost_note, cost_fragile, cost_supportive = _cost_sensitivity_assessment(
        cost_by_rate,
        review_cost_adjusted=review_cost_adjusted,
        config=config,
    )
    if cost_fragile:
        _append_unique(
            fragility_reasons,
            "transaction-cost sensitivity raises portfolio-level fragility",
        )
    if cost_supportive:
        _append_unique(
            support_reasons,
            "portfolio return remains positive across tested transaction-cost rates",
        )

    concentration_note, concentration_risks, concentration_supportive = (
        _concentration_turnover_assessment(
            base_mean_turnover=base_mean_turnover,
            concentration_summary=concentration_summary,
            config=config,
        )
    )
    for concentration_risk in concentration_risks:
        _append_unique(fragility_reasons, concentration_risk)
    if concentration_supportive:
        _append_unique(
            support_reasons,
            "turnover and concentration diagnostics stay within configured guardrails",
        )

    if base_mean_return is not None and base_mean_return > 0.0:
        _append_unique(
            support_reasons,
            "baseline portfolio return is positive under default assumptions",
        )

    severe_fragility_count = sum(
        (
            hold_sign_instability,
            weight_sign_instability,
            benchmark_fragile,
            cost_fragile,
            bool(concentration_risks),
        )
    )
    sensitivity_count = sum((hold_material_sensitivity, weight_material_sensitivity))
    fragile_cutoff = max(1, config.robustness_fragile_min_severe_signal_count)
    sensitive_severe_cutoff = max(1, config.robustness_sensitive_min_severe_signal_count)
    sensitive_material_cutoff = max(1, config.robustness_sensitive_min_material_signal_count)
    needs_refinement_sensitive = (
        config.robustness_needs_refinement_implies_sensitive
        and recommendation == LEVEL2_PORTFOLIO_VALIDATION_RECOMMENDATION_TAXONOMY[1]
    )
    if severe_fragility_count >= fragile_cutoff:
        taxonomy_label = LEVEL2_PORTFOLIO_ROBUSTNESS_TAXONOMY[2]
    elif (
        severe_fragility_count >= sensitive_severe_cutoff
        or sensitivity_count >= sensitive_material_cutoff
        or needs_refinement_sensitive
    ):
        taxonomy_label = LEVEL2_PORTFOLIO_ROBUSTNESS_TAXONOMY[1]
    else:
        taxonomy_label = LEVEL2_PORTFOLIO_ROBUSTNESS_TAXONOMY[0]

    return {
        "taxonomy_label": taxonomy_label,
        "support_reasons": support_reasons,
        "fragility_reasons": fragility_reasons,
        "scenario_sensitivity_notes": scenario_notes,
        "benchmark_relative_support_note": benchmark_note,
        "cost_sensitivity_note": cost_note,
        "concentration_turnover_risk_note": concentration_note,
    }


def _scenario_sensitivity_assessment(
    rows: Sequence[Mapping[str, object]],
    *,
    dimension: str,
    base_mean_return: float | None,
    config: Level2PortfolioValidationConfig,
) -> tuple[str, bool, bool, bool]:
    finite_returns = [
        value
        for value in (_to_float(row.get("mean_portfolio_return")) for row in rows)
        if value is not None
    ]
    if not finite_returns:
        return (
            f"{dimension} sensitivity is unavailable (no finite scenario return evidence).",
            False,
            False,
            False,
        )
    if len(finite_returns) == 1:
        return (
            f"{dimension} sensitivity is limited to one scenario and remains inconclusive.",
            False,
            False,
            False,
        )

    scenario_min = min(finite_returns)
    scenario_max = max(finite_returns)
    if scenario_min < config.sensitivity_sign_flip_pivot_return < scenario_max:
        return (
            (
                f"{dimension} sensitivity shows sign instability "
                f"({scenario_min:.6f} -> {scenario_max:.6f})."
            ),
            True,
            True,
            False,
        )

    spread = scenario_max - scenario_min
    base_abs = abs(base_mean_return) if base_mean_return is not None else None
    if base_abs is not None and base_abs > 0.0:
        spread_ratio = spread / base_abs
        if spread_ratio >= config.sensitivity_material_spread_ratio_warn:
            return (
                (
                    f"{dimension} sensitivity is material "
                    f"(range={spread:.6f}, baseline={base_mean_return:.6f})."
                ),
                False,
                True,
                False,
            )
        if spread_ratio <= config.sensitivity_stable_spread_ratio_max:
            return (
                (
                    f"{dimension} sensitivity is stable "
                    f"(range={spread:.6f}, baseline={base_mean_return:.6f})."
                ),
                False,
                False,
                True,
            )
    return (
        f"{dimension} sensitivity is mixed but does not flip return sign.",
        False,
        False,
        False,
    )


def _benchmark_support_assessment(
    benchmark_eval: BenchmarkRelativeEvaluationPayload,
) -> tuple[str, bool, bool, list[str]]:
    status = _safe_str(benchmark_eval.get("status")) or "not_available"
    risk_flags = _to_text_list(benchmark_eval.get("risk_flags"))
    if status != "available":
        return (
            "Benchmark-relative support is unavailable in current case evidence.",
            False,
            False,
            [],
        )
    if not risk_flags:
        return (
            "Benchmark-relative evidence supports standalone portfolio strength.",
            False,
            True,
            [],
        )
    return (
        "Benchmark-relative support is weak: " + "; ".join(risk_flags),
        True,
        False,
        risk_flags,
    )


def _cost_sensitivity_assessment(
    cost_by_rate: Mapping[str, object],
    *,
    review_cost_adjusted: float | None,
    config: Level2PortfolioValidationConfig,
) -> tuple[str, bool, bool]:
    finite_values = [
        value
        for value in (_to_float(raw) for raw in cost_by_rate.values())
        if value is not None
    ]
    if not finite_values:
        return (
            "Cost sensitivity evidence is unavailable in baseline scenario outputs.",
            False,
            False,
        )
    pivot = config.sensitivity_sign_flip_pivot_return
    if min(finite_values) <= pivot < max(finite_values):
        return (
            "Portfolio return flips sign across tested transaction-cost rates.",
            True,
            False,
        )
    if (
        review_cost_adjusted is not None
        and review_cost_adjusted <= config.min_cost_adjusted_return_warn
    ):
        return (
            "Baseline return does not survive the review transaction-cost assumption.",
            True,
            False,
        )
    if min(finite_values) > pivot:
        return (
            "Portfolio return remains positive across tested transaction-cost rates.",
            False,
            True,
        )
    return (
        "Cost sensitivity is mixed across tested transaction-cost rates.",
        False,
        False,
    )


def _concentration_turnover_assessment(
    *,
    base_mean_turnover: float | None,
    concentration_summary: Mapping[str, object],
    config: Level2PortfolioValidationConfig,
) -> tuple[str, list[str], bool]:
    risks: list[str] = []
    max_abs_weight = _to_float(concentration_summary.get("max_abs_weight_mean"))
    effective_names = _to_float(concentration_summary.get("effective_names_mean"))
    if (
        base_mean_turnover is not None
        and base_mean_turnover > config.max_mean_turnover_warn
    ):
        risks.append("turnover is high under baseline assumptions")
    if (
        max_abs_weight is not None
        and max_abs_weight > config.max_single_name_weight_warn
    ):
        risks.append("single-name concentration is high")
    if (
        effective_names is not None
        and effective_names < config.min_effective_names_warn
    ):
        risks.append("effective diversification is low")
    if not risks:
        return (
            "Turnover and concentration diagnostics stay within configured guardrails.",
            [],
            True,
        )
    return (
        "Turnover/concentration fragility: " + "; ".join(risks),
        risks,
        False,
    )


def _first_metric_float(
    metrics: Mapping[str, object],
    keys: Sequence[str],
) -> float | None:
    for key in keys:
        if key not in metrics:
            continue
        value = _to_float(metrics.get(key))
        if value is not None:
            return value
    return None


def _first_non_none_float(*values: float | None) -> float | None:
    for value in values:
        if value is not None:
            return value
    return None


def _concentration_metrics(weights: pd.DataFrame) -> ConcentrationExposureDiagnosticsPayload:
    if weights.empty:
        return {
            "max_abs_weight_mean": None,
            "top5_abs_weight_share_mean": None,
            "effective_names_mean": None,
            "gross_exposure_mean": None,
            "net_exposure_mean": None,
        }
    grouped = weights.groupby("date", sort=True)["weight"]
    max_abs = grouped.apply(lambda s: float(s.abs().max()))
    top5_abs = grouped.apply(lambda s: float(s.abs().nlargest(5).sum()))

    def _effective_names(series: pd.Series) -> float:
        squared = float(np.square(series.to_numpy(dtype=float)).sum())
        if squared <= 0.0:
            return float("nan")
        return float(1.0 / squared)

    effective_names = grouped.apply(_effective_names)
    gross = grouped.apply(lambda s: float(s.abs().sum()))
    net = grouped.apply(lambda s: float(s.sum()))

    return {
        "max_abs_weight_mean": _finite_or_none(float(max_abs.mean())),
        "top5_abs_weight_share_mean": _finite_or_none(float(top5_abs.mean())),
        "effective_names_mean": _finite_or_none(float(effective_names.mean())),
        "gross_exposure_mean": _finite_or_none(float(gross.mean())),
        "net_exposure_mean": _finite_or_none(float(net.mean())),
    }


def _active_turnover(turnover_df: pd.DataFrame, *, rebalance_step: int) -> pd.DataFrame:
    if turnover_df.empty:
        return turnover_df.copy()
    out = turnover_df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"]).sort_values("date", kind="mergesort").reset_index(drop=True)
    dates = sorted(out["date"].unique())
    active = set(dates[:: max(1, rebalance_step)])
    return out[out["date"].isin(active)].reset_index(drop=True)


def _mean_div_std(values: pd.Series) -> float:
    if len(values) < 2:
        return float("nan")
    mean_val = float(values.mean())
    std_val = float(values.std(ddof=1))
    if not math.isfinite(std_val) or std_val == 0.0:
        return float("nan")
    return mean_val / std_val


def _normalize_methods(candidates: Sequence[str], *, default: str) -> tuple[str, ...]:
    valid = {"equal", "rank", "score"}
    out: list[str] = []
    for raw in candidates:
        method = str(raw).strip().lower()
        if method and method in valid and method not in out:
            out.append(method)
    default_method = str(default).strip().lower()
    if default_method in valid and default_method not in out:
        out.insert(0, default_method)
    if not out:
        out = [default_method if default_method in valid else "rank", "equal", "score"]
    return tuple(out)


def _normalize_holding_grid(candidates: Sequence[int], *, default: int) -> tuple[int, ...]:
    out: list[int] = []
    for raw in candidates:
        value = _to_int(raw)
        if value is not None and value > 0 and value not in out:
            out.append(value)
    if default > 0 and default not in out:
        out.insert(0, default)
    if not out:
        out = [max(1, default)]
    return tuple(sorted(set(out)))


def _normalize_cost_grid(candidates: Sequence[float], *, review_cost: float) -> tuple[float, ...]:
    out: list[float] = []
    for raw in candidates:
        value = _to_float(raw)
        if value is None or value < 0:
            continue
        rounded = round(value, 6)
        if rounded not in out:
            out.append(rounded)
    review = round(max(0.0, review_cost), 6)
    if review not in out:
        out.append(review)
    if 0.0 not in out:
        out.append(0.0)
    out.sort()
    return tuple(out)


def _resolve_rebalance_step(rebalance_frequency: str) -> tuple[int, str]:
    text = rebalance_frequency.strip().upper()
    if not text:
        return 1, "fallback_default"
    try:
        value = int(text)
    except ValueError:
        value = 0
    if value > 0:
        return value, "explicit_integer"

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
    if text in mapping:
        return mapping[text], "calendar_alias"
    return 1, "fallback_default"


def _rate_key(rate: float) -> str:
    return f"{rate:.4f}"


def _jsonable(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return _finite_or_none(value)
    return value


def _finite_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None


def _to_int(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if not isinstance(value, (int, float, str)):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if not isinstance(value, (int, float, str)):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _safe_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _to_text_list(value: object) -> list[str]:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if ";" in text:
            return [token.strip() for token in text.split(";") if token.strip()]
        if "," in text:
            return [token.strip() for token in text.split(",") if token.strip()]
        return [text]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        out: list[str] = []
        for item in value:
            item_text = _safe_str(item)
            if item_text is not None:
                out.append(item_text)
        return out
    coerced = _safe_str(value)
    return [coerced] if coerced is not None else []


def _append_unique(values: list[str], candidate: str) -> None:
    text = candidate.strip()
    if text and text not in values:
        values.append(text)


def _dedupe_text(values: Sequence[str]) -> list[str]:
    out: list[str] = []
    for raw in values:
        text = str(raw).strip()
        if text and text not in out:
            out.append(text)
    return out


def _coerce_mapping(raw: object) -> dict[str, object]:
    return dict(raw) if isinstance(raw, Mapping) else {}


def _format_value(value: object) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    number = _to_float(value)
    if number is not None:
        return f"{number:.6f}"
    text = _safe_str(value)
    return text if text is not None else "N/A"
