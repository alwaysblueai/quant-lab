"""Shared typed contracts for repeated Level 1/2 ``key_metrics`` subsets."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import TypedDict, cast


class CoreSignalEvidenceMetrics(TypedDict):
    n_dates_used: float | None
    n_dates: float | None
    mean_ic: float | None
    mean_rank_ic: float | None
    ic_ir: float | None
    ic_positive_rate: float | None
    rank_ic_positive_rate: float | None
    ic_valid_ratio: float | None
    rank_ic_valid_ratio: float | None
    valid_ratio_min: float | None
    mean_long_short_return: float | None
    long_short_ir: float | None
    mean_long_short_turnover: float | None
    long_short_return_per_turnover: float | None
    subperiod_ic_positive_share: float | None
    subperiod_long_short_positive_share: float | None
    subperiod_positive_share_min: float | None
    coverage_mean: float | None
    coverage_min: float | None
    eval_coverage_ratio_mean: float | None
    eval_coverage_ratio_min: float | None


class UncertaintyEvidenceMetrics(TypedDict):
    mean_ic_ci_lower: float | None
    mean_ic_ci_upper: float | None
    mean_rank_ic_ci_lower: float | None
    mean_rank_ic_ci_upper: float | None
    mean_long_short_return_ci_lower: float | None
    mean_long_short_return_ci_upper: float | None
    uncertainty_method: str | None
    uncertainty_confidence_level: float | None
    uncertainty_bootstrap_resamples: int | None
    uncertainty_bootstrap_block_length: int | None
    uncertainty_flags: tuple[str, ...]
    uncertainty_supportive_ci_count: int
    uncertainty_overlap_zero_count: int


class RollingStabilityMetrics(TypedDict):
    rolling_window_size: int | None
    rolling_ic_positive_share: float | None
    rolling_rank_ic_positive_share: float | None
    rolling_long_short_positive_share: float | None
    rolling_ic_min_mean: float | None
    rolling_rank_ic_min_mean: float | None
    rolling_long_short_min_mean: float | None
    rolling_positive_share_min: float | None
    rolling_worst_mean_min: float | None
    rolling_instability_flags: tuple[str, ...]


class NeutralizationComparisonSlice(TypedDict):
    raw: dict[str, object]
    neutralized: dict[str, object]
    delta: dict[str, object]
    interpretation_flags: tuple[str, ...]
    interpretation_reasons: tuple[str, ...]


class NeutralizationComparisonMetrics(TypedDict):
    neutralization_comparison: NeutralizationComparisonSlice
    neutralization_flags: tuple[str, ...]
    neutralization_reasons: tuple[str, ...]
    neutralization_mean_corr_reduction: float | None
    neutralization_raw_mean_ic: float | None
    neutralization_raw_mean_rank_ic: float | None
    neutralization_raw_mean_long_short_return: float | None
    neutralization_raw_ic_ir: float | None
    neutralization_mean_ic_delta: float | None
    neutralization_mean_rank_ic_delta: float | None
    neutralization_mean_long_short_return_delta: float | None
    neutralization_ic_ir_delta: float | None
    neutralization_valid_ratio_min_delta: float | None
    neutralization_eval_coverage_ratio_mean_delta: float | None
    neutralization_uncertainty_overlap_zero_count_delta: float | None
    neutralization_rolling_positive_share_min_delta: float | None
    neutralization_rolling_worst_mean_min_delta: float | None


class PromotionGateMetrics(TypedDict):
    factor_verdict: str
    campaign_triage: str
    core: CoreSignalEvidenceMetrics
    uncertainty: UncertaintyEvidenceMetrics
    rolling: RollingStabilityMetrics
    neutralization: NeutralizationComparisonMetrics


class PortfolioValidationMetrics(TypedDict):
    case_name: str | None
    rebalance_frequency: str | None
    promotion_decision: str | None
    promotion_reasons: tuple[str, ...]
    promotion_blockers: tuple[str, ...]
    portfolio_validation_status: str | None
    portfolio_validation_recommendation: str | None
    portfolio_validation_major_risks: tuple[str, ...]
    portfolio_validation_robustness_label: str | None
    portfolio_validation_support_reasons: tuple[str, ...]
    portfolio_validation_fragility_reasons: tuple[str, ...]
    portfolio_validation_scenario_sensitivity_notes: tuple[str, ...]
    portfolio_validation_benchmark_support_note: str | None
    portfolio_validation_cost_sensitivity_note: str | None
    portfolio_validation_concentration_turnover_note: str | None
    portfolio_validation_benchmark_relative_status: str | None
    portfolio_validation_benchmark_relative_assessment: str | None
    portfolio_validation_benchmark_excess_return: float | None
    portfolio_validation_benchmark_tracking_error: float | None
    benchmark_name: str | None
    benchmark_excess_return: float | None
    benchmark_active_return: float | None
    benchmark_relative_return: float | None
    benchmark_long_short_excess_return: float | None
    benchmark_information_ratio: float | None
    benchmark_tracking_error: float | None
    benchmark_relative_max_drawdown: float | None
    portfolio_max_drawdown: float | None
    benchmark_max_drawdown: float | None
    mean_eval_assets_per_date: float | None
    n_quantiles: float | None


class Level12TransitionSummaryMetrics(TypedDict):
    level1_status: str
    level2_status: str
    transition_label: str
    transition_interpretation: str
    key_transition_reasons: tuple[str, ...]
    confirmation_vs_degradation_note: str


class CampaignProfileSummaryMetrics(TypedDict):
    factor_verdict: str | None
    factor_verdict_reasons: tuple[str, ...]
    campaign_triage: str | None
    campaign_triage_reasons: tuple[str, ...]
    promotion_decision: str | None
    promotion_reasons: tuple[str, ...]
    promotion_blockers: tuple[str, ...]
    portfolio_validation_status: str | None
    portfolio_validation_recommendation: str | None
    portfolio_validation_major_risks: tuple[str, ...]
    portfolio_validation_robustness_label: str | None
    portfolio_validation_support_reasons: tuple[str, ...]
    portfolio_validation_fragility_reasons: tuple[str, ...]
    level12_transition_summary: Level12TransitionSummaryMetrics
    level12_transition_label: str
    level12_transition_interpretation: str
    level12_transition_reasons: tuple[str, ...]
    level12_transition_confirmation_note: str


class CampaignRankingMetrics(TypedDict):
    ic_ir: float | None
    mean_long_short_return: float | None
    rolling_ic_positive_share: float | None
    rolling_rank_ic_positive_share: float | None
    rolling_long_short_positive_share: float | None


class Level12TransitionDistributionMetrics(TypedDict):
    n_cases: int
    n_cases_with_transition_label: int
    n_cases_missing_transition_label: int
    counts_by_transition_label: dict[str, int]
    proportions_by_transition_label: dict[str, float]
    representative_cases_by_transition_label: dict[str, list[str]]
    representative_case_names_by_transition_label: dict[str, list[str]]
    artifact_pointers_by_transition_label: dict[str, list[str]]
    reason_rollup_by_transition_label: dict[str, dict[str, object]]
    minimum_support_thresholds: dict[str, int]
    support_count: int
    minimum_required_support: int
    support_level: str
    is_sparse: bool
    minimum_support_met: bool
    support_note: str
    confidence_note: str
    interpretation: str


class SupportAnnotation(TypedDict):
    support_level: str
    is_sparse: bool
    minimum_support_met: bool
    support_note: str
    confidence_note: str


CAMPAIGN_PROFILE_COMPARISON_FIELDS: tuple[str, ...] = (
    "factor_verdict",
    "campaign_triage",
    "promotion_decision",
    "portfolio_validation_recommendation",
)

PROFILE_AWARE_LEVEL12_OBSERVED_DIFFERENCE_FIELDS: tuple[str, ...] = (
    "factor_verdict",
    "campaign_triage",
    "promotion_decision",
    "level12_transition_label",
    "portfolio_validation_status",
    "portfolio_validation_recommendation",
)

LEVEL12_TRANSITION_TAXONOMY: tuple[str, ...] = (
    "Confirmed at portfolio level",
    "Weakened at portfolio level",
    "Fragile after promotion",
    "Improved at portfolio level",
    "Inconclusive transition",
)

_TRANSITION_REASON_ROLLUP_NORMALIZATION_METHOD = "explicit_rule_v1"
_TRANSITION_REASON_EXACT_NORMALIZATION_MAP: dict[str, str] = {
    "campaign triage: fragile / monitor": "campaign triage: Fragile / monitor",
    "promotion decision: promote to level 2": "promotion decision: Promote to Level 2",
    "promotion decision: blocked from level 2": "promotion decision: Blocked from Level 2",
    "promotion decision: hold for refinement": "promotion decision: Hold for refinement",
    "portfolio recommendation: credible at portfolio level": (
        "portfolio recommendation: Credible at portfolio level"
    ),
    "portfolio recommendation: needs portfolio refinement": (
        "portfolio recommendation: Needs portfolio refinement"
    ),
    "portfolio recommendation: not evaluated (not promoted)": (
        "portfolio recommendation: Not evaluated (not promoted)"
    ),
}
_TRANSITION_REASON_PREFIX_STRIP_RULES: tuple[str, ...] = (
    "promotion reason:",
    "fragility:",
)
LEVEL12_TRANSITION_SUPPORT_THRESHOLDS: dict[str, int] = {
    "minimum_cases_with_transition_label": 3,
    "minimum_cases_per_transition_label": 2,
    "minimum_cases_with_reasons_per_transition_label": 2,
    "minimum_reason_bucket_count_for_dominance": 2,
}


def normalize_transition_reason_for_rollup(reason: str) -> str:
    """Normalize transition reasons for campaign-level rollups only."""
    token = " ".join(str(reason).strip().split())
    if not token:
        return ""
    canonical = _TRANSITION_REASON_EXACT_NORMALIZATION_MAP.get(token.lower())
    if canonical is not None:
        return canonical

    lowered = token.lower()
    for prefix in _TRANSITION_REASON_PREFIX_STRIP_RULES:
        if lowered.startswith(prefix):
            suffix = token[len(prefix) :].strip()
            if suffix:
                token = suffix
            break

    canonical = _TRANSITION_REASON_EXACT_NORMALIZATION_MAP.get(token.lower())
    if canonical is not None:
        return canonical
    return token


def project_level12_transition_distribution(
    case_rows: Sequence[Mapping[str, object]],
    *,
    case_name_field: str = "case_name",
    transition_label_field: str = "level12_transition_label",
    transition_reasons_field: str = "level12_transition_reasons",
    artifact_pointer_field: str = "artifact_pointer",
    representative_case_limit: int = 2,
    top_reason_limit: int = 3,
    representative_reason_limit: int = 3,
    supporting_case_limit: int = 4,
    artifact_pointer_limit: int = 3,
) -> Level12TransitionDistributionMetrics:
    minimum_cases_with_transition_label = max(
        1,
        int(LEVEL12_TRANSITION_SUPPORT_THRESHOLDS["minimum_cases_with_transition_label"]),
    )
    minimum_cases_per_transition_label = max(
        1,
        int(LEVEL12_TRANSITION_SUPPORT_THRESHOLDS["minimum_cases_per_transition_label"]),
    )
    minimum_cases_with_reasons_per_transition_label = max(
        1,
        int(LEVEL12_TRANSITION_SUPPORT_THRESHOLDS["minimum_cases_with_reasons_per_transition_label"]),
    )
    minimum_reason_bucket_count_for_dominance = max(
        1,
        int(LEVEL12_TRANSITION_SUPPORT_THRESHOLDS["minimum_reason_bucket_count_for_dominance"]),
    )

    labels = LEVEL12_TRANSITION_TAXONOMY
    counts: dict[str, int] = {label: 0 for label in labels}
    representatives: dict[str, list[str]] = {label: [] for label in labels}
    artifact_pointers_by_transition_label: dict[str, list[str]] = {
        label: [] for label in labels
    }
    supporting_case_names_by_transition_label: dict[str, list[str]] = {
        label: [] for label in labels
    }
    supporting_artifact_pointers_by_transition_label: dict[str, list[str]] = {
        label: [] for label in labels
    }
    raw_reason_case_counts: dict[str, dict[str, int]] = {label: {} for label in labels}
    reason_case_counts: dict[str, dict[str, int]] = {label: {} for label in labels}
    reason_supporting_case_names: dict[str, dict[str, list[str]]] = {
        label: {} for label in labels
    }
    reason_artifact_pointers: dict[str, dict[str, list[str]]] = {
        label: {} for label in labels
    }
    n_cases_with_any_reason = {label: 0 for label in labels}

    max_representatives = max(0, representative_case_limit)
    max_top_reasons = max(0, top_reason_limit)
    max_representative_reasons = max(0, representative_reason_limit)
    max_supporting_cases = max(0, supporting_case_limit)
    max_artifact_pointers = max(0, artifact_pointer_limit)
    missing_labels = 0
    for idx, row in enumerate(case_rows):
        case_name = _to_text(row.get(case_name_field)) or f"case_{idx + 1}"
        transition_label = _to_text(row.get(transition_label_field))
        artifact_pointer = _to_text(row.get(artifact_pointer_field))
        if transition_label not in counts:
            missing_labels += 1
            continue
        counts[transition_label] += 1
        _append_unique_text(
            supporting_case_names_by_transition_label[transition_label],
            case_name,
            max_items=max_supporting_cases,
        )
        rep_bucket = representatives[transition_label]
        if (
            max_representatives > 0
            and case_name not in rep_bucket
            and len(rep_bucket) < max_representatives
        ):
            rep_bucket.append(case_name)
            _append_unique_text(
                artifact_pointers_by_transition_label[transition_label],
                artifact_pointer,
                max_items=max_artifact_pointers,
            )
        raw_reasons = _dedupe_texts(_to_text_tuple(row.get(transition_reasons_field)), max_items=32)
        if raw_reasons:
            n_cases_with_any_reason[transition_label] += 1
            _append_unique_text(
                supporting_artifact_pointers_by_transition_label[transition_label],
                artifact_pointer,
                max_items=max_artifact_pointers,
            )
            case_raw_reason_counts = raw_reason_case_counts[transition_label]
            normalized_tokens: list[str] = []
            for reason in raw_reasons:
                case_raw_reason_counts[reason] = case_raw_reason_counts.get(reason, 0) + 1
                normalized = normalize_transition_reason_for_rollup(reason)
                if normalized:
                    normalized_tokens.append(normalized)
            normalized_reasons = _dedupe_texts(normalized_tokens, max_items=32)
            case_reason_counts = reason_case_counts[transition_label]
            for reason in normalized_reasons:
                case_reason_counts[reason] = case_reason_counts.get(reason, 0) + 1
                reason_case_bucket = reason_supporting_case_names[transition_label].setdefault(
                    reason,
                    [],
                )
                _append_unique_text(
                    reason_case_bucket,
                    case_name,
                    max_items=max_supporting_cases,
                )
                reason_artifact_bucket = reason_artifact_pointers[transition_label].setdefault(
                    reason,
                    [],
                )
                _append_unique_text(
                    reason_artifact_bucket,
                    artifact_pointer,
                    max_items=max_artifact_pointers,
                )

    n_cases = len(case_rows)
    n_observed = n_cases - missing_labels
    proportions = {
        label: (counts[label] / n_cases if n_cases > 0 else 0.0)
        for label in labels
    }
    reason_rollup_by_transition_label: dict[str, dict[str, object]] = {}
    for label in labels:
        label_case_count = counts[label]
        raw_counts = raw_reason_case_counts[label]
        reason_counts = reason_case_counts[label]
        sorted_reason_counts = sorted(
            reason_counts.items(),
            key=lambda row: (-row[1], row[0].lower()),
        )
        top_reason_items = sorted_reason_counts[:max_top_reasons]
        top_reasons: list[dict[str, object]] = []
        for reason, count in top_reason_items:
            reason_case_names = reason_supporting_case_names[label].get(reason, [])
            reason_artifact_hints = reason_artifact_pointers[label].get(reason, [])
            reason_support = _support_annotation(
                support_count=count,
                minimum_required_support=minimum_reason_bucket_count_for_dominance,
                sparse_note="tentative due to low support",
                tentative_note="tentative due to low support",
                supported_note="reason bucket is well supported",
            )
            top_reasons.append(
                {
                    "reason": reason,
                    "count": count,
                    "proportion_of_label_cases": (
                        count / label_case_count if label_case_count > 0 else 0.0
                    ),
                    "supporting_case_names": list(reason_case_names),
                    "artifact_pointer_hints": list(reason_artifact_hints),
                    **reason_support,
                }
            )
        dominant_reasons = [
            row for row in top_reasons if bool(row.get("minimum_support_met"))
        ]
        representative_reasons = [
            reason
            for reason, _ in sorted_reason_counts[:max_representative_reasons]
        ]
        label_reason_support = _support_annotation(
            support_count=min(label_case_count, n_cases_with_any_reason[label]),
            minimum_required_support=max(
                minimum_cases_per_transition_label,
                minimum_cases_with_reasons_per_transition_label,
            ),
            sparse_note="sparse transition evidence",
            tentative_note="tentative due to low support",
            supported_note="reason evidence is well supported",
        )
        reason_rollup_by_transition_label[label] = {
            "n_cases_with_label": label_case_count,
            "n_cases_with_any_reason": n_cases_with_any_reason[label],
            "n_unique_reasons_observed": len(reason_counts),
            "n_unique_raw_reasons_observed": len(raw_counts),
            "n_reasons_collapsed_by_normalization": max(
                0,
                len(raw_counts) - len(reason_counts),
            ),
            "reason_normalization": {
                "method": _TRANSITION_REASON_ROLLUP_NORMALIZATION_METHOD,
                "applies_to_rollup_only": True,
                "raw_reasons_preserved_at_case_level": True,
            },
            "minimum_support_thresholds": {
                "minimum_cases_with_label": minimum_cases_per_transition_label,
                "minimum_cases_with_any_reason": minimum_cases_with_reasons_per_transition_label,
                "minimum_reason_bucket_count_for_dominance": (
                    minimum_reason_bucket_count_for_dominance
                ),
            },
            "support_count": min(label_case_count, n_cases_with_any_reason[label]),
            "minimum_required_support": max(
                minimum_cases_per_transition_label,
                minimum_cases_with_reasons_per_transition_label,
            ),
            **label_reason_support,
            "top_reasons": top_reasons,
            "dominant_reasons": dominant_reasons,
            "n_top_reasons_below_minimum_support": max(
                0,
                len(top_reasons) - len(dominant_reasons),
            ),
            "representative_reasons": representative_reasons,
            "representative_case_names": list(representatives[label]),
            "supporting_case_names": list(supporting_case_names_by_transition_label[label]),
            "artifact_pointer_hints": list(
                supporting_artifact_pointers_by_transition_label[label]
            ),
        }
    distribution_support = _support_annotation(
        support_count=n_observed,
        minimum_required_support=minimum_cases_with_transition_label,
        sparse_note="sparse transition evidence",
        tentative_note="tentative due to low support",
        supported_note="transition evidence is well supported",
    )
    interpretation = _level12_transition_distribution_interpretation(
        counts=counts,
        n_cases=n_cases,
        n_observed=n_observed,
        n_missing=missing_labels,
    )
    return {
        "n_cases": n_cases,
        "n_cases_with_transition_label": n_observed,
        "n_cases_missing_transition_label": missing_labels,
        "counts_by_transition_label": counts,
        "proportions_by_transition_label": proportions,
        "representative_cases_by_transition_label": representatives,
        "representative_case_names_by_transition_label": {
            label: list(case_names)
            for label, case_names in representatives.items()
        },
        "artifact_pointers_by_transition_label": {
            label: list(artifact_paths)
            for label, artifact_paths in artifact_pointers_by_transition_label.items()
        },
        "reason_rollup_by_transition_label": reason_rollup_by_transition_label,
        "minimum_support_thresholds": dict(LEVEL12_TRANSITION_SUPPORT_THRESHOLDS),
        "support_count": n_observed,
        "minimum_required_support": minimum_cases_with_transition_label,
        "support_level": distribution_support["support_level"],
        "is_sparse": distribution_support["is_sparse"],
        "minimum_support_met": distribution_support["minimum_support_met"],
        "support_note": distribution_support["support_note"],
        "confidence_note": distribution_support["confidence_note"],
        "interpretation": interpretation,
    }


def project_core_signal_evidence_metrics(
    metrics: Mapping[str, object],
) -> CoreSignalEvidenceMetrics:
    ic_valid_ratio = _to_float(metrics.get("ic_valid_ratio"))
    rank_ic_valid_ratio = _to_float(metrics.get("rank_ic_valid_ratio"))
    subperiod_ic_positive_share = _to_float(metrics.get("subperiod_ic_positive_share"))
    subperiod_long_short_positive_share = _to_float(
        metrics.get("subperiod_long_short_positive_share")
    )
    return {
        "n_dates_used": _to_float(metrics.get("n_dates_used")),
        "n_dates": _to_float(metrics.get("n_dates")),
        "mean_ic": _to_float(metrics.get("mean_ic")),
        "mean_rank_ic": _to_float(metrics.get("mean_rank_ic")),
        "ic_ir": _to_float(metrics.get("ic_ir")),
        "ic_positive_rate": _to_float(metrics.get("ic_positive_rate")),
        "rank_ic_positive_rate": _to_float(metrics.get("rank_ic_positive_rate")),
        "ic_valid_ratio": ic_valid_ratio,
        "rank_ic_valid_ratio": rank_ic_valid_ratio,
        "valid_ratio_min": _min_or_none((ic_valid_ratio, rank_ic_valid_ratio)),
        "mean_long_short_return": _to_float(metrics.get("mean_long_short_return")),
        "long_short_ir": _to_float(metrics.get("long_short_ir")),
        "mean_long_short_turnover": _to_float(metrics.get("mean_long_short_turnover")),
        "long_short_return_per_turnover": _to_float(
            metrics.get("long_short_return_per_turnover")
        ),
        "subperiod_ic_positive_share": subperiod_ic_positive_share,
        "subperiod_long_short_positive_share": subperiod_long_short_positive_share,
        "subperiod_positive_share_min": _min_or_none(
            (subperiod_ic_positive_share, subperiod_long_short_positive_share)
        ),
        "coverage_mean": _to_float(metrics.get("coverage_mean")),
        "coverage_min": _to_float(metrics.get("coverage_min")),
        "eval_coverage_ratio_mean": _to_float(metrics.get("eval_coverage_ratio_mean")),
        "eval_coverage_ratio_min": _to_float(metrics.get("eval_coverage_ratio_min")),
    }


def project_uncertainty_evidence_metrics(
    metrics: Mapping[str, object],
) -> UncertaintyEvidenceMetrics:
    mean_ic_ci_lower = _to_float(metrics.get("mean_ic_ci_lower"))
    mean_ic_ci_upper = _to_float(metrics.get("mean_ic_ci_upper"))
    mean_rank_ic_ci_lower = _to_float(metrics.get("mean_rank_ic_ci_lower"))
    mean_rank_ic_ci_upper = _to_float(metrics.get("mean_rank_ic_ci_upper"))
    mean_long_short_return_ci_lower = _to_float(metrics.get("mean_long_short_return_ci_lower"))
    mean_long_short_return_ci_upper = _to_float(metrics.get("mean_long_short_return_ci_upper"))
    bounds = (
        (mean_ic_ci_lower, mean_ic_ci_upper),
        (mean_rank_ic_ci_lower, mean_rank_ic_ci_upper),
        (mean_long_short_return_ci_lower, mean_long_short_return_ci_upper),
    )
    return {
        "mean_ic_ci_lower": mean_ic_ci_lower,
        "mean_ic_ci_upper": mean_ic_ci_upper,
        "mean_rank_ic_ci_lower": mean_rank_ic_ci_lower,
        "mean_rank_ic_ci_upper": mean_rank_ic_ci_upper,
        "mean_long_short_return_ci_lower": mean_long_short_return_ci_lower,
        "mean_long_short_return_ci_upper": mean_long_short_return_ci_upper,
        "uncertainty_method": _to_text(metrics.get("uncertainty_method")),
        "uncertainty_confidence_level": _to_float(metrics.get("uncertainty_confidence_level")),
        "uncertainty_bootstrap_resamples": _to_int(
            metrics.get("uncertainty_bootstrap_resamples")
        ),
        "uncertainty_bootstrap_block_length": _to_int(
            metrics.get("uncertainty_bootstrap_block_length")
        ),
        "uncertainty_flags": _to_text_tuple(metrics.get("uncertainty_flags")),
        "uncertainty_supportive_ci_count": sum(
            lower is not None and lower > 0.0
            for lower in (
                mean_ic_ci_lower,
                mean_rank_ic_ci_lower,
                mean_long_short_return_ci_lower,
            )
        ),
        "uncertainty_overlap_zero_count": sum(
            lower is not None and upper is not None and lower <= 0.0 <= upper
            for lower, upper in bounds
        ),
    }


def project_rolling_stability_metrics(
    metrics: Mapping[str, object],
) -> RollingStabilityMetrics:
    rolling_ic_positive_share = _to_float(metrics.get("rolling_ic_positive_share"))
    rolling_rank_ic_positive_share = _to_float(metrics.get("rolling_rank_ic_positive_share"))
    rolling_long_short_positive_share = _to_float(metrics.get("rolling_long_short_positive_share"))
    rolling_ic_min_mean = _to_float(metrics.get("rolling_ic_min_mean"))
    rolling_rank_ic_min_mean = _to_float(metrics.get("rolling_rank_ic_min_mean"))
    rolling_long_short_min_mean = _to_float(metrics.get("rolling_long_short_min_mean"))
    return {
        "rolling_window_size": _to_int(metrics.get("rolling_window_size")),
        "rolling_ic_positive_share": rolling_ic_positive_share,
        "rolling_rank_ic_positive_share": rolling_rank_ic_positive_share,
        "rolling_long_short_positive_share": rolling_long_short_positive_share,
        "rolling_ic_min_mean": rolling_ic_min_mean,
        "rolling_rank_ic_min_mean": rolling_rank_ic_min_mean,
        "rolling_long_short_min_mean": rolling_long_short_min_mean,
        "rolling_positive_share_min": _min_or_none(
            (
                rolling_ic_positive_share,
                rolling_rank_ic_positive_share,
                rolling_long_short_positive_share,
            )
        ),
        "rolling_worst_mean_min": _min_or_none(
            (
                rolling_ic_min_mean,
                rolling_rank_ic_min_mean,
                rolling_long_short_min_mean,
            )
        ),
        "rolling_instability_flags": _to_text_tuple(metrics.get("rolling_instability_flags")),
    }


def project_neutralization_comparison_metrics(
    metrics: Mapping[str, object],
) -> NeutralizationComparisonMetrics:
    comparison = _coerce_mapping(metrics.get("neutralization_comparison"))
    nested_flags = _to_text_tuple(comparison.get("interpretation_flags"))
    top_level_flags = _to_text_tuple(metrics.get("neutralization_comparison_flags"))
    nested_reasons = _to_text_tuple(comparison.get("interpretation_reasons"))
    top_level_reasons = _to_text_tuple(metrics.get("neutralization_comparison_reasons"))
    return {
        "neutralization_comparison": {
            "raw": _coerce_mapping(comparison.get("raw")),
            "neutralized": _coerce_mapping(comparison.get("neutralized")),
            "delta": _coerce_mapping(comparison.get("delta")),
            "interpretation_flags": nested_flags,
            "interpretation_reasons": nested_reasons,
        },
        "neutralization_flags": nested_flags or top_level_flags,
        "neutralization_reasons": nested_reasons or top_level_reasons,
        "neutralization_mean_corr_reduction": _to_float(
            metrics.get("neutralization_mean_corr_reduction")
        ),
        "neutralization_raw_mean_ic": _to_float(metrics.get("neutralization_raw_mean_ic")),
        "neutralization_raw_mean_rank_ic": _to_float(
            metrics.get("neutralization_raw_mean_rank_ic")
        ),
        "neutralization_raw_mean_long_short_return": _to_float(
            metrics.get("neutralization_raw_mean_long_short_return")
        ),
        "neutralization_raw_ic_ir": _to_float(metrics.get("neutralization_raw_ic_ir")),
        "neutralization_mean_ic_delta": _to_float(metrics.get("neutralization_mean_ic_delta")),
        "neutralization_mean_rank_ic_delta": _to_float(
            metrics.get("neutralization_mean_rank_ic_delta")
        ),
        "neutralization_mean_long_short_return_delta": _to_float(
            metrics.get("neutralization_mean_long_short_return_delta")
        ),
        "neutralization_ic_ir_delta": _to_float(metrics.get("neutralization_ic_ir_delta")),
        "neutralization_valid_ratio_min_delta": _to_float(
            metrics.get("neutralization_valid_ratio_min_delta")
        ),
        "neutralization_eval_coverage_ratio_mean_delta": _to_float(
            metrics.get("neutralization_eval_coverage_ratio_mean_delta")
        ),
        "neutralization_uncertainty_overlap_zero_count_delta": _to_float(
            metrics.get("neutralization_uncertainty_overlap_zero_count_delta")
        ),
        "neutralization_rolling_positive_share_min_delta": _to_float(
            metrics.get("neutralization_rolling_positive_share_min_delta")
        ),
        "neutralization_rolling_worst_mean_min_delta": _to_float(
            metrics.get("neutralization_rolling_worst_mean_min_delta")
        ),
    }


def project_promotion_gate_metrics(
    metrics: Mapping[str, object],
) -> PromotionGateMetrics:
    return {
        "factor_verdict": _to_text(metrics.get("factor_verdict")) or "",
        "campaign_triage": _to_text(metrics.get("campaign_triage")) or "",
        "core": project_core_signal_evidence_metrics(metrics),
        "uncertainty": project_uncertainty_evidence_metrics(metrics),
        "rolling": project_rolling_stability_metrics(metrics),
        "neutralization": project_neutralization_comparison_metrics(metrics),
    }


def project_portfolio_validation_metrics(
    metrics: Mapping[str, object],
) -> PortfolioValidationMetrics:
    return {
        "case_name": _to_text(metrics.get("case_name")),
        "rebalance_frequency": _to_text(metrics.get("rebalance_frequency")),
        "promotion_decision": _to_text(metrics.get("promotion_decision")),
        "promotion_reasons": _to_text_tuple(metrics.get("promotion_reasons")),
        "promotion_blockers": _to_text_tuple(metrics.get("promotion_blockers")),
        "portfolio_validation_status": _to_text(metrics.get("portfolio_validation_status")),
        "portfolio_validation_recommendation": _to_text(
            metrics.get("portfolio_validation_recommendation")
        ),
        "portfolio_validation_major_risks": _to_text_tuple(
            metrics.get("portfolio_validation_major_risks")
        ),
        "portfolio_validation_robustness_label": _to_text(
            metrics.get("portfolio_validation_robustness_label")
        ),
        "portfolio_validation_support_reasons": _to_text_tuple(
            metrics.get("portfolio_validation_support_reasons")
        ),
        "portfolio_validation_fragility_reasons": _to_text_tuple(
            metrics.get("portfolio_validation_fragility_reasons")
        ),
        "portfolio_validation_scenario_sensitivity_notes": _to_text_tuple(
            metrics.get("portfolio_validation_scenario_sensitivity_notes")
        ),
        "portfolio_validation_benchmark_support_note": _to_text(
            metrics.get("portfolio_validation_benchmark_support_note")
        ),
        "portfolio_validation_cost_sensitivity_note": _to_text(
            metrics.get("portfolio_validation_cost_sensitivity_note")
        ),
        "portfolio_validation_concentration_turnover_note": _to_text(
            metrics.get("portfolio_validation_concentration_turnover_note")
        ),
        "portfolio_validation_benchmark_relative_status": _to_text(
            metrics.get("portfolio_validation_benchmark_relative_status")
        ),
        "portfolio_validation_benchmark_relative_assessment": _to_text(
            metrics.get("portfolio_validation_benchmark_relative_assessment")
        ),
        "portfolio_validation_benchmark_excess_return": _to_float(
            metrics.get("portfolio_validation_benchmark_excess_return")
        ),
        "portfolio_validation_benchmark_tracking_error": _to_float(
            metrics.get("portfolio_validation_benchmark_tracking_error")
        ),
        "benchmark_name": _to_text(metrics.get("benchmark_name")),
        "benchmark_excess_return": _to_float(metrics.get("benchmark_excess_return")),
        "benchmark_active_return": _to_float(metrics.get("benchmark_active_return")),
        "benchmark_relative_return": _to_float(metrics.get("benchmark_relative_return")),
        "benchmark_long_short_excess_return": _to_float(
            metrics.get("benchmark_long_short_excess_return")
        ),
        "benchmark_information_ratio": _to_float(metrics.get("benchmark_information_ratio")),
        "benchmark_tracking_error": _to_float(metrics.get("benchmark_tracking_error")),
        "benchmark_relative_max_drawdown": _to_float(
            metrics.get("benchmark_relative_max_drawdown")
        ),
        "portfolio_max_drawdown": _to_float(metrics.get("portfolio_max_drawdown")),
        "benchmark_max_drawdown": _to_float(metrics.get("benchmark_max_drawdown")),
        "mean_eval_assets_per_date": _to_float(metrics.get("mean_eval_assets_per_date")),
        "n_quantiles": _to_float(metrics.get("n_quantiles")),
    }


def project_campaign_profile_summary_metrics(
    metrics: Mapping[str, object],
) -> CampaignProfileSummaryMetrics:
    portfolio = project_portfolio_validation_metrics(metrics)
    transition = project_level12_transition_summary(metrics)
    return {
        "factor_verdict": _to_text(metrics.get("factor_verdict")),
        "factor_verdict_reasons": _to_text_tuple(metrics.get("factor_verdict_reasons")),
        "campaign_triage": _to_text(metrics.get("campaign_triage")),
        "campaign_triage_reasons": _to_text_tuple(metrics.get("campaign_triage_reasons")),
        "promotion_decision": portfolio["promotion_decision"],
        "promotion_reasons": portfolio["promotion_reasons"],
        "promotion_blockers": portfolio["promotion_blockers"],
        "portfolio_validation_status": portfolio["portfolio_validation_status"],
        "portfolio_validation_recommendation": portfolio[
            "portfolio_validation_recommendation"
        ],
        "portfolio_validation_major_risks": portfolio[
            "portfolio_validation_major_risks"
        ],
        "portfolio_validation_robustness_label": portfolio[
            "portfolio_validation_robustness_label"
        ],
        "portfolio_validation_support_reasons": portfolio[
            "portfolio_validation_support_reasons"
        ],
        "portfolio_validation_fragility_reasons": portfolio[
            "portfolio_validation_fragility_reasons"
        ],
        "level12_transition_summary": transition,
        "level12_transition_label": transition["transition_label"],
        "level12_transition_interpretation": transition["transition_interpretation"],
        "level12_transition_reasons": transition["key_transition_reasons"],
        "level12_transition_confirmation_note": transition[
            "confirmation_vs_degradation_note"
        ],
    }


def project_campaign_ranking_metrics(
    metrics: Mapping[str, object],
) -> CampaignRankingMetrics:
    core = project_core_signal_evidence_metrics(metrics)
    rolling = project_rolling_stability_metrics(metrics)
    return {
        "ic_ir": core["ic_ir"],
        "mean_long_short_return": core["mean_long_short_return"],
        "rolling_ic_positive_share": rolling["rolling_ic_positive_share"],
        "rolling_rank_ic_positive_share": rolling["rolling_rank_ic_positive_share"],
        "rolling_long_short_positive_share": rolling["rolling_long_short_positive_share"],
    }


def project_level12_transition_summary(
    metrics: Mapping[str, object],
) -> Level12TransitionSummaryMetrics:
    verdict = _to_text(metrics.get("factor_verdict")) or "N/A"
    triage = _to_text(metrics.get("campaign_triage")) or "N/A"
    promotion_decision = _to_text(metrics.get("promotion_decision")) or "N/A"
    recommendation = _to_text(metrics.get("portfolio_validation_recommendation")) or "N/A"
    robustness = _to_text(metrics.get("portfolio_validation_robustness_label")) or "N/A"

    verdict_l = verdict.lower()
    promotion_l = promotion_decision.lower()
    recommendation_l = recommendation.lower()
    robustness_l = robustness.lower()

    level1_strong = verdict_l == "strong candidate"
    level1_weak = verdict_l in {"weak / noisy", "fails basic robustness"}
    level1_mixed_or_fragile = verdict_l in {"promising but fragile", "mixed evidence"}
    promoted = promotion_l == "promote to level 2"
    level2_credible = recommendation_l == "credible at portfolio level"
    level2_needs_refinement = recommendation_l == "needs portfolio refinement"
    level2_not_evaluated = recommendation_l == "not evaluated (not promoted)"
    robustness_fragile = robustness_l == "fragile at portfolio level"

    if level2_credible and level1_strong and not robustness_fragile:
        transition_label = "Confirmed at portfolio level"
        transition_interpretation = (
            "Strong Level 1 evidence is confirmed by Level 2 portfolio validation."
        )
        confirmation_note = (
            "Portfolio-level evidence confirms rather than degrades Level 1 strength."
        )
    elif level2_credible and robustness_fragile:
        transition_label = "Fragile after promotion"
        transition_interpretation = (
            "The case is promoted and still credible, but portfolio fragility is explicit."
        )
        confirmation_note = (
            "Signal-level evidence survives, but confidence is degraded by portfolio fragility."
        )
    elif level2_credible and (level1_weak or level1_mixed_or_fragile):
        transition_label = "Improved at portfolio level"
        transition_interpretation = (
            "Level 2 construction evidence improves a borderline Level 1 case."
        )
        confirmation_note = (
            "Portfolio-level evidence strengthens confidence versus the Level 1 baseline."
        )
    elif promoted and (
        level2_needs_refinement
        or robustness_fragile
        or recommendation_l in {"n/a", "not available"}
    ):
        transition_label = "Weakened at portfolio level"
        transition_interpretation = (
            "The case was promoted, but portfolio-level evidence is weaker than Level 1."
        )
        confirmation_note = (
            "Confidence degrades after transition from signal evidence to portfolio evidence."
        )
    elif level2_not_evaluated or not promoted:
        transition_label = "Inconclusive transition"
        transition_interpretation = (
            "Transition cannot be concluded because portfolio-level evaluation is absent."
        )
        confirmation_note = (
            "No clear confirmation or degradation can be established yet."
        )
    else:
        transition_label = "Inconclusive transition"
        transition_interpretation = (
            "Transition evidence is mixed and does not support a clear directional conclusion."
        )
        confirmation_note = (
            "Confirmation-versus-degradation remains unresolved from current evidence."
        )

    level1_status = verdict
    level2_status = (
        recommendation
        if recommendation != "N/A"
        else promotion_decision
    )
    transition_reasons = _build_level12_transition_reasons(
        metrics,
        transition_label=transition_label,
        triage=triage,
        promotion_decision=promotion_decision,
        recommendation=recommendation,
    )
    return {
        "level1_status": level1_status,
        "level2_status": level2_status,
        "transition_label": transition_label,
        "transition_interpretation": transition_interpretation,
        "key_transition_reasons": transition_reasons,
        "confirmation_vs_degradation_note": confirmation_note,
    }


def _build_level12_transition_reasons(
    metrics: Mapping[str, object],
    *,
    transition_label: str,
    triage: str,
    promotion_decision: str,
    recommendation: str,
) -> tuple[str, ...]:
    reasons: list[str] = []

    if triage != "N/A":
        reasons.append(f"campaign triage: {triage}")
    if promotion_decision != "N/A":
        reasons.append(f"promotion decision: {promotion_decision}")
    if recommendation != "N/A":
        reasons.append(f"portfolio recommendation: {recommendation}")

    for token in _to_text_tuple(metrics.get("promotion_reasons"))[:2]:
        reasons.append(f"promotion reason: {token}")
    for token in _to_text_tuple(metrics.get("portfolio_validation_fragility_reasons"))[:2]:
        reasons.append(f"fragility: {token}")
    for token in _to_text_tuple(metrics.get("portfolio_validation_major_risks"))[:2]:
        reasons.append(f"portfolio risk: {token}")

    benchmark_note = _to_text(metrics.get("portfolio_validation_benchmark_support_note"))
    if benchmark_note is not None and "unavailable" not in benchmark_note.lower():
        reasons.append(f"benchmark-relative: {benchmark_note}")
    cost_note = _to_text(metrics.get("portfolio_validation_cost_sensitivity_note"))
    if cost_note is not None and "unavailable" not in cost_note.lower():
        reasons.append(f"cost sensitivity: {cost_note}")
    concentration_note = _to_text(
        metrics.get("portfolio_validation_concentration_turnover_note")
    )
    if concentration_note is not None and "unavailable" not in concentration_note.lower():
        reasons.append(f"concentration/turnover: {concentration_note}")

    if not reasons:
        reasons.append(f"transition classification: {transition_label}")
    return _dedupe_texts(reasons, max_items=6)


def _dedupe_texts(values: Sequence[str], *, max_items: int) -> tuple[str, ...]:
    deduped: list[str] = []
    seen: set[str] = set()
    for raw in values:
        token = raw.strip()
        if not token:
            continue
        key = token.lower()
        if key in seen:
            continue
        deduped.append(token)
        seen.add(key)
        if len(deduped) >= max_items:
            break
    return tuple(deduped)


def _append_unique_text(target: list[str], value: str | None, *, max_items: int) -> None:
    if max_items <= 0:
        return
    token = _to_text(value)
    if token is None:
        return
    lowered = token.lower()
    for existing in target:
        if existing.lower() == lowered:
            return
    if len(target) >= max_items:
        return
    target.append(token)


def _coerce_mapping(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        return {}
    return cast(dict[str, object], dict(value))


def _to_text(value: object) -> str | None:
    if value is None:
        return None
    token = str(value).strip()
    return token if token else None


def _to_text_tuple(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        token = value.strip()
        if not token:
            return ()
        delimiter = ";" if ";" in token else ","
        return tuple(part.strip() for part in token.split(delimiter) if part.strip())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        out = [str(item).strip() for item in value if str(item).strip()]
        return tuple(out)
    text_token = _to_text(value)
    return (text_token,) if text_token is not None else ()


def _to_int(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return int(value)
    token = str(value).strip()
    if not token:
        return None
    try:
        return int(float(token))
    except ValueError:
        return None


def _to_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        out = float(value)
        return out if math.isfinite(out) else None
    token = str(value).strip()
    if not token:
        return None
    try:
        out = float(token)
    except ValueError:
        return None
    return out if math.isfinite(out) else None


def _min_or_none(values: Sequence[float | None]) -> float | None:
    finite = [value for value in values if value is not None]
    return min(finite) if finite else None


def _support_annotation(
    *,
    support_count: int,
    minimum_required_support: int,
    sparse_note: str,
    tentative_note: str,
    supported_note: str,
) -> SupportAnnotation:
    safe_support = max(0, int(support_count))
    safe_minimum = max(1, int(minimum_required_support))
    minimum_support_met = safe_support >= safe_minimum
    tentative_threshold = max(safe_minimum + 1, safe_minimum * 2)
    if safe_support < safe_minimum:
        support_level = "sparse"
        note = sparse_note
    elif safe_support < tentative_threshold:
        support_level = "tentative"
        note = tentative_note
    else:
        support_level = "supported"
        note = supported_note
    return {
        "support_level": support_level,
        "is_sparse": not minimum_support_met,
        "minimum_support_met": minimum_support_met,
        "support_note": note,
        "confidence_note": note,
    }


def _level12_transition_distribution_interpretation(
    *,
    counts: Mapping[str, int],
    n_cases: int,
    n_observed: int,
    n_missing: int,
) -> str:
    if n_cases <= 0:
        return "No campaign cases are available for Level 1->Level 2 transition aggregation."
    if n_observed <= 0:
        return (
            "Transition distribution is unavailable because no case has a recognized "
            "Level 1->Level 2 transition label."
        )

    max_count = max(counts.values(), default=0)
    if max_count <= 0:
        return (
            "No recognized Level 1->Level 2 transition labels are present in "
            "campaign outputs."
        )

    dominant_labels = [label for label, count in counts.items() if count == max_count]
    if len(dominant_labels) == 1:
        dominant_text = dominant_labels[0]
        message = (
            f"Most common transition outcome is `{dominant_text}` "
            f"({max_count}/{n_cases} cases)."
        )
    else:
        dominant_text = "`, `".join(dominant_labels)
        message = (
            "Most common transition outcomes are "
            f"`{dominant_text}` ({max_count}/{n_cases} cases each)."
        )

    if n_missing > 0:
        message += f" {n_missing}/{n_cases} case(s) are missing transition labels."
    return message
