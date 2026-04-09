"""Campaign-level triage and ranking helpers for Level 1/2 research outputs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TypedDict

from alpha_lab.key_metrics_contracts import (
    PromotionGateMetrics,
    project_promotion_gate_metrics,
)
from alpha_lab.reporting.neutralization_comparison import (
    EXPOSURE_DRIVEN_FLAG,
    MATERIAL_REDUCTION_FLAG,
    MODERATE_WEAKENING_FLAG,
    PRESERVES_EVIDENCE_FLAG,
)
from alpha_lab.research_evaluation_config import (
    DEFAULT_RESEARCH_EVALUATION_CONFIG,
    CampaignTriageConfig,
)

CAMPAIGN_TRIAGE_TAXONOMY: tuple[str, ...] = (
    "Advance to Level 2",
    "Strong Level 1 candidate",
    "Needs refinement",
    "Fragile / monitor",
    "Drop for now",
)

_TRIAGE_PRIORITY: dict[str, int] = {
    "Advance to Level 2": 1,
    "Strong Level 1 candidate": 2,
    "Needs refinement": 3,
    "Fragile / monitor": 4,
    "Drop for now": 5,
}

CAMPAIGN_RANK_RULE = (
    "triage_priority asc, ic_ir desc, mean_long_short_return desc, "
    "rolling_positive_share_min desc, risk_count asc, support_count desc"
)


CampaignTriageThresholds = CampaignTriageConfig
DEFAULT_CAMPAIGN_TRIAGE_THRESHOLDS = DEFAULT_RESEARCH_EVALUATION_CONFIG.campaign_triage


class CampaignTriagePayload(TypedDict):
    campaign_triage: str
    campaign_triage_reasons: list[str]
    campaign_triage_priority: int
    campaign_rank_primary_metric_name: str
    campaign_rank_primary_metric: float | None
    campaign_rank_secondary_metric_name: str
    campaign_rank_secondary_metric: float | None
    campaign_rank_stability_metric_name: str
    campaign_rank_stability_metric: float | None
    campaign_rank_support_count: int
    campaign_rank_risk_count: int
    campaign_rank_rule: str


@dataclass(frozen=True)
class CampaignTriage:
    """Compact, auditable campaign triage decision."""

    label: str
    reasons: tuple[str, ...]
    triage_priority: int
    rank_primary_metric: float | None
    rank_secondary_metric: float | None
    rank_stability_metric: float | None
    rank_support_count: int
    rank_risk_count: int

    def to_dict(self) -> CampaignTriagePayload:
        return {
            "campaign_triage": self.label,
            "campaign_triage_reasons": list(self.reasons),
            "campaign_triage_priority": self.triage_priority,
            "campaign_rank_primary_metric_name": "ic_ir",
            "campaign_rank_primary_metric": self.rank_primary_metric,
            "campaign_rank_secondary_metric_name": "mean_long_short_return",
            "campaign_rank_secondary_metric": self.rank_secondary_metric,
            "campaign_rank_stability_metric_name": "rolling_positive_share_min",
            "campaign_rank_stability_metric": self.rank_stability_metric,
            "campaign_rank_support_count": self.rank_support_count,
            "campaign_rank_risk_count": self.rank_risk_count,
            "campaign_rank_rule": CAMPAIGN_RANK_RULE,
        }


def build_campaign_triage(
    metrics: Mapping[str, object],
    *,
    status: str = "success",
    thresholds: CampaignTriageThresholds = DEFAULT_CAMPAIGN_TRIAGE_THRESHOLDS,
) -> CampaignTriage:
    """Classify one case into a campaign-level triage bucket with reasons."""

    gate_metrics = project_promotion_gate_metrics(metrics)
    core_metrics = gate_metrics["core"]
    uncertainty_metrics = gate_metrics["uncertainty"]
    rolling_metrics = gate_metrics["rolling"]
    neutralization_metrics = gate_metrics["neutralization"]

    if status.strip().lower() != "success":
        return _build_decision(
            label="Drop for now",
            reasons=("case did not complete successfully",),
            metrics=metrics,
            gate_metrics=gate_metrics,
            positives=(),
            concerns=("case did not complete successfully",),
        )

    verdict = gate_metrics["factor_verdict"].lower()
    uncertainty_flags = uncertainty_metrics["uncertainty_flags"]
    rolling_instability_flags = rolling_metrics["rolling_instability_flags"]
    neutralization_flags = neutralization_metrics["neutralization_flags"]

    coverage_mean = core_metrics["coverage_mean"]
    if coverage_mean is None:
        coverage_mean = core_metrics["eval_coverage_ratio_mean"]
    coverage_min = core_metrics["coverage_min"]
    if coverage_min is None:
        coverage_min = core_metrics["eval_coverage_ratio_min"]
    valid_ratio_min = core_metrics["valid_ratio_min"]
    ret_per_turnover = core_metrics["long_short_return_per_turnover"]
    turnover = core_metrics["mean_long_short_turnover"]

    subperiod_min = core_metrics["subperiod_positive_share_min"]
    rolling_share_min = rolling_metrics["rolling_positive_share_min"]
    rolling_worst_mean = rolling_metrics["rolling_worst_mean_min"]

    supportive_ci_count = uncertainty_metrics["uncertainty_supportive_ci_count"]
    uncertainty_overlap_count = uncertainty_metrics["uncertainty_overlap_zero_count"]
    uncertainty_supportive = supportive_ci_count >= thresholds.supportive_ci_min_count
    uncertainty_fragile = bool(
        uncertainty_overlap_count >= thresholds.uncertainty_overlap_fragile_min_count
        or any(flag.endswith("_ci_unavailable") for flag in uncertainty_flags)
        or any(flag.endswith("_ci_wide") for flag in uncertainty_flags)
    )

    has_strong_verdict = verdict == "strong candidate"
    has_fragile_verdict = verdict == "promising but fragile"
    has_mixed_verdict = verdict == "mixed evidence"
    has_weak_verdict = verdict in {"weak / noisy", "fails basic robustness"}

    coverage_too_thin = bool(
        (
            coverage_mean is not None
            and coverage_mean < thresholds.min_coverage_mean_fail
        )
        or (
            coverage_min is not None
            and coverage_min < thresholds.min_coverage_min_fail
        )
        or (
            valid_ratio_min is not None
            and valid_ratio_min < thresholds.min_valid_ratio_fail
        )
    )
    coverage_limited = bool(
        not coverage_too_thin
        and (
            (
                coverage_mean is not None
                and coverage_mean < thresholds.min_coverage_mean_warn
            )
            or (
                coverage_min is not None
                and coverage_min < thresholds.min_coverage_min_warn
            )
        )
    )

    turnover_efficiency_weak = bool(
        (
            ret_per_turnover is not None
            and ret_per_turnover <= thresholds.min_return_per_turnover
        )
        or (
            ret_per_turnover is not None
            and turnover is not None
            and turnover >= thresholds.high_turnover
            and ret_per_turnover < thresholds.high_turnover_low_efficiency_rpt
        )
        or (
            ret_per_turnover is None
            and turnover is not None
            and turnover >= thresholds.high_turnover
        )
    )

    subperiod_fails_basic = bool(
        subperiod_min is not None
        and subperiod_min < thresholds.min_subperiod_positive_share_fail
    )
    subperiod_fragile = bool(
        subperiod_min is not None
        and subperiod_min < thresholds.min_subperiod_positive_share_stable
    )
    rolling_stable = bool(
        rolling_share_min is not None
        and rolling_share_min >= thresholds.min_rolling_positive_share_stable
        and (
            rolling_worst_mean is None
            or rolling_worst_mean > thresholds.rolling_worst_mean_positive_min
        )
        and not rolling_instability_flags
    )
    rolling_fragile = bool(
        rolling_instability_flags
        or (
            rolling_share_min is not None
            and rolling_share_min < thresholds.min_rolling_positive_share_fragile
        )
        or (
            rolling_worst_mean is not None
            and rolling_worst_mean <= thresholds.rolling_worst_mean_positive_min
        )
    )

    neutralization_preserves = PRESERVES_EVIDENCE_FLAG in neutralization_flags
    neutralization_material = bool(
        MATERIAL_REDUCTION_FLAG in neutralization_flags
        or EXPOSURE_DRIVEN_FLAG in neutralization_flags
    )
    neutralization_weaken = bool(
        neutralization_material
        or MODERATE_WEAKENING_FLAG in neutralization_flags
    )

    positives: list[str] = []
    concerns: list[str] = []
    blockers: list[str] = []

    if has_strong_verdict:
        positives.append("single-case verdict is strong")
    if neutralization_preserves:
        positives.append("strong raw and neutralized evidence")
    if rolling_stable:
        positives.append("stable across rolling windows")
    if uncertainty_supportive:
        positives.append("confidence intervals remain supportive")
    if (
        not coverage_too_thin
        and not coverage_limited
        and (
            coverage_mean is not None
            or coverage_min is not None
            or valid_ratio_min is not None
        )
    ):
        positives.append("coverage and validity are sufficient")
    if (
        not turnover_efficiency_weak
        and ret_per_turnover is not None
        and ret_per_turnover > 0.0
    ):
        positives.append("turnover efficiency is acceptable")

    if neutralization_material:
        concerns.append("evidence weakens materially after neutralization")
    elif neutralization_weaken:
        concerns.append("evidence weakens after neutralization")

    if rolling_fragile:
        concerns.append("fragile across rolling windows")
    if subperiod_fragile:
        concerns.append("fragile across subperiods")
    if uncertainty_fragile and not uncertainty_supportive:
        concerns.append("uncertainty remains high")
    if coverage_limited:
        concerns.append("coverage is limited")
    if turnover_efficiency_weak:
        concerns.append("turnover efficiency weak")
    if has_fragile_verdict:
        concerns.append("single-case verdict indicates fragility")
    if has_mixed_verdict:
        concerns.append("single-case verdict is mixed")

    if coverage_too_thin:
        blockers.append("coverage too thin")
    if subperiod_fails_basic:
        blockers.append("fragile across subperiods")
    if has_weak_verdict:
        blockers.append("single-case verdict is weak")

    advance_gate = bool(
        has_strong_verdict
        and neutralization_preserves
        and rolling_stable
        and uncertainty_supportive
        and not coverage_limited
        and not turnover_efficiency_weak
        and not neutralization_material
    )
    fragility_signal_count = sum(
        (
            rolling_fragile,
            subperiod_fragile,
            uncertainty_fragile,
            has_fragile_verdict,
        )
    )

    label: str
    reasons: tuple[str, ...]
    if blockers:
        label = "Drop for now"
        reasons = _finalize_reasons(blockers, concerns, positives, max_items=5)
    elif advance_gate:
        label = "Advance to Level 2"
        reasons = _finalize_reasons(positives, max_items=5)
    elif (
        (has_strong_verdict or has_fragile_verdict)
        and not neutralization_material
        and fragility_signal_count <= thresholds.fragile_signal_count_for_strong_candidate_max
        and not coverage_limited
        and not turnover_efficiency_weak
    ):
        label = "Strong Level 1 candidate"
        reasons = _finalize_reasons(positives, concerns, max_items=5)
    elif (
        fragility_signal_count >= thresholds.fragile_signal_count_for_fragile_min
        or has_fragile_verdict
    ):
        label = "Fragile / monitor"
        reasons = _finalize_reasons(concerns, positives, max_items=5)
    else:
        label = "Needs refinement"
        reasons = _finalize_reasons(concerns, positives, max_items=5)

    if not reasons:
        reasons = ("insufficient diagnostics for campaign triage",)

    return _build_decision(
        label=label,
        reasons=reasons,
        metrics=metrics,
        gate_metrics=gate_metrics,
        positives=tuple(positives),
        concerns=tuple(blockers + concerns),
    )


def campaign_rank_sort_key(
    case_name: str,
    *,
    status: str,
    metrics: Mapping[str, object],
    triage: CampaignTriage | None = None,
    thresholds: CampaignTriageThresholds = DEFAULT_CAMPAIGN_TRIAGE_THRESHOLDS,
) -> tuple[object, ...]:
    """Explicit ranking key for campaign comparisons."""

    decision = triage or build_campaign_triage(
        metrics,
        status=status,
        thresholds=thresholds,
    )
    status_penalty = 0 if status.strip().lower() == "success" else 1
    return (
        status_penalty,
        decision.triage_priority,
        _descending_for_sort(decision.rank_primary_metric),
        _descending_for_sort(decision.rank_secondary_metric),
        _descending_for_sort(decision.rank_stability_metric),
        decision.rank_risk_count,
        -decision.rank_support_count,
        case_name.strip().lower(),
    )


def _build_decision(
    *,
    label: str,
    reasons: Sequence[str],
    metrics: Mapping[str, object],
    gate_metrics: PromotionGateMetrics | None = None,
    positives: Sequence[str],
    concerns: Sequence[str],
) -> CampaignTriage:
    projected = gate_metrics or project_promotion_gate_metrics(metrics)
    rank_primary_metric = projected["core"]["ic_ir"]
    rank_secondary_metric = projected["core"]["mean_long_short_return"]
    rank_stability_metric = projected["rolling"]["rolling_positive_share_min"]
    return CampaignTriage(
        label=label,
        reasons=tuple(reasons),
        triage_priority=_TRIAGE_PRIORITY.get(label, 5),
        rank_primary_metric=rank_primary_metric,
        rank_secondary_metric=rank_secondary_metric,
        rank_stability_metric=rank_stability_metric,
        rank_support_count=len(_dedupe(positives)),
        rank_risk_count=len(_dedupe(concerns)),
    )


def _finalize_reasons(*groups: Sequence[str], max_items: int) -> tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for reason in group:
            token = reason.strip()
            if not token or token in seen:
                continue
            out.append(token)
            seen.add(token)
            if len(out) >= max_items:
                return tuple(out)
    return tuple(out)


def _dedupe(values: Sequence[str]) -> tuple[str, ...]:
    return _finalize_reasons(values, max_items=10_000)


def _descending_for_sort(value: float | None) -> float:
    if value is None:
        return float("inf")
    return -value
