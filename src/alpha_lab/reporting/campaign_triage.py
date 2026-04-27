"""Campaign-level triage and ranking helpers for Level 1/2 research outputs."""

from __future__ import annotations

import math
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
from alpha_lab.validation.deflated_sharpe import deflated_sharpe_ratio

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
    tail_risk_metrics = gate_metrics["tail_risk"]
    regime_metrics = gate_metrics["regime"]
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
    rebalance_step_dates = core_metrics["rebalance_step_dates"]
    ic_half_life_horizon = core_metrics["ic_half_life_horizon"]
    ic_half_life_status = core_metrics["ic_half_life_status"]
    ic_half_life_not_reached = core_metrics["ic_half_life_not_reached"]
    ic_decay_rebalance_ratio = core_metrics["ic_decay_rebalance_ratio"]

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
        (coverage_mean is not None and coverage_mean < thresholds.min_coverage_mean_fail)
        or (coverage_min is not None and coverage_min < thresholds.min_coverage_min_fail)
        or (valid_ratio_min is not None and valid_ratio_min < thresholds.min_valid_ratio_fail)
    )
    coverage_limited = bool(
        not coverage_too_thin
        and (
            (coverage_mean is not None and coverage_mean < thresholds.min_coverage_mean_warn)
            or (coverage_min is not None and coverage_min < thresholds.min_coverage_min_warn)
        )
    )

    turnover_efficiency_weak = bool(
        (ret_per_turnover is not None and ret_per_turnover <= thresholds.min_return_per_turnover)
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
    ic_decay_block = bool(
        ic_decay_rebalance_ratio is not None
        and ic_decay_rebalance_ratio >= thresholds.ic_decay_block_rebalance_ratio
    )
    ic_decay_concern = bool(
        ic_decay_rebalance_ratio is not None
        and ic_decay_rebalance_ratio > thresholds.ic_decay_warn_rebalance_ratio
        and not ic_decay_block
    )

    subperiod_fails_basic = bool(
        subperiod_min is not None and subperiod_min < thresholds.min_subperiod_positive_share_fail
    )
    subperiod_fragile = bool(
        subperiod_min is not None and subperiod_min < thresholds.min_subperiod_positive_share_stable
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
        neutralization_material or MODERATE_WEAKENING_FLAG in neutralization_flags
    )

    ls_max_dd = tail_risk_metrics["ls_max_drawdown"]
    ls_max_consec = tail_risk_metrics["ls_max_consecutive_loss_days"]

    tail_risk_severe = bool(
        (ls_max_dd is not None and ls_max_dd >= 0.30)
        or (ls_max_consec is not None and ls_max_consec >= 15)
    )
    tail_risk_elevated = bool(
        not tail_risk_severe
        and (
            (ls_max_dd is not None and ls_max_dd >= 0.15)
            or (ls_max_consec is not None and ls_max_consec >= 8)
        )
    )

    positives: list[str] = []
    concerns: list[str] = []
    blockers: list[str] = []

    # If the caller provides a multi-trial context, compute DSR on demand.
    n_trials = _as_int(metrics.get("n_trials"))
    if n_trials is not None and n_trials > 1:
        dsr_pvalue = _as_float(metrics.get("dsr_pvalue"))
        if dsr_pvalue is None:
            observed_sr = core_metrics["long_short_ir"]
            n_obs = core_metrics["n_dates"] or core_metrics["n_dates_used"]
            if (
                observed_sr is not None
                and n_obs is not None
                and n_obs >= 2
                and math.isfinite(observed_sr)
            ):
                dsr_pvalue = deflated_sharpe_ratio(
                    observed_sr=observed_sr,
                    n_trials=n_trials,
                    n_obs=int(n_obs),
                )
        if dsr_pvalue is not None:
            if dsr_pvalue <= 0.10:
                positives.append("multi-trial deflated sharpe remains supportive")
            elif dsr_pvalue >= 0.50:
                concerns.append("multi-trial deflated sharpe is weak")

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
        and (coverage_mean is not None or coverage_min is not None or valid_ratio_min is not None)
    ):
        positives.append("coverage and validity are sufficient")
    if not turnover_efficiency_weak and ret_per_turnover is not None and ret_per_turnover > 0.0:
        positives.append("turnover efficiency is acceptable")
    if (
        ic_decay_rebalance_ratio is not None
        and ic_half_life_horizon is not None
        and rebalance_step_dates is not None
        and ic_decay_rebalance_ratio <= thresholds.ic_decay_warn_rebalance_ratio
    ):
        positives.append("rebalance cadence matches IC decay profile")
    elif ic_half_life_not_reached or ic_half_life_status == "not_reached":
        positives.append("IC decay remains durable through tested horizons")

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
    if ic_decay_concern:
        concerns.append("rebalance cadence is slower than IC decay")
    if has_fragile_verdict:
        concerns.append("single-case verdict indicates fragility")
    if has_mixed_verdict:
        concerns.append("single-case verdict is mixed")

    if tail_risk_severe:
        blockers.append("long-short tail risk is severe")
    elif tail_risk_elevated:
        concerns.append("long-short tail risk is elevated")
    elif ls_max_dd is not None:
        positives.append("long-short tail risk is controlled")

    if regime_metrics["regime_has_weakness"]:
        concerns.append("factor performance is regime-dependent")

    if coverage_too_thin:
        blockers.append("coverage too thin")
    if subperiod_fails_basic:
        blockers.append("fragile across subperiods")
    if has_weak_verdict:
        blockers.append("single-case verdict is weak")
    if ic_decay_block:
        blockers.append("rebalance cadence materially exceeds IC half-life")

    advance_gate = bool(
        has_strong_verdict
        and neutralization_preserves
        and rolling_stable
        and uncertainty_supportive
        and not coverage_limited
        and not turnover_efficiency_weak
        and not neutralization_material
        and not ic_decay_concern
    )
    extended_block = _append_extended_triage_signals(
        metrics,
        base_mean_ic=core_metrics.get("mean_ic"),
        blockers=blockers,
        concerns=concerns,
        positives=positives,
    )

    fragility_signal_count = sum(
        (
            rolling_fragile,
            subperiod_fragile,
            uncertainty_fragile,
            has_fragile_verdict,
            extended_block["extended_fragility"],
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
        and not ic_decay_concern
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


def _as_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        out = float(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            out = float(text)
        except ValueError:
            return None
    else:
        return None
    if not math.isfinite(out):
        return None
    return out


def _as_int(value: object) -> int | None:
    parsed = _as_float(value)
    if parsed is None:
        return None
    return int(parsed)


def _append_extended_triage_signals(
    metrics: Mapping[str, object],
    *,
    base_mean_ic: float | None,
    blockers: list[str],
    concerns: list[str],
    positives: list[str],
) -> dict[str, int]:
    """Fold P0/P1 diagnostics (tradability, next-open, haircut, random
    baseline, lag, baseline comparison) into campaign triage.

    Returns a small summary dict used by the caller; currently exposes an
    ``extended_fragility`` count that contributes to the fragile-signal tally.
    """
    fragility = 0

    tradability_rate = _as_float(metrics.get("tradability_untradable_rate"))
    if tradability_rate is not None:
        if tradability_rate > 0.25:
            blockers.append("untradable rate exceeds 25%")
        elif tradability_rate > 0.10:
            concerns.append("untradable rate above 10%")
            fragility += 1

    base_ic_val = base_mean_ic if isinstance(base_mean_ic, (int, float)) else None
    if isinstance(base_ic_val, bool):
        base_ic_val = None

    if (
        metrics.get("next_open_execution_available") is True
        and base_ic_val is not None
        and base_ic_val > 0.0
    ):
        next_open_ic = _as_float(metrics.get("next_open_mean_ic"))
        if next_open_ic is not None and next_open_ic < 0.3 * base_ic_val:
            concerns.append("signal collapses under next-open execution")
            fragility += 1

    haircut_ratio = _as_float(metrics.get("haircut_sharpe_ratio"))
    if haircut_ratio is not None:
        if haircut_ratio < 0.3:
            concerns.append("observed Sharpe largely explained by multiple-testing haircut")
            fragility += 1
        elif haircut_ratio > 0.7:
            positives.append("Sharpe survives multiple-testing haircut")

    random_p = _as_float(metrics.get("random_baseline_p_value"))
    if random_p is not None:
        if random_p >= 0.25:
            blockers.append("IC not distinguishable from random factor baseline")
        elif random_p >= 0.10:
            concerns.append("IC is weak relative to random factor baseline")
            fragility += 1
        elif random_p < 0.05:
            positives.append("IC exceeds random factor baseline (p<0.05)")

    baseline_edge = _as_float(metrics.get("baseline_factor_mean_ic_advantage"))
    if baseline_edge is not None:
        if baseline_edge < -0.02:
            blockers.append("factor loses to simple momentum/reversal baselines")
        elif baseline_edge < 0.0:
            concerns.append("factor underperforms simple momentum/reversal baseline")
            fragility += 1
        elif baseline_edge > 0.02:
            positives.append("factor outperforms simple momentum/reversal baselines")

    lag_decay = _as_float(metrics.get("lag_sensitivity_ic_decay_lag_1"))
    lag0 = _as_float(metrics.get("lag_sensitivity_mean_ic_lag_0"))
    if lag_decay is not None and lag0 is not None and lag0 > 0.0:
        if lag_decay < 0.3:
            concerns.append("IC decays sharply after 1-day execution lag")
            fragility += 1
        elif lag_decay > 0.7:
            positives.append("IC retained under 1-day execution lag")

    cost_share = _as_float(metrics.get("daily_pnl_cost_drag_share"))
    if cost_share is not None and cost_share > 0.75:
        concerns.append("transaction costs erode most of gross edge")

    long_share = _as_float(metrics.get("daily_pnl_long_contribution_ratio"))
    if long_share is not None and (long_share < 0.10 or long_share > 0.90):
        concerns.append("PnL attribution is concentrated in a single leg")

    param_ic_std = _as_float(metrics.get("param_sensitivity_mean_ic_std"))
    if (
        param_ic_std is not None
        and base_ic_val is not None
        and abs(base_ic_val) > 1e-6
        and param_ic_std / abs(base_ic_val) > 0.5
    ):
        concerns.append("IC varies materially across n_quantiles choices")
        fragility += 1

    # Clamp contribution to fragility_signal_count: treat as boolean per source.
    return {"extended_fragility": 1 if fragility > 0 else 0}
