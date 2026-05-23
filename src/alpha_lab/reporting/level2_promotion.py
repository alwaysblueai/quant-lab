"""Level 2 promotion gate classification over existing Level 1/2 evidence."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypedDict

from alpha_lab.key_metrics_contracts import project_promotion_gate_metrics
from alpha_lab.reporting._shared import finalize_reasons
from alpha_lab.reporting.neutralization_comparison import (
    EXPOSURE_DRIVEN_FLAG,
    MATERIAL_REDUCTION_FLAG,
    MODERATE_WEAKENING_FLAG,
    PRESERVES_EVIDENCE_FLAG,
)
from alpha_lab.research_evaluation_config import (
    DEFAULT_RESEARCH_EVALUATION_CONFIG,
    Level2PromotionConfig,
)

LEVEL2_PROMOTION_TAXONOMY: tuple[str, ...] = (
    "Promote to Level 2",
    "Hold for refinement",
    "Blocked from Level 2",
)


Level2PromotionThresholds = Level2PromotionConfig
DEFAULT_LEVEL2_PROMOTION_THRESHOLDS = DEFAULT_RESEARCH_EVALUATION_CONFIG.level2_promotion


class Level2PromotionPayload(TypedDict):
    promotion_decision: str
    promotion_reasons: list[str]
    promotion_blockers: list[str]


@dataclass(frozen=True)
class Level2PromotionDecision:
    """Compact, auditable Level 2 promotion decision."""

    label: str
    reasons: tuple[str, ...]
    blockers: tuple[str, ...]

    def to_dict(self) -> Level2PromotionPayload:
        return {
            "promotion_decision": self.label,
            "promotion_reasons": list(self.reasons),
            "promotion_blockers": list(self.blockers),
        }


def build_level2_promotion(
    metrics: Mapping[str, object],
    *,
    status: str = "success",
    thresholds: Level2PromotionThresholds = DEFAULT_LEVEL2_PROMOTION_THRESHOLDS,
) -> Level2PromotionDecision:
    """Classify whether a case should advance from Level 1 screening to Level 2."""

    gate_metrics = project_promotion_gate_metrics(metrics)
    core_metrics = gate_metrics["core"]
    tail_risk_metrics = gate_metrics["tail_risk"]
    regime_metrics = gate_metrics["regime"]
    marginal_metrics = gate_metrics["marginal"]
    uncertainty_metrics = gate_metrics["uncertainty"]
    rolling_metrics = gate_metrics["rolling"]
    neutralization_metrics = gate_metrics["neutralization"]

    if status.strip().lower() != "success":
        status_blockers = ("blocked by unsuccessful case status",)
        return Level2PromotionDecision(
            label="Blocked from Level 2",
            reasons=status_blockers,
            blockers=status_blockers,
        )

    verdict = gate_metrics["factor_verdict"].lower()
    triage_label = gate_metrics["campaign_triage"]
    rolling_instability_flags = rolling_metrics["rolling_instability_flags"]
    neutralization_flags = neutralization_metrics["neutralization_flags"]

    coverage_mean = core_metrics["coverage_mean"]
    if coverage_mean is None:
        coverage_mean = core_metrics["eval_coverage_ratio_mean"]
    coverage_min = core_metrics["coverage_min"]
    if coverage_min is None:
        coverage_min = core_metrics["eval_coverage_ratio_min"]
    valid_ratio_min = core_metrics["valid_ratio_min"]
    subperiod_min = core_metrics["subperiod_positive_share_min"]
    rolling_share_min = rolling_metrics["rolling_positive_share_min"]
    rolling_worst_mean = rolling_metrics["rolling_worst_mean_min"]
    turnover = core_metrics["mean_long_short_turnover"]
    ret_per_turnover = core_metrics["long_short_return_per_turnover"]

    supportive_ci_count = uncertainty_metrics["uncertainty_supportive_ci_count"]
    uncertainty_overlap_count = uncertainty_metrics["uncertainty_overlap_zero_count"]
    uncertainty_supportive = supportive_ci_count >= thresholds.min_supportive_ci_count_promote

    has_strong_verdict = verdict == "strong candidate"
    has_weak_verdict = verdict in {"weak / noisy", "fails basic robustness"}

    neutralization_preserves = PRESERVES_EVIDENCE_FLAG in neutralization_flags
    neutralization_material = bool(
        MATERIAL_REDUCTION_FLAG in neutralization_flags
        or EXPOSURE_DRIVEN_FLAG in neutralization_flags
    )
    neutralization_moderate = MODERATE_WEAKENING_FLAG in neutralization_flags
    neutralization_has_signal = bool(neutralization_flags)

    coverage_block = bool(
        (coverage_mean is not None and coverage_mean < thresholds.min_coverage_mean_block)
        or (coverage_min is not None and coverage_min < thresholds.min_coverage_min_block)
        or (valid_ratio_min is not None and valid_ratio_min < thresholds.min_valid_ratio_block)
    )
    subperiod_block = bool(
        subperiod_min is not None and subperiod_min < thresholds.min_subperiod_positive_share_block
    )
    rolling_block = bool(
        (
            rolling_share_min is not None
            and rolling_share_min < thresholds.min_rolling_positive_share_block
        )
        or (
            rolling_worst_mean is not None
            and rolling_worst_mean <= thresholds.rolling_worst_mean_block_max
        )
        or (thresholds.rolling_instability_is_blocker and bool(rolling_instability_flags))
    )
    uncertainty_block = bool(
        uncertainty_overlap_count >= thresholds.uncertainty_overlap_block_min_count
    )
    turnover_block = bool(
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

    coverage_promote = bool(
        (coverage_mean is not None or coverage_min is not None or valid_ratio_min is not None)
        and (coverage_mean is None or coverage_mean >= thresholds.min_coverage_mean_promote)
        and (coverage_min is None or coverage_min >= thresholds.min_coverage_min_promote)
        and (valid_ratio_min is None or valid_ratio_min >= thresholds.min_valid_ratio_promote)
    )
    subperiod_promote = bool(
        subperiod_min is not None
        and subperiod_min >= thresholds.min_subperiod_positive_share_promote
    )
    rolling_promote = bool(
        rolling_share_min is not None
        and rolling_share_min >= thresholds.min_rolling_positive_share_promote
        and (
            rolling_worst_mean is None
            or rolling_worst_mean > thresholds.rolling_worst_mean_promote_min
        )
        and not rolling_instability_flags
    )

    neutralization_promote = bool(
        neutralization_preserves and not neutralization_material and not neutralization_moderate
    )
    if not thresholds.require_neutralization_support_for_promote:
        neutralization_promote = not neutralization_material

    ls_max_dd = tail_risk_metrics["ls_max_drawdown"]
    ls_max_consec = tail_risk_metrics["ls_max_consecutive_loss_days"]
    tail_risk_block = bool(
        (ls_max_dd is not None and ls_max_dd >= 0.30)
        or (ls_max_consec is not None and ls_max_consec >= 15)
    )
    tail_risk_promote = bool(
        ls_max_dd is not None and ls_max_dd < 0.15 and (ls_max_consec is None or ls_max_consec < 8)
    )

    blockers: list[str] = []
    reasons: tuple[str, ...] = ()
    supports: list[str] = []
    concerns: list[str] = []

    if has_weak_verdict:
        blockers.append("blocked by weak single-case verdict")
    if coverage_block:
        blockers.append("blocked by thin coverage")
    if subperiod_block:
        blockers.append("blocked by fragile subperiod evidence")
    if rolling_block:
        blockers.append("blocked by unstable rolling evidence")
    if uncertainty_block:
        blockers.append("blocked by high uncertainty overlap")
    if neutralization_material:
        blockers.append("blocked by weak neutralized evidence")
    if turnover_block:
        blockers.append("blocked by poor turnover efficiency")
    if tail_risk_block:
        blockers.append("blocked by severe long-short tail risk")

    if has_strong_verdict:
        supports.append("factor verdict is strong")
    if uncertainty_supportive:
        supports.append("uncertainty remains supportive")
    if subperiod_promote:
        supports.append("robust across subperiods")
    if rolling_promote:
        supports.append("stable across rolling windows")
    if coverage_promote:
        supports.append("coverage and validity are sufficient")
    if ret_per_turnover is not None and ret_per_turnover > 0.0 and not turnover_block:
        supports.append("turnover efficiency is acceptable")
    if neutralization_promote:
        supports.append("robust evidence survives neutralization")
    if tail_risk_promote:
        supports.append("long-short tail risk is well-controlled")

    if not has_strong_verdict and not has_weak_verdict:
        concerns.append("factor verdict is not yet strong")
    if not uncertainty_supportive and not uncertainty_block:
        concerns.append("uncertainty support is incomplete")
    if not subperiod_block and not subperiod_promote:
        concerns.append("subperiod robustness is not yet persistent")
    if not rolling_block and not rolling_promote:
        concerns.append("rolling stability is not yet persistent")
    if not coverage_block and not coverage_promote:
        concerns.append("coverage/validity are below promotion target")
    if regime_metrics["regime_has_weakness"]:
        concerns.append("factor performance is regime-dependent")
    if marginal_metrics["marginal_contribution_weak"]:
        concerns.append("marginal contribution is weak in multi-factor context")
    if neutralization_moderate:
        concerns.append("neutralization weakens evidence")
    if thresholds.require_neutralization_support_for_promote and not neutralization_has_signal:
        concerns.append("neutralization evidence is unavailable")
    if triage_label in {"Advance to Level 2", "Strong Level 1 candidate"} and (
        blockers or concerns
    ):
        concerns.append("campaign triage is favorable but promotion gate remains unmet")

    _append_extended_promotion_signals(
        metrics,
        base_mean_ic=core_metrics.get("mean_ic"),
        blockers=blockers,
        concerns=concerns,
        supports=supports,
    )

    promote_gate = bool(
        (has_strong_verdict or not thresholds.require_strong_verdict_for_promote)
        and uncertainty_supportive
        and subperiod_promote
        and rolling_promote
        and coverage_promote
        and neutralization_promote
        and not turnover_block
        and not blockers
    )

    if blockers:
        reasons = finalize_reasons(blockers, concerns, supports, max_items=6)
        return Level2PromotionDecision(
            label="Blocked from Level 2",
            reasons=reasons or tuple(blockers),
            blockers=finalize_reasons(blockers, max_items=6),
        )

    if promote_gate:
        reasons = finalize_reasons(supports, max_items=6)
        if not reasons:
            reasons = ("evidence satisfies Level 2 promotion gate",)
        return Level2PromotionDecision(
            label="Promote to Level 2",
            reasons=reasons,
            blockers=(),
        )

    reasons = finalize_reasons(concerns, supports, max_items=6)
    if not reasons:
        reasons = ("additional robustness evidence is required before Level 2",)
    return Level2PromotionDecision(
        label="Hold for refinement",
        reasons=reasons,
        blockers=(),
    )


def _append_extended_promotion_signals(
    metrics: Mapping[str, object],
    *,
    base_mean_ic: float | None,
    blockers: list[str],
    concerns: list[str],
    supports: list[str],
) -> None:
    """Fold P0/P1 diagnostics (tradability, next-open, haircut, random
    baseline, lag, baseline comparison) into the Level-2 promotion gate.

    Promotion is a higher bar than campaign triage: anything that suggests
    the headline edge is a measurement artifact should hard-block.  Softer
    signals contribute to concerns/supports so the gate label reflects them.
    """

    def _f(key: str) -> float | None:
        v = metrics.get(key)
        if isinstance(v, bool):
            x = float(v)
        elif isinstance(v, (int, float)):
            x = float(v)
        elif isinstance(v, str):
            token = v.strip()
            if not token:
                return None
            try:
                x = float(token)
            except ValueError:
                return None
        else:
            return None
        if x != x:
            return None
        return x

    base_ic = (
        base_mean_ic
        if isinstance(base_mean_ic, (int, float)) and not isinstance(base_mean_ic, bool)
        else None
    )

    tradability_rate = _f("tradability_untradable_rate")
    if tradability_rate is not None:
        if tradability_rate > 0.20:
            blockers.append("blocked by excessive untradable rate")
        elif tradability_rate > 0.10:
            concerns.append("untradable rate is elevated")

    if (
        metrics.get("next_open_execution_available") is True
        and base_ic is not None
        and base_ic > 0.0
    ):
        next_open_ic = _f("next_open_mean_ic")
        if next_open_ic is not None:
            if next_open_ic < 0.3 * base_ic:
                blockers.append("blocked by collapse under next-open execution")
            elif next_open_ic < 0.7 * base_ic:
                concerns.append("signal weakens under next-open execution")

    haircut_ratio = _f("haircut_sharpe_ratio")
    if haircut_ratio is not None:
        if haircut_ratio < 0.3:
            blockers.append("blocked by multiple-testing haircut")
        elif haircut_ratio < 0.5:
            concerns.append("multiple-testing haircut reduces defensible edge")
        elif haircut_ratio > 0.7:
            supports.append("Sharpe survives multiple-testing haircut")

    random_p = _f("random_baseline_p_value")
    if random_p is not None:
        if random_p >= 0.20:
            blockers.append("blocked by inability to beat random factor baseline")
        elif random_p >= 0.10:
            concerns.append("random-baseline p-value is weak")
        elif random_p < 0.05:
            supports.append("IC exceeds random factor baseline (p<0.05)")

    baseline_edge = _f("baseline_factor_mean_ic_advantage")
    if baseline_edge is not None:
        if baseline_edge < 0.0:
            blockers.append("blocked by underperformance vs simple momentum/reversal baselines")
        elif baseline_edge > 0.02:
            supports.append("factor outperforms simple momentum/reversal baselines")

    lag_decay = _f("lag_sensitivity_ic_decay_lag_1")
    lag0 = _f("lag_sensitivity_mean_ic_lag_0")
    if lag_decay is not None and lag0 is not None and lag0 > 0.0:
        if lag_decay < 0.3:
            blockers.append("blocked by sharp IC decay under 1-day execution lag")
        elif lag_decay < 0.5:
            concerns.append("IC decays materially under 1-day execution lag")
        elif lag_decay > 0.7:
            supports.append("IC retained under 1-day execution lag")

    cost_share = _f("daily_pnl_cost_drag_share")
    if cost_share is not None and cost_share > 0.75:
        blockers.append("blocked by transaction costs consuming most of gross edge")
    elif cost_share is not None and cost_share > 0.50:
        concerns.append("transaction costs consume substantial gross edge")

    long_share = _f("daily_pnl_long_contribution_ratio")
    if long_share is not None and (long_share < 0.05 or long_share > 0.95):
        concerns.append("PnL attribution is heavily concentrated in a single leg")

    param_ic_std = _f("param_sensitivity_mean_ic_std")
    if (
        param_ic_std is not None
        and base_ic is not None
        and abs(base_ic) > 1e-6
        and param_ic_std / abs(base_ic) > 0.5
    ):
        concerns.append("IC varies materially across n_quantiles choices")
