"""Compact, auditable factor verdict classification over existing diagnostics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TypedDict

from alpha_lab.key_metrics_contracts import (
    project_core_signal_evidence_metrics,
    project_neutralization_comparison_metrics,
    project_regime_metrics,
    project_rolling_stability_metrics,
    project_tail_risk_metrics,
    project_uncertainty_evidence_metrics,
)
from alpha_lab.reporting._shared import finalize_reasons
from alpha_lab.reporting.neutralization_comparison import (
    EXPOSURE_DRIVEN_FLAG,
    MATERIAL_REDUCTION_FLAG,
    MODERATE_WEAKENING_FLAG,
    PRESERVES_EVIDENCE_FLAG,
    WEAKER_BUT_STABLER_FLAG,
)
from alpha_lab.research_evaluation_config import (
    DEFAULT_RESEARCH_EVALUATION_CONFIG,
    FactorVerdictConfig,
)

FACTOR_VERDICT_TAXONOMY: tuple[str, ...] = (
    "Strong candidate",
    "Promising but fragile",
    "Mixed evidence",
    "Weak / noisy",
    "Fails basic robustness",
)


FactorVerdictThresholds = FactorVerdictConfig
DEFAULT_FACTOR_VERDICT_THRESHOLDS = DEFAULT_RESEARCH_EVALUATION_CONFIG.factor_verdict


class FactorVerdictPayload(TypedDict):
    label: str
    reasons: list[str]


@dataclass(frozen=True)
class FactorVerdict:
    label: str
    reasons: tuple[str, ...]

    def to_dict(self) -> FactorVerdictPayload:
        return {"label": self.label, "reasons": list(self.reasons)}


def build_factor_verdict(
    metrics: Mapping[str, object],
    *,
    thresholds: FactorVerdictThresholds = DEFAULT_FACTOR_VERDICT_THRESHOLDS,
) -> FactorVerdict:
    """Classify factor quality from existing diagnostics with explicit reasons."""

    core = project_core_signal_evidence_metrics(metrics)
    tail = project_tail_risk_metrics(metrics)
    regime = project_regime_metrics(metrics)
    uncertainty = project_uncertainty_evidence_metrics(metrics)
    rolling = project_rolling_stability_metrics(metrics)
    neutralization = project_neutralization_comparison_metrics(metrics)

    n_dates = core["n_dates_used"]
    if n_dates is None:
        n_dates = core["n_dates"]

    mean_ic = core["mean_ic"]
    mean_rank_ic = core["mean_rank_ic"]
    ic_pos_rate = core["ic_positive_rate"]
    rank_ic_pos_rate = core["rank_ic_positive_rate"]
    ic_valid = core["ic_valid_ratio"]
    rank_ic_valid = core["rank_ic_valid_ratio"]

    mean_ls = core["mean_long_short_return"]
    ls_ir = core["long_short_ir"]
    ls_turnover = core["mean_long_short_turnover"]
    ret_per_turnover = core["long_short_return_per_turnover"]

    subperiod_ic_share = core["subperiod_ic_positive_share"]
    subperiod_ls_share = core["subperiod_long_short_positive_share"]
    rolling_ic_share = rolling["rolling_ic_positive_share"]
    rolling_rank_ic_share = rolling["rolling_rank_ic_positive_share"]
    rolling_ls_share = rolling["rolling_long_short_positive_share"]
    rolling_ic_min_mean = rolling["rolling_ic_min_mean"]
    rolling_rank_ic_min_mean = rolling["rolling_rank_ic_min_mean"]
    rolling_ls_min_mean = rolling["rolling_long_short_min_mean"]

    coverage_mean = core["eval_coverage_ratio_mean"]
    if coverage_mean is None:
        coverage_mean = core["coverage_mean"]
    coverage_min = core["eval_coverage_ratio_min"]
    if coverage_min is None:
        coverage_min = core["coverage_min"]
    rebalance_step_dates = core["rebalance_step_dates"]
    ic_half_life_horizon = core["ic_half_life_horizon"]
    ic_half_life_status = core["ic_half_life_status"]
    ic_half_life_not_reached = core["ic_half_life_not_reached"]
    ic_decay_rebalance_ratio = core["ic_decay_rebalance_ratio"]

    instability_flags = _text_tokens(metrics.get("instability_flags"))
    rolling_instability_flags = rolling["rolling_instability_flags"]
    if not rolling_instability_flags:
        rolling_instability_flags = tuple(
            flag for flag in instability_flags if flag.startswith("rolling_")
        )
    uncertainty_flags = uncertainty["uncertainty_flags"]
    mean_ic_ci_lower = uncertainty["mean_ic_ci_lower"]
    mean_ic_ci_upper = uncertainty["mean_ic_ci_upper"]
    mean_rank_ic_ci_lower = uncertainty["mean_rank_ic_ci_lower"]
    mean_rank_ic_ci_upper = uncertainty["mean_rank_ic_ci_upper"]
    mean_ls_ci_lower = uncertainty["mean_long_short_return_ci_lower"]
    mean_ls_ci_upper = uncertainty["mean_long_short_return_ci_upper"]
    neutralization_corr_reduction = neutralization["neutralization_mean_corr_reduction"]
    neutralization_comparison_flags = neutralization["neutralization_flags"]

    observed_count = sum(
        value is not None
        for value in (
            n_dates,
            mean_ic,
            mean_rank_ic,
            ic_pos_rate,
            rank_ic_pos_rate,
            ic_valid,
            rank_ic_valid,
            mean_ls,
            ls_ir,
            ls_turnover,
            ret_per_turnover,
            subperiod_ic_share,
            subperiod_ls_share,
            rolling_ic_share,
            rolling_rank_ic_share,
            rolling_ls_share,
            rolling_ic_min_mean,
            rolling_rank_ic_min_mean,
            rolling_ls_min_mean,
            coverage_mean,
            coverage_min,
            mean_ic_ci_lower,
            mean_ic_ci_upper,
            mean_rank_ic_ci_lower,
            mean_rank_ic_ci_upper,
            mean_ls_ci_lower,
            mean_ls_ci_upper,
            neutralization_corr_reduction,
        )
    )
    if observed_count == 0 and not instability_flags and not uncertainty_flags:
        return FactorVerdict(
            label="Mixed evidence",
            reasons=("insufficient diagnostics to form a clear verdict",),
        )

    positives: list[str] = []
    concerns: list[str] = []
    weak_evidence: list[str] = []
    critical_failures: list[str] = []
    neutralization_positive_reasons: list[str] = []
    neutralization_concern_reasons: list[str] = []

    has_positive_ic_pair = False
    has_sign_consistency = False
    has_validity = False
    has_reliable_ls = False
    has_subperiod_robustness = False
    has_healthy_coverage = False
    has_uncertainty_support = False

    if n_dates is not None:
        if n_dates < thresholds.min_eval_dates_basic:
            critical_failures.append("evaluation window is too short for basic robustness")
        elif n_dates < thresholds.min_eval_dates_preferred:
            concerns.append("evaluation window is short and may be noisy")

    if mean_ic is not None and mean_rank_ic is not None:
        if mean_ic > 0.0 and mean_rank_ic > 0.0:
            has_positive_ic_pair = True
            positives.append("positive IC and RankIC means")
        elif mean_ic <= 0.0 and mean_rank_ic <= 0.0:
            weak_evidence.append("IC and RankIC means are non-positive")
        else:
            concerns.append("IC and RankIC signs are inconsistent")

    if ic_pos_rate is not None and rank_ic_pos_rate is not None:
        if (
            ic_pos_rate >= thresholds.min_sign_positive_rate
            and rank_ic_pos_rate >= thresholds.min_sign_positive_rate
        ):
            has_sign_consistency = True
            positives.append("IC and RankIC signs are consistently positive")
        elif (
            ic_pos_rate < thresholds.weak_sign_positive_rate
            and rank_ic_pos_rate < thresholds.weak_sign_positive_rate
        ):
            weak_evidence.append("IC and RankIC signs are mostly non-positive")
        else:
            concerns.append("IC/RankIC sign consistency is weak")

    min_valid = _min_or_none(ic_valid, rank_ic_valid)
    if min_valid is not None:
        if min_valid < thresholds.min_valid_ratio_fail:
            critical_failures.append("IC/RankIC valid ratio is too low")
        elif min_valid < thresholds.min_valid_ratio_strong:
            concerns.append("IC/RankIC valid ratio is below preferred level")
        else:
            has_validity = True
            positives.append("IC and RankIC validity is high")

    if mean_ls is not None and ls_ir is not None:
        if mean_ls > 0.0 and ls_ir > 0.0:
            has_reliable_ls = True
            positives.append("long-short spread is positive with positive IR")
        elif mean_ls <= 0.0 and ls_ir <= 0.0:
            weak_evidence.append("long-short spread is not reliable")
        else:
            concerns.append("long-short spread is mixed across diagnostics")

    if ret_per_turnover is not None:
        if ret_per_turnover <= thresholds.min_return_per_turnover:
            concerns.append("turnover efficiency is weak")
        elif (
            ls_turnover is not None
            and ls_turnover >= thresholds.high_turnover
            and ret_per_turnover < thresholds.high_turnover_low_efficiency_rpt
        ):
            concerns.append("high turnover with limited return efficiency")
    elif ls_turnover is not None and ls_turnover >= thresholds.high_turnover:
        concerns.append("turnover is high and efficiency evidence is limited")

    min_subperiod_share = _min_or_none(subperiod_ic_share, subperiod_ls_share)
    if min_subperiod_share is not None:
        if min_subperiod_share < thresholds.min_subperiod_share_fail:
            critical_failures.append("subperiod robustness fails basic threshold")
        elif min_subperiod_share < thresholds.min_subperiod_share_strong:
            concerns.append("subperiod robustness is mixed")
        else:
            has_subperiod_robustness = True
            positives.append("robust across subperiods")

    rolling_positive_shares = [
        value
        for value in (rolling_ic_share, rolling_rank_ic_share, rolling_ls_share)
        if value is not None
    ]
    rolling_min_means = [
        value
        for value in (rolling_ic_min_mean, rolling_rank_ic_min_mean, rolling_ls_min_mean)
        if value is not None
    ]
    min_rolling_share = min(rolling_positive_shares) if rolling_positive_shares else None
    min_rolling_mean = min(rolling_min_means) if rolling_min_means else None

    if min_rolling_share is not None:
        if min_rolling_share >= thresholds.min_rolling_positive_share_persistent:
            positives.append("evidence is persistent across rolling windows")
        elif min_rolling_share < thresholds.min_rolling_positive_share_regime_warning:
            concerns.append("rolling evidence suggests regime dependence")
        else:
            concerns.append("signal weakens materially in some periods")
    if min_rolling_mean is not None and min_rolling_mean <= 0.0:
        concerns.append("signal weakens materially in some periods")

    if (
        ic_decay_rebalance_ratio is not None
        and ic_half_life_horizon is not None
        and rebalance_step_dates is not None
    ):
        if ic_decay_rebalance_ratio >= thresholds.ic_decay_block_rebalance_ratio:
            critical_failures.append("rebalance cadence materially exceeds IC half-life")
        elif ic_decay_rebalance_ratio > thresholds.ic_decay_warn_rebalance_ratio:
            concerns.append("rebalance cadence may be too slow for IC decay")
        else:
            positives.append("rebalance cadence is consistent with IC half-life")
    elif ic_half_life_not_reached or ic_half_life_status == "not_reached":
        positives.append("IC decay remains above half-life through tested horizons")

    ls_max_dd = tail["ls_max_drawdown"]
    ls_max_consec = tail["ls_max_consecutive_loss_days"]
    ls_calmar = tail["ls_calmar_ratio"]
    has_tail_risk_ok = False

    if ls_max_dd is not None:
        if ls_max_dd >= thresholds.tail_risk_max_drawdown_fail:
            critical_failures.append("long-short max drawdown exceeds hard limit")
        elif ls_max_dd >= thresholds.tail_risk_max_drawdown_warn:
            concerns.append("long-short max drawdown is elevated")
        else:
            has_tail_risk_ok = True

    if ls_max_consec is not None:
        if ls_max_consec >= thresholds.tail_risk_max_consecutive_loss_days_fail:
            critical_failures.append("long-short has excessively long loss streak")
        elif ls_max_consec >= thresholds.tail_risk_max_consecutive_loss_days_warn:
            concerns.append("long-short has a notable loss streak")

    if ls_calmar is not None and ls_calmar <= 0.0:
        weak_evidence.append("long-short calmar ratio is non-positive")

    if has_tail_risk_ok and ls_max_dd is not None:
        positives.append("long-short tail risk is well-controlled")

    if regime["regime_has_weakness"]:
        concerns.append("factor performance is regime-dependent")

    if rolling_instability_flags:
        if "rolling_regime_dependence" in rolling_instability_flags:
            concerns.append("rolling evidence suggests regime dependence")
        if any(flag.endswith("_sign_flip_instability") for flag in rolling_instability_flags):
            concerns.append("rolling factor performance is unstable through time")

    if coverage_mean is not None or coverage_min is not None:
        if (coverage_mean is not None and coverage_mean < thresholds.min_coverage_mean_fail) or (
            coverage_min is not None and coverage_min < thresholds.min_coverage_min_fail
        ):
            critical_failures.append("coverage is too thin for reliable evaluation")
        elif (coverage_mean is not None and coverage_mean < thresholds.min_coverage_mean_warn) or (
            coverage_min is not None and coverage_min < thresholds.min_coverage_min_strong
        ):
            concerns.append("coverage is uneven across evaluation dates")
        elif (
            coverage_mean is not None
            and coverage_mean >= thresholds.min_coverage_mean_strong
            and (coverage_min is None or coverage_min >= thresholds.min_coverage_min_strong)
        ):
            has_healthy_coverage = True
            positives.append("coverage is healthy")

    if instability_flags:
        concerns.append(
            "instability flags triggered: " + _short_join(instability_flags, max_items=3)
        )

    uncertainty_overlap_metrics: list[str] = []
    if (
        mean_ic_ci_lower is not None
        and mean_ic_ci_upper is not None
        and mean_ic_ci_lower <= 0.0 <= mean_ic_ci_upper
    ):
        uncertainty_overlap_metrics.append("IC")
    if (
        mean_rank_ic_ci_lower is not None
        and mean_rank_ic_ci_upper is not None
        and mean_rank_ic_ci_lower <= 0.0 <= mean_rank_ic_ci_upper
    ):
        uncertainty_overlap_metrics.append("RankIC")
    if (
        mean_ls_ci_lower is not None
        and mean_ls_ci_upper is not None
        and mean_ls_ci_lower <= 0.0 <= mean_ls_ci_upper
    ):
        uncertainty_overlap_metrics.append("long-short")

    if (
        mean_ic_ci_lower is not None
        and mean_rank_ic_ci_lower is not None
        and mean_ls_ci_lower is not None
        and mean_ic_ci_lower > 0.0
        and mean_rank_ic_ci_lower > 0.0
        and mean_ls_ci_lower > 0.0
    ):
        has_uncertainty_support = True
        positives.append("evidence remains positive under uncertainty")

    if len(uncertainty_overlap_metrics) >= thresholds.uncertainty_overlap_zero_fail_count:
        weak_evidence.append("signal direction is unstable under uncertainty")
    elif uncertainty_overlap_metrics:
        concerns.append(
            "confidence interval overlaps zero: " + ", ".join(uncertainty_overlap_metrics)
        )

    wide_flags = tuple(flag for flag in uncertainty_flags if flag.endswith("_ci_wide"))
    unavailable_flags = tuple(
        flag for flag in uncertainty_flags if flag.endswith("_ci_unavailable")
    )
    if wide_flags:
        concerns.append("apparent edge is weak relative to estimation noise")
    if unavailable_flags:
        concerns.append("uncertainty estimates are limited by small sample")

    if (
        neutralization_corr_reduction is not None
        and neutralization_corr_reduction >= thresholds.neutralization_material_corr_reduction
    ):
        concerns.append("neutralization materially reduces independent evidence")

    if PRESERVES_EVIDENCE_FLAG in neutralization_comparison_flags:
        neutralization_positive_reasons.append("signal survives neutralization well")
    if MODERATE_WEAKENING_FLAG in neutralization_comparison_flags:
        neutralization_concern_reasons.append("neutralization moderately weakens evidence")
    if MATERIAL_REDUCTION_FLAG in neutralization_comparison_flags:
        neutralization_concern_reasons.append(
            "neutralization materially reduces independent evidence"
        )
    if EXPOSURE_DRIVEN_FLAG in neutralization_comparison_flags:
        neutralization_concern_reasons.append("raw signal may be driven by common exposure")
    if WEAKER_BUT_STABLER_FLAG in neutralization_comparison_flags:
        neutralization_positive_reasons.append("neutralized signal is weaker but more stable")
    _append_extended_diagnostic_reasons(
        metrics,
        base_mean_ic=mean_ic,
        critical_failures=critical_failures,
        concerns=concerns,
        positives=positives,
    )

    concerns.extend(neutralization_concern_reasons)
    positives.extend(neutralization_positive_reasons)

    positive_score = sum(
        (
            has_positive_ic_pair,
            has_sign_consistency,
            has_validity,
            has_reliable_ls,
            has_subperiod_robustness,
            has_healthy_coverage,
            has_uncertainty_support,
        )
    )
    has_weak_core_signal = bool(
        weak_evidence
        and (
            "IC and RankIC means are non-positive" in weak_evidence
            or "long-short spread is not reliable" in weak_evidence
        )
    )

    if critical_failures:
        label = "Fails basic robustness"
        reasons = finalize_reasons(
            critical_failures,
            neutralization_concern_reasons,
            weak_evidence,
            concerns,
            neutralization_positive_reasons,
            positives,
            max_items=5,
        )
    elif positive_score >= 5 and not concerns and not weak_evidence:
        label = "Strong candidate"
        uncertainty_positive = (
            ("evidence remains positive under uncertainty",)
            if "evidence remains positive under uncertainty" in positives
            else ()
        )
        positives_without_uncertainty = tuple(
            token for token in positives if token not in uncertainty_positive
        )
        reasons = finalize_reasons(
            uncertainty_positive,
            neutralization_positive_reasons,
            positives_without_uncertainty,
            max_items=6,
        )
    elif positive_score >= 3 and not has_weak_core_signal:
        label = "Promising but fragile"
        reasons = finalize_reasons(
            neutralization_positive_reasons,
            positives[:2],
            neutralization_concern_reasons,
            concerns,
            weak_evidence,
            positives[2:],
            max_items=6,
        )
    elif positive_score <= 1 or has_weak_core_signal:
        label = "Weak / noisy"
        reasons = finalize_reasons(
            weak_evidence,
            neutralization_concern_reasons,
            concerns,
            neutralization_positive_reasons,
            positives,
            max_items=5,
        )
    else:
        label = "Mixed evidence"
        reasons = finalize_reasons(
            neutralization_positive_reasons,
            positives,
            neutralization_concern_reasons,
            concerns,
            weak_evidence,
            max_items=6,
        )

    if not reasons:
        reasons = ("insufficient diagnostics to explain verdict",)
    return FactorVerdict(label=label, reasons=reasons)


def _append_extended_diagnostic_reasons(
    metrics: Mapping[str, object],
    *,
    base_mean_ic: float | None,
    critical_failures: list[str],
    concerns: list[str],
    positives: list[str],
) -> None:
    """Translate P0/P1 diagnostic metrics (tradability, next-open, haircut,
    random-baseline, lag, baseline comparison) into verdict reasons.

    Keeps thresholds inline — these diagnostics are newer than the core
    `FactorVerdictConfig` and picking calibrated thresholds per-profile is
    out of scope here.  Only raises concerns when the signal is clearly
    degraded, so a clean factor stays clean.
    """
    tradability_rate = _float_or_none(metrics.get("tradability_untradable_rate"))
    if tradability_rate is not None and tradability_rate > 0.10:
        concerns.append("tradability leakage: >10% of rows were untradable")

    tradability_ic_delta = _float_or_none(metrics.get("tradability_ic_delta"))
    if (
        tradability_ic_delta is not None
        and base_mean_ic is not None
        and base_mean_ic > 0.0
        and tradability_ic_delta < -0.3 * abs(base_mean_ic)
    ):
        concerns.append("IC drops materially after untradable-day filter")

    next_open_available = metrics.get("next_open_execution_available")
    next_open_ic = _float_or_none(metrics.get("next_open_mean_ic"))
    if (
        next_open_available is True
        and base_mean_ic is not None
        and base_mean_ic > 0.0
        and next_open_ic is not None
        and next_open_ic < 0.3 * base_mean_ic
    ):
        concerns.append("signal collapses under next-open execution")

    haircut_ratio = _float_or_none(metrics.get("haircut_sharpe_ratio"))
    if haircut_ratio is not None:
        if haircut_ratio < 0.3:
            concerns.append("most of observed Sharpe explained by multiple-testing haircut")
        elif haircut_ratio > 0.7:
            positives.append("Sharpe survives multiple-testing haircut")

    random_p = _float_or_none(metrics.get("random_baseline_p_value"))
    if random_p is not None:
        if random_p >= 0.10:
            concerns.append("IC not distinguishable from random factor baseline")
        elif random_p < 0.05:
            positives.append("IC exceeds random factor baseline (p<0.05)")

    baseline_edge = _float_or_none(metrics.get("baseline_factor_mean_ic_advantage"))
    if baseline_edge is not None and baseline_edge < 0.0:
        concerns.append("factor underperforms simple momentum/reversal baseline")

    lag_decay = _float_or_none(metrics.get("lag_sensitivity_ic_decay_lag_1"))
    lag0 = _float_or_none(metrics.get("lag_sensitivity_mean_ic_lag_0"))
    if lag_decay is not None and lag0 is not None and lag0 > 0.0 and lag_decay < 0.3:
        concerns.append("IC decays sharply after 1-day execution lag")

    long_share = _float_or_none(metrics.get("daily_pnl_long_contribution_ratio"))
    if long_share is not None:
        if long_share < 0.10:
            concerns.append("PnL is almost entirely short-leg driven")
        elif long_share > 0.90:
            concerns.append("PnL is almost entirely long-leg driven")

    cost_share = _float_or_none(metrics.get("daily_pnl_cost_drag_share"))
    if cost_share is not None and cost_share > 0.75:
        concerns.append("transaction costs erode most of gross edge")

    param_ic_std = _float_or_none(metrics.get("param_sensitivity_mean_ic_std"))
    if (
        param_ic_std is not None
        and base_mean_ic is not None
        and abs(base_mean_ic) > 1e-6
        and param_ic_std / abs(base_mean_ic) > 0.5
    ):
        concerns.append("IC varies materially across n_quantiles choices")


def _float_or_none(value: object) -> float | None:
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if result != result:  # NaN check
        return None
    return result


def reasons_to_text(reasons: Sequence[str]) -> str:
    tokens = [str(reason).strip() for reason in reasons if str(reason).strip()]
    return "; ".join(tokens)


def parse_text_list(value: object) -> tuple[str, ...]:
    return _text_tokens(value)


def _text_tokens(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple, set)):
        out = [str(item).strip() for item in value if str(item).strip()]
        return tuple(out)
    text = str(value).strip()
    if not text:
        return ()
    delim = ";" if ";" in text else ","
    tokens = [part.strip() for part in text.split(delim) if part.strip()]
    return tuple(tokens)


def _min_or_none(left: float | None, right: float | None) -> float | None:
    vals = [value for value in (left, right) if value is not None]
    if not vals:
        return None
    return min(vals)


def _short_join(values: Sequence[str], *, max_items: int) -> str:
    if len(values) <= max_items:
        return ", ".join(values)
    prefix = ", ".join(values[:max_items])
    return f"{prefix}, +{len(values) - max_items} more"


