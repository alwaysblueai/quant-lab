"""Compact, auditable factor verdict classification over existing diagnostics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TypedDict

from alpha_lab.key_metrics_contracts import (
    project_core_signal_evidence_metrics,
    project_neutralization_comparison_metrics,
    project_rolling_stability_metrics,
    project_uncertainty_evidence_metrics,
)
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
            critical_failures.append(
                "evaluation window is too short for basic robustness"
            )
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

    if rolling_instability_flags:
        if "rolling_regime_dependence" in rolling_instability_flags:
            concerns.append("rolling evidence suggests regime dependence")
        if any(
            flag.endswith("_sign_flip_instability")
            for flag in rolling_instability_flags
        ):
            concerns.append("rolling factor performance is unstable through time")

    if coverage_mean is not None or coverage_min is not None:
        if (
            coverage_mean is not None
            and coverage_mean < thresholds.min_coverage_mean_fail
        ) or (
            coverage_min is not None and coverage_min < thresholds.min_coverage_min_fail
        ):
            critical_failures.append("coverage is too thin for reliable evaluation")
        elif (
            coverage_mean is not None
            and coverage_mean < thresholds.min_coverage_mean_warn
        ) or (
            coverage_min is not None
            and coverage_min < thresholds.min_coverage_min_strong
        ):
            concerns.append("coverage is uneven across evaluation dates")
        elif (
            coverage_mean is not None
            and coverage_mean >= thresholds.min_coverage_mean_strong
            and (
                coverage_min is None
                or coverage_min >= thresholds.min_coverage_min_strong
            )
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
            "confidence interval overlaps zero: "
            + ", ".join(uncertainty_overlap_metrics)
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
        and neutralization_corr_reduction
        >= thresholds.neutralization_material_corr_reduction
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
        neutralization_positive_reasons.append(
            "neutralized signal is weaker but more stable"
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
        reasons = _finalize_reasons(
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
        reasons = _finalize_reasons(
            uncertainty_positive,
            neutralization_positive_reasons,
            positives_without_uncertainty,
            max_items=6,
        )
    elif positive_score >= 3 and not has_weak_core_signal:
        label = "Promising but fragile"
        reasons = _finalize_reasons(
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
        reasons = _finalize_reasons(
            weak_evidence,
            neutralization_concern_reasons,
            concerns,
            neutralization_positive_reasons,
            positives,
            max_items=5,
        )
    else:
        label = "Mixed evidence"
        reasons = _finalize_reasons(
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


def _finalize_reasons(*groups: Sequence[str], max_items: int) -> tuple[str, ...]:
    merged: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for reason in group:
            token = reason.strip()
            if not token or token in seen:
                continue
            merged.append(token)
            seen.add(token)
            if len(merged) >= max_items:
                return tuple(merged)
    return tuple(merged)
