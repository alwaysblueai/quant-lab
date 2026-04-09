"""Raw-vs-neutralized comparison helpers for Level 1/2 research outputs."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import median

from alpha_lab.research_evaluation_config import (
    DEFAULT_RESEARCH_EVALUATION_CONFIG,
    NeutralizationComparisonConfig,
)

PRESERVES_EVIDENCE_FLAG = "neutralization preserves most evidence"
MODERATE_WEAKENING_FLAG = "neutralization moderately weakens evidence"
MATERIAL_REDUCTION_FLAG = "neutralization materially reduces independent evidence"
EXPOSURE_DRIVEN_FLAG = "raw signal appears exposure-driven"
WEAKER_BUT_STABLER_FLAG = "neutralization improves stability despite weaker raw performance"


NeutralizationComparisonThresholds = NeutralizationComparisonConfig
DEFAULT_NEUTRALIZATION_COMPARISON_THRESHOLDS = (
    DEFAULT_RESEARCH_EVALUATION_CONFIG.neutralization_comparison
)


@dataclass(frozen=True)
class NeutralizationComparison:
    """Compact, auditable raw-vs-neutralized evidence comparison."""

    raw: dict[str, float | int | None]
    neutralized: dict[str, float | int | None]
    delta: dict[str, float | int | None]
    interpretation_flags: tuple[str, ...]
    interpretation_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "raw": dict(self.raw),
            "neutralized": dict(self.neutralized),
            "delta": dict(self.delta),
            "interpretation_flags": list(self.interpretation_flags),
            "interpretation_reasons": list(self.interpretation_reasons),
        }


def build_raw_vs_neutralized_comparison(
    raw_metrics: Mapping[str, object],
    neutralized_metrics: Mapping[str, object],
    *,
    neutralization_mean_corr_reduction: float | None = None,
    thresholds: NeutralizationComparisonThresholds = DEFAULT_NEUTRALIZATION_COMPARISON_THRESHOLDS,
) -> NeutralizationComparison:
    """Compare raw and neutralized evidence using existing core diagnostics."""

    raw = _extract_evidence(raw_metrics)
    neutralized = _extract_evidence(neutralized_metrics)
    delta = _build_delta(raw, neutralized)

    flags, reasons = _interpret_comparison(
        raw=raw,
        neutralized=neutralized,
        delta=delta,
        neutralization_mean_corr_reduction=neutralization_mean_corr_reduction,
        thresholds=thresholds,
    )
    return NeutralizationComparison(
        raw=raw,
        neutralized=neutralized,
        delta=delta,
        interpretation_flags=flags,
        interpretation_reasons=reasons,
    )


def _extract_evidence(metrics: Mapping[str, object]) -> dict[str, float | int | None]:
    uncertainty_flags = _text_tokens(metrics.get("uncertainty_flags"))
    rolling_instability_flags = _text_tokens(metrics.get("rolling_instability_flags"))

    rolling_shares = [
        _metric(metrics, "rolling_ic_positive_share"),
        _metric(metrics, "rolling_rank_ic_positive_share"),
        _metric(metrics, "rolling_long_short_positive_share"),
    ]
    rolling_mins = [
        _metric(metrics, "rolling_ic_min_mean"),
        _metric(metrics, "rolling_rank_ic_min_mean"),
        _metric(metrics, "rolling_long_short_min_mean"),
    ]

    ic_valid = _metric(metrics, "ic_valid_ratio")
    rank_ic_valid = _metric(metrics, "rank_ic_valid_ratio")

    return {
        "mean_ic": _metric(metrics, "mean_ic"),
        "mean_rank_ic": _metric(metrics, "mean_rank_ic"),
        "mean_long_short_return": _metric(metrics, "mean_long_short_return"),
        "ic_ir": _metric(metrics, "ic_ir"),
        "ic_valid_ratio": ic_valid,
        "rank_ic_valid_ratio": rank_ic_valid,
        "valid_ratio_min": _min_or_none((ic_valid, rank_ic_valid)),
        "eval_coverage_ratio_mean": _metric(metrics, "eval_coverage_ratio_mean"),
        "eval_coverage_ratio_min": _metric(metrics, "eval_coverage_ratio_min"),
        "rolling_ic_positive_share": rolling_shares[0],
        "rolling_rank_ic_positive_share": rolling_shares[1],
        "rolling_long_short_positive_share": rolling_shares[2],
        "rolling_positive_share_min": _min_or_none(tuple(rolling_shares)),
        "rolling_ic_min_mean": rolling_mins[0],
        "rolling_rank_ic_min_mean": rolling_mins[1],
        "rolling_long_short_min_mean": rolling_mins[2],
        "rolling_worst_mean_min": _min_or_none(tuple(rolling_mins)),
        "uncertainty_overlap_zero_count": _uncertainty_overlap_zero_count(metrics),
        "uncertainty_wide_count": _flag_count(uncertainty_flags, suffix="_ci_wide"),
        "uncertainty_unavailable_count": _flag_count(
            uncertainty_flags,
            suffix="_ci_unavailable",
        ),
        "rolling_instability_flag_count": len(rolling_instability_flags),
    }


def _build_delta(
    raw: Mapping[str, float | int | None],
    neutralized: Mapping[str, float | int | None],
) -> dict[str, float | int | None]:
    out: dict[str, float | int | None] = {}
    for key in raw:
        out[f"{key}_delta"] = _delta_value(raw.get(key), neutralized.get(key))
    return out


def _interpret_comparison(
    *,
    raw: Mapping[str, float | int | None],
    neutralized: Mapping[str, float | int | None],
    delta: Mapping[str, float | int | None],
    neutralization_mean_corr_reduction: float | None,
    thresholds: NeutralizationComparisonThresholds,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    mean_ic_delta = _to_float(delta.get("mean_ic_delta"))
    mean_rank_ic_delta = _to_float(delta.get("mean_rank_ic_delta"))
    mean_ls_delta = _to_float(delta.get("mean_long_short_return_delta"))
    ic_ir_delta = _to_float(delta.get("ic_ir_delta"))

    retention = _median_retention(
        raw=raw,
        neutralized=neutralized,
        keys=("mean_ic", "mean_rank_ic", "mean_long_short_return"),
    )

    preserve_loss_limits = (
        (mean_ic_delta is None or mean_ic_delta >= -thresholds.preserve_mean_ic_loss_max)
        and (
            mean_rank_ic_delta is None
            or mean_rank_ic_delta >= -thresholds.preserve_mean_rank_ic_loss_max
        )
        and (mean_ls_delta is None or mean_ls_delta >= -thresholds.preserve_long_short_loss_max)
    )
    preserves = preserve_loss_limits and (
        retention is None or retention >= thresholds.preserve_min_retention
    )

    material_loss_hits = sum(
        condition
        for condition in (
            mean_ic_delta is not None
            and mean_ic_delta <= -thresholds.material_mean_ic_loss,
            mean_rank_ic_delta is not None
            and mean_rank_ic_delta <= -thresholds.material_mean_rank_ic_loss,
            mean_ls_delta is not None
            and mean_ls_delta <= -thresholds.material_long_short_loss,
            ic_ir_delta is not None and ic_ir_delta <= -thresholds.material_ic_ir_loss,
        )
    )

    raw_positive_core = _positive_core_count(raw)
    neutralized_positive_core = _positive_core_count(neutralized)
    material = bool(
        (retention is not None and retention <= thresholds.material_max_retention)
        or material_loss_hits >= thresholds.material_loss_hit_count
        or (
            material_loss_hits >= thresholds.material_loss_hit_count_with_core_shift
            and raw_positive_core >= thresholds.exposure_raw_positive_core_min
            and neutralized_positive_core <= thresholds.exposure_neutralized_positive_core_max
        )
    )

    weaker_core = any(
        value is not None and value < 0.0
        for value in (mean_ic_delta, mean_rank_ic_delta, mean_ls_delta)
    )

    flags: list[str] = []
    reasons: list[str] = []

    if material:
        flags.append(MATERIAL_REDUCTION_FLAG)
        reasons.append(
            "core losses are substantial "
            f"(delta IC={_fmt_metric(mean_ic_delta)}, "
            f"delta RankIC={_fmt_metric(mean_rank_ic_delta)}, "
            f"delta L/S={_fmt_metric(mean_ls_delta)}, "
            f"retention={_fmt_metric(retention)})."
        )
    elif preserves:
        flags.append(PRESERVES_EVIDENCE_FLAG)
        reasons.append(
            "core evidence remains close to raw "
            f"(delta IC={_fmt_metric(mean_ic_delta)}, "
            f"delta RankIC={_fmt_metric(mean_rank_ic_delta)}, "
            f"delta L/S={_fmt_metric(mean_ls_delta)}, "
            f"retention={_fmt_metric(retention)})."
        )
    elif weaker_core:
        flags.append(MODERATE_WEAKENING_FLAG)
        reasons.append(
            "neutralization weakens signal but does not fully remove it "
            f"(delta IC={_fmt_metric(mean_ic_delta)}, "
            f"delta RankIC={_fmt_metric(mean_rank_ic_delta)}, "
            f"delta L/S={_fmt_metric(mean_ls_delta)}, "
            f"retention={_fmt_metric(retention)})."
        )

    corr_reduction = (
        neutralization_mean_corr_reduction
        if neutralization_mean_corr_reduction is not None
        and math.isfinite(neutralization_mean_corr_reduction)
        else None
    )
    if (
        material
        and raw_positive_core >= thresholds.exposure_raw_positive_core_min
        and neutralized_positive_core <= thresholds.exposure_neutralized_positive_core_max
    ):
        if corr_reduction is None or corr_reduction >= thresholds.exposure_corr_reduction_threshold:
            flags.append(EXPOSURE_DRIVEN_FLAG)
            reasons.append(
                "raw core metrics are positive but mostly disappear after neutralization "
                f"(raw positive core={raw_positive_core}, "
                f"neutralized positive core={neutralized_positive_core}, "
                f"mean corr reduction={_fmt_metric(corr_reduction)})."
            )

    stability_improved = _stability_improved(delta, thresholds=thresholds)
    if weaker_core and stability_improved:
        share_delta = _to_float(delta.get("rolling_positive_share_min_delta"))
        worst_mean_delta = _to_float(delta.get("rolling_worst_mean_min_delta"))
        instability_flag_delta = _to_float(delta.get("rolling_instability_flag_count_delta"))
        flags.append(WEAKER_BUT_STABLER_FLAG)
        reasons.append(
            "rolling stability improves while raw performance weakens "
            f"(delta rolling+ min={_fmt_metric(share_delta)}, "
            f"delta worst rolling mean={_fmt_metric(worst_mean_delta)}, "
            "delta rolling instability flags="
            f"{_fmt_metric(instability_flag_delta)})."
        )

    if not flags:
        # If no interpretation fired but deltas are available, default to moderate framing.
        if any(
            value is not None
            for value in (mean_ic_delta, mean_rank_ic_delta, mean_ls_delta, ic_ir_delta)
        ):
            flags.append(MODERATE_WEAKENING_FLAG)
            reasons.append(
                "raw-vs-neutralized comparison is mixed and should be reviewed "
                f"(delta IC={_fmt_metric(mean_ic_delta)}, "
                f"delta RankIC={_fmt_metric(mean_rank_ic_delta)}, "
                f"delta L/S={_fmt_metric(mean_ls_delta)}, "
                f"delta ICIR={_fmt_metric(ic_ir_delta)})."
            )

    return tuple(flags), tuple(reasons)


def _stability_improved(
    delta: Mapping[str, float | int | None],
    *,
    thresholds: NeutralizationComparisonThresholds,
) -> bool:
    share_gain = _to_float(delta.get("rolling_positive_share_min_delta"))
    worst_mean_gain = _to_float(delta.get("rolling_worst_mean_min_delta"))
    instability_drop = _to_float(delta.get("rolling_instability_flag_count_delta"))
    return bool(
        (share_gain is not None and share_gain >= thresholds.stability_share_gain_min)
        or (
            worst_mean_gain is not None
            and worst_mean_gain > thresholds.stability_worst_mean_gain_min
        )
        or (instability_drop is not None and instability_drop < 0.0)
    )


def _uncertainty_overlap_zero_count(metrics: Mapping[str, object]) -> int:
    ranges = (
        ("mean_ic_ci_lower", "mean_ic_ci_upper"),
        ("mean_rank_ic_ci_lower", "mean_rank_ic_ci_upper"),
        ("mean_long_short_return_ci_lower", "mean_long_short_return_ci_upper"),
    )
    count = 0
    for lower_key, upper_key in ranges:
        lower = _metric(metrics, lower_key)
        upper = _metric(metrics, upper_key)
        if lower is None or upper is None:
            continue
        if lower <= 0.0 <= upper:
            count += 1
    return count


def _flag_count(flags: Sequence[str], *, suffix: str) -> int:
    return sum(flag.endswith(suffix) for flag in flags)


def _median_retention(
    *,
    raw: Mapping[str, float | int | None],
    neutralized: Mapping[str, float | int | None],
    keys: Sequence[str],
) -> float | None:
    ratios: list[float] = []
    for key in keys:
        raw_value = _to_float(raw.get(key))
        neutralized_value = _to_float(neutralized.get(key))
        if raw_value is None or neutralized_value is None:
            continue
        if math.isclose(raw_value, 0.0, rel_tol=0.0, abs_tol=1e-12):
            continue
        ratios.append(neutralized_value / raw_value)
    if not ratios:
        return None
    return float(median(ratios))


def _positive_core_count(metrics: Mapping[str, float | int | None]) -> int:
    count = 0
    for key in ("mean_ic", "mean_rank_ic", "mean_long_short_return"):
        value = _to_float(metrics.get(key))
        if value is not None and value > 0.0:
            count += 1
    return count


def _metric(metrics: Mapping[str, object], key: str) -> float | None:
    raw = metrics.get(key)
    return _to_float(raw)


def _to_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        out = float(value)
        return out if math.isfinite(out) else None
    text = str(value).strip()
    if not text:
        return None
    try:
        out = float(text)
    except ValueError:
        return None
    return out if math.isfinite(out) else None


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
    return tuple(token.strip() for token in text.split(delim) if token.strip())


def _min_or_none(values: Sequence[float | None]) -> float | None:
    finite = [value for value in values if value is not None]
    return min(finite) if finite else None


def _delta_value(
    raw_value: float | int | None,
    neutralized_value: float | int | None,
) -> float | int | None:
    if raw_value is None or neutralized_value is None:
        return None
    if isinstance(raw_value, int) and isinstance(neutralized_value, int):
        return neutralized_value - raw_value
    raw_num = _to_float(raw_value)
    neutralized_num = _to_float(neutralized_value)
    if raw_num is None or neutralized_num is None:
        return None
    return neutralized_num - raw_num


def _fmt_metric(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.6f}"
