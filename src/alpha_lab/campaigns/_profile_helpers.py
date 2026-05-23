"""Shared helpers extracted from `alpha_lab.campaigns.profile_comparison`.

This module exists to break the circular import between
`alpha_lab.campaigns.profile_comparison` (which needs the example to
drive `run_campaign_profile_comparison`) and
`alpha_lab.examples.profile_aware_campaign_level12` (which previously
carried byte-identical copies of all helpers below).

These helpers are intentionally private (underscore-prefixed) and pure:
they have no domain dependencies and operate on plain dicts/tuples,
so both consumers can import them without re-introducing the cycle.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from alpha_lab.key_metrics_contracts import LEVEL12_TRANSITION_TAXONOMY


@dataclass(frozen=True)
class CampaignCaseProfileSummary:
    """Per-(case, profile) outcome for the campaign profile comparison.

    Lives here (rather than in ``profile_comparison.py``) so that
    ``examples/profile_aware_campaign_level12.py`` can import it without
    re-introducing the cycle through ``campaigns.profile_comparison``.
    """

    case_name: str
    profile_name: str
    factor_verdict: str
    factor_verdict_reasons: tuple[str, ...]
    campaign_triage: str
    campaign_triage_reasons: tuple[str, ...]
    promotion_decision: str
    promotion_reasons: tuple[str, ...]
    promotion_blockers: tuple[str, ...]
    level12_transition_label: str
    level12_transition_reasons: tuple[str, ...]
    portfolio_validation_status: str
    portfolio_validation_recommendation: str
    portfolio_validation_major_risks: tuple[str, ...]
    status: str = "success"
    output_dir: Path | None = None
    run_manifest_path: Path | None = None
    metrics_path: Path | None = None
    summary_path: Path | None = None
    experiment_card_path: Path | None = None
    factor_definition_json_path: Path | None = None
    signal_validation_json_path: Path | None = None
    portfolio_recipe_json_path: Path | None = None
    backtest_result_json_path: Path | None = None
    case_report_path: Path | None = None


@dataclass(frozen=True)
class ProfileCampaignSummary:
    """One profile's worth of campaign output bundled with rank metadata."""

    profile_name: str
    case_summaries: tuple[CampaignCaseProfileSummary, ...]
    ranked_case_order: tuple[str, ...]
    campaign_output_dir: Path | None = None
    campaign_manifest_path: Path | None = None
    campaign_results_path: Path | None = None
    campaign_summary_path: Path | None = None
    campaign_index_path: Path | None = None
    campaign_report_path: Path | None = None


def _case_profile_lookup(
    profile_campaigns: tuple[ProfileCampaignSummary, ...],
) -> dict[str, dict[str, CampaignCaseProfileSummary]]:
    out: dict[str, dict[str, CampaignCaseProfileSummary]] = {}
    for campaign in profile_campaigns:
        for row in campaign.case_summaries:
            out.setdefault(row.case_name, {})[campaign.profile_name] = row
    return out


def _case_field_differences(
    profile_rows: dict[str, CampaignCaseProfileSummary],
    *,
    fields: tuple[str, ...],
) -> dict[str, dict[str, str]]:
    diffs: dict[str, dict[str, str]] = {}
    for field in fields:
        values = {
            profile: str(getattr(row, field)) for profile, row in sorted(profile_rows.items())
        }
        if len(set(values.values())) > 1:
            diffs[field] = values
    return diffs


_TRANSITION_STRENGTH_SCORE: dict[str, int] = {
    "Inconclusive transition": 0,
    "Fragile after promotion": 1,
    "Weakened at portfolio level": 2,
    "Confirmed at portfolio level": 3,
    "Improved at portfolio level": 4,
}
_TRANSITION_DIRECTION_STABLE = "stable"
_TRANSITION_DIRECTION_WEAKENED = "weakened"
_TRANSITION_DIRECTION_IMPROVED = "improved"
_TRANSITION_DIRECTION_UNKNOWN = "unknown"
_TRANSITION_DELTA_LABEL_STABLE = "transition_stable"
_TRANSITION_DELTA_LABEL_WEAKENED = "transition_weakened_under_stricter_profile"
_TRANSITION_DELTA_LABEL_IMPROVED = "transition_improved_under_profile_change"
_TRANSITION_DELTA_LABEL_MIXED = "transition_mixed_or_nonmonotonic"
_TRANSITION_DELTA_LABELS: tuple[str, ...] = (
    _TRANSITION_DELTA_LABEL_STABLE,
    _TRANSITION_DELTA_LABEL_WEAKENED,
    _TRANSITION_DELTA_LABEL_IMPROVED,
    _TRANSITION_DELTA_LABEL_MIXED,
)
def _sensitivity_label(changed_fields: list[str]) -> str:
    if not changed_fields:
        return "profile_stable"
    if len(changed_fields) >= 3:
        return "highly_profile_sensitive"
    return "profile_sensitive"
def _has_changed_field(row: dict[str, object], field: str) -> bool:
    changed_fields = row.get("changed_fields")
    if not isinstance(changed_fields, list):
        return False
    return any(str(item) == field for item in changed_fields)
def _promoted_only_under_looser_profiles(
    row: dict[str, object],
    *,
    exploratory_profile: str,
    baseline_profiles: tuple[str, ...],
) -> bool:
    profiles_obj = row.get("profiles", {})
    profiles = profiles_obj if isinstance(profiles_obj, dict) else {}

    exploratory = profiles.get(exploratory_profile)
    if not isinstance(exploratory, dict):
        return False
    exploratory_promoted = (
        str(exploratory.get("promotion_decision") or "").strip() == "Promote to Level 2"
    )
    if not exploratory_promoted:
        return False

    for profile in baseline_profiles:
        payload = profiles.get(profile)
        if not isinstance(payload, dict):
            continue
        if str(payload.get("promotion_decision") or "").strip() == "Promote to Level 2":
            return False
    return True
def _consistently_strong(row: dict[str, object]) -> bool:
    profiles_obj = row.get("profiles", {})
    profiles = profiles_obj if isinstance(profiles_obj, dict) else {}
    if not profiles:
        return False
    for payload in profiles.values():
        if not isinstance(payload, dict):
            return False
        if str(payload.get("factor_verdict") or "").strip() != "Strong candidate":
            return False
        if str(payload.get("promotion_decision") or "").strip() != "Promote to Level 2":
            return False
    return True
def _reason_rollup_for_transition_label(
    *,
    distribution: dict[str, object],
    transition_label: str,
) -> dict[str, object]:
    rollups_obj = distribution.get("reason_rollup_by_transition_label")
    rollups = rollups_obj if isinstance(rollups_obj, dict) else {}
    rollup_obj = rollups.get(transition_label)
    return rollup_obj if isinstance(rollup_obj, dict) else {}
def _adjacent_profile_pairs(profiles: list[str]) -> list[tuple[str, str]]:
    return [(profiles[idx], profiles[idx + 1]) for idx in range(max(0, len(profiles) - 1))]
def _transition_step_direction(from_label: str, to_label: str) -> str:
    if from_label not in _TRANSITION_STRENGTH_SCORE or to_label not in _TRANSITION_STRENGTH_SCORE:
        return _TRANSITION_DIRECTION_UNKNOWN
    if from_label == to_label:
        return _TRANSITION_DIRECTION_STABLE
    if _TRANSITION_STRENGTH_SCORE[to_label] < _TRANSITION_STRENGTH_SCORE[from_label]:
        return _TRANSITION_DIRECTION_WEAKENED
    return _TRANSITION_DIRECTION_IMPROVED


def _build_case_level12_transition_profile_delta(
    profile_rows: dict[str, CampaignCaseProfileSummary],
    *,
    profiles: list[str],
) -> dict[str, object]:
    profile_transition_labels = {
        profile: (
            profile_rows[profile].level12_transition_label if profile in profile_rows else "N/A"
        )
        for profile in profiles
    }
    profile_pair_directions: list[dict[str, str]] = []
    has_weakened = False
    has_improved = False
    has_unknown = False

    for from_profile, to_profile in _adjacent_profile_pairs(profiles):
        from_label = profile_transition_labels.get(from_profile, "N/A")
        to_label = profile_transition_labels.get(to_profile, "N/A")
        direction = _transition_step_direction(from_label, to_label)
        profile_pair_directions.append(
            {
                "from_profile": from_profile,
                "to_profile": to_profile,
                "from_label": from_label,
                "to_label": to_label,
                "direction": direction,
            }
        )
        if direction == _TRANSITION_DIRECTION_WEAKENED:
            has_weakened = True
        elif direction == _TRANSITION_DIRECTION_IMPROVED:
            has_improved = True
        elif direction == _TRANSITION_DIRECTION_UNKNOWN:
            has_unknown = True

    if has_unknown:
        delta_label = _TRANSITION_DELTA_LABEL_MIXED
    elif not profile_pair_directions or all(
        row["direction"] == _TRANSITION_DIRECTION_STABLE for row in profile_pair_directions
    ):
        delta_label = _TRANSITION_DELTA_LABEL_STABLE
    elif has_weakened and not has_improved:
        delta_label = _TRANSITION_DELTA_LABEL_WEAKENED
    elif has_improved and not has_weakened:
        delta_label = _TRANSITION_DELTA_LABEL_IMPROVED
    else:
        delta_label = _TRANSITION_DELTA_LABEL_MIXED

    return {
        "delta_label": delta_label,
        "profile_transition_labels": profile_transition_labels,
        "profile_pair_directions": profile_pair_directions,
    }


def _empty_transition_pair_count_matrix() -> dict[str, dict[str, int]]:
    return {
        from_label: {to_label: 0 for to_label in LEVEL12_TRANSITION_TAXONOMY}
        for from_label in LEVEL12_TRANSITION_TAXONOMY
    }
def _transition_pair_proportion_matrix(
    counts_by_from_to_label: dict[str, dict[str, int]],
    *,
    denominator: int,
) -> dict[str, dict[str, float]]:
    return {
        from_label: {
            to_label: (
                counts_by_from_to_label[from_label][to_label] / denominator
                if denominator > 0
                else 0.0
            )
            for to_label in LEVEL12_TRANSITION_TAXONOMY
        }
        for from_label in LEVEL12_TRANSITION_TAXONOMY
    }
def _format_reason_ratio(*, count: int, n_cases: int) -> str:
    safe_count = max(0, int(count))
    safe_n_cases = max(0, int(n_cases))
    return f"{safe_count}/{safe_n_cases}"
def _transition_profile_path_text(labels_obj: object, profiles_obj: object) -> str:
    labels = labels_obj if isinstance(labels_obj, dict) else {}
    profiles: list[object] = profiles_obj if isinstance(profiles_obj, list) else []
    parts: list[str] = []
    for profile_name in profiles:
        if not isinstance(profile_name, str):
            continue
        label = str(labels.get(profile_name) or "N/A")
        parts.append(f"`{profile_name}`=`{label}`")
    return " -> ".join(parts)
def _case_transition_delta_label(row: dict[str, object]) -> str:
    transition_delta_obj = row.get("level12_transition_profile_delta")
    transition_delta = transition_delta_obj if isinstance(transition_delta_obj, dict) else {}
    label = str(transition_delta.get("delta_label") or "")
    if label in _TRANSITION_DELTA_LABELS:
        return label
    return _TRANSITION_DELTA_LABEL_MIXED
def _pair_reduction_counts(*, counts_obj: object) -> tuple[int, int]:
    counts = counts_obj if isinstance(counts_obj, dict) else {}
    promotion_reduction = 0
    robustness_reduction = 0
    for from_label in LEVEL12_TRANSITION_TAXONOMY:
        from_counts_obj = counts.get(from_label)
        from_counts = from_counts_obj if isinstance(from_counts_obj, dict) else {}
        from_score = _TRANSITION_STRENGTH_SCORE.get(from_label)
        if from_score is None:
            continue
        for to_label in LEVEL12_TRANSITION_TAXONOMY:
            raw_count = from_counts.get(to_label, 0)
            count = raw_count if isinstance(raw_count, int) and raw_count >= 0 else 0
            if count <= 0 or to_label == from_label:
                continue
            to_score = _TRANSITION_STRENGTH_SCORE.get(to_label)
            if to_score is None:
                continue
            if to_label == "Inconclusive transition" and from_label != "Inconclusive transition":
                promotion_reduction += count
                continue
            if (
                to_label in {"Weakened at portfolio level", "Fragile after promotion"}
                and to_score < from_score
            ):
                robustness_reduction += count
    return promotion_reduction, robustness_reduction
def _dominant_reduction_mode(
    promotion_reduction_count: int,
    robustness_reduction_count: int,
) -> str:
    promotion = max(0, int(promotion_reduction_count))
    robustness = max(0, int(robustness_reduction_count))
    if promotion > 0 and robustness > 0:
        return "both"
    if promotion > 0:
        return "promotion"
    if robustness > 0:
        return "robustness"
    return "none"
def _to_int_value(value: object, *, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    token = str(value).strip() if value is not None else ""
    if not token:
        return default
    try:
        return int(float(token))
    except ValueError:
        return default
def _to_float_value(value: object, *, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    token = str(value).strip() if value is not None else ""
    if not token:
        return default
    try:
        return float(token)
    except ValueError:
        return default
