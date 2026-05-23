from __future__ import annotations

import datetime
import json
import logging
import os
import re
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from alpha_lab.artifact_contracts import validate_level12_artifact_payload
from alpha_lab.campaigns._profile_helpers import (
    _TRANSITION_DELTA_LABEL_IMPROVED,
    _TRANSITION_DELTA_LABEL_MIXED,
    _TRANSITION_DELTA_LABEL_STABLE,
    _TRANSITION_DELTA_LABEL_WEAKENED,
    _TRANSITION_DELTA_LABELS,
    _TRANSITION_DIRECTION_IMPROVED,
    _TRANSITION_DIRECTION_STABLE,
    _TRANSITION_DIRECTION_UNKNOWN,
    _TRANSITION_DIRECTION_WEAKENED,
    _TRANSITION_STRENGTH_SCORE,
    _adjacent_profile_pairs,
    _case_transition_delta_label,
    _consistently_strong,
    _dominant_reduction_mode,
    _empty_transition_pair_count_matrix,
    _format_reason_ratio,
    _has_changed_field,
    _pair_reduction_counts,
    _promoted_only_under_looser_profiles,
    _reason_rollup_for_transition_label,
    _sensitivity_label,
    _to_float_value,
    _to_int_value,
    _transition_pair_proportion_matrix,
    _transition_profile_path_text,
    _transition_step_direction,
)
from alpha_lab.campaigns.research_campaign_1 import (
    CampaignCaseResult,
    CampaignRunResult,
    load_research_campaign_1_config,
    run_research_campaign_1,
)
from alpha_lab.examples.profile_aware_campaign_level12 import (
    DEFAULT_PROFILE_AWARE_CAMPAIGN_LEVEL12_PROFILES,
    run_profile_aware_campaign_level12_example,
)
from alpha_lab.key_metrics_contracts import (
    CAMPAIGN_PROFILE_COMPARISON_FIELDS,
    LEVEL12_TRANSITION_SUPPORT_THRESHOLDS,
    LEVEL12_TRANSITION_TAXONOMY,
    project_campaign_profile_summary_metrics,
    project_level12_transition_distribution,
)
from alpha_lab.reporting.campaign_triage import campaign_rank_sort_key
from alpha_lab.reporting.display_helpers import format_text_list, parse_text_list
from alpha_lab.reporting.renderers import write_campaign_report
from alpha_lab.research_evaluation_config import (
    AVAILABLE_RESEARCH_EVALUATION_PROFILES,
    DEFAULT_RESEARCH_EVALUATION_CONFIG,
)

ComparisonSource = Literal["example", "campaign"]
ArtifactHintPathMode = Literal["relative", "absolute"]
PairMode = Literal["adjacent", "all_pairs"]

DEFAULT_CAMPAIGN_PROFILE_COMPARISON_OUTPUT_DIR = "dist/examples/profile_aware_campaign_level12"
DEFAULT_CAMPAIGN_PROFILE_COMPARISON_PROFILES: tuple[str, ...] = (
    DEFAULT_PROFILE_AWARE_CAMPAIGN_LEVEL12_PROFILES
)
DEFAULT_CAMPAIGN_PROFILE_COMPARISON_CONFIG_PATH = (
    "configs/campaigns/research_campaign_1/campaign.yaml"
)
DEFAULT_ARTIFACT_HINT_PATH_MODE: ArtifactHintPathMode = "relative"
DEFAULT_PAIR_MODE: PairMode = "adjacent"
logger = logging.getLogger(__name__)
_ABSOLUTE_PATH_TOKEN_RE = re.compile(r"(?:(?<=^)|(?<=[\s=:(;|]))(/[^\s|;,)`]+)")

_MIN_OBSERVED_CASES_PER_PROFILE_PAIR = 3
_MIN_OBSERVED_REASON_LABELS_PER_PROFILE_PAIR = 2
_MIN_CASES_PER_TRANSITION_LABEL_FOR_REASON_SHIFT = max(
    1,
    int(LEVEL12_TRANSITION_SUPPORT_THRESHOLDS["minimum_cases_per_transition_label"]),
)
_MIN_REASON_BUCKET_COUNT_FOR_SHIFT = max(
    1,
    int(LEVEL12_TRANSITION_SUPPORT_THRESHOLDS["minimum_reason_bucket_count_for_dominance"]),
)
_CAMPAIGN_PROFILE_COMPARISON_SUPPORT_THRESHOLDS: dict[str, int] = {
    **LEVEL12_TRANSITION_SUPPORT_THRESHOLDS,
    "minimum_observed_cases_per_profile_pair": _MIN_OBSERVED_CASES_PER_PROFILE_PAIR,
    "minimum_transition_labels_with_reason_evidence_per_profile_pair": (
        _MIN_OBSERVED_REASON_LABELS_PER_PROFILE_PAIR
    ),
    "minimum_cases_per_transition_label_for_reason_shift": (
        _MIN_CASES_PER_TRANSITION_LABEL_FOR_REASON_SHIFT
    ),
    "minimum_reason_bucket_count_for_shift": _MIN_REASON_BUCKET_COUNT_FOR_SHIFT,
}


@dataclass(frozen=True)
class CampaignComparisonCase:
    case_name: str
    case_description: str
    spec_path: Path | None


@dataclass(frozen=True)
class CampaignCaseProfileSummary:
    case_name: str
    profile_name: str
    status: str
    output_dir: Path | None
    run_manifest_path: Path | None
    metrics_path: Path | None
    summary_path: Path | None
    experiment_card_path: Path | None
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
    factor_definition_json_path: Path | None = None
    signal_validation_json_path: Path | None = None
    portfolio_recipe_json_path: Path | None = None
    backtest_result_json_path: Path | None = None


@dataclass(frozen=True)
class ProfileCampaignSummary:
    profile_name: str
    case_summaries: tuple[CampaignCaseProfileSummary, ...]
    ranked_case_order: tuple[str, ...]
    campaign_output_dir: Path | None
    campaign_manifest_path: Path | None
    campaign_results_path: Path | None
    campaign_summary_path: Path | None
    campaign_index_path: Path | None
    campaign_report_path: Path | None


@dataclass(frozen=True)
class CampaignProfileComparisonResult:
    source: ComparisonSource
    root_dir: Path
    profiles: tuple[str, ...]
    comparison_json_path: Path
    comparison_markdown_path: Path
    comparison_csv_path: Path
    pair_mode: PairMode
    stable_cases: tuple[str, ...]
    profile_sensitive_cases: tuple[str, ...]
    highly_profile_sensitive_cases: tuple[str, ...]
    transition_stable_cases: tuple[str, ...]
    transition_sensitive_cases: tuple[str, ...]
    case_evidence_index_case_count: int
    compact_comparison_summary: dict[str, object]
    compact_summary_lines: tuple[str, ...]
    transition_profile_delta_pair_summaries: tuple[str, ...]
    transition_reason_delta_pair_summaries: tuple[str, ...]


def run_campaign_profile_comparison(
    *,
    source: ComparisonSource = "example",
    output_root_dir: str | Path = DEFAULT_CAMPAIGN_PROFILE_COMPARISON_OUTPUT_DIR,
    profiles: tuple[str, ...] = DEFAULT_CAMPAIGN_PROFILE_COMPARISON_PROFILES,
    campaign_config: str | Path = DEFAULT_CAMPAIGN_PROFILE_COMPARISON_CONFIG_PATH,
    case_output_root_dir: str | Path | None = None,
    artifact_hint_path_mode: ArtifactHintPathMode = DEFAULT_ARTIFACT_HINT_PATH_MODE,
    pair_mode: PairMode = DEFAULT_PAIR_MODE,
    render_report: bool = True,
    render_overwrite: bool = False,
    clean_output: bool = True,
) -> CampaignProfileComparisonResult:
    selected_profiles = _normalize_profiles(profiles)
    normalized_hint_path_mode = _normalize_artifact_hint_path_mode(artifact_hint_path_mode)
    normalized_pair_mode = _normalize_pair_mode(pair_mode)
    root_dir = Path(output_root_dir).resolve()
    source_value = str(source).strip().lower()
    if source_value not in {"example", "campaign"}:
        raise ValueError(f"source must be 'example' or 'campaign'; received {source!r}")

    if source_value == "example":
        return _run_example_source(
            output_root_dir=root_dir,
            profiles=selected_profiles,
            artifact_hint_path_mode=normalized_hint_path_mode,
            pair_mode=normalized_pair_mode,
            render_report=render_report,
            clean_output=clean_output,
        )

    return _run_campaign_source(
        output_root_dir=root_dir,
        profiles=selected_profiles,
        campaign_config=campaign_config,
        case_output_root_dir=case_output_root_dir,
        artifact_hint_path_mode=normalized_hint_path_mode,
        pair_mode=normalized_pair_mode,
        render_report=render_report,
        render_overwrite=render_overwrite,
        clean_output=clean_output,
    )


def print_campaign_profile_comparison_summary(
    result: CampaignProfileComparisonResult,
) -> None:
    print("")
    print("  Workflow : campaign-profile-comparison")
    print("  Status   : success")
    print(f"  Source   : {result.source}")
    print(f"  Pair Mode: {result.pair_mode}")
    print(f"  Profiles : {list(result.profiles)}")
    print(f"  Output   : {result.root_dir}")
    print("  Inspect  :")
    print(f"    1) {result.comparison_markdown_path}")
    print(f"    2) {result.comparison_json_path}")
    print(f"    3) {result.comparison_csv_path}")
    if result.case_evidence_index_case_count > 0:
        print(
            "  Case Evidence Index : "
            "available in campaign_profile_comparison.json "
            f"(n={result.case_evidence_index_case_count})"
        )
    print(f"  Profile-Stable Cases     : {_tuple_or_none(result.stable_cases)}")
    print(f"  Profile-Sensitive Cases  : {_tuple_or_none(result.profile_sensitive_cases)}")
    print(f"  Highly Sensitive Cases   : {_tuple_or_none(result.highly_profile_sensitive_cases)}")
    print(f"  Transition-Stable Cases  : {_tuple_or_none(result.transition_stable_cases)}")
    print(f"  Transition-Sensitive Cases: {_tuple_or_none(result.transition_sensitive_cases)}")
    if result.compact_summary_lines:
        print("  Compact Comparison Summary:")
        for row in result.compact_summary_lines:
            print(f"    - {row}")
    if result.transition_profile_delta_pair_summaries:
        print("  Transition Delta Pairs   :")
        for row in result.transition_profile_delta_pair_summaries:
            print(f"    - {row}")
    if result.transition_reason_delta_pair_summaries:
        print("  Transition Reason Delta Pairs:")
        for row in result.transition_reason_delta_pair_summaries:
            print(f"    - {row}")


def print_campaign_profile_case_evidence(
    result: CampaignProfileComparisonResult,
    *,
    case_name: str,
) -> None:
    case_key = str(case_name).strip()
    if not case_key:
        raise ValueError("--show-case-evidence requires a non-empty case name")

    payload = _load_json(result.comparison_json_path)
    case_evidence_index_obj = payload.get("case_evidence_index")
    case_evidence_index = (
        case_evidence_index_obj if isinstance(case_evidence_index_obj, dict) else {}
    )
    if not case_evidence_index:
        raise ValueError(
            "case_evidence_index is unavailable in comparison output "
            f"({result.comparison_json_path})"
        )

    entry_obj = case_evidence_index.get(case_key)
    if not isinstance(entry_obj, dict):
        available_case_names = sorted(
            str(name) for name in case_evidence_index if str(name).strip()
        )
        raise ValueError(
            "requested case "
            f"{case_key!r} not found in case_evidence_index. "
            f"Available cases: {available_case_names}"
        )

    print("  Case Evidence Drill-Down:")
    print(f"    - case: {case_key}")
    print(f"    - profiles_observed: {_list_or_none(entry_obj.get('profiles_observed'))}")
    print(
        "    - sensitivity: "
        f"{entry_obj.get('profile_sensitivity')}; "
        f"profile_delta_label={entry_obj.get('profile_delta_label')}"
    )
    print(
        "    - factor_verdict_by_profile: "
        + _render_profile_value_map(entry_obj.get("factor_verdict_by_profile"))
    )
    print(
        "    - campaign_triage_by_profile: "
        + _render_profile_value_map(entry_obj.get("campaign_triage_by_profile"))
    )
    print(
        "    - promotion_decision_by_profile: "
        + _render_profile_value_map(entry_obj.get("promotion_decision_by_profile"))
    )
    print(
        "    - portfolio_robustness_by_profile: "
        + _render_profile_value_map(entry_obj.get("portfolio_robustness_by_profile"))
    )
    print(
        "    - level12_transition_label_by_profile: "
        + _render_profile_value_map(entry_obj.get("level12_transition_label_by_profile"))
    )
    print(
        "    - key_reason_hints_by_profile: "
        + _render_profile_list_map(entry_obj.get("key_reason_hints_by_profile"))
    )
    print(
        "    - artifact_pointer_hints_by_profile: "
        + _render_profile_list_map(entry_obj.get("artifact_pointer_hints_by_profile"))
    )


def _run_example_source(
    *,
    output_root_dir: Path,
    profiles: tuple[str, ...],
    artifact_hint_path_mode: ArtifactHintPathMode,
    pair_mode: PairMode,
    render_report: bool,
    clean_output: bool,
) -> CampaignProfileComparisonResult:
    result = run_profile_aware_campaign_level12_example(
        output_root_dir=output_root_dir,
        profiles=profiles,
        artifact_hint_path_mode=artifact_hint_path_mode,
        pair_mode=pair_mode,
        render_report=render_report,
        clean_output=clean_output,
    )
    from alpha_lab.reporting.renderers.campaign_profile_dashboard import (
        persist_workflow_closure_artifacts,
    )

    persist_workflow_closure_artifacts(result.comparison_json_path)
    payload = _load_json(result.comparison_json_path)
    return _result_from_payload(
        source="example",
        root_dir=result.root_dir,
        profiles=profiles,
        comparison_json_path=result.comparison_json_path,
        comparison_markdown_path=result.comparison_markdown_path,
        comparison_csv_path=result.comparison_csv_path,
        payload=payload,
    )


def _run_campaign_source(
    *,
    output_root_dir: Path,
    profiles: tuple[str, ...],
    campaign_config: str | Path,
    case_output_root_dir: str | Path | None,
    artifact_hint_path_mode: ArtifactHintPathMode,
    pair_mode: PairMode,
    render_report: bool,
    render_overwrite: bool,
    clean_output: bool,
) -> CampaignProfileComparisonResult:
    if clean_output and output_root_dir.exists():
        shutil.rmtree(output_root_dir)
    output_root_dir.mkdir(parents=True, exist_ok=True)

    campaign_config_path = Path(campaign_config).resolve()
    if not campaign_config_path.exists() or not campaign_config_path.is_file():
        raise FileNotFoundError(f"campaign config does not exist: {campaign_config_path}")
    config = load_research_campaign_1_config(campaign_config_path)

    base_case_output_root = (
        Path(case_output_root_dir).resolve() if case_output_root_dir is not None else None
    )

    cases = tuple(
        CampaignComparisonCase(
            case_name=case.case_name,
            case_description=(
                f"{case.package_type} case from campaign config '{campaign_config_path.name}'"
            ),
            spec_path=Path(case.spec_path),
        )
        for case in config.cases
    )

    profile_campaigns: list[ProfileCampaignSummary] = []
    for profile_name in profiles:
        profile_output_dir = output_root_dir / "runs" / profile_name
        profile_case_output_dir = (
            base_case_output_root / profile_name
            if base_case_output_root is not None
            else profile_output_dir / "cases"
        )
        campaign_result = run_research_campaign_1(
            config,
            output_root_dir=profile_output_dir,
            case_output_root_dir=profile_case_output_dir,
            evaluation_profile=profile_name,
            vault_export_mode="skip",
        )
        campaign_report_path: Path | None = None
        if render_report:
            try:
                campaign_report_path = write_campaign_report(
                    campaign_result.output_dir,
                    overwrite=render_overwrite,
                )
            except Exception as exc:
                logger.warning(
                    "Campaign report rendering failed for %s: %s",
                    campaign_result.output_dir,
                    exc,
                )
                campaign_report_path = None

        case_summaries = tuple(
            _to_case_profile_summary(row, profile_name=profile_name)
            for row in campaign_result.case_results
        )
        ranked_case_order = _ranked_case_order(campaign_result)
        profile_campaigns.append(
            ProfileCampaignSummary(
                profile_name=profile_name,
                case_summaries=case_summaries,
                ranked_case_order=ranked_case_order,
                campaign_output_dir=campaign_result.output_dir,
                campaign_manifest_path=campaign_result.artifact_paths["campaign_manifest"],
                campaign_results_path=campaign_result.artifact_paths["campaign_results"],
                campaign_summary_path=campaign_result.artifact_paths["campaign_summary"],
                campaign_index_path=campaign_result.artifact_paths["campaign_index"],
                campaign_report_path=campaign_report_path,
            )
        )

    comparison_payload = _build_campaign_comparison_payload(
        source="campaign",
        root_dir=output_root_dir,
        case_specs=cases,
        profile_campaigns=tuple(profile_campaigns),
        campaign_config_path=campaign_config_path,
        pair_mode=pair_mode,
    )
    rendered_payload = _render_artifact_hint_paths_in_payload(
        comparison_payload,
        root_dir=output_root_dir,
        path_mode=artifact_hint_path_mode,
    )
    rendered_payload["artifact_hint_path_mode"] = artifact_hint_path_mode
    rendered_payload["artifact_hint_path_base"] = "output_root_dir"

    comparison_json_path = output_root_dir / "campaign_profile_comparison.json"
    comparison_markdown_path = output_root_dir / "campaign_profile_comparison.md"
    comparison_csv_path = output_root_dir / "campaign_profile_case_matrix.csv"

    validate_level12_artifact_payload(
        rendered_payload,
        artifact_name=comparison_json_path.name,
        source=comparison_json_path,
    )
    comparison_json_path.write_text(
        json.dumps(rendered_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    comparison_markdown_path.write_text(
        _render_campaign_comparison_markdown(
            source="campaign",
            root_dir=output_root_dir,
            case_specs=cases,
            profile_campaigns=tuple(profile_campaigns),
            comparison_payload=rendered_payload,
        ),
        encoding="utf-8",
    )
    _write_case_matrix_csv(
        comparison_csv_path,
        case_specs=cases,
        profile_campaigns=tuple(profile_campaigns),
        comparison_payload=rendered_payload,
    )
    from alpha_lab.reporting.renderers.campaign_profile_dashboard import (
        persist_workflow_closure_artifacts,
    )

    persist_workflow_closure_artifacts(comparison_json_path)
    refreshed_payload = _load_json(comparison_json_path)

    return _result_from_payload(
        source="campaign",
        root_dir=output_root_dir,
        profiles=profiles,
        comparison_json_path=comparison_json_path,
        comparison_markdown_path=comparison_markdown_path,
        comparison_csv_path=comparison_csv_path,
        payload=refreshed_payload,
    )


def _result_from_payload(
    *,
    source: ComparisonSource,
    root_dir: Path,
    profiles: tuple[str, ...],
    comparison_json_path: Path,
    comparison_markdown_path: Path,
    comparison_csv_path: Path,
    payload: dict[str, object],
) -> CampaignProfileComparisonResult:
    pair_mode_obj = payload.get("pair_mode")
    pair_mode = _normalize_pair_mode(
        str(pair_mode_obj) if pair_mode_obj is not None else DEFAULT_PAIR_MODE
    )
    campaign_level_summary_obj = payload.get("campaign_level_summary", {})
    campaign_level_summary = (
        campaign_level_summary_obj if isinstance(campaign_level_summary_obj, dict) else {}
    )
    stable_cases = tuple(parse_text_list(campaign_level_summary.get("stable_cases")))
    highly_profile_sensitive_cases = tuple(
        parse_text_list(campaign_level_summary.get("highly_profile_sensitive"))
    )
    transition_stable_cases = tuple(
        parse_text_list(campaign_level_summary.get("transition_stable_cases"))
    )
    transition_sensitive_cases = tuple(
        parse_text_list(campaign_level_summary.get("transition_sensitive_cases"))
    )
    case_evidence_index_obj = payload.get("case_evidence_index")
    case_evidence_index = (
        case_evidence_index_obj if isinstance(case_evidence_index_obj, dict) else {}
    )
    compact_comparison_summary_obj = campaign_level_summary.get("compact_comparison_summary")
    compact_comparison_summary = (
        compact_comparison_summary_obj if isinstance(compact_comparison_summary_obj, dict) else {}
    )
    compact_summary_lines = tuple(_compact_comparison_summary_lines(compact_comparison_summary))
    transition_profile_delta_pair_summaries = _transition_profile_delta_pair_summaries(
        campaign_level_summary
    )
    transition_reason_delta_pair_summaries = _transition_reason_profile_delta_pair_summaries(
        campaign_level_summary
    )

    profile_sensitive_cases: list[str] = []
    case_comparison_obj = payload.get("case_comparison", [])
    case_comparison = case_comparison_obj if isinstance(case_comparison_obj, list) else []
    for row in case_comparison:
        if not isinstance(row, dict):
            continue
        if bool(row.get("is_profile_sensitive")):
            name = str(row.get("case_name") or "").strip()
            if name:
                profile_sensitive_cases.append(name)

    return CampaignProfileComparisonResult(
        source=source,
        root_dir=root_dir,
        profiles=profiles,
        comparison_json_path=comparison_json_path,
        comparison_markdown_path=comparison_markdown_path,
        comparison_csv_path=comparison_csv_path,
        pair_mode=pair_mode,
        stable_cases=stable_cases,
        profile_sensitive_cases=tuple(profile_sensitive_cases),
        highly_profile_sensitive_cases=highly_profile_sensitive_cases,
        transition_stable_cases=transition_stable_cases,
        transition_sensitive_cases=transition_sensitive_cases,
        case_evidence_index_case_count=len(case_evidence_index),
        compact_comparison_summary=compact_comparison_summary,
        compact_summary_lines=compact_summary_lines,
        transition_profile_delta_pair_summaries=transition_profile_delta_pair_summaries,
        transition_reason_delta_pair_summaries=transition_reason_delta_pair_summaries,
    )


def _to_case_profile_summary(
    row: CampaignCaseResult,
    *,
    profile_name: str,
) -> CampaignCaseProfileSummary:
    metrics = project_campaign_profile_summary_metrics(row.key_metrics)
    return CampaignCaseProfileSummary(
        case_name=row.case_name,
        profile_name=profile_name,
        status=row.status,
        output_dir=row.output_dir,
        run_manifest_path=row.run_manifest_path,
        metrics_path=row.metrics_path,
        summary_path=row.summary_path,
        experiment_card_path=row.experiment_card_path,
        factor_verdict=metrics["factor_verdict"] or "N/A",
        factor_verdict_reasons=metrics["factor_verdict_reasons"],
        campaign_triage=metrics["campaign_triage"] or "N/A",
        campaign_triage_reasons=metrics["campaign_triage_reasons"],
        promotion_decision=metrics["promotion_decision"] or "N/A",
        promotion_reasons=metrics["promotion_reasons"],
        promotion_blockers=metrics["promotion_blockers"],
        level12_transition_label=metrics["level12_transition_label"] or "N/A",
        level12_transition_reasons=metrics["level12_transition_reasons"],
        portfolio_validation_status=metrics["portfolio_validation_status"] or "N/A",
        portfolio_validation_recommendation=(
            metrics["portfolio_validation_recommendation"] or "N/A"
        ),
        portfolio_validation_major_risks=metrics["portfolio_validation_major_risks"],
        factor_definition_json_path=row.factor_definition_json_path,
        signal_validation_json_path=row.signal_validation_json_path,
        portfolio_recipe_json_path=row.portfolio_recipe_json_path,
        backtest_result_json_path=row.backtest_result_json_path,
    )


def _ranked_case_order(run_result: CampaignRunResult) -> tuple[str, ...]:
    return tuple(
        row.case_name
        for row in sorted(
            run_result.case_results,
            key=lambda row: campaign_rank_sort_key(
                row.case_name,
                status=row.status,
                metrics=row.key_metrics,
            ),
        )
    )


def _build_campaign_comparison_payload(
    *,
    source: ComparisonSource,
    root_dir: Path,
    case_specs: tuple[CampaignComparisonCase, ...],
    profile_campaigns: tuple[ProfileCampaignSummary, ...],
    campaign_config_path: Path | None,
    pair_mode: PairMode = DEFAULT_PAIR_MODE,
) -> dict[str, object]:
    profiles = [row.profile_name for row in profile_campaigns]
    case_lookup = _case_profile_lookup(profile_campaigns)
    transition_distribution_by_profile = {
        campaign.profile_name: project_level12_transition_distribution(
            [
                {
                    "case_name": row.case_name,
                    "level12_transition_label": row.level12_transition_label,
                    "level12_transition_reasons": list(row.level12_transition_reasons),
                    "artifact_pointer": _preferred_case_artifact_pointer(row),
                }
                for row in campaign.case_summaries
            ]
        )
        for campaign in profile_campaigns
    }
    transition_profile_delta_matrix = _build_level12_transition_profile_delta_matrix(
        case_specs=case_specs,
        case_lookup=case_lookup,
        profiles=profiles,
        pair_mode=pair_mode,
    )
    transition_reason_profile_delta_matrix = _build_level12_transition_reason_profile_delta_matrix(
        transition_distribution_by_profile=transition_distribution_by_profile,
        profiles=profiles,
        pair_mode=pair_mode,
    )

    case_comparison: list[dict[str, object]] = []
    for case in case_specs:
        profile_map = case_lookup.get(case.case_name, {})
        transition_profile_delta = _build_case_level12_transition_profile_delta(
            profile_map,
            profiles=profiles,
        )
        field_differences = _case_field_differences(
            profile_map,
            fields=CAMPAIGN_PROFILE_COMPARISON_FIELDS,
        )
        changed_fields = sorted(field_differences)
        is_profile_sensitive = bool(changed_fields)

        case_comparison.append(
            {
                "case_name": case.case_name,
                "case_description": case.case_description,
                "profile_sensitivity": _sensitivity_label(changed_fields),
                "is_profile_sensitive": is_profile_sensitive,
                "changed_fields": changed_fields,
                "field_differences": field_differences,
                "level12_transition_profile_delta": transition_profile_delta,
                "profiles": {
                    profile: _case_profile_payload(profile_map.get(profile)) for profile in profiles
                },
            }
        )

    field_change_index = {
        field: [row["case_name"] for row in case_comparison if _has_changed_field(row, field)]
        for field in CAMPAIGN_PROFILE_COMPARISON_FIELDS
    }
    stable_cases = [
        row["case_name"] for row in case_comparison if not bool(row.get("is_profile_sensitive"))
    ]
    profile_sensitive_cases = [
        row["case_name"] for row in case_comparison if bool(row.get("is_profile_sensitive"))
    ]
    promoted_only_under_looser_profiles = [
        row["case_name"]
        for row in case_comparison
        if _promoted_only_under_looser_profiles(
            row,
            exploratory_profile="exploratory_screening",
            baseline_profiles=("default_research", "stricter_research"),
        )
    ]
    consistently_strong = [row["case_name"] for row in case_comparison if _consistently_strong(row)]
    highly_profile_sensitive = [
        row["case_name"]
        for row in case_comparison
        if row.get("profile_sensitivity") == "highly_profile_sensitive"
    ]
    transition_stable_cases = [
        case_name
        for row in case_comparison
        if _case_transition_delta_label(row) == _TRANSITION_DELTA_LABEL_STABLE
        for case_name in [str(row.get("case_name") or "").strip()]
        if case_name
    ]
    transition_sensitive_cases = [
        case_name
        for row in case_comparison
        if _case_transition_delta_label(row) != _TRANSITION_DELTA_LABEL_STABLE
        for case_name in [str(row.get("case_name") or "").strip()]
        if case_name
    ]
    transition_delta_label_counts = {
        label: sum(1 for row in case_comparison if _case_transition_delta_label(row) == label)
        for label in _TRANSITION_DELTA_LABELS
    }
    case_evidence_index = _build_case_evidence_index(
        case_comparison=case_comparison,
        profiles=profiles,
    )
    compact_comparison_summary = _build_compact_comparison_summary(
        case_comparison=case_comparison,
        profiles=profiles,
        transition_distribution_by_profile=transition_distribution_by_profile,
        transition_profile_delta_matrix=transition_profile_delta_matrix,
        transition_reason_profile_delta_matrix=transition_reason_profile_delta_matrix,
        transition_stable_cases=transition_stable_cases,
        transition_sensitive_cases=transition_sensitive_cases,
        root_dir=root_dir,
    )

    profile_runs = [
        {
            "profile_name": campaign.profile_name,
            "ranked_case_order": list(campaign.ranked_case_order),
            "level12_transition_distribution": transition_distribution_by_profile[
                campaign.profile_name
            ],
            "campaign_output_dir": (
                str(campaign.campaign_output_dir)
                if campaign.campaign_output_dir is not None
                else None
            ),
            "campaign_artifacts": {
                "campaign_manifest": (
                    str(campaign.campaign_manifest_path)
                    if campaign.campaign_manifest_path is not None
                    else None
                ),
                "campaign_results": (
                    str(campaign.campaign_results_path)
                    if campaign.campaign_results_path is not None
                    else None
                ),
                "campaign_summary": (
                    str(campaign.campaign_summary_path)
                    if campaign.campaign_summary_path is not None
                    else None
                ),
                "campaign_index": (
                    str(campaign.campaign_index_path)
                    if campaign.campaign_index_path is not None
                    else None
                ),
                "campaign_report": (
                    str(campaign.campaign_report_path)
                    if campaign.campaign_report_path is not None
                    else None
                ),
            },
            "case_rows": [_profile_case_row(row) for row in campaign.case_summaries],
        }
        for campaign in profile_campaigns
    ]

    return {
        "schema_version": "1.0.0",
        "workflow_name": "campaign_profile_comparison",
        "source": source,
        "pair_mode": pair_mode,
        "generated_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "output_root_dir": str(root_dir),
        "campaign_config_path": str(campaign_config_path) if campaign_config_path else None,
        "profiles": profiles,
        "default_profile": DEFAULT_RESEARCH_EVALUATION_CONFIG.profile_name,
        "cases": [
            {
                "case_name": row.case_name,
                "case_description": row.case_description,
                "spec_path": str(row.spec_path) if row.spec_path is not None else None,
            }
            for row in case_specs
        ],
        "profile_runs": profile_runs,
        "case_comparison": case_comparison,
        "field_change_index": field_change_index,
        "case_evidence_index": case_evidence_index,
        "campaign_level_summary": {
            "pair_mode": pair_mode,
            "support_thresholds": dict(_CAMPAIGN_PROFILE_COMPARISON_SUPPORT_THRESHOLDS),
            "stable_cases": stable_cases,
            "profile_sensitive_cases": profile_sensitive_cases,
            "promoted_only_under_looser_profiles": promoted_only_under_looser_profiles,
            "consistently_strong": consistently_strong,
            "highly_profile_sensitive": highly_profile_sensitive,
            "level12_transition_distribution_by_profile": transition_distribution_by_profile,
            "level12_transition_profile_delta_matrix": transition_profile_delta_matrix,
            "level12_transition_reason_profile_delta_matrix": (
                transition_reason_profile_delta_matrix
            ),
            "transition_stable_cases": transition_stable_cases,
            "transition_sensitive_cases": transition_sensitive_cases,
            "transition_delta_label_counts": transition_delta_label_counts,
            "compact_comparison_summary": compact_comparison_summary,
        },
    }


def _render_campaign_comparison_markdown(
    *,
    source: ComparisonSource,
    root_dir: Path,
    case_specs: tuple[CampaignComparisonCase, ...],
    profile_campaigns: tuple[ProfileCampaignSummary, ...],
    comparison_payload: dict[str, object],
) -> str:
    lines = [
        "# Campaign Profile Comparison (Level 1/2)",
        "",
        f"- Source: `{source}`",
        f"- Output root: `{root_dir}`",
        f"- Profiles: `{comparison_payload.get('profiles')}`",
        "",
        "## Campaign Cases",
        "",
    ]
    for case in case_specs:
        lines.append(f"- `{case.case_name}`: {case.case_description}")

    lines += ["", "## Per-Profile View", ""]
    for campaign in profile_campaigns:
        lines += [f"### {campaign.profile_name}", ""]
        lines.append(
            "| Rank | Case | Status | Factor Verdict | Campaign Triage | "
            "Level 2 Promotion | L1->L2 Transition | Portfolio Validation Recommendation |"
        )
        lines.append("|---:|---|---|---|---|---|---|---|")
        rank_map = {name: idx + 1 for idx, name in enumerate(campaign.ranked_case_order)}
        by_case = {row.case_name: row for row in campaign.case_summaries}
        for case_name in campaign.ranked_case_order:
            row = by_case[case_name]
            lines.append(
                "| "
                f"{rank_map.get(case_name, 'N/A')} | "
                f"{row.case_name} | "
                f"{row.status} | "
                f"{row.factor_verdict} | "
                f"{row.campaign_triage} | "
                f"{row.promotion_decision} | "
                f"{row.level12_transition_label} | "
                f"{row.portfolio_validation_recommendation} "
                "|"
            )
        lines.append("")

    lines += ["## Case-Level Profile Comparison", ""]
    case_comparison_obj = comparison_payload.get("case_comparison", [])
    case_comparison = case_comparison_obj if isinstance(case_comparison_obj, list) else []
    for row in case_comparison:
        if not isinstance(row, dict):
            continue
        case_name = str(row.get("case_name") or "N/A")
        sensitivity = str(row.get("profile_sensitivity") or "profile_sensitive")
        changed_fields_obj = row.get("changed_fields", [])
        changed_fields = changed_fields_obj if isinstance(changed_fields_obj, list) else []

        lines.append(f"### {case_name}")
        lines.append("")
        lines.append(f"- Sensitivity: `{sensitivity}`")
        transition_delta_obj = row.get("level12_transition_profile_delta")
        transition_delta = transition_delta_obj if isinstance(transition_delta_obj, dict) else {}
        transition_delta_label = str(
            transition_delta.get("delta_label") or _TRANSITION_DELTA_LABEL_MIXED
        )
        lines.append(f"- L1->L2 transition delta: `{transition_delta_label}`")
        transition_path = _transition_profile_path_text(
            transition_delta.get("profile_transition_labels"),
            comparison_payload.get("profiles"),
        )
        if transition_path:
            lines.append(f"- L1->L2 transition path: {transition_path}")
        if changed_fields:
            lines.append(
                "- Changed fields: " + ", ".join(f"`{str(field)}`" for field in changed_fields)
            )
        else:
            lines.append("- Changed fields: none")

        profiles_obj = row.get("profiles", {})
        profile_map = profiles_obj if isinstance(profiles_obj, dict) else {}
        for profile_name, payload in sorted(profile_map.items()):
            if not isinstance(payload, dict):
                continue
            lines.append(
                "- "
                f"{profile_name}: status=`{payload.get('status')}`, "
                f"verdict=`{payload.get('factor_verdict')}`, "
                f"triage=`{payload.get('campaign_triage')}`, "
                f"promotion=`{payload.get('promotion_decision')}`, "
                f"transition=`{payload.get('level12_transition_label')}`, "
                f"portfolio_validation=`{payload.get('portfolio_validation_recommendation')}`"
            )
            reasons = payload.get("major_reasons", {})
            if isinstance(reasons, dict):
                blockers = reasons.get("promotion_blockers", [])
                blockers_text = (
                    ", ".join(str(x) for x in blockers) if isinstance(blockers, list) else ""
                )
                if blockers_text:
                    lines.append(f"- {profile_name} blockers: {blockers_text}")
        lines.append("")

    case_evidence_index_obj = comparison_payload.get("case_evidence_index")
    case_evidence_index = (
        case_evidence_index_obj if isinstance(case_evidence_index_obj, dict) else {}
    )
    if case_evidence_index:
        lines += ["## Case Evidence Index", ""]
        for case_name in sorted(case_evidence_index):
            entry_obj = case_evidence_index.get(case_name)
            entry = entry_obj if isinstance(entry_obj, dict) else {}
            lines.append(f"### {case_name}")
            lines.append("")
            lines.append(f"- Profiles observed: {_list_or_none(entry.get('profiles_observed'))}")
            lines.append(
                "- Sensitivity: "
                f"`{entry.get('profile_sensitivity')}`; "
                f"profile_delta_label=`{entry.get('profile_delta_label')}`"
            )
            changed_fields = _to_str_list(entry.get("changed_fields"), max_items=8)
            lines.append(
                "- Changed fields: "
                + (
                    ", ".join(f"`{field}`" for field in changed_fields)
                    if changed_fields
                    else "none"
                )
            )
            lines.append(
                "- Factor verdict by profile: "
                + _render_profile_value_map(entry.get("factor_verdict_by_profile"))
            )
            lines.append(
                "- Campaign triage by profile: "
                + _render_profile_value_map(entry.get("campaign_triage_by_profile"))
            )
            lines.append(
                "- Promotion decision by profile: "
                + _render_profile_value_map(entry.get("promotion_decision_by_profile"))
            )
            lines.append(
                "- Portfolio robustness by profile: "
                + _render_profile_value_map(entry.get("portfolio_robustness_by_profile"))
            )
            lines.append(
                "- L1->L2 transition label by profile: "
                + _render_profile_value_map(entry.get("level12_transition_label_by_profile"))
            )
            lines.append(
                "- Key reason hints by profile: "
                + _render_profile_list_map(entry.get("key_reason_hints_by_profile"))
            )
            lines.append(
                "- Artifact hints by profile: "
                + _render_profile_list_map(entry.get("artifact_pointer_hints_by_profile"))
            )
            lines.append("")

    summary_obj = comparison_payload.get("campaign_level_summary", {})
    summary = summary_obj if isinstance(summary_obj, dict) else {}
    lines += ["## Campaign-Level Interpretation", ""]
    support_thresholds_obj = summary.get("support_thresholds")
    support_thresholds = support_thresholds_obj if isinstance(support_thresholds_obj, dict) else {}
    if support_thresholds:
        threshold_tokens = [f"{key}={value}" for key, value in sorted(support_thresholds.items())]
        lines.append("- Minimum support thresholds: " + ", ".join(threshold_tokens))
    compact_summary_obj = summary.get("compact_comparison_summary")
    compact_summary = compact_summary_obj if isinstance(compact_summary_obj, dict) else {}
    compact_summary_lines = _compact_comparison_summary_lines(compact_summary)
    if compact_summary_lines:
        lines += ["### Compact Comparison Summary", ""]
        for summary_line in compact_summary_lines:
            lines.append(f"- {summary_line}")
        lines.append("")
    lines.append(f"- Cases stable across profiles: {_list_or_none(summary.get('stable_cases'))}")
    lines.append(
        f"- Cases profile-sensitive: {_list_or_none(summary.get('profile_sensitive_cases'))}"
    )
    lines.append(
        "- Cases promoted only under looser profiles: "
        f"{_list_or_none(summary.get('promoted_only_under_looser_profiles'))}"
    )
    lines.append(
        f"- Cases consistently strong: {_list_or_none(summary.get('consistently_strong'))}"
    )
    lines.append(
        "- Cases highly profile-sensitive: "
        f"{_list_or_none(summary.get('highly_profile_sensitive'))}"
    )
    lines.append(
        "- Transition-stable cases (L1->L2 labels): "
        f"{_list_or_none(summary.get('transition_stable_cases'))}"
    )
    lines.append(
        "- Transition-sensitive cases (L1->L2 labels): "
        f"{_list_or_none(summary.get('transition_sensitive_cases'))}"
    )
    transition_obj = summary.get("level12_transition_distribution_by_profile")
    transition_by_profile = transition_obj if isinstance(transition_obj, dict) else {}
    if transition_by_profile:
        lines.append("- L1->L2 transition distribution by profile:")
        payload_profiles_obj = comparison_payload.get("profiles", [])
        payload_profiles: list[object] = (
            payload_profiles_obj if isinstance(payload_profiles_obj, list) else []
        )
        for profile_name in payload_profiles:
            if not isinstance(profile_name, str):
                continue
            dist_obj = transition_by_profile.get(profile_name)
            dist = dist_obj if isinstance(dist_obj, dict) else {}
            if not dist:
                lines.append(f"- {profile_name}: unavailable")
                continue
            counts = dist.get("counts_by_transition_label")
            if not isinstance(counts, dict):
                counts = {}
            n_cases = dist.get("n_cases")
            interpretation = str(dist.get("interpretation") or "N/A")
            dist_support_suffix = _support_note_suffix(dist, label="support")
            lines.append(
                "- "
                f"{profile_name} (n={n_cases}): "
                f"Confirmed={counts.get('Confirmed at portfolio level', 0)}, "
                f"Weakened={counts.get('Weakened at portfolio level', 0)}, "
                f"Fragile={counts.get('Fragile after promotion', 0)}, "
                f"Improved={counts.get('Improved at portfolio level', 0)}, "
                f"Inconclusive={counts.get('Inconclusive transition', 0)}; "
                f"interpretation={interpretation}{dist_support_suffix}"
            )
            representative_case_names_by_label_obj = dist.get(
                "representative_case_names_by_transition_label"
            )
            representative_case_names_by_label = (
                representative_case_names_by_label_obj
                if isinstance(representative_case_names_by_label_obj, dict)
                else dist.get("representative_cases_by_transition_label")
            )
            if isinstance(representative_case_names_by_label, dict):
                representative_tokens: list[str] = []
                for transition_label in LEVEL12_TRANSITION_TAXONOMY:
                    label_case_names = _to_str_list(
                        representative_case_names_by_label.get(transition_label),
                        max_items=2,
                    )
                    if not label_case_names:
                        continue
                    representative_tokens.append(
                        f"{transition_label}: {', '.join(label_case_names)}"
                    )
                if representative_tokens:
                    lines.append(
                        "- "
                        f"{profile_name} representative cases: " + "; ".join(representative_tokens)
                    )
            artifact_pointers_by_label_obj = dist.get("artifact_pointers_by_transition_label")
            artifact_pointers_by_label = (
                artifact_pointers_by_label_obj
                if isinstance(artifact_pointers_by_label_obj, dict)
                else {}
            )
            if artifact_pointers_by_label:
                pointer_tokens: list[str] = []
                for transition_label in LEVEL12_TRANSITION_TAXONOMY:
                    label_pointers = _to_str_list(
                        artifact_pointers_by_label.get(transition_label),
                        max_items=1,
                    )
                    if not label_pointers:
                        continue
                    pointer_tokens.append(f"{transition_label}: {label_pointers[0]}")
                if pointer_tokens:
                    lines.append(f"- {profile_name} artifact hints: " + "; ".join(pointer_tokens))
            rollup_tokens = _transition_reason_rollup_tokens(dist)
            if rollup_tokens:
                lines.append(
                    f"- {profile_name} dominant transition reasons: " + "; ".join(rollup_tokens)
                )
    transition_delta_matrix_obj = summary.get("level12_transition_profile_delta_matrix")
    transition_delta_matrix = (
        transition_delta_matrix_obj if isinstance(transition_delta_matrix_obj, dict) else {}
    )
    if transition_delta_matrix:
        lines.append(
            "- L1->L2 transition profile-delta matrix "
            f"({_pair_scope_label_text(transition_delta_matrix.get('pair_mode'))}):"
        )
        pair_rows_obj = transition_delta_matrix.get("profile_pairs")
        pair_rows = pair_rows_obj if isinstance(pair_rows_obj, list) else []
        for pair_row in pair_rows:
            if not isinstance(pair_row, dict):
                continue
            from_profile = str(pair_row.get("from_profile") or "N/A")
            to_profile = str(pair_row.get("to_profile") or "N/A")
            observed = pair_row.get("n_cases_with_observed_transition_labels")
            missing = pair_row.get("n_cases_missing_transition_labels")
            stable_count = pair_row.get("stable_count")
            changed_count = pair_row.get("changed_count")
            lines.append(
                "- "
                f"{from_profile} -> {to_profile}: observed={observed}, "
                f"stable={stable_count}, changed={changed_count}, missing={missing}"
                f"{_support_note_suffix(pair_row)}"
            )
            representative_case_names = _to_str_list(
                pair_row.get("representative_case_names"),
                max_items=3,
            )
            if representative_case_names:
                lines.append(
                    "- "
                    f"{from_profile} -> {to_profile} representative cases: "
                    + ", ".join(representative_case_names)
                )
            artifact_hints = _to_str_list(pair_row.get("artifact_pointer_hints"), max_items=2)
            if artifact_hints:
                lines.append(
                    f"- {from_profile} -> {to_profile} artifact hints: " + "; ".join(artifact_hints)
                )
            nonzero_pairs = _render_nonzero_transition_pair_counts(
                pair_row.get("counts_by_from_to_label"),
                pair_row.get("proportions_by_from_to_label"),
                pair_row.get("representative_case_names_by_from_to_label"),
            )
            lines.append(f"- {from_profile} -> {to_profile} pair counts: {nonzero_pairs}")
    transition_reason_delta_matrix_obj = summary.get(
        "level12_transition_reason_profile_delta_matrix"
    )
    transition_reason_delta_matrix = (
        transition_reason_delta_matrix_obj
        if isinstance(transition_reason_delta_matrix_obj, dict)
        else {}
    )
    if transition_reason_delta_matrix:
        lines.append(
            "- L1->L2 dominant reason deltas by profile pair "
            f"({_pair_scope_label_text(transition_reason_delta_matrix.get('pair_mode'))}):"
        )
        pair_rows_obj = transition_reason_delta_matrix.get("profile_pairs")
        pair_rows = pair_rows_obj if isinstance(pair_rows_obj, list) else []
        for pair_row in pair_rows:
            if not isinstance(pair_row, dict):
                continue
            from_profile = str(pair_row.get("from_profile") or "N/A")
            to_profile = str(pair_row.get("to_profile") or "N/A")
            observed_labels = pair_row.get("n_transition_labels_with_observed_reasons")
            shifted_labels = pair_row.get("n_transition_labels_with_reason_shift")
            stable_labels = pair_row.get("n_transition_labels_reason_stable")
            delta_counts_obj = pair_row.get("reason_bucket_delta_counts")
            delta_counts = delta_counts_obj if isinstance(delta_counts_obj, dict) else {}
            tentative_shifted_labels = pair_row.get(
                "n_transition_labels_with_tentative_reason_shift"
            )
            lines.append(
                "- "
                f"{from_profile} -> {to_profile}: observed_labels={observed_labels}, "
                f"shifted_labels={shifted_labels}, stable_labels={stable_labels}, "
                f"tentative_shifted_labels={tentative_shifted_labels}, "
                f"added={delta_counts.get('added', 0)}, "
                f"removed={delta_counts.get('removed', 0)}, "
                f"increased={delta_counts.get('increased', 0)}, "
                f"decreased={delta_counts.get('decreased', 0)}"
                f"{_support_note_suffix(pair_row)}"
            )
            representative_case_names = _to_str_list(
                pair_row.get("representative_case_names"),
                max_items=3,
            )
            if representative_case_names:
                lines.append(
                    "- "
                    f"{from_profile} -> {to_profile} representative cases: "
                    + ", ".join(representative_case_names)
                )
            artifact_hints = _to_str_list(pair_row.get("artifact_pointer_hints"), max_items=2)
            if artifact_hints:
                lines.append(
                    f"- {from_profile} -> {to_profile} artifact hints: " + "; ".join(artifact_hints)
                )
            by_label_obj = pair_row.get("reason_delta_by_transition_label")
            by_label = by_label_obj if isinstance(by_label_obj, dict) else {}
            for transition_label in LEVEL12_TRANSITION_TAXONOMY:
                label_obj = by_label.get(transition_label)
                label_row = label_obj if isinstance(label_obj, dict) else {}
                from_reasons_obj = label_row.get("from_profile_dominant_reasons")
                from_reasons = from_reasons_obj if isinstance(from_reasons_obj, list) else []
                to_reasons_obj = label_row.get("to_profile_dominant_reasons")
                to_reasons = to_reasons_obj if isinstance(to_reasons_obj, list) else []
                deltas_obj = label_row.get("reason_bucket_deltas")
                deltas = deltas_obj if isinstance(deltas_obj, dict) else {}
                is_shifted = bool(label_row.get("is_reason_shifted"))
                is_tentative_shifted = bool(label_row.get("is_reason_shift_tentative"))
                if not from_reasons and not to_reasons:
                    continue
                from_tokens = _render_reason_stat_tokens(from_reasons)
                to_tokens = _render_reason_stat_tokens(to_reasons)
                delta_tokens = _render_reason_delta_bucket_tokens(deltas)
                label_support_suffix = _support_note_suffix(label_row)
                if is_shifted:
                    lines.append(
                        "- "
                        f"{from_profile} -> {to_profile} [{transition_label}]: "
                        f"{from_profile} dominant={from_tokens}; "
                        f"{to_profile} dominant={to_tokens}; "
                        f"shifts={delta_tokens}{label_support_suffix}"
                    )
                elif is_tentative_shifted:
                    lines.append(
                        "- "
                        f"{from_profile} -> {to_profile} [{transition_label}]: "
                        "reason shift observed, but only in a small number of cases; "
                        f"{from_profile} dominant={from_tokens}; "
                        f"{to_profile} dominant={to_tokens}; "
                        f"shifts={delta_tokens}{label_support_suffix}"
                    )
                else:
                    lines.append(
                        "- "
                        f"{from_profile} -> {to_profile} [{transition_label}]: "
                        f"dominant reasons stable; "
                        f"{from_profile}={from_tokens}; "
                        f"{to_profile}={to_tokens}{label_support_suffix}"
                    )

    lines += [
        "",
        "## Artifacts",
        "",
        "- `campaign_profile_comparison.md` (human-readable summary)",
        "- `campaign_profile_comparison.json` (machine-readable profile deltas)",
        "- `campaign_profile_case_matrix.csv` (flat case/profile matrix)",
        "",
    ]

    return "\n".join(lines) + "\n"


def _write_case_matrix_csv(
    path: Path,
    *,
    case_specs: tuple[CampaignComparisonCase, ...],
    profile_campaigns: tuple[ProfileCampaignSummary, ...],
    comparison_payload: dict[str, object],
) -> None:
    case_lookup = _case_profile_lookup(profile_campaigns)

    sensitivity_by_case: dict[str, str] = {}
    transition_delta_by_case: dict[str, str] = {}
    case_comparison_obj = comparison_payload.get("case_comparison", [])
    if isinstance(case_comparison_obj, list):
        for row in case_comparison_obj:
            if not isinstance(row, dict):
                continue
            case_name = str(row.get("case_name") or "").strip()
            if not case_name:
                continue
            sensitivity_by_case[case_name] = str(
                row.get("profile_sensitivity") or "profile_sensitive"
            )
            transition_delta_obj = row.get("level12_transition_profile_delta")
            transition_delta = (
                transition_delta_obj if isinstance(transition_delta_obj, dict) else {}
            )
            transition_delta_by_case[case_name] = str(
                transition_delta.get("delta_label") or _TRANSITION_DELTA_LABEL_MIXED
            )

    rows: list[dict[str, object]] = []
    for case in case_specs:
        by_profile = case_lookup.get(case.case_name, {})
        for profile_name, summary in sorted(by_profile.items()):
            rows.append(
                {
                    "case_name": case.case_name,
                    "profile_name": profile_name,
                    "status": summary.status,
                    "factor_verdict": summary.factor_verdict,
                    "campaign_triage": summary.campaign_triage,
                    "promotion_decision": summary.promotion_decision,
                    "level12_transition_label": summary.level12_transition_label,
                    "portfolio_validation_recommendation": (
                        summary.portfolio_validation_recommendation
                    ),
                    "promotion_blockers": "; ".join(summary.promotion_blockers),
                    "profile_sensitivity": sensitivity_by_case.get(case.case_name, "unknown"),
                    "level12_transition_delta_label": transition_delta_by_case.get(
                        case.case_name,
                        "unknown",
                    ),
                    "metrics_path": (
                        str(summary.metrics_path) if summary.metrics_path is not None else None
                    ),
                    "output_dir": (
                        str(summary.output_dir) if summary.output_dir is not None else None
                    ),
                }
            )

    pd.DataFrame(rows).sort_values(["case_name", "profile_name"], kind="mergesort").to_csv(
        path, index=False
    )


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


def _build_level12_transition_profile_delta_matrix(
    *,
    case_specs: tuple[CampaignComparisonCase, ...],
    case_lookup: dict[str, dict[str, CampaignCaseProfileSummary]],
    profiles: list[str],
    pair_mode: PairMode = DEFAULT_PAIR_MODE,
) -> dict[str, object]:
    profile_pairs = _profile_pairs(profiles, pair_mode=pair_mode)
    pair_rows: list[dict[str, object]] = []
    for from_profile, to_profile in profile_pairs:
        counts_by_from_to_label = _empty_transition_pair_count_matrix()
        representative_case_names_by_from_to_label = _empty_transition_pair_case_matrix()
        n_cases_compared = 0
        n_missing_labels = 0
        stable_count = 0
        changed_count = 0
        supporting_case_names: list[str] = []
        changed_case_names: list[str] = []
        stable_case_names: list[str] = []
        artifact_hint_by_case: dict[str, str] = {}

        for case in case_specs:
            profile_map = case_lookup.get(case.case_name, {})
            from_summary = profile_map.get(from_profile)
            to_summary = profile_map.get(to_profile)
            if from_summary is None or to_summary is None:
                continue
            n_cases_compared += 1
            from_label = from_summary.level12_transition_label
            to_label = to_summary.level12_transition_label
            if (
                from_label not in _TRANSITION_STRENGTH_SCORE
                or to_label not in _TRANSITION_STRENGTH_SCORE
            ):
                n_missing_labels += 1
                continue
            _append_unique_text(supporting_case_names, case.case_name, max_items=8)
            counts_by_from_to_label[from_label][to_label] += 1
            _append_unique_text(
                representative_case_names_by_from_to_label[from_label][to_label],
                case.case_name,
                max_items=3,
            )
            artifact_hint = _profile_pair_case_artifact_hint(
                case_name=case.case_name,
                from_profile=from_profile,
                from_summary=from_summary,
                to_profile=to_profile,
                to_summary=to_summary,
            )
            if artifact_hint is not None:
                artifact_hint_by_case[case.case_name] = artifact_hint
            if from_label == to_label:
                stable_count += 1
                _append_unique_text(stable_case_names, case.case_name, max_items=4)
            else:
                changed_count += 1
                _append_unique_text(changed_case_names, case.case_name, max_items=4)

        n_observed = stable_count + changed_count
        representative_case_names = list(changed_case_names)
        for case_name in stable_case_names:
            if len(representative_case_names) >= 3:
                break
            _append_unique_text(representative_case_names, case_name, max_items=3)
        artifact_pointer_hints = [
            artifact_hint_by_case[case_name]
            for case_name in representative_case_names
            if case_name in artifact_hint_by_case
        ][:3]
        pair_support = _support_annotation(
            support_count=n_observed,
            minimum_required_support=_MIN_OBSERVED_CASES_PER_PROFILE_PAIR,
            sparse_note="sparse transition evidence",
            tentative_note="tentative due to low support",
            supported_note="transition evidence is well supported",
        )
        pair_rows.append(
            {
                "from_profile": from_profile,
                "to_profile": to_profile,
                "profile_pair": f"{from_profile} -> {to_profile}",
                "n_cases_compared": n_cases_compared,
                "n_cases_with_observed_transition_labels": n_observed,
                "n_cases_missing_transition_labels": n_missing_labels,
                "stable_count": stable_count,
                "changed_count": changed_count,
                "stable_proportion": (stable_count / n_observed if n_observed > 0 else 0.0),
                "changed_proportion": (changed_count / n_observed if n_observed > 0 else 0.0),
                "minimum_support_thresholds": {
                    "minimum_observed_cases": _MIN_OBSERVED_CASES_PER_PROFILE_PAIR,
                },
                "support_count": n_observed,
                "minimum_required_support": _MIN_OBSERVED_CASES_PER_PROFILE_PAIR,
                **pair_support,
                "counts_by_from_to_label": counts_by_from_to_label,
                "representative_case_names_by_from_to_label": (
                    representative_case_names_by_from_to_label
                ),
                "proportions_by_from_to_label": _transition_pair_proportion_matrix(
                    counts_by_from_to_label,
                    denominator=n_observed,
                ),
                "representative_case_names": representative_case_names,
                "supporting_case_names": supporting_case_names,
                "artifact_pointer_hints": artifact_pointer_hints,
            }
        )

    return {
        "pair_mode": pair_mode,
        "profile_pair_scope": (
            "adjacent_profiles" if pair_mode == "adjacent" else "all_ordered_profile_pairs"
        ),
        "profile_pairs": pair_rows,
    }


def _build_level12_transition_reason_profile_delta_matrix(
    *,
    transition_distribution_by_profile: Mapping[str, Mapping[str, object]],
    profiles: list[str],
    pair_mode: PairMode = DEFAULT_PAIR_MODE,
) -> dict[str, object]:
    pair_rows: list[dict[str, object]] = []
    for from_profile, to_profile in _profile_pairs(profiles, pair_mode=pair_mode):
        from_distribution_obj = transition_distribution_by_profile.get(from_profile)
        from_distribution = from_distribution_obj if isinstance(from_distribution_obj, dict) else {}
        to_distribution_obj = transition_distribution_by_profile.get(to_profile)
        to_distribution = to_distribution_obj if isinstance(to_distribution_obj, dict) else {}
        reason_delta_by_transition_label: dict[str, dict[str, object]] = {}
        total_added = 0
        total_removed = 0
        total_increased = 0
        total_decreased = 0
        total_stable = 0
        n_observed_labels = 0
        n_shifted_labels = 0
        n_stable_labels = 0
        n_tentative_shifted_labels = 0
        representative_case_names_by_transition_label: dict[str, list[str]] = {}
        pair_representative_case_names: list[str] = []
        pair_supporting_case_names: list[str] = []
        pair_artifact_pointer_hints: list[str] = []

        for transition_label in LEVEL12_TRANSITION_TAXONOMY:
            from_rollup = _reason_rollup_for_transition_label(
                distribution=from_distribution,
                transition_label=transition_label,
            )
            to_rollup = _reason_rollup_for_transition_label(
                distribution=to_distribution,
                transition_label=transition_label,
            )
            (
                from_n_cases,
                from_dominant_reasons,
                from_reason_map,
            ) = _dominant_reason_stats_from_rollup(from_rollup)
            to_n_cases, to_dominant_reasons, to_reason_map = _dominant_reason_stats_from_rollup(
                to_rollup
            )
            from_representative_case_names = _to_str_list(
                from_rollup.get("representative_case_names"),
                max_items=4,
            )
            to_representative_case_names = _to_str_list(
                to_rollup.get("representative_case_names"),
                max_items=4,
            )
            representative_case_names = _merge_unique_text_lists(
                from_representative_case_names,
                to_representative_case_names,
                max_items=4,
            )
            from_supporting_case_names = _to_str_list(
                from_rollup.get("supporting_case_names"),
                max_items=6,
            )
            to_supporting_case_names = _to_str_list(
                to_rollup.get("supporting_case_names"),
                max_items=6,
            )
            supporting_case_names = _merge_unique_text_lists(
                from_supporting_case_names,
                to_supporting_case_names,
                max_items=6,
            )
            from_artifact_pointer_hints = _to_str_list(
                from_rollup.get("artifact_pointer_hints"),
                max_items=4,
            )
            to_artifact_pointer_hints = _to_str_list(
                to_rollup.get("artifact_pointer_hints"),
                max_items=4,
            )
            artifact_pointer_hints = _merge_unique_text_lists(
                from_artifact_pointer_hints,
                to_artifact_pointer_hints,
                max_items=4,
            )
            if from_n_cases > 0 or to_n_cases > 0:
                n_observed_labels += 1
                representative_case_names_by_transition_label[transition_label] = (
                    representative_case_names
                )
                _extend_unique_text(
                    pair_representative_case_names,
                    representative_case_names,
                    max_items=6,
                )
                _extend_unique_text(
                    pair_supporting_case_names,
                    supporting_case_names,
                    max_items=8,
                )
                _extend_unique_text(
                    pair_artifact_pointer_hints,
                    artifact_pointer_hints,
                    max_items=6,
                )

            reason_bucket_deltas = _build_reason_bucket_deltas(
                from_reason_map=from_reason_map,
                to_reason_map=to_reason_map,
                from_n_cases_with_label=from_n_cases,
                to_n_cases_with_label=to_n_cases,
            )
            added_rows = reason_bucket_deltas["added"]
            removed_rows = reason_bucket_deltas["removed"]
            increased_rows = reason_bucket_deltas["increased"]
            decreased_rows = reason_bucket_deltas["decreased"]
            stable_rows = reason_bucket_deltas["stable"]
            is_reason_shift_candidate = bool(
                added_rows or removed_rows or increased_rows or decreased_rows
            )
            label_support = _support_annotation(
                support_count=min(from_n_cases, to_n_cases),
                minimum_required_support=_MIN_CASES_PER_TRANSITION_LABEL_FOR_REASON_SHIFT,
                sparse_note="reason shift observed, but only in a small number of cases",
                tentative_note="tentative due to low support",
                supported_note="reason-shift evidence is well supported",
            )
            is_reason_shifted = bool(label_support["minimum_support_met"]) and bool(
                is_reason_shift_candidate
            )
            is_reason_shift_tentative = bool(is_reason_shift_candidate) and not bool(
                label_support["minimum_support_met"]
            )
            if from_n_cases > 0 or to_n_cases > 0:
                if is_reason_shifted:
                    n_shifted_labels += 1
                elif is_reason_shift_tentative:
                    n_tentative_shifted_labels += 1
                else:
                    n_stable_labels += 1

            total_added += len(added_rows)
            total_removed += len(removed_rows)
            total_increased += len(increased_rows)
            total_decreased += len(decreased_rows)
            total_stable += len(stable_rows)
            reason_delta_by_transition_label[transition_label] = {
                "profile_pair": f"{from_profile} -> {to_profile}",
                "from_profile_n_cases_with_label": from_n_cases,
                "to_profile_n_cases_with_label": to_n_cases,
                "from_profile_dominant_reasons": from_dominant_reasons,
                "to_profile_dominant_reasons": to_dominant_reasons,
                "reason_bucket_deltas": reason_bucket_deltas,
                "representative_case_names": representative_case_names,
                "supporting_case_names": supporting_case_names,
                "artifact_pointer_hints": artifact_pointer_hints,
                "is_reason_shifted": is_reason_shifted,
                "is_reason_shift_tentative": is_reason_shift_tentative,
                "minimum_support_thresholds": {
                    "minimum_cases_per_profile_for_shift": (
                        _MIN_CASES_PER_TRANSITION_LABEL_FOR_REASON_SHIFT
                    ),
                    "minimum_reason_bucket_count_for_shift": (_MIN_REASON_BUCKET_COUNT_FOR_SHIFT),
                },
                "support_count": min(from_n_cases, to_n_cases),
                "minimum_required_support": _MIN_CASES_PER_TRANSITION_LABEL_FOR_REASON_SHIFT,
                **label_support,
            }

        pair_support = _support_annotation(
            support_count=n_observed_labels,
            minimum_required_support=_MIN_OBSERVED_REASON_LABELS_PER_PROFILE_PAIR,
            sparse_note="sparse transition evidence",
            tentative_note="tentative due to low support",
            supported_note="reason-shift evidence is well supported",
        )
        pair_support_note = str(pair_support.get("support_note") or "").strip()
        if n_tentative_shifted_labels > 0 and n_shifted_labels <= 0:
            pair_support_note = "reason shift observed, but only in a small number of cases"

        pair_rows.append(
            {
                "from_profile": from_profile,
                "to_profile": to_profile,
                "profile_pair": f"{from_profile} -> {to_profile}",
                "n_transition_labels_with_observed_reasons": n_observed_labels,
                "n_transition_labels_with_reason_shift": n_shifted_labels,
                "n_transition_labels_reason_stable": n_stable_labels,
                "n_transition_labels_with_tentative_reason_shift": n_tentative_shifted_labels,
                "minimum_support_thresholds": {
                    "minimum_observed_transition_labels_with_reason_evidence": (
                        _MIN_OBSERVED_REASON_LABELS_PER_PROFILE_PAIR
                    ),
                    "minimum_cases_per_transition_label_for_shift": (
                        _MIN_CASES_PER_TRANSITION_LABEL_FOR_REASON_SHIFT
                    ),
                    "minimum_reason_bucket_count_for_shift": _MIN_REASON_BUCKET_COUNT_FOR_SHIFT,
                },
                "support_count": n_observed_labels,
                "minimum_required_support": _MIN_OBSERVED_REASON_LABELS_PER_PROFILE_PAIR,
                **pair_support,
                "support_note": pair_support_note or pair_support["support_note"],
                "confidence_note": pair_support_note or pair_support["confidence_note"],
                "representative_case_names_by_transition_label": (
                    representative_case_names_by_transition_label
                ),
                "representative_case_names": pair_representative_case_names,
                "supporting_case_names": pair_supporting_case_names,
                "artifact_pointer_hints": pair_artifact_pointer_hints,
                "reason_bucket_delta_counts": {
                    "added": total_added,
                    "removed": total_removed,
                    "increased": total_increased,
                    "decreased": total_decreased,
                    "stable": total_stable,
                },
                "reason_delta_by_transition_label": reason_delta_by_transition_label,
            }
        )

    return {
        "pair_mode": pair_mode,
        "profile_pair_scope": (
            "adjacent_profiles" if pair_mode == "adjacent" else "all_ordered_profile_pairs"
        ),
        "profile_pairs": pair_rows,
    }


def _dominant_reason_stats_from_rollup(
    rollup: dict[str, object],
) -> tuple[int, list[dict[str, object]], dict[str, dict[str, object]]]:
    raw_n_cases = rollup.get("n_cases_with_label")
    n_cases_with_label = raw_n_cases if isinstance(raw_n_cases, int) and raw_n_cases >= 0 else 0
    dominant_reasons_obj = rollup.get("dominant_reasons")
    top_reasons_obj = (
        dominant_reasons_obj
        if isinstance(dominant_reasons_obj, list)
        else rollup.get("top_reasons")
    )
    top_reasons = top_reasons_obj if isinstance(top_reasons_obj, list) else []
    dominant_reasons: list[dict[str, object]] = []
    reason_map: dict[str, dict[str, object]] = {}
    for row in top_reasons:
        if not isinstance(row, dict):
            continue
        reason = str(row.get("reason") or "").strip()
        if not reason:
            continue
        raw_count = row.get("count")
        count = raw_count if isinstance(raw_count, int) and raw_count >= 0 else 0
        raw_prop = row.get("proportion_of_label_cases")
        proportion = (
            float(raw_prop)
            if isinstance(raw_prop, int | float) and not isinstance(raw_prop, bool)
            else 0.0
        )
        supporting_case_names = _to_str_list(row.get("supporting_case_names"), max_items=6)
        artifact_pointer_hints = _to_str_list(row.get("artifact_pointer_hints"), max_items=4)
        dominant_reasons.append(
            {
                "reason": reason,
                "count": count,
                "n_cases_with_label": n_cases_with_label,
                "proportion_of_label_cases": proportion,
                "supporting_case_names": supporting_case_names,
                "artifact_pointer_hints": artifact_pointer_hints,
            }
        )
        reason_map[reason] = {
            "count": count,
            "proportion_of_label_cases": proportion,
            "supporting_case_names": supporting_case_names,
            "artifact_pointer_hints": artifact_pointer_hints,
        }
    return n_cases_with_label, dominant_reasons, reason_map


def _build_reason_bucket_deltas(
    *,
    from_reason_map: dict[str, dict[str, object]],
    to_reason_map: dict[str, dict[str, object]],
    from_n_cases_with_label: int,
    to_n_cases_with_label: int,
) -> dict[str, list[dict[str, object]]]:
    buckets: dict[str, list[dict[str, object]]] = {
        "added": [],
        "removed": [],
        "increased": [],
        "decreased": [],
        "stable": [],
    }
    for reason in sorted(set(from_reason_map) | set(to_reason_map)):
        from_row = from_reason_map.get(reason, {})
        to_row = to_reason_map.get(reason, {})
        raw_from_count = from_row.get("count")
        from_count = (
            raw_from_count if isinstance(raw_from_count, int) and raw_from_count >= 0 else 0
        )
        raw_from_prop = from_row.get("proportion_of_label_cases")
        from_prop = (
            float(raw_from_prop)
            if isinstance(raw_from_prop, int | float) and not isinstance(raw_from_prop, bool)
            else 0.0
        )
        raw_to_count = to_row.get("count")
        to_count = raw_to_count if isinstance(raw_to_count, int) and raw_to_count >= 0 else 0
        raw_to_prop = to_row.get("proportion_of_label_cases")
        to_prop = (
            float(raw_to_prop)
            if isinstance(raw_to_prop, int | float) and not isinstance(raw_to_prop, bool)
            else 0.0
        )
        from_case_names = _to_str_list(from_row.get("supporting_case_names"), max_items=6)
        to_case_names = _to_str_list(to_row.get("supporting_case_names"), max_items=6)
        from_artifact_hints = _to_str_list(from_row.get("artifact_pointer_hints"), max_items=4)
        to_artifact_hints = _to_str_list(to_row.get("artifact_pointer_hints"), max_items=4)
        row = {
            "reason": reason,
            "from_count": from_count,
            "from_n_cases_with_label": from_n_cases_with_label,
            "from_proportion_of_label_cases": from_prop,
            "from_supporting_case_names": from_case_names,
            "to_count": to_count,
            "to_n_cases_with_label": to_n_cases_with_label,
            "to_proportion_of_label_cases": to_prop,
            "to_supporting_case_names": to_case_names,
            "delta_count": to_count - from_count,
            "delta_proportion_of_label_cases": to_prop - from_prop,
            "supporting_case_names": _merge_unique_text_lists(
                from_case_names,
                to_case_names,
                max_items=6,
            ),
            "artifact_pointer_hints": _merge_unique_text_lists(
                from_artifact_hints,
                to_artifact_hints,
                max_items=4,
            ),
        }
        if from_count <= 0 and to_count > 0:
            buckets["added"].append(row)
            continue
        if from_count > 0 and to_count <= 0:
            buckets["removed"].append(row)
            continue
        delta_prop = to_prop - from_prop
        if abs(delta_prop) <= 1e-12:
            buckets["stable"].append(row)
        elif delta_prop > 0:
            buckets["increased"].append(row)
        else:
            buckets["decreased"].append(row)

    buckets["added"].sort(
        key=lambda row: (
            -_to_float_value(row.get("to_proportion_of_label_cases")),
            -_to_int_value(row.get("to_count")),
            str(row.get("reason") or "").lower(),
        )
    )
    buckets["removed"].sort(
        key=lambda row: (
            -_to_float_value(row.get("from_proportion_of_label_cases")),
            -_to_int_value(row.get("from_count")),
            str(row.get("reason") or "").lower(),
        )
    )
    buckets["increased"].sort(
        key=lambda row: (
            -_to_float_value(row.get("delta_proportion_of_label_cases")),
            -_to_int_value(row.get("delta_count")),
            str(row.get("reason") or "").lower(),
        )
    )
    buckets["decreased"].sort(
        key=lambda row: (
            _to_float_value(row.get("delta_proportion_of_label_cases")),
            _to_int_value(row.get("delta_count")),
            str(row.get("reason") or "").lower(),
        )
    )
    buckets["stable"].sort(
        key=lambda row: (
            -_to_float_value(row.get("to_proportion_of_label_cases")),
            -_to_int_value(row.get("to_count")),
            str(row.get("reason") or "").lower(),
        )
    )
    return buckets


def _profile_pairs(
    profiles: list[str],
    *,
    pair_mode: PairMode,
) -> list[tuple[str, str]]:
    if pair_mode == "adjacent":
        return _adjacent_profile_pairs(profiles)
    return [
        (profiles[i], profiles[j])
        for i in range(max(0, len(profiles) - 1))
        for j in range(i + 1, len(profiles))
    ]


def _empty_transition_pair_case_matrix() -> dict[str, dict[str, list[str]]]:
    return {
        from_label: {to_label: [] for to_label in LEVEL12_TRANSITION_TAXONOMY}
        for from_label in LEVEL12_TRANSITION_TAXONOMY
    }


def _render_nonzero_transition_pair_counts(
    counts_obj: object,
    proportions_obj: object,
    case_names_obj: object = None,
) -> str:
    counts = counts_obj if isinstance(counts_obj, dict) else {}
    proportions = proportions_obj if isinstance(proportions_obj, dict) else {}
    case_names_map = case_names_obj if isinstance(case_names_obj, dict) else {}
    rows: list[tuple[int, str]] = []
    for from_label in LEVEL12_TRANSITION_TAXONOMY:
        from_counts_obj = counts.get(from_label)
        from_counts = from_counts_obj if isinstance(from_counts_obj, dict) else {}
        from_props_obj = proportions.get(from_label)
        from_props = from_props_obj if isinstance(from_props_obj, dict) else {}
        from_case_names_obj = case_names_map.get(from_label)
        from_case_names = from_case_names_obj if isinstance(from_case_names_obj, dict) else {}
        for to_label in LEVEL12_TRANSITION_TAXONOMY:
            raw_count = from_counts.get(to_label, 0)
            count = raw_count if isinstance(raw_count, int) else 0
            if count <= 0:
                continue
            raw_prop = from_props.get(to_label, 0.0)
            prop = raw_prop if isinstance(raw_prop, float | int) else 0.0
            flow_cases_obj = from_case_names.get(to_label)
            flow_cases = (
                [str(item) for item in flow_cases_obj if str(item).strip()]
                if isinstance(flow_cases_obj, list)
                else []
            )
            case_suffix = f"; cases={', '.join(flow_cases)}" if flow_cases else ""
            rows.append(
                (
                    count,
                    f"{from_label} -> {to_label}: {count} ({float(prop):.1%}){case_suffix}",
                )
            )
    if not rows:
        return "none"
    rows.sort(key=lambda row: (-row[0], row[1]))
    return "; ".join(text for _, text in rows)


def _render_reason_stat_tokens(
    reason_rows_obj: object,
    *,
    max_items: int = 2,
) -> str:
    reason_rows = reason_rows_obj if isinstance(reason_rows_obj, list) else []
    if max_items <= 0:
        return "none"
    tokens: list[str] = []
    for row in reason_rows[:max_items]:
        if not isinstance(row, dict):
            continue
        reason = str(row.get("reason") or "").strip()
        if not reason:
            continue
        raw_count = row.get("count")
        count = raw_count if isinstance(raw_count, int) and raw_count >= 0 else 0
        raw_n_cases = row.get("n_cases_with_label")
        n_cases = raw_n_cases if isinstance(raw_n_cases, int) and raw_n_cases >= 0 else 0
        supporting_case_names = _to_str_list(
            row.get("supporting_case_names"),
            max_items=2,
        )
        case_suffix = f"; cases={','.join(supporting_case_names)}" if supporting_case_names else ""
        tokens.append(
            f"`{reason}` {_format_reason_ratio(count=count, n_cases=n_cases)}{case_suffix}"
        )
    if not tokens:
        return "none"
    return "; ".join(tokens)


def _render_reason_delta_bucket_tokens(
    deltas_obj: object,
    *,
    max_items_per_bucket: int = 1,
) -> str:
    deltas = deltas_obj if isinstance(deltas_obj, dict) else {}
    parts: list[str] = []
    for bucket_name in ("added", "removed", "increased", "decreased"):
        bucket_rows_obj = deltas.get(bucket_name)
        bucket_rows = bucket_rows_obj if isinstance(bucket_rows_obj, list) else []
        if not bucket_rows:
            continue
        rendered_rows = [
            _render_reason_delta_row(row)
            for row in bucket_rows[:max_items_per_bucket]
            if isinstance(row, dict)
        ]
        rendered_rows = [row for row in rendered_rows if row]
        if rendered_rows:
            parts.append(f"{bucket_name}: " + ", ".join(rendered_rows))
    if not parts:
        stable_rows_obj = deltas.get("stable")
        stable_rows = stable_rows_obj if isinstance(stable_rows_obj, list) else []
        rendered_stable = [
            _render_reason_delta_row(row)
            for row in stable_rows[:max_items_per_bucket]
            if isinstance(row, dict)
        ]
        rendered_stable = [row for row in rendered_stable if row]
        if rendered_stable:
            parts.append("stable: " + ", ".join(rendered_stable))
    return "; ".join(parts) if parts else "none"


def _render_reason_delta_row(row: dict[str, object]) -> str:
    reason = str(row.get("reason") or "").strip()
    if not reason:
        return ""
    raw_from_count = row.get("from_count")
    from_count = raw_from_count if isinstance(raw_from_count, int) and raw_from_count >= 0 else 0
    raw_to_count = row.get("to_count")
    to_count = raw_to_count if isinstance(raw_to_count, int) and raw_to_count >= 0 else 0
    raw_from_n_cases = row.get("from_n_cases_with_label")
    from_n_cases = (
        raw_from_n_cases if isinstance(raw_from_n_cases, int) and raw_from_n_cases >= 0 else 0
    )
    raw_to_n_cases = row.get("to_n_cases_with_label")
    to_n_cases = raw_to_n_cases if isinstance(raw_to_n_cases, int) and raw_to_n_cases >= 0 else 0
    raw_delta_prop = row.get("delta_proportion_of_label_cases")
    delta_prop = (
        float(raw_delta_prop)
        if isinstance(raw_delta_prop, int | float) and not isinstance(raw_delta_prop, bool)
        else 0.0
    )
    supporting_case_names = _to_str_list(row.get("supporting_case_names"), max_items=2)
    case_suffix = f"; cases={','.join(supporting_case_names)}" if supporting_case_names else ""
    return (
        f"`{reason}` "
        f"{_format_reason_ratio(count=from_count, n_cases=from_n_cases)} -> "
        f"{_format_reason_ratio(count=to_count, n_cases=to_n_cases)} "
        f"({delta_prop * 100.0:+.1f}pp){case_suffix}"
    )


def _transition_reason_rollup_tokens(
    distribution: dict[str, object],
    *,
    per_label_limit: int = 1,
) -> list[str]:
    rollups_obj = distribution.get("reason_rollup_by_transition_label")
    rollups = rollups_obj if isinstance(rollups_obj, dict) else {}
    max_per_label = max(0, per_label_limit)
    tokens: list[str] = []
    for label in LEVEL12_TRANSITION_TAXONOMY:
        rollup_obj = rollups.get(label)
        rollup = rollup_obj if isinstance(rollup_obj, dict) else {}
        dominant_reasons_obj = rollup.get("dominant_reasons")
        dominant_reasons = dominant_reasons_obj if isinstance(dominant_reasons_obj, list) else []
        top_reasons_obj = rollup.get("top_reasons")
        top_reasons = top_reasons_obj if isinstance(top_reasons_obj, list) else []
        reason_rows = dominant_reasons if dominant_reasons else top_reasons
        if not reason_rows:
            continue
        top_tokens: list[str] = []
        for row in reason_rows[:max_per_label]:
            if not isinstance(row, dict):
                continue
            reason = str(row.get("reason") or "").strip()
            if not reason:
                continue
            raw_count = row.get("count")
            count = raw_count if isinstance(raw_count, int) else 0
            raw_prop = row.get("proportion_of_label_cases")
            prop = raw_prop if isinstance(raw_prop, int | float) else 0.0
            supporting_case_names = _to_str_list(
                row.get("supporting_case_names"),
                max_items=2,
            )
            case_suffix = (
                f"; cases={','.join(supporting_case_names)}" if supporting_case_names else ""
            )
            top_tokens.append(f"`{reason}` ({count}, {float(prop):.1%}{case_suffix})")
        if not top_tokens:
            continue
        if dominant_reasons:
            tokens.append(f"{label}: {', '.join(top_tokens)}")
        else:
            support_note = str(rollup.get("support_note") or "tentative due to low support")
            tokens.append(f"{label}: {support_note} ({', '.join(top_tokens)})")
    return tokens


def _build_case_evidence_index(
    *,
    case_comparison: list[dict[str, object]],
    profiles: list[str],
    max_reason_hints_per_profile: int = 3,
    max_artifact_hints_per_profile: int = 2,
) -> dict[str, dict[str, object]]:
    max_reason_hints = max(0, int(max_reason_hints_per_profile))
    max_artifact_hints = max(0, int(max_artifact_hints_per_profile))
    out: dict[str, dict[str, object]] = {}
    for row in case_comparison:
        if not isinstance(row, dict):
            continue
        case_name = str(row.get("case_name") or "").strip()
        if not case_name:
            continue
        profiles_obj = row.get("profiles")
        profile_map = profiles_obj if isinstance(profiles_obj, dict) else {}
        if not profile_map:
            continue

        ordered_profiles = [
            profile_name
            for profile_name in profiles
            if isinstance(profile_map.get(profile_name), dict)
        ]
        for profile_name in sorted(profile_map):
            if profile_name in ordered_profiles:
                continue
            if isinstance(profile_map.get(profile_name), dict):
                ordered_profiles.append(profile_name)

        factor_verdict_by_profile: dict[str, str] = {}
        campaign_triage_by_profile: dict[str, str] = {}
        promotion_decision_by_profile: dict[str, str] = {}
        portfolio_robustness_by_profile: dict[str, str] = {}
        level12_transition_label_by_profile: dict[str, str] = {}
        key_reason_hints_by_profile: dict[str, list[str]] = {}
        artifact_pointer_hints_by_profile: dict[str, list[str]] = {}
        for profile_name in ordered_profiles:
            payload_obj = profile_map.get(profile_name)
            payload = payload_obj if isinstance(payload_obj, dict) else {}
            factor_verdict_by_profile[profile_name] = str(payload.get("factor_verdict") or "N/A")
            campaign_triage_by_profile[profile_name] = str(payload.get("campaign_triage") or "N/A")
            promotion_decision_by_profile[profile_name] = str(
                payload.get("promotion_decision") or "N/A"
            )
            portfolio_robustness_by_profile[profile_name] = str(
                payload.get("portfolio_validation_recommendation") or "N/A"
            )
            level12_transition_label_by_profile[profile_name] = str(
                payload.get("level12_transition_label") or "N/A"
            )
            key_reason_hints_by_profile[profile_name] = _to_str_list(
                payload.get("level12_transition_reasons"),
                max_items=max_reason_hints,
            )
            artifact_paths_obj = payload.get("artifact_paths")
            artifact_paths = artifact_paths_obj if isinstance(artifact_paths_obj, dict) else {}
            hints: list[str] = []
            for field in (
                "metrics_path",
                "summary_path",
                "experiment_card_path",
                "output_dir",
                "run_manifest_path",
                "signal_validation_json_path",
                "portfolio_recipe_json_path",
                "backtest_result_json_path",
                "factor_definition_json_path",
            ):
                _append_unique_text(
                    hints,
                    str(artifact_paths.get(field) or "").strip(),
                    max_items=max_artifact_hints,
                )
            artifact_pointer_hints_by_profile[profile_name] = hints

        changed_fields = _to_str_list(row.get("changed_fields"), max_items=8)
        transition_delta_obj = row.get("level12_transition_profile_delta")
        transition_delta = transition_delta_obj if isinstance(transition_delta_obj, dict) else {}
        transition_labels_obj = transition_delta.get("profile_transition_labels")
        transition_labels = transition_labels_obj if isinstance(transition_labels_obj, dict) else {}
        transition_label_by_profile = {
            profile_name: str(
                transition_labels.get(profile_name)
                or level12_transition_label_by_profile.get(profile_name)
                or "N/A"
            )
            for profile_name in ordered_profiles
        }
        out[case_name] = {
            "profiles_observed": ordered_profiles,
            "profile_sensitivity": str(row.get("profile_sensitivity") or "profile_sensitive"),
            "changed_fields": changed_fields,
            "profile_delta_label": _case_transition_delta_label(row),
            "factor_verdict_by_profile": factor_verdict_by_profile,
            "campaign_triage_by_profile": campaign_triage_by_profile,
            "promotion_decision_by_profile": promotion_decision_by_profile,
            "portfolio_robustness_by_profile": portfolio_robustness_by_profile,
            "level12_transition_label_by_profile": transition_label_by_profile,
            "key_reason_hints_by_profile": key_reason_hints_by_profile,
            "artifact_pointer_hints_by_profile": artifact_pointer_hints_by_profile,
        }
    return out


def _build_compact_comparison_summary(
    *,
    case_comparison: list[dict[str, object]],
    profiles: list[str],
    transition_distribution_by_profile: Mapping[str, Mapping[str, object]],
    transition_profile_delta_matrix: dict[str, object],
    transition_reason_profile_delta_matrix: dict[str, object],
    transition_stable_cases: list[str],
    transition_sensitive_cases: list[str],
    root_dir: Path | None = None,
) -> dict[str, object]:
    n_cases = len(case_comparison)
    stable_count = len(transition_stable_cases)
    sensitive_count = len(transition_sensitive_cases)
    stable_share = stable_count / n_cases if n_cases > 0 else 0.0
    case_rows_by_name = {
        str(row.get("case_name")): row
        for row in case_comparison
        if isinstance(row, dict) and str(row.get("case_name") or "").strip()
    }
    representative_transition_stable_cases = transition_stable_cases[:3]
    supporting_case_names = [
        str(row.get("case_name") or "").strip()
        for row in case_comparison
        if isinstance(row, dict) and str(row.get("case_name") or "").strip()
    ][:6]
    transition_stability_artifact_hints: list[str] = []
    for case_name in representative_transition_stable_cases:
        case_row = case_rows_by_name.get(case_name)
        if not isinstance(case_row, dict):
            continue
        _extend_unique_text(
            transition_stability_artifact_hints,
            _case_artifact_pointer_hints_from_case_comparison(
                case_row,
                profiles=profiles,
            ),
            max_items=4,
        )
    transition_stability_support = _support_annotation(
        support_count=n_cases,
        minimum_required_support=int(
            LEVEL12_TRANSITION_SUPPORT_THRESHOLDS["minimum_cases_with_transition_label"]
        ),
        sparse_note="sparse transition evidence",
        tentative_note="tentative due to low support",
        supported_note="transition evidence is well supported",
    )

    compact_summary: dict[str, object] = {
        "minimum_support_thresholds": dict(_CAMPAIGN_PROFILE_COMPARISON_SUPPORT_THRESHOLDS),
        "transition_stability": {
            "n_cases": n_cases,
            "n_transition_stable_cases": stable_count,
            "n_transition_sensitive_cases": sensitive_count,
            "stable_case_share": stable_share,
            "representative_transition_stable_cases": representative_transition_stable_cases,
            "representative_case_names": representative_transition_stable_cases,
            "supporting_case_names": supporting_case_names,
            "artifact_pointer_hints": transition_stability_artifact_hints,
            "support_count": n_cases,
            "minimum_required_support": int(
                LEVEL12_TRANSITION_SUPPORT_THRESHOLDS["minimum_cases_with_transition_label"]
            ),
            **transition_stability_support,
        },
        "most_profile_sensitive_cases": _top_profile_sensitive_cases(
            case_comparison=case_comparison,
            profiles=profiles,
            max_items=3,
        ),
        "strongest_profile_pair_shifts": _strongest_profile_pair_shifts(
            transition_profile_delta_matrix=transition_profile_delta_matrix,
            transition_reason_profile_delta_matrix=transition_reason_profile_delta_matrix,
            max_items=2,
        ),
        "weakened_fragile_reason_hotspots": _top_weakened_fragile_reasons(
            transition_distribution_by_profile=transition_distribution_by_profile,
            strictest_profile=(profiles[-1] if profiles else ""),
            max_items=3,
        ),
        "stricter_profile_impact": _stricter_profile_impact_summary(
            transition_profile_delta_matrix=transition_profile_delta_matrix
        ),
    }
    if root_dir is not None:
        compact_summary["artifact_pointer_hints"] = [
            str(root_dir / "campaign_profile_comparison.md"),
            str(root_dir / "campaign_profile_comparison.json"),
            str(root_dir / "campaign_profile_case_matrix.csv"),
        ]
    compact_summary["summary_lines"] = _compact_comparison_summary_lines(compact_summary)
    return compact_summary


def _top_profile_sensitive_cases(
    *,
    case_comparison: list[dict[str, object]],
    profiles: list[str],
    max_items: int,
) -> list[dict[str, object]]:
    max_cases = max(0, max_items)
    if max_cases <= 0:
        return []

    def _sensitivity_rank(label: str) -> int:
        if label == "highly_profile_sensitive":
            return 2
        if label == "profile_sensitive":
            return 1
        return 0

    rows: list[dict[str, object]] = []
    for row in case_comparison:
        case_name = str(row.get("case_name") or "").strip()
        if not case_name:
            continue
        changed_fields_obj = row.get("changed_fields")
        changed_fields = (
            [str(item) for item in changed_fields_obj if str(item).strip()]
            if isinstance(changed_fields_obj, list)
            else []
        )
        n_changed_fields = len(changed_fields)
        if n_changed_fields <= 0:
            continue
        sensitivity = str(row.get("profile_sensitivity") or "profile_sensitive")
        transition_delta_label = _case_transition_delta_label(row)
        transition_delta_obj = row.get("level12_transition_profile_delta")
        transition_delta = transition_delta_obj if isinstance(transition_delta_obj, dict) else {}
        transition_path = _transition_profile_path_text(
            transition_delta.get("profile_transition_labels"),
            profiles,
        )
        rows.append(
            {
                "case_name": case_name,
                "representative_case_names": [case_name],
                "supporting_case_names": [case_name],
                "profile_sensitivity": sensitivity,
                "n_changed_fields": n_changed_fields,
                "changed_fields": changed_fields,
                "transition_delta_label": transition_delta_label,
                "transition_profile_path": transition_path,
                "artifact_pointer_hints": _case_artifact_pointer_hints_from_case_comparison(
                    row,
                    profiles=profiles,
                ),
            }
        )

    rows.sort(
        key=lambda row: (
            -_sensitivity_rank(str(row.get("profile_sensitivity") or "")),
            -_to_int_value(row.get("n_changed_fields")),
            0
            if str(row.get("transition_delta_label") or "") != _TRANSITION_DELTA_LABEL_STABLE
            else 1,
            str(row.get("case_name") or "").lower(),
        )
    )
    return rows[:max_cases]


def _strongest_profile_pair_shifts(
    *,
    transition_profile_delta_matrix: dict[str, object],
    transition_reason_profile_delta_matrix: dict[str, object],
    max_items: int,
) -> list[dict[str, object]]:
    max_pairs = max(0, max_items)
    if max_pairs <= 0:
        return []

    pair_rows_obj = transition_profile_delta_matrix.get("profile_pairs")
    pair_rows = pair_rows_obj if isinstance(pair_rows_obj, list) else []

    reason_pair_rows_obj = transition_reason_profile_delta_matrix.get("profile_pairs")
    reason_pair_rows = reason_pair_rows_obj if isinstance(reason_pair_rows_obj, list) else []
    reason_pair_by_key: dict[tuple[str, str], dict[str, object]] = {}
    for row in reason_pair_rows:
        if not isinstance(row, dict):
            continue
        from_profile = str(row.get("from_profile") or "").strip()
        to_profile = str(row.get("to_profile") or "").strip()
        if not from_profile or not to_profile:
            continue
        reason_pair_by_key[(from_profile, to_profile)] = row

    out_rows: list[dict[str, object]] = []
    for row in pair_rows:
        if not isinstance(row, dict):
            continue
        from_profile = str(row.get("from_profile") or "").strip()
        to_profile = str(row.get("to_profile") or "").strip()
        if not from_profile or not to_profile:
            continue
        observed_raw = row.get("n_cases_with_observed_transition_labels")
        observed = observed_raw if isinstance(observed_raw, int) and observed_raw >= 0 else 0
        changed_raw = row.get("changed_count")
        changed_count = changed_raw if isinstance(changed_raw, int) and changed_raw >= 0 else 0
        changed_proportion = changed_count / observed if observed > 0 else 0.0
        reason_pair = reason_pair_by_key.get((from_profile, to_profile), {})
        shifted_labels_raw = reason_pair.get("n_transition_labels_with_reason_shift")
        shifted_labels = (
            shifted_labels_raw
            if isinstance(shifted_labels_raw, int) and shifted_labels_raw >= 0
            else 0
        )
        observed_reason_labels_raw = reason_pair.get("n_transition_labels_with_observed_reasons")
        observed_reason_labels = (
            observed_reason_labels_raw
            if isinstance(observed_reason_labels_raw, int) and observed_reason_labels_raw >= 0
            else 0
        )
        tentative_shifted_labels_raw = reason_pair.get(
            "n_transition_labels_with_tentative_reason_shift"
        )
        tentative_shifted_labels = (
            tentative_shifted_labels_raw
            if isinstance(tentative_shifted_labels_raw, int) and tentative_shifted_labels_raw >= 0
            else 0
        )
        representative_case_names = _merge_unique_text_lists(
            _to_str_list(row.get("representative_case_names"), max_items=4),
            _to_str_list(reason_pair.get("representative_case_names"), max_items=4),
            max_items=4,
        )
        supporting_case_names = _merge_unique_text_lists(
            _to_str_list(row.get("supporting_case_names"), max_items=6),
            _to_str_list(reason_pair.get("supporting_case_names"), max_items=6),
            max_items=6,
        )
        artifact_pointer_hints = _merge_unique_text_lists(
            _to_str_list(row.get("artifact_pointer_hints"), max_items=4),
            _to_str_list(reason_pair.get("artifact_pointer_hints"), max_items=4),
            max_items=4,
        )
        transition_support_met_raw = row.get("minimum_support_met")
        transition_support_met = (
            bool(transition_support_met_raw)
            if isinstance(transition_support_met_raw, bool)
            else observed >= _MIN_OBSERVED_CASES_PER_PROFILE_PAIR
        )
        reason_support_met_raw = reason_pair.get("minimum_support_met")
        reason_support_met = (
            bool(reason_support_met_raw)
            if isinstance(reason_support_met_raw, bool)
            else observed_reason_labels >= _MIN_OBSERVED_REASON_LABELS_PER_PROFILE_PAIR
        )
        combined_support_met = transition_support_met and reason_support_met
        combined_is_sparse = not combined_support_met
        if shifted_labels > 0 and combined_is_sparse:
            combined_support_note = "reason shift observed, but only in a small number of cases"
            combined_support_level = "sparse"
        elif tentative_shifted_labels > 0 and shifted_labels <= 0:
            combined_support_note = "reason shift observed, but only in a small number of cases"
            combined_support_level = "tentative"
        elif combined_is_sparse:
            combined_support_note = "tentative due to low support"
            combined_support_level = "sparse"
        elif shifted_labels <= 1 and changed_count <= 1:
            combined_support_note = "tentative due to low support"
            combined_support_level = "tentative"
        else:
            combined_support_note = "transition and reason shifts are well supported"
            combined_support_level = "supported"
        out_rows.append(
            {
                "from_profile": from_profile,
                "to_profile": to_profile,
                "profile_pair": f"{from_profile} -> {to_profile}",
                "changed_count": changed_count,
                "n_cases_with_observed_transition_labels": observed,
                "changed_proportion": changed_proportion,
                "reason_shifted_labels": shifted_labels,
                "n_transition_labels_with_observed_reasons": observed_reason_labels,
                "tentative_reason_shifted_labels": tentative_shifted_labels,
                "transition_support_level": str(row.get("support_level") or "sparse"),
                "transition_support_note": str(
                    row.get("support_note")
                    or (
                        "tentative due to low support"
                        if transition_support_met
                        else "sparse transition evidence"
                    )
                ),
                "reason_support_level": str(reason_pair.get("support_level") or "sparse"),
                "reason_support_note": str(
                    reason_pair.get("support_note")
                    or (
                        "tentative due to low support"
                        if reason_support_met
                        else "sparse transition evidence"
                    )
                ),
                "support_level": combined_support_level,
                "is_sparse": combined_is_sparse,
                "minimum_support_met": combined_support_met,
                "support_note": combined_support_note,
                "confidence_note": combined_support_note,
                "representative_case_names": representative_case_names,
                "supporting_case_names": supporting_case_names,
                "artifact_pointer_hints": artifact_pointer_hints,
                "top_shift_flows": _top_shift_flows(
                    counts_obj=row.get("counts_by_from_to_label"),
                    case_names_obj=row.get("representative_case_names_by_from_to_label"),
                    n_observed=observed,
                    max_items=2,
                ),
            }
        )

    out_rows.sort(
        key=lambda row: (
            -_to_float_value(row.get("changed_proportion")),
            -_to_int_value(row.get("changed_count")),
            -_to_int_value(row.get("reason_shifted_labels")),
            str(row.get("from_profile") or "").lower(),
            str(row.get("to_profile") or "").lower(),
        )
    )
    return out_rows[:max_pairs]


def _top_shift_flows(
    *,
    counts_obj: object,
    case_names_obj: object,
    n_observed: int,
    max_items: int,
) -> list[dict[str, object]]:
    max_flows = max(0, max_items)
    if max_flows <= 0:
        return []
    counts = counts_obj if isinstance(counts_obj, dict) else {}
    case_names_map = case_names_obj if isinstance(case_names_obj, dict) else {}
    flows: list[dict[str, object]] = []
    for from_label in LEVEL12_TRANSITION_TAXONOMY:
        from_counts_obj = counts.get(from_label)
        from_counts = from_counts_obj if isinstance(from_counts_obj, dict) else {}
        from_case_names_obj = case_names_map.get(from_label)
        from_case_names = from_case_names_obj if isinstance(from_case_names_obj, dict) else {}
        for to_label in LEVEL12_TRANSITION_TAXONOMY:
            if from_label == to_label:
                continue
            raw_count = from_counts.get(to_label, 0)
            count = raw_count if isinstance(raw_count, int) and raw_count >= 0 else 0
            if count <= 0:
                continue
            representative_case_names = _to_str_list(
                from_case_names.get(to_label),
                max_items=3,
            )
            flows.append(
                {
                    "from_label": from_label,
                    "to_label": to_label,
                    "count": count,
                    "proportion_of_observed": (count / n_observed if n_observed > 0 else 0.0),
                    "representative_case_names": representative_case_names,
                }
            )
    flows.sort(
        key=lambda row: (
            -_to_int_value(row.get("count")),
            str(row.get("from_label") or "").lower(),
            str(row.get("to_label") or "").lower(),
        )
    )
    return flows[:max_flows]


def _top_weakened_fragile_reasons(
    *,
    transition_distribution_by_profile: Mapping[str, Mapping[str, object]],
    strictest_profile: str,
    max_items: int,
) -> dict[str, object]:
    max_reasons = max(0, max_items)
    profile_name = str(strictest_profile).strip()
    dist_obj = transition_distribution_by_profile.get(profile_name)
    distribution = dist_obj if isinstance(dist_obj, dict) else {}
    counts_obj = distribution.get("counts_by_transition_label")
    counts = counts_obj if isinstance(counts_obj, dict) else {}
    rollups_obj = distribution.get("reason_rollup_by_transition_label")
    rollups = rollups_obj if isinstance(rollups_obj, dict) else {}

    top_rows: list[dict[str, object]] = []
    for transition_label in (
        "Weakened at portfolio level",
        "Fragile after promotion",
    ):
        rollup_obj = rollups.get(transition_label)
        rollup = rollup_obj if isinstance(rollup_obj, dict) else {}
        dominant_reasons_obj = rollup.get("dominant_reasons")
        top_reasons_obj = rollup.get("top_reasons")
        dominant_reasons = dominant_reasons_obj if isinstance(dominant_reasons_obj, list) else []
        top_reasons = top_reasons_obj if isinstance(top_reasons_obj, list) else []
        reason_rows = dominant_reasons if dominant_reasons else top_reasons
        n_cases_obj = rollup.get("n_cases_with_label")
        n_cases_with_label = n_cases_obj if isinstance(n_cases_obj, int) and n_cases_obj >= 0 else 0
        for row in reason_rows:
            if not isinstance(row, dict):
                continue
            reason = str(row.get("reason") or "").strip()
            if not reason:
                continue
            raw_count = row.get("count")
            count = raw_count if isinstance(raw_count, int) and raw_count >= 0 else 0
            raw_prop = row.get("proportion_of_label_cases")
            proportion = (
                float(raw_prop)
                if isinstance(raw_prop, int | float) and not isinstance(raw_prop, bool)
                else 0.0
            )
            top_rows.append(
                {
                    "transition_label": transition_label,
                    "reason": reason,
                    "count": count,
                    "n_cases_with_label": n_cases_with_label,
                    "proportion_of_label_cases": proportion,
                    "supporting_case_names": _to_str_list(
                        row.get("supporting_case_names"),
                        max_items=4,
                    ),
                    "artifact_pointer_hints": _to_str_list(
                        row.get("artifact_pointer_hints"),
                        max_items=3,
                    ),
                    "minimum_support_met": bool(row.get("minimum_support_met")),
                    "support_level": str(row.get("support_level") or "sparse"),
                    "support_note": str(row.get("support_note") or "tentative due to low support"),
                }
            )
    top_rows.sort(
        key=lambda row: (
            -_to_int_value(row.get("count")),
            -_to_float_value(row.get("proportion_of_label_cases")),
            str(row.get("transition_label") or "").lower(),
            str(row.get("reason") or "").lower(),
        )
    )
    selected_rows = top_rows[:max_reasons]
    supporting_case_names: list[str] = []
    artifact_pointer_hints: list[str] = []
    for row in selected_rows:
        if not isinstance(row, dict):
            continue
        _extend_unique_text(
            supporting_case_names,
            _to_str_list(row.get("supporting_case_names"), max_items=4),
            max_items=6,
        )
        _extend_unique_text(
            artifact_pointer_hints,
            _to_str_list(row.get("artifact_pointer_hints"), max_items=3),
            max_items=4,
        )
    return {
        "profile_name": profile_name,
        "n_weakened_cases": int(counts.get("Weakened at portfolio level", 0) or 0),
        "n_fragile_cases": int(counts.get("Fragile after promotion", 0) or 0),
        "minimum_support_thresholds": {
            "minimum_reason_bucket_count_for_hotspot": _MIN_REASON_BUCKET_COUNT_FOR_SHIFT,
        },
        "support_count": int(counts.get("Weakened at portfolio level", 0) or 0)
        + int(counts.get("Fragile after promotion", 0) or 0),
        "minimum_required_support": _MIN_CASES_PER_TRANSITION_LABEL_FOR_REASON_SHIFT,
        **_support_annotation(
            support_count=int(counts.get("Weakened at portfolio level", 0) or 0)
            + int(counts.get("Fragile after promotion", 0) or 0),
            minimum_required_support=_MIN_CASES_PER_TRANSITION_LABEL_FOR_REASON_SHIFT,
            sparse_note="sparse transition evidence",
            tentative_note="tentative due to low support",
            supported_note="reason evidence is well supported",
        ),
        "top_reasons": selected_rows,
        "supporting_case_names": supporting_case_names,
        "artifact_pointer_hints": artifact_pointer_hints,
    }


def _stricter_profile_impact_summary(
    *,
    transition_profile_delta_matrix: dict[str, object],
) -> dict[str, object]:
    pair_rows_obj = transition_profile_delta_matrix.get("profile_pairs")
    pair_rows = pair_rows_obj if isinstance(pair_rows_obj, list) else []

    out_rows: list[dict[str, object]] = []
    total_promotion_reduction = 0
    total_robustness_reduction = 0
    total_observed = 0
    for row in pair_rows:
        if not isinstance(row, dict):
            continue
        from_profile = str(row.get("from_profile") or "").strip()
        to_profile = str(row.get("to_profile") or "").strip()
        if not from_profile or not to_profile:
            continue
        observed_obj = row.get("n_cases_with_observed_transition_labels")
        observed = observed_obj if isinstance(observed_obj, int) and observed_obj >= 0 else 0
        counts_obj = row.get("counts_by_from_to_label")
        promotion_reduction, robustness_reduction = _pair_reduction_counts(counts_obj=counts_obj)
        total_promotion_reduction += promotion_reduction
        total_robustness_reduction += robustness_reduction
        total_observed += observed
        pair_support = _support_annotation(
            support_count=observed,
            minimum_required_support=_MIN_OBSERVED_CASES_PER_PROFILE_PAIR,
            sparse_note="sparse transition evidence",
            tentative_note="tentative due to low support",
            supported_note="transition evidence is well supported",
        )
        out_rows.append(
            {
                "from_profile": from_profile,
                "to_profile": to_profile,
                "n_cases_with_observed_transition_labels": observed,
                "promotion_reduction_count": promotion_reduction,
                "robustness_reduction_count": robustness_reduction,
                "dominant_reduction_mode": _dominant_reduction_mode(
                    promotion_reduction,
                    robustness_reduction,
                ),
                "minimum_support_thresholds": {
                    "minimum_observed_cases": _MIN_OBSERVED_CASES_PER_PROFILE_PAIR,
                },
                "support_count": observed,
                "minimum_required_support": _MIN_OBSERVED_CASES_PER_PROFILE_PAIR,
                **pair_support,
            }
        )
    aggregate_support = _support_annotation(
        support_count=total_observed,
        minimum_required_support=max(
            1,
            _MIN_OBSERVED_CASES_PER_PROFILE_PAIR * max(1, len(out_rows)),
        ),
        sparse_note="sparse transition evidence",
        tentative_note="tentative due to low support",
        supported_note="transition evidence is well supported",
    )
    return {
        "profile_pairs": out_rows,
        "aggregate": {
            "n_profile_pairs": len(out_rows),
            "n_cases_with_observed_transition_labels": total_observed,
            "promotion_reduction_count": total_promotion_reduction,
            "robustness_reduction_count": total_robustness_reduction,
            "dominant_reduction_mode": _dominant_reduction_mode(
                total_promotion_reduction,
                total_robustness_reduction,
            ),
            "minimum_support_thresholds": {
                "minimum_observed_cases": _MIN_OBSERVED_CASES_PER_PROFILE_PAIR,
            },
            "support_count": total_observed,
            "minimum_required_support": max(
                1,
                _MIN_OBSERVED_CASES_PER_PROFILE_PAIR * max(1, len(out_rows)),
            ),
            **aggregate_support,
        },
    }


def _compact_comparison_summary_lines(compact_summary: dict[str, object]) -> list[str]:
    if not compact_summary:
        return []
    lines: list[str] = []

    transition_stability_obj = compact_summary.get("transition_stability")
    transition_stability = (
        transition_stability_obj if isinstance(transition_stability_obj, dict) else {}
    )
    n_cases = int(transition_stability.get("n_cases", 0) or 0)
    stable_count = int(transition_stability.get("n_transition_stable_cases", 0) or 0)
    sensitive_count = int(transition_stability.get("n_transition_sensitive_cases", 0) or 0)
    stable_share_raw = transition_stability.get("stable_case_share", 0.0)
    stable_share = (
        float(stable_share_raw)
        if isinstance(stable_share_raw, int | float) and not isinstance(stable_share_raw, bool)
        else 0.0
    )
    stable_case_preview_obj = transition_stability.get("representative_transition_stable_cases")
    stable_case_preview = (
        [str(item) for item in stable_case_preview_obj if str(item).strip()]
        if isinstance(stable_case_preview_obj, list)
        else []
    )
    artifact_pointer_hints = _to_str_list(
        transition_stability.get("artifact_pointer_hints"),
        max_items=2,
    )
    preview_suffix = (
        f"; representative={', '.join(stable_case_preview)}" if stable_case_preview else ""
    )
    pointer_suffix = (
        f"; pointers={'; '.join(artifact_pointer_hints)}" if artifact_pointer_hints else ""
    )
    lines.append(
        f"Transition stability: {stable_count}/{n_cases} stable ({stable_share:.1%}), "
        f"sensitive={sensitive_count}{preview_suffix}"
        f"{pointer_suffix}"
        f"{_support_note_suffix(transition_stability)}."
    )

    top_sensitive_cases_obj = compact_summary.get("most_profile_sensitive_cases")
    top_sensitive_cases = (
        top_sensitive_cases_obj if isinstance(top_sensitive_cases_obj, list) else []
    )
    sensitive_tokens: list[str] = []
    for row in top_sensitive_cases:
        if not isinstance(row, dict):
            continue
        case_name = str(row.get("case_name") or "").strip()
        if not case_name:
            continue
        artifact_hints = _to_str_list(row.get("artifact_pointer_hints"), max_items=1)
        artifact_suffix = f", pointer={artifact_hints[0]}" if artifact_hints else ""
        sensitive_tokens.append(
            f"{case_name} (changed_fields={int(row.get('n_changed_fields') or 0)}, "
            f"delta={row.get('transition_delta_label')}{artifact_suffix})"
        )
    if sensitive_tokens:
        lines.append("Most profile-sensitive cases: " + "; ".join(sensitive_tokens) + ".")

    strongest_pair_shifts_obj = compact_summary.get("strongest_profile_pair_shifts")
    strongest_pair_shifts = (
        strongest_pair_shifts_obj if isinstance(strongest_pair_shifts_obj, list) else []
    )
    if strongest_pair_shifts and isinstance(strongest_pair_shifts[0], dict):
        first_pair = strongest_pair_shifts[0]
        from_profile = str(first_pair.get("from_profile") or "N/A")
        to_profile = str(first_pair.get("to_profile") or "N/A")
        changed_count = int(first_pair.get("changed_count", 0) or 0)
        observed = int(first_pair.get("n_cases_with_observed_transition_labels", 0) or 0)
        changed_prop_raw = first_pair.get("changed_proportion", 0.0)
        changed_prop = (
            float(changed_prop_raw)
            if isinstance(changed_prop_raw, int | float) and not isinstance(changed_prop_raw, bool)
            else 0.0
        )
        shifted_labels = int(first_pair.get("reason_shifted_labels", 0) or 0)
        observed_labels = int(first_pair.get("n_transition_labels_with_observed_reasons", 0) or 0)
        top_flows_obj = first_pair.get("top_shift_flows")
        top_flows = top_flows_obj if isinstance(top_flows_obj, list) else []
        flow_tokens: list[str] = []
        for flow in top_flows:
            if not isinstance(flow, dict):
                continue
            from_label = str(flow.get("from_label") or "").strip()
            to_label = str(flow.get("to_label") or "").strip()
            if not from_label or not to_label:
                continue
            flow_case_names = _to_str_list(flow.get("representative_case_names"), max_items=2)
            case_suffix = f", cases={','.join(flow_case_names)}" if flow_case_names else ""
            flow_tokens.append(
                f"{from_label} -> {to_label} ({int(flow.get('count', 0) or 0)}{case_suffix})"
            )
        flows_text = "; ".join(flow_tokens) if flow_tokens else "none"
        representative_case_names = _to_str_list(
            first_pair.get("representative_case_names"),
            max_items=3,
        )
        representative_suffix = (
            f", representative_cases={','.join(representative_case_names)}"
            if representative_case_names
            else ""
        )
        lines.append(
            f"Strongest profile-pair shift: {from_profile} -> {to_profile} "
            f"changed={changed_count}/{observed} ({changed_prop:.1%}), "
            f"reason_shifted_labels={shifted_labels}/{observed_labels}, "
            f"top_flows={flows_text}{representative_suffix}"
            f"{_support_note_suffix(first_pair)}."
        )

    reason_hotspots_obj = compact_summary.get("weakened_fragile_reason_hotspots")
    reason_hotspots = reason_hotspots_obj if isinstance(reason_hotspots_obj, dict) else {}
    profile_name = str(reason_hotspots.get("profile_name") or "").strip()
    top_reasons_obj = reason_hotspots.get("top_reasons")
    top_reasons = top_reasons_obj if isinstance(top_reasons_obj, list) else []
    reason_tokens: list[str] = []
    for row in top_reasons:
        if not isinstance(row, dict):
            continue
        reason = str(row.get("reason") or "").strip()
        transition_label = str(row.get("transition_label") or "").strip()
        if not reason or not transition_label:
            continue
        count = int(row.get("count", 0) or 0)
        n_cases_with_label = int(row.get("n_cases_with_label", 0) or 0)
        reason_case_names = _to_str_list(row.get("supporting_case_names"), max_items=2)
        reason_case_suffix = f"; cases={','.join(reason_case_names)}" if reason_case_names else ""
        reason_tokens.append(
            f"{reason} [{transition_label}] "
            f"{_format_reason_ratio(count=count, n_cases=n_cases_with_label)}"
            f"{reason_case_suffix}"
            f"{_support_note_suffix(row)}"
        )
    if reason_tokens:
        lines.append(
            f"Most common weakened/fragile reasons under {profile_name}: "
            + "; ".join(reason_tokens)
            + f"{_support_note_suffix(reason_hotspots)}."
        )

    stricter_impact_obj = compact_summary.get("stricter_profile_impact")
    stricter_impact = stricter_impact_obj if isinstance(stricter_impact_obj, dict) else {}
    aggregate_obj = stricter_impact.get("aggregate")
    aggregate = aggregate_obj if isinstance(aggregate_obj, dict) else {}
    dominant_mode = str(aggregate.get("dominant_reduction_mode") or "none")
    promotion_reduction = int(aggregate.get("promotion_reduction_count", 0) or 0)
    robustness_reduction = int(aggregate.get("robustness_reduction_count", 0) or 0)
    n_pairs = int(aggregate.get("n_profile_pairs", 0) or 0)
    lines.append(
        f"Stricter profile impact: {dominant_mode} "
        f"(promotion_reduction={promotion_reduction}, "
        f"robustness_reduction={robustness_reduction}, adjacent_pairs={n_pairs})"
        f"{_support_note_suffix(aggregate)}."
    )

    return lines


def _transition_profile_delta_pair_summaries(
    summary: dict[str, object],
) -> tuple[str, ...]:
    matrix_obj = summary.get("level12_transition_profile_delta_matrix")
    matrix = matrix_obj if isinstance(matrix_obj, dict) else {}
    pair_rows_obj = matrix.get("profile_pairs")
    pair_rows = pair_rows_obj if isinstance(pair_rows_obj, list) else []
    rows: list[str] = []
    for pair in pair_rows:
        if not isinstance(pair, dict):
            continue
        from_profile = str(pair.get("from_profile") or "").strip()
        to_profile = str(pair.get("to_profile") or "").strip()
        if not from_profile or not to_profile:
            continue
        observed = pair.get("n_cases_with_observed_transition_labels")
        missing = pair.get("n_cases_missing_transition_labels")
        stable_count = pair.get("stable_count")
        changed_count = pair.get("changed_count")
        representative_case_names = _to_str_list(
            pair.get("representative_case_names"),
            max_items=3,
        )
        representative_suffix = (
            f", representative_cases={','.join(representative_case_names)}"
            if representative_case_names
            else ""
        )
        rows.append(
            f"{from_profile} -> {to_profile}: stable={stable_count}, "
            f"changed={changed_count}, observed={observed}, missing={missing}"
            f"{representative_suffix}"
            f"{_support_note_suffix(pair)}"
        )
    return tuple(rows)


def _transition_reason_profile_delta_pair_summaries(
    summary: dict[str, object],
) -> tuple[str, ...]:
    matrix_obj = summary.get("level12_transition_reason_profile_delta_matrix")
    matrix = matrix_obj if isinstance(matrix_obj, dict) else {}
    pair_rows_obj = matrix.get("profile_pairs")
    pair_rows = pair_rows_obj if isinstance(pair_rows_obj, list) else []
    rows: list[str] = []
    for pair in pair_rows:
        if not isinstance(pair, dict):
            continue
        from_profile = str(pair.get("from_profile") or "").strip()
        to_profile = str(pair.get("to_profile") or "").strip()
        if not from_profile or not to_profile:
            continue
        observed_labels = pair.get("n_transition_labels_with_observed_reasons")
        shifted_labels = pair.get("n_transition_labels_with_reason_shift")
        tentative_shifted_labels = pair.get("n_transition_labels_with_tentative_reason_shift")
        stable_labels = pair.get("n_transition_labels_reason_stable")
        delta_counts_obj = pair.get("reason_bucket_delta_counts")
        delta_counts = delta_counts_obj if isinstance(delta_counts_obj, dict) else {}
        representative_case_names = _to_str_list(
            pair.get("representative_case_names"),
            max_items=3,
        )
        representative_suffix = (
            f", representative_cases={','.join(representative_case_names)}"
            if representative_case_names
            else ""
        )
        rows.append(
            f"{from_profile} -> {to_profile}: shifted_labels={shifted_labels}/"
            f"{observed_labels}, tentative_shifted_labels={tentative_shifted_labels}, "
            f"stable_labels={stable_labels}, "
            f"added={delta_counts.get('added', 0)}, "
            f"removed={delta_counts.get('removed', 0)}, "
            f"increased={delta_counts.get('increased', 0)}, "
            f"decreased={delta_counts.get('decreased', 0)}"
            f"{representative_suffix}"
            f"{_support_note_suffix(pair)}"
        )
    return tuple(rows)


def _profile_case_row(row: CampaignCaseProfileSummary) -> dict[str, object]:
    return {
        "case_name": row.case_name,
        "profile_name": row.profile_name,
        "status": row.status,
        "output_dir": str(row.output_dir) if row.output_dir is not None else None,
        "run_manifest_path": (
            str(row.run_manifest_path) if row.run_manifest_path is not None else None
        ),
        "metrics_path": str(row.metrics_path) if row.metrics_path is not None else None,
        "factor_definition_json_path": (
            str(row.factor_definition_json_path)
            if row.factor_definition_json_path is not None
            else None
        ),
        "signal_validation_json_path": (
            str(row.signal_validation_json_path)
            if row.signal_validation_json_path is not None
            else None
        ),
        "portfolio_recipe_json_path": (
            str(row.portfolio_recipe_json_path)
            if row.portfolio_recipe_json_path is not None
            else None
        ),
        "backtest_result_json_path": (
            str(row.backtest_result_json_path)
            if row.backtest_result_json_path is not None
            else None
        ),
        "summary_path": str(row.summary_path) if row.summary_path is not None else None,
        "experiment_card_path": (
            str(row.experiment_card_path) if row.experiment_card_path is not None else None
        ),
        "factor_verdict": row.factor_verdict,
        "factor_verdict_reasons": list(row.factor_verdict_reasons),
        "campaign_triage": row.campaign_triage,
        "campaign_triage_reasons": list(row.campaign_triage_reasons),
        "promotion_decision": row.promotion_decision,
        "promotion_reasons": list(row.promotion_reasons),
        "promotion_blockers": list(row.promotion_blockers),
        "level12_transition_label": row.level12_transition_label,
        "level12_transition_reasons": list(row.level12_transition_reasons),
        "portfolio_validation_status": row.portfolio_validation_status,
        "portfolio_validation_recommendation": row.portfolio_validation_recommendation,
        "portfolio_validation_major_risks": list(row.portfolio_validation_major_risks),
    }


def _case_profile_payload(summary: CampaignCaseProfileSummary | None) -> dict[str, object]:
    if summary is None:
        return {
            "status": "missing",
            "factor_verdict": "N/A",
            "campaign_triage": "N/A",
            "promotion_decision": "N/A",
            "level12_transition_label": "N/A",
            "level12_transition_reasons": [],
            "portfolio_validation_status": "N/A",
            "portfolio_validation_recommendation": "N/A",
            "major_reasons": {
                "factor_verdict_reasons": [],
                "campaign_triage_reasons": [],
                "promotion_reasons": [],
                "promotion_blockers": [],
                "level12_transition_reasons": [],
                "portfolio_validation_major_risks": [],
            },
            "artifact_paths": {},
        }

    return {
        "status": summary.status,
        "factor_verdict": summary.factor_verdict,
        "campaign_triage": summary.campaign_triage,
        "promotion_decision": summary.promotion_decision,
        "level12_transition_label": summary.level12_transition_label,
        "level12_transition_reasons": list(summary.level12_transition_reasons),
        "portfolio_validation_status": summary.portfolio_validation_status,
        "portfolio_validation_recommendation": summary.portfolio_validation_recommendation,
        "major_reasons": {
            "factor_verdict_reasons": list(summary.factor_verdict_reasons),
            "campaign_triage_reasons": list(summary.campaign_triage_reasons),
            "promotion_reasons": list(summary.promotion_reasons),
            "promotion_blockers": list(summary.promotion_blockers),
            "level12_transition_reasons": list(summary.level12_transition_reasons),
            "portfolio_validation_major_risks": list(summary.portfolio_validation_major_risks),
        },
        "artifact_paths": {
            "output_dir": str(summary.output_dir) if summary.output_dir is not None else None,
            "run_manifest_path": (
                str(summary.run_manifest_path) if summary.run_manifest_path is not None else None
            ),
            "metrics_path": str(summary.metrics_path) if summary.metrics_path is not None else None,
            "factor_definition_json_path": (
                str(summary.factor_definition_json_path)
                if summary.factor_definition_json_path is not None
                else None
            ),
            "signal_validation_json_path": (
                str(summary.signal_validation_json_path)
                if summary.signal_validation_json_path is not None
                else None
            ),
            "portfolio_recipe_json_path": (
                str(summary.portfolio_recipe_json_path)
                if summary.portfolio_recipe_json_path is not None
                else None
            ),
            "backtest_result_json_path": (
                str(summary.backtest_result_json_path)
                if summary.backtest_result_json_path is not None
                else None
            ),
            "summary_path": str(summary.summary_path) if summary.summary_path is not None else None,
            "experiment_card_path": (
                str(summary.experiment_card_path)
                if summary.experiment_card_path is not None
                else None
            ),
        },
    }


def _preferred_case_artifact_pointer(summary: CampaignCaseProfileSummary | None) -> str | None:
    if summary is None:
        return None
    for path in (
        summary.metrics_path,
        summary.summary_path,
        summary.experiment_card_path,
        summary.run_manifest_path,
        summary.output_dir,
        summary.signal_validation_json_path,
        summary.portfolio_recipe_json_path,
        summary.backtest_result_json_path,
        summary.factor_definition_json_path,
    ):
        if path is not None:
            return str(path)
    return None


def _profile_pair_case_artifact_hint(
    *,
    case_name: str,
    from_profile: str,
    from_summary: CampaignCaseProfileSummary | None,
    to_profile: str,
    to_summary: CampaignCaseProfileSummary | None,
) -> str | None:
    from_pointer = _preferred_case_artifact_pointer(from_summary)
    to_pointer = _preferred_case_artifact_pointer(to_summary)
    if from_pointer is None and to_pointer is None:
        return None
    if from_pointer is not None and to_pointer is not None:
        if from_pointer == to_pointer:
            return f"{case_name}: {from_pointer}"
        return f"{case_name}: {from_profile}={from_pointer}; {to_profile}={to_pointer}"
    if to_pointer is not None:
        return f"{case_name}: {to_profile}={to_pointer}"
    return f"{case_name}: {from_profile}={from_pointer}"


def _case_artifact_pointer_hints_from_case_comparison(
    row: dict[str, object],
    *,
    profiles: list[str],
) -> list[str]:
    case_name = str(row.get("case_name") or "").strip()
    profiles_obj = row.get("profiles")
    profile_map = profiles_obj if isinstance(profiles_obj, dict) else {}
    if not profile_map:
        return []
    preferred_profile_order = list(reversed(profiles))
    hints: list[str] = []
    for profile_name in preferred_profile_order:
        payload = profile_map.get(profile_name)
        if not isinstance(payload, dict):
            continue
        artifact_paths_obj = payload.get("artifact_paths")
        artifact_paths = artifact_paths_obj if isinstance(artifact_paths_obj, dict) else {}
        for field in (
            "metrics_path",
            "summary_path",
            "experiment_card_path",
            "output_dir",
            "run_manifest_path",
            "signal_validation_json_path",
            "portfolio_recipe_json_path",
            "backtest_result_json_path",
            "factor_definition_json_path",
        ):
            pointer = str(artifact_paths.get(field) or "").strip()
            if not pointer:
                continue
            _append_unique_text(
                hints,
                f"{case_name} [{profile_name}] {pointer}",
                max_items=2,
            )
            break
    if hints:
        return hints
    for profile_name, payload in sorted(profile_map.items()):
        if not isinstance(payload, dict):
            continue
        artifact_paths_obj = payload.get("artifact_paths")
        artifact_paths = artifact_paths_obj if isinstance(artifact_paths_obj, dict) else {}
        for field in (
            "metrics_path",
            "summary_path",
            "experiment_card_path",
            "output_dir",
            "run_manifest_path",
            "signal_validation_json_path",
            "portfolio_recipe_json_path",
            "backtest_result_json_path",
            "factor_definition_json_path",
        ):
            pointer = str(artifact_paths.get(field) or "").strip()
            if not pointer:
                continue
            _append_unique_text(
                hints,
                f"{case_name} [{profile_name}] {pointer}",
                max_items=2,
            )
            break
    return hints


def _to_str_list(value: object, *, max_items: int) -> list[str]:
    if max_items <= 0 or not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        _append_unique_text(out, str(item), max_items=max_items)
        if len(out) >= max_items:
            break
    return out


def _append_unique_text(target: list[str], value: str | None, *, max_items: int) -> None:
    if max_items <= 0:
        return
    token = str(value).strip() if value is not None else ""
    if not token:
        return
    lowered = token.lower()
    for existing in target:
        if existing.lower() == lowered:
            return
    if len(target) >= max_items:
        return
    target.append(token)


def _extend_unique_text(target: list[str], values: list[str], *, max_items: int) -> None:
    for value in values:
        _append_unique_text(target, value, max_items=max_items)
        if len(target) >= max_items:
            return


def _merge_unique_text_lists(
    lhs: list[str],
    rhs: list[str],
    *,
    max_items: int,
) -> list[str]:
    out: list[str] = []
    _extend_unique_text(out, lhs, max_items=max_items)
    _extend_unique_text(out, rhs, max_items=max_items)
    return out


def _normalize_pair_mode(pair_mode: str) -> PairMode:
    mode = str(pair_mode).strip().lower()
    if mode == "adjacent":
        return "adjacent"
    if mode == "all_pairs":
        return "all_pairs"
    raise ValueError(f"pair_mode must be 'adjacent' or 'all_pairs'; received {pair_mode!r}")


def _normalize_artifact_hint_path_mode(path_mode: str) -> ArtifactHintPathMode:
    mode = str(path_mode).strip().lower()
    if mode == "relative":
        return "relative"
    if mode == "absolute":
        return "absolute"
    raise ValueError(
        f"artifact_hint_path_mode must be 'relative' or 'absolute'; received {path_mode!r}"
    )


def _render_artifact_hint_paths_in_payload(
    payload: dict[str, object],
    *,
    root_dir: Path,
    path_mode: ArtifactHintPathMode,
) -> dict[str, object]:
    if path_mode == "absolute":
        return payload
    root_prefix = str(root_dir.resolve()).rstrip("/\\")
    rendered = _render_artifact_hint_value(
        payload,
        root_prefix=root_prefix,
        in_artifact_pointer_context=False,
    )
    if isinstance(rendered, dict):
        return rendered
    return payload


def _render_artifact_hint_value(
    value: object,
    *,
    root_prefix: str,
    in_artifact_pointer_context: bool,
) -> object:
    if isinstance(value, dict):
        out: dict[str, object] = {}
        for raw_key, raw_value in value.items():
            key = str(raw_key)
            child_is_artifact_pointer_context = in_artifact_pointer_context or (
                "artifact_pointer" in key
            )
            if key == "summary_lines" and isinstance(raw_value, list):
                out[key] = [
                    _render_hint_text_relative_to_root(
                        row,
                        root_prefix=root_prefix,
                    )
                    if isinstance(row, str)
                    else row
                    for row in raw_value
                ]
                continue
            out[key] = _render_artifact_hint_value(
                raw_value,
                root_prefix=root_prefix,
                in_artifact_pointer_context=child_is_artifact_pointer_context,
            )
        return out
    if isinstance(value, list):
        return [
            _render_artifact_hint_value(
                row,
                root_prefix=root_prefix,
                in_artifact_pointer_context=in_artifact_pointer_context,
            )
            for row in value
        ]
    if isinstance(value, str) and in_artifact_pointer_context:
        return _render_hint_text_relative_to_root(
            value,
            root_prefix=root_prefix,
        )
    return value


def _render_hint_text_relative_to_root(text: str, *, root_prefix: str) -> str:
    if not text:
        return text
    if not root_prefix:
        return text
    normalized_root = str(Path(root_prefix).resolve())
    if not normalized_root:
        return text

    def _replace_path_token(match: re.Match[str]) -> str:
        token = match.group(1)
        try:
            return os.path.relpath(token, start=normalized_root)
        except ValueError:
            return token

    return _ABSOLUTE_PATH_TOKEN_RE.sub(_replace_path_token, text)


def _normalize_profiles(profiles: tuple[str, ...]) -> tuple[str, ...]:
    if not profiles:
        raise ValueError("profiles must contain at least one profile name")
    normalized: list[str] = []
    for profile in profiles:
        name = str(profile).strip()
        if not name:
            continue
        if name not in AVAILABLE_RESEARCH_EVALUATION_PROFILES:
            raise ValueError(
                "unknown research evaluation profile: "
                f"{name!r}; available={list(AVAILABLE_RESEARCH_EVALUATION_PROFILES)}"
            )
        normalized.append(name)
    if not normalized:
        raise ValueError("profiles must contain at least one non-empty profile name")
    return tuple(normalized)


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object at {path}")
    validate_level12_artifact_payload(payload, artifact_name=path.name, source=path)
    return payload


def _support_annotation(
    *,
    support_count: int,
    minimum_required_support: int,
    sparse_note: str,
    tentative_note: str,
    supported_note: str,
) -> dict[str, object]:
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


def _support_note_suffix(payload: object, *, label: str = "support") -> str:
    data = payload if isinstance(payload, dict) else {}
    note = str(data.get("support_note") or "").strip()
    if not note:
        return ""
    return f"; {label}={note}"


def _pair_scope_label_text(pair_mode_obj: object) -> str:
    mode = str(pair_mode_obj or DEFAULT_PAIR_MODE).strip().lower()
    if mode == "all_pairs":
        return "all ordered profile pairs"
    return "adjacent profiles"


def _render_profile_value_map(value_obj: object) -> str:
    value_map = value_obj if isinstance(value_obj, dict) else {}
    tokens: list[str] = []
    for profile_name in sorted(value_map):
        value = str(value_map.get(profile_name) or "").strip()
        if not value:
            continue
        tokens.append(f"{profile_name}=`{value}`")
    return "; ".join(tokens) if tokens else "none"


def _render_profile_list_map(value_obj: object) -> str:
    value_map = value_obj if isinstance(value_obj, dict) else {}
    tokens: list[str] = []
    for profile_name in sorted(value_map):
        values = _to_str_list(value_map.get(profile_name), max_items=3)
        if not values:
            continue
        tokens.append(f"{profile_name}: " + " | ".join(values))
    return "; ".join(tokens) if tokens else "none"


def _list_or_none(value: object) -> str:
    rendered = format_text_list(
        value,
        empty="none",
        separator="`, `",
        split_semicolon=False,
    )
    if rendered == "none":
        return rendered
    return f"`{rendered}`"


def _tuple_or_none(value: tuple[str, ...]) -> str:
    if not value:
        return "none"
    return ", ".join(value)
