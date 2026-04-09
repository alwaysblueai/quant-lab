"""Core Level 1/2 research-validation packaging helpers.

This module packages research workflow outputs without replay/implementability
requirements. Replay-specific add-ons live under `alpha_lab.experimental_level3`.
"""

from __future__ import annotations

import datetime
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict, cast

from alpha_lab.artifact_contracts import validate_level12_artifact_payload
from alpha_lab.key_metrics_contracts import (
    CoreSignalEvidenceMetrics,
    Level12TransitionSummaryMetrics,
    NeutralizationComparisonMetrics,
    RollingStabilityMetrics,
    UncertaintyEvidenceMetrics,
    project_campaign_profile_summary_metrics,
    project_level12_transition_summary,
    project_portfolio_validation_metrics,
    project_promotion_gate_metrics,
)
from alpha_lab.reporting.campaign_triage import CampaignTriagePayload, build_campaign_triage
from alpha_lab.reporting.display_helpers import (
    format_ci,
    parse_text_list,
    safe_text,
    to_finite_float,
)
from alpha_lab.reporting.factor_verdict import FactorVerdictPayload, build_factor_verdict
from alpha_lab.reporting.level2_portfolio_validation import (
    Level2PortfolioValidationMetricsPayload,
    Level2PortfolioValidationPackagePayload,
    Level2PortfolioValidationSummaryPayload,
    build_level2_portfolio_validation_bundle,
)
from alpha_lab.reporting.level2_promotion import Level2PromotionPayload, build_level2_promotion
from alpha_lab.research_evaluation_config import (
    DEFAULT_RESEARCH_EVALUATION_CONFIG,
    ResearchEvaluationAuditSnapshot,
    get_research_evaluation_config,
    research_evaluation_audit_snapshot,
)

RESEARCH_VALIDATION_PACKAGE_SCHEMA_VERSION = "1.0.0"
RESEARCH_VALIDATION_PACKAGE_TYPE = "alpha_lab_research_validation_package"


class ResearchValidationIdentityPayload(TypedDict):
    workflow_summary_path: str | None
    portfolio_validation_package_path: str | None


class ResearchValidationIntentPayload(TypedDict):
    config_path: str | None
    workflow_type: str
    promotion_verdict: str | None
    portfolio_validation_status: str | None
    evaluation_profile: str


class ResearchValidationEvaluationStandardPayload(TypedDict):
    profile_name: str
    snapshot: ResearchEvaluationAuditSnapshot


class UncertaintySummaryPayload(TypedDict, total=False):
    mean_ic_ci_lower: object
    mean_ic_ci_upper: object
    mean_rank_ic_ci_lower: object
    mean_rank_ic_ci_upper: object
    mean_long_short_return_ci_lower: object
    mean_long_short_return_ci_upper: object
    uncertainty_method: object
    uncertainty_confidence_level: object
    uncertainty_bootstrap_resamples: object
    uncertainty_bootstrap_block_length: object
    uncertainty_flags: list[str]


class RollingStabilitySummaryPayload(TypedDict, total=False):
    rolling_window_size: object
    rolling_ic_positive_share: object
    rolling_rank_ic_positive_share: object
    rolling_long_short_positive_share: object
    rolling_ic_min_mean: object
    rolling_rank_ic_min_mean: object
    rolling_long_short_min_mean: object
    rolling_instability_flags: list[str]


class NeutralizationComparisonPayload(TypedDict, total=False):
    raw: dict[str, object]
    neutralized: dict[str, object]
    delta: dict[str, object]
    interpretation_flags: list[str]
    interpretation_reasons: list[str]


class PromotionDecisionPayload(TypedDict):
    verdict: object
    reasons: object
    blockers: object
    source: str


class ResearchValidationResultsPayload(TypedDict):
    key_metrics: dict[str, object]
    evaluation_standard: ResearchValidationEvaluationStandardPayload
    uncertainty: UncertaintySummaryPayload
    rolling_stability: RollingStabilitySummaryPayload
    neutralization_comparison: NeutralizationComparisonPayload
    promotion_decision: dict[str, object]
    level2_promotion: Level2PromotionPayload
    portfolio_validation_summary: Level2PortfolioValidationSummaryPayload
    portfolio_validation_metrics: Level2PortfolioValidationMetricsPayload
    portfolio_validation_package: Level2PortfolioValidationPackagePayload
    factor_verdict: FactorVerdictPayload
    campaign_triage: CampaignTriagePayload
    level12_transition_summary: Level12TransitionSummaryMetrics
    status: str | None


class TrialRegistryMetadataPayload(TypedDict):
    trial_log_path: str | None
    alpha_registry_path: str | None


@dataclass(frozen=True)
class ResearchValidationArtifactRef:
    """One artifact reference included in package-level indexing."""

    name: str
    artifact_type: str
    path: str
    exists: bool
    required: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "artifact_type": self.artifact_type,
            "path": self.path,
            "exists": self.exists,
            "required": self.required,
        }


@dataclass(frozen=True)
class LoadedResearchValidationOutputs:
    """Discovered output paths and payload snippets from one completed case."""

    case_dir: Path
    workflow_summary_path: Path | None
    workflow_summary: dict[str, object] | None
    trial_log_path: Path | None
    alpha_registry_path: Path | None
    rolling_stability_path: Path | None
    portfolio_validation_summary_path: Path | None
    portfolio_validation_metrics_path: Path | None
    portfolio_validation_package_path: Path | None


@dataclass(frozen=True)
class ResearchValidationPackage:
    """Canonical Level 1/2 research-validation package for archival and review."""

    schema_version: str
    package_type: str
    created_at_utc: str
    case_id: str
    case_name: str
    case_output_dir: str
    workflow_type: str
    experiment_name: str
    identity: ResearchValidationIdentityPayload
    research_intent: ResearchValidationIntentPayload
    research_results: ResearchValidationResultsPayload
    trial_registry_metadata: TrialRegistryMetadataPayload
    artifact_index: tuple[ResearchValidationArtifactRef, ...]
    interpretation: str | None
    notes: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "package_type": self.package_type,
            "created_at_utc": self.created_at_utc,
            "case_id": self.case_id,
            "case_name": self.case_name,
            "case_output_dir": self.case_output_dir,
            "workflow_type": self.workflow_type,
            "experiment_name": self.experiment_name,
            "identity": self.identity,
            "research_intent": self.research_intent,
            "research_results": self.research_results,
            "trial_registry_metadata": self.trial_registry_metadata,
            "artifact_index": [x.to_dict() for x in self.artifact_index],
            "interpretation": self.interpretation,
            "notes": self.notes,
        }


def load_research_validation_outputs(
    case_output_dir: str | Path,
) -> LoadedResearchValidationOutputs:
    """Discover Level 1/2 research outputs without replay assumptions."""
    case_dir = Path(case_output_dir).resolve()
    if not case_dir.exists():
        raise FileNotFoundError(f"case output directory does not exist: {case_dir}")
    if not case_dir.is_dir():
        raise ValueError(f"case output path is not a directory: {case_dir}")

    workflow_path = _resolve_workflow_summary_path(case_dir)
    workflow_payload = _read_json_object(workflow_path) if workflow_path is not None else None
    outputs_payload = _coerce_mapping((workflow_payload or {}).get("outputs"))

    trial_log = _resolve_optional_path(outputs_payload.get("trial_log"), case_dir / "trial_log.csv")
    alpha_registry = _resolve_optional_path(
        outputs_payload.get("alpha_registry"), case_dir / "alpha_registry.csv"
    )
    rolling_stability = _resolve_optional_path(
        outputs_payload.get("rolling_stability"),
        case_dir / "rolling_stability.csv",
    )
    portfolio_validation_summary = _resolve_optional_path(
        outputs_payload.get("portfolio_validation_summary"),
        case_dir / "level2_portfolio_validation" / "portfolio_validation_summary.json",
    )
    portfolio_validation_metrics = _resolve_optional_path(
        outputs_payload.get("portfolio_validation_metrics"),
        case_dir / "level2_portfolio_validation" / "portfolio_validation_metrics.json",
    )
    portfolio_validation_package = _resolve_optional_path(
        outputs_payload.get("portfolio_validation_package"),
        case_dir / "level2_portfolio_validation" / "portfolio_validation_package.json",
    )

    return LoadedResearchValidationOutputs(
        case_dir=case_dir,
        workflow_summary_path=workflow_path,
        workflow_summary=workflow_payload,
        trial_log_path=trial_log,
        alpha_registry_path=alpha_registry,
        rolling_stability_path=rolling_stability,
        portfolio_validation_summary_path=portfolio_validation_summary,
        portfolio_validation_metrics_path=portfolio_validation_metrics,
        portfolio_validation_package_path=portfolio_validation_package,
    )


def build_research_validation_package(
    case_output_dir: str | Path,
    *,
    case_id: str,
    case_name: str,
    interpretation: str | None = None,
    notes: str | None = None,
) -> ResearchValidationPackage:
    """Build one core research-validation package from existing case artifacts."""
    loaded = load_research_validation_outputs(case_output_dir)
    workflow = loaded.workflow_summary or {}
    workflow_type = _safe_str(workflow.get("workflow")) or "unknown"
    experiment_name = _safe_str(workflow.get("experiment_name")) or case_id
    workflow_promotion_decision = _coerce_mapping(workflow.get("promotion_decision"))
    key_metrics = _coerce_mapping(workflow.get("key_metrics"))
    profile_name = (
        _safe_str(key_metrics.get("research_evaluation_profile"))
        or DEFAULT_RESEARCH_EVALUATION_CONFIG.profile_name
    )
    try:
        evaluation_config = get_research_evaluation_config(profile_name)
    except ValueError:
        evaluation_config = DEFAULT_RESEARCH_EVALUATION_CONFIG
    raw_evaluation_snapshot = _coerce_mapping(key_metrics.get("research_evaluation_snapshot"))
    evaluation_snapshot = (
        cast(ResearchEvaluationAuditSnapshot, raw_evaluation_snapshot)
        if raw_evaluation_snapshot
        else research_evaluation_audit_snapshot(evaluation_config)
    )

    factor_verdict = build_factor_verdict(
        key_metrics,
        thresholds=evaluation_config.factor_verdict,
    ).to_dict()
    triage_metrics = dict(key_metrics)
    triage_metrics.setdefault("factor_verdict", factor_verdict.get("label"))
    triage_metrics.setdefault("factor_verdict_reasons", factor_verdict.get("reasons"))
    status = _safe_str(workflow.get("status"))
    campaign_triage = build_campaign_triage(
        triage_metrics,
        status=status or "success",
        thresholds=evaluation_config.campaign_triage,
    ).to_dict()
    promotion_metrics = dict(triage_metrics)
    promotion_metrics.setdefault("campaign_triage", campaign_triage.get("campaign_triage"))
    promotion_metrics.setdefault(
        "campaign_triage_reasons",
        campaign_triage.get("campaign_triage_reasons"),
    )
    level2_promotion = build_level2_promotion(
        promotion_metrics,
        status=status or "success",
        thresholds=evaluation_config.level2_promotion,
    ).to_dict()
    promotion_gate_metrics = project_promotion_gate_metrics(promotion_metrics)
    core_signal_metrics = promotion_gate_metrics["core"]
    uncertainty_metrics = promotion_gate_metrics["uncertainty"]
    rolling_metrics = promotion_gate_metrics["rolling"]
    neutralization_metrics = promotion_gate_metrics["neutralization"]
    uncertainty = _extract_uncertainty_summary(uncertainty_metrics)
    rolling_stability = _extract_rolling_stability_summary(rolling_metrics)
    neutralization_comparison = _extract_neutralization_comparison(
        neutralization_metrics,
        core_signal_metrics=core_signal_metrics,
    )
    promotion_decision: dict[str, object] = (
        workflow_promotion_decision
        if workflow_promotion_decision
        else {
            "verdict": level2_promotion.get("promotion_decision"),
            "reasons": level2_promotion.get("promotion_reasons"),
            "blockers": level2_promotion.get("promotion_blockers"),
            "source": "level2_promotion_gate",
        }
    )
    portfolio_validation_summary: Level2PortfolioValidationSummaryPayload = (
        cast(
            Level2PortfolioValidationSummaryPayload,
            _read_json_object(
                loaded.portfolio_validation_summary_path,
                artifact_name="portfolio_validation_summary.json",
            ),
        )
        if loaded.portfolio_validation_summary_path is not None
        else {}
    )
    portfolio_validation_metrics: Level2PortfolioValidationMetricsPayload = (
        cast(
            Level2PortfolioValidationMetricsPayload,
            _read_json_object(
                loaded.portfolio_validation_metrics_path,
                artifact_name="portfolio_validation_metrics.json",
            ),
        )
        if loaded.portfolio_validation_metrics_path is not None
        else {}
    )
    portfolio_validation_package: Level2PortfolioValidationPackagePayload = (
        cast(
            Level2PortfolioValidationPackagePayload,
            _read_json_object(
                loaded.portfolio_validation_package_path,
                artifact_name="portfolio_validation_package.json",
            ),
        )
        if loaded.portfolio_validation_package_path is not None
        else {}
    )
    if (
        not portfolio_validation_summary
        and not portfolio_validation_metrics
        and not portfolio_validation_package
    ):
        portfolio_validation_bundle = build_level2_portfolio_validation_bundle(
            key_metrics=key_metrics,
            case_context={
                "case_id": case_id,
                "case_name": case_name,
                "case_output_dir": str(loaded.case_dir),
                "experiment_name": experiment_name,
            },
            promotion_decision=promotion_decision,
            experiment_result=None,
            config=evaluation_config.level2_portfolio_validation,
        ).to_dict()
        portfolio_validation_summary = cast(
            Level2PortfolioValidationSummaryPayload,
            _coerce_mapping(portfolio_validation_bundle.get("portfolio_validation_summary")),
        )
        portfolio_validation_metrics = cast(
            Level2PortfolioValidationMetricsPayload,
            _coerce_mapping(portfolio_validation_bundle.get("portfolio_validation_metrics")),
        )
        portfolio_validation_package = cast(
            Level2PortfolioValidationPackagePayload,
            _coerce_mapping(portfolio_validation_bundle.get("portfolio_validation_package")),
        )

    projection_input = _build_contract_projection_input(
        key_metrics,
        factor_verdict=factor_verdict,
        campaign_triage=campaign_triage,
        level2_promotion=level2_promotion,
        promotion_decision=workflow_promotion_decision,
        uncertainty=uncertainty,
        rolling_stability=rolling_stability,
        neutralization_comparison=neutralization_comparison,
        portfolio_validation_summary=portfolio_validation_summary,
        portfolio_validation_metrics=portfolio_validation_metrics,
    )
    profile_summary_metrics = project_campaign_profile_summary_metrics(projection_input)
    portfolio_validation_contract = project_portfolio_validation_metrics(projection_input)
    level12_transition_summary = project_level12_transition_summary(projection_input)
    fallback_reasons: object = (
        list(profile_summary_metrics["promotion_reasons"])
        if profile_summary_metrics["promotion_reasons"]
        else level2_promotion.get("promotion_reasons")
    )
    fallback_blockers: object = (
        list(profile_summary_metrics["promotion_blockers"])
        if profile_summary_metrics["promotion_blockers"]
        else level2_promotion.get("promotion_blockers")
    )
    promotion_decision = (
        workflow_promotion_decision
        if workflow_promotion_decision
        else {
            "verdict": profile_summary_metrics["promotion_decision"]
            or level2_promotion.get("promotion_decision"),
            "reasons": fallback_reasons,
            "blockers": fallback_blockers,
            "source": "level2_promotion_gate",
        }
    )
    research_intent: ResearchValidationIntentPayload = {
        "config_path": _safe_str(workflow.get("config_path")),
        "workflow_type": workflow_type,
        "promotion_verdict": (
            _safe_str(promotion_decision.get("verdict"))
            or profile_summary_metrics["promotion_decision"]
            or _safe_str(level2_promotion.get("promotion_decision"))
        ),
        "portfolio_validation_status": _safe_str(
            portfolio_validation_summary.get("validation_status")
        )
        or profile_summary_metrics["portfolio_validation_status"]
        or portfolio_validation_contract["portfolio_validation_status"],
        "evaluation_profile": evaluation_config.profile_name,
    }
    research_results: ResearchValidationResultsPayload = {
        "key_metrics": key_metrics,
        "evaluation_standard": {
            "profile_name": evaluation_config.profile_name,
            "snapshot": evaluation_snapshot,
        },
        "uncertainty": uncertainty,
        "rolling_stability": rolling_stability,
        "neutralization_comparison": neutralization_comparison,
        "promotion_decision": promotion_decision,
        "level2_promotion": level2_promotion,
        "portfolio_validation_summary": portfolio_validation_summary,
        "portfolio_validation_metrics": portfolio_validation_metrics,
        "portfolio_validation_package": portfolio_validation_package,
        "factor_verdict": factor_verdict,
        "campaign_triage": campaign_triage,
        "level12_transition_summary": level12_transition_summary,
        "status": status,
    }
    trial_registry_metadata: TrialRegistryMetadataPayload = {
        "trial_log_path": str(loaded.trial_log_path) if loaded.trial_log_path else None,
        "alpha_registry_path": str(loaded.alpha_registry_path)
        if loaded.alpha_registry_path
        else None,
    }
    identity: ResearchValidationIdentityPayload = {
        "workflow_summary_path": str(loaded.workflow_summary_path)
        if loaded.workflow_summary_path
        else None,
        "portfolio_validation_package_path": str(loaded.portfolio_validation_package_path)
        if loaded.portfolio_validation_package_path
        else None,
    }

    package = ResearchValidationPackage(
        schema_version=RESEARCH_VALIDATION_PACKAGE_SCHEMA_VERSION,
        package_type=RESEARCH_VALIDATION_PACKAGE_TYPE,
        created_at_utc=datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        case_id=case_id,
        case_name=case_name,
        case_output_dir=str(loaded.case_dir),
        workflow_type=workflow_type,
        experiment_name=experiment_name,
        identity=identity,
        research_intent=research_intent,
        research_results=research_results,
        trial_registry_metadata=trial_registry_metadata,
        artifact_index=_build_artifact_index(loaded),
        interpretation=interpretation,
        notes=notes,
    )
    validate_level12_artifact_payload(
        package.to_dict(),
        artifact_name="research_validation_package.json",
    )
    return package


def export_research_validation_package(
    package: ResearchValidationPackage,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write a core research-validation package to JSON and Markdown artifacts."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "research_validation_package.json"
    package_payload = package.to_dict()
    validate_level12_artifact_payload(
        package_payload,
        artifact_name="research_validation_package.json",
        source=json_path,
    )
    json_path.write_text(
        json.dumps(package_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    md_path = out_dir / "research_validation_package.md"
    md_path.write_text(_package_markdown(package), encoding="utf-8")
    exported: dict[str, Path] = {"json": json_path, "markdown": md_path}

    portfolio_summary = _coerce_mapping(
        package.research_results.get("portfolio_validation_summary")
    )
    portfolio_metrics = _coerce_mapping(
        package.research_results.get("portfolio_validation_metrics")
    )
    portfolio_package = _coerce_mapping(
        package.research_results.get("portfolio_validation_package")
    )
    if portfolio_summary or portfolio_metrics or portfolio_package:
        level2_dir = out_dir / "level2_portfolio_validation"
        level2_dir.mkdir(parents=True, exist_ok=True)
        summary_path = level2_dir / "portfolio_validation_summary.json"
        metrics_path = level2_dir / "portfolio_validation_metrics.json"
        package_path = level2_dir / "portfolio_validation_package.json"
        validate_level12_artifact_payload(
            portfolio_summary,
            artifact_name="portfolio_validation_summary.json",
            source=summary_path,
        )
        validate_level12_artifact_payload(
            portfolio_metrics,
            artifact_name="portfolio_validation_metrics.json",
            source=metrics_path,
        )
        validate_level12_artifact_payload(
            portfolio_package,
            artifact_name="portfolio_validation_package.json",
            source=package_path,
        )
        summary_path.write_text(
            json.dumps(portfolio_summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        metrics_path.write_text(
            json.dumps(portfolio_metrics, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        package_path.write_text(
            json.dumps(portfolio_package, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        exported["portfolio_validation_summary"] = summary_path
        exported["portfolio_validation_metrics"] = metrics_path
        exported["portfolio_validation_package"] = package_path

    return exported


def _package_markdown(package: ResearchValidationPackage) -> str:
    research_results = _coerce_mapping(package.research_results)
    key_metrics = _coerce_mapping(research_results.get("key_metrics"))
    verdict_payload = _coerce_mapping(research_results.get("factor_verdict"))
    triage_payload = _coerce_mapping(research_results.get("campaign_triage"))
    level2_promotion = _coerce_mapping(research_results.get("level2_promotion"))
    promotion_decision_payload = _coerce_mapping(research_results.get("promotion_decision"))
    portfolio_validation_summary = _coerce_mapping(
        research_results.get("portfolio_validation_summary")
    )
    portfolio_validation_metrics = _coerce_mapping(
        research_results.get("portfolio_validation_metrics")
    )
    uncertainty_payload = _coerce_mapping(research_results.get("uncertainty"))
    rolling_payload = _coerce_mapping(research_results.get("rolling_stability"))
    neutralization_payload = _coerce_mapping(research_results.get("neutralization_comparison"))
    contract_projection_input = _build_contract_projection_input(
        key_metrics,
        factor_verdict=verdict_payload,
        campaign_triage=triage_payload,
        level2_promotion=level2_promotion,
        promotion_decision=promotion_decision_payload,
        uncertainty=uncertainty_payload,
        rolling_stability=rolling_payload,
        neutralization_comparison=neutralization_payload,
        portfolio_validation_summary=portfolio_validation_summary,
        portfolio_validation_metrics=portfolio_validation_metrics,
    )
    promotion_gate_metrics = project_promotion_gate_metrics(contract_projection_input)
    profile_summary_metrics = project_campaign_profile_summary_metrics(contract_projection_input)
    portfolio_validation_contract = project_portfolio_validation_metrics(
        contract_projection_input
    )
    level12_transition = project_level12_transition_summary(contract_projection_input)
    uncertainty_metrics = promotion_gate_metrics["uncertainty"]
    rolling_metrics = promotion_gate_metrics["rolling"]
    neutralization_contract = promotion_gate_metrics["neutralization"]
    neutralization_for_markdown = _extract_neutralization_comparison(
        neutralization_contract,
        core_signal_metrics=promotion_gate_metrics["core"],
    )
    if not neutralization_for_markdown and neutralization_payload:
        neutralization_for_markdown = cast(
            NeutralizationComparisonPayload,
            neutralization_payload,
        )

    lines = [
        f"# Research Validation Package: {package.case_name}",
        "",
        f"- Case ID: `{package.case_id}`",
        f"- Experiment: `{package.experiment_name}`",
        f"- Workflow: `{package.workflow_type}`",
        "",
        "## Key Metrics",
    ]
    if key_metrics:
        for key, value in sorted(key_metrics.items()):
            lines.append(f"- `{key}`: `{value}`")
    else:
        lines.append("- None")

    evaluation_payload = _coerce_mapping(package.research_results.get("evaluation_standard"))
    if evaluation_payload:
        snapshot = _coerce_mapping(evaluation_payload.get("snapshot"))
        lines.extend(
            [
                "",
                "## Evaluation Standard",
                (
                    "- Profile: "
                    f"`{_safe_str(evaluation_payload.get('profile_name')) or 'N/A'}`"
                ),
            ]
        )
        uncertainty_cfg = _coerce_mapping(snapshot.get("uncertainty"))
        rolling_cfg = _coerce_mapping(snapshot.get("rolling_stability"))
        if uncertainty_cfg:
            lines.append(
                "- Uncertainty method: "
                + _format_value(uncertainty_cfg.get("method"))
            )
            lines.append(
                "- Uncertainty CI level: "
                + _format_value(uncertainty_cfg.get("confidence_level"))
            )
        if rolling_cfg:
            lines.append(
                "- Rolling window size: "
                + _format_value(rolling_cfg.get("rolling_window_size"))
            )

    if verdict_payload:
        lines.extend(
            [
                "",
                "## Factor Verdict",
                f"- Verdict: `{_safe_str(verdict_payload.get('label')) or 'N/A'}`",
            ]
        )
        reasons = _to_text_list(verdict_payload.get("reasons"))
        if reasons:
            for reason in reasons:
                lines.append(f"- Reason: {reason}")
        else:
            lines.append("- Reason: none")

    if triage_payload:
        lines.extend(
            [
                "",
                "## Campaign Triage",
                (
                    "- Triage: "
                    f"`{_safe_str(triage_payload.get('campaign_triage')) or 'N/A'}`"
                ),
                (
                    "- Priority: "
                    + _format_value(triage_payload.get("campaign_triage_priority"))
                ),
                (
                    "- Ranking metrics (ICIR / L-S / rolling+ min): "
                    f"{_format_value(triage_payload.get('campaign_rank_primary_metric'))} / "
                    f"{_format_value(triage_payload.get('campaign_rank_secondary_metric'))} / "
                    f"{_format_value(triage_payload.get('campaign_rank_stability_metric'))}"
                ),
                (
                    "- Support/Risk count: "
                    f"{_format_value(triage_payload.get('campaign_rank_support_count'))} / "
                    f"{_format_value(triage_payload.get('campaign_rank_risk_count'))}"
                ),
            ]
        )
        triage_reasons = _to_text_list(triage_payload.get("campaign_triage_reasons"))
        if triage_reasons:
            for reason in triage_reasons:
                lines.append(f"- Reason: {reason}")
        else:
            lines.append("- Reason: none")

    if level2_promotion or profile_summary_metrics["promotion_decision"]:
        promotion_decision = (
            profile_summary_metrics["promotion_decision"]
            or _safe_str(level2_promotion.get("promotion_decision"))
            or "N/A"
        )
        lines.extend(
            [
                "",
                "## Level 2 Promotion Gate",
                f"- Decision: `{promotion_decision}`",
            ]
        )
        reasons = list(profile_summary_metrics["promotion_reasons"]) or _to_text_list(
            level2_promotion.get("promotion_reasons")
        )
        if reasons:
            for reason in reasons:
                lines.append(f"- Reason: {reason}")
        else:
            lines.append("- Reason: none")
        blockers = list(profile_summary_metrics["promotion_blockers"]) or _to_text_list(
            level2_promotion.get("promotion_blockers")
        )
        if blockers:
            for blocker in blockers:
                lines.append(f"- Blocker: {blocker}")
        else:
            lines.append("- Blocker: none")

    transition_label = _safe_str(level12_transition.get("transition_label"))
    transition_interpretation = _safe_str(level12_transition.get("transition_interpretation"))
    transition_reasons = _to_text_list(level12_transition.get("key_transition_reasons"))
    transition_confirmation_note = _safe_str(
        level12_transition.get("confirmation_vs_degradation_note")
    )
    if (
        transition_label
        or transition_interpretation
        or transition_reasons
        or transition_confirmation_note
    ):
        lines.extend(
            [
                "",
                "## Level 1 to Level 2 Transition",
                f"- Transition label: `{transition_label or 'N/A'}`",
                f"- Interpretation: {transition_interpretation or 'N/A'}",
                (
                    "- Level 1 status: "
                    + _format_value(level12_transition.get("level1_status"))
                ),
                (
                    "- Level 2 status: "
                    + _format_value(level12_transition.get("level2_status"))
                ),
                (
                    "- Confirmation vs degradation: "
                    + _format_value(transition_confirmation_note)
                ),
            ]
        )
        if transition_reasons:
            for reason in transition_reasons:
                lines.append(f"- Transition reason: {reason}")
        else:
            lines.append("- Transition reason: none")

    if portfolio_validation_summary or profile_summary_metrics["portfolio_validation_status"]:
        validation_status = (
            profile_summary_metrics["portfolio_validation_status"]
            or _safe_str(portfolio_validation_summary.get("validation_status"))
            or "N/A"
        )
        validation_recommendation = (
            profile_summary_metrics["portfolio_validation_recommendation"]
            or _safe_str(portfolio_validation_summary.get("recommendation"))
            or "N/A"
        )
        robustness = _coerce_mapping(
            portfolio_validation_summary.get("portfolio_robustness_summary")
        )
        robustness_label = (
            portfolio_validation_contract["portfolio_validation_robustness_label"]
            or _safe_str(robustness.get("taxonomy_label"))
        )
        baseline_return = _format_value(
            portfolio_validation_summary.get("base_mean_portfolio_return")
        )
        baseline_turnover = _format_value(
            portfolio_validation_summary.get("base_mean_turnover")
        )
        baseline_cost_adjusted = _format_value(
            portfolio_validation_summary.get(
                "base_cost_adjusted_return_review_rate"
            )
        )
        lines.extend(
            [
                "",
                "## Level 2 Portfolio Validation",
                f"- Validation status: `{validation_status}`",
                f"- Recommendation: `{validation_recommendation}`",
                (
                    "- Remains credible at portfolio level: "
                    + _format_value(
                        portfolio_validation_summary.get(
                            "remains_credible_at_portfolio_level"
                        )
                    )
                ),
                (
                    "- Baseline (return / turnover / cost-adjusted @ review rate): "
                    f"{baseline_return} / "
                    f"{baseline_turnover} / "
                    f"{baseline_cost_adjusted}"
                ),
            ]
        )
        lines.append(
            "- Portfolio robustness taxonomy: "
            f"`{_safe_str(robustness_label) or 'N/A'}`"
        )
        scenario_notes = list(
            portfolio_validation_contract["portfolio_validation_scenario_sensitivity_notes"]
        ) or _to_text_list(robustness.get("scenario_sensitivity_notes"))
        if scenario_notes:
            for note in scenario_notes:
                lines.append(f"- Scenario sensitivity: {note}")
        support_reasons = list(
            portfolio_validation_contract["portfolio_validation_support_reasons"]
        ) or _to_text_list(robustness.get("support_reasons"))
        if support_reasons:
            for reason in support_reasons:
                lines.append(f"- Robustness support: {reason}")
        fragility_reasons = list(
            portfolio_validation_contract["portfolio_validation_fragility_reasons"]
        ) or _to_text_list(robustness.get("fragility_reasons"))
        if fragility_reasons:
            for reason in fragility_reasons:
                lines.append(f"- Robustness fragility: {reason}")
        benchmark_support_note = (
            portfolio_validation_contract["portfolio_validation_benchmark_support_note"]
            or _safe_str(robustness.get("benchmark_relative_support_note"))
        )
        if benchmark_support_note is not None:
            lines.append(f"- Benchmark-relative support note: {benchmark_support_note}")
        cost_support_note = (
            portfolio_validation_contract["portfolio_validation_cost_sensitivity_note"]
            or _safe_str(robustness.get("cost_sensitivity_note"))
        )
        if cost_support_note is not None:
            lines.append(f"- Cost sensitivity note: {cost_support_note}")
        concentration_note = (
            portfolio_validation_contract["portfolio_validation_concentration_turnover_note"]
            or _safe_str(robustness.get("concentration_turnover_risk_note"))
        )
        if concentration_note is not None:
            lines.append(f"- Concentration/turnover note: {concentration_note}")
        risks = list(profile_summary_metrics["portfolio_validation_major_risks"]) or _to_text_list(
            portfolio_validation_summary.get("major_risks")
        )
        if risks:
            for risk in risks:
                lines.append(f"- Risk: {risk}")
        else:
            lines.append("- Risk: none")
        caveats = _to_text_list(portfolio_validation_summary.get("major_caveats"))
        if caveats:
            for caveat in caveats:
                lines.append(f"- Caveat: {caveat}")
        else:
            lines.append("- Caveat: none")

        if portfolio_validation_metrics:
            benchmark = _coerce_mapping(
                portfolio_validation_metrics.get("benchmark_relative_evaluation")
            )
            if benchmark or any(
                value is not None
                for value in (
                    portfolio_validation_contract[
                        "portfolio_validation_benchmark_relative_status"
                    ],
                    portfolio_validation_contract[
                        "portfolio_validation_benchmark_relative_assessment"
                    ],
                    portfolio_validation_contract[
                        "portfolio_validation_benchmark_excess_return"
                    ],
                    portfolio_validation_contract[
                        "portfolio_validation_benchmark_tracking_error"
                    ],
                    portfolio_validation_contract["benchmark_active_return"],
                    portfolio_validation_contract["benchmark_information_ratio"],
                )
            ):
                lines.append(
                    "- Benchmark-relative evaluation status: "
                    + _format_value(
                        portfolio_validation_contract[
                            "portfolio_validation_benchmark_relative_status"
                        ]
                    )
                )
                lines.append(
                    "- Benchmark-relative assessment: "
                    + _format_value(
                        portfolio_validation_contract[
                            "portfolio_validation_benchmark_relative_assessment"
                        ]
                    )
                )
                lines.append(
                    "- Benchmark-relative excess / active return: "
                    + _format_value(
                        portfolio_validation_contract[
                            "portfolio_validation_benchmark_excess_return"
                        ]
                    )
                    + " / "
                    + _format_value(
                        portfolio_validation_contract["benchmark_active_return"]
                    )
                )
                lines.append(
                    "- Benchmark-relative information ratio / tracking error: "
                    + _format_value(
                        portfolio_validation_contract["benchmark_information_ratio"]
                    )
                    + " / "
                    + _format_value(
                        portfolio_validation_contract[
                            "portfolio_validation_benchmark_tracking_error"
                        ]
                    )
                )
                benchmark_risks = _to_text_list(benchmark.get("risk_flags"))
                if benchmark_risks:
                    for benchmark_risk in benchmark_risks:
                        lines.append(f"- Benchmark-relative risk: {benchmark_risk}")
            turnover_summary = _coerce_mapping(
                portfolio_validation_metrics.get("turnover_summary")
            )
            if turnover_summary:
                lines.append(
                    "- Turnover sensitivity range (mean): "
                    f"{_format_value(turnover_summary.get('scenario_mean_turnover_min'))} -> "
                    f"{_format_value(turnover_summary.get('scenario_mean_turnover_max'))}"
                )

    if uncertainty_payload or any(
        value is not None
        for value in (
            uncertainty_metrics["mean_ic_ci_lower"],
            uncertainty_metrics["mean_ic_ci_upper"],
            uncertainty_metrics["mean_rank_ic_ci_lower"],
            uncertainty_metrics["mean_rank_ic_ci_upper"],
            uncertainty_metrics["mean_long_short_return_ci_lower"],
            uncertainty_metrics["mean_long_short_return_ci_upper"],
            uncertainty_metrics["uncertainty_method"],
            uncertainty_metrics["uncertainty_confidence_level"],
            uncertainty_metrics["uncertainty_bootstrap_resamples"],
            uncertainty_metrics["uncertainty_bootstrap_block_length"],
        )
    ) or uncertainty_metrics["uncertainty_flags"]:
        lines.extend(["", "## Uncertainty"])
        lines.append(
            "- Method / CI level / bootstrap resamples / block length: "
            f"{_format_value(uncertainty_metrics['uncertainty_method'])} / "
            f"{_format_value(uncertainty_metrics['uncertainty_confidence_level'])} / "
            f"{_format_value(uncertainty_metrics['uncertainty_bootstrap_resamples'])} / "
            f"{_format_value(uncertainty_metrics['uncertainty_bootstrap_block_length'])}"
        )
        lines.append(
            "- Mean IC 95% CI: "
            + format_ci(
                uncertainty_metrics["mean_ic_ci_lower"],
                uncertainty_metrics["mean_ic_ci_upper"],
            )
        )
        lines.append(
            "- Mean Rank IC 95% CI: "
            + format_ci(
                uncertainty_metrics["mean_rank_ic_ci_lower"],
                uncertainty_metrics["mean_rank_ic_ci_upper"],
            )
        )
        lines.append(
            "- Mean Long-Short Return 95% CI: "
            + format_ci(
                uncertainty_metrics["mean_long_short_return_ci_lower"],
                uncertainty_metrics["mean_long_short_return_ci_upper"],
            )
        )
        flags = list(uncertainty_metrics["uncertainty_flags"])
        if flags:
            lines.append("- Uncertainty Flags: " + ", ".join(flags))
        else:
            lines.append("- Uncertainty Flags: none")

    if rolling_payload or any(
        value is not None
        for value in (
            rolling_metrics["rolling_window_size"],
            rolling_metrics["rolling_ic_positive_share"],
            rolling_metrics["rolling_rank_ic_positive_share"],
            rolling_metrics["rolling_long_short_positive_share"],
            rolling_metrics["rolling_ic_min_mean"],
            rolling_metrics["rolling_rank_ic_min_mean"],
            rolling_metrics["rolling_long_short_min_mean"],
        )
    ) or rolling_metrics["rolling_instability_flags"]:
        lines.extend(["", "## Rolling Stability"])
        lines.append(
            "- Rolling window size: "
            + _format_value(rolling_metrics["rolling_window_size"])
        )
        lines.append(
            "- Rolling positive share (IC / RankIC / long-short): "
            f"{_format_value(rolling_metrics['rolling_ic_positive_share'])} / "
            f"{_format_value(rolling_metrics['rolling_rank_ic_positive_share'])} / "
            f"{_format_value(rolling_metrics['rolling_long_short_positive_share'])}"
        )
        lines.append(
            "- Worst rolling mean (IC / RankIC / long-short): "
            f"{_format_value(rolling_metrics['rolling_ic_min_mean'])} / "
            f"{_format_value(rolling_metrics['rolling_rank_ic_min_mean'])} / "
            f"{_format_value(rolling_metrics['rolling_long_short_min_mean'])}"
        )
        flags = list(rolling_metrics["rolling_instability_flags"])
        if flags:
            lines.append("- Rolling Stability Flags: " + ", ".join(flags))
        else:
            lines.append("- Rolling Stability Flags: none")

    if neutralization_for_markdown:
        raw = _coerce_mapping(neutralization_for_markdown.get("raw"))
        neutralized = _coerce_mapping(neutralization_for_markdown.get("neutralized"))
        delta = _coerce_mapping(neutralization_for_markdown.get("delta"))
        lines.extend(["", "## Raw vs Neutralized Comparison"])
        lines.append(
            "- Mean IC (raw -> neutralized, delta): "
            + _format_transition(
                raw.get("mean_ic"),
                neutralized.get("mean_ic"),
                delta.get("mean_ic_delta"),
            )
        )
        lines.append(
            "- Mean RankIC (raw -> neutralized, delta): "
            + _format_transition(
                raw.get("mean_rank_ic"),
                neutralized.get("mean_rank_ic"),
                delta.get("mean_rank_ic_delta"),
            )
        )
        lines.append(
            "- Mean long-short return (raw -> neutralized, delta): "
            + _format_transition(
                raw.get("mean_long_short_return"),
                neutralized.get("mean_long_short_return"),
                delta.get("mean_long_short_return_delta"),
            )
        )
        lines.append(
            "- ICIR (raw -> neutralized, delta): "
            + _format_transition(
                raw.get("ic_ir"),
                neutralized.get("ic_ir"),
                delta.get("ic_ir_delta"),
            )
        )
        lines.append(
            "- Validity min ratio (raw -> neutralized, delta): "
            + _format_transition(
                raw.get("valid_ratio_min"),
                neutralized.get("valid_ratio_min"),
                delta.get("valid_ratio_min_delta"),
            )
        )
        lines.append(
            "- Coverage mean ratio (raw -> neutralized, delta): "
            + _format_transition(
                raw.get("eval_coverage_ratio_mean"),
                neutralized.get("eval_coverage_ratio_mean"),
                delta.get("eval_coverage_ratio_mean_delta"),
            )
        )
        lines.append(
            "- Uncertainty overlap-zero count (raw -> neutralized, delta): "
            + _format_transition(
                raw.get("uncertainty_overlap_zero_count"),
                neutralized.get("uncertainty_overlap_zero_count"),
                delta.get("uncertainty_overlap_zero_count_delta"),
            )
        )
        lines.append(
            "- Rolling positive-share min (raw -> neutralized, delta): "
            + _format_transition(
                raw.get("rolling_positive_share_min"),
                neutralized.get("rolling_positive_share_min"),
                delta.get("rolling_positive_share_min_delta"),
            )
        )
        lines.append(
            "- Rolling worst-mean min (raw -> neutralized, delta): "
            + _format_transition(
                raw.get("rolling_worst_mean_min"),
                neutralized.get("rolling_worst_mean_min"),
                delta.get("rolling_worst_mean_min_delta"),
            )
        )
        flags = _to_text_list(neutralization_for_markdown.get("interpretation_flags"))
        if flags:
            lines.append("- Interpretation Flags: " + ", ".join(flags))
        else:
            lines.append("- Interpretation Flags: none")
        reasons = _to_text_list(neutralization_for_markdown.get("interpretation_reasons"))
        if reasons:
            for reason in reasons:
                lines.append(f"- Interpretation Reason: {reason}")
        else:
            lines.append("- Interpretation Reason: none")

    lines.extend(
        [
            "",
            "## Artifacts",
        ]
    )
    if package.artifact_index:
        for artifact in package.artifact_index:
            lines.append(
                f"- `{artifact.name}` ({artifact.artifact_type}) exists={artifact.exists} "
                f"required={artifact.required}"
            )
    else:
        lines.append("- None")

    if package.interpretation:
        lines.extend(["", "## Interpretation", package.interpretation])
    if package.notes:
        lines.extend(["", "## Notes", package.notes])
    lines.append("")
    return "\n".join(lines)


def _build_artifact_index(
    loaded: LoadedResearchValidationOutputs,
) -> tuple[ResearchValidationArtifactRef, ...]:
    refs = [
        ResearchValidationArtifactRef(
            name="workflow_summary_json",
            artifact_type="workflow_summary",
            path=str(loaded.workflow_summary_path) if loaded.workflow_summary_path else "",
            exists=loaded.workflow_summary_path is not None,
            required=True,
        ),
        ResearchValidationArtifactRef(
            name="trial_log_csv",
            artifact_type="trial_log",
            path=str(loaded.trial_log_path) if loaded.trial_log_path else "",
            exists=loaded.trial_log_path is not None,
            required=False,
        ),
        ResearchValidationArtifactRef(
            name="alpha_registry_csv",
            artifact_type="alpha_registry",
            path=str(loaded.alpha_registry_path) if loaded.alpha_registry_path else "",
            exists=loaded.alpha_registry_path is not None,
            required=False,
        ),
        ResearchValidationArtifactRef(
            name="rolling_stability_csv",
            artifact_type="rolling_stability",
            path=str(loaded.rolling_stability_path) if loaded.rolling_stability_path else "",
            exists=loaded.rolling_stability_path is not None,
            required=False,
        ),
        ResearchValidationArtifactRef(
            name="portfolio_validation_summary_json",
            artifact_type="level2_portfolio_validation",
            path=(
                str(loaded.portfolio_validation_summary_path)
                if loaded.portfolio_validation_summary_path
                else ""
            ),
            exists=loaded.portfolio_validation_summary_path is not None,
            required=False,
        ),
        ResearchValidationArtifactRef(
            name="portfolio_validation_metrics_json",
            artifact_type="level2_portfolio_validation",
            path=(
                str(loaded.portfolio_validation_metrics_path)
                if loaded.portfolio_validation_metrics_path
                else ""
            ),
            exists=loaded.portfolio_validation_metrics_path is not None,
            required=False,
        ),
        ResearchValidationArtifactRef(
            name="portfolio_validation_package_json",
            artifact_type="level2_portfolio_validation",
            path=(
                str(loaded.portfolio_validation_package_path)
                if loaded.portfolio_validation_package_path
                else ""
            ),
            exists=loaded.portfolio_validation_package_path is not None,
            required=False,
        ),
    ]
    return tuple(refs)


def _resolve_workflow_summary_path(case_dir: Path) -> Path | None:
    candidates = sorted(case_dir.glob("*_workflow_summary.json"))
    return candidates[0] if candidates else None


def _resolve_optional_path(path_value: object, fallback: Path | None) -> Path | None:
    if isinstance(path_value, str) and path_value.strip():
        candidate = Path(path_value).expanduser().resolve()
        if candidate.exists():
            return candidate
    if fallback is not None and fallback.exists():
        return fallback.resolve()
    return None


def _coerce_mapping(raw: object) -> dict[str, object]:
    return dict(raw) if isinstance(raw, Mapping) else {}


def _read_json_object(
    path: Path | None,
    *,
    artifact_name: str | None = None,
) -> dict[str, object]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    if artifact_name is not None:
        validate_level12_artifact_payload(
            payload,
            artifact_name=artifact_name,
            source=path,
        )
    return payload


def _safe_str(value: object) -> str | None:
    return safe_text(value)


def _to_text_list(value: object) -> list[str]:
    return parse_text_list(value)


def _build_contract_projection_input(
    key_metrics: Mapping[str, object],
    *,
    factor_verdict: Mapping[str, object] | None = None,
    campaign_triage: Mapping[str, object] | None = None,
    level2_promotion: Mapping[str, object] | None = None,
    promotion_decision: Mapping[str, object] | None = None,
    uncertainty: Mapping[str, object] | None = None,
    rolling_stability: Mapping[str, object] | None = None,
    neutralization_comparison: Mapping[str, object] | None = None,
    portfolio_validation_summary: Mapping[str, object] | None = None,
    portfolio_validation_metrics: Mapping[str, object] | None = None,
) -> dict[str, object]:
    projected = dict(key_metrics)
    verdict_payload = _coerce_mapping(factor_verdict)
    triage_payload = _coerce_mapping(campaign_triage)
    promotion_payload = _coerce_mapping(level2_promotion)
    decision_payload = _coerce_mapping(promotion_decision)
    uncertainty_payload = _coerce_mapping(uncertainty)
    rolling_payload = _coerce_mapping(rolling_stability)
    neutralization_payload = _coerce_mapping(neutralization_comparison)
    portfolio_summary_payload = _coerce_mapping(portfolio_validation_summary)
    portfolio_metrics_payload = _coerce_mapping(portfolio_validation_metrics)
    robustness_payload = _coerce_mapping(
        portfolio_summary_payload.get("portfolio_robustness_summary")
    )
    benchmark_payload = _coerce_mapping(
        portfolio_metrics_payload.get("benchmark_relative_evaluation")
    )

    _set_projection_metric(projected, "factor_verdict", verdict_payload.get("label"))
    _set_projection_metric(projected, "factor_verdict_reasons", verdict_payload.get("reasons"))
    _set_projection_metric(projected, "campaign_triage", triage_payload.get("campaign_triage"))
    _set_projection_metric(
        projected,
        "campaign_triage_reasons",
        triage_payload.get("campaign_triage_reasons"),
    )
    _set_projection_metric(
        projected,
        "promotion_decision",
        decision_payload.get("verdict"),
    )
    _set_projection_metric(
        projected,
        "promotion_reasons",
        decision_payload.get("reasons"),
    )
    _set_projection_metric(
        projected,
        "promotion_blockers",
        decision_payload.get("blockers"),
    )
    _set_projection_metric(
        projected,
        "promotion_decision",
        promotion_payload.get("promotion_decision"),
    )
    _set_projection_metric(
        projected,
        "promotion_reasons",
        promotion_payload.get("promotion_reasons"),
    )
    _set_projection_metric(
        projected,
        "promotion_blockers",
        promotion_payload.get("promotion_blockers"),
    )
    _set_projection_metric(
        projected,
        "portfolio_validation_status",
        portfolio_summary_payload.get("validation_status"),
    )
    _set_projection_metric(
        projected,
        "portfolio_validation_recommendation",
        portfolio_summary_payload.get("recommendation"),
    )
    _set_projection_metric(
        projected,
        "portfolio_validation_major_risks",
        portfolio_summary_payload.get("major_risks"),
    )
    _set_projection_metric(
        projected,
        "portfolio_validation_robustness_label",
        robustness_payload.get("taxonomy_label"),
    )
    _set_projection_metric(
        projected,
        "portfolio_validation_support_reasons",
        robustness_payload.get("support_reasons"),
    )
    _set_projection_metric(
        projected,
        "portfolio_validation_fragility_reasons",
        robustness_payload.get("fragility_reasons"),
    )
    _set_projection_metric(
        projected,
        "portfolio_validation_scenario_sensitivity_notes",
        robustness_payload.get("scenario_sensitivity_notes"),
    )
    _set_projection_metric(
        projected,
        "portfolio_validation_benchmark_support_note",
        robustness_payload.get("benchmark_relative_support_note"),
    )
    _set_projection_metric(
        projected,
        "portfolio_validation_cost_sensitivity_note",
        robustness_payload.get("cost_sensitivity_note"),
    )
    _set_projection_metric(
        projected,
        "portfolio_validation_concentration_turnover_note",
        robustness_payload.get("concentration_turnover_risk_note"),
    )

    _set_projection_metric(
        projected,
        "portfolio_validation_benchmark_relative_status",
        benchmark_payload.get("status"),
    )
    _set_projection_metric(
        projected,
        "portfolio_validation_benchmark_relative_assessment",
        benchmark_payload.get("assessment"),
    )
    _set_projection_metric(
        projected,
        "portfolio_validation_benchmark_excess_return",
        benchmark_payload.get("benchmark_excess_return"),
    )
    _set_projection_metric(
        projected,
        "portfolio_validation_benchmark_tracking_error",
        benchmark_payload.get("benchmark_tracking_error"),
    )
    _set_projection_metric(
        projected,
        "benchmark_active_return",
        benchmark_payload.get("benchmark_active_return"),
    )
    _set_projection_metric(
        projected,
        "benchmark_information_ratio",
        benchmark_payload.get("benchmark_information_ratio"),
    )

    for key in (
        "mean_ic_ci_lower",
        "mean_ic_ci_upper",
        "mean_rank_ic_ci_lower",
        "mean_rank_ic_ci_upper",
        "mean_long_short_return_ci_lower",
        "mean_long_short_return_ci_upper",
        "uncertainty_method",
        "uncertainty_confidence_level",
        "uncertainty_bootstrap_resamples",
        "uncertainty_bootstrap_block_length",
        "uncertainty_flags",
    ):
        _set_projection_metric(projected, key, uncertainty_payload.get(key))

    for key in (
        "rolling_window_size",
        "rolling_ic_positive_share",
        "rolling_rank_ic_positive_share",
        "rolling_long_short_positive_share",
        "rolling_ic_min_mean",
        "rolling_rank_ic_min_mean",
        "rolling_long_short_min_mean",
        "rolling_instability_flags",
    ):
        _set_projection_metric(projected, key, rolling_payload.get(key))

    if neutralization_payload:
        _set_projection_metric(
            projected,
            "neutralization_comparison",
            neutralization_payload,
        )
        _set_projection_metric(
            projected,
            "neutralization_comparison_flags",
            neutralization_payload.get("interpretation_flags"),
        )
        _set_projection_metric(
            projected,
            "neutralization_comparison_reasons",
            neutralization_payload.get("interpretation_reasons"),
        )
        delta_payload = _coerce_mapping(neutralization_payload.get("delta"))
        raw_payload = _coerce_mapping(neutralization_payload.get("raw"))
        _set_projection_metric(
            projected,
            "neutralization_raw_mean_ic",
            raw_payload.get("mean_ic"),
        )
        _set_projection_metric(
            projected,
            "neutralization_raw_mean_rank_ic",
            raw_payload.get("mean_rank_ic"),
        )
        _set_projection_metric(
            projected,
            "neutralization_raw_mean_long_short_return",
            raw_payload.get("mean_long_short_return"),
        )
        _set_projection_metric(
            projected,
            "neutralization_raw_ic_ir",
            raw_payload.get("ic_ir"),
        )
        _set_projection_metric(
            projected,
            "neutralization_mean_ic_delta",
            delta_payload.get("mean_ic_delta"),
        )
        _set_projection_metric(
            projected,
            "neutralization_mean_rank_ic_delta",
            delta_payload.get("mean_rank_ic_delta"),
        )
        _set_projection_metric(
            projected,
            "neutralization_mean_long_short_return_delta",
            delta_payload.get("mean_long_short_return_delta"),
        )
        _set_projection_metric(
            projected,
            "neutralization_ic_ir_delta",
            delta_payload.get("ic_ir_delta"),
        )
        _set_projection_metric(
            projected,
            "neutralization_valid_ratio_min_delta",
            delta_payload.get("valid_ratio_min_delta"),
        )
        _set_projection_metric(
            projected,
            "neutralization_eval_coverage_ratio_mean_delta",
            delta_payload.get("eval_coverage_ratio_mean_delta"),
        )
        _set_projection_metric(
            projected,
            "neutralization_uncertainty_overlap_zero_count_delta",
            delta_payload.get("uncertainty_overlap_zero_count_delta"),
        )
        _set_projection_metric(
            projected,
            "neutralization_rolling_positive_share_min_delta",
            delta_payload.get("rolling_positive_share_min_delta"),
        )
        _set_projection_metric(
            projected,
            "neutralization_rolling_worst_mean_min_delta",
            delta_payload.get("rolling_worst_mean_min_delta"),
        )

    return projected


def _set_projection_metric(
    metrics: dict[str, object],
    key: str,
    value: object,
) -> None:
    if value is None:
        return
    if key not in metrics or metrics.get(key) is None:
        metrics[key] = value


def _extract_uncertainty_summary(
    uncertainty_metrics: UncertaintyEvidenceMetrics,
) -> UncertaintySummaryPayload:
    out: UncertaintySummaryPayload = {}
    uncertainty_payload = cast(Mapping[str, object], uncertainty_metrics)
    for key in (
        "mean_ic_ci_lower",
        "mean_ic_ci_upper",
        "mean_rank_ic_ci_lower",
        "mean_rank_ic_ci_upper",
        "mean_long_short_return_ci_lower",
        "mean_long_short_return_ci_upper",
        "uncertainty_method",
        "uncertainty_confidence_level",
        "uncertainty_bootstrap_resamples",
        "uncertainty_bootstrap_block_length",
    ):
        value = uncertainty_payload.get(key)
        if value is not None:
            out[key] = value
    if uncertainty_metrics["uncertainty_flags"]:
        out["uncertainty_flags"] = list(uncertainty_metrics["uncertainty_flags"])
    return out


def _extract_rolling_stability_summary(
    rolling_metrics: RollingStabilityMetrics,
) -> RollingStabilitySummaryPayload:
    out: RollingStabilitySummaryPayload = {}
    rolling_payload = cast(Mapping[str, object], rolling_metrics)
    for key in (
        "rolling_window_size",
        "rolling_ic_positive_share",
        "rolling_rank_ic_positive_share",
        "rolling_long_short_positive_share",
        "rolling_ic_min_mean",
        "rolling_rank_ic_min_mean",
        "rolling_long_short_min_mean",
    ):
        value = rolling_payload.get(key)
        if value is not None:
            out[key] = value
    if rolling_metrics["rolling_instability_flags"]:
        out["rolling_instability_flags"] = list(rolling_metrics["rolling_instability_flags"])
    return out


def _extract_neutralization_comparison(
    neutralization_metrics: NeutralizationComparisonMetrics,
    *,
    core_signal_metrics: CoreSignalEvidenceMetrics,
) -> NeutralizationComparisonPayload:
    comparison_payload = neutralization_metrics["neutralization_comparison"]
    if (
        comparison_payload["raw"]
        or comparison_payload["neutralized"]
        or comparison_payload["delta"]
        or comparison_payload["interpretation_flags"]
        or comparison_payload["interpretation_reasons"]
    ):
        out: NeutralizationComparisonPayload = {
            "raw": comparison_payload["raw"],
            "neutralized": comparison_payload["neutralized"],
            "delta": comparison_payload["delta"],
            "interpretation_flags": list(comparison_payload["interpretation_flags"]),
            "interpretation_reasons": list(comparison_payload["interpretation_reasons"]),
        }
        return out

    has_top_level_signal = any(
        value is not None
        for value in (
            neutralization_metrics["neutralization_raw_mean_ic"],
            neutralization_metrics["neutralization_raw_mean_rank_ic"],
            neutralization_metrics["neutralization_raw_mean_long_short_return"],
            neutralization_metrics["neutralization_raw_ic_ir"],
            neutralization_metrics["neutralization_mean_ic_delta"],
            neutralization_metrics["neutralization_mean_rank_ic_delta"],
            neutralization_metrics["neutralization_mean_long_short_return_delta"],
            neutralization_metrics["neutralization_ic_ir_delta"],
            neutralization_metrics["neutralization_valid_ratio_min_delta"],
            neutralization_metrics["neutralization_eval_coverage_ratio_mean_delta"],
            neutralization_metrics["neutralization_uncertainty_overlap_zero_count_delta"],
            neutralization_metrics["neutralization_rolling_positive_share_min_delta"],
            neutralization_metrics["neutralization_rolling_worst_mean_min_delta"],
        )
    ) or bool(
        neutralization_metrics["neutralization_flags"]
        or neutralization_metrics["neutralization_reasons"]
    )
    if not has_top_level_signal:
        return {}

    return {
        "raw": {
            "mean_ic": neutralization_metrics["neutralization_raw_mean_ic"],
            "mean_rank_ic": neutralization_metrics["neutralization_raw_mean_rank_ic"],
            "mean_long_short_return": neutralization_metrics[
                "neutralization_raw_mean_long_short_return"
            ],
            "ic_ir": neutralization_metrics["neutralization_raw_ic_ir"],
        },
        "neutralized": {
            "mean_ic": core_signal_metrics["mean_ic"],
            "mean_rank_ic": core_signal_metrics["mean_rank_ic"],
            "mean_long_short_return": core_signal_metrics["mean_long_short_return"],
            "ic_ir": core_signal_metrics["ic_ir"],
        },
        "delta": {
            "mean_ic_delta": neutralization_metrics["neutralization_mean_ic_delta"],
            "mean_rank_ic_delta": neutralization_metrics[
                "neutralization_mean_rank_ic_delta"
            ],
            "mean_long_short_return_delta": neutralization_metrics[
                "neutralization_mean_long_short_return_delta"
            ],
            "ic_ir_delta": neutralization_metrics["neutralization_ic_ir_delta"],
            "valid_ratio_min_delta": neutralization_metrics[
                "neutralization_valid_ratio_min_delta"
            ],
            "eval_coverage_ratio_mean_delta": neutralization_metrics[
                "neutralization_eval_coverage_ratio_mean_delta"
            ],
            "uncertainty_overlap_zero_count_delta": neutralization_metrics[
                "neutralization_uncertainty_overlap_zero_count_delta"
            ],
            "rolling_positive_share_min_delta": neutralization_metrics[
                "neutralization_rolling_positive_share_min_delta"
            ],
            "rolling_worst_mean_min_delta": neutralization_metrics[
                "neutralization_rolling_worst_mean_min_delta"
            ],
        },
        "interpretation_flags": list(neutralization_metrics["neutralization_flags"]),
        "interpretation_reasons": list(neutralization_metrics["neutralization_reasons"]),
    }


def _format_value(value: object) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    num = to_finite_float(value)
    if num is not None:
        return f"{num:.6f}"
    text = safe_text(value)
    return text if text is not None else "N/A"


def _format_transition(raw: object, neutralized: object, delta: object) -> str:
    return (
        f"{_format_value(raw)} -> {_format_value(neutralized)} "
        f"(delta={_format_value(delta)})"
    )
