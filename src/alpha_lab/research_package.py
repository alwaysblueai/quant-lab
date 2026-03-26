from __future__ import annotations

import argparse
import datetime
import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

RESEARCH_PACKAGE_SCHEMA_VERSION = "1.0.0"
RESEARCH_PACKAGE_TYPE = "alpha_lab_research_package"
RESEARCH_CAMPAIGN_SCHEMA_VERSION = "1.0.0"

ResearchVerdict = Literal[
    "reject",
    "needs_review",
    "candidate_for_registry",
    "candidate_for_external_backtest",
]
PackageReadiness = Literal["ready", "needs_attention", "blocked"]

_VALID_VERDICTS: frozenset[str] = frozenset(
    {
        "reject",
        "needs_review",
        "candidate_for_registry",
        "candidate_for_external_backtest",
    }
)


@dataclass(frozen=True)
class ResearchPackageArtifactRef:
    """One artifact reference included in package-level indexing."""

    name: str
    artifact_type: str
    path: str
    exists: bool
    required: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "artifact_type": self.artifact_type,
            "path": self.path,
            "exists": self.exists,
            "required": self.required,
        }


@dataclass(frozen=True)
class ResearchPackageEngineSummary:
    """One replay engine summary derived from adapter artifacts."""

    engine: str
    run_path: str
    backtest_summary_path: str | None
    adapter_metadata_path: str | None
    adapter_metadata_ref: str | None
    key_metrics: dict[str, object]
    warnings: tuple[dict[str, object], ...]
    missing_artifacts: tuple[str, ...]
    adapter_version: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "engine": self.engine,
            "run_path": self.run_path,
            "backtest_summary_path": self.backtest_summary_path,
            "adapter_metadata_path": self.adapter_metadata_path,
            "adapter_metadata_ref": self.adapter_metadata_ref,
            "key_metrics": self.key_metrics,
            "warnings": list(self.warnings),
            "missing_artifacts": list(self.missing_artifacts),
            "adapter_version": self.adapter_version,
        }


@dataclass(frozen=True)
class ResearchPackageExecutionImpactSummary:
    """Execution-impact synopsis extracted from report artifacts."""

    report_path: str
    dominant_execution_blocker: str | None
    skipped_order_summary: dict[str, object]
    target_vs_realized_deviation: dict[str, object]
    execution_flags: tuple[dict[str, object], ...]
    warnings: tuple[dict[str, object], ...]
    unavailable_metrics: tuple[dict[str, object], ...]
    missing_artifacts: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "report_path": self.report_path,
            "dominant_execution_blocker": self.dominant_execution_blocker,
            "skipped_order_summary": self.skipped_order_summary,
            "target_vs_realized_deviation": self.target_vs_realized_deviation,
            "execution_flags": list(self.execution_flags),
            "warnings": list(self.warnings),
            "unavailable_metrics": list(self.unavailable_metrics),
            "missing_artifacts": list(self.missing_artifacts),
        }


@dataclass(frozen=True)
class ResearchPackageVerdict:
    """Transparent package-level verdict with explicit rationale fields."""

    verdict: ResearchVerdict
    package_readiness: PackageReadiness
    research_verdict_basis: tuple[str, ...]
    replay_verdict_basis: tuple[str, ...]
    execution_verdict_basis: tuple[str, ...]
    blocking_issues: tuple[str, ...]
    warnings: tuple[str, ...]
    next_recommended_action: str

    def to_dict(self) -> dict[str, object]:
        return {
            "verdict": self.verdict,
            "package_readiness": self.package_readiness,
            "research_verdict_basis": list(self.research_verdict_basis),
            "replay_verdict_basis": list(self.replay_verdict_basis),
            "execution_verdict_basis": list(self.execution_verdict_basis),
            "blocking_issues": list(self.blocking_issues),
            "warnings": list(self.warnings),
            "next_recommended_action": self.next_recommended_action,
        }


@dataclass(frozen=True)
class ResearchPackage:
    """Canonical single-case research package for archival and review."""

    schema_version: str
    package_type: str
    created_at_utc: str
    case_id: str
    case_name: str
    case_output_dir: str
    workflow_type: str | None
    experiment_name: str | None
    identity: dict[str, object]
    research_intent: dict[str, object]
    research_results: dict[str, object]
    trial_registry_metadata: dict[str, object]
    replay_results: tuple[ResearchPackageEngineSummary, ...]
    execution_impact: ResearchPackageExecutionImpactSummary | None
    artifact_index: tuple[ResearchPackageArtifactRef, ...]
    missing_artifacts: tuple[str, ...]
    verdict: ResearchPackageVerdict
    interpretation: str | None = None
    notes: str | None = None

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
            "replay_results": [item.to_dict() for item in self.replay_results],
            "execution_impact": (
                self.execution_impact.to_dict() if self.execution_impact is not None else None
            ),
            "artifact_index": [item.to_dict() for item in self.artifact_index],
            "missing_artifacts": list(self.missing_artifacts),
            "verdict": self.verdict.to_dict(),
            "interpretation": self.interpretation,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class LoadedResearchCaseOutputs:
    """Discovered output paths and payload snippets from one completed case."""

    case_output_dir: Path
    workflow_summary_path: Path | None
    workflow_summary: dict[str, object] | None
    handoff_bundle_path: Path | None
    trial_log_path: Path | None
    alpha_registry_path: Path | None
    replay_runs: tuple[tuple[str | None, Path], ...]
    execution_impact_report_path: Path | None
    missing_artifacts: tuple[str, ...]


@dataclass(frozen=True)
class ResearchCampaignSummary:
    """Light campaign-level aggregation over multiple research packages."""

    schema_version: str
    campaign_id: str
    generated_at_utc: str
    case_ids: tuple[str, ...]
    verdict_distribution: dict[str, int]
    top_candidates: tuple[str, ...]
    common_execution_blockers: tuple[dict[str, object], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "campaign_id": self.campaign_id,
            "generated_at_utc": self.generated_at_utc,
            "case_ids": list(self.case_ids),
            "verdict_distribution": self.verdict_distribution,
            "top_candidates": list(self.top_candidates),
            "common_execution_blockers": list(self.common_execution_blockers),
        }


def load_research_case_outputs(
    case_output_dir: str | Path,
    *,
    workflow_summary_path: str | Path | None = None,
    handoff_bundle_path: str | Path | None = None,
    replay_run_dirs: Mapping[str, str | Path] | None = None,
    execution_impact_report_path: str | Path | None = None,
) -> LoadedResearchCaseOutputs:
    """Discover existing case artifacts without rerunning research logic."""

    case_dir = Path(case_output_dir).resolve()
    if not case_dir.exists() or not case_dir.is_dir():
        raise FileNotFoundError(f"case output directory does not exist: {case_dir}")

    missing: list[str] = []
    workflow_path = _resolve_workflow_summary_path(
        case_dir,
        workflow_summary_path=workflow_summary_path,
    )
    workflow_payload: dict[str, object] | None = None
    outputs_payload: dict[str, object] = {}
    if workflow_path is None:
        missing.append("workflow_summary_json")
    elif not workflow_path.exists():
        missing.append("workflow_summary_json")
    else:
        workflow_payload = _read_json_object(workflow_path, artifact_name="workflow_summary_json")
        outputs_payload = _coerce_mapping(workflow_payload.get("outputs"))

    trial_log_path = _resolve_optional_path(
        path_value=outputs_payload.get("trial_log"),
        base_dir=case_dir,
    )
    if trial_log_path is not None and not trial_log_path.exists():
        missing.append("trial_log_csv")

    alpha_registry_path = _resolve_optional_path(
        path_value=outputs_payload.get("alpha_registry"),
        base_dir=case_dir,
    )
    if alpha_registry_path is not None and not alpha_registry_path.exists():
        missing.append("alpha_registry_csv")

    resolved_handoff = _resolve_optional_path(path_value=handoff_bundle_path, base_dir=case_dir)
    if resolved_handoff is None:
        resolved_handoff = _resolve_optional_path(
            path_value=outputs_payload.get("handoff_artifact"),
            base_dir=case_dir,
        )
    if resolved_handoff is not None and not resolved_handoff.exists():
        missing.append("handoff_bundle")

    replay_runs, replay_missing = _discover_replay_runs(
        case_dir,
        replay_run_dirs=replay_run_dirs,
    )
    missing.extend(replay_missing)
    if not replay_runs:
        missing.append("replay_outputs")

    execution_impact_path = _resolve_optional_path(
        path_value=execution_impact_report_path,
        base_dir=case_dir,
    )
    if execution_impact_path is None:
        execution_impact_path = _discover_execution_impact_report(
            case_dir=case_dir,
            replay_runs=replay_runs,
        )
    if execution_impact_path is None:
        missing.append("execution_impact_report_json")
    elif not execution_impact_path.exists():
        missing.append("execution_impact_report_json")

    return LoadedResearchCaseOutputs(
        case_output_dir=case_dir,
        workflow_summary_path=workflow_path,
        workflow_summary=workflow_payload,
        handoff_bundle_path=resolved_handoff,
        trial_log_path=trial_log_path,
        alpha_registry_path=alpha_registry_path,
        replay_runs=replay_runs,
        execution_impact_report_path=execution_impact_path,
        missing_artifacts=tuple(sorted(set(missing))),
    )


def summarize_replay_outputs(
    run_path: str | Path,
    *,
    engine_hint: str | None = None,
) -> ResearchPackageEngineSummary:
    """Summarize one replay output directory for package inclusion."""

    path = Path(run_path).resolve()
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"replay output directory does not exist: {path}")

    summary_path = path / "backtest_summary.json"
    metadata_path = path / "adapter_run_metadata.json"

    missing: list[str] = []
    summary_payload: dict[str, object] | None = None
    if summary_path.exists():
        summary_payload = _read_json_object(summary_path, artifact_name="backtest_summary_json")
    else:
        missing.append("backtest_summary.json")

    metadata_payload: dict[str, object] | None = None
    if metadata_path.exists():
        metadata_payload = _read_json_object(
            metadata_path,
            artifact_name="adapter_run_metadata_json",
        )
    else:
        missing.append("adapter_run_metadata.json")

    warnings: list[dict[str, object]] = []
    adapter_version: str | None = None
    engine_name: str | None = _normalize_engine(engine_hint)

    key_metrics: dict[str, object] = {
        "total_return": None,
        "sharpe_annualized": None,
        "max_drawdown": None,
        "mean_turnover": None,
        "n_periods": None,
    }

    if summary_payload is not None:
        engine_name = engine_name or _normalize_engine(summary_payload.get("engine"))
        summary_block = _coerce_mapping(summary_payload.get("summary"))
        key_metrics["total_return"] = _safe_float(summary_block.get("total_return"))
        key_metrics["sharpe_annualized"] = _safe_float(summary_block.get("sharpe_annualized"))
        key_metrics["max_drawdown"] = _safe_float(summary_block.get("max_drawdown"))
        key_metrics["mean_turnover"] = _safe_float(summary_block.get("mean_turnover"))
        key_metrics["n_periods"] = _safe_int(summary_block.get("n_periods"))
        warnings.extend(_normalize_warning_rows(summary_payload.get("warnings")))

        inline_metadata = _coerce_mapping(summary_payload.get("adapter_run_metadata"))
        if inline_metadata and metadata_payload is None:
            metadata_payload = inline_metadata

    if metadata_payload is not None:
        engine_name = engine_name or _normalize_engine(metadata_payload.get("engine"))
        adapter_version = _safe_str(metadata_payload.get("adapter_version"))
        warnings.extend(_normalize_warning_rows(metadata_payload.get("warnings")))

    deduped_warnings = tuple(_dedupe_warning_rows(warnings))

    return ResearchPackageEngineSummary(
        engine=engine_name or "unknown",
        run_path=str(path),
        backtest_summary_path=str(summary_path) if summary_path.exists() else None,
        adapter_metadata_path=str(metadata_path) if metadata_path.exists() else None,
        adapter_metadata_ref=str(metadata_path) if metadata_path.exists() else None,
        key_metrics=key_metrics,
        warnings=deduped_warnings,
        missing_artifacts=tuple(sorted(set(missing))),
        adapter_version=adapter_version,
    )


def summarize_execution_impact(
    report_path: str | Path,
) -> ResearchPackageExecutionImpactSummary:
    """Summarize one execution impact report JSON for package inclusion."""

    path = Path(report_path).resolve()
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"execution impact report does not exist: {path}")
    payload = _read_json_object(path, artifact_name="execution_impact_report_json")

    dominant = _safe_str(payload.get("dominant_execution_blocker"))
    turnover_summary = _coerce_mapping(payload.get("turnover_effect_summary"))
    reason_rows = _normalize_dict_rows(payload.get("reason_summary"))

    dominant_ratio: float | None = None
    dominant_count: int | None = None
    if dominant is not None:
        for row in reason_rows:
            if _safe_str(row.get("reason_code")) == dominant:
                dominant_ratio = _safe_float(row.get("skipped_order_ratio"))
                dominant_count = _safe_int(row.get("skipped_order_count"))
                break

    skipped_summary: dict[str, object] = {
        "dominant_reason_code": dominant,
        "dominant_reason_ratio": dominant_ratio,
        "dominant_reason_count": dominant_count,
        "skipped_order_ratio": _safe_float(turnover_summary.get("skipped_order_ratio")),
        "n_orders": _safe_int(turnover_summary.get("n_orders")),
        "n_skipped_orders": _safe_int(turnover_summary.get("n_skipped_orders")),
    }

    deviation_summary = _coerce_mapping(payload.get("execution_deviation_summary"))
    target_vs_realized: dict[str, object] = {
        "mean_abs_weight_diff": _safe_float(deviation_summary.get("mean_abs_weight_diff")),
        "max_abs_weight_diff": _safe_float(deviation_summary.get("max_abs_weight_diff")),
        "gross_abs_diff_mean": _safe_float(deviation_summary.get("gross_abs_diff_mean")),
        "gross_abs_diff_max": _safe_float(deviation_summary.get("gross_abs_diff_max")),
    }

    missing_artifacts = tuple(
        sorted({str(item) for item in _ensure_sequence(payload.get("missing_artifacts"))})
    )
    flags = tuple(_normalize_dict_rows(payload.get("flags")))
    warnings = tuple(_normalize_warning_rows(payload.get("warnings")))
    unavailable_metrics = tuple(_normalize_dict_rows(payload.get("unavailable_metrics")))

    return ResearchPackageExecutionImpactSummary(
        report_path=str(path),
        dominant_execution_blocker=dominant,
        skipped_order_summary=skipped_summary,
        target_vs_realized_deviation=target_vs_realized,
        execution_flags=flags,
        warnings=warnings,
        unavailable_metrics=unavailable_metrics,
        missing_artifacts=missing_artifacts,
    )


def build_research_package(
    case_output_dir: str | Path,
    *,
    case_id: str | None = None,
    case_name: str | None = None,
    created_at_utc: str | None = None,
    workflow_summary_path: str | Path | None = None,
    handoff_bundle_path: str | Path | None = None,
    replay_run_dirs: Mapping[str, str | Path] | None = None,
    execution_impact_report_path: str | Path | None = None,
    interpretation: str | None = None,
    notes: str | None = None,
) -> ResearchPackage:
    """Build one canonical research package from existing output artifacts."""

    loaded = load_research_case_outputs(
        case_output_dir,
        workflow_summary_path=workflow_summary_path,
        handoff_bundle_path=handoff_bundle_path,
        replay_run_dirs=replay_run_dirs,
        execution_impact_report_path=execution_impact_report_path,
    )

    workflow_payload = loaded.workflow_summary or {}
    workflow_name = _safe_str(workflow_payload.get("workflow"))
    experiment_name = _safe_str(workflow_payload.get("experiment_name"))

    resolved_case_id = (case_id or experiment_name or loaded.case_output_dir.name).strip()
    resolved_case_name = (case_name or experiment_name or resolved_case_id).strip()
    if not resolved_case_id:
        raise ValueError("case_id resolves to empty string")
    if not resolved_case_name:
        raise ValueError("case_name resolves to empty string")

    artifact_refs: list[ResearchPackageArtifactRef] = []
    missing_artifacts: list[str] = list(loaded.missing_artifacts)

    if loaded.workflow_summary_path is not None:
        artifact_refs.append(
            ResearchPackageArtifactRef(
                name="workflow_summary",
                artifact_type="workflow",
                path=str(loaded.workflow_summary_path),
                exists=loaded.workflow_summary_path.exists(),
                required=True,
            )
        )

    trial_registry_metadata, trial_missing, trial_refs = _load_trial_registry_metadata(
        experiment_name=experiment_name,
        trial_log_path=loaded.trial_log_path,
        alpha_registry_path=loaded.alpha_registry_path,
    )
    missing_artifacts.extend(trial_missing)
    artifact_refs.extend(trial_refs)

    handoff_metadata, handoff_missing, handoff_refs = _load_handoff_metadata(
        loaded.handoff_bundle_path,
    )
    missing_artifacts.extend(handoff_missing)
    artifact_refs.extend(handoff_refs)

    replay_summaries: list[ResearchPackageEngineSummary] = []
    replay_basis_missing: list[str] = []
    for engine_hint, replay_run in loaded.replay_runs:
        summary = summarize_replay_outputs(replay_run, engine_hint=engine_hint)
        replay_summaries.append(summary)
        artifact_refs.append(
            ResearchPackageArtifactRef(
                name=f"replay_run_{summary.engine}",
                artifact_type="replay_run",
                path=summary.run_path,
                exists=True,
                required=False,
            )
        )
        if summary.backtest_summary_path is not None:
            artifact_refs.append(
                ResearchPackageArtifactRef(
                    name=f"backtest_summary_{summary.engine}",
                    artifact_type="replay_summary",
                    path=summary.backtest_summary_path,
                    exists=True,
                    required=True,
                )
            )
        if summary.adapter_metadata_path is not None:
            artifact_refs.append(
                ResearchPackageArtifactRef(
                    name=f"adapter_metadata_{summary.engine}",
                    artifact_type="replay_metadata",
                    path=summary.adapter_metadata_path,
                    exists=True,
                    required=False,
                )
            )
        if summary.missing_artifacts:
            replay_basis_missing.extend(
                [f"{summary.engine}:{name}" for name in summary.missing_artifacts]
            )

    if replay_basis_missing:
        missing_artifacts.extend(sorted(set(replay_basis_missing)))

    execution_impact: ResearchPackageExecutionImpactSummary | None = None
    if (
        loaded.execution_impact_report_path is not None
        and loaded.execution_impact_report_path.exists()
    ):
        execution_impact = summarize_execution_impact(loaded.execution_impact_report_path)
        artifact_refs.append(
            ResearchPackageArtifactRef(
                name="execution_impact_report",
                artifact_type="execution_impact",
                path=execution_impact.report_path,
                exists=True,
                required=False,
            )
        )
        for item in execution_impact.missing_artifacts:
            missing_artifacts.append(f"execution_impact:{item}")
    elif loaded.execution_impact_report_path is not None:
        missing_artifacts.append("execution_impact_report_json")

    promotion_decision = _coerce_mapping(workflow_payload.get("promotion_decision"))

    research_intent = _build_research_intent(
        workflow_payload=workflow_payload,
        handoff_metadata=handoff_metadata,
    )
    research_results = _build_research_results(
        workflow_payload=workflow_payload,
        promotion_decision=promotion_decision,
    )

    identity = _build_identity(
        case_id=resolved_case_id,
        case_name=resolved_case_name,
        workflow_payload=workflow_payload,
        trial_registry_metadata=trial_registry_metadata,
        handoff_metadata=handoff_metadata,
        replay_summaries=tuple(replay_summaries),
    )

    unique_missing = tuple(sorted(set(missing_artifacts)))
    verdict = _assemble_package_verdict(
        promotion_decision=promotion_decision,
        replay_summaries=tuple(replay_summaries),
        execution_impact=execution_impact,
        missing_artifacts=unique_missing,
    )

    return ResearchPackage(
        schema_version=RESEARCH_PACKAGE_SCHEMA_VERSION,
        package_type=RESEARCH_PACKAGE_TYPE,
        created_at_utc=created_at_utc or _utc_now(),
        case_id=resolved_case_id,
        case_name=resolved_case_name,
        case_output_dir=str(loaded.case_output_dir),
        workflow_type=workflow_name,
        experiment_name=experiment_name,
        identity=identity,
        research_intent=research_intent,
        research_results=research_results,
        trial_registry_metadata=trial_registry_metadata,
        replay_results=tuple(
            sorted(
                replay_summaries,
                key=lambda item: (item.engine, item.run_path),
            )
        ),
        execution_impact=execution_impact,
        artifact_index=_dedupe_artifact_refs(artifact_refs),
        missing_artifacts=unique_missing,
        verdict=verdict,
        interpretation=interpretation,
        notes=notes,
    )


def export_research_package(
    package: ResearchPackage,
    *,
    output_dir: str | Path,
    export_artifact_index: bool = True,
) -> dict[str, Path]:
    """Export package JSON + markdown and optional artifact index."""

    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    package_json_path = out_dir / "research_package.json"
    package_json_path.write_text(
        json.dumps(package.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    package_markdown_path = out_dir / "research_package.md"
    package_markdown_path.write_text(
        _render_research_package_markdown(package),
        encoding="utf-8",
    )

    files: dict[str, Path] = {
        "package_json": package_json_path,
        "package_markdown": package_markdown_path,
    }

    if export_artifact_index:
        index_path = out_dir / "artifact_index.json"
        artifact_payload = {
            "schema_version": RESEARCH_PACKAGE_SCHEMA_VERSION,
            "case_id": package.case_id,
            "artifacts": [item.to_dict() for item in package.artifact_index],
        }
        index_path.write_text(
            json.dumps(artifact_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        files["artifact_index_json"] = index_path

    return files


def build_campaign_summary(
    packages: Iterable[ResearchPackage],
    *,
    campaign_id: str,
    generated_at_utc: str | None = None,
) -> ResearchCampaignSummary:
    """Aggregate multiple packages into a lightweight campaign summary."""

    package_list = sorted(list(packages), key=lambda item: item.case_id)
    verdict_counter = Counter(pkg.verdict.verdict for pkg in package_list)
    blockers = Counter(
        pkg.execution_impact.dominant_execution_blocker
        for pkg in package_list
        if pkg.execution_impact is not None and pkg.execution_impact.dominant_execution_blocker
    )

    top_candidates = tuple(
        pkg.case_id
        for pkg in package_list
        if pkg.verdict.verdict in {"candidate_for_registry", "candidate_for_external_backtest"}
        and pkg.verdict.package_readiness != "blocked"
    )
    common_blockers = tuple(
        {
            "execution_blocker": blocker,
            "count": int(count),
        }
        for blocker, count in sorted(
            blockers.items(),
            key=lambda item: (-item[1], item[0]),
        )
    )

    return ResearchCampaignSummary(
        schema_version=RESEARCH_CAMPAIGN_SCHEMA_VERSION,
        campaign_id=campaign_id,
        generated_at_utc=generated_at_utc or _utc_now(),
        case_ids=tuple(pkg.case_id for pkg in package_list),
        verdict_distribution={k: int(v) for k, v in sorted(verdict_counter.items())},
        top_candidates=top_candidates,
        common_execution_blockers=common_blockers,
    )


def build_cli_parser() -> argparse.ArgumentParser:
    """Build parser for optional post-workflow package generation."""

    parser = argparse.ArgumentParser(
        prog="build_research_package",
        description=(
            "Build canonical research package artifacts from an existing case output directory."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--case-output-dir",
        required=True,
        help="Completed case output directory containing workflow artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Export directory. Defaults to case-output-dir.",
    )
    parser.add_argument(
        "--case-id",
        default=None,
        help="Optional override for package case_id.",
    )
    parser.add_argument(
        "--case-name",
        default=None,
        help="Optional override for package case_name.",
    )
    parser.add_argument(
        "--workflow-summary-path",
        default=None,
        help="Optional explicit workflow summary JSON path.",
    )
    parser.add_argument(
        "--handoff-bundle-path",
        default=None,
        help="Optional explicit handoff bundle path.",
    )
    parser.add_argument(
        "--execution-impact-report-path",
        default=None,
        help="Optional explicit execution impact report JSON path.",
    )
    parser.add_argument(
        "--vectorbt-run-path",
        default=None,
        help="Optional vectorbt replay output directory.",
    )
    parser.add_argument(
        "--backtrader-run-path",
        default=None,
        help="Optional backtrader replay output directory.",
    )
    parser.add_argument(
        "--no-artifact-index",
        action="store_true",
        help="Disable artifact_index.json export.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for post-run research package generation."""

    parser = build_cli_parser()
    args = parser.parse_args(argv)

    replay_paths: dict[str, str | Path] = {}
    if args.vectorbt_run_path:
        replay_paths["vectorbt"] = args.vectorbt_run_path
    if args.backtrader_run_path:
        replay_paths["backtrader"] = args.backtrader_run_path

    package = build_research_package(
        args.case_output_dir,
        case_id=args.case_id,
        case_name=args.case_name,
        workflow_summary_path=args.workflow_summary_path,
        handoff_bundle_path=args.handoff_bundle_path,
        replay_run_dirs=replay_paths or None,
        execution_impact_report_path=args.execution_impact_report_path,
    )
    target_output_dir = Path(args.output_dir).resolve() if args.output_dir else Path(
        args.case_output_dir
    ).resolve()
    files = export_research_package(
        package,
        output_dir=target_output_dir,
        export_artifact_index=not args.no_artifact_index,
    )

    print("")
    print(f"  Case ID           : {package.case_id}")
    print(f"  Verdict           : {package.verdict.verdict}")
    print(f"  Package Readiness : {package.verdict.package_readiness}")
    print(f"  JSON              : {files['package_json']}")
    print(f"  Markdown          : {files['package_markdown']}")
    if "artifact_index_json" in files:
        print(f"  Artifact Index    : {files['artifact_index_json']}")
    if package.missing_artifacts:
        print(f"  Missing           : {', '.join(package.missing_artifacts)}")

    return 0


def _resolve_workflow_summary_path(
    case_dir: Path,
    *,
    workflow_summary_path: str | Path | None,
) -> Path | None:
    if workflow_summary_path is not None:
        return _resolve_optional_path(path_value=workflow_summary_path, base_dir=case_dir)

    matches = sorted(case_dir.glob("*_workflow_summary.json"))
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0].resolve()
    match_names = ", ".join(path.name for path in matches)
    raise ValueError(
        "multiple workflow summary files found; pass workflow_summary_path explicitly: "
        f"{match_names}"
    )


def _resolve_optional_path(
    *,
    path_value: object | None,
    base_dir: Path,
) -> Path | None:
    if path_value is None:
        return None
    if isinstance(path_value, Path):
        candidate = path_value
    elif isinstance(path_value, str):
        if not path_value.strip():
            return None
        candidate = Path(path_value)
    else:
        return None
    if not candidate.is_absolute():
        candidate = base_dir / candidate
    return candidate.resolve()


def _discover_replay_runs(
    case_dir: Path,
    *,
    replay_run_dirs: Mapping[str, str | Path] | None,
) -> tuple[tuple[tuple[str | None, Path], ...], tuple[str, ...]]:
    if replay_run_dirs is not None:
        runs: list[tuple[str | None, Path]] = []
        missing: list[str] = []
        for engine, raw_path in sorted(replay_run_dirs.items()):
            resolved = _resolve_optional_path(path_value=raw_path, base_dir=case_dir)
            if resolved is None or not resolved.exists() or not resolved.is_dir():
                missing.append(f"replay_run_dir:{engine}")
                continue
            runs.append((engine, resolved))
        return tuple(runs), tuple(sorted(set(missing)))

    discovered_runs: list[tuple[str | None, Path]] = []
    seen_paths: set[Path] = set()

    # Restrict fallback discovery to canonical replay locations so residual
    # scratch folders under case_dir cannot silently contaminate package input.
    canonical_paths: tuple[tuple[str | None, Path], ...] = (
        ("vectorbt", case_dir / "replay_compare" / "vectorbt"),
        ("backtrader", case_dir / "replay_compare" / "backtrader"),
        ("vectorbt", case_dir / "replays" / "vectorbt_run"),
        ("backtrader", case_dir / "replays" / "backtrader_run"),
        ("vectorbt", case_dir / "backtest_runs" / "vectorbt"),
        ("backtrader", case_dir / "backtest_runs" / "backtrader"),
    )
    for engine, candidate in canonical_paths:
        resolved = candidate.resolve()
        if resolved in seen_paths:
            continue
        if resolved.is_dir() and (resolved / "backtest_summary.json").exists():
            discovered_runs.append((engine, resolved))
            seen_paths.add(resolved)

    root_summary = case_dir / "backtest_summary.json"
    case_dir_resolved = case_dir.resolve()
    if root_summary.exists() and case_dir_resolved not in seen_paths:
        discovered_runs.append((None, case_dir_resolved))

    return tuple(discovered_runs), ()


def _discover_execution_impact_report(
    *,
    case_dir: Path,
    replay_runs: tuple[tuple[str | None, Path], ...],
) -> Path | None:
    del replay_runs  # deterministic fallback now uses canonical case-level locations only

    candidates = (
        case_dir / "execution_impact" / "execution_impact_report.json",
        case_dir / "execution_impact_report.json",
    )
    for path in candidates:
        if path.exists():
            return path.resolve()
    return None


def _load_trial_registry_metadata(
    *,
    experiment_name: str | None,
    trial_log_path: Path | None,
    alpha_registry_path: Path | None,
) -> tuple[dict[str, object], tuple[str, ...], tuple[ResearchPackageArtifactRef, ...]]:
    missing: list[str] = []
    refs: list[ResearchPackageArtifactRef] = []
    payload: dict[str, object] = {
        "trial_log_path": str(trial_log_path) if trial_log_path is not None else None,
        "trial_log_row": None,
        "alpha_registry_path": (
            str(alpha_registry_path) if alpha_registry_path is not None else None
        ),
        "alpha_registry_row": None,
    }

    if trial_log_path is not None:
        refs.append(
            ResearchPackageArtifactRef(
                name="trial_log",
                artifact_type="trial_log",
                path=str(trial_log_path),
                exists=trial_log_path.exists(),
                required=False,
            )
        )
        if trial_log_path.exists():
            trial_df = pd.read_csv(trial_log_path)
            row = _select_case_row(
                trial_df,
                key_column="experiment_name",
                key_value=experiment_name,
            )
            if row is not None:
                payload["trial_log_row"] = row
        else:
            missing.append("trial_log_csv")

    if alpha_registry_path is not None:
        refs.append(
            ResearchPackageArtifactRef(
                name="alpha_registry",
                artifact_type="alpha_registry",
                path=str(alpha_registry_path),
                exists=alpha_registry_path.exists(),
                required=False,
            )
        )
        if alpha_registry_path.exists():
            registry_df = pd.read_csv(alpha_registry_path)
            row = _select_case_row(
                registry_df,
                key_column="alpha_id",
                key_value=experiment_name,
            )
            if row is not None:
                payload["alpha_registry_row"] = row
        else:
            missing.append("alpha_registry_csv")

    return payload, tuple(sorted(set(missing))), tuple(refs)


def _load_handoff_metadata(
    handoff_bundle_path: Path | None,
) -> tuple[dict[str, object], tuple[str, ...], tuple[ResearchPackageArtifactRef, ...]]:
    if handoff_bundle_path is None:
        return {}, (), ()

    refs: list[ResearchPackageArtifactRef] = [
        ResearchPackageArtifactRef(
            name="handoff_bundle",
            artifact_type="handoff_bundle",
            path=str(handoff_bundle_path),
            exists=handoff_bundle_path.exists(),
            required=False,
        )
    ]
    missing: list[str] = []
    if not handoff_bundle_path.exists() or not handoff_bundle_path.is_dir():
        return {}, ("handoff_bundle",), tuple(refs)

    def _optional_json(
        filename: str,
        *,
        required: bool = False,
    ) -> dict[str, object]:
        file_path = handoff_bundle_path / filename
        refs.append(
            ResearchPackageArtifactRef(
                name=f"handoff_{filename.replace('.json', '')}",
                artifact_type="handoff_component",
                path=str(file_path),
                exists=file_path.exists(),
                required=required,
            )
        )
        if not file_path.exists():
            if required:
                missing.append(f"handoff:{filename}")
            return {}
        return _read_json_object(file_path, artifact_name=f"handoff_{filename}")

    manifest = _optional_json("manifest.json", required=True)
    dataset_fingerprint = _optional_json("dataset_fingerprint.json")
    portfolio = _optional_json("portfolio_construction.json")
    execution = _optional_json("execution_assumptions.json")
    timing = _optional_json("timing.json")
    metadata = _optional_json("experiment_metadata.json")
    validation_context = _optional_json("validation_context.json")

    out: dict[str, object] = {
        "artifact_path": str(handoff_bundle_path),
        "manifest": manifest,
        "dataset_fingerprint_payload": dataset_fingerprint,
        "dataset_fingerprint": (
            _safe_str(manifest.get("dataset_fingerprint"))
            or _safe_str(dataset_fingerprint.get("fingerprint"))
        ),
        "portfolio_construction": portfolio,
        "execution_assumptions": execution,
        "timing": timing,
        "experiment_metadata": metadata,
        "validation_context": validation_context,
    }
    return out, tuple(sorted(set(missing))), tuple(refs)


def _build_research_intent(
    *,
    workflow_payload: Mapping[str, object],
    handoff_metadata: Mapping[str, object],
) -> dict[str, object]:
    experiment_name = _safe_str(workflow_payload.get("experiment_name"))
    decision_payload = _coerce_mapping(workflow_payload.get("promotion_decision"))
    portfolio = _coerce_mapping(handoff_metadata.get("portfolio_construction"))
    execution = _coerce_mapping(handoff_metadata.get("execution_assumptions"))
    validation_context = _coerce_mapping(handoff_metadata.get("validation_context"))
    experiment_metadata_wrapper = _coerce_mapping(handoff_metadata.get("experiment_metadata"))
    experiment_metadata = _coerce_mapping(experiment_metadata_wrapper.get("experiment_metadata"))

    factor_summary = {
        "experiment_name": experiment_name,
        "workflow": _safe_str(workflow_payload.get("workflow")),
        "factor_spec": _safe_str(experiment_metadata.get("factor_spec")),
        "hypothesis": _safe_str(experiment_metadata.get("hypothesis")),
        "research_question": _safe_str(experiment_metadata.get("research_question")),
        "promotion_target": _safe_str(decision_payload.get("verdict")),
    }
    portfolio_summary = {
        "construction_method": _safe_str(portfolio.get("construction_method")),
        "weight_method": _safe_str(portfolio.get("weight_method")),
        "long_short": _safe_bool(portfolio.get("long_short")),
        "top_k": _safe_int(portfolio.get("top_k")),
        "bottom_k": _safe_int(portfolio.get("bottom_k")),
        "gross_limit": _safe_float(portfolio.get("gross_limit")),
        "net_limit": _safe_float(portfolio.get("net_limit")),
        "cash_buffer": _safe_float(portfolio.get("cash_buffer")),
        "rebalance_frequency": _safe_int(portfolio.get("rebalance_frequency")),
    }
    execution_summary = {
        "fill_price_rule": _safe_str(execution.get("fill_price_rule")),
        "execution_delay_bars": _safe_int(execution.get("execution_delay_bars")),
        "commission_model": _safe_str(execution.get("commission_model")),
        "slippage_model": _safe_str(execution.get("slippage_model")),
        "lot_size_rule": _safe_str(execution.get("lot_size_rule")),
        "suspension_policy": _safe_str(execution.get("suspension_policy")),
        "price_limit_policy": _safe_str(execution.get("price_limit_policy")),
        "trade_when_not_tradable": _safe_bool(execution.get("trade_when_not_tradable")),
        "allow_same_day_reentry": _safe_bool(execution.get("allow_same_day_reentry")),
    }

    validation_summary = {
        "validation_context": validation_context,
        "validation_scheme": _safe_str(
            _coerce_mapping(experiment_metadata.get("validation")).get("scheme")
        ),
    }

    return {
        "factor_or_composite_definition": factor_summary,
        "portfolio_construction_summary": portfolio_summary,
        "execution_assumptions_summary": execution_summary,
        "validation_mode_summary": validation_summary,
    }


def _build_research_results(
    *,
    workflow_payload: Mapping[str, object],
    promotion_decision: Mapping[str, object],
) -> dict[str, object]:
    key_metrics = _coerce_mapping(workflow_payload.get("key_metrics"))
    decision_metrics = _coerce_mapping(promotion_decision.get("metrics"))
    blocking = tuple(
        sorted(
            {
                str(item)
                for item in _ensure_sequence(promotion_decision.get("blocking_issues"))
            }
        )
    )
    warnings = tuple(
        sorted(
            {
                str(item)
                for item in _ensure_sequence(promotion_decision.get("warnings"))
            }
        )
    )

    return {
        "key_diagnostics": key_metrics,
        "factor_report_summary": {
            "mean_ic": _safe_float(key_metrics.get("mean_ic")),
            "mean_rank_ic": _safe_float(key_metrics.get("mean_rank_ic")),
            "ic_ir": _safe_float(key_metrics.get("ic_ir")),
        },
        "screening_result_summary": {
            "selected_factor_count": _safe_int(key_metrics.get("selected_factor_count")),
            "decision_metrics": decision_metrics,
        },
        "promotion_decision": {
            "verdict": _safe_str(promotion_decision.get("verdict")),
            "reasons": [str(item) for item in _ensure_sequence(promotion_decision.get("reasons"))],
            "blocking_issues": list(blocking),
            "warnings": list(warnings),
            "metrics": decision_metrics,
        },
        "warnings": list(warnings),
        "blocking_issues": list(blocking),
    }


def _build_identity(
    *,
    case_id: str,
    case_name: str,
    workflow_payload: Mapping[str, object],
    trial_registry_metadata: Mapping[str, object],
    handoff_metadata: Mapping[str, object],
    replay_summaries: tuple[ResearchPackageEngineSummary, ...],
) -> dict[str, object]:
    trial_row = _coerce_mapping(trial_registry_metadata.get("trial_log_row"))
    metadata_wrapper = _coerce_mapping(handoff_metadata.get("experiment_metadata"))
    experiment_metadata = _coerce_mapping(metadata_wrapper.get("experiment_metadata"))
    manifest = _coerce_mapping(handoff_metadata.get("manifest"))
    dataset_payload = _coerce_mapping(handoff_metadata.get("dataset_fingerprint_payload"))

    schema_versions: dict[str, object] = {
        "research_package_schema": RESEARCH_PACKAGE_SCHEMA_VERSION,
        "handoff_schema": _safe_str(manifest.get("schema_version")),
        "portfolio_construction_schema": _safe_str(
            _coerce_mapping(handoff_metadata.get("portfolio_construction")).get("schema_version")
        ),
        "execution_assumptions_schema": _safe_str(
            _coerce_mapping(handoff_metadata.get("execution_assumptions")).get("schema_version")
        ),
    }
    adapter_versions = {
        item.engine: item.adapter_version for item in replay_summaries if item.adapter_version
    }

    return {
        "case_id": case_id,
        "case_name": case_name,
        "workflow_type": _safe_str(workflow_payload.get("workflow")),
        "experiment_name": _safe_str(workflow_payload.get("experiment_name")),
        "trial_id": _safe_str(
            trial_row.get("trial_id") or experiment_metadata.get("trial_id")
        ),
        "trial_count": _safe_int(
            trial_row.get("trial_count") or experiment_metadata.get("trial_count")
        ),
        "dataset_id": _safe_str(
            trial_row.get("dataset_id") or experiment_metadata.get("dataset_id")
        ),
        "dataset_hash": _safe_str(
            trial_row.get("dataset_hash") or experiment_metadata.get("dataset_hash")
        ),
        "dataset_fingerprint": _safe_str(
            handoff_metadata.get("dataset_fingerprint") or dataset_payload.get("fingerprint")
        ),
        "schema_versions": schema_versions,
        "code_versions": {
            "adapter_versions": adapter_versions,
        },
    }


def _assemble_package_verdict(
    *,
    promotion_decision: Mapping[str, object],
    replay_summaries: tuple[ResearchPackageEngineSummary, ...],
    execution_impact: ResearchPackageExecutionImpactSummary | None,
    missing_artifacts: tuple[str, ...],
) -> ResearchPackageVerdict:
    research_basis: list[str] = []
    replay_basis: list[str] = []
    execution_basis: list[str] = []
    blocking_issues: list[str] = []
    warnings: list[str] = []

    base_verdict = _normalize_verdict(promotion_decision.get("verdict"))
    if base_verdict is None:
        base_verdict = "needs_review"
        blocking_issues.append("promotion_decision_missing_or_invalid")
        research_basis.append("promotion_decision_missing_or_invalid")
    else:
        research_basis.append(f"promotion_decision={base_verdict}")

    reasons = [str(item) for item in _ensure_sequence(promotion_decision.get("reasons"))]
    research_basis.extend([f"research_reason:{item}" for item in sorted(set(reasons))])

    for issue in _ensure_sequence(promotion_decision.get("blocking_issues")):
        text = str(issue)
        blocking_issues.append(text)
        research_basis.append(f"research_blocker:{text}")

    for item in _ensure_sequence(promotion_decision.get("warnings")):
        text = str(item)
        warnings.append(text)
        research_basis.append(f"research_warning:{text}")

    if not replay_summaries:
        replay_basis.append("no_replay_outputs_found")
        warnings.append("replay_outputs_missing")
    for replay in replay_summaries:
        replay_basis.append(f"replay_engine:{replay.engine}")
        if replay.missing_artifacts:
            warnings.append(f"{replay.engine}:missing_artifacts")
        for warning in replay.warnings:
            code = _safe_str(warning.get("code"))
            if code:
                replay_basis.append(f"replay_warning:{replay.engine}:{code}")
                if code.startswith("unsupported_"):
                    warnings.append(f"{replay.engine}:{code}")
        if "backtest_summary.json" in replay.missing_artifacts:
            blocking_issues.append(f"{replay.engine}:backtest_summary_missing")

    if execution_impact is None:
        execution_basis.append("execution_impact_report_missing")
        warnings.append("execution_impact_report_missing")
    else:
        if execution_impact.dominant_execution_blocker:
            execution_basis.append(
                f"dominant_execution_blocker:{execution_impact.dominant_execution_blocker}"
            )
        for flag in execution_impact.execution_flags:
            flag_name = _safe_str(flag.get("name"))
            triggered = _safe_bool(flag.get("triggered"))
            if flag_name is None or triggered is None or not triggered:
                continue
            execution_basis.append(f"execution_flag:{flag_name}")
            if flag_name in {"high_execution_deviation", "severe_tradability_constraints"}:
                blocking_issues.append(flag_name)

    for artifact in missing_artifacts:
        if artifact == "workflow_summary_json":
            blocking_issues.append("workflow_summary_json_missing")
        if artifact in {"trial_log_csv", "alpha_registry_csv", "execution_impact_report_json"}:
            warnings.append(f"missing:{artifact}")

    resolved_verdict: ResearchVerdict = base_verdict
    if resolved_verdict != "reject" and blocking_issues:
        resolved_verdict = "needs_review"

    readiness: PackageReadiness
    if blocking_issues:
        readiness = "blocked"
    elif warnings or missing_artifacts:
        readiness = "needs_attention"
    else:
        readiness = "ready"

    next_action = _next_recommended_action(
        verdict=resolved_verdict,
        has_replay=bool(replay_summaries),
        has_execution_impact=execution_impact is not None,
        blocking_issues=tuple(sorted(set(blocking_issues))),
    )

    return ResearchPackageVerdict(
        verdict=resolved_verdict,
        package_readiness=readiness,
        research_verdict_basis=tuple(sorted(set(research_basis))),
        replay_verdict_basis=tuple(sorted(set(replay_basis))),
        execution_verdict_basis=tuple(sorted(set(execution_basis))),
        blocking_issues=tuple(sorted(set(blocking_issues))),
        warnings=tuple(sorted(set(warnings))),
        next_recommended_action=next_action,
    )


def _next_recommended_action(
    *,
    verdict: ResearchVerdict,
    has_replay: bool,
    has_execution_impact: bool,
    blocking_issues: tuple[str, ...],
) -> str:
    if verdict == "reject":
        return "revise_hypothesis_or_signal_design"
    if verdict == "candidate_for_external_backtest":
        if not has_replay:
            return "run_external_backtest_replay"
        if not has_execution_impact:
            return "build_execution_impact_report"
        return "review_replay_limitations_and_prepare_registry_case"
    if verdict == "candidate_for_registry":
        if blocking_issues:
            return "resolve_blocking_issues_before_registry_promotion"
        return "prepare_registry_promotion_review"
    if any(
        issue in {"high_execution_deviation", "severe_tradability_constraints"}
        for issue in blocking_issues
    ):
        return "investigate_execution_constraints_and_replay"
    return "review_blockers_and_warnings"


def _render_research_package_markdown(package: ResearchPackage) -> str:
    lines: list[str] = [
        f"# Research Package - {package.case_name}",
        "",
        "## Verdict",
        "",
        f"- Verdict: `{package.verdict.verdict}`",
        f"- Package Readiness: `{package.verdict.package_readiness}`",
        f"- Next Recommended Action: `{package.verdict.next_recommended_action}`",
        "",
        "## Identity",
        "",
        f"- Case ID: `{package.case_id}`",
        f"- Workflow: `{package.workflow_type or 'unknown'}`",
        f"- Experiment: `{package.experiment_name or 'unknown'}`",
        f"- Created At (UTC): `{package.created_at_utc}`",
    ]

    dataset_fingerprint = _safe_str(package.identity.get("dataset_fingerprint"))
    if dataset_fingerprint:
        lines.append(f"- Dataset Fingerprint: `{dataset_fingerprint}`")

    lines.extend(
        [
            "",
            "## Research Intent",
            "",
            _compact_dict_line(
                "Factor / Composite",
                _coerce_mapping(package.research_intent.get("factor_or_composite_definition")),
            ),
            _compact_dict_line(
                "Portfolio Construction",
                _coerce_mapping(package.research_intent.get("portfolio_construction_summary")),
            ),
            _compact_dict_line(
                "Execution Assumptions",
                _coerce_mapping(package.research_intent.get("execution_assumptions_summary")),
            ),
            "",
            "## Research Results",
            "",
            _compact_dict_line(
                "Key Diagnostics",
                _coerce_mapping(package.research_results.get("key_diagnostics")),
            ),
            _compact_dict_line(
                "Promotion Decision",
                _coerce_mapping(package.research_results.get("promotion_decision")),
            ),
            "",
            "## Replay Results",
            "",
        ]
    )

    if package.replay_results:
        lines.append("| Engine | Total Return | Sharpe | Max Drawdown | Warnings |")
        lines.append("|---|---:|---:|---:|---:|")
        for replay in package.replay_results:
            metrics = replay.key_metrics
            lines.append(
                "| "
                f"{replay.engine} | "
                f"{_fmt_number(metrics.get('total_return'))} | "
                f"{_fmt_number(metrics.get('sharpe_annualized'))} | "
                f"{_fmt_number(metrics.get('max_drawdown'))} | "
                f"{len(replay.warnings)} |"
            )
    else:
        lines.append("- No replay outputs were discovered.")

    lines.extend(["", "## Execution Impact", ""])
    if package.execution_impact is None:
        lines.append("- Execution impact report not available.")
    else:
        impact = package.execution_impact
        skipped = impact.skipped_order_summary
        lines.append(
            f"- Dominant Blocker: `{impact.dominant_execution_blocker or 'none'}`"
        )
        lines.append(
            "- Skipped Order Ratio: "
            f"{_fmt_number(skipped.get('skipped_order_ratio'))}"
        )
        lines.append(
            "- Triggered Flags: "
            f"{_triggered_flag_names(impact.execution_flags)}"
        )
        lines.append(f"- Report Path: `{impact.report_path}`")

    if package.missing_artifacts:
        lines.extend(["", "## Missing Artifacts", ""])
        for missing_item in package.missing_artifacts:
            lines.append(f"- `{missing_item}`")

    lines.extend(["", "## Artifact Paths", ""])
    for artifact in package.artifact_index:
        marker = "required" if artifact.required else "optional"
        status = "exists" if artifact.exists else "missing"
        lines.append(
            f"- `{artifact.name}` ({artifact.artifact_type}, {marker}, {status}): "
            f"`{artifact.path}`"
        )

    if package.interpretation is not None:
        lines.extend(["", "## Interpretation", "", package.interpretation])
    if package.notes is not None:
        lines.extend(["", "## Notes", "", package.notes])

    return "\n".join(lines) + "\n"


def _compact_dict_line(title: str, payload: Mapping[str, object]) -> str:
    if not payload:
        return f"- {title}: none"
    parts: list[str] = []
    for key in sorted(payload):
        value = payload[key]
        rendered = _render_inline_value(value)
        if rendered is None:
            continue
        parts.append(f"{key}={rendered}")
    if not parts:
        return f"- {title}: none"
    return f"- {title}: " + "; ".join(parts)


def _render_inline_value(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        number = _safe_float(value)
        if number is None:
            if isinstance(value, int):
                return str(value)
            return None
        if float(number).is_integer():
            return str(int(number))
        return f"{number:.4f}"
    if isinstance(value, str):
        if not value.strip():
            return None
        return value
    if isinstance(value, Mapping):
        compact = _compact_dict_line("", _coerce_mapping(value))
        if compact.endswith(": none"):
            return None
        return compact.replace("- : ", "", 1)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        tokens = [str(item) for item in value if str(item).strip()]
        return ",".join(tokens) if tokens else None
    return str(value)


def _triggered_flag_names(flags: tuple[dict[str, object], ...]) -> str:
    names: list[str] = []
    for flag in flags:
        name = _safe_str(flag.get("name"))
        triggered = _safe_bool(flag.get("triggered"))
        if name and triggered:
            names.append(name)
    if not names:
        return "none"
    return ",".join(sorted(set(names)))


def _fmt_number(value: object) -> str:
    number = _safe_float(value)
    if number is None:
        return "n/a"
    if float(number).is_integer():
        return str(int(number))
    return f"{number:.4f}"


def _dedupe_artifact_refs(
    refs: Sequence[ResearchPackageArtifactRef],
) -> tuple[ResearchPackageArtifactRef, ...]:
    keyed: dict[tuple[str, str, str], ResearchPackageArtifactRef] = {}
    for item in refs:
        key = (item.artifact_type, item.name, item.path)
        keyed[key] = item
    return tuple(
        keyed[key]
        for key in sorted(
            keyed,
            key=lambda item: (item[0], item[1], item[2]),
        )
    )


def _select_case_row(
    frame: pd.DataFrame,
    *,
    key_column: str,
    key_value: str | None,
) -> dict[str, object] | None:
    if frame.empty:
        return None
    selected = frame
    if key_value is not None and key_column in frame.columns:
        mask = frame[key_column].astype(str) == str(key_value)
        if bool(mask.any()):
            selected = frame[mask]
    row = selected.tail(1).iloc[0]
    return _series_to_jsonable_dict(row)


def _series_to_jsonable_dict(series: pd.Series) -> dict[str, object]:
    out: dict[str, object] = {}
    for key, value in series.items():
        out[str(key)] = _jsonable_value(value)
    return out


def _read_json_object(path: Path, *, artifact_name: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{artifact_name} is not valid JSON ({path}:{exc.lineno}:{exc.colno})"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{artifact_name} must contain a JSON object: {path}")
    return payload


def _coerce_mapping(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        return {}
    return {str(k): _jsonable_value(v) for k, v in value.items()}


def _ensure_sequence(value: object) -> list[object]:
    if isinstance(value, (list, tuple)):
        return list(value)
    return []


def _normalize_dict_rows(value: object) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if not isinstance(value, list):
        return rows
    for item in value:
        if isinstance(item, Mapping):
            rows.append({str(k): _jsonable_value(v) for k, v in item.items()})
    return rows


def _normalize_warning_rows(value: object) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in _normalize_dict_rows(value):
        code = _safe_str(item.get("code")) or "unknown"
        message = _safe_str(item.get("message")) or ""
        rows.append({"code": code, "message": message})
    return rows


def _dedupe_warning_rows(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    seen: set[tuple[str, str]] = set()
    out: list[dict[str, object]] = []
    for row in rows:
        code = _safe_str(row.get("code")) or "unknown"
        message = _safe_str(row.get("message")) or ""
        key = (code, message)
        if key in seen:
            continue
        seen.add(key)
        out.append({"code": code, "message": message})
    return out


def _normalize_engine(value: object) -> str | None:
    raw = _safe_str(value)
    if raw is None:
        return None
    lowered = raw.strip().lower()
    if lowered in {"vectorbt", "backtrader"}:
        return lowered
    return raw


def _normalize_verdict(value: object) -> ResearchVerdict | None:
    text = _safe_str(value)
    if text is None:
        return None
    if text not in _VALID_VERDICTS:
        return None
    return text  # type: ignore[return-value]


def _safe_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text


def _safe_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        if float(value).is_integer():
            return int(value)
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return int(text)
        except ValueError:
            return None
    return None


def _safe_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return float(number)


def _safe_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "1", "yes"}:
            return True
        if text in {"false", "0", "no"}:
            return False
    return None


def _jsonable_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(k): _jsonable_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, datetime.datetime):
        return value.isoformat()
    if isinstance(value, datetime.date):
        return value.isoformat()
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return float(value)
    if pd.isna(value):
        return None
    if isinstance(value, (int, bool, str)) or value is None:
        return value
    return str(value)


def _utc_now() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")


__all__ = [
    "LoadedResearchCaseOutputs",
    "ResearchCampaignSummary",
    "ResearchPackage",
    "ResearchPackageArtifactRef",
    "ResearchPackageEngineSummary",
    "ResearchPackageExecutionImpactSummary",
    "ResearchPackageVerdict",
    "build_campaign_summary",
    "build_research_package",
    "export_research_package",
    "load_research_case_outputs",
    "main",
    "summarize_execution_impact",
    "summarize_replay_outputs",
]
