"""Experimental Level 3 packaging utilities around replay artifacts.

Core Level 1/2 package assembly lives in
`alpha_lab.reporting.research_validation_package`.
This module adds replay/implementability-specific summaries and verdicts.
"""

from __future__ import annotations

import datetime
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from alpha_lab.exceptions import AlphaLabConfigError, AlphaLabDataError
from alpha_lab.reporting.research_validation_package import (
    build_research_validation_package,
)
from alpha_lab.research_integrity.semantic_consistency import (
    summarize_semantic_report_payload,
)

ResearchVerdict = Literal[
    "reject",
    "needs_review",
    "candidate_for_registry",
    "candidate_for_external_backtest",
]
PackageReadiness = Literal["ready", "needs_attention", "blocked"]

RESEARCH_PACKAGE_SCHEMA_VERSION = "1.0.0"
RESEARCH_PACKAGE_TYPE = "alpha_lab_research_package"
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
class ResearchPackageEngineSummary:
    """One replay engine summary derived from adapter artifacts."""

    engine: str
    run_path: str
    backtest_summary_path: str | None
    adapter_metadata_path: str | None
    adapter_metadata_ref: dict[str, object] | None
    semantic_consistency_report_path: str | None
    semantic_consistency_summary: dict[str, object] | None
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
            "semantic_consistency_report_path": self.semantic_consistency_report_path,
            "semantic_consistency_summary": self.semantic_consistency_summary,
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
    unavailable_metrics: tuple[dict[str, object], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "report_path": self.report_path,
            "dominant_execution_blocker": self.dominant_execution_blocker,
            "skipped_order_summary": self.skipped_order_summary,
            "target_vs_realized_deviation": self.target_vs_realized_deviation,
            "execution_flags": list(self.execution_flags),
            "unavailable_metrics": list(self.unavailable_metrics),
        }


@dataclass(frozen=True)
class ResearchPackageVerdict:
    """Transparent package-level verdict with explicit rationale fields."""

    verdict: ResearchVerdict
    package_readiness: PackageReadiness
    research_verdict_basis: str
    replay_verdict_basis: str
    execution_verdict_basis: str
    blocking_issues: tuple[str, ...]
    next_recommended_action: str

    def to_dict(self) -> dict[str, object]:
        return {
            "verdict": self.verdict,
            "package_readiness": self.package_readiness,
            "research_verdict_basis": self.research_verdict_basis,
            "replay_verdict_basis": self.replay_verdict_basis,
            "execution_verdict_basis": self.execution_verdict_basis,
            "blocking_issues": list(self.blocking_issues),
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
    workflow_type: str
    experiment_name: str
    identity: dict[str, object]
    research_intent: dict[str, object]
    research_results: dict[str, object]
    trial_registry_metadata: dict[str, object]
    replay_results: tuple[ResearchPackageEngineSummary, ...]
    execution_impact: ResearchPackageExecutionImpactSummary | None
    artifact_index: tuple[ResearchPackageArtifactRef, ...]
    verdict: ResearchPackageVerdict
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
            "replay_results": [x.to_dict() for x in self.replay_results],
            "execution_impact": (
                None if self.execution_impact is None else self.execution_impact.to_dict()
            ),
            "artifact_index": [x.to_dict() for x in self.artifact_index],
            "verdict": self.verdict.to_dict(),
            "interpretation": self.interpretation,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class LoadedResearchCaseOutputs:
    """Discovered output paths and payload snippets from one completed case."""

    case_dir: Path
    workflow_summary_path: Path | None
    workflow_summary: dict[str, object] | None
    handoff_bundle_path: Path | None
    trial_log_path: Path | None
    alpha_registry_path: Path | None
    replay_runs: tuple[tuple[str | None, Path], ...]
    execution_impact_report_path: Path | None


def load_research_case_outputs(case_output_dir: str | Path) -> LoadedResearchCaseOutputs:
    """Discover existing case artifacts without rerunning research logic."""
    case_dir = Path(case_output_dir).resolve()
    if not case_dir.exists():
        raise FileNotFoundError(f"case output directory does not exist: {case_dir}")
    if not case_dir.is_dir():
        raise AlphaLabConfigError(f"case output path is not a directory: {case_dir}")

    workflow_path = _resolve_workflow_summary_path(case_dir)
    workflow_payload = _read_json_object(workflow_path) if workflow_path is not None else None
    outputs_payload = _coerce_mapping((workflow_payload or {}).get("outputs"))

    trial_log = _resolve_optional_path(outputs_payload.get("trial_log"), case_dir / "trial_log.csv")
    alpha_registry = _resolve_optional_path(
        outputs_payload.get("alpha_registry"), case_dir / "alpha_registry.csv"
    )

    resolved_handoff = _resolve_optional_path(
        outputs_payload.get("handoff_artifact"),
        None,
    )
    if resolved_handoff is None:
        handoff_dir = case_dir / "handoff"
        if handoff_dir.exists() and handoff_dir.is_dir():
            candidates = sorted([p for p in handoff_dir.iterdir() if p.is_dir()])
            resolved_handoff = candidates[0] if candidates else None

    replay_runs = _discover_replay_runs(case_dir)
    execution_impact_path = _discover_execution_impact_report(case_dir)

    return LoadedResearchCaseOutputs(
        case_dir=case_dir,
        workflow_summary_path=workflow_path,
        workflow_summary=workflow_payload,
        handoff_bundle_path=resolved_handoff,
        trial_log_path=trial_log,
        alpha_registry_path=alpha_registry,
        replay_runs=tuple(sorted(replay_runs)),
        execution_impact_report_path=execution_impact_path,
    )


def summarize_replay_outputs(
    replay_run_dir: str | Path,
    *,
    engine_hint: str | None = None,
) -> ResearchPackageEngineSummary:
    """Summarize one replay output directory for package inclusion."""
    run_dir = Path(replay_run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"replay output directory does not exist: {run_dir}")
    if not run_dir.is_dir():
        raise AlphaLabConfigError(f"replay output path is not a directory: {run_dir}")

    summary_path = run_dir / "backtest_summary.json"
    metadata_path = run_dir / "adapter_run_metadata.json"
    semantic_path = run_dir / "semantic_consistency_report.json"
    summary_payload = _read_json_object(summary_path) if summary_path.exists() else None
    metadata_payload = _read_json_object(metadata_path) if metadata_path.exists() else None
    semantic_payload = _read_json_object(semantic_path) if semantic_path.exists() else None
    if semantic_payload is None:
        semantic_payload = _coerce_mapping((metadata_payload or {}).get("semantic_consistency"))

    missing: list[str] = []
    if summary_payload is None:
        missing.append("backtest_summary.json")
    if metadata_payload is None:
        missing.append("adapter_run_metadata.json")
    if not semantic_payload:
        missing.append("semantic_consistency_report.json")

    summary_block = _coerce_mapping((summary_payload or {}).get("summary"))
    engine_name = _normalize_engine(
        engine_hint
        or (summary_payload or {}).get("engine")
        or (metadata_payload or {}).get("engine")
        or "unknown"
    )
    key_metrics = {
        "total_return": _safe_float(summary_block.get("total_return")),
        "sharpe_annualized": _safe_float(summary_block.get("sharpe_annualized")),
        "max_drawdown": _safe_float(summary_block.get("max_drawdown")),
        "mean_turnover": _safe_float(summary_block.get("mean_turnover")),
        "n_periods": _safe_int(summary_block.get("n_periods")),
    }
    warnings = _normalize_warning_rows((summary_payload or {}).get("warnings"))
    adapter_version = _safe_str((metadata_payload or {}).get("adapter_version"))
    semantic_summary = summarize_semantic_report_payload(semantic_payload or None)

    return ResearchPackageEngineSummary(
        engine=engine_name,
        run_path=str(run_dir),
        backtest_summary_path=str(summary_path) if summary_path.exists() else None,
        adapter_metadata_path=str(metadata_path) if metadata_path.exists() else None,
        adapter_metadata_ref=metadata_payload,
        semantic_consistency_report_path=str(semantic_path) if semantic_path.exists() else None,
        semantic_consistency_summary=semantic_summary,
        key_metrics=key_metrics,
        warnings=warnings,
        missing_artifacts=tuple(sorted(set(missing))),
        adapter_version=adapter_version,
    )


def summarize_execution_impact(
    report_path: str | Path,
) -> ResearchPackageExecutionImpactSummary:
    """Summarize one execution impact report JSON for package inclusion."""
    path = Path(report_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"execution impact report does not exist: {path}")
    payload = _read_json_object(path)

    turnover_summary = _coerce_mapping(payload.get("turnover_effect_summary"))
    reason_rows = _normalize_dict_rows(payload.get("reason_summary"))
    dominant = payload.get("dominant_execution_blocker")
    dominant_ratio = None
    dominant_count = None
    for row in reason_rows:
        if _safe_str(row.get("reason_code")) == _safe_str(dominant):
            dominant_ratio = _safe_float(row.get("skipped_order_ratio"))
            dominant_count = _safe_int(row.get("skipped_order_count"))
            break
    skipped_summary = {
        "dominant_reason_code": _safe_str(dominant),
        "dominant_reason_ratio": dominant_ratio,
        "dominant_reason_count": dominant_count,
    }
    deviation_summary = _coerce_mapping(payload.get("execution_deviation_summary"))
    target_vs_realized = {
        "mean_abs_weight_diff": _safe_float(deviation_summary.get("mean_abs_weight_diff")),
        "max_abs_weight_diff": _safe_float(deviation_summary.get("max_abs_weight_diff")),
        "gross_abs_diff_mean": _safe_float(deviation_summary.get("gross_abs_diff_mean")),
        "gross_abs_diff_max": _safe_float(deviation_summary.get("gross_abs_diff_max")),
    }
    flags = tuple(_normalize_dict_rows(payload.get("flags")))
    unavailable = tuple(_normalize_dict_rows(payload.get("unavailable_metrics")))

    return ResearchPackageExecutionImpactSummary(
        report_path=str(path),
        dominant_execution_blocker=_safe_str(dominant),
        skipped_order_summary={**skipped_summary, **turnover_summary},
        target_vs_realized_deviation=target_vs_realized,
        execution_flags=flags,
        unavailable_metrics=unavailable,
    )


def build_research_package(
    case_output_dir: str | Path,
    *,
    case_id: str,
    case_name: str,
    interpretation: str | None = None,
    notes: str | None = None,
) -> ResearchPackage:
    core_package = build_research_validation_package(
        case_output_dir,
        case_id=case_id,
        case_name=case_name,
        interpretation=interpretation,
        notes=notes,
    )
    loaded = load_research_case_outputs(case_output_dir)
    workflow = loaded.workflow_summary or {}

    replay_results = tuple(
        summarize_replay_outputs(path, engine_hint=engine) for engine, path in loaded.replay_runs
    )
    execution_summary = (
        summarize_execution_impact(loaded.execution_impact_report_path)
        if loaded.execution_impact_report_path is not None
        else None
    )
    verdict = _build_package_verdict(workflow, replay_results, execution_summary)
    semantic_overview = _semantic_consistency_overview(replay_results)

    core_index = tuple(
        ResearchPackageArtifactRef(
            name=ref.name,
            artifact_type=ref.artifact_type,
            path=ref.path,
            exists=ref.exists,
            required=ref.required,
        )
        for ref in core_package.artifact_index
    )
    replay_index = _build_replay_artifact_index(loaded)
    artifact_index = tuple([*core_index, *replay_index])
    identity = dict(core_package.identity)
    identity["handoff_bundle_path"] = (
        str(loaded.handoff_bundle_path) if loaded.handoff_bundle_path else None
    )
    research_intent = dict(core_package.research_intent)
    research_results = dict(core_package.research_results)
    research_results["semantic_consistency"] = semantic_overview
    trial_registry_metadata = dict(core_package.trial_registry_metadata)

    return ResearchPackage(
        schema_version=RESEARCH_PACKAGE_SCHEMA_VERSION,
        package_type=RESEARCH_PACKAGE_TYPE,
        created_at_utc=datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        case_id=case_id,
        case_name=case_name,
        case_output_dir=str(loaded.case_dir),
        workflow_type=core_package.workflow_type,
        experiment_name=core_package.experiment_name,
        identity=identity,
        research_intent=research_intent,
        research_results=research_results,
        trial_registry_metadata=trial_registry_metadata,
        replay_results=replay_results,
        execution_impact=execution_summary,
        artifact_index=artifact_index,
        verdict=verdict,
        interpretation=core_package.interpretation,
        notes=core_package.notes,
    )


def export_research_package(
    package: ResearchPackage,
    output_dir: str | Path,
) -> dict[str, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "research_package.json"
    json_path.write_text(json.dumps(package.to_dict(), indent=2, sort_keys=True), encoding="utf-8")

    md_path = out_dir / "research_package.md"
    md_path.write_text(_package_markdown(package), encoding="utf-8")
    return {"json": json_path, "markdown": md_path}


def _package_markdown(package: ResearchPackage) -> str:
    lines = [
        f"# Research Package: {package.case_name}",
        "",
        f"- Case ID: `{package.case_id}`",
        f"- Experiment: `{package.experiment_name}`",
        f"- Workflow: `{package.workflow_type}`",
        f"- Verdict: `{package.verdict.verdict}`",
        f"- Package Readiness: `{package.verdict.package_readiness}`",
        "",
        "## Replay Engines",
    ]
    for replay in package.replay_results:
        semantic_status = _safe_str(
            _coerce_mapping(replay.semantic_consistency_summary).get("status")
        ) or "unknown"
        lines.extend(
            [
                f"- `{replay.engine}` total_return={replay.key_metrics.get('total_return')} "
                f"sharpe={replay.key_metrics.get('sharpe_annualized')} "
                f"max_drawdown={replay.key_metrics.get('max_drawdown')} "
                f"mean_turnover={replay.key_metrics.get('mean_turnover')} "
                f"semantic_status={semantic_status}",
            ]
        )
    if package.execution_impact is not None:
        lines.extend(
            [
                "",
                "## Execution Impact",
                f"- Dominant blocker: `{package.execution_impact.dominant_execution_blocker}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Blocking Issues",
        ]
    )
    if package.verdict.blocking_issues:
        for issue in package.verdict.blocking_issues:
            lines.append(f"- {issue}")
    else:
        lines.append("- None")
    if package.interpretation:
        lines.extend(["", "## Interpretation", package.interpretation])
    if package.notes:
        lines.extend(["", "## Notes", package.notes])
    lines.append("")
    return "\n".join(lines)


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


def _discover_replay_runs(case_dir: Path) -> list[tuple[str | None, Path]]:
    replay_run_dirs: list[tuple[str | None, Path]] = []
    replay_compare = case_dir / "replay_compare"
    if replay_compare.exists():
        for engine_dir in sorted(p for p in replay_compare.iterdir() if p.is_dir()):
            replay_run_dirs.append((engine_dir.name, engine_dir.resolve()))
    replay = case_dir / "replay"
    if replay.exists():
        for run in sorted(p for p in replay.iterdir() if p.is_dir()):
            replay_run_dirs.append((None, run.resolve()))
    return replay_run_dirs


def _discover_execution_impact_report(case_dir: Path) -> Path | None:
    candidates = [
        case_dir / "execution_impact" / "execution_impact_report.json",
        case_dir / "execution_impact_report.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def _build_replay_artifact_index(
    loaded: LoadedResearchCaseOutputs,
) -> tuple[ResearchPackageArtifactRef, ...]:
    refs = [
        ResearchPackageArtifactRef(
            name="handoff_bundle",
            artifact_type="handoff_artifact",
            path=str(loaded.handoff_bundle_path) if loaded.handoff_bundle_path else "",
            exists=loaded.handoff_bundle_path is not None,
            required=True,
        ),
        ResearchPackageArtifactRef(
            name="execution_impact_report_json",
            artifact_type="execution_impact",
            path=str(loaded.execution_impact_report_path)
            if loaded.execution_impact_report_path
            else "",
            exists=loaded.execution_impact_report_path is not None,
            required=True,
        ),
    ]
    for engine, path in loaded.replay_runs:
        refs.append(
            ResearchPackageArtifactRef(
                name=f"replay_{engine or 'unknown'}",
                artifact_type="replay_outputs",
                path=str(path),
                exists=path.exists(),
                required=True,
            )
        )
        refs.append(
            ResearchPackageArtifactRef(
                name=f"semantic_consistency_{engine or 'unknown'}_json",
                artifact_type="semantic_consistency",
                path=str(path / "semantic_consistency_report.json"),
                exists=(path / "semantic_consistency_report.json").exists(),
                required=True,
            )
        )
        refs.append(
            ResearchPackageArtifactRef(
                name=f"semantic_consistency_{engine or 'unknown'}_markdown",
                artifact_type="semantic_consistency",
                path=str(path / "semantic_consistency_report.md"),
                exists=(path / "semantic_consistency_report.md").exists(),
                required=False,
            )
        )
    return tuple(refs)


def _build_package_verdict(
    workflow: Mapping[str, object],
    replay_results: Sequence[ResearchPackageEngineSummary],
    execution_summary: ResearchPackageExecutionImpactSummary | None,
) -> ResearchPackageVerdict:
    promotion = _coerce_mapping(workflow.get("promotion_decision"))
    research_verdict = _safe_str(promotion.get("verdict")) or "needs_review"
    if research_verdict not in _VALID_VERDICTS:
        research_verdict = "needs_review"

    issues: list[str] = []
    semantic_warn_present = False
    if not replay_results:
        issues.append("missing_replay_outputs")
    else:
        for replay in replay_results:
            if replay.missing_artifacts:
                issues.append(f"missing_replay_artifacts:{replay.engine}")
            semantic = _coerce_mapping(replay.semantic_consistency_summary)
            semantic_status = _safe_str(semantic.get("status"))
            if semantic_status is None:
                issues.append(f"missing_semantic_consistency:{replay.engine}")
            elif semantic_status == "fail":
                issues.append(f"semantic_incompatibility:{replay.engine}")
            elif semantic_status == "warn":
                semantic_warn_present = True
    if execution_summary is None:
        issues.append("missing_execution_impact_report")
    if research_verdict == "reject":
        issues.append("research_verdict_reject")

    if issues:
        readiness: PackageReadiness = "blocked"
    elif research_verdict == "needs_review" or semantic_warn_present:
        readiness = "needs_attention"
    else:
        readiness = "ready"

    next_action = (
        "Fix blocking issues and regenerate package artifacts."
        if readiness == "blocked"
        else (
            "Review interpretation and risks, then route package for peer review."
            if readiness == "needs_attention"
            else "Package is review-ready for external replay committee."
        )
    )

    execution_basis = (
        "execution impact report present"
        if execution_summary is not None
        else "execution impact report missing"
    )
    if replay_results:
        semantic_counts = {"pass": 0, "warn": 0, "fail": 0, "unknown": 0}
        for replay in replay_results:
            status = _safe_str(
                _coerce_mapping(replay.semantic_consistency_summary).get("status")
            )
            if status is None:
                semantic_counts["unknown"] += 1
            elif status in semantic_counts:
                semantic_counts[status] += 1
            else:
                semantic_counts["unknown"] += 1
        replay_basis = (
            "replay outputs present; semantic consistency statuses="
            f"pass:{semantic_counts['pass']} warn:{semantic_counts['warn']} "
            f"fail:{semantic_counts['fail']} unknown:{semantic_counts['unknown']}"
        )
    else:
        replay_basis = "replay outputs missing"

    return ResearchPackageVerdict(
        verdict=research_verdict,  # type: ignore[arg-type]
        package_readiness=readiness,
        research_verdict_basis=f"promotion_decision.verdict={research_verdict}",
        replay_verdict_basis=replay_basis,
        execution_verdict_basis=execution_basis,
        blocking_issues=tuple(issues),
        next_recommended_action=next_action,
    )


def _normalize_engine(engine: object) -> str:
    text = _safe_str(engine) or "unknown"
    return text.lower().strip()


def _semantic_consistency_overview(
    replay_results: Sequence[ResearchPackageEngineSummary],
) -> dict[str, object]:
    out: dict[str, object] = {"by_engine": {}, "summary": {}}
    status_counts: dict[str, int] = {"pass": 0, "warn": 0, "fail": 0, "unknown": 0}
    for replay in replay_results:
        summary = replay.semantic_consistency_summary
        status = _safe_str(_coerce_mapping(summary).get("status")) or "unknown"
        if status not in status_counts:
            status = "unknown"
        status_counts[status] += 1
        out["by_engine"][replay.engine] = summary
    out["summary"] = status_counts
    return out


def _normalize_warning_rows(raw: object) -> tuple[dict[str, object], ...]:
    rows = _normalize_dict_rows(raw)
    deduped: list[dict[str, object]] = []
    seen: set[tuple[str | None, str | None]] = set()
    for row in rows:
        key = (_safe_str(row.get("code")), _safe_str(row.get("message")))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return tuple(deduped)


def _normalize_dict_rows(raw: object) -> list[dict[str, object]]:
    if not isinstance(raw, list):
        return []
    out: list[dict[str, object]] = []
    for item in raw:
        if isinstance(item, dict):
            out.append(item)
    return out


def _ensure_sequence(raw: object) -> Sequence[object]:
    return raw if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)) else []


def _coerce_mapping(raw: object) -> dict[str, object]:
    return dict(raw) if isinstance(raw, Mapping) else {}


def _read_json_object(path: Path | None) -> dict[str, object]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AlphaLabDataError(f"{path} must contain a JSON object")
    return payload


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(val) else val


def _safe_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None
