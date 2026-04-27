from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from alpha_lab.research_bridge.decomposition import (
    StructuralDecompositionReport,
    structural_decomposition_check,
)
from alpha_lab.research_bridge.graph_view import NoveltyReport, VaultGraph

KNOWN_NON_PIT_IDENTIFIERS = frozenset(
    {
        "current_constituents",
        "future_constituents",
        "non_pit_fundamentals",
        "restated_fundamentals",
        "survivorship_filtered_universe",
    }
)
KNOWN_NON_PIT_PATTERNS = ("non_pit", "non-pit", "restated", "survivorship", "future_")
PIT_SENSITIVE_IDENTIFIERS = frozenset(
    {
        "analyst_estimates",
        "benchmark_membership",
        "consensus_estimates",
        "corporate_actions",
        "financial_statements",
        "fundamental_data",
        "index_constituents",
        "shares_outstanding",
        "timestamped_releases",
    }
)
CROWDED_MECHANISM_THRESHOLD = 3


@dataclass(frozen=True, slots=True)
class PreflightIssue:
    severity: str
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class PreflightReport:
    is_blocked: bool
    issues: list[PreflightIssue]
    checked_cards: list[str]
    novelty: NoveltyReport | None
    decomposition: StructuralDecompositionReport | None


def run_preflight(
    *,
    vault_root: Path,
    checked_card_paths: list[str] | None = None,
    candidate_name: str = "",
    candidate_family: str = "",
    candidate_mechanism: str = "",
    candidate_similar: list[str] | None = None,
    candidate_uses_data: list[str] | None = None,
    candidate_pit_sensitivity: str = "",
    candidate_decay_class: str = "",
    candidate_capacity_class: str = "",
    enabled_checks: frozenset[str] | None = None,
) -> PreflightReport:
    def _enabled(check_code: str) -> bool:
        return enabled_checks is None or check_code in enabled_checks

    graph = VaultGraph.from_vault_root(vault_root)
    issues: list[PreflightIssue] = []
    checked_cards: list[str] = []

    try:
        graph.build(vault_root=vault_root)
    except Exception as exc:  # pragma: no cover - defensive path
        return PreflightReport(
            is_blocked=True,
            issues=[PreflightIssue("error", "graph_unavailable", str(exc))],
            checked_cards=[],
            novelty=None,
            decomposition=None,
        )

    if _enabled("dependency_completeness"):
        for raw_path in checked_card_paths or []:
            name = _card_name_from_path(Path(vault_root) / raw_path)
            if not name:
                continue
            checked_cards.append(name)
            missing = graph.check_dependency_completeness(name)
            if missing:
                issues.append(
                    PreflightIssue(
                        "warning",
                        "dependency_incomplete",
                        f"{name} is missing depends_on targets: {', '.join(missing)}",
                    )
                )

    novelty: NoveltyReport | None = None
    decomposition: StructuralDecompositionReport | None = None
    normalized_similar = list(candidate_similar or [])
    normalized_uses_data = [
        item.strip() for item in list(candidate_uses_data or []) if str(item).strip()
    ]
    pit_sensitivity = candidate_pit_sensitivity.strip().lower()
    decay_class = candidate_decay_class.strip().lower()
    capacity_class = candidate_capacity_class.strip().lower()
    if candidate_name:
        if _enabled("novelty"):
            novelty = graph.check_novelty(
                candidate_name=candidate_name,
                candidate_similar=normalized_similar,
                candidate_mechanism=candidate_mechanism,
                candidate_family=candidate_family,
            )
            if graph.get_node(candidate_name) is not None:
                issues.append(
                    PreflightIssue(
                        "error",
                        "candidate_exists",
                        f"candidate name already exists in graph: {candidate_name}",
                    )
                )
            elif novelty.warnings:
                issues.append(
                    PreflightIssue(
                        "warning",
                        "novelty_warning",
                        "; ".join(novelty.warnings),
                    )
                )

        if _enabled("pit") and pit_sensitivity == "high":
            blocking_sources, pit_sensitive_sources = _classify_pit_sources(normalized_uses_data)
            if blocking_sources:
                issues.append(
                    PreflightIssue(
                        "error",
                        "pit_non_pit_block",
                        "high PIT sensitivity candidate uses known non-PIT inputs: "
                        + ", ".join(blocking_sources),
                    )
                )
            elif not normalized_uses_data:
                issues.append(
                    PreflightIssue(
                        "warning",
                        "pit_inputs_missing",
                        "candidate is marked high PIT sensitivity but "
                        "candidate_uses_data was not provided",
                    )
                )
            elif pit_sensitive_sources:
                issues.append(
                    PreflightIssue(
                        "warning",
                        "pit_sensitive_data_warning",
                        "candidate uses PIT-sensitive inputs that still require as-of validation: "
                        + ", ".join(pit_sensitive_sources),
                    )
                )

        if _enabled("crowded_mechanism") and candidate_family and candidate_mechanism:
            validated_peers = [
                name
                for name in graph.get_mechanism_family_factors(
                    candidate_family,
                    candidate_mechanism,
                    validated_only=True,
                )
                if name != candidate_name
            ]
            if len(validated_peers) >= CROWDED_MECHANISM_THRESHOLD:
                issues.append(
                    PreflightIssue(
                        "warning",
                        "crowded_mechanism",
                        f"{candidate_family}/{candidate_mechanism} already has "
                        f"{len(validated_peers)} validated factors: {', '.join(validated_peers)}",
                    )
                )

            if _enabled("decomposition"):
                decomposition = structural_decomposition_check(
                    graph,
                    candidate_name=candidate_name,
                    candidate_family=candidate_family,
                    candidate_mechanism=candidate_mechanism,
                    candidate_similar=normalized_similar,
                )
                if decomposition.warnings:
                    issues.append(
                        PreflightIssue(
                            "warning",
                            "decomposition_warning",
                            "; ".join(decomposition.warnings),
                        )
                    )
        elif _enabled("crowded_mechanism"):
            issues.append(
                PreflightIssue(
                    "warning",
                    "candidate_metadata_missing",
                    "candidate_family and candidate_mechanism were not both provided; "
                    "redundancy checks are partial",
                )
            )

        if _enabled("capacity_decay") and decay_class == "fast" and capacity_class == "constrained":
            issues.append(
                PreflightIssue(
                    "warning",
                    "capacity_decay_warning",
                    "fast decay + constrained capacity implies very high implementation difficulty",
                )
            )

    is_blocked = any(issue.severity == "error" for issue in issues)
    return PreflightReport(
        is_blocked=is_blocked,
        issues=issues,
        checked_cards=checked_cards,
        novelty=novelty,
        decomposition=decomposition,
    )


def render_preflight_report(report: PreflightReport) -> str:
    lines = ["## Preflight Checks", ""]
    status = "blocked" if report.is_blocked else "pass_with_warnings" if report.issues else "pass"
    lines.append(f"- `status`: {status}")
    if report.checked_cards:
        lines.append(
            "- `checked_cards`: " + ", ".join(f"`{name}`" for name in report.checked_cards)
        )
    if report.issues:
        lines.append("- `issues`:")
        for issue in report.issues:
            lines.append(f"  - [{issue.severity}] {issue.code}: {issue.message}")
    else:
        lines.append("- `issues`: none")
    if report.decomposition is not None:
        lines.append(f"- `redundancy_risk`: {report.decomposition.redundancy_risk}")
    lines.append("")
    return "\n".join(lines)


def _card_name_from_path(path: Path) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8")
    if text.startswith("---\n"):
        end = text.find("\n---", 4)
        if end != -1:
            fm = text[4:end]
            for line in fm.splitlines():
                if line.startswith("name:"):
                    return line.split(":", 1)[1].strip().strip('"')
    stem = path.stem
    for prefix in ("Factor - ", "Method - ", "Concept - ", "Paper - "):
        if stem.startswith(prefix):
            return stem.removeprefix(prefix).strip()
    return stem.strip()


def _classify_pit_sources(uses_data: list[str]) -> tuple[list[str], list[str]]:
    blocking: set[str] = set()
    sensitive: set[str] = set()
    for item in uses_data:
        normalized = _normalize_identifier(item)
        if not normalized:
            continue
        if normalized in KNOWN_NON_PIT_IDENTIFIERS or any(
            pattern in normalized for pattern in KNOWN_NON_PIT_PATTERNS
        ):
            blocking.add(item)
            continue
        if normalized in PIT_SENSITIVE_IDENTIFIERS:
            sensitive.add(item)
    return sorted(blocking), sorted(sensitive)


def _normalize_identifier(value: str) -> str:
    return value.strip().lower().replace(" ", "_").replace("-", "_").replace("/", "_")
