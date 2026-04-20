from __future__ import annotations

import datetime as dt
import json
import re
from collections import Counter
from csv import DictReader
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from alpha_lab.exceptions import AlphaLabConfigError, AlphaLabDataError, AlphaLabExperimentError
from alpha_lab.research_bridge.categories import (
    CATEGORY_REGISTRY,
    CategoryProfile,
    get_category_profile,
)
from alpha_lab.research_bridge.divergence import build_divergence_context
from alpha_lab.research_bridge.embeddings import SearchResult, VaultEmbeddings
from alpha_lab.research_bridge.exploration import ExplorationMap, FrontierEntry
from alpha_lab.research_bridge.graph_view import VaultGraph
from alpha_lab.research_bridge.models import (
    CURRENT_SCHEMA_VERSION,
    AlphaLabDefaults,
    ProjectConfig,
    ProjectStatus,
    WritebackPolicy,
    load_project_config,
    save_project_config,
    save_yaml_document,
)
from alpha_lab.research_bridge.preflight import render_preflight_report, run_preflight
from alpha_lab.vault_export import ExportResult, export_to_vault, resolve_vault_root
from alpha_lab.vault_export_graph_feedback import (
    GraphFeedbackResult,
    apply_graph_feedback,
    collect_graph_feedback_summary,
)

PROJECTS_DIRNAME = "55_projects"

# Derive valid case types from category registry (backward compatible)
VALID_CASE_TYPES = tuple(
    ct for profile in CATEGORY_REGISTRY.values() for ct in profile.valid_case_types
)

ADVANCED_WORKFLOW_KEYS = frozenset(
    {
        "market_impact",
        "fill_model",
        "order_book",
        "execution_model",
        "slippage_model",
    }
)


@dataclass(frozen=True)
class ProjectPaths:
    vault_root: Path
    project_dir: Path
    project_yaml: Path
    current_case: Path
    decision_log: Path
    runs_dir: Path
    latest_run: Path
    project_brief: Path
    project_rules: Path
    card_map: Path
    active_state: Path
    legacy_decision_log: Path
    recent_history: Path
    rounds_dir: Path
    specs_dir: Path
    drafts_dir: Path


@dataclass(frozen=True)
class ProjectInitResult:
    project: ProjectConfig
    paths: ProjectPaths


@dataclass(frozen=True)
class RefreshProjectPackResult:
    project: ProjectConfig
    paths: ProjectPaths


@dataclass(frozen=True)
class ScaffoldCaseResult:
    project: ProjectConfig
    round_id: str | None
    case_name: str
    current_case_path: Path
    spec_path: Path
    handoff_path: Path
    preflight_path: Path | None = None


@dataclass(frozen=True)
class StartRoundResult:
    project: ProjectConfig
    round_id: str
    round_dir: Path
    round_context_digest: Path
    round_prompt: Path
    web_search_tasks: Path
    discussion_capture: Path


@dataclass(frozen=True)
class StructuredCandidate:
    candidate_name: str
    raw_idea: str
    suggested_mechanism: str
    suggested_factor_family: str
    novelty_score: float
    novelty_warnings: list[str]
    semantic_matches: list[SearchResult]
    related_cards: list[str]


@dataclass(frozen=True)
class StructureCandidatesResult:
    project: ProjectConfig
    round_id: str
    structured_candidates_path: Path
    knowledge_handoff_draft_path: Path
    candidates: list[StructuredCandidate]


@dataclass(frozen=True)
class ExploreIdeaCard:
    path: str
    name: str
    type: str
    lifecycle: str
    mechanism: str
    factor_family: str
    summary: str
    snippet: str
    reasons: list[str]

    def to_payload(self) -> dict[str, object]:
        return {
            "path": self.path,
            "name": self.name,
            "type": self.type,
            "lifecycle": self.lifecycle,
            "mechanism": self.mechanism,
            "factor_family": self.factor_family,
            "summary": self.summary,
            "snippet": self.snippet,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class ExploreIdeaResult:
    idea: str
    mode: str
    related_cards: list[ExploreIdeaCard]
    constraint_report: dict[str, object]
    gpt_prompt: str

    def to_payload(self) -> dict[str, object]:
        return {
            "idea": self.idea,
            "mode": self.mode,
            "related_cards": [card.to_payload() for card in self.related_cards],
            "constraint_report": dict(self.constraint_report),
            "gpt_prompt": self.gpt_prompt,
        }


@dataclass(frozen=True)
class SummarizeRunResult:
    project: ProjectConfig
    round_id: str | None
    summary_path: Path
    latest_path: Path
    decision_log_path: Path
    latest_experiment_feedback: Path
    writeback_draft: Path
    state_update_patch: Path
    graph_feedback: dict[str, object]


@dataclass(frozen=True)
class ApplyWritebackResult:
    project: ProjectConfig
    draft_path: Path
    export_result: ExportResult
    graph_feedback: GraphFeedbackResult


_GRAPH_SIGNAL_OPERATORS: tuple[str, ...] = (
    "lag / delta / pct_change / ratio / difference",
    "rolling_mean / rolling_std / rolling_min / rolling_max / rolling_rank / rolling_corr",
    "cross_section_rank / zscore_cross_section / winsorize_cross_section",
    "residualize / interaction / conditional_gate / min_coverage_gate",
)

_EXPLORE_MODE_ALIASES: dict[str, str] = {
    "start": "start",
    "discussion": "start",
    "kickoff": "start",
    "free": "free",
    "structured": "free",
    "constrained": "constrained",
}


def init_project(
    *,
    vault_root: str | Path | None,
    slug: str,
    title_zh: str,
    category: str,
    owner: str,
    market: str,
    frequency: str,
    chatgpt_project_name: str,
    max_research_level: int = 2,
    source_priority: list[str] | None = None,
    origin_cards: list[str] | None = None,
    supporting_cards: list[str] | None = None,
    failure_cards: list[str] | None = None,
    related_experiment_cards: list[str] | None = None,
    preferred_web_sources: list[str] | None = None,
    alpha_lab_defaults: AlphaLabDefaults | None = None,
    writeback_policy: WritebackPolicy | None = None,
    status: ProjectStatus | None = None,
    mode: str = "fast",
    overwrite: bool = False,
) -> ProjectInitResult:
    safe_slug = _safe_slug(slug)
    resolved_vault = _resolve_bridge_vault_root(vault_root)
    paths = _project_paths(resolved_vault, safe_slug)
    if paths.project_dir.exists() and not overwrite:
        raise FileExistsError(
            f"project already exists: {paths.project_dir}. Pass --overwrite to replace it."
        )

    paths.project_dir.mkdir(parents=True, exist_ok=True)
    paths.runs_dir.mkdir(parents=True, exist_ok=True)

    project = ProjectConfig(
        slug=safe_slug,
        title_zh=title_zh.strip(),
        category=category.strip(),
        owner=owner.strip(),
        market=market.strip(),
        frequency=frequency.strip(),
        chatgpt_project_name=chatgpt_project_name.strip(),
        schema_version=CURRENT_SCHEMA_VERSION,
        max_research_level=max_research_level,
        source_priority=list(
            source_priority or ["quant_knowledge", "alpha_lab_artifacts", "web_search"]
        ),
        origin_cards=list(origin_cards or []),
        supporting_cards=list(supporting_cards or []),
        failure_cards=list(failure_cards or []),
        related_experiment_cards=list(related_experiment_cards or []),
        preferred_web_sources=list(preferred_web_sources or []),
        alpha_lab_defaults=alpha_lab_defaults or AlphaLabDefaults(),
        writeback_policy=writeback_policy or WritebackPolicy(),
        status=status or ProjectStatus(),
    )
    save_project_config(project, paths.project_yaml)
    _refresh_project_pack(project, paths, mode=mode)
    return ProjectInitResult(project=project, paths=paths)


def refresh_project_pack(
    *,
    vault_root: str | Path | None,
    project_slug: str,
    mode: str = "fast",
) -> RefreshProjectPackResult:
    resolved_vault = _resolve_bridge_vault_root(vault_root)
    paths = _project_paths(resolved_vault, project_slug)
    project = load_project_config(paths.project_yaml)
    _refresh_project_pack(project, paths, mode=mode)
    return RefreshProjectPackResult(project=project, paths=paths)


def start_round(
    *,
    vault_root: str | Path | None,
    project_slug: str,
    topic: str,
    round_id: str | None = None,
    mode: str = "standard",
) -> StartRoundResult:
    resolved_vault = _resolve_bridge_vault_root(vault_root)
    paths = _project_paths(resolved_vault, project_slug)
    project = load_project_config(paths.project_yaml)
    resolved_round_id = _safe_slug(round_id) if round_id else _build_round_id(topic)
    round_dir = paths.rounds_dir / resolved_round_id
    round_dir.mkdir(parents=True, exist_ok=True)

    round_context_digest = round_dir / "round_context_digest.md"
    round_prompt = round_dir / "round_prompt.md"
    web_search_tasks = round_dir / "web_search_tasks.md"
    discussion_capture = round_dir / "discussion_capture.md"

    round_context_digest.write_text(
        _render_round_context_digest(
            project=project,
            paths=paths,
            topic=topic,
            mode=mode,
        ),
        encoding="utf-8",
    )
    round_prompt.write_text(
        _render_round_prompt(
            project,
            round_id=resolved_round_id,
            topic=topic,
            mode=mode,
        ),
        encoding="utf-8",
    )
    web_search_tasks.write_text(
        _render_web_search_tasks(project, topic=topic),
        encoding="utf-8",
    )
    if not discussion_capture.exists():
        discussion_capture.write_text(
            _render_discussion_capture_template(
                project=project,
                round_id=resolved_round_id,
                topic=topic,
            ),
            encoding="utf-8",
        )
    return StartRoundResult(
        project=project,
        round_id=resolved_round_id,
        round_dir=round_dir,
        round_context_digest=round_context_digest,
        round_prompt=round_prompt,
        web_search_tasks=web_search_tasks,
        discussion_capture=discussion_capture,
    )


def structure_candidates(
    *,
    vault_root: str | Path | None,
    project_slug: str,
    round_id: str,
    candidate_ideas: list[str] | None = None,
    top_k: int = 8,
    limit: int = 5,
) -> StructureCandidatesResult:
    resolved_vault = _resolve_bridge_vault_root(vault_root)
    paths = _project_paths(resolved_vault, project_slug)
    project = load_project_config(paths.project_yaml)
    round_dir = paths.rounds_dir / _safe_slug(round_id)
    discussion_capture = round_dir / "discussion_capture.md"
    ideas = [item.strip() for item in (candidate_ideas or []) if item and item.strip()]
    if not ideas and discussion_capture.exists():
        ideas = _extract_candidate_ideas(discussion_capture.read_text(encoding="utf-8"))
    if not ideas:
        ideas = [project.status.current_hypothesis or round_id]

    graph = _load_explore_graph(resolved_vault)
    embeddings = _load_explore_embeddings(resolved_vault)
    exploration = _load_explore_exploration_map(resolved_vault)
    index_rows = _load_card_index_rows(resolved_vault)
    structured: list[StructuredCandidate] = []
    for raw_idea in ideas[: max(limit, 1)]:
        semantic_matches = _search_explore_matches(
            idea=raw_idea,
            embeddings=embeddings,
            index_rows=index_rows,
            top_k=max(top_k, 4),
        )
        factor_matches = [match for match in semantic_matches if match.type == "factor"]
        suggested_family = ""
        suggested_mechanism = ""
        if graph is not None:
            suggested_family = _suggest_label(
                graph,
                factor_matches,
                label_getter=lambda name: _graph_node_factor_family(graph, name),
            )
            suggested_mechanism = _suggest_label(
                graph,
                factor_matches,
                label_getter=lambda name: _graph_node_mechanism(graph, name),
            )
        related_cards = _build_explore_related_cards(
            vault_root=resolved_vault,
            idea=raw_idea,
            graph=graph,
            embeddings=embeddings,
            project=project,
            semantic_matches=semantic_matches,
            index_rows=index_rows,
            top_k=max(limit, 3),
        )
        if not suggested_family:
            suggested_family = _first_non_empty(card.factor_family for card in related_cards)
        if not suggested_mechanism:
            suggested_mechanism = _first_non_empty(card.mechanism for card in related_cards)
        constraint_report = _build_explore_constraint_report(
            idea=raw_idea,
            graph=graph,
            exploration=exploration,
            factor_matches=factor_matches,
            related_cards=related_cards,
            suggested_family=suggested_family,
            suggested_mechanism=suggested_mechanism,
        )
        novelty_warnings = _object_to_str_list(constraint_report.get("novelty_warnings"))
        validated_peers = _object_to_str_list(constraint_report.get("validated_peers"))
        novelty_score = max(
            0.0, 1.0 - min(len(validated_peers), 5) / 5.0 - 0.1 * len(novelty_warnings)
        )
        structured.append(
            StructuredCandidate(
                candidate_name=_candidate_name_from_idea(raw_idea),
                raw_idea=raw_idea,
                suggested_mechanism=suggested_mechanism,
                suggested_factor_family=suggested_family,
                novelty_score=novelty_score,
                novelty_warnings=novelty_warnings,
                semantic_matches=semantic_matches[: max(limit, 3)],
                related_cards=[card.name for card in related_cards[: max(limit, 3)]],
            )
        )

    structured_candidates_path = round_dir / "structured_candidates.md"
    knowledge_handoff_draft_path = round_dir / "knowledge_handoff.md"
    structured_candidates_path.write_text(
        _render_structured_candidates(
            project=project,
            round_id=round_id,
            candidates=structured,
        ),
        encoding="utf-8",
    )
    knowledge_handoff_draft_path.write_text(
        _render_structured_handoff_draft(
            project=project,
            round_id=round_id,
            candidates=structured,
        ),
        encoding="utf-8",
    )
    return StructureCandidatesResult(
        project=project,
        round_id=round_id,
        structured_candidates_path=structured_candidates_path,
        knowledge_handoff_draft_path=knowledge_handoff_draft_path,
        candidates=structured,
    )


def scaffold_case(
    *,
    vault_root: str | Path | None,
    project_slug: str,
    round_id: str | None = None,
    case_name: str,
    case_type: str = "factor_recipe",
    factor_name: str | None = None,
    base_method: str = "momentum",
    lookback: int = 20,
    skip_recent: int = 5,
    target_horizon: int = 5,
    rebalance_frequency: str = "W",
    direction: str = "long",
    prices_path: str = "./placeholder_prices.csv",
    universe_path: str = "./placeholder_universe.csv",
    factor_path: str = "./placeholder_factor.csv",
    preflight: bool = False,
    candidate_name: str | None = None,
    candidate_family: str | None = None,
    candidate_mechanism: str | None = None,
    candidate_similar: list[str] | None = None,
    candidate_uses_data: list[str] | None = None,
    candidate_pit_sensitivity: str | None = None,
    candidate_decay_class: str | None = None,
    candidate_capacity_class: str | None = None,
    mode: str | None = None,
) -> ScaffoldCaseResult:
    if case_type not in VALID_CASE_TYPES:
        raise AlphaLabConfigError(
            f"unsupported case_type {case_type!r}; valid types: {', '.join(VALID_CASE_TYPES)}"
        )
    resolved_vault = _resolve_bridge_vault_root(vault_root)
    paths = _project_paths(resolved_vault, project_slug)
    project = load_project_config(paths.project_yaml)

    if project.max_research_level < 2:
        raise AlphaLabConfigError(
            f"project max_research_level is {project.max_research_level}; "
            f"scaffold-case requires at least Level 2."
        )

    safe_case_name = _safe_slug(case_name)
    factor_label = (
        factor_name.strip() if factor_name is not None and factor_name.strip() else safe_case_name
    )

    profile = get_category_profile(project.category)
    if profile.key == "factor_recipe":
        payload = _build_factor_recipe_payload(
            project=project,
            safe_case_name=safe_case_name,
            factor_label=factor_label,
            factor_path=factor_path,
            prices_path=prices_path,
            universe_path=universe_path,
            base_method=base_method,
            lookback=lookback,
            skip_recent=skip_recent,
            target_horizon=target_horizon,
            rebalance_frequency=rebalance_frequency,
            direction=direction,
        )
    else:
        payload = _build_generic_study_payload(
            project=project,
            profile=profile,
            safe_case_name=safe_case_name,
            case_label=factor_label,
        )
    _guard_research_level(payload, project.max_research_level)
    _write_yaml(paths.current_case, payload)
    project.status.current_case = safe_case_name
    save_project_config(project, paths.project_yaml)
    resolved_mode = _resolve_bridge_mode(mode, has_round=bool(round_id))
    spec_path = paths.current_case
    handoff_path = paths.current_case
    preflight_path: Path | None = None
    if resolved_mode == "legacy" or round_id:
        paths.specs_dir.mkdir(parents=True, exist_ok=True)
        spec_path = paths.specs_dir / f"{safe_case_name}.yaml"
        _write_yaml(spec_path, payload)
        discussion_capture = ""
        if round_id:
            capture_path = paths.rounds_dir / _safe_slug(round_id) / "discussion_capture.md"
            discussion_capture = _read_optional_text(capture_path)
        preflight_report = ""
        if preflight:
            report = run_preflight(
                vault_root=resolved_vault,
                checked_card_paths=project.origin_cards + project.supporting_cards,
                candidate_name=candidate_name or factor_label,
                candidate_family=candidate_family or "",
                candidate_mechanism=candidate_mechanism or "",
                candidate_similar=candidate_similar or [],
                candidate_uses_data=candidate_uses_data or [],
                candidate_pit_sensitivity=candidate_pit_sensitivity or "",
                candidate_decay_class=candidate_decay_class or "",
                candidate_capacity_class=candidate_capacity_class or "",
            )
            preflight_report = render_preflight_report(report)
            preflight_path = paths.specs_dir / f"{safe_case_name}__preflight.md"
            preflight_path.write_text(preflight_report, encoding="utf-8")
            if report.is_blocked:
                raise ValueError("preflight blocked scaffold-case")
        handoff_path = paths.specs_dir / f"{safe_case_name}__knowledge_handoff.md"
        handoff_path.write_text(
            _render_knowledge_handoff(
                project=project,
                round_id=round_id or "",
                case_name=safe_case_name,
                case_type=case_type,
                factor_name=factor_label,
                discussion_capture=discussion_capture,
                spec_path=spec_path,
                preflight_report=preflight_report,
            ),
            encoding="utf-8",
        )
    return ScaffoldCaseResult(
        project=project,
        round_id=round_id,
        case_name=safe_case_name,
        current_case_path=paths.current_case,
        spec_path=spec_path,
        handoff_path=handoff_path,
        preflight_path=preflight_path,
    )


def explore_idea(
    *,
    vault_root: str | Path | None,
    idea: str,
    mode: str = "free",
    project_slug: str | None = None,
    top_k: int = 8,
) -> ExploreIdeaResult:
    resolved_vault = _resolve_bridge_vault_root(vault_root)
    normalized_mode = _normalize_explore_mode(mode)
    normalized_idea = idea.strip()
    if not normalized_idea:
        raise ValueError("idea must be non-empty")

    project = _load_project_optional(resolved_vault, project_slug)
    graph = _load_explore_graph(resolved_vault)
    embeddings = _load_explore_embeddings(resolved_vault)
    exploration = _load_explore_exploration_map(resolved_vault)
    index_rows = _load_card_index_rows(resolved_vault)
    semantic_matches = _search_explore_matches(
        idea=normalized_idea,
        embeddings=embeddings,
        index_rows=index_rows,
        top_k=max(top_k * 2, 12),
    )

    factor_matches = [match for match in semantic_matches if match.type == "factor"]
    suggested_family = ""
    suggested_mechanism = ""
    if graph is not None:
        suggested_family = _suggest_label(
            graph,
            factor_matches,
            label_getter=lambda name: _graph_node_factor_family(graph, name),
        )
        suggested_mechanism = _suggest_label(
            graph,
            factor_matches,
            label_getter=lambda name: _graph_node_mechanism(graph, name),
        )

    related_cards = _build_explore_related_cards(
        vault_root=resolved_vault,
        idea=normalized_idea,
        graph=graph,
        embeddings=embeddings,
        project=project,
        semantic_matches=semantic_matches,
        index_rows=index_rows,
        top_k=max(top_k, 1),
    )
    if not suggested_family:
        suggested_family = _first_non_empty(card.factor_family for card in related_cards)
    if not suggested_mechanism:
        suggested_mechanism = _first_non_empty(card.mechanism for card in related_cards)

    prompt_context = _build_explore_constraint_report(
        idea=normalized_idea,
        graph=graph,
        exploration=exploration,
        factor_matches=factor_matches,
        related_cards=related_cards,
        suggested_family=suggested_family,
        suggested_mechanism=suggested_mechanism,
    )
    constraint_report: dict[str, object] = (
        prompt_context if normalized_mode == "constrained" else {}
    )

    category = project.category if project is not None else "factor_recipe"
    gpt_prompt = _build_exploration_prompt(
        idea=normalized_idea,
        mode=normalized_mode,
        cards=related_cards,
        constraint_report=prompt_context,
        category=category,
        project=project,
        graph=graph,
    )
    return ExploreIdeaResult(
        idea=normalized_idea,
        mode=normalized_mode,
        related_cards=related_cards,
        constraint_report=constraint_report,
        gpt_prompt=gpt_prompt,
    )


def _build_factor_recipe_payload(
    *,
    project: ProjectConfig,
    safe_case_name: str,
    factor_label: str,
    factor_path: str,
    prices_path: str,
    universe_path: str,
    base_method: str,
    lookback: int,
    skip_recent: int,
    target_horizon: int,
    rebalance_frequency: str,
    direction: str,
) -> dict[str, Any]:
    return {
        "name": safe_case_name,
        "factor_name": factor_label,
        "factor_path": factor_path,
        "prices_path": prices_path,
        "factor_input": {
            "mode": "recipe",
            "disable_pipeline_preprocess": True,
            "recipe": {
                "base": {
                    "method": base_method,
                    "lookback": lookback,
                    "skip_recent": skip_recent,
                },
                "preprocess": {
                    "winsorize": {
                        "enabled": True,
                        "lower": 0.01,
                        "upper": 0.99,
                        "min_group_size": 5,
                    },
                    "standardization": {
                        "method": "zscore",
                        "min_group_size": 5,
                    },
                    "min_coverage": 0.2,
                },
            },
        },
        "rebalance_frequency": rebalance_frequency,
        "n_quantiles": 5,
        "direction": direction,
        "universe": {
            "name": project.alpha_lab_defaults.universe,
            "path": universe_path,
            "in_universe_column": "in_universe",
        },
        "target": {
            "kind": "forward_return",
            "horizon": target_horizon,
        },
        "preprocess": {
            "winsorize": False,
            "winsorize_lower": 0.01,
            "winsorize_upper": 0.99,
            "standardization": "none",
            "min_group_size": 5,
        },
        "transaction_cost": {
            "one_way_rate": 0.001,
        },
        "output": {
            "root_dir": f"dist/bridge_runs/{project.slug}/{safe_case_name}",
        },
    }


def _build_generic_study_payload(
    *,
    project: ProjectConfig,
    profile: CategoryProfile,
    safe_case_name: str,
    case_label: str,
) -> dict[str, Any]:
    """Build a minimal YAML spec for non-factor research categories."""
    payload: dict[str, Any] = {
        "name": safe_case_name,
        "case_type": profile.valid_case_types[0] if profile.valid_case_types else profile.key,
        "category": profile.key,
        "label": case_label,
        "project_slug": project.slug,
        "market": project.market,
        "frequency": project.frequency,
        "data_source": project.alpha_lab_defaults.data_source,
        "output": {
            "root_dir": f"dist/bridge_runs/{project.slug}/{safe_case_name}",
        },
    }
    # Seed fields from the profile's form_fields defaults
    study_params: dict[str, str] = {}
    for ff in profile.form_fields:
        name = str(ff.get("name", ""))
        default = str(ff.get("default", ""))
        if name:
            study_params[name] = default
    if study_params:
        payload["study_params"] = study_params
    return payload


def _guard_research_level(payload: dict[str, Any], max_level: int) -> None:
    if max_level >= 3:
        return
    violations = ADVANCED_WORKFLOW_KEYS & set(payload.keys())
    if violations:
        raise AlphaLabConfigError(
            f"case payload contains advanced workflow keys {sorted(violations)} "
            f"but project max_research_level is {max_level}. "
            f"Remove these keys or raise max_research_level in project.yaml."
        )


def summarize_run(
    *,
    vault_root: str | Path | None,
    project_slug: str,
    round_id: str | None = None,
    run_root: str | Path,
    mode: str | None = None,
) -> SummarizeRunResult:
    resolved_vault = _resolve_bridge_vault_root(vault_root)
    paths = _project_paths(resolved_vault, project_slug)
    _dedupe_fast_decision_log(paths.decision_log)
    project = load_project_config(paths.project_yaml)

    resolved_run_root = Path(run_root).expanduser().resolve()
    manifest_path = resolved_run_root / "run_manifest.json"
    metrics_path = resolved_run_root / "metrics.json"
    summary_path = resolved_run_root / "summary.md"
    manifest_payload = _load_json_required(manifest_path)
    metrics_payload = _load_json_optional(metrics_path)
    case_name = str(manifest_payload.get("case_name") or resolved_run_root.name).strip()
    metrics_digest = _extract_metrics_digest(metrics_payload)
    verdict_status = _default_verdict_status(metrics_digest.get("factor_verdict"))
    feedback_summary = collect_graph_feedback_summary(
        vault_root=resolved_vault,
        run_root=resolved_run_root,
        project=None,
        include_embeddings=False,
    )
    graph_feedback: dict[str, object] = {
        "suggested_similar_to": feedback_summary.suggested_similar_to,
        "correlation_summary": feedback_summary.correlation_summary,
    }
    summary_output_path = paths.runs_dir / _safe_slug(case_name) / "summary.md"
    summary_output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_output_path.write_text(
        _render_fast_run_summary(
            project=project,
            case_name=case_name,
            run_root=resolved_run_root,
            verdict_status=verdict_status,
            metrics_digest=metrics_digest,
            upstream_summary=summary_path if summary_path.exists() else None,
        ),
        encoding="utf-8",
    )
    paths.latest_run.write_text(
        _render_latest_run_index(
            project=project,
            case_name=case_name,
            verdict_status=verdict_status,
        ),
        encoding="utf-8",
    )
    reason = _build_fast_reason(metrics_digest, verdict_status=verdict_status)
    _append_fast_decision_log(
        paths.decision_log,
        case_name=case_name,
        run_key=resolved_run_root.name or str(resolved_run_root),
        verdict_status=verdict_status,
        reason=reason,
        next_action=project.status.next_action,
    )
    project.status.current_case = _safe_slug(case_name)
    project.status.latest_run = str(summary_output_path.relative_to(paths.project_dir))
    project.status.last_verdict = verdict_status
    save_project_config(project, paths.project_yaml)
    resolved_mode = _resolve_bridge_mode(mode, has_round=bool(round_id))
    artifact_dir = (
        summary_output_path.parent
        if resolved_mode == "fast" and not round_id
        else paths.rounds_dir / _safe_slug(round_id or "latest")
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    paths.drafts_dir.mkdir(parents=True, exist_ok=True)
    round_label = round_id or "latest"
    latest_experiment_feedback = artifact_dir / "latest_experiment_feedback.md"
    latest_experiment_feedback.write_text(
        _render_latest_experiment_feedback(
            project=project,
            round_id=round_label,
            case_name=case_name,
            run_root=resolved_run_root,
            metrics_digest=metrics_digest,
            summary_path=summary_path if summary_path.exists() else None,
            experiment_card_path=(resolved_run_root / "experiment_card.md")
            if (resolved_run_root / "experiment_card.md").exists()
            else None,
        ),
        encoding="utf-8",
    )
    draft_stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    writeback_draft = (
        paths.drafts_dir / f"{draft_stamp}__{_safe_slug(case_name)}__writeback_draft.md"
    )
    writeback_draft.write_text(
        _render_writeback_draft(
            project=project,
            round_id=round_label,
            case_name=case_name,
            run_root=resolved_run_root,
            manifest_path=manifest_path,
            metrics_path=metrics_path if metrics_path.exists() else None,
            summary_path=summary_path if summary_path.exists() else None,
            experiment_card_path=(resolved_run_root / "experiment_card.md")
            if (resolved_run_root / "experiment_card.md").exists()
            else None,
            verdict_status=verdict_status,
            metrics_digest=metrics_digest,
            graph_feedback=graph_feedback,
        ),
        encoding="utf-8",
    )
    state_update_patch = artifact_dir / "state_update_patch.md"
    state_update_patch.write_text(
        _render_state_update_patch(
            project=project,
            round_id=round_label,
            case_name=case_name,
            verdict_status=verdict_status,
            metrics_digest=metrics_digest,
        ),
        encoding="utf-8",
    )

    return SummarizeRunResult(
        project=project,
        round_id=round_id,
        summary_path=summary_output_path,
        latest_path=paths.latest_run,
        decision_log_path=paths.decision_log,
        latest_experiment_feedback=latest_experiment_feedback,
        writeback_draft=writeback_draft,
        state_update_patch=state_update_patch,
        graph_feedback=graph_feedback,
    )


def normalize_fast_decision_log(
    *,
    vault_root: str | Path | None,
    project_slug: str,
) -> Path:
    resolved_vault = _resolve_bridge_vault_root(vault_root)
    paths = _project_paths(resolved_vault, project_slug)
    _dedupe_fast_decision_log(paths.decision_log)
    return paths.decision_log


def apply_writeback(
    *,
    vault_root: str | Path | None,
    project_slug: str,
    draft_path: str | Path,
    mode: str | None = None,
) -> ApplyWritebackResult:
    resolved_vault = _resolve_bridge_vault_root(vault_root)
    paths = _project_paths(resolved_vault, project_slug)
    project = load_project_config(paths.project_yaml)
    resolved_draft_path = Path(draft_path).expanduser().resolve()
    frontmatter, _body = _load_markdown_with_frontmatter(resolved_draft_path)
    review_status = str(frontmatter.get("review_status") or "").strip().lower()
    if review_status != "approved":
        raise ValueError(f"draft {resolved_draft_path} has not been approved")

    export_mode = (
        str(mode or frontmatter.get("vault_export_mode") or "versioned").strip() or "versioned"
    )
    export_result = export_to_vault(
        source_paths={
            "experiment_card_path": _frontmatter_path(frontmatter, "experiment_card_path"),
            "summary_path": _frontmatter_path(frontmatter, "summary_path"),
            "manifest_path": _frontmatter_path(frontmatter, "manifest_path"),
        },
        case_name=str(frontmatter.get("case_name") or "").strip(),
        vault_root=resolved_vault,
        mode=export_mode,
    )
    if not export_result.success:
        raise AlphaLabExperimentError(export_result.error or "vault export failed")

    if str(frontmatter.get("current_focus") or "").strip():
        project.status.current_focus = str(frontmatter.get("current_focus")).strip()
    if str(frontmatter.get("next_action") or "").strip():
        project.status.next_action = str(frontmatter.get("next_action")).strip()
    if str(frontmatter.get("current_hypothesis") or "").strip():
        project.status.current_hypothesis = str(frontmatter.get("current_hypothesis")).strip()
    if str(frontmatter.get("verdict_status") or "").strip():
        project.status.last_verdict = str(frontmatter.get("verdict_status")).strip()
    if str(frontmatter.get("case_name") or "").strip():
        project.status.current_case = _safe_slug(str(frontmatter.get("case_name")))
    save_project_config(project, paths.project_yaml)

    graph_feedback = apply_graph_feedback(
        vault_root=resolved_vault,
        project=project,
        draft_frontmatter=frontmatter,
        export_result=export_result,
    )
    _append_decision_log(
        paths.legacy_decision_log,
        case_name=str(frontmatter.get("case_name") or "").strip(),
        round_id=str(frontmatter.get("round_id") or "").strip(),
        verdict_status=str(frontmatter.get("verdict_status") or "").strip(),
        one_sentence_verdict=str(frontmatter.get("one_sentence_verdict") or "").strip(),
        run_root=str(frontmatter.get("run_root") or "").strip(),
        exported_targets=export_result.target_paths,
    )

    frontmatter["review_status"] = "applied"
    resolved_draft_path.write_text(
        _compose_markdown_with_frontmatter(frontmatter, _body),
        encoding="utf-8",
    )
    return ApplyWritebackResult(
        project=project,
        draft_path=resolved_draft_path,
        export_result=export_result,
        graph_feedback=graph_feedback,
    )


def _refresh_project_pack(
    project: ProjectConfig,
    paths: ProjectPaths,
    *,
    mode: str = "fast",
) -> None:
    _refresh_project_pack_mode(project, paths, mode=mode)


def _refresh_project_pack_mode(
    project: ProjectConfig,
    paths: ProjectPaths,
    *,
    mode: str = "fast",
) -> None:
    resolved_mode = _normalize_bridge_mode(mode)
    paths.project_dir.mkdir(parents=True, exist_ok=True)
    if resolved_mode == "legacy":
        paths.rounds_dir.mkdir(parents=True, exist_ok=True)
        paths.specs_dir.mkdir(parents=True, exist_ok=True)
        paths.drafts_dir.mkdir(parents=True, exist_ok=True)
        paths.project_brief.write_text(_render_project_brief(project), encoding="utf-8")
        paths.project_rules.write_text(_render_project_rules(project), encoding="utf-8")
        paths.card_map.write_text(
            _render_card_map(project=project, vault_root=paths.vault_root),
            encoding="utf-8",
        )
        paths.active_state.write_text(_render_active_state(project), encoding="utf-8")
        paths.recent_history.write_text(
            _render_recent_history(project=project, paths=paths),
            encoding="utf-8",
        )
        if not paths.legacy_decision_log.exists():
            paths.legacy_decision_log.write_text(
                _render_decision_log_header(project),
                encoding="utf-8",
            )
    paths.runs_dir.mkdir(parents=True, exist_ok=True)
    if not paths.current_case.exists():
        _write_yaml(paths.current_case, _render_default_current_case_payload(project))
    if not paths.decision_log.exists():
        paths.decision_log.write_text(_render_fast_decision_log_header(project), encoding="utf-8")
    if not paths.latest_run.exists():
        paths.latest_run.write_text(_render_empty_latest_run(project), encoding="utf-8")


def _resolve_bridge_vault_root(vault_root: str | Path | None) -> Path:
    resolved = resolve_vault_root(vault_root)
    if resolved is None:
        raise AlphaLabConfigError(
            "vault root is unresolved; pass --vault-root or set OBSIDIAN_VAULT_PATH"
        )
    if not resolved.exists():
        raise FileNotFoundError(f"vault root does not exist: {resolved}")
    if not resolved.is_dir():
        raise NotADirectoryError(f"vault root is not a directory: {resolved}")
    return resolved


def _project_paths(vault_root: Path, project_slug: str) -> ProjectPaths:
    safe_slug = _safe_slug(project_slug)
    project_dir = (vault_root / PROJECTS_DIRNAME / safe_slug).resolve()
    project_file = project_dir / "project.md"
    if not project_file.exists() and (project_dir / "project.yaml").exists():
        project_file = project_dir / "project.yaml"
    current_case_file = project_dir / "current_case.md"
    if not current_case_file.exists() and (project_dir / "current_case.yaml").exists():
        current_case_file = project_dir / "current_case.yaml"
    return ProjectPaths(
        vault_root=vault_root,
        project_dir=project_dir,
        project_yaml=project_file,
        current_case=current_case_file,
        decision_log=project_dir / "decision_log.md",
        runs_dir=project_dir / "runs",
        latest_run=project_dir / "runs" / "latest.md",
        project_brief=project_dir / "01_project_brief.md",
        project_rules=project_dir / "02_project_rules.md",
        card_map=project_dir / "03_card_map.md",
        active_state=project_dir / "10_active_state.md",
        legacy_decision_log=project_dir / "20_decision_log.md",
        recent_history=project_dir / "04_recent_history.md",
        rounds_dir=project_dir / "30_rounds",
        specs_dir=project_dir / "40_specs",
        drafts_dir=project_dir / "50_writeback_drafts",
    )


def _safe_slug(value: str) -> str:
    stripped = value.strip()
    if not stripped:
        raise AlphaLabConfigError("value must be non-empty")
    normalized = re.sub(r"\s+", "-", stripped)
    normalized = re.sub(r"[^A-Za-z0-9_.-]", "-", normalized)
    normalized = normalized.strip(".-_")
    if not normalized:
        raise AlphaLabConfigError(f"value is not valid for filesystem paths: {value!r}")
    return normalized


def _build_round_id(topic: str) -> str:
    topic_slug = str(topic).strip().lower().replace(" ", "-")
    topic_slug = re.sub(r"[^A-Za-z0-9_.-]", "-", topic_slug)
    topic_slug = re.sub(r"-+", "-", topic_slug).strip(".-_")
    if not topic_slug:
        topic_slug = "round"
    topic_slug = topic_slug[:24]
    timestamp = dt.datetime.now(dt.UTC).strftime("%Y%m%d-%H%M%S-%f")
    return f"{topic_slug}-{timestamp}"


def _normalize_bridge_mode(mode: str | None) -> str:
    normalized = str(mode or "fast").strip().lower()
    if normalized not in {"fast", "legacy"}:
        raise AlphaLabConfigError("bridge mode must be 'fast' or 'legacy'")
    return normalized


def _resolve_bridge_mode(mode: str | None, *, has_round: bool) -> str:
    if mode is not None and str(mode).strip():
        return _normalize_bridge_mode(mode)
    return "legacy" if has_round else "fast"


def _render_fast_run_summary(
    *,
    project: ProjectConfig,
    case_name: str,
    run_root: Path,
    verdict_status: str,
    metrics_digest: dict[str, str],
    upstream_summary: Path | None,
) -> str:
    lines = [
        f"# Run Summary - {case_name}",
        "",
        f"- `project`: `{project.slug}`",
        f"- `case_name`: `{case_name}`",
        f"- `verdict`: `{verdict_status}`",
        f"- `run_root`: `{run_root}`",
    ]
    if metrics_digest:
        lines.extend(
            [
                f"- `factor_verdict`: `{metrics_digest.get('factor_verdict', 'n/a')}`",
                f"- `mean_ic`: `{metrics_digest.get('mean_ic', 'n/a')}`",
                f"- `mean_rank_ic`: `{metrics_digest.get('mean_rank_ic', 'n/a')}`",
                (
                    f"- `mean_long_short_return`: "
                    f"`{metrics_digest.get('mean_long_short_return', 'n/a')}`"
                ),
                (f"- `promotion_decision`: `{metrics_digest.get('promotion_decision', 'n/a')}`"),
            ]
        )
    lines.extend(
        ["", "## Short Verdict", _build_fast_reason(metrics_digest, verdict_status=verdict_status)]
    )
    if upstream_summary is not None and upstream_summary.exists():
        lines.extend(
            [
                "",
                "## Upstream Summary",
                upstream_summary.read_text(encoding="utf-8").strip(),
            ]
        )
    lines.append("")
    return "\n".join(lines)


def _build_fast_reason(metrics_digest: dict[str, str], *, verdict_status: str) -> str:
    fragments: list[str] = []
    factor_verdict = str(metrics_digest.get("factor_verdict") or "").strip()
    if factor_verdict:
        fragments.append(f"factor_verdict={factor_verdict}")
    mean_ic = str(metrics_digest.get("mean_ic") or "").strip()
    if mean_ic:
        fragments.append(f"mean_ic={mean_ic}")
    mean_rank_ic = str(metrics_digest.get("mean_rank_ic") or "").strip()
    if mean_rank_ic:
        fragments.append(f"mean_rank_ic={mean_rank_ic}")
    mean_long_short = str(metrics_digest.get("mean_long_short_return") or "").strip()
    if mean_long_short:
        fragments.append(f"long_short={mean_long_short}")
    if not fragments:
        return f"verdict={verdict_status}; no structured metrics found."
    return f"verdict={verdict_status}; " + ", ".join(fragments)


def _render_fast_decision_log_header(project: ProjectConfig) -> str:
    return "\n".join(
        [
            f"# Decision Log - {project.title_zh}",
            "",
            "Fast mode project decisions. One run, one short verdict block.",
            "",
        ]
    )


def _render_decision_log_header(project: ProjectConfig) -> str:
    return "\n".join(
        [
            f"# 决策日志 - {project.title_zh}",
            "",
            "按时间倒序追加项目级结论。正式写回只在人工审核后发生。",
            "",
        ]
    )


def _render_default_current_case_payload(project: ProjectConfig) -> dict[str, Any]:
    return {
        "name": "pending_case",
        "project_slug": project.slug,
        "category": project.category,
        "market": project.market,
        "frequency": project.frequency,
        "data_source": project.alpha_lab_defaults.data_source,
        "evaluation_profile": project.alpha_lab_defaults.evaluation_profile,
        "status": "draft",
        "notes": "Replace this file with the active case contract before running.",
    }


def _render_empty_latest_run(project: ProjectConfig) -> str:
    return "\n".join(
        [
            f"# Latest Run - {project.title_zh}",
            "",
            "- `case_name`: `pending_case`",
            "- `verdict`: `pending`",
            "- `summary_path`: `runs/pending_case/summary.md`",
            "",
            "No run has been summarized yet.",
            "",
        ]
    )


def _render_latest_run_index(
    *,
    project: ProjectConfig,
    case_name: str,
    verdict_status: str,
) -> str:
    return "\n".join(
        [
            f"# Latest Run - {project.title_zh}",
            "",
            f"- `case_name`: `{case_name}`",
            f"- `verdict`: `{verdict_status}`",
            f"- `summary_path`: `runs/{_safe_slug(case_name)}/summary.md`",
            "",
            f"Open `runs/{_safe_slug(case_name)}/summary.md` for the full short summary.",
            "",
        ]
    )


def _read_optional_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _card_excerpt(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    return _truncate_text(_strip_frontmatter(text), max_chars=400)


def _render_project_brief(project: ProjectConfig) -> str:
    return "\n".join(
        [
            f"# 项目简报 - {project.title_zh}",
            "",
            f"- `slug`: `{project.slug}`",
            f"- `category`: `{project.category}`",
            f"- `market`: `{project.market}`",
            f"- `frequency`: `{project.frequency}`",
            f"- `current_hypothesis`: {project.status.current_hypothesis}",
            "",
        ]
    )


def _render_project_rules(project: ProjectConfig) -> str:
    return "\n".join(
        [
            f"# 项目规则 - {project.title_zh}",
            "",
            "- 优先压缩为可执行 case。",
            "- 先验证，再决定是否正式写回。",
            "",
        ]
    )


def _render_active_state(project: ProjectConfig) -> str:
    return "\n".join(
        [
            f"# Active State - {project.title_zh}",
            "",
            f"- `current_hypothesis`: {project.status.current_hypothesis}",
            f"- `current_focus`: {project.status.current_focus}",
            f"- `next_action`: {project.status.next_action}",
            "",
        ]
    )


def _render_card_map(*, project: ProjectConfig, vault_root: Path) -> str:
    lines = [f"# Card Map - {project.title_zh}", ""]
    for raw_card in project.origin_cards + project.supporting_cards + project.failure_cards:
        resolved = (vault_root / raw_card).resolve()
        lines.append(f"## {raw_card}")
        lines.append(_card_excerpt(resolved) if resolved.exists() else "missing")
        lines.append("")
    return "\n".join(lines)


def _render_recent_history(
    *,
    project: ProjectConfig,
    paths: ProjectPaths,
    max_decision_entries: int = 5,
    max_feedback_entries: int = 3,
) -> str:
    del max_decision_entries, max_feedback_entries
    return "\n".join(
        [
            f"# 近期历史 - {project.title_zh}",
            "",
            _read_optional_text(paths.decision_log).strip() or "暂无决策记录。",
            "",
        ]
    )


def _read_latest_experiment_feedback(paths: ProjectPaths) -> str:
    candidates = sorted(paths.rounds_dir.rglob("latest_experiment_feedback.md"))
    if not candidates:
        return ""
    return _truncate_text(candidates[-1].read_text(encoding="utf-8"), max_chars=1200)


def _read_latest_discussion_capture(paths: ProjectPaths) -> str:
    candidates = sorted(paths.rounds_dir.rglob("discussion_capture.md"))
    if not candidates:
        return ""
    return _truncate_text(candidates[-1].read_text(encoding="utf-8"), max_chars=1200)


def _render_card_excerpt_block(cards: list[str], vault_root: Path) -> str:
    if not cards:
        return "- 暂无白名单卡片。"
    lines: list[str] = []
    for raw_card in cards:
        resolved = (vault_root / raw_card).resolve()
        lines.append(f"### `{raw_card}`")
        lines.append(_card_excerpt(resolved) if resolved.exists() else "文件不存在")
        lines.append("")
    return "\n".join(lines).rstrip()


def _render_graph_context(project: ProjectConfig, *, vault_root: Path) -> str:
    graph = _load_vault_graph(vault_root)
    if graph is None:
        return "- graph.json unavailable."
    lines: list[str] = []
    for raw_card in project.origin_cards + project.supporting_cards:
        name = _card_name_from_path(vault_root / raw_card)
        if not name:
            continue
        node = graph.get_node(name)
        depends_on = graph.get_neighbors(name, edge_type="depends_on")[:4]
        lines.append(f"### {name}")
        if node is not None:
            lines.append(
                f"- `type`: `{node.type or 'unknown'}`; "
                f"`mechanism`: `{node.mechanism or ''}`; "
                f"`factor_family`: `{node.factor_family or ''}`"
            )
        lines.append(
            "- `depends_on`: "
            + (", ".join(f"`{item}`" for item in depends_on) if depends_on else "无")
        )
        lines.append("")
    return "\n".join(lines).rstrip() or "- 图谱上下文为空。"


def _render_exploration_frontier(
    *,
    project: ProjectConfig,
    vault_root: Path,
    exploration: ExplorationMap | None,
) -> str:
    if exploration is None:
        return "- exploration frontier unavailable."
    mechanisms, families = _project_graph_labels(project=project, vault_root=vault_root)
    items: list[FrontierEntry] = []
    if families:
        items.extend(exploration.frontier(factor_family=families[0])[:3])
    if not items and mechanisms:
        items.extend(exploration.frontier(mechanism=mechanisms[0])[:3])
    if not items:
        items.extend(exploration.frontier()[:3])
    if not items:
        return "- 当前 frontier 为空。"
    lines: list[str] = []
    for item in items:
        lines.append(f"### {item.direction}")
        lines.append(f"- `mechanism`: `{item.mechanism}`; `factor_family`: `{item.factor_family}`")
        lines.append(f"- `reason`: {item.reason}")
        lines.append("")
    return "\n".join(lines).rstrip()


def _render_related_failure_knowledge(
    *,
    project: ProjectConfig,
    topic: str,
    vault_root: Path,
    exploration: ExplorationMap | None,
) -> str:
    if exploration is None:
        return "- failure knowledge unavailable."
    mechanisms, families = _project_graph_labels(project=project, vault_root=vault_root)
    items = exploration.related_failures(
        factor_family=families[0] if families else "",
        mechanism=mechanisms[0] if mechanisms else "",
        text_query=topic,
    )[:3]
    if not items:
        return "- 未命中与当前主题直接相关的 failure knowledge。"
    lines: list[str] = []
    for item in items:
        lines.append(f"### [{item.failure_id}] {item.title}")
        if item.failure_statement:
            lines.append(f"- `failure_statement`: {item.failure_statement}")
        lines.append("")
    return "\n".join(lines).rstrip()


def _render_divergence_context(
    *,
    project: ProjectConfig,
    vault_root: Path,
    exploration: ExplorationMap | None,
) -> str:
    graph = _load_vault_graph(vault_root)
    embeddings = _load_vault_embeddings(vault_root)
    if graph is None:
        return "- graph unavailable."
    context = build_divergence_context(
        project=project,
        vault_root=vault_root,
        graph=graph,
        embeddings=embeddings,
        exploration=exploration,
    )
    lines: list[str] = []
    for item in context.graph_guided[:3]:
        lines.append(f"- `{item.name}`: {item.reason}")
    for item in context.random_walk[:2]:
        lines.append(f"- `{item.name}`: {item.reason}")
    return "\n".join(lines).rstrip() or "- no controlled divergence seeds."


def _render_round_context_digest(
    *,
    project: ProjectConfig,
    paths: ProjectPaths,
    topic: str,
    mode: str = "standard",
) -> str:
    exploration = _load_exploration_map(paths.vault_root)
    lines = [
        f"# 本轮上下文摘要 - {project.title_zh}",
        "",
        "## 当前研究状态",
        _read_optional_text(paths.active_state).strip() or _render_active_state(project),
        "",
        "## 必须继承的已有知识",
        _render_card_excerpt_block(
            project.origin_cards + project.supporting_cards, paths.vault_root
        ),
        "",
        "## 图谱相关上下文",
        _render_graph_context(project, vault_root=paths.vault_root),
        "",
        "## Exploration Frontier",
        _render_exploration_frontier(
            project=project, vault_root=paths.vault_root, exploration=exploration
        ),
        "",
        "## Related Failure Knowledge",
        _render_related_failure_knowledge(
            project=project,
            topic=topic,
            vault_root=paths.vault_root,
            exploration=exploration,
        ),
        "",
        "## 最近实验反馈",
        _read_latest_experiment_feedback(paths) or "暂无实验反馈。",
        "",
        "## 最近一轮讨论摘录",
        _read_latest_discussion_capture(paths) or "暂无历史讨论摘录。",
        "",
    ]
    if mode == "explore":
        lines.extend(
            [
                "## Controlled Divergence Seeds",
                _render_divergence_context(
                    project=project,
                    vault_root=paths.vault_root,
                    exploration=exploration,
                ),
                "",
            ]
        )
    return "\n".join(lines)


def _render_round_prompt(
    project: ProjectConfig,
    *,
    round_id: str,
    topic: str,
    mode: str = "standard",
) -> str:
    return "\n".join(
        [
            f"# Round Prompt - {round_id}",
            "",
            "```text",
            f"项目：{project.title_zh}",
            f"主题：{topic}",
            f"模式：{mode}",
            "请把想法压缩成可执行 case。",
            "```",
            "",
        ]
    )


def _render_web_search_tasks(project: ProjectConfig, *, topic: str) -> str:
    return "\n".join(
        [
            "# Web Search Tasks",
            "",
            f"- topic: {topic}",
            f"- preferred_sources: {', '.join(project.preferred_web_sources) or 'official docs'}",
            "",
        ]
    )


def _render_discussion_capture_template(
    *,
    project: ProjectConfig,
    round_id: str,
    topic: str,
) -> str:
    return "\n".join(
        [
            f"# Discussion Capture - {round_id}",
            "",
            f"- `project`: `{project.slug}`",
            f"- `topic`: {topic}",
            "",
            "## 本轮确认的新假设",
            "- ",
            "",
        ]
    )


def _render_knowledge_handoff(
    *,
    project: ProjectConfig,
    round_id: str,
    case_name: str,
    case_type: str,
    factor_name: str,
    discussion_capture: str,
    spec_path: Path,
    preflight_report: str = "",
) -> str:
    lines = [
        "# 知识交接模板",
        "",
        f"- `project`: `{project.slug}`",
        f"- `round_id`: `{round_id}`",
        f"- `case_name`: `{case_name}`",
        f"- `case_type`: `{case_type}`",
        f"- `factor_name`: `{factor_name}`",
        f"- `candidate_spec_path`: {spec_path}",
        "",
    ]
    if preflight_report.strip():
        lines.extend([preflight_report.strip(), ""])
    lines.extend(
        [
            "## 本轮讨论摘录",
            discussion_capture.strip() or "待补充",
            "",
        ]
    )
    return "\n".join(lines)


def _render_latest_experiment_feedback(
    *,
    project: ProjectConfig,
    round_id: str,
    case_name: str,
    run_root: Path,
    metrics_digest: dict[str, str],
    summary_path: Path | None,
    experiment_card_path: Path | None,
) -> str:
    return "\n".join(
        [
            f"# Latest Experiment Feedback - {case_name}",
            "",
            f"- `project`: `{project.slug}`",
            f"- `round_id`: `{round_id}`",
            f"- `run_root`: `{run_root}`",
            f"- `summary_path`: `{summary_path}`"
            if summary_path is not None
            else "- `summary_path`: 缺失",
            f"- `experiment_card_path`: `{experiment_card_path}`"
            if experiment_card_path is not None
            else "- `experiment_card_path`: 缺失",
            "",
            "## 机器摘要",
            f"- `factor_verdict`: {metrics_digest.get('factor_verdict', 'n/a')}",
            f"- `mean_rank_ic`: {metrics_digest.get('mean_rank_ic', 'n/a')}",
            f"- `mean_ic`: {metrics_digest.get('mean_ic', 'n/a')}",
            f"- `long_short_return`: {metrics_digest.get('mean_long_short_return', 'n/a')}",
            f"- `promotion_decision`: {metrics_digest.get('promotion_decision', 'n/a')}",
            "",
        ]
    )


def _render_writeback_draft(
    *,
    project: ProjectConfig,
    round_id: str,
    case_name: str,
    run_root: Path,
    manifest_path: Path,
    metrics_path: Path | None,
    summary_path: Path | None,
    experiment_card_path: Path | None,
    verdict_status: str,
    metrics_digest: dict[str, str],
    graph_feedback: dict[str, object],
) -> str:
    frontmatter: dict[str, Any] = {
        "type": "research_bridge_writeback_draft",
        "project": project.slug,
        "round_id": round_id,
        "case_name": case_name,
        "run_root": str(run_root),
        "manifest_path": str(manifest_path),
        "metrics_path": str(metrics_path) if metrics_path is not None else "",
        "summary_path": str(summary_path) if summary_path is not None else "",
        "experiment_card_path": str(experiment_card_path)
        if experiment_card_path is not None
        else "",
        "review_status": "pending",
        "reviewed_by": "",
        "reviewed_at": "",
        "verdict_status": verdict_status,
        "one_sentence_verdict": "",
        "status_lifecycle": project.status.lifecycle,
        "current_hypothesis": project.status.current_hypothesis,
        "current_focus": project.status.current_focus,
        "next_action": project.status.next_action,
        "vault_export_mode": "versioned",
    }
    lines = [
        f"# 写回草稿 - {case_name}",
        "",
        "## 机器摘要",
        f"- `factor_verdict`: {metrics_digest.get('factor_verdict', 'n/a')}",
        f"- `mean_rank_ic`: {metrics_digest.get('mean_rank_ic', 'n/a')}",
        f"- `mean_ic`: {metrics_digest.get('mean_ic', 'n/a')}",
        "",
        "## Graph Feedback",
    ]
    for item in _object_to_str_list(graph_feedback.get("suggested_similar_to")):
        lines.append(f"- `suggested_similar_to`: {item}")
    correlation_summary = graph_feedback.get("correlation_summary")
    if isinstance(correlation_summary, str):
        text = correlation_summary.strip()
        if text:
            lines.append("## Correlation Summary")
            lines.append(text)
    elif isinstance(correlation_summary, list):
        for item in correlation_summary:
            if isinstance(item, dict):
                lines.append(f"- {item.get('name', 'unknown')}")
    return _compose_markdown_with_frontmatter(frontmatter, "\n".join(lines))


def _render_state_update_patch(
    *,
    project: ProjectConfig,
    round_id: str,
    case_name: str,
    verdict_status: str,
    metrics_digest: dict[str, str],
) -> str:
    return "\n".join(
        [
            f"# State Update Patch - {case_name}",
            "",
            f"- `project`: `{project.slug}`",
            f"- `round_id`: `{round_id}`",
            f"- `suggested_verdict_status`: `{verdict_status}`",
            f"- `mean_rank_ic`: {metrics_digest.get('mean_rank_ic', 'n/a')}",
            "",
        ]
    )


def _render_structured_candidates(
    *,
    project: ProjectConfig,
    round_id: str,
    candidates: list[StructuredCandidate],
) -> str:
    lines = [
        f"# Structured Candidates - {round_id}",
        "",
        f"- `project`: `{project.slug}`",
        "",
    ]
    for idx, candidate in enumerate(candidates, start=1):
        lines.extend(
            [
                f"## Candidate {idx}: {candidate.candidate_name}",
                f"- `raw_idea`: {candidate.raw_idea}",
                f"- `suggested_mechanism`: {candidate.suggested_mechanism or 'unknown'}",
                f"- `suggested_factor_family`: {candidate.suggested_factor_family or 'unknown'}",
                f"- `novelty_score`: {candidate.novelty_score:.3f}",
                "- `semantic_matches`:",
            ]
        )
        for match in candidate.semantic_matches:
            lines.append(f"  - [{match.score:.3f}] {match.name} ({match.type})")
        lines.append("- `related_cards`:")
        for related in candidate.related_cards:
            lines.append(f"  - {related}")
        lines.append("")
    return "\n".join(lines)


def _render_structured_handoff_draft(
    *,
    project: ProjectConfig,
    round_id: str,
    candidates: list[StructuredCandidate],
) -> str:
    lines = [
        f"# Knowledge Handoff Draft - {round_id}",
        "",
        f"- `project`: `{project.slug}`",
        "",
    ]
    for idx, candidate in enumerate(candidates, start=1):
        lines.extend(
            [
                f"## Candidate {idx}: {candidate.candidate_name}",
                f"- `hypothesis`: {candidate.raw_idea}",
                f"- `mechanism_label`: {candidate.suggested_mechanism or '待确认'}",
                f"- `factor_family_label`: {candidate.suggested_factor_family or '待确认'}",
                "",
            ]
        )
    return "\n".join(lines)


def _extract_candidate_ideas(text: str) -> list[str]:
    lines = text.splitlines()
    in_section = False
    ideas: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("## "):
            in_section = stripped == "## 本轮确认的新假设"
            continue
        if in_section and stripped.startswith("- "):
            idea = stripped[2:].strip()
            if idea:
                ideas.append(idea)
    return ideas


def _load_markdown_with_frontmatter(path: Path) -> tuple[dict[str, Any], str]:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        raise AlphaLabDataError(f"markdown draft is missing YAML frontmatter: {path}")
    try:
        _, raw_frontmatter, body = text.split("---\n", 2)
    except ValueError as exc:
        raise AlphaLabDataError(f"markdown draft has invalid frontmatter fence: {path}") from exc
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover
        raise AlphaLabExperimentError("PyYAML is required to parse bridge drafts") from exc
    payload = yaml.safe_load(raw_frontmatter)
    if not isinstance(payload, dict):
        raise AlphaLabDataError(f"markdown draft frontmatter must be an object: {path}")
    return payload, body.lstrip("\n")


def _compose_markdown_with_frontmatter(frontmatter: dict[str, Any], body: str) -> str:
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover
        raise AlphaLabExperimentError("PyYAML is required to serialize bridge drafts") from exc
    rendered_frontmatter = yaml.safe_dump(frontmatter, sort_keys=False, allow_unicode=True).strip()
    return f"---\n{rendered_frontmatter}\n---\n\n{body.rstrip()}\n"


def _frontmatter_path(frontmatter: dict[str, Any], key: str) -> Path | None:
    raw_value = str(frontmatter.get(key) or "").strip()
    if not raw_value:
        return None
    return Path(raw_value).expanduser().resolve()


def _load_vault_graph(vault_root: Path) -> VaultGraph | None:
    graph_path = (vault_root / "90_computed" / "graph.json").resolve()
    if not graph_path.exists():
        return None
    graph = VaultGraph(graph_path)
    try:
        graph.load()
    except (OSError, ValueError):
        return None
    return graph


def _load_vault_embeddings(vault_root: Path) -> VaultEmbeddings | None:
    embeddings_path = (vault_root / "90_computed" / "embeddings.npz").resolve()
    if not embeddings_path.exists():
        return None
    embeddings = VaultEmbeddings(embeddings_path)
    try:
        embeddings.load()
    except (OSError, ValueError):
        return None
    return embeddings


def _load_exploration_map(vault_root: Path) -> ExplorationMap | None:
    path = (vault_root / "90_computed" / "exploration_map.json").resolve()
    if not path.exists():
        return None
    exploration = ExplorationMap(path)
    try:
        exploration.load()
    except (OSError, ValueError):
        return None
    return exploration


def _project_graph_labels(
    *, project: ProjectConfig, vault_root: Path
) -> tuple[list[str], list[str]]:
    graph = _load_vault_graph(vault_root)
    if graph is None:
        return [], []
    mechanisms: list[str] = []
    families: list[str] = []
    for raw_card in project.origin_cards + project.supporting_cards:
        name = _card_name_from_path(vault_root / raw_card)
        if not name:
            continue
        node = graph.get_node(name)
        if node is None:
            continue
        if node.mechanism and node.mechanism not in mechanisms:
            mechanisms.append(node.mechanism)
        if node.factor_family and node.factor_family not in families:
            families.append(node.factor_family)
    return mechanisms, families


def _card_name_from_path(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return ""
    if text.startswith("---\n"):
        end = text.find("\n---", 4)
        if end != -1:
            frontmatter = text[4:end]
            for line in frontmatter.splitlines():
                if line.startswith("name:"):
                    return str(line.split(":", 1)[1]).strip().strip('"')
    stem = path.stem
    if stem.startswith("Factor - "):
        return stem.removeprefix("Factor - ").strip()
    if stem.startswith("Method - "):
        return stem.removeprefix("Method - ").strip()
    if stem.startswith("Concept - "):
        return stem.removeprefix("Concept - ").strip()
    return stem.strip()


def _candidate_name_from_idea(raw_idea: str) -> str:
    cleaned = raw_idea.strip()
    cleaned = re.split(r"[。.;:：]", cleaned, maxsplit=1)[0].strip()
    return cleaned[:80] or "candidate"


def _suggest_label(
    graph: VaultGraph,
    matches: list[SearchResult],
    *,
    label_getter: Any,
) -> str:
    del graph
    weights: dict[str, float] = {}
    for match in matches:
        label = str(label_getter(match.name) or "").strip()
        if not label:
            continue
        weights[label] = weights.get(label, 0.0) + match.score
    if not weights:
        return ""
    return max(weights.items(), key=lambda item: item[1])[0]


def _graph_node_factor_family(graph: VaultGraph, name: str) -> str:
    node = graph.get_node(name)
    if node is None:
        return ""
    return str(node.factor_family or "").strip()


def _graph_node_mechanism(graph: VaultGraph, name: str) -> str:
    node = graph.get_node(name)
    if node is None:
        return ""
    return str(node.mechanism or "").strip()


def _object_to_str_list(value: object) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    items: list[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            items.append(text)
    return items


def _object_to_dict_list(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    rows: list[dict[str, object]] = []
    for item in value:
        if isinstance(item, dict):
            rows.append({str(key): val for key, val in item.items()})
    return rows


def _object_to_dict(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        return {}
    return {str(key): val for key, val in value.items()}


def _normalize_explore_mode(mode: str) -> str:
    normalized = mode.strip().lower()
    return _EXPLORE_MODE_ALIASES.get(normalized, "free")


def _load_project_optional(vault_root: Path, project_slug: str | None) -> ProjectConfig | None:
    if project_slug is None or not project_slug.strip():
        return None
    paths = _project_paths(vault_root, project_slug)
    if not paths.project_yaml.exists():
        raise FileNotFoundError(f"project not found: {project_slug}")
    return load_project_config(paths.project_yaml)


def _load_card_index_rows(vault_root: Path) -> list[dict[str, str]]:
    index_path = vault_root / "90_moc" / "CARD-INDEX.tsv"
    if not index_path.exists():
        return []
    rows: list[dict[str, str]] = []
    with index_path.open("r", encoding="utf-8") as fh:
        reader = DictReader(fh, delimiter="\t")
        for row in reader:
            rows.append({key: str(value or "").strip() for key, value in row.items()})
    return rows


def _search_explore_matches(
    *,
    idea: str,
    embeddings: VaultEmbeddings | None,
    index_rows: list[dict[str, str]],
    top_k: int,
) -> list[SearchResult]:
    if embeddings is not None:
        results = embeddings.search(idea, top_k=max(top_k, 1))
        if results:
            return results
    keywords = _idea_keywords(idea)
    if not keywords:
        return []
    scored_rows: list[tuple[float, dict[str, str]]] = []
    for row in index_rows:
        haystack = " ".join(
            [
                row.get("path", ""),
                row.get("type", ""),
                row.get("name", ""),
                row.get("domain", ""),
                row.get("lifecycle", ""),
                row.get("tags", ""),
                row.get("parent_moc", ""),
                row.get("summary", ""),
            ]
        ).lower()
        score = float(sum(1 for keyword in keywords if keyword in haystack))
        if score <= 0.0:
            continue
        scored_rows.append((score, row))
    scored_rows.sort(key=lambda item: (-item[0], item[1].get("name", ""), item[1].get("path", "")))
    return [
        SearchResult(
            name=row.get("name", ""),
            score=score,
            type=row.get("type", ""),
            path=row.get("path", ""),
            summary=row.get("summary", ""),
        )
        for score, row in scored_rows[: max(top_k, 1)]
        if row.get("name") and row.get("path")
    ]


def _idea_keywords(text: str) -> set[str]:
    keywords: set[str] = set()
    for token in re.findall(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]+", text):
        normalized = token.lower().strip()
        if len(normalized) <= 1:
            continue
        keywords.add(normalized)
        if re.fullmatch(r"[\u4e00-\u9fff]+", normalized) and len(normalized) >= 4:
            for idx in range(len(normalized) - 1):
                keywords.add(normalized[idx : idx + 2])
    return keywords


def _build_explore_related_cards(
    *,
    vault_root: Path,
    idea: str,
    graph: VaultGraph | None,
    embeddings: VaultEmbeddings | None,
    project: ProjectConfig | None,
    semantic_matches: list[SearchResult],
    index_rows: list[dict[str, str]],
    top_k: int,
) -> list[ExploreIdeaCard]:
    del idea
    index_by_name = {row.get("name", ""): row for row in index_rows if row.get("name")}
    ordered_names: list[str] = []
    reasons_by_name: dict[str, set[str]] = {}
    match_by_name = {match.name: match for match in semantic_matches}

    def enqueue(name: str, reason: str) -> None:
        if not name:
            return
        reasons = reasons_by_name.setdefault(name, set())
        reasons.add(reason)
        if name not in ordered_names:
            ordered_names.append(name)

    for match in semantic_matches[: max(top_k, 4)]:
        enqueue(match.name, "semantic_match")

    if graph is not None:
        for match in semantic_matches[:2]:
            node = graph.get_node(match.name)
            if node is None:
                continue
            for related_name in graph.find_similar(match.name)[:2]:
                enqueue(related_name, "similar_to")
            for related_name in graph.get_neighbors(match.name, edge_type="depends_on")[:2]:
                enqueue(related_name, "depends_on")
            for related_name in graph.get_reverse_dependencies(match.name)[:2]:
                enqueue(related_name, "reverse_dependency")
            if node.factor_family:
                for related_name in graph.get_factor_family(node.factor_family)[:3]:
                    if related_name != match.name:
                        enqueue(related_name, "same_family")
            if node.mechanism:
                for related_name in graph.get_by_mechanism(node.mechanism)[:3]:
                    if related_name != match.name:
                        enqueue(related_name, "same_mechanism")

    if project is not None:
        for raw_card in project.origin_cards + project.supporting_cards:
            enqueue(_card_name_from_path(vault_root / raw_card), "project_context")

    cards: list[ExploreIdeaCard] = []
    for name in ordered_names:
        card = _build_explore_card(
            vault_root=vault_root,
            graph=graph,
            embeddings=embeddings,
            match=match_by_name.get(name),
            index_row=index_by_name.get(name),
            name=name,
            reasons=sorted(reasons_by_name.get(name, set())),
        )
        if card is None:
            continue
        cards.append(card)
        if len(cards) >= max(top_k, 1):
            break
    return cards


def _build_explore_card(
    *,
    vault_root: Path,
    graph: VaultGraph | None,
    embeddings: VaultEmbeddings | None,
    match: SearchResult | None,
    index_row: dict[str, str] | None,
    name: str,
    reasons: list[str],
) -> ExploreIdeaCard | None:
    node = graph.get_node(name) if graph is not None else None
    entry = match or (embeddings.get_entry(name) if embeddings is not None else None)
    rel_path = ""
    if node is not None and node.path:
        rel_path = node.path
    elif entry is not None and entry.path:
        rel_path = entry.path
    elif index_row is not None:
        rel_path = index_row.get("path", "")
    if not rel_path:
        return None

    card_path = (vault_root / rel_path).resolve()
    if not str(card_path).startswith(str(vault_root.resolve())):
        return None
    if not card_path.exists():
        return None

    text = card_path.read_text(encoding="utf-8", errors="replace")
    frontmatter = _extract_simple_frontmatter(text)
    body = _strip_frontmatter(text)
    summary = (
        (index_row or {}).get("summary", "")
        or frontmatter.get("summary", "")
        or (entry.summary if entry is not None else "")
        or _truncate_text(body, max_chars=200)
    )
    snippet = _truncate_text(body, max_chars=280)
    type_name = node.type if node is not None and node.type else (index_row or {}).get("type", "")
    lifecycle = (
        node.lifecycle
        if node is not None and node.lifecycle
        else (index_row or {}).get("lifecycle", "")
    )
    mechanism = (
        node.mechanism if node is not None and node.mechanism else frontmatter.get("mechanism", "")
    )
    factor_family = (
        node.factor_family
        if node is not None and node.factor_family
        else frontmatter.get("factor_family", "")
    )
    return ExploreIdeaCard(
        path=rel_path,
        name=name,
        type=type_name,
        lifecycle=lifecycle,
        mechanism=mechanism,
        factor_family=factor_family,
        summary=summary,
        snippet=snippet,
        reasons=reasons,
    )


def _build_explore_constraint_report(
    *,
    idea: str,
    graph: VaultGraph | None,
    exploration: ExplorationMap | None,
    factor_matches: list[SearchResult],
    related_cards: list[ExploreIdeaCard],
    suggested_family: str,
    suggested_mechanism: str,
) -> dict[str, object]:
    validated_peers: list[str] = []
    novelty_warnings: list[str] = []
    family_counts: dict[str, int] = {}
    candidate_name = _candidate_name_from_idea(idea)
    if graph is not None:
        matrix = graph.mechanism_family_matrix()
        family_counter: Counter[str] = Counter()
        for family_map in matrix.values():
            for family, names in family_map.items():
                family_counter[family] += len(names)
        family_counts = dict(family_counter.most_common(8))
        if suggested_family and suggested_mechanism:
            validated_peers = [
                name
                for name in graph.get_mechanism_family_factors(
                    suggested_family,
                    suggested_mechanism,
                    validated_only=True,
                )
                if name != candidate_name
            ]
            similar_existing = [match.name for match in factor_matches[:3] if match.score >= 0.18]
            novelty = graph.check_novelty(
                candidate_name=candidate_name,
                candidate_similar=similar_existing,
                candidate_mechanism=suggested_mechanism,
                candidate_family=suggested_family,
            )
            novelty_warnings = list(novelty.warnings)
    if not family_counts:
        family_counts = dict(
            Counter(card.factor_family for card in related_cards if card.factor_family).most_common(
                8
            )
        )

    crowding_warning = ""
    if suggested_family and suggested_mechanism and len(validated_peers) >= 3:
        crowding_warning = (
            f"{suggested_family}/{suggested_mechanism} already has "
            f"{len(validated_peers)} validated factors: {', '.join(validated_peers[:5])}"
        )

    frontier_matches: list[dict[str, str]] = []
    failure_refs: list[dict[str, str]] = []
    if exploration is not None:
        frontier = _select_explore_frontier_matches(
            exploration=exploration,
            factor_family=suggested_family,
            mechanism=suggested_mechanism,
        )
        frontier_matches = [
            {
                "direction": item.direction,
                "factor_family": item.factor_family,
                "mechanism": item.mechanism,
                "reason": item.reason,
                "suggested_by": item.suggested_by,
                "priority": item.priority,
            }
            for item in frontier
        ]
        failure_refs = [
            {
                "failure_id": item.failure_id,
                "title": item.title,
                "status": item.status,
                "failure_class": item.failure_class,
                "failure_statement": item.failure_statement,
            }
            for item in exploration.related_failures(
                factor_family=suggested_family,
                mechanism=suggested_mechanism,
                text_query=idea,
            )[:3]
        ]

    return {
        "primary_family": suggested_family,
        "primary_mechanism": suggested_mechanism,
        "family_counts": family_counts,
        "crowding_warning": crowding_warning,
        "novelty_warnings": novelty_warnings,
        "validated_peers": validated_peers,
        "frontier_matches": frontier_matches,
        "failure_refs": failure_refs,
    }


def _select_explore_frontier_matches(
    *,
    exploration: ExplorationMap,
    factor_family: str,
    mechanism: str,
) -> list[FrontierEntry]:
    matches: list[FrontierEntry] = []
    seen: set[tuple[str, str]] = set()
    if factor_family:
        for item in exploration.frontier(factor_family=factor_family)[:3]:
            key = (item.factor_family, item.mechanism)
            if key not in seen:
                seen.add(key)
                matches.append(item)
    if mechanism:
        for item in exploration.frontier(mechanism=mechanism)[:3]:
            key = (item.factor_family, item.mechanism)
            if key not in seen:
                seen.add(key)
                matches.append(item)
    if matches:
        return matches[:3]
    return exploration.frontier(priority="high")[:3] or exploration.frontier()[:3]


def _load_explore_graph(vault_root: Path) -> VaultGraph | None:
    graph = _load_vault_graph(vault_root)
    if graph is not None:
        return graph
    script_path = vault_root / "00_protocols" / "rebuild-graph.py"
    if not script_path.exists():
        return None
    graph = VaultGraph.from_vault_root(vault_root)
    try:
        graph.build(vault_root=vault_root)
    except Exception:
        return None
    return graph


def _load_explore_embeddings(vault_root: Path) -> VaultEmbeddings | None:
    embeddings = _load_vault_embeddings(vault_root)
    if embeddings is not None:
        return embeddings
    script_path = vault_root / "00_protocols" / "rebuild-embeddings.py"
    if not script_path.exists():
        return None
    embeddings = VaultEmbeddings.from_vault_root(vault_root)
    try:
        embeddings.build(vault_root=vault_root)
    except Exception:
        return None
    return embeddings


def _load_explore_exploration_map(vault_root: Path) -> ExplorationMap | None:
    exploration = _load_exploration_map(vault_root)
    if exploration is not None:
        return exploration
    script_path = vault_root / "00_protocols" / "rebuild-exploration-map.py"
    if not script_path.exists():
        return None
    exploration = ExplorationMap.from_vault_root(vault_root)
    try:
        exploration.build(vault_root=vault_root)
    except Exception:
        return None
    return exploration


def _extract_simple_frontmatter(text: str) -> dict[str, str]:
    if not text.startswith("---\n"):
        return {}
    try:
        _, frontmatter, _ = text.split("---\n", 2)
    except ValueError:
        return {}
    payload: dict[str, str] = {}
    for line in frontmatter.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or ":" not in stripped:
            continue
        if line.startswith(" ") or line.startswith("-"):
            continue
        key, value = stripped.split(":", 1)
        payload[key.strip()] = value.strip().strip('"').strip("'")
    return payload


def _strip_frontmatter(text: str) -> str:
    if not text.startswith("---\n"):
        return text.strip()
    end = text.find("\n---", 4)
    if end == -1:
        return text.strip()
    return text[end + 5 :].strip()


def _first_non_empty(values: Any) -> str:
    for value in values:
        normalized = str(value or "").strip()
        if normalized:
            return normalized
    return ""


def _build_exploration_prompt(
    *,
    idea: str,
    mode: str,
    cards: list[ExploreIdeaCard],
    constraint_report: dict[str, object],
    category: str,
    project: ProjectConfig | None,
    graph: VaultGraph | None,
) -> str:
    if category == "factor_recipe":
        return _build_factor_recipe_exploration_prompt(
            idea=idea,
            mode=mode,
            cards=cards,
            constraint_report=constraint_report,
            project=project,
            graph=graph,
        )

    profile = get_category_profile(category)
    mode_label = "约束探索（Graph WEAK）" if mode == "constrained" else "自由发散（Graph OFF）"
    lines: list[str] = [
        f"# 研究探索上下文（{profile.display_name_zh}）",
        f"> 模式：{mode_label}",
        f"> 生成时间：{_utc_now_iso()}",
    ]
    if project is not None:
        lines.append(f"> 项目：{project.slug} | {project.title_zh}")
    lines.extend(
        [
            "",
            "## 你的想法",
            idea,
            "",
            "## 相关知识库卡片",
        ]
    )
    if not cards:
        lines.append("（未找到相关卡片，建议检查 CARD-INDEX.tsv / embeddings 是否可用）")
    for card in cards:
        meta = [f"类型：{card.type or '?'}", f"状态：{card.lifecycle or '?'}"]
        if card.mechanism:
            meta.append(f"机制：{card.mechanism}")
        if card.factor_family:
            meta.append(f"因子族：{card.factor_family}")
        lines.extend(
            [
                f"### {card.name}",
                " | ".join(meta),
                card.summary,
            ]
        )
        if card.snippet and card.snippet != card.summary:
            lines.append(card.snippet)
        lines.append("---")

    if mode == "constrained" and constraint_report:
        lines.extend(["", "## 约束报告（Graph 约束层）"])
        primary_family = str(constraint_report.get("primary_family") or "")
        primary_mechanism = str(constraint_report.get("primary_mechanism") or "")
        if primary_family:
            lines.append(f"- 推断因子族：**{primary_family}**")
        if primary_mechanism:
            lines.append(f"- 推断机制：**{primary_mechanism}**")
        crowding_warning = str(constraint_report.get("crowding_warning") or "")
        if crowding_warning:
            lines.append(f"- ⚠ {crowding_warning}")
        novelty_warnings = _object_to_str_list(constraint_report.get("novelty_warnings"))
        for warning in novelty_warnings:
            lines.append(f"- novelty: {warning}")
        family_counts = _object_to_dict(constraint_report.get("family_counts"))
        if family_counts:
            distribution = ", ".join(
                f"{family}({count})" for family, count in family_counts.items()
            )
            lines.append(f"- 现有知识分布：{distribution}")
        frontier_matches = _object_to_dict_list(constraint_report.get("frontier_matches"))
        if frontier_matches:
            lines.append("- frontier:")
            for frontier_item in frontier_matches[:3]:
                lines.append(f"  - {frontier_item['direction']}: {frontier_item['reason']}")
        failure_refs = _object_to_dict_list(constraint_report.get("failure_refs"))
        if failure_refs:
            lines.append("- failure knowledge:")
            for failure_item in failure_refs[:3]:
                lines.append(f"  - [{failure_item['failure_id']}] {failure_item['title']}")

    lines.extend(["", "## 你的任务"])
    if mode == "constrained":
        lines.extend(list(profile.explore_task_constrained))
    else:
        lines.extend(list(profile.explore_task_free))
    return "\n".join(lines)


def _build_factor_recipe_exploration_prompt(
    *,
    idea: str,
    mode: str,
    cards: list[ExploreIdeaCard],
    constraint_report: dict[str, object],
    project: ProjectConfig | None,
    graph: VaultGraph | None,
) -> str:
    context = _build_factor_recipe_prompt_context(
        cards=cards,
        constraint_report=constraint_report,
        graph=graph,
    )
    if mode == "start":
        return _build_factor_recipe_start_prompt(
            idea=idea,
            project=project,
            context=context,
        )
    if mode == "free":
        return _build_factor_recipe_structured_prompt(
            idea=idea,
            project=project,
            context=context,
        )
    return _build_factor_recipe_constrained_prompt(
        idea=idea,
        project=project,
        context=context,
    )


def _build_factor_recipe_prompt_context(
    *,
    cards: list[ExploreIdeaCard],
    constraint_report: dict[str, object],
    graph: VaultGraph | None,
) -> dict[str, object]:
    primary_family = str(constraint_report.get("primary_family") or "").strip()
    primary_mechanism = str(constraint_report.get("primary_mechanism") or "").strip()
    crowding_warning = str(constraint_report.get("crowding_warning") or "").strip()
    novelty_warnings = _object_to_str_list(constraint_report.get("novelty_warnings"))
    validated_peers = _object_to_str_list(constraint_report.get("validated_peers"))
    frontier_matches = _object_to_dict_list(constraint_report.get("frontier_matches"))
    failure_refs = _object_to_dict_list(constraint_report.get("failure_refs"))
    family_counts = _object_to_dict(constraint_report.get("family_counts"))
    if not primary_family and family_counts:
        primary_family = str(next(iter(family_counts.keys()))).strip()
    if not primary_mechanism:
        primary_mechanism = _first_non_empty(
            str(item.get("mechanism") or "").strip() for item in frontier_matches
        )
    allowed_data_nodes = _collect_explore_allowed_data_nodes(cards=cards, graph=graph)
    if not allowed_data_nodes:
        allowed_data_nodes = _collect_explore_allowed_data_nodes_from_region(
            graph=graph,
            primary_family=primary_family,
            primary_mechanism=primary_mechanism,
        )
    return {
        "cards": cards,
        "primary_family": primary_family,
        "primary_mechanism": primary_mechanism,
        "crowding_warning": crowding_warning,
        "novelty_warnings": novelty_warnings,
        "validated_peers": validated_peers,
        "frontier_matches": frontier_matches,
        "failure_refs": failure_refs,
        "family_counts": family_counts,
        "allowed_data_nodes": allowed_data_nodes,
    }


def _build_factor_recipe_prompt_header(
    *,
    title: str,
    mode_label: str,
    project: ProjectConfig | None,
) -> list[str]:
    lines = [
        title,
        f"> 模式：{mode_label}",
        f"> 生成时间：{_utc_now_iso()}",
    ]
    if project is not None:
        lines.append(f"> 项目：{project.slug} | {project.title_zh}")
    return lines


def _append_factor_recipe_context(
    lines: list[str],
    *,
    context: dict[str, object],
    include_soft_graph: bool = False,
    include_hard_graph: bool = False,
) -> None:
    cards_raw = context.get("cards")
    cards = (
        [item for item in cards_raw if isinstance(item, ExploreIdeaCard)]
        if isinstance(cards_raw, list)
        else []
    )
    failure_refs = _object_to_dict_list(context.get("failure_refs"))
    frontier_matches = _object_to_dict_list(context.get("frontier_matches"))
    family_counts = _object_to_dict(context.get("family_counts"))
    primary_family = str(context.get("primary_family") or "").strip()
    primary_mechanism = str(context.get("primary_mechanism") or "").strip()
    crowding_warning = str(context.get("crowding_warning") or "").strip()
    validated_peers = _object_to_str_list(context.get("validated_peers"))
    allowed_data_nodes = _object_to_str_list(context.get("allowed_data_nodes"))

    lines.extend(["", "## 上下文约束", "", "知识库："])
    if not cards:
        lines.append("- （未命中相关卡片；如果局部知识不足，应主动缩小结论强度，不允许编造背景。）")
    for card in cards[:6]:
        meta: list[str] = []
        if card.type:
            meta.append(f"类型={card.type}")
        if card.lifecycle:
            meta.append(f"状态={card.lifecycle}")
        if card.mechanism:
            meta.append(f"机制={card.mechanism}")
        if card.factor_family:
            meta.append(f"因子族={card.factor_family}")
        meta_text = f" ({'; '.join(meta)})" if meta else ""
        lines.append(f"- {card.name}{meta_text}: {card.summary}")

    lines.extend(["", "历史失败："])
    if failure_refs:
        for item in failure_refs[:3]:
            failure_id = str(item.get("failure_id") or "").strip()
            title = str(item.get("title") or "").strip()
            statement = str(item.get("failure_statement") or "").strip()
            lines.append(f"- [{failure_id}] {title}: {statement}")
    else:
        lines.append("- 未命中直接相关失败案例；这不构成放松约束的理由。")

    lines.extend(["", "研究路径与拥挤度："])
    if primary_family:
        lines.append(f"- 当前最接近的已有因子族：{primary_family}")
    if primary_mechanism:
        lines.append(f"- 当前最接近的已有机制：{primary_mechanism}")
    if crowding_warning:
        lines.append(f"- 拥挤提示：{crowding_warning}")
    if validated_peers:
        lines.append(f"- 已验证同类：{', '.join(validated_peers[:5])}")
    if family_counts:
        distribution = ", ".join(f"{family}({count})" for family, count in family_counts.items())
        lines.append(f"- 知识库分布：{distribution}")
    for frontier_item in frontier_matches[:3]:
        direction = str(frontier_item.get("direction") or "").strip()
        reason = str(frontier_item.get("reason") or "").strip()
        lines.append(f"- frontier: {direction} | {reason}")

    if include_soft_graph:
        lines.extend(["", "可用数据（软约束）："])
        if allowed_data_nodes:
            for data_node in allowed_data_nodes:
                lines.append(f"- {data_node}")
        else:
            lines.append("- 未提供明确数据节点；可提出候选表达，但必须保持不确定性。")
        lines.extend(["", "可用算子（软约束）："])
        for operator_name in _GRAPH_SIGNAL_OPERATORS:
            lines.append(f"- {operator_name}")

    if include_hard_graph:
        lines.extend(
            [
                "",
                "## Graph 约束模式（硬约束）",
                "你只能使用以下数据节点与算子构造信号，不允许引入新变量。",
                "如果某个机制无法仅靠这些节点与算子落地，请直接删除。",
                "",
                "可用数据：",
            ]
        )
        if allowed_data_nodes:
            for data_node in allowed_data_nodes:
                lines.append(f"- {data_node}")
        else:
            lines.append("- graph 未提供可用数据节点；此时不允许虚构新节点，应判定为无法执行。")
        lines.extend(["", "可用算子："])
        for operator_name in _GRAPH_SIGNAL_OPERATORS:
            lines.append(f"- {operator_name}")


def _build_factor_recipe_start_prompt(
    *,
    idea: str,
    project: ProjectConfig | None,
    context: dict[str, object],
) -> str:
    lines = _build_factor_recipe_prompt_header(
        title="# AlphaLab 因子研究起点 Prompt",
        mode_label="Research Kickoff",
        project=project,
    )
    lines.extend(
        [
            "",
            "## 研究主题",
            idea,
            "",
            "## 阶段声明",
            "You are in the research kickoff stage.",
            "Do NOT produce:",
            "- a finalized factor definition",
            "- a complete mathematical formula",
            '- a ranked or selected "best" idea',
            "Your goal is to expand the hypothesis space, not to converge.",
            "你当前处于 hypothesis exploration 阶段，不允许输出最终因子定义或收敛结论。",
            "",
            "## Kickoff 规则",
            "1. 提出 2-3 个有潜力但尚未完全确定的市场机制，重点是值得深入讨论。",
            "2. 每个假设必须在机制层面不同，而不是只改窗口、标准化、缩放或轻微变换。",
            "3. 必须保留不确定性，并主动指出潜在失败路径。",
            "4. 不允许给出最终 ranking、推荐或 single best idea。",
        ]
    )
    _append_factor_recipe_context(lines, context=context, include_soft_graph=True)
    lines.extend(
        [
            "",
            "## 历史路径对抗",
            "If a proposed mechanism resembles previously failed or explored ideas,",
            "you MUST explicitly state:",
            "1) what is similar,",
            "2) what is structurally different,",
            "before continuing.",
            "Failure to differentiate is considered invalid reasoning.",
            "",
            "## 输出要求",
            "[初步机制假设（Mechanism Hypotheses）]",
            "- 提出 2-3 个机制候选，每个都必须包含具体市场参与者行为或结构约束。",
            "",
            "[初步信号思路（Signal Sketch）]",
            "- 可能用到的数据：",
            "- 可能的变换方式：",
            "- 直觉上的预测逻辑：",
            "- 保持可修改空间，不要写完整公式。",
            "",
            "[与已有因子的关系]",
            "- 最接近哪类已有因子：",
            "- 可能的不同点：",
            "",
            "[不确定性与风险点]",
            "- 哪些部分不确定：",
            "- 哪些假设最容易出错：",
            "- 哪些地方需要进一步验证：",
            "",
            "[讨论引导]",
            "1. 这个机制是否真正独立，还是已有因子的变体？",
            "2. 是否存在更直接或更干净的信号表达方式？",
            "3. 是否存在潜在的数据泄露或结构性偏差？",
            "4. 如果要做验证，第一步应该怎么做？",
            "",
            "请输出结构清晰但“未完全收敛”的研究起点，目标是支持后续深入讨论，而不是直接给出最终答案。",
        ]
    )
    return "\n".join(lines)


def _build_factor_recipe_structured_prompt(
    *,
    idea: str,
    project: ProjectConfig | None,
    context: dict[str, object],
) -> str:
    lines = _build_factor_recipe_prompt_header(
        title="# AlphaLab 结构化探索 Prompt",
        mode_label="Structured Exploration",
        project=project,
    )
    lines.extend(
        [
            "",
            "## 研究主题",
            idea,
            "",
            "## 阶段声明",
            "你当前处于 structured exploration 阶段。",
            "目标是把 kickoff 阶段的机制草图做初步形式化、可计算性验证和粗粒度风险识别。",
            "允许写候选表达式，但不允许做最终选择、ranking 或输出 single best idea。",
            "",
            "## 结构化规则",
            "1. 每个候选必须保留为可修改的研究对象，而不是最终因子答案。",
            "2. 必须说明与现有因子的差异，以及 PIT / leakage / 结构偏差风险。",
            "3. 如果两个候选本质上只是参数化变体，只保留一个。",
        ]
    )
    _append_factor_recipe_context(lines, context=context, include_soft_graph=True)
    lines.extend(
        [
            "",
            "## 输出要求",
            "[候选机制]",
            "### 机制 1",
            "- agent behavior:",
            "- structure constraint:",
            "- dynamic process:",
            "",
            "[候选表达]",
            "- 输入数据：",
            "- 候选表达式（可粗略，不要最终公式）：",
            "- 可计算性判断：",
            "",
            "[风险识别]",
            "- PIT 风险：",
            "- 数据泄露风险：",
            "- 结构性偏差：",
            "",
            "[与已有因子的差异]",
            "- 最接近的已有因子类别：",
            "- 相似点：",
            "- 差异点：",
            "",
            "[下一步验证]",
            "- 第一轮应该验证什么：",
            "- 哪些地方最需要继续讨论：",
            "",
            "不要做最终选择，不要 ranking，不要收敛到单一结论。",
        ]
    )
    return "\n".join(lines)


def _build_factor_recipe_constrained_prompt(
    *,
    idea: str,
    project: ProjectConfig | None,
    context: dict[str, object],
) -> str:
    novelty_warnings = _object_to_str_list(context.get("novelty_warnings"))
    lines = _build_factor_recipe_prompt_header(
        title="# AlphaLab 因子假设生成 Prompt",
        mode_label="Constrained Generation",
        project=project,
    )
    lines.extend(
        [
            "",
            "## 约束报告",
            "",
            "## 任务定位",
            (
                "你不是在生成因子、公式、代码或伪代码，而是在做"
                "“强约束下的结构化假设生成（constrained hypothesis generation）”。"
            ),
            "目标是在生成阶段就压缩低质量想法的概率空间，而不是把低质量候选留给后续筛选。",
            "",
            "## 输入主题",
            idea,
            "",
            "## 强制规则",
            (
                "1. 先机制，后信号。任何候选都必须先给出可检验的市场机制，"
                "禁止一上来写因子名、公式、变量组合或算子堆叠。"
            ),
            "2. 机制必须同时包含：具体市场参与者行为或市场结构约束，以及由此产生的动态过程。",
            (
                "3. 禁止使用“市场情绪”“资金博弈”等模糊表述，除非你把它拆成"
                "可操作的行为、约束和可映射数据。"
            ),
            "4. 全程禁止输出具体因子公式、伪代码或参数搜索建议；只能输出结构化假设。",
            "5. 少而精。Step 1 最多提出 5 个候选机制，Step 3 最多保留 2 个最终假设。",
        ]
    )
    _append_factor_recipe_context(lines, context=context, include_hard_graph=True)
    lines.extend(
        [
            "",
            "## 反冗余与 Anti-snooping 约束",
            "1. 避免生成以下内容的简单变体：动量（momentum）、反转（mean reversion）、波动率缩放。",
            (
                "2. 如果只是 horizon、归一化、排序、加权、缩放、去极值或简单组合"
                "发生变化，视为低质量变体，必须删除。"
            ),
            "3. 你必须明确写出：与已有因子的相似点是什么，本质差异又是什么。",
            "",
            "## 多样性约束",
            (
                "1. 最终保留的假设必须来自不同机制类别，"
                "例如：行为驱动、结构约束、信息不对称、流动性冲击。"
            ),
            "2. 如果两个候选机制本质相同，只能保留一个评分更高的版本。",
            "3. 不允许用多个同质候选伪装成“多样性”。",
            "",
            "## 非显然性约束",
            "1. 你必须解释为什么这个想法不是显而易见的。",
            (
                "2. 你必须指出市场上大多数人忽略了什么：是行为偏差、制度约束、"
                "流动性摩擦、披露节奏错配，还是信息处理成本。"
            ),
            "3. 如果无法说明“被忽略之处”，该机制应在 Step 2 被删除。",
            "",
            "## 历史路径约束",
            "1. 你必须判断当前假设属于哪个研究方向。",
            "2. 你必须说明：相比历史尝试，这次推进了什么，而不是重复走过的路径。",
            "3. 如果本质上是重复路径，必须明确说明为什么这次不同；否则应在 Step 2 删除。",
            "",
            "## 三阶段生成流程（必须严格按顺序执行）",
            "Step 1：只生成机制，禁止公式、禁止因子名、禁止变量堆叠。",
            "Step 2：对 Step 1 的机制做带评分的自我筛选，删除低质量机制。",
            "Step 3：仅基于筛选后的机制生成信号构造，仍然禁止公式和伪代码。",
            "Step 4：对最终假设做对抗性检验；如果攻击成立，必须说明回到 Step 2 还是修改假设。",
            "",
            "### Step 1 细则：候选机制池",
            (
                "对每个候选机制，只允许写：agent behavior、structure constraint、"
                "dynamic process、为什么可映射到可观测数据。"
            ),
            "",
            "### Step 2 细则：机制筛选（带评分）",
            "删除任何不满足以下条件的机制：",
            "- 没有明确参与者行为或市场结构约束",
            "- 无法映射到可观测输入数据",
            "- 只是动量 / 反转 / 波动率缩放的换壳",
            "- 与历史失败案例过于相似，却解释不出新的生效条件",
            "- 无法解释为什么它不显然",
            "- 与其他高分候选属于同一机制类别但没有新增信息",
            "对每个机制进行 0-10 评分：可持续性、可交易性、非显然性、与已有因子的差异度。",
            "只保留总评分最高的 1-2 个机制。",
            "",
            "### Step 3 细则：信号构造",
            "对保留机制，信号构造只能写三项：输入数据、变换方式、聚合逻辑。",
            "每一个信号组件都必须回连到机制，不允许出现“有用但解释不清”的装饰性组件。",
            "",
            "### Step 4 细则：对抗性检验（Adversarial Check）",
            "针对最终假设，提出最强反对意见：",
            "- 这个机制可能是错的原因是什么？",
            "- 是否可能是数据挖掘（data snooping）结果？",
            "- 是否可能只是已有因子的变体？",
            "- 在什么市场环境下会彻底失效？",
            "如果反对意见成立，请明确写出：修改假设，还是回到 Step 2 重新选择机制。",
            "",
            "## 输出格式（严格遵守，不要新增栏目）",
            "[Step 1：候选机制]",
            "### 机制 1",
            "- agent behavior:",
            "- structure constraint:",
            "- dynamic process:",
            "- observable implication:",
            "",
            "[Step 2：机制筛选]",
            "### 机制 1",
            "- 可持续性：",
            "- 可交易性：",
            "- 非显然性：",
            "- 差异度：",
            "- 总评分：",
            "- 保留或删除 + 理由：",
            "- 机制类别：",
            "- 与其他保留候选是否重复：",
            "",
            "### 保留结果",
            "- 最终只保留评分最高且机制类别不重复的 1-2 个机制",
            "",
            "[Step 3：最终结构化假设]",
            "### 假设 1",
            "[机制（Mechanism）]",
            "- 行为 + 约束 + 动态过程",
            "[信号构造（Signal Construction）]",
            "- 输入数据：",
            "- 变换方式：",
            "- 聚合逻辑：",
            "[机制→信号映射]",
            "- 每一个信号组件如何体现该机制",
            "[预期表现]",
            "- 收益方向：",
            "- 时间尺度（horizon）：",
            "[失效场景]",
            "- 在什么市场环境下会失效，以及原因",
            "[新颖性说明]",
            "- 最接近的已有因子类别：",
            "- 相似点：",
            "- 本质区别：",
            "[非显然性说明]",
            "- 为什么这个想法不是显而易见的：",
            "- 市场上大多数人忽略了什么：",
            "[历史路径说明]",
            "- 当前假设属于哪个研究方向：",
            "- 相比历史尝试推进了什么：",
            "- 为什么这次不同：",
            "",
            "[Step 4：对抗性检验（Adversarial Check）]",
            "- 最强反对意见：",
            "- 数据挖掘风险：",
            "- 是否只是已有因子的变体：",
            "- 彻底失效的市场环境：",
            "- 结论：保留 / 修改 / 回到 Step 2",
        ]
    )
    if novelty_warnings:
        lines.extend(["", "## 新颖性警示"])
        for item in novelty_warnings:
            lines.append(f"- {item}")
    return "\n".join(lines)


def _collect_explore_allowed_data_nodes(
    *,
    cards: list[ExploreIdeaCard],
    graph: VaultGraph | None,
) -> list[str]:
    if graph is None:
        return []
    data_nodes: list[str] = []
    seen: set[str] = set()
    for card in cards:
        node = graph.get_node(card.name)
        if node is None and card.path:
            node = graph.get_node(Path(card.path).stem.split(" - ", 1)[-1].strip())
        if node is None:
            continue
        for item in graph.get_neighbors(node.name, edge_type="uses_data"):
            normalized = str(item).strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            data_nodes.append(normalized)
    return data_nodes[:12]


def _collect_explore_allowed_data_nodes_from_region(
    *,
    graph: VaultGraph | None,
    primary_family: str,
    primary_mechanism: str,
) -> list[str]:
    if graph is None:
        return []
    candidate_names: list[str] = []
    seen_names: set[str] = set()
    if primary_family and primary_mechanism:
        for name in graph.get_mechanism_family_factors(primary_family, primary_mechanism):
            if name not in seen_names:
                seen_names.add(name)
                candidate_names.append(name)
    if primary_family:
        for name in graph.get_factor_family(primary_family):
            if name not in seen_names:
                seen_names.add(name)
                candidate_names.append(name)
    if primary_mechanism:
        for name in graph.get_by_mechanism(primary_mechanism):
            if name not in seen_names:
                seen_names.add(name)
                candidate_names.append(name)

    data_nodes: list[str] = []
    seen_data: set[str] = set()
    for name in candidate_names:
        for item in graph.get_neighbors(name, edge_type="uses_data"):
            normalized = str(item).strip()
            if not normalized or normalized in seen_data:
                continue
            seen_data.add(normalized)
            data_nodes.append(normalized)
    return data_nodes[:12]


def _append_fast_decision_log(
    path: Path,
    *,
    case_name: str,
    run_key: str,
    verdict_status: str,
    reason: str,
    next_action: str,
) -> None:
    timestamp = dt.datetime.now(dt.UTC).date().isoformat()
    normalized_run_key = _safe_slug(run_key) or "unknown-run"
    marker = f"<!-- run_key:{normalized_run_key} -->"
    block = "\n".join(
        [
            marker,
            f"## {timestamp} - {case_name}",
            f"- run: {normalized_run_key}",
            f"- verdict: {verdict_status or 'pending'}",
            f"- reason: {reason or '待补充'}",
            f"- next: {next_action or '待补充'}",
            "",
        ]
    )
    existing = (
        path.read_text(encoding="utf-8")
        if path.exists()
        else _render_fast_decision_log_header(
            load_project_config(_resolve_project_contract_path(path.parent))
        )
    )
    if marker in existing:
        return
    path.write_text(existing.rstrip() + "\n\n" + block, encoding="utf-8")


def _dedupe_fast_decision_log(path: Path) -> bool:
    if not path.exists():
        return False
    raw = path.read_text(encoding="utf-8")
    if "Fast mode project decisions. One run, one short verdict block." not in raw:
        return False
    entry_pattern = re.compile(
        r"(?ms)(?:^<!-- run_key:[^>]+ -->\n)?^## .+?(?=^(?:<!-- run_key:[^>]+ -->\n)?## |\Z)"
    )
    matches = list(entry_pattern.finditer(raw))
    if not matches:
        return False
    header = raw[: matches[0].start()].strip("\n")
    entries = [m.group(0).strip("\n") for m in matches if m.group(0).strip()]
    if not entries:
        return False

    deduped: list[str] = []
    seen: set[tuple[str, str]] = set()
    for entry in entries:
        marker_match = re.search(r"^<!-- run_key:([^>]+) -->$", entry, flags=re.MULTILINE)
        if marker_match:
            key = ("run_key", marker_match.group(1).strip())
        else:
            normalized = "\n".join(line.rstrip() for line in entry.splitlines())
            key = ("legacy", normalized)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(entry)

    if len(deduped) == len(entries):
        return False

    parts: list[str] = []
    if header:
        parts.append(header)
    parts.append("\n\n".join(deduped))
    normalized_text = "\n\n".join(parts).rstrip() + "\n"
    path.write_text(normalized_text, encoding="utf-8")
    return True


def _append_decision_log(
    path: Path,
    *,
    case_name: str,
    round_id: str,
    verdict_status: str,
    one_sentence_verdict: str,
    run_root: str,
    exported_targets: tuple[str, ...],
) -> None:
    timestamp = dt.datetime.now(dt.UTC).isoformat().replace("+00:00", "Z")
    block = "\n".join(
        [
            f"## {timestamp} - {case_name}",
            "",
            f"- `round_id`: `{round_id}`",
            f"- `verdict_status`: `{verdict_status}`",
            f"- `one_sentence_verdict`: {one_sentence_verdict or '待补充'}",
            f"- `run_root`: `{run_root}`",
            "- `exported_targets`:",
            *[f"  - `{target}`" for target in exported_targets],
            "",
        ]
    )
    existing = (
        path.read_text(encoding="utf-8")
        if path.exists()
        else _render_decision_log_header(
            load_project_config(_resolve_project_contract_path(path.parent))
        )
    )
    path.write_text(existing.rstrip() + "\n\n" + block, encoding="utf-8")


def _load_json_required(path: Path) -> dict[str, Any]:
    payload = _load_json_optional(path)
    if payload is None:
        raise FileNotFoundError(f"required JSON artifact is missing: {path}")
    return payload


def _load_json_optional(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AlphaLabDataError(f"JSON root must be an object: {path}")
    return payload


def _extract_metrics_digest(payload: dict[str, Any] | None) -> dict[str, str]:
    if payload is None:
        return {}
    source = payload
    nested_metrics = payload.get("metrics")
    if isinstance(nested_metrics, dict):
        source = nested_metrics
    keys = [
        "factor_verdict",
        "mean_rank_ic",
        "mean_ic",
        "mean_long_short_return",
        "mean_long_short_turnover",
        "promotion_decision",
        "portfolio_validation_status",
    ]
    digest: dict[str, str] = {}
    for key in keys:
        value = source.get(key)
        if value is None:
            continue
        digest[key] = str(value)
    return digest


def _default_verdict_status(raw_verdict: str | None) -> str:
    verdict = (raw_verdict or "").strip().lower()
    if not verdict:
        return "revise"
    if any(token in verdict for token in ("fail", "reject", "drop", "poor")):
        return "drop"
    if any(token in verdict for token in ("strong", "promising", "pass", "good")):
        return "keep"
    return "revise"


def _truncate_text(text: str, *, max_chars: int) -> str:
    stripped = text.strip()
    if len(stripped) <= max_chars:
        return stripped
    return stripped[: max_chars - 3].rstrip() + "..."


def _utc_now_iso() -> str:
    return dt.datetime.now(dt.UTC).isoformat().replace("+00:00", "Z")


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    title = payload.get("name") if isinstance(payload.get("name"), str) else None
    save_yaml_document(payload, path, title=f"Case - {title}" if title else None)


def _resolve_project_contract_path(project_dir: Path) -> Path:
    markdown = project_dir / "project.md"
    if markdown.exists():
        return markdown
    return project_dir / "project.yaml"
