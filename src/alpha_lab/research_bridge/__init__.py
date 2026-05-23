"""``alpha_lab.research_bridge`` — vault retrieval + research-bridge service layer.

The public names listed in ``__all__`` are still importable via
``from alpha_lab.research_bridge import X`` — they're resolved lazily by
``__getattr__`` so we don't pay the (~400ms cold) cost of pulling in
``service`` / ``graph_view`` / ``embeddings`` / ``decomposition`` /
``divergence`` until somebody actually touches one of those names.

Concretely: the CLI's parser-registration path
(``alpha_lab.cli.build_unified_parser`` → ``alpha_lab.research_bridge.cli.build_bridge_parser``)
only needs to touch ``argparse``, so it should not drag pandas + networkx
into ``alpha-lab --help`` cold start.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .cli import build_bridge_parser, main, resolved_argv_for_bridge
    from .decomposition import (
        StructuralDecompositionReport,
        structural_decomposition_check,
    )
    from .divergence import (
        DivergenceContext,
        DivergenceSeed,
        build_divergence_context,
    )
    from .embeddings import SearchResult, VaultEmbeddings, encode_text
    from .exploration import (
        ExplorationMap,
        ExploredRegion,
        FailureKnowledgeRef,
        FrontierEntry,
    )
    from .graph_view import Edge, NodeAttrs, NoveltyReport, VaultGraph
    from .models import AlphaLabDefaults, ProjectConfig, ProjectStatus, WritebackPolicy
    from .output_lint import (
        LintReport,
        LintViolation,
        describe_lint_contract,
        extract_stage_sections,
        lint_explore_response,
    )
    from .preflight import (
        PreflightIssue,
        PreflightReport,
        render_preflight_report,
        run_preflight,
    )
    from .service import (
        ExploreIdeaCard,
        ExploreIdeaResult,
        StructureCandidatesResult,
        StructuredCandidate,
        apply_writeback,
        explore_idea,
        init_project,
        normalize_fast_decision_log,
        refresh_project_pack,
        scaffold_case,
        start_round,
        structure_candidates,
        summarize_run,
    )
    from .sessions import record_explore_response


__all__ = [
    "AlphaLabDefaults",
    "DivergenceContext",
    "DivergenceSeed",
    "Edge",
    "ExploreIdeaCard",
    "ExploreIdeaResult",
    "ExplorationMap",
    "ExploredRegion",
    "FailureKnowledgeRef",
    "FrontierEntry",
    "LintReport",
    "LintViolation",
    "NodeAttrs",
    "NoveltyReport",
    "PreflightIssue",
    "PreflightReport",
    "ProjectConfig",
    "ProjectStatus",
    "SearchResult",
    "StructuralDecompositionReport",
    "StructureCandidatesResult",
    "StructuredCandidate",
    "VaultEmbeddings",
    "VaultGraph",
    "WritebackPolicy",
    "apply_writeback",
    "build_bridge_parser",
    "build_divergence_context",
    "describe_lint_contract",
    "encode_text",
    "explore_idea",
    "extract_stage_sections",
    "init_project",
    "lint_explore_response",
    "main",
    "normalize_fast_decision_log",
    "record_explore_response",
    "refresh_project_pack",
    "render_preflight_report",
    "resolved_argv_for_bridge",
    "run_preflight",
    "scaffold_case",
    "start_round",
    "structural_decomposition_check",
    "structure_candidates",
    "summarize_run",
]


# Maps each public name to ``(submodule_dotted_path, attribute_name)``.
_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    # cli
    "build_bridge_parser": (".cli", "build_bridge_parser"),
    "main": (".cli", "main"),
    "resolved_argv_for_bridge": (".cli", "resolved_argv_for_bridge"),
    # decomposition
    "StructuralDecompositionReport": (".decomposition", "StructuralDecompositionReport"),
    "structural_decomposition_check": (".decomposition", "structural_decomposition_check"),
    # divergence
    "DivergenceContext": (".divergence", "DivergenceContext"),
    "DivergenceSeed": (".divergence", "DivergenceSeed"),
    "build_divergence_context": (".divergence", "build_divergence_context"),
    # embeddings
    "SearchResult": (".embeddings", "SearchResult"),
    "VaultEmbeddings": (".embeddings", "VaultEmbeddings"),
    "encode_text": (".embeddings", "encode_text"),
    # exploration
    "ExplorationMap": (".exploration", "ExplorationMap"),
    "ExploredRegion": (".exploration", "ExploredRegion"),
    "FailureKnowledgeRef": (".exploration", "FailureKnowledgeRef"),
    "FrontierEntry": (".exploration", "FrontierEntry"),
    # graph_view
    "Edge": (".graph_view", "Edge"),
    "NodeAttrs": (".graph_view", "NodeAttrs"),
    "NoveltyReport": (".graph_view", "NoveltyReport"),
    "VaultGraph": (".graph_view", "VaultGraph"),
    # models
    "AlphaLabDefaults": (".models", "AlphaLabDefaults"),
    "ProjectConfig": (".models", "ProjectConfig"),
    "ProjectStatus": (".models", "ProjectStatus"),
    "WritebackPolicy": (".models", "WritebackPolicy"),
    # output_lint
    "LintReport": (".output_lint", "LintReport"),
    "LintViolation": (".output_lint", "LintViolation"),
    "describe_lint_contract": (".output_lint", "describe_lint_contract"),
    "extract_stage_sections": (".output_lint", "extract_stage_sections"),
    "lint_explore_response": (".output_lint", "lint_explore_response"),
    # preflight
    "PreflightIssue": (".preflight", "PreflightIssue"),
    "PreflightReport": (".preflight", "PreflightReport"),
    "render_preflight_report": (".preflight", "render_preflight_report"),
    "run_preflight": (".preflight", "run_preflight"),
    # service
    "ExploreIdeaCard": (".service", "ExploreIdeaCard"),
    "ExploreIdeaResult": (".service", "ExploreIdeaResult"),
    "StructureCandidatesResult": (".service", "StructureCandidatesResult"),
    "StructuredCandidate": (".service", "StructuredCandidate"),
    "apply_writeback": (".service", "apply_writeback"),
    "explore_idea": (".service", "explore_idea"),
    "init_project": (".service", "init_project"),
    "normalize_fast_decision_log": (".service", "normalize_fast_decision_log"),
    "refresh_project_pack": (".service", "refresh_project_pack"),
    "scaffold_case": (".service", "scaffold_case"),
    "start_round": (".service", "start_round"),
    "structure_candidates": (".service", "structure_candidates"),
    "summarize_run": (".service", "summarize_run"),
    # sessions
    "record_explore_response": (".sessions", "record_explore_response"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(
            f"module 'alpha_lab.research_bridge' has no attribute {name!r}"
        )
    from importlib import import_module

    submodule_path, attr = target
    return getattr(import_module(submodule_path, package=__name__), attr)
