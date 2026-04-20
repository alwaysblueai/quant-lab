from .cli import build_bridge_parser, main, resolved_argv_for_bridge
from .decomposition import StructuralDecompositionReport, structural_decomposition_check
from .divergence import DivergenceContext, DivergenceSeed, build_divergence_context
from .embeddings import SearchResult, VaultEmbeddings, encode_text
from .exploration import ExplorationMap, ExploredRegion, FailureKnowledgeRef, FrontierEntry
from .graph_view import Edge, NodeAttrs, NoveltyReport, VaultGraph
from .models import AlphaLabDefaults, ProjectConfig, ProjectStatus, WritebackPolicy
from .preflight import PreflightIssue, PreflightReport, render_preflight_report, run_preflight
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

__all__ = [
    # Data classes / types
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
    # Service-layer functions
    "apply_writeback",
    "explore_idea",
    "init_project",
    "normalize_fast_decision_log",
    "refresh_project_pack",
    "scaffold_case",
    "start_round",
    "structure_candidates",
    "summarize_run",
    # Utility functions
    "build_bridge_parser",
    "build_divergence_context",
    "encode_text",
    "main",
    "render_preflight_report",
    "resolved_argv_for_bridge",
    "run_preflight",
    "structural_decomposition_check",
]
