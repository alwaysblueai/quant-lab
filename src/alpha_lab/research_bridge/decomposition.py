from __future__ import annotations

from dataclasses import dataclass

from alpha_lab.research_bridge.graph_view import VaultGraph


@dataclass(frozen=True, slots=True)
class StructuralDecompositionReport:
    candidate_name: str
    same_family: list[str]
    same_mechanism: list[str]
    same_mechanism_family: list[str]
    similar_existing: list[str]
    redundancy_risk: str
    warnings: list[str]


def structural_decomposition_check(
    graph: VaultGraph,
    *,
    candidate_name: str,
    candidate_family: str,
    candidate_mechanism: str,
    candidate_similar: list[str],
) -> StructuralDecompositionReport:
    similar_existing = sorted(
        {item for item in candidate_similar if graph.get_node(item) is not None}
    )
    same_family = sorted(
        name for name in graph.get_factor_family(candidate_family) if name != candidate_name
    )
    same_mechanism = sorted(
        name for name in graph.get_by_mechanism(candidate_mechanism) if name != candidate_name
    )
    same_mechanism_family = sorted(set(same_family) & set(same_mechanism))

    warnings: list[str] = []
    redundancy_risk = "low"
    if same_mechanism_family and similar_existing:
        redundancy_risk = "high"
        warnings.append("candidate overlaps with existing factors in the same mechanism and family")
    elif len(same_mechanism_family) >= 2:
        redundancy_risk = "medium"
        warnings.append("candidate enters a crowded mechanism × family slot")
    elif similar_existing:
        redundancy_risk = "medium"
        warnings.append("candidate is close to existing cards supplied via candidate_similar")

    return StructuralDecompositionReport(
        candidate_name=candidate_name,
        same_family=same_family,
        same_mechanism=same_mechanism,
        same_mechanism_family=same_mechanism_family,
        similar_existing=similar_existing,
        redundancy_risk=redundancy_risk,
        warnings=warnings,
    )
