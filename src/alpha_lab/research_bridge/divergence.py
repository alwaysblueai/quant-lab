from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from alpha_lab.research_bridge.embeddings import VaultEmbeddings
from alpha_lab.research_bridge.exploration import ExplorationMap
from alpha_lab.research_bridge.graph_view import VaultGraph
from alpha_lab.research_bridge.models import ProjectConfig


@dataclass(frozen=True, slots=True)
class DivergenceSeed:
    name: str
    seed_type: str
    reason: str
    path: str
    score: float


@dataclass(frozen=True, slots=True)
class DivergenceContext:
    graph_guided: list[DivergenceSeed]
    random_walk: list[DivergenceSeed]


def build_divergence_context(
    *,
    project: ProjectConfig,
    vault_root: Path,
    graph: VaultGraph,
    embeddings: VaultEmbeddings | None = None,
    exploration: ExplorationMap | None = None,
    max_graph: int = 4,
    max_random: int = 2,
) -> DivergenceContext:
    seed_names = _project_card_names(project=project, vault_root=vault_root)
    graph_guided = _graph_guided_candidates(
        seed_names=seed_names,
        graph=graph,
        exploration=exploration,
        max_items=max_graph,
    )
    random_walk = _random_walk_candidates(
        seed_names=seed_names,
        embeddings=embeddings,
        graph=graph,
        max_items=max_random,
    )
    return DivergenceContext(graph_guided=graph_guided, random_walk=random_walk)


def _graph_guided_candidates(
    *,
    seed_names: list[str],
    graph: VaultGraph,
    exploration: ExplorationMap | None,
    max_items: int,
) -> list[DivergenceSeed]:
    frontier = exploration.frontier() if exploration is not None else []
    frontier_pairs = {
        (item.factor_family, item.mechanism)
        for item in frontier
        if item.factor_family and item.mechanism
    }
    seeds: list[tuple[float, DivergenceSeed]] = []
    seen: set[str] = set(seed_names)
    for source in seed_names:
        source_node = graph.get_node(source)
        if source_node is None:
            continue
        candidate_names = set(graph.get_neighbors(source))
        candidate_names.update(graph.get_reverse_dependencies(source))
        if source_node.factor_family:
            candidate_names.update(graph.get_factor_family(source_node.factor_family))
        if source_node.mechanism:
            candidate_names.update(graph.get_by_mechanism(source_node.mechanism))
        for name in sorted(candidate_names):
            if name in seen:
                continue
            node = graph.get_node(name)
            if node is None:
                continue
            score = 0.0
            reasons: list[str] = []
            if node.lifecycle == "theoretical":
                score += 2.0
                reasons.append("theoretical node")
            if source_node.mechanism and node.mechanism and node.mechanism != source_node.mechanism:
                score += 1.5
                reasons.append("cross-mechanism jump")
            if (
                source_node.factor_family
                and node.factor_family
                and node.factor_family != source_node.factor_family
            ):
                score += 1.0
                reasons.append("cross-family jump")
            if (node.factor_family, node.mechanism) in frontier_pairs:
                score += 1.5
                reasons.append("touches exploration frontier")
            if score <= 0.0:
                continue
            seen.add(name)
            seeds.append(
                (
                    -score,
                    DivergenceSeed(
                        name=name,
                        seed_type="graph_guided",
                        reason=", ".join(reasons),
                        path=node.path,
                        score=score,
                    ),
                )
            )
    seeds.sort(key=lambda pair: (pair[0], pair[1].name))
    return [seed for _, seed in seeds[:max_items]]


def _random_walk_candidates(
    *,
    seed_names: list[str],
    embeddings: VaultEmbeddings | None,
    graph: VaultGraph,
    max_items: int,
) -> list[DivergenceSeed]:
    if embeddings is None:
        return []
    origin_families = {
        node.factor_family
        for name in seed_names
        if (node := graph.get_node(name)) is not None and node.factor_family
    }
    candidates: list[tuple[float, DivergenceSeed]] = []
    used_families: set[str] = set()
    for entry in embeddings.iter_entries(type_filter="factor"):
        if entry.name in seed_names:
            continue
        node = graph.get_node(entry.name)
        if node is None:
            continue
        if node.factor_family and node.factor_family in used_families:
            continue
        similarity = embeddings.max_similarity_to(entry.name, seed_names)
        if similarity >= 0.3:
            continue
        if node.factor_family and node.factor_family in origin_families:
            continue
        used_families.add(node.factor_family)
        candidates.append(
            (
                similarity,
                DivergenceSeed(
                    name=entry.name,
                    seed_type="random_walk",
                    reason="semantic distance from current origin cards",
                    path=entry.path,
                    score=1.0 - similarity,
                ),
            )
        )
        if len(candidates) >= max_items:
            break
    candidates.sort(key=lambda pair: (pair[0], pair[1].name))
    return [seed for _, seed in candidates[:max_items]]


def _project_card_names(*, project: ProjectConfig, vault_root: Path) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for raw in project.origin_cards + project.supporting_cards:
        path = (vault_root / raw).resolve()
        if not path.exists():
            continue
        name = ""
        text = path.read_text(encoding="utf-8")
        if text.startswith("---\n"):
            end = text.find("\n---", 4)
            if end != -1:
                for line in text[4:end].splitlines():
                    if line.startswith("name:"):
                        name = line.split(":", 1)[1].strip().strip('"')
                        break
        if not name:
            name = path.stem.split(" - ", 1)[-1].strip()
        if name and name not in seen:
            seen.add(name)
            names.append(name)
    return names
