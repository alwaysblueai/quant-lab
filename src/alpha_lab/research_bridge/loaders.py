from __future__ import annotations

import re
from csv import DictReader
from pathlib import Path

from alpha_lab.research_bridge.embeddings import SearchResult, VaultEmbeddings
from alpha_lab.research_bridge.exploration import ExplorationMap
from alpha_lab.research_bridge.graph_view import VaultGraph


def load_vault_graph(vault_root: Path) -> VaultGraph | None:
    graph_path = (vault_root / "90_computed" / "graph.json").resolve()
    if not graph_path.exists():
        return None
    graph = VaultGraph(graph_path)
    try:
        graph.load()
    except (OSError, ValueError):
        return None
    return graph


def load_vault_embeddings(vault_root: Path) -> VaultEmbeddings | None:
    embeddings_path = (vault_root / "90_computed" / "embeddings.npz").resolve()
    if not embeddings_path.exists():
        return None
    embeddings = VaultEmbeddings(embeddings_path)
    try:
        embeddings.load()
    except (OSError, ValueError):
        return None
    return embeddings


def load_exploration_map(vault_root: Path) -> ExplorationMap | None:
    path = (vault_root / "90_computed" / "exploration_map.json").resolve()
    if not path.exists():
        return None
    exploration = ExplorationMap(path)
    try:
        exploration.load()
    except (OSError, ValueError):
        return None
    return exploration


def load_explore_graph(vault_root: Path) -> VaultGraph | None:
    graph = load_vault_graph(vault_root)
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


def load_explore_embeddings(vault_root: Path) -> VaultEmbeddings | None:
    embeddings = load_vault_embeddings(vault_root)
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


def load_explore_exploration_map(vault_root: Path) -> ExplorationMap | None:
    exploration = load_exploration_map(vault_root)
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


def load_card_index_rows(vault_root: Path) -> list[dict[str, str]]:
    index_path = vault_root / "90_moc" / "CARD-INDEX.tsv"
    if not index_path.exists():
        return []
    rows: list[dict[str, str]] = []
    with index_path.open("r", encoding="utf-8") as fh:
        reader = DictReader(fh, delimiter="\t")
        for row in reader:
            rows.append({key: str(value or "").strip() for key, value in row.items()})
    return rows


def search_explore_matches(
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

