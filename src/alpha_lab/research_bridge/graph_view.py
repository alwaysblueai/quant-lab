from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx

VALIDATED_FACTOR_LIFECYCLES = frozenset({"validated", "production", "live", "deployed"})


@dataclass(frozen=True, slots=True)
class NodeAttrs:
    name: str
    type: str
    domain: str
    lifecycle: str
    market: str
    mechanism: str
    factor_family: str
    path: str


@dataclass(frozen=True, slots=True)
class Edge:
    source: str
    target: str
    type: str
    target_kind: str = "card"
    derived: bool = False


@dataclass(frozen=True, slots=True)
class NoveltyReport:
    is_novel: bool
    similar_existing: list[str]
    same_mechanism_family: list[str]
    warnings: list[str]


class VaultGraph:
    def __init__(self, graph_json_path: Path):
        self.graph_json_path = Path(graph_json_path).expanduser().resolve()
        self._graph = nx.DiGraph()
        self._nodes: dict[str, NodeAttrs] = {}
        self._edges: list[Edge] = []
        self._dangling_edges: list[Edge] = []
        self._orphan_nodes: list[str] = []
        self._explicit_edge_sources: set[str] = set()
        self._meta: dict[str, Any] = {}
        self._loaded = False

    @classmethod
    def from_vault_root(cls, vault_root: Path) -> VaultGraph:
        return cls(Path(vault_root).expanduser().resolve() / "90_computed" / "graph.json")

    def build(
        self,
        card_index_tsv: Path | None = None,
        vault_root: Path | None = None,
    ) -> None:
        if not self.graph_json_path.exists():
            self.rebuild_cache(card_index_tsv=card_index_tsv, vault_root=vault_root)
        self.load()

    def rebuild_cache(
        self,
        card_index_tsv: Path | None = None,
        vault_root: Path | None = None,
    ) -> Path:
        resolved_vault = self._resolve_vault_root(
            vault_root=vault_root, card_index_tsv=card_index_tsv
        )
        script_path = resolved_vault / "00_protocols" / "rebuild-graph.py"
        if not script_path.exists():
            raise FileNotFoundError(f"rebuild-graph.py not found: {script_path}")

        proc = subprocess.run(
            [sys.executable, str(script_path), str(resolved_vault)],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "unknown rebuild-graph failure").strip()
            raise RuntimeError(f"failed to rebuild graph cache: {detail}")
        return self.graph_json_path

    def load(self) -> None:
        payload = json.loads(self.graph_json_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"graph.json root must be an object: {self.graph_json_path}")

        raw_nodes = payload.get("nodes")
        raw_edges = payload.get("edges")
        diagnostics = payload.get("diagnostics", {})
        if not isinstance(raw_nodes, dict):
            raise ValueError(f"graph.json missing nodes object: {self.graph_json_path}")
        if not isinstance(raw_edges, list):
            raise ValueError(f"graph.json missing edges list: {self.graph_json_path}")

        graph = nx.DiGraph()
        nodes: dict[str, NodeAttrs] = {}
        for name, raw in raw_nodes.items():
            if not isinstance(name, str) or not isinstance(raw, dict):
                continue
            attrs = NodeAttrs(
                name=name,
                type=str(raw.get("type") or "").strip(),
                domain=str(raw.get("domain") or "").strip(),
                lifecycle=str(raw.get("lifecycle") or "").strip(),
                market=str(raw.get("market") or "").strip(),
                mechanism=str(raw.get("mechanism") or "").strip(),
                factor_family=str(raw.get("factor_family") or "").strip(),
                path=str(raw.get("path") or "").strip(),
            )
            nodes[name] = attrs
            graph.add_node(
                name,
                name=attrs.name,
                type=attrs.type,
                domain=attrs.domain,
                lifecycle=attrs.lifecycle,
                market=attrs.market,
                mechanism=attrs.mechanism,
                factor_family=attrs.factor_family,
                path=attrs.path,
            )

        edges: list[Edge] = []
        explicit_edge_sources: set[str] = set()
        for raw in raw_edges:
            if not isinstance(raw, dict):
                continue
            edge = Edge(
                source=str(raw.get("source") or "").strip(),
                target=str(raw.get("target") or "").strip(),
                type=str(raw.get("type") or "").strip(),
                target_kind=str(raw.get("target_kind") or "card").strip() or "card",
                derived=bool(raw.get("derived", False)),
            )
            if not edge.source or not edge.target or not edge.type:
                continue
            edges.append(edge)
            if not edge.derived:
                explicit_edge_sources.add(edge.source)
            if edge.target_kind == "card" and edge.source in nodes and edge.target in nodes:
                graph.add_edge(
                    edge.source,
                    edge.target,
                    type=edge.type,
                    target_kind=edge.target_kind,
                    derived=edge.derived,
                )

        raw_dangling = diagnostics.get("dangling_edges", [])
        dangling_edges: list[Edge] = []
        if isinstance(raw_dangling, list):
            for raw in raw_dangling:
                if not isinstance(raw, dict):
                    continue
                source = str(raw.get("source") or "").strip()
                target = str(raw.get("target") or "").strip()
                edge_type = str(raw.get("type") or "").strip()
                if not source or not target or not edge_type:
                    continue
                dangling_edges.append(Edge(source=source, target=target, type=edge_type))

        raw_orphans = diagnostics.get("orphan_nodes", [])
        orphan_nodes = (
            sorted(str(item).strip() for item in raw_orphans if str(item).strip())
            if isinstance(raw_orphans, list)
            else []
        )

        self._graph = graph
        self._nodes = nodes
        self._edges = edges
        self._dangling_edges = dangling_edges
        self._orphan_nodes = orphan_nodes
        self._explicit_edge_sources = explicit_edge_sources
        self._meta = dict(payload.get("meta", {})) if isinstance(payload.get("meta"), dict) else {}
        self._loaded = True

    def get_node(self, name: str) -> NodeAttrs | None:
        self._require_loaded()
        return self._nodes.get(name)

    def get_neighbors(self, name: str, edge_type: str | None = None) -> list[str]:
        self._require_loaded()
        neighbors = {
            edge.target
            for edge in self._edges
            if edge.source == name and (edge_type is None or edge.type == edge_type)
        }
        return sorted(neighbors)

    def get_subgraph(self, names: list[str], hops: int = 1) -> nx.DiGraph:
        self._require_loaded()
        selected = {name for name in names if name in self._nodes}
        frontier = set(selected)
        for _ in range(max(hops, 0)):
            expanded = set(frontier)
            for node in frontier:
                expanded.update(self._graph.predecessors(node))
                expanded.update(self._graph.successors(node))
            frontier = expanded
            selected.update(expanded)
        return self._graph.subgraph(selected).copy()

    def get_dependency_chain(self, name: str) -> list[str]:
        self._require_loaded()
        ordered: list[str] = []
        visited: set[str] = set()

        def visit(node: str) -> None:
            for edge in self._edges:
                if edge.source != node or edge.type != "depends_on" or edge.target_kind != "card":
                    continue
                if edge.target not in self._nodes or edge.target in visited:
                    continue
                visited.add(edge.target)
                visit(edge.target)
                ordered.append(edge.target)

        if name in self._nodes:
            visit(name)
        return ordered

    def get_reverse_dependencies(self, name: str) -> list[str]:
        self._require_loaded()
        matches = {
            edge.source
            for edge in self._edges
            if edge.type == "depends_on" and edge.target_kind == "card" and edge.target == name
        }
        return sorted(matches)

    def get_uses_data(self, name: str) -> list[str]:
        """Return the data identifiers a card declares as inputs.

        These are graph edges with ``type=uses_data`` and
        ``target_kind=data_identifier`` — distinct from card-to-card
        relations and not reachable via ``get_neighbors`` without
        filtering by target_kind.
        """
        self._require_loaded()
        matches = {
            edge.target
            for edge in self._edges
            if edge.source == name
            and edge.type == "uses_data"
            and edge.target_kind == "data_identifier"
        }
        return sorted(matches)

    def get_dependency_cards(self, name: str) -> list[str]:
        """Return cards this card depends on (``depends_on`` edges only).

        Restricted to ``target_kind=card`` so callers don't accidentally
        receive data identifiers when ``depends_on`` is over-applied in
        the source vault.
        """
        self._require_loaded()
        matches = {
            edge.target
            for edge in self._edges
            if edge.source == name
            and edge.type == "depends_on"
            and edge.target_kind == "card"
        }
        return sorted(matches)

    def find_similar(self, name: str) -> list[str]:
        self._require_loaded()
        matches = {
            edge.target
            for edge in self._edges
            if edge.source == name
            and edge.type == "similar_to"
            and edge.target_kind == "card"
            and edge.target in self._nodes
        }
        return sorted(matches)

    def get_factor_family(self, family: str) -> list[str]:
        self._require_loaded()
        return sorted(
            name
            for name, attrs in self._nodes.items()
            if attrs.type == "factor" and attrs.factor_family == family
        )

    def get_by_mechanism(self, mechanism: str) -> list[str]:
        self._require_loaded()
        return sorted(name for name, attrs in self._nodes.items() if attrs.mechanism == mechanism)

    def get_mechanism_family_factors(
        self,
        factor_family: str,
        mechanism: str,
        *,
        validated_only: bool = False,
    ) -> list[str]:
        self._require_loaded()
        return sorted(
            name
            for name, attrs in self._nodes.items()
            if attrs.type == "factor"
            and attrs.factor_family == factor_family
            and attrs.mechanism == mechanism
            and (not validated_only or attrs.lifecycle.lower() in VALIDATED_FACTOR_LIFECYCLES)
        )

    def check_novelty(
        self,
        candidate_name: str,
        candidate_similar: list[str],
        candidate_mechanism: str,
        candidate_family: str,
        node_type_filter: str = "factor",
    ) -> NoveltyReport:
        self._require_loaded()
        similar_existing = sorted({item for item in candidate_similar if item in self._nodes})
        same_mechanism_family = sorted(
            name
            for name, attrs in self._nodes.items()
            if attrs.type == node_type_filter
            and name != candidate_name
            and attrs.mechanism == candidate_mechanism
            and attrs.factor_family == candidate_family
        )
        warnings: list[str] = []
        if candidate_name in self._nodes:
            warnings.append("candidate name already exists in graph")
        if similar_existing:
            warnings.append("candidate overlaps with existing cards via similar_to")
        if same_mechanism_family:
            warnings.append("candidate shares mechanism and factor_family with existing factors")
        return NoveltyReport(
            is_novel=not warnings,
            similar_existing=similar_existing,
            same_mechanism_family=same_mechanism_family,
            warnings=warnings,
        )

    def check_dependency_completeness(self, name: str) -> list[str]:
        self._require_loaded()
        missing = [
            edge.target
            for edge in self._dangling_edges
            if edge.source == name and edge.type == "depends_on"
        ]
        return sorted(set(missing))

    def check_mechanism_redundancy(self, name: str) -> list[str]:
        self._require_loaded()
        node = self._nodes.get(name)
        if node is None or node.type != "factor" or not node.mechanism or not node.factor_family:
            return []
        return [
            other_name
            for other_name in self.get_mechanism_family_factors(
                node.factor_family,
                node.mechanism,
            )
            if other_name != name
        ]

    def dangling_edges(self) -> list[Edge]:
        self._require_loaded()
        return list(self._dangling_edges)

    def orphan_nodes(self) -> list[str]:
        self._require_loaded()
        return list(self._orphan_nodes)

    def coverage_by_type(self) -> dict[str, dict[str, int]]:
        self._require_loaded()
        coverage: dict[str, dict[str, int]] = {}
        for name, attrs in self._nodes.items():
            bucket = coverage.setdefault(
                attrs.type or "unknown", {"annotated": 0, "unannotated": 0, "total": 0}
            )
            bucket["total"] += 1
            is_annotated = bool(
                attrs.mechanism or attrs.factor_family or name in self._explicit_edge_sources
            )
            if is_annotated:
                bucket["annotated"] += 1
            else:
                bucket["unannotated"] += 1
        return coverage

    def mechanism_family_matrix(self) -> dict[str, dict[str, list[str]]]:
        self._require_loaded()
        matrix: dict[str, dict[str, list[str]]] = {}
        for name, attrs in sorted(self._nodes.items()):
            if attrs.type != "factor" or not attrs.mechanism or not attrs.factor_family:
                continue
            family_bucket = matrix.setdefault(attrs.mechanism, {})
            factor_names = family_bucket.setdefault(attrs.factor_family, [])
            factor_names.append(name)
        return matrix

    def domain_coverage_matrix(self) -> dict[str, dict[str, int]]:
        """Return ``{domain: {type: count}}`` matrix covering all node types."""
        self._require_loaded()
        matrix: dict[str, dict[str, int]] = {}
        for _name, attrs in self._nodes.items():
            domain = attrs.domain or "unknown"
            ntype = attrs.type or "unknown"
            type_bucket = matrix.setdefault(domain, {})
            type_bucket[ntype] = type_bucket.get(ntype, 0) + 1
        return matrix

    def _require_loaded(self) -> None:
        if not self._loaded:
            raise RuntimeError("VaultGraph.load() must be called before querying the graph")

    def _resolve_vault_root(
        self,
        *,
        vault_root: Path | None,
        card_index_tsv: Path | None,
    ) -> Path:
        if vault_root is not None:
            return Path(vault_root).expanduser().resolve()
        if card_index_tsv is not None:
            card_index_path = Path(card_index_tsv).expanduser().resolve()
            return card_index_path.parent.parent
        graph_path = self.graph_json_path
        if graph_path.name == "graph.json" and graph_path.parent.name == "90_computed":
            return graph_path.parent.parent
        raise ValueError(
            "vault_root is required when graph path is not under 90_computed/graph.json"
        )
