from __future__ import annotations

import json
from pathlib import Path

import pytest

from alpha_lab.research_bridge.cli import main as bridge_main
from alpha_lab.research_bridge.graph_view import VaultGraph


def _build_graph_vault(tmp_path: Path) -> Path:
    vault = tmp_path / "quant-knowledge"
    (vault / "90_computed").mkdir(parents=True, exist_ok=True)
    (vault / "00_protocols").mkdir(parents=True, exist_ok=True)
    (vault / "90_computed" / "graph.json").write_text(
        json.dumps(
            {
                "meta": {"node_count": 2, "edge_count": 1},
                "nodes": {
                    "Momentum Base": {
                        "type": "factor",
                        "domain": "price_action",
                        "lifecycle": "theoretical",
                        "market": "a_share",
                        "mechanism": "behavioral",
                        "factor_family": "momentum",
                        "path": "30_factors/Factor - Momentum Base.md",
                    },
                    "Momentum Ranking": {
                        "type": "method",
                        "domain": "price_action",
                        "lifecycle": "theoretical",
                        "market": "a_share",
                        "mechanism": "",
                        "factor_family": "",
                        "path": "20_methods/Method - Momentum Ranking.md",
                    },
                },
                "edges": [
                    {
                        "source": "Momentum Base",
                        "target": "Momentum Ranking",
                        "type": "depends_on",
                        "target_kind": "card",
                        "derived": False,
                    }
                ],
                "diagnostics": {
                    "dangling_edges": [],
                    "orphan_nodes": [],
                    "malformed_fields": [],
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return vault


def test_vault_graph_from_vault_root_and_queries(tmp_path: Path) -> None:
    vault = _build_graph_vault(tmp_path)
    graph = VaultGraph.from_vault_root(vault)
    graph.build(vault_root=vault)

    node = graph.get_node("Momentum Base")
    assert node is not None
    assert node.mechanism == "behavioral"
    assert graph.get_neighbors("Momentum Base", "depends_on") == ["Momentum Ranking"]
    assert graph.coverage_by_type()["factor"] == {
        "annotated": 1,
        "unannotated": 0,
        "total": 1,
    }
    assert graph.mechanism_family_matrix() == {
        "behavioral": {"momentum": ["Momentum Base"]},
    }


def test_vault_graph_rebuild_cache_runs_rebuild_script(tmp_path: Path) -> None:
    vault = tmp_path / "quant-knowledge"
    (vault / "00_protocols").mkdir(parents=True, exist_ok=True)
    (vault / "90_computed").mkdir(parents=True, exist_ok=True)
    (vault / "90_computed" / "graph.json").unlink(missing_ok=True)
    (vault / "00_protocols" / "rebuild-graph.py").write_text(
        "from pathlib import Path\n"
        "import json\n"
        "import sys\n"
        "vault = Path(sys.argv[1])\n"
        "(vault / '90_computed').mkdir(parents=True, exist_ok=True)\n"
        "(vault / '90_computed' / 'graph.json').write_text(json.dumps({"
        "'meta': {'node_count': 0, 'edge_count': 0}, 'nodes': {}, 'edges': [], "
        "'diagnostics': {'dangling_edges': [], 'orphan_nodes': [], 'malformed_fields': []}"
        "}), encoding='utf-8')\n",
        encoding="utf-8",
    )

    graph = VaultGraph.from_vault_root(vault)
    built_path = graph.rebuild_cache(vault_root=vault)
    assert built_path.exists()
    graph.load()
    assert graph.coverage_by_type() == {}


def test_bridge_factor_coverage_command(tmp_path: Path, capsys) -> None:
    vault = _build_graph_vault(tmp_path)
    exit_code = bridge_main(["factor-coverage", "--vault-root", str(vault)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "bridge-factor-coverage" in captured.out
    assert "behavioral" in captured.out
    assert "momentum:1" in captured.out


def test_bridge_explore_frontier_command(tmp_path: Path, capsys) -> None:
    vault = tmp_path / "quant-knowledge"
    protocols = vault / "00_protocols"
    protocols.mkdir(parents=True, exist_ok=True)
    (vault / "90_computed").mkdir(parents=True, exist_ok=True)
    (protocols / "rebuild-exploration-map.py").write_text(
        "from pathlib import Path\n"
        "import json\n"
        "import sys\n"
        "vault = Path(sys.argv[1])\n"
        "(vault / '90_computed').mkdir(parents=True, exist_ok=True)\n"
        "(vault / '90_computed' / 'exploration_map.json').write_text(\n"
        "    json.dumps({'frontier':[{'direction':'test','factor_family':'momentum',"
        "'mechanism':'behavioral','reason':'unit test','suggested_by':'test','priority':'high'}],"
        "'explored_regions':[],'failure_registry_refs':[],'meta':{}}),\n"
        "    encoding='utf-8'\n"
        ")\n"
        "(vault / '90_computed' / 'bridge_cmd_marker.txt').write_text('ok', encoding='utf-8')\n",
        encoding="utf-8",
    )
    exit_code = bridge_main(["explore-frontier", "--vault-root", str(vault)])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "bridge-explore-frontier" in captured.out
    assert "Count    : 1" in captured.out
    assert (vault / "90_computed" / "bridge_cmd_marker.txt").exists()


def test_bridge_explore_frontier_missing_script_raises(tmp_path: Path) -> None:
    vault = tmp_path / "quant-knowledge"
    (vault / "00_protocols").mkdir(parents=True, exist_ok=True)
    with pytest.raises(FileNotFoundError, match="rebuild-exploration-map.py not found"):
        bridge_main(["explore-frontier", "--vault-root", str(vault)])
