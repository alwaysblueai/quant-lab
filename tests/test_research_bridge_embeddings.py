from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from alpha_lab.research_bridge.cli import main as bridge_main
from alpha_lab.research_bridge.embeddings import VaultEmbeddings, encode_text
from alpha_lab.research_bridge.service import init_project, start_round, structure_candidates


def _build_semantic_vault(tmp_path: Path) -> Path:
    vault = tmp_path / "quant-knowledge"
    for rel in ["20_methods", "30_factors", "90_computed"]:
        (vault / rel).mkdir(parents=True, exist_ok=True)

    (vault / "30_factors" / "Factor - Momentum Base.md").write_text(
        "---\n"
        "type: factor\n"
        "name: Momentum Base\n"
        "mechanism: behavioral\n"
        "factor_family: momentum\n"
        "---\n\n"
        "volume weighted momentum continuation signal\n",
        encoding="utf-8",
    )
    (vault / "30_factors" / "Factor - Value Base.md").write_text(
        "---\n"
        "type: factor\n"
        "name: Value Base\n"
        "mechanism: behavioral\n"
        "factor_family: value\n"
        "---\n\n"
        "cheap balance sheet valuation signal\n",
        encoding="utf-8",
    )
    (vault / "20_methods" / "Method - Momentum Ranking.md").write_text(
        "---\n"
        "type: method\n"
        "name: Momentum Ranking\n"
        "---\n\n"
        "cross sectional ranking for momentum signals\n",
        encoding="utf-8",
    )

    (vault / "90_computed" / "graph.json").write_text(
        json.dumps(
            {
                "meta": {"node_count": 3, "edge_count": 1},
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
                    "Value Base": {
                        "type": "factor",
                        "domain": "fundamental",
                        "lifecycle": "theoretical",
                        "market": "a_share",
                        "mechanism": "behavioral",
                        "factor_family": "value",
                        "path": "30_factors/Factor - Value Base.md",
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

    dimension = 256
    idf = np.ones(dimension, dtype=np.float32)
    docs = [
        (
            "Momentum Base",
            "factor",
            "30_factors/Factor - Momentum Base.md",
            "volume weighted momentum continuation signal",
            "behavioral momentum factor",
        ),
        (
            "Value Base",
            "factor",
            "30_factors/Factor - Value Base.md",
            "cheap balance sheet valuation signal",
            "behavioral value factor",
        ),
        (
            "Momentum Ranking",
            "method",
            "20_methods/Method - Momentum Ranking.md",
            "cross sectional ranking for momentum signals",
            "momentum method",
        ),
    ]
    matrix = np.vstack([encode_text(text, dimension=dimension, idf=idf) for *_, text in docs])
    np.savez_compressed(
        vault / "90_computed" / "embeddings.npz",
        names=np.asarray([item[0] for item in docs], dtype=str),
        types=np.asarray([item[1] for item in docs], dtype=str),
        paths=np.asarray([item[2] for item in docs], dtype=str),
        summaries=np.asarray([item[3] for item in docs], dtype=str),
        matrix=matrix.astype(np.float32),
        idf=idf,
        dimension=np.asarray([dimension], dtype=np.int32),
        model_name=np.asarray(["hash-tfidf-v1"], dtype=str),
        built_at=np.asarray(["2026-04-07T00:00:00+00:00"], dtype=str),
    )
    return vault


def test_vault_embeddings_search_and_similarity(tmp_path: Path) -> None:
    vault = _build_semantic_vault(tmp_path)
    embeddings = VaultEmbeddings.from_vault_root(vault)
    embeddings.build(vault_root=vault)

    results = embeddings.search("momentum continuation volume signal", top_k=2)
    assert results
    assert "Momentum Base" in {result.name for result in results}
    assert embeddings.pairwise_similarity("Momentum Base", "Value Base") < 0.95


def test_structure_candidates_service_and_cli(tmp_path: Path, capsys) -> None:
    vault = _build_semantic_vault(tmp_path)
    init_project(
        vault_root=vault,
        slug="momentum-factor",
        title_zh="动量因子项目",
        category="factor_family",
        owner="yukun",
        market="ashare",
        frequency="daily",
        chatgpt_project_name="Momentum Factor",
        origin_cards=["30_factors/Factor - Momentum Base.md"],
    )
    round_result = start_round(
        vault_root=vault,
        project_slug="momentum-factor",
        topic="结构化候选",
        round_id="round_001",
    )
    round_result.discussion_capture.write_text(
        "# Discussion Capture - round_001\n\n"
        "## 本轮确认的新假设\n"
        "- volume weighted momentum continuation after short skip\n",
        encoding="utf-8",
    )

    service_result = structure_candidates(
        vault_root=vault,
        project_slug="momentum-factor",
        round_id="round_001",
    )
    assert service_result.structured_candidates_path.exists()
    structured_text = service_result.structured_candidates_path.read_text(encoding="utf-8")
    assert "suggested_factor_family`: momentum" in structured_text
    assert "Momentum Base" in structured_text

    exit_code = bridge_main(
        [
            "structure-candidates",
            "--project",
            "momentum-factor",
            "--round",
            "round_001",
            "--vault-root",
            str(vault),
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "bridge-structure-candidates" in captured.out
