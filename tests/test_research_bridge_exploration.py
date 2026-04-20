from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from alpha_lab.research_bridge.cli import main as bridge_main
from alpha_lab.research_bridge.embeddings import encode_text
from alpha_lab.research_bridge.service import (
    apply_writeback,
    init_project,
    start_round,
    summarize_run,
)


def _build_phase4_vault(tmp_path: Path) -> Path:
    vault = tmp_path / "quant-knowledge"
    for rel in ["20_methods", "30_factors", "50_experiments", "90_computed", "90_moc"]:
        (vault / rel).mkdir(parents=True, exist_ok=True)

    (vault / "30_factors" / "Factor - Momentum Base.md").write_text(
        "---\n"
        "type: factor\n"
        "name: Momentum Base\n"
        "lifecycle: theoretical\n"
        "mechanism: behavioral\n"
        "factor_family: momentum\n"
        "uses_data:\n"
        "  - close\n"
        "---\n\n"
        "# 动量基类\n",
        encoding="utf-8",
    )
    (vault / "30_factors" / "Factor - Remote Value.md").write_text(
        "---\n"
        "type: factor\n"
        "name: Remote Value\n"
        "lifecycle: theoretical\n"
        "mechanism: risk\n"
        "factor_family: value\n"
        "---\n\n"
        "# 远距离灵感\n",
        encoding="utf-8",
    )
    (vault / "20_methods" / "Method - Momentum Ranking.md").write_text(
        "---\ntype: method\nname: Momentum Ranking\n---\n\n# 排名方法\n",
        encoding="utf-8",
    )
    (vault / "90_moc" / "Registry - Failure Knowledge.md").write_text(
        "---\n"
        "type: moc\n"
        "name: Failure Knowledge Registry\n"
        "status: active\n"
        "---\n\n"
        "# Registry - Failure Knowledge\n\n"
        "## 一、当前生效的 Failure Knowledge\n\n"
        "### [FK-001] 旧失败样例\n\n"
        "- `status`: `active`\n"
        "- `failure_class`: `signal-invalidity`\n"
        "- `failure_statement`:\n"
        "  - 某旧方向无效。\n"
        "- `prevention_rule`:\n"
        "  - 不要重复。\n\n"
        "## 二、Watch 条目\n\n"
        "当前暂无独立 `watch` 条目。\n\n"
        "## 三、Retired 条目\n\n"
        "当前暂无 `retired` 条目。\n",
        encoding="utf-8",
    )
    (vault / "90_computed" / "graph.json").write_text(
        json.dumps(
            {
                "meta": {"node_count": 3, "edge_count": 2},
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
                    "Remote Value": {
                        "type": "factor",
                        "domain": "fundamental",
                        "lifecycle": "theoretical",
                        "market": "a_share",
                        "mechanism": "risk",
                        "factor_family": "value",
                        "path": "30_factors/Factor - Remote Value.md",
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
                    },
                    {
                        "source": "Momentum Base",
                        "target": "close",
                        "type": "uses_data",
                        "target_kind": "data_identifier",
                        "derived": False,
                    },
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
    (vault / "90_computed" / "exploration_map.json").write_text(
        json.dumps(
            {
                "meta": {"built_at": "2026-04-07T00:00:00+00:00"},
                "explored_regions": [],
                "frontier": [
                    {
                        "direction": "risk momentum",
                        "factor_family": "momentum",
                        "mechanism": "risk",
                        "reason": "coverage gap",
                        "suggested_by": "coverage gap analysis",
                        "priority": "high",
                    }
                ],
                "failure_registry_refs": [
                    {
                        "failure_id": "FK-001",
                        "title": "旧失败样例",
                        "status": "active",
                        "failure_class": "signal-invalidity",
                        "failure_statement": "某旧方向无效。",
                        "prevention_rule": "不要重复。",
                    }
                ],
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
            "volume weighted momentum",
        ),
        ("Remote Value", "factor", "30_factors/Factor - Remote Value.md", "cheap deep value"),
        ("Momentum Ranking", "method", "20_methods/Method - Momentum Ranking.md", "rank momentum"),
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


def test_start_round_explore_and_frontier_cli(tmp_path: Path, capsys) -> None:
    vault = _build_phase4_vault(tmp_path)
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
    result = start_round(
        vault_root=vault,
        project_slug="momentum-factor",
        topic="探索新的动量方向",
        round_id="round_001",
        mode="explore",
    )
    text = result.round_context_digest.read_text(encoding="utf-8")
    assert "## Exploration Frontier" in text
    assert "risk momentum" in text
    assert "## Related Failure Knowledge" in text
    assert "FK-001" in text or "未命中与当前主题直接相关的 failure knowledge" in text
    assert "## Controlled Divergence Seeds" in text

    exit_code = bridge_main(["explore-frontier", "--vault-root", str(vault)])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "bridge-explore-frontier" in captured.out
    assert "risk momentum" in captured.out


def test_apply_writeback_updates_phase4_feedback(tmp_path: Path) -> None:
    vault = _build_phase4_vault(tmp_path)
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
    start_round(
        vault_root=vault,
        project_slug="momentum-factor",
        topic="失败回写",
        round_id="round_001",
    )
    run_root = tmp_path / "run_root"
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "run_manifest.json").write_text(
        json.dumps({"case_name": "mom_fail_case"}, indent=2),
        encoding="utf-8",
    )
    (run_root / "metrics.json").write_text(
        json.dumps(
            {
                "factor_verdict": "fails basic robustness",
                "mean_rank_ic": 0.001,
                "promotion_decision": "drop",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_root / "summary.md").write_text("# Summary\n\nfailed\n", encoding="utf-8")
    (run_root / "experiment_card.md").write_text("# Experiment Card\n", encoding="utf-8")
    (run_root / "factor_correlation.json").write_text(
        json.dumps({"top_match": "Remote Value", "max_abs_correlation": 0.82}, indent=2),
        encoding="utf-8",
    )

    summary = summarize_run(
        vault_root=vault,
        project_slug="momentum-factor",
        round_id="round_001",
        run_root=run_root,
    )
    assert summary.graph_feedback["suggested_similar_to"] == ["Remote Value"]
    assert isinstance(summary.graph_feedback["correlation_summary"], str)
    assert "Remote Value" in (summary.graph_feedback["correlation_summary"] or "")
    draft_text = summary.writeback_draft.read_text(encoding="utf-8")
    approved = (
        draft_text.replace("review_status: pending", "review_status: approved")
        .replace("reviewed_by: ''", "reviewed_by: yukun")
        .replace("reviewed_at: ''", "reviewed_at: '2026-04-07T12:00:00Z'")
        .replace(
            "one_sentence_verdict: ''",
            "one_sentence_verdict: Fails basic robustness in current setting.",
        )
    )
    summary.writeback_draft.write_text(approved, encoding="utf-8")

    applied = apply_writeback(
        vault_root=vault,
        project_slug="momentum-factor",
        draft_path=summary.writeback_draft,
    )
    assert applied.graph_feedback.suggested_similar_to == ["Remote Value"]
    assert isinstance(applied.graph_feedback.correlation_summary, str)
    assert "Remote Value" in (applied.graph_feedback.correlation_summary or "")

    origin_text = (vault / "30_factors" / "Factor - Momentum Base.md").read_text(encoding="utf-8")
    assert "tested_in:" in origin_text
    assert "mom_fail_case" in origin_text

    exploration_payload = json.loads(
        (vault / "90_computed" / "exploration_map.json").read_text(encoding="utf-8")
    )
    assert exploration_payload["explored_regions"]
    assert exploration_payload["explored_regions"][0]["best_verdict"]

    failure_registry = (vault / "90_moc" / "Registry - Failure Knowledge.md").read_text(
        encoding="utf-8"
    )
    assert "momentum-factor / mom_fail_case failure pattern" in failure_registry
