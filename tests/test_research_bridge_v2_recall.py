from __future__ import annotations

import json
from pathlib import Path

import alpha_lab.research_bridge.model_idea as model_idea_module
from alpha_lab.research_bridge.embeddings import SearchResult
from alpha_lab.research_bridge.llm_rerank import (
    CategorizedCandidate,
    CategorizeOutcome,
    RerankOutcome,
)
from alpha_lab.research_bridge.mechanism_index import CardFingerprint
from alpha_lab.research_bridge.query_expansion import ExpansionOutcome, QueryProbes


class _FakeMechanismEmbeddings:
    def search(
        self,
        query: str,
        top_k: int = 10,
        type_filter: str | None = None,
    ) -> list[SearchResult]:
        del top_k, type_filter
        if "拥挤交易解除" in query or "预期过满" in query:
            return [
                SearchResult(
                    name="Factor - 动量崩盘",
                    score=0.82,
                    type="factor",
                    path="30_factors/Factor - Momentum Crash.md",
                    summary="拥挤交易解除后的非线性回撤。",
                )
            ]
        return []


def _build_vault(tmp_path: Path) -> Path:
    vault = tmp_path / "quant-knowledge"
    for rel in ["30_factors", "90_computed", "90_moc"]:
        (vault / rel).mkdir(parents=True, exist_ok=True)
    (vault / "90_moc" / "CARD-INDEX.tsv").write_text(
        "path\ttype\tname\tdomain\tlifecycle\ttags\tparent_moc\tsummary\n"
        "30_factors/Factor - Value Short.md\tfactor\tFactor - Value Short\tvaluation\t"
        "theoretical\tvaluation\tMOC - Factors\t高估值做空的字面卡。\n"
        "30_factors/Factor - Momentum Crash.md\tfactor\tFactor - 动量崩盘\tprice_action\t"
        "theoretical\tmomentum,crowding\tMOC - Factors\t拥挤交易解除后的非线性回撤。\n",
        encoding="utf-8",
    )
    (vault / "30_factors" / "Factor - Value Short.md").write_text(
        "---\ntype: factor\nname: Factor - Value Short\n"
        "summary: 高估值做空的字面卡。\nmechanism: valuation\n"
        "factor_family: value\n---\n\n高估值做空。\n",
        encoding="utf-8",
    )
    (vault / "30_factors" / "Factor - Momentum Crash.md").write_text(
        "---\ntype: factor\nname: Factor - 动量崩盘\n"
        "summary: 拥挤交易解除后的非线性回撤。\nmechanism: crowding\n"
        "factor_family: momentum\n---\n\n热门动量资产在仓位过满后崩盘。\n",
        encoding="utf-8",
    )
    (vault / "90_computed" / "graph.json").write_text(
        json.dumps(
            {
                "meta": {"node_count": 2, "edge_count": 0},
                "nodes": {
                    "Factor - Value Short": {
                        "type": "factor",
                        "domain": "valuation",
                        "lifecycle": "theoretical",
                        "market": "a_share",
                        "mechanism": "valuation",
                        "factor_family": "value",
                        "path": "30_factors/Factor - Value Short.md",
                    },
                    "Factor - 动量崩盘": {
                        "type": "factor",
                        "domain": "price_action",
                        "lifecycle": "theoretical",
                        "market": "a_share",
                        "mechanism": "crowding",
                        "factor_family": "momentum",
                        "path": "30_factors/Factor - Momentum Crash.md",
                    },
                },
                "edges": [],
                "diagnostics": {"dangling_edges": [], "orphan_nodes": []},
            }
        ),
        encoding="utf-8",
    )
    return vault


def test_v2_pulls_mechanism_card_missed_by_literal(
    tmp_path: Path,
    monkeypatch,
) -> None:
    vault = _build_vault(tmp_path)
    monkeypatch.setenv("ALPHA_LAB_RESEARCH_BRIDGE_V2", "1")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    def fake_search_explore_matches(**kwargs: object) -> list[SearchResult]:
        idea = str(kwargs["idea"])
        if "高估值" in idea:
            return [
                SearchResult(
                    name="Factor - Value Short",
                    score=0.5,
                    type="factor",
                    path="30_factors/Factor - Value Short.md",
                    summary="高估值做空的字面卡。",
                )
            ]
        return []

    def fake_expand_query(**_: object) -> ExpansionOutcome:
        return ExpansionOutcome(
            enabled=True,
            model="claude-sonnet-4-6",
            probes=QueryProbes(
                direct=["做空高估值"],
                mechanism=["拥挤交易解除", "预期过满"],
                analogy=[],
                failure=[],
                construction=[],
            ),
            tokens_input=10,
            tokens_output=5,
            cache_hit_input=0,
            fallback_reason=None,
        )

    def fake_categorize_and_compress(**_: object) -> CategorizeOutcome:
        return CategorizeOutcome(
            enabled=True,
            model="claude-sonnet-4-6",
            categorized=[
                CategorizedCandidate(
                    path="30_factors/Factor - Momentum Crash.md",
                    name="Factor - 动量崩盘",
                    category="transferable",
                    relevance=0.85,
                    transferable_to_idea="拥挤解除可迁移",
                    key_warning="估值继续扩张",
                )
            ],
            insight_brief=["高估值做空需要叠加拥挤和预期过满。"],
            dropped_invalid_paths=[],
            tokens_input=20,
            tokens_output=8,
            cache_hit_input=0,
            fallback_reason=None,
        )

    def fake_fingerprint(**_: object) -> CardFingerprint:
        return CardFingerprint(
            path="30_factors/Factor - Momentum Crash.md",
            name="Factor - 动量崩盘",
            type="factor",
            core_mechanism=["拥挤交易解除", "预期过满", "非线性回撤"],
            transferable_principle="热门资产在预期过满后容易非线性回撤。",
            applicable_scenarios=["截面选股"],
            similar_problems=["高估值做空"],
            failure_conditions=["泡沫延续"],
        )

    monkeypatch.setattr(
        model_idea_module.bridge_loaders,
        "search_explore_matches",
        fake_search_explore_matches,
    )
    monkeypatch.setattr(model_idea_module, "expand_query", fake_expand_query)
    monkeypatch.setattr(
        model_idea_module,
        "categorize_and_compress",
        fake_categorize_and_compress,
    )
    monkeypatch.setattr(
        model_idea_module.mechanism_index,
        "load_mechanism_embeddings",
        lambda **_: _FakeMechanismEmbeddings(),
    )
    monkeypatch.setattr(
        model_idea_module.mechanism_index,
        "load_card_fingerprint",
        fake_fingerprint,
    )

    context = model_idea_module._collect_knowledge_context(
        idea="做空高估值",
        vault_root=vault,
        workspace_root=tmp_path,
        top_k=6,
        mode="start",
    )

    paths = {item["path"] for item in context["knowledge_matches"]}
    assert "30_factors/Factor - Momentum Crash.md" in paths
    diag = context["retrieval_diagnostics"]
    components = diag["score_components_by_name"]["Factor - 动量崩盘"]
    assert components["llm_relevance"] == 0.85
    assert (
        "30_factors/Factor - Momentum Crash.md"
        in diag["mechanism_tier"]["hit_paths"]
    )
    assert "拥挤交易解除" in diag["query_expansion"]["probes"]["mechanism"]
    assert context["insight_brief"] == ["高估值做空需要叠加拥挤和预期过满。"]


def test_v2_fallback_byte_stable_with_v1(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """gpt_prompt must be byte-identical across every v2 fallback branch.

    Three scenarios exercise each rung of the fallback chain — they must
    produce the exact same prompt because the v2 path opted out at a
    different point each time:

    1. flag unset, no key            → _research_bridge_v2_enabled() is False
    2. flag set, key unset           → gate fails on the key check
    3. flag set, key set, expansion fails → gate passes, layer 2 returns enabled=False

    Any drift across these means a v2-only artifact leaked into the
    fallback prompt — the compatibility hard invariant is broken.
    """
    vault = _build_vault(tmp_path)

    def fake_search_explore_matches(**kwargs: object) -> list[SearchResult]:
        idea = str(kwargs["idea"])
        if "高估值" in idea:
            return [
                SearchResult(
                    name="Factor - Value Short",
                    score=0.5,
                    type="factor",
                    path="30_factors/Factor - Value Short.md",
                    summary="高估值做空的字面卡。",
                )
            ]
        return []

    def disabled_rerank(**_: object) -> RerankOutcome:
        return RerankOutcome(
            enabled=False,
            model="claude-sonnet-4-6",
            scores={},
            reasons={},
            tokens_input=0,
            tokens_output=0,
            cache_hit_input=0,
            dropped_invalid_names=[],
            fallback_reason="stubbed",
        )

    def disabled_expansion(**_: object) -> ExpansionOutcome:
        return ExpansionOutcome(
            enabled=False,
            model="claude-sonnet-4-6",
            probes=QueryProbes(
                direct=[],
                mechanism=[],
                analogy=[],
                failure=[],
                construction=[],
            ),
            tokens_input=0,
            tokens_output=0,
            cache_hit_input=0,
            fallback_reason="stubbed",
        )

    monkeypatch.setattr(
        model_idea_module.bridge_loaders,
        "search_explore_matches",
        fake_search_explore_matches,
    )
    monkeypatch.setattr(model_idea_module, "rerank_candidates", disabled_rerank)
    monkeypatch.setattr(model_idea_module, "expand_query", disabled_expansion)

    common_kwargs: dict[str, object] = dict(
        idea="做空高估值",
        mode="explore",
        workspace_root=tmp_path,
        vault_root=vault,
        top_k=4,
        memory_limit=0,
    )

    monkeypatch.delenv("ALPHA_LAB_RESEARCH_BRIDGE_V2", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    flag_off = model_idea_module.explore_model_idea(**common_kwargs)

    monkeypatch.setenv("ALPHA_LAB_RESEARCH_BRIDGE_V2", "1")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    flag_on_no_key = model_idea_module.explore_model_idea(**common_kwargs)

    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    expansion_disabled = model_idea_module.explore_model_idea(**common_kwargs)

    assert flag_off["gpt_prompt"] == flag_on_no_key["gpt_prompt"]
    assert flag_off["gpt_prompt"] == expansion_disabled["gpt_prompt"]
    assert "## Cross-card synthesis" not in flag_off["gpt_prompt"]
