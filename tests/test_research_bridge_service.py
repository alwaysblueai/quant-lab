from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from alpha_lab.research_bridge.models import load_project_config
from alpha_lab.research_bridge.preflight import run_preflight
from alpha_lab.research_bridge.service import (
    _build_factor_recipe_signal_mapping_prompt,
    _build_factor_recipe_validation_kill_tests_prompt,
    apply_writeback,
    explore_idea,
    init_project,
    normalize_fast_decision_log,
    scaffold_case,
    start_round,
    summarize_run,
)


def _build_vault(tmp_path: Path) -> Path:
    vault = tmp_path / "quant-knowledge"
    for rel in [
        "10_concepts",
        "20_methods",
        "30_factors",
        "50_experiments",
        "90_computed",
        "90_moc",
    ]:
        (vault / rel).mkdir(parents=True, exist_ok=True)
    (vault / "90_moc" / "CARD-INDEX.tsv").write_text(
        "path\ttype\tname\tdomain\tlifecycle\ttags\tparent_moc\tsummary\n"
        "30_factors/Factor - Momentum Base.md\tfactor\tMomentum Base\tprice_action\t"
        "theoretical\tmomentum,behavioral\tMOC - Factors\t动量基类，用于测试。\n"
        "20_methods/Method - Momentum Ranking.md\tmethod\tMomentum Ranking\tprice_action\t"
        "theoretical\tranking,momentum\tMOC - Methods\t动量排序方法。\n",
        encoding="utf-8",
    )
    (vault / "30_factors" / "Factor - Momentum Base.md").write_text(
        "---\n"
        "type: factor\n"
        "name: Momentum Base\n"
        "summary: 动量基类，用于测试。\n"
        "mechanism: behavioral\n"
        "factor_family: momentum\n"
        "---\n\n"
        "# 动量基类\n\n这是一个用于测试的 origin card。\n",
        encoding="utf-8",
    )
    (vault / "20_methods" / "Method - Momentum Ranking.md").write_text(
        "---\ntype: method\nname: Momentum Ranking\n---\n\n# 排名方法\n",
        encoding="utf-8",
    )
    (vault / "50_experiments" / "Exp - 202604 - Momentum History.md").write_text(
        "# 历史实验\n\n此前结果一般。\n",
        encoding="utf-8",
    )
    (vault / "90_computed" / "graph.json").write_text(
        json.dumps(
            {
                "meta": {"node_count": 5, "edge_count": 1},
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
                    "Momentum Veteran 1": {
                        "type": "factor",
                        "domain": "price_action",
                        "lifecycle": "validated",
                        "market": "a_share",
                        "mechanism": "behavioral",
                        "factor_family": "momentum",
                        "path": "30_factors/Factor - Momentum Veteran 1.md",
                    },
                    "Momentum Veteran 2": {
                        "type": "factor",
                        "domain": "price_action",
                        "lifecycle": "validated",
                        "market": "a_share",
                        "mechanism": "behavioral",
                        "factor_family": "momentum",
                        "path": "30_factors/Factor - Momentum Veteran 2.md",
                    },
                    "Momentum Veteran 3": {
                        "type": "factor",
                        "domain": "price_action",
                        "lifecycle": "validated",
                        "market": "a_share",
                        "mechanism": "behavioral",
                        "factor_family": "momentum",
                        "path": "30_factors/Factor - Momentum Veteran 3.md",
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
    (vault / "90_computed" / "exploration_map.json").write_text(
        json.dumps(
            {
                "meta": {"frontier_count": 1},
                "explored_regions": [],
                "frontier": [
                    {
                        "direction": "behavioral momentum",
                        "factor_family": "momentum",
                        "mechanism": "behavioral",
                        "reason": "coverage gap test fixture",
                        "suggested_by": "test",
                        "priority": "high",
                    }
                ],
                "failure_registry_refs": [
                    {
                        "failure_id": "FK-001",
                        "title": "Momentum crowding",
                        "status": "watch",
                        "failure_class": "crowding",
                        "failure_statement": "behavioral momentum can crowd",
                        "prevention_rule": "check crowding before promotion",
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return vault


def test_init_project_creates_project_pack(tmp_path: Path) -> None:
    vault = _build_vault(tmp_path)

    result = init_project(
        vault_root=vault,
        slug="momentum-factor",
        title_zh="动量因子项目",
        category="factor_family",
        owner="yukun",
        market="ashare",
        frequency="daily",
        chatgpt_project_name="Momentum Factor",
        origin_cards=["30_factors/Factor - Momentum Base.md"],
        related_experiment_cards=["50_experiments/Exp - 202604 - Momentum History.md"],
    )

    project_dir = result.paths.project_dir
    assert project_dir.exists()
    assert (project_dir / "project.md").exists()
    assert (project_dir / "current_case.md").exists()
    assert (project_dir / "decision_log.md").exists()
    assert (project_dir / "runs" / "latest.md").exists()
    assert not (project_dir / "01_project_brief.md").exists()
    assert not (project_dir / "10_active_state.md").exists()
    assert not (project_dir / "20_decision_log.md").exists()

    project = load_project_config(project_dir / "project.md")
    assert project.slug == "momentum-factor"
    assert project.alpha_lab_defaults.slice_preset == "standard"
    assert project.alpha_lab_defaults.evaluation_profile == "exploratory_screening"
    project_text = (project_dir / "project.md").read_text(encoding="utf-8")
    assert "```yaml" in project_text
    current_case_text = (project_dir / "current_case.md").read_text(encoding="utf-8")
    assert "project_slug: momentum-factor" in current_case_text
    assert "evaluation_profile: exploratory_screening" in current_case_text
    assert "```yaml" in current_case_text
    latest_text = (project_dir / "runs" / "latest.md").read_text(encoding="utf-8")
    assert "pending_case" in latest_text


def test_explore_idea_returns_constrained_graph_context(tmp_path: Path) -> None:
    vault = _build_vault(tmp_path)

    result = explore_idea(
        vault_root=vault,
        idea="动量反转",
        mode="constrained",
    )

    payload = result.to_payload()
    assert payload["mode"] == "constrained"
    assert payload["related_cards"]
    assert "约束报告" in payload["gpt_prompt"]
    constraint_report = payload["constraint_report"]
    assert constraint_report["primary_family"] == "momentum"
    assert "validated_peers" in constraint_report
    assert "frontier_matches" in constraint_report
    assert "failure_refs" in constraint_report


def test_round_case_summary_and_writeback_flow(tmp_path: Path) -> None:
    vault = _build_vault(tmp_path)
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
        mode="legacy",
    )

    round_result = start_round(
        vault_root=vault,
        project_slug="momentum-factor",
        topic="三个月成交额加权动量",
        round_id="round_001",
    )
    assert round_result.round_context_digest.exists()
    assert round_result.round_prompt.exists()
    assert round_result.web_search_tasks.exists()
    assert round_result.discussion_capture.exists()
    round_context = round_result.round_context_digest.read_text(encoding="utf-8")
    assert "## 图谱相关上下文" in round_context
    assert "`depends_on`: `Momentum Ranking`" in round_context
    round_result.discussion_capture.write_text(
        "# Discussion Capture - round_001\n\n## 本轮确认的新假设\n- 使用成交额加权的 60 日动量。\n",
        encoding="utf-8",
    )

    scaffold = scaffold_case(
        vault_root=vault,
        project_slug="momentum-factor",
        round_id="round_001",
        case_name="mom_amt_60",
        factor_name="mom_amt_60",
        base_method="momentum",
        lookback=60,
        skip_recent=5,
        target_horizon=5,
    )
    assert scaffold.handoff_path.exists()
    assert scaffold.spec_path.exists()
    assert scaffold.current_case_path.exists()
    spec_text = scaffold.spec_path.read_text(encoding="utf-8")
    assert "mom_amt_60" in spec_text
    assert "dist/bridge_runs/momentum-factor/mom_amt_60" in spec_text
    spec_payload = yaml.safe_load(spec_text)
    assert spec_payload["factor_input"]["recipe"]["base"]["window"] == 60
    assert "lookback" not in spec_payload["factor_input"]["recipe"]["base"]
    current_case_text = scaffold.current_case_path.read_text(encoding="utf-8")
    assert "name: mom_amt_60" in current_case_text

    run_root = tmp_path / "run_root"
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "run_manifest.json").write_text(
        json.dumps({"case_name": "mom_amt_60"}, indent=2),
        encoding="utf-8",
    )
    (run_root / "metrics.json").write_text(
        json.dumps(
            {
                "metrics": {
                    "factor_verdict": "promising",
                    "mean_rank_ic": 0.041,
                    "mean_ic": 0.032,
                    "mean_long_short_return": 0.006,
                    "mean_long_short_turnover": 0.19,
                    "promotion_decision": "promote_with_review",
                }
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_root / "summary.md").write_text("# Summary\n\nrun summary\n", encoding="utf-8")
    (run_root / "experiment_card.md").write_text(
        "# Experiment Card\n\nmachine owned sections\n",
        encoding="utf-8",
    )
    (run_root / "factor_correlation.json").write_text(
        json.dumps(
            {
                "top_match": "Momentum Base",
                "max_abs_correlation": 0.82,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    summarized = summarize_run(
        vault_root=vault,
        project_slug="momentum-factor",
        round_id="round_001",
        run_root=run_root,
    )
    assert summarized.summary_path.exists()
    assert summarized.latest_path.exists()
    assert summarized.decision_log_path.exists()
    assert summarized.latest_experiment_feedback.exists()
    feedback_text = summarized.latest_experiment_feedback.read_text(encoding="utf-8")
    assert "promising" in feedback_text
    assert "0.041" in feedback_text
    draft_text = summarized.writeback_draft.read_text(encoding="utf-8")
    assert "review_status: pending" in draft_text
    assert "`factor_verdict`: promising" in draft_text
    assert "## Graph Feedback" in draft_text
    assert "Momentum Base" in draft_text
    assert summarized.graph_feedback["suggested_similar_to"] == ["Momentum Base"]
    assert isinstance(summarized.graph_feedback["correlation_summary"], str)
    assert "Momentum Base" in (summarized.graph_feedback["correlation_summary"] or "")

    with pytest.raises(ValueError, match="has not been approved"):
        apply_writeback(
            vault_root=vault,
            project_slug="momentum-factor",
            draft_path=summarized.writeback_draft,
        )

    approved_text = (
        draft_text.replace("review_status: pending", "review_status: approved")
        .replace("reviewed_by: ''", "reviewed_by: yukun")
        .replace("reviewed_at: ''", "reviewed_at: '2026-04-03T12:00:00Z'")
        .replace("one_sentence_verdict: ''", "one_sentence_verdict: 保留并继续迭代。")
        .replace("current_focus: 待开始第一轮讨论", "current_focus: 聚焦成交额加权动量稳健性。")
        .replace(
            "next_action: 刷新项目包并启动第一轮讨论",
            "next_action: 跑更长窗口并检查指数成分约束。",
        )
    )
    summarized.writeback_draft.write_text(approved_text, encoding="utf-8")

    applied = apply_writeback(
        vault_root=vault,
        project_slug="momentum-factor",
        draft_path=summarized.writeback_draft,
    )

    exported_latest = vault / "50_experiments" / "mom_amt_60" / "latest.md"
    assert exported_latest.exists()
    assert applied.export_result.success is True
    project = load_project_config(vault / "55_projects" / "momentum-factor" / "project.md")
    assert project.status.current_focus == "聚焦成交额加权动量稳健性。"
    assert project.status.next_action == "跑更长窗口并检查指数成分约束。"
    assert "mom_amt_60" in (
        vault / "55_projects" / "momentum-factor" / "20_decision_log.md"
    ).read_text(encoding="utf-8")
    assert "review_status: applied" in summarized.writeback_draft.read_text(encoding="utf-8")
    assert applied.graph_feedback.suggested_similar_to == ["Momentum Base"]
    assert isinstance(applied.graph_feedback.correlation_summary, str)
    assert "Momentum Base" in (applied.graph_feedback.correlation_summary or "")


def test_summarize_run_is_idempotent_for_same_run_root(tmp_path: Path) -> None:
    vault = _build_vault(tmp_path)
    init_project(
        vault_root=vault,
        slug="momentum-factor",
        title_zh="动量因子项目",
        category="factor_family",
        owner="yukun",
        market="ashare",
        frequency="daily",
        chatgpt_project_name="Momentum Factor",
    )
    run_root = tmp_path / "run_root"
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "run_manifest.json").write_text(
        json.dumps({"case_name": "mom_amt_60"}, indent=2),
        encoding="utf-8",
    )
    (run_root / "metrics.json").write_text(
        json.dumps(
            {
                "metrics": {
                    "factor_verdict": "promising",
                    "mean_rank_ic": 0.041,
                }
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_root / "summary.md").write_text("# Summary\n\nrun summary\n", encoding="utf-8")

    summarize_run(
        vault_root=vault,
        project_slug="momentum-factor",
        run_root=run_root,
    )
    summarize_run(
        vault_root=vault,
        project_slug="momentum-factor",
        run_root=run_root,
    )

    decision_log = (vault / "55_projects" / "momentum-factor" / "decision_log.md").read_text(
        encoding="utf-8"
    )
    assert decision_log.count("## ") == 1
    assert decision_log.count("- run: run_root") == 1
    assert decision_log.count("<!-- run_key:run_root -->") == 1


def test_normalize_fast_decision_log_dedupes_legacy_duplicates(tmp_path: Path) -> None:
    vault = _build_vault(tmp_path)
    init_project(
        vault_root=vault,
        slug="momentum-factor",
        title_zh="动量因子项目",
        category="factor_family",
        owner="yukun",
        market="ashare",
        frequency="daily",
        chatgpt_project_name="Momentum Factor",
    )
    decision_log = vault / "55_projects" / "momentum-factor" / "decision_log.md"
    duplicate_block = "\n".join(
        [
            "## 2026-04-12 - mom_amt_60",
            "- verdict: drop",
            "- reason: verdict=drop; factor_verdict=fails basic robustness",
            "- next: 刷新项目包并启动第一轮讨论",
            "",
        ]
    )
    decision_log.write_text(
        "\n".join(
            [
                "# Decision Log - 动量因子项目",
                "",
                "Fast mode project decisions. One run, one short verdict block.",
                "",
                duplicate_block,
                duplicate_block,
                duplicate_block,
            ]
        ),
        encoding="utf-8",
    )

    normalize_fast_decision_log(vault_root=vault, project_slug="momentum-factor")

    cleaned = decision_log.read_text(encoding="utf-8")
    assert cleaned.count("## 2026-04-12 - mom_amt_60") == 1


def test_scaffold_case_preflight_writes_report(tmp_path: Path) -> None:
    vault = _build_vault(tmp_path)
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
        mode="legacy",
    )
    start_round(
        vault_root=vault,
        project_slug="momentum-factor",
        topic="预检动量因子",
        round_id="round_001",
    )

    scaffold = scaffold_case(
        vault_root=vault,
        project_slug="momentum-factor",
        round_id="round_001",
        case_name="mom_amt_60",
        factor_name="mom_amt_60",
        base_method="momentum",
        lookback=60,
        skip_recent=5,
        target_horizon=5,
        preflight=True,
        candidate_name="Momentum Variant Candidate",
        candidate_family="momentum",
        candidate_mechanism="behavioral",
        candidate_similar=["Momentum Base"],
        candidate_decay_class="fast",
        candidate_capacity_class="constrained",
    )

    assert scaffold.preflight_path is not None
    assert scaffold.preflight_path.exists()
    report_text = scaffold.preflight_path.read_text(encoding="utf-8")
    assert "## Preflight Checks" in report_text
    assert "novelty_warning" in report_text
    assert "crowded_mechanism" in report_text
    assert "capacity_decay_warning" in report_text
    handoff_text = scaffold.handoff_path.read_text(encoding="utf-8")
    assert "## Preflight Checks" in handoff_text


def test_scaffold_case_preflight_blocks_existing_candidate_name(tmp_path: Path) -> None:
    vault = _build_vault(tmp_path)
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
        mode="legacy",
    )
    start_round(
        vault_root=vault,
        project_slug="momentum-factor",
        topic="预检动量因子",
        round_id="round_001",
    )

    with pytest.raises(
        ValueError, match="preflight blocked scaffold-case|candidate name already exists"
    ):
        scaffold_case(
            vault_root=vault,
            project_slug="momentum-factor",
            round_id="round_001",
            case_name="mom_amt_60",
            factor_name="mom_amt_60",
            base_method="momentum",
            lookback=60,
            skip_recent=5,
            target_horizon=5,
            preflight=True,
            candidate_name="Momentum Base",
            candidate_family="momentum",
            candidate_mechanism="behavioral",
        )


def test_run_preflight_blocks_known_non_pit_inputs(tmp_path: Path) -> None:
    vault = _build_vault(tmp_path)
    report = run_preflight(
        vault_root=vault,
        checked_card_paths=["30_factors/Factor - Momentum Base.md"],
        candidate_name="Restated Momentum Candidate",
        candidate_family="momentum",
        candidate_mechanism="behavioral",
        candidate_uses_data=["restated_fundamentals"],
        candidate_pit_sensitivity="high",
    )

    assert report.is_blocked is True
    assert any(issue.code == "pit_non_pit_block" for issue in report.issues)


# ---------------------------------------------------------------------------
# Typed hybrid retrieval (P0) + dependency hard filter (P1)
# ---------------------------------------------------------------------------


def _build_vault_with_uses_data(tmp_path: Path) -> Path:
    """Vault fixture with explicit ``uses_data`` edges, used for P1 tests."""
    vault = tmp_path / "quant-knowledge-uses-data"
    for rel in ["10_concepts", "30_factors", "90_computed", "90_moc"]:
        (vault / rel).mkdir(parents=True, exist_ok=True)
    (vault / "90_moc" / "CARD-INDEX.tsv").write_text(
        "path\ttype\tname\tdomain\tlifecycle\ttags\tparent_moc\tsummary\n"
        "30_factors/Factor - Daily PV.md\tfactor\tDaily PV\tprice_action\t"
        "validated\tdaily,price\tMOC - Factors\t日频价量动量\n"
        "30_factors/Factor - Intraday Volume Burst.md\tfactor\tIntraday Volume Burst\t"
        "price_action\ttheoretical\tintraday,volume\tMOC - Factors\t盘中成交量异动\n"
        "10_concepts/Concept - Momentum.md\tconcept\tMomentum\tprice_action\t"
        "stable\tmomentum,concept\tMOC - Concepts\t动量概念\n",
        encoding="utf-8",
    )
    (vault / "30_factors" / "Factor - Daily PV.md").write_text(
        "---\ntype: factor\n---\n# 日频价量\n", encoding="utf-8"
    )
    (vault / "30_factors" / "Factor - Intraday Volume Burst.md").write_text(
        "---\ntype: factor\n---\n# 盘中成交量\n", encoding="utf-8"
    )
    (vault / "10_concepts" / "Concept - Momentum.md").write_text(
        "---\ntype: concept\n---\n# 动量\n", encoding="utf-8"
    )
    (vault / "90_computed" / "graph.json").write_text(
        json.dumps(
            {
                "meta": {"node_count": 3, "edge_count": 3},
                "nodes": {
                    "Daily PV": {
                        "type": "factor",
                        "domain": "price_action",
                        "lifecycle": "validated",
                        "market": "a_share",
                        "mechanism": "behavioral",
                        "factor_family": "momentum",
                        "path": "30_factors/Factor - Daily PV.md",
                    },
                    "Intraday Volume Burst": {
                        "type": "factor",
                        "domain": "price_action",
                        "lifecycle": "theoretical",
                        "market": "a_share",
                        "mechanism": "behavioral",
                        "factor_family": "momentum",
                        "path": "30_factors/Factor - Intraday Volume Burst.md",
                    },
                    "Momentum": {
                        "type": "concept",
                        "domain": "price_action",
                        "lifecycle": "stable",
                        "market": "a_share",
                        "mechanism": "",
                        "factor_family": "",
                        "path": "10_concepts/Concept - Momentum.md",
                    },
                },
                "edges": [
                    {
                        "source": "Daily PV",
                        "target": "close",
                        "type": "uses_data",
                        "target_kind": "data_identifier",
                        "derived": False,
                    },
                    {
                        "source": "Daily PV",
                        "target": "volume",
                        "type": "uses_data",
                        "target_kind": "data_identifier",
                        "derived": False,
                    },
                    {
                        "source": "Intraday Volume Burst",
                        "target": "intraday_tick_volume",
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
    return vault


def test_explore_idea_emits_retrieval_diagnostics_in_all_modes(tmp_path: Path) -> None:
    vault = _build_vault_with_uses_data(tmp_path)
    for mode in ("start", "free", "constrained"):
        result = explore_idea(vault_root=vault, idea="日频动量价量", mode=mode)
        payload = result.to_payload()
        diag = payload["retrieval_diagnostics"]
        assert isinstance(diag, dict)
        assert diag["mode"] == mode
        # Five-component weights table is present and matches mode.
        weights = diag["score_weights"]
        assert set(weights.keys()) == {
            "semantic",
            "metadata",
            "mechanism",
            "dependency",
            "failure",
        }
        # Every kept candidate carries score components.
        components_by_name = diag["score_components_by_name"]
        assert isinstance(components_by_name, dict)
        for name, payload_components in components_by_name.items():
            assert isinstance(name, str) and name
            assert "aggregate" in payload_components
            for key in ("semantic", "metadata", "mechanism", "dependency", "failure"):
                assert key in payload_components


def test_explore_idea_prompt_surfaces_per_card_score_components(
    tmp_path: Path,
) -> None:
    vault = _build_vault_with_uses_data(tmp_path)

    result = explore_idea(vault_root=vault, idea="日频动量价量", mode="free")
    prompt = result.to_payload()["gpt_prompt"]

    assert "retrieval score:" in prompt
    assert "semantic=" in prompt
    assert "metadata=" in prompt
    assert "mechanism=" in prompt
    assert "dependency=" in prompt
    assert "failure=" in prompt
    assert "aggregate=" in prompt
    assert "检索分量说明" in prompt


def test_explore_idea_constrained_prompt_requires_score_component_rationale(
    tmp_path: Path,
) -> None:
    vault = _build_vault_with_uses_data(tmp_path)

    result = explore_idea(vault_root=vault, idea="日频动量价量", mode="constrained")
    prompt = result.to_payload()["gpt_prompt"]

    assert "constrained 模式下，引用某张卡片时必须说明你主要依赖哪个检索分量" in prompt


def test_explore_idea_p1_hard_filter_drops_unsatisfiable_dependency(
    tmp_path: Path,
) -> None:
    vault = _build_vault_with_uses_data(tmp_path)
    # Available data set deliberately excludes intraday_tick_volume.
    result = explore_idea(
        vault_root=vault,
        idea="日频动量价量 intraday volume burst",
        mode="free",
        available_data=frozenset({"close", "volume"}),
    )
    payload = result.to_payload()
    diag = payload["retrieval_diagnostics"]
    assert diag["available_data_provided"] is True

    dropped_names = [item["name"] for item in diag["dropped_cards"]]
    assert "Intraday Volume Burst" in dropped_names
    drop_reason = next(
        item["reason"]
        for item in diag["dropped_cards"]
        if item["name"] == "Intraday Volume Burst"
    )
    assert "intraday_tick_volume" in drop_reason

    surviving_names = set(diag["score_components_by_name"].keys())
    assert "Intraday Volume Burst" not in surviving_names

    related_names = {card["name"] for card in payload["related_cards"]}
    assert "Intraday Volume Burst" not in related_names


def test_explore_idea_no_inventory_means_no_hard_filter(tmp_path: Path) -> None:
    vault = _build_vault_with_uses_data(tmp_path)
    result = explore_idea(
        vault_root=vault,
        idea="intraday volume burst",
        mode="free",
        # available_data omitted -> dependency is a soft 0.5 signal only
    )
    diag = result.to_payload()["retrieval_diagnostics"]
    assert diag["available_data_provided"] is False
    assert diag["dropped_cards"] == []


# ---------------------------------------------------------------------------
# Workflow stage axis (P2): stage dispatch, frequency-driven inventory,
# concept-level prohibitions in mechanism_discovery prompts
# ---------------------------------------------------------------------------


def test_explore_idea_default_stage_is_mechanism_discovery(tmp_path: Path) -> None:
    vault = _build_vault_with_uses_data(tmp_path)
    result = explore_idea(vault_root=vault, idea="日频价量动量", mode="free")
    diag = result.to_payload()["retrieval_diagnostics"]
    assert diag["stage"] == "mechanism_discovery"
    assert diag["recommended_next_stage"] == "signal_mapping"


def test_explore_idea_stage_overrides_recommended_next_stage(tmp_path: Path) -> None:
    vault = _build_vault_with_uses_data(tmp_path)
    result = explore_idea(
        vault_root=vault,
        idea="日频价量动量",
        mode="free",
        stage="signal_mapping",
    )
    diag = result.to_payload()["retrieval_diagnostics"]
    assert diag["stage"] == "signal_mapping"
    assert diag["recommended_next_stage"] == "validation_kill_tests"
    # Real prompt (no longer a stub).
    prompt = result.to_payload()["gpt_prompt"]
    assert "stub" not in prompt
    assert "Signal Mapping Prompt" in prompt


def test_explore_idea_signal_mapping_emits_computability_audit(
    tmp_path: Path,
) -> None:
    vault = _build_vault_with_uses_data(tmp_path)
    result = explore_idea(
        vault_root=vault,
        idea="非对称的上下行 realized volatility（log(downside_vol) - log(upside_vol)）",
        mode="free",
        stage="signal_mapping",
    )
    prompt = result.to_payload()["gpt_prompt"]
    assert "Signal Mapping · Open" in prompt
    # Stage declaration: this stage is mechanism→signal computability,
    # NOT mechanism generation and NOT final selection.
    assert "可计算性审计" in prompt
    assert "不是在生成新机制" in prompt
    assert "不是在做最终淘汰" in prompt
    # Three-step mapping section is mandatory.
    assert "Mechanism → Implication → Data 三段映射" in prompt
    # Tags the LLM must apply on each required-data field.
    assert "daily sufficient" in prompt
    assert "intraday required" in prompt
    assert "necessary" in prompt
    assert "decorative" in prompt
    assert "confound control" in prompt
    # Current implementation interpretation is mandatory.
    assert "当前实现的解释" in prompt
    assert "捕捉了哪一个或哪几个机制" in prompt
    assert "漏掉了哪些机制" in prompt
    # Confound list with all five canonical families.
    assert "`reversal`" in prompt
    assert "`total volatility`" in prompt
    assert "`skewness / downside risk`" in prompt
    assert "`liquidity / turnover`" in prompt
    assert "`size / industry / price level`" in prompt
    # No final pick allowed at this stage.
    assert "禁止做最终选择" in prompt
    # mechanism_discovery prohibitions must NOT contaminate signal_mapping —
    # candidates already exist; we're making them testable, not stripping
    # them again.
    assert "概念禁用约束（mechanism_discovery 阶段）" not in prompt


def test_signal_mapping_prompt_builder_contract_direct() -> None:
    prompt = _build_factor_recipe_signal_mapping_prompt(
        idea="up/down realized volatility asymmetry",
        mode="constrained",
        project=None,
        context={},
    )

    assert "# AlphaLab Signal Mapping Prompt" in prompt
    assert "Signal Mapping" in prompt
    assert "Mechanism" in prompt
    assert "Implication" in prompt
    assert "Data" in prompt
    assert "daily sufficient" in prompt
    assert "intraday required" in prompt
    assert "necessary" in prompt
    assert "decorative" in prompt
    assert "confound control" in prompt
    assert "`reversal`" in prompt
    assert "`total volatility`" in prompt
    assert "`skewness / downside risk`" in prompt
    assert "`liquidity / turnover`" in prompt
    assert "`size / industry / price level`" in prompt
    assert "binary alias" in prompt
    assert "reversal / total volatility" in prompt
    assert "## 输出自检（系统会用 lint 校验你的输出）" in prompt


def test_explore_idea_signal_mapping_constrained_adds_strict_rules(
    tmp_path: Path,
) -> None:
    vault = _build_vault_with_uses_data(tmp_path)
    result = explore_idea(
        vault_root=vault,
        idea="非对称的上下行 realized volatility",
        mode="constrained",
        stage="signal_mapping",
    )
    prompt = result.to_payload()["gpt_prompt"]
    assert "Signal Mapping · Strict" in prompt
    # Strict: cite anchor for each implication.
    assert "cite 至少一张知识库卡片" in prompt
    # Strict: binary alias-tag against reversal / total volatility.
    assert "binary alias 标签" in prompt
    assert "reversal / total volatility" in prompt
    # Strict: 2-3 versions enforced.
    assert "输出版本数 ∈ {2, 3}" in prompt
    # `necessary` fields must be argued; otherwise downgrade.
    assert "变量论证" in prompt


def test_explore_idea_validation_kill_tests_emits_full_audit(tmp_path: Path) -> None:
    vault = _build_vault_with_uses_data(tmp_path)
    result = explore_idea(
        vault_root=vault,
        idea="非对称的上下行 realized volatility 因子",
        mode="free",
        stage="validation_kill_tests",
    )
    payload = result.to_payload()
    diag = payload["retrieval_diagnostics"]
    assert diag["stage"] == "validation_kill_tests"
    assert diag["recommended_next_stage"] is None
    prompt = payload["gpt_prompt"]
    # No longer a stub.
    assert "stub" not in prompt
    assert "Validation & Kill Tests Prompt" in prompt
    # Stage declaration: explicitly KILL-oriented.
    assert "try to KILL this factor" in prompt
    # Alias targets: forbidden labels return as audit targets, not as
    # forbidden words. All five canonical targets must be present.
    assert "`reversal`" in prompt
    assert "`volatility`" in prompt
    assert "`skewness / downside risk`" in prompt
    assert "`liquidity / turnover`" in prompt
    assert "`size / industry / price level`" in prompt
    # Required audit sections.
    assert "暴露分解" in prompt
    assert "数据健全性 kill tests" in prompt
    assert "实现稳健性 kill tests" in prompt
    assert "子样本稳定性 kill tests" in prompt
    assert "死亡条件" in prompt
    # mechanism_discovery prohibitions must NOT contaminate the audit
    # prompt — labels are TARGETS here, not forbidden words.
    assert "概念禁用约束（mechanism_discovery 阶段）" not in prompt
    assert "禁止用以下既有标签" not in prompt


def test_validation_kill_tests_prompt_builder_contract_direct() -> None:
    prompt = _build_factor_recipe_validation_kill_tests_prompt(
        idea="up/down realized volatility asymmetry",
        mode="constrained",
        project=None,
        context={},
    )

    assert "# AlphaLab Validation & Kill Tests Prompt" in prompt
    assert "Validation & Kill Tests" in prompt
    assert "try to KILL this factor" in prompt
    assert "`reversal`" in prompt
    assert "`volatility`" in prompt
    assert "`skewness / downside risk`" in prompt
    assert "`liquidity / turnover`" in prompt
    assert "`size / industry / price level`" in prompt
    assert "Exposure Decomposition" in prompt
    assert "Kill Verdict Rules" in prompt
    assert "KILL / HOLD-FOR-AUDIT" in prompt
    assert "## 输出自检（系统会用 lint 校验你的输出）" in prompt


def test_explore_idea_validation_kill_tests_constrained_forces_binary_verdict(
    tmp_path: Path,
) -> None:
    vault = _build_vault_with_uses_data(tmp_path)
    result = explore_idea(
        vault_root=vault,
        idea="非对称的上下行 realized volatility 因子",
        mode="constrained",
        stage="validation_kill_tests",
    )
    prompt = result.to_payload()["gpt_prompt"]
    assert "Validation & Kill Tests · Strict" in prompt
    # Strict variant: cite anchor for each alias verdict.
    assert "cite 至少一张知识库卡片" in prompt
    # Strict variant: binary final verdict, no hedging.
    assert "二值最终判定" in prompt
    assert "KILL / HOLD-FOR-AUDIT" in prompt
    assert "follow-up 实证检查" in prompt


def test_explore_idea_mechanism_discovery_emits_concept_prohibitions(
    tmp_path: Path,
) -> None:
    vault = _build_vault_with_uses_data(tmp_path)
    result = explore_idea(
        vault_root=vault,
        idea="非对称的上下行波动率特征",
        mode="free",
    )
    prompt = result.to_payload()["gpt_prompt"]
    # Section header is present and forbidden labels are listed.
    assert "概念禁用约束（mechanism_discovery 阶段）" in prompt
    assert "reversal" in prompt
    assert "momentum" in prompt
    assert "动量" in prompt
    # Direction-direction prohibition explicit.
    assert "禁止预设收益方向" in prompt
    # Anti-merging requirement explicit.
    assert "禁止把多个机制压成同一个故事" in prompt


def test_explore_idea_mechanism_discovery_constrained_adds_strict_self_check(
    tmp_path: Path,
) -> None:
    vault = _build_vault_with_uses_data(tmp_path)
    result = explore_idea(
        vault_root=vault,
        idea="非对称波动率特征",
        mode="constrained",
    )
    prompt = result.to_payload()["gpt_prompt"]
    # Strict self-check rule (only present in strict variant).
    assert "自检步骤" in prompt
    assert "如果只用 1 个上面禁用标签描述它" in prompt


def test_explore_idea_auto_inventory_from_project_frequency(tmp_path: Path) -> None:
    """Daily-frequency project must auto-filter HFT cards via dependency hard filter."""
    vault = _build_vault_with_uses_data(tmp_path)
    init_project(
        vault_root=vault,
        slug="daily-pv",
        title_zh="日频价量",
        category="factor_recipe",
        owner="yukun",
        market="ashare",
        frequency="daily",
        chatgpt_project_name="Daily PV",
    )
    result = explore_idea(
        vault_root=vault,
        idea="intraday volume burst 价量异动",
        mode="free",
        project_slug="daily-pv",
    )
    diag = result.to_payload()["retrieval_diagnostics"]
    # Inventory came from frequency, not explicit param.
    assert diag["available_data_provided"] is True
    assert diag["available_data_source"] == "frequency:daily"
    # The HFT card got filtered out before any expansion could pull it back in.
    dropped_names = {item["name"] for item in diag["dropped_cards"]}
    assert "Intraday Volume Burst" in dropped_names
    related_names = {card["name"] for card in result.to_payload()["related_cards"]}
    assert "Intraday Volume Burst" not in related_names


def test_explore_idea_persist_session_records_session_id_in_diagnostics(
    tmp_path: Path,
) -> None:
    vault = _build_vault_with_uses_data(tmp_path)
    workspace = tmp_path / "ws"
    result = explore_idea(
        vault_root=vault,
        idea="非对称的上下行 realized volatility",
        mode="free",
        workspace_root=workspace,
        persist_session=True,
    )
    diag = result.to_payload()["retrieval_diagnostics"]
    assert "session_id" in diag and diag["session_id"]
    sessions_dir = workspace / "artifacts" / "alpha_lab_explorer" / "sessions"
    assert sessions_dir.exists()
    saved = list(sessions_dir.glob("*.json"))
    assert len(saved) == 1


def test_explore_idea_persist_without_workspace_raises(tmp_path: Path) -> None:
    vault = _build_vault_with_uses_data(tmp_path)
    with pytest.raises(ValueError, match="persist_session"):
        explore_idea(
            vault_root=vault,
            idea="non-empty",
            mode="free",
            persist_session=True,
        )


def test_explore_idea_drift_injection_round_trip_propagates_violations(
    tmp_path: Path,
) -> None:
    """End-to-end: explore -> persist -> record bad response -> next
    explore reads recent violations and prepends them to the prompt."""
    from alpha_lab.research_bridge.sessions import record_explore_response

    vault = _build_vault_with_uses_data(tmp_path)
    workspace = tmp_path / "ws"

    # Round 1: explore + persist.
    first = explore_idea(
        vault_root=vault,
        idea="非对称的上下行 realized volatility 第一轮",
        mode="free",
        workspace_root=workspace,
        persist_session=True,
    )
    first_session_id = first.to_payload()["retrieval_diagnostics"]["session_id"]
    assert first_session_id

    # User runs the prompt externally; the response is intentionally bad.
    bad_response = (
        "## 输出\n\n"
        "[初步机制假设]\n"
        "### 机制 1: Reversal-style asymmetry\n"
        "做多波动率较低的股票，做空波动率较高的股票。\n"
    )
    report = record_explore_response(
        session_id=first_session_id,
        response_text=bad_response,
        workspace_root=workspace,
    )
    assert report.has_errors
    # Confirm specific drift modes were recorded.
    assert "forbidden_label_in_name" in report.violation_codes
    assert "forbidden_direction" in report.violation_codes

    # Round 2: explore with drift injection enabled.
    second = explore_idea(
        vault_root=vault,
        idea="非对称的上下行 realized volatility 第二轮",
        mode="free",
        workspace_root=workspace,
        inject_recent_drift=True,
    )
    second_payload = second.to_payload()
    diag = second_payload["retrieval_diagnostics"]
    assert diag["drift_injected_count"] >= 1
    prompt = second_payload["gpt_prompt"]
    # The drift header is prepended above the normal prompt body.
    assert prompt.startswith("## 已知漂移模式")
    # And it carries forward the actual codes from round 1.
    assert "forbidden_label_in_name" in prompt
    assert "forbidden_direction" in prompt


def test_explore_idea_drift_injection_filters_by_stage(tmp_path: Path) -> None:
    """A signal_mapping session's violations must not bleed into a
    mechanism_discovery prompt (and vice versa)."""
    from alpha_lab.research_bridge.sessions import record_explore_response

    vault = _build_vault_with_uses_data(tmp_path)
    workspace = tmp_path / "ws"

    # Persist a session at signal_mapping stage with violations.
    sm_run = explore_idea(
        vault_root=vault,
        idea="signal mapping run",
        mode="free",
        stage="signal_mapping",
        workspace_root=workspace,
        persist_session=True,
    )
    sm_session_id = sm_run.to_payload()["retrieval_diagnostics"]["session_id"]
    record_explore_response(
        session_id=sm_session_id,
        response_text="completely empty",
        workspace_root=workspace,
    )

    # Now explore at mechanism_discovery — should not pull signal_mapping violations.
    md_run = explore_idea(
        vault_root=vault,
        idea="mechanism discovery follow-up",
        mode="free",
        stage="mechanism_discovery",
        workspace_root=workspace,
        inject_recent_drift=True,
    )
    diag = md_run.to_payload()["retrieval_diagnostics"]
    assert diag["drift_injected_count"] == 0
    prompt = md_run.to_payload()["gpt_prompt"]
    assert not prompt.startswith("## 已知漂移模式")


def test_explore_idea_injects_upstream_stage_artifact(tmp_path: Path) -> None:
    from alpha_lab.research_bridge.sessions import record_explore_response

    vault = _build_vault_with_uses_data(tmp_path)
    workspace = tmp_path / "ws"

    discovery = explore_idea(
        vault_root=vault,
        idea="上下行 realized volatility asymmetry",
        mode="free",
        stage="mechanism_discovery",
        workspace_root=workspace,
        persist_session=True,
    )
    discovery_session_id = discovery.to_payload()["retrieval_diagnostics"][
        "session_id"
    ]
    response = """\
[初步机制假设]
### 机制 1: 风险厌恶非对称定价
- agent behavior: 投资者在亏损区间对下行波动更敏感
- structure constraint: 风险预算在回撤中被动收缩
- dynamic process: 下行波动升高后风险溢价可能被重新定价

### 机制 2: 负面信息释放速度差异
- agent behavior: 交易者先处理坏消息，再逐步修正估值
- structure constraint: 信息披露与交易约束不同步
- dynamic process: 下行波动可能代表信息集中释放，而非反转

[初步信号思路]
- 可能用到的数据: daily_prices / returns
- 可能的变换方式: 上下行 realized volatility 分解
- 直觉上的预测逻辑: 不同机制对未来收益方向要求不同

[与已有因子的关系]
- 最接近的已有标签: volatility / downside risk
- 可能的不同点: 关注上下行结构差，而不是无符号波动量级

[不确定性与风险点]
- 哪些部分不确定: 是否只是 reversal 或总波动率换壳
- 哪些假设最容易出错: 极端日期与涨跌停驱动
"""
    report = record_explore_response(
        session_id=discovery_session_id,
        response_text=response,
        workspace_root=workspace,
    )
    assert not report.has_errors

    mapped = explore_idea(
        vault_root=vault,
        idea="上下行 realized volatility asymmetry",
        mode="constrained",
        stage="signal_mapping",
        workspace_root=workspace,
        inject_recent_drift=True,
    )
    payload = mapped.to_payload()
    prompt = payload["gpt_prompt"]
    diag = payload["retrieval_diagnostics"]

    assert "## 上游产物" in prompt
    assert "风险厌恶非对称定价" in prompt
    assert "负面信息释放速度差异" in prompt
    assert diag["upstream_session_id"] == discovery_session_id
    assert diag["upstream_sections_injected"] >= 1


def test_explore_idea_prompt_includes_lint_self_check(tmp_path: Path) -> None:
    vault = _build_vault_with_uses_data(tmp_path)

    result = explore_idea(
        vault_root=vault,
        idea="上下行 realized volatility asymmetry",
        mode="constrained",
        stage="validation_kill_tests",
    )
    prompt = result.to_payload()["gpt_prompt"]

    assert "## 输出自检（系统会用 lint 校验你的输出）" in prompt


def test_explore_idea_prompt_includes_per_card_score_chip(tmp_path: Path) -> None:
    """Per-card score line surfaces the 5-component scoring under each
    card so the LLM can see which retrieval signal pulled the card in."""
    vault = _build_vault_with_uses_data(tmp_path)

    result = explore_idea(
        vault_root=vault,
        idea="Daily PV momentum 日频价量动量",
        mode="constrained",
        stage="mechanism_discovery",
    )
    payload = result.to_payload()
    prompt = payload["gpt_prompt"]
    diag = payload["retrieval_diagnostics"]

    # Diagnostics still carry the per-card score components.
    assert isinstance(diag.get("score_components_by_name"), dict)
    assert diag["score_components_by_name"], "expected at least one scored card"

    # Each scored card emits a `- retrieval score:` line with all five
    # components and the aggregate.
    assert "- retrieval score:" in prompt
    for label in ("semantic", "metadata", "mechanism", "dependency", "failure"):
        assert f"{label}=" in prompt, f"score component {label!r} missing"
    assert "aggregate=" in prompt

    # The footer explains what each component means and (in constrained
    # mode) tells the LLM it must justify which component drove a citation.
    assert "检索分量说明" in prompt
    assert "constrained 模式下，引用某张卡片时必须说明" in prompt


def test_explore_idea_explicit_inventory_wins_over_project_frequency(
    tmp_path: Path,
) -> None:
    vault = _build_vault_with_uses_data(tmp_path)
    init_project(
        vault_root=vault,
        slug="daily-pv-2",
        title_zh="日频价量2",
        category="factor_recipe",
        owner="yukun",
        market="ashare",
        frequency="daily",
        chatgpt_project_name="Daily PV 2",
    )
    # Caller explicitly opts into intraday inventory; project.frequency is ignored.
    result = explore_idea(
        vault_root=vault,
        idea="intraday volume burst",
        mode="free",
        project_slug="daily-pv-2",
        available_data=frozenset({"close", "volume", "intraday_tick_volume"}),
    )
    diag = result.to_payload()["retrieval_diagnostics"]
    assert diag["available_data_source"] == "explicit"
    assert diag["dropped_cards"] == []
