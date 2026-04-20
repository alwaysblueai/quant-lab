from __future__ import annotations

import json
from pathlib import Path

import pytest

from alpha_lab.research_bridge.models import load_project_config
from alpha_lab.research_bridge.preflight import run_preflight
from alpha_lab.research_bridge.service import (
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
