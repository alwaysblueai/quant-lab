"""Tests for web_unified.py — service layer, CLI routing, and key invariants."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from alpha_lab.web_unified import (
    _RunRecord,
    _UnifiedService,
    _extract_metrics_summary,
    _index_html_raw,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _build_vault(tmp_path: Path) -> Path:
    vault = tmp_path / "quant-knowledge"
    for rel in [
        "00_inbox",
        "_sources",
        "10_concepts",
        "20_methods",
        "30_factors",
        "50_experiments",
        "90_computed",
        "90_moc",
    ]:
        (vault / rel).mkdir(parents=True, exist_ok=True)

    # CARD-INDEX.tsv with two cards
    (vault / "90_moc" / "CARD-INDEX.tsv").write_text(
        "path\ttype\tname\tdomain\tlifecycle\ttags\tparent_moc\n"
        "30_factors/Factor - Momentum Base.md\tfactor\tMomentum Base\talpha_research\t"
        "theoretical\tmomentum,factor\tMOC - Factors\n"
        "10_concepts/Concept - IC.md\tconcept\tIC\talpha_research\t"
        "stable\tic,evaluation\tMOC - Concepts\n",
        encoding="utf-8",
    )
    (vault / "30_factors" / "Factor - Momentum Base.md").write_text(
        "---\ntype: factor\n---\n# 动量基类\n\n用于测试。\n",
        encoding="utf-8",
    )
    (vault / "10_concepts" / "Concept - IC.md").write_text(
        "---\ntype: concept\n---\n# Information Coefficient\n",
        encoding="utf-8",
    )
    (vault / "90_computed" / "graph.json").write_text(
        json.dumps(
            {
                "meta": {"node_count": 2, "edge_count": 2},
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
                    "IC": {
                        "type": "concept",
                        "domain": "evaluation",
                        "lifecycle": "stable",
                        "market": "a_share",
                        "mechanism": "",
                        "factor_family": "",
                        "path": "10_concepts/Concept - IC.md",
                    },
                },
                "edges": [
                    {
                        "source": "Momentum Base",
                        "target": "close",
                        "type": "uses_data",
                        "target_kind": "data_identifier",
                        "derived": False,
                    },
                    {
                        "source": "Momentum Base",
                        "target": "volume",
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
                "meta": {"built_at": "2026-04-12T00:00:00+00:00"},
                "explored_regions": [],
                "frontier": [
                    {
                        "direction": "liquidity-constrained momentum",
                        "factor_family": "momentum",
                        "mechanism": "behavioral",
                        "reason": "coverage gap",
                        "suggested_by": "graph coverage",
                        "priority": "high",
                    }
                ],
                "failure_registry_refs": [
                    {
                        "failure_id": "FK-001",
                        "title": "动量换壳失败",
                        "status": "active",
                        "failure_class": "redundant-idea",
                        "failure_statement": "仅改变 lookback 的动量变体缺乏新增信息。",
                        "prevention_rule": "不能只改窗口或标准化方式。",
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    # Inbox file
    (vault / "00_inbox" / "raw_note.md").write_text("raw note", encoding="utf-8")
    # Sources file
    (vault / "_sources" / "paper.pdf").write_text("pdf bytes", encoding="utf-8")
    return vault


def _make_service(tmp_path: Path, vault: Path) -> _UnifiedService:
    return _UnifiedService(vault_root=vault, workspace_root=tmp_path)


def _inject_succeeded_run_with_ic_timeseries(
    *,
    svc: _UnifiedService,
    tmp_path: Path,
    project_slug: str,
    run_id: str,
    case_name: str,
    factor_name: str,
    rank_ic_values: list[float],
    dsr_pvalue: float | None = None,
    dsr_from_metrics_artifact: bool = False,
) -> None:
    output_dir = tmp_path / f"run-{run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    ic_path = output_dir / "ic_timeseries.csv"
    lines = ["date,ic,rank_ic"]
    for idx, value in enumerate(rank_ic_values, start=1):
        lines.append(f"2026-01-{idx:02d},{value:.6f},{value:.6f}")
    ic_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    spec_path = tmp_path / f"{case_name}.yaml"
    spec_path.write_text(
        f"name: {case_name}\nfactor_name: {factor_name}\n",
        encoding="utf-8",
    )

    summary: dict[str, object] = {}
    artifact_paths: dict[str, str] = {"ic_timeseries": str(ic_path)}
    if dsr_pvalue is not None:
        if dsr_from_metrics_artifact:
            metrics_path = output_dir / "metrics.json"
            metrics_path.write_text(
                json.dumps({"metrics": {"dsr_pvalue": dsr_pvalue}}),
                encoding="utf-8",
            )
            artifact_paths["metrics"] = str(metrics_path)
        else:
            summary["dsr_pvalue"] = dsr_pvalue

    record = _RunRecord(
        run_id=run_id,
        project_slug=project_slug,
        case_name=case_name,
        round_id=None,
        spec_path=str(spec_path),
        submitted_at_utc=f"2026-04-14T00:00:0{run_id[-1]}Z",
        evaluation_profile="default_research",
        output_root_dir=None,
        render_report=True,
        status="succeeded",
        output_dir=str(output_dir),
        artifact_paths=artifact_paths,
        summary=summary,
    )
    with svc.run_store._lock:  # noqa: SLF001 - tests intentionally seed in-memory store
        svc.run_store._records[run_id] = record  # noqa: SLF001


def test_extract_metrics_summary_includes_scalar_diagnostics(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "metrics": {
                    "factor_verdict": "promising",
                    "mean_rank_ic": 0.03,
                    "ic_t_stat": 2.45,
                    "ic_p_value": 0.017,
                    "dsr_pvalue": 0.12,
                    "split_description": "train<=2021-12-31 / test>=2022-01-01",
                    "data_quality_status": "warn",
                    "data_quality_suspended_rows": 8,
                    "data_quality_stale_rows": 3,
                    "data_quality_suspected_split_rows": 1,
                    "data_quality_integrity_warn_count": 2,
                    "data_quality_integrity_fail_count": 0,
                    "data_quality_hard_fail_count": 0,
                }
            }
        ),
        encoding="utf-8",
    )

    summary = _extract_metrics_summary(metrics_path)

    assert summary["ic_t_stat"] == 2.45
    assert summary["ic_p_value"] == 0.017
    assert summary["dsr_pvalue"] == 0.12
    assert summary["split_description"] == "train<=2021-12-31 / test>=2022-01-01"
    assert summary["data_quality_status"] == "warn"
    assert summary["data_quality_suspended_rows"] == 8
    assert summary["data_quality_stale_rows"] == 3
    assert summary["data_quality_suspected_split_rows"] == 1
    assert summary["data_quality_integrity_warn_count"] == 2
    assert summary["data_quality_integrity_fail_count"] == 0
    assert summary["data_quality_hard_fail_count"] == 0


def test_project_factor_diagnostics_returns_heatmap_and_redundancy_warnings(
    tmp_path: Path,
) -> None:
    vault = _build_vault(tmp_path)
    svc = _make_service(tmp_path, vault)
    project_slug = "diag-project"

    _inject_succeeded_run_with_ic_timeseries(
        svc=svc,
        tmp_path=tmp_path,
        project_slug=project_slug,
        run_id="run1",
        case_name="case_a",
        factor_name="factor_a",
        rank_ic_values=[0.01, 0.02, 0.03, 0.05, 0.08, 0.13],
        dsr_pvalue=0.08,
    )
    _inject_succeeded_run_with_ic_timeseries(
        svc=svc,
        tmp_path=tmp_path,
        project_slug=project_slug,
        run_id="run2",
        case_name="case_b",
        factor_name="factor_b",
        rank_ic_values=[0.02, 0.04, 0.06, 0.10, 0.16, 0.26],
        dsr_pvalue=0.62,
        dsr_from_metrics_artifact=True,
    )

    diagnostics = svc.project_factor_diagnostics(project_slug, threshold=0.7, min_overlap=5)

    assert diagnostics["ok"] is True
    labels = diagnostics["labels"]
    assert isinstance(labels, list)
    assert "factor_a" in labels
    assert "factor_b" in labels
    matrix = diagnostics["matrix"]
    assert isinstance(matrix, list)
    assert len(matrix) == 2
    assert matrix[0][0] == pytest.approx(1.0)
    assert matrix[1][1] == pytest.approx(1.0)
    pairs = diagnostics["redundancy_pairs"]
    assert isinstance(pairs, list)
    assert len(pairs) == 1
    assert pairs[0]["factor_a"] == "factor_a"
    assert pairs[0]["factor_b"] == "factor_b"
    assert pairs[0]["abs_correlation"] == pytest.approx(1.0)
    dsr_summary = diagnostics["dsr_summary"]
    assert dsr_summary["n_runs_total"] == 2
    assert dsr_summary["n_with_dsr"] == 2
    assert dsr_summary["median_dsr_pvalue"] == pytest.approx(0.35)
    assert dsr_summary["robust_count"] == 1
    assert dsr_summary["high_risk_count"] == 1
    dsr_rows = diagnostics["dsr_by_factor"]
    assert isinstance(dsr_rows, list)
    assert len(dsr_rows) == 2
    assert dsr_rows[0]["factor_name"] == "factor_a"
    assert dsr_rows[0]["risk_level"] == "robust"
    assert dsr_rows[1]["factor_name"] == "factor_b"
    assert dsr_rows[1]["risk_level"] == "high_risk"


def test_project_factor_diagnostics_returns_not_ok_when_runs_insufficient(
    tmp_path: Path,
) -> None:
    vault = _build_vault(tmp_path)
    svc = _make_service(tmp_path, vault)
    project_slug = "diag-project"

    _inject_succeeded_run_with_ic_timeseries(
        svc=svc,
        tmp_path=tmp_path,
        project_slug=project_slug,
        run_id="run1",
        case_name="case_a",
        factor_name="factor_a",
        rank_ic_values=[0.01, 0.02, 0.03, 0.04, 0.05],
        dsr_pvalue=0.09,
    )

    diagnostics = svc.project_factor_diagnostics(project_slug)

    assert diagnostics["ok"] is False
    assert diagnostics["matrix"] == []
    assert diagnostics["redundancy_pairs"] == []
    dsr_summary = diagnostics["dsr_summary"]
    assert dsr_summary["n_runs_total"] == 1
    assert dsr_summary["n_with_dsr"] == 1
    assert dsr_summary["robust_count"] == 1
    dsr_rows = diagnostics["dsr_by_factor"]
    assert isinstance(dsr_rows, list)
    assert len(dsr_rows) == 1
    assert dsr_rows[0]["factor_name"] == "factor_a"


def test_index_html_includes_new_diagnostics_renderers() -> None:
    html = _index_html_raw()

    assert "purged_kfold_summary" in html
    assert "purged_kfold_folds" in html
    assert "renderPurgedKfoldSummaryJson" in html
    assert "renderPurgedKfoldFoldsCsv" in html
    assert "renderPortfolioValidationMetricsJson" in html
    assert "renderBarraAttributionSummaryJson" in html
    assert "renderMarketImpactSummaryJson" in html
    assert "renderBarraAttributionTimeseriesCsv" in html
    assert "实验隔离 (L3)" in html


# ---------------------------------------------------------------------------
# Knowledge Ops: vault_stats
# ---------------------------------------------------------------------------


def test_vault_stats_counts_cards_and_inbox(tmp_path: Path) -> None:
    vault = _build_vault(tmp_path)
    svc = _make_service(tmp_path, vault)

    stats = svc.vault_stats()

    assert stats["total_cards"] == 2
    assert stats["inbox_count"] == 2  # one in 00_inbox, one in _sources
    assert stats["by_type"] == {"concept": 1, "factor": 1}
    assert "theoretical" in stats["by_lifecycle"]
    assert "stable" in stats["by_lifecycle"]


def test_vault_stats_missing_index_returns_zeros(tmp_path: Path) -> None:
    vault = tmp_path / "empty-vault"
    vault.mkdir()
    svc = _make_service(tmp_path, vault)

    stats = svc.vault_stats()

    assert stats["total_cards"] == 0
    assert stats["inbox_count"] == 0


# ---------------------------------------------------------------------------
# Knowledge Ops: vault_inbox
# ---------------------------------------------------------------------------


def test_vault_inbox_lists_files(tmp_path: Path) -> None:
    vault = _build_vault(tmp_path)
    svc = _make_service(tmp_path, vault)

    inbox = svc.vault_inbox()

    assert inbox["count"] == 2
    names = {item["name"] for item in inbox["items"]}
    assert "raw_note.md" in names
    assert "paper.pdf" in names


def test_vault_inbox_empty_when_no_dirs(tmp_path: Path) -> None:
    vault = tmp_path / "bare-vault"
    vault.mkdir()
    svc = _make_service(tmp_path, vault)

    inbox = svc.vault_inbox()

    assert inbox["count"] == 0
    assert inbox["items"] == []


# ---------------------------------------------------------------------------
# Knowledge Ops: read_card
# ---------------------------------------------------------------------------


def test_read_card_returns_content(tmp_path: Path) -> None:
    vault = _build_vault(tmp_path)
    svc = _make_service(tmp_path, vault)

    result = svc.read_card("Factor - Momentum Base.md")

    assert "动量基类" in result["content"]
    assert result["truncated"] is False


def test_read_card_missing_raises(tmp_path: Path) -> None:
    vault = _build_vault(tmp_path)
    svc = _make_service(tmp_path, vault)

    with pytest.raises(FileNotFoundError):
        svc.read_card("Factor - Nonexistent.md")


def test_read_card_rejects_path_traversal(tmp_path: Path) -> None:
    vault = _build_vault(tmp_path)
    svc = _make_service(tmp_path, vault)

    with pytest.raises((ValueError, FileNotFoundError, PermissionError)):
        svc.read_card("../../etc/passwd")


def test_read_card_vault_relative_path(tmp_path: Path) -> None:
    # Vault-relative paths (as stored in CARD-INDEX.tsv) must work
    vault = _build_vault(tmp_path)
    svc = _make_service(tmp_path, vault)

    result = svc.read_card("30_factors/Factor - Momentum Base.md")
    assert "动量基类" in result["content"]
    assert result["truncated"] is False


def test_read_card_nested_subdir_path(tmp_path: Path) -> None:
    # Nested subdir paths like "10_concepts/behavioral/Concept - X.md" must work
    vault = _build_vault(tmp_path)
    subdir = vault / "10_concepts" / "behavioral"
    subdir.mkdir(parents=True, exist_ok=True)
    (subdir / "Concept - Habit Formation.md").write_text(
        "---\ntype: concept\n---\n# Habit Formation\n", encoding="utf-8"
    )
    svc = _make_service(tmp_path, vault)

    result = svc.read_card("10_concepts/behavioral/Concept - Habit Formation.md")
    assert "Habit Formation" in result["content"]


def test_read_card_rejects_traversal_via_slash(tmp_path: Path) -> None:
    # Paths with .. must still be rejected even when slash is allowed
    vault = _build_vault(tmp_path)
    svc = _make_service(tmp_path, vault)

    with pytest.raises(PermissionError):
        svc.read_card("../outside_vault/secret.md")


# ---------------------------------------------------------------------------
# Knowledge Ops: list_evaluation_profiles
# ---------------------------------------------------------------------------


def test_list_evaluation_profiles_has_default(tmp_path: Path) -> None:
    vault = _build_vault(tmp_path)
    svc = _make_service(tmp_path, vault)

    result = svc.list_evaluation_profiles()

    assert "profiles" in result
    assert "default_research" in result["profiles"]
    assert result["default_profile"] is not None


# ---------------------------------------------------------------------------
# Knowledge Ops: explore_idea
# ---------------------------------------------------------------------------


def test_explore_idea_start_mode_returns_kickoff_prompt(tmp_path: Path) -> None:
    vault = _build_vault(tmp_path)
    svc = _make_service(tmp_path, vault)

    result = svc.explore_idea("momentum reversal 动量 反转", "start")

    assert result["mode"] == "start"
    assert isinstance(result["related_cards"], list)
    assert isinstance(result["gpt_prompt"], str)
    assert "Research Kickoff" in result["gpt_prompt"]
    assert "You are in the research kickoff stage." in result["gpt_prompt"]
    assert "Your goal is to expand the hypothesis space, not to converge." in result["gpt_prompt"]
    assert "不允许输出最终因子定义或收敛结论" in result["gpt_prompt"]
    assert "Failure to differentiate is considered invalid reasoning." in result["gpt_prompt"]
    assert len(result["related_cards"]) >= 1
    assert result["constraint_report"] == {}


def test_explore_idea_free_mode_returns_structured_prompt(tmp_path: Path) -> None:
    vault = _build_vault(tmp_path)
    svc = _make_service(tmp_path, vault)

    result = svc.explore_idea("momentum reversal 动量 反转", "free")

    assert result["mode"] == "free"
    assert isinstance(result["related_cards"], list)
    assert isinstance(result["gpt_prompt"], str)
    assert "Structured Exploration" in result["gpt_prompt"]
    assert "允许写候选表达式，但不允许做最终选择、ranking 或输出 single best idea。" in result["gpt_prompt"]
    assert "[候选表达]" in result["gpt_prompt"]
    assert "[风险识别]" in result["gpt_prompt"]
    assert "[与已有因子的差异]" in result["gpt_prompt"]
    assert "不要做最终选择，不要 ranking，不要收敛到单一结论。" in result["gpt_prompt"]
    # momentum / reversal tags should match at least one card
    assert len(result["related_cards"]) >= 1
    assert result["constraint_report"] == {}


def test_explore_idea_constrained_mode_returns_report(tmp_path: Path) -> None:
    vault = _build_vault(tmp_path)
    svc = _make_service(tmp_path, vault)

    result = svc.explore_idea("动量", "constrained")

    assert result["mode"] == "constrained"
    assert "Graph 约束模式（硬约束）" in result["gpt_prompt"]
    assert "你只能使用以下数据节点与算子构造信号，不允许引入新变量。" in result["gpt_prompt"]
    assert "只保留总评分最高的 1-2 个机制。" in result["gpt_prompt"]
    assert "如果反对意见成立，请明确写出：修改假设，还是回到 Step 2 重新选择机制。" in result["gpt_prompt"]
    assert "- close" in result["gpt_prompt"]
    assert "- volume" in result["gpt_prompt"]
    cr = result["constraint_report"]
    assert isinstance(cr, dict)
    # keys must be present regardless of vault content
    assert "primary_family" in cr
    assert "primary_mechanism" in cr
    assert "family_counts" in cr
    assert "crowding_warning" in cr


def test_explore_idea_empty_raises(tmp_path: Path) -> None:
    vault = _build_vault(tmp_path)
    svc = _make_service(tmp_path, vault)

    with pytest.raises(ValueError):
        svc.explore_idea("", "free")


def test_explore_idea_unknown_mode_defaults_to_free(tmp_path: Path) -> None:
    vault = _build_vault(tmp_path)
    svc = _make_service(tmp_path, vault)

    result = svc.explore_idea("IC 信息系数", "banana")
    assert result["mode"] == "free"


def test_explore_idea_discussion_alias_maps_to_start(tmp_path: Path) -> None:
    vault = _build_vault(tmp_path)
    svc = _make_service(tmp_path, vault)

    result = svc.explore_idea("动量", "discussion")
    assert result["mode"] == "start"


def test_explore_idea_accepts_project_slug(tmp_path: Path) -> None:
    vault = _build_vault(tmp_path)
    svc = _make_service(tmp_path, vault)
    svc.create_project(
        {
            "slug": "test-momentum",
            "title_zh": "动量测试项目",
            "category": "factor_recipe",
            "owner": "test",
            "market": "ashare",
            "frequency": "daily",
            "chatgpt_project_name": "Test Momentum",
            "origin_cards": ["30_factors/Factor - Momentum Base.md"],
        }
    )

    result = svc.explore_idea("momentum 动量", "constrained", "test-momentum")

    assert result["mode"] == "constrained"
    assert result["related_cards"]


# ---------------------------------------------------------------------------
# Bridge Workspace: project + round + case setup
# ---------------------------------------------------------------------------


def _create_project_and_case(svc: _UnifiedService) -> tuple[str, str]:
    """Create a project and case. Return (slug, case_name)."""
    svc.create_project(
        {
            "slug": "test-momentum",
            "title_zh": "动量测试项目",
            "category": "factor_family",
            "owner": "test",
            "market": "ashare",
            "frequency": "daily",
            "chatgpt_project_name": "Test Momentum",
            "origin_cards": [],
        }
    )
    svc.create_case(
        "test-momentum",
        {
            "case_name": "mom_5d",
            "factor_name": "mom_5d",
            "base_method": "momentum",
            "lookback": 5,
            "skip_recent": 0,
            "target_horizon": 5,
        },
    )
    return "test-momentum", "mom_5d"


def test_list_cases_returns_cases(tmp_path: Path) -> None:
    vault = _build_vault(tmp_path)
    svc = _make_service(tmp_path, vault)
    slug, _ = _create_project_and_case(svc)

    cases = svc.list_cases(slug)

    assert len(cases) == 1
    assert cases[0]["case_name"] == "mom_5d"
    assert cases[0]["spec_exists"] is True


def test_list_cases_empty_for_unknown_project(tmp_path: Path) -> None:
    vault = _build_vault(tmp_path)
    svc = _make_service(tmp_path, vault)

    cases = svc.list_cases("nonexistent-project")
    assert cases == []


# ---------------------------------------------------------------------------
# CLI routing: web unified
# ---------------------------------------------------------------------------


def test_cli_routes_web_unified(monkeypatch: pytest.MonkeyPatch) -> None:
    from typing import Any

    captured: dict[str, Any] = {}

    def _fake_start_unified_server(
        *,
        host: str = "127.0.0.1",
        port: int = 8766,
        workspace_root: str = ".",
        vault_root: str | None = None,
        open_browser: bool = True,
    ) -> None:
        captured["host"] = host
        captured["port"] = port
        captured["workspace_root"] = workspace_root
        captured["vault_root"] = vault_root
        captured["open_browser"] = open_browser

    monkeypatch.setattr(
        "alpha_lab.web_unified.start_unified_server",
        _fake_start_unified_server,
    )

    from alpha_lab.cli import main

    rc = main(
        [
            "web",
            "unified",
            "--host",
            "0.0.0.0",
            "--port",
            "9000",
            "--workspace-root",
            "/tmp/alpha-lab",
            "--vault-root",
            "/tmp/vault",
            "--no-open-browser",
        ]
    )
    assert rc == 0
    assert captured == {
        "host": "0.0.0.0",
        "port": 9000,
        "workspace_root": "/tmp/alpha-lab",
        "vault_root": "/tmp/vault",
        "open_browser": False,
    }


def test_cli_web_unified_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def _fake_start_unified_server(
        *,
        host: str = "127.0.0.1",
        port: int = 8766,
        workspace_root: str = ".",
        vault_root: str | None = None,
        open_browser: bool = True,
    ) -> None:
        captured["host"] = host
        captured["port"] = port
        captured["open_browser"] = open_browser
        captured["vault_root"] = vault_root

    monkeypatch.setattr(
        "alpha_lab.web_unified.start_unified_server",
        _fake_start_unified_server,
    )

    from alpha_lab.cli import main

    rc = main(["web", "unified"])
    assert rc == 0
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 8766
    assert captured["open_browser"] is True
    assert captured["vault_root"] is None


# ---------------------------------------------------------------------------
# Custom Factor Workshop
# ---------------------------------------------------------------------------

_VALID_FACTOR_CODE = """
def builder(prices, *, window=20, skip_recent=0, min_periods=None, **kwargs):
    import pandas as pd
    frame = prices.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values(["asset", "date"]).reset_index(drop=True)
    ret = frame.groupby("asset", sort=False)["close"].pct_change(fill_method=None)
    result = frame[["date", "asset"]].copy()
    result["factor"] = "test_custom"
    result["value"] = -ret.rolling(window).std()
    return result
""".strip()


def test_register_custom_factor(tmp_path: Path) -> None:
    vault = _build_vault(tmp_path)
    svc = _make_service(tmp_path, vault)

    result = svc.register_custom_factor({
        "name": "test_vol",
        "code": _VALID_FACTOR_CODE,
        "description": "test volatility factor",
    })

    assert result["registered"] is True
    assert result["name"] == "test_vol"
    # Verify it's in the registry
    from alpha_lab.factor_recipe import factor_registry
    assert "test_vol" in factor_registry
    # Verify persistence
    meta_path = tmp_path / "custom_factors" / "test_vol.json"
    assert meta_path.exists()
    # Clean up registry
    factor_registry._builders.pop("test_vol", None)


def test_list_custom_factors_includes_builtins(tmp_path: Path) -> None:
    vault = _build_vault(tmp_path)
    svc = _make_service(tmp_path, vault)

    result = svc.list_custom_factors()

    names = [f["name"] for f in result["factors"]]
    assert "momentum" in names
    assert "reversal" in names
    assert result["total"] >= 5


def test_register_and_delete_custom_factor(tmp_path: Path) -> None:
    vault = _build_vault(tmp_path)
    svc = _make_service(tmp_path, vault)

    svc.register_custom_factor({
        "name": "temp_factor",
        "code": _VALID_FACTOR_CODE,
    })

    from alpha_lab.factor_recipe import factor_registry
    assert "temp_factor" in factor_registry

    svc.delete_custom_factor("temp_factor")
    assert "temp_factor" not in factor_registry
    meta_path = tmp_path / "custom_factors" / "temp_factor.json"
    assert not meta_path.exists()


def test_register_custom_factor_invalid_code(tmp_path: Path) -> None:
    vault = _build_vault(tmp_path)
    svc = _make_service(tmp_path, vault)

    with pytest.raises(ValueError, match="syntax error"):
        svc.register_custom_factor({"name": "bad", "code": "def builder(:"})


def test_register_custom_factor_missing_builder(tmp_path: Path) -> None:
    vault = _build_vault(tmp_path)
    svc = _make_service(tmp_path, vault)

    with pytest.raises(ValueError, match="must define a callable named 'builder'"):
        svc.register_custom_factor({"name": "bad2", "code": "x = 42"})


def test_delete_builtin_factor_raises(tmp_path: Path) -> None:
    vault = _build_vault(tmp_path)
    svc = _make_service(tmp_path, vault)

    with pytest.raises(ValueError, match="cannot delete built-in"):
        svc.delete_custom_factor("momentum")


def test_get_custom_factor_code(tmp_path: Path) -> None:
    vault = _build_vault(tmp_path)
    svc = _make_service(tmp_path, vault)

    svc.register_custom_factor({
        "name": "view_test",
        "code": _VALID_FACTOR_CODE,
        "description": "viewable factor",
    })

    result = svc.get_custom_factor_code("view_test")

    assert result["name"] == "view_test"
    assert "def builder" in result["code"]
    assert result["description"] == "viewable factor"

    # Clean up
    from alpha_lab.factor_recipe import factor_registry
    factor_registry._builders.pop("view_test", None)


def test_persisted_factors_reload_on_init(tmp_path: Path) -> None:
    vault = _build_vault(tmp_path)
    svc1 = _make_service(tmp_path, vault)

    svc1.register_custom_factor({
        "name": "persist_test",
        "code": _VALID_FACTOR_CODE,
    })

    from alpha_lab.factor_recipe import factor_registry
    # Remove from in-memory registry to simulate fresh start
    factor_registry._builders.pop("persist_test", None)
    assert "persist_test" not in factor_registry

    # Create new service — should reload from disk
    svc2 = _make_service(tmp_path, vault)
    assert "persist_test" in factor_registry

    # Clean up
    factor_registry._builders.pop("persist_test", None)
