"""Tests for web_unified.py — service layer, CLI routing, and key invariants."""

from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import yaml

import alpha_lab.research_bridge.model_idea as model_idea_module
import alpha_lab.research_bridge.service as research_bridge_service
from alpha_lab.exceptions import AlphaLabIOError
from alpha_lab.research_bridge.llm_rerank import RerankOutcome
from alpha_lab.web_unified import (
    _build_frontend_batch_parallel_config,
    _build_model_lab_batch_worker_count,
    _build_model_lab_subprocess_command,
    _build_model_lab_subprocess_env,
    _extract_metrics_summary,
    _index_html_raw,
    _load_model_factor_artifact_paths_from_manifest,
    _model_lab_html,
    _resolve_run_artifact_for_endpoint,
    _resolve_single_factor_web_output_root_dir,
    _RunRecord,
    _RunStore,
    _RunTask,
    _UnifiedService,
)
from tests.model_factor_case_helpers import write_demo_model_factor_case
from tests.single_factor_case_helpers import write_demo_single_factor_case

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _disable_llm_rerank(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_rerank_candidates(**_: object) -> RerankOutcome:
        return RerankOutcome(
            enabled=False,
            model="claude-sonnet-4-6",
            scores={},
            reasons={},
            tokens_input=0,
            tokens_output=0,
            cache_hit_input=0,
            dropped_invalid_names=[],
            fallback_reason="no_api_key",
        )

    monkeypatch.setattr(
        research_bridge_service,
        "rerank_candidates",
        fake_rerank_candidates,
    )
    monkeypatch.setattr(
        model_idea_module,
        "rerank_candidates",
        fake_rerank_candidates,
    )


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


def test_llm_settings_save_reload_and_clear(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.delenv("ALPHA_LAB_RESEARCH_BRIDGE_V2", raising=False)
    vault = _build_vault(tmp_path)
    svc = _make_service(tmp_path, vault)

    initial = svc.llm_settings_status()
    assert initial["anthropic_api_key_configured"] is False
    assert initial["anthropic_api_key_source"] == "none"

    saved = svc.save_llm_settings(
        {
            "anthropic_api_key": "sk-ant-test-local",
            "anthropic_base_url": "https://anthropic-proxy.example/v1",
            "research_bridge_v2_enabled": True,
        }
    )

    assert saved["anthropic_api_key_configured"] is True
    assert saved["anthropic_api_key_source"] == "saved"
    assert saved["anthropic_base_url"] == "https://anthropic-proxy.example/v1"
    assert saved["anthropic_base_url_source"] == "saved"
    assert saved["research_bridge_v2_enabled"] is True
    assert "anthropic_api_key" not in saved
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-test-local"
    assert os.environ["ANTHROPIC_BASE_URL"] == "https://anthropic-proxy.example/v1"
    assert os.environ["ALPHA_LAB_RESEARCH_BRIDGE_V2"] == "1"

    settings_path = tmp_path / ".research_bridge_cache" / "secret_settings.json"
    raw = json.loads(settings_path.read_text(encoding="utf-8"))
    assert raw["anthropic_api_key"] == "sk-ant-test-local"
    assert raw["anthropic_base_url"] == "https://anthropic-proxy.example/v1"
    assert raw["research_bridge_v2_enabled"] is True

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.delenv("ALPHA_LAB_RESEARCH_BRIDGE_V2", raising=False)
    reloaded = _make_service(tmp_path, vault)
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-test-local"
    assert os.environ["ANTHROPIC_BASE_URL"] == "https://anthropic-proxy.example/v1"
    assert os.environ["ALPHA_LAB_RESEARCH_BRIDGE_V2"] == "1"
    assert reloaded.llm_settings_status()["anthropic_api_key_source"] == "saved"

    cleared = reloaded.save_llm_settings(
        {
            "clear_anthropic_api_key": True,
            "clear_anthropic_base_url": True,
            "research_bridge_v2_enabled": False,
        }
    )

    assert cleared["anthropic_api_key_configured"] is False
    assert cleared["anthropic_api_key_source"] == "none"
    assert cleared["anthropic_base_url"] == ""
    assert cleared["anthropic_base_url_source"] == "default"
    assert cleared["research_bridge_v2_enabled"] is False
    assert "ANTHROPIC_API_KEY" not in os.environ
    assert "ANTHROPIC_BASE_URL" not in os.environ
    assert "ALPHA_LAB_RESEARCH_BRIDGE_V2" not in os.environ
    raw_after_clear = json.loads(settings_path.read_text(encoding="utf-8"))
    assert "anthropic_api_key" not in raw_after_clear
    assert "anthropic_base_url" not in raw_after_clear
    assert raw_after_clear["research_bridge_v2_enabled"] is False


def test_mechanism_index_status_reports_literal_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ALPHA_LAB_RESEARCH_BRIDGE_V2", raising=False)
    vault = _build_vault(tmp_path)
    svc = _make_service(tmp_path, vault)

    status = svc.mechanism_index_status()

    assert status["ok"] is True
    assert status["ready"] is False
    assert status["status"] == "not_built"
    assert status["card_count"] == 0
    assert status["research_bridge_v2_enabled"] is False
    assert status["anthropic_api_key_configured"] is False
    assert status["research_bridge_v2_active"] is False


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


def _inject_model_lab_run_for_compare(
    *,
    svc: _UnifiedService,
    tmp_path: Path,
    run_id: str,
    case_name: str,
    factor_name: str,
    model_family: str,
    metrics: dict[str, object],
    rank_ic_values: list[float],
    top_features: list[str],
    integrity: dict[str, object],
) -> None:
    _inject_succeeded_run_with_ic_timeseries(
        svc=svc,
        tmp_path=tmp_path,
        project_slug="__model_lab__",
        run_id=run_id,
        case_name=case_name,
        factor_name=factor_name,
        rank_ic_values=rank_ic_values,
    )
    output_dir = tmp_path / f"run-{run_id}"
    turnover_path = output_dir / "turnover.csv"
    turnover_path.write_text("date,turnover\n2026-01-01,0.60\n2026-01-02,0.62\n", encoding="utf-8")
    metrics_path = output_dir / "metrics.json"
    metric_payload = {"metrics": dict(metrics)}
    metric_payload["model_family"] = model_family
    metrics_path.write_text(json.dumps(metric_payload), encoding="utf-8")

    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "case_name": case_name,
                "integrity_summary": integrity,
            }
        ),
        encoding="utf-8",
    )

    feature_importance_path = output_dir / "feature_importance.csv"
    feature_importance_path.write_text(
        "feature,mean_abs_importance,latest_importance\n"
        + "\n".join(
            f"{feature},{1.0 / (idx + 1):.6f},{1.0 / (idx + 2):.6f}"
            for idx, feature in enumerate(top_features)
        )
        + "\n",
        encoding="utf-8",
    )
    with svc.run_store._lock:  # noqa: SLF001 - tests intentionally seed in-memory store
        current = svc.run_store._records[run_id]
        if current is not None:
            replacement = current.clone()
            replacement.workflow = "model_factor"
            replacement.artifact_paths = dict(
                replacement.artifact_paths,
                **{
                    "metrics": str(metrics_path),
                    "turnover": str(turnover_path),
                    "feature_importance": str(feature_importance_path),
                    "run_manifest": str(manifest_path),
                },
            )
            replacement.status = "succeeded"
            replacement.summary = {
                "factor_name": factor_name,
                **{str(key): value for key, value in metrics.items()},
                "model_family": model_family,
            }
            svc.run_store._records[run_id] = replacement


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
    assert summary["split_status"] == "missing"


def test_extract_metrics_summary_flattens_strict_split_contract(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.json"
    split_contract = {
        "policy": "auto_holdout_v1",
        "source": "single_factor_pipeline",
        "is_start": "2020-01-01",
        "is_end": "2023-12-29",
        "oos_start": "2024-01-08",
        "oos_end": "2024-12-31",
        "embargo_days": 5,
        "min_oos_dates": 60,
        "min_is_dates": 60,
        "n_dates": 1250,
        "n_is_dates": 995,
        "n_oos_dates": 250,
        "target_horizon": 5,
        "rebalance_step": 5,
    }
    metrics_path.write_text(
        json.dumps(
            {
                "metrics": {
                    "factor_verdict": "promising",
                    "split_contract": split_contract,
                }
            }
        ),
        encoding="utf-8",
    )

    summary = _extract_metrics_summary(metrics_path)

    assert summary["split_status"] == "strict"
    assert summary["oos_start"] == "2024-01-08"
    assert summary["embargo_days"] == 5


def test_extract_metrics_summary_marks_legacy_full_sample(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(
        json.dumps({"metrics": {"split_description": "full_sample"}}),
        encoding="utf-8",
    )

    summary = _extract_metrics_summary(metrics_path)

    assert summary["split_status"] == "full_sample"


def test_extract_metrics_summary_marks_missing_when_split_fields_absent(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(
        json.dumps({"metrics": {"factor_verdict": "watch"}}),
        encoding="utf-8",
    )

    summary = _extract_metrics_summary(metrics_path)

    assert summary["split_status"] == "missing"


def test_frontend_split_date_uses_explicit_contract_fields_only() -> None:
    unified_html = _index_html_raw()
    model_html = _model_lab_html()

    assert "function resolveRunSplitDate" in unified_html
    assert "safeSummary.split_contract" in unified_html
    assert "parseTestStartDate(summary.split_description)" not in unified_html
    assert "parseTestStartDate(run && run.summary" not in unified_html

    assert "function resolveOverviewSplitDate" in model_html
    assert "summary.split_contract" in model_html
    assert "inferOverviewSplitDateFromSnapshot" not in model_html


def test_model_lab_numeric_helpers_preserve_missing_values_and_sample_ir() -> None:
    html = _model_lab_html()

    assert (
        'if (!raw || ["nan", "null", "none", "n/a"].includes(raw.toLowerCase())) '
        "return null;"
    ) in html
    assert (
        'if (!raw || ["nan", "null", "none", "n/a"].includes(raw.toLowerCase())) '
        'return "-";'
    ) in html
    assert "const value = toNum(rawValue);" in html
    assert "const variance = values.reduce(" in html
    assert ") / (values.length - 1);" in html


def test_model_lab_coverage_chart_formats_percent_axis_and_summary_stats() -> None:
    html = _model_lab_html()
    unified_html = _index_html_raw()

    assert "Coverage 时序" in html
    assert 'valueFormat: "percent"' in html
    assert "yClampMax: 1" in html
    assert "showPointTooltips: true" in html
    assert "coverage-summary-grid" in html
    assert "asset_coverage_by_date" in html
    assert "coverageTickDecimals" in html
    assert "period:" in html
    assert "valid_score_count" in html
    assert "valid_forward_return_count" in html
    assert "break_flag:" in html
    assert "break_reason:" in html
    assert "overall_sample_coverage" in html
    assert "Coverage Break Days" in html
    assert "coverage_eval_included" in html
    assert "Warmup Break Days Excluded" in html
    assert "Yearly Factor Stability Heatmap" in html
    assert "Yearly Verdict" in html
    assert "Signal: ${signalTag}" in html
    assert "Signal direction is" in html
    assert "Portfolio conversion is mixed" in html
    assert "Portfolio Conversion: " in html
    assert "OOS years show no clear deterioration" not in html
    assert "low_sample:" in html
    assert "Mean RankIC ↑" in html
    assert "IC>0 ↑" in html
    assert "Avg Asset Coverage ↑" in html
    assert "Est. Cost-adj. L/S IR ↑" in html
    assert "Max Drawdown ↓" in html
    assert "absForColor" in html
    assert "absDisplay" in html
    assert "universe_count" in html
    assert "scored_count" in html
    assert "missing_score_count" in html
    assert "filtered_count" in html
    assert "feature_nan_row_count" in html
    assert "asset_coverage_by_date" in unified_html
    assert "overviewCoverageTickDecimals" in unified_html
    assert "Sample Coverage" in unified_html
    assert "Avg Asset Coverage" in unified_html
    assert "Min Asset Coverage" in unified_html
    assert "Coverage Break Days" in unified_html
    assert "coverage_eval_included" in unified_html
    assert "decisionOverviewCoveragePoints" in unified_html
    assert "Coverage Break Events" in unified_html
    assert "Worst Coverage Date" in unified_html
    assert "Worst 5 Coverage Dates" in unified_html
    assert "First Break Date" in unified_html
    assert "Last Break Date" in unified_html
    assert "OOS Break Days" in unified_html
    assert "early break / warmup candidate" in unified_html
    assert "nonBreakCoveragePoints" in unified_html
    assert "coverageXDomainOptions" in unified_html
    assert "xDomainMin" in unified_html
    assert "Effective Dates" in unified_html
    assert "Avg Valid Assets" in unified_html
    assert "period:" in unified_html
    assert "break_flag:" in unified_html
    assert "break_reason:" in unified_html
    assert "valid_forward_return_count" in unified_html
    assert "Yearly Factor Stability Heatmap" in unified_html
    assert "Yearly Verdict" in unified_html
    assert "yearly-verdict-badges" in unified_html
    assert "Signal: ${signalTag}" in unified_html
    assert "Portfolio: ${portfolioTag}" in unified_html
    assert "Cost: Raw only" in unified_html
    assert "suggesting stable signal direction" in unified_html
    assert "Portfolio conversion is mixed" in unified_html
    assert "Portfolio Conversion: " in unified_html
    assert "OOS years show no clear deterioration" not in unified_html
    assert "low_sample:" in unified_html
    assert "Mean RankIC ↑" in unified_html
    assert "IC>0 ↑" in unified_html
    assert "Avg Asset Coverage ↑" in unified_html
    assert "Raw L/S IR by Year ↑" in unified_html
    assert "Est. Cost-adj. L/S IR ↑" in unified_html
    assert "Max Drawdown ↓" in unified_html
    assert "artifact-fast-diagnostics-row" in unified_html
    assert "artifact-fast-diagnostic-combo" in unified_html
    assert "renderFastDiagnosticsCoverageRow" in unified_html


def test_model_lab_spec_service_round_trip(tmp_path: Path) -> None:
    vault = _build_vault(tmp_path)
    svc = _make_service(tmp_path, vault)
    specs_dir = tmp_path / "configs" / "real_cases" / "model_factor"
    specs_dir.mkdir(parents=True, exist_ok=True)
    spec_path = write_demo_model_factor_case(specs_dir, factor_name="web_model_lab_alpha")
    renamed = spec_path.with_name("web_model_lab_alpha.yaml")
    spec_path.rename(renamed)

    specs = svc.list_model_lab_specs()
    matching = [item for item in specs if item["name"] == "web_model_lab_alpha.yaml"]
    assert len(matching) == 1
    listed = matching[0]
    assert listed["version"] == 1
    assert listed["copied_from"] == ""
    assert isinstance(listed["file_signature"], str)
    assert len(listed["file_signature"]) == 16
    assert listed["file_signature"].isalnum()
    assert listed["file_signature"] == listed["file_signature"].lower()

    payload = svc.read_model_lab_spec("web_model_lab_alpha.yaml")
    assert "feature_columns" in payload["content"]
    assert payload["version"] == 1
    assert payload["copied_from"] == ""
    assert isinstance(payload["file_signature"], str)
    assert len(payload["file_signature"]) == 16
    assert re.match(r"\d{4}-\d{2}-\d{2}T", str(payload["meta"]["updated_at_utc"]))
    assert payload["meta"]["model_family"] == "ridge"
    assert payload["meta"]["version"] == 1
    assert payload["meta"]["copied_from"] == ""

    updated_content = str(payload["content"]).replace("alpha: 1.0", "alpha: 2.5")
    result = svc.update_model_lab_spec(
        "web_model_lab_alpha.yaml",
        {"content": updated_content},
    )
    assert result["ok"] is True
    saved_text = renamed.read_text(encoding="utf-8")
    assert "alpha: 2.5" in saved_text


def test_model_lab_submit_run_enqueues_model_factor_task(tmp_path: Path) -> None:
    vault = _build_vault(tmp_path)
    svc = _make_service(tmp_path, vault)
    specs_dir = tmp_path / "configs" / "real_cases" / "model_factor"
    specs_dir.mkdir(parents=True, exist_ok=True)
    spec_path = write_demo_model_factor_case(specs_dir, factor_name="web_model_lab_run")
    renamed = spec_path.with_name("web_model_lab_run.yaml")
    spec_path.rename(renamed)

    record = svc.submit_model_lab_run({"spec_name": "web_model_lab_run.yaml"})

    assert record["workflow"] == "model_factor"
    assert record["project_slug"] == "__model_lab__"
    assert record["case_name"] == "demo_web_model_lab_run_model_factor"


def test_model_lab_submit_run_preflight_rejects_invalid_split_before_enqueue(
    tmp_path: Path,
) -> None:
    vault = _build_vault(tmp_path)
    svc = _make_service(tmp_path, vault)
    specs_dir = tmp_path / "configs" / "real_cases" / "model_factor"
    specs_dir.mkdir(parents=True, exist_ok=True)
    spec_path = write_demo_model_factor_case(
        specs_dir,
        factor_name="web_model_lab_short_split",
        n_days=120,
    )
    renamed = spec_path.with_name("web_model_lab_short_split.yaml")
    spec_path.rename(renamed)

    with pytest.raises(AlphaLabIOError, match="strict split 启动前检查失败"):
        svc.submit_model_lab_run({"spec_name": "web_model_lab_short_split.yaml"})

    assert svc.run_store.list_records(workflow="model_factor") == []


def test_alpha_lab_submit_run_preflight_rejects_invalid_split_before_enqueue(
    tmp_path: Path,
) -> None:
    vault = _build_vault(tmp_path)
    svc = _make_service(tmp_path, vault)
    project_dir = vault / "55_projects" / "split_project"
    project_dir.mkdir(parents=True, exist_ok=True)
    project_yaml = {
        "slug": "split_project",
        "title_zh": "Split Project",
        "category": "factor",
        "owner": "test",
        "market": "ashare",
        "frequency": "daily",
        "chatgpt_project_name": "Split Project",
        "alpha_lab_defaults": {"evaluation_profile": "default_research"},
    }
    (project_dir / "project.yaml").write_text(
        yaml.safe_dump(project_yaml, sort_keys=False),
        encoding="utf-8",
    )
    spec_path = write_demo_single_factor_case(
        project_dir,
        factor_name="web_alpha_short_split",
        n_days=120,
    )
    spec_payload = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    (project_dir / "current_case.yaml").write_text(
        yaml.safe_dump(spec_payload, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(AlphaLabIOError, match="strict split 启动前检查失败"):
        svc.submit_run("split_project", {"case_name": spec_payload["name"]})

    assert svc.run_store.list_records(project_slug="split_project") == []


def test_model_lab_spec_duplicate_creates_version_chain(tmp_path: Path) -> None:
    vault = _build_vault(tmp_path)
    svc = _make_service(tmp_path, vault)
    specs_dir = tmp_path / "configs" / "real_cases" / "model_factor"
    specs_dir.mkdir(parents=True, exist_ok=True)
    spec_path = write_demo_model_factor_case(specs_dir, factor_name="web_model_lab_versioned")
    renamed = spec_path.with_name("web_model_lab_versioned.yaml")
    spec_path.rename(renamed)

    source_payload = yaml.safe_load(renamed.read_text(encoding="utf-8"))
    assert isinstance(source_payload, dict)
    source_payload["version"] = 7
    source_payload["lineage"] = {"history": [{"event": "seed", "at": "base"}]}
    renamed.write_text(
        yaml.safe_dump(source_payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    duplicated = svc.duplicate_model_lab_spec(
        "web_model_lab_versioned.yaml",
        {"target_name": "stock_versioned_copy.yaml"},
    )
    assert duplicated["ok"] is True
    first_payload = svc.read_model_lab_spec(duplicated["name"])
    first_body = yaml.safe_load(first_payload["content"])
    assert first_body["name"] == "stock_versioned_copy"
    assert first_body["factor_name"] == "stock_versioned_copy"
    assert first_body["version"] == 8
    assert first_body["copied_from"] == "web_model_lab_versioned.yaml"
    assert isinstance(first_body["lineage"], dict)
    assert first_body["lineage"]["copied_from"] == "web_model_lab_versioned.yaml"
    assert first_body["lineage"]["source_version"] == "7"

    duplicated_again = svc.duplicate_model_lab_spec(
        duplicated["name"],
        {"target_name": "stock_versioned_copy2.yaml"},
    )
    second_body = yaml.safe_load(
        svc.read_model_lab_spec(duplicated_again["name"])["content"],
    )
    assert second_body["name"] == "stock_versioned_copy2"
    assert second_body["factor_name"] == "stock_versioned_copy2"
    assert second_body["version"] == 9


def test_model_lab_submit_run_preflight_missing_inputs_fails_early(tmp_path: Path) -> None:
    vault = _build_vault(tmp_path)
    svc = _make_service(tmp_path, vault)
    specs_dir = tmp_path / "configs" / "real_cases" / "model_factor"
    specs_dir.mkdir(parents=True, exist_ok=True)
    spec_path = specs_dir / "missing_inputs.yaml"
    spec_path.write_text(
        "\n".join(
            [
                "name: missing_inputs_case",
                "factor_name: missing_inputs_factor",
                "features_path: ./missing_features.csv",
                "feature_columns: [feature_a]",
                "prices_path: ./missing_prices.csv",
                "rebalance_frequency: W",
                "n_quantiles: 5",
                "direction: long",
                "universe: {name: default}",
                "target: {kind: forward_return, horizon: 5}",
                "feature_preprocess: {missing_policy: median_impute, scale_features: auto}",
                "model: {family: ridge, params: {alpha: 1.0}}",
                "training:",
                "  window_type: rolling",
                "  train_window_n_dates: 60",
                "  min_train_dates: 40",
                "  min_train_rows: 200",
                "  retrain_every_n_dates: 5",
                "  min_score_assets: 5",
                "neutralization: {enabled: false}",
                "transaction_cost: {one_way_rate: 0.001}",
                "output: {root_dir: ./outputs}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(Exception, match="启动前检查失败|does not exist"):
        svc.submit_model_lab_run({"spec_name": "missing_inputs.yaml"})


def test_model_lab_compare_runs_returns_metric_and_leakage_payload(tmp_path: Path) -> None:
    vault = _build_vault(tmp_path)
    svc = _make_service(tmp_path, vault)

    _inject_model_lab_run_for_compare(
        svc=svc,
        tmp_path=tmp_path,
        run_id="mrun_a",
        case_name="case_a",
        factor_name="factor_a",
        model_family="ridge",
        metrics={
            "mean_ic": 0.012,
            "ic_ir": 0.31,
            "mean_rank_ic": 0.010,
            "rank_ic_ir": 0.28,
            "mean_long_short_turnover": 0.62,
            "long_short_ir": 0.35,
        },
        rank_ic_values=[0.01, 0.02, 0.03],
        top_features=["f1", "f2", "f3", "f4", "f5", "f6"],
        integrity={
            "n_checks": 3,
            "n_pass": 1,
            "n_warn": 1,
            "n_fail": 1,
            "highest_severity": "warn",
        },
    )
    _inject_model_lab_run_for_compare(
        svc=svc,
        tmp_path=tmp_path,
        run_id="mrun_b",
        case_name="case_b",
        factor_name="factor_b",
        model_family="gbdt",
        metrics={
            "mean_ic": 0.018,
            "ic_ir": 0.47,
            "mean_rank_ic": 0.013,
            "rank_ic_ir": 0.41,
            "mean_long_short_turnover": 0.55,
            "long_short_ir": 0.30,
        },
        rank_ic_values=[0.03, 0.01, -0.01],
        top_features=["f1", "f7", "f3", "f8", "f9", "f10"],
        integrity={
            "n_checks": 2,
            "n_pass": 2,
            "n_warn": 0,
            "n_fail": 0,
            "highest_severity": "pass",
        },
    )

    result = svc.compare_model_lab_runs(
        {
            "run_ids": ["mrun_a", "mrun_b", "mrun_a"],
            "top_k_features": 4,
        }
    )
    assert result["ok"] is True
    assert result["run_count"] == 2
    assert result["requested_run_count"] == 3
    assert len(result["run_ids"]) == 2
    assert result["case_names"] == ["case_a", "case_b"]
    assert result["case_name_by_run_id"] == {"mrun_a": "case_a", "mrun_b": "case_b"}
    assert {"mean_ic", "ic_ir", "mean_rank_ic"}.issubset(set(result["metric_columns"]))
    assert len(result["metric_rows"]) == 2
    assert all("model_family" in row for row in result["metric_rows"])
    assert result["leakage"]["top_k_features"] == 4
    assert len(result["leakage"]["runs"]) == 2
    assert result["feature_stability"]["pair_count"] == 1
    assert result["feature_stability"]["mean_jaccard"] is not None
    assert result["spec_diff"]["status"] == "ok"
    assert result["spec_diff"]["left"] == "case_a.yaml"
    assert result["spec_diff"]["right"] == "case_b.yaml"
    assert result["spec_diff"]["has_difference"] is True
    assert result["ic_series"]
    assert result["turnover_series"]
    assert result["leakage"]["severity_by_run"] == {"mrun_a": "warn", "mrun_b": "pass"}
    assert isinstance(result["run_failures"], list)
    assert len(result["run_failures"]) == 2
    assert all(item["run_id"] in {"mrun_a", "mrun_b"} for item in result["run_failures"])
    assert all(item["case_name"] in {"case_a", "case_b"} for item in result["run_failures"])


def test_model_lab_compare_runs_supports_eight_runs(tmp_path: Path) -> None:
    vault = _build_vault(tmp_path)
    svc = _make_service(tmp_path, vault)

    run_ids = [f"mrun_p{idx}" for idx in range(8)]
    for idx, run_id in enumerate(run_ids):
        _inject_model_lab_run_for_compare(
            svc=svc,
            tmp_path=tmp_path,
            run_id=run_id,
            case_name=f"case_{idx}",
            factor_name=f"factor_{idx}",
            model_family="ridge" if idx % 2 == 0 else "gbdt",
            metrics={
                "mean_ic": 0.01 + idx * 0.001,
                "ic_ir": 0.3 + idx * 0.01,
                "mean_rank_ic": 0.01,
                "rank_ic_ir": 0.25,
                "mean_long_short_turnover": 0.5,
                "long_short_ir": 0.28,
            },
            rank_ic_values=[0.01, 0.02, 0.03],
            top_features=[f"feat_{idx}_a", "feat_shared", f"feat_{idx}_c"],
            integrity={
                "n_checks": 1,
                "n_pass": 1,
                "n_warn": 0,
                "n_fail": 0,
                "highest_severity": "pass",
            },
        )

    result = svc.compare_model_lab_runs(
        {"run_ids": run_ids, "top_k_features": 3}
    )
    assert result["ok"] is True
    assert result["run_count"] == 8
    assert result["run_ids"] == run_ids  # order preserved after concurrent I/O
    assert len(result["metric_rows"]) == 8
    assert [row["run_id"] for row in result["metric_rows"]] == run_ids
    # N*(N-1)/2 pairs for Jaccard
    assert result["feature_stability"]["pair_count"] == 28
    assert len(result["leakage"]["runs"]) == 8
    assert result["spec_diff"]["status"] == "requires_two_runs"


def test_model_lab_compare_runs_rejects_more_than_eight(tmp_path: Path) -> None:
    vault = _build_vault(tmp_path)
    svc = _make_service(tmp_path, vault)

    # Only 2 real runs needed; the payload itself is what triggers the limit check.
    _inject_model_lab_run_for_compare(
        svc=svc,
        tmp_path=tmp_path,
        run_id="mrun_limit_a",
        case_name="case_a",
        factor_name="factor_a",
        model_family="ridge",
        metrics={"mean_ic": 0.01},
        rank_ic_values=[0.01],
        top_features=["f1"],
        integrity={
            "n_checks": 1,
            "n_pass": 1,
            "n_warn": 0,
            "n_fail": 0,
            "highest_severity": "pass",
        },
    )

    too_many = [f"mrun_limit_{idx}" for idx in range(9)]
    with pytest.raises(ValueError, match="最多支持 8"):
        svc.compare_model_lab_runs({"run_ids": too_many})


def test_model_lab_compare_runs_exposes_failed_run_snapshot(tmp_path: Path) -> None:
    vault = _build_vault(tmp_path)
    svc = _make_service(tmp_path, vault)

    _inject_model_lab_run_for_compare(
        svc=svc,
        tmp_path=tmp_path,
        run_id="mrun_failed",
        case_name="case_fail",
        factor_name="factor_fail",
        model_family="ridge",
        metrics={"mean_ic": 0.0},
        rank_ic_values=[0.0],
        top_features=["f1", "f2", "f3", "f4", "f5"],
        integrity={
            "n_checks": 1,
            "n_pass": 0,
            "n_warn": 0,
            "n_fail": 1,
            "highest_severity": "fail",
        },
    )
    _inject_model_lab_run_for_compare(
        svc=svc,
        tmp_path=tmp_path,
        run_id="mrun_ok",
        case_name="case_ok",
        factor_name="factor_ok",
        model_family="gbdt",
        metrics={"mean_ic": 0.01},
        rank_ic_values=[0.01],
        top_features=["f1", "f2", "f3", "f4", "f5"],
        integrity={
            "n_checks": 1,
            "n_pass": 1,
            "n_warn": 0,
            "n_fail": 0,
            "highest_severity": "pass",
        },
    )
    with svc.run_store._lock:  # noqa: SLF001 - tests intentionally seed in-memory store
        failed = svc.run_store._records["mrun_failed"]
        failed_record = failed.clone()
        failed_record.status = "failed"
        failed_record.error_type = "ValueError"
        failed_record.error_message = "dummy failure"
        failed_record.error_hint = "please retry"
        failed_record.error = "traceback lines..."
        failed_record.summary["error_type"] = "ValueError"
        svc.run_store._records["mrun_failed"] = failed_record

    result = svc.compare_model_lab_runs(
        {
            "run_ids": ["mrun_failed", "mrun_ok"],
            "top_k_features": 3,
        }
    )
    assert result["ok"] is True
    failures = result["run_failures"]
    assert isinstance(failures, list) and len(failures) == 2
    failed_snapshot = next(item for item in failures if item["run_id"] == "mrun_failed")
    assert failed_snapshot["status"] == "failed"
    assert failed_snapshot["error_type"] == "ValueError"
    assert failed_snapshot["error_message"] == "dummy failure"
    assert failed_snapshot["error_hint"] == "please retry"
    assert failed_snapshot["error"] == "traceback lines..."
    assert failed_snapshot["has_error"] is True


def test_model_lab_list_model_lab_runs_supports_filters(tmp_path: Path) -> None:
    vault = _build_vault(tmp_path)
    svc = _make_service(tmp_path, vault)

    _inject_model_lab_run_for_compare(
        svc=svc,
        tmp_path=tmp_path,
        run_id="mrun_filter_1",
        case_name="case_momentum",
        factor_name="factor_momentum",
        model_family="ridge",
        metrics={"mean_ic": 0.01},
        rank_ic_values=[0.01, 0.02],
        top_features=["feat_a", "feat_b", "feat_c"],
        integrity={
            "n_checks": 1,
            "n_pass": 1,
            "n_warn": 0,
            "n_fail": 0,
            "highest_severity": "pass",
        },
    )
    _inject_model_lab_run_for_compare(
        svc=svc,
        tmp_path=tmp_path,
        run_id="mrun_filter_2",
        case_name="case_value",
        factor_name="factor_value",
        model_family="gbdt",
        metrics={"mean_ic": 0.02},
        rank_ic_values=[0.01, -0.01],
        top_features=["feat_a", "feat_b", "feat_c"],
        integrity={
            "n_checks": 1,
            "n_pass": 1,
            "n_warn": 0,
            "n_fail": 0,
            "highest_severity": "pass",
        },
    )
    _inject_model_lab_run_for_compare(
        svc=svc,
        tmp_path=tmp_path,
        run_id="mrun_filter_3",
        case_name="case_momentum_alt",
        factor_name="factor_turnover",
        model_family="elastic_net",
        metrics={"mean_ic": 0.005},
        rank_ic_values=[0.02, 0.03],
        top_features=["feat_a", "feat_b", "feat_c"],
        integrity={
            "n_checks": 1,
            "n_pass": 1,
            "n_warn": 0,
            "n_fail": 0,
            "highest_severity": "pass",
        },
    )
    with svc.run_store._lock:  # noqa: SLF001
        first = svc.run_store._records["mrun_filter_1"]
        first_with_note = first.clone()
        first_with_note.note = "alpha candidate"
        svc.run_store._records["mrun_filter_1"] = first_with_note

        second = svc.run_store._records["mrun_filter_2"]
        second_with_note = second.clone()
        second_with_note.note = "beta note"
        svc.run_store._records["mrun_filter_2"] = second_with_note

        third = svc.run_store._records["mrun_filter_3"]
        third_with_note = third.clone()
        third_with_note.note = "momentum-focused"
        svc.run_store._records["mrun_filter_3"] = third_with_note

    with svc.run_store._lock:  # noqa: SLF001
        current = svc.run_store._records["mrun_filter_2"]
        failed = current.clone()
        failed.status = "failed"
        svc.run_store._records["mrun_filter_2"] = failed

    all_runs = svc.list_model_lab_runs(compact=True)
    assert len(all_runs) >= 3

    failed_runs = svc.list_model_lab_runs(compact=True, status_filter="failed")
    assert len(failed_runs) == 1
    assert failed_runs[0]["run_id"] == "mrun_filter_2"
    assert failed_runs[0]["factor_name"] == "factor_value"
    assert failed_runs[0]["evaluation_title"] == "failed"

    case_runs = svc.list_model_lab_runs(compact=True, case_filter="momentum")
    assert len(case_runs) >= 2
    assert {str(item.get("run_id")) for item in case_runs} == {"mrun_filter_1", "mrun_filter_3"}
    case_run_by_id = {str(item.get("run_id")): item for item in case_runs}
    first_summary = case_run_by_id["mrun_filter_1"]["summary"]
    assert first_summary["model_family"] == "ridge"
    assert first_summary["mean_ic"] == 0.01
    assert case_run_by_id["mrun_filter_1"]["factor_name"] == "factor_momentum"
    assert case_run_by_id["mrun_filter_1"]["evaluation_title"] == "completed"

    note_runs = svc.list_model_lab_runs(compact=True, note_filter="beta")
    assert len(note_runs) == 1
    assert note_runs[0]["run_id"] == "mrun_filter_2"


def test_export_model_lab_run_experiment_card_invokes_vault_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vault = _build_vault(tmp_path)
    svc = _make_service(tmp_path, vault)
    _inject_model_lab_run_for_compare(
        svc=svc,
        tmp_path=tmp_path,
        run_id="mrun_export_ok",
        case_name="case_export",
        factor_name="factor_export",
        model_family="ridge",
        metrics={"mean_ic": 0.02},
        rank_ic_values=[0.02, 0.03],
        top_features=["feat_a", "feat_b", "feat_c"],
        integrity={
            "n_checks": 1,
            "n_pass": 1,
            "n_warn": 0,
            "n_fail": 0,
            "highest_severity": "pass",
        },
    )
    with svc.run_store._lock:  # noqa: SLF001
        export_record = svc.run_store._records["mrun_export_ok"]
        export_record_with_note = export_record.clone()
        export_record_with_note.note = "for export"
        svc.run_store._records["mrun_export_ok"] = export_record_with_note
    output_dir = tmp_path / "run-mrun_export_ok"
    (output_dir / "experiment_card.md").write_text("# experiment card", encoding="utf-8")
    (output_dir / "summary.md").write_text("summary", encoding="utf-8")
    (output_dir / "run_manifest.json").write_text('{"case_name": "case_export"}', encoding="utf-8")

    observed: dict[str, object] = {}

    def _fake_export_to_vault(
        *,
        source_paths: dict[str, Path],
        case_name: str,
        vault_root: Path,
        mode: str = "versioned",
    ) -> Any:
        observed["source_paths"] = source_paths
        observed["case_name"] = case_name
        observed["vault_root"] = vault_root
        observed["mode"] = mode
        return SimpleNamespace(
            status="success",
            success=True,
            target_paths=[vault / "50_experiments" / "exported.md"],
            mode_used=mode,
            error=None,
        )

    monkeypatch.setattr("alpha_lab.web_unified._service.export_to_vault", _fake_export_to_vault)

    result = svc.export_model_lab_run_experiment_card(run_id="mrun_export_ok", mode="versioned")

    assert result["ok"] is True
    assert result["success"] is True
    assert result["status"] == "success"
    assert result["case_name"] == "case_export"
    assert result["mode_used"] == "versioned"
    target_paths = [str(item) for item in result["target_paths"]]
    expected_target_paths = [str(vault / "50_experiments" / "exported.md")]
    assert target_paths == expected_target_paths
    assert isinstance(observed["source_paths"], dict)
    assert "experiment_card_path" in observed["source_paths"]
    assert str(observed["vault_root"]) == str(vault)


def test_model_lab_duplicate_spec_sanitizes_name(tmp_path: Path) -> None:
    vault = _build_vault(tmp_path)
    svc = _make_service(tmp_path, vault)
    specs_dir = tmp_path / "configs" / "real_cases" / "model_factor"
    specs_dir.mkdir(parents=True, exist_ok=True)
    spec_path = write_demo_model_factor_case(specs_dir, factor_name="web_model_lab_copy")
    renamed = spec_path.with_name("web_model_lab_copy.yaml")
    spec_path.rename(renamed)

    duplicated = svc.duplicate_model_lab_spec(
        "web_model_lab_copy.yaml",
        {"target_name": "  ./subdir/../bad name.yml  "},
    )
    assert duplicated["ok"] is True
    assert duplicated["overwrite"] is False
    assert duplicated["name"].endswith(".yml")

    specs = svc.list_model_lab_specs()
    assert any(item["name"] == duplicated["name"] for item in specs)
    duplicated_payload = yaml.safe_load(svc.read_model_lab_spec(duplicated["name"])["content"])
    assert duplicated_payload["name"] == Path(duplicated["name"]).stem
    assert isinstance(duplicated_payload["factor_name"], str)
    assert duplicated_payload["factor_name"]

    duplicated_again = svc.duplicate_model_lab_spec(
        "web_model_lab_copy.yaml",
        {"target_name": duplicated["name"], "overwrite": True},
    )
    assert duplicated_again["overwrite"] is True


def test_model_lab_diff_spec_reports_unified_diff(tmp_path: Path) -> None:
    vault = _build_vault(tmp_path)
    svc = _make_service(tmp_path, vault)
    specs_dir = tmp_path / "configs" / "real_cases" / "model_factor"
    specs_dir.mkdir(parents=True, exist_ok=True)
    spec_path_left = write_demo_model_factor_case(
        specs_dir, factor_name="web_model_lab_diff_left"
    )
    spec_path_right = write_demo_model_factor_case(
        specs_dir, factor_name="web_model_lab_diff_right"
    )
    left = spec_path_left.with_name("web_model_lab_diff_left.yaml")
    right = spec_path_right.with_name("web_model_lab_diff_right.yaml")
    spec_path_left.rename(left)
    spec_path_right.rename(right)
    right.write_text(right.read_text() + "\nnew_key: 1\n", encoding="utf-8")

    result = svc.diff_model_lab_specs({"left": left.name, "right": right.name})
    assert result["ok"] is True
    assert result["left"] == left.name
    assert result["right"] == right.name
    assert result["has_difference"] is True
    assert result["unified"]


def test_model_lab_diff_spec_ignores_copy_metadata_by_default(tmp_path: Path) -> None:
    vault = _build_vault(tmp_path)
    svc = _make_service(tmp_path, vault)
    specs_dir = tmp_path / "configs" / "real_cases" / "model_factor"
    specs_dir.mkdir(parents=True, exist_ok=True)

    source = write_demo_model_factor_case(specs_dir, factor_name="web_model_lab_diff_meta")
    source_yaml = source.with_name("web_model_lab_diff_meta.yaml")
    source.rename(source_yaml)

    copied = svc.duplicate_model_lab_spec(
        source_yaml.name,
        {
            "target_name": "stock_diff_meta_copy.yaml",
            "sync_identifiers": False,
            "sync_factor_name": False,
        },
    )
    assert copied["ok"] is True

    result = svc.diff_model_lab_specs({"left": source_yaml.name, "right": copied["name"]})
    assert result["ok"] is True
    assert result["semantic_equal_ignoring_metadata"] is True
    assert result["ignore_metadata"] is True
    assert result["has_difference"] is False

    raw_result = svc.diff_model_lab_specs(
        {
            "left": source_yaml.name,
            "right": copied["name"],
            "ignore_metadata": False,
        }
    )
    assert raw_result["ok"] is True
    assert raw_result["has_difference"] is True


def test_model_lab_source_service_reads_curated_source(tmp_path: Path) -> None:
    vault = _build_vault(tmp_path)
    svc = _make_service(tmp_path, vault)

    sources = svc.list_model_lab_sources()
    assert any(item["key"] == "core" for item in sources)

    payload = svc.read_model_lab_source("core")
    assert payload["key"] == "core"
    assert "build_model_factor" in payload["content"]
    assert payload["line_count"] > 0


def test_model_lab_source_list_tolerates_missing_registered_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vault = _build_vault(tmp_path)
    svc = _make_service(tmp_path, vault)
    monkeypatch.setattr(
        "alpha_lab.web_unified._service._MODEL_LAB_SOURCE_SPECS",
        (
            {
                "key": "missing_source",
                "label": "missing.py",
                "path": "src/alpha_lab/does_not_exist.py",
                "description": "Missing source should not crash source listing.",
                "focus": "n/a",
            },
        ),
    )

    sources = svc.list_model_lab_sources()

    assert sources == [
        {
            "key": "missing_source",
            "label": "missing.py",
            "path": str((tmp_path / "src/alpha_lab/does_not_exist.py").resolve()),
            "description": "Missing source should not crash source listing.",
            "focus": "n/a",
            "exists": False,
        }
    ]


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

    assert "IC显著性" in html
    assert "Factor Snapshot" in html
    assert "Quick Decision Metrics / 快速决策指标" in html
    assert "Quick Verdict / 快速结论" in html
    assert "quick-verdict-banner" in html
    assert ".run-table th:nth-child(6)" in html
    assert ".run-table .metrics-screening-kv" in html
    assert "metrics-screening-kv-compact" in html
    assert '<strong>${escHtml(label)}:</strong>' in html
    assert 'scope === "full" || scope === "overall"' in html
    assert "is-portfolio" in html
    assert "Raw metrics only; no transaction cost assumption." in html
    assert "5-day horizon stronger" in html
    assert "signal decays by 5d" in html
    assert "Coverage Summary" in html
    assert "Coverage Details / 覆盖率细节" in html
    assert "Main y-axis excludes break days; Min Coverage includes break days" in html
    assert "warmup-only breaks" in html
    assert "tearsheet-exec-summary" in html
    assert "Executive Summary · Core Metrics" in html
    assert "function renderAlphaLabOverviewFixturePayload(payload)" in html
    assert "function loadAlphaLabOverviewFixture(fixtureId)" in html
    assert "function alphaLabOverviewFixtureIdFromLocation()" in html
    assert "/api/dev/alpha-lab/overview-fixtures/" in html
    assert "/dev/alpha-lab/overview-fixture" in html
    assert "/dev/alpha-lab1/overview-fixture" in html
    assert "Verdict:" in html
    assert "Diagnostic Warning" in html
    assert "Trading / Data" in html
    assert "IC@5 / IC@1" in html
    assert "Monotonic Share" in html
    assert "extractNamedRatioFromText" in html
    assert "minmax(128px, 0.86fr) minmax(0, 1.14fr)" in html
    assert "Est. Cost-adj. Long-Short IR" in html
    assert "Raw L/S IR by Year ↑" in html
    assert "Factor Autocorr" in html
    assert "Core Decision Charts / 核心决策图" in html
    assert "tearsheet-core-decision-grid" in html
    assert "tearsheet-core-decision-item" in html
    assert 'wrapCoreDecisionChart(yearlyStabilityChart, "is-wide")' in html
    assert "观察信号稳定性与 OOS 衰减" in html
    assert "观察组合收益与估算成本影响" in html
    assert "观察跨年份稳定性" in html
    assert "观察分组单调性" in html
    assert "monotonic_share=" in html
    assert "Spearman=" in html
    assert "IS Q5-Q1=" in html
    assert "OOS Q5-Q1=" in html
    assert "OOS start" in html
    assert "Advanced IC Diagnostics" in html
    assert "Advanced IC Diagnostics / IC 高级诊断 · ${advancedIcSummaryTitle}" in html
    assert "RankIC<0 ${fmtRatioAsPercent" in html
    assert "Data Quality Diagnostics" in html
    assert "Coverage Break Events" in html
    assert "Main y-axis excludes break days from scaling; min coverage includes break days" in html
    assert "Break Days Count" in html
    assert "Worst Coverage Date" in html
    assert "Worst 5 Coverage Dates" in html
    assert "First Break Date" in html
    assert "Last Break Date" in html
    assert "OOS Break Days" in html
    assert "early break / warmup candidate" in html
    assert "nonBreakCoveragePoints" in html
    assert "coverageXDomainOptions" in html
    assert "Data Quality Diagnostics / 数据质量诊断 · ${dataQualityWarningCount} warnings" in html
    assert "Trading Diagnostics" in html
    assert "Trading Diagnostics / 交易诊断 · ${turnoverP95Title}, ${turnoverSpikeTitle}" in html
    assert "Reproducibility Artifacts" in html
    assert "Reproducibility Artifacts / 复现材料 · ${escHtml(artifactSummaryLabel)}" in html
    assert "Asset Coverage Time Series / 每日资产覆盖率" in html
    assert "Turnover Time Series / 换手率时序" in html
    assert "加载数据库卡片统计中" in html
    assert "总览加载失败" in html
    assert (
        'String((row && (row.group ?? row.quantile ?? row.bucket ?? row.group_id)) ?? "")'
        in html
    )
    assert (
        'String(row && (row.group ?? row.quantile ?? row.bucket ?? row.group_id) ?? "")'
        not in html
    )
    assert "tearsheet-compact-diagnostics" in html
    assert "tearsheet-diagnostic-summary-grid" in html
    assert "Avg Turnover" in html
    assert "P95 Turnover" in html
    assert "OOS P95 Turnover" in html
    assert "Turnover Spike Days" in html
    assert "Worst 20D Return" in html
    assert "computeWorstRollingReturn" in html
    assert "computeCalmarRatio" in html
    assert "RankIC median" in html
    assert "RankIC p25 / p75" in html
    assert "RankIC p5 / p95" in html
    assert "RankIC < 0" in html
    assert "Avg Q5 Holdings" in html
    assert "Avg Q1 Holdings" in html
    assert "Effective Breadth" in html
    assert "Missing Diagnostics" in html
    assert "<th>Module</th><th>Issue</th><th>Evidence</th><th>Impact</th><th>Action</th>" in html
    assert "inspect group_returns.csv / quantile_returns.csv" in html
    assert "Quantile Cumulative NAV" in html
    assert "Quantile NAV hidden" in html
    assert "<th>Notes</th>" in html
    assert "展开查看分位组累计净值和 underwater 曲线等组合端细节" in html
    assert "Permutation Null Distribution of RankIC（experimental，默认折叠）" in html
    assert "n_permutations=" in html
    assert "tearsheet-diagnostic-chart-flow" in html
    assert "Promotion Review Summary" in html
    assert "Hard Fails" in html
    assert "Soft Warnings" in html
    assert "Passed Checks" in html
    assert "Warning Checks" in html
    assert "Failed Checks" in html
    assert "Warning Categories" in html
    assert "Evidence / Reason" in html
    assert "Soft warning evidence" in html
    assert "Top warning reasons" in html
    assert "Promotion is not blocked by hard rules, but soft warnings require review." in html
    assert "Blocker Count" not in html
    assert "buildPromotionDiagnosticModel" in html
    assert "classifyPromotionBlocker" in html
    assert "详细 promotion check 表（默认折叠）" in html
    assert "tearsheet-cell-muted" in html
    assert "IC t-stat" in html
    assert "RankIC@5 / RankIC@1" in html
    assert "Cumulative RankIC" in html
    assert "Cumulative RankIC（默认折叠）" in html
    assert "tearsheetBuildHistogramChart" in html
    assert "IS RankIC" in html
    assert "OOS RankIC" in html
    assert "bin:" in html
    assert "count:" in html
    assert "mean" in html
    assert "median" in html
    assert "p5" in html
    assert "p95" in html
    assert "Rolling RankICIR / Rolling ICIR" in html
    assert "yReferenceLines" in html
    assert "IS mean" in html
    assert "OOS mean" in html
    assert "0 line, IS mean and OOS mean are shown as dashed references" in html
    assert "function buildRollingInformationRatioPoints" in html
    assert "buildRollingIrReferenceLines" in html
    assert "tearsheet-ic-side-box" in html
    assert "tearsheet-ic-distribution-box" in html
    assert "Magnitude Q1~Q5 表示按因子强度分层后的 RankIC" in html
    assert "small / large cross section 表示按每日横截面覆盖规模分组" in html
    assert "Auto interpretation" in html
    assert "RankIC peaks at lag" in html
    assert "RankIC decays quickly; signal is short-lived." in html
    assert "RankIC 时序（含累计）" not in html
    assert "Rolling IC" in html
    assert "IC Decay" in html
    assert "artifact-fast-diagnostics-row" in html
    assert "artifact-fast-diagnostic-combo" in html
    assert "renderFastDiagnosticsCoverageRow" in html
    assert "function fmtDrawdownAsPercent" in html
    assert "function resolveMaxDrawdownForDisplay" in html
    assert "backtestSummary.max_drawdown" in html
    assert "Raw vs Estimated Cost-adjusted Long-Short NAV" in html
    assert "Raw Long-Short NAV" in html
    assert "Estimated Cost-adjusted Long-Short NAV" in html
    assert "No transaction cost assumption; cost-adjusted metrics are hidden." in html
    assert "round-trip" in html
    assert "estimated cost-adjusted return = raw L/S return - turnover * cost_rate" in html
    assert (
        "Estimated by raw L/S return - turnover × cost_rate; "
        "not a full execution-level cost backtest"
    ) in html
    assert "not a full execution-level cost backtest" in html
    assert "period:" in html
    assert "raw long-short return" in html
    assert "estimated cost-adjusted long-short return" in html
    assert "estimated cost" in html
    assert "Raw Long-Short" in html
    assert "Estimated Cost-adjusted Long-Short" in html
    assert "Estimated Cost-adjusted L/S IR by Cost Rate" in html
    assert "Cost-aware" not in html
    assert "diagnostic pending" not in html
    assert "invalid return series" in html
    assert "computeOverviewReturnStatsFromReturns" in html
    assert "const width = 1040;" in html
    assert "const height = 400;" in html
    assert 'stroke-width="2.6" stroke-linejoin="round" stroke-linecap="round"' in html
    assert 'shape-rendering="geometricPrecision"' in html
    assert "因子自相关" in html
    assert "换手率时序" in html
    assert "回测窗口：" in html
    assert "artifact-overview-charts" in html
    assert "metrics-screening-grid" in html
    assert "metrics-screening-grid-fast" in html
    assert "metrics-screening-card" in html
    assert 'label: "Cost assumption"' not in html
    assert "Raw metrics only; no transaction cost assumption." in html
    assert "artifact-group-quick" in html
    assert "metrics-run-header-label" in html
    assert "quantile_returns" in html
    assert "loadRunOverviewSnapshot" in html
    assert "renderRunOverviewSection" in html
    assert "renderOverviewLineChart" in html
    assert "renderDualAxisLineChart" in html
    assert "compactYmd" in html
    assert "compactYm" in html
    assert "purged_kfold_summary" in html
    assert "purged_kfold_folds" in html
    assert "renderPurgedKfoldSummaryJson" in html
    assert "renderPurgedKfoldFoldsCsv" in html
    assert "renderPortfolioValidationMetricsJson" in html
    assert "renderStackedAreaChart" in html
    assert "堆叠面积" in html
    assert "快速筛选模式 · 固定 9 项核心指标" not in html


def test_index_html_factor_workshop_baseline_tables_are_constrained() -> None:
    html = _index_html_raw()

    assert "Research Baseline Suite / 研究基准因子套件" in html
    assert "factor-workshop-table-wrap" in html
    assert "factor-workshop-baseline-table" in html
    assert "overflow-x: auto" in html
    assert "table-layout: fixed" in html
    assert "overflow-wrap: anywhere" in html


def test_index_html_uses_stage0_distribute_without_legacy_idea_draft() -> None:
    html = _index_html_raw()

    assert 'id="stage0DistributePanel"' in html
    assert 'id="stage0IdeaInput"' in html
    assert 'id="btnStage0Distribute"' in html
    assert 'id="stage0Results"' in html
    assert "/api/vault/idea-distribute" in html
    assert "Stage 0: Idea Distribute (single_factor)" in html

    assert "Stage 0 (legacy): Idea Draft" not in html
    assert "想法探索器 / Idea Draft (legacy)" not in html
    assert "旧 ledger-style 输出" not in html
    assert 'id="btnExploreGenerateFactorDraft"' not in html
    assert 'id="exploreIdea"' not in html
    assert 'id="exploreMechanismIndexStatus"' not in html
    assert "/api/settings/mechanism-index" not in html
    assert "generateFactorDraftFromExplore" not in html
    assert 'id="exploreDraftAgentPanel"' not in html
    assert 'id="exploreDraftAgentPanelBody"' not in html
    assert "copyExploreDraftAgentPrompt" not in html
    assert 'api("/api/vault/idea-draft", "POST"' not in html
    assert "minimal_artifacts: true" not in html
    assert "compactRetrievalCardsForPrompt" not in html
    assert "isCompletePromptBullet" not in html
    assert "compactCrossCardSynthesisForPrompt" not in html
    assert "stage1GenerationMaterialText" not in html
    assert "ledger_v1.${model}.yaml" not in html

    assert "Knowledge Writeback Queue" in html
    assert 'id="kwStage2Body"' in html
    assert 'id="btnCreateStage2NoteDraft"' in html
    assert 'id="kwStage3Body"' in html
    assert 'id="btnCreateValidationDraft"' in html
    assert 'id="btnCreateFailureDraft"' in html
    assert "/api/vault/writeback-drafts" in html
    assert "experiment_note" in html
    assert "validation_report" in html
    assert "failure_observation" in html
    assert "summarizeRun" in html
    assert "EXECUTE_FORMAL_WRITEBACK" in html

    assert 'id="exploreDraftFiles"' not in html
    assert 'id="exploreDraftSharedPrompt"' not in html
    assert 'id="exploreDraftModelLedgers"' not in html
    assert 'id="exploreDraftReconcile"' not in html
    assert 'id="exploreDraftFinalLedger"' not in html
    assert 'id="btnExploreIdea"' not in html
    assert 'id="btnCopyExplorePrompt"' not in html
    assert 'id="exploreRecordPanel"' not in html
    assert 'id="exploreResponseText"' not in html
    assert 'id="btnRecordExploreResponse"' not in html
    assert 'id="exploreSessionList"' not in html
    assert 'id="exploreQuickSessions"' not in html
    assert 'id="btnRefreshExploreSessions"' not in html
    assert "/api/vault/explore-idea" not in html
    assert "/api/vault/record-explore-response" not in html
    assert "/api/vault/explore-sessions/" not in html
    assert "openExploreSession" not in html
    assert "deleteExploreSession" not in html
    assert "renderExploreLintReport" not in html
    assert "setExploreRecordSession" not in html
    assert "copyExploreDraftReconcilePrompt" not in html


def test_index_html_has_no_frontend_idea_draft_route() -> None:
    html = _index_html_raw()

    assert "/api/vault/idea-draft" not in html
    assert "generateIdeaDraftFromExplore" not in html
    assert "generateFactorDraftFromExplore" not in html
    assert "renderExploreDraftAgentPanel" not in html
    assert "copyExploreDraftAgentPrompt" not in html
    assert "const promptBox = $(\"explorePromptBox\");" not in html
    assert "previewExploreCard" not in html


def test_index_html_removes_legacy_explore_stage_defaults() -> None:
    html = _index_html_raw()

    assert 'name="exploreMode"' not in html
    assert 'name="exploreStage"' not in html
    assert 'id="exploreInjectDrift"' not in html
    assert 'id="explorePersistSession"' not in html
    assert "function autoExploreStage" not in html
    assert "function autoExploreMode" not in html
    assert "persist_session: true" not in html
    assert 'models: ["claude", "codex"]' not in html


def test_index_html_escapes_script_close_tag_in_print_template() -> None:
    html = _index_html_raw()
    start = html.find("function buildMetricsPrintDocument")
    end = html.find("function exportMetricsReportPdf")

    assert "function renderMetricsExportActions" in html
    assert 'data-action="exportMetricsReportPdf"' in html
    assert "renderMetricsExportActions(run)" in html
    assert '$("metricsRunArtifacts").innerHTML = fastScreen' in html
    assert start >= 0
    assert end > start

    snippet = html[start:end]
    assert "<\\/script>" in snippet
    assert "</script>" not in snippet
    assert "Alpha-Lab 单因子快速评价报告" in snippet
    assert "Alpha-Lab 全面评价报告" in snippet
    assert "break-inside: avoid" in snippet
    assert "grid-template-columns: repeat(3, minmax(0, 1fr))" in snippet
    assert "artifact-overview-charts" in snippet
    assert "tearsheet-ic-layout" in snippet
    assert "tearsheet-summary-grid" in snippet
    assert "tearsheet-diagnostic-summary-grid" in snippet
    assert "tearsheet-promotion-summary-grid" in snippet
    assert "tearsheet-metric-row" in snippet
    assert "grid-template-columns: repeat(3, minmax(0, 1fr)) !important" in snippet
    assert "grid-template-columns: repeat(5, minmax(0, 1fr)) !important" in snippet
    assert "grid-template-columns: repeat(2, minmax(0, 1fr)) !important" in snippet
    assert "@page { size: A4 landscape; margin: 10mm; }" in snippet
    assert "报告尚未加载完成。" in html
    assert "一键导出 PDF（原样）" in html
    assert "下载后端 PDF（兜底）" not in html


def test_model_lab_html_supports_original_pdf_export() -> None:
    html = _model_lab_html()
    start = html.find("function buildModelLabOverviewPrintDocument")
    end = html.find("function exportModelLabOverviewPdf")

    assert "btnExportModelLabOverviewPdf" in html
    assert "一键导出 PDF（原样）" in html
    assert 'data-action="export-model-lab-overview-pdf"' in html
    assert "setModelLabOverviewExportVisible(runId, true)" in html
    assert start >= 0
    assert end > start

    snippet = html[start:end]
    assert "<\\/script>" in snippet
    assert "</script>" not in snippet
    assert "overview-primary-grid" in snippet
    assert "overview-chart-grid" in snippet
    assert "overview-grouped-metrics" in snippet
    assert "coverage-summary-grid" in snippet
    assert "grid-template-columns: repeat(2, minmax(0, 1fr)) !important" in snippet
    assert "grid-template-columns: repeat(4, minmax(0, 1fr)) !important" in snippet
    assert "@page { size: A4 landscape; margin: 10mm; }" in snippet


def test_model_lab_html_omits_duplicate_claude_key_settings() -> None:
    html = _model_lab_html()

    assert 'id="claudeApiKeyInput"' not in html
    assert 'id="claudeApiUrlInput"' not in html
    assert 'id="researchBridgeV2Toggle"' not in html
    assert 'id="btnSaveClaudeKey"' not in html
    assert 'id="btnClearClaudeKey"' not in html
    assert "/api/settings/llm" not in html


def test_model_lab_html_is_backend_first_report_workspace() -> None:
    html = _model_lab_html()

    assert "后端迭代优先" in html
    assert "Validate + Run Full Report" in html
    assert "Validate + Run Screening" not in html
    assert "探索并生成双 Agent Prompt" not in html
    assert 'id="explorerIdea"' not in html
    assert 'id="btnIdeaExplore"' not in html
    assert 'id="explorerAgentPromptCards"' not in html
    assert 'id="explorerQuickSessions"' not in html
    assert 'id="explorerMechanismIndexStatus"' not in html
    assert 'id="explorerPrompt"' not in html
    assert 'id="explorerResponse"' not in html
    assert 'id="btnIdeaRecordResponse"' not in html
    assert 'id="btnIdeaCopyPrompt"' not in html
    assert 'id="btnIdeaApplyPatchHint"' not in html
    assert "共享 Prompt / 回灌响应 / Patch Hint" not in html


def test_model_lab_html_supports_natural_overview_expansion() -> None:
    html = _model_lab_html()

    assert "viewer-box--overview" in html
    assert 'viewer.classList.add("viewer-box--overview")' in html
    assert 'viewer.classList.remove("viewer-box--overview")' in html
    assert "Executive Verdict" in html
    assert "overview-grouped-metrics" in html
    assert "Trading / Data" in html
    assert "Promotion" in html
    assert "IC>0 占比" in html
    assert "Q5-Q1 / Group Monotonicity" in html
    assert "Est. Cost-adj. Long-Short IR" in html
    assert "Raw vs Estimated Cost-adjusted Long-Short NAV" in html
    assert (
        "Extreme NAV growth detected. Verify return unit, "
        "non-overlap compounding, and forward-return alignment."
    ) in html
    assert "function toGroupReturnDecimal(value)" in html
    assert "Quantile NAV hidden / 分位净值已隐藏" in html
    assert (
        "group_return appears cumulative or contains abnormal single-period returns"
        in html
    )
    assert "已跳过单步" not in html
    assert (
        "Main y-axis focuses the normal 95%~100% coverage range when applicable; "
        "break days are shown below"
    ) in html
    assert "Coverage Break Event Strip" in html
    assert (
        "RankIC increases with horizon; signal may be more effective at longer holding periods."
    ) in html
    assert "RankIC decays quickly; signal is short-lived." in html
    assert (
        "Default ${bps} bps estimated cost adjustment is available. "
        "Full cost sensitivity curve was skipped because Level-2 portfolio validation "
        "did not run / run was not promoted."
    ) in html
    assert "Optional Diagnostics / Missing Diagnostics" in html
    assert "Required artifact" in html
    assert "function renderRunOverviewPayload(payload, options = {})" in html
    assert "function loadRunOverviewFixture(fixtureId)" in html
    assert "function renderArtifactPayload(payload)" in html
    assert "function viewArtifactFixture(fixtureId, key)" in html
    assert "function loadDiagnosticsFixture(fixtureId)" in html
    assert "function overviewFixtureIdFromLocation()" in html
    assert "function artifactFixtureParamsFromLocation()" in html
    assert "function diagnosticsFixtureIdFromLocation()" in html
    assert "/api/dev/model-lab/overview-fixtures/" in html
    assert "/api/dev/model-lab/artifact-fixtures/" in html
    assert "/dev/model-lab/overview-fixture" in html
    assert "/dev/model-lab/artifact-fixture" in html
    assert "/dev/model-lab/diagnostics-fixture" in html
    assert "round-trip" in html
    assert "Cost-aware" not in html
    assert "period:" in html


def test_model_lab_html_renders_run_queue_progress_bar() -> None:
    html = _model_lab_html()

    assert "function renderRunProgress(run)" in html
    assert "run-progress-bar" in html
    assert "run-progress-fill" in html
    assert "最近阶段" in html


def test_model_lab_feature_stability_heatmap_uses_green_palette() -> None:
    html = _model_lab_html()

    assert "rgba(79, 107, 62, 0.12)" in html
    assert "lerp(0xf2, 0x4f)" in html
    assert "lerp(0xf6, 0x6b)" in html
    assert "lerp(0xeb, 0x3e)" in html
    assert "rgba(68, 119, 170, 0.10)" not in html


def test_model_lab_compare_timeseries_cards_are_extra_tall() -> None:
    html = _model_lab_html()

    assert ".chart-card.compare-timeseries-card svg" in html
    assert "height: 405px;" in html
    assert ".chart-grid.compare-timeseries-grid" in html
    assert 'id="compareTimeseriesGrid" class="chart-grid compare-timeseries-grid"' in html
    assert 'id="compareTimeseriesGrid" class="chart-grid chart-grid-stacked"' not in html
    assert 'preserveAspectRatio="none"' in html
    assert "const height = 495;" in html
    assert "function renderCompareSpecDiff(specDiff = null)" in html


def test_model_lab_compare_leakage_summary_uses_six_column_rows() -> None:
    html = _model_lab_html()

    assert ".leakage-summary-grid" in html
    assert "grid-template-columns: repeat(6, minmax(0, 1fr));" in html
    assert '<div class="leakage-summary-grid">' in html
    case_idx = html.index("<strong>case</strong>")
    status_idx = html.index("<strong>status</strong>", case_idx)
    severity_idx = html.index("<strong>严重度</strong>", status_idx)
    pass_idx = html.index("<strong>pass</strong>", severity_idx)
    assert case_idx < status_idx < severity_idx < pass_idx


def test_model_lab_overview_primary_chart_cards_are_extra_tall() -> None:
    html = _model_lab_html()

    assert ".overview-primary-grid .chart-card.overview-tall-chart svg" in html
    assert ".overview-primary-grid .chart-card svg" not in html
    assert "height: 390px;" in html
    assert (
        'const OVERVIEW_TALL_CHART_OPTIONS = { cardClass: "overview-tall-chart", height: 330 };'
        in html
    )
    assert html.count("...OVERVIEW_TALL_CHART_OPTIONS") >= 4
    assert "buildGroupReturnCards(groupRows, { ...OVERVIEW_TALL_CHART_OPTIONS" in html


def test_model_lab_html_samples_overview_nav_by_rebalance_frequency() -> None:
    html = _model_lab_html()

    assert "function resolveRebalanceStep(rebalanceFrequency)" in html
    assert "function resolveLabelHorizon(backtest = null)" in html
    assert "function buildLongShortNavRowsFromGroups(rows, options = {})" in html
    assert "const effectiveStep = Math.max(rebalanceStep, labelHorizon)" in html
    assert "sampleStep: effectiveStep" in html
    assert (
        "const overviewEffectiveStep = Math.max(overviewRebalanceStep, overviewLabelHorizon)"
        in html
    )
    assert "sampleStep: overviewEffectiveStep" in html
    assert "按可用日期逐日取点" in html
    assert (
        "buildEquityAndDrawdownCards(backtest, groupRows, { splitDate, turnoverRows, summary })"
        in html
    )
    assert "function buildOverviewSplitCoverageNotice(options = {})" in html
    assert "曲线 IS 段缺失" in html
    assert "曲线仅含 OOS，IS 时序缺失" in html
    assert "${buildOverviewSplitCoverageNotice({ splitDate, icRows, groupRows, backtest })}" in html
    assert (
        "split_phase: normalizeSplitPhase(row.split_phase || row.phase || row.sample_phase)"
        in html
    )
    assert "split_phase: phaseByDate.get(date) || \"\"" in html
    assert "function navRowsEquivalent(leftRows, rightRows, tolerance = 1e-9)" in html
    assert "function backtestNavIsQuantileLongShort(backtest = null)" in html
    assert "quantile_long_short_from_group_returns" in html
    assert "const backtestNavRows = parseBacktestNavRows(backtest)" in html
    assert "const navRows = backtestNavRows.length ? backtestNavRows : dailyNavRows" in html
    assert "多空累计净值（Q5-Q1" in html
    assert "backtest_result NAV=Q5-Q1 多空" in html
    assert "function buildLongShortSpreadCard(rows, options = {})" in html
    assert "includeGroups: false" in html
    assert "includeSpread: false" in html
    assert "suppressDuplicateBacktestNav: true" in html
    assert "buildLongShortSpreadCard(groupRows, { ...OVERVIEW_TALL_CHART_OPTIONS" in html
    assert "let spreadNav = 1.0" in html
    assert "opts.cumulative ? spreadNav * (1 + spread) : spread" in html


def test_model_lab_training_log_charts_use_scored_rows_and_independent_axes() -> None:
    html = _model_lab_html()

    assert "function normalizeTrainingLogRows(rows)" in html
    assert "function isScoredTrainingRow(row)" in html
    assert "Training Health Summary / 训练健康摘要" in html
    assert "Training Health:" in html
    assert "Warmup Skipped" in html
    assert "Post-fit Skipped" in html
    assert "Expected warmup skips, not a training failure." in html
    assert "训练状态时间线（默认按季度聚合）" in html
    assert "切换到月度聚合 / Monthly view" in html
    assert "dominant_status:" in html
    assert "first_fit_date" in html
    assert "模型版本时序" in html
    assert "median fit interval" in html
    assert "max fit gap" in html
    assert "latest version" in html
    assert "fit cadence appears regular" in html
    assert "oos_rank_ic" in html
    assert "悬停可见全部版本号" in html
    assert ".training-fixed-svg-scroll" in html
    assert ".chart-card svg.training-fixed-svg" in html
    assert 'class="training-fixed-svg" width="${W}" height="${H}"' in html
    assert "score/train（资产数 / 训练样本量，按百分比；只取 fit_scored、reused_scored 行）" in html
    assert "n_score_assets（每日打分资产数，warmup 期 skipped 行不绘制）" in html
    assert "No train_rows cliffs detected." in html
    assert "No score_assets cliffs detected." in html
    assert "score/train stable after warmup" in html
    assert "latest marker" in html
    assert "median" in html
    assert "p95" in html
    assert "universe filter change candidate" in html


def test_model_lab_html_deemphasizes_low_frequency_panels() -> None:
    html = _model_lab_html()

    assert "source-drawer-backdrop" in html
    assert 'id="btnOpenSourceDrawer"' in html
    assert "批量操作 / 快速筛选" in html
    assert "Structured Event List" in html
    assert "默认收起；Stage Timeline 为主视图" in html
    assert "Feature Importance Summary / 特征重要性摘要" in html
    assert "Feature Importance Verdict:" in html
    assert "Feature Group Importance / 特征组重要性" in html
    assert "Feature Importance Over Time / 特征重要性随模型版本变化" in html
    assert "cheap_ledger_only" in html
    assert "Zero-importance Features" in html
    assert "Recommended Follow-up Tests / 建议后续验证" in html
    assert "Purged KFold per-fold IC" in html


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


def test_create_case_passes_frontend_controls_and_builder_kwargs(tmp_path: Path) -> None:
    vault = _build_vault(tmp_path)
    svc = _make_service(tmp_path, vault)
    svc.create_project(
        {
            "slug": "test-short-factor",
            "title_zh": "空头因子测试项目",
            "category": "factor_family",
            "owner": "test",
            "market": "ashare",
            "frequency": "daily",
            "chatgpt_project_name": "Test Short Factor",
            "origin_cards": [],
        }
    )

    svc.create_case(
        "test-short-factor",
        {
            "case_name": "failed_cushion_short",
            "factor_name": "failed_cushion_short",
            "base_method": "downshock_failed_cushion_risk_v1",
            "direction": "short",
            "skip_recent": "1",
            "target_horizon": "10",
            "outside_event_policy": "zero",
            "exclude_limit": "true",
            "builder_kwargs_json": (
                '{"shock_gate_mode":"cs_quantile","shock_q":0.7,'
                '"outside_event_policy":"nan","neutralize_basic":true,'
                '"invert":false}'
            ),
        },
    )

    spec_path = vault / "55_projects" / "test-short-factor" / "current_case.md"
    spec_text = spec_path.read_text(encoding="utf-8")
    spec = yaml.safe_load(spec_text.split("```yaml", 1)[1].split("```", 1)[0])
    base = spec["factor_input"]["recipe"]["base"]
    assert spec["direction"] == "short"
    assert spec["target"]["horizon"] == 10
    assert base["method"] == "downshock_failed_cushion_risk_v1"
    assert base["skip_recent"] == 1
    assert base["shock_gate_mode"] == "cs_quantile"
    assert base["shock_q"] == pytest.approx(0.7)
    assert base["outside_event_policy"] == "zero"
    assert base["neutralize_basic"] is True
    assert base["invert"] is False
    assert base["exclude_limit"] is True


def test_list_cases_empty_for_unknown_project(tmp_path: Path) -> None:
    vault = _build_vault(tmp_path)
    svc = _make_service(tmp_path, vault)

    cases = svc.list_cases("nonexistent-project")
    assert cases == []


def test_get_project_truncates_large_documents_for_snapshot(tmp_path: Path) -> None:
    vault = _build_vault(tmp_path)
    svc = _make_service(tmp_path, vault)
    slug, _ = _create_project_and_case(svc)

    project_dir = vault / "55_projects" / slug
    large_block = ("# 标题\n\n" + ("很长的项目内容。\n" * 20000)).encode("utf-8")
    (project_dir / "decision_log.md").write_bytes(large_block)
    (project_dir / "runs").mkdir(parents=True, exist_ok=True)
    (project_dir / "runs" / "latest.md").write_bytes(large_block)

    result = svc.get_project(slug)

    assert result["project"]["slug"] == slug
    documents = result["documents"]
    assert "内容已截断" in documents["decision_log"]
    assert "内容已截断" in documents["latest_run"]
    assert len(documents["decision_log"]) < len(large_block.decode("utf-8"))
    assert len(documents["latest_run"]) < len(large_block.decode("utf-8"))



def test_create_writeback_draft_requires_review_then_applies(tmp_path: Path) -> None:
    vault = _build_vault(tmp_path)
    svc = _make_service(tmp_path, vault)
    slug, _ = _create_project_and_case(svc)

    draft = svc.create_writeback_draft(
        {
            "source_stage": "stage2",
            "card_type": "experiment_note",
            "project_slug": slug,
            "title": "Reviewed Synthesis",
            "body": "User-reviewed synthesis body.",
            "source_artifacts": ["artifacts/stage2.md"],
        }
    )

    assert draft["status"] == "pending"
    assert str(draft["target_path"]).startswith(
        f"55_projects/{slug}/30_research_notes/Note - "
    )
    summaries = svc.list_drafts(slug)
    summary = next(item for item in summaries if item["name"] == draft["draft_name"])
    assert summary["source_stage"] == "stage2"
    assert summary["card_type"] == "experiment_note"
    assert summary["target_path"] == draft["target_path"]
    assert summary["draft_type"] == "knowledge_writeback_draft"

    with pytest.raises(ValueError, match="approved"):
        svc.apply_draft(slug, str(draft["draft_name"]), {})

    svc.patch_draft(
        slug,
        str(draft["draft_name"]),
        {"review_status": "approved", "reviewed_by": "tester", "reviewed_at": "now"},
    )
    applied = svc.apply_draft(slug, str(draft["draft_name"]), {})

    assert applied["success"] is True
    target_paths = cast(list[str], applied["target_paths"])
    assert len(target_paths) == 1
    target_path = Path(target_paths[0])
    assert target_path.exists()
    content = target_path.read_text(encoding="utf-8")
    frontmatter = yaml.safe_load(content.split("---", 2)[1])
    assert frontmatter["type"] == "experiment_note"
    assert frontmatter["project"] == slug
    assert frontmatter["source_stage"] == "stage2"
    assert frontmatter["source_artifacts"] == ["artifacts/stage2.md"]
    assert "User-reviewed synthesis body." in content


def test_create_writeback_draft_rejects_target_path_escape(tmp_path: Path) -> None:
    vault = _build_vault(tmp_path)
    svc = _make_service(tmp_path, vault)
    slug, _ = _create_project_and_case(svc)
    payload: dict[str, object] = {
        "source_stage": "stage3",
        "card_type": "failure_observation",
        "project_slug": slug,
        "title": "Bad Path",
        "body": "Reviewed failure observation.",
    }

    with pytest.raises(PermissionError):
        svc.create_writeback_draft({**payload, "target_path_hint": "../bad.md"})
    with pytest.raises(PermissionError):
        svc.create_writeback_draft(
            {**payload, "target_path_hint": "55_projects/other-project/bad.md"}
        )
    with pytest.raises(PermissionError):
        svc.create_writeback_draft({**payload, "target_path_hint": "C:/tmp/bad.md"})
    with pytest.raises(ValueError):
        svc.create_writeback_draft(
            {
                **payload,
                "target_path_hint": f"55_projects/{slug}/45_failure_observations/bad.txt",
            }
        )

# ---------------------------------------------------------------------------
# CLI routing: web unified
# ---------------------------------------------------------------------------


def test_cli_routes_web_unified(monkeypatch: pytest.MonkeyPatch) -> None:

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

    result = svc.register_custom_factor(
        {
            "name": "test_vol",
            "code": _VALID_FACTOR_CODE,
            "description": "test volatility factor",
        }
    )

    assert result["registered"] is True
    assert result["name"] == "test_vol"
    # Verify it's in the registry
    from alpha_lab.factor_recipe import factor_registry

    assert "test_vol" in factor_registry
    # Verify persistence (subdir layout: research/<name>/factor.json)
    meta_path = tmp_path / "custom_factors" / "research" / "test_vol" / "factor.json"
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
    assert "baseline_factor_suite" in result
    baseline_names = [f["name"] for f in result["baseline_factor_suite"]]
    assert "mom_20d" in baseline_names
    assert "rev_5d" in baseline_names
    assert "vcimom_20_5" in baseline_names
    assert result["baseline_count"] >= 8
    assert result["total"] >= 5


def test_register_and_delete_custom_factor(tmp_path: Path) -> None:
    vault = _build_vault(tmp_path)
    svc = _make_service(tmp_path, vault)

    svc.register_custom_factor(
        {
            "name": "temp_factor",
            "code": _VALID_FACTOR_CODE,
        }
    )

    from alpha_lab.factor_recipe import factor_registry

    assert "temp_factor" in factor_registry

    svc.delete_custom_factor("temp_factor")
    assert "temp_factor" not in factor_registry
    meta_path = tmp_path / "custom_factors" / "research" / "temp_factor" / "factor.json"
    assert not meta_path.exists()
    # Empty parent dir should be cleaned up after delete.
    assert not meta_path.parent.exists()


def test_list_custom_factors_loads_research_subdir(tmp_path: Path) -> None:
    factor_dir = tmp_path / "custom_factors" / "research" / "nested_factor"
    factor_dir.mkdir(parents=True)
    factor_dir.joinpath("factor.json").write_text(
        json.dumps(
            {
                "name": "nested_factor",
                "description": "nested research factor",
                "code": _VALID_FACTOR_CODE,
                "created_at": "2024-01-01T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    factor_dir.joinpath("research_log.md").write_text(
        "# nested_factor research log\n", encoding="utf-8"
    )
    vault = _build_vault(tmp_path)
    svc = _make_service(tmp_path, vault)

    result = svc.list_custom_factors()

    names = [f["name"] for f in result["factors"]]
    assert "nested_factor" in names
    code = svc.get_custom_factor_code("nested_factor")
    assert code["description"] == "nested research factor"

    # Sibling research_log.md must not be loaded as a factor and must survive.
    assert factor_dir.joinpath("research_log.md").exists()

    from alpha_lab.factor_recipe import factor_registry

    factor_registry._builders.pop("nested_factor", None)


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

    svc.register_custom_factor(
        {
            "name": "view_test",
            "code": _VALID_FACTOR_CODE,
            "description": "viewable factor",
        }
    )

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

    svc1.register_custom_factor(
        {
            "name": "persist_test",
            "code": _VALID_FACTOR_CODE,
        }
    )

    from alpha_lab.factor_recipe import factor_registry

    # Remove from in-memory registry to simulate fresh start
    factor_registry._builders.pop("persist_test", None)
    assert "persist_test" not in factor_registry

    # Create new service — should reload from disk
    _make_service(tmp_path, vault)
    assert "persist_test" in factor_registry

    # Clean up
    factor_registry._builders.pop("persist_test", None)


def test_frontend_batch_parallel_config_prefers_process_mode() -> None:
    config = _build_frontend_batch_parallel_config(5)

    assert config.mode == "process"
    assert config.max_workers == 4
    assert config.factors_per_worker == 2


def test_model_lab_batch_worker_count_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ALPHA_LAB_MODEL_LAB_MAX_WORKERS", raising=False)
    monkeypatch.setattr("alpha_lab.web_unified._subprocess.os.cpu_count", lambda: 1)
    assert _build_model_lab_batch_worker_count(3) == 1

    monkeypatch.setattr("alpha_lab.web_unified._subprocess.os.cpu_count", lambda: 2)
    assert _build_model_lab_batch_worker_count(3) == 1

    monkeypatch.setattr("alpha_lab.web_unified._subprocess.os.cpu_count", lambda: 8)
    assert _build_model_lab_batch_worker_count(1) == 1
    assert _build_model_lab_batch_worker_count(2) == 1
    assert _build_model_lab_batch_worker_count(5) == 1

    monkeypatch.setenv("ALPHA_LAB_MODEL_LAB_MAX_WORKERS", "2")
    assert _build_model_lab_batch_worker_count(1) == 1
    assert _build_model_lab_batch_worker_count(2) == 2
    assert _build_model_lab_batch_worker_count(5) == 2


def test_load_model_factor_artifact_paths_from_manifest(tmp_path: Path) -> None:
    output_dir = tmp_path / "case"
    output_dir.mkdir()
    metrics = output_dir / "metrics.json"
    metrics.write_text('{"metrics": {"factor_verdict": "Pass"}}', encoding="utf-8")
    relative_summary = output_dir / "summary.md"
    relative_summary.write_text("# Summary\n", encoding="utf-8")
    manifest = output_dir / "run_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "outputs": {
                    "metrics": str(metrics),
                    "summary": "summary.md",
                }
            }
        ),
        encoding="utf-8",
    )

    paths = _load_model_factor_artifact_paths_from_manifest(output_dir)

    assert paths["run_manifest"] == manifest
    assert paths["metrics"] == metrics
    assert paths["summary"] == relative_summary


def test_resolve_run_artifact_for_endpoint_supports_group_returns_alias(tmp_path: Path) -> None:
    output_dir = tmp_path / "alias_case"
    output_dir.mkdir(parents=True, exist_ok=True)
    quantile_returns = output_dir / "quantile_returns.csv"
    quantile_returns.write_text("date,group,group_return\n2026-01-02,1,0.01\n", encoding="utf-8")

    run = _RunRecord(
        run_id="alias-run",
        project_slug="__model_lab__",
        case_name="alias_case",
        round_id=None,
        spec_path=str(output_dir / "case.yaml"),
        submitted_at_utc="2026-04-26T00:00:00Z",
        evaluation_profile="default_research",
        output_root_dir=None,
        render_report=True,
        status="succeeded",
        output_dir=str(output_dir),
        artifact_paths={"quantile_returns": str(quantile_returns)},
        summary={},
    )

    resolved = _resolve_run_artifact_for_endpoint(run, "group_returns")

    assert resolved == quantile_returns.resolve()


def test_resolve_run_artifact_for_endpoint_falls_back_to_output_dir(tmp_path: Path) -> None:
    output_dir = tmp_path / "fallback_case"
    output_dir.mkdir(parents=True, exist_ok=True)
    group_returns = output_dir / "group_returns.csv"
    group_returns.write_text("date,group,group_return\n2026-01-03,5,0.02\n", encoding="utf-8")

    run = _RunRecord(
        run_id="fallback-run",
        project_slug="__model_lab__",
        case_name="fallback_case",
        round_id=None,
        spec_path=str(output_dir / "case.yaml"),
        submitted_at_utc="2026-04-26T00:00:00Z",
        evaluation_profile="default_research",
        output_root_dir=None,
        render_report=True,
        status="succeeded",
        output_dir=str(output_dir),
        artifact_paths={},
        summary={},
    )

    resolved = _resolve_run_artifact_for_endpoint(run, "group_returns")

    assert resolved == group_returns.resolve()


def test_run_record_payload_filters_missing_artifacts(tmp_path: Path) -> None:
    output_dir = tmp_path / "payload_filter_case"
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = output_dir / "metrics.json"
    metrics.write_text('{"metrics": {"mean_ic": 0.01}}', encoding="utf-8")

    run = _RunRecord(
        run_id="payload-filter-run",
        project_slug="demo",
        case_name="payload_filter_case",
        round_id=None,
        spec_path=str(output_dir / "case.yaml"),
        submitted_at_utc="2026-04-26T00:00:00Z",
        evaluation_profile="default_research",
        output_root_dir=None,
        render_report=True,
        status="succeeded",
        output_dir=str(output_dir),
        artifact_paths={
            "metrics": str(metrics),
            "research_tearsheet": str(output_dir / "research_tearsheet_missing.json"),
        },
        summary={},
    )

    payload = run.to_payload()
    compact = run.to_compact_payload()

    assert "metrics" in payload["artifact_paths"]
    assert "research_tearsheet" not in payload["artifact_paths"]
    assert compact["artifact_paths"].get("metrics") is True
    assert "research_tearsheet" not in compact["artifact_paths"]


def test_run_record_payload_keeps_artifact_when_registered_path_is_stale(tmp_path: Path) -> None:
    output_dir = tmp_path / "payload_filter_fallback_case"
    output_dir.mkdir(parents=True, exist_ok=True)
    tearsheet = output_dir / "research_tearsheet.json"
    tearsheet.write_text('{"artifact_type": "alpha_lab_research_tearsheet"}', encoding="utf-8")

    run = _RunRecord(
        run_id="payload-filter-fallback-run",
        project_slug="demo",
        case_name="payload_filter_fallback_case",
        round_id=None,
        spec_path=str(output_dir / "case.yaml"),
        submitted_at_utc="2026-04-26T00:00:00Z",
        evaluation_profile="default_research",
        output_root_dir=None,
        render_report=True,
        status="succeeded",
        output_dir=str(output_dir),
        artifact_paths={
            "research_tearsheet": str(output_dir / "stale_path.json"),
        },
        summary={},
    )

    payload = run.to_payload()

    assert payload["artifact_paths"]["research_tearsheet"] == str(tearsheet.resolve())


def test_model_factor_web_output_dir_is_scoped_by_run_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _RunStore()
    spec_path = tmp_path / "case.yaml"
    spec_path.write_text("name: same_case\n", encoding="utf-8")
    output_root = tmp_path / "outputs"

    fake_spec = SimpleNamespace(
        name="same_case",
        output=SimpleNamespace(root_dir=str(output_root)),
    )
    monkeypatch.setattr(
        "alpha_lab.web_unified._subprocess.load_model_factor_case_spec",
        lambda _path: fake_spec,
    )

    task_a = _RunTask(
        run_id="run-a",
        project_slug="proj",
        case_name="same_case",
        round_id=None,
        spec_path=str(spec_path),
        evaluation_profile="default_research",
        output_root_dir=None,
        render_report=False,
        workflow="model_factor",
    )
    task_b = _RunTask(
        run_id="run-b",
        project_slug="proj",
        case_name="same_case",
        round_id=None,
        spec_path=str(spec_path),
        evaluation_profile="default_research",
        output_root_dir=None,
        render_report=False,
        workflow="model_factor",
    )

    output_a = Path(store._resolve_model_factor_task_output_dir(task_a))  # noqa: SLF001
    output_b = Path(store._resolve_model_factor_task_output_dir(task_b))  # noqa: SLF001
    cmd_a = _build_model_lab_subprocess_command(task=task_a, spec_path=spec_path.resolve())

    assert output_a == output_root.resolve() / "_web_runs" / "run-a" / "same_case"
    assert output_b == output_root.resolve() / "_web_runs" / "run-b" / "same_case"
    assert output_a != output_b
    assert "--output-root-dir" in cmd_a
    assert cmd_a[cmd_a.index("--output-root-dir") + 1] == str(
        output_root.resolve() / "_web_runs" / "run-a"
    )
    assert "--cache-root-dir" in cmd_a
    assert cmd_a[cmd_a.index("--cache-root-dir") + 1] == str(
        output_root.resolve() / "_model_factor_shared_cache"
    )

    cmd_b = _build_model_lab_subprocess_command(task=task_b, spec_path=spec_path.resolve())
    assert cmd_a[cmd_a.index("--cache-root-dir") + 1] == cmd_b[
        cmd_b.index("--cache-root-dir") + 1
    ], "shared cache root must not depend on run_id"


def test_model_lab_subprocess_command_includes_draft_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec_path = tmp_path / "case.yaml"
    spec_path.write_text("name: draft_case\n", encoding="utf-8")
    output_root = tmp_path / "outputs"
    candidate_path = (
        tmp_path
        / "model_candidates"
        / "research"
        / "candidate_alpha"
        / "model_candidate.json"
    )
    fake_spec = SimpleNamespace(
        name="draft_case",
        output=SimpleNamespace(root_dir=str(output_root)),
    )
    monkeypatch.setattr(
        "alpha_lab.web_unified._subprocess.load_model_factor_case_spec",
        lambda _path: fake_spec,
    )
    task = _RunTask(
        run_id="run-draft",
        project_slug="proj",
        case_name="draft_case",
        round_id=None,
        spec_path=str(spec_path),
        evaluation_profile="exploratory_screening",
        output_root_dir=None,
        render_report=True,
        workflow="model_factor",
        draft_model_candidate_path=str(candidate_path),
        draft_model_candidate_name="candidate_alpha",
        draft_model_candidate_hash="abc123",
        screening_retrain_every_n_dates=40,
    )

    cmd = _build_model_lab_subprocess_command(task=task, spec_path=spec_path.resolve())

    assert "--draft-model-candidate" in cmd
    assert cmd[cmd.index("--draft-model-candidate") + 1] == str(candidate_path)
    assert "--screening-retrain-every-n-dates" in cmd
    assert cmd[cmd.index("--screening-retrain-every-n-dates") + 1] == "40"


def test_model_lab_subprocess_env_prefers_workspace_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYTHONPATH", "existing-path")

    env = _build_model_lab_subprocess_env()

    first_entry = env["PYTHONPATH"].split(os.pathsep)[0]
    assert Path(first_entry).name == "src"
    assert "existing-path" in env["PYTHONPATH"].split(os.pathsep)
    assert env["PYTHONUNBUFFERED"] == "1"
    assert env["ALPHA_LAB_MODEL_LAB_CHILD"] == "1"


def test_single_factor_web_output_root_is_scoped_by_run_id(tmp_path: Path) -> None:
    spec_path = tmp_path / "case.yaml"
    output_root = tmp_path / "outputs"
    fake_spec = SimpleNamespace(
        name="same_case",
        output=SimpleNamespace(root_dir=str(output_root)),
    )
    task_a = _RunTask(
        run_id="run-a",
        project_slug="proj",
        case_name="same_case",
        round_id=None,
        spec_path=str(spec_path),
        evaluation_profile="default_research",
        output_root_dir=None,
        render_report=False,
    )
    task_b = _RunTask(
        run_id="run-b",
        project_slug="proj",
        case_name="same_case",
        round_id=None,
        spec_path=str(spec_path),
        evaluation_profile="default_research",
        output_root_dir=None,
        render_report=False,
    )

    output_a = _resolve_single_factor_web_output_root_dir(task_a, spec=fake_spec)
    output_b = _resolve_single_factor_web_output_root_dir(task_b, spec=fake_spec)

    assert output_a == output_root.resolve() / "_web_runs" / "run-a"
    assert output_b == output_root.resolve() / "_web_runs" / "run-b"
    assert output_a != output_b


def test_run_store_claims_queued_tasks_into_batch_groups() -> None:
    store = _RunStore()
    task_a = _RunTask(
        run_id="run-a",
        project_slug="proj-a",
        case_name="case-a",
        round_id=None,
        spec_path="/tmp/case-a.yaml",
        evaluation_profile="exploratory_screening",
        output_root_dir=None,
        render_report=True,
    )
    task_b = _RunTask(
        run_id="run-b",
        project_slug="proj-b",
        case_name="case-b",
        round_id=None,
        spec_path="/tmp/case-b.yaml",
        evaluation_profile="exploratory_screening",
        output_root_dir=None,
        render_report=True,
    )
    task_c = _RunTask(
        run_id="run-c",
        project_slug="proj-c",
        case_name="case-c",
        round_id=None,
        spec_path="/tmp/case-c.yaml",
        evaluation_profile="default_research",
        output_root_dir=None,
        render_report=True,
    )

    with store._lock:  # noqa: SLF001 - test seeds in-memory state directly
        for task in (task_a, task_b, task_c):
            store._tasks[task.run_id] = task  # noqa: SLF001
            store._records[task.run_id] = _RunRecord(  # noqa: SLF001
                run_id=task.run_id,
                project_slug=task.project_slug,
                case_name=task.case_name,
                round_id=task.round_id,
                spec_path=task.spec_path,
                submitted_at_utc="2026-04-19T00:00:00Z",
                evaluation_profile=task.evaluation_profile,
                output_root_dir=task.output_root_dir,
                render_report=task.render_report,
                status="queued",
            )

    groups = store._claim_queued_task_groups()  # noqa: SLF001

    assert len(groups) == 2
    assert [task.run_id for task in groups[0]] == ["run-a", "run-b"]
    assert [task.run_id for task in groups[1]] == ["run-c"]
    assert store.get("run-a").status == "running"
    assert store.get("run-b").status == "running"
    assert store.get("run-c").status == "running"


def test_run_store_reuses_input_bundle_cache_across_single_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _RunStore()
    spec_path = tmp_path / "case.yaml"
    spec_path.write_text("name: demo\n", encoding="utf-8")
    prices_path = tmp_path / "prices.parquet"
    prices_path.write_text("stub", encoding="utf-8")
    universe_path = tmp_path / "universe.parquet"
    universe_path.write_text("stub", encoding="utf-8")

    fake_spec = SimpleNamespace(
        prices_path=str(prices_path),
        output=SimpleNamespace(root_dir=str(tmp_path / "outputs")),
        universe=SimpleNamespace(
            path=str(universe_path),
            in_universe_column="in_universe",
        ),
    )

    bundle_load_count = 0

    def _fake_load_spec(_path: Path) -> SimpleNamespace:
        return fake_spec

    def _fake_load_inputs(_spec: object) -> object:
        nonlocal bundle_load_count
        bundle_load_count += 1
        return {"bundle_id": bundle_load_count}

    call_bundles: list[object] = []
    call_output_roots: list[Path] = []

    class _FakeResult:
        def __init__(self, output_dir: Path, metrics_path: Path) -> None:
            self.output_dir = output_dir
            self.artifact_paths = {"metrics": metrics_path}

    def _fake_run_single_factor_case(
        _spec: object,
        *,
        output_root_dir: object,
        evaluation_profile: str,
        vault_export_mode: str,
        progress_callback: object,
        input_bundle: object,
    ) -> _FakeResult:
        del evaluation_profile, vault_export_mode, progress_callback
        call_output_roots.append(Path(str(output_root_dir)))
        call_bundles.append(input_bundle)
        run_idx = len(call_bundles)
        output_dir = tmp_path / f"run-{run_idx}"
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = output_dir / "metrics.json"
        metrics_path.write_text(
            json.dumps({"metrics": {"factor_verdict": "Pass"}}),
            encoding="utf-8",
        )
        return _FakeResult(output_dir=output_dir, metrics_path=metrics_path)

    monkeypatch.setattr("alpha_lab.web_unified._run_store.load_single_factor_case_spec", _fake_load_spec)
    monkeypatch.setattr("alpha_lab.web_unified._run_store.load_standard_inputs", _fake_load_inputs)
    monkeypatch.setattr(
        "alpha_lab.web_unified._run_store.run_single_factor_case", _fake_run_single_factor_case
    )

    tasks = [
        _RunTask(
            run_id="run-1",
            project_slug="proj",
            case_name="case",
            round_id=None,
            spec_path=str(spec_path),
            evaluation_profile="exploratory_screening",
            output_root_dir=None,
            render_report=False,
        ),
        _RunTask(
            run_id="run-2",
            project_slug="proj",
            case_name="case",
            round_id=None,
            spec_path=str(spec_path),
            evaluation_profile="exploratory_screening",
            output_root_dir=None,
            render_report=False,
        ),
    ]

    with store._lock:  # noqa: SLF001 - seed run store directly
        for idx, task in enumerate(tasks, start=1):
            store._tasks[task.run_id] = task  # noqa: SLF001
            store._records[task.run_id] = _RunRecord(  # noqa: SLF001
                run_id=task.run_id,
                project_slug=task.project_slug,
                case_name=task.case_name,
                round_id=task.round_id,
                spec_path=task.spec_path,
                submitted_at_utc=f"2026-04-20T00:00:0{idx}Z",
                evaluation_profile=task.evaluation_profile,
                output_root_dir=task.output_root_dir,
                render_report=task.render_report,
                status="running",
            )

    for task in tasks:
        store._execute_single_task(task, allow_fallback=False)  # noqa: SLF001

    assert bundle_load_count == 1
    assert len(call_bundles) == 2
    assert call_bundles[0] is call_bundles[1]
    assert call_output_roots == [
        (tmp_path / "outputs" / "_web_runs" / "run-1").resolve(),
        (tmp_path / "outputs" / "_web_runs" / "run-2").resolve(),
    ]
    assert store.get("run-1").status == "succeeded"
    assert store.get("run-2").status == "succeeded"


def test_run_store_executes_model_factor_batch_in_parallel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _RunStore()
    tasks = [
        _RunTask(
            run_id=f"run-{idx}",
            project_slug="proj",
            case_name=f"case-{idx}",
            round_id=None,
            spec_path=f"/tmp/case-{idx}.yaml",
            evaluation_profile="default_research",
            output_root_dir=None,
            render_report=False,
            workflow="model_factor",
        )
        for idx in range(3)
    ]
    active = 0
    max_active = 0
    seen_run_ids: list[str] = []
    active_lock = threading.Lock()

    def _fake_execute_single_task(task: _RunTask, *, allow_fallback: bool) -> None:
        nonlocal active, max_active
        assert allow_fallback is False
        with active_lock:
            active += 1
            max_active = max(max_active, active)
            seen_run_ids.append(task.run_id)
        time.sleep(0.05)
        with active_lock:
            active -= 1

    monkeypatch.setattr(store, "_execute_single_task", _fake_execute_single_task)
    monkeypatch.setattr(store, "_model_factor_batch_has_output_conflict", lambda _tasks: False)
    monkeypatch.setattr(
        "alpha_lab.web_unified._run_store._build_model_lab_batch_worker_count",
        lambda _n: 3,
    )

    store._execute_task_group(tasks)  # noqa: SLF001

    assert set(seen_run_ids) == {"run-0", "run-1", "run-2"}
    assert max_active >= 2


def test_run_store_falls_back_to_serial_for_model_factor_output_conflicts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _RunStore()
    tasks = [
        _RunTask(
            run_id=f"run-{idx}",
            project_slug="proj",
            case_name=f"case-{idx}",
            round_id=None,
            spec_path=f"/tmp/case-{idx}.yaml",
            evaluation_profile="default_research",
            output_root_dir=None,
            render_report=False,
            workflow="model_factor",
        )
        for idx in range(2)
    ]
    active = 0
    max_active = 0
    active_lock = threading.Lock()

    def _fake_execute_single_task(task: _RunTask, *, allow_fallback: bool) -> None:
        nonlocal active, max_active
        assert allow_fallback is False
        with active_lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.02)
        with active_lock:
            active -= 1

    monkeypatch.setattr(store, "_execute_single_task", _fake_execute_single_task)
    monkeypatch.setattr(store, "_model_factor_batch_has_output_conflict", lambda _tasks: True)

    store._execute_task_group(tasks)  # noqa: SLF001

    assert max_active == 1
