from __future__ import annotations

import json
from pathlib import Path

from alpha_lab.research_bridge.model_idea import (
    explore_model_idea,
    list_model_idea_sessions,
    model_idea_sessions_root,
    read_model_idea_session,
    record_model_idea_response,
    save_model_idea_session,
)
from tests.model_factor_case_helpers import write_demo_model_factor_case


def test_model_idea_session_save_list_read(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)

    payload = explore_model_idea(
        idea="Build a stronger ridge baseline with turnover control.",
        mode="constrained",
        workspace_root=workspace,
        memory_limit=0,
    )
    saved = save_model_idea_session(payload=payload, workspace_root=workspace)

    session_root = model_idea_sessions_root(workspace_root=workspace)
    session_path = session_root / f"{saved['session_id']}.json"
    assert session_path.exists()

    listed = list_model_idea_sessions(workspace_root=workspace, limit=5)
    assert len(listed) >= 1
    assert listed[0]["session_id"] == saved["session_id"]

    loaded = read_model_idea_session(
        session_id=str(saved["session_id"]),
        workspace_root=workspace,
    )
    assert loaded["idea"] == payload["idea"]
    assert loaded["mode"] == payload["mode"]
    assert "constraint_report" in loaded


def test_model_idea_memory_context_uses_recent_sessions(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)

    first = explore_model_idea(
        idea="Try rolling ridge with robust rank transform.",
        mode="constrained",
        workspace_root=workspace,
        memory_limit=0,
    )
    save_model_idea_session(payload=first, workspace_root=workspace)

    second = explore_model_idea(
        idea="Test lightgbm with turnover-aware selection in finance panel.",
        mode="constrained",
        workspace_root=workspace,
        memory_limit=3,
    )

    extras = second["constraint_report"]["recommendations"]["extras"]
    assert extras["session_memory_status"] == "loaded"
    assert isinstance(extras["session_memory"], list)
    assert len(extras["session_memory"]) >= 1
    assert "[M1]" in second["gpt_prompt"]


def test_model_idea_priority_order_and_failure_first_prompt(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    dist = workspace / "dist"
    dist.mkdir(parents=True, exist_ok=True)

    run_ok = dist / "run_ok"
    run_ok.mkdir(parents=True, exist_ok=True)
    (run_ok / "run_manifest.json").write_text(
        json.dumps(
            {
                "workflow": "real_case_model_factor",
                "run_id": "run_ok",
                "case_name": "case_ok",
                "status": "succeeded",
            }
        ),
        encoding="utf-8",
    )
    (run_ok / "metrics.json").write_text(
        json.dumps(
            {
                "metrics": {
                    "factor_name": "priority_factor",
                    "model_family": "ridge",
                    "mean_rank_ic": 0.052,
                    "mean_ic": 0.019,
                    "factor_verdict": "pass",
                }
            }
        ),
        encoding="utf-8",
    )

    run_failed = dist / "run_failed"
    run_failed.mkdir(parents=True, exist_ok=True)
    (run_failed / "run_manifest.json").write_text(
        json.dumps(
            {
                "workflow": "real_case_model_factor",
                "run_id": "run_failed",
                "case_name": "case_failed",
                "status": "failed",
                "error": "integrity failure",
            }
        ),
        encoding="utf-8",
    )
    (run_failed / "metrics.json").write_text(
        json.dumps(
            {
                "metrics": {
                    "factor_name": "priority_factor",
                    "model_family": "ridge",
                    "mean_rank_ic": 0.081,
                    "mean_ic": 0.027,
                    "factor_verdict": "fail",
                    "highest_severity": "fail",
                }
            }
        ),
        encoding="utf-8",
    )

    spec_path = write_demo_model_factor_case(tmp_path, factor_name="priority_factor")
    payload = explore_model_idea(
        idea="Improve ridge model with turnover controls.",
        mode="constrained",
        spec=str(spec_path),
        workspace_root=workspace,
        memory_limit=0,
    )

    extras = payload["constraint_report"]["recommendations"]["extras"]
    assert extras["priority_order"] == [
        "contracts",
        "failures",
        "validated",
        "knowledge",
        "sources",
    ]
    run_matches = extras["run_matches"]
    assert run_matches[0]["run_id"] == "run_ok"
    assert run_matches[0]["status"] == "succeeded"

    prompt = str(payload["gpt_prompt"])
    assert prompt.find("[F1]") != -1
    assert prompt.find("[E1]") != -1
    assert prompt.find("[F1]") < prompt.find("[E1]")


def test_model_idea_record_response_and_chain_upstream(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)

    first = explore_model_idea(
        idea="Discover model mechanisms for nonlinear feature interactions.",
        mode="explore",
        stage="mechanism_discovery",
        workspace_root=workspace,
        memory_limit=0,
    )
    saved = save_model_idea_session(payload=first, workspace_root=workspace)
    response = """\
[模型机制候选]
### 机制 1
- mechanism family: loss/regularization
- touched contract surfaces: model
- early falsifier: rank_ic_ir 不改善
### 机制 2
- mechanism family: feature interaction
- touched contract surfaces: model, feature_preprocess
- early falsifier: feature importance 不稳定

[实现假设草图]
- 只写机制草图，不写 patch。

[与当前 spec / baseline 的关系]
- baseline 是 ridge；差异是 feature interaction。

[不确定性与失败路径]
- PIT / label leakage risk: known_at 对齐。
"""
    report = record_model_idea_response(
        session_id=str(saved["session_id"]),
        response_text=response,
        workspace_root=workspace,
    )
    assert report.has_errors is False

    chained = explore_model_idea(
        idea="Map upstream mechanisms into runnable model variants.",
        mode="constrained",
        stage="signal_mapping",
        workspace_root=workspace,
        memory_limit=0,
        inject_recent_drift=True,
    )
    prompt = str(chained["gpt_prompt"])
    diagnostics = chained["retrieval_diagnostics"]
    assert "上游模型研究产物" in prompt
    assert "模型机制候选" in prompt
    assert diagnostics["upstream_session_id"] == saved["session_id"]
    assert diagnostics["upstream_sections_injected"] >= 1


def test_model_idea_injects_recent_lint_drift(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)

    payload = explore_model_idea(
        idea="Map one model idea but intentionally produce a bad response.",
        mode="explore",
        stage="signal_mapping",
        workspace_root=workspace,
        memory_limit=0,
    )
    saved = save_model_idea_session(payload=payload, workspace_root=workspace)
    report = record_model_idea_response(
        session_id=str(saved["session_id"]),
        response_text="[Model Mechanism Mapping]\n只有一段，缺风险控制。",
        workspace_root=workspace,
    )
    assert report.has_errors is True

    followup = explore_model_idea(
        idea="Retry model signal mapping with drift avoidance.",
        mode="explore",
        stage="signal_mapping",
        workspace_root=workspace,
        memory_limit=0,
        inject_recent_drift=True,
    )
    prompt = str(followup["gpt_prompt"])
    diagnostics = followup["retrieval_diagnostics"]
    assert "已知漂移模式" in prompt
    assert "missing_section" in prompt
    assert diagnostics["recent_lint_violations_injected"] >= 1
