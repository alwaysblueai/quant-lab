from __future__ import annotations

import json
from pathlib import Path

import pytest

from alpha_lab.research_bridge.model_idea import main as model_idea_main
from tests.model_factor_case_helpers import write_demo_model_factor_case


def test_model_idea_cli_empty_idea_errors(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        model_idea_main(["explore", "--idea", "   "])
    captured = capsys.readouterr()
    assert "idea must be non-empty" in captured.err


def test_model_idea_cli_constrained_contains_model_contracts(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec_path = write_demo_model_factor_case(tmp_path, factor_name="model_idea_demo")

    rc = model_idea_main(
        [
            "explore",
            "--idea",
            "Build a stronger tree-based model for daily financial panel data.",
            "--mode",
            "constrained",
            "--spec",
            str(spec_path),
        ]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    report = payload["constraint_report"]
    assert set(report.keys()) == {
        "system_contracts",
        "current_spec",
        "validated_baselines",
        "recent_failures",
        "recommendations",
    }
    assert "ridge" in report["system_contracts"]["supported_model_families"]
    assert payload["mode"] == "constrained"
    assert "Supported model families" in payload["gpt_prompt"]


def test_model_idea_cli_missing_spec_degrades_gracefully(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = model_idea_main(
        [
            "explore",
            "--idea",
            "Try robust model-selection logic for noisy cross-sectional features.",
            "--spec",
            "non_existing_spec.yaml",
        ]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    current_spec = payload["constraint_report"]["current_spec"]
    warnings = payload["constraint_report"]["recommendations"]["extras"]["warnings"]
    assert current_spec["status"] == "unavailable"
    assert isinstance(warnings, list)
    assert len(warnings) >= 1


def test_model_idea_cli_start_mode_is_kickoff(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec_path = write_demo_model_factor_case(tmp_path, factor_name="model_idea_kickoff")

    rc = model_idea_main(
        [
            "explore",
            "--idea",
            "Borrow ideas from imbalanced-classification literature for daily panels.",
            "--mode",
            "start",
            "--spec",
            str(spec_path),
        ]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "start"
    # Kickoff must not collapse the idea into a deterministic spec patch.
    assert payload["spec_patch_hint"] is None
    prompt = payload["gpt_prompt"]
    assert "kickoff" in prompt.lower()
    assert "discussion-only" in prompt
    assert "Candidate Spec Patch Hint" not in prompt
    # Recommendations should reflect the kickoff-specific guidance.
    recommendations = payload["constraint_report"]["recommendations"]
    assert "kickoff" in str(recommendations["mode_guidance"]).lower()
    next_actions = recommendations["next_actions"]
    assert any("do NOT converge" in str(item) for item in next_actions)


def test_model_idea_cli_stage_axis_defaults_to_mechanism_discovery(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = model_idea_main(
        [
            "explore",
            "--idea",
            "Try ridge with turnover-aware selection.",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["stage"] == "mechanism_discovery"
    diag = payload["retrieval_diagnostics"]
    assert diag["stage"] == "mechanism_discovery"
    assert diag["recommended_next_stage"] == "signal_mapping"
    assert "Stage: mechanism_discovery" in payload["gpt_prompt"]


def test_model_idea_explicit_stage_progresses_recommended_next() -> None:
    from alpha_lab.research_bridge.model_idea import explore_model_idea

    payload = explore_model_idea(
        idea="Test signal mapping in model-lab.",
        mode="explore",
        stage="signal_mapping",
    )
    assert payload["stage"] == "signal_mapping"
    diag = payload["retrieval_diagnostics"]
    assert diag["stage"] == "signal_mapping"
    assert diag["recommended_next_stage"] == "validation_kill_tests"
    prompt = payload["gpt_prompt"]
    assert "Stage Notice" not in prompt
    assert "[Model Mechanism Mapping]" in prompt
    assert "[模型风险控制]" in prompt
    assert "`feature availability / PIT`" in prompt
    assert "输出自检" in prompt


def test_model_idea_validation_kill_tests_stage_has_no_next() -> None:
    from alpha_lab.research_bridge.model_idea import explore_model_idea

    payload = explore_model_idea(
        idea="Audit existing model.",
        mode="constrained",
        stage="validation_kill_tests",
    )
    diag = payload["retrieval_diagnostics"]
    assert diag["stage"] == "validation_kill_tests"
    assert diag["recommended_next_stage"] is None


def test_model_idea_cli_save_session_writes_session(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)

    rc = model_idea_main(
        [
            "explore",
            "--idea",
            "Explore ridge baseline with turnover control.",
            "--workspace-root",
            str(workspace),
            "--save-session",
        ]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert isinstance(payload.get("session"), dict)
    session_id = str(payload["session"]["session_id"])
    assert session_id
    session_path = (
        workspace
        / "artifacts"
        / "model_lab_explorer"
        / "sessions"
        / f"{session_id}.json"
    )
    assert session_path.exists()


def test_model_idea_cli_record_response_updates_session(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)

    rc = model_idea_main(
        [
            "explore",
            "--idea",
            "Explore model response recording.",
            "--workspace-root",
            str(workspace),
            "--save-session",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    session_id = str(payload["session"]["session_id"])

    rc2 = model_idea_main(
        [
            "record-response",
            "--session-id",
            session_id,
            "--workspace-root",
            str(workspace),
            "--response-text",
            "[模型机制候选]\n只有一段，缺少其余必需结构。",
        ]
    )
    assert rc2 == 0
    report = json.loads(capsys.readouterr().out)
    assert report["has_errors"] is True
    session_path = (
        workspace
        / "artifacts"
        / "model_lab_explorer"
        / "sessions"
        / f"{session_id}.json"
    )
    session_payload = json.loads(session_path.read_text(encoding="utf-8"))
    assert session_payload["response"]
    assert session_payload["lint_report"]["has_errors"] is True


def test_model_idea_stage_prompts_align_with_lint_anchors() -> None:
    from alpha_lab.research_bridge.model_idea import explore_model_idea

    expected_by_stage = {
        "mechanism_discovery": (
            "[模型机制候选]",
            "[实现假设草图]",
            "[与当前 spec / baseline 的关系]",
            "[不确定性与失败路径]",
        ),
        "signal_mapping": (
            "[Model Mechanism Mapping]",
            "[当前实现解释]",
            "[模型风险控制]",
            "[可测试模型版本]",
        ),
        "validation_kill_tests": (
            "[Alias / 问题归因审计]",
            "[数据与时间完整性]",
            "[训练与验证稳健性]",
            "[特征与解释稳定性]",
            "[成本与组合影响]",
            "[最终判定]",
        ),
    }

    for stage, anchors in expected_by_stage.items():
        payload = explore_model_idea(
            idea=f"Check model-lab prompt anchors for {stage}.",
            mode="constrained",
            stage=stage,
        )
        prompt = str(payload["gpt_prompt"])
        assert "Stage Notice" not in prompt
        assert "not yet specialized" not in prompt
        for anchor in anchors:
            assert anchor in prompt
