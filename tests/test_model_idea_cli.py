from __future__ import annotations

import json

import pytest

import alpha_lab.research_bridge.model_idea as model_idea_module
from alpha_lab.research_bridge.llm_rerank import RerankOutcome
from alpha_lab.research_bridge.model_idea import main as model_idea_main


@pytest.fixture(autouse=True)
def _disable_model_idea_llm_rerank(monkeypatch: pytest.MonkeyPatch) -> None:
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
        model_idea_module,
        "rerank_candidates",
        fake_rerank_candidates,
    )


def _minimal_model_prompt_report() -> dict[str, object]:
    return {
        "system_contracts": {
            "supported_model_families": ["ridge"],
            "supported_feature_preprocess": {},
            "supported_training": {},
            "supported_selection_metrics": [],
            "supported_feature_importance": {},
        },
        "current_spec": {"status": "unavailable"},
        "validated_baselines": [],
        "recent_failures": [],
        "recommendations": {
            "extras": {
                "knowledge_matches": [],
                "knowledge_handling_patterns": [],
                "session_memory": [],
                "warnings": [],
            },
            "source_anchors": [],
        },
    }



def test_model_idea_cli_routes_distribute(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: dict[str, object] = {}

    class _FakeResult:
        def to_payload(self) -> dict[str, object]:
            return {"ok": True, "idea_id": "idea-1", "engines": ["claude", "codex"]}

    def _fake_distribute_model_idea(**kwargs: object) -> _FakeResult:
        captured.update(kwargs)
        return _FakeResult()

    monkeypatch.setattr(
        model_idea_module,
        "distribute_model_idea",
        _fake_distribute_model_idea,
    )

    rc = model_idea_main(
        [
            "distribute",
            "--idea",
            "turnover-aware model idea",
            "--engines",
            "claude,codex",
            "--vault-root",
            "/tmp/vault",
        ]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["idea_id"] == "idea-1"
    assert captured["idea"] == "turnover-aware model idea"
    assert captured["engines"] == "claude,codex"
    assert captured["vault_root"] == "/tmp/vault"

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
    assert diag["recommended_next_stage"] is None
    prompt = payload["gpt_prompt"]
    assert "Stage Notice" not in prompt
    assert "[Model Mechanism Mapping]" in prompt
    assert "[模型风险控制]" in prompt
    assert "`feature availability / PIT`" in prompt
    assert "输出自检" in prompt


def test_model_idea_rejects_validation_kill_tests_public_stage() -> None:
    from alpha_lab.research_bridge.model_idea import explore_model_idea

    with pytest.raises(ValueError, match="moved out of model-lab idea exploration"):
        explore_model_idea(
            idea="Audit existing model.",
            mode="constrained",
            stage="validation_kill_tests",
        )


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


def test_model_idea_mechanism_start_prompt_builder_contract_direct() -> None:
    from alpha_lab.research_bridge.model_idea import (
        _build_model_idea_mechanism_start_prompt,
    )

    prompt = _build_model_idea_mechanism_start_prompt(
        idea="Borrow robust training ideas.",
        mode="start",
        report=_minimal_model_prompt_report(),
        spec_patch_hint=None,
    )

    assert "> Stage: mechanism_discovery" in prompt
    assert "> Mode: start" in prompt
    assert "[模型机制候选]" in prompt
    assert "[实现假设草图]" in prompt
    assert "kickoff" in prompt.lower()
    assert "discussion-only" in prompt
    assert "needs-extension" in prompt
    assert "transfer cost" in prompt
    assert "[Model Mechanism Mapping]" not in prompt
    assert "[Alias / 问题归因审计]" not in prompt


def test_model_idea_mechanism_structured_prompt_builder_contract_direct() -> None:
    from alpha_lab.research_bridge.model_idea import (
        _build_model_idea_mechanism_structured_prompt,
    )

    prompt = _build_model_idea_mechanism_structured_prompt(
        idea="Improve model selection without parameter-only tuning.",
        mode="explore",
        report=_minimal_model_prompt_report(),
        spec_patch_hint=None,
    )

    assert "> Stage: mechanism_discovery" in prompt
    assert "> Mode: explore" in prompt
    assert "[模型机制候选]" in prompt
    assert "[与当前 spec / baseline 的关系]" in prompt
    assert "why it is not just parameter tuning" in prompt
    assert "single best model" in prompt
    assert "contract extension" not in prompt
    assert "输出 2-4 个机制候选" not in prompt
    assert "binary alias-tag" not in prompt
    assert "[Model Mechanism Mapping]" not in prompt


def test_model_idea_mechanism_constrained_prompt_builder_contract_direct() -> None:
    from alpha_lab.research_bridge.model_idea import (
        _build_model_idea_mechanism_constrained_prompt,
    )

    prompt = _build_model_idea_mechanism_constrained_prompt(
        idea="Constrain candidate mechanisms to runnable contracts.",
        mode="constrained",
        report=_minimal_model_prompt_report(),
        spec_patch_hint=None,
    )

    assert "> Stage: mechanism_discovery" in prompt
    assert "> Mode: constrained" in prompt
    assert "[模型机制候选]" in prompt
    assert "[不确定性与失败路径]" in prompt
    assert "输出 2-4 个机制候选" in prompt
    assert "requires_code_change" in prompt
    assert "alpha/lambda/window/depth" in prompt
    assert "[Alias / 问题归因审计]" not in prompt


def test_model_idea_signal_mapping_prompt_builder_contract_direct() -> None:
    from alpha_lab.research_bridge.model_idea import (
        _build_model_idea_signal_mapping_prompt,
    )

    prompt = _build_model_idea_signal_mapping_prompt(
        idea="Map upstream mechanisms into runnable variants.",
        mode="constrained",
        report=_minimal_model_prompt_report(),
        spec_patch_hint=None,
        strict=True,
    )

    assert "> Stage: signal_mapping" in prompt
    assert "> Mode: constrained" in prompt
    assert "[Model Mechanism Mapping]" in prompt
    assert "[当前实现解释]" in prompt
    assert "[模型风险控制]" in prompt
    assert "[可测试模型版本]" in prompt
    assert "`feature availability / PIT`" in prompt
    assert "binary alias-tag" in prompt
    assert "baseline linear/ridge" in prompt
    assert "[模型机制候选]" not in prompt
    assert "[Alias / 问题归因审计]" not in prompt


def test_model_idea_validation_kill_tests_prompt_builder_contract_direct() -> None:
    from alpha_lab.research_bridge.model_idea import (
        _build_model_idea_validation_kill_tests_prompt,
    )

    with pytest.raises(ValueError, match="moved out of model-lab idea exploration"):
        _build_model_idea_validation_kill_tests_prompt(
            idea="Audit whether the model idea survives kill tests.",
            mode="constrained",
            report=_minimal_model_prompt_report(),
            spec_patch_hint=None,
            strict=True,
        )
