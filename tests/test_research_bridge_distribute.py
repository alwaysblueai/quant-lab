"""Tests for the Stage 0 audience-based distribution + Stage 4 experiment-card.

Covers:
- ``engine_prompts.build_prompt`` symmetric task (Claude Code + Codex GUI
  receive byte-identical content except for self-identification)
- ``codebase_index.build_codebase_snapshot`` directory walk
- ``service.distribute_idea`` end-to-end (writes the 5-file layout)
- ``model_idea.distribute_model_idea`` thin wrapper (lab=model_factor)
- CLI ``alpha-lab idea distribute`` routing
- ``experiment_card.scaffold_experiment_card`` + ``--cleanup`` behavior

These tests cover the protocol from docs/end_to_end_workflow.md: both
engines do the **same** generator + reviewer task; web GPT综合两份输出
取长补短 in Stage 2.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from alpha_lab.research_bridge.codebase_index import (
    CodebaseSnapshot,
    build_codebase_snapshot,
)
from alpha_lab.research_bridge.engine_prompts import (
    CardForPrompt,
    Engine,
    Lab,
    PromptContext,
    build_prompt,
    normalize_engines,
    output_filename_for,
    prompt_filename_for,
)
from alpha_lab.research_bridge.experiment_card import (
    ExperimentCardOutcome,
    scaffold_experiment_card,
)
from alpha_lab.research_bridge.model_idea import distribute_model_idea
from alpha_lab.research_bridge.service import distribute_idea


def _build_vault(tmp_path: Path) -> Path:
    vault = tmp_path / "vault"
    (vault / "30_factors").mkdir(parents=True)
    card = vault / "30_factors" / "Factor - Momentum Base.md"
    card.write_text(
        "---\n"
        "type: factor\n"
        "lifecycle: theoretical\n"
        "mechanism: momentum\n"
        "factor_family: momentum\n"
        "transferable_moves:\n"
        '  - "把动量信号按市场状态条件化（regime-conditioning）"\n'
        "operative_claims:\n"
        '  - "动量衰减与市场流动性正相关"\n'
        "---\n"
        "# Momentum Base\n经典动量因子。\n",
        encoding="utf-8",
    )
    return vault


def _build_workspace(tmp_path: Path) -> Path:
    """Workspace with one factor and one model candidate already present."""

    workspace = tmp_path / "workspace"
    (workspace / "custom_factors" / "research" / "alpha_one").mkdir(parents=True)
    (workspace / "custom_factors" / "promoted" / "alpha_promoted").mkdir(parents=True)
    (workspace / "model_candidates" / "research" / "ridge_smoke").mkdir(parents=True)
    (workspace / "configs" / "real_cases" / "single_factor").mkdir(parents=True)
    (workspace / "configs" / "real_cases" / "model_factor").mkdir(parents=True)
    (workspace / "configs" / "real_cases" / "single_factor" / "alpha_one_v1.yaml").write_text(
        "name: alpha_one_v1\n", encoding="utf-8"
    )
    return workspace


def _ctx(*, lab: Lab, draft_dir: Path, vault_root: Path) -> PromptContext:
    return PromptContext(
        idea="test idea",
        idea_id="20260511T000000Z__test",
        lab=lab,
        draft_dir=draft_dir,
        vault_root=vault_root,
        related_cards=(
            CardForPrompt(
                name="K1",
                path="30_factors/Foo.md",
                summary="一张测试卡",
                transferable_moves=("把 X 应用到 Y",),
            ),
        ),
        insight_brief=("跨卡观察 alpha。", "跨卡观察 beta。"),
        codebase=CodebaseSnapshot(
            factors_promoted=("alpha_promoted",),
            factors_research=("alpha_one",),
            model_candidates_promoted=(),
            model_candidates_research=("ridge_smoke",),
            single_factor_cases=("alpha_one_v1",),
            model_factor_cases=(),
        ),
    )


# ---------------------------------------------------------------------------
# engine_prompts: symmetric task
# ---------------------------------------------------------------------------


def test_normalize_engines_default_returns_both() -> None:
    assert normalize_engines(None) == (Engine.CLAUDE, Engine.CODEX)


def test_normalize_engines_parses_csv() -> None:
    assert normalize_engines("codex,claude") == (Engine.CODEX, Engine.CLAUDE)


def test_normalize_engines_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="unknown engine"):
        normalize_engines("rogue_engine")


def test_filename_helpers() -> None:
    assert output_filename_for(Engine.CLAUDE) == "stage1_claude.md"
    assert output_filename_for(Engine.CODEX) == "stage1_codex.md"
    assert prompt_filename_for(Engine.CLAUDE) == "prompt_claude.md"
    assert prompt_filename_for(Engine.CODEX) == "prompt_codex.md"


def test_prompt_body_symmetric_except_identity_and_output_filename(
    tmp_path: Path,
) -> None:
    """Same task, same retrieval, same codebase index.

    The two engine prompts differ only in (a) self-identification line and
    (b) the output filename they're told to write to (stage1_<engine>.md).
    """

    ctx = _ctx(lab=Lab.SINGLE_FACTOR, draft_dir=tmp_path, vault_root=tmp_path)
    claude_prompt = build_prompt(engine=Engine.CLAUDE, ctx=ctx)
    codex_prompt = build_prompt(engine=Engine.CODEX, ctx=ctx)
    assert "你是 **Claude Code**" in claude_prompt
    assert "你是 **Codex GUI**" in codex_prompt
    assert "Stage 1 Prompt — Claude Code" in claude_prompt
    assert "Stage 1 Prompt — Codex GUI" in codex_prompt

    def normalize(text: str) -> str:
        return (
            text.replace("Claude Code", "X")
            .replace("Codex GUI", "X")
            .replace("stage1_claude.md", "stage1_X.md")
            .replace("stage1_codex.md", "stage1_X.md")
        )

    assert normalize(claude_prompt) == normalize(codex_prompt)


def test_prompt_carries_inline_governance_preamble(tmp_path: Path) -> None:
    """Codex GUI 桌面版 Project 没有持久化 Instructions —— governance 必须内联。"""

    ctx = _ctx(lab=Lab.SINGLE_FACTOR, draft_dir=tmp_path, vault_root=tmp_path)
    for engine in (Engine.CLAUDE, Engine.CODEX):
        prompt = build_prompt(engine=engine, ctx=ctx)
        # Section heading + key hard-rule fragments
        assert "## 0. Governance" in prompt
        assert "你的位置" in prompt
        assert "禁止越界" in prompt
        assert "vault = 素材库不是判决书" in prompt
        assert "PIT / future-leakage" in prompt
        # Governance section sits before §1 (so engine reads rules first)
        gov_idx = prompt.find("## 0. Governance")
        idea_idx = prompt.find("## 1. Idea")
        assert 0 < gov_idx < idea_idx
        # Explicit anti-overreach tokens
        assert "Stage 2 网页 GPT 的事" in prompt
        assert "Stage 3 后端的事" in prompt
        assert "Level 3 永久禁止" in prompt


def test_prompt_contains_generator_and_reviewer_sections(tmp_path: Path) -> None:
    ctx = _ctx(lab=Lab.SINGLE_FACTOR, draft_dir=tmp_path, vault_root=tmp_path)
    prompt = build_prompt(engine=Engine.CLAUDE, ctx=ctx)
    assert "你的任务（generator + reviewer 合一）" in prompt
    assert "Part A" in prompt
    assert "Part B" in prompt
    assert "Part A — Mechanism candidates (generator)" in prompt
    assert "Part B — Code feasibility review (reviewer)" in prompt
    assert "stage1_claude.md" in prompt


def test_prompt_carries_codebase_and_validator_rules(tmp_path: Path) -> None:
    ctx = _ctx(lab=Lab.SINGLE_FACTOR, draft_dir=tmp_path, vault_root=tmp_path)
    prompt = build_prompt(engine=Engine.CODEX, ctx=ctx)
    assert "alpha_one" in prompt
    assert "alpha_promoted" in prompt
    assert "ridge_smoke" in prompt
    assert "alpha_one_v1" in prompt
    assert "factor.json required keys" in prompt
    assert "factor validator 硬规则" in prompt
    assert "in_contract_factor_def" in prompt


def test_prompt_uses_model_lab_schema_when_lab_is_model_factor(tmp_path: Path) -> None:
    ctx = _ctx(lab=Lab.MODEL_FACTOR, draft_dir=tmp_path, vault_root=tmp_path)
    prompt = build_prompt(engine=Engine.CLAUDE, ctx=ctx)
    assert "ModelFactorCaseSpec top-level fields" in prompt
    assert "model validator 硬规则" in prompt
    assert "in_contract_spec_variant" in prompt
    assert "factor.json required keys" not in prompt


# ---------------------------------------------------------------------------
# codebase_index
# ---------------------------------------------------------------------------


def test_codebase_snapshot_walks_research_and_promoted(tmp_path: Path) -> None:
    workspace = _build_workspace(tmp_path)
    snap = build_codebase_snapshot(workspace)
    assert "alpha_one" in snap.factors_research
    assert "alpha_promoted" in snap.factors_promoted
    assert "ridge_smoke" in snap.model_candidates_research
    assert "alpha_one_v1" in snap.single_factor_cases
    assert snap.model_factor_cases == ()
    assert "name" in snap.factor_json_required_keys
    assert "feature_columns" in snap.model_case_spec_top_keys
    assert any("snake_case" in rule for rule in snap.factor_validator_rules)
    assert any(
        "ModelFactorCaseSpec" in rule or "spec_variant" in rule
        for rule in snap.model_validator_rules
    )


def test_codebase_snapshot_handles_missing_dirs(tmp_path: Path) -> None:
    snap = build_codebase_snapshot(tmp_path / "does_not_exist")
    assert snap.factors_research == ()
    assert snap.model_candidates_research == ()
    assert snap.single_factor_cases == ()


# ---------------------------------------------------------------------------
# distribute_idea (single-factor) end-to-end — new 5-file layout
# ---------------------------------------------------------------------------


def test_distribute_idea_writes_5_file_layout(tmp_path: Path) -> None:
    vault = _build_vault(tmp_path)
    workspace = _build_workspace(tmp_path)

    result = distribute_idea(
        vault_root=vault,
        idea="intraday turnover regime",
        lab=Lab.SINGLE_FACTOR,
        workspace_root=workspace,
        top_k=2,
    )

    assert result.lab is Lab.SINGLE_FACTOR
    assert result.draft_dir.exists()
    # Canonical 5-file layout under ideas/<idea_id>/.
    expected = {
        "manifest.json",
        "retrieval_pack.md",
        "prompt_claude.md",
        "prompt_codex.md",
        "stage2_input.md",
    }
    actual = {p.name for p in result.draft_dir.iterdir()}
    assert actual == expected, f"unexpected layout: {actual ^ expected}"

    # Legacy files must not appear
    legacy_names = {
        "retrieval_log.md",
        "reconcile.md",
        "ledger_v1.yaml",
        "prompt_claude_mechanism.md",
        "prompt_codex_review.md",
        "mechanism_deepdive.md",
        "code_feasibility_review.md",
    }
    assert not (actual & legacy_names)

    # Manifest carries idea_id + codebase snapshot.
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["idea_id"] == result.idea_id
    assert manifest["lab"] == "single_factor"
    assert manifest["engines"] == ["claude", "codex"]
    assert "codebase_snapshot" in manifest
    assert manifest["codebase_snapshot"]["factors"]["research"] == ["alpha_one"]

    # Both prompts contain identity self-id and refer to their output filename.
    claude = result.engine_prompt_paths[Engine.CLAUDE].read_text(encoding="utf-8")
    codex = result.engine_prompt_paths[Engine.CODEX].read_text(encoding="utf-8")
    assert "Claude Code" in claude
    assert "Codex GUI" in codex
    assert "stage1_claude.md" in claude
    assert "stage1_codex.md" in codex


def test_distribute_idea_stage2_input_references_sources(tmp_path: Path) -> None:
    """stage2_input.md must point at the 3 web-GPT source files."""

    vault = _build_vault(tmp_path)
    workspace = _build_workspace(tmp_path)
    result = distribute_idea(
        vault_root=vault, idea="contract guard", workspace_root=workspace, top_k=2
    )
    text = result.stage2_input_path.read_text(encoding="utf-8")
    assert "single_factor_source_pack.md" in text
    assert "single_factor_stage1_reconcile_contract.md" in text
    assert "single_factor_stage2_candidate_contract.md" in text
    assert "factor_json_payload" in text
    assert "provenance.idea_id" in text


# ---------------------------------------------------------------------------
# distribute_model_idea wrapper
# ---------------------------------------------------------------------------


def test_distribute_model_idea_uses_model_lab(tmp_path: Path) -> None:
    vault = _build_vault(tmp_path)
    workspace = _build_workspace(tmp_path)

    result = distribute_model_idea(
        idea="ridge feature interactions",
        workspace_root=workspace,
        vault_root=vault,
        top_k=2,
    )
    assert result.lab is Lab.MODEL_FACTOR
    claude = result.engine_prompt_paths[Engine.CLAUDE].read_text(encoding="utf-8")
    assert "ModelFactorCaseSpec top-level fields" in claude
    assert "factor.json required keys" not in claude
    stage2 = result.stage2_input_path.read_text(encoding="utf-8")
    assert "model_lab_stage1_reconcile_contract.md" in stage2
    assert "model_lab_stage2_candidate_contract.md" in stage2
    assert "model_candidate_payload" in stage2


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


def test_unified_cli_routes_idea_distribute(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def _fake_distribute_idea(**kwargs):
        captured.update(kwargs)

        class _Stub:
            lab = Lab.SINGLE_FACTOR
            idea_id = "fake_id"
            stage = "mechanism_discovery"
            engines = (Engine.CLAUDE, Engine.CODEX)
            draft_dir = tmp_path
            manifest_path = tmp_path / "manifest.json"
            retrieval_pack_path = tmp_path / "retrieval_pack.md"
            stage2_input_path = tmp_path / "stage2_input.md"

        return _Stub()

    monkeypatch.setattr(
        "alpha_lab.research_bridge.service.distribute_idea", _fake_distribute_idea
    )

    from alpha_lab.cli import main as unified_main

    rc = unified_main(
        [
            "idea",
            "distribute",
            "--idea",
            "regime conditioned reversal",
            "--engines",
            "claude,codex",
            "--lab",
            "single_factor",
            "--vault-root",
            "/tmp/vault",
            "--workspace-root",
            str(tmp_path),
        ]
    )
    assert rc == 0
    assert captured["idea"] == "regime conditioned reversal"
    assert captured["engines"] == "claude,codex"
    assert captured["lab"] == Lab.SINGLE_FACTOR
    assert captured["vault_root"] == "/tmp/vault"


# ---------------------------------------------------------------------------
# Stage 4 experiment-card
# ---------------------------------------------------------------------------


def test_scaffold_experiment_card_writes_skeleton(tmp_path: Path) -> None:
    vault = _build_vault(tmp_path)
    workspace = _build_workspace(tmp_path)
    result = distribute_idea(
        vault_root=vault, idea="card test", workspace_root=workspace, top_k=2
    )
    card_path = scaffold_experiment_card(
        idea_id=result.idea_id,
        outcome=ExperimentCardOutcome.KILLED,
        workspace_root=workspace,
    )
    text = card_path.read_text(encoding="utf-8")
    assert f"idea_id: {result.idea_id}" in text
    assert "outcome: killed" in text
    assert "## emergent_moves" in text
    assert "## operative_claims" in text


def test_scaffold_experiment_card_refuses_overwrite(tmp_path: Path) -> None:
    vault = _build_vault(tmp_path)
    workspace = _build_workspace(tmp_path)
    result = distribute_idea(
        vault_root=vault, idea="overwrite", workspace_root=workspace, top_k=2
    )
    scaffold_experiment_card(
        idea_id=result.idea_id,
        outcome=ExperimentCardOutcome.PARKED,
        workspace_root=workspace,
    )
    with pytest.raises(FileExistsError):
        scaffold_experiment_card(
            idea_id=result.idea_id,
            outcome=ExperimentCardOutcome.KILLED,
            workspace_root=workspace,
        )


def test_scaffold_experiment_card_cleanup_keeps_only_manifest_and_card(
    tmp_path: Path,
) -> None:
    vault = _build_vault(tmp_path)
    workspace = _build_workspace(tmp_path)
    result = distribute_idea(
        vault_root=vault, idea="cleanup", workspace_root=workspace, top_k=2
    )
    # Before cleanup: 5 files
    assert len(list(result.draft_dir.iterdir())) == 5
    scaffold_experiment_card(
        idea_id=result.idea_id,
        outcome=ExperimentCardOutcome.PROMOTED,
        workspace_root=workspace,
        cleanup=True,
    )
    after = {p.name for p in result.draft_dir.iterdir()}
    assert after == {"manifest.json", "experiment_card.md"}


def test_scaffold_experiment_card_missing_idea_dir_raises(tmp_path: Path) -> None:
    workspace = _build_workspace(tmp_path)
    with pytest.raises(FileNotFoundError):
        scaffold_experiment_card(
            idea_id="nonexistent",
            outcome=ExperimentCardOutcome.KILLED,
            workspace_root=workspace,
        )
