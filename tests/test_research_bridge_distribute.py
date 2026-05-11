"""Tests for the new Stage 0 audience-based distribution.

Covers:
- ``audience_prompts.build_prompt`` lab × audience surface
- ``codebase_index.build_codebase_snapshot`` directory walk
- ``service.distribute_idea`` end-to-end (writes ideas/<id>/ layout)
- ``model_idea.distribute_model_idea`` thin wrapper (lab=model_factor)
- CLI ``alpha-lab idea distribute`` routing

These tests cover the protocol shift from docs/research_workflow.md: Stage 1
is generator + reviewer, not two generators. Reviewer prompts must include
the codebase snapshot + schema + validator hard rules.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from alpha_lab.research_bridge.audience_prompts import (
    Audience,
    CardForPrompt,
    Lab,
    PromptContext,
    build_prompt,
    normalize_audiences,
)
from alpha_lab.research_bridge.codebase_index import (
    CodebaseSnapshot,
    build_codebase_snapshot,
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


# ---------------------------------------------------------------------------
# audience_prompts
# ---------------------------------------------------------------------------


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


def test_normalize_audiences_default_returns_both() -> None:
    assert normalize_audiences(None) == (
        Audience.CLAUDE_MECHANISM,
        Audience.CODEX_REVIEW,
    )


def test_normalize_audiences_parses_csv() -> None:
    assert normalize_audiences("codex_review,claude_mechanism") == (
        Audience.CODEX_REVIEW,
        Audience.CLAUDE_MECHANISM,
    )


def test_normalize_audiences_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="unknown audience"):
        normalize_audiences("rogue_audience")


def test_generator_prompt_does_not_leak_codebase_index(tmp_path: Path) -> None:
    ctx = _ctx(lab=Lab.SINGLE_FACTOR, draft_dir=tmp_path, vault_root=tmp_path)
    prompt = build_prompt(audience=Audience.CLAUDE_MECHANISM, ctx=ctx)
    assert "Stage 1 Generator Prompt" in prompt
    assert "claude_mechanism" in prompt
    assert "mechanism_deepdive.md" in prompt
    # Generator does not see codebase index, schema, validator rules.
    assert "代码库索引" not in prompt
    assert "factor.json required keys" not in prompt
    assert "factor validator 硬规则" not in prompt
    # Generator does not write a review or judge feasibility.
    assert "code_feasibility_review.md" not in prompt
    assert "implementation_status" not in prompt


def test_reviewer_prompt_carries_codebase_and_validator(tmp_path: Path) -> None:
    ctx = _ctx(lab=Lab.SINGLE_FACTOR, draft_dir=tmp_path, vault_root=tmp_path)
    prompt = build_prompt(audience=Audience.CODEX_REVIEW, ctx=ctx)
    assert "Stage 1 Reviewer Prompt" in prompt
    assert "codex_review" in prompt
    assert "code_feasibility_review.md" in prompt
    # Reviewer sees codebase index + schema + validator rules.
    assert "alpha_one" in prompt
    assert "alpha_promoted" in prompt
    assert "ridge_smoke" in prompt
    assert "alpha_one_v1" in prompt
    assert "factor.json required keys" in prompt
    assert "factor validator 硬规则" in prompt
    # Reviewer-specific status enum.
    assert "in_contract_factor_def" in prompt
    # Reviewer does not propose new mechanisms.
    assert "你**不写**新机制" in prompt


def test_reviewer_prompt_uses_model_lab_schema_when_lab_is_model_factor(
    tmp_path: Path,
) -> None:
    ctx = _ctx(lab=Lab.MODEL_FACTOR, draft_dir=tmp_path, vault_root=tmp_path)
    prompt = build_prompt(audience=Audience.CODEX_REVIEW, ctx=ctx)
    assert "ModelFactorCaseSpec top-level fields" in prompt
    assert "model validator 硬规则" in prompt
    assert "in_contract_spec_variant" in prompt
    # Single-factor schema must not appear when lab is model_factor.
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
    # Schema cheatsheets are populated from the validator-side constants.
    assert "name" in snap.factor_json_required_keys
    assert "feature_columns" in snap.model_case_spec_top_keys
    # Validator rule cheatsheets non-empty.
    assert any("snake_case" in rule for rule in snap.factor_validator_rules)
    assert any("ModelFactorCaseSpec" in rule or "spec_variant" in rule
               for rule in snap.model_validator_rules)


def test_codebase_snapshot_handles_missing_dirs(tmp_path: Path) -> None:
    snap = build_codebase_snapshot(tmp_path / "does_not_exist")
    assert snap.factors_research == ()
    assert snap.model_candidates_research == ()
    assert snap.single_factor_cases == ()


# ---------------------------------------------------------------------------
# distribute_idea (single-factor) end-to-end
# ---------------------------------------------------------------------------


def test_distribute_idea_writes_canonical_ideas_layout(tmp_path: Path) -> None:
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
    assert result.idea_id.endswith("__intraday-turnover-regime") or "__" in result.idea_id
    assert result.draft_dir.exists()
    # Canonical files under ideas/<idea_id>/.
    expected_names = {
        "manifest.json",
        "retrieval_pack.md",
        "retrieval_log.md",
        "reconcile.md",
        f"prompt_{Audience.CLAUDE_MECHANISM.value}.md",
        f"prompt_{Audience.CODEX_REVIEW.value}.md",
    }
    assert expected_names.issubset({p.name for p in result.draft_dir.iterdir()})

    # Manifest references the codebase snapshot so reviewers can audit it.
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["idea_id"] == result.idea_id
    assert manifest["lab"] == "single_factor"
    assert manifest["audiences"] == ["claude_mechanism", "codex_review"]
    assert "codebase_snapshot" in manifest
    assert manifest["codebase_snapshot"]["factors"]["research"] == ["alpha_one"]

    # Reviewer prompt contains the codebase listings; generator prompt does not.
    reviewer = result.audience_prompt_paths[Audience.CODEX_REVIEW].read_text(
        encoding="utf-8"
    )
    assert "alpha_one" in reviewer
    assert "ridge_smoke" in reviewer
    generator = result.audience_prompt_paths[Audience.CLAUDE_MECHANISM].read_text(
        encoding="utf-8"
    )
    assert "代码库索引" not in generator


def test_distribute_idea_does_not_call_legacy_ledger_layout(tmp_path: Path) -> None:
    """No `dispatch.<model>.md` / `ledger_v1.<model>.yaml` legacy files."""

    vault = _build_vault(tmp_path)
    workspace = _build_workspace(tmp_path)
    result = distribute_idea(
        vault_root=vault,
        idea="legacy guard",
        workspace_root=workspace,
        top_k=2,
    )
    names = {p.name for p in result.draft_dir.iterdir()}
    assert not any(name.startswith("dispatch.") for name in names)
    assert not any(name.startswith("ledger_v1.") and name != "ledger_v1.yaml"
                   for name in names)


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
    reviewer = result.audience_prompt_paths[Audience.CODEX_REVIEW].read_text(
        encoding="utf-8"
    )
    # Model-lab reviewer prompt is scoped to ModelFactorCaseSpec, not factor.json.
    assert "ModelFactorCaseSpec top-level fields" in reviewer
    assert "model validator 硬规则" in reviewer
    assert "factor.json required keys" not in reviewer


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
            audiences = (Audience.CLAUDE_MECHANISM, Audience.CODEX_REVIEW)
            draft_dir = tmp_path
            retrieval_pack_path = tmp_path / "retrieval_pack.md"
            reconcile_path = tmp_path / "reconcile.md"
            manifest_path = tmp_path / "manifest.json"

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
            "--audiences",
            "claude_mechanism,codex_review",
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
    assert captured["audiences"] == "claude_mechanism,codex_review"
    assert captured["lab"] == Lab.SINGLE_FACTOR
    assert captured["vault_root"] == "/tmp/vault"
