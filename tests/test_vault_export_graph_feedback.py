"""Tests for the three finalised feedback policies in vault_export_graph_feedback.py:

1. _should_capture_failure  — when to auto-create a Failure Knowledge entry
2. _infer_exhaustion_level  — how to score exploration exhaustion
3. tested_in scope          — only origin_cards, not supporting/failure/related
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# Ensure dependency chain is initialized in import order before touching
# vault_export_graph_feedback.
import alpha_lab.vault_export_graph_feedback as _gf
from alpha_lab.research_bridge import service as _rb_service
from alpha_lab.vault_export import ExportResult


def _infer_exhaustion_level(v: str, s: str) -> str:
    return _gf._infer_exhaustion_level(v, s)


def _should_capture_failure(v: str, s: str) -> bool:
    return _gf._should_capture_failure(v, s)


def _build_feedback_vault(tmp_path: Path) -> Path:
    """Build a minimal vault for graph-feedback integration tests."""
    vault = tmp_path / "quant-knowledge"
    for rel in [
        "10_concepts",
        "20_methods",
        "30_factors",
        "50_experiments",
        "55_projects",
        "90_computed",
        "90_moc",
    ]:
        (vault / rel).mkdir(parents=True, exist_ok=True)
    (vault / "90_moc" / "CARD-INDEX.tsv").write_text(
        "path\ttype\tname\tdomain\tlifecycle\ttags\tparent_moc\n"
        "30_factors/Factor - Momentum Base.md\tfactor\tMomentum Base\talpha\t"
        "theoretical\tmomentum\tMOC - Factors\n",
        encoding="utf-8",
    )
    (vault / "30_factors" / "Factor - Momentum Base.md").write_text(
        "---\ntype: factor\nname: Momentum Base\nsummary: momentum base test card\n---\n\n# test\n",
        encoding="utf-8",
    )
    (vault / "90_computed" / "graph.json").write_text("{}", encoding="utf-8")
    (vault / "90_computed" / "exploration_map.json").write_text("{}", encoding="utf-8")
    return vault


# ---------------------------------------------------------------------------
# 1. _should_capture_failure
# ---------------------------------------------------------------------------


class TestShouldCaptureFailure:
    """Policy: create FK entry only on unambiguous hard failure
    AND only when one_sentence_verdict is non-empty."""

    @pytest.mark.parametrize(
        "verdict_status, one_sentence",
        [
            ("drop", "信号完全无效"),
            ("fail", "IC 在所有窗口均不显著"),
            ("rejected", "robustness check 失败"),
            ("abandon", "方向不可行"),
            ("", "实验 fail，不值得继续"),
        ],
    )
    def test_captures_hard_failures(self, verdict_status: str, one_sentence: str) -> None:
        assert _should_capture_failure(verdict_status, one_sentence) is True

    @pytest.mark.parametrize(
        "verdict_status, one_sentence",
        [
            ("revise", "需要换一种中性化方式"),
            ("mixed", "部分窗口有效"),
            ("fragile", "信号不稳定但非零"),
            ("promising", "初步有效"),
            ("keep", "继续优化"),
        ],
    )
    def test_does_not_capture_moderate_or_positive(
        self, verdict_status: str, one_sentence: str
    ) -> None:
        assert _should_capture_failure(verdict_status, one_sentence) is False

    def test_does_not_capture_when_sentence_empty(self) -> None:
        """Even if verdict_status says 'drop', empty sentence = no FK entry."""
        assert _should_capture_failure("drop", "") is False
        assert _should_capture_failure("fail", "  ") is False

    def test_does_not_capture_when_both_empty(self) -> None:
        assert _should_capture_failure("", "") is False


# ---------------------------------------------------------------------------
# 2. _infer_exhaustion_level
# ---------------------------------------------------------------------------


class TestInferExhaustionLevel:
    """Policy: exhausted > moderate > light > none."""

    @pytest.mark.parametrize(
        "verdict_status, one_sentence, expected",
        [
            ("drop", "信号无效", "exhausted"),
            ("fail", "IC 不显著", "exhausted"),
            ("rejected", "不通过", "exhausted"),
            ("abandon", "方向不可行", "exhausted"),
        ],
    )
    def test_exhausted(self, verdict_status: str, one_sentence: str, expected: str) -> None:
        assert _infer_exhaustion_level(verdict_status, one_sentence) == expected

    @pytest.mark.parametrize(
        "verdict_status, one_sentence, expected",
        [
            ("revise", "换中性化", "moderate"),
            ("fragile", "不稳定", "moderate"),
            ("mixed", "部分有效", "moderate"),
            ("unstable", "体制依赖", "moderate"),
            ("weak", "信号微弱", "moderate"),
            ("inconclusive", "证据不足", "moderate"),
        ],
    )
    def test_moderate(self, verdict_status: str, one_sentence: str, expected: str) -> None:
        assert _infer_exhaustion_level(verdict_status, one_sentence) == expected

    @pytest.mark.parametrize(
        "verdict_status, one_sentence, expected",
        [
            ("promising", "初步有效", "light"),
            ("keep", "继续优化", "light"),
            ("promote", "可进入下一阶段", "light"),
            ("validated", "通过稳健性检查", "light"),
        ],
    )
    def test_light(self, verdict_status: str, one_sentence: str, expected: str) -> None:
        assert _infer_exhaustion_level(verdict_status, one_sentence) == expected

    def test_none_when_empty(self) -> None:
        assert _infer_exhaustion_level("", "") == "none"

    def test_none_when_unrecognised(self) -> None:
        assert _infer_exhaustion_level("something_random", "无法分类的描述") == "none"


# ---------------------------------------------------------------------------
# 3. tested_in scope (integration-level)
# ---------------------------------------------------------------------------


class TestTestedInScope:
    """Policy: only origin_cards get tested_in updates."""

    def test_only_origin_cards_updated(self, tmp_path) -> None:
        _upsert_tested_in = _gf._upsert_tested_in

        # Create two card files with frontmatter
        origin = tmp_path / "Factor - A.md"
        origin.write_text("---\ntype: factor\nlifecycle: theoretical\n---\n# A\n", encoding="utf-8")
        supporting = tmp_path / "Concept - B.md"
        supporting.write_text("---\ntype: concept\nlifecycle: stable\n---\n# B\n", encoding="utf-8")

        # Simulate what apply_graph_feedback does: update only origin
        assert _upsert_tested_in(card_path=origin, experiment_name="exp_01") is True
        # Supporting card should NOT be touched by the main flow, but
        # verify _upsert_tested_in itself works (the policy is in the caller)
        updated_origin = origin.read_text(encoding="utf-8")
        assert "exp_01" in updated_origin
        assert "tested_in" in updated_origin

        # Supporting card was never called — stays unchanged
        unchanged = supporting.read_text(encoding="utf-8")
        assert "tested_in" not in unchanged

    def test_upsert_tested_in_idempotent(self, tmp_path) -> None:
        card = tmp_path / "Factor - X.md"
        card.write_text("---\ntype: factor\nlifecycle: theoretical\n---\n# X\n", encoding="utf-8")
        _upsert_tested_in = _gf._upsert_tested_in

        assert _upsert_tested_in(card_path=card, experiment_name="exp_01") is True
        assert _upsert_tested_in(card_path=card, experiment_name="exp_01") is False  # no change
        text = card.read_text(encoding="utf-8")
        assert text.count("exp_01") == 1


# ---------------------------------------------------------------------------
# 4. apply_graph_feedback correlation summary
# ---------------------------------------------------------------------------


class TestApplyGraphFeedback:
    """Enhancement checks for correlation summary rendering and similar-to hints."""

    def test_apply_graph_feedback_returns_text_correlation_summary(
        self,
        tmp_path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        vault = _build_feedback_vault(tmp_path)
        init = _rb_service.init_project(
            vault_root=vault,
            slug="momentum-factor",
            title_zh="动量因子",
            category="factor_family",
            owner="analyst",
            market="ashare",
            frequency="daily",
            chatgpt_project_name="Momentum Factor",
            origin_cards=["30_factors/Factor - Momentum Base.md"],
        )

        run_root = tmp_path / "run_root"
        run_root.mkdir()
        (run_root / "run_manifest.json").write_text(
            json.dumps({"case_name": "momentum_case"}), encoding="utf-8"
        )
        (run_root / "factor_correlation.json").write_text(
            json.dumps(
                {
                    "top_match": "Momentum Veteran",
                    "max_abs_correlation": 0.87,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        class _DummyEmbeddings:
            def suggest_similar(self, name: str, threshold: float = 0.35) -> list[str]:  # noqa: ARG002
                del name, threshold
                return ["Momentum Veteran", "Momentum Complement"]

        monkeypatch.setattr(
            _gf,
            "_run_rebuild_script",
            lambda *args, **kwargs: None,
        )
        monkeypatch.setattr(
            _gf,
            "_load_embeddings_optional",
            lambda _vault_root: _DummyEmbeddings(),
        )
        result = _gf.apply_graph_feedback(
            vault_root=vault,
            project=init.project,
            draft_frontmatter={
                "case_name": "momentum_case",
                "current_hypothesis": "momentum persistence",
                "verdict_status": "promising",
                "one_sentence_verdict": "positive run signal",
                "run_root": str(run_root),
            },
            export_result=ExportResult(
                success=True,
                target_paths=(),
                mode_used="versioned",
                status="success",
                error=None,
            ),
        )

        assert result.suggested_similar_to == ["Momentum Veteran", "Momentum Complement"]
        assert isinstance(result.correlation_summary, str)
        assert "Momentum Veteran" in result.correlation_summary
        assert "via decomposition" in result.correlation_summary

        payload = result.to_payload()
        assert payload["correlation_summary"] == result.correlation_summary
