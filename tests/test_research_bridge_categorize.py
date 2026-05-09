from __future__ import annotations

import sys
import types
from typing import Any

from alpha_lab.research_bridge.llm_rerank import (
    CategorizedCandidate,
    RerankCandidate,
    categorize_and_compress,
)


def _install_fake_anthropic(
    monkeypatch: Any,
    *,
    response: object,
) -> None:
    class FakeMessages:
        def create(self, **_: object) -> object:
            return response

    class FakeAnthropic:
        def __init__(self, *, api_key: str, base_url: str | None = None) -> None:
            self.api_key = api_key
            self.base_url = base_url
            self.messages = FakeMessages()

    monkeypatch.setitem(
        sys.modules,
        "anthropic",
        types.SimpleNamespace(Anthropic=FakeAnthropic),
    )


def _response(
    items: list[dict[str, object]],
    *,
    insight_brief: list[str] | None = None,
    usage: dict[str, int] | None = None,
) -> dict[str, object]:
    return {
        "content": [
            {
                "type": "tool_use",
                "name": "submit_categorized_relevance",
                "input": {
                    "items": items,
                    "insight_brief": insight_brief or [],
                },
            }
        ],
        "usage": usage or {},
    }


def test_categorize_invalid_path_dropped(monkeypatch: Any) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    _install_fake_anthropic(
        monkeypatch,
        response=_response(
            [
                {
                    "path": "ghost.md",
                    "name": "Ghost",
                    "category": "transferable",
                    "relevance": 0.9,
                    "transferable_to_idea": "bad",
                    "key_warning": "",
                }
            ]
        ),
    )

    outcome = categorize_and_compress(
        idea="idea",
        mode="start",
        candidates=[
            RerankCandidate(
                name="Real Card",
                summary="summary",
                domain="price_action",
                mechanism="behavioral",
                factor_family="momentum",
                path="real.md",
            )
        ],
        fingerprints={},
        provenance={},
    )

    assert outcome.enabled is True
    assert outcome.categorized == []
    assert outcome.dropped_invalid_paths == ["ghost.md"]


def test_categorize_clamps_and_emits_insight_brief(monkeypatch: Any) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    candidate = RerankCandidate(
        name="Momentum Crash",
        summary="summary",
        domain="price_action",
        mechanism="crowding",
        factor_family="momentum",
        path="30_factors/Factor - Momentum Crash.md",
    )
    _install_fake_anthropic(
        monkeypatch,
        response=_response(
            [
                {
                    "path": candidate.path,
                    "name": candidate.name,
                    "category": "transferable",
                    "relevance": 1.4,
                    "transferable_to_idea": "拥挤解除可迁移",
                    "key_warning": "泡沫延续",
                }
            ],
            insight_brief=["高估值要叠加拥挤和预期过满。"],
            usage={"input_tokens": 20, "output_tokens": 5},
        ),
    )

    outcome = categorize_and_compress(
        idea="做空高估值",
        mode="start",
        candidates=[candidate],
        fingerprints={},
        provenance={candidate.path: ["mechanism:拥挤解除@mechanism"]},
    )

    assert outcome.enabled is True
    assert outcome.categorized == [
        CategorizedCandidate(
            path=candidate.path,
            name=candidate.name,
            category="transferable",
            relevance=1.0,
            transferable_to_idea="拥挤解除可迁移",
            key_warning="泡沫延续",
        )
    ]
    assert outcome.insight_brief == ["高估值要叠加拥挤和预期过满。"]
    assert outcome.tokens_input == 20
    assert outcome.tokens_output == 5
