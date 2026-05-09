from __future__ import annotations

import sys
import types
from typing import Any

from alpha_lab.research_bridge.llm_rerank import (
    DEFAULT_MODEL,
    RerankCandidate,
    anthropic_client_kwargs,
    rerank_candidates,
)


def _candidate(name: str) -> RerankCandidate:
    return RerankCandidate(
        name=name,
        summary=f"{name} summary",
        domain="price_action",
        mechanism="behavioral",
        factor_family="momentum",
    )


def _response(
    items: list[dict[str, object]],
    *,
    usage: dict[str, int] | None = None,
) -> dict[str, object]:
    return {
        "content": [
            {
                "type": "tool_use",
                "name": "submit_relevance",
                "input": {"items": items},
            }
        ],
        "usage": usage or {},
    }


def _categorize_response(
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


def _install_fake_anthropic(
    monkeypatch: Any,
    *,
    response: object | None = None,
    error: Exception | None = None,
    calls: list[dict[str, object]] | None = None,
) -> None:
    call_log = calls if calls is not None else []

    class FakeMessages:
        def create(self, **kwargs: object) -> object:
            call_log.append(kwargs)
            if error is not None:
                raise error
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


def test_no_api_key_returns_disabled(monkeypatch: Any) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delitem(sys.modules, "anthropic", raising=False)

    outcome = rerank_candidates(
        idea="short expensive stocks",
        mode="start",
        candidates=[_candidate("Value Crash")],
    )

    assert outcome.enabled is False
    assert outcome.model == DEFAULT_MODEL
    assert outcome.scores == {}
    assert outcome.fallback_reason == "no_api_key"
    assert "anthropic" not in sys.modules


def test_empty_candidates_short_circuits(monkeypatch: Any) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delitem(sys.modules, "anthropic", raising=False)

    outcome = rerank_candidates(idea="idea", mode="explore", candidates=[])

    assert outcome.enabled is False
    assert outcome.fallback_reason == "no_candidates"
    assert "anthropic" not in sys.modules


def test_anthropic_client_kwargs_include_optional_base_url(monkeypatch: Any) -> None:
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://anthropic-proxy.example/v1")

    assert anthropic_client_kwargs("test-key") == {
        "api_key": "test-key",
        "base_url": "https://anthropic-proxy.example/v1",
    }


def test_invalid_name_dropped(monkeypatch: Any) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    _install_fake_anthropic(
        monkeypatch,
        response=_response([{"name": "幽灵卡", "relevance": 0.9, "reason": "bad"}]),
    )

    outcome = rerank_candidates(
        idea="idea",
        mode="start",
        candidates=[_candidate("Real Card")],
    )

    assert outcome.enabled is True
    assert outcome.scores == {}
    assert outcome.dropped_invalid_names == ["幽灵卡"]


def test_relevance_clamped_to_unit_interval(monkeypatch: Any) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    _install_fake_anthropic(
        monkeypatch,
        response=_response(
            [
                {"name": "High", "relevance": 1.5, "reason": "too high"},
                {"name": "Low", "relevance": -0.2, "reason": "too low"},
            ]
        ),
    )

    outcome = rerank_candidates(
        idea="idea",
        mode="explore",
        candidates=[_candidate("High"), _candidate("Low")],
    )

    assert outcome.scores["High"] == 1.0
    assert outcome.scores["Low"] == 0.0


def test_missing_candidate_defaults_to_zero(monkeypatch: Any) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    _install_fake_anthropic(
        monkeypatch,
        response=_response(
            [{"name": "Only Scored", "relevance": 0.75, "reason": "direct"}]
        ),
    )

    outcome = rerank_candidates(
        idea="idea",
        mode="explore",
        candidates=[
            _candidate("Only Scored"),
            _candidate("Missing A"),
            _candidate("Missing B"),
        ],
    )

    assert outcome.scores["Only Scored"] == 0.75
    assert outcome.scores.get("Missing A", 0.0) == 0.0
    assert outcome.scores.get("Missing B", 0.0) == 0.0


def test_api_error_fallback(monkeypatch: Any) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    _install_fake_anthropic(monkeypatch, error=RuntimeError("boom"))

    outcome = rerank_candidates(
        idea="idea",
        mode="constrained",
        candidates=[_candidate("Real Card")],
    )

    assert outcome.enabled is False
    assert outcome.scores == {}
    assert outcome.fallback_reason is not None
    assert "RuntimeError" in outcome.fallback_reason


def test_cache_token_accounting(monkeypatch: Any) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    _install_fake_anthropic(
        monkeypatch,
        response=_response(
            [{"name": "Real Card", "relevance": 0.5, "reason": "some"}],
            usage={
                "input_tokens": 100,
                "cache_creation_input_tokens": 20,
                "cache_read_input_tokens": 900,
                "output_tokens": 30,
            },
        ),
    )

    outcome = rerank_candidates(
        idea="idea",
        mode="start",
        candidates=[_candidate("Real Card")],
    )

    assert outcome.tokens_input == 1020
    assert outcome.tokens_output == 30
    assert outcome.cache_hit_input == 900


def test_mode_routing_to_distinct_rubric(monkeypatch: Any) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    calls: list[dict[str, object]] = []
    _install_fake_anthropic(
        monkeypatch,
        response=_response(
            [{"name": "Real Card", "relevance": 0.5, "reason": "some"}]
        ),
        calls=calls,
    )

    rerank_candidates(idea="idea", mode="start", candidates=[_candidate("Real Card")])
    rerank_candidates(
        idea="idea", mode="constrained", candidates=[_candidate("Real Card")]
    )

    start_system = calls[0]["system"]
    constrained_system = calls[1]["system"]
    assert isinstance(start_system, list)
    assert isinstance(constrained_system, list)
    assert "跨领域" in str(start_system[0])
    assert "同主题" in str(constrained_system[0])
