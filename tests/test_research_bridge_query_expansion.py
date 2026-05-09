from __future__ import annotations

import sys
import types
from typing import Any

from alpha_lab.research_bridge.query_expansion import (
    QueryProbes,
    _enforce_mode_constraints,
    expand_query,
)


def _response(payload: dict[str, object]) -> dict[str, object]:
    return {
        "content": [
            {
                "type": "tool_use",
                "name": "submit_query_probes",
                "input": payload,
            }
        ],
        "usage": {
            "input_tokens": 10,
            "cache_read_input_tokens": 5,
            "output_tokens": 3,
        },
    }


def _install_fake_anthropic(
    monkeypatch: Any,
    *,
    response: object | None = None,
    error: Exception | None = None,
) -> None:
    class FakeMessages:
        def create(self, **_: object) -> object:
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


def test_no_api_key_falls_back_to_idea_only(monkeypatch: Any) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delitem(sys.modules, "anthropic", raising=False)

    outcome = expand_query(idea="做空高估值", mode="start")

    assert outcome.enabled is False
    assert outcome.probes.direct == ["做空高估值"]
    assert outcome.probes.mechanism == []
    assert outcome.fallback_reason == "no_api_key"
    assert "anthropic" not in sys.modules


def test_constrained_mode_clears_loose_probe_classes() -> None:
    probes = QueryProbes(
        direct=["高估值"],
        mechanism=["预期过满"],
        analogy=["carry unwind"],
        failure=["泡沫延续"],
        construction=["估值分位"],
    )

    constrained = _enforce_mode_constraints(probes, "constrained")

    assert constrained.direct == ["高估值"]
    assert constrained.mechanism == ["预期过满"]
    assert constrained.analogy == []
    assert constrained.failure == []
    assert constrained.construction == []


def test_api_success_returns_budgeted_probes(monkeypatch: Any) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    _install_fake_anthropic(
        monkeypatch,
        response=_response(
            {
                "direct": ["a", "b", "c", "d"],
                "mechanism": ["m1", "m2", "m3", "m4"],
                "analogy": ["x"],
                "failure": ["f"],
                "construction": ["c"],
            }
        ),
    )

    outcome = expand_query(idea="idea", mode="explore")

    assert outcome.enabled is True
    assert outcome.probes.direct == ["a", "b", "c"]
    assert outcome.probes.mechanism == ["m1", "m2", "m3"]
    assert outcome.cache_hit_input == 5


def test_api_error_fallback(monkeypatch: Any) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    _install_fake_anthropic(monkeypatch, error=RuntimeError("boom"))

    outcome = expand_query(idea="idea", mode="start")

    assert outcome.enabled is False
    assert outcome.probes.direct == ["idea"]
    assert outcome.fallback_reason is not None
    assert "RuntimeError" in outcome.fallback_reason
