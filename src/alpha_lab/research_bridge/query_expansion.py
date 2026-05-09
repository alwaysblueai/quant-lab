from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, replace
from typing import Any

from alpha_lab.research_bridge.llm_rerank import (
    DEFAULT_MODEL,
    anthropic_client_kwargs,
    extract_json_object_from_response,
)

LOG = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class QueryProbes:
    direct: list[str]
    mechanism: list[str]
    analogy: list[str]
    failure: list[str]
    construction: list[str]


@dataclass(frozen=True, slots=True)
class ExpansionOutcome:
    enabled: bool
    model: str
    probes: QueryProbes
    tokens_input: int
    tokens_output: int
    cache_hit_input: int
    fallback_reason: str | None


def expand_query(
    *,
    idea: str,
    mode: str,
    model: str | None = None,
    api_key_env: str = "ANTHROPIC_API_KEY",
) -> ExpansionOutcome:
    """Expand a fuzzy idea into typed retrieval probes.

    This layer is fail-closed: without credentials, or on any SDK/API
    failure, callers receive ``direct=[idea]`` so the v1 literal recall path
    remains intact.
    """
    model_name = str(model or DEFAULT_MODEL)
    normalized_idea = str(idea or "").strip()
    fallback_probes = QueryProbes(
        direct=[normalized_idea] if normalized_idea else [],
        mechanism=[],
        analogy=[],
        failure=[],
        construction=[],
    )
    api_key = os.environ.get(api_key_env)
    if not api_key:
        return _fallback_outcome(
            model=model_name,
            probes=fallback_probes,
            reason="no_api_key",
        )
    if not normalized_idea:
        return _fallback_outcome(
            model=model_name,
            probes=fallback_probes,
            reason="empty_idea",
        )

    try:
        import anthropic  # type: ignore[import-not-found]

        client: Any = anthropic.Anthropic(**anthropic_client_kwargs(api_key))
        request_payload = {"idea": normalized_idea, "mode": str(mode or "").strip()}
        try:
            response = client.messages.create(
                model=model_name,
                system=[
                    {
                        "type": "text",
                        "text": _rubric_for_mode(mode),
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[
                    {
                        "role": "user",
                        "content": json.dumps(request_payload, ensure_ascii=False),
                    }
                ],
                max_tokens=1024,
                temperature=0,
                tools=[_submit_query_probes_tool_schema()],
                tool_choice={"type": "tool", "name": "submit_query_probes"},
            )
            tool_input = _extract_tool_input(response)
        except Exception as tool_exc:
            LOG.info("query expansion tool call failed; trying JSON text mode: %s", tool_exc)
            response = client.messages.create(
                model=model_name,
                system=_json_text_system(_rubric_for_mode(mode)),
                messages=[
                    {
                        "role": "user",
                        "content": json.dumps(request_payload, ensure_ascii=False),
                    }
                ],
                max_tokens=1024,
                temperature=0,
            )
            tool_input = extract_json_object_from_response(response)
        probes = _parse_probes(tool_input)
        probes = _enforce_probe_budget(_enforce_mode_constraints(probes, mode), mode)
        usage = _read_attr(response, "usage", {})
        input_tokens = _usage_int(usage, "input_tokens")
        cache_creation = _usage_int(usage, "cache_creation_input_tokens")
        cache_read = _usage_int(usage, "cache_read_input_tokens")
        output_tokens = _usage_int(usage, "output_tokens")
        return ExpansionOutcome(
            enabled=True,
            model=model_name,
            probes=probes,
            tokens_input=input_tokens + cache_creation + cache_read,
            tokens_output=output_tokens,
            cache_hit_input=cache_read,
            fallback_reason=None,
        )
    except Exception as exc:
        reason = f"api_error: {type(exc).__name__}: {exc}"
        LOG.warning("query expansion failed; falling back: %s", reason)
        return _fallback_outcome(
            model=model_name,
            probes=fallback_probes,
            reason=reason,
        )


def _enforce_mode_constraints(probes: QueryProbes, mode: str) -> QueryProbes:
    normalized = _normalize_mode(mode)
    if normalized == "constrained":
        return replace(probes, analogy=[], failure=[], construction=[])
    return probes


def _fallback_outcome(
    *,
    model: str,
    probes: QueryProbes,
    reason: str,
) -> ExpansionOutcome:
    return ExpansionOutcome(
        enabled=False,
        model=model,
        probes=probes,
        tokens_input=0,
        tokens_output=0,
        cache_hit_input=0,
        fallback_reason=reason,
    )


def _normalize_mode(mode: str) -> str:
    normalized = str(mode or "").strip().lower()
    if normalized == "free":
        return "explore"
    if normalized in {"start", "explore", "constrained"}:
        return normalized
    return "explore"


def _rubric_for_mode(mode: str) -> str:
    normalized = _normalize_mode(mode)
    if normalized == "start":
        mode_line = (
            "mode=start: produce 3-5 probes per class; analogy, failure, "
            "and construction probes should be diverse and mechanism-level."
        )
    elif normalized == "constrained":
        mode_line = (
            "mode=constrained: only direct and mechanism probes are allowed; "
            "avoid analogy, failure, and construction expansion."
        )
    else:
        mode_line = (
            "mode=explore: produce 2-3 probes per class; balance direct "
            "topic matches with transferable mechanisms."
        )
    return "\n".join(
        [
            "You are a quant research retrieval assistant.",
            "Expand the user's fuzzy idea into short retrieval probes.",
            "direct probes keep the original topic words.",
            "mechanism probes name causal forces, behavioral mechanisms, or market microstructure.",
            "analogy probes name different domains with transferable mechanisms.",
            "failure probes name ways the idea breaks or reverses.",
            "construction probes name how a factor/model might be built.",
            mode_line,
            "Return only the listed arrays through the tool call.",
        ]
    )


def _json_text_system(rubric: str) -> str:
    return "\n".join(
        [
            rubric,
            "The API gateway may not support tool calls.",
            "Return raw valid JSON only. Do not wrap it in markdown fences.",
            (
                'Return exactly {"direct":[],"mechanism":[],"analogy":[],'
                '"failure":[],"construction":[]}.'
            ),
        ]
    )


def _submit_query_probes_tool_schema() -> dict[str, Any]:
    array_schema = {
        "type": "array",
        "items": {"type": "string", "maxLength": 80},
    }
    return {
        "name": "submit_query_probes",
        "description": "Submit typed retrieval probes for the idea.",
        "input_schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "direct": array_schema,
                "mechanism": array_schema,
                "analogy": array_schema,
                "failure": array_schema,
                "construction": array_schema,
            },
            "required": [
                "direct",
                "mechanism",
                "analogy",
                "failure",
                "construction",
            ],
        },
    }


def _extract_tool_input(response: object) -> dict[str, Any]:
    content = _read_attr(response, "content", [])
    if not isinstance(content, list):
        raise ValueError("response content is not a list")
    for block in content:
        block_type = _read_attr(block, "type", "")
        block_name = _read_attr(block, "name", "")
        if block_type == "tool_use" and block_name == "submit_query_probes":
            raw_input = _read_attr(block, "input", {})
            if isinstance(raw_input, dict):
                return raw_input
            raise ValueError("tool input is not a dict")
    raise ValueError("submit_query_probes tool call not found")


def _parse_probes(tool_input: dict[str, Any]) -> QueryProbes:
    return QueryProbes(
        direct=_clean_probe_list(tool_input.get("direct")),
        mechanism=_clean_probe_list(tool_input.get("mechanism")),
        analogy=_clean_probe_list(tool_input.get("analogy")),
        failure=_clean_probe_list(tool_input.get("failure")),
        construction=_clean_probe_list(tool_input.get("construction")),
    )


def _clean_probe_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    probes: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        probes.append(text[:80])
    return probes


def _enforce_probe_budget(probes: QueryProbes, mode: str) -> QueryProbes:
    normalized = _normalize_mode(mode)
    if normalized == "start":
        cap = 5
    elif normalized == "constrained":
        cap = 3
    else:
        cap = 3
    return QueryProbes(
        direct=probes.direct[:cap],
        mechanism=probes.mechanism[:cap],
        analogy=probes.analogy[:cap],
        failure=probes.failure[:cap],
        construction=probes.construction[:cap],
    )


def _read_attr(obj: object, name: str, default: object) -> object:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _usage_int(usage: object, name: str) -> int:
    value = _read_attr(usage, name, 0)
    if not isinstance(value, int | float | str):
        return 0
    try:
        return int(value)
    except ValueError:
        return 0
