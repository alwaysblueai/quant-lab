from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from alpha_lab.research_bridge._llm_usage import read_attr, usage_int

DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_MAX_CANDIDATES = 30
ANTHROPIC_BASE_URL_ENV = "ANTHROPIC_BASE_URL"
LOG = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RerankCandidate:
    name: str
    summary: str
    domain: str
    mechanism: str
    factor_family: str
    path: str = ""


@dataclass(frozen=True, slots=True)
class RerankOutcome:
    enabled: bool
    model: str
    scores: dict[str, float]
    reasons: dict[str, str]
    tokens_input: int
    tokens_output: int
    cache_hit_input: int
    dropped_invalid_names: list[str]
    fallback_reason: str | None


@dataclass(frozen=True, slots=True)
class CategorizedCandidate:
    path: str
    name: str
    category: str
    relevance: float
    transferable_to_idea: str
    key_warning: str


@dataclass(frozen=True, slots=True)
class CategorizeOutcome:
    enabled: bool
    model: str
    categorized: list[CategorizedCandidate]
    insight_brief: list[str]
    dropped_invalid_paths: list[str]
    tokens_input: int
    tokens_output: int
    cache_hit_input: int
    fallback_reason: str | None

    @property
    def by_path(self) -> dict[str, CategorizedCandidate]:
        return {item.path: item for item in self.categorized}

    @property
    def scores(self) -> dict[str, float]:
        return {item.path: item.relevance for item in self.categorized}


def anthropic_client_kwargs(api_key: str) -> dict[str, Any]:
    """Return shared Anthropic client kwargs, including an optional base URL.

    Return type is ``dict[str, Any]`` rather than ``dict[str, str]`` because the
    upstream ``anthropic.Anthropic`` constructor has many strongly-typed
    parameters; mypy validates ``**kwargs`` unpacking per-key, so a narrow
    ``str`` value type triggers a long cascade of false positives.
    """
    kwargs: dict[str, Any] = {"api_key": api_key}
    base_url = str(os.environ.get(ANTHROPIC_BASE_URL_ENV) or "").strip()
    if base_url:
        kwargs["base_url"] = base_url
    return kwargs


def extract_json_object_from_response(response: object) -> dict[str, Any]:
    """Extract a JSON object from a text response for proxy compatibility."""
    content = read_attr(response, "content", [])
    text_parts: list[str] = []
    if isinstance(content, list):
        for block in content:
            if read_attr(block, "type", "") == "text":
                text_parts.append(str(read_attr(block, "text", "") or ""))
    elif isinstance(content, str):
        text_parts.append(content)
    text = "\n".join(part for part in text_parts if part).strip()
    if not text:
        raise ValueError("response text is empty")
    return _parse_json_object_text(text)


def rerank_candidates(
    *,
    idea: str,
    mode: str,
    candidates: list[RerankCandidate],
    model: str | None = None,
    api_key_env: str = "ANTHROPIC_API_KEY",
) -> RerankOutcome:
    """Score card candidates for transferable idea relevance.

    The function is fail-closed for the caller: missing credentials, SDK
    errors, or malformed responses all return a disabled outcome with empty
    scores, leaving deterministic ranking unchanged.
    """
    model_name = str(model or DEFAULT_MODEL)
    if not candidates:
        return _fallback_outcome(model=model_name, reason="no_candidates")

    api_key = os.environ.get(api_key_env)
    if not api_key:
        return _fallback_outcome(model=model_name, reason="no_api_key")

    try:
        import anthropic  # type: ignore[import-not-found]

        client: Any = anthropic.Anthropic(**anthropic_client_kwargs(api_key))
        request_payload = {
            "idea": str(idea or "").strip(),
            "candidates": [
                _candidate_to_payload(candidate)
                for candidate in candidates[:DEFAULT_MAX_CANDIDATES]
            ],
        }
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
                max_tokens=2048,
                temperature=0,
                tools=[_submit_relevance_tool_schema()],
                tool_choice={"type": "tool", "name": "submit_relevance"},
            )
            tool_input = _extract_tool_input(response)
        except Exception as tool_exc:
            LOG.info("LLM rerank tool call failed; trying JSON text mode: %s", tool_exc)
            response = client.messages.create(
                model=model_name,
                system=_json_text_system(
                    _rubric_for_mode(mode),
                    'Return only {"items":[{"name":"...","relevance":0.0,"reason":"..."}]}.',
                ),
                messages=[
                    {
                        "role": "user",
                        "content": json.dumps(request_payload, ensure_ascii=False),
                    }
                ],
                max_tokens=2048,
                temperature=0,
            )
            tool_input = extract_json_object_from_response(response)
        scores, reasons, dropped = _parse_scores(tool_input, candidates)
        usage = read_attr(response, "usage", {})
        input_tokens = usage_int(usage, "input_tokens")
        cache_creation = usage_int(usage, "cache_creation_input_tokens")
        cache_read = usage_int(usage, "cache_read_input_tokens")
        output_tokens = usage_int(usage, "output_tokens")
        return RerankOutcome(
            enabled=True,
            model=model_name,
            scores=scores,
            reasons=reasons,
            tokens_input=input_tokens + cache_creation + cache_read,
            tokens_output=output_tokens,
            cache_hit_input=cache_read,
            dropped_invalid_names=dropped,
            fallback_reason=None,
        )
    except Exception as exc:
        reason = f"api_error: {type(exc).__name__}: {exc}"
        LOG.warning("LLM rerank failed; falling back: %s", reason)
        return _fallback_outcome(model=model_name, reason=reason)


def categorize_and_compress(
    *,
    idea: str,
    mode: str,
    candidates: list[RerankCandidate],
    fingerprints: Mapping[str, object],
    provenance: Mapping[str, list[str]],
    model: str | None = None,
    api_key_env: str = "ANTHROPIC_API_KEY",
) -> CategorizeOutcome:
    """Categorize candidates and emit cross-card synthesis for v2 recall."""
    model_name = str(model or DEFAULT_MODEL)
    limited_candidates = candidates[:DEFAULT_MAX_CANDIDATES]
    if not limited_candidates:
        return _categorize_fallback_outcome(model=model_name, reason="no_candidates")

    api_key = os.environ.get(api_key_env)
    if not api_key:
        return _categorize_fallback_outcome(model=model_name, reason="no_api_key")

    try:
        import anthropic  # type: ignore[import-not-found]

        client: Any = anthropic.Anthropic(**anthropic_client_kwargs(api_key))
        request_payload = {
            "idea": str(idea or "").strip(),
            "candidates": [
                _categorize_candidate_payload(
                    candidate,
                    fingerprints=fingerprints,
                    provenance=provenance,
                )
                for candidate in limited_candidates
            ],
        }
        try:
            response = client.messages.create(
                model=model_name,
                system=[
                    {
                        "type": "text",
                        "text": _categorize_rubric_for_mode(mode),
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[
                    {
                        "role": "user",
                        "content": json.dumps(request_payload, ensure_ascii=False),
                    }
                ],
                max_tokens=2048,
                temperature=0,
                tools=[_submit_categorized_tool_schema()],
                tool_choice={"type": "tool", "name": "submit_categorized_relevance"},
            )
            tool_input = _extract_named_tool_input(
                response,
                tool_name="submit_categorized_relevance",
            )
        except Exception as tool_exc:
            LOG.info("LLM categorize tool call failed; trying JSON text mode: %s", tool_exc)
            response = client.messages.create(
                model=model_name,
                system=_json_text_system(
                    _categorize_rubric_for_mode(mode),
                    (
                        'Return only {"items":[{"path":"...","name":"...",'
                        '"category":"direct|transferable|risk|construction|unrelated",'
                        '"relevance":0.0,"transferable_to_idea":"...",'
                        '"key_warning":"..."}],"insight_brief":["..."]}.'
                    ),
                ),
                messages=[
                    {
                        "role": "user",
                        "content": json.dumps(request_payload, ensure_ascii=False),
                    }
                ],
                max_tokens=2048,
                temperature=0,
            )
            tool_input = extract_json_object_from_response(response)
        categorized, dropped = _parse_categorized(tool_input, limited_candidates)
        usage = read_attr(response, "usage", {})
        input_tokens = usage_int(usage, "input_tokens")
        cache_creation = usage_int(usage, "cache_creation_input_tokens")
        cache_read = usage_int(usage, "cache_read_input_tokens")
        output_tokens = usage_int(usage, "output_tokens")
        return CategorizeOutcome(
            enabled=True,
            model=model_name,
            categorized=categorized,
            insight_brief=_clean_short_text_list(tool_input.get("insight_brief"), limit=6),
            dropped_invalid_paths=dropped,
            tokens_input=input_tokens + cache_creation + cache_read,
            tokens_output=output_tokens,
            cache_hit_input=cache_read,
            fallback_reason=None,
        )
    except Exception as exc:
        reason = f"api_error: {type(exc).__name__}: {exc}"
        LOG.warning("LLM categorize/compress failed; falling back: %s", reason)
        return _categorize_fallback_outcome(model=model_name, reason=reason)


def _fallback_outcome(*, model: str, reason: str) -> RerankOutcome:
    return RerankOutcome(
        enabled=False,
        model=model,
        scores={},
        reasons={},
        tokens_input=0,
        tokens_output=0,
        cache_hit_input=0,
        dropped_invalid_names=[],
        fallback_reason=reason,
    )


def _json_text_system(rubric: str, schema_hint: str) -> str:
    return "\n".join(
        [
            rubric,
            "The API gateway may not support tool calls.",
            "Return raw valid JSON only. Do not wrap it in markdown fences.",
            schema_hint,
        ]
    )


def _parse_json_object_text(text: str) -> dict[str, Any]:
    candidate = text.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        candidate = "\n".join(lines).strip()
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start < 0 or end <= start:
            raise
        parsed = json.loads(candidate[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("text JSON response is not an object")
    return parsed


def _categorize_fallback_outcome(*, model: str, reason: str) -> CategorizeOutcome:
    return CategorizeOutcome(
        enabled=False,
        model=model,
        categorized=[],
        insight_brief=[],
        dropped_invalid_paths=[],
        tokens_input=0,
        tokens_output=0,
        cache_hit_input=0,
        fallback_reason=reason,
    )


def _rubric_for_mode(mode: str) -> str:
    normalized = str(mode or "").strip().lower()
    if normalized == "free":
        normalized = "explore"
    mode_guidance = {
        "start": "mode=start：鼓励跨领域迁移；不同领域但机制可迁移时可以给高分。",
        "explore": "mode=explore：平衡直接主题与机制迁移；避免只因字面重叠给高分。",
        "constrained": "mode=constrained：仅同主题或同 mechanism 高分；抑制松散类比。",
    }.get(normalized, "mode=explore：平衡直接主题与机制迁移；避免只因字面重叠给高分。")
    return "\n".join(
        [
            "你是 quant 研究助理。",
            "你看到的每张卡片都已经被字面召回选中。",
            "你的任务：判断它对当前 idea 的思想相关度，区间 [0,1]。",
            "1.0 = 直接同主题；0.7+ = 不同领域但机制可迁移；0.3- = 仅字面重叠、机制无关。",
            mode_guidance,
            "必须遵守：仅对 user message 列出的 name 打分；不得添加新 name。",
            "relevance 必须是 [0,1] 浮点；reason 用一句话，30 字以内。",
        ]
    )


def _candidate_to_payload(candidate: RerankCandidate) -> dict[str, str]:
    return {
        "path": candidate.path or candidate.name,
        "name": candidate.name,
        "summary": candidate.summary,
        "domain": candidate.domain,
        "mechanism": candidate.mechanism,
        "factor_family": candidate.factor_family,
    }


def _categorize_candidate_payload(
    candidate: RerankCandidate,
    *,
    fingerprints: Mapping[str, object],
    provenance: Mapping[str, list[str]],
) -> dict[str, object]:
    path = candidate.path or candidate.name
    return {
        **_candidate_to_payload(candidate),
        "path": path,
        "fingerprint": _fingerprint_payload(fingerprints.get(path)),
        "hit_by": list(provenance.get(path, [])),
    }


def _fingerprint_payload(fingerprint: object) -> dict[str, object]:
    if fingerprint is None:
        return {}
    return {
        "core_mechanism": list(getattr(fingerprint, "core_mechanism", []) or []),
        "transferable_principle": str(
            getattr(fingerprint, "transferable_principle", "") or ""
        ),
        "applicable_scenarios": list(
            getattr(fingerprint, "applicable_scenarios", []) or []
        ),
        "similar_problems": list(getattr(fingerprint, "similar_problems", []) or []),
        "failure_conditions": list(
            getattr(fingerprint, "failure_conditions", []) or []
        ),
    }


def _submit_relevance_tool_schema() -> dict[str, Any]:
    return {
        "name": "submit_relevance",
        "description": "Submit relevance scores for the listed card candidates.",
        "input_schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "name": {"type": "string"},
                            "relevance": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 1,
                            },
                            "reason": {"type": "string", "maxLength": 30},
                        },
                        "required": ["name", "relevance", "reason"],
                    },
                }
            },
            "required": ["items"],
        },
    }


def _submit_categorized_tool_schema() -> dict[str, Any]:
    return {
        "name": "submit_categorized_relevance",
        "description": "Categorize listed candidates and summarize cross-card insights.",
        "input_schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "path": {"type": "string"},
                            "name": {"type": "string"},
                            "category": {
                                "type": "string",
                                "enum": [
                                    "direct",
                                    "transferable",
                                    "risk",
                                    "construction",
                                    "unrelated",
                                ],
                            },
                            "relevance": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 1,
                            },
                            "transferable_to_idea": {
                                "type": "string",
                                "maxLength": 30,
                            },
                            "key_warning": {"type": "string", "maxLength": 30},
                        },
                        "required": [
                            "path",
                            "name",
                            "category",
                            "relevance",
                            "transferable_to_idea",
                            "key_warning",
                        ],
                    },
                },
                "insight_brief": {
                    "type": "array",
                    "items": {"type": "string", "maxLength": 120},
                    "minItems": 0,
                    "maxItems": 6,
                },
            },
            "required": ["items", "insight_brief"],
        },
    }


def _extract_tool_input(response: object) -> dict[str, Any]:
    return _extract_named_tool_input(response, tool_name="submit_relevance")


def _extract_named_tool_input(response: object, *, tool_name: str) -> dict[str, Any]:
    content = read_attr(response, "content", [])
    if not isinstance(content, list):
        raise ValueError("response content is not a list")
    for block in content:
        block_type = read_attr(block, "type", "")
        block_name = read_attr(block, "name", "")
        if block_type == "tool_use" and block_name == tool_name:
            raw_input = read_attr(block, "input", {})
            if isinstance(raw_input, dict):
                return raw_input
            raise ValueError("tool input is not a dict")
    raise ValueError(f"{tool_name} tool call not found")


def _parse_scores(
    tool_input: dict[str, Any],
    candidates: list[RerankCandidate],
) -> tuple[dict[str, float], dict[str, str], list[str]]:
    raw_items = tool_input.get("items", [])
    if not isinstance(raw_items, list):
        raise ValueError("tool input items is not a list")
    candidate_names = {
        candidate.name for candidate in candidates[:DEFAULT_MAX_CANDIDATES]
    }
    scores: dict[str, float] = {}
    reasons: dict[str, str] = {}
    dropped_invalid_names: list[str] = []
    for raw_item in raw_items:
        if not isinstance(raw_item, dict):
            continue
        name = str(raw_item.get("name") or "").strip()
        if name not in candidate_names:
            if name:
                dropped_invalid_names.append(name)
            continue
        relevance = _clamp_float(raw_item.get("relevance"))
        scores[name] = relevance
        reasons[name] = str(raw_item.get("reason") or "").strip()
    return scores, reasons, dropped_invalid_names


def _parse_categorized(
    tool_input: dict[str, Any],
    candidates: list[RerankCandidate],
) -> tuple[list[CategorizedCandidate], list[str]]:
    raw_items = tool_input.get("items", [])
    if not isinstance(raw_items, list):
        raise ValueError("tool input items is not a list")
    valid_by_path = {
        (candidate.path or candidate.name): candidate for candidate in candidates
    }
    categorized: list[CategorizedCandidate] = []
    dropped_invalid_paths: list[str] = []
    for raw_item in raw_items:
        if not isinstance(raw_item, dict):
            continue
        path = str(raw_item.get("path") or "").strip()
        if path not in valid_by_path:
            if path:
                dropped_invalid_paths.append(path)
            continue
        candidate = valid_by_path[path]
        category = str(raw_item.get("category") or "").strip().lower()
        if category not in {
            "direct",
            "transferable",
            "risk",
            "construction",
            "unrelated",
        }:
            category = "unrelated"
        categorized.append(
            CategorizedCandidate(
                path=path,
                name=str(raw_item.get("name") or candidate.name).strip()
                or candidate.name,
                category=category,
                relevance=_clamp_float(raw_item.get("relevance")),
                transferable_to_idea=str(
                    raw_item.get("transferable_to_idea") or ""
                ).strip()[:30],
                key_warning=str(raw_item.get("key_warning") or "").strip()[:30],
            )
        )
    return categorized, dropped_invalid_paths


def _clean_short_text_list(value: object, *, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    rows: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        rows.append(text[:120])
        if len(rows) >= limit:
            break
    return rows


def _categorize_rubric_for_mode(mode: str) -> str:
    normalized = str(mode or "").strip().lower()
    if normalized == "free":
        normalized = "explore"
    if normalized == "constrained":
        mode_line = (
            "mode=constrained: be strict; direct and transferable are the useful "
            "categories, and loose analogies should be unrelated."
        )
    elif normalized == "start":
        mode_line = (
            "mode=start: allow cross-domain transfer, risk cards, and construction "
            "cards when they can improve the initial prompt."
        )
    else:
        mode_line = (
            "mode=explore: balance direct relevance with transferable mechanisms."
        )
    return "\n".join(
        [
            "You are a quant research assistant.",
            "Candidates were retrieved by literal and mechanism probes.",
            "For each candidate, output category: direct, transferable, risk, "
            "construction, or unrelated.",
            "Use relevance in [0,1], transferable_to_idea within 30 Chinese "
            "characters, and key_warning within 30 Chinese characters.",
            "Also output 4-6 insight_brief items that synthesize across "
            "multiple cards; avoid single-card restatement.",
            "Strictly use the input candidate path values; never add a new path.",
            mode_line,
        ]
    )


def _clamp_float(value: object) -> float:
    if not isinstance(value, int | float | str):
        return 0.0
    try:
        numeric = float(value)
    except ValueError:
        return 0.0
    if numeric < 0.0:
        return 0.0
    if numeric > 1.0:
        return 1.0
    return numeric


