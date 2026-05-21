"""Shared helpers for parsing Claude/Anthropic LLM response payloads.

``read_attr`` + ``usage_int`` had byte-identical copies in
``mechanism_index.py``, ``query_expansion.py``, and ``llm_rerank.py``.
This module is the single source of truth.
"""

from __future__ import annotations


def read_attr(obj: object, name: str, default: object) -> object:
    """Look up ``name`` on either a mapping or an attribute-style object."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def usage_int(usage: object, name: str) -> int:
    """Extract a token-count field from a Claude usage object as a clamped int."""
    value = read_attr(usage, name, 0)
    if not isinstance(value, int | float | str):
        return 0
    try:
        return int(value)
    except ValueError:
        return 0
