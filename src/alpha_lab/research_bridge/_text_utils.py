"""Shared text-tokenization helpers for research_bridge.

``idea_keywords`` had byte-identical copies in ``loaders.py`` and
``model_idea.py``. This module is the single source of truth.
"""

from __future__ import annotations

import re


def idea_keywords(text: str) -> set[str]:
    """Return a normalized keyword set for a free-form idea string.

    Splits on alphanumeric runs and CJK runs, lowercases, drops one-char
    tokens, and emits 2-gram windows for CJK runs of length >= 4.
    """
    keywords: set[str] = set()
    for token in re.findall(r"[a-zA-Z0-9_]+|[一-鿿]+", text):
        normalized = token.lower().strip()
        if len(normalized) <= 1:
            continue
        keywords.add(normalized)
        if re.fullmatch(r"[一-鿿]+", normalized) and len(normalized) >= 4:
            for idx in range(len(normalized) - 1):
                keywords.add(normalized[idx : idx + 2])
    return keywords
