"""Small, dependency-free helpers shared across ``web_unified`` submodules."""

from __future__ import annotations

import math

from alpha_lab.exceptions import AlphaLabConfigError


def _safe_slug(value: str) -> str:
    """Sanitize a string into a filesystem-safe slug.

    Raises ``AlphaLabConfigError`` if the input is empty or maps to an empty
    slug after stripping disallowed characters.
    """

    raw = str(value).strip()
    if not raw:
        raise AlphaLabConfigError("slug must be non-empty")
    normalized = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "-" for ch in raw)
    normalized = normalized.strip("._-")
    if not normalized:
        raise AlphaLabConfigError(f"slug is invalid: {value!r}")
    return normalized


def _coerce_finite_or_text(value: object) -> str | None:
    """Coerce a JSON-like value to a non-empty trimmed string.

    Returns ``None`` for empty strings, non-finite floats, and anything that
    isn't ``str | int | float``. Used pervasively when sifting through
    untyped manifest payloads in the web frontend.
    """

    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, int | float) and math.isfinite(float(value)):
        text = str(value).strip()
        return text or None
    return None


__all__ = ["_coerce_finite_or_text", "_safe_slug"]
