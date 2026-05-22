"""Small, dependency-free helpers shared across ``web_unified`` submodules."""

from __future__ import annotations

import math
import warnings
from pathlib import Path

from alpha_lab.exceptions import AlphaLabConfigError


class WebUnifiedConfigLoadWarning(UserWarning):
    """Emitted when the web frontend cannot load a stored config artifact.

    The unified web service is a read-mostly UI on top of artifacts written by
    the research pipelines. When a single broken file would otherwise vanish
    from the UI silently — case spec, project config, run record on restore —
    we keep the existing skip-and-continue semantics but surface a warning so
    the operator knows which file is corrupt.
    """


def _warn_web_config_load(*, source: object, action: str, exc: BaseException) -> None:
    """Emit a uniform ``WebUnifiedConfigLoadWarning`` for skip-on-load sites.

    ``source`` is rendered with ``str(...)`` so callers can pass either a
    ``Path`` or a stringified identifier without bespoke formatting.
    """

    label = str(source) if not isinstance(source, Path) else str(source)
    warnings.warn(
        (
            f"{action} skipped for {label}: "
            f"{type(exc).__name__}: {exc}"
        ),
        WebUnifiedConfigLoadWarning,
        stacklevel=3,
    )


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


__all__ = [
    "WebUnifiedConfigLoadWarning",
    "_coerce_finite_or_text",
    "_safe_slug",
    "_warn_web_config_load",
]
