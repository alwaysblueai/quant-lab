from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import cast


def as_object_dict(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return cast(dict[str, object], value)
    if isinstance(value, Mapping):
        return cast(dict[str, object], dict(value))
    return {}


def as_object_list(value: object) -> list[object]:
    if isinstance(value, list):
        return cast(list[object], value)
    return []


def safe_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def to_finite_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        out = float(value)
        if math.isfinite(out):
            return out
    return None


def parse_text_list(value: object, *, split_semicolon: bool = True) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        out: list[str] = []
        for item in value:
            token = safe_text(item)
            if token is not None:
                out.append(token)
        return out

    text = safe_text(value)
    if text is None:
        return []
    if split_semicolon and ";" in text:
        return [token.strip() for token in text.split(";") if token.strip()]
    return [text]


def format_text_list(
    value: object,
    *,
    empty: str,
    separator: str = ", ",
    split_semicolon: bool = True,
) -> str:
    items = parse_text_list(value, split_semicolon=split_semicolon)
    if not items:
        return empty
    return separator.join(items)


def format_ci(
    lower: object,
    upper: object,
    *,
    precision: int = 6,
    na: str = "N/A",
) -> str:
    left = to_finite_float(lower)
    right = to_finite_float(upper)
    if left is None or right is None:
        return na
    return f"[{left:.{precision}f}, {right:.{precision}f}]"


def portfolio_validation_note(
    status: object,
    recommendation: object,
    *,
    na: str = "N/A",
) -> str:
    status_text = safe_text(status) or na
    recommendation_text = safe_text(recommendation) or na
    if status_text == na and recommendation_text == na:
        return na
    return f"{status_text} ({recommendation_text})"


def portfolio_validation_benchmark_note(
    status: object,
    assessment: object,
    excess_return: object,
    tracking_error: object,
    *,
    format_metric: Callable[[object], str],
    na: str = "N/A",
) -> str:
    status_text = safe_text(status) or na
    assessment_text = safe_text(assessment) or na
    if status_text == na and assessment_text == na:
        return na
    excess = format_metric(excess_return)
    tracking = format_metric(tracking_error)
    return f"{status_text} ({assessment_text}), excess={excess}, tracking_error={tracking}"
