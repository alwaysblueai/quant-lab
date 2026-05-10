from __future__ import annotations

import weakref
from collections.abc import Mapping
from typing import Any, cast

import numpy as np


def _indices_as_contiguous_slice(indices: np.ndarray) -> slice | None:
    if len(indices) == 0:
        return slice(0, 0)
    first = int(indices[0])
    last = int(indices[-1])
    if last < first:
        return None
    if last - first + 1 != len(indices):
        return None
    expected = np.arange(first, last + 1, dtype=np.intp)
    if not np.array_equal(indices, expected):
        return None
    return slice(first, last + 1)


def _mapping_bool(mapping: Mapping[str, object] | None, key: str, default: bool) -> bool:
    if not isinstance(mapping, Mapping) or key not in mapping:
        return default
    value = mapping.get(key)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "on"}:
            return True
        if normalized in {"false", "0", "no", "n", "off"}:
            return False
    return bool(value)


def _mapping_int(mapping: Mapping[str, object] | None, key: str, default: int) -> int:
    if not isinstance(mapping, Mapping) or key not in mapping:
        return int(default)
    value = mapping.get(key)
    if isinstance(value, bool):
        return int(default)
    try:
        return int(cast(Any, value))
    except (TypeError, ValueError):
        return int(default)


def _mapping_text(mapping: Mapping[str, object] | None, key: str, default: str) -> str:
    if not isinstance(mapping, Mapping) or key not in mapping:
        return default
    return str(mapping.get(key) or default).strip() or default


def _weakref_or_none(value: object) -> weakref.ReferenceType[object] | None:
    try:
        return weakref.ref(value)
    except TypeError:
        return None


def _sample_float_or_none(value: object) -> float | None:
    if value is None:
        return None
    try:
        number = float(cast(Any, value))
    except (TypeError, ValueError):
        return None
    return _finite_or_none(number)


def _finite_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    return float(value) if np.isfinite(value) else None


def _object_to_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(cast(Any, value))
    except (TypeError, ValueError):
        return None


def _object_to_int(value: object, *, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(cast(Any, value))
    except (TypeError, ValueError):
        return default
