"""Shared JSON normalization helper for real-case artifact writers.

Three artifact packages — ``real_cases.single_factor.artifacts``,
``real_cases.composite.artifacts`` and ``real_cases.model_factor.artifacts``
— used to carry byte-identical copies of ``_to_jsonable``. They are
consolidated here so a change to the JSON-normalization contract only needs
one edit and one set of tests.

The behavior is preserved verbatim from the prior implementations:

* ``Mapping`` -> dict with stringified keys, values recursively normalized.
* ``list`` / ``tuple`` -> list with recursively normalized elements.
* ``Path`` -> ``str(path)``.
* ``pd.Timestamp`` -> ``timestamp.isoformat()``.
* ``float`` -> ``value`` if ``math.isfinite(value)`` else ``None``.
* anything else -> returned unchanged (callers depend on this for
  JSON-native scalars like ``int``, ``bool``, ``str``, and ``None``).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path

import pandas as pd


def to_jsonable(value: object) -> object:
    """Recursively normalize ``value`` into JSON-serializable form.

    See module docstring for the exact mapping. This helper preserves the
    semantics of the previous per-package ``_to_jsonable`` copies including
    the conversion of non-finite floats (NaN, +/-inf) to ``None``.
    """
    if isinstance(value, Mapping):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value
