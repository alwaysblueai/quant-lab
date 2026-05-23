"""Coverage for ``alpha_lab.real_cases._artifact_json.to_jsonable``.

Pins the JSON-normalization contract previously duplicated across three
artifact packages so the consolidated helper preserves byte-for-byte
behavior.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from alpha_lab.real_cases._artifact_json import to_jsonable


def test_to_jsonable_passes_native_scalars_through() -> None:
    assert to_jsonable(1) == 1
    assert to_jsonable("text") == "text"
    assert to_jsonable(True) is True
    assert to_jsonable(None) is None


def test_to_jsonable_normalizes_path_to_str() -> None:
    p = Path("/tmp/example.json")
    assert to_jsonable(p) == str(p)


def test_to_jsonable_normalizes_pandas_timestamp_to_isoformat() -> None:
    ts = pd.Timestamp("2026-01-02T03:04:05Z")
    assert to_jsonable(ts) == ts.isoformat()


def test_to_jsonable_converts_non_finite_float_to_none() -> None:
    assert to_jsonable(float("nan")) is None
    assert to_jsonable(float("inf")) is None
    assert to_jsonable(float("-inf")) is None
    assert to_jsonable(1.5) == 1.5
    assert to_jsonable(0.0) == 0.0


def test_to_jsonable_recurses_into_mappings() -> None:
    out = to_jsonable({"a": 1, "b": float("nan"), 3: "key-coerced"})
    assert out == {"a": 1, "b": None, "3": "key-coerced"}


def test_to_jsonable_recurses_into_lists_and_tuples() -> None:
    out = to_jsonable([1, float("nan"), (2, float("inf"))])
    # tuples collapse to lists in the canonical form.
    assert out == [1, None, [2, None]]


def test_to_jsonable_deeply_nested_structure() -> None:
    payload = {
        "name": "abc",
        "rows": [
            {"date": pd.Timestamp("2026-01-02"), "value": float("nan")},
            {"date": pd.Timestamp("2026-01-03"), "value": 1.0},
        ],
        "paths": (Path("/a"), Path("/b")),
        "nested": {"flag": True, "score": float("-inf")},
    }
    out = to_jsonable(payload)
    assert out == {
        "name": "abc",
        "rows": [
            {"date": "2026-01-02T00:00:00", "value": None},
            {"date": "2026-01-03T00:00:00", "value": 1.0},
        ],
        "paths": ["/a", "/b"],
        "nested": {"flag": True, "score": None},
    }


def test_to_jsonable_matches_legacy_finite_or_none_semantics() -> None:
    # The historical ``_finite_or_none`` helper was a one-liner using
    # ``math.isfinite``. The consolidated helper must produce the same result
    # for every float input including NaN / +inf / -inf / subnormals.
    for x in [
        0.0,
        -0.0,
        1.0,
        -1.0,
        math.pi,
        1e-300,
        1e300,
        float("nan"),
        float("inf"),
        float("-inf"),
    ]:
        expected = x if math.isfinite(x) else None
        assert to_jsonable(x) == expected or (
            expected is None and to_jsonable(x) is None
        )
