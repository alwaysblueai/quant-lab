"""Helpers for marking and reusing sorted panel order.

Use this only where row order is not part of the public or implicit contract.
For example, ``neutralize_signal`` intentionally preserves caller row order, so
it should not be "optimized" by forcing a sorted panel at entry.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

SORTED_ATTR_KEY = "_alpha_lab_sorted"


def mark_sorted(df: pd.DataFrame, by: Iterable[str]) -> pd.DataFrame:
    df.attrs[SORTED_ATTR_KEY] = tuple(str(column) for column in by)
    return df


def ensure_sorted(
    df: pd.DataFrame,
    by: Iterable[str] = ("asset", "date"),
    *,
    reset_index: bool = True,
) -> pd.DataFrame:
    columns = tuple(str(column) for column in by)
    if df.attrs.get(SORTED_ATTR_KEY) == columns:
        return df
    out = df.sort_values(list(columns), kind="mergesort")
    if reset_index:
        out = out.reset_index(drop=True)
    return mark_sorted(out, columns)
