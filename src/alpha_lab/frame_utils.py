from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


def readonly_shallow_copy(
    frame: pd.DataFrame,
    *,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Return a shallow DataFrame view with array-level write protection.

    This is an internal performance helper for hot read-only paths. It avoids
    full-frame copies while still catching accidental in-place writes to shared
    numpy-backed column arrays.
    """
    if columns is None:
        borrowed = frame.copy(deep=False)
    else:
        borrowed = frame.loc[:, list(columns)].copy(deep=False)
    for column in borrowed.columns:
        try:
            borrowed[column].to_numpy(copy=False).setflags(write=False)
        except (AttributeError, TypeError, ValueError):
            continue
    return borrowed
