from __future__ import annotations

from collections.abc import Iterable, Sequence

import pandas as pd

from alpha_lab.exceptions import AlphaLabDataError


def require_columns(
    frame: pd.DataFrame,
    cols: Iterable[str],
    name: str,
) -> None:
    """Raise ``AlphaLabDataError`` if any of ``cols`` are missing from ``frame``.

    Centralizes the previously-duplicated ``_require_columns`` helpers used
    across factor / data-quality / universe / bucket-builder modules. The
    error message is preserved verbatim
    (``"{name} missing required columns: {sorted(missing)}"``) so log and test
    expectations downstream remain stable.
    """
    missing = set(cols) - set(frame.columns)
    if missing:
        raise AlphaLabDataError(f"{name} missing required columns: {sorted(missing)}")


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
