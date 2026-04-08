from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_lab.exceptions import AlphaLabConfigError, AlphaLabDataError


def time_split(
    dates: pd.Series | np.ndarray,
    train_end: str | pd.Timestamp,
    test_start: str | pd.Timestamp,
    val_start: str | pd.Timestamp | None = None,
) -> dict[str, np.ndarray]:
    """Split dates into train / (optional) validation / test boolean masks.

    All splits are strictly time-ordered with no overlap.  The function
    returns boolean masks rather than modified data so the caller controls
    how the masks are applied.

    Parameters
    ----------
    dates:
        1-D array-like of dates.  Need not be sorted.
    train_end:
        Last date (inclusive) of the training set.
    test_start:
        First date (inclusive) of the test set.  Must be strictly after
        ``train_end``.
    val_start:
        First date (inclusive) of the validation set.  Must satisfy
        ``train_end < val_start <= test_start``.  Dates in
        ``[val_start, test_start)`` are validation; dates in
        ``(train_end, val_start)`` are in neither split (gap).
        If ``None`` no validation mask is returned.

    Returns
    -------
    dict
        Maps ``'train'`` and ``'test'`` (and ``'val'`` when requested) to
        boolean numpy arrays of shape ``(len(dates),)``.  The masks are
        mutually exclusive: at most one mask is True at any index position.
    """
    # pd.to_datetime(dates) returns a Series (if dates is Series) or
    # DatetimeIndex (if array-like) — both support boolean comparison and
    # .to_numpy().  Avoid np.asarray() here; it produces a raw ndarray that
    # loses the .to_numpy() method in pandas 2.x.
    dates_ts = pd.to_datetime(dates)
    if pd.isna(dates_ts).any():
        raise AlphaLabDataError("dates contains NaT values; all dates must be valid timestamps")
    train_end_ts = pd.Timestamp(train_end)
    test_start_ts = pd.Timestamp(test_start)

    if train_end_ts >= test_start_ts:
        raise AlphaLabConfigError(
            f"train_end ({train_end_ts.date()}) must be strictly before "
            f"test_start ({test_start_ts.date()})"
        )

    train_mask: np.ndarray = (dates_ts <= train_end_ts).to_numpy(dtype=bool)
    test_mask: np.ndarray = (dates_ts >= test_start_ts).to_numpy(dtype=bool)

    if val_start is not None:
        val_start_ts = pd.Timestamp(val_start)
        if val_start_ts <= train_end_ts:
            raise AlphaLabConfigError(
                f"val_start ({val_start_ts.date()}) must be strictly after "
                f"train_end ({train_end_ts.date()})"
            )
        if val_start_ts > test_start_ts:
            raise AlphaLabConfigError(
                f"val_start ({val_start_ts.date()}) must be at or before "
                f"test_start ({test_start_ts.date()})"
            )
        val_mask: np.ndarray = (
            (dates_ts >= val_start_ts) & (dates_ts < test_start_ts)
        ).to_numpy(dtype=bool)
        return {"train": train_mask, "val": val_mask, "test": test_mask}

    return {"train": train_mask, "test": test_mask}


def walk_forward_split(
    dates: pd.Series | np.ndarray,
    train_size: int,
    test_size: int,
    step: int,
    val_size: int = 0,
) -> list[dict[str, np.ndarray]]:
    """Generate rolling walk-forward train / (optional) val / test masks.

    Each fold is strictly time-ordered: the test window immediately follows
    the training window with no overlap.  Folds advance by ``step`` rows
    between iterations.

    Parameters
    ----------
    dates:
        1-D array of dates in chronological order.  **Must be sorted.**
    train_size:
        Total number of rows in the training window, including the optional
        validation tail.
    test_size:
        Number of rows in the test window.
    step:
        Number of rows to advance the window start between folds.
    val_size:
        Number of rows at the *end* of the training window to reserve for
        validation.  These rows are excluded from the ``'train'`` mask and
        placed in the ``'val'`` mask.  0 (default) means no validation.

    Returns
    -------
    list of dict
        Each element maps ``'train'``, ``'test'``, and optionally ``'val'``
        to boolean numpy arrays of shape ``(len(dates),)``.
        Folds where the test window would exceed the available data are not
        generated.
    """
    if train_size <= 0 or test_size <= 0 or step <= 0:
        raise AlphaLabConfigError("train_size, test_size, and step must be positive integers")
    if val_size < 0:
        raise AlphaLabConfigError("val_size must be non-negative")
    if val_size >= train_size:
        raise AlphaLabConfigError("val_size must be less than train_size")

    dates_arr = np.asarray(pd.to_datetime(np.asarray(dates)))
    n = len(dates_arr)

    if pd.isnull(dates_arr).any():
        raise AlphaLabDataError("dates contains NaT values; all dates must be valid timestamps")

    if n > 1 and not np.all(dates_arr[:-1] <= dates_arr[1:]):
        raise AlphaLabDataError("dates must be sorted in chronological order")

    # Repeated dates indicate panel data (many assets per date).  Row-based
    # slicing would split a shared timestamp across folds, creating leakage.
    # Pass unique dates instead: e.g. df["date"].drop_duplicates().sort_values()
    if len(dates_arr) != len(np.unique(dates_arr)):
        raise AlphaLabDataError(
            "dates contains repeated values. walk_forward_split() operates on "
            "unique dates. For panel data pass the unique sorted date axis: "
            "e.g. df['date'].drop_duplicates().sort_values().reset_index(drop=True)"
        )

    splits: list[dict[str, np.ndarray]] = []
    train_start = 0

    while True:
        train_end = train_start + train_size  # exclusive index
        test_end = train_end + test_size  # exclusive index

        if test_end > n:
            break

        train_mask = np.zeros(n, dtype=bool)
        train_mask[train_start : train_end - val_size] = True

        test_mask = np.zeros(n, dtype=bool)
        test_mask[train_end:test_end] = True

        fold: dict[str, np.ndarray] = {"train": train_mask, "test": test_mask}

        if val_size > 0:
            val_mask = np.zeros(n, dtype=bool)
            val_mask[train_end - val_size : train_end] = True
            fold["val"] = val_mask

        splits.append(fold)
        train_start += step

    return splits
