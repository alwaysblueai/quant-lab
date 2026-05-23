"""Smoke coverage for :mod:`alpha_lab.frame_utils`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_lab.exceptions import AlphaLabDataError
from alpha_lab.frame_utils import readonly_shallow_copy, require_columns


def _sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "a": np.arange(5, dtype=np.float64),
            "b": np.arange(5, 10, dtype=np.float64),
            "c": np.arange(10, 15, dtype=np.float64),
        }
    )


def test_readonly_shallow_copy_preserves_data() -> None:
    frame = _sample_frame()
    view = readonly_shallow_copy(frame)

    assert list(view.columns) == ["a", "b", "c"]
    np.testing.assert_array_equal(view["a"].to_numpy(), frame["a"].to_numpy())


def test_readonly_shallow_copy_selects_columns() -> None:
    frame = _sample_frame()
    view = readonly_shallow_copy(frame, columns=["a", "c"])

    assert list(view.columns) == ["a", "c"]
    assert "b" not in view.columns


def test_readonly_shallow_copy_blocks_inplace_writes() -> None:
    frame = _sample_frame()
    view = readonly_shallow_copy(frame)

    underlying = view["a"].to_numpy(copy=False)
    with pytest.raises(ValueError):
        underlying[0] = 9999.0


def test_require_columns_passes_when_all_present() -> None:
    frame = _sample_frame()
    # No raise expected.
    require_columns(frame, ("a", "b"), "frame")


def test_require_columns_raises_alpha_lab_data_error_on_missing() -> None:
    frame = _sample_frame()
    with pytest.raises(AlphaLabDataError) as excinfo:
        require_columns(frame, ("a", "missing_x", "missing_y"), "frame")
    # Preserve the historical message shape (label prefix + sorted missing).
    assert "frame missing required columns" in str(excinfo.value)
    assert "['missing_x', 'missing_y']" in str(excinfo.value)


def test_require_columns_accepts_arbitrary_iterables() -> None:
    frame = _sample_frame()
    # list, set, generator must all work — call sites pass tuples in production.
    require_columns(frame, ["a", "b"], "frame")
    require_columns(frame, {"a", "c"}, "frame")
    require_columns(frame, (col for col in ("a", "b")), "frame")
