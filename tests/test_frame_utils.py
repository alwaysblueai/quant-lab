"""Smoke coverage for :mod:`alpha_lab.frame_utils`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_lab.frame_utils import readonly_shallow_copy


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
