from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_lab.validation.purged_kfold import purged_kfold_split


def _dates(n: int) -> np.ndarray:
    return np.asarray(pd.date_range("2024-01-01", periods=n, freq="B"))


def test_purged_kfold_returns_non_overlapping_boolean_masks() -> None:
    splits = purged_kfold_split(
        _dates(20),
        n_splits=4,
        label_horizon=2,
        embargo_pct=0.10,
    )
    assert len(splits) == 4
    for fold in splits:
        assert fold["train"].dtype == bool
        assert fold["test"].dtype == bool
        assert not np.any(fold["train"] & fold["test"])


def test_purged_kfold_applies_purge_and_embargo() -> None:
    # n=10, n_splits=2 => fold1 test indices [5..9].
    # With label_horizon=2, train indices 3 and 4 must be purged.
    splits = purged_kfold_split(
        _dates(10),
        n_splits=2,
        label_horizon=2,
        embargo_pct=0.0,
    )
    fold1 = splits[1]
    assert not fold1["train"][3]
    assert not fold1["train"][4]
    assert fold1["train"][2]


def test_purged_kfold_rejects_repeated_dates() -> None:
    repeated = np.asarray(pd.date_range("2024-01-01", periods=5, freq="B").repeat(2))
    with pytest.raises(ValueError, match="unique"):
        purged_kfold_split(repeated, n_splits=3)


def test_purged_kfold_rejects_unsorted_dates() -> None:
    unsorted = np.asarray(
        [
            pd.Timestamp("2024-01-03"),
            pd.Timestamp("2024-01-01"),
            pd.Timestamp("2024-01-02"),
        ]
    )
    with pytest.raises(ValueError, match="sorted"):
        purged_kfold_split(unsorted, n_splits=2)
