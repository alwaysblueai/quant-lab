from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_lab.splits import time_split, walk_forward_split


def _dates(start: str, n: int) -> pd.Series:
    return pd.Series(pd.date_range(start, periods=n, freq="B"))


# ---------------------------------------------------------------------------
# time_split — boundary correctness
# ---------------------------------------------------------------------------


def test_time_split_train_ends_at_train_end():
    dates = _dates("2024-01-01", 20)
    masks = time_split(dates, train_end="2024-01-12", test_start="2024-01-16")
    train_max = dates[masks["train"]].max()
    assert train_max <= pd.Timestamp("2024-01-12")


def test_time_split_test_starts_at_test_start():
    dates = _dates("2024-01-01", 20)
    masks = time_split(dates, train_end="2024-01-12", test_start="2024-01-16")
    test_min = dates[masks["test"]].min()
    assert test_min >= pd.Timestamp("2024-01-16")


def test_time_split_chronological_ordering():
    dates = _dates("2024-01-01", 30)
    masks = time_split(dates, train_end="2024-01-15", test_start="2024-01-22")
    train_max = dates[masks["train"]].max()
    test_min = dates[masks["test"]].min()
    assert train_max < test_min


# ---------------------------------------------------------------------------
# time_split — no overlap
# ---------------------------------------------------------------------------


def test_time_split_no_overlap_train_test():
    dates = _dates("2024-01-01", 30)
    masks = time_split(dates, train_end="2024-01-15", test_start="2024-01-20")
    assert not np.any(masks["train"] & masks["test"])


def test_time_split_three_way_no_overlap():
    dates = _dates("2024-01-01", 30)
    masks = time_split(
        dates,
        train_end="2024-01-12",
        val_start="2024-01-15",
        test_start="2024-01-22",
    )
    assert not np.any(masks["train"] & masks["val"])
    assert not np.any(masks["train"] & masks["test"])
    assert not np.any(masks["val"] & masks["test"])


def test_time_split_three_way_chronological():
    dates = _dates("2024-01-01", 30)
    masks = time_split(
        dates,
        train_end="2024-01-12",
        val_start="2024-01-15",
        test_start="2024-01-22",
    )
    train_max = dates[masks["train"]].max()
    val_min = dates[masks["val"]].min()
    val_max = dates[masks["val"]].max()
    test_min = dates[masks["test"]].min()
    assert train_max < val_min
    assert val_max < test_min


# ---------------------------------------------------------------------------
# time_split — return type
# ---------------------------------------------------------------------------


def test_time_split_returns_boolean_arrays():
    dates = _dates("2024-01-01", 10)
    masks = time_split(dates, train_end="2024-01-05", test_start="2024-01-08")
    for mask in masks.values():
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool
        assert len(mask) == len(dates)


def test_time_split_without_val_has_no_val_key():
    dates = _dates("2024-01-01", 20)
    masks = time_split(dates, train_end="2024-01-10", test_start="2024-01-15")
    assert "val" not in masks


# ---------------------------------------------------------------------------
# time_split — validation
# ---------------------------------------------------------------------------


def test_time_split_rejects_train_end_ge_test_start():
    dates = _dates("2024-01-01", 20)
    with pytest.raises(ValueError, match="strictly before"):
        time_split(dates, train_end="2024-01-20", test_start="2024-01-10")


def test_time_split_rejects_val_start_before_train_end():
    dates = _dates("2024-01-01", 30)
    with pytest.raises(ValueError, match="strictly after"):
        time_split(
            dates,
            train_end="2024-01-15",
            val_start="2024-01-10",
            test_start="2024-01-22",
        )


def test_time_split_rejects_val_start_after_test_start():
    dates = _dates("2024-01-01", 30)
    with pytest.raises(ValueError, match="at or before"):
        time_split(
            dates,
            train_end="2024-01-12",
            val_start="2024-01-25",
            test_start="2024-01-22",
        )


# ---------------------------------------------------------------------------
# walk_forward_split — boundary correctness
# ---------------------------------------------------------------------------


def test_walk_forward_correct_fold_count():
    # train=10, test=5, step=5, n=20
    # fold 0: [0:10] train, [10:15] test
    # fold 1: [5:15] train, [15:20] test
    # fold 2: [10:20] train, [20:25] test → 25 > 20, stop
    dates = _dates("2024-01-01", 20)
    splits = walk_forward_split(dates, train_size=10, test_size=5, step=5)
    assert len(splits) == 2


def test_walk_forward_window_sizes():
    dates = _dates("2024-01-01", 50)
    splits = walk_forward_split(dates, train_size=20, test_size=5, step=5, val_size=4)
    for fold in splits:
        assert fold["train"].sum() == 20 - 4
        assert fold["val"].sum() == 4
        assert fold["test"].sum() == 5


def test_walk_forward_empty_when_data_too_short():
    dates = _dates("2024-01-01", 5)
    splits = walk_forward_split(dates, train_size=10, test_size=5, step=5)
    assert len(splits) == 0


# ---------------------------------------------------------------------------
# walk_forward_split — no overlap
# ---------------------------------------------------------------------------


def test_walk_forward_no_overlap_within_fold():
    dates = _dates("2024-01-01", 30)
    splits = walk_forward_split(dates, train_size=10, test_size=5, step=3)
    for fold in splits:
        assert not np.any(fold["train"] & fold["test"])


def test_walk_forward_with_val_no_overlap():
    dates = _dates("2024-01-01", 40)
    splits = walk_forward_split(dates, train_size=15, test_size=5, step=5, val_size=3)
    for fold in splits:
        assert not np.any(fold["train"] & fold["val"])
        assert not np.any(fold["train"] & fold["test"])
        assert not np.any(fold["val"] & fold["test"])


# ---------------------------------------------------------------------------
# walk_forward_split — chronological ordering
# ---------------------------------------------------------------------------


def test_walk_forward_test_follows_train():
    dates = _dates("2024-01-01", 30)
    splits = walk_forward_split(dates, train_size=10, test_size=5, step=5)
    for fold in splits:
        train_max = dates[fold["train"]].max()
        test_min = dates[fold["test"]].min()
        assert train_max < test_min


def test_walk_forward_val_between_train_and_test():
    dates = _dates("2024-01-01", 40)
    splits = walk_forward_split(dates, train_size=15, test_size=5, step=5, val_size=3)
    for fold in splits:
        train_max = dates[fold["train"]].max()
        val_min = dates[fold["val"]].min()
        val_max = dates[fold["val"]].max()
        test_min = dates[fold["test"]].min()
        assert train_max < val_min
        assert val_max < test_min


# ---------------------------------------------------------------------------
# walk_forward_split — return type
# ---------------------------------------------------------------------------


def test_walk_forward_returns_boolean_arrays():
    dates = _dates("2024-01-01", 20)
    splits = walk_forward_split(dates, train_size=10, test_size=5, step=5)
    for fold in splits:
        for mask in fold.values():
            assert isinstance(mask, np.ndarray)
            assert mask.dtype == bool
            assert len(mask) == len(dates)


# ---------------------------------------------------------------------------
# walk_forward_split — validation
# ---------------------------------------------------------------------------


def test_walk_forward_rejects_unsorted_dates():
    dates = pd.Series(
        [pd.Timestamp("2024-01-05"), pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-07")]
    )
    with pytest.raises(ValueError, match="sorted"):
        walk_forward_split(dates, train_size=2, test_size=1, step=1)


def test_walk_forward_rejects_non_positive_train_size():
    dates = _dates("2024-01-01", 20)
    with pytest.raises(ValueError):
        walk_forward_split(dates, train_size=0, test_size=5, step=5)


def test_walk_forward_rejects_val_size_ge_train_size():
    dates = _dates("2024-01-01", 20)
    with pytest.raises(ValueError):
        walk_forward_split(dates, train_size=5, test_size=3, step=2, val_size=5)


# ---------------------------------------------------------------------------
# Regression: repeated dates (panel data leakage — critical fix)
# ---------------------------------------------------------------------------


def test_walk_forward_rejects_repeated_dates():
    """Panel data has many rows per date.  Passing the full date column to
    walk_forward_split() would split a shared timestamp across train/test,
    creating leakage.  The function must error explicitly."""
    # Simulate a 3-asset panel: each date appears 3 times
    panel_dates = pd.Series(
        pd.date_range("2024-01-01", periods=5, freq="B").repeat(3)
    ).sort_values().reset_index(drop=True)

    with pytest.raises(ValueError, match="repeated values"):
        walk_forward_split(panel_dates, train_size=6, test_size=3, step=3)


def test_time_split_with_panel_data_no_leakage():
    """time_split() uses date-comparison masks so all rows sharing a date
    receive the same split assignment — panel data is safe."""
    # 3 assets × 10 dates
    dates_unique = pd.date_range("2024-01-01", periods=10, freq="B")
    panel_dates = pd.Series(dates_unique.repeat(3))

    masks = time_split(
        panel_dates,
        train_end="2024-01-10",
        test_start="2024-01-12",
    )
    assert not np.any(masks["train"] & masks["test"])

    # Every occurrence of a shared date must be in the same split
    df = pd.DataFrame({"date": panel_dates, "train": masks["train"], "test": masks["test"]})
    for _, group in df.groupby("date"):
        assert group["train"].nunique() == 1, "all assets on the same date must share train mask"
        assert group["test"].nunique() == 1, "all assets on the same date must share test mask"


# ---------------------------------------------------------------------------
# Regression: NaT validation
# ---------------------------------------------------------------------------


def test_time_split_rejects_nat_in_dates():
    dates = pd.Series([pd.Timestamp("2024-01-01"), pd.NaT, pd.Timestamp("2024-01-05")])
    with pytest.raises(ValueError, match="NaT"):
        time_split(dates, train_end="2024-01-03", test_start="2024-01-05")


def test_walk_forward_rejects_nat_in_dates():
    dates = pd.Series([pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02"), pd.NaT])
    with pytest.raises(ValueError, match="NaT"):
        walk_forward_split(dates, train_size=2, test_size=1, step=1)
