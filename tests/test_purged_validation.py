from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from alpha_lab.purged_validation import (
    combinatorial_purged_split,
    overlapping_index,
    purged_fold_summary,
    purged_kfold_split,
)


def _dates(n: int = 8) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=n, freq="B")


def test_purged_kfold_split_no_overlap_no_purge() -> None:
    d = _dates(6)
    samples = pd.DataFrame(
        {
            "date": d,
            "event_start": d,
            "event_end": d,
        }
    )
    folds = purged_kfold_split(samples, n_splits=3, embargo_periods=0)
    assert len(folds) == 3
    assert all(f.n_purged == 0 for f in folds)
    assert all(f.n_test > 0 for f in folds)


def test_purged_kfold_split_with_embargo() -> None:
    d = _dates(6)
    samples = pd.DataFrame(
        {
            "date": d,
            "event_start": d,
            "event_end": d,
        }
    )
    folds = purged_kfold_split(samples, n_splits=3, embargo_periods=1)
    # Fold-0 test is first date block, next block should be embargoed.
    assert folds[0].n_embargoed > 0


def test_purged_kfold_split_purges_overlaps() -> None:
    d = _dates(6)
    samples = pd.DataFrame(
        {
            "date": d,
            "event_start": [d[0], d[1], d[2], d[3], d[4], d[5]],
            "event_end": [d[2], d[3], d[4], d[5], d[5], d[5]],
        }
    )
    folds = purged_kfold_split(samples, n_splits=3, embargo_periods=0)
    assert any(f.n_purged > 0 for f in folds)


def test_overlapping_index_returns_expected_rows() -> None:
    d = _dates(5)
    left = pd.DataFrame(
        {
            "event_start": [d[0], d[1], d[3]],
            "event_end": [d[0], d[2], d[4]],
        }
    )
    right = pd.DataFrame(
        {
            "event_start": [d[2]],
            "event_end": [d[3]],
        }
    )
    idx = overlapping_index(left, right)
    assert set(idx.tolist()) == {1, 2}


def test_combinatorial_purged_split_count_matches_n_choose_k() -> None:
    d = _dates(12)
    samples = pd.DataFrame(
        {
            "date": d,
            "event_start": d,
            "event_end": d,
        }
    )
    folds = combinatorial_purged_split(samples, n_groups=4, n_test_groups=2, embargo_periods=0)
    assert len(folds) == 6
    assert all(f.n_test > 0 for f in folds)


def test_purged_fold_summary_schema() -> None:
    d = _dates(6)
    samples = pd.DataFrame(
        {
            "date": d,
            "event_start": d,
            "event_end": d,
        }
    )
    folds = purged_kfold_split(samples, n_splits=3, embargo_periods=0)
    summary = purged_fold_summary(folds)
    assert list(summary.columns) == [
        "fold_id",
        "n_train",
        "n_test",
        "n_purged",
        "n_embargoed",
        "test_start",
        "test_end",
    ]
    assert math.isfinite(float(summary["n_test"].sum()))


def test_purged_kfold_split_rejects_bad_intervals() -> None:
    d = _dates(2)
    samples = pd.DataFrame(
        {
            "date": d,
            "event_start": [d[1], d[0]],
            "event_end": [d[0], d[1]],
        }
    )
    with pytest.raises(ValueError, match="event_end < event_start"):
        purged_kfold_split(samples, n_splits=2)


def test_overlapping_index_large_input_avoids_pairwise_broadcast() -> None:
    n_samples = 50_000
    n_test = 30_000
    base = pd.Timestamp("2000-01-01")

    sample_start = base + pd.to_timedelta(np.arange(n_samples), unit="D")
    left = pd.DataFrame({"event_start": sample_start, "event_end": sample_start})

    window_start = base + pd.Timedelta(days=1_000)
    window_end = base + pd.Timedelta(days=40_000)
    right = pd.DataFrame(
        {
            "event_start": np.repeat(window_start, n_test),
            "event_end": np.repeat(window_end, n_test),
        }
    )

    idx = overlapping_index(left, right)
    assert int(idx[0]) == 1_000
    assert int(idx[-1]) == 40_000
    assert int(len(idx)) == 39_001


def test_purged_kfold_split_has_clear_cardinality_guard() -> None:
    d = pd.date_range("2010-01-01", periods=5_001, freq="D")
    samples = pd.DataFrame(
        {
            "date": d,
            "asset": ["A"] * len(d),
            "event_start": d,
            "event_end": d,
        }
    )
    with pytest.raises(ValueError, match="purged_kfold_split.prepare_intervals"):
        purged_kfold_split(samples, n_splits=2)
