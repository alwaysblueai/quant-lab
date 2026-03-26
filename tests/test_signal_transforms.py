from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_lab.signal_transforms import (
    apply_min_coverage_gate,
    neutralize_by_group,
    rank_cross_section,
    winsorize_cross_section,
    zscore_cross_section,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-01")] * 4 + [pd.Timestamp("2024-01-02")] * 4,
            "asset": ["A", "B", "C", "D", "A", "B", "C", "D"],
            "group": ["g1", "g1", "g2", "g2", "g1", "g1", "g2", "g2"],
            "value": [1.0, 2.0, 3.0, 100.0, 10.0, np.nan, 30.0, 40.0],
        }
    )


def test_winsorize_cross_section_clips_extremes() -> None:
    out = winsorize_cross_section(_frame(), lower=0.0, upper=0.75, min_group_size=2)
    day1 = out[out["date"] == pd.Timestamp("2024-01-01")]["value"]
    assert float(day1.max()) <= 100.0
    assert float(day1.max()) < 100.0


def test_zscore_cross_section_small_group_returns_nan() -> None:
    small = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-01")],
            "asset": ["A", "B"],
            "value": [1.0, 2.0],
        }
    )
    out = zscore_cross_section(small, min_group_size=3)
    assert out["value"].isna().all()


def test_rank_cross_section_returns_percentile_ranks() -> None:
    out = rank_cross_section(_frame(), min_group_size=2)
    vals = out.loc[out["date"] == pd.Timestamp("2024-01-01"), "value"].dropna()
    assert (vals >= 0).all()
    assert (vals <= 1).all()


def test_neutralize_by_group_demeans_each_group() -> None:
    out = neutralize_by_group(_frame(), group_col="group", min_group_size=2)
    day = out[out["date"] == pd.Timestamp("2024-01-01")]
    for _, group in day.groupby("group"):
        assert abs(float(group["value"].mean())) < 1e-12


def test_apply_min_coverage_gate_filters_dates() -> None:
    out = apply_min_coverage_gate(_frame(), min_coverage=0.9)
    dates = set(pd.to_datetime(out["date"]).unique())
    assert dates == {pd.Timestamp("2024-01-01")}
