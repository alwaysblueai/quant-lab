from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_lab.exceptions import AlphaLabDataError
from alpha_lab.grouped_evaluation import (
    conditional_ic_by_bucket,
    conditional_ic_by_factor_magnitude,
)


def test_conditional_ic_by_bucket_returns_ic_and_rank_ic() -> None:
    date = pd.Timestamp("2024-01-01")
    factor = pd.DataFrame(
        {
            "date": [date] * 4,
            "asset": ["A", "B", "C", "D"],
            "factor": ["f"] * 4,
            "value": [1.0, 2.0, 3.0, 4.0],
        }
    )
    labels = pd.DataFrame(
        {
            "date": [date] * 4,
            "asset": ["A", "B", "C", "D"],
            "factor": ["ret"] * 4,
            "value": [10.0, 20.0, 30.0, 40.0],
        }
    )
    buckets = pd.DataFrame({"date": [date], "market_regime": ["bull"]})

    out = conditional_ic_by_bucket(factor, labels, buckets, group_col="market_regime")

    assert list(out.columns) == ["date", "bucket", "ic", "rank_ic", "n"]
    assert float(out.loc[0, "ic"]) == pytest.approx(1.0)
    assert float(out.loc[0, "rank_ic"]) == pytest.approx(1.0)
    assert int(out.loc[0, "n"]) == 4


def test_conditional_ic_by_bucket_keeps_low_sample_bucket_as_nan() -> None:
    date = pd.Timestamp("2024-01-01")
    factor = pd.DataFrame(
        {
            "date": [date] * 2,
            "asset": ["A", "B"],
            "factor": ["f"] * 2,
            "value": [1.0, 2.0],
        }
    )
    labels = pd.DataFrame(
        {
            "date": [date] * 2,
            "asset": ["A", "B"],
            "factor": ["ret"] * 2,
            "value": [2.0, 1.0],
        }
    )
    buckets = pd.DataFrame(
        {
            "date": [date] * 2,
            "asset": ["A", "B"],
            "bucket": ["thin", "thin"],
        }
    )

    out = conditional_ic_by_bucket(factor, labels, buckets, min_assets=3)

    assert out["bucket"].tolist() == ["thin"]
    assert int(out.loc[0, "n"]) == 2
    assert np.isnan(float(out.loc[0, "ic"]))
    assert np.isnan(float(out.loc[0, "rank_ic"]))


def test_conditional_ic_rejects_mixed_factor_names_before_summarizing() -> None:
    dates = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"])
    factor = pd.DataFrame(
        {
            "date": dates,
            "asset": ["A", "B", "C", "D"],
            "factor": ["f1", "f1", "f2", "f2"],
            "value": [1.0, 2.0, 3.0, 4.0],
        }
    )
    labels = pd.DataFrame(
        {
            "date": dates,
            "asset": ["A", "B", "C", "D"],
            "factor": ["ret"] * 4,
            "value": [1.0, 2.0, 3.0, 4.0],
        }
    )

    with pytest.raises(AlphaLabDataError, match="factor_df must contain exactly one factor name"):
        conditional_ic_by_factor_magnitude(factor, labels, n_buckets=2)
