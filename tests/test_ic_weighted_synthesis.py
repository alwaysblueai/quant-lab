from __future__ import annotations

import pandas as pd
import pytest

from alpha_lab.synthesis.ic_weighted import (
    build_ic_weighted_composite,
    compute_rolling_icir,
)


def test_compute_rolling_icir_matches_hand_calculation() -> None:
    dates = pd.date_range("2024-01-01", periods=4, freq="B")
    ic_dict = {
        "factor_a": pd.DataFrame(
            {
                "date": dates,
                "ic": [1.0, 2.0, 3.0, 4.0],
            }
        )
    }
    out = compute_rolling_icir(ic_dict, window=3)

    row_t3 = out.loc[(out["factor"] == "factor_a") & (out["date"] == dates[2])].iloc[0]
    row_t4 = out.loc[(out["factor"] == "factor_a") & (out["date"] == dates[3])].iloc[0]
    assert float(row_t3["rolling_icir"]) == pytest.approx(2.0)
    assert float(row_t4["rolling_icir"]) == pytest.approx(3.0)


def test_ic_weighted_composite_deweights_zero_icir_factor() -> None:
    dates = pd.date_range("2024-01-01", periods=3, freq="B")
    factor_a = pd.DataFrame(
        {
            "date": [dates[0], dates[0], dates[1], dates[1], dates[2], dates[2]],
            "asset": ["A", "B", "A", "B", "A", "B"],
            "factor": "factor_a",
            "value": [1.0, 2.0, 1.1, 2.1, 1.2, 2.2],
        }
    )
    factor_b = pd.DataFrame(
        {
            "date": [dates[0], dates[0], dates[1], dates[1], dates[2], dates[2]],
            "asset": ["A", "B", "A", "B", "A", "B"],
            "factor": "factor_b",
            "value": [9.0, 8.0, 9.1, 8.1, 9.2, 8.2],
        }
    )
    ic_dict = {
        "factor_a": pd.DataFrame({"date": dates, "ic": [0.10, 0.20, 0.30]}),
        "factor_b": pd.DataFrame({"date": dates, "ic": [0.0, 0.0, 0.0]}),
    }
    composite = build_ic_weighted_composite(
        {"factor_a": factor_a, "factor_b": factor_b},
        ic_dict,
        window=2,
        min_positive_icir=0.0,
    )

    merged = composite.merge(
        factor_a[["date", "asset", "value"]].rename(columns={"value": "a_value"}),
        on=["date", "asset"],
        how="inner",
    )
    # First date has insufficient rolling history (window=2); compare the rest.
    merged = merged[merged["date"] > dates[0]]
    assert (merged["factor"] == "ic_weighted_composite").all()
    assert merged["value"].tolist() == pytest.approx(merged["a_value"].tolist())
