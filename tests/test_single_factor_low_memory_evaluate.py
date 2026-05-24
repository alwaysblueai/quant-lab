from __future__ import annotations

import pandas as pd
import pytest

from alpha_lab.real_cases.single_factor.evaluate.capacity import (
    _compute_market_cap_weighted_long_short_return,
    _compute_mean_traded_adv,
)
from alpha_lab.real_cases.single_factor.evaluate.coverage import (
    _build_effective_coverage_by_date,
)


def test_effective_coverage_counts_key_intersections_without_full_row_merge() -> None:
    prices = pd.DataFrame(
        {
            "date": [
                "2024-01-01",
                "2024-01-01",
                "2024-01-01",
                "2024-01-02",
                "2024-01-02",
            ],
            "asset": ["A", "B", "C", "A", "B"],
        }
    )
    factor = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-01", "2024-01-01", "2024-01-02"],
            "asset": ["A", "B", "D", "B"],
            "value": [1.0, None, 4.0, 2.0],
        }
    )
    labels = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
            "asset": ["A", "C", "A", "B"],
            "value": [0.1, 0.2, None, 0.3],
        }
    )

    coverage = _build_effective_coverage_by_date(
        prices=prices,
        factor_df=factor,
        label_df=labels,
    )

    by_date = coverage.set_index(pd.to_datetime(coverage["date"]).dt.strftime("%Y-%m-%d"))
    first = by_date.loc["2024-01-01"]
    assert first["eligible_count"] == 3
    assert first["valid_score_count"] == 1
    assert first["valid_forward_return_count"] == 2
    assert first["valid_sample_count"] == 1
    assert first["asset_coverage"] == pytest.approx(1.0 / 3.0)
    assert first["sample_coverage"] == pytest.approx(1.0 / 3.0)

    second = by_date.loc["2024-01-02"]
    assert second["eligible_count"] == 2
    assert second["valid_score_count"] == 1
    assert second["valid_forward_return_count"] == 1
    assert second["valid_sample_count"] == 1
    assert second["asset_coverage"] == pytest.approx(0.5)


def test_market_cap_weighted_long_short_filters_extreme_quantiles_before_join() -> None:
    assignments = pd.DataFrame(
        {
            "date": [
                "2024-01-01",
                "2024-01-01",
                "2024-01-01",
                "2024-01-02",
                "2024-01-02",
                "2024-01-02",
                "2024-01-02",
            ],
            "asset": ["A", "B", "C", "A", "B", "C", "D"],
            "quantile": [5, 1, 3, 5, 1, 5, 1],
        }
    )
    labels = pd.DataFrame(
        {
            "date": [
                "2024-01-01",
                "2024-01-01",
                "2024-01-01",
                "2024-01-02",
                "2024-01-02",
                "2024-01-02",
                "2024-01-02",
            ],
            "asset": ["A", "B", "C", "A", "B", "C", "D"],
            "value": [0.10, 0.01, 0.99, 0.10, 0.01, 0.05, -0.02],
        }
    )
    prices = pd.DataFrame(
        {
            "date": [
                "2024-01-01",
                "2024-01-01",
                "2024-01-01",
                "2024-01-02",
                "2024-01-02",
                "2024-01-02",
                "2024-01-02",
            ],
            "asset": ["A", "B", "C", "A", "B", "C", "D"],
            "total_mv": [100.0, 50.0, 999.0, 100.0, 50.0, 300.0, 50.0],
        }
    )

    weighted = _compute_market_cap_weighted_long_short_return(
        prices=prices,
        labels_df=labels,
        quantile_assignments_df=assignments,
        n_quantiles=5,
        cap_col="total_mv",
    )

    assert weighted == pytest.approx((0.09 + 0.0675) / 2.0)


def test_mean_traded_adv_uses_extreme_quantile_transitions() -> None:
    prices = pd.DataFrame(
        {
            "date": [
                "2024-01-01",
                "2024-01-01",
                "2024-01-01",
                "2024-01-02",
                "2024-01-02",
                "2024-01-02",
                "2024-01-03",
                "2024-01-03",
                "2024-01-03",
            ],
            "asset": ["A", "B", "C"] * 3,
            "amount": [100.0, 200.0, 300.0, 110.0, 220.0, 330.0, 120.0, 240.0, 360.0],
        }
    )
    assignments = pd.DataFrame(
        {
            "date": [
                "2024-01-01",
                "2024-01-01",
                "2024-01-01",
                "2024-01-02",
                "2024-01-02",
                "2024-01-02",
                "2024-01-03",
                "2024-01-03",
                "2024-01-03",
            ],
            "asset": ["A", "B", "C"] * 3,
            "quantile": [5, 1, 3, 1, 1, 5, 1, 5, 5],
        }
    )

    mean_adv = _compute_mean_traded_adv(
        prices=prices,
        quantile_assignments_df=assignments,
        n_quantiles=5,
        rebalance_step=1,
        adv_lookback=2,
    )

    assert mean_adv == pytest.approx((210.0 + 230.0) / 2.0)
