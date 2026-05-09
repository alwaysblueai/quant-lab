from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_lab.costs import apply_short_borrow_cost
from alpha_lab.data_quality.outlier_detection import (
    detect_price_jumps,
    detect_stale_prices,
    filter_zero_volume,
)
from alpha_lab.data_quality.survivorship import (
    apply_delisting_return,
    build_delisting_calendar,
    validate_universe_survivorship,
)
from alpha_lab.exceptions import AlphaLabDataError
from alpha_lab.grouped_evaluation import (
    compute_ic_by_group,
    compute_ic_by_size_bucket,
    conditional_ic_by_bucket,
    conditional_ic_by_cross_section_size,
    conditional_ic_by_factor_magnitude,
)
from alpha_lab.optimization.risk_parity import risk_parity_weights
from alpha_lab.risk_model.barra import BarraExposures, extract_pure_alpha
from alpha_lab.signal_transforms import mad_winsorize_cross_section
from alpha_lab.transforms.fracdiff import (
    find_min_d,
    fracdiff_cross_section,
    fracdiff_series,
    fracdiff_weights,
)
from alpha_lab.transforms.information_bars import dollar_bars, tick_bars
from alpha_lab.universe.dynamic_universe import (
    build_liquidity_universe,
    build_market_cap_universe,
    combine_universe_filters,
)
from alpha_lab.validation.cpcv import compute_pbo, cpcv_split
from alpha_lab.validation.multiple_testing import apply_multiple_testing_correction


def test_detect_price_jumps_flags_large_move() -> None:
    prices = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "asset": ["A", "A", "A"],
            "close": [10.0, 10.5, 20.0],
        }
    )
    out = detect_price_jumps(prices, threshold=0.11)
    assert out["is_price_jump"].tolist() == [False, False, True]


def test_filter_zero_volume_flag_and_drop() -> None:
    prices = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "asset": ["A", "A"],
            "volume": [100.0, 0.0],
        }
    )
    flagged = filter_zero_volume(prices, action="flag")
    assert flagged["is_suspended"].tolist() == [False, True]

    dropped = filter_zero_volume(prices, action="drop")
    assert len(dropped) == 1


def test_detect_stale_prices_flags_long_identical_run() -> None:
    dates = pd.date_range("2024-01-01", periods=6, freq="B")
    prices = pd.DataFrame({"date": dates, "asset": "A", "close": 10.0})
    out = detect_stale_prices(prices, max_identical_days=5)
    assert out["is_stale_price"].all()


def test_survivorship_calendar_and_penalty_row() -> None:
    delist = pd.DataFrame({"asset": ["A"], "delist_date": [pd.Timestamp("2024-01-10")]})
    calendar = build_delisting_calendar(delist)
    assert calendar.loc[0, "last_trade_date"] == pd.Timestamp("2024-01-09")

    prices = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-08", "2024-01-09"]),
            "asset": ["A", "A"],
            "close": [10.0, 11.0],
        }
    )
    adjusted = apply_delisting_return(prices, calendar, penalty=-0.10)
    penalty_row = adjusted.loc[
        (adjusted["asset"] == "A") & (adjusted["is_delisting_penalty"])
    ].iloc[0]
    assert penalty_row["date"] == pd.Timestamp("2024-01-10")
    assert float(penalty_row["close"]) == pytest.approx(9.9)
    assert float(penalty_row["delisting_return"]) == pytest.approx(-0.10)


def test_validate_universe_survivorship_fails_when_assets_missing() -> None:
    universe = pd.DataFrame({"date": ["2024-01-01"], "asset": ["A"], "in_universe": [True]})
    result = validate_universe_survivorship(universe, all_historical_assets=["A", "B"])
    assert result.status == "fail"


def test_fracdiff_weights_and_series_endpoints() -> None:
    w = fracdiff_weights(d=1.0, threshold=1e-6)
    assert w[0] == pytest.approx(1.0)
    assert w[1] == pytest.approx(-1.0)

    series = pd.Series([1.0, 2.0, 4.0, 7.0], dtype=float)
    same = fracdiff_series(series, d=0.0)
    pd.testing.assert_series_equal(same, series)

    diff = fracdiff_series(series, d=1.0)
    expected = series.diff()
    pd.testing.assert_series_equal(diff, expected)


def test_find_min_d_returns_in_range() -> None:
    rng = np.random.RandomState(42)
    walk = pd.Series(np.cumsum(rng.normal(0, 1, size=300)), dtype=float)
    d = find_min_d(walk, target_adf_pvalue=0.10, steps=7)
    assert 0.0 <= d <= 1.0


def test_fracdiff_cross_section_returns_canonical_columns() -> None:
    dates = pd.date_range("2024-01-01", periods=6, freq="B")
    rows = []
    for asset, base in [("A", 1.0), ("B", 2.0)]:
        for i, date in enumerate(dates):
            rows.append({"date": date, "asset": asset, "factor": "f", "value": base + i})
    frame = pd.DataFrame(rows)
    out = fracdiff_cross_section(frame, d=0.5)
    assert list(out.columns) == ["date", "asset", "factor", "value"]
    assert len(out) == len(frame)


def test_dollar_and_tick_bars_aggregate() -> None:
    ohlcv = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                ["2024-01-01 09:31", "2024-01-01 09:32", "2024-01-01 09:33", "2024-01-01 09:34"]
            ),
            "asset": ["A"] * 4,
            "open": [10.0, 10.1, 10.2, 10.3],
            "high": [10.1, 10.2, 10.3, 10.4],
            "low": [9.9, 10.0, 10.1, 10.2],
            "close": [10.05, 10.15, 10.25, 10.35],
            "volume": [10.0, 10.0, 10.0, 10.0],
            "amount": [60.0, 60.0, 40.0, 40.0],
        }
    )
    bars = dollar_bars(ohlcv, target_dollar_volume=100.0)
    assert len(bars) == 2
    assert float(bars.iloc[0]["amount"]) == pytest.approx(120.0)

    ticks = pd.DataFrame(
        {
            "datetime": pd.date_range("2024-01-01 09:30", periods=5, freq="min"),
            "asset": ["A"] * 5,
            "price": [10.0, 10.1, 10.2, 10.3, 10.4],
        }
    )
    tick_out = tick_bars(ticks, target_ticks=2)
    assert tick_out["n_ticks"].tolist() == [2, 2, 1]


def test_mad_winsorize_cross_section_clips_outlier() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01"] * 4),
            "asset": ["A", "B", "C", "D"],
            "factor": ["f"] * 4,
            "value": [1.0, 1.0, 2.0, 10.0],
        }
    )
    out = mad_winsorize_cross_section(frame, k=3.0)
    assert float(out.loc[out["asset"] == "D", "value"].iloc[0]) <= 3.0


def test_dynamic_universe_builders_and_combiner() -> None:
    prices = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"] * 3),
            "asset": ["A", "A", "B", "B", "C", "C"],
            "amount": [100.0, 100.0, 50.0, 50.0, 10.0, 10.0],
        }
    )
    liq = build_liquidity_universe(prices, min_adv_pct=34, lookback=2)
    day2 = liq[liq["date"] == pd.Timestamp("2024-01-02")]
    assert day2.loc[day2["asset"] == "A", "in_universe"].iloc[0]

    caps = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02"] * 4),
            "asset": ["A", "B", "C", "D"],
            "circ_mv": [100.0, 80.0, 30.0, 10.0],
        }
    )
    cap_uni = build_market_cap_universe(caps, min_cap_pct=50)
    assert int(cap_uni["in_universe"].sum()) == 2

    combined = combine_universe_filters(liq, cap_uni)
    assert {"date", "asset", "in_universe"} == set(combined.columns)


def test_grouped_ic_and_size_bucket_ic() -> None:
    date = pd.Timestamp("2024-01-01")
    factor = pd.DataFrame(
        {
            "date": [date] * 6,
            "asset": ["A", "B", "C", "D", "E", "F"],
            "factor": ["f"] * 6,
            "value": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
        }
    )
    labels = pd.DataFrame(
        {
            "date": [date] * 6,
            "asset": ["A", "B", "C", "D", "E", "F"],
            "factor": ["ret"] * 6,
            "value": [1.0, 2.0, 3.0, 3.0, 2.0, 1.0],
        }
    )
    groups = pd.DataFrame(
        {
            "date": [date] * 6,
            "asset": ["A", "B", "C", "D", "E", "F"],
            "sector": ["G1", "G1", "G1", "G2", "G2", "G2"],
        }
    )
    out = compute_ic_by_group(factor, labels, groups, group_col="sector")
    g1 = float(out.loc[out["group"] == "G1", "ic"].iloc[0])
    g2 = float(out.loc[out["group"] == "G2", "ic"].iloc[0])
    assert g1 == pytest.approx(1.0)
    assert g2 == pytest.approx(-1.0)

    market_cap = pd.DataFrame(
        {
            "date": [date] * 6,
            "asset": ["A", "B", "C", "D", "E", "F"],
            "circ_mv": [100, 90, 80, 70, 60, 50],
        }
    )
    by_size = compute_ic_by_size_bucket(factor, labels, market_cap, n_buckets=3)
    assert "size_bucket" in by_size.columns


def test_grouped_ic_rejects_duplicate_group_rows() -> None:
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
            "value": [1.0, 2.0],
        }
    )
    groups = pd.DataFrame(
        {
            "date": [date, date, date],
            "asset": ["A", "A", "B"],
            "sector": ["G1", "G1", "G1"],
        }
    )

    with pytest.raises(AlphaLabDataError, match="duplicate"):
        compute_ic_by_group(factor, labels, groups, group_col="sector")


def test_size_bucket_ic_drops_unassigned_small_cross_sections() -> None:
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
            "value": [1.0, 2.0],
        }
    )
    market_cap = pd.DataFrame(
        {
            "date": [date] * 2,
            "asset": ["A", "B"],
            "circ_mv": [100.0, 50.0],
        }
    )

    out = compute_ic_by_size_bucket(factor, labels, market_cap, n_buckets=3)

    assert out.empty
    assert list(out.columns) == ["date", "size_bucket", "factor", "label", "ic"]


def test_conditional_ic_summaries_cover_magnitude_and_cross_section() -> None:
    dates = pd.to_datetime(
        [
            "2024-01-01",
            "2024-01-02",
            "2024-01-03",
            "2024-01-04",
        ]
    )
    rows: list[dict[str, object]] = []
    for date in dates:
        for asset_idx in range(15):
            asset = f"A{asset_idx + 1}"
            factor_value = float(asset_idx - 7.0)
            if abs(factor_value) >= 5.0:
                label_value = factor_value
            elif abs(factor_value) <= 1.0:
                label_value = -factor_value
            else:
                label_value = 0.5 * factor_value
            rows.append(
                {
                    "date": date,
                    "asset": asset,
                    "factor": "f",
                    "value": factor_value,
                    "label": label_value,
                }
            )

    factor = pd.DataFrame(rows)[["date", "asset", "factor", "value"]]
    labels = pd.DataFrame(rows)[["date", "asset"]].copy()
    labels["factor"] = "ret"
    labels["value"] = pd.DataFrame(rows)["label"]

    magnitude = conditional_ic_by_factor_magnitude(factor, labels)
    assert list(magnitude.columns) == [
        "magnitude_quintile",
        "mean_ic",
        "mean_rank_ic",
        "ic_positive_rate",
        "rank_ic_positive_rate",
        "n_dates_used",
        "mean_assets_per_date",
    ]
    assert magnitude["magnitude_quintile"].tolist() == ["Q1", "Q2", "Q3", "Q4", "Q5"]
    q1 = float(magnitude.loc[magnitude["magnitude_quintile"] == "Q1", "mean_ic"].iloc[0])
    q5 = float(magnitude.loc[magnitude["magnitude_quintile"] == "Q5", "mean_ic"].iloc[0])
    assert q5 > q1

    sparse_factor = factor.loc[
        ~(
            (factor["date"] == pd.Timestamp("2024-01-01"))
            & (factor["asset"].isin(["A12", "A13", "A14", "A15"]))
        )
    ].reset_index(drop=True)
    sparse_labels = labels.merge(
        sparse_factor[["date", "asset"]],
        on=["date", "asset"],
        how="inner",
    )
    cross_section = conditional_ic_by_cross_section_size(sparse_factor, sparse_labels)
    assert list(cross_section.columns) == [
        "cross_section_bucket",
        "median_valid_assets_threshold",
        "mean_valid_assets",
        "mean_ic",
        "mean_rank_ic",
        "ic_positive_rate",
        "rank_ic_positive_rate",
        "n_dates_used",
    ]
    assert cross_section["cross_section_bucket"].tolist() == [
        "small_cross_section",
        "large_cross_section",
    ]
    assert cross_section["median_valid_assets_threshold"].notna().all()


def test_conditional_ic_by_bucket_supports_date_level_research_buckets() -> None:
    dates = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
    factor_rows: list[dict[str, object]] = []
    for date_idx, date in enumerate(dates):
        for asset_idx in range(6):
            asset = f"A{asset_idx + 1}"
            value = float(asset_idx)
            factor_rows.append(
                {
                    "date": date,
                    "asset": asset,
                    "factor": "f",
                    "value": value,
                    "label": value + float(date_idx) * 0.01,
                }
            )

    factor = pd.DataFrame(factor_rows)[["date", "asset", "factor", "value"]]
    labels = pd.DataFrame(factor_rows)[["date", "asset"]].copy()
    labels["factor"] = "ret"
    labels["value"] = pd.DataFrame(factor_rows)["label"]

    buckets = pd.DataFrame({"date": dates, "market_regime": ["low", "mid", "high"]})
    out = conditional_ic_by_bucket(factor, labels, buckets, group_col="market_regime")

    assert list(out.columns) == ["date", "bucket", "ic", "rank_ic", "n"]
    assert out["bucket"].tolist() == ["low", "mid", "high"]
    assert out["n"].min() == 6


def test_extract_pure_alpha_near_zero_for_size_explained_signal() -> None:
    date = pd.Timestamp("2024-01-01")
    exposures = BarraExposures(
        exposures=pd.DataFrame(
            {
                "date": [date] * 5,
                "asset": ["A", "B", "C", "D", "E"],
                "circ_mv": [1, 1, 1, 1, 1],
                "size": [1, 2, 3, 4, 5],
            }
        ),
        style_factors=("size",),
        industry_factors=(),
    )
    alpha = pd.DataFrame(
        {
            "date": [date] * 5,
            "asset": ["A", "B", "C", "D", "E"],
            "factor": ["raw_alpha"] * 5,
            "value": [2, 4, 6, 8, 10],
        }
    )
    pure = extract_pure_alpha(alpha, exposures)
    assert np.max(np.abs(pure["value"].to_numpy(dtype=float))) < 1e-8


def test_risk_parity_weights_basic_properties() -> None:
    cov = np.diag([0.04, 0.04, 0.04])
    w = risk_parity_weights(cov)
    assert np.isclose(float(np.sum(w)), 1.0, atol=1e-8)
    assert np.all(w >= 0.0)
    assert np.allclose(w, np.array([1 / 3, 1 / 3, 1 / 3]), atol=1e-2)


def test_multiple_testing_correction_returns_expected_schema() -> None:
    pvals = {"f1": 0.01, "f2": 0.04, "f3": 0.20}
    out = apply_multiple_testing_correction(pvals, method="bh", alpha=0.05)
    assert list(out.columns) == ["factor", "p_value", "corrected_p_value", "reject_null"]
    assert set(out["factor"]) == set(pvals.keys())


def test_cpcv_split_and_pbo() -> None:
    dates = pd.date_range("2024-01-01", periods=12, freq="B").to_numpy()
    splits = cpcv_split(dates, n_splits=4, n_test_splits=2, label_horizon=1, embargo_pct=0.0)
    assert len(splits) == 6  # C(4, 2)
    for split in splits:
        assert not np.any(split["train"] & split["test"])

    is_scores = pd.DataFrame([[0.9, 0.2], [0.8, 0.1], [0.7, 0.3]])
    oos_scores = pd.DataFrame([[0.1, 0.9], [0.2, 0.8], [0.1, 0.7]])
    pbo = compute_pbo(is_scores, oos_scores)
    assert 0.0 <= pbo <= 1.0
    assert pbo > 0.5


def test_apply_short_borrow_cost() -> None:
    returns = pd.Series([0.01, 0.01], index=[0, 1])
    shorts = pd.Series([-0.5, 0.2], index=[0, 1])
    adjusted = apply_short_borrow_cost(returns, shorts, annual_rate=0.08)
    expected_first = 0.01 - 0.5 * 0.08 / 252.0
    assert float(adjusted.iloc[0]) == pytest.approx(expected_first)
    assert float(adjusted.iloc[1]) == pytest.approx(0.01)

    with pytest.raises(ValueError, match="same index"):
        apply_short_borrow_cost(returns, pd.Series([-0.5], index=[3]), annual_rate=0.08)
