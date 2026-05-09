from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from alpha_lab.intraday._formulas import (
    bipower_variation,
    log_returns,
    realized_variance,
    sampled_log_returns,
)
from alpha_lab.intraday.features import (
    BATCH1_FEATURE_COLUMNS,
    BATCH2_FEATURE_COLUMNS,
    BATCH3_FEATURE_COLUMNS,
    BATCH4_FEATURE_COLUMNS,
    compute_batch1_feature_frame,
    compute_batch1_features,
    compute_batch2_feature_frame,
    compute_batch2_features,
    compute_batch3_feature_frame,
    compute_batch3_features,
    compute_batch4_feature_frame,
    compute_batch4_features,
    compute_intraday_moments,
    compute_microfreq_timeseries,
    compute_microstructure,
    compute_pv_correlation,
    compute_realized_volatility,
    compute_return_decomposition,
    compute_volume_timing,
    compute_vwap_deviation,
)


def _row(ts: str, open_: float, close: float) -> dict[str, object]:
    high = max(open_, close) + 0.05
    low = min(open_, close) - 0.05
    return {
        "date": "2024-01-02",
        "asset": "000001.SZ",
        "datetime": pd.Timestamp(ts),
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": 1000.0,
        "amount": close * 1000.0,
    }


def _key_time_day() -> pd.DataFrame:
    rows = [
        _row("2024-01-02 09:30:00", 100.0, 101.0),
        _row("2024-01-02 09:35:00", 101.0, 102.0),
        _row("2024-01-02 10:00:00", 102.0, 103.0),
        _row("2024-01-02 11:30:00", 103.0, 104.0),
        _row("2024-01-02 13:00:00", 104.0, 105.0),
        _row("2024-01-02 14:30:00", 105.0, 106.0),
        _row("2024-01-02 14:55:00", 106.0, 107.0),
        _row("2024-01-02 15:00:00", 107.0, 108.0),
    ]
    return pd.DataFrame(rows)


def _dense_day() -> pd.DataFrame:
    rows = []
    ts = datetime(2024, 1, 2, 9, 30)
    for i in range(40):
        open_ = 100.0 + 0.2 * i
        close = 100.0 + 0.2 * i + ((-1) ** i) * 0.03
        rows.append(_row(ts.strftime("%Y-%m-%d %H:%M:%S"), open_, close))
        ts += timedelta(minutes=1)
    for raw_ts, open_, close in [
        ("2024-01-02 11:30:00", 108.0, 108.3),
        ("2024-01-02 13:00:00", 108.3, 108.1),
        ("2024-01-02 14:30:00", 109.0, 109.4),
        ("2024-01-02 14:55:00", 109.4, 109.8),
        ("2024-01-02 15:00:00", 109.8, 110.2),
    ]:
        rows.append(_row(raw_ts, open_, close))
    return pd.DataFrame(rows)


def _timing_row(ts: str, amount: float, volume: float = 100.0) -> dict[str, object]:
    price = 100.0 + amount / 1000.0
    return {
        "date": "2024-01-02",
        "asset": "000001.SZ",
        "datetime": pd.Timestamp(ts),
        "open": price - 0.02,
        "high": price + 0.05,
        "low": price - 0.05,
        "close": price,
        "volume": volume,
        "amount": amount,
    }


def _timing_day() -> pd.DataFrame:
    rows = [
        _timing_row("2024-01-02 09:30:00", 10.0),
        _timing_row("2024-01-02 09:31:00", 20.0),
        _timing_row("2024-01-02 09:59:00", 30.0),
        _timing_row("2024-01-02 10:00:00", 40.0),
        _timing_row("2024-01-02 11:00:00", 50.0),
        _timing_row("2024-01-02 11:29:00", 60.0),
        _timing_row("2024-01-02 11:30:00", 70.0),
        _timing_row("2024-01-02 13:00:00", 80.0),
        _timing_row("2024-01-02 13:29:00", 90.0),
        _timing_row("2024-01-02 13:30:00", 100.0),
        _timing_row("2024-01-02 14:30:00", 110.0),
        _timing_row("2024-01-02 15:00:00", 120.0),
    ]
    return pd.DataFrame(rows)


def _vwap_day() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": "2024-01-02",
                "asset": "000001.SZ",
                "datetime": pd.Timestamp("2024-01-02 09:30:00"),
                "open": 99.0,
                "high": 100.0,
                "low": 97.0,
                "close": 98.0,
                "volume": 1.0,
                "amount": 98.0,
            },
            {
                "date": "2024-01-02",
                "asset": "000001.SZ",
                "datetime": pd.Timestamp("2024-01-02 10:00:00"),
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 1.0,
                "amount": 100.0,
            },
            {
                "date": "2024-01-02",
                "asset": "000001.SZ",
                "datetime": pd.Timestamp("2024-01-02 14:30:00"),
                "open": 101.0,
                "high": 103.0,
                "low": 100.0,
                "close": 102.0,
                "volume": 1.0,
                "amount": 102.0,
            },
            {
                "date": "2024-01-02",
                "asset": "000001.SZ",
                "datetime": pd.Timestamp("2024-01-02 15:00:00"),
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 1.0,
                "amount": 100.0,
            },
        ]
    )


def test_return_decomposition_uses_documented_time_boundaries() -> None:
    result = compute_return_decomposition(_key_time_day())

    assert result["ret_intraday"] == 108.0 / 100.0 - 1.0
    assert result["ret_morning"] == 104.0 / 100.0 - 1.0
    assert result["ret_afternoon"] == 108.0 / 104.0 - 1.0
    assert result["ret_open5"] == 102.0 / 100.0 - 1.0
    assert result["ret_close5"] == 108.0 / 107.0 - 1.0
    assert result["ret_first30"] == 103.0 / 100.0 - 1.0
    assert result["ret_last30"] == 108.0 / 106.0 - 1.0
    assert result["ret_mid"] == 105.0 / 103.0 - 1.0


def test_realized_volatility_matches_formula_helpers() -> None:
    day = _dense_day()
    result = compute_realized_volatility(day)

    returns_1m = log_returns(day["close"]).dropna()
    returns_5m = sampled_log_returns(day, 5)
    returns_15m = sampled_log_returns(day, 15)
    rv_5m = realized_variance(returns_5m)
    bv_5m = bipower_variation(returns_5m)

    assert np.isclose(result["rv_1m"], realized_variance(returns_1m))
    assert np.isclose(result["rv_5m"], rv_5m)
    assert np.isclose(result["rv_15m"], realized_variance(returns_15m))
    assert np.isclose(result["bv_5m"], bv_5m)
    assert np.isclose(result["jump_5m"], max(rv_5m - bv_5m, 0.0))
    assert np.isclose(result["rv_pos_5m"] + result["rv_neg_5m"], result["rv_5m"])
    assert -1.0 <= result["signed_jump"] <= 1.0
    assert result["rv_morning"] > 0
    assert result["rv_afternoon"] > 0


def test_intraday_moments_require_enough_nonzero_returns() -> None:
    flat_day = _dense_day()
    flat_day["close"] = 100.0
    flat_result = compute_intraday_moments(flat_day)
    assert np.isnan(flat_result["intraday_skew_1m"])
    assert np.isnan(flat_result["intraday_kurt_1m"])
    assert np.isnan(flat_result["intraday_skew_5m"])
    assert np.isnan(flat_result["intraday_kurt_5m"])

    active_result = compute_intraday_moments(_dense_day())
    assert np.isfinite(active_result["intraday_skew_1m"])
    assert np.isfinite(active_result["intraday_kurt_1m"])
    assert np.isfinite(active_result["intraday_skew_5m"])
    assert np.isfinite(active_result["intraday_kurt_5m"])


def test_batch1_feature_contract_has_all_columns() -> None:
    result = compute_batch1_features(_dense_day())

    assert list(result) == BATCH1_FEATURE_COLUMNS
    assert len(BATCH1_FEATURE_COLUMNS) == 22


def test_vectorized_batch1_matches_per_day_formula() -> None:
    day1 = _dense_day()
    day2 = _dense_day()
    day2["date"] = "2024-01-03"
    day2["datetime"] = day2["datetime"] + pd.Timedelta(days=1)
    day2["close"] = day2["close"] * 1.01
    day2["open"] = day2["open"] * 1.01
    day2["high"] = day2[["open", "close"]].max(axis=1) + 0.05
    day2["low"] = day2[["open", "close"]].min(axis=1) - 0.05

    panel = pd.concat([day2, day1], ignore_index=True)
    vectorized = compute_batch1_feature_frame(panel).sort_values(["date", "asset"]).reset_index(
        drop=True
    )

    for idx, day in enumerate([day1, day2]):
        expected = compute_batch1_features(day)
        for column in BATCH1_FEATURE_COLUMNS:
            actual_value = vectorized.loc[idx, column]
            expected_value = expected[column]
            if np.isnan(expected_value):
                assert np.isnan(actual_value), column
            else:
                assert np.isclose(actual_value, expected_value), column


def test_amount_shares_normal() -> None:
    result = compute_volume_timing(_timing_day())
    total = 780.0

    assert np.isclose(result["amount_share_open30"], 60.0 / total)
    assert np.isclose(result["amount_share_pre_lunch30"], 180.0 / total)
    assert np.isclose(result["amount_share_post_lunch30"], 270.0 / total)
    assert np.isclose(result["amount_share_close30"], 230.0 / total)
    assert np.isclose(result["amount_share_morning"], 280.0 / total)
    assert np.isclose(result["amount_share_afternoon"], 500.0 / total)


def test_amount_shares_sum_invariant() -> None:
    result = compute_volume_timing(_timing_day())
    segment_sum = (
        result["amount_share_open30"]
        + result["amount_share_pre_lunch30"]
        + result["amount_share_post_lunch30"]
        + result["amount_share_close30"]
    )

    assert np.isclose(result["amount_share_morning"] + result["amount_share_afternoon"], 1.0)
    assert segment_sum <= 1.0


def test_amount_share_zero_total() -> None:
    day = _timing_day()
    day["amount"] = 0.0
    result = compute_volume_timing(day)

    for column in BATCH2_FEATURE_COLUMNS:
        if column.startswith("amount_") or column == "minutes_to_50pct_amount":
            assert np.isnan(result[column]), column


def test_amount_hhi_bounds() -> None:
    day = _timing_day()
    result = compute_volume_timing(day)
    active_minutes = (day["amount"] > 0).sum()

    assert 1.0 / active_minutes <= result["amount_hhi"] <= 1.0


def test_amount_top10_share_few_minutes() -> None:
    day = pd.DataFrame(
        [
            _timing_row("2024-01-02 09:30:00", 5.0),
            _timing_row("2024-01-02 09:31:00", 1.0),
            _timing_row("2024-01-02 09:32:00", 4.0),
            _timing_row("2024-01-02 09:33:00", 2.0),
            _timing_row("2024-01-02 09:34:00", 3.0),
        ]
    )
    result = compute_volume_timing(day)

    assert result["amount_top10_share"] == 1.0


def test_minutes_to_50pct_chronological() -> None:
    result = compute_volume_timing(_timing_day())

    assert result["minutes_to_50pct_amount"] == 9.0


def test_volume_kurt_threshold() -> None:
    base_ts = datetime(2024, 1, 2, 9, 30)
    sparse_rows = [
        _timing_row((base_ts + timedelta(minutes=i)).strftime("%Y-%m-%d %H:%M:%S"), 1.0, i + 1)
        for i in range(29)
    ]
    dense_rows = sparse_rows + [
        _timing_row("2024-01-02 10:00:00", 1.0, 30.0),
    ]

    assert np.isnan(compute_volume_timing(pd.DataFrame(sparse_rows))["volume_kurt_1m"])
    assert np.isfinite(compute_volume_timing(pd.DataFrame(dense_rows))["volume_kurt_1m"])


def test_vwap_devs_simple() -> None:
    result = compute_vwap_deviation(_vwap_day())

    assert np.isclose(result["vwap_close_dev"], 0.0)
    assert np.isclose(result["vwap_open_dev"], -0.01)
    assert np.isclose(result["vwap_high_dev"], 0.03)
    assert np.isclose(result["vwap_low_dev"], -0.03)
    assert np.isclose(result["vwap_minute_dispersion"], np.sqrt(2.0) / 100.0)


def test_vwap_dispersion_zero_vwap() -> None:
    day = _vwap_day()
    day["volume"] = 0.0
    day["amount"] = 0.0
    result = compute_vwap_deviation(day)

    for column, value in result.items():
        assert np.isnan(value), column


def _signed_pv_day() -> pd.DataFrame:
    """Synthetic minute panel with controlled up/down moves for Batch 3 tests."""

    base_ts = datetime(2024, 1, 2, 9, 30)
    rows = []
    price = 100.0
    closes = []
    moves = [+0.05, -0.03, +0.04, -0.02, +0.06, -0.05, +0.02, -0.04] * 6
    volumes = [1000.0, 1100.0, 900.0, 1200.0, 950.0, 1050.0, 1150.0, 980.0] * 6
    for i, (m, v) in enumerate(zip(moves, volumes, strict=True)):
        ts = (base_ts + timedelta(minutes=i)).strftime("%Y-%m-%d %H:%M:%S")
        new_price = price + m
        rows.append(
            {
                "date": "2024-01-02",
                "asset": "000001.SZ",
                "datetime": pd.Timestamp(ts),
                "open": price,
                "high": max(price, new_price) + 0.01,
                "low": min(price, new_price) - 0.01,
                "close": new_price,
                "volume": v,
                "amount": new_price * v,
            }
        )
        closes.append(new_price)
        price = new_price
    return pd.DataFrame(rows)


def _limit_day(*, up_limit: float = 11.0, prev_close: float = 10.0) -> pd.DataFrame:
    """A 1-yuan stock that hits up_limit early, holds, then opens, etc."""

    base_ts = datetime(2024, 1, 2, 9, 30)
    rows = []
    sequence = (
        [10.5, 10.8, 11.0, 11.0, 11.0, 11.0, 10.95, 10.92]  # touch and open once
        + [10.94, 10.96, 11.0, 11.0, 10.97, 10.99]
        + [10.5] * 16
    )
    for i, close in enumerate(sequence):
        ts = (base_ts + timedelta(minutes=i)).strftime("%Y-%m-%d %H:%M:%S")
        rows.append(
            {
                "date": "2024-01-02",
                "asset": "000001.SZ",
                "datetime": pd.Timestamp(ts),
                "open": close - 0.01 if i > 0 else 10.4,
                "high": close + 0.01,
                "low": close - 0.02,
                "close": float(close),
                "volume": 1000.0,
                "amount": close * 1000.0,
                "up_limit": up_limit,
                "down_limit": 9.0,
                "prev_close": prev_close,
            }
        )
    return pd.DataFrame(rows)


def test_pv_correlation_signed_amount_imbalance_signs_match() -> None:
    day = _signed_pv_day()
    result = compute_pv_correlation(day, min_count=10)

    # Construction: ~half up moves slightly larger than down → positive imbalance.
    assert -1.0 <= result["signed_amount_imbalance"] <= 1.0
    share_sum = (
        result["pos_amount_share"]
        + result["neg_amount_share"]
        + result["zero_ret_amount_share"]
    )
    assert share_sum <= 1.0 + 1e-9
    assert np.isfinite(result["amihud_intraday"])
    assert np.isfinite(result["corr_ret_volume_1m"])


def test_pv_correlation_zero_amount_returns_nan_imbalance() -> None:
    day = _signed_pv_day()
    day["amount"] = 0.0
    result = compute_pv_correlation(day, min_count=10)

    assert np.isnan(result["signed_amount_imbalance"])
    assert np.isnan(result["pos_amount_share"])
    assert np.isnan(result["neg_amount_share"])
    assert np.isnan(result["amihud_intraday"])


def test_microfreq_features_are_finite_on_dense_day() -> None:
    day = _dense_day()
    result = compute_microfreq_timeseries(day, min_count=10)

    assert np.isfinite(result["ret_autocorr_1m_lag1"])
    assert np.isfinite(result["amount_autocorr_1m_lag1"])
    assert result["avg_gap_between_trades"] >= 1.0
    assert 0.0 <= result["time_at_extremes_share"] <= 1.0
    assert result["acceleration_max"] >= 0.0


def test_microfreq_handles_flat_close() -> None:
    day = _dense_day().copy()
    day["close"] = 100.0
    day["high"] = 100.0
    day["low"] = 100.0
    day["amount"] = 1000.0
    day["volume"] = 100.0
    result = compute_microfreq_timeseries(day, min_count=10)

    assert np.isnan(result["time_at_extremes_share"])  # zero range
    assert (
        np.isnan(result["ret_autocorr_1m_lag1"])
        or result["ret_autocorr_1m_lag1"] == 0.0
    )


def test_batch3_vs_perday_consistency() -> None:
    day1 = _signed_pv_day()
    day2 = _signed_pv_day()
    day2["date"] = "2024-01-03"
    day2["datetime"] = day2["datetime"] + pd.Timedelta(days=1)
    day2["close"] = day2["close"] * 1.02
    day2["amount"] = day2["amount"] * 1.02

    panel = pd.concat([day2, day1], ignore_index=True)
    vectorized = (
        compute_batch3_feature_frame(panel, min_count=10)
        .sort_values(["date", "asset"])
        .reset_index(drop=True)
    )

    for idx, day in enumerate([day1, day2]):
        expected = compute_batch3_features(day)
        for column in BATCH3_FEATURE_COLUMNS:
            actual_value = vectorized.loc[idx, column]
            expected_value = expected[column]
            if np.isnan(expected_value):
                assert np.isnan(actual_value), column
            else:
                assert np.isclose(actual_value, expected_value), column


def test_microstructure_limit_touch_and_open() -> None:
    day = _limit_day()
    result = compute_microstructure(day, min_count=10)

    # Sequence has 4 + 2 = 6 minutes at up_limit, with 2 open transitions.
    assert result["limit_up_touch_count"] == 6
    assert result["limit_up_open_count"] == 2
    assert result["limit_down_touch_count"] == 0
    assert result["minutes_at_high_count"] >= 1
    assert result["minutes_at_low_count"] >= 1


def test_microstructure_sign_flips_and_zscore() -> None:
    day = _signed_pv_day()
    result = compute_microstructure(day, up_limit=None, down_limit=None, prev_close=99.99,
                                    min_count=10)

    assert result["sign_flip_count"] >= 1
    assert np.isfinite(result["max_abs_return_zscore"])
    assert np.isfinite(result["roll_spread_proxy"])


def test_microstructure_gap_fill_handles_zero_gap() -> None:
    day = _signed_pv_day().copy()
    day_open = float(day["open"].iloc[0])
    result = compute_microstructure(day, prev_close=day_open, min_count=10)
    assert np.isnan(result["gap_fill_ratio"])  # no gap → undefined


def test_batch4_vs_perday_consistency() -> None:
    day1 = _limit_day()
    day2 = _limit_day(up_limit=11.0, prev_close=10.0)
    day2["date"] = "2024-01-03"
    day2["datetime"] = day2["datetime"] + pd.Timedelta(days=1)
    day2["close"] = day2["close"] * 1.005

    panel = pd.concat([day2, day1], ignore_index=True)
    vectorized = (
        compute_batch4_feature_frame(panel, min_count=10)
        .sort_values(["date", "asset"])
        .reset_index(drop=True)
    )

    for idx, day in enumerate([day1, day2]):
        expected = compute_batch4_features(
            day,
            up_limit=float(day["up_limit"].iloc[0]),
            down_limit=float(day["down_limit"].iloc[0]),
            prev_close=float(day["prev_close"].iloc[0]),
            min_count=10,
        )
        for column in BATCH4_FEATURE_COLUMNS:
            actual_value = vectorized.loc[idx, column]
            expected_value = expected[column]
            if expected_value is None or (
                isinstance(expected_value, float) and np.isnan(expected_value)
            ):
                assert pd.isna(actual_value), column
            else:
                assert np.isclose(actual_value, expected_value), column


def test_batch3_columns_count() -> None:
    assert len(BATCH3_FEATURE_COLUMNS) == 12


def test_batch4_columns_count() -> None:
    assert len(BATCH4_FEATURE_COLUMNS) == 10


def test_batch2_vs_perday_consistency() -> None:
    day1 = _timing_day()
    day2 = _timing_day()
    day2["date"] = "2024-01-03"
    day2["datetime"] = day2["datetime"] + pd.Timedelta(days=1)
    day2["amount"] = day2["amount"] * 1.5
    day2["volume"] = day2["volume"] * 1.5

    panel = pd.concat([day2, day1], ignore_index=True)
    vectorized = compute_batch2_feature_frame(panel).sort_values(["date", "asset"]).reset_index(
        drop=True
    )

    for idx, day in enumerate([day1, day2]):
        expected = compute_batch2_features(day)
        for column in BATCH2_FEATURE_COLUMNS:
            actual_value = vectorized.loc[idx, column]
            expected_value = expected[column]
            if np.isnan(expected_value):
                assert np.isnan(actual_value), column
            else:
                assert np.isclose(actual_value, expected_value), column
