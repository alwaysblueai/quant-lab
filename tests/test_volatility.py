from __future__ import annotations

import math

import pandas as pd

from alpha_lab.factors.volatility import amplitude, downside_volatility
from alpha_lab.interfaces import FACTOR_OUTPUT_COLUMNS, validate_factor_output


def _make_ohlc_df(
    close_prices: dict[str, list[float]],
    *,
    highs: dict[str, list[float]] | None = None,
    lows: dict[str, list[float]] | None = None,
    start: str = "2024-01-01",
) -> pd.DataFrame:
    dates = pd.date_range(start, periods=max(len(v) for v in close_prices.values()), freq="B")
    rows = []
    for asset, closes in close_prices.items():
        asset_highs = highs[asset] if highs is not None else [price * 1.02 for price in closes]
        asset_lows = lows[asset] if lows is not None else [price * 0.98 for price in closes]
        for i, close in enumerate(closes):
            rows.append(
                {
                    "date": dates[i],
                    "asset": asset,
                    "close": close,
                    "high": asset_highs[i],
                    "low": asset_lows[i],
                }
            )
    return pd.DataFrame(rows)


def test_amplitude_is_negative_mean_intraday_range() -> None:
    closes = [100.0, 102.0, 101.0, 103.0, 104.0]
    highs = [101.0, 103.0, 102.0, 104.0, 105.0]
    lows = [99.0, 100.0, 99.0, 101.0, 102.0]
    result = amplitude(_make_ohlc_df({"A": closes}, highs={"A": highs}, lows={"A": lows}), window=3)

    assert tuple(result.columns) == FACTOR_OUTPUT_COLUMNS
    assert set(result["factor"]) == {"amplitude_3d"}

    amp = pd.Series([(highs[i] - lows[i]) / closes[i - 1] if i > 0 else math.nan for i in range(len(closes))])
    expected = -amp.iloc[1:4].mean()
    assert result["value"].iloc[:3].isna().all()
    assert math.isclose(float(result["value"].iloc[3]), float(expected), rel_tol=1e-12)


def test_amplitude_no_future_data() -> None:
    closes = [100.0, 101.0, 99.0, 102.0, 100.0, 103.0]
    full = amplitude(_make_ohlc_df({"A": closes}), window=3)["value"]
    prefix = amplitude(_make_ohlc_df({"A": closes[:5]}), window=3)["value"]
    assert math.isclose(float(full.iloc[4]), float(prefix.iloc[4]), rel_tol=1e-12)


def test_downside_volatility_is_negative_downside_semivol() -> None:
    closes = [100.0, 98.0, 99.0, 97.0, 96.0]
    result = downside_volatility(_make_ohlc_df({"A": closes}), window=3)

    assert tuple(result.columns) == FACTOR_OUTPUT_COLUMNS
    assert set(result["factor"]) == {"downside_volatility_3d"}

    rets = pd.Series(closes).pct_change()
    window_rets = rets.iloc[1:4]
    negative = window_rets[window_rets < 0]
    expected = -math.sqrt(float((negative.pow(2).sum()) / len(negative)))
    assert result["value"].iloc[:2].isna().all()
    assert not math.isnan(float(result["value"].iloc[2]))
    assert math.isclose(float(result["value"].iloc[3]), float(expected), rel_tol=1e-12)


def test_downside_volatility_returns_nan_when_window_has_no_negative_returns() -> None:
    closes = [100.0, 101.0, 102.0, 103.0, 104.0]
    result = downside_volatility(_make_ohlc_df({"A": closes}), window=3)
    assert result["value"].iloc[3:].isna().all()


def test_volatility_builders_validate_canonical_output() -> None:
    frame = _make_ohlc_df({"A": [100.0, 99.0, 101.0, 98.0, 102.0, 100.0]})
    validate_factor_output(amplitude(frame, window=3))
    validate_factor_output(downside_volatility(frame, window=3))
