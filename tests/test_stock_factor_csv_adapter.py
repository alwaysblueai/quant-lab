from __future__ import annotations

import pandas as pd

from scripts.data.adapt_stock_factor_csv import _prepare_raw_prices


def test_prepare_raw_prices_preserves_raw_close_and_qfq_close() -> None:
    raw = pd.DataFrame(
        {
            "trade_date": ["20200102"],
            "ts_code": ["000001.SZ"],
            "open": [10.0],
            "high": [11.0],
            "low": [9.5],
            "close": [10.5],
            "close_qfq": [8.4],
            "pre_close": [10.1],
            "vol": [1000.0],
            "amount": [10500.0],
        }
    )

    prices = _prepare_raw_prices(raw)

    assert list(prices.columns) == [
        "date",
        "asset",
        "open",
        "high",
        "low",
        "close",
        "close_qfq",
        "pre_close",
        "volume",
        "amount",
    ]
    assert prices.loc[0, "close"] == 10.5
    assert prices.loc[0, "close_qfq"] == 8.4
