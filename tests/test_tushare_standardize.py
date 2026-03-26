from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from alpha_lab.data_sources.tushare_standardize import (
    build_standardized_tushare_tables,
    standardize_asset_metadata,
    standardize_daily_basic,
    standardize_prices,
    standardize_trade_calendar,
)


def test_standardize_prices_normalizes_keys_and_volume_amount() -> None:
    daily = pd.DataFrame(
        {
            "ts_code": ["000001.sz"],
            "trade_date": ["20240102"],
            "close": [10.0],
            "open": [9.8],
            "high": [10.1],
            "low": [9.7],
            "pre_close": [9.9],
            "vol": [100.0],
            "amount": [50.0],
        }
    )
    daily_basic = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "trade_date": ["20240102"],
            "pb": [1.2],
            "pe_ttm": [10.5],
            "total_mv": [1000.0],
            "circ_mv": [900.0],
        }
    )
    out = standardize_prices(daily, daily_basic=daily_basic, adj_factor=None)
    row = out.iloc[0]
    assert str(row["asset"]) == "000001.SZ"
    assert str(row["date"].date()) == "2024-01-02"
    assert float(row["volume"]) == 10000.0
    assert float(row["dollar_volume_yuan"]) == 50000.0
    assert float(row["total_mv_yuan"]) == 1000.0 * 10_000.0


def test_standardize_trade_calendar_parses_dates_and_bools() -> None:
    trade_cal = pd.DataFrame(
        {
            "exchange": ["SSE"],
            "cal_date": ["20240102"],
            "is_open": ["1"],
            "pretrade_date": ["20231229"],
        }
    )
    out = standardize_trade_calendar(trade_cal)
    assert bool(out["is_open"].iloc[0]) is True
    assert str(out["pretrade_date"].iloc[0].date()) == "2023-12-29"


def test_standardize_asset_metadata_infers_st_flag() -> None:
    stock_basic = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "name": ["ST Sample"],
            "industry": ["Bank"],
            "market": ["main"],
            "exchange": ["SZSE"],
            "list_status": ["L"],
            "list_date": ["20200101"],
            "delist_date": [None],
            "is_hs": ["N"],
        }
    )
    out = standardize_asset_metadata(stock_basic)
    assert bool(out["is_st"].iloc[0]) is True


def test_standardize_daily_basic_missing_required_columns_raises() -> None:
    with pytest.raises(ValueError, match="required columns"):
        standardize_daily_basic(pd.DataFrame({"ts_code": ["000001.SZ"]}))


def test_build_standardized_tables_accepts_vip_financial_snapshot_name(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "raw_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "trade_date": ["20240102"],
            "close": [10.0],
            "high": [10.1],
            "low": [9.9],
            "vol": [100.0],
            "amount": [50.0],
        }
    ).to_csv(snapshot_dir / "daily.csv", index=False)
    pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "trade_date": ["20240102"],
            "pb": [1.2],
            "pe_ttm": [10.5],
            "total_mv": [1000.0],
            "circ_mv": [900.0],
        }
    ).to_csv(snapshot_dir / "daily_basic.csv", index=False)
    pd.DataFrame(
        {
            "exchange": ["SSE"],
            "cal_date": ["20240102"],
            "is_open": [1],
            "pretrade_date": ["20231229"],
        }
    ).to_csv(snapshot_dir / "trade_cal.csv", index=False)
    pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "name": ["PingAn"],
            "industry": ["Bank"],
            "market": ["main"],
            "exchange": ["SZSE"],
            "list_status": ["L"],
            "list_date": ["19910403"],
            "delist_date": [None],
            "is_hs": ["N"],
        }
    ).to_csv(snapshot_dir / "stock_basic.csv", index=False)
    pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "ann_date": ["20240220"],
            "end_date": ["20231231"],
            "roe_dt": [9.1],
            "roe": [9.2],
            "roa": [0.8],
            "grossprofit_margin": [30.0],
            "netprofit_margin": [10.0],
        }
    ).to_csv(snapshot_dir / "fina_indicator_vip.csv", index=False)

    tables = build_standardized_tushare_tables(
        snapshot_dir,
        output_dir=tmp_path / "standardized",
    )
    assert not tables.financial_indicators.empty
    assert "000001.SZ" in set(tables.financial_indicators["asset"].astype(str))
