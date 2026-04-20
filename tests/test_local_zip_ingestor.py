from __future__ import annotations

import zipfile
from io import BytesIO
from pathlib import Path

import pandas as pd

from alpha_lab.data_store.catalog import DataCatalog, SliceSpec
from alpha_lab.data_store.local_zip import LocalZipAshareDailyIngestor


def _build_nested_zip(path: Path) -> Path:
    inner_buffer = BytesIO()
    with zipfile.ZipFile(inner_buffer, "w", compression=zipfile.ZIP_DEFLATED) as inner:
        inner.writestr(
            "stocks/000001_SZ.csv",
            "\n".join(
                [
                    "ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount,adj_factor,adj_close,adj_open,adj_high,adj_low",
                    "000001.SZ,20240102,10,11,9,10,9.5,0.5,5.0,1000,10000,,10,10,11,9",
                    "000001.SZ,20240103,11,12,10,12,10,2.0,20.0,1200,14400,2.0,12,11,12,10",
                ]
            ),
        )
        inner.writestr(
            "stocks/600000_SH.csv",
            "\n".join(
                [
                    "ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount,adj_factor,adj_close,adj_open,adj_high,adj_low",
                    "600000.SH,20240102,20,21,19,20,19.5,0.5,2.56,2000,40000,1.0,10,10,10.5,9.5",
                    "600000.SH,20240103,21,22,20,22,20,2.0,10.0,2100,46200,2.0,22,21,22,20",
                ]
            ),
        )
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as outer:
        outer.writestr("导出/stocks.zip", inner_buffer.getvalue())
    return path


def test_local_zip_ingestor_writes_asset_partitioned_tables_and_exports_slice(
    tmp_path: Path,
    monkeypatch,
) -> None:
    data_root = tmp_path / "warehouse"
    monkeypatch.setenv("ALPHA_LAB_DATA_ROOT", str(data_root))
    catalog = DataCatalog()
    ingestor = LocalZipAshareDailyIngestor(catalog)
    zip_path = _build_nested_zip(tmp_path / "ashare_daily.zip")

    result = ingestor.ingest_daily_zip(zip_path=zip_path)

    assert result.asset_count == 2
    assert result.row_counts["daily_bars"] == 4
    assert result.row_counts["adj_factor"] == 4
    assert (catalog.table_root("daily_bars") / "asset=000001.SZ" / "part-00000.parquet").exists()
    assert (catalog.table_root("adj_factor") / "asset=600000.SH" / "part-00000.parquet").exists()

    daily_bars = catalog.load_table("daily_bars")
    assert sorted(daily_bars["asset"].unique().tolist()) == ["000001.SZ", "600000.SH"]

    export_result = catalog.export_case_inputs(
        slice_spec=SliceSpec(
            start_date="2024-01-02",
            end_date="2024-01-03",
            adjustment="qfq",
            universe_name="all_ashare",
        ),
        output_dir=tmp_path / "slice",
    )
    prices = pd.read_csv(export_result.output_paths["prices"])
    assert len(prices) == 4
    assert sorted(prices["asset"].unique().tolist()) == ["000001.SZ", "600000.SH"]
    assert {"open", "high", "low", "close", "volume", "amount"} <= set(prices.columns)

    # qfq path should use adj_factor from the asset partitions.
    sh_rows = prices[prices["asset"] == "600000.SH"].sort_values("date")
    assert sh_rows["close"].iloc[0] == 10.0
    assert sh_rows["close"].iloc[1] == 22.0


def test_local_zip_ingestor_registers_trade_calendar_and_instruments(
    tmp_path: Path,
    monkeypatch,
) -> None:
    data_root = tmp_path / "warehouse"
    monkeypatch.setenv("ALPHA_LAB_DATA_ROOT", str(data_root))
    catalog = DataCatalog()
    ingestor = LocalZipAshareDailyIngestor(catalog)
    zip_path = _build_nested_zip(tmp_path / "ashare_daily.zip")

    ingestor.ingest_daily_zip(zip_path=zip_path)

    calendar = catalog.load_table("trade_calendar").sort_values("date")
    instruments = catalog.load_table("instruments").sort_values("asset")

    assert calendar["date"].tolist() == ["2024-01-02", "2024-01-03"]
    assert calendar["is_open"].tolist() == [1, 1]
    assert instruments["asset"].tolist() == ["000001.SZ", "600000.SH"]
    assert instruments["list_date"].tolist() == ["2024-01-02", "2024-01-02"]


def test_local_zip_ingestor_filters_requested_date_window(
    tmp_path: Path,
    monkeypatch,
) -> None:
    data_root = tmp_path / "warehouse"
    monkeypatch.setenv("ALPHA_LAB_DATA_ROOT", str(data_root))
    catalog = DataCatalog()
    ingestor = LocalZipAshareDailyIngestor(catalog)
    zip_path = _build_nested_zip(tmp_path / "ashare_daily.zip")

    result = ingestor.ingest_daily_zip(
        zip_path=zip_path,
        start_date="2024-01-03",
        end_date="2024-01-03",
    )

    assert result.row_counts["daily_bars"] == 2
    assert result.date_range == {
        "start_date": "2024-01-03",
        "end_date": "2024-01-03",
    }
    daily_bars = catalog.load_table("daily_bars").sort_values(["asset", "date"])
    assert daily_bars["date"].tolist() == ["2024-01-03", "2024-01-03"]


def test_local_zip_organize_daily_storage_rewrites_year_month_partitions(
    tmp_path: Path,
    monkeypatch,
) -> None:
    data_root = tmp_path / "warehouse"
    monkeypatch.setenv("ALPHA_LAB_DATA_ROOT", str(data_root))
    catalog = DataCatalog()
    ingestor = LocalZipAshareDailyIngestor(catalog)
    zip_path = _build_nested_zip(tmp_path / "ashare_daily.zip")

    ingestor.ingest_daily_zip(zip_path=zip_path)
    result = ingestor.organize_daily_storage()

    assert "daily_bars" in result.rewritten_tables
    assert (
        catalog.table_root("daily_bars") / "year=2024" / "month=01" / "part-00000.parquet"
    ).exists()
    assert not any(catalog.table_root("daily_bars").glob("asset=*"))
    daily_bars = (
        catalog.load_table("daily_bars").sort_values(["asset", "date"]).reset_index(drop=True)
    )
    assert len(daily_bars) == 4
    assert daily_bars["asset"].tolist() == ["000001.SZ", "000001.SZ", "600000.SH", "600000.SH"]
