from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from alpha_lab.data_store.catalog import DataCatalog, SliceSpec


def test_data_catalog_exports_case_inputs_from_canonical_tables(
    tmp_path: Path,
    monkeypatch,
) -> None:
    data_root = tmp_path / "warehouse"
    monkeypatch.setenv("ALPHA_LAB_DATA_ROOT", str(data_root))
    catalog = DataCatalog()
    catalog.ensure_layout()

    daily_bars = pd.DataFrame(
        [
            {
                "date": "2024-01-02",
                "asset": "000001.SZ",
                "open": 9.9,
                "high": 10.1,
                "low": 9.8,
                "close": 10.0,
                "pre_close": 9.8,
                "volume": 1.0,
                "amount": 10.0,
                "vwap": 100.0,
                "turnover_rate": 1.0,
                "up_limit": 11.0,
                "down_limit": 9.0,
                "is_limit_up": 0,
                "is_limit_down": 0,
            },
            {
                "date": "2024-01-03",
                "asset": "000001.SZ",
                "open": 10.0,
                "high": 10.3,
                "low": 9.9,
                "close": 10.2,
                "pre_close": 10.0,
                "volume": 1.1,
                "amount": 11.22,
                "vwap": 102.0,
                "turnover_rate": 1.1,
                "up_limit": 11.22,
                "down_limit": 9.18,
                "is_limit_up": 0,
                "is_limit_down": 0,
            },
            {
                "date": "2024-01-02",
                "asset": "000002.SZ",
                "open": 19.8,
                "high": 20.2,
                "low": 19.7,
                "close": 20.0,
                "pre_close": 19.8,
                "volume": 2.0,
                "amount": 40.0,
                "vwap": 200.0,
                "turnover_rate": 2.0,
                "up_limit": 22.0,
                "down_limit": 18.0,
                "is_limit_up": 0,
                "is_limit_down": 0,
            },
            {
                "date": "2024-01-03",
                "asset": "000002.SZ",
                "open": 20.0,
                "high": 20.3,
                "low": 19.9,
                "close": 20.2,
                "pre_close": 20.0,
                "volume": 2.1,
                "amount": 42.42,
                "vwap": 202.0,
                "turnover_rate": 2.1,
                "up_limit": 22.22,
                "down_limit": 18.18,
                "is_limit_up": 0,
                "is_limit_down": 0,
            },
        ]
    )
    daily_basic = pd.DataFrame(
        [
            {"date": "2024-01-02", "asset": "000001.SZ", "pb": 2.0},
            {"date": "2024-01-03", "asset": "000001.SZ", "pb": 2.5},
            {"date": "2024-01-02", "asset": "000002.SZ", "pb": 4.0},
            {"date": "2024-01-03", "asset": "000002.SZ", "pb": 5.0},
        ]
    )
    adj_factor = pd.DataFrame(
        [
            {"date": "2024-01-02", "asset": "000001.SZ", "adj_factor": 1.0},
            {"date": "2024-01-03", "asset": "000001.SZ", "adj_factor": 2.0},
            {"date": "2024-01-02", "asset": "000002.SZ", "adj_factor": 1.0},
            {"date": "2024-01-03", "asset": "000002.SZ", "adj_factor": 1.5},
        ]
    )
    financial_indicator = pd.DataFrame(
        [
            {
                "asset": "000001.SZ",
                "ann_date": "2024-01-02",
                "end_date": "2023-12-31",
                "roe_value": 10.0,
                "roe_source_column": "roe_ttm",
            },
            {
                "asset": "000002.SZ",
                "ann_date": "2024-01-03",
                "end_date": "2023-12-31",
                "roe_value": 20.0,
                "roe_source_column": "roe_ttm",
            },
        ]
    )
    balance_sheet = pd.DataFrame(
        [
            {
                "asset": "000001.SZ",
                "ann_date": "2024-01-02",
                "end_date": "2023-12-31",
                "goodwill_balance": 1.0,
            },
            {
                "asset": "000002.SZ",
                "ann_date": "2024-01-03",
                "end_date": "2023-12-31",
                "goodwill_balance": 2.0,
            },
        ]
    )
    income_statement = pd.DataFrame(
        [
            {
                "asset": "000001.SZ",
                "ann_date": "2024-01-02",
                "end_date": "2023-12-31",
                "operating_revenue_ttm": 100.0,
                "operating_cost_ttm": 60.0,
                "rd_expense": 5.0,
            },
            {
                "asset": "000002.SZ",
                "ann_date": "2024-01-03",
                "end_date": "2023-12-31",
                "operating_revenue_ttm": 120.0,
                "operating_cost_ttm": 70.0,
                "rd_expense": 6.0,
            },
        ]
    )
    cash_flow_statement = pd.DataFrame(
        [
            {
                "asset": "000001.SZ",
                "ann_date": "2024-01-02",
                "end_date": "2023-12-31",
                "operating_cash_flow_ttm": 8.0,
            },
            {
                "asset": "000002.SZ",
                "ann_date": "2024-01-03",
                "end_date": "2023-12-31",
                "operating_cash_flow_ttm": 9.0,
            },
        ]
    )
    asset_status = pd.DataFrame(
        [
            {"date": "2024-01-02", "asset": "000001.SZ", "is_suspended": 0, "is_st": 0},
            {"date": "2024-01-03", "asset": "000001.SZ", "is_suspended": 0, "is_st": 1},
            {"date": "2024-01-02", "asset": "000002.SZ", "is_suspended": 0, "is_st": 0},
            {"date": "2024-01-03", "asset": "000002.SZ", "is_suspended": 0, "is_st": 0},
        ]
    )
    index_membership = pd.DataFrame(
        [
            {"date": "2024-01-02", "index_code": "000300.SH", "asset": "000001.SZ", "weight": 1.0},
            {"date": "2024-01-02", "index_code": "000905.SH", "asset": "000002.SZ", "weight": 1.0},
        ]
    )

    catalog.upsert_table(
        "daily_bars", daily_bars, key_cols=("date", "asset"), partition_column="date"
    )
    catalog.upsert_table(
        "adj_factor", adj_factor, key_cols=("date", "asset"), partition_column="date"
    )
    catalog.upsert_table(
        "daily_basic", daily_basic, key_cols=("date", "asset"), partition_column="date"
    )
    catalog.upsert_table(
        "asset_status", asset_status, key_cols=("date", "asset"), partition_column="date"
    )
    catalog.upsert_table(
        "index_membership",
        index_membership,
        key_cols=("date", "index_code", "asset"),
        partition_column="date",
    )
    catalog.upsert_table(
        "financial_indicator",
        financial_indicator,
        key_cols=("asset", "ann_date", "end_date"),
        partition_column="ann_date",
    )
    catalog.upsert_table(
        "balance_sheet",
        balance_sheet,
        key_cols=("asset", "ann_date", "end_date"),
        partition_column="ann_date",
    )
    catalog.upsert_table(
        "income_statement",
        income_statement,
        key_cols=("asset", "ann_date", "end_date"),
        partition_column="ann_date",
    )
    catalog.upsert_table(
        "cash_flow_statement",
        cash_flow_statement,
        key_cols=("asset", "ann_date", "end_date"),
        partition_column="ann_date",
    )
    version = catalog.write_dataset_version(
        dataset_name=DataCatalog.CORE_DATASET_NAME,
        table_names=(
            "daily_bars",
            "adj_factor",
            "daily_basic",
            "asset_status",
            "index_membership",
            "financial_indicator",
            "balance_sheet",
            "income_statement",
            "cash_flow_statement",
        ),
        raw_snapshot_id="snapshot_x",
        notes={"source": "test"},
    )

    export_result = catalog.export_case_inputs(
        slice_spec=SliceSpec(
            start_date="2024-01-02",
            end_date="2024-01-03",
            factors=("bp", "roe_ttm"),
            adjustment="qfq",
        ),
        output_dir=tmp_path / "slice",
    )

    assert export_result.dataset_version_id == version.version_id
    prices = pd.read_csv(export_result.output_paths["prices"])
    bp = pd.read_csv(export_result.output_paths["bp"])
    roe = pd.read_csv(export_result.output_paths["roe_ttm"])
    universe = pd.read_csv(export_result.output_paths["universe"])
    exported_status = pd.read_csv(export_result.output_paths["asset_status"])
    exported_membership = pd.read_csv(export_result.output_paths["index_membership"])

    assert len(prices) == 4
    assert len(universe) == 4
    assert len(exported_status) == 4
    assert len(exported_membership) == 2
    assert bp["factor"].eq("bp").all()
    assert roe["factor"].eq("roe_ttm").all()
    assert {"open", "high", "low", "vwap", "turnover_rate", "is_st", "is_hs300", "is_zz500"} <= set(
        prices.columns
    )
    assert prices.loc[
        (prices["date"] == "2024-01-02") & (prices["asset"] == "000001.SZ"), "close"
    ].iloc[0] == pytest.approx(5.0)
    assert prices.loc[
        (prices["date"] == "2024-01-02") & (prices["asset"] == "000001.SZ"), "open"
    ].iloc[0] == pytest.approx(4.95)
    assert prices.loc[
        (prices["date"] == "2024-01-02") & (prices["asset"] == "000002.SZ"), "close"
    ].iloc[0] == pytest.approx(20.0 / 1.5)
    assert (
        bp.loc[(bp["date"] == "2024-01-02") & (bp["asset"] == "000001.SZ"), "value"].iloc[0] == 0.5
    )
    assert (
        roe.loc[(roe["date"] == "2024-01-02") & (roe["asset"] == "000001.SZ"), "value"].iloc[0]
        == 10.0
    )
    assert (
        roe.loc[(roe["date"] == "2024-01-03") & (roe["asset"] == "000002.SZ"), "value"].iloc[0]
        == 20.0
    )
    assert (
        prices.loc[
            (prices["date"] == "2024-01-03") & (prices["asset"] == "000001.SZ"), "is_st"
        ].iloc[0]
        == 1
    )
    assert (
        prices.loc[
            (prices["date"] == "2024-01-03") & (prices["asset"] == "000001.SZ"), "is_hs300"
        ].iloc[0]
        == 1
    )
    assert (
        prices.loc[
            (prices["date"] == "2024-01-03") & (prices["asset"] == "000002.SZ"), "is_zz500"
        ].iloc[0]
        == 1
    )


def test_data_catalog_exports_raw_prices_without_factor_csvs_by_default(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("ALPHA_LAB_DATA_ROOT", str(tmp_path / "warehouse"))
    catalog = DataCatalog()
    catalog.ensure_layout()

    daily_bars = pd.DataFrame(
        [
            {
                "date": "2024-01-02",
                "asset": "000001.SZ",
                "open": 9.9,
                "high": 10.1,
                "low": 9.8,
                "close": 10.0,
                "pre_close": 9.8,
                "volume": 1.0,
                "amount": 10.0,
            },
            {
                "date": "2024-01-03",
                "asset": "000001.SZ",
                "open": 10.0,
                "high": 10.3,
                "low": 9.9,
                "close": 10.2,
                "pre_close": 10.0,
                "volume": 1.1,
                "amount": 11.22,
            },
        ]
    )
    adj_factor = pd.DataFrame(
        [
            {"date": "2024-01-02", "asset": "000001.SZ", "adj_factor": 1.0},
            {"date": "2024-01-03", "asset": "000001.SZ", "adj_factor": 2.0},
        ]
    )
    catalog.upsert_table(
        "daily_bars", daily_bars, key_cols=("date", "asset"), partition_column="date"
    )
    catalog.upsert_table(
        "adj_factor", adj_factor, key_cols=("date", "asset"), partition_column="date"
    )
    catalog.write_dataset_version(
        dataset_name=DataCatalog.CORE_DATASET_NAME,
        table_names=("daily_bars", "adj_factor"),
        raw_snapshot_id="snapshot_x",
        notes={"source": "test"},
    )

    export_result = catalog.export_case_inputs(
        slice_spec=SliceSpec(start_date="2024-01-02", end_date="2024-01-03"),
        output_dir=tmp_path / "slice",
    )

    assert set(export_result.output_paths) == {"prices", "universe"}
    prices = pd.read_csv(export_result.output_paths["prices"])
    assert prices["close"].tolist() == [10.0, 10.2]
    assert {"open", "high", "low", "is_st", "is_hs300"} <= set(prices.columns)


def test_data_catalog_supports_generic_top_liquidity_universes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("ALPHA_LAB_DATA_ROOT", str(tmp_path / "warehouse"))
    catalog = DataCatalog()
    catalog.ensure_layout()

    daily_bars = pd.DataFrame(
        [
            {
                "date": date,
                "asset": asset,
                "open": 10.0,
                "high": 10.2,
                "low": 9.8,
                "close": 10.1,
                "pre_close": 10.0,
                "volume": 100.0,
                "amount": amount,
            }
            for date in ("2024-01-02", "2024-01-03")
            for asset, amount in (
                ("000001.SZ", 1000.0),
                ("000002.SZ", 800.0),
                ("000003.SZ", 500.0),
            )
        ]
    )
    catalog.upsert_table(
        "daily_bars", daily_bars, key_cols=("date", "asset"), partition_column="date"
    )
    catalog.write_dataset_version(
        dataset_name=DataCatalog.CORE_DATASET_NAME,
        table_names=("daily_bars",),
        raw_snapshot_id="snapshot_x",
        notes={"source": "test"},
    )

    export_result = catalog.export_case_inputs(
        slice_spec=SliceSpec(
            start_date="2024-01-02",
            end_date="2024-01-03",
            universe_name="top_liquid_2",
        ),
        output_dir=tmp_path / "slice_top2",
    )

    prices = pd.read_csv(export_result.output_paths["prices"])
    assert sorted(prices["asset"].unique().tolist()) == ["000001.SZ", "000002.SZ"]


def test_data_catalog_partitions_canonical_tables_by_year_and_month(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("ALPHA_LAB_DATA_ROOT", str(tmp_path / "warehouse"))
    catalog = DataCatalog()
    catalog.ensure_layout()

    daily_bars = pd.DataFrame(
        [
            {
                "date": "2024-01-31",
                "asset": "000001.SZ",
                "close": 10.0,
                "volume": 1.0,
                "amount": 10.0,
            },
            {
                "date": "2024-02-01",
                "asset": "000001.SZ",
                "close": 10.2,
                "volume": 1.1,
                "amount": 11.0,
            },
        ]
    )
    catalog.upsert_table(
        "daily_bars", daily_bars, key_cols=("date", "asset"), partition_column="date"
    )

    assert (
        catalog.table_root("daily_bars") / "year=2024" / "month=01" / "part-00000.parquet"
    ).exists()
    assert (
        catalog.table_root("daily_bars") / "year=2024" / "month=02" / "part-00000.parquet"
    ).exists()


def test_data_catalog_exports_csv_and_parquet_and_loads_slice_from_cache(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("ALPHA_LAB_DATA_ROOT", str(tmp_path / "warehouse"))
    catalog = DataCatalog()
    catalog.ensure_layout()

    daily_bars = pd.DataFrame(
        [
            {
                "date": "2024-01-02",
                "asset": "000001.SZ",
                "open": 9.9,
                "high": 10.1,
                "low": 9.8,
                "close": 10.0,
                "pre_close": 9.8,
                "volume": 1.0,
                "amount": 10.0,
            },
            {
                "date": "2024-01-03",
                "asset": "000001.SZ",
                "open": 10.0,
                "high": 10.3,
                "low": 9.9,
                "close": 10.2,
                "pre_close": 10.0,
                "volume": 1.1,
                "amount": 11.22,
            },
        ]
    )
    adj_factor = pd.DataFrame(
        [
            {"date": "2024-01-02", "asset": "000001.SZ", "adj_factor": 1.0},
            {"date": "2024-01-03", "asset": "000001.SZ", "adj_factor": 2.0},
        ]
    )
    catalog.upsert_table(
        "daily_bars", daily_bars, key_cols=("date", "asset"), partition_column="date"
    )
    catalog.upsert_table(
        "adj_factor", adj_factor, key_cols=("date", "asset"), partition_column="date"
    )
    catalog.write_dataset_version(
        dataset_name=DataCatalog.CORE_DATASET_NAME,
        table_names=("daily_bars", "adj_factor"),
        raw_snapshot_id="snapshot_x",
        notes={"source": "test"},
    )

    slice_spec = SliceSpec(start_date="2024-01-02", end_date="2024-01-03")
    export_result = catalog.export_case_inputs(
        slice_spec=slice_spec,
        output_dir=tmp_path / "slice",
        formats=("csv", "parquet"),
    )

    assert export_result.output_paths["prices"].suffix == ".csv"
    assert export_result.format_output_paths["prices"]["csv"].exists()
    assert export_result.format_output_paths["prices"]["parquet"].exists()
    bundle = catalog.load_case_slice(slice_spec=slice_spec, prefer_cache=True)
    assert bundle.cache_dir == export_result.cache_dir
    assert set(bundle.frames) == {"prices", "universe"}
    assert bundle.frames["prices"]["close"].tolist() == [10.0, 10.2]


def test_validate_raw_snapshot_reports_cross_table_coverage_gaps(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("ALPHA_LAB_DATA_ROOT", str(tmp_path / "warehouse"))
    catalog = DataCatalog()
    catalog.ensure_layout()

    snapshot = catalog.write_raw_snapshot(
        vendor="tushare",
        dataset_name="core",
        tables={
            "prices_raw": pd.DataFrame(
                [
                    {"date": "2024-01-02", "asset": "000001.SZ", "close": 10.0},
                    {"date": "2024-01-02", "asset": "000002.SZ", "close": 20.0},
                ]
            ),
            "adj_factor_raw": pd.DataFrame(
                [
                    {"date": "2024-01-02", "asset": "000001.SZ", "adj_factor": 1.0},
                ]
            ),
        },
        request_params={"assets": ["000001.SZ", "000002.SZ"]},
        time_range={"start_date": "2024-01-02", "end_date": "2024-01-02"},
    )

    report = catalog.validate_raw_snapshot(snapshot.snapshot_id)

    assert report.ok is True
    assert any("adj_factor_raw" in issue.message for issue in report.issues)
    assert report.table_stats["cross_table"]["missing_adj_factor_rows"] == 1


def test_data_catalog_validate_core_dataset_reports_missing_tables(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("ALPHA_LAB_DATA_ROOT", str(tmp_path / "warehouse"))
    catalog = DataCatalog()
    catalog.ensure_layout()

    report = catalog.validate_core_dataset()
    assert report.ok is False
    assert report.report_path.exists()
    assert any(issue.table_name == "daily_bars" for issue in report.issues)


def test_data_catalog_validate_core_dataset_checks_minimum_history_coverage(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("ALPHA_LAB_DATA_ROOT", str(tmp_path / "warehouse"))
    catalog = DataCatalog()
    catalog.ensure_layout()

    daily_bars = pd.DataFrame(
        [
            {
                "date": "2024-01-02",
                "asset": "000001.SZ",
                "close": 10.0,
                "volume": 1.0,
                "amount": 10.0,
            },
            {
                "date": "2025-01-02",
                "asset": "000001.SZ",
                "close": 10.2,
                "volume": 1.1,
                "amount": 11.22,
            },
        ]
    )
    catalog.upsert_table(
        "daily_bars", daily_bars, key_cols=("date", "asset"), partition_column="date"
    )

    report = catalog.validate_core_dataset()

    assert any(
        issue.table_name == "daily_bars" and "minimum 3-year requirement" in issue.message
        for issue in report.issues
    )


def test_data_catalog_validate_core_dataset_accepts_open_day_boundary_for_3y_window(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("ALPHA_LAB_DATA_ROOT", str(tmp_path / "warehouse"))
    catalog = DataCatalog()
    catalog.ensure_layout()

    daily_bars = pd.DataFrame(
        [
            {
                "date": "2023-04-03",
                "asset": "000001.SZ",
                "close": 10.0,
                "volume": 1.0,
                "amount": 10.0,
            },
            {
                "date": "2026-04-01",
                "asset": "000001.SZ",
                "close": 10.2,
                "volume": 1.1,
                "amount": 11.22,
            },
        ]
    )
    trade_calendar = pd.DataFrame(
        [
            {"date": "2023-04-03", "exchange": "SSE", "is_open": 1},
            {"date": "2026-04-01", "exchange": "SSE", "is_open": 1},
        ]
    )
    catalog.upsert_table(
        "daily_bars", daily_bars, key_cols=("date", "asset"), partition_column="date"
    )
    catalog.upsert_table(
        "trade_calendar", trade_calendar, key_cols=("date",), partition_column="date"
    )

    report = catalog.validate_core_dataset()

    assert not any(
        issue.table_name == "daily_bars" and "minimum 3-year requirement" in issue.message
        for issue in report.issues
    )


def test_validate_core_dataset_skips_financial_indicator_warning_in_daily_research_only_mode(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("ALPHA_LAB_DATA_ROOT", str(tmp_path / "warehouse"))
    catalog = DataCatalog()
    catalog.ensure_layout()

    daily_bars = pd.DataFrame(
        [
            {
                "date": "2023-04-03",
                "asset": "000001.SZ",
                "close": 10.0,
                "volume": 1.0,
                "amount": 10.0,
            },
            {
                "date": "2026-04-01",
                "asset": "000001.SZ",
                "close": 10.2,
                "volume": 1.1,
                "amount": 11.22,
            },
        ]
    )
    trade_calendar = pd.DataFrame(
        [
            {"date": "2023-04-03", "exchange": "SSE", "is_open": 1},
            {"date": "2026-04-01", "exchange": "SSE", "is_open": 1},
        ]
    )
    catalog.upsert_table(
        "daily_bars", daily_bars, key_cols=("date", "asset"), partition_column="date"
    )
    catalog.upsert_table(
        "trade_calendar", trade_calendar, key_cols=("date",), partition_column="date"
    )
    catalog.write_dataset_version(
        dataset_name=DataCatalog.CORE_DATASET_NAME,
        table_names=("daily_bars", "trade_calendar"),
        raw_snapshot_id="snapshot_daily_only",
        notes={"daily_research_only": True},
    )

    report = catalog.validate_core_dataset()

    assert not any(
        issue.table_name
        in {
            "financial_indicator",
            "balance_sheet",
            "income_statement",
            "cash_flow_statement",
        }
        for issue in report.issues
    )


def test_export_case_inputs_rejects_window_outside_available_coverage(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("ALPHA_LAB_DATA_ROOT", str(tmp_path / "warehouse"))
    catalog = DataCatalog()
    catalog.ensure_layout()

    daily_bars = pd.DataFrame(
        [
            {
                "date": "2024-01-02",
                "asset": "000001.SZ",
                "close": 10.0,
                "volume": 1.0,
                "amount": 10.0,
            },
            {
                "date": "2024-01-03",
                "asset": "000001.SZ",
                "close": 10.2,
                "volume": 1.1,
                "amount": 11.22,
            },
        ]
    )
    catalog.upsert_table(
        "daily_bars", daily_bars, key_cols=("date", "asset"), partition_column="date"
    )

    with pytest.raises(ValueError, match="predates available daily_bars coverage"):
        catalog.export_case_inputs(
            slice_spec=SliceSpec(start_date="2023-01-01", end_date="2024-01-03"),
            output_dir=tmp_path / "slice",
        )


def test_dataset_version_id_tracks_canonical_content_not_snapshot_id(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("ALPHA_LAB_DATA_ROOT", str(tmp_path / "warehouse"))
    catalog = DataCatalog()
    catalog.ensure_layout()

    daily_bars = pd.DataFrame(
        [
            {
                "date": "2024-01-02",
                "asset": "000001.SZ",
                "close": 10.0,
                "volume": 1.0,
                "amount": 10.0,
            },
        ]
    )
    catalog.upsert_table(
        "daily_bars", daily_bars, key_cols=("date", "asset"), partition_column="date"
    )

    version_a = catalog.write_dataset_version(
        dataset_name=DataCatalog.CORE_DATASET_NAME,
        table_names=("daily_bars",),
        raw_snapshot_id="snapshot_a",
        notes={"start_date": "2024-01-01"},
    )
    version_b = catalog.write_dataset_version(
        dataset_name=DataCatalog.CORE_DATASET_NAME,
        table_names=("daily_bars",),
        raw_snapshot_id="snapshot_b",
        notes={"start_date": "2024-02-01"},
    )

    assert version_a.version_id == version_b.version_id


def test_data_catalog_query_sql_reads_duckdb_views(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("ALPHA_LAB_DATA_ROOT", str(tmp_path / "warehouse"))
    catalog = DataCatalog()
    catalog.ensure_layout()
    daily_bars = pd.DataFrame(
        [
            {
                "date": "2024-01-02",
                "asset": "000001.SZ",
                "close": 10.0,
                "volume": 1.0,
                "amount": 10.0,
            },
            {
                "date": "2024-01-03",
                "asset": "000001.SZ",
                "close": 10.2,
                "volume": 1.1,
                "amount": 11.22,
            },
        ]
    )
    catalog.upsert_table(
        "daily_bars", daily_bars, key_cols=("date", "asset"), partition_column="date"
    )

    result = catalog.query_sql("select count(*) as n_rows from daily_bars")

    assert result.to_dict(orient="records") == [{"n_rows": 2}]


def test_data_catalog_query_sql_rejects_non_read_only(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("ALPHA_LAB_DATA_ROOT", str(tmp_path / "warehouse"))
    catalog = DataCatalog()
    catalog.ensure_layout()

    with pytest.raises(ValueError, match="Only read-only SQL is allowed|Forbidden SQL keyword"):
        catalog.query_sql("drop table daily_bars")


def test_latest_date_prefers_partition_date_roof(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ALPHA_LAB_DATA_ROOT", str(tmp_path / "warehouse"))
    catalog = DataCatalog()
    catalog.ensure_layout()

    daily_bars = pd.DataFrame(
        [
            {
                "date": "2024-01-31",
                "asset": "000001.SZ",
                "close": 10.0,
                "volume": 1.0,
                "amount": 10.0,
            },
            {
                "date": "2024-02-01",
                "asset": "000001.SZ",
                "close": 11.0,
                "volume": 2.0,
                "amount": 12.0,
            },
            {
                "date": "2024-03-15",
                "asset": "000001.SZ",
                "close": 13.0,
                "volume": 3.0,
                "amount": 13.0,
            },
        ]
    )
    catalog.upsert_table(
        "daily_bars", daily_bars, key_cols=("date", "asset"), partition_column="date"
    )

    latest = catalog.latest_date("daily_bars")
    assert latest == "2024-03-15"


def test_latest_date_supports_non_date_partitioned_column(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("ALPHA_LAB_DATA_ROOT", str(tmp_path / "warehouse"))
    catalog = DataCatalog()
    catalog.ensure_layout()

    financial_indicator = pd.DataFrame(
        [
            {
                "asset": "000001.SZ",
                "ann_date": "2024-01-31",
                "end_date": "2023-12-31",
                "roe_value": 10.0,
                "roe_source_column": "roe_ttm",
            },
            {
                "asset": "000002.SZ",
                "ann_date": "2024-04-30",
                "end_date": "2024-03-31",
                "roe_value": 12.0,
                "roe_source_column": "roe_ttm",
            },
        ]
    )
    catalog.upsert_table(
        "financial_indicator",
        financial_indicator,
        key_cols=("asset", "ann_date", "end_date"),
        partition_column="ann_date",
    )

    latest = catalog.latest_date("financial_indicator", date_field="ann_date")
    assert latest == "2024-04-30"


def test_latest_date_falls_back_to_full_scan_when_partition_layout_missing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("ALPHA_LAB_DATA_ROOT", str(tmp_path / "warehouse"))
    catalog = DataCatalog()
    catalog.ensure_layout()

    legacy_root = catalog.canonical_root / "legacy"
    legacy_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"date": "2024-01-05", "asset": "000001.SZ", "close": 10.0},
            {"date": "2024-01-07", "asset": "000002.SZ", "close": 11.0},
        ]
    ).to_parquet(legacy_root / "data.parquet", index=False)

    assert catalog.latest_date("legacy", date_field="date") == "2024-01-07"
