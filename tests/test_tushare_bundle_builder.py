from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from alpha_lab.data_sources.tushare_bundle_builder import (
    build_tushare_research_inputs,
    export_canonical_tushare_case_configs,
)


def _write_df(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _standardized_dir(tmp_path: Path) -> Path:
    root = tmp_path / "standardized"
    _write_df(
        root / "prices.csv",
        pd.DataFrame(
            {
                "date": ["2024-01-02", "2024-01-03", "2024-01-02", "2024-01-03"],
                "asset": ["000001.SZ", "000001.SZ", "000002.SZ", "000002.SZ"],
                "close": [10.0, 10.2, 20.0, 19.8],
                "volume": [10000.0, 12000.0, 15000.0, 16000.0],
                "dollar_volume": [500000.0, 510000.0, 900000.0, 920000.0],
            }
        ),
    )
    _write_df(
        root / "asset_metadata.csv",
        pd.DataFrame(
            {
                "asset": ["000001.SZ", "000002.SZ"],
                "listing_date": ["2020-01-01", "2020-01-01"],
                "delist_date": [None, None],
                "is_st": [False, False],
                "industry": ["Bank", "RealEstate"],
                "market": ["main", "main"],
                "exchange": ["SZSE", "SZSE"],
                "name": ["PingAn", "Vanke"],
                "list_status": ["L", "L"],
                "is_hs": ["N", "N"],
            }
        ),
    )
    _write_df(
        root / "trade_calendar.csv",
        pd.DataFrame(
            {
                "date": ["2024-01-02", "2024-01-03"],
                "exchange": ["SSE", "SSE"],
                "is_open": [True, True],
                "pretrade_date": ["2023-12-29", "2024-01-02"],
            }
        ),
    )
    _write_df(
        root / "market_state.csv",
        pd.DataFrame(
            {
                "date": ["2024-01-02", "2024-01-03", "2024-01-02", "2024-01-03"],
                "asset": ["000001.SZ", "000001.SZ", "000002.SZ", "000002.SZ"],
                "is_halted": [False, False, False, False],
                "is_limit_locked": [False, False, False, False],
                "is_st": [False, False, False, False],
            }
        ),
    )
    _write_df(
        root / "daily_fundamentals.csv",
        pd.DataFrame(
            {
                "date": ["2024-01-02", "2024-01-03", "2024-01-02", "2024-01-03"],
                "asset": ["000001.SZ", "000001.SZ", "000002.SZ", "000002.SZ"],
                "pb": [1.0, 1.1, 2.0, 2.1],
                "pe_ttm": [10.0, 10.1, 20.0, 20.1],
                "total_mv_yuan": [1e9, 1.1e9, 2e9, 2.1e9],
                "circ_mv_yuan": [9e8, 1e9, 1.8e9, 1.9e9],
            }
        ),
    )
    _write_df(
        root / "financial_indicators.csv",
        pd.DataFrame(
            {
                "asset": ["000001.SZ", "000002.SZ"],
                "announce_date": ["2024-01-02", "2024-01-02"],
                "report_period": ["2023-09-30", "2023-09-30"],
                "roe": [10.0, 5.0],
                "roe_dt": [9.5, 4.5],
                "roa": [1.0, 0.8],
                "grossprofit_margin": [30.0, 20.0],
                "netprofit_margin": [10.0, 5.0],
            }
        ),
    )
    (root / "manifest.json").write_text(
        json.dumps({"unavailable_raw_endpoints": []}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return root


def test_build_tushare_research_inputs_outputs_workflow_compatible_tables(tmp_path: Path) -> None:
    standardized = _standardized_dir(tmp_path)
    artifacts = build_tushare_research_inputs(
        standardized,
        output_dir=tmp_path / "research_inputs",
        dataset_id="tushare_case_v1",
    )

    assert artifacts.bundle.prices.shape[0] == 4
    assert artifacts.bundle.factors is not None
    assert set(artifacts.bundle.factors["factor"]) == {
        "momentum_63d",
        "quality_profitability_proxy",
        "value_book_to_price_proxy",
    }
    assert artifacts.neutralization_exposures_path.exists()
    exposures = pd.read_csv(artifacts.neutralization_exposures_path)
    assert {"date", "asset", "size_exposure", "beta_exposure", "industry"} <= set(exposures.columns)


def test_export_canonical_tushare_case_configs_points_to_built_inputs(tmp_path: Path) -> None:
    standardized = _standardized_dir(tmp_path)
    artifacts = build_tushare_research_inputs(
        standardized,
        output_dir=tmp_path / "research_inputs",
        dataset_id="tushare_case_v1",
    )
    case_artifacts = export_canonical_tushare_case_configs(
        artifacts,
        output_dir=tmp_path / "case_configs",
        dataset_id="tushare_case_v1",
    )

    single = json.loads(case_artifacts.single_factor_config_path.read_text(encoding="utf-8"))
    composite = json.loads(case_artifacts.composite_config_path.read_text(encoding="utf-8"))
    assert single["data"]["prices_path"] == str(artifacts.prices_path)
    assert composite["data"]["candidate_signals_path"] == str(artifacts.candidate_signals_path)
