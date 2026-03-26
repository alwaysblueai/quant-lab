from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from alpha_lab.data_sources import (
    build_standardized_tushare_tables,
    build_tushare_research_inputs,
    export_canonical_tushare_case_configs,
    fetch_tushare_raw_snapshots,
)

QueryHandler = Callable[..., pd.DataFrame]


class _FakeClient:
    def __init__(self, handlers: dict[str, pd.DataFrame | Exception | QueryHandler]) -> None:
        self.handlers = handlers

    def query(
        self,
        api_name: str,
        *,
        fields: str | None = None,
        limit: int | None = None,
        offset: int | None = None,
        **params: object,
    ) -> pd.DataFrame:
        _ = fields, limit, offset, params
        handler = self.handlers[api_name]
        if isinstance(handler, Exception):
            raise handler
        if callable(handler):
            return handler(
                api_name=api_name,
                fields=fields,
                limit=limit,
                offset=offset,
                params=dict(params),
            ).copy()
        return handler.copy()


def _required_handlers() -> dict[str, pd.DataFrame | Exception | QueryHandler]:
    return {
        "trade_cal": pd.DataFrame(
            {
                "exchange": ["SSE", "SSE"],
                "cal_date": ["20240102", "20240103"],
                "is_open": [1, 1],
                "pretrade_date": ["20231229", "20240102"],
            }
        ),
        "stock_basic": pd.DataFrame(
            {
                "ts_code": ["000001.SZ", "000002.SZ"],
                "name": ["PingAn", "Vanke"],
                "industry": ["Bank", "RealEstate"],
                "market": ["main", "main"],
                "exchange": ["SZSE", "SZSE"],
                "list_status": ["L", "L"],
                "list_date": ["19910403", "19910129"],
                "delist_date": [None, None],
                "is_hs": ["N", "N"],
            }
        ),
        "daily": pd.DataFrame(
            {
                "ts_code": ["000001.SZ", "000001.SZ", "000002.SZ", "000002.SZ"],
                "trade_date": ["20240102", "20240103", "20240102", "20240103"],
                "close": [10.0, 10.2, 20.0, 20.4],
                "open": [9.8, 10.0, 19.8, 20.1],
                "high": [10.1, 10.3, 20.2, 20.5],
                "low": [9.7, 9.9, 19.6, 20.0],
                "pre_close": [9.9, 10.0, 19.9, 20.0],
                "vol": [100.0, 110.0, 150.0, 160.0],
                "amount": [50.0, 60.0, 120.0, 130.0],
            }
        ),
        "daily_basic": pd.DataFrame(
            {
                "ts_code": ["000001.SZ", "000001.SZ", "000002.SZ", "000002.SZ"],
                "trade_date": ["20240102", "20240103", "20240102", "20240103"],
                "pb": [1.2, 1.1, 2.3, 2.2],
                "pe_ttm": [10.5, 10.3, 15.0, 14.8],
                "total_mv": [1000.0, 1010.0, 2000.0, 2020.0],
                "circ_mv": [900.0, 910.0, 1800.0, 1820.0],
            }
        ),
        "adj_factor": pd.DataFrame({"ts_code": [], "trade_date": [], "adj_factor": []}),
        "suspend_d": pd.DataFrame({"ts_code": [], "trade_date": []}),
        "stk_limit": pd.DataFrame(
            {
                "ts_code": ["000001.SZ", "000002.SZ"],
                "trade_date": ["20240102", "20240102"],
                "up_limit": [11.0, 21.0],
                "down_limit": [9.0, 19.0],
            }
        ),
        "fina_indicator_vip": RuntimeError("permission denied"),
        "fina_indicator": RuntimeError("permission denied"),
    }


def test_financial_degradation_does_not_block_single_case_or_case_config_export(
    tmp_path: Path,
) -> None:
    client = _FakeClient(_required_handlers())
    extracted = fetch_tushare_raw_snapshots(
        client,
        snapshot_name="snapshot_optional_financial",
        start_date="20240101",
        end_date="20240131",
        raw_root=tmp_path,
        extraction_utc="2026-03-25T00:00:00+00:00",
    )

    assert "fina_indicator" in extracted.manifest.unavailable_endpoints

    standardized = build_standardized_tushare_tables(
        extracted.manifest.snapshot_dir,
        output_dir=tmp_path / "standardized",
    )
    assert standardized.financial_indicators.empty

    bundle = build_tushare_research_inputs(
        standardized.snapshot_dir,
        output_dir=tmp_path / "research_inputs",
        dataset_id="tushare_case_v1",
    )
    assert bundle.bundle.prices.shape[0] == 4
    assert "fina_indicator" in bundle.unavailable_inputs
    assert bundle.neutralization_exposures_path.exists()

    factors = bundle.bundle.factors
    quality_rows = factors[factors["factor"] == "quality_profitability_proxy"]
    assert not quality_rows.empty
    assert quality_rows["value"].isna().all()

    payload = json.loads(bundle.manifest_path.read_text(encoding="utf-8"))
    note_text = "\n".join(str(item) for item in payload.get("notes", []))
    assert "quality_profitability_proxy is degraded or missing" in note_text

    cases = export_canonical_tushare_case_configs(bundle, output_dir=tmp_path / "cases")
    single_cfg = json.loads(cases.single_factor_config_path.read_text(encoding="utf-8"))
    composite_cfg = json.loads(cases.composite_config_path.read_text(encoding="utf-8"))
    assert "candidate_signals_path" not in single_cfg["data"]
    assert "candidate_signals_path" in composite_cfg["data"]
