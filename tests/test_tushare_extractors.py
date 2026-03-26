from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest

from alpha_lab.data_sources.tushare_extractors import (
    RequiredEndpointExtractionError,
    fetch_tushare_raw_snapshots,
)

QueryHandler = Callable[..., pd.DataFrame]


class _FakeClient:
    def __init__(self, handlers: dict[str, pd.DataFrame | Exception | QueryHandler]) -> None:
        self.handlers = handlers
        self.calls: list[dict[str, object]] = []

    def query(
        self,
        api_name: str,
        *,
        fields: str | None = None,
        limit: int | None = None,
        offset: int | None = None,
        **params: object,
    ) -> pd.DataFrame:
        self.calls.append(
            {
                "api_name": api_name,
                "fields": fields,
                "limit": limit,
                "offset": offset,
                "params": dict(params),
            }
        )
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


def _base_handlers() -> dict[str, pd.DataFrame | Exception | QueryHandler]:
    return {
        "trade_cal": pd.DataFrame(
            {
                "exchange": ["SSE"],
                "cal_date": ["20240102"],
                "is_open": [1],
                "pretrade_date": ["20231229"],
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
                "ts_code": ["000001.SZ", "000002.SZ"],
                "trade_date": ["20240102", "20240102"],
                "close": [10.0, 20.0],
                "vol": [100.0, 200.0],
                "amount": [50.0, 120.0],
            }
        ),
        "daily_basic": pd.DataFrame(
            {
                "ts_code": ["000001.SZ", "000002.SZ"],
                "trade_date": ["20240102", "20240102"],
                "pb": [1.2, 2.3],
                "pe_ttm": [10.5, 15.0],
                "total_mv": [1000.0, 2000.0],
                "circ_mv": [900.0, 1800.0],
            }
        ),
        "adj_factor": pd.DataFrame({"ts_code": [], "trade_date": [], "adj_factor": []}),
        "suspend_d": pd.DataFrame({"ts_code": [], "trade_date": []}),
        "stk_limit": pd.DataFrame({"ts_code": [], "trade_date": []}),
    }


def test_fetch_snapshots_prefers_vip_for_financial_indicators(tmp_path: Path) -> None:
    handlers = _base_handlers()
    handlers["fina_indicator_vip"] = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "ann_date": ["20240220"],
            "end_date": ["20231231"],
            "roe_dt": [9.1],
            "roe": [9.2],
            "roa": [0.8],
        }
    )

    def _unexpected_fina_indicator_call(**_: object) -> pd.DataFrame:
        raise AssertionError("fallback fina_indicator should not be called when VIP succeeds")

    handlers["fina_indicator"] = _unexpected_fina_indicator_call
    client = _FakeClient(handlers)
    out = fetch_tushare_raw_snapshots(
        client,
        snapshot_name="snapshot_vip",
        start_date="20240101",
        end_date="20240131",
        raw_root=tmp_path,
        extraction_utc="2026-03-25T00:00:00+00:00",
    )

    assert out.manifest.snapshot_dir == tmp_path / "snapshot_vip"
    assert "fina_indicator" not in out.manifest.unavailable_endpoints

    daily_calls = [call for call in client.calls if call["api_name"] == "daily"]
    assert daily_calls
    assert daily_calls[0]["params"]["trade_date"] == "20240102"
    assert "start_date" not in daily_calls[0]["params"]
    assert daily_calls[0]["limit"] is None
    assert daily_calls[0]["offset"] is None

    manifest_payload = json.loads(out.manifest.manifest_path.read_text(encoding="utf-8"))
    fina_meta = manifest_payload["endpoint_extractions"]["fina_indicator"]
    assert fina_meta["endpoint_used"] == "fina_indicator_vip"
    assert fina_meta["fallback_occurred"] is False
    assert fina_meta["extraction_mode"] == "financial_quarterly"
    endpoint_table = manifest_payload["endpoint_status_table"]
    assert any(item["endpoint_requested"] == "fina_indicator" for item in endpoint_table)


def test_fetch_snapshots_financial_fallback_queries_per_stock_with_ts_code(
    tmp_path: Path,
) -> None:
    handlers = _base_handlers()
    handlers["fina_indicator_vip"] = RuntimeError("permission denied")

    def _fina_indicator_handler(**kwargs: object) -> pd.DataFrame:
        params = kwargs["params"]
        if "ts_code" not in params:
            raise AssertionError("fina_indicator must always be queried with ts_code")
        ts_code = str(params["ts_code"])
        if ts_code == "000001.SZ":
            return pd.DataFrame(
                {
                    "ts_code": ["000001.SZ"],
                    "ann_date": ["20240215"],
                    "end_date": ["20231231"],
                    "roe_dt": [8.8],
                    "roe": [9.0],
                    "roa": [0.7],
                }
            )
        return pd.DataFrame(
            {
                "ts_code": ["000002.SZ"],
                "ann_date": ["20240218"],
                "end_date": ["20231231"],
                "roe_dt": [5.5],
                "roe": [5.8],
                "roa": [0.6],
            }
        )

    handlers["fina_indicator"] = _fina_indicator_handler
    client = _FakeClient(handlers)
    out = fetch_tushare_raw_snapshots(
        client,
        snapshot_name="snapshot_fallback",
        start_date="20240101",
        end_date="20240131",
        raw_root=tmp_path,
        extraction_utc="2026-03-25T00:00:00+00:00",
    )

    assert "fina_indicator" not in out.manifest.unavailable_endpoints
    fina_calls = [call for call in client.calls if call["api_name"] == "fina_indicator"]
    assert len(fina_calls) == 2
    assert {call["params"]["ts_code"] for call in fina_calls} == {"000001.SZ", "000002.SZ"}
    assert all(call["params"]["start_date"] == "20240101" for call in fina_calls)
    assert all(call["params"]["end_date"] == "20240131" for call in fina_calls)
    fina_snapshot = pd.read_csv(tmp_path / "snapshot_fallback" / "fina_indicator.csv")
    assert list(fina_snapshot["ts_code"]) == ["000001.SZ", "000002.SZ"]

    manifest_payload = json.loads(out.manifest.manifest_path.read_text(encoding="utf-8"))
    fina_meta = manifest_payload["endpoint_extractions"]["fina_indicator"]
    assert fina_meta["endpoint_used"] == "fina_indicator"
    assert fina_meta["fallback_occurred"] is True
    assert fina_meta["degraded"] is True
    assert fina_meta["stocks_queried_count"] == 2
    assert fina_meta["period_query_count"] == 1


def test_fetch_snapshots_optional_financial_failure_is_isolated_and_recorded(
    tmp_path: Path,
) -> None:
    handlers = _base_handlers()
    handlers["fina_indicator_vip"] = RuntimeError("permission denied")
    handlers["fina_indicator"] = RuntimeError("permission denied")
    client = _FakeClient(handlers)
    out = fetch_tushare_raw_snapshots(
        client,
        snapshot_name="snapshot_no_financial",
        start_date="20240101",
        end_date="20240131",
        raw_root=tmp_path,
        extraction_utc="2026-03-25T00:00:00+00:00",
    )

    assert out.manifest.snapshot_dir == tmp_path / "snapshot_no_financial"
    assert (tmp_path / "snapshot_no_financial" / "daily.csv").exists()
    assert "fina_indicator" in out.manifest.unavailable_endpoints
    assert "fina_indicator" in out.manifest.degraded_endpoints
    assert not (tmp_path / "snapshot_no_financial" / "fina_indicator.csv").exists()

    manifest_payload = json.loads(out.manifest.manifest_path.read_text(encoding="utf-8"))
    fina_meta = manifest_payload["endpoint_extractions"]["fina_indicator"]
    assert fina_meta["status"] == "failed"
    assert fina_meta["fallback_occurred"] is True
    assert fina_meta["degraded"] is True
    assert fina_meta["degradation_used"] is True
    assert fina_meta["error_message"] == "fina_indicator fallback failed for all queried stocks"


def test_required_endpoint_failure_has_contextual_error_and_manifest(
    tmp_path: Path,
) -> None:
    handlers = _base_handlers()
    handlers["daily"] = RuntimeError("查询数据失败，请确认参数！")
    handlers["fina_indicator_vip"] = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "ann_date": ["20240220"],
            "end_date": ["20231231"],
            "roe_dt": [9.1],
            "roe": [9.2],
            "roa": [0.8],
        }
    )
    handlers["fina_indicator"] = RuntimeError("should not be called")
    client = _FakeClient(handlers)

    with pytest.raises(RequiredEndpointExtractionError) as excinfo:
        fetch_tushare_raw_snapshots(
            client,
            snapshot_name="snapshot_required_failure",
            start_date="20240101",
            end_date="20240131",
            raw_root=tmp_path,
            extraction_utc="2026-03-25T00:00:00+00:00",
        )

    err = excinfo.value
    message = str(err)
    assert "endpoint=daily" in message
    assert "mode=trade_date_window" in message
    assert "\"start_date\": \"20240101\"" in message
    assert "查询数据失败，请确认参数！" in message

    manifest_payload = json.loads(err.manifest_path.read_text(encoding="utf-8"))
    daily_meta = manifest_payload["endpoint_extractions"]["daily"]
    assert daily_meta["status"] == "failed"
    assert daily_meta["required"] is True
    assert daily_meta["error_message"] == "查询数据失败，请确认参数！"
    assert daily_meta["params"] == {"start_date": "20240101", "end_date": "20240131"}
    endpoint_table = manifest_payload["endpoint_status_table"]
    daily_rows = [item for item in endpoint_table if item["endpoint_requested"] == "daily"]
    assert len(daily_rows) == 1
    assert daily_rows[0]["status"] == "failed"
    assert daily_rows[0]["endpoint_used"] == "daily"


def test_endpoint_diagnostics_hook_emits_started_and_finished_events(tmp_path: Path) -> None:
    handlers = _base_handlers()
    handlers["fina_indicator_vip"] = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "ann_date": ["20240220"],
            "end_date": ["20231231"],
            "roe_dt": [9.1],
            "roe": [9.2],
            "roa": [0.8],
        }
    )
    handlers["fina_indicator"] = RuntimeError("should not be called")
    client = _FakeClient(handlers)
    events: list[dict[str, object]] = []
    fetch_tushare_raw_snapshots(
        client,
        snapshot_name="snapshot_hook",
        start_date="20240101",
        end_date="20240131",
        raw_root=tmp_path,
        extraction_utc="2026-03-25T00:00:00+00:00",
        endpoint_diagnostic_hook=lambda item: events.append(dict(item)),
    )

    assert events
    assert any(item["event"] == "started" for item in events)
    assert any(item["event"] == "finished" for item in events)
    daily_finished = [
        item
        for item in events
        if item["event"] == "finished" and item["endpoint_requested"] == "daily"
    ]
    assert len(daily_finished) == 1
    assert daily_finished[0]["status"] == "success"
    assert daily_finished[0]["error_message"] is None


def test_daily_endpoint_regression_uses_trade_date_mode_without_generic_pagination(
    tmp_path: Path,
) -> None:
    handlers = _base_handlers()

    def _daily_handler(**kwargs: object) -> pd.DataFrame:
        params = kwargs["params"]
        limit = kwargs["limit"]
        offset = kwargs["offset"]
        if "start_date" in params or "end_date" in params:
            raise RuntimeError("daily does not accept start_date/end_date in this path")
        if limit is not None or offset is not None:
            raise RuntimeError("daily does not support generic limit/offset pagination")
        trade_date = str(params.get("trade_date", "")).strip()
        if trade_date != "20240102":
            return pd.DataFrame(
                {"ts_code": [], "trade_date": [], "close": [], "vol": [], "amount": []}
            )
        return pd.DataFrame(
            {
                "ts_code": ["000001.SZ", "000002.SZ"],
                "trade_date": [trade_date, trade_date],
                "close": [10.0, 20.0],
                "vol": [100.0, 200.0],
                "amount": [50.0, 120.0],
            }
        )

    handlers["daily"] = _daily_handler
    handlers["fina_indicator_vip"] = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "ann_date": ["20240220"],
            "end_date": ["20231231"],
            "roe_dt": [9.1],
            "roe": [9.2],
            "roa": [0.8],
        }
    )
    handlers["fina_indicator"] = RuntimeError("should not be called")
    client = _FakeClient(handlers)
    out = fetch_tushare_raw_snapshots(
        client,
        snapshot_name="snapshot_daily_mode_fix",
        start_date="20240101",
        end_date="20240131",
        raw_root=tmp_path,
        extraction_utc="2026-03-25T00:00:00+00:00",
    )

    assert out.manifest.snapshot_dir == tmp_path / "snapshot_daily_mode_fix"
    daily_calls = [call for call in client.calls if call["api_name"] == "daily"]
    assert len(daily_calls) == 1
    assert daily_calls[0]["params"] == {"trade_date": "20240102"}
    assert daily_calls[0]["limit"] is None
    assert daily_calls[0]["offset"] is None
