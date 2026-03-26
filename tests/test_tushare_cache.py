from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from alpha_lab.data_sources.tushare_cache import (
    load_raw_snapshot,
    write_raw_snapshot,
    write_raw_snapshot_manifest,
)


def test_write_raw_snapshot_is_deterministic(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "trade_date": ["20240102", "20240101"],
            "ts_code": ["000001.SZ", "000001.SZ"],
            "close": [11.0, 10.0],
        }
    )
    write_raw_snapshot(
        df,
        endpoint="daily",
        snapshot_dir=tmp_path,
        params={"start_date": "20240101", "end_date": "20240102"},
        extraction_utc="2026-03-25T00:00:00+00:00",
    )
    first_csv = (tmp_path / "daily.csv").read_text(encoding="utf-8")
    first_meta = (tmp_path / "daily.meta.json").read_text(encoding="utf-8")

    write_raw_snapshot(
        df,
        endpoint="daily",
        snapshot_dir=tmp_path,
        params={"start_date": "20240101", "end_date": "20240102"},
        extraction_utc="2026-03-25T00:00:00+00:00",
    )
    second_csv = (tmp_path / "daily.csv").read_text(encoding="utf-8")
    second_meta = (tmp_path / "daily.meta.json").read_text(encoding="utf-8")

    assert first_csv == second_csv
    assert first_meta == second_meta
    loaded = load_raw_snapshot(tmp_path, endpoint="daily")
    assert list(loaded["trade_date"]) == [20240101, 20240102]


def test_write_raw_snapshot_metadata_records_params(tmp_path: Path) -> None:
    df = pd.DataFrame({"ts_code": [], "trade_date": [], "close": []})
    write_raw_snapshot(
        df,
        endpoint="daily",
        snapshot_dir=tmp_path,
        params={"start_date": "20240101"},
        extraction_utc="2026-03-25T00:00:00+00:00",
    )
    payload = json.loads((tmp_path / "daily.meta.json").read_text(encoding="utf-8"))
    assert payload["params"]["start_date"] == "20240101"
    assert payload["row_count"] == 0


def test_write_raw_snapshot_manifest_persists_endpoint_status_table(tmp_path: Path) -> None:
    manifest = write_raw_snapshot_manifest(
        snapshot_name="snap_manifest",
        snapshot_dir=tmp_path,
        extraction_utc="2026-03-25T00:00:00+00:00",
        endpoints=["trade_cal"],
        unavailable_endpoints=[],
        endpoint_extractions={
            "trade_cal": {
                "endpoint_requested": "trade_cal",
                "endpoint_used": "trade_cal",
                "extraction_mode": "range_query",
                "params": {"start_date": "20240101", "end_date": "20240131"},
                "required": True,
                "status": "success",
                "row_count": 20,
                "fallback_used": False,
                "degradation_used": False,
                "error_message": None,
            }
        },
        endpoint_status_table=[
            {
                "endpoint_requested": "trade_cal",
                "endpoint_used": "trade_cal",
                "extraction_mode": "range_query",
                "params": {"start_date": "20240101", "end_date": "20240131"},
                "required": True,
                "status": "success",
                "row_count": 20,
                "fallback_used": False,
                "degradation_used": False,
                "error_message": None,
            }
        ],
        degraded_endpoints=[],
    )

    payload = json.loads(manifest.manifest_path.read_text(encoding="utf-8"))
    assert len(payload["endpoint_status_table"]) == 1
    assert payload["endpoint_status_table"][0]["endpoint_requested"] == "trade_cal"
    assert payload["endpoint_status_table"][0]["status"] == "success"
    assert len(manifest.endpoint_status_table) == 1
    assert manifest.endpoint_status_table[0]["row_count"] == 20
