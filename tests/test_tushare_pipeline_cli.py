from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from alpha_lab.data_sources.tushare_extractors import RequiredEndpointExtractionError


def _load_script_module():
    root = Path(__file__).resolve().parents[1]
    script_path = root / "scripts" / "tushare_pipeline.py"
    spec = importlib.util.spec_from_file_location("tushare_pipeline_script", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load script module: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fetch_snapshots_cli_surfaces_required_endpoint_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_script_module()

    def _build_client(*_: Any, **__: Any) -> object:
        return object()

    def _raise_failure(*_: Any, **__: Any) -> object:
        raise RequiredEndpointExtractionError(
            endpoint="daily",
            extraction_mode="trade_date_window",
            params={"start_date": "20240101", "end_date": "20240131"},
            raw_error="查询数据失败，请确认参数！",
            manifest_path=tmp_path / "manifest.json",
        )

    monkeypatch.setattr(module, "build_tushare_pro_client", _build_client)
    monkeypatch.setattr(module, "fetch_tushare_raw_snapshots", _raise_failure)

    rc = module.main(
        [
            "fetch-snapshots",
            "--snapshot-name",
            "snap_cli_fail",
            "--start-date",
            "20240101",
            "--end-date",
            "20240131",
        ]
    )

    assert rc == 1
    captured = capsys.readouterr()
    assert "fetch-snapshots failed on endpoint=daily mode=trade_date_window" in captured.err
    assert "\"start_date\": \"20240101\"" in captured.err


def test_fetch_snapshots_cli_debug_endpoints_prints_progress(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_script_module()

    def _build_client(*_: Any, **__: Any) -> object:
        return object()

    def _fake_fetch(*_: Any, **kwargs: Any) -> object:
        hook = kwargs.get("endpoint_diagnostic_hook")
        assert callable(hook)
        hook(
            {
                "event": "started",
                "endpoint_requested": "daily",
                "extraction_mode": "trade_date_window",
                "required": True,
                "params": {"start_date": "20240101", "end_date": "20240131"},
            }
        )
        hook(
            {
                "event": "finished",
                "endpoint_requested": "daily",
                "extraction_mode": "trade_date_window",
                "required": True,
                "status": "success",
                "row_count": 2,
                "fallback_used": False,
                "degradation_used": False,
                "error_message": None,
            }
        )
        manifest = SimpleNamespace(
            snapshot_dir=tmp_path / "snap_cli_debug",
            endpoints=("trade_cal", "stock_basic", "daily", "daily_basic"),
            unavailable_endpoints=(),
            degraded_endpoints=(),
        )
        return SimpleNamespace(manifest=manifest)

    monkeypatch.setattr(module, "build_tushare_pro_client", _build_client)
    monkeypatch.setattr(module, "fetch_tushare_raw_snapshots", _fake_fetch)

    rc = module.main(
        [
            "fetch-snapshots",
            "--snapshot-name",
            "snap_cli_debug",
            "--start-date",
            "20240101",
            "--end-date",
            "20240131",
            "--debug-endpoints",
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert "[endpoint] started endpoint=daily mode=trade_date_window" in captured.out
    assert "[endpoint] finished endpoint=daily mode=trade_date_window" in captured.out
