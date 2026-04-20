from __future__ import annotations

import builtins
from pathlib import Path

import pandas as pd
import pytest

import alpha_lab.data_adapters.baostock_adapter as baostock_adapter
from alpha_lab.data_adapters.baostock_adapter import generate_real_case_inputs
from alpha_lab.data_validation import validate_price_panel
from alpha_lab.interfaces import validate_factor_output


def _price_row(
    date: str,
    asset: str,
    close: float,
    *,
    volume: float,
    amount: float,
) -> dict[str, object]:
    return {
        "date": date,
        "asset": asset,
        "close": close,
        "volume": volume,
        "amount": amount,
    }


def test_asset_code_normalization_roundtrip() -> None:
    assert baostock_adapter._to_baostock_asset("000001.SZ") == "sz.000001"
    assert baostock_adapter._to_baostock_asset("sh.600000") == "sh.600000"
    assert baostock_adapter._to_tushare_asset("sz.000001") == "000001.SZ"
    assert baostock_adapter._to_tushare_asset("600000.SH") == "600000.SH"
    assert baostock_adapter._is_ashare_baostock_asset("sh.600000")
    assert baostock_adapter._is_ashare_baostock_asset("sz.300001")
    assert not baostock_adapter._is_ashare_baostock_asset("sz.200001")


def test_import_baostock_missing_dependency_has_actionable_message(monkeypatch) -> None:
    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name == "baostock":
            raise ModuleNotFoundError("No module named 'baostock'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    with pytest.raises(RuntimeError, match="uv run --with baostock"):
        baostock_adapter._import_baostock()


def test_resolve_parallel_worker_count_defaults_to_serial(monkeypatch) -> None:
    monkeypatch.delenv(baostock_adapter._BAOSTOCK_PARALLEL_ENV, raising=False)
    assert baostock_adapter._resolve_parallel_worker_count(20) == 1


def test_resolve_parallel_worker_count_honors_env_opt_in(monkeypatch) -> None:
    monkeypatch.setenv(baostock_adapter._BAOSTOCK_PARALLEL_ENV, "4")
    monkeypatch.setattr(baostock_adapter.os, "cpu_count", lambda: 16)
    assert baostock_adapter._resolve_parallel_worker_count(20) == 4


def test_resolve_assets_falls_back_to_previous_days(monkeypatch) -> None:
    class _FakeClient:
        def __init__(self) -> None:
            self.days: list[str] = []

        def query_all_stock(self, *, day: str):
            self.days.append(day)
            return {"day": day}

    client = _FakeClient()

    def _fake_result_to_frame(result, *, query_name: str, allow_error: bool = False):  # type: ignore[no-untyped-def]
        assert query_name == "query_all_stock"
        day = str(result["day"])
        if day in {"2024-03-31", "2024-03-30"}:
            return pd.DataFrame({"code": []})
        if day == "2024-03-29":
            return pd.DataFrame({"code": ["sh.600000", "sz.000001", "sz.200001"]})
        return pd.DataFrame({"code": []})

    monkeypatch.setattr(baostock_adapter, "_result_to_frame", _fake_result_to_frame)
    assets = baostock_adapter._resolve_assets(
        client,
        end_date="2024-03-31",
        assets=None,
        asset_limit=None,
    )

    assert assets == ["sh.600000", "sz.000001"]
    assert client.days[:3] == ["2024-03-31", "2024-03-30", "2024-03-29"]


def test_fetch_raw_inputs_can_skip_roe_fetch(monkeypatch) -> None:
    monkeypatch.setattr(
        baostock_adapter,
        "_resolve_assets",
        lambda *_args, **_kwargs: ["sz.000001"],
    )

    def _fake_query_history(_client, *, asset: str, start_date: str, end_date: str) -> pd.DataFrame:
        assert asset == "sz.000001"
        assert start_date == "2024-01-01"
        assert end_date == "2024-01-31"
        return pd.DataFrame(
            [
                {
                    "date": "2024-01-02",
                    "asset": "000001.SZ",
                    "close": 10.0,
                    "volume": 1000.0,
                    "amount": 10000.0,
                    "pb": 2.0,
                },
                {
                    "date": "2024-01-03",
                    "asset": "000001.SZ",
                    "close": 10.2,
                    "volume": 1200.0,
                    "amount": 12240.0,
                    "pb": 2.1,
                },
            ]
        )

    monkeypatch.setattr(baostock_adapter, "_query_history_price_and_pb", _fake_query_history)

    def _fail_fetch_roe(*_args, **_kwargs):
        raise AssertionError("_fetch_roe_raw should not be called when include_roe=False")

    monkeypatch.setattr(baostock_adapter, "_fetch_roe_raw", _fail_fetch_roe)
    out = baostock_adapter._fetch_raw_inputs(
        client=object(),
        start_date="2024-01-01",
        end_date="2024-01-31",
        assets=None,
        asset_limit=None,
        include_roe=False,
    )

    assert not out.prices.empty
    assert not out.pb_raw.empty
    assert out.roe_raw.empty
    assert out.dedup_counts["roe_raw"] == 0
    assert {"volume", "amount"} <= set(out.prices.columns)


def test_generate_real_case_inputs_applies_asset_limit(
    tmp_path: Path,
    monkeypatch,
) -> None:
    prices = pd.DataFrame(
        [
            _price_row("2024-01-02", "000001.SZ", 10.0, volume=1.0, amount=10.0),
            _price_row("2024-01-03", "000001.SZ", 10.1, volume=1.1, amount=11.11),
            _price_row("2024-01-02", "000002.SZ", 20.0, volume=2.0, amount=40.0),
            _price_row("2024-01-03", "000002.SZ", 20.2, volume=2.2, amount=44.44),
            _price_row("2024-01-02", "000003.SZ", 30.0, volume=3.0, amount=90.0),
            _price_row("2024-01-03", "000003.SZ", 30.3, volume=3.3, amount=99.99),
        ]
    )
    pb_raw = pd.DataFrame(
        [
            {"date": "2024-01-02", "asset": "000001.SZ", "pb": 2.0},
            {"date": "2024-01-03", "asset": "000001.SZ", "pb": 2.1},
            {"date": "2024-01-02", "asset": "000002.SZ", "pb": 3.0},
            {"date": "2024-01-03", "asset": "000002.SZ", "pb": 3.1},
            {"date": "2024-01-02", "asset": "000003.SZ", "pb": 4.0},
            {"date": "2024-01-03", "asset": "000003.SZ", "pb": 4.1},
        ]
    )
    roe_raw = pd.DataFrame(
        [
            {
                "asset": "000001.SZ",
                "ann_date": "2024-01-02",
                "end_date": "2023-12-31",
                "roe_value": 10.0,
            },
            {
                "asset": "000002.SZ",
                "ann_date": "2024-01-02",
                "end_date": "2023-12-31",
                "roe_value": 12.0,
            },
            {
                "asset": "000003.SZ",
                "ann_date": "2024-01-02",
                "end_date": "2023-12-31",
                "roe_value": 14.0,
            },
        ]
    )

    raw_inputs = baostock_adapter._BaostockRawInputs(
        prices=prices,
        pb_raw=pb_raw,
        roe_raw=roe_raw,
        dedup_counts={"prices_raw": 0, "pb_raw": 0, "roe_raw": 0},
        roe_source_column="roeAvg",
    )

    def _fake_run_with_baostock_session(_fn):
        return raw_inputs

    monkeypatch.setattr(
        baostock_adapter,
        "_run_with_baostock_session",
        _fake_run_with_baostock_session,
    )
    summary = generate_real_case_inputs(
        output_dir=tmp_path,
        start_date="2024-01-01",
        end_date="2024-03-31",
        asset_limit=2,
    )

    prices_out = pd.read_csv(summary.output_paths["prices"])
    assert sorted(prices_out["asset"].unique().tolist()) == ["000001.SZ", "000002.SZ"]
    assert {"volume", "amount"} <= set(prices_out.columns)
    validate_price_panel(prices_out)

    bp_out = pd.read_csv(summary.output_paths["bp"])
    validate_factor_output(bp_out)
    assert sorted(bp_out["asset"].unique().tolist()) == ["000001.SZ", "000002.SZ"]
