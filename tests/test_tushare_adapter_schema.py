from __future__ import annotations

from pathlib import Path

import pandas as pd

import alpha_lab.data_adapters.tushare_adapter as tushare_adapter
from alpha_lab.data_adapters.tushare_adapter import (
    FundamentalFetchResult,
    build_bp_factor,
    build_roe_factor,
    build_universe,
    fetch_fundamentals,
    generate_real_case_inputs,
)
from alpha_lab.data_validation import validate_price_panel
from alpha_lab.interfaces import validate_factor_output


def _price_row(
    date: str,
    asset: str,
    close: float,
    *,
    open_price: float | None = None,
    high: float | None = None,
    low: float | None = None,
    pre_close: float | None = None,
    volume: float,
    amount: float,
) -> dict[str, object]:
    return {
        "date": date,
        "asset": asset,
        "open": close if open_price is None else open_price,
        "high": close if high is None else high,
        "low": close if low is None else low,
        "close": close,
        "pre_close": close if pre_close is None else pre_close,
        "volume": volume,
        "amount": amount,
    }


def test_bp_and_universe_schema_with_deduplicated_prices() -> None:
    prices = pd.DataFrame(
        [
            _price_row("2024-01-02", "000001.SZ", 10.0, volume=1.0, amount=10.0),
            _price_row("2024-01-03", "000001.SZ", 10.5, volume=1.2, amount=12.6),
            _price_row("2024-01-02", "000002.SZ", 20.0, volume=2.0, amount=40.0),
        ]
    )
    validate_price_panel(prices)

    pb_raw = pd.DataFrame(
        [
            {"date": "2024-01-02", "asset": "000001.SZ", "pb": 2.0},
            {"date": "2024-01-03", "asset": "000001.SZ", "pb": 2.5},
            {"date": "2024-01-02", "asset": "000002.SZ", "pb": 0.0},
            {"date": "2024-01-02", "asset": "000002.SZ", "pb": 4.0},
        ]
    )

    bp = build_bp_factor(pb_raw, prices)
    validate_factor_output(bp)

    assert list(bp.columns) == ["date", "asset", "factor", "value"]
    assert bp["factor"].eq("bp").all()
    assert not bp.duplicated(subset=["date", "asset", "factor"]).any()

    row = bp[(bp["date"] == "2024-01-02") & (bp["asset"] == "000001.SZ")].iloc[0]
    assert row["value"] == 0.5

    universe = build_universe(prices)
    assert list(universe.columns) == ["date", "asset", "in_universe"]
    assert universe["in_universe"].eq(1).all()
    assert not universe.duplicated(subset=["date", "asset"]).any()


def test_roe_factor_forward_fill_with_ann_date_fallback() -> None:
    prices = pd.DataFrame(
        [
            _price_row("2024-01-01", "000001.SZ", 10.0, volume=1.0, amount=10.0),
            _price_row("2024-01-02", "000001.SZ", 10.1, volume=1.1, amount=11.11),
            _price_row("2024-01-03", "000001.SZ", 10.2, volume=1.2, amount=12.24),
            _price_row("2024-01-04", "000001.SZ", 10.3, volume=1.3, amount=13.39),
            _price_row("2024-01-05", "000001.SZ", 10.4, volume=1.4, amount=14.56),
        ]
    )
    validate_price_panel(prices)

    roe_raw = pd.DataFrame(
        [
            {
                "asset": "000001.SZ",
                "ann_date": "2024-01-02",
                "end_date": "2023-12-31",
                "roe_value": 10.0,
            },
            {
                "asset": "000001.SZ",
                "ann_date": None,
                "end_date": "2024-01-05",
                "roe_value": 20.0,
            },
        ]
    )

    roe, fallback_count = build_roe_factor(roe_raw, prices)
    validate_factor_output(roe)

    assert fallback_count == 1
    assert list(roe.columns) == ["date", "asset", "factor", "value"]
    assert roe["factor"].eq("roe_ttm").all()
    assert not roe.duplicated(subset=["date", "asset", "factor"]).any()

    first_date = roe["date"].min()
    assert first_date == "2024-01-02"

    by_date = roe.set_index("date")["value"].to_dict()
    assert by_date["2024-01-02"] == 10.0
    assert by_date["2024-01-03"] == 10.0
    assert by_date["2024-01-04"] == 10.0
    assert by_date["2024-01-05"] == 20.0


def test_generate_real_case_inputs_applies_asset_limit(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("ALPHA_LAB_DATA_ROOT", str(tmp_path / "warehouse"))
    prices = pd.DataFrame(
        [
            _price_row("2024-01-02", "000001.SZ", 10.0, volume=1.0, amount=10.0),
            _price_row("2024-01-03", "000001.SZ", 10.2, volume=1.1, amount=11.22),
            _price_row("2024-01-02", "000002.SZ", 20.0, volume=2.0, amount=40.0),
            _price_row("2024-01-03", "000002.SZ", 20.1, volume=2.2, amount=44.22),
            _price_row("2024-01-02", "000003.SZ", 30.0, volume=3.0, amount=90.0),
            _price_row("2024-01-03", "000003.SZ", 30.5, volume=3.2, amount=97.6),
        ]
    )
    captured_assets: list[str] = []

    def _fake_build_tushare_client(_token: str) -> object:
        return object()

    def _fake_fetch_prices(
        _pro: object,
        *,
        start_date: str,
        end_date: str,
        assets: list[str] | None = None,
    ) -> tuple[pd.DataFrame, int]:
        assert start_date == "2024-01-01"
        assert end_date == "2024-12-31"
        assert assets is None
        return prices.copy(), 0

    def _fake_fetch_fundamentals(
        _pro: object,
        *,
        assets: list[str],
        start_date: str,
        end_date: str,
        include_daily_basic: bool = True,
        include_roe: bool = True,
        token: str | None = None,
    ) -> FundamentalFetchResult:
        assert start_date == "2024-01-01"
        assert end_date == "2024-12-31"
        assert include_daily_basic is True
        assert include_roe is True
        assert token == "dummy"
        captured_assets.extend(assets)
        pb_raw = pd.DataFrame(
            [
                {"date": "2024-01-02", "asset": asset, "pb": 2.0, "turnover_rate": 1.5}
                for asset in assets
            ]
        )
        roe_raw = pd.DataFrame(
            [
                {
                    "asset": asset,
                    "ann_date": "2024-01-02",
                    "end_date": "2023-12-31",
                    "roe_value": 12.0,
                }
                for asset in assets
            ]
        )
        return FundamentalFetchResult(
            pb_raw=pb_raw,
            roe_raw=roe_raw,
            balance_sheet_raw=pd.DataFrame(
                [
                    {
                        "asset": asset,
                        "ann_date": "2024-01-02",
                        "end_date": "2023-12-31",
                        "goodwill_balance": 1.0,
                    }
                    for asset in assets
                ]
            ),
            income_statement_raw=pd.DataFrame(
                [
                    {
                        "asset": asset,
                        "ann_date": "2024-01-02",
                        "end_date": "2023-12-31",
                        "operating_revenue_ttm": 100.0,
                        "rd_expense": 5.0,
                    }
                    for asset in assets
                ]
            ),
            cash_flow_statement_raw=pd.DataFrame(
                [
                    {
                        "asset": asset,
                        "ann_date": "2024-01-02",
                        "end_date": "2023-12-31",
                        "operating_cash_flow_ttm": 10.0,
                    }
                    for asset in assets
                ]
            ),
            dedup_counts={"pb_raw": 0, "roe_raw": 0},
            roe_source_column="roe_ttm",
        )

    def _fake_fetch_adj_factor(
        _pro: object,
        *,
        start_date: str,
        end_date: str,
        assets: tuple[str, ...] | None = None,
    ) -> tuple[pd.DataFrame, int]:
        assert start_date == "2024-01-01"
        assert end_date == "2024-12-31"
        assert assets == ("000001.SZ", "000002.SZ")
        return (
            pd.DataFrame(
                [
                    {"date": "2024-01-02", "asset": "000001.SZ", "adj_factor": 1.0},
                    {"date": "2024-01-03", "asset": "000001.SZ", "adj_factor": 1.1},
                    {"date": "2024-01-02", "asset": "000002.SZ", "adj_factor": 1.0},
                    {"date": "2024-01-03", "asset": "000002.SZ", "adj_factor": 1.1},
                ]
            ),
            0,
        )

    def _fake_fetch_stk_limit(
        _pro: object,
        *,
        start_date: str,
        end_date: str,
        assets: tuple[str, ...] | None = None,
    ) -> tuple[pd.DataFrame, int]:
        assert start_date == "2024-01-01"
        assert end_date == "2024-12-31"
        assert assets == ("000001.SZ", "000002.SZ")
        return (
            pd.DataFrame(
                [
                    {
                        "date": "2024-01-02",
                        "asset": "000001.SZ",
                        "up_limit": 11.0,
                        "down_limit": 9.0,
                    },
                    {
                        "date": "2024-01-03",
                        "asset": "000001.SZ",
                        "up_limit": 11.22,
                        "down_limit": 9.18,
                    },
                    {
                        "date": "2024-01-02",
                        "asset": "000002.SZ",
                        "up_limit": 22.0,
                        "down_limit": 18.0,
                    },
                    {
                        "date": "2024-01-03",
                        "asset": "000002.SZ",
                        "up_limit": 22.11,
                        "down_limit": 18.09,
                    },
                ]
            ),
            0,
        )

    def _fake_fetch_suspend_status(
        _pro: object,
        *,
        start_date: str,
        end_date: str,
        assets: tuple[str, ...] | None = None,
    ) -> tuple[pd.DataFrame, int]:
        assert start_date == "2024-01-01"
        assert end_date == "2024-12-31"
        assert assets == ("000001.SZ", "000002.SZ")
        return (pd.DataFrame(columns=["date", "asset", "is_suspended"]), 0)

    def _fake_fetch_st_name_events(
        _pro: object,
        *,
        start_date: str,
        end_date: str,
        assets: tuple[str, ...],
    ) -> tuple[pd.DataFrame, int]:
        assert start_date == "2024-01-01"
        assert end_date == "2024-12-31"
        assert assets == ("000001.SZ", "000002.SZ")
        return (
            pd.DataFrame(
                [
                    {
                        "asset": "000001.SZ",
                        "start_date": "2024-01-03",
                        "end_date": "2024-12-31",
                        "name": "*ST PingAn",
                        "is_st": 1,
                    }
                ]
            ),
            0,
        )

    def _fake_fetch_index_membership(
        _pro: object,
        *,
        start_date: str,
        end_date: str,
        assets: tuple[str, ...] | None = None,
        index_codes: dict[str, str] | None = None,
    ) -> tuple[pd.DataFrame, int]:
        assert start_date == "2024-01-01"
        assert end_date == "2024-12-31"
        assert assets == ("000001.SZ", "000002.SZ")
        return (
            pd.DataFrame(
                [
                    {
                        "date": "2024-01-02",
                        "index_code": "000300.SH",
                        "index_name": "hs300",
                        "asset": "000001.SZ",
                        "weight": 1.0,
                    }
                ]
            ),
            0,
        )

    def _fake_fetch_moneyflow(
        _pro: object,
        *,
        start_date: str,
        end_date: str,
        assets: tuple[str, ...] | None = None,
    ) -> tuple[pd.DataFrame, int]:
        assert start_date == "2024-01-01"
        assert end_date == "2024-12-31"
        assert assets == ("000001.SZ", "000002.SZ")
        return (
            pd.DataFrame(
                [
                    {
                        "date": "2024-01-02",
                        "asset": "000001.SZ",
                        "buy_sm_amount": 1.0,
                        "sell_sm_amount": 0.5,
                        "net_mf_amount": 0.5,
                    },
                    {
                        "date": "2024-01-03",
                        "asset": "000001.SZ",
                        "buy_sm_amount": 1.1,
                        "sell_sm_amount": 0.4,
                        "net_mf_amount": 0.7,
                    },
                    {
                        "date": "2024-01-02",
                        "asset": "000002.SZ",
                        "buy_sm_amount": 2.0,
                        "sell_sm_amount": 1.0,
                        "net_mf_amount": 1.0,
                    },
                    {
                        "date": "2024-01-03",
                        "asset": "000002.SZ",
                        "buy_sm_amount": 2.2,
                        "sell_sm_amount": 1.1,
                        "net_mf_amount": 1.1,
                    },
                ]
            ),
            0,
        )

    def _fake_fetch_industry_classification(
        _pro: object,
        *,
        snapshot_date: str,
        src: str = "SW2021",
    ) -> tuple[pd.DataFrame, int]:
        assert snapshot_date == "2024-12-31"
        assert src == "SW2021"
        return (
            pd.DataFrame(
                [
                    {
                        "snapshot_date": "2024-12-31",
                        "industry_standard": "SW2021",
                        "index_code": "801000.SI",
                        "industry_name": "申万一级",
                        "parent_code": "",
                        "level": "L1",
                        "industry_code": "801000",
                        "is_published": 1,
                    },
                    {
                        "snapshot_date": "2024-12-31",
                        "industry_standard": "SW2021",
                        "index_code": "801010.SI",
                        "industry_name": "申万二级",
                        "parent_code": "801000.SI",
                        "level": "L2",
                        "industry_code": "801010",
                        "is_published": 1,
                    },
                    {
                        "snapshot_date": "2024-12-31",
                        "industry_standard": "SW2021",
                        "index_code": "801011.SI",
                        "industry_name": "申万三级",
                        "parent_code": "801010.SI",
                        "level": "L3",
                        "industry_code": "801011",
                        "is_published": 1,
                    },
                ]
            ),
            0,
        )

    def _fake_fetch_industry_membership(
        _pro: object,
        *,
        classification: pd.DataFrame,
        src: str = "SW2021",
    ) -> tuple[pd.DataFrame, int]:
        assert src == "SW2021"
        assert not classification.empty
        return (
            pd.DataFrame(
                [
                    {
                        "industry_standard": "SW2021",
                        "asset": "000001.SZ",
                        "in_date": "2024-01-01",
                        "out_date": None,
                        "is_new": 1,
                        "l1_code": "801000.SI",
                        "l1_name": "申万一级",
                        "l2_code": "801010.SI",
                        "l2_name": "申万二级",
                        "l3_code": "801011.SI",
                        "l3_name": "申万三级",
                    },
                    {
                        "industry_standard": "SW2021",
                        "asset": "000002.SZ",
                        "in_date": "2024-01-01",
                        "out_date": None,
                        "is_new": 0,
                        "l1_code": "801000.SI",
                        "l1_name": "申万一级",
                        "l2_code": "801010.SI",
                        "l2_name": "申万二级",
                        "l3_code": "801011.SI",
                        "l3_name": "申万三级",
                    },
                ]
            ),
            0,
        )

    monkeypatch.setattr(
        tushare_adapter,
        "_build_tushare_client",
        _fake_build_tushare_client,
    )
    monkeypatch.setattr(tushare_adapter, "fetch_prices", _fake_fetch_prices)
    monkeypatch.setattr(tushare_adapter, "fetch_fundamentals", _fake_fetch_fundamentals)
    monkeypatch.setattr(tushare_adapter, "fetch_adj_factor", _fake_fetch_adj_factor)
    monkeypatch.setattr(tushare_adapter, "fetch_stk_limit", _fake_fetch_stk_limit)
    monkeypatch.setattr(tushare_adapter, "fetch_suspend_status", _fake_fetch_suspend_status)
    monkeypatch.setattr(tushare_adapter, "fetch_st_name_events", _fake_fetch_st_name_events)
    monkeypatch.setattr(tushare_adapter, "fetch_index_membership", _fake_fetch_index_membership)
    monkeypatch.setattr(tushare_adapter, "fetch_moneyflow", _fake_fetch_moneyflow)
    monkeypatch.setattr(
        tushare_adapter,
        "fetch_industry_classification",
        _fake_fetch_industry_classification,
    )
    monkeypatch.setattr(
        tushare_adapter,
        "fetch_industry_membership",
        _fake_fetch_industry_membership,
    )

    summary = generate_real_case_inputs(
        output_dir=tmp_path,
        token="dummy",
        asset_limit=2,
    )
    assert captured_assets == ["000001.SZ", "000002.SZ"]
    assert summary.dataset_version_id is not None
    assert summary.data_root == (tmp_path / "warehouse").resolve()
    prices_path = summary.output_paths["prices"]
    written_prices = pd.read_csv(prices_path)
    assert sorted(written_prices["asset"].unique().tolist()) == ["000001.SZ", "000002.SZ"]
    assert {"open", "high", "low", "pre_close", "volume", "amount", "is_st"} <= set(
        written_prices.columns
    )


def test_fetch_fundamentals_can_skip_roe_for_daily_research_only() -> None:
    class _FakePro:
        def trade_cal(self, **kwargs):
            return pd.DataFrame(
                [
                    {"cal_date": "20240102", "is_open": 1},
                    {"cal_date": "20240103", "is_open": 1},
                ]
            )

        def daily_basic(self, **kwargs):
            return pd.DataFrame(
                [
                    {
                        "ts_code": "000001.SZ",
                        "trade_date": kwargs["trade_date"],
                        "pb": 2.0,
                        "turnover_rate": 1.2,
                    }
                ]
            )

        def fina_indicator(self, **kwargs):
            raise AssertionError("fina_indicator should not be called when include_roe=False")

    result = fetch_fundamentals(
        _FakePro(),
        assets=["000001.SZ"],
        start_date="2024-01-02",
        end_date="2024-01-03",
        include_daily_basic=True,
        include_roe=False,
    )

    assert not result.pb_raw.empty
    assert list(result.roe_raw.columns) == ["asset", "ann_date", "end_date", "roe_value"]
    assert result.roe_raw.empty
    assert result.balance_sheet_raw.empty
    assert result.income_statement_raw.empty
    assert result.cash_flow_statement_raw.empty
    assert result.roe_source_column == "skipped_daily_research_only"


def test_fetch_fundamentals_collects_accounting_statements() -> None:
    class _FakePro:
        def trade_cal(self, **kwargs):
            return pd.DataFrame([{"cal_date": "20240102", "is_open": 1}])

        def daily_basic(self, **kwargs):
            return pd.DataFrame(
                [
                    {
                        "ts_code": "000001.SZ",
                        "trade_date": kwargs["trade_date"],
                        "pb": 2.0,
                        "turnover_rate": 1.2,
                    }
                ]
            )

        def fina_indicator(self, **kwargs):
            return pd.DataFrame(
                [
                    {
                        "ts_code": "000001.SZ",
                        "ann_date": "20240102",
                        "end_date": "20231231",
                        "roe": 10.0,
                    }
                ]
            )

        def balancesheet(self, **kwargs):
            return pd.DataFrame(
                [
                    {
                        "ts_code": "000001.SZ",
                        "ann_date": "20240102",
                        "end_date": "20231231",
                        "goodwill": 1.0,
                        "st_borr": 2.0,
                    }
                ]
            )

        def income(self, **kwargs):
            return pd.DataFrame(
                [
                    {
                        "ts_code": "000001.SZ",
                        "ann_date": "20240102",
                        "end_date": "20231231",
                        "total_revenue": 100.0,
                        "oper_cost": 60.0,
                        "rd_exp": 5.0,
                        "sell_exp": 4.0,
                        "admin_exp": 3.0,
                    }
                ]
            )

        def cashflow(self, **kwargs):
            return pd.DataFrame(
                [
                    {
                        "ts_code": "000001.SZ",
                        "ann_date": "20240102",
                        "end_date": "20231231",
                        "n_cashflow_act": 12.0,
                    }
                ]
            )

    result = fetch_fundamentals(
        _FakePro(),
        assets=["000001.SZ"],
        start_date="2024-01-02",
        end_date="2024-01-03",
        include_daily_basic=True,
        include_roe=True,
    )

    assert not result.balance_sheet_raw.empty
    assert not result.income_statement_raw.empty
    assert not result.cash_flow_statement_raw.empty
    assert "goodwill_balance" in result.balance_sheet_raw.columns
    assert "short_term_borrow" in result.balance_sheet_raw.columns
    assert "operating_revenue_ttm" in result.income_statement_raw.columns
    assert "rd_expense" in result.income_statement_raw.columns
    assert "operating_cash_flow_ttm" in result.cash_flow_statement_raw.columns


def test_fetch_fundamentals_can_skip_daily_basic_for_fundamental_mode() -> None:
    class _FakePro:
        def fina_indicator(self, **kwargs):
            return pd.DataFrame(
                [
                    {
                        "ts_code": "000001.SZ",
                        "ann_date": "20240102",
                        "end_date": "20231231",
                        "roe": 10.0,
                    }
                ]
            )

        def balancesheet(self, **kwargs):
            return pd.DataFrame(
                [
                    {
                        "ts_code": "000001.SZ",
                        "ann_date": "20240102",
                        "end_date": "20231231",
                        "goodwill": 1.0,
                    }
                ]
            )

        def income(self, **kwargs):
            return pd.DataFrame(
                [
                    {
                        "ts_code": "000001.SZ",
                        "ann_date": "20240102",
                        "end_date": "20231231",
                        "total_revenue": 100.0,
                    }
                ]
            )

        def cashflow(self, **kwargs):
            return pd.DataFrame(
                [
                    {
                        "ts_code": "000001.SZ",
                        "ann_date": "20240102",
                        "end_date": "20231231",
                        "n_cashflow_act": 12.0,
                    }
                ]
            )

        def daily_basic(self, **kwargs):
            raise AssertionError("daily_basic should not be called when include_daily_basic=False")

    result = fetch_fundamentals(
        _FakePro(),
        assets=["000001.SZ"],
        start_date="2024-01-02",
        end_date="2024-01-03",
        include_daily_basic=False,
        include_roe=True,
    )

    assert result.pb_raw.empty
    assert not result.roe_raw.empty
    assert not result.balance_sheet_raw.empty
