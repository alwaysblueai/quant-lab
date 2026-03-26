from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

from alpha_lab.backtest_adapter.backtrader_adapter import _simulate_execution
from alpha_lab.backtest_adapter.base import run_external_backtest
from alpha_lab.backtest_adapter.loader import load_backtest_input_bundle
from alpha_lab.backtest_adapter.schema import BacktestRunConfig
from alpha_lab.experiment import run_factor_experiment
from alpha_lab.factors.momentum import momentum
from alpha_lab.handoff import (
    ExecutionAssumptionsSpec,
    PortfolioConstructionSpec,
    export_handoff_artifact,
)
from alpha_lab.timing import DelaySpec


def _install_fake_backtrader(monkeypatch) -> None:
    fake = types.SimpleNamespace(__version__="fake-backtrader-1.0")
    monkeypatch.setitem(sys.modules, "backtrader", fake)


def _make_prices(n_assets: int = 4, n_days: int = 70, seed: int = 123) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    rows: list[dict[str, object]] = []
    for asset in [f"A{i}" for i in range(n_assets)]:
        close = 100.0
        for date in dates:
            close *= 1.0 + rng.normal(0.0, 0.01)
            rows.append({"date": date, "asset": asset, "close": close, "open": close * 1.001})
    return pd.DataFrame(rows)


def _factor_fn(prices: pd.DataFrame) -> pd.DataFrame:
    return momentum(prices, window=5)


def _bundle(
    tmp_path: Path,
    *,
    execution: ExecutionAssumptionsSpec | None = None,
    tradability_override: pd.DataFrame | None = None,
    exclusion_reasons: pd.DataFrame | None = None,
) -> tuple[Path, pd.DataFrame]:
    prices = _make_prices()
    result = run_factor_experiment(
        prices,
        _factor_fn,
        horizon=5,
        delay_spec=DelaySpec.for_horizon(5, execution_delay_periods=1),
    )
    keys = prices[["date", "asset"]].drop_duplicates().reset_index(drop=True)
    universe = keys.copy()
    universe["in_universe"] = True
    if tradability_override is None:
        tradability = keys.copy()
        tradability["is_tradable"] = True
    else:
        tradability = tradability_override.copy()
    export = export_handoff_artifact(
        result,
        output_dir=tmp_path,
        artifact_name="backtrader_bundle",
        universe_df=universe,
        tradability_df=tradability,
        exclusion_reasons_df=exclusion_reasons,
        portfolio_construction=PortfolioConstructionSpec(
            signal_name="momentum_5d",
            long_short=True,
            top_k=1,
            bottom_k=1,
            weight_method="equal",
            gross_limit=1.0,
            net_limit=0.0,
            cash_buffer=0.0,
            max_weight=1.0,
        ),
        execution_assumptions=execution
        or ExecutionAssumptionsSpec(
            fill_price_rule="next_open",
            execution_delay_bars=1,
            commission_model="bps",
            slippage_model="fixed_bps",
            trade_when_not_tradable=False,
        ),
    )
    return export.artifact_path, prices


def test_backtrader_adapter_smoke(monkeypatch, tmp_path: Path) -> None:
    _install_fake_backtrader(monkeypatch)
    artifact_path, prices = _bundle(tmp_path)
    bundle = load_backtest_input_bundle(artifact_path)
    result = run_external_backtest(
        bundle,
        config=BacktestRunConfig(
            engine="backtrader",
            price_df=prices,
            close_column="close",
            open_column="open",
            output_dir=tmp_path / "bt_output",
            export_target_weights=True,
        ),
    )
    assert result.engine == "backtrader"
    assert len(result.returns_df) > 0
    assert "adapter_run_metadata" in (tmp_path / "bt_output" / "backtest_summary.json").read_text(
        encoding="utf-8"
    )
    assert (tmp_path / "bt_output" / "adapter_run_metadata.json").exists()


def test_backtrader_lot_size_rounding(monkeypatch, tmp_path: Path) -> None:
    _install_fake_backtrader(monkeypatch)
    artifact_path, prices = _bundle(
        tmp_path,
        execution=ExecutionAssumptionsSpec(
            fill_price_rule="next_open",
            execution_delay_bars=1,
            commission_model="bps",
            slippage_model="fixed_bps",
            lot_size_rule="round_to_lot",
            lot_size=100,
            trade_when_not_tradable=False,
        ),
    )
    bundle = load_backtest_input_bundle(artifact_path)
    result = run_external_backtest(
        bundle,
        config=BacktestRunConfig(engine="backtrader", price_df=prices, open_column="open"),
    )
    if result.orders_df is not None and not result.orders_df.empty:
        rounded = result.orders_df["size"].abs()
        assert bool(((rounded % 100) < 1e-12).all())


def test_backtrader_non_tradable_and_price_limit_skip(monkeypatch, tmp_path: Path) -> None:
    _install_fake_backtrader(monkeypatch)
    prices = _make_prices()
    keys = prices[["date", "asset"]].drop_duplicates().reset_index(drop=True)
    tradability = keys.copy()
    all_dates = sorted(pd.to_datetime(tradability["date"]).unique())
    warmup_dates = set(all_dates[:15])
    tradability["is_tradable"] = pd.to_datetime(tradability["date"]).isin(warmup_dates)
    exclusion = keys.copy()
    exclusion["reason"] = "price_limit_lock"

    artifact_path, _ = _bundle(
        tmp_path,
        tradability_override=tradability,
        exclusion_reasons=exclusion,
    )
    bundle = load_backtest_input_bundle(artifact_path)
    result = run_external_backtest(
        bundle,
        config=BacktestRunConfig(engine="backtrader", price_df=prices, open_column="open"),
    )
    assert result.orders_df is not None
    assert result.skipped_orders_df is not None
    assert not result.skipped_orders_df.empty
    assert "skip_trade" in set(result.skipped_orders_df["policy"].astype(str))
    assert "price_limit_locked" in set(result.skipped_orders_df["reason"].astype(str))
    assert "missing_exclusion_reason" not in set(result.skipped_orders_df["reason"].astype(str))
    assert "source_reason" in set(result.skipped_orders_df.columns)
    assert "reason_code" in set(result.skipped_orders_df.columns)


def test_backtrader_execution_delay_reflected(monkeypatch, tmp_path: Path) -> None:
    _install_fake_backtrader(monkeypatch)
    artifact_path, prices = _bundle(
        tmp_path,
        execution=ExecutionAssumptionsSpec(
            fill_price_rule="next_open",
            execution_delay_bars=2,
            commission_model="bps",
            slippage_model="fixed_bps",
            trade_when_not_tradable=False,
        ),
    )
    bundle = load_backtest_input_bundle(artifact_path)
    result = run_external_backtest(
        bundle,
        config=BacktestRunConfig(engine="backtrader", price_df=prices, open_column="open"),
    )
    first_date = result.executed_weights_df["date"].min()
    first_slice = result.executed_weights_df[result.executed_weights_df["date"] == first_date]
    assert bool((first_slice["target_weight"].abs() < 1e-12).all())


def test_backtrader_same_day_reentry_blocked_in_simulation() -> None:
    dates = pd.date_range("2024-01-01", periods=3, freq="B")
    delayed_targets = pd.DataFrame({"A": [1.0, -1.0, -1.0]}, index=dates)
    prices = pd.DataFrame({"A": [10.0, 10.0, 10.0]}, index=dates)
    universe = pd.DataFrame({"A": [True, True, True]}, index=dates)
    tradability = pd.DataFrame({"A": [True, True, True]}, index=dates)
    class _Execution:
        lot_size_rule = "none"
        lot_size = None
        allow_same_day_reentry = False
        trade_when_not_tradable = False
        suspension_policy = "skip_trade"
        price_limit_policy = "skip_trade"
    class _Bundle:
        execution_assumptions = _Execution()
    replay = _simulate_execution(
        delayed_targets=delayed_targets,
        mark_price_matrix=prices,
        execution_price_matrix=prices,
        universe_matrix=universe,
        tradability_matrix=tradability,
        exclusion_reason_map={},
        bundle=_Bundle(),  # type: ignore[arg-type]
        initial_cash=1000.0,
        commission_rate=0.0,
        slippage_rate=0.0,
    )
    skipped = replay["skipped_orders_df"]
    assert isinstance(skipped, pd.DataFrame)
    assert "same_day_reentry_blocked" in set(skipped["policy"].astype(str))


def test_backtrader_no_false_skipped_orders_when_no_action_needed() -> None:
    dates = pd.date_range("2024-01-01", periods=3, freq="B")
    delayed_targets = pd.DataFrame({"A": [0.0, 0.0, 0.0]}, index=dates)
    prices = pd.DataFrame({"A": [10.0, 10.0, 10.0]}, index=dates)
    universe = pd.DataFrame({"A": [True, True, True]}, index=dates)
    tradability = pd.DataFrame({"A": [False, False, False]}, index=dates)

    class _Execution:
        lot_size_rule = "none"
        lot_size = None
        allow_same_day_reentry = False
        trade_when_not_tradable = False
        suspension_policy = "skip_trade"
        price_limit_policy = "skip_trade"

    class _Bundle:
        execution_assumptions = _Execution()

    replay = _simulate_execution(
        delayed_targets=delayed_targets,
        mark_price_matrix=prices,
        execution_price_matrix=prices,
        universe_matrix=universe,
        tradability_matrix=tradability,
        exclusion_reason_map={},
        bundle=_Bundle(),  # type: ignore[arg-type]
        initial_cash=1000.0,
        commission_rate=0.0,
        slippage_rate=0.0,
    )
    skipped = replay["skipped_orders_df"]
    assert isinstance(skipped, pd.DataFrame)
    assert skipped.empty


def test_backtrader_suspension_reason_is_propagated() -> None:
    dates = pd.date_range("2024-01-01", periods=3, freq="B")
    delayed_targets = pd.DataFrame({"A": [1.0, 1.0, 1.0]}, index=dates)
    prices = pd.DataFrame({"A": [10.0, 10.0, 10.0]}, index=dates)
    universe = pd.DataFrame({"A": [True, True, True]}, index=dates)
    tradability = pd.DataFrame({"A": [True, False, False]}, index=dates)

    class _Execution:
        lot_size_rule = "none"
        lot_size = None
        allow_same_day_reentry = False
        trade_when_not_tradable = False
        suspension_policy = "skip_trade"
        price_limit_policy = "skip_trade"

    class _Bundle:
        execution_assumptions = _Execution()

    replay = _simulate_execution(
        delayed_targets=delayed_targets,
        mark_price_matrix=prices,
        execution_price_matrix=prices,
        universe_matrix=universe,
        tradability_matrix=tradability,
        exclusion_reason_map={
            (pd.Timestamp(dates[1]), "A"): "halted_trading",
            (pd.Timestamp(dates[2]), "A"): "halted_trading",
        },
        bundle=_Bundle(),  # type: ignore[arg-type]
        initial_cash=1000.0,
        commission_rate=0.0,
        slippage_rate=0.0,
    )
    skipped = replay["skipped_orders_df"]
    assert isinstance(skipped, pd.DataFrame)
    assert not skipped.empty
    assert "halted_trading" in set(skipped["reason"].astype(str))
