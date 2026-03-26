from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

from alpha_lab.backtest_adapter.loader import load_backtest_input_bundle
from alpha_lab.backtest_adapter.schema import BacktestRunConfig
from alpha_lab.backtest_adapter.vectorbt_adapter import run_vectorbt_backtest
from alpha_lab.experiment import run_factor_experiment
from alpha_lab.factors.momentum import momentum
from alpha_lab.handoff import (
    ExecutionAssumptionsSpec,
    PortfolioConstructionSpec,
    export_handoff_artifact,
)
from alpha_lab.timing import DelaySpec


def _install_fake_vectorbt(monkeypatch) -> type:
    class FakePortfolio:
        last_kwargs: dict[str, object] = {}

        def __init__(
            self,
            *,
            close: pd.DataFrame,
            weights: pd.DataFrame,
            init_cash: float,
            fees: float,
            slippage: float,
            cash_sharing: bool,
            freq: str | None,
        ) -> None:
            _ = cash_sharing, freq
            close = close.astype(float)
            weights = weights.astype(float)
            returns = close.pct_change().fillna(0.0)
            held = weights.shift(1).fillna(0.0)
            turnover = (weights.diff().abs().sum(axis=1).fillna(0.0) / 2.0).astype(float)
            gross_returns = (held * returns).sum(axis=1).astype(float)
            costs = (float(fees) + float(slippage)) * turnover
            self._returns = gross_returns - costs
            self._value = float(init_cash) * (1.0 + self._returns).cumprod()

        @classmethod
        def from_weights(cls, **kwargs):
            cls.last_kwargs = kwargs
            return cls(**kwargs)

        def returns(self) -> pd.Series:
            return self._returns

        def value(self) -> pd.Series:
            return self._value

        def stats(self) -> pd.Series:
            total_return_pct = float(self._value.iloc[-1] / self._value.iloc[0] - 1.0) * 100.0
            return pd.Series({"Total Return [%]": total_return_pct, "N Bars": len(self._value)})

    fake_mod = types.SimpleNamespace(Portfolio=FakePortfolio, __version__="fake-0.1")
    monkeypatch.setitem(sys.modules, "vectorbt", fake_mod)
    return FakePortfolio


def _make_prices(n_assets: int = 4, n_days: int = 70, seed: int = 101) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    rows: list[dict[str, object]] = []
    for asset in [f"A{i}" for i in range(n_assets)]:
        close = 100.0
        for date in dates:
            close *= 1.0 + rng.normal(0.0, 0.01)
            rows.append({"date": date, "asset": asset, "close": close, "open": close * 0.999})
    return pd.DataFrame(rows)


def _bundle(
    tmp_path: Path,
    *,
    execution: ExecutionAssumptionsSpec | None = None,
) -> tuple[Path, pd.DataFrame]:
    prices = _make_prices()
    result = run_factor_experiment(
        prices,
        lambda p: momentum(p, window=5),
        horizon=5,
        delay_spec=DelaySpec.for_horizon(5, execution_delay_periods=1),
    )
    base = prices[["date", "asset"]].drop_duplicates().reset_index(drop=True)
    universe = base.copy()
    universe["in_universe"] = True
    tradability = base.copy()
    tradability["is_tradable"] = True
    export = export_handoff_artifact(
        result,
        output_dir=tmp_path,
        artifact_name="vectorbt_bundle",
        universe_df=universe,
        tradability_df=tradability,
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
        execution_assumptions=execution or ExecutionAssumptionsSpec(
            fill_price_rule="next_close",
            execution_delay_bars=1,
            commission_model="bps",
            slippage_model="fixed_bps",
            trade_when_not_tradable=False,
        ),
    )
    return export.artifact_path, prices


def test_vectorbt_adapter_runs_end_to_end(monkeypatch, tmp_path: Path) -> None:
    _install_fake_vectorbt(monkeypatch)
    artifact_path, prices = _bundle(tmp_path)
    bundle = load_backtest_input_bundle(artifact_path)
    result = run_vectorbt_backtest(
        bundle,
        config=BacktestRunConfig(
            price_df=prices,
            close_column="close",
            open_column="open",
            commission_bps=10.0,
            slippage_bps=5.0,
            output_dir=tmp_path / "adapter_output",
            export_target_weights=True,
            export_series=True,
        ),
    )
    assert result.engine == "vectorbt"
    assert result.experiment_id == bundle.experiment_id
    assert len(result.returns_df) > 0
    assert "total_return" in result.summary
    assert (tmp_path / "adapter_output" / "backtest_summary.json").exists()
    metadata_path = tmp_path / "adapter_output" / "adapter_run_metadata.json"
    assert metadata_path.exists()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["engine"] == "vectorbt"
    assert metadata["mapping_assumptions"]["execution_delay_bars_applied"] == 1
    assert (tmp_path / "adapter_output" / "target_weights.csv").exists()
    first_date = result.executed_weights_df["date"].min()
    first_slice = result.executed_weights_df[result.executed_weights_df["date"] == first_date]
    assert bool((first_slice["target_weight"].abs() < 1e-12).all())


def test_vectorbt_adapter_fee_and_slippage_mapping(monkeypatch, tmp_path: Path) -> None:
    fake_cls = _install_fake_vectorbt(monkeypatch)
    artifact_path, prices = _bundle(tmp_path)
    bundle = load_backtest_input_bundle(artifact_path)
    _ = run_vectorbt_backtest(
        bundle,
        config=BacktestRunConfig(
            price_df=prices,
            commission_bps=12.5,
            slippage_bps=7.5,
            export_summary=False,
        ),
    )
    assert fake_cls.last_kwargs["fees"] == 0.00125
    assert fake_cls.last_kwargs["slippage"] == 0.00075


def test_vectorbt_adapter_warns_on_unsupported_execution_semantics(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _install_fake_vectorbt(monkeypatch)
    artifact_path, prices = _bundle(
        tmp_path,
        execution=ExecutionAssumptionsSpec(
            fill_price_rule="vwap_next_bar",
            execution_delay_bars=1,
            commission_model="flat",
            slippage_model="spread_plus_impact_proxy",
            lot_size_rule="round_to_lot",
            lot_size=100,
            partial_fill_policy="fill_or_kill",
            suspension_policy="defer_trade",
            price_limit_policy="defer_trade",
            allow_same_day_reentry=True,
            trade_when_not_tradable=False,
        ),
    )
    bundle = load_backtest_input_bundle(artifact_path)
    result = run_vectorbt_backtest(
        bundle,
        config=BacktestRunConfig(
            price_df=prices,
            commission_bps=10.0,
            slippage_bps=10.0,
            export_summary=False,
        ),
    )
    warning_codes = {w.code for w in result.warnings}
    assert "unsupported_fill_price_rule" in warning_codes
    assert "unsupported_commission_model" in warning_codes
    assert "unsupported_slippage_model" in warning_codes
    assert "unsupported_lot_size_rule" in warning_codes
    assert "unsupported_partial_fill_policy" in warning_codes
    assert "unsupported_suspension_policy" in warning_codes
    assert "unsupported_price_limit_policy" in warning_codes
    assert "unsupported_same_day_reentry" in warning_codes
