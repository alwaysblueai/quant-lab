"""Tests for alpha_lab.regime module."""

from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_lab.regime import (
    RegimeConditionalSummary,
    classify_market_regimes,
    compute_regime_conditional,
)
from alpha_lab.research_evaluation_config import RegimeAnalysisConfig


def _make_prices(n_dates: int = 50, n_assets: int = 5, seed: int = 42) -> pd.DataFrame:
    """Generate a simple price panel for testing."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_dates)
    rows = []
    for asset_idx in range(n_assets):
        price = 100.0
        for d in dates:
            rows.append({"date": d, "asset": f"A{asset_idx}", "close": price})
            price *= 1.0 + rng.normal(0.001, 0.02)
    return pd.DataFrame(rows)


def _make_ic_df(dates: pd.DatetimeIndex, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    return pd.DataFrame(
        {
            "date": dates,
            "ic": rng.normal(0.03, 0.05, len(dates)),
        }
    )


def _make_rank_ic_df(dates: pd.DatetimeIndex, seed: int = 43) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    return pd.DataFrame(
        {
            "date": dates,
            "rank_ic": rng.normal(0.02, 0.04, len(dates)),
        }
    )


def _make_ls_df(dates: pd.DatetimeIndex, seed: int = 44) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    return pd.DataFrame(
        {
            "date": dates,
            "long_short_return": rng.normal(0.001, 0.01, len(dates)),
        }
    )


class TestClassifyMarketRegimes:
    def test_returns_regime_labels(self) -> None:
        prices = _make_prices(n_dates=50)
        dates = pd.bdate_range("2020-01-02", periods=49)
        ic_df = _make_ic_df(dates)
        ls_df = _make_ls_df(dates)
        regime_df = classify_market_regimes(ic_df, ls_df, prices)
        assert "direction_regime" in regime_df.columns
        assert "vol_regime" in regime_df.columns
        assert set(regime_df["direction_regime"].unique()) <= {"bull", "bear"}
        assert set(regime_df["vol_regime"].unique()) <= {"high_vol", "low_vol"}

    def test_empty_inputs(self) -> None:
        prices = _make_prices()
        empty_ic = pd.DataFrame(columns=["date", "ic"])
        empty_ls = pd.DataFrame(columns=["date", "long_short_return"])
        regime_df = classify_market_regimes(empty_ic, empty_ls, prices)
        assert regime_df.empty

    def test_only_eval_dates_returned(self) -> None:
        prices = _make_prices(n_dates=50)
        # IC only for first 10 business days
        dates = pd.bdate_range("2020-01-02", periods=10)
        ic_df = _make_ic_df(dates)
        ls_df = _make_ls_df(dates)
        regime_df = classify_market_regimes(ic_df, ls_df, prices)
        assert len(regime_df) <= 10


class TestComputeRegimeConditional:
    def test_basic_output_shape(self) -> None:
        prices = _make_prices(n_dates=60)
        dates = pd.bdate_range("2020-01-02", periods=59)
        ic_df = _make_ic_df(dates)
        rank_ic_df = _make_rank_ic_df(dates)
        ls_df = _make_ls_df(dates)
        regime_df = classify_market_regimes(ic_df, ls_df, prices)
        summary = compute_regime_conditional(ic_df, rank_ic_df, ls_df, regime_df)

        assert isinstance(summary, RegimeConditionalSummary)
        assert isinstance(summary.direction_regimes, tuple)
        assert isinstance(summary.volatility_regimes, tuple)
        assert isinstance(summary.regime_flags, tuple)

    def test_regime_stats_have_correct_labels(self) -> None:
        prices = _make_prices(n_dates=60)
        dates = pd.bdate_range("2020-01-02", periods=59)
        ic_df = _make_ic_df(dates)
        rank_ic_df = _make_rank_ic_df(dates)
        ls_df = _make_ls_df(dates)
        regime_df = classify_market_regimes(ic_df, ls_df, prices)
        summary = compute_regime_conditional(ic_df, rank_ic_df, ls_df, regime_df)

        direction_labels = {rs.regime_label for rs in summary.direction_regimes}
        assert direction_labels <= {"bull", "bear"}
        vol_labels = {rs.regime_label for rs in summary.volatility_regimes}
        assert vol_labels <= {"high_vol", "low_vol"}

    def test_empty_regime_df(self) -> None:
        ic_df = pd.DataFrame(columns=["date", "ic"])
        rank_ic_df = pd.DataFrame(columns=["date", "rank_ic"])
        ls_df = pd.DataFrame(columns=["date", "long_short_return"])
        regime_df = pd.DataFrame(
            columns=["date", "market_return", "direction_regime", "rolling_vol", "vol_regime"]
        )
        summary = compute_regime_conditional(ic_df, rank_ic_df, ls_df, regime_df)
        assert summary.direction_regimes == ()
        assert summary.volatility_regimes == ()
        assert summary.regime_flags == ()

    def test_min_regime_dates_filter(self) -> None:
        """Regimes with fewer dates than min_regime_dates are excluded."""
        prices = _make_prices(n_dates=20)
        dates = pd.bdate_range("2020-01-02", periods=15)
        ic_df = _make_ic_df(dates)
        rank_ic_df = _make_rank_ic_df(dates)
        ls_df = _make_ls_df(dates)
        regime_df = classify_market_regimes(ic_df, ls_df, prices)
        # With min_regime_dates=100, no regime should pass
        config = RegimeAnalysisConfig(min_regime_dates=100)
        summary = compute_regime_conditional(ic_df, rank_ic_df, ls_df, regime_df, config=config)
        assert summary.direction_regimes == ()
        assert summary.volatility_regimes == ()

    def test_negative_ic_triggers_flag(self) -> None:
        """A regime with negative mean IC should trigger a flag."""
        prices = _make_prices(n_dates=60, seed=99)
        dates = pd.bdate_range("2020-01-02", periods=59)
        # Force all IC to be very negative
        ic_df = pd.DataFrame(
            {
                "date": dates,
                "ic": np.full(len(dates), -0.05),
            }
        )
        rank_ic_df = _make_rank_ic_df(dates)
        ls_df = pd.DataFrame(
            {
                "date": dates,
                "long_short_return": np.full(len(dates), -0.005),
            }
        )
        regime_df = classify_market_regimes(ic_df, ls_df, prices)
        summary = compute_regime_conditional(ic_df, rank_ic_df, ls_df, regime_df)
        # Should have regime weakness flags
        assert "regime_conditional_weakness" in summary.regime_flags
