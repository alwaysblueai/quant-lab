"""Tests for alpha_lab.tail_risk module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_lab.tail_risk import compute_tail_risk


def _make_ls_df(returns: list[float]) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=len(returns), freq="B")
    return pd.DataFrame({"date": dates, "long_short_return": returns})


class TestComputeTailRisk:
    def test_empty_df(self) -> None:
        result = compute_tail_risk(pd.DataFrame())
        assert np.isnan(result.max_drawdown)
        assert result.max_drawdown_duration == -1
        assert result.max_consecutive_loss_days == -1
        assert result.n_total_dates == 0

    def test_all_positive_returns(self) -> None:
        df = _make_ls_df([0.01, 0.02, 0.01, 0.03, 0.01])
        result = compute_tail_risk(df)
        assert result.max_drawdown == 0.0
        assert result.max_drawdown_duration == 0
        assert result.max_consecutive_loss_days == 0
        assert result.n_loss_dates == 0
        assert result.n_total_dates == 5
        assert np.isnan(result.calmar_ratio)  # dd=0 → nan

    def test_single_drawdown(self) -> None:
        # +1%, -2%, -1%, +3%, +1%
        # cum: 0.01, -0.01, -0.02, 0.01, 0.02
        # peak: 0.01, 0.01, 0.01, 0.01, 0.02
        # dd:   0.00, 0.02, 0.03, 0.00, 0.00
        df = _make_ls_df([0.01, -0.02, -0.01, 0.03, 0.01])
        result = compute_tail_risk(df)
        assert pytest.approx(result.max_drawdown, abs=1e-10) == 0.03
        assert result.max_drawdown_duration == 2  # peak at idx 0, trough at idx 2
        assert result.max_consecutive_loss_days == 2
        assert result.n_loss_dates == 2

    def test_var_cvar(self) -> None:
        np.random.seed(42)
        returns = list(np.random.normal(0.001, 0.02, 100))
        df = _make_ls_df(returns)
        result = compute_tail_risk(df)
        assert result.var_5 < 0  # 5th percentile should be negative for normal
        assert result.cvar_5 <= result.var_5  # CVaR <= VaR (more extreme)
        assert result.n_total_dates == 100

    def test_calmar_ratio(self) -> None:
        df = _make_ls_df([0.02, -0.01, 0.02, -0.005, 0.01])
        result = compute_tail_risk(df)
        mean_ret = np.mean([0.02, -0.01, 0.02, -0.005, 0.01])
        assert result.max_drawdown > 0
        expected_calmar = mean_ret / result.max_drawdown
        assert pytest.approx(result.calmar_ratio, rel=1e-6) == expected_calmar

    def test_all_nan_returns(self) -> None:
        df = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=3, freq="B"),
                "long_short_return": [float("nan"), float("nan"), float("nan")],
            }
        )
        result = compute_tail_risk(df)
        assert np.isnan(result.max_drawdown)
        assert result.n_total_dates == 0

    def test_consecutive_loss_streak(self) -> None:
        # 3 losses, then 1 gain, then 2 losses
        df = _make_ls_df([-0.01, -0.02, -0.01, 0.05, -0.01, -0.02])
        result = compute_tail_risk(df)
        assert result.max_consecutive_loss_days == 3

    def test_zero_returns_count_as_loss(self) -> None:
        # Zero returns are <= 0, so they count for consecutive loss
        df = _make_ls_df([0.0, 0.0, 0.0, 0.01])
        result = compute_tail_risk(df)
        assert result.max_consecutive_loss_days == 3
