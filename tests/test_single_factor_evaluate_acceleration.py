from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import alpha_lab.real_cases.single_factor.evaluate as single_factor_evaluate
from alpha_lab.experiment import run_factor_experiment
from alpha_lab.factors.momentum import momentum
from alpha_lab.research_evaluation_config import get_research_evaluation_config


def _make_prices(n_assets: int = 8, n_days: int = 70, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    rows: list[dict[str, object]] = []
    for idx, asset in enumerate(f"A{i}" for i in range(n_assets)):
        close = 100.0 + idx
        for date in dates:
            ret = rng.normal(0.0005, 0.012)
            open_ = close * (1.0 + rng.normal(0.0, 0.002))
            close = max(close * (1.0 + ret), 1.0)
            high = max(open_, close)
            low = min(open_, close)
            rows.append(
                {
                    "date": date,
                    "asset": asset,
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": 1000.0,
                }
            )
    return pd.DataFrame(rows)


def _base_result(prices: pd.DataFrame):
    factor_df = momentum(prices[["date", "asset", "close"]], window=5)
    result = run_factor_experiment(
        prices[["date", "asset", "close"]],
        lambda _p: factor_df.copy(),
        horizon=5,
        n_quantiles=5,
        allow_full_sample_evaluation=True,
    )
    return factor_df, result


def test_lightweight_variant_summary_matches_full_experiment() -> None:
    prices = _make_prices()
    factor_df, result = _base_result(prices)

    summary = single_factor_evaluate._evaluate_variant_lightweight(
        factor_df=factor_df,
        label_df=result.label_df,
        n_quantiles=5,
    )

    assert summary.mean_ic == pytest.approx(result.summary.mean_ic)
    assert summary.mean_long_short_return == pytest.approx(result.summary.mean_long_short_return)
    assert summary.long_short_ir == pytest.approx(result.summary.long_short_ir)


def test_param_and_baseline_sensitivity_do_not_rerun_full_experiment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prices = _make_prices()
    factor_df, result = _base_result(prices)
    metrics: dict[str, object] = {"mean_ic": float(result.summary.mean_ic)}
    cfg = get_research_evaluation_config("default_research")

    def _fail_run_factor_experiment(*args, **kwargs):
        del args, kwargs
        raise AssertionError("run_factor_experiment should not be used by lightweight diagnostics")

    monkeypatch.setattr(
        single_factor_evaluate, "run_factor_experiment", _fail_run_factor_experiment
    )

    single_factor_evaluate._merge_param_sensitivity_metrics(
        metrics,
        prices=prices[["date", "asset", "close"]],
        factor_df=factor_df,
        horizon=5,
        base_n_quantiles=5,
        evaluation_config=cfg,
        label_df=result.label_df,
        enabled=True,
    )
    single_factor_evaluate._merge_baseline_factor_comparison_metrics(
        metrics,
        prices=prices[["date", "asset", "close"]],
        factor_df=factor_df,
        horizon=5,
        n_quantiles=5,
        evaluation_config=cfg,
        label_df=result.label_df,
        enabled=True,
    )

    assert metrics["param_sensitivity_n_variants"] == 2
    assert np.isfinite(float(metrics["baseline_momentum_mean_ic"]))
    assert np.isfinite(float(metrics["baseline_reversal_mean_ic"]))
    assert int(metrics["baseline_suite_count"]) >= 8
    assert int(metrics["baseline_suite_evaluated_count"]) >= 6
    assert "mom_20d" in metrics["baseline_suite_evaluated_names"]
    assert "rev_5d" in metrics["baseline_suite_evaluated_names"]
    assert metrics["baseline_suite_best_name"]
    assert np.isfinite(float(metrics["baseline_suite_best_mean_ic"]))


def test_lag_sensitivity_reuses_base_run_for_lag_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prices = _make_prices()
    factor_df, result = _base_result(prices)
    metrics: dict[str, object] = {}
    cfg = get_research_evaluation_config("default_research")
    original = single_factor_evaluate._evaluate_variant_lightweight
    calls: list[int] = []

    def _wrapped_variant_summary(
        *,
        factor_df: pd.DataFrame,
        label_df: pd.DataFrame,
        n_quantiles: int,
    ):
        del label_df, n_quantiles
        calls.append(int(factor_df["date"].nunique()))
        return original(
            factor_df=factor_df,
            label_df=result.label_df,
            n_quantiles=5,
        )

    monkeypatch.setattr(
        single_factor_evaluate.pnl_attribution,
        "_evaluate_variant_lightweight",
        _wrapped_variant_summary,
    )

    lag_df = single_factor_evaluate._merge_signal_lag_sensitivity_metrics(
        metrics,
        prices=prices[["date", "asset", "close"]],
        factor_df=factor_df,
        horizon=5,
        n_quantiles=5,
        evaluation_config=cfg,
        label_df=result.label_df,
        base_result=result,
        lags=(0, 1, 2),
        enabled=True,
    )

    assert len(calls) == 2
    assert metrics["lag_sensitivity_mean_ic_lag_0"] == pytest.approx(result.summary.mean_ic)
    assert metrics["lag_sensitivity_long_short_ir_lag_0"] == pytest.approx(
        result.summary.long_short_ir
    )
    assert list(lag_df["lag"]) == [0, 1, 2]
