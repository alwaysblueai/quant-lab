from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import alpha_lab.decay as decay
from alpha_lab.evaluation import compute_ic, compute_ic_summary, compute_rank_ic
from alpha_lab.labels import forward_return


def _factor_panel(*, n_dates: int = 40, n_assets: int = 20, seed: int = 20260424) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-02", periods=n_dates)
    assets = [f"A{idx:03d}" for idx in range(n_assets)]
    return pd.DataFrame(
        {
            "date": np.repeat(dates.values, n_assets),
            "asset": np.tile(assets, n_dates),
            "factor": "bench_factor",
            "value": rng.normal(size=n_dates * n_assets),
        }
    )


def _prices_panel(*, n_dates: int = 30, n_assets: int = 10, seed: int = 17) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-02", periods=n_dates)
    assets = [f"A{idx:03d}" for idx in range(n_assets)]
    rows: list[dict[str, object]] = []
    for asset in assets:
        price = 100.0 + rng.normal()
        for date in dates:
            price *= 1.0 + rng.normal(0.0005, 0.015)
            rows.append({"date": date, "asset": asset, "close": price})
    frame = pd.DataFrame(rows)
    drop_mask = rng.random(len(frame)) < 0.06
    return frame.loc[~drop_mask].sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _ic_decay_reference(
    factor_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    horizons: tuple[int, ...],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for horizon in horizons:
        labels = forward_return(prices_df, horizon=horizon)
        merged_eval = (
            factor_df[["date", "asset", "value"]]
            .rename(columns={"value": "value_factor"})
            .merge(
                labels[["date", "asset", "value"]].rename(columns={"value": "value_label"}),
                on=["date", "asset"],
                how="inner",
                validate="one_to_one",
            )
        )
        ic_df = compute_ic(factor_df, labels, merged_pairs=merged_eval)
        rank_ic_df = compute_rank_ic(factor_df, labels, merged_pairs=merged_eval)

        ic_vals = ic_df["ic"] if not ic_df.empty else pd.Series(dtype=float)
        rank_ic_vals = rank_ic_df["rank_ic"] if not rank_ic_df.empty else pd.Series(dtype=float)
        summary = compute_ic_summary(ic_vals)
        clean_rank = rank_ic_vals.dropna()
        rows.append(
            {
                "horizon": horizon,
                "mean_ic": summary["mean_ic"],
                "mean_rank_ic": (
                    float(clean_rank.mean()) if len(clean_rank) > 0 else float("nan")
                ),
                "ic_ir": summary["ic_ir"],
                "t_stat": summary["t_stat"],
                "p_value": summary["p_value"],
                "n_dates": summary["n_obs"],
            }
        )
    return pd.DataFrame(rows)


def _autocorr_reference(factor_df: pd.DataFrame, lags: tuple[int, ...]) -> pd.DataFrame:
    wide = factor_df.pivot(index="date", columns="asset", values="value").sort_index()
    values = wide.to_numpy(dtype=float, copy=False)
    rows: list[dict[str, object]] = []
    for lag in lags:
        lag_int = int(lag)
        if lag_int >= len(wide):
            rows.append(
                {
                    "lag": lag,
                    "mean_autocorr": float("nan"),
                    "std_autocorr": float("nan"),
                    "n_dates": 0,
                }
            )
            continue
        corrs = [
            decay._spearman_corr_on_overlap(
                values[i],
                values[i - lag_int],
                min_assets=decay.AUTOCORR_MIN_ASSETS,
            )
            for i in range(lag_int, len(wide))
        ]
        clean = np.asarray([c for c in corrs if np.isfinite(c)], dtype=float)
        rows.append(
            {
                "lag": lag,
                "mean_autocorr": float(np.mean(clean)) if clean.size else float("nan"),
                "std_autocorr": (
                    float(np.std(clean, ddof=1)) if int(clean.size) > 1 else float("nan")
                ),
                "n_dates": int(clean.size),
            }
        )
    return pd.DataFrame(rows)


def test_autocorr_fast_path_equivalence_full_coverage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factor_df = _factor_panel()
    lags = (1, 5, 10)

    def fail_fallback(_wide: pd.DataFrame, _lags: tuple[int, ...]) -> pd.DataFrame:
        raise AssertionError("fallback should not run for full-coverage panels")

    monkeypatch.setattr(decay, "_autocorr_per_pair_rank_fallback", fail_fallback)

    actual = decay.compute_factor_autocorrelation(factor_df, lags=lags)
    expected = _autocorr_reference(factor_df, lags)

    assert_frame_equal(actual, expected, check_dtype=False, atol=1e-10, rtol=1e-10)


def test_ic_decay_dense_panel_uses_wide_fast_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prices = _prices_panel(n_dates=28, n_assets=9)
    dates = pd.Index(pd.to_datetime(prices["date"]).drop_duplicates()).sort_values()
    assets = pd.Index(prices["asset"].drop_duplicates()).sort_values()
    complete_prices = (
        prices.pivot(index="date", columns="asset", values="close")
        .reindex(index=dates, columns=assets)
        .ffill()
        .bfill()
        .stack()
        .rename("close")
        .reset_index()
    )
    rng = np.random.default_rng(20260428)
    factor_df = complete_prices[["date", "asset"]].copy()
    factor_df["factor"] = "dense_factor"
    factor_df["value"] = rng.normal(size=len(factor_df))
    horizons = (1, 3, 5)
    expected = _ic_decay_reference(factor_df, complete_prices, horizons)

    def fail_forward_return(*_args: object, **_kwargs: object) -> pd.DataFrame:
        raise AssertionError("dense fast path should not call forward_return")

    monkeypatch.setattr(decay, "forward_return", fail_forward_return)

    actual = decay.compute_ic_decay(factor_df, complete_prices, horizons=horizons)

    assert_frame_equal(actual, expected, check_dtype=False, atol=1e-12, rtol=1e-12)


def test_ic_decay_dense_panel_with_missing_factor_uses_public_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prices = _prices_panel(n_dates=28, n_assets=9)
    dates = pd.Index(pd.to_datetime(prices["date"]).drop_duplicates()).sort_values()
    assets = pd.Index(prices["asset"].drop_duplicates()).sort_values()
    complete_prices = (
        prices.pivot(index="date", columns="asset", values="close")
        .reindex(index=dates, columns=assets)
        .ffill()
        .bfill()
        .stack()
        .rename("close")
        .reset_index()
    )
    rng = np.random.default_rng(20260429)
    factor_df = complete_prices[["date", "asset"]].copy()
    factor_df["factor"] = "dense_factor_with_gap"
    factor_df["value"] = rng.normal(size=len(factor_df))
    factor_df.loc[0, "value"] = np.nan
    horizons = (1, 3)
    expected = _ic_decay_reference(factor_df, complete_prices, horizons)
    called_horizons: list[int] = []

    def wrapped_forward_return(prices_df: pd.DataFrame, *, horizon: int) -> pd.DataFrame:
        called_horizons.append(int(horizon))
        return forward_return(prices_df, horizon=horizon)

    monkeypatch.setattr(decay, "forward_return", wrapped_forward_return)

    actual = decay.compute_ic_decay(factor_df, complete_prices, horizons=horizons)

    assert called_horizons == list(horizons)
    assert_frame_equal(actual, expected, check_dtype=False, atol=1e-12, rtol=1e-12)


def test_ic_decay_matches_public_ic_metric_path_on_sparse_panel() -> None:
    prices = _prices_panel()
    rng = np.random.default_rng(20260427)
    factor_df = prices[["date", "asset"]].copy()
    factor_df["factor"] = "sparse_factor"
    factor_df["value"] = rng.normal(size=len(factor_df))
    factor_df.loc[rng.random(len(factor_df)) < 0.08, "value"] = np.nan
    factor_df = factor_df.sample(frac=1.0, random_state=11).reset_index(drop=True)
    horizons = (1, 2, 5)

    actual = decay.compute_ic_decay(factor_df, prices, horizons=horizons)
    expected = _ic_decay_reference(factor_df, prices, horizons)

    assert_frame_equal(actual, expected, check_dtype=False, atol=1e-12, rtol=1e-12)


def test_autocorr_high_coverage_missing_uses_overlap_ranking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factor_df = _factor_panel(n_dates=16, n_assets=300)
    factor_df.loc[0, "value"] = np.nan
    lags = (1, 2, 5)

    def fail_fast_path(_wide: pd.DataFrame, _lags: tuple[int, ...]) -> pd.DataFrame:
        raise AssertionError("pre-ranked fast path is only exact for complete panels")

    monkeypatch.setattr(decay, "_autocorr_preranked_fast_path", fail_fast_path)

    actual = decay.compute_factor_autocorrelation(factor_df, lags=lags)
    expected = _autocorr_reference(factor_df, lags)

    assert_frame_equal(actual, expected, check_dtype=False, atol=1e-12, rtol=1e-12)


def test_autocorr_fallback_on_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    factor_df = _factor_panel()
    rng = np.random.default_rng(7)
    factor_df.loc[rng.random(len(factor_df)) < 0.05, "value"] = np.nan
    lags = (1, 5, 10)
    called = False
    original = decay._autocorr_per_pair_rank_fallback

    def wrapped_fallback(wide: pd.DataFrame, lags_arg: tuple[int, ...]) -> pd.DataFrame:
        nonlocal called
        called = True
        return original(wide, lags_arg)

    monkeypatch.setattr(decay, "_autocorr_per_pair_rank_fallback", wrapped_fallback)

    actual = decay.compute_factor_autocorrelation(factor_df, lags=lags)
    expected = _autocorr_reference(factor_df, lags)

    assert called
    assert_frame_equal(actual, expected, check_dtype=False, atol=1e-12, rtol=1e-12)
