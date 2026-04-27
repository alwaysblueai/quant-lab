"""Phase 1 Foundation tests: IC t-stat, IC decay, factor redundancy,
corporate actions, PIT fundamentals."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats as scipy_stats

# ═══════════════════════════════════════════════════════════════════════════
# 3.1  IC t-Statistic & Significance
# ═══════════════════════════════════════════════════════════════════════════


class TestComputeICSummary:
    """Verify compute_ic_summary against hand-calculated values."""

    def test_known_values(self):
        from alpha_lab.evaluation import compute_ic_summary

        ic = pd.Series([0.10, 0.20, 0.30])
        result = compute_ic_summary(ic)
        assert result["n_obs"] == 3
        assert result["mean_ic"] == pytest.approx(0.20)
        std = ic.std(ddof=1)
        assert result["std_ic"] == pytest.approx(float(std))
        expected_t = 0.20 / (float(std) / math.sqrt(3))
        assert result["t_stat"] == pytest.approx(expected_t)
        expected_p = float(2 * scipy_stats.t.sf(abs(expected_t), df=2))
        assert result["p_value"] == pytest.approx(expected_p)
        assert result["ic_ir"] == pytest.approx(0.20 / float(std))

    def test_single_observation(self):
        from alpha_lab.evaluation import compute_ic_summary

        result = compute_ic_summary(pd.Series([0.05]))
        assert result["n_obs"] == 1
        assert result["mean_ic"] == pytest.approx(0.05)
        assert math.isnan(result["t_stat"])
        assert math.isnan(result["p_value"])
        assert math.isnan(result["ic_ir"])

    def test_empty_series(self):
        from alpha_lab.evaluation import compute_ic_summary

        result = compute_ic_summary(pd.Series(dtype=float))
        assert result["n_obs"] == 0
        assert math.isnan(result["mean_ic"])
        assert math.isnan(result["t_stat"])
        assert math.isnan(result["p_value"])

    def test_nan_values_dropped(self):
        from alpha_lab.evaluation import compute_ic_summary

        ic = pd.Series([0.10, float("nan"), 0.30])
        result = compute_ic_summary(ic)
        assert result["n_obs"] == 2
        assert result["mean_ic"] == pytest.approx(0.20)

    def test_constant_series_nan_t_stat(self):
        from alpha_lab.evaluation import compute_ic_summary

        ic = pd.Series([0.05, 0.05, 0.05])
        result = compute_ic_summary(ic)
        assert result["n_obs"] == 3
        assert result["mean_ic"] == pytest.approx(0.05)
        assert math.isnan(result["t_stat"])  # std=0

    def test_scipy_cross_validation(self):
        """Cross-validate against scipy.stats.ttest_1samp."""
        from alpha_lab.evaluation import compute_ic_summary

        rng = np.random.RandomState(42)
        ic = pd.Series(rng.normal(0.03, 0.05, size=100))
        result = compute_ic_summary(ic)
        scipy_t, scipy_p = scipy_stats.ttest_1samp(ic.values, 0.0)
        assert result["t_stat"] == pytest.approx(scipy_t, rel=1e-10)
        assert result["p_value"] == pytest.approx(scipy_p, rel=1e-10)

    def test_experiment_summary_has_fields(self):
        """ExperimentSummary should expose ic_t_stat and ic_p_value."""
        import dataclasses

        from alpha_lab.experiment import ExperimentSummary

        field_names = {f.name for f in dataclasses.fields(ExperimentSummary)}
        assert "ic_t_stat" in field_names
        assert "ic_p_value" in field_names


# ═══════════════════════════════════════════════════════════════════════════
# 3.2  IC Decay Analysis
# ═══════════════════════════════════════════════════════════════════════════


def _make_prices(
    dates: list[str], assets: list[str], close_map: dict[str, list[float]]
) -> pd.DataFrame:
    """Build a price panel from a {asset: [close_values]} mapping."""
    rows = []
    for asset in assets:
        for i, d in enumerate(dates):
            rows.append({"date": pd.Timestamp(d), "asset": asset, "close": close_map[asset][i]})
    return pd.DataFrame(rows)


class TestICDecay:
    """Verify IC decay analysis module."""

    def test_decay_returns_all_horizons(self):
        from alpha_lab.decay import compute_ic_decay

        dates = [f"2024-01-{d:02d}" for d in range(1, 25)]
        assets = ["A", "B", "C"]
        rng = np.random.RandomState(42)
        prices = _make_prices(
            dates,
            assets,
            {a: (100 + rng.randn(len(dates)).cumsum()).tolist() for a in assets},
        )
        factor_df = pd.DataFrame(
            {
                "date": pd.to_datetime(dates * len(assets)),
                "asset": [a for a in assets for _ in dates],
                "factor": ["test_factor"] * len(dates) * len(assets),
                "value": rng.randn(len(dates) * len(assets)),
            }
        )
        horizons = (1, 2, 5)
        result = compute_ic_decay(factor_df, prices, horizons=horizons)
        assert set(result["horizon"].tolist()) == set(horizons)
        assert "mean_ic" in result.columns
        assert "t_stat" in result.columns
        assert "n_dates" in result.columns

    def test_decay_single_horizon_matches_compute_ic(self):
        """IC decay at horizon=1 should match direct compute_ic."""
        from alpha_lab.decay import compute_ic_decay
        from alpha_lab.evaluation import compute_ic
        from alpha_lab.labels import forward_return

        dates = [f"2024-01-{d:02d}" for d in range(1, 15)]
        assets = ["A", "B", "C"]
        rng = np.random.RandomState(7)
        prices = _make_prices(
            dates,
            assets,
            {a: (100 + rng.randn(len(dates)).cumsum()).tolist() for a in assets},
        )
        factor_df = pd.DataFrame(
            {
                "date": pd.to_datetime(dates * len(assets)),
                "asset": [a for a in assets for _ in dates],
                "factor": ["f"] * len(dates) * len(assets),
                "value": rng.randn(len(dates) * len(assets)),
            }
        )
        decay = compute_ic_decay(factor_df, prices, horizons=(1,))
        labels = forward_return(prices, horizon=1)
        ic_df = compute_ic(factor_df, labels)
        expected_mean = float(ic_df["ic"].dropna().mean())
        actual_mean = float(decay.loc[decay["horizon"] == 1, "mean_ic"].iloc[0])
        assert actual_mean == pytest.approx(expected_mean)

    def test_estimate_ic_half_life_exact_crossing(self):
        from alpha_lab.decay import estimate_ic_half_life

        decay = pd.DataFrame(
            {
                "horizon": [1, 3, 5],
                "mean_ic": [0.08, 0.04, 0.02],
            }
        )
        summary = estimate_ic_half_life(decay)
        assert summary["ic_half_life_status"] == "estimated"
        assert float(summary["ic_half_life_horizon"]) == pytest.approx(3.0)
        assert bool(summary["ic_half_life_not_reached"]) is False

    def test_estimate_ic_half_life_interpolates_between_horizons(self):
        from alpha_lab.decay import estimate_ic_half_life

        decay = pd.DataFrame(
            {
                "horizon": [1, 3, 5],
                "mean_ic": [0.10, 0.07, 0.04],
            }
        )
        summary = estimate_ic_half_life(decay)
        assert summary["ic_half_life_status"] == "estimated"
        assert float(summary["ic_half_life_horizon"]) == pytest.approx(13.0 / 3.0)
        assert float(summary["ic_half_life_baseline_horizon"]) == pytest.approx(1.0)
        assert float(summary["ic_half_life_baseline_abs_mean_ic"]) == pytest.approx(0.10)

    def test_estimate_ic_half_life_marks_not_reached(self):
        from alpha_lab.decay import estimate_ic_half_life

        decay = pd.DataFrame(
            {
                "horizon": [1, 3, 5],
                "mean_ic": [0.10, 0.08, 0.06],
            }
        )
        summary = estimate_ic_half_life(decay)
        assert summary["ic_half_life_status"] == "not_reached"
        assert bool(summary["ic_half_life_not_reached"]) is True
        assert math.isnan(float(summary["ic_half_life_horizon"]))


class TestFactorAutocorrelation:
    """Verify factor autocorrelation computation."""

    def test_constant_factor_autocorr_is_nan(self):
        from alpha_lab.decay import compute_factor_autocorrelation

        dates = [f"2024-01-{d:02d}" for d in range(1, 11)]
        factor_df = pd.DataFrame(
            {
                "date": pd.to_datetime(dates * 3),
                "asset": [a for a in ["A", "B", "C"] for _ in dates],
                "factor": ["f"] * 30,
                "value": [1.0] * 30,
            }
        )
        result = compute_factor_autocorrelation(factor_df, lags=(1,))
        # Constant factor has zero cross-sectional variance → NaN autocorrelation
        assert result["mean_autocorr"].isna().all()

    def test_perfect_persistence(self):
        """Factor that never changes should have autocorrelation = 1 (when cross-section varies)."""
        from alpha_lab.decay import compute_factor_autocorrelation

        dates = [f"2024-01-{d:02d}" for d in range(1, 11)]
        assets = ["A", "B", "C"]
        # Each asset has a fixed value across all dates — perfect persistence
        factor_df = pd.DataFrame(
            {
                "date": pd.to_datetime(dates * len(assets)),
                "asset": [a for a in assets for _ in dates],
                "factor": ["f"] * len(dates) * len(assets),
                "value": [1.0 * (i + 1) for i, a in enumerate(assets) for _ in dates],
            }
        )
        result = compute_factor_autocorrelation(factor_df, lags=(1,))
        assert result["mean_autocorr"].iloc[0] == pytest.approx(1.0)


# ═══════════════════════════════════════════════════════════════════════════
# 4.2  Factor Correlation & Redundancy Analysis
# ═══════════════════════════════════════════════════════════════════════════


def _make_factor_df(
    dates: list[str], assets: list[str], factor_name: str, values: list[float]
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "asset": assets,
            "factor": [factor_name] * len(values),
            "value": values,
        }
    )


class TestFactorCorrelation:
    """Verify factor correlation matrix computation."""

    def test_identical_factors_corr_one(self):
        from alpha_lab.synthesis.redundancy import compute_factor_correlation_matrix

        # Same date, multiple assets — cross-sectional correlation
        date = "2024-01-01"
        assets = ["A", "B", "C", "D", "E"]
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        fa = _make_factor_df([date] * 5, assets, "alpha", values)
        fb = _make_factor_df([date] * 5, assets, "beta", values)
        corr = compute_factor_correlation_matrix({"alpha": fa, "beta": fb})
        assert corr.loc["alpha", "beta"] == pytest.approx(1.0)
        assert corr.loc["alpha", "alpha"] == pytest.approx(1.0)

    def test_opposite_factors_corr_neg_one(self):
        from alpha_lab.synthesis.redundancy import compute_factor_correlation_matrix

        date = "2024-01-01"
        assets = ["A", "B", "C", "D", "E"]
        fa = _make_factor_df([date] * 5, assets, "alpha", [1.0, 2.0, 3.0, 4.0, 5.0])
        fb = _make_factor_df([date] * 5, assets, "beta", [5.0, 4.0, 3.0, 2.0, 1.0])
        corr = compute_factor_correlation_matrix({"alpha": fa, "beta": fb})
        assert corr.loc["alpha", "beta"] == pytest.approx(-1.0)

    def test_uncorrelated_factors(self):
        from alpha_lab.synthesis.redundancy import compute_factor_correlation_matrix

        date = "2024-01-01"
        assets = ["A", "B", "C", "D", "E"]
        fa = _make_factor_df([date] * 5, assets, "alpha", [1.0, 2.0, 3.0, 4.0, 5.0])
        fb = _make_factor_df([date] * 5, assets, "beta", [3.0, 1.0, 5.0, 2.0, 4.0])
        corr = compute_factor_correlation_matrix({"alpha": fa, "beta": fb}, method="pearson")
        assert abs(corr.loc["alpha", "beta"]) < 0.5


class TestRedundantFactors:
    """Verify redundancy detection."""

    def test_detects_redundant_pair(self):
        from alpha_lab.synthesis.redundancy import (
            compute_factor_correlation_matrix,
            detect_redundant_factors,
        )

        date = "2024-01-01"
        assets = ["A", "B", "C", "D", "E"]
        fa = _make_factor_df([date] * 5, assets, "alpha", [1.0, 2.0, 3.0, 4.0, 5.0])
        fb = _make_factor_df([date] * 5, assets, "beta", [1.0, 2.0, 3.0, 4.0, 5.0])
        corr = compute_factor_correlation_matrix({"alpha": fa, "beta": fb})
        redundant = detect_redundant_factors(corr, threshold=0.7)
        assert len(redundant) == 1
        assert redundant[0][2] == pytest.approx(1.0)

    def test_no_redundancy_below_threshold(self):
        from alpha_lab.synthesis.redundancy import (
            compute_factor_correlation_matrix,
            detect_redundant_factors,
        )

        date = "2024-01-01"
        assets = ["A", "B", "C", "D", "E"]
        fa = _make_factor_df([date] * 5, assets, "alpha", [1.0, 2.0, 3.0, 4.0, 5.0])
        fb = _make_factor_df([date] * 5, assets, "beta", [3.0, 1.0, 5.0, 2.0, 4.0])
        corr = compute_factor_correlation_matrix({"alpha": fa, "beta": fb})
        redundant = detect_redundant_factors(corr, threshold=0.9)
        assert len(redundant) == 0


class TestRollingFactorCorrelation:
    """Verify rolling factor correlation."""

    def test_returns_expected_columns(self):
        from alpha_lab.synthesis.redundancy import compute_rolling_factor_correlation

        dates = [f"2024-01-{d:02d}" for d in range(1, 11)]
        assets = ["A", "B", "C"]
        rng = np.random.RandomState(42)
        n = len(dates) * len(assets)
        fa = pd.DataFrame(
            {
                "date": pd.to_datetime(dates * len(assets)),
                "asset": [a for a in assets for _ in dates],
                "factor": ["alpha"] * n,
                "value": rng.randn(n),
            }
        )
        fb = pd.DataFrame(
            {
                "date": pd.to_datetime(dates * len(assets)),
                "asset": [a for a in assets for _ in dates],
                "factor": ["beta"] * n,
                "value": rng.randn(n),
            }
        )
        result = compute_rolling_factor_correlation({"alpha": fa, "beta": fb}, window=3)
        assert "date" in result.columns
        assert "factor_a" in result.columns
        assert "factor_b" in result.columns
        assert "correlation" in result.columns


# ═══════════════════════════════════════════════════════════════════════════
# 1.1  Corporate Action Adjustment
# ═══════════════════════════════════════════════════════════════════════════


class TestCorporateActions:
    """Verify split detection and adjustment."""

    def test_detect_unadjusted_splits(self):
        from alpha_lab.data_quality.corporate_actions import detect_unadjusted_splits

        prices = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    [
                        "2024-01-01",
                        "2024-01-02",
                        "2024-01-03",
                        "2024-01-01",
                        "2024-01-02",
                        "2024-01-03",
                    ]
                ),
                "asset": ["A", "A", "A", "B", "B", "B"],
                "close": [
                    100.0,
                    10.0,
                    10.5,  # A: 90% drop = split
                    50.0,
                    51.0,
                    52.0,
                ],  # B: normal
            }
        )
        result = detect_unadjusted_splits(prices, threshold=0.45)
        # Only asset A on 2024-01-02 should be flagged
        flagged = result[result["suspected_split"]]
        assert len(flagged) == 1
        assert flagged.iloc[0]["asset"] == "A"
        assert flagged.iloc[0]["date"] == pd.Timestamp("2024-01-02")

    def test_adjust_for_splits(self):
        from alpha_lab.data_quality.corporate_actions import adjust_for_splits

        prices = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
                "asset": ["A", "A", "A"],
                "close": [100.0, 10.0, 10.5],
            }
        )
        adj_factor = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
                "asset": ["A", "A", "A"],
                "adj_factor": [1.0, 10.0, 10.0],  # split happened on day 2
            }
        )
        result = adjust_for_splits(prices, adj_factor)
        # After back-adjustment, all prices should be on the same scale
        # adj_close = close * adj_factor / latest_adj_factor
        # Day 1: 100 * 1.0 / 10.0 = 10.0
        # Day 2: 10 * 10.0 / 10.0 = 10.0
        # Day 3: 10.5 * 10.0 / 10.0 = 10.5
        assert result.loc[result["date"] == pd.Timestamp("2024-01-01"), "close"].iloc[
            0
        ] == pytest.approx(10.0)
        assert result.loc[result["date"] == pd.Timestamp("2024-01-02"), "close"].iloc[
            0
        ] == pytest.approx(10.0)
        assert result.loc[result["date"] == pd.Timestamp("2024-01-03"), "close"].iloc[
            0
        ] == pytest.approx(10.5)

    def test_no_splits_detected_for_normal_prices(self):
        from alpha_lab.data_quality.corporate_actions import detect_unadjusted_splits

        prices = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
                "asset": ["A", "A", "A"],
                "close": [100.0, 101.0, 99.5],
            }
        )
        result = detect_unadjusted_splits(prices)
        assert not result["suspected_split"].any()

    def test_adjust_preserves_other_columns(self):
        from alpha_lab.data_quality.corporate_actions import adjust_for_splits

        prices = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
                "asset": ["A", "A"],
                "close": [100.0, 10.0],
                "volume": [1000, 5000],
            }
        )
        adj_factor = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
                "asset": ["A", "A"],
                "adj_factor": [1.0, 10.0],
            }
        )
        result = adjust_for_splits(prices, adj_factor)
        assert "volume" in result.columns


# ═══════════════════════════════════════════════════════════════════════════
# 1.2  PIT Compliance for Fundamentals
# ═══════════════════════════════════════════════════════════════════════════


class TestPITFundamentals:
    """Verify point-in-time alignment for financial data."""

    def test_build_pit_mapping(self):
        from alpha_lab.data_quality.pit_fundamentals import build_pit_mapping

        fina = pd.DataFrame(
            {
                "asset": ["A", "A", "A", "A"],
                "end_date": pd.to_datetime(
                    ["2023-03-31", "2023-06-30", "2023-09-30", "2023-12-31"]
                ),
                "ann_date": pd.to_datetime(
                    ["2023-04-28", "2023-08-20", "2023-10-30", "2024-04-25"]
                ),
                "roe": [0.05, 0.06, 0.07, 0.08],
            }
        )
        mapping = build_pit_mapping(fina)
        assert len(mapping) == 4
        assert "asset" in mapping.columns
        assert "end_date" in mapping.columns
        assert "ann_date" in mapping.columns

    def test_apply_pit_alignment_uses_announced_data(self):
        from alpha_lab.data_quality.pit_fundamentals import (
            apply_pit_alignment,
            build_pit_mapping,
        )

        fina = pd.DataFrame(
            {
                "asset": ["A", "A"],
                "end_date": pd.to_datetime(["2023-09-30", "2023-12-31"]),
                "ann_date": pd.to_datetime(["2023-10-30", "2024-04-25"]),
                "roe": [0.07, 0.08],
            }
        )
        mapping = build_pit_mapping(fina)

        # On 2024-01-15, only Q3 report (end_date=09-30, ann_date=10-30) is available
        signal_dates = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-15", "2024-05-01"]),
                "asset": ["A", "A"],
            }
        )
        result = apply_pit_alignment(signal_dates, fina, mapping)
        # Jan 15: Q4 not yet announced → use Q3 value (0.07)
        jan_row = result[result["date"] == pd.Timestamp("2024-01-15")]
        assert jan_row["roe"].iloc[0] == pytest.approx(0.07)
        # May 1: Q4 announced (2024-04-25) → use Q4 value (0.08)
        may_row = result[result["date"] == pd.Timestamp("2024-05-01")]
        assert may_row["roe"].iloc[0] == pytest.approx(0.08)

    def test_pit_no_data_before_first_announcement(self):
        from alpha_lab.data_quality.pit_fundamentals import (
            apply_pit_alignment,
            build_pit_mapping,
        )

        fina = pd.DataFrame(
            {
                "asset": ["A"],
                "end_date": pd.to_datetime(["2023-12-31"]),
                "ann_date": pd.to_datetime(["2024-04-25"]),
                "roe": [0.08],
            }
        )
        mapping = build_pit_mapping(fina)
        signal_dates = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01"]),
                "asset": ["A"],
            }
        )
        result = apply_pit_alignment(signal_dates, fina, mapping)
        # No data announced yet → NaN
        assert result["roe"].isna().all()

    def test_pit_duplicate_announcements_uses_earliest(self):
        """When same period has multiple announcement dates, use earliest."""
        from alpha_lab.data_quality.pit_fundamentals import build_pit_mapping

        fina = pd.DataFrame(
            {
                "asset": ["A", "A"],
                "end_date": pd.to_datetime(["2023-12-31", "2023-12-31"]),
                "ann_date": pd.to_datetime(["2024-04-25", "2024-03-15"]),  # correction came earlier
                "roe": [0.08, 0.085],
            }
        )
        mapping = build_pit_mapping(fina)
        # Should keep the earliest announcement date
        a_rows = mapping[mapping["asset"] == "A"]
        assert len(a_rows) == 1
        assert a_rows.iloc[0]["ann_date"] == pd.Timestamp("2024-03-15")
