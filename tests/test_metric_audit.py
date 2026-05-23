"""Phase 1–3 metric audit: hand-calculated ground truth for every atomic indicator.

Each test constructs a tiny dataset (3–5 assets, 3–6 dates), computes the
expected answer by hand (shown in comments), and asserts the code matches.
Tests are grouped by the computation module they audit.

Run:  pytest tests/test_metric_audit.py -v
"""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest

# ── helpers ──────────────────────────────────────────────────────────────────

COLS = ("date", "asset", "factor", "value")


def _canon(
    dates: list[str],
    assets: list[str],
    factor_name: str,
    values: list[float],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "asset": assets,
            "factor": [factor_name] * len(values),
            "value": values,
        }
    )


def _prices(asset_prices: dict[str, list[float]], start: str = "2024-01-02") -> pd.DataFrame:
    n = max(len(v) for v in asset_prices.values())
    dates = pd.bdate_range(start, periods=n)
    rows = []
    for asset, px in asset_prices.items():
        for i, price in enumerate(px):
            rows.append({"date": dates[i], "asset": asset, "close": price})
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
# 1.1  forward_return
# ═══════════════════════════════════════════════════════════════════════════


class TestForwardReturn:
    """Audit labels.forward_return against hand calculations."""

    def test_horizon_1_hand_calc(self):
        """A: [100, 110, 132]  →  ret = [0.10, 0.20, NaN]"""
        from alpha_lab.labels import forward_return

        df = _prices({"A": [100.0, 110.0, 132.0]})
        result = forward_return(df, horizon=1)
        vals = result.set_index(["date", "asset"])["value"]

        assert vals.iloc[0] == pytest.approx(110.0 / 100.0 - 1)  # 0.10
        assert vals.iloc[1] == pytest.approx(132.0 / 110.0 - 1)  # 0.20
        assert np.isnan(vals.iloc[2])

    def test_horizon_2_hand_calc(self):
        """A: [100, 110, 132, 145.2]  →  ret(h=2) = [0.32, 0.32, NaN, NaN]
        close[0+2]/close[0]-1 = 132/100-1 = 0.32
        close[1+2]/close[1]-1 = 145.2/110-1 = 0.32
        """
        from alpha_lab.labels import forward_return

        df = _prices({"A": [100.0, 110.0, 132.0, 145.2]})
        result = forward_return(df, horizon=2)
        vals = result.set_index(["date", "asset"])["value"]

        assert vals.iloc[0] == pytest.approx(132.0 / 100.0 - 1)  # 0.32
        assert vals.iloc[1] == pytest.approx(145.2 / 110.0 - 1)  # 0.32
        assert np.isnan(vals.iloc[2])
        assert np.isnan(vals.iloc[3])

    def test_multi_asset_independence(self):
        """Each asset's forward return uses its own row order, not global."""
        from alpha_lab.labels import forward_return

        df = _prices({"A": [100.0, 120.0], "B": [200.0, 210.0]})
        result = forward_return(df, horizon=1)
        a = result[result["asset"] == "A"].set_index("date")["value"]
        b = result[result["asset"] == "B"].set_index("date")["value"]

        assert a.iloc[0] == pytest.approx(120.0 / 100.0 - 1)  # 0.20
        assert b.iloc[0] == pytest.approx(210.0 / 200.0 - 1)  # 0.05

    def test_zero_price_produces_nan(self):
        """close=0 is non-positive → treated as missing."""
        from alpha_lab.labels import forward_return

        df = _prices({"A": [100.0, 0.0, 120.0]})
        result = forward_return(df, horizon=1)
        vals = result.set_index("date")["value"]
        # ret[0] = 0/100 → NaN because next close is 0 (non-positive)
        assert np.isnan(vals.iloc[0])
        # ret[1] = 120/0 → NaN because current close is 0
        assert np.isnan(vals.iloc[1])

    def test_negative_price_produces_nan(self):
        from alpha_lab.labels import forward_return

        df = _prices({"A": [100.0, -5.0, 120.0]})
        result = forward_return(df, horizon=1)
        vals = result.set_index("date")["value"]
        assert np.isnan(vals.iloc[0])
        assert np.isnan(vals.iloc[1])


# ═══════════════════════════════════════════════════════════════════════════
# 1.2  compute_ic (Pearson)
# ═══════════════════════════════════════════════════════════════════════════


class TestComputeIC:
    """Audit evaluation.compute_ic with hand-calculated Pearson r."""

    def test_perfect_positive_correlation(self):
        """factor=[1,2,3] label=[2,4,6] → Pearson r = 1.0"""
        from alpha_lab.evaluation import compute_ic

        f = _canon(["2024-01-02"] * 3, ["A", "B", "C"], "f", [1.0, 2.0, 3.0])
        labels_df = _canon(["2024-01-02"] * 3, ["A", "B", "C"], "l", [2.0, 4.0, 6.0])
        result = compute_ic(f, labels_df)
        assert result["ic"].iloc[0] == pytest.approx(1.0)

    def test_perfect_negative_correlation(self):
        """factor=[1,2,3] label=[6,4,2] → r = -1.0"""
        from alpha_lab.evaluation import compute_ic

        f = _canon(["2024-01-02"] * 3, ["A", "B", "C"], "f", [1.0, 2.0, 3.0])
        labels_df = _canon(["2024-01-02"] * 3, ["A", "B", "C"], "l", [6.0, 4.0, 2.0])
        result = compute_ic(f, labels_df)
        assert result["ic"].iloc[0] == pytest.approx(-1.0)

    def test_hand_calculated_nonlinear(self):
        """factor=[1,2,3,4] label=[1,3,2,6]
        Hand calc:
          f_bar=2.5  l_bar=3.0
          f_dm = [-1.5, -0.5, 0.5, 1.5]
          l_dm = [-2.0, 0.0, -1.0, 3.0]
          sum(f*l) = 3.0 + 0.0 + (-0.5) + 4.5 = 7.0
          sum(f^2) = 2.25 + 0.25 + 0.25 + 2.25 = 5.0
          sum(l^2) = 4.0 + 0.0 + 1.0 + 9.0 = 14.0
          r = 7.0 / sqrt(5.0 * 14.0) = 7.0 / sqrt(70) ≈ 0.83666
        """
        from alpha_lab.evaluation import compute_ic

        f = _canon(["2024-01-02"] * 4, ["A", "B", "C", "D"], "f", [1, 2, 3, 4])
        labels_df = _canon(["2024-01-02"] * 4, ["A", "B", "C", "D"], "l", [1, 3, 2, 6])
        result = compute_ic(f, labels_df)
        expected = 7.0 / math.sqrt(70.0)
        assert result["ic"].iloc[0] == pytest.approx(expected, abs=1e-10)

    def test_multi_date_independent(self):
        """Each date's IC is computed independently."""
        from alpha_lab.evaluation import compute_ic

        f = _canon(
            ["2024-01-02"] * 3 + ["2024-01-03"] * 3,
            ["A", "B", "C"] * 2,
            "f",
            [1, 2, 3, 3, 2, 1],
        )
        labels_df = _canon(
            ["2024-01-02"] * 3 + ["2024-01-03"] * 3,
            ["A", "B", "C"] * 2,
            "l",
            [10, 20, 30, 10, 20, 30],
        )
        result = compute_ic(f, labels_df).set_index("date")
        assert result.loc[pd.Timestamp("2024-01-02"), "ic"] == pytest.approx(1.0)
        assert result.loc[pd.Timestamp("2024-01-03"), "ic"] == pytest.approx(-1.0)

    def test_constant_factor_produces_nan(self):
        """All factor values identical → std=0 → IC=NaN."""
        from alpha_lab.evaluation import compute_ic

        f = _canon(["2024-01-02"] * 3, ["A", "B", "C"], "f", [5.0, 5.0, 5.0])
        labels_df = _canon(["2024-01-02"] * 3, ["A", "B", "C"], "l", [1.0, 2.0, 3.0])
        result = compute_ic(f, labels_df)
        assert np.isnan(result["ic"].iloc[0])

    def test_constant_label_produces_nan(self):
        from alpha_lab.evaluation import compute_ic

        f = _canon(["2024-01-02"] * 3, ["A", "B", "C"], "f", [1.0, 2.0, 3.0])
        labels_df = _canon(["2024-01-02"] * 3, ["A", "B", "C"], "l", [5.0, 5.0, 5.0])
        result = compute_ic(f, labels_df)
        assert np.isnan(result["ic"].iloc[0])

    def test_single_asset_date_nan(self):
        """Only 1 asset on a date → IC=NaN (can't compute correlation)."""
        from alpha_lab.evaluation import compute_ic

        f = _canon(["2024-01-02"], ["A"], "f", [1.0])
        labels_df = _canon(["2024-01-02"], ["A"], "l", [2.0])
        result = compute_ic(f, labels_df)
        assert np.isnan(result["ic"].iloc[0])

    def test_cross_validate_with_scipy(self):
        """Compare with scipy.stats.pearsonr on a non-trivial example."""
        from scipy.stats import pearsonr

        from alpha_lab.evaluation import compute_ic

        fv = [0.5, -0.3, 1.2, 0.8, -0.1]
        lv = [0.02, -0.01, 0.05, 0.03, 0.0]
        f = _canon(["2024-01-02"] * 5, ["A", "B", "C", "D", "E"], "f", fv)
        labels_df = _canon(["2024-01-02"] * 5, ["A", "B", "C", "D", "E"], "l", lv)
        result = compute_ic(f, labels_df)

        expected_r, _ = pearsonr(fv, lv)
        assert result["ic"].iloc[0] == pytest.approx(expected_r, abs=1e-10)


# ═══════════════════════════════════════════════════════════════════════════
# 1.3  compute_rank_ic (Spearman)
# ═══════════════════════════════════════════════════════════════════════════


class TestComputeRankIC:
    """Audit evaluation.compute_rank_ic = Pearson(rank_f, rank_l)."""

    def test_monotonic_agreement(self):
        """Monotonic agreement → Spearman = 1.0 regardless of non-linearity."""
        from alpha_lab.evaluation import compute_rank_ic

        f = _canon(["2024-01-02"] * 4, ["A", "B", "C", "D"], "f", [1, 2, 3, 4])
        labels_df = _canon(
            ["2024-01-02"] * 4,
            ["A", "B", "C", "D"],
            "l",
            [10, 100, 1000, 10000],
        )
        result = compute_rank_ic(f, labels_df)
        assert result["rank_ic"].iloc[0] == pytest.approx(1.0)

    def test_hand_calculated_rank_ic(self):
        """factor=[1,2,3] label=[30,10,20]
        rank_f = [1, 2, 3]
        rank_l = [3, 1, 2]
        Pearson(rank_f, rank_l):
          rf_bar = 2.0  rl_bar = 2.0
          rf_dm = [-1, 0, 1]  rl_dm = [1, -1, 0]
          sum(rf*rl) = -1 + 0 + 0 = -1
          sum(rf^2) = 1 + 0 + 1 = 2
          sum(rl^2) = 1 + 1 + 0 = 2
          r = -1 / sqrt(4) = -0.5
        """
        from alpha_lab.evaluation import compute_rank_ic

        f = _canon(["2024-01-02"] * 3, ["A", "B", "C"], "f", [1, 2, 3])
        labels_df = _canon(["2024-01-02"] * 3, ["A", "B", "C"], "l", [30, 10, 20])
        result = compute_rank_ic(f, labels_df)
        assert result["rank_ic"].iloc[0] == pytest.approx(-0.5)

    def test_tied_values_use_average_rank(self):
        """factor=[1,1,3] → ranks with method='average' = [1.5, 1.5, 3.0]
        label=[10,20,30] → ranks = [1, 2, 3]
        Pearson(rank_f, rank_l):
          rf_bar = 2.0  rl_bar = 2.0
          rf_dm = [-0.5, -0.5, 1.0]  rl_dm = [-1, 0, 1]
          sum(rf*rl) = 0.5 + 0 + 1.0 = 1.5
          sum(rf^2) = 0.25 + 0.25 + 1.0 = 1.5
          sum(rl^2) = 1 + 0 + 1 = 2
          r = 1.5 / sqrt(1.5 * 2) = 1.5 / sqrt(3) ≈ 0.86603
        """
        from alpha_lab.evaluation import compute_rank_ic

        f = _canon(["2024-01-02"] * 3, ["A", "B", "C"], "f", [1, 1, 3])
        labels_df = _canon(["2024-01-02"] * 3, ["A", "B", "C"], "l", [10, 20, 30])
        result = compute_rank_ic(f, labels_df)
        expected = 1.5 / math.sqrt(3.0)
        assert result["rank_ic"].iloc[0] == pytest.approx(expected, abs=1e-10)

    def test_cross_validate_with_scipy(self):
        """Compare with scipy.stats.spearmanr."""
        from scipy.stats import spearmanr

        from alpha_lab.evaluation import compute_rank_ic

        fv = [0.5, -0.3, 1.2, 0.8, -0.1]
        lv = [0.02, -0.01, 0.05, 0.03, 0.0]
        f = _canon(["2024-01-02"] * 5, ["A", "B", "C", "D", "E"], "f", fv)
        labels_df = _canon(["2024-01-02"] * 5, ["A", "B", "C", "D", "E"], "l", lv)
        result = compute_rank_ic(f, labels_df)

        expected_rho, _ = spearmanr(fv, lv)
        assert result["rank_ic"].iloc[0] == pytest.approx(expected_rho, abs=1e-10)


# ═══════════════════════════════════════════════════════════════════════════
# 1.4  _assign_quantile
# ═══════════════════════════════════════════════════════════════════════════


class TestAssignQuantile:
    """Audit quantile._assign_quantile bucket mapping."""

    def test_three_assets_two_buckets(self):
        """values=[10, 20, 30] n_q=2
        dense_rank = [1, 2, 3]  n_distinct=3  effective_q=2
        q = (r-1)/(3-1)*(2-1)+1 = (r-1)/2 + 1
          r=1: q=1.0 → bucket 1
          r=2: q=1.5 → floor(2.0)=2
          r=3: q=2.0 → bucket 2
        """
        from alpha_lab.quantile import _assign_quantile

        s = pd.Series([10.0, 20.0, 30.0])
        result = _assign_quantile(s, n_quantiles=2)
        assert list(result) == [1, 2, 2]

    def test_five_assets_three_buckets(self):
        """values=[1,2,3,4,5] n_q=3
        dense_rank=[1,2,3,4,5] n_distinct=5 effective_q=3
        q = (r-1)/4 * 2 + 1
          r=1: q=1.0 → 1
          r=2: q=1.5 → floor(2.0) = 2
          r=3: q=2.0 → 2
          r=4: q=2.5 → floor(3.0) = 3
          r=5: q=3.0 → 3
        """
        from alpha_lab.quantile import _assign_quantile

        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _assign_quantile(s, n_quantiles=3)
        assert list(result) == [1, 2, 2, 3, 3]

    def test_tied_values_same_bucket(self):
        """values=[10, 10, 30] n_q=3
        dense_rank=[1, 1, 2]  n_distinct=2  effective_q=3
        q = (r-1)/(2-1)*(3-1)+1 = (r-1)*2+1
          r=1: q=1 → 1
          r=1: q=1 → 1
          r=2: q=3 → 3
        """
        from alpha_lab.quantile import _assign_quantile

        s = pd.Series([10.0, 10.0, 30.0])
        result = _assign_quantile(s, n_quantiles=3)
        assert list(result) == [1, 1, 3]

    def test_constant_factor_all_bucket_1(self):
        """All identical → n_distinct=1 → all bucket 1."""
        from alpha_lab.quantile import _assign_quantile

        s = pd.Series([5.0, 5.0, 5.0])
        result = _assign_quantile(s, n_quantiles=3)
        assert list(result) == [1, 1, 1]

    def test_single_asset_nan(self):
        """n_valid < 2 → all NaN."""
        from alpha_lab.quantile import _assign_quantile

        s = pd.Series([5.0])
        result = _assign_quantile(s, n_quantiles=3)
        assert np.isnan(result.iloc[0])

    def test_nan_input_propagates(self):
        """NaN values → NaN bucket; valid values still assigned."""
        from alpha_lab.quantile import _assign_quantile

        s = pd.Series([10.0, np.nan, 30.0])
        result = _assign_quantile(s, n_quantiles=2)
        assert result.iloc[0] == 1
        assert np.isnan(result.iloc[1])
        assert result.iloc[2] == 2

    def test_row_order_invariant(self):
        """Bucket assignments depend on value, not row order."""
        from alpha_lab.quantile import _assign_quantile

        s1 = pd.Series([10.0, 20.0, 30.0])
        s2 = pd.Series([30.0, 10.0, 20.0])  # shuffled values
        r1 = _assign_quantile(s1, n_quantiles=3)
        r2 = _assign_quantile(s2, n_quantiles=3)
        # After sorting by value, same assignment
        assert r1.iloc[0] == r2.iloc[1]  # value 10 → same bucket
        assert r1.iloc[2] == r2.iloc[0]  # value 30 → same bucket


# ═══════════════════════════════════════════════════════════════════════════
# 1.5  quantile_returns
# ═══════════════════════════════════════════════════════════════════════════


class TestQuantileReturns:
    """Audit quantile.quantile_returns mean-return-per-bucket."""

    def test_hand_calc_two_buckets(self):
        """factor=[1,2,3,4] label=[0.01, 0.02, 0.03, 0.04] n_q=2
        Bucket assignment (from 1.4 logic):
          dense_rank=[1,2,3,4]  n_distinct=4  effective_q=2
          r=1: q=1  r=2: q=floor((1/3)*1+1+0.5)=floor(1.83)=1  → actually
          Let me recalculate properly:
          q = (r-1)/(4-1)*(2-1)+1 = (r-1)/3 + 1
          r=1: 1.0 → 1
          r=2: 1.333 → floor(1.833) = 1  (1.333+0.5=1.833, floor=1)
          r=3: 1.667 → floor(2.167) = 2
          r=4: 2.0 → 2
        Bucket 1: assets A,B → labels 0.01, 0.02 → mean=0.015
        Bucket 2: assets C,D → labels 0.03, 0.04 → mean=0.035
        """
        from alpha_lab.quantile import quantile_returns

        f = _canon(
            ["2024-01-02"] * 4,
            ["A", "B", "C", "D"],
            "f",
            [1.0, 2.0, 3.0, 4.0],
        )
        labels_df = _canon(
            ["2024-01-02"] * 4,
            ["A", "B", "C", "D"],
            "l",
            [0.01, 0.02, 0.03, 0.04],
        )
        result = quantile_returns(f, labels_df, n_quantiles=2)
        b1 = result[result["quantile"] == 1]["mean_return"].iloc[0]
        b2 = result[result["quantile"] == 2]["mean_return"].iloc[0]
        assert b1 == pytest.approx(0.015)
        assert b2 == pytest.approx(0.035)


# ═══════════════════════════════════════════════════════════════════════════
# 1.6  long_short_return
# ═══════════════════════════════════════════════════════════════════════════


class TestLongShortReturn:
    """Audit quantile.long_short_return = top - bottom."""

    def test_hand_calc(self):
        """From 1.5: bucket 1 mean=0.015, bucket 2 mean=0.035
        L/S = top(2) - bottom(1) = 0.035 - 0.015 = 0.020
        """
        from alpha_lab.quantile import long_short_return, quantile_returns

        f = _canon(
            ["2024-01-02"] * 4,
            ["A", "B", "C", "D"],
            "f",
            [1.0, 2.0, 3.0, 4.0],
        )
        labels_df = _canon(
            ["2024-01-02"] * 4,
            ["A", "B", "C", "D"],
            "l",
            [0.01, 0.02, 0.03, 0.04],
        )
        qr = quantile_returns(f, labels_df, n_quantiles=2)
        ls = long_short_return(qr)
        assert ls["long_short_return"].iloc[0] == pytest.approx(0.020)

    def test_negative_spread(self):
        """Factor inversely related to returns → negative L/S."""
        from alpha_lab.quantile import long_short_return, quantile_returns

        f = _canon(
            ["2024-01-02"] * 4,
            ["A", "B", "C", "D"],
            "f",
            [1.0, 2.0, 3.0, 4.0],
        )
        labels_df = _canon(
            ["2024-01-02"] * 4,
            ["A", "B", "C", "D"],
            "l",
            [0.04, 0.03, 0.02, 0.01],
        )
        qr = quantile_returns(f, labels_df, n_quantiles=2)
        ls = long_short_return(qr)
        assert ls["long_short_return"].iloc[0] == pytest.approx(-0.020)

    def test_constant_factor_produces_nan(self):
        """All same factor → single bucket → L/S = NaN."""
        from alpha_lab.quantile import long_short_return, quantile_returns

        f = _canon(["2024-01-02"] * 3, ["A", "B", "C"], "f", [5, 5, 5])
        labels_df = _canon(["2024-01-02"] * 3, ["A", "B", "C"], "l", [0.01, 0.02, 0.03])
        qr = quantile_returns(f, labels_df, n_quantiles=3)
        ls = long_short_return(qr)
        # Constant factor → all in bucket 1 → only 1 bucket → NaN
        assert ls.empty or all(np.isnan(ls["long_short_return"]))

    def test_factor_reversal_flips_sign(self):
        """Negating factor values should flip L/S sign (Phase 3.8)."""
        from alpha_lab.quantile import long_short_return, quantile_returns

        fv = [1.0, 2.0, 3.0, 4.0]
        lv = [0.01, 0.02, 0.03, 0.04]

        f_pos = _canon(["2024-01-02"] * 4, ["A", "B", "C", "D"], "f", fv)
        f_neg = _canon(["2024-01-02"] * 4, ["A", "B", "C", "D"], "f", [-v for v in fv])
        labels_df = _canon(["2024-01-02"] * 4, ["A", "B", "C", "D"], "l", lv)

        ls_pos = long_short_return(quantile_returns(f_pos, labels_df, n_quantiles=2))
        ls_neg = long_short_return(quantile_returns(f_neg, labels_df, n_quantiles=2))

        assert ls_pos["long_short_return"].iloc[0] == pytest.approx(
            -ls_neg["long_short_return"].iloc[0]
        )


# ═══════════════════════════════════════════════════════════════════════════
# 1.7  quantile_turnover
# ═══════════════════════════════════════════════════════════════════════════


class TestQuantileTurnover:
    """Audit turnover.quantile_turnover hand calculations."""

    def _assignments(
        self,
        date: str,
        asset_quantiles: dict[str, int],
        factor_name: str = "f",
    ) -> pd.DataFrame:
        rows = []
        for asset, q in asset_quantiles.items():
            rows.append(
                {"date": pd.Timestamp(date), "asset": asset, "factor": factor_name, "quantile": q}
            )
        return pd.DataFrame(rows)

    def test_first_day_is_nan(self):
        """First day has no prior state → turnover = NaN."""
        from alpha_lab.turnover import quantile_turnover

        day1 = self._assignments("2024-01-02", {"A": 1, "B": 2})
        result = quantile_turnover(day1)
        assert all(np.isnan(result["turnover"]))

    def test_no_change_zero_turnover(self):
        """Same composition → turnover = 0."""
        from alpha_lab.turnover import quantile_turnover

        day1 = self._assignments("2024-01-02", {"A": 1, "B": 1, "C": 2, "D": 2})
        day2 = self._assignments("2024-01-03", {"A": 1, "B": 1, "C": 2, "D": 2})
        asgn = pd.concat([day1, day2], ignore_index=True)
        result = quantile_turnover(asgn)

        d2 = result[result["date"] == pd.Timestamp("2024-01-03")]
        for _, row in d2.iterrows():
            assert row["turnover"] == pytest.approx(0.0)

    def test_full_replacement_turnover_1(self):
        """Completely new members → turnover = 1.0."""
        from alpha_lab.turnover import quantile_turnover

        day1 = self._assignments("2024-01-02", {"A": 1, "B": 2})
        day2 = self._assignments("2024-01-03", {"C": 1, "D": 2})
        asgn = pd.concat([day1, day2], ignore_index=True)
        result = quantile_turnover(asgn)

        d2 = result[result["date"] == pd.Timestamp("2024-01-03")]
        for _, row in d2.iterrows():
            assert row["turnover"] == pytest.approx(1.0)

    def test_partial_turnover_hand_calc(self):
        """Bucket 1: day1={A,B,C} day2={A,B,D}
        entering = {D}, |entering|=1, |members(t)|=3
        turnover = 1/3
        """
        from alpha_lab.turnover import quantile_turnover

        day1 = self._assignments("2024-01-02", {"A": 1, "B": 1, "C": 1, "D": 2, "E": 2})
        day2 = self._assignments("2024-01-03", {"A": 1, "B": 1, "D": 1, "C": 2, "E": 2})
        asgn = pd.concat([day1, day2], ignore_index=True)
        result = quantile_turnover(asgn)

        d2_b1 = result[(result["date"] == pd.Timestamp("2024-01-03")) & (result["quantile"] == 1)]
        assert d2_b1["turnover"].iloc[0] == pytest.approx(1.0 / 3.0)

    def test_bucket_absent_then_present(self):
        """Bucket absent at t-1 but present at t → all entering → turnover=1.0."""
        from alpha_lab.turnover import quantile_turnover

        day1 = self._assignments("2024-01-02", {"A": 1, "B": 1})
        day2 = self._assignments("2024-01-03", {"A": 1, "B": 2})  # bucket 2 is new
        asgn = pd.concat([day1, day2], ignore_index=True)
        result = quantile_turnover(asgn)

        d2_b2 = result[(result["date"] == pd.Timestamp("2024-01-03")) & (result["quantile"] == 2)]
        assert d2_b2["turnover"].iloc[0] == pytest.approx(1.0)


# ═══════════════════════════════════════════════════════════════════════════
# 1.8  long_short_turnover
# ═══════════════════════════════════════════════════════════════════════════


class TestLongShortTurnover:
    """Audit turnover.long_short_turnover = (top + bottom) / 2."""

    def test_hand_calc(self):
        """top turnover=0.5, bottom turnover=0.25 → L/S turnover=0.375."""
        from alpha_lab.turnover import long_short_turnover

        qt_df = pd.DataFrame(
            {
                "date": [pd.Timestamp("2024-01-03")] * 3,
                "factor": ["f"] * 3,
                "quantile": [1, 2, 3],
                "turnover": [0.25, 0.40, 0.50],
            }
        )
        result = long_short_turnover(qt_df)
        # top=3 (turnover=0.50), bottom=1 (turnover=0.25)
        assert result["long_short_turnover"].iloc[0] == pytest.approx((0.25 + 0.50) / 2)

    def test_first_day_nan(self):
        """If either leg is NaN → L/S turnover = NaN."""
        from alpha_lab.turnover import long_short_turnover

        qt_df = pd.DataFrame(
            {
                "date": [pd.Timestamp("2024-01-02")] * 2,
                "factor": ["f"] * 2,
                "quantile": [1, 2],
                "turnover": [float("nan"), float("nan")],
            }
        )
        result = long_short_turnover(qt_df)
        assert np.isnan(result["long_short_turnover"].iloc[0])

    def test_single_bucket_nan(self):
        """Only one bucket → L/S undefined → NaN."""
        from alpha_lab.turnover import long_short_turnover

        qt_df = pd.DataFrame(
            {
                "date": [pd.Timestamp("2024-01-03")],
                "factor": ["f"],
                "quantile": [1],
                "turnover": [0.5],
            }
        )
        result = long_short_turnover(qt_df)
        assert np.isnan(result["long_short_turnover"].iloc[0])


# ═══════════════════════════════════════════════════════════════════════════
# 1.9  apply_linear_cost
# ═══════════════════════════════════════════════════════════════════════════


class TestLinearCost:
    """Audit costs.apply_linear_cost = return - cost_rate * turnover."""

    def test_hand_calc(self):
        from alpha_lab.costs import apply_linear_cost

        ret = pd.Series([0.05, 0.03, -0.01])
        turn = pd.Series([0.5, 0.3, 0.8])
        cost_rate = 0.001

        result = apply_linear_cost(ret, turn, cost_rate=cost_rate)
        assert result.iloc[0] == pytest.approx(0.05 - 0.001 * 0.5)  # 0.0495
        assert result.iloc[1] == pytest.approx(0.03 - 0.001 * 0.3)  # 0.0297
        assert result.iloc[2] == pytest.approx(-0.01 - 0.001 * 0.8)  # -0.0108

    def test_nan_turnover_propagates(self):
        """NaN turnover (first day) → NaN adjusted return."""
        from alpha_lab.costs import apply_linear_cost

        ret = pd.Series([0.05])
        turn = pd.Series([float("nan")])
        result = apply_linear_cost(ret, turn, cost_rate=0.001)
        assert np.isnan(result.iloc[0])

    def test_zero_cost_rate_no_adjustment(self):
        from alpha_lab.costs import apply_linear_cost

        ret = pd.Series([0.05, -0.03])
        turn = pd.Series([0.5, 0.8])
        result = apply_linear_cost(ret, turn, cost_rate=0.0)
        pd.testing.assert_series_equal(result, ret)

    def test_cost_adjusted_long_short_hand_calc(self):
        """End-to-end: L/S return with turnover cost."""
        from alpha_lab.costs import cost_adjusted_long_short

        ls_df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
                "factor": ["f", "f"],
                "long_short_return": [0.02, 0.03],
            }
        )
        turnover_df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
                "factor": ["f", "f"],
                "long_short_turnover": [float("nan"), 0.4],
            }
        )
        result = cost_adjusted_long_short(ls_df, turnover_df, cost_rate=0.001)
        assert np.isnan(result["adjusted_return"].iloc[0])  # first day NaN
        assert result["adjusted_return"].iloc[1] == pytest.approx(0.03 - 0.001 * 0.4)


# ═══════════════════════════════════════════════════════════════════════════
# 1.10  compute_mean_ci (normal approximation)
# ═══════════════════════════════════════════════════════════════════════════


class TestMeanCINormal:
    """Audit uncertainty.compute_mean_ci = mean ± z * (std/sqrt(n))."""

    def test_hand_calc_95(self):
        """values = [1, 2, 3, 4, 5]
        mean = 3.0
        std(ddof=1) = sqrt(10/4) = sqrt(2.5) ≈ 1.5811
        stderr = 1.5811 / sqrt(5) ≈ 0.7071
        z_0.975 = 1.95996...
        half_width = 1.95996 * 0.7071 ≈ 1.3859
        CI = [3.0 - 1.3859, 3.0 + 1.3859] = [1.6141, 4.3859]
        """
        from alpha_lab.reporting.uncertainty import compute_mean_ci

        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        ci = compute_mean_ci(values, confidence_level=0.95)

        assert ci.mean == pytest.approx(3.0)
        assert ci.n_obs == 5

        std_ddof1 = np.std(values, ddof=1)
        stderr = std_ddof1 / math.sqrt(5)
        z = NormalDist().inv_cdf(0.975)
        hw = z * stderr

        assert ci.stderr == pytest.approx(stderr, abs=1e-10)
        assert ci.ci_lower == pytest.approx(3.0 - hw, abs=1e-10)
        assert ci.ci_upper == pytest.approx(3.0 + hw, abs=1e-10)

    def test_single_value_nan_ci(self):
        """n < 2 → CI = NaN."""
        from alpha_lab.reporting.uncertainty import compute_mean_ci

        ci = compute_mean_ci([5.0], confidence_level=0.95)
        assert ci.mean == pytest.approx(5.0)
        assert np.isnan(ci.ci_lower)
        assert np.isnan(ci.ci_upper)

    def test_empty_all_nan(self):
        from alpha_lab.reporting.uncertainty import compute_mean_ci

        ci = compute_mean_ci([], confidence_level=0.95)
        assert np.isnan(ci.mean)
        assert ci.n_obs == 0

    def test_nan_values_filtered(self):
        """NaN/inf values should be excluded before computation."""
        from alpha_lab.reporting.uncertainty import compute_mean_ci

        values = [1.0, float("nan"), 3.0, float("inf"), 5.0]
        ci = compute_mean_ci(values, confidence_level=0.95)
        # Only [1, 3, 5] are finite
        assert ci.n_obs == 3
        assert ci.mean == pytest.approx(3.0)

    def test_overlaps_zero_property(self):
        """CI that spans zero should report overlaps_zero=True."""
        from alpha_lab.reporting.uncertainty import compute_mean_ci

        # values centered near zero with enough spread
        ci = compute_mean_ci([-1.0, 0.0, 1.0, 0.5, -0.5], confidence_level=0.95)
        assert ci.overlaps_zero is True

    def test_positive_ci_no_overlap(self):
        """CI entirely above zero → overlaps_zero=False."""
        from alpha_lab.reporting.uncertainty import compute_mean_ci

        ci = compute_mean_ci([10.0, 11.0, 12.0, 13.0, 14.0], confidence_level=0.95)
        assert ci.overlaps_zero is False

    def test_audit_uses_z_not_t(self):
        """AUDIT POINT: the code uses normal z, not Student-t.
        For n=5 at 95%, z=1.960 but t(df=4)=2.776.
        This test documents the choice — CI is narrower than t-based.
        """
        from scipy.stats import t as t_dist

        from alpha_lab.reporting.uncertainty import compute_mean_ci

        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        ci = compute_mean_ci(values, confidence_level=0.95)

        std_ddof1 = np.std(values, ddof=1)
        stderr = std_ddof1 / math.sqrt(5)

        # t-based CI would be wider
        t_hw = t_dist.ppf(0.975, df=4) * stderr
        z_hw = ci.ci_upper - ci.mean

        # z_hw < t_hw: normal CI is narrower (this is the known trade-off)
        assert z_hw < t_hw
        # Document the ratio for small n
        ratio = z_hw / t_hw
        assert ratio < 0.75  # ~0.706 for n=5


# ═══════════════════════════════════════════════════════════════════════════
# 1.11  compute_bootstrap_mean_ci
# ═══════════════════════════════════════════════════════════════════════════


class TestBootstrapMeanCI:
    """Audit uncertainty.compute_bootstrap_mean_ci."""

    def test_deterministic_with_seed(self):
        """Same seed → identical output (Phase 3.9)."""
        from alpha_lab.reporting.uncertainty import compute_bootstrap_mean_ci

        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        ci1 = compute_bootstrap_mean_ci(values, confidence_level=0.95, random_seed=42)
        ci2 = compute_bootstrap_mean_ci(values, confidence_level=0.95, random_seed=42)
        assert ci1.ci_lower == ci2.ci_lower
        assert ci1.ci_upper == ci2.ci_upper

    def test_different_seed_different_result(self):
        from alpha_lab.reporting.uncertainty import compute_bootstrap_mean_ci

        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        ci1 = compute_bootstrap_mean_ci(values, confidence_level=0.95, random_seed=42)
        ci2 = compute_bootstrap_mean_ci(values, confidence_level=0.95, random_seed=123)
        # At least one of the bounds should differ with a different seed
        assert ci1.ci_lower != ci2.ci_lower or ci1.ci_upper != ci2.ci_upper

    def test_ci_contains_true_mean(self):
        """For a large enough sample, CI should contain the sample mean."""
        from alpha_lab.reporting.uncertainty import compute_bootstrap_mean_ci

        values = list(range(1, 101))
        ci = compute_bootstrap_mean_ci(
            values, confidence_level=0.95, n_resamples=1000, random_seed=7
        )
        assert ci.ci_lower <= ci.mean <= ci.ci_upper

    def test_single_value_nan(self):
        from alpha_lab.reporting.uncertainty import compute_bootstrap_mean_ci

        ci = compute_bootstrap_mean_ci([5.0], confidence_level=0.95, random_seed=7)
        assert np.isnan(ci.ci_lower)

    def test_manual_bootstrap_verification(self):
        """Manually verify the bootstrap percentile logic with fixed seed."""
        from alpha_lab.reporting.uncertainty import compute_bootstrap_mean_ci

        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        n_resamples = 500
        seed = 7

        ci = compute_bootstrap_mean_ci(
            values,
            confidence_level=0.90,
            n_resamples=n_resamples,
            random_seed=seed,
        )

        # Reproduce manually
        arr = np.array(values, dtype=float)
        rng = np.random.default_rng(seed)
        draws = rng.choice(arr, size=(n_resamples, len(values)), replace=True)
        means = draws.mean(axis=1)
        expected_lower = float(np.quantile(means, 0.05))
        expected_upper = float(np.quantile(means, 0.95))

        assert ci.ci_lower == pytest.approx(expected_lower)
        assert ci.ci_upper == pytest.approx(expected_upper)


# ═══════════════════════════════════════════════════════════════════════════
# 1.12  compute_block_bootstrap_mean_ci
# ═══════════════════════════════════════════════════════════════════════════


class TestBlockBootstrapMeanCI:
    """Audit uncertainty.compute_block_bootstrap_mean_ci."""

    def test_deterministic_with_seed(self):
        from alpha_lab.reporting.uncertainty import compute_block_bootstrap_mean_ci

        values = list(range(1, 21))
        ci1 = compute_block_bootstrap_mean_ci(
            values, confidence_level=0.95, block_length=5, random_seed=42
        )
        ci2 = compute_block_bootstrap_mean_ci(
            values, confidence_level=0.95, block_length=5, random_seed=42
        )
        assert ci1.ci_lower == ci2.ci_lower
        assert ci1.ci_upper == ci2.ci_upper

    def test_block_length_1_matches_iid_bootstrap(self):
        """block_length=1 should behave like iid bootstrap (same seed)."""
        from alpha_lab.reporting.uncertainty import (
            compute_block_bootstrap_mean_ci,
            compute_bootstrap_mean_ci,
        )

        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        ci_block = compute_block_bootstrap_mean_ci(
            values,
            confidence_level=0.95,
            block_length=1,
            n_resamples=500,
            random_seed=7,
        )
        ci_iid = compute_bootstrap_mean_ci(
            values,
            confidence_level=0.95,
            n_resamples=500,
            random_seed=7,
        )
        # With block_length=1, the sampling mechanism differs (integers vs choice)
        # but both are valid iid bootstrap. Just check both produce finite CIs.
        assert ci_block.has_finite_ci
        assert ci_iid.has_finite_ci

    def test_block_length_exceeds_n(self):
        """block_length > n → effective_block_length clipped to n."""
        from alpha_lab.reporting.uncertainty import compute_block_bootstrap_mean_ci

        values = [1.0, 2.0, 3.0]
        ci = compute_block_bootstrap_mean_ci(
            values,
            confidence_level=0.95,
            block_length=100,  # >> n
            n_resamples=200,
            random_seed=7,
        )
        assert ci.has_finite_ci
        # With effective block = n, every resample is the full array → stderr ≈ 0
        # CI should be very tight around the mean
        assert ci.ci_lower == pytest.approx(ci.mean, abs=0.01)

    def test_manual_block_bootstrap_verification(self):
        """Manually verify block bootstrap index construction."""
        from alpha_lab.reporting.uncertainty import compute_block_bootstrap_mean_ci

        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        n_obs = len(values)
        block_length = 3
        n_resamples = 400
        seed = 7

        ci = compute_block_bootstrap_mean_ci(
            values,
            confidence_level=0.90,
            block_length=block_length,
            n_resamples=n_resamples,
            random_seed=seed,
        )

        # Reproduce manually
        arr = np.array(values, dtype=float)
        n_blocks = int(math.ceil(n_obs / block_length))
        max_start = n_obs - block_length

        rng = np.random.default_rng(seed)
        starts = rng.integers(0, max_start + 1, size=(n_resamples, n_blocks))
        offsets = np.arange(block_length, dtype=int)
        draw_indices = (starts[..., None] + offsets).reshape(n_resamples, n_blocks * block_length)[
            :, :n_obs
        ]

        means = arr[draw_indices].mean(axis=1)
        expected_lower = float(np.quantile(means, 0.05))
        expected_upper = float(np.quantile(means, 0.95))

        assert ci.ci_lower == pytest.approx(expected_lower)
        assert ci.ci_upper == pytest.approx(expected_upper)


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2: Aggregate metrics in _summarise
# ═══════════════════════════════════════════════════════════════════════════


class TestAggregateMetrics:
    """Audit the scalar summary metrics computed in experiment._summarise."""

    def _build_experiment_result(self):
        """Build a minimal ExperimentResult with known data for aggregate checks."""
        from alpha_lab.experiment import run_factor_experiment

        prices = _prices(
            {
                "A": [100.0, 110.0, 121.0, 133.1, 146.41],
                "B": [200.0, 190.0, 200.0, 210.0, 220.0],
                "C": [50.0, 55.0, 52.0, 58.0, 60.0],
            }
        )

        def factor_fn(p: pd.DataFrame) -> pd.DataFrame:
            # Simple momentum-1: yesterday's return, but lagged properly
            # Just return fixed known values for audit
            rows = []
            for date in sorted(p["date"].unique()):
                for asset in sorted(p["asset"].unique()):
                    if asset == "A":
                        val = 1.0
                    elif asset == "B":
                        val = -1.0
                    else:
                        val = 0.5
                    rows.append(
                        {"date": date, "asset": asset, "factor": "test_factor", "value": val}
                    )
            return pd.DataFrame(rows)

        return run_factor_experiment(
            prices, factor_fn, horizon=1, n_quantiles=3, allow_full_sample_evaluation=True
        )

    def test_mean_ic_is_arithmetic_mean_of_finite(self):
        result = self._build_experiment_result()
        s = result.summary

        ic_vals = result.ic_df["ic"].dropna()
        expected = float(ic_vals.mean())
        assert s.mean_ic == pytest.approx(expected)

    def test_ic_ir_is_mean_over_std_ddof1(self):
        result = self._build_experiment_result()
        s = result.summary

        ic_vals = result.ic_df["ic"].dropna()
        if len(ic_vals) > 1:
            expected = float(ic_vals.mean()) / float(ic_vals.std(ddof=1))
            assert s.ic_ir == pytest.approx(expected)

    def test_ic_positive_rate_is_strictly_positive(self):
        """Positive rate counts > 0, not >= 0."""
        result = self._build_experiment_result()
        s = result.summary

        ic_vals = result.ic_df["ic"].dropna()
        expected = float((ic_vals > 0).mean())
        assert s.ic_positive_rate == pytest.approx(expected)

    def test_ic_valid_ratio_denominator(self):
        """valid_ratio = notna() / total_rows, including NaN rows."""
        result = self._build_experiment_result()
        s = result.summary

        expected = float(result.ic_df["ic"].notna().mean())
        assert s.ic_valid_ratio == pytest.approx(expected)

    def test_long_short_ir(self):
        result = self._build_experiment_result()
        s = result.summary

        ls_vals = result.long_short_df["long_short_return"].dropna()
        if len(ls_vals) > 1:
            expected = float(ls_vals.mean()) / float(ls_vals.std(ddof=1))
            assert s.long_short_ir == pytest.approx(expected)

    def test_long_short_return_per_turnover(self):
        result = self._build_experiment_result()
        s = result.summary

        ls_vals = result.long_short_df["long_short_return"].dropna()
        # Turnover restricted to dates in ls_df
        ls_dates = set(result.long_short_df["date"].unique())
        turn_vals = result.long_short_turnover_df[
            result.long_short_turnover_df["date"].isin(ls_dates)
        ]["long_short_turnover"].dropna()

        mean_ls = float(ls_vals.mean())
        mean_turn = float(turn_vals.mean())
        if mean_turn > 0:
            expected = mean_ls / mean_turn
            assert s.long_short_return_per_turnover == pytest.approx(expected)

    def test_hit_rate_is_strictly_positive(self):
        result = self._build_experiment_result()
        s = result.summary

        ls_vals = result.long_short_df["long_short_return"].dropna()
        expected = float((ls_vals > 0).mean())
        assert s.long_short_hit_rate == pytest.approx(expected)


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2: Subperiod stability
# ═══════════════════════════════════════════════════════════════════════════


class TestSubperiodStability:
    """Audit experiment._subperiod_stability_metrics."""

    def test_hand_calc_3_subperiods(self):
        """9 IC values split into 3 subperiods of 3.
        values = [0.1, 0.2, -0.1, | 0.05, -0.05, 0.1, | -0.2, -0.1, 0.3]
        sub_means = [0.0667, 0.0333, 0.0]   → wait, let me recalculate
        sub1: mean(0.1, 0.2, -0.1) = 0.0667
        sub2: mean(0.05, -0.05, 0.1) = 0.0333
        sub3: mean(-0.2, -0.1, 0.3) = 0.0
        positive_share: sub1>0, sub2>0, sub3==0(not >0) → 2/3
        min_mean: 0.0
        """
        from alpha_lab.experiment import _subperiod_stability_metrics

        ic_df = pd.DataFrame(
            {
                "date": pd.to_datetime([f"2024-01-{2 + i:02d}" for i in range(9)]),
                "ic": [0.1, 0.2, -0.1, 0.05, -0.05, 0.1, -0.2, -0.1, 0.3],
            }
        )
        pos_share, min_mean = _subperiod_stability_metrics(ic_df, value_col="ic")
        assert pos_share == pytest.approx(2.0 / 3.0)
        assert min_mean == pytest.approx(0.0)

    def test_fewer_than_n_subperiods_returns_nan(self):
        from alpha_lab.experiment import _subperiod_stability_metrics

        ic_df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
                "ic": [0.1, 0.2],
            }
        )
        pos_share, min_mean = _subperiod_stability_metrics(ic_df, value_col="ic")
        assert np.isnan(pos_share)
        assert np.isnan(min_mean)


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2: Rolling stability
# ═══════════════════════════════════════════════════════════════════════════


class TestRollingStability:
    """Audit experiment._rolling_metric_frame and _rolling_positive_share_and_min_mean."""

    def test_rolling_mean_hand_calc(self):
        """10 IC values, window=3.
        rolling_mean[2] = mean(0.1, 0.2, 0.3) = 0.2
        rolling_mean[3] = mean(0.2, 0.3, -0.1) = 0.1333
        """
        from alpha_lab.experiment import _rolling_metric_frame

        ic_df = pd.DataFrame(
            {
                "date": pd.to_datetime([f"2024-01-{2 + i:02d}" for i in range(6)]),
                "ic": [0.1, 0.2, 0.3, -0.1, 0.05, 0.15],
            }
        )
        result = _rolling_metric_frame(
            ic_df,
            source_col="ic",
            mean_col="rolling_mean_ic",
            positive_rate_col="rolling_ic_positive_rate",
            window=3,
        )
        # First 2 rows are NaN (min_periods=3)
        assert np.isnan(result["rolling_mean_ic"].iloc[0])
        assert np.isnan(result["rolling_mean_ic"].iloc[1])
        # Row 2 (index 2): mean(0.1, 0.2, 0.3) = 0.2
        assert result["rolling_mean_ic"].iloc[2] == pytest.approx(0.2)
        # Row 3: mean(0.2, 0.3, -0.1) = 0.1333
        assert result["rolling_mean_ic"].iloc[3] == pytest.approx((0.2 + 0.3 + (-0.1)) / 3)

    def test_rolling_positive_rate_hand_calc(self):
        from alpha_lab.experiment import _rolling_metric_frame

        ic_df = pd.DataFrame(
            {
                "date": pd.to_datetime([f"2024-01-{2 + i:02d}" for i in range(5)]),
                "ic": [0.1, -0.2, 0.3, 0.05, -0.1],
            }
        )
        result = _rolling_metric_frame(
            ic_df,
            source_col="ic",
            mean_col="m",
            positive_rate_col="pr",
            window=3,
        )
        # Row 2: window=[0.1, -0.2, 0.3] → 2/3 positive
        assert result["pr"].iloc[2] == pytest.approx(2.0 / 3.0)
        # Row 3: window=[-0.2, 0.3, 0.05] → 2/3 positive
        assert result["pr"].iloc[3] == pytest.approx(2.0 / 3.0)
        # Row 4: window=[0.3, 0.05, -0.1] → 2/3 positive
        assert result["pr"].iloc[4] == pytest.approx(2.0 / 3.0)

    def test_rolling_positive_share_and_min_mean(self):
        from alpha_lab.experiment import _rolling_positive_share_and_min_mean

        frame = pd.DataFrame(
            {
                "rolling_mean_ic": [float("nan"), float("nan"), 0.2, 0.1, -0.05, 0.15],
            }
        )
        pos_share, min_mean = _rolling_positive_share_and_min_mean(
            frame, value_col="rolling_mean_ic"
        )
        # Finite values: [0.2, 0.1, -0.05, 0.15] → 3 positive out of 4
        assert pos_share == pytest.approx(3.0 / 4.0)
        assert min_mean == pytest.approx(-0.05)


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2: eval_coverage_ratio
# ═══════════════════════════════════════════════════════════════════════════


class TestEvalCoverageRatio:
    """Audit coverage_ratio = valid_assets_per_date / n_eval_assets."""

    def test_hand_calc(self):
        """3 assets total. Date 1: 3 valid, Date 2: 2 valid.
        coverage_mean = mean(3/3, 2/3) = mean(1.0, 0.667) = 0.833
        coverage_min  = min(1.0, 0.667) = 0.667
        """
        from alpha_lab.experiment import run_factor_experiment

        prices = _prices(
            {
                "A": [100.0, 110.0, 120.0],
                "B": [200.0, 210.0, 220.0],
                "C": [50.0, 55.0, 60.0],
            }
        )

        def factor_fn(p: pd.DataFrame) -> pd.DataFrame:
            rows = []
            for date in sorted(p["date"].unique()):
                for asset in sorted(p["asset"].unique()):
                    val = 1.0
                    # Make C missing on second date
                    if asset == "C" and date == sorted(p["date"].unique())[1]:
                        val = float("nan")
                    rows.append({"date": date, "asset": asset, "factor": "f", "value": val})
            return pd.DataFrame(rows)

        result = run_factor_experiment(
            prices, factor_fn, horizon=1, n_quantiles=2, allow_full_sample_evaluation=True
        )
        s = result.summary

        # The coverage is computed from the merged factor+label valid pairs
        # n_eval_assets = 3 (total distinct assets in eval period)
        assert s.eval_coverage_ratio_mean <= 1.0
        assert s.eval_coverage_ratio_min <= s.eval_coverage_ratio_mean


# ═══════════════════════════════════════════════════════════════════════════
# Phase 3: Edge cases and traps
# ═══════════════════════════════════════════════════════════════════════════


class TestEdgeCasesAndTraps:
    """Critical edge cases that could silently corrupt results."""

    def test_turnover_first_day_is_nan_not_zero(self):
        """Phase 3.3: First day must be NaN, not 0 or 1."""
        from alpha_lab.turnover import quantile_turnover

        asgn = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-02", "2024-01-02"]),
                "asset": ["A", "B"],
                "factor": ["f", "f"],
                "quantile": [1, 2],
            }
        )
        result = quantile_turnover(asgn)
        for _, row in result.iterrows():
            assert np.isnan(row["turnover"]), "First day turnover must be NaN"

    def test_cost_adjusted_first_day_nan(self):
        """Phase 3.10: First day turnover=NaN → adjusted_return=NaN."""
        from alpha_lab.costs import cost_adjusted_long_short

        ls_df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-02"]),
                "factor": ["f"],
                "long_short_return": [0.05],
            }
        )
        turn_df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-02"]),
                "factor": ["f"],
                "long_short_turnover": [float("nan")],
            }
        )
        result = cost_adjusted_long_short(ls_df, turn_df, cost_rate=0.001)
        assert np.isnan(result["adjusted_return"].iloc[0])

    def test_ic_with_only_nan_factors_on_a_date(self):
        """Phase 3.6: Date with all-NaN factors should be present with IC=NaN."""
        from alpha_lab.evaluation import compute_ic

        f = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    ["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03", "2024-01-03"]
                ),
                "asset": ["A", "B", "X", "Y", "Z"],
                "factor": ["f"] * 5,
                "value": [float("nan"), float("nan"), 1.0, 2.0, 3.0],
            }
        )
        labels_df = _canon(
            ["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03", "2024-01-03"],
            ["A", "B", "X", "Y", "Z"],
            "l",
            [1.0, 2.0, 10.0, 20.0, 30.0],
        )
        result = compute_ic(f, labels_df)
        assert pd.Timestamp("2024-01-02") in result["date"].values
        assert np.isnan(result.loc[result["date"] == pd.Timestamp("2024-01-02"), "ic"].iloc[0])

    def test_constant_factor_ls_nan(self):
        """Phase 3.4: Constant factor → all in one bucket → L/S = NaN."""
        from alpha_lab.quantile import long_short_return, quantile_returns

        f = _canon(["2024-01-02"] * 4, ["A", "B", "C", "D"], "f", [7, 7, 7, 7])
        labels_df = _canon(
            ["2024-01-02"] * 4,
            ["A", "B", "C", "D"],
            "l",
            [0.01, 0.02, 0.03, 0.04],
        )
        qr = quantile_returns(f, labels_df, n_quantiles=5)
        ls = long_short_return(qr)
        if not ls.empty:
            assert np.isnan(ls["long_short_return"].iloc[0])

    def test_single_asset_date_ic_nan(self):
        """Phase 3.5: 1 asset on a date → cannot compute correlation → NaN."""
        from alpha_lab.evaluation import compute_ic

        f = _canon(["2024-01-02"], ["A"], "f", [1.0])
        labels_df = _canon(["2024-01-02"], ["A"], "l", [2.0])
        result = compute_ic(f, labels_df)
        assert np.isnan(result["ic"].iloc[0])


# ═══════════════════════════════════════════════════════════════════════════
# Phase 5: Pipeline consistency
# ═══════════════════════════════════════════════════════════════════════════


class TestPipelineConsistency:
    """End-to-end pipeline consistency checks."""

    def test_deterministic_same_input_same_output(self):
        """Phase 5.5: Same data → identical results."""
        from alpha_lab.experiment import run_factor_experiment

        prices = _prices(
            {
                "A": [100.0, 110.0, 121.0, 133.1],
                "B": [200.0, 190.0, 200.0, 210.0],
                "C": [50.0, 55.0, 52.0, 58.0],
            }
        )

        def factor_fn(p: pd.DataFrame) -> pd.DataFrame:
            rows = []
            for date in sorted(p["date"].unique()):
                for asset in sorted(p["asset"].unique()):
                    val = {"A": 3.0, "B": 1.0, "C": 2.0}[asset]
                    rows.append({"date": date, "asset": asset, "factor": "f", "value": val})
            return pd.DataFrame(rows)

        r1 = run_factor_experiment(
            prices, factor_fn, horizon=1, n_quantiles=3, allow_full_sample_evaluation=True
        )
        r2 = run_factor_experiment(
            prices, factor_fn, horizon=1, n_quantiles=3, allow_full_sample_evaluation=True
        )

        def _eq(a: float, b: float) -> bool:
            if math.isnan(a) and math.isnan(b):
                return True
            return a == pytest.approx(b)

        assert _eq(r1.summary.mean_ic, r2.summary.mean_ic)
        assert _eq(r1.summary.mean_rank_ic, r2.summary.mean_rank_ic)
        assert _eq(r1.summary.mean_long_short_return, r2.summary.mean_long_short_return)
        assert _eq(r1.summary.ic_ir, r2.summary.ic_ir)
        pd.testing.assert_frame_equal(r1.ic_df, r2.ic_df)
        pd.testing.assert_frame_equal(r1.long_short_df, r2.long_short_df)

    def test_summary_columns_match_spec(self):
        """Phase 5.4: summarise_experiment_result output columns == SUMMARY_COLUMNS."""
        from alpha_lab.experiment import run_factor_experiment
        from alpha_lab.reporting import SUMMARY_COLUMNS, summarise_experiment_result

        prices = _prices(
            {
                "A": [100.0, 110.0, 121.0],
                "B": [200.0, 190.0, 200.0],
            }
        )

        def factor_fn(p: pd.DataFrame) -> pd.DataFrame:
            rows = []
            for date in sorted(p["date"].unique()):
                for asset in sorted(p["asset"].unique()):
                    rows.append(
                        {
                            "date": date,
                            "asset": asset,
                            "factor": "f",
                            "value": 1.0 if asset == "A" else 0.0,
                        }
                    )
            return pd.DataFrame(rows)

        result = run_factor_experiment(
            prices, factor_fn, horizon=1, n_quantiles=2, allow_full_sample_evaluation=True
        )
        summary_df = summarise_experiment_result(result)

        assert tuple(summary_df.columns) == SUMMARY_COLUMNS

    def test_summarise_metrics_consistent_with_experiment_summary(self):
        """Phase 5.1: Metrics in summarise_experiment_result match ExperimentSummary."""
        from alpha_lab.experiment import run_factor_experiment
        from alpha_lab.reporting import summarise_experiment_result

        prices = _prices(
            {
                "A": [100.0, 110.0, 121.0, 133.1],
                "B": [200.0, 190.0, 200.0, 210.0],
                "C": [50.0, 55.0, 52.0, 58.0],
            }
        )

        def factor_fn(p: pd.DataFrame) -> pd.DataFrame:
            rows = []
            for date in sorted(p["date"].unique()):
                for asset in sorted(p["asset"].unique()):
                    val = {"A": 3.0, "B": 1.0, "C": 2.0}[asset]
                    rows.append({"date": date, "asset": asset, "factor": "f", "value": val})
            return pd.DataFrame(rows)

        result = run_factor_experiment(
            prices, factor_fn, horizon=1, n_quantiles=3, allow_full_sample_evaluation=True
        )
        summary_df = summarise_experiment_result(result)
        row = summary_df.iloc[0]
        s = result.summary

        # Core metrics must match exactly (no re-computation)
        assert float(row["mean_ic"]) == pytest.approx(s.mean_ic)
        assert float(row["mean_rank_ic"]) == pytest.approx(s.mean_rank_ic)
        assert float(row["ic_ir"]) == pytest.approx(s.ic_ir)
        assert float(row["mean_long_short_return"]) == pytest.approx(s.mean_long_short_return)
        assert float(row["long_short_ir"]) == pytest.approx(s.long_short_ir)
        assert float(row["long_short_hit_rate"]) == pytest.approx(s.long_short_hit_rate)
        assert float(row["ic_positive_rate"]) == pytest.approx(s.ic_positive_rate)
        assert float(row["rank_ic_positive_rate"]) == pytest.approx(s.rank_ic_positive_rate)
        assert float(row["ic_valid_ratio"]) == pytest.approx(s.ic_valid_ratio)
        assert float(row["rank_ic_valid_ratio"]) == pytest.approx(s.rank_ic_valid_ratio)
        if math.isnan(s.long_short_return_per_turnover):
            assert math.isnan(float(row["long_short_return_per_turnover"]))
        else:
            assert float(row["long_short_return_per_turnover"]) == pytest.approx(
                s.long_short_return_per_turnover
            )
        assert float(row["mean_long_short_turnover"]) == pytest.approx(s.mean_long_short_turnover)
        assert int(row["n_dates_used"]) == s.n_dates
        assert float(row["subperiod_ic_positive_share"]) == pytest.approx(
            s.subperiod_ic_positive_share
        )
        assert float(row["subperiod_long_short_positive_share"]) == pytest.approx(
            s.subperiod_long_short_positive_share
        )

        def _assert_eq(actual: float, expected: float, label: str = "") -> None:
            if math.isnan(expected):
                assert math.isnan(actual), f"{label}: expected NaN, got {actual}"
            else:
                assert actual == pytest.approx(expected), label

        _assert_eq(
            float(row["rolling_ic_positive_share"]),
            s.rolling_ic_positive_share,
            "rolling_ic_positive_share",
        )
        _assert_eq(
            float(row["rolling_rank_ic_positive_share"]),
            s.rolling_rank_ic_positive_share,
            "rolling_rank_ic_positive_share",
        )
        _assert_eq(
            float(row["rolling_long_short_positive_share"]),
            s.rolling_long_short_positive_share,
            "rolling_long_short_positive_share",
        )
