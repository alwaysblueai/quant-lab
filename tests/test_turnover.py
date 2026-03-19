from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from alpha_lab.quantile import quantile_assignments
from alpha_lab.turnover import long_short_turnover, quantile_turnover

# ---------------------------------------------------------------------------
# Deterministic fixtures
# ---------------------------------------------------------------------------


def _make_assignments() -> pd.DataFrame:
    """Four dates, three assets (A, B, C), two quantile buckets.

    Date 0:  A→q1, B→q1, C→q2          (first date — baseline)
    Date 1:  A→q1, B→q1, C→q2          (identical to date 0 → zero turnover)
    Date 2:  B→q1, C→q1, A→q2          (A exits q1, C enters q1; A enters q2)
    Date 3:  D→q1, E→q1, F→q2          (complete replacement — 100% turnover)
    """
    dates = pd.date_range("2024-01-01", periods=4, freq="B")
    rows = [
        # date 0
        {"date": dates[0], "asset": "A", "factor": "f", "quantile": 1},
        {"date": dates[0], "asset": "B", "factor": "f", "quantile": 1},
        {"date": dates[0], "asset": "C", "factor": "f", "quantile": 2},
        # date 1 — same membership
        {"date": dates[1], "asset": "A", "factor": "f", "quantile": 1},
        {"date": dates[1], "asset": "B", "factor": "f", "quantile": 1},
        {"date": dates[1], "asset": "C", "factor": "f", "quantile": 2},
        # date 2 — B,C in q1; A in q2
        {"date": dates[2], "asset": "B", "factor": "f", "quantile": 1},
        {"date": dates[2], "asset": "C", "factor": "f", "quantile": 1},
        {"date": dates[2], "asset": "A", "factor": "f", "quantile": 2},
        # date 3 — completely new assets
        {"date": dates[3], "asset": "D", "factor": "f", "quantile": 1},
        {"date": dates[3], "asset": "E", "factor": "f", "quantile": 1},
        {"date": dates[3], "asset": "F", "factor": "f", "quantile": 2},
    ]
    return pd.DataFrame(rows)


def _turnover_at(df: pd.DataFrame, date: pd.Timestamp, q: int) -> float:
    row = df[(df["date"] == date) & (df["quantile"] == q)]
    assert len(row) == 1, f"expected 1 row for date={date}, q={q}, got {len(row)}"
    return float(row["turnover"].iloc[0])


# ---------------------------------------------------------------------------
# 1. quantile_turnover — basic correctness
# ---------------------------------------------------------------------------


def test_quantile_turnover_returns_dataframe():
    df = quantile_turnover(_make_assignments())
    assert isinstance(df, pd.DataFrame)


def test_quantile_turnover_has_expected_columns():
    df = quantile_turnover(_make_assignments())
    assert {"date", "factor", "quantile", "turnover"}.issubset(df.columns)


def test_quantile_turnover_first_date_is_nan():
    assignments = _make_assignments()
    dates = sorted(assignments["date"].unique())
    df = quantile_turnover(assignments)
    first_date_rows = df[df["date"] == dates[0]]
    assert first_date_rows["turnover"].isna().all()


def test_quantile_turnover_zero_when_membership_unchanged():
    assignments = _make_assignments()
    dates = sorted(assignments["date"].unique())
    df = quantile_turnover(assignments)
    # date 1: identical to date 0 → turnover = 0 for both buckets
    assert _turnover_at(df, dates[1], 1) == pytest.approx(0.0)
    assert _turnover_at(df, dates[1], 2) == pytest.approx(0.0)


def test_quantile_turnover_partial_replacement():
    """date 2 q=1: {B,C} vs prev {A,B} → entering={C}, size=2 → 0.5."""
    assignments = _make_assignments()
    dates = sorted(assignments["date"].unique())
    df = quantile_turnover(assignments)
    assert _turnover_at(df, dates[2], 1) == pytest.approx(0.5)


def test_quantile_turnover_complete_replacement():
    """date 2 q=2: {A} vs prev {C} → entering={A}, size=1 → 1.0."""
    assignments = _make_assignments()
    dates = sorted(assignments["date"].unique())
    df = quantile_turnover(assignments)
    assert _turnover_at(df, dates[2], 2) == pytest.approx(1.0)


def test_quantile_turnover_new_assets_is_full_replacement():
    """date 3: entirely new assets → turnover = 1.0 for all buckets."""
    assignments = _make_assignments()
    dates = sorted(assignments["date"].unique())
    df = quantile_turnover(assignments)
    assert _turnover_at(df, dates[3], 1) == pytest.approx(1.0)
    assert _turnover_at(df, dates[3], 2) == pytest.approx(1.0)


def test_quantile_turnover_one_row_per_date_quantile():
    df = quantile_turnover(_make_assignments())
    dupes = df.duplicated(subset=["date", "factor", "quantile"])
    assert not dupes.any()


def test_quantile_turnover_factor_name_preserved():
    df = quantile_turnover(_make_assignments())
    assert (df["factor"] == "f").all()


# ---------------------------------------------------------------------------
# 2. quantile_turnover — edge cases
# ---------------------------------------------------------------------------


def test_quantile_turnover_empty_input_returns_empty():
    empty = pd.DataFrame(columns=["date", "asset", "factor", "quantile"])
    df = quantile_turnover(empty)
    assert df.empty
    assert {"date", "factor", "quantile", "turnover"}.issubset(df.columns)


def test_quantile_turnover_single_date_all_nan():
    """Single date: no prior state → all NaN."""
    single = pd.DataFrame(
        [
            {"date": pd.Timestamp("2024-01-01"), "asset": "A", "factor": "f", "quantile": 1},
            {"date": pd.Timestamp("2024-01-01"), "asset": "B", "factor": "f", "quantile": 2},
        ]
    )
    df = quantile_turnover(single)
    assert df["turnover"].isna().all()


def test_quantile_turnover_rejects_missing_columns():
    bad = pd.DataFrame([{"date": "2024-01-01", "asset": "A", "quantile": 1}])
    with pytest.raises(ValueError, match="missing required columns"):
        quantile_turnover(bad)


def test_quantile_turnover_rejects_duplicate_asset_rows():
    """Duplicate (date, asset) rows must raise rather than silently collapsing."""
    rows = [
        {"date": pd.Timestamp("2024-01-01"), "asset": "A", "factor": "f", "quantile": 1},
        {"date": pd.Timestamp("2024-01-01"), "asset": "A", "factor": "f", "quantile": 1},
    ]
    with pytest.raises(ValueError, match="duplicate"):
        quantile_turnover(pd.DataFrame(rows))


def test_quantile_turnover_rejects_multiple_factor_names():
    rows = [
        {"date": pd.Timestamp("2024-01-01"), "asset": "A", "factor": "f1", "quantile": 1},
        {"date": pd.Timestamp("2024-01-01"), "asset": "B", "factor": "f2", "quantile": 2},
    ]
    with pytest.raises(ValueError, match="exactly one factor name"):
        quantile_turnover(pd.DataFrame(rows))


def test_quantile_turnover_bucket_absent_in_prior_period():
    """A bucket that was absent at t-1 has all entries as new → turnover = 1.0."""
    dates = pd.date_range("2024-01-01", periods=2, freq="B")
    rows = [
        # date 0: only q=1 exists
        {"date": dates[0], "asset": "A", "factor": "f", "quantile": 1},
        # date 1: q=1 unchanged; q=2 appears for the first time
        {"date": dates[1], "asset": "A", "factor": "f", "quantile": 1},
        {"date": dates[1], "asset": "B", "factor": "f", "quantile": 2},
    ]
    df = quantile_turnover(pd.DataFrame(rows))
    assert _turnover_at(df, dates[1], 1) == pytest.approx(0.0)
    assert _turnover_at(df, dates[1], 2) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 3. quantile_turnover — integration with quantile_assignments
# ---------------------------------------------------------------------------


def _make_prices(n_assets: int = 4, n_days: int = 15, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    assets = [f"A{i}" for i in range(n_assets)]
    rows = []
    for asset in assets:
        price = 100.0
        for date in dates:
            price *= 1.0 + rng.normal(0.0, 0.02)
            rows.append({"date": date, "asset": asset, "close": price})
    return pd.DataFrame(rows)


def test_quantile_turnover_from_assignments_has_correct_shape():
    from alpha_lab.factors.momentum import momentum

    prices = _make_prices()
    factors = momentum(prices, window=3)
    assignments = quantile_assignments(factors, n_quantiles=3)
    df = quantile_turnover(assignments)
    n_dates = assignments["date"].nunique()
    n_quantiles_present = assignments["quantile"].nunique()
    # One row per (date, quantile) — max possible = n_dates × n_quantiles_present
    assert len(df) <= n_dates * n_quantiles_present
    assert len(df) > 0


def test_quantile_turnover_in_range_zero_one():
    from alpha_lab.factors.momentum import momentum

    prices = _make_prices(n_assets=6, n_days=20)
    factors = momentum(prices, window=3)
    assignments = quantile_assignments(factors, n_quantiles=4)
    df = quantile_turnover(assignments)
    valid = df["turnover"].dropna()
    assert (valid >= 0.0).all()
    assert (valid <= 1.0).all()


# ---------------------------------------------------------------------------
# 4. long_short_turnover — basic correctness
# ---------------------------------------------------------------------------


def test_long_short_turnover_returns_dataframe():
    qt_df = quantile_turnover(_make_assignments())
    df = long_short_turnover(qt_df)
    assert isinstance(df, pd.DataFrame)


def test_long_short_turnover_has_expected_columns():
    qt_df = quantile_turnover(_make_assignments())
    df = long_short_turnover(qt_df)
    assert {"date", "factor", "long_short_turnover"}.issubset(df.columns)


def test_long_short_turnover_first_date_is_nan():
    assignments = _make_assignments()
    qt_df = quantile_turnover(assignments)
    df = long_short_turnover(qt_df)
    first_date = sorted(df["date"].unique())[0]
    assert math.isnan(float(df[df["date"] == first_date]["long_short_turnover"].iloc[0]))


def test_long_short_turnover_zero_when_both_legs_unchanged():
    """date 1: both q1 and q2 unchanged → L/S turnover = 0."""
    assignments = _make_assignments()
    dates = sorted(assignments["date"].unique())
    qt_df = quantile_turnover(assignments)
    df = long_short_turnover(qt_df)
    val = float(df[df["date"] == dates[1]]["long_short_turnover"].iloc[0])
    assert val == pytest.approx(0.0)


def test_long_short_turnover_average_of_legs():
    """date 2: q1=0.5, q2=1.0 → L/S = 0.75."""
    assignments = _make_assignments()
    dates = sorted(assignments["date"].unique())
    qt_df = quantile_turnover(assignments)
    df = long_short_turnover(qt_df)
    val = float(df[df["date"] == dates[2]]["long_short_turnover"].iloc[0])
    assert val == pytest.approx(0.75)


def test_long_short_turnover_complete_replacement():
    """date 3: both legs 1.0 → L/S = 1.0."""
    assignments = _make_assignments()
    dates = sorted(assignments["date"].unique())
    qt_df = quantile_turnover(assignments)
    df = long_short_turnover(qt_df)
    val = float(df[df["date"] == dates[3]]["long_short_turnover"].iloc[0])
    assert val == pytest.approx(1.0)


def test_long_short_turnover_one_row_per_date():
    qt_df = quantile_turnover(_make_assignments())
    df = long_short_turnover(qt_df)
    assert not df.duplicated(subset=["date", "factor"]).any()


# ---------------------------------------------------------------------------
# 5. long_short_turnover — edge cases
# ---------------------------------------------------------------------------


def test_long_short_turnover_empty_input_returns_empty():
    empty = pd.DataFrame(columns=["date", "factor", "quantile", "turnover"])
    df = long_short_turnover(empty)
    assert df.empty


def test_long_short_turnover_single_bucket_is_nan():
    """Only one quantile bucket occupied → L/S undefined → NaN."""
    rows = [
        {"date": pd.Timestamp("2024-01-01"), "factor": "f", "quantile": 1, "turnover": 0.5},
        {"date": pd.Timestamp("2024-01-02"), "factor": "f", "quantile": 1, "turnover": 0.3},
    ]
    df = long_short_turnover(pd.DataFrame(rows))
    assert df["long_short_turnover"].isna().all()


def test_long_short_turnover_nan_leg_propagates_nan():
    """If either leg is NaN, L/S turnover is NaN."""
    nan = float("nan")
    rows = [
        {"date": pd.Timestamp("2024-01-02"), "factor": "f", "quantile": 1, "turnover": nan},
        {"date": pd.Timestamp("2024-01-02"), "factor": "f", "quantile": 5, "turnover": 0.4},
    ]
    df = long_short_turnover(pd.DataFrame(rows))
    assert math.isnan(float(df["long_short_turnover"].iloc[0]))


def test_long_short_turnover_rejects_missing_columns():
    bad = pd.DataFrame([{"date": "2024-01-01", "quantile": 1, "turnover": 0.5}])
    with pytest.raises(ValueError, match="Missing columns"):
        long_short_turnover(bad)
