from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from alpha_lab.exceptions import AlphaLabConfigError, AlphaLabDataError
from alpha_lab.signal_transforms import (
    apply_min_coverage_gate,
    mad_winsorize_cross_section,
    rank_cross_section,
    winsorize_cross_section,
    zscore_cross_section,
)


def _frame(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["date", "asset", "value"]).assign(
        date=lambda df: pd.to_datetime(df["date"])
    )


# ---------------------------------------------------------------------------
# Schema / input guards
# ---------------------------------------------------------------------------


def test_zscore_requires_value_column():
    df = pd.DataFrame({"date": ["2024-01-01"], "asset": ["a"]})
    with pytest.raises(AlphaLabDataError):
        zscore_cross_section(df)


def test_apply_min_coverage_gate_rejects_out_of_range():
    df = _frame([("2024-01-01", "a", 1.0)])
    with pytest.raises(AlphaLabConfigError):
        apply_min_coverage_gate(df, min_coverage=0.0)
    with pytest.raises(AlphaLabConfigError):
        apply_min_coverage_gate(df, min_coverage=1.5)


# ---------------------------------------------------------------------------
# min_group_size semantics — pinned current behavior
#
# winsorize (both quantile and MAD):  if valid < min_group_size, SKIP → raw values kept
# zscore / rank:                       if valid < min_group_size, NaN-OUT the whole group
#
# This asymmetry is intentional: z-score and rank require a distribution to be
# meaningful; winsorize clips are defined for any sample size ≥1. The tests
# below pin the current behavior so any future change surfaces explicitly.
# ---------------------------------------------------------------------------


def test_winsorize_skips_under_sized_group_keeps_raw_values():
    df = _frame(
        [
            ("2024-01-01", "a", 1.0),
            ("2024-01-01", "b", 100.0),
        ]
    )
    out = winsorize_cross_section(df, min_group_size=3)
    # Raw values flow through untouched when the cross-section is too small.
    assert out.sort_values("asset")["value"].tolist() == [1.0, 100.0]


def test_mad_winsorize_skips_under_sized_group_keeps_raw_values():
    df = _frame(
        [
            ("2024-01-01", "a", 1.0),
            ("2024-01-01", "b", 100.0),
        ]
    )
    out = mad_winsorize_cross_section(df, min_group_size=3)
    assert out.sort_values("asset")["value"].tolist() == [1.0, 100.0]


def test_zscore_nans_out_under_sized_group():
    df = _frame(
        [
            ("2024-01-01", "a", 1.0),
            ("2024-01-01", "b", 100.0),
        ]
    )
    out = zscore_cross_section(df, min_group_size=3)
    assert out["value"].isna().all()


def test_rank_nans_out_under_sized_group():
    df = _frame(
        [
            ("2024-01-01", "a", 1.0),
            ("2024-01-01", "b", 100.0),
        ]
    )
    out = rank_cross_section(df, min_group_size=3)
    assert out["value"].isna().all()


def test_mad_winsorize_rejects_non_positive_k():
    df = _frame([("2024-01-01", "a", 1.0)])
    with pytest.raises(AlphaLabConfigError):
        mad_winsorize_cross_section(df, k=0.0)


# ---------------------------------------------------------------------------
# All-NaN and degenerate cross-sections
# ---------------------------------------------------------------------------


def test_zscore_all_nan_cross_section_stays_nan():
    df = _frame(
        [
            ("2024-01-01", "a", np.nan),
            ("2024-01-01", "b", np.nan),
            ("2024-01-01", "c", np.nan),
        ]
    )
    out = zscore_cross_section(df, min_group_size=3)
    assert out["value"].isna().all()


def test_zscore_constant_cross_section_returns_zeros():
    df = _frame(
        [
            ("2024-01-01", "a", 5.0),
            ("2024-01-01", "b", 5.0),
            ("2024-01-01", "c", 5.0),
        ]
    )
    out = zscore_cross_section(df, min_group_size=3)
    # preprocess.zscore_series returns zeros when std == 0.
    assert (out["value"] == 0.0).all()


def test_rank_handles_ties_via_average_method():
    df = _frame(
        [
            ("2024-01-01", "a", 1.0),
            ("2024-01-01", "b", 1.0),
            ("2024-01-01", "c", 3.0),
        ]
    )
    out = rank_cross_section(df, min_group_size=3, pct=False).sort_values("asset")
    # Ties on 1.0 → average rank of 1 and 2 → 1.5; the third value → rank 3.
    assert out["value"].tolist() == [1.5, 1.5, 3.0]


def test_rank_pct_true_bounded_in_unit_interval():
    df = _frame(
        [
            ("2024-01-01", "a", 1.0),
            ("2024-01-01", "b", 2.0),
            ("2024-01-01", "c", 3.0),
        ]
    )
    out = rank_cross_section(df, min_group_size=3, pct=True)
    values = out["value"].to_numpy()
    assert np.all(values > 0.0)
    assert np.all(values <= 1.0)


# ---------------------------------------------------------------------------
# Winsorize bounds & quantile edges
# ---------------------------------------------------------------------------


def test_winsorize_clips_to_quantile_bounds():
    raw_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 100]
    df = _frame([("2024-01-01", f"a{i}", float(v)) for i, v in enumerate(raw_values)])
    expected_upper = pd.Series(raw_values, dtype=float).quantile(0.9)
    expected_lower = pd.Series(raw_values, dtype=float).quantile(0.1)
    out = winsorize_cross_section(df, lower=0.1, upper=0.9, min_group_size=3)
    assert out["value"].max() <= expected_upper + 1e-9
    assert out["value"].min() >= expected_lower - 1e-9
    # Outlier (100) must have been clipped (i.e. no row equals the raw outlier).
    assert not (out["value"] == 100.0).any()


def test_winsorize_nan_values_pass_through_for_valid_group():
    df = _frame(
        [
            ("2024-01-01", "a", 1.0),
            ("2024-01-01", "b", 2.0),
            ("2024-01-01", "c", 3.0),
            ("2024-01-01", "d", np.nan),
        ]
    )
    out = winsorize_cross_section(df, min_group_size=3).sort_values("asset")
    assert out["value"].iloc[-1] != out["value"].iloc[-1]  # NaN stays NaN
    assert not out["value"].iloc[:3].isna().any()


# ---------------------------------------------------------------------------
# min-coverage gate
# ---------------------------------------------------------------------------


def test_apply_min_coverage_gate_nans_dates_below_threshold():
    df = _frame(
        [
            ("2024-01-01", "a", 1.0),
            ("2024-01-01", "b", np.nan),
            ("2024-01-01", "c", np.nan),
            ("2024-01-02", "a", 1.0),
            ("2024-01-02", "b", 2.0),
            ("2024-01-02", "c", 3.0),
        ]
    )
    out = apply_min_coverage_gate(df, min_coverage=0.5)
    by_date = out.groupby("date")["value"].apply(lambda s: s.notna().sum()).to_dict()
    # Day 1 coverage 1/3 ≈ 0.33 → gate collapses to 0 valid. Day 2 untouched.
    assert by_date[pd.Timestamp("2024-01-01")] == 0
    assert by_date[pd.Timestamp("2024-01-02")] == 3


# ---------------------------------------------------------------------------
# Scope integrity — no new (date, asset) pairs, no dropped pairs, no dupes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "transform",
    [
        lambda df: winsorize_cross_section(df, min_group_size=3),
        lambda df: mad_winsorize_cross_section(df, min_group_size=3),
        lambda df: zscore_cross_section(df, min_group_size=3),
        lambda df: rank_cross_section(df, min_group_size=3),
    ],
)
def test_transform_preserves_date_asset_scope(transform):
    df = _frame(
        [
            ("2024-01-01", "a", 1.0),
            ("2024-01-01", "b", 2.0),
            ("2024-01-01", "c", 3.0),
            ("2024-01-02", "a", 4.0),
            ("2024-01-02", "b", 5.0),
            ("2024-01-02", "c", 6.0),
        ]
    )
    out = transform(df)
    assert set(zip(out["date"], out["asset"], strict=True)) == set(
        zip(df["date"], df["asset"], strict=True)
    )
    scope = out.attrs.get("integrity_scope_check")
    assert scope is not None and scope["status"] == "pass"


def test_enforce_integrity_false_skips_scope_attr():
    df = _frame(
        [
            ("2024-01-01", "a", 1.0),
            ("2024-01-01", "b", 2.0),
            ("2024-01-01", "c", 3.0),
        ]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = zscore_cross_section(df, min_group_size=3, enforce_integrity=False)
    assert "integrity_scope_check" not in out.attrs
