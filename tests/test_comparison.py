from __future__ import annotations

import math

import pandas as pd
import pytest

from alpha_lab.comparison import (
    COMPARISON_COLUMNS,
    compare_experiments,
    rank_experiments,
)
from alpha_lab.reporting import SUMMARY_COLUMNS

# ---------------------------------------------------------------------------
# Shared fixture builder
# ---------------------------------------------------------------------------


def _make_summary(
    factor_name: str = "momentum_20d",
    label_name: str = "forward_return_5",
    n_quantiles: int = 5,
    split_description: str = "full_sample",
    mean_ic: float = 0.05,
    mean_rank_ic: float = 0.06,
    ic_ir: float = 1.2,
    mean_long_short_return: float = 0.003,
    long_short_hit_rate: float = 0.55,
    n_dates_used: int = 100,
    mean_long_short_turnover: float = 0.4,
    cost_rate: float = float("nan"),
    mean_cost_adjusted_long_short_return: float = float("nan"),
) -> pd.DataFrame:
    row = {
        "factor_name": factor_name,
        "label_name": label_name,
        "n_quantiles": n_quantiles,
        "split_description": split_description,
        "mean_ic": mean_ic,
        "mean_rank_ic": mean_rank_ic,
        "ic_ir": ic_ir,
        "mean_long_short_return": mean_long_short_return,
        "long_short_hit_rate": long_short_hit_rate,
        "n_dates_used": n_dates_used,
        "mean_long_short_turnover": mean_long_short_turnover,
        "cost_rate": cost_rate,
        "mean_cost_adjusted_long_short_return": mean_cost_adjusted_long_short_return,
    }
    return pd.DataFrame([row], columns=list(SUMMARY_COLUMNS))


# ---------------------------------------------------------------------------
# 1. compare_experiments — basic correctness
# ---------------------------------------------------------------------------


def test_compare_experiments_returns_dataframe():
    result = compare_experiments([_make_summary()])
    assert isinstance(result, pd.DataFrame)


def test_compare_experiments_single_summary_one_row():
    result = compare_experiments([_make_summary()])
    assert len(result) == 1


def test_compare_experiments_multiple_summaries_stacked():
    summaries = [_make_summary(factor_name=f"f{i}") for i in range(3)]
    result = compare_experiments(summaries)
    assert len(result) == 3


def test_compare_experiments_has_comparison_columns():
    result = compare_experiments([_make_summary()])
    assert set(COMPARISON_COLUMNS).issubset(result.columns)


def test_compare_experiments_renames_label_name_to_label_factor():
    result = compare_experiments([_make_summary(label_name="forward_return_5")])
    assert "label_factor" in result.columns
    assert "label_name" not in result.columns
    assert result["label_factor"].iloc[0] == "forward_return_5"


def test_compare_experiments_renames_n_quantiles_to_quantiles():
    result = compare_experiments([_make_summary(n_quantiles=5)])
    assert "quantiles" in result.columns
    assert "n_quantiles" not in result.columns
    assert int(result["quantiles"].iloc[0]) == 5


def test_compare_experiments_values_preserved():
    s = _make_summary(mean_ic=0.07, ic_ir=1.5, mean_long_short_return=0.004)
    result = compare_experiments([s])
    assert float(result["mean_ic"].iloc[0]) == pytest.approx(0.07)
    assert float(result["ic_ir"].iloc[0]) == pytest.approx(1.5)
    assert float(result["mean_long_short_return"].iloc[0]) == pytest.approx(0.004)


def test_compare_experiments_factor_names_preserved():
    summaries = [_make_summary(factor_name=f"f{i}") for i in range(3)]
    result = compare_experiments(summaries)
    assert list(result["factor_name"]) == ["f0", "f1", "f2"]


def test_compare_experiments_nan_cost_adj_preserved():
    result = compare_experiments(
        [_make_summary(mean_cost_adjusted_long_short_return=float("nan"))]
    )
    assert math.isnan(float(result["mean_cost_adjusted_long_short_return"].iloc[0]))


def test_compare_experiments_finite_cost_adj_preserved():
    result = compare_experiments(
        [_make_summary(mean_cost_adjusted_long_short_return=0.0025, cost_rate=0.001)]
    )
    assert float(result["mean_cost_adjusted_long_short_return"].iloc[0]) == pytest.approx(
        0.0025
    )


def test_compare_experiments_index_reset():
    summaries = [_make_summary(factor_name=f"f{i}") for i in range(4)]
    result = compare_experiments(summaries)
    assert list(result.index) == list(range(4))


# ---------------------------------------------------------------------------
# 2. compare_experiments — input validation
# ---------------------------------------------------------------------------


def test_compare_experiments_rejects_empty_list():
    with pytest.raises(ValueError, match="non-empty"):
        compare_experiments([])


def test_compare_experiments_rejects_non_dataframe():
    with pytest.raises(TypeError):
        compare_experiments([{"factor_name": "f"}])  # type: ignore[list-item]


def test_compare_experiments_rejects_missing_columns():
    bad = pd.DataFrame([{"factor_name": "f", "mean_ic": 0.05}])
    with pytest.raises(ValueError, match="missing required columns"):
        compare_experiments([bad])


def test_compare_experiments_schema_mismatch_second_element_raises():
    good = _make_summary(factor_name="f1")
    bad = good.drop(columns=["mean_ic"])
    with pytest.raises(ValueError, match="missing required columns"):
        compare_experiments([good, bad])


def test_compare_experiments_rejects_non_dataframe_in_middle():
    good = _make_summary()
    with pytest.raises(TypeError):
        compare_experiments([good, "not_a_df", good])  # type: ignore[list-item]


def test_compare_experiments_rejects_extra_columns():
    """Summaries with columns beyond SUMMARY_COLUMNS must be rejected."""
    s = _make_summary()
    s_extra = s.copy()
    s_extra["bonus_col"] = 99
    with pytest.raises(ValueError, match="unexpected columns"):
        compare_experiments([s_extra])


def test_compare_experiments_rejects_multi_row_summary():
    """A summary with more than one row must be rejected explicitly."""
    single = _make_summary(factor_name="f1")
    multi = pd.concat([_make_summary(factor_name="f2"), _make_summary(factor_name="f3")],
                      ignore_index=True)
    with pytest.raises(ValueError, match="exactly one row"):
        compare_experiments([single, multi])


def test_compare_experiments_rejects_empty_summary_dataframe():
    empty = pd.DataFrame(columns=list(SUMMARY_COLUMNS))
    with pytest.raises(ValueError, match="exactly one row"):
        compare_experiments([empty])


# ---------------------------------------------------------------------------
# 3. rank_experiments — basic correctness
# ---------------------------------------------------------------------------


def test_rank_experiments_returns_dataframe():
    comp = compare_experiments([_make_summary(), _make_summary()])
    result = rank_experiments(comp, metric="mean_ic")
    assert isinstance(result, pd.DataFrame)


def test_rank_experiments_same_row_count():
    summaries = [_make_summary(factor_name=f"f{i}", mean_ic=0.01 * i) for i in range(5)]
    comp = compare_experiments(summaries)
    ranked = rank_experiments(comp, metric="mean_ic")
    assert len(ranked) == len(comp)


def test_rank_experiments_descending_by_default():
    s_low = _make_summary(factor_name="low", mean_ic=0.01)
    s_mid = _make_summary(factor_name="mid", mean_ic=0.05)
    s_high = _make_summary(factor_name="high", mean_ic=0.10)
    comp = compare_experiments([s_low, s_mid, s_high])
    ranked = rank_experiments(comp, metric="mean_ic")
    assert list(ranked["factor_name"]) == ["high", "mid", "low"]


def test_rank_experiments_ascending():
    s_low = _make_summary(factor_name="low", mean_ic=0.01)
    s_high = _make_summary(factor_name="high", mean_ic=0.10)
    comp = compare_experiments([s_low, s_high])
    ranked = rank_experiments(comp, metric="mean_ic", ascending=True)
    assert list(ranked["factor_name"]) == ["low", "high"]


def test_rank_experiments_nan_last_descending():
    s_nan = _make_summary(factor_name="nan_factor", mean_ic=float("nan"))
    s_real = _make_summary(factor_name="real_factor", mean_ic=0.05)
    comp = compare_experiments([s_nan, s_real])
    ranked = rank_experiments(comp, metric="mean_ic")
    assert ranked["factor_name"].iloc[-1] == "nan_factor"


def test_rank_experiments_nan_last_ascending():
    s_nan = _make_summary(factor_name="nan_factor", mean_ic=float("nan"))
    s_real = _make_summary(factor_name="real_factor", mean_ic=0.05)
    comp = compare_experiments([s_nan, s_real])
    ranked = rank_experiments(comp, metric="mean_ic", ascending=True)
    assert ranked["factor_name"].iloc[-1] == "nan_factor"


def test_rank_experiments_preserves_all_rows_with_nans():
    summaries = [
        _make_summary(factor_name=f"f{i}", mean_ic=(float("nan") if i == 0 else 0.01 * i))
        for i in range(4)
    ]
    comp = compare_experiments(summaries)
    ranked = rank_experiments(comp, metric="mean_ic")
    assert len(ranked) == 4


def test_rank_experiments_index_reset():
    summaries = [_make_summary(factor_name=f"f{i}", mean_ic=0.01 * i) for i in range(3)]
    comp = compare_experiments(summaries)
    ranked = rank_experiments(comp, metric="mean_ic")
    assert list(ranked.index) == list(range(3))


def test_rank_experiments_by_ic_ir():
    s1 = _make_summary(factor_name="f1", ic_ir=0.5)
    s2 = _make_summary(factor_name="f2", ic_ir=2.0)
    comp = compare_experiments([s1, s2])
    ranked = rank_experiments(comp, metric="ic_ir")
    assert ranked["factor_name"].iloc[0] == "f2"


# ---------------------------------------------------------------------------
# 4. rank_experiments — input validation
# ---------------------------------------------------------------------------


def test_rank_experiments_rejects_unknown_metric():
    comp = compare_experiments([_make_summary()])
    with pytest.raises(ValueError, match="not a column"):
        rank_experiments(comp, metric="nonexistent_metric")


def test_rank_experiments_error_message_lists_available_columns():
    comp = compare_experiments([_make_summary()])
    with pytest.raises(ValueError, match="mean_ic"):
        rank_experiments(comp, metric="bad_metric")


def test_rank_experiments_ties_broken_by_factor_name():
    """Equal metric values must produce a deterministic factor_name order."""
    s1 = _make_summary(factor_name="beta", mean_ic=0.05)
    s2 = _make_summary(factor_name="alpha", mean_ic=0.05)
    s3 = _make_summary(factor_name="gamma", mean_ic=0.05)
    comp = compare_experiments([s1, s2, s3])
    ranked = rank_experiments(comp, metric="mean_ic")
    # Tiebreaker is factor_name ascending: alpha < beta < gamma
    assert list(ranked["factor_name"]) == ["alpha", "beta", "gamma"]


def test_rank_experiments_ties_deterministic_ascending():
    s1 = _make_summary(factor_name="z_factor", mean_ic=0.02)
    s2 = _make_summary(factor_name="a_factor", mean_ic=0.02)
    comp = compare_experiments([s1, s2])
    ranked = rank_experiments(comp, metric="mean_ic", ascending=True)
    assert list(ranked["factor_name"]) == ["a_factor", "z_factor"]


def test_rank_experiments_fully_deterministic_when_factor_name_also_ties():
    """When both metric and factor_name tie, remaining columns must still
    produce a deterministic order — no fallback to unspecified pandas behavior.
    Two rows with the same factor_name and same metric but different ic_ir
    values must be ordered by ic_ir (the next tiebreaker in column order).
    """
    # Same factor_name and mean_ic; differ only in ic_ir
    s1 = _make_summary(factor_name="momentum", mean_ic=0.05, ic_ir=0.8)
    s2 = _make_summary(factor_name="momentum", mean_ic=0.05, ic_ir=1.5)
    comp = compare_experiments([s1, s2])
    ranked = rank_experiments(comp, metric="mean_ic")
    # ic_ir is a remaining column sorted ascending as tiebreaker → 0.8 first
    assert float(ranked["ic_ir"].iloc[0]) == pytest.approx(0.8)
    assert float(ranked["ic_ir"].iloc[1]) == pytest.approx(1.5)


def test_rank_experiments_fully_deterministic_repeated_call():
    """Calling rank_experiments twice on the same input must produce
    identical row order — verifying stability across repeated calls."""
    summaries = [
        _make_summary(factor_name="momentum", mean_ic=0.05, ic_ir=float(v))
        for v in [1.0, 0.5, 1.5]
    ]
    comp = compare_experiments(summaries)
    r1 = rank_experiments(comp, metric="mean_ic")
    r2 = rank_experiments(comp, metric="mean_ic")
    assert list(r1["ic_ir"]) == list(r2["ic_ir"])
