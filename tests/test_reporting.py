from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from alpha_lab.experiment import ExperimentResult, run_factor_experiment
from alpha_lab.factors.momentum import momentum
from alpha_lab.reporting import (
    SUMMARY_COLUMNS,
    export_summary_csv,
    summarise_experiment_result,
    to_obsidian_markdown,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_prices(n_assets: int = 6, n_days: int = 30, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    assets = [f"A{i}" for i in range(n_assets)]
    rows = []
    for asset in assets:
        price = 100.0
        for date in dates:
            price *= 1.0 + rng.normal(0.0, 0.01)
            rows.append({"date": date, "asset": asset, "close": price})
    return pd.DataFrame(rows)


def _momentum_fn(prices: pd.DataFrame) -> pd.DataFrame:
    return momentum(prices, window=5)


def _constant_fn(prices: pd.DataFrame) -> pd.DataFrame:
    dates = pd.to_datetime(prices["date"]).unique()
    assets = prices["asset"].unique()
    rows = [
        {"date": d, "asset": a, "factor": "const", "value": 1.0}
        for d in dates
        for a in assets
    ]
    return pd.DataFrame(rows)


def _standard_result() -> ExperimentResult:
    return run_factor_experiment(_make_prices(), _momentum_fn)


# ---------------------------------------------------------------------------
# 1. summarise_experiment_result — output shape and fields
# ---------------------------------------------------------------------------


def test_summarise_returns_one_row_dataframe():
    result = _standard_result()
    df = summarise_experiment_result(result)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1


def test_summarise_contains_all_summary_columns():
    result = _standard_result()
    df = summarise_experiment_result(result)
    assert set(SUMMARY_COLUMNS).issubset(df.columns)


def test_summarise_columns_in_canonical_order():
    result = _standard_result()
    df = summarise_experiment_result(result)
    assert list(df.columns) == list(SUMMARY_COLUMNS)


def test_summarise_factor_name_matches_factor_df():
    result = _standard_result()
    expected = result.factor_df["factor"].iloc[0]
    df = summarise_experiment_result(result)
    assert df["factor_name"].iloc[0] == expected


def test_summarise_label_name_matches_label_df():
    result = _standard_result()
    expected = result.label_df["factor"].iloc[0]
    df = summarise_experiment_result(result)
    assert df["label_name"].iloc[0] == expected


def test_summarise_label_name_encodes_horizon():
    result = run_factor_experiment(_make_prices(), _momentum_fn, horizon=3)
    df = summarise_experiment_result(result)
    assert df["label_name"].iloc[0] == "forward_return_3"


def test_summarise_n_dates_used_matches_experiment_summary():
    result = _standard_result()
    df = summarise_experiment_result(result)
    assert int(df["n_dates_used"].iloc[0]) == result.summary.n_dates


# ---------------------------------------------------------------------------
# 2. summarise_experiment_result — metric values match experiment summary
# ---------------------------------------------------------------------------


def test_summarise_mean_ic_matches_summary():
    result = _standard_result()
    df = summarise_experiment_result(result)
    assert math.isclose(float(df["mean_ic"].iloc[0]), result.summary.mean_ic)


def test_summarise_mean_rank_ic_matches_summary():
    result = _standard_result()
    df = summarise_experiment_result(result)
    assert math.isclose(float(df["mean_rank_ic"].iloc[0]), result.summary.mean_rank_ic)


def test_summarise_ic_ir_matches_summary():
    result = _standard_result()
    df = summarise_experiment_result(result)
    actual = float(df["ic_ir"].iloc[0])
    if math.isnan(result.summary.ic_ir):
        assert math.isnan(actual)
    else:
        assert math.isclose(actual, result.summary.ic_ir)


def test_summarise_mean_long_short_return_matches_summary():
    result = _standard_result()
    df = summarise_experiment_result(result)
    assert math.isclose(
        float(df["mean_long_short_return"].iloc[0]),
        result.summary.mean_long_short_return,
    )


def test_summarise_long_short_hit_rate_matches_summary():
    result = _standard_result()
    df = summarise_experiment_result(result)
    assert math.isclose(
        float(df["long_short_hit_rate"].iloc[0]),
        result.summary.long_short_hit_rate,
    )


# ---------------------------------------------------------------------------
# 3. summarise_experiment_result — split_description field
# ---------------------------------------------------------------------------


def test_summarise_split_description_full_sample_when_no_dates():
    result = _standard_result()
    df = summarise_experiment_result(result)
    assert df["split_description"].iloc[0] == "full_sample"


def test_summarise_split_description_encodes_both_dates():
    result = run_factor_experiment(
        _make_prices(n_days=40),
        _momentum_fn,
        train_end="2024-01-22",
        test_start="2024-01-25",
    )
    df = summarise_experiment_result(result)
    desc = df["split_description"].iloc[0]
    assert "2024-01-22" in desc
    assert "2024-01-25" in desc


# ---------------------------------------------------------------------------
# 4. summarise_experiment_result — n_quantiles field
# ---------------------------------------------------------------------------


def test_summarise_n_quantiles_matches_runner_parameter():
    """n_quantiles must reflect the exact runner parameter, not the max occupied bucket."""
    result = run_factor_experiment(_make_prices(), _momentum_fn, n_quantiles=7)
    df = summarise_experiment_result(result)
    assert int(df["n_quantiles"].iloc[0]) == 7


def test_summarise_n_quantiles_default_is_five():
    result = _standard_result()  # n_quantiles default = 5
    df = summarise_experiment_result(result)
    assert int(df["n_quantiles"].iloc[0]) == 5


def test_summarise_n_quantiles_independent_of_occupied_buckets():
    """A degenerate cross-section may leave some buckets empty, but n_quantiles
    must still report the configured parameter, not max(quantile)."""
    # 2-asset cross-section with n_quantiles=5: only buckets 1 and 5 are occupied
    result = run_factor_experiment(_make_prices(n_assets=2, n_days=20), _momentum_fn, n_quantiles=5)
    df = summarise_experiment_result(result)
    assert int(df["n_quantiles"].iloc[0]) == 5


# ---------------------------------------------------------------------------
# 5. summarise_experiment_result — degenerate / missing metrics
# ---------------------------------------------------------------------------


def test_summarise_nan_metrics_preserved_for_constant_factor():
    result = run_factor_experiment(_make_prices(), _constant_fn)
    df = summarise_experiment_result(result)
    assert math.isnan(float(df["mean_ic"].iloc[0]))
    assert math.isnan(float(df["mean_rank_ic"].iloc[0]))


def test_summarise_stackable_multiple_results():
    r1 = run_factor_experiment(_make_prices(), _momentum_fn)
    r2 = run_factor_experiment(_make_prices(seed=99), _momentum_fn)
    stacked = pd.concat(
        [summarise_experiment_result(r1), summarise_experiment_result(r2)],
        ignore_index=True,
    )
    assert len(stacked) == 2
    assert list(stacked.columns) == list(SUMMARY_COLUMNS)


def test_summarise_split_description_sourced_from_result():
    """split_description must come from result.train_end/test_start, not caller kwargs."""
    result = run_factor_experiment(
        _make_prices(n_days=40),
        _momentum_fn,
        train_end="2024-01-22",
        test_start="2024-01-25",
    )
    # Call with no extra arguments — split info lives on the result
    df = summarise_experiment_result(result)
    desc = str(df["split_description"].iloc[0])
    assert "2024-01-22" in desc
    assert "2024-01-25" in desc


def test_summarise_split_is_full_sample_when_no_split_used():
    result = _standard_result()
    df = summarise_experiment_result(result)
    assert df["split_description"].iloc[0] == "full_sample"


# ---------------------------------------------------------------------------
# 6. export_summary_csv — file creation and content
# ---------------------------------------------------------------------------


def test_export_summary_csv_creates_file(tmp_path: Path) -> None:
    result = _standard_result()
    df = summarise_experiment_result(result)
    out = tmp_path / "report.csv"
    export_summary_csv(df, out)
    assert out.exists()


def test_export_summary_csv_creates_parent_directories(tmp_path: Path) -> None:
    result = _standard_result()
    df = summarise_experiment_result(result)
    out = tmp_path / "deep" / "nested" / "dir" / "report.csv"
    export_summary_csv(df, out)
    assert out.exists()


def test_export_summary_csv_content_roundtrips(tmp_path: Path) -> None:
    result = _standard_result()
    df = summarise_experiment_result(result)
    out = tmp_path / "report.csv"
    export_summary_csv(df, out)
    loaded = pd.read_csv(out)
    assert list(loaded.columns) == list(SUMMARY_COLUMNS)
    assert len(loaded) == 1
    assert loaded["factor_name"].iloc[0] == df["factor_name"].iloc[0]
    assert loaded["split_description"].iloc[0] == df["split_description"].iloc[0]


def test_export_summary_csv_stacked_rows_roundtrip(tmp_path: Path) -> None:
    r1 = run_factor_experiment(_make_prices(), _momentum_fn)
    r2 = run_factor_experiment(_make_prices(seed=7), _momentum_fn)
    stacked = pd.concat(
        [summarise_experiment_result(r1), summarise_experiment_result(r2)],
        ignore_index=True,
    )
    out = tmp_path / "stacked.csv"
    export_summary_csv(stacked, out)
    loaded = pd.read_csv(out)
    assert len(loaded) == 2


def test_export_summary_csv_rejects_non_dataframe(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="DataFrame"):
        export_summary_csv({"mean_ic": 0.05}, tmp_path / "x.csv")  # type: ignore[arg-type]


def test_export_summary_csv_rejects_empty_dataframe(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="empty"):
        export_summary_csv(pd.DataFrame(), tmp_path / "x.csv")


def test_export_summary_csv_rejects_dataframe_with_wrong_columns(tmp_path: Path) -> None:
    """Exporting an unrelated DataFrame that happens to be non-empty must fail."""
    unrelated = pd.DataFrame([{"foo": 1, "bar": 2}])
    with pytest.raises(ValueError, match="missing expected columns"):
        export_summary_csv(unrelated, tmp_path / "x.csv")


def test_export_summary_csv_accepts_path_string(tmp_path: Path) -> None:
    result = _standard_result()
    df = summarise_experiment_result(result)
    out = str(tmp_path / "str_path.csv")
    export_summary_csv(df, out)
    assert Path(out).exists()


# ---------------------------------------------------------------------------
# 7. to_obsidian_markdown — structure and content
# ---------------------------------------------------------------------------


def test_obsidian_markdown_returns_string():
    result = _standard_result()
    md = to_obsidian_markdown(result)
    assert isinstance(md, str)


def test_obsidian_markdown_ends_with_newline():
    result = _standard_result()
    md = to_obsidian_markdown(result)
    assert md.endswith("\n")


def test_obsidian_markdown_contains_h1_title():
    result = _standard_result()
    md = to_obsidian_markdown(result, title="My Test Factor")
    assert "# My Test Factor" in md


def test_obsidian_markdown_default_title_contains_factor_name():
    result = _standard_result()
    md = to_obsidian_markdown(result)
    factor_name = result.factor_df["factor"].iloc[0]
    assert factor_name in md


def test_obsidian_markdown_contains_factor_name():
    result = _standard_result()
    factor_name = result.factor_df["factor"].iloc[0]
    md = to_obsidian_markdown(result)
    assert factor_name in md


def test_obsidian_markdown_contains_label_name():
    result = _standard_result()
    label_name = result.label_df["factor"].iloc[0]
    md = to_obsidian_markdown(result)
    assert label_name in md


def test_obsidian_markdown_contains_split_description():
    result = run_factor_experiment(
        _make_prices(n_days=40),
        _momentum_fn,
        train_end="2024-01-22",
        test_start="2024-01-25",
    )
    md = to_obsidian_markdown(result)
    assert "2024-01-22" in md
    assert "2024-01-25" in md


def test_obsidian_markdown_full_sample_when_no_split():
    result = _standard_result()
    md = to_obsidian_markdown(result)
    assert "full_sample" in md


def test_obsidian_markdown_contains_summary_metrics_section():
    result = _standard_result()
    md = to_obsidian_markdown(result)
    assert "## Summary Metrics" in md


def test_obsidian_markdown_contains_mean_ic_value():
    result = _standard_result()
    md = to_obsidian_markdown(result)
    assert "Mean IC" in md


def test_obsidian_markdown_contains_ic_ir_value():
    result = _standard_result()
    md = to_obsidian_markdown(result)
    assert "IC IR" in md


def test_obsidian_markdown_contains_interpretation_section():
    result = _standard_result()
    md = to_obsidian_markdown(result)
    assert "## Interpretation" in md


def test_obsidian_markdown_contains_next_steps_section():
    result = _standard_result()
    md = to_obsidian_markdown(result)
    assert "## Next Steps" in md


def test_obsidian_markdown_contains_notes_when_provided():
    result = _standard_result()
    md = to_obsidian_markdown(result, notes="Needs walk-forward validation.")
    assert "## Notes" in md
    assert "Needs walk-forward validation." in md


def test_obsidian_markdown_no_notes_section_when_omitted():
    result = _standard_result()
    md = to_obsidian_markdown(result)
    assert "## Notes" not in md


def test_obsidian_markdown_nan_metrics_render_as_dash():
    result = run_factor_experiment(_make_prices(), _constant_fn)
    md = to_obsidian_markdown(result)
    # constant factor → NaN IC → rendered as em dash
    assert "\u2014" in md


def test_obsidian_markdown_eval_dates_present():
    result = _standard_result()
    md = to_obsidian_markdown(result)
    assert str(result.summary.n_dates) in md


# ---------------------------------------------------------------------------
# 8. Consistency with experiment runner
# ---------------------------------------------------------------------------


def test_summary_df_is_consistent_with_experiment_result():
    """All numeric fields in the summary row must match the experiment summary exactly."""
    result = _standard_result()
    df = summarise_experiment_result(result)
    s = result.summary

    def _close_or_both_nan(a: float, b: float) -> bool:
        if math.isnan(a) and math.isnan(b):
            return True
        if math.isnan(a) or math.isnan(b):
            return False
        return math.isclose(a, b)

    assert _close_or_both_nan(float(df["mean_ic"].iloc[0]), s.mean_ic)
    assert _close_or_both_nan(float(df["mean_rank_ic"].iloc[0]), s.mean_rank_ic)
    assert _close_or_both_nan(float(df["ic_ir"].iloc[0]), s.ic_ir)
    assert _close_or_both_nan(float(df["mean_long_short_return"].iloc[0]), s.mean_long_short_return)
    assert _close_or_both_nan(float(df["long_short_hit_rate"].iloc[0]), s.long_short_hit_rate)
    assert int(df["n_dates_used"].iloc[0]) == s.n_dates
