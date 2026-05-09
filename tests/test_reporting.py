from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from alpha_lab.costs import cost_adjusted_long_short
from alpha_lab.experiment import ExperimentResult, run_factor_experiment
from alpha_lab.factors.momentum import momentum
from alpha_lab.reporting import (
    SUMMARY_COLUMNS,
    export_summary_csv,
    summarise_experiment_result,
    to_obsidian_markdown,
)
from alpha_lab.reporting.factor_verdict import FACTOR_VERDICT_TAXONOMY
from alpha_lab.reporting.research_tearsheet import (
    _build_annual_axis_ticks,
    _build_ic_distribution_chart,
    _build_ic_timeseries_with_cumulative_chart,
    build_research_tearsheet_payload,
)
from alpha_lab.research_evaluation_config import ResearchEvaluationConfig, UncertaintyConfig

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
    rows = [{"date": d, "asset": a, "factor": "const", "value": 1.0} for d in dates for a in assets]
    return pd.DataFrame(rows)


def _standard_result() -> ExperimentResult:
    return run_factor_experiment(_make_prices(), _momentum_fn)


def _close_or_both_nan(a: float, b: float) -> bool:
    if math.isnan(a) and math.isnan(b):
        return True
    if math.isnan(a) or math.isnan(b):
        return False
    return math.isclose(a, b)


def test_tearsheet_date_ticks_use_months_for_one_year_window():
    dates = pd.date_range("2024-01-01", periods=13, freq="MS")
    ticks = _build_annual_axis_ticks(
        x_values=[float(idx) for idx in range(len(dates))],
        x_labels=[date.strftime("%Y-%m-%d") for date in dates],
    )

    labels = [label for _, label in ticks]
    assert "2024.2" in labels
    assert "2024.12" in labels
    assert len(labels) >= 10


def test_tearsheet_date_ticks_use_years_for_ten_year_window():
    dates = pd.date_range("2016-06-01", "2026-06-01", freq="MS")
    ticks = _build_annual_axis_ticks(
        x_values=[float(idx) for idx in range(len(dates))],
        x_labels=[date.strftime("%Y-%m-%d") for date in dates],
    )

    labels = [label for _, label in ticks]
    assert "2017.6" in labels
    assert "2016.7" not in labels
    assert len(labels) <= 12


def test_tearsheet_date_ticks_support_compact_yyyymmdd_labels():
    dates = pd.date_range("2024-01-01", periods=13, freq="MS")
    ticks = _build_annual_axis_ticks(
        x_values=[float(idx) for idx in range(len(dates))],
        x_labels=[date.strftime("%Y%m%d") for date in dates],
    )

    labels = [label for _, label in ticks]
    assert "2024.2" in labels
    assert "2024.12" in labels
    assert len(labels) >= 10


def test_tearsheet_ic_chart_falls_back_to_ic_when_rankic_empty() -> None:
    ic_timeseries = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=3, freq="D"),
            "rank_ic": [np.nan, np.nan, np.nan],
            "ic": [0.10, -0.05, 0.20],
        }
    )

    chart = _build_ic_timeseries_with_cumulative_chart(
        artifacts={"ic_timeseries": ic_timeseries}
    )

    assert chart is not None
    assert chart["series"][0]["name"] == "ic"
    assert chart["series"][0]["points"] == [
        ["2024-01-01", 0.10],
        ["2024-01-02", -0.05],
        ["2024-01-03", 0.20],
    ]
    assert chart["series"][1]["name"] == "cumulative_ic"
    assert [point[0] for point in chart["series"][1]["points"]] == [
        "2024-01-01",
        "2024-01-02",
        "2024-01-03",
    ]
    assert [point[1] for point in chart["series"][1]["points"]] == pytest.approx(
        [0.10, 0.05, 0.25]
    )


def test_tearsheet_ic_distribution_falls_back_to_ic_when_rankic_empty() -> None:
    ic_timeseries = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=4, freq="D"),
            "rank_ic": [np.nan, np.nan, np.nan, np.nan],
            "ic": [0.10, -0.05, 0.20, 0.03],
        }
    )

    chart = _build_ic_distribution_chart(artifacts={"ic_timeseries": ic_timeseries})

    assert chart is not None
    assert chart["series"][0]["name"] == "ic"
    assert sum(bin_["count"] for bin_ in chart["series"][0]["bins"]) == 4


def test_tearsheet_ic_distribution_constant_series_has_renderable_bin() -> None:
    ic_timeseries = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=4, freq="D"),
            "rank_ic": [0.25, 0.25, 0.25, 0.25],
            "ic": [0.10, -0.05, 0.20, 0.03],
        }
    )

    chart = _build_ic_distribution_chart(artifacts={"ic_timeseries": ic_timeseries})

    assert chart is not None
    bins = chart["series"][0]["bins"]
    assert len(bins) == 1
    assert bins[0]["right"] > bins[0]["left"]
    assert bins[0]["count"] == 4


def test_tearsheet_alias_falls_back_when_primary_metric_is_nan(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "metrics": {
                    "coverage_mean": float("nan"),
                    "coverage_min": float("nan"),
                    "factor_verdict": "review",
                    "promotion_decision": "hold",
                },
                "coverage_by_date_summary": {
                    "mean_coverage": 0.75,
                    "min_coverage": 0.50,
                },
            }
        ),
        encoding="utf-8",
    )

    payload = build_research_tearsheet_payload(metrics_path=metrics_path)

    setup_metrics = payload["sections"]["setup"]["metrics"]
    assert setup_metrics["coverage_mean"] == pytest.approx(0.75)
    assert payload["meta"]["field_aliases"]["coverage_mean"] == (
        "coverage_by_date_summary.mean_coverage"
    )


def test_tearsheet_payload_chart_inputs_match_artifact_csvs(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "metrics": {
                    "factor_name": "fixture_factor",
                    "direction": "long",
                    "target_horizon": 5,
                    "factor_verdict": "review",
                    "promotion_decision": "hold",
                    "mean_rank_ic": 0.10,
                    "ic_ir": 1.25,
                    "ic_positive_rate": 0.75,
                    "mean_long_short_return": 0.025,
                    "group_monotonicity_share": 1.0,
                    "group_monotonicity_qtop_qbottom": 0.025,
                }
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03"],
            "rank_ic": [0.10, -0.05],
            "ic": [0.20, 0.30],
        }
    ).to_csv(tmp_path / "ic_timeseries.csv", index=False)
    pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"],
            "group": [1, 2, 1, 2],
            "group_return": [0.01, 0.03, -0.01, 0.02],
        }
    ).to_csv(tmp_path / "group_returns.csv", index=False)
    pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03"],
            "turnover": [float("nan"), 0.5],
        }
    ).to_csv(tmp_path / "turnover.csv", index=False)
    pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03"],
            "coverage": [0.8, 0.9],
        }
    ).to_csv(tmp_path / "coverage.csv", index=False)

    payload = build_research_tearsheet_payload(metrics_path=metrics_path)
    signal_charts = payload["sections"]["signal"]["charts"]
    appendix_charts = payload["appendix"]["charts"]
    chart_by_title = {chart["title"]: chart for chart in [*signal_charts, *appendix_charts]}

    ic_chart = chart_by_title["IC Time Series + Cumulative IC"]
    assert ic_chart["series"][0]["name"] == "rank_ic"
    assert ic_chart["series"][0]["points"] == [["2024-01-02", 0.10], ["2024-01-03", -0.05]]
    assert [point[0] for point in ic_chart["series"][1]["points"]] == [
        "2024-01-02",
        "2024-01-03",
    ]
    assert [point[1] for point in ic_chart["series"][1]["points"]] == pytest.approx(
        [0.10, 0.05]
    )

    nav_chart = chart_by_title["Cumulative Long-Short NAV"]
    assert [point[0] for point in nav_chart["series"][0]["points"]] == [
        "2024-01-02",
        "2024-01-03",
    ]
    assert [point[1] for point in nav_chart["series"][0]["points"]] == pytest.approx(
        [1.02, 1.0506]
    )

    group_bar = chart_by_title["Group Mean Return"]["series"][0]["bars"]
    assert [bar["group"] for bar in group_bar] == ["Q1", "Q2"]
    assert [bar["value"] for bar in group_bar] == pytest.approx([0.0, 0.025])

    turnover_chart = chart_by_title["Turnover Time Series"]
    assert turnover_chart["series"][0]["points"] == [["2024-01-03", 0.5]]
    coverage_chart = chart_by_title["Coverage Time Series"]
    assert coverage_chart["series"][0]["points"] == [
        ["2024-01-02", 0.8],
        ["2024-01-03", 0.9],
    ]


def test_tearsheet_payload_chart_inputs_cover_all_artifact_curves(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "metrics": {
                    "factor_name": "chart_fixture",
                    "direction": "long",
                    "target_horizon": 5,
                    "factor_verdict": "review",
                    "promotion_decision": "hold",
                    "mean_rank_ic": 0.08,
                    "ic_ir": 1.10,
                    "ic_positive_rate": 0.67,
                    "mean_long_short_return": 0.02,
                    "group_monotonicity_share": 1.0,
                    "group_monotonicity_qtop_qbottom": 0.02,
                }
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "backtest_result.json").write_text(
        json.dumps(
            {
                "summary": {
                    "nav_points": [
                        ["2024-01-02", 1.00],
                        ["2024-01-03", 1.05],
                        ["2024-01-04", 1.03],
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03", "2024-01-04"],
            "rank_ic": [0.10, -0.05, 0.20],
            "ic": [0.20, 0.30, 0.40],
        }
    ).to_csv(tmp_path / "ic_timeseries.csv", index=False)
    pd.DataFrame(
        {
            "date": [
                "2024-01-02",
                "2024-01-02",
                "2024-01-03",
                "2024-01-03",
                "2024-01-04",
                "2024-01-04",
            ],
            "group": [1, 2, 1, 2, 1, 2],
            "group_return": [0.01, 0.03, -0.02, 0.01, 0.00, 0.02],
        }
    ).to_csv(tmp_path / "group_returns.csv", index=False)
    pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03", "2024-01-04"],
            "rolling_mean_ic": [0.10, 0.15, 0.20],
            "rolling_mean_rank_ic": [0.05, 0.10, 0.12],
        }
    ).to_csv(tmp_path / "rolling_stability.csv", index=False)
    pd.DataFrame(
        {
            "horizon": [1, 2, 5],
            "mean_ic": [0.20, 0.15, 0.10],
            "mean_rank_ic": [0.18, 0.12, 0.05],
        }
    ).to_csv(tmp_path / "ic_decay.csv", index=False)
    pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03", "2024-01-04"],
            "turnover": [float("nan"), 0.40, 0.60],
        }
    ).to_csv(tmp_path / "turnover.csv", index=False)
    pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03", "2024-01-04"],
            "coverage": [0.80, 0.90, 1.00],
        }
    ).to_csv(tmp_path / "coverage.csv", index=False)

    payload = build_research_tearsheet_payload(metrics_path=metrics_path)
    charts = [
        *payload["sections"]["signal"]["charts"],
        *payload["sections"]["stability"]["charts"],
        *payload["appendix"]["charts"],
    ]
    chart_by_title = {chart["title"]: chart for chart in charts}

    assert set(chart_by_title) == {
        "Cumulative Long-Short NAV",
        "IC Time Series + Cumulative IC",
        "Quantile Cumulative Returns",
        "Group Mean Return",
        "Rolling IC / RankIC",
        "IC Decay",
        "IC Distribution",
        "Turnover Time Series",
        "Coverage Time Series",
    }

    assert chart_by_title["Cumulative Long-Short NAV"]["series"][0]["points"] == [
        ["2024-01-02", 1.00],
        ["2024-01-03", 1.05],
        ["2024-01-04", 1.03],
    ]

    ic_chart = chart_by_title["IC Time Series + Cumulative IC"]
    assert ic_chart["series"][0]["name"] == "rank_ic"
    assert ic_chart["series"][0]["points"] == [
        ["2024-01-02", 0.10],
        ["2024-01-03", -0.05],
        ["2024-01-04", 0.20],
    ]
    assert [point[1] for point in ic_chart["series"][1]["points"]] == pytest.approx(
        [0.10, 0.05, 0.25]
    )

    quantile_series = {
        series["name"]: series["points"]
        for series in chart_by_title["Quantile Cumulative Returns"]["series"]
    }
    assert [point[1] for point in quantile_series["Q1"]] == pytest.approx(
        [1.01, 0.9898, 0.9898]
    )
    assert [point[1] for point in quantile_series["Q2"]] == pytest.approx(
        [1.03, 1.0403, 1.061106]
    )

    group_bars = chart_by_title["Group Mean Return"]["series"][0]["bars"]
    assert [bar["group"] for bar in group_bars] == ["Q1", "Q2"]
    assert [bar["value"] for bar in group_bars] == pytest.approx([-0.01 / 3.0, 0.02])

    rolling_series = {
        series["name"]: series["points"]
        for series in chart_by_title["Rolling IC / RankIC"]["series"]
    }
    assert rolling_series["rolling_ic"] == [
        ["2024-01-02", 0.10],
        ["2024-01-03", 0.15],
        ["2024-01-04", 0.20],
    ]
    assert rolling_series["rolling_rank_ic"] == [
        ["2024-01-02", 0.05],
        ["2024-01-03", 0.10],
        ["2024-01-04", 0.12],
    ]

    decay_series = {
        series["name"]: series["points"] for series in chart_by_title["IC Decay"]["series"]
    }
    assert decay_series["mean_ic"] == [[1.0, 0.20], [2.0, 0.15], [5.0, 0.10]]
    assert decay_series["mean_rank_ic"] == [[1.0, 0.18], [2.0, 0.12], [5.0, 0.05]]

    ic_distribution = chart_by_title["IC Distribution"]["series"][0]
    assert ic_distribution["name"] == "rank_ic"
    assert sum(bin_["count"] for bin_ in ic_distribution["bins"]) == 3

    assert chart_by_title["Turnover Time Series"]["series"][0]["points"] == [
        ["2024-01-03", 0.40],
        ["2024-01-04", 0.60],
    ]
    assert chart_by_title["Coverage Time Series"]["series"][0]["points"] == [
        ["2024-01-02", 0.80],
        ["2024-01-03", 0.90],
        ["2024-01-04", 1.00],
    ]


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


def test_summarise_mean_mutual_information_matches_summary() -> None:
    result = _standard_result()
    df = summarise_experiment_result(result)
    assert math.isclose(
        float(df["mean_mutual_information"].iloc[0]),
        result.summary.mean_mutual_information,
    )


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


def test_summarise_long_short_ir_matches_summary():
    result = _standard_result()
    df = summarise_experiment_result(result)
    actual = float(df["long_short_ir"].iloc[0])
    if math.isnan(result.summary.long_short_ir):
        assert math.isnan(actual)
    else:
        assert math.isclose(actual, result.summary.long_short_ir)


def test_summarise_cost_adjusted_return_excludes_initial_nan_turnover() -> None:
    result = _standard_result()
    cost_rate = 0.001

    summary_df = summarise_experiment_result(result, cost_rate=cost_rate)
    adjusted = cost_adjusted_long_short(
        result.long_short_df,
        result.long_short_turnover_df,
        cost_rate=cost_rate,
    ).sort_values("date", kind="mergesort")

    assert pd.isna(adjusted["turnover"].iloc[0])
    assert pd.isna(adjusted["adjusted_return"].iloc[0])
    assert adjusted["adjusted_return"].notna().sum() == (
        adjusted["long_short_return"].notna().sum() - 1
    )
    assert float(summary_df["mean_cost_adjusted_long_short_return"].iloc[0]) == pytest.approx(
        float(adjusted["adjusted_return"].dropna().mean())
    )


def test_summarise_subperiod_metrics_match_summary():
    result = _standard_result()
    df = summarise_experiment_result(result)
    assert _close_or_both_nan(
        float(df["subperiod_ic_positive_share"].iloc[0]),
        result.summary.subperiod_ic_positive_share,
    )
    assert _close_or_both_nan(
        float(df["subperiod_long_short_positive_share"].iloc[0]),
        result.summary.subperiod_long_short_positive_share,
    )


def test_summarise_rolling_metrics_match_summary():
    result = _standard_result()
    df = summarise_experiment_result(result)
    assert _close_or_both_nan(
        float(df["rolling_ic_positive_share"].iloc[0]),
        result.summary.rolling_ic_positive_share,
    )
    assert _close_or_both_nan(
        float(df["rolling_rank_ic_positive_share"].iloc[0]),
        result.summary.rolling_rank_ic_positive_share,
    )
    assert _close_or_both_nan(
        float(df["rolling_long_short_positive_share"].iloc[0]),
        result.summary.rolling_long_short_positive_share,
    )
    assert _close_or_both_nan(
        float(df["rolling_ic_min_mean"].iloc[0]),
        result.summary.rolling_ic_min_mean,
    )


def test_summarise_instability_flags_serialized_as_semicolon_list():
    result = _standard_result()
    df = summarise_experiment_result(result)
    expected = ";".join(result.summary.instability_flags)
    assert str(df["instability_flags"].iloc[0]) == expected


def test_summarise_rolling_instability_flags_serialized_as_semicolon_list():
    result = _standard_result()
    df = summarise_experiment_result(result)
    expected = ";".join(result.summary.rolling_instability_flags)
    assert str(df["rolling_instability_flags"].iloc[0]) == expected


def test_summarise_factor_verdict_fields_are_populated():
    result = _standard_result()
    df = summarise_experiment_result(result)
    verdict = str(df["factor_verdict"].iloc[0])
    reasons = str(df["factor_verdict_reasons"].iloc[0])
    assert verdict in FACTOR_VERDICT_TAXONOMY
    assert reasons.strip() != ""


def test_summarise_uncertainty_ci_fields_are_present():
    result = _standard_result()
    df = summarise_experiment_result(result)
    for column in (
        "mean_ic_ci_lower",
        "mean_ic_ci_upper",
        "mean_rank_ic_ci_lower",
        "mean_rank_ic_ci_upper",
        "mean_long_short_return_ci_lower",
        "mean_long_short_return_ci_upper",
        "uncertainty_flags",
        "uncertainty_method",
        "uncertainty_confidence_level",
        "uncertainty_bootstrap_resamples",
        "uncertainty_bootstrap_block_length",
    ):
        assert column in df.columns


def test_summarise_uncertainty_flags_show_unavailable_ci_for_constant_factor():
    result = run_factor_experiment(_make_prices(), _constant_fn)
    df = summarise_experiment_result(result)
    flags = str(df["uncertainty_flags"].iloc[0])
    assert "ic_ci_unavailable" in flags


def test_summarise_defaults_to_normal_uncertainty_mode() -> None:
    df = summarise_experiment_result(_standard_result())
    assert str(df["uncertainty_method"].iloc[0]) == "normal"
    assert float(df["uncertainty_confidence_level"].iloc[0]) == pytest.approx(0.95)
    assert math.isnan(float(df["uncertainty_bootstrap_resamples"].iloc[0]))
    assert math.isnan(float(df["uncertainty_bootstrap_block_length"].iloc[0]))


def test_summarise_propagates_bootstrap_uncertainty_metadata() -> None:
    cfg = ResearchEvaluationConfig(
        uncertainty=UncertaintyConfig(
            method="bootstrap",
            bootstrap_resamples=220,
            bootstrap_confidence_level=0.90,
            bootstrap_random_seed=17,
        )
    )
    df = summarise_experiment_result(_standard_result(), evaluation_config=cfg)
    assert str(df["uncertainty_method"].iloc[0]) == "bootstrap"
    assert float(df["uncertainty_confidence_level"].iloc[0]) == pytest.approx(0.90)
    assert int(df["uncertainty_bootstrap_resamples"].iloc[0]) == 220
    assert math.isnan(float(df["uncertainty_bootstrap_block_length"].iloc[0]))


def test_summarise_propagates_block_bootstrap_uncertainty_metadata() -> None:
    cfg = ResearchEvaluationConfig(
        uncertainty=UncertaintyConfig(
            method="block_bootstrap",
            bootstrap_resamples=180,
            bootstrap_confidence_level=0.90,
            bootstrap_random_seed=13,
            block_bootstrap_block_length=6,
        )
    )
    df = summarise_experiment_result(_standard_result(), evaluation_config=cfg)
    assert str(df["uncertainty_method"].iloc[0]) == "block_bootstrap"
    assert float(df["uncertainty_confidence_level"].iloc[0]) == pytest.approx(0.90)
    assert int(df["uncertainty_bootstrap_resamples"].iloc[0]) == 180
    assert int(df["uncertainty_bootstrap_block_length"].iloc[0]) == 6


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
    assert "## 摘要指标" in md


def test_obsidian_markdown_contains_factor_verdict_lines():
    result = _standard_result()
    md = to_obsidian_markdown(result)
    assert "Factor Verdict" in md
    assert "Verdict Reasons" in md


def test_obsidian_markdown_contains_uncertainty_lines():
    result = _standard_result()
    md = to_obsidian_markdown(result)
    assert "Mean IC 95% CI" in md
    assert "Mean Rank IC 95% CI" in md
    assert "Mean L/S Return 95% CI" in md
    assert "Uncertainty Flags" in md


def test_obsidian_markdown_contains_rolling_stability_lines():
    result = _standard_result()
    md = to_obsidian_markdown(result)
    assert "Rolling Stability Window" in md
    assert "Worst Rolling Mean" in md
    assert "Rolling Stability Flags" in md


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
    assert "## 解释" in md


def test_obsidian_markdown_contains_next_steps_section():
    result = _standard_result()
    md = to_obsidian_markdown(result)
    assert "## 下一步" in md


def test_obsidian_markdown_contains_notes_when_provided():
    result = _standard_result()
    md = to_obsidian_markdown(result, notes="Needs walk-forward validation.")
    assert "## 备注" in md
    assert "Needs walk-forward validation." in md


def test_obsidian_markdown_no_notes_section_when_omitted():
    result = _standard_result()
    md = to_obsidian_markdown(result)
    assert "## 备注" not in md


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

    assert _close_or_both_nan(float(df["mean_ic"].iloc[0]), s.mean_ic)
    assert _close_or_both_nan(float(df["mean_rank_ic"].iloc[0]), s.mean_rank_ic)
    assert _close_or_both_nan(float(df["ic_ir"].iloc[0]), s.ic_ir)
    assert _close_or_both_nan(float(df["ic_positive_rate"].iloc[0]), s.ic_positive_rate)
    assert _close_or_both_nan(float(df["rank_ic_positive_rate"].iloc[0]), s.rank_ic_positive_rate)
    assert _close_or_both_nan(float(df["ic_valid_ratio"].iloc[0]), s.ic_valid_ratio)
    assert _close_or_both_nan(float(df["rank_ic_valid_ratio"].iloc[0]), s.rank_ic_valid_ratio)
    assert _close_or_both_nan(float(df["mean_long_short_return"].iloc[0]), s.mean_long_short_return)
    assert _close_or_both_nan(float(df["long_short_ir"].iloc[0]), s.long_short_ir)
    assert _close_or_both_nan(float(df["long_short_hit_rate"].iloc[0]), s.long_short_hit_rate)
    assert _close_or_both_nan(
        float(df["long_short_return_per_turnover"].iloc[0]),
        s.long_short_return_per_turnover,
    )
    assert _close_or_both_nan(
        float(df["subperiod_ic_positive_share"].iloc[0]),
        s.subperiod_ic_positive_share,
    )
    assert _close_or_both_nan(
        float(df["subperiod_long_short_positive_share"].iloc[0]),
        s.subperiod_long_short_positive_share,
    )
    assert _close_or_both_nan(
        float(df["subperiod_ic_min_mean"].iloc[0]),
        s.subperiod_ic_min_mean,
    )
    assert _close_or_both_nan(
        float(df["subperiod_long_short_min_mean"].iloc[0]),
        s.subperiod_long_short_min_mean,
    )
    assert _close_or_both_nan(
        float(df["rolling_ic_positive_share"].iloc[0]),
        s.rolling_ic_positive_share,
    )
    assert _close_or_both_nan(
        float(df["rolling_rank_ic_positive_share"].iloc[0]),
        s.rolling_rank_ic_positive_share,
    )
    assert _close_or_both_nan(
        float(df["rolling_long_short_positive_share"].iloc[0]),
        s.rolling_long_short_positive_share,
    )
    assert _close_or_both_nan(
        float(df["rolling_ic_min_mean"].iloc[0]),
        s.rolling_ic_min_mean,
    )
    assert _close_or_both_nan(
        float(df["rolling_rank_ic_min_mean"].iloc[0]),
        s.rolling_rank_ic_min_mean,
    )
    assert _close_or_both_nan(
        float(df["rolling_long_short_min_mean"].iloc[0]),
        s.rolling_long_short_min_mean,
    )
    assert _close_or_both_nan(
        float(df["mean_eval_assets_per_date"].iloc[0]),
        s.mean_eval_assets_per_date,
    )
    assert _close_or_both_nan(
        float(df["min_eval_assets_per_date"].iloc[0]),
        s.min_eval_assets_per_date,
    )
    assert _close_or_both_nan(
        float(df["eval_coverage_ratio_mean"].iloc[0]),
        s.eval_coverage_ratio_mean,
    )
    assert _close_or_both_nan(
        float(df["eval_coverage_ratio_min"].iloc[0]),
        s.eval_coverage_ratio_min,
    )
    assert str(df["rolling_instability_flags"].iloc[0]) == ";".join(s.rolling_instability_flags)
    assert str(df["instability_flags"].iloc[0]) == ";".join(s.instability_flags)
    assert int(df["n_dates_used"].iloc[0]) == s.n_dates
