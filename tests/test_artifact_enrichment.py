from __future__ import annotations

import math

import pandas as pd
import pytest

from alpha_lab.real_cases.artifact_enrichment import (
    build_backtest_summary_payload,
    build_group_nav_table,
)


def test_nav_and_stats_share_effective_step_for_non_overlap() -> None:
    """NAV and stats must sample at max(rebalance_step, label_horizon).

    Daily compounding of overlapping forward-return labels would over-count;
    both series share the same non-overlapping stride.
    """
    dates = pd.date_range("2026-01-01", periods=10, freq="B")
    group_returns = pd.DataFrame(
        [
            {
                "date": date.strftime("%Y-%m-%d"),
                "group": group,
                "group_return": 0.10 if group == 5 else 0.0,
            }
            for date in dates
            for group in (1, 5)
        ]
    )

    daily_summary, _ = build_backtest_summary_payload(
        group_returns_df=group_returns,
        rebalance_frequency="D",
        metrics_for_payload={},
    )
    weekly_summary, _ = build_backtest_summary_payload(
        group_returns_df=group_returns,
        rebalance_frequency="W",
        metrics_for_payload={},
    )

    # Daily rebalance + 1-day label → effective_step=1, no striding.
    assert len(daily_summary["nav_points"]) == 10
    assert daily_summary["nav_source"] == "quantile_long_short_from_group_returns"
    assert daily_summary["nav_rebalance_step"] == 1
    assert daily_summary["label_horizon"] == 1
    assert daily_summary["nav_point_interval"] == "1D_available"
    assert daily_summary["statistics_rebalance_step"] == 1

    # Weekly rebalance → effective_step=5: 10 daily rows collapse to 2 samples
    # (every 5th business day), and nav uses the same stride as stats.
    assert len(weekly_summary["nav_points"]) == 2
    assert weekly_summary["nav_points"][-1][1] == pytest.approx(1.1 * 1.1)
    assert weekly_summary["nav_rebalance_step"] == 5
    assert weekly_summary["label_horizon"] == 1
    assert weekly_summary["nav_point_interval"] == "5D_non_overlapping"
    assert weekly_summary["statistics_rebalance_step"] == 5
    # Both samples land in January, so the whole compounded month return is the
    # product of the two sampled returns.
    assert weekly_summary["monthly_return_table"][-1][0] == "2026-01"
    assert weekly_summary["monthly_return_table"][-1][1] == pytest.approx(1.1 * 1.1 - 1.0)


def test_label_horizon_strides_nav_even_for_daily_rebalance() -> None:
    """Daily rebalance + 5-day label horizon must still stride by 5."""
    dates = pd.date_range("2026-01-01", periods=10, freq="B")
    group_returns = pd.DataFrame(
        [
            {
                "date": date.strftime("%Y-%m-%d"),
                "group": group,
                "group_return": 0.05 if group == 5 else 0.0,
            }
            for date in dates
            for group in (1, 5)
        ]
    )

    summary, _ = build_backtest_summary_payload(
        group_returns_df=group_returns,
        rebalance_frequency="D",
        metrics_for_payload={},
        label_horizon=5,
    )

    assert summary["nav_rebalance_step"] == 5
    assert summary["label_horizon"] == 5
    assert summary["statistics_rebalance_step"] == 5
    assert len(summary["nav_points"]) == 2
    assert summary["nav_points"][-1][1] == pytest.approx(1.05 * 1.05)


def test_backtest_summary_uses_per_date_occupied_top_bucket() -> None:
    dates = pd.date_range("2026-01-01", periods=2, freq="B")
    group_returns = pd.DataFrame(
        [
            {"date": dates[0].strftime("%Y-%m-%d"), "group": 1, "group_return": 0.01},
            {"date": dates[0].strftime("%Y-%m-%d"), "group": 2, "group_return": 0.03},
            {"date": dates[1].strftime("%Y-%m-%d"), "group": 1, "group_return": 0.02},
            {"date": dates[1].strftime("%Y-%m-%d"), "group": 5, "group_return": 0.07},
        ]
    )

    summary, _ = build_backtest_summary_payload(
        group_returns_df=group_returns,
        rebalance_frequency="D",
        metrics_for_payload={},
    )

    assert len(summary["nav_points"]) == 2
    assert summary["nav_points"][-1][1] == pytest.approx(1.02 * 1.05)
    assert summary["pre_cost_return"] is None
    assert summary["nav_series_policy"] == "daily_available_forward_return_path_for_chart"


def test_backtest_summary_sharpe_uses_annualized_return_stats_not_daily_ir() -> None:
    dates = pd.date_range("2026-01-01", periods=4, freq="B")
    long_short_returns = [0.01, 0.02, -0.005, 0.015]
    group_returns = pd.DataFrame(
        [
            {
                "date": date.strftime("%Y-%m-%d"),
                "group": group,
                "group_return": ret if group == 5 else 0.0,
            }
            for date, ret in zip(dates, long_short_returns, strict=True)
            for group in (1, 5)
        ]
    )

    summary, _ = build_backtest_summary_payload(
        group_returns_df=group_returns,
        rebalance_frequency="D",
        metrics_for_payload={
            "long_short_ir_full": 999.0,
            "cost_aware_long_short_ir_full": 888.0,
            "long_short_ir": 777.0,
        },
    )

    clean = pd.Series(long_short_returns, index=dates, dtype=float)
    nav = (1.0 + clean).cumprod()
    annualized_return = (1.0 + float(nav.iloc[-1] - 1.0)) ** (252 / len(clean)) - 1.0
    annualized_volatility = float(clean.std(ddof=1) * math.sqrt(252))
    expected_sharpe = annualized_return / annualized_volatility

    assert summary["sharpe"] == pytest.approx(expected_sharpe)
    assert summary["sharpe"] != 999.0


def test_backtest_monthly_returns_use_compounded_product() -> None:
    dates = pd.to_datetime(["2026-01-29", "2026-01-30", "2026-02-02"])
    group_returns = pd.DataFrame(
        [
            {
                "date": date.strftime("%Y-%m-%d"),
                "group": group,
                "group_return": 0.01 if group == 5 else 0.0,
            }
            for date in dates
            for group in (1, 5)
        ]
    )

    summary, _ = build_backtest_summary_payload(
        group_returns_df=group_returns,
        rebalance_frequency="D",
        metrics_for_payload={},
    )

    rows = summary["monthly_return_table"]
    assert rows[0][0] == "2026-01"
    assert rows[0][1] == pytest.approx((1.01 * 1.01) - 1.0)
    assert rows[1][0] == "2026-02"
    assert rows[1][1] == pytest.approx(0.01)


def test_group_nav_strides_daily_panel_with_5d_label_horizon() -> None:
    """Daily-grid + 5-day forward label must collapse to a non-overlap NAV path.

    Regression for the overlap compounding bug: a constant 1% Q5 forward
    return on 252 daily rows would yield ``1.01 ** 252 ≈ 12.7`` if naively
    cumprod-ed.  ``build_group_nav_table`` strides by ``label_horizon=5``,
    leaves ~51 sampled rows, and gives ``1.01 ** 51 ≈ 1.66`` instead.
    """
    dates = pd.date_range("2026-01-01", periods=252, freq="B")
    group_returns = pd.DataFrame(
        [
            {
                "date": date.strftime("%Y-%m-%d"),
                "group": group,
                "group_return": 0.01 if group == 5 else -0.005,
            }
            for date in dates
            for group in (1, 5)
        ]
    )

    nav_table = build_group_nav_table(
        group_returns,
        rebalance_frequency="D",
        label_horizon=5,
    )

    assert list(nav_table.columns) == [
        "date",
        "group",
        "period_return",
        "nav",
        "sample_step",
        "rebalance_step",
        "label_horizon",
    ]
    assert (nav_table["sample_step"] == 5).all()
    assert (nav_table["rebalance_step"] == 1).all()
    assert (nav_table["label_horizon"] == 5).all()

    q5 = nav_table[nav_table["group"] == 5].reset_index(drop=True)
    expected_n = len(dates[::5])  # ~51
    assert len(q5) == expected_n
    final_nav = float(q5["nav"].iloc[-1])
    assert final_nav == pytest.approx(1.01 ** expected_n)
    # Buggy naive cumprod over all 252 rows would land near 12.7 — far above
    # the strided result; this asserts the strider really kicked in.
    assert final_nav < 2.0


def test_group_nav_handles_weekly_rebalance_with_1d_label() -> None:
    dates = pd.date_range("2026-01-01", periods=15, freq="B")
    group_returns = pd.DataFrame(
        [
            {
                "date": date.strftime("%Y-%m-%d"),
                "group": 5,
                "group_return": 0.02,
            }
            for date in dates
        ]
    )

    nav_table = build_group_nav_table(
        group_returns,
        rebalance_frequency="W",
        label_horizon=1,
    )

    # Weekly rebalance (step=5) wins over 1-day label, so 15 rows → 3 samples.
    assert len(nav_table) == 3
    assert (nav_table["sample_step"] == 5).all()
    assert (nav_table["rebalance_step"] == 5).all()
    assert (nav_table["label_horizon"] == 1).all()
    assert float(nav_table["nav"].iloc[-1]) == pytest.approx(1.02 ** 3)


def test_group_nav_returns_empty_frame_with_canonical_schema_when_no_input() -> None:
    empty = build_group_nav_table(
        pd.DataFrame(columns=["date", "group", "group_return"]),
        rebalance_frequency="D",
        label_horizon=1,
    )
    assert empty.empty
    assert list(empty.columns) == [
        "date",
        "group",
        "period_return",
        "nav",
        "sample_step",
        "rebalance_step",
        "label_horizon",
    ]
