from __future__ import annotations

import pandas as pd
import pytest

from alpha_lab.real_cases.artifact_enrichment import build_backtest_summary_payload


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
