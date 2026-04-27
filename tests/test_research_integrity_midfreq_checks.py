from __future__ import annotations

import pandas as pd

from alpha_lab.research_integrity.leakage_checks import (
    check_closed_bar_required_before_signal_use,
    check_daily_feature_asof_intraday,
    check_higher_timeframe_feature_not_available_early,
    check_intraday_to_daily_alignment,
    check_multitimeframe_alignment,
    check_same_bar_close_execution_conflict,
)
from alpha_lab.research_integrity.reporting import build_integrity_report


def test_bar_timing_completed_bar_passes() -> None:
    df = pd.DataFrame(
        {
            "signal_computed_at": pd.to_datetime(["2024-01-03 09:35:00"]),
            "bar_close_known_at": pd.to_datetime(["2024-01-03 09:35:00"]),
        }
    )
    result = check_closed_bar_required_before_signal_use(df)
    assert result.status == "pass"


def test_bar_timing_incomplete_bar_fails() -> None:
    df = pd.DataFrame(
        {
            "signal_computed_at": pd.to_datetime(["2024-01-03 09:33:00"]),
            "bar_close_known_at": pd.to_datetime(["2024-01-03 09:35:00"]),
        }
    )
    result = check_closed_bar_required_before_signal_use(df)
    assert result.status == "fail"


def test_same_bar_close_signal_execution_conflict_fails() -> None:
    df = pd.DataFrame(
        {
            "signal_bar_timestamp": pd.to_datetime(["2024-01-03 09:35:00"]),
            "execution_bar_timestamp": pd.to_datetime(["2024-01-03 09:35:00"]),
            "execution_price_rule": ["next_close"],
        }
    )
    result = check_same_bar_close_execution_conflict(df)
    assert result.status == "fail"


def test_multitimeframe_last_completed_bar_passes() -> None:
    df = pd.DataFrame(
        {
            "signal_bar_timestamp": pd.to_datetime(["2024-01-03 09:35:00"]),
            "higher_timeframe_bar_close": pd.to_datetime(["2024-01-03 09:35:00"]),
            "signal_computed_at": pd.to_datetime(["2024-01-03 09:35:00"]),
            "higher_timeframe_known_at": pd.to_datetime(["2024-01-03 09:35:00"]),
        }
    )
    assert check_multitimeframe_alignment(df).status == "pass"
    assert check_higher_timeframe_feature_not_available_early(df).status == "pass"


def test_multitimeframe_unfinished_higher_bar_fails() -> None:
    df = pd.DataFrame(
        {
            "signal_bar_timestamp": pd.to_datetime(["2024-01-03 09:33:00"]),
            "higher_timeframe_bar_close": pd.to_datetime(["2024-01-03 09:35:00"]),
            "signal_computed_at": pd.to_datetime(["2024-01-03 09:33:00"]),
            "higher_timeframe_known_at": pd.to_datetime(["2024-01-03 09:35:00"]),
        }
    )
    assert check_multitimeframe_alignment(df).status == "fail"
    assert check_higher_timeframe_feature_not_available_early(df).status == "fail"


def test_intraday_daily_same_day_final_aggregate_too_early_fails() -> None:
    df = pd.DataFrame(
        {
            "signal_computed_at": pd.to_datetime(["2024-01-03 10:00:00"]),
            "daily_feature_known_at": pd.to_datetime(["2024-01-03 15:00:00"]),
        }
    )
    result = check_intraday_to_daily_alignment(df)
    assert result.status == "fail"


def test_daily_feature_asof_intraday_with_effective_time_passes() -> None:
    signal_df = pd.DataFrame(
        {
            "asset": ["AAA"],
            "signal_computed_at": pd.to_datetime(["2024-01-03 10:00:00"]),
        }
    )
    daily_df = pd.DataFrame(
        {
            "asset": ["AAA"],
            "effective_date": pd.to_datetime(["2024-01-02"]),
            "known_at": pd.to_datetime(["2024-01-03 09:30:00"]),
            "value": [1.0],
        }
    )
    result = check_daily_feature_asof_intraday(signal_df, daily_df)
    assert result.status == "pass"


def test_integrity_report_includes_new_bar_timing_findings() -> None:
    checks = (
        check_closed_bar_required_before_signal_use(
            pd.DataFrame(
                {
                    "signal_computed_at": pd.to_datetime(["2024-01-03 09:33:00"]),
                    "bar_close_known_at": pd.to_datetime(["2024-01-03 09:35:00"]),
                }
            )
        ),
        check_same_bar_close_execution_conflict(
            pd.DataFrame(
                {
                    "signal_bar_timestamp": pd.to_datetime(["2024-01-03 09:35:00"]),
                    "execution_bar_timestamp": pd.to_datetime(["2024-01-03 09:35:00"]),
                    "execution_price_rule": ["next_close"],
                }
            )
        ),
    )
    report = build_integrity_report(checks)
    names = [row["check_name"] for row in report.to_dict()["checks"]]
    assert "check_closed_bar_required_before_signal_use" in names
    assert "check_same_bar_close_execution_conflict" in names
