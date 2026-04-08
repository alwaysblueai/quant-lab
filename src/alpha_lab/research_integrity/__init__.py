"""Core Level 1/2 research-integrity surface.

This package intentionally exposes point-in-time, temporal, and leakage
validation primitives that are required for default research workflows.

Replay/implementability semantic-audit helpers are intentionally excluded from
this core namespace and live under `alpha_lab.experimental_level3`.
"""

from alpha_lab.research_integrity.asof import (
    AsofJoinSummary,
    AsofMonotonicitySummary,
    ForwardFillLagSummary,
    TimeSemanticsMetadata,
    asof_join_frame,
    attach_time_semantics,
    read_time_semantics,
    validate_asof_monotonicity,
    validate_forward_fill_lag,
)
from alpha_lab.research_integrity.contracts import (
    INTEGRITY_REPORT_SCHEMA_VERSION,
    IntegrityCheckResult,
    IntegrityReport,
    IntegrityReportSummary,
)
from alpha_lab.research_integrity.exceptions import IntegrityHardFailure, raise_on_hard_failures
from alpha_lab.research_integrity.leakage_checks import (
    check_asof_inputs_not_after_signal_date,
    check_bar_close_known_before_order_submission,
    check_closed_bar_required_before_signal_use,
    check_cross_section_transform_scope,
    check_daily_feature_asof_intraday,
    check_factor_label_temporal_order,
    check_higher_timeframe_feature_not_available_early,
    check_incomplete_bar_not_used,
    check_intraday_to_daily_alignment,
    check_multitimeframe_alignment,
    check_no_future_dates_in_input,
    check_same_bar_close_execution_conflict,
    check_signal_execution_gap_is_respected,
)
from alpha_lab.research_integrity.reporting import (
    build_integrity_report,
    export_integrity_report,
    render_integrity_report_markdown,
    write_integrity_report_json,
    write_integrity_report_markdown,
)

__all__ = [
    "AsofJoinSummary",
    "AsofMonotonicitySummary",
    "ForwardFillLagSummary",
    "INTEGRITY_REPORT_SCHEMA_VERSION",
    "IntegrityCheckResult",
    "IntegrityHardFailure",
    "IntegrityReport",
    "IntegrityReportSummary",
    "TimeSemanticsMetadata",
    "asof_join_frame",
    "attach_time_semantics",
    "build_integrity_report",
    "check_asof_inputs_not_after_signal_date",
    "check_bar_close_known_before_order_submission",
    "check_closed_bar_required_before_signal_use",
    "check_cross_section_transform_scope",
    "check_daily_feature_asof_intraday",
    "check_factor_label_temporal_order",
    "check_higher_timeframe_feature_not_available_early",
    "check_incomplete_bar_not_used",
    "check_intraday_to_daily_alignment",
    "check_multitimeframe_alignment",
    "check_no_future_dates_in_input",
    "check_same_bar_close_execution_conflict",
    "check_signal_execution_gap_is_respected",
    "export_integrity_report",
    "raise_on_hard_failures",
    "read_time_semantics",
    "render_integrity_report_markdown",
    "validate_asof_monotonicity",
    "validate_forward_fill_lag",
    "write_integrity_report_json",
    "write_integrity_report_markdown",
]
