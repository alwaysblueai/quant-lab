"""Temporal/leakage checks for Level 1/2 research integrity.

Most checks here are core research checks. A small subset at the bottom of
this module supports optional Level 3 replay semantics auditing and is not part
of the default workflow.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd

from alpha_lab.exceptions import AlphaLabConfigError
from alpha_lab.research_integrity.asof import asof_join_frame
from alpha_lab.research_integrity.contracts import IntegrityCheckResult


def check_no_future_dates_in_input(
    frame: pd.DataFrame,
    *,
    max_allowed_date: pd.Timestamp | str,
    date_col: str = "date",
    object_name: str = "input_frame",
) -> IntegrityCheckResult:
    """Fail when a table contains dates beyond the allowed research cutoff."""

    if date_col not in frame.columns:
        return IntegrityCheckResult(
            check_name="check_no_future_dates_in_input",
            status="fail",
            severity="error",
            object_name=object_name,
            module_name="research_integrity.leakage_checks",
            message=f"missing required date column {date_col!r}",
            remediation="Provide the date column used for time-bound validation.",
        )

    dates = pd.to_datetime(frame[date_col], errors="coerce")
    if dates.isna().any():
        return IntegrityCheckResult(
            check_name="check_no_future_dates_in_input",
            status="fail",
            severity="error",
            object_name=object_name,
            module_name="research_integrity.leakage_checks",
            message=f"column {date_col!r} contains invalid timestamp values",
            remediation="Clean timestamp parsing issues before running leakage checks.",
        )

    cutoff = pd.Timestamp(max_allowed_date)
    future_mask = dates > cutoff
    n_future = int(future_mask.sum())

    if n_future > 0:
        return IntegrityCheckResult(
            check_name="check_no_future_dates_in_input",
            status="fail",
            severity="error",
            object_name=object_name,
            module_name="research_integrity.leakage_checks",
            message=(
                f"{n_future} rows are future-dated relative to max_allowed_date={cutoff.date()}"
            ),
            remediation="Drop or delay rows whose timestamps exceed the research cutoff.",
            metrics={
                "future_rows": n_future,
                "max_future_date": str(dates[future_mask].max()),
                "max_allowed_date": str(cutoff),
            },
        )

    return IntegrityCheckResult(
        check_name="check_no_future_dates_in_input",
        status="pass",
        severity="info",
        object_name=object_name,
        module_name="research_integrity.leakage_checks",
        message="no future-dated rows found",
        metrics={"rows_checked": int(len(frame))},
    )


def check_factor_label_temporal_order(
    factor_df: pd.DataFrame,
    label_df: pd.DataFrame,
    *,
    join_keys: tuple[str, ...] = ("date", "asset"),
    factor_date_col: str = "date",
    label_date_col: str = "date",
    object_name: str = "factor_label_alignment",
) -> IntegrityCheckResult:
    """Fail when factor timestamps appear after label timestamps for joined rows."""

    factor_missing = sorted(set([*join_keys, factor_date_col]) - set(factor_df.columns))
    label_missing = sorted(set([*join_keys, label_date_col]) - set(label_df.columns))
    if factor_missing or label_missing:
        missing_parts: list[str] = []
        if factor_missing:
            missing_parts.append(f"factor missing {factor_missing}")
        if label_missing:
            missing_parts.append(f"label missing {label_missing}")
        return IntegrityCheckResult(
            check_name="check_factor_label_temporal_order",
            status="fail",
            severity="error",
            object_name=object_name,
            module_name="research_integrity.leakage_checks",
            message="; ".join(missing_parts),
            remediation="Provide explicit join keys and timestamp columns for factor/label checks.",
        )

    factor_cols = list(dict.fromkeys([*join_keys, factor_date_col]))
    label_cols = list(dict.fromkeys([*join_keys, label_date_col]))

    factor_view = factor_df.loc[:, factor_cols].copy()
    label_view = label_df.loc[:, label_cols].copy()

    factor_view[factor_date_col] = pd.to_datetime(factor_view[factor_date_col], errors="coerce")
    label_view[label_date_col] = pd.to_datetime(label_view[label_date_col], errors="coerce")

    if factor_view[factor_date_col].isna().any() or label_view[label_date_col].isna().any():
        return IntegrityCheckResult(
            check_name="check_factor_label_temporal_order",
            status="fail",
            severity="error",
            object_name=object_name,
            module_name="research_integrity.leakage_checks",
            message="factor or label date columns contain invalid timestamp values",
            remediation="Fix invalid timestamps before temporal-order checks.",
        )

    factor_agg = (
        factor_view.groupby(list(join_keys), dropna=False)[factor_date_col]
        .max()
        .rename("factor_date")
        .reset_index()
    )
    label_agg = (
        label_view.groupby(list(join_keys), dropna=False)[label_date_col]
        .min()
        .rename("label_date")
        .reset_index()
    )

    merged = factor_agg.merge(label_agg, on=list(join_keys), how="inner")
    if merged.empty:
        return IntegrityCheckResult(
            check_name="check_factor_label_temporal_order",
            status="warn",
            severity="warning",
            object_name=object_name,
            module_name="research_integrity.leakage_checks",
            message="factor and label tables have no overlapping join keys",
            remediation="Verify join_keys and timestamp semantics for factor/label alignment.",
        )

    future_mask = merged["factor_date"] > merged["label_date"]
    n_violations = int(future_mask.sum())
    lagged_label_mask = merged["label_date"] > merged["factor_date"]

    if n_violations > 0:
        return IntegrityCheckResult(
            check_name="check_factor_label_temporal_order",
            status="fail",
            severity="error",
            object_name=object_name,
            module_name="research_integrity.leakage_checks",
            message=f"{n_violations} rows have factor_date later than label_date",
            remediation=(
                "Ensure labels are attached to the same or later signal timestamp "
                "than factors."
            ),
            metrics={
                "violations": n_violations,
                "rows_compared": int(len(merged)),
            },
        )

    status: Literal["pass", "warn", "fail"] = "pass"
    severity: Literal["info", "warning", "error"] = "info"
    message = "factor/label temporal ordering check passed"
    remediation: str | None = None

    n_lagged_label = int(lagged_label_mask.sum())
    if n_lagged_label > 0:
        status = "warn"
        severity = "warning"
        message = (
            f"{n_lagged_label} rows have label_date after factor_date; confirm this is intended "
            "signal-date semantics"
        )
        remediation = "Confirm whether label timestamps represent signal date or realization date."

    return IntegrityCheckResult(
        check_name="check_factor_label_temporal_order",
        status=status,
        severity=severity,
        object_name=object_name,
        module_name="research_integrity.leakage_checks",
        message=message,
        remediation=remediation,
        metrics={
            "rows_compared": int(len(merged)),
            "label_after_factor_rows": n_lagged_label,
        },
    )


def check_asof_inputs_not_after_signal_date(
    signal_df: pd.DataFrame,
    aux_df: pd.DataFrame,
    *,
    by: tuple[str, ...] = ("asset",),
    signal_date_col: str = "date",
    aux_effective_date_col: str = "effective_date",
    aux_known_at_col: str | None = None,
    object_name: str = "asof_input_pair",
) -> IntegrityCheckResult:
    """Fail when as-of auxiliary rows are known after the signal timestamp."""

    signal_missing = sorted(set([*by, signal_date_col]) - set(signal_df.columns))
    aux_missing = sorted(set([*by, aux_effective_date_col]) - set(aux_df.columns))
    if signal_missing or aux_missing:
        chunks: list[str] = []
        if signal_missing:
            chunks.append(f"signal missing {signal_missing}")
        if aux_missing:
            chunks.append(f"aux missing {aux_missing}")
        return IntegrityCheckResult(
            check_name="check_asof_inputs_not_after_signal_date",
            status="fail",
            severity="error",
            object_name=object_name,
            module_name="research_integrity.leakage_checks",
            message="; ".join(chunks),
            remediation="Provide required keys and timestamp columns for as-of checks.",
        )

    known_col = aux_known_at_col
    if known_col is None:
        if "known_at" in aux_df.columns:
            known_col = "known_at"
        elif "available_at" in aux_df.columns:
            known_col = "available_at"

    if known_col is None:
        return IntegrityCheckResult(
            check_name="check_asof_inputs_not_after_signal_date",
            status="warn",
            severity="warning",
            object_name=object_name,
            module_name="research_integrity.leakage_checks",
            message="aux table lacks known_at/available_at; cannot verify publication-time leakage",
            remediation="Add known_at or available_at to auxiliary data for strict PIT checks.",
        )

    _, summary = asof_join_frame(
        signal_df,
        aux_df,
        by=by,
        left_on=signal_date_col,
        right_effective_col=aux_effective_date_col,
        right_known_at_col=known_col,
        strict_known_at=False,
    )

    if summary.future_blocked_count > 0:
        return IntegrityCheckResult(
            check_name="check_asof_inputs_not_after_signal_date",
            status="fail",
            severity="error",
            object_name=object_name,
            module_name="research_integrity.leakage_checks",
            message=(
                f"{summary.future_blocked_count} rows align to aux records with known_at "
                "after signal date"
            ),
            remediation=(
                "Filter auxiliary rows by known_at <= signal date before feature "
                "construction."
            ),
            metrics=summary.to_dict(),
        )

    if summary.matched_count == 0:
        return IntegrityCheckResult(
            check_name="check_asof_inputs_not_after_signal_date",
            status="warn",
            severity="warning",
            object_name=object_name,
            module_name="research_integrity.leakage_checks",
            message="as-of join produced no matches; leakage check coverage is empty",
            remediation="Check keys/effective_date coverage for the auxiliary table.",
            metrics=summary.to_dict(),
        )

    return IntegrityCheckResult(
        check_name="check_asof_inputs_not_after_signal_date",
        status="pass",
        severity="info",
        object_name=object_name,
        module_name="research_integrity.leakage_checks",
        message="no known_at > signal_date violations detected in as-of alignment",
        metrics=summary.to_dict(),
    )


def check_cross_section_transform_scope(
    raw_df: pd.DataFrame,
    transformed_df: pd.DataFrame,
    *,
    date_col: str = "date",
    asset_col: str = "asset",
    object_name: str = "cross_section_transform",
) -> IntegrityCheckResult:
    """Fail when a cross-sectional transform emits rows outside the raw date/asset scope."""

    required = [date_col, asset_col]
    raw_missing = sorted(set(required) - set(raw_df.columns))
    transformed_missing = sorted(set(required) - set(transformed_df.columns))
    if raw_missing or transformed_missing:
        details: list[str] = []
        if raw_missing:
            details.append(f"raw missing {raw_missing}")
        if transformed_missing:
            details.append(f"transformed missing {transformed_missing}")
        return IntegrityCheckResult(
            check_name="check_cross_section_transform_scope",
            status="fail",
            severity="error",
            object_name=object_name,
            module_name="research_integrity.leakage_checks",
            message="; ".join(details),
            remediation="Ensure both raw and transformed tables contain date/asset columns.",
        )

    raw_dates = pd.to_datetime(raw_df[date_col], errors="coerce")
    transformed_dates = pd.to_datetime(transformed_df[date_col], errors="coerce")
    if raw_dates.isna().any() or transformed_dates.isna().any():
        return IntegrityCheckResult(
            check_name="check_cross_section_transform_scope",
            status="fail",
            severity="error",
            object_name=object_name,
            module_name="research_integrity.leakage_checks",
            message="raw or transformed date columns contain invalid timestamps",
            remediation="Fix date parsing before scope checks.",
        )

    transformed_keys = pd.DataFrame(
        {
            date_col: transformed_dates,
            asset_col: transformed_df[asset_col].astype(str),
        }
    )
    if transformed_keys.duplicated(subset=[date_col, asset_col]).any():
        return IntegrityCheckResult(
            check_name="check_cross_section_transform_scope",
            status="fail",
            severity="error",
            object_name=object_name,
            module_name="research_integrity.leakage_checks",
            message="transformed output contains duplicate (date, asset) rows",
            remediation="Deduplicate transform output per date/asset before downstream merges.",
        )

    raw_keys = set(
        zip(
            raw_dates.to_numpy(),
            raw_df[asset_col].astype(str).to_numpy(),
            strict=False,
        )
    )
    out_keys = set(
        zip(
            transformed_dates.to_numpy(),
            transformed_df[asset_col].astype(str).to_numpy(),
            strict=False,
        )
    )

    outside_scope = out_keys - raw_keys
    if outside_scope:
        return IntegrityCheckResult(
            check_name="check_cross_section_transform_scope",
            status="fail",
            severity="error",
            object_name=object_name,
            module_name="research_integrity.leakage_checks",
            message=(
                f"transform output contains {len(outside_scope)} (date, asset) pairs absent "
                "from raw input"
            ),
            remediation=(
                "Restrict transformations to same-date cross-sections and avoid emitting unseen "
                "date/asset pairs."
            ),
            metrics={
                "outside_scope_pairs": len(outside_scope),
                "raw_pairs": len(raw_keys),
                "output_pairs": len(out_keys),
            },
        )

    if len(out_keys) < len(raw_keys):
        return IntegrityCheckResult(
            check_name="check_cross_section_transform_scope",
            status="warn",
            severity="warning",
            object_name=object_name,
            module_name="research_integrity.leakage_checks",
            message="transform output drops part of raw cross-sectional universe",
            remediation="Confirm drops are intentional (e.g., coverage gate) and documented.",
            metrics={
                "raw_pairs": len(raw_keys),
                "output_pairs": len(out_keys),
            },
        )

    return IntegrityCheckResult(
        check_name="check_cross_section_transform_scope",
        status="pass",
        severity="info",
        object_name=object_name,
        module_name="research_integrity.leakage_checks",
        message="cross-sectional transform scope check passed",
        metrics={
            "raw_pairs": len(raw_keys),
            "output_pairs": len(out_keys),
        },
    )


def check_closed_bar_required_before_signal_use(
    frame: pd.DataFrame,
    *,
    signal_computed_at_col: str = "signal_computed_at",
    bar_close_known_at_col: str = "bar_close_known_at",
    uses_closed_bar_only: bool = True,
    allows_incomplete_bar_features: bool = False,
    object_name: str = "bar_timing",
) -> IntegrityCheckResult:
    """Fail when signal computation occurs before the bar close is known."""

    if allows_incomplete_bar_features:
        return IntegrityCheckResult(
            check_name="check_closed_bar_required_before_signal_use",
            status="fail",
            severity="error",
            object_name=object_name,
            module_name="research_integrity.leakage_checks",
            message="allows_incomplete_bar_features=True violates closed-bar discipline",
            remediation=(
                "Set allows_incomplete_bar_features=False and compute signals only after the "
                "bar close becomes known."
            ),
        )

    if not uses_closed_bar_only:
        return IntegrityCheckResult(
            check_name="check_closed_bar_required_before_signal_use",
            status="warn",
            severity="warning",
            object_name=object_name,
            module_name="research_integrity.leakage_checks",
            message=(
                "uses_closed_bar_only=False; closed-bar timing cannot be guaranteed for signal "
                "computation"
            ),
            remediation="Enable closed-bar-only semantics for mid-frequency integrity.",
        )

    missing = sorted({signal_computed_at_col, bar_close_known_at_col} - set(frame.columns))
    if missing:
        return _missing_column_result(
            "check_closed_bar_required_before_signal_use", object_name, missing
        )

    signal_time = _coerce_datetime(frame[signal_computed_at_col])
    close_known = _coerce_datetime(frame[bar_close_known_at_col])
    invalid = signal_time.isna() | close_known.isna()
    if bool(invalid.any()):
        return _invalid_timestamp_result(
            "check_closed_bar_required_before_signal_use",
            object_name,
            signal_computed_at_col,
            bar_close_known_at_col,
        )

    violations = signal_time < close_known
    n_violations = int(violations.sum())
    if n_violations > 0:
        return IntegrityCheckResult(
            check_name="check_closed_bar_required_before_signal_use",
            status="fail",
            severity="error",
            object_name=object_name,
            module_name="research_integrity.leakage_checks",
            message=(
                f"{n_violations} rows compute signals before bar close is known "
                f"({signal_computed_at_col} < {bar_close_known_at_col})"
            ),
            remediation="Delay signal computation until bar_close_known_at is reached.",
            metrics={"rows_compared": int(len(frame)), "violations": n_violations},
        )

    return IntegrityCheckResult(
        check_name="check_closed_bar_required_before_signal_use",
        status="pass",
        severity="info",
        object_name=object_name,
        module_name="research_integrity.leakage_checks",
        message="all signals are computed after bar close is known",
        metrics={"rows_compared": int(len(frame))},
    )


def check_incomplete_bar_not_used(
    frame: pd.DataFrame,
    *,
    signal_computed_at_col: str = "signal_computed_at",
    feature_available_at_col: str = "feature_available_at",
    object_name: str = "bar_completeness",
) -> IntegrityCheckResult:
    """Fail when a feature is consumed before it becomes available."""

    missing = sorted({signal_computed_at_col, feature_available_at_col} - set(frame.columns))
    if missing:
        return _missing_column_result("check_incomplete_bar_not_used", object_name, missing)

    signal_time = _coerce_datetime(frame[signal_computed_at_col])
    available_time = _coerce_datetime(frame[feature_available_at_col])
    invalid = signal_time.isna() | available_time.isna()
    if bool(invalid.any()):
        return _invalid_timestamp_result(
            "check_incomplete_bar_not_used",
            object_name,
            signal_computed_at_col,
            feature_available_at_col,
        )

    violations = signal_time < available_time
    n_violations = int(violations.sum())
    if n_violations > 0:
        return IntegrityCheckResult(
            check_name="check_incomplete_bar_not_used",
            status="fail",
            severity="error",
            object_name=object_name,
            module_name="research_integrity.leakage_checks",
            message=(
                f"{n_violations} rows consume features before availability "
                f"({signal_computed_at_col} < {feature_available_at_col})"
            ),
            remediation=(
                "Use the last fully known bar/aggregate only and align feature availability "
                "timestamps explicitly."
            ),
            metrics={"rows_compared": int(len(frame)), "violations": n_violations},
        )

    return IntegrityCheckResult(
        check_name="check_incomplete_bar_not_used",
        status="pass",
        severity="info",
        object_name=object_name,
        module_name="research_integrity.leakage_checks",
        message="no incomplete-bar feature usage detected",
        metrics={"rows_compared": int(len(frame))},
    )


def check_bar_close_known_before_order_submission(
    frame: pd.DataFrame,
    *,
    bar_close_known_at_col: str = "bar_close_known_at",
    order_submitted_at_col: str = "order_submitted_at",
    object_name: str = "order_timing",
) -> IntegrityCheckResult:
    """Fail when orders are submitted before the source bar close is known."""

    missing = sorted({bar_close_known_at_col, order_submitted_at_col} - set(frame.columns))
    if missing:
        return _missing_column_result(
            "check_bar_close_known_before_order_submission",
            object_name,
            missing,
        )

    bar_known = _coerce_datetime(frame[bar_close_known_at_col])
    submitted = _coerce_datetime(frame[order_submitted_at_col])
    invalid = bar_known.isna() | submitted.isna()
    if bool(invalid.any()):
        return _invalid_timestamp_result(
            "check_bar_close_known_before_order_submission",
            object_name,
            bar_close_known_at_col,
            order_submitted_at_col,
        )

    violations = submitted < bar_known
    n_violations = int(violations.sum())
    if n_violations > 0:
        return IntegrityCheckResult(
            check_name="check_bar_close_known_before_order_submission",
            status="fail",
            severity="error",
            object_name=object_name,
            module_name="research_integrity.leakage_checks",
            message=(
                f"{n_violations} rows submit orders before bar close known-at "
                f"({order_submitted_at_col} < {bar_close_known_at_col})"
            ),
            remediation="Submit orders only after bar-close-derived signals are legally known.",
            metrics={"rows_compared": int(len(frame)), "violations": n_violations},
        )

    return IntegrityCheckResult(
        check_name="check_bar_close_known_before_order_submission",
        status="pass",
        severity="info",
        object_name=object_name,
        module_name="research_integrity.leakage_checks",
        message="order submission timestamps respect bar-close known-at timing",
        metrics={"rows_compared": int(len(frame))},
    )


def check_higher_timeframe_feature_not_available_early(
    frame: pd.DataFrame,
    *,
    signal_time_col: str = "signal_computed_at",
    higher_timeframe_known_at_col: str = "higher_timeframe_known_at",
    object_name: str = "higher_timeframe_availability",
) -> IntegrityCheckResult:
    """Fail when higher-timeframe features are used before their known-at timestamp."""

    missing = sorted({signal_time_col, higher_timeframe_known_at_col} - set(frame.columns))
    if missing:
        return _missing_column_result(
            "check_higher_timeframe_feature_not_available_early",
            object_name,
            missing,
        )

    signal_time = _coerce_datetime(frame[signal_time_col])
    known_at = _coerce_datetime(frame[higher_timeframe_known_at_col])
    invalid = signal_time.isna() | known_at.isna()
    if bool(invalid.any()):
        return _invalid_timestamp_result(
            "check_higher_timeframe_feature_not_available_early",
            object_name,
            signal_time_col,
            higher_timeframe_known_at_col,
        )

    violations = signal_time < known_at
    n_violations = int(violations.sum())
    if n_violations > 0:
        return IntegrityCheckResult(
            check_name="check_higher_timeframe_feature_not_available_early",
            status="fail",
            severity="error",
            object_name=object_name,
            module_name="research_integrity.leakage_checks",
            message=(
                f"{n_violations} rows use higher-timeframe features before availability "
                f"({signal_time_col} < {higher_timeframe_known_at_col})"
            ),
            remediation="Shift higher-timeframe features to the first bar when they are known.",
            metrics={"rows_compared": int(len(frame)), "violations": n_violations},
        )

    return IntegrityCheckResult(
        check_name="check_higher_timeframe_feature_not_available_early",
        status="pass",
        severity="info",
        object_name=object_name,
        module_name="research_integrity.leakage_checks",
        message="higher-timeframe features are not accessed early",
        metrics={"rows_compared": int(len(frame))},
    )


def check_multitimeframe_alignment(
    frame: pd.DataFrame,
    *,
    signal_bar_timestamp_col: str = "signal_bar_timestamp",
    higher_timeframe_bar_close_col: str = "higher_timeframe_bar_close",
    uses_closed_higher_timeframe_only: bool = True,
    object_name: str = "multitimeframe_alignment",
) -> IntegrityCheckResult:
    """Fail when lower-timeframe signals reference unfinished higher-timeframe bars."""

    if not uses_closed_higher_timeframe_only:
        return IntegrityCheckResult(
            check_name="check_multitimeframe_alignment",
            status="warn",
            severity="warning",
            object_name=object_name,
            module_name="research_integrity.leakage_checks",
            message="uses_closed_higher_timeframe_only=False; early access risk is not bounded",
            remediation=(
                "Use completed higher-timeframe bars only when driving intraday signals."
            ),
        )

    missing = sorted(
        {signal_bar_timestamp_col, higher_timeframe_bar_close_col} - set(frame.columns)
    )
    if missing:
        return _missing_column_result("check_multitimeframe_alignment", object_name, missing)

    signal_bar = _coerce_datetime(frame[signal_bar_timestamp_col])
    higher_close = _coerce_datetime(frame[higher_timeframe_bar_close_col])
    invalid = signal_bar.isna() | higher_close.isna()
    if bool(invalid.any()):
        return _invalid_timestamp_result(
            "check_multitimeframe_alignment",
            object_name,
            signal_bar_timestamp_col,
            higher_timeframe_bar_close_col,
        )

    violations = signal_bar < higher_close
    n_violations = int(violations.sum())
    if n_violations > 0:
        return IntegrityCheckResult(
            check_name="check_multitimeframe_alignment",
            status="fail",
            severity="error",
            object_name=object_name,
            module_name="research_integrity.leakage_checks",
            message=(
                f"{n_violations} rows align lower-timeframe signals to unfinished "
                "higher-timeframe bars"
            ),
            remediation=(
                "Only use the last completed higher-timeframe bar for lower-timeframe logic."
            ),
            metrics={"rows_compared": int(len(frame)), "violations": n_violations},
        )

    return IntegrityCheckResult(
        check_name="check_multitimeframe_alignment",
        status="pass",
        severity="info",
        object_name=object_name,
        module_name="research_integrity.leakage_checks",
        message="multi-timeframe alignment uses completed higher-timeframe bars",
        metrics={"rows_compared": int(len(frame))},
    )


def check_intraday_to_daily_alignment(
    frame: pd.DataFrame,
    *,
    intraday_signal_time_col: str = "signal_computed_at",
    daily_feature_known_at_col: str = "daily_feature_known_at",
    object_name: str = "intraday_daily_alignment",
) -> IntegrityCheckResult:
    """Fail when intraday logic consumes daily aggregates before they are known."""

    missing = sorted({intraday_signal_time_col, daily_feature_known_at_col} - set(frame.columns))
    if missing:
        return _missing_column_result("check_intraday_to_daily_alignment", object_name, missing)

    intraday_time = _coerce_datetime(frame[intraday_signal_time_col])
    daily_known = _coerce_datetime(frame[daily_feature_known_at_col])
    invalid = intraday_time.isna() | daily_known.isna()
    if bool(invalid.any()):
        return _invalid_timestamp_result(
            "check_intraday_to_daily_alignment",
            object_name,
            intraday_signal_time_col,
            daily_feature_known_at_col,
        )

    violations = intraday_time < daily_known
    n_violations = int(violations.sum())
    if n_violations > 0:
        return IntegrityCheckResult(
            check_name="check_intraday_to_daily_alignment",
            status="fail",
            severity="error",
            object_name=object_name,
            module_name="research_integrity.leakage_checks",
            message=(
                f"{n_violations} rows use daily features before daily effective known-at "
                "time"
            ),
            remediation=(
                "Align daily aggregates/fundamentals with explicit effective-time semantics in "
                "intraday workflows."
            ),
            metrics={"rows_compared": int(len(frame)), "violations": n_violations},
        )

    return IntegrityCheckResult(
        check_name="check_intraday_to_daily_alignment",
        status="pass",
        severity="info",
        object_name=object_name,
        module_name="research_integrity.leakage_checks",
        message="intraday-to-daily alignment respects daily known-at timing",
        metrics={"rows_compared": int(len(frame))},
    )


def check_daily_feature_asof_intraday(
    signal_df: pd.DataFrame,
    daily_feature_df: pd.DataFrame,
    *,
    by: tuple[str, ...] = ("asset",),
    signal_time_col: str = "signal_computed_at",
    daily_effective_date_col: str = "effective_date",
    daily_known_at_col: str = "known_at",
    object_name: str = "daily_feature_asof_intraday",
) -> IntegrityCheckResult:
    """Fail when as-of intraday joins can see daily rows before known-at time."""

    signal_missing = sorted(set([*by, signal_time_col]) - set(signal_df.columns))
    daily_missing = sorted(set([*by, daily_effective_date_col]) - set(daily_feature_df.columns))
    if signal_missing or daily_missing:
        chunks: list[str] = []
        if signal_missing:
            chunks.append(f"signal missing {signal_missing}")
        if daily_missing:
            chunks.append(f"daily missing {daily_missing}")
        return IntegrityCheckResult(
            check_name="check_daily_feature_asof_intraday",
            status="fail",
            severity="error",
            object_name=object_name,
            module_name="research_integrity.leakage_checks",
            message="; ".join(chunks),
            remediation="Provide signal keys, signal timestamp, and daily effective date columns.",
        )

    if daily_known_at_col not in daily_feature_df.columns:
        return IntegrityCheckResult(
            check_name="check_daily_feature_asof_intraday",
            status="warn",
            severity="warning",
            object_name=object_name,
            module_name="research_integrity.leakage_checks",
            message=(
                f"daily feature table lacks {daily_known_at_col!r}; cannot enforce intraday "
                "as-of known-at constraints"
            ),
            remediation="Add explicit daily known-at timestamp to the daily feature table.",
        )

    _, summary = asof_join_frame(
        signal_df,
        daily_feature_df,
        by=by,
        left_on=signal_time_col,
        right_effective_col=daily_effective_date_col,
        right_known_at_col=daily_known_at_col,
        strict_known_at=False,
    )
    if summary.future_blocked_count > 0:
        return IntegrityCheckResult(
            check_name="check_daily_feature_asof_intraday",
            status="fail",
            severity="error",
            object_name=object_name,
            module_name="research_integrity.leakage_checks",
            message=(
                f"{summary.future_blocked_count} intraday rows align to daily rows that are "
                "not known yet"
            ),
            remediation="Shift daily features to their known-at timestamp before intraday joins.",
            metrics=summary.to_dict(),
        )
    if summary.matched_count == 0:
        return IntegrityCheckResult(
            check_name="check_daily_feature_asof_intraday",
            status="warn",
            severity="warning",
            object_name=object_name,
            module_name="research_integrity.leakage_checks",
            message="daily as-of alignment produced no matches; coverage is empty",
            remediation="Check by-keys and effective-date coverage for daily features.",
            metrics=summary.to_dict(),
        )

    return IntegrityCheckResult(
        check_name="check_daily_feature_asof_intraday",
        status="pass",
        severity="info",
        object_name=object_name,
        module_name="research_integrity.leakage_checks",
        message="daily features satisfy intraday as-of known-at constraints",
        metrics=summary.to_dict(),
    )


def check_signal_execution_gap_is_respected(
    frame: pd.DataFrame,
    *,
    signal_bar_timestamp_col: str = "signal_bar_timestamp",
    execution_bar_timestamp_col: str = "execution_bar_timestamp",
    min_gap_bars: int = 1,
    bar_size: str | pd.Timedelta | None = None,
    object_name: str = "execution_gap",
) -> IntegrityCheckResult:
    """Fail when execution occurs earlier than the required signal-to-fill bar gap."""

    if min_gap_bars < 0:
        raise AlphaLabConfigError("min_gap_bars must be >= 0")

    missing = sorted({signal_bar_timestamp_col, execution_bar_timestamp_col} - set(frame.columns))
    if missing:
        return _missing_column_result(
            "check_signal_execution_gap_is_respected",
            object_name,
            missing,
        )

    signal_bar = _coerce_datetime(frame[signal_bar_timestamp_col])
    execution_bar = _coerce_datetime(frame[execution_bar_timestamp_col])
    invalid = signal_bar.isna() | execution_bar.isna()
    if bool(invalid.any()):
        return _invalid_timestamp_result(
            "check_signal_execution_gap_is_respected",
            object_name,
            signal_bar_timestamp_col,
            execution_bar_timestamp_col,
        )

    bar_delta = _resolve_bar_delta(signal_bar, execution_bar, bar_size)
    if bar_delta is None:
        return IntegrityCheckResult(
            check_name="check_signal_execution_gap_is_respected",
            status="warn",
            severity="warning",
            object_name=object_name,
            module_name="research_integrity.leakage_checks",
            message=(
                "cannot infer bar size from timestamps; execution-gap audit is advisory only"
            ),
            remediation="Provide explicit bar_size for strict bar-gap validation.",
        )

    raw_gap = execution_bar - signal_bar
    gap_bars = raw_gap / bar_delta
    violations = gap_bars < float(min_gap_bars)
    n_violations = int(violations.sum())
    if n_violations > 0:
        return IntegrityCheckResult(
            check_name="check_signal_execution_gap_is_respected",
            status="fail",
            severity="error",
            object_name=object_name,
            module_name="research_integrity.leakage_checks",
            message=(
                f"{n_violations} rows violate required signal-to-execution gap of "
                f"{min_gap_bars} bar(s)"
            ),
            remediation="Apply execution delay consistently before replay order fills.",
            metrics={
                "rows_compared": int(len(frame)),
                "violations": n_violations,
                "min_gap_bars_required": int(min_gap_bars),
            },
        )

    return IntegrityCheckResult(
        check_name="check_signal_execution_gap_is_respected",
        status="pass",
        severity="info",
        object_name=object_name,
        module_name="research_integrity.leakage_checks",
        message="signal-to-execution gap is respected",
        metrics={"rows_compared": int(len(frame)), "min_gap_bars_required": int(min_gap_bars)},
    )


def check_same_bar_close_execution_conflict(
    frame: pd.DataFrame,
    *,
    signal_bar_timestamp_col: str = "signal_bar_timestamp",
    execution_bar_timestamp_col: str = "execution_bar_timestamp",
    execution_price_rule_col: str = "execution_price_rule",
    allow_same_bar_close: bool = False,
    object_name: str = "same_bar_close_conflict",
) -> IntegrityCheckResult:
    """Fail when close-based execution happens on the same signal bar."""

    missing = sorted(
        {signal_bar_timestamp_col, execution_bar_timestamp_col, execution_price_rule_col}
        - set(frame.columns)
    )
    if missing:
        return _missing_column_result(
            "check_same_bar_close_execution_conflict",
            object_name,
            missing,
        )

    signal_bar = _coerce_datetime(frame[signal_bar_timestamp_col])
    execution_bar = _coerce_datetime(frame[execution_bar_timestamp_col])
    invalid = signal_bar.isna() | execution_bar.isna()
    if bool(invalid.any()):
        return _invalid_timestamp_result(
            "check_same_bar_close_execution_conflict",
            object_name,
            signal_bar_timestamp_col,
            execution_bar_timestamp_col,
        )

    rule_text = frame[execution_price_rule_col].astype(str).str.strip().str.lower()
    same_bar = signal_bar == execution_bar
    close_rule = rule_text.str.contains("close", na=False)
    conflict = same_bar & close_rule
    n_conflict = int(conflict.sum())
    if n_conflict > 0 and not allow_same_bar_close:
        return IntegrityCheckResult(
            check_name="check_same_bar_close_execution_conflict",
            status="fail",
            severity="error",
            object_name=object_name,
            module_name="research_integrity.leakage_checks",
            message=(
                f"{n_conflict} rows use same-bar close execution with close-derived signals"
            ),
            remediation="Use next-bar or delayed execution when signals are computed on bar close.",
            metrics={"rows_compared": int(len(frame)), "conflicts": n_conflict},
        )

    return IntegrityCheckResult(
        check_name="check_same_bar_close_execution_conflict",
        status="pass",
        severity="info",
        object_name=object_name,
        module_name="research_integrity.leakage_checks",
        message="no prohibited same-bar close execution conflicts detected",
        metrics={"rows_compared": int(len(frame)), "conflicts": n_conflict},
    )


def check_session_boundary_execution_consistency(
    frame: pd.DataFrame,
    *,
    signal_session_col: str = "signal_session",
    execution_session_col: str = "execution_session",
    signal_is_session_end_col: str = "is_session_end_signal",
    session_boundary_policy: str = "next_session",
    object_name: str = "session_boundary_execution",
) -> IntegrityCheckResult:
    """Audit consistency of session-end signal handling and next-session execution."""

    if session_boundary_policy.strip().lower() in {"same_session", "allow_same_session"}:
        return IntegrityCheckResult(
            check_name="check_session_boundary_execution_consistency",
            status="pass",
            severity="info",
            object_name=object_name,
            module_name="research_integrity.leakage_checks",
            message="session policy allows same-session execution",
        )

    required = {signal_session_col, execution_session_col}
    missing = sorted(required - set(frame.columns))
    if missing:
        return _missing_column_result(
            "check_session_boundary_execution_consistency",
            object_name,
            missing,
        )
    if signal_is_session_end_col not in frame.columns:
        return IntegrityCheckResult(
            check_name="check_session_boundary_execution_consistency",
            status="warn",
            severity="warning",
            object_name=object_name,
            module_name="research_integrity.leakage_checks",
            message=(
                f"missing {signal_is_session_end_col!r}; cannot verify session-end execution "
                "timing strictly"
            ),
            remediation="Provide explicit session-end flags for strict boundary validation.",
        )

    end_mask = frame[signal_is_session_end_col].astype(bool)
    if int(end_mask.sum()) == 0:
        return IntegrityCheckResult(
            check_name="check_session_boundary_execution_consistency",
            status="pass",
            severity="info",
            object_name=object_name,
            module_name="research_integrity.leakage_checks",
            message="no session-end signals found; boundary check is vacuously satisfied",
        )

    signal_session = frame.loc[end_mask, signal_session_col].astype(str)
    execution_session = frame.loc[end_mask, execution_session_col].astype(str)
    violations = execution_session <= signal_session
    n_violations = int(violations.sum())
    if n_violations > 0:
        return IntegrityCheckResult(
            check_name="check_session_boundary_execution_consistency",
            status="fail",
            severity="error",
            object_name=object_name,
            module_name="research_integrity.leakage_checks",
            message=(
                f"{n_violations} session-end signals execute without advancing to next session"
            ),
            remediation=(
                "Ensure session-end signals are filled on the next tradable session per policy."
            ),
            metrics={
                "session_end_rows": int(end_mask.sum()),
                "violations": n_violations,
            },
        )

    return IntegrityCheckResult(
        check_name="check_session_boundary_execution_consistency",
        status="pass",
        severity="info",
        object_name=object_name,
        module_name="research_integrity.leakage_checks",
        message="session boundary execution behavior is consistent with next-session policy",
        metrics={"session_end_rows": int(end_mask.sum())},
    )


def check_execution_delay_matches_research_assumption(
    frame: pd.DataFrame,
    *,
    expected_delay_bars: int,
    observed_delay_col: str = "observed_execution_delay_bars",
    fail_threshold_bars: int = 2,
    object_name: str = "execution_delay_assumption",
) -> IntegrityCheckResult:
    """Compare observed replay delay vs expected research execution delay in bars."""

    if expected_delay_bars < 0:
        raise AlphaLabConfigError("expected_delay_bars must be >= 0")
    if fail_threshold_bars < 1:
        raise AlphaLabConfigError("fail_threshold_bars must be >= 1")
    if observed_delay_col not in frame.columns:
        return _missing_column_result(
            "check_execution_delay_matches_research_assumption",
            object_name,
            [observed_delay_col],
        )

    observed = pd.to_numeric(frame[observed_delay_col], errors="coerce")
    if observed.isna().any():
        return IntegrityCheckResult(
            check_name="check_execution_delay_matches_research_assumption",
            status="fail",
            severity="error",
            object_name=object_name,
            module_name="research_integrity.leakage_checks",
            message=f"column {observed_delay_col!r} contains non-numeric delay values",
            remediation="Provide integer bar-delay observations for replay fills.",
        )

    diff = (observed - float(expected_delay_bars)).abs()
    warn_count = int((diff > 0).sum())
    fail_count = int((diff >= float(fail_threshold_bars)).sum())
    if fail_count > 0:
        return IntegrityCheckResult(
            check_name="check_execution_delay_matches_research_assumption",
            status="fail",
            severity="error",
            object_name=object_name,
            module_name="research_integrity.leakage_checks",
            message=(
                f"{fail_count} rows deviate from expected delay by >= {fail_threshold_bars} bar(s)"
            ),
            remediation="Align replay execution delay with research assumptions.",
            metrics={
                "rows_compared": int(len(frame)),
                "expected_delay_bars": int(expected_delay_bars),
                "warn_mismatch_rows": warn_count,
                "fail_mismatch_rows": fail_count,
            },
        )
    if warn_count > 0:
        return IntegrityCheckResult(
            check_name="check_execution_delay_matches_research_assumption",
            status="warn",
            severity="warning",
            object_name=object_name,
            module_name="research_integrity.leakage_checks",
            message=f"{warn_count} rows deviate from expected delay by 1 bar",
            remediation="Check whether delay drift is intentional or adapter-induced.",
            metrics={
                "rows_compared": int(len(frame)),
                "expected_delay_bars": int(expected_delay_bars),
                "warn_mismatch_rows": warn_count,
                "fail_mismatch_rows": fail_count,
            },
        )

    return IntegrityCheckResult(
        check_name="check_execution_delay_matches_research_assumption",
        status="pass",
        severity="info",
        object_name=object_name,
        module_name="research_integrity.leakage_checks",
        message="observed execution delays match research assumptions",
        metrics={"rows_compared": int(len(frame)), "expected_delay_bars": int(expected_delay_bars)},
    )


def check_participation_cap_metadata_present(
    *,
    capacity_model: str | None,
    max_participation_rate: float | None,
    min_tradable_bar_volume: float | None,
    object_name: str = "execution_capacity_metadata",
) -> IntegrityCheckResult:
    """Warn when replay capacity semantics are missing for mid-frequency workflows."""

    mode = (capacity_model or "").strip().lower()
    if mode == "":
        return IntegrityCheckResult(
            check_name="check_participation_cap_metadata_present",
            status="warn",
            severity="warning",
            object_name=object_name,
            module_name="research_integrity.leakage_checks",
            message="capacity_model is missing; execution liquidity assumptions are implicit",
            remediation=(
                "Set capacity_model explicitly (for example unbounded or "
                "participation_capped)."
            ),
        )

    if mode in {"participation_capped", "participation_capped_with_min_volume"} and (
        max_participation_rate is None
    ):
        return IntegrityCheckResult(
            check_name="check_participation_cap_metadata_present",
            status="warn",
            severity="warning",
            object_name=object_name,
            module_name="research_integrity.leakage_checks",
            message="capacity model uses participation caps but max_participation_rate is missing",
            remediation="Provide max_participation_rate for auditable capacity constraints.",
        )

    if mode in {"min_bar_volume_gate", "participation_capped_with_min_volume"} and (
        min_tradable_bar_volume is None
    ):
        return IntegrityCheckResult(
            check_name="check_participation_cap_metadata_present",
            status="warn",
            severity="warning",
            object_name=object_name,
            module_name="research_integrity.leakage_checks",
            message="capacity model uses bar-volume gating but min_tradable_bar_volume is missing",
            remediation="Provide min_tradable_bar_volume for auditable tradability thresholds.",
        )

    return IntegrityCheckResult(
        check_name="check_participation_cap_metadata_present",
        status="pass",
        severity="info",
        object_name=object_name,
        module_name="research_integrity.leakage_checks",
        message="capacity metadata is explicitly defined",
        metrics={
            "capacity_model": mode,
            "max_participation_rate": max_participation_rate,
            "min_tradable_bar_volume": min_tradable_bar_volume,
        },
    )


def check_tradability_mask_respected(
    frame: pd.DataFrame,
    *,
    is_tradable_col: str = "is_tradable",
    executed_col: str = "did_execute",
    object_name: str = "tradability_mask",
) -> IntegrityCheckResult:
    """Fail when replay executes rows explicitly marked non-tradable."""

    missing = sorted({is_tradable_col, executed_col} - set(frame.columns))
    if missing:
        return _missing_column_result("check_tradability_mask_respected", object_name, missing)

    is_tradable = frame[is_tradable_col].astype(bool)
    executed = frame[executed_col].astype(bool)
    violations = (~is_tradable) & executed
    n_violations = int(violations.sum())
    if n_violations > 0:
        return IntegrityCheckResult(
            check_name="check_tradability_mask_respected",
            status="fail",
            severity="error",
            object_name=object_name,
            module_name="research_integrity.leakage_checks",
            message=f"{n_violations} executed rows violate tradability mask",
            remediation="Apply tradability masks before order generation/fill.",
            metrics={"rows_compared": int(len(frame)), "violations": n_violations},
        )

    return IntegrityCheckResult(
        check_name="check_tradability_mask_respected",
        status="pass",
        severity="info",
        object_name=object_name,
        module_name="research_integrity.leakage_checks",
        message="tradability mask is respected by execution rows",
        metrics={"rows_compared": int(len(frame))},
    )


def _coerce_datetime(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce")


def _missing_column_result(
    check_name: str,
    object_name: str,
    missing: list[str],
) -> IntegrityCheckResult:
    return IntegrityCheckResult(
        check_name=check_name,
        status="fail",
        severity="error",
        object_name=object_name,
        module_name="research_integrity.leakage_checks",
        message=f"missing required columns: {missing}",
        remediation="Provide the required timestamp and semantics columns.",
    )


def _invalid_timestamp_result(
    check_name: str,
    object_name: str,
    col_a: str,
    col_b: str,
) -> IntegrityCheckResult:
    return IntegrityCheckResult(
        check_name=check_name,
        status="fail",
        severity="error",
        object_name=object_name,
        module_name="research_integrity.leakage_checks",
        message=f"invalid timestamp values in {col_a!r} or {col_b!r}",
        remediation="Fix timestamp parsing errors before running integrity checks.",
    )


def _resolve_bar_delta(
    signal_bar: pd.Series,
    execution_bar: pd.Series,
    bar_size: str | pd.Timedelta | None,
) -> pd.Timedelta | None:
    if bar_size is not None:
        delta = pd.Timedelta(bar_size)
        return delta if delta > pd.Timedelta(0) else None

    union = pd.Index(signal_bar).union(pd.Index(execution_bar))
    ordered = pd.Series(union).drop_duplicates().sort_values(kind="mergesort")
    if len(ordered) < 2:
        return None
    diffs = ordered.diff().dropna()
    positive = diffs[diffs > pd.Timedelta(0)]
    if positive.empty:
        return None
    return positive.min()
