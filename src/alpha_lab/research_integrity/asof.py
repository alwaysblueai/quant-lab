from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from alpha_lab.exceptions import AlphaLabConfigError, AlphaLabDataError
from alpha_lab.research_integrity.contracts import IntegrityCheckResult


@dataclass(frozen=True)
class TimeSemanticsMetadata:
    """Lightweight metadata describing how table timestamps should be interpreted."""

    dataset_name: str
    date_col: str = "date"
    effective_date_col: str | None = None
    available_at_col: str | None = None
    known_at_col: str | None = None
    assumptions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.dataset_name.strip():
            raise AlphaLabConfigError("dataset_name must be non-empty")

    @property
    def resolved_known_at_col(self) -> str | None:
        if self.known_at_col is not None:
            return self.known_at_col
        return self.available_at_col

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset_name": self.dataset_name,
            "date_col": self.date_col,
            "effective_date_col": self.effective_date_col,
            "available_at_col": self.available_at_col,
            "known_at_col": self.known_at_col,
            "assumptions": list(self.assumptions),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> TimeSemanticsMetadata:
        assumptions_raw = payload.get("assumptions")
        assumptions: tuple[str, ...]
        if isinstance(assumptions_raw, list):
            assumptions = tuple(str(item) for item in assumptions_raw)
        else:
            assumptions = ()
        return cls(
            dataset_name=str(payload.get("dataset_name") or ""),
            date_col=str(payload.get("date_col") or "date"),
            effective_date_col=_optional_str(payload.get("effective_date_col")),
            available_at_col=_optional_str(payload.get("available_at_col")),
            known_at_col=_optional_str(payload.get("known_at_col")),
            assumptions=assumptions,
        )


@dataclass(frozen=True)
class AsofJoinSummary:
    """Audit summary for one explicit as-of join call."""

    left_rows: int
    matched_count: int
    unmatched_count: int
    future_blocked_count: int
    lag_blocked_count: int
    effective_date_column: str
    known_at_column: str | None
    direction: str
    max_lag: str | None
    strict_known_at: bool
    strict_max_lag: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "left_rows": self.left_rows,
            "matched_count": self.matched_count,
            "unmatched_count": self.unmatched_count,
            "future_blocked_count": self.future_blocked_count,
            "lag_blocked_count": self.lag_blocked_count,
            "effective_date_column": self.effective_date_column,
            "known_at_column": self.known_at_column,
            "direction": self.direction,
            "max_lag": self.max_lag,
            "strict_known_at": self.strict_known_at,
            "strict_max_lag": self.strict_max_lag,
        }


@dataclass(frozen=True)
class AsofMonotonicitySummary:
    """Validation summary for as-of monotonicity assumptions."""

    dataset_name: str
    total_rows: int
    group_count: int
    known_at_column: str | None
    non_monotonic_known_at_rows: int
    known_before_effective_rows: int
    duplicate_effective_rows: int
    result: IntegrityCheckResult


@dataclass(frozen=True)
class ForwardFillLagSummary:
    """Validation summary for as-of forward-fill lag boundaries."""

    evaluated_rows: int
    violation_count: int
    max_lag_observed: str | None
    max_lag_allowed: str
    result: IntegrityCheckResult


def attach_time_semantics(
    frame: pd.DataFrame,
    metadata: TimeSemanticsMetadata,
    *,
    inplace: bool = False,
) -> pd.DataFrame:
    """Attach serializable time-semantics metadata to a DataFrame."""

    out = frame if inplace else frame.copy()
    out.attrs["time_semantics"] = metadata.to_dict()
    return out


def read_time_semantics(frame: pd.DataFrame) -> TimeSemanticsMetadata | None:
    """Read time-semantics metadata from DataFrame attrs when present."""

    payload = frame.attrs.get("time_semantics")
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise AlphaLabDataError("frame.attrs['time_semantics'] must be a dict when present")
    return TimeSemanticsMetadata.from_dict(payload)


def validate_asof_monotonicity(
    frame: pd.DataFrame,
    *,
    by: str | tuple[str, ...] = "asset",
    effective_date_col: str = "effective_date",
    known_at_col: str | None = None,
    dataset_name: str = "asof_input",
) -> AsofMonotonicitySummary:
    """Validate monotonicity assumptions required by point-in-time as-of alignment."""

    by_cols = _normalize_by(by)
    missing = _missing_columns(frame, [*by_cols, effective_date_col])
    if missing:
        result = IntegrityCheckResult(
            check_name="validate_asof_monotonicity",
            status="fail",
            severity="error",
            object_name=dataset_name,
            module_name="research_integrity.asof",
            message=f"missing required columns: {missing}",
            remediation="Provide grouping and effective_date columns before PIT validation.",
        )
        return AsofMonotonicitySummary(
            dataset_name=dataset_name,
            total_rows=len(frame),
            group_count=0,
            known_at_column=known_at_col,
            non_monotonic_known_at_rows=0,
            known_before_effective_rows=0,
            duplicate_effective_rows=0,
            result=result,
        )

    known_col = _resolve_known_at_column(frame, known_at_col)

    table = frame.loc[:, [*by_cols, effective_date_col]].copy()
    table[effective_date_col] = _coerce_datetime(
        table[effective_date_col],
        column_name=effective_date_col,
        object_name=dataset_name,
    )

    if known_col is not None:
        table[known_col] = _coerce_datetime(
            frame[known_col],
            column_name=known_col,
            object_name=dataset_name,
        )

    duplicate_effective = int(table.duplicated(subset=[*by_cols, effective_date_col]).sum())

    if known_col is None:
        result = IntegrityCheckResult(
            check_name="validate_asof_monotonicity",
            status="warn",
            severity="warning",
            object_name=dataset_name,
            module_name="research_integrity.asof",
            message=(
                "known_at/available_at column is missing; monotonicity checks cannot verify "
                "publication-time ordering"
            ),
            remediation="Add available_at or known_at to the auxiliary table.",
            metrics={"duplicate_effective_rows": duplicate_effective},
        )
        return AsofMonotonicitySummary(
            dataset_name=dataset_name,
            total_rows=len(table),
            group_count=int(table.groupby(by_cols, dropna=False).ngroups),
            known_at_column=None,
            non_monotonic_known_at_rows=0,
            known_before_effective_rows=0,
            duplicate_effective_rows=duplicate_effective,
            result=result,
        )

    ordered = (
        table.sort_values([*by_cols, effective_date_col], kind="mergesort")
        .reset_index(drop=True)
    )
    known_diff = ordered.groupby(by_cols, dropna=False)[known_col].diff()
    non_monotonic_rows = int((known_diff < pd.Timedelta(0)).fillna(False).sum())
    known_before_effective = int((ordered[known_col] < ordered[effective_date_col]).sum())

    if known_before_effective > 0 or non_monotonic_rows > 0:
        status: Literal["fail", "warn", "pass"] = "fail"
        severity: Literal["error", "warning", "info"] = "error"
        message = (
            "as-of timestamp monotonicity violated: known_at must be non-decreasing by key and "
            "cannot be earlier than effective_date"
        )
        remediation = (
            "Fix auxiliary timestamps or disable rows with non-causal publication times before "
            "PIT joins."
        )
    elif duplicate_effective > 0:
        status = "warn"
        severity = "warning"
        message = "duplicate (by, effective_date) rows detected; as-of joins may be ambiguous"
        remediation = "Deduplicate auxiliary inputs by a deterministic versioning rule."
    else:
        status = "pass"
        severity = "info"
        message = "as-of monotonicity checks passed"
        remediation = None

    result = IntegrityCheckResult(
        check_name="validate_asof_monotonicity",
        status=status,
        severity=severity,
        object_name=dataset_name,
        module_name="research_integrity.asof",
        message=message,
        remediation=remediation,
        metrics={
            "non_monotonic_known_at_rows": non_monotonic_rows,
            "known_before_effective_rows": known_before_effective,
            "duplicate_effective_rows": duplicate_effective,
        },
    )

    return AsofMonotonicitySummary(
        dataset_name=dataset_name,
        total_rows=len(ordered),
        group_count=int(ordered.groupby(by_cols, dropna=False).ngroups),
        known_at_column=known_col,
        non_monotonic_known_at_rows=non_monotonic_rows,
        known_before_effective_rows=known_before_effective,
        duplicate_effective_rows=duplicate_effective,
        result=result,
    )


def asof_join_frame(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    by: str | tuple[str, ...] = "asset",
    left_on: str = "date",
    right_effective_col: str = "effective_date",
    right_known_at_col: str | None = None,
    direction: Literal["backward"] = "backward",
    max_lag: pd.Timedelta | str | None = None,
    strict_known_at: bool = True,
    strict_max_lag: bool = True,
    suffixes: tuple[str, str] = ("", "_asof"),
) -> tuple[pd.DataFrame, AsofJoinSummary]:
    """Perform explicit as-of alignment with optional known_at and lag guards."""

    if left_on == right_effective_col:
        raise AlphaLabConfigError(
            "left_on and right_effective_col must differ for explicit as-of alignment; "
            "use explicit column names such as date/effective_date"
        )

    by_cols = _normalize_by(by)
    left_missing = _missing_columns(left, [*by_cols, left_on])
    right_missing = _missing_columns(right, [*by_cols, right_effective_col])
    if left_missing:
        raise AlphaLabDataError(f"left frame missing required columns: {left_missing}")
    if right_missing:
        raise AlphaLabDataError(f"right frame missing required columns: {right_missing}")

    known_col = _resolve_known_at_column(right, right_known_at_col)

    left_table = left.copy()
    right_table = right.copy()

    left_table[left_on] = _coerce_datetime(
        left_table[left_on],
        column_name=left_on,
        object_name="left",
    )
    right_table[right_effective_col] = _coerce_datetime(
        right_table[right_effective_col],
        column_name=right_effective_col,
        object_name="right",
    )

    if known_col is not None:
        right_table[known_col] = _coerce_datetime(
            right_table[known_col],
            column_name=known_col,
            object_name="right",
        )

    left_table = (
        left_table.sort_values([*by_cols, left_on], kind="mergesort")
        .reset_index(drop=True)
    )
    right_table = right_table.sort_values(
        [*by_cols, right_effective_col], kind="mergesort"
    ).reset_index(drop=True)

    merged = pd.merge_asof(
        left_table,
        right_table,
        left_on=left_on,
        right_on=right_effective_col,
        by=by_cols,
        direction=direction,
        suffixes=suffixes,
    )

    right_payload_cols = [
        column for column in right_table.columns if column not in by_cols
    ]
    right_output_cols = [
        _merged_column_name(column, left_table.columns, suffixes[1])
        for column in right_payload_cols
    ]

    right_effective_out = _merged_column_name(right_effective_col, left_table.columns, suffixes[1])
    known_out = (
        _merged_column_name(known_col, left_table.columns, suffixes[1])
        if known_col is not None
        else None
    )

    matched_mask = merged[right_effective_out].notna()

    future_block_mask = pd.Series(False, index=merged.index)
    if known_out is not None:
        future_block_mask = matched_mask & (merged[known_out] > merged[left_on])

    resolved_max_lag = _coerce_timedelta(max_lag)
    lag_block_mask = pd.Series(False, index=merged.index)
    if resolved_max_lag is not None:
        lag_values = merged[left_on] - merged[right_effective_out]
        lag_block_mask = matched_mask & lag_values.gt(resolved_max_lag)

    if strict_known_at and future_block_mask.any():
        merged.loc[future_block_mask, right_output_cols] = pd.NA

    if strict_max_lag and lag_block_mask.any():
        merged.loc[lag_block_mask, right_output_cols] = pd.NA

    matched_after = int(merged[right_effective_out].notna().sum())

    summary = AsofJoinSummary(
        left_rows=len(merged),
        matched_count=matched_after,
        unmatched_count=int(len(merged) - matched_after),
        future_blocked_count=int(future_block_mask.sum()),
        lag_blocked_count=int(lag_block_mask.sum()),
        effective_date_column=right_effective_col,
        known_at_column=known_col,
        direction=direction,
        max_lag=(str(resolved_max_lag) if resolved_max_lag is not None else None),
        strict_known_at=strict_known_at,
        strict_max_lag=strict_max_lag,
    )
    return merged, summary


def validate_forward_fill_lag(
    aligned_frame: pd.DataFrame,
    *,
    left_date_col: str = "date",
    matched_effective_col: str = "effective_date",
    max_lag: pd.Timedelta | str = "7D",
    object_name: str = "asof_join",
) -> ForwardFillLagSummary:
    """Validate as-of forward-fill lag boundaries on an aligned frame."""

    missing = _missing_columns(aligned_frame, [left_date_col, matched_effective_col])
    if missing:
        result = IntegrityCheckResult(
            check_name="validate_forward_fill_lag",
            status="fail",
            severity="error",
            object_name=object_name,
            module_name="research_integrity.asof",
            message=f"missing required columns: {missing}",
            remediation="Provide both signal date and matched effective-date columns.",
        )
        return ForwardFillLagSummary(
            evaluated_rows=0,
            violation_count=0,
            max_lag_observed=None,
            max_lag_allowed=str(_coerce_timedelta(max_lag)),
            result=result,
        )

    left_dates = _coerce_datetime(
        aligned_frame[left_date_col],
        column_name=left_date_col,
        object_name=object_name,
    )
    right_dates = _coerce_datetime(
        aligned_frame[matched_effective_col],
        column_name=matched_effective_col,
        object_name=object_name,
    )

    usable = right_dates.notna()
    if not usable.any():
        result = IntegrityCheckResult(
            check_name="validate_forward_fill_lag",
            status="warn",
            severity="warning",
            object_name=object_name,
            module_name="research_integrity.asof",
            message="no matched effective_date rows available for forward-fill lag validation",
            remediation="Check upstream as-of inputs and matching keys.",
        )
        return ForwardFillLagSummary(
            evaluated_rows=0,
            violation_count=0,
            max_lag_observed=None,
            max_lag_allowed=str(_coerce_timedelta(max_lag)),
            result=result,
        )

    resolved_max_lag = _coerce_timedelta(max_lag)
    if resolved_max_lag is None:
        raise AlphaLabConfigError("max_lag must resolve to a finite timedelta")

    lag = left_dates[usable] - right_dates[usable]
    violations = lag > resolved_max_lag
    violation_count = int(violations.sum())

    if violation_count > 0:
        status: Literal["fail", "warn", "pass"] = "fail"
        severity: Literal["error", "warning", "info"] = "error"
        message = "forward-fill lag boundary violated"
        remediation = (
            "Use max_lag guard during as-of alignment or reduce fill window for low-frequency data."
        )
    else:
        status = "pass"
        severity = "info"
        message = "forward-fill lag boundary check passed"
        remediation = None

    max_lag_observed = lag.max() if len(lag) > 0 else None

    result = IntegrityCheckResult(
        check_name="validate_forward_fill_lag",
        status=status,
        severity=severity,
        object_name=object_name,
        module_name="research_integrity.asof",
        message=message,
        remediation=remediation,
        metrics={
            "evaluated_rows": int(usable.sum()),
            "violation_count": violation_count,
        },
    )

    return ForwardFillLagSummary(
        evaluated_rows=int(usable.sum()),
        violation_count=violation_count,
        max_lag_observed=(str(max_lag_observed) if max_lag_observed is not None else None),
        max_lag_allowed=str(resolved_max_lag),
        result=result,
    )


def pit_check(
    df: pd.DataFrame,
    *,
    max_allowed_date: pd.Timestamp | str,
    date_col: str = "date",
    object_name: str = "input",
) -> IntegrityCheckResult:
    """Run check_no_future_dates_in_input and immediately raise on hard failure.

    Shared PIT gate used by both :func:`~alpha_lab.experiment.run_factor_experiment`
    and :func:`~alpha_lab.walk_forward._execute_fold`.  Callers that collect
    an integrity-check list should append the returned result; the hard-abort
    raise fires before this function returns when the check fails.

    Parameters
    ----------
    df:
        DataFrame to validate.  Must contain ``date_col``.
    max_allowed_date:
        No row in ``df[date_col]`` may exceed this cutoff.
    date_col:
        Name of the date column.  Defaults to ``"date"``.
    object_name:
        Label used in the integrity-check result for error messages.

    Returns
    -------
    IntegrityCheckResult
        The passing check result (only reached when no hard failure occurs).

    Raises
    ------
    AlphaLabExperimentError
        If ``df`` contains dates beyond ``max_allowed_date``.
    """
    from alpha_lab.research_integrity.exceptions import raise_on_hard_failures
    from alpha_lab.research_integrity.leakage_checks import check_no_future_dates_in_input

    result = check_no_future_dates_in_input(
        df,
        max_allowed_date=max_allowed_date,
        date_col=date_col,
        object_name=object_name,
    )
    raise_on_hard_failures((result,))
    return result


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _normalize_by(by: str | tuple[str, ...]) -> list[str]:
    if isinstance(by, str):
        values = [by]
    else:
        values = list(by)
    if not values:
        raise AlphaLabConfigError("by must contain at least one column")
    bad = [value for value in values if not str(value).strip()]
    if bad:
        raise AlphaLabConfigError("by columns must be non-empty strings")
    return [str(value) for value in values]


def _missing_columns(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    return sorted(set(columns) - set(frame.columns))


def _coerce_datetime(series: pd.Series, *, column_name: str, object_name: str) -> pd.Series:
    out = pd.to_datetime(series, errors="coerce")
    if out.isna().any():
        raise AlphaLabDataError(f"{object_name}.{column_name} contains invalid timestamp values")
    return out


def _coerce_timedelta(value: pd.Timedelta | str | None) -> pd.Timedelta | None:
    if value is None:
        return None
    resolved = pd.Timedelta(value)
    if resolved < pd.Timedelta(0):
        raise AlphaLabConfigError("max_lag must be >= 0")
    return resolved


def _resolve_known_at_column(frame: pd.DataFrame, preferred: str | None) -> str | None:
    if preferred is not None:
        if preferred not in frame.columns:
            raise AlphaLabDataError(f"known-at column {preferred!r} is missing from frame")
        return preferred
    if "known_at" in frame.columns:
        return "known_at"
    if "available_at" in frame.columns:
        return "available_at"
    return None


def _merged_column_name(column: str, left_columns: pd.Index, right_suffix: str) -> str:
    return f"{column}{right_suffix}" if column in left_columns else column
