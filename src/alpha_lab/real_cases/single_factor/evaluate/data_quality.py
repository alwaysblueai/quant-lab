from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from alpha_lab.data_quality.corporate_actions import detect_unadjusted_splits
from alpha_lab.data_quality.outlier_detection import detect_stale_prices, filter_zero_volume


def _build_data_quality_summary(
    *,
    prices: pd.DataFrame,
    integrity_checks: Iterable[object],
) -> dict[str, object]:
    suspended_rows = _count_suspended_rows(prices)
    stale_rows = _count_stale_rows(prices)
    suspected_split_rows = _count_suspected_split_rows(prices)
    warn_count, fail_count, hard_fail_count = _integrity_status_counts(integrity_checks)

    status = "pass"
    if fail_count > 0 or (suspected_split_rows is not None and suspected_split_rows > 0):
        status = "fail"
    elif (
        warn_count > 0
        or (suspended_rows is not None and suspended_rows > 0)
        or (stale_rows is not None and stale_rows > 0)
    ):
        status = "warn"

    return {
        "data_quality_status": status,
        "data_quality_suspended_rows": suspended_rows,
        "data_quality_stale_rows": stale_rows,
        "data_quality_suspected_split_rows": suspected_split_rows,
        "data_quality_integrity_warn_count": warn_count,
        "data_quality_integrity_fail_count": fail_count,
        "data_quality_hard_fail_count": hard_fail_count,
    }


def _count_suspended_rows(prices: pd.DataFrame) -> int | None:
    required = {"date", "asset", "volume"}
    if not required.issubset(set(prices.columns)):
        return None
    flagged = filter_zero_volume(prices, action="flag")
    if "is_suspended" not in flagged.columns:
        return None
    return int(flagged["is_suspended"].fillna(False).astype(bool).sum())


def _count_stale_rows(prices: pd.DataFrame) -> int | None:
    required = {"date", "asset", "close"}
    if not required.issubset(set(prices.columns)):
        return None
    flagged = detect_stale_prices(prices, max_identical_days=5)
    if "is_stale_price" not in flagged.columns:
        return None
    return int(flagged["is_stale_price"].fillna(False).astype(bool).sum())


def _count_suspected_split_rows(prices: pd.DataFrame) -> int | None:
    required = {"date", "asset", "close"}
    if not required.issubset(set(prices.columns)):
        return None
    flagged = detect_unadjusted_splits(prices, threshold=0.45)
    if "suspected_split" not in flagged.columns:
        return None
    return int(flagged["suspected_split"].fillna(False).astype(bool).sum())


def _integrity_status_counts(
    checks: Iterable[object],
) -> tuple[int, int, int]:
    warn_count = 0
    fail_count = 0
    hard_fail_count = 0
    for check in checks:
        status = str(getattr(check, "status", "")).strip().lower()
        severity = str(getattr(check, "severity", "")).strip().lower()
        if status == "warn":
            warn_count += 1
        if status == "fail":
            fail_count += 1
            if severity == "error":
                hard_fail_count += 1
    return warn_count, fail_count, hard_fail_count
