from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

RAW_TUSHARE_DIRNAME = "tushare"
STANDARDIZED_TUSHARE_DIRNAME = "tushare_standardized"
RESEARCH_INPUT_TUSHARE_DIRNAME = "tushare_research_inputs"

ENDPOINT_STATUS_SUCCESS = "success"
ENDPOINT_STATUS_FAILED = "failed"
ENDPOINT_STATUS_DEGRADED = "degraded"
ENDPOINT_STATUS_VALUES = (
    ENDPOINT_STATUS_SUCCESS,
    ENDPOINT_STATUS_FAILED,
    ENDPOINT_STATUS_DEGRADED,
)


RAW_SNAPSHOT_SORT_KEYS: dict[str, tuple[str, ...]] = {
    "trade_cal": ("exchange", "cal_date"),
    "stock_basic": ("ts_code", "list_status"),
    "daily": ("trade_date", "ts_code"),
    "daily_basic": ("trade_date", "ts_code"),
    "adj_factor": ("trade_date", "ts_code"),
    "suspend_d": ("trade_date", "ts_code"),
    "stk_limit": ("trade_date", "ts_code"),
    "fina_indicator": ("ann_date", "end_date", "ts_code"),
    "fina_indicator_vip": ("ann_date", "end_date", "ts_code"),
}


@dataclass(frozen=True)
class StandardizedTushareTables:
    """Normalized vendor tables used by bundle builders."""

    prices: pd.DataFrame
    asset_metadata: pd.DataFrame
    trade_calendar: pd.DataFrame
    market_state: pd.DataFrame
    daily_fundamentals: pd.DataFrame
    financial_indicators: pd.DataFrame
    snapshot_dir: Path
    unavailable_raw_endpoints: tuple[str, ...] = ()


def normalize_tushare_asset(raw: object) -> str:
    """Normalize Tushare security identifiers to canonical asset strings."""

    text = str(raw).strip().upper()
    if not text:
        raise ValueError("ts_code/asset identifier must be non-empty")
    return text


def normalize_trade_date(series: pd.Series, *, column_name: str) -> pd.Series:
    parsed = pd.to_datetime(series, format="%Y%m%d", errors="coerce")
    if parsed.isna().any():
        # Fall back to generic parser for already-normalized strings.
        parsed = pd.to_datetime(series, errors="coerce")
    if parsed.isna().any():
        raise ValueError(f"{column_name} contains invalid date values")
    return parsed
