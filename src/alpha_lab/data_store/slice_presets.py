from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from alpha_lab.exceptions import AlphaLabConfigError


@dataclass(frozen=True)
class SlicePresetConfig:
    name: str
    lookback_years: int
    universe_name: str
    adjustment: str
    description: str


SLICE_PRESETS: dict[str, SlicePresetConfig] = {
    "pilot": SlicePresetConfig(
        name="pilot",
        lookback_years=3,
        universe_name="top_liquid_300",
        adjustment="qfq",
        description="最近 3 年、前复权、按日成交额前 300 做快速因子原型验证。",
    ),
    "standard": SlicePresetConfig(
        name="standard",
        lookback_years=5,
        universe_name="listed_90d",
        adjustment="qfq",
        description="最近 5 年、前复权、上市满 90 天，作为默认日频量价研究窗口。",
    ),
    "robust": SlicePresetConfig(
        name="robust",
        lookback_years=8,
        universe_name="listed_90d",
        adjustment="qfq",
        description="最近 8 年、前复权、上市满 90 天，用于稳健性复核。",
    ),
    "institutional": SlicePresetConfig(
        name="institutional",
        lookback_years=8,
        universe_name="institutional_ashare",
        adjustment="qfq",
        description=(
            "最近 8 年、前复权、上市满 180 天、剔除 ST/停牌，并过滤低流动性尾部，"
            "用于更接近私募常用研究股票池的复核。"
        ),
    ),
}

DEFAULT_SLICE_PRESET = "standard"


def get_slice_preset(name: str) -> SlicePresetConfig:
    try:
        return SLICE_PRESETS[str(name)]
    except KeyError as exc:
        raise AlphaLabConfigError(f"unknown slice preset: {name!r}") from exc


def resolve_slice_window(
    *,
    preset_name: str,
    start_date: str | None,
    end_date: str | None,
    fallback_end_date: str,
) -> tuple[str, str, SlicePresetConfig]:
    preset = get_slice_preset(preset_name)
    end_ts = pd.Timestamp(end_date or fallback_end_date)
    if pd.isna(end_ts):
        raise AlphaLabConfigError(f"Invalid end date: {end_date or fallback_end_date!r}")

    if start_date is None:
        start_ts = (end_ts - pd.DateOffset(years=preset.lookback_years)) + pd.Timedelta(days=1)
    else:
        start_ts = pd.Timestamp(start_date)
        if pd.isna(start_ts):
            raise AlphaLabConfigError(f"Invalid start date: {start_date!r}")

    if start_ts > end_ts:
        raise AlphaLabConfigError("start_date must be <= end_date")
    return start_ts.strftime("%Y-%m-%d"), end_ts.strftime("%Y-%m-%d"), preset
