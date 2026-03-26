from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from alpha_lab.config import PROCESSED_DATA_DIR
from alpha_lab.data_sources.tushare_cache import load_raw_snapshot
from alpha_lab.data_sources.tushare_schemas import (
    STANDARDIZED_TUSHARE_DIRNAME,
    StandardizedTushareTables,
    normalize_trade_date,
    normalize_tushare_asset,
)

_EPSILON = 1e-8


def build_standardized_tushare_tables(
    snapshot_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
) -> StandardizedTushareTables:
    """Transform raw Tushare snapshots into explicit internal research tables."""

    source_dir = Path(snapshot_dir).resolve()
    if not source_dir.exists() or not source_dir.is_dir():
        raise FileNotFoundError(f"raw Tushare snapshot directory does not exist: {source_dir}")

    manifest = _load_optional_manifest(source_dir)
    unavailable_raw = cast(object, manifest.get("unavailable_endpoints", []))
    unavailable_items = (
        unavailable_raw if isinstance(unavailable_raw, list | tuple | set) else []
    )
    unavailable = tuple(sorted({str(item) for item in unavailable_items if str(item).strip()}))

    daily = load_raw_snapshot(source_dir, endpoint="daily")
    stock_basic = load_raw_snapshot(source_dir, endpoint="stock_basic")
    trade_cal = load_raw_snapshot(source_dir, endpoint="trade_cal")
    daily_basic = load_raw_snapshot(source_dir, endpoint="daily_basic")
    adj_factor = _load_optional_snapshot(source_dir, endpoint="adj_factor")
    suspend = _load_optional_snapshot(source_dir, endpoint="suspend_d")
    stk_limit = _load_optional_snapshot(source_dir, endpoint="stk_limit")
    fina_indicator = _load_financial_indicator_snapshot(source_dir)

    prices = standardize_prices(daily, daily_basic=daily_basic, adj_factor=adj_factor)
    asset_metadata = standardize_asset_metadata(stock_basic)
    trade_calendar = standardize_trade_calendar(trade_cal)
    market_state = standardize_market_state(
        prices,
        asset_metadata=asset_metadata,
        suspend=suspend,
        stk_limit=stk_limit,
    )
    daily_fundamentals = standardize_daily_basic(daily_basic)
    financial_indicators = standardize_fina_indicator(fina_indicator)

    standardized_dir = (
        Path(output_dir).resolve()
        if output_dir is not None
        else PROCESSED_DATA_DIR / STANDARDIZED_TUSHARE_DIRNAME / source_dir.name
    )
    standardized_dir.mkdir(parents=True, exist_ok=True)

    _write_df(prices, standardized_dir / "prices.csv")
    _write_df(asset_metadata, standardized_dir / "asset_metadata.csv")
    _write_df(trade_calendar, standardized_dir / "trade_calendar.csv")
    _write_df(market_state, standardized_dir / "market_state.csv")
    _write_df(daily_fundamentals, standardized_dir / "daily_fundamentals.csv")
    _write_df(financial_indicators, standardized_dir / "financial_indicators.csv")

    standardization_manifest = {
        "snapshot_dir": str(source_dir),
        "standardized_dir": str(standardized_dir),
        "unavailable_raw_endpoints": list(unavailable),
        "notes": [
            "prices.close uses raw vendor close to avoid future-adjusted leakage",
            "prices.dollar_volume_yuan uses daily.amount * 1000 when amount is available",
            "market_state limit-lock detection prefers stk_limit and falls back conservatively",
        ],
    }
    (standardized_dir / "manifest.json").write_text(
        json.dumps(standardization_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return StandardizedTushareTables(
        prices=prices,
        asset_metadata=asset_metadata,
        trade_calendar=trade_calendar,
        market_state=market_state,
        daily_fundamentals=daily_fundamentals,
        financial_indicators=financial_indicators,
        snapshot_dir=standardized_dir,
        unavailable_raw_endpoints=unavailable,
    )


def standardize_prices(
    daily: pd.DataFrame,
    *,
    daily_basic: pd.DataFrame | None = None,
    adj_factor: pd.DataFrame | None = None,
) -> pd.DataFrame:
    required = {"ts_code", "trade_date", "close"}
    missing = required - set(daily.columns)
    if missing:
        raise ValueError(f"daily snapshot is missing required columns: {sorted(missing)}")

    out = daily.copy()
    out["asset"] = out["ts_code"].map(normalize_tushare_asset)
    out["date"] = normalize_trade_date(out["trade_date"], column_name="trade_date")
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    if out["close"].isna().any():
        raise ValueError("daily.close contains non-numeric values")

    for col in ("open", "high", "low", "pre_close", "pct_chg", "change"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if "vol" in out.columns:
        out["volume"] = pd.to_numeric(out["vol"], errors="coerce") * 100.0
    else:
        out["volume"] = np.nan
    if "amount" in out.columns:
        out["dollar_volume_yuan"] = pd.to_numeric(out["amount"], errors="coerce") * 1000.0
    else:
        out["dollar_volume_yuan"] = np.nan

    if daily_basic is not None and not daily_basic.empty:
        basic = standardize_daily_basic(daily_basic)[
            ["date", "asset", "pb", "pe_ttm", "total_mv_yuan", "circ_mv_yuan"]
        ].copy()
        out = out.merge(basic, on=["date", "asset"], how="left")
    else:
        out["pb"] = np.nan
        out["pe_ttm"] = np.nan
        out["total_mv_yuan"] = np.nan
        out["circ_mv_yuan"] = np.nan

    if adj_factor is not None and not adj_factor.empty:
        factor = adj_factor.copy()
        factor["asset"] = factor["ts_code"].map(normalize_tushare_asset)
        factor["date"] = normalize_trade_date(factor["trade_date"], column_name="trade_date")
        factor["adj_factor"] = pd.to_numeric(factor["adj_factor"], errors="coerce")
        out = out.merge(
            factor[["date", "asset", "adj_factor"]],
            on=["date", "asset"],
            how="left",
        )
    else:
        out["adj_factor"] = np.nan

    out["dollar_volume"] = out["dollar_volume_yuan"]
    keep = [
        "date",
        "asset",
        "close",
        "open",
        "high",
        "low",
        "pre_close",
        "pct_chg",
        "change",
        "volume",
        "dollar_volume",
        "dollar_volume_yuan",
        "pb",
        "pe_ttm",
        "total_mv_yuan",
        "circ_mv_yuan",
        "adj_factor",
    ]
    resolved = [col for col in keep if col in out.columns]
    return out[resolved].sort_values(["asset", "date"], kind="mergesort").reset_index(drop=True)


def standardize_asset_metadata(stock_basic: pd.DataFrame) -> pd.DataFrame:
    required = {"ts_code", "list_date"}
    missing = required - set(stock_basic.columns)
    if missing:
        raise ValueError(f"stock_basic snapshot is missing required columns: {sorted(missing)}")

    out = stock_basic.copy()
    out["asset"] = out["ts_code"].map(normalize_tushare_asset)
    out["listing_date"] = normalize_trade_date(out["list_date"], column_name="list_date")
    if "delist_date" in out.columns:
        delist = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns]")
        present = out["delist_date"].notna() & (out["delist_date"].astype(str).str.strip() != "")
        if present.any():
            delist.loc[present] = normalize_trade_date(
                out.loc[present, "delist_date"],
                column_name="delist_date",
            )
    else:
        delist = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns]")
    out["delist_date"] = delist
    name_series = (
        out["name"].astype(str)
        if "name" in out.columns
        else pd.Series("", index=out.index, dtype="object")
    )
    out["is_st"] = name_series.str.upper().str.contains("ST", regex=False)
    for col in ("industry", "market", "exchange", "name", "list_status", "is_hs"):
        if col not in out.columns:
            out[col] = None
    out = out[
        [
            "asset",
            "listing_date",
            "delist_date",
            "is_st",
            "industry",
            "market",
            "exchange",
            "name",
            "list_status",
            "is_hs",
        ]
    ].copy()
    out = out.sort_values(["asset", "listing_date"], kind="mergesort")
    out = out.drop_duplicates(subset=["asset"], keep="last").reset_index(drop=True)
    return out


def standardize_trade_calendar(trade_cal: pd.DataFrame) -> pd.DataFrame:
    required = {"exchange", "cal_date", "is_open"}
    missing = required - set(trade_cal.columns)
    if missing:
        raise ValueError(f"trade_cal snapshot is missing required columns: {sorted(missing)}")

    out = trade_cal.copy()
    out["date"] = normalize_trade_date(out["cal_date"], column_name="cal_date")
    out["is_open"] = _coerce_bool_series(out["is_open"], column_name="is_open")
    if "pretrade_date" in out.columns:
        out["pretrade_date"] = normalize_trade_date(
            out["pretrade_date"],
            column_name="pretrade_date",
        )
    else:
        out["pretrade_date"] = pd.NaT
    return out[["date", "exchange", "is_open", "pretrade_date"]].sort_values(
        ["exchange", "date"],
        kind="mergesort",
    ).reset_index(drop=True)


def standardize_daily_basic(daily_basic: pd.DataFrame) -> pd.DataFrame:
    required = {"ts_code", "trade_date"}
    missing = required - set(daily_basic.columns)
    if missing:
        raise ValueError(f"daily_basic snapshot is missing required columns: {sorted(missing)}")
    out = daily_basic.copy()
    out["asset"] = out["ts_code"].map(normalize_tushare_asset)
    out["date"] = normalize_trade_date(out["trade_date"], column_name="trade_date")
    for col in (
        "turnover_rate",
        "turnover_rate_f",
        "volume_ratio",
        "pe",
        "pe_ttm",
        "pb",
        "ps",
        "ps_ttm",
        "dv_ratio",
        "dv_ttm",
        "total_share",
        "float_share",
        "free_share",
        "total_mv",
        "circ_mv",
    ):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        else:
            out[col] = np.nan
    out["total_mv_yuan"] = out["total_mv"] * 10_000.0
    out["circ_mv_yuan"] = out["circ_mv"] * 10_000.0
    return out.sort_values(["asset", "date"], kind="mergesort").reset_index(drop=True)


def standardize_fina_indicator(fina_indicator: pd.DataFrame | None) -> pd.DataFrame:
    if fina_indicator is None or fina_indicator.empty:
        return pd.DataFrame(
            columns=[
                "asset",
                "announce_date",
                "report_period",
                "roe",
                "roe_dt",
                "roa",
                "grossprofit_margin",
                "netprofit_margin",
            ]
        )
    required = {"ts_code", "ann_date", "end_date"}
    missing = required - set(fina_indicator.columns)
    if missing:
        raise ValueError(f"fina_indicator snapshot is missing required columns: {sorted(missing)}")
    out = fina_indicator.copy()
    out["asset"] = out["ts_code"].map(normalize_tushare_asset)
    out["announce_date"] = normalize_trade_date(out["ann_date"], column_name="ann_date")
    out["report_period"] = normalize_trade_date(out["end_date"], column_name="end_date")
    for col in ("roe", "roe_dt", "roa", "grossprofit_margin", "netprofit_margin"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        else:
            out[col] = np.nan
    out = out[
        [
            "asset",
            "announce_date",
            "report_period",
            "roe",
            "roe_dt",
            "roa",
            "grossprofit_margin",
            "netprofit_margin",
        ]
    ].copy()
    return out.sort_values(
        ["asset", "announce_date", "report_period"],
        kind="mergesort",
    ).reset_index(drop=True)


def standardize_market_state(
    prices: pd.DataFrame,
    *,
    asset_metadata: pd.DataFrame,
    suspend: pd.DataFrame | None,
    stk_limit: pd.DataFrame | None,
) -> pd.DataFrame:
    out = prices[["date", "asset", "close", "high", "low", "volume"]].copy()
    out = out.merge(asset_metadata[["asset", "is_st"]], on="asset", how="left")
    out["is_st"] = out["is_st"].fillna(False).astype(bool)
    out["is_halted"] = False
    if suspend is not None and not suspend.empty:
        suspend_keys = suspend.copy()
        suspend_keys["asset"] = suspend_keys["ts_code"].map(normalize_tushare_asset)
        suspend_keys["date"] = normalize_trade_date(
            suspend_keys["trade_date"],
            column_name="trade_date",
        )
        suspend_keys["is_halted_raw"] = True
        suspend_frame = suspend_keys[["date", "asset", "is_halted_raw"]].drop_duplicates()
        out = out.merge(suspend_frame, on=["date", "asset"], how="left")
        out["is_halted"] = out["is_halted"] | out["is_halted_raw"].fillna(False).astype(bool)
        out = out.drop(columns=["is_halted_raw"])
    out["is_halted"] = out["is_halted"] | (
        out["volume"].fillna(0.0).abs() <= _EPSILON
    )

    out["is_limit_locked"] = False
    if stk_limit is not None and not stk_limit.empty:
        limits = stk_limit.copy()
        limits["asset"] = limits["ts_code"].map(normalize_tushare_asset)
        limits["date"] = normalize_trade_date(limits["trade_date"], column_name="trade_date")
        limits["up_limit"] = pd.to_numeric(limits["up_limit"], errors="coerce")
        limits["down_limit"] = pd.to_numeric(limits["down_limit"], errors="coerce")
        out = out.merge(
            limits[["date", "asset", "up_limit", "down_limit"]],
            on=["date", "asset"],
            how="left",
        )
        up_hit = (out["high"] - out["up_limit"]).abs() <= 1e-6
        down_hit = (out["low"] - out["down_limit"]).abs() <= 1e-6
        close_hit = (
            (out["close"] - out["up_limit"]).abs() <= 1e-6
        ) | ((out["close"] - out["down_limit"]).abs() <= 1e-6)
        out["is_limit_locked"] = (up_hit | down_hit) & close_hit
        out = out.drop(columns=["up_limit", "down_limit"])
    else:
        out["is_limit_locked"] = False

    return out[["date", "asset", "is_halted", "is_limit_locked", "is_st"]].sort_values(
        ["date", "asset"],
        kind="mergesort",
    ).reset_index(drop=True)


def _load_optional_snapshot(snapshot_dir: Path, *, endpoint: str) -> pd.DataFrame | None:
    path = snapshot_dir / f"{endpoint}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def _load_financial_indicator_snapshot(snapshot_dir: Path) -> pd.DataFrame | None:
    frames: list[pd.DataFrame] = []
    fallback = _load_optional_snapshot(snapshot_dir, endpoint="fina_indicator")
    if fallback is not None:
        frames.append(fallback)
    vip = _load_optional_snapshot(snapshot_dir, endpoint="fina_indicator_vip")
    if vip is not None:
        frames.append(vip)
    if not frames:
        return None
    if len(frames) == 1:
        return frames[0]
    return pd.concat(frames, ignore_index=True).drop_duplicates().reset_index(drop=True)


def _load_optional_manifest(snapshot_dir: Path) -> dict[str, object]:
    path = snapshot_dir / "manifest.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    return payload


def _write_df(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, lineterminator="\n")


def _coerce_bool_series(series: pd.Series, *, column_name: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    lowered = series.astype(str).str.strip().str.lower()
    mapping = {"1": True, "0": False, "true": True, "false": False}
    if not lowered.isin(mapping).all():
        raise ValueError(f"{column_name} contains non-boolean values")
    return lowered.map(mapping).astype(bool)
