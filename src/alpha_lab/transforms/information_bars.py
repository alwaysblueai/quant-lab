from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_lab.exceptions import AlphaLabDataError


def dollar_bars(ohlcv_df: pd.DataFrame, target_dollar_volume: float) -> pd.DataFrame:
    """Build dollar bars from intraday OHLCV rows."""
    _require_columns(
        ohlcv_df,
        ("datetime", "asset", "open", "high", "low", "close", "volume", "amount"),
        "ohlcv_df",
    )
    return _build_information_bars(
        ohlcv_df,
        target=float(target_dollar_volume),
        target_mode="amount",
    )


def volume_bars(ohlcv_df: pd.DataFrame, target_volume: float) -> pd.DataFrame:
    """Build volume bars from intraday OHLCV rows."""
    _require_columns(
        ohlcv_df,
        ("datetime", "asset", "open", "high", "low", "close", "volume", "amount"),
        "ohlcv_df",
    )
    return _build_information_bars(
        ohlcv_df,
        target=float(target_volume),
        target_mode="volume",
    )


def tick_bars(tick_df: pd.DataFrame, target_ticks: int) -> pd.DataFrame:
    """Build tick bars from tick-level rows."""
    _require_columns(tick_df, ("datetime", "asset"), "tick_df")
    if "price" not in tick_df.columns and "close" not in tick_df.columns:
        raise AlphaLabDataError("tick_df must contain either 'price' or 'close'")

    price_col = "price" if "price" in tick_df.columns else "close"
    default_volume = pd.Series(1.0, index=tick_df.index, dtype=float)
    normalized = pd.DataFrame(
        {
            "datetime": tick_df["datetime"],
            "asset": tick_df["asset"],
            "open": pd.to_numeric(tick_df[price_col], errors="coerce"),
            "high": pd.to_numeric(tick_df[price_col], errors="coerce"),
            "low": pd.to_numeric(tick_df[price_col], errors="coerce"),
            "close": pd.to_numeric(tick_df[price_col], errors="coerce"),
            "volume": pd.to_numeric(tick_df.get("volume", default_volume), errors="coerce").fillna(
                1.0
            ),
            "amount": pd.to_numeric(tick_df.get("amount"), errors="coerce"),
        }
    )
    normalized["amount"] = normalized["amount"].fillna(normalized["volume"] * normalized["close"])
    return _build_information_bars(normalized, target=float(target_ticks), target_mode="ticks")


def _build_information_bars(
    frame: pd.DataFrame,
    *,
    target: float,
    target_mode: str,
) -> pd.DataFrame:
    if target <= 0:
        raise ValueError("target must be > 0")
    if target_mode not in {"amount", "volume", "ticks"}:
        raise ValueError("target_mode must be one of: amount, volume, ticks")

    table = frame.copy()
    table["datetime"] = pd.to_datetime(table["datetime"], errors="coerce")
    for col in ("open", "high", "low", "close", "volume", "amount"):
        table[col] = pd.to_numeric(table[col], errors="coerce")
    table = table.dropna(subset=["datetime", "asset", "open", "high", "low", "close"])
    table = table.sort_values(["asset", "datetime"], kind="mergesort").reset_index(drop=True)

    rows: list[dict[str, object]] = []
    for asset, idx in table.groupby("asset", sort=False).groups.items():
        block = table.loc[idx].reset_index(drop=True)
        bar_start: pd.Timestamp | None = None
        bar_open = np.nan
        bar_high = -np.inf
        bar_low = np.inf
        bar_close = np.nan
        bar_volume = 0.0
        bar_amount = 0.0
        bar_ticks = 0
        accum_target = 0.0

        for row in block.itertuples(index=False):
            dt = pd.Timestamp(row.datetime)
            open_ = float(row.open)
            high_ = float(row.high)
            low_ = float(row.low)
            close_ = float(row.close)
            vol_ = float(row.volume) if np.isfinite(row.volume) else 0.0
            amt_ = float(row.amount) if np.isfinite(row.amount) else vol_ * close_
            step = 1.0 if target_mode == "ticks" else (amt_ if target_mode == "amount" else vol_)

            if bar_start is None:
                bar_start = dt
                bar_open = open_
                bar_high = high_
                bar_low = low_
                bar_close = close_
                bar_volume = vol_
                bar_amount = amt_
                bar_ticks = 1
                accum_target = float(step)
            else:
                bar_high = max(bar_high, high_)
                bar_low = min(bar_low, low_)
                bar_close = close_
                bar_volume += vol_
                bar_amount += amt_
                bar_ticks += 1
                accum_target += float(step)

            if accum_target >= target:
                rows.append(
                    {
                        "bar_start": bar_start,
                        "bar_end": dt,
                        "asset": asset,
                        "open": bar_open,
                        "high": bar_high,
                        "low": bar_low,
                        "close": bar_close,
                        "volume": bar_volume,
                        "amount": bar_amount,
                        "n_ticks": bar_ticks,
                    }
                )
                bar_start = None
                bar_open = np.nan
                bar_high = -np.inf
                bar_low = np.inf
                bar_close = np.nan
                bar_volume = 0.0
                bar_amount = 0.0
                bar_ticks = 0
                accum_target = 0.0

        if bar_start is not None:
            rows.append(
                {
                    "bar_start": bar_start,
                    "bar_end": pd.Timestamp(block.iloc[-1]["datetime"]),
                    "asset": asset,
                    "open": bar_open,
                    "high": bar_high,
                    "low": bar_low,
                    "close": bar_close,
                    "volume": bar_volume,
                    "amount": bar_amount,
                    "n_ticks": bar_ticks,
                }
            )

    return pd.DataFrame(
        rows,
        columns=[
            "bar_start",
            "bar_end",
            "asset",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "amount",
            "n_ticks",
        ],
    )


def _require_columns(frame: pd.DataFrame, cols: tuple[str, ...], name: str) -> None:
    missing = set(cols) - set(frame.columns)
    if missing:
        raise AlphaLabDataError(f"{name} missing required columns: {sorted(missing)}")
