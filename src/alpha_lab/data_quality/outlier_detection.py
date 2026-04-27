from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_lab.exceptions import AlphaLabDataError


def detect_price_jumps(
    prices_df: pd.DataFrame,
    threshold: float = 0.11,
) -> pd.DataFrame:
    """Flag suspicious daily price jumps.

    Parameters
    ----------
    prices_df:
        Long-form price panel with ``[date, asset, close]``.
    threshold:
        Absolute daily pct-change threshold for anomaly flagging.
    """
    if threshold <= 0:
        raise ValueError("threshold must be > 0")
    _require_columns(prices_df, ("date", "asset", "close"), "prices_df")

    out = prices_df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out = out.sort_values(["asset", "date"], kind="mergesort").reset_index(drop=True)

    out["pct_change"] = out.groupby("asset", sort=False)["close"].pct_change()
    out["is_price_jump"] = out["pct_change"].abs() > float(threshold)
    out.loc[out["pct_change"].isna(), "is_price_jump"] = False
    return out


def filter_zero_volume(
    prices_df: pd.DataFrame,
    action: str = "flag",
) -> pd.DataFrame:
    """Flag or drop zero-volume rows.

    ``action="flag"`` adds ``is_suspended``.
    ``action="drop"`` removes rows where volume <= 0.
    """
    _require_columns(prices_df, ("date", "asset", "volume"), "prices_df")
    mode = action.strip().lower()
    if mode not in {"flag", "drop"}:
        raise ValueError("action must be 'flag' or 'drop'")

    out = prices_df.copy()
    out["volume"] = pd.to_numeric(out["volume"], errors="coerce")
    suspended = (out["volume"] <= 0.0) | out["volume"].isna()

    if mode == "drop":
        return out.loc[~suspended].reset_index(drop=True)

    out["is_suspended"] = suspended
    return out


def detect_stale_prices(
    prices_df: pd.DataFrame,
    max_identical_days: int = 5,
) -> pd.DataFrame:
    """Flag assets whose close is unchanged for ``max_identical_days``+ runs."""
    if max_identical_days < 2:
        raise ValueError("max_identical_days must be >= 2")
    _require_columns(prices_df, ("date", "asset", "close"), "prices_df")

    out = prices_df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out = out.sort_values(["asset", "date"], kind="mergesort").reset_index(drop=True)

    stale_flags = np.zeros(len(out), dtype=bool)
    for _, idx in out.groupby("asset", sort=False).groups.items():
        block = out.loc[idx].copy()
        run_id = block["close"].ne(block["close"].shift(1)).cumsum()
        run_len = block.groupby(run_id, sort=False)["close"].transform("size")
        stale_flags[block.index.to_numpy()] = (
            (run_len >= int(max_identical_days)) & block["close"].notna()
        ).to_numpy()

    out["is_stale_price"] = stale_flags
    return out


def _require_columns(frame: pd.DataFrame, cols: tuple[str, ...], name: str) -> None:
    missing = set(cols) - set(frame.columns)
    if missing:
        raise AlphaLabDataError(f"{name} missing required columns: {sorted(missing)}")
