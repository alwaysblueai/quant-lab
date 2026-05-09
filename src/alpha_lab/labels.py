from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from alpha_lab.exceptions import AlphaLabConfigError, AlphaLabDataError
from alpha_lab.frame_utils import readonly_shallow_copy
from alpha_lab.interfaces import FACTOR_OUTPUT_COLUMNS
from alpha_lab.sorted_panel import ensure_sorted

_REQUIRED_COLS = {"date", "asset", "close"}

ExecutionPriceMode = Literal["close", "next_open", "vwap"]


class LabelCache:
    """Reuse normalized price state across repeated forward-return horizons."""

    def __init__(
        self,
        prices: pd.DataFrame,
        *,
        execution_price_mode: ExecutionPriceMode = "close",
    ) -> None:
        self._mode = _normalize_execution_price_mode(execution_price_mode)
        self._prices = _prepare_forward_return_prices(prices, mode=self._mode)
        self._cache: dict[int, pd.DataFrame] = {}

    def forward_return(self, horizon: int, *, copy: bool = True) -> pd.DataFrame:
        horizon_int = int(horizon)
        if horizon_int not in self._cache:
            self._cache[horizon_int] = _forward_return_from_prepared(
                self._prices,
                horizon=horizon_int,
                mode=self._mode,
            )
        cached = self._cache[horizon_int]
        if copy:
            return cached.copy()
        return readonly_shallow_copy(cached)

    def forward_returns(self, horizons: tuple[int, ...] | list[int]) -> dict[int, pd.DataFrame]:
        return {int(horizon): self.forward_return(int(horizon)) for horizon in horizons}

    def forward_returns_wide(
        self,
        horizons: tuple[int, ...] | list[int],
    ) -> dict[int, np.ndarray]:
        import numpy as np

        out: dict[int, np.ndarray] = {}
        for horizon in horizons:
            labels = self.forward_return(int(horizon))
            if labels.empty:
                out[int(horizon)] = np.empty((0, 0), dtype=float)
                continue
            wide = labels.pivot(index="date", columns="asset", values="value").sort_index()
            out[int(horizon)] = wide.to_numpy(dtype=float, copy=True)
        return out


def forward_return(
    df: pd.DataFrame,
    *,
    horizon: int = 1,
    execution_price_mode: ExecutionPriceMode = "close",
) -> pd.DataFrame:
    """Compute forward returns using each asset's own ordered history.

    The label is the return of a position opened *after* the signal date and
    closed ``horizon`` bars later.  The ``execution_price_mode`` selects the
    realistic execution prices:

    - ``"close"`` (default, legacy behaviour): ``close[t + horizon] / close[t] - 1``.
      Assumes instantaneous execution at the signal-day close, which is
      optimistic for intraday strategies.
    - ``"next_open"``: ``close[t + horizon] / open[t + 1] - 1``.  Models a
      trader who sees the signal at close of day ``t`` and executes a market
      order at the next open.  Requires an ``open`` column.
    - ``"vwap"``: ``close[t + horizon] / vwap[t + 1] - 1`` when a ``vwap``
      column is available.  Models average-price execution on the day
      following the signal.  Requires a ``vwap`` column.

    In every mode the label at date ``t`` depends only on prices strictly
    after ``t`` (for non-close modes, on day ``t+1`` onward).

    Parameters
    ----------
    df:
        Long-form price table.  Must include ``[date, asset, close]``; must
        additionally include ``open`` when ``execution_price_mode="next_open"``
        or ``vwap`` when ``execution_price_mode="vwap"``.  Rows need not be
        sorted.  Duplicate ``(date, asset)`` pairs are rejected.  Non-positive
        prices (<= 0) are treated as missing.
    horizon:
        Number of future per-asset rows used in the forward return.  Must be
        a positive integer.  Interpreted as "close-after-horizon" in every
        mode: the position closes at ``close[t + horizon]``.
    execution_price_mode:
        Entry-price convention.  See above.

    Returns
    -------
    pd.DataFrame
        Canonical long-form output with columns ``[date, asset, factor, value]``.
        ``factor`` is ``forward_return_{horizon}`` for close-to-close mode and
        ``forward_return_{horizon}_{mode}`` otherwise so downstream consumers
        can distinguish the convention.
    """
    if df.empty:
        return pd.DataFrame(columns=FACTOR_OUTPUT_COLUMNS)

    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        raise AlphaLabDataError(f"Input DataFrame is missing required columns: {missing}")
    if horizon <= 0:
        raise AlphaLabConfigError("'horizon' must be a positive integer")

    mode = _normalize_execution_price_mode(execution_price_mode)
    prepared = _prepare_forward_return_prices(df, mode=mode)
    return _forward_return_from_prepared(prepared, horizon=int(horizon), mode=mode)


def _normalize_execution_price_mode(execution_price_mode: str) -> ExecutionPriceMode:
    mode = execution_price_mode.strip().lower()
    if mode not in ("close", "next_open", "vwap"):
        raise AlphaLabConfigError(
            f"execution_price_mode must be 'close', 'next_open', or 'vwap'; got {mode!r}"
        )
    return mode  # type: ignore[return-value]


def _prepare_forward_return_prices(
    df: pd.DataFrame,
    *,
    mode: ExecutionPriceMode,
) -> pd.DataFrame:
    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        raise AlphaLabDataError(f"Input DataFrame is missing required columns: {missing}")

    if mode == "next_open" and "open" not in df.columns:
        raise AlphaLabDataError(
            "execution_price_mode='next_open' requires an 'open' column in the price frame"
        )
    if mode == "vwap" and "vwap" not in df.columns:
        raise AlphaLabDataError(
            "execution_price_mode='vwap' requires a 'vwap' column in the price frame"
        )

    if df["date"].isna().any():
        raise AlphaLabDataError("Input 'date' column contains NaN/NaT values.")

    if df["asset"].isna().any():
        raise AlphaLabDataError("Input 'asset' column contains NaN values.")

    dupes = df.duplicated(subset=["date", "asset"])
    if dupes.any():
        raise AlphaLabDataError(
            f"Duplicate (date, asset) pairs found:\n{df[dupes][['date', 'asset']]}"
        )

    df_copy = df.copy()
    df_copy["date"] = pd.to_datetime(df_copy["date"])
    return ensure_sorted(df_copy, by=("asset", "date"))


def _forward_return_from_prepared(
    df_copy: pd.DataFrame,
    *,
    horizon: int,
    mode: ExecutionPriceMode,
) -> pd.DataFrame:
    if horizon <= 0:
        raise AlphaLabConfigError("'horizon' must be a positive integer")
    if df_copy.empty:
        return pd.DataFrame(columns=FACTOR_OUTPUT_COLUMNS)
    close = df_copy["close"].where(df_copy["close"] > 0)
    grouped = close.groupby(df_copy["asset"], sort=False)
    exit_price = grouped.shift(-horizon)

    if mode == "close":
        entry_price = close
        factor_name = f"forward_return_{horizon}"
    elif mode == "next_open":
        open_col = df_copy["open"].where(df_copy["open"] > 0)
        entry_price = open_col.groupby(df_copy["asset"], sort=False).shift(-1)
        factor_name = f"forward_return_{horizon}_next_open"
    else:  # vwap
        vwap_col = df_copy["vwap"].where(df_copy["vwap"] > 0)
        entry_price = vwap_col.groupby(df_copy["asset"], sort=False).shift(-1)
        factor_name = f"forward_return_{horizon}_vwap"

    value = exit_price.div(entry_price).sub(1.0)

    result = df_copy[["date", "asset"]].copy()
    result["factor"] = factor_name
    result["value"] = value
    result = result.loc[:, FACTOR_OUTPUT_COLUMNS].reset_index(drop=True)

    return result
