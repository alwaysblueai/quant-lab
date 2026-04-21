"""Corporate action detection and adjustment for equity price data.

Provides split detection (suspected unadjusted data) and back-adjustment
using exchange-provided adjustment factors.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def detect_unadjusted_splits(
    prices_df: pd.DataFrame,
    threshold: float = 0.45,
) -> pd.DataFrame:
    """Detect suspected unadjusted stock splits in price data.

    Flags rows where the absolute day-over-day price change exceeds
    ``threshold`` (e.g. 0.45 = 45%), which is well beyond normal A-share
    daily limits (10%/20%) and likely indicates an unadjusted split.

    Parameters
    ----------
    prices_df:
        Long-form price panel with ``[date, asset, close]``.
    threshold:
        Absolute fractional price change threshold for flagging.

    Returns
    -------
    pd.DataFrame
        Copy of input with added columns: ``pct_change``, ``suspected_split``.
        Only rows with a valid previous close are included.
    """
    df = prices_df.copy()
    df = df.sort_values(["asset", "date"]).reset_index(drop=True)
    df["prev_close"] = df.groupby("asset")["close"].shift(1)
    df["pct_change"] = (df["close"] - df["prev_close"]) / df["prev_close"]
    df["suspected_split"] = df["pct_change"].abs() > threshold
    # First row per asset has no previous close — not a split
    df.loc[df["prev_close"].isna(), "suspected_split"] = False
    return df.drop(columns=["prev_close"])


def adjust_for_splits(
    prices_df: pd.DataFrame,
    adj_factor_df: pd.DataFrame,
) -> pd.DataFrame:
    """Back-adjust prices using adjustment factors.

    Applies back-adjustment so that prices are comparable across split
    events: ``adjusted_close = close * adj_factor / latest_adj_factor``.

    Parameters
    ----------
    prices_df:
        Long-form price panel with at least ``[date, asset, close]``.
    adj_factor_df:
        Adjustment factors with ``[date, asset, adj_factor]``.
        The latest adj_factor per asset is used as the reference.

    Returns
    -------
    pd.DataFrame
        Copy of ``prices_df`` with ``close`` replaced by adjusted values.
        All other columns are preserved.
    """
    df = prices_df.copy()
    adj = adj_factor_df[["date", "asset", "adj_factor"]].copy()

    # Find the latest adj_factor per asset (the reference point)
    latest = adj.sort_values("date").groupby("asset").last().reset_index()
    latest = latest.rename(columns={"adj_factor": "latest_adj_factor"})
    latest = latest[["asset", "latest_adj_factor"]]

    # Merge adjustment factors
    df = df.merge(adj, on=["date", "asset"], how="left")
    df = df.merge(latest, on="asset", how="left")

    # Apply adjustment where both factors are available
    has_adj = df["adj_factor"].notna() & df["latest_adj_factor"].notna()
    df.loc[has_adj, "close"] = (
        df.loc[has_adj, "close"]
        * df.loc[has_adj, "adj_factor"]
        / df.loc[has_adj, "latest_adj_factor"]
    )

    return df.drop(columns=["adj_factor", "latest_adj_factor"])


def adjust_for_dividends(
    prices_df: pd.DataFrame,
    dividend_df: pd.DataFrame,
    method: str = "back_adjust",
) -> pd.DataFrame:
    """Adjust prices for dividend events.

    Parameters
    ----------
    prices_df:
        Long-form price panel with ``[date, asset, close]``.
    dividend_df:
        Dividend events with ``[date, asset, dividend_per_share]``.
        ``date`` is the ex-dividend date.
    method:
        Adjustment method. Only ``"back_adjust"`` is supported.

    Returns
    -------
    pd.DataFrame
        Copy of ``prices_df`` with ``close`` adjusted for dividends.
    """
    if method != "back_adjust":
        raise ValueError(f"unsupported dividend adjustment method: {method!r}")

    df = prices_df.copy()
    df = df.sort_values(["asset", "date"]).reset_index(drop=True)
    if dividend_df.empty:
        return df

    @dataclass
    class _AssetState:
        start: int
        end: int
        dates: np.ndarray
        close_values: np.ndarray
        diff_log: np.ndarray
        fenwick_tree: np.ndarray

        @property
        def size(self) -> int:
            return self.end - self.start

    def _fenwick_add(tree: np.ndarray, idx: int, delta: float) -> None:
        i = idx + 1
        n = tree.size - 1
        while i <= n:
            tree[i] += delta
            i += i & -i

    def _fenwick_point_query(tree: np.ndarray, idx: int) -> float:
        i = idx + 1
        total = 0.0
        while i > 0:
            total += float(tree[i])
            i -= i & -i
        return total

    close_values = pd.to_numeric(df["close"], errors="coerce").to_numpy(copy=True, dtype=float)
    date_values = pd.to_datetime(df["date"], errors="coerce").to_numpy(dtype="datetime64[ns]")

    asset_states: dict[object, _AssetState] = {}
    for asset, group in df.groupby("asset", sort=False, observed=True):
        idx = group.index.to_numpy(dtype=int)
        if idx.size == 0:
            continue
        start = int(idx[0])
        end = int(idx[-1]) + 1
        size = end - start
        asset_states[asset] = _AssetState(
            start=start,
            end=end,
            dates=date_values[start:end],
            close_values=close_values[start:end],
            diff_log=np.zeros(size + 1, dtype=float),
            fenwick_tree=np.zeros(size + 1, dtype=float),
        )

    events = dividend_df[["asset", "date", "dividend_per_share"]]
    for asset, ex_date_raw, div_raw in events.itertuples(index=False, name=None):
        state = asset_states.get(asset)
        if state is None:
            continue

        ex_date = pd.to_datetime(ex_date_raw, errors="coerce")
        if pd.isna(ex_date):
            continue
        div = pd.to_numeric(div_raw, errors="coerce")
        if not pd.notna(div):
            continue
        div_value = float(div)

        ex_np = np.datetime64(pd.Timestamp(ex_date), "ns")
        insert_pos = int(np.searchsorted(state.dates, ex_np, side="left"))
        if insert_pos <= 0:
            continue

        prev_idx = insert_pos - 1
        prev_scale_log = _fenwick_point_query(state.fenwick_tree, prev_idx)
        prev_close = float(state.close_values[prev_idx] * np.exp(prev_scale_log))
        if not np.isfinite(prev_close) or prev_close <= 0.0:
            continue

        ratio = 1.0 - div_value / prev_close
        if not np.isfinite(ratio) or ratio <= 0.0:
            continue

        log_ratio = float(np.log(ratio))
        state.diff_log[0] += log_ratio
        state.diff_log[insert_pos] -= log_ratio

        _fenwick_add(state.fenwick_tree, 0, log_ratio)
        if insert_pos < state.size:
            _fenwick_add(state.fenwick_tree, insert_pos, -log_ratio)

    for state in asset_states.values():
        log_scale = np.cumsum(state.diff_log[:-1], dtype=float)
        state.close_values[:] = state.close_values * np.exp(log_scale)

    df["close"] = close_values
    return df
