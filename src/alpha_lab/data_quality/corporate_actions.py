"""Corporate action detection and adjustment for equity price data.

Provides split detection (suspected unadjusted data) and back-adjustment
using exchange-provided adjustment factors.
"""

from __future__ import annotations

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

    # For each dividend event, compute the adjustment ratio
    # On ex-date: adjusted_price = price * (1 - dividend / prev_close)
    # All prior prices are scaled by this ratio
    for _, row in dividend_df.iterrows():
        asset = row["asset"]
        ex_date = row["date"]
        div = row["dividend_per_share"]

        mask = (df["asset"] == asset) & (df["date"] < ex_date)
        # Get the closing price on the day before ex-date
        pre_ex = df[(df["asset"] == asset) & (df["date"] < ex_date)]
        if pre_ex.empty:
            continue
        prev_close = pre_ex.sort_values("date").iloc[-1]["close"]
        if prev_close <= 0:
            continue
        ratio = 1.0 - div / prev_close
        if ratio <= 0:
            continue
        df.loc[mask, "close"] = df.loc[mask, "close"] * ratio

    return df
