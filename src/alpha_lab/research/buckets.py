from __future__ import annotations

import pandas as pd

from alpha_lab.bucket_builders import build_numeric_quantile_bucket
from alpha_lab.frame_utils import require_columns


def build_past_ret_lookback_bucket(
    prices: pd.DataFrame,
    *,
    lookback: int,
    skip_recent: int = 0,
    n_buckets: int = 3,
    bucket_col: str = "bucket",
    bucket_labels: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Build experimental past-return buckets with arbitrary lookback and skip."""
    if lookback < 1:
        raise ValueError("lookback must be >= 1")
    if skip_recent < 0:
        raise ValueError("skip_recent must be >= 0")

    require_columns(prices, ("date", "asset", "close"), "prices")
    values = prices.loc[:, ["date", "asset", "close"]].copy()
    values["date"] = pd.to_datetime(values["date"], errors="coerce")
    values["close"] = pd.to_numeric(values["close"], errors="coerce")
    values = values.sort_values(["asset", "date"], kind="mergesort").reset_index(drop=True)
    grouped_close = values.groupby("asset", sort=False)["close"]
    anchor_close = grouped_close.shift(int(skip_recent)) if skip_recent else values["close"]
    values["past_ret"] = anchor_close.groupby(values["asset"], sort=False).pct_change(
        int(lookback)
    )
    return build_numeric_quantile_bucket(
        values,
        value_col="past_ret",
        n_buckets=n_buckets,
        bucket_col=bucket_col,
        bucket_labels=bucket_labels,
    )


def build_factor_self_bucket(
    factor_df: pd.DataFrame,
    *,
    n_buckets: int = 10,
    bucket_col: str = "bucket",
    bucket_labels: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Build experimental buckets by raw factor value rather than magnitude."""
    require_columns(factor_df, ("date", "asset", "value"), "factor_df")
    values = factor_df.loc[:, ["date", "asset", "value"]].copy()
    values = values.rename(columns={"value": "factor_value"})
    return build_numeric_quantile_bucket(
        values,
        value_col="factor_value",
        n_buckets=n_buckets,
        bucket_col=bucket_col,
        bucket_labels=bucket_labels,
    )


def build_two_dim_bucket(
    left_bucket: pd.DataFrame,
    right_bucket: pd.DataFrame,
    *,
    left_name: str,
    right_name: str,
    left_col: str = "bucket",
    right_col: str = "bucket",
    bucket_col: str = "bucket",
) -> pd.DataFrame:
    """Combine two asset-level bucket frames into a single crossed bucket label."""
    require_columns(left_bucket, ("date", "asset", left_col), "left_bucket")
    require_columns(right_bucket, ("date", "asset", right_col), "right_bucket")
    left = left_bucket.loc[:, ["date", "asset", left_col]].rename(columns={left_col: "_left"})
    right = right_bucket.loc[:, ["date", "asset", right_col]].rename(columns={right_col: "_right"})
    merged = left.merge(right, on=["date", "asset"], how="inner", validate="one_to_one")
    if merged.empty:
        return pd.DataFrame(columns=["date", "asset", bucket_col])
    merged[bucket_col] = (
        left_name
        + "="
        + merged["_left"].astype(str)
        + "|"
        + right_name
        + "="
        + merged["_right"].astype(str)
    )
    return merged.loc[:, ["date", "asset", bucket_col]].reset_index(drop=True)


