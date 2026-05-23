from __future__ import annotations

import pandas as pd

from alpha_lab.exceptions import AlphaLabDataError
from alpha_lab.regime import classify_market_regimes

DEFAULT_TIER1_TRAILING_RETURN_HORIZON = 20

SIZE_SOURCE_COLUMNS: tuple[str, ...] = (
    "market_cap_log",
    "market_cap",
    "circ_mv",
    "total_mv",
)
LIQUIDITY_SOURCE_COLUMNS: tuple[str, ...] = (
    "dollar_volume",
    "amount",
    "amt",
    "turnover",
    "adv",
)


def build_numeric_quantile_bucket(
    frame: pd.DataFrame,
    *,
    value_col: str,
    n_buckets: int = 3,
    bucket_col: str = "bucket",
    bucket_labels: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Build per-date cross-sectional quantile buckets."""
    if n_buckets < 2:
        raise ValueError("n_buckets must be >= 2")

    labels = (
        tuple(f"Q{i + 1}" for i in range(int(n_buckets)))
        if bucket_labels is None
        else tuple(str(label) for label in bucket_labels)
    )
    if len(labels) != int(n_buckets):
        raise ValueError("bucket_labels length must equal n_buckets")

    _require_columns(frame, ("date", "asset", value_col), "frame")
    out = frame.loc[:, ["date", "asset", value_col]].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out[value_col] = pd.to_numeric(out[value_col], errors="coerce")
    out = out.dropna(subset=["date", "asset", value_col]).copy()
    if out.empty:
        return pd.DataFrame(columns=["date", "asset", bucket_col])

    out[bucket_col] = out.groupby("date", sort=False)[value_col].transform(
        lambda series: _assign_quantile_labels(series, labels=labels)
    )
    return out.loc[:, ["date", "asset", bucket_col]].dropna(subset=[bucket_col]).reset_index(
        drop=True
    )


def build_factor_magnitude_bucket(
    factor_df: pd.DataFrame,
    *,
    n_buckets: int = 5,
    bucket_col: str = "bucket",
) -> pd.DataFrame:
    """Build per-date buckets by absolute factor value."""
    _require_columns(factor_df, ("date", "asset", "value"), "factor_df")
    values = factor_df.loc[:, ["date", "asset", "value"]].copy()
    values["factor_abs"] = pd.to_numeric(values["value"], errors="coerce").abs()
    return build_numeric_quantile_bucket(
        values,
        value_col="factor_abs",
        n_buckets=n_buckets,
        bucket_col=bucket_col,
    )


def build_size_bucket(
    frame: pd.DataFrame,
    *,
    source_col: str | None = None,
    n_buckets: int = 3,
    bucket_col: str = "bucket",
    bucket_labels: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Build stable Tier-1 size buckets from the first available size column."""
    resolved_col = source_col or _resolve_source_column(frame, SIZE_SOURCE_COLUMNS, "size")
    return build_numeric_quantile_bucket(
        frame,
        value_col=resolved_col,
        n_buckets=n_buckets,
        bucket_col=bucket_col,
        bucket_labels=bucket_labels,
    )


def build_liquidity_bucket(
    frame: pd.DataFrame,
    *,
    source_col: str | None = None,
    n_buckets: int = 3,
    bucket_col: str = "bucket",
    bucket_labels: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Build stable Tier-1 liquidity buckets from the first available liquidity column."""
    resolved_col = source_col or _resolve_source_column(
        frame,
        LIQUIDITY_SOURCE_COLUMNS,
        "liquidity",
    )
    return build_numeric_quantile_bucket(
        frame,
        value_col=resolved_col,
        n_buckets=n_buckets,
        bucket_col=bucket_col,
        bucket_labels=bucket_labels,
    )


def build_trailing_return_bucket(
    prices: pd.DataFrame,
    *,
    horizon: int = DEFAULT_TIER1_TRAILING_RETURN_HORIZON,
    source_col: str | None = None,
    n_buckets: int = 3,
    bucket_col: str = "bucket",
    bucket_labels: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Build trailing-return buckets, reusing ret_{horizon}d when present."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    resolved_col = source_col or f"ret_{int(horizon)}d"
    if resolved_col not in prices.columns:
        _require_columns(prices, ("date", "asset", "close"), "prices")
        values = prices.loc[:, ["date", "asset", "close"]].copy()
        values["date"] = pd.to_datetime(values["date"], errors="coerce")
        values = values.sort_values(["asset", "date"], kind="mergesort").reset_index(drop=True)
        values["close"] = pd.to_numeric(values["close"], errors="coerce")
        values[resolved_col] = values.groupby("asset", sort=False)["close"].pct_change(
            int(horizon)
        )
    else:
        values = prices.loc[:, ["date", "asset", resolved_col]].copy()

    return build_numeric_quantile_bucket(
        values,
        value_col=resolved_col,
        n_buckets=n_buckets,
        bucket_col=bucket_col,
        bucket_labels=bucket_labels,
    )


def build_regime_bucket(
    *,
    ic_df: pd.DataFrame,
    long_short_df: pd.DataFrame,
    prices: pd.DataFrame,
    regime_col: str = "direction_regime",
    bucket_col: str = "bucket",
    vol_window: int = 20,
) -> pd.DataFrame:
    """Build date-level market regime buckets from the stable regime classifier."""
    regime_df = classify_market_regimes(
        ic_df=ic_df,
        long_short_df=long_short_df,
        prices=prices,
        vol_window=vol_window,
    )
    if regime_df.empty:
        return pd.DataFrame(columns=["date", bucket_col])
    _require_columns(regime_df, ("date", regime_col), "regime_df")
    out = regime_df.loc[:, ["date", regime_col]].rename(columns={regime_col: bucket_col})
    return out.dropna(subset=["date", bucket_col]).reset_index(drop=True)


def _assign_quantile_labels(
    series: pd.Series,
    *,
    labels: list[str] | tuple[str, ...],
) -> pd.Series:
    out = pd.Series(pd.NA, index=series.index, dtype="object")
    valid = series.notna()
    if int(valid.sum()) < len(labels):
        return out
    ranks = series.loc[valid].rank(method="first")
    out.loc[valid] = pd.qcut(ranks, q=len(labels), labels=labels).astype(str)
    return out


def _resolve_source_column(
    frame: pd.DataFrame,
    candidates: tuple[str, ...],
    label: str,
) -> str:
    for col in candidates:
        if col in frame.columns:
            return col
    raise AlphaLabDataError(
        f"{label} bucket source missing; expected one of: {list(candidates)}"
    )


def _require_columns(frame: pd.DataFrame, cols: tuple[str, ...], name: str) -> None:
    missing = set(cols) - set(frame.columns)
    if missing:
        raise AlphaLabDataError(f"{name} missing required columns: {sorted(missing)}")
