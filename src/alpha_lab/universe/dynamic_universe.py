from __future__ import annotations

import pandas as pd

from alpha_lab.exceptions import AlphaLabDataError
from alpha_lab.frame_utils import require_columns


def build_liquidity_universe(
    prices_df: pd.DataFrame,
    min_adv_pct: float = 20,
    lookback: int = 60,
) -> pd.DataFrame:
    """Build a dynamic top-liquidity universe by trailing ADV."""
    if min_adv_pct <= 0 or min_adv_pct > 100:
        raise ValueError("min_adv_pct must be in (0, 100]")
    if lookback < 2:
        raise ValueError("lookback must be >= 2")
    require_columns(prices_df, ("date", "asset"), "prices_df")

    panel = prices_df.copy()
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce")
    if "amount" in panel.columns:
        panel["amount"] = pd.to_numeric(panel["amount"], errors="coerce")
    else:
        require_columns(panel, ("volume", "close"), "prices_df")
        panel["amount"] = pd.to_numeric(panel["volume"], errors="coerce") * pd.to_numeric(
            panel["close"],
            errors="coerce",
        )

    panel = panel.dropna(subset=["date", "asset", "amount"])
    panel = panel.sort_values(["asset", "date"], kind="mergesort").reset_index(drop=True)
    panel["adv"] = (
        panel.groupby("asset", sort=False)["amount"]
        .rolling(window=int(lookback), min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    q = 1.0 - float(min_adv_pct) / 100.0
    threshold = panel.groupby("date", sort=False)["adv"].transform(lambda s: s.quantile(q))
    out = panel[["date", "asset"]].copy()
    out["in_universe"] = panel["adv"] >= threshold
    return out.sort_values(["date", "asset"], kind="mergesort").reset_index(drop=True)


def build_market_cap_universe(
    daily_basic_df: pd.DataFrame,
    min_cap_pct: float = 20,
) -> pd.DataFrame:
    """Build a dynamic top-market-cap universe."""
    if min_cap_pct <= 0 or min_cap_pct > 100:
        raise ValueError("min_cap_pct must be in (0, 100]")
    require_columns(daily_basic_df, ("date", "asset"), "daily_basic_df")

    cap_col = _resolve_cap_col(daily_basic_df)
    panel = daily_basic_df[["date", "asset", cap_col]].copy()
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce")
    panel[cap_col] = pd.to_numeric(panel[cap_col], errors="coerce")
    panel = panel.dropna(subset=["date", "asset", cap_col]).reset_index(drop=True)

    q = 1.0 - float(min_cap_pct) / 100.0
    threshold = panel.groupby("date", sort=False)[cap_col].transform(lambda s: s.quantile(q))
    out = panel[["date", "asset"]].copy()
    out["in_universe"] = panel[cap_col] >= threshold
    return out.sort_values(["date", "asset"], kind="mergesort").reset_index(drop=True)


def combine_universe_filters(*filters: pd.DataFrame) -> pd.DataFrame:
    """Combine multiple universe masks with logical AND."""
    if not filters:
        return pd.DataFrame(columns=["date", "asset", "in_universe"])

    merged: pd.DataFrame | None = None
    flag_cols: list[str] = []
    for i, filt in enumerate(filters):
        require_columns(filt, ("date", "asset", "in_universe"), f"filters[{i}]")
        part = filt[["date", "asset", "in_universe"]].copy()
        part["date"] = pd.to_datetime(part["date"], errors="coerce")
        flag_col = f"in_universe_{i}"
        part = part.rename(columns={"in_universe": flag_col})
        part[flag_col] = part[flag_col].astype(bool)
        flag_cols.append(flag_col)
        if merged is None:
            merged = part
        else:
            merged = merged.merge(part, on=["date", "asset"], how="outer")

    assert merged is not None
    combined = merged[["date", "asset"]].copy()
    combined["in_universe"] = merged[flag_cols].fillna(False).all(axis=1)
    return combined.sort_values(["date", "asset"], kind="mergesort").reset_index(drop=True)


def _resolve_cap_col(frame: pd.DataFrame) -> str:
    for col in ("circ_mv", "total_mv", "market_cap", "value"):
        if col in frame.columns:
            return col
    raise AlphaLabDataError(
        "daily_basic_df must contain one of: circ_mv, total_mv, market_cap, value"
    )
