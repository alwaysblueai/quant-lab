from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from alpha_lab.frame_utils import require_columns
from alpha_lab.research_integrity.contracts import IntegrityCheckResult


def build_delisting_calendar(delisting_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize delisting metadata to ``[asset, delist_date, last_trade_date]``."""
    require_columns(delisting_df, ("asset", "delist_date"), "delisting_df")

    out = delisting_df.copy()
    out["asset"] = out["asset"].astype(str).str.strip()
    out["delist_date"] = pd.to_datetime(out["delist_date"], errors="coerce")

    if "last_trade_date" in out.columns:
        out["last_trade_date"] = pd.to_datetime(out["last_trade_date"], errors="coerce")
    else:
        out["last_trade_date"] = pd.NaT

    missing_last = out["last_trade_date"].isna() & out["delist_date"].notna()
    out.loc[missing_last, "last_trade_date"] = out.loc[
        missing_last, "delist_date"
    ] - pd.offsets.BDay(1)
    invalid_order = out["last_trade_date"] > out["delist_date"]
    out.loc[invalid_order, "last_trade_date"] = out.loc[invalid_order, "delist_date"]

    out = out.dropna(subset=["asset", "delist_date", "last_trade_date"])
    out = out.drop_duplicates(subset=["asset"], keep="last")
    return (
        out[["asset", "delist_date", "last_trade_date"]]
        .sort_values(
            ["delist_date", "asset"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


def apply_delisting_return(
    prices_df: pd.DataFrame,
    delisting_calendar: pd.DataFrame,
    penalty: float = -0.10,
) -> pd.DataFrame:
    """Inject a one-day delisting penalty after each asset's last trade date."""
    require_columns(prices_df, ("date", "asset", "close"), "prices_df")
    require_columns(
        delisting_calendar,
        ("asset", "delist_date", "last_trade_date"),
        "delisting_calendar",
    )

    out = prices_df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out["asset"] = out["asset"].astype(str)

    if "delisting_return" not in out.columns:
        out["delisting_return"] = np.nan
    if "is_delisting_penalty" not in out.columns:
        out["is_delisting_penalty"] = False

    calendar = delisting_calendar.copy()
    calendar["last_trade_date"] = pd.to_datetime(calendar["last_trade_date"], errors="coerce")
    calendar = calendar.dropna(subset=["asset", "last_trade_date"])

    new_rows: list[dict[str, object]] = []
    for row in calendar.itertuples(index=False):
        asset = str(row.asset)
        last_trade_date = pd.Timestamp(row.last_trade_date)

        hist = out[(out["asset"] == asset) & (out["date"] <= last_trade_date)].sort_values(
            "date",
            kind="mergesort",
        )
        if hist.empty:
            continue

        last_close = float(hist.iloc[-1]["close"])
        if not np.isfinite(last_close) or last_close <= 0.0:
            continue

        penalty_date = (last_trade_date + pd.offsets.BDay(1)).normalize()
        row_payload: dict[str, object] = {col: np.nan for col in out.columns}
        row_payload["date"] = penalty_date
        row_payload["asset"] = asset
        row_payload["close"] = last_close * (1.0 + float(penalty))
        row_payload["delisting_return"] = float(penalty)
        row_payload["is_delisting_penalty"] = True
        new_rows.append(row_payload)

    if new_rows:
        out = pd.concat([out, pd.DataFrame(new_rows)], ignore_index=True, sort=False)
        out = out.sort_values(["date", "asset"], kind="mergesort")
        out = out.drop_duplicates(subset=["date", "asset"], keep="last").reset_index(drop=True)

    return out


def validate_universe_survivorship(
    universe_df: pd.DataFrame,
    all_historical_assets: Iterable[str],
) -> IntegrityCheckResult:
    """Check whether the universe excludes historically listed assets."""
    require_columns(universe_df, ("asset",), "universe_df")

    historical = {str(asset).strip() for asset in all_historical_assets if str(asset).strip()}
    present = {
        str(asset).strip()
        for asset in universe_df["asset"].dropna().astype(str)
        if str(asset).strip()
    }
    missing_assets = sorted(historical - present)

    if missing_assets:
        return IntegrityCheckResult(
            check_name="validate_universe_survivorship",
            status="fail",
            severity="error",
            message="universe excludes historically listed assets",
            object_name="universe_df",
            module_name="data_quality.survivorship",
            remediation="Use dynamic historical universe including delisted assets.",
            metrics={
                "missing_asset_count": len(missing_assets),
                "missing_assets_sample": missing_assets[:20],
            },
        )

    return IntegrityCheckResult(
        check_name="validate_universe_survivorship",
        status="pass",
        severity="info",
        message="universe includes all historical assets",
        object_name="universe_df",
        module_name="data_quality.survivorship",
        metrics={"historical_asset_count": len(historical)},
    )


