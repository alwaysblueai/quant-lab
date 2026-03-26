from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from alpha_lab.interfaces import FACTOR_OUTPUT_COLUMNS

_PRICE_REQUIRED = ("date", "asset", "close")
_UNIVERSE_REQUIRED = ("date", "asset", "in_universe")
_TRADABILITY_REQUIRED = ("date", "asset", "is_tradable")


@dataclass(frozen=True)
class DatasetSnapshot:
    """Minimal dataset snapshot/provenance descriptor."""

    dataset_id: str
    as_of_utc: str | None = None
    source: str | None = None
    version: str | None = None
    dataset_hash: str | None = None
    notes: str | None = None

    def __post_init__(self) -> None:
        if not self.dataset_id.strip():
            raise ValueError("dataset_id must be a non-empty string")


@dataclass(frozen=True)
class ResearchBundle:
    """Canonical typed input bundle for factor research workflows."""

    prices: pd.DataFrame
    factors: pd.DataFrame | None = None
    labels: pd.DataFrame | None = None
    universe: pd.DataFrame | None = None
    tradability: pd.DataFrame | None = None
    metadata: pd.DataFrame | None = None
    snapshot: DatasetSnapshot | None = None

    def validate(self) -> None:
        validate_prices_table(self.prices, require_monotonic_by_asset=True)
        if self.factors is not None:
            validate_canonical_signal_table(self.factors, table_name="factors")
        if self.labels is not None:
            validate_canonical_signal_table(self.labels, table_name="labels")
        if self.universe is not None:
            validate_universe_table(self.universe)
        if self.tradability is not None:
            validate_tradability_table(self.tradability)

        if self.universe is not None and self.tradability is not None:
            _validate_unique_index_alignment(self.universe, self.tradability)


def validate_prices_table(
    df: pd.DataFrame,
    *,
    require_monotonic_by_asset: bool = True,
) -> None:
    missing = set(_PRICE_REQUIRED) - set(df.columns)
    if missing:
        raise ValueError(f"prices is missing required columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("prices is empty")

    date = pd.to_datetime(df["date"], errors="coerce")
    if date.isna().any():
        raise ValueError("prices contains NaT/unparseable values in 'date'")
    if df["asset"].isna().any() or (df["asset"].astype(str).str.strip() == "").any():
        raise ValueError("prices contains null/empty asset identifiers")
    if df["close"].isna().any():
        raise ValueError("prices contains NaN close values")
    if (df["close"] <= 0).any():
        raise ValueError("prices contains non-positive close values")
    if df.duplicated(subset=["date", "asset"]).any():
        raise ValueError("prices contains duplicate (date, asset) rows")

    if require_monotonic_by_asset:
        tmp = df[["asset", "date"]].copy()
        tmp["date"] = date
        # Strictly increasing date order per asset in observed row order.
        non_monotonic_assets = []
        for asset, group in tmp.groupby("asset", sort=False):
            if not group["date"].is_monotonic_increasing:
                non_monotonic_assets.append(str(asset))
        if non_monotonic_assets:
            raise ValueError(
                "prices must be monotonic-increasing by date within each asset; "
                f"violations: {non_monotonic_assets[:5]}"
            )


def validate_canonical_signal_table(df: pd.DataFrame, *, table_name: str) -> None:
    missing = set(FACTOR_OUTPUT_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{table_name} is missing required columns: {sorted(missing)}")
    if df.empty:
        raise ValueError(f"{table_name} is empty")

    date = pd.to_datetime(df["date"], errors="coerce")
    if date.isna().any():
        raise ValueError(f"{table_name} contains NaT/unparseable values in 'date'")
    if df["asset"].isna().any() or (df["asset"].astype(str).str.strip() == "").any():
        raise ValueError(f"{table_name} contains null/empty asset identifiers")
    if df["factor"].isna().any() or (df["factor"].astype(str).str.strip() == "").any():
        raise ValueError(f"{table_name} contains null/empty factor identifiers")
    if df.duplicated(subset=["date", "asset", "factor"]).any():
        raise ValueError(f"{table_name} contains duplicate (date, asset, factor) rows")
    if not pd.api.types.is_numeric_dtype(df["value"]):
        raise ValueError(f"{table_name}.value must be numeric")


def validate_universe_table(df: pd.DataFrame) -> None:
    missing = set(_UNIVERSE_REQUIRED) - set(df.columns)
    if missing:
        raise ValueError(f"universe is missing required columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("universe is empty")

    date = pd.to_datetime(df["date"], errors="coerce")
    if date.isna().any():
        raise ValueError("universe contains NaT/unparseable values in 'date'")
    if df["asset"].isna().any() or (df["asset"].astype(str).str.strip() == "").any():
        raise ValueError("universe contains null/empty asset identifiers")
    if df.duplicated(subset=["date", "asset"]).any():
        raise ValueError("universe contains duplicate (date, asset) rows")
    if not pd.api.types.is_bool_dtype(df["in_universe"]):
        raise ValueError("universe.in_universe must be bool dtype")


def validate_tradability_table(df: pd.DataFrame) -> None:
    missing = set(_TRADABILITY_REQUIRED) - set(df.columns)
    if missing:
        raise ValueError(f"tradability is missing required columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("tradability is empty")

    date = pd.to_datetime(df["date"], errors="coerce")
    if date.isna().any():
        raise ValueError("tradability contains NaT/unparseable values in 'date'")
    if df["asset"].isna().any() or (df["asset"].astype(str).str.strip() == "").any():
        raise ValueError("tradability contains null/empty asset identifiers")
    if df.duplicated(subset=["date", "asset"]).any():
        raise ValueError("tradability contains duplicate (date, asset) rows")
    if not pd.api.types.is_bool_dtype(df["is_tradable"]):
        raise ValueError("tradability.is_tradable must be bool dtype")


def _validate_unique_index_alignment(left: pd.DataFrame, right: pd.DataFrame) -> None:
    left_key = set(zip(pd.to_datetime(left["date"]), left["asset"], strict=False))
    right_key = set(zip(pd.to_datetime(right["date"]), right["asset"], strict=False))
    if left_key != right_key:
        raise ValueError(
            "universe and tradability must share the same (date, asset) support"
        )
