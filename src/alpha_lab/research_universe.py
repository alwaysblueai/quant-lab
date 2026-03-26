from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from alpha_lab.data_validation import validate_price_panel

_REQUIRED_PRICE_COLUMNS: tuple[str, ...] = ("date", "asset", "close")
_REQUIRED_METADATA_COLUMNS: tuple[str, ...] = ("asset",)
_REQUIRED_STATE_COLUMNS: tuple[str, ...] = ("date", "asset")


@dataclass(frozen=True)
class ResearchUniverseRules:
    """Explicit research sample-construction rules.

    These rules are deterministic and inspectable. They define how a broad
    candidate sample becomes `in_universe`, then how executable names become
    `is_tradable`.
    """

    min_listing_age_days: int = 60
    min_adv: float | None = None
    adv_window: int = 20
    listing_date_col: str = "listing_date"
    st_flag_col: str = "is_st"
    halted_flag_col: str = "is_halted"
    limit_locked_flag_col: str = "is_limit_locked"
    dollar_volume_col: str = "dollar_volume"
    volume_col: str = "volume"

    def __post_init__(self) -> None:
        if self.min_listing_age_days < 0:
            raise ValueError("min_listing_age_days must be >= 0")
        if self.min_adv is not None and self.min_adv < 0:
            raise ValueError("min_adv must be >= 0 when provided")
        if self.adv_window <= 0:
            raise ValueError("adv_window must be > 0")


@dataclass(frozen=True)
class ResearchUniverseResult:
    """Deterministic sample-construction outputs."""

    universe: pd.DataFrame
    tradability: pd.DataFrame
    exclusion_reasons: pd.DataFrame
    diagnostics: pd.DataFrame


def construct_research_universe(
    prices: pd.DataFrame,
    *,
    asset_metadata: pd.DataFrame | None = None,
    market_state: pd.DataFrame | None = None,
    rules: ResearchUniverseRules | None = None,
) -> ResearchUniverseResult:
    """Build PIT-safe universe/tradability tables with exclusion reasons."""
    validate_price_panel(prices)
    resolved_rules = rules or ResearchUniverseRules()
    price = _prepare_prices(prices)
    meta = _prepare_asset_metadata(
        asset_metadata,
        listing_date_col=resolved_rules.listing_date_col,
        st_flag_col=resolved_rules.st_flag_col,
    )
    state = _prepare_market_state(
        market_state,
        halted_flag_col=resolved_rules.halted_flag_col,
        limit_locked_flag_col=resolved_rules.limit_locked_flag_col,
        st_flag_col=resolved_rules.st_flag_col,
    )

    base = price[["date", "asset"]].drop_duplicates().copy()
    merged = base.merge(meta, on="asset", how="left")
    merged = merged.merge(state, on=["date", "asset"], how="left")
    merged = merged.merge(
        _adv_table(
            price,
            rules=resolved_rules,
        ),
        on=["date", "asset"],
        how="left",
    )

    # asset_metadata and market_state can both provide ST flags.
    # Keep the combined effective flag explicit and deterministic.
    st_candidates = [col for col in ("is_st_x", "is_st_y", "is_st") if col in merged.columns]
    if st_candidates:
        merged["is_st"] = False
        for col in st_candidates:
            merged["is_st"] = merged["is_st"] | merged[col].fillna(False).astype(bool)
        drop_cols = [col for col in ("is_st_x", "is_st_y") if col in merged.columns]
        if drop_cols:
            merged = merged.drop(columns=drop_cols)
    else:
        merged["is_st"] = False

    merged["listing_date"] = pd.to_datetime(
        merged["listing_date"],
        errors="coerce",
    )
    merged["is_st"] = merged["is_st"].fillna(False).astype(bool)
    merged["is_halted"] = merged["is_halted"].fillna(False).astype(bool)
    merged["is_limit_locked"] = merged["is_limit_locked"].fillna(False).astype(bool)

    listing_age = (merged["date"] - merged["listing_date"]).dt.days
    listing_ok = (
        merged["listing_date"].notna()
        & (listing_age >= resolved_rules.min_listing_age_days)
    )
    st_ok = ~merged["is_st"]

    merged["in_universe"] = (listing_ok & st_ok).astype(bool)

    tradable = merged["in_universe"].copy()
    tradable = tradable & (~merged["is_halted"])
    tradable = tradable & (~merged["is_limit_locked"])
    if resolved_rules.min_adv is not None:
        tradable = tradable & (
            merged["adv_dollar_volume"].fillna(0.0) >= resolved_rules.min_adv
        )
    merged["is_tradable"] = tradable.astype(bool)

    exclusion_reasons = _build_exclusion_reasons(
        merged,
        listing_ok=listing_ok,
        st_ok=st_ok,
        rules=resolved_rules,
    )

    universe = merged[["date", "asset", "in_universe"]].copy()
    tradability = merged[["date", "asset", "is_tradable"]].copy()

    diagnostics = _build_diagnostics(merged)
    universe = universe.sort_values(["date", "asset"], kind="mergesort").reset_index(drop=True)
    tradability = tradability.sort_values(
        ["date", "asset"],
        kind="mergesort",
    ).reset_index(drop=True)

    return ResearchUniverseResult(
        universe=universe,
        tradability=tradability,
        exclusion_reasons=exclusion_reasons,
        diagnostics=diagnostics,
    )


def _prepare_prices(prices: pd.DataFrame) -> pd.DataFrame:
    missing = set(_REQUIRED_PRICE_COLUMNS) - set(prices.columns)
    if missing:
        raise ValueError(f"prices missing required columns: {sorted(missing)}")
    out = prices.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    if out["date"].isna().any():
        raise ValueError("prices contains invalid date values")
    return out.sort_values(["asset", "date"], kind="mergesort").reset_index(drop=True)


def _prepare_asset_metadata(
    metadata: pd.DataFrame | None,
    *,
    listing_date_col: str,
    st_flag_col: str,
) -> pd.DataFrame:
    if metadata is None:
        return pd.DataFrame(columns=["asset", "listing_date", "is_st"])
    missing = set(_REQUIRED_METADATA_COLUMNS) - set(metadata.columns)
    if missing:
        raise ValueError(f"asset_metadata missing required columns: {sorted(missing)}")

    out = metadata.copy()
    out = out.rename(columns={listing_date_col: "listing_date", st_flag_col: "is_st"})
    if "listing_date" not in out.columns:
        out["listing_date"] = pd.NaT
    if "is_st" not in out.columns:
        out["is_st"] = False
    out["is_st"] = out["is_st"].fillna(False).astype(bool)
    out = out[["asset", "listing_date", "is_st"]].copy()
    dupes = out.duplicated(subset=["asset"])
    if dupes.any():
        raise ValueError("asset_metadata contains duplicate asset rows")
    return out.reset_index(drop=True)


def _prepare_market_state(
    market_state: pd.DataFrame | None,
    *,
    halted_flag_col: str,
    limit_locked_flag_col: str,
    st_flag_col: str,
) -> pd.DataFrame:
    if market_state is None:
        return pd.DataFrame(columns=["date", "asset", "is_halted", "is_limit_locked", "is_st"])
    missing = set(_REQUIRED_STATE_COLUMNS) - set(market_state.columns)
    if missing:
        raise ValueError(f"market_state missing required columns: {sorted(missing)}")
    out = market_state.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    if out["date"].isna().any():
        raise ValueError("market_state contains invalid date values")
    out = out.rename(
        columns={
            halted_flag_col: "is_halted",
            limit_locked_flag_col: "is_limit_locked",
            st_flag_col: "is_st",
        }
    )
    if "is_halted" not in out.columns:
        out["is_halted"] = False
    if "is_limit_locked" not in out.columns:
        out["is_limit_locked"] = False
    if "is_st" not in out.columns:
        out["is_st"] = False
    out["is_halted"] = out["is_halted"].fillna(False).astype(bool)
    out["is_limit_locked"] = out["is_limit_locked"].fillna(False).astype(bool)
    out["is_st"] = out["is_st"].fillna(False).astype(bool)
    dupes = out.duplicated(subset=["date", "asset"])
    if dupes.any():
        raise ValueError("market_state contains duplicate (date, asset) rows")
    return out[["date", "asset", "is_halted", "is_limit_locked", "is_st"]].reset_index(drop=True)


def _adv_table(
    prices: pd.DataFrame,
    *,
    rules: ResearchUniverseRules,
) -> pd.DataFrame:
    out = prices[["date", "asset", "close"]].copy()
    if rules.dollar_volume_col in prices.columns:
        out["dollar_volume"] = pd.to_numeric(
            prices[rules.dollar_volume_col],
            errors="coerce",
        )
    elif rules.volume_col in prices.columns:
        volume = pd.to_numeric(prices[rules.volume_col], errors="coerce")
        out["dollar_volume"] = prices["close"] * volume
    else:
        out["dollar_volume"] = pd.NA
    out["dollar_volume"] = pd.to_numeric(out["dollar_volume"], errors="coerce")
    out = out.sort_values(["asset", "date"], kind="mergesort").reset_index(drop=True)
    out["adv_dollar_volume"] = (
        out.groupby("asset", sort=False)["dollar_volume"]
        .rolling(window=rules.adv_window, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    return out[["date", "asset", "adv_dollar_volume"]]


def _build_exclusion_reasons(
    merged: pd.DataFrame,
    *,
    listing_ok: pd.Series,
    st_ok: pd.Series,
    rules: ResearchUniverseRules,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []

    listing_fail = merged[~listing_ok][["date", "asset"]].copy()
    if not listing_fail.empty:
        listing_fail["reason"] = "listing_age_or_missing_listing_date"
        listing_fail["detail"] = (
            f"min_listing_age_days={rules.min_listing_age_days}"
        )
        rows.append(listing_fail)

    st_fail = merged[~st_ok][["date", "asset"]].copy()
    if not st_fail.empty:
        st_fail["reason"] = "st_filter"
        st_fail["detail"] = "asset flagged as ST"
        rows.append(st_fail)

    halted_fail = merged[merged["in_universe"] & merged["is_halted"]][["date", "asset"]].copy()
    if not halted_fail.empty:
        halted_fail["reason"] = "halted_trading"
        halted_fail["detail"] = "is_halted=true"
        rows.append(halted_fail)

    limit_fail = merged[
        merged["in_universe"] & (~merged["is_halted"]) & merged["is_limit_locked"]
    ][["date", "asset"]].copy()
    if not limit_fail.empty:
        limit_fail["reason"] = "limit_locked_non_executable"
        limit_fail["detail"] = "is_limit_locked=true"
        rows.append(limit_fail)

    if rules.min_adv is not None:
        adv_fail = merged[
            merged["in_universe"]
            & (~merged["is_halted"])
            & (~merged["is_limit_locked"])
            & (merged["adv_dollar_volume"].fillna(0.0) < rules.min_adv)
        ][["date", "asset"]].copy()
        if not adv_fail.empty:
            adv_fail["reason"] = "min_adv_filter"
            adv_fail["detail"] = f"min_adv={rules.min_adv}"
            rows.append(adv_fail)

    if not rows:
        return pd.DataFrame(columns=["date", "asset", "reason", "detail"])
    out = pd.concat(rows, ignore_index=True)
    return out.sort_values(["date", "asset", "reason"], kind="mergesort").reset_index(drop=True)


def _build_diagnostics(merged: pd.DataFrame) -> pd.DataFrame:
    grouped = merged.groupby("date", sort=True).agg(
        n_assets=("asset", "nunique"),
        n_in_universe=("in_universe", "sum"),
        n_tradable=("is_tradable", "sum"),
    )
    grouped["universe_ratio"] = grouped["n_in_universe"] / grouped["n_assets"].replace(0, pd.NA)
    grouped["tradable_ratio"] = grouped["n_tradable"] / grouped["n_assets"].replace(0, pd.NA)
    return grouped.reset_index()
