"""Shared IO and validation helpers for real-case research pipelines.

Extracted from ``single_factor.pipeline`` and ``composite.pipeline`` to
eliminate duplication.  Both pipelines delegate prices/universe/factor loading
and universe filtering to this module.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from alpha_lab.exceptions import AlphaLabDataError, AlphaLabIOError
from alpha_lab.research_contracts import validate_canonical_signal_table, validate_prices_table
from alpha_lab.real_cases.common_spec import UniverseSpec


# ---------------------------------------------------------------------------
# Prices
# ---------------------------------------------------------------------------


def load_prices(path_value: str) -> pd.DataFrame:
    """Load, validate, and return a price panel from a CSV path.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If required columns are missing or validation fails.
    """
    path = Path(path_value)
    if not path.exists() or not path.is_file():
        raise AlphaLabIOError(f"prices file does not exist: {path}")

    prices = pd.read_csv(path)
    required = {"date", "asset", "close"}
    missing = required - set(prices.columns)
    if missing:
        raise AlphaLabDataError(f"prices is missing required columns: {sorted(missing)}")

    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
    prices = prices.sort_values(["asset", "date"], kind="mergesort").reset_index(drop=True)
    validate_prices_table(prices)
    return prices


# ---------------------------------------------------------------------------
# Universe mask
# ---------------------------------------------------------------------------


def load_universe_mask(universe_spec: UniverseSpec) -> pd.DataFrame | None:
    """Load an optional universe mask CSV.

    Returns ``None`` when ``universe_spec.path`` is ``None`` (no universe
    filter configured).

    Raises
    ------
    FileNotFoundError
        If the configured path does not exist.
    ValueError
        If required columns are missing or duplicates are found.
    """
    if universe_spec.path is None:
        return None

    path = Path(universe_spec.path)
    if not path.exists() or not path.is_file():
        raise AlphaLabIOError(f"universe file does not exist: {path}")

    universe = pd.read_csv(path)
    col = universe_spec.in_universe_column
    required = {"date", "asset", col}
    missing = required - set(universe.columns)
    if missing:
        raise AlphaLabDataError(f"universe file is missing required columns: {sorted(missing)}")

    out = universe[["date", "asset", col]].copy()
    out = out.rename(columns={col: "in_universe"})
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date", "asset"]).copy()
    if out.duplicated(subset=["date", "asset"]).any():
        raise AlphaLabDataError("universe file contains duplicate (date, asset) rows")
    out["in_universe"] = out["in_universe"].astype(bool)
    return out


# ---------------------------------------------------------------------------
# Universe filtering
# ---------------------------------------------------------------------------


def apply_universe_to_prices(
    prices: pd.DataFrame, universe_mask: pd.DataFrame
) -> pd.DataFrame:
    """Inner-join prices against the active universe rows.

    Raises
    ------
    ValueError
        If the result is empty after filtering.
    """
    active = universe_mask[universe_mask["in_universe"]][["date", "asset"]]
    out = prices.merge(active, on=["date", "asset"], how="inner", validate="many_to_one")
    if out.empty:
        raise AlphaLabDataError("prices became empty after universe filtering")
    return out.sort_values(["asset", "date"], kind="mergesort").reset_index(drop=True)


def apply_universe_to_factor(
    factor_df: pd.DataFrame, universe_mask: pd.DataFrame
) -> pd.DataFrame:
    """Inner-join a factor DataFrame against the active universe rows.

    Raises
    ------
    ValueError
        If the result is empty after filtering.
    """
    active = universe_mask[universe_mask["in_universe"]][["date", "asset"]]
    out = factor_df.merge(active, on=["date", "asset"], how="inner", validate="many_to_one")
    if out.empty:
        raise AlphaLabDataError("factor data became empty after universe filtering")
    return out.sort_values(["date", "asset"], kind="mergesort").reset_index(drop=True)
