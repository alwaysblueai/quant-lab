from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from alpha_lab.backtest_adapter.schema import BacktestInputBundle
from alpha_lab.data_validation import validate_price_panel

_SIGNAL_COLUMNS: tuple[str, ...] = ("date", "asset", "signal_name", "signal_value")
_UNIVERSE_COLUMNS: tuple[str, ...] = ("date", "asset", "in_universe")
_TRADABILITY_COLUMNS: tuple[str, ...] = ("date", "asset", "is_tradable")


def validate_backtest_input_bundle(bundle: BacktestInputBundle) -> None:
    """Validate the in-memory bundle contract consumed by adapters."""

    _assert_columns(bundle.signal_snapshot_df, _SIGNAL_COLUMNS, "signal_snapshot_df")
    _assert_columns(bundle.universe_mask_df, _UNIVERSE_COLUMNS, "universe_mask_df")
    _assert_columns(bundle.tradability_mask_df, _TRADABILITY_COLUMNS, "tradability_mask_df")

    _assert_unique_keys(bundle.signal_snapshot_df, ("date", "asset"), "signal_snapshot_df")
    _assert_unique_keys(bundle.universe_mask_df, ("date", "asset"), "universe_mask_df")
    _assert_unique_keys(bundle.tradability_mask_df, ("date", "asset"), "tradability_mask_df")

    signal_keys = _key_set(bundle.signal_snapshot_df, ("date", "asset"))
    universe_keys = _key_set(bundle.universe_mask_df, ("date", "asset"))
    tradability_keys = _key_set(bundle.tradability_mask_df, ("date", "asset"))
    missing_in_universe = signal_keys - universe_keys
    if missing_in_universe:
        raise ValueError(
            "signal_snapshot_df keys must be covered by universe_mask_df; "
            f"{len(missing_in_universe)} signal key(s) are missing from universe mask"
        )
    missing_in_tradability = signal_keys - tradability_keys
    if missing_in_tradability:
        raise ValueError(
            "signal_snapshot_df keys must be covered by tradability_mask_df; "
            f"{len(missing_in_tradability)} signal key(s) are missing from tradability mask"
        )
    if bundle.signal_name != bundle.portfolio_construction.signal_name:
        raise ValueError(
            "bundle signal_name does not match portfolio_construction.signal_name"
        )


def validate_price_inputs(
    price_df: pd.DataFrame,
    *,
    close_column: str,
    open_column: str | None,
) -> None:
    """Validate external price panel required by the replay adapter."""

    if close_column not in price_df.columns:
        raise ValueError(f"price_df is missing close column {close_column!r}")
    base = price_df[["date", "asset", close_column]].rename(columns={close_column: "close"})
    validate_price_panel(base)

    if open_column is not None:
        if open_column not in price_df.columns:
            raise ValueError(f"price_df is missing open column {open_column!r}")
        opens = price_df[open_column]
        n_nan = int(opens.isna().sum())
        if n_nan > 0:
            raise ValueError(f"price_df[{open_column!r}] contains {n_nan} NaN value(s)")
        n_nonpos = int((opens <= 0).sum())
        if n_nonpos > 0:
            raise ValueError(
                f"price_df[{open_column!r}] contains {n_nonpos} non-positive value(s)"
            )


def _assert_columns(
    df: pd.DataFrame,
    required: Iterable[str],
    table_name: str,
) -> None:
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"{table_name} missing required columns: {sorted(missing)}")


def _assert_unique_keys(df: pd.DataFrame, keys: tuple[str, ...], table_name: str) -> None:
    n_dupes = int(df.duplicated(subset=list(keys)).sum())
    if n_dupes:
        raise ValueError(
            f"{table_name} has {n_dupes} duplicate row(s) for key columns {list(keys)}"
        )


def _key_set(df: pd.DataFrame, keys: tuple[str, ...]) -> set[tuple[object, ...]]:
    items = df.loc[:, list(keys)]
    return {tuple(row) for row in items.itertuples(index=False, name=None)}
