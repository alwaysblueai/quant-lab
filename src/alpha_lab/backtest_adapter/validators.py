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

    signal_keys = _key_set(bundle.signal_snapshot_df)
    universe_keys = _key_set(bundle.universe_mask_df)
    tradability_keys = _key_set(bundle.tradability_mask_df)

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

    expected_signal = bundle.portfolio_construction.signal_name
    if expected_signal is not None and bundle.signal_name != expected_signal:
        raise ValueError(
            "bundle signal_name does not match portfolio_construction.signal_name"
        )


def validate_price_inputs(
    price_df: pd.DataFrame,
    close_column: str,
    open_column: str | None,
) -> None:
    """Validate external price panel required by the replay adapter."""
    validate_price_panel(price_df.rename(columns={close_column: "close"}))

    if close_column not in price_df.columns:
        raise ValueError(f"price_df is missing close column {close_column!r}")
    if open_column is not None and open_column not in price_df.columns:
        raise ValueError(f"price_df is missing open column {open_column!r}")

    for column_name in [close_column, open_column]:
        if column_name is None:
            continue
        base = pd.to_numeric(price_df[column_name], errors="coerce")
        n_nan = int(base.isna().sum())
        if n_nan > 0:
            raise ValueError(f"price_df[{column_name!r}] contains {n_nan} NaN value(s)")
        n_nonpos = int((base <= 0).sum())
        if n_nonpos > 0:
            raise ValueError(f"price_df[{column_name!r}] contains {n_nonpos} non-positive value(s)")


def _assert_columns(table: pd.DataFrame, required: Iterable[str], table_name: str) -> None:
    missing = set(required) - set(table.columns)
    if missing:
        raise ValueError(f"{table_name} missing required columns: {sorted(missing)}")


def _assert_unique_keys(table: pd.DataFrame, keys: tuple[str, ...], name: str) -> None:
    n_dupes = int(table.duplicated(list(keys)).sum())
    if n_dupes > 0:
        raise ValueError(f"{name} has {n_dupes} duplicate row(s) for key columns {list(keys)}")


def _key_set(table: pd.DataFrame) -> set[tuple[object, ...]]:
    return set(table.loc[:, ["date", "asset"]].itertuples(index=False, name=None))
