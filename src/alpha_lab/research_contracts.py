from __future__ import annotations

import pandas as pd

from alpha_lab.data_validation import validate_price_panel
from alpha_lab.exceptions import AlphaLabDataError
from alpha_lab.interfaces import validate_factor_output


def validate_prices_table(prices: pd.DataFrame) -> None:
    """Validate canonical long-form prices table: [date, asset, close]."""

    validate_price_panel(prices)


def validate_canonical_signal_table(
    signal_df: pd.DataFrame,
    *,
    table_name: str = "signal",
) -> None:
    """Validate canonical long-form signal table: [date, asset, factor, value]."""

    try:
        validate_factor_output(signal_df)
    except ValueError as exc:
        raise AlphaLabDataError(f"{table_name} violates canonical signal contract: {exc}") from exc
