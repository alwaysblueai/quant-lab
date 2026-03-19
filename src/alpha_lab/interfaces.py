from __future__ import annotations

from typing import Protocol

import pandas as pd

FACTOR_OUTPUT_COLUMNS = ("date", "asset", "factor", "value")


class Factor(Protocol):
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Input:
            df must contain:
                - date
                - asset
                - required raw features

        Output:
            DataFrame with columns:
                - date
                - asset
                - factor
                - value
        """
        ...


def validate_factor_output(df: pd.DataFrame) -> None:
    required_cols = set(FACTOR_OUTPUT_COLUMNS)

    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df.empty:
        raise ValueError("Factor output is empty")

    if df["value"].isna().all():
        raise ValueError("Factor values are all NaN")

    dupes = df.duplicated(subset=["date", "asset", "factor"])
    if dupes.any():
        raise ValueError("Factor output contains duplicate (date, asset, factor) rows")
