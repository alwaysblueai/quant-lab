from __future__ import annotations

import pandas as pd

from alpha_lab.factors.momentum import momentum


def reversal(
    df: pd.DataFrame,
    *,
    window: int = 5,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """Compute a short-term reversal factor from each asset's own history.

    The factor at date ``t`` is defined as the negative of the trailing
    ``window``-row simple return:

    ``-(close[t] / close[t - window] - 1)``

    Higher values therefore indicate larger recent losers, which aligns the
    signal with the usual long-recent-losers / short-recent-winners
    interpretation of short-term reversal.
    """
    result = momentum(df, window=window, min_periods=min_periods).copy()
    result["factor"] = f"reversal_{window}d"
    result["value"] = -result["value"]
    return result
