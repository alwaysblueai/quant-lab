"""Point-in-time (PIT) alignment for fundamental financial data.

Ensures that factor values at signal date *t* only use financial reports
that have been publicly announced by *t*, preventing look-ahead bias.
"""

from __future__ import annotations

import pandas as pd


def build_pit_mapping(
    fina_df: pd.DataFrame,
    ann_date_col: str = "ann_date",
    period_col: str = "end_date",
) -> pd.DataFrame:
    """Build a point-in-time mapping from financial report data.

    When the same ``(asset, end_date)`` has multiple announcement dates
    (e.g. a correction filing), the **earliest** announcement date is kept
    — the data became known to the market on that date.

    Parameters
    ----------
    fina_df:
        Financial data with at least ``[asset, end_date, ann_date]``.
    ann_date_col:
        Column name for the announcement date.
    period_col:
        Column name for the reporting period end date.

    Returns
    -------
    pd.DataFrame
        Deduplicated mapping with ``[asset, end_date, ann_date]``, one row
        per (asset, end_date) with the earliest announcement date.
    """
    df = fina_df[["asset", period_col, ann_date_col]].copy()
    df = df.rename(columns={period_col: "end_date", ann_date_col: "ann_date"})
    df["end_date"] = pd.to_datetime(df["end_date"])
    df["ann_date"] = pd.to_datetime(df["ann_date"])

    # Keep earliest announcement per (asset, end_date)
    df = df.sort_values("ann_date").drop_duplicates(subset=["asset", "end_date"], keep="first")
    return df.reset_index(drop=True)


def apply_pit_alignment(
    signal_dates_df: pd.DataFrame,
    fina_df: pd.DataFrame,
    pit_mapping: pd.DataFrame,
) -> pd.DataFrame:
    """Align financial data to signal dates using PIT rules.

    For each ``(date, asset)`` in ``signal_dates_df``, finds the most
    recent financial report whose announcement date is on or before the
    signal date.

    Parameters
    ----------
    signal_dates_df:
        DataFrame with ``[date, asset]`` — the dates at which we need
        fundamental values.
    fina_df:
        Financial data with at least ``[asset, end_date, ann_date]`` plus
        value columns (e.g. ``roe``, ``pb``).
    pit_mapping:
        Output of :func:`build_pit_mapping`.

    Returns
    -------
    pd.DataFrame
        One row per (date, asset) from ``signal_dates_df``, with the value
        columns from ``fina_df`` filled using PIT-compliant lookup.
        Rows where no report has been announced yet have NaN values.
    """
    # Merge fina data with PIT mapping to get the authoritative ann_date
    value_cols = [c for c in fina_df.columns if c not in ("asset", "end_date", "ann_date")]
    fina = fina_df[["asset", "end_date", "ann_date"] + value_cols].copy()
    fina["end_date"] = pd.to_datetime(fina["end_date"])
    fina["ann_date"] = pd.to_datetime(fina["ann_date"])

    # Use the PIT mapping's ann_date (earliest announcement)
    fina = fina.drop(columns=["ann_date"]).merge(
        pit_mapping[["asset", "end_date", "ann_date"]],
        on=["asset", "end_date"],
        how="inner",
    )

    signals = signal_dates_df[["date", "asset"]].copy()
    signals["date"] = pd.to_datetime(signals["date"])

    # For each (date, asset), find the latest report announced on or before date
    results = []
    for _, sig_row in signals.iterrows():
        dt = sig_row["date"]
        asset = sig_row["asset"]

        available = fina[(fina["asset"] == asset) & (fina["ann_date"] <= dt)]
        row_data = {"date": dt, "asset": asset}
        if available.empty:
            for col in value_cols:
                row_data[col] = float("nan")
        else:
            # Pick the report with the latest period end date
            best = available.sort_values("end_date").iloc[-1]
            for col in value_cols:
                row_data[col] = best[col]
        results.append(row_data)

    return pd.DataFrame(results)


def validate_pit_compliance(
    factor_df: pd.DataFrame,
    pit_mapping: pd.DataFrame,
    factor_date_col: str = "date",
) -> dict[str, object]:
    """Check whether a factor DataFrame might use unannounced data.

    Compares factor dates against the PIT mapping to detect rows where
    the factor might be using a financial report that has not yet been
    announced.

    Parameters
    ----------
    factor_df:
        Factor values with at least ``[date, asset]``.
    pit_mapping:
        Output of :func:`build_pit_mapping`.
    factor_date_col:
        Column in ``factor_df`` containing the signal date.

    Returns
    -------
    dict with keys:
        - ``status``: ``"pass"`` or ``"fail"``
        - ``n_violations``: number of (date, asset) pairs with look-ahead
        - ``violations``: DataFrame of violating rows (empty if pass)
    """
    # For each asset, find the latest end_date in the factor data
    # and check that the corresponding ann_date <= factor_date
    factor = factor_df[[factor_date_col, "asset"]].copy()
    factor = factor.rename(columns={factor_date_col: "signal_date"})
    factor["signal_date"] = pd.to_datetime(factor["signal_date"])

    mapping = pit_mapping.copy()

    # Cross join: for each (signal_date, asset), check all reports
    merged = factor.merge(mapping, on="asset", how="inner")

    # More precisely: a violation is when a factor_date is between ann_date and end_date
    # meaning the report period has ended but not yet announced
    precise_violations = merged[
        (merged["signal_date"] >= merged["end_date"]) & (merged["signal_date"] < merged["ann_date"])
    ]

    n_violations = len(precise_violations.drop_duplicates(subset=["signal_date", "asset"]))
    return {
        "status": "pass" if n_violations == 0 else "fail",
        "n_violations": n_violations,
        "violations": precise_violations,
    }
