from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from alpha_lab.interfaces import validate_factor_output

_COMPOSITE_FACTOR_NAME = "ic_weighted_composite"


def compute_rolling_icir(
    ic_dict: Mapping[str, pd.DataFrame],
    window: int = 60,
) -> pd.DataFrame:
    """Rolling ICIR by factor and date."""
    if window < 2:
        raise ValueError("window must be >= 2")
    if not ic_dict:
        return pd.DataFrame(columns=["date", "factor", "rolling_icir"])

    rows: list[pd.DataFrame] = []
    for factor_name, ic_df in ic_dict.items():
        if ic_df.empty:
            continue
        if "date" not in ic_df.columns or "ic" not in ic_df.columns:
            raise ValueError(f"ic_dict[{factor_name!r}] must contain columns ['date', 'ic']")

        per_date = ic_df[["date", "ic"]].copy()
        per_date["date"] = pd.to_datetime(per_date["date"], errors="coerce")
        per_date["ic"] = pd.to_numeric(per_date["ic"], errors="coerce")
        per_date = (
            per_date.dropna(subset=["date"]).groupby("date", sort=True)["ic"].mean().sort_index()
        )
        if per_date.empty:
            continue

        rolling_mean = per_date.rolling(window=window, min_periods=2).mean()
        rolling_std = per_date.rolling(window=window, min_periods=2).std(ddof=1)
        rolling_icir = pd.Series(np.nan, index=per_date.index, dtype=float)
        valid = rolling_std > 0.0
        rolling_icir.loc[valid] = rolling_mean.loc[valid] / rolling_std.loc[valid]

        rows.append(
            pd.DataFrame(
                {
                    "date": rolling_icir.index,
                    "factor": str(factor_name),
                    "rolling_icir": rolling_icir.to_numpy(dtype=float),
                }
            )
        )

    if not rows:
        return pd.DataFrame(columns=["date", "factor", "rolling_icir"])
    return (
        pd.concat(rows, ignore_index=True)
        .sort_values(["date", "factor"], kind="mergesort")
        .reset_index(drop=True)
    )


def build_ic_weighted_composite(
    factor_dict: Mapping[str, pd.DataFrame],
    ic_dict: Mapping[str, pd.DataFrame],
    window: int = 60,
    min_positive_icir: float = 0.0,
) -> pd.DataFrame:
    """Build a time-varying ICIR-weighted composite factor."""
    if not factor_dict:
        return pd.DataFrame(columns=["date", "asset", "factor", "value"])

    rolling_icir = compute_rolling_icir(ic_dict, window=window)

    weighted_parts: list[pd.DataFrame] = []
    for factor_name, factor_df in factor_dict.items():
        if factor_df.empty:
            continue
        validate_factor_output(factor_df)

        base = factor_df[["date", "asset", "value"]].copy()
        base["date"] = pd.to_datetime(base["date"], errors="coerce")
        base["value"] = pd.to_numeric(base["value"], errors="coerce")

        icir_slice = rolling_icir.loc[
            rolling_icir["factor"] == str(factor_name), ["date", "rolling_icir"]
        ]
        base = base.merge(icir_slice, on="date", how="left")

        weight = pd.to_numeric(base["rolling_icir"], errors="coerce")
        weight = weight.where(weight >= float(min_positive_icir), float(min_positive_icir))
        effective_weight = weight.where(base["value"].notna(), 0.0).fillna(0.0)

        part = base[["date", "asset"]].copy()
        part["weighted_value"] = base["value"] * effective_weight
        part["effective_weight"] = effective_weight
        weighted_parts.append(part)

    if not weighted_parts:
        return pd.DataFrame(columns=["date", "asset", "factor", "value"])

    stacked = pd.concat(weighted_parts, ignore_index=True)
    agg = (
        stacked.groupby(["date", "asset"], sort=True)[["weighted_value", "effective_weight"]]
        .sum(min_count=1)
        .reset_index()
    )

    denom = agg["effective_weight"].replace(0.0, np.nan)
    out = agg[["date", "asset"]].copy()
    out["factor"] = _COMPOSITE_FACTOR_NAME
    out["value"] = agg["weighted_value"] / denom
    return (
        out[["date", "asset", "factor", "value"]]
        .sort_values(["date", "asset"], kind="mergesort")
        .reset_index(drop=True)
    )
