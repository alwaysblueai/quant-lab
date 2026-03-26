from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from alpha_lab.evaluation import compute_rank_ic
from alpha_lab.interfaces import FACTOR_OUTPUT_COLUMNS


@dataclass(frozen=True)
class CompositeSignalResult:
    """Composite alpha output with weight history and attribution diagnostics."""

    composite: pd.DataFrame
    weights: pd.DataFrame
    diagnostics: pd.DataFrame
    attribution: pd.DataFrame


def compose_signals(
    signals: pd.DataFrame,
    *,
    method: str = "equal",
    labels: pd.DataFrame | None = None,
    lookback: int = 63,
    min_history: int = 5,
    output_factor: str = "composite_signal",
) -> CompositeSignalResult:
    """Compose multiple factor signals into one cross-sectional composite."""
    _validate_canonical(signals, table_name="signals")
    method_l = method.lower()
    if method_l not in {"equal", "ic", "icir"}:
        raise ValueError("method must be one of {'equal', 'ic', 'icir'}")
    if lookback <= 0:
        raise ValueError("lookback must be > 0")
    if min_history <= 0:
        raise ValueError("min_history must be > 0")

    factors = sorted(pd.unique(signals["factor"].astype(str)).tolist())
    if len(factors) < 2:
        raise ValueError("compose_signals requires at least 2 input factors")

    s = signals.copy()
    s["date"] = pd.to_datetime(s["date"], errors="coerce")
    wide = s.pivot(index=["date", "asset"], columns="factor", values="value")
    wide = wide.sort_index(kind="mergesort")
    z = wide.groupby(level="date", sort=True).transform(_zscore_series)
    weights = _build_weights(
        s,
        factors=factors,
        method=method_l,
        labels=labels,
        lookback=lookback,
        min_history=min_history,
    )

    z_long = z.stack().rename("zscore").reset_index()
    z_long = z_long.rename(columns={"factor": "factor_name"})
    comp = z_long.merge(
        weights.rename(columns={"factor": "factor_name"}),
        on=["date", "factor_name"],
        how="left",
        validate="many_to_one",
    )
    comp["weighted_component"] = comp["zscore"] * comp["weight"]
    comp["abs_weight_present"] = comp["weight"].abs().where(comp["zscore"].notna(), 0.0)

    agg = comp.groupby(["date", "asset"], sort=True).agg(
        weighted_sum=("weighted_component", "sum"),
        abs_weight_sum=("abs_weight_present", "sum"),
    )
    agg["value"] = agg["weighted_sum"] / agg["abs_weight_sum"].replace(0.0, np.nan)
    composite = agg.reset_index()[["date", "asset", "value"]]
    composite["factor"] = output_factor
    composite = composite[["date", "asset", "factor", "value"]].sort_values(
        ["date", "asset"],
        kind="mergesort",
    ).reset_index(drop=True)

    attribution = (
        comp.groupby(["date", "factor_name"], sort=True)["weighted_component"]
        .apply(_mean_abs_component)
        .rename("mean_abs_component")
        .reset_index()
        .rename(columns={"factor_name": "factor"})
    )

    diagnostics = (
        composite.groupby("date", sort=True)["asset"]
        .nunique()
        .rename("n_assets")
        .reset_index()
    )
    diagnostics["n_factors"] = len(factors)
    diagnostics["coverage"] = composite.groupby("date", sort=True)["value"].apply(
        lambda x: float(x.notna().mean())
    ).to_numpy()
    return CompositeSignalResult(
        composite=composite,
        weights=weights.sort_values(["date", "factor"], kind="mergesort").reset_index(
            drop=True
        ),
        diagnostics=diagnostics.sort_values("date", kind="mergesort").reset_index(drop=True),
        attribution=attribution.sort_values(["date", "factor"], kind="mergesort").reset_index(
            drop=True
        ),
    )


def _build_weights(
    signals: pd.DataFrame,
    *,
    factors: list[str],
    method: str,
    labels: pd.DataFrame | None,
    lookback: int,
    min_history: int,
) -> pd.DataFrame:
    dates = pd.Index(pd.to_datetime(signals["date"]).drop_duplicates().sort_values())
    if method == "equal":
        weight = 1.0 / len(factors)
        equal_rows: list[dict[str, object]] = []
        for date in dates:
            for factor in factors:
                equal_rows.append(
                    {
                        "date": pd.Timestamp(date),
                        "factor": factor,
                        "weight": weight,
                    }
                )
        return pd.DataFrame(equal_rows)

    if labels is None:
        raise ValueError(f"labels is required when method={method!r}")
    _validate_canonical(labels, table_name="labels")

    ic_hist = _ic_history(signals, labels, factors=factors)
    rows: list[dict[str, object]] = []
    for date in dates:
        score: dict[str, float] = {}
        for factor in factors:
            h = ic_hist[(ic_hist["factor"] == factor) & (ic_hist["date"] < pd.Timestamp(date))]
            h = h.sort_values("date", kind="mergesort").tail(lookback)
            if len(h) < min_history:
                score[factor] = np.nan
                continue
            mean = float(h["rank_ic"].mean())
            if method == "ic":
                score[factor] = max(mean, 0.0)
            else:
                std = float(h["rank_ic"].std(ddof=1))
                icir = mean / std if np.isfinite(std) and std > 0 else np.nan
                score[factor] = max(icir, 0.0) if np.isfinite(icir) else np.nan

        if all(pd.isna(v) for v in score.values()) or np.nansum(list(score.values())) <= 0:
            weight_map = {factor: 1.0 / len(factors) for factor in factors}
        else:
            total = float(np.nansum(list(score.values())))
            weight_map = {
                factor: (
                    0.0 if pd.isna(score[factor]) else float(score[factor] / total)
                )
                for factor in factors
            }
        for factor in factors:
            rows.append(
                {
                    "date": pd.Timestamp(date),
                    "factor": factor,
                    "weight": weight_map[factor],
                }
            )
    return pd.DataFrame(rows)


def _ic_history(
    signals: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    factors: list[str],
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for factor in factors:
        f = signals[signals["factor"].astype(str) == factor][list(FACTOR_OUTPUT_COLUMNS)].copy()
        ric = compute_rank_ic(f, labels)
        rows.append(ric[["date", "factor", "rank_ic"]])
    return pd.concat(rows, ignore_index=True)


def _zscore_series(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    if vals.notna().sum() < 2:
        return pd.Series(np.nan, index=series.index, dtype=float)
    mu = vals.mean()
    std = vals.std(ddof=0)
    if not np.isfinite(std) or std == 0:
        return pd.Series(0.0, index=series.index, dtype=float)
    return (vals - mu) / std


def _mean_abs_component(series: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(vals)
    if not finite.any():
        return float("nan")
    return float(np.abs(vals[finite]).mean())


def _validate_canonical(df: pd.DataFrame, *, table_name: str) -> None:
    missing = set(FACTOR_OUTPUT_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{table_name} missing required columns: {sorted(missing)}")
