from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from alpha_lab.interfaces import FACTOR_OUTPUT_COLUMNS

_REQUIRED_COLS = {"date", "asset", "close"}
UNIFIED_LABEL_COLUMNS: tuple[str, ...] = (
    "date",
    "asset",
    "label_name",
    "label_type",
    "label_value",
    "event_start",
    "event_end",
    "trigger",
    "realized_horizon",
    "confidence",
)


@dataclass(frozen=True)
class LabelResult:
    """Unified label result with typed metadata."""

    labels: pd.DataFrame
    metadata: dict[str, object]


def forward_return(df: pd.DataFrame, *, horizon: int = 1) -> pd.DataFrame:
    """Compute forward returns using each asset's own ordered history.

    The label at date ``t`` is defined as ``close[t + horizon] / close[t] - 1``,
    where ``t + horizon`` is measured in row count within each asset's own
    sorted history. This preserves timestamp discipline: the label is stored at
    timestamp ``t`` for later evaluation against features observed at ``t``,
    while the value itself depends only on strictly future prices.

    Parameters
    ----------
    df:
        Long-form price table with columns ``[date, asset, close]``. Rows need
        not be sorted. Duplicate ``(date, asset)`` pairs are rejected.
        Non-positive prices (<= 0) are treated as missing for the return
        calculation.
    horizon:
        Number of future per-asset rows used in the forward return. Must be a
        positive integer.

    Returns
    -------
    pd.DataFrame
        Canonical long-form output with columns ``[date, asset, factor, value]``.
        ``factor`` is set to ``forward_return_{horizon}``.
    """
    if df.empty:
        return pd.DataFrame(columns=FACTOR_OUTPUT_COLUMNS)

    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        raise KeyError(f"Input DataFrame is missing required columns: {missing}")

    if horizon <= 0:
        raise ValueError("'horizon' must be a positive integer")

    if df["date"].isna().any():
        raise ValueError("Input 'date' column contains NaN/NaT values.")

    if df["asset"].isna().any():
        raise ValueError("Input 'asset' column contains NaN values.")

    dupes = df.duplicated(subset=["date", "asset"])
    if dupes.any():
        raise ValueError(f"Duplicate (date, asset) pairs found:\n{df[dupes][['date', 'asset']]}")

    df_copy = df.copy()
    df_copy["date"] = pd.to_datetime(df_copy["date"])
    df_copy = df_copy.sort_values(["asset", "date"]).reset_index(drop=True)

    close = df_copy["close"].where(df_copy["close"] > 0)
    next_close = close.groupby(df_copy["asset"], sort=False).shift(-horizon)
    value = next_close.div(close).sub(1.0)

    result = df_copy[["date", "asset"]].copy()
    result["factor"] = f"forward_return_{horizon}"
    result["value"] = value
    result = result.loc[:, FACTOR_OUTPUT_COLUMNS].reset_index(drop=True)

    return result


def regression_forward_label(
    df: pd.DataFrame,
    *,
    horizon: int = 1,
    label_name: str | None = None,
) -> LabelResult:
    """Return a unified-schema regression label based on forward return."""
    canonical = forward_return(df, horizon=horizon)
    resolved_name = label_name or f"forward_return_{horizon}"
    out = canonical[["date", "asset", "value"]].copy()
    out = out.rename(columns={"value": "label_value"})
    out["label_name"] = resolved_name
    out["label_type"] = "regression"
    out["event_start"] = out["date"]
    out["event_end"] = _event_end_map(df, horizon=horizon)
    out["trigger"] = "horizon_end"
    out["realized_horizon"] = horizon
    out["confidence"] = np.nan
    out = _finalise_unified_label_table(out)
    return LabelResult(
        labels=out,
        metadata={
            "label_name": resolved_name,
            "label_type": "regression",
            "horizon": horizon,
            "schema_version": "1.0.0",
        },
    )


def rankpct_label(
    df: pd.DataFrame,
    *,
    horizon: int = 1,
    label_name: str | None = None,
) -> LabelResult:
    """Cross-sectional rank-percentile label for relative return prediction."""
    base = regression_forward_label(df, horizon=horizon)
    out = base.labels.copy()
    out["label_name"] = label_name or f"rankpct_forward_return_{horizon}"
    out["label_type"] = "ranking"
    out["label_value"] = (
        out.groupby("date", sort=True)["label_value"]
        .rank(method="average", pct=True, na_option="keep")
    )
    out["trigger"] = "cross_section_rank"
    out["confidence"] = (out["label_value"] - 0.5).abs()
    out = _finalise_unified_label_table(out)
    return LabelResult(
        labels=out,
        metadata={
            "label_name": str(out["label_name"].iloc[0]) if not out.empty else "unknown",
            "label_type": "ranking",
            "horizon": horizon,
            "schema_version": "1.0.0",
        },
    )


def triple_barrier_labels(
    df: pd.DataFrame,
    *,
    horizon: int,
    pt_mult: float = 1.0,
    sl_mult: float = 1.0,
    volatility_lookback: int = 20,
    label_name: str = "triple_barrier",
) -> LabelResult:
    """Generate event labels via triple barrier method.

    label_value:
    - +1: upper barrier first
    - -1: lower barrier first
    -  0: vertical barrier (timeout)
    """
    if horizon <= 0:
        raise ValueError("horizon must be > 0")
    if pt_mult <= 0 or sl_mult <= 0:
        raise ValueError("pt_mult and sl_mult must be > 0")
    if volatility_lookback <= 1:
        raise ValueError("volatility_lookback must be > 1")

    _validate_label_input(df)
    data = _sorted_prices(df)

    rows: list[dict[str, object]] = []
    for asset, group in data.groupby("asset", sort=False):
        g = group.reset_index(drop=True)
        close = g["close"].to_numpy(dtype=float)
        dates = pd.to_datetime(g["date"]).to_numpy()
        ret = pd.Series(close).pct_change()
        sigma = ret.rolling(volatility_lookback, min_periods=2).std(ddof=0).to_numpy()

        for i in range(len(g)):
            event_start = pd.Timestamp(dates[i])
            row = {
                "date": event_start,
                "asset": asset,
                "label_name": label_name,
                "label_type": "event_classification",
                "label_value": np.nan,
                "event_start": event_start,
                "event_end": pd.NaT,
                "trigger": "insufficient_forward_window",
                "realized_horizon": np.nan,
                "confidence": np.nan,
            }
            if i + horizon >= len(g):
                rows.append(row)
                continue
            vol = sigma[i]
            if np.isnan(vol) or vol <= 0:
                rows.append(row)
                continue

            upper = close[i] * (1.0 + pt_mult * vol)
            lower = close[i] * (1.0 - sl_mult * vol)
            path = close[i + 1 : i + horizon + 1]

            trigger_idx: int | None = None
            trigger = "vertical_barrier"
            label_value = 0.0
            for j, px in enumerate(path, start=1):
                if px >= upper:
                    trigger_idx = j
                    trigger = "upper_barrier"
                    label_value = 1.0
                    break
                if px <= lower:
                    trigger_idx = j
                    trigger = "lower_barrier"
                    label_value = -1.0
                    break

            if trigger_idx is None:
                trigger_idx = horizon
            end_idx = i + trigger_idx
            event_end = pd.Timestamp(dates[end_idx])
            confidence = abs(close[end_idx] / close[i] - 1.0) / max(vol, 1e-12)

            row.update(
                {
                    "label_value": label_value,
                    "event_end": event_end,
                    "trigger": trigger,
                    "realized_horizon": trigger_idx,
                    "confidence": confidence,
                }
            )
            rows.append(row)

    out = pd.DataFrame(rows, columns=list(UNIFIED_LABEL_COLUMNS))
    out = _finalise_unified_label_table(out)
    return LabelResult(
        labels=out,
        metadata={
            "label_name": label_name,
            "label_type": "event_classification",
            "horizon": horizon,
            "pt_mult": pt_mult,
            "sl_mult": sl_mult,
            "volatility_lookback": volatility_lookback,
            "schema_version": "1.0.0",
        },
    )


def trend_scanning_labels(
    df: pd.DataFrame,
    *,
    min_horizon: int = 2,
    max_horizon: int = 20,
    label_name: str = "trend_scan",
) -> LabelResult:
    """Generate labels by selecting the horizon with strongest trend t-stat."""
    if min_horizon < 2:
        raise ValueError("min_horizon must be >= 2")
    if max_horizon < min_horizon:
        raise ValueError("max_horizon must be >= min_horizon")

    _validate_label_input(df)
    data = _sorted_prices(df)
    rows: list[dict[str, object]] = []

    for asset, group in data.groupby("asset", sort=False):
        g = group.reset_index(drop=True)
        log_close = np.log(g["close"].to_numpy(dtype=float))
        dates = pd.to_datetime(g["date"]).to_numpy()
        for i in range(len(g)):
            row = {
                "date": pd.Timestamp(dates[i]),
                "asset": asset,
                "label_name": label_name,
                "label_type": "event_classification",
                "label_value": np.nan,
                "event_start": pd.Timestamp(dates[i]),
                "event_end": pd.NaT,
                "trigger": "insufficient_forward_window",
                "realized_horizon": np.nan,
                "confidence": np.nan,
            }

            best_h = None
            best_t = None
            for h in range(min_horizon, max_horizon + 1):
                end = i + h
                if end >= len(g):
                    break
                t_stat = _linear_trend_tstat(log_close[i : end + 1])
                if best_t is None or abs(t_stat) > abs(best_t):
                    best_t = t_stat
                    best_h = h

            if best_h is None or best_t is None:
                rows.append(row)
                continue

            row.update(
                {
                    "label_value": float(np.sign(best_t)),
                    "event_end": pd.Timestamp(dates[i + best_h]),
                    "trigger": "trend_scan",
                    "realized_horizon": best_h,
                    "confidence": abs(float(best_t)),
                }
            )
            rows.append(row)

    out = pd.DataFrame(rows, columns=list(UNIFIED_LABEL_COLUMNS))
    out = _finalise_unified_label_table(out)
    return LabelResult(
        labels=out,
        metadata={
            "label_name": label_name,
            "label_type": "event_classification",
            "min_horizon": min_horizon,
            "max_horizon": max_horizon,
            "schema_version": "1.0.0",
        },
    )


def validate_unified_label_table(df: pd.DataFrame) -> None:
    missing = set(UNIFIED_LABEL_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"unified label table missing columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("unified label table is empty")
    date = pd.to_datetime(df["date"], errors="coerce")
    if date.isna().any():
        raise ValueError("unified label table has invalid date values")
    if df["asset"].isna().any():
        raise ValueError("unified label table has null asset values")
    if (
        df["label_name"].isna().any()
        or (df["label_name"].astype(str).str.strip() == "").any()
    ):
        raise ValueError("unified label table has invalid label_name values")
    dupes = df.duplicated(subset=["date", "asset", "label_name"])
    if dupes.any():
        raise ValueError("unified label table has duplicate (date, asset, label_name)")


def _validate_label_input(df: pd.DataFrame) -> None:
    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        raise KeyError(f"Input DataFrame is missing required columns: {missing}")
    if df.empty:
        raise ValueError("input DataFrame is empty")
    if df.duplicated(subset=["date", "asset"]).any():
        raise ValueError("input has duplicate (date, asset) rows")
    if (df["close"] <= 0).any():
        raise ValueError("close must be > 0 for advanced labeling")


def _sorted_prices(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    if out["date"].isna().any():
        raise ValueError("date contains invalid values")
    return out.sort_values(["asset", "date"], kind="mergesort").reset_index(drop=True)


def _event_end_map(df: pd.DataFrame, *, horizon: int) -> pd.Series:
    data = _sorted_prices(df)
    end = (
        data.groupby("asset", sort=False)["date"]
        .shift(-horizon)
        .reset_index(drop=True)
    )
    key = data[["date", "asset"]].copy()
    key["event_end"] = end
    return key["event_end"]


def _linear_trend_tstat(values: np.ndarray) -> float:
    n = len(values)
    if n < 3:
        return float("nan")
    x = np.arange(n, dtype=float)
    x_mean = float(x.mean())
    y_mean = float(values.mean())
    x_centered = x - x_mean
    y_centered = values - y_mean
    denom = float(np.sum(x_centered**2))
    if denom == 0:
        return float("nan")
    slope = float(np.sum(x_centered * y_centered) / denom)
    fitted = slope * x + (y_mean - slope * x_mean)
    residual = values - fitted
    s2 = float(np.sum(residual**2) / max(n - 2, 1))
    se = np.sqrt(s2 / denom) if denom > 0 else np.nan
    if se == 0 or np.isnan(se):
        return float("nan")
    return slope / float(se)


def _finalise_unified_label_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out[list(UNIFIED_LABEL_COLUMNS)]
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["event_start"] = pd.to_datetime(out["event_start"], errors="coerce")
    out["event_end"] = pd.to_datetime(out["event_end"], errors="coerce")
    out = out.sort_values(["date", "asset", "label_name"], kind="mergesort").reset_index(drop=True)
    return out
