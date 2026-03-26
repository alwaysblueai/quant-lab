from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def _safe_nanquantile(series: pd.Series, q: float) -> float:
    vals = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan")
    return float(np.quantile(vals, q))


@dataclass(frozen=True)
class ResearchCostResult:
    """Layered research friction diagnostics (not execution simulation)."""

    per_trade: pd.DataFrame
    by_date: pd.DataFrame
    summary: pd.DataFrame


def flat_fee_cost(trade_dollar: float, *, fee_bps: float) -> float:
    """Linear flat-fee proxy cost in dollars."""
    if fee_bps < 0:
        raise ValueError("fee_bps must be >= 0")
    return abs(float(trade_dollar)) * float(fee_bps) * 1e-4


def spread_proxy_cost(trade_dollar: float, *, spread_bps: float) -> float:
    """Half-spread proxy cost in dollars for one-way trade."""
    if spread_bps < 0:
        raise ValueError("spread_bps must be >= 0")
    return abs(float(trade_dollar)) * float(spread_bps) * 1e-4 * 0.5


def sqrt_impact_cost(
    trade_dollar: float,
    *,
    adv_dollar: float,
    daily_volatility: float,
    eta: float = 0.1,
) -> float:
    """Square-root impact proxy in dollars."""
    if adv_dollar <= 0:
        return float("inf")
    if daily_volatility < 0:
        raise ValueError("daily_volatility must be >= 0")
    if eta < 0:
        raise ValueError("eta must be >= 0")
    participation = abs(float(trade_dollar)) / float(adv_dollar)
    impact_rate = float(eta) * float(daily_volatility) * np.sqrt(max(participation, 0.0))
    return float(abs(float(trade_dollar)) * float(impact_rate))


def layered_research_costs(
    trades: pd.DataFrame,
    *,
    flat_fee_bps: float = 1.0,
    spread_bps: float = 5.0,
    impact_eta: float = 0.1,
    trade_col: str = "trade_dollar",
    adv_col: str = "adv_dollar",
    vol_col: str = "daily_volatility",
) -> ResearchCostResult:
    """Apply layered friction proxies to a trade plan table."""
    for col in ("date", "asset", trade_col, adv_col, vol_col):
        if col not in trades.columns:
            raise ValueError(f"trades missing required column {col!r}")

    out = trades.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    if out["date"].isna().any():
        raise ValueError("trades contains invalid date values")
    out[trade_col] = pd.to_numeric(out[trade_col], errors="coerce")
    out[adv_col] = pd.to_numeric(out[adv_col], errors="coerce")
    out[vol_col] = pd.to_numeric(out[vol_col], errors="coerce")
    if out[[trade_col, adv_col, vol_col]].isna().any().any():
        raise ValueError("trades contains invalid numeric values")

    out["cost_flat_fee"] = out[trade_col].apply(lambda x: flat_fee_cost(x, fee_bps=flat_fee_bps))
    out["cost_spread_proxy"] = out[trade_col].apply(
        lambda x: spread_proxy_cost(x, spread_bps=spread_bps)
    )
    out["cost_impact_proxy"] = out.apply(
        lambda row: sqrt_impact_cost(
            row[trade_col],
            adv_dollar=float(row[adv_col]),
            daily_volatility=float(row[vol_col]),
            eta=impact_eta,
        ),
        axis=1,
    )
    out["cost_total"] = out["cost_flat_fee"] + out["cost_spread_proxy"] + out["cost_impact_proxy"]
    abs_trade = out[trade_col].abs()
    out["cost_total_bps"] = np.divide(
        out["cost_total"],
        abs_trade.replace(0, np.nan),
    ) * 1e4
    out["cost_total_bps"] = out["cost_total_bps"].replace([np.inf, -np.inf], np.nan)

    by_date = (
        out.groupby("date", sort=True)
        .agg(
            gross_trade_dollar=(trade_col, lambda x: float(np.abs(x).sum())),
            total_cost_dollar=("cost_total", "sum"),
            mean_cost_bps=("cost_total_bps", "mean"),
            p95_cost_bps=("cost_total_bps", lambda x: _safe_nanquantile(x, 0.95)),
        )
        .reset_index()
    )
    summary = pd.DataFrame(
        [
            {
                "n_trades": int(len(out)),
                "total_cost_dollar": float(out["cost_total"].sum()),
                "mean_cost_bps": float(out["cost_total_bps"].mean()),
                "max_cost_bps": float(out["cost_total_bps"].max()),
            }
        ]
    )
    return ResearchCostResult(
        per_trade=out.sort_values(["date", "asset"], kind="mergesort").reset_index(drop=True),
        by_date=by_date,
        summary=summary,
    )
