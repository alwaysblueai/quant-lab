from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CapacityDiagnosticsResult:
    """Research-grade liquidity/capacity diagnostics (non-execution)."""

    adv_penetration: pd.DataFrame
    turnover_liquidity: pd.DataFrame
    concentration_flags: pd.DataFrame
    warnings: pd.DataFrame


def run_capacity_diagnostics(
    trades: pd.DataFrame,
    *,
    portfolio_value: float,
    max_adv_participation: float = 0.05,
    concentration_weight_threshold: float = 0.05,
) -> CapacityDiagnosticsResult:
    """Compute capacity warning layers from planned trade table."""
    if portfolio_value <= 0:
        raise ValueError("portfolio_value must be > 0")
    if max_adv_participation <= 0:
        raise ValueError("max_adv_participation must be > 0")
    if concentration_weight_threshold <= 0:
        raise ValueError("concentration_weight_threshold must be > 0")

    pen = adv_penetration_table(trades, max_adv_participation=max_adv_participation)
    stress = turnover_liquidity_stress(pen, portfolio_value=portfolio_value)
    conc = concentration_liquidity_flags(
        pen,
        concentration_weight_threshold=concentration_weight_threshold,
        max_adv_participation=max_adv_participation,
    )
    warn = capacity_warning_summary(pen, stress, conc)
    return CapacityDiagnosticsResult(
        adv_penetration=pen,
        turnover_liquidity=stress,
        concentration_flags=conc,
        warnings=warn,
    )


def adv_penetration_table(
    trades: pd.DataFrame,
    *,
    max_adv_participation: float = 0.05,
) -> pd.DataFrame:
    """Per-trade ADV penetration table."""
    required = {"date", "asset", "trade_dollar", "adv_dollar"}
    missing = required - set(trades.columns)
    if missing:
        raise ValueError(f"trades missing required columns: {sorted(missing)}")
    if max_adv_participation <= 0:
        raise ValueError("max_adv_participation must be > 0")

    out = trades.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    if out["date"].isna().any():
        raise ValueError("trades contains invalid date values")
    out["trade_dollar"] = pd.to_numeric(out["trade_dollar"], errors="coerce")
    out["adv_dollar"] = pd.to_numeric(out["adv_dollar"], errors="coerce")
    if out["trade_dollar"].isna().any() or out["adv_dollar"].isna().any():
        raise ValueError("trades contains invalid numeric values")

    out["adv_participation"] = np.divide(
        out["trade_dollar"].abs(),
        out["adv_dollar"].replace(0, np.nan),
    )
    out["adv_participation"] = out["adv_participation"].replace([np.inf, -np.inf], np.nan)
    out["adv_participation"] = out["adv_participation"].fillna(np.inf)
    out["adv_limit_flag"] = out["adv_participation"] > max_adv_participation
    return out.sort_values(["date", "asset"], kind="mergesort").reset_index(drop=True)


def turnover_liquidity_stress(
    penetration: pd.DataFrame,
    *,
    portfolio_value: float,
) -> pd.DataFrame:
    """Date-level turnover x liquidity stress summary."""
    if portfolio_value <= 0:
        raise ValueError("portfolio_value must be > 0")
    required = {"date", "trade_dollar", "adv_participation"}
    missing = required - set(penetration.columns)
    if missing:
        raise ValueError(f"penetration missing required columns: {sorted(missing)}")

    grouped = penetration.groupby("date", sort=True).agg(
        gross_trade_dollar=("trade_dollar", lambda x: float(np.abs(x).sum())),
        mean_adv_participation=("adv_participation", "mean"),
        p95_adv_participation=("adv_participation", lambda x: float(np.nanquantile(x, 0.95))),
    )
    grouped["turnover"] = grouped["gross_trade_dollar"] / float(portfolio_value)
    grouped["turnover_x_liquidity"] = grouped["turnover"] * grouped["mean_adv_participation"]
    return grouped.reset_index()


def concentration_liquidity_flags(
    penetration: pd.DataFrame,
    *,
    concentration_weight_threshold: float,
    max_adv_participation: float,
) -> pd.DataFrame:
    """Flag concentrated names that are also liquidity-stressed."""
    required = {"date", "asset", "adv_participation"}
    missing = required - set(penetration.columns)
    if missing:
        raise ValueError(f"penetration missing required columns: {sorted(missing)}")

    out = penetration.copy()
    if "target_weight" in out.columns:
        out["target_weight"] = pd.to_numeric(out["target_weight"], errors="coerce")
    else:
        out["target_weight"] = np.nan
    conc = out["target_weight"].abs() >= concentration_weight_threshold
    liq = out["adv_participation"] > max_adv_participation
    out["concentration_liquidity_flag"] = conc & liq
    return out[out["concentration_liquidity_flag"]].copy().reset_index(drop=True)


def capacity_warning_summary(
    penetration: pd.DataFrame,
    stress: pd.DataFrame,
    concentration_flags: pd.DataFrame,
) -> pd.DataFrame:
    """Compact warning table for audit/reporting."""
    total_rows = len(penetration)
    n_adv_limit = int(penetration["adv_limit_flag"].sum()) if "adv_limit_flag" in penetration else 0
    ratio_adv = n_adv_limit / total_rows if total_rows > 0 else np.nan
    max_stress = float(stress["turnover_x_liquidity"].max()) if not stress.empty else np.nan
    n_conc = int(len(concentration_flags))
    return pd.DataFrame(
        [
            {
                "n_rows": total_rows,
                "n_adv_limit_flags": n_adv_limit,
                "adv_limit_flag_ratio": ratio_adv,
                "max_turnover_x_liquidity": max_stress,
                "n_concentration_liquidity_flags": n_conc,
            }
        ]
    )
