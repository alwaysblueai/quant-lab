from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from alpha_lab.exceptions import AlphaLabDataError
from alpha_lab.frame_utils import require_columns
from alpha_lab.interfaces import validate_factor_output

_STYLE_FACTORS: tuple[str, ...] = ("size", "value", "momentum", "volatility", "beta")


@dataclass(frozen=True)
class BarraExposures:
    """Barra-like cross-sectional exposure panel."""

    exposures: pd.DataFrame
    style_factors: tuple[str, ...]
    industry_factors: tuple[str, ...]


def build_barra_exposures(
    prices_df: pd.DataFrame,
    daily_basic_df: pd.DataFrame,
    industry_col: str = "industry",
) -> BarraExposures:
    """Build a Barra-style exposure matrix from daily prices and fundamentals."""
    require_columns(prices_df, ("date", "asset", "close"), "prices_df")
    require_columns(
        daily_basic_df,
        ("date", "asset", "circ_mv", "pb", industry_col),
        "daily_basic_df",
    )

    prices = prices_df[["date", "asset", "close"]].copy()
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
    prices["close"] = pd.to_numeric(prices["close"], errors="coerce")
    prices = prices.dropna(subset=["date", "asset", "close"]).sort_values(
        ["asset", "date"], kind="mergesort"
    )

    prices["ret_1d"] = prices.groupby("asset", sort=False)["close"].pct_change()
    prices["momentum"] = prices.groupby("asset", sort=False)["close"].pct_change(
        252
    ) - prices.groupby("asset", sort=False)["close"].pct_change(21)
    prices["volatility"] = (
        prices.groupby("asset", sort=False)["ret_1d"]
        .rolling(window=60, min_periods=20)
        .std(ddof=1)
        .reset_index(level=0, drop=True)
    )

    market_ret = prices.groupby("date", sort=True)["ret_1d"].mean().rename("market_ret")
    prices = prices.merge(market_ret, on="date", how="left")
    prices["beta"] = _rolling_beta(prices)

    basic = daily_basic_df[["date", "asset", "circ_mv", "pb", industry_col]].copy()
    basic["date"] = pd.to_datetime(basic["date"], errors="coerce")
    basic["circ_mv"] = pd.to_numeric(basic["circ_mv"], errors="coerce")
    basic["pb"] = pd.to_numeric(basic["pb"], errors="coerce")
    basic["size"] = np.log(basic["circ_mv"].where(basic["circ_mv"] > 0.0))
    basic["value"] = 1.0 / basic["pb"].replace(0.0, np.nan)
    basic[industry_col] = basic[industry_col].fillna("Unknown").astype(str)

    base = basic.merge(
        prices[["date", "asset", "momentum", "volatility", "beta"]],
        on=["date", "asset"],
        how="inner",
    )
    if base.empty:
        raise AlphaLabDataError(
            "no overlap between prices_df and daily_basic_df for Barra exposures"
        )

    dummies = pd.get_dummies(_sanitize_industry(base[industry_col]), prefix="industry", dtype=float)
    industry_factors = tuple(str(col) for col in dummies.columns)

    exposures = pd.concat(
        [
            base[["date", "asset", "circ_mv"] + list(_STYLE_FACTORS)].reset_index(drop=True),
            dummies.reset_index(drop=True),
        ],
        axis=1,
    )
    exposures = _cross_sectional_zscore(
        exposures,
        factor_cols=list(_STYLE_FACTORS) + list(industry_factors),
    )
    exposures = exposures.sort_values(["date", "asset"], kind="mergesort").reset_index(drop=True)

    return BarraExposures(
        exposures=exposures,
        style_factors=_STYLE_FACTORS,
        industry_factors=industry_factors,
    )


def estimate_factor_returns(
    exposures: BarraExposures,
    returns_df: pd.DataFrame,
    weight_col: str = "circ_mv",
) -> pd.DataFrame:
    """Estimate per-date factor returns via weighted cross-sectional regression."""
    expo = exposures.exposures.copy()
    require_columns(expo, ("date", "asset"), "exposures.exposures")
    factor_cols = list(exposures.style_factors) + list(exposures.industry_factors)
    if not factor_cols:
        raise AlphaLabDataError("Barra exposures must include at least one factor column")
    require_columns(expo, tuple(factor_cols), "exposures.exposures")

    return_col = "value" if "value" in returns_df.columns else "return"
    require_columns(returns_df, ("date", "asset", return_col), "returns_df")

    rets = returns_df[["date", "asset", return_col]].copy()
    rets["date"] = pd.to_datetime(rets["date"], errors="coerce")
    rets["asset_return"] = pd.to_numeric(rets[return_col], errors="coerce")
    rets = rets.drop(columns=[return_col])

    expo["date"] = pd.to_datetime(expo["date"], errors="coerce")
    for col in factor_cols:
        expo[col] = pd.to_numeric(expo[col], errors="coerce")

    if weight_col in expo.columns:
        expo["_w"] = pd.to_numeric(expo[weight_col], errors="coerce").clip(lower=0.0)
    else:
        expo["_w"] = 1.0

    merged = expo.merge(rets, on=["date", "asset"], how="inner")
    if merged.empty:
        return pd.DataFrame(columns=["date", "factor", "factor_return"])

    rows: list[dict[str, object]] = []
    for date, group in merged.groupby("date", sort=True):
        valid = group.dropna(subset=factor_cols + ["asset_return", "_w"])
        if valid.empty:
            rows.extend(
                {"date": date, "factor": factor, "factor_return": float("nan")}
                for factor in factor_cols
            )
            continue

        weights = valid["_w"].to_numpy(dtype=float)
        if np.all(weights <= 0.0):
            weights = np.ones(len(valid), dtype=float)
        sqrt_w = np.sqrt(np.clip(weights, 1e-12, None))

        x_mat = valid[factor_cols].to_numpy(dtype=float)
        y_vec = valid["asset_return"].to_numpy(dtype=float)
        xw = x_mat * sqrt_w[:, None]
        yw = y_vec * sqrt_w

        try:
            coef, *_ = np.linalg.lstsq(xw, yw, rcond=None)
        except np.linalg.LinAlgError:
            coef = np.full(len(factor_cols), np.nan, dtype=float)

        rows.extend(
            {"date": date, "factor": factor_name, "factor_return": float(coef[idx])}
            for idx, factor_name in enumerate(factor_cols)
        )

    return (
        pd.DataFrame(rows, columns=["date", "factor", "factor_return"])
        .sort_values(
            ["date", "factor"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


def extract_pure_alpha(
    alpha_df: pd.DataFrame,
    exposures: BarraExposures,
    weight_col: str = "circ_mv",
) -> pd.DataFrame:
    """Remove Barra-factor exposure from raw alpha cross-sections."""
    validate_factor_output(alpha_df)
    expo = exposures.exposures.copy()
    require_columns(expo, ("date", "asset"), "exposures.exposures")

    factor_cols = list(exposures.style_factors) + list(exposures.industry_factors)
    if not factor_cols:
        raise AlphaLabDataError("Barra exposures must include factor columns")
    require_columns(expo, tuple(factor_cols), "exposures.exposures")

    alpha = alpha_df[["date", "asset", "value"]].copy()
    alpha["date"] = pd.to_datetime(alpha["date"], errors="coerce")
    alpha["alpha_raw"] = pd.to_numeric(alpha["value"], errors="coerce")
    alpha = alpha.drop(columns=["value"])

    expo["date"] = pd.to_datetime(expo["date"], errors="coerce")
    for col in factor_cols:
        expo[col] = pd.to_numeric(expo[col], errors="coerce")
    if weight_col in expo.columns:
        expo["_w"] = pd.to_numeric(expo[weight_col], errors="coerce").clip(lower=0.0)
    else:
        expo["_w"] = 1.0

    merged = alpha.merge(
        expo[["date", "asset", "_w", *factor_cols]],
        on=["date", "asset"],
        how="inner",
    )
    if merged.empty:
        return pd.DataFrame(columns=["date", "asset", "factor", "value"])

    rows: list[dict[str, object]] = []
    for date, group in merged.groupby("date", sort=True):
        valid = group.dropna(subset=["alpha_raw", "_w", *factor_cols])
        if valid.empty:
            continue

        x_mat = valid[factor_cols].to_numpy(dtype=float)
        y_vec = valid["alpha_raw"].to_numpy(dtype=float)
        weights = valid["_w"].to_numpy(dtype=float)
        if np.all(weights <= 0.0):
            weights = np.ones(len(valid), dtype=float)
        sqrt_w = np.sqrt(np.clip(weights, 1e-12, None))

        xw = x_mat * sqrt_w[:, None]
        yw = y_vec * sqrt_w
        try:
            beta_hat, *_ = np.linalg.lstsq(xw, yw, rcond=None)
        except np.linalg.LinAlgError:
            beta_hat = np.zeros(len(factor_cols), dtype=float)

        residual = y_vec - x_mat @ beta_hat
        rows.extend(
            {
                "date": date,
                "asset": str(asset),
                "factor": "pure_alpha",
                "value": float(value),
            }
            for asset, value in zip(valid["asset"], residual, strict=False)
        )

    return (
        pd.DataFrame(rows, columns=["date", "asset", "factor", "value"])
        .sort_values(
            ["date", "asset"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


def _rolling_beta(prices: pd.DataFrame) -> pd.Series:
    parts: list[pd.Series] = []
    for _, group in prices.groupby("asset", sort=False):
        cov = group["ret_1d"].rolling(window=252, min_periods=60).cov(group["market_ret"])
        var = group["market_ret"].rolling(window=252, min_periods=60).var(ddof=1)
        beta = cov / var.replace(0.0, np.nan)
        beta.index = group.index
        parts.append(beta)
    if not parts:
        return pd.Series(dtype=float)
    out = pd.concat(parts).sort_index()
    out.name = "beta"
    return out


def _sanitize_industry(series: pd.Series) -> pd.Series:
    out = series.astype(str).str.strip()
    out = out.replace("", "Unknown")
    return out.str.replace(r"[^A-Za-z0-9_]+", "_", regex=True)


def _cross_sectional_zscore(frame: pd.DataFrame, factor_cols: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for col in factor_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    means = out.groupby("date", sort=False)[factor_cols].transform("mean")
    stds = out.groupby("date", sort=False)[factor_cols].transform("std").replace(0.0, np.nan)
    out[factor_cols] = (out[factor_cols] - means) / stds
    out[factor_cols] = out[factor_cols].fillna(0.0)
    return out


