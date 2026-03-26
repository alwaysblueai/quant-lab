from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ExposureAuditResult:
    """Research exposure-audit output."""

    industry_exposure: pd.DataFrame
    style_exposure: pd.DataFrame
    summary: pd.DataFrame


def run_exposure_audit(
    weights: pd.DataFrame,
    exposures: pd.DataFrame,
    *,
    weight_col: str = "weight",
    industry_col: str | None = "industry",
    size_col: str | None = "size_exposure",
    beta_col: str | None = "beta_exposure",
    style_cols: list[str] | None = None,
) -> ExposureAuditResult:
    """Compute industry/size/beta/style weighted exposure diagnostics."""
    for col in ("date", "asset", weight_col):
        if col not in weights.columns:
            raise ValueError(f"weights missing required column {col!r}")
    for col in ("date", "asset"):
        if col not in exposures.columns:
            raise ValueError(f"exposures missing required column {col!r}")

    w = weights.copy()
    x = exposures.copy()
    w["date"] = pd.to_datetime(w["date"], errors="coerce")
    x["date"] = pd.to_datetime(x["date"], errors="coerce")
    merged = w.merge(x, on=["date", "asset"], how="left", validate="many_to_one")
    merged[weight_col] = pd.to_numeric(merged[weight_col], errors="coerce")
    if merged[weight_col].isna().any():
        raise ValueError("weights contains invalid numeric values")

    industry = _industry_exposure(merged, weight_col=weight_col, industry_col=industry_col)
    styles = _style_exposure(
        merged,
        weight_col=weight_col,
        size_col=size_col,
        beta_col=beta_col,
        style_cols=style_cols,
    )
    summary = _exposure_summary(industry, styles)
    return ExposureAuditResult(industry_exposure=industry, style_exposure=styles, summary=summary)


def _industry_exposure(
    merged: pd.DataFrame,
    *,
    weight_col: str,
    industry_col: str | None,
) -> pd.DataFrame:
    if industry_col is None or industry_col not in merged.columns:
        return pd.DataFrame(columns=["date", "industry", "net_weight"])
    out = (
        merged.groupby(["date", industry_col], sort=True)[weight_col]
        .sum()
        .rename("net_weight")
        .reset_index()
        .rename(columns={industry_col: "industry"})
    )
    return out.sort_values(["date", "industry"], kind="mergesort").reset_index(drop=True)


def _style_exposure(
    merged: pd.DataFrame,
    *,
    weight_col: str,
    size_col: str | None,
    beta_col: str | None,
    style_cols: list[str] | None,
) -> pd.DataFrame:
    cols: list[str] = []
    if size_col is not None and size_col in merged.columns:
        cols.append(size_col)
    if beta_col is not None and beta_col in merged.columns:
        cols.append(beta_col)
    if style_cols is not None:
        cols.extend([c for c in style_cols if c in merged.columns])
    cols = sorted(set(cols))
    if not cols:
        return pd.DataFrame(columns=["date", "exposure_name", "weighted_exposure"])

    rows: list[dict[str, object]] = []
    for date, group in merged.groupby("date", sort=True):
        w = group[weight_col].to_numpy(dtype=float)
        for col in cols:
            val = pd.to_numeric(group[col], errors="coerce").to_numpy(dtype=float)
            valid = np.isfinite(val) & np.isfinite(w)
            if valid.sum() == 0:
                exp = np.nan
            else:
                exp = float(np.nansum(w[valid] * val[valid]))
            rows.append(
                {
                    "date": pd.Timestamp(date),
                    "exposure_name": col,
                    "weighted_exposure": exp,
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["date", "exposure_name"],
        kind="mergesort",
    ).reset_index(drop=True)


def _exposure_summary(industry: pd.DataFrame, styles: pd.DataFrame) -> pd.DataFrame:
    max_ind = float(industry["net_weight"].abs().max()) if not industry.empty else np.nan
    out = {
        "max_abs_industry_exposure": max_ind,
        "n_industry_cells": int(len(industry)),
        "n_style_cells": int(len(styles)),
    }
    if styles.empty:
        out["max_abs_style_exposure"] = np.nan
        return pd.DataFrame([out])
    out["max_abs_style_exposure"] = float(styles["weighted_exposure"].abs().max())
    return pd.DataFrame([out])
