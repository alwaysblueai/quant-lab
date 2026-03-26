from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class NeutralizationResult:
    """Neutralized signal output with regression diagnostics."""

    data: pd.DataFrame
    diagnostics: pd.DataFrame
    coefficients: pd.DataFrame


def neutralize_signal(
    df: pd.DataFrame,
    *,
    value_col: str = "value",
    by: str = "date",
    size_col: str | None = None,
    industry_col: str | None = None,
    beta_col: str | None = None,
    min_obs: int = 20,
    ridge: float = 1e-8,
    output_col: str = "value_neutralized",
) -> NeutralizationResult:
    """Cross-sectional residual neutralization with explicit exposure controls."""
    if min_obs <= 0:
        raise ValueError("min_obs must be > 0")
    if ridge < 0:
        raise ValueError("ridge must be >= 0")
    if value_col not in df.columns:
        raise ValueError(f"df missing value_col {value_col!r}")
    if by not in df.columns:
        raise ValueError(f"df missing by column {by!r}")

    work = df.copy()
    exposure = pd.DataFrame(index=work.index)
    raw_exposure_names: list[str] = []

    if size_col is not None:
        _require_column(work, size_col)
        exposure["size_exposure"] = pd.to_numeric(work[size_col], errors="coerce")
        raw_exposure_names.append("size_exposure")
    if beta_col is not None:
        _require_column(work, beta_col)
        exposure["beta_exposure"] = pd.to_numeric(work[beta_col], errors="coerce")
        raw_exposure_names.append("beta_exposure")
    if industry_col is not None:
        _require_column(work, industry_col)
        dummies = pd.get_dummies(
            work[industry_col].astype("string"),
            prefix="industry",
            dummy_na=False,
            dtype=float,
        )
        if dummies.shape[1] > 0:
            # Drop one column for identifiability with intercept.
            dummies = dummies.iloc[:, 1:]
            exposure = pd.concat([exposure, dummies], axis=1)
            raw_exposure_names.extend(dummies.columns.tolist())

    if exposure.shape[1] == 0:
        raise ValueError("at least one exposure is required for neutralization")

    work = pd.concat([work, exposure], axis=1)
    coef_rows: list[dict[str, object]] = []
    resid = pd.Series(np.nan, index=work.index, dtype=float)

    for date, group in work.groupby(by, sort=True):
        y = pd.to_numeric(group[value_col], errors="coerce")
        x = group[exposure.columns].apply(pd.to_numeric, errors="coerce")
        valid = y.notna() & x.notna().all(axis=1)
        if int(valid.sum()) < max(min_obs, x.shape[1] + 2):
            continue

        yv = y[valid].to_numpy(dtype=float)
        xv = x.loc[valid].to_numpy(dtype=float)
        x_with_const = np.column_stack([np.ones(len(xv), dtype=float), xv])
        beta = _ridge_ols(x_with_const, yv, ridge=ridge)
        fitted = x_with_const @ beta
        residuals = yv - fitted
        resid.loc[group.index[valid]] = residuals

        row: dict[str, object] = {"date": pd.Timestamp(date), "intercept": float(beta[0])}
        for i, col in enumerate(exposure.columns, start=1):
            row[col] = float(beta[i])
        coef_rows.append(row)

    out = df.copy()
    out[output_col] = resid

    diag_frame = work[[by, *list(exposure.columns)]].copy()
    diag_frame["_raw_value"] = pd.to_numeric(df[value_col], errors="coerce")
    diag_frame["_neutralized_value"] = out[output_col]
    diag = exposure_diagnostics(
        diag_frame,
        value_before_col="_raw_value",
        value_after_col="_neutralized_value",
        exposure_cols=list(exposure.columns),
        by=by,
    )
    if coef_rows:
        coefs = pd.DataFrame(coef_rows).sort_values("date", kind="mergesort").reset_index(
            drop=True
        )
    else:
        coefs = pd.DataFrame(columns=["date", "intercept", *list(exposure.columns)])
    return NeutralizationResult(data=out, diagnostics=diag, coefficients=coefs)


def neutralize_size(
    df: pd.DataFrame,
    *,
    size_col: str,
    value_col: str = "value",
    by: str = "date",
    min_obs: int = 20,
    ridge: float = 1e-8,
    output_col: str = "value_neutralized",
) -> NeutralizationResult:
    return neutralize_signal(
        df,
        value_col=value_col,
        by=by,
        size_col=size_col,
        min_obs=min_obs,
        ridge=ridge,
        output_col=output_col,
    )


def neutralize_beta(
    df: pd.DataFrame,
    *,
    beta_col: str,
    value_col: str = "value",
    by: str = "date",
    min_obs: int = 20,
    ridge: float = 1e-8,
    output_col: str = "value_neutralized",
) -> NeutralizationResult:
    return neutralize_signal(
        df,
        value_col=value_col,
        by=by,
        beta_col=beta_col,
        min_obs=min_obs,
        ridge=ridge,
        output_col=output_col,
    )


def neutralize_industry(
    df: pd.DataFrame,
    *,
    industry_col: str,
    value_col: str = "value",
    by: str = "date",
    min_obs: int = 20,
    ridge: float = 1e-8,
    output_col: str = "value_neutralized",
) -> NeutralizationResult:
    return neutralize_signal(
        df,
        value_col=value_col,
        by=by,
        industry_col=industry_col,
        min_obs=min_obs,
        ridge=ridge,
        output_col=output_col,
    )


def exposure_diagnostics(
    df: pd.DataFrame,
    *,
    value_before_col: str,
    value_after_col: str,
    exposure_cols: list[str],
    by: str = "date",
) -> pd.DataFrame:
    """Mean absolute exposure correlation before/after neutralization."""
    for col in (value_before_col, value_after_col, by):
        _require_column(df, col)
    for col in exposure_cols:
        _require_column(df, col)

    rows: list[dict[str, object]] = []
    for exp in exposure_cols:
        before_vals: list[float] = []
        after_vals: list[float] = []
        for _, group in df.groupby(by, sort=True):
            g = group[[value_before_col, value_after_col, exp]].copy()
            g = g.dropna()
            if len(g) < 3:
                continue
            corr_before = g[value_before_col].corr(g[exp])
            corr_after = g[value_after_col].corr(g[exp])
            if pd.notna(corr_before):
                before_vals.append(float(abs(corr_before)))
            if pd.notna(corr_after):
                after_vals.append(float(abs(corr_after)))
        before = float(np.mean(before_vals)) if before_vals else float("nan")
        after = float(np.mean(after_vals)) if after_vals else float("nan")
        rows.append(
            {
                "exposure": exp,
                "mean_abs_corr_before": before,
                "mean_abs_corr_after": after,
                "corr_reduction": (
                    before - after
                    if np.isfinite(before) and np.isfinite(after)
                    else np.nan
                ),
                "n_dates_used": min(len(before_vals), len(after_vals)),
            }
        )
    return pd.DataFrame(rows).sort_values("exposure", kind="mergesort").reset_index(drop=True)


def _ridge_ols(x: np.ndarray, y: np.ndarray, *, ridge: float) -> np.ndarray:
    xtx = x.T @ x
    if ridge > 0:
        xtx = xtx + ridge * np.eye(xtx.shape[0], dtype=float)
    xty = x.T @ y
    return np.linalg.solve(xtx, xty)


def _require_column(df: pd.DataFrame, col: str) -> None:
    if col not in df.columns:
        raise ValueError(f"df missing required column {col!r}")
