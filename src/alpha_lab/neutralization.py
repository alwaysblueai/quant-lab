from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from alpha_lab.exceptions import AlphaLabConfigError, AlphaLabDataError
from alpha_lab.research_integrity.contracts import IntegrityCheckResult
from alpha_lab.research_integrity.leakage_checks import (
    check_asof_inputs_not_after_signal_date,
    check_cross_section_transform_scope,
)


@dataclass(frozen=True)
class NeutralizationResult:
    """Result of cross-sectional signal neutralization."""

    data: pd.DataFrame
    diagnostics: pd.DataFrame
    integrity_checks: tuple[IntegrityCheckResult, ...] = ()


def neutralize_signal(
    df: pd.DataFrame,
    *,
    value_col: str,
    by: str,
    size_col: str | None = None,
    industry_col: str | None = None,
    beta_col: str | None = None,
    min_obs: int = 20,
    ridge: float = 1e-8,
    output_col: str = "value_neutralized",
    known_at_col: str | None = None,
    enforce_integrity: bool = True,
) -> NeutralizationResult:
    """Neutralize a signal by cross-sectional regression within each date.

    The implementation is intentionally narrow for real-case research packages:
    linear de-meaning against optional size/beta/industry exposures plus a
    small ridge penalty for numeric stability.
    """

    required = {by, "asset", value_col}
    missing = required - set(df.columns)
    if missing:
        raise AlphaLabDataError(f"neutralize_signal input missing columns: {sorted(missing)}")
    if min_obs <= 0:
        raise AlphaLabConfigError("min_obs must be > 0")
    if ridge < 0:
        raise AlphaLabConfigError("ridge must be >= 0")
    integrity_checks: list[IntegrityCheckResult] = []

    exposure_families: list[tuple[str, str]] = []
    if size_col is not None:
        if size_col not in df.columns:
            raise AlphaLabDataError(f"size_col not found: {size_col}")
        exposure_families.append(("size", size_col))
    if beta_col is not None:
        if beta_col not in df.columns:
            raise AlphaLabDataError(f"beta_col not found: {beta_col}")
        exposure_families.append(("beta", beta_col))
    if industry_col is not None:
        if industry_col not in df.columns:
            raise AlphaLabDataError(f"industry_col not found: {industry_col}")
        exposure_families.append(("industry", industry_col))
    if known_at_col is not None and known_at_col not in df.columns:
        raise AlphaLabDataError(f"known_at_col not found: {known_at_col}")

    if enforce_integrity and known_at_col is not None and by == "date":
        signal_dates = df[[by, "asset"]].copy().rename(columns={by: "date"})
        aux_dates = (
            df[[by, "asset", known_at_col]]
            .copy()
            .rename(columns={by: "effective_date"})
        )
        asof_check = check_asof_inputs_not_after_signal_date(
            signal_dates,
            aux_dates,
            by=("asset",),
            signal_date_col="date",
            aux_effective_date_col="effective_date",
            aux_known_at_col=known_at_col,
            object_name="neutralization_exposure_asof",
        )
        integrity_checks.append(asof_check)
        if asof_check.status == "fail":
            raise AlphaLabDataError(
                "neutralization exposure timing failed integrity check: "
                f"{asof_check.message}"
            )
        if asof_check.status == "warn":
            warnings.warn(
                f"neutralize_signal integrity warning: {asof_check.message}",
                UserWarning,
                stacklevel=2,
            )

    out = df.copy()
    y_all = pd.to_numeric(out[value_col], errors="coerce")
    out[output_col] = y_all

    stats: dict[str, dict[str, list[float]]] = {
        family: {"before": [], "after": []} for family, _ in exposure_families
    }

    for _, group_idx in out.groupby(by, sort=True).groups.items():
        idx = pd.Index(group_idx)
        g = out.loc[idx].copy()
        y = pd.to_numeric(g[value_col], errors="coerce")

        family_matrix: dict[str, pd.DataFrame] = {}
        for family, column in exposure_families:
            if family == "industry":
                cat = g[column].astype("string")
                x = pd.get_dummies(cat, prefix="ind", dummy_na=False, dtype=float)
            else:
                x = pd.DataFrame(
                    {column: pd.to_numeric(g[column], errors="coerce")},
                    index=g.index,
                )
            family_matrix[family] = x

        if not family_matrix:
            out.loc[idx, output_col] = y
            continue

        x_concat = pd.concat(family_matrix.values(), axis=1)
        valid = y.notna() & x_concat.notna().all(axis=1)
        n_obs = int(valid.sum())
        if n_obs < min_obs:
            out.loc[idx, output_col] = y
            continue

        x = x_concat.loc[valid].to_numpy(dtype=float)
        yv = y.loc[valid].to_numpy(dtype=float)

        x_design = np.column_stack([np.ones(n_obs, dtype=float), x])
        xtx = x_design.T @ x_design
        if ridge > 0:
            penalty = np.eye(xtx.shape[0], dtype=float) * ridge
            penalty[0, 0] = 0.0
            xtx = xtx + penalty

        beta = np.linalg.solve(xtx, x_design.T @ yv)
        residual = yv - (x_design @ beta)

        neutralized = y.copy()
        neutralized.loc[valid] = residual
        out.loc[idx, output_col] = neutralized

        y_after = neutralized.loc[valid]
        for family, _column in exposure_families:
            fam_x = family_matrix[family].loc[valid]
            before = _mean_abs_corr(y.loc[valid], fam_x)
            after = _mean_abs_corr(y_after, fam_x)
            if np.isfinite(before):
                stats[family]["before"].append(float(before))
            if np.isfinite(after):
                stats[family]["after"].append(float(after))

    diag_rows: list[dict[str, object]] = []
    label_map = {
        "size": size_col,
        "beta": beta_col,
        "industry": industry_col,
    }
    for family, _ in exposure_families:
        before_vals = stats[family]["before"]
        after_vals = stats[family]["after"]
        before_mean = float(np.mean(before_vals)) if before_vals else float("nan")
        after_mean = float(np.mean(after_vals)) if after_vals else float("nan")
        diag_rows.append(
            {
                "exposure": label_map[family] or family,
                "mean_abs_corr_before": _finite_or_nan(before_mean),
                "mean_abs_corr_after": _finite_or_nan(after_mean),
                "corr_reduction": _finite_or_nan(before_mean - after_mean),
                "n_dates_used": int(min(len(before_vals), len(after_vals))),
            }
        )

    diagnostics = pd.DataFrame(
        diag_rows,
        columns=[
            "exposure",
            "mean_abs_corr_before",
            "mean_abs_corr_after",
            "corr_reduction",
            "n_dates_used",
        ],
    )

    if enforce_integrity and by == "date":
        scope_check = check_cross_section_transform_scope(
            raw_df=df[[by, "asset"]].rename(columns={by: "date"}),
            transformed_df=out[[by, "asset", output_col]].rename(
                columns={by: "date", output_col: "value"}
            ),
            date_col="date",
            asset_col="asset",
            object_name="neutralize_signal_scope",
        )
        integrity_checks.append(scope_check)
        if scope_check.status == "fail":
            raise AlphaLabDataError(
                "neutralize_signal output scope failed integrity check: "
                f"{scope_check.message}"
            )
        if scope_check.status == "warn":
            warnings.warn(
                f"neutralize_signal integrity warning: {scope_check.message}",
                UserWarning,
                stacklevel=2,
            )

    return NeutralizationResult(
        data=out,
        diagnostics=diagnostics,
        integrity_checks=tuple(integrity_checks),
    )


def _mean_abs_corr(y: pd.Series, x_frame: pd.DataFrame) -> float:
    yv = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)
    corrs: list[float] = []
    for _, col in x_frame.items():
        xv = pd.to_numeric(col, errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(yv) & np.isfinite(xv)
        if int(valid.sum()) < 2:
            continue
        y_sub = yv[valid]
        x_sub = xv[valid]
        if np.nanstd(y_sub) == 0 or np.nanstd(x_sub) == 0:
            continue
        corr = np.corrcoef(y_sub, x_sub)[0, 1]
        if np.isfinite(corr):
            corrs.append(float(abs(corr)))
    if not corrs:
        return float("nan")
    return float(np.mean(corrs))


def _finite_or_nan(value: float) -> float:
    return value if np.isfinite(value) else float("nan")
