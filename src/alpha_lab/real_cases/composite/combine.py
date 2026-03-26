from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from alpha_lab.research_contracts import validate_canonical_signal_table
from alpha_lab.signal_transforms import (
    apply_min_coverage_gate,
    rank_cross_section,
    winsorize_cross_section,
    zscore_cross_section,
)

from .spec import ComponentSpec, CompositeCaseSpec

ComponentLoader = Callable[[ComponentSpec], pd.DataFrame]


@dataclass(frozen=True)
class CombineResult:
    """Composite construction artifacts before evaluation."""

    composite_factor: pd.DataFrame
    component_values: pd.DataFrame
    coverage_by_date: pd.DataFrame
    component_summary: pd.DataFrame


def build_linear_composite(
    spec: CompositeCaseSpec,
    *,
    component_loader: ComponentLoader | None = None,
    universe_mask: pd.DataFrame | None = None,
) -> CombineResult:
    """Build linear weighted composite factor from configured components."""

    loader = component_loader or _default_component_loader
    prepared: list[pd.DataFrame] = []

    for component in spec.components:
        raw = loader(component)
        component_df = _prepare_component(raw, component=component, spec=spec)
        if universe_mask is not None:
            component_df = _apply_universe_mask(component_df, universe_mask)
        if component_df.empty:
            raise ValueError(f"component {component.name!r} has no rows after filtering")
        prepared.append(component_df)

    component_values = (
        pd.concat(prepared, ignore_index=True)
        .sort_values(["date", "asset", "component"], kind="mergesort")
        .reset_index(drop=True)
    )

    grouped = component_values.groupby(["date", "asset"], sort=True)
    composite = grouped.agg(
        weighted_sum=("weighted_value", "sum"),
        abs_weight_sum=("abs_weight_present", "sum"),
    ).reset_index()
    composite["value"] = (
        composite["weighted_sum"]
        / composite["abs_weight_sum"].replace(0.0, np.nan)
    )
    composite["factor"] = spec.name
    composite_factor = composite[["date", "asset", "factor", "value"]].copy()

    coverage_by_date = _coverage_by_date(
        component_values,
        composite_factor,
        n_components=len(spec.components),
    )
    if spec.preprocess.min_coverage is not None:
        keep_dates = set(
            coverage_by_date.loc[
                coverage_by_date["composite_coverage"] >= spec.preprocess.min_coverage,
                "date",
            ]
        )
        component_values = component_values[
            component_values["date"].isin(keep_dates)
        ].reset_index(drop=True)
        composite_factor = composite_factor[
            composite_factor["date"].isin(keep_dates)
        ].reset_index(drop=True)
        coverage_by_date = _coverage_by_date(
            component_values,
            composite_factor,
            n_components=len(spec.components),
        )

    component_summary = _component_summary(component_values, spec=spec)

    if composite_factor.empty:
        raise ValueError("composite factor is empty after combination")

    return CombineResult(
        composite_factor=composite_factor,
        component_values=component_values,
        coverage_by_date=coverage_by_date,
        component_summary=component_summary,
    )


def _default_component_loader(component: ComponentSpec) -> pd.DataFrame:
    path = component.path
    try:
        return pd.read_csv(path)
    except Exception as exc:
        raise ValueError(
            f"failed to load component CSV for {component.name!r} ({path}): {exc}"
        ) from exc


def _prepare_component(
    raw: pd.DataFrame,
    *,
    component: ComponentSpec,
    spec: CompositeCaseSpec,
) -> pd.DataFrame:
    missing = {"date", "asset", "factor", "value"} - set(raw.columns)
    if missing:
        raise ValueError(
            f"component {component.name!r} is missing required columns: {sorted(missing)}"
        )

    frame = raw.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")

    selected = _select_component_factor(frame, component=component)
    validate_canonical_signal_table(selected, table_name=f"component[{component.name}]")

    transformed = selected[["date", "asset", "value"]].copy()
    if spec.preprocess.winsorize:
        transformed = winsorize_cross_section(
            transformed,
            lower=spec.preprocess.winsorize_lower,
            upper=spec.preprocess.winsorize_upper,
            min_group_size=spec.preprocess.min_group_size,
        )

    transformed = _apply_component_transform(
        transformed,
        transform=component.transform,
        min_group_size=spec.preprocess.min_group_size,
    )

    # Optional per-component coverage gate before combination.
    if spec.preprocess.min_coverage is not None:
        transformed = apply_min_coverage_gate(
            transformed,
            min_coverage=spec.preprocess.min_coverage,
        )

    sign = 1.0 if component.direction == "positive" else -1.0
    effective_weight = float(component.weight) * sign

    transformed = transformed.rename(columns={"value": "transformed_value"})
    transformed["component"] = component.name
    transformed["weight"] = float(component.weight)
    transformed["direction"] = component.direction
    transformed["transform"] = component.transform
    transformed["effective_weight"] = effective_weight
    transformed["signed_value"] = transformed["transformed_value"] * sign
    transformed["weighted_value"] = transformed["transformed_value"] * effective_weight
    transformed["abs_weight_present"] = (
        np.where(transformed["transformed_value"].notna(), abs(effective_weight), 0.0)
    )

    cols = [
        "date",
        "asset",
        "component",
        "weight",
        "direction",
        "transform",
        "effective_weight",
        "transformed_value",
        "signed_value",
        "weighted_value",
        "abs_weight_present",
    ]
    return transformed[cols].sort_values(["date", "asset"], kind="mergesort").reset_index(drop=True)


def _select_component_factor(frame: pd.DataFrame, *, component: ComponentSpec) -> pd.DataFrame:
    out = frame.copy()
    if component.factor is not None:
        out = out[out["factor"].astype(str) == component.factor].copy()
        if out.empty:
            raise ValueError(
                f"component {component.name!r} has no rows with factor={component.factor!r}"
            )
    else:
        factor_names = pd.unique(out["factor"].astype(str))
        if len(factor_names) != 1:
            raise ValueError(
                "component "
                f"{component.name!r} contains multiple factors; "
                "set component.factor explicitly"
            )

    out = out[["date", "asset", "factor", "value"]].copy()
    out["factor"] = component.name
    return out


def _apply_component_transform(
    df: pd.DataFrame,
    *,
    transform: str,
    min_group_size: int,
) -> pd.DataFrame:
    if transform == "zscore":
        return zscore_cross_section(df, min_group_size=min_group_size)
    if transform == "rank":
        return rank_cross_section(df, min_group_size=max(2, min_group_size), pct=True)
    if transform == "none":
        return df
    raise ValueError(f"unsupported component transform: {transform!r}")


def _apply_universe_mask(df: pd.DataFrame, universe_mask: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "asset", "in_universe"}
    missing = required - set(universe_mask.columns)
    if missing:
        raise ValueError(f"universe mask is missing columns: {sorted(missing)}")

    mask = universe_mask.copy()
    mask["date"] = pd.to_datetime(mask["date"], errors="coerce")
    mask = mask.dropna(subset=["date", "asset"])
    mask = mask[mask["in_universe"].astype(bool)][["date", "asset"]].drop_duplicates()
    out = df.merge(mask, on=["date", "asset"], how="inner", validate="many_to_one")
    return out.reset_index(drop=True)


def _coverage_by_date(
    component_values: pd.DataFrame,
    composite_factor: pd.DataFrame,
    *,
    n_components: int,
) -> pd.DataFrame:
    if component_values.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "n_assets",
                "mean_component_coverage",
                "n_complete_assets",
                "composite_coverage",
            ]
        )

    wide = component_values.pivot_table(
        index=["date", "asset"],
        columns="component",
        values="transformed_value",
        aggfunc="first",
    )
    per_asset = wide.notna().sum(axis=1).rename("n_components_present")
    coverage = per_asset / float(n_components)
    coverage_df = pd.concat([per_asset, coverage.rename("coverage_ratio")], axis=1).reset_index()

    per_date = coverage_df.groupby("date", sort=True).agg(
        n_assets=("asset", "nunique"),
        mean_component_coverage=("coverage_ratio", "mean"),
        n_complete_assets=("n_components_present", lambda s: int((s == n_components).sum())),
    )

    composite_cov = composite_factor.copy()
    composite_cov["non_null"] = composite_cov["value"].notna().astype(float)
    comp_by_date = composite_cov.groupby("date", sort=True)["non_null"].mean().rename(
        "composite_coverage"
    )

    out = per_date.join(comp_by_date, how="left").reset_index()
    out["composite_coverage"] = out["composite_coverage"].fillna(0.0)
    return out


def _component_summary(component_values: pd.DataFrame, *, spec: CompositeCaseSpec) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for component in spec.components:
        view = component_values[component_values["component"] == component.name]
        vals = pd.to_numeric(view["transformed_value"], errors="coerce")
        rows.append(
            {
                "component": component.name,
                "weight": component.weight,
                "direction": component.direction,
                "transform": component.transform,
                "n_rows": int(len(view)),
                "n_dates": int(view["date"].nunique()),
                "n_assets": int(view["asset"].nunique()),
                "non_null_ratio": float(vals.notna().mean()) if len(vals) > 0 else float("nan"),
                "mean_value": float(vals.mean()) if len(vals) > 0 else float("nan"),
                "std_value": float(vals.std(ddof=0)) if len(vals) > 0 else float("nan"),
            }
        )
    return pd.DataFrame(rows)
