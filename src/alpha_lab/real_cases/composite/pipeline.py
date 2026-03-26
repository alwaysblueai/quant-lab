from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from alpha_lab.interfaces import validate_factor_output
from alpha_lab.neutralization import neutralize_signal
from alpha_lab.research_contracts import validate_prices_table

from .artifacts import export_artifact_bundle
from .combine import CombineResult, ComponentLoader, build_linear_composite
from .evaluate import CompositeEvaluationResult, evaluate_composite_case
from .spec import CompositeCaseSpec, load_composite_case_spec


@dataclass(frozen=True)
class CompositeCaseRunResult:
    """End-to-end run result for one real-case composite package."""

    spec: CompositeCaseSpec
    output_dir: Path
    combine_result: CombineResult
    evaluation_result: CompositeEvaluationResult
    artifact_paths: dict[str, Path]


def run_composite_case(
    spec_or_path: CompositeCaseSpec | str | Path,
    *,
    output_root_dir: str | Path | None = None,
    component_loader: ComponentLoader | None = None,
) -> CompositeCaseRunResult:
    """Run one real-case composite study end-to-end and export artifacts."""

    spec_path: Path | None = None
    if isinstance(spec_or_path, CompositeCaseSpec):
        spec = spec_or_path
    else:
        spec_path = Path(spec_or_path).resolve()
        spec = load_composite_case_spec(spec_path)

    universe_mask = _load_universe_mask(spec)
    prices = _load_prices(spec.prices_path)
    if universe_mask is not None:
        prices = _apply_universe_to_prices(prices, universe_mask)

    combine_result = build_linear_composite(
        spec,
        component_loader=component_loader,
        universe_mask=universe_mask,
    )

    composite_factor, exposure_summary = _maybe_neutralize_composite(
        combine_result.composite_factor,
        spec=spec,
        universe_mask=universe_mask,
    )
    validate_factor_output(composite_factor)

    evaluation_result = evaluate_composite_case(
        prices=prices,
        composite_factor=composite_factor,
        spec=spec,
        coverage_by_date=combine_result.coverage_by_date,
        exposure_summary=exposure_summary,
    )

    root_dir = (
        Path(output_root_dir).resolve()
        if output_root_dir is not None
        else Path(spec.output.root_dir)
    )
    root_dir = root_dir.resolve()
    output_dir = (root_dir / spec.name).resolve()

    artifact_paths = export_artifact_bundle(
        spec=spec,
        combine_result=combine_result,
        evaluation_result=evaluation_result,
        output_dir=output_dir,
        spec_path=spec_path,
    )

    return CompositeCaseRunResult(
        spec=spec,
        output_dir=output_dir,
        combine_result=combine_result,
        evaluation_result=evaluation_result,
        artifact_paths=artifact_paths,
    )


def _load_prices(path_value: str) -> pd.DataFrame:
    path = Path(path_value)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"prices file does not exist: {path}")
    prices = pd.read_csv(path)

    required = {"date", "asset", "close"}
    missing = required - set(prices.columns)
    if missing:
        raise ValueError(f"prices is missing required columns: {sorted(missing)}")

    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
    prices = prices.sort_values(["asset", "date"], kind="mergesort").reset_index(drop=True)
    validate_prices_table(prices)
    return prices


def _load_universe_mask(spec: CompositeCaseSpec) -> pd.DataFrame | None:
    path_value = spec.universe.path
    if path_value is None:
        return None

    path = Path(path_value)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"universe file does not exist: {path}")

    universe = pd.read_csv(path)
    col = spec.universe.in_universe_column
    required = {"date", "asset", col}
    missing = required - set(universe.columns)
    if missing:
        raise ValueError(f"universe file is missing required columns: {sorted(missing)}")

    out = universe[["date", "asset", col]].copy()
    out = out.rename(columns={col: "in_universe"})
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date", "asset"]).copy()
    if out.duplicated(subset=["date", "asset"]).any():
        raise ValueError("universe file contains duplicate (date, asset) rows")
    out["in_universe"] = out["in_universe"].astype(bool)
    return out


def _apply_universe_to_prices(prices: pd.DataFrame, universe_mask: pd.DataFrame) -> pd.DataFrame:
    active = universe_mask[universe_mask["in_universe"]][["date", "asset"]]
    out = prices.merge(active, on=["date", "asset"], how="inner", validate="many_to_one")
    if out.empty:
        raise ValueError("prices became empty after universe filtering")
    return out.sort_values(["asset", "date"], kind="mergesort").reset_index(drop=True)


def _maybe_neutralize_composite(
    composite_factor: pd.DataFrame,
    *,
    spec: CompositeCaseSpec,
    universe_mask: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    if not spec.neutralization.enabled:
        return composite_factor, None

    exposures_path = spec.neutralization.exposures_path
    if exposures_path is None:
        raise ValueError("neutralization.exposures_path is required when neutralization is enabled")

    exposures = pd.read_csv(exposures_path)
    exposures["date"] = pd.to_datetime(exposures["date"], errors="coerce")

    required_cols = {"date", "asset"}
    for col in (
        spec.neutralization.size_col,
        spec.neutralization.industry_col,
        spec.neutralization.beta_col,
    ):
        if col is not None:
            required_cols.add(col)
    missing = required_cols - set(exposures.columns)
    if missing:
        raise ValueError(
            f"neutralization exposure file is missing required columns: {sorted(missing)}"
        )

    if universe_mask is not None:
        active = universe_mask[universe_mask["in_universe"]][["date", "asset"]]
        exposures = exposures.merge(
            active,
            on=["date", "asset"],
            how="inner",
            validate="many_to_one",
        )

    base = composite_factor[["date", "asset", "value"]].copy()
    merged = base.merge(
        exposures,
        on=["date", "asset"],
        how="left",
        validate="one_to_one",
    )

    # Avoid duplicate-column collisions inside neutralize_signal.
    size_col = spec.neutralization.size_col
    industry_col = spec.neutralization.industry_col
    beta_col = spec.neutralization.beta_col

    if size_col is not None:
        merged["__size_input"] = merged[size_col]
        size_col = "__size_input"
    if industry_col is not None:
        merged["__industry_input"] = merged[industry_col]
        industry_col = "__industry_input"
    if beta_col is not None:
        merged["__beta_input"] = merged[beta_col]
        beta_col = "__beta_input"

    cols = ["date", "asset", "value"]
    for col in (size_col, industry_col, beta_col):
        if col is not None:
            cols.append(col)
    neutral_input = merged[cols].copy()

    neutralized = neutralize_signal(
        neutral_input,
        value_col="value",
        by="date",
        size_col=size_col,
        industry_col=industry_col,
        beta_col=beta_col,
        min_obs=spec.neutralization.min_obs,
        ridge=spec.neutralization.ridge,
        output_col="value_neutralized",
    )

    out = composite_factor[["date", "asset", "factor"]].copy()
    out = out.merge(
        neutralized.data[["date", "asset", "value_neutralized"]],
        on=["date", "asset"],
        how="left",
        validate="one_to_one",
    )
    out = out.rename(columns={"value_neutralized": "value"})

    return out, neutralized.diagnostics
