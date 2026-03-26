from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from alpha_lab.interfaces import validate_factor_output
from alpha_lab.neutralization import neutralize_signal
from alpha_lab.research_contracts import validate_canonical_signal_table, validate_prices_table
from alpha_lab.signal_transforms import (
    apply_min_coverage_gate,
    rank_cross_section,
    winsorize_cross_section,
    zscore_cross_section,
)

from .artifacts import export_artifact_bundle
from .evaluate import SingleFactorEvaluationResult, evaluate_single_factor_case
from .spec import SingleFactorCaseSpec, load_single_factor_case_spec

FactorLoader = Callable[[SingleFactorCaseSpec], pd.DataFrame]


@dataclass(frozen=True)
class SingleFactorCaseRunResult:
    """End-to-end run result for one real-case single-factor package."""

    spec: SingleFactorCaseSpec
    output_dir: Path
    factor_df: pd.DataFrame
    evaluation_result: SingleFactorEvaluationResult
    artifact_paths: dict[str, Path]


def run_single_factor_case(
    spec_or_path: SingleFactorCaseSpec | str | Path,
    *,
    output_root_dir: str | Path | None = None,
    factor_loader: FactorLoader | None = None,
    vault_root: str | Path | None = None,
    vault_export_mode: str = "versioned",
) -> SingleFactorCaseRunResult:
    """Run one real-case single-factor study end-to-end and export artifacts."""

    spec_path: Path | None = None
    if isinstance(spec_or_path, SingleFactorCaseSpec):
        spec = spec_or_path
    else:
        spec_path = Path(spec_or_path).resolve()
        spec = load_single_factor_case_spec(spec_path)

    universe_mask = _load_universe_mask(spec)
    prices = _load_prices(spec.prices_path)
    if universe_mask is not None:
        prices = _apply_universe_to_prices(prices, universe_mask)

    raw_factor = (factor_loader or _default_factor_loader)(spec)
    factor_df = _prepare_factor(raw_factor, spec=spec)
    if universe_mask is not None:
        factor_df = _apply_universe_to_factor(factor_df, universe_mask)

    factor_df, neutral_diag = _maybe_neutralize_factor(
        factor_df,
        spec=spec,
        universe_mask=universe_mask,
    )
    coverage_by_date = _coverage_by_date(factor_df)

    validate_factor_output(factor_df)

    evaluation_result = evaluate_single_factor_case(
        prices=prices,
        factor_df=factor_df,
        spec=spec,
        coverage_by_date=coverage_by_date,
        neutralization_summary=neutral_diag,
    )

    root_dir = (
        Path(output_root_dir).resolve()
        if output_root_dir is not None
        else Path(spec.output.root_dir)
    )
    output_dir = (root_dir.resolve() / spec.name).resolve()

    artifact_paths = export_artifact_bundle(
        spec=spec,
        evaluation_result=evaluation_result,
        output_dir=output_dir,
        spec_path=spec_path,
        vault_root=vault_root,
        vault_export_mode=vault_export_mode,
    )

    return SingleFactorCaseRunResult(
        spec=spec,
        output_dir=output_dir,
        factor_df=factor_df,
        evaluation_result=evaluation_result,
        artifact_paths=artifact_paths,
    )


def _default_factor_loader(spec: SingleFactorCaseSpec) -> pd.DataFrame:
    path = Path(spec.factor_path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"factor file does not exist: {path}")
    return pd.read_csv(path)


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


def _load_universe_mask(spec: SingleFactorCaseSpec) -> pd.DataFrame | None:
    if spec.universe.path is None:
        return None

    path = Path(spec.universe.path)
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


def _prepare_factor(raw: pd.DataFrame, *, spec: SingleFactorCaseSpec) -> pd.DataFrame:
    missing = {"date", "asset", "factor", "value"} - set(raw.columns)
    if missing:
        raise ValueError(f"factor file is missing required columns: {sorted(missing)}")

    frame = raw.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame[frame["factor"].astype(str) == spec.factor_name].copy()
    if frame.empty:
        raise ValueError(
            f"factor file has no rows for factor_name={spec.factor_name!r}"
        )

    frame["factor"] = spec.factor_name
    frame = frame[["date", "asset", "factor", "value"]].copy()
    validate_canonical_signal_table(frame, table_name="single_factor")

    transformed = frame[["date", "asset", "value"]].copy()
    if spec.preprocess.winsorize:
        transformed = winsorize_cross_section(
            transformed,
            lower=spec.preprocess.winsorize_lower,
            upper=spec.preprocess.winsorize_upper,
            min_group_size=spec.preprocess.min_group_size,
        )

    if spec.preprocess.standardization == "zscore":
        transformed = zscore_cross_section(
            transformed,
            min_group_size=spec.preprocess.min_group_size,
        )
    elif spec.preprocess.standardization == "rank":
        transformed = rank_cross_section(
            transformed,
            min_group_size=max(2, spec.preprocess.min_group_size),
            pct=True,
        )

    if spec.direction == "short":
        transformed["value"] = -transformed["value"]

    if spec.preprocess.min_coverage is not None:
        transformed = apply_min_coverage_gate(
            transformed,
            min_coverage=spec.preprocess.min_coverage,
        )

    out = transformed.copy()
    out["factor"] = spec.factor_name
    out = out[["date", "asset", "factor", "value"]]
    return out.sort_values(["date", "asset"], kind="mergesort").reset_index(drop=True)


def _apply_universe_to_prices(prices: pd.DataFrame, universe_mask: pd.DataFrame) -> pd.DataFrame:
    active = universe_mask[universe_mask["in_universe"]][["date", "asset"]]
    out = prices.merge(active, on=["date", "asset"], how="inner", validate="many_to_one")
    if out.empty:
        raise ValueError("prices became empty after universe filtering")
    return out.sort_values(["asset", "date"], kind="mergesort").reset_index(drop=True)


def _apply_universe_to_factor(factor_df: pd.DataFrame, universe_mask: pd.DataFrame) -> pd.DataFrame:
    active = universe_mask[universe_mask["in_universe"]][["date", "asset"]]
    out = factor_df.merge(active, on=["date", "asset"], how="inner", validate="many_to_one")
    if out.empty:
        raise ValueError("factor data became empty after universe filtering")
    return out.sort_values(["date", "asset"], kind="mergesort").reset_index(drop=True)


def _maybe_neutralize_factor(
    factor_df: pd.DataFrame,
    *,
    spec: SingleFactorCaseSpec,
    universe_mask: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    if not spec.neutralization.enabled:
        return factor_df, None

    exposures_path = spec.neutralization.exposures_path
    if exposures_path is None:
        raise ValueError("neutralization.exposures_path is required when neutralization is enabled")

    exposures = pd.read_csv(exposures_path)
    exposures["date"] = pd.to_datetime(exposures["date"], errors="coerce")

    required = {"date", "asset"}
    if spec.neutralization.size_col is not None:
        required.add(spec.neutralization.size_col)
    if spec.neutralization.industry_col is not None:
        required.add(spec.neutralization.industry_col)

    missing = required - set(exposures.columns)
    if missing:
        raise ValueError(
            "neutralization exposure file is missing required columns: "
            f"{sorted(missing)}"
        )

    if universe_mask is not None:
        active = universe_mask[universe_mask["in_universe"]][["date", "asset"]]
        exposures = exposures.merge(
            active,
            on=["date", "asset"],
            how="inner",
            validate="many_to_one",
        )

    merged = factor_df[["date", "asset", "value"]].merge(
        exposures,
        on=["date", "asset"],
        how="left",
        validate="one_to_one",
    )

    size_col = spec.neutralization.size_col
    industry_col = spec.neutralization.industry_col

    if size_col is not None:
        merged["__size_input"] = merged[size_col]
        size_col = "__size_input"
    if industry_col is not None:
        merged["__industry_input"] = merged[industry_col]
        industry_col = "__industry_input"

    cols = ["date", "asset", "value"]
    for col in (size_col, industry_col):
        if col is not None:
            cols.append(col)

    neutralized = neutralize_signal(
        merged[cols].copy(),
        value_col="value",
        by="date",
        size_col=size_col,
        industry_col=industry_col,
        beta_col=None,
        min_obs=spec.neutralization.min_obs,
        ridge=spec.neutralization.ridge,
        output_col="value_neutralized",
    )

    out = factor_df[["date", "asset", "factor"]].copy()
    out = out.merge(
        neutralized.data[["date", "asset", "value_neutralized"]],
        on=["date", "asset"],
        how="left",
        validate="one_to_one",
    )
    out = out.rename(columns={"value_neutralized": "value"})
    return out, neutralized.diagnostics


def _coverage_by_date(factor_df: pd.DataFrame) -> pd.DataFrame:
    if factor_df.empty:
        return pd.DataFrame(columns=["date", "n_assets", "coverage", "missingness"])

    summary = factor_df.groupby("date", sort=True).agg(
        n_assets=("asset", "nunique"),
        n_non_null=("value", lambda s: int(s.notna().sum())),
    )
    summary["coverage"] = summary["n_non_null"] / summary["n_assets"].replace(0, pd.NA)
    summary["missingness"] = 1.0 - summary["coverage"]
    return summary.reset_index()[["date", "n_assets", "coverage", "missingness"]]
