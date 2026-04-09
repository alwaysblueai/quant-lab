from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from alpha_lab.exceptions import AlphaLabConfigError, AlphaLabDataError
from alpha_lab.interfaces import validate_factor_output
from alpha_lab.neutralization import neutralize_signal
from alpha_lab.real_cases.common_io import (
    apply_universe_to_factor,
    apply_universe_to_prices,
    load_prices,
    load_universe_mask,
)
from alpha_lab.research_contracts import validate_canonical_signal_table, validate_prices_table
from alpha_lab.research_evaluation_config import get_research_evaluation_config
from alpha_lab.research_integrity.contracts import IntegrityCheckResult, IntegrityReport
from alpha_lab.research_integrity.exceptions import raise_on_hard_failures
from alpha_lab.research_integrity.leakage_checks import (
    check_asof_inputs_not_after_signal_date,
    check_cross_section_transform_scope,
    check_factor_label_temporal_order,
    check_no_future_dates_in_input,
)
from alpha_lab.research_integrity.reporting import build_integrity_report
from alpha_lab.signal_transforms import (
    apply_min_coverage_gate,
    rank_cross_section,
    winsorize_cross_section,
    zscore_cross_section,
)

from .artifacts import SingleFactorArtifactPaths, export_artifact_bundle
from .evaluate import SingleFactorEvaluationResult, evaluate_single_factor_case
from .spec import SingleFactorCaseSpec, load_single_factor_case_spec

FactorLoader = Callable[[SingleFactorCaseSpec], pd.DataFrame]


@dataclass(frozen=True)
class SingleFactorCaseRunResult:
    """End-to-end run result for one real-case single-factor research package."""

    spec: SingleFactorCaseSpec
    output_dir: Path
    factor_df: pd.DataFrame
    evaluation_result: SingleFactorEvaluationResult
    artifact_paths: SingleFactorArtifactPaths
    integrity_report: IntegrityReport


def run_single_factor_case(
    spec_or_path: SingleFactorCaseSpec | str | Path,
    *,
    output_root_dir: str | Path | None = None,
    factor_loader: FactorLoader | None = None,
    evaluation_profile: str = "default_research",
    vault_root: str | Path | None = None,
    vault_export_mode: str = "versioned",
) -> SingleFactorCaseRunResult:
    """Run one real-case single-factor study end-to-end and export artifacts."""
    integrity_checks: list[IntegrityCheckResult] = []

    def _record_integrity(check: IntegrityCheckResult) -> None:
        integrity_checks.append(check)
        raise_on_hard_failures((check,))

    spec_path: Path | None = None
    if isinstance(spec_or_path, SingleFactorCaseSpec):
        spec = spec_or_path
    else:
        spec_path = Path(spec_or_path).resolve()
        spec = load_single_factor_case_spec(spec_path)

    evaluation_config = get_research_evaluation_config(evaluation_profile)

    universe_mask = load_universe_mask(spec.universe)
    prices = load_prices(spec.prices_path)
    max_price_date = pd.Timestamp(prices["date"].max())
    _record_integrity(
        check_no_future_dates_in_input(
            prices,
            max_allowed_date=max_price_date,
            date_col="date",
            object_name="single_factor_prices",
        )
    )
    if universe_mask is not None:
        _record_integrity(
            check_no_future_dates_in_input(
                universe_mask,
                max_allowed_date=max_price_date,
                date_col="date",
                object_name="single_factor_universe",
            )
        )
        _record_integrity(
            check_asof_inputs_not_after_signal_date(
                prices[["date", "asset"]],
                universe_mask,
                by=("asset",),
                signal_date_col="date",
                aux_effective_date_col="date",
                aux_known_at_col=None,
                object_name="single_factor_universe_asof",
            )
        )
        prices = apply_universe_to_prices(prices, universe_mask)

    raw_factor = (factor_loader or _default_factor_loader)(spec)
    if "date" in raw_factor.columns:
        _record_integrity(
            check_no_future_dates_in_input(
                raw_factor,
                max_allowed_date=max_price_date,
                date_col="date",
                object_name="single_factor_raw_factor",
            )
        )
    factor_df = _prepare_factor(raw_factor, spec=spec)
    if universe_mask is not None:
        factor_df = apply_universe_to_factor(factor_df, universe_mask)

    raw_factor_df = factor_df.copy()
    factor_df, neutral_diag = _maybe_neutralize_factor(
        factor_df,
        spec=spec,
        universe_mask=universe_mask,
        integrity_checks=integrity_checks,
        max_price_date=max_price_date,
    )
    coverage_by_date = _coverage_by_date(factor_df)

    validate_factor_output(factor_df)
    _record_integrity(
        check_cross_section_transform_scope(
            prices[["date", "asset"]],
            factor_df[["date", "asset", "value"]],
            date_col="date",
            asset_col="asset",
            object_name="single_factor_final_factor_scope",
        )
    )

    evaluation_result = evaluate_single_factor_case(
        prices=prices,
        factor_df=factor_df,
        raw_factor_df=raw_factor_df,
        spec=spec,
        coverage_by_date=coverage_by_date,
        neutralization_summary=neutral_diag,
        evaluation_config=evaluation_config,
    )
    for check in evaluation_result.experiment_result.integrity_checks:
        _record_integrity(check)
    _record_integrity(
        check_factor_label_temporal_order(
            evaluation_result.experiment_result.factor_df,
            evaluation_result.experiment_result.label_df,
            join_keys=("date", "asset"),
            factor_date_col="date",
            label_date_col="date",
            object_name="single_factor_factor_label_alignment",
        )
    )
    integrity_report = build_integrity_report(
        tuple(integrity_checks),
        context={
            "pipeline": "run_single_factor_case",
            "case_name": spec.name,
            "prices_path": spec.prices_path,
            "factor_path": spec.factor_path,
            "factor_name": spec.factor_name,
            "neutralization_enabled": bool(spec.neutralization.enabled),
        },
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
        integrity_report=integrity_report,
        output_dir=output_dir,
        spec_path=spec_path,
        evaluation_config=evaluation_config,
        vault_root=vault_root,
        vault_export_mode=vault_export_mode,
    )

    return SingleFactorCaseRunResult(
        spec=spec,
        output_dir=output_dir,
        factor_df=factor_df,
        evaluation_result=evaluation_result,
        artifact_paths=artifact_paths,
        integrity_report=integrity_report,
    )


def _default_factor_loader(spec: SingleFactorCaseSpec) -> pd.DataFrame:
    path = Path(spec.factor_path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"factor file does not exist: {path}")
    return pd.read_csv(path)


def _prepare_factor(raw: pd.DataFrame, *, spec: SingleFactorCaseSpec) -> pd.DataFrame:
    missing = {"date", "asset", "factor", "value"} - set(raw.columns)
    if missing:
        raise AlphaLabDataError(f"factor file is missing required columns: {sorted(missing)}")

    frame = raw.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame[frame["factor"].astype(str) == spec.factor_name].copy()
    if frame.empty:
        raise AlphaLabDataError(
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


def _maybe_neutralize_factor(
    factor_df: pd.DataFrame,
    *,
    spec: SingleFactorCaseSpec,
    universe_mask: pd.DataFrame | None,
    integrity_checks: list[IntegrityCheckResult] | None = None,
    max_price_date: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    if not spec.neutralization.enabled:
        return factor_df, None

    exposures_path = spec.neutralization.exposures_path
    if exposures_path is None:
        raise AlphaLabConfigError("neutralization.exposures_path is required when neutralization is enabled")

    exposures = pd.read_csv(exposures_path)
    exposures["date"] = pd.to_datetime(exposures["date"], errors="coerce")

    required = {"date", "asset"}
    if spec.neutralization.size_col is not None:
        required.add(spec.neutralization.size_col)
    if spec.neutralization.industry_col is not None:
        required.add(spec.neutralization.industry_col)

    missing = required - set(exposures.columns)
    if missing:
        raise AlphaLabDataError(
            "neutralization exposure file is missing required columns: "
            f"{sorted(missing)}"
        )
    known_at_col = None
    if "known_at" in exposures.columns:
        known_at_col = "known_at"
    elif "available_at" in exposures.columns:
        known_at_col = "available_at"

    if integrity_checks is not None and max_price_date is not None:
        no_future_check = check_no_future_dates_in_input(
            exposures,
            max_allowed_date=max_price_date,
            date_col="date",
            object_name="single_factor_neutralization_exposures",
        )
        integrity_checks.append(no_future_check)
        raise_on_hard_failures((no_future_check,))

        asof_check = check_asof_inputs_not_after_signal_date(
            factor_df[["date", "asset"]],
            exposures,
            by=("asset",),
            signal_date_col="date",
            aux_effective_date_col="date",
            aux_known_at_col=known_at_col,
            object_name="single_factor_neutralization_exposures_asof",
        )
        integrity_checks.append(asof_check)
        raise_on_hard_failures((asof_check,))

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
    known_at_input = None
    if known_at_col is not None:
        merged["__known_at_input"] = pd.to_datetime(
            merged[known_at_col],
            errors="coerce",
        )
        known_at_input = "__known_at_input"

    cols = ["date", "asset", "value"]
    for col in (size_col, industry_col):
        if col is not None:
            cols.append(col)
    if known_at_input is not None:
        cols.append(known_at_input)

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
        known_at_col=known_at_input,
        enforce_integrity=True,
    )
    if integrity_checks is not None:
        integrity_checks.extend(list(neutralized.integrity_checks))
        raise_on_hard_failures(neutralized.integrity_checks)

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
