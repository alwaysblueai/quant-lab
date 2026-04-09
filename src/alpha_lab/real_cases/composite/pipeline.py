from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from alpha_lab.interfaces import validate_factor_output
from alpha_lab.neutralization import neutralize_signal
from alpha_lab.real_cases.common_io import (
    apply_universe_to_prices,
    load_prices,
    load_universe_mask,
)
from alpha_lab.research_contracts import validate_prices_table
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

from .artifacts import CompositeArtifactPaths, export_artifact_bundle
from .combine import CombineResult, ComponentLoader, build_linear_composite
from .evaluate import CompositeEvaluationResult, evaluate_composite_case
from .spec import CompositeCaseSpec, load_composite_case_spec


@dataclass(frozen=True)
class CompositeCaseRunResult:
    """End-to-end run result for one real-case composite research package."""

    spec: CompositeCaseSpec
    output_dir: Path
    combine_result: CombineResult
    evaluation_result: CompositeEvaluationResult
    artifact_paths: CompositeArtifactPaths
    integrity_report: IntegrityReport


def run_composite_case(
    spec_or_path: CompositeCaseSpec | str | Path,
    *,
    output_root_dir: str | Path | None = None,
    component_loader: ComponentLoader | None = None,
    evaluation_profile: str = "default_research",
    vault_root: str | Path | None = None,
    vault_export_mode: str = "versioned",
) -> CompositeCaseRunResult:
    """Run one real-case composite study end-to-end and export artifacts."""
    integrity_checks: list[IntegrityCheckResult] = []

    def _record_integrity(check: IntegrityCheckResult) -> None:
        integrity_checks.append(check)
        raise_on_hard_failures((check,))

    spec_path: Path | None = None
    if isinstance(spec_or_path, CompositeCaseSpec):
        spec = spec_or_path
    else:
        spec_path = Path(spec_or_path).resolve()
        spec = load_composite_case_spec(spec_path)

    evaluation_config = get_research_evaluation_config(evaluation_profile)

    universe_mask = load_universe_mask(spec.universe)
    prices = load_prices(spec.prices_path)
    max_price_date = pd.Timestamp(prices["date"].max())
    _record_integrity(
        check_no_future_dates_in_input(
            prices,
            max_allowed_date=max_price_date,
            date_col="date",
            object_name="composite_prices",
        )
    )
    if universe_mask is not None:
        _record_integrity(
            check_no_future_dates_in_input(
                universe_mask,
                max_allowed_date=max_price_date,
                date_col="date",
                object_name="composite_universe",
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
                object_name="composite_universe_asof",
            )
        )
        prices = apply_universe_to_prices(prices, universe_mask)

    combine_result = build_linear_composite(
        spec,
        component_loader=component_loader,
        universe_mask=universe_mask,
    )
    raw_composite_factor = combine_result.composite_factor.copy()

    composite_factor, exposure_summary = _maybe_neutralize_composite(
        combine_result.composite_factor,
        spec=spec,
        universe_mask=universe_mask,
        integrity_checks=integrity_checks,
        max_price_date=max_price_date,
    )
    validate_factor_output(composite_factor)
    _record_integrity(
        check_cross_section_transform_scope(
            prices[["date", "asset"]],
            composite_factor[["date", "asset", "value"]],
            date_col="date",
            asset_col="asset",
            object_name="composite_final_factor_scope",
        )
    )

    evaluation_result = evaluate_composite_case(
        prices=prices,
        composite_factor=composite_factor,
        raw_composite_factor=raw_composite_factor,
        spec=spec,
        coverage_by_date=combine_result.coverage_by_date,
        exposure_summary=exposure_summary,
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
            object_name="composite_factor_label_alignment",
        )
    )
    integrity_report = build_integrity_report(
        tuple(integrity_checks),
        context={
            "pipeline": "run_composite_case",
            "case_name": spec.name,
            "prices_path": spec.prices_path,
            "n_components": len(spec.components),
            "neutralization_enabled": bool(spec.neutralization.enabled),
        },
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
        integrity_report=integrity_report,
        output_dir=output_dir,
        spec_path=spec_path,
        evaluation_config=evaluation_config,
        vault_root=vault_root,
        vault_export_mode=vault_export_mode,
    )

    return CompositeCaseRunResult(
        spec=spec,
        output_dir=output_dir,
        combine_result=combine_result,
        evaluation_result=evaluation_result,
        artifact_paths=artifact_paths,
        integrity_report=integrity_report,
    )


def _maybe_neutralize_composite(
    composite_factor: pd.DataFrame,
    *,
    spec: CompositeCaseSpec,
    universe_mask: pd.DataFrame | None,
    integrity_checks: list[IntegrityCheckResult] | None = None,
    max_price_date: pd.Timestamp | None = None,
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
            object_name="composite_neutralization_exposures",
        )
        integrity_checks.append(no_future_check)
        raise_on_hard_failures((no_future_check,))

        asof_check = check_asof_inputs_not_after_signal_date(
            composite_factor[["date", "asset"]],
            exposures,
            by=("asset",),
            signal_date_col="date",
            aux_effective_date_col="date",
            aux_known_at_col=known_at_col,
            object_name="composite_neutralization_exposures_asof",
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
    known_at_input = None
    if known_at_col is not None:
        merged["__known_at_input"] = pd.to_datetime(
            merged[known_at_col],
            errors="coerce",
        )
        known_at_input = "__known_at_input"

    cols = ["date", "asset", "value"]
    for col in (size_col, industry_col, beta_col):
        if col is not None:
            cols.append(col)
    if known_at_input is not None:
        cols.append(known_at_input)
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
        known_at_col=known_at_input,
        enforce_integrity=True,
    )
    if integrity_checks is not None:
        integrity_checks.extend(list(neutralized.integrity_checks))
        raise_on_hard_failures(neutralized.integrity_checks)

    out = composite_factor[["date", "asset", "factor"]].copy()
    out = out.merge(
        neutralized.data[["date", "asset", "value_neutralized"]],
        on=["date", "asset"],
        how="left",
        validate="one_to_one",
    )
    out = out.rename(columns={"value_neutralized": "value"})

    return out, neutralized.diagnostics
