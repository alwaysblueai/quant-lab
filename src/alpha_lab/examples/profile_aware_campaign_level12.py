from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from alpha_lab.artifact_contracts import validate_level12_artifact_payload
from alpha_lab.campaigns._profile_helpers import (
    _TRANSITION_DELTA_LABEL_MIXED,
    _TRANSITION_DELTA_LABEL_STABLE,
    _TRANSITION_DELTA_LABELS,
    _TRANSITION_STRENGTH_SCORE,
    CampaignCaseProfileSummary,
    ProfileCampaignSummary,
    _adjacent_profile_pairs,
    _build_case_level12_transition_profile_delta,
    _case_field_differences,
    _case_profile_lookup,
    _case_transition_delta_label,
    _consistently_strong,
    _dominant_reduction_mode,
    _empty_transition_pair_count_matrix,
    _format_reason_ratio,
    _has_changed_field,
    _pair_reduction_counts,
    _promoted_only_under_looser_profiles,
    _reason_rollup_for_transition_label,
    _sensitivity_label,
    _to_float_value,
    _to_int_value,
    _transition_pair_proportion_matrix,
    _transition_profile_path_text,
)
from alpha_lab.exceptions import AlphaLabConfigError, AlphaLabDataError
from alpha_lab.key_metrics_contracts import (
    CAMPAIGN_PROFILE_COMPARISON_FIELDS,
    LEVEL12_TRANSITION_TAXONOMY,
    project_campaign_profile_summary_metrics,
    project_campaign_ranking_metrics,
    project_level12_transition_distribution,
)
from alpha_lab.real_cases.single_factor.pipeline import run_single_factor_case
from alpha_lab.reporting.campaign_triage import campaign_rank_sort_key
from alpha_lab.reporting.renderers import write_case_report
from alpha_lab.research_evaluation_config import (
    AVAILABLE_RESEARCH_EVALUATION_PROFILES,
    CAMPAIGN_PROFILE_COMPARE_DEFAULTS,
    DEFAULT_RESEARCH_EVALUATION_CONFIG,
)

DEFAULT_PROFILE_AWARE_CAMPAIGN_LEVEL12_PROFILES: tuple[str, ...] = (
    CAMPAIGN_PROFILE_COMPARE_DEFAULTS
    if CAMPAIGN_PROFILE_COMPARE_DEFAULTS
    else AVAILABLE_RESEARCH_EVALUATION_PROFILES
)


@dataclass(frozen=True)
class CampaignExampleCaseSpec:
    case_name: str
    case_description: str
    spec_path: Path


@dataclass(frozen=True)
class ProfileAwareCampaignLevel12ExampleResult:
    root_dir: Path
    case_specs: tuple[CampaignExampleCaseSpec, ...]
    profile_campaigns: tuple[ProfileCampaignSummary, ...]
    comparison_json_path: Path
    comparison_markdown_path: Path
    comparison_csv_path: Path


def run_profile_aware_campaign_level12_example(
    *,
    output_root_dir: str | Path = "dist/examples/profile_aware_campaign_level12",
    profiles: tuple[str, ...] = DEFAULT_PROFILE_AWARE_CAMPAIGN_LEVEL12_PROFILES,
    pair_mode: str = "adjacent",
    artifact_hint_path_mode: str = "relative",
    render_report: bool = True,
    clean_output: bool = True,
) -> ProfileAwareCampaignLevel12ExampleResult:
    selected_profiles = _normalize_profiles(profiles)
    root_dir = Path(output_root_dir).resolve()

    if clean_output and root_dir.exists():
        shutil.rmtree(root_dir)
    root_dir.mkdir(parents=True, exist_ok=True)

    case_specs = _write_campaign_case_specs(root_dir)

    profile_campaigns: list[ProfileCampaignSummary] = []
    for profile_name in selected_profiles:
        case_summaries: list[CampaignCaseProfileSummary] = []
        ranking_metrics_by_case: dict[str, dict[str, object]] = {}
        for case_spec in case_specs:
            case_result = run_single_factor_case(
                case_spec.spec_path,
                output_root_dir=root_dir / "runs" / profile_name,
                evaluation_profile=profile_name,
                vault_export_mode="skip",
            )
            metrics = _load_metrics_payload(case_result.artifact_paths["metrics"])
            case_report_path = (
                write_case_report(case_result.output_dir, overwrite=True) if render_report else None
            )
            profile_summary = project_campaign_profile_summary_metrics(metrics)
            ranking_projection = project_campaign_ranking_metrics(metrics)
            factor_verdict = profile_summary["factor_verdict"] or "N/A"
            campaign_triage = profile_summary["campaign_triage"] or "N/A"
            ranking_metrics_by_case[case_spec.case_name] = {
                **ranking_projection,
                "factor_verdict": factor_verdict,
                "campaign_triage": campaign_triage,
                "campaign_triage_reasons": list(profile_summary["campaign_triage_reasons"]),
            }
            case_summaries.append(
                CampaignCaseProfileSummary(
                    case_name=case_spec.case_name,
                    profile_name=profile_name,
                    output_dir=case_result.output_dir,
                    run_manifest_path=case_result.artifact_paths["run_manifest"],
                    metrics_path=case_result.artifact_paths["metrics"],
                    factor_definition_json_path=case_result.artifact_paths[
                        "factor_definition_json"
                    ],
                    signal_validation_json_path=case_result.artifact_paths[
                        "signal_validation_json"
                    ],
                    portfolio_recipe_json_path=case_result.artifact_paths["portfolio_recipe_json"],
                    backtest_result_json_path=case_result.artifact_paths["backtest_result_json"],
                    summary_path=case_result.artifact_paths["summary"],
                    experiment_card_path=case_result.artifact_paths["experiment_card"],
                    case_report_path=case_report_path,
                    factor_verdict=factor_verdict,
                    factor_verdict_reasons=profile_summary["factor_verdict_reasons"],
                    campaign_triage=campaign_triage,
                    campaign_triage_reasons=profile_summary["campaign_triage_reasons"],
                    promotion_decision=profile_summary["promotion_decision"] or "N/A",
                    promotion_reasons=profile_summary["promotion_reasons"],
                    promotion_blockers=profile_summary["promotion_blockers"],
                    level12_transition_label=(profile_summary["level12_transition_label"] or "N/A"),
                    level12_transition_reasons=profile_summary["level12_transition_reasons"],
                    portfolio_validation_status=(
                        profile_summary["portfolio_validation_status"] or "N/A"
                    ),
                    portfolio_validation_recommendation=(
                        profile_summary["portfolio_validation_recommendation"] or "N/A"
                    ),
                    portfolio_validation_major_risks=(
                        profile_summary["portfolio_validation_major_risks"]
                    ),
                )
            )

        ranked_case_order = tuple(
            row.case_name
            for row in sorted(
                case_summaries,
                key=lambda row: campaign_rank_sort_key(
                    row.case_name,
                    status="success",
                    metrics=ranking_metrics_by_case[row.case_name],
                ),
            )
        )

        profile_campaigns.append(
            ProfileCampaignSummary(
                profile_name=profile_name,
                case_summaries=tuple(case_summaries),
                ranked_case_order=ranked_case_order,
            )
        )

    from alpha_lab.campaigns import profile_comparison as profile_comparison_module

    normalized_artifact_hint_path_mode = (
        profile_comparison_module._normalize_artifact_hint_path_mode(artifact_hint_path_mode)
    )
    normalized_pair_mode = profile_comparison_module._normalize_pair_mode(pair_mode)

    comparison_case_specs = tuple(
        profile_comparison_module.CampaignComparisonCase(
            case_name=row.case_name,
            case_description=row.case_description,
            spec_path=row.spec_path,
        )
        for row in case_specs
    )
    comparison_profile_campaigns = tuple(profile_campaigns)
    comparison_payload = profile_comparison_module._build_campaign_comparison_payload(
        source="example",
        root_dir=root_dir,
        case_specs=comparison_case_specs,
        profile_campaigns=comparison_profile_campaigns,
        campaign_config_path=None,
        pair_mode=normalized_pair_mode,
    )
    rendered_payload = profile_comparison_module._render_artifact_hint_paths_in_payload(
        comparison_payload,
        root_dir=root_dir,
        path_mode=normalized_artifact_hint_path_mode,
    )
    rendered_payload["artifact_hint_path_mode"] = normalized_artifact_hint_path_mode
    rendered_payload["artifact_hint_path_base"] = "output_root_dir"

    comparison_json_path = root_dir / "campaign_profile_comparison.json"
    comparison_markdown_path = root_dir / "campaign_profile_comparison.md"
    comparison_csv_path = root_dir / "campaign_profile_case_matrix.csv"

    validate_level12_artifact_payload(
        rendered_payload,
        artifact_name=comparison_json_path.name,
        source=comparison_json_path,
    )
    comparison_json_path.write_text(
        json.dumps(rendered_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    comparison_markdown_path.write_text(
        profile_comparison_module._render_campaign_comparison_markdown(
            source="example",
            root_dir=root_dir,
            case_specs=comparison_case_specs,
            profile_campaigns=comparison_profile_campaigns,
            comparison_payload=rendered_payload,
        ),
        encoding="utf-8",
    )
    profile_comparison_module._write_case_matrix_csv(
        comparison_csv_path,
        case_specs=comparison_case_specs,
        profile_campaigns=comparison_profile_campaigns,
        comparison_payload=rendered_payload,
    )

    return ProfileAwareCampaignLevel12ExampleResult(
        root_dir=root_dir,
        case_specs=case_specs,
        profile_campaigns=tuple(profile_campaigns),
        comparison_json_path=comparison_json_path,
        comparison_markdown_path=comparison_markdown_path,
        comparison_csv_path=comparison_csv_path,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_profile_aware_campaign_level12_example",
        description=(
            "Run a compact deterministic multi-case campaign-style Level 1/2 "
            "example under multiple evaluation profiles and export case-by-case "
            "profile-sensitivity comparisons."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-root-dir",
        default="dist/examples/profile_aware_campaign_level12",
        help="Output directory for generated cases, profile runs, and comparison artifacts.",
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=list(DEFAULT_PROFILE_AWARE_CAMPAIGN_LEVEL12_PROFILES),
        choices=sorted(AVAILABLE_RESEARCH_EVALUATION_PROFILES),
        help="Evaluation profiles to compare for the same compact campaign.",
    )
    parser.add_argument(
        "--pair-mode",
        choices=["adjacent", "all_pairs"],
        default="adjacent",
        help=(
            "Profile-pair coverage for transition/reason delta matrices. "
            "`adjacent` keeps neighboring profile pairs only; "
            "`all_pairs` includes non-adjacent ordered pairs."
        ),
    )
    parser.add_argument(
        "--artifact-hint-path-mode",
        choices=["relative", "absolute"],
        default="relative",
        help=(
            "Render artifact pointer hints as paths relative to --output-root-dir "
            "(`relative`) or keep absolute filesystem paths (`absolute`)."
        ),
    )
    parser.add_argument(
        "--no-render-report",
        action="store_true",
        help="Skip case_report.md rendering for each profile/case run.",
    )
    parser.add_argument(
        "--no-clean-output",
        action="store_true",
        help="Keep existing output content and append/overwrite in place.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    result = run_profile_aware_campaign_level12_example(
        output_root_dir=args.output_root_dir,
        profiles=tuple(args.profiles),
        pair_mode=args.pair_mode,
        artifact_hint_path_mode=args.artifact_hint_path_mode,
        render_report=not bool(args.no_render_report),
        clean_output=not bool(args.no_clean_output),
    )

    print("")
    print("  Workflow : profile-aware-campaign-level12-example")
    print("  Status   : success")
    print(f"  Output   : {result.root_dir}")
    print(f"  Compare  : {result.comparison_markdown_path}")
    print(f"  CompareJ : {result.comparison_json_path}")
    print(f"  CompareC : {result.comparison_csv_path}")
    for campaign in result.profile_campaigns:
        print(f"  Profile  : {campaign.profile_name}")
        print(f"    Ranked Order : {list(campaign.ranked_case_order)}")
        for row in campaign.case_summaries:
            print(f"    Case                 : {row.case_name}")
            print(f"      Factor Verdict     : {row.factor_verdict}")
            print(f"      Campaign Triage    : {row.campaign_triage}")
            print(f"      Level 2 Promotion  : {row.promotion_decision}")
            print(
                "      Portfolio Validation: "
                f"{row.portfolio_validation_status} ({row.portfolio_validation_recommendation})"
            )
            print(f"      Output             : {row.output_dir}")
            print(f"      Metrics            : {row.metrics_path}")
            print(f"      Report             : {row.case_report_path}")
    return 0


def _normalize_profiles(profiles: tuple[str, ...]) -> tuple[str, ...]:
    if not profiles:
        raise AlphaLabConfigError("profiles must contain at least one profile name")
    normalized: list[str] = []
    for profile in profiles:
        name = str(profile).strip()
        if not name:
            continue
        if name not in AVAILABLE_RESEARCH_EVALUATION_PROFILES:
            raise AlphaLabConfigError(
                "unknown research evaluation profile: "
                f"{name!r}; available={list(AVAILABLE_RESEARCH_EVALUATION_PROFILES)}"
            )
        normalized.append(name)
    if not normalized:
        raise AlphaLabConfigError("profiles must contain at least one non-empty profile name")
    return tuple(normalized)


def _write_campaign_case_specs(root_dir: Path) -> tuple[CampaignExampleCaseSpec, ...]:
    case_root = root_dir / "cases"
    case_root.mkdir(parents=True, exist_ok=True)

    stable_case = _write_case_stable_promoted(case_root)
    short_case = _write_case_short_window_sensitive(case_root)
    triage_case = _write_case_triage_sensitive(case_root)

    return (
        stable_case,
        short_case,
        triage_case,
    )


def _write_case_stable_promoted(case_root: Path) -> CampaignExampleCaseSpec:
    case_name = "case_stable_promoted"
    rng = np.random.default_rng(505)
    n_days = 200
    n_assets = 14
    dates = pd.date_range("2023-01-03", periods=n_days, freq="B")
    assets = [f"A{i:03d}" for i in range(n_assets)]

    case_dir = case_root / case_name
    inputs_dir = case_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)

    latent = {asset: rng.normal(0.0, 1.0, size=n_days) for asset in assets}
    macro = rng.normal(0.0, 1.0, size=n_days)

    rows_price: list[dict[str, object]] = []
    for idx, asset in enumerate(assets):
        price = 40.0 + idx
        asset_latent = latent[asset]
        eps = rng.normal(0.0, 0.006, size=n_days)
        for t, date in enumerate(dates):
            pred = (1.0 * asset_latent[t - 1] if t > 0 else 0.0) + (
                0.35 * macro[t - 1] if t > 0 else 0.0
            )
            ret = 0.0046 * pred + eps[t]
            price = max(1.0, price * (1.0 + ret))
            open_price = max(price * (0.998 + 0.0002 * ((idx + t) % 7)), 1.0)
            rows_price.append(
                {"date": date, "asset": asset, "open": float(open_price), "close": float(price)}
            )

    prices = pd.DataFrame(rows_price)
    universe = pd.DataFrame(
        [{"date": date, "asset": asset, "in_universe": True} for date in dates for asset in assets]
    )

    factor_matrix = np.zeros((n_days, n_assets), dtype=float)
    for j, asset in enumerate(assets):
        factor_matrix[:, j] = 3.0 * latent[asset] + rng.normal(0.0, 0.07, size=n_days)

    rows_factor: list[dict[str, object]] = []
    rows_exposure: list[dict[str, object]] = []
    for t, date in enumerate(dates):
        x = factor_matrix[t, :]
        z = rng.normal(0.0, 1.0, size=n_assets)
        x0 = x - x.mean()
        z0 = z - z.mean()
        denom = float(np.dot(x0, x0))
        if denom > 1e-12:
            z0 = z0 - (float(np.dot(z0, x0)) / denom) * x0
        for j, asset in enumerate(assets):
            rows_factor.append(
                {
                    "date": date,
                    "asset": asset,
                    "factor": "bp",
                    "value": float(x[j]),
                }
            )
            rows_exposure.append(
                {
                    "date": date,
                    "asset": asset,
                    "size": float(z0[j]),
                }
            )

    prices_path = inputs_dir / "prices.csv"
    universe_path = inputs_dir / "universe.csv"
    factor_path = inputs_dir / "bp.csv"
    exposure_path = inputs_dir / "exposure.csv"
    prices.to_csv(prices_path, index=False)
    universe.to_csv(universe_path, index=False)
    pd.DataFrame(rows_factor).to_csv(factor_path, index=False)
    pd.DataFrame(rows_exposure).to_csv(exposure_path, index=False)

    spec_path = case_dir / "single_factor_case.json"
    _write_single_factor_spec(
        spec_path,
        case_name=case_name,
        prices_path=prices_path,
        factor_path=factor_path,
        universe_path=universe_path,
        preprocess_min_coverage=0.4,
        transaction_cost=0.0005,
        neutralization={
            "enabled": True,
            "exposures_path": str(exposure_path),
            "size_col": "size",
            "industry_col": None,
            "min_obs": 5,
            "ridge": 1e-8,
        },
    )

    return CampaignExampleCaseSpec(
        case_name=case_name,
        case_description=(
            "High-signal neutralized case engineered to remain stable and promoted across profiles."
        ),
        spec_path=spec_path,
    )


def _write_case_short_window_sensitive(case_root: Path) -> CampaignExampleCaseSpec:
    case_name = "case_short_window_sensitive"
    rng = np.random.default_rng(77)
    n_days = 200
    n_assets = 20
    dates = pd.date_range("2025-01-02", periods=n_days, freq="B")
    assets = [f"A{i:03d}" for i in range(n_assets)]

    case_dir = case_root / case_name
    inputs_dir = case_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)

    latent = {asset: rng.normal(0.0, 1.0, size=n_days) for asset in assets}
    macro = rng.normal(0.0, 1.0, size=n_days)

    rows_price: list[dict[str, object]] = []
    rows_factor: list[dict[str, object]] = []
    for idx, asset in enumerate(assets):
        price = 40.0 + idx
        asset_latent = latent[asset]
        eps = rng.normal(0.0, 0.006, size=n_days)
        factor_noise = rng.normal(0.0, 0.1, size=n_days)
        for t, date in enumerate(dates):
            pred = (1.0 * asset_latent[t - 1] if t > 0 else 0.0) + (
                0.3 * macro[t - 1] if t > 0 else 0.0
            )
            ret = 0.0042 * pred + eps[t]
            price = max(1.0, price * (1.0 + ret))
            open_price = max(price * (0.998 + 0.0002 * ((idx + t) % 7)), 1.0)
            rows_price.append(
                {"date": date, "asset": asset, "open": float(open_price), "close": float(price)}
            )
            rows_factor.append(
                {
                    "date": date,
                    "asset": asset,
                    "factor": "bp",
                    "value": float(2.2 * asset_latent[t] + factor_noise[t]),
                }
            )

    prices = pd.DataFrame(rows_price)
    factors = pd.DataFrame(rows_factor)
    universe = pd.DataFrame(
        [{"date": date, "asset": asset, "in_universe": True} for date in dates for asset in assets]
    )

    prices_path = inputs_dir / "prices.csv"
    universe_path = inputs_dir / "universe.csv"
    factor_path = inputs_dir / "bp.csv"
    prices.to_csv(prices_path, index=False)
    factors.to_csv(factor_path, index=False)
    universe.to_csv(universe_path, index=False)

    spec_path = case_dir / "single_factor_case.json"
    _write_single_factor_spec(
        spec_path,
        case_name=case_name,
        prices_path=prices_path,
        factor_path=factor_path,
        universe_path=universe_path,
        preprocess_min_coverage=0.4,
        transaction_cost=0.001,
        neutralization={
            "enabled": False,
            "min_obs": 5,
            "ridge": 1e-8,
        },
    )

    return CampaignExampleCaseSpec(
        case_name=case_name,
        case_description=(
            "Noisy stability-sensitive case that changes Level 1 verdict and "
            "promotion behavior across profiles."
        ),
        spec_path=spec_path,
    )


def _write_case_triage_sensitive(case_root: Path) -> CampaignExampleCaseSpec:
    case_name = "case_triage_sensitive"
    rng = np.random.default_rng(1357)
    n_days = 200
    n_assets = 16
    dates = pd.date_range("2024-01-02", periods=n_days, freq="B")
    assets = [f"A{i:03d}" for i in range(n_assets)]

    case_dir = case_root / case_name
    inputs_dir = case_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)

    latent = {asset: rng.normal(0.0, 1.0, size=n_days) for asset in assets}
    macro = rng.normal(0.0, 1.0, size=n_days)

    rows_price: list[dict[str, object]] = []
    rows_factor: list[dict[str, object]] = []

    for idx, asset in enumerate(assets):
        price = 60.0 + idx
        asset_latent = latent[asset]
        eps = rng.normal(0.0, 0.009, size=n_days)
        for t, date in enumerate(dates):
            pred = (0.8 * asset_latent[t - 1] if t > 0 else 0.0) + (
                0.2 * macro[t - 1] if t > 0 else 0.0
            )
            ret = 0.0024 * pred + eps[t]
            price = max(1.0, price * (1.0 + ret))
            open_price = max(price * (0.998 + 0.0002 * ((idx + t) % 7)), 1.0)
            rows_price.append(
                {"date": date, "asset": asset, "open": float(open_price), "close": float(price)}
            )

    for asset in assets:
        asset_latent = latent[asset]
        factor_noise = rng.normal(0.0, 0.45, size=n_days)
        for t, date in enumerate(dates):
            shift = -1.40 * asset_latent[t] if t >= int(n_days * 0.84) else 0.0
            value = 1.0 * asset_latent[t] + shift + factor_noise[t]
            if rng.random() < 0.10:
                value = float("nan")
            rows_factor.append(
                {
                    "date": date,
                    "asset": asset,
                    "factor": "bp",
                    "value": value,
                }
            )

    prices = pd.DataFrame(rows_price)
    factors = pd.DataFrame(rows_factor)
    universe = pd.DataFrame(
        [{"date": date, "asset": asset, "in_universe": True} for date in dates for asset in assets]
    )

    prices_path = inputs_dir / "prices.csv"
    universe_path = inputs_dir / "universe.csv"
    factor_path = inputs_dir / "bp.csv"
    prices.to_csv(prices_path, index=False)
    factors.to_csv(factor_path, index=False)
    universe.to_csv(universe_path, index=False)

    spec_path = case_dir / "single_factor_case.json"
    _write_single_factor_spec(
        spec_path,
        case_name=case_name,
        prices_path=prices_path,
        factor_path=factor_path,
        universe_path=universe_path,
        preprocess_min_coverage=0.35,
        transaction_cost=0.001,
        neutralization={
            "enabled": False,
            "min_obs": 5,
            "ridge": 1e-8,
        },
    )

    return CampaignExampleCaseSpec(
        case_name=case_name,
        case_description=(
            "Borderline case with profile-dependent campaign triage and portfolio "
            "validation recommendation."
        ),
        spec_path=spec_path,
    )


def _write_single_factor_spec(
    spec_path: Path,
    *,
    case_name: str,
    prices_path: Path,
    factor_path: Path,
    universe_path: Path,
    preprocess_min_coverage: float,
    transaction_cost: float,
    neutralization: dict[str, object],
) -> None:
    payload = {
        "name": case_name,
        "factor_name": "bp",
        "factor_path": str(factor_path),
        "prices_path": str(prices_path),
        "rebalance_frequency": "W",
        "n_quantiles": 5,
        "direction": "long",
        "universe": {
            "name": "demo_universe",
            "path": str(universe_path),
            "in_universe_column": "in_universe",
        },
        "target": {
            "kind": "forward_return",
            "horizon": 5,
        },
        "preprocess": {
            "winsorize": True,
            "winsorize_lower": 0.01,
            "winsorize_upper": 0.99,
            "standardization": "zscore",
            "min_group_size": 3,
            "min_coverage": preprocess_min_coverage,
        },
        "neutralization": neutralization,
        "transaction_cost": {
            "one_way_rate": transaction_cost,
        },
        "output": {
            "root_dir": "runs/default_research",
        },
    }
    spec_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def _build_campaign_comparison_payload(
    *,
    root_dir: Path,
    case_specs: tuple[CampaignExampleCaseSpec, ...],
    profile_campaigns: tuple[ProfileCampaignSummary, ...],
) -> dict[str, object]:
    profiles = [row.profile_name for row in profile_campaigns]
    case_lookup = _case_profile_lookup(profile_campaigns)
    transition_distribution_by_profile = {
        campaign.profile_name: project_level12_transition_distribution(
            [
                {
                    "case_name": row.case_name,
                    "level12_transition_label": row.level12_transition_label,
                    "level12_transition_reasons": list(row.level12_transition_reasons),
                }
                for row in campaign.case_summaries
            ]
        )
        for campaign in profile_campaigns
    }
    transition_profile_delta_matrix = _build_level12_transition_profile_delta_matrix(
        case_specs=case_specs,
        case_lookup=case_lookup,
        profiles=profiles,
    )
    transition_reason_profile_delta_matrix = _build_level12_transition_reason_profile_delta_matrix(
        transition_distribution_by_profile=transition_distribution_by_profile,
        profiles=profiles,
    )

    case_comparison: list[dict[str, object]] = []
    for case in case_specs:
        profile_map = case_lookup.get(case.case_name, {})
        transition_profile_delta = _build_case_level12_transition_profile_delta(
            profile_map,
            profiles=profiles,
        )
        field_differences = _case_field_differences(
            profile_map,
            fields=CAMPAIGN_PROFILE_COMPARISON_FIELDS,
        )
        changed_fields = sorted(field_differences)
        is_profile_sensitive = bool(changed_fields)
        sensitivity_label = _sensitivity_label(changed_fields)

        case_comparison.append(
            {
                "case_name": case.case_name,
                "case_description": case.case_description,
                "profile_sensitivity": sensitivity_label,
                "is_profile_sensitive": is_profile_sensitive,
                "changed_fields": changed_fields,
                "field_differences": field_differences,
                "level12_transition_profile_delta": transition_profile_delta,
                "profiles": {
                    profile: _case_profile_payload(profile_map.get(profile)) for profile in profiles
                },
            }
        )

    field_change_index = {
        field: [row["case_name"] for row in case_comparison if _has_changed_field(row, field)]
        for field in CAMPAIGN_PROFILE_COMPARISON_FIELDS
    }

    stable_cases = [
        row["case_name"] for row in case_comparison if not bool(row.get("is_profile_sensitive"))
    ]
    promoted_only_under_looser_profiles = [
        row["case_name"]
        for row in case_comparison
        if _promoted_only_under_looser_profiles(
            row,
            exploratory_profile="exploratory_screening",
            baseline_profiles=("default_research", "stricter_research"),
        )
    ]
    consistently_strong = [row["case_name"] for row in case_comparison if _consistently_strong(row)]
    highly_profile_sensitive = [
        row["case_name"]
        for row in case_comparison
        if row.get("profile_sensitivity") == "highly_profile_sensitive"
    ]
    transition_stable_cases = [
        case_name
        for row in case_comparison
        if _case_transition_delta_label(row) == _TRANSITION_DELTA_LABEL_STABLE
        for case_name in [str(row.get("case_name") or "").strip()]
        if case_name
    ]
    transition_sensitive_cases = [
        case_name
        for row in case_comparison
        if _case_transition_delta_label(row) != _TRANSITION_DELTA_LABEL_STABLE
        for case_name in [str(row.get("case_name") or "").strip()]
        if case_name
    ]
    transition_delta_label_counts = {
        label: sum(1 for row in case_comparison if _case_transition_delta_label(row) == label)
        for label in _TRANSITION_DELTA_LABELS
    }
    compact_comparison_summary = _build_compact_comparison_summary(
        case_comparison=case_comparison,
        profiles=profiles,
        transition_distribution_by_profile=transition_distribution_by_profile,
        transition_profile_delta_matrix=transition_profile_delta_matrix,
        transition_reason_profile_delta_matrix=transition_reason_profile_delta_matrix,
        transition_stable_cases=transition_stable_cases,
        transition_sensitive_cases=transition_sensitive_cases,
    )

    profile_runs = [
        {
            "profile_name": campaign.profile_name,
            "ranked_case_order": list(campaign.ranked_case_order),
            "level12_transition_distribution": transition_distribution_by_profile[
                campaign.profile_name
            ],
            "case_rows": [_profile_case_row(row) for row in campaign.case_summaries],
        }
        for campaign in profile_campaigns
    ]

    return {
        "schema_version": "1.0.0",
        "example_name": "profile_aware_campaign_level12",
        "output_root_dir": str(root_dir),
        "profiles": profiles,
        "default_profile": DEFAULT_RESEARCH_EVALUATION_CONFIG.profile_name,
        "cases": [
            {
                "case_name": row.case_name,
                "case_description": row.case_description,
                "spec_path": str(row.spec_path),
            }
            for row in case_specs
        ],
        "profile_runs": profile_runs,
        "case_comparison": case_comparison,
        "field_change_index": field_change_index,
        "campaign_level_summary": {
            "stable_cases": stable_cases,
            "promoted_only_under_looser_profiles": promoted_only_under_looser_profiles,
            "consistently_strong": consistently_strong,
            "highly_profile_sensitive": highly_profile_sensitive,
            "level12_transition_distribution_by_profile": transition_distribution_by_profile,
            "level12_transition_profile_delta_matrix": transition_profile_delta_matrix,
            "level12_transition_reason_profile_delta_matrix": (
                transition_reason_profile_delta_matrix
            ),
            "transition_stable_cases": transition_stable_cases,
            "transition_sensitive_cases": transition_sensitive_cases,
            "transition_delta_label_counts": transition_delta_label_counts,
            "compact_comparison_summary": compact_comparison_summary,
        },
    }


def _render_campaign_comparison_markdown(
    *,
    root_dir: Path,
    case_specs: tuple[CampaignExampleCaseSpec, ...],
    profile_campaigns: tuple[ProfileCampaignSummary, ...],
    comparison_payload: dict[str, object],
) -> str:
    lines = [
        "# Profile-Aware Campaign Level 1/2 Example",
        "",
        f"- Output root: `{root_dir}`",
        f"- Profiles: `{comparison_payload.get('profiles')}`",
        "",
        "## Campaign Cases",
        "",
    ]

    for case in case_specs:
        lines.append(f"- `{case.case_name}`: {case.case_description}")

    lines += ["", "## Per-Profile Campaign View", ""]

    for campaign in profile_campaigns:
        lines += [f"### {campaign.profile_name}", ""]
        lines.append(
            "| Rank | Case | Factor Verdict | Campaign Triage | Level 2 Promotion | "
            "L1->L2 Transition | Portfolio Validation Recommendation |"
        )
        lines.append("|---:|---|---|---|---|---|---|")

        rank_map = {name: idx + 1 for idx, name in enumerate(campaign.ranked_case_order)}
        by_case = {row.case_name: row for row in campaign.case_summaries}
        for case_name in campaign.ranked_case_order:
            row = by_case[case_name]
            lines.append(
                "| "
                f"{rank_map.get(case_name, 'N/A')} | "
                f"{row.case_name} | "
                f"{row.factor_verdict} | "
                f"{row.campaign_triage} | "
                f"{row.promotion_decision} | "
                f"{row.level12_transition_label} | "
                f"{row.portfolio_validation_recommendation} "
                "|"
            )

        lines.append("")

    lines += ["## Case-Level Profile Comparison", ""]
    case_comparison = comparison_payload.get("case_comparison", [])
    if not isinstance(case_comparison, list):
        case_comparison = []

    for row in case_comparison:
        if not isinstance(row, dict):
            continue
        case_name = str(row.get("case_name") or "N/A")
        sensitivity = str(row.get("profile_sensitivity") or "profile_sensitive")
        changed_fields_obj = row.get("changed_fields", [])
        changed_fields = changed_fields_obj if isinstance(changed_fields_obj, list) else []

        lines.append(f"### {case_name}")
        lines.append("")
        lines.append(f"- Sensitivity: `{sensitivity}`")
        transition_delta_obj = row.get("level12_transition_profile_delta")
        transition_delta = transition_delta_obj if isinstance(transition_delta_obj, dict) else {}
        transition_delta_label = str(
            transition_delta.get("delta_label") or _TRANSITION_DELTA_LABEL_MIXED
        )
        lines.append(f"- L1->L2 transition delta: `{transition_delta_label}`")
        transition_path = _transition_profile_path_text(
            transition_delta.get("profile_transition_labels"),
            comparison_payload.get("profiles"),
        )
        if transition_path:
            lines.append(f"- L1->L2 transition path: {transition_path}")
        if changed_fields:
            lines.append(
                "- Changed fields: " + ", ".join(f"`{str(field)}`" for field in changed_fields)
            )
        else:
            lines.append("- Changed fields: none")

        profiles_obj = row.get("profiles", {})
        profiles = profiles_obj if isinstance(profiles_obj, dict) else {}
        for profile_name, payload in sorted(profiles.items()):
            if not isinstance(payload, dict):
                continue
            lines.append(
                "- "
                f"{profile_name}: verdict=`{payload.get('factor_verdict')}`, "
                f"triage=`{payload.get('campaign_triage')}`, "
                f"promotion=`{payload.get('promotion_decision')}`, "
                f"transition=`{payload.get('level12_transition_label')}`, "
                f"portfolio_validation=`{payload.get('portfolio_validation_recommendation')}`"
            )
            reasons = payload.get("major_reasons", {})
            if isinstance(reasons, dict):
                blockers = reasons.get("promotion_blockers", [])
                blockers_text = (
                    ", ".join(str(x) for x in blockers) if isinstance(blockers, list) else ""
                )
                if blockers_text:
                    lines.append(f"- {profile_name} blockers: {blockers_text}")

        lines.append("")

    summary_obj = comparison_payload.get("campaign_level_summary", {})
    summary = summary_obj if isinstance(summary_obj, dict) else {}
    lines += ["## Campaign-Level Interpretation", ""]
    compact_summary_obj = summary.get("compact_comparison_summary")
    compact_summary = compact_summary_obj if isinstance(compact_summary_obj, dict) else {}
    compact_summary_lines = _compact_comparison_summary_lines(compact_summary)
    if compact_summary_lines:
        lines += ["### Compact Comparison Summary", ""]
        for summary_line in compact_summary_lines:
            lines.append(f"- {summary_line}")
        lines.append("")
    lines.append(f"- Cases stable across profiles: {_list_or_none(summary.get('stable_cases'))}")
    lines.append(
        "- Cases promoted only under looser profiles: "
        f"{_list_or_none(summary.get('promoted_only_under_looser_profiles'))}"
    )
    lines.append(
        f"- Cases consistently strong: {_list_or_none(summary.get('consistently_strong'))}"
    )
    lines.append(
        "- Cases highly profile-sensitive: "
        f"{_list_or_none(summary.get('highly_profile_sensitive'))}"
    )
    lines.append(
        "- Transition-stable cases (L1->L2 labels): "
        f"{_list_or_none(summary.get('transition_stable_cases'))}"
    )
    lines.append(
        "- Transition-sensitive cases (L1->L2 labels): "
        f"{_list_or_none(summary.get('transition_sensitive_cases'))}"
    )
    transition_obj = summary.get("level12_transition_distribution_by_profile")
    transition_by_profile = transition_obj if isinstance(transition_obj, dict) else {}
    if transition_by_profile:
        lines.append("- L1->L2 transition distribution by profile:")
        payload_profiles_obj = comparison_payload.get("profiles", [])
        payload_profiles: list[object] = (
            payload_profiles_obj if isinstance(payload_profiles_obj, list) else []
        )
        for profile_name in payload_profiles:
            if not isinstance(profile_name, str):
                continue
            dist_obj = transition_by_profile.get(profile_name)
            dist = dist_obj if isinstance(dist_obj, dict) else {}
            if not dist:
                lines.append(f"- {profile_name}: unavailable")
                continue
            counts = dist.get("counts_by_transition_label")
            if not isinstance(counts, dict):
                counts = {}
            n_cases = dist.get("n_cases")
            interpretation = str(dist.get("interpretation") or "N/A")
            lines.append(
                "- "
                f"{profile_name} (n={n_cases}): "
                f"Confirmed={counts.get('Confirmed at portfolio level', 0)}, "
                f"Weakened={counts.get('Weakened at portfolio level', 0)}, "
                f"Fragile={counts.get('Fragile after promotion', 0)}, "
                f"Improved={counts.get('Improved at portfolio level', 0)}, "
                f"Inconclusive={counts.get('Inconclusive transition', 0)}; "
                f"interpretation={interpretation}"
            )
            rollup_tokens = _transition_reason_rollup_tokens(dist)
            if rollup_tokens:
                lines.append(
                    f"- {profile_name} dominant transition reasons: " + "; ".join(rollup_tokens)
                )
    transition_delta_matrix_obj = summary.get("level12_transition_profile_delta_matrix")
    transition_delta_matrix = (
        transition_delta_matrix_obj if isinstance(transition_delta_matrix_obj, dict) else {}
    )
    if transition_delta_matrix:
        lines.append("- L1->L2 transition profile-delta matrix (adjacent profiles):")
        pair_rows_obj = transition_delta_matrix.get("profile_pairs")
        pair_rows = pair_rows_obj if isinstance(pair_rows_obj, list) else []
        for pair_row in pair_rows:
            if not isinstance(pair_row, dict):
                continue
            from_profile = str(pair_row.get("from_profile") or "N/A")
            to_profile = str(pair_row.get("to_profile") or "N/A")
            observed = pair_row.get("n_cases_with_observed_transition_labels")
            missing = pair_row.get("n_cases_missing_transition_labels")
            stable_count = pair_row.get("stable_count")
            changed_count = pair_row.get("changed_count")
            lines.append(
                "- "
                f"{from_profile} -> {to_profile}: observed={observed}, "
                f"stable={stable_count}, changed={changed_count}, missing={missing}"
            )
            nonzero_pairs = _render_nonzero_transition_pair_counts(
                pair_row.get("counts_by_from_to_label"),
                pair_row.get("proportions_by_from_to_label"),
            )
            lines.append(f"- {from_profile} -> {to_profile} pair counts: {nonzero_pairs}")
    transition_reason_delta_matrix_obj = summary.get(
        "level12_transition_reason_profile_delta_matrix"
    )
    transition_reason_delta_matrix = (
        transition_reason_delta_matrix_obj
        if isinstance(transition_reason_delta_matrix_obj, dict)
        else {}
    )
    if transition_reason_delta_matrix:
        lines.append("- L1->L2 dominant reason deltas by profile pair (adjacent profiles):")
        pair_rows_obj = transition_reason_delta_matrix.get("profile_pairs")
        pair_rows = pair_rows_obj if isinstance(pair_rows_obj, list) else []
        for pair_row in pair_rows:
            if not isinstance(pair_row, dict):
                continue
            from_profile = str(pair_row.get("from_profile") or "N/A")
            to_profile = str(pair_row.get("to_profile") or "N/A")
            observed_labels = pair_row.get("n_transition_labels_with_observed_reasons")
            shifted_labels = pair_row.get("n_transition_labels_with_reason_shift")
            stable_labels = pair_row.get("n_transition_labels_reason_stable")
            delta_counts_obj = pair_row.get("reason_bucket_delta_counts")
            delta_counts = delta_counts_obj if isinstance(delta_counts_obj, dict) else {}
            lines.append(
                "- "
                f"{from_profile} -> {to_profile}: observed_labels={observed_labels}, "
                f"shifted_labels={shifted_labels}, stable_labels={stable_labels}, "
                f"added={delta_counts.get('added', 0)}, "
                f"removed={delta_counts.get('removed', 0)}, "
                f"increased={delta_counts.get('increased', 0)}, "
                f"decreased={delta_counts.get('decreased', 0)}"
            )
            by_label_obj = pair_row.get("reason_delta_by_transition_label")
            by_label = by_label_obj if isinstance(by_label_obj, dict) else {}
            for transition_label in LEVEL12_TRANSITION_TAXONOMY:
                label_obj = by_label.get(transition_label)
                label_row = label_obj if isinstance(label_obj, dict) else {}
                from_reasons_obj = label_row.get("from_profile_dominant_reasons")
                from_reasons = from_reasons_obj if isinstance(from_reasons_obj, list) else []
                to_reasons_obj = label_row.get("to_profile_dominant_reasons")
                to_reasons = to_reasons_obj if isinstance(to_reasons_obj, list) else []
                deltas_obj = label_row.get("reason_bucket_deltas")
                deltas = deltas_obj if isinstance(deltas_obj, dict) else {}
                is_shifted = bool(label_row.get("is_reason_shifted"))
                if not from_reasons and not to_reasons:
                    continue
                from_tokens = _render_reason_stat_tokens(from_reasons)
                to_tokens = _render_reason_stat_tokens(to_reasons)
                delta_tokens = _render_reason_delta_bucket_tokens(deltas)
                if is_shifted:
                    lines.append(
                        "- "
                        f"{from_profile} -> {to_profile} [{transition_label}]: "
                        f"{from_profile} dominant={from_tokens}; "
                        f"{to_profile} dominant={to_tokens}; "
                        f"shifts={delta_tokens}"
                    )
                else:
                    lines.append(
                        "- "
                        f"{from_profile} -> {to_profile} [{transition_label}]: "
                        f"dominant reasons stable; "
                        f"{from_profile}={from_tokens}; "
                        f"{to_profile}={to_tokens}"
                    )

    lines += [
        "",
        "## Artifacts",
        "",
        "- `campaign_profile_comparison.json` (machine-readable profile deltas)",
        "- `campaign_profile_case_matrix.csv` (flat case/profile matrix)",
        "- `campaign_profile_comparison.md` (human-readable summary)",
        "",
    ]

    return "\n".join(lines) + "\n"


def _write_case_matrix_csv(
    path: Path,
    *,
    case_specs: tuple[CampaignExampleCaseSpec, ...],
    profile_campaigns: tuple[ProfileCampaignSummary, ...],
    comparison_payload: dict[str, object],
) -> None:
    case_lookup = _case_profile_lookup(profile_campaigns)

    sensitivity_by_case: dict[str, str] = {}
    transition_delta_by_case: dict[str, str] = {}
    case_comparison_obj = comparison_payload.get("case_comparison", [])
    if isinstance(case_comparison_obj, list):
        for row in case_comparison_obj:
            if not isinstance(row, dict):
                continue
            case_name = str(row.get("case_name") or "")
            if not case_name:
                continue
            sensitivity_by_case[case_name] = str(
                row.get("profile_sensitivity") or "profile_sensitive"
            )
            transition_delta_obj = row.get("level12_transition_profile_delta")
            transition_delta = (
                transition_delta_obj if isinstance(transition_delta_obj, dict) else {}
            )
            transition_delta_by_case[case_name] = str(
                transition_delta.get("delta_label") or _TRANSITION_DELTA_LABEL_MIXED
            )

    rows: list[dict[str, object]] = []
    for case in case_specs:
        by_profile = case_lookup.get(case.case_name, {})
        for profile_name, summary in sorted(by_profile.items()):
            rows.append(
                {
                    "case_name": case.case_name,
                    "profile_name": profile_name,
                    "factor_verdict": summary.factor_verdict,
                    "campaign_triage": summary.campaign_triage,
                    "promotion_decision": summary.promotion_decision,
                    "level12_transition_label": summary.level12_transition_label,
                    "portfolio_validation_recommendation": (
                        summary.portfolio_validation_recommendation
                    ),
                    "promotion_blockers": "; ".join(summary.promotion_blockers),
                    "profile_sensitivity": sensitivity_by_case.get(case.case_name, "unknown"),
                    "level12_transition_delta_label": transition_delta_by_case.get(
                        case.case_name,
                        "unknown",
                    ),
                    "metrics_path": str(summary.metrics_path),
                    "output_dir": str(summary.output_dir),
                }
            )

    pd.DataFrame(rows).sort_values(["case_name", "profile_name"], kind="mergesort").to_csv(
        path, index=False
    )


def _build_level12_transition_profile_delta_matrix(
    *,
    case_specs: tuple[CampaignExampleCaseSpec, ...],
    case_lookup: dict[str, dict[str, CampaignCaseProfileSummary]],
    profiles: list[str],
) -> dict[str, object]:
    profile_pairs = _adjacent_profile_pairs(profiles)
    pair_rows: list[dict[str, object]] = []
    for from_profile, to_profile in profile_pairs:
        counts_by_from_to_label = _empty_transition_pair_count_matrix()
        n_cases_compared = 0
        n_missing_labels = 0
        stable_count = 0
        changed_count = 0

        for case in case_specs:
            profile_map = case_lookup.get(case.case_name, {})
            from_summary = profile_map.get(from_profile)
            to_summary = profile_map.get(to_profile)
            if from_summary is None or to_summary is None:
                continue
            n_cases_compared += 1
            from_label = from_summary.level12_transition_label
            to_label = to_summary.level12_transition_label
            if (
                from_label not in _TRANSITION_STRENGTH_SCORE
                or to_label not in _TRANSITION_STRENGTH_SCORE
            ):
                n_missing_labels += 1
                continue
            counts_by_from_to_label[from_label][to_label] += 1
            if from_label == to_label:
                stable_count += 1
            else:
                changed_count += 1

        n_observed = stable_count + changed_count
        pair_rows.append(
            {
                "from_profile": from_profile,
                "to_profile": to_profile,
                "n_cases_compared": n_cases_compared,
                "n_cases_with_observed_transition_labels": n_observed,
                "n_cases_missing_transition_labels": n_missing_labels,
                "stable_count": stable_count,
                "changed_count": changed_count,
                "stable_proportion": (stable_count / n_observed if n_observed > 0 else 0.0),
                "changed_proportion": (changed_count / n_observed if n_observed > 0 else 0.0),
                "counts_by_from_to_label": counts_by_from_to_label,
                "proportions_by_from_to_label": _transition_pair_proportion_matrix(
                    counts_by_from_to_label,
                    denominator=n_observed,
                ),
            }
        )

    return {"profile_pairs": pair_rows}


def _build_level12_transition_reason_profile_delta_matrix(
    *,
    transition_distribution_by_profile: Mapping[str, Mapping[str, object]],
    profiles: list[str],
) -> dict[str, object]:
    pair_rows: list[dict[str, object]] = []
    for from_profile, to_profile in _adjacent_profile_pairs(profiles):
        from_distribution_obj = transition_distribution_by_profile.get(from_profile)
        from_distribution = from_distribution_obj if isinstance(from_distribution_obj, dict) else {}
        to_distribution_obj = transition_distribution_by_profile.get(to_profile)
        to_distribution = to_distribution_obj if isinstance(to_distribution_obj, dict) else {}
        reason_delta_by_transition_label: dict[str, dict[str, object]] = {}
        total_added = 0
        total_removed = 0
        total_increased = 0
        total_decreased = 0
        total_stable = 0
        n_observed_labels = 0
        n_shifted_labels = 0
        n_stable_labels = 0

        for transition_label in LEVEL12_TRANSITION_TAXONOMY:
            from_rollup = _reason_rollup_for_transition_label(
                distribution=from_distribution,
                transition_label=transition_label,
            )
            to_rollup = _reason_rollup_for_transition_label(
                distribution=to_distribution,
                transition_label=transition_label,
            )
            (
                from_n_cases,
                from_dominant_reasons,
                from_reason_map,
            ) = _dominant_reason_stats_from_rollup(from_rollup)
            to_n_cases, to_dominant_reasons, to_reason_map = _dominant_reason_stats_from_rollup(
                to_rollup
            )
            if from_n_cases > 0 or to_n_cases > 0:
                n_observed_labels += 1

            reason_bucket_deltas = _build_reason_bucket_deltas(
                from_reason_map=from_reason_map,
                to_reason_map=to_reason_map,
                from_n_cases_with_label=from_n_cases,
                to_n_cases_with_label=to_n_cases,
            )
            added_rows = reason_bucket_deltas["added"]
            removed_rows = reason_bucket_deltas["removed"]
            increased_rows = reason_bucket_deltas["increased"]
            decreased_rows = reason_bucket_deltas["decreased"]
            stable_rows = reason_bucket_deltas["stable"]
            is_reason_shifted = bool(added_rows or removed_rows or increased_rows or decreased_rows)
            if from_n_cases > 0 or to_n_cases > 0:
                if is_reason_shifted:
                    n_shifted_labels += 1
                else:
                    n_stable_labels += 1

            total_added += len(added_rows)
            total_removed += len(removed_rows)
            total_increased += len(increased_rows)
            total_decreased += len(decreased_rows)
            total_stable += len(stable_rows)
            reason_delta_by_transition_label[transition_label] = {
                "from_profile_n_cases_with_label": from_n_cases,
                "to_profile_n_cases_with_label": to_n_cases,
                "from_profile_dominant_reasons": from_dominant_reasons,
                "to_profile_dominant_reasons": to_dominant_reasons,
                "reason_bucket_deltas": reason_bucket_deltas,
                "is_reason_shifted": is_reason_shifted,
            }

        pair_rows.append(
            {
                "from_profile": from_profile,
                "to_profile": to_profile,
                "n_transition_labels_with_observed_reasons": n_observed_labels,
                "n_transition_labels_with_reason_shift": n_shifted_labels,
                "n_transition_labels_reason_stable": n_stable_labels,
                "reason_bucket_delta_counts": {
                    "added": total_added,
                    "removed": total_removed,
                    "increased": total_increased,
                    "decreased": total_decreased,
                    "stable": total_stable,
                },
                "reason_delta_by_transition_label": reason_delta_by_transition_label,
            }
        )

    return {"profile_pairs": pair_rows}


def _dominant_reason_stats_from_rollup(
    rollup: dict[str, object],
) -> tuple[int, list[dict[str, object]], dict[str, tuple[int, float]]]:
    raw_n_cases = rollup.get("n_cases_with_label")
    n_cases_with_label = raw_n_cases if isinstance(raw_n_cases, int) and raw_n_cases >= 0 else 0
    top_reasons_obj = rollup.get("top_reasons")
    top_reasons = top_reasons_obj if isinstance(top_reasons_obj, list) else []
    dominant_reasons: list[dict[str, object]] = []
    reason_map: dict[str, tuple[int, float]] = {}
    for row in top_reasons:
        if not isinstance(row, dict):
            continue
        reason = str(row.get("reason") or "").strip()
        if not reason:
            continue
        raw_count = row.get("count")
        count = raw_count if isinstance(raw_count, int) and raw_count >= 0 else 0
        raw_prop = row.get("proportion_of_label_cases")
        proportion = (
            float(raw_prop)
            if isinstance(raw_prop, int | float) and not isinstance(raw_prop, bool)
            else 0.0
        )
        dominant_reasons.append(
            {
                "reason": reason,
                "count": count,
                "n_cases_with_label": n_cases_with_label,
                "proportion_of_label_cases": proportion,
            }
        )
        reason_map[reason] = (count, proportion)
    return n_cases_with_label, dominant_reasons, reason_map


def _build_reason_bucket_deltas(
    *,
    from_reason_map: dict[str, tuple[int, float]],
    to_reason_map: dict[str, tuple[int, float]],
    from_n_cases_with_label: int,
    to_n_cases_with_label: int,
) -> dict[str, list[dict[str, object]]]:
    buckets: dict[str, list[dict[str, object]]] = {
        "added": [],
        "removed": [],
        "increased": [],
        "decreased": [],
        "stable": [],
    }
    for reason in sorted(set(from_reason_map) | set(to_reason_map)):
        from_count, from_prop = from_reason_map.get(reason, (0, 0.0))
        to_count, to_prop = to_reason_map.get(reason, (0, 0.0))
        row = {
            "reason": reason,
            "from_count": from_count,
            "from_n_cases_with_label": from_n_cases_with_label,
            "from_proportion_of_label_cases": from_prop,
            "to_count": to_count,
            "to_n_cases_with_label": to_n_cases_with_label,
            "to_proportion_of_label_cases": to_prop,
            "delta_count": to_count - from_count,
            "delta_proportion_of_label_cases": to_prop - from_prop,
        }
        if from_count <= 0 and to_count > 0:
            buckets["added"].append(row)
            continue
        if from_count > 0 and to_count <= 0:
            buckets["removed"].append(row)
            continue
        delta_prop = to_prop - from_prop
        if abs(delta_prop) <= 1e-12:
            buckets["stable"].append(row)
        elif delta_prop > 0:
            buckets["increased"].append(row)
        else:
            buckets["decreased"].append(row)

    buckets["added"].sort(
        key=lambda row: (
            -_to_float_value(row.get("to_proportion_of_label_cases")),
            -_to_int_value(row.get("to_count")),
            str(row.get("reason") or "").lower(),
        )
    )
    buckets["removed"].sort(
        key=lambda row: (
            -_to_float_value(row.get("from_proportion_of_label_cases")),
            -_to_int_value(row.get("from_count")),
            str(row.get("reason") or "").lower(),
        )
    )
    buckets["increased"].sort(
        key=lambda row: (
            -_to_float_value(row.get("delta_proportion_of_label_cases")),
            -_to_int_value(row.get("delta_count")),
            str(row.get("reason") or "").lower(),
        )
    )
    buckets["decreased"].sort(
        key=lambda row: (
            _to_float_value(row.get("delta_proportion_of_label_cases")),
            _to_int_value(row.get("delta_count")),
            str(row.get("reason") or "").lower(),
        )
    )
    buckets["stable"].sort(
        key=lambda row: (
            -_to_float_value(row.get("to_proportion_of_label_cases")),
            -_to_int_value(row.get("to_count")),
            str(row.get("reason") or "").lower(),
        )
    )
    return buckets


def _render_nonzero_transition_pair_counts(
    counts_obj: object,
    proportions_obj: object,
) -> str:
    counts = counts_obj if isinstance(counts_obj, dict) else {}
    proportions = proportions_obj if isinstance(proportions_obj, dict) else {}
    rows: list[tuple[int, str]] = []
    for from_label in LEVEL12_TRANSITION_TAXONOMY:
        from_counts_obj = counts.get(from_label)
        from_counts = from_counts_obj if isinstance(from_counts_obj, dict) else {}
        from_props_obj = proportions.get(from_label)
        from_props = from_props_obj if isinstance(from_props_obj, dict) else {}
        for to_label in LEVEL12_TRANSITION_TAXONOMY:
            raw_count = from_counts.get(to_label, 0)
            count = raw_count if isinstance(raw_count, int) else 0
            if count <= 0:
                continue
            raw_prop = from_props.get(to_label, 0.0)
            prop = raw_prop if isinstance(raw_prop, float | int) else 0.0
            rows.append(
                (
                    count,
                    f"{from_label} -> {to_label}: {count} ({float(prop):.1%})",
                )
            )
    if not rows:
        return "none"
    rows.sort(key=lambda row: (-row[0], row[1]))
    return "; ".join(text for _, text in rows)


def _render_reason_stat_tokens(
    reason_rows_obj: object,
    *,
    max_items: int = 2,
) -> str:
    reason_rows = reason_rows_obj if isinstance(reason_rows_obj, list) else []
    if max_items <= 0:
        return "none"
    tokens: list[str] = []
    for row in reason_rows[:max_items]:
        if not isinstance(row, dict):
            continue
        reason = str(row.get("reason") or "").strip()
        if not reason:
            continue
        raw_count = row.get("count")
        count = raw_count if isinstance(raw_count, int) and raw_count >= 0 else 0
        raw_n_cases = row.get("n_cases_with_label")
        n_cases = raw_n_cases if isinstance(raw_n_cases, int) and raw_n_cases >= 0 else 0
        tokens.append(f"`{reason}` {_format_reason_ratio(count=count, n_cases=n_cases)}")
    if not tokens:
        return "none"
    return "; ".join(tokens)


def _render_reason_delta_bucket_tokens(
    deltas_obj: object,
    *,
    max_items_per_bucket: int = 1,
) -> str:
    deltas = deltas_obj if isinstance(deltas_obj, dict) else {}
    parts: list[str] = []
    for bucket_name in ("added", "removed", "increased", "decreased"):
        bucket_rows_obj = deltas.get(bucket_name)
        bucket_rows = bucket_rows_obj if isinstance(bucket_rows_obj, list) else []
        if not bucket_rows:
            continue
        rendered_rows = [
            _render_reason_delta_row(row)
            for row in bucket_rows[:max_items_per_bucket]
            if isinstance(row, dict)
        ]
        rendered_rows = [row for row in rendered_rows if row]
        if rendered_rows:
            parts.append(f"{bucket_name}: " + ", ".join(rendered_rows))
    if not parts:
        stable_rows_obj = deltas.get("stable")
        stable_rows = stable_rows_obj if isinstance(stable_rows_obj, list) else []
        rendered_stable = [
            _render_reason_delta_row(row)
            for row in stable_rows[:max_items_per_bucket]
            if isinstance(row, dict)
        ]
        rendered_stable = [row for row in rendered_stable if row]
        if rendered_stable:
            parts.append("stable: " + ", ".join(rendered_stable))
    return "; ".join(parts) if parts else "none"


def _render_reason_delta_row(row: dict[str, object]) -> str:
    reason = str(row.get("reason") or "").strip()
    if not reason:
        return ""
    raw_from_count = row.get("from_count")
    from_count = raw_from_count if isinstance(raw_from_count, int) and raw_from_count >= 0 else 0
    raw_to_count = row.get("to_count")
    to_count = raw_to_count if isinstance(raw_to_count, int) and raw_to_count >= 0 else 0
    raw_from_n_cases = row.get("from_n_cases_with_label")
    from_n_cases = (
        raw_from_n_cases if isinstance(raw_from_n_cases, int) and raw_from_n_cases >= 0 else 0
    )
    raw_to_n_cases = row.get("to_n_cases_with_label")
    to_n_cases = raw_to_n_cases if isinstance(raw_to_n_cases, int) and raw_to_n_cases >= 0 else 0
    raw_delta_prop = row.get("delta_proportion_of_label_cases")
    delta_prop = (
        float(raw_delta_prop)
        if isinstance(raw_delta_prop, int | float) and not isinstance(raw_delta_prop, bool)
        else 0.0
    )
    return (
        f"`{reason}` "
        f"{_format_reason_ratio(count=from_count, n_cases=from_n_cases)} -> "
        f"{_format_reason_ratio(count=to_count, n_cases=to_n_cases)} "
        f"({delta_prop * 100.0:+.1f}pp)"
    )


def _transition_reason_rollup_tokens(
    distribution: dict[str, object],
    *,
    per_label_limit: int = 1,
) -> list[str]:
    rollups_obj = distribution.get("reason_rollup_by_transition_label")
    rollups = rollups_obj if isinstance(rollups_obj, dict) else {}
    max_per_label = max(0, per_label_limit)
    tokens: list[str] = []
    for label in LEVEL12_TRANSITION_TAXONOMY:
        rollup_obj = rollups.get(label)
        rollup = rollup_obj if isinstance(rollup_obj, dict) else {}
        top_reasons_obj = rollup.get("top_reasons")
        top_reasons = top_reasons_obj if isinstance(top_reasons_obj, list) else []
        if not top_reasons:
            continue
        label_tokens: list[str] = []
        for row in top_reasons[:max_per_label]:
            if not isinstance(row, dict):
                continue
            reason = str(row.get("reason") or "").strip()
            if not reason:
                continue
            raw_count = row.get("count")
            count = raw_count if isinstance(raw_count, int) else 0
            raw_prop = row.get("proportion_of_label_cases")
            prop = raw_prop if isinstance(raw_prop, int | float) else 0.0
            label_tokens.append(f"`{reason}` ({count}, {float(prop):.1%})")
        if not label_tokens:
            continue
        tokens.append(f"{label}: {', '.join(label_tokens)}")
    return tokens


def _build_compact_comparison_summary(
    *,
    case_comparison: list[dict[str, object]],
    profiles: list[str],
    transition_distribution_by_profile: Mapping[str, Mapping[str, object]],
    transition_profile_delta_matrix: dict[str, object],
    transition_reason_profile_delta_matrix: dict[str, object],
    transition_stable_cases: list[str],
    transition_sensitive_cases: list[str],
) -> dict[str, object]:
    n_cases = len(case_comparison)
    stable_count = len(transition_stable_cases)
    sensitive_count = len(transition_sensitive_cases)
    stable_share = stable_count / n_cases if n_cases > 0 else 0.0

    compact_summary: dict[str, object] = {
        "transition_stability": {
            "n_cases": n_cases,
            "n_transition_stable_cases": stable_count,
            "n_transition_sensitive_cases": sensitive_count,
            "stable_case_share": stable_share,
            "representative_transition_stable_cases": transition_stable_cases[:3],
        },
        "most_profile_sensitive_cases": _top_profile_sensitive_cases(
            case_comparison=case_comparison,
            profiles=profiles,
            max_items=3,
        ),
        "strongest_profile_pair_shifts": _strongest_profile_pair_shifts(
            transition_profile_delta_matrix=transition_profile_delta_matrix,
            transition_reason_profile_delta_matrix=transition_reason_profile_delta_matrix,
            max_items=2,
        ),
        "weakened_fragile_reason_hotspots": _top_weakened_fragile_reasons(
            transition_distribution_by_profile=transition_distribution_by_profile,
            strictest_profile=(profiles[-1] if profiles else ""),
            max_items=3,
        ),
        "stricter_profile_impact": _stricter_profile_impact_summary(
            transition_profile_delta_matrix=transition_profile_delta_matrix
        ),
    }
    compact_summary["summary_lines"] = _compact_comparison_summary_lines(compact_summary)
    return compact_summary


def _top_profile_sensitive_cases(
    *,
    case_comparison: list[dict[str, object]],
    profiles: list[str],
    max_items: int,
) -> list[dict[str, object]]:
    max_cases = max(0, max_items)
    if max_cases <= 0:
        return []

    def _sensitivity_rank(label: str) -> int:
        if label == "highly_profile_sensitive":
            return 2
        if label == "profile_sensitive":
            return 1
        return 0

    rows: list[dict[str, object]] = []
    for row in case_comparison:
        case_name = str(row.get("case_name") or "").strip()
        if not case_name:
            continue
        changed_fields_obj = row.get("changed_fields")
        changed_fields = (
            [str(item) for item in changed_fields_obj if str(item).strip()]
            if isinstance(changed_fields_obj, list)
            else []
        )
        n_changed_fields = len(changed_fields)
        if n_changed_fields <= 0:
            continue
        sensitivity = str(row.get("profile_sensitivity") or "profile_sensitive")
        transition_delta_label = _case_transition_delta_label(row)
        transition_delta_obj = row.get("level12_transition_profile_delta")
        transition_delta = transition_delta_obj if isinstance(transition_delta_obj, dict) else {}
        transition_path = _transition_profile_path_text(
            transition_delta.get("profile_transition_labels"),
            profiles,
        )
        rows.append(
            {
                "case_name": case_name,
                "profile_sensitivity": sensitivity,
                "n_changed_fields": n_changed_fields,
                "changed_fields": changed_fields,
                "transition_delta_label": transition_delta_label,
                "transition_profile_path": transition_path,
            }
        )

    rows.sort(
        key=lambda row: (
            -_sensitivity_rank(str(row.get("profile_sensitivity") or "")),
            -_to_int_value(row.get("n_changed_fields")),
            0
            if str(row.get("transition_delta_label") or "") != _TRANSITION_DELTA_LABEL_STABLE
            else 1,
            str(row.get("case_name") or "").lower(),
        )
    )
    return rows[:max_cases]


def _strongest_profile_pair_shifts(
    *,
    transition_profile_delta_matrix: dict[str, object],
    transition_reason_profile_delta_matrix: dict[str, object],
    max_items: int,
) -> list[dict[str, object]]:
    max_pairs = max(0, max_items)
    if max_pairs <= 0:
        return []

    pair_rows_obj = transition_profile_delta_matrix.get("profile_pairs")
    pair_rows = pair_rows_obj if isinstance(pair_rows_obj, list) else []

    reason_pair_rows_obj = transition_reason_profile_delta_matrix.get("profile_pairs")
    reason_pair_rows = reason_pair_rows_obj if isinstance(reason_pair_rows_obj, list) else []
    reason_pair_by_key: dict[tuple[str, str], dict[str, object]] = {}
    for row in reason_pair_rows:
        if not isinstance(row, dict):
            continue
        from_profile = str(row.get("from_profile") or "").strip()
        to_profile = str(row.get("to_profile") or "").strip()
        if not from_profile or not to_profile:
            continue
        reason_pair_by_key[(from_profile, to_profile)] = row

    out_rows: list[dict[str, object]] = []
    for row in pair_rows:
        if not isinstance(row, dict):
            continue
        from_profile = str(row.get("from_profile") or "").strip()
        to_profile = str(row.get("to_profile") or "").strip()
        if not from_profile or not to_profile:
            continue
        observed_raw = row.get("n_cases_with_observed_transition_labels")
        observed = observed_raw if isinstance(observed_raw, int) and observed_raw >= 0 else 0
        changed_raw = row.get("changed_count")
        changed_count = changed_raw if isinstance(changed_raw, int) and changed_raw >= 0 else 0
        changed_proportion = changed_count / observed if observed > 0 else 0.0
        reason_pair = reason_pair_by_key.get((from_profile, to_profile), {})
        shifted_labels_raw = reason_pair.get("n_transition_labels_with_reason_shift")
        shifted_labels = (
            shifted_labels_raw
            if isinstance(shifted_labels_raw, int) and shifted_labels_raw >= 0
            else 0
        )
        observed_reason_labels_raw = reason_pair.get("n_transition_labels_with_observed_reasons")
        observed_reason_labels = (
            observed_reason_labels_raw
            if isinstance(observed_reason_labels_raw, int) and observed_reason_labels_raw >= 0
            else 0
        )
        out_rows.append(
            {
                "from_profile": from_profile,
                "to_profile": to_profile,
                "changed_count": changed_count,
                "n_cases_with_observed_transition_labels": observed,
                "changed_proportion": changed_proportion,
                "reason_shifted_labels": shifted_labels,
                "n_transition_labels_with_observed_reasons": observed_reason_labels,
                "top_shift_flows": _top_shift_flows(
                    counts_obj=row.get("counts_by_from_to_label"),
                    n_observed=observed,
                    max_items=2,
                ),
            }
        )

    out_rows.sort(
        key=lambda row: (
            -_to_float_value(row.get("changed_proportion")),
            -_to_int_value(row.get("changed_count")),
            -_to_int_value(row.get("reason_shifted_labels")),
            str(row.get("from_profile") or "").lower(),
            str(row.get("to_profile") or "").lower(),
        )
    )
    return out_rows[:max_pairs]


def _top_shift_flows(
    *,
    counts_obj: object,
    n_observed: int,
    max_items: int,
) -> list[dict[str, object]]:
    max_flows = max(0, max_items)
    if max_flows <= 0:
        return []
    counts = counts_obj if isinstance(counts_obj, dict) else {}
    flows: list[dict[str, object]] = []
    for from_label in LEVEL12_TRANSITION_TAXONOMY:
        from_counts_obj = counts.get(from_label)
        from_counts = from_counts_obj if isinstance(from_counts_obj, dict) else {}
        for to_label in LEVEL12_TRANSITION_TAXONOMY:
            if from_label == to_label:
                continue
            raw_count = from_counts.get(to_label, 0)
            count = raw_count if isinstance(raw_count, int) and raw_count >= 0 else 0
            if count <= 0:
                continue
            flows.append(
                {
                    "from_label": from_label,
                    "to_label": to_label,
                    "count": count,
                    "proportion_of_observed": (count / n_observed if n_observed > 0 else 0.0),
                }
            )
    flows.sort(
        key=lambda row: (
            -_to_int_value(row.get("count")),
            str(row.get("from_label") or "").lower(),
            str(row.get("to_label") or "").lower(),
        )
    )
    return flows[:max_flows]


def _top_weakened_fragile_reasons(
    *,
    transition_distribution_by_profile: Mapping[str, Mapping[str, object]],
    strictest_profile: str,
    max_items: int,
) -> dict[str, object]:
    max_reasons = max(0, max_items)
    profile_name = str(strictest_profile).strip()
    dist_obj = transition_distribution_by_profile.get(profile_name)
    distribution = dist_obj if isinstance(dist_obj, dict) else {}
    counts_obj = distribution.get("counts_by_transition_label")
    counts = counts_obj if isinstance(counts_obj, dict) else {}
    rollups_obj = distribution.get("reason_rollup_by_transition_label")
    rollups = rollups_obj if isinstance(rollups_obj, dict) else {}

    top_rows: list[dict[str, object]] = []
    for transition_label in (
        "Weakened at portfolio level",
        "Fragile after promotion",
    ):
        rollup_obj = rollups.get(transition_label)
        rollup = rollup_obj if isinstance(rollup_obj, dict) else {}
        top_reasons_obj = rollup.get("top_reasons")
        top_reasons = top_reasons_obj if isinstance(top_reasons_obj, list) else []
        n_cases_obj = rollup.get("n_cases_with_label")
        n_cases_with_label = n_cases_obj if isinstance(n_cases_obj, int) and n_cases_obj >= 0 else 0
        for row in top_reasons:
            if not isinstance(row, dict):
                continue
            reason = str(row.get("reason") or "").strip()
            if not reason:
                continue
            raw_count = row.get("count")
            count = raw_count if isinstance(raw_count, int) and raw_count >= 0 else 0
            raw_prop = row.get("proportion_of_label_cases")
            proportion = (
                float(raw_prop)
                if isinstance(raw_prop, int | float) and not isinstance(raw_prop, bool)
                else 0.0
            )
            top_rows.append(
                {
                    "transition_label": transition_label,
                    "reason": reason,
                    "count": count,
                    "n_cases_with_label": n_cases_with_label,
                    "proportion_of_label_cases": proportion,
                }
            )
    top_rows.sort(
        key=lambda row: (
            -_to_int_value(row.get("count")),
            -_to_float_value(row.get("proportion_of_label_cases")),
            str(row.get("transition_label") or "").lower(),
            str(row.get("reason") or "").lower(),
        )
    )
    return {
        "profile_name": profile_name,
        "n_weakened_cases": int(counts.get("Weakened at portfolio level", 0) or 0),
        "n_fragile_cases": int(counts.get("Fragile after promotion", 0) or 0),
        "top_reasons": top_rows[:max_reasons],
    }


def _stricter_profile_impact_summary(
    *,
    transition_profile_delta_matrix: dict[str, object],
) -> dict[str, object]:
    pair_rows_obj = transition_profile_delta_matrix.get("profile_pairs")
    pair_rows = pair_rows_obj if isinstance(pair_rows_obj, list) else []

    out_rows: list[dict[str, object]] = []
    total_promotion_reduction = 0
    total_robustness_reduction = 0
    total_observed = 0
    for row in pair_rows:
        if not isinstance(row, dict):
            continue
        from_profile = str(row.get("from_profile") or "").strip()
        to_profile = str(row.get("to_profile") or "").strip()
        if not from_profile or not to_profile:
            continue
        observed_obj = row.get("n_cases_with_observed_transition_labels")
        observed = observed_obj if isinstance(observed_obj, int) and observed_obj >= 0 else 0
        counts_obj = row.get("counts_by_from_to_label")
        promotion_reduction, robustness_reduction = _pair_reduction_counts(counts_obj=counts_obj)
        total_promotion_reduction += promotion_reduction
        total_robustness_reduction += robustness_reduction
        total_observed += observed
        out_rows.append(
            {
                "from_profile": from_profile,
                "to_profile": to_profile,
                "n_cases_with_observed_transition_labels": observed,
                "promotion_reduction_count": promotion_reduction,
                "robustness_reduction_count": robustness_reduction,
                "dominant_reduction_mode": _dominant_reduction_mode(
                    promotion_reduction,
                    robustness_reduction,
                ),
            }
        )
    return {
        "profile_pairs": out_rows,
        "aggregate": {
            "n_profile_pairs": len(out_rows),
            "n_cases_with_observed_transition_labels": total_observed,
            "promotion_reduction_count": total_promotion_reduction,
            "robustness_reduction_count": total_robustness_reduction,
            "dominant_reduction_mode": _dominant_reduction_mode(
                total_promotion_reduction,
                total_robustness_reduction,
            ),
        },
    }


def _compact_comparison_summary_lines(compact_summary: dict[str, object]) -> list[str]:
    if not compact_summary:
        return []
    lines: list[str] = []

    transition_stability_obj = compact_summary.get("transition_stability")
    transition_stability = (
        transition_stability_obj if isinstance(transition_stability_obj, dict) else {}
    )
    n_cases = int(transition_stability.get("n_cases", 0) or 0)
    stable_count = int(transition_stability.get("n_transition_stable_cases", 0) or 0)
    sensitive_count = int(transition_stability.get("n_transition_sensitive_cases", 0) or 0)
    stable_share_raw = transition_stability.get("stable_case_share", 0.0)
    stable_share = (
        float(stable_share_raw)
        if isinstance(stable_share_raw, int | float) and not isinstance(stable_share_raw, bool)
        else 0.0
    )
    stable_case_preview_obj = transition_stability.get("representative_transition_stable_cases")
    stable_case_preview = (
        [str(item) for item in stable_case_preview_obj if str(item).strip()]
        if isinstance(stable_case_preview_obj, list)
        else []
    )
    preview_suffix = (
        f"; representative={', '.join(stable_case_preview)}" if stable_case_preview else ""
    )
    lines.append(
        f"Transition stability: {stable_count}/{n_cases} stable ({stable_share:.1%}), "
        f"sensitive={sensitive_count}{preview_suffix}."
    )

    top_sensitive_cases_obj = compact_summary.get("most_profile_sensitive_cases")
    top_sensitive_cases = (
        top_sensitive_cases_obj if isinstance(top_sensitive_cases_obj, list) else []
    )
    sensitive_tokens: list[str] = []
    for row in top_sensitive_cases:
        if not isinstance(row, dict):
            continue
        case_name = str(row.get("case_name") or "").strip()
        if not case_name:
            continue
        sensitive_tokens.append(
            f"{case_name} (changed_fields={int(row.get('n_changed_fields') or 0)}, "
            f"delta={row.get('transition_delta_label')})"
        )
    if sensitive_tokens:
        lines.append("Most profile-sensitive cases: " + "; ".join(sensitive_tokens) + ".")

    strongest_pair_shifts_obj = compact_summary.get("strongest_profile_pair_shifts")
    strongest_pair_shifts = (
        strongest_pair_shifts_obj if isinstance(strongest_pair_shifts_obj, list) else []
    )
    if strongest_pair_shifts and isinstance(strongest_pair_shifts[0], dict):
        first_pair = strongest_pair_shifts[0]
        from_profile = str(first_pair.get("from_profile") or "N/A")
        to_profile = str(first_pair.get("to_profile") or "N/A")
        changed_count = int(first_pair.get("changed_count", 0) or 0)
        observed = int(first_pair.get("n_cases_with_observed_transition_labels", 0) or 0)
        changed_prop_raw = first_pair.get("changed_proportion", 0.0)
        changed_prop = (
            float(changed_prop_raw)
            if isinstance(changed_prop_raw, int | float) and not isinstance(changed_prop_raw, bool)
            else 0.0
        )
        shifted_labels = int(first_pair.get("reason_shifted_labels", 0) or 0)
        observed_labels = int(first_pair.get("n_transition_labels_with_observed_reasons", 0) or 0)
        top_flows_obj = first_pair.get("top_shift_flows")
        top_flows = top_flows_obj if isinstance(top_flows_obj, list) else []
        flow_tokens: list[str] = []
        for flow in top_flows:
            if not isinstance(flow, dict):
                continue
            from_label = str(flow.get("from_label") or "").strip()
            to_label = str(flow.get("to_label") or "").strip()
            if not from_label or not to_label:
                continue
            flow_tokens.append(f"{from_label} -> {to_label} ({int(flow.get('count', 0) or 0)})")
        flows_text = "; ".join(flow_tokens) if flow_tokens else "none"
        lines.append(
            f"Strongest profile-pair shift: {from_profile} -> {to_profile} "
            f"changed={changed_count}/{observed} ({changed_prop:.1%}), "
            f"reason_shifted_labels={shifted_labels}/{observed_labels}, "
            f"top_flows={flows_text}."
        )

    reason_hotspots_obj = compact_summary.get("weakened_fragile_reason_hotspots")
    reason_hotspots = reason_hotspots_obj if isinstance(reason_hotspots_obj, dict) else {}
    profile_name = str(reason_hotspots.get("profile_name") or "").strip()
    top_reasons_obj = reason_hotspots.get("top_reasons")
    top_reasons = top_reasons_obj if isinstance(top_reasons_obj, list) else []
    reason_tokens: list[str] = []
    for row in top_reasons:
        if not isinstance(row, dict):
            continue
        reason = str(row.get("reason") or "").strip()
        transition_label = str(row.get("transition_label") or "").strip()
        if not reason or not transition_label:
            continue
        count = int(row.get("count", 0) or 0)
        n_cases_with_label = int(row.get("n_cases_with_label", 0) or 0)
        reason_tokens.append(
            f"{reason} [{transition_label}] "
            f"{_format_reason_ratio(count=count, n_cases=n_cases_with_label)}"
        )
    if reason_tokens:
        lines.append(
            f"Most common weakened/fragile reasons under {profile_name}: "
            + "; ".join(reason_tokens)
            + "."
        )

    stricter_impact_obj = compact_summary.get("stricter_profile_impact")
    stricter_impact = stricter_impact_obj if isinstance(stricter_impact_obj, dict) else {}
    aggregate_obj = stricter_impact.get("aggregate")
    aggregate = aggregate_obj if isinstance(aggregate_obj, dict) else {}
    dominant_mode = str(aggregate.get("dominant_reduction_mode") or "none")
    promotion_reduction = int(aggregate.get("promotion_reduction_count", 0) or 0)
    robustness_reduction = int(aggregate.get("robustness_reduction_count", 0) or 0)
    n_pairs = int(aggregate.get("n_profile_pairs", 0) or 0)
    lines.append(
        f"Stricter profile impact: {dominant_mode} "
        f"(promotion_reduction={promotion_reduction}, "
        f"robustness_reduction={robustness_reduction}, adjacent_pairs={n_pairs})."
    )

    return lines


def _profile_case_row(row: CampaignCaseProfileSummary) -> dict[str, object]:
    return {
        "case_name": row.case_name,
        "profile_name": row.profile_name,
        "output_dir": str(row.output_dir),
        "run_manifest_path": str(row.run_manifest_path),
        "metrics_path": str(row.metrics_path),
        "summary_path": str(row.summary_path),
        "experiment_card_path": str(row.experiment_card_path),
        "case_report_path": str(row.case_report_path) if row.case_report_path else None,
        "factor_verdict": row.factor_verdict,
        "factor_verdict_reasons": list(row.factor_verdict_reasons),
        "campaign_triage": row.campaign_triage,
        "campaign_triage_reasons": list(row.campaign_triage_reasons),
        "promotion_decision": row.promotion_decision,
        "promotion_reasons": list(row.promotion_reasons),
        "promotion_blockers": list(row.promotion_blockers),
        "level12_transition_label": row.level12_transition_label,
        "level12_transition_reasons": list(row.level12_transition_reasons),
        "portfolio_validation_status": row.portfolio_validation_status,
        "portfolio_validation_recommendation": row.portfolio_validation_recommendation,
        "portfolio_validation_major_risks": list(row.portfolio_validation_major_risks),
    }


def _case_profile_payload(summary: CampaignCaseProfileSummary | None) -> dict[str, object]:
    if summary is None:
        return {
            "factor_verdict": "N/A",
            "campaign_triage": "N/A",
            "promotion_decision": "N/A",
            "level12_transition_label": "N/A",
            "level12_transition_reasons": [],
            "portfolio_validation_status": "N/A",
            "portfolio_validation_recommendation": "N/A",
            "major_reasons": {
                "factor_verdict_reasons": [],
                "campaign_triage_reasons": [],
                "promotion_reasons": [],
                "promotion_blockers": [],
                "level12_transition_reasons": [],
                "portfolio_validation_major_risks": [],
            },
            "artifact_paths": {},
        }

    return {
        "factor_verdict": summary.factor_verdict,
        "campaign_triage": summary.campaign_triage,
        "promotion_decision": summary.promotion_decision,
        "level12_transition_label": summary.level12_transition_label,
        "level12_transition_reasons": list(summary.level12_transition_reasons),
        "portfolio_validation_status": summary.portfolio_validation_status,
        "portfolio_validation_recommendation": summary.portfolio_validation_recommendation,
        "major_reasons": {
            "factor_verdict_reasons": list(summary.factor_verdict_reasons),
            "campaign_triage_reasons": list(summary.campaign_triage_reasons),
            "promotion_reasons": list(summary.promotion_reasons),
            "promotion_blockers": list(summary.promotion_blockers),
            "level12_transition_reasons": list(summary.level12_transition_reasons),
            "portfolio_validation_major_risks": list(summary.portfolio_validation_major_risks),
        },
        "artifact_paths": {
            "output_dir": str(summary.output_dir),
            "run_manifest_path": str(summary.run_manifest_path),
            "metrics_path": str(summary.metrics_path),
            "summary_path": str(summary.summary_path),
            "experiment_card_path": str(summary.experiment_card_path),
            "case_report_path": (
                str(summary.case_report_path) if summary.case_report_path else None
            ),
        },
    }


def _load_metrics_payload(metrics_path: Path) -> dict[str, object]:
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AlphaLabDataError("metrics payload root must be an object")
    validate_level12_artifact_payload(
        payload,
        artifact_name=metrics_path.name,
        source=metrics_path,
    )
    metrics = payload.get("metrics", {})
    if not isinstance(metrics, dict):
        raise AlphaLabDataError("metrics payload field must be an object")
    return metrics


def _list_or_none(value: object) -> str:
    if isinstance(value, list) and value:
        return ", ".join(f"`{str(item)}`" for item in value)
    return "none"


