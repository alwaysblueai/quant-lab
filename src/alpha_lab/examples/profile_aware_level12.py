from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from alpha_lab.exceptions import AlphaLabConfigError, AlphaLabDataError
from alpha_lab.key_metrics_contracts import (
    PROFILE_AWARE_LEVEL12_OBSERVED_DIFFERENCE_FIELDS,
    project_campaign_profile_summary_metrics,
)
from alpha_lab.real_cases.single_factor.pipeline import run_single_factor_case
from alpha_lab.reporting.renderers import write_case_report
from alpha_lab.research_evaluation_config import (
    AVAILABLE_RESEARCH_EVALUATION_PROFILES,
    DEFAULT_RESEARCH_EVALUATION_CONFIG,
)

DEFAULT_PROFILE_AWARE_LEVEL12_PROFILES: tuple[str, str] = (
    "exploratory_screening",
    "default_research",
)


@dataclass(frozen=True)
class ProfileRunSummary:
    profile_name: str
    output_dir: Path
    run_manifest_path: Path
    metrics_path: Path
    summary_path: Path
    experiment_card_path: Path
    case_report_path: Path | None
    factor_verdict: str
    campaign_triage: str
    promotion_decision: str
    level12_transition_label: str
    portfolio_validation_status: str
    portfolio_validation_recommendation: str


@dataclass(frozen=True)
class ProfileAwareLevel12ExampleResult:
    root_dir: Path
    case_name: str
    spec_path: Path
    profile_runs: tuple[ProfileRunSummary, ...]
    comparison_json_path: Path
    comparison_markdown_path: Path


def run_profile_aware_level12_example(
    *,
    output_root_dir: str | Path = "dist/examples/profile_aware_level12",
    profiles: tuple[str, ...] = DEFAULT_PROFILE_AWARE_LEVEL12_PROFILES,
    case_name: str = "profile_aware_bp_single_factor",
    render_report: bool = True,
    clean_output: bool = True,
) -> ProfileAwareLevel12ExampleResult:
    selected_profiles = _normalize_profiles(profiles)
    root_dir = Path(output_root_dir).resolve()

    if clean_output and root_dir.exists():
        shutil.rmtree(root_dir)
    root_dir.mkdir(parents=True, exist_ok=True)

    spec_path = _write_example_single_factor_case(root_dir, case_name=case_name)
    run_summaries: list[ProfileRunSummary] = []

    for profile in selected_profiles:
        case_result = run_single_factor_case(
            spec_path,
            output_root_dir=root_dir / "runs" / profile,
            evaluation_profile=profile,
            vault_export_mode="skip",
        )
        case_report_path = (
            write_case_report(case_result.output_dir, overwrite=True) if render_report else None
        )
        metrics_payload = _load_metrics_payload(case_result.artifact_paths["metrics"])
        profile_summary = project_campaign_profile_summary_metrics(metrics_payload)
        run_summaries.append(
            ProfileRunSummary(
                profile_name=profile,
                output_dir=case_result.output_dir,
                run_manifest_path=case_result.artifact_paths["run_manifest"],
                metrics_path=case_result.artifact_paths["metrics"],
                summary_path=case_result.artifact_paths["summary"],
                experiment_card_path=case_result.artifact_paths["experiment_card"],
                case_report_path=case_report_path,
                factor_verdict=profile_summary["factor_verdict"] or "N/A",
                campaign_triage=profile_summary["campaign_triage"] or "N/A",
                promotion_decision=profile_summary["promotion_decision"] or "N/A",
                level12_transition_label=(profile_summary["level12_transition_label"] or "N/A"),
                portfolio_validation_status=(
                    profile_summary["portfolio_validation_status"] or "N/A"
                ),
                portfolio_validation_recommendation=(
                    profile_summary["portfolio_validation_recommendation"] or "N/A"
                ),
            )
        )

    comparison_payload = _build_comparison_payload(
        root_dir=root_dir,
        case_name=case_name,
        spec_path=spec_path,
        profile_runs=tuple(run_summaries),
    )
    comparison_json_path = root_dir / "profile_comparison.json"
    comparison_markdown_path = root_dir / "profile_comparison.md"
    observed_differences_obj = comparison_payload.get("observed_differences", {})
    observed_differences: dict[str, Any]
    if isinstance(observed_differences_obj, dict):
        observed_differences = observed_differences_obj
    else:
        observed_differences = {}
    comparison_json_path.write_text(
        json.dumps(comparison_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    comparison_markdown_path.write_text(
        _render_comparison_markdown(
            case_name=case_name,
            spec_path=spec_path,
            root_dir=root_dir,
            profile_runs=tuple(run_summaries),
            observed_differences=observed_differences,
        ),
        encoding="utf-8",
    )

    return ProfileAwareLevel12ExampleResult(
        root_dir=root_dir,
        case_name=case_name,
        spec_path=spec_path,
        profile_runs=tuple(run_summaries),
        comparison_json_path=comparison_json_path,
        comparison_markdown_path=comparison_markdown_path,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_profile_aware_level12_example",
        description=(
            "Run one deterministic Level 1/2 single-factor example under multiple "
            "evaluation profiles and export a side-by-side comparison."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-root-dir",
        default="dist/examples/profile_aware_level12",
        help="Output directory for generated inputs, profile runs, and comparison artifacts.",
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=list(DEFAULT_PROFILE_AWARE_LEVEL12_PROFILES),
        choices=sorted(AVAILABLE_RESEARCH_EVALUATION_PROFILES),
        help="Evaluation profiles to run for the same example case.",
    )
    parser.add_argument(
        "--case-name",
        default="profile_aware_bp_single_factor",
        help="Case name embedded into generated artifacts.",
    )
    parser.add_argument(
        "--no-render-report",
        action="store_true",
        help="Skip case_report.md rendering for each profile run.",
    )
    parser.add_argument(
        "--no-clean-output",
        action="store_true",
        help="Keep any existing output directory content and append/overwrite in place.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    result = run_profile_aware_level12_example(
        output_root_dir=args.output_root_dir,
        profiles=tuple(args.profiles),
        case_name=args.case_name,
        render_report=not bool(args.no_render_report),
        clean_output=not bool(args.no_clean_output),
    )

    print("")
    print("  Workflow : profile-aware-level12-example")
    print("  Status   : success")
    print(f"  Output   : {result.root_dir}")
    print(f"  Case Spec: {result.spec_path}")
    print(f"  Compare  : {result.comparison_markdown_path}")
    print(f"  CompareJ : {result.comparison_json_path}")
    for row in result.profile_runs:
        print(f"  Profile  : {row.profile_name}")
        print(f"    Output               : {row.output_dir}")
        print(f"    Factor Verdict       : {row.factor_verdict}")
        print(f"    Campaign Triage      : {row.campaign_triage}")
        print(f"    Level 2 Promotion    : {row.promotion_decision}")
        print(f"    L1->L2 Transition    : {row.level12_transition_label}")
        print(
            "    Portfolio Validation : "
            f"{row.portfolio_validation_status} ({row.portfolio_validation_recommendation})"
        )
        print(f"    Manifest             : {row.run_manifest_path}")
        print(f"    Metrics              : {row.metrics_path}")
        print(f"    Summary              : {row.summary_path}")
        print(f"    Card                 : {row.experiment_card_path}")
        print(f"    Report               : {row.case_report_path}")
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


def _write_example_single_factor_case(root_dir: Path, *, case_name: str) -> Path:
    inputs_dir = root_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)

    prices, factors, universe = _synthetic_case_tables()
    prices_path = inputs_dir / "prices.csv"
    factor_path = inputs_dir / "bp.csv"
    universe_path = inputs_dir / "universe.csv"

    prices.to_csv(prices_path, index=False)
    factors.to_csv(factor_path, index=False)
    universe.to_csv(universe_path, index=False)

    payload = {
        "name": case_name,
        "factor_name": "bp",
        "factor_path": "inputs/bp.csv",
        "prices_path": "inputs/prices.csv",
        "rebalance_frequency": "W",
        "n_quantiles": 5,
        "direction": "long",
        "universe": {
            "name": "demo_universe",
            "path": "inputs/universe.csv",
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
            "min_coverage": 0.5,
        },
        "neutralization": {
            "enabled": False,
            "min_obs": 5,
            "ridge": 1e-8,
        },
        "transaction_cost": {
            "one_way_rate": 0.001,
        },
        "output": {
            "root_dir": "runs/default_research",
        },
    }
    spec_path = root_dir / "profile_aware_single_factor_case.json"
    spec_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    return spec_path


def _synthetic_case_tables(
    *,
    n_assets: int = 12,
    n_days: int = 200,
    seed: int = 20260326,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    assets = [f"A{i:03d}" for i in range(n_assets)]

    rows_price: list[dict[str, object]] = []
    rows_factor: list[dict[str, object]] = []
    rows_universe: list[dict[str, object]] = []

    for i, asset in enumerate(assets):
        price = 50.0 + i
        latent = rng.normal(0.0, 1.0, size=n_days)
        for t, date in enumerate(dates):
            open_price = price
            pred = latent[t - 1] if t > 0 else 0.0
            ret = 0.0018 * pred + rng.normal(0.0, 0.01)
            price = max(price * (1.0 + ret), 1.0)
            factor_val = latent[t] + rng.normal(0.0, 0.25)

            rows_price.append(
                {"date": date, "asset": asset, "open": float(open_price), "close": float(price)}
            )
            rows_factor.append(
                {
                    "date": date,
                    "asset": asset,
                    "factor": "bp",
                    "value": float(factor_val),
                }
            )
            rows_universe.append(
                {
                    "date": date,
                    "asset": asset,
                    "in_universe": not (i == 0 and t < 3),
                }
            )

    return (
        pd.DataFrame(rows_price),
        pd.DataFrame(rows_factor),
        pd.DataFrame(rows_universe),
    )


def _load_metrics_payload(metrics_path: Path) -> dict[str, object]:
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AlphaLabDataError("metrics payload root must be an object")
    metrics = payload.get("metrics", {})
    if not isinstance(metrics, dict):
        raise AlphaLabDataError("metrics payload field must be an object")
    return metrics


def _build_comparison_payload(
    *,
    root_dir: Path,
    case_name: str,
    spec_path: Path,
    profile_runs: tuple[ProfileRunSummary, ...],
) -> dict[str, object]:
    rows = [
        {
            "profile_name": row.profile_name,
            "output_dir": str(row.output_dir),
            "run_manifest_path": str(row.run_manifest_path),
            "metrics_path": str(row.metrics_path),
            "summary_path": str(row.summary_path),
            "experiment_card_path": str(row.experiment_card_path),
            "case_report_path": str(row.case_report_path) if row.case_report_path else None,
            "factor_verdict": row.factor_verdict,
            "campaign_triage": row.campaign_triage,
            "promotion_decision": row.promotion_decision,
            "level12_transition_label": row.level12_transition_label,
            "portfolio_validation_status": row.portfolio_validation_status,
            "portfolio_validation_recommendation": row.portfolio_validation_recommendation,
        }
        for row in profile_runs
    ]
    observed_differences = _collect_observed_differences(profile_runs)
    return {
        "schema_version": "1.0.0",
        "example_name": "profile_aware_level12",
        "case_name": case_name,
        "case_spec_path": str(spec_path),
        "output_root_dir": str(root_dir),
        "profiles": [row.profile_name for row in profile_runs],
        "default_profile": DEFAULT_RESEARCH_EVALUATION_CONFIG.profile_name,
        "runs": rows,
        "observed_differences": observed_differences,
    }


def _collect_observed_differences(
    profile_runs: tuple[ProfileRunSummary, ...],
) -> dict[str, dict[str, str]]:
    differences: dict[str, dict[str, str]] = {}
    for field in PROFILE_AWARE_LEVEL12_OBSERVED_DIFFERENCE_FIELDS:
        by_profile = {row.profile_name: str(getattr(row, field)) for row in profile_runs}
        if len(set(by_profile.values())) > 1:
            differences[field] = by_profile
    return differences


def _render_comparison_markdown(
    *,
    case_name: str,
    spec_path: Path,
    root_dir: Path,
    profile_runs: tuple[ProfileRunSummary, ...],
    observed_differences: dict[str, Any],
) -> str:
    lines = [
        "# Profile-Aware Level 1/2 Example",
        "",
        f"- Case: `{case_name}`",
        f"- Spec: `{spec_path}`",
        f"- Output root: `{root_dir}`",
        "",
        "## Run Results",
        "",
        (
            "| Profile | Factor Verdict | Campaign Triage | Level 2 Promotion | "
            "L1->L2 Transition | "
            "Portfolio Validation | Recommendation | Output Directory |"
        ),
        "|---|---|---|---|---|---|---|---|",
    ]

    for row in profile_runs:
        lines.append(
            "| "
            f"{row.profile_name} | "
            f"{row.factor_verdict} | "
            f"{row.campaign_triage} | "
            f"{row.promotion_decision} | "
            f"{row.level12_transition_label} | "
            f"{row.portfolio_validation_status} | "
            f"{row.portfolio_validation_recommendation} | "
            f"{row.output_dir} |"
        )

    lines += ["", "## Observed Profile Differences", ""]
    if not observed_differences:
        lines.append("- None in this run.")
    else:
        for field, by_profile in observed_differences.items():
            details = ", ".join(
                f"{profile}={value}" for profile, value in sorted(by_profile.items())
            )
            lines.append(f"- `{field}`: {details}")

    lines += [
        "",
        "## Promotion Check",
        "",
        (
            "- Inspect `metrics.json` for `promotion_decision` "
            "(`Promote to Level 2` means promoted)."
        ),
    ]
    return "\n".join(lines) + "\n"
