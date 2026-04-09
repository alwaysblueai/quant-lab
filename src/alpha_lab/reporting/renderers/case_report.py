from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import cast

import pandas as pd

from alpha_lab.key_metrics_contracts import (
    CampaignProfileSummaryMetrics,
    NeutralizationComparisonMetrics,
    PortfolioValidationMetrics,
    RollingStabilityMetrics,
    UncertaintyEvidenceMetrics,
    project_campaign_profile_summary_metrics,
    project_portfolio_validation_metrics,
    project_promotion_gate_metrics,
)
from alpha_lab.reporting.display_helpers import (
    as_object_dict,
    as_object_list,
    format_text_list,
    parse_text_list,
    portfolio_validation_benchmark_note,
    portfolio_validation_note,
    to_finite_float,
)

from .templates import (
    CASE_SECTION_TITLES,
    PLACEHOLDER_INTERPRETATION,
    PLACEHOLDER_NEXT_STEPS,
    PLACEHOLDER_OBJECTIVE,
    format_metric,
    format_text,
    markdown_table,
    section_lines,
)


def render_case_report(case_output_dir: str | Path) -> str:
    """Render a standardized markdown report for one case output directory."""

    case_dir = Path(case_output_dir).resolve()
    manifest = _load_required_json(case_dir / "run_manifest.json")
    metrics_payload = _load_required_json(case_dir / "metrics.json")
    metrics = _extract_metrics(metrics_payload)
    promotion_gate_metrics = project_promotion_gate_metrics(metrics)
    profile_summary_metrics = project_campaign_profile_summary_metrics(metrics)
    portfolio_metrics = project_portfolio_validation_metrics(metrics)
    core_metrics = promotion_gate_metrics["core"]
    uncertainty_metrics = promotion_gate_metrics["uncertainty"]
    rolling_metrics = promotion_gate_metrics["rolling"]
    neutralization_metrics = promotion_gate_metrics["neutralization"]
    spec = as_object_dict(manifest.get("spec"))
    package_type = _resolve_package_type(manifest.get("artifact_type"))
    case_name = format_text(manifest.get("case_name"), na=case_dir.name)

    summary_text = _read_text_if_exists(case_dir / "summary.md")

    group_returns_path = case_dir / "group_returns.csv"
    turnover_path = case_dir / "turnover.csv"
    coverage_path = case_dir / "coverage.csv"
    group_summary = _group_return_summary(group_returns_path)
    mean_turnover = _mean_csv_metric(turnover_path, "turnover")
    mean_coverage = _mean_csv_metric(coverage_path, "coverage")

    objective_text = _extract_objective(summary_text)
    preprocess = as_object_dict(spec.get("preprocess"))
    if not preprocess:
        preprocess = as_object_dict(spec.get("feature_preprocess"))
    target = as_object_dict(spec.get("target"))
    universe = as_object_dict(spec.get("universe"))
    neutralization = as_object_dict(spec.get("neutralization"))
    model_spec = as_object_dict(spec.get("model"))

    lines: list[str] = [f"# Case Report: {case_name}", ""]

    lines.extend(
        section_lines(
            CASE_SECTION_TITLES[0],
            [
                f"- Case name: `{case_name}`",
                f"- Package type: `{package_type}`",
            ],
        )
    )

    lines.extend(
        section_lines(
            CASE_SECTION_TITLES[1],
            [objective_text or PLACEHOLDER_OBJECTIVE],
        )
    )

    lines.extend(
        section_lines(
            CASE_SECTION_TITLES[2],
            _signal_definition_lines(
                package_type=package_type,
                spec=spec,
                metrics=metrics,
                model_spec=model_spec,
                preprocess=preprocess,
            ),
        )
    )

    target_kind = target.get("kind", metrics.get("target_kind"))
    target_horizon = target.get("horizon", metrics.get("target_horizon"))
    data_setup_lines = [
        f"- Universe: `{format_text(universe.get('name'))}`",
        (
            "- Label/target: "
            f"`{format_text(target_kind)}` (horizon={format_text(target_horizon)})"
        ),
        "- Rebalance frequency: "
        f"`{format_text(spec.get('rebalance_frequency', metrics.get('rebalance_frequency')))}`",
    ]
    lines.extend(section_lines(CASE_SECTION_TITLES[3], data_setup_lines))

    lines.extend(
        section_lines(
            CASE_SECTION_TITLES[4],
            _methodology_lines(
                package_type=package_type,
                model_spec=model_spec,
                preprocess=preprocess,
                neutralization=neutralization,
            ),
        )
    )

    diagnostics_rows = [
        ("Factor Verdict", format_text(promotion_gate_metrics["factor_verdict"])),
        ("Verdict Reasons", _format_reason_list(profile_summary_metrics["factor_verdict_reasons"])),
        ("Level 2 Promotion", format_text(profile_summary_metrics["promotion_decision"])),
        ("Promotion Reasons", _format_reason_list(profile_summary_metrics["promotion_reasons"])),
        ("Promotion Blockers", _format_reason_list(profile_summary_metrics["promotion_blockers"])),
        (
            "Level 1->Level 2 Transition",
            format_text(profile_summary_metrics["level12_transition_label"]),
        ),
        (
            "Transition Interpretation",
            format_text(profile_summary_metrics["level12_transition_interpretation"]),
        ),
        (
            "Transition Reasons",
            _format_reason_list(profile_summary_metrics["level12_transition_reasons"]),
        ),
        (
            "Confirmation vs Degradation",
            format_text(profile_summary_metrics["level12_transition_confirmation_note"]),
        ),
        (
            "Level 2 Portfolio Validation",
            _portfolio_validation_note(profile_summary_metrics),
        ),
        (
            "Portfolio Robustness Taxonomy",
            format_text(portfolio_metrics["portfolio_validation_robustness_label"]),
        ),
        (
            "Portfolio Robustness Supports",
            _format_reason_list(portfolio_metrics["portfolio_validation_support_reasons"]),
        ),
        (
            "Portfolio Robustness Fragilities",
            _format_reason_list(portfolio_metrics["portfolio_validation_fragility_reasons"]),
        ),
        (
            "Portfolio Scenario Sensitivity",
            _format_reason_list(
                portfolio_metrics["portfolio_validation_scenario_sensitivity_notes"]
            ),
        ),
        (
            "Portfolio Benchmark Support Note",
            format_text(portfolio_metrics["portfolio_validation_benchmark_support_note"]),
        ),
        (
            "Portfolio Cost Sensitivity Note",
            format_text(portfolio_metrics["portfolio_validation_cost_sensitivity_note"]),
        ),
        (
            "Portfolio Concentration/Turnover Note",
            format_text(portfolio_metrics["portfolio_validation_concentration_turnover_note"]),
        ),
        (
            "Portfolio Validation Risks",
            _format_reason_list(profile_summary_metrics["portfolio_validation_major_risks"]),
        ),
        (
            "Portfolio Validation Baseline (Return / Turnover / Cost-Adj)",
            _metric_triplet(
                metrics.get("portfolio_validation_base_mean_portfolio_return"),
                metrics.get("portfolio_validation_base_mean_turnover"),
                metrics.get("portfolio_validation_base_cost_adjusted_return_review_rate"),
            ),
        ),
        (
            "Portfolio Validation Benchmark Relative",
            _portfolio_validation_benchmark_note(portfolio_metrics),
        ),
        (
            "Portfolio Validation Benchmark Risks",
            _format_reason_list(metrics.get("portfolio_validation_benchmark_relative_risks")),
        ),
        ("Evaluation Standard", _evaluation_standard_note(metrics)),
        ("Uncertainty Method", _uncertainty_method_note(uncertainty_metrics)),
        ("IC / ICIR", _metric_pair(core_metrics["mean_ic"], core_metrics["ic_ir"])),
        (
            "IC 95% CI",
            _metric_ci_pair(
                uncertainty_metrics["mean_ic_ci_lower"],
                uncertainty_metrics["mean_ic_ci_upper"],
            ),
        ),
        (
            "IC / RankIC Positive Rate",
            _metric_pair(
                core_metrics["ic_positive_rate"],
                core_metrics["rank_ic_positive_rate"],
            ),
        ),
        (
            "RankIC 95% CI",
            _metric_ci_pair(
                uncertainty_metrics["mean_rank_ic_ci_lower"],
                uncertainty_metrics["mean_rank_ic_ci_upper"],
            ),
        ),
        (
            "IC / RankIC Valid Ratio",
            _metric_pair(
                core_metrics["ic_valid_ratio"],
                core_metrics["rank_ic_valid_ratio"],
            ),
        ),
        (
            "Long-Short Performance",
            format_metric(core_metrics["mean_long_short_return"]),
        ),
        (
            "L/S IR / Return-per-Turnover",
            _metric_pair(
                core_metrics["long_short_ir"],
                core_metrics["long_short_return_per_turnover"],
            ),
        ),
        (
            "Long-Short Mean 95% CI",
            _metric_ci_pair(
                uncertainty_metrics["mean_long_short_return_ci_lower"],
                uncertainty_metrics["mean_long_short_return_ci_upper"],
            ),
        ),
        (
            "Subperiod Robustness (IC / L-S)",
            _metric_pair(
                core_metrics["subperiod_ic_positive_share"],
                core_metrics["subperiod_long_short_positive_share"],
            ),
        ),
        (
            "Rolling Stability",
            _rolling_window_text(rolling_metrics["rolling_window_size"]),
        ),
        (
            "Rolling IC positive share",
            format_metric(rolling_metrics["rolling_ic_positive_share"]),
        ),
        (
            "Rolling RankIC positive share",
            format_metric(rolling_metrics["rolling_rank_ic_positive_share"]),
        ),
        (
            "Rolling L/S positive share",
            format_metric(rolling_metrics["rolling_long_short_positive_share"]),
        ),
        (
            "Worst rolling IC window",
            format_metric(rolling_metrics["rolling_ic_min_mean"]),
        ),
        (
            "Worst rolling RankIC window",
            format_metric(rolling_metrics["rolling_rank_ic_min_mean"]),
        ),
        (
            "Worst rolling L/S window",
            format_metric(rolling_metrics["rolling_long_short_min_mean"]),
        ),
        (
            "Rolling Stability Flags",
            _format_flags(rolling_metrics["rolling_instability_flags"]),
        ),
        (
            "Rolling Stability Note",
            _rolling_stability_note(rolling_metrics),
        ),
        (
            "Group Return Summary",
            group_summary if group_summary is not None else "N/A",
        ),
        (
            "Turnover",
            format_metric(core_metrics["mean_long_short_turnover"])
            if core_metrics["mean_long_short_turnover"] is not None
            else format_metric(mean_turnover),
        ),
        (
            "Coverage",
            format_metric(core_metrics["coverage_mean"])
            if core_metrics["coverage_mean"] is not None
            else format_metric(mean_coverage),
        ),
        (
            "Coverage Ratio Mean/Min",
            _metric_pair(
                core_metrics["eval_coverage_ratio_mean"],
                core_metrics["eval_coverage_ratio_min"],
            ),
        ),
        (
            "Neutralization Corr Reduction",
            format_metric(neutralization_metrics["neutralization_mean_corr_reduction"]),
        ),
        *_neutralization_comparison_rows(neutralization_metrics),
        ("Uncertainty Flags", _format_flags(uncertainty_metrics["uncertainty_flags"])),
        ("Instability Flags", _format_flags(metrics.get("instability_flags"))),
    ]
    lines.extend(
        section_lines(
            CASE_SECTION_TITLES[5],
            markdown_table(("Metric", "Value"), diagnostics_rows),
        )
    )

    lines.extend(
        section_lines(CASE_SECTION_TITLES[6], [PLACEHOLDER_INTERPRETATION])
    )

    lines.extend(
        section_lines(
            CASE_SECTION_TITLES[7],
            _risks_and_limitations_lines(
                metrics=metrics,
                neutralization=neutralization,
                transaction_cost=as_object_dict(spec.get("transaction_cost")),
            ),
        )
    )

    lines.extend(section_lines(CASE_SECTION_TITLES[8], [PLACEHOLDER_NEXT_STEPS]))
    return "\n".join(lines).rstrip() + "\n"


def write_case_report(case_output_dir: str | Path, *, overwrite: bool = False) -> Path:
    """Render and write ``case_report.md`` in the provided case output directory."""

    case_dir = Path(case_output_dir).resolve()
    case_dir.mkdir(parents=True, exist_ok=True)
    report_path = case_dir / "case_report.md"
    if report_path.exists() and not overwrite:
        raise FileExistsError(
            f"{report_path} already exists. Pass overwrite=True to replace it."
        )
    report_path.write_text(render_case_report(case_dir), encoding="utf-8")
    return report_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="case-report-renderer",
        description="Render case_report.md from an existing case artifact directory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("case_output_dir", help="Case output directory containing manifests.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing case_report.md when present.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report_path = write_case_report(args.case_output_dir, overwrite=args.overwrite)
    except (FileNotFoundError, ValueError, KeyError, FileExistsError) as exc:
        parser.error(str(exc))

    print(f"Case report written: {report_path}")
    return 0


def _signal_definition_lines(
    *,
    package_type: str,
    spec: dict[str, object],
    metrics: dict[str, object],
    model_spec: dict[str, object],
    preprocess: dict[str, object],
) -> list[str]:
    lines: list[str] = []
    if package_type == "single_factor":
        lines.append(
            f"- Factor name: `{format_text(spec.get('factor_name', metrics.get('factor_name')))}`"
        )
        lines.append(
            f"- Direction: `{format_text(spec.get('direction', metrics.get('direction')))}`"
        )
        if model_spec:
            lines.append(
                f"- Model family: `{format_text(model_spec.get('family'))}`"
            )
            feature_columns = as_object_list(spec.get("feature_columns"))
            if feature_columns:
                lines.append(
                    f"- Feature count: `{len(feature_columns)}`"
                )
    else:
        components = as_object_list(spec.get("components"))
        if components:
            lines.append("- Components:")
            for component in components:
                row = as_object_dict(component)
                lines.append(
                    "- "
                    f"`{format_text(row.get('name'))}` "
                    f"(weight={format_metric(to_finite_float(row.get('weight')))}, "
                    f"direction={format_text(row.get('direction'))}, "
                    f"transform={format_text(row.get('transform'))})"
                )
        else:
            lines.append("- Components: N/A")

    lines.append(
        "- Transformation steps: "
        f"{_render_preprocess_description(preprocess, include_standardization=True)}"
    )
    return lines


def _methodology_lines(
    *,
    package_type: str,
    model_spec: dict[str, object],
    preprocess: dict[str, object],
    neutralization: dict[str, object],
) -> list[str]:
    lines = [
        (
            "- Preprocessing: "
            f"{_render_preprocess_description(preprocess, include_standardization=True)}"
        ),
        f"- Neutralization: {_render_neutralization_description(neutralization)}",
        (
            "- Timestamp alignment: factor values at `t` are evaluated against "
            "forward returns after `t`."
        ),
    ]

    if package_type == "composite":
        lines.append(
            "- Combination logic: weighted linear blend of transformed component factors."
        )
    elif model_spec:
        lines.append(
            "- Score generation: rolling/expanding model training first, then emit "
            "canonical factor scores."
        )
    else:
        lines.append("- Combination logic: single-factor signal (no component blending).")
    return lines


def _risks_and_limitations_lines(
    *,
    metrics: dict[str, object],
    neutralization: dict[str, object],
    transaction_cost: dict[str, object],
) -> list[str]:
    lines: list[str] = []

    coverage = to_finite_float(metrics.get("coverage_mean"))
    if coverage is not None and coverage < 0.6:
        lines.append(
            f"- Coverage is low ({format_metric(coverage)}), so signals may be thin on some dates."
        )

    n_dates = to_finite_float(metrics.get("n_dates_used"))
    if n_dates is not None and n_dates < 30:
        lines.append(
            f"- Evaluation sample is short ({int(n_dates)} dates), increasing estimate uncertainty."
        )

    missingness = to_finite_float(metrics.get("missingness_mean"))
    if missingness is not None and missingness > 0.4:
        lines.append(
            f"- Missingness is elevated ({format_metric(missingness)}), which can bias diagnostics."
        )

    if neutralization.get("enabled") is False:
        lines.append("- Neutralization is disabled; factor may retain style/sector exposures.")

    cost = to_finite_float(transaction_cost.get("one_way_rate"))
    if cost is not None and math.isclose(cost, 0.0, rel_tol=0.0, abs_tol=1e-12):
        lines.append("- Transaction cost is set to 0.0; live frictions are not reflected.")

    if not lines:
        lines.append(
            "- Auto-generated from recorded artifacts; manual review is still "
            "required for deployment."
        )

    for flag in parse_text_list(metrics.get("instability_flags")):
        mapped = _risk_line_for_flag(flag)
        if mapped is not None:
            lines.append(mapped)
    for flag in parse_text_list(metrics.get("uncertainty_flags")):
        mapped = _uncertainty_risk_line(flag)
        if mapped is not None:
            lines.append(mapped)
    return lines


def _metric_pair(left: object, right: object) -> str:
    return f"{format_metric(left)} / {format_metric(right)}"


def _metric_triplet(left: object, middle: object, right: object) -> str:
    return (
        f"{format_metric(left)} / {format_metric(middle)} / {format_metric(right)}"
    )


def _metric_ci_pair(lower: object, upper: object) -> str:
    left = format_metric(lower)
    right = format_metric(upper)
    if left == "N/A" or right == "N/A":
        return "N/A"
    return f"[{left}, {right}]"


def _rolling_window_text(value: object) -> str:
    window = to_finite_float(value)
    if window is None:
        return "N/A"
    if float(window).is_integer():
        return f"{int(window)} observations"
    return f"{format_metric(window)} observations"


def _rolling_stability_note(rolling_metrics: RollingStabilityMetrics) -> str:
    rolling_flags = list(rolling_metrics["rolling_instability_flags"])
    if "rolling_regime_dependence" in rolling_flags:
        return "Rolling evidence suggests regime dependence."

    ic_share = rolling_metrics["rolling_ic_positive_share"]
    rank_share = rolling_metrics["rolling_rank_ic_positive_share"]
    ls_share = rolling_metrics["rolling_long_short_positive_share"]
    available = [value for value in (ic_share, rank_share, ls_share) if value is not None]
    if available and min(available) >= 0.6:
        return "Rolling evidence is persistent across windows."
    return "N/A"


def _evaluation_standard_note(metrics: dict[str, object]) -> str:
    profile = format_text(metrics.get("research_evaluation_profile"))
    if profile == "N/A":
        return "N/A"
    snapshot = as_object_dict(metrics.get("research_evaluation_snapshot"))
    if not snapshot:
        return profile
    uncertainty = as_object_dict(snapshot.get("uncertainty"))
    rolling = as_object_dict(snapshot.get("rolling_stability"))
    uncertainty_method = format_text(uncertainty.get("method"), na="")
    ci_level = uncertainty.get("confidence_level")
    window_size = rolling.get("rolling_window_size")
    if uncertainty_method == "" and ci_level is None and window_size is None:
        return profile
    uncertainty_note = format_metric(ci_level)
    if uncertainty_method:
        uncertainty_note = f"{uncertainty_method}@{uncertainty_note}"
    return (
        f"{profile} "
        f"(uncertainty={uncertainty_note}, rolling_window={format_text(window_size)})"
    )


def _uncertainty_method_note(uncertainty_metrics: UncertaintyEvidenceMetrics) -> str:
    method = uncertainty_metrics["uncertainty_method"]
    if method is None:
        return "N/A"
    ci_level = uncertainty_metrics["uncertainty_confidence_level"]
    resamples = uncertainty_metrics["uncertainty_bootstrap_resamples"]
    block_length = uncertainty_metrics["uncertainty_bootstrap_block_length"]
    if method.lower() == "bootstrap":
        if ci_level is None and resamples is None:
            return "bootstrap"
        return (
            "bootstrap "
            f"(CI={format_metric(ci_level)}, "
            f"resamples={format_metric(resamples)})"
        )
    if method.lower() == "block_bootstrap":
        if ci_level is None and resamples is None and block_length is None:
            return "block_bootstrap"
        return (
            "block_bootstrap "
            f"(CI={format_metric(ci_level)}, "
            f"resamples={format_metric(resamples)}, "
            f"block_length={format_metric(block_length)})"
        )
    if ci_level is None:
        return method
    return f"{method} (CI={format_metric(ci_level)})"


def _neutralization_comparison_rows(
    neutralization_metrics: NeutralizationComparisonMetrics,
) -> list[tuple[str, str]]:
    comparison = neutralization_metrics["neutralization_comparison"]
    raw = comparison["raw"]
    neutralized = comparison["neutralized"]
    delta = comparison["delta"]
    if (
        not raw
        and not neutralized
        and not delta
        and not comparison["interpretation_flags"]
        and not comparison["interpretation_reasons"]
    ):
        return []
    return [
        (
            "Raw vs Neutralized Mean IC",
            _metric_transition(
                raw.get("mean_ic"),
                neutralized.get("mean_ic"),
                delta.get("mean_ic_delta"),
            ),
        ),
        (
            "Raw vs Neutralized Mean RankIC",
            _metric_transition(
                raw.get("mean_rank_ic"),
                neutralized.get("mean_rank_ic"),
                delta.get("mean_rank_ic_delta"),
            ),
        ),
        (
            "Raw vs Neutralized Mean L/S Return",
            _metric_transition(
                raw.get("mean_long_short_return"),
                neutralized.get("mean_long_short_return"),
                delta.get("mean_long_short_return_delta"),
            ),
        ),
        (
            "Raw vs Neutralized ICIR",
            _metric_transition(
                raw.get("ic_ir"),
                neutralized.get("ic_ir"),
                delta.get("ic_ir_delta"),
            ),
        ),
        (
            "Raw vs Neutralized Validity Min",
            _metric_transition(
                raw.get("valid_ratio_min"),
                neutralized.get("valid_ratio_min"),
                delta.get("valid_ratio_min_delta"),
            ),
        ),
        (
            "Raw vs Neutralized Coverage Mean",
            _metric_transition(
                raw.get("eval_coverage_ratio_mean"),
                neutralized.get("eval_coverage_ratio_mean"),
                delta.get("eval_coverage_ratio_mean_delta"),
            ),
        ),
        (
            "Raw vs Neutralized Uncertainty Overlap Count",
            _metric_transition(
                raw.get("uncertainty_overlap_zero_count"),
                neutralized.get("uncertainty_overlap_zero_count"),
                delta.get("uncertainty_overlap_zero_count_delta"),
            ),
        ),
        (
            "Raw vs Neutralized Rolling+ Min Share",
            _metric_transition(
                raw.get("rolling_positive_share_min"),
                neutralized.get("rolling_positive_share_min"),
                delta.get("rolling_positive_share_min_delta"),
            ),
        ),
        (
            "Raw vs Neutralized Rolling Worst Mean",
            _metric_transition(
                raw.get("rolling_worst_mean_min"),
                neutralized.get("rolling_worst_mean_min"),
                delta.get("rolling_worst_mean_min_delta"),
            ),
        ),
        (
            "Neutralization Comparison Flags",
            _format_flags(comparison["interpretation_flags"]),
        ),
        (
            "Neutralization Comparison Reasons",
            _format_reason_list(comparison["interpretation_reasons"]),
        ),
    ]


def _metric_transition(raw_value: object, neutralized_value: object, delta_value: object) -> str:
    return (
        f"{format_metric(raw_value)} -> {format_metric(neutralized_value)} "
        f"(delta={format_metric(delta_value)})"
    )


def _extract_objective(summary_text: str | None) -> str | None:
    if summary_text is None:
        return None
    lines = [line.strip() for line in summary_text.splitlines()]
    if not lines:
        return None

    for line in lines:
        lowered = line.lower()
        if lowered.startswith("objective:"):
            return line.split(":", 1)[1].strip() or None

    for idx, line in enumerate(lines):
        if line.lower().startswith("## objective"):
            for candidate in lines[idx + 1 :]:
                if candidate and not candidate.startswith("#"):
                    return candidate
            break

    return None


def _group_return_summary(path: Path) -> str | None:
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    if frame.empty or "group" not in frame.columns or "group_return" not in frame.columns:
        return None
    grouped = frame.groupby("group", dropna=True)["group_return"].mean()
    if grouped.empty:
        return None
    spread = float(grouped.max() - grouped.min())
    return (
        f"mean top-bottom spread={format_metric(spread)} "
        f"(groups={int(grouped.index.nunique())})"
    )


def _mean_csv_metric(path: Path, column: str) -> float | None:
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    if frame.empty or column not in frame.columns:
        return None
    value = float(frame[column].mean())
    if not math.isfinite(value):
        return None
    return value


def _render_preprocess_description(
    preprocess: dict[str, object],
    *,
    include_standardization: bool,
) -> str:
    if not preprocess:
        return "N/A"
    winsorize = bool(preprocess.get("winsorize", False))
    lower = format_metric(to_finite_float(preprocess.get("winsorize_lower")))
    upper = format_metric(to_finite_float(preprocess.get("winsorize_upper")))
    parts = [f"winsorize={winsorize} ({lower}, {upper})"]
    if include_standardization:
        parts.append(f"standardization={format_text(preprocess.get('standardization'))}")
    min_group = preprocess.get("min_group_size")
    if min_group is not None:
        parts.append(f"min_group_size={format_text(min_group)}")
    min_coverage = preprocess.get("min_coverage")
    if min_coverage is not None:
        parts.append(f"min_coverage={format_metric(to_finite_float(min_coverage))}")
    return ", ".join(parts)


def _render_neutralization_description(neutralization: dict[str, object]) -> str:
    if not neutralization:
        return "N/A"
    enabled = bool(neutralization.get("enabled", False))
    if not enabled:
        return "disabled"

    fields: list[str] = ["enabled"]
    for name in ("size_col", "industry_col", "beta_col"):
        value = neutralization.get(name)
        if value is not None:
            fields.append(f"{name}={value}")
    min_obs = neutralization.get("min_obs")
    if min_obs is not None:
        fields.append(f"min_obs={min_obs}")
    ridge = neutralization.get("ridge")
    if ridge is not None:
        fields.append(f"ridge={ridge}")
    return ", ".join(fields)


def _extract_metrics(payload: dict[str, object]) -> dict[str, object]:
    raw = payload.get("metrics")
    if isinstance(raw, dict):
        return cast(dict[str, object], raw)
    return {}


def _resolve_package_type(artifact_type: object) -> str:
    if not isinstance(artifact_type, str):
        return "N/A"
    text = artifact_type.lower()
    if "single_factor" in text:
        return "single_factor"
    if "composite" in text:
        return "composite"
    return "N/A"


def _read_text_if_exists(path: Path) -> str | None:
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def _load_required_json(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"required artifact not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return cast(dict[str, object], payload)


def _format_flags(value: object) -> str:
    return format_text_list(value, empty="none")


def _format_reason_list(value: object) -> str:
    return format_text_list(value, empty="none", separator="; ")


def _portfolio_validation_note(
    metrics: CampaignProfileSummaryMetrics | PortfolioValidationMetrics,
) -> str:
    return portfolio_validation_note(
        metrics["portfolio_validation_status"],
        metrics["portfolio_validation_recommendation"],
    )


def _portfolio_validation_benchmark_note(
    metrics: PortfolioValidationMetrics,
) -> str:
    return portfolio_validation_benchmark_note(
        metrics["portfolio_validation_benchmark_relative_status"],
        metrics["portfolio_validation_benchmark_relative_assessment"],
        metrics["portfolio_validation_benchmark_excess_return"],
        metrics["portfolio_validation_benchmark_tracking_error"],
        format_metric=format_metric,
    )


def _risk_line_for_flag(flag: str) -> str | None:
    mapping = {
        "short_eval_window": "- Instability flag: short_eval_window (limited sample length).",
        "low_ic_valid_ratio": "- Instability flag: low_ic_valid_ratio (many IC points undefined).",
        "low_rank_ic_valid_ratio": (
            "- Instability flag: low_rank_ic_valid_ratio (many RankIC points undefined)."
        ),
        "ic_sign_instability": (
            "- Instability flag: ic_sign_instability (IC sign flips frequently)."
        ),
        "ic_subperiod_instability": (
            "- Instability flag: ic_subperiod_instability (IC weak in chronological subperiods)."
        ),
        "long_short_subperiod_instability": (
            "- Instability flag: long_short_subperiod_instability "
            "(spread unstable across subperiods)."
        ),
        "thin_universe_coverage": (
            "- Instability flag: thin_universe_coverage (effective universe is sparse)."
        ),
        "high_turnover_negative_spread": (
            "- Instability flag: high_turnover_negative_spread "
            "(trading churn without positive spread)."
        ),
        "negative_long_short_ir": (
            "- Instability flag: negative_long_short_ir "
            "(spread risk-adjusted return < 0)."
        ),
        "rolling_regime_dependence": (
            "- Instability flag: rolling_regime_dependence "
            "(rolling evidence is regime-dependent)."
        ),
        "rolling_ic_below_zero_share_high": (
            "- Instability flag: rolling_ic_below_zero_share_high "
            "(rolling IC is non-positive too often)."
        ),
        "rolling_rank_ic_below_zero_share_high": (
            "- Instability flag: rolling_rank_ic_below_zero_share_high "
            "(rolling RankIC is non-positive too often)."
        ),
        "rolling_long_short_below_zero_share_high": (
            "- Instability flag: rolling_long_short_below_zero_share_high "
            "(rolling long-short mean is non-positive too often)."
        ),
        "rolling_ic_sign_flip_instability": (
            "- Instability flag: rolling_ic_sign_flip_instability "
            "(rolling IC changes sign frequently)."
        ),
        "rolling_rank_ic_sign_flip_instability": (
            "- Instability flag: rolling_rank_ic_sign_flip_instability "
            "(rolling RankIC changes sign frequently)."
        ),
        "rolling_long_short_sign_flip_instability": (
            "- Instability flag: rolling_long_short_sign_flip_instability "
            "(rolling long-short mean changes sign frequently)."
        ),
    }
    return mapping.get(flag)


def _uncertainty_risk_line(flag: str) -> str | None:
    mapping = {
        "ic_ci_overlaps_zero": (
            "- Uncertainty flag: ic_ci_overlaps_zero (mean IC confidence interval crosses zero)."
        ),
        "rank_ic_ci_overlaps_zero": (
            "- Uncertainty flag: rank_ic_ci_overlaps_zero "
            "(mean RankIC confidence interval crosses zero)."
        ),
        "long_short_ci_overlaps_zero": (
            "- Uncertainty flag: long_short_ci_overlaps_zero "
            "(mean long-short confidence interval crosses zero)."
        ),
        "ic_ci_wide": "- Uncertainty flag: ic_ci_wide (IC estimate is noisy relative to edge).",
        "rank_ic_ci_wide": (
            "- Uncertainty flag: rank_ic_ci_wide (RankIC estimate is noisy relative to edge)."
        ),
        "long_short_ci_wide": (
            "- Uncertainty flag: long_short_ci_wide "
            "(long-short estimate is noisy relative to edge)."
        ),
        "ic_ci_unavailable": "- Uncertainty flag: ic_ci_unavailable (insufficient IC samples).",
        "rank_ic_ci_unavailable": (
            "- Uncertainty flag: rank_ic_ci_unavailable (insufficient RankIC samples)."
        ),
        "long_short_ci_unavailable": (
            "- Uncertainty flag: long_short_ci_unavailable "
            "(insufficient long-short samples)."
        ),
    }
    return mapping.get(flag)


if __name__ == "__main__":
    raise SystemExit(main())
