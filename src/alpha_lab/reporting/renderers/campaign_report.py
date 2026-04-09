from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

from alpha_lab.key_metrics_contracts import (
    CampaignProfileSummaryMetrics,
    Level12TransitionDistributionMetrics,
    NeutralizationComparisonMetrics,
    PortfolioValidationMetrics,
    PromotionGateMetrics,
    RollingStabilityMetrics,
    project_campaign_profile_summary_metrics,
    project_level12_transition_distribution,
    project_portfolio_validation_metrics,
    project_promotion_gate_metrics,
)
from alpha_lab.reporting.campaign_triage import (
    CAMPAIGN_RANK_RULE,
    build_campaign_triage,
    campaign_rank_sort_key,
)
from alpha_lab.reporting.display_helpers import (
    as_object_dict,
    as_object_list,
    format_ci,
    format_text_list,
    portfolio_validation_benchmark_note,
    portfolio_validation_note,
    to_finite_float,
)
from alpha_lab.reporting.level2_promotion import build_level2_promotion
from alpha_lab.research_evaluation_config import (
    DEFAULT_RESEARCH_EVALUATION_CONFIG,
    ResearchEvaluationConfig,
    get_research_evaluation_config,
)

from .templates import (
    CAMPAIGN_SECTION_TITLES,
    COMPARISON_TABLE_COLUMNS,
    PLACEHOLDER_NEXT_STEPS,
    PLACEHOLDER_OBJECTIVE,
    format_metric,
    format_text,
    markdown_table,
    section_lines,
)


def render_campaign_report(campaign_output_dir: str | Path) -> str:
    """Render a standardized campaign-level markdown report."""

    campaign_dir = Path(campaign_output_dir).resolve()
    manifest = _load_optional_json(campaign_dir / "campaign_manifest.json")
    results = _load_required_json(campaign_dir / "campaign_results.json")

    campaign_name = format_text(
        (manifest or {}).get("campaign_name", results.get("campaign_name")),
        na=campaign_dir.name,
    )
    objective = format_text((manifest or {}).get("campaign_description"), na=PLACEHOLDER_OBJECTIVE)

    cases = as_object_list(results.get("cases"))
    rows = [_case_row(raw, campaign_dir=campaign_dir) for raw in cases]
    ranked_rows = _rank_rows(rows)
    transition_distribution = project_level12_transition_distribution(
        _transition_distribution_rows(rows)
    )

    lines: list[str] = [f"# Campaign Report: {campaign_name}", ""]

    lines.extend(
        section_lines(
            CAMPAIGN_SECTION_TITLES[0],
            [
                f"- Campaign name: `{campaign_name}`",
                (
                    "- Evaluation standard profile: "
                    f"`{_campaign_profile_note(rows)}`"
                ),
            ],
        )
    )
    lines.extend(section_lines(CAMPAIGN_SECTION_TITLES[1], [objective]))

    included = [
        f"- `{row['case_name']}` ({row['package_type']}) - status: `{row['status']}`"
        for row in rows
    ]
    lines.extend(
        section_lines(
            CAMPAIGN_SECTION_TITLES[2],
            included if included else ["- N/A"],
        )
    )

    table_rows: list[tuple[object, ...]] = []
    for row in ranked_rows:
        promotion_gate_metrics = _as_promotion_gate_metrics(row.get("promotion_gate_metrics"))
        profile_summary_metrics = _as_campaign_profile_summary_metrics(
            row.get("profile_summary_metrics")
        )
        portfolio_metrics = _as_portfolio_validation_metrics(
            row.get("portfolio_validation_metrics")
        )
        core_metrics = promotion_gate_metrics["core"]
        uncertainty_metrics = promotion_gate_metrics["uncertainty"]
        neutralization_metrics = promotion_gate_metrics["neutralization"]
        triage = as_object_dict(row.get("triage"))
        promotion = as_object_dict(row.get("promotion"))
        rank = row.get("rank")
        rank_text = str(rank) if isinstance(rank, int) else "N/A"
        table_rows.append(
            (
                rank_text,
                row["case_name"],
                _display_case_type(row["package_type"]),
                _metric_pair(core_metrics["mean_ic"], core_metrics["ic_ir"]),
                format_ci(
                    uncertainty_metrics["mean_ic_ci_lower"],
                    uncertainty_metrics["mean_ic_ci_upper"],
                ),
                format_metric(core_metrics["mean_long_short_return"]),
                format_metric(core_metrics["mean_long_short_turnover"]),
                format_metric(core_metrics["coverage_mean"]),
                _format_flag_list(uncertainty_metrics["uncertainty_flags"]),
                _neutralization_comparison_note(neutralization_metrics),
                format_text(profile_summary_metrics["factor_verdict"]),
                format_text(triage.get("campaign_triage")),
                _format_reason_list(triage.get("campaign_triage_reasons")),
                format_text(promotion.get("promotion_decision")),
                _format_reason_list(promotion.get("promotion_reasons")),
                _format_reason_list(promotion.get("promotion_blockers")),
                format_text(profile_summary_metrics["level12_transition_label"]),
                _portfolio_validation_note(profile_summary_metrics),
                _portfolio_validation_robustness_note(portfolio_metrics),
                _portfolio_validation_benchmark_note(portfolio_metrics),
                _portfolio_validation_risks(profile_summary_metrics),
                row["status"],
            )
        )
    lines.extend(
        section_lines(
            CAMPAIGN_SECTION_TITLES[3],
            markdown_table(COMPARISON_TABLE_COLUMNS, table_rows),
        )
    )

    highlight_lines = [_highlight_line(row) for row in ranked_rows]
    lines.extend(section_lines(CAMPAIGN_SECTION_TITLES[4], highlight_lines or ["- N/A"]))

    lines.extend(section_lines(CAMPAIGN_SECTION_TITLES[5], _cross_case_insights(rows)))

    failed = [row for row in rows if row["status"] != "success"]
    if failed:
        failure_lines = [
            f"- `{row['case_name']}` ({row['status']}): {format_text(row['error'])}"
            for row in failed
        ]
    else:
        failure_lines = ["- No failed cases recorded in campaign_results.json."]
    lines.extend(section_lines(CAMPAIGN_SECTION_TITLES[6], failure_lines))

    n_success = sum(1 for row in rows if row["status"] == "success")
    conclusions = [
        f"- Total cases: {len(rows)}",
        f"- Successful cases: {n_success}",
        f"- Failed/skipped cases: {len(rows) - n_success}",
    ]
    best_case = _best_case(rows, metric="ic_ir")
    if best_case is not None:
        conclusions.append(
            "- Best ICIR among successful cases: "
            f"`{best_case}`"
        )
    top_ranked = next(
        (row for row in ranked_rows if format_text(row.get("status")) == "success"),
        None,
    )
    if top_ranked is not None:
        triage = as_object_dict(top_ranked.get("triage"))
        conclusions.append(
            "- Top campaign triage candidate: "
            f"`{format_text(top_ranked.get('case_name'))}` "
            f"({format_text(triage.get('campaign_triage'))})"
        )
    top_promoted = next(
        (
            row
            for row in ranked_rows
            if format_text(row.get("status")) == "success"
            and format_text(as_object_dict(row.get("promotion")).get("promotion_decision"))
            == "Promote to Level 2"
        ),
        None,
    )
    if top_promoted is not None:
        conclusions.append(
            "- Top Level 2 promotion candidate: "
            f"`{format_text(top_promoted.get('case_name'))}` "
            "(Promote to Level 2)"
        )
    else:
        conclusions.append("- Level 2 promotion gate: no case passed this run.")
    top_portfolio_validated = next(
        (
            row
            for row in ranked_rows
            if format_text(row.get("status")) == "success"
            and format_text(
                _as_campaign_profile_summary_metrics(
                    row.get("profile_summary_metrics")
                )["portfolio_validation_recommendation"]
            )
            == "Credible at portfolio level"
        ),
        None,
    )
    if top_portfolio_validated is not None:
        conclusions.append(
            "- Top portfolio-credible Level 2 candidate: "
            f"`{format_text(top_portfolio_validated.get('case_name'))}`"
        )
    else:
        conclusions.append("- Level 2 portfolio validation: no case is yet portfolio-credible.")
    conclusions.extend(_transition_distribution_lines(transition_distribution))
    conclusions.append(f"- Campaign rank rule: `{CAMPAIGN_RANK_RULE}`")
    lines.extend(section_lines(CAMPAIGN_SECTION_TITLES[7], conclusions))

    lines.extend(
        section_lines(
            CAMPAIGN_SECTION_TITLES[8],
            [PLACEHOLDER_NEXT_STEPS],
        )
    )

    return "\n".join(lines).rstrip() + "\n"


def write_campaign_report(
    campaign_output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Render and write ``campaign_report.md`` in the campaign output directory."""

    campaign_dir = Path(campaign_output_dir).resolve()
    campaign_dir.mkdir(parents=True, exist_ok=True)
    report_path = campaign_dir / "campaign_report.md"
    if report_path.exists() and not overwrite:
        raise FileExistsError(
            f"{report_path} already exists. Pass overwrite=True to replace it."
        )
    report_path.write_text(render_campaign_report(campaign_dir), encoding="utf-8")
    return report_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="campaign-report-renderer",
        description="Render campaign_report.md from campaign artifacts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("campaign_output_dir", help="Campaign output directory.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing campaign_report.md when present.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report_path = write_campaign_report(args.campaign_output_dir, overwrite=args.overwrite)
    except (FileNotFoundError, ValueError, KeyError, FileExistsError) as exc:
        parser.error(str(exc))

    print(f"Campaign report written: {report_path}")
    return 0


def _case_row(raw: object, *, campaign_dir: Path) -> dict[str, object]:
    payload = as_object_dict(raw)
    case_name = format_text(payload.get("case_name"))
    package_type = format_text(payload.get("package_type"))
    status = format_text(payload.get("status"))
    error = payload.get("error")

    metrics = as_object_dict(payload.get("key_metrics")).copy()
    if not metrics:
        loaded = _load_case_metrics(payload, campaign_dir=campaign_dir)
        metrics.update(loaded)
    promotion_gate_metrics = project_promotion_gate_metrics(metrics)
    profile_summary_metrics = project_campaign_profile_summary_metrics(metrics)
    portfolio_validation_metrics = project_portfolio_validation_metrics(metrics)

    return {
        "case_name": case_name,
        "package_type": package_type,
        "status": status,
        "metrics": metrics,
        "promotion_gate_metrics": promotion_gate_metrics,
        "profile_summary_metrics": profile_summary_metrics,
        "portfolio_validation_metrics": portfolio_validation_metrics,
        "error": error,
    }


def _load_case_metrics(payload: dict[str, object], *, campaign_dir: Path) -> dict[str, object]:
    metrics_path = payload.get("metrics_path")
    output_dir = payload.get("output_dir")

    candidates: list[Path] = []
    if isinstance(metrics_path, str) and metrics_path.strip():
        candidates.append(_resolve_path(metrics_path, base=campaign_dir))
    if isinstance(output_dir, str) and output_dir.strip():
        candidates.append(_resolve_path(output_dir, base=campaign_dir) / "metrics.json")

    for path in candidates:
        loaded = _load_optional_json(path)
        if loaded is None:
            continue
        metrics = as_object_dict(loaded.get("metrics"))
        if metrics:
            return metrics
    return {}


def _cross_case_insights(rows: list[dict[str, object]]) -> list[str]:
    lines: list[str] = []

    value_case = _best_case(rows, metric="ic_ir", name_tokens=("value", "bp"))
    quality_case = _best_case(rows, metric="ic_ir", name_tokens=("quality", "roe"))
    composite_case = _best_case(rows, metric="ic_ir", package_type="composite")

    lines.append(
        "- Value proxy (name contains value/bp): "
        f"{_insight_entry(value_case, rows, metric='ic_ir')}"
    )
    lines.append(
        "- Quality proxy (name contains quality/roe): "
        f"{_insight_entry(quality_case, rows, metric='ic_ir')}"
    )
    lines.append(
        "- Composite cases: "
        f"{_insight_entry(composite_case, rows, metric='ic_ir')}"
    )

    overall_best = _best_case(rows, metric="mean_long_short_return")
    if overall_best is not None:
        value = _case_metric(rows, overall_best, "mean_long_short_return")
        lines.append(
            "- Best long-short performance among successful cases: "
            f"`{overall_best}` ({format_metric(value)})"
        )
    else:
        lines.append("- Long-short comparison: N/A (no successful cases with finite returns).")

    return lines


def _highlight_line(row: dict[str, object]) -> str:
    case_name = format_text(row.get("case_name"))
    status = format_text(row.get("status"))
    promotion_gate_metrics = _as_promotion_gate_metrics(row.get("promotion_gate_metrics"))
    profile_summary_metrics = _as_campaign_profile_summary_metrics(
        row.get("profile_summary_metrics")
    )
    portfolio_metrics = _as_portfolio_validation_metrics(
        row.get("portfolio_validation_metrics")
    )
    core_metrics = promotion_gate_metrics["core"]
    uncertainty_metrics = promotion_gate_metrics["uncertainty"]
    rolling_metrics = promotion_gate_metrics["rolling"]
    neutralization_metrics = promotion_gate_metrics["neutralization"]
    triage = as_object_dict(row.get("triage"))
    promotion = as_object_dict(row.get("promotion"))
    triage_label = format_text(triage.get("campaign_triage"))
    triage_reasons = _format_reason_list(triage.get("campaign_triage_reasons"))
    promotion_label = format_text(promotion.get("promotion_decision"))
    promotion_reasons = _format_reason_list(promotion.get("promotion_reasons"))
    promotion_blockers = _format_reason_list(promotion.get("promotion_blockers"))
    transition_label = format_text(profile_summary_metrics["level12_transition_label"])
    transition_reasons = _format_reason_list(profile_summary_metrics["level12_transition_reasons"])
    portfolio_validation = _portfolio_validation_note(profile_summary_metrics)
    portfolio_robustness = _portfolio_validation_robustness_note(portfolio_metrics)
    portfolio_benchmark = _portfolio_validation_benchmark_note(portfolio_metrics)
    portfolio_risks = _portfolio_validation_risks(profile_summary_metrics)
    rank = row.get("rank")
    rank_text = str(rank) if isinstance(rank, int) else "N/A"

    if status != "success":
        return (
            f"- `#{rank_text}` `{case_name}` did not complete successfully ({status}); "
            f"triage={triage_label}, reasons={triage_reasons}, "
            f"promotion={promotion_label}, blockers={promotion_blockers}."
            f" transition={transition_label}, transition_reasons={transition_reasons}."
            f" portfolio_validation={portfolio_validation}, "
            f"portfolio_robustness={portfolio_robustness}, "
            f"portfolio_benchmark={portfolio_benchmark}, "
            f"portfolio_risks={portfolio_risks}."
        )

    subperiod_robustness = _metric_pair(
        core_metrics["subperiod_ic_positive_share"],
        core_metrics["subperiod_long_short_positive_share"],
    )
    verdict = format_text(profile_summary_metrics["factor_verdict"])
    verdict_reasons = _format_reason_list(profile_summary_metrics["factor_verdict_reasons"])
    uncertainty = _format_flag_list(uncertainty_metrics["uncertainty_flags"])
    rolling_note = _rolling_stability_note(rolling_metrics)
    neutralization_note = _neutralization_comparison_note(neutralization_metrics)
    ic_ci = format_ci(
        uncertainty_metrics["mean_ic_ci_lower"],
        uncertainty_metrics["mean_ic_ci_upper"],
    )
    return (
        f"- `#{rank_text}` `{case_name}`: "
        f"IC/ICIR={_metric_pair(core_metrics['mean_ic'], core_metrics['ic_ir'])}, "
        f"IC95%CI={ic_ci}, "
        f"L/S={format_metric(core_metrics['mean_long_short_return'])}, "
        f"turnover={format_metric(core_metrics['mean_long_short_turnover'])}, "
        f"coverage={format_metric(core_metrics['coverage_mean'])}, "
        f"uncertainty={uncertainty}, "
        f"subperiod robustness={subperiod_robustness}, "
        f"rolling IC+ share={format_metric(rolling_metrics['rolling_ic_positive_share'])}, "
        f"worst rolling IC={format_metric(rolling_metrics['rolling_ic_min_mean'])}, "
        f"rolling note={rolling_note}, "
        f"neutralization={neutralization_note}, "
        f"verdict={verdict}, "
        f"reasons={verdict_reasons}, "
        f"triage={triage_label}, "
        f"triage_reasons={triage_reasons}, "
        f"promotion={promotion_label}, "
        f"promotion_reasons={promotion_reasons}, "
        f"promotion_blockers={promotion_blockers}, "
        f"transition={transition_label}, "
        f"transition_reasons={transition_reasons}, "
        f"portfolio_validation={portfolio_validation}, "
        f"portfolio_robustness={portfolio_robustness}, "
        f"portfolio_benchmark={portfolio_benchmark}, "
        f"portfolio_risks={portfolio_risks}."
    )


def _rank_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    ranked: list[dict[str, object]] = []
    for row in rows:
        metrics = as_object_dict(row.get("metrics"))
        status = format_text(row.get("status"))
        evaluation_config = _evaluation_config_for_metrics(metrics)
        triage = build_campaign_triage(
            metrics,
            status=status,
            thresholds=evaluation_config.campaign_triage,
        ).to_dict()
        promotion = build_level2_promotion(
            metrics,
            status=status,
            thresholds=evaluation_config.level2_promotion,
        ).to_dict()
        ranked.append({**row, "triage": triage, "promotion": promotion, "rank": None})

    ranked.sort(
        key=lambda row: campaign_rank_sort_key(
            case_name=format_text(row.get("case_name")),
            status=format_text(row.get("status")),
            metrics=as_object_dict(row.get("metrics")),
            thresholds=_evaluation_config_for_metrics(
                as_object_dict(row.get("metrics"))
            ).campaign_triage,
        )
    )

    next_rank = 1
    for row in ranked:
        if format_text(row.get("status")) == "success":
            row["rank"] = next_rank
            next_rank += 1
    return ranked


def _best_case(
    rows: list[dict[str, object]],
    *,
    metric: str,
    name_tokens: tuple[str, ...] | None = None,
    package_type: str | None = None,
) -> str | None:
    best_case: str | None = None
    best_value: float | None = None
    for row in rows:
        if format_text(row.get("status")) != "success":
            continue
        case_name = format_text(row.get("case_name")).lower()
        row_package_type = format_text(row.get("package_type")).lower()

        if name_tokens is not None and not any(token in case_name for token in name_tokens):
            continue
        if package_type is not None and package_type.lower() != row_package_type:
            continue

        value = to_finite_float(as_object_dict(row.get("metrics")).get(metric))
        if value is None:
            continue
        if best_value is None or value > best_value:
            best_value = value
            best_case = format_text(row.get("case_name"))
    return best_case


def _case_metric(rows: list[dict[str, object]], case_name: str, metric: str) -> float | None:
    for row in rows:
        if format_text(row.get("case_name")) == case_name:
            return to_finite_float(as_object_dict(row.get("metrics")).get(metric))
    return None


def _insight_entry(case_name: str | None, rows: list[dict[str, object]], *, metric: str) -> str:
    if case_name is None:
        return "N/A"
    value = _case_metric(rows, case_name, metric)
    return f"`{case_name}` ({format_metric(value)})"


def _campaign_profile_note(rows: list[dict[str, object]]) -> str:
    profiles = sorted(
        {
            format_text(as_object_dict(row.get("metrics")).get("research_evaluation_profile"))
            for row in rows
            if format_text(as_object_dict(row.get("metrics")).get("research_evaluation_profile"))
            != "N/A"
        }
    )
    if not profiles:
        return DEFAULT_RESEARCH_EVALUATION_CONFIG.profile_name
    if len(profiles) == 1:
        return profiles[0]
    return "mixed: " + ", ".join(profiles)


def _evaluation_config_for_metrics(metrics: dict[str, object]) -> ResearchEvaluationConfig:
    profile_name = format_text(metrics.get("research_evaluation_profile"), na="").strip()
    if profile_name:
        try:
            return get_research_evaluation_config(profile_name)
        except ValueError:
            return DEFAULT_RESEARCH_EVALUATION_CONFIG
    return DEFAULT_RESEARCH_EVALUATION_CONFIG


def _display_case_type(value: object) -> str:
    text = format_text(value).lower()
    if text == "single_factor":
        return "single"
    if text == "composite":
        return "composite"
    return "N/A"


def _metric_pair(left: object, right: object) -> str:
    return f"{format_metric(left)} / {format_metric(right)}"


def _transition_distribution_rows(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for row in rows:
        metrics = as_object_dict(row.get("metrics"))
        profile_summary = _as_campaign_profile_summary_metrics(row.get("profile_summary_metrics"))
        transition_label = profile_summary["level12_transition_label"] or metrics.get(
            "level12_transition_label"
        )
        transition_reasons = profile_summary["level12_transition_reasons"] or metrics.get(
            "level12_transition_reasons"
        )
        out.append(
            {
                "case_name": row.get("case_name"),
                "level12_transition_label": transition_label,
                "level12_transition_reasons": transition_reasons,
            }
        )
    return out


def _transition_distribution_lines(
    distribution: Level12TransitionDistributionMetrics,
) -> list[str]:
    counts = distribution["counts_by_transition_label"]
    proportions = distribution["proportions_by_transition_label"]
    reason_rollups_obj = distribution.get("reason_rollup_by_transition_label", {})
    reason_rollups = reason_rollups_obj if isinstance(reason_rollups_obj, dict) else {}
    lines = [
        (
            "- L1->L2 transitions (total/observed/missing): "
            f"{distribution['n_cases']}/"
            f"{distribution['n_cases_with_transition_label']}/"
            f"{distribution['n_cases_missing_transition_label']}"
        ),
    ]
    for label in (
        "Confirmed at portfolio level",
        "Weakened at portfolio level",
        "Fragile after promotion",
        "Improved at portfolio level",
        "Inconclusive transition",
    ):
        lines.append(
            f"- {label}: {counts.get(label, 0)} ({proportions.get(label, 0.0):.1%})"
        )
    lines.append("- Dominant transition reasons by label:")
    for label in (
        "Confirmed at portfolio level",
        "Weakened at portfolio level",
        "Fragile after promotion",
        "Improved at portfolio level",
        "Inconclusive transition",
    ):
        rollup_obj = reason_rollups.get(label)
        rollup = rollup_obj if isinstance(rollup_obj, dict) else {}
        top_reasons_obj = rollup.get("top_reasons")
        top_reasons = top_reasons_obj if isinstance(top_reasons_obj, list) else []
        if not top_reasons:
            lines.append(f"- {label}: none")
            continue
        tokens: list[str] = []
        for row in top_reasons:
            if not isinstance(row, dict):
                continue
            reason = str(row.get("reason") or "").strip()
            if not reason:
                continue
            raw_count = row.get("count")
            count = raw_count if isinstance(raw_count, int) else 0
            raw_prop = row.get("proportion_of_label_cases")
            prop = raw_prop if isinstance(raw_prop, int | float) else 0.0
            tokens.append(f"`{reason}` ({count}, {float(prop):.1%})")
        lines.append(f"- {label}: {', '.join(tokens) if tokens else 'none'}")
    lines.append(f"- Transition interpretation: {distribution['interpretation']}")
    return lines


def _resolve_path(raw: str, *, base: Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return (base / path).resolve()


def _load_required_json(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"required artifact not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return cast(dict[str, object], payload)


def _load_optional_json(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return cast(dict[str, object], payload)


def _as_promotion_gate_metrics(value: object) -> PromotionGateMetrics:
    if isinstance(value, dict):
        return cast(PromotionGateMetrics, value)
    return project_promotion_gate_metrics({})


def _as_campaign_profile_summary_metrics(value: object) -> CampaignProfileSummaryMetrics:
    if isinstance(value, dict):
        return cast(CampaignProfileSummaryMetrics, value)
    return project_campaign_profile_summary_metrics({})


def _as_portfolio_validation_metrics(value: object) -> PortfolioValidationMetrics:
    if isinstance(value, dict):
        return cast(PortfolioValidationMetrics, value)
    return project_portfolio_validation_metrics({})


def _format_flag_list(value: object) -> str:
    return format_text_list(value, empty="none")


def _format_reason_list(value: object) -> str:
    return format_text_list(value, empty="N/A", separator="; ")


def _portfolio_validation_note(
    metrics: CampaignProfileSummaryMetrics | PortfolioValidationMetrics,
) -> str:
    return portfolio_validation_note(
        metrics["portfolio_validation_status"],
        metrics["portfolio_validation_recommendation"],
    )


def _portfolio_validation_risks(metrics: CampaignProfileSummaryMetrics) -> str:
    return _format_reason_list(metrics["portfolio_validation_major_risks"])


def _portfolio_validation_robustness_note(
    metrics: PortfolioValidationMetrics,
) -> str:
    return format_text_list(
        metrics["portfolio_validation_robustness_label"],
        empty="N/A",
        split_semicolon=False,
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


def _rolling_stability_note(rolling_metrics: RollingStabilityMetrics) -> str:
    rolling_flags = list(rolling_metrics["rolling_instability_flags"])
    if "rolling_regime_dependence" in rolling_flags:
        return "regime-dependent"

    shares = [
        rolling_metrics["rolling_ic_positive_share"],
        rolling_metrics["rolling_rank_ic_positive_share"],
        rolling_metrics["rolling_long_short_positive_share"],
    ]
    finite = [value for value in shares if value is not None]
    if finite and min(finite) >= 0.6:
        return "persistent"
    return "N/A"


def _neutralization_comparison_note(
    metrics: NeutralizationComparisonMetrics,
) -> str:
    comparison = metrics["neutralization_comparison"]
    nested_flags = comparison["interpretation_flags"]
    if nested_flags:
        return ", ".join(nested_flags)
    top_level_flags = list(metrics["neutralization_flags"])
    if top_level_flags:
        return ", ".join(top_level_flags)

    delta_ic = metrics["neutralization_mean_ic_delta"]
    delta_ls = metrics["neutralization_mean_long_short_return_delta"]
    if delta_ic is None and delta_ls is None:
        return "N/A"
    return (
        f"delta IC={format_metric(delta_ic)}, "
        f"delta L/S={format_metric(delta_ls)}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
