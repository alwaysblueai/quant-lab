from __future__ import annotations

import argparse
import datetime
import json
import logging
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, TypedDict, cast

from alpha_lab.artifact_contracts import validate_level12_artifact_payload
from alpha_lab.key_metrics_contracts import (
    CampaignProfileSummaryMetrics,
    CampaignRankingMetrics,
    Level12TransitionDistributionMetrics,
    NeutralizationComparisonMetrics,
    PortfolioValidationMetrics,
    PromotionGateMetrics,
    RollingStabilityMetrics,
    project_campaign_profile_summary_metrics,
    project_campaign_ranking_metrics,
    project_level12_transition_distribution,
    project_portfolio_validation_metrics,
    project_promotion_gate_metrics,
)
from alpha_lab.real_cases.composite.pipeline import CompositeCaseRunResult, run_composite_case
from alpha_lab.real_cases.single_factor.pipeline import (
    SingleFactorCaseRunResult,
    run_single_factor_case,
)
from alpha_lab.reporting.campaign_triage import (
    CampaignTriagePayload,
    build_campaign_triage,
    campaign_rank_sort_key,
)
from alpha_lab.reporting.level2_promotion import Level2PromotionPayload, build_level2_promotion
from alpha_lab.reporting.renderers import write_campaign_report
from alpha_lab.research_evaluation_config import (
    AVAILABLE_RESEARCH_EVALUATION_PROFILES,
    DEFAULT_RESEARCH_EVALUATION_CONFIG,
    ResearchEvaluationConfig,
    get_research_evaluation_config,
    research_evaluation_audit_snapshot,
)

PackageType = Literal["single_factor", "composite"]
Status = Literal["success", "failed", "skipped"]

_ALLOWED_PACKAGE_TYPES: frozenset[str] = frozenset({"single_factor", "composite"})
_ALLOWED_VAULT_MODES: frozenset[str] = frozenset({"skip", "overwrite", "versioned"})
_REQUIRED_CASE_NAMES: tuple[str, str, str] = (
    "bp_single_factor_v1",
    "roe_ttm_single_factor_v1",
    "value_quality_lowvol_v1",
)
logger = logging.getLogger(__name__)


class CampaignArtifactPaths(TypedDict):
    campaign_manifest: Path
    campaign_results: Path
    campaign_summary: Path
    campaign_index: Path


class RankedCasePayload(TypedDict):
    row: CampaignCaseResult
    metrics: dict[str, object]
    promotion_gate_metrics: PromotionGateMetrics
    profile_summary_metrics: CampaignProfileSummaryMetrics
    portfolio_validation_metrics: PortfolioValidationMetrics
    ranking_metrics: CampaignRankingMetrics
    triage: CampaignTriagePayload
    promotion: Level2PromotionPayload
    rank: int | None


class CampaignMetricViews(TypedDict):
    promotion_gate: PromotionGateMetrics
    profile_summary: CampaignProfileSummaryMetrics
    portfolio_validation: PortfolioValidationMetrics
    ranking: CampaignRankingMetrics


class CampaignRenderMeta(TypedDict):
    rendered_report: bool
    rendered_report_path: str | None
    render_status: str
    render_error: str | None


@dataclass(frozen=True)
class CampaignCase:
    case_name: str
    package_type: PackageType
    spec_path: str


@dataclass(frozen=True)
class CampaignConfig:
    campaign_name: str
    campaign_description: str
    output_root_dir: str
    cases: tuple[CampaignCase, ...]
    execution_order: tuple[str, ...]
    case_output_root_dir: str | None = None
    vault_root: str | None = None
    vault_export_mode: str = "versioned"


@dataclass(frozen=True)
class CampaignCaseResult:
    case_name: str
    package_type: PackageType
    status: Status
    output_dir: Path | None
    summary_path: Path | None
    experiment_card_path: Path | None
    run_manifest_path: Path | None
    metrics_path: Path | None
    key_metrics: dict[str, object]
    vault_export: dict[str, object]
    error: str | None = None
    factor_definition_json_path: Path | None = None
    signal_validation_json_path: Path | None = None
    portfolio_recipe_json_path: Path | None = None
    backtest_result_json_path: Path | None = None


@dataclass(frozen=True)
class CampaignRunResult:
    config: CampaignConfig
    output_dir: Path
    case_results: tuple[CampaignCaseResult, ...]
    artifact_paths: CampaignArtifactPaths


def load_research_campaign_1_config(path: str | Path) -> CampaignConfig:
    config_path = Path(path).resolve()
    if not config_path.exists() or not config_path.is_file():
        raise FileNotFoundError(f"campaign config does not exist: {config_path}")

    text = config_path.read_text(encoding="utf-8")
    parsed = _parse_mapping_payload(text, suffix=config_path.suffix.lower())
    config = campaign_config_from_mapping(parsed)
    return resolve_campaign_paths(config, base_dir=config_path.parent)


def campaign_config_from_mapping(data: dict[str, object]) -> CampaignConfig:
    campaign_name = _required_str(data, "campaign_name")
    campaign_description = _required_str(data, "campaign_description")
    output_root_dir = _required_str(data, "output_root_dir")

    raw_cases = data.get("cases")
    if not isinstance(raw_cases, list) or len(raw_cases) == 0:
        raise ValueError("cases must be a non-empty list")

    cases: list[CampaignCase] = []
    for idx, raw in enumerate(raw_cases):
        if not isinstance(raw, dict):
            raise ValueError(f"cases[{idx}] must be an object")
        case_name = _required_str(raw, "case_name")
        raw_package = _required_str(raw, "package_type").lower()
        if raw_package not in _ALLOWED_PACKAGE_TYPES:
            raise ValueError(
                f"cases[{idx}].package_type must be one of {sorted(_ALLOWED_PACKAGE_TYPES)}"
            )
        spec_path = _required_str(raw, "spec_path")
        cases.append(
            CampaignCase(
                case_name=case_name,
                package_type=cast(PackageType, raw_package),
                spec_path=spec_path,
            )
        )

    raw_order = data.get("execution_order")
    if not isinstance(raw_order, list) or len(raw_order) == 0:
        raise ValueError("execution_order must be a non-empty list")
    execution_order: tuple[str, ...] = tuple(_required_list_item_str(raw_order, "execution_order"))

    case_names = [case.case_name for case in cases]
    if len(set(case_names)) != len(case_names):
        raise ValueError("case names in campaign config must be unique")

    if set(execution_order) != set(case_names):
        raise ValueError("execution_order must contain exactly the configured case names")

    if set(case_names) != set(_REQUIRED_CASE_NAMES):
        raise ValueError(
            "research_campaign_1 must include exactly: "
            f"{list(_REQUIRED_CASE_NAMES)}"
        )

    case_output_root_dir = _optional_str(data.get("case_output_root_dir"))

    vault_root: str | None = None
    vault_export_mode = "versioned"
    raw_vault = data.get("vault_export", {})
    if raw_vault is not None:
        if not isinstance(raw_vault, dict):
            raise ValueError("vault_export must be an object when provided")
        vault_root = _optional_str(raw_vault.get("vault_root"))
        raw_mode = _optional_str(raw_vault.get("mode"))
        if raw_mode is not None:
            mode = raw_mode.lower()
            if mode not in _ALLOWED_VAULT_MODES:
                raise ValueError(
                    f"vault_export.mode must be one of {sorted(_ALLOWED_VAULT_MODES)}"
                )
            vault_export_mode = mode

    return CampaignConfig(
        campaign_name=campaign_name,
        campaign_description=campaign_description,
        output_root_dir=output_root_dir,
        cases=tuple(cases),
        execution_order=execution_order,
        case_output_root_dir=case_output_root_dir,
        vault_root=vault_root,
        vault_export_mode=vault_export_mode,
    )


def resolve_campaign_paths(config: CampaignConfig, *, base_dir: Path) -> CampaignConfig:
    def _resolve(path_value: str) -> str:
        path = Path(path_value)
        if not path.is_absolute():
            path = (base_dir / path).resolve()
        else:
            path = path.resolve()
        return str(path)

    cases = tuple(
        CampaignCase(
            case_name=case.case_name,
            package_type=case.package_type,
            spec_path=_resolve(case.spec_path),
        )
        for case in config.cases
    )

    case_output_root = (
        _resolve(config.case_output_root_dir)
        if config.case_output_root_dir is not None
        else None
    )
    vault_root = _resolve(config.vault_root) if config.vault_root is not None else None

    return CampaignConfig(
        campaign_name=config.campaign_name,
        campaign_description=config.campaign_description,
        output_root_dir=_resolve(config.output_root_dir),
        cases=cases,
        execution_order=config.execution_order,
        case_output_root_dir=case_output_root,
        vault_root=vault_root,
        vault_export_mode=config.vault_export_mode,
    )


def run_research_campaign_1(
    config_or_path: CampaignConfig | str | Path,
    *,
    output_root_dir: str | Path | None = None,
    case_output_root_dir: str | Path | None = None,
    evaluation_profile: str | None = None,
    vault_root: str | Path | None = None,
    vault_export_mode: str | None = None,
) -> CampaignRunResult:
    config = (
        config_or_path
        if isinstance(config_or_path, CampaignConfig)
        else load_research_campaign_1_config(config_or_path)
    )

    campaign_out = (
        Path(output_root_dir).resolve()
        if output_root_dir is not None
        else Path(config.output_root_dir).resolve()
    )
    campaign_out.mkdir(parents=True, exist_ok=True)

    case_out = (
        Path(case_output_root_dir).resolve()
        if case_output_root_dir is not None
        else (
            Path(config.case_output_root_dir).resolve()
            if config.case_output_root_dir is not None
            else None
        )
    )

    resolved_vault_root = (
        str(Path(vault_root).resolve())
        if vault_root is not None and str(vault_root).strip()
        else config.vault_root
    )

    resolved_mode = (vault_export_mode or config.vault_export_mode).strip().lower()
    if resolved_mode not in _ALLOWED_VAULT_MODES:
        raise ValueError(f"vault_export_mode must be one of {sorted(_ALLOWED_VAULT_MODES)}")
    resolved_profile = (
        (evaluation_profile or DEFAULT_RESEARCH_EVALUATION_CONFIG.profile_name).strip()
        or DEFAULT_RESEARCH_EVALUATION_CONFIG.profile_name
    )
    evaluation_config = get_research_evaluation_config(resolved_profile)

    case_map = {case.case_name: case for case in config.cases}
    ordered_cases = [case_map[name] for name in config.execution_order]

    case_results: list[CampaignCaseResult] = []
    for case in ordered_cases:
        result = _run_case(
            case,
            case_output_root_dir=case_out,
            evaluation_profile=evaluation_config.profile_name,
            vault_root=resolved_vault_root,
            vault_export_mode=resolved_mode,
        )
        case_results.append(result)
        _write_case_pointer(
            campaign_out,
            result,
            evaluation_config=evaluation_config,
        )

    now_utc = datetime.datetime.now(datetime.UTC).isoformat()
    manifest_payload = {
        "schema_version": "1.0.0",
        "campaign_name": config.campaign_name,
        "campaign_description": config.campaign_description,
        "run_timestamp_utc": now_utc,
        "execution_order": list(config.execution_order),
        "output_root_dir": str(campaign_out),
        "case_output_root_dir": str(case_out) if case_out is not None else None,
        "vault_export_defaults": {
            "vault_root": resolved_vault_root,
            "mode": resolved_mode,
        },
        "evaluation_standard": {
            "profile_name": evaluation_config.profile_name,
            "snapshot": research_evaluation_audit_snapshot(evaluation_config),
        },
        "cases": [asdict(case) for case in ordered_cases],
    }

    transition_distribution = project_level12_transition_distribution(
        _transition_distribution_rows(case_results)
    )
    results_payload = {
        "campaign_name": config.campaign_name,
        "run_timestamp_utc": now_utc,
        "n_cases": len(case_results),
        "n_success": sum(1 for row in case_results if row.status == "success"),
        "n_failed": sum(1 for row in case_results if row.status == "failed"),
        "n_skipped": sum(1 for row in case_results if row.status == "skipped"),
        "evaluation_profile": evaluation_config.profile_name,
        "level12_transition_distribution": transition_distribution,
        "cases": [
            _case_result_to_dict(
                row,
                evaluation_config=evaluation_config,
            )
            for row in case_results
        ],
    }

    manifest_path = campaign_out / "campaign_manifest.json"
    results_path = campaign_out / "campaign_results.json"
    summary_path = campaign_out / "campaign_summary.md"
    index_path = campaign_out / "campaign_index.md"

    _write_json(manifest_path, manifest_payload)
    _write_json(results_path, results_payload)
    summary_path.write_text(
        render_campaign_summary(
            config=config,
            case_results=tuple(case_results),
            run_timestamp_utc=now_utc,
            evaluation_config=evaluation_config,
        ),
        encoding="utf-8",
    )
    index_path.write_text(
        render_campaign_index(
            config=config,
            case_results=tuple(case_results),
            run_timestamp_utc=now_utc,
        ),
        encoding="utf-8",
    )

    return CampaignRunResult(
        config=config,
        output_dir=campaign_out,
        case_results=tuple(case_results),
        artifact_paths={
            "campaign_manifest": manifest_path,
            "campaign_results": results_path,
            "campaign_summary": summary_path,
            "campaign_index": index_path,
        },
    )


def render_campaign_summary(
    *,
    config: CampaignConfig,
    case_results: tuple[CampaignCaseResult, ...],
    run_timestamp_utc: str,
    evaluation_config: ResearchEvaluationConfig = DEFAULT_RESEARCH_EVALUATION_CONFIG,
) -> str:
    ranked_rows = _rank_case_results(
        case_results,
        evaluation_config=evaluation_config,
    )
    transition_distribution = project_level12_transition_distribution(
        _transition_distribution_rows(case_results)
    )
    lines = [
        f"# Campaign Summary: {config.campaign_name}",
        "",
        "## 1. Campaign Objective",
        "",
        config.campaign_description,
        "",
        "## 2. Cases Included",
        "",
    ]

    for row in case_results:
        lines.append(
            f"- `{row.case_name}` ({row.package_type}) - status: `{row.status}`"
        )

    lines += [
        "",
        "## 3. Method Overview",
        "",
        "- Reused existing real-case single-factor and composite research-validation runners.",
        "- Each case writes standardized artifacts under `outputs/real_cases/<case_name>/`.",
        "- Campaign outputs aggregate per-case manifests, metrics, and vault export status.",
        (
            "- Evaluation standard profile: "
            f"`{_campaign_evaluation_profile(case_results, evaluation_config.profile_name)}`"
        ),
        "",
        "## 4. High-Level Findings By Case",
        "",
    ]

    for ranked in ranked_rows:
        row = ranked["row"]
        metrics = ranked["metrics"]
        promotion_gate_metrics = ranked["promotion_gate_metrics"]
        profile_summary_metrics = ranked["profile_summary_metrics"]
        portfolio_metrics = ranked["portfolio_validation_metrics"]
        ranking_metrics = ranked["ranking_metrics"]
        core_metrics = promotion_gate_metrics["core"]
        uncertainty_metrics = promotion_gate_metrics["uncertainty"]
        rolling_metrics = promotion_gate_metrics["rolling"]
        neutralization_metrics = promotion_gate_metrics["neutralization"]
        triage = ranked["triage"]
        promotion = ranked["promotion"]
        rank_text = str(ranked["rank"]) if ranked["rank"] is not None else "N/A"
        ic_ci_text = _fmt_ci(
            uncertainty_metrics["mean_ic_ci_lower"],
            uncertainty_metrics["mean_ic_ci_upper"],
        )
        transition_reasons = _fmt_reasons(
            profile_summary_metrics["level12_transition_reasons"]
        )
        if row.status != "success":
            lines.append(
                "- "
                f"`{row.case_name}`: failed ({row.error or 'no error detail'}), "
                f"Triage={triage['campaign_triage']}, "
                f"Reasons={_fmt_reasons(triage.get('campaign_triage_reasons'))}, "
                f"Promotion={promotion['promotion_decision']}, "
                f"PromotionBlockers={_fmt_reasons(promotion.get('promotion_blockers'))}, "
                f"Transition={_fmt(profile_summary_metrics['level12_transition_label'])}, "
                f"PortfolioValidation={_portfolio_validation_note(profile_summary_metrics)}, "
                f"PortfolioRobustness={_portfolio_validation_robustness_note(portfolio_metrics)}, "
                f"PortfolioBenchmark={_portfolio_validation_benchmark_note(portfolio_metrics)}"
            )
            continue

        lines.append(
            "- "
            f"Rank={rank_text}, "
            f"`{row.case_name}`: "
            f"IC={_fmt(core_metrics['mean_ic'])}, "
            f"IC95%CI={ic_ci_text}, "
            f"ICIR={_fmt(ranking_metrics['ic_ir'])}, "
            f"L/S={_fmt(ranking_metrics['mean_long_short_return'])}, "
            f"RollingIC+={_fmt(rolling_metrics['rolling_ic_positive_share'])}, "
            f"WorstRollingIC={_fmt(rolling_metrics['rolling_ic_min_mean'])}, "
            f"RollingNote={_rolling_stability_note(rolling_metrics)}, "
            f"Turnover={_fmt(core_metrics['mean_long_short_turnover'])}, "
            f"Coverage={_fmt(core_metrics['coverage_mean'])}, "
            f"Uncertainty={_fmt_flags(uncertainty_metrics['uncertainty_flags'])}, "
            f"Neutralization={_neutralization_comparison_note(neutralization_metrics)}, "
            f"Verdict={_fmt(profile_summary_metrics['factor_verdict'])}, "
            f"Triage={triage['campaign_triage']}, "
            f"TriageReasons={_fmt_reasons(triage.get('campaign_triage_reasons'))}, "
            f"Promotion={promotion['promotion_decision']}, "
            f"PromotionReasons={_fmt_reasons(promotion.get('promotion_reasons'))}, "
            f"PromotionBlockers={_fmt_reasons(promotion.get('promotion_blockers'))}, "
            f"Transition={_fmt(profile_summary_metrics['level12_transition_label'])}, "
            f"TransitionReasons={transition_reasons}, "
            f"PortfolioValidation={_portfolio_validation_note(profile_summary_metrics)}, "
            f"PortfolioRobustness={_portfolio_validation_robustness_note(portfolio_metrics)}, "
            f"PortfolioBenchmark={_portfolio_validation_benchmark_note(portfolio_metrics)}, "
            f"PortfolioRisks={_portfolio_validation_risks(profile_summary_metrics)}"
        )

    lines += [
        "",
        "## 5. Comparative Observations",
        "",
        (
            "| Rank | Case | Factor Type | Direction | IC | IC 95% CI | ICIR | Long-Short | "
            "Rolling IC+ | Worst Rolling IC | Rolling Note | "
            "Turnover | Coverage | Uncertainty | Neutralization | Verdict | "
            "Campaign Triage | Triage Reasons | Level 2 Promotion | "
            "Promotion Reasons | Promotion Blockers | L1->L2 Transition | "
            "Level 2 Portfolio Validation | Portfolio Robustness | Portfolio Benchmark Relative | "
            "Portfolio Validation Risks | Vault Export |"
        ),
        (
            "|---:|---|---|---|---:|---|---:|---:|---:|---:|---|---:|---:|---:|---|---|"
            "---|---|---|---|---|---|---|---|---|---|---|"
        ),
    ]

    for ranked in ranked_rows:
        row = ranked["row"]
        metrics = ranked["metrics"]
        promotion_gate_metrics = ranked["promotion_gate_metrics"]
        profile_summary_metrics = ranked["profile_summary_metrics"]
        portfolio_metrics = ranked["portfolio_validation_metrics"]
        ranking_metrics = ranked["ranking_metrics"]
        core_metrics = promotion_gate_metrics["core"]
        uncertainty_metrics = promotion_gate_metrics["uncertainty"]
        rolling_metrics = promotion_gate_metrics["rolling"]
        neutralization_metrics = promotion_gate_metrics["neutralization"]
        triage = ranked["triage"]
        promotion = ranked["promotion"]
        rank_text = str(ranked["rank"]) if ranked["rank"] is not None else "N/A"
        ic_ci_text = _fmt_ci(
            uncertainty_metrics["mean_ic_ci_lower"],
            uncertainty_metrics["mean_ic_ci_upper"],
        )
        lines.append(
            "| "
            f"{rank_text} | "
            f"{row.case_name} | "
            f"{row.package_type} | "
            f"{metrics.get('direction', 'N/A')} | "
            f"{_fmt(core_metrics['mean_ic'])} | "
            f"{ic_ci_text} | "
            f"{_fmt(ranking_metrics['ic_ir'])} | "
            f"{_fmt(ranking_metrics['mean_long_short_return'])} | "
            f"{_fmt(rolling_metrics['rolling_ic_positive_share'])} | "
            f"{_fmt(rolling_metrics['rolling_ic_min_mean'])} | "
            f"{_rolling_stability_note(rolling_metrics)} | "
            f"{_fmt(core_metrics['mean_long_short_turnover'])} | "
            f"{_fmt(core_metrics['coverage_mean'])} | "
            f"{_fmt_flags(uncertainty_metrics['uncertainty_flags'])} | "
            f"{_neutralization_comparison_note(neutralization_metrics)} | "
            f"{_fmt(profile_summary_metrics['factor_verdict'])} | "
            f"{triage['campaign_triage']} | "
            f"{_fmt_reasons(triage.get('campaign_triage_reasons'))} | "
            f"{promotion['promotion_decision']} | "
            f"{_fmt_reasons(promotion.get('promotion_reasons'))} | "
            f"{_fmt_reasons(promotion.get('promotion_blockers'))} | "
            f"{_fmt(profile_summary_metrics['level12_transition_label'])} | "
            f"{_portfolio_validation_note(profile_summary_metrics)} | "
            f"{_portfolio_validation_robustness_note(portfolio_metrics)} | "
            f"{_portfolio_validation_benchmark_note(portfolio_metrics)} | "
            f"{_portfolio_validation_risks(profile_summary_metrics)} | "
            f"{row.vault_export.get('status', 'N/A')} "
            "|"
        )

    lines += [
        "",
        "### Level 1->Level 2 Transition Distribution",
        "",
    ]
    lines.extend(_transition_distribution_markdown_lines(transition_distribution))

    best_icir = _best_case_by_metric(case_results, "ic_ir")
    best_promoted = next(
        (
            payload["row"].case_name
            for payload in ranked_rows
            if payload["rank"] is not None
            and payload["promotion"].get("promotion_decision") == "Promote to Level 2"
        ),
        None,
    )
    lines += ["", "## 6. Campaign Triage Ranking", ""]
    ranked_success = [row for row in ranked_rows if row["rank"] is not None]
    if ranked_success:
        for ranked in ranked_success:
            row = ranked["row"]
            ranking_metrics = ranked["ranking_metrics"]
            profile_summary_metrics = ranked["profile_summary_metrics"]
            portfolio_metrics = ranked["portfolio_validation_metrics"]
            triage = ranked["triage"]
            promotion = ranked["promotion"]
            lines.append(
                "- "
                f"#{ranked['rank']} `{row.case_name}`: "
                f"Triage={triage['campaign_triage']}, "
                f"ICIR={_fmt(ranking_metrics['ic_ir'])}, "
                f"L/S={_fmt(ranking_metrics['mean_long_short_return'])}, "
                f"Rolling+Min={_fmt(triage.get('campaign_rank_stability_metric'))}, "
                f"RiskCount={_fmt(triage.get('campaign_rank_risk_count'))}, "
                f"SupportCount={_fmt(triage.get('campaign_rank_support_count'))}, "
                f"Reasons={_fmt_reasons(triage.get('campaign_triage_reasons'))}, "
                f"Promotion={promotion['promotion_decision']}, "
                f"PromotionBlockers={_fmt_reasons(promotion.get('promotion_blockers'))}, "
                f"Transition={_fmt(profile_summary_metrics['level12_transition_label'])}, "
                f"PortfolioValidation={_portfolio_validation_note(profile_summary_metrics)}, "
                f"PortfolioRobustness={_portfolio_validation_robustness_note(portfolio_metrics)}, "
                f"PortfolioBenchmark={_portfolio_validation_benchmark_note(portfolio_metrics)}, "
                f"PortfolioRisks={_portfolio_validation_risks(profile_summary_metrics)}"
            )
    else:
        lines.append("- N/A (no successful cases to rank).")

    lines += [
        "",
        (
            "- Best ICIR case (successful runs only): "
            f"`{best_icir}`" if best_icir is not None else
            "- Best ICIR case: N/A (no successful runs with finite ICIR)."
        ),
        (
            "- Promotion gate pass (highest-ranked triage): "
            f"`{best_promoted}`"
            if best_promoted is not None
            else "- Promotion gate pass: N/A (no case promoted to Level 2)."
        ),
        "",
        "## 7. Limitations",
        "",
        (
            "- Campaign-level conclusions rely only on generated metrics; "
            "no manual interpretation added."
        ),
        "- Failed/missing cases are marked explicitly and not imputed.",
        "- This v1 campaign runner is intentionally explicit to research_campaign_1 only.",
        "",
        "## 8. Recommended Next Steps",
        "",
        "- Add deeper attribution diagnostics for successful cases.",
        "- Add a follow-up campaign with robustness windows and alternate universes.",
        "- Move accepted findings into quant-knowledge interpretation sections.",
        "",
        f"_Run timestamp (UTC): `{run_timestamp_utc}`_",
        "",
    ]
    return "\n".join(lines)


def render_campaign_index(
    *,
    config: CampaignConfig,
    case_results: tuple[CampaignCaseResult, ...],
    run_timestamp_utc: str,
) -> str:
    lines = [
        f"# Campaign Index: {config.campaign_name}",
        "",
        config.campaign_description,
        "",
        "## Cases",
        "",
    ]

    for row in case_results:
        pointer_rel = f"{row.case_name}/case_output_pointer.json"
        lines.append(
            "- "
            f"`{row.case_name}` ({row.package_type}) - {row.status} - "
            f"[pointer]({pointer_rel})"
        )
        if row.experiment_card_path is not None:
            lines.append(f"  card: `{row.experiment_card_path}`")

    lines += [
        "",
        "## Notes",
        "",
        "- This index is campaign-level and vault-friendly.",
        "- Use each case pointer file to locate canonical run artifacts.",
        "",
        f"_Run timestamp (UTC): `{run_timestamp_utc}`_",
        "",
    ]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="research-campaign-1",
        description=(
            "Run research_campaign_1 through the default Level 1/2 workflow "
            "(Level 1 evaluation -> campaign triage -> Level 2 promotion gate -> "
            "Level 2 portfolio validation)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "campaign_config",
        nargs="?",
        default="configs/campaigns/research_campaign_1/campaign.yaml",
        help="Path to campaign manifest YAML/JSON.",
    )
    parser.add_argument(
        "--output-root-dir",
        default=None,
        help="Override campaign output root directory.",
    )
    parser.add_argument(
        "--case-output-root-dir",
        default=None,
        help="Optional override for per-case output root passed to case runners.",
    )
    parser.add_argument(
        "--evaluation-profile",
        default=DEFAULT_RESEARCH_EVALUATION_CONFIG.profile_name,
        choices=sorted(AVAILABLE_RESEARCH_EVALUATION_PROFILES),
        help=(
            "Research evaluation profile controlling factor verdict standards, "
            "case-level triage, Level 2 promotion gate thresholds, and Level 2 "
            "portfolio-validation guardrails."
        ),
    )
    parser.add_argument(
        "--vault-root",
        default=None,
        help="Optional vault root override passed to case runners.",
    )
    parser.add_argument(
        "--vault-export-mode",
        choices=["skip", "overwrite", "versioned"],
        default=None,
        help="Optional vault export mode override passed to case runners.",
    )
    parser.add_argument(
        "--render-report",
        action="store_true",
        help="Render campaign_report.md after a successful campaign run.",
    )
    parser.add_argument(
        "--render-overwrite",
        action="store_true",
        help="Overwrite existing campaign_report.md when rendering is enabled.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        result = run_research_campaign_1(
            args.campaign_config,
            output_root_dir=args.output_root_dir,
            case_output_root_dir=args.case_output_root_dir,
            evaluation_profile=args.evaluation_profile,
            vault_root=args.vault_root,
            vault_export_mode=args.vault_export_mode,
        )
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        parser.error(str(exc))

    render_meta = _render_campaign_report(
        output_dir=result.output_dir,
        enabled=bool(args.render_report),
        overwrite=bool(args.render_overwrite),
    )
    _update_campaign_results(
        result.artifact_paths["campaign_results"],
        render_meta,
    )

    success = sum(1 for row in result.case_results if row.status == "success")
    failed = sum(1 for row in result.case_results if row.status == "failed")
    promoted = sum(
        1
        for row in result.case_results
        if str(row.key_metrics.get("promotion_decision") or "").strip()
        == "Promote to Level 2"
    )
    portfolio_completed = sum(
        1
        for row in result.case_results
        if str(row.key_metrics.get("portfolio_validation_status") or "").strip()
        == "completed"
    )
    results_payload = _load_json(result.artifact_paths["campaign_results"])
    raw_profile = str(results_payload.get("evaluation_profile") or "").strip()
    evaluation_profile = raw_profile or _campaign_evaluation_profile(
        result.case_results,
        DEFAULT_RESEARCH_EVALUATION_CONFIG.profile_name,
    )

    print("")
    print(f"  Campaign : {result.config.campaign_name}")
    print("  Status   : completed")
    print(f"  Cases    : {len(result.case_results)} total / {success} success / {failed} failed")
    print(f"  Evaluation Profile  : {evaluation_profile}")
    print(f"  Promotion Gate Pass : {promoted}")
    print(f"  Portfolio Validated : {portfolio_completed}")
    print(f"  Output   : {result.output_dir}")
    print(f"  Manifest : {result.artifact_paths['campaign_manifest']}")
    print(f"  Results  : {result.artifact_paths['campaign_results']}")
    print(f"  Summary  : {result.artifact_paths['campaign_summary']}")
    print(f"  Index    : {result.artifact_paths['campaign_index']}")
    print(f"  Report Render Status: {render_meta.get('render_status')}")
    print(f"  Report Path         : {render_meta.get('rendered_report_path')}")

    return 0


def _render_campaign_report(
    *,
    output_dir: Path,
    enabled: bool,
    overwrite: bool,
) -> CampaignRenderMeta:
    if not enabled:
        return {
            "rendered_report": False,
            "rendered_report_path": None,
            "render_status": "skipped",
            "render_error": None,
        }

    try:
        report_path = write_campaign_report(output_dir, overwrite=overwrite)
        return {
            "rendered_report": True,
            "rendered_report_path": str(report_path),
            "render_status": "success",
            "render_error": None,
        }
    except Exception as exc:
        logger.warning(
            "Campaign report rendering failed for %s: %s",
            output_dir,
            exc,
        )
        return {
            "rendered_report": False,
            "rendered_report_path": None,
            "render_status": "failed",
            "render_error": str(exc),
        }


def _update_campaign_results(
    campaign_results_path: Path,
    render_meta: CampaignRenderMeta,
) -> None:
    try:
        payload = _load_json(campaign_results_path)
        payload.update(render_meta)
        _write_json(campaign_results_path, payload)
    except Exception as exc:
        logger.warning(
            "Failed to persist campaign render metadata into %s: %s",
            campaign_results_path,
            exc,
        )


def _run_case(
    case: CampaignCase,
    *,
    case_output_root_dir: Path | None,
    evaluation_profile: str,
    vault_root: str | None,
    vault_export_mode: str,
) -> CampaignCaseResult:
    try:
        run: SingleFactorCaseRunResult | CompositeCaseRunResult
        if case.package_type == "single_factor":
            run = run_single_factor_case(
                case.spec_path,
                output_root_dir=case_output_root_dir,
                evaluation_profile=evaluation_profile,
                vault_root=vault_root,
                vault_export_mode=vault_export_mode,
            )
        else:
            run = run_composite_case(
                case.spec_path,
                output_root_dir=case_output_root_dir,
                evaluation_profile=evaluation_profile,
                vault_root=vault_root,
                vault_export_mode=vault_export_mode,
            )

        manifest_path = run.artifact_paths["run_manifest"]
        metrics_path = run.artifact_paths["metrics"]
        summary_path = run.artifact_paths["summary"]
        card_path = run.artifact_paths["experiment_card"]
        factor_definition_json_path = run.artifact_paths["factor_definition_json"]
        signal_validation_json_path = run.artifact_paths["signal_validation_json"]
        portfolio_recipe_json_path = run.artifact_paths["portfolio_recipe_json"]
        backtest_result_json_path = run.artifact_paths["backtest_result_json"]

        manifest = _load_json(manifest_path)
        key_metrics = _extract_key_metrics(metrics_path)

        direction = key_metrics.get("direction")
        if direction is None:
            direction = _extract_direction_from_manifest(manifest, package_type=case.package_type)
        if direction is not None:
            key_metrics["direction"] = direction

        vault_export = cast(
            dict[str, object],
            manifest.get(
                "vault_export",
                {"enabled": False, "mode": "skip", "status": "skipped", "error": None},
            ),
        )

        return CampaignCaseResult(
            case_name=case.case_name,
            package_type=case.package_type,
            status="success",
            output_dir=run.output_dir,
            summary_path=summary_path,
            experiment_card_path=card_path,
            run_manifest_path=manifest_path,
            metrics_path=metrics_path,
            key_metrics=key_metrics,
            vault_export=vault_export,
            error=None,
            factor_definition_json_path=factor_definition_json_path,
            signal_validation_json_path=signal_validation_json_path,
            portfolio_recipe_json_path=portfolio_recipe_json_path,
            backtest_result_json_path=backtest_result_json_path,
        )
    except Exception as exc:
        return CampaignCaseResult(
            case_name=case.case_name,
            package_type=case.package_type,
            status="failed",
            output_dir=None,
            summary_path=None,
            experiment_card_path=None,
            run_manifest_path=None,
            metrics_path=None,
            key_metrics={},
            vault_export={
                "enabled": bool(vault_root),
                "mode": vault_export_mode,
                "status": "failed",
                "error": str(exc),
            },
            error=str(exc),
            factor_definition_json_path=None,
            signal_validation_json_path=None,
            portfolio_recipe_json_path=None,
            backtest_result_json_path=None,
        )


def _extract_key_metrics(metrics_path: Path) -> dict[str, object]:
    payload = _load_json(metrics_path)
    raw_metrics = payload.get("metrics", {})
    if not isinstance(raw_metrics, dict):
        return {}

    projected = _project_campaign_metric_views(raw_metrics)
    extracted: dict[str, object] = {}

    promotion_gate = projected["promotion_gate"]
    _copy_projected_fields(
        extracted,
        raw_metrics=raw_metrics,
        projected={
            "factor_verdict": promotion_gate["factor_verdict"],
            "campaign_triage": promotion_gate["campaign_triage"],
        },
    )
    _copy_projected_fields(
        extracted,
        raw_metrics=raw_metrics,
        projected=promotion_gate["core"],
    )
    _copy_projected_fields(
        extracted,
        raw_metrics=raw_metrics,
        projected=promotion_gate["uncertainty"],
    )
    _copy_projected_fields(
        extracted,
        raw_metrics=raw_metrics,
        projected=promotion_gate["rolling"],
    )
    _copy_projected_fields(
        extracted,
        raw_metrics=raw_metrics,
        projected=promotion_gate["neutralization"],
    )
    _copy_projected_fields(
        extracted,
        raw_metrics=raw_metrics,
        projected=projected["profile_summary"],
    )
    # Always include Level 1->2 transition diagnostics so campaign views can
    # compare transition outcomes even for legacy metrics payloads.
    extracted["level12_transition_summary"] = projected["profile_summary"][
        "level12_transition_summary"
    ]
    extracted["level12_transition_label"] = projected["profile_summary"][
        "level12_transition_label"
    ]
    extracted["level12_transition_interpretation"] = projected["profile_summary"][
        "level12_transition_interpretation"
    ]
    extracted["level12_transition_reasons"] = projected["profile_summary"][
        "level12_transition_reasons"
    ]
    extracted["level12_transition_confirmation_note"] = projected["profile_summary"][
        "level12_transition_confirmation_note"
    ]
    _copy_projected_fields(
        extracted,
        raw_metrics=raw_metrics,
        projected=projected["portfolio_validation"],
    )
    _copy_projected_fields(
        extracted,
        raw_metrics=raw_metrics,
        projected=projected["ranking"],
    )

    for key in _CAMPAIGN_KEY_METRICS_COMPAT_PASSTHROUGH_FIELDS:
        if key in raw_metrics:
            extracted[key] = raw_metrics[key]
    return extracted


_CAMPAIGN_KEY_METRICS_COMPAT_PASSTHROUGH_FIELDS: tuple[str, ...] = (
    "direction",
    "instability_flags",
    "campaign_triage_priority",
    "campaign_rank_primary_metric_name",
    "campaign_rank_primary_metric",
    "campaign_rank_secondary_metric_name",
    "campaign_rank_secondary_metric",
    "campaign_rank_stability_metric_name",
    "campaign_rank_stability_metric",
    "campaign_rank_support_count",
    "campaign_rank_risk_count",
    "campaign_rank_rule",
    "portfolio_validation_remains_credible",
    "portfolio_validation_base_mean_portfolio_return",
    "portfolio_validation_base_mean_turnover",
    "portfolio_validation_base_cost_adjusted_return_review_rate",
    "portfolio_validation_robustness_label",
    "portfolio_validation_support_reasons",
    "portfolio_validation_fragility_reasons",
    "portfolio_validation_scenario_sensitivity_notes",
    "portfolio_validation_benchmark_support_note",
    "portfolio_validation_cost_sensitivity_note",
    "portfolio_validation_concentration_turnover_note",
    "portfolio_validation_benchmark_name",
    "portfolio_validation_benchmark_active_return",
    "portfolio_validation_benchmark_information_ratio",
    "portfolio_validation_benchmark_relative_max_drawdown",
    "portfolio_validation_benchmark_relative_risks",
    "research_evaluation_profile",
    "research_evaluation_snapshot",
    "neutralization_comparison_flags",
    "missingness_mean",
)


def _project_campaign_metric_views(metrics: Mapping[str, object]) -> CampaignMetricViews:
    return {
        "promotion_gate": project_promotion_gate_metrics(metrics),
        "profile_summary": project_campaign_profile_summary_metrics(metrics),
        "portfolio_validation": project_portfolio_validation_metrics(metrics),
        "ranking": project_campaign_ranking_metrics(metrics),
    }


def _copy_projected_fields(
    extracted: dict[str, object],
    *,
    raw_metrics: Mapping[str, object],
    projected: Mapping[str, object],
) -> None:
    for key, value in projected.items():
        if key in raw_metrics:
            extracted[key] = value


def _extract_direction_from_manifest(
    manifest: dict[str, object],
    *,
    package_type: PackageType,
) -> str | None:
    spec = manifest.get("spec", {})
    if not isinstance(spec, dict):
        return None

    if package_type == "single_factor":
        value = spec.get("direction")
        return str(value) if value is not None else None

    raw_components = spec.get("components", [])
    if not isinstance(raw_components, list):
        return None

    directions = sorted(
        {
            str(row.get("direction"))
            for row in raw_components
            if isinstance(row, dict) and row.get("direction") is not None
        }
    )
    if not directions:
        return None
    return "/".join(directions)


def _write_case_pointer(
    campaign_output_dir: Path,
    result: CampaignCaseResult,
    *,
    evaluation_config: ResearchEvaluationConfig = DEFAULT_RESEARCH_EVALUATION_CONFIG,
) -> None:
    case_dir = campaign_output_dir / result.case_name
    case_dir.mkdir(parents=True, exist_ok=True)
    pointer_path = case_dir / "case_output_pointer.json"

    triage = build_campaign_triage(
        result.key_metrics,
        status=result.status,
        thresholds=evaluation_config.campaign_triage,
    ).to_dict()
    promotion = build_level2_promotion(
        result.key_metrics,
        status=result.status,
        thresholds=evaluation_config.level2_promotion,
    ).to_dict()
    payload = {
        "case_name": result.case_name,
        "package_type": result.package_type,
        "status": result.status,
        "output_dir": str(result.output_dir) if result.output_dir is not None else None,
        "summary_path": (
            str(result.summary_path)
            if result.summary_path is not None
            else None
        ),
        "experiment_card_path": (
            str(result.experiment_card_path)
            if result.experiment_card_path is not None
            else None
        ),
        "run_manifest_path": (
            str(result.run_manifest_path)
            if result.run_manifest_path is not None
            else None
        ),
        "metrics_path": (
            str(result.metrics_path)
            if result.metrics_path is not None
            else None
        ),
        "factor_definition_json_path": (
            str(result.factor_definition_json_path)
            if result.factor_definition_json_path is not None
            else None
        ),
        "signal_validation_json_path": (
            str(result.signal_validation_json_path)
            if result.signal_validation_json_path is not None
            else None
        ),
        "portfolio_recipe_json_path": (
            str(result.portfolio_recipe_json_path)
            if result.portfolio_recipe_json_path is not None
            else None
        ),
        "backtest_result_json_path": (
            str(result.backtest_result_json_path)
            if result.backtest_result_json_path is not None
            else None
        ),
        "campaign_triage": triage,
        "level2_promotion": promotion,
        "vault_export": result.vault_export,
        "error": result.error,
    }
    _write_json(pointer_path, payload)


def _case_result_to_dict(
    result: CampaignCaseResult,
    *,
    evaluation_config: ResearchEvaluationConfig = DEFAULT_RESEARCH_EVALUATION_CONFIG,
) -> dict[str, object]:
    triage = build_campaign_triage(
        result.key_metrics,
        status=result.status,
        thresholds=evaluation_config.campaign_triage,
    ).to_dict()
    promotion = build_level2_promotion(
        result.key_metrics,
        status=result.status,
        thresholds=evaluation_config.level2_promotion,
    ).to_dict()
    return {
        "case_name": result.case_name,
        "package_type": result.package_type,
        "status": result.status,
        "output_dir": str(result.output_dir) if result.output_dir is not None else None,
        "summary_path": (
            str(result.summary_path)
            if result.summary_path is not None
            else None
        ),
        "experiment_card_path": (
            str(result.experiment_card_path)
            if result.experiment_card_path is not None
            else None
        ),
        "run_manifest_path": (
            str(result.run_manifest_path)
            if result.run_manifest_path is not None
            else None
        ),
        "metrics_path": (
            str(result.metrics_path)
            if result.metrics_path is not None
            else None
        ),
        "factor_definition_json_path": (
            str(result.factor_definition_json_path)
            if result.factor_definition_json_path is not None
            else None
        ),
        "signal_validation_json_path": (
            str(result.signal_validation_json_path)
            if result.signal_validation_json_path is not None
            else None
        ),
        "portfolio_recipe_json_path": (
            str(result.portfolio_recipe_json_path)
            if result.portfolio_recipe_json_path is not None
            else None
        ),
        "backtest_result_json_path": (
            str(result.backtest_result_json_path)
            if result.backtest_result_json_path is not None
            else None
        ),
        "key_metrics": result.key_metrics,
        "campaign_triage": triage,
        "level2_promotion": promotion,
        "vault_export": result.vault_export,
        "error": result.error,
    }


def _rank_case_results(
    case_results: tuple[CampaignCaseResult, ...],
    *,
    evaluation_config: ResearchEvaluationConfig = DEFAULT_RESEARCH_EVALUATION_CONFIG,
) -> list[RankedCasePayload]:
    ranked: list[RankedCasePayload] = []
    for row in case_results:
        projected = _project_campaign_metric_views(row.key_metrics)
        triage = build_campaign_triage(
            row.key_metrics,
            status=row.status,
            thresholds=evaluation_config.campaign_triage,
        ).to_dict()
        promotion = build_level2_promotion(
            row.key_metrics,
            status=row.status,
            thresholds=evaluation_config.level2_promotion,
        ).to_dict()
        ranked.append(
            {
                "row": row,
                "metrics": row.key_metrics,
                "promotion_gate_metrics": projected["promotion_gate"],
                "profile_summary_metrics": projected["profile_summary"],
                "portfolio_validation_metrics": projected["portfolio_validation"],
                "ranking_metrics": projected["ranking"],
                "triage": triage,
                "promotion": promotion,
                "rank": None,
            }
        )

    ranked.sort(
        key=lambda payload: campaign_rank_sort_key(
            payload["row"].case_name,
            status=payload["row"].status,
            metrics=payload["metrics"],
            thresholds=evaluation_config.campaign_triage,
        )
    )

    next_rank = 1
    for payload in ranked:
        row = payload["row"]
        if row.status == "success":
            payload["rank"] = next_rank
            next_rank += 1
    return ranked


def _best_case_by_metric(
    case_results: tuple[CampaignCaseResult, ...],
    metric_name: str,
) -> str | None:
    best_case: str | None = None
    best_value: float | None = None
    for row in case_results:
        if row.status != "success":
            continue
        raw = row.key_metrics.get(metric_name)
        if not isinstance(raw, (int, float)):
            continue
        value = float(raw)
        if not math.isfinite(value):
            continue
        if best_value is None or value > best_value:
            best_value = value
            best_case = row.case_name
    return best_case


def _transition_distribution_rows(
    case_results: tuple[CampaignCaseResult, ...] | list[CampaignCaseResult],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in case_results:
        projected = project_campaign_profile_summary_metrics(row.key_metrics)
        transition_label = row.key_metrics.get("level12_transition_label")
        if not isinstance(transition_label, str) or not transition_label.strip():
            transition_label = projected["level12_transition_label"]
        transition_reasons = row.key_metrics.get("level12_transition_reasons")
        if not isinstance(transition_reasons, list | tuple):
            transition_reasons = list(projected["level12_transition_reasons"])
        rows.append(
            {
                "case_name": row.case_name,
                "level12_transition_label": transition_label,
                "level12_transition_reasons": transition_reasons,
            }
        )
    return rows


def _transition_distribution_markdown_lines(
    distribution: Level12TransitionDistributionMetrics,
) -> list[str]:
    counts = distribution["counts_by_transition_label"]
    proportions = distribution["proportions_by_transition_label"]
    representatives = distribution["representative_cases_by_transition_label"]
    reason_rollups_obj = distribution.get("reason_rollup_by_transition_label", {})
    reason_rollups = reason_rollups_obj if isinstance(reason_rollups_obj, dict) else {}
    lines = [
        (
            "- Cases total / observed transition labels / missing labels: "
            f"`{distribution['n_cases']}` / "
            f"`{distribution['n_cases_with_transition_label']}` / "
            f"`{distribution['n_cases_missing_transition_label']}`"
        )
    ]
    support_note = str(distribution.get("support_note") or "").strip()
    if support_note:
        lines.append(f"- Transition distribution support: {support_note}")
    thresholds_obj = distribution.get("minimum_support_thresholds")
    thresholds = thresholds_obj if isinstance(thresholds_obj, dict) else {}
    if thresholds:
        threshold_tokens = [f"{key}={value}" for key, value in sorted(thresholds.items())]
        lines.append("- Transition support thresholds: " + ", ".join(threshold_tokens))
    for label in (
        "Confirmed at portfolio level",
        "Weakened at portfolio level",
        "Fragile after promotion",
        "Improved at portfolio level",
        "Inconclusive transition",
    ):
        case_tokens = representatives.get(label) or []
        case_text = ", ".join(f"`{name}`" for name in case_tokens) if case_tokens else "none"
        lines.append(
            f"- {label}: {counts.get(label, 0)} ({proportions.get(label, 0.0):.1%}); "
            f"representative cases: {case_text}"
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
        dominant_reasons_obj = rollup.get("dominant_reasons")
        dominant_reasons = (
            dominant_reasons_obj if isinstance(dominant_reasons_obj, list) else []
        )
        top_reasons_obj = rollup.get("top_reasons")
        top_reasons = top_reasons_obj if isinstance(top_reasons_obj, list) else []
        reason_rows = dominant_reasons if dominant_reasons else top_reasons
        if not reason_rows:
            lines.append(f"- {label}: none")
            continue
        reason_tokens: list[str] = []
        for row in reason_rows:
            if not isinstance(row, dict):
                continue
            reason = str(row.get("reason") or "").strip()
            if not reason:
                continue
            raw_count = row.get("count")
            count = raw_count if isinstance(raw_count, int) else 0
            raw_prop = row.get("proportion_of_label_cases")
            prop = raw_prop if isinstance(raw_prop, int | float) else 0.0
            reason_tokens.append(f"`{reason}` ({count}, {float(prop):.1%})")
        support_suffix = ""
        if not dominant_reasons:
            rollup_support_note = str(rollup.get("support_note") or "tentative due to low support")
            support_suffix = f" [{rollup_support_note}]"
        lines.append(
            f"- {label}: {', '.join(reason_tokens) if reason_tokens else 'none'}{support_suffix}"
        )
    lines.append(f"- Interpretation: {distribution['interpretation']}")
    lines.append("")
    return lines


def _fmt(value: object) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return "N/A"
        return f"{value:.6f}"
    return str(value)


def _fmt_ci(lower: object, upper: object) -> str:
    if not isinstance(lower, (int, float)) or not isinstance(upper, (int, float)):
        return "N/A"
    left = float(lower)
    right = float(upper)
    if not math.isfinite(left) or not math.isfinite(right):
        return "N/A"
    return f"[{left:.6f}, {right:.6f}]"


def _campaign_evaluation_profile(
    case_results: tuple[CampaignCaseResult, ...],
    fallback_profile: str = DEFAULT_RESEARCH_EVALUATION_CONFIG.profile_name,
) -> str:
    profiles = sorted(
        {
            str(row.key_metrics.get("research_evaluation_profile")).strip()
            for row in case_results
            if str(row.key_metrics.get("research_evaluation_profile")).strip()
        }
    )
    if not profiles:
        return fallback_profile
    if len(profiles) == 1:
        return profiles[0]
    return "mixed: " + ", ".join(profiles)


def _fmt_flags(value: object) -> str:
    if value is None:
        return "none"
    if isinstance(value, (list, tuple)):
        tokens = [str(v).strip() for v in value if str(v).strip()]
        return ", ".join(tokens) if tokens else "none"
    text = str(value).strip()
    if not text:
        return "none"
    if ";" in text:
        return ", ".join(token.strip() for token in text.split(";") if token.strip())
    return text


def _fmt_reasons(value: object) -> str:
    if value is None:
        return "none"
    if isinstance(value, (list, tuple)):
        tokens = [str(v).strip() for v in value if str(v).strip()]
        return "; ".join(tokens) if tokens else "none"
    text = str(value).strip()
    if not text:
        return "none"
    if ";" in text:
        return "; ".join(token.strip() for token in text.split(";") if token.strip())
    if "," in text:
        return "; ".join(token.strip() for token in text.split(",") if token.strip())
    return text


def _portfolio_validation_note(
    metrics: CampaignProfileSummaryMetrics | PortfolioValidationMetrics,
) -> str:
    status = _fmt(metrics["portfolio_validation_status"])
    recommendation = _fmt(metrics["portfolio_validation_recommendation"])
    if status == "N/A" and recommendation == "N/A":
        return "N/A"
    return f"{status} ({recommendation})"


def _portfolio_validation_risks(metrics: CampaignProfileSummaryMetrics) -> str:
    return _fmt_reasons(metrics["portfolio_validation_major_risks"])


def _portfolio_validation_robustness_note(metrics: PortfolioValidationMetrics) -> str:
    return _fmt(metrics["portfolio_validation_robustness_label"])


def _portfolio_validation_benchmark_note(metrics: PortfolioValidationMetrics) -> str:
    status = _fmt(metrics["portfolio_validation_benchmark_relative_status"])
    assessment = _fmt(metrics["portfolio_validation_benchmark_relative_assessment"])
    excess = _fmt(metrics["portfolio_validation_benchmark_excess_return"])
    tracking_error = _fmt(metrics["portfolio_validation_benchmark_tracking_error"])
    if status == "N/A" and assessment == "N/A":
        return "N/A"
    return (
        f"{status} ({assessment}), "
        f"excess={excess}, "
        f"tracking_error={tracking_error}"
    )


def _rolling_stability_note(metrics: RollingStabilityMetrics) -> str:
    flags = _fmt_flags(metrics["rolling_instability_flags"])
    if "rolling_regime_dependence" in flags:
        return "regime-dependent"

    shares = [
        metrics["rolling_ic_positive_share"],
        metrics["rolling_rank_ic_positive_share"],
        metrics["rolling_long_short_positive_share"],
    ]
    finite = [value for value in shares if value is not None]

    if finite and min(finite) >= 0.6:
        return "persistent"
    return "N/A"


def _neutralization_comparison_note(metrics: NeutralizationComparisonMetrics) -> str:
    comparison = metrics["neutralization_comparison"]
    nested_flags = _fmt_flags(comparison["interpretation_flags"])
    if nested_flags != "none":
        return nested_flags

    flags = _fmt_flags(metrics["neutralization_flags"])
    if flags != "none":
        return flags

    delta_ic = metrics["neutralization_mean_ic_delta"]
    delta_ls = metrics["neutralization_mean_long_short_return_delta"]
    if delta_ic is None and delta_ls is None:
        return "N/A"
    return f"delta IC={_fmt(delta_ic)}, delta L/S={_fmt(delta_ls)}"


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    validate_level12_artifact_payload(payload, artifact_name=path.name, source=path)
    return cast(dict[str, object], payload)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    payload_obj = dict(payload)
    validate_level12_artifact_payload(payload_obj, artifact_name=path.name, source=path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload_obj, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def _parse_mapping_payload(text: str, *, suffix: str) -> dict[str, object]:
    parsed: object
    if suffix == ".json":
        parsed = json.loads(text)
    else:
        parsed = _yaml_load(text)

    if not isinstance(parsed, dict):
        raise ValueError("campaign config root must be an object")
    return cast(dict[str, object], parsed)


def _yaml_load(text: str) -> object:
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - import guard
        raise RuntimeError("PyYAML is required to load campaign YAML configs") from exc

    return yaml.safe_load(text)


def _required_str(data: dict[str, object], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("expected a string or null")
    text = value.strip()
    return text if text else None


def _required_list_item_str(values: list[object], field_name: str) -> list[str]:
    out: list[str] = []
    for idx, value in enumerate(values):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name}[{idx}] must be a non-empty string")
        out.append(value)
    return out


if __name__ == "__main__":
    raise SystemExit(main())
