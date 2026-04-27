from __future__ import annotations

# ruff: noqa: E501
import html
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

import pandas as pd

from alpha_lab.artifact_contracts import validate_level12_artifact_payload
from alpha_lab.reporting.display_helpers import (
    as_object_dict,
    as_object_list,
    parse_text_list,
    safe_text,
)
from alpha_lab.reporting.research_artifact_manifest import (
    CANONICAL_ARTIFACT_REQUIREMENTS as _CANONICAL_ARTIFACT_REQUIREMENTS,
)
from alpha_lab.reporting.research_artifact_manifest import (
    WORKFLOW_CLOSURE_ARTIFACT_REQUIREMENTS as _WORKFLOW_CLOSURE_ARTIFACT_REQUIREMENTS,
)
from alpha_lab.reporting.workflow_artifact_service import (
    persist_dashboard_workflow_artifacts as _persist_dashboard_workflow_artifacts,
)
from alpha_lab.reporting.workflow_artifact_service import (
    persist_workflow_closure_artifacts as _persist_workflow_closure_artifacts,
)

from .research_dashboard_schema import (
    ArtifactLoadDiagnostic,
    CandidateRecipe,
    CandidateRecipeGenerationConfig,
    CandidateRecipeGenerationResult,
    DashboardOverview,
    ExperimentRegistryEntry,
    FactorComparisonRow,
    FactorDetail,
    FactorSetConstructionConfig,
    FactorSetConstructionResult,
    FactorSetDefinition,
    FactorSetScoreSummary,
    FactorShortlistConfig,
    FactorShortlistEntry,
    FactorShortlistResult,
    FactorSummary,
    NextStepRecommendation,
    NextStepRecommendationResult,
    PortfolioBacktestSummary,
    PortfolioRecipeSummary,
    RecipeComparisonRow,
    RecipeComparisonView,
    RecipeHeadToHeadInsight,
    RecipeLeaderboardEntry,
    ResearchDashboardData,
    ResearchLineageLink,
    ResearchLineageRegistry,
    RobustnessSummary,
    ValidationSummary,
    WinnerSelectionPolicy,
    WinnerSelectionResult,
)

_FACTOR_FAMILY_RULES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("mom", "momentum", "rsi", "trend"), "momentum"),
    (("reversal", "rev", "mean_revert"), "reversal"),
    (("bp", "pb", "pe", "valuation", "value"), "valuation"),
    (("quality", "roe", "roa", "profit"), "quality"),
    (("growth", "sales_growth", "earnings_growth"), "growth"),
    (("sentiment", "news", "attention"), "sentiment"),
    (("micro", "order", "quote", "book"), "microstructure"),
    (("liquidity", "turnover", "illiq"), "liquidity / turnover"),
    (("vol", "volatility", "beta", "risk"), "volatility / risk"),
    (("analyst", "estimate", "expect"), "analyst / expectation"),
    (("alt", "satellite", "web", "credit"), "alternative data"),
)

_STATUS_ORDER: dict[str, int] = {
    "portfolio-active": 0,
    "validated": 1,
    "generated": 2,
    "draft": 3,
    "rejected": 4,
    "archived": 5,
}

_SHORTLIST_SCORE_RANGES: dict[str, tuple[float, float]] = {
    "ic_mean": (0.0, 0.08),
    "rank_ic_mean": (0.0, 0.12),
    "icir": (0.0, 1.20),
    "monotonicity_share": (0.40, 1.00),
    "turnover_efficiency": (0.0, 1.00),
    "oos_stability_share": (0.40, 1.00),
}

_DEFAULT_SHORTLIST_CONFIG = FactorShortlistConfig()
_DEFAULT_FACTOR_SET_CONFIG = FactorSetConstructionConfig()
_DEFAULT_CANDIDATE_RECIPE_CONFIG = CandidateRecipeGenerationConfig()
_DEFAULT_WINNER_SELECTION_POLICY = WinnerSelectionPolicy()
_NEXT_STEP_POLICY_ID = "next_step_policy_v1"
_NEXT_STEP_POLICY_FORMULA = (
    "recommendations = deterministic_rules(shortlist, factor_sets, candidate_recipes, "
    "recipe_comparison, winner_selection)"
)
_DEFAULT_ARTIFACT_LOAD_MODE = "permissive"
_MISSING_CANONICAL_ARTIFACT_CODE = "MISSING_CANONICAL_ARTIFACT"
_INVALID_CANONICAL_ARTIFACT_CODE = "INVALID_CANONICAL_ARTIFACT"
_MISSING_WORKFLOW_ARTIFACT_CODE = "MISSING_WORKFLOW_ARTIFACT"
_INVALID_WORKFLOW_ARTIFACT_CODE = "INVALID_WORKFLOW_ARTIFACT"
_FALLBACK_USED_CODE = "FALLBACK_USED"
_STRICT_LOAD_ABORTED_CODE = "STRICT_LOAD_ABORTED"

_RECIPE_LEADERBOARD_OBJECTIVES: tuple[tuple[str, str], ...] = (
    ("Sharpe", "sharpe"),
    ("Annualized Return", "annualized_return"),
    ("Max Drawdown", "max_drawdown"),
    ("Information Ratio", "information_ratio"),
    ("Post-cost Return", "post_cost_return"),
)

_FACTOR_SET_STATUS_ORDER: dict[str, int] = {
    "selected": 0,
    "candidate": 1,
    "watchlist": 2,
    "rejected": 3,
}

_CANDIDATE_RECIPE_VARIANTS: tuple[tuple[str, str, str, str, str], ...] = (
    (
        "baseline_rank_neutralized_strict",
        "rank",
        "neutralization_on",
        "strict",
        "benchmark_relative",
    ),
    (
        "alpha_rank_unneutralized_balanced",
        "rank",
        "neutralization_off",
        "balanced",
        "absolute",
    ),
    (
        "diversified_equal_neutralized_strict",
        "equal_weight",
        "neutralization_on",
        "strict",
        "absolute",
    ),
)

_UI_LABELS: dict[str, str] = {
    "na": "N/A",
    "section_overview": "A. 研究首页总览 (Research Home / Overview)",
    "section_factor_library": "B. 因子库 (Factor Library)",
    "section_factor_detail": "C. 因子详情页 (Factor Detail)",
    "section_cross_factor": "D. 因子横向对比 (Cross-Factor Comparison)",
    "section_selected_factor_sets": "D2. 入选因子集合 (Selected Factor Sets)",
    "section_candidate_recipe_generation": "D3. 候选配方生成 (Candidate Recipe Generation)",
    "section_portfolio_construction": "E. 组合构建 (Portfolio Construction)",
    "section_winner_selection": "E2. 冠军方案选择 (Winner Selection)",
    "section_next_step_recommendations": "E3. 下一步建议 (Next-Step Recommendations)",
    "section_backtest_evaluation": "F. 回测评估 (Backtest Evaluation)",
    "section_lineage_registry": "F2. 研究血缘与实验登记 (Research Lineage / Registry)",
    "section_robustness": "G. 稳健性与审计（次级） (Robustness / Audit, Secondary)",
    "nav_overview": "研究首页 (Research Home)",
    "nav_factor_library": "因子库 (Factor Library)",
    "nav_factor_detail": "因子详情 (Factor Detail)",
    "nav_cross_factor": "因子对比 (Cross-Factor)",
    "nav_selected_factor_sets": "因子集合 (Factor Sets)",
    "nav_candidate_recipe_generation": "候选配方 (Candidate Recipes)",
    "nav_portfolio_construction": "组合构建 (Portfolio)",
    "nav_winner_selection": "冠军选择 (Winner)",
    "nav_next_step_recommendations": "下一步建议 (Next Actions)",
    "nav_backtest_evaluation": "回测评估 (Backtest)",
    "nav_lineage_registry": "血缘登记 (Lineage)",
    "nav_robustness": "稳健性审计 (Robustness)",
    "card_candidate_factors": "候选因子数 (Candidate Factors)",
    "card_validated_factors": "已验证因子数 (Validated Factors)",
    "card_active_portfolio_recipes": "活跃组合配方数 (Active Portfolio Recipes)",
    "card_completed_backtests": "已完成回测数 (Completed Backtests)",
}

_STATUS_LABELS: dict[str, str] = {
    "portfolio-active": "组合活跃 (portfolio-active)",
    "validated": "已验证 (validated)",
    "generated": "已生成 (generated)",
    "draft": "草稿 (draft)",
    "rejected": "已淘汰 (rejected)",
    "archived": "已归档 (archived)",
}

_SHORTLIST_RECOMMENDATION_LABELS: dict[str, str] = {
    "keep": "保留 (keep)",
    "watchlist": "观察 (watchlist)",
    "drop": "剔除 (drop)",
    "rejected": "淘汰 (rejected)",
}

_FACTOR_SET_STATUS_LABELS: dict[str, str] = {
    "selected": "入选 (selected)",
    "candidate": "候选 (candidate)",
    "watchlist": "观察 (watchlist)",
    "rejected": "淘汰 (rejected)",
}

_PRIORITY_LABELS: dict[str, str] = {
    "P1": "高优先级 (P1)",
    "P2": "中优先级 (P2)",
    "P3": "低优先级 (P3)",
}

_NEXT_STEP_CATEGORY_LABELS: dict[str, str] = {
    "promotion": "晋级推进 (promotion)",
    "stress_test": "压力测试 (stress_test)",
    "factor_pruning": "因子裁剪 (factor_pruning)",
    "recipe_generation": "配方生成 (recipe_generation)",
    "turnover_sensitivity": "换手率敏感性 (turnover_sensitivity)",
    "evidence_gap": "证据缺口 (evidence_gap)",
    "archival": "归档处理 (archival)",
}

_DISPLAY_TEXT_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    ("IC mean", "信息系数均值 (IC Mean)"),
    ("Rank IC mean", "秩信息系数均值 (Rank IC Mean)"),
    ("RankIC", "秩信息系数 (Rank IC)"),
    ("Max Drawdown", "最大回撤 (Max Drawdown)"),
    ("Annualized Return", "年化收益 (Annualized Return)"),
    ("Information Ratio", "信息比率 (Information Ratio)"),
    ("Post-cost Return", "成本后收益 (Post-cost Return)"),
    ("Turnover", "换手率 (Turnover)"),
    ("OOS stability", "样本外稳定性 (OOS Stability)"),
    ("OOS Stability", "样本外稳定性 (OOS Stability)"),
    ("AnnRet", "年化收益 (AnnRet)"),
    ("selected as winner", "被选为冠军方案 (selected as winner)"),
    ("winner tradeoff", "冠军权衡 (winner tradeoff)"),
    ("winner outranks top challenger", "冠军得分高于首位挑战者 (winner outranks top challenger)"),
    (
        "no winner: no recipe met minimum Sharpe/post-cost/drawdown guardrails",
        "无冠军方案：没有配方满足最低 Sharpe/成本后收益/回撤护栏 (no winner)",
    ),
    ("challenger:", "挑战者 (challenger):"),
    ("rejected:", "淘汰 (rejected):"),
    ("promote winner", "推进冠军方案 (promote winner)"),
    (
        "keep challengers under active comparison",
        "保持挑战者并行比较 (keep challengers under active comparison)",
    ),
    (
        "watchlist recipes need more evidence",
        "观察名单配方需要更多证据 (watchlist recipes need more evidence)",
    ),
    ("archive weak candidates", "归档弱候选方案 (archive weak candidates)"),
    ("promote ", "推进 (promote) "),
    (
        "run stricter neutralization stress test",
        "执行更严格中性化压力测试 (run stricter neutralization stress test)",
    ),
    ("drop redundant factor", "移除冗余因子 (drop redundant factor)"),
    (
        "generate additional recipes from factor set",
        "从因子集合生成更多配方 (generate additional recipes from factor set)",
    ),
    ("inspect turnover sensitivity for", "检查换手率敏感性 (inspect turnover sensitivity for)"),
    (
        "collect more history before trusting watchlist candidates",
        "补充更长历史后再信任观察名单候选 (collect more history before trusting watchlist candidates)",
    ),
    (
        "archive weak candidates and keep registry lineage for audit",
        "归档弱候选并保留登记血缘用于审计 (archive weak candidates and keep registry lineage for audit)",
    ),
)


def _ui(key: str) -> str:
    return _UI_LABELS.get(key, key)


def _display_enum(value: str | None, mapping: dict[str, str]) -> str:
    text = safe_text(value)
    if text is None:
        return _ui("na")
    return mapping.get(text, text)


def _display_status(value: str | None) -> str:
    return _display_enum(value, _STATUS_LABELS)


def _display_shortlist_recommendation(value: str | None) -> str:
    return _display_enum(value, _SHORTLIST_RECOMMENDATION_LABELS)


def _display_factor_set_status(value: str | None) -> str:
    return _display_enum(value, _FACTOR_SET_STATUS_LABELS)


def _display_priority(value: str | None) -> str:
    return _display_enum(value, _PRIORITY_LABELS)


def _display_next_step_category(value: str | None) -> str:
    return _display_enum(value, _NEXT_STEP_CATEGORY_LABELS)


def _display_text(value: str | None) -> str:
    text = safe_text(value)
    if text is None:
        return _ui("na")
    out = text
    for source, target in _DISPLAY_TEXT_REPLACEMENTS:
        out = out.replace(source, target)
    return out


def _display_name_with_zh(*, english: str, zh: str | None) -> str:
    zh_text = safe_text(zh)
    if zh_text is None:
        return english
    if zh_text == english:
        return english
    return f"{zh_text} ({english})"


def _display_desc_with_zh(*, english: str, zh: str | None) -> str:
    zh_text = safe_text(zh)
    if zh_text is None:
        return english
    if zh_text == english:
        return english
    return f"{zh_text} ({english})"


def _preferred_lines(
    zh_lines: tuple[str, ...] | list[str],
    fallback_lines: tuple[str, ...] | list[str],
) -> list[str]:
    if any(safe_text(item) for item in zh_lines):
        return list(zh_lines)
    return list(fallback_lines)


@dataclass(frozen=True)
class _CaseArtifacts:
    output_dir: Path | None
    metrics_payload: dict[str, object]
    metrics: dict[str, object]
    factor_definition_payload: dict[str, object]
    signal_validation_payload: dict[str, object]
    portfolio_recipe_payload: dict[str, object]
    backtest_result_payload: dict[str, object]
    fallback_derived_fields: dict[str, tuple[str, ...]]
    portfolio_validation_summary: dict[str, object]
    portfolio_validation_metrics: dict[str, object]
    portfolio_validation_package: dict[str, object]
    manifest: dict[str, object]
    integrity_report: dict[str, object]
    coverage_df: pd.DataFrame | None
    group_returns_df: pd.DataFrame | None
    ic_df: pd.DataFrame | None
    turnover_df: pd.DataFrame | None
    rolling_df: pd.DataFrame | None
    factor_series: pd.Series | None
    artifact_paths: dict[str, Path]


ArtifactLoadMode = Literal["permissive", "strict"]


@dataclass(frozen=True)
class ArtifactLoadPolicy:
    mode: ArtifactLoadMode
    require_canonical_artifacts: bool
    require_workflow_closure_artifacts: bool
    allow_legacy_case_fallback: bool
    allow_workflow_fallback: bool
    prefer_persisted_workflow_artifacts: bool
    required_canonical_objects: tuple[str, ...]
    required_workflow_closure_objects: tuple[str, ...]


def _normalize_artifact_load_mode(mode: str | None) -> ArtifactLoadMode:
    normalized = (safe_text(mode) or "").lower() or _DEFAULT_ARTIFACT_LOAD_MODE
    if normalized not in {"permissive", "strict"}:
        raise ValueError(f"artifact_load_mode must be 'permissive' or 'strict'; received {mode!r}")
    return cast(ArtifactLoadMode, normalized)


def _build_artifact_load_policy(
    *,
    artifact_load_mode: str | None,
    prefer_persisted_workflow_artifacts: bool,
) -> ArtifactLoadPolicy:
    mode = _normalize_artifact_load_mode(artifact_load_mode)
    canonical_objects = tuple(row[0] for row in _CANONICAL_ARTIFACT_REQUIREMENTS)
    workflow_objects = tuple(row[0] for row in _WORKFLOW_CLOSURE_ARTIFACT_REQUIREMENTS)
    if mode == "strict":
        return ArtifactLoadPolicy(
            mode=mode,
            require_canonical_artifacts=True,
            require_workflow_closure_artifacts=True,
            allow_legacy_case_fallback=False,
            allow_workflow_fallback=False,
            prefer_persisted_workflow_artifacts=True,
            required_canonical_objects=canonical_objects,
            required_workflow_closure_objects=workflow_objects,
        )
    return ArtifactLoadPolicy(
        mode=mode,
        require_canonical_artifacts=False,
        require_workflow_closure_artifacts=False,
        allow_legacy_case_fallback=True,
        allow_workflow_fallback=True,
        prefer_persisted_workflow_artifacts=bool(prefer_persisted_workflow_artifacts),
        required_canonical_objects=canonical_objects,
        required_workflow_closure_objects=workflow_objects,
    )


def _artifact_load_policy_summary(policy: ArtifactLoadPolicy) -> tuple[str, ...]:
    return (
        f"mode={policy.mode}",
        (f"require_canonical_artifacts={'yes' if policy.require_canonical_artifacts else 'no'}"),
        (
            "require_workflow_closure_artifacts="
            f"{'yes' if policy.require_workflow_closure_artifacts else 'no'}"
        ),
        f"allow_legacy_case_fallback={'yes' if policy.allow_legacy_case_fallback else 'no'}",
        f"allow_workflow_fallback={'yes' if policy.allow_workflow_fallback else 'no'}",
        (
            "prefer_persisted_workflow_artifacts="
            f"{'yes' if policy.prefer_persisted_workflow_artifacts else 'no'}"
        ),
        "required_canonical_objects=" + ", ".join(policy.required_canonical_objects),
        "required_workflow_closure_objects=" + ", ".join(policy.required_workflow_closure_objects),
    )


class ArtifactLoadRuntimeError(RuntimeError):
    """Strict-mode artifact loading failure with structured diagnostic context."""

    def __init__(
        self,
        message: str,
        *,
        diagnostics: tuple[ArtifactLoadDiagnostic, ...],
    ) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics


def _append_artifact_issue(
    *,
    code: str,
    severity: Literal["warning", "error"],
    artifact_type: str,
    object_scope: str,
    message: str,
    diagnostics: list[ArtifactLoadDiagnostic],
    warnings: list[str],
    errors: list[str],
    mode: ArtifactLoadMode,
    path: Path | None = None,
    case_name: str | None = None,
    profile_name: str | None = None,
    fallback_used: bool = False,
    remediation_hint: str | None = None,
) -> ArtifactLoadDiagnostic:
    diagnostic = ArtifactLoadDiagnostic(
        code=code,
        severity=severity,
        artifact_type=artifact_type,
        object_scope=object_scope,
        message=message,
        path=str(path) if path is not None else None,
        case_name=case_name,
        profile_name=profile_name,
        mode=mode,
        fallback_used=fallback_used,
        remediation_hint=remediation_hint,
    )
    diagnostics.append(diagnostic)
    line = _artifact_diagnostic_to_text(diagnostic)
    if severity == "error":
        errors.append(line)
        return diagnostic
    warnings.append(line)
    return diagnostic


def _artifact_diagnostic_to_text(diagnostic: ArtifactLoadDiagnostic) -> str:
    return diagnostic.message


def _artifact_issue_severity(
    *,
    required: bool,
) -> Literal["warning", "error"]:
    return "error" if required else "warning"


def render_campaign_profile_dashboard_html(
    comparison_json_path: str | Path,
    *,
    title: str | None = None,
    artifact_load_mode: str = _DEFAULT_ARTIFACT_LOAD_MODE,
) -> str:
    """Render the factor-first local research dashboard HTML."""

    comparison_path = Path(comparison_json_path).resolve()
    data = _build_research_dashboard_data(
        comparison_path,
        artifact_load_mode=artifact_load_mode,
    )
    report_title = title or "Factor Research Workbench 因子研究工作台"
    return _render_campaign_profile_dashboard_html_content(
        data=data,
        report_title=report_title,
    )


def _render_campaign_profile_dashboard_html_content(
    *,
    data: ResearchDashboardData,
    report_title: str,
) -> str:
    subtitle = (
        "因子发现 → 信号验证 → 组合构建 → 回测评估；稳健性/审计位于下游模块。 "
        "(Factor Discovery → Signal Validation → Portfolio Construction → Backtest Evaluation; "
        "Robustness/Audit remains a downstream module.)"
    )

    return (
        "<!doctype html>\n"
        '<html lang="zh-CN">\n'
        "<head>\n"
        '  <meta charset="utf-8" />\n'
        '  <meta name="viewport" content="width=device-width, initial-scale=1" />\n'
        f"  <title>{_h(report_title)}</title>\n"
        "  <style>\n"
        f"{_dashboard_css()}\n"
        "  </style>\n"
        "</head>\n"
        "<body>\n"
        '  <div class="page-backdrop"></div>\n'
        '  <div class="page-wrap">\n'
        '    <header class="hero">\n'
        f"      <h1>{_h(report_title)}</h1>\n"
        f'      <p class="subtitle">{_h(subtitle)}</p>\n'
        '      <p class="meta-line">'
        f"默认Profile (Default profile): <code>{_h(data.default_profile)}</code> | "
        f"对比JSON (Comparison JSON): <code>{_h(data.source_json_path)}</code>"
        "</p>\n"
        '      <p class="meta-line">'
        f"生成时间 (Generated at, UTC): <code>{_h(data.generated_at_utc or _ui('na'))}</code>"
        "</p>\n"
        '      <p class="meta-line">'
        f"工件加载模式 (Artifact load mode): <code>{_h(data.artifact_load_mode)}</code>"
        "</p>\n"
        "    </header>\n"
        f"{_render_quick_nav()}\n"
        f'    <section id="overview" class="panel">{_render_overview(data.overview)}</section>\n'
        f'    <section id="factor-library" class="panel">{_render_factor_library(data.factor_summaries)}</section>\n'
        f'    <section id="factor-detail" class="panel">{_render_factor_detail(data.factor_details)}</section>\n'
        f'    <section id="cross-factor" class="panel">{_render_cross_factor(data)}</section>\n'
        f'    <section id="selected-factor-sets" class="panel">{_render_selected_factor_sets(data.factor_sets)}</section>\n'
        f'    <section id="candidate-recipe-generation" class="panel">{_render_candidate_recipe_generation(data.candidate_recipe_generation)}</section>\n'
        f'    <section id="portfolio-construction" class="panel">{_render_portfolio_construction(data.portfolio_recipes, data.recipe_comparison)}</section>\n'
        f'    <section id="winner-selection" class="panel">{_render_winner_selection(data.winner_selection)}</section>\n'
        f'    <section id="next-step-recommendations" class="panel">{_render_next_step_recommendations(data.next_step_recommendations)}</section>\n'
        f'    <section id="backtest-evaluation" class="panel">{_render_backtest_evaluation(data.backtests)}</section>\n'
        f'    <section id="lineage-registry" class="panel">{_render_lineage_registry(data.lineage_registry)}</section>\n'
        f'    <section id="robustness-audit" class="panel panel-secondary">{_render_robustness(data.robustness_summaries)}</section>\n'
        "  </div>\n"
        "  <script>\n"
        f"{_dashboard_js()}\n"
        "  </script>\n"
        "</body>\n"
        "</html>\n"
    )


def write_campaign_profile_dashboard_html(
    comparison_json_path: str | Path,
    *,
    output_path: str | Path | None = None,
    overwrite: bool = False,
    title: str | None = None,
    artifact_load_mode: str = _DEFAULT_ARTIFACT_LOAD_MODE,
) -> Path:
    """Write the factor-first research dashboard HTML next to comparison artifacts."""

    comparison_path = Path(comparison_json_path).resolve()
    out_path = (
        Path(output_path).resolve()
        if output_path is not None
        else comparison_path.with_name("campaign_profile_dashboard_zh.html")
    )
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"{out_path} already exists. Pass overwrite=True to replace it.")
    data = _build_research_dashboard_data(
        comparison_path,
        artifact_load_mode=artifact_load_mode,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report_title = title or "Factor Research Workbench 因子研究工作台"
    out_path.write_text(
        _render_campaign_profile_dashboard_html_content(
            data=data,
            report_title=report_title,
        ),
        encoding="utf-8",
    )
    _persist_dashboard_workflow_artifacts(
        comparison_path=comparison_path,
        data=data,
        generated_at_utc=datetime.now(UTC).isoformat(),
        source_artifacts={
            "campaign_profile_comparison_json_path": str(comparison_path),
            "dashboard_html_path": str(out_path),
        },
    )
    return out_path


def persist_workflow_closure_artifacts(
    comparison_json_path: str | Path,
) -> dict[str, Path]:
    """Persist workflow-closure outputs as canonical artifact JSONs."""

    comparison_path = Path(comparison_json_path).resolve()
    data = _build_research_dashboard_data(
        comparison_path,
        artifact_load_mode="permissive",
        prefer_persisted_workflow_artifacts=False,
    )
    return _persist_workflow_closure_artifacts(
        comparison_path=comparison_path,
        data=data,
        winner_selection_policy=_DEFAULT_WINNER_SELECTION_POLICY,
        winner_policy_formula_text=_winner_policy_formula_text(_DEFAULT_WINNER_SELECTION_POLICY),
        generated_at_utc=datetime.now(UTC).isoformat(),
        source_artifacts={
            "campaign_profile_comparison_json_path": str(comparison_path),
            "factor_shortlist_reference": "campaign_profile_comparison.json#case_comparison",
        },
    )


def _build_research_dashboard_data(
    comparison_path: Path,
    *,
    artifact_load_mode: str | None = _DEFAULT_ARTIFACT_LOAD_MODE,
    prefer_persisted_workflow_artifacts: bool = True,
) -> ResearchDashboardData:
    load_policy = _build_artifact_load_policy(
        artifact_load_mode=artifact_load_mode,
        prefer_persisted_workflow_artifacts=prefer_persisted_workflow_artifacts,
    )
    artifact_diagnostics: list[ArtifactLoadDiagnostic] = []
    artifact_warnings: list[str] = []
    artifact_errors: list[str] = []
    payload = _load_json(comparison_path)
    workflow_artifact_paths = _workflow_closure_artifact_paths_from_payload(
        payload=payload,
        comparison_path=comparison_path,
        load_policy=load_policy,
        diagnostics=artifact_diagnostics,
        warnings=artifact_warnings,
        errors=artifact_errors,
    )
    profiles = [str(item) for item in as_object_list(payload.get("profiles")) if str(item)]
    case_rows = as_object_list(payload.get("case_comparison"))
    default_profile = safe_text(payload.get("default_profile")) or (
        profiles[0] if profiles else "default_research"
    )
    generated_at_utc = safe_text(payload.get("generated_at_utc"))

    factor_summaries: list[FactorSummary] = []
    factor_details: list[FactorDetail] = []
    comparison_rows: list[FactorComparisonRow] = []
    recipes: list[PortfolioRecipeSummary] = []
    backtests: list[PortfolioBacktestSummary] = []
    robustness_rows: list[RobustnessSummary] = []
    lineage_entries: list[ExperimentRegistryEntry] = []
    lineage_links: list[ResearchLineageLink] = []
    warnings: list[str] = []
    recent_runs: list[tuple[str, str]] = []
    factor_series_lookup: dict[str, pd.Series] = {}

    for raw_row in case_rows:
        row = as_object_dict(raw_row)
        case_name = safe_text(row.get("case_name")) or "unknown_case"
        case_description = safe_text(row.get("case_description")) or "N/A"
        case_description_zh = safe_text(row.get("case_description_zh"))
        profiles_payload = as_object_dict(row.get("profiles"))
        profile_name, profile_payload = _select_profile_payload(
            profiles_payload,
            preferred_profile=default_profile,
            ordered_profiles=profiles,
        )
        artifacts = _load_case_artifacts(
            profile_payload,
            case_name=case_name,
            profile_name=profile_name or "unknown_profile",
            load_policy=load_policy,
            diagnostics=artifact_diagnostics,
            warnings=artifact_warnings,
            errors=artifact_errors,
        )

        if profile_name is None:
            warnings.append(f"{case_name}: missing profile payload")
            continue

        run_timestamp = safe_text(artifacts.manifest.get("run_timestamp_utc"))
        if run_timestamp:
            recent_runs.append((run_timestamp, case_name))

        metrics = artifacts.metrics
        spec = as_object_dict(artifacts.factor_definition_payload.get("spec")) or as_object_dict(
            artifacts.manifest.get("spec")
        )
        target = as_object_dict(spec.get("target"))
        preprocess = as_object_dict(spec.get("preprocess"))
        neutralization = as_object_dict(spec.get("neutralization"))
        universe = as_object_dict(spec.get("universe"))

        factor_name = (
            safe_text(artifacts.factor_definition_payload.get("factor_name"))
            or safe_text(metrics.get("factor_name"))
            or safe_text(spec.get("factor_name"))
            or case_name
        )
        factor_display_name_zh = (
            safe_text(artifacts.factor_definition_payload.get("display_name_zh"))
            or safe_text(artifacts.factor_definition_payload.get("factor_name_zh"))
            or safe_text(spec.get("factor_name_zh"))
            or safe_text(metrics.get("factor_name_zh"))
        )
        short_description_zh = (
            case_description_zh
            or safe_text(artifacts.factor_definition_payload.get("short_description_zh"))
            or safe_text(metrics.get("short_description_zh"))
        )
        factor_family = _infer_factor_family(factor_name=factor_name, case_name=case_name)

        rank_ic_mean = _safe_float(metrics.get("mean_rank_ic"))
        icir = _safe_float(metrics.get("ic_ir"))
        long_short_ir = _safe_float(metrics.get("long_short_ir"))
        coverage_ratio = _safe_float(metrics.get("coverage_mean"))
        if coverage_ratio is None:
            coverage_ratio = _safe_float(
                as_object_dict(artifacts.metrics_payload.get("coverage_by_date_summary")).get(
                    "mean_coverage"
                )
            )
        missingness_mean = _safe_float(metrics.get("missingness_mean"))

        signal_quality_score = _signal_quality_score(
            icir=icir,
            rank_ic_mean=rank_ic_mean,
            long_short_ir=long_short_ir,
            coverage_ratio=coverage_ratio,
        )

        factor_status = _factor_status(
            profile_status=safe_text(profile_payload.get("status")) or "unknown",
            factor_verdict=safe_text(profile_payload.get("factor_verdict")) or "N/A",
            promotion_decision=safe_text(profile_payload.get("promotion_decision")) or "N/A",
            portfolio_recommendation=safe_text(
                profile_payload.get("portfolio_validation_recommendation")
            )
            or "N/A",
        )

        required_inputs = _required_input_fields(spec=spec)
        lookback_parameters = _lookback_parameters(
            target_horizon=_safe_float(target.get("horizon")),
            preprocess=preprocess,
        )
        lag_delay_rule = _lag_rule(target=target)
        expected_sign = _expected_sign(safe_text(spec.get("direction")))
        economic_intuition = _economic_intuition(
            factor_family=factor_family, expected_sign=expected_sign
        )

        summary = FactorSummary(
            factor_id=case_name,
            factor_name=factor_name,
            display_name_zh=factor_display_name_zh,
            short_description=case_description,
            short_description_zh=short_description_zh,
            factor_family=factor_family,
            mathematical_definition=_formal_definition_text(
                factor_name=factor_name,
                target=target,
                spec=spec,
            ),
            required_input_fields=required_inputs,
            frequency=safe_text(spec.get("rebalance_frequency"))
            or safe_text(metrics.get("rebalance_frequency"))
            or "N/A",
            lookback_parameters=lookback_parameters,
            lag_delay_rule=lag_delay_rule,
            expected_sign=expected_sign,
            economic_intuition=economic_intuition,
            coverage_ratio=coverage_ratio,
            missingness_summary=_missingness_summary(missingness_mean),
            last_updated_time=run_timestamp,
            research_status=factor_status,
            signal_quality_score=signal_quality_score,
        )
        factor_summaries.append(summary)

        validation = _build_validation_summary(
            metrics=metrics,
            case_row=row,
            artifacts=artifacts,
        )
        concise_verdict = _concise_verdict(profile_payload=profile_payload)
        concise_verdict_zh = safe_text(profile_payload.get("concise_verdict_zh")) or _display_text(
            concise_verdict
        )

        detail = FactorDetail(
            summary=summary,
            formal_definition=summary.mathematical_definition,
            implementation_notes=_implementation_notes(spec=spec),
            pit_anti_lookahead_notes=_pit_notes(artifacts.integrity_report),
            data_dependencies=required_inputs,
            parameter_settings=_parameter_settings(spec=spec),
            intended_holding_horizon=_holding_horizon_text(target=target),
            coverage_over_time=_coverage_over_time_text(artifacts.coverage_df),
            cross_sectional_coverage=_cross_sectional_coverage(
                metrics=metrics, coverage_df=artifacts.coverage_df
            ),
            missingness=summary.missingness_summary,
            winsorization_clipping_summary=_winsorization_summary(preprocess),
            standardization_neutralization_summary=_standardization_summary(
                preprocess=preprocess,
                neutralization=neutralization,
            ),
            distribution_snapshots=_distribution_snapshots(artifacts.factor_series),
            turnover_of_factor_values=_turnover_snapshot(artifacts.turnover_df),
            stability_over_time=_stability_text(metrics=metrics, rolling_df=artifacts.rolling_df),
            validation=validation,
            concise_verdict=concise_verdict,
            concise_verdict_zh=concise_verdict_zh,
            strengths=tuple(
                _to_reason_list(
                    profile_payload, key="major_reasons", nested_key="factor_verdict_reasons"
                )
            ),
            weaknesses=tuple(
                _to_reason_list(
                    profile_payload,
                    key="major_reasons",
                    nested_key="portfolio_validation_major_risks",
                )
            ),
            likely_failure_modes=tuple(
                _portfolio_fragility_reasons(artifacts.portfolio_validation_summary)
            ),
            proceed_to_portfolio_layer=factor_status in {"validated", "portfolio-active"},
            related_artifacts=_related_artifacts(artifacts),
        )
        factor_details.append(detail)

        comparison_rows.append(
            FactorComparisonRow(
                factor_id=summary.factor_id,
                factor_name=summary.factor_name,
                factor_family=summary.factor_family,
                ic_mean=validation.ic_mean,
                rank_ic_mean=validation.rank_ic_mean,
                icir=validation.icir,
                turnover=_safe_float(metrics.get("mean_long_short_turnover")),
                coverage=summary.coverage_ratio,
                long_short_return=_safe_float(metrics.get("mean_long_short_return")),
                monotonicity_share=_monotonicity_share(artifacts.group_returns_df),
                oos_stability_share=_oos_stability_share(metrics),
                monotonicity=validation.monotonicity_diagnostics,
                oos_stability=validation.oos_stability_comparison,
            )
        )

        recipe = _build_portfolio_recipe(
            case_name=case_name,
            factor_name=factor_name,
            profile_payload=profile_payload,
            metrics=metrics,
            portfolio_recipe_payload=artifacts.portfolio_recipe_payload,
            portfolio_summary=artifacts.portfolio_validation_summary,
            portfolio_metrics=artifacts.portfolio_validation_metrics,
            spec=spec,
            universe=universe,
            neutralization=neutralization,
        )
        recipes.append(recipe)

        backtest = _build_backtest_summary(
            recipe_id=recipe.recipe_id,
            factor_id=case_name,
            metrics=metrics,
            portfolio_summary=artifacts.portfolio_validation_summary,
            portfolio_metrics=artifacts.portfolio_validation_metrics,
            backtest_payload=artifacts.backtest_result_payload,
            group_returns_df=artifacts.group_returns_df,
            turnover_df=artifacts.turnover_df,
            rebalance_frequency=summary.frequency,
        )
        backtests.append(backtest)

        lineage_entry = _build_registry_entry(
            case_name=case_name,
            profile_name=profile_name,
            run_timestamp=run_timestamp,
            factor_id=summary.factor_id,
            recipe_id=recipe.recipe_id,
            artifacts=artifacts,
        )
        lineage_entries.append(lineage_entry)
        lineage_links.extend(
            _build_lineage_links(
                profile_name=profile_name,
                factor_id=summary.factor_id,
                recipe_id=recipe.recipe_id,
            )
        )

        robustness_rows.append(
            _build_robustness_summary(
                case_name=case_name,
                case_row=row,
                artifacts=artifacts,
                profile_payload=profile_payload,
                portfolio_summary=artifacts.portfolio_validation_summary,
                portfolio_metrics=artifacts.portfolio_validation_metrics,
            )
        )

        if artifacts.factor_series is not None and not artifacts.factor_series.empty:
            factor_series_lookup[case_name] = artifacts.factor_series

        warnings.extend(
            _collect_case_warnings(
                case_name=case_name,
                profile_payload=profile_payload,
                artifacts=artifacts,
                metrics=metrics,
            )
        )

    factor_summaries.sort(key=_factor_sort_key)
    factor_details.sort(key=lambda item: _factor_sort_key(item.summary))
    comparison_rows.sort(
        key=lambda row: (
            -_float_or_default(row.icir, default=-999.0),
            -_float_or_default(row.long_short_return, default=-999.0),
            row.factor_name,
        )
    )

    recipe_lookup = {recipe.recipe_id: recipe for recipe in recipes}
    recipes.sort(key=lambda item: item.recipe_name)
    backtests.sort(
        key=lambda item: (
            -_float_or_default(item.sharpe, default=-999.0),
            -_float_or_default(item.annualized_return, default=-999.0),
            item.factor_id,
        )
    )
    robustness_rows.sort(key=lambda item: item.factor_id)

    correlation_matrix = _factor_correlation_matrix(
        summaries=factor_summaries,
        factor_series_lookup=factor_series_lookup,
    )
    family_summary = _factor_family_summary(factor_summaries)
    shortlist_result = _build_factor_shortlist_result(
        comparison_rows=comparison_rows,
        correlation_matrix=correlation_matrix,
        config=_DEFAULT_SHORTLIST_CONFIG,
    )
    shortlist = shortlist_result.recommendation_summary
    fallback_factor_set_result = _build_factor_set_result(
        shortlist=shortlist_result,
        comparison_rows=comparison_rows,
        correlation_matrix=correlation_matrix,
        config=_DEFAULT_FACTOR_SET_CONFIG,
    )
    fallback_candidate_recipe_generation = _build_candidate_recipe_generation_result(
        factor_sets=fallback_factor_set_result.factor_sets,
        config=_DEFAULT_CANDIDATE_RECIPE_CONFIG,
    )
    factor_set_result = fallback_factor_set_result
    candidate_recipe_generation = fallback_candidate_recipe_generation
    if load_policy.prefer_persisted_workflow_artifacts:
        factor_set_result = _load_factor_set_result_artifact(
            workflow_artifact_paths.get("factor_set_result_json_path"),
            fallback=fallback_factor_set_result,
            load_policy=load_policy,
            diagnostics=artifact_diagnostics,
            warnings=artifact_warnings,
            errors=artifact_errors,
        )
        candidate_recipe_generation = _load_candidate_recipe_generation_artifact(
            workflow_artifact_paths.get("candidate_recipe_generation_json_path"),
            fallback=fallback_candidate_recipe_generation,
            load_policy=load_policy,
            diagnostics=artifact_diagnostics,
            warnings=artifact_warnings,
            errors=artifact_errors,
        )
    candidate_recipe_summaries = _candidate_recipes_to_portfolio_summaries(
        generated=candidate_recipe_generation.generated_recipes,
        factor_sets=factor_set_result.factor_sets,
    )
    recipe_comparison = _build_recipe_comparison_view(
        recipes=[*recipes, *candidate_recipe_summaries],
        backtests=backtests,
        factor_summaries=factor_summaries,
    )
    fallback_winner_selection = _build_winner_selection_result(
        recipe_comparison=recipe_comparison,
        factor_sets=factor_set_result,
        candidate_recipe_generation=candidate_recipe_generation,
        policy=_DEFAULT_WINNER_SELECTION_POLICY,
    )
    fallback_next_step_recommendations = _build_next_step_recommendations(
        shortlist=shortlist_result,
        factor_sets=factor_set_result,
        candidate_recipe_generation=candidate_recipe_generation,
        recipe_comparison=recipe_comparison,
        winner_selection=fallback_winner_selection,
    )
    winner_selection = fallback_winner_selection
    next_step_recommendations = fallback_next_step_recommendations
    if load_policy.prefer_persisted_workflow_artifacts:
        winner_selection = _load_winner_selection_artifact(
            workflow_artifact_paths.get("winner_selection_json_path"),
            fallback=fallback_winner_selection,
            load_policy=load_policy,
            diagnostics=artifact_diagnostics,
            warnings=artifact_warnings,
            errors=artifact_errors,
        )
        next_step_recommendations = _load_next_step_recommendations_artifact(
            workflow_artifact_paths.get("next_step_recommendations_json_path"),
            fallback=fallback_next_step_recommendations,
            load_policy=load_policy,
            diagnostics=artifact_diagnostics,
            warnings=artifact_warnings,
            errors=artifact_errors,
        )
    if artifact_errors:
        _append_artifact_issue(
            code=_STRICT_LOAD_ABORTED_CODE,
            severity="error",
            artifact_type="artifact_load_runtime",
            object_scope="artifact_load",
            message="strict artifact load checks failed; aborting dashboard data build",
            diagnostics=artifact_diagnostics,
            warnings=artifact_warnings,
            errors=artifact_errors,
            mode=load_policy.mode,
            remediation_hint=(
                "Ensure required canonical/workflow artifacts are present and "
                "schema-valid before enabling strict mode."
            ),
        )
    unique_diagnostics = tuple(dict.fromkeys(artifact_diagnostics))
    artifact_warning_rows = tuple(
        dict.fromkeys(
            _artifact_diagnostic_to_text(item)
            for item in unique_diagnostics
            if item.severity == "warning"
        )
    )
    if artifact_errors:
        unique_errors = tuple(dict.fromkeys(artifact_errors))
        error_detail_text = "\n".join(f"  - {row}" for row in unique_errors)
        raise ArtifactLoadRuntimeError(
            f"strict artifact load checks failed:\n{error_detail_text}",
            diagnostics=unique_diagnostics,
        )
    warnings.extend(artifact_warning_rows)
    lineage_registry = _build_lineage_registry(
        entries=lineage_entries,
        links=lineage_links,
        workflow_artifact_paths=workflow_artifact_paths,
        default_profile=default_profile,
        additional_warnings=artifact_warning_rows,
    )

    overview = DashboardOverview(
        total_candidate_factors=len(factor_summaries),
        validated_factors=sum(
            1
            for item in factor_summaries
            if item.research_status in {"validated", "portfolio-active"}
        ),
        active_portfolio_recipes=sum(
            1
            for item in recipes
            if not item.infeasible_configuration_warnings and item.recipe_id in recipe_lookup
        ),
        completed_backtests=sum(1 for item in backtests if item.annualized_return is not None),
        top_factors_by_signal_quality=tuple(
            _top_factor_lines(factor_summaries=factor_summaries, top_n=5)
        ),
        top_portfolios_by_objective=tuple(_top_portfolio_lines(backtests=backtests, top_n=5)),
        recent_research_runs=tuple(_recent_run_lines(recent_runs=recent_runs, top_n=8)),
        warnings=tuple(dict.fromkeys(warnings)),
    )

    return ResearchDashboardData(
        overview=overview,
        factor_summaries=tuple(factor_summaries),
        factor_details=tuple(factor_details),
        comparison_rows=tuple(comparison_rows),
        factor_shortlist=shortlist_result,
        factor_sets=factor_set_result,
        candidate_recipe_generation=candidate_recipe_generation,
        portfolio_recipes=tuple(recipes),
        recipe_comparison=recipe_comparison,
        winner_selection=winner_selection,
        next_step_recommendations=next_step_recommendations,
        backtests=tuple(backtests),
        lineage_registry=lineage_registry,
        robustness_summaries=tuple(robustness_rows),
        factor_correlation_matrix=correlation_matrix,
        factor_family_summary=family_summary,
        shortlist_recommendations=shortlist,
        generated_at_utc=generated_at_utc,
        default_profile=default_profile,
        source_json_path=str(comparison_path),
        artifact_load_mode=load_policy.mode,
        artifact_load_policy_summary=_artifact_load_policy_summary(load_policy),
        artifact_load_warnings=artifact_warning_rows,
        artifact_load_diagnostics=unique_diagnostics,
    )


def _collect_case_warnings(
    *,
    case_name: str,
    profile_payload: dict[str, object],
    artifacts: _CaseArtifacts,
    metrics: dict[str, object],
) -> list[str]:
    rows: list[str] = []
    status = safe_text(profile_payload.get("status")) or "unknown"
    if status != "success":
        rows.append(f"{case_name}: run status is {status}")

    coverage = _safe_float(metrics.get("coverage_mean"))
    if coverage is not None and coverage < 0.7:
        rows.append(f"{case_name}: low coverage mean ({coverage:.2f})")

    recommendation = safe_text(profile_payload.get("portfolio_validation_recommendation"))
    if recommendation and recommendation == "Needs portfolio refinement":
        rows.append(f"{case_name}: portfolio recipe still requires refinement")

    summary = as_object_dict(artifacts.integrity_report.get("summary"))
    n_warn = int(_float_or_default(summary.get("n_warn"), default=0.0))
    n_fail = int(_float_or_default(summary.get("n_fail"), default=0.0))
    if n_fail > 0:
        rows.append(f"{case_name}: integrity checks contain failures ({n_fail})")
    elif n_warn > 0:
        rows.append(f"{case_name}: integrity checks contain warnings ({n_warn})")

    for artifact_name, field_names in sorted(artifacts.fallback_derived_fields.items()):
        if not field_names:
            continue
        preview = ", ".join(field_names[:3])
        suffix = " ..." if len(field_names) > 3 else ""
        rows.append(
            f"{case_name}: {artifact_name} keeps fallback-derived fields ({preview}{suffix})"
        )

    return rows


def _build_portfolio_recipe(
    *,
    case_name: str,
    factor_name: str,
    profile_payload: dict[str, object],
    metrics: dict[str, object],
    portfolio_recipe_payload: dict[str, object],
    portfolio_summary: dict[str, object],
    portfolio_metrics: dict[str, object],
    spec: dict[str, object],
    universe: dict[str, object],
    neutralization: dict[str, object],
) -> PortfolioRecipeSummary:
    protocol = as_object_dict(portfolio_metrics.get("protocol_settings"))
    weighting_scheme_obj = as_object_dict(protocol.get("weighting_scheme"))
    weighting_scheme = (
        safe_text(weighting_scheme_obj.get("default"))
        or safe_text(metrics.get("base_weighting_method"))
        or "rank"
    )

    benchmark_status = (
        safe_text(portfolio_summary.get("benchmark_relative_status")) or "not_available"
    )
    benchmark_mode = (
        "benchmark-relative" if benchmark_status not in {"not_available", "none"} else "absolute"
    )

    neutralization_enabled = bool(neutralization.get("enabled"))
    neutralization_constraints = (
        "size-neutralization enabled"
        if neutralization_enabled
        else "no explicit neutralization constraint"
    )

    industry_col = safe_text(neutralization.get("industry_col"))
    industry_constraints = (
        f"industry neutral by `{industry_col}`" if industry_col else "none declared"
    )
    style_col = safe_text(neutralization.get("size_col"))
    style_constraints = f"size proxy `{style_col}`" if style_col else "none declared"

    pv_snapshot = as_object_dict(metrics.get("research_evaluation_snapshot"))
    pv_thresholds = as_object_dict(pv_snapshot.get("level2_portfolio_validation"))
    turnover_penalty = _safe_float(pv_thresholds.get("max_mean_turnover_warn"))
    canonical_turnover_penalty = safe_text(
        portfolio_recipe_payload.get("turnover_penalty_settings")
    )

    transaction_cost = _safe_float(metrics.get("transaction_cost_one_way_rate"))
    cost_grid = as_object_list(protocol.get("transaction_cost_sensitivity"))
    cost_text = (
        f"one-way={_fmt(transaction_cost)}; grid={','.join(str(item) for item in cost_grid)}"
        if cost_grid
        else f"one-way={_fmt(transaction_cost)}"
    )
    canonical_cost_text = safe_text(portfolio_recipe_payload.get("transaction_cost_assumptions"))

    concentration = as_object_dict(portfolio_metrics.get("concentration_exposure_diagnostics"))
    max_abs_weight = _safe_float(concentration.get("max_abs_weight_mean"))
    effective_names = _safe_float(concentration.get("effective_names_mean"))
    position_limits = (
        f"max|w|~{_fmt(max_abs_weight)}; effective names~{_fmt(effective_names)}"
        if max_abs_weight is not None or effective_names is not None
        else "N/A"
    )
    canonical_position_limits = safe_text(portfolio_recipe_payload.get("position_limits"))

    base_return = _safe_float(portfolio_summary.get("base_mean_portfolio_return"))
    base_turnover = _safe_float(portfolio_summary.get("base_mean_turnover"))
    expected_risk_summary = (
        f"mean turnover={_fmt(base_turnover)}, max abs weight={_fmt(max_abs_weight)}, "
        f"effective names={_fmt(effective_names)}"
    )

    scenario_metrics = as_object_list(portfolio_metrics.get("scenario_metrics"))
    optimizer_diagnostics = (
        f"scenario_count={len(scenario_metrics)}; baseline weighting={weighting_scheme}; "
        "single-factor heuristic portfolio constructor"
    )

    support_reasons = _portfolio_support_reasons(portfolio_summary)
    warnings = _portfolio_fragility_reasons(portfolio_summary)
    profile_recommendation = safe_text(profile_payload.get("portfolio_validation_recommendation"))
    if profile_recommendation and profile_recommendation != "Credible at portfolio level":
        warnings = list(dict.fromkeys([*warnings, f"recommendation={profile_recommendation}"]))

    rebalance_freq = (
        safe_text(spec.get("rebalance_frequency"))
        or safe_text(metrics.get("rebalance_frequency"))
        or "N/A"
    )
    rebalance_step = _safe_float(portfolio_summary.get("rebalance_step_assumption"))
    rebalance_text = (
        f"{rebalance_freq} / step={int(rebalance_step) if rebalance_step is not None else 'N/A'}"
    )

    return PortfolioRecipeSummary(
        recipe_id=f"recipe-{case_name}",
        recipe_name=f"{factor_name} ({case_name})",
        selected_factors=(factor_name,),
        weighting_scheme=weighting_scheme,
        neutralization_constraints=neutralization_constraints,
        benchmark_mode=benchmark_mode,
        industry_constraints=industry_constraints,
        style_constraints=style_constraints,
        turnover_penalty_settings=canonical_turnover_penalty
        or (
            f"warn if mean turnover > {turnover_penalty:.2f}"
            if turnover_penalty is not None
            else "N/A"
        ),
        rebalance_frequency=rebalance_text,
        transaction_cost_assumptions=canonical_cost_text or cost_text,
        universe_definition=safe_text(universe.get("name")) or "N/A",
        position_limits=canonical_position_limits or position_limits,
        factor_contributions=tuple(support_reasons) or ("single-factor long-short contribution",),
        expected_risk_summary=expected_risk_summary,
        expected_return_proxy=(
            f"base mean portfolio return={_fmt(base_return)}" if base_return is not None else "N/A"
        ),
        optimizer_diagnostics=optimizer_diagnostics,
        infeasible_configuration_warnings=tuple(warnings),
    )


def _build_backtest_summary(
    *,
    recipe_id: str,
    factor_id: str,
    metrics: dict[str, object],
    portfolio_summary: dict[str, object],
    portfolio_metrics: dict[str, object],
    backtest_payload: dict[str, object],
    group_returns_df: pd.DataFrame | None,
    turnover_df: pd.DataFrame | None,
    rebalance_frequency: str,
) -> PortfolioBacktestSummary:
    backtest_summary = as_object_dict(backtest_payload.get("summary"))
    long_short_series = _long_short_series(group_returns_df)
    periods_per_year = _periods_per_year(rebalance_frequency)

    stats: dict[str, object] = {}
    if long_short_series is not None and len(long_short_series) > 1:
        stats = _return_stats(long_short_series, periods_per_year)

    annualized_return = _coalesce_float(
        backtest_summary.get("annualized_return"),
        stats.get("annualized_return"),
    )
    annualized_volatility = _coalesce_float(
        backtest_summary.get("annualized_volatility"),
        stats.get("annualized_volatility"),
    )
    sharpe = _coalesce_float(backtest_summary.get("sharpe"), stats.get("sharpe"))
    sortino = _coalesce_float(backtest_summary.get("sortino"), stats.get("sortino"))
    max_drawdown = _coalesce_float(
        backtest_summary.get("max_drawdown"),
        stats.get("max_drawdown"),
    )
    calmar = _coalesce_float(backtest_summary.get("calmar"), stats.get("calmar"))
    win_rate = _coalesce_float(backtest_summary.get("win_rate"), stats.get("win_rate"))
    rolling_sharpe = _coalesce_float(
        backtest_summary.get("rolling_sharpe"),
        stats.get("rolling_sharpe"),
    )
    rolling_drawdown = _coalesce_float(
        backtest_summary.get("rolling_drawdown"),
        stats.get("rolling_drawdown"),
    )
    nav_points = _coalesce_rows(
        _to_time_value_rows(backtest_summary.get("nav_points")),
        _rows_from_stats(stats.get("nav_points")),
    )
    monthly_returns = _coalesce_rows(
        _to_time_value_rows(backtest_summary.get("monthly_return_table")),
        _rows_from_stats(stats.get("monthly_returns")),
    )
    drawdown_table = _coalesce_rows(
        _to_time_value_rows(backtest_summary.get("drawdown_table")),
        _rows_from_stats(stats.get("drawdown_table")),
    )
    subperiod_analysis = (
        safe_text(backtest_summary.get("subperiod_analysis"))
        or safe_text(stats.get("subperiod_analysis"))
        or "N/A"
    )
    regime_analysis = (
        safe_text(backtest_summary.get("regime_analysis"))
        or safe_text(stats.get("regime_analysis"))
        or "N/A"
    )

    benchmark_excess = _coalesce_float(
        backtest_summary.get("excess_return_vs_benchmark"),
        metrics.get("portfolio_validation_benchmark_excess_return"),
    )
    tracking_error = _coalesce_float(
        backtest_summary.get("tracking_error"),
        metrics.get("portfolio_validation_benchmark_tracking_error"),
    )

    scenario = _baseline_scenario(portfolio_metrics)
    information_ratio = _coalesce_float(
        backtest_summary.get("information_ratio"),
        scenario.get("portfolio_ir"),
    )
    if information_ratio is None:
        information_ratio = _safe_float(
            metrics.get("portfolio_validation_benchmark_information_ratio")
        )

    turnover = _coalesce_float(
        backtest_summary.get("turnover"),
        metrics.get("mean_long_short_turnover"),
    )
    if turnover is None and turnover_df is not None and "turnover" in turnover_df.columns:
        series = pd.to_numeric(turnover_df["turnover"], errors="coerce")
        finite = series.dropna()
        if not finite.empty:
            turnover = float(finite.mean())

    pre_cost_return = _coalesce_float(
        backtest_summary.get("pre_cost_return"),
        metrics.get("mean_long_short_return"),
    )
    post_cost_return = _coalesce_float(
        backtest_summary.get("post_cost_return"),
        metrics.get("mean_cost_adjusted_long_short_return"),
    )

    concentration = as_object_dict(portfolio_metrics.get("concentration_exposure_diagnostics"))
    portfolio_composition = (
        f"max|w|={_fmt(_safe_float(concentration.get('max_abs_weight_mean')))}, "
        f"top5 abs share={_fmt(_safe_float(concentration.get('top5_abs_weight_share_mean')))}, "
        f"effective names={_fmt(_safe_float(concentration.get('effective_names_mean')))}"
    )

    trade_statistics = (
        f"hit_rate={_fmt(_safe_float(scenario.get('portfolio_hit_rate')))}, "
        f"mean_turnover={_fmt(_safe_float(scenario.get('mean_turnover')))}, "
        f"n_return_dates={_fmt(_safe_float(scenario.get('n_return_dates')))}"
    )

    summary_robust = as_object_dict(portfolio_summary.get("portfolio_robustness_summary"))
    capacity_notes = (
        safe_text(summary_robust.get("concentration_turnover_risk_note"))
        or "Capacity/implementability notes unavailable from current artifacts."
    )

    attribution = (
        "Attribution proxy: single-factor long-short return stream and baseline scenario IR "
        "from portfolio_validation_metrics."
    )

    return PortfolioBacktestSummary(
        recipe_id=recipe_id,
        factor_id=factor_id,
        annualized_return=annualized_return,
        annualized_volatility=annualized_volatility,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=max_drawdown,
        calmar=calmar,
        win_rate=win_rate,
        turnover=turnover,
        information_ratio=information_ratio,
        excess_return_vs_benchmark=benchmark_excess,
        tracking_error=tracking_error,
        pre_cost_return=pre_cost_return,
        post_cost_return=post_cost_return,
        nav_points=nav_points,
        monthly_return_table=monthly_returns,
        drawdown_table=drawdown_table,
        period_by_period_attribution=attribution,
        subperiod_analysis=subperiod_analysis,
        regime_analysis=regime_analysis,
        rolling_sharpe=rolling_sharpe,
        rolling_drawdown=rolling_drawdown,
        portfolio_composition_snapshot=portfolio_composition,
        trade_statistics=trade_statistics,
        capacity_implementability_notes=capacity_notes,
    )


def _build_robustness_summary(
    *,
    case_name: str,
    case_row: dict[str, object],
    artifacts: _CaseArtifacts,
    profile_payload: dict[str, object],
    portfolio_summary: dict[str, object],
    portfolio_metrics: dict[str, object],
) -> RobustnessSummary:
    holding_rows = as_object_list(portfolio_metrics.get("holding_period_sensitivity"))
    weighting_rows = as_object_list(portfolio_metrics.get("weighting_sensitivity"))

    parameter_sensitivity = _spread_note(
        values=[
            _safe_float(as_object_dict(item).get("mean_portfolio_return"))
            for item in weighting_rows
        ],
        label="weighting method",
    )
    lookback_sensitivity = _spread_note(
        values=[
            _safe_float(as_object_dict(item).get("mean_portfolio_return")) for item in holding_rows
        ],
        label="holding horizon",
    )
    transaction_cost_sensitivity = (
        safe_text(
            as_object_dict(portfolio_summary.get("portfolio_robustness_summary")).get(
                "cost_sensitivity_note"
            )
        )
        or "N/A"
    )

    summary = as_object_dict(artifacts.integrity_report.get("summary"))
    n_pass = int(_float_or_default(summary.get("n_pass"), default=0.0))
    n_warn = int(_float_or_default(summary.get("n_warn"), default=0.0))
    n_fail = int(_float_or_default(summary.get("n_fail"), default=0.0))
    leakage_checks = f"pass={n_pass}, warn={n_warn}, fail={n_fail}"

    pit_warnings: list[str] = []
    impl_warnings: list[str] = []
    for check_obj in as_object_list(artifacts.integrity_report.get("checks")):
        check = as_object_dict(check_obj)
        message = safe_text(check.get("message"))
        remediation = safe_text(check.get("remediation"))
        status = safe_text(check.get("status")) or ""
        if status in {"warn", "fail"}:
            if message:
                impl_warnings.append(message)
            if remediation:
                impl_warnings.append(remediation)
        text = (message or "") + " " + (remediation or "")
        if "known_at" in text or "available_at" in text or "PIT" in text or "as-of" in text:
            pit_warnings.append(text.strip())

    profile_sensitivity = safe_text(case_row.get("profile_sensitivity")) or "N/A"
    robustness_label = safe_text(portfolio_summary.get("recommendation")) or (
        safe_text(profile_payload.get("portfolio_validation_recommendation")) or "N/A"
    )

    rebalance_assumption = as_object_dict(
        as_object_dict(portfolio_metrics.get("protocol_settings")).get("rebalance_assumption")
    )
    rebalance_sensitivity = (
        f"input_frequency={safe_text(rebalance_assumption.get('input_frequency')) or 'N/A'}, "
        f"rebalance_step={safe_text(rebalance_assumption.get('rebalance_step')) or 'N/A'}"
    )

    return RobustnessSummary(
        factor_id=case_name,
        parameter_sensitivity=parameter_sensitivity,
        lookback_sensitivity=lookback_sensitivity,
        universe_sensitivity=(
            "profile sensitive across evaluation profiles"
            if "sensitive" in profile_sensitivity
            else "profile stable under compared profiles"
        ),
        rebalance_sensitivity=rebalance_sensitivity,
        transaction_cost_sensitivity=transaction_cost_sensitivity,
        profile_sensitivity=profile_sensitivity,
        leakage_checks=leakage_checks,
        survivorship_pit_checks=(
            " | ".join(pit_warnings[:2])
            if pit_warnings
            else "No explicit PIT warning in integrity report."
        ),
        implementation_warnings=tuple(dict.fromkeys(impl_warnings[:5])),
        robustness_verdict=robustness_label,
    )


def _render_quick_nav() -> str:
    return (
        '<nav class="quick-nav">'
        f'<a href="#overview">{_h(_ui("nav_overview"))}</a>'
        f'<a href="#factor-library">{_h(_ui("nav_factor_library"))}</a>'
        f'<a href="#factor-detail">{_h(_ui("nav_factor_detail"))}</a>'
        f'<a href="#cross-factor">{_h(_ui("nav_cross_factor"))}</a>'
        f'<a href="#selected-factor-sets">{_h(_ui("nav_selected_factor_sets"))}</a>'
        f'<a href="#candidate-recipe-generation">{_h(_ui("nav_candidate_recipe_generation"))}</a>'
        f'<a href="#portfolio-construction">{_h(_ui("nav_portfolio_construction"))}</a>'
        f'<a href="#winner-selection">{_h(_ui("nav_winner_selection"))}</a>'
        f'<a href="#next-step-recommendations">{_h(_ui("nav_next_step_recommendations"))}</a>'
        f'<a href="#backtest-evaluation">{_h(_ui("nav_backtest_evaluation"))}</a>'
        f'<a href="#lineage-registry">{_h(_ui("nav_lineage_registry"))}</a>'
        f'<a href="#robustness-audit">{_h(_ui("nav_robustness"))}</a>'
        "</nav>"
    )


def _render_overview(overview: DashboardOverview) -> str:
    cards = [
        (_ui("card_candidate_factors"), str(overview.total_candidate_factors)),
        (_ui("card_validated_factors"), str(overview.validated_factors)),
        (_ui("card_active_portfolio_recipes"), str(overview.active_portfolio_recipes)),
        (_ui("card_completed_backtests"), str(overview.completed_backtests)),
    ]
    card_html = "".join(
        '<div class="summary-card reveal">'
        f'<div class="summary-label">{_h(label)}</div>'
        f'<div class="summary-value">{_h(value)}</div>'
        "</div>"
        for label, value in cards
    )

    top_factors = _render_line_list(
        lines=overview.top_factors_by_signal_quality,
        empty_text="暂无因子信号质量排名 (No factor signal-quality ranking yet).",
    )
    top_portfolios = _render_line_list(
        lines=overview.top_portfolios_by_objective,
        empty_text="暂无组合排名 (No portfolio ranking yet).",
    )
    recent_runs = _render_line_list(
        lines=overview.recent_research_runs,
        empty_text="暂无近期运行元数据 (No recent run metadata).",
    )
    warnings = _render_line_list(
        lines=overview.warnings,
        empty_text="当前工件未检测到关键告警 (No critical warnings detected in current artifacts).",
        warning_mode=True,
    )

    return (
        f"<h2>{_h(_ui('section_overview'))}</h2>"
        '<p class="section-note">'
        "因子发现、信号验证、组合构建与回测结果的总控视图。"
        " (Command center for factor discovery, validation, portfolio construction, and backtest outcomes.)"
        "</p>"
        f'<div class="summary-grid">{card_html}</div>'
        '<div class="two-col">'
        '<article class="content-card reveal">'
        "<h3>信号质量领先因子 (Top Factors by Signal Quality)</h3>"
        f"{top_factors}"
        "</article>"
        '<article class="content-card reveal">'
        "<h3>按目标领先的组合 (Top Portfolios by Objective: Sharpe / Return / IR)</h3>"
        f"{top_portfolios}"
        "</article>"
        "</div>"
        '<div class="two-col">'
        '<article class="content-card reveal">'
        "<h3>近期研究运行与更新 (Recent Research Runs / Latest Updates)</h3>"
        f"{recent_runs}"
        "</article>"
        '<article class="content-card content-card-warn reveal">'
        "<h3>告警/失败运行/覆盖缺口 (Warnings / Failed Runs / Missing Coverage)</h3>"
        f"{warnings}"
        "</article>"
        "</div>"
    )


def _render_factor_library(factors: tuple[FactorSummary, ...]) -> str:
    families = sorted({item.factor_family for item in factors})
    statuses = sorted(
        {item.research_status for item in factors}, key=lambda item: _STATUS_ORDER.get(item, 999)
    )

    family_options = "".join(
        f'<option value="{_h(family)}">{_h(family)}</option>' for family in families
    )
    status_options = "".join(
        f'<option value="{_h(status)}">{_h(_display_status(status))}</option>'
        for status in statuses
    )

    rows: list[str] = []
    for item in factors:
        detail_anchor = f"factor-detail-{_slug(item.factor_id)}"
        display_factor_name = _display_name_with_zh(
            english=item.factor_name,
            zh=item.display_name_zh,
        )
        display_short_description = _display_desc_with_zh(
            english=item.short_description,
            zh=item.short_description_zh,
        )
        search_tokens = [
            token
            for token in (
                safe_text(item.factor_name),
                safe_text(item.display_name_zh),
                safe_text(item.factor_id),
            )
            if token
        ]
        search_blob = " ".join(search_tokens).lower()
        rows.append(
            "<tr "
            f'data-factor-name="{_h(search_blob)}" '
            f'data-factor-id="{_h(item.factor_id.lower())}" '
            f'data-family="{_h(item.factor_family)}" '
            f'data-status="{_h(item.research_status)}" '
            f'data-score="{_h(_sort_value(item.signal_quality_score))}" '
            f'data-coverage="{_h(_sort_value(item.coverage_ratio))}">'
            f'<td><a href="#{_h(detail_anchor)}">{_h(display_factor_name)}</a></td>'
            f"<td><code>{_h(item.factor_id)}</code></td>"
            f"<td>{_h(display_short_description)}</td>"
            f"<td>{_h(item.factor_family)}</td>"
            f"<td><code>{_h(item.mathematical_definition)}</code></td>"
            f"<td>{_h(', '.join(item.required_input_fields) if item.required_input_fields else _ui('na'))}</td>"
            f"<td>{_h(item.frequency)}</td>"
            f"<td>{_h(', '.join(item.lookback_parameters) if item.lookback_parameters else _ui('na'))}</td>"
            f"<td>{_h(item.lag_delay_rule)}</td>"
            f"<td>{_h(item.expected_sign)}</td>"
            f"<td>{_h(_fmt(item.coverage_ratio))}</td>"
            f"<td>{_h(item.missingness_summary)}</td>"
            f"<td>{_h(_display_status(item.research_status))}</td>"
            "</tr>"
        )

    return (
        f"<h2>{_h(_ui('section_factor_library'))}</h2>"
        '<p class="section-note">将已发现因子作为一等研究对象进行检索、筛选与排序。'
        " (Search, filter, and sort discovered factors as first-class research objects.)</p>"
        '<div class="filter-bar">'
        '<input id="factor-search-input" type="search" placeholder="搜索因子名 / 案例ID (Search factor name / case id)" />'
        '<select id="factor-family-filter">'
        '<option value="">全部家族 (All Families)</option>'
        f"{family_options}"
        "</select>"
        '<select id="factor-status-filter">'
        '<option value="">全部状态 (All Status)</option>'
        f"{status_options}"
        "</select>"
        '<select id="factor-sort-select">'
        '<option value="signal_desc">排序：信号质量降序 (Signal Quality ↓)</option>'
        '<option value="coverage_desc">排序：覆盖率降序 (Coverage ↓)</option>'
        '<option value="name_asc">排序：名称升序 (Name A→Z)</option>'
        "</select>"
        '<span id="factor-library-count" class="count-chip">0 可见 (visible)</span>'
        "</div>"
        '<div class="table-wrap">'
        '<table id="factor-library-table">'
        "<thead><tr>"
        "<th>因子名称 (Factor Name)</th><th>因子ID (Factor ID)</th><th>描述 (Description)</th><th>家族 (Family)</th>"
        "<th>定义 (Definition)</th><th>输入字段 (Input Fields)</th><th>频率 (Freq)</th><th>回看参数 (Lookback)</th>"
        "<th>滞后规则 (Lag Rule)</th><th>预期方向 (Expected Sign)</th><th>覆盖率 (Coverage)</th><th>缺失率 (Missingness)</th><th>状态 (Status)</th>"
        "</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
        "</div>"
        '<p id="factor-library-empty" class="section-note" style="display:none;">当前筛选条件无匹配因子 (No factors match current filters).</p>'
    )


def _render_factor_detail(details: tuple[FactorDetail, ...]) -> str:
    blocks: list[str] = []
    for detail in details:
        summary = detail.summary
        validation = detail.validation
        display_factor_name = _display_name_with_zh(
            english=summary.factor_name,
            zh=summary.display_name_zh,
        )
        display_short_description = _display_desc_with_zh(
            english=summary.short_description,
            zh=summary.short_description_zh,
        )
        block = (
            f'<article id="factor-detail-{_h(_slug(summary.factor_id))}" class="factor-detail-card reveal">'
            "<header>"
            f"<h3>{_h(display_factor_name)} <small><code>{_h(summary.factor_id)}</code></small></h3>"
            f'<p class="section-note">{_h(display_short_description)}</p>'
            "</header>"
            '<div class="detail-grid">'
            '<section class="detail-block">'
            "<h4>1. 定义与元数据 (Definition & Metadata)</h4>"
            f"<p><strong>形式化定义 (Formal definition):</strong> <code>{_h(detail.formal_definition)}</code></p>"
            f"<p><strong>实现说明 (Implementation notes):</strong> {_h(detail.implementation_notes)}</p>"
            f"<p><strong>PIT/防前视 (PIT / anti-lookahead):</strong> {_h(detail.pit_anti_lookahead_notes)}</p>"
            f"<p><strong>数据依赖 (Data dependencies):</strong> {_h(', '.join(detail.data_dependencies) if detail.data_dependencies else _ui('na'))}</p>"
            f"<p><strong>参数设置 (Parameter settings):</strong> {_h(', '.join(detail.parameter_settings) if detail.parameter_settings else _ui('na'))}</p>"
            f"<p><strong>目标持有期 (Intended holding horizon):</strong> {_h(detail.intended_holding_horizon)}</p>"
            "</section>"
            '<section class="detail-block">'
            "<h4>2. 数据质量与分布 (Data Quality / Distribution)</h4>"
            f"<p><strong>时序覆盖 (Coverage over time):</strong> {_h(detail.coverage_over_time)}</p>"
            f"<p><strong>截面覆盖 (Cross-sectional coverage):</strong> {_h(detail.cross_sectional_coverage)}</p>"
            f"<p><strong>缺失情况 (Missingness):</strong> {_h(detail.missingness)}</p>"
            f"<p><strong>去极值/截断 (Winsorization / clipping):</strong> {_h(detail.winsorization_clipping_summary)}</p>"
            f"<p><strong>标准化/中性化 (Standardization / neutralization):</strong> {_h(detail.standardization_neutralization_summary)}</p>"
            f"<p><strong>分布快照 (Distribution snapshots):</strong> {_h('; '.join(detail.distribution_snapshots) if detail.distribution_snapshots else _ui('na'))}</p>"
            f"<p><strong>因子值换手率 (Turnover of factor values):</strong> {_h(detail.turnover_of_factor_values)}</p>"
            f"<p><strong>时序稳定性 (Stability over time):</strong> {_h(detail.stability_over_time)}</p>"
            "</section>"
            '<section class="detail-block">'
            "<h4>3. 信号验证 (Signal Validation)</h4>"
            '<table class="mini-table">'
            "<tbody>"
            f"<tr><th>信息系数均值 (IC Mean)</th><td>{_h(_fmt(validation.ic_mean))}</td></tr>"
            f"<tr><th>秩信息系数均值 (Rank IC Mean)</th><td>{_h(_fmt(validation.rank_ic_mean))}</td></tr>"
            f"<tr><th>ICIR</th><td>{_h(_fmt(validation.icir))}</td></tr>"
            f"<tr><th>t统计代理 (t-stat proxy)</th><td>{_h(_fmt(validation.t_stat_proxy))}</td></tr>"
            f"<tr><th>命中率/IC为正频率 (Hit rate / positive IC frequency)</th><td>{_h(_fmt(validation.hit_rate))} / {_h(_fmt(validation.positive_ic_frequency))}</td></tr>"
            f"<tr><th>衰减轮廓 (Decay profile)</th><td>{_h('; '.join(validation.decay_profile) if validation.decay_profile else _ui('na'))}</td></tr>"
            f"<tr><th>期限分析 (t+1/t+5/t+10)</th><td>{_h('; '.join(validation.horizon_analysis) if validation.horizon_analysis else _ui('na'))}</td></tr>"
            f"<tr><th>分组收益差 (Quantile/decile return spread)</th><td>{_h(_fmt(validation.quantile_return_spread))}</td></tr>"
            f"<tr><th>多空表现 (Long-short performance)</th><td>{_h(_display_text(validation.long_short_performance_summary))}</td></tr>"
            f"<tr><th>单调性诊断 (Monotonicity diagnostics)</th><td>{_h(_display_text(validation.monotonicity_diagnostics))}</td></tr>"
            f"<tr><th>市场状态拆解 (Regime breakdown)</th><td>{_h(_display_text(validation.regime_breakdown))}</td></tr>"
            f"<tr><th>行业中性对比 (Industry-neutral comparison)</th><td>{_h(_display_text(validation.industry_neutral_comparison))}</td></tr>"
            f"<tr><th>规模中性对比 (Size-neutral comparison)</th><td>{_h(_display_text(validation.size_neutral_comparison))}</td></tr>"
            f"<tr><th>训练/验证/样本外拆分 (Train/validation/OOS split)</th><td>{_h(_display_text(validation.split_summary))}</td></tr>"
            "</tbody>"
            "</table>"
            "</section>"
            '<section class="detail-block">'
            "<h4>4. 解读 (Interpretation)</h4>"
            f"<p><strong>结论 (Verdict):</strong> {_h(detail.concise_verdict_zh or _display_text(detail.concise_verdict))}</p>"
            f"<p><strong>优势 (Strengths):</strong> {_h(_display_text('; '.join(detail.strengths) if detail.strengths else _ui('na')))}</p>"
            f"<p><strong>弱点 (Weaknesses):</strong> {_h(_display_text('; '.join(detail.weaknesses) if detail.weaknesses else _ui('na')))}</p>"
            f"<p><strong>潜在失效模式 (Likely failure modes):</strong> {_h(_display_text('; '.join(detail.likely_failure_modes) if detail.likely_failure_modes else _ui('na')))}</p>"
            f"<p><strong>是否进入组合层 (Proceed to portfolio layer):</strong> {_h('是 (yes)' if detail.proceed_to_portfolio_layer else '否 (no)')}</p>"
            "</section>"
            '<section class="detail-block">'
            "<h4>5. 相关工件 (Related Artifacts)</h4>"
            f'<p><a href="#{_h(_lineage_anchor(summary.factor_id))}">打开血缘/溯源条目 (Open lineage / provenance entry)</a></p>'
            f"{_render_line_list(detail.related_artifacts, empty_text='暂无工件指针 (No artifact pointers available).')}"
            "</section>"
            "</div>"
            "</article>"
        )
        blocks.append(block)

    return (
        f"<h2>{_h(_ui('section_factor_detail'))}</h2>"
        '<p class="section-note">逐因子深度检查，从因子库行进入详情。 (Deep inspection for one factor at a time; start from library rows and drill down.)</p>'
        f"{''.join(blocks) if blocks else '<p class="section-note">暂无因子详情 (No factor detail available).</p>'}"
    )


def _render_cross_factor(data: ResearchDashboardData) -> str:
    table_rows = "".join(
        "<tr>"
        f"<td><code>{_h(row.factor_id)}</code></td>"
        f"<td>{_h(row.factor_name)}</td>"
        f"<td>{_h(row.factor_family)}</td>"
        f"<td>{_h(_fmt(row.ic_mean))}</td>"
        f"<td>{_h(_fmt(row.rank_ic_mean))}</td>"
        f"<td>{_h(_fmt(row.icir))}</td>"
        f"<td>{_h(_fmt(row.turnover))}</td>"
        f"<td>{_h(_fmt(row.coverage))}</td>"
        f"<td>{_h(_fmt(row.long_short_return))}</td>"
        f"<td>{_h(_fmt_pct(row.monotonicity_share))}</td>"
        f"<td>{_h(_fmt_pct(row.oos_stability_share))}</td>"
        f"<td>{_h(_display_text(row.monotonicity))}</td>"
        f"<td>{_h(_display_text(row.oos_stability))}</td>"
        "</tr>"
        for row in data.comparison_rows
    )

    matrix_html = _render_correlation_matrix(data.factor_correlation_matrix)
    family_summary_html = "".join(
        f"<li><code>{_h(name)}</code>: {_h(str(count))} 个因子 (factors)</li>"
        for name, count in data.factor_family_summary
    )
    shortlist = _render_line_list(
        data.factor_shortlist.recommendation_summary,
        empty_text="暂无候选清单建议 (No shortlist recommendation available).",
    )
    shortlist_rows = "".join(
        "<tr>"
        f"<td>{_h(str(entry.rank))}</td>"
        f"<td><code>{_h(entry.factor_id)}</code></td>"
        f"<td>{_h(entry.factor_name)}</td>"
        f"<td>{_h(_display_shortlist_recommendation(entry.recommendation))}</td>"
        f"<td>{_h(_fmt(entry.composite_score))}</td>"
        f"<td>{_h(_fmt(entry.ic_mean))}</td>"
        f"<td>{_h(_fmt(entry.rank_ic_mean))}</td>"
        f"<td>{_h(_fmt(entry.icir))}</td>"
        f"<td>{_h(_fmt_pct(entry.monotonicity_share))}</td>"
        f"<td>{_h(_fmt(entry.turnover))}</td>"
        f"<td>{_h(_fmt_pct(entry.oos_stability_share))}</td>"
        f"<td>{_h(_fmt(entry.max_correlation_to_selected))}</td>"
        f"<td>{_h(entry.redundancy_with or _ui('na'))}</td>"
        f"<td>{_h(_display_text('; '.join(entry.rationale) if entry.rationale else _ui('na')))}</td>"
        "</tr>"
        for entry in data.factor_shortlist.entries
    )
    cfg = data.factor_shortlist.config
    weight_lines = ", ".join(f"{name}={weight:.2f}" for name, weight in cfg.component_weights)
    threshold_lines = (
        f"保留 keep>={cfg.keep_score_min:.2f}, 观察 watchlist>={cfg.watchlist_score_min:.2f}, "
        f"信息系数 IC>={cfg.min_ic_mean:.3f}, 秩信息系数 Rank IC>={cfg.min_rank_ic_mean:.3f}, ICIR>={cfg.min_icir:.2f}, "
        f"单调性 monotonicity>={cfg.min_monotonicity_share:.0%}, 换手率 Turnover<={cfg.max_turnover:.2f}, "
        f"样本外稳定性 OOS Stability>={cfg.min_oos_stability_share:.0%}, 冗余相关性 corr<={cfg.redundancy_correlation_max:.2f}"
    )

    return (
        f"<h2>{_h(_ui('section_cross_factor'))}</h2>"
        '<p class="section-note">'
        "回答核心问题：哪些因子具备区分度、预测力，并值得进入在线候选清单。"
        " (Which factors are distinct, predictive, and worth keeping in the live shortlist.)"
        "</p>"
        '<div class="table-wrap">'
        "<table>"
        "<thead><tr>"
        "<th>因子ID (Factor ID)</th><th>因子 (Factor)</th><th>家族 (Family)</th><th>信息系数 (IC)</th><th>秩信息系数 (Rank IC)</th><th>ICIR</th>"
        "<th>换手率 (Turnover)</th><th>覆盖率 (Coverage)</th><th>多空收益 (L/S Return)</th><th>单调性占比 (Monotonicity Share)</th><th>样本外稳定性占比 (OOS Stability Share)</th><th>单调性 (Monotonicity)</th><th>样本外稳定性 (OOS Stability)</th>"
        "</tr></thead>"
        f"<tbody>{table_rows}</tbody>"
        "</table>"
        "</div>"
        '<div class="two-col">'
        '<article class="content-card reveal">'
        "<h3>冗余度/相关性矩阵 (Redundancy / Correlation Matrix)</h3>"
        f"{matrix_html}"
        "</article>"
        '<article class="content-card reveal">'
        "<h3>聚类/分组视图 (Cluster / Group View)</h3>"
        f"<ul>{family_summary_html or ('<li>' + _h(_ui('na')) + '</li>')}</ul>"
        "<h3>候选清单建议区 (Shortlist Recommendation Area)</h3>"
        f"{shortlist}"
        "</article>"
        "</div>"
        '<article class="content-card reveal">'
        "<h3>候选清单综合评分（规范优先） (Shortlist Composite Score, Canonical-First)</h3>"
        f"<p><strong>公式 (Formula):</strong> {_h(cfg.formula)}</p>"
        f"<p><strong>权重 (Weights):</strong> {_h(weight_lines)}</p>"
        f"<p><strong>阈值 (Thresholds):</strong> {_h(threshold_lines)}</p>"
        '<div class="table-wrap">'
        "<table>"
        "<thead><tr>"
        "<th>排名 (Rank)</th><th>因子ID (Factor ID)</th><th>因子 (Factor)</th><th>建议 (Recommendation)</th><th>综合分 (Composite)</th>"
        "<th>信息系数 (IC)</th><th>秩信息系数 (Rank IC)</th><th>ICIR</th><th>单调性 (Monotonicity)</th><th>换手率 (Turnover)</th><th>样本外稳定性 (OOS)</th>"
        "<th>与入选最大相关性 (Max Corr vs Selected)</th><th>冗余来源 (Redundant With)</th><th>理由 (Rationale)</th>"
        "</tr></thead>"
        f"<tbody>{shortlist_rows}</tbody>"
        "</table>"
        "</div>"
        "</article>"
    )


def _render_selected_factor_sets(result: FactorSetConstructionResult) -> str:
    cfg = result.config
    summary = _render_line_list(
        result.recommendation_summary,
        empty_text="暂无因子集合建议 (No factor-set recommendation generated).",
    )
    rows = "".join(
        "<tr>"
        f'<td><code>{_h(item.factor_set_id)}</code><br/><span class="section-note">{_h(item.label_zh or _display_factor_set_status(item.status))}</span></td>'
        f"<td>{_h(_display_factor_set_status(item.status))}</td>"
        f"<td>{_h(', '.join(item.factor_names) if item.factor_names else _ui('na'))}</td>"
        f"<td>{_h(item.construction_rule)}</td>"
        f"<td>{_h(_fmt(item.score_summary.mean_shortlist_score))}</td>"
        f"<td>{_h(_fmt(item.score_summary.mean_icir))}</td>"
        f"<td>{_h(_fmt(item.score_summary.mean_turnover))}</td>"
        f"<td>{_h(_fmt_pct(item.score_summary.mean_oos_stability_share))}</td>"
        f"<td>{_h(_fmt(item.score_summary.max_pair_correlation))}</td>"
        f"<td>{_h(_fmt_pct(item.score_summary.family_balance_ratio))}</td>"
        f"<td>{_h('; '.join(item.rationale_zh) if item.rationale_zh else _display_text('; '.join(item.rationale) if item.rationale else _ui('na')))}</td>"
        f"<td>{_h(_display_text('; '.join(item.warnings) if item.warnings else _ui('na')))}</td>"
        f"<td>{_h('; '.join(item.source_shortlist_entries) if item.source_shortlist_entries else _ui('na'))}</td>"
        "</tr>"
        for item in result.factor_sets
    )
    threshold_text = (
        f"selected_score>={cfg.min_selected_score:.2f}, candidate_score>={cfg.min_candidate_score:.2f}, "
        f"turnover<={cfg.turnover_max:.2f}, OOS>={cfg.oos_stability_min:.0%}, "
        f"redundancy_corr<={cfg.redundancy_correlation_max:.2f}"
    )
    return (
        f"<h2>{_h(_ui('section_selected_factor_sets'))}</h2>"
        '<p class="section-note">'
        "流程第2步：基于候选清单构建明确且可复现的因子集合对象。"
        " (Workflow step 2: build explicit and reproducible factor-set objects from shortlist outputs.)"
        "</p>"
        f"<p><strong>策略ID (Policy ID):</strong> <code>{_h(cfg.policy_id)}</code></p>"
        f"<p><strong>构建公式 (Construction Formula):</strong> {_h(cfg.formula_text)}</p>"
        f"<p><strong>阈值 (Thresholds):</strong> {_h(threshold_text)}</p>"
        "<h3>集合建议摘要 (Set Recommendation Summary)</h3>"
        f"{summary}"
        '<div class="table-wrap">'
        "<table>"
        "<thead><tr>"
        "<th>因子集合ID (Factor Set ID)</th><th>状态 (Status)</th><th>因子名称 (Factor Names)</th><th>构建规则 (Construction Rule)</th>"
        "<th>平均得分 (Mean Score)</th><th>平均ICIR (Mean ICIR)</th><th>平均换手率 (Mean Turnover)</th><th>平均样本外稳定性 (Mean OOS)</th><th>最大相关性 (Max Corr)</th><th>家族均衡度 (Family Balance)</th>"
        "<th>理由 (Rationale)</th><th>告警 (Warnings)</th><th>来源候选条目 (Source Shortlist Entries)</th>"
        "</tr></thead>"
        f"<tbody>{rows or '<tr><td colspan="13">{}</td></tr>'.format(_ui('na'))}</tbody>"
        "</table>"
        "</div>"
    )


def _render_candidate_recipe_generation(result: CandidateRecipeGenerationResult) -> str:
    cfg = result.config
    summary = _render_line_list(
        result.recommendation_summary,
        empty_text="暂无候选配方生成结果 (No candidate recipes generated).",
    )
    rows = "".join(
        "<tr>"
        f"<td><code>{_h(item.recipe_id)}</code></td>"
        f"<td>{_h(item.recipe_name)}</td>"
        f"<td><code>{_h(item.source_factor_set_id)}</code></td>"
        f"<td>{_h(', '.join(item.source_factor_ids) if item.source_factor_ids else _ui('na'))}</td>"
        f"<td>{_h(item.construction_variant)}</td>"
        f"<td>{_h(item.weighting_scheme)}</td>"
        f"<td>{_h(item.neutralization_mode)}</td>"
        f"<td>{_h(item.turnover_penalty_mode)}</td>"
        f"<td>{_h(item.benchmark_mode)}</td>"
        f"<td>{_h(_display_text('; '.join(item.rationale) if item.rationale else _ui('na')))}</td>"
        f"<td>{_h(_display_text('; '.join(item.assumptions) if item.assumptions else _ui('na')))}</td>"
        f"<td>{_h(_display_text('; '.join(item.warnings) if item.warnings else _ui('na')))}</td>"
        "</tr>"
        for item in result.generated_recipes
    )
    return (
        f"<h2>{_h(_ui('section_candidate_recipe_generation'))}</h2>"
        '<p class="section-note">'
        "流程第3步：从入选/候选因子集合生成明确的候选组合配方。"
        " (Workflow step 3: generate explicit candidate portfolio recipes from selected/candidate factor sets.)"
        "</p>"
        f"<p><strong>策略ID (Policy ID):</strong> <code>{_h(cfg.policy_id)}</code></p>"
        f"<p><strong>生成公式 (Generation Formula):</strong> {_h(cfg.formula_text)}</p>"
        f"<p><strong>变体网格 (Variant Grid):</strong> weighting={_h(', '.join(cfg.weighting_schemes))}; "
        f"neutralization={_h(', '.join(cfg.neutralization_modes))}; "
        f"turnover_penalty={_h(', '.join(cfg.turnover_penalty_modes))}; "
        f"benchmark={_h(', '.join(cfg.benchmark_modes))}; "
        f"max_per_set={_h(str(cfg.max_recipes_per_factor_set))}</p>"
        "<h3>生成摘要 (Generation Summary)</h3>"
        f"{summary}"
        '<div class="table-wrap">'
        "<table>"
        "<thead><tr>"
        "<th>配方ID (Recipe ID)</th><th>名称 (Name)</th><th>来源因子集合 (Source Factor Set)</th><th>来源因子 (Source Factors)</th><th>构建变体 (Construction Variant)</th>"
        "<th>权重方式 (Weighting)</th><th>中性化 (Neutralization)</th><th>换手率惩罚 (Turnover Penalty)</th><th>基准模式 (Benchmark)</th>"
        "<th>理由 (Rationale)</th><th>假设 (Assumptions)</th><th>告警 (Warnings)</th>"
        "</tr></thead>"
        f"<tbody>{rows or '<tr><td colspan="12">{}</td></tr>'.format(_ui('na'))}</tbody>"
        "</table>"
        "</div>"
    )


def _render_winner_selection(result: WinnerSelectionResult) -> str:
    score_rows = "".join(
        f"<tr><td><code>{_h(recipe_id)}</code></td><td>{_h(_fmt(score))}</td></tr>"
        for recipe_id, score in result.score_table
    )
    decision_lines = _preferred_lines(
        result.decision_reasons_zh,
        result.decision_reasons,
    )
    challenger_lines = _preferred_lines(
        result.challenger_reasons_zh,
        result.challenger_reasons,
    )
    rejection_lines = _preferred_lines(
        result.rejection_reasons_zh,
        result.rejection_reasons,
    )
    next_action_lines = _preferred_lines(
        result.next_actions_zh,
        result.next_actions,
    )
    decision_reasons = _render_line_list(
        decision_lines,
        empty_text="暂无冠军决策理由 (No winner decision reason available).",
    )
    challenger_reasons = _render_line_list(
        challenger_lines,
        empty_text="暂无挑战者理由 (No challenger rationale).",
    )
    rejection_reasons = _render_line_list(
        rejection_lines,
        empty_text="暂无淘汰理由 (No rejected recipe rationale).",
    )
    next_actions = _render_line_list(
        next_action_lines,
        empty_text="暂无建议的下一步动作 (No suggested next action).",
    )
    return (
        f"<h2>{_h(_ui('section_winner_selection'))}</h2>"
        '<p class="section-note">'
        "流程第5步：使用显式决策策略划分冠军/挑战者/观察/淘汰。"
        " (Workflow step 5: apply an explicit decision policy to winner/challenger/watchlist/rejected buckets.)"
        "</p>"
        f"<p><strong>决策策略ID (Decision Policy ID):</strong> <code>{_h(result.decision_policy_id or _ui('na'))}</code></p>"
        f"<p><strong>策略公式 (Policy Formula):</strong> {_h(result.policy_formula_text)}</p>"
        '<table class="mini-table"><tbody>'
        f"<tr><th>当前冠军方案 (Current Winner)</th><td><code>{_h(result.winner_recipe_id or _ui('na'))}</code></td></tr>"
        f"<tr><th>挑战者 (Challengers)</th><td>{_h(', '.join(result.challenger_recipe_ids) if result.challenger_recipe_ids else _ui('na'))}</td></tr>"
        f"<tr><th>观察名单配方 (Watchlist Recipes)</th><td>{_h(', '.join(result.watchlist_recipe_ids) if result.watchlist_recipe_ids else _ui('na'))}</td></tr>"
        f"<tr><th>淘汰配方 (Rejected Recipes)</th><td>{_h(', '.join(result.rejected_recipe_ids) if result.rejected_recipe_ids else _ui('na'))}</td></tr>"
        "</tbody></table>"
        '<div class="two-col">'
        '<article class="content-card reveal"><h3>冠军决策理由 (Winner Decision Reasons)</h3>'
        f"{decision_reasons}</article>"
        '<article class="content-card reveal"><h3>决策评分表 (Decision Score Table)</h3>'
        '<div class="table-wrap"><table><thead><tr><th>配方 (Recipe)</th><th>综合分 (Composite Score)</th></tr></thead>'
        f"<tbody>{score_rows or '<tr><td colspan="2">{}</td></tr>'.format(_ui('na'))}</tbody></table></div></article>"
        "</div>"
        '<div class="two-col">'
        '<article class="content-card reveal"><h3>挑战者为何仍值得关注 (Why Challengers Stay Interesting)</h3>'
        f"{challenger_reasons}</article>"
        '<article class="content-card reveal"><h3>部分配方为何被淘汰 (Why Some Recipes Are Rejected)</h3>'
        f"{rejection_reasons}</article>"
        "</div>"
        "<h3>建议的下一步动作（决策层） (Recommended Next Actions, Decision Layer)</h3>"
        f"{next_actions}"
    )


def _render_next_step_recommendations(result: NextStepRecommendationResult) -> str:
    summary_lines = _preferred_lines(result.summary_zh, result.summary)
    summary = _render_line_list(
        summary_lines,
        empty_text="暂无下一步建议 (No next-step recommendation generated).",
    )
    rows = "".join(
        "<tr>"
        f"<td><code>{_h(item.recommendation_id)}</code></td>"
        f"<td>{_h(_display_priority(item.priority))}</td>"
        f"<td>{_h(item.label_zh or _display_next_step_category(item.category))}</td>"
        f"<td>{_h(item.action_text_zh or _display_text(item.action))}</td>"
        f"<td>{_h(item.rationale_zh or _display_text(item.rationale))}</td>"
        f"<td>{_h('; '.join(item.trigger_objects) if item.trigger_objects else _ui('na'))}</td>"
        f"<td>{_h(_display_text('; '.join(item.supporting_evidence) if item.supporting_evidence else _ui('na')))}</td>"
        "</tr>"
        for item in result.recommendations
    )
    return (
        f"<h2>{_h(_ui('section_next_step_recommendations'))}</h2>"
        '<p class="section-note">'
        "流程第6步：将候选清单/配方证据转换为明确且可审计的研究动作。"
        " (Workflow step 6: convert shortlist/recipe evidence into explicit and auditable next research actions.)"
        "</p>"
        f"<p><strong>建议策略ID (Recommendation Policy ID):</strong> <code>{_h(result.policy_id)}</code></p>"
        f"<p><strong>策略公式 (Policy Formula):</strong> {_h(result.policy_formula_text)}</p>"
        "<h3>建议摘要 (Recommendation Summary)</h3>"
        f"{summary}"
        '<div class="table-wrap">'
        "<table>"
        "<thead><tr><th>ID</th><th>优先级 (Priority)</th><th>类别 (Category)</th><th>动作 (Action)</th><th>理由 (Rationale)</th><th>触发对象 (Trigger Objects)</th><th>支持证据 (Supporting Evidence)</th></tr></thead>"
        f"<tbody>{rows or '<tr><td colspan="7">{}</td></tr>'.format(_ui('na'))}</tbody>"
        "</table>"
        "</div>"
    )


def _render_portfolio_construction(
    recipes: tuple[PortfolioRecipeSummary, ...],
    recipe_comparison: RecipeComparisonView,
) -> str:
    cards: list[str] = []
    for recipe in recipes:
        warnings = _render_line_list(
            recipe.infeasible_configuration_warnings,
            empty_text="暂无不可行配置告警 (No infeasible configuration warning).",
            warning_mode=True,
        )
        cards.append(
            '<article class="content-card reveal">'
            f"<h3>{_h(recipe.recipe_name)}</h3>"
            '<table class="mini-table"><tbody>'
            f'<tr><th>登记/血缘 (Registry / lineage)</th><td><a href="#{_h(_lineage_anchor(_recipe_case_name(recipe.recipe_id)))}">打开溯源条目 (Open provenance entry)</a></td></tr>'
            f"<tr><th>入选因子 (Selected factors)</th><td>{_h(', '.join(recipe.selected_factors) if recipe.selected_factors else _ui('na'))}</td></tr>"
            f"<tr><th>权重方式 (Weighting scheme)</th><td>{_h(recipe.weighting_scheme)}</td></tr>"
            f"<tr><th>中性化约束 (Neutralization constraints)</th><td>{_h(recipe.neutralization_constraints)}</td></tr>"
            f"<tr><th>基准模式 (Benchmark mode)</th><td>{_h(recipe.benchmark_mode)}</td></tr>"
            f"<tr><th>行业约束 (Industry constraints)</th><td>{_h(recipe.industry_constraints)}</td></tr>"
            f"<tr><th>风格约束 (Style constraints)</th><td>{_h(recipe.style_constraints)}</td></tr>"
            f"<tr><th>换手率惩罚 (Turnover penalty)</th><td>{_h(recipe.turnover_penalty_settings)}</td></tr>"
            f"<tr><th>再平衡频率 (Rebalance frequency)</th><td>{_h(recipe.rebalance_frequency)}</td></tr>"
            f"<tr><th>交易成本假设 (Transaction cost assumptions)</th><td>{_h(recipe.transaction_cost_assumptions)}</td></tr>"
            f"<tr><th>股票池定义 (Universe definition)</th><td>{_h(recipe.universe_definition)}</td></tr>"
            f"<tr><th>仓位/暴露约束 (Position limits / exposure controls)</th><td>{_h(recipe.position_limits)}</td></tr>"
            f"<tr><th>因子贡献 (Factor contributions)</th><td>{_h('; '.join(recipe.factor_contributions) if recipe.factor_contributions else _ui('na'))}</td></tr>"
            f"<tr><th>预期风险摘要 (Expected risk summary)</th><td>{_h(recipe.expected_risk_summary)}</td></tr>"
            f"<tr><th>预期收益代理 (Expected return proxy)</th><td>{_h(recipe.expected_return_proxy)}</td></tr>"
            f"<tr><th>优化器诊断 (Optimizer diagnostics)</th><td>{_h(recipe.optimizer_diagnostics)}</td></tr>"
            "</tbody></table>"
            "<h4>不可行配置告警 (Infeasible Configuration Warnings)</h4>"
            f"{warnings}"
            "</article>"
        )

    comparison_rows = "".join(
        "<tr>"
        f"<td><code>{_h(row.recipe_id)}</code></td>"
        f"<td>{_h(row.recipe_name)}</td>"
        f"<td>{_h(', '.join(row.selected_factors) if row.selected_factors else _ui('na'))}</td>"
        f"<td>{_h(', '.join(row.factor_family_mix) if row.factor_family_mix else _ui('na'))}</td>"
        f"<td>{_h(row.objective_tag)}</td>"
        f"<td>{_h(_display_text(row.construction_style))}</td>"
        f"<td>{_h(row.weighting_scheme)}</td>"
        f"<td>{_h(row.neutralization_constraints)}</td>"
        f"<td>{_h(row.turnover_penalty_settings)}</td>"
        f"<td>{_h(row.transaction_cost_assumptions)}</td>"
        f"<td>{_h(row.benchmark_mode)}</td>"
        f"<td>{_h(row.position_limits)}</td>"
        f"<td>{_h(row.expected_return_proxy)}</td>"
        f"<td>{_h(row.expected_risk_summary)}</td>"
        f"<td>{_h(_fmt(row.sharpe))}</td>"
        f"<td>{_h(_fmt_pct(row.annualized_return))}</td>"
        f"<td>{_h(_fmt_pct(row.max_drawdown))}</td>"
        f"<td>{_h(_fmt(row.information_ratio))}</td>"
        f"<td>{_h(_fmt_pct(row.post_cost_return))}</td>"
        "</tr>"
        for row in recipe_comparison.rows
    )
    leaderboard_rows = "".join(
        "<tr>"
        f"<td>{_h(_display_text(entry.objective))}</td>"
        f"<td>{_h(str(entry.rank))}</td>"
        f"<td><code>{_h(entry.recipe_id)}</code></td>"
        f"<td>{_h(entry.recipe_name)}</td>"
        f"<td>{_h(_fmt(entry.metric_value))}</td>"
        "</tr>"
        for entry in recipe_comparison.leaderboards
    )
    head_to_head_rows = "".join(
        "<tr>"
        f"<td>{_h(_display_text(insight.objective))}</td>"
        f"<td><code>{_h(insight.winner_recipe_id)}</code></td>"
        f"<td><code>{_h(insight.loser_recipe_id)}</code></td>"
        f"<td>{_h(_display_text(insight.summary))}</td>"
        f"<td>{_h(_display_text('; '.join(insight.reasons) if insight.reasons else _ui('na')))}</td>"
        "</tr>"
        for insight in recipe_comparison.head_to_head
    )
    grouping = "".join(
        f"<li><code>{_h(_display_text(label))}</code>: {_h(str(count))} 个配方 (recipes)</li>"
        for label, count in recipe_comparison.grouping_summary
    )

    return (
        f"<h2>{_h(_ui('section_portfolio_construction'))}</h2>"
        '<p class="section-note">组合配方是一等对象，并与已验证因子直接关联。 (Portfolio recipes are first-class objects and directly tied to validated factors.)</p>'
        f"{''.join(cards) if cards else '<p class="section-note">暂无组合配方 (No portfolio recipe available).</p>'}"
        '<article class="content-card reveal">'
        "<h3>流程第4步：配方对比层 (Workflow Step 4: Recipe Comparison Layer, Canonical Portfolio Recipe + Backtest)</h3>"
        '<div class="table-wrap">'
        "<table>"
        "<thead><tr>"
        "<th>配方ID (Recipe ID)</th><th>名称 (Name)</th><th>入选因子 (Selected Factors)</th><th>家族构成 (Factor Family Mix)</th><th>目标标签 (Objective Tag)</th>"
        "<th>构建风格 (Construction Style)</th><th>权重 (Weighting)</th><th>中性化 (Neutralization)</th><th>换手率惩罚 (Turnover Penalty)</th>"
        "<th>交易成本 (Tx Cost)</th><th>基准 (Benchmark)</th><th>仓位约束 (Position Limits)</th><th>预期收益 (Expected Return)</th><th>预期风险 (Expected Risk)</th>"
        "<th>Sharpe</th><th>年化收益 (Ann Return)</th><th>最大回撤 (Max Drawdown)</th><th>信息比率 (IR)</th><th>成本后收益 (Post-cost Return)</th>"
        "</tr></thead>"
        f"<tbody>{comparison_rows}</tbody>"
        "</table>"
        "</div>"
        "</article>"
        '<div class="two-col">'
        '<article class="content-card reveal">'
        "<h3>配方排行榜 (Recipe Leaderboard: Sharpe / Return / MDD / IR / Post-cost)</h3>"
        '<div class="table-wrap">'
        "<table><thead><tr><th>目标 (Objective)</th><th>排名 (Rank)</th><th>配方ID (Recipe ID)</th><th>配方名称 (Recipe Name)</th><th>指标 (Metric)</th></tr></thead>"
        f"<tbody>{leaderboard_rows}</tbody></table>"
        "</div>"
        "</article>"
        '<article class="content-card reveal">'
        "<h3>为何配方A优于配方B (Why Recipe A Beats Recipe B)</h3>"
        '<div class="table-wrap">'
        "<table><thead><tr><th>目标 (Objective)</th><th>赢家 (Winner)</th><th>落后者 (Loser)</th><th>摘要 (Summary)</th><th>原因 (Reasons)</th></tr></thead>"
        f"<tbody>{head_to_head_rows or '<tr><td colspan="5">{}</td></tr>'.format(_ui('na'))}</tbody></table>"
        "</div>"
        "<h3>配方分组摘要 (Recipe Grouping Summary)</h3>"
        f"<ul>{grouping or '<li>{}</li>'.format(_ui('na'))}</ul>"
        "</article>"
        "</div>"
    )


def _render_backtest_evaluation(backtests: tuple[PortfolioBacktestSummary, ...]) -> str:
    cards: list[str] = []
    for item in backtests:
        metrics_table = (
            '<table class="mini-table"><tbody>'
            f"<tr><th>年化收益 (Annualized Return)</th><td>{_h(_fmt_pct(item.annualized_return))}</td></tr>"
            f"<tr><th>年化波动 (Annualized Volatility)</th><td>{_h(_fmt_pct(item.annualized_volatility))}</td></tr>"
            f"<tr><th>Sharpe</th><td>{_h(_fmt(item.sharpe))}</td></tr>"
            f"<tr><th>Sortino</th><td>{_h(_fmt(item.sortino))}</td></tr>"
            f"<tr><th>最大回撤 (Max Drawdown)</th><td>{_h(_fmt_pct(item.max_drawdown))}</td></tr>"
            f"<tr><th>Calmar</th><td>{_h(_fmt(item.calmar))}</td></tr>"
            f"<tr><th>胜率 (Win Rate)</th><td>{_h(_fmt_pct(item.win_rate))}</td></tr>"
            f"<tr><th>换手率 (Turnover)</th><td>{_h(_fmt(item.turnover))}</td></tr>"
            f"<tr><th>信息比率 (Information Ratio)</th><td>{_h(_fmt(item.information_ratio))}</td></tr>"
            f"<tr><th>相对基准超额收益 (Excess Return vs Benchmark)</th><td>{_h(_fmt_pct(item.excess_return_vs_benchmark))}</td></tr>"
            f"<tr><th>跟踪误差 (Tracking Error)</th><td>{_h(_fmt_pct(item.tracking_error))}</td></tr>"
            f"<tr><th>成本前/后收益 (Pre-cost vs Post-cost)</th><td>{_h(_fmt(item.pre_cost_return))} / {_h(_fmt(item.post_cost_return))}</td></tr>"
            f"<tr><th>滚动Sharpe (Rolling Sharpe)</th><td>{_h(_fmt(item.rolling_sharpe))}</td></tr>"
            f"<tr><th>滚动回撤 (Rolling Drawdown)</th><td>{_h(_fmt_pct(item.rolling_drawdown))}</td></tr>"
            "</tbody></table>"
        )

        monthly_rows = "".join(
            f"<tr><td>{_h(month)}</td><td>{_h(_fmt_pct(value))}</td></tr>"
            for month, value in item.monthly_return_table[:12]
        )
        drawdown_rows = "".join(
            f"<tr><td>{_h(label)}</td><td>{_h(_fmt_pct(value))}</td></tr>"
            for label, value in item.drawdown_table[:8]
        )

        cards.append(
            '<article class="content-card reveal">'
            f"<h3>{_h(item.recipe_id)} | factor <code>{_h(item.factor_id)}</code></h3>"
            f'<p><a href="#{_h(_lineage_anchor(item.factor_id))}">打开血缘/溯源条目 (Open lineage / provenance entry)</a></p>'
            f"{_render_nav_chart(item.nav_points)}"
            f"{metrics_table}"
            '<div class="two-col">'
            "<section>"
            "<h4>月度收益表 (Monthly Return Table)</h4>"
            '<table class="mini-table"><thead><tr><th>月份 (Month)</th><th>收益 (Return)</th></tr></thead>'
            f"<tbody>{monthly_rows or '<tr><td colspan="2">{}</td></tr>'.format(_ui('na'))}</tbody></table>"
            "</section>"
            "<section>"
            "<h4>回撤表 (Drawdown Table)</h4>"
            '<table class="mini-table"><thead><tr><th>日期 (Date)</th><th>回撤 (Drawdown)</th></tr></thead>'
            f"<tbody>{drawdown_rows or '<tr><td colspan="2">{}</td></tr>'.format(_ui('na'))}</tbody></table>"
            "</section>"
            "</div>"
            "<h4>分阶段/状态/归因 (Subperiod / Regime / Attribution)</h4>"
            f"<p>{_h(_display_text(item.subperiod_analysis))}</p>"
            f"<p>{_h(_display_text(item.regime_analysis))}</p>"
            f"<p>{_h(_display_text(item.period_by_period_attribution))}</p>"
            "<h4>组合构成/交易统计/容量 (Portfolio Composition / Trade Stats / Capacity)</h4>"
            f"<p>{_h(_display_text(item.portfolio_composition_snapshot))}</p>"
            f"<p>{_h(_display_text(item.trade_statistics))}</p>"
            f"<p>{_h(_display_text(item.capacity_implementability_notes))}</p>"
            "</article>"
        )

    return (
        f"<h2>{_h(_ui('section_backtest_evaluation'))}</h2>"
        '<p class="section-note">'
        "主输出模块：评估最终策略结果、成本前后表现、回撤与稳定性。"
        " (Primary output module: evaluate final strategy outcomes, pre/post cost behavior, drawdowns, and stability.)"
        "</p>"
        f"{''.join(cards) if cards else '<p class="section-note">暂无回测评估结果 (No backtest evaluation available).</p>'}"
    )


def _render_lineage_registry(registry: ResearchLineageRegistry) -> str:
    entry_rows = "".join(
        "<tr "
        f'id="{_h(_lineage_anchor(entry.case_name))}">'
        f"<td><code>{_h(entry.case_name)}</code></td>"
        f"<td>{_h(entry.profile_name)}</td>"
        f"<td><code>{_h(entry.run_id)}</code></td>"
        f"<td>{_h(entry.run_timestamp_utc or _ui('na'))}</td>"
        f"<td><code>{_h(entry.factor_id)}</code></td>"
        f"<td><code>{_h(entry.recipe_id)}</code></td>"
        f"<td><code>{_h(entry.backtest_id)}</code></td>"
        f"<td><code>{_h(entry.output_dir or _ui('na'))}</code></td>"
        f"<td>{_h('; '.join(entry.provenance_links) if entry.provenance_links else _ui('na'))}</td>"
        "</tr>"
        for entry in registry.entries
    )

    linkage_rows = "".join(
        "<tr>"
        f"<td><code>{_h(link.from_object)}</code></td>"
        f"<td>{_h(link.relation)}</td>"
        f"<td><code>{_h(link.to_object)}</code></td>"
        "</tr>"
        for link in registry.links
    )

    warning_rows = _render_line_list(
        list(registry.warnings),
        empty_text="暂无血缘告警 (No lineage warnings).",
        warning_mode=True,
    )

    return (
        f"<h2>{_h(_ui('section_lineage_registry'))}</h2>"
        '<p class="section-note">'
        "追踪规范对象流：factor_definition -> signal_validation -> portfolio_recipe -> backtest_result。"
        " (Trace canonical object flow.)"
        "</p>"
        '<div class="table-wrap">'
        "<table>"
        "<thead><tr>"
        "<th>案例 (Case)</th><th>Profile</th><th>运行ID (Run ID)</th><th>运行时间 (Run Timestamp)</th><th>因子 (Factor)</th><th>配方 (Recipe)</th><th>回测 (Backtest)</th><th>输出目录 (Output Dir)</th><th>溯源链接 (Provenance Links)</th>"
        "</tr></thead>"
        f"<tbody>{entry_rows or '<tr><td colspan="9">{}</td></tr>'.format(_ui('na'))}</tbody>"
        "</table>"
        "</div>"
        "<h3>血缘链接 (Lineage Links)</h3>"
        '<div class="table-wrap">'
        "<table>"
        "<thead><tr><th>来源 (From)</th><th>关系 (Relation)</th><th>目标 (To)</th></tr></thead>"
        f"<tbody>{linkage_rows or '<tr><td colspan="3">{}</td></tr>'.format(_ui('na'))}</tbody>"
        "</table>"
        "</div>"
        "<h3>血缘告警 (Lineage Warnings)</h3>"
        f"{warning_rows}"
    )


def _render_robustness(rows: tuple[RobustnessSummary, ...]) -> str:
    table_rows = "".join(
        "<tr>"
        f"<td><code>{_h(item.factor_id)}</code></td>"
        f"<td>{_h(item.parameter_sensitivity)}</td>"
        f"<td>{_h(item.lookback_sensitivity)}</td>"
        f"<td>{_h(item.universe_sensitivity)}</td>"
        f"<td>{_h(item.rebalance_sensitivity)}</td>"
        f"<td>{_h(item.transaction_cost_sensitivity)}</td>"
        f"<td>{_h(item.profile_sensitivity)}</td>"
        f"<td>{_h(item.leakage_checks)}</td>"
        f"<td>{_h(item.survivorship_pit_checks)}</td>"
        f"<td>{_h(_display_text('; '.join(item.implementation_warnings) if item.implementation_warnings else _ui('na')))}</td>"
        f"<td>{_h(item.robustness_verdict)}</td>"
        "</tr>"
        for item in rows
    )

    return (
        f"<h2>{_h(_ui('section_robustness'))}</h2>"
        '<p class="section-note">次级模块：在因子/组合/回测审阅后，用于识别脆弱性和审计风险。'
        " (Secondary module after factor/portfolio/backtest review.)</p>"
        '<div class="table-wrap">'
        "<table>"
        "<thead><tr>"
        "<th>因子 (Factor)</th><th>参数敏感性 (Parameter Sensitivity)</th><th>回看敏感性 (Lookback Sensitivity)</th><th>股票池敏感性 (Universe Sensitivity)</th>"
        "<th>再平衡敏感性 (Rebalance Sensitivity)</th><th>交易成本敏感性 (Tx-Cost Sensitivity)</th><th>Profile敏感性 (Profile Sensitivity)</th>"
        "<th>泄漏检查 (Leakage Checks)</th><th>生存偏差/PIT (Survivorship/PIT)</th><th>实现告警 (Implementation Warnings)</th><th>结论 (Verdict)</th>"
        "</tr></thead>"
        f"<tbody>{table_rows}</tbody>"
        "</table>"
        "</div>"
    )


def _render_line_list(
    lines: tuple[str, ...] | list[str],
    *,
    empty_text: str,
    warning_mode: bool = False,
) -> str:
    if not lines:
        return f'<p class="section-note">{_h(_display_text(empty_text))}</p>'
    cls = "line-list line-list-warning" if warning_mode else "line-list"
    row_html = "".join(f"<li>{_h(_display_text(line))}</li>" for line in lines)
    return f'<ul class="{cls}">{row_html}</ul>'


def _render_correlation_matrix(
    matrix: tuple[tuple[str, tuple[tuple[str, float | None], ...]], ...],
) -> str:
    if not matrix:
        return '<p class="section-note">暂无相关性矩阵（因子取值文件不可用） (No correlation matrix available).</p>'

    names = [name for name, _ in matrix]
    rows = []
    lookup = {name: dict(values) for name, values in matrix}
    for row_name in names:
        cells = [f"<th><code>{_h(row_name)}</code></th>"]
        for col_name in names:
            value = lookup.get(row_name, {}).get(col_name)
            text = "N/A" if value is None else f"{value:.2f}"
            cls = "corr-cell"
            if (
                value is not None
                and abs(value) >= _DEFAULT_SHORTLIST_CONFIG.redundancy_correlation_max
                and row_name != col_name
            ):
                cls += " corr-cell-high"
            cells.append(f'<td class="{cls}">{_h(text)}</td>')
        rows.append(f"<tr>{''.join(cells)}</tr>")

    head_cells = "".join(f"<th><code>{_h(name)}</code></th>" for name in names)
    return (
        '<div class="table-wrap">'
        '<table class="corr-table">'
        f"<thead><tr><th></th>{head_cells}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
        "</div>"
    )


def _render_nav_chart(nav_points: tuple[tuple[str, float], ...]) -> str:
    if len(nav_points) < 2:
        return '<p class="section-note">当前序列无法绘制累计收益/NAV曲线 (Cumulative return / NAV chart unavailable).</p>'

    values = [value for _, value in nav_points]
    x_count = len(values)
    y_min = min(values)
    y_max = max(values)
    span = y_max - y_min if y_max > y_min else 1.0
    width = 520
    height = 160
    points: list[str] = []
    for idx, value in enumerate(values):
        x = (idx / max(1, x_count - 1)) * width
        y = height - ((value - y_min) / span) * height
        points.append(f"{x:.2f},{y:.2f}")

    return (
        '<div class="nav-chart">'
        "<h4>累计收益/NAV曲线 (Cumulative Return / NAV Chart)</h4>"
        f'<svg viewBox="0 0 {width} {height}" preserveAspectRatio="none">'
        f'<polyline points="{" ".join(points)}" class="nav-line" />'
        "</svg>"
        "</div>"
    )


def _dashboard_css() -> str:
    return """
:root {
  --bg: #f4f6f8;
  --bg-alt: #e8eef2;
  --ink: #0f1a21;
  --muted: #4b5f6b;
  --line: #c7d5de;
  --brand: #0f766e;
  --brand-soft: #d8f2ef;
  --accent: #b45309;
  --warn: #c2410c;
  --warn-soft: #ffedd5;
  --card: #ffffff;
  --shadow: 0 12px 40px rgba(15, 26, 33, 0.08);
}

* { box-sizing: border-box; }

body {
  margin: 0;
  color: var(--ink);
  background: radial-gradient(circle at top right, #dceff4 0%, var(--bg) 48%, #f8fafb 100%);
  font-family: "MiSans", "PingFang SC", "Microsoft YaHei UI", "Segoe UI", sans-serif;
  line-height: 1.5;
}

.page-backdrop {
  position: fixed;
  inset: -20% -10% auto auto;
  width: 36rem;
  height: 36rem;
  background: radial-gradient(circle, rgba(15, 118, 110, 0.14) 0%, rgba(15, 118, 110, 0) 72%);
  pointer-events: none;
}

.page-wrap {
  max-width: 1440px;
  margin: 0 auto;
  padding: 20px 18px 44px;
}

.hero {
  background: linear-gradient(135deg, #0f1720, #143042 72%);
  color: #eff7fb;
  border-radius: 18px;
  padding: 20px 22px;
  box-shadow: var(--shadow);
}

.hero h1 {
  margin: 0;
  font-family: "MiSans", "PingFang SC", "Microsoft YaHei UI", sans-serif;
  letter-spacing: 0.01em;
}

.subtitle {
  margin: 8px 0 10px;
  color: #cfe5f3;
}

.meta-line {
  margin: 4px 0;
  color: #d4e8f5;
}

code {
  background: #edf3f7;
  padding: 1px 5px;
  border-radius: 6px;
  font-family: "Fira Code", "Cascadia Code", monospace;
  color: #0d3b47;
}

.quick-nav {
  margin: 14px 0 18px;
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.quick-nav a {
  text-decoration: none;
  color: #0e4f60;
  background: #e4f4f6;
  border: 1px solid #b7dce0;
  padding: 6px 10px;
  border-radius: 999px;
  font-size: 0.9rem;
}

.quick-nav a:hover {
  background: #d0ecef;
}

.panel {
  margin-top: 14px;
  background: var(--card);
  border: 1px solid #d4e0e8;
  border-radius: 14px;
  padding: 16px;
  box-shadow: 0 6px 20px rgba(8, 24, 35, 0.05);
  animation: fade-up 260ms ease;
}

.panel-secondary {
  border-top: 4px solid #c0cdd7;
  opacity: 0.96;
}

h2 {
  margin: 0 0 8px;
  font-family: "MiSans", "PingFang SC", "Microsoft YaHei UI", sans-serif;
  font-size: 1.35rem;
}

h3 {
  margin: 0 0 8px;
  font-size: 1.02rem;
}

h4 {
  margin: 0 0 6px;
  font-size: 0.96rem;
}

.section-note {
  margin: 6px 0 10px;
  color: var(--muted);
}

.summary-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 10px;
}

.summary-card {
  background: linear-gradient(160deg, var(--brand-soft), #f4fbfb);
  border: 1px solid #b8e3de;
  border-radius: 12px;
  padding: 10px;
}

.summary-label {
  color: #17595d;
  font-size: 0.84rem;
}

.summary-value {
  margin-top: 3px;
  font-size: 1.4rem;
  font-weight: 700;
}

.two-col {
  margin-top: 10px;
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
  gap: 12px;
}

.content-card {
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 10px;
  background: #f9fcfe;
}

.content-card-warn {
  border-color: #f4b58b;
  background: var(--warn-soft);
}

.filter-bar {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin: 6px 0 10px;
}

.filter-bar input,
.filter-bar select {
  padding: 7px 9px;
  border-radius: 8px;
  border: 1px solid #bbccd8;
  font: inherit;
  min-width: 170px;
}

.count-chip {
  align-self: center;
  background: #e7f4f3;
  color: #175f58;
  border: 1px solid #b7ddd6;
  border-radius: 999px;
  padding: 5px 10px;
  font-size: 0.84rem;
}

.table-wrap {
  overflow-x: auto;
}

table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.9rem;
}

th,
td {
  border-bottom: 1px solid #dde7ed;
  text-align: left;
  padding: 7px 8px;
  vertical-align: top;
}

th {
  background: #eef4f8;
  position: sticky;
  top: 0;
  z-index: 1;
}

.mini-table th {
  width: 42%;
  position: static;
}

.line-list {
  margin: 0;
  padding-left: 18px;
}

.line-list-warning li {
  color: #8f2d00;
}

.factor-detail-card {
  margin-top: 12px;
  border: 1px solid #d7e4ea;
  border-radius: 12px;
  padding: 12px;
  background: #fbfdfe;
}

.detail-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
  gap: 10px;
}

.detail-block {
  border: 1px solid #d9e6ed;
  border-radius: 10px;
  padding: 10px;
  background: #fdfefe;
}

.corr-table td,
.corr-table th {
  text-align: center;
  min-width: 84px;
}

.corr-cell-high {
  background: #ffe8d8;
  color: #8f2f00;
  font-weight: 700;
}

.nav-chart {
  margin: 8px 0 10px;
  border: 1px solid #d5e4ea;
  border-radius: 10px;
  background: #fcffff;
  padding: 8px;
}

.nav-chart svg {
  width: 100%;
  height: 170px;
  display: block;
}

.nav-line {
  fill: none;
  stroke: #0b7d74;
  stroke-width: 2.8;
  stroke-linecap: round;
  stroke-linejoin: round;
}

.reveal {
  animation: reveal 420ms ease;
}

@keyframes fade-up {
  from {
    opacity: 0.78;
    transform: translateY(5px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

@keyframes reveal {
  from {
    opacity: 0;
    transform: translateY(8px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

@media (max-width: 860px) {
  .page-wrap {
    padding: 14px 10px 32px;
  }

  .hero {
    padding: 16px;
  }

  .filter-bar input,
  .filter-bar select {
    min-width: 140px;
  }

  table {
    font-size: 0.82rem;
  }
}
"""


def _dashboard_js() -> str:
    return """
const factorSearchInput = document.getElementById("factor-search-input");
const factorFamilyFilter = document.getElementById("factor-family-filter");
const factorStatusFilter = document.getElementById("factor-status-filter");
const factorSortSelect = document.getElementById("factor-sort-select");
const factorTable = document.getElementById("factor-library-table");
const factorEmpty = document.getElementById("factor-library-empty");
const factorCount = document.getElementById("factor-library-count");

const applyFactorLibraryFilters = () => {
  if (!factorTable) {
    return;
  }
  const tbody = factorTable.querySelector("tbody");
  if (!tbody) {
    return;
  }
  const rows = Array.from(tbody.querySelectorAll("tr"));
  const search = factorSearchInput ? (factorSearchInput.value || "").toLowerCase().trim() : "";
  const family = factorFamilyFilter ? (factorFamilyFilter.value || "") : "";
  const status = factorStatusFilter ? (factorStatusFilter.value || "") : "";
  const sortMode = factorSortSelect ? (factorSortSelect.value || "signal_desc") : "signal_desc";

  const visibleRows = rows.filter((row) => {
    const rowName = row.dataset.factorName || "";
    const rowId = row.dataset.factorId || "";
    const rowFamily = row.dataset.family || "";
    const rowStatus = row.dataset.status || "";
    const matchesSearch = !search || rowName.includes(search) || rowId.includes(search);
    const matchesFamily = !family || rowFamily === family;
    const matchesStatus = !status || rowStatus === status;
    const visible = matchesSearch && matchesFamily && matchesStatus;
    row.style.display = visible ? "" : "none";
    return visible;
  });

  const sortedRows = visibleRows.slice().sort((a, b) => {
    if (sortMode === "name_asc") {
      return (a.dataset.factorName || "").localeCompare(b.dataset.factorName || "");
    }
    if (sortMode === "coverage_desc") {
      return Number(b.dataset.coverage || "-999") - Number(a.dataset.coverage || "-999");
    }
    return Number(b.dataset.score || "-999") - Number(a.dataset.score || "-999");
  });

  sortedRows.forEach((row) => {
    tbody.appendChild(row);
  });

  if (factorCount) {
    factorCount.textContent = `${sortedRows.length} 可见 (visible)`;
  }
  if (factorEmpty) {
    factorEmpty.style.display = sortedRows.length > 0 ? "none" : "";
  }
};

if (factorSearchInput) {
  factorSearchInput.addEventListener("input", applyFactorLibraryFilters);
}
if (factorFamilyFilter) {
  factorFamilyFilter.addEventListener("change", applyFactorLibraryFilters);
}
if (factorStatusFilter) {
  factorStatusFilter.addEventListener("change", applyFactorLibraryFilters);
}
if (factorSortSelect) {
  factorSortSelect.addEventListener("change", applyFactorLibraryFilters);
}

applyFactorLibraryFilters();
"""


def _select_profile_payload(
    profiles_payload: dict[str, object],
    *,
    preferred_profile: str,
    ordered_profiles: list[str],
) -> tuple[str | None, dict[str, object]]:
    preferred = as_object_dict(profiles_payload.get(preferred_profile))
    if preferred:
        return preferred_profile, preferred

    for profile_name in ordered_profiles:
        payload = as_object_dict(profiles_payload.get(profile_name))
        if payload:
            return profile_name, payload

    for profile_name, payload_obj in profiles_payload.items():
        payload = as_object_dict(payload_obj)
        if payload:
            return str(profile_name), payload

    return None, {}


def _load_case_artifacts(
    profile_payload: dict[str, object],
    *,
    case_name: str = "unknown_case",
    profile_name: str = "unknown_profile",
    load_policy: ArtifactLoadPolicy | None = None,
    diagnostics: list[ArtifactLoadDiagnostic] | None = None,
    warnings: list[str] | None = None,
    errors: list[str] | None = None,
) -> _CaseArtifacts:
    resolved_policy = load_policy or _build_artifact_load_policy(
        artifact_load_mode=_DEFAULT_ARTIFACT_LOAD_MODE,
        prefer_persisted_workflow_artifacts=True,
    )
    diagnostic_rows = diagnostics if diagnostics is not None else []
    warning_rows = warnings if warnings is not None else []
    error_rows = errors if errors is not None else []
    path_payload = as_object_dict(profile_payload.get("artifact_paths"))
    output_dir = _to_path(path_payload.get("output_dir"))

    def _case_artifact_path(payload_key: str, fallback_name: str) -> Path | None:
        payload_value = _to_path(path_payload.get(payload_key))
        if payload_value is not None:
            return payload_value
        if output_dir is None:
            return None
        return output_dir / fallback_name

    metrics_path = _case_artifact_path("metrics_path", "metrics.json")
    manifest_path = _case_artifact_path("run_manifest_path", "run_manifest.json")
    factor_definition_path = _case_artifact_path(
        "factor_definition_json_path",
        "factor_definition.json",
    )
    signal_validation_path = _case_artifact_path(
        "signal_validation_json_path",
        "signal_validation.json",
    )
    portfolio_recipe_path = _case_artifact_path(
        "portfolio_recipe_json_path",
        "portfolio_recipe.json",
    )
    backtest_result_path = _case_artifact_path(
        "backtest_result_json_path",
        "backtest_result.json",
    )

    integrity_path = output_dir / "integrity_report.json" if output_dir is not None else None
    coverage_path = output_dir / "coverage.csv" if output_dir is not None else None
    group_returns_path = output_dir / "group_returns.csv" if output_dir is not None else None
    ic_path = output_dir / "ic_timeseries.csv" if output_dir is not None else None
    turnover_path = output_dir / "turnover.csv" if output_dir is not None else None
    rolling_path = output_dir / "rolling_stability.csv" if output_dir is not None else None

    legacy_metrics_payload = _load_optional_json(metrics_path)
    manifest = _load_optional_json(manifest_path)
    status = safe_text(profile_payload.get("status")) or "success"
    canonical_required = resolved_policy.require_canonical_artifacts and status == "success"
    factor_definition_payload = _load_case_artifact_json_payload(
        path=factor_definition_path,
        artifact_name="factor_definition.json",
        object_label="factor_definition",
        case_name=case_name,
        profile_name=profile_name,
        required=canonical_required,
        allow_fallback=resolved_policy.allow_legacy_case_fallback,
        mode=resolved_policy.mode,
        diagnostics=diagnostic_rows,
        warnings=warning_rows,
        errors=error_rows,
    )
    signal_validation_payload = _load_case_artifact_json_payload(
        path=signal_validation_path,
        artifact_name="signal_validation.json",
        object_label="signal_validation",
        case_name=case_name,
        profile_name=profile_name,
        required=canonical_required,
        allow_fallback=resolved_policy.allow_legacy_case_fallback,
        mode=resolved_policy.mode,
        diagnostics=diagnostic_rows,
        warnings=warning_rows,
        errors=error_rows,
    )
    portfolio_recipe_payload = _load_case_artifact_json_payload(
        path=portfolio_recipe_path,
        artifact_name="portfolio_recipe.json",
        object_label="portfolio_recipe",
        case_name=case_name,
        profile_name=profile_name,
        required=canonical_required,
        allow_fallback=resolved_policy.allow_legacy_case_fallback,
        mode=resolved_policy.mode,
        diagnostics=diagnostic_rows,
        warnings=warning_rows,
        errors=error_rows,
    )
    backtest_result_payload = _load_case_artifact_json_payload(
        path=backtest_result_path,
        artifact_name="backtest_result.json",
        object_label="backtest_result",
        case_name=case_name,
        profile_name=profile_name,
        required=canonical_required,
        allow_fallback=resolved_policy.allow_legacy_case_fallback,
        mode=resolved_policy.mode,
        diagnostics=diagnostic_rows,
        warnings=warning_rows,
        errors=error_rows,
    )
    integrity_report = _load_optional_json(integrity_path)

    metrics_payload: dict[str, object] = dict(legacy_metrics_payload)
    for key in ("metrics", "coverage_by_date_summary", "neutralization_summary"):
        if key in signal_validation_payload:
            metrics_payload[key] = signal_validation_payload[key]
    metrics = as_object_dict(metrics_payload.get("metrics"))

    portfolio_validation_summary = as_object_dict(
        portfolio_recipe_payload.get("portfolio_validation_summary")
    ) or as_object_dict(metrics_payload.get("portfolio_validation_summary"))
    portfolio_validation_metrics = as_object_dict(
        portfolio_recipe_payload.get("portfolio_validation_metrics")
    ) or as_object_dict(metrics_payload.get("portfolio_validation_metrics"))
    portfolio_validation_package = as_object_dict(
        portfolio_recipe_payload.get("portfolio_validation_package")
    ) or as_object_dict(metrics_payload.get("portfolio_validation_package"))

    spec = as_object_dict(factor_definition_payload.get("spec")) or as_object_dict(
        manifest.get("spec")
    )
    factor_series = _load_factor_series(_to_path(spec.get("factor_path")))

    fallback_derived_fields: dict[str, tuple[str, ...]] = {}
    for artifact_name, payload in (
        ("signal_validation.json", signal_validation_payload),
        ("portfolio_recipe.json", portfolio_recipe_payload),
        ("backtest_result.json", backtest_result_payload),
    ):
        fields = tuple(
            parse_text_list(payload.get("fallback_derived_fields"), split_semicolon=False)
        )
        if fields:
            fallback_derived_fields[artifact_name] = fields

    artifact_paths: dict[str, Path] = {}
    for key, value in path_payload.items():
        path = _to_path(value)
        if path is not None:
            artifact_paths[key] = path
    if output_dir is not None:
        for name in [
            "summary.md",
            "experiment_card.md",
            "factor_definition.json",
            "signal_validation.json",
            "portfolio_recipe.json",
            "backtest_result.json",
            "factor_definition.yaml",
            "level2_portfolio_validation/portfolio_validation_summary.json",
            "level2_portfolio_validation/portfolio_validation_metrics.json",
            "level2_portfolio_validation/portfolio_validation_package.json",
            "level2_portfolio_validation/portfolio_validation_package.md",
            "integrity_report.json",
            "case_report.md",
        ]:
            path = output_dir / name
            if path.exists():
                artifact_paths[name] = path

    return _CaseArtifacts(
        output_dir=output_dir,
        metrics_payload=metrics_payload,
        metrics=metrics,
        factor_definition_payload=factor_definition_payload,
        signal_validation_payload=signal_validation_payload,
        portfolio_recipe_payload=portfolio_recipe_payload,
        backtest_result_payload=backtest_result_payload,
        fallback_derived_fields=fallback_derived_fields,
        portfolio_validation_summary=portfolio_validation_summary,
        portfolio_validation_metrics=portfolio_validation_metrics,
        portfolio_validation_package=portfolio_validation_package,
        manifest=manifest,
        integrity_report=integrity_report,
        coverage_df=_load_optional_csv(coverage_path),
        group_returns_df=_load_optional_csv(group_returns_path),
        ic_df=_load_optional_csv(ic_path),
        turnover_df=_load_optional_csv(turnover_path),
        rolling_df=_load_optional_csv(rolling_path),
        factor_series=factor_series,
        artifact_paths=artifact_paths,
    )


def _load_case_artifact_json_payload(
    *,
    path: Path | None,
    artifact_name: str,
    object_label: str,
    case_name: str,
    profile_name: str,
    required: bool,
    allow_fallback: bool,
    mode: ArtifactLoadMode,
    diagnostics: list[ArtifactLoadDiagnostic],
    warnings: list[str],
    errors: list[str],
) -> dict[str, object]:
    context = f"{case_name} ({profile_name}) {object_label}"
    fallback_used = allow_fallback and not required
    if path is None:
        _append_artifact_issue(
            code=_MISSING_CANONICAL_ARTIFACT_CODE,
            severity=_artifact_issue_severity(required=required),
            artifact_type="canonical_artifact",
            object_scope=object_label,
            message=f"{context}: missing artifact path ({artifact_name})",
            diagnostics=diagnostics,
            warnings=warnings,
            errors=errors,
            mode=mode,
            case_name=case_name,
            profile_name=profile_name,
            fallback_used=fallback_used,
            remediation_hint=(f"Persist {artifact_name} and ensure artifact_paths points to it."),
        )
        if fallback_used:
            _append_artifact_issue(
                code=_FALLBACK_USED_CODE,
                severity="warning",
                artifact_type="canonical_artifact",
                object_scope=object_label,
                message=(
                    f"{context}: fallback used because canonical artifact "
                    f"{artifact_name} is unavailable"
                ),
                diagnostics=diagnostics,
                warnings=warnings,
                errors=errors,
                mode=mode,
                case_name=case_name,
                profile_name=profile_name,
                fallback_used=True,
                remediation_hint=(f"Generate canonical {artifact_name} to disable this fallback."),
            )
        return {}
    payload = _load_optional_json(path)
    if not payload:
        _append_artifact_issue(
            code=_MISSING_CANONICAL_ARTIFACT_CODE,
            severity=_artifact_issue_severity(required=required),
            artifact_type="canonical_artifact",
            object_scope=object_label,
            message=f"{context}: artifact missing or unreadable at {path}",
            diagnostics=diagnostics,
            warnings=warnings,
            errors=errors,
            mode=mode,
            path=path,
            case_name=case_name,
            profile_name=profile_name,
            fallback_used=fallback_used,
            remediation_hint=(f"Regenerate {artifact_name} and ensure the file is readable."),
        )
        if fallback_used:
            _append_artifact_issue(
                code=_FALLBACK_USED_CODE,
                severity="warning",
                artifact_type="canonical_artifact",
                object_scope=object_label,
                message=(
                    f"{context}: fallback used because canonical artifact "
                    f"{artifact_name} could not be loaded"
                ),
                diagnostics=diagnostics,
                warnings=warnings,
                errors=errors,
                mode=mode,
                path=path,
                case_name=case_name,
                profile_name=profile_name,
                fallback_used=True,
                remediation_hint=(
                    f"Regenerate canonical {artifact_name} to disable this fallback."
                ),
            )
        return {}
    try:
        validate_level12_artifact_payload(
            payload,
            artifact_name=artifact_name,
            source=path,
        )
    except Exception as exc:
        _append_artifact_issue(
            code=_INVALID_CANONICAL_ARTIFACT_CODE,
            severity=_artifact_issue_severity(required=required),
            artifact_type="canonical_artifact",
            object_scope=object_label,
            message=f"{context}: invalid artifact payload ({artifact_name}): {exc}",
            diagnostics=diagnostics,
            warnings=warnings,
            errors=errors,
            mode=mode,
            path=path,
            case_name=case_name,
            profile_name=profile_name,
            remediation_hint=(
                f"Fix schema issues in {artifact_name} and re-run artifact generation."
            ),
        )
    return payload


def _build_validation_summary(
    *,
    metrics: dict[str, object],
    case_row: dict[str, object],
    artifacts: _CaseArtifacts,
) -> ValidationSummary:
    ic_mean = _safe_float(metrics.get("mean_ic"))
    rank_ic_mean = _safe_float(metrics.get("mean_rank_ic"))
    icir = _safe_float(metrics.get("ic_ir"))
    n_dates = _safe_float(metrics.get("n_dates_used"))
    t_stat_proxy = None
    if icir is not None and n_dates is not None and n_dates > 0:
        t_stat_proxy = icir * math.sqrt(n_dates)

    decay_profile = _decay_profile(artifacts.rolling_df)
    horizon_analysis = _horizon_analysis(artifacts.portfolio_validation_metrics)

    neutralization = as_object_dict(metrics.get("neutralization_comparison"))
    neutralization_delta = as_object_dict(neutralization.get("delta"))

    transition_reasons = parse_text_list(case_row.get("changed_fields"), split_semicolon=False)
    oos = (
        f"rolling_ic_positive_share={_fmt(_safe_float(metrics.get('rolling_ic_positive_share')))}; "
        f"rolling_long_short_positive_share={_fmt(_safe_float(metrics.get('rolling_long_short_positive_share')))}"
    )

    return ValidationSummary(
        ic_mean=ic_mean,
        rank_ic_mean=rank_ic_mean,
        icir=icir,
        t_stat_proxy=t_stat_proxy,
        hit_rate=_safe_float(metrics.get("long_short_hit_rate")),
        positive_ic_frequency=_safe_float(metrics.get("ic_positive_rate")),
        decay_profile=decay_profile,
        horizon_analysis=horizon_analysis,
        quantile_return_spread=_safe_float(metrics.get("mean_long_short_return")),
        long_short_performance_summary=(
            f"mean={_fmt(_safe_float(metrics.get('mean_long_short_return')))}, "
            f"ir={_fmt(_safe_float(metrics.get('long_short_ir')))}, "
            f"hit_rate={_fmt(_safe_float(metrics.get('long_short_hit_rate')))}"
        ),
        monotonicity_diagnostics=_monotonicity_note(artifacts.group_returns_df),
        regime_breakdown=(
            f"subperiod_ic_positive_share={_fmt(_safe_float(metrics.get('subperiod_ic_positive_share')))}, "
            f"subperiod_long_short_positive_share={_fmt(_safe_float(metrics.get('subperiod_long_short_positive_share')))}"
        ),
        industry_neutral_comparison=(
            f"delta mean IC={_fmt(_safe_float(neutralization_delta.get('mean_ic_delta')))}, "
            f"delta RankIC={_fmt(_safe_float(neutralization_delta.get('mean_rank_ic_delta')))}"
            if neutralization_delta
            else "N/A"
        ),
        size_neutral_comparison=(
            f"delta coverage={_fmt(_safe_float(neutralization_delta.get('eval_coverage_ratio_mean_delta')))}; "
            f"delta L/S={_fmt(_safe_float(neutralization_delta.get('mean_long_short_return_delta')))}"
            if neutralization_delta
            else "N/A"
        ),
        split_summary=(
            f"profile={safe_text(metrics.get('research_evaluation_profile')) or 'N/A'}; "
            f"n_dates={_fmt(n_dates)}"
        ),
        oos_stability_comparison=oos
        + (f"; changed_fields={len(transition_reasons)}" if transition_reasons else ""),
    )


def _related_artifacts(artifacts: _CaseArtifacts) -> tuple[str, ...]:
    rows = []
    for key, path in sorted(artifacts.artifact_paths.items(), key=lambda item: str(item[0])):
        rows.append(f"{key}: {path}")
    for artifact_name, fields in sorted(artifacts.fallback_derived_fields.items()):
        rows.append(f"{artifact_name}.fallback_derived_fields: {', '.join(fields)}")
    return tuple(rows)


def _portfolio_fragility_reasons(portfolio_summary: dict[str, object]) -> list[str]:
    robustness_summary = as_object_dict(portfolio_summary.get("portfolio_robustness_summary"))
    return parse_text_list(robustness_summary.get("fragility_reasons"), split_semicolon=False)


def _portfolio_support_reasons(portfolio_summary: dict[str, object]) -> list[str]:
    robustness_summary = as_object_dict(portfolio_summary.get("portfolio_robustness_summary"))
    return parse_text_list(robustness_summary.get("support_reasons"), split_semicolon=False)


def _to_reason_list(profile_payload: dict[str, object], *, key: str, nested_key: str) -> list[str]:
    nested = as_object_dict(profile_payload.get(key))
    return parse_text_list(nested.get(nested_key), split_semicolon=False)


def _concise_verdict(profile_payload: dict[str, object]) -> str:
    factor_verdict = safe_text(profile_payload.get("factor_verdict")) or "N/A"
    promotion_decision = safe_text(profile_payload.get("promotion_decision")) or "N/A"
    portfolio_recommendation = (
        safe_text(profile_payload.get("portfolio_validation_recommendation")) or "N/A"
    )
    return (
        f"factor_verdict={factor_verdict}; promotion={promotion_decision}; "
        f"portfolio={portfolio_recommendation}"
    )


def _signal_quality_score(
    *,
    icir: float | None,
    rank_ic_mean: float | None,
    long_short_ir: float | None,
    coverage_ratio: float | None,
) -> float | None:
    values = [
        (icir, 1.0),
        (rank_ic_mean, 0.7),
        (long_short_ir, 0.6),
        (coverage_ratio, 0.2),
    ]
    score = 0.0
    used = 0
    for value, weight in values:
        if value is None:
            continue
        score += value * weight
        used += 1
    if used == 0:
        return None
    return score


def _factor_sort_key(summary: FactorSummary) -> tuple[float, int, str]:
    signal = _float_or_default(summary.signal_quality_score, default=-999.0)
    status_rank = _STATUS_ORDER.get(summary.research_status, 999)
    return (-signal, status_rank, summary.factor_name)


def _factor_status(
    *,
    profile_status: str,
    factor_verdict: str,
    promotion_decision: str,
    portfolio_recommendation: str,
) -> str:
    if profile_status != "success":
        return "rejected"
    if portfolio_recommendation == "Credible at portfolio level":
        return "portfolio-active"
    if promotion_decision == "Promote to Level 2":
        return "validated"
    if factor_verdict == "Weak candidate":
        return "rejected"
    if factor_verdict == "Strong candidate":
        return "generated"
    return "draft"


def _factor_family_summary(
    factors: list[FactorSummary] | tuple[FactorSummary, ...],
) -> tuple[tuple[str, int], ...]:
    counter: dict[str, int] = {}
    for item in factors:
        counter[item.factor_family] = counter.get(item.factor_family, 0) + 1
    return tuple(sorted(counter.items(), key=lambda pair: (-pair[1], pair[0])))


def _factor_correlation_matrix(
    *,
    summaries: list[FactorSummary] | tuple[FactorSummary, ...],
    factor_series_lookup: dict[str, pd.Series],
) -> tuple[tuple[str, tuple[tuple[str, float | None], ...]], ...]:
    names = [item.factor_id for item in summaries]
    rows: list[tuple[str, tuple[tuple[str, float | None], ...]]] = []
    for row_name in names:
        row_series = factor_series_lookup.get(row_name)
        cell_rows: list[tuple[str, float | None]] = []
        for col_name in names:
            col_series = factor_series_lookup.get(col_name)
            corr: float | None
            if row_series is None or col_series is None:
                corr = None
            elif row_name == col_name:
                corr = 1.0
            else:
                merged = pd.concat([row_series, col_series], axis=1, join="inner").dropna()
                if len(merged) < 20:
                    corr = None
                else:
                    corr_value = merged.iloc[:, 0].corr(merged.iloc[:, 1])
                    corr = float(corr_value) if pd.notna(corr_value) else None
            cell_rows.append((col_name, corr))
        rows.append((row_name, tuple(cell_rows)))
    return tuple(rows)


def _shortlist_recommendations(
    *,
    summaries: list[FactorSummary] | tuple[FactorSummary, ...],
    correlation_matrix: tuple[tuple[str, tuple[tuple[str, float | None], ...]], ...],
) -> tuple[str, ...]:
    rows = [
        FactorComparisonRow(
            factor_id=item.factor_id,
            factor_name=item.factor_name,
            factor_family=item.factor_family,
        )
        for item in summaries
    ]
    shortlist = _build_factor_shortlist_result(
        comparison_rows=rows,
        correlation_matrix=correlation_matrix,
        config=_DEFAULT_SHORTLIST_CONFIG,
    )
    return shortlist.recommendation_summary


def _build_factor_shortlist_result(
    *,
    comparison_rows: list[FactorComparisonRow] | tuple[FactorComparisonRow, ...],
    correlation_matrix: tuple[tuple[str, tuple[tuple[str, float | None], ...]], ...],
    config: FactorShortlistConfig,
) -> FactorShortlistResult:
    if not comparison_rows:
        return FactorShortlistResult(config=config)

    corr_lookup = {name: dict(values) for name, values in correlation_matrix}

    scored_rows: list[tuple[FactorComparisonRow, float | None]] = []
    for row in comparison_rows:
        scored_rows.append((row, _factor_composite_score(row=row, config=config)))

    ranked_rows = sorted(
        scored_rows,
        key=lambda item: (
            -_float_or_default(item[1], default=-999.0),
            item[0].factor_name,
        ),
    )

    selected_keep: list[str] = []
    entries: list[FactorShortlistEntry] = []

    for rank, (row, score) in enumerate(ranked_rows, start=1):
        max_corr, redundant_with = _max_correlation_to_selected(
            factor_id=row.factor_id,
            selected_factor_ids=selected_keep,
            corr_lookup=corr_lookup,
        )
        recommendation, rationale = _factor_shortlist_recommendation(
            row=row,
            score=score,
            max_corr=max_corr,
            redundant_with=redundant_with,
            config=config,
        )
        if recommendation == "keep":
            selected_keep.append(row.factor_id)
        entries.append(
            FactorShortlistEntry(
                rank=rank,
                factor_id=row.factor_id,
                factor_name=row.factor_name,
                factor_family=row.factor_family,
                composite_score=score,
                recommendation=recommendation,
                ic_mean=row.ic_mean,
                rank_ic_mean=row.rank_ic_mean,
                icir=row.icir,
                turnover=row.turnover,
                monotonicity_share=row.monotonicity_share,
                oos_stability_share=row.oos_stability_share,
                max_correlation_to_selected=max_corr,
                redundancy_with=redundant_with,
                rationale=tuple(rationale),
            )
        )

    summary = [
        (
            f"{entry.rank}. {entry.factor_name} ({entry.factor_id}) => {entry.recommendation}; "
            f"composite={_fmt(entry.composite_score)}; rationale={'; '.join(entry.rationale[:3])}"
        )
        for entry in entries[:8]
    ]
    return FactorShortlistResult(
        config=config,
        selected_factor_ids=tuple(selected_keep),
        entries=tuple(entries),
        recommendation_summary=tuple(summary),
    )


def _build_factor_set_result(
    *,
    shortlist: FactorShortlistResult,
    comparison_rows: list[FactorComparisonRow] | tuple[FactorComparisonRow, ...],
    correlation_matrix: tuple[tuple[str, tuple[tuple[str, float | None], ...]], ...],
    config: FactorSetConstructionConfig,
) -> FactorSetConstructionResult:
    if not shortlist.entries:
        return FactorSetConstructionResult(config=config)

    shortlist_by_id = {entry.factor_id: entry for entry in shortlist.entries}
    row_by_id = {row.factor_id: row for row in comparison_rows}
    corr_lookup = {name: dict(values) for name, values in correlation_matrix}

    keep_entries = [
        entry
        for entry in shortlist.entries
        if entry.recommendation == "keep"
        and entry.composite_score is not None
        and entry.composite_score >= config.min_selected_score
    ]
    candidate_pool = [
        entry
        for entry in shortlist.entries
        if entry.recommendation in {"keep", "watchlist"}
        and entry.composite_score is not None
        and entry.composite_score >= config.min_candidate_score
    ]
    watchlist_entries = [
        entry for entry in shortlist.entries if entry.recommendation == "watchlist"
    ]
    rejected_entries = [
        entry
        for entry in shortlist.entries
        if entry.recommendation in {"drop", "rejected"}
        or (entry.turnover is not None and entry.turnover > config.turnover_max)
        or (
            entry.oos_stability_share is not None
            and entry.oos_stability_share < config.oos_stability_min
        )
    ]

    selected_factor_ids = tuple(
        entry.factor_id for entry in keep_entries[: config.selected_set_size]
    )
    candidate_factor_ids = _build_diversified_factor_ids(
        pool=candidate_pool,
        seed=selected_factor_ids,
        max_size=config.candidate_set_size,
        corr_lookup=corr_lookup,
        redundancy_correlation_max=config.redundancy_correlation_max,
    )
    watchlist_factor_ids = _build_watchlist_factor_ids(
        entries=watchlist_entries,
        max_size=config.watchlist_set_size,
    )
    rejected_factor_ids = tuple(dict.fromkeys(entry.factor_id for entry in rejected_entries))

    factor_sets: list[FactorSetDefinition] = []
    if selected_factor_ids:
        factor_sets.append(
            _build_factor_set_definition(
                factor_set_id="set-selected-core-v1",
                status="selected",
                factor_ids=selected_factor_ids,
                construction_rule="selected_core_top_keep_by_shortlist_score",
                shortlist_by_id=shortlist_by_id,
                row_by_id=row_by_id,
                corr_lookup=corr_lookup,
                config=config,
            )
        )
    if candidate_factor_ids:
        factor_sets.append(
            _build_factor_set_definition(
                factor_set_id="set-candidate-diversified-v1",
                status="candidate",
                factor_ids=candidate_factor_ids,
                construction_rule="candidate_diversified_keep_watchlist_mix_low_redundancy",
                shortlist_by_id=shortlist_by_id,
                row_by_id=row_by_id,
                corr_lookup=corr_lookup,
                config=config,
            )
        )
    if watchlist_factor_ids:
        factor_sets.append(
            _build_factor_set_definition(
                factor_set_id="set-watchlist-family-balance-v1",
                status="watchlist",
                factor_ids=watchlist_factor_ids,
                construction_rule="watchlist_top_per_family_then_rank_fill",
                shortlist_by_id=shortlist_by_id,
                row_by_id=row_by_id,
                corr_lookup=corr_lookup,
                config=config,
            )
        )
    if rejected_factor_ids:
        factor_sets.append(
            _build_factor_set_definition(
                factor_set_id="set-rejected-quality-guardrail-v1",
                status="rejected",
                factor_ids=rejected_factor_ids,
                construction_rule="rejected_by_shortlist_or_guardrail_failure",
                shortlist_by_id=shortlist_by_id,
                row_by_id=row_by_id,
                corr_lookup=corr_lookup,
                config=config,
            )
        )

    factor_sets.sort(
        key=lambda item: (
            _FACTOR_SET_STATUS_ORDER.get(item.status, 999),
            item.factor_set_id,
        )
    )
    selected_factor_set_ids = tuple(
        item.factor_set_id for item in factor_sets if item.status == "selected"
    )
    recommendation_summary = tuple(
        f"{idx}. {item.factor_set_id} ({item.status}) | factors={', '.join(item.factor_ids)} | "
        f"mean_score={_fmt(item.score_summary.mean_shortlist_score)} | "
        f"max_corr={_fmt(item.score_summary.max_pair_correlation)}"
        for idx, item in enumerate(factor_sets, start=1)
    )
    return FactorSetConstructionResult(
        config=config,
        factor_sets=tuple(factor_sets),
        selected_factor_set_ids=selected_factor_set_ids,
        recommendation_summary=recommendation_summary,
    )


def _build_factor_set_definition(
    *,
    factor_set_id: str,
    status: str,
    factor_ids: tuple[str, ...],
    construction_rule: str,
    shortlist_by_id: dict[str, FactorShortlistEntry],
    row_by_id: dict[str, FactorComparisonRow],
    corr_lookup: dict[str, dict[str, float | None]],
    config: FactorSetConstructionConfig,
) -> FactorSetDefinition:
    factor_names = tuple(
        shortlist_by_id.get(
            factor_id, FactorShortlistEntry(0, factor_id, factor_id, "N/A")
        ).factor_name
        for factor_id in factor_ids
    )
    source_shortlist_entries = tuple(
        f"{entry.factor_id}#rank={entry.rank}#rec={entry.recommendation}"
        for factor_id in factor_ids
        for entry in [shortlist_by_id.get(factor_id)]
        if entry is not None
    )
    score_summary = _factor_set_score_summary(
        factor_ids=factor_ids,
        shortlist_by_id=shortlist_by_id,
        corr_lookup=corr_lookup,
    )
    rationale: list[str] = []
    warnings: list[str] = []

    mean_score = score_summary.mean_shortlist_score
    if mean_score is not None and mean_score >= config.min_candidate_score:
        rationale.append(
            f"high signal quality: mean shortlist score {_fmt(mean_score)} >= {config.min_candidate_score:.2f}"
        )
    else:
        warnings.append(
            f"signal quality below candidate threshold: mean shortlist score {_fmt(mean_score)}"
        )

    max_pair_corr = score_summary.max_pair_correlation
    if max_pair_corr is not None and abs(max_pair_corr) <= config.redundancy_correlation_max:
        rationale.append("diversification / low redundancy: pairwise correlation within threshold")
    else:
        warnings.append(
            f"diversification risk: max pair correlation {_fmt(max_pair_corr)} > {config.redundancy_correlation_max:.2f}"
        )

    family_balance = score_summary.family_balance_ratio
    if family_balance is not None and family_balance >= 0.50:
        rationale.append("family balance: multiple factor families represented")
    else:
        warnings.append("family balance weak: concentration in too few families")

    mean_turnover = score_summary.mean_turnover
    if mean_turnover is not None and mean_turnover <= config.turnover_max:
        rationale.append("acceptable turnover profile under guardrail")
    else:
        warnings.append(
            f"turnover profile elevated: mean turnover {_fmt(mean_turnover)} > {config.turnover_max:.2f}"
        )

    mean_oos = score_summary.mean_oos_stability_share
    if mean_oos is not None and mean_oos >= config.oos_stability_min:
        rationale.append("OOS stability considerations: stability share passes threshold")
    else:
        warnings.append(
            f"OOS stability weak: mean stability {_fmt_pct(mean_oos)} < {config.oos_stability_min:.0%}"
        )

    missing_rows = [factor_id for factor_id in factor_ids if factor_id not in row_by_id]
    if missing_rows:
        warnings.append(f"missing comparison rows for: {', '.join(missing_rows)}")
    label_zh = f"{_display_factor_set_status(status)} · {factor_set_id}"
    rationale_zh = tuple(_display_text(item) for item in rationale)

    return FactorSetDefinition(
        factor_set_id=factor_set_id,
        label_zh=label_zh,
        factor_ids=factor_ids,
        factor_names=factor_names,
        source_shortlist_entries=source_shortlist_entries,
        construction_rule=construction_rule,
        status=status,
        rationale=tuple(rationale),
        rationale_zh=rationale_zh,
        warnings=tuple(dict.fromkeys(warnings)),
        score_summary=score_summary,
    )


def _factor_set_score_summary(
    *,
    factor_ids: tuple[str, ...],
    shortlist_by_id: dict[str, FactorShortlistEntry],
    corr_lookup: dict[str, dict[str, float | None]],
) -> FactorSetScoreSummary:
    scores: list[float] = []
    icirs: list[float] = []
    turnovers: list[float] = []
    oos_values: list[float] = []
    for factor_id in factor_ids:
        entry = shortlist_by_id.get(factor_id)
        if entry is None:
            continue
        if entry.composite_score is not None:
            scores.append(entry.composite_score)
        if entry.icir is not None:
            icirs.append(entry.icir)
        if entry.turnover is not None:
            turnovers.append(entry.turnover)
        if entry.oos_stability_share is not None:
            oos_values.append(entry.oos_stability_share)
    families = [
        shortlist_by_id[factor_id].factor_family
        for factor_id in factor_ids
        if factor_id in shortlist_by_id
    ]
    family_balance_ratio: float | None = None
    if families:
        family_balance_ratio = len(set(families)) / len(families)

    return FactorSetScoreSummary(
        mean_shortlist_score=(sum(scores) / len(scores)) if scores else None,
        mean_icir=(sum(icirs) / len(icirs)) if icirs else None,
        mean_turnover=(sum(turnovers) / len(turnovers)) if turnovers else None,
        mean_oos_stability_share=(sum(oos_values) / len(oos_values)) if oos_values else None,
        max_pair_correlation=_factor_set_max_pair_correlation(
            factor_ids=factor_ids,
            corr_lookup=corr_lookup,
        ),
        family_balance_ratio=family_balance_ratio,
    )


def _factor_set_max_pair_correlation(
    *,
    factor_ids: tuple[str, ...],
    corr_lookup: dict[str, dict[str, float | None]],
) -> float | None:
    max_corr: float | None = None
    for idx, left in enumerate(factor_ids):
        for right in factor_ids[idx + 1 :]:
            corr = corr_lookup.get(left, {}).get(right)
            if corr is None:
                corr = corr_lookup.get(right, {}).get(left)
            if corr is None:
                continue
            if max_corr is None or abs(corr) > abs(max_corr):
                max_corr = corr
    return max_corr


def _build_diversified_factor_ids(
    *,
    pool: list[FactorShortlistEntry],
    seed: tuple[str, ...],
    max_size: int,
    corr_lookup: dict[str, dict[str, float | None]],
    redundancy_correlation_max: float,
) -> tuple[str, ...]:
    if max_size <= 0:
        return tuple()
    chosen: list[str] = list(seed[:max_size])
    ordered_pool = sorted(
        pool,
        key=lambda entry: (
            -_float_or_default(entry.composite_score, default=-999.0),
            entry.rank,
            entry.factor_id,
        ),
    )
    for entry in ordered_pool:
        if len(chosen) >= max_size:
            break
        if entry.factor_id in chosen:
            continue
        addable = True
        for picked in chosen:
            corr = corr_lookup.get(entry.factor_id, {}).get(picked)
            if corr is None:
                corr = corr_lookup.get(picked, {}).get(entry.factor_id)
            if corr is not None and abs(corr) > redundancy_correlation_max:
                addable = False
                break
        if addable:
            chosen.append(entry.factor_id)
    if len(chosen) < max_size:
        for entry in ordered_pool:
            if len(chosen) >= max_size:
                break
            if entry.factor_id not in chosen:
                chosen.append(entry.factor_id)
    return tuple(chosen)


def _build_watchlist_factor_ids(
    *,
    entries: list[FactorShortlistEntry],
    max_size: int,
) -> tuple[str, ...]:
    if max_size <= 0:
        return tuple()
    ordered = sorted(entries, key=lambda item: (item.rank, item.factor_id))
    chosen: list[str] = []
    seen_family: set[str] = set()
    for entry in ordered:
        if len(chosen) >= max_size:
            break
        if entry.factor_family in seen_family:
            continue
        chosen.append(entry.factor_id)
        seen_family.add(entry.factor_family)
    if len(chosen) < max_size:
        for entry in ordered:
            if len(chosen) >= max_size:
                break
            if entry.factor_id not in chosen:
                chosen.append(entry.factor_id)
    return tuple(chosen)


def _build_candidate_recipe_generation_result(
    *,
    factor_sets: tuple[FactorSetDefinition, ...],
    config: CandidateRecipeGenerationConfig,
) -> CandidateRecipeGenerationResult:
    if not factor_sets:
        return CandidateRecipeGenerationResult(config=config)

    generated: list[CandidateRecipe] = []
    for factor_set in factor_sets:
        if factor_set.status not in {"selected", "candidate"}:
            continue
        limit = config.max_recipes_per_factor_set
        if factor_set.status == "candidate":
            limit = max(1, config.max_recipes_per_factor_set - 1)
        for idx, (
            variant_name,
            weighting,
            neutralization,
            turnover_penalty,
            benchmark_mode,
        ) in enumerate(_CANDIDATE_RECIPE_VARIANTS[:limit], start=1):
            recipe_id = f"candidate-{factor_set.factor_set_id}-v{idx}"
            recipe_name = f"Candidate {factor_set.factor_set_id} v{idx}"
            rationale = [
                f"source factor set status is {factor_set.status}",
                f"variant {variant_name} explores weighting/neutralization/turnover/benchmark tradeoff",
            ]
            if factor_set.score_summary.mean_shortlist_score is not None:
                rationale.append(
                    f"mean factor-set shortlist score {_fmt(factor_set.score_summary.mean_shortlist_score)}"
                )
            assumptions = [
                "transaction cost proxy inherited from existing campaign profile defaults",
                "factor exposures remain stable between shortlist window and recipe test window",
            ]
            warnings = list(factor_set.warnings[:2])
            generated.append(
                CandidateRecipe(
                    recipe_id=recipe_id,
                    recipe_name=recipe_name,
                    source_factor_set_id=factor_set.factor_set_id,
                    source_factor_ids=factor_set.factor_ids,
                    construction_variant=variant_name,
                    weighting_scheme=weighting,
                    neutralization_mode=neutralization,
                    turnover_penalty_mode=turnover_penalty,
                    benchmark_mode=benchmark_mode,
                    rationale=tuple(rationale),
                    assumptions=tuple(assumptions),
                    warnings=tuple(warnings),
                )
            )

    generated.sort(key=lambda item: item.recipe_id)
    recommendation_summary = tuple(
        f"{idx}. {item.recipe_id} <- {item.source_factor_set_id}; variant={item.construction_variant}; "
        f"weighting={item.weighting_scheme}; neutralization={item.neutralization_mode}"
        for idx, item in enumerate(generated, start=1)
    )
    return CandidateRecipeGenerationResult(
        config=config,
        generated_recipes=tuple(generated),
        recommendation_summary=recommendation_summary,
    )


def _candidate_recipes_to_portfolio_summaries(
    *,
    generated: tuple[CandidateRecipe, ...],
    factor_sets: tuple[FactorSetDefinition, ...],
) -> list[PortfolioRecipeSummary]:
    factor_set_lookup = {item.factor_set_id: item for item in factor_sets}
    rows: list[PortfolioRecipeSummary] = []
    for candidate in generated:
        source_set = factor_set_lookup.get(candidate.source_factor_set_id)
        selected_factors = (
            source_set.factor_names if source_set is not None else candidate.source_factor_ids
        )
        neutralization_constraints = (
            "size-neutralization enabled"
            if candidate.neutralization_mode == "neutralization_on"
            else "no explicit neutralization constraint"
        )
        turnover_penalty_settings = (
            "candidate strict turnover penalty"
            if candidate.turnover_penalty_mode == "strict"
            else "candidate balanced turnover penalty"
        )
        benchmark_mode = (
            "benchmark-relative" if candidate.benchmark_mode == "benchmark_relative" else "absolute"
        )
        expected_return_proxy = "N/A"
        expected_risk_summary = "N/A"
        if source_set is not None:
            expected_return_proxy = f"derived from factor-set mean shortlist score {_fmt(source_set.score_summary.mean_shortlist_score)}"
            expected_risk_summary = (
                f"max pair corr={_fmt(source_set.score_summary.max_pair_correlation)}, "
                f"mean turnover={_fmt(source_set.score_summary.mean_turnover)}"
            )
        warnings = list(candidate.warnings)
        warnings.append("generated candidate recipe: no canonical backtest result yet")
        rows.append(
            PortfolioRecipeSummary(
                recipe_id=candidate.recipe_id,
                recipe_name=candidate.recipe_name,
                selected_factors=selected_factors,
                weighting_scheme=candidate.weighting_scheme,
                neutralization_constraints=neutralization_constraints,
                benchmark_mode=benchmark_mode,
                turnover_penalty_settings=turnover_penalty_settings,
                transaction_cost_assumptions="candidate assumption: one-way transaction cost proxy",
                factor_contributions=selected_factors,
                expected_risk_summary=expected_risk_summary,
                expected_return_proxy=expected_return_proxy,
                optimizer_diagnostics=(
                    f"generated from {candidate.source_factor_set_id} via {candidate.construction_variant}"
                ),
                infeasible_configuration_warnings=tuple(warnings),
            )
        )
    return rows


def _build_winner_selection_result(
    *,
    recipe_comparison: RecipeComparisonView,
    factor_sets: FactorSetConstructionResult,
    candidate_recipe_generation: CandidateRecipeGenerationResult,
    policy: WinnerSelectionPolicy,
) -> WinnerSelectionResult:
    rows = list(recipe_comparison.rows)
    if not rows:
        return WinnerSelectionResult(
            decision_policy_id=policy.decision_policy_id,
            policy_formula_text=_winner_policy_formula_text(policy),
        )

    candidate_recipe_ids = {
        item.recipe_id for item in candidate_recipe_generation.generated_recipes
    }
    recipe_to_factor_set = _recipe_factor_set_id_lookup(
        rows=rows,
        factor_sets=factor_sets.factor_sets,
        candidate_recipe_generation=candidate_recipe_generation,
    )
    factor_set_lookup = {item.factor_set_id: item for item in factor_sets.factor_sets}

    sharpe_norm = _normalize_metric_values(rows, "sharpe")
    post_cost_norm = _normalize_metric_values(rows, "post_cost_return")
    ann_return_norm = _normalize_metric_values(rows, "annualized_return")
    drawdown_norm = _normalize_metric_values(rows, "max_drawdown")

    score_lookup: dict[str, float | None] = {}
    for row in rows:
        source_set = factor_set_lookup.get(recipe_to_factor_set.get(row.recipe_id, ""))
        if source_set is not None:
            diversity_quality = _float_or_default(
                source_set.score_summary.family_balance_ratio,
                default=0.5,
            )
            max_corr = source_set.score_summary.max_pair_correlation
            redundancy_quality = (
                0.5 if max_corr is None else max(0.0, 1.0 - min(1.0, abs(max_corr)))
            )
            composition_quality = 0.60 * diversity_quality + 0.40 * redundancy_quality
            robustness_quality = _float_or_default(
                source_set.score_summary.mean_oos_stability_share,
                default=0.5,
            )
        else:
            n_factors = max(1, len(row.selected_factors))
            n_families = max(1, len(set(row.factor_family_mix)))
            composition_quality = min(1.0, n_families / n_factors)
            robustness_quality = 0.5

        components: dict[str, float | None] = {
            "sharpe": sharpe_norm.get(row.recipe_id),
            "post_cost_return": post_cost_norm.get(row.recipe_id),
            "annualized_return": ann_return_norm.get(row.recipe_id),
            "drawdown_quality": drawdown_norm.get(row.recipe_id),
            "composition_quality": composition_quality,
            "robustness_quality": robustness_quality,
        }
        weighted = 0.0
        total_weight = 0.0
        for key, weight in policy.component_weights:
            value = components.get(key)
            if value is None:
                continue
            weighted += weight * value
            total_weight += weight
        score_lookup[row.recipe_id] = (weighted / total_weight) if total_weight > 0 else None

    ranked_rows = sorted(
        rows,
        key=lambda item: (
            -_float_or_default(score_lookup.get(item.recipe_id), default=-999.0),
            item.recipe_id,
        ),
    )
    eligible_rows = [
        row
        for row in ranked_rows
        if row.sharpe is not None
        and row.post_cost_return is not None
        and row.max_drawdown is not None
        and row.sharpe >= policy.min_sharpe_for_winner
        and row.post_cost_return >= policy.min_post_cost_return_for_winner
        and row.max_drawdown >= policy.max_drawdown_floor
    ]
    winner = eligible_rows[0] if eligible_rows else None
    winner_recipe_id = winner.recipe_id if winner is not None else ""
    challengers = [row for row in eligible_rows if row.recipe_id != winner_recipe_id][
        : policy.challenger_count
    ]
    challenger_ids = tuple(item.recipe_id for item in challengers)

    watchlist_ids: list[str] = []
    rejected_ids: list[str] = []
    rejection_reasons: list[str] = []

    excluded = {winner_recipe_id, *challenger_ids}
    for row in ranked_rows:
        if row.recipe_id in excluded:
            continue
        score = score_lookup.get(row.recipe_id)
        hard_fail = (
            row.post_cost_return is not None
            and row.post_cost_return < policy.min_post_cost_return_for_winner
        ) or (row.max_drawdown is not None and row.max_drawdown < policy.max_drawdown_floor)
        missing_core_metrics = (
            row.sharpe is None or row.post_cost_return is None or row.max_drawdown is None
        )
        if hard_fail:
            rejected_ids.append(row.recipe_id)
            rejection_reasons.append(
                f"{row.recipe_id} rejected: post-cost/drawdown guardrail failed"
            )
            continue
        if row.recipe_id in candidate_recipe_ids and missing_core_metrics:
            watchlist_ids.append(row.recipe_id)
            continue
        if score is not None and score <= policy.reject_score_max:
            rejected_ids.append(row.recipe_id)
            rejection_reasons.append(
                f"{row.recipe_id} rejected: composite score {_fmt(score)} <= {policy.reject_score_max:.2f}"
            )
            continue
        if missing_core_metrics or (score is not None and score >= policy.watchlist_score_min):
            watchlist_ids.append(row.recipe_id)
        else:
            rejected_ids.append(row.recipe_id)
            rejection_reasons.append(f"{row.recipe_id} rejected: insufficient score/confidence")

    decision_reasons: list[str] = []
    challenger_reasons: list[str] = []
    if winner is not None:
        winner_score = score_lookup.get(winner.recipe_id)
        decision_reasons.append(
            f"{winner.recipe_id} selected as winner with composite score {_fmt(winner_score)}"
        )
        decision_reasons.append(
            f"winner tradeoff: Sharpe={_fmt(winner.sharpe)}, AnnRet={_fmt_pct(winner.annualized_return)}, "
            f"MaxDD={_fmt_pct(winner.max_drawdown)}, Post-cost={_fmt_pct(winner.post_cost_return)}"
        )
        if challengers:
            top_challenger = challengers[0]
            decision_reasons.append(
                f"winner outranks top challenger {top_challenger.recipe_id} "
                f"({_fmt(score_lookup.get(winner.recipe_id))} vs {_fmt(score_lookup.get(top_challenger.recipe_id))})"
            )
    else:
        decision_reasons.append(
            "no winner: no recipe met minimum Sharpe/post-cost/drawdown guardrails"
        )

    for challenger in challengers:
        challenger_reasons.append(
            f"{challenger.recipe_id} challenger: score={_fmt(score_lookup.get(challenger.recipe_id))}, "
            f"Sharpe={_fmt(challenger.sharpe)}, Post-cost={_fmt_pct(challenger.post_cost_return)}"
        )

    next_actions: list[str] = []
    if winner_recipe_id:
        next_actions.append(
            f"promote winner {winner_recipe_id} to deeper validation with stricter stress tests"
        )
    if challenger_ids:
        next_actions.append(
            f"keep challengers under active comparison: {', '.join(challenger_ids)}"
        )
    if watchlist_ids:
        next_actions.append(f"watchlist recipes need more evidence: {', '.join(watchlist_ids)}")
    if rejected_ids:
        next_actions.append(f"archive weak candidates: {', '.join(rejected_ids)}")

    score_table = tuple((row.recipe_id, score_lookup.get(row.recipe_id)) for row in ranked_rows)
    decision_reasons_zh = tuple(_display_text(row) for row in decision_reasons)
    challenger_reasons_zh = tuple(_display_text(row) for row in challenger_reasons)
    rejection_reasons_zh = tuple(_display_text(row) for row in dict.fromkeys(rejection_reasons))
    next_actions_zh = tuple(_display_text(row) for row in next_actions)
    return WinnerSelectionResult(
        decision_policy_id=policy.decision_policy_id,
        winner_recipe_id=winner_recipe_id,
        challenger_recipe_ids=challenger_ids,
        watchlist_recipe_ids=tuple(dict.fromkeys(watchlist_ids)),
        rejected_recipe_ids=tuple(dict.fromkeys(rejected_ids)),
        policy_formula_text=_winner_policy_formula_text(policy),
        decision_reasons=tuple(decision_reasons),
        decision_reasons_zh=decision_reasons_zh,
        challenger_reasons=tuple(challenger_reasons),
        challenger_reasons_zh=challenger_reasons_zh,
        rejection_reasons=tuple(dict.fromkeys(rejection_reasons)),
        rejection_reasons_zh=rejection_reasons_zh,
        next_actions=tuple(next_actions),
        next_actions_zh=next_actions_zh,
        score_table=score_table,
    )


def _recipe_factor_set_id_lookup(
    *,
    rows: list[RecipeComparisonRow],
    factor_sets: tuple[FactorSetDefinition, ...],
    candidate_recipe_generation: CandidateRecipeGenerationResult,
) -> dict[str, str]:
    candidate_lookup = {
        item.recipe_id: item.source_factor_set_id
        for item in candidate_recipe_generation.generated_recipes
    }
    rows_lookup: dict[str, str] = {}
    for row in rows:
        if row.recipe_id in candidate_lookup:
            rows_lookup[row.recipe_id] = candidate_lookup[row.recipe_id]
            continue
        inferred = _infer_factor_set_for_recipe(row=row, factor_sets=factor_sets)
        if inferred is not None:
            rows_lookup[row.recipe_id] = inferred
    return rows_lookup


def _infer_factor_set_for_recipe(
    *,
    row: RecipeComparisonRow,
    factor_sets: tuple[FactorSetDefinition, ...],
) -> str | None:
    if not row.selected_factors:
        return None
    selected = {item.lower() for item in row.selected_factors}
    best_id: str | None = None
    best_overlap = 0.0
    for factor_set in factor_sets:
        keys = {
            *{item.lower() for item in factor_set.factor_ids},
            *{item.lower() for item in factor_set.factor_names},
        }
        overlap = len(selected & keys)
        if overlap <= 0:
            continue
        ratio = overlap / max(1, len(selected))
        if ratio > best_overlap:
            best_overlap = ratio
            best_id = factor_set.factor_set_id
    return best_id


def _normalize_metric_values(
    rows: list[RecipeComparisonRow],
    field_name: str,
) -> dict[str, float | None]:
    values = [
        cast(float, getattr(row, field_name))
        for row in rows
        if getattr(row, field_name) is not None
    ]
    if not values:
        return {row.recipe_id: None for row in rows}
    min_value = min(values)
    max_value = max(values)
    normalized: dict[str, float | None] = {}
    for row in rows:
        value = cast(float | None, getattr(row, field_name))
        if value is None:
            normalized[row.recipe_id] = None
            continue
        if math.isclose(max_value, min_value):
            normalized[row.recipe_id] = 1.0
            continue
        normalized[row.recipe_id] = (value - min_value) / (max_value - min_value)
    return normalized


def _winner_policy_formula_text(policy: WinnerSelectionPolicy) -> str:
    weights = ", ".join(f"{name}={weight:.2f}" for name, weight in policy.component_weights)
    return (
        f"{policy.formula_text}; weights({weights}); "
        f"winner_guardrails[sharpe>={policy.min_sharpe_for_winner:.2f}, "
        f"post_cost>={policy.min_post_cost_return_for_winner:.2%}, "
        f"max_drawdown>={policy.max_drawdown_floor:.0%}]"
    )


def _build_next_step_recommendations(
    *,
    shortlist: FactorShortlistResult,
    factor_sets: FactorSetConstructionResult,
    candidate_recipe_generation: CandidateRecipeGenerationResult,
    recipe_comparison: RecipeComparisonView,
    winner_selection: WinnerSelectionResult,
) -> NextStepRecommendationResult:
    recommendations: list[NextStepRecommendation] = []
    rid = 1
    score_lookup = {recipe_id: score for recipe_id, score in winner_selection.score_table}

    if winner_selection.winner_recipe_id:
        winner_score = score_lookup.get(winner_selection.winner_recipe_id)
        recommendations.append(
            NextStepRecommendation(
                recommendation_id=f"rec-{rid:02d}",
                category="promotion",
                priority="P1",
                action=f"promote {winner_selection.winner_recipe_id} to deeper validation",
                action_text_zh=_display_text(
                    f"promote {winner_selection.winner_recipe_id} to deeper validation"
                ),
                rationale=(
                    f"winner selected by explicit policy with strongest score {_fmt(winner_score)}"
                ),
                rationale_zh=_display_text(
                    f"winner selected by explicit policy with strongest score {_fmt(winner_score)}"
                ),
                label_zh=_display_next_step_category("promotion"),
                trigger_objects=(
                    winner_selection.winner_recipe_id,
                    winner_selection.decision_policy_id,
                ),
                supporting_evidence=(
                    f"winner_score={_fmt(winner_score)}",
                    "decision_layer=winner_selection",
                ),
            )
        )
        rid += 1

    if winner_selection.challenger_recipe_ids:
        recommendations.append(
            NextStepRecommendation(
                recommendation_id=f"rec-{rid:02d}",
                category="stress_test",
                priority="P1",
                action=(
                    "run stricter neutralization stress test for challengers "
                    + ", ".join(winner_selection.challenger_recipe_ids)
                ),
                action_text_zh=_display_text(
                    "run stricter neutralization stress test for challengers "
                    + ", ".join(winner_selection.challenger_recipe_ids)
                ),
                rationale="challengers remain competitive under current ranking and deserve focused stress tests",
                rationale_zh=_display_text(
                    "challengers remain competitive under current ranking and deserve focused stress tests"
                ),
                label_zh=_display_next_step_category("stress_test"),
                trigger_objects=winner_selection.challenger_recipe_ids,
                supporting_evidence=tuple(winner_selection.challenger_reasons[:2]),
            )
        )
        rid += 1

    low_redundancy_watch = next(
        (
            entry
            for entry in shortlist.entries
            if entry.recommendation == "watchlist" and entry.redundancy_with
        ),
        None,
    )
    if low_redundancy_watch is not None:
        recommendations.append(
            NextStepRecommendation(
                recommendation_id=f"rec-{rid:02d}",
                category="factor_pruning",
                priority="P2",
                action=f"drop redundant factor {low_redundancy_watch.factor_id} from active set draft",
                action_text_zh=_display_text(
                    f"drop redundant factor {low_redundancy_watch.factor_id} from active set draft"
                ),
                rationale="watchlist factor flagged as redundant against selected factors",
                rationale_zh=_display_text(
                    "watchlist factor flagged as redundant against selected factors"
                ),
                label_zh=_display_next_step_category("factor_pruning"),
                trigger_objects=(
                    low_redundancy_watch.factor_id,
                    low_redundancy_watch.redundancy_with or "",
                ),
                supporting_evidence=low_redundancy_watch.rationale[:2],
            )
        )
        rid += 1

    generated_by_set: dict[str, int] = {}
    for item in candidate_recipe_generation.generated_recipes:
        generated_by_set[item.source_factor_set_id] = (
            generated_by_set.get(item.source_factor_set_id, 0) + 1
        )
    for factor_set in factor_sets.factor_sets:
        if factor_set.status != "candidate":
            continue
        if (
            generated_by_set.get(factor_set.factor_set_id, 0)
            >= candidate_recipe_generation.config.max_recipes_per_factor_set
        ):
            continue
        recommendations.append(
            NextStepRecommendation(
                recommendation_id=f"rec-{rid:02d}",
                category="recipe_generation",
                priority="P2",
                action=f"generate additional recipes from factor set {factor_set.factor_set_id}",
                action_text_zh=_display_text(
                    f"generate additional recipes from factor set {factor_set.factor_set_id}"
                ),
                rationale="candidate factor set has room for more explicit construction variants",
                rationale_zh=_display_text(
                    "candidate factor set has room for more explicit construction variants"
                ),
                label_zh=_display_next_step_category("recipe_generation"),
                trigger_objects=(factor_set.factor_set_id,),
                supporting_evidence=(
                    f"generated_count={generated_by_set.get(factor_set.factor_set_id, 0)}",
                    f"max_per_set={candidate_recipe_generation.config.max_recipes_per_factor_set}",
                ),
            )
        )
        rid += 1
        break

    high_turnover_set = next(
        (
            item
            for item in factor_sets.factor_sets
            if item.score_summary.mean_turnover is not None
            and item.score_summary.mean_turnover > _DEFAULT_FACTOR_SET_CONFIG.turnover_max * 0.90
        ),
        None,
    )
    if high_turnover_set is not None:
        recommendations.append(
            NextStepRecommendation(
                recommendation_id=f"rec-{rid:02d}",
                category="turnover_sensitivity",
                priority="P2",
                action=f"inspect turnover sensitivity for {high_turnover_set.factor_set_id}",
                action_text_zh=_display_text(
                    f"inspect turnover sensitivity for {high_turnover_set.factor_set_id}"
                ),
                rationale="factor set turnover is near or above guardrail threshold",
                rationale_zh=_display_text(
                    "factor set turnover is near or above guardrail threshold"
                ),
                label_zh=_display_next_step_category("turnover_sensitivity"),
                trigger_objects=(high_turnover_set.factor_set_id,),
                supporting_evidence=(
                    f"mean_turnover={_fmt(high_turnover_set.score_summary.mean_turnover)}",
                    f"guardrail={_DEFAULT_FACTOR_SET_CONFIG.turnover_max:.2f}",
                ),
            )
        )
        rid += 1

    if winner_selection.watchlist_recipe_ids:
        recommendations.append(
            NextStepRecommendation(
                recommendation_id=f"rec-{rid:02d}",
                category="evidence_gap",
                priority="P3",
                action="collect more history before trusting watchlist candidates",
                action_text_zh=_display_text(
                    "collect more history before trusting watchlist candidates"
                ),
                rationale="watchlist recipes lack enough post-cost robustness evidence",
                rationale_zh=_display_text(
                    "watchlist recipes lack enough post-cost robustness evidence"
                ),
                label_zh=_display_next_step_category("evidence_gap"),
                trigger_objects=winner_selection.watchlist_recipe_ids,
                supporting_evidence=(
                    f"watchlist_count={len(winner_selection.watchlist_recipe_ids)}",
                    "missing_or_borderline_metrics in decision layer",
                ),
            )
        )
        rid += 1

    if winner_selection.rejected_recipe_ids:
        recommendations.append(
            NextStepRecommendation(
                recommendation_id=f"rec-{rid:02d}",
                category="archival",
                priority="P3",
                action="archive weak candidates and keep registry lineage for audit",
                action_text_zh=_display_text(
                    "archive weak candidates and keep registry lineage for audit"
                ),
                rationale="recipes failed guardrails or scored too low for active research budget",
                rationale_zh=_display_text(
                    "recipes failed guardrails or scored too low for active research budget"
                ),
                label_zh=_display_next_step_category("archival"),
                trigger_objects=winner_selection.rejected_recipe_ids,
                supporting_evidence=winner_selection.rejection_reasons[:2],
            )
        )

    summary = tuple(
        f"{idx}. [{item.priority}] {item.action} | triggers={', '.join(item.trigger_objects) or 'N/A'}"
        for idx, item in enumerate(recommendations, start=1)
    )
    summary_zh = tuple(
        (
            f"{idx}. [{_display_priority(item.priority)}] "
            f"{item.action_text_zh or _display_text(item.action)} | "
            f"触发对象 (triggers)={', '.join(item.trigger_objects) or _ui('na')}"
        )
        for idx, item in enumerate(recommendations, start=1)
    )
    return NextStepRecommendationResult(
        policy_id=_NEXT_STEP_POLICY_ID,
        policy_formula_text=_NEXT_STEP_POLICY_FORMULA,
        recommendations=tuple(recommendations),
        summary=summary,
        summary_zh=summary_zh,
    )


def _factor_composite_score(
    *,
    row: FactorComparisonRow,
    config: FactorShortlistConfig,
) -> float | None:
    components = {
        "ic_mean": _normalize_range(row.ic_mean, _SHORTLIST_SCORE_RANGES["ic_mean"]),
        "rank_ic_mean": _normalize_range(
            row.rank_ic_mean,
            _SHORTLIST_SCORE_RANGES["rank_ic_mean"],
        ),
        "icir": _normalize_range(row.icir, _SHORTLIST_SCORE_RANGES["icir"]),
        "monotonicity_share": _normalize_range(
            row.monotonicity_share,
            _SHORTLIST_SCORE_RANGES["monotonicity_share"],
        ),
        "turnover_efficiency": _normalize_range(
            _turnover_efficiency(row.turnover),
            _SHORTLIST_SCORE_RANGES["turnover_efficiency"],
        ),
        "oos_stability_share": _normalize_range(
            row.oos_stability_share,
            _SHORTLIST_SCORE_RANGES["oos_stability_share"],
        ),
    }

    weighted = 0.0
    total_weight = 0.0
    for name, weight in config.component_weights:
        value = components.get(name)
        if value is None:
            continue
        weighted += value * weight
        total_weight += weight
    if total_weight <= 0:
        return None
    return weighted / total_weight


def _factor_shortlist_recommendation(
    *,
    row: FactorComparisonRow,
    score: float | None,
    max_corr: float | None,
    redundant_with: str | None,
    config: FactorShortlistConfig,
) -> tuple[str, list[str]]:
    rationale: list[str] = []
    if row.ic_mean is not None and row.ic_mean >= config.min_ic_mean:
        rationale.append(f"IC passes threshold ({_fmt(row.ic_mean)} >= {config.min_ic_mean:.3f})")
    else:
        rationale.append(f"IC below threshold ({_fmt(row.ic_mean)} < {config.min_ic_mean:.3f})")

    if row.rank_ic_mean is not None and row.rank_ic_mean >= config.min_rank_ic_mean:
        rationale.append(
            f"RankIC passes threshold ({_fmt(row.rank_ic_mean)} >= {config.min_rank_ic_mean:.3f})"
        )
    else:
        rationale.append(
            f"RankIC below threshold ({_fmt(row.rank_ic_mean)} < {config.min_rank_ic_mean:.3f})"
        )

    if row.icir is not None and row.icir >= config.min_icir:
        rationale.append(f"ICIR strong ({_fmt(row.icir)} >= {config.min_icir:.2f})")
    else:
        rationale.append(f"ICIR weak ({_fmt(row.icir)} < {config.min_icir:.2f})")

    if (
        row.monotonicity_share is not None
        and row.monotonicity_share >= config.min_monotonicity_share
    ):
        rationale.append(
            f"monotonicity acceptable ({_fmt_pct(row.monotonicity_share)} >= {config.min_monotonicity_share:.0%})"
        )
    else:
        rationale.append(
            f"monotonicity weak ({_fmt_pct(row.monotonicity_share)} < {config.min_monotonicity_share:.0%})"
        )

    if row.turnover is not None and row.turnover <= config.max_turnover:
        rationale.append(f"turnover acceptable ({_fmt(row.turnover)} <= {config.max_turnover:.2f})")
    else:
        rationale.append(f"turnover high ({_fmt(row.turnover)} > {config.max_turnover:.2f})")

    if (
        row.oos_stability_share is not None
        and row.oos_stability_share >= config.min_oos_stability_share
    ):
        rationale.append(
            f"OOS stability acceptable ({_fmt_pct(row.oos_stability_share)} >= {config.min_oos_stability_share:.0%})"
        )
    else:
        rationale.append(
            f"OOS stability weak ({_fmt_pct(row.oos_stability_share)} < {config.min_oos_stability_share:.0%})"
        )

    redundancy_flag = max_corr is not None and abs(max_corr) > config.redundancy_correlation_max
    if redundancy_flag:
        rationale.append(
            f"redundant with selected factor {redundant_with or 'N/A'} (|corr|={_fmt(max_corr)} > {config.redundancy_correlation_max:.2f})"
        )
    else:
        rationale.append("redundancy acceptable vs selected factors")

    hard_quality_pass = (
        row.ic_mean is not None
        and row.ic_mean >= config.min_ic_mean
        and row.rank_ic_mean is not None
        and row.rank_ic_mean >= config.min_rank_ic_mean
        and row.icir is not None
        and row.icir >= config.min_icir
        and row.monotonicity_share is not None
        and row.monotonicity_share >= config.min_monotonicity_share
        and row.turnover is not None
        and row.turnover <= config.max_turnover
        and row.oos_stability_share is not None
        and row.oos_stability_share >= config.min_oos_stability_share
    )
    if score is None:
        return "drop", rationale
    if hard_quality_pass and not redundancy_flag and score >= config.keep_score_min:
        return "keep", rationale
    if score >= config.watchlist_score_min:
        return "watchlist", rationale
    return "drop", rationale


def _max_correlation_to_selected(
    *,
    factor_id: str,
    selected_factor_ids: list[str],
    corr_lookup: dict[str, dict[str, float | None]],
) -> tuple[float | None, str | None]:
    if not selected_factor_ids:
        return None, None
    best_name: str | None = None
    best_value: float | None = None
    for selected in selected_factor_ids:
        corr = corr_lookup.get(factor_id, {}).get(selected)
        if corr is None:
            continue
        if best_value is None or abs(corr) > abs(best_value):
            best_value = corr
            best_name = selected
    return best_value, best_name


def _normalize_range(value: float | None, bounds: tuple[float, float]) -> float | None:
    if value is None:
        return None
    lower, upper = bounds
    if upper <= lower:
        return None
    clipped = min(upper, max(lower, value))
    return (clipped - lower) / (upper - lower)


def _turnover_efficiency(turnover: float | None) -> float | None:
    if turnover is None:
        return None
    efficiency = 1.0 - turnover
    return min(1.0, max(0.0, efficiency))


def _build_recipe_comparison_view(
    *,
    recipes: list[PortfolioRecipeSummary] | tuple[PortfolioRecipeSummary, ...],
    backtests: list[PortfolioBacktestSummary] | tuple[PortfolioBacktestSummary, ...],
    factor_summaries: list[FactorSummary] | tuple[FactorSummary, ...],
) -> RecipeComparisonView:
    if not recipes:
        return RecipeComparisonView()

    backtest_lookup = {item.recipe_id: item for item in backtests}
    factor_family_lookup: dict[str, str] = {}
    for summary in factor_summaries:
        factor_family_lookup[summary.factor_name] = summary.factor_family
        factor_family_lookup[summary.factor_id] = summary.factor_family

    rows: list[RecipeComparisonRow] = []
    for recipe in recipes:
        backtest = backtest_lookup.get(recipe.recipe_id)
        factor_families = tuple(
            sorted({factor_family_lookup.get(name, "unknown") for name in recipe.selected_factors})
        )
        objective_tag = _recipe_objective_tag(recipe)
        construction_style = _recipe_construction_style(recipe)
        rows.append(
            RecipeComparisonRow(
                recipe_id=recipe.recipe_id,
                recipe_name=recipe.recipe_name,
                selected_factors=recipe.selected_factors,
                factor_family_mix=factor_families,
                objective_tag=objective_tag,
                construction_style=construction_style,
                weighting_scheme=recipe.weighting_scheme,
                neutralization_constraints=recipe.neutralization_constraints,
                turnover_penalty_settings=recipe.turnover_penalty_settings,
                transaction_cost_assumptions=recipe.transaction_cost_assumptions,
                benchmark_mode=recipe.benchmark_mode,
                position_limits=recipe.position_limits,
                expected_return_proxy=recipe.expected_return_proxy,
                expected_risk_summary=recipe.expected_risk_summary,
                sharpe=backtest.sharpe if backtest is not None else None,
                annualized_return=(backtest.annualized_return if backtest is not None else None),
                max_drawdown=backtest.max_drawdown if backtest is not None else None,
                information_ratio=(backtest.information_ratio if backtest is not None else None),
                post_cost_return=(backtest.post_cost_return if backtest is not None else None),
            )
        )

    rows.sort(
        key=lambda item: (
            -_float_or_default(item.sharpe, default=-999.0),
            -_float_or_default(item.annualized_return, default=-999.0),
            item.recipe_name,
        )
    )

    leaderboards: list[RecipeLeaderboardEntry] = []
    for objective, field_name in _RECIPE_LEADERBOARD_OBJECTIVES:
        ranked_rows = sorted(
            rows,
            key=lambda item: (
                -_float_or_default(getattr(item, field_name), default=-999.0),
                item.recipe_id,
            ),
        )
        for rank, row in enumerate(ranked_rows[:5], start=1):
            leaderboards.append(
                RecipeLeaderboardEntry(
                    objective=objective,
                    rank=rank,
                    recipe_id=row.recipe_id,
                    recipe_name=row.recipe_name,
                    metric_value=cast(float | None, getattr(row, field_name)),
                )
            )

    head_to_head = _build_recipe_head_to_head(rows)

    grouping_counter: dict[str, int] = {}
    for row in rows:
        group_key = f"{','.join(row.factor_family_mix) or 'unknown'} | {row.construction_style}"
        grouping_counter[group_key] = grouping_counter.get(group_key, 0) + 1
    grouping_summary = tuple(sorted(grouping_counter.items(), key=lambda item: (-item[1], item[0])))

    return RecipeComparisonView(
        rows=tuple(rows),
        leaderboards=tuple(leaderboards),
        head_to_head=tuple(head_to_head),
        grouping_summary=grouping_summary,
    )


def _build_recipe_head_to_head(
    rows: list[RecipeComparisonRow],
) -> list[RecipeHeadToHeadInsight]:
    insights: list[RecipeHeadToHeadInsight] = []
    for objective, field_name in _RECIPE_LEADERBOARD_OBJECTIVES:
        ranked = sorted(
            rows,
            key=lambda item: (
                -_float_or_default(getattr(item, field_name), default=-999.0),
                item.recipe_id,
            ),
        )
        if len(ranked) < 2:
            continue
        winner = ranked[0]
        loser = ranked[1]
        winner_value = cast(float | None, getattr(winner, field_name))
        loser_value = cast(float | None, getattr(loser, field_name))
        if winner_value is None and loser_value is None:
            continue
        reasons = _recipe_comparison_reasons(winner=winner, loser=loser)
        summary = (
            f"{winner.recipe_id} leads on {objective} ({_fmt(winner_value)} vs {_fmt(loser_value)})"
        )
        insights.append(
            RecipeHeadToHeadInsight(
                objective=objective,
                winner_recipe_id=winner.recipe_id,
                loser_recipe_id=loser.recipe_id,
                summary=summary,
                reasons=tuple(reasons),
            )
        )
    return insights


def _recipe_comparison_reasons(
    *,
    winner: RecipeComparisonRow,
    loser: RecipeComparisonRow,
) -> list[str]:
    rows: list[str] = []
    if (
        winner.post_cost_return is not None
        and loser.post_cost_return is not None
        and winner.post_cost_return > loser.post_cost_return
    ):
        rows.append(
            f"higher post-cost return ({_fmt_pct(winner.post_cost_return)} vs {_fmt_pct(loser.post_cost_return)})"
        )
    if (
        winner.information_ratio is not None
        and loser.information_ratio is not None
        and winner.information_ratio > loser.information_ratio
    ):
        rows.append(
            f"higher information ratio ({_fmt(winner.information_ratio)} vs {_fmt(loser.information_ratio)})"
        )
    if (
        winner.max_drawdown is not None
        and loser.max_drawdown is not None
        and winner.max_drawdown > loser.max_drawdown
    ):
        rows.append(
            f"shallower drawdown ({_fmt_pct(winner.max_drawdown)} vs {_fmt_pct(loser.max_drawdown)})"
        )
    if winner.construction_style != loser.construction_style:
        rows.append(
            f"construction style differs ({winner.construction_style} vs {loser.construction_style})"
        )
    if winner.turnover_penalty_settings != loser.turnover_penalty_settings:
        rows.append(
            f"turnover policy differs ({winner.turnover_penalty_settings} vs {loser.turnover_penalty_settings})"
        )
    return rows


def _recipe_objective_tag(recipe: PortfolioRecipeSummary) -> str:
    benchmark_text = recipe.benchmark_mode.lower()
    turnover_text = recipe.turnover_penalty_settings.lower()
    if "benchmark-relative" in benchmark_text:
        return "benchmark_relative_alpha"
    if "warn if mean turnover" in turnover_text:
        return "turnover_aware_absolute_return"
    return "absolute_return"


def _recipe_construction_style(recipe: PortfolioRecipeSummary) -> str:
    neutralization = (
        "neutralized"
        if "enabled" in recipe.neutralization_constraints.lower()
        else "non_neutralized"
    )
    return f"{recipe.weighting_scheme} | {neutralization} | {recipe.benchmark_mode}"


def _build_registry_entry(
    *,
    case_name: str,
    profile_name: str,
    run_timestamp: str | None,
    factor_id: str,
    recipe_id: str,
    artifacts: _CaseArtifacts,
) -> ExperimentRegistryEntry:
    run_id = (
        safe_text(artifacts.manifest.get("run_id"))
        or f"{profile_name}:{case_name}:{run_timestamp or 'unknown'}"
    )
    output_dir = str(artifacts.output_dir) if artifacts.output_dir is not None else ""
    run_manifest_path = _artifact_path_text(
        artifacts,
        "run_manifest_path",
        "run_manifest.json",
    )
    factor_definition_path = _artifact_path_text(
        artifacts,
        "factor_definition_json_path",
        "factor_definition.json",
    )
    signal_validation_path = _artifact_path_text(
        artifacts,
        "signal_validation_json_path",
        "signal_validation.json",
    )
    portfolio_recipe_path = _artifact_path_text(
        artifacts,
        "portfolio_recipe_json_path",
        "portfolio_recipe.json",
    )
    backtest_result_path = _artifact_path_text(
        artifacts,
        "backtest_result_json_path",
        "backtest_result.json",
    )
    backtest_id = f"backtest-{recipe_id}"

    provenance_links = _provenance_links(artifacts)
    return ExperimentRegistryEntry(
        case_name=case_name,
        profile_name=profile_name,
        run_id=run_id,
        run_timestamp_utc=run_timestamp,
        factor_id=factor_id,
        recipe_id=recipe_id,
        backtest_id=backtest_id,
        output_dir=output_dir,
        run_manifest_path=run_manifest_path,
        factor_definition_path=factor_definition_path,
        signal_validation_path=signal_validation_path,
        portfolio_recipe_path=portfolio_recipe_path,
        backtest_result_path=backtest_result_path,
        provenance_links=provenance_links,
    )


def _build_lineage_links(
    *,
    profile_name: str,
    factor_id: str,
    recipe_id: str,
) -> tuple[ResearchLineageLink, ...]:
    factor_obj = f"factor_definition:{profile_name}:{factor_id}"
    validation_obj = f"signal_validation:{profile_name}:{factor_id}"
    recipe_obj = f"portfolio_recipe:{profile_name}:{recipe_id}"
    backtest_obj = f"backtest_result:{profile_name}:{recipe_id}"
    return (
        ResearchLineageLink(
            from_object=factor_obj,
            relation="validated_by",
            to_object=validation_obj,
        ),
        ResearchLineageLink(
            from_object=validation_obj,
            relation="feeds_recipe",
            to_object=recipe_obj,
        ),
        ResearchLineageLink(
            from_object=factor_obj,
            relation="selected_into_recipe",
            to_object=recipe_obj,
        ),
        ResearchLineageLink(
            from_object=recipe_obj,
            relation="evaluated_by_backtest",
            to_object=backtest_obj,
        ),
    )


def _build_lineage_registry(
    *,
    entries: list[ExperimentRegistryEntry],
    links: list[ResearchLineageLink],
    workflow_artifact_paths: dict[str, Path] | None = None,
    default_profile: str = "default_research",
    additional_warnings: tuple[str, ...] = (),
) -> ResearchLineageRegistry:
    sorted_entries = sorted(
        entries,
        key=lambda item: (
            item.profile_name,
            item.case_name,
            item.run_timestamp_utc or "",
        ),
    )
    dedup_links: dict[tuple[str, str, str], ResearchLineageLink] = {}
    for link in links:
        key = (link.from_object, link.relation, link.to_object)
        dedup_links[key] = link
    warnings: list[str] = []
    for entry in sorted_entries:
        missing = []
        if not _path_text_exists(entry.factor_definition_path):
            missing.append("factor_definition")
        if not _path_text_exists(entry.signal_validation_path):
            missing.append("signal_validation")
        if not _path_text_exists(entry.portfolio_recipe_path):
            missing.append("portfolio_recipe")
        if not _path_text_exists(entry.backtest_result_path):
            missing.append("backtest_result")
        if missing:
            warnings.append(
                f"{entry.case_name} ({entry.profile_name}) missing canonical paths: {', '.join(missing)}"
            )
    workflow_paths = workflow_artifact_paths or {}
    workflow_link_specs = (
        (
            f"factor_shortlist:{default_profile}:campaign_profile_comparison",
            "feeds_factor_sets",
            "factor_set_result",
            "factor_set_result_json_path",
        ),
        (
            f"factor_set_result:{default_profile}",
            "generates_candidate_recipes",
            "candidate_recipe_generation",
            "candidate_recipe_generation_json_path",
        ),
        (
            f"candidate_recipe_generation:{default_profile}",
            "feeds_winner_selection",
            "winner_selection",
            "winner_selection_json_path",
        ),
        (
            f"winner_selection:{default_profile}",
            "feeds_next_step_recommendations",
            "next_step_recommendations",
            "next_step_recommendations_json_path",
        ),
        (
            f"next_step_recommendations:{default_profile}",
            "emits_artifact_load_diagnostics",
            "artifact_load_diagnostics",
            "artifact_load_diagnostics_json_path",
        ),
    )
    if workflow_paths:
        for from_object, relation, to_label, path_key in workflow_link_specs:
            path = workflow_paths.get(path_key)
            if path is None:
                warnings.append(f"workflow closure artifact missing for lineage: {path_key}")
                continue
            dedup_links[(from_object, relation, f"{to_label}:{default_profile}:{path.name}")] = (
                ResearchLineageLink(
                    from_object=from_object,
                    relation=relation,
                    to_object=f"{to_label}:{default_profile}:{path.name}",
                )
            )
    warnings.extend(additional_warnings)
    return ResearchLineageRegistry(
        entries=tuple(sorted_entries),
        links=tuple(dedup_links.values()),
        warnings=tuple(dict.fromkeys(warnings)),
    )


def _provenance_links(artifacts: _CaseArtifacts) -> tuple[str, ...]:
    rows: list[str] = []
    for label, payload in (
        ("factor_definition", artifacts.factor_definition_payload),
        ("signal_validation", artifacts.signal_validation_payload),
        ("portfolio_recipe", artifacts.portfolio_recipe_payload),
        ("backtest_result", artifacts.backtest_result_payload),
    ):
        source = as_object_dict(payload.get("source_artifacts"))
        for key, value in sorted(source.items(), key=lambda item: str(item[0])):
            text = safe_text(value)
            if text:
                rows.append(f"{label}.{key}={text}")
    return tuple(rows)


def _artifact_path_text(artifacts: _CaseArtifacts, *keys: str) -> str:
    for key in keys:
        path = artifacts.artifact_paths.get(key)
        if path is not None:
            return str(path)
    return ""


def _path_text_exists(path_text: str) -> bool:
    text = safe_text(path_text)
    if not text:
        return False
    return Path(text).exists()


def _lineage_anchor(case_name: str) -> str:
    return f"lineage-{_slug(case_name)}"


def _recipe_case_name(recipe_id: str) -> str:
    prefix = "recipe-"
    if recipe_id.startswith(prefix):
        return recipe_id[len(prefix) :]
    return recipe_id


def _top_factor_lines(
    *,
    factor_summaries: list[FactorSummary] | tuple[FactorSummary, ...],
    top_n: int,
) -> list[str]:
    rows = sorted(factor_summaries, key=_factor_sort_key)[:top_n]
    return [
        f"{idx}. {_display_name_with_zh(english=item.factor_name, zh=item.display_name_zh)} ({item.factor_id}) | "
        f"信号质量分 (IC quality score)={_fmt(item.signal_quality_score)} | "
        f"状态 (status)={_display_status(item.research_status)}"
        for idx, item in enumerate(rows, start=1)
    ]


def _top_portfolio_lines(
    *,
    backtests: list[PortfolioBacktestSummary] | tuple[PortfolioBacktestSummary, ...],
    top_n: int,
) -> list[str]:
    rows = sorted(
        backtests,
        key=lambda item: (
            -_float_or_default(item.sharpe, default=-999.0),
            -_float_or_default(item.annualized_return, default=-999.0),
            item.recipe_id,
        ),
    )[:top_n]
    return [
        f"{idx}. {item.recipe_id} | Sharpe={_fmt(item.sharpe)} | "
        f"年化收益 (AnnRet)={_fmt_pct(item.annualized_return)} | "
        f"信息比率 (IR)={_fmt(item.information_ratio)}"
        for idx, item in enumerate(rows, start=1)
    ]


def _recent_run_lines(
    *,
    recent_runs: list[tuple[str, str]],
    top_n: int,
) -> list[str]:
    rows = sorted(recent_runs, key=lambda item: item[0], reverse=True)[:top_n]
    return [
        f"{timestamp} | {case_name} | 研究运行记录 (research run)" for timestamp, case_name in rows
    ]


def _long_short_series(group_returns_df: pd.DataFrame | None) -> pd.Series | None:
    if group_returns_df is None or group_returns_df.empty:
        return None
    required = {"date", "group", "group_return"}
    if not required.issubset(set(group_returns_df.columns)):
        return None

    frame = group_returns_df.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["group"] = pd.to_numeric(frame["group"], errors="coerce")
    frame["group_return"] = pd.to_numeric(frame["group_return"], errors="coerce")
    frame = frame.dropna(subset=["date", "group", "group_return"])
    if frame.empty:
        return None

    pivot = frame.pivot_table(index="date", columns="group", values="group_return", aggfunc="mean")
    if pivot.shape[1] < 2:
        return None

    bottom = pivot.columns.min()
    top = pivot.columns.max()
    long_short = (pivot[top] - pivot[bottom]).sort_index().dropna()
    if long_short.empty:
        return None
    return long_short


def _return_stats(series: pd.Series, periods_per_year: int) -> dict[str, object]:
    clean = pd.to_numeric(series, errors="coerce").dropna().sort_index()
    if len(clean) < 2:
        return {
            "annualized_return": None,
            "annualized_volatility": None,
            "sharpe": None,
            "sortino": None,
            "max_drawdown": None,
            "calmar": None,
            "win_rate": None,
            "rolling_sharpe": None,
            "rolling_drawdown": None,
            "nav_points": (),
            "monthly_returns": (),
            "drawdown_table": (),
            "subperiod_analysis": "N/A",
            "regime_analysis": "N/A",
        }

    nav = (1.0 + clean).cumprod()
    total_return = float(nav.iloc[-1] - 1.0)
    annualized_return = float((1.0 + total_return) ** (periods_per_year / len(clean)) - 1.0)
    annualized_volatility = float(clean.std(ddof=1) * math.sqrt(periods_per_year))

    sharpe = None
    if annualized_volatility > 0:
        sharpe = annualized_return / annualized_volatility

    downside = clean[clean < 0]
    sortino = None
    if len(downside) >= 2:
        downside_vol = float(downside.std(ddof=1) * math.sqrt(periods_per_year))
        if downside_vol > 0:
            sortino = annualized_return / downside_vol

    drawdown = nav / nav.cummax() - 1.0
    max_drawdown = float(drawdown.min())
    calmar = None
    if max_drawdown < 0:
        calmar = annualized_return / abs(max_drawdown)

    win_rate = float((clean > 0).mean())

    window = min(20, len(clean))
    rolling_sharpe = None
    if window >= 5:
        rolling_mean = clean.rolling(window).mean()
        rolling_std = clean.rolling(window).std(ddof=1)
        rolling_ratio = rolling_mean / rolling_std
        rolling_ratio = rolling_ratio.replace([math.inf, -math.inf], pd.NA).dropna()
        if not rolling_ratio.empty:
            rolling_sharpe = float(rolling_ratio.iloc[-1] * math.sqrt(periods_per_year))

    rolling_drawdown = float(drawdown.iloc[-1])

    monthly = clean.resample("ME").apply(lambda values: float((1.0 + values).prod() - 1.0))
    monthly_rows = tuple((idx.strftime("%Y-%m"), float(value)) for idx, value in monthly.items())

    worst_drawdowns = drawdown.nsmallest(8)
    drawdown_rows = tuple(
        (idx.strftime("%Y-%m-%d"), float(value)) for idx, value in worst_drawdowns.items()
    )

    split = len(clean) // 2
    first_half = clean.iloc[:split]
    second_half = clean.iloc[split:]
    first_ann = _annualized_from_series(first_half, periods_per_year)
    second_ann = _annualized_from_series(second_half, periods_per_year)
    subperiod_analysis = (
        f"first_half_ann={_fmt_pct(first_ann)}; second_half_ann={_fmt_pct(second_ann)}"
    )

    volatility_cut = clean.abs().median()
    high_vol = clean[clean.abs() >= volatility_cut]
    low_vol = clean[clean.abs() < volatility_cut]
    regime_analysis = (
        f"high-vol mean={_fmt(high_vol.mean() if len(high_vol) > 0 else None)}; "
        f"low-vol mean={_fmt(low_vol.mean() if len(low_vol) > 0 else None)}"
    )

    nav_points = tuple((idx.strftime("%Y-%m-%d"), float(value)) for idx, value in nav.items())

    return {
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_drawdown,
        "calmar": calmar,
        "win_rate": win_rate,
        "rolling_sharpe": rolling_sharpe,
        "rolling_drawdown": rolling_drawdown,
        "nav_points": nav_points,
        "monthly_returns": monthly_rows,
        "drawdown_table": drawdown_rows,
        "subperiod_analysis": subperiod_analysis,
        "regime_analysis": regime_analysis,
    }


def _annualized_from_series(series: pd.Series, periods_per_year: int) -> float | None:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return None
    nav = (1.0 + clean).cumprod()
    total_return = float(nav.iloc[-1] - 1.0)
    return float((1.0 + total_return) ** (periods_per_year / len(clean)) - 1.0)


def _baseline_scenario(portfolio_metrics: dict[str, object]) -> dict[str, object]:
    scenarios = as_object_list(portfolio_metrics.get("scenario_metrics"))
    if not scenarios:
        return {}

    by_rank = [as_object_dict(item) for item in scenarios]
    preferred = [item for item in by_rank if safe_text(item.get("weighting_method")) == "rank"]
    if preferred:
        return preferred[0]
    return by_rank[0]


def _periods_per_year(rebalance_frequency: str) -> int:
    freq = (rebalance_frequency or "").strip().upper()
    if freq.startswith("D"):
        return 252
    if freq.startswith("W"):
        return 52
    if freq.startswith("M"):
        return 12
    return 252


def _distribution_snapshots(factor_series: pd.Series | None) -> tuple[str, ...]:
    if factor_series is None or factor_series.empty:
        return ()
    finite = pd.to_numeric(factor_series, errors="coerce").dropna()
    if finite.empty:
        return ()
    quantiles = finite.quantile([0.05, 0.5, 0.95])
    return (
        f"mean={_fmt(float(finite.mean()))}",
        f"std={_fmt(float(finite.std(ddof=1)))}",
        f"p05={_fmt(float(quantiles.loc[0.05]))}",
        f"median={_fmt(float(quantiles.loc[0.5]))}",
        f"p95={_fmt(float(quantiles.loc[0.95]))}",
    )


def _turnover_snapshot(turnover_df: pd.DataFrame | None) -> str:
    if turnover_df is None or "turnover" not in turnover_df.columns:
        return "N/A"
    values = pd.to_numeric(turnover_df["turnover"], errors="coerce").dropna()
    if values.empty:
        return "N/A"
    return f"mean={_fmt(float(values.mean()))}; p95={_fmt(float(values.quantile(0.95)))}"


def _coverage_over_time_text(coverage_df: pd.DataFrame | None) -> str:
    if coverage_df is None or coverage_df.empty or "coverage" not in coverage_df.columns:
        return "N/A"
    values = pd.to_numeric(coverage_df["coverage"], errors="coerce").dropna()
    if values.empty:
        return "N/A"
    return (
        f"mean={_fmt(float(values.mean()))}; min={_fmt(float(values.min()))}; n_dates={len(values)}"
    )


def _cross_sectional_coverage(
    *,
    metrics: dict[str, object],
    coverage_df: pd.DataFrame | None,
) -> str:
    mean_assets = _safe_float(metrics.get("mean_eval_assets_per_date"))
    min_assets = _safe_float(metrics.get("min_eval_assets_per_date"))
    if mean_assets is None and coverage_df is not None and "n_assets" in coverage_df.columns:
        values = pd.to_numeric(coverage_df["n_assets"], errors="coerce").dropna()
        if not values.empty:
            mean_assets = float(values.mean())
            min_assets = float(values.min())
    return f"mean assets/date={_fmt(mean_assets)}; min assets/date={_fmt(min_assets)}"


def _stability_text(*, metrics: dict[str, object], rolling_df: pd.DataFrame | None) -> str:
    if rolling_df is not None and not rolling_df.empty:
        series = pd.to_numeric(rolling_df.get("rolling_mean_ic"), errors="coerce").dropna()
        if not series.empty:
            return f"rolling mean IC latest={_fmt(float(series.iloc[-1]))}; min={_fmt(float(series.min()))}"
    return (
        f"rolling_ic_positive_share={_fmt(_safe_float(metrics.get('rolling_ic_positive_share')))}; "
        f"rolling_long_short_positive_share={_fmt(_safe_float(metrics.get('rolling_long_short_positive_share')))}"
    )


def _winsorization_summary(preprocess: dict[str, object]) -> str:
    enabled = bool(preprocess.get("winsorize"))
    if not enabled:
        return "winsorization disabled"
    lower = _safe_float(preprocess.get("winsorize_lower"))
    upper = _safe_float(preprocess.get("winsorize_upper"))
    return f"winsorize={enabled}; lower={_fmt(lower)}; upper={_fmt(upper)}"


def _standardization_summary(
    *, preprocess: dict[str, object], neutralization: dict[str, object]
) -> str:
    standardization = safe_text(preprocess.get("standardization")) or "N/A"
    neutralization_enabled = bool(neutralization.get("enabled"))
    size_col = safe_text(neutralization.get("size_col"))
    industry_col = safe_text(neutralization.get("industry_col"))
    return (
        f"standardization={standardization}; neutralization_enabled={neutralization_enabled}; "
        f"size_col={size_col or 'N/A'}; industry_col={industry_col or 'N/A'}"
    )


def _pit_notes(integrity_report: dict[str, object]) -> str:
    summary = as_object_dict(integrity_report.get("summary"))
    n_warn = int(_float_or_default(summary.get("n_warn"), default=0.0))
    n_fail = int(_float_or_default(summary.get("n_fail"), default=0.0))
    checks = as_object_list(integrity_report.get("checks"))
    pit_flag = False
    for check_obj in checks:
        check = as_object_dict(check_obj)
        text = (
            (safe_text(check.get("message")) or "")
            + " "
            + (safe_text(check.get("remediation")) or "")
        )
        if "known_at" in text or "available_at" in text or "asof" in text.lower():
            pit_flag = True
            break
    if n_fail > 0:
        return f"integrity checks include failures (n_fail={n_fail}); inspect integrity_report.json"
    if n_warn > 0:
        return (
            f"integrity checks include warnings (n_warn={n_warn}); "
            f"publication-time fields review required={pit_flag}"
        )
    return "no integrity warnings; no future-data leakage flagged in current artifact checks"


def _implementation_notes(spec: dict[str, object]) -> str:
    direction = safe_text(spec.get("direction")) or "N/A"
    n_quantiles = safe_text(spec.get("n_quantiles")) or "N/A"
    rebalance = safe_text(spec.get("rebalance_frequency")) or "N/A"
    return f"direction={direction}; quantiles={n_quantiles}; rebalance={rebalance}"


def _holding_horizon_text(target: dict[str, object]) -> str:
    kind = safe_text(target.get("kind")) or "N/A"
    horizon = safe_text(target.get("horizon")) or "N/A"
    return f"{kind} horizon={horizon}"


def _formal_definition_text(
    *,
    factor_name: str,
    target: dict[str, object],
    spec: dict[str, object],
) -> str:
    target_kind = safe_text(target.get("kind")) or "forward_return"
    horizon = safe_text(target.get("horizon")) or "N/A"
    direction = safe_text(spec.get("direction")) or "long"
    return (
        f"{factor_name}(t) cross-sectional signal; predict {target_kind}(t+{horizon}); "
        f"portfolio direction={direction}"
    )


def _required_input_fields(spec: dict[str, object]) -> tuple[str, ...]:
    rows: list[str] = []
    for key in ["factor_path", "prices_path"]:
        if safe_text(spec.get(key)):
            rows.append(key)

    universe = as_object_dict(spec.get("universe"))
    if safe_text(universe.get("path")):
        rows.append("universe.path")

    neutralization = as_object_dict(spec.get("neutralization"))
    if safe_text(neutralization.get("exposures_path")):
        rows.append("neutralization.exposures_path")

    return tuple(rows)


def _parameter_settings(spec: dict[str, object]) -> tuple[str, ...]:
    rows: list[str] = []
    preprocess = as_object_dict(spec.get("preprocess"))
    for key in [
        "winsorize",
        "winsorize_lower",
        "winsorize_upper",
        "standardization",
        "min_group_size",
        "min_coverage",
    ]:
        if key in preprocess:
            rows.append(f"preprocess.{key}={preprocess.get(key)}")

    target = as_object_dict(spec.get("target"))
    for key in ["kind", "horizon"]:
        if key in target:
            rows.append(f"target.{key}={target.get(key)}")

    rows.append(f"n_quantiles={spec.get('n_quantiles', 'N/A')}")
    rows.append(f"rebalance_frequency={spec.get('rebalance_frequency', 'N/A')}")
    return tuple(rows)


def _lookback_parameters(
    *, target_horizon: float | None, preprocess: dict[str, object]
) -> tuple[str, ...]:
    rows: list[str] = []
    if target_horizon is not None:
        rows.append(f"target_horizon={int(target_horizon)}")
    min_group = _safe_float(preprocess.get("min_group_size"))
    if min_group is not None:
        rows.append(f"min_group_size={_fmt(min_group)}")
    coverage = _safe_float(preprocess.get("min_coverage"))
    if coverage is not None:
        rows.append(f"min_coverage={_fmt(coverage)}")
    return tuple(rows)


def _lag_rule(target: dict[str, object]) -> str:
    horizon = safe_text(target.get("horizon")) or "N/A"
    return f"signal(t) -> label(t+{horizon})"


def _expected_sign(direction: str | None) -> str:
    if not direction:
        return "N/A"
    text = direction.strip().lower()
    if text == "long":
        return "higher factor -> higher expected return"
    if text == "short":
        return "higher factor -> lower expected return"
    return direction


def _economic_intuition(*, factor_family: str, expected_sign: str) -> str:
    return f"family={factor_family}; expected relation={expected_sign}"


def _missingness_summary(missingness_mean: float | None) -> str:
    if missingness_mean is None:
        return "N/A"
    return f"mean missingness={missingness_mean:.4f}"


def _decay_profile(rolling_df: pd.DataFrame | None) -> tuple[str, ...]:
    if rolling_df is None or rolling_df.empty:
        return ()
    if "rolling_mean_ic" not in rolling_df.columns:
        return ()
    series = pd.to_numeric(rolling_df["rolling_mean_ic"], errors="coerce").dropna()
    if series.empty:
        return ()
    start = float(series.iloc[0])
    median = float(series.median())
    end = float(series.iloc[-1])
    return (
        f"rolling_mean_ic start={_fmt(start)}",
        f"rolling_mean_ic median={_fmt(median)}",
        f"rolling_mean_ic end={_fmt(end)}",
    )


def _horizon_analysis(portfolio_metrics: dict[str, object]) -> tuple[str, ...]:
    rows = as_object_list(portfolio_metrics.get("holding_period_sensitivity"))
    lines: list[str] = []
    for row_obj in rows:
        row = as_object_dict(row_obj)
        horizon = safe_text(row.get("holding_period"))
        mean_return = _safe_float(row.get("mean_portfolio_return"))
        cost_adj = _safe_float(row.get("mean_cost_adjusted_return_review_rate"))
        if horizon is None:
            continue
        lines.append(f"t+{horizon}: mean_return={_fmt(mean_return)}, cost_adj={_fmt(cost_adj)}")
    return tuple(lines)


def _spread_note(*, values: list[float | None], label: str) -> str:
    finite = [value for value in values if value is not None]
    if not finite:
        return "N/A"
    spread = max(finite) - min(finite)
    return f"{label} spread={_fmt(spread)} (min={_fmt(min(finite))}, max={_fmt(max(finite))})"


def _monotonicity_note(group_returns_df: pd.DataFrame | None) -> str:
    share = _monotonicity_share(group_returns_df)
    if share is None:
        return "N/A"
    return f"monotonic dates share={share:.2%}"


def _monotonicity_share(group_returns_df: pd.DataFrame | None) -> float | None:
    if group_returns_df is None or group_returns_df.empty:
        return None
    required = {"date", "group", "group_return"}
    if not required.issubset(set(group_returns_df.columns)):
        return None

    frame = group_returns_df.copy()
    frame["group"] = pd.to_numeric(frame["group"], errors="coerce")
    frame["group_return"] = pd.to_numeric(frame["group_return"], errors="coerce")
    frame = frame.dropna(subset=["date", "group", "group_return"])
    if frame.empty:
        return None

    grouped = frame.groupby("date")
    checks: list[float] = []
    for _, block in grouped:
        ordered = block.sort_values("group")
        values = ordered["group_return"].to_numpy()
        if len(values) < 3:
            continue
        diffs = pd.Series(values).diff().dropna()
        checks.append(1.0 if bool((diffs >= 0).all()) else 0.0)
    if not checks:
        return None

    return float(sum(checks) / len(checks))


def _oos_stability_share(metrics: dict[str, object]) -> float | None:
    rolling_ic = _safe_float(metrics.get("rolling_ic_positive_share"))
    rolling_long_short = _safe_float(metrics.get("rolling_long_short_positive_share"))
    parts = [value for value in (rolling_ic, rolling_long_short) if value is not None]
    if not parts:
        return None
    return float(sum(parts) / len(parts))


def _infer_factor_family(*, factor_name: str, case_name: str) -> str:
    text = f"{factor_name} {case_name}".lower()
    for keywords, family in _FACTOR_FAMILY_RULES:
        if any(keyword in text for keyword in keywords):
            return family
    return "custom / experimental"


def _load_factor_series(path: Path | None) -> pd.Series | None:
    if path is None or not path.exists():
        return None
    try:
        frame = pd.read_csv(path)
    except Exception:
        return None

    required = {"date", "asset", "value"}
    if not required.issubset(set(frame.columns)):
        return None

    frame = frame[["date", "asset", "value"]].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.dropna(subset=["date", "asset", "value"])
    if frame.empty:
        return None
    frame["key"] = frame["date"].dt.strftime("%Y-%m-%d") + "|" + frame["asset"].astype(str)
    series = frame.drop_duplicates(subset=["key"]).set_index("key")["value"].sort_index()
    return series


def _workflow_closure_artifact_paths_from_payload(
    *,
    payload: dict[str, object],
    comparison_path: Path,
    load_policy: ArtifactLoadPolicy,
    diagnostics: list[ArtifactLoadDiagnostic],
    warnings: list[str],
    errors: list[str],
) -> dict[str, Path]:
    base_dir = comparison_path.parent
    artifact_payload = as_object_dict(payload.get("workflow_closure_artifacts"))
    paths: dict[str, Path] = {}
    for object_label, filename, key in _WORKFLOW_CLOSURE_ARTIFACT_REQUIREMENTS:
        raw_pointer = artifact_payload.get(key) if artifact_payload else None
        candidate = _resolve_artifact_path(
            raw_pointer,
            base_dir=base_dir,
        )
        if candidate is not None:
            if candidate.exists():
                paths[key] = candidate
                continue
            severity = _artifact_issue_severity(
                required=(
                    load_policy.require_workflow_closure_artifacts
                    and load_policy.prefer_persisted_workflow_artifacts
                ),
            )
            _append_artifact_issue(
                code=_MISSING_WORKFLOW_ARTIFACT_CODE,
                severity=severity,
                artifact_type="workflow_closure_artifact",
                object_scope=object_label,
                message=(
                    f"workflow closure artifact path does not exist for {object_label}: {candidate}"
                ),
                diagnostics=diagnostics,
                warnings=warnings,
                errors=errors,
                mode=load_policy.mode,
                path=candidate,
                remediation_hint=(
                    "Update workflow_closure_artifacts pointer or regenerate "
                    "workflow closure artifacts."
                ),
            )
            continue
        if artifact_payload and raw_pointer is not None:
            severity = _artifact_issue_severity(
                required=(
                    load_policy.require_workflow_closure_artifacts
                    and load_policy.prefer_persisted_workflow_artifacts
                ),
            )
            _append_artifact_issue(
                code=_INVALID_WORKFLOW_ARTIFACT_CODE,
                severity=severity,
                artifact_type="workflow_closure_artifact",
                object_scope=object_label,
                message=(
                    f"workflow closure artifact pointer is invalid for {object_label}: key={key}"
                ),
                diagnostics=diagnostics,
                warnings=warnings,
                errors=errors,
                mode=load_policy.mode,
                remediation_hint=(
                    "Use a valid path in workflow_closure_artifacts or regenerate "
                    "workflow closure artifacts."
                ),
            )
            continue
        fallback_path = (base_dir / filename).resolve()
        if fallback_path.exists():
            paths[key] = fallback_path
            continue
        if not load_policy.prefer_persisted_workflow_artifacts:
            continue
        severity = _artifact_issue_severity(
            required=load_policy.require_workflow_closure_artifacts,
        )
        _append_artifact_issue(
            code=_MISSING_WORKFLOW_ARTIFACT_CODE,
            severity=severity,
            artifact_type="workflow_closure_artifact",
            object_scope=object_label,
            message=(
                f"workflow closure artifact missing for {object_label} ({filename}) near {base_dir}"
            ),
            diagnostics=diagnostics,
            warnings=warnings,
            errors=errors,
            mode=load_policy.mode,
            path=fallback_path,
            remediation_hint=(f"Persist {filename} near comparison output before strict loading."),
        )
    return paths


def _resolve_artifact_path(value: object, *, base_dir: Path) -> Path | None:
    text = safe_text(value)
    if not text:
        return None
    candidate = Path(text)
    if not candidate.is_absolute():
        candidate = base_dir / candidate
    return candidate.resolve()


def _load_workflow_artifact_payload(
    *,
    path: Path | None,
    artifact_name: str,
    object_label: str,
    required: bool,
    allow_fallback: bool,
    mode: ArtifactLoadMode,
    diagnostics: list[ArtifactLoadDiagnostic],
    warnings: list[str],
    errors: list[str],
) -> dict[str, object]:
    fallback_used = allow_fallback and not required
    if path is None:
        _append_artifact_issue(
            code=_MISSING_WORKFLOW_ARTIFACT_CODE,
            severity=_artifact_issue_severity(required=required),
            artifact_type="workflow_closure_artifact",
            object_scope=object_label,
            message=f"{object_label}: missing workflow artifact path ({artifact_name})",
            diagnostics=diagnostics,
            warnings=warnings,
            errors=errors,
            mode=mode,
            fallback_used=fallback_used,
            remediation_hint=(f"Persist workflow artifact {artifact_name} and store its pointer."),
        )
        if fallback_used:
            _append_artifact_issue(
                code=_FALLBACK_USED_CODE,
                severity="warning",
                artifact_type="workflow_closure_artifact",
                object_scope=object_label,
                message=(
                    f"{object_label}: fallback used because workflow artifact "
                    f"{artifact_name} is unavailable"
                ),
                diagnostics=diagnostics,
                warnings=warnings,
                errors=errors,
                mode=mode,
                fallback_used=True,
                remediation_hint=(f"Persist canonical {artifact_name} to disable this fallback."),
            )
        return {}
    payload = _load_optional_json(path)
    if not payload:
        _append_artifact_issue(
            code=_MISSING_WORKFLOW_ARTIFACT_CODE,
            severity=_artifact_issue_severity(required=required),
            artifact_type="workflow_closure_artifact",
            object_scope=object_label,
            message=f"{object_label}: missing or unreadable workflow artifact at {path}",
            diagnostics=diagnostics,
            warnings=warnings,
            errors=errors,
            mode=mode,
            path=path,
            fallback_used=fallback_used,
            remediation_hint=(f"Regenerate {artifact_name} and verify it is readable."),
        )
        if fallback_used:
            _append_artifact_issue(
                code=_FALLBACK_USED_CODE,
                severity="warning",
                artifact_type="workflow_closure_artifact",
                object_scope=object_label,
                message=(
                    f"{object_label}: fallback used because workflow artifact "
                    f"{artifact_name} could not be loaded"
                ),
                diagnostics=diagnostics,
                warnings=warnings,
                errors=errors,
                mode=mode,
                path=path,
                fallback_used=True,
                remediation_hint=(
                    f"Regenerate canonical {artifact_name} to disable this fallback."
                ),
            )
        return {}
    try:
        validate_level12_artifact_payload(
            payload,
            artifact_name=artifact_name,
            source=path,
        )
    except Exception as exc:
        _append_artifact_issue(
            code=_INVALID_WORKFLOW_ARTIFACT_CODE,
            severity=_artifact_issue_severity(required=required),
            artifact_type="workflow_closure_artifact",
            object_scope=object_label,
            message=(f"{object_label}: invalid workflow artifact payload ({artifact_name}): {exc}"),
            diagnostics=diagnostics,
            warnings=warnings,
            errors=errors,
            mode=mode,
            path=path,
            remediation_hint=(f"Fix schema issues in {artifact_name} and regenerate it."),
        )
        if fallback_used:
            _append_artifact_issue(
                code=_FALLBACK_USED_CODE,
                severity="warning",
                artifact_type="workflow_closure_artifact",
                object_scope=object_label,
                message=(
                    f"{object_label}: fallback used because workflow artifact "
                    f"{artifact_name} failed validation"
                ),
                diagnostics=diagnostics,
                warnings=warnings,
                errors=errors,
                mode=mode,
                path=path,
                fallback_used=True,
                remediation_hint=(
                    f"Regenerate canonical {artifact_name} to disable this fallback."
                ),
            )
        return {}
    return payload


def _load_factor_set_result_artifact(
    path: Path | None,
    *,
    fallback: FactorSetConstructionResult,
    load_policy: ArtifactLoadPolicy,
    diagnostics: list[ArtifactLoadDiagnostic],
    warnings: list[str],
    errors: list[str],
) -> FactorSetConstructionResult:
    required = (
        load_policy.require_workflow_closure_artifacts
        and load_policy.prefer_persisted_workflow_artifacts
    )
    payload = _load_workflow_artifact_payload(
        path=path,
        artifact_name="factor_set_result.json",
        object_label="factor_set_result",
        required=required,
        allow_fallback=load_policy.allow_workflow_fallback,
        mode=load_policy.mode,
        diagnostics=diagnostics,
        warnings=warnings,
        errors=errors,
    )
    if not payload:
        return fallback
    return _parse_factor_set_result_payload(payload, fallback=fallback)


def _parse_factor_set_result_payload(
    payload: dict[str, object],
    *,
    fallback: FactorSetConstructionResult,
) -> FactorSetConstructionResult:
    policy = as_object_dict(payload.get("policy"))
    config_obj = as_object_dict(policy.get("config"))
    base_cfg = fallback.config
    config = FactorSetConstructionConfig(
        policy_id=safe_text(policy.get("policy_id")) or base_cfg.policy_id,
        formula_text=safe_text(policy.get("formula_text")) or base_cfg.formula_text,
        selected_set_size=_safe_int(config_obj.get("selected_set_size"))
        or base_cfg.selected_set_size,
        candidate_set_size=_safe_int(config_obj.get("candidate_set_size"))
        or base_cfg.candidate_set_size,
        watchlist_set_size=_safe_int(config_obj.get("watchlist_set_size"))
        or base_cfg.watchlist_set_size,
        redundancy_correlation_max=_safe_float(config_obj.get("redundancy_correlation_max"))
        or base_cfg.redundancy_correlation_max,
        turnover_max=_safe_float(config_obj.get("turnover_max")) or base_cfg.turnover_max,
        oos_stability_min=_safe_float(config_obj.get("oos_stability_min"))
        or base_cfg.oos_stability_min,
        min_selected_score=_safe_float(config_obj.get("min_selected_score"))
        or base_cfg.min_selected_score,
        min_candidate_score=_safe_float(config_obj.get("min_candidate_score"))
        or base_cfg.min_candidate_score,
    )
    factor_sets: list[FactorSetDefinition] = []
    for raw in as_object_list(payload.get("factor_sets")):
        item = as_object_dict(raw)
        score = as_object_dict(item.get("score_summary"))
        factor_sets.append(
            FactorSetDefinition(
                factor_set_id=safe_text(item.get("factor_set_id")) or "N/A",
                label_zh=safe_text(item.get("label_zh")),
                factor_ids=tuple(parse_text_list(item.get("factor_ids"), split_semicolon=False)),
                factor_names=tuple(
                    parse_text_list(item.get("factor_names"), split_semicolon=False)
                ),
                source_shortlist_entries=tuple(
                    parse_text_list(item.get("source_shortlist_entries"), split_semicolon=False)
                ),
                construction_rule=safe_text(item.get("construction_rule")) or "N/A",
                status=safe_text(item.get("status")) or "candidate",
                rationale=tuple(parse_text_list(item.get("rationale"), split_semicolon=False)),
                rationale_zh=tuple(
                    parse_text_list(item.get("rationale_zh"), split_semicolon=False)
                ),
                warnings=tuple(parse_text_list(item.get("warnings"), split_semicolon=False)),
                score_summary=FactorSetScoreSummary(
                    mean_shortlist_score=_safe_float(score.get("mean_shortlist_score")),
                    mean_icir=_safe_float(score.get("mean_icir")),
                    mean_turnover=_safe_float(score.get("mean_turnover")),
                    mean_oos_stability_share=_safe_float(score.get("mean_oos_stability_share")),
                    max_pair_correlation=_safe_float(score.get("max_pair_correlation")),
                    family_balance_ratio=_safe_float(score.get("family_balance_ratio")),
                ),
            )
        )
    return FactorSetConstructionResult(
        config=config,
        factor_sets=tuple(factor_sets),
        selected_factor_set_ids=tuple(
            parse_text_list(payload.get("selected_factor_set_ids"), split_semicolon=False)
        ),
        recommendation_summary=tuple(
            parse_text_list(payload.get("recommendation_summary"), split_semicolon=False)
        ),
    )


def _load_candidate_recipe_generation_artifact(
    path: Path | None,
    *,
    fallback: CandidateRecipeGenerationResult,
    load_policy: ArtifactLoadPolicy,
    diagnostics: list[ArtifactLoadDiagnostic],
    warnings: list[str],
    errors: list[str],
) -> CandidateRecipeGenerationResult:
    required = (
        load_policy.require_workflow_closure_artifacts
        and load_policy.prefer_persisted_workflow_artifacts
    )
    payload = _load_workflow_artifact_payload(
        path=path,
        artifact_name="candidate_recipe_generation.json",
        object_label="candidate_recipe_generation",
        required=required,
        allow_fallback=load_policy.allow_workflow_fallback,
        mode=load_policy.mode,
        diagnostics=diagnostics,
        warnings=warnings,
        errors=errors,
    )
    if not payload:
        return fallback
    return _parse_candidate_recipe_generation_payload(payload, fallback=fallback)


def _parse_candidate_recipe_generation_payload(
    payload: dict[str, object],
    *,
    fallback: CandidateRecipeGenerationResult,
) -> CandidateRecipeGenerationResult:
    policy = as_object_dict(payload.get("policy"))
    config_obj = as_object_dict(policy.get("config"))
    base_cfg = fallback.config
    config = CandidateRecipeGenerationConfig(
        policy_id=safe_text(policy.get("policy_id")) or base_cfg.policy_id,
        formula_text=safe_text(policy.get("formula_text")) or base_cfg.formula_text,
        max_recipes_per_factor_set=_safe_int(config_obj.get("max_recipes_per_factor_set"))
        or base_cfg.max_recipes_per_factor_set,
        weighting_schemes=tuple(
            parse_text_list(config_obj.get("weighting_schemes"), split_semicolon=False)
        )
        or base_cfg.weighting_schemes,
        neutralization_modes=tuple(
            parse_text_list(config_obj.get("neutralization_modes"), split_semicolon=False)
        )
        or base_cfg.neutralization_modes,
        turnover_penalty_modes=tuple(
            parse_text_list(config_obj.get("turnover_penalty_modes"), split_semicolon=False)
        )
        or base_cfg.turnover_penalty_modes,
        benchmark_modes=tuple(
            parse_text_list(config_obj.get("benchmark_modes"), split_semicolon=False)
        )
        or base_cfg.benchmark_modes,
    )
    generated: list[CandidateRecipe] = []
    for raw in as_object_list(payload.get("generated_recipes")):
        item = as_object_dict(raw)
        generated.append(
            CandidateRecipe(
                recipe_id=safe_text(item.get("recipe_id")) or "N/A",
                recipe_name=safe_text(item.get("recipe_name")) or "N/A",
                source_factor_set_id=safe_text(item.get("source_factor_set_id")) or "N/A",
                source_factor_ids=tuple(
                    parse_text_list(item.get("source_factor_ids"), split_semicolon=False)
                ),
                construction_variant=safe_text(item.get("construction_variant")) or "N/A",
                weighting_scheme=safe_text(item.get("weighting_scheme")) or "N/A",
                neutralization_mode=safe_text(item.get("neutralization_mode")) or "N/A",
                turnover_penalty_mode=safe_text(item.get("turnover_penalty_mode")) or "N/A",
                benchmark_mode=safe_text(item.get("benchmark_mode")) or "N/A",
                rationale=tuple(parse_text_list(item.get("rationale"), split_semicolon=False)),
                assumptions=tuple(parse_text_list(item.get("assumptions"), split_semicolon=False)),
                warnings=tuple(parse_text_list(item.get("warnings"), split_semicolon=False)),
            )
        )
    return CandidateRecipeGenerationResult(
        config=config,
        generated_recipes=tuple(generated),
        recommendation_summary=tuple(
            parse_text_list(payload.get("recommendation_summary"), split_semicolon=False)
        ),
    )


def _load_winner_selection_artifact(
    path: Path | None,
    *,
    fallback: WinnerSelectionResult,
    load_policy: ArtifactLoadPolicy,
    diagnostics: list[ArtifactLoadDiagnostic],
    warnings: list[str],
    errors: list[str],
) -> WinnerSelectionResult:
    required = (
        load_policy.require_workflow_closure_artifacts
        and load_policy.prefer_persisted_workflow_artifacts
    )
    payload = _load_workflow_artifact_payload(
        path=path,
        artifact_name="winner_selection.json",
        object_label="winner_selection",
        required=required,
        allow_fallback=load_policy.allow_workflow_fallback,
        mode=load_policy.mode,
        diagnostics=diagnostics,
        warnings=warnings,
        errors=errors,
    )
    if not payload:
        return fallback
    return _parse_winner_selection_payload(payload, fallback=fallback)


def _parse_winner_selection_payload(
    payload: dict[str, object],
    *,
    fallback: WinnerSelectionResult,
) -> WinnerSelectionResult:
    policy = as_object_dict(payload.get("decision_policy"))
    score_table: list[tuple[str, float | None]] = []
    for raw in as_object_list(payload.get("score_table")):
        item = as_object_dict(raw)
        recipe_id = safe_text(item.get("recipe_id")) or ""
        if not recipe_id:
            continue
        score_table.append((recipe_id, _safe_float(item.get("composite_score"))))
    decision_reasons = tuple(
        parse_text_list(payload.get("decision_reasons"), split_semicolon=False)
    )
    challenger_reasons = tuple(
        parse_text_list(payload.get("challenger_reasons"), split_semicolon=False)
    )
    rejection_reasons = tuple(
        parse_text_list(payload.get("rejection_reasons"), split_semicolon=False)
    )
    next_actions = tuple(parse_text_list(payload.get("next_actions"), split_semicolon=False))
    decision_reasons_zh = tuple(
        parse_text_list(payload.get("decision_reasons_zh"), split_semicolon=False)
    ) or tuple(_display_text(item) for item in decision_reasons)
    challenger_reasons_zh = tuple(
        parse_text_list(payload.get("challenger_reasons_zh"), split_semicolon=False)
    ) or tuple(_display_text(item) for item in challenger_reasons)
    rejection_reasons_zh = tuple(
        parse_text_list(payload.get("rejection_reasons_zh"), split_semicolon=False)
    ) or tuple(_display_text(item) for item in rejection_reasons)
    next_actions_zh = tuple(
        parse_text_list(payload.get("next_actions_zh"), split_semicolon=False)
    ) or tuple(_display_text(item) for item in next_actions)
    return WinnerSelectionResult(
        decision_policy_id=(
            safe_text(policy.get("decision_policy_id")) or fallback.decision_policy_id
        ),
        winner_recipe_id=safe_text(payload.get("winner_recipe_id")) or "",
        challenger_recipe_ids=tuple(
            parse_text_list(payload.get("challenger_recipe_ids"), split_semicolon=False)
        ),
        watchlist_recipe_ids=tuple(
            parse_text_list(payload.get("watchlist_recipe_ids"), split_semicolon=False)
        ),
        rejected_recipe_ids=tuple(
            parse_text_list(payload.get("rejected_recipe_ids"), split_semicolon=False)
        ),
        policy_formula_text=(
            safe_text(policy.get("policy_formula_text")) or fallback.policy_formula_text or "N/A"
        ),
        decision_reasons=decision_reasons,
        decision_reasons_zh=decision_reasons_zh,
        challenger_reasons=challenger_reasons,
        challenger_reasons_zh=challenger_reasons_zh,
        rejection_reasons=rejection_reasons,
        rejection_reasons_zh=rejection_reasons_zh,
        next_actions=next_actions,
        next_actions_zh=next_actions_zh,
        score_table=tuple(score_table),
    )


def _load_next_step_recommendations_artifact(
    path: Path | None,
    *,
    fallback: NextStepRecommendationResult,
    load_policy: ArtifactLoadPolicy,
    diagnostics: list[ArtifactLoadDiagnostic],
    warnings: list[str],
    errors: list[str],
) -> NextStepRecommendationResult:
    required = (
        load_policy.require_workflow_closure_artifacts
        and load_policy.prefer_persisted_workflow_artifacts
    )
    payload = _load_workflow_artifact_payload(
        path=path,
        artifact_name="next_step_recommendations.json",
        object_label="next_step_recommendations",
        required=required,
        allow_fallback=load_policy.allow_workflow_fallback,
        mode=load_policy.mode,
        diagnostics=diagnostics,
        warnings=warnings,
        errors=errors,
    )
    if not payload:
        return fallback
    return _parse_next_step_recommendations_payload(payload, fallback=fallback)


def _parse_next_step_recommendations_payload(
    payload: dict[str, object],
    *,
    fallback: NextStepRecommendationResult,
) -> NextStepRecommendationResult:
    policy = as_object_dict(payload.get("policy"))
    recommendations: list[NextStepRecommendation] = []
    for raw in as_object_list(payload.get("recommendations")):
        item = as_object_dict(raw)
        recommendation_id = safe_text(item.get("recommendation_id")) or ""
        if not recommendation_id:
            continue
        recommendations.append(
            NextStepRecommendation(
                recommendation_id=recommendation_id,
                category=safe_text(item.get("category")) or "research_action",
                priority=safe_text(item.get("priority")) or "P2",
                action=safe_text(item.get("action")) or "N/A",
                action_text_zh=(
                    safe_text(item.get("action_text_zh"))
                    or _display_text(safe_text(item.get("action")) or "N/A")
                ),
                rationale=safe_text(item.get("rationale")) or "N/A",
                rationale_zh=(
                    safe_text(item.get("rationale_zh"))
                    or _display_text(safe_text(item.get("rationale")) or "N/A")
                ),
                label_zh=safe_text(item.get("label_zh"))
                or _display_next_step_category(
                    safe_text(item.get("category")) or "research_action"
                ),
                trigger_objects=tuple(
                    parse_text_list(item.get("triggered_by"), split_semicolon=False)
                ),
                supporting_evidence=tuple(
                    parse_text_list(
                        item.get("supporting_evidence"),
                        split_semicolon=False,
                    )
                ),
            )
        )
    summary = tuple(parse_text_list(payload.get("summary"), split_semicolon=False))
    summary_zh = tuple(parse_text_list(payload.get("summary_zh"), split_semicolon=False)) or tuple(
        _display_text(item) for item in summary
    )
    return NextStepRecommendationResult(
        policy_id=safe_text(policy.get("policy_id")) or fallback.policy_id,
        policy_formula_text=(
            safe_text(policy.get("policy_formula_text")) or fallback.policy_formula_text
        ),
        recommendations=tuple(recommendations),
        summary=summary,
        summary_zh=summary_zh,
    )


def _load_json(path: Path) -> dict[str, object]:
    payload = _load_optional_json(path)
    if not payload:
        raise FileNotFoundError(f"unable to read comparison payload: {path}")
    return payload


def _load_optional_json(path: Path | None) -> dict[str, object]:
    if path is None or not path.exists():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return as_object_dict(loaded)


def _load_optional_csv(path: Path | None) -> pd.DataFrame | None:
    if path is None or not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _to_path(value: object) -> Path | None:
    text = safe_text(value)
    if not text:
        return None
    return Path(text).resolve()


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        result = float(value)
        if math.isfinite(result):
            return result
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        result = float(text)
    except ValueError:
        return None
    if not math.isfinite(result):
        return None
    return result


def _safe_int(value: object) -> int | None:
    parsed = _safe_float(value)
    if parsed is None:
        return None
    if not float(parsed).is_integer():
        return None
    return int(parsed)


def _coalesce_float(primary: object, fallback: object) -> float | None:
    parsed = _safe_float(primary)
    if parsed is not None:
        return parsed
    return _safe_float(fallback)


def _rows_from_stats(value: object) -> tuple[tuple[str, float], ...]:
    if not isinstance(value, tuple):
        return ()
    rows: list[tuple[str, float]] = []
    for row in value:
        if not isinstance(row, tuple) or len(row) != 2:
            continue
        timestamp = safe_text(row[0])
        point = _safe_float(row[1])
        if timestamp and point is not None:
            rows.append((timestamp, point))
    return tuple(rows)


def _to_time_value_rows(value: object) -> tuple[tuple[str, float], ...]:
    if not isinstance(value, list):
        return ()
    rows: list[tuple[str, float]] = []
    for row in value:
        if not isinstance(row, list) or len(row) != 2:
            continue
        timestamp = safe_text(row[0])
        point = _safe_float(row[1])
        if timestamp and point is not None:
            rows.append((timestamp, point))
    return tuple(rows)


def _coalesce_rows(
    primary: tuple[tuple[str, float], ...],
    fallback: tuple[tuple[str, float], ...],
) -> tuple[tuple[str, float], ...]:
    return primary if primary else fallback


def _float_or_default(value: object, *, default: float) -> float:
    parsed = _safe_float(value)
    return parsed if parsed is not None else default


def _fmt(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.4f}"


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.2%}"


def _sort_value(value: float | None) -> str:
    if value is None:
        return "-999"
    return f"{value:.8f}"


def _h(value: str) -> str:
    return html.escape(value, quote=True)


def _slug(value: str) -> str:
    cleaned = []
    for char in value.lower():
        if char.isalnum():
            cleaned.append(char)
        else:
            cleaned.append("-")
    return "".join(cleaned).strip("-")
