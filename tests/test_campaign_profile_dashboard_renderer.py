from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import alpha_lab.reporting.renderers.campaign_profile_dashboard as dashboard_module
from alpha_lab.campaigns.profile_comparison import run_campaign_profile_comparison
from alpha_lab.reporting.renderers.campaign_profile_dashboard import (
    render_campaign_profile_dashboard_html,
    write_campaign_profile_dashboard_html,
)


def _build_dashboard(tmp_path: Path) -> tuple[Path, str]:
    output_root = tmp_path / "profile_compare_dashboard"
    comparison_result = run_campaign_profile_comparison(
        source="example",
        output_root_dir=output_root,
        profiles=("exploratory_screening", "default_research", "stricter_research"),
        render_report=False,
        clean_output=True,
    )
    dashboard_path = write_campaign_profile_dashboard_html(
        comparison_result.comparison_json_path,
    )
    html_text = dashboard_path.read_text(encoding="utf-8")
    return dashboard_path, html_text


def test_campaign_profile_dashboard_renderer_renders_factor_first_research_workbench(
    tmp_path: Path,
) -> None:
    dashboard_path, html_text = _build_dashboard(tmp_path)

    assert dashboard_path.exists()
    assert dashboard_path.name == "campaign_profile_dashboard_zh.html"
    assert '<html lang="zh-CN">' in html_text

    # Overview / factor / portfolio / backtest flow must be primary.
    assert "A. 研究首页总览 (Research Home / Overview)" in html_text
    assert "B. 因子库 (Factor Library)" in html_text
    assert "C. 因子详情页 (Factor Detail)" in html_text
    assert "D. 因子横向对比 (Cross-Factor Comparison)" in html_text
    assert "D2. 入选因子集合 (Selected Factor Sets)" in html_text
    assert "D3. 候选配方生成 (Candidate Recipe Generation)" in html_text
    assert "E. 组合构建 (Portfolio Construction)" in html_text
    assert "E2. 冠军方案选择 (Winner Selection)" in html_text
    assert "E3. 下一步建议 (Next-Step Recommendations)" in html_text
    assert "F. 回测评估 (Backtest Evaluation)" in html_text
    assert "G. 稳健性与审计（次级） (Robustness / Audit, Secondary)" in html_text

    # Overview command-center cards and summary blocks.
    assert "Candidate Factors" in html_text
    assert "Validated Factors" in html_text
    assert "Active Portfolio Recipes" in html_text
    assert "Completed Backtests" in html_text
    assert "Top Factors by Signal Quality" in html_text
    assert "Top Portfolios by Objective" in html_text
    assert "Recent Research Runs / Latest Updates" in html_text
    assert "Warnings / Failed Runs / Missing Coverage" in html_text

    # Factor library interactions.
    assert 'id="factor-search-input"' in html_text
    assert 'id="factor-family-filter"' in html_text
    assert 'id="factor-status-filter"' in html_text
    assert 'id="factor-sort-select"' in html_text
    assert 'id="factor-library-table"' in html_text

    # Factor detail validation metrics.
    assert "<th>信息系数均值 (IC Mean)</th>" in html_text
    assert "<th>秩信息系数均值 (Rank IC Mean)</th>" in html_text
    assert "<th>ICIR</th>" in html_text
    assert "<th>t统计代理 (t-stat proxy)</th>" in html_text
    assert "<th>期限分析 (t+1/t+5/t+10)</th>" in html_text
    assert "<th>分组收益差 (Quantile/decile return spread)</th>" in html_text
    assert "<th>单调性诊断 (Monotonicity diagnostics)</th>" in html_text

    # Portfolio recipe/risk controls.
    assert "<th>权重方式 (Weighting scheme)</th>" in html_text
    assert "<th>中性化约束 (Neutralization constraints)</th>" in html_text
    assert "<th>仓位/暴露约束 (Position limits / exposure controls)</th>" in html_text
    assert "<th>预期风险摘要 (Expected risk summary)</th>" in html_text
    assert "不可行配置告警 (Infeasible Configuration Warnings)" in html_text

    # Backtest evaluation metrics.
    assert "<th>年化收益 (Annualized Return)</th>" in html_text
    assert "<th>年化波动 (Annualized Volatility)</th>" in html_text
    assert "<th>Sharpe</th>" in html_text
    assert "<th>Sortino</th>" in html_text
    assert "<th>最大回撤 (Max Drawdown)</th>" in html_text
    assert "<th>Calmar</th>" in html_text
    assert "<th>信息比率 (Information Ratio)</th>" in html_text
    assert "<th>相对基准超额收益 (Excess Return vs Benchmark)</th>" in html_text
    assert "<th>跟踪误差 (Tracking Error)</th>" in html_text
    assert "<th>成本前/后收益 (Pre-cost vs Post-cost)</th>" in html_text
    assert "月度收益表 (Monthly Return Table)" in html_text
    assert "回撤表 (Drawdown Table)" in html_text
    assert "累计收益/NAV曲线 (Cumulative Return / NAV Chart)" in html_text

    # Cross-factor section essentials.
    assert "Redundancy / Correlation Matrix" in html_text
    assert "Cluster / Group View" in html_text
    assert "Shortlist Recommendation Area" in html_text
    assert "候选清单综合评分（规范优先） (Shortlist Composite Score, Canonical-First)" in html_text
    assert "公式 (Formula):" in html_text
    assert "Recommendation" in html_text
    assert "集合建议摘要 (Set Recommendation Summary)" in html_text
    assert "生成摘要 (Generation Summary)" in html_text

    # Recipe comparison + lineage sections.
    assert (
        "流程第4步：配方对比层 (Workflow Step 4: Recipe Comparison Layer, "
        "Canonical Portfolio Recipe + Backtest)" in html_text
    )
    assert "配方排行榜 (Recipe Leaderboard: Sharpe / Return / MDD / IR / Post-cost)" in html_text
    assert "为何配方A优于配方B (Why Recipe A Beats Recipe B)" in html_text
    assert "当前冠军方案 (Current Winner)" in html_text
    assert "建议策略ID (Recommendation Policy ID)" in html_text
    assert "研究血缘与实验登记 (Research Lineage / Registry)" in html_text
    assert "血缘链接 (Lineage Links)" in html_text
    assert "Trace canonical object flow" in html_text

    # Robustness must remain a downstream section.
    assert "次级模块：在因子/组合/回测审阅后" in html_text
    assert 'id="robustness-audit" class="panel panel-secondary"' in html_text

    overview_pos = html_text.index('id="overview"')
    factor_library_pos = html_text.index('id="factor-library"')
    factor_detail_pos = html_text.index('id="factor-detail"')
    cross_factor_pos = html_text.index('id="cross-factor"')
    factor_set_pos = html_text.index('id="selected-factor-sets"')
    candidate_recipe_pos = html_text.index('id="candidate-recipe-generation"')
    portfolio_pos = html_text.index('id="portfolio-construction"')
    winner_pos = html_text.index('id="winner-selection"')
    next_step_pos = html_text.index('id="next-step-recommendations"')
    backtest_pos = html_text.index('id="backtest-evaluation"')
    robustness_pos = html_text.index('id="robustness-audit"')

    assert (
        overview_pos
        < factor_library_pos
        < factor_detail_pos
        < cross_factor_pos
        < factor_set_pos
        < candidate_recipe_pos
        < portfolio_pos
        < winner_pos
        < next_step_pos
        < backtest_pos
        < robustness_pos
    )


def test_campaign_profile_dashboard_renderer_default_overwrite_behavior(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "profile_compare_dashboard_overwrite"
    comparison_result = run_campaign_profile_comparison(
        source="example",
        output_root_dir=output_root,
        profiles=("exploratory_screening", "default_research"),
        render_report=False,
        clean_output=True,
    )

    dashboard_path = write_campaign_profile_dashboard_html(comparison_result.comparison_json_path)
    assert dashboard_path.exists()

    with pytest.raises(FileExistsError):
        write_campaign_profile_dashboard_html(comparison_result.comparison_json_path)

    overwritten = write_campaign_profile_dashboard_html(
        comparison_result.comparison_json_path,
        overwrite=True,
    )
    assert overwritten == dashboard_path

    rendered = render_campaign_profile_dashboard_html(comparison_result.comparison_json_path)
    assert "Factor Research Workbench 因子研究工作台" in rendered
    assert "A. 研究首页总览 (Research Home / Overview)" in rendered


def test_campaign_profile_dashboard_factor_summary_display_fields_support_bilingual_name() -> None:
    html = dashboard_module._render_factor_library(
        (
            dashboard_module.FactorSummary(
                factor_id="f_demo",
                factor_name="quality_factor",
                display_name_zh="质量因子",
                short_description="quality signal",
                short_description_zh="质量信号",
                factor_family="quality",
                mathematical_definition="quality_factor(t)",
            ),
        )
    )
    assert "质量因子 (quality_factor)" in html
    assert "质量信号 (quality signal)" in html


def test_campaign_profile_dashboard_loader_prefers_canonical_artifacts_when_present(
    tmp_path: Path,
) -> None:
    case_dir = tmp_path / "case_with_canonical"
    case_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        case_dir / "metrics.json",
        {
            "metrics": {"factor_name": "legacy_factor", "mean_rank_ic": 0.10},
            "coverage_by_date_summary": {"n_dates": 10, "mean_coverage": 0.8, "min_coverage": 0.7},
            "portfolio_validation_summary": {"recommendation": "legacy"},
            "portfolio_validation_metrics": {
                "protocol_settings": {"weighting_scheme": {"default": "rank"}}
            },
            "portfolio_validation_package": {"schema_version": "1.0.0"},
        },
    )
    _write_json(
        case_dir / "run_manifest.json",
        {"spec": {"factor_name": "legacy_from_manifest", "target": {"horizon": 5}}},
    )
    _write_json(
        case_dir / "factor_definition.json",
        {
            "factor_name": "canonical_factor",
            "spec": {"factor_name": "canonical_factor", "target": {"horizon": 5}},
        },
    )
    _write_json(
        case_dir / "signal_validation.json",
        {
            "metrics": {"factor_name": "canonical_factor", "mean_rank_ic": 0.42},
            "coverage_by_date_summary": {"n_dates": 12, "mean_coverage": 0.9, "min_coverage": 0.8},
            "fallback_derived_fields": [],
        },
    )
    _write_json(
        case_dir / "portfolio_recipe.json",
        {
            "portfolio_validation_summary": {"recommendation": "canonical"},
            "portfolio_validation_metrics": {
                "protocol_settings": {"weighting_scheme": {"default": "long_only"}}
            },
            "portfolio_validation_package": {"schema_version": "1.0.0"},
            "fallback_derived_fields": ["turnover_penalty_settings"],
        },
    )
    _write_json(
        case_dir / "backtest_result.json",
        {
            "summary": {
                "pre_cost_return": 0.123,
                "post_cost_return": 0.111,
                "nav_points": [],
                "monthly_return_table": [],
                "drawdown_table": [],
            },
            "fallback_derived_fields": ["annualized_return"],
        },
    )

    artifacts = dashboard_module._load_case_artifacts(
        {"artifact_paths": {"output_dir": str(case_dir)}}
    )
    assert artifacts.metrics["factor_name"] == "canonical_factor"
    assert artifacts.metrics["mean_rank_ic"] == 0.42
    assert artifacts.portfolio_validation_summary["recommendation"] == "canonical"
    assert artifacts.backtest_result_payload["summary"]["pre_cost_return"] == 0.123
    assert artifacts.fallback_derived_fields["backtest_result.json"] == ("annualized_return",)


def test_campaign_profile_dashboard_loader_legacy_fallback_without_canonical_artifacts(
    tmp_path: Path,
) -> None:
    case_dir = tmp_path / "case_legacy_only"
    case_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        case_dir / "metrics.json",
        {
            "metrics": {"factor_name": "legacy_factor", "mean_rank_ic": 0.15},
            "coverage_by_date_summary": {"n_dates": 8, "mean_coverage": 0.75, "min_coverage": 0.6},
            "portfolio_validation_summary": {"recommendation": "legacy"},
            "portfolio_validation_metrics": {
                "protocol_settings": {"weighting_scheme": {"default": "rank"}}
            },
            "portfolio_validation_package": {"schema_version": "1.0.0"},
        },
    )
    _write_json(
        case_dir / "run_manifest.json",
        {"spec": {"factor_name": "legacy_from_manifest", "target": {"horizon": 5}}},
    )

    artifacts = dashboard_module._load_case_artifacts(
        {"artifact_paths": {"output_dir": str(case_dir)}}
    )
    assert artifacts.metrics["factor_name"] == "legacy_factor"
    assert artifacts.metrics["mean_rank_ic"] == 0.15
    assert artifacts.portfolio_validation_summary["recommendation"] == "legacy"
    assert artifacts.signal_validation_payload == {}
    assert artifacts.portfolio_recipe_payload == {}
    assert artifacts.backtest_result_payload == {}


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_minimal_comparison_json(path: Path, *, case_output_dir: Path) -> Path:
    payload: dict[str, object] = {
        "profiles": ["default_research"],
        "default_profile": "default_research",
        "generated_at_utc": "2026-01-20T08:00:00+00:00",
        "case_comparison": [
            {
                "case_name": "case_alpha",
                "case_description": "minimal test case",
                "profiles": {
                    "default_research": {
                        "status": "success",
                        "artifact_paths": {"output_dir": str(case_output_dir)},
                    }
                },
            }
        ],
    }
    _write_json(path, payload)
    return path


def test_campaign_profile_dashboard_portfolio_recipe_prefers_canonical_controls() -> None:
    recipe = dashboard_module._build_portfolio_recipe(
        case_name="demo_case",
        factor_name="demo_factor",
        profile_payload={},
        metrics={
            "research_evaluation_snapshot": {
                "level2_portfolio_validation": {"max_mean_turnover_warn": 0.42}
            },
            "transaction_cost_one_way_rate": 0.001,
            "base_weighting_method": "rank",
            "rebalance_frequency": "W",
        },
        portfolio_recipe_payload={
            "turnover_penalty_settings": "canonical-turnover",
            "transaction_cost_assumptions": "canonical-cost",
            "position_limits": "canonical-limits",
        },
        portfolio_summary={
            "base_mean_portfolio_return": 0.01,
            "base_mean_turnover": 0.20,
        },
        portfolio_metrics={
            "protocol_settings": {
                "weighting_scheme": {"default": "rank"},
                "transaction_cost_sensitivity": [0.001, 0.002],
            },
            "concentration_exposure_diagnostics": {
                "max_abs_weight_mean": 0.30,
                "effective_names_mean": 15.0,
            },
            "scenario_metrics": [],
        },
        spec={"rebalance_frequency": "W"},
        universe={"name": "all_a_share"},
        neutralization={},
    )

    assert recipe.turnover_penalty_settings == "canonical-turnover"
    assert recipe.transaction_cost_assumptions == "canonical-cost"
    assert recipe.position_limits == "canonical-limits"


def test_campaign_profile_dashboard_backtest_prefers_canonical_backtest_result_fields() -> None:
    group_returns = pd.DataFrame(
        {
            "date": [
                "2026-01-01",
                "2026-01-01",
                "2026-01-08",
                "2026-01-08",
                "2026-01-15",
                "2026-01-15",
            ],
            "group": [1, 5, 1, 5, 1, 5],
            "group_return": [-0.01, 0.01, -0.03, 0.02, -0.02, 0.03],
        }
    )
    backtest = dashboard_module._build_backtest_summary(
        recipe_id="recipe-demo",
        factor_id="factor-demo",
        metrics={},
        portfolio_summary={},
        portfolio_metrics={
            "scenario_metrics": [
                {
                    "weighting_method": "rank",
                    "portfolio_ir": 0.11,
                    "portfolio_hit_rate": 0.62,
                    "mean_turnover": 0.19,
                    "n_return_dates": 12,
                }
            ],
            "concentration_exposure_diagnostics": {
                "max_abs_weight_mean": 0.22,
                "top5_abs_weight_share_mean": 0.44,
                "effective_names_mean": 17.0,
            },
        },
        backtest_payload={
            "summary": {
                "annualized_return": 0.99,
                "annualized_volatility": 0.21,
                "sharpe": 1.5,
                "sortino": 1.8,
                "max_drawdown": -0.08,
                "calmar": 12.0,
                "rolling_sharpe": 1.2,
                "rolling_drawdown": -0.03,
                "nav_points": [["2026-01-01", 1.0], ["2026-01-08", 1.1]],
                "monthly_return_table": [["2026-01", 0.07]],
                "drawdown_table": [["2026-01-08", -0.03]],
                "subperiod_analysis": "canonical-subperiod",
                "regime_analysis": "canonical-regime",
            }
        },
        group_returns_df=group_returns,
        turnover_df=None,
        rebalance_frequency="W",
    )

    assert backtest.annualized_return == pytest.approx(0.99)
    assert backtest.annualized_volatility == pytest.approx(0.21)
    assert backtest.sortino == pytest.approx(1.8)
    assert backtest.max_drawdown == pytest.approx(-0.08)
    assert backtest.calmar == pytest.approx(12.0)
    assert backtest.rolling_sharpe == pytest.approx(1.2)
    assert backtest.rolling_drawdown == pytest.approx(-0.03)
    assert backtest.nav_points == (("2026-01-01", 1.0), ("2026-01-08", 1.1))
    assert backtest.monthly_return_table == (("2026-01", 0.07),)
    assert backtest.drawdown_table == (("2026-01-08", -0.03),)
    assert backtest.subperiod_analysis == "canonical-subperiod"
    assert backtest.regime_analysis == "canonical-regime"


def test_campaign_profile_dashboard_shortlist_scoring_respects_redundancy_and_thresholds() -> None:
    rows = [
        dashboard_module.FactorComparisonRow(
            factor_id="f1",
            factor_name="factor_1",
            factor_family="value",
            ic_mean=0.070,
            rank_ic_mean=0.090,
            icir=1.10,
            turnover=0.20,
            monotonicity_share=0.90,
            oos_stability_share=0.88,
        ),
        dashboard_module.FactorComparisonRow(
            factor_id="f2",
            factor_name="factor_2",
            factor_family="value",
            ic_mean=0.068,
            rank_ic_mean=0.088,
            icir=1.08,
            turnover=0.22,
            monotonicity_share=0.89,
            oos_stability_share=0.86,
        ),
        dashboard_module.FactorComparisonRow(
            factor_id="f3",
            factor_name="factor_3",
            factor_family="reversal",
            ic_mean=0.004,
            rank_ic_mean=0.010,
            icir=0.10,
            turnover=1.15,
            monotonicity_share=0.42,
            oos_stability_share=0.40,
        ),
    ]
    corr = (
        ("f1", (("f1", 1.0), ("f2", 0.85), ("f3", 0.05))),
        ("f2", (("f1", 0.85), ("f2", 1.0), ("f3", 0.06))),
        ("f3", (("f1", 0.05), ("f2", 0.06), ("f3", 1.0))),
    )

    shortlist = dashboard_module._build_factor_shortlist_result(
        comparison_rows=rows,
        correlation_matrix=corr,
        config=dashboard_module._DEFAULT_SHORTLIST_CONFIG,
    )
    by_id = {entry.factor_id: entry for entry in shortlist.entries}

    assert shortlist.config.formula.startswith("score = weighted_average")
    keep_ids = [
        factor_id for factor_id in ("f1", "f2") if by_id[factor_id].recommendation == "keep"
    ]
    watch_ids = [
        factor_id for factor_id in ("f1", "f2") if by_id[factor_id].recommendation == "watchlist"
    ]
    assert len(keep_ids) == 1
    assert len(watch_ids) == 1
    assert by_id[watch_ids[0]].redundancy_with == keep_ids[0]
    assert by_id["f3"].recommendation == "drop"
    assert "redundant with selected factor" in "; ".join(by_id[watch_ids[0]].rationale)


def test_campaign_profile_dashboard_factor_set_builder_creates_explicit_status_objects() -> None:
    rows = [
        dashboard_module.FactorComparisonRow(
            factor_id="f1",
            factor_name="factor_1",
            factor_family="value",
            ic_mean=0.070,
            rank_ic_mean=0.090,
            icir=1.10,
            turnover=0.20,
            monotonicity_share=0.90,
            oos_stability_share=0.88,
        ),
        dashboard_module.FactorComparisonRow(
            factor_id="f2",
            factor_name="factor_2",
            factor_family="value",
            ic_mean=0.068,
            rank_ic_mean=0.088,
            icir=1.08,
            turnover=0.22,
            monotonicity_share=0.89,
            oos_stability_share=0.86,
        ),
        dashboard_module.FactorComparisonRow(
            factor_id="f3",
            factor_name="factor_3",
            factor_family="momentum",
            ic_mean=0.052,
            rank_ic_mean=0.072,
            icir=0.78,
            turnover=0.48,
            monotonicity_share=0.77,
            oos_stability_share=0.74,
        ),
        dashboard_module.FactorComparisonRow(
            factor_id="f4",
            factor_name="factor_4",
            factor_family="reversal",
            ic_mean=0.004,
            rank_ic_mean=0.009,
            icir=0.12,
            turnover=1.15,
            monotonicity_share=0.40,
            oos_stability_share=0.35,
        ),
    ]
    corr = (
        ("f1", (("f1", 1.0), ("f2", 0.86), ("f3", 0.25), ("f4", 0.05))),
        ("f2", (("f1", 0.86), ("f2", 1.0), ("f3", 0.24), ("f4", 0.02))),
        ("f3", (("f1", 0.25), ("f2", 0.24), ("f3", 1.0), ("f4", 0.07))),
        ("f4", (("f1", 0.05), ("f2", 0.02), ("f3", 0.07), ("f4", 1.0))),
    )
    shortlist = dashboard_module._build_factor_shortlist_result(
        comparison_rows=rows,
        correlation_matrix=corr,
        config=dashboard_module._DEFAULT_SHORTLIST_CONFIG,
    )
    factor_sets = dashboard_module._build_factor_set_result(
        shortlist=shortlist,
        comparison_rows=rows,
        correlation_matrix=corr,
        config=dashboard_module._DEFAULT_FACTOR_SET_CONFIG,
    )

    statuses = {item.status for item in factor_sets.factor_sets}
    assert {"selected", "candidate", "watchlist", "rejected"}.issubset(statuses)
    selected_set = next(item for item in factor_sets.factor_sets if item.status == "selected")
    assert selected_set.factor_set_id.startswith("set-selected")
    assert selected_set.factor_ids
    assert selected_set.source_shortlist_entries
    assert "high signal quality" in "; ".join(selected_set.rationale)
    assert selected_set.score_summary.mean_shortlist_score is not None
    assert factor_sets.recommendation_summary


def test_campaign_profile_dashboard_candidate_recipe_generation_tracks_source_factor_sets() -> None:
    factor_sets = (
        dashboard_module.FactorSetDefinition(
            factor_set_id="set-selected-core-v1",
            factor_ids=("f1", "f3"),
            factor_names=("factor_1", "factor_3"),
            construction_rule="selected_core_top_keep_by_shortlist_score",
            status="selected",
            score_summary=dashboard_module.FactorSetScoreSummary(
                mean_shortlist_score=0.81,
                mean_turnover=0.36,
                mean_oos_stability_share=0.76,
                max_pair_correlation=0.32,
                family_balance_ratio=1.00,
            ),
        ),
    )
    generated = dashboard_module._build_candidate_recipe_generation_result(
        factor_sets=factor_sets,
        config=dashboard_module._DEFAULT_CANDIDATE_RECIPE_CONFIG,
    )
    recipe_summaries = dashboard_module._candidate_recipes_to_portfolio_summaries(
        generated=generated.generated_recipes,
        factor_sets=factor_sets,
    )

    assert generated.generated_recipes
    first = generated.generated_recipes[0]
    assert first.source_factor_set_id == "set-selected-core-v1"
    assert first.construction_variant != "N/A"
    assert first.weighting_scheme in {"rank", "equal_weight"}
    assert recipe_summaries[0].recipe_id == first.recipe_id
    assert "no canonical backtest result yet" in "; ".join(
        recipe_summaries[0].infeasible_configuration_warnings
    )


def test_campaign_profile_dashboard_winner_selection_and_next_steps_are_explicit() -> None:
    factor_sets = dashboard_module.FactorSetConstructionResult(
        config=dashboard_module._DEFAULT_FACTOR_SET_CONFIG,
        factor_sets=(
            dashboard_module.FactorSetDefinition(
                factor_set_id="set-selected-core-v1",
                factor_ids=("f1", "f3"),
                factor_names=("factor_1", "factor_3"),
                construction_rule="selected_core_top_keep_by_shortlist_score",
                status="selected",
                score_summary=dashboard_module.FactorSetScoreSummary(
                    mean_shortlist_score=0.82,
                    mean_turnover=0.34,
                    mean_oos_stability_share=0.78,
                    max_pair_correlation=0.30,
                    family_balance_ratio=1.00,
                ),
            ),
            dashboard_module.FactorSetDefinition(
                factor_set_id="set-candidate-diversified-v1",
                factor_ids=("f2",),
                factor_names=("factor_2",),
                construction_rule="candidate_diversified_keep_watchlist_mix_low_redundancy",
                status="candidate",
                score_summary=dashboard_module.FactorSetScoreSummary(
                    mean_shortlist_score=0.58,
                    mean_turnover=0.62,
                    mean_oos_stability_share=0.60,
                    max_pair_correlation=0.20,
                    family_balance_ratio=1.00,
                ),
            ),
        ),
        selected_factor_set_ids=("set-selected-core-v1",),
        recommendation_summary=(),
    )
    candidate_generation = dashboard_module.CandidateRecipeGenerationResult(
        config=dashboard_module._DEFAULT_CANDIDATE_RECIPE_CONFIG,
        generated_recipes=(
            dashboard_module.CandidateRecipe(
                recipe_id="candidate-set-candidate-diversified-v1-v1",
                recipe_name="Candidate set-candidate-diversified-v1 v1",
                source_factor_set_id="set-candidate-diversified-v1",
                source_factor_ids=("f2",),
                construction_variant="alpha_rank_unneutralized_balanced",
                weighting_scheme="rank",
                neutralization_mode="neutralization_off",
                turnover_penalty_mode="balanced",
                benchmark_mode="absolute",
            ),
        ),
        recommendation_summary=(),
    )
    rows = (
        dashboard_module.RecipeComparisonRow(
            recipe_id="recipe-a",
            recipe_name="Recipe A",
            selected_factors=("f1", "f3"),
            factor_family_mix=("value", "momentum"),
            sharpe=1.25,
            annualized_return=0.19,
            max_drawdown=-0.10,
            information_ratio=0.60,
            post_cost_return=0.15,
        ),
        dashboard_module.RecipeComparisonRow(
            recipe_id="recipe-b",
            recipe_name="Recipe B",
            selected_factors=("f1",),
            factor_family_mix=("value",),
            sharpe=0.95,
            annualized_return=0.14,
            max_drawdown=-0.14,
            information_ratio=0.45,
            post_cost_return=0.10,
        ),
        dashboard_module.RecipeComparisonRow(
            recipe_id="recipe-c",
            recipe_name="Recipe C",
            selected_factors=("f4",),
            factor_family_mix=("reversal",),
            sharpe=0.20,
            annualized_return=0.03,
            max_drawdown=-0.45,
            information_ratio=0.05,
            post_cost_return=-0.02,
        ),
        dashboard_module.RecipeComparisonRow(
            recipe_id="candidate-set-candidate-diversified-v1-v1",
            recipe_name="Candidate Recipe",
            selected_factors=("f2",),
            factor_family_mix=("value",),
        ),
    )
    recipe_comparison = dashboard_module.RecipeComparisonView(rows=rows)
    winner = dashboard_module._build_winner_selection_result(
        recipe_comparison=recipe_comparison,
        factor_sets=factor_sets,
        candidate_recipe_generation=candidate_generation,
        policy=dashboard_module._DEFAULT_WINNER_SELECTION_POLICY,
    )
    shortlist = dashboard_module.FactorShortlistResult(
        config=dashboard_module._DEFAULT_SHORTLIST_CONFIG,
        selected_factor_ids=("f1",),
        entries=(
            dashboard_module.FactorShortlistEntry(
                rank=1,
                factor_id="f1",
                factor_name="factor_1",
                factor_family="value",
                recommendation="keep",
            ),
            dashboard_module.FactorShortlistEntry(
                rank=2,
                factor_id="f2",
                factor_name="factor_2",
                factor_family="value",
                recommendation="watchlist",
                redundancy_with="f1",
                rationale=("redundant with selected factor f1",),
            ),
        ),
    )
    recommendations = dashboard_module._build_next_step_recommendations(
        shortlist=shortlist,
        factor_sets=factor_sets,
        candidate_recipe_generation=candidate_generation,
        recipe_comparison=recipe_comparison,
        winner_selection=winner,
    )

    assert winner.winner_recipe_id == "recipe-a"
    assert "recipe-b" in winner.challenger_recipe_ids
    assert "candidate-set-candidate-diversified-v1-v1" in winner.watchlist_recipe_ids
    assert "recipe-c" in winner.rejected_recipe_ids
    assert "weights(" in winner.policy_formula_text
    assert winner.decision_reasons_zh
    assert winner.next_actions_zh
    assert recommendations.recommendations
    assert recommendations.summary_zh
    assert all(
        item.action_text_zh is not None and item.rationale_zh is not None
        for item in recommendations.recommendations
    )
    assert any("promote recipe-a" in item.action for item in recommendations.recommendations)
    assert any("archive weak candidates" in item.action for item in recommendations.recommendations)


def test_campaign_profile_dashboard_prefers_persisted_workflow_closure_artifacts(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "profile_compare_dashboard_persisted_workflow"
    comparison_result = run_campaign_profile_comparison(
        source="example",
        output_root_dir=output_root,
        profiles=("exploratory_screening", "default_research"),
        render_report=False,
        clean_output=True,
    )
    workflow_paths = dashboard_module.persist_workflow_closure_artifacts(
        comparison_result.comparison_json_path
    )

    factor_set_path = workflow_paths["factor_set_result_json_path"]
    factor_set_payload = json.loads(factor_set_path.read_text(encoding="utf-8"))
    factor_sets = factor_set_payload["factor_sets"]
    assert isinstance(factor_sets, list)
    assert factor_sets
    first_set = factor_sets[0]
    assert isinstance(first_set, dict)
    first_set["factor_set_id"] = "persisted-factor-set-vtest"
    factor_set_payload["selected_factor_set_ids"] = ["persisted-factor-set-vtest"]
    factor_set_path.write_text(
        json.dumps(factor_set_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    winner_path = workflow_paths["winner_selection_json_path"]
    winner_payload = json.loads(winner_path.read_text(encoding="utf-8"))
    winner_payload["winner_recipe_id"] = "persisted-winner-recipe"
    winner_path.write_text(
        json.dumps(winner_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    next_step_path = workflow_paths["next_step_recommendations_json_path"]
    next_step_payload = json.loads(next_step_path.read_text(encoding="utf-8"))
    recs = next_step_payload["recommendations"]
    assert isinstance(recs, list)
    assert recs
    first_rec = recs[0]
    assert isinstance(first_rec, dict)
    first_rec["action"] = "persisted next-step action"
    next_step_path.write_text(
        json.dumps(next_step_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    diagnostics_path = workflow_paths["artifact_load_diagnostics_json_path"]
    diagnostics_payload = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    assert diagnostics_payload["artifact_type"] == "alpha_lab_artifact_load_diagnostics"
    assert diagnostics_payload["artifact_load_mode"] == "permissive"
    assert isinstance(diagnostics_payload["diagnostics"], list)
    manifest_path = workflow_paths["research_artifact_manifest_json_path"]
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest_payload["artifact_type"] == "alpha_lab_research_artifact_manifest"
    manifest_entries = manifest_payload["artifact_entries"]
    assert isinstance(manifest_entries, list)
    assert any(
        isinstance(item, dict)
        and item.get("artifact_name") == "artifact_load_diagnostics.json"
        and item.get("required_in_strict_mode") is True
        for item in manifest_entries
    )
    assert any(
        isinstance(item, dict)
        and item.get("artifact_name") == "research_artifact_manifest.json"
        and item.get("required_in_strict_mode") is False
        for item in manifest_entries
    )
    comparison_payload = json.loads(
        Path(comparison_result.comparison_json_path).read_text(encoding="utf-8")
    )
    workflow_context = comparison_payload.get("workflow_closure_artifacts")
    assert isinstance(workflow_context, dict)
    assert workflow_context["artifact_load_diagnostics_json_path"] == str(diagnostics_path)
    assert workflow_context["research_artifact_manifest_json_path"] == str(manifest_path)
    summary = comparison_payload.get("campaign_level_summary")
    assert isinstance(summary, dict)
    summary_workflow = summary.get("workflow_closure_artifacts")
    assert isinstance(summary_workflow, dict)
    assert summary_workflow["artifact_load_diagnostics_json_path"] == str(diagnostics_path)
    assert summary_workflow["research_artifact_manifest_json_path"] == str(manifest_path)
    profile_runs = comparison_payload.get("profile_runs")
    assert isinstance(profile_runs, list)
    assert profile_runs
    for run in profile_runs:
        assert isinstance(run, dict)
        campaign_artifacts = run.get("campaign_artifacts")
        assert isinstance(campaign_artifacts, dict)
        run_workflow = campaign_artifacts.get("workflow_closure_artifacts")
        assert isinstance(run_workflow, dict)
        assert run_workflow["artifact_load_diagnostics_json_path"] == str(diagnostics_path)
        assert run_workflow["research_artifact_manifest_json_path"] == str(manifest_path)

    data = dashboard_module._build_research_dashboard_data(comparison_result.comparison_json_path)
    assert data.factor_sets.factor_sets[0].factor_set_id == "persisted-factor-set-vtest"
    assert data.winner_selection.winner_recipe_id == "persisted-winner-recipe"
    assert any(
        item.action == "persisted next-step action"
        for item in data.next_step_recommendations.recommendations
    )


def test_campaign_profile_dashboard_permissive_mode_keeps_fallback_when_artifacts_missing(
    tmp_path: Path,
) -> None:
    case_dir = tmp_path / "case_missing_artifacts"
    case_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        case_dir / "metrics.json",
        {
            "metrics": {"factor_name": "legacy_factor", "mean_rank_ic": 0.12},
            "coverage_by_date_summary": {"n_dates": 8, "mean_coverage": 0.75, "min_coverage": 0.6},
            "portfolio_validation_summary": {"recommendation": "legacy"},
            "portfolio_validation_metrics": {
                "protocol_settings": {"weighting_scheme": {"default": "rank"}}
            },
            "portfolio_validation_package": {"schema_version": "1.0.0"},
        },
    )
    _write_json(
        case_dir / "run_manifest.json",
        {"spec": {"factor_name": "legacy_from_manifest", "target": {"horizon": 5}}},
    )
    comparison_json = _write_minimal_comparison_json(
        tmp_path / "campaign_profile_comparison.json",
        case_output_dir=case_dir,
    )

    data = dashboard_module._build_research_dashboard_data(
        comparison_json,
        artifact_load_mode="permissive",
    )

    assert data.artifact_load_mode == "permissive"
    assert data.factor_summaries
    assert data.factor_summaries[0].factor_name == "legacy_factor"
    assert data.factor_sets.factor_sets
    assert data.candidate_recipe_generation.generated_recipes
    assert data.winner_selection is not None
    assert data.next_step_recommendations.recommendations
    assert any("factor_definition" in row for row in data.artifact_load_warnings)
    assert any("workflow closure artifact missing" in row for row in data.artifact_load_warnings)
    factor_definition_diagnostics = [
        item for item in data.artifact_load_diagnostics if item.object_scope == "factor_definition"
    ]
    assert any(
        item.code == "MISSING_CANONICAL_ARTIFACT"
        and item.severity == "warning"
        and item.case_name == "case_alpha"
        and item.profile_name == "default_research"
        and item.mode == "permissive"
        for item in factor_definition_diagnostics
    )
    assert any(
        item.code == "FALLBACK_USED" and item.fallback_used and item.mode == "permissive"
        for item in factor_definition_diagnostics
    )


def test_persist_workflow_closure_artifacts_writes_structured_diagnostics_artifact(
    tmp_path: Path,
) -> None:
    case_dir = tmp_path / "case_persist_diagnostics"
    case_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        case_dir / "metrics.json",
        {
            "metrics": {"factor_name": "legacy_factor", "mean_rank_ic": 0.12},
            "coverage_by_date_summary": {"n_dates": 8, "mean_coverage": 0.75, "min_coverage": 0.6},
            "portfolio_validation_summary": {"recommendation": "legacy"},
            "portfolio_validation_metrics": {
                "protocol_settings": {"weighting_scheme": {"default": "rank"}}
            },
            "portfolio_validation_package": {"schema_version": "1.0.0"},
        },
    )
    _write_json(
        case_dir / "run_manifest.json",
        {"spec": {"factor_name": "legacy_from_manifest", "target": {"horizon": 5}}},
    )
    comparison_json = _write_minimal_comparison_json(
        tmp_path / "campaign_profile_comparison_persist_diagnostics.json",
        case_output_dir=case_dir,
    )

    workflow_paths = dashboard_module.persist_workflow_closure_artifacts(comparison_json)
    diagnostics_path = workflow_paths["artifact_load_diagnostics_json_path"]
    assert diagnostics_path.exists()
    payload = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    assert payload["artifact_type"] == "alpha_lab_artifact_load_diagnostics"
    assert payload["artifact_load_mode"] == "permissive"
    diagnostics = payload["diagnostics"]
    assert isinstance(diagnostics, list)
    assert diagnostics
    first = diagnostics[0]
    assert isinstance(first, dict)
    assert {
        "code",
        "severity",
        "artifact_type",
        "object_scope",
        "message",
        "mode",
        "fallback_used",
    } <= set(first)
    assert any(
        isinstance(item, dict) and item.get("code") == "MISSING_CANONICAL_ARTIFACT"
        for item in diagnostics
    )

    comparison_payload = json.loads(Path(comparison_json).read_text(encoding="utf-8"))
    workflow_context = comparison_payload.get("workflow_closure_artifacts")
    assert isinstance(workflow_context, dict)
    assert workflow_context["artifact_load_diagnostics_json_path"] == str(diagnostics_path)
    summary = comparison_payload.get("campaign_level_summary")
    assert isinstance(summary, dict)
    summary_workflow = summary.get("workflow_closure_artifacts")
    assert isinstance(summary_workflow, dict)
    assert summary_workflow["artifact_load_diagnostics_json_path"] == str(diagnostics_path)
    assert "research_artifact_manifest_json_path" in workflow_paths
    manifest_path = workflow_paths["research_artifact_manifest_json_path"]
    assert manifest_path.exists()
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest_payload["artifact_type"] == "alpha_lab_research_artifact_manifest"
    entries = manifest_payload["artifact_entries"]
    assert isinstance(entries, list)
    assert any(
        isinstance(item, dict)
        and item.get("artifact_name") == "artifact_load_diagnostics.json"
        and item.get("artifact_layer") == "governance"
        and item.get("required_in_strict_mode") is True
        for item in entries
    )
    assert any(
        isinstance(item, dict)
        and item.get("artifact_name") == "factor_definition.json"
        and item.get("artifact_layer") == "canonical"
        and item.get("scope") == "case"
        and item.get("required_in_strict_mode") is True
        for item in entries
    )
    workflow_context = comparison_payload.get("workflow_closure_artifacts")
    assert isinstance(workflow_context, dict)
    assert workflow_context["research_artifact_manifest_json_path"] == str(manifest_path)


def test_write_dashboard_html_emits_structured_diagnostics_artifact(
    tmp_path: Path,
) -> None:
    case_dir = tmp_path / "case_render_diagnostics"
    case_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        case_dir / "metrics.json",
        {
            "metrics": {"factor_name": "legacy_factor", "mean_rank_ic": 0.12},
            "coverage_by_date_summary": {"n_dates": 8, "mean_coverage": 0.75, "min_coverage": 0.6},
            "portfolio_validation_summary": {"recommendation": "legacy"},
            "portfolio_validation_metrics": {
                "protocol_settings": {"weighting_scheme": {"default": "rank"}}
            },
            "portfolio_validation_package": {"schema_version": "1.0.0"},
        },
    )
    _write_json(
        case_dir / "run_manifest.json",
        {"spec": {"factor_name": "legacy_from_manifest", "target": {"horizon": 5}}},
    )
    comparison_json = _write_minimal_comparison_json(
        tmp_path / "campaign_profile_comparison_render_diagnostics.json",
        case_output_dir=case_dir,
    )

    dashboard_path = write_campaign_profile_dashboard_html(comparison_json)
    assert dashboard_path.exists()
    comparison_payload = json.loads(Path(comparison_json).read_text(encoding="utf-8"))
    workflow_context = comparison_payload.get("workflow_closure_artifacts")
    assert isinstance(workflow_context, dict)
    diagnostics_pointer = workflow_context.get("artifact_load_diagnostics_json_path")
    assert isinstance(diagnostics_pointer, str)
    diagnostics_path = Path(diagnostics_pointer)
    assert diagnostics_path.exists()
    payload = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    assert payload["artifact_type"] == "alpha_lab_artifact_load_diagnostics"
    diagnostics = payload["diagnostics"]
    assert isinstance(diagnostics, list)
    assert diagnostics
    assert all(isinstance(item, dict) for item in diagnostics)
    manifest_pointer = workflow_context.get("research_artifact_manifest_json_path")
    assert isinstance(manifest_pointer, str)
    manifest_path = Path(manifest_pointer)
    assert manifest_path.exists()
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest_payload["artifact_type"] == "alpha_lab_research_artifact_manifest"
    entries = manifest_payload["artifact_entries"]
    assert isinstance(entries, list)
    assert any(
        isinstance(item, dict) and item.get("artifact_name") == "artifact_load_diagnostics.json"
        for item in entries
    )


def test_campaign_profile_dashboard_strict_mode_fails_when_required_artifacts_missing(
    tmp_path: Path,
) -> None:
    case_dir = tmp_path / "case_missing_artifacts_strict"
    case_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        case_dir / "metrics.json",
        {
            "metrics": {"factor_name": "legacy_factor", "mean_rank_ic": 0.12},
            "coverage_by_date_summary": {"n_dates": 8, "mean_coverage": 0.75, "min_coverage": 0.6},
        },
    )
    comparison_json = _write_minimal_comparison_json(
        tmp_path / "campaign_profile_comparison_strict.json",
        case_output_dir=case_dir,
    )

    with pytest.raises(
        dashboard_module.ArtifactLoadRuntimeError,
        match="strict artifact load checks failed",
    ) as exc_info:
        dashboard_module._build_research_dashboard_data(
            comparison_json,
            artifact_load_mode="strict",
        )
    diagnostics = exc_info.value.diagnostics
    assert any(
        item.code == "MISSING_CANONICAL_ARTIFACT"
        and item.severity == "error"
        and item.mode == "strict"
        for item in diagnostics
    )
    assert any(item.code == "STRICT_LOAD_ABORTED" for item in diagnostics)


def test_campaign_profile_dashboard_strict_mode_fails_on_invalid_workflow_payload(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "profile_compare_dashboard_strict_invalid"
    comparison_result = run_campaign_profile_comparison(
        source="example",
        output_root_dir=output_root,
        profiles=("exploratory_screening", "default_research"),
        render_report=False,
        clean_output=True,
    )
    workflow_paths = dashboard_module.persist_workflow_closure_artifacts(
        comparison_result.comparison_json_path
    )
    winner_path = workflow_paths["winner_selection_json_path"]
    winner_payload = json.loads(winner_path.read_text(encoding="utf-8"))
    winner_payload["artifact_type"] = "invalid_winner_type"
    winner_path.write_text(
        json.dumps(winner_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with pytest.raises(
        dashboard_module.ArtifactLoadRuntimeError,
        match="strict artifact load checks failed",
    ) as exc_info:
        dashboard_module._build_research_dashboard_data(
            comparison_result.comparison_json_path,
            artifact_load_mode="strict",
        )
    assert any(
        item.code == "INVALID_WORKFLOW_ARTIFACT"
        and item.object_scope == "winner_selection"
        and item.mode == "strict"
        for item in exc_info.value.diagnostics
    )


def test_campaign_profile_dashboard_strict_mode_succeeds_with_valid_artifacts(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "profile_compare_dashboard_strict_success"
    comparison_result = run_campaign_profile_comparison(
        source="example",
        output_root_dir=output_root,
        profiles=("exploratory_screening", "default_research"),
        render_report=False,
        clean_output=True,
    )

    data = dashboard_module._build_research_dashboard_data(
        comparison_result.comparison_json_path,
        artifact_load_mode="strict",
    )

    assert data.artifact_load_mode == "strict"
    assert data.artifact_load_warnings == ()
    assert not any(item.severity == "error" for item in data.artifact_load_diagnostics)
    assert data.factor_sets.factor_sets
    assert data.candidate_recipe_generation.generated_recipes
    assert data.winner_selection.winner_recipe_id


def test_campaign_profile_dashboard_recipe_comparison_builds_leaderboard_and_head_to_head() -> None:
    recipes = [
        dashboard_module.PortfolioRecipeSummary(
            recipe_id="recipe-a",
            recipe_name="Recipe A",
            selected_factors=("factor_a",),
            weighting_scheme="rank",
            neutralization_constraints="size-neutralization enabled",
            benchmark_mode="benchmark-relative",
            turnover_penalty_settings="warn if mean turnover > 0.45",
            transaction_cost_assumptions="one-way=0.001",
            position_limits="max|w|~0.30",
        ),
        dashboard_module.PortfolioRecipeSummary(
            recipe_id="recipe-b",
            recipe_name="Recipe B",
            selected_factors=("factor_b",),
            weighting_scheme="equal_weight",
            neutralization_constraints="no explicit neutralization constraint",
            benchmark_mode="absolute",
            turnover_penalty_settings="N/A",
            transaction_cost_assumptions="one-way=0.001",
            position_limits="max|w|~0.30",
        ),
    ]
    backtests = [
        dashboard_module.PortfolioBacktestSummary(
            recipe_id="recipe-a",
            factor_id="factor_a",
            sharpe=1.20,
            annualized_return=0.18,
            max_drawdown=-0.09,
            information_ratio=0.55,
            post_cost_return=0.15,
        ),
        dashboard_module.PortfolioBacktestSummary(
            recipe_id="recipe-b",
            factor_id="factor_b",
            sharpe=0.95,
            annualized_return=0.13,
            max_drawdown=-0.12,
            information_ratio=0.40,
            post_cost_return=0.10,
        ),
    ]
    factor_summaries = [
        dashboard_module.FactorSummary(
            factor_id="factor_a",
            factor_name="factor_a",
            short_description="A",
            factor_family="value",
            mathematical_definition="a",
        ),
        dashboard_module.FactorSummary(
            factor_id="factor_b",
            factor_name="factor_b",
            short_description="B",
            factor_family="momentum",
            mathematical_definition="b",
        ),
    ]

    view = dashboard_module._build_recipe_comparison_view(
        recipes=recipes,
        backtests=backtests,
        factor_summaries=factor_summaries,
    )

    sharpe_rows = [row for row in view.leaderboards if row.objective == "Sharpe"]
    assert sharpe_rows[0].recipe_id == "recipe-a"
    assert view.rows[0].recipe_id == "recipe-a"
    assert any(
        insight.winner_recipe_id == "recipe-a" and insight.objective == "Sharpe"
        for insight in view.head_to_head
    )


def test_campaign_profile_dashboard_lineage_registry_builds_provenance_links(
    tmp_path: Path,
) -> None:
    case_dir = tmp_path / "lineage_case"
    case_dir.mkdir(parents=True, exist_ok=True)
    _write_json(case_dir / "metrics.json", {"metrics": {}, "coverage_by_date_summary": {}})
    _write_json(
        case_dir / "run_manifest.json",
        {
            "run_timestamp_utc": "2026-01-20T08:00:00+00:00",
            "spec": {"factor_name": "lineage_factor"},
        },
    )
    _write_json(
        case_dir / "factor_definition.json",
        {
            "factor_name": "lineage_factor",
            "spec": {"factor_name": "lineage_factor"},
            "source_artifacts": {"run_manifest_path": str(case_dir / "run_manifest.json")},
        },
    )
    _write_json(
        case_dir / "signal_validation.json",
        {
            "metrics": {},
            "coverage_by_date_summary": {},
            "source_artifacts": {"metrics_path": str(case_dir / "metrics.json")},
        },
    )
    _write_json(
        case_dir / "portfolio_recipe.json",
        {
            "portfolio_validation_summary": {},
            "portfolio_validation_metrics": {},
            "portfolio_validation_package": {},
            "source_artifacts": {
                "portfolio_validation_summary_path": str(
                    case_dir / "portfolio_validation_summary.json"
                )
            },
        },
    )
    _write_json(
        case_dir / "backtest_result.json",
        {
            "summary": {},
            "source_artifacts": {"group_returns_path": str(case_dir / "group_returns.csv")},
        },
    )

    artifacts = dashboard_module._load_case_artifacts(
        {"artifact_paths": {"output_dir": str(case_dir)}}
    )
    entry = dashboard_module._build_registry_entry(
        case_name="lineage_case",
        profile_name="default_research",
        run_timestamp="2026-01-20T08:00:00+00:00",
        factor_id="lineage_case",
        recipe_id="recipe-lineage_case",
        artifacts=artifacts,
    )
    links = dashboard_module._build_lineage_links(
        profile_name="default_research",
        factor_id="lineage_case",
        recipe_id="recipe-lineage_case",
    )
    registry = dashboard_module._build_lineage_registry(
        entries=[entry],
        links=list(links),
    )

    assert entry.run_id.startswith("default_research:lineage_case:2026-01-20")
    assert entry.factor_definition_path.endswith("factor_definition.json")
    assert any("signal_validation.metrics_path" in row for row in entry.provenance_links)
    assert len(registry.entries) == 1
    assert any(link.relation == "validated_by" for link in registry.links)
