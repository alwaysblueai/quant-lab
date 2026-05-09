"""Category Registry — dispatch layer for multi-category research.

Each ``CategoryProfile`` describes how the research bridge templates, preflight
checks, and frontend forms adapt to a particular research category.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from alpha_lab.config import PROJECT_ROOT

_DEFAULT_FACTOR_RECIPE_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "real_case_inputs"
    / "ashare_institutional_20160418_20260415_supplemented"
)
_LEGACY_FACTOR_RECIPE_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "real_case_inputs"
    / "tushare_qfq_listed90_20200401_20260401"
)


@dataclass(frozen=True, slots=True)
class CategoryProfile:
    """Template/config bundle for one research category."""

    key: str
    display_name_zh: str
    valid_case_types: tuple[str, ...]
    # Numbered items in round_prompt "请按以下结构回答" section
    round_prompt_sections: tuple[str, ...]
    # (label, placeholder) pairs for discussion_capture template
    discussion_fields: tuple[tuple[str, str], ...]
    # Lines for knowledge_handoff domain-specific sections (after generic header)
    handoff_domain_sections: tuple[str, ...]
    # Node types to filter in structure_candidates semantic matching
    relevant_node_types: tuple[str, ...]
    # Preflight check codes that apply; None means all
    preflight_checks: frozenset[str] | None
    # Explore prompt "你的任务" lines (constrained / free)
    explore_task_constrained: tuple[str, ...]
    explore_task_free: tuple[str, ...]
    # Frontend form field definitions for scaffold-case (JSON-serializable)
    form_fields: tuple[dict[str, Any], ...]


# ---------------------------------------------------------------------------
# Built-in profiles
# ---------------------------------------------------------------------------

FACTOR_RECIPE = CategoryProfile(
    key="factor_recipe",
    display_name_zh="因子配方研究",
    valid_case_types=("factor_recipe",),
    round_prompt_sections=(
        "1. 已知事实",
        "2. 推断",
        "3. 待验证问题",
        "4. 推荐的因子/信号定义",
        "5. 与历史失败案例的区别",
        "6. 进入 alpha-lab 前还缺什么",
    ),
    discussion_fields=(
        ("factor_definition", "如：20 日成交量加权动量"),
        ("universe", "如：listed_90d"),
        ("target_horizon", "如：5"),
        ("evaluation_profile", "如：exploratory_screening"),
    ),
    handoff_domain_sections=(
        "## 因子或信号定义",
        "",
        "- `raw_definition`: {case_label}",
        "- `formula_or_pseudocode`: 待根据本轮讨论和 case spec 草案细化。",
        "- `lookbacks_and_parameters`: 待确认",
        "- `expected_sign`: 待确认",
        "",
        "## 预处理与中性化方案",
        "",
        "- `winsorize`: recipe 层默认启用 1%/99%",
        "- `standardize_or_rank`: recipe 层默认 z-score",
        "- `neutralize`: 待按本轮讨论决定",
        "- `coverage_rule`: recipe 层默认最小覆盖 0.2",
        "- `missing_value_rule`: 保持默认 no-lookahead 填充策略，不做未来填补",
        "",
        "## 目标与评估",
        "",
        "- `target_kind`: forward_return",
        "- `target_horizon`: 待按 case 参数确认",
        "- `primary_metrics`: mean_rank_ic, long_short_return, turnover, factor_verdict",
        "- `robustness_checks`: 子区间稳定性、rolling 稳定性、覆盖率",
        "- `failure_conditions`: 统计显著性不足、单一阶段驱动、覆盖率过低或高换手",
    ),
    relevant_node_types=("factor",),
    preflight_checks=None,  # all checks
    explore_task_constrained=(
        "基于以上背景，请帮我：",
        "1. 将模糊想法结构化为一个**可测试的因子假设**（明确信号来源、频率、中性化方式）",
        "2. 提出 2-3 个具体的变体设计（不同 lookback、数据源、标准化方式）",
        "3. 参考约束报告，说明你的因子与现有同族因子的**核心差异点**",
        "4. 指出主要的 PIT 风险点和可行性障碍",
    ),
    explore_task_free=(
        "基于以上背景，请帮我：",
        "1. 从模糊想法出发自由联想，提出 5-8 个因子变体方向（不限机制、不限数据）",
        "2. 对每个方向简要说明：预期有效的经济逻辑、适合的市场环境",
        "3. 选出你认为**最值得验证的 2 个方向**，给出初步的实验设计建议",
    ),
    form_fields=(
        {
            "name": "factor_name",
            "label": "factor_name（因子名称）",
            "type": "text",
            "placeholder": "mom_amt_60",
        },
        {
            "name": "base_method",
            "label": "base_method（基础方法）",
            "type": "text",
            "default": "momentum",
        },
        {
            "name": "direction",
            "label": "direction（long / short）",
            "type": "select",
            "options": ("long", "short"),
            "default": "long",
        },
        {
            "name": "lookback",
            "label": "lookback（回看窗口）",
            "type": "number",
            "default": "20",
        },
        {
            "name": "skip_recent",
            "label": "skip_recent（信号滞后）",
            "type": "number",
            "default": "5",
        },
        {
            "name": "target_horizon",
            "label": "target_horizon（持有期）",
            "type": "number",
            "default": "5",
        },
        {
            "name": "rebalance_frequency",
            "label": "rebalance_frequency（调仓频率）",
            "type": "select",
            "options": ("D", "W", "M"),
            "default": "W",
        },
        {
            "name": "shock_gate_mode",
            "label": "shock_gate_mode（冲击事件门控）",
            "type": "select",
            "options": ("", "cs_quantile", "ts_threshold", "none"),
            "default": "",
        },
        {
            "name": "shock_q",
            "label": "shock_q（截面分位阈值）",
            "type": "number",
            "placeholder": "0.70",
        },
        {
            "name": "shock_threshold",
            "label": "shock_threshold（时序阈值）",
            "type": "number",
            "placeholder": "1.0",
        },
        {
            "name": "outside_event_policy",
            "label": "outside_event_policy（事件外处理）",
            "type": "select",
            "options": ("", "nan", "zero"),
            "default": "",
        },
        {
            "name": "neutralize_basic",
            "label": "neutralize_basic（基础残差化）",
            "type": "select",
            "options": ("", "true", "false"),
            "default": "",
        },
        {
            "name": "invert",
            "label": "invert（输出取反）",
            "type": "select",
            "options": ("", "true", "false"),
            "default": "",
        },
        {
            "name": "exclude_limit",
            "label": "exclude_limit（过滤涨跌停）",
            "type": "select",
            "options": ("", "true", "false"),
            "default": "",
        },
        {
            "name": "exclude_st",
            "label": "exclude_st（过滤 ST）",
            "type": "select",
            "options": ("", "true", "false"),
            "default": "",
        },
        {
            "name": "exclude_suspended",
            "label": "exclude_suspended（过滤停牌）",
            "type": "select",
            "options": ("", "true", "false"),
            "default": "",
        },
        {
            "name": "builder_kwargs_json",
            "label": "builder_kwargs_json（可选，JSON）",
            "type": "textarea",
            "placeholder": (
                "{\"shock_gate_mode\":\"cs_quantile\",\"shock_q\":0.7,"
                "\"neutralize_basic\":true}"
            ),
        },
        {
            "name": "prices_path",
            "label": "prices_path（价格数据路径）",
            "type": "text",
            "default": str(_DEFAULT_FACTOR_RECIPE_DATA_DIR / "prices.parquet"),
        },
        {
            "name": "universe_path",
            "label": "universe_path（选股域路径）",
            "type": "text",
            "default": str(_DEFAULT_FACTOR_RECIPE_DATA_DIR / "universe_mask.parquet"),
        },
    ),
)

FACTOR_EVALUATION = CategoryProfile(
    key="factor_evaluation",
    display_name_zh="因子评价方法论",
    valid_case_types=("evaluation_study",),
    round_prompt_sections=(
        "1. 当前已有的评价方法及其局限性",
        "2. 文献中提出的改进方向",
        "3. 推荐的评价框架改进方案",
        "4. 可量化的验证方案（含基准 / 对照实验设计）",
        "5. 与已有实践的关键差异",
        "6. 需要的数据和计算资源",
    ),
    discussion_fields=(
        ("evaluation_framework", "如：基于 rolling IC 的衰减半衰期估计"),
        ("metrics_to_compare", "如：IC_mean, IC_IR, quantile_spread, factor_verdict"),
        ("benchmark_studies", "如：Harvey et al. 2016, McLean & Pontiff 2016"),
        ("validation_approach", "如：模拟 + 历史回测对照"),
    ),
    handoff_domain_sections=(
        "## 评价框架定义",
        "",
        "- `framework_name`: {case_label}",
        "- `target_problem`: 待根据本轮讨论补充（如：过拟合检测、因子衰减度量、多重检验校正）",
        "- `approach`: 待确认",
        "",
        "## 度量指标",
        "",
        "- `metrics_under_study`: 待列举需要比较的指标",
        "- `baseline_metrics`: 当前系统已有的评价指标",
        "- `proposed_new_metrics`: 待确认",
        "",
        "## 验证方案",
        "",
        "- `validation_method`: 模拟数据 / 已知因子对照 / 历史回测",
        "- `sample_factors`: 待选定用于验证的代表性因子集",
        "- `success_criteria`: 待确认",
        "- `failure_conditions`: 待确认",
    ),
    relevant_node_types=("method", "concept", "factor"),
    preflight_checks=frozenset({"dependency_completeness", "novelty"}),
    explore_task_constrained=(
        "基于以上背景，请帮我：",
        "1. 梳理当前因子评价体系的**具体缺陷**（哪些场景下现有指标会给出误导性结论）",
        "2. 提出 2-3 个改进方向，每个包含：改进的指标定义、适用场景、与现有指标的关系",
        "3. 设计一个可执行的**验证实验**（用什么因子、什么数据、怎么对比新旧方法）",
        "4. 参考约束报告，指出相关的已有方法和概念",
    ),
    explore_task_free=(
        "基于以上背景，请帮我：",
        "1. 自由联想因子评价的各个维度（统计显著性、经济逻辑、稳健性、可交易性、衰减性……）",
        "2. 对每个维度提出一个非常规的评价角度或方法",
        "3. 选出 2 个最有潜力的方向，给出初步的研究设计",
    ),
    form_fields=(
        {
            "name": "framework_name",
            "label": "评价框架名称",
            "type": "text",
            "placeholder": "如：rolling_ic_decay_estimator",
        },
        {
            "name": "metrics_to_compare",
            "label": "待比较指标（逗号分隔）",
            "type": "text",
            "placeholder": "IC_mean, IC_IR, quantile_spread",
        },
        {
            "name": "validation_approach",
            "label": "验证方法",
            "type": "select",
            "options": ["simulation", "historical_backtest", "cross_validation", "comparison"],
            "default": "historical_backtest",
        },
    ),
)

PORTFOLIO_CONSTRUCTION = CategoryProfile(
    key="portfolio_construction",
    display_name_zh="组合构建研究",
    valid_case_types=("portfolio_study",),
    round_prompt_sections=(
        "1. 当前已有的组合构建方法",
        "2. 约束条件与现实限制（换手、容量、风险预算）",
        "3. 推荐的组合构建方案",
        "4. 风险预算与再平衡策略",
        "5. 与已有实践的关键差异",
        "6. 需要的输入数据和模型假设",
    ),
    discussion_fields=(
        ("portfolio_strategy", "如：risk_parity, mean_variance, hierarchical_risk_parity"),
        ("constraint_set", "如：max_weight=5%, turnover_limit=30%"),
        ("risk_budget_method", "如：equal_risk_contribution"),
        ("rebalance_method", "如：monthly_calendar, threshold_trigger"),
    ),
    handoff_domain_sections=(
        "## 组合策略定义",
        "",
        "- `strategy_name`: {case_label}",
        "- `optimization_objective`: 待确认（如：最小方差、最大夏普、风险预算）",
        "- `constraints`: 待列举约束条件",
        "",
        "## 输入与假设",
        "",
        "- `factor_inputs`: 待确认（哪些因子信号作为输入）",
        "- `risk_model`: 待确认（样本协方差 / Barra / 收缩估计）",
        "- `transaction_cost_model`: 待确认",
        "",
        "## 评估方案",
        "",
        "- `primary_metrics`: sharpe, max_drawdown, turnover, information_ratio",
        "- `robustness_checks`: 参数敏感性、不同市场环境、样本外表现",
        "- `failure_conditions`: 夏普显著低于基准、换手不可控、集中度过高",
    ),
    relevant_node_types=("method", "concept", "factor"),
    preflight_checks=frozenset({"dependency_completeness", "novelty"}),
    explore_task_constrained=(
        "基于以上背景，请帮我：",
        "1. 分析当前组合构建方法的优劣势",
        "2. 提出 2-3 个改进方向（优化目标、约束处理、再平衡规则）",
        "3. 设计可执行的**对比实验**（基准策略 vs 改进策略，明确评估指标）",
        "4. 指出关键的模型假设和数据要求",
    ),
    explore_task_free=(
        "基于以上背景，请帮我：",
        "1. 自由联想组合构建的各个维度（目标函数、风险模型、约束、再平衡、执行）",
        "2. 提出 5-8 个研究方向（不限于传统均值方差框架）",
        "3. 选出 2 个最值得深入的方向，给出初步设计",
    ),
    form_fields=(
        {
            "name": "strategy_name",
            "label": "组合策略名称",
            "type": "text",
            "placeholder": "如：risk_parity_v2",
        },
        {
            "name": "portfolio_strategy",
            "label": "策略类型",
            "type": "select",
            "options": [
                "risk_parity",
                "mean_variance",
                "hierarchical_risk_parity",
                "equal_weight",
                "min_variance",
                "max_diversification",
                "other",
            ],
            "default": "risk_parity",
        },
        {
            "name": "constraint_set",
            "label": "约束条件",
            "type": "text",
            "placeholder": "如：max_weight=5%, turnover_limit=30%",
        },
    ),
)

RESEARCH_METHODOLOGY = CategoryProfile(
    key="research_methodology",
    display_name_zh="研究方法论",
    valid_case_types=("methodology_study",),
    round_prompt_sections=(
        "1. 当前已有的研究方法及其局限性",
        "2. 文献中的改进建议",
        "3. 推荐的方法论改进方案",
        "4. 验证方案（如何证明新方法更好）",
        "5. 对现有研究流程的影响",
        "6. 实施路径和依赖条件",
    ),
    discussion_fields=(
        ("methodology_name", "如：purged_walk_forward_cv"),
        ("target_problem", "如：回测过拟合检测、多重检验校正"),
        ("comparison_baselines", "如：standard_cv, time_series_split"),
        ("validation_approach", "如：蒙特卡洛模拟 + 已知因子对照"),
    ),
    handoff_domain_sections=(
        "## 方法论定义",
        "",
        "- `methodology_name`: {case_label}",
        "- `target_problem`: 待确认（如：回测过拟合、样本外退化、参数不确定性）",
        "- `approach`: 待确认",
        "",
        "## 基准与对照",
        "",
        "- `baseline_methods`: 待列举当前使用的方法",
        "- `proposed_improvement`: 待确认",
        "- `theoretical_justification`: 待确认",
        "",
        "## 验证方案",
        "",
        "- `validation_method`: 蒙特卡洛模拟 / 已知因子对照 / 对比实验",
        "- `success_criteria`: 待确认",
        "- `failure_conditions`: 待确认",
        "- `computational_requirements`: 待确认",
    ),
    relevant_node_types=("method", "concept", "paper"),
    preflight_checks=frozenset({"dependency_completeness"}),
    explore_task_constrained=(
        "基于以上背景，请帮我：",
        "1. 分析当前研究方法的**具体缺陷**和适用边界",
        "2. 提出 2-3 个改进方向，每个包含：方法定义、理论依据、与当前方法的对比",
        "3. 设计验证实验（用什么数据、怎么证明新方法更优）",
        "4. 参考约束报告，指出相关的已有方法和论文",
    ),
    explore_task_free=(
        "基于以上背景，请帮我：",
        "1. 自由联想量化研究方法论的各个维度（回测设计、统计检验、模型选择、特征工程……）",
        "2. 对每个维度提出一个非常规的改进想法",
        "3. 选出 2 个最有潜力的方向，给出初步研究设计",
    ),
    form_fields=(
        {
            "name": "methodology_name",
            "label": "方法论名称",
            "type": "text",
            "placeholder": "如：purged_walk_forward_cv",
        },
        {
            "name": "target_problem",
            "label": "目标问题",
            "type": "text",
            "placeholder": "如：回测过拟合检测",
        },
        {
            "name": "validation_approach",
            "label": "验证方法",
            "type": "select",
            "options": [
                "monte_carlo",
                "known_factor_benchmark",
                "comparison_experiment",
                "theoretical_proof",
            ],
            "default": "comparison_experiment",
        },
    ),
)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

CATEGORY_REGISTRY: dict[str, CategoryProfile] = {
    p.key: p
    for p in (FACTOR_RECIPE, FACTOR_EVALUATION, PORTFOLIO_CONSTRUCTION, RESEARCH_METHODOLOGY)
}

# Backward-compat aliases: existing projects may use "factor_family" as category
_ALIASES: dict[str, str] = {
    "factor_family": "factor_recipe",
}


def _resolve_factor_recipe_data_defaults() -> dict[str, str]:
    env_vars = (
        "ALPHA_LAB_FACTOR_RECIPE_DATA_DIR",
        "ALPHA_LAB_REAL_CASE_INPUT_DIR",
    )
    env_roots: list[Path] = []
    for env_var in env_vars:
        raw_roots = os.getenv(env_var, "").strip()
        if not raw_roots:
            continue
        for raw_root in raw_roots.split(os.pathsep):
            normalized = raw_root.strip()
            if normalized:
                env_roots.append(Path(normalized).expanduser().resolve())

    dataset_bases: list[Path] = []
    for root in env_roots:
        dataset_bases.append(root)
        dataset_bases.append(root / "alpha_lab_single_factor_inputs")
        dataset_bases.append(root / "ashare_institutional_20160418_20260415")
        dataset_bases.append(root / "tushare_qfq_listed90_20200401_20260401")
        dataset_bases.append(
            root
            / "data"
            / "processed"
            / "real_case_inputs"
            / "ashare_institutional_20160418_20260415"
        )
        dataset_bases.append(
            root
            / "data"
            / "processed"
            / "real_case_inputs"
            / "tushare_qfq_listed90_20200401_20260401"
        )

    dataset_bases.append(_DEFAULT_FACTOR_RECIPE_DATA_DIR)
    for root in (PROJECT_ROOT, PROJECT_ROOT / "data", Path(tempfile.gettempdir())):
        dataset_bases.append(root)
        dataset_bases.append(root / "alpha_lab_single_factor_inputs")
        dataset_bases.append(
            root
            / "data"
            / "processed"
            / "real_case_inputs"
            / "ashare_institutional_20160418_20260415"
        )
        dataset_bases.append(
            root
            / "data"
            / "processed"
            / "real_case_inputs"
            / "tushare_qfq_listed90_20200401_20260401"
        )
    dataset_bases.append(_LEGACY_FACTOR_RECIPE_DATA_DIR)

    seen_bases: dict[Path, None] = {}
    for base in dataset_bases:
        seen_bases[base] = None

    for base in seen_bases:
        prices_parquet = base / "alpha_lab_single_factor_inputs" / "prices.parquet"
        universe_parquet = base / "alpha_lab_single_factor_inputs" / "universe_mask.parquet"
        if prices_parquet.exists() and universe_parquet.exists():
            return {
                "prices_path": str(prices_parquet),
                "universe_path": str(universe_parquet),
            }

        prices_csv = base / "alpha_lab_single_factor_inputs" / "prices.csv"
        universe_csv = base / "alpha_lab_single_factor_inputs" / "universe_mask.csv"
        if prices_csv.exists() and universe_csv.exists():
            return {
                "prices_path": str(prices_csv),
                "universe_path": str(universe_csv),
            }

        prices_direct_parquet = base / "prices.parquet"
        for universe_direct_parquet in (
            base / "universe_mask.parquet",
            base / "universe.parquet",
        ):
            if prices_direct_parquet.exists() and universe_direct_parquet.exists():
                return {
                    "prices_path": str(prices_direct_parquet),
                    "universe_path": str(universe_direct_parquet),
                }

        prices_legacy = base / "prices.csv"
        for universe_legacy in (base / "universe_mask.csv", base / "universe.csv"):
            if prices_legacy.exists() and universe_legacy.exists():
                return {
                    "prices_path": str(prices_legacy),
                    "universe_path": str(universe_legacy),
                }

    fallback = _DEFAULT_FACTOR_RECIPE_DATA_DIR
    return {
        "prices_path": str(fallback / "prices.parquet"),
        "universe_path": str(fallback / "universe_mask.parquet"),
    }


def _apply_runtime_defaults(profile: CategoryProfile) -> CategoryProfile:
    if profile.key != "factor_recipe":
        return profile
    defaults = _resolve_factor_recipe_data_defaults()
    fields: list[dict[str, Any]] = []
    for field in profile.form_fields:
        updated = dict(field)
        name = str(updated.get("name") or "")
        if name in defaults:
            updated["default"] = defaults[name]
        fields.append(updated)
    return replace(profile, form_fields=tuple(fields))


def default_form_field_value(category: str, field_name: str) -> str:
    profile = get_category_profile(category)
    for field in profile.form_fields:
        if str(field.get("name") or "") == field_name:
            return str(field.get("default") or "")
    return ""


def get_category_profile(category: str) -> CategoryProfile:
    """Look up a category profile, falling back to *factor_recipe*."""
    key = _ALIASES.get(category, category)
    return _apply_runtime_defaults(CATEGORY_REGISTRY.get(key, FACTOR_RECIPE))


def list_categories() -> list[dict[str, object]]:
    """Return registry metadata suitable for the frontend ``/api/categories``."""
    return [
        {
            "key": p.key,
            "display_name_zh": p.display_name_zh,
            "valid_case_types": list(p.valid_case_types),
            "form_fields": list(p.form_fields),
        }
        for p in (_apply_runtime_defaults(profile) for profile in CATEGORY_REGISTRY.values())
    ]
