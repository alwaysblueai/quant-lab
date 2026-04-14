from __future__ import annotations

import datetime
import math
from pathlib import Path

from alpha_lab.experiment import ExperimentResult

from .spec import ModelFactorCaseSpec


def render_summary_markdown(
    *,
    spec: ModelFactorCaseSpec,
    metrics: dict[str, object],
    model_diagnostics: dict[str, object],
    output_dir: Path,
) -> str:
    """Render a concise Chinese-first run summary."""

    top_features = _fmt_reason_list(model_diagnostics.get("top_features"))
    lines = [
        f"# 实战模型因子研究摘要: {spec.name}",
        "",
        "## 运行上下文",
        "",
        f"- 案例名: `{spec.name}`",
        f"- 输出因子: `{spec.factor_name}`",
        f"- 模型族: `{spec.model.family}`",
        f"- 特征数: {len(spec.feature_columns)}",
        f"- 调仓频率: `{spec.rebalance_frequency}`",
        f"- 标签: `{spec.target.kind}` (horizon={spec.target.horizon})",
        f"- 输出目录: `{output_dir}`",
        "",
        "## 核心结果",
        "",
        "| 指标 | 数值 |",
        "|---|---|",
        f"| Mean IC | {_fmt(metrics.get('mean_ic'))} |",
        f"| Mean Rank IC | {_fmt(metrics.get('mean_rank_ic'))} |",
        f"| ICIR | {_fmt(metrics.get('ic_ir'))} |",
        f"| Mean Long-Short Return | {_fmt(metrics.get('mean_long_short_return'))} |",
        f"| Long-Short IR | {_fmt(metrics.get('long_short_ir'))} |",
        f"| 因子结论 | {_fmt(metrics.get('factor_verdict'))} |",
        f"| Campaign Triage | {_fmt(metrics.get('campaign_triage'))} |",
        f"| Level 2 Promotion | {_fmt(metrics.get('promotion_decision'))} |",
        f"| 组合层验证 | {_portfolio_validation_note(metrics)} |",
        f"| 训练模型版本数 | {_fmt(model_diagnostics.get('trained_model_versions'))} |",
        f"| 平均训练样本数 | {_fmt(model_diagnostics.get('mean_train_rows'))} |",
        f"| 平均打分资产数 | {_fmt(model_diagnostics.get('mean_score_assets'))} |",
        f"| Top Features | {top_features} |",
        "",
        "## 备注",
        "",
        "- 本文件自动生成，用于快速审阅模型端是否成功收敛为可评估的标准因子。",
        "- 更详细的训练过程请查看 `training_log.csv` 与 `feature_importance.csv`。",
        "",
    ]
    return "\n".join(lines)


def render_experiment_card_markdown(
    *,
    spec: ModelFactorCaseSpec,
    metrics: dict[str, object],
    model_diagnostics: dict[str, object],
    result: ExperimentResult,
) -> str:
    """Render a vault-friendly experiment card for the model-factor route."""

    today = datetime.date.today().isoformat()
    eval_dates = int(result.n_eval_dates)
    top_features = _fmt_reason_list(model_diagnostics.get("top_features"))
    lines = [
        "---",
        "type: experiment",
        f"name: {spec.name}",
        'source: "alpha-lab / real-case model-factor research package"',
        "tags: [experiment, model_factor, quant]",
        "status: draft",
        f"factor: {spec.factor_name}",
        f"model_family: {spec.model.family}",
        f"horizon: {spec.target.horizon}",
        f"quantiles: {spec.n_quantiles}",
        f"rebalance_frequency: {spec.rebalance_frequency}",
        f"run_date: {today}",
        "---",
        "",
        f"# {spec.name}",
        "",
        "> *由 `alpha-lab` 模型因子实战研究包自动生成。*  ",
        "> *模型端负责把研究特征转成分数，评估端仍按标准因子链路执行。*",
        "",
        "## 基本信息",
        "",
        "| 字段 | 数值 |",
        "|---|---|",
        f"| Factor | `{spec.factor_name}` |",
        f"| Model Family | `{spec.model.family}` |",
        f"| Direction | `{spec.direction}` |",
        f"| Feature Count | {len(spec.feature_columns)} |",
        f"| Eval Dates | {eval_dates} |",
        f"| Top Features | {top_features} |",
        "",
        "## 结果",
        "",
        "| 指标 | 数值 |",
        "|---|---|",
        f"| Mean IC | {_fmt(metrics.get('mean_ic'))} |",
        f"| Mean Rank IC | {_fmt(metrics.get('mean_rank_ic'))} |",
        f"| ICIR | {_fmt(metrics.get('ic_ir'))} |",
        f"| Mean Long-Short Return | {_fmt(metrics.get('mean_long_short_return'))} |",
        f"| Long-Short IR | {_fmt(metrics.get('long_short_ir'))} |",
        f"| Factor Verdict | {_fmt(metrics.get('factor_verdict'))} |",
        f"| Campaign Triage | {_fmt(metrics.get('campaign_triage'))} |",
        f"| Level 2 Promotion | {_fmt(metrics.get('promotion_decision'))} |",
        f"| 组合层验证 | {_portfolio_validation_note(metrics)} |",
        "",
        "## 解释",
        "",
        "- 待补充：记录为什么这些特征和该模型族有机会生成稳定横截面排序。",
        "",
        "## 下一步",
        "",
        "- 待补充：判断是否需要更严格 profile、替换特征集、或改进训练窗口。",
        "",
    ]
    return "\n".join(lines)


def _portfolio_validation_note(metrics: dict[str, object]) -> str:
    status = _fmt(metrics.get("portfolio_validation_status"))
    recommendation = _fmt(metrics.get("portfolio_validation_recommendation"))
    return f"{status} ({recommendation})"


def _fmt(value: object) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return "N/A"
        return f"{value:.6f}"
    text = str(value).strip()
    return text if text else "N/A"


def _fmt_reason_list(value: object) -> str:
    if isinstance(value, list):
        tokens = [str(item).strip() for item in value if str(item).strip()]
        return ", ".join(tokens) if tokens else "N/A"
    return _fmt(value)
