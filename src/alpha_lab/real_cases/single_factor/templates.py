from __future__ import annotations

import datetime
import math
from pathlib import Path

from alpha_lab.experiment import ExperimentResult

from .spec import SingleFactorCaseSpec


def render_summary_markdown(
    *,
    spec: SingleFactorCaseSpec,
    metrics: dict[str, object],
    output_dir: Path,
) -> str:
    """Render a compact human-readable run summary."""

    lines = [
        f"# 实验摘要：{spec.name}",
        "",
        "## 基本信息",
        "",
        f"- 因子：`{spec.factor_name}`",
        f"- 方向：`{spec.direction}`",
        f"- 股票池：`{spec.universe.name}`",
        f"- 调仓频率：`{spec.rebalance_frequency}`",
        f"- 目标：`{spec.target.kind}` / horizon=`{spec.target.horizon}`",
        "",
        "## 初筛结论",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Evaluation Profile | {_fmt(metrics.get('research_evaluation_profile'))} |",
        f"| Factor Verdict | {_fmt(metrics.get('factor_verdict'))} |",
        f"| Campaign Triage | {_fmt(metrics.get('campaign_triage'))} |",
        f"| Level 2 Promotion | {_fmt(metrics.get('promotion_decision'))} |",
        f"| Level 1->2 Transition | {_fmt(metrics.get('level12_transition_label'))} |",
        f"| Portfolio Validation | {_portfolio_validation_note(metrics)} |",
        f"| Mean Rank IC | {_fmt_dual_metric(metrics, 'mean_rank_ic')} |",
        f"| Mean MI | {_fmt_dual_metric(metrics, 'mean_mutual_information')} |",
        f"| ICIR | {_fmt_dual_metric(metrics, 'ic_ir')} |",
        f"| IC Half-Life | {_fmt_half_life(metrics)} |",
        f"| Decay vs Rebalance | {_fmt_decay_consistency(metrics)} |",
        f"| Mean Long-Short Return | {_fmt_dual_metric(metrics, 'mean_long_short_return')} |",
        f"| Mean Turnover | {_fmt_dual_metric(metrics, 'mean_long_short_turnover')} |",
        f"| Coverage Mean | {_fmt_dual_metric(metrics, 'eval_coverage_ratio_mean')} |",
        f"| Capacity | {_fmt_capacity_summary(metrics)} |",
        f"| Conditional IC | {_fmt_conditional_ic_summary(metrics)} |",
        (f"| 主要诊断 | {_fmt_reason_list(metrics.get('factor_verdict_reasons'))} |"),
        (f"| 主要阻断项 | {_fmt_reason_list(metrics.get('promotion_blockers'))} |"),
        (f"| 主要风险 | {_fmt_reason_list(metrics.get('portfolio_validation_major_risks'))} |"),
        "",
        "## 产物路径",
        "",
        f"- 输出目录：`{output_dir}`",
        "",
    ]
    return "\n".join(lines)


def render_experiment_card_markdown(
    *,
    spec: SingleFactorCaseSpec,
    metrics: dict[str, object],
    result: ExperimentResult,
) -> str:
    """Render a compact vault-friendly experiment card."""

    today = datetime.date.today().isoformat()
    tags = "[experiment, single_factor, quant]"
    lines = [
        "---",
        "type: experiment",
        f"name: {spec.name}",
        'source: "alpha-lab / real-case single-factor research package"',
        f"tags: {tags}",
        "status: draft",
        f"factor: {spec.factor_name}",
        f"direction: {spec.direction}",
        "emergent_moves: []",
        "operative_claims: []",
        f"horizon: {spec.target.horizon}",
        f"quantiles: {spec.n_quantiles}",
        f"rebalance_frequency: {spec.rebalance_frequency}",
        f"run_date: {today}",
        "---",
        "",
        f"# {spec.name}",
        "",
        "> 由 `alpha-lab` 自动生成的精简实验卡。",
        "",
        "## 基本信息",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Factor | `{spec.factor_name}` |",
        f"| Direction | `{spec.direction}` |",
        f"| Universe | `{spec.universe.name}` |",
        f"| Target | `{spec.target.kind}` |",
        f"| Horizon | {spec.target.horizon} |",
        f"| Rebalance frequency | `{spec.rebalance_frequency}` |",
        f"| Transaction cost (one-way) | {_fmt(spec.transaction_cost.one_way_rate)} |",
        f"| Eval dates (finite IC) | {_fmt(metrics.get('n_dates_used'))} |",
        "",
        "## 关键结果",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Mean Rank IC | {_fmt_dual_metric(metrics, 'mean_rank_ic')} |",
        f"| Mean MI | {_fmt_dual_metric(metrics, 'mean_mutual_information')} |",
        f"| ICIR | {_fmt_dual_metric(metrics, 'ic_ir')} |",
        f"| IC Half-Life | {_fmt_half_life(metrics)} |",
        f"| Decay vs Rebalance | {_fmt_decay_consistency(metrics)} |",
        f"| Factor Verdict | {_fmt(metrics.get('factor_verdict'))} |",
        f"| Campaign Triage | {_fmt(metrics.get('campaign_triage'))} |",
        f"| Level 2 Promotion | {_fmt(metrics.get('promotion_decision'))} |",
        (f"| Level 2 Portfolio Validation | {_portfolio_validation_note(metrics)} |"),
        f"| Mean Long-Short Return | {_fmt_dual_metric(metrics, 'mean_long_short_return')} |",
        f"| Mean Long-Short Turnover | {_fmt_dual_metric(metrics, 'mean_long_short_turnover')} |",
        f"| Coverage Mean | {_fmt_dual_metric(metrics, 'eval_coverage_ratio_mean')} |",
        f"| Capacity | {_fmt_capacity_summary(metrics)} |",
        f"| Conditional IC | {_fmt_conditional_ic_summary(metrics)} |",
        (f"| 主要诊断 | {_fmt_reason_list(metrics.get('factor_verdict_reasons'))} |"),
        "",
        "## 解释",
        "",
        "<!-- Manual: 只补充最关键的经济解释或失败原因 -->",
        "",
        "## 回灌素材",
        "",
        "- `emergent_moves`: <!-- Manual: 这次实验浮现、可被未来因子复用的新 move -->",
        "- `operative_claims`: <!-- Manual: 观察到的现象 / 经验 / 边界条件；弱 hint，不作为 kill 条件 -->",
        "",
        "## 下一步",
        "",
        "<!-- Manual: 只写下一步最重要的 1-3 个动作 -->",
        "",
        "## 备注",
        "",
        f"- Provenance run timestamp UTC: `{result.provenance.run_timestamp_utc}`",
        f"- Git commit: `{result.provenance.git_commit or 'unknown'}`",
        "",
    ]
    return "\n".join(lines)


def _fmt(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return "—"
        return f"{value:.6f}"
    return str(value)


def _fmt_dual_metric(metrics: dict[str, object], key: str) -> str:
    full_value = metrics.get(f"{key}_full")
    oos_value = metrics.get(f"{key}_oos")
    base_value = metrics.get(key)
    primary = _fmt(full_value if full_value is not None else base_value)
    oos = _fmt(oos_value)
    if oos_value is None or oos == "—":
        return primary
    return f"{primary} (OOS: {oos})"


def _fmt_flags(value: object) -> str:
    if value is None:
        return "none"
    if isinstance(value, (list, tuple)):
        tokens = [str(v).strip() for v in value if str(v).strip()]
        return ", ".join(tokens) if tokens else "none"
    text = str(value).strip()
    return text if text else "none"


def _fmt_reason_list(value: object) -> str:
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
    return text


def _portfolio_validation_note(metrics: dict[str, object]) -> str:
    status = _fmt(metrics.get("portfolio_validation_status"))
    recommendation = _fmt(metrics.get("portfolio_validation_recommendation"))
    if status == "—" and recommendation == "—":
        return "—"
    return f"{status} ({recommendation})"


def _fmt_half_life(metrics: dict[str, object]) -> str:
    status = str(metrics.get("ic_half_life_status") or "").strip().lower()
    half_life = _to_float(metrics.get("ic_half_life_horizon"))
    if status == "not_reached":
        return "not reached"
    if half_life is None:
        return "—"
    return f"{half_life:.2f}"


def _fmt_decay_consistency(metrics: dict[str, object]) -> str:
    ratio = _to_float(metrics.get("ic_decay_rebalance_ratio"))
    rebalance = _to_float(metrics.get("rebalance_step_dates"))
    if ratio is None or rebalance is None:
        status = str(metrics.get("ic_half_life_status") or "").strip().lower()
        if status == "not_reached":
            return f"rebalance={int(rebalance) if rebalance is not None else 'N/A'}; durable"
        return "—"
    return f"rebalance={int(rebalance)}; ratio={ratio:.2f}"


def _fmt_capacity_summary(metrics: dict[str, object]) -> str:
    status = str(metrics.get("capacity_status") or "").strip()
    capacity = _to_float(metrics.get("estimated_capacity_upper_bound"))
    adv = _to_float(metrics.get("mean_traded_adv"))
    if capacity is not None:
        return (
            f"{status or 'available'}; upper={capacity:.2f}; adv={adv:.2f}"
            if adv is not None
            else f"{status or 'available'}; upper={capacity:.2f}"
        )
    if status:
        note = str(metrics.get("capacity_notes") or "").strip()
        return f"{status}; {note}" if note else status
    return "—"


def _fmt_conditional_ic_summary(metrics: dict[str, object]) -> str:
    delta = _to_float(metrics.get("conditional_ic_extreme_minus_base_ic"))
    large = _to_float(metrics.get("conditional_ic_large_cross_section_mean_ic"))
    small = _to_float(metrics.get("conditional_ic_small_cross_section_mean_ic"))
    parts: list[str] = []
    if delta is not None:
        parts.append(f"Q5-Q1={delta:.4f}")
    if large is not None and small is not None:
        parts.append(f"large-small={large - small:.4f}")
    return "; ".join(parts) if parts else "—"


def _uncertainty_method_note(metrics: dict[str, object]) -> str:
    method = str(metrics.get("uncertainty_method") or "").strip().lower()
    if not method:
        return "—"
    level = _to_float(metrics.get("uncertainty_confidence_level"))
    resamples = metrics.get("uncertainty_bootstrap_resamples")
    block_length = metrics.get("uncertainty_bootstrap_block_length")
    if method == "bootstrap":
        resample_text = _fmt(resamples)
        if level is None:
            return f"bootstrap (resamples={resample_text})"
        return f"bootstrap (CI={level:.2f}, resamples={resample_text})"
    if method == "block_bootstrap":
        resample_text = _fmt(resamples)
        block_length_text = _fmt(block_length)
        if level is None:
            return f"block_bootstrap (resamples={resample_text}, block_length={block_length_text})"
        return (
            "block_bootstrap "
            f"(CI={level:.2f}, resamples={resample_text}, block_length={block_length_text})"
        )
    if level is None:
        return method
    return f"{method} (CI={level:.2f})"


def _fmt_ci(lower: object, upper: object) -> str:
    left = _to_float(lower)
    right = _to_float(upper)
    if left is None or right is None:
        return "—"
    return f"[{left:.6f}, {right:.6f}]"


def _to_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        out = float(value)
        if math.isfinite(out):
            return out
    return None


def _neutralization_comparison_rows(metrics: dict[str, object]) -> list[str]:
    comparison = _as_dict(metrics.get("neutralization_comparison"))
    if not comparison:
        return []
    raw = _as_dict(comparison.get("raw"))
    neutralized = _as_dict(comparison.get("neutralized"))
    delta = _as_dict(comparison.get("delta"))

    def _cmp(raw_key: str, delta_key: str) -> str:
        return _fmt_transition(
            raw.get(raw_key),
            neutralized.get(raw_key),
            delta.get(delta_key),
        )

    return [
        (f"| Raw vs Neutralized Mean IC | {_cmp('mean_ic', 'mean_ic_delta')} |"),
        (f"| Raw vs Neutralized Mean RankIC | {_cmp('mean_rank_ic', 'mean_rank_ic_delta')} |"),
        (
            "| Raw vs Neutralized Mean L/S Return | "
            f"{_cmp('mean_long_short_return', 'mean_long_short_return_delta')} |"
        ),
        (f"| Raw vs Neutralized ICIR | {_cmp('ic_ir', 'ic_ir_delta')} |"),
        (
            "| Raw vs Neutralized Validity Min | "
            f"{_cmp('valid_ratio_min', 'valid_ratio_min_delta')} |"
        ),
        (
            "| Raw vs Neutralized Coverage Mean | "
            f"{_cmp('eval_coverage_ratio_mean', 'eval_coverage_ratio_mean_delta')} |"
        ),
        (
            "| Raw vs Neutralized Uncertainty Overlap Count | "
            f"{_cmp('uncertainty_overlap_zero_count', 'uncertainty_overlap_zero_count_delta')} |"
        ),
        (
            "| Raw vs Neutralized Rolling+ Min Share | "
            f"{_cmp('rolling_positive_share_min', 'rolling_positive_share_min_delta')} |"
        ),
        (
            "| Raw vs Neutralized Rolling Worst Mean | "
            f"{_cmp('rolling_worst_mean_min', 'rolling_worst_mean_min_delta')} |"
        ),
        (
            "| Neutralization Comparison Flags | "
            f"{_fmt_flags(comparison.get('interpretation_flags'))} |"
        ),
        (
            "| Neutralization Comparison Reasons | "
            f"{_fmt_reason_list(comparison.get('interpretation_reasons'))} |"
        ),
    ]


def _fmt_transition(raw_value: object, neutralized_value: object, delta_value: object) -> str:
    return f"{_fmt(raw_value)} -> {_fmt(neutralized_value)} (delta={_fmt(delta_value)})"


def _as_dict(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return value
    return {}
