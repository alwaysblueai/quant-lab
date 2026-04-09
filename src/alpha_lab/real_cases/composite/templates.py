from __future__ import annotations

import datetime
import math
from pathlib import Path

from alpha_lab.experiment import ExperimentResult

from .spec import CompositeCaseSpec


def render_summary_markdown(
    *,
    spec: CompositeCaseSpec,
    metrics: dict[str, object],
    output_dir: Path,
) -> str:
    """Render a concise human-readable run summary."""

    ic_ci = _fmt_ci(metrics.get("mean_ic_ci_lower"), metrics.get("mean_ic_ci_upper"))
    rank_ic_ci = _fmt_ci(
        metrics.get("mean_rank_ic_ci_lower"),
        metrics.get("mean_rank_ic_ci_upper"),
    )
    ls_ci = _fmt_ci(
        metrics.get("mean_long_short_return_ci_lower"),
        metrics.get("mean_long_short_return_ci_upper"),
    )
    lines = [
        f"# Real-Case Composite Research Summary: {spec.name}",
        "",
        "## Run Context",
        "",
        f"- Case name: `{spec.name}`",
        f"- Universe: `{spec.universe.name}`",
        f"- Rebalance frequency: `{spec.rebalance_frequency}`",
        f"- Target: `{spec.target.kind}` (horizon={spec.target.horizon})",
        f"- Components: {', '.join(c.name for c in spec.components)}",
        f"- Output directory: `{output_dir}`",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Mean IC | {_fmt(metrics.get('mean_ic'))} |",
        f"| Mean IC 95% CI | {ic_ci} |",
        f"| Mean Rank IC | {_fmt(metrics.get('mean_rank_ic'))} |",
        f"| Mean Rank IC 95% CI | {rank_ic_ci} |",
        f"| ICIR | {_fmt(metrics.get('ic_ir'))} |",
        (
            "| Research Evaluation Profile | "
            f"{_fmt(metrics.get('research_evaluation_profile'))} |"
        ),
        f"| Factor Verdict | {_fmt(metrics.get('factor_verdict'))} |",
        (
            "| Verdict Reasons | "
            f"{_fmt_reason_list(metrics.get('factor_verdict_reasons'))} |"
        ),
        f"| Campaign Triage | {_fmt(metrics.get('campaign_triage'))} |",
        (
            "| Triage Reasons | "
            f"{_fmt_reason_list(metrics.get('campaign_triage_reasons'))} |"
        ),
        f"| Level 2 Promotion | {_fmt(metrics.get('promotion_decision'))} |",
        (
            "| Promotion Reasons | "
            f"{_fmt_reason_list(metrics.get('promotion_reasons'))} |"
        ),
        (
            "| Promotion Blockers | "
            f"{_fmt_reason_list(metrics.get('promotion_blockers'))} |"
        ),
        (
            "| Level 1->Level 2 Transition | "
            f"{_fmt(metrics.get('level12_transition_label'))} |"
        ),
        (
            "| Transition Interpretation | "
            f"{_fmt(metrics.get('level12_transition_interpretation'))} |"
        ),
        (
            "| Transition Reasons | "
            f"{_fmt_reason_list(metrics.get('level12_transition_reasons'))} |"
        ),
        (
            "| Confirmation vs Degradation | "
            f"{_fmt(metrics.get('level12_transition_confirmation_note'))} |"
        ),
        (
            "| Level 2 Portfolio Validation | "
            f"{_portfolio_validation_note(metrics)} |"
        ),
        (
            "| Portfolio Validation Risks | "
            f"{_fmt_reason_list(metrics.get('portfolio_validation_major_risks'))} |"
        ),
        (
            "| IC/RankIC Positive Rate | "
            f"{_fmt(metrics.get('ic_positive_rate'))} / "
            f"{_fmt(metrics.get('rank_ic_positive_rate'))} |"
        ),
        f"| Mean Long-Short Return | {_fmt(metrics.get('mean_long_short_return'))} |",
        f"| Mean Long-Short Return 95% CI | {ls_ci} |",
        f"| Long-Short IR | {_fmt(metrics.get('long_short_ir'))} |",
        (
            "| Long-Short Return per Turnover | "
            f"{_fmt(metrics.get('long_short_return_per_turnover'))} |"
        ),
        (
            "| Subperiod Positive Share (IC / L-S) | "
            f"{_fmt(metrics.get('subperiod_ic_positive_share'))} / "
            f"{_fmt(metrics.get('subperiod_long_short_positive_share'))} |"
        ),
        f"| Rolling Stability Window | {_fmt(metrics.get('rolling_window_size'))} |",
        (
            "| Rolling Positive Share (IC / RankIC / L-S) | "
            f"{_fmt(metrics.get('rolling_ic_positive_share'))} / "
            f"{_fmt(metrics.get('rolling_rank_ic_positive_share'))} / "
            f"{_fmt(metrics.get('rolling_long_short_positive_share'))} |"
        ),
        (
            "| Worst Rolling Mean (IC / RankIC / L-S) | "
            f"{_fmt(metrics.get('rolling_ic_min_mean'))} / "
            f"{_fmt(metrics.get('rolling_rank_ic_min_mean'))} / "
            f"{_fmt(metrics.get('rolling_long_short_min_mean'))} |"
        ),
        (
            "| Rolling Stability Flags | "
            f"{_fmt_flags(metrics.get('rolling_instability_flags'))} |"
        ),
        f"| Mean Long-Short Turnover | {_fmt(metrics.get('mean_long_short_turnover'))} |",
        (
            "| Coverage Ratio Mean/Min | "
            f"{_fmt(metrics.get('eval_coverage_ratio_mean'))} / "
            f"{_fmt(metrics.get('eval_coverage_ratio_min'))} |"
        ),
        f"| Coverage Mean | {_fmt(metrics.get('coverage_mean'))} |",
        f"| Missingness Mean | {_fmt(metrics.get('missingness_mean'))} |",
        (
            "| Neutralization Mean Corr Reduction | "
            f"{_fmt(metrics.get('neutralization_mean_corr_reduction'))} |"
        ),
        *_neutralization_comparison_rows(metrics),
        f"| Uncertainty Method | {_uncertainty_method_note(metrics)} |",
        f"| Uncertainty Flags | {_fmt_flags(metrics.get('uncertainty_flags'))} |",
        f"| Instability Flags | {_fmt_flags(metrics.get('instability_flags'))} |",
        (
            "| Cost-Adjusted Mean L/S Return | "
            f"{_fmt(metrics.get('mean_cost_adjusted_long_short_return'))} |"
        ),
        "",
        "## 备注",
        "",
        (
            "- This file is auto-generated. "
            "Use `experiment_card.md` for research interpretation notes."
        ),
        "",
    ]
    return "\n".join(lines)


def render_experiment_card_markdown(
    *,
    spec: CompositeCaseSpec,
    metrics: dict[str, object],
    result: ExperimentResult,
) -> str:
    """Render vault-friendly experiment card markdown."""

    today = datetime.date.today().isoformat()
    tags = "[experiment, composite, quant]"
    ic_ci = _fmt_ci(metrics.get("mean_ic_ci_lower"), metrics.get("mean_ic_ci_upper"))
    rank_ic_ci = _fmt_ci(
        metrics.get("mean_rank_ic_ci_lower"),
        metrics.get("mean_rank_ic_ci_upper"),
    )
    ls_ci = _fmt_ci(
        metrics.get("mean_long_short_return_ci_lower"),
        metrics.get("mean_long_short_return_ci_upper"),
    )
    lines = [
        "---",
        "type: experiment",
        f"name: {spec.name}",
        'source: "alpha-lab / real-case composite research package"',
        f"tags: {tags}",
        "status: draft",
        f"factor: {spec.name}",
        f"horizon: {spec.target.horizon}",
        f"quantiles: {spec.n_quantiles}",
        f"rebalance_frequency: {spec.rebalance_frequency}",
        f"run_date: {today}",
        "---",
        "",
        f"# {spec.name}",
        "",
        "> *由 `alpha-lab` 组合实战研究包自动生成。*  ",
        (
            "> *“基本信息”和“结果”为自动生成部分。“解释”、“下一步”、“开放问题”和“备注”"
            "为手工填写部分。*"
        ),
        "",
        "## 基本信息",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Universe | `{spec.universe.name}` |",
        f"| Target | `{spec.target.kind}` |",
        f"| Horizon | {spec.target.horizon} |",
        f"| Rebalance frequency | `{spec.rebalance_frequency}` |",
        f"| Components | {', '.join(c.name for c in spec.components)} |",
        f"| Transaction cost (one-way) | {_fmt(spec.transaction_cost.one_way_rate)} |",
        f"| Eval dates (finite IC) | {_fmt(metrics.get('n_dates_used'))} |",
        "",
        "## 结果",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Mean IC | {_fmt(metrics.get('mean_ic'))} |",
        f"| Mean IC 95% CI | {ic_ci} |",
        f"| Mean Rank IC | {_fmt(metrics.get('mean_rank_ic'))} |",
        f"| Mean Rank IC 95% CI | {rank_ic_ci} |",
        f"| ICIR | {_fmt(metrics.get('ic_ir'))} |",
        (
            "| Research Evaluation Profile | "
            f"{_fmt(metrics.get('research_evaluation_profile'))} |"
        ),
        f"| Factor Verdict | {_fmt(metrics.get('factor_verdict'))} |",
        (
            "| Verdict Reasons | "
            f"{_fmt_reason_list(metrics.get('factor_verdict_reasons'))} |"
        ),
        f"| Campaign Triage | {_fmt(metrics.get('campaign_triage'))} |",
        (
            "| Triage Reasons | "
            f"{_fmt_reason_list(metrics.get('campaign_triage_reasons'))} |"
        ),
        f"| Level 2 Promotion | {_fmt(metrics.get('promotion_decision'))} |",
        (
            "| Promotion Reasons | "
            f"{_fmt_reason_list(metrics.get('promotion_reasons'))} |"
        ),
        (
            "| Promotion Blockers | "
            f"{_fmt_reason_list(metrics.get('promotion_blockers'))} |"
        ),
        (
            "| Level 1->Level 2 Transition | "
            f"{_fmt(metrics.get('level12_transition_label'))} |"
        ),
        (
            "| Transition Interpretation | "
            f"{_fmt(metrics.get('level12_transition_interpretation'))} |"
        ),
        (
            "| Transition Reasons | "
            f"{_fmt_reason_list(metrics.get('level12_transition_reasons'))} |"
        ),
        (
            "| Confirmation vs Degradation | "
            f"{_fmt(metrics.get('level12_transition_confirmation_note'))} |"
        ),
        (
            "| Level 2 Portfolio Validation | "
            f"{_portfolio_validation_note(metrics)} |"
        ),
        (
            "| Portfolio Validation Risks | "
            f"{_fmt_reason_list(metrics.get('portfolio_validation_major_risks'))} |"
        ),
        (
            "| IC/RankIC Positive Rate | "
            f"{_fmt(metrics.get('ic_positive_rate'))} / "
            f"{_fmt(metrics.get('rank_ic_positive_rate'))} |"
        ),
        f"| Mean Long-Short Return | {_fmt(metrics.get('mean_long_short_return'))} |",
        f"| Mean Long-Short Return 95% CI | {ls_ci} |",
        f"| Long-Short IR | {_fmt(metrics.get('long_short_ir'))} |",
        f"| Long-Short Hit Rate | {_fmt(metrics.get('long_short_hit_rate'))} |",
        (
            "| Long-Short Return per Turnover | "
            f"{_fmt(metrics.get('long_short_return_per_turnover'))} |"
        ),
        (
            "| Subperiod Positive Share (IC / L-S) | "
            f"{_fmt(metrics.get('subperiod_ic_positive_share'))} / "
            f"{_fmt(metrics.get('subperiod_long_short_positive_share'))} |"
        ),
        f"| Rolling Stability Window | {_fmt(metrics.get('rolling_window_size'))} |",
        (
            "| Rolling Positive Share (IC / RankIC / L-S) | "
            f"{_fmt(metrics.get('rolling_ic_positive_share'))} / "
            f"{_fmt(metrics.get('rolling_rank_ic_positive_share'))} / "
            f"{_fmt(metrics.get('rolling_long_short_positive_share'))} |"
        ),
        (
            "| Worst Rolling Mean (IC / RankIC / L-S) | "
            f"{_fmt(metrics.get('rolling_ic_min_mean'))} / "
            f"{_fmt(metrics.get('rolling_rank_ic_min_mean'))} / "
            f"{_fmt(metrics.get('rolling_long_short_min_mean'))} |"
        ),
        (
            "| Rolling Stability Flags | "
            f"{_fmt_flags(metrics.get('rolling_instability_flags'))} |"
        ),
        f"| Mean Long-Short Turnover | {_fmt(metrics.get('mean_long_short_turnover'))} |",
        (
            "| Coverage Ratio Mean/Min | "
            f"{_fmt(metrics.get('eval_coverage_ratio_mean'))} / "
            f"{_fmt(metrics.get('eval_coverage_ratio_min'))} |"
        ),
        (
            "| Mean Cost-Adjusted L/S Return | "
            f"{_fmt(metrics.get('mean_cost_adjusted_long_short_return'))} |"
        ),
        f"| Coverage Mean | {_fmt(metrics.get('coverage_mean'))} |",
        (
            "| Neutralization Mean Corr Reduction | "
            f"{_fmt(metrics.get('neutralization_mean_corr_reduction'))} |"
        ),
        *_neutralization_comparison_rows(metrics),
        f"| Uncertainty Method | {_uncertainty_method_note(metrics)} |",
        f"| Uncertainty Flags | {_fmt_flags(metrics.get('uncertainty_flags'))} |",
        f"| Instability Flags | {_fmt_flags(metrics.get('instability_flags'))} |",
        "",
        "## 解释",
        "",
        "<!-- Manual: interpret statistical and economic significance -->",
        "",
        "## 下一步",
        "",
        "<!-- Manual: define the next concrete research actions -->",
        "",
        "## 开放问题",
        "",
        "<!-- Manual: unresolved assumptions / data questions -->",
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
            return (
                "block_bootstrap "
                f"(resamples={resample_text}, block_length={block_length_text})"
            )
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
        (
            "| Raw vs Neutralized Mean IC | "
            f"{_cmp('mean_ic', 'mean_ic_delta')} |"
        ),
        (
            "| Raw vs Neutralized Mean RankIC | "
            f"{_cmp('mean_rank_ic', 'mean_rank_ic_delta')} |"
        ),
        (
            "| Raw vs Neutralized Mean L/S Return | "
            f"{_cmp('mean_long_short_return', 'mean_long_short_return_delta')} |"
        ),
        (
            "| Raw vs Neutralized ICIR | "
            f"{_cmp('ic_ir', 'ic_ir_delta')} |"
        ),
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
    return (
        f"{_fmt(raw_value)} -> {_fmt(neutralized_value)} "
        f"(delta={_fmt(delta_value)})"
    )


def _as_dict(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return value
    return {}
