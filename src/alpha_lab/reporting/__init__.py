from __future__ import annotations

import datetime
import math
from pathlib import Path

import pandas as pd

from alpha_lab.costs import cost_adjusted_long_short
from alpha_lab.execution_impact_report import ExecutionImpactReport
from alpha_lab.experiment import ExperimentResult
from alpha_lab.obsidian import write_obsidian_note
from alpha_lab.reporting.factor_verdict import (
    build_factor_verdict,
    reasons_to_text,
)
from alpha_lab.reporting.uncertainty import compute_core_uncertainty
from alpha_lab.research_evaluation_config import (
    DEFAULT_RESEARCH_EVALUATION_CONFIG,
    ResearchEvaluationConfig,
)

# Ordered column names for summary DataFrames produced by this module.
SUMMARY_COLUMNS: tuple[str, ...] = (
    "factor_name",
    "label_name",
    "n_quantiles",
    "split_description",
    "mean_ic",
    "mean_ic_ci_lower",
    "mean_ic_ci_upper",
    "mean_rank_ic",
    "mean_rank_ic_ci_lower",
    "mean_rank_ic_ci_upper",
    "ic_ir",
    "ic_positive_rate",
    "rank_ic_positive_rate",
    "ic_valid_ratio",
    "rank_ic_valid_ratio",
    "mean_long_short_return",
    "mean_long_short_return_ci_lower",
    "mean_long_short_return_ci_upper",
    "long_short_ir",
    "long_short_hit_rate",
    "long_short_return_per_turnover",
    "subperiod_ic_positive_share",
    "subperiod_long_short_positive_share",
    "subperiod_ic_min_mean",
    "subperiod_long_short_min_mean",
    "rolling_window_size",
    "rolling_ic_positive_share",
    "rolling_rank_ic_positive_share",
    "rolling_long_short_positive_share",
    "rolling_ic_min_mean",
    "rolling_rank_ic_min_mean",
    "rolling_long_short_min_mean",
    "n_dates_used",
    "mean_long_short_turnover",
    "mean_eval_assets_per_date",
    "min_eval_assets_per_date",
    "eval_coverage_ratio_mean",
    "eval_coverage_ratio_min",
    "uncertainty_flags",
    "uncertainty_method",
    "uncertainty_confidence_level",
    "uncertainty_bootstrap_resamples",
    "uncertainty_bootstrap_block_length",
    "rolling_instability_flags",
    "instability_flags",
    "factor_verdict",
    "factor_verdict_reasons",
    "cost_rate",
    "mean_cost_adjusted_long_short_return",
)


def summarise_experiment_result(
    result: ExperimentResult,
    *,
    cost_rate: float | None = None,
    evaluation_config: ResearchEvaluationConfig = DEFAULT_RESEARCH_EVALUATION_CONFIG,
) -> pd.DataFrame:
    """Produce a compact one-row summary DataFrame from an ExperimentResult.

    All metadata (factor name, label name, quantile count, split boundaries,
    turnover) is read directly from ``result`` — nothing is inferred from
    heuristics or supplied by the caller.  Metric values are taken from
    ``result.summary`` without re-computation.

    The returned DataFrame has exactly one row and columns in
    :data:`SUMMARY_COLUMNS`.  Multiple rows can be stacked with
    ``pd.concat([row1, row2], ignore_index=True)``.

    Parameters
    ----------
    result:
        Output of :func:`~alpha_lab.experiment.run_factor_experiment`.
    cost_rate:
        Optional one-way transaction cost rate (e.g., 0.001 for 10 bps).
        When provided, ``mean_cost_adjusted_long_short_return`` is computed
        via :func:`~alpha_lab.costs.cost_adjusted_long_short`.  When
        ``None``, that column is NaN.  This is a minimal research friction
        estimate only; see :mod:`alpha_lab.costs` for the full disclaimer.

    Returns
    -------
    pd.DataFrame
        One-row DataFrame with columns in :data:`SUMMARY_COLUMNS`.
    """
    factor_name = (
        str(result.factor_df["factor"].iloc[0])
        if not result.factor_df.empty
        else "unknown"
    )
    label_name = (
        str(result.label_df["factor"].iloc[0])
        if not result.label_df.empty
        else "unknown"
    )

    # Cost-adjusted return — only computed when cost_rate is provided.
    mean_cost_adj: float
    if cost_rate is not None:
        adj_df = cost_adjusted_long_short(
            result.long_short_df,
            result.long_short_turnover_df,
            cost_rate=cost_rate,
        )
        adj_vals = adj_df["adjusted_return"].dropna()
        mean_cost_adj = float(adj_vals.mean()) if len(adj_vals) > 0 else float("nan")
    else:
        mean_cost_adj = float("nan")

    s = result.summary
    uncertainty = compute_core_uncertainty(
        ic_values=result.ic_df["ic"] if "ic" in result.ic_df.columns else [],
        rank_ic_values=(
            result.rank_ic_df["rank_ic"] if "rank_ic" in result.rank_ic_df.columns else []
        ),
        long_short_values=(
            result.long_short_df["long_short_return"]
            if "long_short_return" in result.long_short_df.columns
            else []
        ),
        thresholds=evaluation_config.uncertainty,
    )
    row: dict[str, object] = {
        "factor_name": factor_name,
        "label_name": label_name,
        "n_quantiles": result.n_quantiles,
        "split_description": _split_description(result.train_end, result.test_start),
        "mean_ic": s.mean_ic,
        "mean_ic_ci_lower": uncertainty.mean_ic_ci_lower,
        "mean_ic_ci_upper": uncertainty.mean_ic_ci_upper,
        "mean_rank_ic": s.mean_rank_ic,
        "mean_rank_ic_ci_lower": uncertainty.mean_rank_ic_ci_lower,
        "mean_rank_ic_ci_upper": uncertainty.mean_rank_ic_ci_upper,
        "ic_ir": s.ic_ir,
        "ic_positive_rate": s.ic_positive_rate,
        "rank_ic_positive_rate": s.rank_ic_positive_rate,
        "ic_valid_ratio": s.ic_valid_ratio,
        "rank_ic_valid_ratio": s.rank_ic_valid_ratio,
        "mean_long_short_return": s.mean_long_short_return,
        "mean_long_short_return_ci_lower": uncertainty.mean_long_short_return_ci_lower,
        "mean_long_short_return_ci_upper": uncertainty.mean_long_short_return_ci_upper,
        "long_short_ir": s.long_short_ir,
        "long_short_hit_rate": s.long_short_hit_rate,
        "long_short_return_per_turnover": s.long_short_return_per_turnover,
        "subperiod_ic_positive_share": s.subperiod_ic_positive_share,
        "subperiod_long_short_positive_share": s.subperiod_long_short_positive_share,
        "subperiod_ic_min_mean": s.subperiod_ic_min_mean,
        "subperiod_long_short_min_mean": s.subperiod_long_short_min_mean,
        "rolling_window_size": s.rolling_window_size,
        "rolling_ic_positive_share": s.rolling_ic_positive_share,
        "rolling_rank_ic_positive_share": s.rolling_rank_ic_positive_share,
        "rolling_long_short_positive_share": s.rolling_long_short_positive_share,
        "rolling_ic_min_mean": s.rolling_ic_min_mean,
        "rolling_rank_ic_min_mean": s.rolling_rank_ic_min_mean,
        "rolling_long_short_min_mean": s.rolling_long_short_min_mean,
        "n_dates_used": s.n_dates,
        "mean_long_short_turnover": s.mean_long_short_turnover,
        "mean_eval_assets_per_date": s.mean_eval_assets_per_date,
        "min_eval_assets_per_date": s.min_eval_assets_per_date,
        "eval_coverage_ratio_mean": s.eval_coverage_ratio_mean,
        "eval_coverage_ratio_min": s.eval_coverage_ratio_min,
        "uncertainty_flags": ";".join(uncertainty.uncertainty_flags),
        "uncertainty_method": uncertainty.uncertainty_method,
        "uncertainty_confidence_level": uncertainty.uncertainty_confidence_level,
        "uncertainty_bootstrap_resamples": (
            uncertainty.uncertainty_bootstrap_resamples
            if uncertainty.uncertainty_bootstrap_resamples is not None
            else float("nan")
        ),
        "uncertainty_bootstrap_block_length": (
            uncertainty.uncertainty_bootstrap_block_length
            if uncertainty.uncertainty_bootstrap_block_length is not None
            else float("nan")
        ),
        "rolling_instability_flags": ";".join(s.rolling_instability_flags),
        "instability_flags": ";".join(s.instability_flags),
    }
    verdict = build_factor_verdict(row, thresholds=evaluation_config.factor_verdict)
    row["factor_verdict"] = verdict.label
    row["factor_verdict_reasons"] = reasons_to_text(verdict.reasons)
    row["cost_rate"] = cost_rate if cost_rate is not None else float("nan")
    row["mean_cost_adjusted_long_short_return"] = mean_cost_adj
    return pd.DataFrame([row], columns=list(SUMMARY_COLUMNS))


def export_summary_csv(
    summary: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Write a summary DataFrame to CSV, creating parent directories as needed.

    Parameters
    ----------
    summary:
        Non-empty DataFrame whose columns must include all of
        :data:`SUMMARY_COLUMNS`.  Typically the output of
        :func:`summarise_experiment_result` or a vertical stack of such rows.
    output_path:
        Destination file path.  Parent directories are created if they do
        not exist.

    Raises
    ------
    TypeError
        If ``summary`` is not a :class:`pandas.DataFrame`.
    ValueError
        If ``summary`` is empty or is missing expected columns.
    """
    if not isinstance(summary, pd.DataFrame):
        raise TypeError(
            f"summary must be a pandas DataFrame, got {type(summary).__name__}"
        )
    if summary.empty:
        raise ValueError("summary DataFrame is empty; nothing to export")
    missing = set(SUMMARY_COLUMNS) - set(summary.columns)
    if missing:
        raise ValueError(
            f"summary is missing expected columns: {sorted(missing)}"
        )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(path, index=False)


def to_obsidian_markdown(
    result: ExperimentResult,
    *,
    title: str | None = None,
    cost_rate: float | None = None,
    notes: str | None = None,
    horizon: int | None = None,
    tags: list[str] | None = None,
) -> str:
    """Render an experiment result as Obsidian-friendly markdown.

    All metadata is sourced from ``result`` directly.  No scientific
    conclusions are generated; interpretation and next-step sections are
    explicit placeholders for the researcher.

    Parameters
    ----------
    result:
        Output of :func:`~alpha_lab.experiment.run_factor_experiment`.
    title:
        H1 heading for the note.  Defaults to ``"Experiment: {factor_name}"``.
    cost_rate:
        Optional one-way cost rate.  When provided, a cost-adjusted return
        row is added to the metrics table.  This is a minimal research
        friction estimate only.
    notes:
        Optional free-text appended under a ``## 备注`` section.
    horizon:
        Forward-return horizon in per-asset rows.  When provided, included in
        the YAML frontmatter.
    tags:
        YAML frontmatter tags.  Defaults to ``["quant", "factor"]``.

    Returns
    -------
    str
        Markdown text with YAML frontmatter, ending with a single newline.
    """
    summary_df = summarise_experiment_result(result, cost_rate=cost_rate)
    row = summary_df.iloc[0]

    factor_name = str(row["factor_name"])
    label_name = str(row["label_name"])
    split_desc = str(row["split_description"])
    n_dates = int(row["n_dates_used"])  # type: ignore[arg-type]
    nq_str = str(int(row["n_quantiles"]))  # type: ignore[arg-type]
    n_quantiles = int(row["n_quantiles"])  # type: ignore[arg-type]

    if title is None:
        title = f"Experiment: {factor_name}"

    resolved_tags = tags if tags is not None else ["quant", "factor"]
    tag_lines = "\n".join(f"  - {t}" for t in resolved_tags)
    frontmatter_lines = [
        "---",
        f"factor: {factor_name}",
        f"label: {label_name}",
        f"quantiles: {n_quantiles}",
    ]
    if horizon is not None:
        frontmatter_lines.append(f"horizon: {horizon}")
    frontmatter_lines += [
        f"date: {datetime.date.today().isoformat()}",
        f"tags:\n{tag_lines}",
        "---",
    ]

    s = result.summary
    verdict = str(row["factor_verdict"]).strip()
    verdict_reasons = _fmt_reason_text(row["factor_verdict_reasons"])
    uncertainty_flags = _fmt_reason_text(row["uncertainty_flags"])
    ic_ci = _fmt_ci(float(row["mean_ic_ci_lower"]), float(row["mean_ic_ci_upper"]))
    rank_ic_ci = _fmt_ci(
        float(row["mean_rank_ic_ci_lower"]),
        float(row["mean_rank_ic_ci_upper"]),
    )
    ls_ci = _fmt_ci(
        float(row["mean_long_short_return_ci_lower"]),
        float(row["mean_long_short_return_ci_upper"]),
    )
    metrics_rows = [
        f"| Mean IC | {_fmt_float(s.mean_ic)} |",
        f"| Mean IC 95% CI | {ic_ci} |",
        f"| Mean Rank IC | {_fmt_float(s.mean_rank_ic)} |",
        f"| Mean Rank IC 95% CI | {rank_ic_ci} |",
        f"| IC IR | {_fmt_float(s.ic_ir)} |",
        f"| IC Positive Rate | {_fmt_float(s.ic_positive_rate)} |",
        f"| RankIC Positive Rate | {_fmt_float(s.rank_ic_positive_rate)} |",
        f"| IC Valid Ratio | {_fmt_float(s.ic_valid_ratio)} |",
        f"| RankIC Valid Ratio | {_fmt_float(s.rank_ic_valid_ratio)} |",
        f"| Mean L/S Return | {_fmt_float(s.mean_long_short_return)} |",
        f"| Mean L/S Return 95% CI | {ls_ci} |",
        f"| L/S IR | {_fmt_float(s.long_short_ir)} |",
        f"| L/S Hit Rate | {_fmt_float(s.long_short_hit_rate)} |",
        (
            "| L/S Return per Turnover | "
            f"{_fmt_float(s.long_short_return_per_turnover)} |"
        ),
        (
            "| Subperiod Positive Share (IC / L/S) | "
            f"{_fmt_float(s.subperiod_ic_positive_share)} / "
            f"{_fmt_float(s.subperiod_long_short_positive_share)} |"
        ),
        (
            "| Subperiod Worst Mean (IC / L/S) | "
            f"{_fmt_float(s.subperiod_ic_min_mean)} / "
            f"{_fmt_float(s.subperiod_long_short_min_mean)} |"
        ),
        (
            "| Rolling Stability Window | "
            f"{s.rolling_window_size} observations |"
        ),
        (
            "| Rolling Positive Share (IC / RankIC / L/S) | "
            f"{_fmt_float(s.rolling_ic_positive_share)} / "
            f"{_fmt_float(s.rolling_rank_ic_positive_share)} / "
            f"{_fmt_float(s.rolling_long_short_positive_share)} |"
        ),
        (
            "| Worst Rolling Mean (IC / RankIC / L/S) | "
            f"{_fmt_float(s.rolling_ic_min_mean)} / "
            f"{_fmt_float(s.rolling_rank_ic_min_mean)} / "
            f"{_fmt_float(s.rolling_long_short_min_mean)} |"
        ),
        (
            "| Rolling Stability Flags | "
            f"{_fmt_reason_text(row['rolling_instability_flags'])} |"
        ),
        f"| Mean L/S Turnover (one-way) | {_fmt_float(s.mean_long_short_turnover)} |",
        (
            "| Coverage Ratio Mean/Min | "
            f"{_fmt_float(s.eval_coverage_ratio_mean)} / "
            f"{_fmt_float(s.eval_coverage_ratio_min)} |"
        ),
        f"| Mean Eval Assets per Date | {_fmt_float(s.mean_eval_assets_per_date)} |",
        f"| Min Eval Assets per Date | {_fmt_float(s.min_eval_assets_per_date)} |",
        f"| Uncertainty Flags | {uncertainty_flags} |",
        f"| Instability Flags | {_fmt_flags(s.instability_flags)} |",
        f"| Factor Verdict | {verdict} |",
        f"| Verdict Reasons | {verdict_reasons} |",
    ]
    if cost_rate is not None:
        adj = float(row["mean_cost_adjusted_long_short_return"])  # type: ignore[arg-type]
        metrics_rows.append(
            f"| Mean L/S Return (cost-adj, rate={cost_rate}) | {_fmt_float(adj)} |"
        )

    lines = [
        *frontmatter_lines,
        "",
        f"# {title}",
        "",
        "## 实验信息",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Factor | `{factor_name}` |",
        f"| Label | `{label_name}` |",
        f"| N quantiles | {nq_str} |",
        f"| Split | {split_desc} |",
        f"| Eval dates (finite IC) | {n_dates} |",
        "",
        "## 摘要指标",
        "",
        "| Metric | Value |",
        "|---|---|",
        *metrics_rows,
        "",
        "## 解释",
        "",
        "<!-- Add interpretation here -->",
        "",
        "## 下一步",
        "",
        "<!-- Add next steps here -->",
    ]

    if notes is not None:
        lines += [
            "",
        "## 备注",
            "",
            notes,
        ]

    return "\n".join(lines) + "\n"


def export_experiment_card(
    result: ExperimentResult,
    name: str,
    *,
    vault_path: str | Path | None = None,
    cost_rate: float | None = None,
    notes: str | None = None,
    tags: list[str] | None = None,
    overwrite: bool = False,
) -> Path:
    """Export an ExperimentResult as a quant-knowledge experiment card.

    The generated card should be Chinese-first in its human-readable sections.
    Preserve English only where it is necessary for technical terms, proper
    nouns, abbreviations, formulas, code symbols, file paths, or quoted source
    titles.

    Writes to ``{vault_path}/50_experiments/Exp - YYYYMM - {name}.md``
    using the quant-knowledge frontmatter schema (``type: experiment``).
    The generated card is a research log: it records setup, metrics, and
    placeholder sections for interpretation and next steps.

    Parameters
    ----------
    result:
        Output of :func:`~alpha_lab.experiment.run_factor_experiment`.
    name:
        Short description for the filename (e.g.
        ``"momentum-5d-Ashare"``).  Combined with today's YYYYMM to form
        the filename ``Exp - YYYYMM - {name}.md``.  Must not contain path
        separators.
    vault_path:
        Path to the quant-knowledge vault root.  Defaults to
        Resolved via :func:`~alpha_lab.vault_export.resolve_vault_root`:
        CLI value takes priority, then the ``OBSIDIAN_VAULT_PATH`` environment
        variable, then ``None``.  Raises :exc:`ValueError` when unresolved.
    cost_rate:
        Optional one-way transaction cost rate.  When provided, a
        cost-adjusted return row is appended to the metrics table.
    notes:
        Optional free-text appended under a ``## 备注`` section.
    tags:
        YAML frontmatter tags.  Defaults to
        ``[experiment, factor, quant]``.
    overwrite:
        If ``False`` (default), raises :exc:`FileExistsError` when the
        destination file already exists.

    Returns
    -------
    Path
        Resolved path of the written card.

    Raises
    ------
    ValueError
        If ``name`` is empty/whitespace or contains path separators, or
        if no vault path can be resolved.
    FileNotFoundError
        If the resolved vault path does not exist on disk.
    NotADirectoryError
        If the resolved vault path exists but is not a directory.
    FileExistsError
        If the destination file already exists and ``overwrite`` is
        ``False``.
    """
    name = name.strip()
    if not name:
        raise ValueError("name must not be empty or whitespace")
    if "/" in name or "\\" in name:
        raise ValueError(
            f"name must not contain path separators; got {name!r}"
        )

    # Resolve vault path.
    from alpha_lab.vault_export import resolve_vault_root

    _resolved = resolve_vault_root(vault_path)
    if _resolved is None:
        raise ValueError(
            "vault_path was not provided and OBSIDIAN_VAULT_PATH is not "
            "set.  Pass vault_path explicitly or set the "
            "OBSIDIAN_VAULT_PATH environment variable."
        )
    resolved_vault: Path = _resolved

    # Vault root must already exist.  We create the 50_experiments subdir on
    # demand, but we do not create the vault root — a missing root most likely
    # indicates a misconfigured path rather than a first-time setup.
    if not resolved_vault.exists():
        raise FileNotFoundError(
            f"Vault path {resolved_vault!s} does not exist.  "
            "Ensure vault_path (or OBSIDIAN_VAULT_PATH) points to an "
            "existing quant-knowledge vault directory."
        )
    if not resolved_vault.is_dir():
        raise NotADirectoryError(
            f"Vault path {resolved_vault!s} exists but is not a directory."
        )

    yyyymm = datetime.date.today().strftime("%Y%m")
    card_name = f"Exp - {yyyymm} - {name}"
    output_path = resolved_vault / "50_experiments" / f"{card_name}.md"

    markdown = _render_experiment_card(
        result,
        card_name=card_name,
        cost_rate=cost_rate,
        notes=notes,
        tags=tags,
    )
    return write_obsidian_note(
        markdown,
        output_path,
        overwrite=overwrite,
        restricted_root=resolved_vault / "50_experiments",
    )


def export_execution_impact_experiment_card(
    report: ExecutionImpactReport,
    name: str,
    *,
    vault_path: str | Path | None = None,
    case_name: str | None = None,
    workflow_kind: str | None = None,
    git_commit: str | None = None,
    notes: str | None = None,
    tags: list[str] | None = None,
    related_cards: list[str] | None = None,
    overwrite: bool = False,
) -> Path:
    """Export an ExecutionImpactReport as a quant-knowledge experiment card.

    This path is intended for explicit Level 3 replay / execution-realism
    diagnostics. It does not replace the default Level 1/2
    :func:`export_experiment_card` path for standard ``ExperimentResult``.
    """
    name = name.strip()
    if not name:
        raise ValueError("name must not be empty or whitespace")
    if "/" in name or "\\" in name:
        raise ValueError(f"name must not contain path separators; got {name!r}")

    from alpha_lab.vault_export import resolve_vault_root

    _resolved = resolve_vault_root(vault_path)
    if _resolved is None:
        raise ValueError(
            "vault_path was not provided and OBSIDIAN_VAULT_PATH is not "
            "set. Pass vault_path explicitly or set OBSIDIAN_VAULT_PATH."
        )
    resolved_vault: Path = _resolved

    if not resolved_vault.exists():
        raise FileNotFoundError(
            f"Vault path {resolved_vault!s} does not exist. Ensure vault_path "
            "(or OBSIDIAN_VAULT_PATH) points to an existing quant-knowledge vault."
        )
    if not resolved_vault.is_dir():
        raise NotADirectoryError(
            f"Vault path {resolved_vault!s} exists but is not a directory."
        )

    yyyymm = datetime.date.today().strftime("%Y%m")
    card_name = f"Exp - {yyyymm} - {name}"
    output_path = resolved_vault / "50_experiments" / f"{card_name}.md"
    markdown = _render_execution_impact_experiment_card(
        report,
        card_name=card_name,
        case_name=case_name,
        workflow_kind=workflow_kind,
        git_commit=git_commit,
        notes=notes,
        tags=tags,
        related_cards=related_cards,
    )
    return write_obsidian_note(
        markdown,
        output_path,
        overwrite=overwrite,
        restricted_root=resolved_vault / "50_experiments",
    )


def _render_experiment_card(
    result: ExperimentResult,
    *,
    card_name: str,
    cost_rate: float | None,
    notes: str | None,
    tags: list[str] | None,
) -> str:
    """Render an ExperimentResult as a quant-knowledge experiment card."""
    summary_df = summarise_experiment_result(result, cost_rate=cost_rate)
    row = summary_df.iloc[0]

    factor_name = str(row["factor_name"])
    split_desc = str(row["split_description"])
    n_dates = int(row["n_dates_used"])  # type: ignore[arg-type]
    prov = result.provenance

    resolved_tags = tags if tags is not None else ["experiment", "factor", "quant"]
    tag_str = "[" + ", ".join(resolved_tags) + "]"

    # quant-knowledge §4 frontmatter schema
    frontmatter_lines = [
        "---",
        "type: experiment",
        f"name: {card_name}",
        'source: "alpha-lab / Self-developed"',
        f"tags: {tag_str}",
        "status: draft",
        f"factor: {factor_name}",
        f"horizon: {prov.horizon}",
        f"quantiles: {prov.n_quantiles}",
        f"split: {split_desc}",
        f'git_commit: {prov.git_commit or "unknown"}',
        f"run_timestamp_utc: {prov.run_timestamp_utc}",
        "---",
    ]

    s = result.summary
    verdict = str(row["factor_verdict"]).strip()
    verdict_reasons = _fmt_reason_text(row["factor_verdict_reasons"])
    uncertainty_flags = _fmt_reason_text(row["uncertainty_flags"])
    ic_ci = _fmt_ci(float(row["mean_ic_ci_lower"]), float(row["mean_ic_ci_upper"]))
    rank_ic_ci = _fmt_ci(
        float(row["mean_rank_ic_ci_lower"]),
        float(row["mean_rank_ic_ci_upper"]),
    )
    ls_ci = _fmt_ci(
        float(row["mean_long_short_return_ci_lower"]),
        float(row["mean_long_short_return_ci_upper"]),
    )
    metrics_rows = [
        f"| Mean IC | {_fmt_float(s.mean_ic)} |",
        f"| Mean IC 95% CI | {ic_ci} |",
        f"| Mean Rank IC | {_fmt_float(s.mean_rank_ic)} |",
        f"| Mean Rank IC 95% CI | {rank_ic_ci} |",
        f"| IC IR | {_fmt_float(s.ic_ir)} |",
        f"| IC Positive Rate | {_fmt_float(s.ic_positive_rate)} |",
        f"| RankIC Positive Rate | {_fmt_float(s.rank_ic_positive_rate)} |",
        f"| IC Valid Ratio | {_fmt_float(s.ic_valid_ratio)} |",
        f"| RankIC Valid Ratio | {_fmt_float(s.rank_ic_valid_ratio)} |",
        f"| Mean L/S Return | {_fmt_float(s.mean_long_short_return)} |",
        f"| Mean L/S Return 95% CI | {ls_ci} |",
        f"| L/S IR | {_fmt_float(s.long_short_ir)} |",
        f"| L/S Hit Rate | {_fmt_float(s.long_short_hit_rate)} |",
        (
            "| L/S Return per Turnover | "
            f"{_fmt_float(s.long_short_return_per_turnover)} |"
        ),
        (
            "| Subperiod Positive Share (IC / L/S) | "
            f"{_fmt_float(s.subperiod_ic_positive_share)} / "
            f"{_fmt_float(s.subperiod_long_short_positive_share)} |"
        ),
        (
            "| Subperiod Worst Mean (IC / L/S) | "
            f"{_fmt_float(s.subperiod_ic_min_mean)} / "
            f"{_fmt_float(s.subperiod_long_short_min_mean)} |"
        ),
        (
            "| Rolling Stability Window | "
            f"{s.rolling_window_size} observations |"
        ),
        (
            "| Rolling Positive Share (IC / RankIC / L/S) | "
            f"{_fmt_float(s.rolling_ic_positive_share)} / "
            f"{_fmt_float(s.rolling_rank_ic_positive_share)} / "
            f"{_fmt_float(s.rolling_long_short_positive_share)} |"
        ),
        (
            "| Worst Rolling Mean (IC / RankIC / L/S) | "
            f"{_fmt_float(s.rolling_ic_min_mean)} / "
            f"{_fmt_float(s.rolling_rank_ic_min_mean)} / "
            f"{_fmt_float(s.rolling_long_short_min_mean)} |"
        ),
        (
            "| Rolling Stability Flags | "
            f"{_fmt_reason_text(row['rolling_instability_flags'])} |"
        ),
        f"| Mean L/S Turnover (one-way) | {_fmt_float(s.mean_long_short_turnover)} |",
        (
            "| Coverage Ratio Mean/Min | "
            f"{_fmt_float(s.eval_coverage_ratio_mean)} / "
            f"{_fmt_float(s.eval_coverage_ratio_min)} |"
        ),
        f"| Mean Eval Assets per Date | {_fmt_float(s.mean_eval_assets_per_date)} |",
        f"| Min Eval Assets per Date | {_fmt_float(s.min_eval_assets_per_date)} |",
        f"| Uncertainty Flags | {uncertainty_flags} |",
        f"| Instability Flags | {_fmt_flags(s.instability_flags)} |",
        f"| Factor Verdict | {verdict} |",
        f"| Verdict Reasons | {verdict_reasons} |",
    ]
    if cost_rate is not None:
        adj = float(row["mean_cost_adjusted_long_short_return"])  # type: ignore[arg-type]
        metrics_rows.append(
            f"| Mean L/S Return (cost-adj, rate={cost_rate}) | {_fmt_float(adj)} |"
        )

    lines = [
        *frontmatter_lines,
        "",
        f"# {card_name}",
        "",
        "> *Auto-generated by `alpha-lab`. Setup, Results, and frontmatter "
        "reflect the recorded experiment and must not be edited manually.*  ",
        "> *Interpretation, Next Steps, Open Questions, and Notes are for "
        "manual completion.*",
        "",
        "## 基本信息",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Factor | `{factor_name}` |",
        f"| Horizon | {prov.horizon} bars |",
        f"| N quantiles | {prov.n_quantiles} |",
        f"| Split | {split_desc} |",
        f"| Eval dates (finite IC) | {n_dates} |",
        f"| Assets in eval period | {result.n_eval_assets} |",
        f'| Git commit | `{prov.git_commit or "unknown"}` |',
        "",
        "## 结果",
        "",
        "| Metric | Value |",
        "|---|---|",
        *metrics_rows,
        "",
        "## 解释",
        "",
        "<!-- Add interpretation here -->",
        "",
        "## 下一步",
        "",
        "<!-- Add next steps here -->",
        "",
        "## 相关卡片",
        "",
        f"- [[30_factors/Factor - {factor_name}]]",
        "",
        "## 开放问题",
        "",
        "<!-- What needs further investigation? -->",
    ]

    if notes is not None:
        lines += [
            "",
        "## 备注",
            "",
            notes,
        ]

    return "\n".join(lines) + "\n"


def _render_execution_impact_experiment_card(
    report: ExecutionImpactReport,
    *,
    card_name: str,
    case_name: str | None,
    workflow_kind: str | None,
    git_commit: str | None,
    notes: str | None,
    tags: list[str] | None,
    related_cards: list[str] | None,
) -> str:
    """Render an ExecutionImpactReport as a quant-knowledge experiment card."""
    resolved_tags = tags if tags is not None else [
        "experiment",
        "execution-impact",
        "quant",
        "level3",
    ]
    tag_str = "[" + ", ".join(resolved_tags) + "]"
    resolved_case_name = (case_name or "unknown").strip() or "unknown"
    resolved_workflow = (workflow_kind or "execution_impact").strip() or "execution_impact"
    resolved_git_commit = (git_commit or "unknown").strip() or "unknown"

    frontmatter_lines = [
        "---",
        "type: experiment",
        f"name: {card_name}",
        'source: "alpha-lab / Experimental Level 3"',
        f"tags: {tag_str}",
        "status: draft",
        "experiment_kind: execution_impact",
        f"case_name: {resolved_case_name}",
        f"workflow: {resolved_workflow}",
        f'git_commit: {resolved_git_commit}',
        f"run_timestamp_utc: {report.generated_at_utc}",
        "---",
    ]

    comparison = report.comparison_summary or {}
    reason_rows = [
        (
            f"| {row.get('reason_code', 'unknown')} | "
            f"{row.get('skipped_order_count', 0)} | "
            f"{_fmt_float(float(row.get('skipped_order_ratio', float('nan'))))} |"
        )
        for row in report.to_dict().get("reason_summary", [])
        if isinstance(row, dict)
    ]
    if not reason_rows:
        reason_rows = ["| none | 0 | — |"]

    flag_rows = [
        (
            f"| {flag.name} | {str(flag.triggered).lower() if flag.triggered is not None else 'none'} | "
            f"{_fmt_observed(flag.observed)} | {_fmt_threshold(flag.threshold)} |"
        )
        for flag in report.flags
    ]
    if not flag_rows:
        flag_rows = ["| none | none | — | — |"]

    related_lines = (
        [f"- {entry}" for entry in related_cards]
        if related_cards
        else ["<!-- Add related cards here -->"]
    )

    lines = [
        *frontmatter_lines,
        "",
        f"# {card_name}",
        "",
        "> *Auto-generated by `alpha-lab` experimental Level 3 tooling. "
        "Frontmatter, setup, and machine metrics must not be edited manually.*  ",
        "> *Interpretation, Next Steps, Related Cards, and Notes are for manual completion.*",
        "",
        "## 基本信息",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Case | `{resolved_case_name}` |",
        f"| Workflow | `{resolved_workflow}` |",
        f"| Primary Run Path | `{report.run_path}` |",
        f"| Comparison Run Path | `{report.comparison_run_path}` |",
        f"| Dominant Execution Blocker | `{report.dominant_execution_blocker or 'none'}` |",
        f"| Generated At | `{report.generated_at_utc}` |",
        f"| Git commit | `{resolved_git_commit}` |",
        "",
        "## 结果",
        "",
        "| Metric | Value |",
        "|---|---|",
        (
            "| Mean Abs Weight Diff | "
            f"{_fmt_float(float(report.execution_deviation_summary.get('mean_abs_weight_diff', float('nan'))))} |"
        ),
        (
            "| Max Abs Weight Diff | "
            f"{_fmt_float(float(report.execution_deviation_summary.get('max_abs_weight_diff', float('nan'))))} |"
        ),
        (
            "| Realized Gross Mean | "
            f"{_fmt_float(float(report.execution_deviation_summary.get('realized_gross_mean', float('nan'))))} |"
        ),
        (
            "| Target Gross Mean | "
            f"{_fmt_float(float(report.execution_deviation_summary.get('target_gross_mean', float('nan'))))} |"
        ),
        (
            "| Primary Mean Turnover | "
            f"{_fmt_float(float(report.turnover_effect_summary.get('primary_mean_turnover', float('nan'))))} |"
        ),
        (
            "| Comparison Mean Turnover | "
            f"{_fmt_float(float(report.turnover_effect_summary.get('comparison_mean_turnover', float('nan'))))} |"
        ),
        (
            "| Turnover Reduction Ratio | "
            f"{_fmt_float(float(report.turnover_effect_summary.get('turnover_reduction_ratio_vs_comparison', float('nan'))))} |"
        ),
        (
            "| Primary Total Return | "
            f"{_fmt_nested_comparison(comparison, 'total_return', 'primary')} |"
        ),
        (
            "| Comparison Total Return | "
            f"{_fmt_nested_comparison(comparison, 'total_return', 'comparison')} |"
        ),
        (
            "| Total Return Delta | "
            f"{_fmt_nested_comparison(comparison, 'total_return', 'delta_primary_minus_comparison')} |"
        ),
        (
            "| Primary Sharpe | "
            f"{_fmt_nested_comparison(comparison, 'sharpe_annualized', 'primary')} |"
        ),
        (
            "| Comparison Sharpe | "
            f"{_fmt_nested_comparison(comparison, 'sharpe_annualized', 'comparison')} |"
        ),
        (
            "| Sharpe Delta | "
            f"{_fmt_nested_comparison(comparison, 'sharpe_annualized', 'delta_primary_minus_comparison')} |"
        ),
        (
            "| Primary Max Drawdown | "
            f"{_fmt_nested_comparison(comparison, 'max_drawdown', 'primary')} |"
        ),
        (
            "| Comparison Max Drawdown | "
            f"{_fmt_nested_comparison(comparison, 'max_drawdown', 'comparison')} |"
        ),
        (
            "| Max Drawdown Delta | "
            f"{_fmt_nested_comparison(comparison, 'max_drawdown', 'delta_primary_minus_comparison')} |"
        ),
        "",
        "## Skipped Order Reasons",
        "",
        "| Reason | Count | Ratio |",
        "|---|---:|---:|",
        *reason_rows,
        "",
        "## Execution Flags",
        "",
        "| Flag | Triggered | Observed | Threshold |",
        "|---|---|---:|---:|",
        *flag_rows,
        "",
        "## Semantic Consistency",
        "",
        "| Scope | Status | Highest Severity | N Fail | N Warn |",
        "|---|---|---|---:|---:|",
        (
            f"| primary | "
            f"{_fmt_semantic_value(report.semantic_consistency, 'primary', 'status')} | "
            f"{_fmt_semantic_value(report.semantic_consistency, 'primary', 'highest_severity')} | "
            f"{_fmt_semantic_int(report.semantic_consistency, 'primary', 'n_fail')} | "
            f"{_fmt_semantic_int(report.semantic_consistency, 'primary', 'n_warn')} |"
        ),
        (
            f"| comparison | "
            f"{_fmt_semantic_value(report.semantic_consistency, 'comparison', 'status')} | "
            f"{_fmt_semantic_value(report.semantic_consistency, 'comparison', 'highest_severity')} | "
            f"{_fmt_semantic_int(report.semantic_consistency, 'comparison', 'n_fail')} | "
            f"{_fmt_semantic_int(report.semantic_consistency, 'comparison', 'n_warn')} |"
        ),
        "",
        "## 解释",
        "",
        "<!-- Add interpretation here -->",
        "",
        "## 下一步",
        "",
        "<!-- Add next steps here -->",
        "",
        "## 相关卡片",
        "",
        *related_lines,
        "",
        "## 开放问题",
        "",
        "<!-- What needs further investigation? -->",
    ]

    if notes is not None:
        lines += ["", "## 备注", "", notes]

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _split_description(
    train_end: pd.Timestamp | None,
    test_start: pd.Timestamp | None,
) -> str:
    """Return a concise human-readable split label sourced from the result."""
    if train_end is not None and test_start is not None:
        return f"train<={train_end.date()} / test>={test_start.date()}"
    return "full_sample"


def _fmt_float(value: float, precision: int = 4) -> str:
    """Format a float for display; NaN renders as '\u2014' (em dash)."""
    if math.isnan(value):
        return "\u2014"
    return f"{value:.{precision}f}"


def _fmt_ci(lower: float, upper: float, precision: int = 4) -> str:
    if not math.isfinite(lower) or not math.isfinite(upper):
        return "\u2014"
    return f"[{lower:.{precision}f}, {upper:.{precision}f}]"


def _fmt_flags(flags: tuple[str, ...]) -> str:
    if not flags:
        return "none"
    return ", ".join(flags)


def _fmt_reason_text(value: object) -> str:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return "none"
    return text


def _fmt_observed(value: object) -> str:
    if value is None:
        return "\u2014"
    if isinstance(value, (int, float)):
        return _fmt_float(float(value))
    text = str(value).strip()
    return text or "\u2014"


def _fmt_threshold(value: float | None) -> str:
    if value is None:
        return "\u2014"
    return _fmt_float(float(value))


def _fmt_nested_comparison(
    payload: dict[str, object],
    metric_name: str,
    field_name: str,
) -> str:
    metric = payload.get(metric_name)
    if not isinstance(metric, dict):
        return "\u2014"
    value = metric.get(field_name)
    if not isinstance(value, (int, float)):
        return "\u2014"
    return _fmt_float(float(value))


def _fmt_semantic_value(
    payload: dict[str, object] | None,
    scope: str,
    field_name: str,
) -> str:
    if not isinstance(payload, dict):
        return "none"
    scope_payload = payload.get(scope)
    if not isinstance(scope_payload, dict):
        return "none"
    value = scope_payload.get(field_name)
    text = str(value).strip() if value is not None else ""
    return text or "none"


def _fmt_semantic_int(
    payload: dict[str, object] | None,
    scope: str,
    field_name: str,
) -> str:
    if not isinstance(payload, dict):
        return "0"
    scope_payload = payload.get(scope)
    if not isinstance(scope_payload, dict):
        return "0"
    value = scope_payload.get(field_name)
    if isinstance(value, (int, float)):
        return str(int(value))
    return "0"
