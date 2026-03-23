from __future__ import annotations

import datetime
import math
from pathlib import Path

import pandas as pd

from alpha_lab.costs import cost_adjusted_long_short
from alpha_lab.experiment import ExperimentResult
from alpha_lab.obsidian import write_obsidian_note

# Ordered column names for summary DataFrames produced by this module.
SUMMARY_COLUMNS: tuple[str, ...] = (
    "factor_name",
    "label_name",
    "n_quantiles",
    "split_description",
    "mean_ic",
    "mean_rank_ic",
    "ic_ir",
    "mean_long_short_return",
    "long_short_hit_rate",
    "n_dates_used",
    "mean_long_short_turnover",
    "cost_rate",
    "mean_cost_adjusted_long_short_return",
)


def summarise_experiment_result(
    result: ExperimentResult,
    *,
    cost_rate: float | None = None,
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
    row: dict[str, object] = {
        "factor_name": factor_name,
        "label_name": label_name,
        "n_quantiles": result.n_quantiles,
        "split_description": _split_description(result.train_end, result.test_start),
        "mean_ic": s.mean_ic,
        "mean_rank_ic": s.mean_rank_ic,
        "ic_ir": s.ic_ir,
        "mean_long_short_return": s.mean_long_short_return,
        "long_short_hit_rate": s.long_short_hit_rate,
        "n_dates_used": s.n_dates,
        "mean_long_short_turnover": s.mean_long_short_turnover,
        "cost_rate": cost_rate if cost_rate is not None else float("nan"),
        "mean_cost_adjusted_long_short_return": mean_cost_adj,
    }
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
        Optional free-text appended under a ``## Notes`` section.
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
    metrics_rows = [
        f"| Mean IC | {_fmt_float(s.mean_ic)} |",
        f"| Mean Rank IC | {_fmt_float(s.mean_rank_ic)} |",
        f"| IC IR | {_fmt_float(s.ic_ir)} |",
        f"| Mean L/S Return | {_fmt_float(s.mean_long_short_return)} |",
        f"| L/S Hit Rate | {_fmt_float(s.long_short_hit_rate)} |",
        f"| Mean L/S Turnover (one-way) | {_fmt_float(s.mean_long_short_turnover)} |",
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
        "## Experiment",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Factor | `{factor_name}` |",
        f"| Label | `{label_name}` |",
        f"| N quantiles | {nq_str} |",
        f"| Split | {split_desc} |",
        f"| Eval dates (finite IC) | {n_dates} |",
        "",
        "## Summary Metrics",
        "",
        "| Metric | Value |",
        "|---|---|",
        *metrics_rows,
        "",
        "## Interpretation",
        "",
        "<!-- Add interpretation here -->",
        "",
        "## Next Steps",
        "",
        "<!-- Add next steps here -->",
    ]

    if notes is not None:
        lines += [
            "",
            "## Notes",
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
        :data:`~alpha_lab.config.OBSIDIAN_VAULT_PATH`.  Raises
        :exc:`ValueError` when both this argument and the config value
        are ``None``.
    cost_rate:
        Optional one-way transaction cost rate.  When provided, a
        cost-adjusted return row is appended to the metrics table.
    notes:
        Optional free-text appended under a ``## Notes`` section.
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
    resolved_vault: Path
    if vault_path is not None:
        resolved_vault = Path(vault_path).resolve()
    else:
        from alpha_lab.config import OBSIDIAN_VAULT_PATH

        if OBSIDIAN_VAULT_PATH is None:
            raise ValueError(
                "vault_path was not provided and OBSIDIAN_VAULT_PATH is not "
                "set.  Pass vault_path explicitly or set the "
                "OBSIDIAN_VAULT_PATH environment variable."
            )
        resolved_vault = OBSIDIAN_VAULT_PATH

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
    return write_obsidian_note(markdown, output_path, overwrite=overwrite)


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
    metrics_rows = [
        f"| Mean IC | {_fmt_float(s.mean_ic)} |",
        f"| Mean Rank IC | {_fmt_float(s.mean_rank_ic)} |",
        f"| IC IR | {_fmt_float(s.ic_ir)} |",
        f"| Mean L/S Return | {_fmt_float(s.mean_long_short_return)} |",
        f"| L/S Hit Rate | {_fmt_float(s.long_short_hit_rate)} |",
        f"| Mean L/S Turnover (one-way) | {_fmt_float(s.mean_long_short_turnover)} |",
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
        "## Setup",
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
        "## Results",
        "",
        "| Metric | Value |",
        "|---|---|",
        *metrics_rows,
        "",
        "## Interpretation",
        "",
        "<!-- Add interpretation here -->",
        "",
        "## Next Steps",
        "",
        "<!-- Add next steps here -->",
        "",
        "## Related Cards",
        "",
        f"- [[30_factors/Factor - {factor_name}]]",
        "",
        "## Open Questions",
        "",
        "<!-- What needs further investigation? -->",
    ]

    if notes is not None:
        lines += [
            "",
            "## Notes",
            "",
            notes,
        ]

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
