from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from alpha_lab.experiment import ExperimentResult

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
)


def summarise_experiment_result(result: ExperimentResult) -> pd.DataFrame:
    """Produce a compact one-row summary DataFrame from an ExperimentResult.

    All metadata (factor name, label name, quantile count, split boundaries)
    is read directly from ``result`` — nothing is inferred from heuristics or
    supplied by the caller.  Metric values are taken from ``result.summary``
    without re-computation.

    The returned DataFrame has exactly one row and columns in
    :data:`SUMMARY_COLUMNS`.  Multiple rows can be stacked with
    ``pd.concat([row1, row2], ignore_index=True)``.

    Parameters
    ----------
    result:
        Output of :func:`~alpha_lab.experiment.run_factor_experiment`.

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
    notes: str | None = None,
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
    notes:
        Optional free-text appended under a ``## Notes`` section.

    Returns
    -------
    str
        Markdown text ending with a single newline.
    """
    summary_df = summarise_experiment_result(result)
    row = summary_df.iloc[0]

    factor_name = str(row["factor_name"])
    label_name = str(row["label_name"])
    split_desc = str(row["split_description"])
    n_dates = int(row["n_dates_used"])  # type: ignore[arg-type]
    nq_str = str(int(row["n_quantiles"]))  # type: ignore[arg-type]

    if title is None:
        title = f"Experiment: {factor_name}"

    s = result.summary
    lines = [
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
        f"| Mean IC | {_fmt_float(s.mean_ic)} |",
        f"| Mean Rank IC | {_fmt_float(s.mean_rank_ic)} |",
        f"| IC IR | {_fmt_float(s.ic_ir)} |",
        f"| Mean L/S Return | {_fmt_float(s.mean_long_short_return)} |",
        f"| L/S Hit Rate | {_fmt_float(s.long_short_hit_rate)} |",
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
