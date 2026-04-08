from __future__ import annotations

import pandas as pd

from alpha_lab.exceptions import AlphaLabDataError
from alpha_lab.reporting import SUMMARY_COLUMNS

# Ordered output columns for comparison DataFrames.
# ``label_name`` is renamed to ``label_factor`` and ``n_quantiles`` to
# ``quantiles`` for readability; all other columns preserve their names.
COMPARISON_COLUMNS: tuple[str, ...] = (
    "factor_name",
    "label_factor",
    "quantiles",
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

_RENAME: dict[str, str] = {
    "label_name": "label_factor",
    "n_quantiles": "quantiles",
}


def compare_experiments(summaries: list[pd.DataFrame]) -> pd.DataFrame:
    """Stack a list of experiment summaries into a single comparison DataFrame.

    Each element of ``summaries`` must be a one-row DataFrame produced by
    :func:`~alpha_lab.reporting.summarise_experiment_result`.  All elements
    must share the :data:`~alpha_lab.reporting.SUMMARY_COLUMNS` schema.

    Metrics are not recomputed.  Values are taken directly from the input
    summaries.

    Parameters
    ----------
    summaries:
        Non-empty list of one-row summary DataFrames.

    Returns
    -------
    pd.DataFrame
        One row per experiment, columns in :data:`COMPARISON_COLUMNS`.

    Raises
    ------
    ValueError
        If ``summaries`` is empty, or if any element is missing columns
        required by :data:`~alpha_lab.reporting.SUMMARY_COLUMNS`.
    TypeError
        If any element is not a :class:`pandas.DataFrame`.
    """
    if not summaries:
        raise AlphaLabDataError("summaries must be a non-empty list")

    for i, s in enumerate(summaries):
        if not isinstance(s, pd.DataFrame):
            raise AlphaLabDataError(
                f"summaries[{i}] must be a pandas DataFrame, got {type(s).__name__}"
            )
        if len(s) != 1:
            raise AlphaLabDataError(
                f"summaries[{i}] must contain exactly one row, got {len(s)}"
            )
        missing = set(SUMMARY_COLUMNS) - set(s.columns)
        if missing:
            raise AlphaLabDataError(
                f"summaries[{i}] is missing required columns: {sorted(missing)}"
            )
        extra = set(s.columns) - set(SUMMARY_COLUMNS)
        if extra:
            raise AlphaLabDataError(
                f"summaries[{i}] contains unexpected columns: {sorted(extra)}"
            )

    combined = pd.concat(summaries, ignore_index=True)
    combined = combined.rename(columns=_RENAME)
    return combined[list(COMPARISON_COLUMNS)].reset_index(drop=True)


def rank_experiments(
    comparison_df: pd.DataFrame,
    metric: str,
    *,
    ascending: bool = False,
) -> pd.DataFrame:
    """Return ``comparison_df`` sorted by ``metric``.

    All rows are preserved; none are dropped.  NaN values are placed last
    regardless of sort direction.

    Parameters
    ----------
    comparison_df:
        Output of :func:`compare_experiments`.
    metric:
        Column name to sort by.  Must be present in ``comparison_df``.
    ascending:
        Sort direction.  Defaults to ``False`` (highest value first), which
        is the natural ordering for most performance metrics.

    Returns
    -------
    pd.DataFrame
        A sorted copy of ``comparison_df`` with the index reset.

    Raises
    ------
    ValueError
        If ``metric`` is not a column in ``comparison_df``.

    Notes
    -----
    Ties in ``metric`` are broken by all remaining columns in their natural
    column order (each ascending), making the ranking fully deterministic for
    any fixed input regardless of how many rows share the same metric value.
    """
    if metric not in comparison_df.columns:
        raise AlphaLabDataError(
            f"metric {metric!r} is not a column in comparison_df; "
            f"available columns: {list(comparison_df.columns)}"
        )
    # Use all remaining columns as tiebreakers in their natural column order so
    # that the result is fully deterministic for any fixed input, including rows
    # that share both the primary metric and factor_name.
    remaining = [c for c in comparison_df.columns if c != metric]
    sort_cols = [metric] + remaining
    sort_ascending = [ascending] + [True] * len(remaining)
    return (
        comparison_df.sort_values(
            sort_cols, ascending=sort_ascending, na_position="last"
        )
        .reset_index(drop=True)
    )
