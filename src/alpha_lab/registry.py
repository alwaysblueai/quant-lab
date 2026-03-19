from __future__ import annotations

import datetime
from pathlib import Path

import pandas as pd

from alpha_lab.reporting import SUMMARY_COLUMNS

DEFAULT_REGISTRY_PATH: Path = Path("data/processed/experiment_registry.csv")

REGISTRY_COLUMNS: tuple[str, ...] = (
    "experiment_name",
    "factor_name",
    "label_factor",
    "quantiles",
    "split_description",
    "cost_rate",
    "mean_ic",
    "ic_ir",
    "mean_long_short_return",
    "mean_cost_adjusted_long_short_return",
    "timestamp",
    "obsidian_path",
)


def register_experiment(
    name: str,
    summary: pd.DataFrame,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
    *,
    obsidian_path: str | None = None,
) -> None:
    """Register one experiment result in the CSV registry.

    Converts a one-row summary DataFrame (output of
    :func:`~alpha_lab.reporting.summarise_experiment_result`) into a registry
    row and appends it to ``registry_path``.  The file is created if it does
    not exist; if it exists its schema is validated before writing to prevent
    silent schema drift.

    Parameters
    ----------
    name:
        Human-readable experiment identifier (e.g.
        ``"momentum_20d_5q_oos_2023"``).  Not enforced to be unique — the
        registry is an append-only log; duplicate names are permitted.
    summary:
        One-row DataFrame produced by
        :func:`~alpha_lab.reporting.summarise_experiment_result`.
    registry_path:
        Path to the registry CSV file.  Defaults to
        ``data/processed/experiment_registry.csv``.
    obsidian_path:
        Optional path to an associated Obsidian markdown note.

    Raises
    ------
    TypeError
        If ``summary`` is not a :class:`pandas.DataFrame`.
    ValueError
        If ``summary`` is empty or missing expected columns.
    ValueError
        If the existing registry file has an incompatible schema.
    """
    if not isinstance(summary, pd.DataFrame):
        raise TypeError(
            f"summary must be a pandas DataFrame, got {type(summary).__name__}"
        )
    if summary.empty:
        raise ValueError("summary DataFrame is empty")
    if len(summary) != 1:
        raise ValueError(
            f"summary must contain exactly one row, got {len(summary)}"
        )
    missing = set(SUMMARY_COLUMNS) - set(summary.columns)
    if missing:
        raise ValueError(
            f"summary is missing required columns: {sorted(missing)}"
        )
    extra = set(summary.columns) - set(SUMMARY_COLUMNS)
    if extra:
        raise ValueError(
            f"summary contains unexpected columns: {sorted(extra)}"
        )

    row = _summary_to_registry_row(name, summary, obsidian_path=obsidian_path)
    append_to_registry(row, registry_path)


def load_registry(
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
) -> pd.DataFrame:
    """Load the experiment registry from a CSV file.

    Returns an empty DataFrame with :data:`REGISTRY_COLUMNS` columns if the
    file does not exist.

    Parameters
    ----------
    registry_path:
        Path to the registry CSV file.

    Returns
    -------
    pd.DataFrame
        Registry DataFrame with columns in :data:`REGISTRY_COLUMNS`.

    Raises
    ------
    ValueError
        If the file exists but its columns do not include all
        :data:`REGISTRY_COLUMNS`.
    """
    path = Path(registry_path)
    if not path.exists():
        return pd.DataFrame(columns=list(REGISTRY_COLUMNS))

    df = pd.read_csv(path)
    _check_exact_schema(df.columns, path)
    return df[list(REGISTRY_COLUMNS)].reset_index(drop=True)


def append_to_registry(
    row: pd.DataFrame,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
) -> None:
    """Append a registry row to the CSV file.

    Creates the file (and any missing parent directories) if it does not
    exist.  If the file exists its column schema is validated before writing.

    Parameters
    ----------
    row:
        One-row DataFrame whose columns must include all
        :data:`REGISTRY_COLUMNS`.
    registry_path:
        Destination CSV file path.

    Raises
    ------
    ValueError
        If ``row`` is missing required registry columns or contains extra
        columns not in :data:`REGISTRY_COLUMNS`.
    ValueError
        If the existing registry file has a schema that does not exactly match
        :data:`REGISTRY_COLUMNS`.
    TypeError
        If ``row`` is not a :class:`pandas.DataFrame`.
    """
    if not isinstance(row, pd.DataFrame):
        raise TypeError(f"row must be a pandas DataFrame, got {type(row).__name__}")

    missing = set(REGISTRY_COLUMNS) - set(row.columns)
    if missing:
        raise ValueError(
            f"row is missing required registry columns: {sorted(missing)}"
        )
    extra = set(row.columns) - set(REGISTRY_COLUMNS)
    if extra:
        raise ValueError(
            f"row contains unexpected columns: {sorted(extra)}"
        )

    path = Path(registry_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    out = row[list(REGISTRY_COLUMNS)]

    if path.exists():
        existing = pd.read_csv(path)
        _check_exact_schema(existing.columns, path)
        out.to_csv(path, mode="a", header=False, index=False)
    else:
        out.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _check_exact_schema(
    columns: pd.Index,  # type: ignore[type-arg]
    path: Path,
) -> None:
    """Raise ValueError if *columns* does not exactly match REGISTRY_COLUMNS.

    Both column *set* and column *order* are validated.  CSV appends are
    positional: if the existing file's header has the same columns in a
    different order, appended rows would be misaligned.  Strict ordered
    equality is therefore required for append safety.
    """
    col_list = list(columns)
    expected = list(REGISTRY_COLUMNS)

    col_set = set(col_list)
    expected_set = set(expected)
    missing = expected_set - col_set
    extra = col_set - expected_set
    wrong_order = (not missing) and (not extra) and (col_list != expected)

    if missing or extra or wrong_order:
        parts: list[str] = []
        if missing:
            parts.append(f"missing: {sorted(missing)}")
        if extra:
            parts.append(f"unexpected: {sorted(extra)}")
        if wrong_order:
            parts.append(f"wrong order: got {col_list}, expected {expected}")
        raise ValueError(
            f"Registry file {path} has an incompatible schema "
            f"({'; '.join(parts)}).  Refusing to proceed to avoid schema drift."
        )


def _summary_to_registry_row(
    name: str,
    summary: pd.DataFrame,
    *,
    obsidian_path: str | None,
) -> pd.DataFrame:
    """Convert a one-row summary DataFrame to a one-row registry DataFrame."""
    s = summary.iloc[0]
    row: dict[str, object] = {
        "experiment_name": name,
        "factor_name": s["factor_name"],
        "label_factor": s["label_name"],
        "quantiles": s["n_quantiles"],
        "split_description": s["split_description"],
        "cost_rate": s["cost_rate"],
        "mean_ic": s["mean_ic"],
        "ic_ir": s["ic_ir"],
        "mean_long_short_return": s["mean_long_short_return"],
        "mean_cost_adjusted_long_short_return": s[
            "mean_cost_adjusted_long_short_return"
        ],
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "obsidian_path": obsidian_path if obsidian_path is not None else "",
    }
    return pd.DataFrame([row], columns=list(REGISTRY_COLUMNS))
