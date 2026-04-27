"""Shared IO and validation helpers for real-case research pipelines.

Extracted from ``single_factor.pipeline`` and ``composite.pipeline`` to
eliminate duplication.  Both pipelines delegate prices/universe/factor loading
and universe filtering to this module.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from alpha_lab.data_quality.corporate_actions import adjust_for_dividends
from alpha_lab.exceptions import AlphaLabDataError, AlphaLabIOError
from alpha_lab.real_cases.common_spec import UniverseSpec
from alpha_lab.research_contracts import validate_prices_table
from alpha_lab.sorted_panel import ensure_sorted, mark_sorted

# ---------------------------------------------------------------------------
# Prices
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParquetFrameResolution:
    """Concrete parquet path resolved for a tabular input."""

    path: Path
    source_path: Path
    materialized: bool


def load_tabular_frame(
    path_value: str,
    *,
    object_name: str,
    columns: Iterable[str] | None = None,
    optional_columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Load a tabular file from CSV or Parquet.

    Supported suffixes:
    - ``.csv``
    - ``.parquet``
    - ``.pq``

    Parquet-first resolution: when ``path_value`` points at a ``.csv`` and a
    sibling ``.parquet`` (or ``.pq``) exists alongside it whose mtime is at
    least as recent as the CSV, the parquet file is loaded instead. This keeps
    specs CSV-oriented while letting case inputs transparently upgrade to
    parquet for faster IO. If the parquet is older than the CSV (stale export),
    the CSV still wins.

    ``columns`` and ``optional_columns`` enable column pruning for wide tables.
    Required columns must exist; optional columns are read only when present.
    """

    resolved = resolve_tabular_frame_path(path_value, object_name=object_name)
    suffix = resolved.suffix.lower()
    read_columns = _resolve_read_columns(
        resolved,
        suffix=suffix,
        object_name=object_name,
        columns=columns,
        optional_columns=optional_columns,
    )
    if suffix == ".csv":
        return pd.read_csv(resolved, usecols=list(read_columns) if read_columns else None)
    return pd.read_parquet(resolved, columns=list(read_columns) if read_columns else None)


def resolve_tabular_frame_path(path_value: str, *, object_name: str) -> Path:
    """Resolve the concrete file path to read from a CSV/Parquet input.

    Resolution rules:
    - ``.parquet`` / ``.pq`` inputs are used directly.
    - ``.csv`` inputs prefer a fresh-enough sibling parquet file.
    """
    path = Path(path_value)
    if not path.exists() or not path.is_file():
        raise AlphaLabIOError(f"{object_name} file does not exist: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        parquet_sibling = _resolve_parquet_sibling(path)
        return parquet_sibling if parquet_sibling is not None else path
    if suffix in {".parquet", ".pq"}:
        return path

    raise AlphaLabIOError(
        f"{object_name} file must use one of ['.csv', '.parquet', '.pq']; got: {path}"
    )


def ensure_parquet_tabular_frame(path_value: str, *, object_name: str) -> ParquetFrameResolution:
    """Resolve or materialize a parquet file for a CSV/Parquet input.

    Model-factor feature tables can be large enough that CSV parsing dominates
    repeated lab runs. A CSV input is therefore treated as a one-time source:
    if a fresh sibling parquet already exists it is reused; otherwise the CSV is
    read once and written to ``<stem>.parquet`` before the caller loads data.
    """

    path = Path(path_value)
    if not path.exists() or not path.is_file():
        raise AlphaLabIOError(f"{object_name} file does not exist: {path}")

    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return ParquetFrameResolution(path=path, source_path=path, materialized=False)
    if suffix != ".csv":
        raise AlphaLabIOError(
            f"{object_name} file must use one of ['.csv', '.parquet', '.pq']; got: {path}"
        )

    parquet_sibling = _resolve_parquet_sibling(path)
    if parquet_sibling is not None:
        return ParquetFrameResolution(
            path=parquet_sibling,
            source_path=path,
            materialized=False,
        )

    target = path.with_suffix(".parquet")
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        raise AlphaLabIOError(
            f"{object_name} CSV could not be read for parquet materialization: {path}: {exc}"
        ) from exc
    try:
        frame.to_parquet(target, index=False)
    except Exception as exc:
        raise AlphaLabIOError(
            f"{object_name} parquet materialization failed: {target}: {exc}"
        ) from exc
    return ParquetFrameResolution(path=target, source_path=path, materialized=True)


def _resolve_parquet_sibling(csv_path: Path) -> Path | None:
    """Return a fresh-enough sibling .parquet/.pq for a CSV, or None."""
    try:
        csv_mtime = csv_path.stat().st_mtime
    except OSError:
        return None
    for suffix in (".parquet", ".pq"):
        sibling = csv_path.with_suffix(suffix)
        try:
            sibling_mtime = sibling.stat().st_mtime
        except OSError:
            continue
        if sibling_mtime >= csv_mtime:
            return sibling
    return None


def _resolve_read_columns(
    path: Path,
    *,
    suffix: str,
    object_name: str,
    columns: Iterable[str] | None,
    optional_columns: Iterable[str] | None,
) -> tuple[str, ...] | None:
    required = _normalize_column_request(columns)
    optional = tuple(
        column for column in _normalize_column_request(optional_columns) if column not in required
    )
    if not required and not optional:
        return None

    available = _read_tabular_column_names(path, suffix=suffix)
    missing = [column for column in required if column not in available]
    if missing:
        raise AlphaLabDataError(f"{object_name} is missing required columns: {missing}")
    present_optional = [column for column in optional if column in available]
    return (*required, *present_optional)


def _normalize_column_request(columns: Iterable[str] | None) -> tuple[str, ...]:
    if columns is None:
        return ()
    out: list[str] = []
    seen: set[str] = set()
    for raw in columns:
        column = str(raw).strip()
        if not column or column in seen:
            continue
        out.append(column)
        seen.add(column)
    return tuple(out)


def _read_tabular_column_names(path: Path, *, suffix: str) -> frozenset[str]:
    if suffix == ".csv":
        return frozenset(str(column) for column in pd.read_csv(path, nrows=0).columns)
    try:
        import pyarrow.parquet as pq  # type: ignore[import-untyped]
    except ImportError:  # pragma: no cover - pandas parquet engines vary by environment
        return frozenset(str(column) for column in pd.read_parquet(path).columns)
    parquet_file = pq.ParquetFile(path)
    return frozenset(str(column) for column in parquet_file.schema_arrow.names)


def load_prices(
    path_value: str,
    *,
    columns: Iterable[str] | None = None,
    optional_columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Load, validate, and return a price panel from CSV/Parquet.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If required columns are missing or validation fails.
    """
    read_columns: tuple[str, ...] | None = None
    read_optional_columns: tuple[str, ...] | None = None
    if columns is not None or optional_columns is not None:
        read_columns = _normalize_column_request(("date", "asset", "close", *(columns or ())))
        read_optional_columns = _normalize_column_request(
            (
                "dividend_per_share",
                *(optional_columns or ()),
            )
        )

    prices = load_tabular_frame(
        path_value,
        object_name="prices",
        columns=read_columns,
        optional_columns=read_optional_columns,
    )
    required = {"date", "asset", "close"}
    missing = required - set(prices.columns)
    if missing:
        raise AlphaLabDataError(f"prices is missing required columns: {sorted(missing)}")

    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
    prices = ensure_sorted(prices, by=("asset", "date"))
    prices = _apply_optional_dividend_adjustment(prices)
    prices = mark_sorted(prices, ("asset", "date"))
    validate_prices_table(prices)
    return prices


def _apply_optional_dividend_adjustment(prices: pd.DataFrame) -> pd.DataFrame:
    """Apply default dividend back-adjustment when event column is present.

    The daily price panel may include an optional ``dividend_per_share`` column
    where non-zero entries mark ex-dividend dates. When present, the single-
    factor pipeline should use adjusted close prices by default so all downstream
    labels (including cached horizon=1 labels) are aligned to the same adjusted
    close path.
    """

    if "dividend_per_share" not in prices.columns:
        return prices

    div = pd.to_numeric(prices["dividend_per_share"], errors="coerce")
    event_mask = div.notna() & div.ne(0.0)
    if not bool(event_mask.any()):
        return prices

    events = prices.loc[event_mask, ["asset", "date"]].copy()
    events["dividend_per_share"] = div.loc[event_mask].astype(float).to_numpy()
    return adjust_for_dividends(prices, events)


# ---------------------------------------------------------------------------
# Universe mask
# ---------------------------------------------------------------------------


def load_universe_mask(universe_spec: UniverseSpec) -> pd.DataFrame | None:
    """Load an optional universe mask from CSV/Parquet.

    Returns ``None`` when ``universe_spec.path`` is ``None`` (no universe
    filter configured).

    Raises
    ------
    FileNotFoundError
        If the configured path does not exist.
    ValueError
        If required columns are missing or duplicates are found.
    """
    if universe_spec.path is None:
        return None

    col = universe_spec.in_universe_column
    universe = load_tabular_frame(
        universe_spec.path,
        object_name="universe",
        columns=("date", "asset", col),
    )
    required = {"date", "asset", col}
    missing = required - set(universe.columns)
    if missing:
        raise AlphaLabDataError(f"universe file is missing required columns: {sorted(missing)}")

    out = universe[["date", "asset", col]].copy()
    out = out.rename(columns={col: "in_universe"})
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date", "asset"]).copy()
    if out.duplicated(subset=["date", "asset"]).any():
        raise AlphaLabDataError("universe file contains duplicate (date, asset) rows")
    out["in_universe"] = _coerce_in_universe_flags(out["in_universe"])
    return out


def _coerce_in_universe_flags(raw: pd.Series) -> pd.Series:
    """Coerce universe inclusion flags with strict semantics.

    Accepted values:
    - booleans: ``True`` / ``False``
    - integers: ``1`` / ``0``
    - strings: ``"1"/"0"``, ``"true"/"false"``, ``"yes"/"no"``, ``"y"/"n"``
    """
    if raw.dtype == bool:
        return raw.astype(bool)

    values = raw.copy()

    # Handle numeric values first so 0/1 remain unambiguous.
    numeric = pd.to_numeric(values, errors="coerce")
    numeric_mask = numeric.notna()
    if bool(numeric_mask.any()):
        numeric_values = numeric.loc[numeric_mask]
        invalid_numeric = numeric_values[~numeric_values.isin([0, 1])]
        if not invalid_numeric.empty:
            sample = ", ".join(str(v) for v in invalid_numeric.head(5).tolist())
            raise AlphaLabDataError(
                f"universe in_universe column contains numeric values outside {{0,1}}: {sample}"
            )

    normalized = values.astype(str).str.strip().str.lower()
    true_tokens = {"1", "true", "t", "yes", "y"}
    false_tokens = {"0", "false", "f", "no", "n", "", "nan", "none", "null"}
    token_mask = normalized.isin(true_tokens | false_tokens)

    unresolved = values.loc[~numeric_mask & ~token_mask]
    if not unresolved.empty:
        sample = ", ".join(str(v) for v in unresolved.head(5).tolist())
        raise AlphaLabDataError(
            "universe in_universe column contains unsupported values: "
            f"{sample}; allowed boolean/0/1 token values only"
        )

    out = pd.Series(False, index=values.index, dtype=bool)
    out.loc[numeric_mask] = numeric.loc[numeric_mask].astype(int).eq(1)
    out.loc[~numeric_mask] = normalized.loc[~numeric_mask].isin(true_tokens)
    return out


# ---------------------------------------------------------------------------
# Universe filtering
# ---------------------------------------------------------------------------


def apply_universe_to_prices(prices: pd.DataFrame, universe_mask: pd.DataFrame) -> pd.DataFrame:
    """Inner-join prices against the active universe rows.

    Raises
    ------
    ValueError
        If the result is empty after filtering.
    """
    active = universe_mask[universe_mask["in_universe"]][["date", "asset"]]
    out = prices.merge(active, on=["date", "asset"], how="inner", validate="many_to_one")
    if out.empty:
        raise AlphaLabDataError("prices became empty after universe filtering")
    return ensure_sorted(out, by=("asset", "date"))


def apply_universe_to_factor(factor_df: pd.DataFrame, universe_mask: pd.DataFrame) -> pd.DataFrame:
    """Inner-join a factor DataFrame against the active universe rows.

    Raises
    ------
    ValueError
        If the result is empty after filtering.
    """
    active = universe_mask[universe_mask["in_universe"]][["date", "asset"]]
    out = factor_df.merge(active, on=["date", "asset"], how="inner", validate="many_to_one")
    if out.empty:
        raise AlphaLabDataError("factor data became empty after universe filtering")
    return ensure_sorted(out, by=("date", "asset"))
