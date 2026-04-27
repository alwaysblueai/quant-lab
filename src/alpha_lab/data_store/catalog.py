from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.dataset as ds

from alpha_lab.config import resolve_data_root
from alpha_lab.data_validation import validate_price_panel
from alpha_lab.exceptions import AlphaLabConfigError, AlphaLabDataError, AlphaLabExperimentError
from alpha_lab.interfaces import validate_factor_output

_CORE_TABLES: tuple[str, ...] = (
    "trade_calendar",
    "daily_bars",
    "adj_factor",
    "daily_basic",
    "asset_status",
    "index_membership",
    "moneyflow",
    "liquidity_profile",
    "financial_indicator",
    "balance_sheet",
    "income_statement",
    "cash_flow_statement",
    "industry_classification",
    "industry_membership",
)

_INDEX_MEMBERSHIP_FLAG_COLUMNS: dict[str, str] = {
    "000300.SH": "is_hs300",
    "000905.SH": "is_zz500",
    "000852.SH": "is_zz1000",
    "000016.SH": "is_sz50",
}
_TOP_LIQUID_UNIVERSE_RE = re.compile(r"^top_liquid_(\d+)$")
_INSTITUTIONAL_UNIVERSE_NAME = "institutional_ashare"
_INSTITUTIONAL_MIN_LISTED_DAYS = 180
_INSTITUTIONAL_LIQUIDITY_LOOKBACK_DAYS = 20
_INSTITUTIONAL_MIN_ACTIVE_DAYS = 15
_INSTITUTIONAL_MIN_AVG_AMOUNT = 20_000.0
_INSTITUTIONAL_MIN_AMOUNT_PERCENTILE = 0.20


@dataclass(frozen=True)
class SnapshotManifest:
    snapshot_id: str
    vendor: str
    dataset_name: str
    requested_at_utc: str
    request_params: dict[str, object]
    row_counts: dict[str, int]
    file_hashes: dict[str, str]
    time_range: dict[str, str]
    notes: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class DatasetVersion:
    dataset_name: str
    version_id: str
    created_at_utc: str
    raw_snapshot_id: str | None
    table_hashes: dict[str, str]
    notes: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SliceSpec:
    start_date: str
    end_date: str
    slice_preset: str | None = None
    universe_name: str = "all_ashare"
    assets: tuple[str, ...] | None = None
    asset_limit: int | None = None
    factors: tuple[str, ...] = ()
    adjustment: str = "raw"

    def cache_key(self, *, dataset_version_id: str) -> str:
        normalized_adjustment = _normalize_adjustment(self.adjustment)
        payload = {
            "start_date": self.start_date,
            "end_date": self.end_date,
            "universe_name": self.universe_name,
            "assets": list(self.assets) if self.assets is not None else None,
            "asset_limit": self.asset_limit,
            "factors": list(self.factors),
            "adjustment": normalized_adjustment,
            "dataset_version_id": dataset_version_id,
        }
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


@dataclass(frozen=True)
class CaseInputExportResult:
    output_dir: Path
    output_paths: dict[str, Path]
    format_output_paths: dict[str, dict[str, Path]]
    row_counts: dict[str, int]
    dataset_version_id: str | None
    cache_key: str
    cache_dir: Path
    formats_written: tuple[str, ...]


@dataclass(frozen=True)
class CaseSliceBundle:
    cache_dir: Path
    cache_key: str
    dataset_version_id: str | None
    frames: dict[str, pd.DataFrame]


@dataclass(frozen=True)
class RawSnapshotQualityReport:
    snapshot_id: str
    vendor: str
    dataset_name: str
    ok: bool
    created_at_utc: str
    issues: tuple[QualityIssue, ...]
    table_stats: dict[str, dict[str, object]]
    report_path: Path

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["issues"] = [issue.to_dict() for issue in self.issues]
        payload["report_path"] = str(self.report_path)
        return payload


@dataclass(frozen=True)
class QualityIssue:
    table_name: str
    severity: str
    message: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class QualityReport:
    dataset_name: str
    ok: bool
    created_at_utc: str
    checked_tables: tuple[str, ...]
    issues: tuple[QualityIssue, ...]
    report_path: Path
    summary: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["issues"] = [issue.to_dict() for issue in self.issues]
        payload["report_path"] = str(self.report_path)
        return payload


class DataCatalog:
    """External research-data catalog backed by Parquet datasets."""

    CORE_DATASET_NAME = "tushare_core"
    REQUIRED_EXPORT_PRICE_COLUMNS: tuple[str, ...] = ("date", "asset", "close", "volume", "amount")
    RESEARCH_EXPORT_PRICE_COLUMNS: tuple[str, ...] = (
        "date",
        "asset",
        "open",
        "high",
        "low",
        "close",
        "pre_close",
        "volume",
        "amount",
        "vwap",
        "turnover_rate",
        "up_limit",
        "down_limit",
        "is_limit_up",
        "is_limit_down",
        "is_suspended",
        "is_st",
        "is_hs300",
        "is_zz500",
        "is_zz1000",
        "is_sz50",
    )

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = resolve_data_root(root)
        self._refresh_defer_depth = 0
        self._deferred_refresh_tables: set[str] = set()

    @property
    def raw_root(self) -> Path:
        return self.root / "raw"

    @property
    def canonical_root(self) -> Path:
        return self.root / "canonical"

    @property
    def cache_root(self) -> Path:
        return self.root / "cache"

    @property
    def metadata_root(self) -> Path:
        return self.root / "metadata"

    @property
    def dataset_versions_root(self) -> Path:
        return self.metadata_root / "dataset_versions"

    @property
    def tables_metadata_root(self) -> Path:
        return self.metadata_root / "tables"

    @property
    def validation_root(self) -> Path:
        return self.metadata_root / "validation"

    @property
    def raw_validation_root(self) -> Path:
        return self.validation_root / "raw"

    @property
    def canonical_validation_root(self) -> Path:
        return self.validation_root / "canonical"

    @property
    def duckdb_catalog_path(self) -> Path:
        return self.metadata_root / "catalog" / "catalog.duckdb"

    @property
    def slices_cache_root(self) -> Path:
        return self.cache_root / "slices"

    def ensure_layout(self) -> None:
        for path in (
            self.root,
            self.raw_root,
            self.canonical_root,
            self.cache_root,
            self.slices_cache_root,
            self.metadata_root,
            self.dataset_versions_root,
            self.tables_metadata_root,
            self.validation_root,
            self.raw_validation_root,
            self.canonical_validation_root,
            self.duckdb_catalog_path.parent,
        ):
            path.mkdir(parents=True, exist_ok=True)
        self._write_json(
            self.metadata_root / "layout.json",
            {
                "root": str(self.root),
                "canonical_root": str(self.canonical_root),
                "raw_root": str(self.raw_root),
                "cache_root": str(self.cache_root),
                "duckdb_available": bool(_optional_duckdb() is not None),
            },
        )

    @contextmanager
    def upsert_session(self) -> Iterator[None]:
        """Batch DuckDB catalog refreshes across multiple upserts.

        Within the session, ``upsert_table`` records changed table names and
        delays the DuckDB view refresh until the outermost session exits
        successfully. If the session fails with an exception, deferred refresh
        requests are discarded to avoid exposing a partially written batch.
        """
        self._refresh_defer_depth += 1
        success = False
        try:
            yield
            success = True
        finally:
            self._refresh_defer_depth = max(0, self._refresh_defer_depth - 1)
            if self._refresh_defer_depth == 0:
                if success:
                    self._flush_deferred_refresh()
                else:
                    self._deferred_refresh_tables.clear()

    def table_root(self, table_name: str) -> Path:
        return self.canonical_root / table_name

    def table_metadata_path(self, table_name: str) -> Path:
        return self.tables_metadata_root / f"{table_name}.json"

    def current_dataset_version_path(self, dataset_name: str) -> Path:
        return self.dataset_versions_root / f"{dataset_name}_current.json"

    def dataset_version_history_dir(self, dataset_name: str) -> Path:
        return self.dataset_versions_root / dataset_name

    def raw_snapshot_dir(self, *, vendor: str, dataset_name: str, snapshot_id: str) -> Path:
        return self.raw_root / vendor / dataset_name / snapshot_id

    def write_raw_snapshot(
        self,
        *,
        vendor: str,
        dataset_name: str,
        tables: dict[str, pd.DataFrame],
        request_params: dict[str, object],
        time_range: dict[str, str],
        notes: dict[str, object] | None = None,
    ) -> SnapshotManifest:
        self.ensure_layout()
        requested_at = _utc_now_text()
        digest = hashlib.sha256(
            json.dumps(
                {
                    "vendor": vendor,
                    "dataset_name": dataset_name,
                    "request_params": request_params,
                    "time_range": time_range,
                    "requested_at_utc": requested_at,
                },
                sort_keys=True,
                ensure_ascii=True,
            ).encode("utf-8")
        ).hexdigest()[:12]
        snapshot_id = f"{requested_at.replace(':', '').replace('-', '')}_{digest}"
        snapshot_dir = self.raw_snapshot_dir(
            vendor=vendor,
            dataset_name=dataset_name,
            snapshot_id=snapshot_id,
        )
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        row_counts: dict[str, int] = {}
        file_hashes: dict[str, str] = {}
        for name, frame in tables.items():
            path = snapshot_dir / f"{name}.parquet"
            frame.to_parquet(path, index=False, compression="zstd")
            row_counts[name] = int(len(frame))
            file_hashes[name] = _sha256_file(path)

        manifest = SnapshotManifest(
            snapshot_id=snapshot_id,
            vendor=vendor,
            dataset_name=dataset_name,
            requested_at_utc=requested_at,
            request_params=dict(request_params),
            row_counts=row_counts,
            file_hashes=file_hashes,
            time_range=dict(time_range),
            notes=dict(notes or {}),
        )
        self._write_json(snapshot_dir / "manifest.json", manifest.to_dict())
        return manifest

    def upsert_table(
        self,
        table_name: str,
        frame: pd.DataFrame,
        *,
        key_cols: tuple[str, ...],
        partition_column: str,
    ) -> tuple[Path, ...]:
        self.ensure_layout()
        if frame.empty:
            return tuple()
        out = frame.copy()
        out = out.reset_index(drop=True)
        out["_partition_year"] = _partition_year_values(out, partition_column)
        out["_partition_month"] = _partition_month_values(out, partition_column)
        table_root = self.table_root(table_name)
        table_root.mkdir(parents=True, exist_ok=True)
        written_dirs: list[Path] = []

        for (partition_year, partition_month), part_frame in out.groupby(
            ["_partition_year", "_partition_month"],
            sort=True,
        ):
            partition_dir = table_root / f"year={partition_year}" / f"month={partition_month}"
            incoming = part_frame.drop(columns=["_partition_year", "_partition_month"]).reset_index(
                drop=True
            )
            existing = self._read_partition(partition_dir)
            if not existing.empty:
                combined = pd.concat([existing, incoming], ignore_index=True)
            else:
                combined = incoming
            combined = combined.drop_duplicates(subset=list(key_cols), keep="last")
            combined = combined.sort_values(list(key_cols), kind="mergesort").reset_index(drop=True)
            self._write_partition(partition_dir, combined)
            written_dirs.append(partition_dir)

        self._write_json(
            self.table_metadata_path(table_name),
            {
                "table_name": table_name,
                "updated_at_utc": _utc_now_text(),
                "key_cols": list(key_cols),
                "partition_column": partition_column,
                "partition_granularity": "year_month",
                "columns": list(frame.columns),
                "written_partitions": [str(path) for path in written_dirs],
            },
        )
        self._schedule_catalog_refresh([table_name])
        return tuple(written_dirs)

    def load_table(
        self,
        table_name: str,
        *,
        columns: tuple[str, ...] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        assets: tuple[str, ...] | None = None,
        date_field: str | None = None,
    ) -> pd.DataFrame:
        table_root = self.table_root(table_name)
        if not table_root.exists():
            return pd.DataFrame(columns=list(columns or ()))

        parquet_files = sorted(table_root.rglob("*.parquet"))
        if not parquet_files:
            return pd.DataFrame(columns=list(columns or ()))

        dataset = ds.dataset(table_root, format="parquet", partitioning="hive")
        schema_names = set(dataset.schema.names)
        resolved_date_field = date_field
        if resolved_date_field is None:
            if "date" in schema_names:
                resolved_date_field = "date"
            elif "ann_date" in schema_names:
                resolved_date_field = "ann_date"

        filter_expr = None
        if resolved_date_field is not None and resolved_date_field in schema_names:
            if start_date is not None:
                expr = ds.field(resolved_date_field) >= start_date
                filter_expr = expr if filter_expr is None else filter_expr & expr
            if end_date is not None:
                expr = ds.field(resolved_date_field) <= end_date
                filter_expr = expr if filter_expr is None else filter_expr & expr
        if assets:
            expr = ds.field("asset").isin(list(assets))
            filter_expr = expr if filter_expr is None else filter_expr & expr

        resolved_columns: list[str] | None = None
        if columns is not None:
            resolved_columns = [column for column in columns if column in schema_names]
            if not resolved_columns:
                return pd.DataFrame(columns=list(columns))

        table = dataset.to_table(columns=resolved_columns, filter=filter_expr)
        return table.to_pandas()

    def latest_date(self, table_name: str, *, date_field: str = "date") -> str | None:
        if not table_name or not date_field:
            return None
        partition_latest = self._latest_date_from_partitions(
            table_name=table_name,
            date_field=date_field,
        )
        if partition_latest is not None:
            return partition_latest

        frame = self.load_table(table_name, columns=(date_field,), date_field=date_field)
        if frame.empty or date_field not in frame.columns:
            return None
        return self._latest_date_from_frame(frame, date_field)

    def _latest_date_from_partitions(
        self,
        *,
        table_name: str,
        date_field: str,
    ) -> str | None:
        table_root = self.table_root(table_name)
        if not table_root.exists():
            return None

        partition_months: list[tuple[int, int, Path]] = []
        for year_dir in table_root.glob("year=*"):
            if not year_dir.is_dir():
                continue
            try:
                year = int(year_dir.name.split("=", maxsplit=1)[1])
            except (IndexError, ValueError):
                continue
            for month_dir in year_dir.glob("month=*"):
                if not month_dir.is_dir():
                    continue
                try:
                    month = int(month_dir.name.split("=", maxsplit=1)[1])
                except (IndexError, ValueError):
                    continue
                if not 1 <= month <= 12:
                    continue
                partition_months.append((year, month, month_dir))

        if not partition_months:
            return None

        for _, _, partition_dir in sorted(partition_months, reverse=True):
            for parquet_path in sorted(partition_dir.glob("*.parquet")):
                frame = pd.read_parquet(parquet_path, columns=[date_field])
                if date_field not in frame.columns:
                    continue
                latest = self._latest_date_from_frame(frame, date_field)
                if latest is not None:
                    return latest
        return None

    def write_dataset_version(
        self,
        *,
        dataset_name: str,
        table_names: tuple[str, ...],
        raw_snapshot_id: str | None,
        notes: dict[str, object] | None = None,
    ) -> DatasetVersion:
        self.ensure_layout()
        table_hashes = {name: self._table_fingerprint(name) for name in table_names}
        payload = {
            "dataset_name": dataset_name,
            "table_hashes": table_hashes,
        }
        version_id = hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
        ).hexdigest()[:16]
        version = DatasetVersion(
            dataset_name=dataset_name,
            version_id=version_id,
            created_at_utc=_utc_now_text(),
            raw_snapshot_id=raw_snapshot_id,
            table_hashes=table_hashes,
            notes=dict(notes or {}),
        )
        self.dataset_version_history_dir(dataset_name).mkdir(parents=True, exist_ok=True)
        self._write_json(
            self.dataset_version_history_dir(dataset_name) / f"{version_id}.json",
            version.to_dict(),
        )
        self._write_json(self.current_dataset_version_path(dataset_name), version.to_dict())
        return version

    def get_current_dataset_version(self, dataset_name: str) -> DatasetVersion | None:
        path = self.current_dataset_version_path(dataset_name)
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        return DatasetVersion(
            dataset_name=str(payload["dataset_name"]),
            version_id=str(payload["version_id"]),
            created_at_utc=str(payload["created_at_utc"]),
            raw_snapshot_id=(
                str(payload["raw_snapshot_id"])
                if payload.get("raw_snapshot_id") is not None
                else None
            ),
            table_hashes={str(k): str(v) for k, v in dict(payload["table_hashes"]).items()},
            notes={str(k): v for k, v in dict(payload.get("notes", {})).items()},
        )

    def validate_raw_snapshot(
        self,
        snapshot_id: str,
        *,
        vendor: str = "tushare",
        dataset_name: str = "core",
    ) -> RawSnapshotQualityReport:
        self.ensure_layout()
        snapshot_dir = self.raw_snapshot_dir(
            vendor=vendor,
            dataset_name=dataset_name,
            snapshot_id=snapshot_id,
        )
        manifest_path = snapshot_dir / "manifest.json"
        if not manifest_path.exists():
            raise AlphaLabDataError(f"raw snapshot manifest not found: {manifest_path}")

        manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        issues: list[QualityIssue] = []
        table_stats: dict[str, dict[str, object]] = {}
        loaded_frames: dict[str, pd.DataFrame] = {}
        for parquet_path in sorted(snapshot_dir.glob("*.parquet")):
            table_name = parquet_path.stem
            frame = pd.read_parquet(parquet_path)
            loaded_frames[table_name] = frame
            stats = {
                "row_count": int(len(frame)),
                "columns": list(frame.columns),
            }
            key_cols = _raw_table_key_columns(table_name)
            if key_cols:
                present_key_cols = [column for column in key_cols if column in frame.columns]
                if len(present_key_cols) == len(key_cols):
                    dup_count = int(frame.duplicated(subset=present_key_cols).sum())
                    stats["duplicate_rows"] = dup_count
                    if dup_count:
                        issues.append(
                            QualityIssue(
                                table_name,
                                "warning",
                                f"raw snapshot contains duplicate rows: {dup_count}",
                            )
                        )
            required_cols = _raw_table_required_columns(table_name)
            if required_cols:
                missing_columns = sorted(set(required_cols) - set(frame.columns))
                stats["missing_required_columns"] = missing_columns
                if missing_columns:
                    issues.append(
                        QualityIssue(
                            table_name,
                            "error",
                            f"raw snapshot is missing required columns: {missing_columns}",
                        )
                    )
            if (
                table_name in {"prices_raw", "adj_factor_raw", "pb_raw", "roe_raw"}
                and not frame.empty
            ):
                stats["null_rows"] = _count_null_key_rows(frame)
            if table_name == "prices_raw" and not frame.empty:
                invalid_close = int(
                    (pd.to_numeric(frame.get("close"), errors="coerce") <= 0).fillna(False).sum()
                )
                stats["invalid_close_rows"] = invalid_close
                if invalid_close:
                    issues.append(
                        QualityIssue(
                            table_name, "error", f"close <= 0 rows detected: {invalid_close}"
                        )
                    )
                stats["asset_count"] = (
                    int(frame["asset"].astype(str).nunique()) if "asset" in frame else 0
                )
            if table_name == "adj_factor_raw" and not frame.empty:
                invalid_adj = int(
                    (pd.to_numeric(frame.get("adj_factor"), errors="coerce") <= 0)
                    .fillna(False)
                    .sum()
                )
                stats["invalid_adj_factor_rows"] = invalid_adj
                if invalid_adj:
                    issues.append(
                        QualityIssue(
                            table_name, "error", f"adj_factor <= 0 rows detected: {invalid_adj}"
                        )
                    )
            if table_name == "roe_raw" and not frame.empty:
                ann_date_series = (
                    frame["ann_date"]
                    if "ann_date" in frame.columns
                    else pd.Series(pd.NA, index=frame.index)
                )
                roe_value_series = (
                    pd.to_numeric(frame["roe_value"], errors="coerce")
                    if "roe_value" in frame.columns
                    else pd.Series(pd.NA, index=frame.index)
                )
                missing_ann_date = int((ann_date_series.isna() & roe_value_series.notna()).sum())
                stats["missing_ann_date_rows"] = missing_ann_date
                if missing_ann_date:
                    issues.append(
                        QualityIssue(
                            table_name,
                            "warning",
                            f"roe rows with value but missing ann_date: {missing_ann_date}",
                        )
                    )
            table_stats[table_name] = stats

        prices_raw = loaded_frames.get("prices_raw")
        adj_factor_raw = loaded_frames.get("adj_factor_raw")
        if prices_raw is not None and adj_factor_raw is not None and not prices_raw.empty:
            price_keys = prices_raw[["date", "asset"]].drop_duplicates()
            adj_keys = (
                adj_factor_raw[["date", "asset"]].drop_duplicates()
                if {"date", "asset"} <= set(adj_factor_raw.columns)
                else pd.DataFrame(columns=["date", "asset"])
            )
            missing_adj = int(
                price_keys.merge(adj_keys, on=["date", "asset"], how="left", indicator=True)[
                    "_merge"
                ]
                .eq("left_only")
                .sum()
            )
            table_stats.setdefault("cross_table", {})["missing_adj_factor_rows"] = missing_adj
            if missing_adj:
                issues.append(
                    QualityIssue(
                        "cross_table",
                        "warning",
                        f"price rows without matching adj_factor_raw rows: {missing_adj}",
                    )
                )

        requested_assets = tuple(
            str(asset)
            for asset in manifest_payload.get("request_params", {}).get("assets", [])
            if str(asset).strip()
        )
        if requested_assets and prices_raw is not None and "asset" in prices_raw.columns:
            actual_assets = set(prices_raw["asset"].astype(str))
            missing_assets = sorted(set(requested_assets) - actual_assets)
            table_stats.setdefault("coverage", {})["missing_requested_assets"] = missing_assets
            if missing_assets:
                issues.append(
                    QualityIssue(
                        "coverage",
                        "warning",
                        f"requested assets with no price rows: {missing_assets[:20]}",
                    )
                )

        report = RawSnapshotQualityReport(
            snapshot_id=snapshot_id,
            vendor=vendor,
            dataset_name=dataset_name,
            ok=not any(issue.severity == "error" for issue in issues),
            created_at_utc=_utc_now_text(),
            issues=tuple(issues),
            table_stats=table_stats,
            report_path=self.raw_validation_root / f"{snapshot_id}.json",
        )
        self._write_json(report.report_path, report.to_dict())
        return report

    def validate_core_dataset(self, *, dataset_name: str = CORE_DATASET_NAME) -> QualityReport:
        self.ensure_layout()
        issues: list[QualityIssue] = []
        summary: dict[str, object] = {}
        dataset_version = self.get_current_dataset_version(dataset_name)
        daily_research_only = bool(
            (dataset_version.notes if dataset_version is not None else {}).get(
                "daily_research_only"
            )
        )

        daily_bars = self.load_table("daily_bars")
        if daily_bars.empty:
            issues.append(QualityIssue("daily_bars", "error", "daily_bars is empty or missing"))
        else:
            try:
                validate_price_panel(
                    daily_bars.loc[
                        :,
                        [c for c in daily_bars.columns if c in self.REQUIRED_EXPORT_PRICE_COLUMNS],
                    ]
                )
            except ValueError as exc:
                issues.append(QualityIssue("daily_bars", "error", str(exc)))
            dup_count = int(daily_bars.duplicated(subset=["date", "asset"]).sum())
            if dup_count:
                issues.append(
                    QualityIssue(
                        "daily_bars", "error", f"duplicate (date, asset) rows detected: {dup_count}"
                    )
                )
            earliest_price_date = self._earliest_date_from_frame(daily_bars, "date")
            latest_price_date = self._latest_date_from_frame(daily_bars, "date")
            if earliest_price_date is not None and latest_price_date is not None:
                earliest_required = _subtract_years(latest_price_date, 3)
                earliest_target = _subtract_years(latest_price_date, 8)
                effective_required = self._resolve_open_boundary(
                    start_date=earliest_required,
                    end_date=latest_price_date,
                    side="left",
                )
                effective_target = self._resolve_open_boundary(
                    start_date=earliest_target,
                    end_date=latest_price_date,
                    side="left",
                )
                if earliest_price_date > effective_required:
                    issues.append(
                        QualityIssue(
                            "daily_bars",
                            "error",
                            (
                                "daily_bars coverage is below the minimum 3-year requirement: "
                                f"earliest={earliest_price_date}, latest={latest_price_date}, "
                                f"required_open_boundary={effective_required}"
                            ),
                        )
                    )
                elif earliest_price_date > effective_target:
                    issues.append(
                        QualityIssue(
                            "daily_bars",
                            "warning",
                            (
                                "daily_bars coverage does not yet reach the 8-year robust target: "
                                f"earliest={earliest_price_date}, latest={latest_price_date}, "
                                f"target_open_boundary={effective_target}"
                            ),
                        )
                    )
                summary["daily_bars_coverage"] = {
                    "earliest": earliest_price_date,
                    "latest": latest_price_date,
                }
            if {"date", "asset"} <= set(daily_bars.columns):
                asset_counts = (
                    daily_bars.groupby("date", sort=False)["asset"]
                    .nunique()
                    .sort_index(kind="mergesort")
                )
                summary["daily_asset_count_range"] = {
                    "min": int(asset_counts.min()),
                    "max": int(asset_counts.max()),
                }

        daily_basic = self.load_table("daily_basic")
        if daily_basic.empty:
            issues.append(QualityIssue("daily_basic", "warning", "daily_basic is empty or missing"))
        elif int(daily_basic.duplicated(subset=["date", "asset"]).sum()):
            issues.append(
                QualityIssue("daily_basic", "error", "daily_basic has duplicate (date, asset) rows")
            )

        trade_calendar = self.load_table("trade_calendar")
        if trade_calendar.empty:
            issues.append(
                QualityIssue("trade_calendar", "warning", "trade_calendar is empty or missing")
            )
        elif not daily_bars.empty and {"date", "is_open"} <= set(trade_calendar.columns):
            open_dates = set(
                pd.to_datetime(
                    trade_calendar.loc[
                        pd.to_numeric(trade_calendar["is_open"], errors="coerce").fillna(0) > 0,
                        "date",
                    ],
                    errors="coerce",
                )
                .dropna()
                .dt.strftime("%Y-%m-%d")
                .tolist()
            )
            price_dates = set(
                pd.to_datetime(daily_bars["date"], errors="coerce")
                .dropna()
                .dt.strftime("%Y-%m-%d")
                .tolist()
            )
            missing_open_dates = sorted(price_dates - open_dates)
            summary["trade_calendar_missing_open_dates"] = missing_open_dates[:20]
            if missing_open_dates:
                issues.append(
                    QualityIssue(
                        "trade_calendar",
                        "warning",
                        f"trade_calendar is missing {len(missing_open_dates)} open dates "
                        "observed in daily_bars",
                    )
                )

        asset_status = self.load_table("asset_status")
        if asset_status.empty:
            issues.append(
                QualityIssue("asset_status", "warning", "asset_status is empty or missing")
            )
        else:
            if int(asset_status.duplicated(subset=["date", "asset"]).sum()):
                issues.append(
                    QualityIssue(
                        "asset_status", "error", "asset_status has duplicate (date, asset) rows"
                    )
                )
            if not daily_bars.empty:
                price_keys = daily_bars[["date", "asset"]].drop_duplicates()
                status_keys = asset_status[["date", "asset"]].drop_duplicates()
                missing_status_rows = int(
                    price_keys.merge(status_keys, on=["date", "asset"], how="left", indicator=True)[
                        "_merge"
                    ]
                    .eq("left_only")
                    .sum()
                )
                summary["asset_status_missing_rows"] = missing_status_rows
                if missing_status_rows:
                    issues.append(
                        QualityIssue(
                            "asset_status",
                            "warning",
                            f"asset_status is missing for {missing_status_rows} price rows",
                        )
                    )

        index_membership = self.load_table("index_membership")
        if index_membership.empty:
            issues.append(
                QualityIssue("index_membership", "warning", "index_membership is empty or missing")
            )
        elif int(index_membership.duplicated(subset=["date", "index_code", "asset"]).sum()):
            issues.append(
                QualityIssue(
                    "index_membership",
                    "error",
                    "index_membership has duplicate (date, index_code, asset) rows",
                )
            )

        moneyflow = self.load_table("moneyflow")
        if moneyflow.empty:
            issues.append(QualityIssue("moneyflow", "warning", "moneyflow is empty or missing"))
        elif int(moneyflow.duplicated(subset=["date", "asset"]).sum()):
            issues.append(
                QualityIssue("moneyflow", "error", "moneyflow has duplicate (date, asset) rows")
            )

        liquidity_profile = self.load_table("liquidity_profile")
        if liquidity_profile.empty:
            issues.append(
                QualityIssue(
                    "liquidity_profile", "warning", "liquidity_profile is empty or missing"
                )
            )
        else:
            if int(liquidity_profile.duplicated(subset=["date", "asset"]).sum()):
                issues.append(
                    QualityIssue(
                        "liquidity_profile",
                        "error",
                        "liquidity_profile has duplicate (date, asset) rows",
                    )
                )
            if "liquidity_tier" in liquidity_profile.columns:
                invalid_tiers = int(
                    (
                        ~pd.to_numeric(liquidity_profile["liquidity_tier"], errors="coerce").isin(
                            [1, 2, 3, 4, 5]
                        )
                    ).sum()
                )
                if invalid_tiers:
                    issues.append(
                        QualityIssue(
                            "liquidity_profile",
                            "error",
                            f"invalid liquidity_tier rows detected: {invalid_tiers}",
                        )
                    )

        adj_factor = self.load_table("adj_factor")
        if adj_factor.empty:
            issues.append(QualityIssue("adj_factor", "warning", "adj_factor is empty or missing"))
        else:
            if adj_factor["adj_factor"].isna().any():
                issues.append(
                    QualityIssue("adj_factor", "error", "adj_factor contains null values")
                )
            if (pd.to_numeric(adj_factor["adj_factor"], errors="coerce") <= 0).any():
                issues.append(
                    QualityIssue("adj_factor", "error", "adj_factor contains non-positive values")
                )
            dup_count = int(adj_factor.duplicated(subset=["date", "asset"]).sum())
            if dup_count:
                issues.append(
                    QualityIssue(
                        "adj_factor", "error", f"duplicate (date, asset) rows detected: {dup_count}"
                    )
                )
            if not daily_bars.empty:
                price_keys = daily_bars[["date", "asset"]].drop_duplicates()
                adj_keys = adj_factor[["date", "asset"]].drop_duplicates()
                missing_adj_rows = int(
                    price_keys.merge(adj_keys, on=["date", "asset"], how="left", indicator=True)[
                        "_merge"
                    ]
                    .eq("left_only")
                    .sum()
                )
                summary["adj_factor_missing_rows"] = missing_adj_rows
                if missing_adj_rows:
                    issues.append(
                        QualityIssue(
                            "adj_factor",
                            "warning",
                            f"adj_factor is missing for {missing_adj_rows} price rows",
                        )
                    )

        financial_indicator = self.load_table("financial_indicator", date_field="ann_date")
        if financial_indicator.empty:
            if not daily_research_only:
                issues.append(
                    QualityIssue(
                        "financial_indicator", "warning", "financial_indicator is empty or missing"
                    )
                )
        else:
            if financial_indicator["ann_date"].isna().any():
                issues.append(
                    QualityIssue(
                        "financial_indicator", "error", "financial_indicator contains null ann_date"
                    )
                )
            dup_count = int(
                financial_indicator.duplicated(subset=["asset", "ann_date", "end_date"]).sum()
            )
            if dup_count:
                issues.append(
                    QualityIssue(
                        "financial_indicator",
                        "error",
                        f"duplicate (asset, ann_date, end_date) rows detected: {dup_count}",
                    )
                )

        for table_name in ("balance_sheet", "income_statement", "cash_flow_statement"):
            statement = self.load_table(table_name, date_field="ann_date")
            if statement.empty:
                if not daily_research_only:
                    issues.append(
                        QualityIssue(table_name, "warning", f"{table_name} is empty or missing")
                    )
                continue
            if statement["ann_date"].isna().any():
                issues.append(
                    QualityIssue(table_name, "error", f"{table_name} contains null ann_date")
                )
            dup_count = int(statement.duplicated(subset=["asset", "ann_date", "end_date"]).sum())
            if dup_count:
                issues.append(
                    QualityIssue(
                        table_name,
                        "error",
                        f"duplicate (asset, ann_date, end_date) rows detected: {dup_count}",
                    )
                )

        industry_classification = self.load_table(
            "industry_classification", date_field="snapshot_date"
        )
        if industry_classification.empty:
            issues.append(
                QualityIssue(
                    "industry_classification",
                    "warning",
                    "industry_classification is empty or missing",
                )
            )
        elif int(
            industry_classification.duplicated(
                subset=["snapshot_date", "industry_standard", "index_code"]
            ).sum()
        ):
            issues.append(
                QualityIssue(
                    "industry_classification",
                    "error",
                    "industry_classification has duplicate "
                    "(snapshot_date, industry_standard, index_code) rows",
                )
            )

        industry_membership = self.load_table("industry_membership", date_field="in_date")
        if industry_membership.empty:
            issues.append(
                QualityIssue(
                    "industry_membership", "warning", "industry_membership is empty or missing"
                )
            )
        elif int(
            industry_membership.duplicated(
                subset=["industry_standard", "asset", "l3_code", "in_date", "out_date"]
            ).sum()
        ):
            issues.append(
                QualityIssue(
                    "industry_membership",
                    "error",
                    "industry_membership has duplicate "
                    "(industry_standard, asset, l3_code, in_date, out_date) rows",
                )
            )

        report = QualityReport(
            dataset_name=dataset_name,
            ok=not any(issue.severity == "error" for issue in issues),
            created_at_utc=_utc_now_text(),
            checked_tables=_CORE_TABLES,
            issues=tuple(issues),
            report_path=self.canonical_validation_root
            / f"{dataset_name}_{_utc_now_text_for_filename()}.json",
            summary=summary,
        )
        self._write_json(report.report_path, report.to_dict())
        return report

    def export_case_inputs(
        self,
        *,
        slice_spec: SliceSpec,
        output_dir: str | Path,
        formats: Iterable[str] | None = None,
        prefer_cache: bool = True,
    ) -> CaseInputExportResult:
        self.ensure_layout()
        normalized_formats = _normalize_export_formats(formats)
        dataset_version = self.get_current_dataset_version(self.CORE_DATASET_NAME)
        dataset_version_id = dataset_version.version_id if dataset_version is not None else None
        cache_key = slice_spec.cache_key(dataset_version_id=dataset_version_id or "unknown")
        cache_dir = self.slices_cache_root / cache_key
        out_dir = Path(output_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        if not (
            prefer_cache
            and self._slice_cache_ready(cache_dir, dataset_version_id, normalized_formats)
        ):
            frames = self._build_case_slice_frames(slice_spec)
            self._write_slice_outputs(
                base_dir=cache_dir,
                frames=frames,
                slice_spec=slice_spec,
                dataset_version_id=dataset_version_id,
                cache_key=cache_key,
                formats=normalized_formats,
            )

        self._copy_slice_outputs(
            cache_dir=cache_dir, output_dir=out_dir, formats=normalized_formats
        )
        manifest = json.loads((cache_dir / "slice_manifest.json").read_text(encoding="utf-8"))
        row_counts = {str(key): int(value) for key, value in dict(manifest["row_counts"]).items()}
        format_output_paths = self._resolve_slice_output_paths(
            base_dir=out_dir,
            table_names=tuple(row_counts.keys()),
            formats=normalized_formats,
        )
        output_paths = _resolve_primary_output_paths(format_output_paths)

        return CaseInputExportResult(
            output_dir=out_dir,
            output_paths=output_paths,
            format_output_paths=format_output_paths,
            row_counts=row_counts,
            dataset_version_id=dataset_version_id,
            cache_key=cache_key,
            cache_dir=cache_dir,
            formats_written=normalized_formats,
        )

    def materialize_slice(
        self,
        *,
        slice_spec: SliceSpec,
        formats: Iterable[str] | None = None,
    ) -> CaseInputExportResult:
        dataset_version = self.get_current_dataset_version(self.CORE_DATASET_NAME)
        dataset_version_id = dataset_version.version_id if dataset_version is not None else None
        cache_key = slice_spec.cache_key(dataset_version_id=dataset_version_id or "unknown")
        return self.export_case_inputs(
            slice_spec=slice_spec,
            output_dir=self.slices_cache_root / cache_key,
            formats=formats,
            prefer_cache=True,
        )

    def load_case_slice(
        self,
        *,
        slice_spec: SliceSpec,
        prefer_cache: bool = True,
    ) -> CaseSliceBundle:
        dataset_version = self.get_current_dataset_version(self.CORE_DATASET_NAME)
        dataset_version_id = dataset_version.version_id if dataset_version is not None else None
        cache_key = slice_spec.cache_key(dataset_version_id=dataset_version_id or "unknown")
        cache_dir = self.slices_cache_root / cache_key
        if not (
            prefer_cache and self._slice_cache_ready(cache_dir, dataset_version_id, ("parquet",))
        ):
            self.export_case_inputs(
                slice_spec=slice_spec,
                output_dir=cache_dir,
                formats=("parquet",),
                prefer_cache=False,
            )
        manifest = json.loads((cache_dir / "slice_manifest.json").read_text(encoding="utf-8"))
        frames = {
            table_name: pd.read_parquet(cache_dir / f"{table_name}.parquet")
            for table_name in dict(manifest["row_counts"]).keys()
            if (cache_dir / f"{table_name}.parquet").exists()
        }
        return CaseSliceBundle(
            cache_dir=cache_dir,
            cache_key=cache_key,
            dataset_version_id=dataset_version_id,
            frames=frames,
        )

    def _build_case_slice_frames(self, slice_spec: SliceSpec) -> dict[str, pd.DataFrame]:
        normalized_adjustment = _normalize_adjustment(slice_spec.adjustment)
        assets = slice_spec.assets
        self._assert_daily_bars_coverage(
            start_date=slice_spec.start_date,
            end_date=slice_spec.end_date,
        )
        prices = self.load_table(
            "daily_bars",
            columns=self.RESEARCH_EXPORT_PRICE_COLUMNS,
            start_date=slice_spec.start_date,
            end_date=slice_spec.end_date,
            assets=assets,
        )
        if prices.empty:
            raise AlphaLabDataError(
                "No daily_bars rows matched the requested slice. "
                "Run 'alpha-lab data ingest tushare core ...' first."
            )

        prices = _ensure_export_price_columns(prices)
        prices = self._apply_universe(prices, universe_name=slice_spec.universe_name)
        if slice_spec.assets is not None:
            prices = prices[prices["asset"].isin(slice_spec.assets)].copy()
        if slice_spec.asset_limit is not None:
            selected_assets = sorted(prices["asset"].astype(str).unique().tolist())[
                : slice_spec.asset_limit
            ]
            prices = prices[prices["asset"].isin(selected_assets)].copy()
        if normalized_adjustment == "qfq":
            adj_factor = self.load_table(
                "adj_factor",
                columns=("date", "asset", "adj_factor"),
                start_date=slice_spec.start_date,
                end_date=slice_spec.end_date,
                assets=tuple(sorted(prices["asset"].astype(str).unique().tolist())),
            )
            prices = _apply_price_adjustment(
                prices=prices, adj_factor=adj_factor, adjustment=normalized_adjustment
            )

        slice_assets = tuple(sorted(prices["asset"].astype(str).unique().tolist()))
        asset_status = self.load_table(
            "asset_status",
            columns=("date", "asset", "is_suspended", "is_st"),
            start_date=slice_spec.start_date,
            end_date=slice_spec.end_date,
            assets=slice_assets,
        )
        prices = _merge_asset_status(prices=prices, asset_status=asset_status)
        index_membership = self.load_table(
            "index_membership",
            columns=("date", "index_code", "asset", "weight"),
            start_date=slice_spec.start_date,
            end_date=slice_spec.end_date,
            assets=slice_assets,
        )
        prices = _merge_index_membership_flags(prices=prices, index_membership=index_membership)
        prices = prices.sort_values(["date", "asset"], kind="mergesort").reset_index(drop=True)
        validate_price_panel(prices)

        universe = prices[["date", "asset"]].drop_duplicates().copy()
        universe["in_universe"] = 1
        universe = universe[["date", "asset", "in_universe"]]

        frames: dict[str, pd.DataFrame] = {
            "prices": prices,
            "universe": universe,
        }
        if not asset_status.empty:
            frames["asset_status"] = asset_status.sort_values(
                ["date", "asset"], kind="mergesort"
            ).reset_index(drop=True)
        if not index_membership.empty:
            frames["index_membership"] = index_membership.sort_values(
                ["date", "index_code", "asset"],
                kind="mergesort",
            ).reset_index(drop=True)

        requested_factors = set(slice_spec.factors)
        if "bp" in requested_factors:
            daily_basic = self.load_table(
                "daily_basic",
                columns=("date", "asset", "pb"),
                start_date=slice_spec.start_date,
                end_date=slice_spec.end_date,
                assets=slice_assets,
            )
            bp = _build_bp_from_daily_basic(daily_basic=daily_basic, prices=prices)
            validate_factor_output(bp)
            frames["bp"] = bp

        if "roe_ttm" in requested_factors:
            financial_indicator = self.load_table(
                "financial_indicator",
                columns=("asset", "ann_date", "end_date", "roe_value", "roe_source_column"),
                end_date=slice_spec.end_date,
                assets=slice_assets,
                date_field="ann_date",
            )
            roe = _build_roe_factor_strict(events=financial_indicator, prices=prices)
            validate_factor_output(roe)
            frames["roe_ttm"] = roe
        return frames

    def _write_slice_outputs(
        self,
        *,
        base_dir: Path,
        frames: dict[str, pd.DataFrame],
        slice_spec: SliceSpec,
        dataset_version_id: str | None,
        cache_key: str,
        formats: tuple[str, ...],
    ) -> None:
        base_dir.mkdir(parents=True, exist_ok=True)
        for table_name, frame in frames.items():
            if "csv" in formats:
                frame.to_csv(base_dir / f"{table_name}.csv", index=False)
            if "parquet" in formats:
                frame.to_parquet(
                    base_dir / f"{table_name}.parquet", index=False, compression="zstd"
                )
        self._write_json(
            base_dir / "slice_manifest.json",
            {
                "slice_spec": {
                    **asdict(slice_spec),
                    "adjustment": _normalize_adjustment(slice_spec.adjustment),
                },
                "dataset_version_id": dataset_version_id,
                "cache_key": cache_key,
                "formats_written": list(formats),
                "row_counts": {name: int(len(frame)) for name, frame in frames.items()},
            },
        )

    def _slice_cache_ready(
        self,
        cache_dir: Path,
        dataset_version_id: str | None,
        formats: tuple[str, ...],
    ) -> bool:
        manifest_path = cache_dir / "slice_manifest.json"
        if not manifest_path.exists():
            return False
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return False
        if manifest.get("dataset_version_id") != dataset_version_id:
            return False
        row_counts = dict(manifest.get("row_counts", {}))
        if not row_counts:
            return False
        for table_name in row_counts.keys():
            for output_format in formats:
                if not (cache_dir / f"{table_name}.{output_format}").exists():
                    return False
        return True

    def _copy_slice_outputs(
        self,
        *,
        cache_dir: Path,
        output_dir: Path,
        formats: tuple[str, ...],
    ) -> None:
        if cache_dir == output_dir:
            return
        manifest = json.loads((cache_dir / "slice_manifest.json").read_text(encoding="utf-8"))
        for table_name in dict(manifest["row_counts"]).keys():
            for output_format in formats:
                src = cache_dir / f"{table_name}.{output_format}"
                if src.exists():
                    shutil.copy2(src, output_dir / src.name)
        shutil.copy2(cache_dir / "slice_manifest.json", output_dir / "slice_manifest.json")

    def _resolve_slice_output_paths(
        self,
        *,
        base_dir: Path,
        table_names: tuple[str, ...],
        formats: tuple[str, ...],
    ) -> dict[str, dict[str, Path]]:
        return {
            table_name: {
                output_format: base_dir / f"{table_name}.{output_format}"
                for output_format in formats
                if (base_dir / f"{table_name}.{output_format}").exists()
            }
            for table_name in table_names
        }

    def query_sql(
        self,
        sql: str,
        *,
        refresh_catalog: bool = False,
    ) -> pd.DataFrame:
        """Execute one read-only SQL statement against DuckDB catalog views."""
        duckdb = _optional_duckdb()
        if duckdb is None:
            raise AlphaLabExperimentError("duckdb is not available in the current environment.")
        normalized_sql = _validate_read_only_sql(sql)
        if refresh_catalog or not self.duckdb_catalog_path.exists():
            self.refresh_duckdb_catalog()
        if not self.duckdb_catalog_path.exists():
            raise AlphaLabExperimentError(
                "DuckDB catalog is not initialized yet. Run 'alpha-lab data init' first."
            )

        conn = duckdb.connect(str(self.duckdb_catalog_path), read_only=True)
        try:
            return conn.execute(normalized_sql).fetchdf()
        finally:
            conn.close()

    def refresh_duckdb_catalog(
        self, table_names: list[str] | tuple[str, ...] | None = None
    ) -> None:
        duckdb = _optional_duckdb()
        if duckdb is None:
            self._write_json(
                self.metadata_root / "catalog" / "duckdb_status.json",
                {
                    "available": False,
                    "updated_at_utc": _utc_now_text(),
                },
            )
            return

        self.ensure_layout()
        names = (
            list(table_names)
            if table_names is not None
            else [path.name for path in self.canonical_root.iterdir() if path.is_dir()]
        )
        conn = duckdb.connect(str(self.duckdb_catalog_path))
        try:
            for name in names:
                table_root = self.table_root(name)
                if not table_root.exists():
                    continue
                parquet_glob = str(table_root / "**" / "*.parquet").replace("'", "''")
                conn.execute(
                    f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_parquet('{parquet_glob}')"
                )
        finally:
            conn.close()
        self._write_json(
            self.metadata_root / "catalog" / "duckdb_status.json",
            {
                "available": True,
                "updated_at_utc": _utc_now_text(),
                "catalog_path": str(self.duckdb_catalog_path),
            },
        )

    def _schedule_catalog_refresh(self, table_names: Iterable[str]) -> None:
        names = sorted({str(name) for name in table_names if str(name).strip()})
        if not names:
            return
        if self._refresh_defer_depth > 0:
            self._deferred_refresh_tables.update(names)
            return
        self.refresh_duckdb_catalog(names)

    def _flush_deferred_refresh(self) -> None:
        if not self._deferred_refresh_tables:
            return
        names = sorted(self._deferred_refresh_tables)
        self._deferred_refresh_tables.clear()
        self.refresh_duckdb_catalog(names)

    def _apply_universe(self, prices: pd.DataFrame, *, universe_name: str) -> pd.DataFrame:
        frame = prices.copy()
        if universe_name == "all_ashare":
            return frame
        top_liquid_size = _parse_top_liquid_universe_size(universe_name)
        if top_liquid_size is not None:
            return _apply_top_liquidity_universe(frame, n_assets=top_liquid_size)
        if universe_name == _INSTITUTIONAL_UNIVERSE_NAME:
            return self._apply_institutional_universe(frame)
        if universe_name == "listed_90d":
            return self._apply_listed_age_universe(frame, min_listed_days=90)
        raise AlphaLabDataError(
            "universe_name must be one of ['all_ashare', 'listed_90d', "
            "'institutional_ashare', 'top_liquid_300', 'top_liquid_500', "
            "'top_liquid_800']"
        )

    def _apply_listed_age_universe(
        self,
        prices: pd.DataFrame,
        *,
        min_listed_days: int,
        allow_missing_list_date: bool = True,
    ) -> pd.DataFrame:
        instruments = self.load_table(
            "instruments",
            columns=("asset", "list_date", "delist_date"),
            assets=tuple(sorted(prices["asset"].astype(str).unique().tolist())),
            date_field=None,
        )
        return _apply_listed_age_filter(
            prices,
            instruments=instruments,
            min_listed_days=min_listed_days,
            allow_missing_list_date=allow_missing_list_date,
        )

    def _apply_institutional_universe(self, prices: pd.DataFrame) -> pd.DataFrame:
        frame = self._apply_listed_age_universe(
            prices,
            min_listed_days=_INSTITUTIONAL_MIN_LISTED_DAYS,
            allow_missing_list_date=False,
        )
        if frame.empty:
            return frame

        date_values = pd.to_datetime(frame["date"], errors="coerce").dropna()
        if date_values.empty:
            return frame.iloc[0:0].copy()
        asset_status = self.load_table(
            "asset_status",
            columns=("date", "asset", "is_suspended", "is_st"),
            start_date=str(date_values.min().strftime("%Y-%m-%d")),
            end_date=str(date_values.max().strftime("%Y-%m-%d")),
            assets=tuple(sorted(frame["asset"].astype(str).unique().tolist())),
        )
        frame = _merge_asset_status(prices=frame, asset_status=asset_status)
        frame = _apply_institutional_liquidity_universe(frame)
        return frame.loc[:, list(prices.columns)].copy()

    def _assert_daily_bars_coverage(self, *, start_date: str, end_date: str) -> None:
        bounds = self.load_table("daily_bars", columns=("date",))
        if bounds.empty:
            return
        earliest = self._earliest_date_from_frame(bounds, "date")
        latest = self._latest_date_from_frame(bounds, "date")
        if earliest is None or latest is None:
            return
        effective_start = self._resolve_open_boundary(
            start_date=start_date, end_date=end_date, side="left"
        )
        effective_end = self._resolve_open_boundary(
            start_date=start_date, end_date=end_date, side="right"
        )
        if earliest > effective_start:
            raise AlphaLabDataError(
                "Requested slice start_date predates available daily_bars coverage: "
                f"requested_start={start_date}, effective_start={effective_start}, "
                f"earliest_available={earliest}"
            )
        if latest < effective_end:
            raise AlphaLabDataError(
                "Requested slice end_date exceeds available daily_bars coverage: "
                f"requested_end={end_date}, effective_end={effective_end}, "
                f"latest_available={latest}"
            )

    @staticmethod
    def _earliest_date_from_frame(frame: pd.DataFrame, date_field: str) -> str | None:
        values = pd.to_datetime(frame[date_field], errors="coerce").dropna()
        if values.empty:
            return None
        return str(values.min().strftime("%Y-%m-%d"))

    @staticmethod
    def _latest_date_from_frame(frame: pd.DataFrame, date_field: str) -> str | None:
        values = pd.to_datetime(frame[date_field], errors="coerce").dropna()
        if values.empty:
            return None
        return str(values.max().strftime("%Y-%m-%d"))

    def _resolve_open_boundary(self, *, start_date: str, end_date: str, side: str) -> str:
        calendar = self.load_table("trade_calendar", columns=("date", "is_open"))
        if calendar.empty:
            return start_date if side == "left" else end_date
        dates = pd.to_datetime(calendar["date"], errors="coerce")
        is_open = pd.to_numeric(calendar.get("is_open"), errors="coerce").fillna(1) > 0
        open_dates = dates[is_open].dropna().sort_values(kind="mergesort")
        if open_dates.empty:
            return start_date if side == "left" else end_date
        if side == "left":
            matches = open_dates[open_dates >= pd.Timestamp(start_date)]
            if matches.empty:
                return start_date
            return str(matches.iloc[0].strftime("%Y-%m-%d"))
        matches = open_dates[open_dates <= pd.Timestamp(end_date)]
        if matches.empty:
            return end_date
        return str(matches.iloc[-1].strftime("%Y-%m-%d"))

    def _read_partition(self, partition_dir: Path) -> pd.DataFrame:
        parquet_files = sorted(partition_dir.glob("*.parquet"))
        if not parquet_files:
            return pd.DataFrame()
        return pd.read_parquet(parquet_files[0])

    def _write_partition(self, partition_dir: Path, frame: pd.DataFrame) -> None:
        tmp_dir = partition_dir.parent / f".tmp_{partition_dir.name}"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(tmp_dir / "part-00000.parquet", index=False, compression="zstd")
        if partition_dir.exists():
            shutil.rmtree(partition_dir)
        tmp_dir.rename(partition_dir)

    def _table_fingerprint(self, table_name: str) -> str:
        table_root = self.table_root(table_name)
        if not table_root.exists():
            return "missing"
        hashes = {
            str(path.relative_to(table_root)): _sha256_file(path)
            for path in sorted(table_root.rglob("*.parquet"))
        }
        return hashlib.sha256(
            json.dumps(hashes, sort_keys=True, ensure_ascii=True).encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _write_json(path: Path, payload: dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _build_bp_from_daily_basic(*, daily_basic: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    if daily_basic.empty:
        raise AlphaLabDataError("Cannot export bp.csv because daily_basic is empty.")
    frame = daily_basic.copy()
    frame["pb"] = pd.to_numeric(frame["pb"], errors="coerce")
    frame = frame[frame["pb"] > 0].copy()
    frame["factor"] = "bp"
    frame["value"] = 1.0 / frame["pb"]
    factor = frame[["date", "asset", "factor", "value"]]
    price_keys = prices[["date", "asset"]].drop_duplicates()
    factor = factor.merge(price_keys, on=["date", "asset"], how="inner", validate="many_to_one")
    factor = factor.drop_duplicates(subset=["date", "asset", "factor"], keep="last")
    factor = factor.sort_values(["date", "asset"], kind="mergesort").reset_index(drop=True)
    return factor


def _build_roe_factor_strict(*, events: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        raise AlphaLabDataError("Cannot export roe_ttm.csv because financial_indicator is empty.")
    frame = events.copy()
    frame["ann_date"] = pd.to_datetime(frame["ann_date"], errors="coerce")
    frame["end_date"] = pd.to_datetime(frame["end_date"], errors="coerce")
    frame["roe_value"] = pd.to_numeric(frame["roe_value"], errors="coerce")
    frame = frame.dropna(subset=["asset", "ann_date", "roe_value"]).copy()
    frame = frame.sort_values(["asset", "ann_date", "end_date"], kind="mergesort")
    frame = frame.drop_duplicates(subset=["asset", "ann_date"], keep="last")
    if frame.empty:
        raise AlphaLabDataError(
            "No valid ann_date-backed financial_indicator rows available for roe_ttm export."
        )

    price_keys = prices[["date", "asset"]].drop_duplicates().copy()
    price_keys["date"] = pd.to_datetime(price_keys["date"], errors="coerce")
    aligned_parts: list[pd.DataFrame] = []
    for asset, px in price_keys.groupby("asset"):
        asset_events = frame[frame["asset"] == asset][["ann_date", "roe_value"]].copy()
        if asset_events.empty:
            continue
        aligned = pd.merge_asof(
            px.sort_values("date", kind="mergesort"),
            asset_events.sort_values("ann_date", kind="mergesort"),
            left_on="date",
            right_on="ann_date",
            direction="backward",
            allow_exact_matches=False,
        )
        aligned["asset"] = asset
        aligned_parts.append(aligned[["date", "asset", "roe_value"]])

    if not aligned_parts:
        raise AlphaLabDataError("No ROE rows could be aligned to the requested slice.")
    factor = pd.concat(aligned_parts, ignore_index=True)
    factor = factor.dropna(subset=["roe_value"]).copy()
    factor["date"] = pd.to_datetime(factor["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    factor["factor"] = "roe_ttm"
    factor = factor.rename(columns={"roe_value": "value"})
    factor = factor[["date", "asset", "factor", "value"]]
    factor = factor.drop_duplicates(subset=["date", "asset", "factor"], keep="last")
    factor = factor.sort_values(["date", "asset"], kind="mergesort").reset_index(drop=True)
    return factor


def _ensure_export_price_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    defaults: dict[str, object] = {
        "open": pd.NA,
        "high": pd.NA,
        "low": pd.NA,
        "pre_close": pd.NA,
        "volume": pd.NA,
        "amount": pd.NA,
        "vwap": pd.NA,
        "turnover_rate": pd.NA,
        "up_limit": pd.NA,
        "down_limit": pd.NA,
        "is_limit_up": 0,
        "is_limit_down": 0,
        "is_suspended": 0,
        "is_st": 0,
        "is_hs300": 0,
        "is_zz500": 0,
        "is_zz1000": 0,
        "is_sz50": 0,
    }
    for column, default in defaults.items():
        if column not in out.columns:
            out[column] = default
    missing = {"date", "asset", "close"} - set(out.columns)
    if missing:
        raise AlphaLabDataError(f"daily_bars is missing required export columns: {sorted(missing)}")
    return out.loc[:, list(DataCatalog.RESEARCH_EXPORT_PRICE_COLUMNS)].copy()


def _normalize_adjustment(adjustment: str) -> str:
    normalized = str(adjustment or "").strip().lower()
    if normalized in {"", "none", "raw"}:
        return "raw"
    if normalized == "qfq":
        return "qfq"
    raise AlphaLabDataError("adjustment must be one of ['raw', 'qfq']")


def _apply_price_adjustment(
    *,
    prices: pd.DataFrame,
    adj_factor: pd.DataFrame,
    adjustment: str,
) -> pd.DataFrame:
    normalized = _normalize_adjustment(adjustment)
    if normalized == "raw":
        return prices.copy()
    if adj_factor.empty:
        raise AlphaLabDataError("Cannot export qfq prices because adj_factor is empty.")

    factor = adj_factor.copy()
    required = {"date", "asset", "adj_factor"}
    missing = required - set(factor.columns)
    if missing:
        raise AlphaLabDataError(f"adj_factor is missing required columns: {sorted(missing)}")
    factor["adj_factor"] = pd.to_numeric(factor["adj_factor"], errors="coerce")
    factor = factor.dropna(subset=["date", "asset", "adj_factor"]).copy()
    factor = factor.sort_values(["date", "asset"], kind="mergesort").drop_duplicates(
        subset=["date", "asset"],
        keep="last",
    )

    adjusted = prices.merge(
        factor[["date", "asset", "adj_factor"]],
        on=["date", "asset"],
        how="left",
        validate="one_to_one",
    )
    missing_rows = int(adjusted["adj_factor"].isna().sum())
    if missing_rows:
        raise AlphaLabDataError(
            f"adj_factor is missing for {missing_rows} price rows; cannot export qfq prices."
        )

    adjusted = adjusted.sort_values(["asset", "date"], kind="mergesort").reset_index(drop=True)
    adjusted["adj_factor"] = pd.to_numeric(adjusted["adj_factor"], errors="coerce")
    latest_factor = adjusted.groupby("asset", sort=False)["adj_factor"].transform("last")
    if latest_factor.isna().any() or (latest_factor <= 0).any():
        raise AlphaLabDataError(
            "adj_factor contains invalid latest factors; cannot export qfq prices."
        )

    adjusted["close"] = pd.to_numeric(adjusted["close"], errors="coerce") * (
        adjusted["adj_factor"] / latest_factor
    )
    ratio = adjusted["adj_factor"] / latest_factor
    for column in ("open", "high", "low", "pre_close", "vwap", "up_limit", "down_limit"):
        if column in adjusted.columns:
            adjusted[column] = pd.to_numeric(adjusted[column], errors="coerce") * ratio
    adjusted = adjusted.drop(columns="adj_factor")
    return _ensure_export_price_columns(adjusted)


def _parse_top_liquid_universe_size(universe_name: str) -> int | None:
    match = _TOP_LIQUID_UNIVERSE_RE.match(str(universe_name).strip())
    if match is None:
        return None
    size = int(match.group(1))
    if size <= 0:
        raise AlphaLabDataError(f"top_liquid universe size must be positive: {universe_name!r}")
    return size


def _apply_listed_age_filter(
    prices: pd.DataFrame,
    *,
    instruments: pd.DataFrame,
    min_listed_days: int,
    allow_missing_list_date: bool,
) -> pd.DataFrame:
    if prices.empty or instruments.empty:
        return prices.copy()
    frame = prices.copy()
    instrument_frame = instruments.copy()
    instrument_frame["list_date"] = pd.to_datetime(
        instrument_frame["list_date"], errors="coerce"
    )
    instrument_frame["delist_date"] = pd.to_datetime(
        instrument_frame["delist_date"], errors="coerce"
    )
    enriched = frame.merge(instrument_frame, on="asset", how="left", validate="many_to_one")
    enriched["date_ts"] = pd.to_datetime(enriched["date"], errors="coerce")
    enriched["min_live_date"] = enriched["list_date"] + timedelta(days=min_listed_days)
    if allow_missing_list_date:
        keep = enriched["list_date"].isna() | (enriched["date_ts"] >= enriched["min_live_date"])
    else:
        keep = enriched["list_date"].notna() & (enriched["date_ts"] >= enriched["min_live_date"])
    if "delist_date" in enriched:
        keep &= enriched["delist_date"].isna() | (enriched["date_ts"] <= enriched["delist_date"])
    return enriched.loc[keep, list(frame.columns)].copy()


def _apply_institutional_liquidity_universe(
    prices: pd.DataFrame,
    *,
    lookback_days: int = _INSTITUTIONAL_LIQUIDITY_LOOKBACK_DAYS,
    min_active_days: int = _INSTITUTIONAL_MIN_ACTIVE_DAYS,
    min_avg_amount: float = _INSTITUTIONAL_MIN_AVG_AMOUNT,
    min_amount_percentile: float = _INSTITUTIONAL_MIN_AMOUNT_PERCENTILE,
) -> pd.DataFrame:
    if prices.empty:
        return prices.copy()
    if lookback_days <= 0 or min_active_days <= 0:
        raise AlphaLabDataError(
            "institutional universe lookback and active-day thresholds must be positive"
        )
    if not 0 <= min_amount_percentile < 1:
        raise AlphaLabDataError("institutional universe amount percentile must be in [0, 1)")

    frame = prices.copy()
    original_columns = list(prices.columns)
    frame["date_ts"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["amount_numeric"] = pd.to_numeric(frame["amount"], errors="coerce")
    frame["is_suspended_numeric"] = (
        pd.to_numeric(frame.get("is_suspended"), errors="coerce").fillna(0).astype(int)
    )
    frame["is_st_numeric"] = (
        pd.to_numeric(frame.get("is_st"), errors="coerce").fillna(0).astype(int)
    )
    current_tradable = (
        frame["date_ts"].notna()
        & frame["amount_numeric"].notna()
        & (frame["amount_numeric"] > 0)
        & (frame["is_suspended_numeric"] == 0)
        & (frame["is_st_numeric"] == 0)
    )
    frame["current_tradable"] = current_tradable
    frame["active_amount"] = frame["amount_numeric"].where(current_tradable)
    frame["active_day"] = current_tradable.astype(int)
    frame = frame.sort_values(["asset", "date_ts", "date"], kind="mergesort").reset_index(
        drop=True
    )
    grouped = frame.groupby("asset", sort=False)
    frame["avg_amount_lookback"] = (
        grouped["active_amount"]
        .rolling(window=lookback_days, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    frame["active_days_lookback"] = (
        grouped["active_day"]
        .rolling(window=lookback_days, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )
    frame["observed_days_lookback"] = (
        grouped["active_day"]
        .rolling(window=lookback_days, min_periods=1)
        .count()
        .reset_index(level=0, drop=True)
    )
    required_active_days = frame["observed_days_lookback"].clip(upper=min_active_days)
    frame["floor_amount_candidate"] = frame["avg_amount_lookback"].where(
        frame["current_tradable"]
    )
    amount_floor_by_date = frame.groupby("date", sort=False)["floor_amount_candidate"].transform(
        lambda values: values.quantile(min_amount_percentile)
    )
    keep = (
        frame["current_tradable"]
        & (frame["active_days_lookback"] >= required_active_days)
        & (frame["avg_amount_lookback"] >= min_avg_amount)
        & (frame["avg_amount_lookback"] >= amount_floor_by_date.fillna(min_avg_amount))
    )
    return frame.loc[keep, original_columns].copy()


def _apply_top_liquidity_universe(
    prices: pd.DataFrame,
    *,
    n_assets: int,
    lookback_days: int = 60,
) -> pd.DataFrame:
    if prices.empty:
        return prices.copy()
    ranked = prices.copy()
    ranked["date_ts"] = pd.to_datetime(ranked["date"], errors="coerce")
    ranked["amount_numeric"] = pd.to_numeric(ranked["amount"], errors="coerce")
    ranked = ranked.sort_values(["asset", "date_ts", "date"], kind="mergesort").reset_index(
        drop=True
    )
    ranked["avg_amount_lookback"] = (
        ranked.groupby("asset", sort=False)["amount_numeric"]
        .rolling(window=lookback_days, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    ranked = ranked.sort_values(
        ["date_ts", "avg_amount_lookback", "asset"],
        ascending=[True, False, True],
        kind="mergesort",
        na_position="last",
    ).reset_index(drop=True)
    ranked["liquidity_rank"] = ranked.groupby("date", sort=False).cumcount() + 1
    filtered = ranked[ranked["liquidity_rank"] <= n_assets].copy()
    return filtered.loc[:, list(prices.columns)]


def _merge_asset_status(*, prices: pd.DataFrame, asset_status: pd.DataFrame) -> pd.DataFrame:
    frame = prices.copy()
    if asset_status.empty:
        frame["is_suspended"] = 0
        frame["is_st"] = 0
        return _ensure_export_price_columns(frame)

    status = asset_status.copy()
    status["is_suspended"] = (
        pd.to_numeric(status["is_suspended"], errors="coerce").fillna(0).astype(int)
    )
    status["is_st"] = pd.to_numeric(status["is_st"], errors="coerce").fillna(0).astype(int)
    merged = frame.merge(
        status[["date", "asset", "is_suspended", "is_st"]].drop_duplicates(
            subset=["date", "asset"], keep="last"
        ),
        on=["date", "asset"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_status"),
    )
    merged["is_suspended"] = (
        merged["is_suspended_status"].fillna(merged["is_suspended"]).fillna(0).astype(int)
    )
    merged["is_st"] = merged["is_st_status"].fillna(merged["is_st"]).fillna(0).astype(int)
    merged = merged.drop(columns=["is_suspended_status", "is_st_status"])
    return _ensure_export_price_columns(merged)


def _merge_index_membership_flags(
    *, prices: pd.DataFrame, index_membership: pd.DataFrame
) -> pd.DataFrame:
    frame = prices.copy()
    for flag in _INDEX_MEMBERSHIP_FLAG_COLUMNS.values():
        if flag not in frame.columns:
            frame[flag] = 0
    if index_membership.empty:
        return _ensure_export_price_columns(frame)

    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    unique_dates = (
        frame[["date"]]
        .drop_duplicates()
        .sort_values("date", kind="mergesort")
        .reset_index(drop=True)
    )
    membership = index_membership.copy()
    membership["date"] = pd.to_datetime(membership["date"], errors="coerce")
    membership = membership.dropna(subset=["date", "index_code", "asset"]).copy()
    membership = membership.sort_values(["index_code", "date", "asset"], kind="mergesort")

    for index_code, flag in _INDEX_MEMBERSHIP_FLAG_COLUMNS.items():
        subset = membership[membership["index_code"] == index_code][["date", "asset"]].copy()
        if subset.empty:
            continue
        snapshot_dates = (
            subset[["date"]]
            .drop_duplicates()
            .rename(columns={"date": "snapshot_date"})
            .sort_values("snapshot_date", kind="mergesort")
        )
        resolved = pd.merge_asof(
            unique_dates,
            snapshot_dates,
            left_on="date",
            right_on="snapshot_date",
            direction="backward",
            allow_exact_matches=True,
        )
        members = subset.rename(columns={"date": "snapshot_date"}).copy()
        temp_flag = f"{flag}__member"
        members[temp_flag] = 1
        frame = frame.merge(resolved, on="date", how="left", validate="many_to_one")
        frame = frame.merge(
            members,
            on=["snapshot_date", "asset"],
            how="left",
            validate="many_to_one",
        )
        frame[flag] = frame[temp_flag].fillna(frame[flag]).fillna(0).astype(int)
        frame = frame.drop(columns=[temp_flag])
        frame = frame.drop(columns=["snapshot_date"])

    frame["date"] = frame["date"].dt.strftime("%Y-%m-%d")
    return _ensure_export_price_columns(frame)


def _raw_table_key_columns(table_name: str) -> tuple[str, ...]:
    return {
        "prices_raw": ("date", "asset"),
        "adj_factor_raw": ("date", "asset"),
        "stk_limit_raw": ("date", "asset"),
        "suspend_status_raw": ("date", "asset"),
        "st_name_events_raw": ("asset", "start_date", "end_date"),
        "index_membership_raw": ("date", "index_code", "asset"),
        "moneyflow_raw": ("date", "asset"),
        "industry_classification_raw": ("snapshot_date", "industry_standard", "index_code"),
        "industry_membership_raw": ("industry_standard", "asset", "l3_code", "in_date", "out_date"),
        "pb_raw": ("date", "asset"),
        "roe_raw": ("asset", "ann_date", "end_date"),
        "balance_sheet_raw": ("asset", "ann_date", "end_date"),
        "income_statement_raw": ("asset", "ann_date", "end_date"),
        "cash_flow_statement_raw": ("asset", "ann_date", "end_date"),
        "trade_calendar": ("date",),
        "instruments": ("asset",),
    }.get(table_name, ())


def _raw_table_required_columns(table_name: str) -> tuple[str, ...]:
    return {
        "prices_raw": ("date", "asset", "close"),
        "adj_factor_raw": ("date", "asset", "adj_factor"),
        "pb_raw": ("date", "asset", "pb"),
        "roe_raw": ("asset", "ann_date", "end_date", "roe_value"),
        "moneyflow_raw": ("date", "asset", "net_mf_amount"),
        "industry_classification_raw": ("snapshot_date", "industry_standard", "index_code"),
        "industry_membership_raw": ("industry_standard", "asset", "l3_code", "in_date"),
    }.get(table_name, ())


def _count_null_key_rows(frame: pd.DataFrame) -> int:
    if frame.empty:
        return 0
    key_like_columns = [
        column for column in ("date", "asset", "ann_date", "end_date") if column in frame.columns
    ]
    if not key_like_columns:
        return 0
    return int(frame[key_like_columns].isna().any(axis=1).sum())


def _subtract_years(date_text: str, years: int) -> str:
    value = pd.Timestamp(date_text)
    try:
        shifted = value.replace(year=value.year - years)
    except ValueError:
        shifted = value - pd.DateOffset(years=years)
    return str(pd.Timestamp(shifted).strftime("%Y-%m-%d"))


def _normalize_export_formats(formats: Iterable[str] | None) -> tuple[str, ...]:
    if formats is None:
        return ("csv", "parquet")
    resolved: list[str] = []
    for output_format in formats:
        normalized = str(output_format).strip().lower()
        if normalized == "both":
            resolved.extend(["csv", "parquet"])
            continue
        if normalized not in {"csv", "parquet"}:
            raise AlphaLabConfigError("export format must be one of ['csv', 'parquet', 'both']")
        resolved.append(normalized)
    deduped = tuple(dict.fromkeys(resolved))
    if not deduped:
        raise AlphaLabConfigError("at least one export format is required")
    return deduped


def _resolve_primary_output_paths(
    format_output_paths: dict[str, dict[str, Path]],
) -> dict[str, Path]:
    resolved: dict[str, Path] = {}
    for table_name, formats in format_output_paths.items():
        if "csv" in formats:
            resolved[table_name] = formats["csv"]
        elif "parquet" in formats:
            resolved[table_name] = formats["parquet"]
    return resolved


def _partition_year_values(frame: pd.DataFrame, partition_column: str) -> pd.Series:
    values = pd.to_datetime(frame[partition_column], errors="coerce")
    if values.isna().any():
        raise AlphaLabDataError(
            f"{partition_column} contains unparseable values; cannot partition dataset."
        )
    return values.dt.year.astype(str)


def _partition_month_values(frame: pd.DataFrame, partition_column: str) -> pd.Series:
    values = pd.to_datetime(frame[partition_column], errors="coerce")
    if values.isna().any():
        raise AlphaLabDataError(
            f"{partition_column} contains unparseable values; cannot partition dataset."
        )
    return values.dt.month.map(lambda value: f"{int(value):02d}")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _utc_now_text() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _utc_now_text_for_filename() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _optional_duckdb() -> Any | None:
    try:
        import duckdb  # type: ignore[import-not-found]
    except ImportError:
        return None
    return duckdb


_READ_ONLY_SQL_PREFIXES: frozenset[str] = frozenset(
    {"select", "with", "show", "describe", "pragma", "explain"}
)
_FORBIDDEN_SQL_KEYWORDS: tuple[str, ...] = (
    "insert",
    "update",
    "delete",
    "create",
    "replace",
    "alter",
    "drop",
    "attach",
    "detach",
    "copy",
    "call",
    "merge",
    "truncate",
)


def _validate_read_only_sql(sql: str) -> str:
    normalized = str(sql or "").strip()
    if not normalized:
        raise AlphaLabConfigError("sql must be non-empty")

    statements = [part.strip() for part in normalized.split(";") if part.strip()]
    if len(statements) != 1:
        raise AlphaLabConfigError("Only one SQL statement is allowed per query.")
    statement = statements[0]
    first_token = statement.split(maxsplit=1)[0].lower()
    if first_token not in _READ_ONLY_SQL_PREFIXES:
        raise AlphaLabConfigError(
            "Only read-only SQL is allowed. Start the query with SELECT, WITH, "
            "SHOW, DESCRIBE, PRAGMA, or EXPLAIN."
        )

    lowered = f" {statement.lower()} "
    for keyword in _FORBIDDEN_SQL_KEYWORDS:
        if re.search(rf"\b{re.escape(keyword)}\b", lowered):
            raise AlphaLabConfigError(
                f"Forbidden SQL keyword detected in read-only query: {keyword}"
            )
    return statement
