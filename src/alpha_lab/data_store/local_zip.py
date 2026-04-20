from __future__ import annotations

import shutil
import tempfile
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from alpha_lab.exceptions import AlphaLabConfigError, AlphaLabDataError

from .catalog import DataCatalog, DatasetVersion

_CSV_USECOLS: tuple[str, ...] = (
    "ts_code",
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "pre_close",
    "change",
    "pct_chg",
    "vol",
    "amount",
    "adj_factor",
    "adj_close",
    "adj_open",
    "adj_high",
    "adj_low",
)
_WRITTEN_TABLES: tuple[str, ...] = (
    "daily_bars",
    "adj_factor",
    "asset_status",
    "trade_calendar",
    "instruments",
)


@dataclass(frozen=True)
class LocalZipAshareDailyIngestResult:
    dataset_version: DatasetVersion
    written_tables: tuple[str, ...]
    row_counts: dict[str, int]
    asset_count: int
    date_range: dict[str, str | None]
    notes: dict[str, object]


@dataclass(frozen=True)
class LocalZipAshareDailyOrganizeResult:
    dataset_version: DatasetVersion
    rewritten_tables: tuple[str, ...]
    row_counts: dict[str, int]
    notes: dict[str, object]


@dataclass(frozen=True)
class _AssetFrames:
    asset: str
    daily_bars: pd.DataFrame
    adj_factor: pd.DataFrame
    asset_status: pd.DataFrame
    list_date: str
    last_date: str
    filled_adj_factor_rows: int
    inferred_adj_factor_mode: str


class LocalZipAshareDailyIngestor:
    """Ingest nested ZIP archives of per-asset A-share daily CSV files."""

    def __init__(self, catalog: DataCatalog | None = None) -> None:
        self.catalog = catalog or DataCatalog()

    def ingest_daily_zip(
        self,
        *,
        zip_path: str | Path,
        start_date: str | None = None,
        end_date: str | None = None,
        progress_callback: Callable[[str], None] | None = None,
    ) -> LocalZipAshareDailyIngestResult:
        self.catalog.ensure_layout()
        source_path = Path(zip_path).expanduser().resolve()
        if not source_path.exists() or not source_path.is_file():
            raise AlphaLabDataError(f"zip_path does not exist: {source_path}")
        start_ts, end_ts = _resolve_date_window(start_date=start_date, end_date=end_date)

        for table_name in _WRITTEN_TABLES:
            table_root = self.catalog.table_root(table_name)
            if table_root.exists():
                shutil.rmtree(table_root)

        trade_dates: set[str] = set()
        instruments_rows: list[dict[str, object]] = []
        seen_assets: set[str] = set()
        row_counts = {
            "daily_bars": 0,
            "adj_factor": 0,
            "asset_status": 0,
            "trade_calendar": 0,
            "instruments": 0,
        }
        filled_adj_factor_rows = 0
        all_missing_adj_factor_assets = 0

        with tempfile.TemporaryDirectory(prefix="alpha_lab_local_zip_") as tmp_dir_text:
            tmp_dir = Path(tmp_dir_text)
            archive_path, archive_label = _prepare_member_archive(source_path, tmp_dir=tmp_dir)
            if progress_callback is not None:
                progress_callback(f"opened archive source: {archive_label}")
            with zipfile.ZipFile(archive_path) as archive:
                member_names = sorted(
                    info.filename
                    for info in archive.infolist()
                    if not info.is_dir() and info.filename.lower().endswith(".csv")
                )
                if not member_names:
                    raise AlphaLabDataError(
                        f"archive does not contain per-asset CSV files: {source_path}"
                    )
                total_members = len(member_names)
                for idx, member_name in enumerate(member_names, start=1):
                    if progress_callback is not None and (
                        idx == 1 or idx == total_members or idx % 250 == 0
                    ):
                        progress_callback(
                            f"processing asset file {idx}/{total_members}: {Path(member_name).name}"
                        )
                    with archive.open(member_name) as raw:
                        frame = pd.read_csv(
                            raw,
                            usecols=lambda col: col in _CSV_USECOLS,
                            dtype={"ts_code": "string", "trade_date": "string"},
                        )
                    asset_frames = _canonicalize_asset_frame(
                        frame,
                        member_name=member_name,
                        start_date=start_ts,
                        end_date=end_ts,
                    )
                    if asset_frames is None:
                        continue
                    if asset_frames.asset in seen_assets:
                        raise AlphaLabDataError(
                            f"duplicate asset encountered in archive: {asset_frames.asset}"
                        )
                    seen_assets.add(asset_frames.asset)
                    trade_dates.update(asset_frames.daily_bars["date"].tolist())
                    instruments_rows.append(
                        {
                            "asset": asset_frames.asset,
                            "symbol": asset_frames.asset.split(".", 1)[0],
                            "name": pd.NA,
                            "area": pd.NA,
                            "industry": pd.NA,
                            "market": asset_frames.asset.split(".", 1)[1],
                            "list_date": asset_frames.list_date,
                            "delist_date": pd.NA,
                        }
                    )
                    if asset_frames.inferred_adj_factor_mode == "default_1.0_all_missing":
                        all_missing_adj_factor_assets += 1
                    filled_adj_factor_rows += asset_frames.filled_adj_factor_rows
                    row_counts["daily_bars"] += int(len(asset_frames.daily_bars))
                    row_counts["adj_factor"] += int(len(asset_frames.adj_factor))
                    row_counts["asset_status"] += int(len(asset_frames.asset_status))
                    _write_asset_partition(
                        table_root=self.catalog.table_root("daily_bars"),
                        asset=asset_frames.asset,
                        frame=asset_frames.daily_bars,
                    )
                    _write_asset_partition(
                        table_root=self.catalog.table_root("adj_factor"),
                        asset=asset_frames.asset,
                        frame=asset_frames.adj_factor,
                    )
                    _write_asset_partition(
                        table_root=self.catalog.table_root("asset_status"),
                        asset=asset_frames.asset,
                        frame=asset_frames.asset_status,
                    )

        trade_calendar = pd.DataFrame(
            {
                "date": sorted(trade_dates),
                "exchange": "CN_A",
                "is_open": 1,
            }
        )
        instruments = (
            pd.DataFrame(instruments_rows)
            .sort_values(["asset"], kind="mergesort")
            .reset_index(drop=True)
        )

        row_counts["trade_calendar"] = int(len(trade_calendar))
        row_counts["instruments"] = int(len(instruments))
        _write_single_file_table(
            table_root=self.catalog.table_root("trade_calendar"),
            frame=trade_calendar,
            sort_cols=("date",),
        )
        _write_single_file_table(
            table_root=self.catalog.table_root("instruments"),
            frame=instruments,
            sort_cols=("asset",),
        )
        self.catalog.refresh_duckdb_catalog(list(_WRITTEN_TABLES))

        date_range = {
            "start_date": trade_calendar["date"].iloc[0] if not trade_calendar.empty else None,
            "end_date": trade_calendar["date"].iloc[-1] if not trade_calendar.empty else None,
        }
        notes = {
            "source_vendor": "local_zip",
            "source_path": str(source_path),
            "asset_partitioned_tables": ["daily_bars", "adj_factor", "asset_status"],
            "filled_adj_factor_rows": filled_adj_factor_rows,
            "all_missing_adj_factor_assets": all_missing_adj_factor_assets,
            "requested_start_date": str(start_ts.strftime("%Y-%m-%d"))
            if start_ts is not None
            else None,
            "requested_end_date": str(end_ts.strftime("%Y-%m-%d")) if end_ts is not None else None,
        }
        dataset_version = self.catalog.write_dataset_version(
            dataset_name=self.catalog.CORE_DATASET_NAME,
            table_names=_WRITTEN_TABLES,
            raw_snapshot_id=None,
            notes=notes,
        )
        return LocalZipAshareDailyIngestResult(
            dataset_version=dataset_version,
            written_tables=_WRITTEN_TABLES,
            row_counts=row_counts,
            asset_count=len(seen_assets),
            date_range=date_range,
            notes=notes,
        )

    def organize_daily_storage(
        self,
        *,
        progress_callback: Callable[[str], None] | None = None,
    ) -> LocalZipAshareDailyOrganizeResult:
        """Rewrite local ZIP-imported tables into canonical year/month partitions."""
        self.catalog.ensure_layout()
        table_specs: tuple[tuple[str, tuple[str, ...], str], ...] = (
            ("daily_bars", ("date", "asset"), "date"),
            ("adj_factor", ("date", "asset"), "date"),
            ("asset_status", ("date", "asset"), "date"),
            ("trade_calendar", ("date",), "date"),
            ("instruments", ("asset",), "list_date"),
        )
        row_counts: dict[str, int] = {}
        rewritten_tables: list[str] = []

        for table_name, key_cols, partition_column in table_specs:
            frame = self.catalog.load_table(table_name, date_field=partition_column)
            row_counts[table_name] = int(len(frame))
            if frame.empty:
                continue
            if progress_callback is not None:
                progress_callback(
                    f"rewriting {table_name} into year/month partitions ({len(frame)} rows)"
                )
            table_root = self.catalog.table_root(table_name)
            if table_root.exists():
                shutil.rmtree(table_root)
            self.catalog.upsert_table(
                table_name,
                frame,
                key_cols=key_cols,
                partition_column=partition_column,
            )
            rewritten_tables.append(table_name)

        notes: dict[str, object] = {
            "source_vendor": "local_zip",
            "storage_layout": "year_month",
            "reorganized_from": "asset_partitioned_local_zip_import",
        }
        dataset_version = self.catalog.write_dataset_version(
            dataset_name=self.catalog.CORE_DATASET_NAME,
            table_names=tuple(rewritten_tables),
            raw_snapshot_id=None,
            notes=notes,
        )
        return LocalZipAshareDailyOrganizeResult(
            dataset_version=dataset_version,
            rewritten_tables=tuple(rewritten_tables),
            row_counts=row_counts,
            notes=notes,
        )


def _prepare_member_archive(source_path: Path, *, tmp_dir: Path) -> tuple[Path, str]:
    with zipfile.ZipFile(source_path) as outer:
        csv_members = [
            info.filename
            for info in outer.infolist()
            if not info.is_dir() and info.filename.lower().endswith(".csv")
        ]
        if csv_members:
            return source_path, source_path.name
        inner_archives = [
            info.filename
            for info in outer.infolist()
            if not info.is_dir() and info.filename.lower().endswith(".zip")
        ]
        if len(inner_archives) != 1:
            raise AlphaLabConfigError(
                "zip archive must contain either direct CSV members or exactly "
                "one nested ZIP archive"
            )
        inner_name = inner_archives[0]
        staged_path = tmp_dir / Path(inner_name).name
        with outer.open(inner_name) as src, staged_path.open("wb") as dst:
            shutil.copyfileobj(src, dst, length=1024 * 1024)
        return staged_path, f"{source_path.name}::{inner_name}"


def _canonicalize_asset_frame(
    frame: pd.DataFrame,
    *,
    member_name: str,
    start_date: pd.Timestamp | None,
    end_date: pd.Timestamp | None,
) -> _AssetFrames | None:
    if frame.empty:
        raise AlphaLabDataError(f"asset CSV is empty: {member_name}")
    required = {"ts_code", "trade_date", "close"}
    missing = required - set(frame.columns)
    if missing:
        raise AlphaLabDataError(
            f"asset CSV missing required columns {sorted(missing)}: {member_name}"
        )

    out = frame.copy()
    fallback_asset = Path(member_name).stem.replace("_", ".").upper()
    out["asset"] = (
        out["ts_code"].astype("string").str.strip().replace({"": pd.NA}).fillna(fallback_asset)
    )
    asset_values = sorted(set(out["asset"].dropna().astype(str).tolist()))
    if len(asset_values) != 1:
        raise AlphaLabDataError(
            f"asset CSV must contain exactly one ts_code, got {asset_values[:5]} in {member_name}"
        )
    asset = asset_values[0]
    out["date"] = pd.to_datetime(out["trade_date"], format="%Y%m%d", errors="coerce")
    if out["date"].isna().any():
        raise AlphaLabDataError(f"trade_date contains invalid values: {member_name}")
    if start_date is not None:
        out = out[out["date"] >= start_date].copy()
    if end_date is not None:
        out = out[out["date"] <= end_date].copy()
    if out.empty:
        return None
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "pre_close",
        "change",
        "pct_chg",
        "vol",
        "amount",
        "adj_factor",
    ]
    for column in numeric_columns:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")

    bars = out.rename(columns={"vol": "volume"})[
        [
            "date",
            "asset",
            "open",
            "high",
            "low",
            "close",
            "pre_close",
            "change",
            "pct_chg",
            "volume",
            "amount",
        ]
    ].copy()
    bars = bars.sort_values(["date"], kind="mergesort").drop_duplicates(
        subset=["date"], keep="last"
    )
    if bars["close"].isna().any():
        raise AlphaLabDataError(f"close contains NaN rows after parsing: {member_name}")
    if (pd.to_numeric(bars["close"], errors="coerce") <= 0).any():
        raise AlphaLabDataError(f"close contains non-positive rows: {member_name}")
    bars = bars.reset_index(drop=True)

    adj_series = pd.to_numeric(out["adj_factor"], errors="coerce")
    if adj_series.notna().sum() == 0:
        filled_adj = pd.Series(1.0, index=out.index, dtype=float)
        inferred_mode = "default_1.0_all_missing"
    elif adj_series.isna().any():
        filled_adj = adj_series.bfill().ffill()
        inferred_mode = "bfill_ffill_partial_missing"
    else:
        filled_adj = adj_series.astype(float)
        inferred_mode = "as_is"
    if filled_adj.isna().any():
        raise AlphaLabDataError(f"adj_factor still contains NaN rows after fill: {member_name}")
    if (filled_adj <= 0).any():
        raise AlphaLabDataError(f"adj_factor contains non-positive rows: {member_name}")

    adj_factor = out[["date", "asset"]].copy()
    adj_factor["adj_factor"] = filled_adj.to_numpy(dtype=float)
    adj_factor = adj_factor.sort_values(["date"], kind="mergesort").drop_duplicates(
        subset=["date"], keep="last"
    )
    adj_factor = adj_factor.reset_index(drop=True)

    asset_status = bars[["date", "asset"]].copy()
    volume = pd.to_numeric(bars["volume"], errors="coerce").fillna(0.0)
    amount = pd.to_numeric(bars["amount"], errors="coerce").fillna(0.0)
    asset_status["is_suspended"] = ((volume <= 0.0) | (amount <= 0.0)).astype(int)
    asset_status["is_st"] = 0

    filled_adj_rows = int(adj_series.isna().sum())
    return _AssetFrames(
        asset=asset,
        daily_bars=bars,
        adj_factor=adj_factor,
        asset_status=asset_status,
        list_date=str(bars["date"].iloc[0]),
        last_date=str(bars["date"].iloc[-1]),
        filled_adj_factor_rows=filled_adj_rows,
        inferred_adj_factor_mode=inferred_mode,
    )


def _write_asset_partition(*, table_root: Path, asset: str, frame: pd.DataFrame) -> None:
    partition_dir = table_root / f"asset={asset}"
    partition_dir.mkdir(parents=True, exist_ok=True)
    to_write = frame.drop(columns=["asset"]).reset_index(drop=True)
    to_write.to_parquet(partition_dir / "part-00000.parquet", index=False, compression="zstd")


def _write_single_file_table(
    *,
    table_root: Path,
    frame: pd.DataFrame,
    sort_cols: tuple[str, ...] | None = None,
) -> None:
    table_root.mkdir(parents=True, exist_ok=True)
    out = frame.copy()
    if sort_cols:
        out = out.sort_values(list(sort_cols), kind="mergesort").reset_index(drop=True)
    out.to_parquet(table_root / "part-00000.parquet", index=False, compression="zstd")


def _resolve_date_window(
    *,
    start_date: str | None,
    end_date: str | None,
) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    start_ts = pd.Timestamp(start_date) if start_date is not None else None
    end_ts = pd.Timestamp(end_date) if end_date is not None else None
    if start_ts is not None and pd.isna(start_ts):
        raise AlphaLabConfigError(f"invalid start_date: {start_date!r}")
    if end_ts is not None and pd.isna(end_ts):
        raise AlphaLabConfigError(f"invalid end_date: {end_date!r}")
    if start_ts is not None and end_ts is not None and start_ts > end_ts:
        raise AlphaLabConfigError(
            f"start_date must be <= end_date, got start_date={start_date!r} end_date={end_date!r}"
        )
    return start_ts, end_ts
