from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from alpha_lab.exceptions import AlphaLabConfigError, AlphaLabDataError

from .slice_presets import (
    DEFAULT_SLICE_PRESET,
    SLICE_PRESETS,
    SlicePresetConfig,
    resolve_slice_window,
)

if TYPE_CHECKING:
    import pandas as pd

    from .catalog import DataCatalog, SliceSpec


# Lazily resolved on first attribute access. Keeping these names addressable as
# module attributes (rather than function-local imports) preserves the
# monkeypatch surface that ``tests/test_data_store_cli.py`` relies on
# (``monkeypatch.setattr("alpha_lab.data_store.cli.TushareIngestor", ...)``).
# Tests that install a fake _before_ ``main()`` runs will hit the fake on
# attribute access; the lazy resolver only fires when the attribute is
# missing.
_LAZY_CLI_EXPORTS: dict[str, tuple[str, str]] = {
    "DataCatalog": ("alpha_lab.data_store.catalog", "DataCatalog"),
    "SliceSpec": ("alpha_lab.data_store.catalog", "SliceSpec"),
    "LocalZipAshareDailyIngestor": (
        "alpha_lab.data_store.local_zip",
        "LocalZipAshareDailyIngestor",
    ),
    "TushareIngestor": ("alpha_lab.data_store.tushare", "TushareIngestor"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY_CLI_EXPORTS.get(name)
    if target is None:
        raise AttributeError(
            f"module 'alpha_lab.data_store.cli' has no attribute {name!r}"
        )
    from importlib import import_module

    submodule_path, attr = target
    return getattr(import_module(submodule_path), attr)

UNIVERSE_CHOICES = [
    "all_ashare",
    "listed_90d",
    "institutional_ashare",
    "top_liquid_300",
    "top_liquid_500",
    "top_liquid_800",
]
MODE_CHOICES = ["daily", "fundamental", "full"]
EXPORT_FORMAT_CHOICES = ["csv", "parquet", "both"]
VALIDATE_LEVEL_CHOICES = ["raw", "canonical", "all"]


def build_data_parser(parser: argparse.ArgumentParser) -> None:
    parser.description = "Manage external Parquet-backed research datasets for Level 1/2 workflows."
    commands = parser.add_subparsers(dest="data_action", required=True)

    init_cmd = commands.add_parser(
        "init",
        help="Initialize the external data root layout.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    init_cmd.add_argument(
        "--data-root",
        default=None,
        help=(
            "Optional external data root override. Defaults to "
            "ALPHA_LAB_DATA_ROOT or ~/.local/share/alpha-lab/data."
        ),
    )

    ingest = commands.add_parser(
        "ingest",
        help="Ingest vendor data into the canonical data store.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ingest_vendor = ingest.add_subparsers(dest="vendor", required=True)
    ingest_tushare = ingest_vendor.add_parser(
        "tushare",
        help="Ingest Tushare data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ingest_tushare.add_argument("dataset_scope", choices=["core"])
    _add_common_tushare_args(ingest_tushare)
    ingest_local_zip = ingest_vendor.add_parser(
        "local-zip",
        help="Ingest nested ZIP archives of per-asset A-share daily bars.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ingest_local_zip.add_argument("dataset_scope", choices=["ashare-daily"])
    ingest_local_zip.add_argument(
        "--zip-path",
        required=True,
        help=(
            "Path to the outer ZIP file. Supports either direct CSV members "
            "or one nested ZIP containing per-asset CSV files."
        ),
    )
    ingest_local_zip.add_argument(
        "--start-date",
        default=None,
        help="Optional inclusive start date YYYY-MM-DD for filtered ingest.",
    )
    ingest_local_zip.add_argument(
        "--end-date",
        default=None,
        help="Optional inclusive end date YYYY-MM-DD for filtered ingest.",
    )
    ingest_local_zip.add_argument(
        "--data-root",
        default=None,
        help="Optional external data root override.",
    )

    update = commands.add_parser(
        "update",
        help="Incrementally update an existing vendor dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    update_vendor = update.add_subparsers(dest="vendor", required=True)
    update_tushare = update_vendor.add_parser(
        "tushare",
        help="Update Tushare-backed canonical tables.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    update_tushare.add_argument("dataset_scope", choices=["core"])
    update_tushare.add_argument(
        "--end-date", required=True, help="Update until this YYYY-MM-DD date."
    )
    update_tushare.add_argument("--token", default=None, help="Optional Tushare token override.")
    update_tushare.add_argument("--assets-file", default=None, help="Optional asset list file.")
    update_tushare.add_argument(
        "--asset-limit", type=int, default=None, help="Optional cap on assets."
    )
    update_tushare.add_argument(
        "--mode",
        default="daily",
        choices=MODE_CHOICES,
        help=(
            "Update mode. daily refreshes fast day tables, "
            "fundamental refreshes slow财务 tables, full runs both."
        ),
    )
    update_tushare.add_argument(
        "--daily-research-only",
        action="store_true",
        help=(
            "Skip slow ROE financial-indicator fetches and keep the "
            "ingest focused on daily price-volume research."
        ),
    )
    update_tushare.add_argument(
        "--chunk-months",
        type=int,
        default=6,
        help="Split long update windows into N-month chunks. Use 0 to disable chunking.",
    )
    update_tushare.add_argument(
        "--data-root", default=None, help="Optional external data root override."
    )

    validate = commands.add_parser(
        "validate",
        help="Validate the canonical Level 1/2 data store tables.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    validate.add_argument(
        "--level",
        default="all",
        choices=VALIDATE_LEVEL_CHOICES,
        help="Validate the latest raw snapshot, the canonical tables, or both.",
    )
    validate.add_argument("--data-root", default=None, help="Optional external data root override.")

    export = commands.add_parser(
        "export-case-inputs",
        help="Export canonical case inputs (prices/universe/factors) for real-case workflows.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    export.add_argument(
        "--slice-preset",
        default=DEFAULT_SLICE_PRESET,
        choices=sorted(SLICE_PRESETS),
        help="Built-in slice preset for daily price-volume research.",
    )
    export.add_argument(
        "--start-date",
        default=None,
        help=(
            "Optional slice start date YYYY-MM-DD. Defaults to the selected preset lookback window."
        ),
    )
    export.add_argument(
        "--end-date",
        default=None,
        help=(
            "Optional slice end date YYYY-MM-DD. Defaults to the latest available daily_bars date."
        ),
    )
    export.add_argument("--output-dir", required=True, help="Directory for exported case inputs.")
    export.add_argument(
        "--universe",
        default=None,
        choices=UNIVERSE_CHOICES,
        help="Optional universe override. Defaults to the selected slice preset.",
    )
    export.add_argument(
        "--factors",
        nargs="*",
        default=[],
        choices=["bp", "roe_ttm"],
        help=(
            "Optional factor CSVs to export beside prices/universe. Omit "
            "for price-volume-only slices."
        ),
    )
    export.add_argument(
        "--adjustment",
        default=None,
        choices=["raw", "qfq"],
        help="Optional price adjustment override. Defaults to the selected slice preset.",
    )
    export.add_argument("--assets-file", default=None, help="Optional explicit asset list file.")
    export.add_argument("--asset-limit", type=int, default=None, help="Optional cap on assets.")
    export.add_argument(
        "--format",
        default="both",
        choices=EXPORT_FORMAT_CHOICES,
        help="Write CSV, Parquet, or both.",
    )
    export.add_argument(
        "--prefer-cache",
        action="store_true",
        help="Reuse an existing slice cache when dataset_version_id matches.",
    )
    export.add_argument("--data-root", default=None, help="Optional external data root override.")

    materialize = commands.add_parser(
        "materialize-slice",
        help=(
            "Build and cache a reusable Level 1/2 slice without choosing "
            "a separate output directory."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    materialize.add_argument(
        "--slice-preset",
        default=DEFAULT_SLICE_PRESET,
        choices=sorted(SLICE_PRESETS),
        help="Built-in slice preset for daily price-volume research.",
    )
    materialize.add_argument(
        "--start-date", default=None, help="Optional slice start date YYYY-MM-DD."
    )
    materialize.add_argument("--end-date", default=None, help="Optional slice end date YYYY-MM-DD.")
    materialize.add_argument(
        "--universe",
        default=None,
        choices=UNIVERSE_CHOICES,
        help="Optional universe override. Defaults to the selected slice preset.",
    )
    materialize.add_argument(
        "--factors",
        nargs="*",
        default=[],
        choices=["bp", "roe_ttm"],
        help="Optional factor files to cache beside prices/universe.",
    )
    materialize.add_argument(
        "--adjustment",
        default=None,
        choices=["raw", "qfq"],
        help="Optional price adjustment override. Defaults to the selected slice preset.",
    )
    materialize.add_argument(
        "--assets-file", default=None, help="Optional explicit asset list file."
    )
    materialize.add_argument(
        "--asset-limit", type=int, default=None, help="Optional cap on assets."
    )
    materialize.add_argument(
        "--format",
        default="parquet",
        choices=EXPORT_FORMAT_CHOICES,
        help="Cache CSV, Parquet, or both.",
    )
    materialize.add_argument(
        "--data-root", default=None, help="Optional external data root override."
    )

    query = commands.add_parser(
        "query",
        help="Execute one read-only SQL query against DuckDB catalog views.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    query.add_argument(
        "--sql",
        required=True,
        help=(
            "One read-only SQL statement. Views are created for canonical tables "
            "such as daily_bars, daily_basic, financial_indicator, trade_calendar, and instruments."
        ),
    )
    query.add_argument(
        "--format",
        default="table",
        choices=["table", "csv", "json"],
        help="Output format for query results.",
    )
    query.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional output row cap applied after the SQL result is fetched.",
    )
    query.add_argument(
        "--refresh-catalog",
        action="store_true",
        help="Refresh DuckDB views before running the query.",
    )
    query.add_argument("--data-root", default=None, help="Optional external data root override.")

    organize_local_zip = commands.add_parser(
        "organize-local-zip",
        help="Rewrite local ZIP-imported A-share daily tables into year/month partitions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    organize_local_zip.add_argument(
        "--data-root",
        default=None,
        help="Optional external data root override.",
    )


def main(argv: list[str] | None = None) -> int:
    # ``DataCatalog`` / ``LocalZipAshareDailyIngestor`` / ``TushareIngestor``
    # resolve via this module's ``__getattr__`` (lazy import) — pulled into
    # local bindings here so the dispatch code below can use the bare names.
    # Tests that ``monkeypatch.setattr("alpha_lab.data_store.cli.X", fake)``
    # before calling ``main()`` still take effect: ``sys.modules[__name__]``
    # is the same object the monkeypatch wrote to.
    _self = sys.modules[__name__]
    DataCatalog = _self.DataCatalog  # noqa: N806
    LocalZipAshareDailyIngestor = _self.LocalZipAshareDailyIngestor  # noqa: N806
    TushareIngestor = _self.TushareIngestor  # noqa: N806

    parser = argparse.ArgumentParser(
        prog="alpha-lab data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    build_data_parser(parser)
    args = parser.parse_args(argv)

    if args.data_action == "init":
        catalog = DataCatalog(root=args.data_root)
        catalog.ensure_layout()
        print("")
        print("  Workflow : data-init")
        print("  Status   : success")
        print(f"  Data root: {catalog.root}")
        duckdb_status = (
            "available" if catalog.duckdb_catalog_path.exists() else "optional/unavailable"
        )
        print(f"  DuckDB   : {duckdb_status}")
        return 0

    if args.data_action == "ingest":
        if args.vendor == "tushare" and args.dataset_scope == "core":
            tushare_ingestor = TushareIngestor(DataCatalog(root=args.data_root))
            tushare_result = tushare_ingestor.ingest_core_chunked(
                start_date=args.start_date,
                end_date=args.end_date,
                token=args.token,
                assets=_load_assets_file(args.assets_file),
                asset_limit=args.asset_limit,
                chunk_months=args.chunk_months,
                mode=args.mode,
                daily_research_only=bool(args.daily_research_only),
                progress_callback=_build_progress_callback(),
            )
            print("")
            print("  Workflow          : data-ingest-tushare-core")
            print("  Status            : success")
            print(f"  Data root         : {tushare_ingestor.catalog.root}")
            quality_notes = getattr(tushare_result, "quality_notes", {}) or {}
            print(f"  Mode              : {quality_notes.get('mode', args.mode)}")
            print(f"  Snapshot          : {tushare_result.snapshot_manifest.snapshot_id}")
            print(f"  Dataset version   : {tushare_result.dataset_version.version_id}")
            print(f"  Chunks            : {tushare_result.chunk_count}")
            raw_report_path = quality_notes.get("raw_validation_report_path")
            if raw_report_path is not None:
                print(f"  Raw validation    : {raw_report_path}")
            for name, count in tushare_result.row_counts.items():
                print(f"  {name:18}: {count}")
            return 0
        if args.vendor == "local-zip" and args.dataset_scope == "ashare-daily":
            local_zip_ingestor = LocalZipAshareDailyIngestor(DataCatalog(root=args.data_root))
            local_zip_result = local_zip_ingestor.ingest_daily_zip(
                zip_path=args.zip_path,
                start_date=args.start_date,
                end_date=args.end_date,
                progress_callback=_build_progress_callback(),
            )
            print("")
            print("  Workflow          : data-ingest-local-zip-ashare-daily")
            print("  Status            : success")
            print(f"  Data root         : {local_zip_ingestor.catalog.root}")
            print(f"  Dataset version   : {local_zip_result.dataset_version.version_id}")
            print(f"  Assets            : {local_zip_result.asset_count}")
            start_date = local_zip_result.date_range.get("start_date")
            end_date = local_zip_result.date_range.get("end_date")
            print(f"  Window            : {start_date} -> {end_date}")
            for name, count in local_zip_result.row_counts.items():
                print(f"  {name:18}: {count}")
            return 0
        parser.error(
            "supported ingest commands are "
            "'alpha-lab data ingest tushare core' and "
            "'alpha-lab data ingest local-zip ashare-daily'"
        )

    if args.data_action == "update":
        if args.vendor != "tushare" or args.dataset_scope != "core":
            parser.error("only 'alpha-lab data update tushare core' is currently supported")
        update_ingestor = TushareIngestor(DataCatalog(root=args.data_root))
        update_result = update_ingestor.update_core(
            end_date=args.end_date,
            token=args.token,
            assets=_load_assets_file(args.assets_file),
            asset_limit=args.asset_limit,
            chunk_months=args.chunk_months,
            mode=args.mode,
            daily_research_only=bool(args.daily_research_only),
            progress_callback=_build_progress_callback(),
        )
        print("")
        print("  Workflow          : data-update-tushare-core")
        print("  Status            : success")
        print(f"  Data root         : {update_ingestor.catalog.root}")
        quality_notes = getattr(update_result, "quality_notes", {}) or {}
        print(f"  Mode              : {quality_notes.get('mode', args.mode)}")
        print(f"  Dataset version   : {update_result.dataset_version.version_id}")
        print(f"  Snapshot          : {update_result.snapshot_manifest.snapshot_id}")
        print(f"  Chunks            : {update_result.chunk_count}")
        raw_report_path = quality_notes.get("raw_validation_report_path")
        if raw_report_path is not None:
            print(f"  Raw validation    : {raw_report_path}")
        return 0

    if args.data_action == "validate":
        catalog = DataCatalog(root=args.data_root)
        raw_ok = True
        canonical_ok = True
        print("")
        print("  Workflow : data-validate")
        if args.level in {"raw", "all"}:
            version = catalog.get_current_dataset_version(DataCatalog.CORE_DATASET_NAME)
            snapshot_id = version.raw_snapshot_id if version is not None else None
            if snapshot_id is None:
                raw_ok = False
                print("  Raw      : no current raw snapshot is registered")
            else:
                raw_report = catalog.validate_raw_snapshot(snapshot_id)
                raw_ok = raw_report.ok
                print(f"  Raw      : {'success' if raw_report.ok else 'issues-found'}")
                print(f"  RawReport: {raw_report.report_path}")
                for issue in raw_report.issues:
                    print(f"  [{issue.severity}] {issue.table_name}: {issue.message}")
        if args.level in {"canonical", "all"}:
            report = catalog.validate_core_dataset()
            canonical_ok = report.ok
            print(f"  Canonical: {'success' if report.ok else 'issues-found'}")
            print(f"  Report   : {report.report_path}")
            for issue in report.issues:
                print(f"  [{issue.severity}] {issue.table_name}: {issue.message}")
        ok = (
            raw_ok
            if args.level == "raw"
            else canonical_ok
            if args.level == "canonical"
            else raw_ok and canonical_ok
        )
        print(f"  Status   : {'success' if ok else 'issues-found'}")
        return 0 if ok else 1

    if args.data_action == "export-case-inputs":
        catalog = DataCatalog(root=args.data_root)
        slice_spec, preset = _resolve_export_slice_spec(args, catalog)
        export_result = catalog.export_case_inputs(
            slice_spec=slice_spec,
            output_dir=args.output_dir,
            formats=_resolve_output_formats(args.format),
            prefer_cache=bool(args.prefer_cache),
        )
        print("")
        print("  Workflow          : data-export-case-inputs")
        print("  Status            : success")
        print(f"  Data root         : {catalog.root}")
        print(f"  Slice preset      : {preset.name}")
        print(f"  Window            : {slice_spec.start_date} -> {slice_spec.end_date}")
        print(f"  Universe          : {slice_spec.universe_name}")
        print(f"  Adjustment        : {slice_spec.adjustment}")
        print(f"  Output dir        : {export_result.output_dir}")
        print(f"  Cache dir         : {export_result.cache_dir}")
        print(f"  Dataset version   : {export_result.dataset_version_id}")
        for name, path in export_result.output_paths.items():
            print(f"  {name:18}: {path}")
        return 0

    if args.data_action == "materialize-slice":
        catalog = DataCatalog(root=args.data_root)
        slice_spec, preset = _resolve_export_slice_spec(args, catalog)
        slice_result = catalog.materialize_slice(
            slice_spec=slice_spec,
            formats=_resolve_output_formats(args.format),
        )
        print("")
        print("  Workflow          : data-materialize-slice")
        print("  Status            : success")
        print(f"  Data root         : {catalog.root}")
        print(f"  Slice preset      : {preset.name}")
        print(f"  Window            : {slice_spec.start_date} -> {slice_spec.end_date}")
        print(f"  Cache dir         : {slice_result.cache_dir}")
        print(f"  Dataset version   : {slice_result.dataset_version_id}")
        return 0

    if args.data_action == "query":
        catalog = DataCatalog(root=args.data_root)
        frame = catalog.query_sql(
            args.sql,
            refresh_catalog=bool(args.refresh_catalog),
        )
        if args.limit is not None:
            if args.limit <= 0:
                parser.error("--limit must be > 0 when provided")
            frame = frame.head(args.limit)
        _print_query_frame(frame, output_format=args.format)
        return 0

    if args.data_action == "organize-local-zip":
        local_zip_ingestor = LocalZipAshareDailyIngestor(DataCatalog(root=args.data_root))
        organize_result = local_zip_ingestor.organize_daily_storage(
            progress_callback=_build_progress_callback()
        )
        print("")
        print("  Workflow          : data-organize-local-zip")
        print("  Status            : success")
        print(f"  Data root         : {local_zip_ingestor.catalog.root}")
        print(f"  Dataset version   : {organize_result.dataset_version.version_id}")
        for name, count in organize_result.row_counts.items():
            print(f"  {name:18}: {count}")
        return 0

    parser.error(f"unsupported data action: {args.data_action!r}")


def _add_common_tushare_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--start-date", required=True, help="Slice start date YYYY-MM-DD.")
    parser.add_argument("--end-date", required=True, help="Slice end date YYYY-MM-DD.")
    parser.add_argument("--token", default=None, help="Optional Tushare token override.")
    parser.add_argument("--assets-file", default=None, help="Optional asset list file.")
    parser.add_argument("--asset-limit", type=int, default=None, help="Optional cap on assets.")
    parser.add_argument(
        "--mode",
        default="daily",
        choices=MODE_CHOICES,
        help=(
            "Ingest mode. daily (default) covers price-volume research and "
            "skips ROE/industry_classification; fundamental refreshes slow "
            "财务 tables; full runs both."
        ),
    )
    parser.add_argument(
        "--daily-research-only",
        action="store_true",
        help=(
            "Belt-and-suspenders flag for PV research: downgrades --mode full "
            "back to daily. Redundant when --mode daily is already selected."
        ),
    )
    parser.add_argument(
        "--chunk-months",
        type=int,
        default=6,
        help="Split long ingest windows into N-month chunks. Use 0 to disable chunking.",
    )
    parser.add_argument("--data-root", default=None, help="Optional external data root override.")


def _load_assets_file(path_text: str | None) -> tuple[str, ...] | None:
    if path_text is None:
        return None
    path = Path(path_text).resolve()
    tokens: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        token = raw.strip()
        if not token or token.startswith("#"):
            continue
        tokens.append(token)
    if not tokens:
        raise AlphaLabDataError(f"assets file is empty: {path}")
    return tuple(sorted(set(tokens)))


def _resolve_export_slice_spec(
    args: argparse.Namespace,
    catalog: DataCatalog,
) -> tuple[SliceSpec, SlicePresetConfig]:
    from .catalog import SliceSpec

    fallback_end_date = args.end_date or catalog.latest_date("daily_bars")
    if fallback_end_date is None:
        raise AlphaLabDataError(
            "end_date is required when daily_bars is empty or missing. "
            "Run ingest first or pass --end-date explicitly."
        )
    resolved_start_date, resolved_end_date, preset = resolve_slice_window(
        preset_name=str(args.slice_preset),
        start_date=args.start_date,
        end_date=args.end_date,
        fallback_end_date=str(fallback_end_date),
    )

    resolved_universe = str(args.universe or preset.universe_name)
    resolved_adjustment = str(args.adjustment or preset.adjustment)
    return (
        SliceSpec(
            start_date=resolved_start_date,
            end_date=resolved_end_date,
            slice_preset=preset.name,
            universe_name=resolved_universe,
            assets=_load_assets_file(args.assets_file),
            asset_limit=args.asset_limit,
            factors=tuple(args.factors),
            adjustment=resolved_adjustment,
        ),
        preset,
    )


def _print_query_frame(frame: pd.DataFrame, *, output_format: str) -> None:
    if output_format == "table":
        print("")
        if frame.empty:
            print("  Query returned 0 rows.")
            return
        print(frame.to_string(index=False))
        return
    if output_format == "csv":
        print(frame.to_csv(index=False), end="")
        return
    if output_format == "json":
        print(frame.to_json(orient="records", force_ascii=False, indent=2))
        return
    raise AlphaLabConfigError(f"unsupported output format: {output_format!r}")


def _build_progress_callback() -> Callable[[str], None]:
    def _emit(message: str) -> None:
        print(f"  Progress          : {message}")

    return _emit


def _resolve_output_formats(format_name: str) -> tuple[str, ...]:
    if format_name == "both":
        return ("csv", "parquet")
    return (str(format_name),)
