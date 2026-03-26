#!/usr/bin/env python3
"""Lightweight Tushare ingestion/build CLI for canonical A-share research cases."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping

from alpha_lab.data_sources import (
    RequiredEndpointExtractionError,
    build_standardized_tushare_tables,
    build_tushare_pro_client,
    build_tushare_research_inputs,
    export_canonical_tushare_case_configs,
    fetch_tushare_raw_snapshots,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tushare_pipeline",
        description="Fetch Tushare raw snapshots and build workflow-compatible research inputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch = subparsers.add_parser("fetch-snapshots")
    fetch.add_argument("--snapshot-name", required=True)
    fetch.add_argument("--start-date", required=True, help="YYYYMMDD")
    fetch.add_argument("--end-date", required=True, help="YYYYMMDD")
    fetch.add_argument("--raw-root", default=None)
    fetch.add_argument("--token", default=None)
    fetch.add_argument("--token-env-var", default="TUSHARE_TOKEN")
    fetch.add_argument(
        "--debug-endpoints",
        action="store_true",
        help="Print per-endpoint extraction progress and failure diagnostics.",
    )

    standardize = subparsers.add_parser("build-standardized")
    standardize.add_argument("--snapshot-dir", required=True)
    standardize.add_argument("--output-dir", default=None)

    cases = subparsers.add_parser("build-cases")
    cases.add_argument("--standardized-dir", required=True)
    cases.add_argument("--output-dir", default=None)
    cases.add_argument("--dataset-id", default=None)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "fetch-snapshots":
        client = build_tushare_pro_client(token=args.token, token_env_var=args.token_env_var)
        hook = _debug_endpoint_event if args.debug_endpoints else None
        try:
            result = fetch_tushare_raw_snapshots(
                client,
                snapshot_name=args.snapshot_name,
                start_date=args.start_date,
                end_date=args.end_date,
                raw_root=args.raw_root,
                endpoint_diagnostic_hook=hook,
            )
        except RequiredEndpointExtractionError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print("")
        print(f"  Snapshot Dir : {result.manifest.snapshot_dir}")
        print(f"  Endpoints    : {', '.join(result.manifest.endpoints)}")
        if result.manifest.unavailable_endpoints:
            print(f"  Unavailable  : {', '.join(result.manifest.unavailable_endpoints)}")
        if result.manifest.degraded_endpoints:
            print(f"  Degraded     : {', '.join(result.manifest.degraded_endpoints)}")
        return 0

    if args.command == "build-standardized":
        tables = build_standardized_tushare_tables(
            args.snapshot_dir,
            output_dir=args.output_dir,
        )
        print("")
        print(f"  Standardized Dir : {tables.snapshot_dir}")
        print(f"  Prices Rows      : {len(tables.prices)}")
        print(f"  Asset Rows       : {len(tables.asset_metadata)}")
        if tables.unavailable_raw_endpoints:
            print(f"  Unavailable      : {', '.join(tables.unavailable_raw_endpoints)}")
        return 0

    if args.command == "build-cases":
        bundle = build_tushare_research_inputs(
            args.standardized_dir,
            output_dir=args.output_dir,
            dataset_id=args.dataset_id,
        )
        case_artifacts = export_canonical_tushare_case_configs(
            bundle,
            output_dir=bundle.output_dir,
            dataset_id=args.dataset_id,
        )
        print("")
        print(f"  Research Input Dir : {bundle.output_dir}")
        print(f"  Single Config      : {case_artifacts.single_factor_config_path}")
        print(f"  Composite Config   : {case_artifacts.composite_config_path}")
        single_cmd = (
            "uv run python scripts/run_research_workflow.py run-single-factor "
            f"--config-path {case_artifacts.single_factor_config_path} "
            f"--output-dir {bundle.output_dir / 'single_case_output'}"
        )
        composite_cmd = (
            "uv run python scripts/run_research_workflow.py run-composite "
            f"--config-path {case_artifacts.composite_config_path} "
            f"--output-dir {bundle.output_dir / 'composite_case_output'}"
        )
        print(f"  Run Single         : {single_cmd}")
        print(f"  Run Composite      : {composite_cmd}")
        return 0

    parser.error(f"unsupported command: {args.command!r}")
    return 2

def _debug_endpoint_event(event: Mapping[str, object]) -> None:
    endpoint = str(event.get("endpoint_requested", ""))
    mode = str(event.get("extraction_mode", ""))
    required = bool(event.get("required", False))
    if str(event.get("event")) == "started":
        print(
            "[endpoint] started "
            f"endpoint={endpoint} mode={mode} required={required} "
            f"params={event.get('params', {})}"
        )
        return
    status = str(event.get("status", ""))
    row_count = event.get("row_count")
    fallback_used = bool(event.get("fallback_used", False))
    degradation_used = bool(event.get("degradation_used", False))
    error_message = event.get("error_message")
    suffix = f" error={error_message}" if error_message else ""
    print(
        "[endpoint] finished "
        f"endpoint={endpoint} mode={mode} required={required} status={status} "
        f"row_count={row_count} fallback_used={fallback_used} "
        f"degradation_used={degradation_used}{suffix}"
    )


if __name__ == "__main__":
    sys.exit(main())
