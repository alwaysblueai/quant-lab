#!/usr/bin/env python3
"""Generate canonical real-case CSV inputs from BaoStock."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from alpha_lab.data_adapters.baostock_adapter import generate_real_case_inputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="generate_baostock_real_case_inputs",
        description=(
            "Fetch BaoStock A-share data and write canonical CSV inputs required "
            "by real-case single-factor pipelines."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--start-date", default="2024-01-01", help="Start date in YYYY-MM-DD.")
    parser.add_argument("--end-date", default="2024-12-31", help="End date in YYYY-MM-DD.")
    parser.add_argument(
        "--output-dir",
        default="data/processed/real_case_inputs/baostock_v1",
        help="Output directory for generated CSV files.",
    )
    parser.add_argument(
        "--assets-file",
        default=None,
        help=("Optional text file containing one stock code per line (000001.SZ or sz.000001)."),
    )
    parser.add_argument(
        "--asset-limit",
        type=int,
        default=None,
        help="Optional cap on number of assets to keep (sorted by code).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Python logging level.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    assets = _load_assets_file(args.assets_file)
    summary = generate_real_case_inputs(
        output_dir=Path(args.output_dir),
        start_date=args.start_date,
        end_date=args.end_date,
        assets=assets,
        asset_limit=args.asset_limit,
    )

    print("")
    print("  BaoStock real-case input generation complete")
    print(f"  Output dir         : {summary.output_dir}")
    print(f"  Prices CSV         : {summary.output_paths['prices']}")
    print(f"  BP CSV             : {summary.output_paths['bp']}")
    print(f"  ROE_TTM CSV        : {summary.output_paths['roe_ttm']}")
    print(f"  Universe CSV       : {summary.output_paths['universe']}")
    print("  Row counts:")
    print(f"    prices           : {summary.row_counts['prices']}")
    print(f"    bp               : {summary.row_counts['bp']}")
    print(f"    roe_ttm          : {summary.row_counts['roe_ttm']}")
    print(f"    universe         : {summary.row_counts['universe']}")
    print("  Dedup raw rows:")
    print(f"    prices_raw       : {summary.dedup_counts['prices_raw']}")
    print(f"    pb_raw           : {summary.dedup_counts['pb_raw']}")
    print(f"    roe_raw          : {summary.dedup_counts['roe_raw']}")
    print(f"  ROE source column  : {summary.roe_source_column}")
    print(f"  ROE end_date fallback rows: {summary.roe_rows_using_end_date_fallback}")
    print("")
    print("  Note: BaoStock ROE coverage depends on fundamental endpoint availability.")
    return 0


def _load_assets_file(path_text: str | None) -> list[str] | None:
    if path_text is None:
        return None
    path = Path(path_text)
    rows: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        token = raw.strip()
        if not token or token.startswith("#"):
            continue
        rows.append(token)
    if not rows:
        raise ValueError(f"assets file is empty: {path}")
    return sorted(set(rows))


if __name__ == "__main__":
    raise SystemExit(main())
