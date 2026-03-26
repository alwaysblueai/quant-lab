#!/usr/bin/env python3
"""Run the real A-share single-factor case from handoff to package outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from alpha_lab.backtest_adapter.backtrader_adapter import run_backtrader_backtest
from alpha_lab.backtest_adapter.loader import load_backtest_input_bundle
from alpha_lab.backtest_adapter.schema import BacktestRunConfig
from alpha_lab.backtest_adapter.vectorbt_adapter import run_vectorbt_backtest
from alpha_lab.execution_impact_report import (
    build_execution_impact_report,
    export_execution_impact_report,
)
from alpha_lab.research_package import build_research_package, export_research_package


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_real_case_single_factor_package",
        description=(
            "Run real single-factor case artifacts: handoff -> vectorbt replay -> "
            "backtrader replay -> execution impact -> research package"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--handoff-bundle",
        default=(
            "data/processed/tushare_research_inputs/ashare_202401_202412/"
            "single_case_output/handoff/tushare_single_reversal_liquidity_screened"
        ),
        help="Path to handoff bundle directory.",
    )
    p.add_argument(
        "--price-path",
        default="data/processed/tushare_research_inputs/ashare_202401_202412/prices.csv",
        help="Path to long-form price panel CSV.",
    )
    p.add_argument(
        "--case-output-dir",
        default=(
            "data/processed/tushare_research_inputs/ashare_202401_202412/"
            "single_case_output"
        ),
        help="Root output directory for this case.",
    )
    p.add_argument(
        "--case-id",
        default="real_single_factor_ashare_reversal_5d",
        help="Stable ID used in research package metadata.",
    )
    p.add_argument(
        "--case-name",
        default="Real A-share Single Factor Reversal 5D",
        help="Human-friendly case name used in package markdown.",
    )
    p.add_argument("--initial-cash", type=float, default=1_000_000.0)
    p.add_argument("--commission-bps", type=float, default=0.0)
    p.add_argument("--slippage-bps", type=float, default=0.0)
    p.add_argument("--freq", default=None)
    p.add_argument(
        "--skip-vectorbt",
        action="store_true",
        help="Skip vectorbt replay stage.",
    )
    p.add_argument(
        "--skip-backtrader",
        action="store_true",
        help="Skip backtrader replay stage.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.initial_cash <= 0:
        parser.error("--initial-cash must be > 0")
    if args.commission_bps < 0:
        parser.error("--commission-bps must be >= 0")
    if args.slippage_bps < 0:
        parser.error("--slippage-bps must be >= 0")

    bundle = load_backtest_input_bundle(args.handoff_bundle)
    prices = pd.read_csv(args.price_path)

    case_output_dir = Path(args.case_output_dir).resolve()
    replay_dir = case_output_dir / "replay_compare"
    vectorbt_dir = replay_dir / "vectorbt"
    backtrader_dir = replay_dir / "backtrader"
    impact_dir = case_output_dir / "execution_impact"
    package_dir = case_output_dir / "research_package"

    if not args.skip_vectorbt:
        vcfg = BacktestRunConfig(
            price_df=prices,
            engine="vectorbt",
            close_column="close",
            open_column=None,
            initial_cash=args.initial_cash,
            freq=args.freq,
            commission_bps=args.commission_bps,
            slippage_bps=args.slippage_bps,
            output_dir=vectorbt_dir,
            export_summary=True,
            export_target_weights=True,
            export_series=True,
        )
        run_vectorbt_backtest(bundle, vcfg)

    if not args.skip_backtrader:
        bcfg = BacktestRunConfig(
            price_df=prices,
            engine="backtrader",
            close_column="close",
            open_column=None,
            initial_cash=args.initial_cash,
            freq=args.freq,
            commission_bps=args.commission_bps,
            slippage_bps=args.slippage_bps,
            output_dir=backtrader_dir,
            export_summary=True,
            export_target_weights=True,
            export_series=True,
        )
        run_backtrader_backtest(bundle, bcfg)

    impact = build_execution_impact_report(
        run_path=backtrader_dir,
        comparison_run_path=vectorbt_dir if vectorbt_dir.exists() else None,
    )
    impact_files = export_execution_impact_report(impact, impact_dir)

    package = build_research_package(
        case_output_dir,
        case_id=args.case_id,
        case_name=args.case_name,
    )
    package_files = export_research_package(package, package_dir)

    summary = {
        "handoff_bundle": str(Path(args.handoff_bundle).resolve()),
        "vectorbt_run_dir": str(vectorbt_dir),
        "backtrader_run_dir": str(backtrader_dir),
        "execution_impact_report": str(impact_files.get("report_json")),
        "research_package_json": str(package_files.get("json")),
        "research_package_markdown": str(package_files.get("markdown")),
    }
    summary_path = case_output_dir / "real_single_factor_package_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print("")
    print("  Real Single-Factor Package Complete")
    for key, value in summary.items():
        print(f"  {key:>24} : {value}")
    print(f"  {'summary':>24} : {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
