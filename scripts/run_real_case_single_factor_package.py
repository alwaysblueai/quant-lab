#!/usr/bin/env python3
"""Run the real A-share single-factor case from handoff to package outputs."""

from __future__ import annotations

import argparse
import datetime
import json
import re
import shutil
from dataclasses import dataclass
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

RUNS_ROOT_NAME = "real_single_factor_package_runs"
_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class RunPaths:
    case_output_dir: Path
    run_output_dir: Path
    replay_dir: Path
    vectorbt_dir: Path
    backtrader_dir: Path
    impact_dir: Path
    package_dir: Path
    summary_path: Path


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
        "--workflow-summary-path",
        default=None,
        help=(
            "Optional explicit workflow summary JSON path. "
            "If omitted, resolves deterministically from handoff manifest experiment_id "
            "or a single unambiguous *_workflow_summary.json under case-output-dir."
        ),
    )
    p.add_argument(
        "--run-id",
        default=None,
        help=(
            "Run identifier under "
            f"<case-output-dir>/{RUNS_ROOT_NAME}/<run-id>. "
            "Defaults to UTC timestamp YYYYMMDDTHHMMSSZ."
        ),
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


def _default_run_id() -> str:
    return datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")


def _resolve_path(path_value: str | Path, *, base_dir: Path) -> Path:
    candidate = Path(path_value)
    if not candidate.is_absolute():
        candidate = base_dir / candidate
    return candidate.resolve()


def _validate_run_id(raw_run_id: str) -> str:
    run_id = raw_run_id.strip()
    if not run_id:
        raise ValueError("run-id must be non-empty")
    if not _RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError(
            "run-id must match [A-Za-z0-9][A-Za-z0-9._-]* "
            "(no whitespace or path separators)"
        )
    return run_id


def _prepare_run_paths(case_output_dir: Path, run_id: str) -> RunPaths:
    case_dir = case_output_dir.resolve()
    run_output_dir = case_dir / RUNS_ROOT_NAME / run_id

    if run_output_dir.exists():
        if run_output_dir.is_dir():
            shutil.rmtree(run_output_dir)
        else:
            run_output_dir.unlink()

    replay_dir = run_output_dir / "replay_compare"
    vectorbt_dir = replay_dir / "vectorbt"
    backtrader_dir = replay_dir / "backtrader"
    impact_dir = run_output_dir / "execution_impact"
    package_dir = run_output_dir / "research_package"
    summary_path = run_output_dir / "real_single_factor_package_summary.json"

    return RunPaths(
        case_output_dir=case_dir,
        run_output_dir=run_output_dir,
        replay_dir=replay_dir,
        vectorbt_dir=vectorbt_dir,
        backtrader_dir=backtrader_dir,
        impact_dir=impact_dir,
        package_dir=package_dir,
        summary_path=summary_path,
    )


def _resolve_workflow_summary_path(
    *,
    case_output_dir: Path,
    handoff_bundle_path: Path,
    explicit_path: str | Path | None,
) -> Path:
    if explicit_path is not None:
        workflow_path = _resolve_path(explicit_path, base_dir=case_output_dir)
        if not workflow_path.exists():
            raise FileNotFoundError(f"workflow summary does not exist: {workflow_path}")
        return workflow_path

    manifest_path = handoff_bundle_path / "manifest.json"
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            raw_experiment_id = payload.get("experiment_id")
            if isinstance(raw_experiment_id, str) and raw_experiment_id.strip():
                candidate = (
                    case_output_dir / f"{raw_experiment_id.strip()}_single_factor_workflow_summary.json"
                )
                if candidate.exists():
                    return candidate.resolve()

    matches = sorted(case_output_dir.glob("*_workflow_summary.json"))
    if len(matches) == 1:
        return matches[0].resolve()
    if not matches:
        raise FileNotFoundError(
            "workflow summary not found; pass --workflow-summary-path explicitly"
        )
    names = ", ".join(path.name for path in matches)
    raise ValueError(
        "multiple workflow summaries found; pass --workflow-summary-path explicitly: "
        f"{names}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.initial_cash <= 0:
        parser.error("--initial-cash must be > 0")
    if args.commission_bps < 0:
        parser.error("--commission-bps must be >= 0")
    if args.slippage_bps < 0:
        parser.error("--slippage-bps must be >= 0")
    if args.skip_backtrader:
        parser.error("--skip-backtrader is not supported for package assembly")

    cwd = Path.cwd()
    case_output_dir = _resolve_path(args.case_output_dir, base_dir=cwd)
    handoff_bundle_path = _resolve_path(args.handoff_bundle, base_dir=cwd)
    price_path = _resolve_path(args.price_path, base_dir=cwd)

    run_id = _validate_run_id(args.run_id or _default_run_id())
    run_paths = _prepare_run_paths(case_output_dir, run_id)
    workflow_summary_path = _resolve_workflow_summary_path(
        case_output_dir=run_paths.case_output_dir,
        handoff_bundle_path=handoff_bundle_path,
        explicit_path=args.workflow_summary_path,
    )

    bundle = load_backtest_input_bundle(handoff_bundle_path)
    prices = pd.read_csv(price_path)

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
            output_dir=run_paths.vectorbt_dir,
            export_summary=True,
            export_target_weights=True,
            export_series=True,
        )
        run_vectorbt_backtest(bundle, vcfg)

    bcfg = BacktestRunConfig(
        price_df=prices,
        engine="backtrader",
        close_column="close",
        open_column=None,
        initial_cash=args.initial_cash,
        freq=args.freq,
        commission_bps=args.commission_bps,
        slippage_bps=args.slippage_bps,
        output_dir=run_paths.backtrader_dir,
        export_summary=True,
        export_target_weights=True,
        export_series=True,
    )
    run_backtrader_backtest(bundle, bcfg)

    impact = build_execution_impact_report(
        run_path=run_paths.backtrader_dir,
        comparison_run_path=(
            run_paths.vectorbt_dir if (run_paths.vectorbt_dir / "backtest_summary.json").exists() else None
        ),
    )
    impact_files = export_execution_impact_report(impact, output_dir=run_paths.impact_dir)

    replay_run_dirs: dict[str, Path] = {"backtrader": run_paths.backtrader_dir}
    if (run_paths.vectorbt_dir / "backtest_summary.json").exists():
        replay_run_dirs["vectorbt"] = run_paths.vectorbt_dir

    package = build_research_package(
        run_paths.run_output_dir,
        case_id=args.case_id,
        case_name=args.case_name,
        workflow_summary_path=workflow_summary_path,
        handoff_bundle_path=handoff_bundle_path,
        replay_run_dirs=replay_run_dirs,
        execution_impact_report_path=impact_files["report_json"],
    )
    package_files = export_research_package(package, output_dir=run_paths.package_dir)

    summary = {
        "run_id": run_id,
        "run_output_dir": str(run_paths.run_output_dir),
        "handoff_bundle": str(handoff_bundle_path),
        "workflow_summary_path": str(workflow_summary_path),
        "vectorbt_run_dir": str(run_paths.vectorbt_dir) if "vectorbt" in replay_run_dirs else None,
        "backtrader_run_dir": str(run_paths.backtrader_dir),
        "execution_impact_report": str(impact_files["report_json"]),
        "research_package_json": str(package_files["package_json"]),
        "research_package_markdown": str(package_files["package_markdown"]),
    }
    run_paths.summary_path.parent.mkdir(parents=True, exist_ok=True)
    run_paths.summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("")
    print("  Real Single-Factor Package Complete")
    for key, value in summary.items():
        print(f"  {key:>24} : {value}")
    print(f"  {'summary':>24} : {run_paths.summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
