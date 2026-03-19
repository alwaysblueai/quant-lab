from __future__ import annotations

import argparse
import datetime
import math
import re
import sys
from pathlib import Path

import pandas as pd

import alpha_lab.registry as _registry
from alpha_lab.experiment import run_factor_experiment
from alpha_lab.factors.momentum import momentum
from alpha_lab.obsidian import write_obsidian_note
from alpha_lab.reporting import (
    export_summary_csv,
    summarise_experiment_result,
    to_obsidian_markdown,
)

# ---------------------------------------------------------------------------
# Factor registry
# ---------------------------------------------------------------------------

# Columns the input CSV must contain for any currently supported factor.
REQUIRED_PRICE_COLUMNS: frozenset[str] = frozenset({"date", "asset", "close"})

# Supported factor names (used for argparse choices and dispatch).
SUPPORTED_FACTORS: frozenset[str] = frozenset({"momentum"})


def _build_factor_fn(
    factor: str,
    *,
    momentum_window: int,
) -> object:
    """Return a callable ``(prices) -> factor_df`` for the requested factor."""
    if factor == "momentum":
        return lambda prices: momentum(prices, window=momentum_window)
    # argparse choices= guards this path; kept for explicit safety.
    raise ValueError(
        f"Unknown factor {factor!r}.  Supported: {sorted(SUPPORTED_FACTORS)}"
    )


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Return the CLI argument parser.

    Separated from ``main`` so tests can inspect parser behaviour directly.
    """
    p = argparse.ArgumentParser(
        prog="run_experiment",
        description=(
            "Run a factor experiment using the alpha-lab pipeline.  "
            "Writes a summary CSV and optionally Obsidian markdown and a "
            "registry entry.  This is a thin CLI wrapper over the existing "
            "research modules — it does not redesign the pipeline."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Required ---
    p.add_argument(
        "--input-path",
        required=True,
        help=(
            "Path to the input price CSV.  Must contain columns: "
            "date (parseable datetime), asset (str), close (float)."
        ),
    )
    p.add_argument(
        "--factor",
        required=True,
        choices=sorted(SUPPORTED_FACTORS),
        help="Factor to compute.",
    )
    p.add_argument(
        "--label-horizon",
        required=True,
        type=int,
        help="Forward-return horizon in per-asset rows (>= 1).",
    )
    p.add_argument(
        "--quantiles",
        required=True,
        type=int,
        help="Number of quantile buckets (>= 2).",
    )

    # --- Factor-specific ---
    p.add_argument(
        "--momentum-window",
        type=int,
        default=20,
        help="Look-back window (per-asset rows) for the momentum factor.",
    )

    # --- Split ---
    p.add_argument(
        "--train-end",
        default=None,
        metavar="YYYY-MM-DD",
        help=(
            "Last inclusive date of the training period.  "
            "Must be provided together with --test-start."
        ),
    )
    p.add_argument(
        "--test-start",
        default=None,
        metavar="YYYY-MM-DD",
        help=(
            "First inclusive date of the evaluation period.  "
            "Must be provided together with --train-end."
        ),
    )

    # --- Cost ---
    p.add_argument(
        "--cost-rate",
        type=float,
        default=None,
        metavar="RATE",
        help=(
            "One-way transaction cost rate (e.g. 0.001 for 10 bps).  "
            "When provided, mean_cost_adjusted_long_short_return is included "
            "in the summary.  Minimal research estimate only."
        ),
    )

    # --- Output ---
    p.add_argument(
        "--experiment-name",
        default=None,
        help=(
            "Human-readable experiment identifier used in output filenames "
            "and the registry.  Defaults to "
            "'{factor}_h{horizon}_q{quantiles}'."
        ),
    )
    p.add_argument(
        "--output-dir",
        default="output",
        help="Directory for summary CSV output.  Created if it does not exist.",
    )
    p.add_argument(
        "--append-registry",
        action="store_true",
        help=(
            "Append this experiment to the CSV registry at "
            "data/processed/experiment_registry.csv."
        ),
    )
    p.add_argument(
        "--obsidian-markdown-path",
        default=None,
        metavar="PATH",
        help=(
            "If set, write an Obsidian-friendly markdown note to this path.  "
            "If PATH ends with '/' or is an existing directory, a filename is "
            "generated automatically as YYYY-MM-DD_{experiment_name}.md.  "
            "Parent directories are created automatically."
        ),
    )
    p.add_argument(
        "--obsidian-overwrite",
        action="store_true",
        help=(
            "Allow overwriting an existing Obsidian note.  "
            "By default the CLI refuses to overwrite."
        ),
    )

    return p


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# Only letters, digits, hyphens, underscores, and dots are allowed in a name
# used as part of an output filename.  This prevents path-separator injection
# and directory traversal via --experiment-name.
_SAFE_FILENAME_RE = re.compile(r"^[A-Za-z0-9_\-.]+$")


def _safe_filename(name: str) -> str:
    """Return *name* unchanged if it is safe for use as a filename component.

    Raises
    ------
    ValueError
        If *name* contains characters that could alter the output path (path
        separators, ``..``, etc.).
    """
    if not _SAFE_FILENAME_RE.match(name) or name in (".", ".."):
        raise ValueError(
            f"experiment name {name!r} is not safe for use as a filename.  "
            "Use only letters, digits, hyphens, underscores, and dots."
        )
    return name


def _load_prices(input_path: Path) -> pd.DataFrame:
    """Read and minimally validate the input price CSV."""
    try:
        df = pd.read_csv(input_path)
    except Exception as exc:
        raise SystemExit(
            f"Error: could not read input file {input_path!s}: {exc}"
        ) from exc

    missing = REQUIRED_PRICE_COLUMNS - set(df.columns)
    if missing:
        raise SystemExit(
            f"Error: input CSV is missing required columns: {sorted(missing)}.\n"
            f"Required for all factors: {sorted(REQUIRED_PRICE_COLUMNS)}."
        )
    # Coerce date column explicitly so that unparseable strings become NaT
    # rather than being silently passed through as object dtype.
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        raise SystemExit(
            "Error: input CSV 'date' column contains unparseable values.  "
            "Ensure all dates are in a parseable format (e.g. YYYY-MM-DD)."
        )
    return df


def _fmt_float(value: float) -> str:
    if math.isnan(value):
        return "—"
    return f"{value:.4f}"


def _print_stdout_summary(
    experiment_name: str,
    summary: pd.DataFrame,
    summary_csv_path: Path,
) -> None:
    row = summary.iloc[0]
    lines = [
        "",
        f"  Experiment : {experiment_name}",
        f"  Factor     : {row['factor_name']}",
        f"  Label      : {row['label_name']}",
        f"  Mean IC    : {_fmt_float(float(row['mean_ic']))}",
        f"  Mean Rank IC : {_fmt_float(float(row['mean_rank_ic']))}",
        f"  Mean L/S Return : {_fmt_float(float(row['mean_long_short_return']))}",
        f"  Mean Cost-Adj L/S Return : "
        f"{_fmt_float(float(row['mean_cost_adjusted_long_short_return']))}",
        "",
        f"  Summary CSV : {summary_csv_path}",
    ]
    print("\n".join(lines))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Run a factor experiment from the command line.

    Parameters
    ----------
    argv:
        Argument list.  ``None`` means read from ``sys.argv[1:]``, which is
        the normal CLI usage.  Pass an explicit list for in-process testing.

    Returns
    -------
    int
        Exit code: 0 on success.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    # --- Validate numeric arguments ---
    if args.label_horizon <= 0:
        parser.error("--label-horizon must be a positive integer")
    if args.quantiles < 2:
        parser.error("--quantiles must be at least 2")
    if args.momentum_window <= 0:
        parser.error("--momentum-window must be a positive integer")
    if args.cost_rate is not None and args.cost_rate < 0:
        parser.error("--cost-rate must be >= 0")

    # --- Validate split arguments ---
    if (args.train_end is None) != (args.test_start is None):
        parser.error(
            "--train-end and --test-start must be provided together or not at all"
        )

    # --- Load data ---
    prices = _load_prices(Path(args.input_path))

    # --- Derive and validate experiment name ---
    experiment_name: str = args.experiment_name or (
        f"{args.factor}_h{args.label_horizon}_q{args.quantiles}"
    )
    try:
        _safe_filename(experiment_name)
    except ValueError as exc:
        parser.error(str(exc))

    # --- Build factor function ---
    factor_fn = _build_factor_fn(args.factor, momentum_window=args.momentum_window)

    # --- Run pipeline ---
    result = run_factor_experiment(
        prices,
        factor_fn,  # type: ignore[arg-type]
        horizon=args.label_horizon,
        n_quantiles=args.quantiles,
        train_end=args.train_end,
        test_start=args.test_start,
    )

    # --- Summarise ---
    summary = summarise_experiment_result(result, cost_rate=args.cost_rate)

    # --- Write summary CSV ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"{experiment_name}_summary.csv"
    export_summary_csv(summary, summary_path)

    # --- Stdout ---
    _print_stdout_summary(experiment_name, summary, summary_path)

    # --- Optional Obsidian markdown ---
    # Resolve the actual file path first (directory → auto filename).
    resolved_md_path: Path | None = None
    if args.obsidian_markdown_path:
        md_arg_str: str = args.obsidian_markdown_path
        md_arg = Path(md_arg_str)
        if md_arg_str.endswith(("/", "\\")) or md_arg.is_dir():
            date_str = datetime.date.today().isoformat()
            auto_name = f"{date_str}_{experiment_name}.md"
            resolved_md_path = md_arg / auto_name
        else:
            resolved_md_path = md_arg

        md = to_obsidian_markdown(
            result,
            title=experiment_name,
            cost_rate=args.cost_rate,
            horizon=args.label_horizon,
        )
        try:
            write_obsidian_note(md, resolved_md_path, overwrite=args.obsidian_overwrite)
        except FileExistsError as exc:
            parser.error(str(exc))
        print(f"  Obsidian   : {resolved_md_path}")

    # --- Optional registry ---
    if args.append_registry:
        # Access DEFAULT_REGISTRY_PATH at call time (not import time) so that
        # tests can monkeypatch the module attribute reliably.
        # Pass the resolved file path so the registry captures the actual note,
        # not the directory that was supplied when auto-naming is used.
        _registry.register_experiment(
            experiment_name,
            summary,
            _registry.DEFAULT_REGISTRY_PATH,
            obsidian_path=str(resolved_md_path) if resolved_md_path is not None else None,
        )
        print(f"  Registry   : appended '{experiment_name}'")

    return 0


if __name__ == "__main__":
    sys.exit(main())
