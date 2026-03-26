from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pandas as pd

from alpha_lab.cli import _build_factor_fn, _load_prices, _safe_filename
from alpha_lab.config import PROCESSED_DATA_DIR
from alpha_lab.data_validation import validate_price_panel
from alpha_lab.obsidian import write_obsidian_note
from alpha_lab.walk_forward import WalkForwardResult, run_walk_forward_experiment

SUPPORTED_FACTORS = frozenset({"momentum", "reversal", "low_volatility"})


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_walk_forward_experiment",
        description=(
            "Run a walk-forward factor experiment and write fold-level / aggregate "
            "artifacts for out-of-sample research."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input-path", required=True)
    p.add_argument("--factor", required=True, choices=sorted(SUPPORTED_FACTORS))
    p.add_argument("--label-horizon", required=True, type=int)
    p.add_argument("--quantiles", required=True, type=int)
    p.add_argument("--train-size", required=True, type=int, help="Training window in unique dates.")
    p.add_argument("--test-size", required=True, type=int, help="Test window in unique dates.")
    p.add_argument("--step", required=True, type=int, help="Date advance between folds.")
    p.add_argument("--val-size", default=0, type=int, help="Gap between train and test windows.")
    p.add_argument("--purge-periods", default=0, type=int, help="Metadata-only purge periods.")
    p.add_argument(
        "--embargo-periods",
        default=0,
        type=int,
        help="Metadata-only embargo periods.",
    )
    p.add_argument("--cost-rate", type=float, default=None, metavar="RATE")
    p.add_argument("--momentum-window", type=int, default=20)
    p.add_argument("--reversal-window", type=int, default=5)
    p.add_argument("--low-volatility-window", type=int, default=20)
    p.add_argument("--experiment-name", default=None)
    p.add_argument(
        "--output-dir",
        default=str(PROCESSED_DATA_DIR / "output"),
        help="Directory for aggregate and fold-summary CSVs.",
    )
    p.add_argument(
        "--obsidian-markdown-path",
        default=None,
        metavar="PATH",
        help="If set, write a walk-forward markdown note to this file path.",
    )
    p.add_argument("--obsidian-overwrite", action="store_true")
    return p


def _fmt_float(value: float) -> str:
    if math.isnan(value):
        return "—"
    return f"{value:.4f}"


def _walk_forward_markdown(
    *, experiment_name: str, wf: WalkForwardResult, factor: str, horizon: int
) -> str:
    agg = wf.aggregate_summary
    lines = [
        "---",
        "type: walk_forward_experiment",
        f"name: {experiment_name}",
        f"factor: {factor}",
        f"horizon: {horizon}",
        f"n_folds: {agg.n_folds}",
        "---",
        "",
        f"# {experiment_name}",
        "",
        "## Setup",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Factor | `{factor}` |",
        f"| Horizon | {horizon} bars |",
        f"| Folds | {agg.n_folds} |",
        "",
        "## Aggregate Results",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Mean fold IC | {_fmt_float(agg.mean_ic)} |",
        f"| Pooled IC mean | {_fmt_float(agg.pooled_ic_mean)} |",
        f"| Pooled IC IR | {_fmt_float(agg.pooled_ic_ir)} |",
        f"| Mean fold long-short return | {_fmt_float(agg.mean_long_short)} |",
        f"| Mean fold turnover | {_fmt_float(agg.mean_turnover)} |",
        "",
        "## Interpretation",
        "",
        "<!-- Add interpretation here -->",
        "",
        "## Next Steps",
        "",
        "<!-- Add next steps here -->",
        "",
        "## Open Questions",
        "",
        "<!-- Add open questions here -->",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    for name in (
        "label_horizon",
        "quantiles",
        "train_size",
        "test_size",
        "step",
        "momentum_window",
        "reversal_window",
        "low_volatility_window",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be a positive integer")
    if args.val_size < 0:
        parser.error("--val-size must be >= 0")
    if args.purge_periods < 0:
        parser.error("--purge-periods must be >= 0")
    if args.embargo_periods < 0:
        parser.error("--embargo-periods must be >= 0")
    if args.quantiles < 2:
        parser.error("--quantiles must be at least 2")
    if args.cost_rate is not None and args.cost_rate < 0:
        parser.error("--cost-rate must be >= 0")

    prices = _load_prices(Path(args.input_path))
    try:
        validate_price_panel(prices)
    except ValueError as exc:
        raise SystemExit(f"Error: invalid price data: {exc}") from exc

    experiment_name: str = args.experiment_name or (
        f"wf_{args.factor}_h{args.label_horizon}_q{args.quantiles}"
    )
    try:
        _safe_filename(experiment_name)
    except ValueError as exc:
        parser.error(str(exc))

    factor_fn = _build_factor_fn(
        args.factor,
        momentum_window=args.momentum_window,
        reversal_window=args.reversal_window,
        low_volatility_window=args.low_volatility_window,
    )

    wf = run_walk_forward_experiment(
        prices,
        factor_fn,  # type: ignore[arg-type]
        train_size=args.train_size,
        test_size=args.test_size,
        step=args.step,
        horizon=args.label_horizon,
        n_quantiles=args.quantiles,
        cost_rate=args.cost_rate,
        val_size=args.val_size,
        purge_periods=args.purge_periods,
        embargo_periods=args.embargo_periods,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    aggregate_path = out_dir / f"{experiment_name}_aggregate.csv"
    folds_path = out_dir / f"{experiment_name}_folds.csv"
    aggregate_df = pd.DataFrame(
        [
            {
                "experiment_name": experiment_name,
                "factor": args.factor,
                "label_horizon": args.label_horizon,
                "quantiles": args.quantiles,
                "train_size": args.train_size,
                "test_size": args.test_size,
                "step": args.step,
                "val_size": args.val_size,
                "n_folds": wf.aggregate_summary.n_folds,
                "mean_ic": wf.aggregate_summary.mean_ic,
                "pooled_ic_mean": wf.aggregate_summary.pooled_ic_mean,
                "pooled_ic_ir": wf.aggregate_summary.pooled_ic_ir,
                "mean_long_short": wf.aggregate_summary.mean_long_short,
                "mean_turnover": wf.aggregate_summary.mean_turnover,
                "mean_cost_adjusted_return": wf.aggregate_summary.mean_cost_adjusted_return,
            }
        ]
    )
    aggregate_df.to_csv(aggregate_path, index=False)
    wf.fold_summary_df.to_csv(folds_path, index=False)

    print("")
    print(f"  Experiment : {experiment_name}")
    print(f"  Factor     : {args.factor}")
    print(f"  Folds      : {wf.aggregate_summary.n_folds}")
    print(f"  Pooled IC  : {_fmt_float(wf.aggregate_summary.pooled_ic_mean)}")
    print(f"  Pooled IC IR : {_fmt_float(wf.aggregate_summary.pooled_ic_ir)}")
    print(f"  Aggregate CSV : {aggregate_path}")
    print(f"  Fold CSV      : {folds_path}")

    if args.obsidian_markdown_path:
        note = _walk_forward_markdown(
            experiment_name=experiment_name,
            wf=wf,
            factor=args.factor,
            horizon=args.label_horizon,
        )
        note_path = write_obsidian_note(
            note,
            args.obsidian_markdown_path,
            overwrite=args.obsidian_overwrite,
        )
        print(f"  Obsidian      : {note_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
