from __future__ import annotations

import argparse
import datetime
import math
import re
import sys
from pathlib import Path

import pandas as pd

from alpha_lab.exceptions import AlphaLabConfigError

import alpha_lab.registry as _registry
from alpha_lab.config import PROCESSED_DATA_DIR
from alpha_lab.data_validation import validate_price_panel
from alpha_lab.experiment import run_factor_experiment
from alpha_lab.factors.low_volatility import low_volatility
from alpha_lab.factors.momentum import momentum
from alpha_lab.factors.reversal import reversal
from alpha_lab.obsidian import write_obsidian_note
from alpha_lab.reporting import (
    export_summary_csv,
    summarise_experiment_result,
    to_obsidian_markdown,
)
from alpha_lab.research_evaluation_config import (
    AVAILABLE_RESEARCH_EVALUATION_PROFILES,
    get_research_evaluation_config,
    get_research_evaluation_profile_intent,
)

# ---------------------------------------------------------------------------
# Factor registry
# ---------------------------------------------------------------------------

# Columns the input CSV must contain for any currently supported factor.
REQUIRED_PRICE_COLUMNS: frozenset[str] = frozenset({"date", "asset", "close"})

# Supported factor names (used for argparse choices and dispatch).
SUPPORTED_FACTORS: frozenset[str] = frozenset({"momentum", "reversal", "low_volatility"})
_UNIFIED_TOP_LEVEL_COMMANDS: frozenset[str] = frozenset(
    {"run", "real-case", "campaign", "bridge", "experimental", "profiles", "web", "data"}
)
_SUPPORTED_CAMPAIGNS: frozenset[str] = frozenset({"research_campaign_1"})


def _build_factor_fn(
    factor: str,
    *,
    momentum_window: int,
    reversal_window: int,
    low_volatility_window: int,
) -> object:
    """Return a callable ``(prices) -> factor_df`` for the requested factor."""
    if factor == "momentum":
        return lambda prices: momentum(prices, window=momentum_window)
    if factor == "reversal":
        return lambda prices: reversal(prices, window=reversal_window)
    if factor == "low_volatility":
        return lambda prices: low_volatility(prices, window=low_volatility_window)
    # argparse choices= guards this path; kept for explicit safety.
    raise AlphaLabConfigError(f"Unknown factor {factor!r}.  Supported: {sorted(SUPPORTED_FACTORS)}")


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
            "Run a Level 1/2 factor-research experiment using the alpha-lab pipeline.  "
            "Writes a summary CSV and optionally Obsidian markdown and a "
            "registry entry.  This is a thin CLI wrapper over the existing "
            "research modules and does not run execution replay."
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
    p.add_argument(
        "--reversal-window",
        type=int,
        default=5,
        help="Look-back window (per-asset rows) for the reversal factor.",
    )
    p.add_argument(
        "--low-volatility-window",
        type=int,
        default=20,
        help="Rolling return-volatility window (per-asset rows) for the low-volatility factor.",
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
        default=str(PROCESSED_DATA_DIR / "output"),
        help=(
            "Directory for summary CSV output.  Created if it does not exist.  "
            "Defaults to <project_root>/data/processed/output to keep artifacts "
            "inside the project regardless of current working directory."
        ),
    )
    p.add_argument(
        "--append-registry",
        action="store_true",
        help=(
            "Append this experiment to the CSV registry at data/processed/experiment_registry.csv."
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
            "Allow overwriting an existing Obsidian note.  By default the CLI refuses to overwrite."
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
        raise AlphaLabConfigError(
            f"experiment name {name!r} is not safe for use as a filename.  "
            "Use only letters, digits, hyphens, underscores, and dots."
        )
    return name


def _load_prices(input_path: Path) -> pd.DataFrame:
    """Read and minimally validate the input price CSV."""
    try:
        df = pd.read_csv(input_path)
    except Exception as exc:
        raise SystemExit(f"Error: could not read input file {input_path!s}: {exc}") from exc

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


def _legacy_main(argv: list[str] | None = None) -> int:
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
    if args.reversal_window <= 0:
        parser.error("--reversal-window must be a positive integer")
    if args.low_volatility_window <= 0:
        parser.error("--low-volatility-window must be a positive integer")
    if args.cost_rate is not None and args.cost_rate < 0:
        parser.error("--cost-rate must be >= 0")

    # --- Validate split arguments ---
    if (args.train_end is None) != (args.test_start is None):
        parser.error("--train-end and --test-start must be provided together or not at all")

    # --- Load data ---
    prices = _load_prices(Path(args.input_path))
    try:
        validate_price_panel(prices)
    except ValueError as exc:
        raise SystemExit(f"Error: invalid price data: {exc}") from exc

    # --- Derive and validate experiment name ---
    experiment_name: str = args.experiment_name or (
        f"{args.factor}_h{args.label_horizon}_q{args.quantiles}"
    )
    try:
        _safe_filename(experiment_name)
    except ValueError as exc:
        parser.error(str(exc))

    # --- Build factor function ---
    factor_fn = _build_factor_fn(
        args.factor,
        momentum_window=args.momentum_window,
        reversal_window=args.reversal_window,
        low_volatility_window=args.low_volatility_window,
    )

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


def build_unified_parser() -> argparse.ArgumentParser:
    """Return the unified top-level router parser."""
    parser = argparse.ArgumentParser(
        prog="alpha-lab",
        description=(
            "Unified routing CLI for Level 1/2 research workflows "
            "(Level 1 evaluation -> campaign triage -> Level 2 promotion gate -> "
            "Level 2 portfolio validation). Research-bridge project packaging "
            "is available under 'bridge'. "
            "Execution replay/implementability commands are available only under "
            "'experimental'. Use 'alpha-lab profiles' to list available evaluation "
            "profiles."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    top = parser.add_subparsers(dest="top_command", required=True)

    run = top.add_parser(
        "run",
        help="Run the legacy single-experiment CLI (backward-compatible route).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    run.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to the legacy run CLI (use 'alpha-lab run --help').",
    )

    real_case = top.add_parser(
        "real-case",
        help="Run real-case Level 1/2 research-validation workflows.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    real_case_kinds = real_case.add_subparsers(dest="real_case_kind", required=True)

    single_factor = real_case_kinds.add_parser(
        "single-factor",
        help="Route to the single-factor real-case CLI (supports --evaluation-profile).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    single_factor.add_argument("action", choices=["run"])
    single_factor.add_argument("args", nargs=argparse.REMAINDER)

    composite = real_case_kinds.add_parser(
        "composite",
        help="Route to the composite real-case CLI (supports --evaluation-profile).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    composite.add_argument("action", choices=["run"])
    composite.add_argument("args", nargs=argparse.REMAINDER)

    model_factor = real_case_kinds.add_parser(
        "model-factor",
        help="Route to the model-factor real-case CLI (supports --evaluation-profile).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    model_factor.add_argument("action", choices=["run"])
    model_factor.add_argument("args", nargs=argparse.REMAINDER)

    campaign = top.add_parser(
        "campaign",
        help="Run research campaign workflows.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    campaign_commands = campaign.add_subparsers(dest="campaign_action", required=True)
    campaign_run = campaign_commands.add_parser(
        "run",
        help=("Run one supported campaign workflow. Forwarded flags include --evaluation-profile."),
        description=(
            "Run a campaign workflow by name and pass campaign-specific flags after "
            "the campaign name. Example: "
            "'alpha-lab campaign run research_campaign_1 --evaluation-profile "
            "default_research'."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    campaign_run.add_argument("campaign_name", choices=sorted(_SUPPORTED_CAMPAIGNS))
    campaign_run.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to the campaign runner.",
    )
    campaign_compare = campaign_commands.add_parser(
        "compare-profiles",
        help="Compare campaign outcomes across multiple Level 1/2 evaluation profiles.",
        description=(
            "Run profile-aware campaign comparison in Level 1/2 mode. "
            "Use '--source example' for the built-in lightweight deterministic "
            "campaign, or '--source campaign' to compare an existing campaign "
            "definition under multiple evaluation profiles."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    campaign_compare.add_argument(
        "--source",
        choices=["example", "campaign"],
        default="example",
        help=(
            "Comparison source. "
            "'example' runs the built-in compact campaign example. "
            "'campaign' runs the campaign config provided by --campaign-config."
        ),
    )
    campaign_compare.add_argument(
        "--campaign-config",
        default="configs/campaigns/research_campaign_1/campaign.yaml",
        help=(
            "Campaign manifest YAML/JSON used when --source=campaign. "
            "Ignored when --source=example."
        ),
    )
    campaign_compare.add_argument(
        "--output-root-dir",
        default="dist/examples/profile_aware_campaign_level12",
        help="Output directory for profile runs and campaign profile comparison artifacts.",
    )
    campaign_compare.add_argument(
        "--profiles",
        nargs="+",
        default=["exploratory_screening", "default_research", "stricter_research"],
        choices=sorted(AVAILABLE_RESEARCH_EVALUATION_PROFILES),
        help="Evaluation profiles to compare for the same campaign/cases.",
    )
    campaign_compare.add_argument(
        "--pair-mode",
        choices=["adjacent", "all_pairs"],
        default="adjacent",
        help=(
            "Profile-pair coverage for transition/reason delta matrices. "
            "`adjacent` keeps neighboring profile pairs only; "
            "`all_pairs` includes non-adjacent ordered pairs."
        ),
    )
    campaign_compare.add_argument(
        "--case-output-root-dir",
        default=None,
        help=(
            "Optional root directory for per-case outputs when --source=campaign. "
            "When omitted, cases are written under each profile run output directory."
        ),
    )
    campaign_compare.add_argument(
        "--artifact-hint-path-mode",
        choices=["relative", "absolute"],
        default="relative",
        help=(
            "Render artifact pointer hints as paths relative to --output-root-dir "
            "(`relative`) or keep absolute filesystem paths (`absolute`)."
        ),
    )
    campaign_compare.add_argument(
        "--no-render-report",
        action="store_true",
        help=(
            "Skip rendering report markdown. "
            "For --source=example this skips per-case case_report.md; "
            "for --source=campaign this skips per-profile campaign_report.md."
        ),
    )
    campaign_compare.add_argument(
        "--render-overwrite",
        action="store_true",
        help="Allow overwriting existing report markdown when rendering is enabled.",
    )
    campaign_compare.add_argument(
        "--no-clean-output",
        action="store_true",
        help="Keep existing output directory content and append/overwrite in place.",
    )
    campaign_compare.add_argument(
        "--show-case-evidence",
        default=None,
        metavar="CASE_NAME",
        help=(
            "After comparison completes, print a compact case_evidence_index drill-down "
            "for the specified case name."
        ),
    )
    campaign_dashboard = campaign_commands.add_parser(
        "render-dashboard",
        help=(
            "Render a local factor-first research dashboard from campaign profile "
            "comparison artifacts."
        ),
        description=(
            "Render a local quant research workbench dashboard from "
            "campaign_profile_comparison.json with factor discovery, signal "
            "validation, portfolio construction, and backtest evaluation views."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    campaign_dashboard.add_argument(
        "--comparison-json",
        default="dist/examples/profile_aware_campaign_level12/campaign_profile_comparison.json",
        help="Path to campaign_profile_comparison.json.",
    )
    campaign_dashboard.add_argument(
        "--output-html",
        default=None,
        help=(
            "Optional output HTML path. "
            "When omitted, writes campaign_profile_dashboard_zh.html beside "
            "--comparison-json."
        ),
    )
    campaign_dashboard.add_argument(
        "--title",
        default=None,
        help="Optional dashboard title override.",
    )
    campaign_dashboard.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing output HTML file.",
    )
    campaign_dashboard.add_argument(
        "--artifact-load-mode",
        choices=["permissive", "strict"],
        default="permissive",
        help=(
            "Artifact loading mode for dashboard rendering. "
            "`permissive` keeps canonical-first fallback behavior; "
            "`strict` requires valid persisted canonical/workflow artifacts."
        ),
    )

    profiles = top.add_parser(
        "profiles",
        help="List available research evaluation profiles with usage guidance.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    profiles.add_argument(
        "action",
        nargs="?",
        choices=["list"],
        default="list",
        help="Profile command action.",
    )

    web = top.add_parser(
        "web",
        help="Start a local browser-based UI for single-factor real-case runs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    web_commands = web.add_subparsers(dest="web_action", required=True)
    web_ui = web_commands.add_parser(
        "ui",
        help="Run local web UI server and operate via browser.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    web_ui.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind host for the local web server.",
    )
    web_ui.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Bind port for the local web server.",
    )
    web_ui.add_argument(
        "--workspace-root",
        default=".",
        help="Workspace root used for resolving dist/web_ui_* directories.",
    )
    web_ui.add_argument(
        "--no-open-browser",
        action="store_true",
        help="Do not auto-open browser; print URL only.",
    )
    web_cockpit = web_commands.add_parser(
        "cockpit",
        help=(
            "Run the local research cockpit UI for project-round-case-run-writeback workflows."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    web_cockpit.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind host for the local web server.",
    )
    web_cockpit.add_argument(
        "--port",
        type=int,
        default=8766,
        help="Bind port for the local web server.",
    )
    web_cockpit.add_argument(
        "--workspace-root",
        default=".",
        help="Workspace root used for resolving dist/cockpit_* directories.",
    )
    web_cockpit.add_argument(
        "--vault-root",
        default=None,
        help="Quant-knowledge vault root. Defaults to OBSIDIAN_VAULT_PATH.",
    )
    web_cockpit.add_argument(
        "--no-open-browser",
        action="store_true",
        help="Do not auto-open browser; print URL only.",
    )

    bridge = top.add_parser(
        "bridge",
        help="Manage project-level research packs for quant-knowledge and ChatGPT Projects.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    from alpha_lab.research_bridge.cli import build_bridge_parser

    build_bridge_parser(bridge)

    data = top.add_parser(
        "data",
        help="Manage external Level 1/2 research datasets and export case inputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    from alpha_lab.data_store.cli import build_data_parser

    build_data_parser(data)

    experimental = top.add_parser(
        "experimental",
        help="Run future Level 3 replay/implementability workflows.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    experimental_commands = experimental.add_subparsers(
        dest="experimental_command",
        required=True,
    )
    experimental_single_factor = experimental_commands.add_parser(
        "single-factor-package",
        help="Run the experimental Level 3 single-factor replay package flow.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    experimental_single_factor.add_argument("action", choices=["run"])
    experimental_single_factor.add_argument("args", nargs=argparse.REMAINDER)
    experimental_execution_realism = experimental_commands.add_parser(
        "execution-realism-package",
        help="Run the experimental Level 3 execution-realism regression package flow.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    experimental_execution_realism.add_argument("action", choices=["run"])
    experimental_execution_realism.add_argument("args", nargs=argparse.REMAINDER)
    experimental_factor_health = experimental_commands.add_parser(
        "factor-health-monitor",
        help="Run the experimental factor lifecycle health monitor flow.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    experimental_factor_health.add_argument("action", choices=["run"])
    experimental_factor_health.add_argument("args", nargs=argparse.REMAINDER)
    experimental_vault_export_gate = experimental_commands.add_parser(
        "vault-export-gate",
        help="Detect or apply pending vault exports from alpha-lab output manifests.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    experimental_vault_export_gate.add_argument("action", choices=["detect", "apply"])
    experimental_vault_export_gate.add_argument("args", nargs=argparse.REMAINDER)

    return parser


def unified_main(argv: list[str] | None = None) -> int:
    """Route unified CLI commands to existing module entrypoints."""
    parser = build_unified_parser()
    args = parser.parse_args(argv)

    if args.top_command == "run":
        return _legacy_main(args.args)

    if args.top_command == "real-case":
        forwarded = [args.action, *args.args]
        if args.real_case_kind == "single-factor":
            from alpha_lab.real_cases.single_factor.cli import main as single_factor_main

            return single_factor_main(forwarded)

        if args.real_case_kind == "composite":
            from alpha_lab.real_cases.composite.cli import main as composite_main

            return composite_main(forwarded)

        if args.real_case_kind == "model-factor":
            from alpha_lab.real_cases.model_factor.cli import main as model_factor_main

            return model_factor_main(forwarded)

        parser.error(f"unsupported real-case workflow: {args.real_case_kind!r}")

    if args.top_command == "campaign":
        if args.campaign_action == "run":
            if args.campaign_name not in _SUPPORTED_CAMPAIGNS:
                parser.error(
                    "unsupported campaign name "
                    f"{args.campaign_name!r}; supported: {sorted(_SUPPORTED_CAMPAIGNS)}"
                )

            from alpha_lab.campaigns.research_campaign_1 import (
                main as research_campaign_1_main,
            )

            return research_campaign_1_main(args.args)

        if args.campaign_action == "compare-profiles":
            from alpha_lab.campaigns.profile_comparison import (
                print_campaign_profile_case_evidence,
                print_campaign_profile_comparison_summary,
                run_campaign_profile_comparison,
            )

            try:
                comparison_result = run_campaign_profile_comparison(
                    source=args.source,
                    output_root_dir=args.output_root_dir,
                    profiles=tuple(args.profiles),
                    pair_mode=args.pair_mode,
                    campaign_config=args.campaign_config,
                    case_output_root_dir=args.case_output_root_dir,
                    artifact_hint_path_mode=args.artifact_hint_path_mode,
                    render_report=not bool(args.no_render_report),
                    render_overwrite=bool(args.render_overwrite),
                    clean_output=not bool(args.no_clean_output),
                )
            except (ValueError, FileNotFoundError, RuntimeError) as exc:
                parser.error(str(exc))
            print_campaign_profile_comparison_summary(comparison_result)
            if args.show_case_evidence is not None:
                try:
                    print_campaign_profile_case_evidence(
                        comparison_result,
                        case_name=str(args.show_case_evidence),
                    )
                except ValueError as exc:
                    parser.error(str(exc))
            return 0

        if args.campaign_action == "render-dashboard":
            from alpha_lab.reporting.renderers import (
                write_campaign_profile_dashboard_html,
            )

            try:
                dashboard_path = write_campaign_profile_dashboard_html(
                    args.comparison_json,
                    output_path=args.output_html,
                    overwrite=bool(args.overwrite),
                    title=args.title,
                    artifact_load_mode=args.artifact_load_mode,
                )
            except (ValueError, FileNotFoundError, RuntimeError, OSError) as exc:
                parser.error(str(exc))
            print("")
            print("  Workflow : campaign-render-dashboard")
            print("  Status   : success")
            print(f"  Input    : {Path(args.comparison_json).resolve()}")
            print(f"  Output   : {dashboard_path}")
            return 0

        parser.error(f"unsupported campaign command: {args.campaign_action!r}")

    if args.top_command == "profiles":
        _print_profiles()
        return 0

    if args.top_command == "web":
        if args.port <= 0 or args.port > 65535:
            parser.error("--port must be within 1..65535")
        if args.web_action == "ui":
            from alpha_lab.web_ui import start_web_ui_server

            try:
                start_web_ui_server(
                    host=args.host,
                    port=args.port,
                    workspace_root=args.workspace_root,
                    open_browser=not bool(args.no_open_browser),
                )
            except OSError as exc:
                parser.error(str(exc))
            return 0
        if args.web_action == "cockpit":
            from alpha_lab.web_cockpit import start_web_cockpit_server

            try:
                start_web_cockpit_server(
                    host=args.host,
                    port=args.port,
                    workspace_root=args.workspace_root,
                    vault_root=args.vault_root,
                    open_browser=not bool(args.no_open_browser),
                )
            except OSError as exc:
                parser.error(str(exc))
            return 0
        parser.error(f"unsupported web command: {args.web_action!r}")
        return 0

    if args.top_command == "bridge":
        from alpha_lab.research_bridge.cli import main as bridge_main
        from alpha_lab.research_bridge.cli import resolved_argv_for_bridge

        return bridge_main(resolved_argv_for_bridge(args))

    if args.top_command == "data":
        from alpha_lab.data_store.cli import main as data_main

        return data_main(resolved_argv_for_data(args))

    if args.top_command == "experimental":
        if args.experimental_command == "single-factor-package":
            from alpha_lab.experimental_level3.single_factor_package import (
                main as single_factor_package_main,
            )

            return single_factor_package_main(args.args)
        if args.experimental_command == "execution-realism-package":
            from alpha_lab.experimental_level3.execution_realism_package import (
                main as execution_realism_package_main,
            )

            return execution_realism_package_main(args.args)
        if args.experimental_command == "factor-health-monitor":
            from alpha_lab.experimental_level3.factor_health_monitor import (
                main as factor_health_monitor_main,
            )

            return factor_health_monitor_main(args.args)
        if args.experimental_command == "vault-export-gate":
            from alpha_lab.vault_export_gate import main as vault_export_gate_main

            return vault_export_gate_main([args.action, *args.args])
        parser.error(f"unsupported experimental command: {args.experimental_command!r}")

    parser.error(f"unsupported top-level command: {args.top_command!r}")


def resolved_argv_for_data(args: argparse.Namespace) -> list[str]:
    forwarded: list[str] = [str(args.data_action)]
    if args.data_action in {"ingest", "update"}:
        forwarded.extend([str(args.vendor), str(args.dataset_scope)])
    for key, value in vars(args).items():
        if key in {
            "top_command",
            "data_action",
            "vendor",
            "dataset_scope",
        }:
            continue
        if value is None:
            continue
        if isinstance(value, bool):
            if value:
                forwarded.append(f"--{key.replace('_', '-')}")
            continue
        if isinstance(value, list):
            if not value:
                continue
            forwarded.append(f"--{key.replace('_', '-')}")
            forwarded.extend(str(item) for item in value)
            continue
        forwarded.extend([f"--{key.replace('_', '-')}", str(value)])
    return forwarded


def main(argv: list[str] | None = None) -> int:
    """Entrypoint that preserves the legacy CLI and adds unified routing."""
    resolved_argv = list(sys.argv[1:] if argv is None else argv)
    if not resolved_argv:
        return unified_main(["--help"])
    if resolved_argv[0] in {"-h", "--help"}:
        return unified_main(resolved_argv)
    if resolved_argv[0] == "run":
        return _legacy_main(resolved_argv[1:])
    if resolved_argv[0] in _UNIFIED_TOP_LEVEL_COMMANDS:
        return unified_main(resolved_argv)
    return _legacy_main(resolved_argv)


def _print_profiles() -> None:
    """Print available Level 1/2 evaluation profiles and key settings."""
    print("")
    print("  Available Evaluation Profiles (Level 1/2):")
    for name in AVAILABLE_RESEARCH_EVALUATION_PROFILES:
        config = get_research_evaluation_config(name)
        snapshot = config.to_audit_snapshot()
        factor_verdict = snapshot["factor_verdict"]
        campaign_triage = snapshot["campaign_triage"]
        uncertainty = snapshot["uncertainty"]
        promotion = snapshot["level2_promotion"]
        portfolio_validation = snapshot["level2_portfolio_validation"]
        print(f"  - {name}")
        print(f"      intent={get_research_evaluation_profile_intent(name)}")
        print(
            "      factor_verdict.min_eval_dates_basic="
            f"{factor_verdict.get('min_eval_dates_basic')}"
        )
        print(
            "      campaign_triage.min_rolling_positive_share_stable="
            f"{campaign_triage.get('min_rolling_positive_share_stable')}"
        )
        print(f"      uncertainty.method={uncertainty.get('method')}")
        print(
            f"      level2_promotion.min_valid_ratio_block={promotion.get('min_valid_ratio_block')}"
        )
        print(
            "      level2_portfolio_validation.max_mean_turnover_warn="
            f"{portfolio_validation.get('max_mean_turnover_warn')}"
        )


if __name__ == "__main__":
    sys.exit(main())
