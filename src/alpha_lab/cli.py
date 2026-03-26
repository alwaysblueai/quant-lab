from __future__ import annotations

import argparse
import datetime
import json
import math
import re
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast

import pandas as pd

import alpha_lab.registry as _registry
from alpha_lab.alpha_registry import ALPHA_REGISTRY_PATH
from alpha_lab.config import PROCESSED_DATA_DIR
from alpha_lab.data_validation import validate_price_panel
from alpha_lab.experiment import run_factor_experiment
from alpha_lab.experiment_metadata import ExperimentMetadata, ValidationMetadata
from alpha_lab.factors.low_volatility import low_volatility
from alpha_lab.factors.momentum import momentum
from alpha_lab.factors.reversal import reversal
from alpha_lab.obsidian import write_obsidian_note
from alpha_lab.reporting import (
    export_summary_csv,
    summarise_experiment_result,
    to_obsidian_markdown,
)
from alpha_lab.research_templates import (
    CompositeDecisionThresholds,
    CompositeWorkflowResult,
    CompositeWorkflowSpec,
    NeutralizationSpec,
    SignalPreprocessSpec,
    SingleFactorDecisionThresholds,
    SingleFactorWorkflowResult,
    SingleFactorWorkflowSpec,
    run_composite_signal_research_workflow,
    run_single_factor_research_workflow,
)
from alpha_lab.research_universe import ResearchUniverseRules
from alpha_lab.timing import DelaySpec
from alpha_lab.trial_log import DEFAULT_TRIAL_LOG_PATH

# ---------------------------------------------------------------------------
# Factor registry
# ---------------------------------------------------------------------------

# Columns the input CSV must contain for any currently supported factor.
REQUIRED_PRICE_COLUMNS: frozenset[str] = frozenset({"date", "asset", "close"})

# Supported factor names (used for argparse choices and dispatch).
SUPPORTED_FACTORS: frozenset[str] = frozenset(
    {"momentum", "reversal", "low_volatility"}
)
WORKFLOW_COMMANDS: frozenset[str] = frozenset(
    {"run-single-factor", "run-composite"}
)


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
# Workflow CLI (thin wrapper over research_templates)
# ---------------------------------------------------------------------------


def build_workflow_parser() -> argparse.ArgumentParser:
    """Return parser for canonical workflow commands."""
    parser = argparse.ArgumentParser(
        prog="run_research_workflow",
        description=(
            "Run canonical single-factor/composite workflows from JSON config.  "
            "This CLI is a thin orchestration wrapper and does not add research logic."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    single = subparsers.add_parser(
        "run-single-factor",
        help="Run run_single_factor_research_workflow(...) from config.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_workflow_common_args(single)

    composite = subparsers.add_parser(
        "run-composite",
        help="Run run_composite_signal_research_workflow(...) from config.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_workflow_common_args(composite)
    return parser


def _add_workflow_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config-path",
        required=True,
        help="Path to JSON workflow config.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for deterministic workflow summary artifacts.",
    )

    trial_group = parser.add_mutually_exclusive_group()
    trial_group.add_argument(
        "--write-trial-log",
        dest="write_trial_log",
        action="store_true",
        default=None,
        help="Override config and enable trial log writes.",
    )
    trial_group.add_argument(
        "--no-write-trial-log",
        dest="write_trial_log",
        action="store_false",
        help="Override config and disable trial log writes.",
    )

    registry_group = parser.add_mutually_exclusive_group()
    registry_group.add_argument(
        "--update-registry",
        dest="update_registry",
        action="store_true",
        default=None,
        help="Override config and enable alpha registry updates.",
    )
    registry_group.add_argument(
        "--no-update-registry",
        dest="update_registry",
        action="store_false",
        help="Override config and disable alpha registry updates.",
    )

    handoff_group = parser.add_mutually_exclusive_group()
    handoff_group.add_argument(
        "--export-handoff",
        dest="export_handoff",
        action="store_true",
        default=None,
        help="Override config and enable strict external-backtest handoff export.",
    )
    handoff_group.add_argument(
        "--no-export-handoff",
        dest="export_handoff",
        action="store_false",
        help="Override config and disable handoff export.",
    )


def _is_workflow_command(argv: list[str]) -> bool:
    return bool(argv) and argv[0] in WORKFLOW_COMMANDS


def _load_json_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise ValueError(f"config file does not exist: {config_path}")
    if not config_path.is_file():
        raise ValueError(f"config path is not a file: {config_path}")
    try:
        text = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"failed to read config file {config_path}: {exc}") from exc
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"config file is not valid JSON ({config_path}:{exc.lineno}:{exc.colno}): "
            f"{exc.msg}"
        ) from exc
    if not isinstance(parsed, dict):
        raise ValueError("config root must be a JSON object")
    return cast(dict[str, Any], parsed)


def _as_mapping(value: object, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object")
    return cast(Mapping[str, Any], value)


def _string_key_dict(mapping: Mapping[str, Any], *, field_name: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, val in mapping.items():
        if not isinstance(key, str):
            raise ValueError(f"{field_name} must use string keys")
        out[key] = val
    return out


def _as_str_tuple(value: object, *, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field_name} must be a list of strings")
    out: list[str] = []
    for idx, item in enumerate(value):
        if not isinstance(item, str):
            raise ValueError(f"{field_name}[{idx}] must be a string")
        out.append(item)
    return tuple(out)


def _as_bool(value: object, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean")
    return value


def _resolve_bool_override(override: bool | None, *, config_default: bool) -> bool:
    return config_default if override is None else override


def _resolve_config_path(
    path_value: object,
    *,
    field_name: str,
    config_dir: Path,
    required: bool,
) -> Path | None:
    if path_value is None:
        if required:
            raise ValueError(f"{field_name} is required")
        return None
    if not isinstance(path_value, str) or not path_value.strip():
        raise ValueError(f"{field_name} must be a non-empty string path")
    candidate = Path(path_value)
    if not candidate.is_absolute():
        candidate = config_dir / candidate
    return candidate.resolve()


def _load_csv_table(path: Path, *, field_name: str) -> pd.DataFrame:
    if not path.exists():
        raise ValueError(f"{field_name} does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"{field_name} is not a file: {path}")
    try:
        return pd.read_csv(path)
    except Exception as exc:
        raise ValueError(f"could not read {field_name} ({path}): {exc}") from exc


def _parse_delay_spec(raw: object, *, field_name: str) -> DelaySpec:
    data = _string_key_dict(_as_mapping(raw, field_name=field_name), field_name=field_name)
    try:
        return DelaySpec(**data)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {field_name}: {exc}") from exc


def _parse_universe_rules(raw: object, *, field_name: str) -> ResearchUniverseRules:
    data = _string_key_dict(_as_mapping(raw, field_name=field_name), field_name=field_name)
    try:
        return ResearchUniverseRules(**data)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {field_name}: {exc}") from exc


def _parse_preprocess(raw: object, *, field_name: str) -> SignalPreprocessSpec:
    data = _string_key_dict(_as_mapping(raw, field_name=field_name), field_name=field_name)
    try:
        return SignalPreprocessSpec(**data)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {field_name}: {exc}") from exc


def _parse_neutralization(raw: object, *, field_name: str) -> NeutralizationSpec:
    data = _string_key_dict(_as_mapping(raw, field_name=field_name), field_name=field_name)
    try:
        return NeutralizationSpec(**data)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {field_name}: {exc}") from exc


def _parse_single_thresholds(raw: object, *, field_name: str) -> SingleFactorDecisionThresholds:
    data = _string_key_dict(_as_mapping(raw, field_name=field_name), field_name=field_name)
    try:
        return SingleFactorDecisionThresholds(**data)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {field_name}: {exc}") from exc


def _parse_composite_thresholds(
    raw: object,
    *,
    field_name: str,
) -> CompositeDecisionThresholds:
    data = _string_key_dict(_as_mapping(raw, field_name=field_name), field_name=field_name)
    try:
        return CompositeDecisionThresholds(**data)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {field_name}: {exc}") from exc


def _optional_timestamp(value: object, *, field_name: str) -> pd.Timestamp | None:
    if value is None:
        return None
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"{field_name} must be a valid datetime")
    return pd.Timestamp(ts)


def _parse_validation_metadata(raw: object, *, field_name: str) -> ValidationMetadata:
    data = _string_key_dict(_as_mapping(raw, field_name=field_name), field_name=field_name)
    if "scheme" not in data:
        raise ValueError(f"{field_name}.scheme is required")
    scheme = data["scheme"]
    if not isinstance(scheme, str) or not scheme.strip():
        raise ValueError(f"{field_name}.scheme must be a non-empty string")
    try:
        return ValidationMetadata(
            scheme=scheme,
            train_end=_optional_timestamp(
                data.get("train_end"),
                field_name=f"{field_name}.train_end",
            ),
            test_start=_optional_timestamp(
                data.get("test_start"),
                field_name=f"{field_name}.test_start",
            ),
            val_start=_optional_timestamp(
                data.get("val_start"),
                field_name=f"{field_name}.val_start",
            ),
            purge_periods=int(data.get("purge_periods", 0)),
            embargo_periods=int(data.get("embargo_periods", 0)),
            notes=cast(str | None, data.get("notes")),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {field_name}: {exc}") from exc


def _parse_experiment_metadata(raw: object, *, field_name: str) -> ExperimentMetadata:
    data = _string_key_dict(_as_mapping(raw, field_name=field_name), field_name=field_name)
    kwargs: dict[str, Any] = {}
    for key in (
        "hypothesis",
        "research_question",
        "factor_spec",
        "dataset_id",
        "dataset_hash",
        "trial_id",
        "verdict",
        "interpretation",
    ):
        if key in data:
            kwargs[key] = data[key]
    if "trial_count" in data and data["trial_count"] is not None:
        kwargs["trial_count"] = int(data["trial_count"])
    if "assumptions" in data:
        kwargs["assumptions"] = _as_str_tuple(
            data["assumptions"],
            field_name=f"{field_name}.assumptions",
        )
    if "caveats" in data:
        kwargs["caveats"] = _as_str_tuple(
            data["caveats"],
            field_name=f"{field_name}.caveats",
        )
    if "warnings" in data:
        kwargs["warnings"] = _as_str_tuple(
            data["warnings"],
            field_name=f"{field_name}.warnings",
        )
    if "validation" in data and data["validation"] is not None:
        kwargs["validation"] = _parse_validation_metadata(
            data["validation"],
            field_name=f"{field_name}.validation",
        )
    if "delay" in data and data["delay"] is not None:
        kwargs["delay"] = _parse_delay_spec(
            data["delay"],
            field_name=f"{field_name}.delay",
        )
    try:
        return ExperimentMetadata(**kwargs)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {field_name}: {exc}") from exc


def _factor_callable_from_name(
    factor_name: str,
    *,
    params: Mapping[str, Any] | None = None,
) -> Callable[[pd.DataFrame], pd.DataFrame]:
    if factor_name not in SUPPORTED_FACTORS:
        raise ValueError(
            f"unsupported factor {factor_name!r}; supported factors: {sorted(SUPPORTED_FACTORS)}"
        )
    kwargs = dict(params or {})
    if factor_name == "momentum":
        def _momentum_fn(prices: pd.DataFrame) -> pd.DataFrame:
            return momentum(prices, **kwargs)  # type: ignore[arg-type]

        return _momentum_fn
    if factor_name == "reversal":
        def _reversal_fn(prices: pd.DataFrame) -> pd.DataFrame:
            return reversal(prices, **kwargs)  # type: ignore[arg-type]

        return _reversal_fn

    def _low_vol_fn(prices: pd.DataFrame) -> pd.DataFrame:
        return low_volatility(prices, **kwargs)  # type: ignore[arg-type]

    return _low_vol_fn


def _single_factor_fn_from_config(
    config: Mapping[str, Any],
) -> Callable[[pd.DataFrame], pd.DataFrame]:
    if "name" not in config:
        raise ValueError("factor.name is required")
    raw_name = config["name"]
    if not isinstance(raw_name, str):
        raise ValueError("factor.name must be a string")
    params_raw = config.get("params", {})
    params = _string_key_dict(
        _as_mapping(params_raw, field_name="factor.params"),
        field_name="factor.params",
    )
    return _factor_callable_from_name(raw_name, params=params)


def _composite_factor_fns_from_config(
    factors_raw: object,
) -> dict[str, Callable[[pd.DataFrame], pd.DataFrame]]:
    if not isinstance(factors_raw, list) or not factors_raw:
        raise ValueError("factors must be a non-empty list")
    out: dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {}
    for idx, entry in enumerate(factors_raw):
        item = _as_mapping(entry, field_name=f"factors[{idx}]")
        if "name" not in item:
            raise ValueError(f"factors[{idx}].name is required")
        raw_name = item["name"]
        if not isinstance(raw_name, str):
            raise ValueError(f"factors[{idx}].name must be a string")
        params_raw = item.get("params", {})
        params = _string_key_dict(
            _as_mapping(params_raw, field_name=f"factors[{idx}].params"),
            field_name=f"factors[{idx}].params",
        )
        key = f"factor_{idx:02d}_{raw_name}"
        out[key] = _factor_callable_from_name(raw_name, params=params)
    return out


def _to_jsonable(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_workflow_summary(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_to_jsonable(dict(payload)), f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def _friendly_workflow_error(exc: Exception) -> ValueError:
    message = str(exc).strip() or exc.__class__.__name__
    if message in {"Factor output is empty", "Factor values are all NaN"}:
        return ValueError(
            "factor output is empty after masking/preprocessing. "
            "Check `spec.universe_rules` (especially `min_adv` and listing-age filters), "
            "`spec.preprocess.min_coverage`, and factor window settings."
        )
    if message == "compose_signals requires at least 2 input factors":
        return ValueError(
            "composite composition has fewer than 2 usable factors after screening. "
            "Check candidate signal coverage and screening thresholds."
        )
    return ValueError(message)


def _print_single_workflow_summary(
    result: SingleFactorWorkflowResult,
    *,
    summary_path: Path,
    trial_log_path: Path | None,
    registry_path: Path | None,
) -> None:
    summary = result.experiment_result.summary
    print("")
    print("  Workflow : run-single-factor")
    print("  Status   : success")
    print(f"  PromotionDecision : {result.decision.verdict}")
    print(f"  Mean Rank IC : {_fmt_float(float(summary.mean_rank_ic))}")
    print(f"  IC IR        : {_fmt_float(float(summary.ic_ir))}")
    print(f"  Mean L/S Ret : {_fmt_float(float(summary.mean_long_short_return))}")
    print(f"  Summary JSON : {summary_path}")
    if trial_log_path is not None:
        print(f"  Trial Log    : {trial_log_path}")
    if registry_path is not None:
        print(f"  Registry     : {registry_path}")
    if result.handoff_export is not None:
        print(f"  Handoff      : {result.handoff_export.artifact_path}")
    if result.decision.blocking_issues:
        print(f"  Blocking     : {', '.join(result.decision.blocking_issues)}")
    if result.decision.warnings:
        print(f"  Warnings     : {', '.join(result.decision.warnings)}")


def _print_composite_workflow_summary(
    result: CompositeWorkflowResult,
    *,
    summary_path: Path,
    trial_log_path: Path | None,
    registry_path: Path | None,
) -> None:
    summary = result.composite_experiment.summary
    n_selected = int(result.selected_signals["factor"].nunique())
    breadth = float("nan")
    if not result.alpha_pool_diagnostics.breadth_summary.empty:
        breadth = float(result.alpha_pool_diagnostics.breadth_summary["effective_breadth"].iloc[0])
    print("")
    print("  Workflow : run-composite")
    print("  Status   : success")
    print(f"  PromotionDecision : {result.decision.verdict}")
    print(f"  Mean Rank IC : {_fmt_float(float(summary.mean_rank_ic))}")
    print(f"  IC IR        : {_fmt_float(float(summary.ic_ir))}")
    print(f"  Selected Factors : {n_selected}")
    print(f"  Effective Breadth : {_fmt_float(breadth)}")
    print(f"  Summary JSON : {summary_path}")
    if trial_log_path is not None:
        print(f"  Trial Log    : {trial_log_path}")
    if registry_path is not None:
        print(f"  Registry     : {registry_path}")
    if result.handoff_export is not None:
        print(f"  Handoff      : {result.handoff_export.artifact_path}")
    if result.decision.blocking_issues:
        print(f"  Blocking     : {', '.join(result.decision.blocking_issues)}")
    if result.decision.warnings:
        print(f"  Warnings     : {', '.join(result.decision.warnings)}")


def _run_single_factor_workflow_command(args: argparse.Namespace) -> int:
    config_path = Path(args.config_path).resolve()
    config = _load_json_config(config_path)
    config_dir = config_path.parent
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"--output-dir must be a directory path: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = _as_mapping(config.get("data"), field_name="data")
    spec_cfg = _as_mapping(config.get("spec", {}), field_name="spec")
    factor_cfg = _as_mapping(config.get("factor"), field_name="factor")

    prices_path = _resolve_config_path(
        data_cfg.get("prices_path"),
        field_name="data.prices_path",
        config_dir=config_dir,
        required=True,
    )
    assert prices_path is not None
    prices = _load_prices(prices_path)
    try:
        validate_price_panel(prices)
    except ValueError as exc:
        raise ValueError(f"invalid prices data: {exc}") from exc

    asset_metadata_path = _resolve_config_path(
        data_cfg.get("asset_metadata_path"),
        field_name="data.asset_metadata_path",
        config_dir=config_dir,
        required=True,
    )
    assert asset_metadata_path is not None
    asset_metadata = _load_csv_table(asset_metadata_path, field_name="data.asset_metadata_path")

    market_state: pd.DataFrame | None = None
    market_state_path = _resolve_config_path(
        data_cfg.get("market_state_path"),
        field_name="data.market_state_path",
        config_dir=config_dir,
        required=False,
    )
    if market_state_path is not None:
        market_state = _load_csv_table(market_state_path, field_name="data.market_state_path")

    neutralization_exposures: pd.DataFrame | None = None
    neutralization_exposures_path = _resolve_config_path(
        data_cfg.get("neutralization_exposures_path"),
        field_name="data.neutralization_exposures_path",
        config_dir=config_dir,
        required=False,
    )
    if neutralization_exposures_path is not None:
        neutralization_exposures = _load_csv_table(
            neutralization_exposures_path,
            field_name="data.neutralization_exposures_path",
        )

    experiment_name = spec_cfg.get("experiment_name", config.get("experiment_name"))
    if not isinstance(experiment_name, str) or not experiment_name.strip():
        raise ValueError("spec.experiment_name is required")
    _safe_filename(experiment_name)

    factor_fn = _single_factor_fn_from_config(_string_key_dict(factor_cfg, field_name="factor"))
    spec_kwargs: dict[str, Any] = {
        "experiment_name": experiment_name,
        "factor_fn": factor_fn,
    }
    for key in (
        "horizon",
        "n_quantiles",
        "label_method",
        "screening_min_abs_monotonicity",
        "screening_max_pairwise_corr",
        "screening_max_vif",
        "validation_mode",
        "train_end",
        "test_start",
        "purged_n_splits",
        "purged_embargo_periods",
        "walk_forward_train_size",
        "walk_forward_test_size",
        "walk_forward_step",
        "walk_forward_val_size",
        "hypothesis",
        "research_question",
        "factor_spec",
        "dataset_id",
        "dataset_hash",
        "trial_id",
        "trial_count",
        "handoff_artifact_name",
        "handoff_include_label_snapshot",
        "handoff_overwrite",
        "registry_alpha_id",
        "registry_taxonomy",
        "registry_notes",
    ):
        if key in spec_cfg:
            spec_kwargs[key] = spec_cfg[key]
    if "label_kwargs" in spec_cfg:
        spec_kwargs["label_kwargs"] = _string_key_dict(
            _as_mapping(spec_cfg["label_kwargs"], field_name="spec.label_kwargs"),
            field_name="spec.label_kwargs",
        )
    if "assumptions" in spec_cfg:
        spec_kwargs["assumptions"] = _as_str_tuple(
            spec_cfg["assumptions"],
            field_name="spec.assumptions",
        )
    if "caveats" in spec_cfg:
        spec_kwargs["caveats"] = _as_str_tuple(spec_cfg["caveats"], field_name="spec.caveats")
    if "registry_tags" in spec_cfg:
        spec_kwargs["registry_tags"] = _as_str_tuple(
            spec_cfg["registry_tags"],
            field_name="spec.registry_tags",
        )
    if "delay_spec" in spec_cfg and spec_cfg["delay_spec"] is not None:
        spec_kwargs["delay_spec"] = _parse_delay_spec(
            spec_cfg["delay_spec"],
            field_name="spec.delay_spec",
        )
    if "universe_rules" in spec_cfg and spec_cfg["universe_rules"] is not None:
        spec_kwargs["universe_rules"] = _parse_universe_rules(
            spec_cfg["universe_rules"],
            field_name="spec.universe_rules",
        )
    if "preprocess" in spec_cfg and spec_cfg["preprocess"] is not None:
        spec_kwargs["preprocess"] = _parse_preprocess(
            spec_cfg["preprocess"],
            field_name="spec.preprocess",
        )
    if "neutralization" in spec_cfg and spec_cfg["neutralization"] is not None:
        spec_kwargs["neutralization"] = _parse_neutralization(
            spec_cfg["neutralization"],
            field_name="spec.neutralization",
        )
    if "decision_thresholds" in spec_cfg and spec_cfg["decision_thresholds"] is not None:
        spec_kwargs["decision_thresholds"] = _parse_single_thresholds(
            spec_cfg["decision_thresholds"],
            field_name="spec.decision_thresholds",
        )
    if "metadata" in spec_cfg and spec_cfg["metadata"] is not None:
        spec_kwargs["metadata"] = _parse_experiment_metadata(
            spec_cfg["metadata"],
            field_name="spec.metadata",
        )

    append_trial_log_cfg = _as_bool(
        spec_cfg.get("append_trial_log", False),
        field_name="spec.append_trial_log",
    )
    append_trial_log = _resolve_bool_override(
        cast(bool | None, args.write_trial_log),
        config_default=append_trial_log_cfg,
    )
    spec_kwargs["append_trial_log"] = append_trial_log
    trial_log_path: Path | None = None
    if append_trial_log:
        trial_log_path = _resolve_config_path(
            spec_cfg.get("trial_log_path"),
            field_name="spec.trial_log_path",
            config_dir=config_dir,
            required=False,
        )
        if trial_log_path is None:
            trial_log_path = output_dir / "trial_log.csv"
        spec_kwargs["trial_log_path"] = trial_log_path

    update_registry_cfg = _as_bool(
        spec_cfg.get("update_registry", False),
        field_name="spec.update_registry",
    )
    update_registry = _resolve_bool_override(
        cast(bool | None, args.update_registry),
        config_default=update_registry_cfg,
    )
    spec_kwargs["update_registry"] = update_registry
    registry_path: Path | None = None
    if update_registry:
        registry_path = _resolve_config_path(
            spec_cfg.get("registry_path"),
            field_name="spec.registry_path",
            config_dir=config_dir,
            required=False,
        )
        if registry_path is None:
            registry_path = output_dir / "alpha_registry.csv"
        spec_kwargs["registry_path"] = registry_path

    export_handoff_cfg = _as_bool(
        spec_cfg.get("export_handoff", False),
        field_name="spec.export_handoff",
    )
    export_handoff = _resolve_bool_override(
        cast(bool | None, args.export_handoff),
        config_default=export_handoff_cfg,
    )
    spec_kwargs["export_handoff"] = export_handoff
    if export_handoff:
        handoff_output_dir = _resolve_config_path(
            spec_cfg.get("handoff_output_dir"),
            field_name="spec.handoff_output_dir",
            config_dir=config_dir,
            required=False,
        )
        if handoff_output_dir is None:
            handoff_output_dir = output_dir / "handoff"
        spec_kwargs["handoff_output_dir"] = handoff_output_dir

    try:
        spec = SingleFactorWorkflowSpec(**spec_kwargs)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid single-factor workflow spec: {exc}") from exc

    try:
        result = run_single_factor_research_workflow(
            prices,
            spec=spec,
            asset_metadata=asset_metadata,
            market_state=market_state,
            neutralization_exposures=neutralization_exposures,
        )
    except Exception as exc:
        raise _friendly_workflow_error(exc) from exc

    summary_path = output_dir / f"{spec.experiment_name}_single_factor_workflow_summary.json"
    summary_payload: dict[str, object] = {
        "workflow": "run-single-factor",
        "status": "success",
        "experiment_name": spec.experiment_name,
        "config_path": str(config_path),
        "promotion_decision": {
            "verdict": result.decision.verdict,
            "reasons": list(result.decision.reasons),
            "blocking_issues": list(result.decision.blocking_issues),
            "warnings": list(result.decision.warnings),
            "metrics": result.decision.metrics,
        },
        "key_metrics": {
            "mean_ic": result.experiment_result.summary.mean_ic,
            "mean_rank_ic": result.experiment_result.summary.mean_rank_ic,
            "ic_ir": result.experiment_result.summary.ic_ir,
            "mean_long_short_return": (
                result.experiment_result.summary.mean_long_short_return
            ),
        },
        "outputs": {
            "summary_json": str(summary_path),
            "trial_log": str(
                trial_log_path if trial_log_path is not None else DEFAULT_TRIAL_LOG_PATH
            )
            if append_trial_log
            else None,
            "alpha_registry": str(
                registry_path if registry_path is not None else ALPHA_REGISTRY_PATH
            )
            if update_registry
            else None,
            "handoff_artifact": (
                str(result.handoff_export.artifact_path)
                if result.handoff_export is not None
                else None
            ),
            "handoff_manifest": (
                str(result.handoff_export.manifest_path)
                if result.handoff_export is not None
                else None
            ),
        },
    }
    _write_workflow_summary(summary_path, summary_payload)
    _print_single_workflow_summary(
        result,
        summary_path=summary_path,
        trial_log_path=trial_log_path if append_trial_log else None,
        registry_path=registry_path if update_registry else None,
    )
    return 0


def _run_composite_workflow_command(args: argparse.Namespace) -> int:
    config_path = Path(args.config_path).resolve()
    config = _load_json_config(config_path)
    config_dir = config_path.parent
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"--output-dir must be a directory path: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = _as_mapping(config.get("data"), field_name="data")
    spec_cfg = _as_mapping(config.get("spec", {}), field_name="spec")

    prices_path = _resolve_config_path(
        data_cfg.get("prices_path"),
        field_name="data.prices_path",
        config_dir=config_dir,
        required=True,
    )
    assert prices_path is not None
    prices = _load_prices(prices_path)
    try:
        validate_price_panel(prices)
    except ValueError as exc:
        raise ValueError(f"invalid prices data: {exc}") from exc

    asset_metadata_path = _resolve_config_path(
        data_cfg.get("asset_metadata_path"),
        field_name="data.asset_metadata_path",
        config_dir=config_dir,
        required=True,
    )
    assert asset_metadata_path is not None
    asset_metadata = _load_csv_table(asset_metadata_path, field_name="data.asset_metadata_path")

    market_state: pd.DataFrame | None = None
    market_state_path = _resolve_config_path(
        data_cfg.get("market_state_path"),
        field_name="data.market_state_path",
        config_dir=config_dir,
        required=False,
    )
    if market_state_path is not None:
        market_state = _load_csv_table(market_state_path, field_name="data.market_state_path")

    neutralization_exposures: pd.DataFrame | None = None
    neutralization_exposures_path = _resolve_config_path(
        data_cfg.get("neutralization_exposures_path"),
        field_name="data.neutralization_exposures_path",
        config_dir=config_dir,
        required=False,
    )
    if neutralization_exposures_path is not None:
        neutralization_exposures = _load_csv_table(
            neutralization_exposures_path,
            field_name="data.neutralization_exposures_path",
        )

    exposure_data: pd.DataFrame | None = None
    exposure_data_path = _resolve_config_path(
        data_cfg.get("exposure_data_path"),
        field_name="data.exposure_data_path",
        config_dir=config_dir,
        required=False,
    )
    if exposure_data_path is not None:
        exposure_data = _load_csv_table(exposure_data_path, field_name="data.exposure_data_path")

    factor_fns: dict[str, Callable[[pd.DataFrame], pd.DataFrame]] | None = None
    candidate_signals: pd.DataFrame | None = None
    factors_raw = config.get("factors")
    candidate_signals_path = _resolve_config_path(
        data_cfg.get("candidate_signals_path"),
        field_name="data.candidate_signals_path",
        config_dir=config_dir,
        required=False,
    )
    if factors_raw is not None and candidate_signals_path is not None:
        raise ValueError("provide either factors or data.candidate_signals_path, not both")
    if factors_raw is not None:
        factor_fns = _composite_factor_fns_from_config(factors_raw)
    elif candidate_signals_path is not None:
        candidate_signals = _load_csv_table(
            candidate_signals_path,
            field_name="data.candidate_signals_path",
        )
    else:
        raise ValueError("one of factors or data.candidate_signals_path is required")

    experiment_name = spec_cfg.get("experiment_name", config.get("experiment_name"))
    if not isinstance(experiment_name, str) or not experiment_name.strip():
        raise ValueError("spec.experiment_name is required")
    _safe_filename(experiment_name)

    spec_kwargs: dict[str, Any] = {
        "experiment_name": experiment_name,
    }
    for key in (
        "horizon",
        "n_quantiles",
        "label_method",
        "screening_min_coverage",
        "screening_min_abs_monotonicity",
        "screening_max_pairwise_corr",
        "screening_max_vif",
        "composite_method",
        "composite_lookback",
        "composite_min_history",
        "composite_factor_name",
        "portfolio_top_k",
        "portfolio_bottom_k",
        "portfolio_weighting_method",
        "portfolio_value",
        "adv_window",
        "capacity_max_adv_participation",
        "capacity_concentration_weight_threshold",
        "cost_flat_fee_bps",
        "cost_spread_bps",
        "cost_impact_eta",
        "hypothesis",
        "research_question",
        "factor_spec",
        "dataset_id",
        "dataset_hash",
        "trial_id",
        "trial_count",
        "handoff_artifact_name",
        "handoff_include_label_snapshot",
        "handoff_overwrite",
        "registry_alpha_id",
        "registry_taxonomy",
        "registry_notes",
    ):
        if key in spec_cfg:
            spec_kwargs[key] = spec_cfg[key]
    if "label_kwargs" in spec_cfg:
        spec_kwargs["label_kwargs"] = _string_key_dict(
            _as_mapping(spec_cfg["label_kwargs"], field_name="spec.label_kwargs"),
            field_name="spec.label_kwargs",
        )
    if "assumptions" in spec_cfg:
        spec_kwargs["assumptions"] = _as_str_tuple(
            spec_cfg["assumptions"],
            field_name="spec.assumptions",
        )
    if "caveats" in spec_cfg:
        spec_kwargs["caveats"] = _as_str_tuple(spec_cfg["caveats"], field_name="spec.caveats")
    if "registry_tags" in spec_cfg:
        spec_kwargs["registry_tags"] = _as_str_tuple(
            spec_cfg["registry_tags"],
            field_name="spec.registry_tags",
        )
    if "delay_spec" in spec_cfg and spec_cfg["delay_spec"] is not None:
        spec_kwargs["delay_spec"] = _parse_delay_spec(
            spec_cfg["delay_spec"],
            field_name="spec.delay_spec",
        )
    if "universe_rules" in spec_cfg and spec_cfg["universe_rules"] is not None:
        spec_kwargs["universe_rules"] = _parse_universe_rules(
            spec_cfg["universe_rules"],
            field_name="spec.universe_rules",
        )
    if "preprocess" in spec_cfg and spec_cfg["preprocess"] is not None:
        spec_kwargs["preprocess"] = _parse_preprocess(
            spec_cfg["preprocess"],
            field_name="spec.preprocess",
        )
    if "neutralization" in spec_cfg and spec_cfg["neutralization"] is not None:
        spec_kwargs["neutralization"] = _parse_neutralization(
            spec_cfg["neutralization"],
            field_name="spec.neutralization",
        )
    if "decision_thresholds" in spec_cfg and spec_cfg["decision_thresholds"] is not None:
        spec_kwargs["decision_thresholds"] = _parse_composite_thresholds(
            spec_cfg["decision_thresholds"],
            field_name="spec.decision_thresholds",
        )
    if "metadata" in spec_cfg and spec_cfg["metadata"] is not None:
        spec_kwargs["metadata"] = _parse_experiment_metadata(
            spec_cfg["metadata"],
            field_name="spec.metadata",
        )

    append_trial_log_cfg = _as_bool(
        spec_cfg.get("append_trial_log", False),
        field_name="spec.append_trial_log",
    )
    append_trial_log = _resolve_bool_override(
        cast(bool | None, args.write_trial_log),
        config_default=append_trial_log_cfg,
    )
    spec_kwargs["append_trial_log"] = append_trial_log
    trial_log_path: Path | None = None
    if append_trial_log:
        trial_log_path = _resolve_config_path(
            spec_cfg.get("trial_log_path"),
            field_name="spec.trial_log_path",
            config_dir=config_dir,
            required=False,
        )
        if trial_log_path is None:
            trial_log_path = output_dir / "trial_log.csv"
        spec_kwargs["trial_log_path"] = trial_log_path

    update_registry_cfg = _as_bool(
        spec_cfg.get("update_registry", False),
        field_name="spec.update_registry",
    )
    update_registry = _resolve_bool_override(
        cast(bool | None, args.update_registry),
        config_default=update_registry_cfg,
    )
    spec_kwargs["update_registry"] = update_registry
    registry_path: Path | None = None
    if update_registry:
        registry_path = _resolve_config_path(
            spec_cfg.get("registry_path"),
            field_name="spec.registry_path",
            config_dir=config_dir,
            required=False,
        )
        if registry_path is None:
            registry_path = output_dir / "alpha_registry.csv"
        spec_kwargs["registry_path"] = registry_path

    export_handoff_cfg = _as_bool(
        spec_cfg.get("export_handoff", False),
        field_name="spec.export_handoff",
    )
    export_handoff = _resolve_bool_override(
        cast(bool | None, args.export_handoff),
        config_default=export_handoff_cfg,
    )
    spec_kwargs["export_handoff"] = export_handoff
    if export_handoff:
        handoff_output_dir = _resolve_config_path(
            spec_cfg.get("handoff_output_dir"),
            field_name="spec.handoff_output_dir",
            config_dir=config_dir,
            required=False,
        )
        if handoff_output_dir is None:
            handoff_output_dir = output_dir / "handoff"
        spec_kwargs["handoff_output_dir"] = handoff_output_dir

    try:
        spec = CompositeWorkflowSpec(**spec_kwargs)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid composite workflow spec: {exc}") from exc

    try:
        result = run_composite_signal_research_workflow(
            prices,
            spec=spec,
            factor_fns=factor_fns,
            candidate_signals=candidate_signals,
            asset_metadata=asset_metadata,
            market_state=market_state,
            neutralization_exposures=neutralization_exposures,
            exposure_data=exposure_data,
        )
    except Exception as exc:
        raise _friendly_workflow_error(exc) from exc

    summary_path = output_dir / f"{spec.experiment_name}_composite_workflow_summary.json"
    summary_payload: dict[str, object] = {
        "workflow": "run-composite",
        "status": "success",
        "experiment_name": spec.experiment_name,
        "config_path": str(config_path),
        "promotion_decision": {
            "verdict": result.decision.verdict,
            "reasons": list(result.decision.reasons),
            "blocking_issues": list(result.decision.blocking_issues),
            "warnings": list(result.decision.warnings),
            "metrics": result.decision.metrics,
        },
        "key_metrics": {
            "mean_ic": result.composite_experiment.summary.mean_ic,
            "mean_rank_ic": result.composite_experiment.summary.mean_rank_ic,
            "ic_ir": result.composite_experiment.summary.ic_ir,
            "selected_factor_count": int(result.selected_signals["factor"].nunique()),
        },
        "outputs": {
            "summary_json": str(summary_path),
            "trial_log": str(
                trial_log_path if trial_log_path is not None else DEFAULT_TRIAL_LOG_PATH
            )
            if append_trial_log
            else None,
            "alpha_registry": str(
                registry_path if registry_path is not None else ALPHA_REGISTRY_PATH
            )
            if update_registry
            else None,
            "handoff_artifact": (
                str(result.handoff_export.artifact_path)
                if result.handoff_export is not None
                else None
            ),
            "handoff_manifest": (
                str(result.handoff_export.manifest_path)
                if result.handoff_export is not None
                else None
            ),
        },
    }
    _write_workflow_summary(summary_path, summary_payload)
    _print_composite_workflow_summary(
        result,
        summary_path=summary_path,
        trial_log_path=trial_log_path if append_trial_log else None,
        registry_path=registry_path if update_registry else None,
    )
    return 0


def _main_workflow_cli(argv: list[str]) -> int:
    parser = build_workflow_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "run-single-factor":
            return _run_single_factor_workflow_command(args)
        if args.command == "run-composite":
            return _run_composite_workflow_command(args)
    except ValueError as exc:
        parser.error(str(exc))
    parser.error(f"unsupported workflow command: {args.command!r}")
    return 2


# ---------------------------------------------------------------------------
# Legacy experiment CLI
# ---------------------------------------------------------------------------


def _main_legacy_experiment_cli(argv: list[str] | None = None) -> int:
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
        _registry.register_experiment(
            experiment_name,
            summary,
            _registry.DEFAULT_REGISTRY_PATH,
            obsidian_path=str(resolved_md_path) if resolved_md_path is not None else None,
        )
        print(f"  Registry   : appended '{experiment_name}'")

    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Run legacy experiment CLI or workflow CLI based on argv."""
    resolved_argv = list(sys.argv[1:] if argv is None else argv)
    if _is_workflow_command(resolved_argv):
        return _main_workflow_cli(resolved_argv)
    return _main_legacy_experiment_cli(resolved_argv)


if __name__ == "__main__":
    sys.exit(main())
