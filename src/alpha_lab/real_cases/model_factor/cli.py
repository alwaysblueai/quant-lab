from __future__ import annotations

import argparse
import glob
import json
import logging
import sys
from pathlib import Path

from alpha_lab.artifact_contracts import validate_level12_artifact_payload
from alpha_lab.real_cases._cli_io import (
    export_to_vault_after_contract,
    finalize_contract_if_research_draft,
    render_case_report,
    update_run_manifest,
)
from alpha_lab.research_evaluation_config import (
    AVAILABLE_RESEARCH_EVALUATION_PROFILES,
    DEFAULT_RESEARCH_EVALUATION_CONFIG,
)

from .benchmark import ModelFactorBenchmarkOptions, run_model_factor_benchmark
from .pipeline import ModelFactorCaseRunResult, run_model_factor_case

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="alpha-lab real-case model-factor",
        description=(
            "Run one Level 1/2 model-factor workflow from feature input to auditable artifacts "
            "(feature table -> model score -> canonical factor -> Level 1 evaluation -> "
            "campaign triage -> Level 2 promotion gate -> Level 2 portfolio validation)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run",
        help="Run one model-factor case and export artifacts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    run_parser.add_argument("spec_path", help="Path to model-factor case YAML/JSON spec.")
    run_parser.add_argument(
        "--evaluation-profile",
        default=DEFAULT_RESEARCH_EVALUATION_CONFIG.profile_name,
        choices=sorted(AVAILABLE_RESEARCH_EVALUATION_PROFILES),
        help=(
            "Research evaluation profile controlling factor verdict standards, "
            "campaign triage, Level 2 promotion gate thresholds, and Level 2 "
            "portfolio-validation guardrails. Use exploratory_screening for "
            "faster model-factor development iteration."
        ),
    )
    run_parser.add_argument(
        "--disable-preparation-cache",
        action="store_true",
        help="Disable local model-factor data/preprocessed-input cache for this run.",
    )
    run_parser.add_argument(
        "--screening-retrain-every-n-dates",
        type=int,
        default=None,
        help=(
            "Optional runtime retrain cadence override for exploratory_screening runs. "
            "Use larger values such as 40 or 60 for faster screening; ignored for "
            "default_research/stricter_research."
        ),
    )
    run_parser.add_argument(
        "--output-root-dir",
        default=None,
        help=(
            "Optional output root override. Final artifact directory is "
            "<output-root-dir>/<case_name>."
        ),
    )
    run_parser.add_argument(
        "--cache-root-dir",
        default=None,
        help=(
            "Optional preparation cache root override. When set, the model-factor "
            "dataset cache lives at <cache-root-dir>/_model_factor_cache and is "
            "shared across runs that point at the same cache root. When unset, "
            "the cache sits next to the case output dir (CLI default: shared "
            "across cases under the same output root)."
        ),
    )
    run_parser.add_argument(
        "--vault-root",
        default=None,
        help=(
            "Optional quant-knowledge vault root path. Resolution priority: "
            "CLI flag -> OBSIDIAN_VAULT_PATH env -> disabled."
        ),
    )
    run_parser.add_argument(
        "--vault-export-mode",
        default="versioned",
        choices=["skip", "overwrite", "versioned"],
        help="Vault export behavior when a vault root is available.",
    )
    run_parser.add_argument(
        "--render-report",
        action="store_true",
        help="Render case_report.md after a successful run.",
    )
    run_parser.add_argument(
        "--render-overwrite",
        action="store_true",
        help="Overwrite existing case_report.md when rendering is enabled.",
    )
    run_parser.add_argument(
        "--draft-model-candidate",
        default=None,
        help=(
            "Optional path to a Stage3 model_candidate.json. When set, the run "
            "writes a draft_model_source audit block (candidate hash, case-spec "
            "hash, feature-contract hash) into run_manifest.json, "
            "model_definition.json, and feature_manifest.json."
        ),
    )

    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Run one model-factor case and write a benchmark recorder JSON.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    benchmark_parser.add_argument("spec_path", help="Path to model-factor case YAML/JSON spec.")
    benchmark_parser.add_argument(
        "--evaluation-profile",
        default=DEFAULT_RESEARCH_EVALUATION_CONFIG.profile_name,
        choices=sorted(AVAILABLE_RESEARCH_EVALUATION_PROFILES),
        help=(
            "Research evaluation profile for this benchmark. Use exploratory_screening "
            "for fast development timing and default_research for full research timing."
        ),
    )
    benchmark_parser.add_argument(
        "--disable-preparation-cache",
        action="store_true",
        help="Disable local model-factor data/preprocessed-input cache for this benchmark.",
    )
    benchmark_parser.add_argument(
        "--screening-retrain-every-n-dates",
        type=int,
        default=None,
        help="Optional runtime retrain cadence override for exploratory_screening runs.",
    )
    benchmark_parser.add_argument(
        "--output-root-dir",
        default=None,
        help=(
            "Optional case artifact output root override. Final artifact directory is "
            "<output-root-dir>/<case_name>."
        ),
    )
    benchmark_parser.add_argument(
        "--cache-root-dir",
        default=None,
        help=(
            "Optional preparation cache root override. See run --cache-root-dir."
        ),
    )
    benchmark_parser.add_argument(
        "--benchmark-output-dir",
        default=None,
        help=(
            "Directory for benchmark JSON records. Defaults to "
            "<output-root-dir>/_benchmark_records or the spec output root."
        ),
    )
    benchmark_parser.add_argument(
        "--run-id",
        default=None,
        help="Optional stable benchmark run id. Defaults to timestamp + case name.",
    )
    benchmark_parser.add_argument(
        "--note",
        default=None,
        help="Optional free-form note stored in the benchmark record.",
    )
    benchmark_parser.add_argument(
        "--memory-sample-interval-seconds",
        type=float,
        default=1.0,
        help="Interval for lightweight /proc/self memory sampling.",
    )
    benchmark_parser.add_argument(
        "--memory-limit-gb",
        type=float,
        default=24.0,
        help=(
            "Process address-space limit for benchmark runs. This turns many OOM "
            "cases into catchable MemoryError/ArrowMemoryError records instead of "
            "letting the kernel OOM-kill WSL. Set to 0 to disable."
        ),
    )
    benchmark_parser.add_argument(
        "--memory-profile",
        action="store_true",
        help=(
            "Write a per-sample RSS/swap timeline with the current model-factor stage "
            "to a sidecar memory profile artifact."
        ),
    )
    benchmark_parser.add_argument(
        "--tracemalloc-profile",
        action="store_true",
        help=(
            "Capture tracemalloc snapshots at selected stage boundaries and write "
            "top allocation diffs to the memory profile artifact."
        ),
    )
    benchmark_parser.add_argument(
        "--tracemalloc-nframes",
        type=int,
        default=10,
        help="Number of Python frames retained in tracemalloc stage snapshots.",
    )
    benchmark_parser.add_argument(
        "--vault-root",
        default=None,
        help=(
            "Optional quant-knowledge vault root path. Resolution priority: "
            "CLI flag -> OBSIDIAN_VAULT_PATH env -> disabled."
        ),
    )
    benchmark_parser.add_argument(
        "--vault-export-mode",
        default="skip",
        choices=["skip", "overwrite", "versioned"],
        help="Vault export behavior when a vault root is available.",
    )

    batch_parser = subparsers.add_parser(
        "run-batch",
        help="Run multiple model-factor case specs in one Python process.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    batch_parser.add_argument(
        "spec_patterns",
        nargs="+",
        help="Spec paths or glob patterns, e.g. configs/real_cases/model_factor/*.yaml.",
    )
    batch_parser.add_argument(
        "--evaluation-profile",
        default=DEFAULT_RESEARCH_EVALUATION_CONFIG.profile_name,
        choices=sorted(AVAILABLE_RESEARCH_EVALUATION_PROFILES),
        help=(
            "Research evaluation profile controlling factor verdict standards, "
            "campaign triage, Level 2 promotion gate thresholds, and Level 2 "
            "portfolio-validation guardrails. Use exploratory_screening for "
            "faster model-factor development iteration."
        ),
    )
    batch_parser.add_argument(
        "--disable-preparation-cache",
        action="store_true",
        help="Disable local model-factor data/preprocessed-input cache for this batch.",
    )
    batch_parser.add_argument(
        "--screening-retrain-every-n-dates",
        type=int,
        default=None,
        help=(
            "Optional runtime retrain cadence override for exploratory_screening batch runs. "
            "Use larger values such as 40 or 60 for faster screening; ignored for "
            "default_research/stricter_research."
        ),
    )
    batch_parser.add_argument(
        "--output-root-dir",
        default=None,
        help=(
            "Optional output root override. Each case writes to "
            "<output-root-dir>/<case_name>."
        ),
    )
    batch_parser.add_argument(
        "--cache-root-dir",
        default=None,
        help=(
            "Optional preparation cache root override applied to every case in the batch."
        ),
    )
    batch_parser.add_argument(
        "--vault-root",
        default=None,
        help=(
            "Optional quant-knowledge vault root path. Resolution priority: "
            "CLI flag -> OBSIDIAN_VAULT_PATH env -> disabled."
        ),
    )
    batch_parser.add_argument(
        "--vault-export-mode",
        default="versioned",
        choices=["skip", "overwrite", "versioned"],
        help="Vault export behavior when a vault root is available.",
    )
    batch_parser.add_argument(
        "--render-report",
        action="store_true",
        help="Render case_report.md after each successful run.",
    )
    batch_parser.add_argument(
        "--render-overwrite",
        action="store_true",
        help="Overwrite existing case_report.md when rendering is enabled.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run-batch":
        return _run_batch(args, parser)

    if args.command == "benchmark":
        return _run_benchmark(args, parser)

    if args.command != "run":
        parser.error(f"unsupported command: {args.command!r}")

    result, contract_rc = _run_one(args.spec_path, args, parser)
    _print_single_success(result, contract_rc=contract_rc)
    return contract_rc


def _run_batch(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    try:
        spec_paths = _expand_spec_patterns(args.spec_patterns)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        from alpha_lab.numba_kernels import warmup_numba_kernels

        warmup_numba_kernels()
    except Exception as exc:  # noqa: BLE001
        logger.debug("numba warmup skipped: %s", exc)

    results: list[ModelFactorCaseRunResult] = []
    worst_rc = 0
    for spec_path in spec_paths:
        result, contract_rc = _run_one(spec_path, args, parser)
        worst_rc = max(worst_rc, contract_rc)
        results.append(result)

    status_line = "success" if worst_rc == 0 else "contract_failed"
    print("")
    print("  Workflow : real-case-model-factor-batch")
    print(f"  Status   : {status_line}")
    print(f"  Cases    : {len(results)}")
    for result in results:
        print(f"  - {result.spec.name}: {result.output_dir}")
    return worst_rc


def _run_one(
    spec_path: str | Path,
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> tuple[ModelFactorCaseRunResult, int]:
    draft_model_source = None
    candidate_arg = getattr(args, "draft_model_candidate", None)
    if candidate_arg:
        from alpha_lab.custom_models import read_draft_model_source

        try:
            draft_model_source = read_draft_model_source(candidate_arg)
        except (ValueError, FileNotFoundError) as exc:
            parser.error(f"--draft-model-candidate failed to load: {exc}")
    try:
        result = run_model_factor_case(
            Path(spec_path),
            output_root_dir=args.output_root_dir,
            cache_root_dir=getattr(args, "cache_root_dir", None),
            evaluation_profile=args.evaluation_profile,
            use_preparation_cache=not bool(args.disable_preparation_cache),
            screening_retrain_every_n_dates=args.screening_retrain_every_n_dates,
            vault_root=args.vault_root,
            vault_export_mode=args.vault_export_mode,
            draft_model_source=draft_model_source,
            defer_vault_export=True,
        )
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        parser.error(str(exc))

    render_meta = render_case_report(
        output_dir=result.output_dir,
        enabled=bool(args.render_report),
        overwrite=bool(args.render_overwrite),
    )
    update_run_manifest(result.artifact_paths["run_manifest"], render_meta)
    contract_rc = finalize_contract_if_research_draft(
        output_dir=result.output_dir,
        workflow="model_factor",
        case_spec_path=spec_path,
        evaluation_profile=args.evaluation_profile,
        command=tuple(sys.argv),
    )
    export_to_vault_after_contract(
        case_name=result.spec.name,
        vault_root=args.vault_root,
        vault_export_mode=args.vault_export_mode,
        experiment_card_path=result.artifact_paths["experiment_card"],
        summary_path=result.artifact_paths["summary"],
        manifest_path=result.artifact_paths["run_manifest"],
        workflow_label="model-factor",
    )
    return result, contract_rc


def _print_single_success(
    result: ModelFactorCaseRunResult,
    *,
    contract_rc: int = 0,
) -> None:
    status_line = "success" if contract_rc == 0 else "contract_failed"
    print("")
    print("  Workflow : real-case-model-factor")
    print(f"  Status   : {status_line}")
    print(f"  Case     : {result.spec.name}")
    print(f"  Output   : {result.output_dir}")
    print(f"  Manifest : {result.artifact_paths['run_manifest']}")
    print(f"  Metrics  : {result.artifact_paths['metrics']}")
    print(f"  Summary  : {result.artifact_paths['summary']}")
    print(f"  Card     : {result.artifact_paths['experiment_card']}")
    print(f"  Model    : {result.artifact_paths['model_definition_json']}")
    print(f"  Level 2  : {result.output_dir / 'level2_portfolio_validation'}")

    manifest_payload = json.loads(result.artifact_paths["run_manifest"].read_text(encoding="utf-8"))
    metrics_payload = json.loads(result.artifact_paths["metrics"].read_text(encoding="utf-8"))
    if not isinstance(manifest_payload, dict):
        raise ValueError("run_manifest.json root must be an object")
    if not isinstance(metrics_payload, dict):
        raise ValueError("metrics.json root must be an object")
    validate_level12_artifact_payload(
        manifest_payload,
        artifact_name=result.artifact_paths["run_manifest"].name,
        source=result.artifact_paths["run_manifest"],
    )
    validate_level12_artifact_payload(
        metrics_payload,
        artifact_name=result.artifact_paths["metrics"].name,
        source=result.artifact_paths["metrics"],
    )
    metrics = metrics_payload.get("metrics", {}) if isinstance(metrics_payload, dict) else {}
    evaluation_profile = _fmt_text(metrics.get("research_evaluation_profile"))
    triage_label = _fmt_text(metrics.get("campaign_triage"))
    promotion_label = _fmt_text(metrics.get("promotion_decision"))
    portfolio_status = _fmt_text(metrics.get("portfolio_validation_status"))
    portfolio_reco = _fmt_text(metrics.get("portfolio_validation_recommendation"))
    vault_meta = manifest_payload.get("vault_export", {})
    print(f"  Evaluation Profile : {evaluation_profile}")
    print(f"  Campaign Triage    : {triage_label}")
    print(f"  Level 2 Promotion  : {promotion_label}")
    print(f"  Level 2 Validation : {portfolio_status} ({portfolio_reco})")
    print(f"  Vault Export Status : {vault_meta.get('status')}")
    print(f"  Vault Export Mode   : {vault_meta.get('mode')}")
    print(f"  Report Render Status: {manifest_payload.get('render_status')}")
    print(f"  Report Path         : {manifest_payload.get('rendered_report_path')}")


def _run_benchmark(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    try:
        result = run_model_factor_benchmark(
            ModelFactorBenchmarkOptions(
                spec_path=args.spec_path,
                output_root_dir=args.output_root_dir,
                cache_root_dir=getattr(args, "cache_root_dir", None),
                benchmark_output_dir=args.benchmark_output_dir,
                evaluation_profile=args.evaluation_profile,
                use_preparation_cache=not bool(args.disable_preparation_cache),
                screening_retrain_every_n_dates=args.screening_retrain_every_n_dates,
                vault_root=args.vault_root,
                vault_export_mode=args.vault_export_mode,
                run_id=args.run_id,
                note=args.note,
                memory_sample_interval_seconds=args.memory_sample_interval_seconds,
                memory_limit_gb=(
                    None
                    if args.memory_limit_gb is not None and args.memory_limit_gb <= 0
                    else args.memory_limit_gb
                ),
                memory_profile=bool(args.memory_profile),
                tracemalloc_profile=bool(args.tracemalloc_profile),
                tracemalloc_nframes=int(args.tracemalloc_nframes),
            )
        )
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        parser.error(str(exc))

    record = result.record
    print("")
    print("  Workflow : real-case-model-factor-benchmark")
    print(f"  Status   : {record.get('status')}")
    config = record.get("config", {})
    spec = config.get("spec", {}) if isinstance(config, dict) else {}
    print(f"  Case     : {spec.get('name') if isinstance(spec, dict) else 'N/A'}")
    print(f"  Record   : {result.record_path}")
    print(f"  Wall Sec : {record.get('wall_seconds')}")
    memory = record.get("memory", {})
    if isinstance(memory, dict):
        print(f"  Peak RSS : {memory.get('peak_rss_mb')} MB")
        print(f"  Peak Swap: {memory.get('peak_swap_mb')} MB")
    training = record.get("training", {})
    if isinstance(training, dict):
        print(f"  Fit Count: {training.get('fit_count')}")
    return 0



def _expand_spec_patterns(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        matches = [Path(item) for item in glob.glob(pattern)]
        if not matches and Path(pattern).exists():
            matches = [Path(pattern)]
        if not matches:
            raise ValueError(f"run-batch pattern matched no specs: {pattern}")
        for match in sorted(matches):
            resolved = match.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            paths.append(resolved)
    return paths


def _fmt_text(value: object) -> str:
    text = str(value).strip() if value is not None else ""
    return text if text else "N/A"


if __name__ == "__main__":
    sys.exit(main())
