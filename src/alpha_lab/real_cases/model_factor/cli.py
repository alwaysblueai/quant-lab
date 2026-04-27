from __future__ import annotations

import argparse
import glob
import json
import logging
import sys
from pathlib import Path

from alpha_lab.artifact_contracts import validate_level12_artifact_payload
from alpha_lab.reporting.renderers import write_case_report
from alpha_lab.research_evaluation_config import (
    AVAILABLE_RESEARCH_EVALUATION_PROFILES,
    DEFAULT_RESEARCH_EVALUATION_CONFIG,
)

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
            "portfolio-validation guardrails."
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
            "portfolio-validation guardrails."
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

    if args.command != "run":
        parser.error(f"unsupported command: {args.command!r}")

    result = _run_one(args.spec_path, args, parser)
    _print_single_success(result)
    return 0


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

    results = []
    for spec_path in spec_paths:
        results.append(_run_one(spec_path, args, parser))

    print("")
    print("  Workflow : real-case-model-factor-batch")
    print("  Status   : success")
    print(f"  Cases    : {len(results)}")
    for result in results:
        print(f"  - {result.spec.name}: {result.output_dir}")
    return 0


def _run_one(
    spec_path: str | Path,
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> ModelFactorCaseRunResult:
    try:
        result = run_model_factor_case(
            Path(spec_path),
            output_root_dir=args.output_root_dir,
            evaluation_profile=args.evaluation_profile,
            vault_root=args.vault_root,
            vault_export_mode=args.vault_export_mode,
        )
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        parser.error(str(exc))

    render_meta = _render_case_report(
        output_dir=result.output_dir,
        enabled=bool(args.render_report),
        overwrite=bool(args.render_overwrite),
    )
    _update_run_manifest(result.artifact_paths["run_manifest"], render_meta)
    return result


def _print_single_success(result: ModelFactorCaseRunResult) -> None:
    print("")
    print("  Workflow : real-case-model-factor")
    print("  Status   : success")
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


def _render_case_report(
    *,
    output_dir: Path,
    enabled: bool,
    overwrite: bool,
) -> dict[str, object]:
    if not enabled:
        return {
            "rendered_report": False,
            "rendered_report_path": None,
            "render_status": "skipped",
            "render_error": None,
        }

    try:
        report_path = write_case_report(output_dir, overwrite=overwrite)
        return {
            "rendered_report": True,
            "rendered_report_path": str(report_path),
            "render_status": "success",
            "render_error": None,
        }
    except Exception as exc:
        logger.warning(
            "Case report rendering failed for %s: %s",
            output_dir,
            exc,
        )
        return {
            "rendered_report": False,
            "rendered_report_path": None,
            "render_status": "failed",
            "render_error": str(exc),
        }


def _update_run_manifest(
    manifest_path: Path,
    render_meta: dict[str, object],
) -> None:
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("run_manifest.json root must be an object")
        payload.update(render_meta)
        validate_level12_artifact_payload(
            payload,
            artifact_name=manifest_path.name,
            source=manifest_path,
        )
        manifest_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except Exception as exc:
        logger.warning(
            "Failed to persist render metadata into %s: %s",
            manifest_path,
            exc,
        )


def _fmt_text(value: object) -> str:
    text = str(value).strip() if value is not None else ""
    return text if text else "N/A"


if __name__ == "__main__":
    sys.exit(main())
