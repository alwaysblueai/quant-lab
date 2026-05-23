from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from alpha_lab.artifact_contracts import validate_level12_artifact_payload
from alpha_lab.exceptions import AlphaLabDataError
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

from .pipeline import run_single_factor_case

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="alpha-lab real-case single-factor",
        description=(
            "Run one Level 1/2 single-factor workflow from spec to auditable artifacts "
            "(Level 1 evaluation -> campaign triage -> Level 2 promotion gate -> "
            "Level 2 portfolio validation)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run",
        help="Run one single-factor case and export artifacts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    run_parser.add_argument("spec_path", help="Path to single-factor case YAML/JSON spec.")
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
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command != "run":
        parser.error(f"unsupported command: {args.command!r}")

    try:
        result = run_single_factor_case(
            Path(args.spec_path),
            output_root_dir=args.output_root_dir,
            evaluation_profile=args.evaluation_profile,
            vault_root=args.vault_root,
            vault_export_mode=args.vault_export_mode,
            defer_vault_export=True,
        )
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        parser.error(str(exc))

    render_meta = render_case_report(
        output_dir=result.output_dir,
        enabled=bool(args.render_report),
        overwrite=bool(args.render_overwrite),
    )
    update_run_manifest(
        result.artifact_paths["run_manifest"],
        render_meta,
    )
    contract_rc = finalize_contract_if_research_draft(
        output_dir=result.output_dir,
        workflow="single_factor",
        case_spec_path=args.spec_path,
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
        workflow_label="single-factor",
    )

    print("")
    print("  Workflow : real-case-single-factor")
    print("  Status   : success" if contract_rc == 0 else "  Status   : contract_failed")
    print(f"  Case     : {result.spec.name}")
    print(f"  Output   : {result.output_dir}")
    print(f"  Manifest : {result.artifact_paths['run_manifest']}")
    print(f"  Metrics  : {result.artifact_paths['metrics']}")
    print(f"  Summary  : {result.artifact_paths['summary']}")
    print(f"  Card     : {result.artifact_paths['experiment_card']}")
    print(f"  Level 2  : {result.output_dir / 'level2_portfolio_validation'}")

    manifest_payload = json.loads(result.artifact_paths["run_manifest"].read_text(encoding="utf-8"))
    metrics_payload = json.loads(result.artifact_paths["metrics"].read_text(encoding="utf-8"))
    if not isinstance(manifest_payload, dict):
        raise AlphaLabDataError("run_manifest.json root must be an object")
    if not isinstance(metrics_payload, dict):
        raise AlphaLabDataError("metrics.json root must be an object")
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

    return contract_rc


def _fmt_text(value: object) -> str:
    text = str(value).strip() if value is not None else ""
    return text if text else "N/A"


if __name__ == "__main__":
    sys.exit(main())
