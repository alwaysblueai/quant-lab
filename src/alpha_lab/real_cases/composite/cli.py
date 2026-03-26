from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from alpha_lab.reporting.renderers import write_case_report

from .pipeline import run_composite_case

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="alpha-lab real-case composite",
        description=(
            "Run one real-case composite factor study from spec to standardized artifacts."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run",
        help="Run one composite case and export artifacts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    run_parser.add_argument("spec_path", help="Path to composite case YAML/JSON spec.")
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

    spec_path = Path(args.spec_path)
    try:
        result = run_composite_case(
            spec_path,
            output_root_dir=args.output_root_dir,
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
    _update_run_manifest(
        result.artifact_paths["run_manifest"],
        render_meta,
    )

    print("")
    print("  Workflow : real-case-composite")
    print("  Status   : success")
    print(f"  Case     : {result.spec.name}")
    print(f"  Output   : {result.output_dir}")
    print(f"  Manifest : {result.artifact_paths['run_manifest']}")
    print(f"  Metrics  : {result.artifact_paths['metrics']}")
    print(f"  Summary  : {result.artifact_paths['summary']}")
    print(f"  Card     : {result.artifact_paths['experiment_card']}")
    manifest_payload = json.loads(result.artifact_paths["run_manifest"].read_text(encoding="utf-8"))
    vault_meta = manifest_payload.get("vault_export", {})
    print(f"  Vault Export Status : {vault_meta.get('status')}")
    print(f"  Vault Export Mode   : {vault_meta.get('mode')}")
    print(f"  Report Render Status: {manifest_payload.get('render_status')}")
    print(f"  Report Path         : {manifest_payload.get('rendered_report_path')}")
    return 0


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


if __name__ == "__main__":
    sys.exit(main())
