from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .pipeline import run_single_factor_case


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="alpha-lab real-case single-factor",
        description=(
            "Run one real-case single-factor study from spec to standardized artifacts."
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
            vault_root=args.vault_root,
            vault_export_mode=args.vault_export_mode,
        )
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        parser.error(str(exc))

    print("")
    print("  Workflow : real-case-single-factor")
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

    return 0


if __name__ == "__main__":
    sys.exit(main())
