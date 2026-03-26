from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .pipeline import run_composite_case


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
        )
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        parser.error(str(exc))

    print("")
    print("  Workflow : real-case-composite")
    print("  Status   : success")
    print(f"  Case     : {result.spec.name}")
    print(f"  Output   : {result.output_dir}")
    print(f"  Manifest : {result.artifact_paths['run_manifest']}")
    print(f"  Metrics  : {result.artifact_paths['metrics']}")
    print(f"  Summary  : {result.artifact_paths['summary']}")
    print(f"  Card     : {result.artifact_paths['experiment_card']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
