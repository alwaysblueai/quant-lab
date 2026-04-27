"""CLI subcommands for vault computed-layer management.

Canonical entry point: ``alpha-lab vault <subcommand>``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys

from alpha_lab.vault_export import resolve_vault_root

_SINGLE_SCRIPTS = (
    ("rebuild-graph", "Rebuild 90_computed/graph.json from vault cards.", "rebuild-graph.py"),
    (
        "rebuild-embeddings",
        "Rebuild 90_computed/embeddings.npz from vault cards.",
        "rebuild-embeddings.py",
    ),
    (
        "rebuild-exploration-map",
        "Rebuild 90_computed/exploration_map.json.",
        "rebuild-exploration-map.py",
    ),
    ("suggest-edges", "Run embedding-based typed-edge suggestions.", "suggest-edges.py"),
)

_REBUILD_ALL_SCRIPTS = ("rebuild-graph.py", "rebuild-embeddings.py", "rebuild-exploration-map.py")


def build_vault_parser(parser: argparse.ArgumentParser) -> None:
    parser.description = "Manage quant-knowledge vault computed layer."
    commands = parser.add_subparsers(dest="vault_action", required=True)

    for name, help_text, _ in _SINGLE_SCRIPTS:
        cmd = commands.add_parser(
            name,
            help=help_text,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        cmd.add_argument(
            "--vault-root",
            default=None,
            help="Quant-knowledge vault root. Defaults to OBSIDIAN_VAULT_PATH.",
        )

    all_cmd = commands.add_parser(
        "rebuild-all",
        help="Run rebuild-graph, rebuild-embeddings, and rebuild-exploration-map in sequence.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    all_cmd.add_argument(
        "--vault-root",
        default=None,
        help="Quant-knowledge vault root. Defaults to OBSIDIAN_VAULT_PATH.",
    )


def run_vault_command(args: argparse.Namespace) -> int:
    vault_root = resolve_vault_root(getattr(args, "vault_root", None))
    if vault_root is None:
        print(
            "Error: vault root unresolved; pass --vault-root or set OBSIDIAN_VAULT_PATH",
            file=sys.stderr,
        )
        return 1

    if args.vault_action == "rebuild-all":
        scripts = list(_REBUILD_ALL_SCRIPTS)
    else:
        script_map = {name: script for name, _, script in _SINGLE_SCRIPTS}
        scripts = [script_map[args.vault_action]]

    exit_code = 0
    for script_name in scripts:
        script_path = vault_root / "00_protocols" / script_name
        if not script_path.exists():
            print(f"Warning: {script_path} not found, skipping.", file=sys.stderr)
            continue
        result = subprocess.run(
            [sys.executable, str(script_path), str(vault_root)],
            check=False,
        )
        if result.returncode != 0:
            exit_code = 1
    return exit_code
