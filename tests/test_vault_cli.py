"""Tests for vault CLI command wiring."""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import patch

import pytest

from alpha_lab.vault_cli import build_vault_parser, run_vault_command


def test_parser_accepts_all_subcommands() -> None:
    parser = argparse.ArgumentParser()
    build_vault_parser(parser)
    for name in (
        "rebuild-graph",
        "rebuild-embeddings",
        "rebuild-exploration-map",
        "suggest-edges",
        "rebuild-all",
    ):
        args = parser.parse_args([name, "--vault-root", "/tmp/fake"])
        assert args.vault_action == name
        assert args.vault_root == "/tmp/fake"


def test_missing_script_warns_but_returns_int(tmp_path: Path) -> None:
    args = argparse.Namespace(vault_action="rebuild-graph", vault_root=str(tmp_path))
    with patch("alpha_lab.vault_cli.resolve_vault_root", return_value=tmp_path):
        code = run_vault_command(args)
    assert code == 0


def test_rebuild_all_runs_scripts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    protocols = tmp_path / "00_protocols"
    protocols.mkdir()
    for name in ("rebuild-graph.py", "rebuild-embeddings.py", "rebuild-exploration-map.py"):
        (protocols / name).write_text("import sys; print('ok')", encoding="utf-8")

    calls: list[tuple[list[str], bool]] = []

    def _fake_run(cmd: list[str], check: bool) -> object:
        calls.append((cmd, check))

        class _R:
            returncode = 0

        return _R()

    args = argparse.Namespace(vault_action="rebuild-all", vault_root=str(tmp_path))
    with (
        patch("alpha_lab.vault_cli.resolve_vault_root", return_value=tmp_path),
        patch(
            "alpha_lab.vault_cli.subprocess.run",
            side_effect=_fake_run,
        ) as run_mock,
    ):
        code = run_vault_command(args)
    assert code == 0
    assert run_mock.call_count == 3
    assert calls
    assert all(item[1] is False for item in calls)
    assert [cmd[1] for cmd, _ in calls] == [
        str(tmp_path / "00_protocols" / "rebuild-graph.py"),
        str(tmp_path / "00_protocols" / "rebuild-embeddings.py"),
        str(tmp_path / "00_protocols" / "rebuild-exploration-map.py"),
    ]


def test_unresolved_vault_returns_error() -> None:
    args = argparse.Namespace(vault_action="rebuild-graph", vault_root=None)
    with patch("alpha_lab.vault_cli.resolve_vault_root", return_value=None):
        code = run_vault_command(args)
    assert code == 1
