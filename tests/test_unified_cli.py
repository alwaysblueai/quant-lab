from __future__ import annotations

from typing import Any

import pytest

from alpha_lab.cli import main


def test_unified_cli_routes_single_factor_run(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def _fake_single_factor_main(argv: list[str] | None = None) -> int:
        captured["argv"] = argv
        return 11

    monkeypatch.setattr("alpha_lab.real_cases.single_factor.cli.main", _fake_single_factor_main)

    rc = main(["real-case", "single-factor", "run", "spec.yaml"])
    assert rc == 11
    assert captured["argv"] == ["run", "spec.yaml"]


def test_unified_cli_routes_composite_run(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def _fake_composite_main(argv: list[str] | None = None) -> int:
        captured["argv"] = argv
        return 22

    monkeypatch.setattr("alpha_lab.real_cases.composite.cli.main", _fake_composite_main)

    rc = main(["real-case", "composite", "run", "spec.yaml"])
    assert rc == 22
    assert captured["argv"] == ["run", "spec.yaml"]


def test_unified_cli_routes_campaign_run(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def _fake_campaign_main(argv: list[str] | None = None) -> int:
        captured["argv"] = argv
        return 33

    monkeypatch.setattr("alpha_lab.campaigns.research_campaign_1.main", _fake_campaign_main)

    rc = main(["campaign", "run", "research_campaign_1", "--render-report"])
    assert rc == 33
    assert captured["argv"] == ["--render-report"]


def test_unified_cli_passes_through_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def _fake_single_factor_main(argv: list[str] | None = None) -> int:
        captured["argv"] = argv
        return 0

    monkeypatch.setattr("alpha_lab.real_cases.single_factor.cli.main", _fake_single_factor_main)

    rc = main(
        [
            "real-case",
            "single-factor",
            "run",
            "spec.yaml",
            "--render-report",
            "--render-overwrite",
            "--vault-root",
            "/tmp/vault",
            "--vault-export-mode",
            "overwrite",
            "--output-root-dir",
            "/tmp/out",
        ]
    )
    assert rc == 0
    assert captured["argv"] == [
        "run",
        "spec.yaml",
        "--render-report",
        "--render-overwrite",
        "--vault-root",
        "/tmp/vault",
        "--vault-export-mode",
        "overwrite",
        "--output-root-dir",
        "/tmp/out",
    ]


def test_unified_cli_invalid_command_is_helpful(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["campaign", "run", "not_supported"])

    captured = capsys.readouterr()
    assert "invalid choice" in captured.err
