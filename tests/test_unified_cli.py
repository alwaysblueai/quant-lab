from __future__ import annotations

from pathlib import Path
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


def test_unified_cli_routes_model_factor_run(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def _fake_model_factor_main(argv: list[str] | None = None) -> int:
        captured["argv"] = argv
        return 23

    monkeypatch.setattr("alpha_lab.real_cases.model_factor.cli.main", _fake_model_factor_main)

    rc = main(["real-case", "model-factor", "run", "spec.yaml"])
    assert rc == 23
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


def test_unified_cli_routes_data_command(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def _fake_data_main(argv: list[str] | None = None) -> int:
        captured["argv"] = argv
        return 44

    monkeypatch.setattr("alpha_lab.data_store.cli.main", _fake_data_main)

    rc = main(
        [
            "data",
            "export-case-inputs",
            "--start-date",
            "2024-01-01",
            "--end-date",
            "2024-03-31",
            "--output-dir",
            "/tmp/out",
            "--adjustment",
            "qfq",
        ]
    )
    assert rc == 44
    assert captured["argv"] == [
        "export-case-inputs",
        "--slice-preset",
        "standard",
        "--start-date",
        "2024-01-01",
        "--end-date",
        "2024-03-31",
        "--output-dir",
        "/tmp/out",
        "--adjustment",
        "qfq",
    ]


def test_unified_cli_routes_data_query_command(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def _fake_data_main(argv: list[str] | None = None) -> int:
        captured["argv"] = argv
        return 45

    monkeypatch.setattr("alpha_lab.data_store.cli.main", _fake_data_main)

    rc = main(
        [
            "data",
            "query",
            "--sql",
            "select count(*) as n_rows from daily_bars",
            "--format",
            "json",
            "--limit",
            "5",
        ]
    )
    assert rc == 45
    assert captured["argv"] == [
        "query",
        "--sql",
        "select count(*) as n_rows from daily_bars",
        "--format",
        "json",
        "--limit",
        "5",
    ]


def test_unified_cli_routes_bridge_command(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def _fake_bridge_main(argv: list[str] | None = None) -> int:
        captured["argv"] = argv
        return 46

    monkeypatch.setattr("alpha_lab.research_bridge.cli.main", _fake_bridge_main)

    rc = main(
        [
            "bridge",
            "start-round",
            "--project",
            "momentum-factor",
            "--topic",
            "三个月成交额加权动量",
            "--vault-root",
            "/tmp/vault",
        ]
    )
    assert rc == 46
    assert captured["argv"] == [
        "start-round",
        "--project",
        "momentum-factor",
        "--topic",
        "三个月成交额加权动量",
        "--vault-root",
        "/tmp/vault",
    ]


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
            "--evaluation-profile",
            "default_research",
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
        "--evaluation-profile",
        "default_research",
    ]


def test_unified_cli_routes_campaign_profile_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def _fake_campaign_main(argv: list[str] | None = None) -> int:
        captured["argv"] = argv
        return 0

    monkeypatch.setattr("alpha_lab.campaigns.research_campaign_1.main", _fake_campaign_main)

    rc = main(
        [
            "campaign",
            "run",
            "research_campaign_1",
            "--evaluation-profile",
            "default_research",
        ]
    )
    assert rc == 0
    assert captured["argv"] == ["--evaluation-profile", "default_research"]


def test_unified_cli_routes_campaign_compare_profiles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    printed: dict[str, Any] = {}

    result_token = object()

    def _fake_compare_run(**kwargs: Any) -> object:
        captured.update(kwargs)
        return result_token

    def _fake_compare_print(result: object) -> None:
        printed["result"] = result

    monkeypatch.setattr(
        "alpha_lab.campaigns.profile_comparison.run_campaign_profile_comparison",
        _fake_compare_run,
    )
    monkeypatch.setattr(
        "alpha_lab.campaigns.profile_comparison.print_campaign_profile_comparison_summary",
        _fake_compare_print,
    )

    rc = main(
        [
            "campaign",
            "compare-profiles",
            "--source",
            "campaign",
            "--campaign-config",
            "configs/campaigns/research_campaign_1/campaign.yaml",
            "--output-root-dir",
            "/tmp/profile_compare",
            "--profiles",
            "exploratory_screening",
            "default_research",
            "--case-output-root-dir",
            "/tmp/profile_compare_cases",
            "--no-render-report",
            "--render-overwrite",
            "--no-clean-output",
        ]
    )
    assert rc == 0
    assert captured["source"] == "campaign"
    assert captured["campaign_config"] == "configs/campaigns/research_campaign_1/campaign.yaml"
    assert captured["output_root_dir"] == "/tmp/profile_compare"
    assert captured["profiles"] == ("exploratory_screening", "default_research")
    assert captured["pair_mode"] == "adjacent"
    assert captured["case_output_root_dir"] == "/tmp/profile_compare_cases"
    assert captured["artifact_hint_path_mode"] == "relative"
    assert captured["render_report"] is False
    assert captured["render_overwrite"] is True
    assert captured["clean_output"] is False
    assert printed["result"] is result_token


def test_unified_cli_routes_campaign_compare_profiles_artifact_hint_path_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def _fake_compare_run(**kwargs: Any) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        "alpha_lab.campaigns.profile_comparison.run_campaign_profile_comparison",
        _fake_compare_run,
    )
    monkeypatch.setattr(
        "alpha_lab.campaigns.profile_comparison.print_campaign_profile_comparison_summary",
        lambda _: None,
    )

    rc = main(
        [
            "campaign",
            "compare-profiles",
            "--source",
            "example",
            "--artifact-hint-path-mode",
            "absolute",
        ]
    )
    assert rc == 0
    assert captured["artifact_hint_path_mode"] == "absolute"
    assert captured["pair_mode"] == "adjacent"


def test_unified_cli_routes_campaign_compare_profiles_pair_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def _fake_compare_run(**kwargs: Any) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        "alpha_lab.campaigns.profile_comparison.run_campaign_profile_comparison",
        _fake_compare_run,
    )
    monkeypatch.setattr(
        "alpha_lab.campaigns.profile_comparison.print_campaign_profile_comparison_summary",
        lambda _: None,
    )

    rc = main(
        [
            "campaign",
            "compare-profiles",
            "--source",
            "example",
            "--pair-mode",
            "all_pairs",
        ]
    )
    assert rc == 0
    assert captured["pair_mode"] == "all_pairs"


def test_unified_cli_routes_campaign_compare_profiles_case_evidence_drill_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    printed: dict[str, Any] = {}
    drilled: dict[str, Any] = {}

    result_token = object()

    def _fake_compare_run(**kwargs: Any) -> object:
        captured.update(kwargs)
        return result_token

    def _fake_compare_print(result: object) -> None:
        printed["result"] = result

    def _fake_case_evidence_print(result: object, *, case_name: str) -> None:
        drilled["result"] = result
        drilled["case_name"] = case_name

    monkeypatch.setattr(
        "alpha_lab.campaigns.profile_comparison.run_campaign_profile_comparison",
        _fake_compare_run,
    )
    monkeypatch.setattr(
        "alpha_lab.campaigns.profile_comparison.print_campaign_profile_comparison_summary",
        _fake_compare_print,
    )
    monkeypatch.setattr(
        "alpha_lab.campaigns.profile_comparison.print_campaign_profile_case_evidence",
        _fake_case_evidence_print,
    )

    rc = main(
        [
            "campaign",
            "compare-profiles",
            "--source",
            "example",
            "--show-case-evidence",
            "case_alpha",
        ]
    )
    assert rc == 0
    assert captured["source"] == "example"
    assert printed["result"] is result_token
    assert drilled["result"] is result_token
    assert drilled["case_name"] == "case_alpha"


def test_unified_cli_routes_campaign_render_dashboard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def _fake_write(
        comparison_json_path: str,
        *,
        output_path: str | None = None,
        overwrite: bool = False,
        title: str | None = None,
        artifact_load_mode: str = "permissive",
    ) -> Path:
        captured["comparison_json_path"] = comparison_json_path
        captured["output_path"] = output_path
        captured["overwrite"] = overwrite
        captured["title"] = title
        captured["artifact_load_mode"] = artifact_load_mode
        return Path("/tmp/fake_dashboard.html")

    monkeypatch.setattr(
        "alpha_lab.reporting.renderers.write_campaign_profile_dashboard_html",
        _fake_write,
    )

    rc = main(
        [
            "campaign",
            "render-dashboard",
            "--comparison-json",
            "/tmp/campaign_profile_comparison.json",
            "--output-html",
            "/tmp/dashboard.html",
            "--overwrite",
            "--title",
            "中文看板",
        ]
    )
    assert rc == 0
    assert captured["comparison_json_path"] == "/tmp/campaign_profile_comparison.json"
    assert captured["output_path"] == "/tmp/dashboard.html"
    assert captured["overwrite"] is True
    assert captured["title"] == "中文看板"
    assert captured["artifact_load_mode"] == "permissive"


def test_unified_cli_routes_campaign_render_dashboard_with_strict_artifact_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def _fake_write(
        comparison_json_path: str,
        *,
        output_path: str | None = None,
        overwrite: bool = False,
        title: str | None = None,
        artifact_load_mode: str = "permissive",
    ) -> Path:
        captured["comparison_json_path"] = comparison_json_path
        captured["artifact_load_mode"] = artifact_load_mode
        return Path("/tmp/fake_dashboard.html")

    monkeypatch.setattr(
        "alpha_lab.reporting.renderers.write_campaign_profile_dashboard_html",
        _fake_write,
    )

    rc = main(
        [
            "campaign",
            "render-dashboard",
            "--comparison-json",
            "/tmp/campaign_profile_comparison.json",
            "--artifact-load-mode",
            "strict",
        ]
    )
    assert rc == 0
    assert captured["comparison_json_path"] == "/tmp/campaign_profile_comparison.json"
    assert captured["artifact_load_mode"] == "strict"


def test_unified_cli_render_dashboard_surfaces_artifact_load_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from alpha_lab.reporting.renderers.campaign_profile_dashboard import (
        ArtifactLoadRuntimeError,
    )
    from alpha_lab.reporting.renderers.research_dashboard_schema import (
        ArtifactLoadDiagnostic,
    )

    def _fake_write(
        comparison_json_path: str,
        *,
        output_path: str | None = None,
        overwrite: bool = False,
        title: str | None = None,
        artifact_load_mode: str = "permissive",
    ) -> Path:
        raise ArtifactLoadRuntimeError(
            "strict artifact load checks failed:\n  - missing canonical artifact",
            diagnostics=(
                ArtifactLoadDiagnostic(
                    code="MISSING_CANONICAL_ARTIFACT",
                    severity="error",
                    artifact_type="canonical_artifact",
                    object_scope="factor_definition",
                    message=(
                        "case_alpha (default_research) factor_definition: "
                        "missing artifact path"
                    ),
                    mode="strict",
                ),
            ),
        )

    monkeypatch.setattr(
        "alpha_lab.reporting.renderers.write_campaign_profile_dashboard_html",
        _fake_write,
    )

    with pytest.raises(SystemExit):
        main(
            [
                "campaign",
                "render-dashboard",
                "--comparison-json",
                "/tmp/campaign_profile_comparison.json",
                "--artifact-load-mode",
                "strict",
            ]
        )
    captured = capsys.readouterr()
    assert "strict artifact load checks failed" in captured.err


@pytest.mark.skip(
    reason="alpha_lab.experimental_level3 is not yet committed; remains out of Round 4 scope"
)
def test_unified_cli_routes_experimental_single_factor_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def _fake_main(argv: list[str] | None = None) -> int:
        captured["argv"] = argv
        return 44

    monkeypatch.setattr("alpha_lab.experimental_level3.single_factor_package.main", _fake_main)

    rc = main(
        [
            "experimental",
            "single-factor-package",
            "run",
            "--include-experimental-level3",
        ]
    )
    assert rc == 44
    assert captured["argv"] == ["--include-experimental-level3"]


@pytest.mark.skip(
    reason="alpha_lab.experimental_level3 is not yet committed; remains out of Round 4 scope"
)
def test_unified_cli_routes_experimental_execution_realism_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def _fake_main(argv: list[str] | None = None) -> int:
        captured["argv"] = argv
        return 55

    monkeypatch.setattr(
        "alpha_lab.experimental_level3.execution_realism_package.main",
        _fake_main,
    )

    rc = main(
        [
            "experimental",
            "execution-realism-package",
            "run",
            "--case-id",
            "ashare_execution_realism",
        ]
    )
    assert rc == 55
    assert captured["argv"] == ["--case-id", "ashare_execution_realism"]


@pytest.mark.skip(
    reason="alpha_lab.experimental_level3 is not yet committed; remains out of Round 4 scope"
)
def test_unified_cli_routes_experimental_factor_health_monitor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def _fake_main(argv: list[str] | None = None) -> int:
        captured["argv"] = argv
        return 66

    monkeypatch.setattr(
        "alpha_lab.experimental_level3.factor_health_monitor.main",
        _fake_main,
    )

    rc = main(
        [
            "experimental",
            "factor-health-monitor",
            "run",
            "--case-id",
            "factor_health_monitor_v1",
        ]
    )
    assert rc == 66
    assert captured["argv"] == ["--case-id", "factor_health_monitor_v1"]


def test_unified_cli_routes_experimental_vault_export_gate_detect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def _fake_main(argv: list[str] | None = None) -> int:
        captured["argv"] = argv
        return 77

    monkeypatch.setattr("alpha_lab.vault_export_gate.main", _fake_main)

    rc = main(
        [
            "experimental",
            "vault-export-gate",
            "detect",
            "--transcript-path",
            "/tmp/session.jsonl",
        ]
    )
    assert rc == 77
    assert captured["argv"] == ["detect", "--transcript-path", "/tmp/session.jsonl"]


def test_unified_cli_routes_experimental_vault_export_gate_apply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def _fake_main(argv: list[str] | None = None) -> int:
        captured["argv"] = argv
        return 88

    monkeypatch.setattr("alpha_lab.vault_export_gate.main", _fake_main)

    rc = main(
        [
            "experimental",
            "vault-export-gate",
            "apply",
            "--manifest-path",
            "/tmp/run_manifest.json",
            "--vault-root",
            "/tmp/vault",
        ]
    )
    assert rc == 88
    assert captured["argv"] == [
        "apply",
        "--manifest-path",
        "/tmp/run_manifest.json",
        "--vault-root",
        "/tmp/vault",
    ]


def test_unified_cli_invalid_command_is_helpful(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["campaign", "run", "not_supported"])

    captured = capsys.readouterr()
    assert "invalid choice" in captured.err


def test_unified_cli_profiles_lists_available_profiles(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = main(["profiles"])
    assert rc == 0
    captured = capsys.readouterr()
    assert "Available Evaluation Profiles" in captured.out
    assert "default_research" in captured.out
    assert "exploratory_screening" in captured.out
    assert "stricter_research" in captured.out
    assert "candidate discovery" in captured.out


def test_unified_cli_routes_web_ui(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def _fake_start_web_ui_server(
        *,
        host: str = "127.0.0.1",
        port: int = 8765,
        workspace_root: str = ".",
        open_browser: bool = True,
    ) -> None:
        captured["host"] = host
        captured["port"] = port
        captured["workspace_root"] = workspace_root
        captured["open_browser"] = open_browser

    monkeypatch.setattr("alpha_lab.web_ui.start_web_ui_server", _fake_start_web_ui_server)

    rc = main(
        [
            "web",
            "ui",
            "--host",
            "0.0.0.0",
            "--port",
            "8899",
            "--workspace-root",
            "/tmp/alpha-lab",
            "--no-open-browser",
        ]
    )
    assert rc == 0
    assert captured == {
        "host": "0.0.0.0",
        "port": 8899,
        "workspace_root": "/tmp/alpha-lab",
        "open_browser": False,
    }


def test_unified_cli_routes_web_cockpit(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def _fake_start_web_cockpit_server(
        *,
        host: str = "127.0.0.1",
        port: int = 8766,
        workspace_root: str = ".",
        vault_root: str | None = None,
        open_browser: bool = True,
    ) -> None:
        captured["host"] = host
        captured["port"] = port
        captured["workspace_root"] = workspace_root
        captured["vault_root"] = vault_root
        captured["open_browser"] = open_browser

    monkeypatch.setattr(
        "alpha_lab.web_cockpit.start_web_cockpit_server",
        _fake_start_web_cockpit_server,
    )

    rc = main(
        [
            "web",
            "cockpit",
            "--host",
            "0.0.0.0",
            "--port",
            "8999",
            "--workspace-root",
            "/tmp/alpha-lab",
            "--vault-root",
            "/tmp/vault",
            "--no-open-browser",
        ]
    )
    assert rc == 0
    assert captured == {
        "host": "0.0.0.0",
        "port": 8999,
        "workspace_root": "/tmp/alpha-lab",
        "vault_root": "/tmp/vault",
        "open_browser": False,
    }


def test_unified_cli_web_ui_invalid_port(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["web", "ui", "--port", "0"])
    captured = capsys.readouterr()
    assert "--port must be within 1..65535" in captured.err


def test_unified_cli_run_routes_to_legacy_main(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def _fake_legacy_main(argv: list[str] | None = None) -> int:
        captured["argv"] = argv
        return 55

    monkeypatch.setattr("alpha_lab.cli._legacy_main", _fake_legacy_main)

    rc = main(["run", "--input-path", "prices.csv"])
    assert rc == 55
    assert captured["argv"] == ["--input-path", "prices.csv"]


def test_unified_cli_top_level_help_is_router(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit):
        main(["--help"])
    captured = capsys.readouterr()
    assert "{run,real-case,campaign,profiles,web,bridge,data,experimental}" in captured.out


def test_unified_cli_campaign_help_lists_compare_profiles(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit):
        main(["campaign", "--help"])
    captured = capsys.readouterr()
    assert "compare-profiles" in captured.out
    assert "render-dashboard" in captured.out


def test_unified_cli_campaign_compare_profiles_help_is_discoverable(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit):
        main(["campaign", "compare-profiles", "--help"])
    captured = capsys.readouterr()
    assert "--source" in captured.out
    assert "--campaign-config" in captured.out
    assert "--profiles" in captured.out
    assert "--pair-mode" in captured.out
    assert "--output-root-dir" in captured.out
    assert "--artifact-hint-path-mode" in captured.out
    assert "--show-case-evidence" in captured.out
