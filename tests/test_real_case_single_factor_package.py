from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

_RUNNER_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_real_case_single_factor_package.py"
_RUNNER_SPEC = importlib.util.spec_from_file_location(
    "run_real_case_single_factor_package",
    _RUNNER_PATH,
)
assert _RUNNER_SPEC is not None and _RUNNER_SPEC.loader is not None
runner = importlib.util.module_from_spec(_RUNNER_SPEC)
sys.modules[_RUNNER_SPEC.name] = runner
_RUNNER_SPEC.loader.exec_module(runner)


def _write_case_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    case_dir = tmp_path / "single_case_output"
    case_dir.mkdir(parents=True, exist_ok=True)

    handoff_bundle = case_dir / "handoff" / "demo_bundle"
    handoff_bundle.mkdir(parents=True, exist_ok=True)
    (handoff_bundle / "manifest.json").write_text(
        json.dumps({"experiment_id": "demo_case"}, sort_keys=True),
        encoding="utf-8",
    )

    workflow_summary_path = case_dir / "demo_case_single_factor_workflow_summary.json"
    workflow_summary_path.write_text(
        json.dumps({"workflow": "run-single-factor", "experiment_name": "demo_case"}),
        encoding="utf-8",
    )

    price_path = tmp_path / "prices.csv"
    pd.DataFrame(
        [
            {"date": "2024-01-02", "asset": "A", "close": 10.0},
            {"date": "2024-01-03", "asset": "A", "close": 10.2},
        ]
    ).to_csv(price_path, index=False)

    return case_dir, handoff_bundle, workflow_summary_path, price_path


def _install_runner_stubs(
    *,
    monkeypatch: Any,
    captured_package_kwargs: list[dict[str, Any]],
) -> None:
    def _fake_load_bundle(path: str | Path) -> object:
        return {"bundle_path": str(Path(path).resolve())}

    def _fake_run_vectorbt_backtest(_bundle: object, config: Any) -> object:
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "backtest_summary.json").write_text(
            json.dumps({"engine": "vectorbt"}),
            encoding="utf-8",
        )
        return object()

    def _fake_run_backtrader_backtest(_bundle: object, config: Any) -> object:
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "backtest_summary.json").write_text(
            json.dumps({"engine": "backtrader"}),
            encoding="utf-8",
        )
        return object()

    def _fake_build_execution_impact_report(
        run_path: str | Path,
        *,
        comparison_run_path: str | Path | None = None,
    ) -> dict[str, str]:
        return {
            "run_path": str(Path(run_path)),
            "comparison_run_path": (
                str(Path(comparison_run_path)) if comparison_run_path is not None else ""
            ),
        }

    def _fake_export_execution_impact_report(
        _report: object,
        *,
        output_dir: str | Path | None = None,
        export_reason_csv: bool = True,
        export_timeseries_csv: bool = True,
    ) -> dict[str, Path]:
        del export_reason_csv, export_timeseries_csv
        out_dir = Path(output_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        report_path = out_dir / "execution_impact_report.json"
        report_path.write_text(json.dumps({"ok": True}), encoding="utf-8")
        return {"report_json": report_path}

    def _fake_build_research_package(case_output_dir: str | Path, **kwargs: Any) -> object:
        captured: dict[str, Any] = {"case_output_dir": Path(case_output_dir).resolve()}
        captured.update(kwargs)
        captured_package_kwargs.append(captured)
        return object()

    def _fake_export_research_package(
        _package: object,
        *,
        output_dir: str | Path,
        export_artifact_index: bool = True,
    ) -> dict[str, Path]:
        del export_artifact_index
        out_dir = Path(output_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / "research_package.json"
        md_path = out_dir / "research_package.md"
        json_path.write_text(json.dumps({"ok": True}), encoding="utf-8")
        md_path.write_text("# package\n", encoding="utf-8")
        return {"package_json": json_path, "package_markdown": md_path}

    monkeypatch.setattr(runner, "load_backtest_input_bundle", _fake_load_bundle)
    monkeypatch.setattr(runner, "run_vectorbt_backtest", _fake_run_vectorbt_backtest)
    monkeypatch.setattr(runner, "run_backtrader_backtest", _fake_run_backtrader_backtest)
    monkeypatch.setattr(runner, "build_execution_impact_report", _fake_build_execution_impact_report)
    monkeypatch.setattr(runner, "export_execution_impact_report", _fake_export_execution_impact_report)
    monkeypatch.setattr(runner, "build_research_package", _fake_build_research_package)
    monkeypatch.setattr(runner, "export_research_package", _fake_export_research_package)


def test_dirty_run_dir_is_cleaned_and_packaging_uses_explicit_current_run_paths(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    case_dir, handoff_bundle, workflow_summary_path, price_path = _write_case_inputs(tmp_path)
    run_id = "run-001"
    stale_file = case_dir / runner.RUNS_ROOT_NAME / run_id / "stale.txt"
    stale_file.parent.mkdir(parents=True, exist_ok=True)
    stale_file.write_text("stale", encoding="utf-8")

    captured_package_kwargs: list[dict[str, Any]] = []
    _install_runner_stubs(
        monkeypatch=monkeypatch,
        captured_package_kwargs=captured_package_kwargs,
    )

    rc = runner.main(
        [
            "--handoff-bundle",
            str(handoff_bundle),
            "--price-path",
            str(price_path),
            "--case-output-dir",
            str(case_dir),
            "--run-id",
            run_id,
        ]
    )
    assert rc == 0
    assert not stale_file.exists()

    run_output_dir = (case_dir / runner.RUNS_ROOT_NAME / run_id).resolve()
    assert len(captured_package_kwargs) == 1
    call_kwargs = captured_package_kwargs[0]
    assert call_kwargs["case_output_dir"] == run_output_dir
    assert call_kwargs["workflow_summary_path"] == workflow_summary_path.resolve()
    assert call_kwargs["handoff_bundle_path"] == handoff_bundle.resolve()
    assert call_kwargs["replay_run_dirs"] == {
        "backtrader": run_output_dir / "replay_compare" / "backtrader",
        "vectorbt": run_output_dir / "replay_compare" / "vectorbt",
    }
    assert call_kwargs["execution_impact_report_path"] == (
        run_output_dir / "execution_impact" / "execution_impact_report.json"
    )

    summary_path = run_output_dir / "real_single_factor_package_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["run_id"] == run_id
    assert summary["run_output_dir"] == str(run_output_dir)
    assert summary["research_package_json"].startswith(str(run_output_dir))
    assert summary["execution_impact_report"].startswith(str(run_output_dir))


def test_repeated_runs_package_only_current_run_artifacts(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    case_dir, handoff_bundle, _workflow_summary_path, price_path = _write_case_inputs(tmp_path)
    captured_package_kwargs: list[dict[str, Any]] = []
    _install_runner_stubs(
        monkeypatch=monkeypatch,
        captured_package_kwargs=captured_package_kwargs,
    )

    rc_first = runner.main(
        [
            "--handoff-bundle",
            str(handoff_bundle),
            "--price-path",
            str(price_path),
            "--case-output-dir",
            str(case_dir),
            "--run-id",
            "run-a",
        ]
    )
    rc_second = runner.main(
        [
            "--handoff-bundle",
            str(handoff_bundle),
            "--price-path",
            str(price_path),
            "--case-output-dir",
            str(case_dir),
            "--run-id",
            "run-b",
        ]
    )
    assert rc_first == 0
    assert rc_second == 0
    assert len(captured_package_kwargs) == 2

    first_run_dir = (case_dir / runner.RUNS_ROOT_NAME / "run-a").resolve()
    second_run_dir = (case_dir / runner.RUNS_ROOT_NAME / "run-b").resolve()
    first_kwargs = captured_package_kwargs[0]
    second_kwargs = captured_package_kwargs[1]

    assert first_kwargs["case_output_dir"] == first_run_dir
    assert second_kwargs["case_output_dir"] == second_run_dir
    assert second_kwargs["replay_run_dirs"] == {
        "backtrader": second_run_dir / "replay_compare" / "backtrader",
        "vectorbt": second_run_dir / "replay_compare" / "vectorbt",
    }
    assert second_kwargs["execution_impact_report_path"] == (
        second_run_dir / "execution_impact" / "execution_impact_report.json"
    )

    summary_second = json.loads(
        (second_run_dir / "real_single_factor_package_summary.json").read_text(encoding="utf-8")
    )
    assert summary_second["run_id"] == "run-b"
    assert str(first_run_dir) not in json.dumps(summary_second, sort_keys=True)
