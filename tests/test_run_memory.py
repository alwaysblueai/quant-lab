"""Tests for the soft run-memory budget + RSS telemetry (P1-D, telemetry scope)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from alpha_lab.exceptions import AlphaLabMemoryError
from alpha_lab.real_cases.single_factor.pipeline import run_single_factor_case
from alpha_lab.run_memory import (
    MAX_RSS_ENV_VAR,
    RESOURCE_USAGE_ARTIFACT_NAME,
    RunMemoryMonitor,
)
from tests.single_factor_case_helpers import write_demo_single_factor_case


def test_monitor_without_budget_never_raises() -> None:
    monitor = RunMemoryMonitor(None)
    monitor.sample("load")
    monitor.check("load")  # no budget -> no enforcement
    assert monitor.max_rss_mb is None
    assert monitor.peak_rss_mb is not None and monitor.peak_rss_mb > 0


def test_monitor_raises_when_peak_exceeds_budget() -> None:
    # The process is comfortably larger than 1 MB, so any sample trips a 1 MB budget.
    monitor = RunMemoryMonitor(1.0, label="demo_case")
    monitor.sample("load_inputs")
    with pytest.raises(AlphaLabMemoryError, match="memory budget exceeded"):
        monitor.check("load_inputs")


def test_stage_context_records_rss_and_enforces_on_success() -> None:
    monitor = RunMemoryMonitor(1.0)
    with pytest.raises(AlphaLabMemoryError):
        with monitor.stage("evaluate"):
            pass
    # RSS for the stage was still recorded despite the budget breach.
    assert "evaluate" in monitor.snapshot()["stage_rss_mb"]


def test_stage_context_does_not_mask_body_error() -> None:
    monitor = RunMemoryMonitor(1.0)
    # A genuine error inside the stage must propagate, not be replaced by the
    # budget error, and the stage RSS is still recorded.
    with pytest.raises(ValueError, match="boom"):
        with monitor.stage("load_inputs"):
            raise ValueError("boom")
    assert "load_inputs" in monitor.snapshot()["stage_rss_mb"]


def test_from_env_parses_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(MAX_RSS_ENV_VAR, "1500")
    assert RunMemoryMonitor.from_env().max_rss_mb == 1500.0

    monkeypatch.setenv(MAX_RSS_ENV_VAR, "not-a-number")
    assert RunMemoryMonitor.from_env().max_rss_mb is None

    monkeypatch.delenv(MAX_RSS_ENV_VAR, raising=False)
    assert RunMemoryMonitor.from_env().max_rss_mb is None


def test_snapshot_has_expected_shape() -> None:
    monitor = RunMemoryMonitor(2000.0, label="c")
    monitor.sample("a")
    snap = monitor.snapshot()
    assert snap["artifact_type"] == "alpha_lab_resource_usage"
    assert snap["max_rss_mb_budget"] == 2000.0
    assert snap["monitor_available"] is True
    assert isinstance(snap["stage_rss_mb"], dict)


def test_write_resource_usage_handles_missing_dir(tmp_path: Path) -> None:
    monitor = RunMemoryMonitor(None)
    monitor.sample("x")
    # Missing directory -> no write, returns None (early-failure safe).
    assert monitor.write_resource_usage(tmp_path / "does_not_exist") is None
    # Existing directory -> artifact written.
    path = monitor.write_resource_usage(tmp_path)
    assert path is not None and path.name == RESOURCE_USAGE_ARTIFACT_NAME
    assert json.loads(path.read_text())["artifact_type"] == "alpha_lab_resource_usage"


def test_run_single_factor_case_writes_resource_usage(tmp_path: Path) -> None:
    spec_path = write_demo_single_factor_case(tmp_path, factor_name="bp")
    result = run_single_factor_case(spec_path, evaluation_profile="exploratory_screening")

    payload = json.loads((result.output_dir / RESOURCE_USAGE_ARTIFACT_NAME).read_text())
    assert payload["artifact_type"] == "alpha_lab_resource_usage"
    assert payload["max_rss_mb_budget"] is None  # no budget set by default
    stages = payload["stage_rss_mb"]
    assert {"load_inputs", "evaluate"}.issubset(set(stages))
    assert payload["peak_rss_mb"] >= max(stages.values())


def test_run_single_factor_case_enforces_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec_path = write_demo_single_factor_case(tmp_path, factor_name="bp")
    # A 1 MB budget is far below any real run -> the first stage check trips it.
    monkeypatch.setenv(MAX_RSS_ENV_VAR, "1")
    with pytest.raises(AlphaLabMemoryError, match="memory budget exceeded"):
        run_single_factor_case(spec_path, evaluation_profile="exploratory_screening")
