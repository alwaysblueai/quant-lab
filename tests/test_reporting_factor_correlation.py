from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from alpha_lab.reporting import inspect_run_factor_correlation as inspect_from_reporting_root
from alpha_lab.reporting.factor_correlation import (
    collect_run_factor_correlation_summary,
    compute_factor_correlation,
    inspect_run_factor_correlation,
)
from alpha_lab.reporting.factor_decomposition import inspect_run_decomposition


def test_inspect_run_factor_correlation_reads_top_match_payload(tmp_path: Path) -> None:
    run_root = tmp_path / "run_root"
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "factor_correlation.json").write_text(
        json.dumps(
            {
                "top_match": "Remote Value",
                "max_abs_correlation": 0.82,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    signal = inspect_run_factor_correlation(run_root)
    assert signal is not None
    assert signal.top_match == "Remote Value"
    assert signal.max_abs_correlation == 0.82
    assert signal.redundant is True


def test_collect_run_factor_correlation_summary_reads_ranked_matches(tmp_path: Path) -> None:
    run_root = tmp_path / "run_root"
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "decomposition.json").write_text(
        json.dumps(
            {
                "ranked_matches": [
                    {"name": "Momentum Base", "score": 0.66},
                    {"name": "Remote Value", "score": 0.81},
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    summary = collect_run_factor_correlation_summary(run_root, limit=3)
    assert len(summary) >= 2
    names = {item.name for item in summary}
    assert "Momentum Base" in names
    assert "Remote Value" in names
    remote = next(item for item in summary if item.name == "Remote Value")
    assert remote.redundant is True
    assert remote.source == "decomposition"


def test_inspect_run_decomposition_remains_compatible(tmp_path: Path) -> None:
    run_root = tmp_path / "run_root"
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "factor_correlation.json").write_text(
        json.dumps(
            {
                "top_match": "Momentum Base",
                "max_abs_correlation": 0.73,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    signal = inspect_run_decomposition(run_root)
    assert signal is not None
    assert signal.top_match == "Momentum Base"
    assert signal.redundant is True


def test_reporting_root_exports_factor_correlation_api(tmp_path: Path) -> None:
    run_root = tmp_path / "run_root"
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "factor_correlation.json").write_text(
        json.dumps({"top_match": "Alpha", "max_abs_correlation": 0.75}, indent=2),
        encoding="utf-8",
    )
    signal = inspect_from_reporting_root(run_root)
    assert signal is not None
    assert signal.top_match == "Alpha"


def test_reporting_factor_correlation_compute_function() -> None:
    compute = compute_factor_correlation(
        pd.Series([1.0, 0.2, 0.9, 0.3, -0.1, 0.0, 0.8, 0.5], index=range(8)),
        {"ref": pd.Series([1.1, 0.25, 1.0, 0.28, -0.05, -0.01, 0.88, 0.55])},
        candidate_name="candidate",
    )
    assert compute.candidate_name == "candidate"
    assert len(compute.correlations) == 1
    assert compute.correlations[0].factor_name == "ref"
