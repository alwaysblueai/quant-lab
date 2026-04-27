from __future__ import annotations

import difflib
import json
import os
from pathlib import Path

from alpha_lab.artifact_contracts import validate_level12_artifact_payload
from alpha_lab.examples.profile_aware_campaign_level12 import (
    run_profile_aware_campaign_level12_example,
)
from alpha_lab.real_cases.single_factor.pipeline import run_single_factor_case
from alpha_lab.reporting.research_validation_package import (
    build_research_validation_package,
    export_research_validation_package,
)
from tests.single_factor_case_helpers import write_demo_single_factor_case

GOLDEN_ROOT = Path(__file__).resolve().parent / "goldens" / "artifact_regression"
UPDATE_GOLDENS_ENV = "ALPHA_LAB_UPDATE_GOLDENS"
SCRUBBED_TIMESTAMP_TOKEN = "<SCRUBBED_TIMESTAMP>"
VOLATILE_JSON_KEYS = frozenset(
    {
        "created_at_utc",
        "generated_at_utc",
        "run_timestamp_utc",
    }
)


def test_single_factor_level12_core_artifacts_match_golden(tmp_path: Path) -> None:
    """Golden coverage for one deterministic Level 1/2 single-factor run."""
    spec_path = write_demo_single_factor_case(tmp_path, factor_name="bp")
    output_root = tmp_path / "single_factor_level12"
    result = run_single_factor_case(
        spec_path,
        output_root_dir=output_root,
        vault_export_mode="skip",
    )

    replacements = {
        str(output_root.resolve()): "<OUTPUT_ROOT>",
    }

    _assert_json_matches_golden(
        artifact_path=result.artifact_paths["metrics"],
        golden_relative_path=Path("single_factor/metrics.json"),
        replacements=replacements,
    )
    _assert_text_matches_golden(
        golden_relative_path=Path("single_factor/summary.md"),
        actual_text=(result.artifact_paths["summary"]).read_text(encoding="utf-8"),
        replacements=replacements,
    )
    _assert_json_matches_golden(
        artifact_path=result.artifact_paths["portfolio_validation_summary"],
        golden_relative_path=Path(
            "single_factor/level2_portfolio_validation/portfolio_validation_summary.json"
        ),
        replacements=replacements,
    )
    _assert_json_matches_golden(
        artifact_path=result.artifact_paths["portfolio_validation_package"],
        golden_relative_path=Path(
            "single_factor/level2_portfolio_validation/portfolio_validation_package.json"
        ),
        replacements=replacements,
    )
    _write_research_validation_workflow_summary(
        output_dir=result.output_dir,
        case_name=result.spec.name,
        spec_path=spec_path,
        metrics_path=result.artifact_paths["metrics"],
        portfolio_validation_summary_path=result.artifact_paths["portfolio_validation_summary"],
        portfolio_validation_metrics_path=result.artifact_paths["portfolio_validation_metrics"],
        portfolio_validation_package_path=result.artifact_paths["portfolio_validation_package"],
    )
    package = build_research_validation_package(
        result.output_dir,
        case_id=result.spec.name,
        case_name=result.spec.name,
    )
    exported_package = export_research_validation_package(
        package,
        output_root / "research_validation_package",
    )
    package_payload = json.loads(exported_package["json"].read_text(encoding="utf-8"))
    if not isinstance(package_payload, dict):
        raise AssertionError(f"{exported_package['json']} must contain a JSON object")
    validate_level12_artifact_payload(
        package_payload,
        artifact_name=exported_package["json"].name,
        source=exported_package["json"],
    )
    _assert_text_matches_golden(
        golden_relative_path=Path("single_factor/research_validation_package.md"),
        actual_text=exported_package["markdown"].read_text(encoding="utf-8"),
        replacements=replacements,
    )


def test_campaign_profile_comparison_artifacts_match_golden(tmp_path: Path) -> None:
    """Golden coverage for deterministic Level 1/2 campaign profile comparison outputs."""
    output_root = tmp_path / "campaign_profile_comparison"
    result = run_profile_aware_campaign_level12_example(
        output_root_dir=output_root,
        profiles=("exploratory_screening", "default_research"),
        render_report=False,
        clean_output=True,
    )

    replacements = {
        str(output_root.resolve()): "<OUTPUT_ROOT>",
    }

    _assert_json_matches_golden(
        artifact_path=result.comparison_json_path,
        golden_relative_path=Path("campaign_profile_comparison/campaign_profile_comparison.json"),
        replacements=replacements,
    )
    _assert_text_matches_golden(
        golden_relative_path=Path("campaign_profile_comparison/campaign_profile_comparison.md"),
        actual_text=result.comparison_markdown_path.read_text(encoding="utf-8"),
        replacements=replacements,
    )
    _assert_text_matches_golden(
        golden_relative_path=Path("campaign_profile_comparison/campaign_profile_case_matrix.csv"),
        actual_text=result.comparison_csv_path.read_text(encoding="utf-8"),
        replacements=replacements,
    )


def _assert_json_matches_golden(
    *,
    artifact_path: Path,
    golden_relative_path: Path,
    replacements: dict[str, str],
) -> None:
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AssertionError(f"{artifact_path} must contain a JSON object")
    validate_level12_artifact_payload(
        payload,
        artifact_name=artifact_path.name,
        source=artifact_path,
    )
    normalized_payload = _normalize_json(payload, replacements=replacements)
    normalized_text = (
        json.dumps(
            normalized_payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    _assert_text_matches_golden(
        golden_relative_path=golden_relative_path,
        actual_text=normalized_text,
        replacements={},
    )


def _normalize_json(value: object, *, replacements: dict[str, str]) -> object:
    if isinstance(value, dict):
        normalized: dict[str, object] = {}
        for key, raw in value.items():
            key_text = str(key)
            if key_text in VOLATILE_JSON_KEYS:
                normalized[key_text] = SCRUBBED_TIMESTAMP_TOKEN
                continue
            normalized[key_text] = _normalize_json(raw, replacements=replacements)
        return normalized
    if isinstance(value, list):
        return [_normalize_json(item, replacements=replacements) for item in value]
    if isinstance(value, str):
        return _normalize_text(value, replacements=replacements)
    return value


def _assert_text_matches_golden(
    *,
    golden_relative_path: Path,
    actual_text: str,
    replacements: dict[str, str],
) -> None:
    golden_path = GOLDEN_ROOT / golden_relative_path
    normalized_actual = _normalize_text(actual_text, replacements=replacements)
    if not normalized_actual.endswith("\n"):
        normalized_actual = normalized_actual + "\n"

    if _update_goldens_enabled():
        golden_path.parent.mkdir(parents=True, exist_ok=True)
        golden_path.write_text(normalized_actual, encoding="utf-8")
        return

    if not golden_path.exists():
        raise AssertionError(
            f"missing golden file: {golden_path}. "
            f"Set {UPDATE_GOLDENS_ENV}=1 to create/update baselines intentionally."
        )

    expected = golden_path.read_text(encoding="utf-8")
    if normalized_actual == expected:
        return

    diff = "".join(
        difflib.unified_diff(
            expected.splitlines(keepends=True),
            normalized_actual.splitlines(keepends=True),
            fromfile=f"expected/{golden_relative_path}",
            tofile=f"actual/{golden_relative_path}",
        )
    )
    raise AssertionError(
        "artifact drift detected for "
        f"{golden_relative_path}. "
        f"If intentional, re-run with {UPDATE_GOLDENS_ENV}=1.\n{diff}"
    )


def _normalize_text(text: str, *, replacements: dict[str, str]) -> str:
    normalized = text.replace("\r\n", "\n")
    for old, new in sorted(replacements.items(), key=lambda row: len(row[0]), reverse=True):
        normalized = normalized.replace(old, new)
    return normalized


def _write_research_validation_workflow_summary(
    *,
    output_dir: Path,
    case_name: str,
    spec_path: Path,
    metrics_path: Path,
    portfolio_validation_summary_path: Path,
    portfolio_validation_metrics_path: Path,
    portfolio_validation_package_path: Path,
) -> Path:
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AssertionError(f"{metrics_path} must contain a JSON object")
    key_metrics = payload.get("metrics")
    if not isinstance(key_metrics, dict):
        raise AssertionError(f"{metrics_path}.metrics must be a JSON object")

    summary_payload = {
        "workflow": "run-single-factor",
        "experiment_name": case_name,
        "status": "success",
        "config_path": str(spec_path),
        "key_metrics": key_metrics,
        "promotion_decision": {
            "verdict": key_metrics.get("promotion_decision"),
            "reasons": key_metrics.get("promotion_reasons"),
            "blockers": key_metrics.get("promotion_blockers"),
            "source": "level2_promotion_gate",
        },
        "outputs": {
            "portfolio_validation_summary": str(portfolio_validation_summary_path),
            "portfolio_validation_metrics": str(portfolio_validation_metrics_path),
            "portfolio_validation_package": str(portfolio_validation_package_path),
        },
    }
    summary_path = output_dir / f"{case_name}_workflow_summary.json"
    summary_path.write_text(
        json.dumps(summary_payload, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    return summary_path


def _update_goldens_enabled() -> bool:
    raw = os.getenv(UPDATE_GOLDENS_ENV, "")
    return raw.strip().lower() in {"1", "true", "yes", "on"}
