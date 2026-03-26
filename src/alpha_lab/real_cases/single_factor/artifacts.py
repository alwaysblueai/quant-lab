from __future__ import annotations

import datetime
import json
import logging
import math
import shutil
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from alpha_lab.vault_export import export_to_vault, resolve_vault_root

from .evaluate import SingleFactorEvaluationResult
from .spec import SingleFactorCaseSpec, dump_spec_yaml, spec_to_dict
from .templates import render_experiment_card_markdown, render_summary_markdown

logger = logging.getLogger(__name__)

REQUIRED_BUNDLE_FILES: tuple[str, ...] = (
    "run_manifest.json",
    "metrics.json",
    "ic_timeseries.csv",
    "group_returns.csv",
    "turnover.csv",
    "coverage.csv",
    "factor_definition.yaml",
    "summary.md",
    "experiment_card.md",
)


def export_artifact_bundle(
    *,
    spec: SingleFactorCaseSpec,
    evaluation_result: SingleFactorEvaluationResult,
    output_dir: str | Path,
    spec_path: str | Path | None = None,
    vault_root: str | Path | None = None,
    vault_export_mode: str = "versioned",
) -> dict[str, Path]:
    """Write standardized artifact bundle for one single-factor case run."""

    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "run_manifest": out_dir / "run_manifest.json",
        "metrics": out_dir / "metrics.json",
        "ic_timeseries": out_dir / "ic_timeseries.csv",
        "group_returns": out_dir / "group_returns.csv",
        "turnover": out_dir / "turnover.csv",
        "coverage": out_dir / "coverage.csv",
        "factor_definition": out_dir / "factor_definition.yaml",
        "summary": out_dir / "summary.md",
        "experiment_card": out_dir / "experiment_card.md",
    }

    evaluation_result.ic_timeseries.to_csv(paths["ic_timeseries"], index=False)
    evaluation_result.group_returns.to_csv(paths["group_returns"], index=False)
    evaluation_result.turnover.to_csv(paths["turnover"], index=False)
    evaluation_result.coverage.to_csv(paths["coverage"], index=False)

    spec_yaml = dump_spec_yaml(spec)
    paths["factor_definition"].write_text(spec_yaml, encoding="utf-8")

    summary_md = render_summary_markdown(
        spec=spec,
        metrics=evaluation_result.metrics,
        output_dir=out_dir,
    )
    paths["summary"].write_text(summary_md, encoding="utf-8")

    card_md = render_experiment_card_markdown(
        spec=spec,
        metrics=evaluation_result.metrics,
        result=evaluation_result.experiment_result,
    )
    paths["experiment_card"].write_text(card_md, encoding="utf-8")

    metrics_payload = {
        "metrics": _to_jsonable(evaluation_result.metrics),
        "coverage_by_date_summary": {
            "n_dates": int(evaluation_result.coverage["date"].nunique())
            if not evaluation_result.coverage.empty
            else 0,
            "mean_coverage": _finite_or_none(
                evaluation_result.coverage["coverage"].mean()
                if not evaluation_result.coverage.empty
                else float("nan")
            ),
            "min_coverage": _finite_or_none(
                evaluation_result.coverage["coverage"].min()
                if not evaluation_result.coverage.empty
                else float("nan")
            ),
        },
        "neutralization_summary": _to_jsonable(
            evaluation_result.neutralization_summary.to_dict(orient="records")
        ),
    }
    _write_json(paths["metrics"], metrics_payload)

    manifest = {
        "schema_version": "1.0.0",
        "artifact_type": "real_case_single_factor_bundle",
        "run_timestamp_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "case_name": spec.name,
        "spec_path": str(Path(spec_path).resolve()) if spec_path is not None else None,
        "inputs": {
            "prices_path": spec.prices_path,
            "factor_path": spec.factor_path,
            "factor_name": spec.factor_name,
            "universe_path": spec.universe.path,
            "neutralization_exposures_path": spec.neutralization.exposures_path,
        },
        "spec": _to_jsonable(spec_to_dict(spec)),
        "outputs": {name: str(path) for name, path in paths.items()},
        "required_bundle_files": list(REQUIRED_BUNDLE_FILES),
        "vault_export": {
            "enabled": False,
            "mode": "skip",
            "target_paths": [],
            "status": "skipped",
            "error": None,
        },
    }

    _write_json(paths["run_manifest"], manifest)

    resolved_vault = resolve_vault_root(vault_root)
    enabled = resolved_vault is not None and vault_export_mode.strip().lower() != "skip"
    vault_result = export_to_vault(
        {
            "experiment_card_path": paths["experiment_card"],
            "summary_path": paths["summary"],
            "manifest_path": paths["run_manifest"],
        },
        case_name=spec.name,
        vault_root=vault_root,
        mode=vault_export_mode,
    )
    manifest["vault_export"] = vault_result.to_manifest_dict(enabled=enabled)
    _write_json(paths["run_manifest"], manifest)

    if vault_result.status == "failed":
        logger.warning(
            "Vault export failed for single-factor case %s: %s",
            spec.name,
            vault_result.error,
        )
    if vault_result.success and vault_result.target_paths:
        _sync_exported_manifest_copies(paths["run_manifest"], vault_result.target_paths)

    return paths


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_to_jsonable(payload), f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def _to_jsonable(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float):
        return _finite_or_none(value)
    return value


def _finite_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None


def _sync_exported_manifest_copies(
    local_manifest_path: Path,
    target_paths: tuple[str, ...],
) -> None:
    """Ensure vault-side manifest copies contain final vault_export payload."""

    for raw in target_paths:
        target = Path(raw)
        if not target.name.endswith("run_manifest.json"):
            continue
        try:
            shutil.copy2(local_manifest_path, target)
        except OSError as exc:
            logger.warning(
                "Failed to sync single-factor vault manifest copy %s: %s",
                target,
                exc,
            )
