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

from .combine import CombineResult
from .evaluate import CompositeEvaluationResult
from .spec import CompositeCaseSpec, dump_spec_yaml, spec_to_dict
from .templates import render_experiment_card_markdown, render_summary_markdown

logger = logging.getLogger(__name__)

REQUIRED_BUNDLE_FILES: tuple[str, ...] = (
    "run_manifest.json",
    "metrics.json",
    "ic_timeseries.csv",
    "group_returns.csv",
    "turnover.csv",
    "exposures.csv",
    "composite_definition.yaml",
    "summary.md",
    "experiment_card.md",
)


def export_artifact_bundle(
    *,
    spec: CompositeCaseSpec,
    combine_result: CombineResult,
    evaluation_result: CompositeEvaluationResult,
    output_dir: str | Path,
    spec_path: str | Path | None = None,
    vault_root: str | Path | None = None,
    vault_export_mode: str = "versioned",
) -> dict[str, Path]:
    """Write standardized artifact bundle for one composite case run."""

    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "run_manifest": out_dir / "run_manifest.json",
        "metrics": out_dir / "metrics.json",
        "ic_timeseries": out_dir / "ic_timeseries.csv",
        "group_returns": out_dir / "group_returns.csv",
        "turnover": out_dir / "turnover.csv",
        "exposures": out_dir / "exposures.csv",
        "composite_definition": out_dir / "composite_definition.yaml",
        "summary": out_dir / "summary.md",
        "experiment_card": out_dir / "experiment_card.md",
    }

    evaluation_result.ic_timeseries.to_csv(paths["ic_timeseries"], index=False)
    evaluation_result.group_returns.to_csv(paths["group_returns"], index=False)
    evaluation_result.turnover.to_csv(paths["turnover"], index=False)
    evaluation_result.exposure_summary.to_csv(paths["exposures"], index=False)

    spec_yaml = dump_spec_yaml(spec)
    paths["composite_definition"].write_text(spec_yaml, encoding="utf-8")

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
        "component_summary": _to_jsonable(
            combine_result.component_summary.to_dict(orient="records")
        ),
        "coverage_by_date_summary": {
            "n_dates": int(combine_result.coverage_by_date["date"].nunique())
            if not combine_result.coverage_by_date.empty
            else 0,
            "mean_composite_coverage": _finite_or_none(
                combine_result.coverage_by_date["composite_coverage"].mean()
                if not combine_result.coverage_by_date.empty
                else float("nan")
            ),
            "min_composite_coverage": _finite_or_none(
                combine_result.coverage_by_date["composite_coverage"].min()
                if not combine_result.coverage_by_date.empty
                else float("nan")
            ),
        },
    }
    _write_json(paths["metrics"], metrics_payload)

    manifest = {
        "schema_version": "1.0.0",
        "artifact_type": "real_case_composite_bundle",
        "run_timestamp_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "case_name": spec.name,
        "spec_path": str(Path(spec_path).resolve()) if spec_path is not None else None,
        "inputs": {
            "prices_path": spec.prices_path,
            "component_paths": [component.path for component in spec.components],
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

    # Write once so manifest_path exists for optional vault sync.
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
            "Vault export failed for case %s: %s",
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
                "Failed to sync vault manifest copy %s: %s",
                target,
                exc,
            )
