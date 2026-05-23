from __future__ import annotations

import json
import logging
import math
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import TypedDict

from alpha_lab.artifact_contracts import validate_level12_artifact_payload
from alpha_lab.real_cases._artifact_json import to_jsonable as _to_jsonable

# Re-export so sibling modules can keep ``from ._utils import _to_jsonable``.
__all__ = ["_to_jsonable"]


class ModelFactorArtifactPaths(TypedDict):
    run_manifest: Path
    metrics: Path
    factor_definition_json: Path
    signal_validation_json: Path
    portfolio_recipe_json: Path
    backtest_result_json: Path
    purged_kfold_summary: Path
    purged_kfold_folds: Path
    purged_kfold_fold_daily: Path
    model_selection_json: Path
    model_definition_json: Path
    feature_manifest_json: Path
    diagnostics: Path
    research_tearsheet: Path
    research_tearsheet_pdf: Path
    training_log: Path
    training_metrics: Path
    feature_importance: Path
    feature_importance_ledger: Path
    feature_oos_ic: Path
    ic_timeseries: Path
    ic_decay: Path
    rolling_stability: Path
    group_returns: Path
    group_nav: Path
    turnover: Path
    coverage: Path
    factor_definition: Path
    summary: Path
    experiment_card: Path
    integrity_report_json: Path
    integrity_report_markdown: Path
    portfolio_validation_summary: Path
    portfolio_validation_metrics: Path
    portfolio_validation_package: Path
    portfolio_validation_markdown: Path


logger = logging.getLogger(__name__)


def _write_json(path: Path, payload: Mapping[str, object], *, pretty: bool = True) -> None:
    jsonable_payload = _to_jsonable(payload)
    if not isinstance(jsonable_payload, Mapping):
        raise ValueError(f"{path} JSON payload root must be an object")
    validate_level12_artifact_payload(
        jsonable_payload,
        artifact_name=path.name,
        source=path,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        if pretty:
            json.dump(jsonable_payload, file_obj, ensure_ascii=False, indent=2, sort_keys=True)
        else:
            json.dump(
                jsonable_payload,
                file_obj,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=False,
            )
        file_obj.write("\n")


def _finite_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None


def _finite_if_number(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if not isinstance(value, (int, float)):
        return None
    return _finite_or_none(float(value))


def _text_or_none(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _normalized_text_or_none(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan" or text in {"N/A", "None", "null"}:
        return None
    return text


def _as_object(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _split_contract_payload(metrics: Mapping[str, object]) -> dict[str, object]:
    raw = metrics.get("split_contract")
    return {str(key): value for key, value in raw.items()} if isinstance(raw, Mapping) else {}


def _split_contract_top_level_fields(split_contract: Mapping[str, object]) -> dict[str, object]:
    fields: dict[str, object] = {"split_contract": dict(split_contract)}
    for key in ("is_start", "is_end", "oos_start", "oos_end"):
        if key in split_contract:
            fields[key] = split_contract[key]
    return fields


def _sync_exported_manifest_copies(
    local_manifest_path: Path,
    target_paths: tuple[str, ...],
) -> None:
    for raw in target_paths:
        target = Path(raw)
        if not target.name.endswith("run_manifest.json"):
            continue
        try:
            shutil.copy2(local_manifest_path, target)
        except OSError as exc:
            logger.warning(
                "Failed to sync model-factor vault manifest copy %s: %s",
                target,
                exc,
            )
