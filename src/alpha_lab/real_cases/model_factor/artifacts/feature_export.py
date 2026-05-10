from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from alpha_lab.real_cases.common_spec import parse_mapping_payload

from ..spec import ModelFactorCaseSpec, spec_to_dict

# Cross-module imports (auto-added)
from ._utils import _as_object


def _feature_preprocess_payload_for_artifacts(
    *,
    spec: ModelFactorCaseSpec,
    spec_path: str | Path | None,
) -> dict[str, object]:
    payload = dict(_as_object(spec_to_dict(spec).get("feature_preprocess")))
    payload["cross_sectional_transform_default_applied"] = (
        _cross_sectional_transform_default_applied(spec_path)
    )
    return payload


def _cross_sectional_transform_default_applied(spec_path: str | Path | None) -> bool:
    if spec_path is None:
        return False
    path = Path(spec_path)
    if not path.exists() or not path.is_file():
        return False
    try:
        parsed = parse_mapping_payload(path.read_text(encoding="utf-8"), suffix=path.suffix.lower())
    except Exception:  # noqa: BLE001
        return False
    raw_feature_preprocess = parsed.get("feature_preprocess")
    if not isinstance(raw_feature_preprocess, Mapping):
        return True
    return "cross_sectional_transform" not in raw_feature_preprocess


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, na_rep="N/A")
    if path.exists() and path.stat().st_size > 0:
        return
    # Defensive fallback: keep file-level content non-empty even for pathological empty inputs.
    path.write_text("status,reason\nnot_available,empty_dataframe\n", encoding="utf-8")


def _prepare_training_log_for_export(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    exported = frame.copy()
    notes: list[dict[str, object]] = []
    fill_specs: tuple[tuple[str, str], ...] = (
        ("skip_reason", "not_skipped"),
        ("model_version", "N/A"),
        ("trained_date_start", "N/A"),
        ("trained_date_end", "N/A"),
        ("scale_mode", "N/A"),
    )
    missing_total = 0
    impacted_fields: list[str] = []
    for field, replacement in fill_specs:
        if field not in exported.columns:
            continue
        values = exported[field]
        missing_mask = values.isna() | values.astype(str).str.strip().eq("")
        missing_count = int(missing_mask.sum())
        if missing_count <= 0:
            continue
        missing_total += missing_count
        impacted_fields.append(field)
        exported[field] = values.astype(object)
        exported.loc[missing_mask, field] = replacement
    if missing_total > 0:
        notes.append(
            {
                "artifact": "training_log.csv",
                "fields": impacted_fields,
                "missing_value_count": missing_total,
                "reason": (
                    "训练日志存在按行条件不适用字段（例如非 skipped 行的 skip_reason、"
                    "skipped 行的模型版本/训练窗口）；导出时统一写为 N/A 或 not_skipped。"
                ),
            }
        )
    return exported, notes


def _prepare_feature_importance_for_export(
    frame: pd.DataFrame,
    *,
    model_family: str,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    exported = frame.copy()
    notes: list[dict[str, object]] = []
    required_defaults: dict[str, object] = {
        "feature": "N/A",
        "mean_abs_importance": "N/A",
        "latest_importance": "N/A",
        "mean_signed_importance": "N/A",
        "latest_abs_importance": "N/A",
        "positive_version_count": 0,
        "negative_version_count": 0,
        "zero_version_count": 0,
        "sign_stability": "N/A",
        "importance_source": "unknown",
        "n_model_versions": 0,
    }
    for field, default in required_defaults.items():
        if field not in exported.columns:
            exported[field] = default

    if exported.empty:
        exported = pd.DataFrame(
            [
                {
                    "feature": "N/A",
                    "mean_abs_importance": "N/A",
                    "latest_importance": "N/A",
                    "importance_source": "not_available",
                    "n_model_versions": 0,
                    "missing_value_reason": (
                        "feature_importance 数据为空：训练阶段未产出可用重要性统计。"
                    ),
                }
            ]
        )
        notes.append(
            {
                "artifact": "feature_importance.csv",
                "fields": ["mean_abs_importance", "latest_importance"],
                "missing_value_count": 2,
                "reason": "训练阶段未产出可用重要性统计，已写入 N/A 与缺失原因。",
            }
        )
        return exported, notes

    missing_total = 0
    for field in (
        "mean_abs_importance",
        "latest_importance",
        "mean_signed_importance",
        "latest_abs_importance",
        "sign_stability",
    ):
        values = exported[field]
        missing_mask = values.isna() | values.astype(str).str.strip().eq("")
        missing_count = int(missing_mask.sum())
        if missing_count <= 0:
            continue
        missing_total += missing_count
        exported[field] = values.astype(object)
        exported.loc[missing_mask, field] = "N/A"

    source_values = exported["importance_source"].astype(str).str.strip()
    exported["importance_source"] = source_values.where(source_values.ne(""), "unknown")
    exported["missing_value_reason"] = exported.apply(
        lambda row: _feature_importance_missing_reason(row, model_family=model_family),
        axis=1,
    )

    if missing_total > 0:
        notes.append(
            {
                "artifact": "feature_importance.csv",
                "fields": ["mean_abs_importance", "latest_importance"],
                "missing_value_count": missing_total,
                "reason": (
                    "部分特征重要性字段缺失，已写入 N/A；"
                    "逐行原因见 missing_value_reason（例如模型族不支持 importance 提取）。"
                ),
            }
        )
    return exported, notes


def _feature_importance_missing_reason(row: pd.Series, *, model_family: str) -> str:
    mean_value = str(row.get("mean_abs_importance", "")).strip()
    latest_value = str(row.get("latest_importance", "")).strip()
    latest_abs_value = str(row.get("latest_abs_importance", "")).strip()
    has_missing = mean_value in {"", "N/A"} or latest_value in {"", "N/A"}
    if not has_missing:
        return "无缺失"
    source = str(row.get("importance_source", "")).strip().lower()
    raw_versions = row.get("n_model_versions")
    versions = 0
    if isinstance(raw_versions, (int, float)) and math.isfinite(float(raw_versions)):
        versions = int(raw_versions)
    elif isinstance(raw_versions, str):
        text = raw_versions.strip()
        if text.isdigit():
            versions = int(text)
    if source == "unsupported":
        return (
            f"模型族 `{model_family}` 当前不暴露可解释特征重要性接口，"
            "因此 mean/latest importance 记为 N/A。"
        )
    if source in {"built_in", "feature_importances"} and latest_abs_value not in {"", "N/A"}:
        return (
            "树模型内置 importance 仅提供非负重要性；"
            "latest_importance 的方向符号不可用，请使用 latest_abs_importance。"
        )
    if source == "unsupported_mlp_default":
        return (
            "MLP 默认不生成版本级 importance；permutation importance 成本高，"
            "需要手动开启 latest_only + sampled 诊断。"
        )
    if source == "built_in_unavailable":
        return (
            f"模型族 `{model_family}` 当前估计器未暴露 feature_importances_；"
            "已跳过默认 permutation fallback。"
        )
    if source == "permutation_skipped_guardrail":
        reason = str(row.get("permutation_guardrail_reason", "")).strip()
        return reason or "permutation importance 被计算 guardrail 跳过。"
    if source == "permutation_sampled" and latest_abs_value not in {"", "N/A"}:
        return "permutation importance 为手动 sampled 诊断，仅提供非负重要性，不表示特征方向。"
    if source == "disabled":
        return "feature_importance.mode='disabled'，本次运行按配置跳过重要性计算。"
    if versions <= 0:
        return "训练阶段未生成可用模型版本，无法汇总 mean/latest importance。"
    return "重要性统计不可用（存在非有限值或中间结果缺失）。"


def _prepare_feature_importance_ledger_for_export(
    frame: pd.DataFrame,
    *,
    spec: ModelFactorCaseSpec,
) -> pd.DataFrame:
    columns = [
        "run_id",
        "case",
        "factor",
        "model_family",
        "model_version",
        "fit_date",
        "feature",
        "signed_importance",
        "abs_importance",
        "normalized_share",
        "rank",
        "importance_source",
        "permutation_sampled",
        "permutation_sample_rows",
        "permutation_n_repeats",
        "permutation_guardrail_reason",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    exported = frame.copy()
    exported.insert(0, "factor", spec.factor_name)
    exported.insert(0, "case", spec.name)
    exported.insert(0, "run_id", spec.name)
    for column in columns:
        if column not in exported.columns:
            exported[column] = pd.NA
    return exported.loc[:, columns]
