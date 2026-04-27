from __future__ import annotations

import datetime
import math
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import pandas as pd

DiagnosticsSeverity = str
DiagnosticsLevel = str

_DEFAULT_TOP_MISSING_FEATURES = 10
_NEAR_CONSTANT_THRESHOLD = 0.995
_TARGET_OUTLIER_ZSCORE = 3.0


def _utc_now_iso() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat(timespec="milliseconds")


def _to_jsonable(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        return float(value) if math.isfinite(value) else None
    return value


def _coerce_finite_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if not isinstance(value, (int, float, np.number)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _coerce_date_text(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return str(value.date().isoformat())
    text = str(value).strip()
    return text or None


class ModelFactorDiagnosticsObserver(Protocol):
    @contextmanager
    def stage(
        self,
        name: str,
        *,
        payload: dict[str, object] | None = None,
    ) -> Any: ...

    def event(
        self,
        *,
        level: DiagnosticsLevel,
        stage: str,
        message: str,
        payload: dict[str, object] | None = None,
    ) -> None: ...

    def warning(
        self,
        *,
        title: str,
        severity: DiagnosticsSeverity,
        stage: str,
        description: str,
        suggested_action: str,
    ) -> None: ...

    def set_data_health(self, payload: dict[str, object]) -> None: ...


class _NullStageContext:
    def attach(self, **values: object) -> None:  # noqa: ARG002
        return

    def update(self, payload: dict[str, object] | None = None) -> None:  # noqa: ARG002
        return


_NULL_STAGE_CONTEXT = _NullStageContext()


class NullModelFactorDiagnosticsObserver:
    @contextmanager
    def stage(
        self,
        name: str,  # noqa: ARG002
        *,
        payload: dict[str, object] | None = None,  # noqa: ARG002
    ) -> Any:
        yield _NULL_STAGE_CONTEXT

    def event(
        self,
        *,
        level: DiagnosticsLevel,  # noqa: ARG002
        stage: str,  # noqa: ARG002
        message: str,  # noqa: ARG002
        payload: dict[str, object] | None = None,  # noqa: ARG002
    ) -> None:
        return

    def warning(
        self,
        *,
        title: str,  # noqa: ARG002
        severity: DiagnosticsSeverity,  # noqa: ARG002
        stage: str,  # noqa: ARG002
        description: str,  # noqa: ARG002
        suggested_action: str,  # noqa: ARG002
    ) -> None:
        return

    def set_data_health(self, payload: dict[str, object]) -> None:  # noqa: ARG002
        return


_NULL_DIAGNOSTICS_OBSERVER = NullModelFactorDiagnosticsObserver()


@dataclass(frozen=True)
class _StageHandle:
    stage_id: int
    started_perf_counter: float


class _StageContext:
    """Mutable handle returned by ``ModelFactorDiagnosticsRecorder.stage``.

    Callers can call ``attach(key=value, ...)`` inside the ``with`` block to
    record additional fields that end up in the stage's ``result`` payload.
    """

    def __init__(self, handle: _StageHandle) -> None:
        self._handle = handle
        self._result: dict[str, object] = {}

    @property
    def handle(self) -> _StageHandle:
        return self._handle

    def attach(self, **values: object) -> None:
        for key, value in values.items():
            if not isinstance(key, str) or not key.strip():
                continue
            self._result[key] = value

    def update(self, payload: dict[str, object] | None = None) -> None:
        if payload:
            for key, value in payload.items():
                if not isinstance(key, str) or not key.strip():
                    continue
                self._result[key] = value

    def collected_result(self) -> dict[str, object]:
        return dict(self._result)


class ModelFactorDiagnosticsRecorder:
    """Collect run diagnostics with stage timing, events, warnings, and health stats."""

    def __init__(self) -> None:
        self._next_stage_id = 1
        self._stages: list[dict[str, object]] = []
        self._stage_index: dict[int, int] = {}
        self._open_stages: dict[int, _StageHandle] = {}
        self._events: list[dict[str, object]] = []
        self._warnings: list[dict[str, object]] = []
        self._data_health: dict[str, object] = {}

    @contextmanager
    def stage(
        self,
        name: str,
        *,
        payload: dict[str, object] | None = None,
    ) -> Any:
        handle = self.begin_stage(name, payload=payload)
        ctx = _StageContext(handle)
        try:
            yield ctx
        except Exception as exc:
            error_payload: dict[str, object] = dict(ctx.collected_result())
            error_payload["error_type"] = type(exc).__name__
            error_payload["error_message"] = str(exc)
            self.end_stage(handle, status="failed", payload=error_payload)
            raise
        else:
            collected = ctx.collected_result()
            self.end_stage(handle, status="success", payload=collected or None)

    def begin_stage(
        self,
        name: str,
        *,
        payload: dict[str, object] | None = None,
    ) -> _StageHandle:
        stage_name = str(name or "").strip() or "unknown"
        stage_id = self._next_stage_id
        self._next_stage_id += 1
        started_perf_counter = time.perf_counter()
        stage_row: dict[str, object] = {
            "name": stage_name,
            "status": "running",
            "start_time": _utc_now_iso(),
            "end_time": None,
            "duration_ms": None,
        }
        if payload:
            stage_row["payload"] = _to_jsonable(payload)
        self._stage_index[stage_id] = len(self._stages)
        self._stages.append(stage_row)
        self._open_stages[stage_id] = _StageHandle(
            stage_id=stage_id,
            started_perf_counter=started_perf_counter,
        )
        self.event(level="info", stage=stage_name, message="stage started", payload=payload)
        return self._open_stages[stage_id]

    def end_stage(
        self,
        handle: _StageHandle,
        *,
        status: str,
        payload: dict[str, object] | None = None,
    ) -> None:
        stage_idx = self._stage_index.get(handle.stage_id)
        if stage_idx is None:
            return
        row = self._stages[stage_idx]
        if row.get("end_time") is not None:
            return
        duration_ms = int(round((time.perf_counter() - handle.started_perf_counter) * 1000.0))
        row["status"] = str(status or "unknown")
        row["end_time"] = _utc_now_iso()
        row["duration_ms"] = max(duration_ms, 0)
        if payload:
            row["result"] = _to_jsonable(payload)
        stage_name = str(row.get("name") or "unknown")
        level = "error" if status == "failed" else "info"
        self.event(level=level, stage=stage_name, message=f"stage {status}", payload=payload)
        self._open_stages.pop(handle.stage_id, None)

    def event(
        self,
        *,
        level: DiagnosticsLevel,
        stage: str,
        message: str,
        payload: dict[str, object] | None = None,
    ) -> None:
        row: dict[str, object] = {
            "ts": _utc_now_iso(),
            "level": str(level or "info"),
            "stage": str(stage or "").strip() or "unknown",
            "message": str(message or "").strip() or "(empty)",
        }
        if payload:
            row["payload"] = _to_jsonable(payload)
        self._events.append(row)

    def warning(
        self,
        *,
        title: str,
        severity: DiagnosticsSeverity,
        stage: str,
        description: str,
        suggested_action: str,
    ) -> None:
        row: dict[str, object] = {
            "title": str(title or "").strip() or "warning",
            "severity": str(severity or "warning").strip() or "warning",
            "stage": str(stage or "").strip() or "unknown",
            "description": str(description or "").strip() or "(empty)",
            "suggested_action": str(suggested_action or "").strip() or "N/A",
        }
        self._warnings.append(row)

    def set_data_health(self, payload: dict[str, object]) -> None:
        jsonable_payload = _to_jsonable(payload)
        if isinstance(jsonable_payload, dict):
            self._data_health = jsonable_payload
        else:
            self._data_health = {}

    def finalize_open_stages(self, *, status: str = "interrupted") -> None:
        handles = list(self._open_stages.values())
        for handle in handles:
            self.end_stage(handle, status=status)

    def build_payload(
        self,
        *,
        run_meta: dict[str, object] | None = None,
        raw_log_ref: str | None = None,
    ) -> dict[str, object]:
        self.finalize_open_stages()
        payload: dict[str, object] = {
            "schema_version": "1.0.0",
            "artifact_type": "alpha_lab_model_run_diagnostics",
            "generated_at_utc": _utc_now_iso(),
            "run_meta": _to_jsonable(run_meta or {}),
            "stages": _to_jsonable(self._stages),
            "events": _to_jsonable(self._events),
            "warnings": _to_jsonable(self._warnings),
            "data_health": _to_jsonable(self._data_health),
        }
        ref_text = str(raw_log_ref or "").strip()
        if ref_text:
            payload["raw_log_ref"] = ref_text
        return payload


def diagnostics_observer_or_null(
    observer: ModelFactorDiagnosticsObserver | None,
) -> ModelFactorDiagnosticsObserver:
    if observer is None:
        return _NULL_DIAGNOSTICS_OBSERVER
    return observer


def compute_data_health_snapshot(
    *,
    features: pd.DataFrame,
    labels: pd.DataFrame,
    feature_columns: tuple[str, ...],
) -> dict[str, object]:
    n_rows = int(len(features))
    n_assets = int(features["asset"].nunique()) if "asset" in features.columns else 0
    dates = pd.to_datetime(features.get("date"), errors="coerce")
    start_date = _coerce_date_text(dates.min()) if not dates.empty else None
    end_date = _coerce_date_text(dates.max()) if not dates.empty else None

    feature_frame = features.loc[:, list(feature_columns)].copy()
    denom = float(max(n_rows * max(len(feature_columns), 1), 1))
    non_missing = float(feature_frame.notna().sum().sum()) if n_rows > 0 else 0.0
    coverage_ratio = non_missing / denom if denom > 0 else None

    missing_ratios: list[dict[str, object]] = []
    constant_features: list[str] = []
    near_constant_features: list[str] = []
    for column in feature_columns:
        series = feature_frame[column]
        missing_ratio = float(series.isna().mean()) if len(series) > 0 else 1.0
        missing_ratios.append({"feature": column, "missing_ratio": missing_ratio})

        non_na = series.dropna()
        if non_na.empty:
            constant_features.append(column)
            continue
        unique_count = int(non_na.nunique(dropna=True))
        if unique_count <= 1:
            constant_features.append(column)
            continue
        max_freq_ratio = float(non_na.value_counts(normalize=True, dropna=True).max())
        if max_freq_ratio >= _NEAR_CONSTANT_THRESHOLD:
            near_constant_features.append(column)

    missing_ratios = sorted(
        missing_ratios,
        key=lambda item: float(_coerce_finite_float(item.get("missing_ratio")) or 0.0),
        reverse=True,
    )[:_DEFAULT_TOP_MISSING_FEATURES]

    target_series = pd.to_numeric(labels.get("label"), errors="coerce").dropna()
    target_mean = _coerce_finite_float(target_series.mean()) if not target_series.empty else None
    target_std = _coerce_finite_float(target_series.std(ddof=0)) if len(target_series) > 1 else 0.0
    outlier_ratio: float | None
    if target_series.empty:
        outlier_ratio = None
    else:
        std = float(target_series.std(ddof=0))
        if std <= 0.0 or not math.isfinite(std):
            outlier_ratio = 0.0
        else:
            zscore = ((target_series - float(target_series.mean())) / std).abs()
            outlier_ratio = float((zscore > _TARGET_OUTLIER_ZSCORE).mean())

    distribution_shift = _compute_train_valid_distribution_shift(
        feature_frame=feature_frame,
        dates=dates,
        feature_columns=feature_columns,
    )

    return {
        "n_rows": n_rows,
        "n_assets": n_assets,
        "date_range": {
            "start": start_date,
            "end": end_date,
        },
        "coverage_ratio": _coerce_finite_float(coverage_ratio),
        "missing_ratio_top_features": missing_ratios,
        "constant_features": sorted(set(constant_features)),
        "near_constant_features": sorted(set(near_constant_features)),
        "target_mean": target_mean,
        "target_std": target_std,
        "outlier_ratio": _coerce_finite_float(outlier_ratio),
        "train_valid_distribution_shift": distribution_shift,
    }


def _compute_train_valid_distribution_shift(
    *,
    feature_frame: pd.DataFrame,
    dates: pd.Series,
    feature_columns: tuple[str, ...],
) -> dict[str, object]:
    if dates is None or dates.empty or feature_frame.empty:
        return {"split_strategy": "first_half_vs_second_half", "n_features_compared": 0}

    sorted_dates = dates.dropna().sort_values()
    if sorted_dates.empty:
        return {"split_strategy": "first_half_vs_second_half", "n_features_compared": 0}

    cut = sorted_dates.iloc[len(sorted_dates) // 2]
    train_mask = dates <= cut
    valid_mask = dates > cut
    if not bool(valid_mask.any()):
        # Fall back to last date out so split is non-degenerate when possible.
        unique_sorted = sorted_dates.drop_duplicates()
        if len(unique_sorted) < 2:
            return {
                "split_strategy": "first_half_vs_second_half",
                "n_features_compared": 0,
            }
        cut = unique_sorted.iloc[-2]
        train_mask = dates <= cut
        valid_mask = dates > cut

    train_frame = feature_frame.loc[train_mask, list(feature_columns)]
    valid_frame = feature_frame.loc[valid_mask, list(feature_columns)]
    n_train = int(len(train_frame))
    n_valid = int(len(valid_frame))

    abs_mean_drifts: list[float] = []
    abs_std_drifts: list[float] = []
    per_feature_top: list[dict[str, object]] = []
    n_compared = 0
    for column in feature_columns:
        train_series = pd.to_numeric(train_frame[column], errors="coerce").dropna()
        valid_series = pd.to_numeric(valid_frame[column], errors="coerce").dropna()
        if train_series.empty or valid_series.empty:
            continue
        train_mean = float(train_series.mean())
        valid_mean = float(valid_series.mean())
        train_std = float(train_series.std(ddof=0))
        valid_std = float(valid_series.std(ddof=0))
        scale = max(abs(train_std), 1e-12)
        mean_drift = (valid_mean - train_mean) / scale
        std_ratio = (valid_std / scale) - 1.0 if scale > 0 else 0.0
        if not math.isfinite(mean_drift):
            mean_drift = 0.0
        if not math.isfinite(std_ratio):
            std_ratio = 0.0
        abs_mean_drifts.append(abs(mean_drift))
        abs_std_drifts.append(abs(std_ratio))
        per_feature_top.append(
            {
                "feature": column,
                "mean_drift_z": _coerce_finite_float(mean_drift),
                "std_ratio_minus_1": _coerce_finite_float(std_ratio),
            }
        )
        n_compared += 1

    per_feature_top.sort(
        key=lambda item: max(
            abs(_coerce_finite_float(item.get("mean_drift_z") ) or 0.0),
            abs(_coerce_finite_float(item.get("std_ratio_minus_1")) or 0.0),
        ),
        reverse=True,
    )
    return {
        "split_strategy": "first_half_vs_second_half",
        "split_cut_date": _coerce_date_text(cut),
        "n_train_rows": n_train,
        "n_valid_rows": n_valid,
        "n_features_compared": n_compared,
        "mean_abs_mean_drift_z": _coerce_finite_float(
            float(np.mean(abs_mean_drifts)) if abs_mean_drifts else None
        ),
        "max_abs_mean_drift_z": _coerce_finite_float(
            float(max(abs_mean_drifts)) if abs_mean_drifts else None
        ),
        "mean_abs_std_ratio_minus_1": _coerce_finite_float(
            float(np.mean(abs_std_drifts)) if abs_std_drifts else None
        ),
        "top_drift_features": per_feature_top[:_DEFAULT_TOP_MISSING_FEATURES],
    }


def derive_data_health_warnings(data_health: dict[str, object]) -> list[dict[str, str]]:
    warnings: list[dict[str, str]] = []

    coverage_ratio = _coerce_finite_float(data_health.get("coverage_ratio"))
    if coverage_ratio is not None and coverage_ratio < 0.70:
        warnings.append(
            {
                "title": "特征覆盖率偏低",
                "severity": "warning",
                "stage": "feature_validate",
                "description": (
                    f"当前特征有效覆盖率约为 {coverage_ratio:.3f}，可能导致可训练样本不足。"
                ),
                "suggested_action": "检查特征缺失来源，优先修复高缺失列或缩小特征白名单。",
            }
        )

    constant_features = data_health.get("constant_features")
    if isinstance(constant_features, list) and constant_features:
        preview = ", ".join(str(item) for item in constant_features[:5])
        warnings.append(
            {
                "title": "检测到常量特征",
                "severity": "warning",
                "stage": "feature_validate",
                "description": f"存在常量特征（示例：{preview}）。",
                "suggested_action": "从 feature_columns 中移除常量列，或修复上游特征生成逻辑。",
            }
        )

    near_constant_features = data_health.get("near_constant_features")
    if isinstance(near_constant_features, list) and near_constant_features:
        preview = ", ".join(str(item) for item in near_constant_features[:5])
        warnings.append(
            {
                "title": "检测到近常量特征",
                "severity": "info",
                "stage": "feature_validate",
                "description": f"存在近常量特征（示例：{preview}），信息量可能较低。",
                "suggested_action": "审查近常量列的分布并考虑降权或剔除。",
            }
        )

    target_std = _coerce_finite_float(data_health.get("target_std"))
    if target_std is not None and target_std <= 0.0:
        warnings.append(
            {
                "title": "目标方差接近 0",
                "severity": "warning",
                "stage": "target_build",
                "description": "目标序列方差接近 0，模型难以学习有效信号。",
                "suggested_action": "检查 target horizon、价格口径和样本区间是否合理。",
            }
        )

    outlier_ratio = _coerce_finite_float(data_health.get("outlier_ratio"))
    if outlier_ratio is not None and outlier_ratio >= 0.05:
        warnings.append(
            {
                "title": "目标异常值占比偏高",
                "severity": "warning",
                "stage": "target_build",
                "description": f"目标异常值占比约 {outlier_ratio:.3f}，可能引入训练不稳定。",
                "suggested_action": "考虑 winsorize/robust loss，或排查除权与脏数据影响。",
            }
        )

    return warnings
