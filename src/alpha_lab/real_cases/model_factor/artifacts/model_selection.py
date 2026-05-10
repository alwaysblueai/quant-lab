from __future__ import annotations

import math
from collections.abc import Mapping

import pandas as pd

from alpha_lab.model_factor import ModelFactorBuildResult, ModelSpec, resolve_model_spec_params

from ..spec import ModelFactorCaseSpec, spec_to_dict

# Cross-module imports (auto-added)
from ._utils import _as_object, _finite_if_number, _normalized_text_or_none, _to_jsonable


def _build_model_selection_payload(
    *,
    spec: ModelFactorCaseSpec,
    model_factor_result: ModelFactorBuildResult,
) -> dict[str, object]:
    model_selection_df = model_factor_result.model_selection_df.copy()
    diagnostics_selection = model_factor_result.model_diagnostics.get("model_selection", {})
    if isinstance(diagnostics_selection, Mapping):
        diagnostics_selection_payload: Mapping[str, object] = diagnostics_selection
    else:
        diagnostics_selection_payload = {}
    if model_selection_df.empty:
        status = "disabled" if not spec.model_selection.enabled else "not_available"
        rows: list[dict[str, object]] = []
    else:
        status = "ok"
        rows = _to_jsonable(model_selection_df.to_dict(orient="records"))  # type: ignore[assignment]
    return {
        "schema_version": "1.0.0",
        "artifact_type": "alpha_lab_model_selection",
        "case_name": spec.name,
        "factor_name": spec.factor_name,
        "status": status,
        "configured_model": _to_jsonable(spec_to_dict(spec).get("model", {})),
        "configured_model_selection": _to_jsonable(spec_to_dict(spec).get("model_selection", {})),
        "summary": _to_jsonable(dict(diagnostics_selection_payload)),
        "selection_rows": rows,
    }


def _candidate_id_for_payload(idx: int, candidate: ModelSpec) -> str:
    return f"candidate_{idx + 1}_{candidate.family}"


def _selection_candidates_for_payload(spec: ModelFactorCaseSpec) -> tuple[ModelSpec, ...]:
    if spec.model_selection.enabled and spec.model_selection.candidates:
        return spec.model_selection.candidates
    return (spec.model,)


def _build_resolved_model_params_payload(*, spec: ModelFactorCaseSpec) -> dict[str, object]:
    candidates = _selection_candidates_for_payload(spec)
    candidate_rows: list[dict[str, object]] = []
    for idx, candidate in enumerate(candidates):
        candidate_rows.append(
            {
                "candidate_id": _candidate_id_for_payload(idx, candidate),
                "family": candidate.family,
                "params": _to_jsonable(resolve_model_spec_params(candidate)),
            }
        )
    return {
        "configured_model": {
            "family": spec.model.family,
            "params": _to_jsonable(resolve_model_spec_params(spec.model)),
        },
        "selection_candidates": candidate_rows,
        "selection_candidates_count": int(len(candidate_rows)),
    }


def _build_label_temporal_contract_payload(
    *,
    spec: ModelFactorCaseSpec,
    model_factor_result: ModelFactorBuildResult,
) -> dict[str, object]:
    diagnostics = _as_object(model_factor_result.model_diagnostics)
    raw_gap = diagnostics.get("purged_train_gap_dates")
    if isinstance(raw_gap, (int, float)) and math.isfinite(float(raw_gap)):
        purge_gap = int(raw_gap)
    else:
        purge_gap = max(int(spec.target.horizon) - 1, 0)
    raw_clipped_rows = diagnostics.get("label_winsor_clipped_rows")
    if isinstance(raw_clipped_rows, (int, float)) and math.isfinite(float(raw_clipped_rows)):
        clipped_rows = int(raw_clipped_rows)
    else:
        clipped_rows = 0
    raw_extreme_filtered_rows = model_factor_result.target_diagnostics.get(
        "label_extreme_filtered_rows"
    )
    if isinstance(raw_extreme_filtered_rows, (int, float)) and math.isfinite(
        float(raw_extreme_filtered_rows)
    ):
        extreme_filtered_rows = int(raw_extreme_filtered_rows)
    else:
        extreme_filtered_rows = 0
    return {
        "target_kind": spec.target.kind,
        "target_horizon": int(spec.target.horizon),
        "target_price_column": spec.target.price_column,
        "max_abs_forward_return": _finite_if_number(spec.target.max_abs_forward_return),
        "purged_train_gap_dates": purge_gap,
        "walk_forward_purged_training": True,
        "label_winsorize_zscore": _finite_if_number(diagnostics.get("label_winsorize_zscore")),
        "label_winsor_clipped_rows": clipped_rows,
        "label_extreme_filtered_rows": extreme_filtered_rows,
        "label_extreme_max_abs_raw_return": _finite_if_number(
            model_factor_result.target_diagnostics.get("label_extreme_max_abs_raw_return")
        ),
    }


def _build_model_selection_outcome_payload(
    *,
    spec: ModelFactorCaseSpec,
    model_factor_result: ModelFactorBuildResult,
) -> dict[str, object]:
    training_log = model_factor_result.training_log_df
    selection_rows = model_factor_result.model_selection_df
    diagnostics_selection = _as_object(
        _as_object(model_factor_result.model_diagnostics).get("model_selection")
    )
    raw_selection_events = diagnostics_selection.get("n_selection_events")
    if isinstance(raw_selection_events, (int, float)) and math.isfinite(
        float(raw_selection_events)
    ):
        n_selection_events = int(raw_selection_events)
    else:
        n_selection_events = 0

    status_series = (
        training_log["status"].astype(str).str.strip().str.lower()
        if "status" in training_log.columns
        else pd.Series(dtype=str)
    )
    n_fit_events = int((status_series == "fit_scored").sum()) if not status_series.empty else 0
    n_reuse_events = int((status_series == "reused_scored").sum()) if not status_series.empty else 0
    n_skip_events = int((status_series == "skipped").sum()) if not status_series.empty else 0

    selected_candidates = selection_rows
    if "selected" in selected_candidates.columns:
        selected_candidates = selected_candidates[selected_candidates["selected"] == True].copy()  # noqa: E712
    if "score_date" in selected_candidates.columns:
        selected_candidates = selected_candidates.sort_values("score_date", kind="mergesort")

    latest_candidate_id: str | None = None
    latest_candidate_score: float | None = None
    latest_selection_score_date: str | None = None
    if not selected_candidates.empty:
        latest_row = selected_candidates.iloc[-1]
        latest_candidate_id = _normalized_text_or_none(latest_row.get("candidate_id"))
        latest_candidate_score = _finite_if_number(latest_row.get("selection_score"))
        latest_selection_score_date = _normalized_text_or_none(latest_row.get("score_date"))

    if latest_candidate_id is None and "selected_candidate_id" in training_log.columns:
        candidate_series = training_log["selected_candidate_id"].map(_normalized_text_or_none)
        candidate_series = candidate_series[candidate_series.notna()]
        if not candidate_series.empty:
            latest_candidate_id = str(candidate_series.iloc[-1])

    if latest_candidate_score is None and "selected_candidate_score" in training_log.columns:
        score_series = training_log["selected_candidate_score"].map(_finite_if_number)
        score_series = score_series[score_series.notna()]
        if not score_series.empty:
            latest_candidate_score = float(score_series.iloc[-1])

    latest_candidate_turnover: float | None = None
    if "selected_candidate_turnover" in training_log.columns:
        turnover_frame = training_log
        if latest_candidate_id is not None and "selected_candidate_id" in turnover_frame.columns:
            ids = turnover_frame["selected_candidate_id"].map(_normalized_text_or_none)
            turnover_frame = turnover_frame[ids == latest_candidate_id]
        turnover_series = turnover_frame["selected_candidate_turnover"].map(_finite_if_number)
        turnover_series = turnover_series[turnover_series.notna()]
        if not turnover_series.empty:
            latest_candidate_turnover = float(turnover_series.iloc[-1])

    candidate_lookup: dict[str, ModelSpec] = {
        _candidate_id_for_payload(idx, candidate): candidate
        for idx, candidate in enumerate(_selection_candidates_for_payload(spec))
    }
    latest_candidate = (
        candidate_lookup.get(latest_candidate_id) if latest_candidate_id is not None else None
    )

    return {
        "enabled": bool(spec.model_selection.enabled),
        "selection_metric": spec.model_selection.metric if spec.model_selection.enabled else None,
        "n_selection_events": n_selection_events,
        "n_fit_events": n_fit_events,
        "n_reuse_events": n_reuse_events,
        "n_skip_events": n_skip_events,
        "latest_selected_candidate_id": latest_candidate_id,
        "latest_selected_candidate_family": (
            latest_candidate.family if latest_candidate is not None else None
        ),
        "latest_selected_candidate_params": (
            _to_jsonable(resolve_model_spec_params(latest_candidate))
            if latest_candidate is not None
            else None
        ),
        "latest_selected_candidate_score": latest_candidate_score,
        "latest_selected_candidate_turnover": latest_candidate_turnover,
        "latest_selection_score_date": latest_selection_score_date,
    }
