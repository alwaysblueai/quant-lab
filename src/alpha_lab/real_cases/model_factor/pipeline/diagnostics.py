from __future__ import annotations

from pathlib import Path

from ..spec import (
    ModelFactorCaseSpec,
)


def _build_diagnostics_run_meta(
    *,
    spec: ModelFactorCaseSpec | None,
    evaluation_profile: str,
    output_dir: Path | None,
    status: str,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "workflow": "real_case_model_factor",
        "status": status,
        "evaluation_profile": evaluation_profile,
        "output_dir": str(output_dir) if output_dir is not None else None,
    }
    if spec is not None:
        payload.update(
            {
                "case_name": spec.name,
                "factor_name": spec.factor_name,
                "model_family": spec.model.family,
                "target_horizon": int(spec.target.horizon),
                "feature_count": len(spec.feature_columns),
                "prices_path": spec.prices_path,
                "features_path": spec.features_path,
            }
        )
    return payload


def _annotate_exception_with_diagnostics(
    exc: Exception,
    *,
    output_dir: Path | None,
    diagnostics_path: Path | None,
) -> None:
    try:
        if output_dir is not None:
            exc.model_lab_output_dir = str(output_dir)  # type: ignore[attr-defined]
        if diagnostics_path is not None:
            exc.model_lab_artifact_paths = {  # type: ignore[attr-defined]
                "diagnostics": str(diagnostics_path),
            }
    except Exception:  # noqa: BLE001
        return
