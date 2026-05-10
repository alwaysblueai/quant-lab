from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path

from alpha_lab.model_factor import (
    TrainingSpec,
)
from alpha_lab.model_factor.dataset_cache import (
    ModelFactorDatasetCache,
)
from alpha_lab.model_factor.diagnostics import (
    ModelFactorDiagnosticsRecorder,
)

from ..spec import (
    ModelFactorCaseSpec,
)


def _resolve_case_output_dir(
    spec: ModelFactorCaseSpec,
    *,
    output_root_dir: str | Path | None,
) -> Path:
    root_dir = (
        Path(output_root_dir).resolve()
        if output_root_dir is not None
        else Path(spec.output.root_dir)
    )
    return (root_dir.resolve() / spec.name).resolve()


def _resolve_preparation_cache_dir(
    output_dir: Path,
    *,
    cache_root_dir: str | Path | None = None,
) -> Path:
    if cache_root_dir is not None:
        return Path(cache_root_dir).expanduser().resolve() / "_model_factor_cache"
    fallback = output_dir.parent.resolve() / "_model_factor_cache"
    # Web runs land output under <root>/_web_runs/<run_id>/<case>. Falling back
    # to output_dir.parent in that case would land the cache under the per-run
    # directory and silently duplicate ~3-4GB of feature matrices per submission.
    # The web launcher always passes --cache-root-dir; reaching this branch with
    # an _web_runs output dir means the launcher contract was violated. Fail
    # loudly so the cache leak does not return.
    if "_web_runs" in fallback.parts:
        raise ValueError(
            "model_factor preparation cache cannot fall back to a per-run "
            f"directory under _web_runs ({fallback}). The web launcher must "
            "pass --cache-root-dir pointing at a shared location (typically "
            "<output_root>/_model_factor_shared_cache); without it, prepared "
            "inputs would be duplicated per run."
        )
    return fallback


def _resolve_screening_training_override(
    *,
    spec: ModelFactorCaseSpec,
    evaluation_profile: str,
    screening_retrain_every_n_dates: int | None,
    diagnostics: ModelFactorDiagnosticsRecorder,
) -> TrainingSpec:
    if screening_retrain_every_n_dates is None:
        return spec.training
    if screening_retrain_every_n_dates <= 0:
        raise ValueError("screening_retrain_every_n_dates must be > 0")
    if evaluation_profile != "exploratory_screening":
        diagnostics.warning(
            title="筛选重训间隔覆盖未生效",
            severity="warning",
            stage="spec_load",
            description=(
                "screening_retrain_every_n_dates 只在 exploratory_screening profile 下生效，"
                f"当前 profile={evaluation_profile!r}，将保持合同中的训练节奏。"
            ),
            suggested_action=(
                "若要启用快速筛选重训间隔，请使用 --evaluation-profile exploratory_screening。"
            ),
        )
        return spec.training
    original = int(spec.training.retrain_every_n_dates)
    effective = max(original, int(screening_retrain_every_n_dates))
    diagnostics.event(
        level="info",
        stage="spec_load",
        message="screening retrain cadence override applied",
        payload={
            "original_retrain_every_n_dates": original,
            "effective_retrain_every_n_dates": effective,
            "expected_fit_count_ratio": (
                (float(original) / float(effective)) if effective > 0 else None
            ),
        },
    )
    return replace(spec.training, retrain_every_n_dates=effective)


def _build_preparation_cache_key(
    *,
    dataset_cache: ModelFactorDatasetCache,
    spec: ModelFactorCaseSpec,
    feature_storage_path: Path,
    feature_source_path: Path,
    price_columns: tuple[str, ...],
    optional_price_columns: tuple[str, ...],
    evaluation_profile: str,
) -> str:
    payload = {
        "features_path": dataset_cache.file_signature(feature_storage_path),
        "features_source_path": dataset_cache.file_signature(feature_source_path),
        "prices_path": dataset_cache.file_signature(Path(spec.prices_path)),
        "universe": asdict(spec.universe),
        "universe_path": (
            dataset_cache.file_signature(Path(spec.universe.path))
            if spec.universe.path is not None
            else None
        ),
        "feature_columns": list(spec.feature_columns),
        "feature_availability": asdict(spec.feature_availability),
        "feature_preprocess": asdict(spec.feature_preprocess),
        "target": asdict(spec.target),
        "price_columns": list(price_columns),
        "optional_price_columns": list(optional_price_columns),
        "evaluation_profile": str(evaluation_profile),
    }
    return dataset_cache.build_key(payload)
