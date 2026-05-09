from __future__ import annotations

import multiprocessing as mp
import os
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, cast

import pandas as pd

from alpha_lab.custom_factors import (
    CustomFactorSource,
    find_custom_factor_workspace_root,
    load_persisted_custom_factors,
)
from alpha_lab.exceptions import AlphaLabConfigError, AlphaLabDataError
from alpha_lab.factor_recipe import FactorRecipeError, build_factor_from_recipe_mapping
from alpha_lab.interfaces import validate_factor_output
from alpha_lab.labels import forward_return
from alpha_lab.neutralization import neutralize_signal
from alpha_lab.real_cases.common_io import (
    apply_universe_to_factor,
    apply_universe_to_prices,
    load_prices,
    load_tabular_frame,
    load_universe_mask,
    resolve_tabular_frame_path,
)
from alpha_lab.research_contracts import validate_canonical_signal_table
from alpha_lab.research_evaluation_config import get_research_evaluation_config
from alpha_lab.research_integrity.contracts import IntegrityCheckResult, IntegrityReport
from alpha_lab.research_integrity.exceptions import raise_on_hard_failures
from alpha_lab.research_integrity.leakage_checks import (
    check_asof_inputs_not_after_signal_date,
    check_cross_section_transform_scope,
    check_factor_label_temporal_order,
    check_no_future_dates_in_input,
)
from alpha_lab.research_integrity.reporting import build_integrity_report
from alpha_lab.signal_transforms import (
    apply_min_coverage_gate,
    rank_cross_section,
    winsorize_cross_section,
    zscore_cross_section,
)
from alpha_lab.splits import (
    TimeSeriesSplitContract,
    infer_default_time_series_split_contract,
    rebalance_frequency_to_step,
)

from .artifacts import SingleFactorArtifactPaths, export_artifact_bundle
from .evaluate import SingleFactorEvaluationResult, evaluate_single_factor_case
from .spec import FactorInputSpec, SingleFactorCaseSpec, load_single_factor_case_spec

FactorLoader = Callable[[SingleFactorCaseSpec], pd.DataFrame]
InputBundleKey = tuple[str, str | None, str]
BatchParallelMode = Literal["serial", "thread", "process"]


@dataclass(frozen=True)
class SingleFactorCaseRunResult:
    """End-to-end run result for one real-case single-factor research package."""

    spec: SingleFactorCaseSpec
    output_dir: Path
    factor_df: pd.DataFrame
    evaluation_result: SingleFactorEvaluationResult
    artifact_paths: SingleFactorArtifactPaths
    integrity_report: IntegrityReport
    custom_factor_source: CustomFactorSource | None = None


@dataclass(frozen=True)
class SingleFactorBaseFeatureCache:
    """Reusable precomputed returns + forward labels on a standardized panel."""

    prices_enriched: pd.DataFrame
    trailing_return_columns: tuple[str, ...]
    forward_labels_by_horizon: dict[int, pd.DataFrame]


@dataclass(frozen=True)
class SingleFactorInputBundle:
    """Standardized in-memory input bundle reusable across multiple factors."""

    prices_path: str
    universe_path: str | None
    universe_in_column: str
    prices_all: pd.DataFrame
    prices_panel: pd.DataFrame
    universe_mask: pd.DataFrame | None
    max_price_date: pd.Timestamp
    base_feature_cache: SingleFactorBaseFeatureCache


@dataclass(frozen=True)
class SingleFactorBatchParallelConfig:
    """Execution config for multi-factor batch runs."""

    mode: BatchParallelMode = "serial"
    max_workers: int | None = None
    factors_per_worker: int = 1

    def __post_init__(self) -> None:
        normalized = self.mode.strip().lower()
        if normalized not in {"serial", "thread", "process"}:
            raise AlphaLabConfigError(
                "batch parallel mode must be one of ['serial', 'thread', 'process']"
            )
        object.__setattr__(self, "mode", cast(BatchParallelMode, normalized))
        if self.max_workers is not None and self.max_workers <= 0:
            raise AlphaLabConfigError("max_workers must be > 0 when provided")
        if self.factors_per_worker <= 0:
            raise AlphaLabConfigError("factors_per_worker must be > 0")


@dataclass(frozen=True)
class SingleFactorBatchDefinition:
    """One factor-level override used to derive a case spec from a base spec."""

    factor_name: str
    case_name: str | None = None
    factor_path: str | None = None
    factor_input: FactorInputSpec | None = None

    def __post_init__(self) -> None:
        if not self.factor_name.strip():
            raise AlphaLabConfigError("batch factor definition factor_name must be non-empty")
        if self.case_name is not None and not self.case_name.strip():
            raise AlphaLabConfigError("batch factor definition case_name must be non-empty")
        if self.factor_path is not None and not self.factor_path.strip():
            raise AlphaLabConfigError("batch factor definition factor_path must be non-empty")


def prepare_base_features(
    prices: pd.DataFrame,
    *,
    trailing_return_horizons: Sequence[int] = (1, 5, 10, 20, 60),
    forward_label_horizons: Sequence[int] = (5, 10, 20),
) -> SingleFactorBaseFeatureCache:
    """Precompute frequently reused return series and close-to-close labels."""
    panel = prices.copy()
    panel = panel.sort_values(["asset", "date"], kind="mergesort").reset_index(drop=True)
    panel["close"] = pd.to_numeric(panel["close"], errors="coerce")

    grouped_close = panel.groupby("asset", sort=False)["close"]
    trailing_cols: list[str] = []
    for horizon in sorted({int(x) for x in trailing_return_horizons if int(x) > 0}):
        col = f"ret_{horizon}d"
        panel[col] = grouped_close.pct_change(horizon)
        trailing_cols.append(col)

    labels: dict[int, pd.DataFrame] = {}
    for horizon in sorted({int(x) for x in forward_label_horizons if int(x) > 0}):
        labels[horizon] = forward_return(panel, horizon=horizon)

    return SingleFactorBaseFeatureCache(
        prices_enriched=panel,
        trailing_return_columns=tuple(trailing_cols),
        forward_labels_by_horizon=labels,
    )


def _strict_split_contract_check(
    contract: TimeSeriesSplitContract,
    *,
    object_name: str,
    module_name: str,
) -> IntegrityCheckResult:
    metadata = contract.to_metadata()
    return IntegrityCheckResult(
        check_name="strict_time_series_split_contract",
        status="pass",
        severity="info",
        object_name=object_name,
        module_name=module_name,
        message=(
            "Strict chronological IS/OOS split resolved before evaluation: "
            f"IS {metadata['is_start']}..{metadata['is_end']}, "
            f"OOS {metadata['oos_start']}..{metadata['oos_end']}, "
            f"embargo={metadata['embargo_days']}."
        ),
        metrics=metadata,
    )


def load_standard_inputs(
    spec_or_path: SingleFactorCaseSpec | str | Path,
) -> SingleFactorInputBundle:
    """Load and standardize prices/universe once for multi-factor reuse."""
    spec = _resolve_single_factor_spec(spec_or_path)
    resolved_prices_path = str(resolve_tabular_frame_path(spec.prices_path, object_name="prices"))
    resolved_universe_path: str | None = None
    if spec.universe.path is not None:
        resolved_universe_path = str(
            resolve_tabular_frame_path(spec.universe.path, object_name="universe")
        )
    resolved_universe_spec = replace(spec.universe, path=resolved_universe_path)

    universe_mask = load_universe_mask(resolved_universe_spec)
    if universe_mask is not None:
        universe_mask = universe_mask.sort_values(["date", "asset"], kind="mergesort").reset_index(
            drop=True
        )
        universe_mask["in_universe"] = universe_mask["in_universe"].astype(bool)

    prices_all = load_prices(resolved_prices_path)
    prices_panel = (
        apply_universe_to_prices(prices_all, universe_mask)
        if universe_mask is not None
        else prices_all
    )
    base_feature_cache = prepare_base_features(
        prices_panel,
        trailing_return_horizons=(1, 5, 10, 20, 60),
        forward_label_horizons=tuple(sorted({1, 2, 3, 5, 10, 20, int(spec.target.horizon)})),
    )
    prices_panel_enriched = base_feature_cache.prices_enriched

    return SingleFactorInputBundle(
        prices_path=resolved_prices_path,
        universe_path=resolved_universe_path,
        universe_in_column=spec.universe.in_universe_column,
        prices_all=prices_all,
        prices_panel=prices_panel_enriched,
        universe_mask=universe_mask,
        max_price_date=pd.Timestamp(prices_all["date"].max()),
        base_feature_cache=base_feature_cache,
    )


def run_single_factor_batch(
    base_spec_or_path: SingleFactorCaseSpec | str | Path,
    factor_definitions: Sequence[SingleFactorBatchDefinition | Mapping[str, object]],
    *,
    output_root_dir: str | Path | None = None,
    factor_loader: FactorLoader | None = None,
    evaluation_profile: str = "default_research",
    vault_root: str | Path | None = None,
    vault_export_mode: str = "versioned",
    progress_callback: Callable[[str, int], None] | None = None,
    batch_parallel_config: SingleFactorBatchParallelConfig | None = None,
    reuse_input_bundle: bool = True,
) -> list[SingleFactorCaseRunResult]:
    """Run multiple factors from one base spec using factor-level definitions."""
    base_spec, base_dir = _resolve_base_spec_and_base_dir(base_spec_or_path)
    specs = _build_specs_from_batch_definitions(
        base_spec,
        factor_definitions,
        base_dir=base_dir,
    )
    return run_single_factor_cases(
        specs,
        output_root_dir=output_root_dir,
        factor_loader=factor_loader,
        evaluation_profile=evaluation_profile,
        vault_root=vault_root,
        vault_export_mode=vault_export_mode,
        progress_callback=progress_callback,
        batch_parallel_config=batch_parallel_config,
        reuse_input_bundle=reuse_input_bundle,
    )


def run_single_factor_cases(
    specs_or_paths: Sequence[SingleFactorCaseSpec | str | Path],
    *,
    output_root_dir: str | Path | None = None,
    factor_loader: FactorLoader | None = None,
    evaluation_profile: str = "default_research",
    vault_root: str | Path | None = None,
    vault_export_mode: str = "versioned",
    progress_callback: Callable[[str, int], None] | None = None,
    batch_parallel_config: SingleFactorBatchParallelConfig | None = None,
    reuse_input_bundle: bool = True,
) -> list[SingleFactorCaseRunResult]:
    """Run multiple single-factor cases with reusable shared read-only inputs."""
    resolved_specs = [_resolve_single_factor_spec(item) for item in specs_or_paths]
    if not resolved_specs:
        return []
    parallel_config = batch_parallel_config or SingleFactorBatchParallelConfig()
    if parallel_config.mode == "serial":
        return _run_single_factor_cases_serial(
            resolved_specs,
            output_root_dir=output_root_dir,
            factor_loader=factor_loader,
            evaluation_profile=evaluation_profile,
            vault_root=vault_root,
            vault_export_mode=vault_export_mode,
            progress_callback=progress_callback,
            reuse_input_bundle=reuse_input_bundle,
        )
    if parallel_config.mode == "process":
        return _run_single_factor_cases_process(
            resolved_specs,
            output_root_dir=output_root_dir,
            factor_loader=factor_loader,
            evaluation_profile=evaluation_profile,
            vault_root=vault_root,
            vault_export_mode=vault_export_mode,
            progress_callback=progress_callback,
            parallel_config=parallel_config,
            reuse_input_bundle=reuse_input_bundle,
        )
    return _run_single_factor_cases_threaded(
        resolved_specs,
        output_root_dir=output_root_dir,
        factor_loader=factor_loader,
        evaluation_profile=evaluation_profile,
        vault_root=vault_root,
        vault_export_mode=vault_export_mode,
        progress_callback=progress_callback,
        parallel_config=parallel_config,
        reuse_input_bundle=reuse_input_bundle,
    )


def _run_single_factor_cases_serial(
    resolved_specs: Sequence[SingleFactorCaseSpec],
    *,
    output_root_dir: str | Path | None,
    factor_loader: FactorLoader | None,
    evaluation_profile: str,
    vault_root: str | Path | None,
    vault_export_mode: str,
    progress_callback: Callable[[str, int], None] | None,
    reuse_input_bundle: bool,
) -> list[SingleFactorCaseRunResult]:
    if not reuse_input_bundle:
        return [
            run_single_factor_case(
                spec,
                output_root_dir=output_root_dir,
                factor_loader=factor_loader,
                evaluation_profile=evaluation_profile,
                vault_root=vault_root,
                vault_export_mode=vault_export_mode,
                progress_callback=progress_callback,
                input_bundle=None,
            )
            for spec in resolved_specs
        ]
    bundles: dict[InputBundleKey, SingleFactorInputBundle] = {}
    results: list[SingleFactorCaseRunResult] = []
    for spec in resolved_specs:
        key = _input_bundle_key(spec)
        bundle = bundles.get(key)
        if bundle is None:
            bundle = load_standard_inputs(spec)
            bundles[key] = bundle
        result = run_single_factor_case(
            spec,
            output_root_dir=output_root_dir,
            factor_loader=factor_loader,
            evaluation_profile=evaluation_profile,
            vault_root=vault_root,
            vault_export_mode=vault_export_mode,
            progress_callback=progress_callback,
            input_bundle=bundle,
        )
        results.append(result)
    return results


def _run_single_factor_cases_threaded(
    resolved_specs: Sequence[SingleFactorCaseSpec],
    *,
    output_root_dir: str | Path | None,
    factor_loader: FactorLoader | None,
    evaluation_profile: str,
    vault_root: str | Path | None,
    vault_export_mode: str,
    progress_callback: Callable[[str, int], None] | None,
    parallel_config: SingleFactorBatchParallelConfig,
    reuse_input_bundle: bool,
) -> list[SingleFactorCaseRunResult]:
    grouped_specs: dict[InputBundleKey, list[tuple[int, SingleFactorCaseSpec]]] = defaultdict(list)
    for idx, spec in enumerate(resolved_specs):
        key = _input_bundle_key(spec)
        grouped_specs[key].append((idx, spec))

    indexed_tasks: list[
        tuple[InputBundleKey, tuple[int, ...], tuple[SingleFactorCaseSpec, ...]]
    ] = []
    for key, indexed_specs in grouped_specs.items():
        for chunk in _chunk_list(indexed_specs, parallel_config.factors_per_worker):
            indices = tuple(item[0] for item in chunk)
            specs = tuple(item[1] for item in chunk)
            indexed_tasks.append((key, indices, specs))

    if not indexed_tasks:
        return []

    max_workers = parallel_config.max_workers
    if max_workers is None:
        cpu = max(1, os.cpu_count() or 1)
        max_workers = min(cpu, len(indexed_tasks))
    else:
        max_workers = min(max_workers, len(indexed_tasks))

    if max_workers <= 1:
        return _run_single_factor_cases_serial(
            resolved_specs,
            output_root_dir=output_root_dir,
            factor_loader=factor_loader,
            evaluation_profile=evaluation_profile,
            vault_root=vault_root,
            vault_export_mode=vault_export_mode,
            progress_callback=progress_callback,
            reuse_input_bundle=reuse_input_bundle,
        )

    bundles: dict[InputBundleKey, SingleFactorInputBundle] = {}
    if reuse_input_bundle:
        for key, indexed_specs in grouped_specs.items():
            if indexed_specs:
                bundles[key] = load_standard_inputs(indexed_specs[0][1])

    tasks: list[
        tuple[tuple[int, ...], tuple[SingleFactorCaseSpec, ...], SingleFactorInputBundle | None]
    ] = []
    for key, indices, specs in indexed_tasks:
        bundle = bundles.get(key) if reuse_input_bundle else None
        tasks.append((indices, specs, bundle))

    if progress_callback is not None:
        progress_callback(
            f"批量并行启动 mode=thread workers={max_workers} tasks={len(tasks)}",
            0,
        )

    results: list[SingleFactorCaseRunResult | None] = [None] * len(resolved_specs)
    completed_cases = 0
    futures: dict[
        Future[list[SingleFactorCaseRunResult]],
        tuple[int, ...],
    ] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for indices, specs, bundle in tasks:
            future = executor.submit(
                _run_single_factor_case_chunk,
                specs,
                output_root_dir=output_root_dir,
                factor_loader=factor_loader,
                evaluation_profile=evaluation_profile,
                vault_root=vault_root,
                vault_export_mode=vault_export_mode,
                input_bundle=bundle,
            )
            futures[future] = indices

        try:
            for future in as_completed(futures):
                indices = futures[future]
                chunk_results = future.result()
                if len(chunk_results) != len(indices):
                    raise RuntimeError("threaded batch returned mismatched result size")
                for idx, result in zip(indices, chunk_results, strict=True):
                    results[idx] = result
                completed_cases += len(chunk_results)
                if progress_callback is not None:
                    percent = int(completed_cases * 100 / max(1, len(resolved_specs)))
                    progress_callback(
                        f"批量并行进度 {completed_cases}/{len(resolved_specs)}",
                        min(100, percent),
                    )
        except Exception:
            for future in futures:
                future.cancel()
            raise

    if progress_callback is not None:
        progress_callback("批量并行完成", 100)

    final_results: list[SingleFactorCaseRunResult] = []
    for item in results:
        if item is None:
            raise RuntimeError("threaded batch missing case result")
        final_results.append(item)
    return final_results


def _run_single_factor_cases_process(
    resolved_specs: Sequence[SingleFactorCaseSpec],
    *,
    output_root_dir: str | Path | None,
    factor_loader: FactorLoader | None,
    evaluation_profile: str,
    vault_root: str | Path | None,
    vault_export_mode: str,
    progress_callback: Callable[[str, int], None] | None,
    parallel_config: SingleFactorBatchParallelConfig,
    reuse_input_bundle: bool,
) -> list[SingleFactorCaseRunResult]:
    if factor_loader is not None:
        raise AlphaLabConfigError(
            "process batch mode does not support custom factor_loader; "
            "use spec.factor_input.recipe or pre-generated factor files"
        )
    grouped_specs: dict[InputBundleKey, list[tuple[int, SingleFactorCaseSpec]]] = defaultdict(list)
    for idx, spec in enumerate(resolved_specs):
        key = _input_bundle_key(spec)
        grouped_specs[key].append((idx, spec))

    indexed_tasks: list[tuple[tuple[int, ...], tuple[SingleFactorCaseSpec, ...]]] = []
    for indexed_specs in grouped_specs.values():
        for chunk in _chunk_list(indexed_specs, parallel_config.factors_per_worker):
            indices = tuple(item[0] for item in chunk)
            specs = tuple(item[1] for item in chunk)
            indexed_tasks.append((indices, specs))

    if not indexed_tasks:
        return []

    max_workers = parallel_config.max_workers
    if max_workers is None:
        cpu = max(1, os.cpu_count() or 1)
        max_workers = min(cpu, len(indexed_tasks))
    else:
        max_workers = min(max_workers, len(indexed_tasks))

    if max_workers <= 1:
        return _run_single_factor_cases_serial(
            resolved_specs,
            output_root_dir=output_root_dir,
            factor_loader=factor_loader,
            evaluation_profile=evaluation_profile,
            vault_root=vault_root,
            vault_export_mode=vault_export_mode,
            progress_callback=progress_callback,
            reuse_input_bundle=reuse_input_bundle,
        )

    if progress_callback is not None:
        progress_callback(
            f"批量并行启动 mode=process workers={max_workers} tasks={len(indexed_tasks)}",
            0,
        )

    results: list[SingleFactorCaseRunResult | None] = [None] * len(resolved_specs)
    completed_cases = 0
    futures: dict[
        Future[list[SingleFactorCaseRunResult]],
        tuple[int, ...],
    ] = {}
    with ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=mp.get_context("spawn"),
    ) as executor:
        for indices, specs in indexed_tasks:
            future = executor.submit(
                _run_single_factor_case_chunk_process,
                specs,
                output_root_dir=output_root_dir,
                evaluation_profile=evaluation_profile,
                vault_root=vault_root,
                vault_export_mode=vault_export_mode,
                reuse_input_bundle=reuse_input_bundle,
            )
            futures[future] = indices

        try:
            for future in as_completed(futures):
                indices = futures[future]
                chunk_results = future.result()
                if len(chunk_results) != len(indices):
                    raise RuntimeError("process batch returned mismatched result size")
                for idx, result in zip(indices, chunk_results, strict=True):
                    results[idx] = result
                completed_cases += len(chunk_results)
                if progress_callback is not None:
                    percent = int(completed_cases * 100 / max(1, len(resolved_specs)))
                    progress_callback(
                        f"批量并行进度 {completed_cases}/{len(resolved_specs)}",
                        min(100, percent),
                    )
        except Exception:
            for future in futures:
                future.cancel()
            raise

    if progress_callback is not None:
        progress_callback("批量并行完成", 100)

    final_results: list[SingleFactorCaseRunResult] = []
    for item in results:
        if item is None:
            raise RuntimeError("process batch missing case result")
        final_results.append(item)
    return final_results


def _run_single_factor_case_chunk(
    specs: Sequence[SingleFactorCaseSpec],
    *,
    output_root_dir: str | Path | None,
    factor_loader: FactorLoader | None,
    evaluation_profile: str,
    vault_root: str | Path | None,
    vault_export_mode: str,
    input_bundle: SingleFactorInputBundle | None,
) -> list[SingleFactorCaseRunResult]:
    results: list[SingleFactorCaseRunResult] = []
    for spec in specs:
        results.append(
            run_single_factor_case(
                spec,
                output_root_dir=output_root_dir,
                factor_loader=factor_loader,
                evaluation_profile=evaluation_profile,
                vault_root=vault_root,
                vault_export_mode=vault_export_mode,
                progress_callback=None,
                input_bundle=input_bundle,
            )
        )
    return results


def _run_single_factor_case_chunk_process(
    specs: Sequence[SingleFactorCaseSpec],
    *,
    output_root_dir: str | Path | None,
    evaluation_profile: str,
    vault_root: str | Path | None,
    vault_export_mode: str,
    reuse_input_bundle: bool,
) -> list[SingleFactorCaseRunResult]:
    bundle: SingleFactorInputBundle | None = None
    if reuse_input_bundle:
        if not specs:
            return []
        bundle = load_standard_inputs(specs[0])
        for spec in specs[1:]:
            _ensure_bundle_compatible(bundle, spec=spec)

    results: list[SingleFactorCaseRunResult] = []
    for spec in specs:
        results.append(
            run_single_factor_case(
                spec,
                output_root_dir=output_root_dir,
                factor_loader=None,
                evaluation_profile=evaluation_profile,
                vault_root=vault_root,
                vault_export_mode=vault_export_mode,
                progress_callback=None,
                input_bundle=bundle,
            )
        )
    return results


def _resolve_base_spec_and_base_dir(
    base_spec_or_path: SingleFactorCaseSpec | str | Path,
) -> tuple[SingleFactorCaseSpec, Path]:
    if isinstance(base_spec_or_path, SingleFactorCaseSpec):
        return base_spec_or_path, Path.cwd().resolve()
    spec_path = Path(base_spec_or_path).resolve()
    return load_single_factor_case_spec(spec_path), spec_path.parent


def _build_specs_from_batch_definitions(
    base_spec: SingleFactorCaseSpec,
    factor_definitions: Sequence[SingleFactorBatchDefinition | Mapping[str, object]],
    *,
    base_dir: Path,
) -> list[SingleFactorCaseSpec]:
    if not factor_definitions:
        raise AlphaLabConfigError("factor_definitions must be a non-empty sequence")
    specs: list[SingleFactorCaseSpec] = []
    seen_case_names: set[str] = set()
    for idx, raw in enumerate(factor_definitions, start=1):
        definition = _parse_batch_definition(raw, index=idx, base_dir=base_dir)
        factor_name = definition.factor_name.strip()
        case_name = (definition.case_name or f"{base_spec.name}__{factor_name}").strip()
        if case_name in seen_case_names:
            raise AlphaLabConfigError(f"duplicate batch case_name generated: {case_name!r}")
        seen_case_names.add(case_name)

        factor_input = (
            definition.factor_input
            if definition.factor_input is not None
            else base_spec.factor_input
        )
        factor_path = definition.factor_path or base_spec.factor_path
        specs.append(
            replace(
                base_spec,
                name=case_name,
                factor_name=factor_name,
                factor_path=factor_path,
                factor_input=factor_input,
            )
        )
    return specs


def _parse_batch_definition(
    raw: SingleFactorBatchDefinition | Mapping[str, object],
    *,
    index: int,
    base_dir: Path,
) -> SingleFactorBatchDefinition:
    if isinstance(raw, SingleFactorBatchDefinition):
        definition = raw
    elif isinstance(raw, Mapping):
        definition = _batch_definition_from_mapping(raw, index=index)
    else:
        raise AlphaLabConfigError(
            f"factor_definitions[{index - 1}] must be a mapping or SingleFactorBatchDefinition"
        )

    factor_path = definition.factor_path
    if factor_path is not None:
        factor_path = str(_resolve_batch_path(factor_path, base_dir=base_dir))
    return SingleFactorBatchDefinition(
        factor_name=definition.factor_name.strip(),
        case_name=definition.case_name.strip() if definition.case_name is not None else None,
        factor_path=factor_path,
        factor_input=definition.factor_input,
    )


def _batch_definition_from_mapping(
    data: Mapping[str, object],
    *,
    index: int,
) -> SingleFactorBatchDefinition:
    allowed_keys = {"factor_name", "case_name", "name", "factor_path", "factor_input"}
    unknown = sorted(set(data) - allowed_keys)
    if unknown:
        raise AlphaLabConfigError(
            f"factor_definitions[{index - 1}] has unsupported keys: {unknown}"
        )

    factor_name_raw = data.get("factor_name")
    if not isinstance(factor_name_raw, str) or not factor_name_raw.strip():
        raise AlphaLabConfigError(
            f"factor_definitions[{index - 1}].factor_name must be a non-empty string"
        )

    case_name_raw = data.get("case_name", data.get("name"))
    case_name: str | None = None
    if case_name_raw is not None:
        if not isinstance(case_name_raw, str) or not case_name_raw.strip():
            raise AlphaLabConfigError(
                f"factor_definitions[{index - 1}].case_name must be a non-empty string"
            )
        case_name = case_name_raw.strip()

    factor_path_raw = data.get("factor_path")
    factor_path: str | None = None
    if factor_path_raw is not None:
        if not isinstance(factor_path_raw, str) or not factor_path_raw.strip():
            raise AlphaLabConfigError(
                f"factor_definitions[{index - 1}].factor_path must be a non-empty string"
            )
        factor_path = factor_path_raw.strip()

    factor_input = _coerce_factor_input_spec(
        data.get("factor_input"),
        index=index,
    )
    return SingleFactorBatchDefinition(
        factor_name=factor_name_raw.strip(),
        case_name=case_name,
        factor_path=factor_path,
        factor_input=factor_input,
    )


def _coerce_factor_input_spec(
    raw: object,
    *,
    index: int,
) -> FactorInputSpec | None:
    if raw is None:
        return None
    if isinstance(raw, FactorInputSpec):
        return raw
    if not isinstance(raw, Mapping):
        raise AlphaLabConfigError(f"factor_definitions[{index - 1}].factor_input must be an object")

    allowed_keys = {"mode", "recipe", "disable_pipeline_preprocess"}
    unknown = sorted(set(raw) - allowed_keys)
    if unknown:
        raise AlphaLabConfigError(
            f"factor_definitions[{index - 1}].factor_input has unsupported keys: {unknown}"
        )

    mode_raw = raw.get("mode", "recipe")
    if not isinstance(mode_raw, str):
        raise AlphaLabConfigError(
            f"factor_definitions[{index - 1}].factor_input.mode must be a string"
        )
    recipe_raw = raw.get("recipe")
    if recipe_raw is not None and not isinstance(recipe_raw, Mapping):
        raise AlphaLabConfigError(
            f"factor_definitions[{index - 1}].factor_input.recipe must be an object"
        )
    disable_raw = raw.get("disable_pipeline_preprocess", True)
    if not isinstance(disable_raw, bool):
        raise AlphaLabConfigError("factor_input.disable_pipeline_preprocess must be a boolean")
    return FactorInputSpec(
        mode=mode_raw.strip().lower(),
        recipe=cast(Mapping[str, object] | None, recipe_raw),
        disable_pipeline_preprocess=disable_raw,
    )


def _resolve_batch_path(path_text: str, *, base_dir: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def _input_bundle_key(spec: SingleFactorCaseSpec) -> InputBundleKey:
    return _resolved_input_bundle_key(spec)


def _chunk_list(
    items: Sequence[tuple[int, SingleFactorCaseSpec]],
    chunk_size: int,
) -> list[list[tuple[int, SingleFactorCaseSpec]]]:
    return [list(items[start : start + chunk_size]) for start in range(0, len(items), chunk_size)]


def run_single_factor_case(
    spec_or_path: SingleFactorCaseSpec | str | Path,
    *,
    output_root_dir: str | Path | None = None,
    factor_loader: FactorLoader | None = None,
    evaluation_profile: str = "default_research",
    vault_root: str | Path | None = None,
    vault_export_mode: str = "versioned",
    progress_callback: Callable[[str, int], None] | None = None,
    fast_screen_artifact_root: str | Path | None = None,
    fast_screen_run_id: str | None = None,
    input_bundle: SingleFactorInputBundle | None = None,
) -> SingleFactorCaseRunResult:
    """Run one real-case single-factor study end-to-end and export artifacts."""
    integrity_checks: list[IntegrityCheckResult] = []

    def _emit_progress(message: str, percent: int) -> None:
        if progress_callback is not None:
            progress_callback(message, percent)

    def _record_integrity(check: IntegrityCheckResult) -> None:
        integrity_checks.append(check)
        raise_on_hard_failures((check,))

    _emit_progress("读取实验合同文件", 3)
    spec_path: Path | None = None
    if isinstance(spec_or_path, SingleFactorCaseSpec):
        spec = spec_or_path
    else:
        spec_path = Path(spec_or_path).resolve()
        spec = load_single_factor_case_spec(spec_path)

    custom_factor_source = _load_custom_factor_source_for_spec(
        spec,
        spec_path=spec_path,
        enabled=factor_loader is None,
    )

    evaluation_config = get_research_evaluation_config(evaluation_profile)
    _emit_progress("实验合同与评估配置已加载", 10)

    _emit_progress("加载行情与可选股票池", 15)
    bundle = input_bundle if input_bundle is not None else load_standard_inputs(spec)
    _ensure_bundle_compatible(bundle, spec=spec)
    universe_mask = bundle.universe_mask
    prices_all = bundle.prices_all
    prices = bundle.prices_panel
    max_price_date = bundle.max_price_date
    _record_integrity(
        check_no_future_dates_in_input(
            prices_all,
            max_allowed_date=max_price_date,
            date_col="date",
            object_name="single_factor_prices",
        )
    )
    if universe_mask is not None:
        _record_integrity(
            check_no_future_dates_in_input(
                universe_mask,
                max_allowed_date=max_price_date,
                date_col="date",
                object_name="single_factor_universe",
            )
        )
        _record_integrity(
            check_asof_inputs_not_after_signal_date(
                prices_all[["date", "asset"]],
                universe_mask,
                by=("asset",),
                signal_date_col="date",
                aux_effective_date_col="date",
                aux_known_at_col=None,
                object_name="single_factor_universe_asof",
            )
        )
    split_contract = infer_default_time_series_split_contract(
        prices["date"],
        target_horizon=int(spec.target.horizon),
        rebalance_step=rebalance_frequency_to_step(spec.rebalance_frequency),
        source="single_factor_pipeline",
    )
    _record_integrity(
        _strict_split_contract_check(
            split_contract,
            object_name="single_factor_strict_split",
            module_name="real_cases.single_factor.pipeline",
        )
    )

    _emit_progress("加载因子输入", 30)
    raw_factor = (
        _load_factor_from_recipe(spec, prices=prices)
        if (
            factor_loader is None
            and spec.factor_input is not None
            and spec.factor_input.mode == "recipe"
        )
        else (factor_loader or _default_factor_loader)(spec)
    )
    if "date" in raw_factor.columns:
        _record_integrity(
            check_no_future_dates_in_input(
                raw_factor,
                max_allowed_date=max_price_date,
                date_col="date",
                object_name="single_factor_raw_factor",
            )
        )

    _emit_progress("预处理因子并应用股票池", 42)
    factor_df = _prepare_factor(raw_factor, spec=spec)
    if universe_mask is not None:
        factor_df = apply_universe_to_factor(factor_df, universe_mask)

    raw_factor_df = factor_df.copy()
    _emit_progress("处理中性化与覆盖率检查", 52)
    factor_df, neutral_diag = _maybe_neutralize_factor(
        factor_df,
        spec=spec,
        universe_mask=universe_mask,
        integrity_checks=integrity_checks,
        max_price_date=max_price_date,
    )
    coverage_by_date = _coverage_by_date(factor_df)

    validate_factor_output(factor_df)
    _record_integrity(
        check_cross_section_transform_scope(
            prices[["date", "asset"]],
            factor_df[["date", "asset", "value"]],
            date_col="date",
            asset_col="asset",
            object_name="single_factor_final_factor_scope",
        )
    )

    _emit_progress("运行评估与完整性检查", 68)
    evaluation_result = evaluate_single_factor_case(
        prices=prices,
        factor_df=factor_df,
        raw_factor_df=raw_factor_df,
        spec=spec,
        coverage_by_date=coverage_by_date,
        neutralization_summary=neutral_diag,
        precomputed_forward_labels=bundle.base_feature_cache.forward_labels_by_horizon,
        evaluation_config=evaluation_config,
        split_contract=split_contract,
        progress_callback=lambda message, percent: _emit_progress(
            message,
            min(83, 68 + max(0, min(int(percent), 100)) * 15 // 100),
        ),
    )
    for check in evaluation_result.experiment_result.integrity_checks:
        _record_integrity(check)
    _record_integrity(
        check_factor_label_temporal_order(
            evaluation_result.experiment_result.factor_df,
            evaluation_result.experiment_result.label_df,
            join_keys=("date", "asset"),
            factor_date_col="date",
            label_date_col="date",
            object_name="single_factor_factor_label_alignment",
        )
    )
    integrity_report = build_integrity_report(
        tuple(integrity_checks),
        context={
            "pipeline": "run_single_factor_case",
            "case_name": spec.name,
            "prices_path": spec.prices_path,
            "factor_path": spec.factor_path,
            "factor_name": spec.factor_name,
            "neutralization_enabled": bool(spec.neutralization.enabled),
            "split_contract": split_contract.to_metadata(),
        },
    )

    _emit_progress("导出实验产物", 84)
    root_dir = (
        Path(output_root_dir).resolve()
        if output_root_dir is not None
        else Path(spec.output.root_dir)
    )
    output_dir = (root_dir.resolve() / spec.name).resolve()

    artifact_paths = export_artifact_bundle(
        spec=spec,
        evaluation_result=evaluation_result,
        integrity_report=integrity_report,
        output_dir=output_dir,
        spec_path=spec_path,
        evaluation_config=evaluation_config,
        vault_root=vault_root,
        vault_export_mode=vault_export_mode,
        custom_factor_source=(
            custom_factor_source.to_audit_dict() if custom_factor_source is not None else None
        ),
    )
    _emit_progress("实验产物导出完成", 90)

    if fast_screen_artifact_root is not None:
        try:
            from alpha_lab.fast_screen import (
                Tier1Inputs,
                run_tier1,
                save_tier1_result,
            )

            fs_inputs = Tier1Inputs(
                factor_name=spec.factor_name,
                factor_df=factor_df,
                prices=prices,
                horizon=spec.target.horizon,
                n_quantiles=spec.n_quantiles,
                cost_rate=spec.transaction_cost.one_way_rate,
                universe=spec.universe.name,
                frequency="daily",
            )
            fs_result = run_tier1(fs_inputs, run_id=fast_screen_run_id)
            save_tier1_result(fast_screen_artifact_root, fs_result)
            _emit_progress("Fast Screen 产物已生成", 92)
        except Exception:  # noqa: BLE001 — don't fail main run on Tier-1 emission issue
            _emit_progress("Fast Screen 产物生成失败（已跳过）", 92)

    return SingleFactorCaseRunResult(
        spec=spec,
        output_dir=output_dir,
        factor_df=factor_df,
        evaluation_result=evaluation_result,
        artifact_paths=artifact_paths,
        integrity_report=integrity_report,
        custom_factor_source=custom_factor_source,
    )


def _load_custom_factor_source_for_spec(
    spec: SingleFactorCaseSpec,
    *,
    spec_path: Path | None,
    enabled: bool,
) -> CustomFactorSource | None:
    if not enabled:
        return None
    workspace_root = find_custom_factor_workspace_root(spec_path)
    sources = load_persisted_custom_factors(workspace_root, ignore_errors=True)
    method = _recipe_base_method(spec) or spec.factor_name
    return sources.get(method.strip().lower()) if method else None


def _recipe_base_method(spec: SingleFactorCaseSpec) -> str | None:
    factor_input = spec.factor_input
    if factor_input is None or factor_input.recipe is None:
        return None
    base = factor_input.recipe.get("base")
    if not isinstance(base, Mapping):
        return None
    method = base.get("method")
    return method.strip() if isinstance(method, str) and method.strip() else None


def _resolve_single_factor_spec(
    spec_or_path: SingleFactorCaseSpec | str | Path,
) -> SingleFactorCaseSpec:
    if isinstance(spec_or_path, SingleFactorCaseSpec):
        return spec_or_path
    return load_single_factor_case_spec(Path(spec_or_path).resolve())


def _ensure_bundle_compatible(
    bundle: SingleFactorInputBundle,
    *,
    spec: SingleFactorCaseSpec,
) -> None:
    (
        expected_prices_path,
        expected_universe_path,
        expected_universe_col,
    ) = _resolved_input_bundle_key(spec)
    if bundle.prices_path != expected_prices_path:
        raise AlphaLabConfigError("input_bundle.prices_path does not match spec.prices_path")
    if bundle.universe_path != expected_universe_path:
        raise AlphaLabConfigError("input_bundle.universe_path does not match spec.universe.path")
    if bundle.universe_in_column != expected_universe_col:
        raise AlphaLabConfigError(
            "input_bundle.universe_in_column does not match spec.universe.in_universe_column"
        )


def _resolved_input_bundle_key(spec: SingleFactorCaseSpec) -> InputBundleKey:
    prices_path = str(resolve_tabular_frame_path(spec.prices_path, object_name="prices"))
    universe_path: str | None = None
    if spec.universe.path is not None:
        universe_path = str(resolve_tabular_frame_path(spec.universe.path, object_name="universe"))
    return (prices_path, universe_path, spec.universe.in_universe_column)


def _default_factor_loader(spec: SingleFactorCaseSpec) -> pd.DataFrame:
    return load_tabular_frame(spec.factor_path, object_name="factor")


def _load_factor_from_recipe(
    spec: SingleFactorCaseSpec,
    *,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    factor_input = spec.factor_input
    if factor_input is None or factor_input.mode != "recipe" or factor_input.recipe is None:
        raise AlphaLabConfigError("factor_input.recipe is required when using recipe mode")
    try:
        return build_factor_from_recipe_mapping(
            prices=prices,
            recipe=factor_input.recipe,
            factor_name=spec.factor_name,
        )
    except FactorRecipeError as exc:
        raise AlphaLabDataError(
            f"invalid factor recipe for factor_name={spec.factor_name!r}: {exc}"
        ) from exc


def _prepare_factor(raw: pd.DataFrame, *, spec: SingleFactorCaseSpec) -> pd.DataFrame:
    missing = {"date", "asset", "factor", "value"} - set(raw.columns)
    if missing:
        raise AlphaLabDataError(f"factor file is missing required columns: {sorted(missing)}")

    frame = raw.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame[frame["factor"].astype(str) == spec.factor_name].copy()
    if frame.empty:
        raise AlphaLabDataError(f"factor file has no rows for factor_name={spec.factor_name!r}")

    frame["factor"] = spec.factor_name
    frame = frame[["date", "asset", "factor", "value"]].copy()
    validate_canonical_signal_table(frame, table_name="single_factor")

    transformed = frame[["date", "asset", "value"]].copy()
    if spec.preprocess.winsorize:
        transformed = winsorize_cross_section(
            transformed,
            lower=spec.preprocess.winsorize_lower,
            upper=spec.preprocess.winsorize_upper,
            min_group_size=spec.preprocess.min_group_size,
        )

    if spec.preprocess.standardization == "zscore":
        transformed = zscore_cross_section(
            transformed,
            min_group_size=spec.preprocess.min_group_size,
        )
    elif spec.preprocess.standardization == "rank":
        transformed = rank_cross_section(
            transformed,
            min_group_size=max(2, spec.preprocess.min_group_size),
            pct=True,
        )

    if spec.direction == "short":
        transformed["value"] = -transformed["value"]

    if spec.preprocess.min_coverage is not None:
        transformed = apply_min_coverage_gate(
            transformed,
            min_coverage=spec.preprocess.min_coverage,
        )

    out = transformed.copy()
    out["factor"] = spec.factor_name
    out = out[["date", "asset", "factor", "value"]]
    return out.sort_values(["date", "asset"], kind="mergesort").reset_index(drop=True)


def _maybe_neutralize_factor(
    factor_df: pd.DataFrame,
    *,
    spec: SingleFactorCaseSpec,
    universe_mask: pd.DataFrame | None,
    integrity_checks: list[IntegrityCheckResult] | None = None,
    max_price_date: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    if not spec.neutralization.enabled:
        return factor_df, None

    exposures_path = spec.neutralization.exposures_path
    if exposures_path is None:
        raise AlphaLabConfigError(
            "neutralization.exposures_path is required when neutralization is enabled"
        )

    exposures = load_tabular_frame(
        exposures_path,
        object_name="neutralization exposures",
    )
    exposures["date"] = pd.to_datetime(exposures["date"], errors="coerce")

    required = {"date", "asset"}
    if spec.neutralization.size_col is not None:
        required.add(spec.neutralization.size_col)
    if spec.neutralization.industry_col is not None:
        required.add(spec.neutralization.industry_col)

    missing = required - set(exposures.columns)
    if missing:
        raise AlphaLabDataError(
            f"neutralization exposure file is missing required columns: {sorted(missing)}"
        )
    known_at_col = None
    if "known_at" in exposures.columns:
        known_at_col = "known_at"
    elif "available_at" in exposures.columns:
        known_at_col = "available_at"

    if integrity_checks is not None and max_price_date is not None:
        no_future_check = check_no_future_dates_in_input(
            exposures,
            max_allowed_date=max_price_date,
            date_col="date",
            object_name="single_factor_neutralization_exposures",
        )
        integrity_checks.append(no_future_check)
        raise_on_hard_failures((no_future_check,))

        asof_check = check_asof_inputs_not_after_signal_date(
            factor_df[["date", "asset"]],
            exposures,
            by=("asset",),
            signal_date_col="date",
            aux_effective_date_col="date",
            aux_known_at_col=known_at_col,
            object_name="single_factor_neutralization_exposures_asof",
        )
        integrity_checks.append(asof_check)
        raise_on_hard_failures((asof_check,))

    if universe_mask is not None:
        active = universe_mask[universe_mask["in_universe"]][["date", "asset"]]
        exposures = exposures.merge(
            active,
            on=["date", "asset"],
            how="inner",
            validate="many_to_one",
        )

    merged = factor_df[["date", "asset", "value"]].merge(
        exposures,
        on=["date", "asset"],
        how="left",
        validate="one_to_one",
    )

    size_col = spec.neutralization.size_col
    industry_col = spec.neutralization.industry_col

    if size_col is not None:
        merged["__size_input"] = merged[size_col]
        size_col = "__size_input"
    if industry_col is not None:
        merged["__industry_input"] = merged[industry_col]
        industry_col = "__industry_input"
    known_at_input = None
    if known_at_col is not None:
        merged["__known_at_input"] = pd.to_datetime(
            merged[known_at_col],
            errors="coerce",
        )
        known_at_input = "__known_at_input"

    cols = ["date", "asset", "value"]
    for col in (size_col, industry_col):
        if col is not None:
            cols.append(col)
    if known_at_input is not None:
        cols.append(known_at_input)

    neutralized = neutralize_signal(
        merged[cols].copy(),
        value_col="value",
        by="date",
        size_col=size_col,
        industry_col=industry_col,
        beta_col=None,
        min_obs=spec.neutralization.min_obs,
        ridge=spec.neutralization.ridge,
        output_col="value_neutralized",
        known_at_col=known_at_input,
        enforce_integrity=True,
    )
    if integrity_checks is not None:
        integrity_checks.extend(list(neutralized.integrity_checks))
        raise_on_hard_failures(neutralized.integrity_checks)

    out = factor_df[["date", "asset", "factor"]].copy()
    out = out.merge(
        neutralized.data[["date", "asset", "value_neutralized"]],
        on=["date", "asset"],
        how="left",
        validate="one_to_one",
    )
    out = out.rename(columns={"value_neutralized": "value"})
    return out, neutralized.diagnostics


def _coverage_by_date(factor_df: pd.DataFrame) -> pd.DataFrame:
    if factor_df.empty:
        return pd.DataFrame(columns=["date", "n_assets", "coverage", "missingness"])

    summary = factor_df.groupby("date", sort=True).agg(
        n_assets=("asset", "nunique"),
        n_non_null=("value", lambda s: int(s.notna().sum())),
    )
    summary["coverage"] = summary["n_non_null"] / summary["n_assets"].replace(0, pd.NA)
    summary["missingness"] = 1.0 - summary["coverage"]
    return summary.reset_index()[["date", "n_assets", "coverage", "missingness"]]
