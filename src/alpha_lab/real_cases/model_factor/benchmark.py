from __future__ import annotations

import hashlib
import json
import subprocess
import threading
import time
import tracemalloc
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from alpha_lab.real_cases.model_factor.pipeline import (
    ModelFactorCaseRunResult,
    run_model_factor_case,
)
from alpha_lab.real_cases.model_factor.spec import (
    ModelFactorCaseSpec,
    load_model_factor_case_spec,
)

try:
    import resource as _resource
except ImportError:  # pragma: no cover - exercised on Windows.
    _resource = None  # type: ignore[assignment]

resource = _resource

_BENCHMARK_SCHEMA_VERSION = "1.0.0"
_MEMORY_PROFILE_SCHEMA_VERSION = "1.0.0"
_TRACEMALLOC_STAGE_NAMES: frozenset[str] = frozenset(
    {
        "data_load",
        "feature_validate",
        "target_build",
        "training_window_index",
        "model_fit",
        "predict",
        "evaluate",
    }
)


@dataclass(frozen=True)
class ModelFactorBenchmarkOptions:
    spec_path: str | Path
    output_root_dir: str | Path | None = None
    cache_root_dir: str | Path | None = None
    benchmark_output_dir: str | Path | None = None
    evaluation_profile: str = "default_research"
    use_preparation_cache: bool = True
    screening_retrain_every_n_dates: int | None = None
    vault_root: str | Path | None = None
    vault_export_mode: str = "skip"
    run_id: str | None = None
    note: str | None = None
    memory_sample_interval_seconds: float = 1.0
    memory_limit_gb: float | None = None
    memory_profile: bool = False
    tracemalloc_profile: bool = False
    tracemalloc_nframes: int = 10

    def __post_init__(self) -> None:
        if self.memory_sample_interval_seconds <= 0:
            raise ValueError("memory_sample_interval_seconds must be > 0")
        if self.memory_limit_gb is not None and self.memory_limit_gb <= 0:
            raise ValueError("memory_limit_gb must be > 0 when provided")
        if self.tracemalloc_nframes <= 0:
            raise ValueError("tracemalloc_nframes must be > 0")


@dataclass(frozen=True)
class ModelFactorBenchmarkResult:
    record: dict[str, object]
    record_path: Path
    case_result: ModelFactorCaseRunResult | None


def run_model_factor_benchmark(
    options: ModelFactorBenchmarkOptions,
) -> ModelFactorBenchmarkResult:
    """Run one model-factor case and persist a comparable benchmark record."""

    spec_path = Path(options.spec_path).resolve()
    spec = load_model_factor_case_spec(spec_path)
    run_id = options.run_id or _default_run_id(spec)
    record_dir = _resolve_benchmark_output_dir(options, spec=spec)
    record_path = record_dir / f"{run_id}.json"
    config_snapshot = _build_config_snapshot(options, spec_path=spec_path, spec=spec)
    git_snapshot = _git_snapshot(spec_path.parent)
    resource_limits = _apply_memory_limit(options.memory_limit_gb)

    stage_memory_tracker = _StageMemoryTracker(
        tracemalloc_enabled=bool(options.tracemalloc_profile),
        tracemalloc_nframes=int(options.tracemalloc_nframes),
    )
    sampler = _ProcessMemorySampler(
        interval_seconds=float(options.memory_sample_interval_seconds),
        capture_timeline=bool(options.memory_profile),
        current_stage_provider=stage_memory_tracker.current_stage,
    )
    started_at = _utc_now_iso()
    wall_start = time.perf_counter()
    case_result: ModelFactorCaseRunResult | None = None
    status = "success"
    error: dict[str, object] | None = None
    failure_artifact_paths: dict[str, str] = {}

    try:
        sampler.start()
        case_result = run_model_factor_case(
            spec_path,
            output_root_dir=options.output_root_dir,
            cache_root_dir=options.cache_root_dir,
            evaluation_profile=options.evaluation_profile,
            use_preparation_cache=options.use_preparation_cache,
            screening_retrain_every_n_dates=options.screening_retrain_every_n_dates,
            vault_root=options.vault_root,
            vault_export_mode=options.vault_export_mode,
            stage_lifecycle_callback=stage_memory_tracker.stage_lifecycle_callback,
        )
    except Exception as exc:
        status = "failed"
        failure_artifact_paths = _failure_artifact_paths(exc)
        error = {
            "type": type(exc).__name__,
            "message": str(exc),
            "model_lab_output_dir": getattr(exc, "model_lab_output_dir", None),
            "artifact_paths": failure_artifact_paths,
        }
    finally:
        wall_seconds = time.perf_counter() - wall_start
        sampler.stop()
        stage_memory_tracker.stop()
    finished_at = _utc_now_iso()
    memory_profile_path = _write_memory_profile_artifact(
        record_dir=record_dir,
        run_id=run_id,
        enabled=bool(options.memory_profile or options.tracemalloc_profile),
        sampler=sampler,
        stage_memory_tracker=stage_memory_tracker,
        started_at=started_at,
        finished_at=finished_at,
    )

    record = _build_record(
        run_id=run_id,
        status=status,
        started_at=started_at,
        finished_at=finished_at,
        wall_seconds=wall_seconds,
        config_snapshot=config_snapshot,
        git_snapshot=git_snapshot,
        memory_summary=sampler.summary(),
        resource_limits=resource_limits,
        memory_profile_path=memory_profile_path,
        case_result=case_result,
        failure_artifact_paths=failure_artifact_paths,
        error=error,
        note=options.note,
    )
    _write_record(record_path, record)
    _write_record(record_dir / "latest.json", record)

    if error is not None:
        raise RuntimeError(
            f"model-factor benchmark failed; record written to {record_path}: "
            f"{error['type']}: {error['message']}"
        )
    return ModelFactorBenchmarkResult(
        record=record,
        record_path=record_path,
        case_result=case_result,
    )


def _build_record(
    *,
    run_id: str,
    status: str,
    started_at: str,
    finished_at: str,
    wall_seconds: float,
    config_snapshot: dict[str, object],
    git_snapshot: dict[str, object],
    memory_summary: dict[str, object],
    resource_limits: dict[str, object],
    memory_profile_path: Path | None,
    case_result: ModelFactorCaseRunResult | None,
    failure_artifact_paths: dict[str, str],
    error: dict[str, object] | None,
    note: str | None,
) -> dict[str, object]:
    diagnostics_payload = _read_json_artifact(
        case_result,
        "diagnostics",
        failure_artifact_paths=failure_artifact_paths,
    )
    metrics_payload = _read_json_artifact(case_result, "metrics")
    run_manifest_payload = _read_json_artifact(case_result, "run_manifest")
    model_definition_payload = _read_json_artifact(case_result, "model_definition_json")

    stage_timings: dict[str, object] = (
        dict(case_result.stage_timings) if case_result is not None else {}
    )
    evaluation_stage_timings: dict[str, object] = (
        dict(case_result.evaluation_result.stage_timings)
        if case_result is not None
        else {}
    )
    metrics = _as_mapping(metrics_payload.get("metrics")) if metrics_payload else {}
    warnings = diagnostics_payload.get("warnings")
    events = diagnostics_payload.get("events")

    return {
        "schema_version": _BENCHMARK_SCHEMA_VERSION,
        "artifact_type": "alpha_lab_model_factor_benchmark_record",
        "workflow": "real_case_model_factor_benchmark",
        "run_id": run_id,
        "status": status,
        "note": note,
        "started_at": started_at,
        "finished_at": finished_at,
        "wall_seconds": round(float(wall_seconds), 6),
        "config_hash": _stable_hash(config_snapshot),
        "config": config_snapshot,
        "git": git_snapshot,
        "memory": {
            **memory_summary,
            "resource_limits": resource_limits,
            "profile_path": str(memory_profile_path) if memory_profile_path is not None else None,
        },
        "timings": {
            "stage_timings": stage_timings,
            "evaluation_stage_timings": evaluation_stage_timings,
        },
        "training": _training_summary(
            stage_timings=stage_timings,
            model_definition_payload=model_definition_payload,
        ),
        "cache_lineage": _cache_lineage(diagnostics_payload),
        "metrics": _selected_metrics(metrics),
        "artifacts": _artifact_paths(case_result),
        "diagnostics": {
            "warning_count": len(warnings) if isinstance(warnings, list) else 0,
            "event_count": len(events) if isinstance(events, list) else 0,
            "run_manifest_status": run_manifest_payload.get("status")
            if run_manifest_payload
            else None,
        },
        "error": error,
    }


def _build_config_snapshot(
    options: ModelFactorBenchmarkOptions,
    *,
    spec_path: Path,
    spec: ModelFactorCaseSpec,
) -> dict[str, object]:
    return {
        "spec_path": str(spec_path),
        "spec": asdict(spec),
        "runtime": {
            "output_root_dir": (
                str(Path(options.output_root_dir).resolve())
                if options.output_root_dir is not None
                else None
            ),
            "cache_root_dir": (
                str(Path(options.cache_root_dir).expanduser().resolve())
                if options.cache_root_dir is not None
                else None
            ),
            "evaluation_profile": options.evaluation_profile,
            "use_preparation_cache": bool(options.use_preparation_cache),
            "screening_retrain_every_n_dates": options.screening_retrain_every_n_dates,
            "vault_root": (
                str(Path(options.vault_root).resolve())
                if options.vault_root is not None
                else None
            ),
            "vault_export_mode": options.vault_export_mode,
            "memory_sample_interval_seconds": (
                float(options.memory_sample_interval_seconds)
            ),
            "memory_limit_gb": options.memory_limit_gb,
            "memory_profile": bool(options.memory_profile),
            "tracemalloc_profile": bool(options.tracemalloc_profile),
            "tracemalloc_nframes": int(options.tracemalloc_nframes),
        },
    }


def _apply_memory_limit(memory_limit_gb: float | None) -> dict[str, object]:
    if resource is None:
        return {
            "address_space_limit_enabled": False,
            "requested_memory_limit_gb": memory_limit_gb,
            "previous_soft_bytes": None,
            "previous_hard_bytes": None,
            "applied_soft_bytes": None,
            "unsupported_reason": "python_resource_module_unavailable",
        }
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    payload: dict[str, object] = {
        "address_space_limit_enabled": memory_limit_gb is not None,
        "requested_memory_limit_gb": memory_limit_gb,
        "previous_soft_bytes": _rlimit_value(soft),
        "previous_hard_bytes": _rlimit_value(hard),
    }
    if memory_limit_gb is None:
        payload["applied_soft_bytes"] = _rlimit_value(soft)
        return payload
    requested_bytes = int(float(memory_limit_gb) * 1024**3)
    applied_soft = requested_bytes
    if hard != resource.RLIM_INFINITY:
        applied_soft = min(applied_soft, int(hard))
    resource.setrlimit(resource.RLIMIT_AS, (applied_soft, hard))
    payload["applied_soft_bytes"] = applied_soft
    payload["applied_memory_limit_gb"] = round(applied_soft / 1024**3, 6)
    return payload


def _rlimit_value(value: int) -> int | str:
    if resource is not None and value == resource.RLIM_INFINITY:
        return "unlimited"
    return int(value)


def _resolve_benchmark_output_dir(
    options: ModelFactorBenchmarkOptions,
    *,
    spec: ModelFactorCaseSpec,
) -> Path:
    if options.benchmark_output_dir is not None:
        return Path(options.benchmark_output_dir).resolve()
    if options.output_root_dir is not None:
        return Path(options.output_root_dir).resolve() / "_benchmark_records"
    return Path(spec.output.root_dir).resolve() / "_benchmark_records"


def _default_run_id(spec: ModelFactorCaseSpec) -> str:
    stamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_{spec.name}_{uuid4().hex[:8]}"


def _utc_now_iso() -> str:
    return datetime.now(tz=UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _stable_hash(payload: object) -> str:
    text = json.dumps(payload, ensure_ascii=True, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _write_record(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_json_artifact(
    case_result: ModelFactorCaseRunResult | None,
    key: str,
    *,
    failure_artifact_paths: Mapping[str, str] | None = None,
) -> dict[str, object]:
    path: Path | None = None
    if case_result is not None:
        raw_path = case_result.artifact_paths.get(key)
        path = Path(str(raw_path)) if raw_path is not None else None
    elif failure_artifact_paths is not None:
        raw_path = failure_artifact_paths.get(key)
        path = Path(str(raw_path)) if raw_path is not None else None
    if path is None or not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _failure_artifact_paths(exc: Exception) -> dict[str, str]:
    raw_paths = getattr(exc, "model_lab_artifact_paths", {})
    if not isinstance(raw_paths, Mapping):
        return {}
    return {
        str(key): str(value)
        for key, value in raw_paths.items()
        if value is not None
    }


def _as_mapping(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _selected_metrics(metrics: dict[str, object]) -> dict[str, object]:
    keys = (
        "research_evaluation_profile",
        "model_family",
        "feature_count",
        "target_horizon",
        "mean_ic",
        "mean_rank_ic",
        "ic_ir",
        "rank_ic_ir",
        "mean_mutual_information",
        "campaign_triage",
        "promotion_decision",
        "portfolio_validation_status",
        "portfolio_validation_recommendation",
    )
    return {key: metrics.get(key) for key in keys if key in metrics}


def _training_summary(
    *,
    stage_timings: Mapping[str, object],
    model_definition_payload: dict[str, object],
) -> dict[str, object]:
    outcome = _as_mapping(model_definition_payload.get("model_selection_outcome"))
    return {
        "fit_count": stage_timings.get("model_fit_count", 0.0),
        "fit_total_seconds": stage_timings.get("model_fit", 0.0),
        "fit_mean_seconds": stage_timings.get("model_fit_mean", 0.0),
        "fit_p95_seconds": stage_timings.get("model_fit_p95", 0.0),
        "predict_count": stage_timings.get("predict_count", 0.0),
        "predict_total_seconds": stage_timings.get("predict", 0.0),
        "model_selection_count": stage_timings.get("model_selection_count", 0.0),
        "n_fit_events": outcome.get("n_fit_events"),
        "n_reuse_events": outcome.get("n_reuse_events"),
    }


def _cache_lineage(diagnostics_payload: dict[str, object]) -> dict[str, object]:
    stages = diagnostics_payload.get("stages")
    if not isinstance(stages, list):
        return {}
    by_name: dict[str, list[dict[str, object]]] = {}
    for item in stages:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "")
        if not name:
            continue
        by_name.setdefault(name, []).append(item)

    data_load = _stage_result(by_name, "data_load")
    feature_validate = _stage_result(by_name, "feature_validate")
    target_build = _stage_result(by_name, "target_build")
    return {
        "preparation_cache_enabled": data_load.get("preparation_cache_enabled"),
        "preparation_cache_hit": data_load.get("preparation_cache_hit"),
        "preparation_cache_key": data_load.get("preparation_cache_key"),
        "preparation_cache_dir": data_load.get("preparation_cache_dir"),
        "feature_validate_cache_hit": feature_validate.get("cache_hit"),
        "target_build_cache_hit": target_build.get("cache_hit"),
    }


def _stage_result(
    stages_by_name: dict[str, list[dict[str, object]]],
    name: str,
) -> dict[str, object]:
    items = stages_by_name.get(name, [])
    if not items:
        return {}
    result = items[-1].get("result")
    return result if isinstance(result, dict) else {}


def _artifact_paths(
    case_result: ModelFactorCaseRunResult | None,
) -> dict[str, str]:
    if case_result is None:
        return {}
    return {
        key: str(Path(str(value)).resolve())
        for key, value in sorted(case_result.artifact_paths.items())
    }


def _git_snapshot(cwd: Path) -> dict[str, object]:
    root = _git_output(["git", "rev-parse", "--show-toplevel"], cwd=cwd)
    if root is None:
        return {"available": False}
    commit = _git_output(["git", "rev-parse", "HEAD"], cwd=Path(root))
    status = _git_output(["git", "status", "--short"], cwd=Path(root))
    return {
        "available": True,
        "root": root,
        "commit": commit,
        "dirty": bool(status),
        "status_short": status.splitlines() if status else [],
    }


def _git_output(command: list[str], *, cwd: Path) -> str | None:
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _write_memory_profile_artifact(
    *,
    record_dir: Path,
    run_id: str,
    enabled: bool,
    sampler: _ProcessMemorySampler,
    stage_memory_tracker: _StageMemoryTracker,
    started_at: str,
    finished_at: str,
) -> Path | None:
    if not enabled:
        return None
    path = record_dir / f"{run_id}_memory_profile.json"
    payload: dict[str, object] = {
        "schema_version": _MEMORY_PROFILE_SCHEMA_VERSION,
        "artifact_type": "alpha_lab_model_factor_memory_profile",
        "run_id": run_id,
        "started_at": started_at,
        "finished_at": finished_at,
        "timeline": sampler.timeline(),
        "stage_peaks": sampler.stage_peaks(),
        "tracemalloc": stage_memory_tracker.tracemalloc_payload(),
    }
    _write_record(path, payload)
    return path


class _StageMemoryTracker:
    def __init__(
        self,
        *,
        tracemalloc_enabled: bool,
        tracemalloc_nframes: int,
    ) -> None:
        self._lock = threading.Lock()
        self._stage_stack: list[tuple[int, str]] = []
        self._tracemalloc_enabled = bool(tracemalloc_enabled)
        self._tracemalloc_records: list[dict[str, object]] = []
        self._start_snapshots: dict[int, tracemalloc.Snapshot] = {}
        if self._tracemalloc_enabled:
            tracemalloc.start(int(tracemalloc_nframes))

    def stop(self) -> None:
        if self._tracemalloc_enabled and tracemalloc.is_tracing():
            tracemalloc.stop()

    def current_stage(self) -> str:
        with self._lock:
            return self._stage_stack[-1][1] if self._stage_stack else "idle"

    def stage_lifecycle_callback(self, event: str, stage_name: str, stage_id: int) -> None:
        if event == "begin":
            self._begin_stage(stage_name=stage_name, stage_id=stage_id)
        elif event == "end":
            self._end_stage(stage_name=stage_name, stage_id=stage_id)

    def tracemalloc_payload(self) -> dict[str, object]:
        return {
            "enabled": self._tracemalloc_enabled,
            "records": list(self._tracemalloc_records),
        }

    def _begin_stage(self, *, stage_name: str, stage_id: int) -> None:
        with self._lock:
            self._stage_stack.append((int(stage_id), str(stage_name)))
        if self._should_trace_stage(stage_name) and tracemalloc.is_tracing():
            try:
                self._start_snapshots[int(stage_id)] = tracemalloc.take_snapshot()
            except Exception:
                return

    def _end_stage(self, *, stage_name: str, stage_id: int) -> None:
        snapshot = self._start_snapshots.pop(int(stage_id), None)
        if snapshot is not None and tracemalloc.is_tracing():
            try:
                end_snapshot = tracemalloc.take_snapshot()
                stats = end_snapshot.compare_to(snapshot, "lineno")[:20]
                self._tracemalloc_records.append(
                    {
                        "stage": str(stage_name),
                        "stage_id": int(stage_id),
                        "top_allocators": [
                            _tracemalloc_stat_payload(stat) for stat in stats
                        ],
                    }
                )
            except Exception:
                pass
        with self._lock:
            self._stage_stack = [
                item for item in self._stage_stack if item[0] != int(stage_id)
            ]

    def _should_trace_stage(self, stage_name: str) -> bool:
        return self._tracemalloc_enabled and str(stage_name) in _TRACEMALLOC_STAGE_NAMES


def _tracemalloc_stat_payload(stat: tracemalloc.StatisticDiff) -> dict[str, object]:
    frame = stat.traceback[0] if stat.traceback else None
    return {
        "size_diff_kb": round(float(stat.size_diff) / 1024.0, 3),
        "count_diff": int(stat.count_diff),
        "file": frame.filename if frame is not None else None,
        "line": frame.lineno if frame is not None else None,
        "traceback": [
            {"file": frame_item.filename, "line": frame_item.lineno}
            for frame_item in stat.traceback
        ],
    }


class _ProcessMemorySampler:
    def __init__(
        self,
        *,
        interval_seconds: float,
        capture_timeline: bool,
        current_stage_provider: Callable[[], str],
    ) -> None:
        self._interval_seconds = float(interval_seconds)
        self._capture_timeline = bool(capture_timeline)
        self._current_stage_provider = current_stage_provider
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._peak_rss_kb = 0
        self._peak_swap_kb = 0
        self._samples = 0
        self._started_perf_counter: float | None = None
        self._timeline: list[dict[str, object]] = []

    def start(self) -> None:
        self._started_perf_counter = time.perf_counter()
        self._sample()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(self._interval_seconds * 2.0, 1.0))
        self._sample()

    def summary(self) -> dict[str, object]:
        rusage_maxrss_kb = (
            int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            if resource is not None
            else 0
        )
        peak_rss_kb = max(self._peak_rss_kb, rusage_maxrss_kb)
        return {
            "peak_rss_kb": peak_rss_kb,
            "peak_rss_mb": round(peak_rss_kb / 1024.0, 3),
            "peak_swap_kb": self._peak_swap_kb,
            "peak_swap_mb": round(self._peak_swap_kb / 1024.0, 3),
            "sample_count": self._samples,
            "sample_interval_seconds": self._interval_seconds,
            "rusage_maxrss_kb": rusage_maxrss_kb,
        }

    def timeline(self) -> list[dict[str, object]]:
        return list(self._timeline)

    def stage_peaks(self) -> dict[str, dict[str, object]]:
        peaks: dict[str, dict[str, object]] = {}
        for row in self._timeline:
            stage = str(row.get("stage") or "unknown")
            rss_kb = _object_to_int(row.get("rss_kb"), default=0)
            swap_kb = _object_to_int(row.get("swap_kb"), default=0)
            current = peaks.setdefault(
                stage,
                {
                    "peak_rss_kb": 0,
                    "peak_rss_mb": 0.0,
                    "peak_swap_kb": 0,
                    "peak_swap_mb": 0.0,
                    "sample_count": 0,
                },
            )
            current["sample_count"] = (
                _object_to_int(current.get("sample_count"), default=0) + 1
            )
            if rss_kb > _object_to_int(current.get("peak_rss_kb"), default=0):
                current["peak_rss_kb"] = rss_kb
                current["peak_rss_mb"] = round(rss_kb / 1024.0, 3)
            if swap_kb > _object_to_int(current.get("peak_swap_kb"), default=0):
                current["peak_swap_kb"] = swap_kb
                current["peak_swap_mb"] = round(swap_kb / 1024.0, 3)
        return peaks

    def _run(self) -> None:
        while not self._stop_event.wait(self._interval_seconds):
            self._sample()

    def _sample(self) -> None:
        rss_kb = 0
        swap_kb = 0
        try:
            for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
                if line.startswith("VmRSS:"):
                    rss_kb = _status_kb_value(line)
                elif line.startswith("VmSwap:"):
                    swap_kb = _status_kb_value(line)
        except Exception:
            return
        self._peak_rss_kb = max(self._peak_rss_kb, rss_kb)
        self._peak_swap_kb = max(self._peak_swap_kb, swap_kb)
        self._samples += 1
        if self._capture_timeline:
            started = self._started_perf_counter or time.perf_counter()
            self._timeline.append(
                {
                    "t_rel_seconds": round(time.perf_counter() - started, 3),
                    "stage": self._safe_current_stage(),
                    "rss_kb": rss_kb,
                    "rss_mb": round(rss_kb / 1024.0, 3),
                    "swap_kb": swap_kb,
                    "swap_mb": round(swap_kb / 1024.0, 3),
                }
            )

    def _safe_current_stage(self) -> str:
        try:
            return str(self._current_stage_provider())
        except Exception:
            return "unknown"


def _status_kb_value(line: str) -> int:
    parts = line.split()
    if len(parts) < 2:
        return 0
    try:
        return int(parts[1])
    except ValueError:
        return 0


def _object_to_int(value: object, *, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default
