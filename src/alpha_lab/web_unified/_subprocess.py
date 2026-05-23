"""Model-lab subprocess orchestration helpers.

This module owns the contract between the unified web service and the
out-of-process ``python -m alpha_lab.real_cases.model_factor.cli run`` worker:

- command construction (``_build_model_lab_subprocess_command``)
- environment shaping (``_build_model_lab_subprocess_env``, thread pinning)
- output / cache directory resolution (the ``_resolve_*`` helpers; the
  ``_model_factor_shared_cache`` invariant lives here)
- result inspection (artifact discovery, RSS parsing, error annotation)
- failure formatting (``_format_*``)

Extracted from ``web_unified/__init__.py`` to keep subprocess plumbing
isolated from request handling and run-store state machines.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import sys
from collections.abc import Mapping
from pathlib import Path

from alpha_lab.real_cases.model_factor.spec import load_model_factor_case_spec
from alpha_lab.real_cases.single_factor.pipeline import (
    SingleFactorBatchParallelConfig,
)
from alpha_lab.real_cases.single_factor.spec import (
    SingleFactorCaseSpec,
    load_single_factor_case_spec,
)
from alpha_lab.web_unified._models import _RunTask
from alpha_lab.web_unified._utils import _coerce_finite_or_text, _safe_slug

# ---------------------------------------------------------------------------
# Tunable knobs
# ---------------------------------------------------------------------------

_FRONTEND_BATCH_MAX_WORKERS: int = 4
_FRONTEND_BATCH_FACTORS_PER_WORKER: int = 2
_MODEL_LAB_BATCH_MAX_WORKERS: int = 3
_MODEL_LAB_BATCH_DEFAULT_WORKERS: int = 1

# Artifact filename fallbacks (used when the manifest is missing or partial).
_ARTIFACT_FALLBACK_FILENAMES: dict[str, str] = {
    "research_tearsheet": "research_tearsheet.json",
    "research_tearsheet_pdf": "research_tearsheet.pdf",
    "metrics": "metrics.json",
    "summary": "summary.md",
    "case_report": "case_report.md",
}


def _parse_positive_int_env(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return None
    try:
        value = int(str(raw).strip())
    except ValueError:
        return None
    return value if value > 0 else None


# ---------------------------------------------------------------------------
# Batch sizing
# ---------------------------------------------------------------------------


def _build_frontend_batch_parallel_config(
    n_tasks: int,
) -> SingleFactorBatchParallelConfig:
    worker_slots = max(1, min(_FRONTEND_BATCH_MAX_WORKERS, n_tasks))
    return SingleFactorBatchParallelConfig(
        mode="process",
        max_workers=worker_slots,
        factors_per_worker=_FRONTEND_BATCH_FACTORS_PER_WORKER,
    )


def _build_model_lab_batch_worker_count(n_tasks: int) -> int:
    if n_tasks <= 0:
        return 1
    requested_workers = _parse_positive_int_env("ALPHA_LAB_MODEL_LAB_MAX_WORKERS")
    configured_workers = (
        requested_workers
        if requested_workers is not None
        else _MODEL_LAB_BATCH_DEFAULT_WORKERS
    )
    cpu_count = max(1, os.cpu_count() or 1)
    # Model-factor runs are memory-heavy and many estimators are internally threaded.
    # Default to serial execution; opt in with ALPHA_LAB_MODEL_LAB_MAX_WORKERS.
    baseline_cap = 1 if cpu_count <= 1 else max(2, cpu_count // 2)
    worker_cap = min(_MODEL_LAB_BATCH_MAX_WORKERS, baseline_cap, configured_workers)
    return max(1, min(worker_cap, n_tasks))


# ---------------------------------------------------------------------------
# Subprocess command + env
# ---------------------------------------------------------------------------


def _build_model_lab_subprocess_command(
    *,
    task: _RunTask,
    spec_path: Path,
) -> list[str]:
    output_root_dir, _case_dir_name = _resolve_model_factor_web_output_parts(
        task,
        spec_path=spec_path,
    )
    cache_root_dir = _resolve_model_factor_web_cache_root_dir(
        task,
        spec_path=spec_path,
    )
    cmd = [
        sys.executable,
        "-m",
        "alpha_lab.real_cases.model_factor.cli",
        "run",
        str(spec_path),
        "--evaluation-profile",
        task.evaluation_profile,
        "--vault-export-mode",
        "skip",
        "--output-root-dir",
        str(output_root_dir),
        "--cache-root-dir",
        str(cache_root_dir),
    ]
    if task.screening_retrain_every_n_dates is not None:
        cmd.extend(
            [
                "--screening-retrain-every-n-dates",
                str(task.screening_retrain_every_n_dates),
            ]
        )
    if task.draft_model_candidate_path:
        cmd.extend(["--draft-model-candidate", task.draft_model_candidate_path])
    return cmd


def _build_single_factor_subprocess_command(
    *,
    task: _RunTask,
    spec_path: Path,
) -> list[str]:
    output_root_dir = _resolve_single_factor_web_output_root_dir(
        task,
        spec_path=spec_path,
    )
    cmd = [
        sys.executable,
        "-m",
        "alpha_lab.real_cases.single_factor.cli",
        "run",
        str(spec_path),
        "--evaluation-profile",
        task.evaluation_profile,
        "--vault-export-mode",
        "skip",
        "--output-root-dir",
        str(output_root_dir),
    ]
    if task.render_report:
        cmd.extend(["--render-report", "--render-overwrite"])
    return cmd


def _build_model_lab_subprocess_env() -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("ALPHA_LAB_MODEL_LAB_CHILD", "1")
    # __file__ is src/alpha_lab/web_unified/_subprocess.py; parents[2] == src/
    source_root = str(Path(__file__).resolve().parents[2])
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        source_root
        if not existing_pythonpath
        else os.pathsep.join([source_root, existing_pythonpath])
    )
    thread_count = str(_parse_positive_int_env("ALPHA_LAB_MODEL_LAB_THREADS") or 1)
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        env[key] = thread_count
    return env


def _build_single_factor_subprocess_env() -> dict[str, str]:
    env = _build_model_lab_subprocess_env()
    env["ALPHA_LAB_SINGLE_FACTOR_CHILD"] = "1"
    return env


def _wrap_command_with_time(cmd: list[str]) -> list[str]:
    time_bin = shutil.which("time")
    if os.name != "nt" and time_bin:
        return [time_bin, "-v", *cmd]
    return list(cmd)


# ---------------------------------------------------------------------------
# Output/cache directory resolution
# ---------------------------------------------------------------------------


def _resolve_model_factor_web_cache_root_dir(
    task: _RunTask,
    *,
    spec_path: Path | None = None,
) -> Path:
    """Resolve a shared model-factor cache root for web runs.

    Returns ``<base_root>/_model_factor_shared_cache`` so the dataset cache
    sits beside ``_web_runs/`` and is reused across web submissions instead of
    being duplicated under every per-run output directory.
    """

    resolved_spec_path = Path(spec_path or task.spec_path).expanduser().resolve()
    try:
        spec = load_model_factor_case_spec(resolved_spec_path)
        base_root = (
            Path(task.output_root_dir).expanduser().resolve()
            if task.output_root_dir is not None
            else Path(spec.output.root_dir).expanduser().resolve()
        )
    except Exception:
        base_root = Path(task.output_root_dir or "__default_output_root__").expanduser().resolve()

    return base_root / "_model_factor_shared_cache"


def _resolve_model_factor_web_output_parts(
    task: _RunTask,
    *,
    spec_path: Path | None = None,
) -> tuple[Path, str]:
    resolved_spec_path = Path(spec_path or task.spec_path).expanduser().resolve()
    try:
        spec = load_model_factor_case_spec(resolved_spec_path)
        base_root = (
            Path(task.output_root_dir).expanduser().resolve()
            if task.output_root_dir is not None
            else Path(spec.output.root_dir).expanduser().resolve()
        )
        case_dir_name = spec.name
    except Exception:
        base_root = Path(task.output_root_dir or "__default_output_root__").expanduser().resolve()
        case_dir_name = _safe_slug(task.case_name)

    return base_root / "_web_runs" / _safe_slug(task.run_id), case_dir_name


def _resolve_single_factor_web_output_root_dir(
    task: _RunTask,
    *,
    spec: SingleFactorCaseSpec | None = None,
    spec_path: Path | None = None,
) -> Path:
    try:
        resolved_spec = spec
        if resolved_spec is None:
            resolved_spec_path = Path(spec_path or task.spec_path).expanduser().resolve()
            resolved_spec = load_single_factor_case_spec(resolved_spec_path)
        base_root = (
            Path(task.output_root_dir).expanduser().resolve()
            if task.output_root_dir is not None
            else Path(resolved_spec.output.root_dir).expanduser().resolve()
        )
    except Exception:
        base_root = Path(task.output_root_dir or "__default_output_root__").expanduser().resolve()
    return base_root / "_web_runs" / _safe_slug(task.run_id)


# ---------------------------------------------------------------------------
# Subprocess output inspection
# ---------------------------------------------------------------------------


def _write_json_file(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_text_tail(path: Path, *, max_bytes: int = 16 * 1024) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - max_bytes))
            return handle.read(max_bytes).decode("utf-8", errors="replace").strip()
    except OSError:
        return ""


def _parse_time_peak_rss_kb(stderr_path: Path) -> int | None:
    tail = _read_text_tail(stderr_path, max_bytes=64 * 1024)
    match = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", tail)
    if match is None:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _load_model_factor_artifact_paths_from_manifest(output_dir: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    manifest_path = output_dir / "run_manifest.json"
    if manifest_path.exists():
        paths["run_manifest"] = manifest_path
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
        outputs = payload.get("outputs") if isinstance(payload, dict) else None
        if isinstance(outputs, dict):
            for key, value in outputs.items():
                key_text = str(key or "").strip()
                value_text = _coerce_finite_or_text(value)
                if not key_text or not value_text:
                    continue
                candidate = Path(value_text).expanduser()
                if not candidate.is_absolute():
                    candidate = output_dir / candidate
                paths[key_text] = candidate.resolve()

    fallback_filenames = {
        **_ARTIFACT_FALLBACK_FILENAMES,
        "run_manifest": "run_manifest.json",
        "diagnostics": "diagnostics.json",
        "training_log": "training_log.csv",
        "feature_importance": "feature_importance.csv",
        "model_definition_json": "model_definition.json",
        "feature_manifest_json": "feature_manifest.json",
        "experiment_card": "experiment_card.md",
    }
    for key, filename in fallback_filenames.items():
        candidate = output_dir / filename
        if candidate.exists():
            paths.setdefault(key, candidate.resolve())
    return paths


def _load_single_factor_artifact_paths_from_manifest(output_dir: Path) -> dict[str, Path]:
    return _load_model_factor_artifact_paths_from_manifest(output_dir)


def _annotate_exception_with_model_lab_subprocess_artifacts(
    exc: Exception,
    *,
    output_dir: Path,
    artifact_paths: Mapping[str, Path],
) -> None:
    existing_raw = getattr(exc, "model_lab_artifact_paths", None)
    existing: dict[str, str] = {}
    if isinstance(existing_raw, dict):
        existing = {str(key): str(value) for key, value in existing_raw.items()}
    try:
        exc.model_lab_output_dir = str(output_dir)  # type: ignore[attr-defined]
        exc.model_lab_artifact_paths = {  # type: ignore[attr-defined]
            **existing,
            **{key: str(path) for key, path in artifact_paths.items()},
        }
    except Exception:  # noqa: BLE001
        return


# ---------------------------------------------------------------------------
# Failure formatting
# ---------------------------------------------------------------------------


def _model_lab_subprocess_failure_hint(
    *,
    returncode: int | None,
    stderr_tail: str,
    stdout_tail: str,
) -> str:
    combined_tail = f"{stderr_tail}\n{stdout_tail}".lower()
    if returncode in {137, -9} or "killed" in combined_tail or "out of memory" in combined_tail:
        return (
            "模型因子子进程很可能被 OOM killer 杀掉。优先降低数据窗口/特征数量，"
            "保持 ALPHA_LAB_MODEL_LAB_MAX_WORKERS=1，或继续提高 WSL memory/swap。"
        )
    return (
        "模型因子子进程已失败，但 web 前端进程仍保持存活。请打开 subprocess_stderr、"
        "subprocess_stdout 和 diagnostics artifact 查看具体异常。"
    )


def _format_model_lab_subprocess_failure(
    *,
    command: list[str],
    returncode: int | None,
    stdout_tail: str,
    stderr_tail: str,
    elapsed_seconds: float,
    peak_rss_kb: int | None,
) -> str:
    lines = [
        "model-factor subprocess failed",
        f"returncode: {returncode}",
        f"elapsed_seconds: {elapsed_seconds}",
        f"peak_rss_kb: {peak_rss_kb}",
        f"command: {_format_shell_command(command)}",
    ]
    if stderr_tail:
        lines.extend(["", "stderr_tail:", stderr_tail])
    if stdout_tail:
        lines.extend(["", "stdout_tail:", stdout_tail])
    return "\n".join(lines).rstrip()


def _format_shell_command(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def _format_run_error_text(
    *,
    stage: str | None,
    error_type: str,
    error_message: str,
    error_hint: str,
    traceback_text: str,
) -> str:
    lines = [
        f"stage: {stage or 'unknown'}",
        f"type: {error_type}",
        f"message: {error_message}",
        f"hint: {error_hint}",
        "",
        "traceback:",
        traceback_text.rstrip(),
    ]
    return "\n".join(lines).rstrip()


__all__ = [
    "_ARTIFACT_FALLBACK_FILENAMES",
    "_FRONTEND_BATCH_FACTORS_PER_WORKER",
    "_FRONTEND_BATCH_MAX_WORKERS",
    "_MODEL_LAB_BATCH_DEFAULT_WORKERS",
    "_MODEL_LAB_BATCH_MAX_WORKERS",
    "_annotate_exception_with_model_lab_subprocess_artifacts",
    "_build_frontend_batch_parallel_config",
    "_build_model_lab_batch_worker_count",
    "_build_model_lab_subprocess_command",
    "_build_model_lab_subprocess_env",
    "_format_model_lab_subprocess_failure",
    "_format_run_error_text",
    "_format_shell_command",
    "_load_model_factor_artifact_paths_from_manifest",
    "_model_lab_subprocess_failure_hint",
    "_parse_positive_int_env",
    "_parse_time_peak_rss_kb",
    "_read_text_tail",
    "_resolve_model_factor_web_cache_root_dir",
    "_resolve_model_factor_web_output_parts",
    "_resolve_single_factor_web_output_root_dir",
    "_wrap_command_with_time",
    "_write_json_file",
]
