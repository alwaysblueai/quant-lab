"""Unified Research Frontend — single local HTTP server integrating
Knowledge Ops, Bridge Workspace, Validation Console, and Writeback Review.

Evolved from web_cockpit.py; provides the ``start_unified_server`` entry-point.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import re
import threading
import traceback
import uuid
import webbrowser
from collections.abc import Mapping
from csv import DictReader
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Literal
from urllib.parse import parse_qs, unquote, urlparse

from alpha_lab.exceptions import AlphaLabConfigError, AlphaLabDataError, AlphaLabExperimentError
from alpha_lab.factor_recipe import factor_registry
from alpha_lab.real_cases.single_factor.pipeline import (
    SingleFactorBatchParallelConfig,
    SingleFactorInputBundle,
    load_standard_inputs,
    run_single_factor_case,
    run_single_factor_cases,
)
from alpha_lab.real_cases.single_factor.spec import (
    SingleFactorCaseSpec,
    load_single_factor_case_spec,
)
from alpha_lab.reporting.renderers import write_case_report
from alpha_lab.research_bridge.categories import get_category_profile, list_categories
from alpha_lab.research_bridge.graph_view import VaultGraph
from alpha_lab.research_bridge.models import (
    load_project_config,
    load_yaml_document,
    save_project_config,
)
from alpha_lab.research_bridge.preflight import run_preflight
from alpha_lab.research_bridge.service import (
    PROJECTS_DIRNAME,
    apply_writeback,
    init_project,
    refresh_project_pack,
    scaffold_case,
    start_round,
    summarize_run,
)
from alpha_lab.research_bridge.service import (
    explore_idea as bridge_explore_idea,
)
from alpha_lab.research_evaluation_config import (
    AVAILABLE_RESEARCH_EVALUATION_PROFILES,
    CAMPAIGN_PROFILE_COMPARE_DEFAULTS,
    RESEARCH_EVALUATION_PROFILE_LABELS,
)
from alpha_lab.vault_export import resolve_vault_root

RunStatus = Literal["queued", "running", "succeeded", "failed"]

# Maximum bytes read from any text file served to the browser.
# Prevents the server from reading/sending huge artifacts that would freeze the UI.
_MAX_TEXT_BYTES: int = 512 * 1024  # 512 KB
_MAX_REPORT_TEXT_BYTES: int = 8 * 1024 * 1024  # 8 MB for full tearsheet JSON
_PROJECT_DOC_PREVIEW_BYTES: int = 128 * 1024  # 128 KB for project snapshot docs

# Maximum request body size accepted from the browser.
_MAX_REQUEST_BODY_BYTES: int = 2 * 1024 * 1024  # 2 MB
_FRONTEND_BATCH_WINDOW_SECONDS: float = 0.20
_FRONTEND_BATCH_MAX_WORKERS: int = 4
_FRONTEND_BATCH_FACTORS_PER_WORKER: int = 2
_FRONTEND_INPUT_BUNDLE_CACHE_MAX_ITEMS: int = 8
_RUN_OVERVIEW_MAX_CSV_ROWS: int = 20000

_RUN_SUMMARY_COMPACT_KEYS: tuple[str, ...] = (
    "research_evaluation_profile",
    "factor_name",
    "factor_verdict",
    "mean_rank_ic",
    "rank_ic_ir",
    "ic_ir",
    "ic_positive_rate",
    "rank_ic_positive_rate",
    "group_monotonicity_summary",
    "group_spread_summary",
    "ic_decay_half_life_summary",
    "ic_decay_retention_5_over_1",
    "ic_half_life_summary",
    "ic_half_life_horizon",
    "mean_long_short_turnover",
    "cost_aware_long_short_sharpe",
    "cost_aware_long_short_ir",
    "long_short_ir",
    "ic_t_stat",
    "max_drawdown",
    "ls_max_drawdown",
    "coverage_summary",
    "n_dates_used",
    "mean_eval_assets_per_date",
    "eval_coverage_ratio_mean",
)

_ARTIFACT_FALLBACK_FILENAMES: dict[str, str] = {
    "research_tearsheet": "research_tearsheet.json",
    "research_tearsheet_pdf": "research_tearsheet.pdf",
    "metrics": "metrics.json",
    "summary": "summary.md",
}

# ---------------------------------------------------------------------------
# Server entry-point
# ---------------------------------------------------------------------------


def start_unified_server(
    *,
    host: str = "127.0.0.1",
    port: int = 8766,
    workspace_root: str | Path | None = None,
    vault_root: str | Path | None = None,
    open_browser: bool = True,
) -> None:
    resolved_workspace = (
        Path.cwd().resolve() if workspace_root is None else Path(workspace_root).resolve()
    )
    resolved_vault = resolve_vault_root(vault_root)
    if resolved_vault is None:
        raise AlphaLabConfigError(
            "vault root is unresolved; pass --vault-root or set OBSIDIAN_VAULT_PATH"
        )
    if not resolved_vault.exists() or not resolved_vault.is_dir():
        raise AlphaLabConfigError(
            f"vault root does not exist or is not a directory: {resolved_vault}"
        )

    service = _UnifiedService(vault_root=resolved_vault, workspace_root=resolved_workspace)

    class _Handler(_UnifiedRequestHandler):
        svc = service

    server = ThreadingHTTPServer((host, port), _Handler)
    url = f"http://{host}:{port}/"
    print("")
    print("  Workflow : unified-research-frontend")
    print("  Status   : running")
    print(f"  URL      : {url}")
    print("  Hint     : press Ctrl+C to stop")
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("")
        print("  Workflow : unified-research-frontend")
        print("  Status   : stopped")
    finally:
        server.server_close()


# ---------------------------------------------------------------------------
# Run store
# ---------------------------------------------------------------------------


def _compact_metrics_summary(summary: Mapping[str, object]) -> dict[str, object]:
    if not summary:
        return {}
    compact: dict[str, object] = {}
    for key in _RUN_SUMMARY_COMPACT_KEYS:
        if key in summary:
            compact[key] = summary[key]
    return compact


@dataclass
class _RunRecord:
    run_id: str
    project_slug: str
    case_name: str
    round_id: str | None
    spec_path: str
    submitted_at_utc: str
    evaluation_profile: str
    output_root_dir: str | None
    render_report: bool
    status: RunStatus = "queued"
    started_at_utc: str | None = None
    finished_at_utc: str | None = None
    updated_at_utc: str | None = None
    output_dir: str | None = None
    progress_percent: int | None = None
    progress_message: str | None = None
    progress_events: list[dict[str, object]] = field(default_factory=list)
    artifact_paths: dict[str, str] = field(default_factory=dict)
    summary: dict[str, object] = field(default_factory=dict)
    summarize_feedback_path: str | None = None
    summarize_draft_path: str | None = None
    summarize_state_patch_path: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    error_hint: str | None = None
    error: str | None = None

    def clone(self) -> _RunRecord:
        return _RunRecord(
            run_id=self.run_id,
            project_slug=self.project_slug,
            case_name=self.case_name,
            round_id=self.round_id,
            spec_path=self.spec_path,
            submitted_at_utc=self.submitted_at_utc,
            evaluation_profile=self.evaluation_profile,
            output_root_dir=self.output_root_dir,
            render_report=self.render_report,
            status=self.status,
            started_at_utc=self.started_at_utc,
            finished_at_utc=self.finished_at_utc,
            updated_at_utc=self.updated_at_utc,
            output_dir=self.output_dir,
            progress_percent=self.progress_percent,
            progress_message=self.progress_message,
            progress_events=[dict(item) for item in self.progress_events],
            artifact_paths=dict(self.artifact_paths),
            summary=dict(self.summary),
            summarize_feedback_path=self.summarize_feedback_path,
            summarize_draft_path=self.summarize_draft_path,
            summarize_state_patch_path=self.summarize_state_patch_path,
            error_type=self.error_type,
            error_message=self.error_message,
            error_hint=self.error_hint,
            error=self.error,
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "project_slug": self.project_slug,
            "case_name": self.case_name,
            "round_id": self.round_id,
            "spec_path": self.spec_path,
            "submitted_at_utc": self.submitted_at_utc,
            "evaluation_profile": self.evaluation_profile,
            "output_root_dir": self.output_root_dir,
            "render_report": self.render_report,
            "status": self.status,
            "started_at_utc": self.started_at_utc,
            "finished_at_utc": self.finished_at_utc,
            "updated_at_utc": self.updated_at_utc,
            "output_dir": self.output_dir,
            "progress_percent": self.progress_percent,
            "progress_message": self.progress_message,
            "progress_events": [dict(item) for item in self.progress_events],
            "artifact_paths": dict(self.artifact_paths),
            "summary": dict(self.summary),
            "summarize_feedback_path": self.summarize_feedback_path,
            "summarize_draft_path": self.summarize_draft_path,
            "summarize_state_patch_path": self.summarize_state_patch_path,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "error_hint": self.error_hint,
            "error": self.error,
        }

    def to_compact_payload(self) -> dict[str, object]:
        # Lightweight payload for run polling.
        return {
            "run_id": self.run_id,
            "project_slug": self.project_slug,
            "case_name": self.case_name,
            "round_id": self.round_id,
            "spec_path": self.spec_path,
            "submitted_at_utc": self.submitted_at_utc,
            "evaluation_profile": self.evaluation_profile,
            "output_root_dir": self.output_root_dir,
            "render_report": self.render_report,
            "status": self.status,
            "started_at_utc": self.started_at_utc,
            "finished_at_utc": self.finished_at_utc,
            "updated_at_utc": self.updated_at_utc,
            "output_dir": self.output_dir,
            "progress_percent": self.progress_percent,
            "progress_message": self.progress_message,
            "progress_events": [dict(item) for item in self.progress_events[-2:]],
            "artifact_paths": {key: True for key in self.artifact_paths.keys()},
            "summary": _compact_metrics_summary(self.summary),
            "summarize_feedback_path": self.summarize_feedback_path,
            "summarize_draft_path": self.summarize_draft_path,
            "summarize_state_patch_path": self.summarize_state_patch_path,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "error_hint": self.error_hint,
            "error": self.error,
            "_compact": True,
        }


@dataclass(frozen=True)
class _RunTask:
    run_id: str
    project_slug: str
    case_name: str
    round_id: str | None
    spec_path: str
    evaluation_profile: str
    output_root_dir: str | None
    render_report: bool


@dataclass
class _InputBundleCacheEntry:
    bundle: SingleFactorInputBundle
    last_used_seq: int


class _RunStore:
    def __init__(self) -> None:
        self._records: dict[str, _RunRecord] = {}
        self._tasks: dict[str, _RunTask] = {}
        self._input_bundle_cache: dict[
            tuple[str, str | None, str, int, int],
            _InputBundleCacheEntry,
        ] = {}
        self._input_bundle_cache_clock: int = 0
        self._lock = threading.Lock()
        self._dispatch_event = threading.Event()
        self._dispatcher = threading.Thread(target=self._dispatch_loop, daemon=True)
        self._dispatcher.start()

    def submit(self, task: _RunTask) -> _RunRecord:
        submitted_at = _utc_now_iso()
        record = _RunRecord(
            run_id=task.run_id,
            project_slug=task.project_slug,
            case_name=task.case_name,
            round_id=task.round_id,
            spec_path=task.spec_path,
            submitted_at_utc=submitted_at,
            evaluation_profile=task.evaluation_profile,
            output_root_dir=task.output_root_dir,
            render_report=task.render_report,
            updated_at_utc=submitted_at,
            progress_percent=0,
            progress_message="已提交到队列，等待调度",
            progress_events=[
                {
                    "ts": submitted_at,
                    "message": "已提交到队列，等待调度",
                    "percent": 0,
                }
            ],
        )
        with self._lock:
            self._records[record.run_id] = record
            self._tasks[record.run_id] = task
        self._dispatch_event.set()
        return record.clone()

    def get(self, run_id: str) -> _RunRecord | None:
        with self._lock:
            record = self._records.get(run_id)
            if record is None:
                return None
            self._hydrate_summary_locked(record)
            return record.clone()

    def list_records(self, *, project_slug: str | None = None) -> list[_RunRecord]:
        with self._lock:
            for record in self._records.values():
                self._hydrate_summary_locked(record)
            records = [rec.clone() for rec in self._records.values()]
        if project_slug is None:
            return sorted(records, key=lambda item: item.submitted_at_utc, reverse=True)
        filtered = [item for item in records if item.project_slug == project_slug]
        return sorted(filtered, key=lambda item: item.submitted_at_utc, reverse=True)

    def _hydrate_summary_locked(self, record: _RunRecord) -> None:
        """Backfill run.summary from metrics.json when summary is missing.

        This keeps the run table robust for runs created by older code paths
        where summary extraction might not have been stored in memory.
        """
        if record.status != "succeeded":
            return
        if record.summary:
            return

        metrics_path: Path | None = None
        metrics_text = record.artifact_paths.get("metrics")
        if metrics_text:
            candidate = Path(metrics_text).expanduser().resolve()
            if candidate.exists() and candidate.is_file():
                metrics_path = candidate
        if metrics_path is None and record.output_dir:
            candidate = Path(record.output_dir).expanduser().resolve() / "metrics.json"
            if candidate.exists() and candidate.is_file():
                metrics_path = candidate
        if metrics_path is None:
            return
        try:
            record.summary = _extract_metrics_summary(metrics_path)
        except Exception:
            # Keep run listing resilient even if one metrics file is malformed.
            return

    def delete(self, run_id: str) -> _RunRecord | None:
        with self._lock:
            record = self._records.pop(run_id, None)
            self._tasks.pop(run_id, None)
            return record.clone() if record is not None else None

    def attach_summary(
        self,
        *,
        run_id: str,
        feedback_path: Path,
        draft_path: Path,
        state_patch_path: Path,
    ) -> None:
        with self._lock:
            record = self._records.get(run_id)
            if record is None:
                return
            record.summarize_feedback_path = str(feedback_path)
            record.summarize_draft_path = str(draft_path)
            record.summarize_state_patch_path = str(state_patch_path)

    def _load_cached_input_bundle(
        self,
        spec: SingleFactorCaseSpec,
    ) -> tuple[SingleFactorInputBundle, bool]:
        key = self._build_input_bundle_cache_key(spec)
        with self._lock:
            cached = self._input_bundle_cache.get(key)
            if cached is not None:
                self._input_bundle_cache_clock += 1
                cached.last_used_seq = self._input_bundle_cache_clock
                return cached.bundle, True

        bundle = load_standard_inputs(spec)
        with self._lock:
            self._input_bundle_cache_clock += 1
            self._input_bundle_cache[key] = _InputBundleCacheEntry(
                bundle=bundle,
                last_used_seq=self._input_bundle_cache_clock,
            )
            while len(self._input_bundle_cache) > _FRONTEND_INPUT_BUNDLE_CACHE_MAX_ITEMS:
                oldest_key = min(
                    self._input_bundle_cache.items(),
                    key=lambda item: item[1].last_used_seq,
                )[0]
                self._input_bundle_cache.pop(oldest_key, None)
        return bundle, False

    @staticmethod
    def _build_input_bundle_cache_key(
        spec: SingleFactorCaseSpec,
    ) -> tuple[str, str | None, str, int, int]:
        return (
            spec.prices_path,
            spec.universe.path,
            spec.universe.in_universe_column,
            _file_mtime_ns(spec.prices_path),
            _file_mtime_ns(spec.universe.path),
        )

    def _push_progress(
        self,
        run_id: str,
        *,
        message: str,
        percent: int | None = None,
    ) -> None:
        with self._lock:
            record = self._records.get(run_id)
            if record is None:
                return
            self._push_progress_locked(record, message=message, percent=percent)

    def _push_progress_locked(
        self,
        record: _RunRecord,
        *,
        message: str,
        percent: int | None = None,
    ) -> None:
        ts = _utc_now_iso()
        record.updated_at_utc = ts
        record.progress_message = message
        if percent is not None:
            record.progress_percent = max(0, min(int(percent), 100))
        event: dict[str, object] = {
            "ts": ts,
            "message": message,
            "percent": record.progress_percent,
        }
        record.progress_events = [*record.progress_events[-7:], event]

    def _dispatch_loop(self) -> None:
        while True:
            self._dispatch_event.wait()
            self._dispatch_event.clear()
            while self._dispatch_event.wait(timeout=_FRONTEND_BATCH_WINDOW_SECONDS):
                self._dispatch_event.clear()
            while True:
                groups = self._claim_queued_task_groups()
                if not groups:
                    break
                for tasks in groups:
                    self._execute_task_group(tasks)

    def _claim_queued_task_groups(self) -> list[list[_RunTask]]:
        with self._lock:
            queued_with_records: list[tuple[_RunTask, _RunRecord]] = []
            for run_id, task in self._tasks.items():
                record = self._records.get(run_id)
                if record is None or record.status != "queued":
                    continue
                queued_with_records.append((task, record))
            if not queued_with_records:
                return []
            started_at = _utc_now_iso()
            grouped: dict[tuple[str, str], list[_RunTask]] = {}
            for task, _record in sorted(
                queued_with_records,
                key=lambda item: item[1].submitted_at_utc,
            ):
                key = (
                    task.evaluation_profile,
                    task.output_root_dir or "",
                )
                grouped.setdefault(key, []).append(task)
            ordered_groups = list(grouped.values())
            for tasks in ordered_groups:
                batch_message = (
                    "已进入前端批量调度窗口，等待复用输入与并行执行"
                    if len(tasks) > 1
                    else "任务已启动，准备执行 single-factor pipeline"
                )
                batch_percent = 1 if len(tasks) > 1 else 2
                for task in tasks:
                    record = self._records.get(task.run_id)
                    if record is None:
                        continue
                    record.status = "running"
                    record.started_at_utc = started_at
                    self._push_progress_locked(
                        record,
                        message=batch_message,
                        percent=batch_percent,
                    )
            return ordered_groups

    def _execute_task_group(self, tasks: list[_RunTask]) -> None:
        if len(tasks) <= 1:
            if tasks:
                self._execute_single_task(tasks[0], allow_fallback=False)
            return

        batch_config = _build_frontend_batch_parallel_config(len(tasks))
        batch_message = (
            f"前端批量调度命中，共 {len(tasks)} 个实验，"
            f"mode={batch_config.mode} workers={batch_config.max_workers or 1} "
            f"chunk={batch_config.factors_per_worker}"
        )
        for task in tasks:
            self._push_progress(task.run_id, message=batch_message, percent=4)

        try:
            results = run_single_factor_cases(
                [task.spec_path for task in tasks],
                output_root_dir=tasks[0].output_root_dir,
                evaluation_profile=tasks[0].evaluation_profile,
                vault_export_mode="skip",
                batch_parallel_config=batch_config,
                reuse_input_bundle=True,
                progress_callback=lambda message, percent: self._push_batch_progress(
                    tasks,
                    message=message,
                    percent=percent,
                ),
            )
        except Exception:
            for task in tasks:
                self._push_progress(
                    task.run_id,
                    message="前端批量执行失败，自动回退到逐个执行",
                    percent=6,
                )
            for task in tasks:
                self._execute_single_task(task, allow_fallback=False)
            return

        if len(results) != len(tasks):
            for task in tasks:
                self._push_progress(
                    task.run_id,
                    message="批量结果数量异常，自动回退到逐个执行",
                    percent=6,
                )
            for task in tasks:
                self._execute_single_task(task, allow_fallback=False)
            return

        for task, result in zip(tasks, results, strict=True):
            self._finalize_success(task=task, result=result)

    def _push_batch_progress(
        self,
        tasks: list[_RunTask],
        *,
        message: str,
        percent: int | None,
    ) -> None:
        for task in tasks:
            self._push_progress(task.run_id, message=message, percent=percent)

    def _execute_single_task(
        self,
        task: _RunTask,
        *,
        allow_fallback: bool,
    ) -> None:
        del allow_fallback
        run_id = task.run_id
        with self._lock:
            record = self._records.get(run_id)
            stored_task = self._tasks.get(run_id)
            if record is None:
                return
            if stored_task is None:
                return
            if record.started_at_utc is None:
                record.status = "running"
                record.started_at_utc = _utc_now_iso()
                self._push_progress_locked(
                    record,
                    message="任务已启动，准备执行 single-factor pipeline",
                    percent=2,
                )
        try:

            def progress_callback(message: str, percent: int) -> None:
                self._push_progress(
                    run_id,
                    message=message,
                    percent=percent,
                )

            spec = load_single_factor_case_spec(Path(task.spec_path).resolve())
            bundle, _ = self._load_cached_input_bundle(spec)
            result = run_single_factor_case(
                spec,
                output_root_dir=task.output_root_dir,
                evaluation_profile=task.evaluation_profile,
                vault_export_mode="skip",
                progress_callback=progress_callback,
                input_bundle=bundle,
            )
            self._finalize_success(task=task, result=result)
        except Exception as exc:
            error_payload = _build_run_error_payload(exc)
            with self._lock:
                stored = self._records[run_id]
                stored.status = "failed"
                stored.finished_at_utc = _utc_now_iso()
                stored.updated_at_utc = stored.finished_at_utc
                stored.progress_message = f"失败于：{stored.progress_message or '未知阶段'}"
                stored.progress_events = [
                    *stored.progress_events[-7:],
                    {
                        "ts": stored.finished_at_utc,
                        "message": stored.progress_message,
                        "percent": stored.progress_percent,
                    },
                ]
                stored.error_type = error_payload["error_type"]
                stored.error_message = error_payload["error_message"]
                stored.error_hint = error_payload["error_hint"]
                stored.error = _format_run_error_text(
                    stage=stored.progress_message,
                    error_type=error_payload["error_type"],
                    error_message=error_payload["error_message"],
                    error_hint=error_payload["error_hint"],
                    traceback_text=traceback.format_exc(limit=20),
                )
        finally:
            with self._lock:
                self._tasks.pop(run_id, None)

    def _finalize_success(self, *, task: _RunTask, result: Any) -> None:
        run_id = task.run_id
        self._push_progress(run_id, message="整理产物清单", percent=93)
        artifact_paths = {key: str(path) for key, path in result.artifact_paths.items()}
        if task.render_report:
            self._push_progress(run_id, message="生成 case report", percent=96)
            report_path = write_case_report(result.output_dir, overwrite=True)
            artifact_paths["case_report"] = str(report_path)
        self._push_progress(run_id, message="提取关键指标摘要", percent=98)
        summary = _extract_metrics_summary(result.artifact_paths.get("metrics"))
        with self._lock:
            stored = self._records[run_id]
            stored.status = "succeeded"
            stored.finished_at_utc = _utc_now_iso()
            stored.updated_at_utc = stored.finished_at_utc
            stored.output_dir = str(result.output_dir)
            stored.progress_percent = 100
            stored.progress_message = "运行完成"
            stored.progress_events = [
                *stored.progress_events[-7:],
                {
                    "ts": stored.finished_at_utc,
                    "message": "运行完成",
                    "percent": 100,
                },
            ]
            stored.artifact_paths = artifact_paths
            stored.summary = summary
            stored.error_type = None
            stored.error_message = None
            stored.error_hint = None
            stored.error = None
            self._tasks.pop(run_id, None)


def _build_run_error_payload(exc: Exception) -> dict[str, str]:
    error_type = type(exc).__name__
    error_message = str(exc).strip() or repr(exc)
    if isinstance(exc, FileNotFoundError):
        hint = "检查 case spec 中的 prices_path、factor_path、exposures_path 是否存在且路径正确。"
    elif isinstance(exc, AlphaLabDataError):
        hint = (
            "检查输入 CSV 的列名、日期列格式、factor_name 过滤后是否仍有数据，以及是否存在空文件。"
        )
    elif isinstance(exc, AlphaLabConfigError):
        hint = "检查 case spec、evaluation profile、neutralization 配置是否完整且取值合法。"
    elif isinstance(exc, ValueError):
        hint = "检查参数取值、日期格式、空值以及 YAML/JSON/CSV 内容是否合法。"
    elif isinstance(exc, KeyError):
        hint = "通常表示输入表缺少必需列，或配置中引用了不存在的字段名。"
    else:
        hint = (
            "优先查看失败阶段、核心报错和 traceback，定位是路径、数据 schema、"
            "配置还是代码逻辑问题。"
        )
    return {
        "error_type": error_type,
        "error_message": error_message,
        "error_hint": hint,
    }


def _build_frontend_batch_parallel_config(
    n_tasks: int,
) -> SingleFactorBatchParallelConfig:
    worker_slots = max(1, min(_FRONTEND_BATCH_MAX_WORKERS, n_tasks))
    return SingleFactorBatchParallelConfig(
        mode="process",
        max_workers=worker_slots,
        factors_per_worker=_FRONTEND_BATCH_FACTORS_PER_WORKER,
    )


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


# ---------------------------------------------------------------------------
# Unified Service
# ---------------------------------------------------------------------------


class _UnifiedService:
    def __init__(self, *, vault_root: Path, workspace_root: Path) -> None:
        self.vault_root = vault_root.resolve()
        self.workspace_root = workspace_root.resolve()
        self.run_store = _RunStore()
        self._custom_factors_dir = self.workspace_root / "custom_factors"
        self._load_persisted_custom_factors()

    @property
    def projects_root(self) -> Path:
        return (self.vault_root / PROJECTS_DIRNAME).resolve()

    # ---- Dashboard --------------------------------------------------------

    def dashboard(self) -> dict[str, object]:
        projects = self.list_projects()
        runs = [item.to_payload() for item in self.run_store.list_records()]
        status_counts: dict[str, int] = {"queued": 0, "running": 0, "succeeded": 0, "failed": 0}
        for record in runs:
            status = str(record["status"])
            status_counts[status] = status_counts.get(status, 0) + 1
        vault_stats = self.vault_stats()
        return {
            "vault_root": str(self.vault_root),
            "workspace_root": str(self.workspace_root),
            "project_count": len(projects),
            "run_status_counts": status_counts,
            "vault_card_count": vault_stats.get("total_cards", 0),
            "vault_inbox_count": vault_stats.get("inbox_count", 0),
            "active_projects": [
                project
                for project in projects
                if str(project.get("lifecycle", "")).strip() == "active"
            ],
            "recent_runs": runs[:10],
            "next_actions": [
                {
                    "project_slug": project["slug"],
                    "next_action": project["next_action"],
                }
                for project in projects
                if str(project.get("next_action", "")).strip()
            ][:10],
        }

    # ---- Knowledge Ops ----------------------------------------------------

    def vault_stats(self) -> dict[str, object]:
        index_path = (self.vault_root / "90_moc" / "CARD-INDEX.tsv").resolve()
        if not index_path.exists():
            return {
                "total_cards": 0,
                "inbox_count": self._count_inbox(),
                "by_type": {},
                "by_lifecycle": {},
            }
        by_type: dict[str, int] = {}
        by_lifecycle: dict[str, int] = {}
        total = 0
        with index_path.open("r", encoding="utf-8") as fh:
            reader = DictReader(fh, delimiter="\t")
            for row in reader:
                total += 1
                ctype = str(row.get("type") or "unknown").strip()
                lifecycle = str(row.get("lifecycle") or "unknown").strip()
                by_type[ctype] = by_type.get(ctype, 0) + 1
                by_lifecycle[lifecycle] = by_lifecycle.get(lifecycle, 0) + 1
        return {
            "total_cards": total,
            "inbox_count": self._count_inbox(),
            "by_type": dict(sorted(by_type.items())),
            "by_lifecycle": dict(sorted(by_lifecycle.items())),
        }

    def _count_inbox(self) -> int:
        count = 0
        for dirname in ("00_inbox", "_sources"):
            d = self.vault_root / dirname
            if d.exists():
                count += sum(1 for f in d.iterdir() if f.is_file())
        return count

    def vault_inbox(self) -> dict[str, object]:
        items: list[dict[str, str]] = []
        for dirname in ("00_inbox", "_sources"):
            d = self.vault_root / dirname
            if not d.exists():
                continue
            for f in sorted(d.iterdir()):
                if f.is_file():
                    items.append(
                        {
                            "name": f.name,
                            "directory": dirname,
                            "path": str(f),
                            "size_bytes": str(f.stat().st_size),
                            "modified": dt.datetime.fromtimestamp(f.stat().st_mtime, tz=dt.UTC)
                            .isoformat()
                            .replace("+00:00", "Z"),
                        }
                    )
        return {"items": items, "count": len(items)}

    def read_card(self, card_name: str) -> dict[str, object]:
        # Reject traversal attempts
        if ".." in card_name or card_name.startswith("/") or "\\" in card_name:
            raise PermissionError("invalid card name")
        if not card_name.strip():
            raise ValueError("card name must be non-empty")

        vault = self.vault_root.resolve()

        # Case A: vault-relative path as stored in CARD-INDEX.tsv
        # e.g. "10_concepts/behavioral/Concept - Habit Formation.md"
        if "/" in card_name:
            path = (vault / card_name).resolve()
            if not str(path).startswith(str(vault) + "/") and path != vault:
                raise PermissionError("invalid card name")
            if path.exists() and path.is_file():
                return self._read_card_file(card_name, path)
            raise FileNotFoundError(f"card not found: {card_name}")

        # Case B: bare filename — search CARD-INDEX.tsv first (O(n) single file
        # read), avoids slow rglob on WSL2/network filesystems.
        candidate = card_name if card_name.endswith(".md") else card_name + ".md"
        index_file = vault / "90_moc" / "CARD-INDEX.tsv"
        if index_file.exists():
            try:
                with index_file.open(encoding="utf-8") as fh:
                    for line in fh:
                        parts = line.rstrip("\n").split("\t")
                        if not parts:
                            continue
                        rel_path = parts[0]
                        if not rel_path:
                            continue
                        filename = rel_path.rsplit("/", 1)[-1]
                        if filename == candidate:
                            resolved = (vault / rel_path).resolve()
                            vault_str = str(vault)
                            if str(resolved).startswith(vault_str + "/") or str(
                                resolved
                            ).startswith(vault_str + "\\"):
                                if resolved.is_file():
                                    return self._read_card_file(card_name, resolved)
            except OSError:
                pass  # fall through to directory scan

        # Fallback: shallow glob (top-level only per subdir) — no recursive scan
        for subdir in (
            "30_factors",
            "20_methods",
            "10_concepts",
            "40_papers",
            "60_playbooks",
            "80_pipelines",
            "70_code_patterns",
            "50_experiments",
        ):
            subdir_path = vault / subdir
            if not subdir_path.exists():
                continue
            # Shallow: only direct children, no rglob
            hit = subdir_path / candidate
            if hit.is_file():
                resolved = hit.resolve()
                if str(resolved).startswith(str(vault)):
                    return self._read_card_file(card_name, resolved)
        raise FileNotFoundError(f"card not found: {card_name}")

    def _read_card_file(self, card_name: str, path: Path) -> dict[str, object]:
        size = path.stat().st_size
        if size > _MAX_TEXT_BYTES:
            return {
                "name": card_name,
                "path": str(path),
                "content": path.read_text(encoding="utf-8")[:_MAX_TEXT_BYTES],
                "truncated": True,
                "size_bytes": size,
            }
        return {
            "name": card_name,
            "path": str(path),
            "content": path.read_text(encoding="utf-8"),
            "truncated": False,
            "size_bytes": size,
        }

    def search_cards(self, query: str, *, limit: int = 50) -> dict[str, object]:
        index_path = (self.vault_root / "90_moc" / "CARD-INDEX.tsv").resolve()
        if not index_path.exists():
            return {
                "cards": [],
                "index_path": str(index_path),
                "warning": "CARD-INDEX.tsv not found",
            }
        needle = query.strip().lower()
        rows: list[dict[str, str]] = []
        with index_path.open("r", encoding="utf-8") as fh:
            reader = DictReader(fh, delimiter="\t")
            for row in reader:
                normalized = {key: str(value or "") for key, value in row.items()}
                if not needle:
                    rows.append(normalized)
                    if len(rows) >= limit:
                        break
                    continue
                haystack = " ".join(
                    [
                        normalized.get("path", ""),
                        normalized.get("type", ""),
                        normalized.get("name", ""),
                        normalized.get("domain", ""),
                        normalized.get("lifecycle", ""),
                        normalized.get("tags", ""),
                        normalized.get("parent_moc", ""),
                    ]
                ).lower()
                if needle in haystack:
                    rows.append(normalized)
                    if len(rows) >= limit:
                        break
        return {"cards": rows, "index_path": str(index_path), "query": query, "limit": limit}

    def explore_idea(
        self,
        idea: str,
        mode: str,
        project_slug: str | None = None,
    ) -> dict[str, object]:
        return bridge_explore_idea(
            vault_root=self.vault_root,
            idea=idea,
            mode=mode,
            project_slug=project_slug,
        ).to_payload()

    # ---- Graph / Preflight ------------------------------------------------

    def graph_coverage(self) -> dict[str, object]:
        """Return mechanism × family matrix + graph health stats from VaultGraph."""
        g = VaultGraph.from_vault_root(self.vault_root)
        try:
            g.build(vault_root=self.vault_root)
        except Exception as exc:
            return {"ok": False, "error": str(exc), "matrix": {}, "coverage": {}, "stats": {}}
        matrix = g.mechanism_family_matrix()
        coverage = g.coverage_by_type()
        domain_coverage = g.domain_coverage_matrix()
        stats = {
            "node_count": len(g._graph.nodes) if hasattr(g, "_graph") and g._graph else 0,
            "edge_count": len(g._graph.edges) if hasattr(g, "_graph") and g._graph else 0,
            "orphan_nodes": g.orphan_nodes(),
            "dangling_edge_count": len(g.dangling_edges()),
        }
        # Summarise matrix: for each family, list mechanisms and their validated counts
        summary: dict[str, dict[str, int]] = {}
        for family, mech_dict in matrix.items():
            summary[family] = {mech: len(nodes) for mech, nodes in mech_dict.items()}
        return {
            "ok": True,
            "matrix": summary,
            "coverage": coverage,
            "domain_coverage": domain_coverage,
            "stats": stats,
        }

    def run_preflight_check(self, payload: dict[str, object]) -> dict[str, object]:
        """Run graph-based preflight checks for a candidate (category-aware)."""
        candidate_similar = _coerce_text_list(payload.get("candidate_similar"), delimiter=",")
        candidate_uses_data = _coerce_text_list(payload.get("candidate_uses_data"), delimiter=",")
        checked_card_paths = _coerce_text_list(payload.get("checked_card_paths"), delimiter="\n")

        # Category-aware: only run relevant preflight checks
        category = str(payload.get("category") or "factor_recipe")
        profile = get_category_profile(category)

        report = run_preflight(
            vault_root=self.vault_root,
            checked_card_paths=checked_card_paths or None,
            candidate_name=str(payload.get("candidate_name") or ""),
            candidate_family=str(payload.get("candidate_family") or ""),
            candidate_mechanism=str(payload.get("candidate_mechanism") or ""),
            candidate_similar=candidate_similar,
            candidate_uses_data=candidate_uses_data,
            candidate_pit_sensitivity=str(payload.get("candidate_pit_sensitivity") or ""),
            candidate_decay_class=str(payload.get("candidate_decay_class") or ""),
            candidate_capacity_class=str(payload.get("candidate_capacity_class") or ""),
            enabled_checks=profile.preflight_checks,
        )
        issues_payload = [
            {"severity": i.severity, "code": i.code, "message": i.message} for i in report.issues
        ]
        novelty_payload: dict[str, object] = {}
        if report.novelty:
            novelty_payload = {
                "similar_existing": report.novelty.similar_existing,
                "same_mechanism_family": report.novelty.same_mechanism_family,
                "warnings": report.novelty.warnings,
            }
        decomp_payload: dict[str, object] = {}
        if report.decomposition:
            decomp_payload = {
                "warnings": report.decomposition.warnings,
            }
        return {
            "ok": True,
            "is_blocked": report.is_blocked,
            "checked_cards": report.checked_cards,
            "issues": issues_payload,
            "novelty": novelty_payload,
            "decomposition": decomp_payload,
        }

    # ---- Bridge Workspace -------------------------------------------------

    def list_projects(self) -> list[dict[str, object]]:
        root = self.projects_root
        if not root.exists():
            return []
        rows: list[dict[str, object]] = []
        for project_yaml in _iter_project_contracts(root):
            try:
                project = load_project_config(project_yaml)
            except Exception:
                continue
            paths = _project_paths(self.vault_root, project_yaml.parent.name)
            rows.append(
                {
                    "slug": project.slug,
                    "title_zh": project.title_zh,
                    "owner": project.owner,
                    "market": project.market,
                    "frequency": project.frequency,
                    "lifecycle": project.status.lifecycle,
                    "current_focus": project.status.current_focus,
                    "next_action": project.status.next_action,
                    "current_case": project.status.current_case,
                    "last_verdict": project.status.last_verdict,
                    "case_count": len(_list_cases(paths)),
                    "path": str(paths["project_dir"]),
                }
            )
        return sorted(rows, key=lambda row: str(row["slug"]))

    def get_project(self, slug: str) -> dict[str, object]:
        paths = _project_paths(self.vault_root, slug)
        if not paths["project_yaml"].exists():
            raise FileNotFoundError(f"project not found: {slug}")
        project = load_project_config(paths["project_yaml"])
        cases = _list_cases(paths)
        docs = {
            "decision_log": _read_text_preview(
                paths["decision_log"],
                limit_bytes=_PROJECT_DOC_PREVIEW_BYTES,
            ),
            "current_case": _read_text_preview(
                paths["current_case"],
                limit_bytes=_PROJECT_DOC_PREVIEW_BYTES,
            ),
            "latest_run": _read_text_preview(
                paths["latest_run"],
                limit_bytes=_PROJECT_DOC_PREVIEW_BYTES,
            ),
        }
        return {
            "project": {
                "slug": project.slug,
                "title_zh": project.title_zh,
                "category": project.category,
                "owner": project.owner,
                "market": project.market,
                "frequency": project.frequency,
                "chatgpt_project_name": project.chatgpt_project_name,
                "max_research_level": project.max_research_level,
                "origin_cards": list(project.origin_cards),
                "supporting_cards": list(project.supporting_cards),
                "failure_cards": list(project.failure_cards),
                "related_experiment_cards": list(project.related_experiment_cards),
                "preferred_web_sources": list(project.preferred_web_sources),
                "status": {
                    "lifecycle": project.status.lifecycle,
                    "current_hypothesis": project.status.current_hypothesis,
                    "current_focus": project.status.current_focus,
                    "next_action": project.status.next_action,
                    "current_case": project.status.current_case,
                    "latest_run": project.status.latest_run,
                    "last_verdict": project.status.last_verdict,
                },
                "alpha_lab_defaults": {
                    "data_source": project.alpha_lab_defaults.data_source,
                    "slice_preset": project.alpha_lab_defaults.slice_preset,
                    "universe": project.alpha_lab_defaults.universe,
                    "adjustment": project.alpha_lab_defaults.adjustment,
                    "evaluation_profile": project.alpha_lab_defaults.evaluation_profile,
                },
            },
            "paths": {key: str(path) for key, path in paths.items()},
            "documents": docs,
            "cases": cases,
            "runs": [
                item.to_payload() for item in self.run_store.list_records(project_slug=project.slug)
            ],
        }

    def create_project(self, payload: dict[str, object]) -> dict[str, object]:
        required_fields = [
            "slug",
            "title_zh",
            "category",
            "owner",
            "market",
            "frequency",
            "chatgpt_project_name",
        ]
        missing = [field for field in required_fields if not str(payload.get(field) or "").strip()]
        if missing:
            raise ValueError(f"missing required fields: {missing}")
        result = init_project(
            vault_root=self.vault_root,
            slug=str(payload.get("slug")),
            title_zh=str(payload.get("title_zh")),
            category=str(payload.get("category")),
            owner=str(payload.get("owner")),
            market=str(payload.get("market")),
            frequency=str(payload.get("frequency")),
            chatgpt_project_name=str(payload.get("chatgpt_project_name")),
            max_research_level=_as_int(payload.get("max_research_level"), default=2),
            origin_cards=_as_text_list(payload.get("origin_cards")),
            supporting_cards=_as_text_list(payload.get("supporting_cards")),
            failure_cards=_as_text_list(payload.get("failure_cards")),
            related_experiment_cards=_as_text_list(payload.get("related_experiment_cards")),
            preferred_web_sources=_as_text_list(payload.get("preferred_web_sources")),
            mode=str(payload.get("mode") or "fast"),
            overwrite=bool(payload.get("overwrite", False)),
        )
        return {
            "slug": result.project.slug,
            "project_dir": str(result.paths.project_dir),
        }

    def update_project_status(self, slug: str, payload: dict[str, object]) -> dict[str, object]:
        paths = _project_paths(self.vault_root, slug)
        project = load_project_config(paths["project_yaml"])
        status = project.status
        if "lifecycle" in payload:
            status.lifecycle = str(payload.get("lifecycle") or "").strip() or status.lifecycle
        if "current_hypothesis" in payload:
            status.current_hypothesis = (
                str(payload.get("current_hypothesis") or "").strip() or status.current_hypothesis
            )
        if "current_focus" in payload:
            status.current_focus = (
                str(payload.get("current_focus") or "").strip() or status.current_focus
            )
        if "next_action" in payload:
            status.next_action = str(payload.get("next_action") or "").strip() or status.next_action
        save_project_config(project, paths["project_yaml"])
        refresh_project_pack(vault_root=self.vault_root, project_slug=slug)
        return self.get_project(slug)

    def refresh_project(self, slug: str) -> dict[str, object]:
        result = refresh_project_pack(vault_root=self.vault_root, project_slug=slug)
        return {"slug": result.project.slug, "project_dir": str(result.paths.project_dir)}

    # ---- Validation Console -----------------------------------------------

    def create_round(self, slug: str, payload: dict[str, object]) -> dict[str, object]:
        topic = str(payload.get("topic") or "").strip()
        if not topic:
            raise ValueError("topic is required")
        round_result = start_round(
            vault_root=self.vault_root,
            project_slug=slug,
            topic=topic,
            round_id=str(payload.get("round_id") or "").strip() or None,
            mode=str(payload.get("mode") or "standard"),
        )
        return {
            "project": round_result.project.slug,
            "round_id": round_result.round_id,
            "round_dir": str(round_result.round_dir),
            "round_context_digest": str(round_result.round_context_digest),
            "round_prompt": str(round_result.round_prompt),
            "web_search_tasks": str(round_result.web_search_tasks),
            "discussion_capture": str(round_result.discussion_capture),
        }

    def list_rounds(self, slug: str) -> list[dict[str, object]]:
        paths = _project_paths(self.vault_root, slug)
        if not paths["project_yaml"].exists():
            raise FileNotFoundError(f"project not found: {slug}")
        return _list_rounds(paths["rounds_dir"])

    def create_case(self, slug: str, payload: dict[str, object]) -> dict[str, object]:
        case_name = str(payload.get("case_name") or "").strip()
        if not case_name:
            raise ValueError("case_name is required")
        result = scaffold_case(
            vault_root=self.vault_root,
            project_slug=slug,
            case_name=case_name,
            case_type=str(payload.get("case_type") or "factor_recipe"),
            factor_name=_optional_text(payload.get("factor_name")),
            base_method=str(payload.get("base_method") or "momentum"),
            lookback=_as_int(payload.get("lookback"), default=20),
            skip_recent=_as_int(payload.get("skip_recent"), default=5),
            target_horizon=_as_int(payload.get("target_horizon"), default=5),
            rebalance_frequency=str(payload.get("rebalance_frequency") or "W"),
            direction=str(payload.get("direction") or "long"),
            prices_path=str(payload.get("prices_path") or "./placeholder_prices.csv"),
            universe_path=str(payload.get("universe_path") or "./placeholder_universe.csv"),
            factor_path=str(payload.get("factor_path") or "./placeholder_factor.csv"),
        )
        return {
            "project": result.project.slug,
            "case_name": result.case_name,
            "current_case_path": str(result.current_case_path),
        }

    def list_cases(self, slug: str) -> list[dict[str, object]]:
        paths = _project_paths(self.vault_root, slug)
        return _list_cases(paths)

    def list_drafts(self, slug: str) -> list[dict[str, object]]:
        drafts_dir = _project_paths(self.vault_root, slug)["drafts_dir"]
        project_yaml = _project_paths(self.vault_root, slug)["project_yaml"]
        if not project_yaml.exists():
            raise FileNotFoundError(f"project not found: {slug}")
        return _list_draft_summaries(drafts_dir)

    def read_draft(self, slug: str, draft_name: str) -> dict[str, object]:
        paths = _project_paths(self.vault_root, slug)
        if not paths["project_yaml"].exists():
            raise FileNotFoundError(f"project not found: {slug}")
        draft_path = _resolve_draft_path(paths["drafts_dir"], draft_name)
        frontmatter, _ = _load_markdown_with_frontmatter(draft_path)
        preview = _read_text_preview(
            draft_path,
            limit_bytes=_PROJECT_DOC_PREVIEW_BYTES,
        )
        size_bytes = draft_path.stat().st_size
        return {
            "name": draft_path.name,
            "path": str(draft_path),
            "frontmatter": frontmatter,
            "body": preview,
            "content": preview,
            "size_bytes": size_bytes,
            "truncated": size_bytes > _PROJECT_DOC_PREVIEW_BYTES,
        }

    def patch_draft(
        self,
        slug: str,
        draft_name: str,
        payload: dict[str, object],
    ) -> dict[str, object]:
        paths = _project_paths(self.vault_root, slug)
        if not paths["project_yaml"].exists():
            raise FileNotFoundError(f"project not found: {slug}")
        draft_path = _resolve_draft_path(paths["drafts_dir"], draft_name)
        frontmatter, body = _load_markdown_with_frontmatter(draft_path)
        allowed = {
            "review_status",
            "reviewed_by",
            "reviewed_at",
            "one_sentence_verdict",
            "status_lifecycle",
            "current_hypothesis",
            "current_focus",
            "next_action",
            "vault_export_mode",
            "review_note",
        }
        for key, value in payload.items():
            if key not in allowed:
                continue
            if key == "reviewed_at" and str(value).strip().lower() == "now":
                frontmatter[key] = _utc_now_iso()
            else:
                frontmatter[key] = str(value)
        draft_path.write_text(
            _compose_markdown_with_frontmatter(frontmatter, body),
            encoding="utf-8",
        )
        return _draft_summary(draft_path)

    def apply_draft(
        self,
        slug: str,
        draft_name: str,
        payload: dict[str, object] | None = None,
    ) -> dict[str, object]:
        paths = _project_paths(self.vault_root, slug)
        if not paths["project_yaml"].exists():
            raise FileNotFoundError(f"project not found: {slug}")
        draft_path = _resolve_draft_path(paths["drafts_dir"], draft_name)
        mode = _optional_text(payload.get("mode")) if payload is not None else None
        result = apply_writeback(
            vault_root=self.vault_root,
            project_slug=slug,
            draft_path=draft_path,
            mode=mode,
        )
        return {
            "project": result.project.slug,
            "draft_path": str(result.draft_path),
            "status": result.export_result.status,
            "success": result.export_result.success,
            "target_paths": list(result.export_result.target_paths),
            "mode_used": result.export_result.mode_used,
            "error": result.export_result.error,
        }

    def list_evaluation_profiles(self) -> dict[str, object]:
        configured_profiles = [
            p
            for p in CAMPAIGN_PROFILE_COMPARE_DEFAULTS
            if p in AVAILABLE_RESEARCH_EVALUATION_PROFILES
        ]
        if len(configured_profiles) < len(AVAILABLE_RESEARCH_EVALUATION_PROFILES):
            configured_profiles.extend(
                p
                for p in AVAILABLE_RESEARCH_EVALUATION_PROFILES
                if p not in set(configured_profiles)
            )
        profile_labels = {
            profile: RESEARCH_EVALUATION_PROFILE_LABELS.get(
                profile,
                profile.replace("_", " "),
            )
            for profile in configured_profiles
        }
        return {
            "profiles": configured_profiles,
            "default_profile": "exploratory_screening",
            "profile_labels": profile_labels,
        }

    def project_factor_diagnostics(
        self,
        slug: str,
        *,
        threshold: float = 0.7,
        min_overlap: int = 5,
    ) -> dict[str, object]:
        runs = [
            item
            for item in self.run_store.list_records(project_slug=slug)
            if item.status == "succeeded"
        ]
        dsr_by_factor: list[dict[str, object]] = []
        for run in runs:
            dsr_pvalue = _resolve_run_dsr_pvalue(run)
            if dsr_pvalue is None:
                continue
            dsr_by_factor.append(
                {
                    "run_id": run.run_id,
                    "case_name": run.case_name,
                    "factor_name": _resolve_run_factor_label(run),
                    "dsr_pvalue": dsr_pvalue,
                    "risk_level": _classify_dsr_pvalue(dsr_pvalue),
                }
            )
        dsr_by_factor.sort(
            key=lambda row: (
                _coerce_finite_float(row.get("dsr_pvalue")) or 0.0,
                str(row.get("factor_name") or ""),
            )
        )
        dsr_summary = _build_project_dsr_summary(dsr_by_factor, n_runs_total=len(runs))

        response: dict[str, object] = {
            "ok": False,
            "message": "成功运行记录不足（至少需要 2 个已完成 run）。",
            "labels": [],
            "matrix": [],
            "redundancy_pairs": [],
            "n_runs_used": 0,
            "n_runs_total": len(runs),
            "threshold": threshold,
            "min_overlap": min_overlap,
            "metric": "rank_ic_timeseries_spearman",
            "dsr_summary": dsr_summary,
            "dsr_by_factor": dsr_by_factor,
        }
        if len(runs) < 2:
            return response

        series_by_label: dict[str, dict[str, float]] = {}
        for run in runs:
            series = _load_run_rank_ic_timeseries(run)
            if len(series) < min_overlap:
                continue
            base_label = _resolve_run_factor_label(run)
            label = base_label
            suffix = 2
            while label in series_by_label:
                label = f"{base_label}#{suffix}"
                suffix += 1
            series_by_label[label] = series

        labels = sorted(series_by_label.keys())
        if len(labels) < 2:
            response["message"] = "可用于相关性计算的 run 不足（有效时序太短或缺失）。"
            return response

        matrix: list[list[float | None]] = []
        redundancy_pairs: list[dict[str, object]] = []
        for i, left_label in enumerate(labels):
            row: list[float | None] = []
            for j, right_label in enumerate(labels):
                if i == j:
                    row.append(1.0)
                    continue
                if j < i:
                    row.append(matrix[j][i])
                    continue
                corr, overlap = _pairwise_spearman_from_timeseries(
                    series_by_label[left_label],
                    series_by_label[right_label],
                    min_overlap=min_overlap,
                )
                row.append(corr)
                if corr is not None and abs(corr) >= threshold:
                    redundancy_pairs.append(
                        {
                            "factor_a": left_label,
                            "factor_b": right_label,
                            "correlation": corr,
                            "abs_correlation": abs(corr),
                            "overlap_dates": overlap,
                            "warning": "high"
                            if abs(corr) >= max(threshold + 0.15, 0.85)
                            else "medium",
                        }
                    )
            matrix.append(row)

        redundancy_pairs.sort(
            key=lambda row: _coerce_finite_float(row.get("abs_correlation")) or 0.0,
            reverse=True,
        )
        response.update(
            {
                "ok": True,
                "message": "",
                "labels": labels,
                "matrix": matrix,
                "redundancy_pairs": redundancy_pairs,
                "n_runs_used": len(labels),
            }
        )
        return response

    def submit_run(self, slug: str, payload: dict[str, object]) -> dict[str, object]:
        case_name = str(payload.get("case_name") or "").strip()
        if not case_name:
            raise ValueError("case_name is required")
        paths = _project_paths(self.vault_root, slug)
        spec_path = _resolve_case_spec_path(paths, case_name)
        if not spec_path.exists():
            raise FileNotFoundError(f"case spec does not exist: {spec_path}")
        project = load_project_config(paths["project_yaml"])
        task = _RunTask(
            run_id=uuid.uuid4().hex,
            project_slug=slug,
            case_name=case_name,
            round_id=_optional_text(payload.get("round_id")),
            spec_path=str(spec_path),
            evaluation_profile=str(
                payload.get("evaluation_profile") or project.alpha_lab_defaults.evaluation_profile
            ),
            output_root_dir=_optional_text(payload.get("output_root_dir")),
            render_report=bool(payload.get("render_report", True)),
        )
        record = self.run_store.submit(task)
        return record.to_payload()

    def summarize_run(
        self, slug: str, run_id: str, payload: dict[str, object]
    ) -> dict[str, object]:
        run_record = self.run_store.get(run_id)
        if run_record is None or run_record.project_slug != slug:
            raise FileNotFoundError(f"run not found: {run_id}")
        if run_record.status != "succeeded":
            raise AlphaLabConfigError(
                f"run {run_id} is not succeeded; current status: {run_record.status}"
            )
        if not run_record.output_dir:
            raise AlphaLabConfigError(f"run {run_id} has no output_dir")
        result = summarize_run(
            vault_root=self.vault_root,
            project_slug=slug,
            run_root=Path(run_record.output_dir),
        )
        self.run_store.attach_summary(
            run_id=run_id,
            feedback_path=result.latest_experiment_feedback,
            draft_path=result.writeback_draft,
            state_patch_path=result.state_update_patch,
        )
        return {
            "project": result.project.slug,
            "summary_path": str(result.summary_path),
            "latest_path": str(result.latest_path),
            "decision_log_path": str(result.decision_log_path),
            "graph_feedback": dict(result.graph_feedback),
        }

    def delete_run(self, slug: str, run_id: str) -> dict[str, object]:
        """Delete a run record and all associated artifacts from disk."""
        record = self.run_store.get(run_id)
        if record is None or record.project_slug != slug:
            raise FileNotFoundError(f"run not found: {run_id}")
        if record.status in ("queued", "running"):
            raise AlphaLabConfigError(f"cannot delete run {run_id} while it is {record.status}")
        import shutil

        deleted_paths: list[str] = []
        # 1. Delete output_dir (dist/bridge_runs/{case_name}/)
        if record.output_dir:
            output_dir = Path(record.output_dir)
            if output_dir.exists() and output_dir.is_dir():
                shutil.rmtree(output_dir)
                deleted_paths.append(f"output_dir: {output_dir}")
        # 2. Delete summarize artifacts in the vault project dir
        paths = _project_paths(self.vault_root, slug)
        case_slug = _safe_slug(record.case_name)
        runs_dir = paths["runs_dir"]
        run_summary_dir = runs_dir / case_slug
        if run_summary_dir.exists() and run_summary_dir.is_dir():
            shutil.rmtree(run_summary_dir)
            deleted_paths.append(f"run_summary: {run_summary_dir}")
        # 3. Delete writeback drafts matching this case
        drafts_dir = paths["project_dir"] / "50_writeback_drafts"
        if drafts_dir.exists():
            for draft in drafts_dir.glob(f"*__{case_slug}__writeback_draft.md"):
                draft.unlink(missing_ok=True)
                deleted_paths.append(f"draft: {draft.name}")
        # 4. Remove in-memory record
        self.run_store.delete(run_id)
        return {
            "ok": True,
            "run_id": run_id,
            "deleted_paths": deleted_paths,
        }

    # ---- Custom Factor Workshop ---------------------------------------------

    def _load_persisted_custom_factors(self) -> None:
        """Load previously saved custom factors from disk and register them."""
        if not self._custom_factors_dir.exists():
            return
        for meta_path in sorted(self._custom_factors_dir.glob("*.json")):
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                name = meta["name"]
                code = meta["code"]
                fn = _compile_custom_factor(name, code)
                if name not in factor_registry:
                    factor_registry.register(name, fn)
            except Exception:
                pass  # skip broken persisted factors silently

    def list_custom_factors(self) -> dict[str, object]:
        """List all registered factor methods (built-in + custom)."""
        builtin = {"momentum", "reversal", "low_volatility", "amplitude", "downside_volatility"}
        all_methods = factor_registry.supported_methods()
        items: list[dict[str, object]] = []
        for method in all_methods:
            is_custom = method not in builtin
            meta: dict[str, object] = {"name": method, "is_custom": is_custom}
            if is_custom:
                meta_path = self._custom_factors_dir / f"{method}.json"
                if meta_path.exists():
                    try:
                        saved = json.loads(meta_path.read_text(encoding="utf-8"))
                        meta["description"] = saved.get("description", "")
                        meta["created_at"] = saved.get("created_at", "")
                    except Exception:
                        pass
            items.append(meta)
        return {
            "factors": items,
            "total": len(items),
            "custom_count": sum(1 for i in items if i.get("is_custom")),
        }

    def register_custom_factor(self, payload: dict[str, object]) -> dict[str, object]:
        """Register a custom factor from user-provided Python code."""
        name = str(payload.get("name") or "").strip().lower()
        if not name:
            raise ValueError("name is required")
        if not re.match(r"^[a-z][a-z0-9_]*$", name):
            raise ValueError(
                "name must be lowercase alphanumeric with underscores, starting with a letter"
            )
        code = str(payload.get("code") or "").strip()
        if not code:
            raise ValueError("code is required")
        description = str(payload.get("description") or "").strip()

        # Compile and validate the code
        fn = _compile_custom_factor(name, code)

        # Register in the global factor_registry
        factor_registry.register(name, fn)

        # Persist to disk
        self._custom_factors_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "name": name,
            "description": description,
            "code": code,
            "created_at": _utc_now_iso(),
        }
        meta_path = self._custom_factors_dir / f"{name}.json"
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        return {"name": name, "registered": True, "persisted": str(meta_path)}

    def delete_custom_factor(self, name: str) -> dict[str, object]:
        """Unregister a custom factor and remove its persisted file."""
        builtin = {"momentum", "reversal", "low_volatility", "amplitude", "downside_volatility"}
        name = name.strip().lower()
        if name in builtin:
            raise ValueError(f"cannot delete built-in factor: {name}")
        if name not in factor_registry:
            raise FileNotFoundError(f"factor not found: {name}")

        # Remove from registry
        factor_registry._builders.pop(name, None)

        # Remove persisted file
        meta_path = self._custom_factors_dir / f"{name}.json"
        if meta_path.exists():
            meta_path.unlink()

        return {"name": name, "deleted": True}

    def get_custom_factor_code(self, name: str) -> dict[str, object]:
        """Return the source code of a persisted custom factor."""
        meta_path = self._custom_factors_dir / f"{name}.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"custom factor not found: {name}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return {
            "name": meta["name"],
            "code": meta.get("code", ""),
            "description": meta.get("description", ""),
        }


# ---------------------------------------------------------------------------
# HTTP Request Handler
# ---------------------------------------------------------------------------


class _UnifiedRequestHandler(BaseHTTPRequestHandler):
    svc: _UnifiedService

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path

        # Root page
        if path == "/":
            self._send_html(_index_html())
            return

        # Dashboard
        if path == "/api/dashboard":
            self._send_json(self.svc.dashboard())
            return

        # Project list
        if path == "/api/projects":
            self._send_json({"projects": self.svc.list_projects()})
            return

        # Knowledge Ops
        if path == "/api/vault/stats":
            self._send_json(self.svc.vault_stats())
            return
        if path == "/api/vault/inbox":
            self._send_json(self.svc.vault_inbox())
            return
        if path == "/api/vault/graph/coverage":
            try:
                self._send_json(self.svc.graph_coverage())
            except Exception as exc:
                self._send_error_payload(exc)
            return
        if path == "/api/cards/search":
            params = parse_qs(parsed.query)
            query_text = str((params.get("q") or [""])[0])
            limit = _safe_limit((params.get("limit") or ["50"])[0], default=50)
            self._send_json(self.svc.search_cards(query_text, limit=limit))
            return
        if path == "/api/evaluation-profiles":
            self._send_json(self.svc.list_evaluation_profiles())
            return
        if path == "/api/categories":
            self._send_json({"categories": list_categories()})
            return

        # Custom factor routes
        if path == "/api/custom-factors":
            self._send_json(self.svc.list_custom_factors())
            return

        # Project-scoped routes
        parts = _path_parts(path)
        if len(parts) >= 3 and parts[0] == "api" and parts[1] == "projects":
            slug = parts[2]
            try:
                if len(parts) == 3:
                    self._send_json(self.svc.get_project(slug))
                    return
                if len(parts) == 4 and parts[3] == "rounds":
                    self._send_json({"project_slug": slug, "rounds": self.svc.list_rounds(slug)})
                    return
                if len(parts) == 4 and parts[3] == "cases":
                    self._send_json({"project_slug": slug, "cases": self.svc.list_cases(slug)})
                    return
                if len(parts) == 4 and parts[3] == "runs":
                    compact_query = parse_qs(parsed.query).get("compact") or [""]
                    compact_raw = str(compact_query[0]).strip().lower()
                    compact = compact_raw in {"1", "true", "yes", "y"}
                    records = self.svc.run_store.list_records(project_slug=slug)
                    runs = [
                        (item.to_compact_payload() if compact else item.to_payload())
                        for item in records
                    ]
                    self._send_json({"project_slug": slug, "runs": runs})
                    return
                if (
                    len(parts) == 5
                    and parts[3] == "diagnostics"
                    and parts[4] == "factor-correlation"
                ):
                    self._send_json(self.svc.project_factor_diagnostics(slug))
                    return
                if len(parts) == 5 and parts[3] == "runs":
                    run = self.svc.run_store.get(parts[4])
                    if run is None or run.project_slug != slug:
                        self._send_json(
                            {"ok": False, "error": f"run not found: {parts[4]}"},
                            status=HTTPStatus.NOT_FOUND,
                        )
                        return
                    self._send_json(run.to_payload())
                    return
                if len(parts) == 6 and parts[3] == "runs" and parts[5] == "overview":
                    self._handle_get_run_overview(slug=slug, run_id=parts[4])
                    return
                if len(parts) == 7 and parts[3] == "runs" and parts[5] == "artifact":
                    artifact_query = parse_qs(parsed.query or "")
                    download = str(artifact_query.get("download", ["0"])[0]).strip().lower() in {
                        "1",
                        "true",
                        "yes",
                    }
                    self._handle_get_run_artifact(
                        slug=slug,
                        run_id=parts[4],
                        artifact_key=parts[6],
                        download=download,
                    )
                    return
                if len(parts) == 4 and parts[3] == "drafts":
                    self._send_json({"project_slug": slug, "drafts": self.svc.list_drafts(slug)})
                    return
                if len(parts) == 5 and parts[3] == "drafts":
                    self._send_json(self.svc.read_draft(slug, parts[4]))
                    return
            except Exception as exc:
                self._send_error_payload(exc)
                return

        # Custom factor code: GET /api/custom-factors/{name}
        if len(parts) == 3 and parts[0] == "api" and parts[1] == "custom-factors":
            try:
                self._send_json(self.svc.get_custom_factor_code(parts[2]))
            except Exception as exc:
                self._send_error_payload(exc)
            return

        # Card read: GET /api/vault/card/{name}
        if len(parts) == 4 and parts[0] == "api" and parts[1] == "vault" and parts[2] == "card":
            try:
                self._send_json(self.svc.read_card(parts[3]))
            except Exception as exc:
                self._send_error_payload(exc)
            return

        self._send_json({"ok": False, "error": "not found"}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        payload = self._read_json_body_or_empty()
        parts = _path_parts(parsed.path)
        try:
            if parsed.path == "/api/vault/explore-idea":
                idea = str(payload.get("idea") or "").strip()
                mode = str(payload.get("mode") or "free").strip()
                project_slug = str(payload.get("project_slug") or "").strip() or None
                self._send_json(self.svc.explore_idea(idea, mode, project_slug))
                return
            if parsed.path == "/api/vault/preflight":
                self._send_json(self.svc.run_preflight_check(payload))
                return
            if parsed.path == "/api/custom-factors":
                self._send_json(self.svc.register_custom_factor(payload), status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/projects":
                created = self.svc.create_project(payload)
                self._send_json(created, status=HTTPStatus.CREATED)
                return
            if len(parts) >= 3 and parts[0] == "api" and parts[1] == "projects":
                slug = parts[2]
                if len(parts) == 4 and parts[3] == "refresh":
                    self._send_json(self.svc.refresh_project(slug))
                    return
                if len(parts) == 4 and parts[3] == "rounds":
                    self._send_json(self.svc.create_round(slug, payload), status=HTTPStatus.CREATED)
                    return
                if len(parts) == 5 and parts[3] == "drafts" and parts[4] == "patch":
                    draft_name = str(payload.get("draft_name") or "").strip()
                    if not draft_name:
                        raise ValueError("draft_name is required")
                    patch_payload = dict(payload)
                    patch_payload.pop("draft_name", None)
                    self._send_json(self.svc.patch_draft(slug, draft_name, patch_payload))
                    return
                if len(parts) == 4 and parts[3] == "cases":
                    self._send_json(self.svc.create_case(slug, payload), status=HTTPStatus.CREATED)
                    return
                if len(parts) == 4 and parts[3] == "runs":
                    self._send_json(self.svc.submit_run(slug, payload), status=HTTPStatus.CREATED)
                    return
                if len(parts) == 6 and parts[3] == "runs" and parts[5] == "summarize":
                    self._send_json(self.svc.summarize_run(slug, parts[4], payload))
                    return
                if len(parts) == 6 and parts[3] == "drafts" and parts[5] == "apply":
                    self._send_json(self.svc.apply_draft(slug, parts[4], payload))
                    return
        except Exception as exc:
            self._send_error_payload(exc)
            return
        self._send_json({"ok": False, "error": "not found"}, status=HTTPStatus.NOT_FOUND)

    def do_PATCH(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        parts = _path_parts(parsed.path)
        payload = self._read_json_body_or_empty()
        try:
            if len(parts) >= 3 and parts[0] == "api" and parts[1] == "projects":
                slug = parts[2]
                if len(parts) == 3:
                    self._send_json(self.svc.update_project_status(slug, payload))
                    return
                if len(parts) == 5 and parts[3] == "drafts":
                    self._send_json(self.svc.patch_draft(slug, parts[4], payload))
                    return
        except Exception as exc:
            self._send_error_payload(exc)
            return
        self._send_json({"ok": False, "error": "not found"}, status=HTTPStatus.NOT_FOUND)

    def do_DELETE(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        parts = _path_parts(parsed.path)
        try:
            # DELETE /api/custom-factors/{name}
            if len(parts) == 3 and parts[0] == "api" and parts[1] == "custom-factors":
                self._send_json(self.svc.delete_custom_factor(parts[2]))
                return
            # DELETE /api/projects/{slug}/runs/{run_id}
            if (
                len(parts) == 5
                and parts[0] == "api"
                and parts[1] == "projects"
                and parts[3] == "runs"
            ):
                self._send_json(self.svc.delete_run(parts[2], parts[4]))
                return
        except Exception as exc:
            self._send_error_payload(exc)
            return
        self._send_json({"ok": False, "error": "not found"}, status=HTTPStatus.NOT_FOUND)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003
        return

    def _handle_get_run_artifact(
        self,
        *,
        slug: str,
        run_id: str,
        artifact_key: str,
        download: bool = False,
    ) -> None:
        run = self.svc.run_store.get(run_id)
        if run is None or run.project_slug != slug:
            self._send_json(
                {"ok": False, "error": f"run not found: {run_id}"},
                status=HTTPStatus.NOT_FOUND,
            )
            return
        path_text = run.artifact_paths.get(artifact_key)
        if not path_text:
            self._send_json(
                {"ok": False, "error": f"artifact key not found: {artifact_key}"},
                status=HTTPStatus.NOT_FOUND,
            )
            return
        artifact_path = Path(path_text).resolve()
        if not artifact_path.exists() or not artifact_path.is_file():
            self._send_json(
                {"ok": False, "error": f"artifact file not found: {artifact_path}"},
                status=HTTPStatus.NOT_FOUND,
            )
            return
        file_size = artifact_path.stat().st_size
        ctype = _guess_content_type(artifact_path)
        # For text/JSON artifacts cap at a safe size; full tearsheet JSON gets
        # a higher ceiling than generic artifacts. Binary artifacts are served as-is.
        if "text" in ctype or "json" in ctype:
            raw = artifact_path.read_bytes()
            if len(raw) > _MAX_TEXT_BYTES:
                # Return JSON error instead of dumping huge content to browser
                self._send_json(
                    {
                        "error": "artifact too large to display inline",
                        "size_bytes": file_size,
                        "limit_bytes": _MAX_TEXT_BYTES,
                        "path": str(artifact_path),
                    },
                    status=HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                )
                return
            content = raw
        else:
            content = artifact_path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(content)))
        self.send_header(
            "Content-Disposition",
            f'{"attachment" if download else "inline"}; filename="{artifact_path.name}"',
        )
        self.end_headers()
        self.wfile.write(content)

    def _handle_get_run_overview(self, *, slug: str, run_id: str) -> None:
        run = self.svc.run_store.get(run_id)
        if run is None or run.project_slug != slug:
            self._send_json(
                {"ok": False, "error": f"run not found: {run_id}"},
                status=HTTPStatus.NOT_FOUND,
            )
            return
        snapshot = _build_run_overview_snapshot(run)
        self._send_json(
            {
                "ok": True,
                "project_slug": slug,
                "run_id": run_id,
                "summary": dict(run.summary),
                "snapshot": snapshot,
            }
        )

    def _read_json_body_or_empty(self) -> dict[str, object]:
        length_text = self.headers.get("Content-Length", "").strip()
        if not length_text:
            return {}
        try:
            length = int(length_text)
        except ValueError as exc:
            raise AlphaLabDataError("invalid Content-Length") from exc
        if length <= 0:
            return {}
        if length > _MAX_REQUEST_BODY_BYTES:
            raise AlphaLabDataError(
                f"request body too large: {length} bytes (limit {_MAX_REQUEST_BODY_BYTES})"
            )
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise AlphaLabDataError("invalid JSON body") from exc
        if not isinstance(payload, dict):
            raise AlphaLabDataError("JSON body must be an object")
        return payload

    def _send_json(
        self,
        payload: dict[str, object],
        *,
        status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        encoded = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _send_html(self, body: str) -> None:
        encoded = body.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.end_headers()
        self.wfile.write(encoded)

    def _send_error_payload(self, exc: Exception) -> None:
        status = HTTPStatus.BAD_REQUEST
        if isinstance(exc, FileNotFoundError):
            status = HTTPStatus.NOT_FOUND
        elif isinstance(exc, PermissionError):
            status = HTTPStatus.FORBIDDEN
        payload = {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "trace": traceback.format_exc(limit=6),
        }
        self._send_json(payload, status=status)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _project_paths(vault_root: Path, slug: str) -> dict[str, Path]:
    safe_slug = _safe_slug(slug)
    project_dir = (vault_root / PROJECTS_DIRNAME / safe_slug).resolve()
    project_file = project_dir / "project.md"
    if not project_file.exists() and (project_dir / "project.yaml").exists():
        project_file = project_dir / "project.yaml"
    current_case_file = project_dir / "current_case.md"
    if not current_case_file.exists() and (project_dir / "current_case.yaml").exists():
        current_case_file = project_dir / "current_case.yaml"
    return {
        "project_dir": project_dir,
        "project_yaml": project_file,
        "current_case": current_case_file,
        "latest_run": project_dir / "runs" / "latest.md",
        "rounds_dir": project_dir / "30_rounds",
        "decision_log": project_dir / "decision_log.md",
        "runs_dir": project_dir / "runs",
        "drafts_dir": project_dir / "50_writeback_drafts",
    }


def _iter_project_contracts(root: Path) -> list[Path]:
    rows: list[Path] = []
    seen: set[Path] = set()
    for candidate in sorted(root.glob("*/project.md")) + sorted(root.glob("*/project.yaml")):
        resolved = candidate.resolve()
        project_dir = resolved.parent
        if project_dir in seen:
            continue
        seen.add(project_dir)
        rows.append(resolved)
    return rows


def _list_cases(paths: dict[str, Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    current_case_path = paths["current_case"]
    if current_case_path.exists():
        case_name = _yaml_case_name(current_case_path) or current_case_path.stem
        rows.append(
            {
                "case_name": case_name,
                "spec_path": str(current_case_path),
                "handoff_path": "",
                "spec_exists": True,
                "handoff_exists": False,
                "is_current": True,
            }
        )
    return rows


def _list_rounds(rounds_dir: Path) -> list[dict[str, object]]:
    if not rounds_dir.exists():
        return []
    rows: list[dict[str, object]] = []
    for round_dir in sorted([item for item in rounds_dir.iterdir() if item.is_dir()]):
        round_id = round_dir.name
        discussion_path = round_dir / "discussion_capture.md"
        rows.append(
            {
                "round_id": round_id,
                "path": str(round_dir),
                "discussion_capture_path": str(discussion_path),
                "has_discussion_capture": discussion_path.exists(),
                "has_feedback": (round_dir / "latest_experiment_feedback.md").exists(),
                "files": {
                    "round_context_digest": str(round_dir / "round_context_digest.md"),
                    "round_prompt": str(round_dir / "round_prompt.md"),
                    "web_search_tasks": str(round_dir / "web_search_tasks.md"),
                    "discussion_capture": str(discussion_path),
                    "latest_experiment_feedback": str(round_dir / "latest_experiment_feedback.md"),
                },
            }
        )
    return rows


def _resolve_case_spec_path(paths: dict[str, Path], case_name: str) -> Path:
    current_case = paths["current_case"]
    if current_case.exists():
        current_name = _yaml_case_name(current_case)
        if current_name == case_name or current_case.stem == case_name:
            return current_case
    return current_case


def _yaml_case_name(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        payload = load_yaml_document(path)
    except Exception:
        return ""
    return str(payload.get("name") or "").strip()


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _list_draft_summaries(drafts_dir: Path) -> list[dict[str, object]]:
    if not drafts_dir.exists():
        return []
    rows = []
    for draft_path in sorted(drafts_dir.glob("*__writeback_draft.md"), reverse=True):
        rows.append(_draft_summary(draft_path))
    return rows


def _draft_summary(draft_path: Path) -> dict[str, object]:
    frontmatter, _ = _load_markdown_with_frontmatter(draft_path)
    return {
        "name": draft_path.name,
        "path": str(draft_path),
        "project": str(frontmatter.get("project") or ""),
        "round_id": str(frontmatter.get("round_id") or ""),
        "case_name": str(frontmatter.get("case_name") or ""),
        "review_status": str(frontmatter.get("review_status") or ""),
        "reviewed_by": str(frontmatter.get("reviewed_by") or ""),
        "reviewed_at": str(frontmatter.get("reviewed_at") or ""),
        "vault_export_mode": str(frontmatter.get("vault_export_mode") or ""),
    }


def _compose_markdown_with_frontmatter(frontmatter: dict[str, object], body: str) -> str:
    yaml = _require_yaml()
    rendered_frontmatter = yaml.safe_dump(frontmatter, sort_keys=False, allow_unicode=True).strip()
    return f"---\n{rendered_frontmatter}\n---\n\n{body.rstrip()}\n"


def _load_markdown_with_frontmatter(path: Path) -> tuple[dict[str, object], str]:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        raise AlphaLabDataError(f"markdown draft is missing YAML frontmatter: {path}")
    try:
        _, raw_frontmatter, body = text.split("---\n", 2)
    except ValueError as exc:
        raise AlphaLabDataError(f"markdown draft has invalid frontmatter fence: {path}") from exc
    yaml = _require_yaml()
    payload = yaml.safe_load(raw_frontmatter)
    if not isinstance(payload, dict):
        raise AlphaLabDataError(f"markdown draft frontmatter must be an object: {path}")
    return payload, body.lstrip("\n")


def _resolve_draft_path(drafts_dir: Path, draft_name: str) -> Path:
    candidate = (drafts_dir / draft_name).resolve()
    if not str(candidate).startswith(str(drafts_dir.resolve())):
        raise PermissionError("invalid draft path")
    if not candidate.exists():
        raise FileNotFoundError(f"draft not found: {candidate}")
    return candidate


def _read_text_preview(path: Path, *, limit_bytes: int) -> str:
    if not path.exists():
        return ""
    file_size = path.stat().st_size
    with path.open("rb") as fh:
        raw = fh.read(limit_bytes + 1)
    truncated = len(raw) > limit_bytes or file_size > limit_bytes
    content = raw[:limit_bytes].decode("utf-8", errors="replace")
    if not truncated:
        return content
    size_kb = max(1, round(file_size / 1024))
    limit_kb = max(1, round(limit_bytes / 1024))
    return content.rstrip() + f"\n\n> [内容已截断 — 显示前 {limit_kb} KB，文件共 {size_kb} KB]"


def _extract_metrics_summary(metrics_path: Path | None) -> dict[str, object]:
    if metrics_path is None or not metrics_path.exists():
        return {}
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    source = payload.get("metrics")
    if isinstance(source, dict):
        metrics = source
    else:
        metrics = payload
    # Keep all available metric fields so the UI can:
    # 1) render a strict compact set in quick-screen mode, and
    # 2) render a full metric surface in full-evaluation mode.
    return {str(key): value for key, value in metrics.items()}


def _read_json_artifact(path: Path | None) -> dict[str, object] | None:
    if path is None:
        return None
    try:
        raw = path.read_bytes()
    except Exception:
        return None
    if len(raw) > _MAX_TEXT_BYTES:
        return None
    try:
        payload = json.loads(raw.decode("utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _read_csv_artifact_rows(path: Path | None) -> list[dict[str, str]]:
    if path is None:
        return []
    rows: list[dict[str, str]] = []
    try:
        with path.open("r", encoding="utf-8") as fh:
            reader = DictReader(fh)
            for idx, row in enumerate(reader):
                if idx >= _RUN_OVERVIEW_MAX_CSV_ROWS:
                    break
                if not isinstance(row, dict):
                    continue
                normalized: dict[str, str] = {}
                for key, value in row.items():
                    name = str(key or "").strip()
                    if not name:
                        continue
                    normalized[name] = "" if value is None else str(value)
                if normalized:
                    rows.append(normalized)
    except Exception:
        return []
    return rows


def _build_run_overview_snapshot(run: _RunRecord) -> dict[str, object]:
    group_rows = _read_csv_artifact_rows(
        _resolve_run_artifact_path(
            run,
            artifact_key="group_returns",
            fallback_name="group_returns.csv",
        )
    )
    if not group_rows:
        group_rows = _read_csv_artifact_rows(
            _resolve_run_artifact_path(
                run,
                artifact_key="quantile_returns",
                fallback_name="quantile_returns.csv",
            )
        )
    return {
        "backtest": _read_json_artifact(
            _resolve_run_artifact_path(
                run,
                artifact_key="backtest_result_json",
                fallback_name="backtest_result.json",
            )
        ),
        "icRows": _read_csv_artifact_rows(
            _resolve_run_artifact_path(
                run,
                artifact_key="ic_timeseries",
                fallback_name="ic_timeseries.csv",
            )
        ),
        "rollingRows": _read_csv_artifact_rows(
            _resolve_run_artifact_path(
                run,
                artifact_key="rolling_stability",
                fallback_name="rolling_stability.csv",
            )
        ),
        "decayRows": _read_csv_artifact_rows(
            _resolve_run_artifact_path(
                run,
                artifact_key="ic_decay",
                fallback_name="ic_decay.csv",
            )
        ),
        "groupRows": group_rows,
        "autocorrRows": _read_csv_artifact_rows(
            _resolve_run_artifact_path(
                run,
                artifact_key="factor_autocorrelation",
                fallback_name="factor_autocorrelation.csv",
            )
        ),
        "turnoverRows": _read_csv_artifact_rows(
            _resolve_run_artifact_path(
                run,
                artifact_key="turnover",
                fallback_name="turnover.csv",
            )
        ),
        "coverageRows": _read_csv_artifact_rows(
            _resolve_run_artifact_path(
                run,
                artifact_key="coverage",
                fallback_name="coverage.csv",
            )
        ),
    }


def _load_run_rank_ic_timeseries(run: _RunRecord) -> dict[str, float]:
    path = _resolve_run_artifact_path(
        run, artifact_key="ic_timeseries", fallback_name="ic_timeseries.csv"
    )
    if path is None:
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            reader = DictReader(fh)
            rows: dict[str, float] = {}
            for row in reader:
                date = str((row or {}).get("date") or "").strip()
                if not date:
                    continue
                raw = (row or {}).get("rank_ic")
                value = _coerce_finite_float(raw)
                if value is None:
                    value = _coerce_finite_float((row or {}).get("ic"))
                if value is None:
                    continue
                rows[date] = value
            return rows
    except Exception:
        return {}


def _resolve_run_artifact_path(
    run: _RunRecord,
    *,
    artifact_key: str,
    fallback_name: str,
) -> Path | None:
    path_text = run.artifact_paths.get(artifact_key)
    if path_text:
        path = Path(path_text).expanduser().resolve()
        if path.exists() and path.is_file():
            return path
    if run.output_dir:
        fallback = Path(run.output_dir).expanduser().resolve() / fallback_name
        if fallback.exists() and fallback.is_file():
            return fallback
    return None


def _resolve_run_factor_label(run: _RunRecord) -> str:
    summary_name = str(run.summary.get("factor_name") or "").strip()
    if summary_name:
        return summary_name
    spec_path = Path(run.spec_path).expanduser().resolve()
    if spec_path.exists():
        try:
            payload = load_yaml_document(spec_path)
            factor_name = str(payload.get("factor_name") or "").strip()
            if factor_name:
                return factor_name
        except Exception:
            pass
    return run.case_name


def _resolve_run_dsr_pvalue(run: _RunRecord) -> float | None:
    summary_value = _coerce_finite_float(run.summary.get("dsr_pvalue"))
    if summary_value is not None:
        return summary_value
    metrics_path = _resolve_run_artifact_path(
        run,
        artifact_key="metrics",
        fallback_name="metrics.json",
    )
    if metrics_path is None:
        return None
    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    source = payload.get("metrics")
    metrics = source if isinstance(source, dict) else payload
    if not isinstance(metrics, dict):
        return None
    return _coerce_finite_float(metrics.get("dsr_pvalue"))


def _classify_dsr_pvalue(value: float) -> str:
    if value <= 0.10:
        return "robust"
    if value >= 0.50:
        return "high_risk"
    return "watch"


def _build_project_dsr_summary(
    rows: list[dict[str, object]],
    *,
    n_runs_total: int,
) -> dict[str, object]:
    values = [
        numeric
        for numeric in (_coerce_finite_float(item.get("dsr_pvalue")) for item in rows)
        if numeric is not None
    ]
    values_sorted = sorted(values)
    median: float | None = None
    if values_sorted:
        mid = len(values_sorted) // 2
        if len(values_sorted) % 2 == 1:
            median = values_sorted[mid]
        else:
            median = (values_sorted[mid - 1] + values_sorted[mid]) / 2.0
    robust_count = sum(1 for value in values_sorted if value <= 0.10)
    high_risk_count = sum(1 for value in values_sorted if value >= 0.50)
    return {
        "n_runs_total": n_runs_total,
        "n_with_dsr": len(values_sorted),
        "coverage_ratio": (len(values_sorted) / n_runs_total if n_runs_total > 0 else None),
        "median_dsr_pvalue": median,
        "min_dsr_pvalue": values_sorted[0] if values_sorted else None,
        "max_dsr_pvalue": values_sorted[-1] if values_sorted else None,
        "robust_count": robust_count,
        "watch_count": len(values_sorted) - robust_count - high_risk_count,
        "high_risk_count": high_risk_count,
    }


def _coerce_finite_float(value: object) -> float | None:
    if isinstance(value, bool):
        numeric = float(value)
    elif isinstance(value, (int, float)):
        numeric = float(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            numeric = float(text)
        except ValueError:
            return None
    else:
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _pairwise_spearman_from_timeseries(
    left: dict[str, float],
    right: dict[str, float],
    *,
    min_overlap: int,
) -> tuple[float | None, int]:
    overlap_dates = sorted(set(left.keys()) & set(right.keys()))
    if len(overlap_dates) < min_overlap:
        return None, len(overlap_dates)
    left_values = [left[date] for date in overlap_dates]
    right_values = [right[date] for date in overlap_dates]
    corr = _spearman_correlation(left_values, right_values)
    return corr, len(overlap_dates)


def _spearman_correlation(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_rank = _average_ranks(left)
    right_rank = _average_ranks(right)
    return _pearson_correlation(left_rank, right_rank)


def _average_ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        # Average rank for ties. Rank base is 1 to match Spearman convention.
        rank = ((i + 1) + j) / 2.0
        for k in range(i, j):
            original_idx = indexed[k][0]
            ranks[original_idx] = rank
        i = j
    return ranks


def _pearson_correlation(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    n = float(len(left))
    mean_left = sum(left) / n
    mean_right = sum(right) / n
    centered_left = [value - mean_left for value in left]
    centered_right = [value - mean_right for value in right]
    denom_left = math.sqrt(sum(value * value for value in centered_left))
    denom_right = math.sqrt(sum(value * value for value in centered_right))
    if denom_left == 0.0 or denom_right == 0.0:
        return None
    numerator = sum(a * b for a, b in zip(centered_left, centered_right, strict=True))
    return numerator / (denom_left * denom_right)


def _as_text_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _optional_text(value: object) -> str | None:
    text = str(value or "").strip()
    return text or None


def _as_int(value: object, *, default: int) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return default
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return default
        try:
            return int(text)
        except ValueError:
            return default
    return default


def _coerce_text_list(value: object, *, delimiter: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(delimiter) if part.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _safe_slug(value: str) -> str:
    raw = str(value).strip()
    if not raw:
        raise AlphaLabConfigError("slug must be non-empty")
    normalized = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "-" for ch in raw)
    normalized = normalized.strip("._-")
    if not normalized:
        raise AlphaLabConfigError(f"slug is invalid: {value!r}")
    return normalized


def _path_parts(path: str) -> list[str]:
    return [unquote(part) for part in path.split("/") if part]


def _guess_content_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return "application/json; charset=utf-8"
    if suffix in {".md", ".txt", ".csv", ".log", ".yaml", ".yml"}:
        return "text/plain; charset=utf-8"
    if suffix in {".html", ".htm"}:
        return "text/html; charset=utf-8"
    if suffix == ".pdf":
        return "application/pdf"
    return "application/octet-stream"


def _safe_limit(value: str, *, default: int) -> int:
    try:
        parsed = int(value)
    except ValueError:
        return default
    return max(1, min(parsed, 200))


def _utc_now_iso() -> str:
    return dt.datetime.now(dt.UTC).isoformat().replace("+00:00", "Z")


def _file_mtime_ns(path_text: str | None) -> int:
    if path_text is None:
        return -1
    try:
        return int(Path(path_text).stat().st_mtime_ns)
    except OSError:
        return -1


def _require_yaml() -> Any:
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover
        raise AlphaLabExperimentError("PyYAML is required for draft editing") from exc
    return yaml


def _compile_custom_factor(name: str, code: str) -> Any:
    """Compile user-provided Python code into a callable factor builder.

    The code must define a function named ``builder`` that accepts
    ``(prices, *, window=20, skip_recent=0, min_periods=None, **kwargs)``
    and returns a DataFrame with columns ``[date, asset, factor, value]``.
    """
    import numpy as np  # noqa: F401
    import pandas as pd  # noqa: F401

    namespace: dict[str, Any] = {"np": np, "pd": pd}
    try:
        compiled = compile(code, f"<custom_factor:{name}>", "exec")
    except SyntaxError as exc:
        raise ValueError(f"syntax error in custom factor code: {exc}") from exc
    exec(compiled, namespace)  # noqa: S102
    fn = namespace.get("builder")
    if fn is None or not callable(fn):
        raise ValueError(
            "custom factor code must define a callable named 'builder'; "
            "e.g. def builder(prices, *, window=20, **kwargs): ..."
        )
    return fn


# ---------------------------------------------------------------------------
# HTML Frontend — 5-page single-page app
# ---------------------------------------------------------------------------


_MD_RENDER_JS_PATH = Path(__file__).with_name("web_unified_md_render.js")
_MD_RENDER_JS: str | None = None


def _md_render_js() -> str:
    """Load mdRender JS function from file with in-memory caching."""
    global _MD_RENDER_JS
    if _MD_RENDER_JS is None:
        _MD_RENDER_JS = _MD_RENDER_JS_PATH.read_text(encoding="utf-8")
    return _MD_RENDER_JS


# Cached inline HTML template cache path.
_INDEX_HTML_TEMPLATE_PATH = Path(__file__).with_name("web_unified_index.html")
_INDEX_HTML_TEMPLATE: str | None = None


def _load_index_html_template() -> str:
    """Load frontend HTML template with in-memory caching."""
    global _INDEX_HTML_TEMPLATE
    if _INDEX_HTML_TEMPLATE is None:
        _INDEX_HTML_TEMPLATE = _INDEX_HTML_TEMPLATE_PATH.read_text(encoding="utf-8")
    return _INDEX_HTML_TEMPLATE


def _index_html() -> str:
    return _index_html_raw().replace("@@MD_RENDER_JS@@", _md_render_js())


def _index_html_raw() -> str:
    return _load_index_html_template()
