"""Run record + run store for the unified web frontend.

Holds:

- ``_RunRecord``: dataclass for a single run's mutable state (status,
  progress, artifacts, summary).
- ``_InputBundleCacheEntry`` + ``_RunStore``: in-memory queue, dispatcher
  thread, single-factor + model-factor execution drivers, input-bundle
  cache.

The few symbols this module needs from the rest of ``web_unified``
(``_resolve_run_artifact_for_endpoint``, ``_extract_metrics_summary``,
``_build_run_error_payload``, ``_safe_rmtree``, ``_utc_now_iso``,
``_file_mtime_ns``) are imported lazily inside methods to avoid the
circular import that would arise if we pulled them at module load time —
``__init__.py`` is still executing when this module first loads.
"""

from __future__ import annotations

import gc
import subprocess
import threading
import time
import traceback
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from alpha_lab.real_cases.single_factor.pipeline import (
    SingleFactorCaseRunResult,
)
from alpha_lab.real_cases.single_factor.spec import (
    load_single_factor_case_spec,
)
from alpha_lab.reporting.renderers import write_case_report
from alpha_lab.web_unified._models import (
    RunStatus,
    RunWorkflow,
    _ModelLabSubprocessError,
    _RunTask,
    _SubprocessCaseRunResult,
)
from alpha_lab.web_unified._subprocess import (
    _annotate_exception_with_model_lab_subprocess_artifacts,
    _build_model_lab_batch_worker_count,
    _build_model_lab_subprocess_command,
    _build_model_lab_subprocess_env,
    _build_single_factor_subprocess_command,
    _build_single_factor_subprocess_env,
    _format_model_lab_subprocess_failure,
    _format_run_error_text,
    _load_model_factor_artifact_paths_from_manifest,
    _load_single_factor_artifact_paths_from_manifest,
    _model_lab_subprocess_failure_hint,
    _parse_time_peak_rss_kb,
    _read_text_tail,
    _resolve_model_factor_web_output_parts,
    _resolve_single_factor_web_output_root_dir,
    _wrap_command_with_time,
)
from alpha_lab.web_unified._utils import _coerce_finite_or_text

_FRONTEND_BATCH_WINDOW_SECONDS: float = 0.20
_MODEL_LAB_SUBPROCESS_POLL_SECONDS: float = 0.5

_RUN_SUMMARY_COMPACT_KEYS: tuple[str, ...] = (
    "research_evaluation_profile",
    "factor_name",
    "factor_verdict",
    "campaign_triage",
    "promotion_decision",
    "portfolio_validation_recommendation",
    "level12_transition_label",
    "evaluation_title",
    "evaluation_action",
    "evaluation_next_step",
    "model_family",
    "mean_ic",
    "mean_ic_full",
    "mean_ic_is",
    "mean_ic_oos",
    "mean_ic_oos_decay_ratio",
    "mean_rank_ic",
    "mean_rank_ic_full",
    "mean_rank_ic_is",
    "mean_rank_ic_oos",
    "mean_rank_ic_oos_decay_ratio",
    "rank_ic_ir",
    "rank_ic_ir_full",
    "rank_ic_ir_is",
    "rank_ic_ir_oos",
    "rank_ic_ir_oos_decay_ratio",
    "ic_ir",
    "ic_ir_full",
    "ic_ir_is",
    "ic_ir_oos",
    "ic_ir_oos_decay_ratio",
    "ic_positive_rate",
    "ic_positive_rate_full",
    "ic_positive_rate_is",
    "ic_positive_rate_oos",
    "rank_ic_positive_rate",
    "rank_ic_positive_rate_full",
    "rank_ic_positive_rate_is",
    "rank_ic_positive_rate_oos",
    "group_monotonicity_summary",
    "group_monotonicity_share",
    "monotonic_share",
    "group_spread_summary",
    "ic_decay_half_life_summary",
    "ic_decay_retention_5_over_1",
    "ic_half_life_summary",
    "ic_half_life_horizon",
    "mean_long_short_return",
    "mean_long_short_return_full",
    "mean_long_short_return_is",
    "mean_long_short_return_oos",
    "mean_long_short_return_oos_decay_ratio",
    "mean_long_short_turnover",
    "mean_long_short_turnover_full",
    "mean_long_short_turnover_is",
    "mean_long_short_turnover_oos",
    "cost_aware_long_short_sharpe",
    "cost_aware_long_short_ir",
    "cost_aware_long_short_ir_full",
    "cost_aware_long_short_ir_is",
    "cost_aware_long_short_ir_oos",
    "cost_aware_long_short_ir_oos_decay_ratio",
    "long_short_ir",
    "long_short_ir_full",
    "long_short_ir_is",
    "long_short_ir_oos",
    "long_short_ir_oos_decay_ratio",
    "raw_long_short_ir",
    "ic_t_stat",
    "max_drawdown",
    "max_drawdown_full",
    "max_drawdown_is",
    "max_drawdown_oos",
    "ls_max_drawdown",
    "coverage_summary",
    "coverage_break_days",
    "n_dates_used",
    "mean_eval_assets_per_date",
    "eval_coverage_ratio_mean",
)


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
    evaluation_profile_source: str = "request"
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
    workflow: RunWorkflow = "single_factor"
    note: str | None = None
    draft_model_candidate_path: str | None = None
    draft_model_candidate_name: str | None = None
    draft_model_candidate_hash: str | None = None

    def _artifact_paths_for_api(self) -> dict[str, str]:
        """Return artifact paths that are actually retrievable by endpoint."""
        # Lazy import — _resolve_run_artifact_for_endpoint lives in
        # ``__init__`` (still executing when this module first loads).
        from alpha_lab.web_unified import _resolve_run_artifact_for_endpoint

        resolved_paths: dict[str, str] = {}
        for raw_key in self.artifact_paths.keys():
            key = str(raw_key or "").strip()
            if not key:
                continue
            resolved = _resolve_run_artifact_for_endpoint(self, key)
            if resolved is None:
                continue
            resolved_paths[key] = str(resolved)
        return resolved_paths

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
            evaluation_profile_source=self.evaluation_profile_source,
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
            workflow=self.workflow,
            note=self.note,
            draft_model_candidate_path=self.draft_model_candidate_path,
            draft_model_candidate_name=self.draft_model_candidate_name,
            draft_model_candidate_hash=self.draft_model_candidate_hash,
        )

    def to_payload(self) -> dict[str, object]:
        artifact_paths = self._artifact_paths_for_api()
        return {
            "run_id": self.run_id,
            "project_slug": self.project_slug,
            "case_name": self.case_name,
            "round_id": self.round_id,
            "spec_path": self.spec_path,
            "submitted_at_utc": self.submitted_at_utc,
            "evaluation_profile": self.evaluation_profile,
            "evaluation_profile_source": self.evaluation_profile_source,
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
            "artifact_paths": artifact_paths,
            "summary": dict(self.summary),
            "summarize_feedback_path": self.summarize_feedback_path,
            "summarize_draft_path": self.summarize_draft_path,
            "summarize_state_patch_path": self.summarize_state_patch_path,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "error_hint": self.error_hint,
            "error": self.error,
            "workflow": self.workflow,
            "note": self.note,
            "draft_model_candidate_path": self.draft_model_candidate_path,
            "draft_model_candidate_name": self.draft_model_candidate_name,
            "draft_model_candidate_hash": self.draft_model_candidate_hash,
        }

    def to_compact_payload(self) -> dict[str, object]:
        # Lightweight payload for run polling.
        artifact_paths = self._artifact_paths_for_api()
        return {
            "run_id": self.run_id,
            "project_slug": self.project_slug,
            "case_name": self.case_name,
            "round_id": self.round_id,
            "spec_path": self.spec_path,
            "submitted_at_utc": self.submitted_at_utc,
            "evaluation_profile": self.evaluation_profile,
            "evaluation_profile_source": self.evaluation_profile_source,
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
            "artifact_paths": {key: True for key in artifact_paths.keys()},
            "summary": _compact_metrics_summary(self.summary),
            "summarize_feedback_path": self.summarize_feedback_path,
            "summarize_draft_path": self.summarize_draft_path,
            "summarize_state_patch_path": self.summarize_state_patch_path,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "error_hint": self.error_hint,
            "error": self.error,
            "_compact": True,
            "workflow": self.workflow,
            "note": self.note,
            "draft_model_candidate_path": self.draft_model_candidate_path,
            "draft_model_candidate_name": self.draft_model_candidate_name,
            "draft_model_candidate_hash": self.draft_model_candidate_hash,
        }


RunSuccessResult = SingleFactorCaseRunResult | _SubprocessCaseRunResult


class _RunStore:
    def __init__(self) -> None:
        self._records: dict[str, _RunRecord] = {}
        self._tasks: dict[str, _RunTask] = {}
        self._cancel_requests: set[str] = set()
        self._lock = threading.Lock()
        self._dispatch_event = threading.Event()
        self._dispatcher = threading.Thread(target=self._dispatch_loop, daemon=True)
        self._dispatcher.start()

    def submit(self, task: _RunTask) -> _RunRecord:
        from alpha_lab.web_unified import _utc_now_iso

        submitted_at = _utc_now_iso()
        record = _RunRecord(
            run_id=task.run_id,
            project_slug=task.project_slug,
            case_name=task.case_name,
            round_id=task.round_id,
            spec_path=task.spec_path,
            submitted_at_utc=submitted_at,
            evaluation_profile=task.evaluation_profile,
            evaluation_profile_source=task.evaluation_profile_source,
            output_root_dir=task.output_root_dir,
            render_report=task.render_report,
            workflow=task.workflow,
            note=task.note,
            draft_model_candidate_path=task.draft_model_candidate_path,
            draft_model_candidate_name=task.draft_model_candidate_name,
            draft_model_candidate_hash=task.draft_model_candidate_hash,
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

    def restore_completed(self, record: _RunRecord) -> None:
        with self._lock:
            if record.run_id in self._records:
                return
            self._records[record.run_id] = record

    def get(self, run_id: str) -> _RunRecord | None:
        with self._lock:
            record = self._records.get(run_id)
            if record is None:
                return None
            self._hydrate_summary_locked(record)
            return record.clone()

    def list_records(
        self,
        *,
        project_slug: str | None = None,
        workflow: RunWorkflow | None = None,
    ) -> list[_RunRecord]:
        with self._lock:
            for record in self._records.values():
                self._hydrate_summary_locked(record)
            records = [rec.clone() for rec in self._records.values()]
        if workflow is not None:
            records = [item for item in records if item.workflow == workflow]
        if project_slug is None:
            return sorted(records, key=lambda item: item.submitted_at_utc, reverse=True)
        filtered = [item for item in records if item.project_slug == project_slug]
        return sorted(filtered, key=lambda item: item.submitted_at_utc, reverse=True)

    def _hydrate_summary_locked(self, record: _RunRecord) -> None:
        """Backfill run.summary from metrics.json when summary is missing.

        This keeps the run table robust for runs created by older code paths
        where summary extraction might not have been stored in memory.
        """
        from alpha_lab.web_unified import _extract_metrics_summary

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
            record.summary = _extract_metrics_summary(
                metrics_path,
                run_status=record.status,
            )
        except Exception:
            # Keep run listing resilient even if one metrics file is malformed.
            return

    def delete(self, run_id: str) -> _RunRecord | None:
        with self._lock:
            record = self._records.pop(run_id, None)
            self._tasks.pop(run_id, None)
            self._cancel_requests.discard(run_id)
            return record.clone() if record is not None else None

    def request_cancel_and_delete(self, run_id: str) -> dict[str, object]:
        """Cancel a run and request deletion.

        Queued / terminal runs are removed immediately. Running runs are flagged
        for cancellation — the worker finalizer sees the flag, skips state
        updates, and cleans up the output_dir once the current stage returns.
        """
        with self._lock:
            record = self._records.get(run_id)
            if record is None:
                self._tasks.pop(run_id, None)
                self._cancel_requests.discard(run_id)
                return {"immediate": True, "prior_status": None, "output_dir": None}
            self._tasks.pop(run_id, None)
            prior_status = record.status
            output_dir = record.output_dir
            if prior_status == "running":
                self._cancel_requests.add(run_id)
                record.status = "cancelled"
                self._push_progress_locked(
                    record,
                    message="已请求取消：当前阶段结束后将自动清理产物",
                    percent=record.progress_percent,
                )
                return {
                    "immediate": False,
                    "prior_status": prior_status,
                    "output_dir": output_dir,
                }
            self._records.pop(run_id, None)
            self._cancel_requests.discard(run_id)
            return {
                "immediate": True,
                "prior_status": prior_status,
                "output_dir": output_dir,
            }

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
        from alpha_lab.web_unified import _utc_now_iso

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
        from alpha_lab.web_unified import _utc_now_iso

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
            grouped: dict[tuple[RunWorkflow, str, str], list[_RunTask]] = {}
            for task, _record in sorted(
                queued_with_records,
                key=lambda item: item[1].submitted_at_utc,
            ):
                key = (
                    task.workflow,
                    task.evaluation_profile,
                    task.output_root_dir or "",
                )
                grouped.setdefault(key, []).append(task)
            ordered_groups = list(grouped.values())
            for tasks in ordered_groups:
                workflow = tasks[0].workflow
                batch_message = (
                    "已进入前端调度窗口，按 run_id 隔离产物并复用输入缓存"
                    if len(tasks) > 1 and workflow == "single_factor"
                    else (
                        "已进入模型批量调度窗口，准备并行执行"
                        if len(tasks) > 1 and workflow == "model_factor"
                        else (
                            "任务已启动，准备执行模型因子训练与评估"
                            if workflow == "model_factor"
                            else "任务已启动，准备执行 single-factor pipeline"
                        )
                    )
                )
                batch_percent = (
                    1 if len(tasks) > 1 and workflow in {"single_factor", "model_factor"} else 2
                )
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
        if tasks and tasks[0].workflow == "model_factor":
            self._execute_model_factor_task_group(tasks)
            return
        self._execute_single_factor_task_group(tasks)

    def _execute_single_factor_task_group(self, tasks: list[_RunTask]) -> None:
        if len(tasks) > 1:
            for task in tasks:
                self._push_progress(
                    task.run_id,
                    message="按 run_id 隔离输出目录，逐个执行以避免同名 case 覆盖历史产物",
                    percent=4,
                )
        for task in tasks:
            self._execute_single_task(task, allow_fallback=False)

    def _execute_model_factor_task_group(self, tasks: list[_RunTask]) -> None:
        if len(tasks) <= 1:
            if tasks:
                self._execute_single_task(tasks[0], allow_fallback=False)
            return

        if self._model_factor_batch_has_output_conflict(tasks):
            for task in tasks:
                self._push_progress(
                    task.run_id,
                    message="检测到相同 output_dir，为避免产物互相覆盖，自动回退到串行执行",
                    percent=4,
                )
            for task in tasks:
                self._execute_single_task(task, allow_fallback=False)
            return

        worker_slots = _build_model_lab_batch_worker_count(len(tasks))
        if worker_slots <= 1:
            for task in tasks:
                self._execute_single_task(task, allow_fallback=False)
            return

        batch_message = (
            f"模型批量调度命中，共 {len(tasks)} 个实验，并行 workers={worker_slots}"
        )
        for task in tasks:
            self._push_progress(task.run_id, message=batch_message, percent=4)

        with ThreadPoolExecutor(max_workers=worker_slots) as executor:
            futures = [
                executor.submit(self._execute_single_task, task, allow_fallback=False)
                for task in tasks
            ]
            for future in futures:
                future.result()

    def _model_factor_batch_has_output_conflict(self, tasks: list[_RunTask]) -> bool:
        seen_output_dirs: set[str] = set()
        for task in tasks:
            output_dir = self._resolve_model_factor_task_output_dir(task)
            if output_dir in seen_output_dirs:
                return True
            seen_output_dirs.add(output_dir)
        return False

    def _resolve_model_factor_task_output_dir(self, task: _RunTask) -> str:
        root_dir, case_dir_name = _resolve_model_factor_web_output_parts(task)
        return str((root_dir / case_dir_name).resolve())

    def _execute_model_factor_subprocess_task(
        self,
        task: _RunTask,
        *,
        progress_callback: Any,
    ) -> _SubprocessCaseRunResult:
        from alpha_lab.web_unified import _utc_now_iso, _write_json_file

        spec_path = Path(task.spec_path).expanduser().resolve()
        output_dir = Path(self._resolve_model_factor_task_output_dir(task)).expanduser().resolve()
        log_dir = output_dir / "_web_run_logs" / task.run_id
        log_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = log_dir / "stdout.log"
        stderr_path = log_dir / "stderr.log"
        status_path = log_dir / "status.json"
        artifact_paths = {
            "subprocess_stdout": stdout_path,
            "subprocess_stderr": stderr_path,
            "subprocess_status": status_path,
        }
        cli_cmd = _build_model_lab_subprocess_command(task=task, spec_path=spec_path)
        cmd = _wrap_command_with_time(cli_cmd)
        started_at = _utc_now_iso()
        _write_json_file(
            status_path,
            {
                "status": "starting",
                "run_id": task.run_id,
                "case_name": task.case_name,
                "workflow": task.workflow,
                "started_at_utc": started_at,
                "command": cli_cmd,
                "effective_command": cmd,
                "cwd": str(Path.cwd().resolve()),
                "stdout_log": str(stdout_path),
                "stderr_log": str(stderr_path),
            },
        )
        with self._lock:
            record = self._records.get(task.run_id)
            if record is not None:
                record.output_dir = str(output_dir)
                record.artifact_paths.update(
                    {key: str(path) for key, path in artifact_paths.items()}
                )
                self._push_progress_locked(
                    record,
                    message="已启动隔离子进程执行模型因子实验",
                    percent=8,
                )

        env = _build_model_lab_subprocess_env()
        start_monotonic = time.monotonic()
        try:
            with stdout_path.open("ab") as stdout, stderr_path.open("ab") as stderr:
                proc = subprocess.Popen(  # noqa: S603 - argv is built without shell=True.
                    cmd,
                    cwd=str(Path.cwd().resolve()),
                    env=env,
                    stdout=stdout,
                    stderr=stderr,
                )
                _write_json_file(
                    status_path,
                    {
                        "status": "running",
                        "run_id": task.run_id,
                        "case_name": task.case_name,
                        "pid": proc.pid,
                        "started_at_utc": started_at,
                        "command": cli_cmd,
                        "effective_command": cmd,
                        "cwd": str(Path.cwd().resolve()),
                        "stdout_log": str(stdout_path),
                        "stderr_log": str(stderr_path),
                    },
                )
                progress_callback(
                    message=(
                        f"模型因子子进程运行中 pid={proc.pid}；日志写入 "
                        f"{stdout_path.name}/{stderr_path.name}"
                    ),
                    percent=30,
                )
                returncode = self._wait_for_model_factor_subprocess(
                    task=task,
                    proc=proc,
                )
        except Exception as exc:
            _annotate_exception_with_model_lab_subprocess_artifacts(
                exc,
                output_dir=output_dir,
                artifact_paths=artifact_paths,
            )
            raise

        finished_at = _utc_now_iso()
        elapsed_seconds = round(time.monotonic() - start_monotonic, 3)
        peak_rss_kb = _parse_time_peak_rss_kb(stderr_path)
        status_payload: dict[str, object] = {
            "status": "succeeded" if returncode == 0 else "failed",
            "run_id": task.run_id,
            "case_name": task.case_name,
            "returncode": returncode,
            "started_at_utc": started_at,
            "finished_at_utc": finished_at,
            "elapsed_seconds": elapsed_seconds,
            "peak_rss_kb": peak_rss_kb,
            "command": cli_cmd,
            "effective_command": cmd,
            "cwd": str(Path.cwd().resolve()),
            "stdout_log": str(stdout_path),
            "stderr_log": str(stderr_path),
        }
        _write_json_file(status_path, status_payload)

        if returncode != 0:
            stderr_tail = _read_text_tail(stderr_path)
            stdout_tail = _read_text_tail(stdout_path)
            hint = _model_lab_subprocess_failure_hint(
                returncode=returncode,
                stderr_tail=stderr_tail,
                stdout_tail=stdout_tail,
            )
            message = _format_model_lab_subprocess_failure(
                command=cmd,
                returncode=returncode,
                stdout_tail=stdout_tail,
                stderr_tail=stderr_tail,
                elapsed_seconds=elapsed_seconds,
                peak_rss_kb=peak_rss_kb,
            )
            subprocess_error = _ModelLabSubprocessError(
                message,
                returncode=returncode,
                hint=hint,
            )
            _annotate_exception_with_model_lab_subprocess_artifacts(
                subprocess_error,
                output_dir=output_dir,
                artifact_paths=artifact_paths,
            )
            raise subprocess_error

        manifest_paths = _load_model_factor_artifact_paths_from_manifest(output_dir)
        return _SubprocessCaseRunResult(
            output_dir=output_dir,
            artifact_paths={**manifest_paths, **artifact_paths},
        )

    def _execute_single_factor_subprocess_task(
        self,
        task: _RunTask,
        *,
        progress_callback: Any,
    ) -> _SubprocessCaseRunResult:
        from alpha_lab.web_unified import _utc_now_iso, _write_json_file

        spec_path = Path(task.spec_path).expanduser().resolve()
        spec = load_single_factor_case_spec(spec_path)
        output_root_dir = _resolve_single_factor_web_output_root_dir(
            task,
            spec=spec,
        )
        output_dir = (output_root_dir / spec.name).expanduser().resolve()
        log_dir = output_dir / "_web_run_logs" / task.run_id
        log_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = log_dir / "stdout.log"
        stderr_path = log_dir / "stderr.log"
        status_path = log_dir / "status.json"
        artifact_paths = {
            "subprocess_stdout": stdout_path,
            "subprocess_stderr": stderr_path,
            "subprocess_status": status_path,
        }
        cli_cmd = _build_single_factor_subprocess_command(
            task=task,
            spec_path=spec_path,
        )
        cmd = _wrap_command_with_time(cli_cmd)
        started_at = _utc_now_iso()
        _write_json_file(
            status_path,
            {
                "status": "starting",
                "run_id": task.run_id,
                "case_name": task.case_name,
                "workflow": task.workflow,
                "started_at_utc": started_at,
                "command": cli_cmd,
                "effective_command": cmd,
                "cwd": str(Path.cwd().resolve()),
                "stdout_log": str(stdout_path),
                "stderr_log": str(stderr_path),
            },
        )
        with self._lock:
            record = self._records.get(task.run_id)
            if record is not None:
                record.output_dir = str(output_dir)
                record.artifact_paths.update(
                    {key: str(path) for key, path in artifact_paths.items()}
                )
                self._push_progress_locked(
                    record,
                    message="已启动隔离子进程执行单因子实验",
                    percent=8,
                )

        env = _build_single_factor_subprocess_env()
        start_monotonic = time.monotonic()
        try:
            with stdout_path.open("ab") as stdout, stderr_path.open("ab") as stderr:
                proc = subprocess.Popen(  # noqa: S603 - argv is built without shell=True.
                    cmd,
                    cwd=str(Path.cwd().resolve()),
                    env=env,
                    stdout=stdout,
                    stderr=stderr,
                )
                _write_json_file(
                    status_path,
                    {
                        "status": "running",
                        "run_id": task.run_id,
                        "case_name": task.case_name,
                        "pid": proc.pid,
                        "started_at_utc": started_at,
                        "command": cli_cmd,
                        "effective_command": cmd,
                        "cwd": str(Path.cwd().resolve()),
                        "stdout_log": str(stdout_path),
                        "stderr_log": str(stderr_path),
                    },
                )
                progress_callback(
                    message=(
                        f"单因子子进程运行中 pid={proc.pid}；日志写入 "
                        f"{stdout_path.name}/{stderr_path.name}"
                    ),
                    percent=30,
                )
                returncode = self._wait_for_model_factor_subprocess(
                    task=task,
                    proc=proc,
                )
        except Exception as exc:
            _annotate_exception_with_model_lab_subprocess_artifacts(
                exc,
                output_dir=output_dir,
                artifact_paths=artifact_paths,
            )
            raise

        finished_at = _utc_now_iso()
        elapsed_seconds = round(time.monotonic() - start_monotonic, 3)
        peak_rss_kb = _parse_time_peak_rss_kb(stderr_path)
        status_payload: dict[str, object] = {
            "status": "succeeded" if returncode == 0 else "failed",
            "run_id": task.run_id,
            "case_name": task.case_name,
            "returncode": returncode,
            "started_at_utc": started_at,
            "finished_at_utc": finished_at,
            "elapsed_seconds": elapsed_seconds,
            "peak_rss_kb": peak_rss_kb,
            "command": cli_cmd,
            "effective_command": cmd,
            "cwd": str(Path.cwd().resolve()),
            "stdout_log": str(stdout_path),
            "stderr_log": str(stderr_path),
        }
        _write_json_file(status_path, status_payload)

        if returncode != 0:
            stderr_tail = _read_text_tail(stderr_path)
            stdout_tail = _read_text_tail(stdout_path)
            hint = _model_lab_subprocess_failure_hint(
                returncode=returncode,
                stderr_tail=stderr_tail,
                stdout_tail=stdout_tail,
            )
            message = _format_model_lab_subprocess_failure(
                command=cmd,
                returncode=returncode,
                stdout_tail=stdout_tail,
                stderr_tail=stderr_tail,
                elapsed_seconds=elapsed_seconds,
                peak_rss_kb=peak_rss_kb,
            )
            subprocess_error = _ModelLabSubprocessError(
                message,
                returncode=returncode,
                hint=hint,
            )
            _annotate_exception_with_model_lab_subprocess_artifacts(
                subprocess_error,
                output_dir=output_dir,
                artifact_paths=artifact_paths,
            )
            raise subprocess_error

        manifest_paths = _load_single_factor_artifact_paths_from_manifest(output_dir)
        return _SubprocessCaseRunResult(
            output_dir=output_dir,
            artifact_paths={**manifest_paths, **artifact_paths},
        )

    def _wait_for_model_factor_subprocess(
        self,
        *,
        task: _RunTask,
        proc: subprocess.Popen[bytes],
    ) -> int:
        last_heartbeat = time.monotonic()
        while True:
            returncode = proc.poll()
            if returncode is not None:
                return int(returncode)
            with self._lock:
                cancelled = task.run_id in self._cancel_requests or task.run_id not in self._tasks
            if cancelled:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)
                raise RuntimeError("模型因子子进程已按用户请求取消")
            now = time.monotonic()
            if now - last_heartbeat >= 30:
                self._push_progress(
                    task.run_id,
                    message=f"实验子进程仍在运行 pid={proc.pid}",
                    percent=30,
                )
                last_heartbeat = now
            time.sleep(_MODEL_LAB_SUBPROCESS_POLL_SECONDS)

    def _execute_single_task(
        self,
        task: _RunTask,
        *,
        allow_fallback: bool,
    ) -> None:
        from alpha_lab.web_unified import (
            _build_run_error_payload,
            _safe_rmtree,
            _utc_now_iso,
        )

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
                    message=(
                        "任务已启动，准备执行模型因子训练与评估"
                        if stored_task.workflow == "model_factor"
                        else "任务已启动，准备执行 single-factor pipeline"
                    ),
                    percent=2,
                )
        try:

            def progress_callback(message: str, percent: int) -> None:
                self._push_progress(
                    run_id,
                    message=message,
                    percent=percent,
                )

            run_result: RunSuccessResult
            if task.workflow == "model_factor":
                run_result = self._execute_model_factor_subprocess_task(
                    task,
                    progress_callback=progress_callback,
                )
            else:
                run_result = self._execute_single_factor_subprocess_task(
                    task,
                    progress_callback=progress_callback,
                )
            self._finalize_success(task=task, result=run_result)
        except Exception as exc:
            error_payload = _build_run_error_payload(exc)
            output_dir_from_exc = _coerce_finite_or_text(getattr(exc, "model_lab_output_dir", None))
            artifact_paths_from_exc: dict[str, str] = {}
            raw_artifacts_from_exc = getattr(exc, "model_lab_artifact_paths", None)
            if isinstance(raw_artifacts_from_exc, dict):
                for key, value in raw_artifacts_from_exc.items():
                    key_text = str(key or "").strip()
                    value_text = _coerce_finite_or_text(value)
                    if key_text and value_text:
                        artifact_paths_from_exc[key_text] = value_text
            with self._lock:
                stored = self._records.get(run_id)
                if stored is None or run_id in self._cancel_requests:
                    self._records.pop(run_id, None)
                    self._cancel_requests.discard(run_id)
                    self._tasks.pop(run_id, None)
                    cancelled_output = (
                        stored.output_dir if stored is not None else output_dir_from_exc
                    )
                else:
                    cancelled_output = None
                    if output_dir_from_exc and not stored.output_dir:
                        stored.output_dir = output_dir_from_exc
                    if artifact_paths_from_exc:
                        stored.artifact_paths = {
                            **stored.artifact_paths,
                            **artifact_paths_from_exc,
                        }
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
            if cancelled_output:
                _safe_rmtree(cancelled_output)
        finally:
            with self._lock:
                self._tasks.pop(run_id, None)
            gc.collect()

    def _finalize_success(self, *, task: _RunTask, result: RunSuccessResult) -> None:
        from alpha_lab.web_unified import (
            _extract_metrics_summary,
            _safe_rmtree,
            _utc_now_iso,
        )

        run_id = task.run_id
        artifact_paths_map = cast(Mapping[str, Path], result.artifact_paths)
        with self._lock:
            if run_id in self._cancel_requests or run_id not in self._records:
                self._records.pop(run_id, None)
                self._cancel_requests.discard(run_id)
                self._tasks.pop(run_id, None)
                cancelled_output = str(result.output_dir) if result.output_dir else None
            else:
                cancelled_output = None
        if cancelled_output:
            _safe_rmtree(cancelled_output)
            return
        self._push_progress(run_id, message="整理产物清单", percent=93)
        artifact_paths = {key: str(path) for key, path in artifact_paths_map.items()}
        if task.render_report:
            self._push_progress(run_id, message="生成 case report", percent=96)
            report_path = write_case_report(result.output_dir, overwrite=True)
            artifact_paths["case_report"] = str(report_path)
        self._push_progress(run_id, message="提取关键指标摘要", percent=98)
        summary = _extract_metrics_summary(
            artifact_paths_map.get("metrics"),
            run_status="succeeded",
        )
        with self._lock:
            stored = self._records.get(run_id)
            if stored is None or run_id in self._cancel_requests:
                self._records.pop(run_id, None)
                self._cancel_requests.discard(run_id)
                self._tasks.pop(run_id, None)
                cancelled_output = str(result.output_dir) if result.output_dir else None
            else:
                cancelled_output = None
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
        if cancelled_output:
            _safe_rmtree(cancelled_output)


__all__ = [
    "RunSuccessResult",
    "_RUN_SUMMARY_COMPACT_KEYS",
    "_RunRecord",
    "_RunStore",
    "_compact_metrics_summary",
]
