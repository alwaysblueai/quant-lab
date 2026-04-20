# ruff: noqa: E501
from __future__ import annotations

import datetime as dt
import json
import threading
import traceback
import uuid
import webbrowser
from csv import DictReader
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Literal
from urllib.parse import parse_qs, unquote, urlparse

from alpha_lab.exceptions import AlphaLabConfigError, AlphaLabDataError, AlphaLabExperimentError
from alpha_lab.real_cases.single_factor.pipeline import (
    SingleFactorBatchParallelConfig,
    run_single_factor_case,
    run_single_factor_cases,
)
from alpha_lab.reporting.renderers import write_case_report
from alpha_lab.research_bridge.models import (
    load_project_config,
    load_yaml_document,
    save_project_config,
)
from alpha_lab.research_bridge.service import (
    PROJECTS_DIRNAME,
    apply_writeback,
    init_project,
    refresh_project_pack,
    scaffold_case,
    start_round,
    summarize_run,
)
from alpha_lab.vault_export import resolve_vault_root

RunStatus = Literal["queued", "running", "succeeded", "failed"]
_FRONTEND_BATCH_WINDOW_SECONDS: float = 0.20
_FRONTEND_BATCH_MAX_WORKERS: int = 4
_FRONTEND_BATCH_FACTORS_PER_WORKER: int = 2


def start_web_cockpit_server(
    *,
    host: str = "127.0.0.1",
    port: int = 8766,
    workspace_root: str | Path | None = None,
    vault_root: str | Path | None = None,
    open_browser: bool = True,
) -> None:
    print(
        "[DEPRECATED] `alpha_lab.web_cockpit` is deprecated. "
        "Prefer `alpha-lab web unified`.",
    )
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

    cockpit = _CockpitService(vault_root=resolved_vault, workspace_root=resolved_workspace)

    class _Handler(_CockpitRequestHandler):
        service = cockpit

    server = ThreadingHTTPServer((host, port), _Handler)
    url = f"http://{host}:{port}/"
    print("")
    print("  Workflow : web-cockpit")
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
        print("  Workflow : web-cockpit")
        print("  Status   : stopped")
    finally:
        server.server_close()


@dataclass
class _CockpitRunRecord:
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
    output_dir: str | None = None
    artifact_paths: dict[str, str] = field(default_factory=dict)
    summary: dict[str, object] = field(default_factory=dict)
    summarize_feedback_path: str | None = None
    summarize_draft_path: str | None = None
    summarize_state_patch_path: str | None = None
    error: str | None = None

    def clone(self) -> _CockpitRunRecord:
        return _CockpitRunRecord(
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
            output_dir=self.output_dir,
            artifact_paths=dict(self.artifact_paths),
            summary=dict(self.summary),
            summarize_feedback_path=self.summarize_feedback_path,
            summarize_draft_path=self.summarize_draft_path,
            summarize_state_patch_path=self.summarize_state_patch_path,
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
            "output_dir": self.output_dir,
            "artifact_paths": dict(self.artifact_paths),
            "summary": dict(self.summary),
            "summarize_feedback_path": self.summarize_feedback_path,
            "summarize_draft_path": self.summarize_draft_path,
            "summarize_state_patch_path": self.summarize_state_patch_path,
            "error": self.error,
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


class _CockpitRunStore:
    def __init__(self) -> None:
        self._records: dict[str, _CockpitRunRecord] = {}
        self._tasks: dict[str, _RunTask] = {}
        self._lock = threading.Lock()
        self._dispatch_event = threading.Event()
        self._dispatcher = threading.Thread(target=self._dispatch_loop, daemon=True)
        self._dispatcher.start()

    def submit(self, task: _RunTask) -> _CockpitRunRecord:
        record = _CockpitRunRecord(
            run_id=task.run_id,
            project_slug=task.project_slug,
            case_name=task.case_name,
            round_id=task.round_id,
            spec_path=task.spec_path,
            submitted_at_utc=_utc_now_iso(),
            evaluation_profile=task.evaluation_profile,
            output_root_dir=task.output_root_dir,
            render_report=task.render_report,
        )
        with self._lock:
            self._records[record.run_id] = record
            self._tasks[record.run_id] = task
        self._dispatch_event.set()
        return record.clone()

    def get(self, run_id: str) -> _CockpitRunRecord | None:
        with self._lock:
            record = self._records.get(run_id)
            return None if record is None else record.clone()

    def list_records(self, *, project_slug: str | None = None) -> list[_CockpitRunRecord]:
        with self._lock:
            records = [rec.clone() for rec in self._records.values()]
        if project_slug is None:
            return sorted(records, key=lambda item: item.submitted_at_utc, reverse=True)
        filtered = [item for item in records if item.project_slug == project_slug]
        return sorted(filtered, key=lambda item: item.submitted_at_utc, reverse=True)

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
            queued_with_records: list[tuple[_RunTask, _CockpitRunRecord]] = []
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
                key = (task.evaluation_profile, task.output_root_dir or "")
                grouped.setdefault(key, []).append(task)
            ordered_groups = list(grouped.values())
            for tasks in ordered_groups:
                for task in tasks:
                    record = self._records.get(task.run_id)
                    if record is None:
                        continue
                    record.status = "running"
                    record.started_at_utc = started_at
            return ordered_groups

    def _execute_task_group(self, tasks: list[_RunTask]) -> None:
        if len(tasks) <= 1:
            for task in tasks:
                self._execute_single_task(task)
            return

        try:
            results = run_single_factor_cases(
                [task.spec_path for task in tasks],
                output_root_dir=tasks[0].output_root_dir,
                evaluation_profile=tasks[0].evaluation_profile,
                vault_export_mode="skip",
                batch_parallel_config=_build_frontend_batch_parallel_config(len(tasks)),
                reuse_input_bundle=True,
            )
        except Exception:
            for task in tasks:
                self._execute_single_task(task)
            return

        if len(results) != len(tasks):
            for task in tasks:
                self._execute_single_task(task)
            return

        for task, result in zip(tasks, results, strict=True):
            self._finalize_success(task.run_id, task.render_report, result)

    def _execute_single_task(self, task: _RunTask) -> None:
        try:
            result = run_single_factor_case(
                task.spec_path,
                output_root_dir=task.output_root_dir,
                evaluation_profile=task.evaluation_profile,
                vault_export_mode="skip",
            )
            self._finalize_success(task.run_id, task.render_report, result)
        except Exception as exc:
            self._mark_failed(task.run_id, exc)

    def _finalize_success(self, run_id: str, render_report: bool, result: Any) -> None:
        artifact_paths = {key: str(path) for key, path in result.artifact_paths.items()}
        if render_report:
            report_path = write_case_report(result.output_dir, overwrite=True)
            artifact_paths["case_report"] = str(report_path)
        summary = _extract_metrics_summary(result.artifact_paths.get("metrics"))
        with self._lock:
            stored = self._records[run_id]
            stored.status = "succeeded"
            stored.finished_at_utc = _utc_now_iso()
            stored.output_dir = str(result.output_dir)
            stored.artifact_paths = artifact_paths
            stored.summary = summary
            stored.error = None
            self._tasks.pop(run_id, None)

    def _mark_failed(self, run_id: str, exc: Exception) -> None:
        with self._lock:
            stored = self._records[run_id]
            stored.status = "failed"
            stored.finished_at_utc = _utc_now_iso()
            stored.error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=12)}"
            self._tasks.pop(run_id, None)


def _build_frontend_batch_parallel_config(
    n_tasks: int,
) -> SingleFactorBatchParallelConfig:
    worker_slots = max(1, min(_FRONTEND_BATCH_MAX_WORKERS, n_tasks))
    return SingleFactorBatchParallelConfig(
        mode="process",
        max_workers=worker_slots,
        factors_per_worker=_FRONTEND_BATCH_FACTORS_PER_WORKER,
    )


class _CockpitService:
    def __init__(self, *, vault_root: Path, workspace_root: Path) -> None:
        self.vault_root = vault_root.resolve()
        self.workspace_root = workspace_root.resolve()
        self.run_store = _CockpitRunStore()

    @property
    def projects_root(self) -> Path:
        return (self.vault_root / PROJECTS_DIRNAME).resolve()

    def dashboard(self) -> dict[str, object]:
        projects = self.list_projects()
        runs = [item.to_payload() for item in self.run_store.list_records()]
        status_counts: dict[str, int] = {"queued": 0, "running": 0, "succeeded": 0, "failed": 0}
        for record in runs:
            status = str(record["status"])
            status_counts[status] = status_counts.get(status, 0) + 1
        pending_writebacks = sum(
            _as_int(project.get("pending_writeback_count"), default=0) for project in projects
        )
        return {
            "vault_root": str(self.vault_root),
            "workspace_root": str(self.workspace_root),
            "project_count": len(projects),
            "pending_writebacks": pending_writebacks,
            "run_status_counts": status_counts,
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
            draft_rows = _list_draft_summaries(paths["drafts_dir"])
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
                    "round_count": _count_dirs(paths["rounds_dir"]),
                    "case_count": len(_list_cases(paths)),
                    "draft_count": len(draft_rows),
                    "pending_writeback_count": sum(
                        1 for draft in draft_rows if draft.get("review_status") == "pending"
                    ),
                    "path": str(paths["project_dir"]),
                }
            )
        return sorted(rows, key=lambda row: str(row["slug"]))

    def get_project(self, slug: str) -> dict[str, object]:
        paths = _project_paths(self.vault_root, slug)
        if not paths["project_yaml"].exists():
            raise FileNotFoundError(f"project not found: {slug}")
        project = load_project_config(paths["project_yaml"])
        rounds = _list_rounds(paths["rounds_dir"])
        cases = _list_cases(paths)
        drafts = _list_draft_summaries(paths["drafts_dir"])
        docs = {
            "project_brief": _read_text(paths["project_brief"]),
            "project_rules": _read_text(paths["project_rules"]),
            "card_map": _read_text(paths["card_map"]),
            "active_state": _read_text(paths["active_state"]) or _render_project_status(project),
            "recent_history": _read_text(paths["recent_history"]),
            "decision_log": _read_text(paths["decision_log"])
            or _read_text(paths["legacy_decision_log"]),
            "current_case": _read_text(paths["current_case"]),
            "latest_run": _read_text(paths["latest_run"]),
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
            "rounds": rounds,
            "cases": cases,
            "drafts": drafts,
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
        refresh_project_pack(vault_root=self.vault_root, project_slug=slug, mode="fast")
        return self.get_project(slug)

    def refresh_project(self, slug: str) -> dict[str, object]:
        result = refresh_project_pack(vault_root=self.vault_root, project_slug=slug, mode="fast")
        return {"slug": result.project.slug, "project_dir": str(result.paths.project_dir)}

    def create_round(self, slug: str, payload: dict[str, object]) -> dict[str, object]:
        topic = str(payload.get("topic") or "").strip()
        if not topic:
            raise ValueError("topic is required")
        round_result = start_round(
            vault_root=self.vault_root,
            project_slug=slug,
            topic=topic,
            round_id=str(payload.get("round_id") or "").strip() or None,
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

    def get_discussion_capture(self, slug: str, round_id: str) -> dict[str, object]:
        round_dir = _project_paths(self.vault_root, slug)["rounds_dir"] / round_id
        file_path = round_dir / "discussion_capture.md"
        if not file_path.exists():
            raise FileNotFoundError(f"discussion_capture does not exist: {file_path}")
        return {
            "project_slug": slug,
            "round_id": round_id,
            "path": str(file_path),
            "content": file_path.read_text(encoding="utf-8"),
        }

    def update_discussion_capture(
        self, slug: str, round_id: str, content: str
    ) -> dict[str, object]:
        round_dir = _project_paths(self.vault_root, slug)["rounds_dir"] / round_id
        round_dir.mkdir(parents=True, exist_ok=True)
        file_path = round_dir / "discussion_capture.md"
        file_path.write_text(content, encoding="utf-8")
        return {
            "project_slug": slug,
            "round_id": round_id,
            "path": str(file_path),
            "content": content,
        }

    def create_case(self, slug: str, payload: dict[str, object]) -> dict[str, object]:
        round_id = str(payload.get("round_id") or "").strip()
        case_name = str(payload.get("case_name") or "").strip()
        if not case_name:
            raise ValueError("case_name is required")
        result = scaffold_case(
            vault_root=self.vault_root,
            project_slug=slug,
            round_id=round_id or None,
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
            "round_id": result.round_id,
            "case_name": result.case_name,
            "current_case_path": str(result.current_case_path),
            "handoff_path": str(result.handoff_path),
            "spec_path": str(result.spec_path),
        }

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
        round_id = (
            str(payload.get("round_id") or "").strip() or str(run_record.round_id or "").strip()
        )
        result = summarize_run(
            vault_root=self.vault_root,
            project_slug=slug,
            round_id=round_id or None,
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
            "round_id": result.round_id,
            "summary_path": str(result.summary_path),
            "latest_path": str(result.latest_path),
            "decision_log_path": str(result.decision_log_path),
            "latest_experiment_feedback": str(result.latest_experiment_feedback),
            "writeback_draft": str(result.writeback_draft),
            "state_update_patch": str(result.state_update_patch),
        }

    def list_drafts(self, slug: str) -> list[dict[str, object]]:
        drafts_dir = _project_paths(self.vault_root, slug)["drafts_dir"]
        return _list_draft_summaries(drafts_dir)

    def patch_draft(
        self, slug: str, draft_name: str, payload: dict[str, object]
    ) -> dict[str, object]:
        draft_path = _resolve_draft_path(
            _project_paths(self.vault_root, slug)["drafts_dir"], draft_name
        )
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
                continue
            frontmatter[key] = str(value)
        draft_path.write_text(
            _compose_markdown_with_frontmatter(frontmatter, body), encoding="utf-8"
        )
        return _draft_summary(draft_path)

    def apply_draft(
        self, slug: str, draft_name: str, payload: dict[str, object] | None = None
    ) -> dict[str, object]:
        draft_path = _resolve_draft_path(
            _project_paths(self.vault_root, slug)["drafts_dir"], draft_name
        )
        mode = None
        if payload is not None:
            mode = _optional_text(payload.get("mode"))
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


class _CockpitRequestHandler(BaseHTTPRequestHandler):
    service: _CockpitService

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_html(_index_html())
            return
        if parsed.path == "/api/dashboard":
            self._send_json(self.service.dashboard())
            return
        if parsed.path == "/api/projects":
            self._send_json({"projects": self.service.list_projects()})
            return
        if parsed.path == "/api/cards/search":
            params = parse_qs(parsed.query)
            query = str((params.get("q") or [""])[0])
            limit = _safe_limit((params.get("limit") or ["50"])[0], default=50)
            self._send_json(self.service.search_cards(query, limit=limit))
            return
        parts = _path_parts(parsed.path)
        if len(parts) >= 3 and parts[0] == "api" and parts[1] == "projects":
            slug = parts[2]
            try:
                if len(parts) == 3:
                    self._send_json(self.service.get_project(slug))
                    return
                if len(parts) == 6 and parts[3] == "rounds" and parts[5] == "discussion":
                    self._send_json(self.service.get_discussion_capture(slug, parts[4]))
                    return
                if len(parts) == 4 and parts[3] == "runs":
                    runs = [
                        item.to_payload()
                        for item in self.service.run_store.list_records(project_slug=slug)
                    ]
                    self._send_json({"project_slug": slug, "runs": runs})
                    return
                if len(parts) == 5 and parts[3] == "runs":
                    run = self.service.run_store.get(parts[4])
                    if run is None or run.project_slug != slug:
                        self._send_json(
                            {"error": f"run not found: {parts[4]}"}, status=HTTPStatus.NOT_FOUND
                        )
                        return
                    self._send_json(run.to_payload())
                    return
                if len(parts) == 7 and parts[3] == "runs" and parts[5] == "artifact":
                    self._handle_get_run_artifact(slug=slug, run_id=parts[4], artifact_key=parts[6])
                    return
                if len(parts) == 4 and parts[3] == "drafts":
                    self._send_json(
                        {"project_slug": slug, "drafts": self.service.list_drafts(slug)}
                    )
                    return
            except Exception as exc:
                self._send_error_payload(exc)
                return
        self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        payload = self._read_json_body_or_empty()
        parts = _path_parts(parsed.path)
        try:
            if parsed.path == "/api/projects":
                created = self.service.create_project(payload)
                self._send_json(created, status=HTTPStatus.CREATED)
                return
            if len(parts) >= 3 and parts[0] == "api" and parts[1] == "projects":
                slug = parts[2]
                if len(parts) == 4 and parts[3] == "refresh":
                    self._send_json(self.service.refresh_project(slug))
                    return
                if len(parts) == 4 and parts[3] == "rounds":
                    self._send_json(
                        self.service.create_round(slug, payload), status=HTTPStatus.CREATED
                    )
                    return
                if len(parts) == 4 and parts[3] == "cases":
                    self._send_json(
                        self.service.create_case(slug, payload), status=HTTPStatus.CREATED
                    )
                    return
                if len(parts) == 4 and parts[3] == "runs":
                    self._send_json(
                        self.service.submit_run(slug, payload), status=HTTPStatus.CREATED
                    )
                    return
                if len(parts) == 6 and parts[3] == "runs" and parts[5] == "summarize":
                    self._send_json(self.service.summarize_run(slug, parts[4], payload))
                    return
                if len(parts) == 6 and parts[3] == "drafts" and parts[5] == "apply":
                    self._send_json(self.service.apply_draft(slug, parts[4], payload))
                    return
        except Exception as exc:
            self._send_error_payload(exc)
            return
        self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)

    def do_PATCH(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        parts = _path_parts(parsed.path)
        payload = self._read_json_body_or_empty()
        try:
            if len(parts) >= 3 and parts[0] == "api" and parts[1] == "projects":
                slug = parts[2]
                if len(parts) == 3:
                    self._send_json(self.service.update_project_status(slug, payload))
                    return
                if len(parts) == 5 and parts[3] == "drafts":
                    self._send_json(self.service.patch_draft(slug, parts[4], payload))
                    return
        except Exception as exc:
            self._send_error_payload(exc)
            return
        self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)

    def do_PUT(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        parts = _path_parts(parsed.path)
        payload = self._read_json_body_or_empty()
        try:
            if (
                len(parts) == 6
                and parts[0] == "api"
                and parts[1] == "projects"
                and parts[3] == "rounds"
                and parts[5] == "discussion"
            ):
                content = str(payload.get("content") or "")
                self._send_json(self.service.update_discussion_capture(parts[2], parts[4], content))
                return
        except Exception as exc:
            self._send_error_payload(exc)
            return
        self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003
        return

    def _handle_get_run_artifact(self, *, slug: str, run_id: str, artifact_key: str) -> None:
        run = self.service.run_store.get(run_id)
        if run is None or run.project_slug != slug:
            self._send_json({"error": f"run not found: {run_id}"}, status=HTTPStatus.NOT_FOUND)
            return
        path_text = run.artifact_paths.get(artifact_key)
        if not path_text:
            self._send_json(
                {"error": f"artifact key not found: {artifact_key}"},
                status=HTTPStatus.NOT_FOUND,
            )
            return
        artifact_path = Path(path_text).resolve()
        if not artifact_path.exists() or not artifact_path.is_file():
            self._send_json(
                {"error": f"artifact file not found: {artifact_path}"},
                status=HTTPStatus.NOT_FOUND,
            )
            return
        content = artifact_path.read_bytes()
        ctype = _guess_content_type(artifact_path)
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(content)))
        self.send_header(
            "Content-Disposition",
            f'inline; filename="{artifact_path.name}"',
        )
        self.end_headers()
        self.wfile.write(content)

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
        self.end_headers()
        self.wfile.write(encoded)

    def _send_error_payload(self, exc: Exception) -> None:
        status = HTTPStatus.BAD_REQUEST
        if isinstance(exc, FileNotFoundError):
            status = HTTPStatus.NOT_FOUND
        elif isinstance(exc, PermissionError):
            status = HTTPStatus.FORBIDDEN
        payload: dict[str, object] = {
            "error": f"{type(exc).__name__}: {exc}",
            "trace": traceback.format_exc(limit=6),
        }
        self._send_json(payload, status=status)


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
        "project_brief": project_dir / "01_project_brief.md",
        "project_rules": project_dir / "02_project_rules.md",
        "card_map": project_dir / "03_card_map.md",
        "recent_history": project_dir / "04_recent_history.md",
        "active_state": project_dir / "10_active_state.md",
        "decision_log": project_dir / "decision_log.md",
        "legacy_decision_log": project_dir / "20_decision_log.md",
        "rounds_dir": project_dir / "30_rounds",
        "specs_dir": project_dir / "40_specs",
        "drafts_dir": project_dir / "50_writeback_drafts",
    }


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
    specs_dir = paths["specs_dir"]
    if not specs_dir.exists():
        return rows
    for spec_path in sorted(specs_dir.glob("*.yaml")):
        case_name = spec_path.stem
        handoff_path = specs_dir / f"{case_name}__knowledge_handoff.md"
        if any(str(existing["case_name"]) == case_name for existing in rows):
            continue
        rows.append(
            {
                "case_name": case_name,
                "spec_path": str(spec_path),
                "handoff_path": str(handoff_path),
                "spec_exists": spec_path.exists(),
                "handoff_exists": handoff_path.exists(),
                "is_current": spec_path == current_case_path,
            }
        )
    return rows


def _resolve_case_spec_path(paths: dict[str, Path], case_name: str) -> Path:
    legacy_spec = paths["specs_dir"] / f"{case_name}.yaml"
    if legacy_spec.exists():
        return legacy_spec
    current_case = paths["current_case"]
    if current_case.exists():
        current_name = _yaml_case_name(current_case)
        if current_name == case_name or current_case.stem == case_name:
            return current_case
    return legacy_spec


def _yaml_case_name(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        payload = load_yaml_document(path)
    except Exception:
        return ""
    return str(payload.get("name") or "").strip()


def _render_project_status(project: Any) -> str:
    return "\n".join(
        [
            f"- lifecycle: {project.status.lifecycle}",
            f"- current_hypothesis: {project.status.current_hypothesis}",
            f"- current_focus: {project.status.current_focus}",
            f"- next_action: {project.status.next_action}",
            f"- current_case: {project.status.current_case or 'pending'}",
            f"- latest_run: {project.status.latest_run or 'pending'}",
            f"- last_verdict: {project.status.last_verdict or 'pending'}",
        ]
    )


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


def _compose_markdown_with_frontmatter(frontmatter: dict[str, object], body: str) -> str:
    yaml = _require_yaml()
    rendered_frontmatter = yaml.safe_dump(frontmatter, sort_keys=False, allow_unicode=True).strip()
    return f"---\n{rendered_frontmatter}\n---\n\n{body.rstrip()}\n"


def _resolve_draft_path(drafts_dir: Path, draft_name: str) -> Path:
    candidate = (drafts_dir / draft_name).resolve()
    if not str(candidate).startswith(str(drafts_dir.resolve())):
        raise PermissionError("invalid draft path")
    if not candidate.exists():
        raise FileNotFoundError(f"draft not found: {candidate}")
    return candidate


def _count_dirs(path: Path) -> int:
    if not path.exists():
        return 0
    return len([item for item in path.iterdir() if item.is_dir()])


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


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
    keys = (
        "factor_verdict",
        "campaign_triage",
        "promotion_decision",
        "portfolio_validation_status",
        "mean_ic",
        "mean_rank_ic",
        "ic_ir",
        "mean_long_short_return",
        "mean_long_short_turnover",
        "annualized_return",
        "sharpe",
        "max_drawdown",
    )
    summary: dict[str, object] = {}
    for key in keys:
        if key in metrics:
            summary[key] = metrics[key]
    return summary


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
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return default
        try:
            return int(text)
        except ValueError:
            return default
    return default


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
    return "application/octet-stream"


def _safe_limit(value: str, *, default: int) -> int:
    try:
        parsed = int(value)
    except ValueError:
        return default
    return max(1, min(parsed, 200))


def _utc_now_iso() -> str:
    return dt.datetime.now(dt.UTC).isoformat().replace("+00:00", "Z")


def _require_yaml() -> Any:
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover
        raise AlphaLabExperimentError("PyYAML is required for cockpit draft editing") from exc
    return yaml


def _index_html() -> str:
    return """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Alpha Lab Research Cockpit</title>
  <style>
    :root {
      --bg: #f4f1ea;
      --panel: #fffdf8;
      --ink: #1f2a30;
      --muted: #667882;
      --brand: #005f73;
      --brand-soft: #cfe7e5;
      --ok: #2a9d8f;
      --warn: #d08c00;
      --bad: #bc4749;
      --line: #d6d8db;
      --mono: "JetBrains Mono", "Consolas", "SFMono-Regular", monospace;
      --sans: "MiSans", "PingFang SC", "Microsoft YaHei UI", "Segoe UI", sans-serif;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--ink);
      font-family: var(--sans);
      background:
        radial-gradient(circle at 0% 0%, #dde8e8 0%, transparent 35%),
        radial-gradient(circle at 100% 0%, #efe7d8 0%, transparent 30%),
        var(--bg);
    }
    .layout {
      display: grid;
      grid-template-columns: 280px 1fr;
      min-height: 100vh;
    }
    .sidebar {
      background: #f9f7f0;
      border-right: 1px solid var(--line);
      padding: 16px;
      position: sticky;
      top: 0;
      height: 100vh;
      overflow: auto;
    }
    .brand {
      margin: 0 0 4px 0;
      font-size: 18px;
      letter-spacing: .04em;
      text-transform: uppercase;
      color: var(--brand);
      font-weight: 750;
    }
    .sub {
      margin: 0 0 16px 0;
      color: var(--muted);
      font-size: 13px;
    }
    .nav button {
      width: 100%;
      border: 1px solid var(--line);
      background: var(--panel);
      color: var(--ink);
      padding: 10px 12px;
      margin: 6px 0;
      text-align: left;
      border-radius: 8px;
      font-weight: 600;
      cursor: pointer;
    }
    .nav button.active {
      border-color: var(--brand);
      background: var(--brand-soft);
      color: #023646;
    }
    .main {
      padding: 20px;
      overflow: auto;
    }
    .toolbar {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 16px;
    }
    .toolbar button {
      border: 0;
      background: var(--brand);
      color: #fff;
      padding: 8px 12px;
      border-radius: 8px;
      cursor: pointer;
      font-weight: 600;
    }
    .toolbar button.ghost {
      background: #e8ecef;
      color: #20323a;
    }
    .cards {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
      gap: 12px;
      margin-bottom: 16px;
    }
    .card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 12px;
    }
    .card h3 {
      margin: 0 0 6px 0;
      font-size: 14px;
      text-transform: uppercase;
      letter-spacing: .04em;
      color: var(--muted);
    }
    .card .value {
      font-size: 24px;
      font-weight: 700;
      color: var(--brand);
    }
    .grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 12px;
      margin-bottom: 12px;
    }
    .panel h2 {
      margin: 0 0 8px 0;
      font-size: 16px;
      color: #17343e;
    }
    textarea, input, select {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px;
      font-family: var(--sans);
      margin: 4px 0 10px 0;
      background: #fff;
    }
    textarea {
      min-height: 120px;
      resize: vertical;
    }
    pre {
      white-space: pre-wrap;
      word-break: break-word;
      border-radius: 8px;
      border: 1px solid var(--line);
      background: #fdfcf8;
      padding: 10px;
      max-height: 340px;
      overflow: auto;
      font-family: var(--mono);
      font-size: 12px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }
    th, td {
      border-bottom: 1px solid var(--line);
      padding: 6px;
      text-align: left;
      vertical-align: top;
    }
    .status {
      display: inline-block;
      font-size: 12px;
      font-weight: 700;
      padding: 2px 8px;
      border-radius: 999px;
      background: #e7ebef;
    }
    .status.running { background: #d8ecf2; color: #0b4c63; }
    .status.succeeded { background: #d7efe9; color: #0f5f4b; }
    .status.failed { background: #f4d7d9; color: #7a2228; }
    .status.queued { background: #f3ecd1; color: #6d5200; }
    .muted { color: var(--muted); font-size: 12px; }
    .row {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }
    @media (max-width: 1100px) {
      .layout { grid-template-columns: 1fr; }
      .sidebar { height: auto; position: static; border-right: 0; border-bottom: 1px solid var(--line); }
      .grid, .row { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="layout">
    <aside class="sidebar">
      <h1 class="brand">Research Cockpit</h1>
      <p class="sub">Project → Current Case → Latest Result → Formal Export</p>
      <div class="nav">
        <button data-view="dashboard" class="active">Dashboard</button>
        <button data-view="projects">Projects</button>
        <button data-view="cards">Card Search</button>
      </div>
      <hr />
      <div class="panel">
        <h2>Selected Project</h2>
        <select id="projectSelect"></select>
        <button id="reloadProjectBtn">Reload Project</button>
      </div>
      <div class="panel">
        <h2>Actions</h2>
        <button id="refreshProjectBtn">Refresh Pack</button>
        <button id="reloadDashboardBtn" class="ghost">Reload Dashboard</button>
      </div>
    </aside>
    <main class="main">
      <section id="view-dashboard">
        <div class="toolbar">
          <button id="btnLoadDashboard">Reload Dashboard</button>
        </div>
        <div id="dashboardCards" class="cards"></div>
        <div class="grid">
          <div class="panel">
            <h2>Recent Runs</h2>
            <div id="recentRuns"></div>
          </div>
          <div class="panel">
            <h2>Next Actions</h2>
            <div id="nextActions"></div>
          </div>
        </div>
      </section>

      <section id="view-projects" style="display:none">
        <div class="grid">
          <div>
            <div class="panel">
              <h2>Create Project</h2>
              <div class="row">
                <div><label>slug</label><input id="createSlug" /></div>
                <div><label>title_zh</label><input id="createTitle" /></div>
              </div>
              <div class="row">
                <div><label>category</label><input id="createCategory" value="factor_family" /></div>
                <div><label>owner</label><input id="createOwner" value="yukun" /></div>
              </div>
              <div class="row">
                <div><label>market</label><input id="createMarket" value="ashare" /></div>
                <div><label>frequency</label><input id="createFrequency" value="daily" /></div>
              </div>
              <label>chatgpt_project_name</label><input id="createChatgptName" />
              <label>origin_cards (one per line)</label><textarea id="createOriginCards"></textarea>
              <button id="btnCreateProject">Create Project</button>
            </div>

            <div class="panel">
              <h2>Project Status Patch</h2>
              <label>current_hypothesis</label><textarea id="patchHypothesis"></textarea>
              <label>current_focus</label><textarea id="patchFocus"></textarea>
              <label>next_action</label><textarea id="patchAction"></textarea>
              <button id="btnPatchProject">Update Status</button>
            </div>
          </div>
          <div>
            <div class="panel">
              <h2>Project Detail</h2>
              <pre id="projectDetail"></pre>
            </div>
          </div>
        </div>
        <div class="grid">
          <div class="panel">
            <h2>Legacy Rounds</h2>
            <div class="row">
              <div><label>topic</label><input id="roundTopic" /></div>
              <div><label>round_id(optional)</label><input id="roundId" /></div>
            </div>
            <button id="btnCreateRound">Create Legacy Round</button>
            <label>Round for legacy capture editing</label><input id="captureRoundId" />
            <label>legacy discussion_capture.md</label><textarea id="captureContent"></textarea>
            <div class="toolbar">
              <button id="btnLoadCapture" class="ghost">Load Capture</button>
              <button id="btnSaveCapture">Save Capture</button>
            </div>
            <div id="roundTable"></div>
          </div>

          <div class="panel">
            <h2>Current Case + Runs</h2>
            <div class="row">
              <div><label>round_id(optional legacy)</label><input id="caseRoundId" /></div>
              <div><label>case_name</label><input id="caseName" /></div>
            </div>
            <div class="row">
              <div><label>factor_name(optional)</label><input id="caseFactorName" /></div>
              <div><label>base_method</label><input id="caseBaseMethod" value="momentum" /></div>
            </div>
            <button id="btnCreateCase">Refresh Current Case</button>
            <hr />
            <div class="row">
              <div><label>run case_name</label><input id="runCaseName" /></div>
              <div><label>run round_id(optional legacy)</label><input id="runRoundId" /></div>
            </div>
            <button id="btnStartRun">Start Run</button>
            <div class="toolbar">
              <button id="btnRefreshRuns" class="ghost">Refresh Runs</button>
            </div>
            <div id="runTable"></div>
          </div>
        </div>

        <div class="grid">
          <div class="panel">
            <h2>Legacy Export Drafts</h2>
            <label>export draft file name</label><input id="draftName" />
            <div class="row">
              <div><label>export_status</label><input id="draftStatus" value="approved" /></div>
              <div><label>reviewed_by</label><input id="draftReviewer" value="yukun" /></div>
            </div>
            <label>reviewed_at ("now" allowed)</label><input id="draftReviewedAt" value="now" />
            <label>one_sentence_verdict</label><input id="draftVerdict" />
            <div class="toolbar">
              <button id="btnPatchDraft">Save Export Review</button>
              <button id="btnApplyDraft">Execute Formal Writeback</button>
              <button id="btnRefreshDrafts" class="ghost">Refresh Export Drafts</button>
            </div>
            <div id="draftTable"></div>
          </div>
          <div class="panel">
            <h2>Responses</h2>
            <pre id="responseBox"></pre>
          </div>
        </div>
      </section>

      <section id="view-cards" style="display:none">
        <div class="panel">
          <h2>CARD-INDEX Search</h2>
          <div class="row">
            <div><label>query</label><input id="cardQuery" /></div>
            <div><label>limit</label><input id="cardLimit" value="30" /></div>
          </div>
          <button id="btnSearchCards">Search</button>
          <div id="cardResults"></div>
        </div>
      </section>
    </main>
  </div>

  <script>
    const state = {
      view: "dashboard",
      projects: [],
      selectedProject: "",
      projectDetail: null,
    };

    const $ = (id) => document.getElementById(id);

    async function api(path, method = "GET", body = null) {
      const opts = { method, headers: {} };
      if (body !== null) {
        opts.headers["Content-Type"] = "application/json";
        opts.body = JSON.stringify(body);
      }
      const res = await fetch(path, opts);
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        const message = data.error || `${res.status} ${res.statusText}`;
        throw new Error(message);
      }
      return data;
    }

    function showResponse(payload) {
      $("responseBox").textContent = JSON.stringify(payload, null, 2);
    }

    function switchView(view) {
      state.view = view;
      for (const button of document.querySelectorAll(".nav button")) {
        button.classList.toggle("active", button.dataset.view === view);
      }
      $("view-dashboard").style.display = view === "dashboard" ? "block" : "none";
      $("view-projects").style.display = view === "projects" ? "block" : "none";
      $("view-cards").style.display = view === "cards" ? "block" : "none";
    }

    async function loadDashboard() {
      const data = await api("/api/dashboard");
      const cards = [
        ["Projects", data.project_count],
        ["Pending Formal Exports", data.pending_writebacks],
        ["Running", data.run_status_counts.running || 0],
        ["Failed", data.run_status_counts.failed || 0],
      ];
      $("dashboardCards").innerHTML = cards.map(([title, value]) => `
        <div class="card"><h3>${title}</h3><div class="value">${value}</div></div>
      `).join("");

      $("recentRuns").innerHTML = (data.recent_runs || []).map((run) => `
        <div class="panel" style="margin:8px 0">
          <div><strong>${run.project_slug}/${run.case_name}</strong></div>
          <div><span class="status ${run.status}">${run.status}</span></div>
          <div class="muted">${run.submitted_at_utc}</div>
        </div>
      `).join("") || "<div class='muted'>No runs yet.</div>";

      $("nextActions").innerHTML = (data.next_actions || []).map((item) => `
        <div class="panel" style="margin:8px 0">
          <div><strong>${item.project_slug}</strong></div>
          <div>${item.next_action}</div>
        </div>
      `).join("") || "<div class='muted'>No actions.</div>";
      showResponse(data);
    }

    async function loadProjects() {
      const data = await api("/api/projects");
      state.projects = data.projects || [];
      const select = $("projectSelect");
      select.innerHTML = "<option value=''>-- choose project --</option>" + state.projects.map((p) =>
        `<option value="${p.slug}">${p.slug} | ${p.title_zh}</option>`
      ).join("");
      if (!state.selectedProject && state.projects.length > 0) {
        state.selectedProject = state.projects[0].slug;
      }
      if (state.selectedProject) {
        select.value = state.selectedProject;
        await loadProjectDetail();
      }
    }

    async function loadProjectDetail() {
      if (!state.selectedProject) return;
      const data = await api(`/api/projects/${encodeURIComponent(state.selectedProject)}`);
      state.projectDetail = data;
      $("projectDetail").textContent = JSON.stringify(data, null, 2);
      renderRoundTable(data.rounds || []);
      renderRunTable(data.runs || []);
      renderDraftTable(data.drafts || []);
      showResponse(data);
    }

    function renderRoundTable(rounds) {
      if (!rounds.length) {
        $("roundTable").innerHTML = "<div class='muted'>No rounds yet.</div>";
        return;
      }
      $("roundTable").innerHTML = `
        <table>
          <thead><tr><th>round_id</th><th>feedback</th><th>discussion</th></tr></thead>
          <tbody>${rounds.map((r) => `
            <tr>
              <td>${r.round_id}</td>
              <td>${r.has_feedback ? "yes" : "no"}</td>
              <td>${r.has_discussion_capture ? "yes" : "no"}</td>
            </tr>
          `).join("")}</tbody>
        </table>
      `;
    }

    function renderRunTable(runs) {
      if (!runs.length) {
        $("runTable").innerHTML = "<div class='muted'>No runs yet.</div>";
        return;
      }
      $("runTable").innerHTML = `
        <table>
          <thead><tr><th>run_id</th><th>case</th><th>status</th><th>actions</th></tr></thead>
          <tbody>${runs.map((run) => `
            <tr>
              <td><code>${run.run_id.slice(0, 10)}</code></td>
              <td>${run.case_name}</td>
              <td><span class="status ${run.status}">${run.status}</span></td>
              <td>
                <button data-run="${run.run_id}" class="btnSummarizeRun">refresh result</button>
              </td>
            </tr>
          `).join("")}</tbody>
        </table>
      `;
      document.querySelectorAll(".btnSummarizeRun").forEach((button) => {
        button.addEventListener("click", async () => {
          try {
            const runId = button.dataset.run;
            const roundId = prompt("optional legacy round_id for summarize-run (leave blank for latest result only)");
            const data = await api(
              `/api/projects/${encodeURIComponent(state.selectedProject)}/runs/${encodeURIComponent(runId)}/summarize`,
              "POST",
              roundId ? { round_id: roundId } : {}
            );
            showResponse(data);
            await loadProjectDetail();
          } catch (error) {
            alert(error.message);
          }
        });
      });
    }

    function renderDraftTable(drafts) {
      if (!drafts.length) {
        $("draftTable").innerHTML = "<div class='muted'>No legacy export drafts yet.</div>";
        return;
      }
      $("draftTable").innerHTML = `
        <table>
          <thead><tr><th>name</th><th>export status</th><th>reviewer</th><th>case</th></tr></thead>
          <tbody>${drafts.map((d) => `
            <tr>
              <td>${d.name}</td>
              <td>${d.review_status}</td>
              <td>${d.reviewed_by || "-"}</td>
              <td>${d.case_name || "-"}</td>
            </tr>
          `).join("")}</tbody>
        </table>
      `;
    }

    async function refreshRuns() {
      if (!state.selectedProject) return;
      const data = await api(`/api/projects/${encodeURIComponent(state.selectedProject)}/runs`);
      renderRunTable(data.runs || []);
      showResponse(data);
    }

    async function refreshDrafts() {
      if (!state.selectedProject) return;
      const data = await api(`/api/projects/${encodeURIComponent(state.selectedProject)}/drafts`);
      renderDraftTable(data.drafts || []);
      showResponse(data);
    }

    function linesToList(text) {
      return (text || "").split("\\n").map((x) => x.trim()).filter(Boolean);
    }

    async function init() {
      for (const button of document.querySelectorAll(".nav button")) {
        button.addEventListener("click", () => switchView(button.dataset.view));
      }
      $("projectSelect").addEventListener("change", async (event) => {
        state.selectedProject = event.target.value;
        await loadProjectDetail();
      });
      $("btnLoadDashboard").addEventListener("click", loadDashboard);
      $("reloadDashboardBtn").addEventListener("click", loadDashboard);
      $("reloadProjectBtn").addEventListener("click", loadProjectDetail);
      $("refreshProjectBtn").addEventListener("click", async () => {
        try {
          if (!state.selectedProject) return;
          const data = await api(`/api/projects/${encodeURIComponent(state.selectedProject)}/refresh`, "POST", {});
          showResponse(data);
          await loadProjectDetail();
        } catch (error) {
          alert(error.message);
        }
      });
      $("btnCreateProject").addEventListener("click", async () => {
        try {
          const payload = {
            slug: $("createSlug").value,
            title_zh: $("createTitle").value,
            category: $("createCategory").value,
            owner: $("createOwner").value,
            market: $("createMarket").value,
            frequency: $("createFrequency").value,
            chatgpt_project_name: $("createChatgptName").value,
            origin_cards: linesToList($("createOriginCards").value),
          };
          const data = await api("/api/projects", "POST", payload);
          showResponse(data);
          state.selectedProject = payload.slug;
          await loadProjects();
        } catch (error) {
          alert(error.message);
        }
      });
      $("btnPatchProject").addEventListener("click", async () => {
        try {
          if (!state.selectedProject) return;
          const payload = {
            current_hypothesis: $("patchHypothesis").value,
            current_focus: $("patchFocus").value,
            next_action: $("patchAction").value,
          };
          const data = await api(`/api/projects/${encodeURIComponent(state.selectedProject)}`, "PATCH", payload);
          showResponse(data);
          await loadProjectDetail();
        } catch (error) {
          alert(error.message);
        }
      });
      $("btnCreateRound").addEventListener("click", async () => {
        try {
          if (!state.selectedProject) return;
          const payload = { topic: $("roundTopic").value, round_id: $("roundId").value || null };
          const data = await api(`/api/projects/${encodeURIComponent(state.selectedProject)}/rounds`, "POST", payload);
          showResponse(data);
          $("captureRoundId").value = data.round_id || "";
          await loadProjectDetail();
        } catch (error) {
          alert(error.message);
        }
      });
      $("btnLoadCapture").addEventListener("click", async () => {
        try {
          if (!state.selectedProject) return;
          const roundId = $("captureRoundId").value;
          const data = await api(`/api/projects/${encodeURIComponent(state.selectedProject)}/rounds/${encodeURIComponent(roundId)}/discussion`);
          $("captureContent").value = data.content || "";
          showResponse(data);
        } catch (error) {
          alert(error.message);
        }
      });
      $("btnSaveCapture").addEventListener("click", async () => {
        try {
          if (!state.selectedProject) return;
          const roundId = $("captureRoundId").value;
          const data = await api(
            `/api/projects/${encodeURIComponent(state.selectedProject)}/rounds/${encodeURIComponent(roundId)}/discussion`,
            "PUT",
            { content: $("captureContent").value }
          );
          showResponse(data);
          await loadProjectDetail();
        } catch (error) {
          alert(error.message);
        }
      });
      $("btnCreateCase").addEventListener("click", async () => {
        try {
          if (!state.selectedProject) return;
          const payload = {
            round_id: $("caseRoundId").value,
            case_name: $("caseName").value,
            factor_name: $("caseFactorName").value || null,
            base_method: $("caseBaseMethod").value || "momentum",
          };
          const data = await api(`/api/projects/${encodeURIComponent(state.selectedProject)}/cases`, "POST", payload);
          showResponse(data);
          $("runCaseName").value = payload.case_name;
          $("runRoundId").value = payload.round_id;
          await loadProjectDetail();
        } catch (error) {
          alert(error.message);
        }
      });
      $("btnStartRun").addEventListener("click", async () => {
        try {
          if (!state.selectedProject) return;
          const payload = {
            case_name: $("runCaseName").value,
            round_id: $("runRoundId").value || null,
            render_report: true,
          };
          const data = await api(`/api/projects/${encodeURIComponent(state.selectedProject)}/runs`, "POST", payload);
          showResponse(data);
          await refreshRuns();
        } catch (error) {
          alert(error.message);
        }
      });
      $("btnRefreshRuns").addEventListener("click", refreshRuns);
      $("btnPatchDraft").addEventListener("click", async () => {
        try {
          if (!state.selectedProject) return;
          const draftName = $("draftName").value;
          const payload = {
            review_status: $("draftStatus").value,
            reviewed_by: $("draftReviewer").value,
            reviewed_at: $("draftReviewedAt").value,
            one_sentence_verdict: $("draftVerdict").value,
          };
          const data = await api(`/api/projects/${encodeURIComponent(state.selectedProject)}/drafts/${encodeURIComponent(draftName)}`, "PATCH", payload);
          showResponse(data);
          await refreshDrafts();
        } catch (error) {
          alert(error.message);
        }
      });
      $("btnApplyDraft").addEventListener("click", async () => {
        try {
          if (!state.selectedProject) return;
          const draftName = $("draftName").value;
          const data = await api(`/api/projects/${encodeURIComponent(state.selectedProject)}/drafts/${encodeURIComponent(draftName)}/apply`, "POST", {});
          showResponse(data);
          await loadProjectDetail();
        } catch (error) {
          alert(error.message);
        }
      });
      $("btnRefreshDrafts").addEventListener("click", refreshDrafts);
      $("btnSearchCards").addEventListener("click", async () => {
        try {
          const q = encodeURIComponent($("cardQuery").value || "");
          const limit = encodeURIComponent($("cardLimit").value || "30");
          const data = await api(`/api/cards/search?q=${q}&limit=${limit}`);
          const rows = data.cards || [];
          if (!rows.length) {
            $("cardResults").innerHTML = "<div class='muted'>No cards found.</div>";
          } else {
            $("cardResults").innerHTML = `
              <table>
                <thead><tr><th>name</th><th>type</th><th>domain</th><th>lifecycle</th><th>path</th></tr></thead>
                <tbody>${rows.map((row) => `
                  <tr>
                    <td>${row.name || ""}</td>
                    <td>${row.type || ""}</td>
                    <td>${row.domain || ""}</td>
                    <td>${row.lifecycle || ""}</td>
                    <td><code>${row.path || ""}</code></td>
                  </tr>
                `).join("")}</tbody>
              </table>
            `;
          }
          showResponse(data);
        } catch (error) {
          alert(error.message);
        }
      });

      await loadDashboard();
      await loadProjects();
      switchView("dashboard");
    }
    init().catch((error) => {
      document.body.innerHTML = `<pre>Boot failed: ${String(error)}</pre>`;
    });
  </script>
</body>
</html>
"""
