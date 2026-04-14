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
from csv import DictReader
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Literal
from urllib.parse import parse_qs, unquote, urlparse

from alpha_lab.exceptions import AlphaLabConfigError, AlphaLabDataError, AlphaLabExperimentError
from alpha_lab.factor_recipe import factor_registry
from alpha_lab.real_cases.single_factor.pipeline import run_single_factor_case
from alpha_lab.reporting.renderers import write_case_report
from alpha_lab.research_bridge.categories import get_category_profile, list_categories
from alpha_lab.research_bridge.graph_view import VaultGraph
from alpha_lab.research_bridge.models import load_project_config, load_yaml_document, save_project_config
from alpha_lab.research_bridge.preflight import run_preflight
from alpha_lab.research_bridge.service import (
    PROJECTS_DIRNAME,
    explore_idea as bridge_explore_idea,
    init_project,
    normalize_fast_decision_log,
    refresh_project_pack,
    scaffold_case,
    summarize_run,
)
from alpha_lab.research_evaluation_config import (
    AVAILABLE_RESEARCH_EVALUATION_PROFILES,
    DEFAULT_RESEARCH_EVALUATION_CONFIG,
)
from alpha_lab.vault_export import resolve_vault_root

RunStatus = Literal["queued", "running", "succeeded", "failed"]

# Maximum bytes read from any text file served to the browser.
# Prevents the server from reading/sending huge artifacts that would freeze the UI.
_MAX_TEXT_BYTES: int = 512 * 1024  # 512 KB

# Maximum request body size accepted from the browser.
_MAX_REQUEST_BODY_BYTES: int = 2 * 1024 * 1024  # 2 MB

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
        raise AlphaLabConfigError(f"vault root does not exist or is not a directory: {resolved_vault}")

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


class _RunStore:
    def __init__(self) -> None:
        self._records: dict[str, _RunRecord] = {}
        self._tasks: dict[str, _RunTask] = {}
        self._lock = threading.Lock()

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
            progress_events=[{"ts": submitted_at, "message": "已提交到队列，等待调度", "percent": 0}],
        )
        with self._lock:
            self._records[record.run_id] = record
            self._tasks[record.run_id] = task
        worker = threading.Thread(target=self._execute_run, args=(record.run_id,), daemon=True)
        worker.start()
        return record.clone()

    def get(self, run_id: str) -> _RunRecord | None:
        with self._lock:
            record = self._records.get(run_id)
            return None if record is None else record.clone()

    def list(self, *, project_slug: str | None = None) -> list[_RunRecord]:
        with self._lock:
            records = [rec.clone() for rec in self._records.values()]
        if project_slug is None:
            return sorted(records, key=lambda item: item.submitted_at_utc, reverse=True)
        filtered = [item for item in records if item.project_slug == project_slug]
        return sorted(filtered, key=lambda item: item.submitted_at_utc, reverse=True)

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
        event = {
            "ts": ts,
            "message": message,
            "percent": record.progress_percent,
        }
        record.progress_events = [*record.progress_events[-7:], event]

    def _execute_run(self, run_id: str) -> None:
        with self._lock:
            record = self._records.get(run_id)
            task = self._tasks.get(run_id)
            if record is None or task is None:
                return
            record.status = "running"
            record.started_at_utc = _utc_now_iso()
            self._push_progress_locked(record, message="任务已启动，准备执行 single-factor pipeline", percent=2)
        try:
            progress_callback = lambda message, percent: self._push_progress(
                run_id,
                message=message,
                percent=percent,
            )
            result = run_single_factor_case(
                task.spec_path,
                output_root_dir=task.output_root_dir,
                evaluation_profile=task.evaluation_profile,
                vault_export_mode="skip",
                progress_callback=progress_callback,
            )
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
        except Exception as exc:
            error_payload = _build_run_error_payload(exc)
            with self._lock:
                stored = self._records[run_id]
                stored.status = "failed"
                stored.finished_at_utc = _utc_now_iso()
                stored.updated_at_utc = stored.finished_at_utc
                stored.progress_message = (
                    f"失败于：{stored.progress_message or '未知阶段'}"
                )
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


def _build_run_error_payload(exc: Exception) -> dict[str, str]:
    error_type = type(exc).__name__
    error_message = str(exc).strip() or repr(exc)
    if isinstance(exc, FileNotFoundError):
        hint = "检查 case spec 中的 prices_path、factor_path、exposures_path 是否存在且路径正确。"
    elif isinstance(exc, AlphaLabDataError):
        hint = "检查输入 CSV 的列名、日期列格式、factor_name 过滤后是否仍有数据，以及是否存在空文件。"
    elif isinstance(exc, AlphaLabConfigError):
        hint = "检查 case spec、evaluation profile、neutralization 配置是否完整且取值合法。"
    elif isinstance(exc, ValueError):
        hint = "检查参数取值、日期格式、空值以及 YAML/JSON/CSV 内容是否合法。"
    elif isinstance(exc, KeyError):
        hint = "通常表示输入表缺少必需列，或配置中引用了不存在的字段名。"
    else:
        hint = "优先查看失败阶段、核心报错和 traceback，定位是路径、数据 schema、配置还是代码逻辑问题。"
    return {
        "error_type": error_type,
        "error_message": error_message,
        "error_hint": hint,
    }


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
        runs = [item.to_payload() for item in self.run_store.list()]
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
                project for project in projects if str(project.get("lifecycle", "")).strip() == "active"
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
            return {"total_cards": 0, "inbox_count": self._count_inbox(), "by_type": {}, "by_lifecycle": {}}
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
                    items.append({
                        "name": f.name,
                        "directory": dirname,
                        "path": str(f),
                        "size_bytes": str(f.stat().st_size),
                        "modified": dt.datetime.fromtimestamp(
                            f.stat().st_mtime, tz=dt.UTC
                        ).isoformat().replace("+00:00", "Z"),
                    })
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
                            if (str(resolved).startswith(vault_str + "/") or
                                    str(resolved).startswith(vault_str + "\\")):
                                if resolved.is_file():
                                    return self._read_card_file(card_name, resolved)
            except OSError:
                pass  # fall through to directory scan

        # Fallback: shallow glob (top-level only per subdir) — no recursive scan
        for subdir in ("30_factors", "20_methods", "10_concepts", "40_papers",
                       "60_playbooks", "80_pipelines", "70_code_patterns", "50_experiments"):
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
            return {"cards": [], "index_path": str(index_path), "warning": "CARD-INDEX.tsv not found"}
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
        return {"ok": True, "matrix": summary, "coverage": coverage, "domain_coverage": domain_coverage, "stats": stats}

    def run_preflight_check(self, payload: dict[str, object]) -> dict[str, object]:
        """Run graph-based preflight checks for a candidate (category-aware)."""
        candidate_similar_raw = payload.get("candidate_similar") or []
        if isinstance(candidate_similar_raw, str):
            candidate_similar = [s.strip() for s in candidate_similar_raw.split(",") if s.strip()]
        else:
            candidate_similar = [str(s) for s in candidate_similar_raw if str(s).strip()]
        candidate_uses_data_raw = payload.get("candidate_uses_data") or []
        if isinstance(candidate_uses_data_raw, str):
            candidate_uses_data = [s.strip() for s in candidate_uses_data_raw.split(",") if s.strip()]
        else:
            candidate_uses_data = [str(s) for s in candidate_uses_data_raw if str(s).strip()]
        checked_paths_raw = payload.get("checked_card_paths") or []
        if isinstance(checked_paths_raw, str):
            checked_card_paths = [s.strip() for s in checked_paths_raw.split("\n") if s.strip()]
        else:
            checked_card_paths = [str(s) for s in checked_paths_raw if str(s).strip()]

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
            {"severity": i.severity, "code": i.code, "message": i.message}
            for i in report.issues
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
        normalize_fast_decision_log(vault_root=self.vault_root, project_slug=slug)
        project = load_project_config(paths["project_yaml"])
        cases = _list_cases(paths)
        docs = {
            "decision_log": _read_text(paths["decision_log"]),
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
            "cases": cases,
            "runs": [item.to_payload() for item in self.run_store.list(project_slug=project.slug)],
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
            max_research_level=int(payload.get("max_research_level") or 2),
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
            status.current_focus = str(payload.get("current_focus") or "").strip() or status.current_focus
        if "next_action" in payload:
            status.next_action = str(payload.get("next_action") or "").strip() or status.next_action
        save_project_config(project, paths["project_yaml"])
        refresh_project_pack(vault_root=self.vault_root, project_slug=slug)
        return self.get_project(slug)

    def refresh_project(self, slug: str) -> dict[str, object]:
        result = refresh_project_pack(vault_root=self.vault_root, project_slug=slug)
        return {"slug": result.project.slug, "project_dir": str(result.paths.project_dir)}

    # ---- Validation Console -----------------------------------------------

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
            lookback=int(payload.get("lookback") or 20),
            skip_recent=int(payload.get("skip_recent") or 5),
            target_horizon=int(payload.get("target_horizon") or 5),
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

    def list_evaluation_profiles(self) -> dict[str, object]:
        return {
            "profiles": sorted(AVAILABLE_RESEARCH_EVALUATION_PROFILES),
            "default_profile": "exploratory_screening",
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
            for item in self.run_store.list(project_slug=slug)
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
                float(row.get("dsr_pvalue") or 0.0),
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
            key=lambda row: float(row.get("abs_correlation") or 0.0),
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

    def summarize_run(self, slug: str, run_id: str, payload: dict[str, object]) -> dict[str, object]:
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
            raise AlphaLabConfigError(
                f"cannot delete run {run_id} while it is {record.status}"
            )
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
        return {"factors": items, "total": len(items), "custom_count": sum(1 for i in items if i.get("is_custom"))}

    def register_custom_factor(self, payload: dict[str, object]) -> dict[str, object]:
        """Register a custom factor from user-provided Python code."""
        name = str(payload.get("name") or "").strip().lower()
        if not name:
            raise ValueError("name is required")
        if not re.match(r"^[a-z][a-z0-9_]*$", name):
            raise ValueError("name must be lowercase alphanumeric with underscores, starting with a letter")
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
        return {"name": meta["name"], "code": meta.get("code", ""), "description": meta.get("description", "")}


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
            query = str((params.get("q") or [""])[0])
            limit = _safe_limit((params.get("limit") or ["50"])[0], default=50)
            self._send_json(self.svc.search_cards(query, limit=limit))
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
                if len(parts) == 4 and parts[3] == "cases":
                    self._send_json({"project_slug": slug, "cases": self.svc.list_cases(slug)})
                    return
                if len(parts) == 4 and parts[3] == "runs":
                    runs = [item.to_payload() for item in self.svc.run_store.list(project_slug=slug)]
                    self._send_json({"project_slug": slug, "runs": runs})
                    return
                if len(parts) == 5 and parts[3] == "diagnostics" and parts[4] == "factor-correlation":
                    self._send_json(self.svc.project_factor_diagnostics(slug))
                    return
                if len(parts) == 5 and parts[3] == "runs":
                    run = self.svc.run_store.get(parts[4])
                    if run is None or run.project_slug != slug:
                        self._send_json({"ok": False, "error": f"run not found: {parts[4]}"}, status=HTTPStatus.NOT_FOUND)
                        return
                    self._send_json(run.to_payload())
                    return
                if len(parts) == 7 and parts[3] == "runs" and parts[5] == "artifact":
                    self._handle_get_run_artifact(slug=slug, run_id=parts[4], artifact_key=parts[6])
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
                if len(parts) == 4 and parts[3] == "cases":
                    self._send_json(self.svc.create_case(slug, payload), status=HTTPStatus.CREATED)
                    return
                if len(parts) == 4 and parts[3] == "runs":
                    self._send_json(self.svc.submit_run(slug, payload), status=HTTPStatus.CREATED)
                    return
                if len(parts) == 6 and parts[3] == "runs" and parts[5] == "summarize":
                    self._send_json(self.svc.summarize_run(slug, parts[4], payload))
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

    def _handle_get_run_artifact(self, *, slug: str, run_id: str, artifact_key: str) -> None:
        run = self.svc.run_store.get(run_id)
        if run is None or run.project_slug != slug:
            self._send_json({"ok": False, "error": f"run not found: {run_id}"}, status=HTTPStatus.NOT_FOUND)
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
        # For text/JSON artifacts cap at _MAX_TEXT_BYTES; binary artifacts (e.g. plots) have
        # no cap but are served as-is since the browser handles them.
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
        "decision_log": project_dir / "decision_log.md",
        "runs_dir": project_dir / "runs",
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
        "portfolio_validation_recommendation",
        "level12_transition_label",
        "mean_ic",
        "mean_rank_ic",
        "ic_ir",
        "ic_t_stat",
        "ic_p_value",
        "dsr_pvalue",
        "split_description",
        "mean_long_short_return",
        "mean_long_short_turnover",
        "eval_coverage_ratio_mean",
        "coverage_mean",
        "coverage_min",
        "data_quality_status",
        "data_quality_suspended_rows",
        "data_quality_stale_rows",
        "data_quality_suspected_split_rows",
        "data_quality_integrity_warn_count",
        "data_quality_integrity_fail_count",
        "data_quality_hard_fail_count",
        "annualized_return",
        "sharpe",
        "max_drawdown",
        "factor_verdict_reasons",
        "campaign_triage_reasons",
        "promotion_reasons",
        "promotion_blockers",
        "portfolio_validation_major_risks",
        "rolling_instability_flags",
        "uncertainty_flags",
        "instability_flags",
    )
    summary: dict[str, object] = {}
    for key in keys:
        if key in metrics:
            summary[key] = metrics[key]
    return summary


def _load_run_rank_ic_timeseries(run: _RunRecord) -> dict[str, float]:
    path = _resolve_run_artifact_path(run, artifact_key="ic_timeseries", fallback_name="ic_timeseries.csv")
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
        fallback = (Path(run.output_dir).expanduser().resolve() / fallback_name)
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
        "coverage_ratio": (
            len(values_sorted) / n_runs_total if n_runs_total > 0 else None
        ),
        "median_dsr_pvalue": median,
        "min_dsr_pvalue": values_sorted[0] if values_sorted else None,
        "max_dsr_pvalue": values_sorted[-1] if values_sorted else None,
        "robust_count": robust_count,
        "watch_count": len(values_sorted) - robust_count - high_risk_count,
        "high_risk_count": high_risk_count,
    }


def _coerce_finite_float(value: object) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
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
    numerator = sum(a * b for a, b in zip(centered_left, centered_right))
    return numerator / (denom_left * denom_right)


def _as_text_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _optional_text(value: object) -> str | None:
    text = str(value or "").strip()
    return text or None


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


def _md_render_js() -> str:
    """Return the mdRender JS function as a raw string (no Python escape processing)."""
    return r"""
    function mdRender(text) {
      const SENTINEL = "@@BLK";
      const MATH_SENT = "@@MTH";
      function esc(s) {
        return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
      }
      function isTableRow(line) {
        const s = String(line || "").trim();
        return s.includes("|") && /^\|?.+\|.+\|?$/.test(s);
      }
      function isTableSeparator(line) {
        const s = String(line || "").trim();
        if (!s.includes("|")) return false;
        const body = s.replace(/^\|/, "").replace(/\|$/, "");
        const cells = body.split("|").map((cell) => cell.trim());
        return cells.length >= 2 && cells.every((cell) => /^:?-{3,}:?$/.test(cell));
      }
      function parseTableCells(line) {
        return String(line || "")
          .trim()
          .replace(/^\|/, "")
          .replace(/\|$/, "")
          .split("|")
          .map((cell) => cell.trim());
      }
      function inline(s) {
        s = esc(s);
        // Wikilinks — replace with placeholders first to protect from later regexes
        var wikiHolds = [];
        s = s.replace(/\[\[(.+?)(?:\|(.+?))?\]\]/g, function(_, page, alias) {
          var idx = wikiHolds.length;
          var label = (alias || page).trim();
          var cardPath = page.trim() + (page.trim().endsWith(".md") ? "" : ".md");
          wikiHolds.push('<span class="wikilink" data-action="selectCard" data-card-path="' +
            cardPath.replace(/"/g, "&quot;") + '" style="cursor:pointer">' + label + '</span>');
          return "@@WLK" + idx + "@@";
        });
        s = s.replace(/\*\*\*(.+?)\*\*\*/g, "<strong><em>$1</em></strong>");
        s = s.replace(/\*\*(.+?)\*\*/g,     "<strong>$1</strong>");
        s = s.replace(/\*(.+?)\*/g,         "<em>$1</em>");
        // Treat underscores as emphasis only at token boundaries so snake_case
        // identifiers such as asym_vol_reversal_v1 remain intact.
        s = s.replace(/(^|[^0-9A-Za-z])_([^_\n]+?)_(?=[^0-9A-Za-z]|$)/g, function(_, prefix, inner) {
          return prefix + "<em>" + inner + "</em>";
        });
        s = s.replace(/~~(.+?)~~/g,         "<del>$1</del>");
        s = s.replace(/`([^`]+)`/g,         "<code>$1</code>");
        s = s.replace(/\[(.+?)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
        s = s.replace(/\n/g, "<br>");
        // Restore wikilinks after all inline regexes
        s = s.replace(/@@WLK(\d+)@@/g, function(_, i) { return wikiHolds[+i]; });
        return s;
      }
      // 1. Protect fenced code blocks
      const blocks = [];
      text = text.replace(/```[^\n]*\n([\s\S]*?)```/g, function(_, code) {
        const idx = blocks.length;
        blocks.push('<pre class="code-block"><code>' + esc(code.replace(/\n$/, "")) + "</code></pre>");
        return SENTINEL + idx + "@@";
      });
      // 2. Protect math blocks so inline() doesn't mangle them
      const mathBlocks = [];
      // Display math $$...$$ (must come before inline $)
      text = text.replace(/\$\$([\s\S]+?)\$\$/g, function(m) {
        const idx = mathBlocks.length; mathBlocks.push(m); return MATH_SENT + idx + "@@";
      });
      // Inline math $...$ (single line, non-empty)
      text = text.replace(/\$([^\n$]+?)\$/g, function(m) {
        const idx = mathBlocks.length; mathBlocks.push(m); return MATH_SENT + idx + "@@";
      });
      // 3. Strip YAML frontmatter
      text = text.replace(/^---\n[\s\S]*?\n---\n?/, "");
      const lines = text.split("\n");
      const out = [];
      let i = 0;
      while (i < lines.length) {
        const raw = lines[i];
        if (raw.indexOf(SENTINEL) !== -1) {
          const m = raw.match(/@@BLK(\d+)@@/);
          if (m) { out.push(blocks[+m[1]]); i++; continue; }
        }
        if (raw.indexOf(MATH_SENT) !== -1) {
          // Math placeholder line — pass through as-is; restored to LaTeX below
          out.push(raw); i++; continue;
        }
        if (
          i + 1 < lines.length
          && isTableRow(raw)
          && isTableSeparator(lines[i + 1])
        ) {
          const header = parseTableCells(raw);
          const rows = [];
          i += 2;
          while (i < lines.length && isTableRow(lines[i]) && !isTableSeparator(lines[i])) {
            rows.push(parseTableCells(lines[i]));
            i++;
          }
          out.push(
            '<div class="artifact-table-wrap"><table><thead><tr>'
            + header.map((cell) => "<th>" + inline(cell) + "</th>").join("")
            + "</tr></thead><tbody>"
            + rows.map((row) => "<tr>" + header.map((_, idx) => "<td>" + inline(row[idx] || "") + "</td>").join("") + "</tr>").join("")
            + "</tbody></table></div>"
          );
          continue;
        }
        const hm = raw.match(/^(#{1,4}) +(.*)/);
        if (hm) { const lv = hm[1].length; out.push("<h"+lv+">"+inline(hm[2])+"</h"+lv+">"); i++; continue; }
        if (/^-{3,}\s*$/.test(raw)) { out.push("<hr>"); i++; continue; }
        if (raw.startsWith("> ")) {
          const bq = [];
          while (i < lines.length && lines[i].startsWith("> ")) { bq.push(lines[i].slice(2)); i++; }
          out.push("<blockquote>" + inline(bq.join("\n")) + "</blockquote>");
          continue;
        }
        if (/^[*+-] /.test(raw)) {
          const items = [];
          while (i < lines.length && /^[*+-] /.test(lines[i])) {
            items.push("<li>" + inline(lines[i].replace(/^[*+-] /, "")) + "</li>"); i++;
          }
          out.push("<ul>" + items.join("") + "</ul>"); continue;
        }
        if (/^\d+[.)]\s/.test(raw)) {
          const items = [];
          while (i < lines.length && /^\d+[.)]\s/.test(lines[i])) {
            items.push("<li>" + inline(lines[i].replace(/^\d+[.)]\s/, "")) + "</li>"); i++;
          }
          out.push("<ol>" + items.join("") + "</ol>"); continue;
        }
        if (!raw.trim()) { i++; continue; }
        const para = [];
        while (i < lines.length && lines[i].trim() &&
               !/^(#{1,4} |-{3,}|[*+-] |\d+[.)]\s|> |@@BLK|@@MTH)/.test(lines[i])) {
          para.push(lines[i]); i++;
        }
        if (para.length) out.push("<p>" + inline(para.join("\n")) + "</p>");
      }
      // 4. Restore math (leave raw LaTeX — MathJax will process it in the DOM)
      let result = out.join("\n");
      result = result.replace(/@@MTH(\d+)@@/g, function(_, idx) {
        var m = mathBlocks[+idx];
        if (m.substring(0, 2) === "$$") return '<div class="math-display">' + m + '</div>';
        return m;
      });
      return result;
    }
"""


def _index_html() -> str:
    return _index_html_raw().replace("@@MD_RENDER_JS@@", _md_render_js())


def _index_html_raw() -> str:
    return """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Alpha Lab - Unified Research Frontend</title>
  <style>
    :root {
      --bg: #f8fafc;
      --sidebar-bg: #f1f5f9;
      --sidebar-ink: #1e293b;
      --sidebar-muted: #64748b;
      --panel: #ffffff;
      --panel-hover: #f8fafc;
      --ink: #0f172a;
      --muted: #64748b;
      --brand: #0284c7;
      --brand-soft: #e0f2fe;
      --brand-dark: #0369a1;
      --ok: #10b981;
      --ok-soft: #d1fae5;
      --warn: #f59e0b;
      --warn-soft: #fef3c7;
      --bad: #ef4444;
      --bad-soft: #fee2e2;
      --line: #e2e8f0;
      --shadow: 0 1px 3px 0 rgb(0 0 0 / 0.1), 0 1px 2px -1px rgb(0 0 0 / 0.1);
      --shadow-lg: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
      --mono: "JetBrains Mono", "Menlo", "Monaco", "Consolas", monospace;
      --sans: "Inter", "system-ui", "-apple-system", "BlinkMacSystemFont", "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--ink);
      font-family: var(--sans);
      background-color: var(--bg);
      line-height: 1.5;
    }
    .layout {
      display: grid;
      grid-template-columns: 280px 1fr;
      min-height: 100vh;
    }
    .sidebar {
      background: var(--sidebar-bg);
      color: var(--sidebar-ink);
      padding: 24px 16px;
      position: sticky;
      top: 0;
      height: 100vh;
      overflow: auto;
      border-right: 1px solid var(--line);
      z-index: 100;
    }
    .brand {
      margin: 0 0 4px 0;
      font-size: 20px;
      letter-spacing: .05em;
      text-transform: uppercase;
      color: var(--sidebar-ink);
      font-weight: 800;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .sub {
      margin: 0 0 24px 0;
      color: var(--sidebar-muted);
      font-size: 14px;
      font-weight: 600;
    }
    .nav button {
      width: 100%;
      border: 1px solid transparent;
      background: transparent;
      color: var(--sidebar-ink);
      padding: 10px 14px;
      margin: 4px 0;
      text-align: left;
      border-radius: 8px;
      font-weight: 600;
      font-size: 16px;
      cursor: pointer;
      transition: all 0.2s ease;
    }
    .nav button:hover {
      background: rgba(0,0,0,0.04);
      color: var(--brand-dark);
    }
    .nav button.active {
      background: #ffffff;
      color: var(--brand);
      border: 1px solid var(--line);
      box-shadow: var(--shadow);
    }
    .nav button .badge {
      float: right;
      background: var(--brand-soft);
      color: var(--brand-dark);
      font-size: 13px;
      padding: 2px 8px;
      border-radius: 999px;
      font-weight: 700;
    }
    .main {
      padding: 32px;
      max-width: 92vw;
      margin: 0 auto;
      width: 100%;
    }
    .toolbar {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-bottom: 20px;
    }
    .toolbar button, button.action {
      border: 1px solid transparent;
      background: var(--brand);
      color: #ffffff;
      padding: 8px 18px;
      border-radius: 6px;
      cursor: pointer;
      font-weight: 600;
      font-size: 15px;
      transition: all 0.2s;
      box-shadow: var(--shadow);
    }
    .toolbar button:hover, button.action:hover {
      background: var(--brand-dark);
      box-shadow: var(--shadow-lg);
    }
    .toolbar button:active, button.action:active {
      transform: translateY(1px);
      box-shadow: none;
    }
    .toolbar button.ghost, button.ghost {
      background: #ffffff;
      color: var(--ink);
      border: 1px solid var(--line);
      box-shadow: none;
    }
    .toolbar button.ghost:hover, button.ghost:hover {
      background: var(--panel-hover);
      border-color: var(--muted);
    }
    .toolbar button.small, button.small {
      padding: 6px 12px;
      font-size: 14px;
    }
    button.action.action-violet {
      background: #8b5cf6;
      border-color: #7c3aed;
      color: #ffffff;
    }
    button.action.action-violet:hover {
      background: #7c3aed;
      border-color: #6d28d9;
      color: #ffffff;
    }
    button.action.action-success {
      background: var(--ok);
      border-color: #15803d;
      color: #ffffff;
    }
    button.action.action-success:hover {
      background: #15803d;
      border-color: #166534;
      color: #ffffff;
    }
    .cards {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
      gap: 16px;
      margin-bottom: 24px;
    }
    .card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 16px;
      box-shadow: var(--shadow);
      transition: transform 0.2s, box-shadow 0.2s;
    }
    .card:hover {
      transform: translateY(-2px);
      border-color: var(--brand-soft);
      box-shadow: var(--shadow-lg);
    }
    .card h3 {
      margin: 0 0 6px 0;
      font-size: 14px;
      text-transform: uppercase;
      letter-spacing: .05em;
      color: var(--muted);
      font-weight: 600;
    }
    .card .value {
      font-size: 28px;
      font-weight: 800;
      color: var(--ink);
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 20px;
    }
    .grid-3 {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 20px;
    }
    .grid > *,
    .grid-3 > *,
    .row > *,
    .row-3 > *,
    .workspace-dual > *,
    .bridge-layout > * {
      min-width: 0;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 20px;
      margin-bottom: 20px;
      box-shadow: var(--shadow);
    }
    .panel h2 {
      margin: 0 0 16px 0;
      font-size: 18px;
      color: var(--ink);
      font-weight: 700;
      border-left: 4px solid var(--brand);
      padding-left: 12px;
    }
    .panel.full { grid-column: 1 / -1; }
    textarea, input, select {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px 12px;
      font-family: var(--sans);
      font-size: 16px;
      margin: 4px 0 12px 0;
      background: #ffffff;
      color: var(--ink);
      transition: border-color 0.2s, box-shadow 0.2s;
    }
    textarea:focus, input:focus, select:focus {
      outline: none;
      border-color: var(--brand);
      box-shadow: 0 0 0 3px var(--brand-soft);
    }
    textarea {
      min-height: 120px;
      resize: vertical;
      font-family: var(--mono);
      font-size: 15px;
    }
    pre {
      white-space: pre-wrap;
      word-break: break-word;
      border-radius: 8px;
      border: 1px solid var(--line);
      background: #f1f5f9;
      color: var(--ink);
      padding: 14px;
      max-height: 400px;
      overflow: auto;
      font-family: var(--mono);
      font-size: 14px;
      line-height: 1.6;
    }
    .md-box {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      padding: 24px;
      max-height: 600px;
      overflow: auto;
      font-size: 16px;
      line-height: 1.8;
      color: var(--ink);
      min-width: 0;
      overflow-wrap: anywhere;
    }
    .md-box.raw-mode {
      white-space: pre-wrap;
      word-break: break-word;
      font-family: var(--mono);
      font-size: 15px;
      background: #f1f5f9;
    }
    .md-box h1 { font-size: 1.5em; margin: 1em 0 .5em; border-bottom: 2px solid var(--line); padding-bottom: .3em; color: var(--ink); }
    .md-box h2 { font-size: 1.25em; margin: 1.2em 0 .4em; color: var(--ink); font-weight: 700; }
    .md-box h3 { font-size: 1.1em; margin: 1em 0 .3em; color: var(--ink); }
    .md-box p  { margin: 0.8em 0; }
    .md-box ul, .md-box ol { margin: .6em 0 .6em 1.6em; padding: 0; }
    .md-box li { margin: .3em 0; }
    .md-box blockquote { border-left: 4px solid var(--brand); margin: 1em 0; padding: .5em 1.2em; color: var(--muted); background: var(--bg); border-radius: 0 8px 8px 0; }
    .md-box hr { border: none; border-top: 1px solid var(--line); margin: 1.5em 0; }
    .md-box code { font-family: var(--mono); font-size: .9em; background: var(--brand-soft); padding: .2em .4em; border-radius: 4px; color: var(--brand-dark); }
    .md-box pre.code-block { background: #f1f5f9; color: var(--ink); border: 1px solid var(--line); border-radius: 10px; padding: 16px; overflow: auto; margin: 1em 0; }
    .md-box pre.code-block code { background: none; color: inherit; padding: 0; font-size: .9em; }
    .md-box a { color: var(--brand); text-decoration: none; border-bottom: 1px solid transparent; transition: border-color 0.2s; }
    .md-box a:hover { border-bottom-color: var(--brand); }
    label {
      font-size: 14px;
      font-weight: 700;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.02em;
    }
    table {
      width: 100%;
      border-collapse: separate;
      border-spacing: 0;
      font-size: 15px;
      margin: 8px 0;
    }
    th, td {
      border-bottom: 1px solid var(--line);
      padding: 12px 10px;
      text-align: left;
      vertical-align: middle;
    }
    th { font-size: 13px; color: var(--muted); text-transform: uppercase; letter-spacing: .05em; font-weight: 700; background: #f8fafc; }
    tr:hover td { background: var(--panel-hover); }
    .status {
      display: inline-flex;
      align-items: center;
      font-size: 13px;
      font-weight: 700;
      padding: 2px 10px;
      border-radius: 999px;
      background: #f1f5f9;
      color: var(--muted);
      text-transform: uppercase;
    }
    .status.running { background: var(--brand-soft); color: var(--brand-dark); }
    .status.succeeded { background: var(--ok-soft); color: var(--ok); }
    .status.failed { background: var(--bad-soft); color: var(--bad); }
    .status.queued { background: var(--warn-soft); color: var(--warn); }
    .status.pending { background: #f1f5f9; color: var(--muted); }
    .status.approved { background: var(--ok-soft); color: var(--ok); }
    .status.applied { background: var(--brand-soft); color: var(--brand-dark); }
    .muted { color: var(--muted); font-size: 14px; }
    .row {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
    }
    .row-3 {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 16px;
    }
    .copy-btn {
      border: 1px solid var(--line);
      background: #ffffff;
      color: var(--brand);
      padding: 6px 14px;
      border-radius: 8px;
      cursor: pointer;
      font-size: 14px;
      font-weight: 600;
      transition: all 0.2s;
    }
    .copy-btn:hover { background: var(--brand-soft); border-color: var(--brand); }
    .artifact-link {
      color: var(--brand);
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      font-size: 14px;
      font-weight: 600;
      padding: 4px 9px;
      border-radius: 999px;
      background: var(--brand-soft);
      transition: background 0.2s;
      white-space: nowrap;
    }
    .artifact-link:hover { background: #bae6fd; text-decoration: underline; }
    .artifact-link.artifact-doc {
      background: #ecfeff;
      color: #0f766e;
    }
    .artifact-link.artifact-doc:hover {
      background: #ccfbf1;
    }
    .artifact-link.artifact-data {
      background: #eff6ff;
      color: #1d4ed8;
    }
    .artifact-link.artifact-data:hover {
      background: #dbeafe;
    }
    .artifact-link.artifact-detail {
      background: #f8fafc;
      color: #475569;
    }
    .artifact-link.artifact-detail:hover {
      background: #e2e8f0;
    }
    .metrics-screening {
      display: grid;
      gap: 8px;
      font-size: 14px;
      min-width: 300px;
    }
    .metrics-screening-row {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      align-items: center;
    }
    .metrics-chip {
      display: inline-flex;
      align-items: center;
      gap: 4px;
      padding: 3px 8px;
      border-radius: 999px;
      background: #f1f5f9;
      color: var(--ink);
      font-size: 13px;
      font-weight: 700;
    }
    .metrics-chip.good {
      background: var(--ok-soft);
      color: #047857;
    }
    .metrics-chip.warn {
      background: var(--warn-soft);
      color: #b45309;
    }
    .metrics-chip.bad {
      background: var(--bad-soft);
      color: #b91c1c;
    }
    .metrics-screening-kv {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      color: var(--muted);
      font-size: 13px;
    }
    .metrics-screening-notes {
      display: grid;
      gap: 4px;
    }
    .metrics-screening-note {
      color: var(--ink);
      font-size: 13px;
      line-height: 1.45;
    }
    .metrics-screening-note strong {
      margin-right: 4px;
      color: var(--muted);
      font-size: 12px;
      letter-spacing: 0.03em;
    }
    .artifact-links {
      display: flex;
      flex-wrap: wrap;
      gap: 8px 6px;
      min-width: 320px;
      align-items: flex-start;
    }
    .artifact-cell {
      min-width: 380px;
      vertical-align: top;
    }
    .artifact-groups {
      display: grid;
      gap: 8px;
      min-width: 340px;
    }
    .artifact-group {
      border: 1px solid var(--line);
      border-radius: 10px;
      background: #ffffff;
      padding: 8px 10px;
    }
    .artifact-group-title {
      margin-bottom: 8px;
      color: var(--muted);
      font-size: 13px;
      font-weight: 700;
      letter-spacing: 0.03em;
    }
    .artifact-group summary {
      cursor: pointer;
      color: var(--muted);
      font-size: 13px;
      font-weight: 700;
      letter-spacing: 0.03em;
      list-style: none;
    }
    .artifact-group summary::-webkit-details-marker {
      display: none;
    }
    .artifact-group[open] summary {
      margin-bottom: 8px;
    }
    .artifact-viewer-shell {
      display: grid;
      gap: 12px;
    }
    .artifact-viewer-head {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      flex-wrap: wrap;
    }
    .artifact-viewer-title {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
    }
    .artifact-viewer-kind {
      display: inline-flex;
      align-items: center;
      padding: 2px 8px;
      border-radius: 999px;
      background: var(--brand-soft);
      color: var(--brand-dark);
      font-size: 13px;
      font-weight: 700;
    }
    .artifact-viewer-meta {
      color: var(--muted);
      font-size: 14px;
    }
    .artifact-viewer-raw {
      margin: 0;
      max-height: 560px;
      background: #f8fafc;
    }
    .artifact-json-grid {
      display: grid;
      gap: 10px;
    }
    .artifact-json-card {
      border: 1px solid var(--line);
      border-radius: 10px;
      background: #ffffff;
      padding: 12px 14px;
    }
    .artifact-json-card strong {
      color: var(--muted);
      font-size: 13px;
      letter-spacing: 0.02em;
    }
    .artifact-json-card pre {
      margin-top: 8px;
      max-height: 260px;
    }
    .artifact-table-wrap {
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: #ffffff;
    }
    .artifact-table-wrap table {
      margin: 0;
      min-width: 640px;
    }
    .artifact-chart-card {
      border: 1px solid var(--line);
      border-radius: 10px;
      background: #ffffff;
      padding: 10px 12px;
      display: grid;
      gap: 8px;
    }
    .artifact-chart-title {
      margin: 0;
      color: var(--ink);
      font-size: 14px;
      font-weight: 700;
    }
    .artifact-line-chart {
      width: 100%;
      height: auto;
      display: block;
      border-radius: 8px;
      background: #f8fafc;
    }
    .artifact-chart-legend {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      color: var(--muted);
      font-size: 12px;
    }
    .artifact-chart-legend-item {
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }
    .artifact-bar-list {
      display: grid;
      gap: 8px;
    }
    .artifact-bar-row {
      display: grid;
      gap: 4px;
      font-size: 12px;
    }
    .artifact-bar-head {
      display: flex;
      justify-content: space-between;
      color: var(--muted);
    }
    .artifact-bar-track {
      height: 9px;
      border-radius: 999px;
      background: #e2e8f0;
      overflow: hidden;
    }
    .artifact-bar-fill {
      height: 100%;
      border-radius: 999px;
      background: linear-gradient(90deg, #0ea5e9, #22c55e);
    }
    .artifact-chart-dot {
      width: 10px;
      height: 10px;
      border-radius: 999px;
      display: inline-block;
      flex-shrink: 0;
    }
    .project-diagnostics-box {
      border: 1px solid var(--line);
      border-radius: 10px;
      background: #ffffff;
      padding: 12px 14px;
      min-height: 74px;
    }
    .diag-meta {
      font-size: 13px;
      color: var(--muted);
      margin-bottom: 10px;
    }
    .diag-heatmap-wrap {
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
      margin-bottom: 10px;
    }
    .diag-heatmap {
      border-collapse: collapse;
      width: 100%;
      min-width: 520px;
      font-size: 12px;
      font-family: var(--mono);
    }
    .diag-heatmap th,
    .diag-heatmap td {
      border: 1px solid #e2e8f0;
      padding: 6px 8px;
      text-align: center;
      white-space: nowrap;
    }
    .diag-heatmap th {
      background: #f8fafc;
      color: var(--muted);
      font-weight: 700;
    }
    .diag-redundancy-list {
      display: grid;
      gap: 6px;
    }
    .diag-redundancy-item {
      border-left: 3px solid #f59e0b;
      background: #fffbeb;
      padding: 6px 8px;
      font-size: 13px;
      color: #92400e;
    }
    .diag-redundancy-item.high {
      border-left-color: #ef4444;
      background: #fef2f2;
      color: #991b1b;
    }
    .diag-dsr-list {
      display: grid;
      gap: 6px;
    }
    .diag-dsr-item {
      border-left: 3px solid #94a3b8;
      background: #f8fafc;
      padding: 6px 8px;
      font-size: 13px;
      color: #334155;
    }
    .diag-dsr-item.good {
      border-left-color: #22c55e;
      background: #f0fdf4;
      color: #166534;
    }
    .diag-dsr-item.watch {
      border-left-color: #f59e0b;
      background: #fffbeb;
      color: #92400e;
    }
    .diag-dsr-item.high {
      border-left-color: #ef4444;
      background: #fef2f2;
      color: #991b1b;
    }
    .run-error-box {
      margin-top: 4px;
      padding: 10px 12px;
      background: var(--bad-soft);
      border: 1px solid #fca5a5;
      border-radius: 8px;
      color: #991b1b;
    }
    .run-error-box > strong {
      display: block;
      margin-bottom: 6px;
      font-size: 13px;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }
    .run-error-box pre {
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      overflow-x: auto;
      font-family: var(--mono);
      font-size: 13px;
      line-height: 1.5;
    }
    .run-status-cell {
      min-width: 230px;
    }
    .run-progress-text {
      margin-top: 6px;
      font-size: 14px;
      color: var(--ink);
    }
    .run-progress-bar {
      width: 100%;
      height: 6px;
      margin-top: 8px;
      background: #e2e8f0;
      border-radius: 999px;
      overflow: hidden;
    }
    .run-progress-fill {
      height: 100%;
      background: linear-gradient(90deg, var(--brand), var(--brand-dark));
      border-radius: 999px;
      transition: width 0.25s ease;
    }
    .run-progress-meta {
      margin-top: 6px;
      font-size: 13px;
      color: var(--muted);
    }
    .run-event-trail {
      margin-top: 8px;
    }
    .run-event-trail summary {
      cursor: pointer;
      font-size: 13px;
      color: var(--brand-dark);
      user-select: none;
    }
    .run-event-list {
      margin-top: 8px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #f8fafc;
      overflow: hidden;
    }
    .run-event-item {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      padding: 8px 10px;
      font-size: 13px;
      border-bottom: 1px solid var(--line);
    }
    .run-event-item:last-child {
      border-bottom: 0;
    }
    .run-event-item time {
      color: var(--muted);
      white-space: nowrap;
      font-family: var(--mono);
    }
    .run-error-summary {
      display: grid;
      gap: 6px;
      font-size: 14px;
      margin-bottom: 8px;
    }
    .run-error-summary strong {
      display: inline;
      margin: 0 6px 0 0;
      font-size: 13px;
      letter-spacing: 0.02em;
      text-transform: uppercase;
    }
    .run-error-summary code {
      font-family: var(--mono);
      font-size: 13px;
      background: rgba(255,255,255,0.6);
      padding: 1px 6px;
      border-radius: 4px;
    }
    .sidebar-section {
      margin-top: 24px;
      padding-top: 16px;
      border-top: 1px solid var(--line);
    }
    .sidebar-section h2 {
      margin: 0 0 10px 0;
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: .08em;
      color: var(--sidebar-muted);
      font-weight: 700;
    }
    .sidebar-section select {
      background: #ffffff;
      border-color: var(--line);
      color: var(--ink);
    }
    .sidebar-section button.ghost {
      background: #ffffff;
      border-color: var(--line);
      color: var(--ink);
    }
    .sidebar-section button.ghost:hover {
      background: var(--bg);
    }
    @media (max-width: 1200px) {
      .layout { grid-template-columns: 1fr; }
      .sidebar { height: auto; position: static; border-right: 0; border-bottom: 1px solid var(--line); }
      .grid, .row, .grid-3, .row-3, .bridge-layout { grid-template-columns: 1fr; }
      .main { padding: 16px; max-width: 100vw; overflow-x: hidden; }
      .bridge-layout { gap: 16px; }
      .project-hub-sticky { position: static; width: 100%; max-width: 800px; margin: 0 auto; }
      .hub-body { padding: 16px; }
    }
    
    /* Custom scrollbar for professional look */
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 10px; }
    ::-webkit-scrollbar-thumb:hover { background: #94a3b8; }
    .sidebar::-webkit-scrollbar-thumb { background: #cbd5e1; }
    /* === Command Center Styles === */
    .workflow-header {
      display: flex; align-items: center; gap: 12px; margin-bottom: 24px; padding: 14px 24px;
      background: #ffffff; border: 1px solid var(--line); border-left: 4px solid var(--brand);
      border-radius: 8px; color: var(--ink); box-shadow: var(--shadow);
    }
    .workflow-step-num {
      width: 30px; height: 30px; background: var(--brand-soft); border: 2px solid #bae6fd;
      border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 800; font-size: 16px;
      color: var(--brand-dark); font-family: var(--mono);
    }
    .panel-inspiration {
      background: var(--panel);
      border: 1px solid var(--line); border-left: 4px solid var(--brand);
    }
    .panel-guardrail {
      background: var(--panel); border: 1px solid var(--line); border-left: 4px solid var(--muted);
    }
    .panel-ai-workspace {
      border: 1px solid var(--line); border-top: 3px solid var(--brand); background: var(--panel);
    }
    /* === Improved Idea Lab Layout === */
    .inspiration-workspace {
      display: flex;
      gap: 24px;
      align-items: flex-start;
    }
    .inspiration-input-area {
      flex: 1.6;
      min-width: 0; /* Critical for preventing flex blowout */
    }
    .explore-mode-panel {
      flex: 1;
      min-width: 360px;
      max-width: 500px;
    }
    .explore-mode-option {
      display: grid;
      grid-template-columns: 16px 64px minmax(0, 1fr);
      align-items: flex-start;
      column-gap: 12px;
      cursor: pointer;
      margin-bottom: 10px;
      padding: 12px 16px;
      border-radius: 12px;
      border: 1px solid var(--line);
      background: #ffffff;
      transition: all 0.2s ease;
    }
    .explore-mode-option:hover {
      border-color: var(--brand);
      transform: translateX(4px);
      box-shadow: var(--shadow);
    }
    .explore-mode-key {
      width: 64px;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 24px;
      padding: 0 6px;
      border-radius: 6px;
      background: var(--bg);
      color: var(--muted);
      font-size: 11px;
      font-weight: 800;
      font-family: var(--mono);
      text-transform: uppercase;
      border: 1px solid var(--line);
    }
    .explore-mode-copy {
      min-width: 0;
    }
    .explore-mode-desc {
      display: block;
      font-weight: 700;
      font-size: 13px;
      color: var(--ink);
      margin-bottom: 4px;
    }
    .explore-mode-intent {
      display: block;
      font-size: 11px;
      line-height: 1.5;
      color: var(--muted);
    }
    .explore-mode-option:has(input:checked) {
      border-color: var(--brand);
      background: var(--brand-soft);
      box-shadow: 0 4px 12px rgba(2, 132, 199, 0.08);
    }
    .explore-mode-option:has(input:checked) .explore-mode-key {
      background: var(--brand);
      color: #fff;
      border-color: var(--brand);
    }
    .explore-mode-option:has(input:checked) .explore-mode-desc {
      color: var(--brand-dark);
    }
    .section-tag {
      display: inline-block; padding: 3px 10px; background: var(--bg); color: var(--muted);
      border-radius: 6px; font-size: 12px; font-weight: 800; text-transform: uppercase; margin-bottom: 12px; letter-spacing: 0.05em; border: 1px solid var(--line);
    }
    .section-tag.section-tag-violet {
      background: #8b5cf6;
      border-color: #7c3aed;
      color: #ffffff;
    }
    .idea-prompt-hint { font-style: italic; color: var(--muted); font-size: 15px; margin: -4px 0 16px 0; }
    .bridge-layout { display: grid; grid-template-columns: 7fr 3fr; gap: 32px; align-items: start; }
    .project-hub-sticky { position: sticky; top: 32px; min-width: 0; }

    .project-hub-stack {
      display: grid;
      gap: 16px;
    }
    .workspace-dual {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
      align-items: start;
    }
    .workspace-dual .md-box {
      max-height: 260px;
      padding: 16px;
    }
    .report-doc-box {
      max-height: none !important;
      overflow: visible !important;
      min-height: 200px;
      padding: 16px;
    }
    .factor-workshop-details {
      border: 1px solid rgba(139, 92, 246, 0.25);
      border-radius: 10px;
      background: #faf5ff;
      padding: 10px 12px;
    }
    .factor-workshop-summary {
      cursor: pointer;
      font-family: var(--mono);
      font-size: 13px;
      letter-spacing: 0.05em;
      text-transform: uppercase;
      color: #6d28d9;
      font-weight: 700;
    }
    .project-modal-backdrop {
      position: fixed;
      inset: 0;
      z-index: 50;
      background: rgba(15, 23, 42, 0.45);
      display: none;
      align-items: center;
      justify-content: center;
      padding: 20px;
    }
    .project-modal-backdrop.active { display: flex; }
    .project-modal {
      width: min(760px, 96vw);
      max-height: 90vh;
      overflow: auto;
      background: #ffffff;
      border: 1px solid var(--line);
      border-radius: 12px;
      box-shadow: 0 24px 64px rgba(15, 23, 42, 0.25);
    }
    .project-modal .hub-header {
      position: sticky;
      top: 0;
      z-index: 1;
      background: #f8fafc;
    }
    
    /* === Geeky Hub Styles === */
    .hub-panel {
      background: var(--panel); border: 1px solid var(--line); border-radius: 10px;
      box-shadow: var(--shadow); overflow: hidden;
      min-width: 0; /* Essential for grid child shrinking */
    }
    .hub-header {
      background: #f8fafc; color: var(--ink); padding: 12px 20px;
      border-bottom: 1px solid var(--line); display: flex; justify-content: space-between; align-items: center;
      min-width: 0;
    }
    .hub-header h2 { 
      margin: 0; font-size: 14px; font-weight: 700; letter-spacing: 0.08em; border: none; padding: 0; 
      color: var(--ink); text-transform: uppercase; font-family: var(--mono);
      white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }
    .hub-header .pulse { flex-shrink: 0; width: 8px; height: 8px; background: var(--ok); border-radius: 50%; box-shadow: 0 0 8px var(--ok-soft); }
    .hub-body { 
      padding: 20px; 
      min-width: 0; 
      overflow-wrap: anywhere; /* Prevent long strings from stretching the panel */
    }
    .hub-section-title {
      font-size: 12px; text-transform: uppercase; color: var(--muted); font-weight: 800;
      letter-spacing: 0.1em; margin: 20px 0 12px 0; display: flex; align-items: center; gap: 8px;
      min-width: 0;
    }
    .hub-section-title::after { content: ""; flex: 1; height: 1px; background: var(--line); min-width: 10px; }
    .hub-section-explainer {
      margin: -6px 0 10px 0;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.45;
    }
    .hub-box {
      background: #ffffff; border: 1px solid var(--line); border-radius: 8px; padding: 14px;
      font-family: var(--mono); font-size: 14px; box-shadow: 0 1px 2px rgba(0,0,0,0.02);
      min-width: 0;
      overflow-wrap: anywhere;
    }
    .hub-box table { width: 100%; table-layout: fixed; }
    .hub-box td { overflow-wrap: anywhere; word-break: break-word; }
    .hub-box label { font-family: var(--sans); font-size: 12px; color: var(--muted); display: block; margin-bottom: 4px; font-weight: 700; }
    .hub-box input, .hub-box textarea, .hub-box select {
      font-family: var(--mono); font-size: 14px; background: #f8fafc; border-color: var(--line); color: var(--ink); margin-bottom: 12px;
    }
    .hub-box input:focus, .hub-box textarea:focus { border-color: var(--brand); background: #ffffff; box-shadow: 0 0 0 2px var(--brand-soft); }
    .case-table-wrap, .run-table-wrap {
      width: 100%;
      overflow-x: auto;
    }
    .case-table { table-layout: fixed; }
    .run-table {
      table-layout: auto;
      width: 100%;
    }
    .run-table th:nth-child(1) { width: 90px; }   /* run_id */
    .run-table th:nth-child(2) { width: 18%; }     /* case */
    .run-table th:nth-child(3) { width: 10%; }     /* profile */
    .run-table th:nth-child(4) { width: 70px; }    /* status */
    .run-table th:nth-child(5) { min-width: 160px; } /* metrics */
    .run-table th:nth-child(6) { min-width: 120px; } /* artifacts */
    .run-table th:nth-child(7) { min-width: 100px; } /* actions */
    .run-table td { vertical-align: top; }
    .case-name-cell,
    .run-case-cell {
      overflow-wrap: anywhere;
      word-break: break-word;
    }
    .case-name-cell code,
    .run-case-cell code {
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      word-break: break-word;
    }
    .case-flag-cell,
    .case-action-cell {
      white-space: nowrap;
      width: 1%;
    }
    /* Keyboard navigation hints */
    .kbd {
      display: inline-block; width: 18px; height: 18px; line-height: 16px;
      text-align: center; font-size: 12px; font-weight: 700; font-family: var(--mono);
      border: 1px solid var(--line); border-radius: 4px;
      margin-right: 8px; color: var(--muted); background: var(--bg); vertical-align: middle;
    }
    .nav button.active .kbd { border-color: var(--brand); color: var(--brand); background: #ffffff; }
    .nav button:hover .kbd { border-color: var(--muted); }
    /* Dark panel heading */
    .panel h2 { color: var(--ink); }
    /* Section tag in dark */
    .section-tag { color: var(--muted); background: var(--bg); }
    .idea-prompt-hint { color: var(--muted); }
    @media (max-width: 900px) {
      .explore-mode-panel { min-width: 0; max-width: none; width: 100%; }
      .explore-mode-option {
        grid-template-columns: 14px 48px minmax(0, 1fr);
      }
      .explore-mode-key {
        width: 48px;
      }
      .workspace-dual {
        grid-template-columns: 1fr;
      }
    }
  </style>
  <script>
    /* MathJax config — only $...$ and $$...$$ delimiters (no backslash variants
       to avoid Python/JS string escaping conflicts). typeset:false so we call
       typesetPromise manually after each card render. */
    MathJax = {
      tex: {
        inlineMath: [["$","$"]],
        displayMath: [["$$","$$"]],
        processEscapes: true
      },
      options: { skipHtmlTags: ["script","noscript","style","textarea","pre","code"] },
      startup: { typeset: false }
    };
  </script>
  <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js" async></script>
</head>
<body>
  <div class="layout">
    <aside class="sidebar">
      <h1 class="brand"><span style="color:var(--brand);font-family:var(--mono)">α</span> ALPHA LAB</h1>
      <p class="sub">Research Terminal v2</p>
      <div class="nav">
        <button data-view="dashboard" class="active"><span class="kbd">0</span>总览 Overview</button>
        <button data-view="knowledge"><span class="kbd">1</span>知识库 Knowledge</button>
        <button data-view="bridge"><span class="kbd">2</span>工作台 Workspace</button>
        <button data-view="writeback"><span class="kbd">3</span>导出台 Export</button>
      </div>
      <div class="sidebar-section">
        <h2>当前项目</h2>
        <select id="projectSelect"><option value="">-- 加载中 --</option></select>
        <button class="ghost small" id="reloadProjectBtn" style="width:100%">重新加载项目</button>
      </div>
      <div class="sidebar-section">
        <h2>快捷操作</h2>
        <button class="ghost small" id="refreshProjectBtn" style="width:100%">刷新上下文包</button>
        <button class="ghost small" id="btnOpenCreateProject" style="width:100%;margin-top:8px">新建项目</button>
        <div id="sidebarStatus" style="margin-top:6px;font-size: 13px;color:var(--muted);min-height:16px"></div>
      </div>
    </aside>

    <main class="main">

      <!-- ============ DASHBOARD ============ -->
      <section id="view-dashboard">
        <div class="workflow-header">
          <div class="workflow-step-num">0</div>
          <h2 style="margin:0;font-size: 17px;border:0;padding:0;letter-spacing:0.08em;font-family:var(--mono)">SYSTEM_OVERVIEW</h2>
          <span class="muted" style="margin-left:auto;font-size: 13px;font-family:var(--mono)">系统总览</span>
        </div>
        
        <div id="dashboardCards" class="cards"></div>
        
        <div class="grid">
          <div class="hub-panel">
            <div class="hub-header">
              <h2>最近运行 (Recent Runs)</h2>
              <div class="pulse" title="Live"></div>
            </div>
            <div class="hub-body">
              <div id="recentRuns"></div>
            </div>
          </div>
          <div class="hub-panel">
            <div class="hub-header" style="border-bottom-color: var(--warn)">
              <h2>待办事项 (Next Actions)</h2>
            </div>
            <div class="hub-body">
              <div id="nextActions"></div>
            </div>
          </div>
        </div>

        <!-- ============ 使用指南 ============ -->
        <details class="hub-panel" style="margin-top:24px">
          <summary style="cursor:pointer;font-weight:700;font-size: 14px;padding:14px 20px;background:#f8fafc;color:var(--brand);letter-spacing:0.08em;text-transform:uppercase;font-family:var(--mono)">
            ALPHA_LAB USER_GUIDE
          </summary>
          <div class="hub-body" style="line-height:1.8;font-size: 15px;border-top:1px solid var(--line)">

            <h3 style="margin:0 0 8px 0;font-size: 14px;color:var(--brand);text-transform:uppercase;letter-spacing:0.05em">前提条件 (Prerequisites)</h3>
            <ul style="margin:0 0 20px 0;padding-left:18px;color:var(--muted)">
              <li>已在终端执行 <code>alpha-lab web unified --vault-root /path/to/quant-knowledge</code> 启动服务</li>
              <li>或设置了环境变量 <code>OBSIDIAN_VAULT_PATH</code> 后直接执行 <code>alpha-lab web unified</code></li>
              <li>vault 目录下需存在 <code>90_moc/CARD-INDEX.tsv</code>，卡片搜索依赖它</li>
            </ul>

            <h3 style="margin:0 0 8px 0;font-size: 14px;color:var(--brand);text-transform:uppercase;letter-spacing:0.05em">整体工作流 (Workflow)</h3>
            <ol style="margin:0 0 20px 0;padding-left:18px;color:var(--muted)">
              <li><strong>A. 知识库</strong> — 搜索相关因子卡片，确认研究假设；查看 inbox 待处理文件</li>
              <li><strong>B. 研究工作台</strong> — 新建项目（弹窗）→ 明确 hypothesis / focus → 刷新 current case</li>
              <li><strong>C. 工作台内验证</strong> — 在同一页 Step 4/5 完成运行、结果解读与状态同步</li>
              <li><strong>D. 导出台</strong> — 仅在需要正式写回知识库时使用</li>
            </ol>

            <div class="grid" style="gap:24px;margin-top:24px">
              <div>
                <h3 style="margin:0 0 8px 0;font-size: 14px;color:var(--brand-dark);text-transform:uppercase">A. 知识库操作</h3>
                <ul style="margin:0;padding-left:18px;color:var(--muted);font-size: 14px">
                  <li><strong>卡片搜索</strong>：输入关键词（如 <code>momentum</code>），点击"搜索"</li>
                  <li><strong>卡片查看器</strong>：输入卡片文件名，点"读取"查看全文</li>
                  <li><strong>待处理文件</strong>：显示 <code>00_inbox/</code> 中的文件</li>
                </ul>
              </div>
              <div>
                <h3 style="margin:0 0 8px 0;font-size: 14px;color:var(--brand-dark);text-transform:uppercase">B. 工作台操作</h3>
                <ul style="margin:0;padding-left:18px;color:var(--muted);font-size: 14px">
                  <li><strong>新建项目</strong>：点击侧栏或右侧 NEW_PROJECT 打开弹窗初始化</li>
                  <li><strong>状态同步</strong>：维护 hypothesis / focus / next action</li>
                  <li><strong>刷新 Current Case</strong>：填写参数生成当前实验合同</li>
                </ul>
              </div>
            </div>

            <div class="grid" style="gap:24px;margin-top:24px">
              <div>
                <h3 style="margin:0 0 8px 0;font-size: 14px;color:var(--brand-dark);text-transform:uppercase">C. 工作台内验证（Step 4/5）</h3>
                <ul style="margin:0;padding-left:18px;color:var(--muted);font-size: 14px">
                  <li><strong>启动实验</strong>：在 Workspace 的 Step 4 选择 current case 和 profile，后台运行</li>
                  <li><strong>查看产物</strong>：点击 artifact 链接内联查看</li>
                  <li><strong>结果解读 + 同步</strong>：在 Step 5 审阅 latest result / decision log 并回填状态</li>
                </ul>
              </div>
              <div>
                <h3 style="margin:0 0 8px 0;font-size: 14px;color:var(--brand-dark);text-transform:uppercase">D. 审核操作</h3>
                <ul style="margin:0;padding-left:18px;color:var(--muted);font-size: 14px">
                  <li><strong>审阅</strong>：只在需要正式 writeback 时查看草稿</li>
                  <li><strong>标注</strong>：填写结论和审阅状态</li>
                  <li><strong>回写</strong>：把正式结论写回知识库</li>
                </ul>
              </div>
            </div>

          </div>
        </details>
      </section>

      <!-- ============ A. KNOWLEDGE OPS ============ -->
      <section id="view-knowledge" style="display:none">
        <div class="workflow-header">
          <div class="workflow-step-num">1</div>
          <h2 style="margin:0;font-size: 17px;border:0;padding:0;letter-spacing:0.08em;font-family:var(--mono)">KNOWLEDGE_OPS</h2>
          <span class="muted" style="margin-left:auto;font-size: 13px;font-family:var(--mono)">知识库</span>
        </div>
        
        <div class="grid-3">
          <div class="hub-panel">
            <div class="hub-header">
              <h2>知识库统计 (Stats)</h2>
            </div>
            <div class="hub-body">
              <div id="vaultStats"></div>
              <button class="ghost small" id="btnRefreshVaultStats" style="margin-top:12px;width:100%;font-family:var(--mono)">REFRESH_STATS</button>
            </div>
          </div>
          <div class="hub-panel" style="grid-column: span 2">
            <div class="hub-header">
              <h2>待处理文件 (Inbox / Sources)</h2>
            </div>
            <div class="hub-body">
              <div id="inboxList"></div>
              <button class="ghost small" id="btnRefreshInbox" style="margin-top:12px;font-family:var(--mono)">REFRESH_INBOX</button>
            </div>
          </div>
        </div>

        <div class="grid" style="margin-top:24px">
          <div class="hub-panel">
            <div class="hub-header">
              <h2>卡片搜索 (Search Engine)</h2>
            </div>
            <div class="hub-body">
              <p class="muted" style="margin:0 0 12px 0;font-size: 14px">在 CARD-INDEX.tsv 中全文检索；支持因子名、类型、领域等关键词</p>
              <div class="hub-box">
                <label>KEYWORDS (如：momentum、factor、ashare)</label>
                <input id="cardQuery" placeholder="输入搜索关键词...">
                <label>MAX_RESULTS</label>
                <input id="cardLimit" value="30">
                <button class="action small" id="btnSearchCards" style="width:100%;margin-top:8px;font-family:var(--mono)">EXECUTE_SEARCH</button>
              </div>
              <div id="cardResults" style="margin-top:16px;max-height:400px;overflow-y:auto;padding-right:8px"></div>
            </div>
          </div>

          <div class="hub-panel">
            <div class="hub-header">
              <h2>知识覆盖矩阵 (Graph Coverage)</h2>
            </div>
            <div class="hub-body">
              <p class="muted" style="margin:0 0 12px 0;font-size: 14px">因子研究：mechanism × family 交叉矩阵；其他类别：domain × type 分布</p>
              <button class="ghost small" id="btnLoadGraphCoverage" style="width:100%;font-family:var(--mono)">LOAD_MATRIX</button>
              <div id="graphCoverageResult" style="margin-top:16px;max-height:450px;overflow:auto;padding-right:8px"></div>
            </div>
          </div>
        </div>

        <div class="hub-panel" style="margin-top:24px">
          <div class="hub-header">
            <h2>卡片内容查看器 (Card Viewer)</h2>
          </div>
          <div class="hub-body">
            <p class="muted" style="margin:0 0 12px 0;font-size: 14px">输入卡片文件名（点击搜索结果中的名称可自动填入）</p>
            <div class="row" style="align-items:flex-end;margin-bottom:12px">
              <div style="flex:1">
                <label style="font-size: 12px;color:var(--muted);display:block;margin-bottom:4px;font-family:var(--sans)">TARGET_CARD_PATH</label>
                <input id="cardViewName" placeholder="如：Factor - Momentum Base.md" style="width:100%;padding:8px 12px;border:1px solid var(--line);border-radius:8px;font-family:var(--mono);font-size: 14px">
              </div>
              <button class="action small" id="btnReadCard" style="height:35px;margin-bottom:0;font-family:var(--mono)">READ_FILE</button>
            </div>
            
            <div style="background:#f8fafc;border:1px solid var(--line);border-radius:8px;padding:16px">
              <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;border-bottom:1px solid var(--line);padding-bottom:8px">
                <span id="cardContentMeta" class="muted" style="font-family:var(--mono);font-size: 13px"></span>
                <label style="font-size: 13px;font-weight:600;cursor:pointer;display:flex;align-items:center;gap:6px;color:var(--brand-dark)">
                  <input type="checkbox" id="cardRenderToggle" checked> Markdown 渲染
                </label>
              </div>
              <div id="cardContent" class="md-box" style="border:none;padding:0;background:transparent;max-height:600px">请先选择并读取一张卡片。</div>
            </div>
          </div>
        </div>
      </section>

      <!-- ============ B. BRIDGE WORKSPACE ============ -->
      <section id="view-bridge" style="display:none">
        <div class="workflow-header">
          <div class="workflow-step-num">2</div>
          <h2 style="margin:0;font-size: 17px;border:0;padding:0;letter-spacing:0.08em;font-family:var(--mono)">RESEARCH_BRIDGE</h2>
          <span class="muted" style="margin-left:auto;font-size: 13px;font-family:var(--mono)">项目控制面</span>
        </div>

        <div class="bridge-layout">
          <!-- Left: Main Workflow -->
          <div class="bridge-flow">
            
            <!-- Step 1: Idea Lab -->
            <div class="panel panel-inspiration">
              <span class="section-tag">Step 1: Project Hypothesis</span>
              <h2 style="border:0;padding:0;margin-bottom:4px">想法探索器</h2>
              <p class="idea-prompt-hint">先把模糊假设压成当前研究方向，再决定是否值得进入 current case</p>
              
              <div class="inspiration-workspace">
                <div class="inspiration-input-area">
                  <textarea id="exploreIdea" style="min-height:180px;border-color:rgba(2, 132, 199, 0.2)" placeholder="例：我想基于非对称的波动率建立一个反转策略…"></textarea>
                </div>
                
                <div class="explore-mode-panel">
                  <div style="background:#f8fafc;padding:16px;border-radius:12px;border:1px solid var(--line)">
                    <label class="explore-mode-option">
                      <input type="radio" name="exploreMode" value="start" checked>
                      <span class="explore-mode-key">Kickoff</span>
                      <span class="explore-mode-copy">
                        <span class="explore-mode-desc" id="exploreStartDesc">扩展候选机制空间</span>
                        <span class="explore-mode-intent" id="exploreStartIntent">意图：先发散，再决定哪些方向值得继续讨论。</span>
                      </span>
                    </label>
                    
                    <label class="explore-mode-option">
                      <input type="radio" name="exploreMode" value="free">
                      <span class="explore-mode-key">Explore</span>
                      <span class="explore-mode-copy">
                        <span class="explore-mode-desc" id="exploreFreeDesc">结构化细化与可计算性检查</span>
                        <span class="explore-mode-intent" id="exploreFreeIntent">意图：把机制压成候选表达，并提前暴露风险，但暂不做最终选择。</span>
                      </span>
                    </label>
                    
                    <label class="explore-mode-option">
                      <input type="radio" name="exploreMode" value="constrained">
                      <span class="explore-mode-key">Graph</span>
                      <span class="explore-mode-copy">
                        <span class="explore-mode-desc" id="exploreConstrainedDesc">强约束筛选与最终决策</span>
                        <span class="explore-mode-intent" id="exploreConstrainedIntent">意图：在节点、算子、失败案例与拥挤度约束下做选择和淘汰。</span>
                      </span>
                    </label>
                  </div>
                </div>
              </div>
              
              <div class="toolbar" style="margin-top:16px">
                <button class="action" id="btnExploreIdea">启动探索 (Explore)</button>
                <button class="copy-btn" id="btnCopyExplorePrompt" style="display:none">复制 AI 上下文</button>
              </div>
              <div id="exploreResults" style="display:none;margin-top:16px;background:#f8fafc;padding:16px;border-radius:8px;border:1px solid rgba(2, 132, 199, 0.15)">
                <div class="row" style="align-items:flex-start;gap:16px">
                  <div style="flex:1;min-width:0">
                    <strong style="font-size: 15px;color:var(--brand-dark)">关联知识卡片</strong>
                    <div id="exploreCardList" style="margin-top:8px"></div>
                  </div>
                  <div id="exploreRightPane" style="flex:1;min-width:0;display:none">
                    <div id="exploreRightTabs" style="display:flex;gap:4px;margin-bottom:8px">
                      <button id="exploreTabCard" class="action small" style="font-size: 13px;padding:2px 10px;opacity:0.5" onclick="switchExploreRightTab('card')">卡片内容</button>
                      <button id="exploreTabConstraint" class="action small" style="font-size: 13px;padding:2px 10px;opacity:0.5;display:none" onclick="switchExploreRightTab('constraint')">图谱约束</button>
                    </div>
                    <div id="exploreCardPreview" style="display:none">
                      <div id="exploreCardPreviewHeader" style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px">
                        <strong id="exploreCardPreviewTitle" style="font-size: 15px;color:var(--brand-dark)"></strong>
                        <button class="action small" style="font-size: 12px;padding:1px 8px" onclick="$('cardViewName').value=$('exploreCardPreviewTitle').dataset.path||'';$('btnReadCard').click();switchView('view-knowledge')">在知识库中打开</button>
                      </div>
                      <div id="exploreCardPreviewBody" style="font-size: 15px;max-height:600px;overflow:auto;padding:8px 10px;background:#fff;border:1px solid var(--line);border-radius:6px"></div>
                    </div>
                    <div id="exploreConstraintBox" style="display:none">
                      <strong style="font-size: 15px;color:var(--brand-dark)">图谱约束分析</strong>
                      <div id="exploreConstraintReport" style="margin-top:8px;font-size: 15px"></div>
                    </div>
                    <div id="exploreRightPlaceholder" style="color:var(--muted);font-size: 14px;padding:20px 0;text-align:center">← 点击左侧卡片查看内容</div>
                  </div>
                </div>
                <details style="margin-top:10px">
                  <summary style="cursor:pointer;font-weight:bold;color:var(--brand);font-size: 15px">GPT 上下文 Prompt（点击展开）</summary>
                  <pre id="explorePromptBox" style="max-height:320px;overflow:auto;margin-top:6px;font-size: 14px"></pre>
                </details>
              </div>
            </div>

            <!-- Step 2: Integrity Guard -->
            <div class="panel panel-guardrail">
              <span class="section-tag">Step 2: Guardrails</span>
              <h2 style="border:0;padding:0;margin-bottom:8px">候选预检</h2>
              <div class="row">
                <div style="grid-column: span 2"><label>候选名称 (Candidate Name)</label><input id="pfCandidateName" placeholder="Momentum - Skew Adjusted" /></div>
              </div>
              <div id="pfFactorFields">
                <div class="row-3">
                  <div>
                    <label>因子族 (Family)</label>
                    <select id="pfFamily">
                      <option value="">-- 选择 --</option>
                      <option>momentum</option><option>value</option><option>quality</option><option>size</option><option>volatility</option><option>reversal</option>
                    </select>
                  </div>
                  <div>
                    <label>动力机制 (Mechanism)</label>
                    <select id="pfMechanism">
                      <option value="">-- 选择 --</option>
                      <option>behavioral</option><option>risk</option><option>microstructure</option><option>statistical</option>
                    </select>
                  </div>
                  <div>
                    <label>衰减特性 (Decay)</label>
                    <select id="pfDecayClass"><option value="">-- 选择 --</option><option>fast</option><option>medium</option><option>slow</option></select>
                  </div>
                </div>
              </div>
              <label>附加检查卡片路径 (可选)</label>
              <textarea id="pfCheckedCards" style="min-height:50px" placeholder="30_factors/Factor - Momentum Base.md"></textarea>
              <button class="action ghost small" id="btnRunPreflight" style="margin-top:8px">运行图约束扫描</button>
              <div id="preflightResult" style="display:none;margin-top:12px"></div>
            </div>

            <!-- Factor Workshop -->
            <div class="panel" style="border-left:3px solid #8b5cf6">
              <span class="section-tag section-tag-violet">Factor Workshop</span>
              <h2 style="border:0;padding:0;margin-bottom:8px">因子工坊</h2>
              <p class="muted" style="font-size: 13px;margin:0 0 12px">低频高级功能：将 GPT 生成的表达式注册为自定义因子。日常流程可跳过。</p>
              <details class="factor-workshop-details">
                <summary class="factor-workshop-summary">展开 Factor Workshop</summary>
                <div class="grid" style="margin-top:12px">
                  <div>
                    <label>因子方法名 (method name)</label>
                    <input id="cfName" placeholder="asym_vol_reversal" style="font-family:var(--mono)" />
                    <label style="margin-top:8px">因子描述</label>
                    <input id="cfDescription" placeholder="非对称波动率反转因子 — 下行波动高于上行时做多反转" />
                    <label style="margin-top:8px">因子代码 (Python)</label>
                    <textarea id="cfCode" style="min-height:260px;font-family:var(--mono);font-size: 13px;line-height:1.5;tab-size:4;white-space:pre;overflow-wrap:normal;overflow-x:auto" placeholder="def builder(prices, *, window=20, skip_recent=5, min_periods=None, **kwargs):
    import numpy as np
    import pandas as pd

    frame = prices.copy()
    frame['date'] = pd.to_datetime(frame['date'])
    frame = frame.sort_values(['asset', 'date']).reset_index(drop=True)

    ret = frame.groupby('asset', sort=False)['close'].pct_change(fill_method=None)
    # ... your logic ...

    result = frame[['date', 'asset']].copy()
    result['factor'] = 'my_factor'
    result['value'] = ...
    return result"></textarea>
                    <div style="display:flex;gap:8px;margin-top:10px">
                      <button class="action action-violet" id="btnRegisterFactor" style="flex:1">注册因子</button>
                      <button class="ghost small" id="btnLoadFactorTemplate" style="flex:0 0 auto">加载模板</button>
                    </div>
                    <pre id="cfResponseBox" style="margin-top:8px;font-size: 12px;max-height:60px;overflow:auto" class="muted"></pre>
                  </div>
                  <div>
                    <label>已注册因子方法</label>
                    <div id="cfFactorList" style="font-size: 14px;font-family:var(--mono)">加载中...</div>
                  </div>
                </div>
              </details>
            </div>

            <!-- Step 3: Current Case -->
            <div class="panel panel-ai-workspace">
              <span class="section-tag">Step 3: Current Case</span>
              <h2 style="border:0;padding:0;margin-bottom:12px">当前实验合同</h2>
              <div class="grid">
                <div>
                  <label>Current Case Name</label>
                  <input id="caseName" placeholder="mom_v1" />
                  <div id="caseDynamicFields"></div>
                  <button class="action action-success" id="btnCreateCase" style="width:100%;margin-top:12px">刷新 Current Case</button>
                  <pre id="bridgeResponseBox" style="margin-top:12px;font-size: 12px;max-height:80px;overflow:auto" class="muted"></pre>
                </div>
                <div>
                  <label>Current Case Preview</label>
                  <div id="currentCasePreviewBox" class="md-box report-doc-box" style="background:#f8fafc">等待 current case…</div>
                </div>
              </div>
            </div>

            <!-- Step 4: Run Validation -->
            <div class="panel panel-ai-workspace">
              <span class="section-tag">Step 4: Run Validation</span>
              <h2 style="border:0;padding:0;margin-bottom:12px">运行与结果</h2>
              <div class="grid">
                <div class="hub-box">
                  <label>Case 列表 (Current)</label>
                  <p class="muted" style="margin:0 0 8px 0;font-size: 13px">显示当前活跃的 case 合同，点击“选择”可直接填入运行目标。</p>
                  <div id="caseTable"></div>
                  <button class="ghost small" id="btnRefreshCases" style="margin-top:12px;font-family:var(--mono)">REFRESH_CASES</button>
                </div>
                <div class="hub-box">
                  <label>CASE_NAME (Target)</label>
                  <input id="runCaseName" placeholder="选定或输入 Case">
                  <label>EVALUATION_PROFILE (Engine)</label>
                  <select id="runProfile"></select>
                  <label>OUTPUT_DIR_OVERRIDE (Optional)</label>
                  <input id="runOutputDir" placeholder="默认路径">
                  <button class="action" id="btnStartRun" style="width:100%;margin-top:12px;background:var(--brand-dark);font-family:var(--mono)">EXECUTE_RUN</button>
                </div>
              </div>
              <div class="hub-panel" style="margin-top:16px">
                <div class="hub-header" style="border-bottom-color:var(--ok)">
                  <h2>Run Queue & Artifacts</h2>
                </div>
                <div class="hub-body">
                  <div class="toolbar" style="margin-bottom:12px">
                    <button class="ghost small" id="btnRefreshRuns" style="font-family:var(--mono)">REFRESH_QUEUE</button>
                    <button class="ghost small" id="btnAutoRefresh" style="font-family:var(--mono)">AUTO_REFRESH: 5s</button>
                  </div>
                  <p class="muted" style="margin:0 0 16px 0;font-size: 14px">状态：queued → running → succeeded / failed。成功后点“刷新结果”更新 Latest Result 与 Decision Log。</p>
                  <div id="runTable" style="margin-bottom:16px;overflow-x:auto"></div>
                  <div class="hub-section-title">项目级诊断</div>
                  <div class="toolbar" style="margin-bottom:12px">
                    <button class="ghost small" id="btnRefreshProjectDiagnostics" style="font-family:var(--mono)">REFRESH_PROJECT_DIAGNOSTICS</button>
                  </div>
                  <div id="projectDiagnostics" class="project-diagnostics-box" style="margin-bottom:16px">
                    <span class="muted">等待加载项目级相关性诊断…</span>
                  </div>
                  <div class="hub-section-title">Artifact Viewer</div>
                  <div id="artifactViewer" class="md-box" style="background:#f8fafc;color:var(--ink);padding:16px;border-radius:8px;font-size: 14px;min-height:100px;border:1px solid var(--line)">
                    <span class="muted">点击运行记录中的 artifact 链接（如 metrics、summary）在此查看内容。</span>
                  </div>
                  <div id="validationResponseBox" style="margin-top:12px;font-family:var(--mono);font-size: 13px;color:var(--brand-dark);"></div>
                </div>
              </div>
            </div>

            <!-- Step 5: Interpretation & State -->
            <div class="panel panel-ai-workspace">
              <span class="section-tag">Step 5: Interpretation & State</span>
              <h2 style="border:0;padding:0;margin-bottom:12px">解释结果并更新状态</h2>
              <div>
                <label>Current Case (Document)</label>
                <p class="muted" style="font-size: 13px;margin:0 0 8px">这是已落地的实验合同正文，用于核对“刚才运行的到底是什么”。</p>
                <div id="currentCaseHub" class="md-box report-doc-box">等待 current case…</div>
              </div>
              <div class="workspace-dual">
                <div>
                  <label>Latest Result</label>
                  <p class="muted" style="font-size: 13px;margin:0 0 8px">最近一次已汇总的实验结果摘要，用于判断是否继续该方向。</p>
                  <div id="latestResultHub" class="md-box report-doc-box">等待 latest result…</div>
                </div>
                <div>
                  <label>Decision Log</label>
                  <p class="muted" style="font-size: 13px;margin:0 0 8px">项目级决策记录，说明为什么继续 / 暂停 / 放弃，避免重复路径。</p>
                  <div id="decisionLogHub" class="md-box report-doc-box">等待 decision log…</div>
                </div>
              </div>
              <div class="hub-box" style="margin-top:14px">
                <label>RESEARCH_HYPOTHESIS</label>
                <textarea id="patchHypothesis" style="min-height:50px"></textarea>
                <label>CURRENT_FOCUS</label>
                <input id="patchFocus">
                <label>NEXT_ACTION</label>
                <input id="patchAction" style="margin-bottom:12px">
                <button class="ghost small" id="btnPatchProject" style="width:100%;background:var(--brand);color:#fff;border:none;">SYNC_STATE</button>
              </div>
            </div>

          </div>

          <!-- Right: Project Hub -->
          <div class="project-hub-sticky">
            <div class="project-hub-stack">
            <div class="hub-panel">
              <div class="hub-header">
                <h2>Project Snapshot</h2>
                <div class="pulse" title="System Online"></div>
              </div>
              
              <div class="hub-body">
                <p class="hub-section-explainer" style="margin:0 0 14px 0;font-size:11.5px">
                  这里回答两个问题：这个项目是什么，以及它现在处于哪一步。先看项目标识，再看研究状态，不要把它当成结果面板。
                </p>
                <div class="hub-section-title">项目标识 (Identity)</div>
                <p class="hub-section-explainer">说明这是哪个项目、在哪个研究轨道上、默认用什么评估设定。</p>
                <div id="projectMetaView" style="font-family: var(--mono); font-size: 14px;"></div>

                <div class="hub-section-title">研究状态 (Research State)</div>
                <p class="hub-section-explainer">说明当前母命题、这一轮 focus、下一步动作、当前实验合同，以及最近一次项目级结论。</p>
                <div id="projectStateView" style="font-family: var(--mono); font-size: 14px;"></div>
              </div>
            </div>

            <div class="hub-panel">
              <div class="hub-header">
                <h2>Workspace Tips</h2>
              </div>
              <div class="hub-body">
                <p class="hub-section-explainer" style="margin:0 0 14px 0;font-size:11.5px">
                  主流程：Idea → Guardrails → Current Case → Run → Refresh Result → Sync State。右侧不再承载正文文档，避免来回找信息。
                </p>
                <button class="ghost small" id="btnOpenCreateProject2" style="width:100%">NEW_PROJECT</button>
              </div>
            </div>
            </div>
          </div>
        </div>
      </section>

      <!-- Validation panels merged into Project Workspace -->

      <!-- ============ D. WRITEBACK REVIEW ============ -->
      <section id="view-writeback" style="display:none">
        <div class="workflow-header">
          <div class="workflow-step-num">4</div>
          <h2 style="margin:0;font-size: 17px;border:0;padding:0;letter-spacing:0.08em;font-family:var(--mono)">WRITEBACK_REVIEW</h2>
          <span class="muted" style="margin-left:auto;font-size: 13px;font-family:var(--mono)">可选正式导出</span>
        </div>

        <div class="grid">
          <div class="hub-panel">
            <div class="hub-header">
              <h2>导出草案队列 (Export Drafts)</h2>
            </div>
            <div class="hub-body">
              <p class="muted" style="margin:0 0 12px 0;font-size: 14px">这个页面不是日常主路径。只有你需要把结论正式写回知识库时，才在这里处理导出草案。</p>
              <div id="draftTable"></div>
              <button class="ghost small" id="btnRefreshDrafts" style="margin-top:12px;font-family:var(--mono)">REFRESH_EXPORTS</button>
            </div>
          </div>
          
          <div class="hub-panel">
            <div class="hub-header" style="border-bottom-color:var(--brand)">
              <h2>正式导出编辑器 (Export Editor)</h2>
            </div>
            <div class="hub-body">
              <p class="muted" style="margin:0 0 16px 0;font-size: 14px">这里只处理正式知识写回。日常研究循环只看 project / current case / latest result / decision log。</p>
              <div class="hub-box">
                <label>EXPORT_DRAFT</label>
                <input id="draftName" placeholder="draft_mom_5d_r01.md">
              </div>
              <div class="row">
                <div class="hub-box">
                  <label>EXPORT_STATUS</label>
                  <select id="draftStatus">
                    <option value="approved">approved</option>
                    <option value="rejected">rejected</option>
                  </select>
                </div>
                <div class="hub-box">
                  <label>REVIEWED_BY</label>
                  <input id="draftReviewer" value="yukun">
                </div>
              </div>
              <div class="hub-box">
                <label>REVIEWED_AT (填 'now' 自动转为当前时间)</label>
                <input id="draftReviewedAt" value="now">
                <label>FORMAL_VERDICT</label>
                <input id="draftVerdict" placeholder="该因子 IC 均值 0.04，建议进入候选池">
              </div>
              <div class="toolbar" style="margin-top:12px">
                <button class="action" id="btnPatchDraft" style="background:var(--brand-dark);font-family:var(--mono)">SAVE_EXPORT_REVIEW</button>
                <button class="action" id="btnApplyDraft" style="background:var(--ok);font-family:var(--mono)">EXECUTE_FORMAL_WRITEBACK</button>
              </div>
            </div>
          </div>
        </div>

        <div class="hub-panel" style="margin-top:24px">
          <div class="hub-header">
            <h2>导出草案预览 (Preview)</h2>
          </div>
          <div class="hub-body">
            <button class="ghost small" id="btnPreviewDraft" style="font-family:var(--mono);margin-bottom:12px">LOAD_EXPORT_PREVIEW</button>
            <pre id="draftPreviewContent" style="background:#f1f5f9;color:var(--ink);padding:16px;border-radius:8px;font-family:var(--mono);font-size: 14px;margin:0;max-height:500px;overflow:auto;border:1px solid var(--line)">选择导出草案后点“加载预览”查看全文。</pre>
            <div id="writebackResponseBox" style="margin-top:12px;font-family:var(--mono);font-size: 13px;color:var(--brand-dark);"></div>
          </div>
        </div>
      </section>

      <div id="createProjectModal" class="project-modal-backdrop" style="display:none">
        <div class="project-modal">
          <div class="hub-header">
            <h2>New Project (Init)</h2>
            <button class="ghost small" id="btnCloseCreateProject">CLOSE</button>
          </div>
          <div class="hub-body">
            <p class="hub-section-explainer" style="margin:0 0 12px 0">仅在创建全新研究主题时使用。已有项目请在工作台更新 current case 与状态同步。</p>
            <div class="hub-box">
              <label>PROJECT_SLUG</label>
              <input id="createSlug" placeholder="asym-vol-reversal">
              <label>PROJECT_TITLE</label>
              <input id="createTitle" placeholder="非对称波动率反转因子验证">
              <label>CATEGORY_PROFILE</label>
              <select id="createCategory" style="margin-bottom:12px"><option value="factor_recipe">因子配方研究</option></select>
              <label>OWNER</label>
              <input id="createOwner" value="yukun">
              <label>MARKET / FREQUENCY</label>
              <div style="display:flex;gap:6px">
                <input id="createMarket" value="ashare" style="flex:1">
                <input id="createFrequency" value="daily" style="flex:1">
              </div>
              <label>CHATGPT_PROJECT_NAME</label>
              <input id="createChatgptName" placeholder="Asym Vol Reversal">
              <label>ORIGIN_CARDS（每行一个路径）</label>
              <textarea id="createOriginCards" style="min-height:48px" placeholder="30_factors/Factor - Momentum Base.md"></textarea>
              <button class="ghost small" id="btnCreateProject" style="width:100%;margin-top:8px">INIT_WORKSPACE</button>
            </div>
          </div>
        </div>
      </div>

    </main>
  </div>

  <script>
    // ========== State ==========
    const state = {
      view: "dashboard",
      projects: [],
      selectedProject: "",
      projectDetail: null,
      autoRefreshTimer: null,
      autoRefreshMode: "off",
      artifactRawText: "",
      categories: [],          // [{key, display_name_zh, form_fields, ...}]
      currentCategoryKey: "factor_recipe",
    };

    const $ = (id) => document.getElementById(id);

    // ========== API helper ==========
    async function api(path, method = "GET", body = null) {
      const controller = new AbortController();
      const tId = setTimeout(() => controller.abort(), 15000);
      const opts = { method, headers: {}, signal: controller.signal };
      if (body !== null) {
        opts.headers["Content-Type"] = "application/json";
        opts.body = JSON.stringify(body);
      }
      try {
        const res = await fetch(path, opts);
        clearTimeout(tId);
        const data = await res.json().catch(() => ({}));
        if (!res.ok) {
          const message = data.error || `HTTP ${res.status} ${res.statusText}`;
          throw new Error(message);
        }
        return data;
      } catch(e) {
        clearTimeout(tId);
        if (e.name === "AbortError") throw new Error("请求超时（服务器无响应，请检查服务是否正常启动）");
        throw e;
      }
    }

    function showResponse(boxId, payload) {
      const el = $(boxId);
      if (el) el.textContent = JSON.stringify(payload, null, 2);
    }

    function openCreateProjectModal() {
      const modal = $("createProjectModal");
      if (!modal) return;
      modal.style.display = "flex";
      modal.classList.add("active");
    }

    function closeCreateProjectModal() {
      const modal = $("createProjectModal");
      if (!modal) return;
      modal.classList.remove("active");
      modal.style.display = "none";
    }

    // ========== Navigation ==========
    const VIEWS = ["dashboard", "knowledge", "bridge", "writeback"];

    function switchView(view) {
      state.view = view;
      for (const button of document.querySelectorAll(".nav button")) {
        button.classList.toggle("active", button.dataset.view === view);
      }
      for (const v of VIEWS) {
        $("view-" + v).style.display = v === view ? "block" : "none";
      }
      // Load view-specific data
      if (view === "dashboard") loadDashboard();
      if (view === "knowledge") { loadVaultStats(); loadInbox(); }
      if (view === "bridge") {
        loadProjectDetail();
        loadCases();
        loadRuns();
      }
      if (view === "writeback") loadDrafts();
    }

    // ========== Dashboard ==========
    async function loadDashboard() {
      try {
        const data = await api("/api/dashboard");
        const cards = [
          ["项目数", data.project_count],
          ["知识卡片", data.vault_card_count],
          ["待处理文件", data.vault_inbox_count],
          ["运行中", data.run_status_counts.running || 0],
          ["失败", data.run_status_counts.failed || 0],
        ];
        $("dashboardCards").innerHTML = cards.map(([t, v]) => `
          <div class="card"><h3>${t}</h3><div class="value">${v}</div></div>
        `).join("");
        $("recentRuns").innerHTML = (data.recent_runs || []).map((r) => `
          <div style="margin:6px 0;padding:6px;border-bottom:1px solid var(--line)">
            <strong>${r.project_slug}/${r.case_name}</strong>
            <span class="status ${r.status}" style="margin-left:8px">${r.status}</span>
            <div class="muted">${r.submitted_at_utc}</div>
          </div>
        `).join("") || "<div class='muted'>暂无运行记录。</div>";
        $("nextActions").innerHTML = (data.next_actions || []).map((a) => `
          <div style="margin:6px 0;padding:6px;border-bottom:1px solid var(--line)">
            <strong>${a.project_slug}</strong>
            <div>${a.next_action}</div>
          </div>
        `).join("") || "<div class='muted'>暂无待办事项。</div>";
      } catch(e) { console.error(e); }
    }

    // ========== Knowledge Ops ==========
    async function loadVaultStats() {
      try {
        const data = await api("/api/vault/stats");
        let html = `<div><strong>卡片总数：</strong> ${data.total_cards}</div>`;
        html += `<div><strong>待处理文件：</strong> ${data.inbox_count}</div>`;
        if (data.by_type && Object.keys(data.by_type).length) {
          html += `<div style="margin-top:8px"><strong>按类型：</strong></div><table>`;
          for (const [k, v] of Object.entries(data.by_type)) {
            html += `<tr><td>${k}</td><td>${v}</td></tr>`;
          }
          html += `</table>`;
        }
        if (data.by_lifecycle && Object.keys(data.by_lifecycle).length) {
          html += `<div style="margin-top:8px"><strong>按生命周期：</strong></div><table>`;
          for (const [k, v] of Object.entries(data.by_lifecycle)) {
            html += `<tr><td>${k}</td><td>${v}</td></tr>`;
          }
          html += `</table>`;
        }
        $("vaultStats").innerHTML = html;
      } catch(e) { $("vaultStats").innerHTML = `<div class="muted">Error: ${e.message}</div>`; }
    }

    async function loadInbox() {
      try {
        const data = await api("/api/vault/inbox");
        if (!data.items.length) {
          $("inboxList").innerHTML = "<div class='muted'>待处理文件夹为空（00_inbox / _sources）。</div>";
          return;
        }
        $("inboxList").innerHTML = `<table>
          <thead><tr><th>文件名</th><th>目录</th><th>大小</th><th>修改时间</th></tr></thead>
          <tbody>${data.items.map((f) => `
            <tr><td>${f.name}</td><td>${f.directory}</td><td>${f.size_bytes}B</td><td class="muted">${f.modified}</td></tr>
          `).join("")}</tbody></table>`;
      } catch(e) { $("inboxList").innerHTML = `<div class="muted">Error: ${e.message}</div>`; }
    }

    // ========== Categories ==========
    async function loadCategories() {
      try {
        const data = await api("/api/categories");
        state.categories = data.categories || [];
        // Populate category select in project creation form
        const sel = $("createCategory");
        sel.innerHTML = state.categories.map(c =>
          `<option value="${c.key}">${c.display_name_zh} (${c.key})</option>`
        ).join("");
      } catch(e) { console.error("loadCategories:", e); }
    }

    function getCategoryProfile(key) {
      return state.categories.find(c => c.key === key) || state.categories[0] || null;
    }

    function updateCategoryUI() {
      // Determine current project category
      const p = state.projectDetail && state.projectDetail.project;
      const catKey = (p && p.category) || "factor_recipe";
      state.currentCategoryKey = catKey;
      const isFactorRecipe = (catKey === "factor_recipe");

      // Preflight: show/hide factor-specific fields
      const pfFactorFields = $("pfFactorFields");
      if (pfFactorFields) pfFactorFields.style.display = isFactorRecipe ? "" : "none";
      const pfDesc = $("preflightDescription");
      if (pfDesc) {
        pfDesc.textContent = isFactorRecipe
          ? "填写候选因子元属性，运行 5 项图约束检查（依赖完整性、新颖性、PIT、机制拥挤度、容量衰减）"
          : "填写候选研究名称，运行图约束检查（依赖完整性 + 新颖性）";
      }

      // Explore mode descriptions
      const profile = getCategoryProfile(catKey);
      const displayName = profile ? profile.display_name_zh : catKey;
      const startDesc = $("exploreStartDesc");
      const startIntent = $("exploreStartIntent");
      const freeDesc = $("exploreFreeDesc");
      const freeIntent = $("exploreFreeIntent");
      const constrDesc = $("exploreConstrainedDesc");
      const constrIntent = $("exploreConstrainedIntent");
      if (startDesc) {
        startDesc.textContent = isFactorRecipe
          ? "扩展候选机制空间"
          : `${displayName}：先扩展问题空间`;
      }
      if (startIntent) {
        startIntent.textContent = isFactorRecipe
          ? "意图：先发散，再决定哪些方向值得继续讨论。"
          : `意图：先打开 ${displayName} 的候选空间，不提前收敛。`;
      }
      if (freeDesc) {
        freeDesc.textContent = isFactorRecipe
          ? "结构化细化与可计算性检查"
          : `${displayName}：结构化细化`;
      }
      if (freeIntent) {
        freeIntent.textContent = isFactorRecipe
          ? "意图：把机制压成候选表达，并提前暴露风险，但暂不做最终选择。"
          : `意图：细化 ${displayName} 的候选对象，同时保留讨论空间。`;
      }
      if (constrDesc) {
        constrDesc.textContent = isFactorRecipe
          ? "强约束筛选与最终决策"
          : `${displayName}：强约束筛选`;
      }
      if (constrIntent) {
        constrIntent.textContent = isFactorRecipe
          ? "意图：在节点、算子、失败案例与拥挤度约束下做选择和淘汰。"
          : `意图：在知识与约束边界内收敛到更强候选。`;
      }

      // Case creation form: render dynamic fields based on category profile
      renderCaseForm(catKey);
    }

    function renderCaseForm(categoryKey) {
      const container = $("caseDynamicFields");
      if (!container) return;
      const profile = getCategoryProfile(categoryKey);
      if (!profile || !profile.form_fields || !profile.form_fields.length) {
        container.innerHTML = "";
        return;
      }
      let html = '<div class="row">';
      for (const field of profile.form_fields) {
        html += '<div>';
        html += `<label>${escHtml(field.label || field.name)}</label>`;
        if (field.type === "select" && field.options) {
          html += `<select id="caseField_${field.name}">`;
          for (const opt of field.options) {
            const selected = (opt === field.default) ? " selected" : "";
            html += `<option value="${escAttr(opt)}"${selected}>${escHtml(opt)}</option>`;
          }
          html += `</select>`;
        } else {
          const val = field.default || "";
          const ph = field.placeholder || "";
          html += `<input id="caseField_${field.name}" value="${escAttr(val)}" placeholder="${escAttr(ph)}" />`;
        }
        html += '</div>';
      }
      html += '</div>';
      container.innerHTML = html;
    }

    function collectCaseDynamicFields() {
      const profile = getCategoryProfile(state.currentCategoryKey);
      if (!profile || !profile.form_fields) return {};
      const result = {};
      for (const field of profile.form_fields) {
        const el = $("caseField_" + field.name);
        if (el) result[field.name] = el.value || null;
      }
      return result;
    }

    // ========== Factor Workshop ==========
    async function loadCustomFactors() {
      try {
        const data = await api("/api/custom-factors");
        const list = $("cfFactorList");
        if (!list) return;
        const factors = data.factors || [];
        if (!factors.length) { list.innerHTML = "<span class='muted'>无已注册因子</span>"; return; }
        let html = '<table style="width:100%;border-collapse:collapse;font-size: 14px">';
        html += '<tr style="border-bottom:1px solid #e2e8f0"><th style="text-align:left;padding:4px">名称</th><th style="text-align:left;padding:4px">类型</th><th style="padding:4px"></th></tr>';
        for (const f of factors) {
          const badge = f.is_custom
            ? '<span style="background:#8b5cf6;color:#fff;padding:1px 6px;border-radius:3px;font-size: 12px">custom</span>'
            : '<span style="background:#94a3b8;color:#fff;padding:1px 6px;border-radius:3px;font-size: 12px">built-in</span>';
          const actions = f.is_custom
            ? `<button class="ghost small cf-view-btn" data-name="${escAttr(f.name)}" style="font-size: 12px;padding:1px 4px">查看</button> <button class="ghost small cf-del-btn" data-name="${escAttr(f.name)}" style="font-size: 12px;padding:1px 4px;color:var(--fail)">删除</button>`
            : '';
          html += `<tr style="border-bottom:1px solid #f1f5f9"><td style="padding:4px">${escHtml(f.name)}</td><td style="padding:4px">${badge}</td><td style="padding:4px;text-align:right">${actions}</td></tr>`;
        }
        html += '</table>';
        list.innerHTML = html;
        // Bind view/delete buttons
        for (const btn of list.querySelectorAll(".cf-view-btn")) {
          btn.addEventListener("click", async () => {
            try {
              const data = await api(`/api/custom-factors/${enc(btn.dataset.name)}`);
              $("cfName").value = data.name || "";
              $("cfDescription").value = data.description || "";
              $("cfCode").value = data.code || "";
            } catch(e) { showResponse("cfResponseBox", {error: String(e)}); }
          });
        }
        for (const btn of list.querySelectorAll(".cf-del-btn")) {
          btn.addEventListener("click", async () => {
            if (!confirm(`确认删除自定义因子 ${btn.dataset.name}？`)) return;
            try {
              await api(`/api/custom-factors/${enc(btn.dataset.name)}`, "DELETE");
              await loadCustomFactors();
            } catch(e) { showResponse("cfResponseBox", {error: String(e)}); }
          });
        }
      } catch(e) { console.error(e); }
    }

    const FACTOR_TEMPLATE = `def builder(prices, *, window=20, skip_recent=5, min_periods=None, **kwargs):
    # Custom factor builder.
    # Args:
    #   prices: DataFrame with columns [date, asset, close, high, low, volume, amount]
    #   window: lookback window
    #   skip_recent: skip recent N days
    # Returns:
    #   DataFrame with columns [date, asset, factor, value]
    import numpy as np
    import pandas as pd

    frame = prices.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values(["asset", "date"]).reset_index(drop=True)

    # Simple returns
    ret = frame.groupby("asset", sort=False)["close"].pct_change(fill_method=None)

    # --- Replace with your logic ---
    # Example: negative downside volatility (short low-vol stocks)
    neg_ret = ret.where(ret < 0, 0.0)
    pos_ret = ret.where(ret > 0, 0.0)
    down_vol = neg_ret.pow(2).groupby(frame["asset"], sort=False).rolling(window, min_periods=max(3, window // 2)).mean().reset_index(level=0, drop=True).pipe(np.sqrt)
    up_vol = pos_ret.pow(2).groupby(frame["asset"], sort=False).rolling(window, min_periods=max(3, window // 2)).mean().reset_index(level=0, drop=True).pipe(np.sqrt)
    asym = down_vol / up_vol.replace(0, np.nan)

    result = frame[["date", "asset"]].copy()
    result["factor"] = "custom_factor"
    result["value"] = -asym  # negative = prefer low asymmetry
    return result
`;

    // ========== Bridge Workspace ==========
    async function loadProjects() {
      try {
        const data = await api("/api/projects");
        state.projects = data.projects || [];
        const select = $("projectSelect");
        select.innerHTML = "<option value=''>-- 请选择项目 --</option>" + state.projects.map((p) =>
          `<option value="${p.slug}">${p.slug} | ${p.title_zh}</option>`
        ).join("");
        if (!state.selectedProject && state.projects.length > 0) {
          state.selectedProject = state.projects[0].slug;
        }
        if (state.selectedProject) {
          select.value = state.selectedProject;
          await loadProjectDetail();
        }
      } catch(e) { console.error(e); }
    }

    async function loadProjectDetail() {
      if (!state.selectedProject) {
        $("projectMetaView").innerHTML = "<div class='muted'>请在左侧选择一个项目。</div>";
        $("projectStateView").innerHTML = "<div class='muted'>选中项目后显示当前研究状态。</div>";
        $("currentCaseHub").textContent = "等待 current case…";
        $("latestResultHub").textContent = "等待 latest result…";
        $("decisionLogHub").textContent = "等待 decision log…";
        if ($("projectDiagnostics")) {
          $("projectDiagnostics").innerHTML = "<div class='muted'>请选择项目后查看项目级诊断。</div>";
        }
        return;
      }
      try {
        const data = await api(`/api/projects/${enc(state.selectedProject)}`);
        state.projectDetail = data;
        const p = data.project;
        let metaHtml = `<table>`;
        metaHtml += `<tr><td><strong>slug</strong></td><td>${p.slug}</td></tr>`;
        metaHtml += `<tr><td><strong>title</strong></td><td>${p.title_zh}</td></tr>`;
        metaHtml += `<tr><td><strong>category</strong></td><td>${p.category}</td></tr>`;
        metaHtml += `<tr><td><strong>chatgpt_project_name</strong></td><td>${p.chatgpt_project_name || "-"}</td></tr>`;
        metaHtml += `<tr><td><strong>market</strong></td><td>${p.market} / ${p.frequency}</td></tr>`;
        metaHtml += `<tr><td><strong>eval_profile</strong></td><td>${p.alpha_lab_defaults.evaluation_profile}</td></tr>`;
        metaHtml += `<tr><td><strong>origin_cards</strong></td><td>${(p.origin_cards||[]).join(", ") || "-"}</td></tr>`;
        metaHtml += `</table>`;
        let stateHtml = `<table>`;
        stateHtml += `<tr><td><strong>lifecycle</strong></td><td><span class="status ${p.status.lifecycle}">${p.status.lifecycle}</span></td></tr>`;
        stateHtml += `<tr><td><strong>hypothesis</strong></td><td>${p.status.current_hypothesis || "-"}</td></tr>`;
        stateHtml += `<tr><td><strong>focus</strong></td><td>${p.status.current_focus || "-"}</td></tr>`;
        stateHtml += `<tr><td><strong>next_action</strong></td><td>${p.status.next_action || "-"}</td></tr>`;
        stateHtml += `<tr><td><strong>current_case</strong></td><td>${p.status.current_case || "-"}</td></tr>`;
        stateHtml += `<tr><td><strong>last_verdict</strong></td><td>${p.status.last_verdict || "-"}</td></tr>`;
        stateHtml += `</table>`;
        $("projectMetaView").innerHTML = metaHtml;
        $("projectStateView").innerHTML = stateHtml;
        renderMarkdownHub("currentCaseHub", data.documents.current_case || "暂无 current case");
        renderMarkdownHub("latestResultHub", data.documents.latest_run || "暂无 latest result");
        renderMarkdownHub("decisionLogHub", data.documents.decision_log || "暂无 decision log");
        renderMarkdownHub("currentCasePreviewBox", data.documents.current_case || "暂无 current case");
        // Populate status patch fields
        $("patchHypothesis").value = p.status.current_hypothesis || "";
        $("patchFocus").value = p.status.current_focus || "";
        $("patchAction").value = p.status.next_action || "";
        // Render rounds table
        // Update category-dependent UI (case form, preflight, etc.)
        updateCategoryUI();
      } catch(e) {
        $("projectMetaView").innerHTML = `<div class="muted">Error: ${e.message}</div>`;
        $("projectStateView").innerHTML = `<div class="muted">项目状态加载失败。</div>`;
      }
    }

    function copyText(elementId, btn) {
      const text = $(elementId).textContent;
      navigator.clipboard.writeText(text).then(() => {
        const orig = btn.textContent;
        btn.textContent = "Copied!";
        setTimeout(() => btn.textContent = orig, 1500);
      });
    }

    // ========== Validation Console ==========
    async function loadCases() {
      if (!state.selectedProject) return;
      try {
        const data = await api(`/api/projects/${enc(state.selectedProject)}/cases`);
        const cases = data.cases || [];
        if (!cases.length) {
          $("caseTable").innerHTML = "<div class='muted'>暂无 current case，请先在研究工作台刷新 current case。</div>";
          return;
        }
        $("caseTable").innerHTML = `<div class="case-table-wrap"><table class="case-table">
          <thead><tr><th>case_name</th><th>current</th><th>spec</th><th>handoff</th><th></th></tr></thead>
          <tbody>${cases.map(c => `
            <tr>
              <td class="case-name-cell"><code>${c.case_name}</code></td>
              <td class="case-flag-cell">${c.is_current ? "<span class='status succeeded'>current</span>" : "-"}</td>
              <td class="case-flag-cell">${c.spec_exists ? "yes" : "no"}</td>
              <td class="case-flag-cell">${c.handoff_exists ? "yes" : "no"}</td>
              <td class="case-action-cell"><button class="ghost small" data-action="selectCase" data-case-name="${escAttr(c.case_name)}">选择</button></td>
            </tr>
          `).join("")}</tbody></table></div>`;
      } catch(e) { $("caseTable").innerHTML = `<div class="muted">${e.message}</div>`; }
    }

    function selectCase(name) {
      $("runCaseName").value = name;
    }

    async function loadEvaluationProfiles() {
      try {
        const data = await api("/api/evaluation-profiles");
        const sel = $("runProfile");
        sel.innerHTML = (data.profiles || []).map(p =>
          `<option value="${p}" ${p === data.default_profile ? "selected" : ""}>${p}</option>`
        ).join("");
      } catch(e) { console.error(e); }
    }

    async function loadRuns() {
      if (!state.selectedProject) return;
      try {
        const data = await api(`/api/projects/${enc(state.selectedProject)}/runs`);
        const runs = data.runs || [];
        renderRunTable(runs);
        await loadProjectDiagnostics();
        const hasActiveRuns = runs.some(r => r.status === "queued" || r.status === "running");
        ensureRunAutoRefresh(hasActiveRuns);
      } catch(e) {
        $("runTable").innerHTML = `<div class="muted">${e.message}</div>`;
        $("projectDiagnostics").innerHTML = `<div class="muted">诊断加载失败：${escHtml(e.message || "unknown error")}</div>`;
      }
    }

    async function loadProjectDiagnostics() {
      if (!state.selectedProject) return;
      const container = $("projectDiagnostics");
      if (!container) return;
      try {
        const data = await api(`/api/projects/${enc(state.selectedProject)}/diagnostics/factor-correlation`);
        container.innerHTML = renderProjectDiagnostics(data);
      } catch (e) {
        container.innerHTML = `<div class="muted">项目级诊断加载失败：${escHtml(e.message || "unknown error")}</div>`;
      }
    }

    function renderProjectDiagnostics(data) {
      const dsrHtml = renderProjectDsrSection(data);
      if (!data || data.ok === false) {
        const msg = data && data.message ? data.message : "暂无可用诊断。";
        return `<div class="muted">${escHtml(msg)}</div>${dsrHtml}`;
      }
      const labels = Array.isArray(data.labels) ? data.labels : [];
      const matrix = Array.isArray(data.matrix) ? data.matrix : [];
      if (!labels.length || !matrix.length) {
        return `<div class="muted">相关性矩阵为空。</div>${dsrHtml}`;
      }
      const heatmapHtml = renderCorrelationHeatmap(labels, matrix);
      const pairs = Array.isArray(data.redundancy_pairs) ? data.redundancy_pairs : [];
      const pairHtml = pairs.length
        ? `<div class="diag-redundancy-list">${pairs.slice(0, 8).map((row) => {
            const level = String(row.warning || "medium");
            const left = String(row.factor_a || "");
            const right = String(row.factor_b || "");
            const corr = Number(row.correlation);
            const overlap = Number(row.overlap_dates);
            return `<div class="diag-redundancy-item ${escAttr(level)}">${escHtml(left)} vs ${escHtml(right)} · corr=${Number.isFinite(corr) ? corr.toFixed(3) : "-"} · overlap=${Number.isFinite(overlap) ? String(overlap) : "-"}</div>`;
          }).join("")}</div>`
        : "<div class='muted'>暂无超过阈值的冗余因子对。</div>";
      const threshold = Number(data.threshold);
      const minOverlap = Number(data.min_overlap);
      const nRunsUsed = Number(data.n_runs_used);
      return `<div class="diag-meta">阈值 |corr| ≥ ${Number.isFinite(threshold) ? threshold.toFixed(2) : "0.70"}，最小重叠日期 ${Number.isFinite(minOverlap) ? minOverlap : 5}，参与因子 ${Number.isFinite(nRunsUsed) ? nRunsUsed : labels.length} 个。</div>
        ${heatmapHtml}
        <div class="diag-meta" style="margin-top:2px">冗余因子警告</div>
        ${pairHtml}
        ${dsrHtml}`;
    }

    function renderProjectDsrSection(data) {
      const summary = data && typeof data === "object" && data.dsr_summary && typeof data.dsr_summary === "object"
        ? data.dsr_summary
        : {};
      const rows = data && Array.isArray(data.dsr_by_factor) ? data.dsr_by_factor : [];
      const nRunsTotal = Number(summary.n_runs_total);
      const nWith = Number(summary.n_with_dsr);
      const median = Number(summary.median_dsr_pvalue);
      const robust = Number(summary.robust_count);
      const highRisk = Number(summary.high_risk_count);
      const meta = `覆盖 ${Number.isFinite(nWith) ? nWith : rows.length}/${Number.isFinite(nRunsTotal) ? nRunsTotal : "-"}，中位数 ${Number.isFinite(median) ? median.toFixed(3) : "-"}，稳健 ${Number.isFinite(robust) ? robust : 0}，高风险 ${Number.isFinite(highRisk) ? highRisk : 0}`;

      let listHtml = "<div class='muted'>暂无 DSR p-value（需单次运行产出该标量）。</div>";
      if (rows.length) {
        listHtml = `<div class="diag-dsr-list">${rows.slice(0, 10).map((row) => {
          const factor = String(row.factor_name || row.case_name || "-");
          const dsr = Number(row.dsr_pvalue);
          const level = String(row.risk_level || "watch");
          const levelClass = level === "robust" ? "good" : (level === "high_risk" ? "high" : "watch");
          const levelText = level === "robust" ? "稳健" : (level === "high_risk" ? "高风险" : "观察");
          const runId = String(row.run_id || "");
          const runShort = runId ? runId.slice(0, 8) : "-";
          return `<div class="diag-dsr-item ${escAttr(levelClass)}"><strong>${escHtml(factor)}</strong> · DSR p=${Number.isFinite(dsr) ? dsr.toFixed(3) : "-"} · ${levelText} · run ${escHtml(runShort)}</div>`;
        }).join("")}</div>`;
      }
      return `<div class="diag-meta" style="margin-top:10px">项目级 DSR p-value</div>
        <div class="diag-meta">${meta}</div>
        ${listHtml}`;
    }

    function renderCorrelationHeatmap(labels, matrix) {
      const header = `<tr><th>Factor</th>${labels.map((name) => `<th>${escHtml(name)}</th>`).join("")}</tr>`;
      const rows = labels.map((name, i) => {
        const cells = labels.map((_, j) => {
          const value = matrix[i] && matrix[i][j] !== undefined ? matrix[i][j] : null;
          if (value === null || value === undefined || !Number.isFinite(Number(value))) {
            return `<td class="muted">-</td>`;
          }
          const numeric = Number(value);
          return `<td style="background:${corrHeatColor(numeric)}">${escHtml(numeric.toFixed(3))}</td>`;
        }).join("");
        return `<tr><th>${escHtml(name)}</th>${cells}</tr>`;
      }).join("");
      return `<div class="diag-heatmap-wrap"><table class="diag-heatmap"><thead>${header}</thead><tbody>${rows}</tbody></table></div>`;
    }

    function corrHeatColor(value) {
      const v = Number(value);
      if (!Number.isFinite(v)) return "transparent";
      const mag = Math.max(0, Math.min(1, Math.abs(v)));
      if (v >= 0) {
        return `rgba(239, 68, 68, ${0.08 + mag * 0.45})`;
      }
      return `rgba(59, 130, 246, ${0.08 + mag * 0.45})`;
    }

    function renderRunTable(runs) {
      if (!runs.length) {
        $("runTable").innerHTML = "<div class='muted'>暂无运行记录。</div>";
        return;
      }
      $("runTable").innerHTML = `<div class="run-table-wrap"><table class="run-table">
        <thead><tr><th>run_id</th><th>case</th><th>profile</th><th>status</th><th>metrics</th><th>artifacts</th><th>actions</th></tr></thead>
        <tbody>${runs.map(r => {
          const shortId = r.run_id.slice(0, 10);
          const metricsHtml = renderMetricsSummary(r.summary || {});
          const artifactHtml = renderArtifactLinks(r);
          const progressHtml = renderRunProgress(r);
          const eventTrailHtml = renderRunEventTrail(r);
          const alreadySummarized = Boolean(r.summarize_draft_path);
          const errorText = r.status === "failed"
            ? (r.error || "运行失败，但后端没有返回详细错误信息。")
            : "";
          const errorType = r.error_type || "UnknownError";
          const errorMessage = r.error_message || "后端没有返回核心报错信息。";
          const errorHint = r.error_hint || "优先看 traceback 和失败阶段，定位是路径、schema、配置还是代码逻辑问题。";
          const failedStage = r.progress_message || "未知阶段";
          const errorRow = r.status === "failed"
            ? `<tr>
                <td colspan="7" style="padding:8px 10px 12px 10px">
                  <div class="run-error-box">
                    <strong>Run Error</strong>
                    <div class="run-error-summary">
                      <div><strong>失败阶段</strong> <code>${escHtml(failedStage)}</code></div>
                      <div><strong>错误类型</strong> <code>${escHtml(errorType)}</code></div>
                      <div><strong>核心报错</strong> ${escHtml(errorMessage)}</div>
                      <div><strong>改进建议</strong> ${escHtml(errorHint)}</div>
                    </div>
                    <details>
                      <summary style="cursor:pointer;font-size: 13px;color:#991b1b">查看完整 traceback</summary>
                      <pre style="margin-top:8px">${escHtml(errorText)}</pre>
                    </details>
                  </div>
                </td>
              </tr>`
            : "";
          return `<tr>
            <td><code>${shortId}</code></td>
            <td class="run-case-cell"><code>${r.case_name}</code></td>
            <td class="muted">${r.evaluation_profile}</td>
            <td>${progressHtml}${eventTrailHtml}</td>
            <td>${metricsHtml}</td>
            <td class="artifact-cell">${artifactHtml}</td>
            <td style="white-space:nowrap">
              ${r.status === "succeeded" && !alreadySummarized
                ? `<button class="ghost small" data-action="summarizeRun" data-run-id="${escAttr(r.run_id)}">刷新结果</button> `
                : ""}
              ${r.status === "succeeded" && alreadySummarized
                ? `<button class="ghost small" data-action="writebackRun" data-run-id="${escAttr(r.run_id)}" style="color:var(--brand)">写回知识库</button> `
                : ""}
              ${r.status === "succeeded" || r.status === "failed"
                ? `<button class="ghost small" data-action="deleteRun" data-run-id="${escAttr(r.run_id)}" data-case-name="${escAttr(r.case_name)}" style="color:var(--fail)">删除实验</button>`
                : ""}
            </td>
          </tr>${errorRow}`;
        }).join("")}</tbody></table></div>`;
    }

    function renderMetricsSummary(summary) {
      if (!summary || !Object.keys(summary).length) return "<span class='muted'>-</span>";
      const chips = [];
      if (summary.factor_verdict) {
        chips.push(renderMetricsChip("因子结论", summary.factor_verdict, classifyVerdict(summary.factor_verdict)));
      }
      if (summary.promotion_decision) {
        chips.push(renderMetricsChip("L2", summary.promotion_decision, classifyPromotion(summary.promotion_decision)));
      }
      if (summary.portfolio_validation_recommendation || summary.portfolio_validation_status) {
        chips.push(
          renderMetricsChip(
            "组合层",
            summary.portfolio_validation_recommendation || summary.portfolio_validation_status,
            classifyPortfolioValidation(summary.portfolio_validation_recommendation || summary.portfolio_validation_status),
          )
        );
      }
      if (summary.data_quality_status) {
        chips.push(
          renderMetricsChip(
            "数据质量",
            summary.data_quality_status,
            classifyDataQuality(summary.data_quality_status),
          )
        );
      }

      const kvPairs = [];
      if (summary.mean_rank_ic !== undefined) kvPairs.push(`RankIC ${fmtMetric(summary.mean_rank_ic)}`);
      if (summary.ic_ir !== undefined) kvPairs.push(`ICIR ${fmtMetric(summary.ic_ir)}`);
      if (summary.ic_t_stat !== undefined) kvPairs.push(`IC t-stat ${fmtMetric(summary.ic_t_stat)}`);
      if (summary.ic_p_value !== undefined) kvPairs.push(`IC p-value ${fmtMetric(summary.ic_p_value)}`);
      if (summary.dsr_pvalue !== undefined) kvPairs.push(`DSR p-value ${fmtMetric(summary.dsr_pvalue)}`);
      if (summary.split_description) kvPairs.push(`拆分 ${summary.split_description}`);
      if (summary.mean_long_short_turnover !== undefined) kvPairs.push(`换手 ${fmtMetric(summary.mean_long_short_turnover)}`);
      if (summary.eval_coverage_ratio_mean !== undefined || summary.coverage_mean !== undefined) {
        kvPairs.push(`覆盖 ${fmtMetric(summary.eval_coverage_ratio_mean ?? summary.coverage_mean)}`);
      }

      const noteSections = [];
      const blockers = firstTextList(summary.promotion_blockers);
      const risks = firstTextList(summary.portfolio_validation_major_risks);
      const verdictReasons = firstTextList(summary.factor_verdict_reasons || summary.campaign_triage_reasons);
      const flags = firstTextList(summary.rolling_instability_flags || summary.uncertainty_flags || summary.instability_flags);
      const qualityStats = [];
      if (summary.data_quality_suspended_rows !== undefined && summary.data_quality_suspended_rows !== null) {
        qualityStats.push(`停牌行 ${summary.data_quality_suspended_rows}`);
      }
      if (summary.data_quality_stale_rows !== undefined && summary.data_quality_stale_rows !== null) {
        qualityStats.push(`僵尸价行 ${summary.data_quality_stale_rows}`);
      }
      if (summary.data_quality_suspected_split_rows !== undefined && summary.data_quality_suspected_split_rows !== null) {
        qualityStats.push(`疑似拆股行 ${summary.data_quality_suspected_split_rows}`);
      }
      if (summary.data_quality_integrity_warn_count !== undefined && summary.data_quality_integrity_warn_count !== null) {
        qualityStats.push(`Integrity warn ${summary.data_quality_integrity_warn_count}`);
      }
      if (summary.data_quality_integrity_fail_count !== undefined && summary.data_quality_integrity_fail_count !== null) {
        qualityStats.push(`Integrity fail ${summary.data_quality_integrity_fail_count}`);
      }
      if (summary.data_quality_hard_fail_count !== undefined && summary.data_quality_hard_fail_count !== null) {
        qualityStats.push(`Hard fail ${summary.data_quality_hard_fail_count}`);
      }
      if (blockers.length) noteSections.push(renderMetricsNote("阻断项", blockers));
      if (risks.length) noteSections.push(renderMetricsNote("主要风险", risks));
      if (verdictReasons.length) noteSections.push(renderMetricsNote("诊断", verdictReasons));
      if (flags.length) noteSections.push(renderMetricsNote("提示", flags));
      if (qualityStats.length) noteSections.push(renderMetricsNote("数据质量摘要", qualityStats));

      return `<div class="metrics-screening">
        <div class="metrics-screening-row">${chips.join("") || "<span class='muted'>无结论摘要</span>"}</div>
        <div class="metrics-screening-kv">${kvPairs.map(v => `<span>${escHtml(v)}</span>`).join("")}</div>
        <div class="metrics-screening-notes">${noteSections.join("") || "<span class='muted'>暂无附加诊断。</span>"}</div>
      </div>`;
    }

    function renderArtifactLinks(run) {
      const groups = buildArtifactGroups(run);
      if (!groups.length) return "-";
      return `<div class="artifact-groups">${groups.map((group) => renderArtifactGroup(run, group)).join("")}</div>`;
    }

    function buildArtifactGroups(run) {
      const artifactPaths = run.artifact_paths || {};
      const labels = {
        summary: "摘要",
        case_report: "案例报告",
        experiment_card: "实验卡",
        integrity_report_markdown: "完整性报告",
        portfolio_validation_markdown: "组合验证报告",
        metrics: "核心指标",
        signal_validation_json: "信号校验",
        backtest_result_json: "回测摘要",
        group_returns: "分组收益",
        ic_decay: "IC Decay",
        factor_autocorrelation: "因子自相关",
        rolling_stability: "滚动稳定性",
        turnover: "换手率",
        coverage: "覆盖率",
        portfolio_validation_summary: "组合验证摘要",
        portfolio_validation_metrics: "组合验证指标",
        portfolio_validation_package: "组合验证包",
        run_manifest: "运行清单",
        factor_definition: "因子定义",
        factor_definition_json: "因子定义JSON",
        portfolio_recipe_json: "组合配方",
        purged_kfold_summary: "Purged K-Fold 摘要",
        purged_kfold_folds: "Purged K-Fold 分折明细",
        barra_attribution_summary: "Barra 归因摘要（实验）",
        barra_attribution_timeseries: "Barra 归因时序（实验）",
        market_impact_summary: "冲击成本摘要（实验）",
        market_impact_orders: "冲击成本明细（实验）",
        ic_timeseries: "IC时序",
        integrity_report_json: "完整性JSON",
      };
      const classByKey = {
        summary: "artifact-doc",
        case_report: "artifact-doc",
        experiment_card: "artifact-doc",
        integrity_report_markdown: "artifact-doc",
        portfolio_validation_markdown: "artifact-doc",
        metrics: "artifact-data",
        signal_validation_json: "artifact-data",
        backtest_result_json: "artifact-data",
        group_returns: "artifact-data",
        ic_decay: "artifact-data",
        factor_autocorrelation: "artifact-data",
        rolling_stability: "artifact-data",
        turnover: "artifact-data",
        coverage: "artifact-data",
        portfolio_validation_summary: "artifact-data",
        portfolio_validation_metrics: "artifact-data",
        portfolio_validation_package: "artifact-detail",
        run_manifest: "artifact-detail",
        factor_definition: "artifact-detail",
        factor_definition_json: "artifact-detail",
        portfolio_recipe_json: "artifact-detail",
        purged_kfold_summary: "artifact-data",
        purged_kfold_folds: "artifact-data",
        barra_attribution_summary: "artifact-data",
        barra_attribution_timeseries: "artifact-data",
        market_impact_summary: "artifact-data",
        market_impact_orders: "artifact-detail",
        ic_timeseries: "artifact-detail",
        integrity_report_json: "artifact-detail",
      };
      const screeningKeys = [
        "summary",
        "metrics",
        "experiment_card",
        "signal_validation_json",
        "integrity_report_markdown",
      ];
      const diagnosticKeys = [
        "case_report",
        "backtest_result_json",
        "group_returns",
        "ic_decay",
        "factor_autocorrelation",
        "purged_kfold_summary",
        "purged_kfold_folds",
        "rolling_stability",
        "turnover",
        "coverage",
        "ic_timeseries",
      ];
      const metadataKeys = [
        "run_manifest",
        "factor_definition",
        "factor_definition_json",
        "portfolio_recipe_json",
        "integrity_report_json",
      ];
      const portfolioKeys = [
        "portfolio_validation_summary",
        "portfolio_validation_metrics",
        "portfolio_validation_markdown",
        "portfolio_validation_package",
      ];
      const experimentalKeys = [
        "barra_attribution_summary",
        "barra_attribution_timeseries",
        "market_impact_summary",
        "market_impact_orders",
      ];
      const decorate = (keys) => keys
        .filter((key) => Boolean(artifactPaths[key]))
        .map((key) => ({
          key,
          label: labels[key] || key,
          cssClass: classByKey[key] || "artifact-detail",
        }));
      const summary = run.summary || {};
      const promotionText = String(summary.promotion_decision || "").toLowerCase();
      const portfolioText = String(
        summary.portfolio_validation_recommendation || summary.portfolio_validation_status || ""
      ).toLowerCase();
      const showPortfolioGroup = decorate(portfolioKeys).length > 0 && (
        promotionText.includes("promote")
        || !portfolioText.includes("not evaluated")
      );
      const groups = [
        {
          title: "初筛结论",
          entries: decorate(screeningKeys),
          folded: false,
        },
        {
          title: "诊断详情",
          entries: decorate(diagnosticKeys),
          folded: true,
        },
        {
          title: "复现留档",
          entries: decorate(metadataKeys),
          folded: true,
        },
      ];
      if (showPortfolioGroup) {
        groups.push({
          title: "L2 结果",
          entries: decorate(portfolioKeys),
          folded: true,
        });
      }
      const experimentalEntries = decorate(experimentalKeys);
      if (experimentalEntries.length) {
        groups.push({
          title: "实验隔离 (L3)",
          entries: experimentalEntries,
          folded: true,
        });
      }
      return groups.filter((group) => group.entries.length > 0);
    }

    function renderArtifactGroup(run, group) {
      const links = `<div class="artifact-links">${group.entries.map(({key, label, cssClass}) =>
        `<span class="artifact-link ${cssClass}" data-action="viewArtifact" data-run-id="${escAttr(run.run_id)}" data-artifact-key="${escAttr(key)}">${escHtml(label)}</span>`
      ).join("")}</div>`;
      if (!group.folded) {
        return `<div class="artifact-group"><div class="artifact-group-title">${escHtml(group.title)}</div>${links}</div>`;
      }
      return `<details class="artifact-group">
        <summary>${escHtml(group.title)} (${group.entries.length})</summary>
        ${links}
      </details>`;
    }

    function renderMetricsChip(label, value, tone) {
      return `<span class="metrics-chip ${escAttr(tone)}">${escHtml(label)}: ${escHtml(translateDiagnosticText(value))}</span>`;
    }

    function renderMetricsNote(label, items) {
      return `<div class="metrics-screening-note"><strong>${escHtml(label)}</strong>${escHtml(items.map((item) => translateDiagnosticText(item)).join("；"))}</div>`;
    }

    function firstTextList(value) {
      if (!Array.isArray(value)) return [];
      return value.map((item) => String(item || "").trim()).filter(Boolean).slice(0, 3);
    }

    function fmtMetric(value) {
      if (value === null || value === undefined || value === "") return "-";
      if (typeof value === "number") return value.toFixed(4);
      return String(value);
    }

    function translateDiagnosticText(value) {
      const text = String(value ?? "").trim();
      if (!text) return "";
      const exactMap = new Map([
        ["Strong candidate", "强候选因子"],
        ["Promising but fragile", "有潜力但偏脆弱"],
        ["Mixed evidence", "证据分化"],
        ["Weak / noisy", "偏弱 / 噪声较大"],
        ["Fails basic robustness", "未通过基础稳健性"],
        ["Advance to Level 2", "建议进入 Level 2"],
        ["Strong Level 1 candidate", "强 Level 1 候选"],
        ["Needs refinement", "需要继续打磨"],
        ["Fragile / monitor", "偏脆弱 / 持续观察"],
        ["Drop for now", "当前建议淘汰"],
        ["Promote to Level 2", "建议晋升到 Level 2"],
        ["Hold for refinement", "暂缓晋升，先继续打磨"],
        ["Blocked from Level 2", "暂不允许进入 Level 2"],
        ["Credible at portfolio level", "组合层面可信"],
        ["Needs portfolio refinement", "组合层面还需继续打磨"],
        ["Not evaluated (not promoted)", "未进入组合层验证"],
        ["blocked by weak single-case verdict", "被阻断：单案例结论偏弱"],
        ["blocked by thin coverage", "被阻断：覆盖率过低"],
        ["blocked by fragile subperiod evidence", "被阻断：子区间证据不稳"],
        ["blocked by unstable rolling evidence", "被阻断：滚动窗口稳定性不足"],
        ["blocked by high uncertainty overlap", "被阻断：不确定性区间重叠过大"],
        ["blocked by weak neutralized evidence", "被阻断：中性化后证据偏弱"],
        ["blocked by poor turnover efficiency", "被阻断：换手效率偏差"],
        ["blocked by unsuccessful case status", "被阻断：运行未成功完成"],
        ["promotion context still carries unresolved blockers", "晋升上下文里仍有未解决的阻断项"],
        ["evaluation window is too short for basic robustness", "评估窗口太短，尚不足以支持基础稳健性判断"],
        ["coverage is too thin for reliable evaluation", "覆盖率过低，评估结论不够可靠"],
        ["signal direction is unstable under uncertainty", "在不确定性区间下，信号方向不稳定"],
      ]);
      if (exactMap.has(text)) return exactMap.get(text);
      return text
        .replaceAll("positive IC and RankIC means", "IC 与 RankIC 均值为正")
        .replaceAll("IC and RankIC signs are consistently positive", "IC 与 RankIC 方向整体保持为正")
        .replaceAll("IC and RankIC validity is high", "IC 与 RankIC 有效率较高")
        .replaceAll("long-short spread is positive with positive IR", "多空收益为正且 IR 为正")
        .replaceAll("robust across subperiods", "子区间表现较稳健")
        .replaceAll("evidence is persistent across rolling windows", "滚动窗口中的证据具备持续性")
        .replaceAll("signal weakens materially in some periods", "信号在部分阶段明显走弱")
        .replaceAll("rolling evidence suggests regime dependence", "滚动证据提示存在阶段依赖")
        .replaceAll("rolling factor performance is unstable through time", "因子表现随时间波动较大")
        .replaceAll("confidence interval overlaps zero", "置信区间跨过 0")
        .replaceAll("apparent edge is weak relative to estimation noise", "表面优势相对估计噪声偏弱")
        .replaceAll("coverage and validity are sufficient", "覆盖率与有效率达标")
        .replaceAll("turnover efficiency is acceptable", "换手效率可接受")
        .replaceAll("confidence intervals remain supportive", "置信区间仍提供支持")
        .replaceAll("stable across rolling windows", "滚动窗口稳定性较好")
        .replaceAll("single-case verdict is strong", "单案例结论较强")
        .replaceAll("single-case verdict indicates fragility", "单案例结论提示存在脆弱性")
        .replaceAll("single-case verdict is mixed", "单案例结论分化")
        .replaceAll("single-case verdict is weak", "单案例结论偏弱")
        .replaceAll("coverage too thin", "覆盖率过低")
        .replaceAll("fragile across subperiods", "子区间稳定性不足")
        .replaceAll("fragile across rolling windows", "滚动窗口稳定性不足")
        .replaceAll("uncertainty remains high", "不确定性仍然偏高")
        .replaceAll("coverage is limited", "覆盖率偏有限")
        .replaceAll("turnover efficiency weak", "换手效率偏弱")
        .replaceAll("factor verdict is not yet strong", "因子结论还不够强")
        .replaceAll("uncertainty support is incomplete", "不确定性支持仍不充分")
        .replaceAll("subperiod robustness is not yet persistent", "子区间稳健性还不够持续")
        .replaceAll("rolling stability is not yet persistent", "滚动稳定性还不够持续")
        .replaceAll("coverage/validity are below promotion target", "覆盖率 / 有效率仍低于晋升目标")
        .replaceAll("neutralization weakens evidence", "中性化后证据走弱")
        .replaceAll("neutralization evidence is unavailable", "缺少中性化证据")
        .replaceAll("campaign triage is favorable but promotion gate remains unmet", "初筛结果尚可，但晋升门槛仍未满足")
        .replaceAll("factor verdict is strong", "因子结论较强")
        .replaceAll("robust evidence survives neutralization", "中性化后仍保留较强证据");
    }

    function inferArtifactKind(key, ctype) {
      const text = String(ctype || "").toLowerCase();
      if (key.endsWith("_markdown") || key === "summary" || key === "case_report" || key === "experiment_card") return "markdown";
      if (key.endsWith(".md") || text.includes("markdown")) return "markdown";
      if (text.includes("html")) return "html";
      if (text.includes("json")) return "json";
      if (key.endsWith("json")) return "json";
      if (key.includes("csv") || text.includes("csv")) return "csv";
      if (text.includes("yaml") || key.includes("yaml")) return "yaml";
      if (text.includes("text") || text.includes("plain")) return "text";
      return "binary";
    }

    function renderArtifactViewerShell(key, kindLabel, bodyHtml, rawText = "") {
      state.artifactRawText = String(rawText || "");
      return `<div class="artifact-viewer-shell">
        <div class="artifact-viewer-head">
          <div class="artifact-viewer-title">
            <strong style="color:var(--brand)">${escHtml(key)}</strong>
            <span class="artifact-viewer-kind">${escHtml(kindLabel)}</span>
          </div>
          <button class="copy-btn" data-action="copyArtifact">复制原文</button>
        </div>
        ${bodyHtml}
      </div>`;
    }

    function renderPlainArtifact(key, text, kindLabel = "文本") {
      return renderArtifactViewerShell(
        key,
        kindLabel,
        `<pre id="artifactText" class="artifact-viewer-raw">${escHtml(text)}</pre>`,
        text,
      );
    }

    function renderMarkdownArtifact(key, text) {
      return renderArtifactViewerShell(
        key,
        "Markdown",
        `<div id="artifactText" class="md-box" style="max-height:640px;background:#ffffff">${mdRender(text || "")}</div>`,
        text,
      );
    }

    function renderJsonArtifact(key, text) {
      try {
        const parsed = JSON.parse(text);
        if (key === "purged_kfold_summary") {
          return renderPurgedKfoldSummaryJson(key, parsed, text);
        }
        if (key === "portfolio_validation_metrics") {
          return renderPortfolioValidationMetricsJson(key, parsed, text);
        }
        if (key === "barra_attribution_summary") {
          return renderBarraAttributionSummaryJson(key, parsed, text);
        }
        if (key === "market_impact_summary") {
          return renderMarketImpactSummaryJson(key, parsed, text);
        }
        const entries = parsed && typeof parsed === "object" && !Array.isArray(parsed)
          ? Object.entries(parsed).slice(0, 12)
          : [];
        const cards = entries.length
          ? `<div class="artifact-json-grid">${entries.map(([name, value]) => `
              <div class="artifact-json-card">
                <strong>${escHtml(name)}</strong>
                <pre>${escHtml(typeof value === "string" ? value : JSON.stringify(value, null, 2))}</pre>
              </div>
            `).join("")}</div>`
          : "";
        const raw = `<details><summary>查看完整 JSON</summary><pre id="artifactText" class="artifact-viewer-raw">${escHtml(JSON.stringify(parsed, null, 2))}</pre></details>`;
        return renderArtifactViewerShell(key, "JSON", cards + raw, JSON.stringify(parsed, null, 2));
      } catch (_) {
        return renderPlainArtifact(key, text, "JSON");
      }
    }

    function renderPurgedKfoldSummaryJson(key, parsed, rawText) {
      const nFolds = Number(parsed.n_folds);
      const nUsed = Number(parsed.n_splits_used);
      const meanIc = Number(parsed.mean_ic);
      const meanRankIc = Number(parsed.mean_rank_ic);
      const meanSharpe = Number(parsed.mean_sharpe);
      const verdict = String(parsed.verdict || "-");
      const status = String(parsed.status || "-");
      const reasons = Array.isArray(parsed.reasons) ? parsed.reasons.slice(0, 4) : [];
      const body = `<div class="artifact-json-grid">
          <div class="artifact-json-card"><strong>Status</strong><pre>${escHtml(status)}</pre></div>
          <div class="artifact-json-card"><strong>Verdict</strong><pre>${escHtml(verdict)}</pre></div>
          <div class="artifact-json-card"><strong>n_folds</strong><pre>${Number.isFinite(nFolds) ? nFolds : "-"}</pre></div>
          <div class="artifact-json-card"><strong>n_splits_used</strong><pre>${Number.isFinite(nUsed) ? nUsed : "-"}</pre></div>
          <div class="artifact-json-card"><strong>mean_ic</strong><pre>${Number.isFinite(meanIc) ? meanIc.toFixed(4) : "-"}</pre></div>
          <div class="artifact-json-card"><strong>mean_rank_ic</strong><pre>${Number.isFinite(meanRankIc) ? meanRankIc.toFixed(4) : "-"}</pre></div>
          <div class="artifact-json-card"><strong>mean_sharpe</strong><pre>${Number.isFinite(meanSharpe) ? meanSharpe.toFixed(4) : "-"}</pre></div>
          <div class="artifact-json-card"><strong>purge/embargo</strong><pre>${escHtml(String(parsed.purge_days ?? "-"))}/${escHtml(String(parsed.embargo_days ?? "-"))}</pre></div>
        </div>
        ${reasons.length ? `<div class="artifact-viewer-meta">原因：${escHtml(reasons.join("；"))}</div>` : ""}
        <details><summary>查看完整 JSON</summary><pre id="artifactText" class="artifact-viewer-raw">${escHtml(rawText)}</pre></details>`;
      return renderArtifactViewerShell(key, "JSON · Purged K-Fold", body, rawText);
    }

    function renderPortfolioValidationMetricsJson(key, parsed, rawText) {
      const concentration = parsed && typeof parsed === "object" && parsed.concentration_exposure_diagnostics && typeof parsed.concentration_exposure_diagnostics === "object"
        ? parsed.concentration_exposure_diagnostics
        : {};
      const scenarioRows = Array.isArray(parsed.scenario_metrics) ? parsed.scenario_metrics : [];
      const holdingRows = Array.isArray(parsed.holding_period_sensitivity) ? parsed.holding_period_sensitivity : [];
      const weightingRows = Array.isArray(parsed.weighting_sensitivity) ? parsed.weighting_sensitivity : [];

      const concentrationCards = [
        ["max_abs_weight_mean", concentration.max_abs_weight_mean],
        ["top5_abs_weight_share_mean", concentration.top5_abs_weight_share_mean],
        ["effective_names_mean", concentration.effective_names_mean],
        ["gross_exposure_mean", concentration.gross_exposure_mean],
        ["net_exposure_mean", concentration.net_exposure_mean],
      ].map(([name, value]) =>
        `<div class="artifact-json-card"><strong>${escHtml(String(name))}</strong><pre>${Number.isFinite(Number(value)) ? Number(value).toFixed(4) : "-"}</pre></div>`
      ).join("");

      const holdingChart = renderMultiLineChart("Holding Period 敏感性", [
        {label: "Mean Return", color: "#0ea5e9", points: makeLinePoints(holdingRows, "holding_period", "mean_portfolio_return")},
        {label: "Cost-Adjusted Return", color: "#22c55e", points: makeLinePoints(holdingRows, "holding_period", "mean_cost_adjusted_return_review_rate")},
      ]);

      const byMethodReturns = weightingRows.map((row) => ({
        label: String(row.weighting_method || "unknown"),
        value: Number(row.mean_portfolio_return),
      }));
      const byMethodConcentration = averageByMethod(scenarioRows, "max_abs_weight_mean");
      const returnBars = renderCategoryBarChart("不同权重法：Mean Portfolio Return", byMethodReturns);
      const concentrationBars = renderCategoryBarChart("不同权重法：Max |Weight|", byMethodConcentration);

      const body = `<div class="artifact-viewer-meta">组合约束诊断（MVO / 风险约束近似）：关注集中度、敏感性与不同权重法表现。</div>
        <div class="artifact-json-grid">${concentrationCards}</div>
        ${holdingChart}
        ${returnBars}
        ${concentrationBars}
        <details><summary>查看完整 JSON</summary><pre id="artifactText" class="artifact-viewer-raw">${escHtml(rawText)}</pre></details>`;
      return renderArtifactViewerShell(key, "JSON · 组合约束诊断", body, rawText);
    }

    function renderBarraAttributionSummaryJson(key, parsed, rawText) {
      const cards = [
        ["start_date", parsed.start_date],
        ["end_date", parsed.end_date],
        ["total_return", parsed.total_return],
        ["specific_return", parsed.specific_return],
        ["residual_return", parsed.residual_return],
      ].map(([name, value]) =>
        `<div class="artifact-json-card"><strong>${escHtml(String(name))}</strong><pre>${escHtml(String(value ?? "-"))}</pre></div>`
      ).join("");
      const body = `<div class="artifact-viewer-meta">实验隔离（Level 3）: Barra 归因仅用于实验诊断，不进入默认结论门控。</div>
        <div class="artifact-json-grid">${cards}</div>
        <details><summary>查看完整 JSON</summary><pre id="artifactText" class="artifact-viewer-raw">${escHtml(rawText)}</pre></details>`;
      return renderArtifactViewerShell(key, "JSON · 实验归因", body, rawText);
    }

    function renderMarketImpactSummaryJson(key, parsed, rawText) {
      const cards = [
        ["model_name", parsed.model_name],
        ["avg_impact_bps", parsed.avg_impact_bps],
        ["median_impact_bps", parsed.median_impact_bps],
        ["p95_impact_bps", parsed.p95_impact_bps],
        ["max_impact_bps", parsed.max_impact_bps],
        ["n_orders", parsed.n_orders],
      ].map(([name, value]) =>
        `<div class="artifact-json-card"><strong>${escHtml(String(name))}</strong><pre>${escHtml(String(value ?? "-"))}</pre></div>`
      ).join("");
      const body = `<div class="artifact-viewer-meta">实验隔离（Level 3）: 预估冲击成本仅作实验参考，不进入默认推荐结论。</div>
        <div class="artifact-json-grid">${cards}</div>
        <details><summary>查看完整 JSON</summary><pre id="artifactText" class="artifact-viewer-raw">${escHtml(rawText)}</pre></details>`;
      return renderArtifactViewerShell(key, "JSON · 实验冲击成本", body, rawText);
    }

    function parseCsvLines(text) {
      const rows = String(text || "").trim().split("\\n").map((line) => line.replace(/\\r$/, "")).filter(Boolean);
      if (!rows.length) return null;
      const splitRow = (row) => row.split(",").map((cell) => cell.trim());
      const header = splitRow(rows[0]);
      const body = rows.slice(1, 13).map(splitRow);
      return {header, body, totalRows: Math.max(rows.length - 1, 0)};
    }

    function parseCsvRows(text) {
      const parsed = parseCsvLines(text);
      if (!parsed) return [];
      const rows = String(text || "").trim().split("\\n").map((line) => line.replace(/\\r$/, "")).filter(Boolean);
      if (rows.length <= 1) return [];
      const splitRow = (row) => row.split(",").map((cell) => cell.trim());
      return rows.slice(1).map((row) => {
        const values = splitRow(row);
        const item = {};
        parsed.header.forEach((name, idx) => {
          item[String(name)] = values[idx] || "";
        });
        return item;
      });
    }

    function renderCsvPreviewTable(parsed, text) {
      return `<div class="artifact-viewer-meta">预览前 ${parsed.body.length} 行，共 ${parsed.totalRows} 行数据。</div>
        <div class="artifact-table-wrap">
          <table>
            <thead><tr>${parsed.header.map((cell) => `<th>${escHtml(cell)}</th>`).join("")}</tr></thead>
            <tbody>${parsed.body.map((row) => `<tr>${parsed.header.map((_, idx) => `<td>${escHtml(row[idx] || "")}</td>`).join("")}</tr>`).join("")}</tbody>
          </table>
        </div>
        <details><summary>查看原始 CSV</summary><pre id="artifactText" class="artifact-viewer-raw">${escHtml(text)}</pre></details>`;
    }

    function toFiniteNumber(value) {
      const n = Number(value);
      return Number.isFinite(n) ? n : null;
    }

    function makeLinePoints(rows, xKey, yKey) {
      return rows
        .map((row) => ({
          x: toFiniteNumber(row[xKey]),
          y: toFiniteNumber(row[yKey]),
        }))
        .filter((p) => p.x !== null && p.y !== null)
        .map((p) => ({x: Number(p.x), y: Number(p.y)}))
        .sort((a, b) => a.x - b.x);
    }

    function renderMultiLineChart(title, lines) {
      const lineRows = lines.filter((line) => Array.isArray(line.points) && line.points.length >= 2);
      if (!lineRows.length) return `<div class="muted">图表数据不足（至少需要 2 个有效点）。</div>`;

      const width = 520;
      const height = 200;
      const padL = 44;
      const padR = 16;
      const padT = 16;
      const padB = 30;
      const innerW = width - padL - padR;
      const innerH = height - padT - padB;

      let xMin = Infinity;
      let xMax = -Infinity;
      let yMin = Infinity;
      let yMax = -Infinity;
      for (const line of lineRows) {
        for (const point of line.points) {
          if (point.x < xMin) xMin = point.x;
          if (point.x > xMax) xMax = point.x;
          if (point.y < yMin) yMin = point.y;
          if (point.y > yMax) yMax = point.y;
        }
      }
      if (!Number.isFinite(xMin) || !Number.isFinite(xMax) || !Number.isFinite(yMin) || !Number.isFinite(yMax)) {
        return `<div class="muted">图表数据不可解析。</div>`;
      }
      if (Math.abs(xMax - xMin) < 1e-12) {
        xMax = xMin + 1.0;
      }
      if (Math.abs(yMax - yMin) < 1e-12) {
        yMax += 1.0;
        yMin -= 1.0;
      }

      const xPos = (x) => padL + ((x - xMin) / (xMax - xMin)) * innerW;
      const yPos = (y) => padT + (1 - (y - yMin) / (yMax - yMin)) * innerH;
      const zeroY = (yMin <= 0 && yMax >= 0) ? yPos(0) : null;

      const svgLines = lineRows.map((line) => {
        const points = line.points.map((p) => `${xPos(p.x).toFixed(2)},${yPos(p.y).toFixed(2)}`).join(" ");
        return `<polyline points="${points}" fill="none" stroke="${escAttr(line.color)}" stroke-width="2.2" />`;
      }).join("");

      const legend = lineRows.map((line) =>
        `<span class="artifact-chart-legend-item"><span class="artifact-chart-dot" style="background:${escAttr(line.color)}"></span>${escHtml(line.label)}</span>`
      ).join("");

      const xTicks = [xMin, xMax].map((value) => Number.isInteger(value) ? String(value) : value.toFixed(2));
      const yTicks = [yMin, yMax].map((value) => value.toFixed(4));

      return `<div class="artifact-chart-card">
        <p class="artifact-chart-title">${escHtml(title)}</p>
        <svg class="artifact-line-chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="${escAttr(title)}">
          <line x1="${padL}" x2="${width - padR}" y1="${height - padB}" y2="${height - padB}" stroke="#cbd5e1" stroke-width="1" />
          <line x1="${padL}" x2="${padL}" y1="${padT}" y2="${height - padB}" stroke="#e2e8f0" stroke-width="1" />
          ${zeroY === null ? "" : `<line x1="${padL}" x2="${width - padR}" y1="${zeroY.toFixed(2)}" y2="${zeroY.toFixed(2)}" stroke="#fecaca" stroke-width="1" stroke-dasharray="4 3" />`}
          ${svgLines}
          <text x="${padL}" y="${height - 8}" fill="#64748b" font-size="11">${escHtml(xTicks[0])}</text>
          <text x="${width - padR}" y="${height - 8}" fill="#64748b" font-size="11" text-anchor="end">${escHtml(xTicks[1])}</text>
          <text x="${padL - 6}" y="${padT + 4}" fill="#64748b" font-size="11" text-anchor="end">${escHtml(yTicks[1])}</text>
          <text x="${padL - 6}" y="${height - padB + 4}" fill="#64748b" font-size="11" text-anchor="end">${escHtml(yTicks[0])}</text>
        </svg>
        <div class="artifact-chart-legend">${legend}</div>
      </div>`;
    }

    function renderCategoryBarChart(title, rows) {
      const items = (Array.isArray(rows) ? rows : [])
        .map((row) => ({
          label: String(row.label || "").trim(),
          value: Number(row.value),
        }))
        .filter((row) => row.label && Number.isFinite(row.value));
      if (!items.length) {
        return `<div class="artifact-chart-card"><p class="artifact-chart-title">${escHtml(title)}</p><div class="muted">暂无可视化数据。</div></div>`;
      }
      const maxAbs = Math.max(...items.map((row) => Math.abs(row.value)), 1e-9);
      const body = items.map((row) => {
        const ratio = Math.max(0, Math.min(1, Math.abs(row.value) / maxAbs));
        return `<div class="artifact-bar-row">
          <div class="artifact-bar-head"><span>${escHtml(row.label)}</span><span>${escHtml(row.value.toFixed(4))}</span></div>
          <div class="artifact-bar-track"><div class="artifact-bar-fill" style="width:${(ratio * 100).toFixed(1)}%"></div></div>
        </div>`;
      }).join("");
      return `<div class="artifact-chart-card">
        <p class="artifact-chart-title">${escHtml(title)}</p>
        <div class="artifact-bar-list">${body}</div>
      </div>`;
    }

    function averageByMethod(rows, metricKey) {
      const bucket = {};
      for (const row of (Array.isArray(rows) ? rows : [])) {
        const method = String(row.weighting_method || "").trim();
        const value = Number(row[metricKey]);
        if (!method || !Number.isFinite(value)) continue;
        if (!bucket[method]) bucket[method] = [];
        bucket[method].push(value);
      }
      return Object.entries(bucket).map(([label, values]) => ({
        label,
        value: values.reduce((acc, item) => acc + item, 0) / values.length,
      }));
    }

    function renderIcDecayCsv(key, text, parsed) {
      const rows = parseCsvRows(text);
      const icPoints = makeLinePoints(rows, "horizon", "mean_ic");
      const rankIcPoints = makeLinePoints(rows, "horizon", "mean_rank_ic");
      const chart = renderMultiLineChart("IC Decay（X=Horizon）", [
        {label: "Mean IC", color: "#0ea5e9", points: icPoints},
        {label: "Mean RankIC", color: "#22c55e", points: rankIcPoints},
      ]);
      const bodyHtml = `<div class="artifact-viewer-meta">IC 衰减诊断：随 horizon 增大观察预测力变化。</div>
        ${chart}
        ${renderCsvPreviewTable(parsed, text)}`;
      return renderArtifactViewerShell(key, "CSV · 诊断图", bodyHtml, text);
    }

    function renderFactorAutocorrCsv(key, text, parsed) {
      const rows = parseCsvRows(text);
      const acPoints = makeLinePoints(rows, "lag", "mean_autocorr");
      const chart = renderMultiLineChart("因子自相关（X=Lag）", [
        {label: "Mean Autocorr", color: "#f97316", points: acPoints},
      ]);
      const bodyHtml = `<div class="artifact-viewer-meta">因子持久性诊断：Lag 越大，自相关通常逐步下降。</div>
        ${chart}
        ${renderCsvPreviewTable(parsed, text)}`;
      return renderArtifactViewerShell(key, "CSV · 诊断图", bodyHtml, text);
    }

    function renderPurgedKfoldFoldsCsv(key, text, parsed) {
      const rows = parseCsvRows(text);
      const icChart = renderMultiLineChart("Purged K-Fold: IC / RankIC", [
        {label: "Mean IC", color: "#0ea5e9", points: makeLinePoints(rows, "fold_id", "mean_ic")},
        {label: "Mean RankIC", color: "#22c55e", points: makeLinePoints(rows, "fold_id", "mean_rank_ic")},
      ]);
      const sharpeChart = renderMultiLineChart("Purged K-Fold: Long-Short Sharpe", [
        {label: "Fold Sharpe", color: "#f97316", points: makeLinePoints(rows, "fold_id", "long_short_sharpe")},
      ]);
      const bodyHtml = `<div class="artifact-viewer-meta">OOS 多折诊断：观察各 fold 的 IC 与 Sharpe 一致性。</div>
        ${icChart}
        ${sharpeChart}
        ${renderCsvPreviewTable(parsed, text)}`;
      return renderArtifactViewerShell(key, "CSV · Purged K-Fold", bodyHtml, text);
    }

    function renderBarraAttributionTimeseriesCsv(key, text, parsed) {
      const rows = parseCsvRows(text);
      const numericColumns = parsed.header.filter((name) => {
        if (name === "date") return false;
        return rows.some((row) => Number.isFinite(Number(row[name])));
      });
      const selected = numericColumns.slice(0, 4);
      const lines = selected.map((col, idx) => ({
        label: col,
        color: ["#0ea5e9", "#22c55e", "#f97316", "#a855f7"][idx % 4],
        points: rows
          .map((row, pos) => ({x: pos + 1, y: Number(row[col])}))
          .filter((p) => Number.isFinite(p.y)),
      }));
      const chart = renderMultiLineChart("Barra 归因时序（实验）", lines);
      const bodyHtml = `<div class="artifact-viewer-meta">实验隔离：默认展示前 4 列贡献时序（按行序号作为横轴）。</div>
        ${chart}
        ${renderCsvPreviewTable(parsed, text)}`;
      return renderArtifactViewerShell(key, "CSV · 实验归因", bodyHtml, text);
    }

    function renderCsvArtifact(key, text) {
      const parsed = parseCsvLines(text);
      if (!parsed) return renderPlainArtifact(key, text, "CSV");
      if (key === "ic_decay") {
        return renderIcDecayCsv(key, text, parsed);
      }
      if (key === "factor_autocorrelation") {
        return renderFactorAutocorrCsv(key, text, parsed);
      }
      if (key === "purged_kfold_folds") {
        return renderPurgedKfoldFoldsCsv(key, text, parsed);
      }
      if (key === "barra_attribution_timeseries") {
        return renderBarraAttributionTimeseriesCsv(key, text, parsed);
      }
      const bodyHtml = renderCsvPreviewTable(parsed, text);
      return renderArtifactViewerShell(key, "CSV", bodyHtml, text);
    }

    function classifyVerdict(value) {
      const text = String(value || "").toLowerCase();
      if (text.includes("strong") || text.includes("promising")) return "good";
      if (text.includes("fragile") || text.includes("warning")) return "warn";
      if (text.includes("fail") || text.includes("blocked") || text.includes("weak")) return "bad";
      return "warn";
    }

    function classifyPromotion(value) {
      const text = String(value || "").toLowerCase();
      if (text.includes("promote")) return "good";
      if (text.includes("blocked") || text.includes("reject")) return "bad";
      return "warn";
    }

    function classifyPortfolioValidation(value) {
      const text = String(value || "").toLowerCase();
      if (text.includes("credible") || text.includes("robust")) return "good";
      if (text.includes("not evaluated") || text.includes("fragile")) return "bad";
      return "warn";
    }

    function classifyDataQuality(value) {
      const text = String(value || "").toLowerCase().trim();
      if (text === "pass") return "good";
      if (text === "fail") return "bad";
      return "warn";
    }

    function renderRunProgress(run) {
      const progressPercent = Number.isFinite(run.progress_percent)
        ? Math.max(0, Math.min(100, Number(run.progress_percent)))
        : null;
      const progressMessage = run.progress_message
        || (run.status === "succeeded" ? "运行完成" : run.status === "failed" ? "运行失败" : "-");
      const barHtml = (run.status === "running" || run.status === "queued") && progressPercent !== null
        ? `<div class="run-progress-bar"><div class="run-progress-fill" style="width:${progressPercent}%"></div></div>`
        : "";
      const percentHtml = progressPercent !== null
        ? ` · ${progressPercent}%`
        : "";
      const updatedHtml = run.updated_at_utc
        ? `<div class="run-progress-meta">最近更新：${escHtml(formatUtc(run.updated_at_utc))}${percentHtml}</div>`
        : "";
      return `<div class="run-status-cell">
        <span class="status ${escAttr(run.status || "pending")}">${escHtml(run.status || "pending")}</span>
        <div class="run-progress-text">${escHtml(progressMessage)}</div>
        ${barHtml}
        ${updatedHtml}
      </div>`;
    }

    function renderRunEventTrail(run) {
      const events = Array.isArray(run.progress_events) ? run.progress_events : [];
      if (!events.length) return "";
      const rows = events.slice(-5).reverse().map((event) => {
        const percent = Number.isFinite(event.percent) ? `${Number(event.percent)}%` : "";
        return `<div class="run-event-item">
          <div>${escHtml(event.message || "-")}${percent ? ` <span class="muted">${escHtml(percent)}</span>` : ""}</div>
          <time>${escHtml(formatUtc(event.ts || ""))}</time>
        </div>`;
      }).join("");
      return `<details class="run-event-trail">
        <summary>查看进度轨迹</summary>
        <div class="run-event-list">${rows}</div>
      </details>`;
    }

    function formatUtc(ts) {
      if (!ts) return "-";
      const date = new Date(ts);
      if (Number.isNaN(date.getTime())) return ts;
      return date.toLocaleString("zh-CN", {
        hour12: false,
        month: "2-digit",
        day: "2-digit",
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
      });
    }

    async function viewArtifact(runId, key) {
      if (!state.selectedProject) return;
      try {
        const res = await fetch(`/api/projects/${enc(state.selectedProject)}/runs/${enc(runId)}/artifact/${enc(key)}`);
        const ctype = res.headers.get("Content-Type") || "";
        if (ctype.includes("text") || ctype.includes("json") || ctype.includes("yaml") || ctype.includes("markdown")) {
          const text = await res.text();
          const kind = inferArtifactKind(key, ctype);
          if (kind === "markdown") {
            $("artifactViewer").innerHTML = renderMarkdownArtifact(key, text);
          } else if (kind === "json") {
            $("artifactViewer").innerHTML = renderJsonArtifact(key, text);
          } else if (kind === "csv") {
            $("artifactViewer").innerHTML = renderCsvArtifact(key, text);
          } else if (kind === "yaml") {
            $("artifactViewer").innerHTML = renderPlainArtifact(key, text, "YAML");
          } else {
            $("artifactViewer").innerHTML = renderPlainArtifact(key, text, "文本");
          }
        } else if (ctype.includes("html")) {
          const text = await res.text();
          state.artifactRawText = text;
          $("artifactViewer").innerHTML = renderArtifactViewerShell(
            key,
            "HTML",
            `<iframe srcdoc="${escAttr(text)}" style="width:100%;height:600px;border:1px solid var(--line);border-radius:8px;background:#ffffff"></iframe>`,
            text,
          );
        } else {
          state.artifactRawText = "";
          $("artifactViewer").innerHTML = `<div class="muted">Binary artifact: ${key} (${ctype})</div>`;
        }
      } catch(e) { $("artifactViewer").innerHTML = `<div class="muted">Error: ${e.message}</div>`; }
    }

    async function summarizeRun(runId) {
      try {
        const data = await api(
          `/api/projects/${enc(state.selectedProject)}/runs/${enc(runId)}/summarize`,
          "POST", {}
        );
        showResponse("validationResponseBox", data);
        await loadRuns();
        await loadProjectDetail();
      } catch(e) { alert(e.message); }
    }

    async function deleteRun(runId, caseName) {
      if (!confirm(`确认删除实验 ${caseName}（${runId.slice(0,10)}）？\n\n将删除：输出产物目录、运行摘要、writeback 草稿。此操作不可撤回。`)) return;
      try {
        await api(
          `/api/projects/${enc(state.selectedProject)}/runs/${enc(runId)}`,
          "DELETE"
        );
        await loadRuns();
        await loadProjectDetail();
      } catch(e) { alert("删除失败：" + e.message); }
    }

    async function writebackRun(runId) {
      switchView("view-writeback");
      try {
        await loadDrafts();
      } catch(e) { /* drafts tab will show its own error */ }
    }

    function renderMarkdownHub(id, text) {
      const box = $(id);
      if (!box) return;
      box.classList.remove("raw-mode");
      box.innerHTML = mdRender(text || "");
    }

    function updateAutoRefreshButton() {
      const btn = $("btnAutoRefresh");
      if (!btn) return;
      if (state.autoRefreshMode === "manual") {
        btn.textContent = "AUTO_REFRESH: MANUAL";
      } else if (state.autoRefreshMode === "auto") {
        btn.textContent = "AUTO_REFRESH: ACTIVE RUN";
      } else {
        btn.textContent = "AUTO_REFRESH: 5s";
      }
    }

    function startAutoRefresh(mode = "manual") {
      if (state.autoRefreshTimer) clearInterval(state.autoRefreshTimer);
      state.autoRefreshMode = mode;
      state.autoRefreshTimer = setInterval(() => loadRuns(), 3000);
      updateAutoRefreshButton();
    }

    function stopAutoRefresh() {
      if (state.autoRefreshTimer) clearInterval(state.autoRefreshTimer);
      state.autoRefreshTimer = null;
      state.autoRefreshMode = "off";
      updateAutoRefreshButton();
    }

    function ensureRunAutoRefresh(hasActiveRuns) {
      if (hasActiveRuns) {
        if (!state.autoRefreshTimer || state.autoRefreshMode === "off") {
          startAutoRefresh("auto");
        }
        return;
      }
      if (state.autoRefreshMode === "auto") {
        stopAutoRefresh();
      }
    }

    function toggleAutoRefresh() {
      if (state.autoRefreshTimer) {
        stopAutoRefresh();
      } else {
        startAutoRefresh("manual");
      }
    }

    // ========== Writeback Review ==========
    async function loadDrafts() {
      if (!state.selectedProject) return;
      try {
        const data = await api(`/api/projects/${enc(state.selectedProject)}/drafts`);
        renderDraftTable(data.drafts || []);
      } catch(e) { $("draftTable").innerHTML = `<div class="muted">${e.message}</div>`; }
    }

    function renderDraftTable(drafts) {
      if (!drafts.length) {
        $("draftTable").innerHTML = "<div class='muted'>暂无导出草案。只有在你显式走正式写回流程时，这里才会出现内容。</div>";
        return;
      }
      $("draftTable").innerHTML = `<table>
        <thead><tr><th>name</th><th>status</th><th>reviewer</th><th>case</th><th></th></tr></thead>
        <tbody>${drafts.map(d => `
          <tr>
            <td style="font-size: 14px">${d.name}</td>
            <td><span class="status ${d.review_status}">${d.review_status}</span></td>
            <td>${d.reviewed_by || "-"}</td>
            <td>${d.case_name || "-"}</td>
            <td><button class="ghost small" data-action="selectDraft" data-draft-name="${escAttr(d.name)}">选择</button></td>
          </tr>
        `).join("")}</tbody></table>`;
    }

    function selectDraft(name) {
      $("draftName").value = name;
    }

    async function previewDraft() {
      const name = $("draftName").value;
      if (!state.selectedProject || !name) return;
      try {
        const data = await api(`/api/projects/${enc(state.selectedProject)}/drafts/${enc(name)}`);
        let preview = "--- FRONTMATTER ---\\n";
        preview += JSON.stringify(data.frontmatter, null, 2);
        preview += "\\n\\n--- BODY ---\\n";
        preview += data.body;
        if (data.truncated) preview += `\\n\\n[TRUNCATED — showing first 512 KB of ${(data.size_bytes/1024).toFixed(0)} KB total]`;
        $("draftPreviewContent").textContent = preview;
      } catch(e) { $("draftPreviewContent").textContent = `Error: ${e.message}`; }
    }

    // ========== Utility ==========

    // withLoading: wraps an async action with button loading state + inline feedback.
    // btn     - the button element
    // label   - original button label to restore after
    // asyncFn - async function to call; may return a string summary to flash on btn
    // onError - optional override; default shows message in red on the button area
    // ========== Markdown Renderer ==========
    @@MD_RENDER_JS@@

    async function withLoading(btn, label, asyncFn) {
      btn.disabled = true;
      btn.textContent = "加载中…";
      // Safety net: unconditionally re-enable after 20 s so a hanging request
      // never leaves the UI permanently locked.
      const safetyId = setTimeout(() => {
        btn.textContent = label;
        btn.style.color = "";
        btn.disabled = false;
        const s = $("sidebarStatus");
        if (s) { s.textContent = `${label}：请求超时，请重试`; s.style.color = "var(--bad, #f87171)"; }
      }, 20000);
      try {
        const msg = await asyncFn();
        clearTimeout(safetyId);
        btn.textContent = msg ? `✓ ${msg}` : `✓ ${label}`;
        setTimeout(() => { btn.textContent = label; btn.disabled = false; }, 1800);
      } catch(e) {
        clearTimeout(safetyId);
        btn.textContent = `✗ 出错`;
        btn.style.color = "var(--bad, #f87171)";
        setTimeout(() => {
          btn.textContent = label;
          btn.style.color = "";
          btn.disabled = false;
        }, 3000);
        // Also surface the message in a small area below (caller may override)
        const statusEl = $("sidebarStatus");
        if (statusEl) { statusEl.textContent = `错误：${e.message}`; statusEl.style.color = "var(--bad, #f87171)"; }
        console.error(label, e);
      }
    }

    // Flash a transient status message in the sidebar status bar.
    function sidebarMsg(msg, isError = false) {
      const el = $("sidebarStatus");
      if (!el) return;
      el.textContent = msg;
      el.style.color = isError ? "var(--bad, #f87171)" : "var(--brand)";
      setTimeout(() => { el.textContent = ""; el.style.color = ""; }, 3500);
    }

    function enc(s) { return encodeURIComponent(s); }
    function escHtml(s) { return String(s ?? "").replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;"); }
    function escAttr(s) { return String(s ?? "").replace(/&/g,"&amp;").replace(/"/g,"&quot;").replace(/</g,"&lt;").replace(/>/g,"&gt;"); }

    function linesToList(text) {
      return (text || "").split("\\n").map(x => x.trim()).filter(Boolean);
    }

    // ========== Explore right pane: card preview + constraint tabs ==========
    function switchExploreRightTab(tab) {
      const cardPreview = $("exploreCardPreview");
      const constraintBox = $("exploreConstraintBox");
      const placeholder = $("exploreRightPlaceholder");
      const tabCard = $("exploreTabCard");
      const tabConstraint = $("exploreTabConstraint");
      cardPreview.style.display = "none";
      constraintBox.style.display = "none";
      placeholder.style.display = "none";
      tabCard.style.opacity = "0.5";
      tabConstraint.style.opacity = "0.5";
      if (tab === "card") {
        cardPreview.style.display = "";
        tabCard.style.opacity = "1";
      } else if (tab === "constraint") {
        constraintBox.style.display = "";
        tabConstraint.style.opacity = "1";
      } else {
        placeholder.style.display = "";
      }
    }
    window.switchExploreRightTab = switchExploreRightTab;

    async function previewExploreCard(cardPath, cardName) {
      const titleEl = $("exploreCardPreviewTitle");
      const bodyEl = $("exploreCardPreviewBody");
      titleEl.textContent = cardName || cardPath;
      titleEl.dataset.path = cardPath;
      bodyEl.innerHTML = '<span class="muted">加载中…</span>';
      switchExploreRightTab("card");
      // Highlight selected card
      document.querySelectorAll("#exploreCardList [data-action=previewExploreCard]").forEach(el => {
        el.style.borderColor = el.dataset.cardPath === cardPath ? "var(--brand)" : "var(--line)";
      });
      try {
        const data = await api(`/api/vault/card/${enc(cardPath)}`);
        const content = data.content || "(空)";
        const truncNote = data.truncated
          ? `\n\n> [内容已截断 — 显示前 512 KB，文件共 ${(data.size_bytes/1024).toFixed(0)} KB]`
          : "";
        bodyEl.innerHTML = mdRender(content + truncNote);
        if (window.MathJax && typeof MathJax.typesetPromise === "function") {
          setTimeout(() => MathJax.typesetPromise([bodyEl]).catch(() => {}), 0);
        }
      } catch(e) {
        bodyEl.innerHTML = `<span style="color:#f87171">读取失败：${escHtml(e.message)}</span>`;
      }
    }

    // ========== Event delegation for dynamically rendered tables ==========
    // Replaces all inline onclick= patterns; handles data-action attributes set on
    // table rows, spans, and buttons rendered by renderRoundTable / renderRunTable /
    // loadCases / renderDraftTable / card search results / viewArtifact.
    document.addEventListener("click", (e) => {
      const el = e.target.closest("[data-action]");
      if (!el) return;
      const action = el.dataset.action;
      if (action === "selectCase") {
        selectCase(el.dataset.caseName);
      } else if (action === "viewArtifact") {
        viewArtifact(el.dataset.runId, el.dataset.artifactKey);
      } else if (action === "summarizeRun") {
        summarizeRun(el.dataset.runId);
      } else if (action === "selectDraft") {
        selectDraft(el.dataset.draftName);
      } else if (action === "previewExploreCard") {
        previewExploreCard(el.dataset.cardPath, el.dataset.cardName);
      } else if (action === "selectCard") {
        $("cardViewName").value = el.dataset.cardPath;
        $("btnReadCard").click();
      } else if (action === "deleteRun") {
        deleteRun(el.dataset.runId, el.dataset.caseName);
      } else if (action === "writebackRun") {
        writebackRun(el.dataset.runId);
      } else if (action === "copyArtifact") {
        const pre = $("artifactText");
        const text = state.artifactRawText || (pre ? pre.textContent : "");
        if (text) navigator.clipboard.writeText(text);
      }
    });

    // ========== Init ==========
    async function init() {
      // Navigation
      for (const button of document.querySelectorAll(".nav button")) {
        button.addEventListener("click", () => switchView(button.dataset.view));
      }
      // Keyboard shortcuts: press 0-3 to switch views (when not in input field)
      document.addEventListener("keydown", (e) => {
        if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA" || e.target.tagName === "SELECT") return;
        const viewMap = {"0":"dashboard","1":"knowledge","2":"bridge","3":"writeback"};
        if (viewMap[e.key]) switchView(viewMap[e.key]);
      });
      $("projectSelect").addEventListener("change", async (e) => {
        state.selectedProject = e.target.value;
        await loadProjectDetail();
        if (state.view === "bridge") { loadCases(); loadRuns(); }
        if (state.view === "writeback") loadDrafts();
      });
      $("reloadProjectBtn").addEventListener("click", (e) => {
        withLoading(e.currentTarget, "重新加载项目", async () => {
          await loadProjectDetail();
          return state.selectedProject ? `已加载 ${state.selectedProject}` : "请先选择项目";
        });
      });
      $("refreshProjectBtn").addEventListener("click", (e) => {
        withLoading(e.currentTarget, "刷新上下文包", async () => {
          if (!state.selectedProject) { sidebarMsg("请先选择项目", true); return; }
          const data = await api(`/api/projects/${enc(state.selectedProject)}/refresh`, "POST", {});
          await loadProjectDetail();
          sidebarMsg(`上下文包已刷新：${state.selectedProject}`);
          showResponse("bridgeResponseBox", data);
        });
      });
      $("btnOpenCreateProject").addEventListener("click", () => openCreateProjectModal());
      $("btnOpenCreateProject2").addEventListener("click", () => openCreateProjectModal());
      $("btnCloseCreateProject").addEventListener("click", () => closeCreateProjectModal());
      $("createProjectModal").addEventListener("click", (e) => {
        if (e.target.id === "createProjectModal") closeCreateProjectModal();
      });

      // Knowledge Ops
      $("btnRefreshVaultStats").addEventListener("click", (e) => {
        withLoading(e.currentTarget, "刷新", async () => { await loadVaultStats(); });
      });
      $("btnRefreshInbox").addEventListener("click", (e) => {
        withLoading(e.currentTarget, "刷新", async () => { await loadInbox(); });
      });
      $("btnSearchCards").addEventListener("click", (e) => {
        withLoading(e.currentTarget, "搜索", async () => {
          const q = enc($("cardQuery").value || "");
          const limit = enc($("cardLimit").value || "30");
          const data = await api(`/api/cards/search?q=${q}&limit=${limit}`);
          const rows = data.cards || [];
          if (!rows.length) {
            $("cardResults").innerHTML = "<div class='muted'>未找到匹配的卡片。</div>";
            return "无结果";
          } {
            $("cardResults").innerHTML = `<table>
              <thead><tr><th>名称</th><th>类型</th><th>领域</th><th>生命周期</th></tr></thead>
              <tbody>${rows.map(r => `
                <tr>
                  <td><span class="artifact-link" data-action="selectCard" data-card-path="${escAttr(r.path||r.name||"")}">${escHtml(r.name||"")}</span></td>
                  <td>${r.type||""}</td>
                  <td>${r.domain||""}</td>
                  <td>${r.lifecycle||""}</td>
                </tr>
              `).join("")}</tbody></table>`;
            return `找到 ${rows.length} 张卡片`;
          }
        });
      });
      let _cardRawText = "";
      function _renderCard() {
        const box = $("cardContent");
        const rendered = $("cardRenderToggle").checked;
        if (rendered) {
          box.classList.remove("raw-mode");
          box.innerHTML = mdRender(_cardRawText);
          // Defer MathJax: run after withLoading resolves and button re-enables,
          // so a slow typeset doesn't freeze the UI or block the async chain.
          // Guard with typeof so we don't throw if CDN hasn't loaded yet.
          setTimeout(function() {
            try {
              if (window.MathJax && typeof MathJax.typesetPromise === "function") {
                MathJax.typesetPromise([box]).catch(function(e) { console.warn("MathJax:", e); });
              }
            } catch(e) { console.warn("MathJax setup error:", e); }
          }, 0);
        } else {
          box.classList.add("raw-mode");
          box.textContent = _cardRawText;
        }
      }
      $("cardRenderToggle").addEventListener("change", _renderCard);
      $("btnReadCard").addEventListener("click", (e) => {
        withLoading(e.currentTarget, "读取", async () => {
          const name = $("cardViewName").value.trim();
          if (!name) { $("cardContent").textContent = "请先输入卡片文件名。"; return; }
          const data = await api(`/api/vault/card/${enc(name)}`);
          const truncNote = data.truncated
            ? `\n\n> [内容已截断 — 显示前 512 KB，文件共 ${(data.size_bytes/1024).toFixed(0)} KB]`
            : "";
          _cardRawText = (data.content || "(空)") + truncNote;
          const meta = $("cardContentMeta");
          if (meta) meta.textContent = data.truncated ? `⚠ 已截断 (${(data.size_bytes/1024).toFixed(0)} KB)` : `${(data.size_bytes/1024).toFixed(1)} KB`;
          _renderCard();
        });
      });

      // Knowledge Ops — Graph Coverage Matrix
      $("btnLoadGraphCoverage").addEventListener("click", (e) => {
        withLoading(e.currentTarget, "加载", async () => {
          const data = await api("/api/vault/graph/coverage");
          const el = $("graphCoverageResult");
          if (!data.ok) {
            el.innerHTML = `<div class="muted">Graph 不可用：${escHtml(data.error || "未知错误")}</div>`;
            return "加载失败";
          }
          const matrix = data.matrix || {};
          const stats = data.stats || {};
          const coverage = data.coverage || {};
          const domainCoverage = data.domain_coverage || {};
          let html = "";
          // Stats bar
          const orphanCount = (stats.orphan_nodes || []).length;
          html += `<div style="display:flex;gap:16px;flex-wrap:wrap;margin-bottom:10px;font-size: 14px">`;
          html += `<span>节点数：<strong>${escHtml(String(stats.node_count||0))}</strong></span>`;
          html += `<span>边数：<strong>${escHtml(String(stats.edge_count||0))}</strong></span>`;
          html += `<span>孤立节点：<strong style="color:${orphanCount>0?'#f87171':'#34d399'}">${escHtml(String(orphanCount))}</strong></span>`;
          html += `<span>悬空边：<strong style="color:${(stats.dangling_edge_count||0)>0?'#fbbf24':'#34d399'}">${escHtml(String(stats.dangling_edge_count||0))}</strong></span>`;
          if (coverage.factor) {
            html += `<span>Factor 标注：<strong>${escHtml(String(coverage.factor.annotated||0))}/${escHtml(String(coverage.factor.total||0))}</strong></span>`;
          }
          html += `</div>`;

          const isFactorCategory = state.currentCategoryKey === "factor_recipe";

          if (isFactorCategory) {
            // Factor recipe: mechanism × family matrix
            const families = [...new Set(Object.values(matrix).flatMap(m => Object.keys(m)))].sort();
            const mechanisms = Object.keys(matrix).sort();
            if (mechanisms.length > 0 && families.length > 0) {
              html += `<h3 style="margin:8px 0 6px 0;font-size: 15px">Mechanism × Family 矩阵</h3>`;
              html += `<div style="overflow-x:auto"><table style="font-size: 14px;border-collapse:collapse">`;
              html += `<thead><tr><th style="text-align:left;padding:4px 10px 4px 4px;border-bottom:2px solid var(--line)">mechanism \\ family</th>`;
              for (const fam of families) {
                html += `<th style="padding:4px 8px;border-bottom:2px solid var(--line);text-align:center">${escHtml(fam)}</th>`;
              }
              html += `</tr></thead><tbody>`;
              for (const mech of mechanisms) {
                html += `<tr><td style="padding:4px 4px;font-weight:bold;white-space:nowrap">${escHtml(mech)}</td>`;
                for (const fam of families) {
                  const count = (matrix[mech] || {})[fam] || 0;
                  const bg = count === 0 ? "var(--panel-hover)" : count >= 3 ? "var(--bad-soft)" : "var(--ok-soft)";
                  const color = count === 0 ? "var(--muted)" : count >= 3 ? "var(--bad)" : "var(--ok)";
                  html += `<td style="text-align:center;padding:4px 8px;background:${bg};color:${color};font-weight:bold;border:1px solid var(--line)">${count === 0 ? "·" : count}</td>`;
                }
                html += `</tr>`;
              }
              html += `</tbody></table></div>`;
              html += `<div class="muted" style="font-size: 13px;margin-top:4px">绿=已有 | 红>=3=拥挤 | ·=空白方向（研究机会）</div>`;
            } else {
              html += `<div class="muted">暂无 mechanism/family 数据。请先在 Factor 卡片 frontmatter 中补充 mechanism 和 factor_family 字段。</div>`;
            }
          } else {
            // Other categories: domain × type coverage matrix
            const domains = Object.keys(domainCoverage).sort();
            const allTypes = [...new Set(domains.flatMap(d => Object.keys(domainCoverage[d])))].sort();
            if (domains.length > 0 && allTypes.length > 0) {
              html += `<h3 style="margin:8px 0 6px 0;font-size: 15px">Domain × Type 知识分布</h3>`;
              html += `<div style="overflow-x:auto"><table style="font-size: 14px;border-collapse:collapse">`;
              html += `<thead><tr><th style="text-align:left;padding:4px 10px 4px 4px;border-bottom:2px solid var(--line)">domain \\ type</th>`;
              for (const t of allTypes) {
                html += `<th style="padding:4px 8px;border-bottom:2px solid var(--line);text-align:center">${escHtml(t)}</th>`;
              }
              html += `</tr></thead><tbody>`;
              for (const d of domains) {
                html += `<tr><td style="padding:4px 4px;font-weight:bold;white-space:nowrap">${escHtml(d)}</td>`;
                for (const t of allTypes) {
                  const count = (domainCoverage[d] || {})[t] || 0;
                  const bg = count === 0 ? "var(--panel-hover)" : "var(--ok-soft)";
                  const color = count === 0 ? "var(--muted)" : "var(--ok)";
                  html += `<td style="text-align:center;padding:4px 8px;background:${bg};color:${color};font-weight:bold;border:1px solid var(--line)">${count === 0 ? "·" : count}</td>`;
                }
                html += `</tr>`;
              }
              html += `</tbody></table></div>`;
              html += `<div class="muted" style="font-size: 13px;margin-top:4px">显示知识库中各 domain 下各类型卡片数量</div>`;
            } else {
              html += `<div class="muted">暂无 domain/type 数据。</div>`;
            }
          }

          // Type coverage summary
          const typeKeys = Object.keys(coverage).sort();
          if (typeKeys.length > 0) {
            html += `<h3 style="margin:12px 0 6px 0;font-size: 15px">Type 覆盖汇总</h3>`;
            html += `<div style="display:flex;gap:12px;flex-wrap:wrap;font-size: 14px">`;
            for (const tk of typeKeys) {
              const c = coverage[tk];
              html += `<span>${escHtml(tk)}: <strong>${c.annotated||0}</strong>/${c.total||0} 已标注</span>`;
            }
            html += `</div>`;
          }

          el.innerHTML = html;
          return "已加载";
        });
      });

      // Bridge Workspace — Preflight Check
      $("btnRunPreflight").addEventListener("click", (e) => {
        withLoading(e.currentTarget, "运行预检", async () => {
          const isFactorRecipe = state.currentCategoryKey === "factor_recipe";
          const v = (id) => ($(id) || {value: ""}).value;
          const payload = {
            candidate_name: v("pfCandidateName").trim(),
            candidate_family: isFactorRecipe ? v("pfFamily") : "",
            candidate_mechanism: isFactorRecipe ? v("pfMechanism") : "",
            candidate_pit_sensitivity: isFactorRecipe ? v("pfPitSensitivity") : "",
            candidate_decay_class: isFactorRecipe ? v("pfDecayClass") : "",
            candidate_capacity_class: isFactorRecipe ? v("pfCapacityClass") : "",
            candidate_similar: isFactorRecipe ? v("pfSimilar").split(",").map(s=>s.trim()).filter(Boolean) : [],
            candidate_uses_data: isFactorRecipe ? v("pfUsesData").split(",").map(s=>s.trim()).filter(Boolean) : [],
            checked_card_paths: v("pfCheckedCards").split("\\n").map(s=>s.trim()).filter(Boolean),
            category: state.currentCategoryKey,
          };
          if (!payload.candidate_name) { sidebarMsg("请填写候选名称", true); return; }
          const data = await api("/api/vault/preflight", "POST", payload);
          const el = $("preflightResult");
          el.style.display = "";
          const severityIcon = s => s === "error" ? "🔴" : s === "warning" ? "🟡" : "🟢";
          const blocked = data.is_blocked;
          let html = `<div style="padding:8px 12px;border-radius:6px;font-weight:bold;margin-bottom:10px;background:${blocked?"rgba(248,113,113,0.1)":"rgba(52,211,153,0.1)"};color:${blocked?"#f87171":"#34d399"}">`;
          html += blocked ? "BLOCKED — 存在 error 级别问题" : (data.issues && data.issues.length ? "PASS (warnings)" : "ALL CLEAR");
          html += `</div>`;
          if (data.issues && data.issues.length) {
            html += `<div style="margin-bottom:8px">`;
            for (const issue of data.issues) {
              html += `<div style="padding:5px 8px;margin-bottom:4px;border-left:3px solid ${issue.severity==="error"?"#f87171":"#fbbf24"};background:#f8fafc;font-size: 15px">`;
              html += `<code style="font-size: 14px;color:var(--brand)">${escHtml(issue.code)}</code> — ${escHtml(issue.message)}`;
              html += `</div>`;
            }
            html += `</div>`;
          }
          if (data.novelty && (data.novelty.similar_existing||[]).length) {
            html += `<div style="font-size: 14px;color:var(--ink);margin-bottom:6px"><strong>相似因子：</strong>${escHtml((data.novelty.similar_existing||[]).join(", "))}</div>`;
          }
          if (data.novelty && (data.novelty.same_mechanism_family||[]).length) {
            html += `<div style="font-size: 14px;color:var(--ink);margin-bottom:6px"><strong>同机制+族因子：</strong>${escHtml((data.novelty.same_mechanism_family||[]).join(", "))}</div>`;
          }
          if (data.checked_cards && data.checked_cards.length) {
            html += `<div class="muted" style="font-size: 13px">已检查卡片：${escHtml(data.checked_cards.join(", "))}</div>`;
          }
          el.innerHTML = html;
          return blocked ? "预检：已阻断" : `预检：${data.issues ? data.issues.length : 0} 个问题`;
        });
      });

      // Bridge Workspace — Idea Explorer
      $("btnExploreIdea").addEventListener("click", (e) => {
        withLoading(e.currentTarget, "探索", async () => {
          const idea = $("exploreIdea").value.trim();
          if (!idea) { sidebarMsg("请输入想法", true); return; }
          const mode = document.querySelector('input[name="exploreMode"]:checked').value;
          const data = await api("/api/vault/explore-idea", "POST", { idea, mode, project_slug: state.selectedProject || null });

          // Render card list
          const cards = data.related_cards || [];
          const cardListEl = $("exploreCardList");
          if (cards.length === 0) {
            cardListEl.innerHTML = '<span class="muted">未找到相关卡片。建议在 CARD-INDEX.tsv 中完善 tags 字段。</span>';
          } else {
            const reasonLabel = {
              semantic_match: "语义匹配", similar_to: "相似卡片", depends_on: "依赖",
              reverse_dependency: "被依赖", same_family: "同族", same_mechanism: "同机制",
              project_context: "项目上下文",
            };
            const lifecycleColor = lc => {
              if (!lc) return "";
              const m = {validated:"#34d399",production:"#34d399",live:"#34d399",deployed:"#34d399",
                         theoretical:"#64748b",candidate:"#22d3ee",active:"#22d3ee",deprecated:"#f87171",retired:"#f87171"};
              return m[lc.toLowerCase()] || "#64748b";
            };
            cardListEl.innerHTML = cards.map(c => {
              const name = c.name || c.path || "?";
              const reasons = (c.reasons || []).filter(Boolean);
              const summary = c.summary || "";
              const snippet = c.snippet || "";
              const showSnippet = snippet && snippet !== summary;
              // Meta line: type badge + lifecycle badge + mechanism + family
              let metaHtml = "";
              if (c.type) metaHtml += `<span style="background:rgba(34,211,238,0.08);padding:1px 5px;border-radius:3px;font-size: 12px;color:var(--brand)">${escHtml(c.type)}</span> `;
              if (c.lifecycle) {
                const lcc = lifecycleColor(c.lifecycle);
                metaHtml += `<span style="color:${lcc};font-size: 13px">${escHtml(c.lifecycle)}</span> `;
              }
              if (c.mechanism) metaHtml += `<span class="muted" style="font-size: 13px">机制:${escHtml(c.mechanism)}</span> `;
              if (c.factor_family) metaHtml += `<span class="muted" style="font-size: 13px">族:${escHtml(c.factor_family)}</span> `;
              // Reason badges
              let reasonHtml = "";
              if (reasons.length) {
                reasonHtml = reasons.map(r => {
                  const label = reasonLabel[r] || r;
                  const bg = r === "semantic_match" ? "rgba(34,211,238,0.1)" : r === "project_context" ? "rgba(139,92,246,0.1)" : "rgba(52,211,153,0.1)";
                  const fg = r === "semantic_match" ? "#22d3ee" : r === "project_context" ? "#a78bfa" : "#34d399";
                  return `<span style="display:inline-block;background:${bg};color:${fg};padding:1px 5px;border-radius:3px;font-size: 12px;margin-right:3px">${escHtml(label)}</span>`;
                }).join("");
              }
              return `<div style="margin-bottom:7px;padding:7px 10px;background:#f8fafc;border:1px solid var(--line);border-radius:6px;cursor:pointer"
                           data-action="previewExploreCard" data-card-path="${escAttr(c.path || name)}" data-card-name="${escAttr(name)}">
                <div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap">
                  <strong style="font-size: 15px">${escHtml(name)}</strong>
                  ${reasonHtml}
                </div>
                <div style="margin:3px 0 0 0">${metaHtml}</div>
                ${summary ? `<div style="font-size: 14px;color:var(--ink);margin-top:3px">${escHtml(summary)}</div>` : ""}
                ${showSnippet ? `<details style="margin-top:2px"><summary style="font-size: 13px;color:var(--muted);cursor:pointer">展开原文摘录</summary><div style="font-size: 13px;color:var(--muted);margin-top:2px">${escHtml(snippet)}</div></details>` : ""}
              </div>`;
            }).join("");
          }

          // Setup right pane: always visible, with constraint tab in constrained mode
          const rightPane = $("exploreRightPane");
          rightPane.style.display = "";
          const constraintTab = $("exploreTabConstraint");
          const cr = data.constraint_report || {};
          if (mode === "constrained") {
            constraintTab.style.display = "";
            let html = "";
            if (cr.primary_family) html += `<div>推断因子族：<strong>${escHtml(cr.primary_family)}</strong></div>`;
            if (cr.primary_mechanism) html += `<div>推断机制：<strong>${escHtml(cr.primary_mechanism)}</strong></div>`;
            if (cr.crowding_warning) {
              html += `<div style="color:#f87171;margin-top:5px">${escHtml(cr.crowding_warning)}</div>`;
            }
            const peers = cr.validated_peers || [];
            if (peers.length > 0) {
              html += `<div style="margin-top:6px;font-size: 14px"><strong>同族已验证因子</strong>（${peers.length}）：`;
              html += `<span class="muted">${peers.map(p => escHtml(p)).join("、")}</span></div>`;
            }
            if (cr.family_counts && Object.keys(cr.family_counts).length > 0) {
              const fc = Object.entries(cr.family_counts).map(([k, v]) => `${escHtml(k)}(${v})`).join(", ");
              html += `<div class="muted" style="font-size: 13px;margin-top:5px">现有因子族分布：${fc}</div>`;
            }
            const nw = cr.novelty_warnings || [];
            if (nw.length > 0) {
              html += `<div style="margin-top:6px">`;
              for (const w of nw) {
                html += `<div style="font-size: 14px;color:#fbbf24">${escHtml(w)}</div>`;
              }
              html += `</div>`;
            }
            const frontier = cr.frontier_matches || [];
            if (frontier.length > 0) {
              html += `<div style="margin-top:8px;padding-top:6px;border-top:1px solid var(--line)">`;
              html += `<strong style="font-size: 14px">探索前沿方向</strong>`;
              for (const f of frontier) {
                const prioColor = f.priority === "high" ? "#34d399" : f.priority === "medium" ? "#fbbf24" : "#64748b";
                html += `<div style="margin-top:4px;padding:5px 8px;background:#f8fafc;border-radius:4px;font-size: 14px">`;
                html += `<div><strong>${escHtml(f.direction)}</strong> <span style="color:${prioColor};font-size: 13px">[${escHtml(f.priority || "?")}]</span></div>`;
                html += `<div class="muted" style="font-size: 13px">${escHtml(f.reason)}</div>`;
                if (f.factor_family || f.mechanism) {
                  html += `<div class="muted" style="font-size: 12px">${[f.factor_family, f.mechanism].filter(Boolean).join(" / ")}`;
                  if (f.suggested_by) html += ` · suggested by: ${escHtml(f.suggested_by)}`;
                  html += `</div>`;
                }
                html += `</div>`;
              }
              html += `</div>`;
            }
            const failures = cr.failure_refs || [];
            if (failures.length > 0) {
              html += `<div style="margin-top:8px;padding-top:6px;border-top:1px solid var(--line)">`;
              html += `<strong style="font-size: 14px;color:#f87171">相关失败案例</strong>`;
              for (const f of failures) {
                html += `<div style="margin-top:4px;padding:5px 8px;background:rgba(248,113,113,0.06);border-left:3px solid #f87171;border-radius:0 4px 4px 0;font-size: 14px">`;
                html += `<div><code style="font-size: 13px">${escHtml(f.failure_id)}</code> <strong>${escHtml(f.title)}</strong>`;
                if (f.status) html += ` <span class="muted" style="font-size: 13px">[${escHtml(f.status)}]</span>`;
                html += `</div>`;
                if (f.failure_statement) {
                  html += `<div class="muted" style="font-size: 13px;margin-top:2px">${escHtml(f.failure_statement)}</div>`;
                }
                html += `</div>`;
              }
              html += `</div>`;
            }
            if (!html) html = '<span class="muted">（未找到 mechanism/factor_family 字段，建议完善卡片 frontmatter）</span>';
            $("exploreConstraintReport").innerHTML = html;
            // Default to constraint tab in constrained mode
            switchExploreRightTab("constraint");
          } else {
            constraintTab.style.display = "none";
            // In non-constrained mode, show placeholder until a card is clicked
            switchExploreRightTab("placeholder");
          }

          // Show prompt + copy button
          $("explorePromptBox").textContent = data.gpt_prompt || "";
          $("exploreResults").style.display = "";
          $("btnCopyExplorePrompt").style.display = "";
          return `找到 ${cards.length} 张相关卡片`;
        });
      });
      $("btnCopyExplorePrompt").addEventListener("click", (e) => copyText("explorePromptBox", e.currentTarget));

      // Bridge Workspace
      const bind = (id, evt, fn) => { const el = $(id); if (el) el.addEventListener(evt, fn); };

      bind("btnCreateProject", "click", (e) => {
        withLoading(e.currentTarget, "新建项目", async () => {
          const payload = {
            slug: ($("createSlug") || {}).value || "",
            title_zh: ($("createTitle") || {}).value || "",
            category: ($("createCategory") || {}).value || "factor_recipe",
            owner: ($("createOwner") || {value: "yukun"}).value,
            market: ($("createMarket") || {value: "ashare"}).value,
            frequency: ($("createFrequency") || {value: "daily"}).value,
            chatgpt_project_name: ($("createChatgptName") || {}).value || "",
            origin_cards: linesToList(($("createOriginCards") || {}).value || ""),
          };
          if (!payload.slug) throw new Error("请输入项目 slug");
          const data = await api("/api/projects", "POST", payload);
          showResponse("bridgeResponseBox", data);
          state.selectedProject = payload.slug;
          await loadProjects();
          await loadProjectDetail();
          closeCreateProjectModal();
          return `已创建：${payload.slug}`;
        });
      });
      bind("btnPatchProject", "click", (e) => {
        withLoading(e.currentTarget, "保存状态", async () => {
          if (!state.selectedProject) { sidebarMsg("请先选择项目", true); return; }
          const payload = {
            current_hypothesis: ($("patchHypothesis") || {}).value || "",
            current_focus: ($("patchFocus") || {}).value || "",
            next_action: ($("patchAction") || {}).value || "",
          };
          const data = await api(`/api/projects/${enc(state.selectedProject)}`, "PATCH", payload);
          showResponse("bridgeResponseBox", data);
          await loadProjectDetail();
          return "已保存";
        });
      });
      // Factor Workshop
      bind("btnRegisterFactor", "click", (e) => {
        withLoading(e.currentTarget, "注册因子", async () => {
          const name = ($("cfName") || {}).value || "";
          const code = ($("cfCode") || {}).value || "";
          const description = ($("cfDescription") || {}).value || "";
          if (!name) { showResponse("cfResponseBox", {error: "请填写因子方法名"}); return; }
          if (!code) { showResponse("cfResponseBox", {error: "请填写因子代码"}); return; }
          const data = await api("/api/custom-factors", "POST", {name, code, description});
          showResponse("cfResponseBox", data);
          await loadCustomFactors();
          return `因子已注册：${name}`;
        });
      });
      bind("btnLoadFactorTemplate", "click", () => {
        $("cfCode").value = FACTOR_TEMPLATE;
      });

      bind("btnCreateCase", "click", (e) => {
        withLoading(e.currentTarget, "创建 Case", async () => {
          if (!state.selectedProject) { sidebarMsg("请先选择项目", true); return; }
          const dynamicFields = collectCaseDynamicFields();
          const payload = {
            case_name: ($("caseName") || {}).value || "",
            factor_name: dynamicFields.factor_name || null,
            base_method: dynamicFields.base_method || null,
            ...dynamicFields,
          };
          const data = await api(`/api/projects/${enc(state.selectedProject)}/cases`, "POST", payload);
          showResponse("bridgeResponseBox", data);
          if ($("runCaseName")) $("runCaseName").value = payload.case_name;
          return `Case 已创建：${payload.case_name}`;
        });
      });

      // Validation Console
      $("btnRefreshCases").addEventListener("click", (e) => {
        withLoading(e.currentTarget, "刷新", async () => { await loadCases(); });
      });
      $("btnStartRun").addEventListener("click", (e) => {
        withLoading(e.currentTarget, "启动实验", async () => {
          if (!state.selectedProject) { sidebarMsg("请先选择项目", true); return; }
          const payload = {
            case_name: $("runCaseName").value,
            evaluation_profile: $("runProfile").value,
            output_root_dir: $("runOutputDir").value || null,
            render_report: true,
          };
          const data = await api(`/api/projects/${enc(state.selectedProject)}/runs`, "POST", payload);
          showResponse("validationResponseBox", data);
          await loadRuns();
          startAutoRefresh("auto");
          return `实验已启动：${payload.case_name}`;
        });
      });
      $("btnRefreshRuns").addEventListener("click", (e) => {
        withLoading(e.currentTarget, "刷新", async () => { await loadRuns(); });
      });
      $("btnRefreshProjectDiagnostics").addEventListener("click", (e) => {
        withLoading(e.currentTarget, "刷新", async () => { await loadProjectDiagnostics(); });
      });
      $("btnAutoRefresh").addEventListener("click", toggleAutoRefresh);

      // Writeback Review
      $("btnRefreshDrafts").addEventListener("click", (e) => {
        withLoading(e.currentTarget, "刷新", async () => { await loadDrafts(); });
      });
      $("btnPreviewDraft").addEventListener("click", (e) => {
        withLoading(e.currentTarget, "预览导出草案", async () => { await previewDraft(); });
      });
      $("btnPatchDraft").addEventListener("click", (e) => {
        withLoading(e.currentTarget, "保存导出审阅", async () => {
          if (!state.selectedProject) { sidebarMsg("请先选择项目", true); return; }
          const draftName = $("draftName").value;
          const payload = {
            review_status: $("draftStatus").value,
            reviewed_by: $("draftReviewer").value,
            reviewed_at: $("draftReviewedAt").value,
            one_sentence_verdict: $("draftVerdict").value,
          };
          const data = await api(`/api/projects/${enc(state.selectedProject)}/drafts/${enc(draftName)}`, "PATCH", payload);
          showResponse("writebackResponseBox", data);
          await loadDrafts();
          return "导出审阅已保存";
        });
      });
      $("btnApplyDraft").addEventListener("click", (e) => {
        withLoading(e.currentTarget, "执行正式写回", async () => {
          if (!state.selectedProject) { sidebarMsg("请先选择项目", true); return; }
          const draftName = $("draftName").value;
          const data = await api(`/api/projects/${enc(state.selectedProject)}/drafts/${enc(draftName)}/apply`, "POST", {});
          showResponse("writebackResponseBox", data);
          await loadDrafts();
          return "正式写回已完成";
        });
      });

      // Boot
      await loadCategories();
      await loadProjects();
      await loadEvaluationProfiles();
      await loadCustomFactors();
      await loadDashboard();
      switchView("dashboard");
    }

    init().catch((e) => {
      document.body.innerHTML = `<pre>Boot failed: ${String(e)}</pre>`;
    });
  </script>
</body>
</html>"""
