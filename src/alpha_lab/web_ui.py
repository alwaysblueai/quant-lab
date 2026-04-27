from __future__ import annotations

import datetime
import hashlib
import json
import math
import threading
import traceback
import uuid
import webbrowser
from collections import defaultdict
from collections.abc import Mapping
from csv import DictReader
from dataclasses import dataclass, field, replace
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

import pandas as pd

from alpha_lab.data_adapters.baostock_adapter import (
    generate_real_case_inputs as generate_baostock_real_case_inputs,
)
from alpha_lab.data_store.catalog import DataCatalog
from alpha_lab.data_store.slice_presets import (
    DEFAULT_SLICE_PRESET,
    SLICE_PRESETS,
    SlicePresetConfig,
    get_slice_preset,
    resolve_slice_window,
)
from alpha_lab.data_store.tushare import TushareIngestor
from alpha_lab.exceptions import AlphaLabConfigError, AlphaLabDataError
from alpha_lab.factor_recipe import FactorRecipeError, build_factor_from_recipe_mapping
from alpha_lab.real_cases.single_factor.pipeline import (
    SingleFactorBatchParallelConfig,
    run_single_factor_case,
)
from alpha_lab.real_cases.single_factor.spec import (
    PreprocessSpec,
    dump_spec_yaml,
    load_single_factor_case_spec,
)
from alpha_lab.reporting.renderers import write_case_report
from alpha_lab.research_evaluation_config import (
    AVAILABLE_RESEARCH_EVALUATION_PROFILES,
    DEFAULT_RESEARCH_EVALUATION_CONFIG,
)

RunStatus = Literal["queued", "running", "succeeded", "failed"]
DataSource = Literal["manual", "tushare", "baostock"]
_DATA_SOURCE_BUILD_LOCKS: dict[str, threading.Lock] = {}
_DATA_SOURCE_BUILD_LOCKS_GUARD = threading.Lock()
_FRONTEND_BATCH_WINDOW_SECONDS: float = 0.20
_FRONTEND_BATCH_MAX_WORKERS: int = 4
_FRONTEND_BATCH_FACTORS_PER_WORKER: int = 2


def start_web_ui_server(
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    workspace_root: str | Path | None = None,
    open_browser: bool = True,
) -> None:
    """Start a lightweight local web UI for single-factor real-case runs."""
    root = Path.cwd().resolve() if workspace_root is None else Path(workspace_root).resolve()
    upload_root = root / "dist" / "web_ui_uploads"
    default_output_root = root / "dist" / "web_ui_runs"
    store = _WebRunStore(
        workspace_root=root,
        upload_root=upload_root,
        default_output_root=default_output_root,
    )

    class _Handler(_WebUIRequestHandler):
        run_store = store

    server = ThreadingHTTPServer((host, port), _Handler)
    url = f"http://{host}:{port}/"
    print("")
    print("  Workflow : web-ui")
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
        print("  Workflow : web-ui")
        print("  Status   : stopped")
    finally:
        server.server_close()


@dataclass
class _WebRunRecord:
    run_id: str
    status: RunStatus
    submitted_at_utc: str
    evaluation_profile: str
    data_source: DataSource
    render_report: bool
    output_root_dir: str
    spec_path: str
    data_start_date: str | None = None
    data_end_date: str | None = None
    data_slice_preset: str = DEFAULT_SLICE_PRESET
    data_asset_limit: int | None = None
    started_at_utc: str | None = None
    finished_at_utc: str | None = None
    output_dir: str | None = None
    artifact_paths: dict[str, str] = field(default_factory=dict)
    summary: dict[str, object] = field(default_factory=dict)
    visualization: dict[str, object] = field(default_factory=dict)
    error: str | None = None

    def clone(self) -> _WebRunRecord:
        return _WebRunRecord(
            run_id=self.run_id,
            status=self.status,
            submitted_at_utc=self.submitted_at_utc,
            evaluation_profile=self.evaluation_profile,
            data_source=self.data_source,
            render_report=self.render_report,
            output_root_dir=self.output_root_dir,
            spec_path=self.spec_path,
            data_start_date=self.data_start_date,
            data_end_date=self.data_end_date,
            data_slice_preset=self.data_slice_preset,
            data_asset_limit=self.data_asset_limit,
            started_at_utc=self.started_at_utc,
            finished_at_utc=self.finished_at_utc,
            output_dir=self.output_dir,
            artifact_paths=dict(self.artifact_paths),
            summary=dict(self.summary),
            visualization=dict(self.visualization),
            error=self.error,
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "submitted_at_utc": self.submitted_at_utc,
            "started_at_utc": self.started_at_utc,
            "finished_at_utc": self.finished_at_utc,
            "evaluation_profile": self.evaluation_profile,
            "data_source": self.data_source,
            "data_start_date": self.data_start_date,
            "data_end_date": self.data_end_date,
            "data_slice_preset": self.data_slice_preset,
            "data_asset_limit": self.data_asset_limit,
            "render_report": self.render_report,
            "output_root_dir": self.output_root_dir,
            "spec_path": self.spec_path,
            "output_dir": self.output_dir,
            "artifact_paths": dict(self.artifact_paths),
            "summary": dict(self.summary),
            "visualization": dict(self.visualization),
            "error": self.error,
        }


@dataclass
class _RunTask:
    run_id: str
    spec_text: str
    spec_filename: str
    evaluation_profile: str
    data_source: DataSource
    data_start_date: str | None
    data_end_date: str | None
    data_asset_limit: int | None
    tushare_token: str | None
    render_report: bool
    output_root_dir: str
    data_slice_preset: str = DEFAULT_SLICE_PRESET


@dataclass(frozen=True)
class _FactorRecipeConfig:
    recipe: Mapping[str, object]
    disable_pipeline_preprocess: bool = True


class _WebRunStore:
    def __init__(
        self,
        *,
        workspace_root: Path,
        upload_root: Path,
        default_output_root: Path,
    ) -> None:
        self.workspace_root = workspace_root
        self.upload_root = upload_root
        self.default_output_root = default_output_root
        self.source_cache_root = self.workspace_root / "dist" / "web_ui_source_cache"
        self._records: dict[str, _WebRunRecord] = {}
        self._tasks: dict[str, _RunTask] = {}
        self._lock = threading.Lock()
        self._dispatch_event = threading.Event()
        self._dispatcher = threading.Thread(target=self._dispatch_loop, daemon=True)
        self._dispatcher.start()

    def submit(self, task: _RunTask) -> _WebRunRecord:
        self.upload_root.mkdir(parents=True, exist_ok=True)
        self.default_output_root.mkdir(parents=True, exist_ok=True)
        run_dir = self.upload_root / task.run_id
        run_dir.mkdir(parents=True, exist_ok=False)

        filename = _safe_upload_filename(task.spec_filename or "single_factor_case.yaml")
        spec_path = run_dir / filename
        spec_path.write_text(task.spec_text, encoding="utf-8")

        record = _WebRunRecord(
            run_id=task.run_id,
            status="queued",
            submitted_at_utc=_utc_now_iso(),
            evaluation_profile=task.evaluation_profile,
            data_source=task.data_source,
            data_start_date=task.data_start_date,
            data_end_date=task.data_end_date,
            data_slice_preset=task.data_slice_preset,
            data_asset_limit=task.data_asset_limit,
            render_report=task.render_report,
            output_root_dir=task.output_root_dir,
            spec_path=str(spec_path),
        )
        with self._lock:
            self._records[record.run_id] = record
            self._tasks[record.run_id] = task
        self._dispatch_event.set()
        return record

    def get(self, run_id: str) -> _WebRunRecord | None:
        with self._lock:
            record = self._records.get(run_id)
            if record is None:
                return None
            return record.clone()

    def list_records(self) -> list[_WebRunRecord]:
        with self._lock:
            return [record.clone() for record in self._records.values()]

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
            queued_with_records: list[tuple[_RunTask, _WebRunRecord]] = []
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
        prepared: list[tuple[_RunTask, Path]] = []
        for task in tasks:
            try:
                prepared.append((task, self._prepare_task_spec(task)))
            except Exception as exc:
                self._mark_failed(task.run_id, exc)

        for task, spec_path in prepared:
            self._execute_single_task(task, spec_path)

    def _prepare_task_spec(self, task: _RunTask) -> Path:
        spec_path = Path(self._records[task.run_id].spec_path)
        if task.data_source == "manual":
            return spec_path
        return _prepare_spec_for_data_source(
            task=task,
            original_spec_path=spec_path,
            cache_root=self.source_cache_root,
        )

    def _execute_single_task(self, task: _RunTask, spec_path: Path) -> None:
        try:
            result = run_single_factor_case(
                spec_path,
                output_root_dir=_resolve_web_ui_run_output_root_dir(task),
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
        backtest_result_path = _to_path_or_none(artifact_paths.get("backtest_result_json"))
        metrics_summary = _extract_metrics_summary(
            result.artifact_paths["metrics"],
            backtest_result_path=backtest_result_path,
        )
        visualization = _extract_visualization_payload(artifact_paths)
        with self._lock:
            stored = self._records[run_id]
            stored.status = "succeeded"
            stored.finished_at_utc = _utc_now_iso()
            stored.output_dir = str(result.output_dir)
            stored.artifact_paths = artifact_paths
            stored.summary = metrics_summary
            stored.visualization = visualization
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


class _WebUIRequestHandler(BaseHTTPRequestHandler):
    run_store: _WebRunStore

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_html(_index_html())
            return
        if parsed.path == "/api/profiles":
            self._send_json(
                {
                    "profiles": sorted(AVAILABLE_RESEARCH_EVALUATION_PROFILES),
                    "default_profile": DEFAULT_RESEARCH_EVALUATION_CONFIG.profile_name,
                }
            )
            return
        if parsed.path == "/api/runs":
            self._send_json(
                {
                    "runs": [record.to_payload() for record in self.run_store.list_records()],
                }
            )
            return
        if parsed.path.startswith("/api/runs/") and "/artifact/" in parsed.path:
            self._handle_get_artifact(parsed.path)
            return
        if parsed.path.startswith("/api/runs/"):
            run_id = parsed.path.removeprefix("/api/runs/").strip()
            if not run_id:
                self._send_json({"error": "run_id is required"}, status=HTTPStatus.BAD_REQUEST)
                return
            record = self.run_store.get(run_id)
            if record is None:
                self._send_json(
                    {"error": f"run_id not found: {run_id}"},
                    status=HTTPStatus.NOT_FOUND,
                )
                return
            self._send_json(record.to_payload())
            return

        self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != "/api/runs":
            self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)
            return
        try:
            payload = self._read_json_body()
            task = _parse_run_task(payload)
            record = self.run_store.submit(task)
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return
        self._send_json(record.to_payload(), status=HTTPStatus.CREATED)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003
        return

    def _handle_get_artifact(self, path: str) -> None:
        suffix = path.removeprefix("/api/runs/")
        run_id, _, artifact_key = suffix.partition("/artifact/")
        run_id = run_id.strip()
        artifact_key = artifact_key.strip()
        if not run_id or not artifact_key:
            self._send_json(
                {"error": "run_id and artifact key are required"},
                status=HTTPStatus.BAD_REQUEST,
            )
            return
        record = self.run_store.get(run_id)
        if record is None:
            self._send_json(
                {"error": f"run_id not found: {run_id}"},
                status=HTTPStatus.NOT_FOUND,
            )
            return
        artifact_path = record.artifact_paths.get(artifact_key)
        if not artifact_path:
            self._send_json(
                {"error": f"artifact not found for key: {artifact_key}"},
                status=HTTPStatus.NOT_FOUND,
            )
            return
        path_obj = Path(artifact_path)
        if not path_obj.exists() or not path_obj.is_file():
            self._send_json(
                {"error": f"artifact file not found: {path_obj}"},
                status=HTTPStatus.NOT_FOUND,
            )
            return
        content = path_obj.read_bytes()
        ctype = "application/octet-stream"
        suffix = path_obj.suffix.lower()
        if suffix in {".json"}:
            ctype = "application/json; charset=utf-8"
        elif suffix in {".md", ".txt", ".log", ".csv"}:
            ctype = "text/plain; charset=utf-8"
        elif suffix in {".html", ".htm"}:
            ctype = "text/html; charset=utf-8"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(content)))
        self.send_header(
            "Content-Disposition",
            f'inline; filename="{path_obj.name}"',
        )
        self.end_headers()
        self.wfile.write(content)

    def _read_json_body(self) -> dict[str, object]:
        length_text = self.headers.get("Content-Length", "").strip()
        if not length_text:
            raise AlphaLabDataError("missing Content-Length")
        try:
            length = int(length_text)
        except ValueError as exc:
            raise AlphaLabDataError("invalid Content-Length") from exc
        if length <= 0:
            raise AlphaLabDataError("request body is empty")
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


def _parse_run_task(payload: dict[str, object]) -> _RunTask:
    spec_text = _required_text(payload.get("spec_text"), "spec_text")
    spec_filename = _required_text(payload.get("spec_filename"), "spec_filename")
    eval_profile = _required_text(payload.get("evaluation_profile"), "evaluation_profile")
    if eval_profile not in AVAILABLE_RESEARCH_EVALUATION_PROFILES:
        raise AlphaLabConfigError(
            f"evaluation_profile must be one of {sorted(AVAILABLE_RESEARCH_EVALUATION_PROFILES)}"
        )
    render_report_raw = payload.get("render_report", True)
    if not isinstance(render_report_raw, bool):
        raise AlphaLabConfigError("render_report must be a boolean")

    output_root_raw = payload.get("output_root_dir")
    if output_root_raw is None:
        output_root_dir = "dist/web_ui_runs"
    else:
        output_root_dir = _required_text(output_root_raw, "output_root_dir")

    data_source_raw = payload.get("data_source", "manual")
    data_source_text = _required_text(data_source_raw, "data_source").lower()
    if data_source_text not in {"manual", "tushare", "baostock"}:
        raise ValueError("data_source must be one of ['manual', 'tushare', 'baostock']")
    if data_source_text == "manual":
        data_source: DataSource = "manual"
    elif data_source_text == "tushare":
        data_source = "tushare"
    else:
        data_source = "baostock"

    data_start_date = _optional_date(payload.get("data_start_date"), "data_start_date")
    data_end_date = _optional_date(payload.get("data_end_date"), "data_end_date")
    data_slice_preset = (
        _optional_text(payload.get("data_slice_preset"), "data_slice_preset")
        or DEFAULT_SLICE_PRESET
    )
    if data_slice_preset not in SLICE_PRESETS:
        raise ValueError(f"data_slice_preset must be one of {sorted(SLICE_PRESETS)}")
    data_asset_limit = _optional_positive_int(payload.get("data_asset_limit"), "data_asset_limit")
    tushare_token = _optional_text(payload.get("tushare_token"), "tushare_token")
    if data_source == "tushare" and tushare_token is None:
        raise ValueError("tushare_token is required when data_source=tushare")

    return _RunTask(
        run_id=uuid.uuid4().hex,
        spec_text=spec_text,
        spec_filename=spec_filename,
        evaluation_profile=eval_profile,
        data_source=data_source,
        data_start_date=data_start_date,
        data_end_date=data_end_date,
        data_slice_preset=data_slice_preset,
        data_asset_limit=data_asset_limit,
        tushare_token=tushare_token,
        render_report=render_report_raw,
        output_root_dir=output_root_dir,
    )


def _required_text(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    text = value.strip()
    if not text:
        raise ValueError(f"{field} must be non-empty")
    return text


def _optional_text(value: object, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string when provided")
    token = value.strip()
    return token if token else None


def _optional_positive_int(value: object, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a positive integer when provided")
    if isinstance(value, int):
        out = value
    elif isinstance(value, str):
        token = value.strip()
        if not token:
            return None
        try:
            out = int(token)
        except ValueError as exc:
            raise ValueError(f"{field} must be a positive integer when provided") from exc
    else:
        raise ValueError(f"{field} must be a positive integer when provided")
    if out <= 0:
        raise ValueError(f"{field} must be > 0 when provided")
    return out


def _optional_date(value: object, field: str) -> str | None:
    token = _optional_text(value, field)
    if token is None:
        return None
    try:
        parsed = datetime.date.fromisoformat(token)
    except ValueError as exc:
        raise ValueError(f"{field} must be in YYYY-MM-DD format") from exc
    return parsed.isoformat()


def _safe_upload_filename(filename: str) -> str:
    text = filename.strip().replace("\\", "/").split("/")[-1]
    if not text:
        return "single_factor_case.yaml"
    allowed = []
    for ch in text:
        if ch.isalnum() or ch in {"_", "-", "."}:
            allowed.append(ch)
        else:
            allowed.append("_")
    safe = "".join(allowed)
    if safe in {".", ".."}:
        return "single_factor_case.yaml"
    if "." not in safe:
        return safe + ".yaml"
    return safe


def _resolve_web_ui_run_output_root_dir(task: _RunTask) -> Path:
    return Path(task.output_root_dir).expanduser().resolve() / "_web_runs" / task.run_id


def _prepare_spec_for_data_source(
    *,
    task: _RunTask,
    original_spec_path: Path,
    cache_root: Path | None = None,
) -> Path:
    start_date, end_date, _preset = _resolve_web_data_window(task)
    if task.data_source == "manual":
        return original_spec_path

    raw_spec_mapping = _load_mapping_payload(original_spec_path)
    source_spec = load_single_factor_case_spec(original_spec_path)
    required_factor_csv = _select_factor_csv_for_spec(source_spec.factor_name)
    factor_recipe_config = _extract_factor_recipe_config(raw_spec_mapping)
    factor_recipe = factor_recipe_config.recipe if factor_recipe_config is not None else None
    custom_factor_csv = (
        _factor_csv_filename(source_spec.factor_name)
        if required_factor_csv == "" and factor_recipe is not None
        else ""
    )
    factor_recipe_hash = _stable_hash_mapping(factor_recipe) if factor_recipe is not None else ""

    if cache_root is None:
        input_dir = original_spec_path.parent / f"{task.data_source}_inputs"
        input_dir.mkdir(parents=True, exist_ok=True)
        _generate_source_inputs(
            task=task,
            input_dir=input_dir,
            start_date=start_date,
            end_date=end_date,
            required_factor_csv=required_factor_csv,
        )
        if custom_factor_csv:
            _build_custom_factor_csv(
                input_dir=input_dir,
                factor_recipe=factor_recipe,
                factor_name=source_spec.factor_name,
                factor_csv=custom_factor_csv,
            )
    else:
        cache_key = _build_data_source_cache_key(
            task=task,
            start_date=start_date,
            end_date=end_date,
            factor_name=source_spec.factor_name,
            required_factor_csv=required_factor_csv,
            factor_recipe_hash=factor_recipe_hash,
        )
        input_dir = cache_root / f"{task.data_source}_{cache_key}"
        lock = _get_data_source_build_lock(cache_key)
        with lock:
            if not _source_base_inputs_ready(
                input_dir=input_dir,
                required_factor_csv=required_factor_csv,
            ):
                input_dir.mkdir(parents=True, exist_ok=True)
                _generate_source_inputs(
                    task=task,
                    input_dir=input_dir,
                    start_date=start_date,
                    end_date=end_date,
                    required_factor_csv=required_factor_csv,
                )
            if custom_factor_csv and not (input_dir / custom_factor_csv).exists():
                _build_custom_factor_csv(
                    input_dir=input_dir,
                    factor_recipe=factor_recipe,
                    factor_name=source_spec.factor_name,
                    factor_csv=custom_factor_csv,
                )

    return _rewrite_spec_with_source_inputs(
        original_spec_path=original_spec_path,
        source_input_dir=input_dir,
        custom_factor_csv=custom_factor_csv or None,
        disable_pipeline_preprocess_for_recipe=(
            custom_factor_csv != ""
            and factor_recipe_config is not None
            and factor_recipe_config.disable_pipeline_preprocess
        ),
    )


def _get_data_source_build_lock(cache_key: str) -> threading.Lock:
    with _DATA_SOURCE_BUILD_LOCKS_GUARD:
        lock = _DATA_SOURCE_BUILD_LOCKS.get(cache_key)
        if lock is None:
            lock = threading.Lock()
            _DATA_SOURCE_BUILD_LOCKS[cache_key] = lock
        return lock


def _build_data_source_cache_key(
    *,
    task: _RunTask,
    start_date: str,
    end_date: str,
    factor_name: str,
    required_factor_csv: str,
    factor_recipe_hash: str,
) -> str:
    payload = {
        "source": task.data_source,
        "slice_preset": task.data_slice_preset,
        "start_date": start_date,
        "end_date": end_date,
        "asset_limit": task.data_asset_limit,
        "factor_name": factor_name,
        "required_factor_csv": required_factor_csv,
        "factor_recipe_hash": factor_recipe_hash,
        "tushare_token_hash": (
            hashlib.sha256(task.tushare_token.encode("utf-8")).hexdigest()[:16]
            if task.tushare_token is not None
            else ""
        ),
        "tushare_dataset_version": (
            (_current_tushare_dataset_version_id() or "") if task.data_source == "tushare" else ""
        ),
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def _current_tushare_dataset_version_id() -> str | None:
    version = DataCatalog().get_current_dataset_version(DataCatalog.CORE_DATASET_NAME)
    if version is None:
        return None
    return version.version_id


def _source_inputs_ready(
    *,
    input_dir: Path,
    required_factor_csv: str,
    custom_factor_csv: str,
) -> bool:
    required_files = _required_source_input_filenames(
        required_factor_csv=required_factor_csv,
        custom_factor_csv=custom_factor_csv,
    )
    return all((input_dir / filename).exists() for filename in required_files)


def _source_base_inputs_ready(*, input_dir: Path, required_factor_csv: str) -> bool:
    required_files = _required_source_input_filenames(
        required_factor_csv=required_factor_csv,
        custom_factor_csv="",
    )
    return all((input_dir / filename).exists() for filename in required_files)


def _required_source_input_filenames(
    *,
    required_factor_csv: str,
    custom_factor_csv: str,
) -> tuple[str, ...]:
    required_files = (
        ("prices.csv", "universe.csv")
        if required_factor_csv == ""
        else ("prices.csv", "universe.csv", required_factor_csv)
    )
    if custom_factor_csv:
        required_files = (*required_files, custom_factor_csv)
    return required_files


def _generate_source_inputs(
    *,
    task: _RunTask,
    input_dir: Path,
    start_date: str,
    end_date: str,
    required_factor_csv: str,
) -> None:
    if task.data_source == "tushare":
        requested_factors: tuple[str, ...]
        if required_factor_csv == "bp.csv":
            requested_factors = ("bp",)
        elif required_factor_csv == "roe_ttm.csv":
            requested_factors = ("roe_ttm",)
        else:
            requested_factors = ()
        preset = get_slice_preset(task.data_slice_preset)
        ingestor = TushareIngestor()
        ingestor.ingest_core(
            start_date=start_date,
            end_date=end_date,
            token=task.tushare_token,
            asset_limit=task.data_asset_limit,
        )
        ingestor.export_case_inputs(
            start_date=start_date,
            end_date=end_date,
            output_dir=input_dir,
            asset_limit=task.data_asset_limit,
            factors=requested_factors,
            adjustment=preset.adjustment,
            universe_name=preset.universe_name,
        )
        return
    if task.data_source == "baostock":
        generate_baostock_real_case_inputs(
            output_dir=input_dir,
            start_date=start_date,
            end_date=end_date,
            asset_limit=task.data_asset_limit,
            include_roe=required_factor_csv == "roe_ttm.csv",
        )
        return
    raise ValueError(f"unsupported data_source: {task.data_source!r}")


def _resolve_web_data_window(task: _RunTask) -> tuple[str, str, SlicePresetConfig]:
    return resolve_slice_window(
        preset_name=task.data_slice_preset,
        start_date=task.data_start_date,
        end_date=task.data_end_date,
        fallback_end_date=datetime.date.today().isoformat(),
    )


def _rewrite_spec_with_source_inputs(
    *,
    original_spec_path: Path,
    source_input_dir: Path,
    custom_factor_csv: str | None = None,
    disable_pipeline_preprocess_for_recipe: bool = False,
) -> Path:
    spec = load_single_factor_case_spec(original_spec_path)
    if custom_factor_csv is not None:
        factor_path = source_input_dir / custom_factor_csv
        factor_csv = custom_factor_csv
    else:
        factor_csv = _select_factor_csv_for_spec(spec.factor_name)
        if factor_csv:
            factor_path = source_input_dir / factor_csv
        else:
            factor_path = Path(spec.factor_path)
    prices_path = source_input_dir / "prices.csv"
    universe_path = source_input_dir / "universe.csv"
    if factor_csv and not factor_path.exists():
        raise FileNotFoundError(f"generated factor file not found: {factor_path}")
    if not prices_path.exists():
        raise FileNotFoundError(f"generated prices file not found: {prices_path}")
    if not universe_path.exists():
        raise FileNotFoundError(f"generated universe file not found: {universe_path}")

    updated_universe = replace(spec.universe, path=str(universe_path))
    updated_preprocess = (
        PreprocessSpec(
            winsorize=False,
            winsorize_lower=spec.preprocess.winsorize_lower,
            winsorize_upper=spec.preprocess.winsorize_upper,
            standardization="none",
            min_group_size=spec.preprocess.min_group_size,
            min_coverage=None,
        )
        if disable_pipeline_preprocess_for_recipe
        else spec.preprocess
    )
    updated_spec = replace(
        spec,
        factor_path=str(factor_path),
        prices_path=str(prices_path),
        universe=updated_universe,
        preprocess=updated_preprocess,
    )
    rewritten_path = original_spec_path.parent / "web_ui_source_spec.yaml"
    rewritten_path.write_text(dump_spec_yaml(updated_spec), encoding="utf-8")
    return rewritten_path


def _load_mapping_payload(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    payload: object
    if suffix == ".json":
        payload = json.loads(text)
    elif suffix in {".yml", ".yaml"}:
        payload = _yaml_load(text)
    else:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = _yaml_load(text)
    if not isinstance(payload, dict):
        raise ValueError("spec payload must be a mapping object")
    return payload


def _yaml_load(text: str) -> object:
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - import guard
        raise RuntimeError("PyYAML is required to parse YAML specs") from exc
    return yaml.safe_load(text)


def _extract_factor_recipe_config(
    spec_mapping: Mapping[str, object],
) -> _FactorRecipeConfig | None:
    legacy_recipe_raw = spec_mapping.get("factor_recipe")
    factor_input_raw = spec_mapping.get("factor_input")

    if legacy_recipe_raw is not None and factor_input_raw is not None:
        raise ValueError("use one schema only: factor_recipe (legacy) or factor_input")

    if legacy_recipe_raw is not None:
        if not isinstance(legacy_recipe_raw, Mapping):
            raise ValueError("factor_recipe must be a mapping when provided")
        return _FactorRecipeConfig(recipe=legacy_recipe_raw)

    if factor_input_raw is None:
        return None
    if not isinstance(factor_input_raw, Mapping):
        raise ValueError("factor_input must be a mapping when provided")

    mode_raw = factor_input_raw.get("mode", "recipe")
    if not isinstance(mode_raw, str):
        raise ValueError("factor_input.mode must be a string")
    mode = mode_raw.strip().lower()
    if mode != "recipe":
        return None

    recipe_raw = factor_input_raw.get("recipe")
    if not isinstance(recipe_raw, Mapping):
        raise ValueError("factor_input.recipe must be a mapping when mode=recipe")
    disable_pipeline_preprocess = _optional_bool(
        factor_input_raw.get("disable_pipeline_preprocess"),
        default=True,
        field="factor_input.disable_pipeline_preprocess",
    )
    return _FactorRecipeConfig(
        recipe=recipe_raw,
        disable_pipeline_preprocess=disable_pipeline_preprocess,
    )


def _optional_bool(value: object, *, default: bool, field: str) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    raise ValueError(f"{field} must be a boolean when provided")


def _stable_hash_mapping(mapping: Mapping[str, object]) -> str:
    serialized = json.dumps(mapping, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:20]


def _factor_csv_filename(factor_name: str) -> str:
    token = factor_name.strip().lower()
    safe_chars: list[str] = []
    for ch in token:
        if ch.isalnum() or ch in {"_", "-"}:
            safe_chars.append(ch)
        else:
            safe_chars.append("_")
    safe = "".join(safe_chars).strip("_")
    if not safe:
        safe = "custom_factor"
    return f"{safe}.csv"


def _build_custom_factor_csv(
    *,
    input_dir: Path,
    factor_recipe: Mapping[str, object] | None,
    factor_name: str,
    factor_csv: str,
) -> None:
    if factor_recipe is None:
        raise ValueError(
            "factor_recipe is required for non-builtin factor_name when using auto data_source"
        )
    prices_path = input_dir / "prices.csv"
    if not prices_path.exists():
        raise FileNotFoundError(f"generated prices file not found: {prices_path}")
    prices = pd.read_csv(prices_path)
    try:
        factor_df = build_factor_from_recipe_mapping(
            prices=prices,
            recipe=factor_recipe,
            factor_name=factor_name,
        )
    except FactorRecipeError as exc:
        raise ValueError(f"invalid factor recipe: {exc}") from exc
    factor_path = input_dir / factor_csv
    factor_df.to_csv(factor_path, index=False)


def _select_factor_csv_for_spec(factor_name: str) -> str:
    token = factor_name.strip().lower()
    if token == "bp":
        return "bp.csv"
    if token in {"roe_ttm", "roe"}:
        return "roe_ttm.csv"
    return ""


def _extract_metrics_summary(
    metrics_path: Path,
    *,
    backtest_result_path: Path | None = None,
) -> dict[str, object]:
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    metrics_obj = payload.get("metrics")
    metrics = metrics_obj if isinstance(metrics_obj, dict) else {}
    out: dict[str, object] = {
        "factor_verdict": _clean_text(metrics.get("factor_verdict")),
        "campaign_triage": _clean_text(metrics.get("campaign_triage")),
        "promotion_decision": _clean_text(metrics.get("promotion_decision")),
        "level12_transition_label": _clean_text(metrics.get("level12_transition_label")),
        "portfolio_validation_status": _clean_text(metrics.get("portfolio_validation_status")),
        "portfolio_validation_recommendation": _clean_text(
            metrics.get("portfolio_validation_recommendation")
        ),
        "mean_ic": _as_float(metrics.get("mean_ic")),
        "mean_rank_ic": _as_float(metrics.get("mean_rank_ic")),
        "ic_ir": _as_float(metrics.get("ic_ir")),
        "mean_long_short_return": _as_float(metrics.get("mean_long_short_return")),
        "mean_long_short_turnover": _as_float(metrics.get("mean_long_short_turnover")),
        "coverage_mean": _as_float(metrics.get("coverage_mean")),
        "ic_positive_rate": _as_float(metrics.get("ic_positive_rate")),
        "rank_ic_positive_rate": _as_float(metrics.get("rank_ic_positive_rate")),
        "long_short_hit_rate": _as_float(metrics.get("long_short_hit_rate")),
        "rolling_ic_positive_share": _as_float(metrics.get("rolling_ic_positive_share")),
        "rolling_rank_ic_positive_share": _as_float(metrics.get("rolling_rank_ic_positive_share")),
        "rolling_long_short_positive_share": _as_float(
            metrics.get("rolling_long_short_positive_share")
        ),
        "subperiod_ic_positive_share": _as_float(metrics.get("subperiod_ic_positive_share")),
        "subperiod_long_short_positive_share": _as_float(
            metrics.get("subperiod_long_short_positive_share")
        ),
        "factor_verdict_reasons": _to_text_list(
            metrics.get("factor_verdict_reasons"),
            max_items=5,
        ),
        "campaign_triage_reasons": _to_text_list(
            metrics.get("campaign_triage_reasons"),
            max_items=5,
        ),
        "promotion_reasons": _to_text_list(metrics.get("promotion_reasons"), max_items=5),
        "promotion_blockers": _to_text_list(metrics.get("promotion_blockers"), max_items=5),
        "portfolio_validation_major_risks": _to_text_list(
            metrics.get("portfolio_validation_major_risks"),
            max_items=5,
        ),
    }
    backtest_summary = (
        _extract_backtest_summary(backtest_result_path) if backtest_result_path is not None else {}
    )
    out.update(backtest_summary)
    out["interview_brief"] = _build_interview_brief(out)
    out["decision_analysis"] = _build_decision_analysis(out)
    return out


def _extract_backtest_summary(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    summary_obj = payload.get("summary")
    summary = summary_obj if isinstance(summary_obj, dict) else {}
    return {
        "annualized_return": _as_float(summary.get("annualized_return")),
        "annualized_volatility": _as_float(summary.get("annualized_volatility")),
        "sharpe": _as_float(summary.get("sharpe")),
        "sortino": _as_float(summary.get("sortino")),
        "max_drawdown": _as_float(summary.get("max_drawdown")),
        "calmar": _as_float(summary.get("calmar")),
        "win_rate": _as_float(summary.get("win_rate")),
        "information_ratio": _as_float(summary.get("information_ratio")),
        "tracking_error": _as_float(summary.get("tracking_error")),
        "turnover": _as_float(summary.get("turnover")),
        "pre_cost_return": _as_float(summary.get("pre_cost_return")),
        "post_cost_return": _as_float(summary.get("post_cost_return")),
    }


def _build_interview_brief(summary: dict[str, object]) -> dict[str, object]:
    factor_verdict = _clean_text(summary.get("factor_verdict")) or "N/A"
    campaign_triage = _clean_text(summary.get("campaign_triage")) or "N/A"
    promotion_decision = _clean_text(summary.get("promotion_decision")) or "N/A"
    transition_label = _clean_text(summary.get("level12_transition_label")) or "N/A"
    portfolio_status = _clean_text(summary.get("portfolio_validation_status")) or "N/A"

    opening_30s = (
        "这是一个面向投研流程的量化研究工作台，覆盖因子发现、信号验证、"
        "组合构建和回测评估，并把关键判断标准透明化。"
        f"当前案例的因子结论是“{factor_verdict}”，"
        f"活动分诊为“{campaign_triage}”，"
        f"晋升决策为“{promotion_decision}”。"
    )
    deep_dive_90s = (
        "我会先用信息系数 (IC) / 秩信息系数 (Rank IC) 判断横截面有效性，"
        "再看长短组合收益与换手效率，最后结合子区间与滚动稳定性做稳健性审查。"
        f"在这个案例里，L1→L2 过渡标签是“{transition_label}”，"
        f"组合层状态是“{portfolio_status}”，"
        "所以可以快速判断这个信号当前更适合继续打磨还是进入组合验证。"
    )

    highlight_rows = [
        f"ICIR: {_fmt_float_metric(_as_float(summary.get('ic_ir')))}",
        f"Mean IC: {_fmt_float_metric(_as_float(summary.get('mean_ic')))}",
        f"Mean Rank IC: {_fmt_float_metric(_as_float(summary.get('mean_rank_ic')))}",
        (
            "Long-Short Mean Return: "
            f"{_fmt_float_metric(_as_float(summary.get('mean_long_short_return')))}"
        ),
        (f"Annualized Return: {_fmt_pct_metric(_as_float(summary.get('annualized_return')))}"),
        f"Sharpe: {_fmt_float_metric(_as_float(summary.get('sharpe')))}",
        f"Max Drawdown: {_fmt_pct_metric(_as_float(summary.get('max_drawdown')))}",
    ]
    risk_controls = [
        "PIT / as-of 时间对齐与未来数据泄漏检查",
        "统一 artifact contract，保证复现实验口径一致",
        "因子→组合的分层决策链路，避免黑盒结论",
    ]
    talking_points = [
        "先讲结论标签，再讲证据指标，最后讲风险控制边界。",
        "强调该系统适用于 Level 1/2 投研，不混入 execution replay 语义。",
        "展示同一页面可回看指标、图像和产物链接，便于团队协同评审。",
    ]
    return {
        "opening_30s": opening_30s,
        "deep_dive_90s": deep_dive_90s,
        "highlights": highlight_rows,
        "risk_controls": risk_controls,
        "talking_points": talking_points,
    }


def _build_decision_analysis(summary: dict[str, object]) -> dict[str, object]:
    factor_verdict = _clean_text(summary.get("factor_verdict")) or "N/A"
    campaign_triage = _clean_text(summary.get("campaign_triage")) or "N/A"
    promotion_decision = _clean_text(summary.get("promotion_decision")) or "N/A"
    transition_label = _clean_text(summary.get("level12_transition_label")) or "N/A"
    portfolio_status = _clean_text(summary.get("portfolio_validation_status")) or "N/A"
    portfolio_recommendation = (
        _clean_text(summary.get("portfolio_validation_recommendation")) or "N/A"
    )

    l1_evidence = _build_l1_evidence(summary)
    promotion_evidence = [
        f"因子判定 (Factor Verdict): {factor_verdict}",
        f"活动分诊 (Campaign Triage): {campaign_triage}",
        f"L1→L2 过渡标签 (Transition Label): {transition_label}",
    ]
    portfolio_evidence = [
        f"组合验证状态 (Portfolio Validation Status): {portfolio_status}",
        f"组合验证建议 (Recommendation): {portfolio_recommendation}",
    ]
    nodes: list[dict[str, object]] = [
        {
            "title": "因子判定 (Factor Verdict, L1)",
            "status": factor_verdict,
            "meaning": _factor_meaning(factor_verdict),
            "reasons": _select_reasons(
                primary=summary.get("factor_verdict_reasons"),
                fallback=[
                    _factor_reason_fallback(factor_verdict),
                ],
            ),
            "evidence": l1_evidence[:4],
            "next_action": _factor_next_action(factor_verdict),
        },
        {
            "title": "活动分诊 (Campaign Triage, L1)",
            "status": campaign_triage,
            "meaning": _campaign_triage_meaning(campaign_triage),
            "reasons": _select_reasons(
                primary=summary.get("campaign_triage_reasons"),
                fallback=[
                    _campaign_reason_fallback(campaign_triage),
                ],
            ),
            "evidence": l1_evidence[:3],
            "next_action": _campaign_next_action(campaign_triage),
        },
        {
            "title": "晋升决策 (Promotion Decision, L1→L2 Gate)",
            "status": promotion_decision,
            "meaning": _promotion_meaning(promotion_decision),
            "reasons": _select_reasons(
                primary=summary.get("promotion_reasons"),
                secondary=summary.get("promotion_blockers"),
                fallback=[
                    _promotion_reason_fallback(promotion_decision),
                ],
            ),
            "evidence": promotion_evidence,
            "next_action": _promotion_next_action(promotion_decision),
        },
        {
            "title": "L1→L2 过渡标签 (Transition Label)",
            "status": transition_label,
            "meaning": _transition_meaning(transition_label),
            "reasons": _select_reasons(
                primary=summary.get("promotion_blockers"),
                fallback=[
                    _transition_reason_fallback(transition_label),
                ],
            ),
            "evidence": promotion_evidence,
            "next_action": _transition_next_action(transition_label),
        },
        {
            "title": "组合验证状态 (Portfolio Validation, L2)",
            "status": portfolio_status,
            "meaning": _portfolio_status_meaning(portfolio_status),
            "reasons": _select_reasons(
                primary=summary.get("portfolio_validation_major_risks"),
                fallback=[
                    _portfolio_reason_fallback(portfolio_status, portfolio_recommendation),
                ],
            ),
            "evidence": portfolio_evidence,
            "next_action": portfolio_recommendation,
        },
    ]
    workflow = {
        "l1_title": "Level 1: 因子发现与稳健性 (Factor Discovery & Robustness)",
        "l1_focus": [
            "验证信息系数 (IC) / 秩信息系数 (Rank IC) 是否稳定为正。",
            "评估多空收益、覆盖率、换手率、子区间稳定性等基本可用性。",
            "输出因子判定、活动分诊与晋升决策。",
        ],
        "l2_title": "Level 2: 组合构建验证 (Portfolio Construction Validation)",
        "l2_focus": [
            "仅对通过 L1 门槛的信号进行组合层验证。",
            "检查组合收益风险、回撤、容量与稳健性。",
            "形成可执行的组合研究结论，而不是单点因子结论。",
        ],
        "gate_rule": "Gate 规则：只有晋升决策为 Promote，才进入 Level 2 组合验证。",
    }
    return {"workflow": workflow, "nodes": nodes}


def _build_l1_evidence(summary: dict[str, object]) -> list[str]:
    rows: list[str] = []
    _append_signed_evidence(
        rows,
        label="信息系数 (IC) 均值",
        value=_as_float(summary.get("mean_ic")),
    )
    _append_signed_evidence(
        rows,
        label="ICIR",
        value=_as_float(summary.get("ic_ir")),
    )
    _append_ratio_evidence(
        rows,
        label="IC 正值占比",
        value=_as_float(summary.get("ic_positive_rate")),
        threshold=0.5,
    )
    _append_signed_evidence(
        rows,
        label="多空收益均值 (Long-Short Mean Return)",
        value=_as_float(summary.get("mean_long_short_return")),
    )
    _append_ratio_evidence(
        rows,
        label="多空胜率 (Long-Short Hit Rate)",
        value=_as_float(summary.get("long_short_hit_rate")),
        threshold=0.5,
    )
    _append_ratio_evidence(
        rows,
        label="覆盖率均值 (Coverage Mean)",
        value=_as_float(summary.get("coverage_mean")),
        threshold=0.6,
    )
    max_drawdown = _as_float(summary.get("max_drawdown"))
    if max_drawdown is not None and math.isfinite(max_drawdown):
        note = "回撤可控" if max_drawdown >= -0.2 else "回撤偏大"
        rows.append(f"最大回撤 (Max Drawdown) = {_fmt_pct_metric(max_drawdown)}（{note}）")
    return rows


def _append_signed_evidence(rows: list[str], *, label: str, value: float | None) -> None:
    if value is None or not math.isfinite(value):
        return
    note = "偏正向" if value > 0 else ("中性" if value == 0 else "偏弱")
    rows.append(f"{label} = {_fmt_float_metric(value)}（{note}）")


def _append_ratio_evidence(
    rows: list[str],
    *,
    label: str,
    value: float | None,
    threshold: float,
) -> None:
    if value is None or not math.isfinite(value):
        return
    note = "偏稳健" if value >= threshold else "偏弱"
    rows.append(f"{label} = {_fmt_pct_metric(value)}（{note}）")


def _select_reasons(
    *,
    primary: object,
    secondary: object | None = None,
    fallback: list[str] | None = None,
) -> list[str]:
    primary_rows = _to_text_list(primary, max_items=4)
    secondary_rows = _to_text_list(secondary, max_items=4) if secondary is not None else []
    merged = _dedupe_texts(primary_rows + secondary_rows, max_items=5)
    if merged:
        return merged
    if fallback is None:
        return []
    return _dedupe_texts([item for item in fallback if item.strip()], max_items=3)


def _dedupe_texts(rows: list[str], *, max_items: int) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for row in rows:
        token = row.strip()
        if not token or token in seen:
            continue
        seen.add(token)
        out.append(token)
        if len(out) >= max_items:
            break
    return out


def _factor_meaning(status: str) -> str:
    token = status.lower()
    if "fail" in token:
        return "L1 判断该因子未通过基础稳健性门槛，通常表示方向性、稳定性或覆盖度存在短板。"
    if "pass" in token:
        return "L1 基础稳健性通过，可继续进入下一步筛选。"
    return "L1 对因子稳健性的判断结果。"


def _campaign_triage_meaning(status: str) -> str:
    token = status.lower()
    if "drop" in token:
        return "在当前证据下建议暂不继续投入，优先把资源给更有潜力的候选因子。"
    if "keep" in token:
        return "当前研究线索仍有价值，建议继续跟踪或迭代。"
    return "研究活动层的资源分配建议。"


def _promotion_meaning(status: str) -> str:
    token = status.lower()
    if "block" in token or "not promoted" in token:
        return "L1→L2 晋升门槛未满足，因此不会进入组合构建验证。"
    if "promote" in token:
        return "满足晋升门槛，可进入 L2 组合构建验证。"
    return "决定是否从 L1 晋升到 L2 的关口结论。"


def _transition_meaning(status: str) -> str:
    token = status.lower()
    if "inconclusive" in token:
        return "L1 到 L2 的证据链不充分或存在冲突，过渡判断暂不明确。"
    return "描述 L1 到 L2 之间证据一致性的标签。"


def _portfolio_status_meaning(status: str) -> str:
    token = status.lower()
    if "skipped" in token:
        return "L2 组合验证步骤被跳过，通常因为该因子未晋升。"
    if "pass" in token:
        return "L2 组合验证通过，组合层风险收益表现满足预期。"
    return "L2 组合验证阶段的执行状态。"


def _factor_reason_fallback(status: str) -> str:
    token = status.lower()
    if "fail" in token:
        return "基础稳健性证据不足，需先修复 L1 指标短板。"
    return "因子判定依据来自 L1 的稳定性与有效性指标。"


def _campaign_reason_fallback(status: str) -> str:
    token = status.lower()
    if "drop" in token:
        return "在当前证据条件下，继续投入该因子的边际收益较低。"
    return "分诊结果基于当前因子证据质量与研究优先级。"


def _promotion_reason_fallback(status: str) -> str:
    token = status.lower()
    if "block" in token or "not promoted" in token:
        return "晋升门槛未满足，因此被阻断在 L1。"
    return "晋升依据来自 L1 因子判定与活动分诊。"


def _transition_reason_fallback(status: str) -> str:
    token = status.lower()
    if "inconclusive" in token:
        return "关键证据存在不一致，暂不支持明确过渡结论。"
    return "过渡标签用于描述 L1 与 L2 之间证据一致性。"


def _portfolio_reason_fallback(status: str, recommendation: str) -> str:
    token = status.lower()
    if "skipped" in token:
        return "由于未晋升，L2 验证按规则跳过。"
    if recommendation != "N/A":
        return recommendation
    return "组合验证建议来自 L2 风险收益评估结果。"


def _factor_next_action(status: str) -> str:
    if "fail" in status.lower():
        return "优先修复 L1 指标，再重跑评估。"
    return "保持当前配置，继续观察稳定性。"


def _campaign_next_action(status: str) -> str:
    if "drop" in status.lower():
        return "暂时下线该候选因子，后续有新证据再回归。"
    return "维持跟踪并继续迭代。"


def _promotion_next_action(status: str) -> str:
    token = status.lower()
    if "block" in token or "not promoted" in token:
        return "不进入 L2，先完成 L1 层面的修复与再验证。"
    return "进入 L2 组合构建验证。"


def _transition_next_action(status: str) -> str:
    if "inconclusive" in status.lower():
        return "补充样本外稳定性证据，再判断是否具备晋升条件。"
    return "按当前过渡标签执行后续流程。"


def _extract_visualization_payload(artifact_paths: dict[str, str]) -> dict[str, object]:
    ic_path = _to_path_or_none(artifact_paths.get("ic_timeseries"))
    turnover_path = _to_path_or_none(artifact_paths.get("turnover"))
    group_returns_path = _to_path_or_none(artifact_paths.get("group_returns"))
    rolling_path = _to_path_or_none(artifact_paths.get("rolling_stability"))

    series: dict[str, list[dict[str, object]]] = {}
    series["ic"] = (
        _read_numeric_series_csv(ic_path, date_col="date", value_col="ic")
        if ic_path is not None
        else []
    )
    series["rank_ic"] = (
        _read_numeric_series_csv(ic_path, date_col="date", value_col="rank_ic")
        if ic_path is not None
        else []
    )
    series["turnover"] = (
        _read_numeric_series_csv(turnover_path, date_col="date", value_col="turnover")
        if turnover_path is not None
        else []
    )
    series["rolling_mean_ic"] = (
        _read_numeric_series_csv(
            rolling_path,
            date_col="date",
            value_col="rolling_mean_ic",
        )
        if rolling_path is not None
        else []
    )
    long_short_points = (
        _read_long_short_from_group_returns(group_returns_path, max_points=0)
        if group_returns_path is not None
        else []
    )
    series["long_short"] = long_short_points
    series["cum_long_short"] = _cumulative_series(long_short_points)
    series["long_short_drawdown"] = _drawdown_series(series["cum_long_short"])
    series["rolling_mean_long_short"] = (
        _read_numeric_series_csv(
            rolling_path,
            date_col="date",
            value_col="rolling_mean_long_short_return",
        )
        if rolling_path is not None
        else []
    )
    series["rolling_long_short_positive_rate"] = (
        _read_numeric_series_csv(
            rolling_path,
            date_col="date",
            value_col="rolling_long_short_positive_rate",
        )
        if rolling_path is not None
        else []
    )
    group_mean_returns = (
        _read_group_mean_returns(group_returns_path) if group_returns_path is not None else []
    )
    ic_histogram = _histogram_from_series(series["ic"], n_bins=12)
    rank_ic_histogram = _histogram_from_series(series["rank_ic"], n_bins=12)

    return {
        "series": series,
        "series_point_counts": {name: len(points) for name, points in series.items()},
        "group_mean_returns": group_mean_returns,
        "ic_histogram": ic_histogram,
        "rank_ic_histogram": rank_ic_histogram,
    }


def _to_path_or_none(path_text: str | None) -> Path | None:
    if path_text is None:
        return None
    token = str(path_text).strip()
    if not token:
        return None
    path = Path(token)
    if not path.exists() or not path.is_file():
        return None
    return path


def _read_numeric_series_csv(
    path: Path,
    *,
    date_col: str,
    value_col: str,
    max_points: int = 240,
) -> list[dict[str, object]]:
    rows = _read_csv_rows(path)
    points: list[tuple[str, float]] = []
    for row in rows:
        date = str(row.get(date_col) or "").strip()
        value = _as_float(row.get(value_col))
        if not date or value is None or not math.isfinite(value):
            continue
        points.append((date, float(value)))
    points = _downsample_points(points, max_points=max_points)
    return [
        {
            "date": date,
            "value": value,
        }
        for date, value in points
    ]


def _read_long_short_from_group_returns(
    path: Path,
    *,
    max_points: int = 240,
) -> list[dict[str, object]]:
    rows = _read_csv_rows(path)
    by_date: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        date = str(row.get("date") or "").strip()
        group_return = _as_float(row.get("group_return"))
        if not date or group_return is None or not math.isfinite(group_return):
            continue
        by_date[date].append(float(group_return))

    points: list[tuple[str, float]] = []
    for date in sorted(by_date):
        values = by_date[date]
        if not values:
            continue
        long_short = max(values) - min(values)
        points.append((date, long_short))
    points = _downsample_points(points, max_points=max_points)
    return [
        {
            "date": date,
            "value": value,
        }
        for date, value in points
    ]


def _read_group_mean_returns(path: Path) -> list[dict[str, object]]:
    rows = _read_csv_rows(path)
    by_group: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        group = str(row.get("group") or "").strip()
        value = _as_float(row.get("group_return"))
        if not group or value is None or not math.isfinite(value):
            continue
        by_group[group].append(float(value))
    out: list[dict[str, object]] = []
    for group in sorted(by_group, key=_group_sort_key):
        vals = by_group[group]
        if not vals:
            continue
        out.append(
            {
                "group": group,
                "value": sum(vals) / float(len(vals)),
                "n_obs": len(vals),
            }
        )
    return out


def _group_sort_key(token: str) -> tuple[int, str]:
    num = _as_float(token)
    if num is None:
        return (10_000, token)
    return (int(num * 1000), token)


def _cumulative_series(points: list[dict[str, object]]) -> list[dict[str, object]]:
    nav = 1.0
    out: list[dict[str, object]] = []
    for item in points:
        date = str(item.get("date") or "").strip()
        value = _as_float(item.get("value"))
        if not date or value is None or not math.isfinite(value):
            continue
        nav *= 1.0 + value
        out.append({"date": date, "value": nav - 1.0})
    return out


def _drawdown_series(points: list[dict[str, object]]) -> list[dict[str, object]]:
    peak = 1.0
    out: list[dict[str, object]] = []
    for item in points:
        date = str(item.get("date") or "").strip()
        cum = _as_float(item.get("value"))
        if not date or cum is None or not math.isfinite(cum):
            continue
        nav = 1.0 + cum
        if nav > peak:
            peak = nav
        drawdown = nav / peak - 1.0 if peak > 0 else 0.0
        out.append({"date": date, "value": drawdown})
    return out


def _histogram_from_series(
    points: list[dict[str, object]],
    *,
    n_bins: int,
) -> list[dict[str, object]]:
    values = [
        value
        for value in (_as_float(item.get("value")) for item in points)
        if value is not None and math.isfinite(value)
    ]
    if not values:
        return []
    n = max(1, n_bins)
    left = min(values)
    right = max(values)
    if abs(right - left) < 1e-12:
        return [{"left": left, "right": right, "count": len(values)}]
    width = (right - left) / float(n)
    bins = [0 for _ in range(n)]
    for value in values:
        idx = int((value - left) / width)
        if idx >= n:
            idx = n - 1
        if idx < 0:
            idx = 0
        bins[idx] += 1
    out: list[dict[str, object]] = []
    for idx, count in enumerate(bins):
        b_left = left + idx * width
        b_right = b_left + width
        out.append({"left": b_left, "right": b_right, "count": count})
    return out


def _read_csv_rows(path: Path) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8", newline="") as fp:
        reader = DictReader(fp)
        for row in reader:
            out.append(dict(row))
    return out


def _downsample_points(
    points: list[tuple[str, float]],
    *,
    max_points: int,
) -> list[tuple[str, float]]:
    if max_points <= 0 or len(points) <= max_points:
        return points
    idxs = _linspace_indices(len(points), max_points)
    return [points[idx] for idx in idxs]


def _linspace_indices(n_points: int, target: int) -> list[int]:
    if target <= 1:
        return [n_points - 1]
    if n_points <= target:
        return list(range(n_points))
    step = (n_points - 1) / float(target - 1)
    idxs: list[int] = []
    seen: set[int] = set()
    for i in range(target):
        idx = int(round(i * step))
        idx = max(0, min(n_points - 1, idx))
        if idx in seen:
            continue
        seen.add(idx)
        idxs.append(idx)
    if idxs[-1] != n_points - 1:
        idxs.append(n_points - 1)
    return idxs


def _as_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    token = str(value).strip()
    if not token:
        return None
    try:
        return float(token)
    except ValueError:
        return None


def _to_text_list(value: object, *, max_items: int) -> list[str]:
    if max_items <= 0 or not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        token = _clean_text(item)
        if token is None:
            continue
        out.append(token)
        if len(out) >= max_items:
            break
    return out


def _fmt_float_metric(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "N/A"
    return f"{value:.4f}"


def _fmt_pct_metric(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "N/A"
    return f"{value * 100:.2f}%"


def _clean_text(value: object) -> str | None:
    if value is None:
        return None
    token = str(value).strip()
    return token if token else None


def _utc_now_iso() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat()


def _index_html() -> str:
    return """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>alpha-lab Web UI</title>
  <style>
    :root {
      --bg: #f6f8fa;
      --panel: #ffffff;
      --line: #d0d7de;
      --text: #1f2328;
      --muted: #59636e;
      --brand: #0969da;
      --brand-2: #0a7ea4;
      --ok: #1a7f37;
      --err: #d1242f;
    }
    body {
      margin: 0;
      font-family: "MiSans", "PingFang SC", "Microsoft YaHei UI", sans-serif;
      background: var(--bg);
      color: var(--text);
    }
    .wrap { max-width: 960px; margin: 24px auto; padding: 0 16px; }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 16px;
      margin-bottom: 16px;
    }
    .panel.hero {
      border-color: #bfd3ea;
      background: linear-gradient(160deg, #ffffff 0%, #f1f8ff 100%);
    }
    h1 { margin: 0 0 8px 0; font-size: 22px; }
    h2 { margin: 0 0 10px 0; font-size: 18px; }
    p { margin: 6px 0; color: var(--muted); }
    .hero-grid {
      display: grid;
      grid-template-columns: 1.6fr 1fr;
      gap: 12px;
      align-items: start;
    }
    .hero-title {
      margin: 0;
      font-size: 24px;
      line-height: 1.3;
      color: #0d1d31;
    }
    .hero-sub {
      font-size: 14px;
      margin-top: 8px;
    }
    .badge-wrap {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 4px;
      justify-content: flex-end;
    }
    .badge {
      font-size: 12px;
      border-radius: 999px;
      border: 1px solid #bfd3ea;
      background: #ffffff;
      color: #0d1d31;
      padding: 4px 10px;
      font-weight: 700;
      white-space: nowrap;
    }
    label { display: block; margin-top: 10px; font-weight: 600; font-size: 14px; }
    input[type="file"], input[type="text"], select {
      width: 100%;
      margin-top: 6px;
      padding: 8px;
      border: 1px solid var(--line);
      border-radius: 6px;
      box-sizing: border-box;
    }
    input[type="date"], input[type="password"], input[type="number"] {
      width: 100%;
      margin-top: 6px;
      padding: 8px;
      border: 1px solid var(--line);
      border-radius: 6px;
      box-sizing: border-box;
    }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    .row3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; }
    .source-box {
      margin-top: 10px;
      padding: 10px;
      border: 1px solid #d8dee4;
      border-radius: 8px;
      background: #fafbfc;
    }
    .source-hint {
      font-size: 12px;
      margin: 4px 0 0 0;
      color: var(--muted);
    }
    .btn {
      margin-top: 14px;
      border: 0;
      background: var(--brand);
      color: #fff;
      padding: 10px 14px;
      border-radius: 6px;
      cursor: pointer;
    }
    .btn:disabled { opacity: 0.6; cursor: not-allowed; }
    .status { font-weight: 700; }
    .ok { color: var(--ok); }
    .err { color: var(--err); }
    code { background: #f0f3f6; padding: 1px 4px; border-radius: 4px; }
    ul { margin: 8px 0 0 0; padding-left: 18px; }
    li { margin: 4px 0; }
    .chart-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 12px;
      margin-top: 10px;
    }
    .chart-card {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px;
      background: #fafbfc;
    }
    .chart-title {
      font-size: 13px;
      font-weight: 700;
      margin: 0 0 6px 0;
      color: var(--text);
    }
    .chart-meta {
      font-size: 12px;
      color: var(--muted);
      margin: 4px 0 0 0;
    }
    .chart-svg {
      width: 100%;
      height: 120px;
      display: block;
      background: #fff;
      border: 1px solid #eaeef2;
      border-radius: 6px;
    }
    .flow-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      gap: 10px;
      margin-top: 8px;
    }
    .flow-card {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fafbfc;
      padding: 8px;
    }
    .flow-step {
      font-size: 12px;
      color: var(--muted);
      margin: 0 0 4px 0;
    }
    .flow-value {
      font-size: 13px;
      font-weight: 700;
      margin: 0;
      color: var(--text);
    }
    .kpi-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 10px;
      margin-top: 8px;
    }
    .kpi-card {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px;
      background: #fff;
    }
    .kpi-name {
      font-size: 12px;
      color: var(--muted);
      margin: 0 0 4px 0;
    }
    .kpi-value {
      font-size: 16px;
      font-weight: 700;
      margin: 0;
      color: var(--text);
    }
    .kpi-explain {
      font-size: 12px;
      color: var(--muted);
      margin: 4px 0 0 0;
    }
    .rate-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 10px;
      margin-top: 8px;
    }
    .rate-card {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      padding: 8px;
    }
    .rate-label {
      font-size: 12px;
      color: var(--text);
      margin: 0 0 4px 0;
    }
    .rate-bar {
      width: 100%;
      height: 10px;
      background: #eaeef2;
      border-radius: 999px;
      overflow: hidden;
    }
    .rate-fill {
      height: 100%;
      border-radius: 999px;
    }
    .rate-value {
      font-size: 12px;
      color: var(--muted);
      margin: 4px 0 0 0;
    }
    .explain-list {
      margin-top: 8px;
      padding-left: 18px;
      color: var(--muted);
      font-size: 13px;
    }
    .primer-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
      gap: 10px;
      margin-top: 10px;
    }
    .primer-card {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      padding: 10px;
    }
    .primer-title {
      margin: 0 0 6px 0;
      font-size: 13px;
      font-weight: 700;
      color: #0d1d31;
    }
    .primer-list {
      margin: 0;
      padding-left: 18px;
      font-size: 13px;
      color: #2f3f52;
    }
    .gate-note {
      margin-top: 8px;
      padding: 8px 10px;
      border-radius: 8px;
      border: 1px solid #bfd3ea;
      background: #f3f8ff;
      font-size: 13px;
      color: #0d1d31;
    }
    .analysis-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 10px;
      margin-top: 10px;
    }
    .analysis-card {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      padding: 10px;
    }
    .analysis-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      margin-bottom: 6px;
    }
    .analysis-title {
      margin: 0;
      font-size: 13px;
      font-weight: 700;
      color: #0d1d31;
    }
    .status-pill {
      font-size: 12px;
      padding: 2px 8px;
      white-space: nowrap;
      border-radius: 999px;
      border: 1px solid #d0d7de;
      color: #59636e;
      background: #f6f8fa;
    }
    .status-pill.good {
      background: #e6f6ea;
      border-color: #9bd4a8;
      color: #1a7f37;
    }
    .status-pill.warn {
      background: #fff8e6;
      border-color: #f0cf8d;
      color: #9a6700;
    }
    .status-pill.bad {
      background: #fff0f0;
      border-color: #f1b5b8;
      color: #b4232c;
    }
    .analysis-meaning {
      margin: 0;
      font-size: 13px;
      color: #2f3f52;
      line-height: 1.6;
    }
    .analysis-block {
      margin-top: 8px;
    }
    .analysis-label {
      margin: 0 0 4px 0;
      font-size: 12px;
      color: #59636e;
      font-weight: 700;
    }
    .analysis-list {
      margin: 0;
      padding-left: 18px;
      font-size: 13px;
      color: #2f3f52;
    }
    .analysis-list li {
      margin: 4px 0;
    }
    .analysis-next {
      margin-top: 8px;
      font-size: 13px;
      color: #0d1d31;
      font-weight: 600;
    }
    @media (max-width: 780px) {
      .row { grid-template-columns: 1fr; }
      .row3 { grid-template-columns: 1fr; }
      .hero-grid { grid-template-columns: 1fr; }
      .badge-wrap { justify-content: flex-start; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="panel hero">
      <div class="hero-grid">
        <div>
          <h1 class="hero-title">alpha-lab 量化研究工作台 (Professional Dashboard)</h1>
          <p class="hero-sub">
            上传单因子 case spec（YAML/JSON），自动触发 Level 1/2 流程，
            生成可讲解的研究结论、关键图表和回测证据。
          </p>
          <p>
            说明：spec 内的 <code>factor_path</code>/<code>prices_path</code>
            建议使用绝对路径，或相对于 spec 文件所在目录。
          </p>
        </div>
        <div class="badge-wrap">
          <span class="badge">信息系数 (IC)</span>
          <span class="badge">秩信息系数 (Rank IC)</span>
          <span class="badge">最大回撤 (Max Drawdown)</span>
          <span class="badge">换手率 (Turnover)</span>
          <span class="badge">样本外稳定性 (OOS Stability)</span>
        </div>
      </div>
    </div>

    <div class="panel">
      <h2>1) 提交运行</h2>
      <label for="spec-file">Case Spec 文件</label>
      <input id="spec-file" type="file" accept=".yaml,.yml,.json,.txt" />
      <label for="data-source">数据源 (Data Source)</label>
      <select id="data-source">
        <option value="manual" selected>Manual（仅使用 Spec 内现有路径）</option>
        <option value="tushare">Tushare（自动拉取并覆盖 spec 数据路径）</option>
        <option value="baostock">Baostock（自动拉取并覆盖 spec 数据路径）</option>
      </select>
      <div id="source-box" class="source-box" style="display:none;">
        <p class="source-hint">
          选择自动数据源后，系统会先拉取输入 CSV；若 factor_name 是 bp/roe 则自动映射，
          其余因子可在 spec 里提供 factor_input.recipe（兼容 legacy factor_recipe）
          自动生成标准因子文件。
        </p>
        <div>
          <label for="data-slice-preset">默认切片预设</label>
          <select id="data-slice-preset">
            <option value="pilot">pilot: 最近 3 年 / top_liquid_300 / qfq</option>
            <option value="standard" selected>standard: 最近 5 年 / listed_90d / qfq</option>
            <option value="robust">robust: 最近 8 年 / listed_90d / qfq</option>
            <option value="institutional">institutional: 最近 8 年 / 私募口径 / qfq</option>
          </select>
          <p id="data-slice-preset-hint" class="source-hint">
            standard 作为默认预设；institutional 会剔除 ST/停牌和低流动性尾部，
            可继续手动改开始/结束日期。
          </p>
        </div>
        <div class="row3">
          <div>
            <label for="data-start-date">开始日期</label>
            <input id="data-start-date" type="date" />
          </div>
          <div>
            <label for="data-end-date">结束日期</label>
            <input id="data-end-date" type="date" />
          </div>
          <div>
            <label for="data-asset-limit">资产上限 (可选)</label>
            <input id="data-asset-limit" type="number" min="1" step="1" />
          </div>
        </div>
        <div id="tushare-token-wrap" style="display:none;">
          <label for="tushare-token">Tushare Token</label>
          <input id="tushare-token" type="password" />
        </div>
      </div>

      <div class="row">
        <div>
          <label for="evaluation-profile">评估配置 (Evaluation Profile)</label>
          <select id="evaluation-profile"></select>
        </div>
        <div>
          <label for="output-root-dir">输出目录根路径</label>
          <input id="output-root-dir" type="text" value="dist/web_ui_runs" />
        </div>
      </div>
      <label><input id="render-report" type="checkbox" checked /> 生成 case_report.md</label>
      <button id="run-btn" class="btn">开始运行</button>
      <p id="submit-msg"></p>
    </div>

    <div class="panel">
      <h2>2) 运行状态与结果</h2>
      <p>Run ID: <code id="run-id">N/A</code></p>
      <p>状态: <span id="run-status" class="status">N/A</span></p>
      <p id="run-error" class="err"></p>
      <p id="run-output"></p>
      <div id="summary-wrap"></div>
      <div id="visualization-wrap"></div>
      <div id="artifacts-wrap"></div>
    </div>
  </div>

  <script>
    const profileSelect = document.getElementById("evaluation-profile");
    const runBtn = document.getElementById("run-btn");
    const specFileInput = document.getElementById("spec-file");
    const dataSourceSelect = document.getElementById("data-source");
    const sourceBox = document.getElementById("source-box");
    const dataSlicePresetSelect = document.getElementById("data-slice-preset");
    const dataSlicePresetHint = document.getElementById("data-slice-preset-hint");
    const dataStartDateInput = document.getElementById("data-start-date");
    const dataEndDateInput = document.getElementById("data-end-date");
    const dataAssetLimitInput = document.getElementById("data-asset-limit");
    const tushareTokenWrap = document.getElementById("tushare-token-wrap");
    const tushareTokenInput = document.getElementById("tushare-token");
    const outputRootInput = document.getElementById("output-root-dir");
    const renderReportInput = document.getElementById("render-report");
    const submitMsg = document.getElementById("submit-msg");
    const runIdEl = document.getElementById("run-id");
    const runStatusEl = document.getElementById("run-status");
    const runErrorEl = document.getElementById("run-error");
    const runOutputEl = document.getElementById("run-output");
    const summaryWrap = document.getElementById("summary-wrap");
    const visualizationWrap = document.getElementById("visualization-wrap");
    const artifactsWrap = document.getElementById("artifacts-wrap");

    let pollTimer = null;
    const slicePresetMeta = {
      pilot: {
        years: 3,
        hint: "pilot: 最近 3 年，top_liquid_300，qfq。按过去 60 日平均成交额选前 300，"
          + "适合快速原型验证。",
      },
      standard: {
        years: 5,
        hint: "standard: 最近 5 年，listed_90d，qfq。适合默认日频量价研究。",
      },
      robust: {
        years: 8,
        hint: "robust: 最近 8 年，listed_90d，qfq。适合稳健性复核。",
      },
      institutional: {
        years: 8,
        hint: "institutional: 最近 8 年，institutional_ashare，qfq。"
          + "上市满 180 天，剔除 ST/停牌，并过滤低流动性尾部。",
      },
    };

    function formatIsoDate(dateObj) {
      const y = dateObj.getFullYear();
      const m = String(dateObj.getMonth() + 1).padStart(2, "0");
      const d = String(dateObj.getDate()).padStart(2, "0");
      return `${y}-${m}-${d}`;
    }

    function applySlicePreset(force = false) {
      const presetName = String(dataSlicePresetSelect.value || "standard");
      const preset = slicePresetMeta[presetName] || slicePresetMeta.standard;
      const end = new Date();
      const start = new Date(end.getTime());
      start.setFullYear(start.getFullYear() - preset.years);
      start.setDate(start.getDate() + 1);
      if (force || !String(dataStartDateInput.value || "").trim()) {
        dataStartDateInput.value = formatIsoDate(start);
      }
      if (force || !String(dataEndDateInput.value || "").trim()) {
        dataEndDateInput.value = formatIsoDate(end);
      }
      dataSlicePresetHint.textContent = preset.hint + " 如需自定义窗口，可直接修改日期。";
    }

    function syncSourceControls() {
      const source = String(dataSourceSelect.value || "manual");
      const autoMode = source === "tushare" || source === "baostock";
      sourceBox.style.display = autoMode ? "" : "none";
      tushareTokenWrap.style.display = source === "tushare" ? "" : "none";
      if (autoMode) {
        applySlicePreset(false);
      }
    }

    async function loadProfiles() {
      const resp = await fetch("/api/profiles");
      const data = await resp.json();
      profileSelect.innerHTML = "";
      for (const profile of data.profiles || []) {
        const opt = document.createElement("option");
        opt.value = profile;
        opt.textContent = profile;
        if (profile === data.default_profile) {
          opt.selected = true;
        }
        profileSelect.appendChild(opt);
      }
    }

    function setStatus(text, ok = null) {
      runStatusEl.textContent = text;
      runStatusEl.className = "status";
      if (ok === true) runStatusEl.classList.add("ok");
      if (ok === false) runStatusEl.classList.add("err");
    }

    function renderSummary(summary) {
      summaryWrap.innerHTML = "";
      if (!summary || Object.keys(summary).length === 0) return;

      const title = document.createElement("p");
      title.innerHTML = "<strong>L1/L2 决策解析（不仅是结论）</strong>";
      summaryWrap.appendChild(title);

      const flowGrid = document.createElement("div");
      flowGrid.className = "flow-grid";
      const flowRows = [
        ["因子判定", summary.factor_verdict],
        ["活动分诊", summary.campaign_triage],
        ["晋升决策", summary.promotion_decision],
        ["L1→L2过渡", summary.level12_transition_label],
        ["组合验证状态", summary.portfolio_validation_status],
        ["组合验证建议", summary.portfolio_validation_recommendation],
      ];
      for (const [step, value] of flowRows) {
        if (!value) continue;
        const card = document.createElement("div");
        card.className = "flow-card";
        card.innerHTML =
          `<p class="flow-step">${step}</p>` +
          `<p class="flow-value">${String(value)}</p>`;
        flowGrid.appendChild(card);
      }
      if (flowGrid.children.length > 0) {
        summaryWrap.appendChild(flowGrid);
      }

      const decisionAnalysis = normalizeDecisionAnalysis(summary);
      const primerSection = buildWorkflowPrimer(decisionAnalysis.workflow);
      if (primerSection) {
        summaryWrap.appendChild(primerSection);
      }
      const decisionCards = buildDecisionCards(decisionAnalysis.nodes);
      if (decisionCards) {
        summaryWrap.appendChild(decisionCards);
      }

      const kpiGrid = document.createElement("div");
      kpiGrid.className = "kpi-grid";
      const kpis = [
        ["IC 均值", summary.mean_ic, "越高越好，反映因子与未来收益的线性相关方向。"],
        ["Rank IC 均值", summary.mean_rank_ic, "越高越好，反映排序能力是否稳定。"],
        ["ICIR", summary.ic_ir, "信噪比，通常 > 0 更稳健。"],
        ["多空收益均值", summary.mean_long_short_return, "分组最高减最低后的平均收益差。"],
        ["年化收益", summary.annualized_return, "策略收益年化后的直观表现。"],
        ["年化波动", summary.annualized_volatility, "收益波动幅度，越低通常风险越小。"],
        ["Sharpe", summary.sharpe, "单位波动收益，通常越高越好。"],
        ["最大回撤", summary.max_drawdown, "历史最深亏损，绝对值越小越好。"],
        ["平均换手率", summary.mean_long_short_turnover, "调仓频率代理，过高会吃掉收益。"],
        ["覆盖率均值", summary.coverage_mean, "可计算资产覆盖比例，越高越好。"],
      ];
      for (const [name, valueRaw, explain] of kpis) {
        const value = Number(valueRaw);
        if (!Number.isFinite(value)) continue;
        const card = document.createElement("div");
        card.className = "kpi-card";
        card.innerHTML =
          `<p class="kpi-name">${name}</p>` +
          `<p class="kpi-value">${formatMetricValue(name, value)}</p>` +
          `<p class="kpi-explain">${explain}</p>`;
        kpiGrid.appendChild(card);
      }
      if (kpiGrid.children.length > 0) {
        summaryWrap.appendChild(kpiGrid);
      }

      const rateGrid = document.createElement("div");
      rateGrid.className = "rate-grid";
      const rates = [
        ["IC 正值占比", summary.ic_positive_rate],
        ["Rank IC 正值占比", summary.rank_ic_positive_rate],
        ["多空胜率", summary.long_short_hit_rate],
        ["滚动IC正值占比", summary.rolling_ic_positive_share],
        ["滚动多空正值占比", summary.rolling_long_short_positive_share],
        ["子区间IC正值占比", summary.subperiod_ic_positive_share],
      ];
      for (const [label, valueRaw] of rates) {
        const value = Number(valueRaw);
        if (!Number.isFinite(value)) continue;
        const clamped = Math.max(0, Math.min(1, value));
        const color = clamped >= 0.55
          ? "#1a7f37"
          : (clamped >= 0.45 ? "#bf8700" : "#d1242f");
        const widthPct = (clamped * 100).toFixed(1);
        const card = document.createElement("div");
        card.className = "rate-card";
        card.innerHTML =
          `<p class="rate-label">${label}</p>` +
          `<div class="rate-bar"><div class="rate-fill" ` +
          `style="width:${widthPct}%;background:${color};"></div></div>` +
          `<p class="rate-value">${widthPct}%</p>`;
        rateGrid.appendChild(card);
      }
      if (rateGrid.children.length > 0) {
        summaryWrap.appendChild(rateGrid);
      }

      const explain = document.createElement("ul");
      explain.className = "explain-list";
      for (const reason of [
        ...(summary.factor_verdict_reasons || []),
        ...(summary.campaign_triage_reasons || []),
      ]) {
        const li = document.createElement("li");
        li.textContent = String(reason);
        explain.appendChild(li);
      }
      if (explain.children.length > 0) {
        summaryWrap.appendChild(explain);
      }
    }

    function toText(value) {
      if (value === null || value === undefined) return "";
      return String(value).trim();
    }

    function toTextArray(value) {
      if (!Array.isArray(value)) return [];
      const out = [];
      for (const item of value) {
        const text = toText(item);
        if (text) out.push(text);
      }
      return out;
    }

    function normalizeDecisionAnalysis(summary) {
      const raw = summary.decision_analysis;
      const fallback = buildFallbackDecisionAnalysis(summary);
      if (!raw || typeof raw !== "object") return fallback;

      const workflowRaw = raw.workflow && typeof raw.workflow === "object"
        ? raw.workflow
        : {};
      const workflow = {
        l1_title: toText(workflowRaw.l1_title) || fallback.workflow.l1_title,
        l1_focus: toTextArray(workflowRaw.l1_focus).length > 0
          ? toTextArray(workflowRaw.l1_focus)
          : fallback.workflow.l1_focus,
        l2_title: toText(workflowRaw.l2_title) || fallback.workflow.l2_title,
        l2_focus: toTextArray(workflowRaw.l2_focus).length > 0
          ? toTextArray(workflowRaw.l2_focus)
          : fallback.workflow.l2_focus,
        gate_rule: toText(workflowRaw.gate_rule) || fallback.workflow.gate_rule,
      };

      const nodesRaw = Array.isArray(raw.nodes) ? raw.nodes : [];
      const nodes = [];
      for (const node of nodesRaw) {
        if (!node || typeof node !== "object") continue;
        const title = toText(node.title);
        if (!title) continue;
        nodes.push({
          title,
          status: toText(node.status) || "N/A",
          meaning: toText(node.meaning),
          reasons: toTextArray(node.reasons),
          evidence: toTextArray(node.evidence),
          next_action: toText(node.next_action),
        });
      }
      return { workflow, nodes: nodes.length > 0 ? nodes : fallback.nodes };
    }

    function buildFallbackDecisionAnalysis(summary) {
      return {
        workflow: {
          l1_title: "Level 1: 因子发现与稳健性",
          l1_focus: [
            "检验 IC / Rank IC、覆盖率、稳定性与多空收益。",
            "给出因子判定、活动分诊与晋升建议。",
          ],
          l2_title: "Level 2: 组合构建验证",
          l2_focus: [
            "仅对晋升信号做组合层风险收益评估。",
            "检验回撤、波动、换手与稳健性。",
          ],
          gate_rule: "仅当晋升决策为 Promote 时进入 Level 2。",
        },
        nodes: [
          {
            title: "因子判定 (L1)",
            status: toText(summary.factor_verdict) || "N/A",
            meaning: "L1 对因子基础稳健性的结论。",
            reasons: toTextArray(summary.factor_verdict_reasons),
            evidence: [],
            next_action: "依据证据结果继续迭代或暂缓。",
          },
          {
            title: "活动分诊 (L1)",
            status: toText(summary.campaign_triage) || "N/A",
            meaning: "研究资源分配建议。",
            reasons: toTextArray(summary.campaign_triage_reasons),
            evidence: [],
            next_action: "按分诊结果安排后续研究优先级。",
          },
          {
            title: "晋升决策 (L1→L2)",
            status: toText(summary.promotion_decision) || "N/A",
            meaning: "决定是否进入组合构建验证。",
            reasons: toTextArray(summary.promotion_reasons),
            evidence: [],
            next_action: toText(summary.portfolio_validation_recommendation),
          },
        ],
      };
    }

    function buildWorkflowPrimer(workflow) {
      if (!workflow || typeof workflow !== "object") return null;
      const l1Title = toText(workflow.l1_title);
      const l2Title = toText(workflow.l2_title);
      const l1Focus = toTextArray(workflow.l1_focus);
      const l2Focus = toTextArray(workflow.l2_focus);
      if (!l1Title && !l2Title) return null;

      const container = document.createElement("div");
      const title = document.createElement("p");
      title.innerHTML = "<strong>L1 / L2 是什么</strong>";
      container.appendChild(title);

      const grid = document.createElement("div");
      grid.className = "primer-grid";
      const l1Card = createPrimerCard(l1Title, l1Focus);
      const l2Card = createPrimerCard(l2Title, l2Focus);
      if (l1Card) grid.appendChild(l1Card);
      if (l2Card) grid.appendChild(l2Card);
      if (grid.children.length > 0) {
        container.appendChild(grid);
      }

      const gateRule = toText(workflow.gate_rule);
      if (gateRule) {
        const note = document.createElement("p");
        note.className = "gate-note";
        note.textContent = gateRule;
        container.appendChild(note);
      }
      return container;
    }

    function createPrimerCard(titleText, items) {
      const title = toText(titleText);
      const rows = toTextArray(items);
      if (!title) return null;
      const card = document.createElement("div");
      card.className = "primer-card";

      const titleEl = document.createElement("p");
      titleEl.className = "primer-title";
      titleEl.textContent = title;
      card.appendChild(titleEl);

      if (rows.length > 0) {
        const ul = document.createElement("ul");
        ul.className = "primer-list";
        for (const row of rows) {
          const li = document.createElement("li");
          li.textContent = row;
          ul.appendChild(li);
        }
        card.appendChild(ul);
      }
      return card;
    }

    function buildDecisionCards(nodes) {
      if (!Array.isArray(nodes) || nodes.length === 0) return null;
      const container = document.createElement("div");
      const title = document.createElement("p");
      title.innerHTML = "<strong>为什么会得到这些结论</strong>";
      container.appendChild(title);

      const grid = document.createElement("div");
      grid.className = "analysis-grid";
      for (const node of nodes) {
        if (!node || typeof node !== "object") continue;
        const card = createDecisionCard(node);
        if (card) grid.appendChild(card);
      }
      if (grid.children.length === 0) return null;
      container.appendChild(grid);
      return container;
    }

    function createDecisionCard(node) {
      const title = toText(node.title);
      if (!title) return null;

      const card = document.createElement("div");
      card.className = "analysis-card";

      const head = document.createElement("div");
      head.className = "analysis-head";
      const titleEl = document.createElement("p");
      titleEl.className = "analysis-title";
      titleEl.textContent = title;
      head.appendChild(titleEl);

      const status = toText(node.status) || "N/A";
      const pill = document.createElement("span");
      pill.className = `status-pill ${statusTone(status)}`;
      pill.textContent = status;
      head.appendChild(pill);
      card.appendChild(head);

      const meaning = toText(node.meaning);
      if (meaning) {
        const p = document.createElement("p");
        p.className = "analysis-meaning";
        p.textContent = meaning;
        card.appendChild(p);
      }

      const reasons = createAnalysisBlock("直接原因", toTextArray(node.reasons));
      if (reasons) card.appendChild(reasons);
      const evidence = createAnalysisBlock("证据指标", toTextArray(node.evidence));
      if (evidence) card.appendChild(evidence);

      const nextAction = toText(node.next_action);
      if (nextAction) {
        const next = document.createElement("p");
        next.className = "analysis-next";
        next.textContent = `下一步: ${nextAction}`;
        card.appendChild(next);
      }
      return card;
    }

    function createAnalysisBlock(label, rows) {
      if (!Array.isArray(rows) || rows.length === 0) return null;
      const block = document.createElement("div");
      block.className = "analysis-block";

      const labelEl = document.createElement("p");
      labelEl.className = "analysis-label";
      labelEl.textContent = label;
      block.appendChild(labelEl);

      const ul = document.createElement("ul");
      ul.className = "analysis-list";
      for (const row of rows) {
        const text = toText(row);
        if (!text) continue;
        const li = document.createElement("li");
        li.textContent = text;
        ul.appendChild(li);
      }
      if (ul.children.length === 0) return null;
      block.appendChild(ul);
      return block;
    }

    function statusTone(value) {
      const token = toText(value).toLowerCase();
      if (!token) return "";
      if (
        token.includes("fail") ||
        token.includes("drop") ||
        token.includes("blocked") ||
        token.includes("skip") ||
        token.includes("not promoted")
      ) {
        return "bad";
      }
      if (
        token.includes("inconclusive") ||
        token.includes("watch") ||
        token.includes("hold") ||
        token.includes("tentative")
      ) {
        return "warn";
      }
      if (
        token.includes("pass") ||
        token.includes("promote") ||
        token.includes("keep") ||
        token.includes("approved")
      ) {
        return "good";
      }
      return "";
    }

    function formatMetricValue(name, value) {
      if (
        name.includes("占比") ||
        name.includes("胜率") ||
        name.includes("覆盖率")
      ) {
        return `${(value * 100).toFixed(1)}%`;
      }
      if (name.includes("回撤")) {
        return `${(value * 100).toFixed(2)}%`;
      }
      return value.toFixed(4);
    }

    function renderArtifacts(runId, artifactPaths) {
      artifactsWrap.innerHTML = "";
      if (!artifactPaths || Object.keys(artifactPaths).length === 0) return;
      const title = document.createElement("p");
      title.innerHTML = "<strong>产物链接</strong>";
      artifactsWrap.appendChild(title);
      const ul = document.createElement("ul");
      for (const [key, _value] of Object.entries(artifactPaths)) {
        const li = document.createElement("li");
        const a = document.createElement("a");
        a.href = `/api/runs/${encodeURIComponent(runId)}/artifact/${encodeURIComponent(key)}`;
        a.target = "_blank";
        a.rel = "noopener";
        a.textContent = key;
        li.appendChild(a);
        ul.appendChild(li);
      }
      artifactsWrap.appendChild(ul);
    }

    function renderVisualization(visualization) {
      visualizationWrap.innerHTML = "";
      if (!visualization || !visualization.series) return;
      const series = visualization.series;
      const chartDefs = [
        ["ic", "IC 时间序列", "#0969da"],
        ["rank_ic", "Rank IC 时间序列", "#1a7f37"],
        ["long_short", "多空收益差时间序列", "#bf8700"],
        ["turnover", "换手率时间序列", "#8250df"],
        ["rolling_mean_ic", "滚动IC均值", "#cf222e"],
        ["cum_long_short", "累计多空收益", "#0a7ea4"],
        ["long_short_drawdown", "多空收益回撤", "#d1242f"],
        ["rolling_mean_long_short", "滚动多空收益均值", "#9a6700"],
        ["rolling_long_short_positive_rate", "滚动多空正收益占比", "#6e7781"],
      ];
      const cards = [];
      for (const [key, title, color] of chartDefs) {
        const points = Array.isArray(series[key]) ? series[key] : [];
        if (points.length < 2) continue;
        const card = buildLineChartCard(title, points, color);
        if (card) cards.push(card);
      }
      const groupMeans = Array.isArray(visualization.group_mean_returns)
        ? visualization.group_mean_returns
        : [];
      if (groupMeans.length > 0) {
        const bar = buildBarChartCard(
          "分组平均收益（group 低→高）",
          groupMeans.map((row) => ({
            label: String(row.group || ""),
            value: Number(row.value),
          })),
          "#0969da"
        );
        if (bar) cards.push(bar);
      }

      const icHist = Array.isArray(visualization.ic_histogram)
        ? visualization.ic_histogram
        : [];
      if (icHist.length > 0) {
        const hist = buildBarChartCard(
          "IC 分布直方图",
          icHist.map((row) => ({
            label: `${Number(row.left).toFixed(2)}~${Number(row.right).toFixed(2)}`,
            value: Number(row.count),
          })),
          "#1a7f37"
        );
        if (hist) cards.push(hist);
      }

      if (cards.length === 0) return;

      const header = document.createElement("p");
      header.innerHTML = "<strong>可视化结果</strong>";
      visualizationWrap.appendChild(header);
      const grid = document.createElement("div");
      grid.className = "chart-grid";
      for (const card of cards) {
        grid.appendChild(card);
      }
      visualizationWrap.appendChild(grid);
    }

    function parseAxisDate(value) {
      const text = String(value || "").trim();
      if (!text) return null;
      const normalized = text.replace(/[/.]/g, "-");
      let match = normalized.match(/^(\\d{4})-(\\d{1,2})(?:-(\\d{1,2}))?/);
      if (!match) {
        const compactYmd = normalized.match(/(?:^|\\D)(\\d{4})(\\d{2})(\\d{2})(?:\\D|$)/);
        if (compactYmd) {
          match = ["", compactYmd[1], compactYmd[2], compactYmd[3]];
        } else {
          const compactYm = normalized.match(/(?:^|\\D)(\\d{4})(\\d{2})(?:\\D|$)/);
          if (compactYm) {
            match = ["", compactYm[1], compactYm[2], "1"];
          }
        }
      }
      if (!match) return null;
      const year = Number(match[1]);
      const month = Number(match[2]);
      const day = Number(match[3] || "1");
      if (!Number.isFinite(year) || !Number.isFinite(month) || !Number.isFinite(day)) return null;
      if (month < 1 || month > 12 || day < 1 || day > 31) return null;
      const parsed = new Date(year, month - 1, day);
      if (
        parsed.getFullYear() !== year
        || parsed.getMonth() !== month - 1
        || parsed.getDate() !== day
      ) {
        return null;
      }
      return parsed;
    }

    function formatAxisDateLabel(date, granularity = "month") {
      const base = `${date.getFullYear()}.${date.getMonth() + 1}`;
      return granularity === "day" ? `${base}.${date.getDate()}` : base;
    }

    function addAxisDays(date, days) {
      return new Date(date.getFullYear(), date.getMonth(), date.getDate() + days);
    }

    function addAxisMonths(date, months) {
      const target = new Date(date.getFullYear(), date.getMonth() + months, 1);
      const lastDay = new Date(target.getFullYear(), target.getMonth() + 1, 0).getDate();
      target.setDate(Math.min(date.getDate(), lastDay));
      return target;
    }

    function chooseAxisDateTickInterval(start, end) {
      const spanDays = Math.max(0, (end - start) / (86400 * 1000));
      if (spanDays <= 45) return { days: 7, months: 0, granularity: "day", minLastGapDays: 4 };
      if (spanDays <= 120) return { days: 14, months: 0, granularity: "day", minLastGapDays: 7 };
      if (spanDays <= 550) return { days: 0, months: 1, granularity: "month", minLastGapDays: 15 };
      if (spanDays <= 1095) return { days: 0, months: 3, granularity: "month", minLastGapDays: 45 };
      if (spanDays <= 2190) return { days: 0, months: 6, granularity: "month", minLastGapDays: 90 };
      return { days: 0, months: 12, granularity: "month", minLastGapDays: 180 };
    }

    function addAxisDateTickInterval(date, interval) {
      if (interval.days > 0) return addAxisDays(date, interval.days);
      return addAxisMonths(date, Math.max(1, interval.months || 1));
    }

    function buildAnnualDateTicks(dateRows) {
      const parsed = (Array.isArray(dateRows) ? dateRows : [])
        .map((value, idx) => ({ idx, date: parseAxisDate(value) }))
        .filter((item) => item.date instanceof Date && Number.isFinite(item.date.getTime()));
      if (parsed.length < 2) return [];
      const start = parsed[0].date;
      const end = parsed[parsed.length - 1].date;
      const interval = chooseAxisDateTickInterval(start, end);
      const ticks = [];
      let lastIdx = -1;
      let target = new Date(start.getTime());
      while (target <= end) {
        const item = parsed.find(
          (candidate) => candidate.idx > lastIdx && candidate.date >= target,
        );
        if (item) {
          const label = formatAxisDateLabel(item.date, interval.granularity);
          if (!ticks.length || ticks[ticks.length - 1].label !== label) {
            ticks.push({ x: item.idx + 1, label, date: item.date });
            lastIdx = item.idx;
          }
        }
        target = addAxisDateTickInterval(target, interval);
      }
      const last = parsed[parsed.length - 1];
      const lastGapDays = ticks.length
        ? (last.date - ticks[ticks.length - 1].date) / (86400 * 1000)
        : Infinity;
      if (!ticks.length || (last.idx > lastIdx && lastGapDays >= interval.minLastGapDays)) {
        ticks.push({
          x: last.idx + 1,
          label: formatAxisDateLabel(last.date, interval.granularity),
          date: last.date,
        });
      }
      if (ticks.length < 2 && last.idx > lastIdx) {
        ticks.push({
          x: last.idx + 1,
          label: formatAxisDateLabel(last.date, interval.granularity),
          date: last.date,
        });
      }
      return ticks.map((tick) => ({ x: tick.x, label: tick.label }));
    }

    function buildLineChartCard(title, points, color) {
      const parsed = [];
      for (const item of points) {
        if (!item || typeof item !== "object") continue;
        const date = String(item.date || "");
        const value = Number(item.value);
        if (!date || !Number.isFinite(value)) continue;
        parsed.push({ date, value });
      }
      if (parsed.length < 2) return null;

      let min = parsed[0].value;
      let max = parsed[0].value;
      for (const p of parsed) {
        if (p.value < min) min = p.value;
        if (p.value > max) max = p.value;
      }
      if (Math.abs(max - min) < 1e-12) {
        max += 1.0;
        min -= 1.0;
      }

      const width = 420;
      const height = 120;
      const padL = 28;
      const padR = 10;
      const padT = 10;
      const padB = 24;
      const innerW = width - padL - padR;
      const innerH = height - padT - padB;
      const n = parsed.length;
      const lineParts = [];
      for (let i = 0; i < n; i += 1) {
        const x = padL + (i / (n - 1)) * innerW;
        const y = padT + (1 - (parsed[i].value - min) / (max - min)) * innerH;
        lineParts.push(`${x.toFixed(2)},${y.toFixed(2)}`);
      }

      const card = document.createElement("div");
      card.className = "chart-card";

      const t = document.createElement("p");
      t.className = "chart-title";
      t.textContent = title;
      card.appendChild(t);

      const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
      svg.setAttribute("class", "chart-svg");
      svg.setAttribute("viewBox", `0 0 ${width} ${height}`);

      const base = document.createElementNS("http://www.w3.org/2000/svg", "line");
      base.setAttribute("x1", String(padL));
      base.setAttribute("x2", String(width - padR));
      base.setAttribute("y1", String(height - padB));
      base.setAttribute("y2", String(height - padB));
      base.setAttribute("stroke", "#d0d7de");
      base.setAttribute("stroke-width", "1");
      svg.appendChild(base);

      const poly = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
      poly.setAttribute("points", lineParts.join(" "));
      poly.setAttribute("fill", "none");
      poly.setAttribute("stroke", color);
      poly.setAttribute("stroke-width", "2");
      svg.appendChild(poly);

      const ticks = buildAnnualDateTicks(parsed.map((item) => item.date));
      for (const tick of ticks) {
        const x = padL + ((tick.x - 1) / (n - 1)) * innerW;
        const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
        label.setAttribute("x", x.toFixed(2));
        label.setAttribute("y", String(height - 6));
        label.setAttribute("fill", "#57606a");
        label.setAttribute("font-size", "9");
        label.setAttribute("text-anchor", tick.x === 1 ? "start" : tick.x === n ? "end" : "middle");
        label.textContent = tick.label;
        svg.appendChild(label);
      }

      card.appendChild(svg);

      const first = parsed[0];
      const last = parsed[parsed.length - 1];
      const meta = document.createElement("p");
      meta.className = "chart-meta";
      meta.textContent =
        `样本点=${parsed.length}, 起始=${first.date}, 结束=${last.date}, ` +
        `最新值=${last.value.toFixed(4)}, 区间=[${min.toFixed(4)}, ${max.toFixed(4)}]`;
      card.appendChild(meta);
      return card;
    }

    function buildBarChartCard(title, items, color) {
      const parsed = [];
      for (const item of items) {
        if (!item || typeof item !== "object") continue;
        const label = String(item.label || "");
        const value = Number(item.value);
        if (!label || !Number.isFinite(value)) continue;
        parsed.push({ label, value });
      }
      if (parsed.length === 0) return null;

      let maxAbs = 0.0;
      for (const item of parsed) {
        const abs = Math.abs(item.value);
        if (abs > maxAbs) maxAbs = abs;
      }
      if (maxAbs < 1e-12) maxAbs = 1.0;

      const width = 420;
      const height = 140;
      const pad = 12;
      const innerW = width - pad * 2;
      const innerH = height - pad * 2;
      const barW = innerW / parsed.length;

      const card = document.createElement("div");
      card.className = "chart-card";

      const t = document.createElement("p");
      t.className = "chart-title";
      t.textContent = title;
      card.appendChild(t);

      const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
      svg.setAttribute("class", "chart-svg");
      svg.setAttribute("viewBox", `0 0 ${width} ${height}`);

      const baseY = pad + innerH / 2;
      const base = document.createElementNS("http://www.w3.org/2000/svg", "line");
      base.setAttribute("x1", String(pad));
      base.setAttribute("x2", String(width - pad));
      base.setAttribute("y1", String(baseY));
      base.setAttribute("y2", String(baseY));
      base.setAttribute("stroke", "#d0d7de");
      base.setAttribute("stroke-width", "1");
      svg.appendChild(base);

      for (let i = 0; i < parsed.length; i += 1) {
        const x = pad + i * barW + barW * 0.15;
        const w = barW * 0.7;
        const h = (Math.abs(parsed[i].value) / maxAbs) * (innerH * 0.45);
        const y = parsed[i].value >= 0 ? baseY - h : baseY;
        const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
        rect.setAttribute("x", x.toFixed(2));
        rect.setAttribute("y", y.toFixed(2));
        rect.setAttribute("width", w.toFixed(2));
        rect.setAttribute("height", h.toFixed(2));
        rect.setAttribute("fill", parsed[i].value >= 0 ? color : "#d1242f");
        rect.setAttribute("opacity", "0.85");
        svg.appendChild(rect);
      }

      card.appendChild(svg);

      const meta = document.createElement("p");
      meta.className = "chart-meta";
      meta.textContent =
        `样本点=${parsed.length}, 最大绝对值=${maxAbs.toFixed(4)}；` +
        `正值为上方柱，负值为下方柱。`;
      card.appendChild(meta);
      return card;
    }

    async function pollRun(runId) {
      const resp = await fetch(`/api/runs/${encodeURIComponent(runId)}`);
      const data = await resp.json();
      runIdEl.textContent = data.run_id || runId;
      runErrorEl.textContent = data.error || "";
      runOutputEl.textContent = data.output_dir ? `输出目录: ${data.output_dir}` : "";
      renderSummary(data.summary || {});
      renderVisualization(data.visualization || {});
      renderArtifacts(runId, data.artifact_paths || {});
      if (data.status === "succeeded") {
        setStatus("succeeded", true);
        if (pollTimer) clearTimeout(pollTimer);
        runBtn.disabled = false;
        return;
      }
      if (data.status === "failed") {
        setStatus("failed", false);
        if (pollTimer) clearTimeout(pollTimer);
        runBtn.disabled = false;
        return;
      }
      setStatus(data.status || "running");
      pollTimer = setTimeout(() => { pollRun(runId); }, 1500);
    }

    async function submitRun() {
      runErrorEl.textContent = "";
      runOutputEl.textContent = "";
      summaryWrap.innerHTML = "";
      visualizationWrap.innerHTML = "";
      artifactsWrap.innerHTML = "";
      submitMsg.textContent = "";

      const source = String(dataSourceSelect.value || "manual");
      if (!specFileInput.files || specFileInput.files.length === 0) {
        submitMsg.textContent = "请先选择 spec 文件。";
        return;
      }
      if (source === "tushare" && !String(tushareTokenInput.value || "").trim()) {
        submitMsg.textContent = "选择 Tushare 数据源时必须填写 Token。";
        return;
      }

      let assetLimit = null;
      const assetLimitText = String(dataAssetLimitInput.value || "").trim();
      if (assetLimitText) {
        const parsed = Number.parseInt(assetLimitText, 10);
        if (!Number.isFinite(parsed) || parsed <= 0) {
          submitMsg.textContent = "资产上限必须是正整数。";
          return;
        }
        assetLimit = parsed;
      }

      const specFile = specFileInput.files[0];
      const specText = await specFile.text();
      const body = {
        spec_filename: specFile.name,
        spec_text: specText,
        data_source: source,
        data_slice_preset: source === "manual"
          ? null
          : String(dataSlicePresetSelect.value || "standard"),
        data_start_date: source === "manual"
          ? null
          : String(dataStartDateInput.value || "").trim() || null,
        data_end_date: source === "manual"
          ? null
          : String(dataEndDateInput.value || "").trim() || null,
        data_asset_limit: source === "manual" ? null : assetLimit,
        tushare_token: source === "tushare"
          ? String(tushareTokenInput.value || "").trim()
          : null,
        evaluation_profile: profileSelect.value,
        output_root_dir: outputRootInput.value,
        render_report: Boolean(renderReportInput.checked),
      };

      runBtn.disabled = true;
      setStatus("submitting");
      const resp = await fetch("/api/runs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const data = await resp.json();
      if (!resp.ok) {
        setStatus("submit_failed", false);
        runErrorEl.textContent = data.error || "提交失败";
        runBtn.disabled = false;
        return;
      }
      const runId = data.run_id;
      if (!runId) {
        setStatus("submit_failed", false);
        runErrorEl.textContent = "后端返回缺少 run_id";
        runBtn.disabled = false;
        return;
      }
      submitMsg.textContent = "已提交，开始执行。";
      setStatus(data.status || "queued");
      if (pollTimer) clearTimeout(pollTimer);
      await pollRun(runId);
    }

    runBtn.addEventListener("click", () => {
      submitRun().catch((err) => {
        setStatus("client_error", false);
        runErrorEl.textContent = String(err);
        runBtn.disabled = false;
      });
    });
    dataSlicePresetSelect.addEventListener("change", () => {
      applySlicePreset(true);
    });
    dataSourceSelect.addEventListener("change", () => {
      syncSourceControls();
    });
    applySlicePreset(true);
    syncSourceControls();

    loadProfiles().catch((err) => {
      runErrorEl.textContent = `加载配置失败: ${String(err)}`;
    });
  </script>
</body>
</html>
"""
