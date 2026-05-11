"""Unified Research Frontend — single local HTTP server integrating
Knowledge Ops, Bridge Workspace, Validation Console, and Writeback Review.

Provides the ``start_unified_server`` entry-point.
"""

from __future__ import annotations

import datetime as dt
import difflib
import gc
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
import threading
import time
import traceback
import uuid
import webbrowser
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from csv import DictReader
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Literal, cast
from urllib.parse import parse_qs, unquote, urlparse

from alpha_lab.baseline_factor_suite import baseline_factor_suite_payload
from alpha_lab.custom_factors import (
    BUILTIN_FACTOR_NAMES,
    compile_custom_factor,
    custom_factor_meta_path,
    custom_factor_write_path,
    iter_custom_factor_meta_paths,
    load_persisted_custom_factors,
)
from alpha_lab.draft_model_validation import validate_draft_model_file
from alpha_lab.exceptions import (
    AlphaLabConfigError,
    AlphaLabDataError,
    AlphaLabExperimentError,
    AlphaLabIOError,
)
from alpha_lab.factor_recipe import factor_registry
from alpha_lab.model_candidates import (
    model_candidate_write_path,
    read_draft_model_source,
)
from alpha_lab.real_cases.common_io import (
    ensure_parquet_tabular_frame,
    resolve_tabular_frame_path,
)
from alpha_lab.real_cases.model_factor.spec import load_model_factor_case_spec
from alpha_lab.real_cases.single_factor.pipeline import (
    SingleFactorBatchParallelConfig,
    SingleFactorCaseRunResult,
    SingleFactorInputBundle,
    load_standard_inputs,
    run_single_factor_case,
)
from alpha_lab.real_cases.single_factor.spec import (
    SingleFactorCaseSpec,
    load_single_factor_case_spec,
)
from alpha_lab.reporting.renderers import write_case_report
from alpha_lab.research_bridge.categories import get_category_profile, list_categories
from alpha_lab.research_bridge.graph_view import VaultGraph
from alpha_lab.research_bridge.mechanism_index import (
    mechanism_index_status as bridge_mechanism_index_status,
)
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
from alpha_lab.research_evaluation_config import (
    AVAILABLE_RESEARCH_EVALUATION_PROFILES,
    CAMPAIGN_PROFILE_COMPARE_DEFAULTS,
    RESEARCH_EVALUATION_PROFILE_LABELS,
)
from alpha_lab.splits import preflight_split_contract, rebalance_frequency_to_step
from alpha_lab.vault_export import export_to_vault, resolve_vault_root

RunStatus = Literal["queued", "running", "succeeded", "failed", "cancelled"]
RunWorkflow = Literal["single_factor", "model_factor"]
_KNOWLEDGE_WRITEBACK_STAGES: frozenset[str] = frozenset({"stage2", "stage3", "run"})
_KNOWLEDGE_WRITEBACK_CARD_TYPES: frozenset[str] = frozenset(
    {
        "experiment_note",
        "mechanism_note",
        "validation_report",
        "failure_observation",
        "experiment_result",
    }
)
_KNOWLEDGE_WRITEBACK_TARGET_DIRS: dict[str, str] = {
    "experiment_note": "30_research_notes",
    "mechanism_note": "30_research_notes",
    "validation_report": "40_validation",
    "failure_observation": "60_failure_observations",
    "experiment_result": "50_experiments",
}
_KNOWLEDGE_WRITEBACK_FILENAME_PREFIXES: dict[str, str] = {
    "experiment_note": "Note",
    "mechanism_note": "Mechanism",
    "validation_report": "Validation",
    "failure_observation": "Failure",
    "experiment_result": "Experiment",
}
_MODEL_LAB_PROJECT_SLUG = "__model_lab__"
_WEB_SECRET_SETTINGS_REL_PATH = (
    Path(".research_bridge_cache") / "secret_settings.json"
)
_MODEL_LAB_COMPARE_METRIC_KEYS: tuple[str, ...] = (
    "model_family",
    "factor_verdict",
    "mean_ic",
    "ic_ir",
    "mean_rank_ic",
    "rank_ic_ir",
    "mean_long_short_turnover",
    "long_short_ir",
    "cost_aware_long_short_ir",
    "max_drawdown",
    "ls_max_drawdown",
    "coverage_mean",
)
_MODEL_LAB_MAX_COMPARE_RUNS: int = 8
_MODEL_LAB_SOURCE_SPECS: tuple[dict[str, str], ...] = (
    {
        "key": "core",
        "label": "model_factor/core/build.py",
        "path": "src/alpha_lab/model_factor/core/build.py",
        "description": "模型因子构建主循环：训练窗口、预测、训练日志都在这里串起来。",
        "focus": "先看 build_model_factor，以及训练窗口如何进入 fit/predict。",
    },
    {
        "key": "pipeline",
        "label": "real_cases/model_factor/pipeline/core.py",
        "path": "src/alpha_lab/real_cases/model_factor/pipeline/core.py",
        "description": "端到端 case 执行入口：读数据、训练、评估、导出 artifact。",
        "focus": "先看 run_model_factor_case 和 progress_callback 的阶段划分。",
    },
    {
        "key": "spec",
        "label": "real_cases/model_factor/spec.py",
        "path": "src/alpha_lab/real_cases/model_factor/spec.py",
        "description": "YAML/JSON spec 合同解析与路径解析。",
        "focus": "先看 ModelFactorCaseSpec、load_model_factor_case_spec、resolve_spec_paths。",
    },
    {
        "key": "artifacts",
        "label": "real_cases/model_factor/artifacts/core.py",
        "path": "src/alpha_lab/real_cases/model_factor/artifacts/core.py",
        "description": "metrics、training_log、feature_importance 等 artifact 的写出逻辑。",
        "focus": "先看 export_artifact_bundle 和 metrics_payload 的组织方式。",
    },
    {
        "key": "cli",
        "label": "real_cases/model_factor/cli.py",
        "path": "src/alpha_lab/real_cases/model_factor/cli.py",
        "description": "CLI 入口和运行参数，对应终端命令的行为。",
        "focus": "先看 main 和 run_model_factor_case 的调用方式。",
    },
    {
        "key": "model_lab_ui",
        "label": "web_model_lab.html",
        "path": "src/alpha_lab/web_model_lab.html",
        "description": "当前这个本地研究平台的前端模板。",
        "focus": "先看 spec 编辑器、run 队列、artifact/source viewer 的 JS。",
    },
)

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
_MODEL_LAB_BATCH_MAX_WORKERS: int = 3
_MODEL_LAB_BATCH_DEFAULT_WORKERS: int = 1
_MODEL_LAB_SUBPROCESS_POLL_SECONDS: float = 0.5
_RUN_OVERVIEW_MAX_CSV_ROWS: int = 20000

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

_ARTIFACT_FALLBACK_FILENAMES: dict[str, str] = {
    "research_tearsheet": "research_tearsheet.json",
    "research_tearsheet_pdf": "research_tearsheet.pdf",
    "metrics": "metrics.json",
    "summary": "summary.md",
}

# Names used to recover an artifact when the path stored on the run record
# is stale (e.g., output dir moved) or the key was never registered. Keys
# without an entry here fall back to "<key>.csv" / "<key>.json" heuristics.
_ARTIFACT_DISK_FILENAMES: dict[str, str] = {
    "research_tearsheet": "research_tearsheet.json",
    "research_tearsheet_pdf": "research_tearsheet.pdf",
    "metrics": "metrics.json",
    "summary": "summary.md",
    "case_report": "case_report.md",
    "experiment_card": "experiment_card.md",
    "integrity_report_markdown": "integrity_report.md",
    "integrity_report_json": "integrity_report.json",
    "portfolio_validation_markdown": "level2_portfolio_validation/portfolio_validation.md",
    "portfolio_validation_summary": "level2_portfolio_validation/portfolio_validation_summary.json",
    "portfolio_validation_metrics": "level2_portfolio_validation/portfolio_validation_metrics.json",
    "portfolio_validation_package": "level2_portfolio_validation/portfolio_validation_package.json",
    "signal_validation_json": "signal_validation.json",
    "backtest_result_json": "backtest_result.json",
    "group_returns": "group_returns.csv",
    "quantile_returns": "quantile_returns.csv",
    "ic_decay": "ic_decay.csv",
    "ic_significance": "ic_significance.json",
    "ic_timeseries": "ic_timeseries.csv",
    "factor_autocorrelation": "factor_autocorrelation.csv",
    "rolling_stability": "rolling_stability.csv",
    "turnover": "turnover.csv",
    "coverage": "coverage.csv",
    "purged_kfold_summary": "purged_kfold_summary.json",
    "purged_kfold_folds": "purged_kfold_folds.csv",
    "run_manifest": "run_manifest.json",
    "factor_definition": "factor_definition.md",
    "factor_definition_json": "factor_definition.json",
    "portfolio_recipe_json": "portfolio_recipe.json",
    "diagnostics": "diagnostics.json",
    "training_log": "training_log.csv",
    "feature_importance": "feature_importance.csv",
    "feature_importance_ledger": "feature_importance_ledger.csv",
    "model_definition_json": "model_definition.json",
    "feature_manifest_json": "feature_manifest.json",
}

# Some artifact keys refer to the same on-disk file under different names.
# When the requested key is missing/stale, try the synonym before giving up.
_ARTIFACT_KEY_SYNONYMS: dict[str, tuple[str, ...]] = {
    "group_returns": ("quantile_returns",),
    "quantile_returns": ("group_returns",),
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
    workflow: RunWorkflow = "single_factor"
    note: str | None = None
    draft_model_candidate_path: str | None = None
    draft_model_candidate_name: str | None = None
    draft_model_candidate_hash: str | None = None

    def _artifact_paths_for_api(self) -> dict[str, str]:
        """Return artifact paths that are actually retrievable by endpoint."""
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
    workflow: RunWorkflow = "single_factor"
    note: str | None = None
    draft_model_candidate_path: str | None = None
    draft_model_candidate_name: str | None = None
    draft_model_candidate_hash: str | None = None
    screening_retrain_every_n_dates: int | None = None


@dataclass(frozen=True)
class _SubprocessCaseRunResult:
    output_dir: Path
    artifact_paths: Mapping[str, Path]


class _ModelLabSubprocessError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        returncode: int | None,
        hint: str,
    ) -> None:
        super().__init__(message)
        self.returncode = returncode
        self.hint = hint


RunSuccessResult = SingleFactorCaseRunResult | _SubprocessCaseRunResult


@dataclass
class _InputBundleCacheEntry:
    bundle: SingleFactorInputBundle
    last_used_seq: int


class _RunStore:
    def __init__(self) -> None:
        self._records: dict[str, _RunRecord] = {}
        self._tasks: dict[str, _RunTask] = {}
        self._cancel_requests: set[str] = set()
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

    def _push_batch_progress(
        self,
        tasks: list[_RunTask],
        *,
        message: str,
        percent: int | None,
    ) -> None:
        for task in tasks:
            self._push_progress(task.run_id, message=message, percent=percent)

    def _execute_model_factor_subprocess_task(
        self,
        task: _RunTask,
        *,
        progress_callback: Any,
    ) -> _SubprocessCaseRunResult:
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
                    message=f"模型因子子进程仍在运行 pid={proc.pid}",
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
                spec = load_single_factor_case_spec(Path(task.spec_path).resolve())
                bundle, _ = self._load_cached_input_bundle(spec)
                run_result = run_single_factor_case(
                    spec,
                    output_root_dir=_resolve_single_factor_web_output_root_dir(
                        task,
                        spec=spec,
                    ),
                    evaluation_profile=task.evaluation_profile,
                    vault_export_mode="skip",
                    progress_callback=progress_callback,
                    input_bundle=bundle,
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
            if task.workflow == "model_factor":
                gc.collect()

    def _finalize_success(self, *, task: _RunTask, result: RunSuccessResult) -> None:
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
    elif isinstance(exc, _ModelLabSubprocessError):
        hint = exc.hint
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


def _parse_positive_int_env(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return None
    try:
        value = int(str(raw).strip())
    except ValueError:
        return None
    return value if value > 0 else None


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


def _build_model_lab_subprocess_env() -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("ALPHA_LAB_MODEL_LAB_CHILD", "1")
    source_root = str(Path(__file__).resolve().parents[1])
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


def _wrap_command_with_time(cmd: list[str]) -> list[str]:
    time_bin = shutil.which("time")
    if os.name != "nt" and time_bin:
        return [time_bin, "-v", *cmd]
    return list(cmd)


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
    match = re.search(r"Maximum resident set size \\(kbytes\\):\\s*(\\d+)", tail)
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


# ---------------------------------------------------------------------------
# Unified Service
# ---------------------------------------------------------------------------


class _UnifiedService:
    def __init__(self, *, vault_root: Path, workspace_root: Path) -> None:
        self.vault_root = vault_root.resolve()
        self.workspace_root = workspace_root.resolve()
        self.run_store = _RunStore()
        self._custom_factors_dir = self.workspace_root / "custom_factors"
        self._apply_saved_llm_settings()
        self._load_persisted_custom_factors()
        self._restore_model_lab_web_runs()

    @property
    def projects_root(self) -> Path:
        return (self.vault_root / PROJECTS_DIRNAME).resolve()

    @property
    def model_lab_specs_root(self) -> Path:
        return (self.workspace_root / "configs" / "real_cases" / "model_factor").resolve()

    @property
    def model_lab_candidates_root(self) -> Path:
        return (self.workspace_root / "model_candidates" / "research").resolve()

    @property
    def _secret_settings_path(self) -> Path:
        return (self.workspace_root / _WEB_SECRET_SETTINGS_REL_PATH).resolve()

    def _load_secret_settings(self) -> dict[str, object]:
        path = self._secret_settings_path
        if not path.exists():
            return {}
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return raw if isinstance(raw, dict) else {}

    def _write_secret_settings(self, payload: dict[str, object]) -> None:
        path = self._secret_settings_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        try:
            path.chmod(stat.S_IRUSR | stat.S_IWUSR)
        except OSError:
            pass

    def _apply_saved_llm_settings(self) -> None:
        settings = self._load_secret_settings()
        api_key = str(settings.get("anthropic_api_key") or "").strip()
        base_url = str(settings.get("anthropic_base_url") or "").strip()
        if api_key:
            os.environ["ANTHROPIC_API_KEY"] = api_key
        if base_url:
            os.environ["ANTHROPIC_BASE_URL"] = base_url
        if bool(settings.get("research_bridge_v2_enabled")):
            os.environ["ALPHA_LAB_RESEARCH_BRIDGE_V2"] = "1"

    def llm_settings_status(self) -> dict[str, object]:
        settings = self._load_secret_settings()
        saved_key = str(settings.get("anthropic_api_key") or "").strip()
        env_key = str(os.environ.get("ANTHROPIC_API_KEY") or "").strip()
        saved_base_url = str(settings.get("anthropic_base_url") or "").strip()
        env_base_url = str(os.environ.get("ANTHROPIC_BASE_URL") or "").strip()
        if saved_key:
            key_source = "saved"
        elif env_key:
            key_source = "env"
        else:
            key_source = "none"
        if saved_base_url:
            base_url_source = "saved"
        elif env_base_url:
            base_url_source = "env"
        else:
            base_url_source = "default"
        return {
            "ok": True,
            "anthropic_api_key_configured": bool(saved_key or env_key),
            "anthropic_api_key_source": key_source,
            "anthropic_base_url": saved_base_url or env_base_url,
            "anthropic_base_url_source": base_url_source,
            "research_bridge_v2_enabled": (
                os.environ.get("ALPHA_LAB_RESEARCH_BRIDGE_V2") == "1"
            ),
            "settings_path": str(self._secret_settings_path),
        }

    def mechanism_index_status(self) -> dict[str, object]:
        status = bridge_mechanism_index_status(
            workspace_root=self.workspace_root,
            vault_root=self.vault_root,
        )
        key_configured = bool(str(os.environ.get("ANTHROPIC_API_KEY") or "").strip())
        v2_enabled = os.environ.get("ALPHA_LAB_RESEARCH_BRIDGE_V2") == "1"
        return {
            **status,
            "anthropic_api_key_configured": key_configured,
            "research_bridge_v2_enabled": v2_enabled,
            "research_bridge_v2_active": v2_enabled and key_configured,
        }

    def save_llm_settings(self, payload: dict[str, object]) -> dict[str, object]:
        settings = self._load_secret_settings()
        existing_saved_key = str(settings.get("anthropic_api_key") or "").strip()
        existing_saved_base_url = str(settings.get("anthropic_base_url") or "").strip()
        raw_key = str(payload.get("anthropic_api_key") or "").strip()
        raw_base_url = str(payload.get("anthropic_base_url") or "").strip()
        if bool(payload.get("clear_anthropic_api_key")):
            settings.pop("anthropic_api_key", None)
            if (
                existing_saved_key
                and os.environ.get("ANTHROPIC_API_KEY") == existing_saved_key
            ):
                os.environ.pop("ANTHROPIC_API_KEY", None)
        elif raw_key:
            settings["anthropic_api_key"] = raw_key
            os.environ["ANTHROPIC_API_KEY"] = raw_key

        if bool(payload.get("clear_anthropic_base_url")):
            settings.pop("anthropic_base_url", None)
            if (
                existing_saved_base_url
                and os.environ.get("ANTHROPIC_BASE_URL") == existing_saved_base_url
            ):
                os.environ.pop("ANTHROPIC_BASE_URL", None)
        elif "anthropic_base_url" in payload:
            if raw_base_url:
                settings["anthropic_base_url"] = raw_base_url
                os.environ["ANTHROPIC_BASE_URL"] = raw_base_url
            else:
                settings.pop("anthropic_base_url", None)
                if (
                    existing_saved_base_url
                    and os.environ.get("ANTHROPIC_BASE_URL") == existing_saved_base_url
                ):
                    os.environ.pop("ANTHROPIC_BASE_URL", None)

        if "research_bridge_v2_enabled" in payload:
            v2_enabled = bool(payload.get("research_bridge_v2_enabled"))
            settings["research_bridge_v2_enabled"] = v2_enabled
            if v2_enabled:
                os.environ["ALPHA_LAB_RESEARCH_BRIDGE_V2"] = "1"
            else:
                os.environ.pop("ALPHA_LAB_RESEARCH_BRIDGE_V2", None)

        self._write_secret_settings(settings)
        return self.llm_settings_status()

    def _restore_model_lab_web_runs(self) -> None:
        web_runs_root = self.workspace_root / "outputs" / "real_cases" / "_web_runs"
        if not web_runs_root.exists():
            return
        for manifest_path in sorted(web_runs_root.glob("*/*/run_manifest.json")):
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                output_dir = manifest_path.parent.resolve()
                run_id = output_dir.parent.name
                case_name = str(manifest.get("case_name") or output_dir.name)
                outputs = manifest.get("outputs")
                artifact_paths = {
                    str(key): str(value)
                    for key, value in (outputs.items() if isinstance(outputs, Mapping) else [])
                    if key and value
                }
                artifact_paths.setdefault("run_manifest", str(manifest_path))
                metrics_path = output_dir / "metrics.json"
                summary = (
                    _extract_metrics_summary(metrics_path, run_status="succeeded")
                    if metrics_path.exists()
                    else {}
                )
                evaluation_standard = manifest.get("evaluation_standard")
                evaluation_profile = "default_research"
                if isinstance(evaluation_standard, Mapping):
                    evaluation_profile = str(
                        evaluation_standard.get("profile_name") or evaluation_profile
                    )
                submitted_at = str(
                    manifest.get("run_timestamp_utc")
                    or manifest.get("generated_at_utc")
                    or _utc_now_iso()
                )
                draft_model_source = (
                    manifest.get("draft_model_source")
                    if isinstance(manifest.get("draft_model_source"), Mapping)
                    else None
                )
                draft_model_candidate_path = (
                    _coerce_finite_or_text(draft_model_source.get("path"))
                    if isinstance(draft_model_source, Mapping)
                    else None
                )
                draft_model_candidate_name = (
                    _coerce_finite_or_text(draft_model_source.get("name"))
                    if isinstance(draft_model_source, Mapping)
                    else None
                )
                draft_model_candidate_hash = (
                    _coerce_finite_or_text(
                        draft_model_source.get("candidate_json_sha256")
                    )
                    if isinstance(draft_model_source, Mapping)
                    else None
                )
                record = _RunRecord(
                    run_id=run_id,
                    project_slug=_MODEL_LAB_PROJECT_SLUG,
                    case_name=case_name,
                    round_id=None,
                    spec_path=str(manifest.get("spec_path") or ""),
                    submitted_at_utc=submitted_at,
                    evaluation_profile=evaluation_profile,
                    output_root_dir=None,
                    render_report=True,
                    status="succeeded",
                    started_at_utc=submitted_at,
                    finished_at_utc=submitted_at,
                    updated_at_utc=submitted_at,
                    output_dir=str(output_dir),
                    progress_percent=100,
                    progress_message="已从本地产物恢复",
                    progress_events=[
                        {
                            "ts": submitted_at,
                            "message": "已从本地产物恢复",
                            "percent": 100,
                        }
                    ],
                    artifact_paths=artifact_paths,
                    summary=summary,
                    workflow="model_factor",
                    draft_model_candidate_path=draft_model_candidate_path,
                    draft_model_candidate_name=draft_model_candidate_name,
                    draft_model_candidate_hash=(
                        draft_model_candidate_hash[:12]
                        if draft_model_candidate_hash
                        else None
                    ),
                )
                self.run_store.restore_completed(record)
            except Exception:
                continue

    # ---- Dashboard --------------------------------------------------------

    def dashboard(self) -> dict[str, object]:
        projects = self.list_projects()
        records = self.run_store.list_records()
        status_counts: dict[str, int] = {"queued": 0, "running": 0, "succeeded": 0, "failed": 0}
        for record in records:
            status = str(record.status)
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
            "recent_runs": [record.to_compact_payload() for record in records[:10]],
            "next_actions": [
                {
                    "project_slug": project["slug"],
                    "next_action": project["next_action"],
                }
                for project in projects
                if str(project.get("next_action", "")).strip()
            ][:10],
        }

    # ---- Model Lab -------------------------------------------------------

    def list_model_lab_specs(self) -> list[dict[str, object]]:
        specs_root = self.model_lab_specs_root
        if not specs_root.exists():
            return []
        rows: list[dict[str, object]] = []
        for path in sorted(specs_root.iterdir()):
            if not path.is_file() or path.suffix.lower() not in {".yaml", ".yml", ".json"}:
                continue
            raw_spec = _read_yaml_document_safe(str(path))
            lineage_meta = _build_spec_lineage_meta(path, raw_spec)
            item: dict[str, object] = {
                "name": path.name,
                "path": str(path),
                "mtime_utc": _iso_from_timestamp(path.stat().st_mtime),
                "version": lineage_meta["version"],
                "lineage": lineage_meta["lineage"],
                "copied_from": lineage_meta["copied_from"],
                "file_signature": lineage_meta["file_signature"],
            }
            try:
                spec = load_model_factor_case_spec(path)
                item.update(
                    {
                        "valid": True,
                        "case_name": spec.name,
                        "factor_name": spec.factor_name,
                        "model_family": spec.model.family,
                        "feature_count": len(spec.feature_columns),
                        "target_horizon": int(spec.target.horizon),
                        "features_path": spec.features_path,
                        "prices_path": spec.prices_path,
                    }
                )
            except Exception as exc:
                item.update({"valid": False, "error": str(exc)})
            rows.append(item)
        return rows

    def read_model_lab_spec(self, spec_name: str) -> dict[str, object]:
        spec_path = self._resolve_model_lab_spec_path(spec_name)
        raw_spec = _read_yaml_document_safe(str(spec_path))
        lineage_meta = _build_spec_lineage_meta(spec_path, raw_spec)
        payload: dict[str, object] = {
            "name": spec_path.name,
            "path": str(spec_path),
            "content": _read_text_with_limit(spec_path, limit_bytes=_MAX_TEXT_BYTES),
            "size_bytes": spec_path.stat().st_size,
            "version": lineage_meta["version"],
            "lineage": lineage_meta["lineage"],
            "copied_from": lineage_meta["copied_from"],
            "file_signature": lineage_meta["file_signature"],
            "mtime_utc": _iso_from_timestamp(spec_path.stat().st_mtime),
        }
        try:
            spec = load_model_factor_case_spec(spec_path)
            payload["meta"] = {
                "case_name": spec.name,
                "factor_name": spec.factor_name,
                "model_family": spec.model.family,
                "feature_count": len(spec.feature_columns),
                "target_horizon": int(spec.target.horizon),
                "output_root_dir": str(spec.output.root_dir),
                "version": lineage_meta["version"],
                "lineage": lineage_meta["lineage"],
                "copied_from": lineage_meta["copied_from"],
                "file_signature": lineage_meta["file_signature"],
                "updated_at_utc": _iso_from_timestamp(spec_path.stat().st_mtime),
                "feature_preprocess": {
                    "missing_policy": spec.feature_preprocess.missing_policy,
                    "scale_features": spec.feature_preprocess.scale_features,
                    "cross_sectional_transform": spec.feature_preprocess.cross_sectional_transform,
                    "cross_sectional_group_scope": (
                        spec.feature_preprocess.cross_sectional_group_scope
                    ),
                    "industry_group_column": spec.feature_preprocess.industry_group_column,
                },
                "model_selection": {
                    "enabled": spec.model_selection.enabled,
                    "n_splits": spec.model_selection.n_splits,
                    "embargo_pct": spec.model_selection.embargo_pct,
                    "metric": spec.model_selection.metric,
                    "turnover_penalty_lambda": spec.model_selection.turnover_penalty_lambda,
                    "turnover_bucket_quantile": spec.model_selection.turnover_bucket_quantile,
                    "candidate_count": len(spec.model_selection.candidates),
                    "candidate_families": sorted(
                        {candidate.family for candidate in spec.model_selection.candidates}
                    ),
                },
            }
        except Exception as exc:
            payload["meta"] = {"valid": False, "error": str(exc)}
        return payload

    def list_model_lab_sources(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for item in _MODEL_LAB_SOURCE_SPECS:
            try:
                path = self._resolve_model_lab_source_path(item["key"])
                path_text = str(path)
                exists = path.exists()
            except FileNotFoundError:
                path_text = str((self.workspace_root / item["path"]).resolve())
                exists = False
            rows.append(
                {
                    "key": item["key"],
                    "label": item["label"],
                    "path": path_text,
                    "description": item["description"],
                    "focus": item["focus"],
                    "exists": exists,
                }
            )
        return rows

    def list_model_lab_candidates(self) -> list[dict[str, object]]:
        root = self.model_lab_candidates_root
        if not root.exists():
            return []
        rows: list[dict[str, object]] = []
        for path in sorted(root.glob("*/model_candidate.json")):
            if not path.is_file():
                continue
            try:
                rows.append(self._model_lab_candidate_summary(path))
            except Exception as exc:  # noqa: BLE001
                rows.append(
                    {
                        "name": path.parent.name,
                        "path": str(path),
                        "valid": False,
                        "validation_status": "failed",
                        "error": str(exc),
                    }
                )
        return rows

    def read_model_lab_candidate(self, candidate: str) -> dict[str, object]:
        path = self._resolve_model_lab_candidate_path(candidate)
        return {
            **self._model_lab_candidate_summary(path),
            "content": _read_text_with_limit(path, limit_bytes=_MAX_TEXT_BYTES),
            "research_log": _read_text_with_limit(
                path.with_name("research_log.md"),
                limit_bytes=_MAX_TEXT_BYTES,
            ),
        }

    def save_model_lab_candidate(self, payload: dict[str, object]) -> dict[str, object]:
        candidate_payload = _extract_model_candidate_payload(payload)
        candidate_name = _safe_model_candidate_name(
            str(candidate_payload.get("candidate_name") or "")
        )
        target = self._resolve_model_lab_candidate_path(candidate_name, require_exists=False)
        overwrite = bool(payload.get("overwrite", True))
        if target.exists() and not overwrite:
            raise FileExistsError(f"model candidate already exists: {candidate_name}")
        target.parent.mkdir(parents=True, exist_ok=True)
        rendered = json.dumps(
            candidate_payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        target.write_text(rendered + "\n", encoding="utf-8")
        source = read_draft_model_source(target)
        self._append_model_candidate_research_log(
            candidate_name,
            "created/imported",
            f"candidate_json_sha256={source.candidate_json_sha256}",
        )
        return {"ok": True, **self.read_model_lab_candidate(candidate_name)}

    def validate_model_lab_candidate(
        self,
        candidate: str,
        payload: dict[str, object] | None = None,
    ) -> dict[str, object]:
        payload = payload or {}
        path = self._resolve_model_lab_candidate_path(candidate)
        available_fields = _coerce_available_fields(payload.get("available_fields"))
        result = validate_draft_model_file(
            path,
            available_fields=available_fields,
            require_features_file=not bool(payload.get("skip_features_file_check", False)),
        )
        result_payload = result.to_payload()
        if result.ok:
            detail = (
                f"candidate_json_sha256={result.candidate_json_sha256} "
                f"case_spec_sha256={result.case_spec_sha256} "
                f"feature_contract_sha256={result.feature_contract_sha256}"
            )
            event = "validated"
        else:
            codes = ",".join(str(item.code) for item in result.errors)
            detail = f"error_codes={codes or 'unknown'}"
            event = "failed"
        self._append_model_candidate_research_log(candidate, event, detail)
        return result_payload

    def materialize_model_lab_candidate_spec(
        self,
        candidate: str,
        payload: dict[str, object] | None = None,
    ) -> dict[str, object]:
        payload = payload or {}
        path = self._resolve_model_lab_candidate_path(candidate)
        validation = validate_draft_model_file(path)
        if not validation.ok:
            return {"ok": False, "validation": validation.to_payload()}
        raw_payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw_payload, dict):
            raise ValueError("model_candidate.json root must be an object")
        case_spec_payload = raw_payload.get("case_spec_payload")
        if not isinstance(case_spec_payload, dict):
            raise ValueError("case_spec_payload must be an object")
        target_path = self._next_model_lab_candidate_spec_path(
            candidate,
            target_name=_optional_text(payload.get("target_name")),
            overwrite=bool(payload.get("overwrite", False)),
        )
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(
            _dump_spec_payload(case_spec_payload, target_path.suffix),
            encoding="utf-8",
        )
        spec = load_model_factor_case_spec(target_path)
        self._append_model_candidate_research_log(
            candidate,
            "materialized",
            f"case={target_path.name}",
        )
        return {
            "ok": True,
            "candidate": candidate,
            "name": target_path.name,
            "path": str(target_path),
            "case_name": spec.name,
            "factor_name": spec.factor_name,
            "model_family": spec.model.family,
            "feature_count": len(spec.feature_columns),
            "validation": validation.to_payload(),
        }

    def run_model_lab_candidate(
        self,
        candidate: str,
        payload: dict[str, object] | None = None,
    ) -> dict[str, object]:
        payload = payload or {}
        validation = self.validate_model_lab_candidate(candidate, payload)
        if not bool(validation.get("ok")):
            return {"ok": False, "validation": validation}
        materialized = self.materialize_model_lab_candidate_spec(candidate, payload)
        if not bool(materialized.get("ok")):
            return {"ok": False, "validation": validation, "materialized": materialized}
        candidate_path = self._resolve_model_lab_candidate_path(candidate)
        run_payload = {
            "spec_name": materialized["name"],
            "evaluation_profile": str(
                payload.get("evaluation_profile") or "default_research"
            ),
            "screening_retrain_every_n_dates": _as_int(
                payload.get("screening_retrain_every_n_dates"),
                default=0,
            )
            or None,
            "vault_export_mode": str(payload.get("vault_export_mode") or "skip"),
            "render_report": bool(payload.get("render_report", True)),
            "output_root_dir": _optional_text(payload.get("output_root_dir")),
            "note": _optional_text(payload.get("note")) or f"draft:{candidate}",
            "draft_model_candidate_path": str(candidate_path),
        }
        submitted = self.submit_model_lab_run(run_payload)
        self._append_model_candidate_research_log(
            candidate,
            "run_submitted",
            "case="
            f"{materialized['name']} run={submitted['run_id']} "
            f"profile={run_payload['evaluation_profile']}",
        )
        return {
            "ok": True,
            "candidate": candidate,
            "validation": validation,
            "materialized": materialized,
            "run": submitted,
        }

    def read_model_lab_source(self, source_key: str) -> dict[str, object]:
        item = next((row for row in _MODEL_LAB_SOURCE_SPECS if row["key"] == source_key), None)
        if item is None:
            raise FileNotFoundError(f"model-lab source not found: {source_key}")
        path = self._resolve_model_lab_source_path(source_key)
        text = _read_text_with_limit(path, limit_bytes=_MAX_REPORT_TEXT_BYTES)
        return {
            "key": item["key"],
            "label": item["label"],
            "description": item["description"],
            "focus": item["focus"],
            "path": str(path),
            "content": text,
            "size_bytes": path.stat().st_size,
            "line_count": text.count("\n") + (0 if not text else 1),
        }

    def update_model_lab_spec(
        self,
        spec_name: str,
        payload: dict[str, object],
    ) -> dict[str, object]:
        spec_path = self._resolve_model_lab_spec_path(spec_name)
        content = str(payload.get("content") or "")
        if not content.strip():
            raise ValueError("content is required")
        temp_path = Path("/tmp") / f"alpha_lab_model_lab_{uuid.uuid4().hex}{spec_path.suffix}"
        temp_path.write_text(content, encoding="utf-8")
        try:
            spec = load_model_factor_case_spec(temp_path)
        finally:
            temp_path.unlink(missing_ok=True)
        spec_path.write_text(content, encoding="utf-8")
        return {
            "ok": True,
            "name": spec_path.name,
            "case_name": spec.name,
            "factor_name": spec.factor_name,
            "model_family": spec.model.family,
            "feature_count": len(spec.feature_columns),
        }

    def submit_model_lab_run(self, payload: dict[str, object]) -> dict[str, object]:
        spec_name = str(payload.get("spec_name") or "").strip()
        if not spec_name:
            raise ValueError("spec_name is required")
        spec_path = self._resolve_model_lab_spec_path(spec_name)
        spec = load_model_factor_case_spec(spec_path)
        _preflight_model_lab_spec_inputs(spec)
        draft_model_candidate_path = _optional_text(payload.get("draft_model_candidate_path"))
        draft_model_candidate_name: str | None = None
        draft_model_candidate_hash: str | None = None
        if draft_model_candidate_path is not None:
            candidate_path = Path(draft_model_candidate_path).expanduser().resolve()
            try:
                candidate_path.relative_to(self.model_lab_candidates_root)
            except ValueError as exc:
                raise PermissionError(
                    "draft_model_candidate_path must be under research candidates"
                ) from exc
            if candidate_path.name != "model_candidate.json" or not candidate_path.is_file():
                raise FileNotFoundError(
                    f"draft model candidate not found: {draft_model_candidate_path}"
                )
            source = read_draft_model_source(candidate_path)
            draft_model_candidate_path = str(candidate_path)
            draft_model_candidate_name = source.name
            draft_model_candidate_hash = source.candidate_json_sha256[:12]
        task = _RunTask(
            run_id=uuid.uuid4().hex,
            project_slug=_MODEL_LAB_PROJECT_SLUG,
            case_name=spec.name,
            round_id=None,
            spec_path=str(spec_path),
            evaluation_profile=str(payload.get("evaluation_profile") or "default_research"),
            output_root_dir=_optional_text(payload.get("output_root_dir")),
            render_report=bool(payload.get("render_report", True)),
            workflow="model_factor",
            note=_optional_text(payload.get("note")),
            draft_model_candidate_path=draft_model_candidate_path,
            draft_model_candidate_name=draft_model_candidate_name,
            draft_model_candidate_hash=draft_model_candidate_hash,
            screening_retrain_every_n_dates=(
                _as_int(payload.get("screening_retrain_every_n_dates"), default=0)
                or None
            ),
        )
        submitted = self.run_store.submit(task).to_payload()
        return {"ok": True, **submitted}

    def duplicate_model_lab_spec(
        self,
        spec_name: str,
        payload: dict[str, object],
    ) -> dict[str, object]:
        source_path = self._resolve_model_lab_spec_path(spec_name)
        target_name = _optional_text(payload.get("target_name"))
        if target_name is None:
            raise ValueError("target_name is required")
        target_name = _safe_spec_filename(target_name)
        overwrite = bool(payload.get("overwrite", False))
        target_path = (self.model_lab_specs_root / target_name).resolve()
        root = self.model_lab_specs_root.resolve()
        if not str(target_path).startswith(str(root) + "/") and target_path != root:
            raise PermissionError("invalid target spec path")
        if target_path.exists() and not overwrite:
            raise FileNotFoundError(f"target spec already exists: {target_name}")
        source_payload = _read_yaml_document_safe(str(source_path))
        if isinstance(source_payload, dict):
            payload_copy: dict[str, object] = {
                key: value for key, value in source_payload.items() if isinstance(key, str)
            }
            sync_identifiers = bool(payload.get("sync_identifiers", True))
            sync_factor_name = bool(payload.get("sync_factor_name", True))
            target_stem = Path(target_path.name).stem
            if sync_identifiers:
                payload_copy["name"] = target_stem
                if sync_factor_name:
                    payload_copy["factor_name"] = _derive_factor_name_from_spec_stem(target_stem)
            source_lineage = _extract_spec_lineage(payload_copy)
            source_version = _coerce_spec_version(payload_copy.get("version"))
            payload_copy["copied_from"] = source_path.name
            payload_copy["lineage"] = {
                **source_lineage,
                "copied_from": source_path.name,
                "copied_at": _utc_now_iso(),
                "source_version": str(source_version) if source_version is not None else "",
            }
            payload_copy["version"] = _next_spec_version(source_version)
            target_path.write_text(
                _dump_spec_payload(payload_copy, target_path.suffix.lower()),
                encoding="utf-8",
            )
        else:
            target_path.write_text(source_path.read_text(encoding="utf-8"), encoding="utf-8")
        return {
            "ok": True,
            "source": source_path.name,
            "name": target_path.name,
            "path": str(target_path),
            "overwrite": overwrite,
        }

    def delete_model_lab_spec(self, spec_name: str) -> dict[str, object]:
        spec_path = self._resolve_model_lab_spec_path(spec_name)
        if spec_path.suffix.lower() not in {".yaml", ".yml"}:
            raise ValueError("仅支持删除 .yaml/.yml spec 文件")

        resolved_spec_path = spec_path.resolve()
        blocking_runs: list[dict[str, str]] = []
        for run in self.run_store.list_records(workflow="model_factor"):
            run_spec_path = Path(str(run.spec_path)).expanduser().resolve(strict=False)
            if run_spec_path != resolved_spec_path:
                continue
            if run.status not in {"queued", "running"}:
                continue
            blocking_runs.append(
                {
                    "run_id": run.run_id,
                    "status": run.status,
                }
            )
        if blocking_runs:
            preview = ", ".join(
                f"{item['run_id'][:10]}({item['status']})" for item in blocking_runs[:5]
            )
            more = "" if len(blocking_runs) <= 5 else f" +{len(blocking_runs) - 5}"
            raise ValueError(
                "该 spec 正被排队/运行中的 run 引用，无法删除；请先取消对应 run："
                f" {preview}{more}"
            )

        spec_path.unlink(missing_ok=False)
        remaining_specs = self.list_model_lab_specs()
        next_spec_name = str(remaining_specs[0].get("name") or "") if remaining_specs else ""
        return {
            "ok": True,
            "deleted": True,
            "name": spec_path.name,
            "path": str(spec_path),
            "remaining_count": len(remaining_specs),
            "next_spec_name": next_spec_name,
        }

    def diff_model_lab_specs(self, payload: dict[str, object]) -> dict[str, object]:
        left_name = _optional_text(payload.get("left"))
        right_name = _optional_text(payload.get("right"))
        if left_name is None or right_name is None:
            raise ValueError("left and right spec names are required")
        ignore_metadata = bool(payload.get("ignore_metadata", True))
        left_path = self._resolve_model_lab_spec_path(left_name)
        right_path = self._resolve_model_lab_spec_path(right_name)
        left_text = left_path.read_text(encoding="utf-8").splitlines()
        right_text = right_path.read_text(encoding="utf-8").splitlines()
        semantic_equal_ignoring_meta = False
        left_payload = _read_yaml_document_safe(str(left_path))
        right_payload = _read_yaml_document_safe(str(right_path))
        if ignore_metadata and isinstance(left_payload, dict) and isinstance(right_payload, dict):
            semantic_equal_ignoring_meta = _strip_spec_diff_metadata(
                left_payload
            ) == _strip_spec_diff_metadata(right_payload)
        unified = "\n".join(
            difflib.unified_diff(
                left_text,
                right_text,
                fromfile=left_path.name,
                tofile=right_path.name,
                lineterm="",
            )
        )
        if semantic_equal_ignoring_meta:
            unified = ""
        return {
            "ok": True,
            "left": left_path.name,
            "right": right_path.name,
            "unified": unified,
            "has_difference": bool(unified.strip()),
            "semantic_equal_ignoring_metadata": semantic_equal_ignoring_meta,
            "ignore_metadata": ignore_metadata,
        }

    def compare_model_lab_runs(self, payload: dict[str, object]) -> dict[str, object]:
        run_ids_raw = payload.get("run_ids")
        if not isinstance(run_ids_raw, list):
            raise ValueError("run_ids must be a list")
        run_ids = [str(item).strip() for item in run_ids_raw if str(item).strip()]
        if len(run_ids) < 2:
            raise ValueError("at least 2 run ids are required")
        if len(run_ids) > _MODEL_LAB_MAX_COMPARE_RUNS:
            raise ValueError(
                f"最多支持 {_MODEL_LAB_MAX_COMPARE_RUNS} 个 run 对比"
            )

        seen: set[str] = set()
        ordered_run_ids: list[str] = []
        for run_id in run_ids:
            if run_id in seen:
                continue
            seen.add(run_id)
            ordered_run_ids.append(run_id)

        records: list[_RunRecord] = []
        for run_id in ordered_run_ids:
            run = self.get_model_lab_run(run_id)
            if run.workflow != "model_factor":
                raise ValueError(f"非 model-lab run 不支持对比: {run_id}")
            records.append(run)

        top_k_features = _as_int(payload.get("top_k_features"), default=20)
        if top_k_features <= 0:
            top_k_features = 20
        top_k_features = max(1, min(top_k_features, 200))

        top_features_by_run: dict[str, list[str]] = {}
        metric_rows: list[dict[str, object]] = []
        ic_series_by_run: dict[str, dict[str, float]] = {}
        turnover_series_by_run: dict[str, dict[str, float]] = {}
        failure_rows: list[dict[str, object]] = []
        leakage_rows: list[dict[str, object]] = []

        collected_by_run_id: dict[str, dict[str, object]] = {}
        max_workers = min(len(records), _MODEL_LAB_MAX_COMPARE_RUNS)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_to_run_id = {
                pool.submit(
                    _collect_model_lab_run_compare_payload,
                    run,
                    top_k_features,
                ): run.run_id
                for run in records
            }
            for future in future_to_run_id:
                run_id = future_to_run_id[future]
                collected_by_run_id[run_id] = future.result()

        for run in records:
            collected = collected_by_run_id[run.run_id]
            top_features_by_run[run.run_id] = cast(
                list[str], collected["top_features"]
            )
            failure_rows.append(
                cast(dict[str, object], collected["failure_snapshot"])
            )
            metric_rows.append(cast(dict[str, object], collected["metric_row"]))
            ic_series_by_run[run.run_id] = cast(
                dict[str, float], collected["ic_series"]
            )
            turnover_series_by_run[run.run_id] = cast(
                dict[str, float], collected["turnover_series"]
            )
            leakage_rows.append(cast(dict[str, object], collected["leakage"]))

        comparison = _build_top_feature_stability(top_features_by_run, run_count=len(records))
        compare_dates = _build_rank_ic_merge_rows(ic_series_by_run)
        turnover_dates = _build_metric_timeseries_rows(turnover_series_by_run)
        severity_by_run: dict[str, str] = {}
        for item in leakage_rows:
            integrity_summary = item.get("integrity_summary")
            run_id_value = item.get("run_id")
            highest_severity: object = "pass"
            if isinstance(integrity_summary, Mapping):
                highest_severity = (
                    integrity_summary.get("highest_severity") or "pass"
                )
            severity_by_run[str(run_id_value or "")] = str(highest_severity)
        return {
            "ok": True,
            "run_count": len(records),
            "requested_run_count": len(run_ids),
            "run_ids": [run.run_id for run in records],
            "case_names": [run.case_name for run in records],
            "case_name_by_run_id": {run.run_id: run.case_name for run in records},
            "run_failures": failure_rows,
            "metric_columns": list(_MODEL_LAB_COMPARE_METRIC_KEYS),
            "metric_rows": metric_rows,
            "top_features_by_run": top_features_by_run,
            "feature_stability": comparison,
            "spec_diff": _build_model_lab_run_spec_diff(records),
            "ic_series": compare_dates,
            "turnover_series": turnover_dates,
            "leakage": {
                "runs": leakage_rows,
                "top_k_features": top_k_features,
                "severity_by_run": severity_by_run,
            },
        }

    def list_model_lab_runs(
        self,
        *,
        compact: bool = False,
        status_filter: str | None = None,
        case_filter: str | None = None,
        note_filter: str | None = None,
    ) -> list[dict[str, object]]:
        records = self.run_store.list_records(workflow="model_factor")
        status = (status_filter or "").strip().lower()
        case = (case_filter or "").strip().lower()
        note = (note_filter or "").strip().lower()
        if status:
            records = [item for item in records if str(item.status).lower() == status]
        if case:
            records = [item for item in records if case in str(item.case_name).lower()]
        if note:
            records = [item for item in records if note in str(item.note or "").lower()]
        payloads: list[dict[str, object]] = []
        for item in records:
            row = item.to_compact_payload() if compact else item.to_payload()
            draft_source = _load_run_draft_model_source(item)
            if draft_source is not None:
                row["draft_model_source"] = draft_source
                row["draft_model_candidate_name"] = str(
                    draft_source.get("name") or row.get("draft_model_candidate_name") or ""
                )
                row["draft_model_candidate_hash"] = str(
                    draft_source.get("candidate_json_sha256")
                    or row.get("draft_model_candidate_hash")
                    or ""
                )[:12]
            summary = _ensure_run_summary(item)
            action, next_step = _derive_evaluation_action_and_next_step(
                summary,
                run_status=item.status,
            )
            row["summary"] = _compact_metrics_summary(summary) if compact else dict(summary)
            row["factor_name"] = _resolve_run_factor_label(item)
            row["evaluation_title"] = _resolve_run_evaluation_title(item)
            row["evaluation_action"] = (
                _coerce_finite_or_text(summary.get("evaluation_action")) or action
            )
            row["evaluation_next_step"] = (
                _coerce_finite_or_text(summary.get("evaluation_next_step")) or next_step
            )
            payloads.append(row)
        return payloads

    def get_model_lab_run(self, run_id: str) -> _RunRecord:
        run = self.run_store.get(run_id)
        if run is None or run.workflow != "model_factor":
            raise FileNotFoundError(f"model-lab run not found: {run_id}")
        return run

    def delete_model_lab_run(self, run_id: str) -> dict[str, object]:
        record = self.get_model_lab_run(run_id)
        outcome = self.run_store.request_cancel_and_delete(run_id)
        deleted_paths: list[str] = []
        if outcome.get("immediate") and outcome.get("output_dir"):
            if _safe_rmtree(str(outcome["output_dir"])):
                deleted_paths.append(str(outcome["output_dir"]))
        return {
            "ok": True,
            "run_id": run_id,
            "prior_status": record.status,
            "cancelled": not bool(outcome.get("immediate")),
            "deleted_paths": deleted_paths,
            "message": (
                "已请求取消：当前阶段结束后将自动清理产物。"
                if not outcome.get("immediate")
                else "已删除。"
            ),
        }

    def export_model_lab_run_experiment_card(
        self,
        *,
        run_id: str,
        mode: str = "versioned",
    ) -> dict[str, object]:
        run = self.get_model_lab_run(run_id)
        if run.workflow != "model_factor":
            raise ValueError("only model_factor runs support experiment-card export")
        if run.status != "succeeded":
            raise ValueError("run must be succeeded before exporting experiment card")

        source_paths: dict[str, str | Path | None] = {
            "experiment_card_path": _resolve_run_artifact_path(
                run,
                artifact_key="experiment_card",
                fallback_name="experiment_card.md",
            ),
            "summary_path": _resolve_run_artifact_path(
                run,
                artifact_key="summary",
                fallback_name="summary.md",
            ),
            "manifest_path": _resolve_run_artifact_path(
                run,
                artifact_key="run_manifest",
                fallback_name="run_manifest.json",
            ),
        }
        result = export_to_vault(
            source_paths=source_paths,
            case_name=run.case_name,
            vault_root=self.vault_root,
            mode=mode,
        )
        return {
            "ok": result.success,
            "run_id": run_id,
            "case_name": run.case_name,
            "status": result.status,
            "success": result.success,
            "target_paths": list(result.target_paths),
            "mode_used": result.mode_used,
            "error": result.error,
        }

    def _resolve_model_lab_spec_path(self, spec_name: str) -> Path:
        raw = str(spec_name or "").strip()
        if not raw:
            raise ValueError("spec_name must be non-empty")
        candidate = (self.model_lab_specs_root / raw).resolve()
        root = self.model_lab_specs_root
        if not str(candidate).startswith(str(root)):
            raise PermissionError("invalid spec path")
        if not candidate.exists() or not candidate.is_file():
            raise FileNotFoundError(f"model-lab spec not found: {raw}")
        return candidate

    def _resolve_model_lab_candidate_path(
        self,
        candidate: str,
        *,
        require_exists: bool = True,
    ) -> Path:
        name = _safe_model_candidate_name(candidate)
        expected = model_candidate_write_path(self.workspace_root, name).resolve()
        root = self.model_lab_candidates_root.resolve()
        try:
            expected.relative_to(root)
        except ValueError as exc:
            raise PermissionError("invalid model candidate path") from exc
        if require_exists and (not expected.exists() or not expected.is_file()):
            raise FileNotFoundError(f"model candidate not found: {name}")
        return expected

    def _model_lab_candidate_summary(self, path: Path) -> dict[str, object]:
        path = path.expanduser().resolve()
        source = read_draft_model_source(path)
        validation = validate_draft_model_file(path, require_features_file=False)
        audit = source.to_audit_dict()
        return {
            "name": source.name,
            "path": str(path),
            "mtime_utc": _iso_from_timestamp(path.stat().st_mtime),
            "valid": validation.ok,
            "validation_status": "ok" if validation.ok else "failed",
            "validation": validation.to_payload(),
            "model_family": source.model_family or "",
            "feature_count": len(source.feature_columns),
            "feature_columns": list(source.feature_columns),
            "candidate_json_sha256": source.candidate_json_sha256,
            "case_spec_sha256": source.case_spec_sha256,
            "feature_contract_sha256": source.feature_contract_sha256,
            "candidate_json_sha256_short": source.candidate_json_sha256[:12],
            "case_spec_sha256_short": source.case_spec_sha256[:12],
            "feature_contract_sha256_short": source.feature_contract_sha256[:12],
            "audit": audit,
        }

    def _next_model_lab_candidate_spec_path(
        self,
        candidate: str,
        *,
        target_name: str | None,
        overwrite: bool,
    ) -> Path:
        specs_root = self.model_lab_specs_root.resolve()
        specs_root.mkdir(parents=True, exist_ok=True)
        if target_name is not None:
            safe_name = _safe_candidate_case_filename(target_name)
            target = (specs_root / safe_name).resolve()
            try:
                target.relative_to(specs_root)
            except ValueError as exc:
                raise PermissionError("invalid target spec path") from exc
            if target.exists() and not overwrite:
                raise FileExistsError(f"target spec already exists: {target.name}")
            return target

        candidate_name = _safe_model_candidate_name(candidate)
        version = 1
        while True:
            target = (specs_root / f"{candidate_name}_v{version}.yaml").resolve()
            if overwrite or not target.exists():
                return target
            version += 1

    def _append_model_candidate_research_log(
        self,
        candidate: str,
        event: str,
        detail: str = "",
    ) -> None:
        path = self._resolve_model_lab_candidate_path(candidate, require_exists=False)
        path.parent.mkdir(parents=True, exist_ok=True)
        log_path = path.with_name("research_log.md")
        line = f"- {_utc_now_iso()} {event}"
        if detail.strip():
            line = f"{line} {detail.strip()}"
        prior = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
        if prior and not prior.endswith("\n"):
            prior += "\n"
        log_path.write_text(prior + line + "\n", encoding="utf-8")

    def _resolve_model_lab_source_path(self, source_key: str) -> Path:
        item = next((row for row in _MODEL_LAB_SOURCE_SPECS if row["key"] == source_key), None)
        if item is None:
            raise FileNotFoundError(f"model-lab source not found: {source_key}")
        repo_root = Path(__file__).resolve().parents[2]
        candidates = [
            (self.workspace_root / item["path"]).resolve(),
            (repo_root / item["path"]).resolve(),
        ]
        allowed_roots = [self.workspace_root.resolve(), repo_root.resolve()]
        for candidate in candidates:
            if not any(str(candidate).startswith(str(root)) for root in allowed_roots):
                continue
            if candidate.exists() and candidate.is_file():
                return candidate
        raise FileNotFoundError(f"model-lab source file not found for key={source_key}")

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

    def create_idea_distribute(
        self,
        idea: str,
        *,
        lab: str = "single_factor",
        engines: list[str] | tuple[str, ...] | str | None = None,
        top_k: int = 8,
    ) -> dict[str, object]:
        """Stage 0 entry exposed to the unified frontend.

        Calls :func:`alpha_lab.research_bridge.service.distribute_idea` and
        returns the 5-file layout. Each file's content is read back and
        embedded in the response so the frontend can preview + copy without
        a follow-up GET per file.
        """

        from alpha_lab.research_bridge.engine_prompts import Lab
        from alpha_lab.research_bridge.service import distribute_idea

        if not idea.strip():
            return {"ok": False, "error": "idea must be non-empty"}
        try:
            target_lab = Lab(lab)
        except ValueError:
            return {
                "ok": False,
                "error": (
                    f"lab must be one of {[lab_value.value for lab_value in Lab]}; "
                    f"got {lab!r}"
                ),
            }
        try:
            result = distribute_idea(
                vault_root=self.vault_root,
                idea=idea,
                engines=engines,
                lab=target_lab,
                workspace_root=self.workspace_root,
                top_k=top_k,
            )
        except (ValueError, FileExistsError, OSError) as exc:
            return {"ok": False, "error": str(exc)}

        files: list[dict[str, str]] = []
        claude_engine = next((e for e in result.engines if e.value == "claude"), None)
        codex_engine = next((e for e in result.engines if e.value == "codex"), None)
        for label, path in (
            ("manifest.json", result.manifest_path),
            ("retrieval_pack.md", result.retrieval_pack_path),
            (
                "prompt_claude.md",
                result.engine_prompt_paths.get(claude_engine) if claude_engine else None,
            ),
            (
                "prompt_codex.md",
                result.engine_prompt_paths.get(codex_engine) if codex_engine else None,
            ),
            ("stage2_input.md", result.stage2_input_path),
        ):
            if path is None:
                continue
            try:
                content = Path(path).read_text(encoding="utf-8")
            except OSError as exc:
                content = f"<read error: {exc}>"
            files.append({"name": label, "path": str(path), "content": content})

        payload = result.to_payload()
        payload["ok"] = True
        payload["files"] = files
        return payload

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
            builder_kwargs=_parse_builder_kwargs(payload),
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

    def create_writeback_draft(self, payload: dict[str, object]) -> dict[str, object]:
        slug = _safe_slug(str(payload.get("project_slug") or "").strip())
        paths = _project_paths(self.vault_root, slug)
        if not paths["project_yaml"].exists():
            raise FileNotFoundError(f"project not found: {slug}")

        source_stage = str(payload.get("source_stage") or "").strip().lower()
        if source_stage not in _KNOWLEDGE_WRITEBACK_STAGES:
            raise ValueError(
                f"source_stage must be one of {sorted(_KNOWLEDGE_WRITEBACK_STAGES)}"
            )
        card_type = str(payload.get("card_type") or "").strip().lower()
        if card_type not in _KNOWLEDGE_WRITEBACK_CARD_TYPES:
            raise ValueError(
                f"card_type must be one of {sorted(_KNOWLEDGE_WRITEBACK_CARD_TYPES)}"
            )
        title = str(payload.get("title") or "").strip()
        if not title:
            raise ValueError("title is required")
        body = str(payload.get("body") or "").strip()
        if not body:
            raise ValueError("body is required")

        source_artifacts = _coerce_source_artifacts(payload.get("source_artifacts"))
        target_path_hint = str(payload.get("target_path_hint") or "").strip()
        if target_path_hint:
            target_rel = _normalize_knowledge_target_hint(
                vault_root=self.vault_root,
                project_slug=slug,
                target_path_hint=target_path_hint,
            )
        else:
            target_rel = _default_knowledge_target_path(
                project_slug=slug,
                card_type=card_type,
                title=title,
            )

        drafts_dir = paths["drafts_dir"]
        drafts_dir.mkdir(parents=True, exist_ok=True)
        stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
        safe_title = _safe_file_stem(title, fallback=card_type)
        draft_path = (
            drafts_dir
            / f"{stamp}__{card_type}__{safe_title}__writeback_draft.md"
        )
        frontmatter: dict[str, object] = {
            "type": "knowledge_writeback_draft",
            "project": slug,
            "source_stage": source_stage,
            "card_type": card_type,
            "title": title,
            "target_path": target_rel,
            "source_artifacts": source_artifacts,
            "review_status": "pending",
            "reviewed_by": "",
            "reviewed_at": "",
            "writeback_allowed": True,
            "vault_export_mode": "versioned",
        }
        draft_body = _render_knowledge_writeback_draft_body(
            title=title,
            source_stage=source_stage,
            card_type=card_type,
            target_path=target_rel,
            body=body,
            source_artifacts=source_artifacts,
        )
        draft_path.write_text(
            _compose_markdown_with_frontmatter(frontmatter, draft_body),
            encoding="utf-8",
        )
        preview = _read_text_preview(
            draft_path,
            limit_bytes=_PROJECT_DOC_PREVIEW_BYTES,
        )
        summary = _draft_summary(draft_path)
        return {
            "ok": True,
            "draft": summary,
            "draft_name": draft_path.name,
            "draft_path": str(draft_path),
            "target_path": target_rel,
            "status": "pending",
            "preview": preview,
        }

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
            "target_path",
            "title",
        }
        for key, value in payload.items():
            if key not in allowed:
                continue
            if key == "target_path":
                frontmatter[key] = _normalize_knowledge_target_hint(
                    vault_root=self.vault_root,
                    project_slug=slug,
                    target_path_hint=str(value),
                )
            elif key == "reviewed_at" and str(value).strip().lower() == "now":
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
        frontmatter, body = _load_markdown_with_frontmatter(draft_path)
        if str(frontmatter.get("type") or "").strip() == "knowledge_writeback_draft":
            return _apply_knowledge_writeback_draft(
                vault_root=self.vault_root,
                project_slug=slug,
                draft_path=draft_path,
                frontmatter=frontmatter,
                body=body,
                mode=mode,
            )
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
        spec = load_single_factor_case_spec(spec_path)
        evaluation_profile = str(
            payload.get("evaluation_profile") or project.alpha_lab_defaults.evaluation_profile
        )
        _preflight_strict_split_for_spec(
            spec,
            object_name="alpha-lab",
            source="single_factor_submit_preflight",
        )
        task = _RunTask(
            run_id=uuid.uuid4().hex,
            project_slug=slug,
            case_name=case_name,
            round_id=_optional_text(payload.get("round_id")),
            spec_path=str(spec_path),
            evaluation_profile=evaluation_profile,
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
        # 1. Delete the recorded run output directory.
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
        """Load previously saved custom factors from disk and register them.

        On-disk layout: ``custom_factors/{research,promoted}/<name>/factor.json``.
        The sibling ``research_log.md`` (if present) is the iteration log and is
        not loaded here.
        """
        load_persisted_custom_factors(self.workspace_root, ignore_errors=True)

    def list_custom_factors(self) -> dict[str, object]:
        """List all registered factor methods (built-in + custom)."""
        all_methods = factor_registry.supported_methods()
        items: list[dict[str, object]] = []
        for method in all_methods:
            is_custom = method not in BUILTIN_FACTOR_NAMES
            meta: dict[str, object] = {
                "name": method,
                "is_custom": is_custom,
                "role": "custom_factor" if is_custom else "base_method",
                "baseline_role": "base_method_only" if not is_custom else "candidate_or_custom",
            }
            if is_custom:
                meta_path = self._custom_factor_meta_path(method)
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
            "baseline_factor_suite": baseline_factor_suite_payload(include_non_default=True),
            "total": len(items),
            "custom_count": sum(1 for i in items if i.get("is_custom")),
            "baseline_count": len(baseline_factor_suite_payload(include_non_default=True)),
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
        fn = compile_custom_factor(name, code)

        # Register in the global factor_registry
        factor_registry.register(name, fn)

        # Persist to disk
        meta_path = self._custom_factor_write_path(name)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "name": name,
            "description": description,
            "code": code,
            "created_at": _utc_now_iso(),
        }
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        return {"name": name, "registered": True, "persisted": str(meta_path)}

    def delete_custom_factor(self, name: str) -> dict[str, object]:
        """Unregister a custom factor and remove its persisted file.

        Removes ``factor.json`` and the enclosing ``<name>/`` directory if it is
        empty. A sibling ``research_log.md`` (or any other artifact) keeps the
        directory around — iteration history outlives a single registration.
        """
        name = name.strip().lower()
        if name in BUILTIN_FACTOR_NAMES:
            raise ValueError(f"cannot delete built-in factor: {name}")
        if name not in factor_registry:
            raise FileNotFoundError(f"factor not found: {name}")

        factor_registry._builders.pop(name, None)

        meta_path = self._custom_factor_meta_path(name)
        if meta_path.exists():
            meta_path.unlink()
            parent = meta_path.parent
            try:
                parent.rmdir()
            except OSError:
                pass

        return {"name": name, "deleted": True}

    def get_custom_factor_code(self, name: str) -> dict[str, object]:
        """Return the source code of a persisted custom factor."""
        meta_path = self._custom_factor_meta_path(name)
        if not meta_path.exists():
            raise FileNotFoundError(f"custom factor not found: {name}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return {
            "name": meta["name"],
            "code": meta.get("code", ""),
            "description": meta.get("description", ""),
        }

    def _iter_custom_factor_meta_paths(self) -> list[Path]:
        return iter_custom_factor_meta_paths(self.workspace_root)

    def _custom_factor_meta_path(self, name: str) -> Path:
        return custom_factor_meta_path(self.workspace_root, name)

    def _custom_factor_write_path(self, name: str) -> Path:
        return custom_factor_write_path(self.workspace_root, name)


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
        if path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return
        if path in {"/dev/alpha-lab/overview-fixture", "/dev/alpha-lab1/overview-fixture"}:
            self._send_html(_index_html(reload_template=True))
            return
        if path == "/model-lab":
            self._send_html(_model_lab_html())
            return
        if path == "/dev/model-lab/overview-fixture":
            self._send_html(_model_lab_html(reload_template=True))
            return
        if path == "/dev/model-lab/artifact-fixture":
            self._send_html(_model_lab_html(reload_template=True))
            return
        if path == "/dev/model-lab/diagnostics-fixture":
            self._send_html(_model_lab_html(reload_template=True))
            return

        if path == "/api/dev/model-lab/overview-fixtures":
            self._send_json({"ok": True, "fixtures": _list_model_lab_overview_fixtures()})
            return
        if path == "/api/dev/model-lab/artifact-fixtures":
            self._send_json({"ok": True, "fixtures": _list_model_lab_overview_fixtures()})
            return
        if path in {
            "/api/dev/alpha-lab/overview-fixtures",
            "/api/dev/alpha-lab1/overview-fixtures",
        }:
            self._send_json({"ok": True, "fixtures": _list_alpha_lab_overview_fixtures()})
            return

        # Dashboard
        if path == "/api/dashboard":
            self._send_json(self.svc.dashboard())
            return
        if path == "/api/settings/llm":
            self._send_json(self.svc.llm_settings_status())
            return
        if path == "/api/settings/mechanism-index":
            self._send_json(self.svc.mechanism_index_status())
            return
        if path == "/api/model-lab/specs":
            self._send_json({"specs": self.svc.list_model_lab_specs()})
            return
        if path == "/api/model-lab/candidates":
            self._send_json({"candidates": self.svc.list_model_lab_candidates()})
            return
        if path == "/api/model-lab/sources":
            self._send_json({"sources": self.svc.list_model_lab_sources()})
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
        if (
            len(parts) == 5
            and parts[0] == "api"
            and parts[1] == "dev"
            and parts[2] == "model-lab"
            and parts[3] == "overview-fixtures"
        ):
            try:
                self._send_json(_load_model_lab_overview_fixture(parts[4]))
            except Exception as exc:
                self._send_error_payload(exc)
            return
        if (
            len(parts) == 5
            and parts[0] == "api"
            and parts[1] == "dev"
            and parts[2] in {"alpha-lab", "alpha-lab1"}
            and parts[3] == "overview-fixtures"
        ):
            try:
                self._send_json(_load_alpha_lab_overview_fixture(parts[4]))
            except Exception as exc:
                self._send_error_payload(exc)
            return
        if (
            len(parts) == 7
            and parts[0] == "api"
            and parts[1] == "dev"
            and parts[2] == "model-lab"
            and parts[3] == "artifact-fixtures"
            and parts[5] == "artifact"
        ):
            try:
                content_type, content = _load_model_lab_artifact_fixture(parts[4], parts[6])
                self._send_text(content, content_type=content_type)
            except Exception as exc:
                self._send_error_payload(exc)
            return
        if len(parts) >= 3 and parts[0] == "api" and parts[1] == "model-lab":
            try:
                if len(parts) == 4 and parts[2] == "specs":
                    self._send_json(self.svc.read_model_lab_spec(parts[3]))
                    return
                if len(parts) == 4 and parts[2] == "candidates":
                    self._send_json(self.svc.read_model_lab_candidate(parts[3]))
                    return
                if len(parts) == 4 and parts[2] == "sources":
                    self._send_json(self.svc.read_model_lab_source(parts[3]))
                    return
                if len(parts) == 3 and parts[2] == "runs":
                    compact_query = parse_qs(parsed.query).get("compact") or [""]
                    compact_raw = str(compact_query[0]).strip().lower()
                    compact = compact_raw in {"1", "true", "yes", "y"}
                    query = parse_qs(parsed.query)
                    self._send_json(
                        {
                            "runs": self.svc.list_model_lab_runs(
                                compact=compact,
                                status_filter=str(query.get("status", [""])[0]),
                                case_filter=str(query.get("case", [""])[0]),
                                note_filter=str(query.get("note", [""])[0]),
                            )
                        }
                    )
                    return
                if len(parts) == 4 and parts[2] == "runs":
                    self._send_json(self.svc.get_model_lab_run(parts[3]).to_payload())
                    return
                if len(parts) == 5 and parts[2] == "runs" and parts[4] == "overview":
                    self._handle_get_model_lab_run_overview(run_id=parts[3])
                    return
                if len(parts) == 6 and parts[2] == "runs" and parts[4] == "artifact":
                    artifact_query = parse_qs(parsed.query or "")
                    download = str(artifact_query.get("download", ["0"])[0]).strip().lower() in {
                        "1",
                        "true",
                        "yes",
                    }
                    self._handle_get_model_lab_run_artifact(
                        run_id=parts[3],
                        artifact_key=parts[5],
                        download=download,
                    )
                    return
            except Exception as exc:
                self._send_error_payload(exc)
                return
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
            if parsed.path == "/api/vault/idea-distribute":
                idea = str(payload.get("idea") or "").strip()
                lab = str(payload.get("lab") or "single_factor").strip()
                engines = _coerce_engines_payload(payload.get("engines"))
                top_k = _as_int(payload.get("top_k"), default=8)
                self._send_json(
                    self.svc.create_idea_distribute(
                        idea=idea,
                        lab=lab,
                        engines=engines,
                        top_k=top_k,
                    )
                )
                return
            if parsed.path == "/api/model-lab/idea-distribute":
                idea = str(payload.get("idea") or "").strip()
                engines = _coerce_engines_payload(payload.get("engines"))
                top_k = _as_int(payload.get("top_k"), default=8)
                self._send_json(
                    self.svc.create_idea_distribute(
                        idea=idea,
                        lab="model_factor",
                        engines=engines,
                        top_k=top_k,
                    )
                )
                return
            if parsed.path == "/api/vault/writeback-drafts":
                self._send_json(
                    self.svc.create_writeback_draft(payload),
                    status=HTTPStatus.CREATED,
                )
                return
            if parsed.path == "/api/vault/preflight":
                self._send_json(self.svc.run_preflight_check(payload))
                return
            if parsed.path == "/api/settings/llm":
                self._send_json(self.svc.save_llm_settings(payload))
                return
            if parsed.path == "/api/custom-factors":
                self._send_json(self.svc.register_custom_factor(payload), status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/projects":
                created = self.svc.create_project(payload)
                self._send_json(created, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/model-lab/candidates":
                self._send_json(
                    self.svc.save_model_lab_candidate(payload),
                    status=HTTPStatus.CREATED,
                )
                return
            if (
                len(parts) == 5
                and parts[0] == "api"
                and parts[1] == "model-lab"
                and parts[2] == "candidates"
                and parts[4] == "validate"
            ):
                self._send_json(self.svc.validate_model_lab_candidate(parts[3], payload))
                return
            if (
                len(parts) == 5
                and parts[0] == "api"
                and parts[1] == "model-lab"
                and parts[2] == "candidates"
                and parts[4] == "materialize-spec"
            ):
                self._send_json(
                    self.svc.materialize_model_lab_candidate_spec(parts[3], payload)
                )
                return
            if (
                len(parts) == 5
                and parts[0] == "api"
                and parts[1] == "model-lab"
                and parts[2] == "candidates"
                and parts[4] == "run"
            ):
                self._send_json(self.svc.run_model_lab_candidate(parts[3], payload))
                return
            if parsed.path == "/api/model-lab/runs":
                self._send_json(self.svc.submit_model_lab_run(payload))
                return
            if (
                len(parts) == 5
                and parts[0] == "api"
                and parts[1] == "model-lab"
                and parts[2] == "runs"
                and parts[4] == "export-card"
            ):
                self._send_json(
                    self.svc.export_model_lab_run_experiment_card(
                        run_id=parts[3],
                        mode=str(payload.get("mode") or "versioned").strip() or "versioned",
                    )
                )
                return
            if parsed.path == "/api/model-lab/runs/compare":
                self._send_json(self.svc.compare_model_lab_runs(payload))
                return
            if parsed.path == "/api/model-lab/specs/diff":
                self._send_json(self.svc.diff_model_lab_specs(payload))
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

    def do_PUT(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        parts = _path_parts(parsed.path)
        payload = self._read_json_body_or_empty()
        try:
            if (
                len(parts) == 4
                and parts[0] == "api"
                and parts[1] == "model-lab"
                and parts[2] == "specs"
            ):
                self._send_json(self.svc.update_model_lab_spec(parts[3], payload))
                return
            if (
                len(parts) == 5
                and parts[0] == "api"
                and parts[1] == "model-lab"
                and parts[2] == "specs"
                and parts[4] == "duplicate"
            ):
                self._send_json(self.svc.duplicate_model_lab_spec(parts[3], payload))
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
            if (
                len(parts) == 4
                and parts[0] == "api"
                and parts[1] == "model-lab"
                and parts[2] == "specs"
            ):
                self._send_json(self.svc.delete_model_lab_spec(parts[3]))
                return
            if (
                len(parts) == 4
                and parts[0] == "api"
                and parts[1] == "model-lab"
                and parts[2] == "runs"
            ):
                self._send_json(self.svc.delete_model_lab_run(parts[3]))
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
        artifact_path = _resolve_run_artifact_for_endpoint(run, artifact_key)
        if artifact_path is None:
            registered = bool(run.artifact_paths.get(artifact_key))
            error_text = (
                f"artifact file not found for key: {artifact_key}"
                if registered
                else f"artifact key not found: {artifact_key}"
            )
            self._send_json(
                {"ok": False, "error": error_text},
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

    def _handle_get_model_lab_run_artifact(
        self,
        *,
        run_id: str,
        artifact_key: str,
        download: bool = False,
    ) -> None:
        run = self.svc.get_model_lab_run(run_id)
        self._send_run_artifact(run=run, artifact_key=artifact_key, download=download)

    def _handle_get_model_lab_run_overview(self, *, run_id: str) -> None:
        run = self.svc.get_model_lab_run(run_id)
        snapshot = _build_run_overview_snapshot(run)
        self._send_json(
            {"ok": True, "run_id": run_id, "summary": dict(run.summary), "snapshot": snapshot}
        )

    def _send_run_artifact(
        self,
        *,
        run: _RunRecord,
        artifact_key: str,
        download: bool = False,
    ) -> None:
        artifact_path = _resolve_run_artifact_for_endpoint(run, artifact_key)
        if artifact_path is None:
            registered = bool(run.artifact_paths.get(artifact_key))
            error_text = (
                f"artifact file not found for key: {artifact_key}"
                if registered
                else f"artifact key not found: {artifact_key}"
            )
            self._send_json(
                {"ok": False, "error": error_text},
                status=HTTPStatus.NOT_FOUND,
            )
            return
        file_size = artifact_path.stat().st_size
        ctype = _guess_content_type(artifact_path)
        if "text" in ctype or "json" in ctype:
            raw = artifact_path.read_bytes()
            if len(raw) > _MAX_TEXT_BYTES:
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

    def _send_text(self, body: str, *, content_type: str) -> None:
        encoded = body.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
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


def _coerce_source_artifacts(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [line.strip() for line in value.splitlines() if line.strip()]
    return []


def _safe_file_stem(value: str, *, fallback: str) -> str:
    raw = str(value).strip() or fallback
    normalized = "".join(ch if ch.isalnum() or ch in {"-", "_", ".", " "} else "-" for ch in raw)
    normalized = re.sub(r"\s+", "-", normalized).strip("._-")
    if not normalized:
        normalized = fallback
    return normalized[:96].strip("._-") or fallback


def _default_knowledge_target_path(
    *,
    project_slug: str,
    card_type: str,
    title: str,
) -> str:
    target_dir = _KNOWLEDGE_WRITEBACK_TARGET_DIRS[card_type]
    prefix = _KNOWLEDGE_WRITEBACK_FILENAME_PREFIXES[card_type]
    safe_title = _safe_file_stem(title, fallback=card_type)
    return (
        f"{PROJECTS_DIRNAME}/{_safe_slug(project_slug)}/"
        f"{target_dir}/{prefix} - {safe_title}.md"
    )


def _normalize_knowledge_target_hint(
    *,
    vault_root: Path,
    project_slug: str,
    target_path_hint: str,
) -> str:
    raw = str(target_path_hint or "").strip().replace("\\", "/")
    if not raw:
        raise ValueError("target_path_hint is required")
    if re.match(r"^[A-Za-z]:", raw):
        raise PermissionError("target_path_hint must be vault-relative")
    root = vault_root.resolve()
    rel = raw.lstrip("/")
    candidate = (root / rel).resolve()
    project_dir = (root / PROJECTS_DIRNAME / _safe_slug(project_slug)).resolve()
    try:
        candidate.relative_to(project_dir)
    except ValueError as exc:
        raise PermissionError("target path must stay inside the selected project") from exc
    if candidate.suffix.lower() != ".md":
        raise ValueError("target path must be a markdown file")
    return candidate.relative_to(root).as_posix()


def _render_knowledge_writeback_draft_body(
    *,
    title: str,
    source_stage: str,
    card_type: str,
    target_path: str,
    body: str,
    source_artifacts: list[str],
) -> str:
    lines = [
        f"# {title}",
        "",
        "## 写回元信息",
        f"- 来源阶段: {source_stage}",
        f"- 推荐卡片类型: `{card_type}`",
        f"- 目标 vault 路径: `{target_path}`",
        "- 写回原则: 只收用户审阅后的判断或本地验证后的事实，不收 agent 原始过程稿。",
        "",
        "## 正文",
        body.strip(),
    ]
    if source_artifacts:
        lines.extend(["", "## Source Artifacts"])
        lines.extend(f"- `{item}`" for item in source_artifacts)
    return "\n".join(lines)


def _apply_knowledge_writeback_draft(
    *,
    vault_root: Path,
    project_slug: str,
    draft_path: Path,
    frontmatter: dict[str, object],
    body: str,
    mode: str | None,
) -> dict[str, object]:
    review_status = str(frontmatter.get("review_status") or "").strip().lower()
    if review_status != "approved":
        raise ValueError(f"draft {draft_path} has not been approved")
    if not bool(frontmatter.get("writeback_allowed", True)):
        raise ValueError(f"draft {draft_path} is not allowed for writeback")

    mode_l = str(mode or frontmatter.get("vault_export_mode") or "versioned").strip().lower()
    if mode_l not in {"skip", "overwrite", "versioned"}:
        raise ValueError("mode must be one of skip, overwrite, versioned")
    if mode_l == "skip":
        return {
            "project": project_slug,
            "draft_path": str(draft_path),
            "status": "skipped",
            "success": True,
            "target_paths": [],
            "mode_used": "skip",
            "error": None,
        }

    target_rel = _normalize_knowledge_target_hint(
        vault_root=vault_root,
        project_slug=project_slug,
        target_path_hint=str(frontmatter.get("target_path") or ""),
    )
    target_path = (vault_root.resolve() / target_rel).resolve()
    if target_path.exists() and mode_l == "versioned":
        stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
        target_path = target_path.with_name(
            f"{target_path.stem}__{stamp}{target_path.suffix}"
        )

    target_frontmatter: dict[str, object] = {
        "type": str(frontmatter.get("card_type") or "experiment_note"),
        "project": project_slug,
        "source_stage": str(frontmatter.get("source_stage") or ""),
        "source_artifacts": _coerce_source_artifacts(frontmatter.get("source_artifacts")),
        "reviewed_by": str(frontmatter.get("reviewed_by") or ""),
        "reviewed_at": str(frontmatter.get("reviewed_at") or ""),
        "created_from_draft": draft_path.name,
    }
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(
        _compose_markdown_with_frontmatter(target_frontmatter, body),
        encoding="utf-8",
    )
    return {
        "project": project_slug,
        "draft_path": str(draft_path),
        "status": "success",
        "success": True,
        "target_paths": [str(target_path)],
        "mode_used": mode_l,
        "error": None,
    }


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


def _read_text_with_limit(path: Path, *, limit_bytes: int) -> str:
    file_size = path.stat().st_size
    with path.open("rb") as fh:
        raw = fh.read(limit_bytes + 1)
    if len(raw) <= limit_bytes and file_size <= limit_bytes:
        return raw.decode("utf-8", errors="replace")
    raise AlphaLabDataError(
        f"file too large to edit inline: {path} ({file_size} bytes, limit {limit_bytes})"
    )


def _iso_from_timestamp(timestamp: float) -> str:
    return dt.datetime.fromtimestamp(timestamp, tz=dt.UTC).isoformat().replace("+00:00", "Z")


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
        "source_stage": str(frontmatter.get("source_stage") or ""),
        "card_type": str(frontmatter.get("card_type") or ""),
        "target_path": str(frontmatter.get("target_path") or ""),
        "draft_type": str(frontmatter.get("type") or ""),
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


def _extract_metrics_summary(
    metrics_path: Path | None,
    *,
    run_status: str | None = None,
) -> dict[str, object]:
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
    return _enrich_evaluation_summary(
        {str(key): value for key, value in metrics.items()},
        run_status=run_status,
    )


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


def _load_run_draft_model_source(run: _RunRecord) -> dict[str, object] | None:
    for artifact_key, fallback_name in (
        ("run_manifest", "run_manifest.json"),
        ("model_definition_json", "model_definition.json"),
        ("feature_manifest_json", "feature_manifest.json"),
    ):
        payload = _read_json_artifact(
            _resolve_run_artifact_path(
                run,
                artifact_key=artifact_key,
                fallback_name=fallback_name,
            )
        )
        if not isinstance(payload, dict):
            continue
        source = payload.get("draft_model_source")
        if isinstance(source, dict):
            return {str(key): value for key, value in source.items()}
        inputs = payload.get("inputs")
        if isinstance(inputs, dict) and isinstance(inputs.get("draft_model_source"), dict):
            nested = cast(dict[object, object], inputs["draft_model_source"])
            return {str(key): value for key, value in nested.items()}
    return None


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


def _read_partition_rows(
    run: _RunRecord,
    alias_paths: tuple[tuple[str, str], ...],
) -> list[dict[str, str]]:
    for artifact_key, fallback_name in alias_paths:
        rows = _read_csv_artifact_rows(
            _resolve_run_artifact_path(
                run,
                artifact_key=artifact_key,
                fallback_name=fallback_name,
            )
        )
        if rows:
            return rows
    return []


def _load_model_factor_portfolio_validation_snapshot(
    run: _RunRecord,
) -> dict[str, object]:
    summary_payload = _read_json_artifact(
        _resolve_run_artifact_path(
            run,
            artifact_key="portfolio_validation_summary",
            fallback_name="level2_portfolio_validation/portfolio_validation_summary.json",
        )
    )
    metrics_payload = _read_json_artifact(
        _resolve_run_artifact_path(
            run,
            artifact_key="portfolio_validation_metrics",
            fallback_name="level2_portfolio_validation/portfolio_validation_metrics.json",
        )
    )
    package_payload = _read_json_artifact(
        _resolve_run_artifact_path(
            run,
            artifact_key="portfolio_validation_package",
            fallback_name="level2_portfolio_validation/portfolio_validation_package.json",
        )
    )
    summary = summary_payload if isinstance(summary_payload, dict) else {}
    metrics = metrics_payload if isinstance(metrics_payload, dict) else {}
    package = package_payload if isinstance(package_payload, dict) else {}
    return {
        "summary": summary,
        "metrics": metrics,
        "package": package,
        "status": _coerce_finite_or_text(summary.get("validation_status"))
        or _coerce_finite_or_text(summary.get("recommendation"))
        or "not_available",
        "portfolio_validation_recommendation": _coerce_finite_or_text(
            summary.get("recommendation")
        ),
        "has_data": bool(summary or metrics or package),
    }


def _build_run_failure_snapshot(run: _RunRecord) -> dict[str, object]:
    message = run.error_message or ""
    raw = run.error or ""
    hint = run.error_hint or ""
    has_error = (
        run.status == "failed"
        and bool((message or "").strip() or (raw or "").strip() or (hint or "").strip())
    )
    return {
        "run_id": run.run_id,
        "case_name": run.case_name,
        "status": run.status,
        "error_type": run.error_type,
        "error_message": message,
        "error_hint": hint,
        "error": raw,
        "has_error": has_error,
    }


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
        "failure": _build_run_failure_snapshot(run),
        "draftModelSource": _load_run_draft_model_source(run),
        "portfolioValidation": _load_model_factor_portfolio_validation_snapshot(run),
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
        "randomBaselineRows": _read_csv_artifact_rows(
            _resolve_run_artifact_path(
                run,
                artifact_key="random_baseline_null",
                fallback_name="random_baseline_null.csv",
            )
        ),
        "conditionalMagnitudeRows": _read_csv_artifact_rows(
            _resolve_run_artifact_path(
                run,
                artifact_key="conditional_ic_by_magnitude",
                fallback_name="conditional_ic_by_magnitude.csv",
            )
        ),
        "conditionalCrossSectionRows": _read_csv_artifact_rows(
            _resolve_run_artifact_path(
                run,
                artifact_key="conditional_ic_by_cross_section_size",
                fallback_name="conditional_ic_by_cross_section_size.csv",
            )
        ),
        "industryRows": _read_partition_rows(
            run,
            (
                ("industry_returns", "industry_returns.csv"),
                ("industry_group_returns", "industry_group_returns.csv"),
                ("group_returns_by_industry", "group_returns_by_industry.csv"),
                ("returns_by_industry", "returns_by_industry.csv"),
                ("conditional_ic_by_industry", "conditional_ic_by_industry.csv"),
            ),
        ),
        "sizeRows": _read_partition_rows(
            run,
            (
                ("size_returns", "size_returns.csv"),
                ("size_group_returns", "size_group_returns.csv"),
                ("group_returns_by_size", "group_returns_by_size.csv"),
                ("returns_by_size", "returns_by_size.csv"),
                ("cross_section_size", "conditional_ic_by_cross_section_size.csv"),
                (
                    "conditional_ic_by_cross_section_size",
                    "conditional_ic_by_cross_section_size.csv",
                ),
                ("conditional_ic_by_size", "conditional_ic_by_size.csv"),
            ),
        ),
        "regimeRows": _read_partition_rows(
            run,
            (
                ("regime_returns", "regime_returns.csv"),
                ("group_returns_by_regime", "group_returns_by_regime.csv"),
                ("returns_by_regime", "returns_by_regime.csv"),
                ("volatility_regime_returns", "volatility_regime_returns.csv"),
                ("regime_group_returns", "regime_group_returns.csv"),
            ),
        ),
        "integrity": _load_model_factor_run_leakage_summary(run),
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


def _load_run_turnover_timeseries(run: _RunRecord) -> dict[str, float]:
    path = _resolve_run_artifact_path(
        run, artifact_key="turnover", fallback_name="turnover.csv"
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
                value = _coerce_finite_float((row or {}).get("turnover"))
                if value is None:
                    value = _coerce_finite_float((row or {}).get("mean_turnover"))
                if value is None:
                    value = _coerce_finite_float((row or {}).get("value"))
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


def _resolve_run_artifact_for_endpoint(
    run: _RunRecord, artifact_key: str
) -> Path | None:
    """Locate an artifact file for the per-key download endpoint.

    Falls back, in order, to: the registered path for `artifact_key`, the
    canonical filename under `output_dir`, and any registered synonym keys
    (e.g., `group_returns` ↔ `quantile_returns`). Returns None when nothing
    on disk matches.
    """

    def _lookup(key: str) -> Path | None:
        path_text = run.artifact_paths.get(key)
        if path_text:
            path = Path(path_text).expanduser().resolve()
            if path.exists() and path.is_file():
                return path
        fallback_name = _ARTIFACT_DISK_FILENAMES.get(key)
        if fallback_name and run.output_dir:
            fallback = Path(run.output_dir).expanduser().resolve() / fallback_name
            if fallback.exists() and fallback.is_file():
                return fallback
        return None

    direct = _lookup(artifact_key)
    if direct is not None:
        return direct
    for synonym in _ARTIFACT_KEY_SYNONYMS.get(artifact_key, ()):
        recovered = _lookup(synonym)
        if recovered is not None:
            return recovered
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


def _resolve_run_evaluation_title(run: _RunRecord) -> str:
    summary = _ensure_run_summary(run)
    title = _coerce_finite_or_text(summary.get("evaluation_title")) or _pick_evaluation_title(
        summary
    )
    if title:
        return title
    status_label = {
        "queued": "queued",
        "running": "running",
        "succeeded": "completed",
        "failed": "failed",
        "cancelled": "cancelled",
    }.get(str(run.status or "").strip().lower())
    if status_label:
        return status_label
    fallback = _coerce_finite_or_text(run.status)
    return fallback or "-"


def _read_run_payload(path: Path | None) -> dict[str, object] | None:
    if path is None or not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _coerce_finite_or_text(value: object) -> str | None:
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        text = str(value).strip()
        return text or None
    return None


def _pick_evaluation_title(summary: Mapping[str, object]) -> str | None:
    transition = _coerce_finite_or_text(summary.get("level12_transition_label"))
    if transition and not _is_inconclusive_transition(transition):
        return transition
    for key in (
        "promotion_decision",
        "campaign_triage",
        "factor_verdict",
        "portfolio_validation_recommendation",
    ):
        value = _coerce_finite_or_text(summary.get(key))
        if value:
            return value
    if transition:
        return transition
    return None


def _is_inconclusive_transition(value: str) -> bool:
    return value.strip().lower() == "inconclusive transition"


def _derive_evaluation_action_and_next_step(
    summary: Mapping[str, object],
    *,
    run_status: str | None = None,
) -> tuple[str, str]:
    status = str(run_status or "").strip().lower()
    if status in {"failed", "cancelled"}:
        return "STOP", "先查看报错并修复数据或配置后再重跑。"
    if status in {"queued", "running"}:
        return "HOLD", "运行中，完成后再依据结论决定是否推进。"

    promotion = str(_coerce_finite_or_text(summary.get("promotion_decision")) or "").strip().lower()
    recommendation = str(
        _coerce_finite_or_text(summary.get("portfolio_validation_recommendation")) or ""
    ).strip().lower()
    triage = str(_coerce_finite_or_text(summary.get("campaign_triage")) or "").strip().lower()
    verdict = str(_coerce_finite_or_text(summary.get("factor_verdict")) or "").strip().lower()
    transition = str(
        _coerce_finite_or_text(summary.get("level12_transition_label")) or ""
    ).strip().lower()

    if (
        promotion == "blocked from level 2"
        or triage == "drop for now"
        or verdict in {"weak / noisy", "fails basic robustness"}
    ):
        if promotion == "blocked from level 2":
            return "STOP", "先处理 promotion blockers，再用 default_research 复跑验证。"
        if triage == "drop for now":
            return "STOP", "建议先重做因子或特征方案，再决定是否继续该方向。"
        return "STOP", "先修复核心稳健性问题（IC、覆盖率、子区间）后再重跑。"

    if recommendation == "credible at portfolio level" or transition in {
        "confirmed at portfolio level",
        "improved at portfolio level",
    }:
        return "GO", "进入候选池并开展组合约束与交易成本复核。"
    if promotion == "promote to level 2":
        return "GO", "已满足晋级门槛，下一步执行 Level 2 组合验证。"
    if (
        triage in {"advance to level 2", "strong level 1 candidate"}
        and verdict == "strong candidate"
    ):
        return "GO", "优先进入下一轮验证，并补齐组合层证据。"

    if recommendation == "needs portfolio refinement":
        return "HOLD", "优先优化换手、成本与集中度，再复跑组合验证。"
    if recommendation == "not evaluated (not promoted)" or transition == "inconclusive transition":
        return "HOLD", "当前未形成明确转化结论，先提升到 Promote to Level 2。"
    if triage in {"fragile / monitor", "needs refinement"} or verdict in {
        "promising but fragile",
        "mixed evidence",
    }:
        return "HOLD", "先补充滚动稳定性与不确定性证据，再判断推进。"

    return "HOLD", "维持观察，补充关键诊断后再决策。"


def _enrich_evaluation_summary(
    summary: Mapping[str, object],
    *,
    run_status: str | None = None,
) -> dict[str, object]:
    enriched = {str(key): value for key, value in summary.items()}
    _enrich_split_summary_fields(enriched)
    title = _pick_evaluation_title(enriched)
    action, next_step = _derive_evaluation_action_and_next_step(
        enriched,
        run_status=run_status,
    )
    if title:
        enriched["evaluation_title"] = title
    enriched["evaluation_action"] = action
    enriched["evaluation_next_step"] = next_step
    return enriched


def _enrich_split_summary_fields(summary: dict[str, object]) -> None:
    raw_contract = summary.get("split_contract")
    if isinstance(raw_contract, Mapping):
        contract = {str(key): value for key, value in raw_contract.items()}
        summary["split_contract"] = contract
        summary["split_status"] = "strict"
        for key in ("is_start", "is_end", "oos_start", "oos_end"):
            if key in contract:
                summary.setdefault(key, contract[key])
        if "embargo_days" in contract:
            summary["embargo_days"] = contract["embargo_days"]
            summary.setdefault("split_embargo_days", contract["embargo_days"])
        return

    split_description = str(summary.get("split_description") or "").strip().lower()
    if split_description in {"full_sample", "full-sample", "full sample"}:
        summary["split_status"] = "full_sample"
    else:
        summary.setdefault("split_status", "missing")


def _read_yaml_document_safe(path_text: str) -> dict[str, object] | None:
    try:
        payload = load_yaml_document(Path(path_text))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _coerce_spec_version(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 1 else None
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        version = int(value)
        return version if version >= 1 and float(version) == value else None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            version = int(text)
        except ValueError:
            return None
        return version if version >= 1 else None
    return None


def _next_spec_version(value: int | None) -> int:
    if value is None or value < 1:
        return 1
    return value + 1


def _coerce_lineage_history(value: object) -> list[dict[str, object]]:
    if value is None:
        return []
    if isinstance(value, dict):
        lineage_entries: list[dict[str, object]] = []
        for key, item_value in value.items():
            candidate = _coerce_lineage_item(item_value)
            if candidate:
                candidate["event_key"] = str(key).strip()
                lineage_entries.append(candidate)
        return lineage_entries
    if not isinstance(value, (list, tuple)):
        return []
    lineage_entries_from_iterable: list[dict[str, object]] = []
    for item in value:
        candidate = _coerce_lineage_item(item)
        if candidate:
            lineage_entries_from_iterable.append(candidate)
    return lineage_entries_from_iterable


def _coerce_lineage_item(raw_item: object) -> dict[str, object] | None:
    if isinstance(raw_item, dict):
        payload: dict[str, object] = {}
        for key, value in raw_item.items():
            if not isinstance(key, str):
                continue
            name = key.strip()
            if not name:
                continue
            payload[name] = value
        return payload
    if isinstance(raw_item, str):
        text = raw_item.strip()
        if not text:
            return None
        return {"raw": text}
    return None


def _extract_spec_lineage(payload: dict[str, object]) -> dict[str, object]:
    if not isinstance(payload, dict):
        return {}
    raw_lineage = payload.get("lineage")
    if isinstance(raw_lineage, dict):
        lineage: dict[str, object] = {}
        for key, value in raw_lineage.items():
            if isinstance(key, str):
                name = key.strip()
                if name:
                    lineage[name] = value
        history = _coerce_lineage_history(raw_lineage.get("history"))
        if history:
            lineage["history"] = history
        return lineage
    if isinstance(raw_lineage, (list, tuple)):
        return {"history": _coerce_lineage_history(raw_lineage)}
    if isinstance(raw_lineage, str):
        text = raw_lineage.strip()
        if text:
            return {"history": [{"raw": text}]}
    return {}


def _build_spec_lineage_meta(
    spec_path: Path,
    raw_spec: dict[str, object] | None,
) -> dict[str, object]:
    payload = raw_spec or {}
    lineage = _extract_spec_lineage(payload)
    copied_from = _coerce_finite_or_text(payload.get("copied_from"))
    version = _coerce_spec_version(payload.get("version"))
    if version is None:
        version = 1
    return {
        "version": version,
        "lineage": lineage,
        "copied_from": copied_from or "",
        "file_signature": _file_signature(spec_path),
    }


def _dump_spec_payload(payload: object, suffix: str) -> str:
    normalized: dict[str, object] | list[object]
    if isinstance(payload, dict):
        normalized = payload
    elif isinstance(payload, list):
        normalized = cast(list[object], payload)
    elif isinstance(payload, (str, bytes)):
        return str(payload)
    else:
        return json.dumps(payload, ensure_ascii=False, indent=2)

    if suffix == ".json":
        return json.dumps(normalized, ensure_ascii=False, indent=2)

    yaml = _require_yaml()
    return str(yaml.safe_dump(normalized, sort_keys=False, allow_unicode=True))


def _derive_factor_name_from_spec_stem(stem: str) -> str:
    raw = str(stem or "").strip()
    normalized = re.sub(r"[^A-Za-z0-9_]+", "_", raw).strip("_").lower()
    return normalized or "model_factor_copy"


def _strip_spec_diff_metadata(payload: dict[str, object]) -> dict[str, object]:
    stripped = {
        key: value
        for key, value in payload.items()
        if key not in {"copied_from", "lineage", "version"}
    }
    return stripped


def _preflight_strict_split_for_spec(
    spec: object,
    *,
    object_name: str,
    source: str,
) -> dict[str, object]:
    prices_path = str(getattr(spec, "prices_path", "") or "").strip()
    if not prices_path:
        raise AlphaLabIOError(f"{object_name} 启动前检查失败：prices_path 不能为空")
    try:
        prices_resolved = resolve_tabular_frame_path(prices_path, object_name="prices")
        date_values = _read_tabular_date_values(prices_resolved, object_name="prices")
        target = getattr(spec, "target", None)
        target_horizon = int(getattr(target, "horizon", 1) or 1)
        rebalance_step = rebalance_frequency_to_step(
            getattr(spec, "rebalance_frequency", None)
        )
    except Exception as exc:
        raise AlphaLabIOError(
            f"{object_name} split 启动前检查失败，请先修复 prices 数据或配置：{exc}"
        ) from exc

    contract, remediations = preflight_split_contract(
        date_values,
        target_horizon=target_horizon,
        rebalance_step=rebalance_step,
        source=source,
    )
    if contract is None:
        bullet_lines = "\n".join(f"- {item}" for item in remediations)
        raise AlphaLabIOError(
            f"{object_name} strict split 启动前检查失败，任务未启动：\n{bullet_lines}"
        )
    return contract.to_metadata()


def _read_tabular_date_values(path: Path, *, object_name: str) -> Any:
    import pandas as pd

    suffix = path.suffix.lower()
    try:
        if suffix == ".csv":
            return pd.read_csv(path, usecols=["date"])["date"]
        if suffix in {".parquet", ".pq"}:
            return pd.read_parquet(path, columns=["date"])["date"]
    except Exception as exc:
        raise AlphaLabDataError(
            f"{object_name} 无法读取 date 列用于 split preflight: {path} ({exc})"
        ) from exc
    raise AlphaLabIOError(f"{object_name} 文件后缀不支持: {path}")


def _preflight_model_lab_spec_inputs(spec: object) -> None:
    prices_path = str(getattr(spec, "prices_path", "") or "").strip()
    features_path = str(getattr(spec, "features_path", "") or "").strip()
    feature_columns = list(getattr(spec, "feature_columns", ()) or ())

    if not prices_path or not features_path:
        raise ValueError("model-lab preflight requires non-empty prices_path/features_path")

    failures: list[str] = []
    prices_resolved: Path | None = None
    features_resolved: Path | None = None

    try:
        prices_resolved = resolve_tabular_frame_path(prices_path, object_name="prices")
    except Exception as exc:
        failures.append(str(exc))
    try:
        feature_storage = ensure_parquet_tabular_frame(
            features_path,
            object_name="features",
        )
        features_resolved = feature_storage.path
    except Exception as exc:
        failures.append(str(exc))

    universe_payload = getattr(spec, "universe", None)
    universe_path = str(getattr(universe_payload, "path", "") or "").strip()
    universe_col = str(
        getattr(universe_payload, "in_universe_column", "in_universe") or "in_universe"
    )
    if universe_path:
        try:
            _ = resolve_tabular_frame_path(universe_path, object_name="universe")
        except Exception as exc:
            failures.append(str(exc))

    if prices_resolved is not None:
        try:
            _preflight_tabular_columns(
                prices_resolved,
                required_columns=("date", "asset", "close"),
                object_name="prices",
            )
        except Exception as exc:
            failures.append(str(exc))

    if features_resolved is not None:
        required_feature_columns = ("date", "asset", *tuple(str(col) for col in feature_columns))
        try:
            _preflight_tabular_columns(
                features_resolved,
                required_columns=required_feature_columns,
                object_name="features",
            )
        except Exception as exc:
            failures.append(str(exc))

    if universe_path:
        try:
            universe_resolved = resolve_tabular_frame_path(universe_path, object_name="universe")
            _preflight_tabular_columns(
                universe_resolved,
                required_columns=("date", "asset", universe_col),
                object_name="universe",
            )
        except Exception as exc:
            failures.append(str(exc))

    try:
        _preflight_strict_split_for_spec(
            spec,
            object_name="model-lab",
            source="model_factor_submit_preflight",
        )
    except Exception as exc:
        failures.append(str(exc))

    if failures:
        bullet_lines = "\n".join(f"- {item}" for item in failures)
        raise AlphaLabIOError(
            "model-lab 启动前检查失败，请先修复数据/路径后再运行：\n" + bullet_lines
        )


def _preflight_tabular_columns(
    path: Path,
    *,
    required_columns: tuple[str, ...],
    object_name: str,
) -> None:
    required = tuple(
        dict.fromkeys(str(col).strip() for col in required_columns if str(col).strip())
    )
    if not required:
        return

    suffix = path.suffix.lower()
    import pandas as pd

    if suffix == ".csv":
        try:
            columns = [str(col) for col in pd.read_csv(path, nrows=0).columns]
        except Exception as exc:
            raise AlphaLabDataError(f"{object_name} 无法读取 CSV 头部: {path} ({exc})") from exc
        missing = [col for col in required if col not in set(columns)]
        if missing:
            raise AlphaLabDataError(
                f"{object_name} 缺少必需列: {missing} ({path})"
            )
        return

    if suffix not in {".parquet", ".pq"}:
        raise AlphaLabIOError(f"{object_name} 文件后缀不支持: {path}")

    try:
        import pyarrow.parquet as pq  # type: ignore[import-untyped]

        schema_columns = set(str(name) for name in pq.read_schema(path).names)
        missing = [col for col in required if col not in schema_columns]
        if missing:
            raise AlphaLabDataError(
                f"{object_name} 缺少必需列: {missing} ({path})"
            )
        return
    except ImportError:
        pass
    except Exception as exc:
        raise AlphaLabDataError(f"{object_name} 无法读取 Parquet schema: {path} ({exc})") from exc

    # Fallback when pyarrow schema is unavailable: read the selected columns.
    try:
        pd.read_parquet(path, columns=list(required))
    except Exception as exc:
        raise AlphaLabDataError(
            f"{object_name} Parquet 列检查失败，请确认列存在且可读取: {path} ({exc})"
        ) from exc


def _file_signature(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1 << 20)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()[:16]


def _extract_model_from_spec(spec_path: str) -> str:
    payload = _read_yaml_document_safe(spec_path)
    if payload is None:
        return ""
    model_payload = payload.get("model")
    if isinstance(model_payload, dict):
        model = _coerce_finite_or_text(model_payload.get("family"))
        if model:
            return model
    return _coerce_finite_or_text(payload.get("model_family")) or ""


def _coerce_feature_list(raw: object) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(item).strip() for item in raw if str(item).strip()]
    if isinstance(raw, str):
        parts = [part.strip() for part in raw.split(",") if part.strip()]
        if parts:
            return parts
        return [part.strip() for part in raw.split(";") if part.strip()]
    return []


def _ensure_run_summary(run: _RunRecord) -> dict[str, object]:
    summary: dict[str, object] = dict(run.summary)
    if summary:
        return _enrich_evaluation_summary(summary, run_status=run.status)

    metrics_payload = _read_run_payload(
        _resolve_run_artifact_path(
            run,
            artifact_key="metrics",
            fallback_name="metrics.json",
        )
    )
    if metrics_payload is None:
        return summary

    source = metrics_payload.get("metrics")
    metrics = source if isinstance(source, dict) else metrics_payload
    if isinstance(metrics, dict):
        summary = {str(key): value for key, value in metrics.items()}
    return _enrich_evaluation_summary(summary, run_status=run.status)


def _resolve_run_model_family(run: _RunRecord) -> str:
    summary = _ensure_run_summary(run)
    model_family = _coerce_finite_or_text(summary.get("model_family"))
    if model_family:
        return model_family

    model_definition_payload = _read_run_payload(
        _resolve_run_artifact_path(
            run,
            artifact_key="model_definition_json",
            fallback_name="model_definition.json",
        )
    )
    if isinstance(model_definition_payload, dict):
        value = _coerce_finite_or_text(model_definition_payload.get("model_family"))
        if value:
            return value

    manifest_payload = _read_run_payload(
        _resolve_run_artifact_path(
            run,
            artifact_key="run_manifest",
            fallback_name="run_manifest.json",
        )
    )
    if isinstance(manifest_payload, dict):
        inputs = manifest_payload.get("inputs")
        if isinstance(inputs, dict):
            value = _coerce_finite_or_text(inputs.get("model_family"))
            if value:
                return value

    value = _extract_model_from_spec(run.spec_path)
    return value or "-"


def _extract_model_factor_top_features(run: _RunRecord, *, top_k: int) -> list[str]:
    summary = _ensure_run_summary(run)
    for key in ("model_top_features", "top_features"):
        features = _coerce_feature_list(summary.get(key))
        if features:
            return features[:top_k]

    feature_path = _resolve_run_artifact_path(
        run,
        artifact_key="feature_importance",
        fallback_name="feature_importance.csv",
    )
    if feature_path is None:
        return []

    rows = _read_csv_artifact_rows(feature_path)
    with_importance: list[tuple[str, float]] = []
    for row in rows:
        feature = str(row.get("feature") or "").strip()
        if not feature:
            continue
        importance = _coerce_finite_float(row.get("mean_abs_importance"))
        if importance is None:
            importance = _coerce_finite_float(row.get("latest_importance"))
        if importance is None:
            importance = 0.0
        with_importance.append((feature, importance))

    with_importance.sort(key=lambda item: (item[1], item[0]), reverse=True)
    return [item[0] for item in with_importance[:top_k]]


def _collect_model_lab_run_compare_payload(
    run: _RunRecord,
    top_k_features: int,
) -> dict[str, object]:
    summary = _ensure_run_summary(run)
    metric_row: dict[str, object] = {
        "run_id": run.run_id,
        "case_name": run.case_name,
        "note": run.note or "",
        "factor_name": _resolve_run_factor_label(run),
        "status": run.status,
        "model_family": _resolve_run_model_family(run),
    }
    for key in _MODEL_LAB_COMPARE_METRIC_KEYS:
        if key == "factor_verdict":
            value = summary.get(key) or summary.get("factor_verdict")
        else:
            value = summary.get(key)
        if value is not None:
            metric_row[key] = value

    return {
        "top_features": _extract_model_factor_top_features(run, top_k=top_k_features),
        "failure_snapshot": _build_run_failure_snapshot(run),
        "metric_row": metric_row,
        "ic_series": _load_run_rank_ic_timeseries(run),
        "turnover_series": _load_run_turnover_timeseries(run),
        "leakage": _load_model_factor_run_leakage_summary(run),
        }


def _build_model_lab_run_spec_diff(records: list[_RunRecord]) -> dict[str, object]:
    if len(records) != 2:
        return {
            "status": "requires_two_runs",
            "message": "spec diff is shown when exactly two runs are selected",
        }
    left, right = records
    left_path = Path(left.spec_path)
    right_path = Path(right.spec_path)
    if not left_path.exists() or not right_path.exists():
        return {
            "status": "unavailable",
            "message": "one or both run spec files are unavailable",
            "left_run_id": left.run_id,
            "right_run_id": right.run_id,
            "left": left_path.name,
            "right": right_path.name,
        }
    left_text = left_path.read_text(encoding="utf-8").splitlines()
    right_text = right_path.read_text(encoding="utf-8").splitlines()
    left_payload = _read_yaml_document_safe(str(left_path))
    right_payload = _read_yaml_document_safe(str(right_path))
    semantic_equal_ignoring_meta = False
    if isinstance(left_payload, dict) and isinstance(right_payload, dict):
        semantic_equal_ignoring_meta = _strip_spec_diff_metadata(
            left_payload
        ) == _strip_spec_diff_metadata(right_payload)
    unified = "\n".join(
        difflib.unified_diff(
            left_text,
            right_text,
            fromfile=left_path.name,
            tofile=right_path.name,
            lineterm="",
        )
    )
    if semantic_equal_ignoring_meta:
        unified = ""
    return {
        "status": "ok",
        "left_run_id": left.run_id,
        "right_run_id": right.run_id,
        "left": left_path.name,
        "right": right_path.name,
        "unified": unified,
        "has_difference": bool(unified.strip()),
        "semantic_equal_ignoring_metadata": semantic_equal_ignoring_meta,
    }


def _build_top_feature_stability(
    top_features_by_run: dict[str, list[str]],
    *,
    run_count: int,
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    pairwise_scores: list[float] = []
    run_ids = list(top_features_by_run.keys())

    for left_index in range(len(run_ids)):
        left = run_ids[left_index]
        left_set = set(top_features_by_run.get(left, []))
        for right_index in range(left_index + 1, len(run_ids)):
            right = run_ids[right_index]
            right_set = set(top_features_by_run.get(right, []))
            union = left_set | right_set
            intersection = left_set & right_set
            if union:
                jaccard = len(intersection) / len(union)
            elif left_set or right_set:
                jaccard = 0.0
            else:
                jaccard = 1.0
            rows.append(
                {
                    "run_a": left,
                    "run_b": right,
                    "n_features_a": len(left_set),
                    "n_features_b": len(right_set),
                    "n_overlap": len(intersection),
                    "n_union": len(union),
                    "jaccard": jaccard,
                }
            )
            if union:
                pairwise_scores.append(jaccard)

    rows.sort(
        key=lambda item: float(_coerce_finite_float(item.get("jaccard")) or 0.0),
        reverse=True,
    )
    if pairwise_scores:
        mean_jaccard = sum(pairwise_scores) / len(pairwise_scores)
        min_jaccard = min(pairwise_scores)
        max_jaccard = max(pairwise_scores)
    else:
        mean_jaccard = None
        min_jaccard = None
        max_jaccard = None

    return {
        "run_count": run_count,
        "pair_count": len(rows),
        "pairwise": rows,
        "mean_jaccard": mean_jaccard,
        "min_jaccard": min_jaccard,
        "max_jaccard": max_jaccard,
    }


def _build_rank_ic_merge_rows(
    ic_series_by_run: dict[str, dict[str, float]],
) -> list[dict[str, object]]:
    rows = _build_metric_timeseries_rows(ic_series_by_run)
    for row in rows:
        row["mean_rank_ic"] = row.pop("mean_value", None)
    return rows


def _build_metric_timeseries_rows(
    metric_series_by_run: dict[str, dict[str, float]],
) -> list[dict[str, object]]:
    all_dates: set[str] = set()
    for series in metric_series_by_run.values():
        all_dates.update(series.keys())

    rows: list[dict[str, object]] = []
    for date in sorted(all_dates):
        row: dict[str, object] = {"date": date}
        date_values: list[float] = []
        for run_id, series in metric_series_by_run.items():
            value = series.get(date)
            if value is None:
                continue
            key = f"run:{run_id}"
            row[key] = value
            date_values.append(value)
        row["n_runs"] = len(date_values)
        row["mean_value"] = sum(date_values) / len(date_values) if date_values else None
        rows.append(row)
    return rows


def _load_model_factor_run_leakage_summary(run: _RunRecord) -> dict[str, object]:
    summary: dict[str, object] = {
        "run_id": run.run_id,
        "case_name": run.case_name,
        "factor_name": _resolve_run_factor_label(run),
        "status": run.status,
        "integrity_summary": {},
        "integrity_checks": [],
    }

    manifest_payload = _read_run_payload(
        _resolve_run_artifact_path(
            run,
            artifact_key="run_manifest",
            fallback_name="run_manifest.json",
        )
    )
    if isinstance(manifest_payload, dict):
        integrity_summary = manifest_payload.get("integrity_summary")
        if isinstance(integrity_summary, dict):
            summary["integrity_summary"] = {
                "n_checks": integrity_summary.get("n_checks"),
                "n_pass": integrity_summary.get("n_pass"),
                "n_warn": integrity_summary.get("n_warn"),
                "n_fail": integrity_summary.get("n_fail"),
                "highest_severity": integrity_summary.get("highest_severity"),
            }

    integrity_payload = _read_run_payload(
        _resolve_run_artifact_path(
            run,
            artifact_key="integrity_report_json",
            fallback_name="integrity_report.json",
        )
    )
    if isinstance(integrity_payload, dict):
        integrity_summary = integrity_payload.get("summary")
        if isinstance(integrity_summary, dict):
            summary["integrity_summary"] = {
                "n_checks": integrity_summary.get("n_checks"),
                "n_pass": integrity_summary.get("n_pass"),
                "n_warn": integrity_summary.get("n_warn"),
                "n_fail": integrity_summary.get("n_fail"),
                "highest_severity": integrity_summary.get("highest_severity"),
            }
        checks = integrity_payload.get("checks")
        if isinstance(checks, list):
            parsed: list[dict[str, object]] = []
            for item in checks:
                if not isinstance(item, dict):
                    continue
                check_name = str(item.get("check_name") or "").strip()
                if not check_name:
                    continue
                parsed.append(
                    {
                        "check_name": check_name,
                        "status": str(item.get("status") or "").strip(),
                        "severity": str(item.get("severity") or "").strip(),
                        "module_name": _coerce_finite_or_text(item.get("module_name")),
                        "object_name": _coerce_finite_or_text(item.get("object_name")),
                        "message": _coerce_finite_or_text(item.get("message")),
                        "remediation": _coerce_finite_or_text(item.get("remediation")),
                        "metrics": (
                            item.get("metrics")
                            if isinstance(item.get("metrics"), dict)
                            else {}
                        ),
                    }
                )
            summary["integrity_checks"] = parsed

    return summary


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


def _coerce_engines_payload(
    value: object,
) -> list[str] | tuple[str, ...] | str | None:
    """Narrow a JSON-decoded ``engines`` payload to the shapes Stage 0 accepts."""

    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, tuple):
        return tuple(str(item) for item in value)
    return None


def _coerce_builder_kwarg(value: object) -> object:
    if isinstance(value, str):
        text = value.strip()
        lowered = text.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        if lowered == "null":
            return None
        try:
            parsed_float = float(text)
        except ValueError:
            return text
        if math.isfinite(parsed_float):
            parsed_int = int(parsed_float)
            return parsed_int if parsed_float == parsed_int else parsed_float
        return text
    return value


def _parse_builder_kwargs(payload: Mapping[str, object]) -> dict[str, object]:
    explicit_fields = {
        "shock_gate_mode",
        "shock_q",
        "shock_threshold",
        "outside_event_policy",
        "neutralize_basic",
        "invert",
        "exclude_limit",
        "exclude_st",
        "exclude_suspended",
    }
    explicit_kwargs = {
        key: _coerce_builder_kwarg(payload[key])
        for key in explicit_fields
        if key in payload and payload[key] not in (None, "")
    }
    raw = payload.get("builder_kwargs")
    json_text = str(payload.get("builder_kwargs_json") or "").strip()
    if raw is None and not json_text:
        raw_kwargs: dict[str, object] = {}
    elif raw is None:
        try:
            raw = json.loads(json_text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"builder_kwargs_json must be valid JSON: {exc}") from exc
        if not isinstance(raw, Mapping):
            raise ValueError("builder_kwargs must be a JSON object")
        raw_kwargs = {str(key): _coerce_builder_kwarg(value) for key, value in raw.items()}
    else:
        if not isinstance(raw, Mapping):
            raise ValueError("builder_kwargs must be a JSON object")
        raw_kwargs = {str(key): _coerce_builder_kwarg(value) for key, value in raw.items()}
    raw_kwargs.update(explicit_kwargs)
    reserved = {"method", "lookback", "window", "skip_recent"}
    return {
        key: value
        for key, value in raw_kwargs.items()
        if str(key) not in reserved
    }


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


_MODEL_LAB_SPEC_NAME_PATTERN = re.compile(r"^stock_[a-z0-9]+(?:_[a-z0-9]+)*\.ya?ml$")
_MODEL_LAB_SPEC_STEM_MAX_LEN = 30
_MODEL_LAB_SPEC_NAME_HINT = (
    f"spec 文件名必须符合 stock_{{name}}.yaml，name 仅允许小写字母/数字，下划线分段，"
    f"文件名（不含后缀）≤ {_MODEL_LAB_SPEC_STEM_MAX_LEN} 字符；"
    f"例如 stock_ridge.yaml、stock_gbdt_smoke.yaml"
)
_MODEL_LAB_CANDIDATE_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{2,63}$")
_MODEL_LAB_CANDIDATE_CASE_FILENAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_.-]*\.ya?ml$")


def _safe_spec_filename(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        raise ValueError("target_name is required")

    # 阻止目录穿越并保留可读的文件名语义
    safe_name = re.sub(r"[/\\\\]+", "", raw)
    if not safe_name:
        raise ValueError("target_name is invalid")

    # 仅允许基本文件名字符
    safe_name = re.sub(r"[^A-Za-z0-9._-]", "-", safe_name)
    safe_name = re.sub(r"-{2,}", "-", safe_name).strip("-._")
    if not safe_name:
        raise ValueError("target_name is invalid")

    stem = Path(safe_name).stem
    suffix = Path(safe_name).suffix.lower()
    if suffix not in {".yaml", ".yml"}:
        suffix = ".yaml"

    # 规范化命名：统一 stock_{name}.yaml 以防止出现过长 / 旧式命名
    stem_norm = stem.lower().replace("-", "_").replace(".", "_")
    stem_norm = re.sub(r"_{2,}", "_", stem_norm).strip("_")
    if not stem_norm:
        raise ValueError("target_name is invalid")
    if not stem_norm.startswith("stock_"):
        stem_norm = f"stock_{stem_norm}"
    if len(stem_norm) > _MODEL_LAB_SPEC_STEM_MAX_LEN:
        raise ValueError(_MODEL_LAB_SPEC_NAME_HINT)
    normalized = f"{stem_norm}{suffix}"
    if not _MODEL_LAB_SPEC_NAME_PATTERN.match(normalized):
        raise ValueError(_MODEL_LAB_SPEC_NAME_HINT)
    return normalized


def _safe_model_candidate_name(value: str) -> str:
    normalized = str(value or "").strip().lower()
    if not _MODEL_LAB_CANDIDATE_NAME_PATTERN.match(normalized):
        raise ValueError(
            "candidate_name must be snake_case, start with a letter, and use "
            "3-64 lowercase chars"
        )
    return normalized


def _safe_candidate_case_filename(value: str) -> str:
    raw = str(value or "").strip().lower()
    if not raw:
        raise ValueError("target_name is required")
    if "/" in raw or "\\" in raw:
        raise PermissionError("invalid target spec path")
    if Path(raw).suffix.lower() not in {".yaml", ".yml"}:
        raw = f"{raw}.yaml"
    raw = re.sub(r"[^a-z0-9._-]", "_", raw)
    raw = re.sub(r"_{2,}", "_", raw).strip("._-")
    if Path(raw).suffix.lower() not in {".yaml", ".yml"}:
        raw = f"{Path(raw).stem}.yaml"
    if not _MODEL_LAB_CANDIDATE_CASE_FILENAME_PATTERN.match(raw):
        raise ValueError("target_name must be a simple YAML filename")
    return raw


def _coerce_available_fields(value: object) -> set[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        fields = {item.strip() for item in value.split(",") if item.strip()}
        return fields or None
    if isinstance(value, (list, tuple, set)):
        fields = {str(item).strip() for item in value if str(item).strip()}
        return fields or None
    return None


def _extract_model_candidate_payload(payload: dict[str, object]) -> dict[str, object]:
    for key in ("model_candidate_payload", "candidate", "payload"):
        value = payload.get(key)
        if isinstance(value, dict):
            return _normalize_model_candidate_payload(value)
    if _looks_like_model_candidate_payload(payload):
        return _normalize_model_candidate_payload(payload)

    for key in ("text", "content", "stage2_output"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            parsed = _extract_model_candidate_payload_from_text(value)
            if parsed is not None:
                return parsed
    raise ValueError(
        "model_candidate_payload not found; paste a JSON/YAML object with "
        "contract_version and case_spec_payload"
    )


def _extract_model_candidate_payload_from_text(text: str) -> dict[str, object] | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    whole = _parse_model_candidate_block(raw)
    if whole is not None:
        return whole
    fence_pattern = re.compile(r"```(?:json|yaml|yml)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
    for match in fence_pattern.finditer(raw):
        parsed = _parse_model_candidate_block(match.group(1).strip())
        if parsed is not None:
            return parsed
    return None


def _parse_model_candidate_block(text: str) -> dict[str, object] | None:
    for loader in (_parse_json_mapping, _parse_yaml_mapping):
        parsed = loader(text)
        if parsed is None:
            continue
        if isinstance(parsed.get("model_candidate_payload"), dict):
            parsed = cast(dict[str, object], parsed["model_candidate_payload"])
        if _looks_like_model_candidate_payload(parsed):
            return _normalize_model_candidate_payload(parsed)
    return None


def _parse_json_mapping(text: str) -> dict[str, object] | None:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return {str(key): value for key, value in parsed.items()}


def _parse_yaml_mapping(text: str) -> dict[str, object] | None:
    try:
        yaml = _require_yaml()
        parsed = yaml.safe_load(text)
    except Exception:
        return None
    if not isinstance(parsed, dict):
        return None
    return {str(key): value for key, value in parsed.items()}


def _looks_like_model_candidate_payload(payload: Mapping[str, object]) -> bool:
    return "contract_version" in payload and "case_spec_payload" in payload


def _normalize_model_candidate_payload(payload: Mapping[str, object]) -> dict[str, object]:
    normalized = {str(key): value for key, value in payload.items()}
    candidate_name = _safe_model_candidate_name(str(normalized.get("candidate_name") or ""))
    normalized["candidate_name"] = candidate_name
    case_spec_payload = normalized.get("case_spec_payload")
    if not isinstance(case_spec_payload, dict):
        raise ValueError("case_spec_payload must be an object")
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


def _safe_rmtree(path_text: str | None) -> bool:
    if not path_text:
        return False
    import shutil

    try:
        target = Path(path_text)
        if target.exists() and target.is_dir():
            shutil.rmtree(target)
            return True
    except OSError:
        return False
    return False


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
_ALPHA_LAB_OVERVIEW_FIXTURE_DIR = (
    Path(__file__).with_name("dev_fixtures") / "alpha_lab_overview"
)
_MODEL_LAB_OVERVIEW_FIXTURE_DIR = (
    Path(__file__).with_name("dev_fixtures") / "model_lab_overview"
)


def _md_render_js() -> str:
    """Load mdRender JS function from file with in-memory caching."""
    global _MD_RENDER_JS
    if _MD_RENDER_JS is None:
        _MD_RENDER_JS = _MD_RENDER_JS_PATH.read_text(encoding="utf-8")
    return _MD_RENDER_JS


# Cached inline HTML template cache path.
_INDEX_HTML_TEMPLATE_PATH = Path(__file__).with_name("web_unified_index.html")
_INDEX_HTML_TEMPLATE: str | None = None
_MODEL_LAB_HTML_TEMPLATE_PATH = Path(__file__).with_name("web_model_lab.html")
_MODEL_LAB_HTML_TEMPLATE: str | None = None


def _load_index_html_template() -> str:
    """Load frontend HTML template with in-memory caching."""
    global _INDEX_HTML_TEMPLATE
    if _INDEX_HTML_TEMPLATE is None:
        _INDEX_HTML_TEMPLATE = _INDEX_HTML_TEMPLATE_PATH.read_text(encoding="utf-8")
    return _INDEX_HTML_TEMPLATE


def _index_html(*, reload_template: bool = False) -> str:
    template = (
        _INDEX_HTML_TEMPLATE_PATH.read_text(encoding="utf-8")
        if reload_template
        else _index_html_raw()
    )
    return template.replace("@@MD_RENDER_JS@@", _md_render_js())


def _index_html_raw() -> str:
    return _load_index_html_template()


def _load_model_lab_html_template() -> str:
    global _MODEL_LAB_HTML_TEMPLATE
    if _MODEL_LAB_HTML_TEMPLATE is None:
        _MODEL_LAB_HTML_TEMPLATE = _MODEL_LAB_HTML_TEMPLATE_PATH.read_text(encoding="utf-8")
    return _MODEL_LAB_HTML_TEMPLATE


def _model_lab_html(*, reload_template: bool = False) -> str:
    template = (
        _MODEL_LAB_HTML_TEMPLATE_PATH.read_text(encoding="utf-8")
        if reload_template
        else _load_model_lab_html_template()
    )
    return template.replace("@@MD_RENDER_JS@@", _md_render_js())


def _safe_model_lab_fixture_id(value: str) -> str:
    fixture_id = str(value or "").strip()
    if not fixture_id:
        raise FileNotFoundError("overview fixture id is required")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", fixture_id):
        raise FileNotFoundError(f"invalid overview fixture id: {fixture_id}")
    return fixture_id


def _safe_alpha_lab_fixture_id(value: str) -> str:
    fixture_id = str(value or "").strip()
    if not fixture_id:
        raise FileNotFoundError("overview fixture id is required")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", fixture_id):
        raise FileNotFoundError(f"invalid overview fixture id: {fixture_id}")
    return fixture_id


def _list_alpha_lab_overview_fixtures() -> list[dict[str, object]]:
    if not _ALPHA_LAB_OVERVIEW_FIXTURE_DIR.exists():
        return []
    fixtures: list[dict[str, object]] = []
    for path in sorted(_ALPHA_LAB_OVERVIEW_FIXTURE_DIR.glob("*.json")):
        fixture_id = path.stem
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        fixtures.append(
            {
                "id": fixture_id,
                "label": str(payload.get("label") or fixture_id),
                "description": str(payload.get("description") or ""),
            }
        )
    return fixtures


def _load_alpha_lab_overview_fixture(fixture_id: str) -> dict[str, object]:
    safe_id = _safe_alpha_lab_fixture_id(fixture_id)
    path = (_ALPHA_LAB_OVERVIEW_FIXTURE_DIR / f"{safe_id}.json").resolve()
    fixture_root = _ALPHA_LAB_OVERVIEW_FIXTURE_DIR.resolve()
    if not path.exists() or fixture_root not in path.parents:
        raise FileNotFoundError(f"overview fixture not found: {safe_id}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"overview fixture must be a JSON object: {safe_id}")
    summary = payload.get("summary")
    snapshot = payload.get("snapshot")
    if not isinstance(summary, dict) or not isinstance(snapshot, dict):
        raise ValueError(f"overview fixture must include summary and snapshot objects: {safe_id}")
    run_id = str(payload.get("run_id") or f"fixture-{safe_id}")
    project_slug = str(payload.get("project_slug") or "alpha-lab1-fixture")
    return {
        "ok": True,
        "fixture_id": safe_id,
        "label": str(payload.get("label") or safe_id),
        "description": str(payload.get("description") or ""),
        "project_slug": project_slug,
        "run_id": run_id,
        "case_name": str(payload.get("case_name") or safe_id),
        "summary": summary,
        "snapshot": snapshot,
        "run": payload.get("run") if isinstance(payload.get("run"), dict) else {},
    }


def _list_model_lab_overview_fixtures() -> list[dict[str, object]]:
    if not _MODEL_LAB_OVERVIEW_FIXTURE_DIR.exists():
        return []
    fixtures: list[dict[str, object]] = []
    for path in sorted(_MODEL_LAB_OVERVIEW_FIXTURE_DIR.glob("*.json")):
        fixture_id = path.stem
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        fixtures.append(
            {
                "id": fixture_id,
                "label": str(payload.get("label") or fixture_id),
                "description": str(payload.get("description") or ""),
            }
        )
    return fixtures


def _load_model_lab_overview_fixture(fixture_id: str) -> dict[str, object]:
    safe_id = _safe_model_lab_fixture_id(fixture_id)
    path = (_MODEL_LAB_OVERVIEW_FIXTURE_DIR / f"{safe_id}.json").resolve()
    fixture_root = _MODEL_LAB_OVERVIEW_FIXTURE_DIR.resolve()
    if not path.exists() or fixture_root not in path.parents:
        raise FileNotFoundError(f"overview fixture not found: {safe_id}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"overview fixture must be a JSON object: {safe_id}")
    summary = payload.get("summary")
    snapshot = payload.get("snapshot")
    if not isinstance(summary, dict) or not isinstance(snapshot, dict):
        raise ValueError(f"overview fixture must include summary and snapshot objects: {safe_id}")
    run_id = str(payload.get("run_id") or f"fixture-{safe_id}")
    return {
        "ok": True,
        "fixture_id": safe_id,
        "label": str(payload.get("label") or safe_id),
        "description": str(payload.get("description") or ""),
        "run_id": run_id,
        "summary": summary,
        "snapshot": snapshot,
    }


def _artifact_fixture_content_type(artifact_key: str) -> str:
    key = artifact_key.strip().lower()
    if key in {
        "metrics",
        "run_manifest",
        "model_definition_json",
        "feature_manifest_json",
        "diagnostics",
    }:
        return "application/json; charset=utf-8"
    if key in {
        "training_log",
        "feature_importance",
        "feature_importance_ledger",
        "purged_kfold_folds",
    }:
        return "text/csv; charset=utf-8"
    if key in {"summary", "experiment_card"} or key.endswith("_markdown"):
        return "text/markdown; charset=utf-8"
    return "text/plain; charset=utf-8"


def _fixture_csv(rows: list[dict[str, object]], headers: list[str]) -> str:
    lines = [",".join(headers)]
    for row in rows:
        values: list[str] = []
        for header in headers:
            text = str(row.get(header, ""))
            if any(token in text for token in [",", "\n", '"']):
                text = '"' + text.replace('"', '""') + '"'
            values.append(text)
        lines.append(",".join(values))
    return "\n".join(lines) + "\n"


def _default_model_lab_artifact_fixture_content(
    fixture: dict[str, object],
    artifact_key: str,
) -> str:
    key = artifact_key.strip()
    raw_summary = fixture.get("summary")
    summary: dict[str, object] = raw_summary if isinstance(raw_summary, dict) else {}
    raw_snapshot = fixture.get("snapshot")
    snapshot: dict[str, object] = raw_snapshot if isinstance(raw_snapshot, dict) else {}
    run_id = str(fixture.get("run_id") or f"fixture-{fixture.get('fixture_id', 'artifact')}")
    if key == "metrics":
        return json.dumps({"metrics": summary}, ensure_ascii=False, indent=2)
    if key == "run_manifest":
        return json.dumps(
            {
                "run_id": run_id,
                "case_name": summary.get("case_name", "fixture_case"),
                "factor_name": summary.get("factor_name", "fixture_factor"),
                "workflow": "model_factor",
                "run_timestamp_utc": "2026-01-29T00:00:00Z",
                "outputs": {
                    "metrics": "fixture://metrics.json",
                    "training_log": "fixture://training_log.csv",
                    "feature_importance": "fixture://feature_importance.csv",
                    "summary": "fixture://summary.md",
                },
                "evaluation_standard": {"profile_name": "default_research"},
            },
            ensure_ascii=False,
            indent=2,
        )
    if key == "model_definition_json":
        return json.dumps(
            {
                "model_family": summary.get("model_family", "ridge"),
                "factor_name": summary.get("factor_name", "fixture_factor"),
                "target_horizon": 1,
                "rebalance_frequency": "D",
                "feature_columns": [
                    "mom_20d",
                    "turnover_rate_f",
                    "volatility_20d",
                    "size_zscore",
                    "valuation_pb",
                ],
                "hyperparameters": {"alpha": 1.0, "fit_intercept": True},
                "split_contract": summary.get("split_contract", {}),
            },
            ensure_ascii=False,
            indent=2,
        )
    if key == "feature_manifest_json":
        return json.dumps(
            {
                "artifact_type": "alpha_lab_feature_manifest",
                "factor_name": summary.get("factor_name", "fixture_factor"),
                "feature_columns": [
                    "mom_20d",
                    "turnover_rate_f",
                    "volatility_20d",
                    "size_zscore",
                    "valuation_pb",
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    if key == "summary" or key == "experiment_card":
        return (
            "# 实验摘要\n\n"
            "Strong signal metrics, but promotion was skipped because coverage and "
            "rolling evidence "
            "need additional review.\n\n"
            "## 关键结论\n\n"
            "- Signal: Strong\n"
            "- Promotion: skipped_not_promoted\n"
            "- Data Quality: warmup coverage breaks\n"
            "- Action: review coverage breaks, rolling stability, and NAV compounding sanity.\n"
        )
    if key == "training_log":
        rows: list[dict[str, object]] = [
            {
                "score_date": "2026-01-02",
                "status": "skipped",
                "model_version": "",
                "n_train_rows": "",
                "n_score_assets": "",
                "skip_reason": "insufficient_train_rows",
            },
            {
                "score_date": "2026-01-05",
                "status": "skipped",
                "model_version": "",
                "n_train_rows": "",
                "n_score_assets": "",
                "skip_reason": "insufficient_train_rows",
            },
            {
                "score_date": "2026-01-06",
                "status": "fit_scored",
                "model_version": 1,
                "n_train_rows": 12000,
                "n_score_assets": 4800,
                "skip_reason": "",
            },
            {
                "score_date": "2026-01-12",
                "status": "reused_scored",
                "model_version": 1,
                "n_train_rows": 12150,
                "n_score_assets": 4980,
                "skip_reason": "",
            },
            {
                "score_date": "2026-01-19",
                "status": "fit_scored",
                "model_version": 2,
                "n_train_rows": 13200,
                "n_score_assets": 4995,
                "skip_reason": "",
            },
            {
                "score_date": "2026-01-26",
                "status": "reused_scored",
                "model_version": 2,
                "n_train_rows": 13420,
                "n_score_assets": 4996,
                "skip_reason": "",
            },
        ]
        return _fixture_csv(
            rows,
            [
                "score_date",
                "status",
                "model_version",
                "n_train_rows",
                "n_score_assets",
                "skip_reason",
            ],
        )
    if key == "feature_importance":
        rows = [
            {
                "feature": "turnover_rate_f",
                "mean_abs_importance": 0.42,
                "latest_importance": 0.39,
                "sign_stability": 0.92,
                "importance_source": "fixture",
            },
            {
                "feature": "mom_20d",
                "mean_abs_importance": 0.25,
                "latest_importance": 0.28,
                "sign_stability": 0.88,
                "importance_source": "fixture",
            },
            {
                "feature": "volatility_20d",
                "mean_abs_importance": 0.16,
                "latest_importance": 0.18,
                "sign_stability": 0.77,
                "importance_source": "fixture",
            },
            {
                "feature": "size_zscore",
                "mean_abs_importance": 0.11,
                "latest_importance": 0.09,
                "sign_stability": 0.64,
                "importance_source": "fixture",
            },
            {
                "feature": "valuation_pb",
                "mean_abs_importance": 0.06,
                "latest_importance": 0.06,
                "sign_stability": 0.58,
                "importance_source": "fixture",
            },
        ]
        return _fixture_csv(
            rows,
            [
                "feature",
                "mean_abs_importance",
                "latest_importance",
                "sign_stability",
                "importance_source",
            ],
        )
    if key == "feature_importance_ledger":
        rows = [
            {
                "model_version": 1,
                "fit_date": "2026-01-06",
                "feature": "turnover_rate_f",
                "signed_importance": 0.40,
                "abs_importance": 0.40,
                "rank": 1,
                "importance_source": "fixture",
            },
            {
                "model_version": 1,
                "fit_date": "2026-01-06",
                "feature": "mom_20d",
                "signed_importance": 0.22,
                "abs_importance": 0.22,
                "rank": 2,
                "importance_source": "fixture",
            },
            {
                "model_version": 2,
                "fit_date": "2026-01-19",
                "feature": "turnover_rate_f",
                "signed_importance": 0.39,
                "abs_importance": 0.39,
                "rank": 1,
                "importance_source": "fixture",
            },
            {
                "model_version": 2,
                "fit_date": "2026-01-19",
                "feature": "mom_20d",
                "signed_importance": 0.28,
                "abs_importance": 0.28,
                "rank": 2,
                "importance_source": "fixture",
            },
        ]
        return _fixture_csv(
            rows,
            [
                "model_version",
                "fit_date",
                "feature",
                "signed_importance",
                "abs_importance",
                "rank",
                "importance_source",
            ],
        )
    if key == "purged_kfold_folds":
        rows = [
            {"fold": 1, "mean_ic": 0.021, "mean_rank_ic": 0.044, "n_dates": 20},
            {"fold": 2, "mean_ic": 0.028, "mean_rank_ic": 0.056, "n_dates": 20},
            {"fold": 3, "mean_ic": 0.024, "mean_rank_ic": 0.052, "n_dates": 20},
        ]
        return _fixture_csv(rows, ["fold", "mean_ic", "mean_rank_ic", "n_dates"])
    if key == "diagnostics":
        return json.dumps(
            {
                "artifact_type": "model_lab_diagnostics_fixture",
                "run_summary": {
                    "run_id": run_id,
                    "status": "succeeded",
                    "case_name": summary.get("case_name", "fixture_case"),
                    "factor_name": summary.get("factor_name", "fixture_factor"),
                    "highest_severity": "warn",
                    "warning_count": 1,
                    "error_count": 0,
                },
                "stages": [
                    {"stage": "load_data", "status": "ok", "duration_ms": 1200},
                    {"stage": "train_model", "status": "ok", "duration_ms": 3400},
                    {"stage": "artifact_export", "status": "warn", "duration_ms": 900},
                ],
                "events": [
                    {
                        "ts": "2026-01-29T00:00:00Z",
                        "level": "warning",
                        "stage": "artifact_export",
                        "message": "coverage breaks concentrated in warmup",
                        "payload": {"break_days": summary.get("coverage_break_days", 0)},
                    }
                ],
                "training_log": _default_model_lab_artifact_fixture_content(
                    fixture,
                    "training_log",
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    group_rows = snapshot.get("groupRows") if isinstance(snapshot, dict) else []
    if key == "group_returns" and isinstance(group_rows, list):
        return _fixture_csv(
            [dict(row) for row in group_rows if isinstance(row, dict)],
            ["date", "group", "group_return", "split_phase"],
        )
    return f"Fixture artifact '{key}' is not defined.\n"


def _load_model_lab_artifact_fixture(fixture_id: str, artifact_key: str) -> tuple[str, str]:
    fixture = _load_model_lab_overview_fixture(fixture_id)
    key = str(artifact_key or "").strip()
    if not key:
        raise FileNotFoundError("artifact key is required")
    artifacts = fixture.get("artifacts")
    if isinstance(artifacts, Mapping) and key in artifacts:
        raw = artifacts[key]
        if isinstance(raw, str):
            content = raw
        else:
            content = json.dumps(raw, ensure_ascii=False, indent=2)
    else:
        content = _default_model_lab_artifact_fixture_content(fixture, key)
    return _artifact_fixture_content_type(key), content
