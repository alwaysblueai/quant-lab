"""Unified Research Frontend — single local HTTP server integrating
Knowledge Ops, Bridge Workspace, Validation Console, and Writeback Review.

Provides the ``start_unified_server`` entry-point.
"""

from __future__ import annotations

import datetime as dt
import difflib
import hashlib
import json
import math
import re
import webbrowser
from collections.abc import Mapping
from csv import DictReader
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast
from urllib.parse import unquote

from alpha_lab.exceptions import (
    AlphaLabConfigError,
    AlphaLabDataError,
    AlphaLabExperimentError,
    AlphaLabIOError,
)
from alpha_lab.real_cases.common_io import (
    ensure_parquet_tabular_frame,
    resolve_tabular_frame_path,
)
from alpha_lab.research_bridge.models import load_yaml_document
from alpha_lab.research_bridge.service import PROJECTS_DIRNAME
from alpha_lab.splits import preflight_split_contract, rebalance_frequency_to_step
from alpha_lab.vault_export import resolve_vault_root

# HTTP request handler.
from alpha_lab.web_unified._handler import (
    _UnifiedRequestHandler as _UnifiedRequestHandler,
)

# Plain dataclasses + type aliases live in ``_models`` so subprocess /
# run_store / handler splits can import them without circular deps.
from alpha_lab.web_unified._models import RunStatus as RunStatus
from alpha_lab.web_unified._models import RunWorkflow as RunWorkflow
from alpha_lab.web_unified._models import _ModelLabSubprocessError as _ModelLabSubprocessError
from alpha_lab.web_unified._models import _RunTask as _RunTask
from alpha_lab.web_unified._models import _SubprocessCaseRunResult as _SubprocessCaseRunResult
from alpha_lab.web_unified._run_store import (
    _RUN_SUMMARY_COMPACT_KEYS as _RUN_SUMMARY_COMPACT_KEYS,
)

# Run record + run store — the in-memory queue/dispatcher.
from alpha_lab.web_unified._run_store import RunSuccessResult as RunSuccessResult
from alpha_lab.web_unified._run_store import (
    _compact_metrics_summary as _compact_metrics_summary,
)
from alpha_lab.web_unified._run_store import _RunRecord as _RunRecord
from alpha_lab.web_unified._run_store import _RunStore as _RunStore

# Subprocess orchestration helpers live in ``_subprocess``.
from alpha_lab.web_unified._subprocess import (
    _ARTIFACT_FALLBACK_FILENAMES as _ARTIFACT_FALLBACK_FILENAMES,  # noqa: E501
)
from alpha_lab.web_unified._subprocess import (
    _annotate_exception_with_model_lab_subprocess_artifacts as _annotate_exception_with_model_lab_subprocess_artifacts,  # noqa: E501
)
from alpha_lab.web_unified._subprocess import (
    _build_frontend_batch_parallel_config as _build_frontend_batch_parallel_config,
)
from alpha_lab.web_unified._subprocess import (
    _build_model_lab_batch_worker_count as _build_model_lab_batch_worker_count,
)
from alpha_lab.web_unified._subprocess import (
    _build_model_lab_subprocess_command as _build_model_lab_subprocess_command,
)
from alpha_lab.web_unified._subprocess import (
    _build_model_lab_subprocess_env as _build_model_lab_subprocess_env,
)
from alpha_lab.web_unified._subprocess import (
    _format_model_lab_subprocess_failure as _format_model_lab_subprocess_failure,
)
from alpha_lab.web_unified._subprocess import _format_run_error_text as _format_run_error_text
from alpha_lab.web_unified._subprocess import _format_shell_command as _format_shell_command
from alpha_lab.web_unified._subprocess import (
    _load_model_factor_artifact_paths_from_manifest as _load_model_factor_artifact_paths_from_manifest,  # noqa: E501
)
from alpha_lab.web_unified._subprocess import (
    _model_lab_subprocess_failure_hint as _model_lab_subprocess_failure_hint,
)
from alpha_lab.web_unified._subprocess import _parse_positive_int_env as _parse_positive_int_env
from alpha_lab.web_unified._subprocess import _parse_time_peak_rss_kb as _parse_time_peak_rss_kb
from alpha_lab.web_unified._subprocess import _read_text_tail as _read_text_tail
from alpha_lab.web_unified._subprocess import (
    _resolve_model_factor_web_cache_root_dir as _resolve_model_factor_web_cache_root_dir,
)
from alpha_lab.web_unified._subprocess import (
    _resolve_model_factor_web_output_parts as _resolve_model_factor_web_output_parts,
)
from alpha_lab.web_unified._subprocess import (
    _resolve_single_factor_web_output_root_dir as _resolve_single_factor_web_output_root_dir,
)
from alpha_lab.web_unified._subprocess import _wrap_command_with_time as _wrap_command_with_time
from alpha_lab.web_unified._subprocess import _write_json_file as _write_json_file

# Template + asset-path helpers live in ``_templates``; re-export everything
# that the rest of this module (and tests / monkeypatch consumers) reaches for.
from alpha_lab.web_unified._templates import (
    _ALPHA_LAB_OVERVIEW_FIXTURE_DIR as _ALPHA_LAB_OVERVIEW_FIXTURE_DIR,  # noqa: E501
)
from alpha_lab.web_unified._templates import _INDEX_HTML_TEMPLATE_PATH as _INDEX_HTML_TEMPLATE_PATH
from alpha_lab.web_unified._templates import _MD_RENDER_JS_PATH as _MD_RENDER_JS_PATH
from alpha_lab.web_unified._templates import (
    _MODEL_LAB_HTML_TEMPLATE_PATH as _MODEL_LAB_HTML_TEMPLATE_PATH,  # noqa: E501
)
from alpha_lab.web_unified._templates import (
    _MODEL_LAB_OVERVIEW_FIXTURE_DIR as _MODEL_LAB_OVERVIEW_FIXTURE_DIR,  # noqa: E501
)
from alpha_lab.web_unified._templates import _index_html as _index_html
from alpha_lab.web_unified._templates import _index_html_raw as _index_html_raw
from alpha_lab.web_unified._templates import _load_index_html_template as _load_index_html_template
from alpha_lab.web_unified._templates import (
    _load_model_lab_html_template as _load_model_lab_html_template,  # noqa: E501
)
from alpha_lab.web_unified._templates import _md_render_js as _md_render_js
from alpha_lab.web_unified._templates import _model_lab_html as _model_lab_html

# Small dependency-free helpers (slug + JSON-value coercion).
from alpha_lab.web_unified._utils import _coerce_finite_or_text as _coerce_finite_or_text
from alpha_lab.web_unified._utils import _safe_slug as _safe_slug

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
# _FRONTEND_BATCH_WINDOW_SECONDS / _MODEL_LAB_SUBPROCESS_POLL_SECONDS now
# live in ``alpha_lab.web_unified._run_store`` next to ``_RunStore``.
# _FRONTEND_BATCH_MAX_WORKERS / _FRONTEND_BATCH_FACTORS_PER_WORKER /
# _MODEL_LAB_BATCH_MAX_WORKERS / _MODEL_LAB_BATCH_DEFAULT_WORKERS now live in
# ``alpha_lab.web_unified._subprocess`` next to the helpers that consume them.
_RUN_OVERVIEW_MAX_CSV_ROWS: int = 20000

# _RUN_SUMMARY_COMPACT_KEYS now lives in
# ``alpha_lab.web_unified._run_store`` next to ``_compact_metrics_summary``.

# _ARTIFACT_FALLBACK_FILENAMES now lives in
# ``alpha_lab.web_unified._subprocess`` — see the re-export block at the top
# of this module.

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
        "specs_dir": project_dir / "40_specs",
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
    seen_names: set[str] = set()
    current_case_path = paths["current_case"]
    if current_case_path.exists():
        case_name = _yaml_case_name(current_case_path) or current_case_path.stem
        seen_names.add(case_name)
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
    specs_dir = paths.get("specs_dir")
    if specs_dir is not None and specs_dir.exists():
        spec_paths = [
            *sorted(specs_dir.glob("*.yaml")),
            *sorted(specs_dir.glob("*.yml")),
            *sorted(specs_dir.glob("*.md")),
        ]
        for spec_path in spec_paths:
            case_name = _yaml_case_name(spec_path) or spec_path.stem
            if not case_name or case_name in seen_names:
                continue
            seen_names.add(case_name)
            handoff_path = spec_path.with_name(f"{spec_path.stem}__knowledge_handoff.md")
            rows.append(
                {
                    "case_name": case_name,
                    "spec_path": str(spec_path),
                    "handoff_path": str(handoff_path),
                    "spec_exists": True,
                    "handoff_exists": handoff_path.exists(),
                    "is_current": False,
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
    normalized_case_name = case_name.strip()
    if current_case.exists():
        current_name = _yaml_case_name(current_case)
        if current_name == normalized_case_name or current_case.stem == normalized_case_name:
            return current_case
    specs_dir = paths.get("specs_dir")
    if specs_dir is not None and specs_dir.exists():
        for suffix in (".yaml", ".yml", ".md"):
            direct = specs_dir / f"{normalized_case_name}{suffix}"
            if direct.exists():
                return direct
        spec_paths = [
            *sorted(specs_dir.glob("*.yaml")),
            *sorted(specs_dir.glob("*.yml")),
            *sorted(specs_dir.glob("*.md")),
        ]
        for spec_path in spec_paths:
            if _yaml_case_name(spec_path) == normalized_case_name:
                return spec_path
    return current_case


def _yaml_case_name(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        payload = load_yaml_document(path)
    except Exception:
        return ""
    return str(payload.get("name") or "").strip()


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
    draft_paths = [
        *drafts_dir.glob("*__writeback_draft.md"),
        *drafts_dir.glob("*__archive_draft.md"),
    ]
    for draft_path in sorted(draft_paths, reverse=True):
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
        "workflow": str(frontmatter.get("workflow") or ""),
        "archive_identity": str(frontmatter.get("archive_identity") or ""),
        "archive_identity_inferred": bool(frontmatter.get("archive_identity_inferred")),
        "origin": str(frontmatter.get("origin") or ""),
        "audit_level": str(frontmatter.get("audit_level") or ""),
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


# ---------------------------------------------------------------------------
# HTML Frontend — 5-page single-page app
# ---------------------------------------------------------------------------


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


# _UnifiedService is imported last on purpose: it eagerly pulls helpers
# and constants from this module's namespace, so it must come after
# everything it depends on is defined.
from alpha_lab.web_unified._service import _UnifiedService as _UnifiedService  # noqa: E402
