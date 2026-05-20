"""``_UnifiedService``: business-logic layer for the unified web frontend.

Methods are grouped by concern:

- LLM/secrets settings (``llm_settings_status`` / ``save_llm_settings``)
- model-lab specs/candidates/sources/runs/compare
- vault stats/inbox/cards/idea-distribute/preflight
- projects/cases/rounds/diagnostics
- writeback drafts (read/patch/apply)
- run submission/summarize/delete
- custom factor registry

This module is imported by ``__init__.py`` *after* all the helper
functions and constants the service needs are defined — see the
"Service" import block at the very bottom of ``__init__.py``. That
ordering is what lets us pull helpers eagerly from
``alpha_lab.web_unified`` here without hitting a circular-import error.
"""

from __future__ import annotations

import datetime as dt
import difflib
import hashlib
import json
import os
import re
import stat
import uuid
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from csv import DictReader
from pathlib import Path
from typing import cast

from alpha_lab.archive import (
    ARCHIVE_DRAFT_TYPE,
    ArchiveRunIndex,
    apply_archive_draft,
    build_archive_preview,
    cleanup_deprecated_writebacks,
    write_archive_draft,
)
from alpha_lab.baseline_factor_suite import baseline_factor_suite_payload
from alpha_lab.custom_factors import (
    BUILTIN_FACTOR_NAMES,
    compile_custom_factor,
    custom_factor_meta_path,
    custom_factor_write_path,
    iter_custom_factor_meta_paths,
    load_persisted_custom_factors,
)
from alpha_lab.custom_models import (
    model_candidate_write_path,
    read_draft_model_source,
)
from alpha_lab.draft_model_validation import validate_draft_model_file
from alpha_lab.exceptions import AlphaLabConfigError
from alpha_lab.factor_recipe import factor_registry
from alpha_lab.real_cases.model_factor.spec import load_model_factor_case_spec
from alpha_lab.real_cases.single_factor.spec import load_single_factor_case_spec
from alpha_lab.research_bridge.categories import get_category_profile
from alpha_lab.research_bridge.graph_view import VaultGraph
from alpha_lab.research_bridge.mechanism_index import (
    mechanism_index_status as bridge_mechanism_index_status,
)
from alpha_lab.research_bridge.models import (
    load_project_config,
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
from alpha_lab.vault_export import export_to_vault
from alpha_lab.web_unified import (
    _KNOWLEDGE_WRITEBACK_CARD_TYPES,
    _KNOWLEDGE_WRITEBACK_STAGES,
    _MAX_REPORT_TEXT_BYTES,
    _MAX_TEXT_BYTES,
    _MODEL_LAB_COMPARE_METRIC_KEYS,
    _MODEL_LAB_MAX_COMPARE_RUNS,
    _MODEL_LAB_PROJECT_SLUG,
    _MODEL_LAB_SOURCE_SPECS,
    _PROJECT_DOC_PREVIEW_BYTES,
    _WEB_SECRET_SETTINGS_REL_PATH,
    _apply_knowledge_writeback_draft,
    _as_int,
    _as_text_list,
    _build_metric_timeseries_rows,
    _build_model_lab_run_spec_diff,
    _build_project_dsr_summary,
    _build_rank_ic_merge_rows,
    _build_spec_lineage_meta,
    _build_top_feature_stability,
    _classify_dsr_pvalue,
    _coerce_available_fields,
    _coerce_finite_float,
    _coerce_source_artifacts,
    _coerce_spec_version,
    _coerce_text_list,
    _collect_model_lab_run_compare_payload,
    _compact_metrics_summary,
    _compose_markdown_with_frontmatter,
    _default_knowledge_target_path,
    _derive_evaluation_action_and_next_step,
    _derive_factor_name_from_spec_stem,
    _draft_summary,
    _dump_spec_payload,
    _ensure_run_summary,
    _extract_metrics_summary,
    _extract_model_candidate_payload,
    _extract_spec_lineage,
    _iso_from_timestamp,
    _iter_project_contracts,
    _list_cases,
    _list_draft_summaries,
    _list_rounds,
    _load_markdown_with_frontmatter,
    _load_run_draft_model_source,
    _load_run_rank_ic_timeseries,
    _next_spec_version,
    _normalize_knowledge_target_hint,
    _optional_text,
    _pairwise_spearman_from_timeseries,
    _parse_builder_kwargs,
    _preflight_model_lab_spec_inputs,
    _preflight_strict_split_for_spec,
    _project_paths,
    _read_text_preview,
    _read_text_with_limit,
    _read_yaml_document_safe,
    _render_knowledge_writeback_draft_body,
    _resolve_case_spec_path,
    _resolve_draft_path,
    _resolve_run_artifact_path,
    _resolve_run_dsr_pvalue,
    _resolve_run_evaluation_title,
    _resolve_run_factor_label,
    _safe_candidate_case_filename,
    _safe_file_stem,
    _safe_model_candidate_name,
    _safe_rmtree,
    _safe_spec_filename,
    _strip_spec_diff_metadata,
    _utc_now_iso,
)
from alpha_lab.web_unified._models import RunWorkflow, _RunTask
from alpha_lab.web_unified._run_store import _RunRecord, _RunStore
from alpha_lab.web_unified._utils import _coerce_finite_or_text, _safe_slug

_BACKEND_SINGLE_FACTOR_CASE_ROOT = Path("configs") / "real_cases" / "single_factor"
_CLAIMED_BACKEND_CASES_FILENAME = "claimed_backend_cases.json"


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _clean_text(value: object) -> str:
    return str(value or "").strip()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _case_row_updated_at_epoch(row: Mapping[str, object]) -> float:
    value = row.get("updated_at_epoch")
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0


class _UnifiedService:
    def __init__(self, *, vault_root: Path, workspace_root: Path) -> None:
        self.vault_root = vault_root.resolve()
        self.workspace_root = workspace_root.resolve()
        self.run_store = _RunStore()
        self._custom_factors_dir = self.workspace_root / "custom_factors"
        self._apply_saved_llm_settings()
        self._load_persisted_custom_factors()
        self._restore_completed_web_runs()
        self.archive_index = ArchiveRunIndex.build(
            workspace_root=self.workspace_root,
            records=self.run_store.list_records(),
        )
        try:
            cleanup_deprecated_writebacks(vault_root=self.vault_root)
        except OSError:
            pass

    @property
    def projects_root(self) -> Path:
        return (self.vault_root / PROJECTS_DIRNAME).resolve()

    @property
    def model_lab_specs_root(self) -> Path:
        return (self.workspace_root / "configs" / "real_cases" / "model_factor").resolve()

    @property
    def single_factor_specs_root(self) -> Path:
        return (self.workspace_root / "configs" / "real_cases" / "single_factor").resolve()

    @property
    def model_lab_candidates_root(self) -> Path:
        return (self.workspace_root / "custom_models" / "research").resolve()

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

    def _restore_completed_web_runs(self) -> None:
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
                case_report_path = output_dir / "case_report.md"
                if case_report_path.exists():
                    artifact_paths.setdefault("case_report", str(case_report_path))
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
                is_model_factor = (
                    (output_dir / "model_definition.json").exists()
                    or (output_dir / "feature_manifest.json").exists()
                    or isinstance(manifest.get("draft_model_source"), Mapping)
                )
                is_single_factor = (output_dir / "factor_definition.json").exists()
                workflow: RunWorkflow
                if is_model_factor:
                    project_slug = _MODEL_LAB_PROJECT_SLUG
                    workflow = "model_factor"
                elif is_single_factor:
                    restored_project_slug = self._restore_single_factor_project_slug(manifest)
                    if restored_project_slug is None:
                        continue
                    project_slug = restored_project_slug
                    workflow = "single_factor"
                else:
                    continue
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
                    project_slug=project_slug,
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
                    workflow=workflow,
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

    def _restore_single_factor_project_slug(self, manifest: Mapping[str, object]) -> str | None:
        spec_path_raw = _coerce_finite_or_text(manifest.get("spec_path"))
        if spec_path_raw:
            try:
                spec_path = Path(spec_path_raw).expanduser().resolve()
                rel = spec_path.relative_to(self.projects_root)
                if len(rel.parts) >= 2:
                    return rel.parts[0]
            except (OSError, ValueError):
                pass
        return None

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
        # __file__ is src/alpha_lab/web_unified/__init__.py; parents[3] == repo root
        repo_root = Path(__file__).resolve().parents[3]
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
        if not paths["project_yaml"].exists():
            return []
        project = load_project_config(paths["project_yaml"])
        project_rows = [
            self._decorate_project_case_row(row, slug=project.slug, project=project)
            for row in _list_cases(paths)
        ]
        claim_owners = self._claim_owner_by_spec_path()
        backend_rows = [
            self._decorate_backend_case_row(
                row,
                slug=project.slug,
                project=project,
                claim_owners=claim_owners,
            )
            for row in self._list_backend_single_factor_cases()
        ]
        self._mark_recommended_backend_case(
            backend_rows,
            archive_identity=self._current_project_archive_identity(paths),
            factor_name=self._current_project_factor_name(paths),
        )
        return [*backend_rows, *project_rows]

    def _list_backend_single_factor_cases(self) -> list[dict[str, object]]:
        specs_root = self.single_factor_specs_root
        if not specs_root.exists():
            return []
        rows: list[dict[str, object]] = []
        for spec_path in [*sorted(specs_root.glob("*.yaml")), *sorted(specs_root.glob("*.yml"))]:
            if spec_path.name.startswith("_"):
                continue
            raw = _read_yaml_document_safe(str(spec_path)) or {}
            case_name = _clean_text(raw.get("name") or spec_path.stem)
            factor_name = _clean_text(raw.get("factor_name"))
            if not case_name:
                continue
            stat_result = spec_path.stat()
            rows.append(
                {
                    "case_name": case_name,
                    "factor_name": factor_name,
                    "archive_identity": _clean_text(
                        raw.get("archive_identity") or raw.get("factor_name")
                    ),
                    "project_slug": _clean_text(raw.get("project_slug")),
                    "evaluation_profile": _clean_text(raw.get("evaluation_profile")),
                    "evaluation_profile_source": (
                        "case_spec"
                        if _clean_text(raw.get("evaluation_profile"))
                        else "project_default"
                    ),
                    "spec_path": str(spec_path),
                    "spec_relative_path": self._workspace_relative_path(spec_path),
                    "spec_preview": _read_text_preview(
                        spec_path,
                        limit_bytes=_PROJECT_DOC_PREVIEW_BYTES,
                    ),
                    "spec_summary": self._single_factor_spec_summary(raw),
                    "handoff_path": "",
                    "spec_exists": True,
                    "handoff_exists": False,
                    "is_current": False,
                    "is_recommended": False,
                    "source": "backend_optimized",
                    "updated_at": dt.datetime.fromtimestamp(
                        stat_result.st_mtime,
                        dt.UTC,
                    ).isoformat(),
                    "updated_at_epoch": stat_result.st_mtime,
                }
            )
        return sorted(
            rows,
            key=_case_row_updated_at_epoch,
            reverse=True,
        )

    def _resolve_backend_single_factor_case_spec_path(self, case_name: str) -> Path | None:
        normalized = case_name.strip()
        if not normalized:
            return None
        specs_root = self.single_factor_specs_root
        if not specs_root.exists():
            return None
        for suffix in (".yaml", ".yml"):
            direct = specs_root / f"{normalized}{suffix}"
            if direct.exists():
                return direct.resolve()
        for row in self._list_backend_single_factor_cases():
            if str(row.get("case_name") or "") == normalized:
                return Path(str(row["spec_path"])).resolve()
        return None

    def claim_backend_case(self, slug: str, payload: dict[str, object]) -> dict[str, object]:
        paths = _project_paths(self.vault_root, slug)
        if not paths["project_yaml"].exists():
            raise FileNotFoundError(f"project not found: {slug}")
        project = load_project_config(paths["project_yaml"])
        spec_path, relative_path = self._resolve_claimable_backend_case_path(
            payload.get("spec_path")
        )
        raw = _read_yaml_document_safe(str(spec_path)) or {}
        yaml_project = _clean_text(raw.get("project_slug"))
        if yaml_project:
            if _safe_slug(yaml_project) == project.slug:
                return self._claim_response(
                    project_slug=project.slug,
                    spec_path=spec_path,
                    relative_path=relative_path,
                    status="already_owned_by_project_slug",
                    claimed=False,
                    raw=raw,
                )
            raise AlphaLabConfigError(
                f"backend case is owned by another project: {yaml_project}"
            )

        claim_owners = self._claim_owner_by_spec_path()
        owner = claim_owners.get(relative_path, "")
        if owner:
            if owner == project.slug:
                return self._claim_response(
                    project_slug=project.slug,
                    spec_path=spec_path,
                    relative_path=relative_path,
                    status="already_claimed_by_project",
                    claimed=False,
                    raw=raw,
                )
            raise AlphaLabConfigError(f"backend case is already claimed by project: {owner}")

        claims = self._load_claims(project.slug)
        claim: dict[str, object] = {
            "spec_path": relative_path,
            "case_name": _clean_text(raw.get("name") or spec_path.stem),
            "factor_name": _clean_text(raw.get("factor_name")),
            "archive_identity": _clean_text(raw.get("archive_identity") or raw.get("factor_name")),
            "spec_sha256_at_claim": _file_sha256(spec_path),
            "claimed_at_utc": _utc_now_iso(),
            "claimed_by": "web_unified",
        }
        claims.append(claim)
        self._write_claims(project.slug, claims)
        return self._claim_response(
            project_slug=project.slug,
            spec_path=spec_path,
            relative_path=relative_path,
            status="claimed",
            claimed=True,
            raw=raw,
        )

    def _claim_response(
        self,
        *,
        project_slug: str,
        spec_path: Path,
        relative_path: str,
        status: str,
        claimed: bool,
        raw: Mapping[str, object],
    ) -> dict[str, object]:
        return {
            "ok": True,
            "project_slug": project_slug,
            "status": status,
            "claimed": claimed,
            "spec_path": str(spec_path),
            "spec_relative_path": relative_path,
            "case_name": _clean_text(raw.get("name") or spec_path.stem),
            "factor_name": _clean_text(raw.get("factor_name")),
            "archive_identity": _clean_text(raw.get("archive_identity") or raw.get("factor_name")),
        }

    def _decorate_project_case_row(
        self,
        row: dict[str, object],
        *,
        slug: str,
        project: object,
    ) -> dict[str, object]:
        spec_path = Path(str(row.get("spec_path") or ""))
        raw = _read_yaml_document_safe(str(spec_path)) or {}
        evaluation_profile = _clean_text(raw.get("evaluation_profile"))
        project_default = _clean_text(
            getattr(getattr(project, "alpha_lab_defaults", None), "evaluation_profile", "")
        )
        return {
            **row,
            "source": "project",
            "project_slug": slug,
            "claimed_by_project": "",
            "claim_status": "project_local",
            "archive_identity": _clean_text(raw.get("archive_identity") or raw.get("factor_name")),
            "factor_name": _clean_text(raw.get("factor_name")),
            "evaluation_profile": evaluation_profile or project_default,
            "evaluation_profile_source": "case_spec" if evaluation_profile else "project_default",
            "is_recommended": False,
            "recommendation_reason": "",
            "requires_explicit_selection": False,
            "spec_preview": _read_text_preview(
                spec_path,
                limit_bytes=_PROJECT_DOC_PREVIEW_BYTES,
            ),
            "spec_summary": self._single_factor_spec_summary(raw),
        }

    def _decorate_backend_case_row(
        self,
        row: dict[str, object],
        *,
        slug: str,
        project: object,
        claim_owners: Mapping[str, str],
    ) -> dict[str, object]:
        relative_path = _clean_text(row.get("spec_relative_path"))
        yaml_project = _clean_text(row.get("project_slug"))
        claim_owner = _clean_text(claim_owners.get(relative_path, ""))
        evaluation_profile = _clean_text(row.get("evaluation_profile"))
        project_default = _clean_text(
            getattr(getattr(project, "alpha_lab_defaults", None), "evaluation_profile", "")
        )
        claim_status = "unclaimed"
        if yaml_project:
            claim_status = (
                "owned_by_current_project"
                if _safe_slug(yaml_project) == slug
                else "owned_by_other_project"
            )
        elif claim_owner:
            claim_status = (
                "claimed_by_current_project"
                if claim_owner == slug
                else "owned_by_other_project"
            )
        decorated = dict(row)
        decorated.update(
            {
                "claimed_by_project": claim_owner,
                "claim_status": claim_status,
                "requires_explicit_selection": claim_status == "unclaimed",
                "evaluation_profile": evaluation_profile or project_default,
                "is_recommended": False,
                "recommendation_reason": "",
            }
        )
        return decorated

    def _mark_recommended_backend_case(
        self,
        rows: list[dict[str, object]],
        *,
        archive_identity: str,
        factor_name: str,
    ) -> None:
        eligible = [
            row
            for row in rows
            if row.get("claim_status") in {"owned_by_current_project", "claimed_by_current_project"}
        ]
        if not eligible:
            return
        selected: dict[str, object] | None = None
        reason = "project_latest"
        if archive_identity:
            matches = [
                row
                for row in eligible
                if _clean_text(row.get("archive_identity")) == archive_identity
            ]
            if matches:
                selected = self._latest_backend_case(matches)
                reason = "archive_identity_match"
        if selected is None and factor_name:
            matches = [
                row for row in eligible if _clean_text(row.get("factor_name")) == factor_name
            ]
            if matches:
                selected = self._latest_backend_case(matches)
                reason = "factor_name_match"
        if selected is None:
            selected = self._latest_backend_case(eligible)
        if selected is None:
            return
        selected["is_recommended"] = True
        selected["recommendation_reason"] = reason

    def _latest_backend_case(self, rows: list[dict[str, object]]) -> dict[str, object] | None:
        if not rows:
            return None
        return sorted(
            rows,
            key=_case_row_updated_at_epoch,
            reverse=True,
        )[0]

    def _current_project_archive_identity(self, paths: dict[str, Path]) -> str:
        payload = _read_yaml_document_safe(str(paths["current_case"])) or {}
        return _clean_text(payload.get("archive_identity"))

    def _current_project_factor_name(self, paths: dict[str, Path]) -> str:
        payload = _read_yaml_document_safe(str(paths["current_case"])) or {}
        return _clean_text(payload.get("factor_name"))

    def _single_factor_spec_summary(self, payload: Mapping[str, object]) -> dict[str, object]:
        target = payload.get("target") if isinstance(payload.get("target"), Mapping) else {}
        universe = payload.get("universe") if isinstance(payload.get("universe"), Mapping) else {}
        return {
            "factor_name": _clean_text(payload.get("factor_name")),
            "target_horizon": target.get("horizon") if isinstance(target, Mapping) else None,
            "execution_price_mode": (
                _clean_text(target.get("execution_price_mode"))
                if isinstance(target, Mapping)
                else ""
            ),
            "rebalance_frequency": _clean_text(payload.get("rebalance_frequency")),
            "direction": _clean_text(payload.get("direction")),
            "prices_path": _clean_text(payload.get("prices_path")),
            "universe_path": (
                _clean_text(universe.get("path")) if isinstance(universe, Mapping) else ""
            ),
            "factor_path": _clean_text(payload.get("factor_path")),
        }

    def _workspace_relative_path(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.workspace_root).as_posix()
        except ValueError:
            return path.resolve().as_posix()

    def _resolve_claimable_backend_case_path(self, value: object) -> tuple[Path, str]:
        raw = _clean_text(value)
        if not raw:
            raise ValueError("spec_path is required")
        candidate = Path(raw).expanduser()
        resolved = (
            candidate.resolve()
            if candidate.is_absolute()
            else (self.workspace_root / candidate).resolve()
        )
        if resolved.parent != self.single_factor_specs_root:
            raise PermissionError("spec_path must be a one-level single-factor backend case YAML")
        if resolved.suffix.lower() not in {".yaml", ".yml"}:
            raise PermissionError("spec_path must be a YAML single-factor backend case")
        if not resolved.exists() or not resolved.is_file():
            raise FileNotFoundError(f"case spec does not exist: {resolved}")
        try:
            relative = resolved.relative_to(self.workspace_root).as_posix()
        except ValueError as exc:
            raise PermissionError("spec_path must be under the workspace root") from exc
        rel_path = Path(relative)
        if rel_path.parent != _BACKEND_SINGLE_FACTOR_CASE_ROOT:
            raise PermissionError("spec_path must be under configs/real_cases/single_factor")
        return resolved, relative

    def _claims_path(self, slug: str) -> Path:
        return (
            _project_paths(self.vault_root, slug)["project_dir"]
            / _CLAIMED_BACKEND_CASES_FILENAME
        )

    def _load_claims(self, slug: str) -> list[dict[str, object]]:
        path = self._claims_path(slug)
        if not path.exists():
            return []
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        claims = payload.get("claims") if isinstance(payload, dict) else None
        if not isinstance(claims, list):
            return []
        return [dict(item) for item in claims if isinstance(item, dict)]

    def _write_claims(self, slug: str, claims: list[dict[str, object]]) -> None:
        path = self._claims_path(slug)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "claims": sorted(claims, key=lambda item: _clean_text(item.get("spec_path"))),
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _claim_owner_by_spec_path(self) -> dict[str, str]:
        owners: dict[str, str] = {}
        if not self.projects_root.exists():
            return owners
        for project_yaml in _iter_project_contracts(self.projects_root):
            try:
                project = load_project_config(project_yaml)
                slug = project.slug
            except Exception:
                slug = _safe_slug(project_yaml.parent.name)
            if not slug:
                continue
            for claim in self._load_claims(slug):
                rel_path = _clean_text(claim.get("spec_path"))
                if rel_path and rel_path not in owners:
                    owners[rel_path] = slug
        return owners

    def _resolve_allowed_single_factor_spec_path(
        self,
        *,
        project_paths: dict[str, Path],
        raw_path: str,
    ) -> Path:
        candidate = Path(raw_path).expanduser().resolve()
        allowed_roots = [
            project_paths["project_dir"].resolve(),
            self.single_factor_specs_root.resolve(),
        ]
        if not any(_is_relative_to(candidate, root) for root in allowed_roots):
            raise PermissionError(
                "spec_path must be under the project or backend case config roots"
            )
        if not candidate.exists():
            raise FileNotFoundError(f"case spec does not exist: {candidate}")
        return candidate

    def list_drafts(self, slug: str) -> list[dict[str, object]]:
        drafts_dir = _project_paths(self.vault_root, slug)["drafts_dir"]
        project_yaml = _project_paths(self.vault_root, slug)["project_yaml"]
        if not project_yaml.exists() and not drafts_dir.exists():
            raise FileNotFoundError(f"project not found: {slug}")
        return _list_draft_summaries(drafts_dir)

    def _refresh_archive_index(self) -> None:
        self.archive_index.sync_run_records(self.run_store.list_records())

    def archive_preview(self, workflow: str, run_id: str) -> dict[str, object]:
        self._refresh_archive_index()
        return build_archive_preview(
            index=self.archive_index,
            vault_root=self.vault_root,
            workflow=workflow,
            run_id=run_id,
        )

    def archive_draft(
        self,
        workflow: str,
        run_id: str,
        payload: dict[str, object],
    ) -> dict[str, object]:
        preview = self.archive_preview(workflow, run_id)
        record = self.archive_index.get(run_id)
        project_slug = ""
        if record is not None:
            project_slug = _safe_slug(record.project_slug or "")
        if not project_slug:
            project_slug = _MODEL_LAB_PROJECT_SLUG if workflow == "model_factor" else "__archive__"
        result = write_archive_draft(
            vault_root=self.vault_root,
            project_slug=project_slug,
            preview=preview,
            payload=payload,
        )
        result["project_slug"] = project_slug
        return result

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
        if not paths["project_yaml"].exists() and not paths["drafts_dir"].exists():
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
        if not paths["project_yaml"].exists() and not paths["drafts_dir"].exists():
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
        if not paths["project_yaml"].exists() and not paths["drafts_dir"].exists():
            raise FileNotFoundError(f"project not found: {slug}")
        draft_path = _resolve_draft_path(paths["drafts_dir"], draft_name)
        mode = _optional_text(payload.get("mode")) if payload is not None else None
        frontmatter, body = _load_markdown_with_frontmatter(draft_path)
        if str(frontmatter.get("type") or "").strip() == ARCHIVE_DRAFT_TYPE:
            return apply_archive_draft(
                vault_root=self.vault_root,
                draft_path=draft_path,
                frontmatter=frontmatter,
                body=body,
                mode=mode,
            )
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
        explicit_spec_path = _optional_text(payload.get("spec_path"))
        if explicit_spec_path is not None:
            spec_path = self._resolve_allowed_single_factor_spec_path(
                project_paths=paths,
                raw_path=explicit_spec_path,
            )
        else:
            spec_path = _resolve_case_spec_path(paths, case_name)
            should_try_backend = not spec_path.exists()
            if spec_path == paths["current_case"] and spec_path.exists():
                raw_current_case = _read_yaml_document_safe(str(spec_path)) or {}
                current_case_name = _clean_text(raw_current_case.get("name"))
                should_try_backend = (
                    current_case_name != case_name and spec_path.stem != case_name
                )
            if should_try_backend:
                backend_spec_path = self._resolve_backend_single_factor_case_spec_path(case_name)
                if backend_spec_path is not None:
                    spec_path = backend_spec_path
        if not spec_path.exists():
            raise FileNotFoundError(f"case spec does not exist: {spec_path}")
        project = load_project_config(paths["project_yaml"])
        spec = load_single_factor_case_spec(spec_path)
        case_name = spec.name
        raw_spec = _read_yaml_document_safe(str(spec_path)) or {}
        requested_profile = _optional_text(payload.get("evaluation_profile"))
        spec_profile = _optional_text(raw_spec.get("evaluation_profile"))
        if requested_profile is not None:
            evaluation_profile = requested_profile
            evaluation_profile_source = "request"
        elif spec_profile is not None:
            evaluation_profile = spec_profile
            evaluation_profile_source = "case_spec"
        else:
            evaluation_profile = project.alpha_lab_defaults.evaluation_profile
            evaluation_profile_source = "project_default"
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
            evaluation_profile_source=evaluation_profile_source,
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


__all__ = ["_UnifiedService"]
