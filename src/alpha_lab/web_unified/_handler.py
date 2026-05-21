"""HTTP request handler for the unified web frontend.

``_UnifiedRequestHandler`` dispatches GET/POST/PUT/PATCH/DELETE routes to
``_UnifiedService``. Kept thin — it parses URLs, validates payloads, and
serializes responses; the business logic lives on ``self.svc``.

Forward references (``_UnifiedService`` for the type annotation, and the
helper functions still in ``__init__.py``) are resolved lazily inside
method bodies to avoid the circular import that would otherwise fire when
``__init__.py`` imports this module during its own load.
"""

from __future__ import annotations

import json
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from typing import TYPE_CHECKING
from urllib.parse import parse_qs, urlparse

from alpha_lab.exceptions import AlphaLabDataError
from alpha_lab.research_bridge.categories import list_categories
from alpha_lab.web_unified._templates import _index_html, _model_lab_html
from alpha_lab.web_unified._utils import _coerce_finite_or_text  # noqa: F401  - kept for parity

if TYPE_CHECKING:
    from alpha_lab.web_unified._run_store import _RunRecord
    from alpha_lab.web_unified._service import _UnifiedService


# Maximum bytes returned inline for an artifact before we tell the browser
# to download / refuse. Mirrors the constant in ``__init__.py``.
_MAX_TEXT_BYTES: int = 512 * 1024
# Maximum request body size accepted from the browser.
_MAX_REQUEST_BODY_BYTES: int = 2 * 1024 * 1024


class _UnifiedRequestHandler(BaseHTTPRequestHandler):
    """HTTP handler for the unified web UI.

    ``do_GET`` / ``do_POST`` / ``do_PUT`` / ``do_PATCH`` / ``do_DELETE``
    are dispatched by name from ``BaseHTTPRequestHandler.handle_one_request``
    based on the request method; they have no visible Python callers.
    ``log_message`` is the base class's per-request stderr-logging hook,
    overridden here to a no-op so we don't spam access logs.
    """

    svc: _UnifiedService

    def do_GET(self) -> None:  # noqa: N802
        from alpha_lab.web_unified import (
            _as_int,
            _coerce_engines_payload,
            _list_alpha_lab_overview_fixtures,
            _list_model_lab_overview_fixtures,
            _load_alpha_lab_overview_fixture,
            _load_model_lab_artifact_fixture,
            _load_model_lab_overview_fixture,
            _path_parts,
            _safe_limit,
        )

        del _as_int, _coerce_engines_payload  # only POST handlers need these

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
        if (
            len(parts) == 6
            and parts[0] == "api"
            and parts[1] == "workflows"
            and parts[3] == "runs"
            and parts[5] == "archive-preview"
        ):
            try:
                self._send_json(self.svc.archive_preview(parts[2], parts[4]))
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
        from alpha_lab.web_unified import (
            _as_int,
            _coerce_engines_payload,
            _path_parts,
        )

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
            if (
                len(parts) == 6
                and parts[0] == "api"
                and parts[1] == "workflows"
                and parts[3] == "runs"
                and parts[5] == "archive-draft"
            ):
                self._send_json(
                    self.svc.archive_draft(parts[2], parts[4], payload),
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
                if len(parts) == 5 and parts[3] == "cases" and parts[4] == "claim":
                    self._send_json(self.svc.claim_backend_case(slug, payload))
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
        from alpha_lab.web_unified import _path_parts

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
        from alpha_lab.web_unified import _path_parts

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
        from alpha_lab.web_unified import _path_parts

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
        from alpha_lab.web_unified import _guess_content_type, _resolve_run_artifact_for_endpoint

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
        from alpha_lab.web_unified import _build_run_overview_snapshot

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
        from alpha_lab.web_unified import _build_run_overview_snapshot

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
        from alpha_lab.web_unified import _guess_content_type, _resolve_run_artifact_for_endpoint

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


__all__ = ["_UnifiedRequestHandler"]
