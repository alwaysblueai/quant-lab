"""File-backed alpha-lab explorer session memory (P3).

Each ``explore_idea`` call may persist a session row at:

    <workspace_root>/artifacts/alpha_lab_explorer/sessions/<session_id>.json

The row starts life with no response. Once the user has run the prompt
externally and pasted the LLM's response back via
``record_explore_response``, we run the stage-aware lint and store the
report on the same row. ``list_recent_violations`` then exposes recent
violations to the next ``explore_idea`` call so they can be injected
into the prompt as a "known drift patterns" header — closing the
self-improving prompt loop.

This is a deliberately small, file-based store. We don't need a
database; sessions are short-lived research artifacts and the user
already trusts the workspace dir.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from alpha_lab.research_bridge.output_lint import (
    LintReport,
    extract_stage_sections,
    lint_explore_response,
)
from alpha_lab.research_bridge.scoring import (
    MECHANISM_DISCOVERY,
    SIGNAL_MAPPING,
    VALIDATION_KILL_TESTS,
    normalize_workflow_stage,
)

_SESSIONS_REL = Path("artifacts") / "alpha_lab_explorer" / "sessions"


@dataclass(frozen=True, slots=True)
class ExploreSessionStartResult:
    session_id: str
    created_at_utc: str
    path: Path


def explore_sessions_root(workspace_root: str | Path) -> Path:
    return Path(workspace_root).expanduser().resolve() / _SESSIONS_REL


def _utc_iso_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def _generate_session_id(payload: Mapping[str, object]) -> str:
    raw = json.dumps(
        {
            "idea": str(payload.get("idea") or ""),
            "stage": str(payload.get("stage") or ""),
            "mode": str(payload.get("mode") or ""),
            "ts": _utc_iso_now(),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _safe_session_id(session_id: str) -> str:
    cleaned = "".join(
        ch for ch in (session_id or "").strip() if ch.isalnum() or ch in {"_", "-"}
    )
    if not cleaned:
        raise ValueError("session_id must be non-empty alphanumeric / _ / -")
    return cleaned


def start_explore_session(
    *,
    payload: Mapping[str, object],
    workspace_root: str | Path,
    session_id: str | None = None,
    parent_session_id: str | None = None,
) -> ExploreSessionStartResult:
    """Persist the initial explore call. Response and lint come later.

    ``payload`` should be the dict returned by ``ExploreIdeaResult.to_payload``;
    we copy a small, structured subset for memory and keep the full prompt
    text for replay.
    """
    root = explore_sessions_root(workspace_root)
    root.mkdir(parents=True, exist_ok=True)
    sid = _safe_session_id(session_id) if session_id else _generate_session_id(payload)

    diagnostics_raw = payload.get("retrieval_diagnostics")
    diagnostics = (
        diagnostics_raw if isinstance(diagnostics_raw, dict) else {}
    )
    record: dict[str, object] = {
        "session_id": sid,
        "parent_session_id": (
            _safe_session_id(parent_session_id) if parent_session_id else None
        ),
        "created_at_utc": _utc_iso_now(),
        "idea": str(payload.get("idea") or ""),
        "mode": str(payload.get("mode") or ""),
        "stage": str(diagnostics.get("stage") or ""),
        "project_slug": str(payload.get("project_slug") or ""),
        "recommended_next_stage": diagnostics.get("recommended_next_stage"),
        "related_cards": (
            payload.get("related_cards")
            if isinstance(payload.get("related_cards"), list)
            else []
        ),
        "constraint_report": (
            payload.get("constraint_report")
            if isinstance(payload.get("constraint_report"), dict)
            else {}
        ),
        "gpt_prompt": str(payload.get("gpt_prompt") or ""),
        "retrieval_diagnostics": diagnostics,
        "response": None,
        "response_sections": None,
        "responded_at_utc": None,
        "lint_report": None,
    }
    path = root / f"{sid}.json"
    path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return ExploreSessionStartResult(
        session_id=sid,
        created_at_utc=str(record["created_at_utc"]),
        path=path,
    )


def record_explore_response(
    *,
    session_id: str,
    response_text: str,
    workspace_root: str | Path,
) -> LintReport:
    """Run stage-aware lint on the response and persist the report."""
    root = explore_sessions_root(workspace_root)
    sid = _safe_session_id(session_id)
    path = root / f"{sid}.json"
    if not path.exists():
        raise FileNotFoundError(f"explore session not found: {sid}")
    record = json.loads(path.read_text(encoding="utf-8"))
    stage = str(record.get("stage") or "mechanism_discovery")
    mode = str(record.get("mode") or "free")
    report = lint_explore_response(response_text, stage=stage, mode=mode)
    record["response"] = response_text
    record["response_sections"] = extract_stage_sections(response_text, stage=stage)
    record["responded_at_utc"] = _utc_iso_now()
    record["lint_report"] = report.to_dict()
    path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def read_explore_session(
    *,
    session_id: str,
    workspace_root: str | Path,
) -> dict[str, object]:
    sid = _safe_session_id(session_id)
    path = explore_sessions_root(workspace_root) / f"{sid}.json"
    if not path.exists():
        raise FileNotFoundError(f"explore session not found: {sid}")
    return json.loads(path.read_text(encoding="utf-8"))


def list_explore_sessions(
    *,
    workspace_root: str | Path,
    limit: int = 20,
) -> list[dict[str, object]]:
    root = explore_sessions_root(workspace_root)
    if not root.exists():
        return []
    paths = sorted(root.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    rows: list[dict[str, object]] = []
    for path in paths[: max(int(limit), 0)]:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        rows.append(
            {
                "session_id": record.get("session_id"),
                "parent_session_id": record.get("parent_session_id"),
                "created_at_utc": record.get("created_at_utc"),
                "idea": record.get("idea"),
                "stage": record.get("stage"),
                "mode": record.get("mode"),
                "project_slug": record.get("project_slug"),
                "recommended_next_stage": record.get("recommended_next_stage"),
                "has_response": record.get("response") is not None,
                "has_lint_errors": bool(
                    (record.get("lint_report") or {}).get("has_errors")
                ),
                "violation_codes": [
                    v.get("code")
                    for v in (record.get("lint_report") or {}).get("violations", [])
                    if isinstance(v, dict)
                ],
            }
        )
    return rows


_PREVIOUS_STAGE: dict[str, str] = {
    SIGNAL_MAPPING: MECHANISM_DISCOVERY,
    VALIDATION_KILL_TESTS: SIGNAL_MAPPING,
}

_UPSTREAM_SECTION_PRIORITY: dict[str, tuple[str, ...]] = {
    SIGNAL_MAPPING: ("机制候选", "信号思路"),
    VALIDATION_KILL_TESTS: (
        "Mechanism Mapping",
        "当前实现解释",
        "Confound 控制",
        "可测试信号版本",
    ),
}


def previous_workflow_stage(stage: str | None) -> str | None:
    """Return the immediate upstream workflow stage, if any."""
    return _PREVIOUS_STAGE.get(normalize_workflow_stage(stage))


def _iter_session_records(
    *,
    workspace_root: str | Path,
    limit: int,
) -> list[dict[str, object]]:
    root = explore_sessions_root(workspace_root)
    if not root.exists():
        return []
    paths = sorted(root.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    records: list[dict[str, object]] = []
    for path in paths[: max(int(limit), 0)]:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def _has_response_sections(record: Mapping[str, object]) -> bool:
    sections = record.get("response_sections")
    return isinstance(sections, dict) and any(
        str(value or "").strip() for value in sections.values()
    )


def find_upstream_session(
    *,
    workspace_root: str | Path,
    stage: str,
    parent_session_id: str | None = None,
    project_slug: str | None = None,
    limit: int = 20,
) -> dict[str, object] | None:
    """Find the best previous-stage session for workflow chaining.

    Explicit ``parent_session_id`` wins. Without it, we use the most
    recent responded session from the immediate upstream stage. Project
    slug is a soft fence: if both sides know it, they must match; older
    sessions without the field remain eligible.
    """
    current_stage = normalize_workflow_stage(stage)
    expected_stage = previous_workflow_stage(current_stage)
    if expected_stage is None:
        return None

    if parent_session_id:
        record = read_explore_session(
            session_id=parent_session_id,
            workspace_root=workspace_root,
        )
        if record.get("stage") != expected_stage or not _has_response_sections(record):
            return None
        return record

    wanted_project = (project_slug or "").strip()
    for record in _iter_session_records(workspace_root=workspace_root, limit=limit):
        if record.get("stage") != expected_stage:
            continue
        if not _has_response_sections(record):
            continue
        record_project = str(record.get("project_slug") or "").strip()
        if wanted_project and record_project and record_project != wanted_project:
            continue
        return record
    return None


def count_upstream_sections(record: Mapping[str, object], *, current_stage: str) -> int:
    sections = record.get("response_sections")
    if not isinstance(sections, dict):
        return 0
    priority = _UPSTREAM_SECTION_PRIORITY.get(
        normalize_workflow_stage(current_stage), tuple(str(key) for key in sections)
    )
    return sum(1 for label in priority if str(sections.get(label) or "").strip())


def _truncate_body(text: str, *, max_chars: int) -> str:
    stripped = text.strip()
    if len(stripped) <= max_chars:
        return stripped
    return stripped[: max(max_chars - 40, 0)].rstrip() + "\n...（上游产物已截断）"


def render_upstream_artifact_header(
    record: Mapping[str, object],
    *,
    current_stage: str,
    max_chars: int = 6000,
) -> str:
    """Render prior-stage structured response sections for the next prompt."""
    sections = record.get("response_sections")
    if not isinstance(sections, dict):
        return ""
    normalized_current = normalize_workflow_stage(current_stage)
    priority = _UPSTREAM_SECTION_PRIORITY.get(
        normalized_current, tuple(str(key) for key in sections)
    )
    section_bodies: list[tuple[str, str]] = []
    for label in priority:
        body = str(sections.get(label) or "").strip()
        if body:
            section_bodies.append((label, body))
    if not section_bodies:
        return ""

    sid = str(record.get("session_id") or "").strip()
    source_stage = str(record.get("stage") or "").strip()
    lines = [
        "## 上游产物",
        f"- 来源 session: `{sid}`" if sid else "- 来源 session: unknown",
        f"- 来源阶段: {source_stage or 'unknown'} → 当前阶段: {normalized_current}",
        "",
        "以下内容来自上一阶段的结构化响应，是本阶段的输入锚点；如果它与当前想法冲突，请显式指出冲突，而不是静默改写。",
        "",
    ]
    remaining = max(max_chars, 500)
    for label, body in section_bodies:
        if remaining <= 0:
            break
        rendered_body = _truncate_body(body, max_chars=remaining)
        lines.extend([f"### {label}", rendered_body, ""])
        remaining -= len(rendered_body)
    return "\n".join(lines).rstrip()


def list_recent_violations(
    *,
    workspace_root: str | Path,
    stage: str | None = None,
    mode: str | None = None,
    limit: int = 5,
    max_violations: int = 12,
) -> list[dict[str, object]]:
    """Return the most-recent violations across persisted sessions.

    Used by ``explore_idea`` to build the "known drift patterns" header.
    Sessions without a lint report are skipped. ``stage`` (and ``mode``)
    filter to violations from sessions of the same workflow stage so we
    don't, e.g., inject mechanism_discovery violations into a
    validation_kill_tests prompt.
    """
    root = explore_sessions_root(workspace_root)
    if not root.exists():
        return []
    paths = sorted(root.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    out: list[dict[str, object]] = []
    sessions_consumed = 0
    for path in paths:
        if sessions_consumed >= max(int(limit), 0):
            break
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        report = record.get("lint_report")
        if not isinstance(report, dict):
            continue
        if stage is not None and record.get("stage") != stage:
            continue
        if mode is not None and record.get("mode") != mode:
            continue
        sessions_consumed += 1
        for violation in report.get("violations", []) or []:
            if not isinstance(violation, dict):
                continue
            out.append(
                {
                    "session_id": record.get("session_id"),
                    "stage": record.get("stage"),
                    "mode": record.get("mode"),
                    "created_at_utc": record.get("created_at_utc"),
                    "code": violation.get("code"),
                    "severity": violation.get("severity"),
                    "section": violation.get("section"),
                    "detail": violation.get("detail"),
                }
            )
            if len(out) >= max(int(max_violations), 0):
                return out
    return out


def render_drift_header(violations: list[dict[str, object]]) -> str:
    """Render recent violations as a Markdown header for prompt injection.

    Returns an empty string when there's nothing to inject. The header is
    deliberately curt: the LLM should treat it as a self-check checklist,
    not a discussion item.
    """
    if not violations:
        return ""
    lines = [
        "## 已知漂移模式（自上次审计回灌）",
        "",
        "最近一轮探索响应触发了以下 lint 违规——本次输出**禁止**重复出现：",
        "",
    ]
    seen: set[str] = set()
    for item in violations:
        code = str(item.get("code") or "").strip()
        detail = str(item.get("detail") or "").strip()
        section = str(item.get("section") or "").strip()
        key = f"{code}|{section}|{detail}"
        if key in seen:
            continue
        seen.add(key)
        suffix = f"（section: {section}）" if section else ""
        lines.append(f"- `{code}`{suffix}: {detail}")
    lines.extend(
        [
            "",
            "在产出前先自检上述每一条是否仍可能复现；如不可避免，请显式说明原因。",
            "",
        ]
    )
    return "\n".join(lines)
