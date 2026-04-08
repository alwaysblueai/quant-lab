from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from alpha_lab.vault_export_gate import (
    apply_pending_vault_export_candidates,
    collect_pending_vault_export_candidates,
    infer_session_start_utc,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_manifest(
    base: Path,
    *,
    rel_root: str,
    case_name: str,
    run_timestamp_utc: str,
    export_status: str,
    export_mode: str = "skip",
    export_error: str | None = None,
) -> Path:
    case_dir = base / rel_root / case_name
    case_dir.mkdir(parents=True, exist_ok=True)
    experiment_card = case_dir / "experiment_card.md"
    experiment_card.write_text("# demo\n", encoding="utf-8")
    summary_path = case_dir / "summary.md"
    summary_path.write_text("# summary\n", encoding="utf-8")
    manifest_path = case_dir / "run_manifest.json"
    _write_json(
        manifest_path,
        {
            "case_name": case_name,
            "run_timestamp_utc": run_timestamp_utc,
            "outputs": {
                "experiment_card": str(experiment_card),
                "summary": str(summary_path),
            },
            "vault_export": {
                "enabled": export_mode != "skip",
                "mode": export_mode,
                "status": export_status,
                "target_paths": [],
                "error": export_error,
            },
        },
    )
    return manifest_path


def test_infer_session_start_utc_reads_first_timestamp(tmp_path: Path) -> None:
    transcript = tmp_path / "session.jsonl"
    transcript.write_text(
        "\n".join(
            [
                json.dumps({"timestamp": "2026-04-01T10:00:00.000Z"}),
                json.dumps({"timestamp": "2026-04-01T10:05:00.000Z"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    parsed = infer_session_start_utc(transcript)
    assert parsed == dt.datetime(2026, 4, 1, 10, 0, 0, tzinfo=dt.UTC)


def test_collect_pending_vault_export_candidates_filters_recent_active_roots(tmp_path: Path) -> None:
    recent = "2026-04-01T10:05:00+00:00"
    old = "2026-03-01T10:05:00+00:00"
    _write_manifest(
        tmp_path,
        rel_root="outputs/real_cases",
        case_name="case_recent_pending",
        run_timestamp_utc=recent,
        export_status="skipped",
    )
    _write_manifest(
        tmp_path,
        rel_root="outputs/real_cases",
        case_name="case_recent_success",
        run_timestamp_utc=recent,
        export_status="success",
        export_mode="versioned",
    )
    _write_manifest(
        tmp_path,
        rel_root="dist/examples/runs/default_research",
        case_name="case_examples_skipped",
        run_timestamp_utc=recent,
        export_status="skipped",
    )
    _write_manifest(
        tmp_path,
        rel_root="outputs/real_cases",
        case_name="case_old_pending",
        run_timestamp_utc=old,
        export_status="skipped",
    )

    candidates = collect_pending_vault_export_candidates(
        repo_root=tmp_path,
        since_utc=dt.datetime(2026, 4, 1, 10, 0, 0, tzinfo=dt.UTC),
    )
    assert [candidate.case_name for candidate in candidates] == ["case_recent_pending"]


def test_apply_pending_vault_export_candidates_updates_manifest_and_writes_vault(tmp_path: Path) -> None:
    manifest_path = _write_manifest(
        tmp_path,
        rel_root="outputs/real_cases",
        case_name="case_pending_apply",
        run_timestamp_utc="2026-04-01T10:05:00+00:00",
        export_status="skipped",
    )
    candidates = collect_pending_vault_export_candidates(repo_root=tmp_path)
    assert [candidate.case_name for candidate in candidates] == ["case_pending_apply"]

    vault_root = tmp_path / "vault"
    vault_root.mkdir(parents=True, exist_ok=True)
    results = apply_pending_vault_export_candidates(
        candidates,
        vault_root=vault_root,
        mode="versioned",
    )

    assert len(results) == 1
    candidate, export_result = results[0]
    assert candidate.case_name == "case_pending_apply"
    assert export_result.success is True
    exported_latest = vault_root / "50_experiments" / "case_pending_apply" / "latest.md"
    assert exported_latest.exists()

    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest_payload["vault_export"]["status"] == "success"
    assert manifest_payload["vault_export"]["mode"] == "versioned"
    assert manifest_payload["vault_export"]["target_paths"]
