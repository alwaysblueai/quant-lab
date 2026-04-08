from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from alpha_lab.exceptions import AlphaLabConfigError
from alpha_lab.vault_export import ExportResult, export_to_vault, resolve_vault_root

DEFAULT_ACTIVE_OUTPUT_ROOTS = (
    "outputs/real_cases",
    "dist/bridge_runs",
    "dist/web_ui_runs",
)


@dataclass(frozen=True)
class PendingVaultExportCandidate:
    manifest_path: Path
    case_name: str
    run_timestamp_utc: str | None
    experiment_card_path: Path
    summary_path: Path | None
    current_status: str
    current_mode: str
    current_error: str | None


def _repo_root_default() -> Path:
    return Path(__file__).resolve().parents[2]


def _parse_iso_timestamp(raw: str | None) -> dt.datetime | None:
    if raw is None:
        return None
    text = raw.strip()
    if not text:
        return None
    try:
        parsed = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=dt.UTC)
    return parsed.astimezone(dt.UTC)


def infer_session_start_utc(transcript_path: str | Path) -> dt.datetime | None:
    path = Path(transcript_path).expanduser().resolve()
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                payload = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            timestamp = payload.get("timestamp")
            if isinstance(timestamp, str):
                parsed = _parse_iso_timestamp(timestamp)
                if parsed is not None:
                    return parsed
    return None


def _iter_run_manifests(
    repo_root: Path,
    *,
    active_output_roots: tuple[str, ...] = DEFAULT_ACTIVE_OUTPUT_ROOTS,
) -> list[Path]:
    manifests: list[Path] = []
    for rel_root in active_output_roots:
        root = (repo_root / rel_root).resolve()
        if not root.exists():
            continue
        manifests.extend(sorted(root.rglob("run_manifest.json")))
    return manifests


def _is_recent_enough(
    *,
    manifest_path: Path,
    manifest_payload: dict[str, Any],
    since_utc: dt.datetime | None,
) -> bool:
    if since_utc is None:
        return True
    run_timestamp = _parse_iso_timestamp(
        str(manifest_payload.get("run_timestamp_utc") or "").strip() or None
    )
    if run_timestamp is not None:
        return run_timestamp >= since_utc
    modified_utc = dt.datetime.fromtimestamp(manifest_path.stat().st_mtime, tz=dt.UTC)
    return modified_utc >= since_utc


def _load_manifest_candidate(
    manifest_path: Path,
    *,
    since_utc: dt.datetime | None,
) -> PendingVaultExportCandidate | None:
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if not _is_recent_enough(
        manifest_path=manifest_path,
        manifest_payload=payload,
        since_utc=since_utc,
    ):
        return None

    outputs = payload.get("outputs")
    if not isinstance(outputs, dict):
        return None
    raw_experiment_card = outputs.get("experiment_card")
    if not isinstance(raw_experiment_card, str) or not raw_experiment_card.strip():
        return None
    experiment_card_path = Path(raw_experiment_card).expanduser().resolve()
    if not experiment_card_path.exists() or not experiment_card_path.is_file():
        return None

    raw_summary = outputs.get("summary")
    summary_path = None
    if isinstance(raw_summary, str) and raw_summary.strip():
        summary_candidate = Path(raw_summary).expanduser().resolve()
        if summary_candidate.exists() and summary_candidate.is_file():
            summary_path = summary_candidate

    vault_export = payload.get("vault_export")
    if not isinstance(vault_export, dict):
        return None
    status = str(vault_export.get("status") or "").strip().lower()
    mode = str(vault_export.get("mode") or "").strip().lower()
    target_paths = vault_export.get("target_paths")
    if status == "success" and isinstance(target_paths, list) and target_paths:
        return None
    if status not in {"skipped", "failed"}:
        return None

    case_name = str(payload.get("case_name") or "").strip()
    if not case_name:
        case_name = manifest_path.parent.name

    return PendingVaultExportCandidate(
        manifest_path=manifest_path.resolve(),
        case_name=case_name,
        run_timestamp_utc=str(payload.get("run_timestamp_utc") or "").strip() or None,
        experiment_card_path=experiment_card_path,
        summary_path=summary_path,
        current_status=status or "unknown",
        current_mode=mode or "skip",
        current_error=(
            str(vault_export.get("error")).strip()
            if vault_export.get("error") not in (None, "")
            else None
        ),
    )


def collect_pending_vault_export_candidates(
    repo_root: str | Path | None = None,
    *,
    since_utc: dt.datetime | None = None,
    active_output_roots: tuple[str, ...] = DEFAULT_ACTIVE_OUTPUT_ROOTS,
) -> list[PendingVaultExportCandidate]:
    resolved_root = _repo_root_default() if repo_root is None else Path(repo_root).resolve()
    candidates: list[PendingVaultExportCandidate] = []
    for manifest_path in _iter_run_manifests(
        resolved_root,
        active_output_roots=active_output_roots,
    ):
        candidate = _load_manifest_candidate(manifest_path, since_utc=since_utc)
        if candidate is not None:
            candidates.append(candidate)
    candidates.sort(
        key=lambda item: (
            item.run_timestamp_utc or "",
            str(item.manifest_path),
        ),
        reverse=True,
    )
    return candidates


def _sync_exported_manifest_copies(
    manifest_path: Path,
    target_paths: tuple[str, ...],
) -> None:
    for raw_target in target_paths:
        target = Path(raw_target).expanduser().resolve()
        if not target.name.endswith("run_manifest.json"):
            continue
        try:
            shutil.copy2(manifest_path, target)
        except OSError:
            continue


def apply_pending_vault_export_candidates(
    candidates: list[PendingVaultExportCandidate],
    *,
    vault_root: str | Path | None,
    mode: str | None = None,
) -> list[tuple[PendingVaultExportCandidate, ExportResult]]:
    resolved_vault = resolve_vault_root(vault_root)
    if resolved_vault is None:
        raise AlphaLabConfigError("vault root is unresolved; pass --vault-root or set OBSIDIAN_VAULT_PATH")

    results: list[tuple[PendingVaultExportCandidate, ExportResult]] = []
    for candidate in candidates:
        applied_mode = mode or (
            candidate.current_mode if candidate.current_mode and candidate.current_mode != "skip" else "versioned"
        )
        export_result = export_to_vault(
            {
                "experiment_card_path": candidate.experiment_card_path,
                "summary_path": candidate.summary_path,
                "manifest_path": candidate.manifest_path,
            },
            case_name=candidate.case_name,
            vault_root=resolved_vault,
            mode=applied_mode,
        )
        payload = json.loads(candidate.manifest_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload["vault_export"] = export_result.to_manifest_dict(
                enabled=applied_mode != "skip"
            )
            candidate.manifest_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        if export_result.success and export_result.target_paths:
            _sync_exported_manifest_copies(candidate.manifest_path, export_result.target_paths)
        results.append((candidate, export_result))
    return results


def _render_detect_output(candidates: list[PendingVaultExportCandidate]) -> str:
    if not candidates:
        return "No pending vault exports detected.\n"
    lines = [
        f"Pending vault exports detected: {len(candidates)}",
        "",
    ]
    for candidate in candidates:
        lines.extend(
            [
                f"- case: {candidate.case_name}",
                f"  status: {candidate.current_status} (mode={candidate.current_mode})",
                f"  manifest: {candidate.manifest_path}",
                f"  card: {candidate.experiment_card_path}",
            ]
        )
        if candidate.current_error:
            lines.append(f"  error: {candidate.current_error}")
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m alpha_lab.vault_export_gate",
        description="Detect or export pending alpha-lab vault artifacts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    detect_parser = subparsers.add_parser("detect")
    detect_parser.add_argument(
        "--repo-root",
        default=str(_repo_root_default()),
    )
    detect_parser.add_argument(
        "--transcript-path",
        default=None,
        help="Optional Claude transcript path used to bound detection to the current session.",
    )
    detect_parser.add_argument(
        "--since-hours",
        type=float,
        default=None,
        help="Optional rolling lookback when transcript_path is not provided.",
    )

    apply_parser = subparsers.add_parser("apply")
    apply_parser.add_argument(
        "--repo-root",
        default=str(_repo_root_default()),
    )
    apply_parser.add_argument(
        "--manifest-path",
        action="append",
        default=[],
        help="Explicit run_manifest.json path to export; repeatable.",
    )
    apply_parser.add_argument(
        "--transcript-path",
        default=None,
        help="Optional Claude transcript path used to select pending manifests from the current session.",
    )
    apply_parser.add_argument(
        "--since-hours",
        type=float,
        default=None,
        help="Optional rolling lookback when transcript_path is not provided.",
    )
    apply_parser.add_argument(
        "--vault-root",
        default=None,
        help="Quant-knowledge vault root. Defaults to OBSIDIAN_VAULT_PATH.",
    )
    apply_parser.add_argument(
        "--mode",
        default=None,
        choices=["skip", "overwrite", "versioned"],
        help="Override vault export mode for all selected manifests.",
    )

    return parser


def _resolve_since_utc(args: argparse.Namespace) -> dt.datetime | None:
    transcript_path = getattr(args, "transcript_path", None)
    if isinstance(transcript_path, str) and transcript_path.strip():
        return infer_session_start_utc(transcript_path)
    since_hours = getattr(args, "since_hours", None)
    if since_hours is None:
        return None
    return dt.datetime.now(dt.UTC) - dt.timedelta(hours=float(since_hours))


def _load_explicit_candidates(manifest_paths: list[str]) -> list[PendingVaultExportCandidate]:
    candidates: list[PendingVaultExportCandidate] = []
    for raw_path in manifest_paths:
        manifest_path = Path(raw_path).expanduser().resolve()
        candidate = _load_manifest_candidate(manifest_path, since_utc=None)
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "detect":
        candidates = collect_pending_vault_export_candidates(
            repo_root=args.repo_root,
            since_utc=_resolve_since_utc(args),
        )
        print(_render_detect_output(candidates), end="")
        return 0

    if args.command == "apply":
        if args.manifest_path:
            candidates = _load_explicit_candidates(args.manifest_path)
        else:
            candidates = collect_pending_vault_export_candidates(
                repo_root=args.repo_root,
                since_utc=_resolve_since_utc(args),
            )
        if not candidates:
            print("No pending vault exports selected.")
            return 0
        results = apply_pending_vault_export_candidates(
            candidates,
            vault_root=args.vault_root,
            mode=args.mode,
        )
        failed = 0
        for candidate, export_result in results:
            print(
                f"{candidate.case_name}: status={export_result.status} "
                f"mode={export_result.mode_used} targets={len(export_result.target_paths)}"
            )
            if export_result.error:
                print(f"  error: {export_result.error}")
            if not export_result.success:
                failed += 1
        return 1 if failed else 0

    parser.error(f"unsupported command: {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
