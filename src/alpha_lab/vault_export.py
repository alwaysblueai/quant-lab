from __future__ import annotations

import datetime
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from alpha_lab.exceptions import AlphaLabConfigError

VaultExportMode = Literal["skip", "overwrite", "versioned"]
VaultExportStatus = Literal["success", "failed", "skipped"]

_VALID_MODES: frozenset[str] = frozenset({"skip", "overwrite", "versioned"})


@dataclass(frozen=True)
class ExportResult:
    """Outcome of one quant-lab -> quant-knowledge export attempt."""

    success: bool
    target_paths: tuple[str, ...]
    mode_used: VaultExportMode
    status: VaultExportStatus
    error: str | None = None

    def to_manifest_dict(self, *, enabled: bool) -> dict[str, object]:
        return {
            "enabled": enabled,
            "mode": self.mode_used,
            "target_paths": list(self.target_paths),
            "status": self.status,
            "error": self.error,
        }


def resolve_vault_root(vault_root: str | Path | None) -> Path | None:
    """Resolve vault root using CLI value first, then env, then None."""

    if vault_root is not None:
        text = str(vault_root).strip()
        if text:
            return Path(text).resolve()

    env_value = (os.environ.get("OBSIDIAN_VAULT_PATH") or "").strip()
    if env_value:
        return Path(env_value).resolve()

    return None


def export_to_vault(
    source_paths: dict[str, str | Path | None],
    case_name: str,
    vault_root: str | Path | None,
    mode: str = "versioned",
) -> ExportResult:
    """Export experiment artifacts into quant-knowledge/50_experiments/."""

    mode_l = mode.strip().lower()
    if mode_l not in _VALID_MODES:
        return ExportResult(
            success=False,
            target_paths=(),
            mode_used="skip",
            status="failed",
            error=f"invalid vault export mode: {mode!r}",
        )

    resolved_root = resolve_vault_root(vault_root)
    if mode_l == "skip":
        return ExportResult(
            success=True,
            target_paths=(),
            mode_used="skip",
            status="skipped",
            error=None,
        )
    if resolved_root is None:
        return ExportResult(
            success=False,
            target_paths=(),
            mode_used=mode_l,  # type: ignore[arg-type]
            status="failed",
            error="vault_root was not provided and OBSIDIAN_VAULT_PATH is not set",
        )

    if not resolved_root.exists():
        return ExportResult(
            success=False,
            target_paths=(),
            mode_used=mode_l,  # type: ignore[arg-type]
            status="failed",
            error=f"vault root does not exist: {resolved_root}",
        )
    if not resolved_root.is_dir():
        return ExportResult(
            success=False,
            target_paths=(),
            mode_used=mode_l,  # type: ignore[arg-type]
            status="failed",
            error=f"vault root is not a directory: {resolved_root}",
        )

    experiment_card_src = _resolve_source_path(
        source_paths.get("experiment_card_path"),
        name="experiment_card_path",
        required=True,
    )
    if experiment_card_src is None:
        return ExportResult(
            success=False,
            target_paths=(),
            mode_used=mode_l,  # type: ignore[arg-type]
            status="failed",
            error="experiment_card_path is required",
        )

    summary_src = _resolve_source_path(
        source_paths.get("summary_path"),
        name="summary_path",
        required=False,
    )
    manifest_src = _resolve_source_path(
        source_paths.get("manifest_path"),
        name="manifest_path",
        required=False,
    )
    contract_sidecars = _resolve_backend_contract_sidecars(manifest_src)

    if not experiment_card_src.exists() or not experiment_card_src.is_file():
        return ExportResult(
            success=False,
            target_paths=(),
            mode_used=mode_l,  # type: ignore[arg-type]
            status="failed",
            error=f"experiment_card_path does not exist: {experiment_card_src}",
        )

    timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H-%M-%S")
    safe_case = _safe_case_name(case_name)
    case_dir = (resolved_root / "50_experiments" / safe_case).resolve()

    try:
        case_dir.mkdir(parents=True, exist_ok=True)
        targets: list[Path] = []

        if mode_l == "overwrite":
            targets.extend(
                _copy_set(
                    experiment_card_src,
                    summary_src,
                    manifest_src,
                    case_dir,
                    experiment_name="latest.md",
                    summary_name="latest_summary.md",
                    manifest_name="latest_run_manifest.json",
                    sidecars=_sidecar_targets(contract_sidecars, prefix=None),
                )
            )
        else:
            targets.extend(
                _copy_set(
                    experiment_card_src,
                    summary_src,
                    manifest_src,
                    case_dir,
                    experiment_name=f"{timestamp}__experiment_card.md",
                    summary_name=f"{timestamp}__summary.md",
                    manifest_name=f"{timestamp}__run_manifest.json",
                    sidecars=_sidecar_targets(contract_sidecars, prefix=f"{timestamp}__"),
                )
            )
            targets.extend(
                _copy_set(
                    experiment_card_src,
                    summary_src,
                    manifest_src,
                    case_dir,
                    experiment_name="latest.md",
                    summary_name="latest_summary.md",
                    manifest_name="latest_run_manifest.json",
                    sidecars=_sidecar_targets(contract_sidecars, prefix=None),
                )
            )

        return ExportResult(
            success=True,
            target_paths=tuple(str(path) for path in targets),
            mode_used=mode_l,  # type: ignore[arg-type]
            status="success",
            error=None,
        )
    except Exception as exc:
        return ExportResult(
            success=False,
            target_paths=(),
            mode_used=mode_l,  # type: ignore[arg-type]
            status="failed",
            error=str(exc),
        )


def _resolve_source_path(
    value: str | Path | None,
    *,
    name: str,
    required: bool,
) -> Path | None:
    if value is None:
        if required:
            return None
        return None

    text = str(value).strip()
    if not text:
        if required:
            return None
        return None

    return Path(text).resolve()


def _copy_set(
    experiment_card_src: Path,
    summary_src: Path | None,
    manifest_src: Path | None,
    case_dir: Path,
    *,
    experiment_name: str,
    summary_name: str,
    manifest_name: str,
    sidecars: tuple[tuple[Path, str], ...] = (),
) -> tuple[Path, ...]:
    targets: list[Path] = []

    exp_target = case_dir / experiment_name
    shutil.copy2(experiment_card_src, exp_target)
    targets.append(exp_target)

    if summary_src is not None and summary_src.exists() and summary_src.is_file():
        summary_target = case_dir / summary_name
        shutil.copy2(summary_src, summary_target)
        targets.append(summary_target)

    if manifest_src is not None and manifest_src.exists() and manifest_src.is_file():
        manifest_target = case_dir / manifest_name
        shutil.copy2(manifest_src, manifest_target)
        targets.append(manifest_target)

    for src, dest_name in sidecars:
        if src.exists() and src.is_file():
            dest = case_dir / dest_name
            shutil.copy2(src, dest)
            targets.append(dest)

    return tuple(targets)


def _resolve_backend_contract_sidecars(
    manifest_src: Path | None,
) -> tuple[tuple[Path, str], ...]:
    """Detect backend-run contract sidecars next to ``run_manifest.json``.

    Returns ``(src_path, base_name)`` pairs for any sidecar present beside the
    manifest. Non-draft runs (no sidecars on disk) yield an empty tuple, which
    leaves vault export behavior unchanged.
    """

    if manifest_src is None:
        return ()
    if not manifest_src.exists() or not manifest_src.is_file():
        return ()
    run_dir = manifest_src.parent
    found: list[tuple[Path, str]] = []
    for base_name in ("backend_run_receipt.json", "comparison_summary.json"):
        candidate = run_dir / base_name
        if candidate.is_file():
            found.append((candidate, base_name))
    return tuple(found)


def _sidecar_targets(
    sidecars: tuple[tuple[Path, str], ...],
    *,
    prefix: str | None,
) -> tuple[tuple[Path, str], ...]:
    if prefix is None:
        return tuple((src, f"latest_{base_name}") for src, base_name in sidecars)
    return tuple((src, f"{prefix}{base_name}") for src, base_name in sidecars)


def _safe_case_name(case_name: str) -> str:
    stripped = case_name.strip()
    if not stripped:
        raise AlphaLabConfigError("case_name must be non-empty")

    normalized = re.sub(r"\s+", "_", stripped)
    normalized = re.sub(r"[^A-Za-z0-9_.-]", "-", normalized)
    normalized = normalized.strip(".-_")
    if not normalized:
        raise AlphaLabConfigError(f"case_name is not valid for vault path: {case_name!r}")
    return normalized
