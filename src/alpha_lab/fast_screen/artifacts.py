"""Dual-tier artifact layout on disk.

Layout::

    <artifact_root>/<factor>/<run_id>/
        tier1/
            result.json            # FastScreenResult
            charts/<key>.json      # individual chart payloads (dup of result.json)
        tier2/
            index.json             # {module: status}
            <module>/
                result.json
                status.json        # Tier2ModuleStatus
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import (
    ChartSeries,
    FastScreenResult,
    MetricCard,
    MetricStatus,
    Tier2ModuleStatus,
    Verdict,
)


@dataclass(frozen=True)
class FastScreenArtifactPaths:
    """Concrete on-disk paths for a single fast-screen run."""

    run_dir: Path
    tier1_dir: Path
    tier1_result: Path
    tier2_dir: Path
    tier2_index: Path


def _run_dir(artifact_root: Path, factor_name: str, run_id: str) -> Path:
    return artifact_root / _sanitize(factor_name) / _sanitize(run_id)


def tier1_dir(artifact_root: Path, factor_name: str, run_id: str) -> Path:
    return _run_dir(artifact_root, factor_name, run_id) / "tier1"


def tier2_module_dir(artifact_root: Path, factor_name: str, run_id: str, module: str) -> Path:
    return _run_dir(artifact_root, factor_name, run_id) / "tier2" / _sanitize(module)


def save_tier1_result(
    artifact_root: str | Path,
    result: FastScreenResult,
) -> FastScreenArtifactPaths:
    """Write a Tier-1 bundle to disk. Creates parent directories as needed."""
    root = Path(artifact_root).resolve()
    run = _run_dir(root, result.factor_name, result.run_id)
    t1 = run / "tier1"
    t2 = run / "tier2"
    charts = t1 / "charts"
    charts.mkdir(parents=True, exist_ok=True)
    t2.mkdir(parents=True, exist_ok=True)

    _write_json(t1 / "result.json", result.to_dict())
    for chart in result.charts:
        _write_json(charts / f"{_sanitize(chart.key)}.json", chart.to_dict())

    index_path = t2 / "index.json"
    if not index_path.exists():
        _write_json(index_path, {"modules": {}})

    return FastScreenArtifactPaths(
        run_dir=run,
        tier1_dir=t1,
        tier1_result=t1 / "result.json",
        tier2_dir=t2,
        tier2_index=index_path,
    )


def load_tier1_result(artifact_root: str | Path, factor_name: str, run_id: str) -> FastScreenResult:
    path = tier1_dir(Path(artifact_root).resolve(), factor_name, run_id) / "result.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return _result_from_dict(payload)


def save_tier2_module(
    artifact_root: str | Path,
    factor_name: str,
    run_id: str,
    module: str,
    *,
    result_payload: dict[str, Any],
    status: Tier2ModuleStatus,
) -> Path:
    """Persist one Tier-2 module's output and status.

    Atomic for a single module: writing this does not invalidate sibling
    modules. The module-level ``status.json`` is the source of truth for the
    UI; ``index.json`` is regenerated on every write for convenience.
    """
    root = Path(artifact_root).resolve()
    mod_dir = tier2_module_dir(root, factor_name, run_id, module)
    mod_dir.mkdir(parents=True, exist_ok=True)
    _write_json(mod_dir / "result.json", result_payload)
    _write_json(mod_dir / "status.json", status.to_dict())

    index = _rebuild_index(root, factor_name, run_id)
    t2_dir = tier2_module_dir(root, factor_name, run_id, module).parent
    _write_json(t2_dir / "index.json", {"modules": index})
    return mod_dir


def load_tier2_index(
    artifact_root: str | Path, factor_name: str, run_id: str
) -> dict[str, Tier2ModuleStatus]:
    root = Path(artifact_root).resolve()
    t2_dir = _run_dir(root, factor_name, run_id) / "tier2"
    if not t2_dir.exists():
        return {}
    index = _rebuild_index(root, factor_name, run_id)
    return {k: _module_status_from_dict(v) for k, v in index.items()}


def _rebuild_index(root: Path, factor_name: str, run_id: str) -> dict[str, dict[str, Any]]:
    t2_dir = _run_dir(root, factor_name, run_id) / "tier2"
    out: dict[str, dict[str, Any]] = {}
    if not t2_dir.exists():
        return out
    for child in sorted(t2_dir.iterdir()):
        if not child.is_dir():
            continue
        status_file = child / "status.json"
        if not status_file.exists():
            continue
        try:
            out[child.name] = json.loads(status_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
    return out


def _result_from_dict(payload: dict[str, Any]) -> FastScreenResult:
    metrics = [
        MetricCard(
            key=m["key"],
            label=m["label"],
            value=m.get("value"),
            status=MetricStatus(m["status"]),
            unit=m.get("unit", ""),
            secondary=dict(m.get("secondary") or {}),
            note=m.get("note", ""),
        )
        for m in payload.get("metrics", [])
    ]
    charts = [
        ChartSeries(
            key=c["key"],
            label=c["label"],
            kind=c["kind"],
            x=list(c.get("x") or []),
            y=list(c.get("y") or []),
            status=MetricStatus(c["status"]),
            extras=dict(c.get("extras") or {}),
            note=c.get("note", ""),
        )
        for c in payload.get("charts", [])
    ]
    v = payload.get("verdict") or {}
    verdict = Verdict(
        status=v.get("status", "fail"),
        triggered_rules=list(v.get("triggered_rules") or []),
        next_step=v.get("next_step", ""),
    )
    return FastScreenResult(
        factor_name=payload["factor_name"],
        run_id=payload["run_id"],
        universe=payload.get("universe", ""),
        frequency=payload.get("frequency", ""),
        window=dict(payload.get("window") or {}),
        metrics=metrics,
        charts=charts,
        verdict=verdict,
        inputs_hash=payload.get("inputs_hash", ""),
        generated_at=payload.get("generated_at", ""),
    )


def _module_status_from_dict(payload: dict[str, Any]) -> Tier2ModuleStatus:
    return Tier2ModuleStatus(
        module=payload["module"],
        status=MetricStatus(payload["status"]),
        computed_at=payload.get("computed_at", ""),
        duration_sec=float(payload.get("duration_sec", 0.0)),
        inputs_hash=payload.get("inputs_hash", ""),
        stale=bool(payload.get("stale", False)),
        message=payload.get("message", ""),
    )


def _sanitize(name: str) -> str:
    keep = []
    for ch in name:
        if ch.isalnum() or ch in ("-", "_", "."):
            keep.append(ch)
        else:
            keep.append("_")
    out = "".join(keep).strip("._") or "unnamed"
    return out[:120]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8"
    )
    tmp.replace(path)


def _json_default(obj: Any) -> Any:
    # Pandas timestamps and numpy scalars, just in case
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:  # noqa: BLE001
            return str(obj)
    return str(obj)
