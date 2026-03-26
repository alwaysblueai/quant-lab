from __future__ import annotations

import datetime
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from alpha_lab.config import RAW_DATA_DIR
from alpha_lab.data_sources.tushare_schemas import RAW_SNAPSHOT_SORT_KEYS, RAW_TUSHARE_DIRNAME


@dataclass(frozen=True)
class TushareRawSnapshot:
    """One endpoint snapshot on disk."""

    endpoint: str
    csv_path: Path
    metadata_path: Path


@dataclass(frozen=True)
class RawSnapshotManifest:
    """Disk layout metadata for a raw Tushare extraction batch."""

    snapshot_dir: Path
    manifest_path: Path
    endpoints: tuple[str, ...]
    unavailable_endpoints: tuple[str, ...]
    degraded_endpoints: tuple[str, ...] = ()
    endpoint_extractions: Mapping[str, object] = field(default_factory=dict)
    endpoint_status_table: tuple[Mapping[str, object], ...] = ()


def write_raw_snapshot(
    df: pd.DataFrame,
    *,
    endpoint: str,
    snapshot_dir: str | Path,
    params: dict[str, object],
    extraction_utc: str | None = None,
    sort_keys: tuple[str, ...] | None = None,
    metadata_extras: Mapping[str, object] | None = None,
) -> TushareRawSnapshot:
    """Write one deterministic raw endpoint snapshot."""

    output_dir = Path(snapshot_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{endpoint}.csv"
    metadata_path = output_dir / f"{endpoint}.meta.json"

    stable_df = _stable_sort(df, endpoint=endpoint, sort_keys=sort_keys)
    stable_df.to_csv(csv_path, index=False, lineterminator="\n")

    metadata = {
        "endpoint": endpoint,
        "source": "tushare_pro",
        "extraction_utc": extraction_utc or _utc_now(),
        "row_count": int(len(stable_df)),
        "columns": stable_df.columns.tolist(),
        "params": params,
    }
    if metadata_extras is not None:
        metadata.update(dict(metadata_extras))
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return TushareRawSnapshot(
        endpoint=endpoint,
        csv_path=csv_path,
        metadata_path=metadata_path,
    )


def write_raw_snapshot_manifest(
    *,
    snapshot_name: str,
    snapshot_dir: str | Path,
    extraction_utc: str | None,
    endpoints: list[str],
    unavailable_endpoints: list[str],
    notes: list[str] | None = None,
    endpoint_extractions: Mapping[str, object] | None = None,
    endpoint_status_table: list[Mapping[str, object]] | None = None,
    degraded_endpoints: list[str] | None = None,
) -> RawSnapshotManifest:
    """Write a deterministic manifest for one extraction batch."""

    output_dir = Path(snapshot_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    status_table = [dict(item) for item in (endpoint_status_table or [])]
    payload = {
        "snapshot_name": snapshot_name,
        "source": "tushare_pro",
        "extraction_utc": extraction_utc or _utc_now(),
        "endpoints": sorted(endpoints),
        "unavailable_endpoints": sorted(set(unavailable_endpoints)),
        "degraded_endpoints": sorted(set(degraded_endpoints or [])),
        "notes": list(notes or []),
        "endpoint_extractions": dict(endpoint_extractions or {}),
        "endpoint_status_table": status_table,
    }
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return RawSnapshotManifest(
        snapshot_dir=output_dir,
        manifest_path=manifest_path,
        endpoints=tuple(sorted(endpoints)),
        unavailable_endpoints=tuple(sorted(set(unavailable_endpoints))),
        degraded_endpoints=tuple(sorted(set(degraded_endpoints or []))),
        endpoint_extractions=dict(endpoint_extractions or {}),
        endpoint_status_table=tuple(status_table),
    )


def load_raw_snapshot(
    snapshot_dir: str | Path,
    *,
    endpoint: str,
) -> pd.DataFrame:
    path = Path(snapshot_dir).resolve() / f"{endpoint}.csv"
    if not path.exists():
        raise FileNotFoundError(f"raw snapshot file does not exist: {path}")
    return pd.read_csv(path)


def default_raw_snapshot_dir(snapshot_name: str) -> Path:
    return RAW_DATA_DIR / RAW_TUSHARE_DIRNAME / snapshot_name


def _stable_sort(
    df: pd.DataFrame,
    *,
    endpoint: str,
    sort_keys: tuple[str, ...] | None,
) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    resolved_sort_keys = sort_keys or RAW_SNAPSHOT_SORT_KEYS.get(endpoint)
    if resolved_sort_keys is None:
        return df.copy().reset_index(drop=True)
    if not all(col in df.columns for col in resolved_sort_keys):
        return df.copy().reset_index(drop=True)
    return df.sort_values(list(resolved_sort_keys), kind="mergesort").reset_index(drop=True)


def _utc_now() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")
