from __future__ import annotations

import gc
import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

DATASET_CACHE_SCHEMA_VERSION = "1.0.0"
DATASET_CACHE_VERSION = "model_factor_preparation_v2"
DATA_LOAD_CACHE_LAYER = "data_load"
PREPARED_INPUTS_CACHE_LAYER = "prepared_inputs"
DATA_LOAD_CACHE_ROW_GROUP_ROWS = 128_000
PREPARED_INPUTS_NUMPY_CHUNK_ROWS = 128_000
PREPARED_INPUTS_ROW_ORDER = "date_asset"
PREPARED_INPUTS_FEATURE_DTYPE = "float32"


@dataclass(frozen=True)
class ResolvedFeatureAvailability:
    mode: str
    requested_column: str | None
    source_column: str | None
    known_at_col: str
    safety_lag_days: int | None
    shifted_rows: int
    dropped_rows: int


@dataclass(frozen=True)
class DataLoadCacheEntry:
    prices: pd.DataFrame
    features: pd.DataFrame
    resolved_feature_availability: ResolvedFeatureAvailability
    metadata: dict[str, object]
    features_loaded: bool = True


@dataclass(frozen=True)
class PreparedInputsNumpyCacheEntry:
    feature_values: np.ndarray
    feature_columns: tuple[str, ...]
    index_df: pd.DataFrame
    labeled_feature_values: np.ndarray
    labeled_index_df: pd.DataFrame
    metadata: dict[str, object]


@dataclass(frozen=True)
class PreparedInputsCacheEntry:
    features: pd.DataFrame
    labels: pd.DataFrame
    forward_label_df: pd.DataFrame
    data_health: dict[str, object]
    winsor_clip_count: int
    metadata: dict[str, object]
    numpy_entry: PreparedInputsNumpyCacheEntry | None = None


class ModelFactorDatasetCache:
    """Local Parquet cache for model-factor reusable dataset preparation layers."""

    def __init__(self, root_dir: str | Path) -> None:
        self._root_dir = Path(root_dir).resolve()

    @property
    def root_dir(self) -> Path:
        return self._root_dir

    @property
    def data_load_root_dir(self) -> Path:
        return self._root_dir / DATA_LOAD_CACHE_LAYER

    @property
    def prepared_inputs_root_dir(self) -> Path:
        return self._root_dir / "model_build"

    @staticmethod
    def file_signature(path: str | Path) -> dict[str, object]:
        return file_signature(path)

    @staticmethod
    def jsonable_for_cache(value: object) -> object:
        return jsonable_for_cache(value)

    def build_key(self, payload: Mapping[str, object]) -> str:
        return build_dataset_cache_key(payload)

    def data_load_path(self, cache_key: str) -> Path:
        return self.data_load_root_dir / str(cache_key)

    def prepared_inputs_exists(self, cache_key: str) -> bool:
        return prepared_inputs_cache_exists(
            cache_dir=self.prepared_inputs_root_dir,
            cache_key=cache_key,
        )

    def prepared_inputs_metadata(self, cache_key: str) -> dict[str, object] | None:
        return prepared_inputs_cache_metadata(
            cache_dir=self.prepared_inputs_root_dir,
            cache_key=cache_key,
        )

    def load_data_load(
        self,
        cache_key: str,
        *,
        include_features: bool = True,
    ) -> DataLoadCacheEntry | None:
        cache_path = self.data_load_path(cache_key)
        metadata_path = cache_path / "metadata.json"
        prices_path = cache_path / "prices.parquet"
        features_path = cache_path / "features.parquet"
        if not (metadata_path.exists() and prices_path.exists() and features_path.exists()):
            return None
        try:
            metadata = _read_metadata(metadata_path)
            if metadata.get("cache_version") != DATASET_CACHE_VERSION:
                return None
            if metadata.get("cache_key") != cache_key:
                return None
            cache_layer = metadata.get("cache_layer")
            if cache_layer is not None and cache_layer != DATA_LOAD_CACHE_LAYER:
                return None
            resolved_payload = metadata.get("resolved_feature_availability")
            if not isinstance(resolved_payload, dict):
                return None
            return DataLoadCacheEntry(
                prices=pd.read_parquet(prices_path),
                features=(
                    pd.read_parquet(features_path)
                    if include_features
                    else pd.DataFrame()
                ),
                resolved_feature_availability=_resolved_feature_availability_from_mapping(
                    resolved_payload
                ),
                metadata=metadata,
                features_loaded=bool(include_features),
            )
        except Exception:
            return None

    def write_data_load(
        self,
        *,
        cache_key: str,
        prices: pd.DataFrame,
        features: pd.DataFrame,
        resolved_feature_availability: ResolvedFeatureAvailability,
    ) -> bool:
        cache_path = self.data_load_path(cache_key)
        try:
            cache_path.mkdir(parents=True, exist_ok=True)
            prices_row_groups = _write_parquet_row_groups(
                prices,
                cache_path / "prices.parquet",
                row_group_rows=DATA_LOAD_CACHE_ROW_GROUP_ROWS,
            )
            features_row_groups = _write_parquet_row_groups(
                features,
                cache_path / "features.parquet",
                row_group_rows=DATA_LOAD_CACHE_ROW_GROUP_ROWS,
            )
            metadata = {
                "schema_version": DATASET_CACHE_SCHEMA_VERSION,
                "cache_version": DATASET_CACHE_VERSION,
                "cache_layer": DATA_LOAD_CACHE_LAYER,
                "cache_key": cache_key,
                "resolved_feature_availability": asdict(resolved_feature_availability),
                "n_prices_rows": int(len(prices)),
                "n_prices_assets": int(prices["asset"].nunique()) if "asset" in prices else None,
                "n_prices_dates": int(prices["date"].nunique()) if "date" in prices else None,
                "n_features_rows": int(len(features)),
                "n_features_assets": (
                    int(features["asset"].nunique()) if "asset" in features else None
                ),
                "n_features_dates": (
                    int(features["date"].nunique()) if "date" in features else None
                ),
                "prices_row_groups": prices_row_groups,
                "features_row_groups": features_row_groups,
                "row_group_rows": DATA_LOAD_CACHE_ROW_GROUP_ROWS,
            }
            _write_metadata(cache_path / "metadata.json", metadata)
            return True
        except Exception:
            return False

    def load_prepared_inputs(self, cache_key: str) -> PreparedInputsCacheEntry | None:
        return load_prepared_inputs_cache(
            cache_dir=self.prepared_inputs_root_dir,
            cache_key=cache_key,
        )

    def write_prepared_inputs(
        self,
        *,
        cache_key: str,
        features: pd.DataFrame,
        labels: pd.DataFrame,
        forward_label_df: pd.DataFrame,
        data_health: dict[str, object],
        winsor_clip_count: int,
        feature_columns: tuple[str, ...] | None = None,
    ) -> bool:
        return write_prepared_inputs_cache(
            cache_dir=self.prepared_inputs_root_dir,
            cache_key=cache_key,
            features=features,
            labels=labels,
            forward_label_df=forward_label_df,
            data_health=data_health,
            winsor_clip_count=winsor_clip_count,
            feature_columns=feature_columns,
        )


def build_dataset_cache_key(payload: Mapping[str, object]) -> str:
    stable_payload = {
        "cache_version": DATASET_CACHE_VERSION,
        **dict(payload),
    }
    text = json.dumps(
        jsonable_for_cache(stable_payload),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


def file_signature(path: str | Path) -> dict[str, object]:
    resolved = Path(path).resolve()
    try:
        stat = resolved.stat()
    except FileNotFoundError:
        return {"path": str(resolved), "exists": False}
    return {
        "path": str(resolved),
        "exists": True,
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def jsonable_for_cache(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): jsonable_for_cache(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [jsonable_for_cache(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def prepared_inputs_cache_path(
    *,
    cache_dir: str | Path | None,
    cache_key: str | None,
) -> Path | None:
    if cache_dir is None or cache_key is None:
        return None
    if not str(cache_key).strip():
        return None
    return Path(cache_dir).resolve() / str(cache_key)


def prepared_inputs_cache_exists(
    *,
    cache_dir: str | Path | None,
    cache_key: str | None,
) -> bool:
    cache_path = prepared_inputs_cache_path(cache_dir=cache_dir, cache_key=cache_key)
    if cache_path is None:
        return False
    return all(
        (cache_path / filename).exists()
        for filename in (
            "metadata.json",
            "features.npy",
            "labeled_features.npy",
            "columns.json",
            "index.parquet",
            "labeled_index.parquet",
            "forward_label.parquet",
        )
    )


def prepared_inputs_cache_metadata(
    *,
    cache_dir: str | Path | None,
    cache_key: str | None,
) -> dict[str, object] | None:
    cache_path = prepared_inputs_cache_path(cache_dir=cache_dir, cache_key=cache_key)
    if cache_path is None:
        return None
    metadata_path = cache_path / "metadata.json"
    if not metadata_path.exists():
        return None
    try:
        metadata = _read_metadata(metadata_path)
    except Exception:
        return None
    if metadata.get("cache_key") != cache_key:
        return None
    if metadata.get("cache_version") != DATASET_CACHE_VERSION:
        return None
    if metadata.get("cache_layer") != PREPARED_INPUTS_CACHE_LAYER:
        return None
    return metadata


def load_prepared_inputs_cache(
    *,
    cache_dir: str | Path | None,
    cache_key: str | None,
) -> PreparedInputsCacheEntry | None:
    cache_path = prepared_inputs_cache_path(cache_dir=cache_dir, cache_key=cache_key)
    if cache_path is None:
        return None
    metadata_path = cache_path / "metadata.json"
    features_path = cache_path / "features.npy"
    labeled_features_path = cache_path / "labeled_features.npy"
    columns_path = cache_path / "columns.json"
    index_path = cache_path / "index.parquet"
    labeled_index_path = cache_path / "labeled_index.parquet"
    forward_label_path = cache_path / "forward_label.parquet"
    if not (
        metadata_path.exists()
        and features_path.exists()
        and labeled_features_path.exists()
        and columns_path.exists()
        and index_path.exists()
        and labeled_index_path.exists()
        and forward_label_path.exists()
    ):
        return None
    try:
        metadata = _read_metadata(metadata_path)
        if metadata.get("cache_key") != cache_key:
            return None
        cache_version = metadata.get("cache_version")
        if cache_version is not None and cache_version != DATASET_CACHE_VERSION:
            return None
        cache_layer = metadata.get("cache_layer")
        if cache_layer is not None and cache_layer != PREPARED_INPUTS_CACHE_LAYER:
            return None
        data_health = metadata.get("data_health")
        if not isinstance(data_health, dict):
            return None
        if metadata.get("feature_dtype") != PREPARED_INPUTS_FEATURE_DTYPE:
            return None
        if metadata.get("row_order") != PREPARED_INPUTS_ROW_ORDER:
            return None
        raw_columns = json.loads(columns_path.read_text(encoding="utf-8"))
        if not isinstance(raw_columns, list) or not all(
            isinstance(item, str) and item for item in raw_columns
        ):
            return None
        feature_columns = tuple(raw_columns)
        feature_values = np.load(features_path, mmap_mode="r")
        labeled_feature_values = np.load(labeled_features_path, mmap_mode="r")
        if str(feature_values.dtype) != PREPARED_INPUTS_FEATURE_DTYPE:
            return None
        if str(labeled_feature_values.dtype) != PREPARED_INPUTS_FEATURE_DTYPE:
            return None
        if feature_values.ndim != 2 or feature_values.shape[1] != len(feature_columns):
            return None
        if (
            labeled_feature_values.ndim != 2
            or labeled_feature_values.shape[1] != len(feature_columns)
        ):
            return None
        index_df = pd.read_parquet(index_path)
        labeled_index_df = pd.read_parquet(labeled_index_path)
        if len(index_df) != int(feature_values.shape[0]):
            return None
        if len(labeled_index_df) != int(labeled_feature_values.shape[0]):
            return None
        if not {"date", "asset", "label"}.issubset(index_df.columns):
            return None
        if not {"date", "asset", "label"}.issubset(labeled_index_df.columns):
            return None
        labels = index_df[["date", "asset", "label"]].copy()
        return PreparedInputsCacheEntry(
            features=pd.DataFrame(),
            labels=labels,
            forward_label_df=pd.read_parquet(forward_label_path),
            data_health=dict(data_health),
            winsor_clip_count=_int_from_metadata(metadata.get("winsor_clip_count"), default=0),
            metadata=metadata,
            numpy_entry=PreparedInputsNumpyCacheEntry(
                feature_values=feature_values,
                feature_columns=feature_columns,
                index_df=index_df,
                labeled_feature_values=labeled_feature_values,
                labeled_index_df=labeled_index_df,
                metadata=metadata,
            ),
        )
    except Exception:
        return None


def write_prepared_inputs_cache(
    *,
    cache_dir: str | Path | None,
    cache_key: str | None,
    features: pd.DataFrame,
    labels: pd.DataFrame,
    forward_label_df: pd.DataFrame,
    data_health: dict[str, object],
    winsor_clip_count: int,
    feature_columns: tuple[str, ...] | None = None,
) -> bool:
    cache_path = prepared_inputs_cache_path(cache_dir=cache_dir, cache_key=cache_key)
    if cache_path is None:
        return False
    try:
        cache_path.mkdir(parents=True, exist_ok=True)
        selected_feature_columns = tuple(feature_columns or _infer_feature_columns(features))
        if not selected_feature_columns:
            return False
        missing = set(selected_feature_columns) - set(features.columns)
        if missing:
            return False
        feature_manifest_stats = _feature_manifest_stats(
            features,
            feature_columns=selected_feature_columns,
        )
        merged = features.merge(labels, on=["date", "asset"], how="left", validate="one_to_one")
        merged = merged.sort_values(["date", "asset"], kind="mergesort").reset_index(drop=True)
        del features, labels
        _write_feature_matrix_memmap(
            merged,
            cache_path / "features.npy",
            feature_columns=selected_feature_columns,
            chunk_rows=PREPARED_INPUTS_NUMPY_CHUNK_ROWS,
        )
        index_df = _prepared_index_frame(
            merged,
            feature_columns=selected_feature_columns,
        )
        index_df.to_parquet(cache_path / "index.parquet", index=False)

        labeled_mask = pd.to_numeric(index_df["label"], errors="coerce").notna().to_numpy()
        labeled_indices = np.flatnonzero(labeled_mask)
        labeled_index_df = index_df.loc[labeled_mask].reset_index(drop=True)
        labeled_index_df.to_parquet(cache_path / "labeled_index.parquet", index=False)
        n_features_rows = int(len(index_df))
        n_features_assets = int(index_df["asset"].nunique()) if "asset" in index_df else None
        n_features_dates = int(index_df["date"].nunique()) if "date" in index_df else None
        n_labeled_rows = int(len(labeled_index_df))
        n_labeled_dates = (
            int(labeled_index_df["date"].nunique()) if "date" in labeled_index_df else None
        )
        del merged, index_df, labeled_index_df, labeled_mask
        gc.collect()
        _write_labeled_feature_matrix_memmap(
            cache_path / "features.npy",
            cache_path / "labeled_features.npy",
            labeled_indices=labeled_indices,
            n_features=len(selected_feature_columns),
            chunk_rows=PREPARED_INPUTS_NUMPY_CHUNK_ROWS,
        )
        del labeled_indices
        gc.collect()
        (cache_path / "columns.json").write_text(
            json.dumps(list(selected_feature_columns), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        forward_label_df.to_parquet(cache_path / "forward_label.parquet", index=False)
        metadata = {
            "schema_version": DATASET_CACHE_SCHEMA_VERSION,
            "cache_version": DATASET_CACHE_VERSION,
            "cache_layer": PREPARED_INPUTS_CACHE_LAYER,
            "cache_key": cache_key,
            "data_health": data_health,
            "winsor_clip_count": int(winsor_clip_count),
            "feature_dtype": PREPARED_INPUTS_FEATURE_DTYPE,
            "feature_columns": list(selected_feature_columns),
            "feature_count": int(len(selected_feature_columns)),
            "row_order": PREPARED_INPUTS_ROW_ORDER,
            "layout": "numpy_with_compact_labeled_matrix",
            "matrix_write_mode": "chunked_memmap",
            "numpy_chunk_rows": int(PREPARED_INPUTS_NUMPY_CHUNK_ROWS),
            "n_features_rows": n_features_rows,
            "n_features_assets": n_features_assets,
            "n_features_dates": n_features_dates,
            "n_labeled_rows": n_labeled_rows,
            "n_labeled_dates": n_labeled_dates,
            "feature_manifest_stats": feature_manifest_stats,
            "migration_hint": "prepared cache schema v1 -> v2, please rerun cold once",
        }
        _write_metadata(cache_path / "metadata.json", metadata)
        return True
    except Exception:
        return False


def _infer_feature_columns(features: pd.DataFrame) -> tuple[str, ...]:
    reserved = {"date", "asset", "label", "target", "known_at"}
    return tuple(
        str(column)
        for column in features.columns
        if str(column) not in reserved and pd.api.types.is_numeric_dtype(features[column])
    )


def _prepared_index_frame(
    merged: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...],
) -> pd.DataFrame:
    sidecar_columns = [
        column
        for column in merged.columns
        if column not in {*feature_columns, "forward_return", "target", "factor", "value"}
        and column not in {"label"}
    ]
    ordered_columns = [*sidecar_columns, "label"]
    seen: set[str] = set()
    unique_columns: list[str] = []
    for column in ordered_columns:
        if column in seen:
            continue
        seen.add(column)
        unique_columns.append(column)
    return merged.loc[:, unique_columns].copy()


def _write_feature_matrix_memmap(
    frame: pd.DataFrame,
    path: Path,
    *,
    feature_columns: tuple[str, ...],
    chunk_rows: int,
) -> None:
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be > 0")
    feature_positions = [frame.columns.get_loc(column) for column in feature_columns]
    matrix = np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=np.float32,
        shape=(len(frame), len(feature_columns)),
    )
    try:
        for start in range(0, len(frame), chunk_rows):
            stop = min(start + chunk_rows, len(frame))
            values = frame.iloc[start:stop, feature_positions].to_numpy(
                dtype=np.float32,
                copy=False,
            )
            matrix[start:stop, :] = values
        matrix.flush()
    finally:
        del matrix


def _write_labeled_feature_matrix_memmap(
    source_path: Path,
    output_path: Path,
    *,
    labeled_indices: np.ndarray,
    n_features: int,
    chunk_rows: int,
) -> None:
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be > 0")
    source = np.load(source_path, mmap_mode="r")
    matrix = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(labeled_indices), n_features),
    )
    try:
        for start in range(0, len(labeled_indices), chunk_rows):
            stop = min(start + chunk_rows, len(labeled_indices))
            matrix[start:stop, :] = source[labeled_indices[start:stop]]
        matrix.flush()
    finally:
        del matrix, source


def _feature_manifest_stats(
    merged: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for column in feature_columns:
        series = pd.to_numeric(merged[column], errors="coerce")
        rows.append(
            {
                "feature": column,
                "non_null_ratio": float(series.notna().mean()) if len(series) else None,
                "mean": float(series.mean()) if series.notna().any() else None,
                "std": float(series.std(ddof=1)) if series.notna().sum() > 1 else None,
            }
        )
    return rows


def _read_metadata(path: Path) -> dict[str, object]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("cache metadata must be a JSON object")
    return dict(raw)


def _write_metadata(path: Path, metadata: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_parquet_row_groups(
    frame: pd.DataFrame,
    path: Path,
    *,
    row_group_rows: int,
) -> int:
    if row_group_rows <= 0:
        raise ValueError("row_group_rows must be > 0")
    if path.exists():
        path.unlink()
    writer: pq.ParquetWriter | None = None
    row_groups = 0
    try:
        if frame.empty:
            table = pa.Table.from_pandas(frame, preserve_index=False)
            pq.write_table(table, path, compression="zstd")
            return 1
        for start in range(0, len(frame), row_group_rows):
            chunk = frame.iloc[start : start + row_group_rows]
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(path, table.schema, compression="zstd")
            writer.write_table(table)
            row_groups += 1
    finally:
        if writer is not None:
            writer.close()
    return row_groups


def _resolved_feature_availability_from_mapping(
    payload: Mapping[object, object],
) -> ResolvedFeatureAvailability:
    raw_safety_lag_days = payload.get("safety_lag_days")
    return ResolvedFeatureAvailability(
        mode=str(payload.get("mode") or ""),
        requested_column=_optional_string(payload.get("requested_column")),
        source_column=_optional_string(payload.get("source_column")),
        known_at_col=str(payload.get("known_at_col") or ""),
        safety_lag_days=_optional_int_from_metadata(raw_safety_lag_days),
        shifted_rows=_int_from_metadata(payload.get("shifted_rows"), default=0),
        dropped_rows=_int_from_metadata(payload.get("dropped_rows"), default=0),
    )


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_int_from_metadata(value: object) -> int | None:
    if value is None:
        return None
    return _int_from_metadata(value, default=0)


def _int_from_metadata(value: object, *, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        return int(value)
    raise ValueError(f"cache metadata integer field has unsupported value: {value!r}")
