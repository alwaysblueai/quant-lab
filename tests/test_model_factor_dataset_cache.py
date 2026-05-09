from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from alpha_lab.model_factor import dataset_cache as dataset_cache_module
from alpha_lab.model_factor.dataset_cache import (
    DATA_LOAD_CACHE_LAYER,
    DATASET_CACHE_VERSION,
    PREPARED_INPUTS_CACHE_LAYER,
    ModelFactorDatasetCache,
    ResolvedFeatureAvailability,
)


def test_model_factor_dataset_cache_key_changes_for_reuse_dimensions(tmp_path: Path) -> None:
    cache = ModelFactorDatasetCache(tmp_path)
    base_payload: dict[str, object] = {
        "features_path": {
            "path": "/data/features.parquet",
            "exists": True,
            "size": 100,
            "mtime_ns": 1,
        },
        "features_source_path": {
            "path": "/data/features.csv",
            "exists": True,
            "size": 100,
            "mtime_ns": 1,
        },
        "prices_path": {
            "path": "/data/prices.parquet",
            "exists": True,
            "size": 200,
            "mtime_ns": 2,
        },
        "universe": {"enabled": False, "path": None},
        "universe_path": None,
        "feature_columns": ["mom_20", "turnover_5"],
        "feature_availability": {
            "mode": "required_timestamp",
            "column": "known_at",
            "safety_lag_days": None,
        },
        "feature_preprocess": {
            "missing_policy": "median_impute",
            "scale_features": "auto",
            "cross_sectional_transform": "winsorize_zscore",
            "cross_sectional_group_scope": "date",
            "industry_group_column": None,
        },
        "target": {"horizon": 5, "winsorize_zscore": None},
        "price_columns": ["date", "asset", "open", "close"],
        "optional_price_columns": ["limit_up", "limit_down"],
        "evaluation_profile": "default_research",
    }
    base_key = cache.build_key(base_payload)

    mutations: dict[str, object] = {
        "feature_columns": ["mom_20"],
        "target": {"horizon": 10, "winsorize_zscore": None},
        "feature_preprocess": {
            "missing_policy": "median_impute",
            "scale_features": "none",
            "cross_sectional_transform": "rank",
            "cross_sectional_group_scope": "date",
            "industry_group_column": None,
        },
        "universe": {"enabled": True, "path": "/data/universe.parquet"},
    }
    for field, value in mutations.items():
        mutated = deepcopy(base_payload)
        mutated[field] = value
        assert cache.build_key(mutated) != base_key


def test_model_factor_dataset_cache_roundtrip(
    tmp_path: Path,
) -> None:
    cache = ModelFactorDatasetCache(tmp_path)
    cache_key = cache.build_key({"dataset_id": "demo", "feature_columns": ["x1"]})
    prices = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "asset": ["A", "A"],
            "close": [10.0, 10.5],
        }
    )
    features = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "asset": ["A", "A"],
            "x1": [0.1, 0.2],
            "known_at": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        }
    )
    resolved = ResolvedFeatureAvailability(
        mode="required_timestamp",
        requested_column="known_at",
        source_column="known_at",
        known_at_col="known_at",
        safety_lag_days=None,
        shifted_rows=0,
        dropped_rows=0,
    )

    assert cache.load_data_load(cache_key) is None
    assert cache.write_data_load(
        cache_key=cache_key,
        prices=prices,
        features=features,
        resolved_feature_availability=resolved,
    )
    data_entry = cache.load_data_load(cache_key)
    assert data_entry is not None
    assert data_entry.metadata["cache_version"] == DATASET_CACHE_VERSION
    assert data_entry.metadata["cache_layer"] == DATA_LOAD_CACHE_LAYER
    assert data_entry.resolved_feature_availability == resolved
    pd.testing.assert_frame_equal(data_entry.prices, prices)
    pd.testing.assert_frame_equal(data_entry.features, features)
    metadata_only_entry = cache.load_data_load(cache_key, include_features=False)
    assert metadata_only_entry is not None
    assert metadata_only_entry.features_loaded is False
    assert metadata_only_entry.features.empty
    pd.testing.assert_frame_equal(metadata_only_entry.prices, prices)
    assert metadata_only_entry.resolved_feature_availability == resolved

    labels = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01"]),
            "asset": ["A"],
            "label": [0.05],
        }
    )
    forward_label_df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01"]),
            "asset": ["A"],
            "factor": ["forward_return_1"],
            "value": [0.05],
        }
    )
    data_health = {"coverage_ratio": 1.0, "target_mean": 0.05}
    assert cache.write_prepared_inputs(
        cache_key=cache_key,
        features=features,
        labels=labels,
        forward_label_df=forward_label_df,
        data_health=data_health,
        winsor_clip_count=2,
    )
    assert cache.prepared_inputs_exists(cache_key)
    prepared_entry = cache.load_prepared_inputs(cache_key)
    assert prepared_entry is not None
    assert prepared_entry.metadata["cache_version"] == DATASET_CACHE_VERSION
    assert prepared_entry.metadata["cache_layer"] == PREPARED_INPUTS_CACHE_LAYER
    assert prepared_entry.data_health == data_health
    assert prepared_entry.winsor_clip_count == 2
    assert prepared_entry.features.empty
    assert prepared_entry.numpy_entry is not None
    assert prepared_entry.numpy_entry.feature_columns == ("x1",)
    assert str(prepared_entry.numpy_entry.feature_values.dtype) == "float32"
    assert prepared_entry.numpy_entry.feature_values.shape == (2, 1)
    assert prepared_entry.numpy_entry.labeled_feature_values.shape == (1, 1)
    assert prepared_entry.metadata["row_order"] == "date_asset"
    assert prepared_entry.metadata["layout"] == "numpy_with_compact_labeled_matrix"
    assert prepared_entry.metadata["n_features_rows"] == 2
    assert prepared_entry.metadata["n_labeled_rows"] == 1
    assert len(prepared_entry.labels) == 2
    assert prepared_entry.labels["label"].notna().sum() == 1
    pd.testing.assert_frame_equal(prepared_entry.forward_label_df, forward_label_df)


def test_prepared_inputs_cache_streams_numpy_matrices_in_chunks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(dataset_cache_module, "PREPARED_INPUTS_NUMPY_CHUNK_ROWS", 2)
    cache = ModelFactorDatasetCache(tmp_path)
    cache_key = cache.build_key({"dataset_id": "chunked", "feature_columns": ["x1", "x2"]})
    features = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2024-01-02",
                    "2024-01-01",
                    "2024-01-01",
                    "2024-01-03",
                    "2024-01-02",
                ]
            ),
            "asset": ["B", "B", "A", "A", "A"],
            "x1": [5.0, 2.0, 1.0, 7.0, 4.0],
            "x2": [50.0, 20.0, 10.0, 70.0, 40.0],
            "known_at": pd.to_datetime(
                [
                    "2024-01-02",
                    "2024-01-01",
                    "2024-01-01",
                    "2024-01-03",
                    "2024-01-02",
                ]
            ),
        }
    )
    labels = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "asset": ["A", "B", "A"],
            "label": [0.1, -0.2, 0.3],
        }
    )
    forward_label_df = labels.rename(columns={"label": "value"}).assign(
        factor="forward_return_1"
    )[["date", "asset", "factor", "value"]]

    assert cache.write_prepared_inputs(
        cache_key=cache_key,
        features=features,
        labels=labels,
        forward_label_df=forward_label_df,
        data_health={"coverage_ratio": 0.6},
        winsor_clip_count=0,
        feature_columns=("x1", "x2"),
    )

    entry = cache.load_prepared_inputs(cache_key)
    assert entry is not None
    assert entry.numpy_entry is not None
    merged = features.merge(labels, on=["date", "asset"], how="left", validate="one_to_one")
    merged = merged.sort_values(["date", "asset"], kind="mergesort").reset_index(drop=True)
    expected_features = merged[["x1", "x2"]].to_numpy(dtype=np.float32)
    expected_labeled = merged.loc[merged["label"].notna(), ["x1", "x2"]].to_numpy(
        dtype=np.float32
    )
    np.testing.assert_array_equal(
        np.asarray(entry.numpy_entry.feature_values),
        expected_features,
    )
    np.testing.assert_array_equal(
        np.asarray(entry.numpy_entry.labeled_feature_values),
        expected_labeled,
    )
    assert entry.metadata["matrix_write_mode"] == "chunked_memmap"
    assert entry.metadata["numpy_chunk_rows"] == 2


def test_data_load_cache_writes_multiple_row_groups_for_large_frames(
    tmp_path: Path,
) -> None:
    cache = ModelFactorDatasetCache(tmp_path)
    cache_key = cache.build_key({"dataset_id": "row_groups", "feature_columns": ["x1"]})
    n_rows = 260_001
    dates = pd.date_range("2024-01-01", periods=n_rows, freq="min")
    prices = pd.DataFrame(
        {
            "date": dates,
            "asset": "A",
            "close": 10.0,
        }
    )
    features = pd.DataFrame(
        {
            "date": dates,
            "asset": "A",
            "x1": pd.Series(range(n_rows), dtype="float32"),
            "known_at": dates,
        }
    )
    resolved = ResolvedFeatureAvailability(
        mode="required_timestamp",
        requested_column="known_at",
        source_column="known_at",
        known_at_col="known_at",
        safety_lag_days=None,
        shifted_rows=0,
        dropped_rows=0,
    )

    assert cache.write_data_load(
        cache_key=cache_key,
        prices=prices,
        features=features,
        resolved_feature_availability=resolved,
    )
    cache_path = cache.data_load_path(cache_key)
    features_file = pq.ParquetFile(cache_path / "features.parquet")
    prices_file = pq.ParquetFile(cache_path / "prices.parquet")
    assert features_file.num_row_groups == 3
    assert prices_file.num_row_groups == 3
    entry = cache.load_data_load(cache_key)
    assert entry is not None
    assert entry.metadata["features_row_groups"] == 3
    assert entry.metadata["prices_row_groups"] == 3
    assert len(entry.features) == n_rows
    assert len(entry.prices) == n_rows
