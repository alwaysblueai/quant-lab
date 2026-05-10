from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_lab.model_factor.dataset_cache import (
    PreparedInputsNumpyCacheEntry,
)

# Cross-module imports (auto-added)
from .config import TrainingSpec
from .internals import _DateIndexedRows, _PreparedModelArrays, _TrainingWindowCache
from .types import _PERMUTATION_IMPORTANCE_RANDOM_SEED, _RowSelection


def _build_date_indexed_rows(
    frame: pd.DataFrame,
    ordered_dates: pd.DatetimeIndex,
    *,
    allow_missing: bool = False,
) -> _DateIndexedRows:
    grouped = {
        pd.Timestamp(date): np.asarray(indices, dtype=np.intp)
        for date, indices in frame.groupby("date", sort=False).indices.items()
    }
    row_indices_by_pos: list[np.ndarray] = []
    missing_dates: list[str] = []
    for raw_date in ordered_dates:
        date = pd.Timestamp(raw_date)
        indices = grouped.get(date)
        if indices is None:
            if allow_missing:
                row_indices_by_pos.append(np.empty(0, dtype=np.intp))
            else:
                missing_dates.append(date.date().isoformat())
            continue
        row_indices_by_pos.append(indices)
    if missing_dates:
        preview = ", ".join(missing_dates[:5])
        raise ValueError(f"date-indexed frame is missing score dates: {preview}")
    return _DateIndexedRows(
        ordered_dates=pd.DatetimeIndex(ordered_dates),
        row_indices_by_pos=tuple(row_indices_by_pos),
    )


def _build_training_window_cache(
    *,
    date_index: _DateIndexedRows,
    merged: pd.DataFrame,
    training: TrainingSpec,
    target_horizon: int,
) -> _TrainingWindowCache:
    n_dates = int(len(date_index.row_indices_by_pos))
    labeled_by_pos = _build_labeled_date_indexed_rows(merged, date_index.ordered_dates)
    labeled_counts = np.asarray([len(indices) for indices in labeled_by_pos], dtype=np.intp)
    labeled_prefix = np.concatenate(
        [np.zeros(1, dtype=np.intp), np.cumsum(labeled_counts, dtype=np.intp)]
    )
    window_start_by_pos = np.zeros(n_dates, dtype=np.intp)
    eligible_end_by_pos = np.zeros(n_dates, dtype=np.intp)
    n_train_dates_by_pos = np.zeros(n_dates, dtype=np.intp)
    n_labeled_rows_by_pos = np.zeros(n_dates, dtype=np.intp)

    for score_idx in range(n_dates):
        window_start, eligible_end = _training_window_bounds(
            score_idx=score_idx,
            training=training,
            target_horizon=target_horizon,
        )
        window_start_by_pos[score_idx] = window_start
        eligible_end_by_pos[score_idx] = eligible_end
        n_train_dates_by_pos[score_idx] = max(eligible_end - window_start, 0)
        n_labeled_rows_by_pos[score_idx] = int(
            labeled_prefix[eligible_end] - labeled_prefix[window_start]
        )

    return _TrainingWindowCache(
        row_indices_by_date_pos=date_index.row_indices_by_pos,
        labeled_row_indices_by_date_pos=labeled_by_pos,
        window_start_by_pos=window_start_by_pos,
        eligible_end_by_pos=eligible_end_by_pos,
        n_train_dates_by_pos=n_train_dates_by_pos,
        n_labeled_rows_by_pos=n_labeled_rows_by_pos,
    )


def _prepare_model_arrays(
    merged: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...],
) -> _PreparedModelArrays:
    feature_frame = merged.loc[:, list(feature_columns)]
    feature_values = feature_frame.to_numpy(dtype=np.float32, copy=False)
    if not feature_values.flags.c_contiguous:
        feature_values = np.ascontiguousarray(feature_values, dtype=np.float32)
    labels = pd.to_numeric(merged["label"], errors="coerce").to_numpy(
        dtype=np.float32,
        copy=True,
    )
    dates = pd.to_datetime(merged["date"]).to_numpy(copy=True)
    assets = pd.Categorical(merged["asset"])
    return _PreparedModelArrays(
        feature_values=feature_values,
        labels=labels,
        dates=dates,
        assets=assets,
    )


def _prepare_model_arrays_from_numpy_cache(
    entry: PreparedInputsNumpyCacheEntry,
) -> _PreparedModelArrays:
    index_df = entry.index_df
    labeled_index_df = entry.labeled_index_df
    return _PreparedModelArrays(
        feature_values=entry.feature_values,
        labels=pd.to_numeric(index_df["label"], errors="coerce").to_numpy(
            dtype=np.float32,
            copy=True,
        ),
        dates=pd.to_datetime(index_df["date"]).to_numpy(copy=True),
        assets=pd.Categorical(index_df["asset"]),
        training_feature_values=entry.labeled_feature_values,
        training_labels=pd.to_numeric(labeled_index_df["label"], errors="coerce").to_numpy(
            dtype=np.float32,
            copy=True,
        ),
        training_dates=pd.to_datetime(labeled_index_df["date"]).to_numpy(copy=True),
    )


def _row_selection_mode(selection: _RowSelection) -> str:
    return "contiguous_slice" if isinstance(selection, slice) else "advanced_index"


def _row_selection_length(selection: _RowSelection, *, n_rows: int) -> int:
    if isinstance(selection, slice):
        start, stop, step = selection.indices(n_rows)
        if step <= 0:
            return 0
        return max((stop - start + step - 1) // step, 0)
    return int(len(selection))


def _row_selection_to_indices(selection: _RowSelection, *, n_rows: int) -> np.ndarray:
    if isinstance(selection, slice):
        start, stop, step = selection.indices(n_rows)
        return np.arange(start, stop, step, dtype=np.intp)
    return selection.astype(np.intp, copy=False)


def _training_matrix_from_selection(
    feature_values: np.ndarray,
    selection: _RowSelection,
) -> np.ndarray:
    selected = feature_values[selection]
    if isinstance(selection, slice):
        return np.array(selected, dtype=np.float32, order="C", copy=True)
    return np.asarray(selected, dtype=np.float32, order="C")


def _feature_importance_training_slice_from_arrays(
    prepared_arrays: _PreparedModelArrays,
    *,
    row_selection: _RowSelection,
    feature_columns: tuple[str, ...],
    model_version: int,
    max_rows: int,
) -> pd.DataFrame:
    columns = [*feature_columns, "label"]
    feature_values = (
        prepared_arrays.training_feature_values
        if prepared_arrays.training_feature_values is not None
        else prepared_arrays.feature_values
    )
    labels_array = (
        prepared_arrays.training_labels
        if prepared_arrays.training_labels is not None
        else prepared_arrays.labels
    )
    row_indices = _row_selection_to_indices(
        row_selection,
        n_rows=len(labels_array),
    )
    if len(row_indices) == 0:
        return pd.DataFrame(columns=columns)

    labels = labels_array[row_indices]
    valid_pos = np.flatnonzero(np.isfinite(labels))
    if valid_pos.size < 2:
        return pd.DataFrame(columns=columns)
    selected_rows = row_indices[valid_pos]
    if len(selected_rows) > max_rows:
        rng = np.random.default_rng(_PERMUTATION_IMPORTANCE_RANDOM_SEED + model_version)
        sample_pos = np.sort(rng.choice(len(selected_rows), size=max_rows, replace=False))
        selected_rows = selected_rows[sample_pos]

    data: dict[str, object] = {
        column: feature_values[selected_rows, idx] for idx, column in enumerate(feature_columns)
    }
    data["label"] = labels_array[selected_rows].astype(float, copy=False)
    return pd.DataFrame(data, columns=columns)


def _build_labeled_date_indexed_rows(
    merged: pd.DataFrame,
    ordered_dates: pd.DatetimeIndex,
) -> tuple[np.ndarray, ...]:
    if "label" not in merged.columns:
        return tuple(np.empty(0, dtype=np.intp) for _ in ordered_dates)
    labeled = merged[merged["label"].notna()]
    grouped = {
        pd.Timestamp(date): np.asarray(indices, dtype=np.intp)
        for date, indices in labeled.groupby("date", sort=False).groups.items()
    }
    return tuple(
        grouped.get(pd.Timestamp(raw_date), np.empty(0, dtype=np.intp))
        for raw_date in ordered_dates
    )


def _training_window_bounds(
    *,
    score_idx: int,
    training: TrainingSpec,
    target_horizon: int,
) -> tuple[int, int]:
    horizon = max(int(target_horizon), 1)
    purge = horizon - 1
    eligible_end = int(score_idx) - purge
    if eligible_end <= 0:
        return 0, 0

    if training.window_type == "expanding":
        window_start = 0
    else:
        window_start = max(0, eligible_end - int(training.train_window_n_dates or 0))
    if window_start >= eligible_end:
        return 0, 0
    return window_start, eligible_end


def _training_slice_from_preindexed_dates(
    merged: pd.DataFrame,
    *,
    date_index: _DateIndexedRows,
    score_idx: int,
    training: TrainingSpec,
    target_horizon: int,
) -> pd.DataFrame:
    # Same purged walk-forward contract as _training_slice, but using date
    # positions instead of rescanning the full table for every score date.
    window_start, eligible_end = _training_window_bounds(
        score_idx=score_idx,
        training=training,
        target_horizon=target_horizon,
    )
    if window_start >= eligible_end:
        return merged.iloc[0:0].copy()

    selected_indices = date_index.row_indices_by_pos[window_start:eligible_end]
    if not selected_indices:
        return merged.iloc[0:0].copy()
    row_idx = np.concatenate(selected_indices)
    return merged.take(row_idx).copy()


def _training_slice(
    merged: pd.DataFrame,
    *,
    score_date: pd.Timestamp,
    training: TrainingSpec,
    target_horizon: int,
) -> pd.DataFrame:
    # Purged walk-forward: a sample at date t has label that resolves at t+h, so we
    # drop the most-recent (h-1) dates from the training pool. This guarantees every
    # label in the training slice was realized strictly before score_date.
    horizon = max(int(target_horizon), 1)
    all_dates = pd.Index(
        merged.loc[merged["date"] < score_date, "date"].drop_duplicates()
    ).sort_values()
    if len(all_dates) == 0:
        return merged.iloc[0:0].copy()
    purge = horizon - 1
    if purge > 0:
        eligible_dates = all_dates[:-purge] if len(all_dates) > purge else all_dates[:0]
    else:
        eligible_dates = all_dates
    if len(eligible_dates) == 0:
        return merged.iloc[0:0].copy()

    if training.window_type == "expanding":
        return merged[merged["date"].isin(eligible_dates)].copy()

    tail_dates = eligible_dates[-int(training.train_window_n_dates or 0) :]
    return merged[merged["date"].isin(tail_dates)].copy()
