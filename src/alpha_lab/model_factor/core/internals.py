from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

# Cross-module imports (auto-added)
from ._utils import _indices_as_contiguous_slice
from .config import ModelSpec
from .types import _RowSelection


@dataclass(frozen=True)
class _FittedModelBundle:
    pipeline: Pipeline
    model_version: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    n_train_dates: int
    n_train_rows: int
    scale_mode: str
    model_family: str
    model_params: dict[str, object]
    selected_candidate_id: str | None = None
    selection_score: float | None = None
    selected_candidate_turnover: float | None = None


@dataclass(frozen=True)
class _FeatureImportanceRequest:
    pipeline: Pipeline
    train_slice: pd.DataFrame
    feature_columns: tuple[str, ...]
    model_family: str
    model_version: int
    fit_date: pd.Timestamp
    trained_until: pd.Timestamp


@dataclass(frozen=True)
class _ModelSelectionOutcome:
    selected_model: ModelSpec
    selected_candidate_id: str
    selected_score: float | None
    selected_turnover: float | None
    status: str
    n_splits_used: int
    rows: list[dict[str, object]]


@dataclass(frozen=True)
class _DateIndexedRows:
    ordered_dates: pd.DatetimeIndex
    row_indices_by_pos: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class _PreparedModelArrays:
    feature_values: np.ndarray
    labels: np.ndarray
    dates: np.ndarray
    assets: pd.Categorical
    training_feature_values: np.ndarray | None = None
    training_labels: np.ndarray | None = None
    training_dates: np.ndarray | None = None


@dataclass
class _TrainingWindowCache:
    row_indices_by_date_pos: tuple[np.ndarray, ...]
    labeled_row_indices_by_date_pos: tuple[np.ndarray, ...]
    window_start_by_pos: np.ndarray
    eligible_end_by_pos: np.ndarray
    n_train_dates_by_pos: np.ndarray
    n_labeled_rows_by_pos: np.ndarray
    _labeled_window_rows: dict[int, np.ndarray] = field(default_factory=dict)

    def labeled_row_selection(self, score_idx: int) -> _RowSelection:
        window_start = int(self.window_start_by_pos[score_idx])
        eligible_end = int(self.eligible_end_by_pos[score_idx])
        if window_start >= eligible_end:
            return np.empty(0, dtype=np.intp)

        start: int | None = None
        end: int | None = None
        for indices in self.labeled_row_indices_by_date_pos[window_start:eligible_end]:
            if len(indices) == 0:
                continue
            date_slice = _indices_as_contiguous_slice(indices)
            if date_slice is None:
                return self.labeled_row_indices(score_idx)
            if start is None:
                start = int(date_slice.start)
                end = int(date_slice.stop)
                continue
            if int(date_slice.start) != end:
                return self.labeled_row_indices(score_idx)
            end = int(date_slice.stop)

        if start is None or end is None:
            return np.empty(0, dtype=np.intp)
        return slice(start, end)

    def labeled_row_indices(self, score_idx: int) -> np.ndarray:
        window_start = int(self.window_start_by_pos[score_idx])
        eligible_end = int(self.eligible_end_by_pos[score_idx])
        if window_start >= eligible_end:
            row_idx = np.empty(0, dtype=np.intp)
        else:
            selected = self.labeled_row_indices_by_date_pos[window_start:eligible_end]
            non_empty = [indices for indices in selected if len(indices) > 0]
            row_idx = (
                np.concatenate(non_empty).astype(np.intp, copy=False)
                if non_empty
                else np.empty(0, dtype=np.intp)
            )
        return row_idx

    def labeled_slice(self, merged: pd.DataFrame, score_idx: int) -> pd.DataFrame:
        row_idx = self.labeled_row_indices(score_idx)
        if len(row_idx) == 0:
            return merged.iloc[0:0].copy()
        return merged.take(row_idx).copy()
