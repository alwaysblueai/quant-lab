from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_MAX_BOOLEAN_MASK_LENGTH = 50_000_000
_MAX_DECISION_DATES = 5_000
_MAX_ASSETS = 10_000


@dataclass(frozen=True)
class PurgedFold:
    """One purged/embargoed split with explicit diagnostics."""

    fold_id: int
    train_indices: np.ndarray
    test_indices: np.ndarray
    purged_indices: np.ndarray
    embargoed_indices: np.ndarray
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    @property
    def n_train(self) -> int:
        return int(len(self.train_indices))

    @property
    def n_test(self) -> int:
        return int(len(self.test_indices))

    @property
    def n_purged(self) -> int:
        return int(len(self.purged_indices))

    @property
    def n_embargoed(self) -> int:
        return int(len(self.embargoed_indices))

    def to_dict(self) -> dict[str, object]:
        return {
            "fold_id": self.fold_id,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "n_purged": self.n_purged,
            "n_embargoed": self.n_embargoed,
            "test_start": self.test_start.isoformat(),
            "test_end": self.test_end.isoformat(),
        }


def purged_kfold_split(
    samples: pd.DataFrame,
    *,
    n_splits: int,
    decision_col: str = "date",
    start_col: str = "event_start",
    end_col: str = "event_end",
    embargo_periods: int = 0,
) -> list[PurgedFold]:
    """Build purged K-fold splits using interval overlap and date embargo.

    The split unit is the unique decision date axis, so all rows with the same
    decision date stay in the same test fold.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    if embargo_periods < 0:
        raise ValueError("embargo_periods must be >= 0")

    frame = _prepare_intervals(
        samples,
        decision_col=decision_col,
        start_col=start_col,
        end_col=end_col,
    )
    _log_shape(
        step="purged_kfold_split.prepare_intervals",
        frame=frame,
    )
    unique_dates = pd.Index(frame["decision_date"].drop_duplicates().sort_values())
    _assert_max_cardinality(
        step="purged_kfold_split.prepare_intervals",
        unique_dates=int(len(unique_dates)),
        unique_assets=int(frame["asset"].nunique()) if "asset" in frame.columns else None,
    )
    if len(unique_dates) < n_splits:
        raise ValueError(
            "n_splits is larger than the number of unique decision dates; "
            f"got n_splits={n_splits}, unique_dates={len(unique_dates)}"
        )

    date_folds = [
        np.asarray(x, dtype="datetime64[ns]")
        for x in np.array_split(unique_dates, n_splits)
    ]
    splits: list[PurgedFold] = []
    for fold_id, test_dates in enumerate(date_folds):
        _assert_boolean_mask_length(
            step="purged_kfold_split.test_mask",
            length=len(frame),
        )
        test_mask = frame["decision_date"].isin(pd.to_datetime(test_dates))
        logger.info(
            "step=purged_kfold_split.test_mask fold_id=%d n_rows=%d n_test_dates=%d n_test_rows=%d",
            fold_id,
            int(len(frame)),
            int(len(test_dates)),
            int(test_mask.sum()),
        )
        split = _build_purged_fold(
            frame=frame,
            fold_id=fold_id,
            test_mask=test_mask.to_numpy(dtype=bool),
            unique_dates=unique_dates,
            embargo_periods=embargo_periods,
        )
        splits.append(split)
    return splits


def combinatorial_purged_split(
    samples: pd.DataFrame,
    *,
    n_groups: int,
    n_test_groups: int,
    decision_col: str = "date",
    start_col: str = "event_start",
    end_col: str = "event_end",
    embargo_periods: int = 0,
) -> list[PurgedFold]:
    """Simplified CPCV scaffold using contiguous date groups and combinations."""
    if n_groups < 2:
        raise ValueError("n_groups must be >= 2")
    if n_test_groups <= 0 or n_test_groups >= n_groups:
        raise ValueError("n_test_groups must be in [1, n_groups - 1]")
    if embargo_periods < 0:
        raise ValueError("embargo_periods must be >= 0")

    frame = _prepare_intervals(
        samples,
        decision_col=decision_col,
        start_col=start_col,
        end_col=end_col,
    )
    _log_shape(
        step="combinatorial_purged_split.prepare_intervals",
        frame=frame,
    )
    unique_dates = pd.Index(frame["decision_date"].drop_duplicates().sort_values())
    _assert_max_cardinality(
        step="combinatorial_purged_split.prepare_intervals",
        unique_dates=int(len(unique_dates)),
        unique_assets=int(frame["asset"].nunique()) if "asset" in frame.columns else None,
    )
    if len(unique_dates) < n_groups:
        raise ValueError(
            "n_groups is larger than the number of unique decision dates; "
            f"got n_groups={n_groups}, unique_dates={len(unique_dates)}"
        )

    groups = [pd.Index(x) for x in np.array_split(unique_dates, n_groups)]
    splits: list[PurgedFold] = []
    for fold_id, combo in enumerate(combinations(range(n_groups), n_test_groups)):
        combo_dates = pd.Index([])
        for g in combo:
            combo_dates = combo_dates.append(groups[g])
        _assert_boolean_mask_length(
            step="combinatorial_purged_split.test_mask",
            length=len(frame),
        )
        test_mask = frame["decision_date"].isin(combo_dates)
        logger.info(
            "step=combinatorial_purged_split.test_mask "
            "fold_id=%d n_rows=%d n_test_dates=%d n_test_rows=%d",
            fold_id,
            int(len(frame)),
            int(len(combo_dates)),
            int(test_mask.sum()),
        )
        split = _build_purged_fold(
            frame=frame,
            fold_id=fold_id,
            test_mask=test_mask.to_numpy(dtype=bool),
            unique_dates=unique_dates,
            embargo_periods=embargo_periods,
        )
        splits.append(split)
    return splits


def purged_fold_summary(splits: list[PurgedFold]) -> pd.DataFrame:
    """Convert split diagnostics into a stable DataFrame."""
    if not splits:
        return pd.DataFrame(
            columns=[
                "fold_id",
                "n_train",
                "n_test",
                "n_purged",
                "n_embargoed",
                "test_start",
                "test_end",
            ]
        )
    rows = [s.to_dict() for s in splits]
    out = pd.DataFrame(rows)
    out["test_start"] = pd.to_datetime(out["test_start"], errors="coerce")
    out["test_end"] = pd.to_datetime(out["test_end"], errors="coerce")
    return out[
        [
            "fold_id",
            "n_train",
            "n_test",
            "n_purged",
            "n_embargoed",
            "test_start",
            "test_end",
        ]
    ].sort_values("fold_id", kind="mergesort").reset_index(drop=True)


def overlapping_index(
    samples: pd.DataFrame,
    test_samples: pd.DataFrame,
    *,
    start_col: str = "event_start",
    end_col: str = "event_end",
) -> np.ndarray:
    """Return sample indices from ``samples`` overlapping any test interval.

    This implementation avoids pairwise ``n_samples x n_test`` boolean matrices.
    It merges the test intervals first, then checks overlaps via binary search.
    """
    if samples.empty or test_samples.empty:
        return np.asarray([], dtype=int)

    starts = pd.to_datetime(samples[start_col], errors="coerce").to_numpy(dtype="datetime64[ns]")
    ends = pd.to_datetime(samples[end_col], errors="coerce").to_numpy(dtype="datetime64[ns]")
    test_starts = pd.to_datetime(test_samples[start_col], errors="coerce").to_numpy(
        dtype="datetime64[ns]"
    )
    test_ends = pd.to_datetime(
        test_samples[end_col], errors="coerce"
    ).to_numpy(dtype="datetime64[ns]")

    if np.isnat(starts).any() or np.isnat(ends).any():
        raise ValueError("overlapping_index: samples contains invalid interval timestamps")
    if np.isnat(test_starts).any() or np.isnat(test_ends).any():
        raise ValueError("overlapping_index: test_samples contains invalid interval timestamps")
    if np.any(ends < starts):
        raise ValueError("overlapping_index: samples has event_end < event_start")
    if np.any(test_ends < test_starts):
        raise ValueError("overlapping_index: test_samples has event_end < event_start")

    starts_ns = starts.astype(np.int64, copy=False)
    ends_ns = ends.astype(np.int64, copy=False)
    test_starts_ns = test_starts.astype(np.int64, copy=False)
    test_ends_ns = test_ends.astype(np.int64, copy=False)
    merged_starts, merged_ends = _merge_intervals(test_starts_ns, test_ends_ns)

    logger.info(
        "step=purged_overlap.prep n_samples=%d n_test_samples=%d n_merged_test_intervals=%d",
        int(len(starts_ns)),
        int(len(test_starts_ns)),
        int(len(merged_starts)),
    )

    right_idx = np.searchsorted(merged_starts, ends_ns, side="right") - 1
    valid = right_idx >= 0
    overlaps = np.zeros(len(starts_ns), dtype=bool)
    candidate_idx = right_idx[valid]
    overlaps[valid] = merged_ends[candidate_idx] >= starts_ns[valid]
    return np.flatnonzero(overlaps).astype(int)


def _build_purged_fold(
    *,
    frame: pd.DataFrame,
    fold_id: int,
    test_mask: np.ndarray,
    unique_dates: pd.Index,
    embargo_periods: int,
) -> PurgedFold:
    _assert_boolean_mask_length(
        step="purged_kfold_split._build_purged_fold.test_mask",
        length=len(test_mask),
    )
    test_indices = np.flatnonzero(test_mask).astype(int)
    if len(test_indices) == 0:
        raise ValueError("test fold is empty; check split configuration")
    train_candidate = ~test_mask

    _assert_boolean_mask_length(
        step="purged_kfold_split._build_purged_fold.train_candidate",
        length=len(train_candidate),
    )
    test_df = frame.iloc[test_indices]
    train_df = frame.iloc[np.flatnonzero(train_candidate).astype(int)]
    logger.info(
        "step=purged_kfold_split.fold_input fold_id=%d n_total=%d n_train_candidate=%d n_test=%d",
        fold_id,
        int(len(frame)),
        int(len(train_df)),
        int(len(test_df)),
    )
    purge_rel_idx = overlapping_index(
        train_df,
        test_df,
        start_col="event_start",
        end_col="event_end",
    )
    train_candidate_indices = np.flatnonzero(train_candidate).astype(int)
    purged_indices = train_candidate_indices[purge_rel_idx]
    _assert_boolean_mask_length(
        step="purged_kfold_split._build_purged_fold.purged_mask",
        length=len(frame),
    )
    purged_mask = np.zeros(len(frame), dtype=bool)
    purged_mask[purged_indices] = True

    test_end_date = pd.Timestamp(test_df["decision_date"].max())
    embargo_dates = _embargo_dates(
        unique_dates,
        test_end_date=test_end_date,
        embargo_periods=embargo_periods,
    )
    _assert_boolean_mask_length(
        step="purged_kfold_split._build_purged_fold.embargo_mask",
        length=len(frame),
    )
    embargo_mask = frame["decision_date"].isin(embargo_dates).to_numpy(dtype=bool)
    embargoed_indices = np.flatnonzero(
        train_candidate & (~purged_mask) & embargo_mask
    ).astype(int)

    final_train_mask = train_candidate & (~purged_mask) & (~embargo_mask)
    train_indices = np.flatnonzero(final_train_mask).astype(int)
    logger.info(
        "step=purged_kfold_split.fold_output "
        "fold_id=%d n_train=%d n_test=%d n_purged=%d n_embargoed=%d",
        fold_id,
        int(len(train_indices)),
        int(len(test_indices)),
        int(len(purged_indices)),
        int(len(embargoed_indices)),
    )

    return PurgedFold(
        fold_id=fold_id,
        train_indices=train_indices,
        test_indices=test_indices,
        purged_indices=purged_indices,
        embargoed_indices=embargoed_indices,
        test_start=pd.Timestamp(test_df["decision_date"].min()),
        test_end=pd.Timestamp(test_df["decision_date"].max()),
    )


def _embargo_dates(
    unique_dates: pd.Index,
    *,
    test_end_date: pd.Timestamp,
    embargo_periods: int,
) -> pd.Index:
    if embargo_periods <= 0:
        return pd.Index([])
    ordered = pd.Index(pd.to_datetime(unique_dates, errors="coerce")).sort_values()
    end_pos = int(ordered.get_indexer([test_end_date])[0])
    if end_pos < 0:
        return pd.Index([])
    start = end_pos + 1
    stop = min(start + embargo_periods, len(ordered))
    return pd.Index(ordered[start:stop])


def _prepare_intervals(
    samples: pd.DataFrame,
    *,
    decision_col: str,
    start_col: str,
    end_col: str,
) -> pd.DataFrame:
    if decision_col not in samples.columns:
        raise ValueError(f"samples missing decision_col {decision_col!r}")
    frame = samples.copy().reset_index(drop=True)
    frame["decision_date"] = pd.to_datetime(frame[decision_col], errors="coerce")
    if frame["decision_date"].isna().any():
        raise ValueError("samples contains invalid decision dates")

    if start_col in frame.columns:
        frame["event_start"] = pd.to_datetime(frame[start_col], errors="coerce")
    else:
        frame["event_start"] = frame["decision_date"]

    if end_col in frame.columns:
        frame["event_end"] = pd.to_datetime(frame[end_col], errors="coerce")
    else:
        frame["event_end"] = frame["decision_date"]

    if frame["event_start"].isna().any() or frame["event_end"].isna().any():
        raise ValueError("samples contains invalid event interval timestamps")
    if (frame["event_end"] < frame["event_start"]).any():
        raise ValueError("samples has event_end < event_start")

    return frame.sort_values(
        ["decision_date", "event_start", "event_end"],
        kind="mergesort",
    ).reset_index(drop=True)


def _merge_intervals(
    starts_ns: np.ndarray,
    ends_ns: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if len(starts_ns) == 0:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64)
    order = np.argsort(starts_ns, kind="mergesort")
    sorted_starts = starts_ns[order]
    sorted_ends = ends_ns[order]

    merged_starts: list[int] = [int(sorted_starts[0])]
    merged_ends: list[int] = [int(sorted_ends[0])]
    for start, end in zip(sorted_starts[1:], sorted_ends[1:], strict=False):
        start_i = int(start)
        end_i = int(end)
        if start_i <= merged_ends[-1]:
            if end_i > merged_ends[-1]:
                merged_ends[-1] = end_i
        else:
            merged_starts.append(start_i)
            merged_ends.append(end_i)
    return np.asarray(merged_starts, dtype=np.int64), np.asarray(merged_ends, dtype=np.int64)


def _assert_boolean_mask_length(*, step: str, length: int) -> None:
    if length > _MAX_BOOLEAN_MASK_LENGTH:
        raise ValueError(
            f"{step}: boolean mask length {length} exceeds safety limit "
            f"{_MAX_BOOLEAN_MASK_LENGTH}. This indicates an unexpected broadcast "
            "or panel-shape explosion."
        )


def _assert_max_cardinality(
    *,
    step: str,
    unique_dates: int,
    unique_assets: int | None,
) -> None:
    if unique_dates > _MAX_DECISION_DATES:
        raise ValueError(
            f"{step}: unique decision dates={unique_dates} exceeds safety limit "
            f"{_MAX_DECISION_DATES}."
        )
    if unique_assets is not None and unique_assets > _MAX_ASSETS:
        raise ValueError(
            f"{step}: unique assets={unique_assets} exceeds safety limit "
            f"{_MAX_ASSETS}."
        )


def _log_shape(*, step: str, frame: pd.DataFrame) -> None:
    n_rows = int(len(frame))
    n_dates = int(frame["decision_date"].nunique()) if "decision_date" in frame.columns else int(
        frame["date"].nunique() if "date" in frame.columns else 0
    )
    n_assets = int(frame["asset"].nunique()) if "asset" in frame.columns else -1
    logger.info(
        "step=%s n_rows=%d n_dates=%d n_assets=%s",
        step,
        n_rows,
        n_dates,
        "NA" if n_assets < 0 else str(n_assets),
    )
