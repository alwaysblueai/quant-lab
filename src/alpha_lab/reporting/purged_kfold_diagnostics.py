from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from alpha_lab.experiment import ExperimentResult
from alpha_lab.validation.purged_kfold import purged_kfold_split

PURGED_KFOLD_FOLDS_COLUMNS: tuple[str, ...] = (
    "fold_id",
    "train_start",
    "train_end",
    "test_start",
    "test_end",
    "n_train_dates",
    "n_test_dates",
    "mean_ic",
    "mean_rank_ic",
    "long_short_sharpe",
    "mean_long_short_return",
)


@dataclass(frozen=True)
class PurgedKFoldDiagnosticsResult:
    summary: dict[str, object]
    folds: pd.DataFrame


def build_purged_kfold_diagnostics(
    *,
    experiment_result: ExperimentResult,
    label_horizon: int,
    n_splits: int = 5,
    embargo_pct: float = 0.01,
) -> PurgedKFoldDiagnosticsResult:
    if label_horizon < 0:
        raise ValueError("label_horizon must be >= 0")
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    if embargo_pct < 0.0 or embargo_pct >= 1.0:
        raise ValueError("embargo_pct must be in [0, 1)")

    date_axis = _resolve_eval_dates(experiment_result)
    if len(date_axis) < 2:
        return PurgedKFoldDiagnosticsResult(
            summary={
                "schema_version": "1.0.0",
                "artifact_type": "alpha_lab_purged_kfold_summary",
                "status": "not_available",
                "message": "insufficient evaluation dates for purged k-fold diagnostics",
                "n_eval_dates": int(len(date_axis)),
                "n_splits_requested": int(n_splits),
                "n_splits_used": 0,
                "label_horizon": int(label_horizon),
                "embargo_pct": float(embargo_pct),
                "embargo_days": 0,
                "purge_days": int(label_horizon),
                "n_folds": 0,
                "fold_metrics_available": 0,
                "mean_ic": None,
                "mean_rank_ic": None,
                "mean_sharpe": None,
                "ic_positive_folds": 0,
                "rank_ic_positive_folds": 0,
                "verdict": "not_available",
                "reasons": ["insufficient evaluation dates"],
            },
            folds=pd.DataFrame(columns=PURGED_KFOLD_FOLDS_COLUMNS),
        )

    n_splits_used = min(int(n_splits), int(len(date_axis)))
    split_rows = purged_kfold_split(
        date_axis.to_numpy(),
        n_splits=n_splits_used,
        label_horizon=int(label_horizon),
        embargo_pct=float(embargo_pct),
    )
    embargo_days = int(math.ceil(len(date_axis) * float(embargo_pct)))

    ic_series = _series_by_date(experiment_result.ic_df, "ic")
    rank_ic_series = _series_by_date(experiment_result.rank_ic_df, "rank_ic")
    long_short_series = _series_by_date(
        experiment_result.long_short_df,
        "long_short_return",
    )

    folds_rows: list[dict[str, object]] = []
    for fold_idx, masks in enumerate(split_rows, start=1):
        train_dates = date_axis[masks["train"]]
        test_dates = date_axis[masks["test"]]

        ic_values = ic_series.reindex(test_dates).dropna()
        rank_ic_values = rank_ic_series.reindex(test_dates).dropna()
        long_short_values = long_short_series.reindex(test_dates).dropna()

        mean_ic = _mean_or_none(ic_values)
        mean_rank_ic = _mean_or_none(rank_ic_values)
        mean_long_short = _mean_or_none(long_short_values)
        long_short_sharpe = _sharpe_or_none(long_short_values)

        folds_rows.append(
            {
                "fold_id": fold_idx,
                "train_start": _first_date_or_none(train_dates),
                "train_end": _last_date_or_none(train_dates),
                "test_start": _first_date_or_none(test_dates),
                "test_end": _last_date_or_none(test_dates),
                "n_train_dates": int(len(train_dates)),
                "n_test_dates": int(len(test_dates)),
                "mean_ic": mean_ic,
                "mean_rank_ic": mean_rank_ic,
                "long_short_sharpe": long_short_sharpe,
                "mean_long_short_return": mean_long_short,
            }
        )

    folds_df = pd.DataFrame(folds_rows, columns=PURGED_KFOLD_FOLDS_COLUMNS)
    validate_purged_kfold_folds_frame(folds_df)

    mean_ic = _mean_or_none(folds_df["mean_ic"])
    mean_rank_ic = _mean_or_none(folds_df["mean_rank_ic"])
    mean_sharpe = _mean_or_none(folds_df["long_short_sharpe"])
    ic_positive_folds = int((pd.to_numeric(folds_df["mean_ic"], errors="coerce") > 0).sum())
    rank_ic_positive_folds = int(
        (pd.to_numeric(folds_df["mean_rank_ic"], errors="coerce") > 0).sum()
    )
    fold_metrics_available = int(
        pd.to_numeric(folds_df["mean_ic"], errors="coerce").notna().sum()
    )
    verdict, reasons = _build_verdict(
        n_folds=int(len(folds_df)),
        ic_positive_folds=ic_positive_folds,
        rank_ic_positive_folds=rank_ic_positive_folds,
        mean_sharpe=mean_sharpe,
    )

    summary = {
        "schema_version": "1.0.0",
        "artifact_type": "alpha_lab_purged_kfold_summary",
        "status": "ok",
        "message": "",
        "n_eval_dates": int(len(date_axis)),
        "n_splits_requested": int(n_splits),
        "n_splits_used": int(n_splits_used),
        "label_horizon": int(label_horizon),
        "embargo_pct": float(embargo_pct),
        "embargo_days": int(embargo_days),
        "purge_days": int(label_horizon),
        "n_folds": int(len(folds_df)),
        "fold_metrics_available": int(fold_metrics_available),
        "mean_ic": mean_ic,
        "mean_rank_ic": mean_rank_ic,
        "mean_sharpe": mean_sharpe,
        "ic_positive_folds": ic_positive_folds,
        "rank_ic_positive_folds": rank_ic_positive_folds,
        "verdict": verdict,
        "reasons": reasons,
    }
    return PurgedKFoldDiagnosticsResult(summary=summary, folds=folds_df)


def validate_purged_kfold_folds_frame(frame: pd.DataFrame) -> None:
    missing = [name for name in PURGED_KFOLD_FOLDS_COLUMNS if name not in frame.columns]
    if missing:
        raise ValueError(f"purged k-fold folds frame missing columns: {missing}")


def _series_by_date(frame: pd.DataFrame, value_col: str) -> pd.Series:
    if frame.empty or "date" not in frame.columns or value_col not in frame.columns:
        return pd.Series(dtype=float)
    scoped = frame[["date", value_col]].copy()
    scoped["date"] = pd.to_datetime(scoped["date"], errors="coerce")
    scoped[value_col] = pd.to_numeric(scoped[value_col], errors="coerce")
    scoped = scoped.dropna(subset=["date"])
    if scoped.empty:
        return pd.Series(dtype=float)
    grouped = scoped.groupby("date", sort=True)[value_col].mean()
    grouped.index = pd.DatetimeIndex(grouped.index)
    return grouped.sort_index()


def _resolve_eval_dates(result: ExperimentResult) -> pd.DatetimeIndex:
    sources = (
        result.rank_ic_df,
        result.ic_df,
        result.long_short_df,
    )
    for frame in sources:
        if frame is None or frame.empty or "date" not in frame.columns:
            continue
        dates = pd.to_datetime(frame["date"], errors="coerce").dropna()
        if len(dates) == 0:
            continue
        return pd.DatetimeIndex(sorted(dates.unique()))
    return pd.DatetimeIndex([])


def _mean_or_none(values: pd.Series) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if len(numeric) == 0:
        return None
    return float(numeric.mean())


def _sharpe_or_none(values: pd.Series) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if len(numeric) < 2:
        return None
    std = float(numeric.std(ddof=1))
    if not math.isfinite(std) or std <= 0.0:
        return None
    return float(numeric.mean() / std)


def _first_date_or_none(dates: pd.DatetimeIndex) -> str | None:
    if len(dates) == 0:
        return None
    return pd.Timestamp(dates.min()).date().isoformat()


def _last_date_or_none(dates: pd.DatetimeIndex) -> str | None:
    if len(dates) == 0:
        return None
    return pd.Timestamp(dates.max()).date().isoformat()


def _build_verdict(
    *,
    n_folds: int,
    ic_positive_folds: int,
    rank_ic_positive_folds: int,
    mean_sharpe: float | None,
) -> tuple[str, list[str]]:
    if n_folds <= 0:
        return "not_available", ["no folds were generated"]

    support_ratio = min(
        ic_positive_folds / float(n_folds),
        rank_ic_positive_folds / float(n_folds),
    )
    sharpe_supportive = mean_sharpe is not None and mean_sharpe > 0.0

    if support_ratio >= 0.60 and sharpe_supportive:
        return "robust", [
            "most folds show positive IC and RankIC",
            "mean fold long-short sharpe is positive",
        ]
    if support_ratio >= 0.40:
        return "mixed", [
            "fold-level signal quality is uneven across time",
            "review fold diagnostics before promotion",
        ]
    return "weak", [
        "too few folds show positive IC and RankIC",
        "out-of-sample consistency is weak",
    ]
