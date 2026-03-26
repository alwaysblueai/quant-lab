from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold


@dataclass(frozen=True)
class FeatureImportanceReport:
    """Combined feature-importance views for research diagnostics."""

    mdi: pd.DataFrame
    mda: pd.DataFrame
    sfi: pd.DataFrame
    clusters: pd.DataFrame
    cluster_importance: pd.DataFrame


def mdi_importance(model: Any, feature_names: Sequence[str]) -> pd.DataFrame:
    """Model-dependent impurity-based importance (MDI) when available."""
    if not hasattr(model, "feature_importances_"):
        raise ValueError("model does not expose feature_importances_ for MDI")
    vals = np.asarray(model.feature_importances_, dtype=float)  # type: ignore[attr-defined]
    if len(vals) != len(feature_names):
        raise ValueError("feature_importances_ length does not match feature_names")
    out = pd.DataFrame({"feature": list(feature_names), "mdi_importance": vals})
    return out.sort_values(
        "mdi_importance",
        ascending=False,
        kind="mergesort",
    ).reset_index(drop=True)


def mda_importance(
    estimator: Any,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    splits: Sequence[tuple[np.ndarray, np.ndarray]] | None = None,
    scorer: Callable[[np.ndarray, np.ndarray], float] | None = None,
    random_state: int = 0,
) -> pd.DataFrame:
    """Permutation importance across user-provided folds (MDA)."""
    _validate_xy(X, y)
    resolved_splits = _resolve_splits(X, splits)
    rng = np.random.default_rng(random_state)

    drops: dict[str, list[float]] = {col: [] for col in X.columns}
    for train_idx, test_idx in resolved_splits:
        model = clone(estimator)
        x_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        x_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]
        model.fit(x_train, y_train)
        baseline = _score(model, x_test, y_test, scorer=scorer)
        for col in X.columns:
            x_perm = x_test.copy()
            x_perm[col] = rng.permutation(x_perm[col].to_numpy())
            perm_score = _score(model, x_perm, y_test, scorer=scorer)
            drops[col].append(float(baseline - perm_score))

    rows = []
    for col, vals in drops.items():
        rows.append(
            {
                "feature": col,
                "mda_importance_mean": float(np.mean(vals)),
                "mda_importance_std": (
                    float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")
                ),
                "n_folds": int(len(vals)),
            }
        )
    return pd.DataFrame(rows).sort_values(
        "mda_importance_mean",
        ascending=False,
        kind="mergesort",
    ).reset_index(drop=True)


def sfi_importance(
    estimator_factory: Callable[[], Any],
    X: pd.DataFrame,
    y: pd.Series,
    *,
    splits: Sequence[tuple[np.ndarray, np.ndarray]] | None = None,
    scorer: Callable[[np.ndarray, np.ndarray], float] | None = None,
) -> pd.DataFrame:
    """Single-feature importance (SFI): model score using one feature at a time."""
    _validate_xy(X, y)
    resolved_splits = _resolve_splits(X, splits)

    rows: list[dict[str, object]] = []
    for col in X.columns:
        scores: list[float] = []
        for train_idx, test_idx in resolved_splits:
            model = estimator_factory()
            x_train = X.iloc[train_idx][[col]]
            y_train = y.iloc[train_idx]
            x_test = X.iloc[test_idx][[col]]
            y_test = y.iloc[test_idx]
            model.fit(x_train, y_train)
            scores.append(_score(model, x_test, y_test, scorer=scorer))
        rows.append(
            {
                "feature": col,
                "sfi_score_mean": float(np.mean(scores)),
                "sfi_score_std": (
                    float(np.std(scores, ddof=1)) if len(scores) > 1 else float("nan")
                ),
                "n_folds": int(len(scores)),
            }
        )
    return pd.DataFrame(rows).sort_values(
        "sfi_score_mean",
        ascending=False,
        kind="mergesort",
    ).reset_index(drop=True)


def correlation_clusters(
    X: pd.DataFrame,
    *,
    threshold: float = 0.8,
) -> pd.DataFrame:
    """Simple correlation-threshold connected-component clusters."""
    if threshold <= 0 or threshold > 1:
        raise ValueError("threshold must be in (0, 1]")
    if X.shape[1] == 0:
        return pd.DataFrame(columns=["feature", "cluster_id"])

    corr = X.corr(method="pearson").abs().fillna(0.0)
    features = corr.columns.tolist()
    visited: set[str] = set()
    rows: list[dict[str, object]] = []
    cluster_id = 0
    for feat in features:
        if feat in visited:
            continue
        stack = [feat]
        members: list[str] = []
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            members.append(cur)
            neigh = corr.index[corr.loc[cur] >= threshold].tolist()
            for nxt in neigh:
                if nxt not in visited:
                    stack.append(nxt)
        for m in sorted(members):
            rows.append({"feature": m, "cluster_id": cluster_id})
        cluster_id += 1
    return pd.DataFrame(rows).sort_values(
        ["cluster_id", "feature"],
        kind="mergesort",
    ).reset_index(drop=True)


def cluster_importance(
    importance: pd.DataFrame,
    clusters: pd.DataFrame,
    *,
    importance_col: str,
) -> pd.DataFrame:
    """Aggregate feature importance at correlation-cluster level."""
    if importance_col not in importance.columns:
        raise ValueError(f"importance missing importance_col {importance_col!r}")
    merged = clusters.merge(importance[["feature", importance_col]], on="feature", how="left")
    out = (
        merged.groupby("cluster_id", sort=True)
        .agg(
            n_features=("feature", "nunique"),
            importance_sum=(importance_col, "sum"),
            importance_mean=(importance_col, "mean"),
        )
        .reset_index()
    )
    return out.sort_values("cluster_id", kind="mergesort").reset_index(drop=True)


def build_feature_importance_report(
    fitted_model: Any,
    estimator_for_mda: Any,
    estimator_factory_for_sfi: Callable[[], Any],
    X: pd.DataFrame,
    y: pd.Series,
    *,
    splits: Sequence[tuple[np.ndarray, np.ndarray]] | None = None,
    scorer: Callable[[np.ndarray, np.ndarray], float] | None = None,
    cluster_threshold: float = 0.8,
) -> FeatureImportanceReport:
    """Build a combined MDI/MDA/SFI + cluster-importance report."""
    mdi = mdi_importance(fitted_model, X.columns.tolist())
    mda = mda_importance(estimator_for_mda, X, y, splits=splits, scorer=scorer)
    sfi = sfi_importance(estimator_factory_for_sfi, X, y, splits=splits, scorer=scorer)
    clusters = correlation_clusters(X, threshold=cluster_threshold)
    cluster_imp = cluster_importance(mda, clusters, importance_col="mda_importance_mean")
    return FeatureImportanceReport(
        mdi=mdi,
        mda=mda,
        sfi=sfi,
        clusters=clusters,
        cluster_importance=cluster_imp,
    )


def _resolve_splits(
    X: pd.DataFrame,
    splits: Sequence[tuple[np.ndarray, np.ndarray]] | None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if splits is not None:
        return [(np.asarray(tr, dtype=int), np.asarray(te, dtype=int)) for tr, te in splits]
    kf = KFold(n_splits=3, shuffle=False)
    idx = np.arange(len(X), dtype=int)
    return [(tr.astype(int), te.astype(int)) for tr, te in kf.split(idx)]


def _score(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    scorer: Callable[[np.ndarray, np.ndarray], float] | None,
) -> float:
    if scorer is None:
        if not hasattr(model, "score"):
            raise ValueError("model does not expose score() and no scorer was provided")
        return float(model.score(X_test, y_test))
    if not hasattr(model, "predict"):
        raise ValueError("model does not expose predict() for custom scorer")
    pred = model.predict(X_test)
    return float(scorer(y_test.to_numpy(), np.asarray(pred)))


def _validate_xy(X: pd.DataFrame, y: pd.Series) -> None:
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a DataFrame")
    if not isinstance(y, pd.Series):
        raise TypeError("y must be a Series")
    if len(X) != len(y):
        raise ValueError("X and y must have the same length")
    if X.empty:
        raise ValueError("X is empty")
