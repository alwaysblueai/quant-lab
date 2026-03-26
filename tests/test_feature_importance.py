from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from alpha_lab.feature_importance import (
    build_feature_importance_report,
    correlation_clusters,
    mda_importance,
    mdi_importance,
    sfi_importance,
)


def _dataset(seed: int = 7) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    n = 240
    x1 = rng.normal(0, 1, size=n)
    x2 = rng.normal(0, 1, size=n)
    x3 = rng.normal(0, 1, size=n)
    y = 2.5 * x1 + 0.1 * x2 + rng.normal(0, 0.5, size=n)
    X = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})
    return X, pd.Series(y, name="y")


def test_mdi_importance_identifies_primary_feature() -> None:
    X, y = _dataset()
    model = RandomForestRegressor(n_estimators=80, random_state=0)
    model.fit(X, y)
    out = mdi_importance(model, X.columns.tolist())
    assert out.iloc[0]["feature"] == "x1"


def test_mda_and_sfi_rank_primary_feature_higher() -> None:
    X, y = _dataset()
    est = RandomForestRegressor(n_estimators=60, random_state=0)
    mda = mda_importance(est, X, y)
    sfi = sfi_importance(lambda: LinearRegression(), X, y)
    top_mda = mda.sort_values("mda_importance_mean", ascending=False).iloc[0]["feature"]
    top_sfi = sfi.sort_values("sfi_score_mean", ascending=False).iloc[0]["feature"]
    assert top_mda == "x1"
    assert top_sfi == "x1"


def test_correlation_clusters_groups_highly_correlated_features() -> None:
    X, _ = _dataset()
    X = X.copy()
    X["x1_dup"] = X["x1"] + 1e-6
    clusters = correlation_clusters(X, threshold=0.95)
    cid = clusters.set_index("feature").loc["x1", "cluster_id"]
    assert clusters.set_index("feature").loc["x1_dup", "cluster_id"] == cid


def test_build_feature_importance_report_sections() -> None:
    X, y = _dataset()
    fitted = RandomForestRegressor(n_estimators=50, random_state=0).fit(X, y)
    report = build_feature_importance_report(
        fitted_model=fitted,
        estimator_for_mda=RandomForestRegressor(n_estimators=40, random_state=1),
        estimator_factory_for_sfi=lambda: LinearRegression(),
        X=X,
        y=y,
    )
    assert len(report.mdi) == X.shape[1]
    assert len(report.mda) == X.shape[1]
    assert len(report.sfi) == X.shape[1]
    assert len(report.clusters) == X.shape[1]

