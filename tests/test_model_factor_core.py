from __future__ import annotations

from typing import get_args

import pandas as pd
import pytest

from alpha_lab.model_factor import (
    FeatureImportanceConfig,
    ModelFactorBuildConfig,
    build_model_factor,
)
from alpha_lab.model_factor.core import (
    FeaturePreprocessConfig,
    ModelFamily,
    ModelSelectionSpec,
    ModelSpec,
    TrainingSpec,
    _build_estimator,
    _feature_importance_extractors_for_family,
    _normalize_features,
    _prepare_training_matrix,
)
from alpha_lab.model_factor.diagnostics import ModelFactorDiagnosticsRecorder
from alpha_lab.real_cases.model_factor.spec import load_model_factor_case_spec
from tests.model_factor_case_helpers import write_demo_model_factor_case


def test_build_model_factor_uses_past_only_training_windows(tmp_path) -> None:
    spec_path = write_demo_model_factor_case(tmp_path, factor_name="ml_score")
    spec = load_model_factor_case_spec(spec_path)

    prices = pd.read_csv(spec.prices_path)
    features = pd.read_csv(spec.features_path)

    progress_events: list[tuple[str, int]] = []

    result = build_model_factor(
        features,
        prices,
        ModelFactorBuildConfig(
            factor_name=spec.factor_name,
            feature_columns=spec.feature_columns,
            target_horizon=spec.target.horizon,
            feature_preprocess=spec.feature_preprocess,
            model=spec.model,
            training=spec.training,
            known_at_col="known_at",
        ),
        progress_callback=lambda message, percent: progress_events.append((message, percent)),
    )

    assert set(result.factor_df.columns) == {"date", "asset", "factor", "value"}
    scored = result.training_log_df[result.training_log_df["status"] != "skipped"].copy()
    assert not scored.empty
    assert (pd.to_datetime(scored["trained_date_end"]) < pd.to_datetime(scored["score_date"])).all()
    assert result.feature_importance_df["feature"].tolist() == sorted(spec.feature_columns)
    assert set(result.forward_label_df.columns) == {"date", "asset", "factor", "value"}
    assert result.forward_label_df["factor"].eq(f"forward_return_{spec.target.horizon}").all()
    assert any("训练模型生成因子：第 " in message for message, _ in progress_events)
    assert progress_events[-1][1] == 100


def test_build_model_factor_batches_predictions_for_reused_model_version(tmp_path) -> None:
    spec_path = write_demo_model_factor_case(tmp_path, factor_name="ml_score")
    spec = load_model_factor_case_spec(spec_path)

    prices = pd.read_csv(spec.prices_path)
    features = pd.read_csv(spec.features_path)
    observer = ModelFactorDiagnosticsRecorder()

    result = build_model_factor(
        features,
        prices,
        ModelFactorBuildConfig(
            factor_name=spec.factor_name,
            feature_columns=spec.feature_columns,
            target_horizon=spec.target.horizon,
            feature_preprocess=spec.feature_preprocess,
            model=spec.model,
            training=TrainingSpec(
                window_type="rolling",
                train_window_n_dates=45,
                min_train_dates=20,
                min_train_rows=120,
                retrain_every_n_dates=999,
                min_score_assets=5,
            ),
            known_at_col="known_at",
        ),
        observer=observer,
    )

    scored = result.training_log_df[result.training_log_df["status"] != "skipped"]
    assert len(scored) > 1
    assert int((result.training_log_df["status"] == "fit_scored").sum()) == 1
    assert int((result.training_log_df["status"] == "reused_scored").sum()) > 0

    payload = observer.build_payload(run_meta={"case_name": spec.name})
    predict_stages = [
        item for item in payload["stages"] if str(item.get("name")) == "predict"
    ]
    split_stages = [item for item in payload["stages"] if str(item.get("name")) == "split"]
    window_index_stages = [
        item for item in payload["stages"] if str(item.get("name")) == "training_window_index"
    ]
    assert not split_stages
    assert len(window_index_stages) == 1
    assert window_index_stages[0]["result"]["row_index_cache_mode"] == (
        "lazy_materialize_fit_windows"
    )
    assert len(predict_stages) == 1
    predict_result = predict_stages[0]["result"]
    assert predict_result["n_score_dates"] == int(len(scored))
    assert predict_result["n_score_rows"] == int(len(result.factor_df))
    assert set(predict_result["statuses"]) == {"fit_scored", "reused_scored"}


def test_build_model_factor_defaults_feature_importance_to_latest_fit_only(tmp_path) -> None:
    spec_path = write_demo_model_factor_case(tmp_path, factor_name="ml_score")
    spec = load_model_factor_case_spec(spec_path)

    prices = pd.read_csv(spec.prices_path)
    features = pd.read_csv(spec.features_path)

    result = build_model_factor(
        features,
        prices,
        ModelFactorBuildConfig(
            factor_name=spec.factor_name,
            feature_columns=spec.feature_columns,
            target_horizon=spec.target.horizon,
            feature_preprocess=spec.feature_preprocess,
            model=spec.model,
            training=spec.training,
            known_at_col="known_at",
        ),
    )

    fit_count = int((result.training_log_df["status"] == "fit_scored").sum())
    assert fit_count > 1
    assert result.model_diagnostics["trained_model_versions"] == fit_count
    assert result.model_diagnostics["feature_importance"]["mode"] == "latest_only"
    assert result.model_diagnostics["feature_importance"]["n_importance_model_versions"] == 1
    assert result.feature_importance_df["n_model_versions"].eq(1).all()


def test_build_model_factor_can_disable_feature_importance(tmp_path) -> None:
    spec_path = write_demo_model_factor_case(tmp_path, factor_name="ml_score")
    spec = load_model_factor_case_spec(spec_path)

    prices = pd.read_csv(spec.prices_path)
    features = pd.read_csv(spec.features_path)

    result = build_model_factor(
        features,
        prices,
        ModelFactorBuildConfig(
            factor_name=spec.factor_name,
            feature_columns=spec.feature_columns,
            target_horizon=spec.target.horizon,
            feature_preprocess=spec.feature_preprocess,
            feature_importance=FeatureImportanceConfig(mode="disabled"),
            model=spec.model,
            training=spec.training,
            known_at_col="known_at",
        ),
    )

    assert result.model_diagnostics["feature_importance"]["mode"] == "disabled"
    assert result.model_diagnostics["top_features"] == []
    assert result.feature_importance_df["importance_source"].eq("disabled").all()
    assert result.feature_importance_df["n_model_versions"].eq(0).all()
    assert pd.to_numeric(
        result.feature_importance_df["mean_abs_importance"],
        errors="coerce",
    ).isna().all()


def test_build_model_factor_fails_on_future_known_at_feature(tmp_path) -> None:
    spec_path = write_demo_model_factor_case(tmp_path, factor_name="ml_score")
    spec = load_model_factor_case_spec(spec_path)

    prices = pd.read_csv(spec.prices_path)
    features = pd.read_csv(spec.features_path)
    features["date"] = pd.to_datetime(features["date"], errors="coerce")
    features["known_at"] = pd.to_datetime(features["known_at"], errors="coerce")
    features.loc[0, "known_at"] = features.loc[0, "date"] + pd.Timedelta(days=1)

    with pytest.raises(ValueError):
        build_model_factor(
            features,
            prices,
            ModelFactorBuildConfig(
                factor_name=spec.factor_name,
                feature_columns=spec.feature_columns,
                target_horizon=spec.target.horizon,
                feature_preprocess=spec.feature_preprocess,
                model=spec.model,
                training=spec.training,
                known_at_col="known_at",
            ),
        )


def test_model_spec_accepts_optional_tree_families() -> None:
    assert ModelSpec(family="xgboost").family == "xgboost"
    assert ModelSpec(family="lightgbm").family == "lightgbm"


def test_feature_preprocess_defaults_use_cross_sectional_winsorize_zscore() -> None:
    preprocess = FeaturePreprocessConfig()
    assert preprocess.cross_sectional_transform == "winsorize_zscore"


def test_build_estimator_gbdt_uses_robust_finance_defaults() -> None:
    estimator = _build_estimator(ModelSpec(family="gbdt"))
    assert getattr(estimator, "loss", None) == "absolute_error"
    assert getattr(estimator, "min_samples_leaf", None) == 200


def test_feature_importance_registry_covers_all_model_families() -> None:
    families = get_args(ModelFamily)
    assert families
    for family in families:
        extractors = _feature_importance_extractors_for_family(family)
        assert extractors


def test_prepare_training_matrix_uses_tree_float32_and_linear_float64() -> None:
    frame = pd.DataFrame({"f1": [1.0, 2.0, 3.0], "f2": [4.0, 5.0, 6.0]})

    tree_matrix = _prepare_training_matrix(
        frame,
        feature_columns=("f1", "f2"),
        model_family="lightgbm",
    )
    linear_matrix = _prepare_training_matrix(
        frame,
        feature_columns=("f1", "f2"),
        model_family="ridge",
    )

    assert str(tree_matrix["f1"].dtype) == "float32"
    assert str(tree_matrix["f2"].dtype) == "float32"
    assert str(linear_matrix["f1"].dtype) == "float64"
    assert str(linear_matrix["f2"].dtype) == "float64"


def test_normalize_features_casts_asset_and_industry_to_category() -> None:
    features = pd.DataFrame(
        {
            "date": ["2025-01-01", "2025-01-01", "2025-01-02", "2025-01-02"],
            "asset": ["A1", "A2", "A1", "A2"],
            "f1": [1.0, 2.0, 3.0, 4.0],
            "industry_group": ["IND_A", "IND_B", "IND_A", "IND_B"],
        }
    )
    config = ModelFactorBuildConfig(
        factor_name="ml_score",
        feature_columns=("f1",),
        target_horizon=1,
        feature_preprocess=FeaturePreprocessConfig(
            cross_sectional_group_scope="date_and_industry",
            industry_group_column="industry_group",
        ),
    )

    normalized = _normalize_features(features, config=config)

    assert isinstance(normalized["asset"].dtype, pd.CategoricalDtype)
    assert isinstance(normalized["industry_group"].dtype, pd.CategoricalDtype)


@pytest.mark.parametrize(
    ("family", "params"),
    [
        ("gbdt", {"max_iter": 50, "max_leaf_nodes": 15}),
        ("mlp", {"max_iter": 50, "hidden_layer_sizes": (8,), "early_stopping": False}),
    ],
)
def test_build_model_factor_uses_permutation_importance_fallback_for_gbdt_and_mlp(
    tmp_path,
    family: str,
    params: dict[str, object],
) -> None:
    spec_path = write_demo_model_factor_case(tmp_path, factor_name="ml_score")
    spec = load_model_factor_case_spec(spec_path)
    prices = pd.read_csv(spec.prices_path)
    features = pd.read_csv(spec.features_path)

    result = build_model_factor(
        features,
        prices,
        ModelFactorBuildConfig(
            factor_name=spec.factor_name,
            feature_columns=spec.feature_columns,
            target_horizon=spec.target.horizon,
            feature_preprocess=spec.feature_preprocess,
            model=ModelSpec(family=family, params=params),
            training=spec.training,
            known_at_col="known_at",
        ),
    )

    feature_importance_df = result.feature_importance_df
    assert not feature_importance_df.empty
    assert (feature_importance_df["importance_source"] == "permutation").all()
    mean_importance = pd.to_numeric(feature_importance_df["mean_abs_importance"], errors="coerce")
    latest_importance = pd.to_numeric(feature_importance_df["latest_importance"], errors="coerce")
    assert mean_importance.notna().all()
    assert latest_importance.notna().all()


@pytest.mark.parametrize(
    ("family", "expected_class", "dependency_name"),
    [
        ("xgboost", "XGBRegressor", "xgboost"),
        ("lightgbm", "LGBMRegressor", "lightgbm"),
    ],
)
def test_build_estimator_optional_tree_families_or_dependency_hint(
    family: str,
    expected_class: str,
    dependency_name: str,
) -> None:
    spec = ModelSpec(family=family)
    try:
        estimator = _build_estimator(spec)
    except RuntimeError as exc:
        message = str(exc)
        assert dependency_name in message
        assert "uv add" in message
    else:
        assert estimator.__class__.__name__ == expected_class


def test_build_model_factor_records_inner_purged_cv_model_selection(tmp_path) -> None:
    spec_path = write_demo_model_factor_case(tmp_path, factor_name="ml_score")
    spec = load_model_factor_case_spec(spec_path)

    prices = pd.read_csv(spec.prices_path)
    features = pd.read_csv(spec.features_path)

    result = build_model_factor(
        features,
        prices,
        ModelFactorBuildConfig(
            factor_name=spec.factor_name,
            feature_columns=spec.feature_columns,
            target_horizon=spec.target.horizon,
            feature_preprocess=spec.feature_preprocess,
            model=spec.model,
            model_selection=ModelSelectionSpec(
                enabled=True,
                n_splits=3,
                embargo_pct=0.0,
                metric="rank_ic",
                candidates=(
                    ModelSpec(family="ridge", params={"alpha": 1.0}),
                    ModelSpec(family="ridge", params={"alpha": 10.0}),
                ),
            ),
            training=spec.training,
            known_at_col="known_at",
        ),
    )

    assert not result.model_selection_df.empty
    selected = result.model_selection_df[result.model_selection_df["selected"]].copy()
    assert not selected.empty
    assert selected.groupby("score_date").size().eq(1).all()
    selection_diag = result.model_diagnostics["model_selection"]
    assert selection_diag["enabled"] is True
    assert selection_diag["candidate_count"] == 2
    assert selection_diag["n_selection_events"] >= 1


def test_build_model_factor_warns_when_industry_group_is_static(tmp_path) -> None:
    spec_path = write_demo_model_factor_case(tmp_path, factor_name="ml_score")
    spec = load_model_factor_case_spec(spec_path)
    prices = pd.read_csv(spec.prices_path)
    features = pd.read_csv(spec.features_path)
    features["industry_group"] = features["asset"].map(
        lambda asset: "IND_A" if int(str(asset).replace("A", "")) % 2 == 0 else "IND_B"
    )

    observer = ModelFactorDiagnosticsRecorder()
    _ = build_model_factor(
        features,
        prices,
        ModelFactorBuildConfig(
            factor_name=spec.factor_name,
            feature_columns=spec.feature_columns,
            target_horizon=spec.target.horizon,
            feature_preprocess=FeaturePreprocessConfig(
                cross_sectional_group_scope="date_and_industry",
                industry_group_column="industry_group",
            ),
            model=spec.model,
            training=spec.training,
            known_at_col="known_at",
        ),
        observer=observer,
    )
    payload = observer.build_payload(run_meta={"case_name": spec.name})
    warnings = payload.get("warnings", [])
    assert isinstance(warnings, list)
    assert any("行业分组列疑似静态映射" in str(item.get("title")) for item in warnings)


def test_build_model_factor_warns_for_mlp_early_stopping_limitation(tmp_path) -> None:
    spec_path = write_demo_model_factor_case(tmp_path, factor_name="ml_score")
    spec = load_model_factor_case_spec(spec_path)
    prices = pd.read_csv(spec.prices_path)
    features = pd.read_csv(spec.features_path)

    observer = ModelFactorDiagnosticsRecorder()
    _ = build_model_factor(
        features,
        prices,
        ModelFactorBuildConfig(
            factor_name=spec.factor_name,
            feature_columns=spec.feature_columns,
            target_horizon=spec.target.horizon,
            feature_preprocess=spec.feature_preprocess,
            model=ModelSpec(
                family="mlp",
                params={"max_iter": 5, "hidden_layer_sizes": (8,), "early_stopping": True},
            ),
            training=TrainingSpec(
                window_type="rolling",
                train_window_n_dates=30,
                min_train_dates=20,
                min_train_rows=120,
                retrain_every_n_dates=999,
                min_score_assets=5,
            ),
            known_at_col="known_at",
        ),
        observer=observer,
    )
    payload = observer.build_payload(run_meta={"case_name": spec.name})
    warnings = payload.get("warnings", [])
    assert isinstance(warnings, list)
    assert any("MLP 早停验证存在时序局限" in str(item.get("title")) for item in warnings)
