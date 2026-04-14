from __future__ import annotations

import pandas as pd
import pytest

from alpha_lab.model_factor import ModelFactorBuildConfig, build_model_factor
from alpha_lab.real_cases.model_factor.spec import load_model_factor_case_spec
from tests.model_factor_case_helpers import write_demo_model_factor_case


def test_build_model_factor_uses_past_only_training_windows(tmp_path) -> None:
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

    assert set(result.factor_df.columns) == {"date", "asset", "factor", "value"}
    scored = result.training_log_df[result.training_log_df["status"] != "skipped"].copy()
    assert not scored.empty
    assert (pd.to_datetime(scored["trained_date_end"]) < pd.to_datetime(scored["score_date"])).all()
    assert result.feature_importance_df["feature"].tolist() == sorted(spec.feature_columns)


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
