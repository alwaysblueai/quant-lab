from __future__ import annotations

import pandas as pd

from alpha_lab.model_factor import (
    FeaturePreprocessConfig,
)
from alpha_lab.neutralization import neutralize_signal
from alpha_lab.real_cases.common_io import (
    load_tabular_frame,
)
from alpha_lab.research_integrity.contracts import IntegrityCheckResult
from alpha_lab.research_integrity.exceptions import raise_on_hard_failures
from alpha_lab.research_integrity.leakage_checks import (
    check_asof_inputs_not_after_signal_date,
    check_no_future_dates_in_input,
)

from ..spec import (
    FeatureAvailabilitySpec,
    ModelFactorCaseSpec,
)


def _load_features(
    path_value: str,
    *,
    feature_columns: tuple[str, ...] = (),
    feature_availability: FeatureAvailabilitySpec | None = None,
    feature_preprocess: FeaturePreprocessConfig | None = None,
) -> pd.DataFrame:
    if not feature_columns and feature_availability is None and feature_preprocess is None:
        features = load_tabular_frame(path_value, object_name="features")
        return _normalize_loaded_features(features)

    required_columns: list[str] = ["date", "asset", *feature_columns]
    optional_columns: list[str] = []

    if feature_availability is not None:
        if feature_availability.mode == "required_timestamp":
            if feature_availability.column is not None:
                required_columns.append(feature_availability.column)
            else:
                optional_columns.extend(["known_at", "available_at"])
    else:
        optional_columns.extend(["known_at", "available_at"])

    if (
        feature_preprocess is not None
        and feature_preprocess.cross_sectional_group_scope == "date_and_industry"
        and feature_preprocess.industry_group_column is not None
    ):
        required_columns.append(feature_preprocess.industry_group_column)

    features = load_tabular_frame(
        path_value,
        object_name="features",
        columns=required_columns,
        optional_columns=optional_columns,
    )
    return _normalize_loaded_features(features)


def _normalize_loaded_features(features: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "asset"}
    missing = required - set(features.columns)
    if missing:
        raise ValueError(f"features is missing required columns: {sorted(missing)}")

    features = features.copy()
    features["date"] = pd.to_datetime(features["date"], errors="coerce")
    if "known_at" in features.columns:
        features["known_at"] = pd.to_datetime(features["known_at"], errors="coerce")
    if "available_at" in features.columns:
        features["available_at"] = pd.to_datetime(features["available_at"], errors="coerce")
    return features.sort_values(["date", "asset"], kind="mergesort").reset_index(drop=True)


def _maybe_neutralize_factor(
    factor_df: pd.DataFrame,
    *,
    spec: ModelFactorCaseSpec,
    universe_mask: pd.DataFrame | None,
    integrity_checks: list[IntegrityCheckResult] | None = None,
    max_price_date: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    if not spec.neutralization.enabled:
        return factor_df, None

    exposures_path = spec.neutralization.exposures_path
    if exposures_path is None:
        raise ValueError("neutralization.exposures_path is required when neutralization is enabled")

    required = {"date", "asset"}
    if spec.neutralization.size_col is not None:
        required.add(spec.neutralization.size_col)
    if spec.neutralization.industry_col is not None:
        required.add(spec.neutralization.industry_col)

    exposures = load_tabular_frame(
        exposures_path,
        object_name="neutralization exposure",
        columns=sorted(required),
        optional_columns=("known_at", "available_at"),
    )
    exposures["date"] = pd.to_datetime(exposures["date"], errors="coerce")

    missing = required - set(exposures.columns)
    if missing:
        raise ValueError(
            f"neutralization exposure file is missing required columns: {sorted(missing)}"
        )
    known_at_col = None
    if "known_at" in exposures.columns:
        known_at_col = "known_at"
    elif "available_at" in exposures.columns:
        known_at_col = "available_at"

    if integrity_checks is not None and max_price_date is not None:
        no_future_check = check_no_future_dates_in_input(
            exposures,
            max_allowed_date=max_price_date,
            date_col="date",
            object_name="model_factor_neutralization_exposures",
        )
        integrity_checks.append(no_future_check)
        raise_on_hard_failures((no_future_check,))

        asof_check = check_asof_inputs_not_after_signal_date(
            factor_df[["date", "asset"]],
            exposures,
            by=("asset",),
            signal_date_col="date",
            aux_effective_date_col="date",
            aux_known_at_col=known_at_col,
            object_name="model_factor_neutralization_exposures_asof",
        )
        integrity_checks.append(asof_check)
        raise_on_hard_failures((asof_check,))

    if universe_mask is not None:
        active = universe_mask[universe_mask["in_universe"]][["date", "asset"]]
        exposures = exposures.merge(
            active,
            on=["date", "asset"],
            how="inner",
            validate="many_to_one",
        )

    merged = factor_df[["date", "asset", "value"]].merge(
        exposures,
        on=["date", "asset"],
        how="left",
        validate="one_to_one",
    )

    size_col = spec.neutralization.size_col
    industry_col = spec.neutralization.industry_col

    if size_col is not None:
        merged["__size_input"] = merged[size_col]
        size_col = "__size_input"
    if industry_col is not None:
        merged["__industry_input"] = merged[industry_col]
        industry_col = "__industry_input"
    known_at_input = None
    if known_at_col is not None:
        merged["__known_at_input"] = pd.to_datetime(
            merged[known_at_col],
            errors="coerce",
        )
        known_at_input = "__known_at_input"

    cols = ["date", "asset", "value"]
    for col in (size_col, industry_col):
        if col is not None:
            cols.append(col)
    if known_at_input is not None:
        cols.append(known_at_input)

    neutralized = neutralize_signal(
        merged[cols].copy(),
        value_col="value",
        by="date",
        size_col=size_col,
        industry_col=industry_col,
        beta_col=None,
        min_obs=spec.neutralization.min_obs,
        ridge=spec.neutralization.ridge,
        output_col="value_neutralized",
        known_at_col=known_at_input,
        enforce_integrity=True,
    )
    if integrity_checks is not None:
        integrity_checks.extend(list(neutralized.integrity_checks))
        raise_on_hard_failures(neutralized.integrity_checks)

    out = factor_df[["date", "asset", "factor"]].copy()
    out = out.merge(
        neutralized.data[["date", "asset", "value_neutralized"]],
        on=["date", "asset"],
        how="left",
        validate="one_to_one",
    )
    out = out.rename(columns={"value_neutralized": "value"})
    return out, neutralized.diagnostics


def _coverage_by_date(
    factor_df: pd.DataFrame,
    *,
    coverage_base_df: pd.DataFrame | None = None,
    target_label_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    columns = [
        "date",
        "n_assets",
        "n_non_null",
        "coverage",
        "missingness",
        "universe_count",
        "feature_row_count",
        "complete_feature_count",
        "feature_nan_row_count",
        "label_available_count",
        "eligible_count",
        "scored_count",
        "scored_evaluable_count",
        "missing_feature_count",
        "missing_label_count",
        "missing_score_count",
        "filtered_count",
        "score_coverage",
        "universe_coverage",
    ]
    if factor_df.empty and (coverage_base_df is None or coverage_base_df.empty):
        return pd.DataFrame(columns=columns)

    scored = pd.DataFrame(columns=["date", "asset", "value"])
    if not factor_df.empty:
        scored = factor_df.loc[:, ["date", "asset", "value"]].copy()
        scored["date"] = pd.to_datetime(scored["date"], errors="coerce")
        scored["asset"] = scored["asset"].astype(str)
        finite_score = pd.to_numeric(scored["value"], errors="coerce").notna()
        scored = scored.loc[finite_score, ["date", "asset"]].drop_duplicates()

    scored_counts = (
        scored.groupby("date", sort=True)["asset"].nunique().rename("scored_count")
        if not scored.empty
        else pd.Series(dtype="int64", name="scored_count")
    )

    scored_evaluable_counts = scored_counts.rename("scored_evaluable_count")
    if target_label_df is not None and not target_label_df.empty and not scored.empty:
        labels = target_label_df.loc[:, ["date", "asset", "value"]].copy()
        labels["date"] = pd.to_datetime(labels["date"], errors="coerce")
        labels["asset"] = labels["asset"].astype(str)
        valid_label = pd.to_numeric(labels["value"], errors="coerce").notna()
        label_pairs = labels.loc[valid_label, ["date", "asset"]].drop_duplicates()
        scored_evaluable_counts = (
            scored.merge(label_pairs, on=["date", "asset"], how="inner", validate="one_to_one")
            .groupby("date", sort=True)["asset"]
            .nunique()
            .rename("scored_evaluable_count")
        )

    if coverage_base_df is not None and not coverage_base_df.empty:
        summary = coverage_base_df.copy()
        summary["date"] = pd.to_datetime(summary["date"], errors="coerce")
        summary = summary.dropna(subset=["date"]).sort_values("date", kind="mergesort")
        summary = summary.drop_duplicates(subset=["date"], keep="last").set_index("date")
    else:
        summary = pd.DataFrame(index=scored_counts.index)
        summary["universe_count"] = scored_counts
        summary["feature_row_count"] = scored_counts
        summary["complete_feature_count"] = scored_counts
        summary["feature_nan_row_count"] = 0
        summary["label_available_count"] = scored_counts
        summary["eligible_count"] = scored_counts
        summary["missing_feature_count"] = 0
        summary["missing_label_count"] = 0
        summary["filtered_count"] = 0

    summary["scored_count"] = scored_counts.reindex(summary.index).fillna(0)
    summary["scored_evaluable_count"] = scored_evaluable_counts.reindex(summary.index).fillna(0)

    for count_column in [
        "universe_count",
        "feature_row_count",
        "complete_feature_count",
        "feature_nan_row_count",
        "label_available_count",
        "eligible_count",
        "scored_count",
        "scored_evaluable_count",
        "missing_feature_count",
        "missing_label_count",
        "filtered_count",
    ]:
        if count_column not in summary.columns:
            summary[count_column] = pd.NA
        summary[count_column] = pd.to_numeric(summary[count_column], errors="coerce")

    if "eligible_count" not in summary.columns:
        summary["eligible_count"] = summary["label_available_count"]
    eligible = pd.to_numeric(summary["eligible_count"], errors="coerce")
    scored_evaluable = pd.to_numeric(summary["scored_evaluable_count"], errors="coerce")
    scored_total = pd.to_numeric(summary["scored_count"], errors="coerce")
    feature_total = pd.to_numeric(summary["feature_row_count"], errors="coerce")
    universe_total = pd.to_numeric(summary["universe_count"], errors="coerce")
    summary["missing_score_count"] = (eligible - scored_evaluable).clip(lower=0)
    summary["coverage"] = scored_evaluable / eligible.replace(0, pd.NA)
    summary["score_coverage"] = scored_total / feature_total.replace(0, pd.NA)
    summary["universe_coverage"] = scored_total / universe_total.replace(0, pd.NA)
    summary["missingness"] = 1.0 - summary["coverage"]
    summary["n_assets"] = eligible
    summary["n_non_null"] = scored_evaluable
    out = summary.reset_index()
    for column in columns:
        if column not in out.columns:
            out[column] = pd.NA
    return out[columns]
