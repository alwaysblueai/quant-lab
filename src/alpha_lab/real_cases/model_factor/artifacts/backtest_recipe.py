from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from alpha_lab.real_cases.artifact_enrichment import (
    build_backtest_summary_payload,
    build_portfolio_recipe_controls,
)

from ..spec import ModelFactorCaseSpec

# Cross-module imports (auto-added)
from ._utils import (
    ModelFactorArtifactPaths,
    _as_object,
    _finite_if_number,
    _split_contract_payload,
    _split_contract_top_level_fields,
    _text_or_none,
    _to_jsonable,
)


def _build_portfolio_recipe_payload(
    *,
    spec: ModelFactorCaseSpec,
    metrics_for_payload: Mapping[str, object],
    portfolio_validation_payload: Mapping[str, object],
    output_paths: ModelFactorArtifactPaths,
) -> dict[str, object]:
    controls = build_portfolio_recipe_controls(
        metrics_for_payload=metrics_for_payload,
        portfolio_validation_payload=portfolio_validation_payload,
    )
    return {
        "schema_version": "1.0.0",
        "artifact_type": "alpha_lab_portfolio_recipe",
        "case_name": spec.name,
        "package_type": "single_factor",
        "recipe_context": {
            "factor_name": spec.factor_name,
            "rebalance_frequency": spec.rebalance_frequency,
            "universe_name": spec.universe.name,
            "target_horizon": spec.target.horizon,
            "neutralization_enabled": bool(spec.neutralization.enabled),
            "model_family": spec.model.family,
        },
        "portfolio_validation_summary": _to_jsonable(
            _as_object(portfolio_validation_payload.get("portfolio_validation_summary"))
        ),
        "portfolio_validation_metrics": _to_jsonable(
            _as_object(portfolio_validation_payload.get("portfolio_validation_metrics"))
        ),
        "portfolio_validation_package": _to_jsonable(
            _as_object(portfolio_validation_payload.get("portfolio_validation_package"))
        ),
        "turnover_penalty_settings": controls["turnover_penalty_settings"],
        "transaction_cost_assumptions": controls["transaction_cost_assumptions"],
        "position_limits": controls["position_limits"],
        "source_artifacts": {
            "portfolio_validation_summary_path": str(output_paths["portfolio_validation_summary"]),
            "portfolio_validation_metrics_path": str(output_paths["portfolio_validation_metrics"]),
            "portfolio_validation_package_path": str(output_paths["portfolio_validation_package"]),
            "metrics_path": str(output_paths["metrics"]),
        },
        "fallback_derived_fields": [],
        "metrics_reference": {
            "transaction_cost_one_way_rate": _finite_if_number(
                metrics_for_payload.get("transaction_cost_one_way_rate")
            ),
            "base_weighting_method": _text_or_none(
                metrics_for_payload.get("base_weighting_method")
            ),
            "research_evaluation_profile": _text_or_none(
                metrics_for_payload.get("research_evaluation_profile")
            ),
        },
    }


def _build_backtest_result_payload(
    *,
    spec: ModelFactorCaseSpec,
    metrics_for_payload: Mapping[str, object],
    group_returns_df: pd.DataFrame,
    output_paths: ModelFactorArtifactPaths,
) -> dict[str, object]:
    summary, fallback_fields = build_backtest_summary_payload(
        group_returns_df=group_returns_df,
        rebalance_frequency=spec.rebalance_frequency,
        metrics_for_payload=metrics_for_payload,
        label_horizon=int(spec.target.horizon),
    )
    split_contract = _split_contract_payload(metrics_for_payload)
    payload: dict[str, object] = {
        "schema_version": "1.0.0",
        "artifact_type": "alpha_lab_backtest_result",
        "case_name": spec.name,
        "package_type": "single_factor",
        "rebalance_frequency": spec.rebalance_frequency,
        "target_horizon": int(spec.target.horizon),
        "summary": summary,
        "source_artifacts": {
            "group_returns_path": str(output_paths["group_returns"]),
            "group_nav_path": str(output_paths["group_nav"]),
            "turnover_path": str(output_paths["turnover"]),
            "metrics_path": str(output_paths["metrics"]),
        },
        "fallback_derived_fields": fallback_fields,
    }
    if split_contract:
        payload.update(_split_contract_top_level_fields(split_contract))
    return payload
