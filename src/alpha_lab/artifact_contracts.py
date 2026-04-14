from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import NoReturn, cast

from alpha_lab.exceptions import AlphaLabDataError

_LEVEL2_PACKAGE_TYPE = "alpha_lab_level2_portfolio_validation_package"
_RESEARCH_VALIDATION_PACKAGE_TYPE = "alpha_lab_research_validation_package"
_FACTOR_DEFINITION_TYPE = "alpha_lab_factor_definition"
_SIGNAL_VALIDATION_TYPE = "alpha_lab_signal_validation"
_PORTFOLIO_RECIPE_TYPE = "alpha_lab_portfolio_recipe"
_BACKTEST_RESULT_TYPE = "alpha_lab_backtest_result"
_FACTOR_SET_RESULT_TYPE = "alpha_lab_factor_set_result"
_CANDIDATE_RECIPE_GENERATION_TYPE = "alpha_lab_candidate_recipe_generation"
_WINNER_SELECTION_TYPE = "alpha_lab_winner_selection"
_NEXT_STEP_RECOMMENDATIONS_TYPE = "alpha_lab_next_step_recommendations"
_ARTIFACT_LOAD_DIAGNOSTICS_TYPE = "alpha_lab_artifact_load_diagnostics"
_RESEARCH_ARTIFACT_MANIFEST_TYPE = "alpha_lab_research_artifact_manifest"

_KNOWN_RUN_ARTIFACT_TYPES: frozenset[str] = frozenset(
    {
        "real_case_single_factor_bundle",
        "real_case_composite_bundle",
    }
)

_KNOWN_CASE_PACKAGE_TYPES: frozenset[str] = frozenset({"single_factor", "composite"})
_KNOWN_CASE_STATUSES: frozenset[str] = frozenset({"success", "failed", "skipped"})


CORE_LEVEL12_JSON_ARTIFACTS: tuple[str, ...] = (
    "run_manifest.json",
    "metrics.json",
    "factor_definition.json",
    "signal_validation.json",
    "portfolio_recipe.json",
    "backtest_result.json",
    "campaign_manifest.json",
    "campaign_results.json",
    "research_validation_package.json",
    "portfolio_validation_summary.json",
    "portfolio_validation_metrics.json",
    "portfolio_validation_package.json",
    "campaign_profile_comparison.json",
    "factor_set_result.json",
    "candidate_recipe_generation.json",
    "winner_selection.json",
    "next_step_recommendations.json",
    "artifact_load_diagnostics.json",
    "research_artifact_manifest.json",
)


def validate_level12_artifact_payload(
    payload: Mapping[str, object],
    *,
    artifact_name: str,
    source: str | Path | None = None,
) -> None:
    """Validate core Level 1/2 JSON artifact payloads.

    Unknown artifact names are ignored so callers can use one shared JSON write
    helper without branching.
    """

    label = str(source) if source is not None else artifact_name
    dispatch = {
        "run_manifest.json": validate_run_manifest_payload,
        "metrics.json": validate_metrics_payload,
        "factor_definition.json": validate_factor_definition_payload,
        "signal_validation.json": validate_signal_validation_payload,
        "portfolio_recipe.json": validate_portfolio_recipe_payload,
        "backtest_result.json": validate_backtest_result_payload,
        "campaign_manifest.json": validate_campaign_manifest_payload,
        "campaign_results.json": validate_campaign_results_payload,
        "research_validation_package.json": validate_research_validation_package_payload,
        "portfolio_validation_summary.json": validate_portfolio_validation_summary_payload,
        "portfolio_validation_metrics.json": validate_portfolio_validation_metrics_payload,
        "portfolio_validation_package.json": validate_portfolio_validation_package_payload,
        "campaign_profile_comparison.json": validate_campaign_profile_comparison_payload,
        "factor_set_result.json": validate_factor_set_result_payload,
        "candidate_recipe_generation.json": validate_candidate_recipe_generation_payload,
        "winner_selection.json": validate_winner_selection_payload,
        "next_step_recommendations.json": validate_next_step_recommendations_payload,
        "artifact_load_diagnostics.json": validate_artifact_load_diagnostics_payload,
        "research_artifact_manifest.json": validate_research_artifact_manifest_payload,
    }
    validator = dispatch.get(artifact_name)
    if validator is None:
        return
    validator(payload, source=label)


def validate_run_manifest_payload(
    payload: Mapping[str, object],
    *,
    source: str | Path | None = None,
) -> None:
    label = str(source) if source is not None else "run_manifest.json"
    _require_non_empty_string(payload, "schema_version", label)
    artifact_type = _require_non_empty_string(payload, "artifact_type", label)
    if artifact_type not in _KNOWN_RUN_ARTIFACT_TYPES:
        _raise(
            label,
            f"`artifact_type` must be one of {sorted(_KNOWN_RUN_ARTIFACT_TYPES)}",
        )
    _require_non_empty_string(payload, "run_timestamp_utc", label)
    _require_non_empty_string(payload, "case_name", label)
    outputs = _require_object(payload, "outputs", label)
    for key, value in outputs.items():
        field = f"{label}.outputs[{key!r}]"
        if not _is_non_empty_string(value):
            _raise(field, "must be a non-empty string")

    required_bundle_files = _require_list(payload, "required_bundle_files", label)
    _require_string_list(required_bundle_files, f"{label}.required_bundle_files")

    _require_object(payload, "integrity_summary", label)
    evaluation_standard = _require_object(payload, "evaluation_standard", label)
    _require_non_empty_string(
        evaluation_standard,
        "profile_name",
        f"{label}.evaluation_standard",
    )
    _require_object(
        evaluation_standard,
        "snapshot",
        f"{label}.evaluation_standard",
    )


def validate_metrics_payload(
    payload: Mapping[str, object],
    *,
    source: str | Path | None = None,
) -> None:
    label = str(source) if source is not None else "metrics.json"
    metrics = _require_object(payload, "metrics", label)
    _require_non_empty_string(metrics, "research_evaluation_profile", f"{label}.metrics")
    _require_non_empty_string(metrics, "campaign_triage", f"{label}.metrics")
    _require_non_empty_string(metrics, "promotion_decision", f"{label}.metrics")

    for key in ("factor_verdict_reasons", "campaign_triage_reasons", "promotion_reasons"):
        _validate_string_list_if_present(metrics, key, f"{label}.metrics")
    for key in (
        "promotion_blockers",
        "portfolio_validation_major_risks",
    ):
        _validate_string_list_if_present(metrics, key, f"{label}.metrics")
    _validate_string_list_or_none_if_present(
        metrics,
        "portfolio_validation_benchmark_relative_risks",
        f"{label}.metrics",
    )
    _validate_finite_number_or_none_if_present(metrics, "ic_t_stat", f"{label}.metrics")
    _validate_finite_number_or_none_if_present(metrics, "ic_p_value", f"{label}.metrics")
    _validate_finite_number_or_none_if_present(metrics, "dsr_pvalue", f"{label}.metrics")
    if "split_description" in metrics:
        _require_non_empty_string(metrics, "split_description", f"{label}.metrics")
    if "data_quality_status" in metrics:
        status = _require_non_empty_string(metrics, "data_quality_status", f"{label}.metrics")
        if status not in {"pass", "warn", "fail"}:
            _raise(
                f"{label}.metrics.data_quality_status",
                "must be one of ['fail', 'pass', 'warn']",
            )
    for key in (
        "data_quality_suspended_rows",
        "data_quality_stale_rows",
        "data_quality_suspected_split_rows",
        "data_quality_integrity_warn_count",
        "data_quality_integrity_fail_count",
        "data_quality_hard_fail_count",
    ):
        _validate_int_or_none_if_present(metrics, key, f"{label}.metrics")

    coverage = _require_object(payload, "coverage_by_date_summary", label)
    _require_int(coverage, "n_dates", f"{label}.coverage_by_date_summary")
    for key in ("mean_coverage", "min_coverage"):
        _validate_finite_number_or_none_if_present(
            coverage,
            key,
            f"{label}.coverage_by_date_summary",
        )

    _validate_object_if_present(payload, "portfolio_validation_summary", label)
    _validate_object_if_present(payload, "portfolio_validation_metrics", label)
    _validate_object_if_present(payload, "portfolio_validation_package", label)


def validate_campaign_manifest_payload(
    payload: Mapping[str, object],
    *,
    source: str | Path | None = None,
) -> None:
    label = str(source) if source is not None else "campaign_manifest.json"
    _require_non_empty_string(payload, "schema_version", label)
    _require_non_empty_string(payload, "campaign_name", label)
    _require_non_empty_string(payload, "campaign_description", label)
    _require_non_empty_string(payload, "run_timestamp_utc", label)
    _require_non_empty_string(payload, "output_root_dir", label)

    execution_order = _require_list(payload, "execution_order", label)
    _require_string_list(execution_order, f"{label}.execution_order", allow_empty=False)

    cases = _require_list(payload, "cases", label)
    if not cases:
        _raise(f"{label}.cases", "must be a non-empty list")
    for idx, row in enumerate(cases):
        if not isinstance(row, Mapping):
            _raise(f"{label}.cases[{idx}]", "must be an object")
        row_obj = cast(Mapping[str, object], row)
        _require_non_empty_string(row_obj, "case_name", f"{label}.cases[{idx}]")
        package_type = _require_non_empty_string(row_obj, "package_type", f"{label}.cases[{idx}]")
        if package_type not in _KNOWN_CASE_PACKAGE_TYPES:
            _raise(
                f"{label}.cases[{idx}].package_type",
                f"must be one of {sorted(_KNOWN_CASE_PACKAGE_TYPES)}",
            )
        _require_non_empty_string(row_obj, "spec_path", f"{label}.cases[{idx}]")

    evaluation_standard = _require_object(payload, "evaluation_standard", label)
    _require_non_empty_string(
        evaluation_standard,
        "profile_name",
        f"{label}.evaluation_standard",
    )
    _require_object(evaluation_standard, "snapshot", f"{label}.evaluation_standard")


def validate_factor_definition_payload(
    payload: Mapping[str, object],
    *,
    source: str | Path | None = None,
) -> None:
    label = str(source) if source is not None else "factor_definition.json"
    _require_non_empty_string(payload, "schema_version", label)
    artifact_type = _require_non_empty_string(payload, "artifact_type", label)
    if artifact_type != _FACTOR_DEFINITION_TYPE:
        _raise(f"{label}.artifact_type", f"must be `{_FACTOR_DEFINITION_TYPE}`")
    _require_non_empty_string(payload, "case_name", label)
    package_type = _require_non_empty_string(payload, "package_type", label)
    if package_type not in _KNOWN_CASE_PACKAGE_TYPES:
        _raise(
            f"{label}.package_type",
            f"must be one of {sorted(_KNOWN_CASE_PACKAGE_TYPES)}",
        )
    _require_non_empty_string(payload, "factor_name", label)
    _require_object(payload, "spec", label)
    _require_object(payload, "source_artifacts", label)
    _validate_string_list_if_present(payload, "fallback_derived_fields", label)


def validate_signal_validation_payload(
    payload: Mapping[str, object],
    *,
    source: str | Path | None = None,
) -> None:
    label = str(source) if source is not None else "signal_validation.json"
    _require_non_empty_string(payload, "schema_version", label)
    artifact_type = _require_non_empty_string(payload, "artifact_type", label)
    if artifact_type != _SIGNAL_VALIDATION_TYPE:
        _raise(f"{label}.artifact_type", f"must be `{_SIGNAL_VALIDATION_TYPE}`")
    _require_non_empty_string(payload, "case_name", label)
    package_type = _require_non_empty_string(payload, "package_type", label)
    if package_type not in _KNOWN_CASE_PACKAGE_TYPES:
        _raise(
            f"{label}.package_type",
            f"must be one of {sorted(_KNOWN_CASE_PACKAGE_TYPES)}",
        )
    metrics = _require_object(payload, "metrics", label)
    coverage = _require_object(payload, "coverage_by_date_summary", label)
    legacy_shape: Mapping[str, object] = {
        "metrics": metrics,
        "coverage_by_date_summary": coverage,
    }
    validate_metrics_payload(
        legacy_shape,
        source=f"{label}.legacy_metrics_shape",
    )
    _require_object(payload, "source_artifacts", label)
    _validate_string_list_if_present(payload, "fallback_derived_fields", label)


def validate_portfolio_recipe_payload(
    payload: Mapping[str, object],
    *,
    source: str | Path | None = None,
) -> None:
    label = str(source) if source is not None else "portfolio_recipe.json"
    _require_non_empty_string(payload, "schema_version", label)
    artifact_type = _require_non_empty_string(payload, "artifact_type", label)
    if artifact_type != _PORTFOLIO_RECIPE_TYPE:
        _raise(f"{label}.artifact_type", f"must be `{_PORTFOLIO_RECIPE_TYPE}`")
    _require_non_empty_string(payload, "case_name", label)
    package_type = _require_non_empty_string(payload, "package_type", label)
    if package_type not in _KNOWN_CASE_PACKAGE_TYPES:
        _raise(
            f"{label}.package_type",
            f"must be one of {sorted(_KNOWN_CASE_PACKAGE_TYPES)}",
        )
    _require_object(payload, "recipe_context", label)
    validate_portfolio_validation_summary_payload(
        _require_object(payload, "portfolio_validation_summary", label),
        source=f"{label}.portfolio_validation_summary",
    )
    validate_portfolio_validation_metrics_payload(
        _require_object(payload, "portfolio_validation_metrics", label),
        source=f"{label}.portfolio_validation_metrics",
    )
    validate_portfolio_validation_package_payload(
        _require_object(payload, "portfolio_validation_package", label),
        source=f"{label}.portfolio_validation_package",
    )
    for key in (
        "turnover_penalty_settings",
        "transaction_cost_assumptions",
        "position_limits",
    ):
        if key in payload:
            value = payload.get(key)
            if value is not None and not _is_non_empty_string(value):
                _raise(f"{label}.{key}", "must be a non-empty string or null")
    _require_object(payload, "source_artifacts", label)
    _validate_string_list_if_present(payload, "fallback_derived_fields", label)


def validate_backtest_result_payload(
    payload: Mapping[str, object],
    *,
    source: str | Path | None = None,
) -> None:
    label = str(source) if source is not None else "backtest_result.json"
    _require_non_empty_string(payload, "schema_version", label)
    artifact_type = _require_non_empty_string(payload, "artifact_type", label)
    if artifact_type != _BACKTEST_RESULT_TYPE:
        _raise(f"{label}.artifact_type", f"must be `{_BACKTEST_RESULT_TYPE}`")
    _require_non_empty_string(payload, "case_name", label)
    package_type = _require_non_empty_string(payload, "package_type", label)
    if package_type not in _KNOWN_CASE_PACKAGE_TYPES:
        _raise(
            f"{label}.package_type",
            f"must be one of {sorted(_KNOWN_CASE_PACKAGE_TYPES)}",
        )
    _require_non_empty_string(payload, "rebalance_frequency", label)
    summary = _require_object(payload, "summary", label)
    for key in (
        "annualized_return",
        "annualized_volatility",
        "sharpe",
        "sortino",
        "max_drawdown",
        "calmar",
        "win_rate",
        "turnover",
        "information_ratio",
        "excess_return_vs_benchmark",
        "tracking_error",
        "pre_cost_return",
        "post_cost_return",
        "rolling_sharpe",
        "rolling_drawdown",
    ):
        _validate_finite_number_or_none_if_present(summary, key, f"{label}.summary")
    for key in ("subperiod_analysis", "regime_analysis"):
        if key in summary:
            value = summary.get(key)
            if value is not None and not _is_non_empty_string(value):
                _raise(f"{label}.summary.{key}", "must be a non-empty string or null")
    _validate_time_value_rows(summary.get("nav_points"), f"{label}.summary.nav_points")
    _validate_time_value_rows(
        summary.get("monthly_return_table"),
        f"{label}.summary.monthly_return_table",
    )
    _validate_time_value_rows(
        summary.get("drawdown_table"),
        f"{label}.summary.drawdown_table",
    )
    _require_object(payload, "source_artifacts", label)
    _validate_string_list_if_present(payload, "fallback_derived_fields", label)


def validate_campaign_results_payload(
    payload: Mapping[str, object],
    *,
    source: str | Path | None = None,
) -> None:
    label = str(source) if source is not None else "campaign_results.json"
    _require_non_empty_string(payload, "campaign_name", label)
    _require_non_empty_string(payload, "run_timestamp_utc", label)
    _require_non_empty_string(payload, "evaluation_profile", label)

    for key in ("n_cases", "n_success", "n_failed", "n_skipped"):
        _require_int(payload, key, label)

    cases = _require_list(payload, "cases", label)
    for idx, row in enumerate(cases):
        if not isinstance(row, Mapping):
            _raise(f"{label}.cases[{idx}]", "must be an object")
        row_obj = cast(Mapping[str, object], row)
        _require_non_empty_string(row_obj, "case_name", f"{label}.cases[{idx}]")
        package_type = _require_non_empty_string(row_obj, "package_type", f"{label}.cases[{idx}]")
        if package_type not in _KNOWN_CASE_PACKAGE_TYPES:
            _raise(
                f"{label}.cases[{idx}].package_type",
                f"must be one of {sorted(_KNOWN_CASE_PACKAGE_TYPES)}",
            )
        status = _require_non_empty_string(row_obj, "status", f"{label}.cases[{idx}]")
        if status not in _KNOWN_CASE_STATUSES:
            _raise(
                f"{label}.cases[{idx}].status",
                f"must be one of {sorted(_KNOWN_CASE_STATUSES)}",
            )
        key_metrics = _require_object(row_obj, "key_metrics", f"{label}.cases[{idx}]")
        if status == "success":
            _require_non_empty_string(
                key_metrics,
                "research_evaluation_profile",
                f"{label}.cases[{idx}].key_metrics",
            )
            _require_non_empty_string(
                key_metrics,
                "campaign_triage",
                f"{label}.cases[{idx}].key_metrics",
            )
            _require_non_empty_string(
                key_metrics,
                "promotion_decision",
                f"{label}.cases[{idx}].key_metrics",
            )

        _validate_object_if_present(row_obj, "campaign_triage", f"{label}.cases[{idx}]")
        _validate_object_if_present(row_obj, "level2_promotion", f"{label}.cases[{idx}]")


def validate_research_validation_package_payload(
    payload: Mapping[str, object],
    *,
    source: str | Path | None = None,
) -> None:
    label = str(source) if source is not None else "research_validation_package.json"
    _require_non_empty_string(payload, "schema_version", label)
    package_type = _require_non_empty_string(payload, "package_type", label)
    if package_type != _RESEARCH_VALIDATION_PACKAGE_TYPE:
        _raise(
            f"{label}.package_type",
            f"must be `{_RESEARCH_VALIDATION_PACKAGE_TYPE}`",
        )

    for key in (
        "created_at_utc",
        "case_id",
        "case_name",
        "case_output_dir",
        "workflow_type",
        "experiment_name",
    ):
        _require_non_empty_string(payload, key, label)

    _require_object(payload, "identity", label)

    research_intent = _require_object(payload, "research_intent", label)
    _require_non_empty_string(research_intent, "workflow_type", f"{label}.research_intent")
    _require_non_empty_string(
        research_intent,
        "evaluation_profile",
        f"{label}.research_intent",
    )

    research_results = _require_object(payload, "research_results", label)
    _require_object(research_results, "key_metrics", f"{label}.research_results")
    evaluation_standard = _require_object(
        research_results,
        "evaluation_standard",
        f"{label}.research_results",
    )
    _require_non_empty_string(
        evaluation_standard,
        "profile_name",
        f"{label}.research_results.evaluation_standard",
    )
    _require_object(
        evaluation_standard,
        "snapshot",
        f"{label}.research_results.evaluation_standard",
    )
    validate_portfolio_validation_summary_payload(
        _require_object(
            research_results,
            "portfolio_validation_summary",
            f"{label}.research_results",
        ),
        source=f"{label}.research_results.portfolio_validation_summary",
    )
    validate_portfolio_validation_metrics_payload(
        _require_object(
            research_results,
            "portfolio_validation_metrics",
            f"{label}.research_results",
        ),
        source=f"{label}.research_results.portfolio_validation_metrics",
    )
    validate_portfolio_validation_package_payload(
        _require_object(
            research_results,
            "portfolio_validation_package",
            f"{label}.research_results",
        ),
        source=f"{label}.research_results.portfolio_validation_package",
    )

    _require_object(payload, "trial_registry_metadata", label)
    artifact_index = _require_list(payload, "artifact_index", label)
    for idx, row in enumerate(artifact_index):
        item_label = f"{label}.artifact_index[{idx}]"
        if not isinstance(row, Mapping):
            _raise(item_label, "must be an object")
        item = cast(Mapping[str, object], row)
        _require_non_empty_string(item, "name", item_label)
        _require_non_empty_string(item, "artifact_type", item_label)
        path_value = item.get("path")
        if not isinstance(path_value, str):
            _raise(f"{item_label}.path", "must be a string")
        _require_bool(item, "exists", item_label)
        _require_bool(item, "required", item_label)


def validate_portfolio_validation_summary_payload(
    payload: Mapping[str, object],
    *,
    source: str | Path | None = None,
) -> None:
    label = str(source) if source is not None else "portfolio_validation_summary.json"
    _require_non_empty_string(payload, "validation_status", label)
    _require_non_empty_string(payload, "promotion_decision", label)
    _require_non_empty_string(payload, "recommendation", label)
    _validate_bool_or_none_if_present(payload, "remains_credible_at_portfolio_level", label)
    _validate_string_list_if_present(payload, "major_risks", label)
    _validate_string_list_if_present(payload, "major_caveats", label)
    _validate_string_list_if_present(payload, "benchmark_relative_risks", label)
    _validate_portfolio_robustness_summary_if_present(payload, label=label)
    _validate_int_or_none_if_present(payload, "base_holding_period", label)
    _validate_int_or_none_if_present(payload, "rebalance_step_assumption", label)
    for key in (
        "base_mean_portfolio_return",
        "base_mean_turnover",
        "base_cost_adjusted_return_review_rate",
        "benchmark_relative_excess_return",
        "benchmark_relative_tracking_error",
    ):
        _validate_finite_number_or_none_if_present(payload, key, label)


def validate_portfolio_validation_metrics_payload(
    payload: Mapping[str, object],
    *,
    source: str | Path | None = None,
) -> None:
    label = str(source) if source is not None else "portfolio_validation_metrics.json"
    _require_object(payload, "protocol_settings", label)
    scenario_metrics = _require_list(payload, "scenario_metrics", label)
    for idx, row in enumerate(scenario_metrics):
        if not isinstance(row, Mapping):
            _raise(f"{label}.scenario_metrics[{idx}]", "must be an object")
        scenario = cast(Mapping[str, object], row)
        _require_non_empty_string(
            scenario,
            "weighting_method",
            f"{label}.scenario_metrics[{idx}]",
        )
        _require_int(scenario, "holding_period", f"{label}.scenario_metrics[{idx}]")

    _require_list(payload, "holding_period_sensitivity", label)
    _require_list(payload, "weighting_sensitivity", label)
    _require_object(payload, "turnover_summary", label)
    _require_object(payload, "transaction_cost_sensitivity", label)
    _require_object(payload, "benchmark_relative_evaluation", label)
    _validate_object_if_present(payload, "concentration_exposure_diagnostics", label)


def validate_portfolio_validation_package_payload(
    payload: Mapping[str, object],
    *,
    source: str | Path | None = None,
) -> None:
    label = str(source) if source is not None else "portfolio_validation_package.json"
    _require_non_empty_string(payload, "schema_version", label)
    package_type = _require_non_empty_string(payload, "package_type", label)
    if package_type != _LEVEL2_PACKAGE_TYPE:
        _raise(
            f"{label}.package_type",
            f"must be `{_LEVEL2_PACKAGE_TYPE}`",
        )
    _require_non_empty_string(payload, "created_at_utc", label)
    _require_object(payload, "input_case_identity", label)
    _require_object(payload, "promotion_decision_context", label)
    _require_object(payload, "portfolio_validation_settings", label)
    _require_object(payload, "key_portfolio_results", label)
    recommendation = _require_object(payload, "recommendation", label)
    _require_non_empty_string(recommendation, "label", f"{label}.recommendation")
    _validate_string_list_if_present(payload, "major_risks", label)
    _validate_string_list_if_present(payload, "major_caveats", label)
    _validate_portfolio_robustness_summary_if_present(payload, label=label)


def validate_campaign_profile_comparison_payload(
    payload: Mapping[str, object],
    *,
    source: str | Path | None = None,
) -> None:
    label = str(source) if source is not None else "campaign_profile_comparison.json"
    _require_non_empty_string(payload, "schema_version", label)
    _require_non_empty_string(payload, "output_root_dir", label)
    if not (
        _is_non_empty_string(payload.get("workflow_name"))
        or _is_non_empty_string(payload.get("example_name"))
    ):
        _raise(
            label,
            "must include either non-empty `workflow_name` or non-empty `example_name`",
        )

    profiles = _require_list(payload, "profiles", label)
    _require_string_list(profiles, f"{label}.profiles", allow_empty=False)
    _require_list(payload, "cases", label)

    profile_runs = _require_list(payload, "profile_runs", label)
    for idx, row in enumerate(profile_runs):
        item_label = f"{label}.profile_runs[{idx}]"
        if not isinstance(row, Mapping):
            _raise(item_label, "must be an object")
        profile_row = cast(Mapping[str, object], row)
        _require_non_empty_string(profile_row, "profile_name", item_label)
        ranked = _require_list(profile_row, "ranked_case_order", item_label)
        _require_string_list(ranked, f"{item_label}.ranked_case_order")
        _require_list(profile_row, "case_rows", item_label)

    field_change_index = _require_object(payload, "field_change_index", label)
    for key, value in field_change_index.items():
        if not isinstance(value, list):
            _raise(f"{label}.field_change_index[{key!r}]", "must be a list of strings")
        _require_string_list(
            cast(list[object], value),
            f"{label}.field_change_index[{key!r}]",
        )

    case_comparison = _require_list(payload, "case_comparison", label)
    for idx, row in enumerate(case_comparison):
        item_label = f"{label}.case_comparison[{idx}]"
        if not isinstance(row, Mapping):
            _raise(item_label, "must be an object")
        comparison_row = cast(Mapping[str, object], row)
        _require_non_empty_string(comparison_row, "case_name", item_label)
        changed_fields = _require_list(comparison_row, "changed_fields", item_label)
        _require_string_list(changed_fields, f"{item_label}.changed_fields")
        _require_object(comparison_row, "profiles", item_label)
        transition_delta = comparison_row.get("level12_transition_profile_delta")
        if transition_delta is not None:
            if not isinstance(transition_delta, Mapping):
                _raise(
                    f"{item_label}.level12_transition_profile_delta",
                    "must be an object",
                )
            transition_delta_obj = cast(Mapping[str, object], transition_delta)
            _require_non_empty_string(
                transition_delta_obj,
                "delta_label",
                f"{item_label}.level12_transition_profile_delta",
            )
            profile_labels = _require_object(
                transition_delta_obj,
                "profile_transition_labels",
                f"{item_label}.level12_transition_profile_delta",
            )
            for key, value in profile_labels.items():
                key_label = (
                    f"{item_label}.level12_transition_profile_delta."
                    f"profile_transition_labels[{key!r}]"
                )
                if not _is_non_empty_string(key):
                    _raise(key_label, "key must be a non-empty string")
                if not _is_non_empty_string(value):
                    _raise(key_label, "must be a non-empty string")
            pair_directions = _require_list(
                transition_delta_obj,
                "profile_pair_directions",
                f"{item_label}.level12_transition_profile_delta",
            )
            for pair_idx, pair in enumerate(pair_directions):
                pair_label = (
                    f"{item_label}.level12_transition_profile_delta."
                    f"profile_pair_directions[{pair_idx}]"
                )
                if not isinstance(pair, Mapping):
                    _raise(pair_label, "must be an object")
                pair_obj = cast(Mapping[str, object], pair)
                for field in (
                    "from_profile",
                    "to_profile",
                    "from_label",
                    "to_label",
                    "direction",
                ):
                    _require_non_empty_string(pair_obj, field, pair_label)

    summary = _require_object(payload, "campaign_level_summary", label)
    for key in (
        "stable_cases",
        "profile_sensitive_cases",
        "promoted_only_under_looser_profiles",
        "consistently_strong",
        "highly_profile_sensitive",
        "transition_stable_cases",
        "transition_sensitive_cases",
    ):
        _validate_string_list_if_present(summary, key, f"{label}.campaign_level_summary")
    transition_delta_label_counts = summary.get("transition_delta_label_counts")
    if transition_delta_label_counts is not None:
        if not isinstance(transition_delta_label_counts, Mapping):
            _raise(
                f"{label}.campaign_level_summary.transition_delta_label_counts",
                "must be an object",
            )
        for key, value in cast(Mapping[str, object], transition_delta_label_counts).items():
            key_label = (
                f"{label}.campaign_level_summary.transition_delta_label_counts[{key!r}]"
            )
            if not _is_non_empty_string(key):
                _raise(key_label, "key must be a non-empty string")
            if isinstance(value, bool) or not isinstance(value, int):
                _raise(key_label, "must be an integer")
    transition_delta_matrix = summary.get("level12_transition_profile_delta_matrix")
    if transition_delta_matrix is not None:
        if not isinstance(transition_delta_matrix, Mapping):
            _raise(
                f"{label}.campaign_level_summary.level12_transition_profile_delta_matrix",
                "must be an object",
            )
        matrix_obj = cast(Mapping[str, object], transition_delta_matrix)
        pair_rows = _require_list(
            matrix_obj,
            "profile_pairs",
            f"{label}.campaign_level_summary.level12_transition_profile_delta_matrix",
        )
        for pair_idx, pair_row in enumerate(pair_rows):
            pair_label = (
                f"{label}.campaign_level_summary.level12_transition_profile_delta_matrix."
                f"profile_pairs[{pair_idx}]"
            )
            if not isinstance(pair_row, Mapping):
                _raise(pair_label, "must be an object")
            pair_obj = cast(Mapping[str, object], pair_row)
            _require_non_empty_string(pair_obj, "from_profile", pair_label)
            _require_non_empty_string(pair_obj, "to_profile", pair_label)
            for field in (
                "n_cases_compared",
                "n_cases_with_observed_transition_labels",
                "n_cases_missing_transition_labels",
                "stable_count",
                "changed_count",
            ):
                _require_int(pair_obj, field, pair_label)
            counts = _require_object(pair_obj, "counts_by_from_to_label", pair_label)
            proportions = _require_object(pair_obj, "proportions_by_from_to_label", pair_label)
            for from_label, to_counts in counts.items():
                from_label_key = (
                    f"{pair_label}.counts_by_from_to_label[{from_label!r}]"
                )
                if not _is_non_empty_string(from_label):
                    _raise(from_label_key, "key must be a non-empty string")
                if not isinstance(to_counts, Mapping):
                    _raise(from_label_key, "must be an object")
                for to_label, count in cast(Mapping[str, object], to_counts).items():
                    to_label_key = f"{from_label_key}[{to_label!r}]"
                    if not _is_non_empty_string(to_label):
                        _raise(to_label_key, "key must be a non-empty string")
                    if isinstance(count, bool) or not isinstance(count, int):
                        _raise(to_label_key, "must be an integer")
            for from_label, to_props in proportions.items():
                from_label_key = (
                    f"{pair_label}.proportions_by_from_to_label[{from_label!r}]"
                )
                if not _is_non_empty_string(from_label):
                    _raise(from_label_key, "key must be a non-empty string")
                if not isinstance(to_props, Mapping):
                    _raise(from_label_key, "must be an object")
                for to_label, prop in cast(Mapping[str, object], to_props).items():
                    to_label_key = f"{from_label_key}[{to_label!r}]"
                    if not _is_non_empty_string(to_label):
                        _raise(to_label_key, "key must be a non-empty string")
                    if not isinstance(prop, (int, float)) or isinstance(prop, bool):
                        _raise(to_label_key, "must be a number")
    transition_reason_delta_matrix = summary.get("level12_transition_reason_profile_delta_matrix")
    if transition_reason_delta_matrix is not None:
        if not isinstance(transition_reason_delta_matrix, Mapping):
            _raise(
                f"{label}.campaign_level_summary.level12_transition_reason_profile_delta_matrix",
                "must be an object",
            )
        matrix_obj = cast(Mapping[str, object], transition_reason_delta_matrix)
        pair_rows = _require_list(
            matrix_obj,
            "profile_pairs",
            (
                f"{label}.campaign_level_summary."
                "level12_transition_reason_profile_delta_matrix"
            ),
        )
        for pair_idx, pair_row in enumerate(pair_rows):
            pair_label = (
                f"{label}.campaign_level_summary."
                "level12_transition_reason_profile_delta_matrix."
                f"profile_pairs[{pair_idx}]"
            )
            if not isinstance(pair_row, Mapping):
                _raise(pair_label, "must be an object")
            pair_obj = cast(Mapping[str, object], pair_row)
            _require_non_empty_string(pair_obj, "from_profile", pair_label)
            _require_non_empty_string(pair_obj, "to_profile", pair_label)
            for field in (
                "n_transition_labels_with_observed_reasons",
                "n_transition_labels_with_reason_shift",
                "n_transition_labels_reason_stable",
            ):
                _require_int(pair_obj, field, pair_label)
            delta_counts = _require_object(pair_obj, "reason_bucket_delta_counts", pair_label)
            for field in ("added", "removed", "increased", "decreased", "stable"):
                _require_int(
                    delta_counts,
                    field,
                    f"{pair_label}.reason_bucket_delta_counts",
                )
            by_label = _require_object(pair_obj, "reason_delta_by_transition_label", pair_label)
            for transition_label, label_payload in by_label.items():
                item_label = (
                    f"{pair_label}.reason_delta_by_transition_label[{transition_label!r}]"
                )
                if not _is_non_empty_string(transition_label):
                    _raise(item_label, "key must be a non-empty string")
                if not isinstance(label_payload, Mapping):
                    _raise(item_label, "must be an object")
                label_obj = cast(Mapping[str, object], label_payload)
                for field in (
                    "from_profile_n_cases_with_label",
                    "to_profile_n_cases_with_label",
                ):
                    _require_int(label_obj, field, item_label)
                _validate_transition_reason_stat_list(
                    label_obj.get("from_profile_dominant_reasons"),
                    f"{item_label}.from_profile_dominant_reasons",
                )
                _validate_transition_reason_stat_list(
                    label_obj.get("to_profile_dominant_reasons"),
                    f"{item_label}.to_profile_dominant_reasons",
                )
                deltas = _require_object(label_obj, "reason_bucket_deltas", item_label)
                for bucket_name in ("added", "removed", "increased", "decreased", "stable"):
                    _validate_transition_reason_delta_list(
                        deltas.get(bucket_name),
                        f"{item_label}.reason_bucket_deltas.{bucket_name}",
                    )
                _require_bool(label_obj, "is_reason_shifted", item_label)


def validate_factor_set_result_payload(
    payload: Mapping[str, object],
    *,
    source: str | Path | None = None,
) -> None:
    label = str(source) if source is not None else "factor_set_result.json"
    _require_non_empty_string(payload, "schema_version", label)
    artifact_type = _require_non_empty_string(payload, "artifact_type", label)
    if artifact_type != _FACTOR_SET_RESULT_TYPE:
        _raise(f"{label}.artifact_type", f"must be `{_FACTOR_SET_RESULT_TYPE}`")
    _require_non_empty_string(payload, "generated_at_utc", label)
    _require_non_empty_string(payload, "comparison_json_path", label)
    _require_non_empty_string(payload, "default_profile", label)
    _require_object(payload, "source_artifacts", label)
    policy = _require_object(payload, "policy", label)
    _require_non_empty_string(policy, "policy_id", f"{label}.policy")
    _require_non_empty_string(policy, "formula_text", f"{label}.policy")

    factor_sets = _require_list(payload, "factor_sets", label)
    for idx, row in enumerate(factor_sets):
        item_label = f"{label}.factor_sets[{idx}]"
        if not isinstance(row, Mapping):
            _raise(item_label, "must be an object")
        item = cast(Mapping[str, object], row)
        _require_non_empty_string(item, "factor_set_id", item_label)
        _require_non_empty_string(item, "construction_rule", item_label)
        _require_non_empty_string(item, "status", item_label)
        if "label_zh" in item and item.get("label_zh") is not None:
            _require_non_empty_string(item, "label_zh", item_label)
        _require_string_list(
            _require_list(item, "factor_ids", item_label),
            f"{item_label}.factor_ids",
        )
        _require_string_list(
            _require_list(item, "factor_names", item_label),
            f"{item_label}.factor_names",
        )
        _require_string_list(
            _require_list(item, "source_shortlist_entries", item_label),
            f"{item_label}.source_shortlist_entries",
        )
        _require_string_list(
            _require_list(item, "rationale", item_label),
            f"{item_label}.rationale",
        )
        if "rationale_zh" in item and item.get("rationale_zh") is not None:
            _require_string_list(
                _require_list(item, "rationale_zh", item_label),
                f"{item_label}.rationale_zh",
            )
        _require_string_list(
            _require_list(item, "warnings", item_label),
            f"{item_label}.warnings",
        )
        score_summary = _require_object(item, "score_summary", item_label)
        for key in (
            "mean_shortlist_score",
            "mean_icir",
            "mean_turnover",
            "mean_oos_stability_share",
            "max_pair_correlation",
            "family_balance_ratio",
        ):
            _validate_finite_number_or_none_if_present(
                score_summary,
                key,
                f"{item_label}.score_summary",
            )

    _require_string_list(
        _require_list(payload, "selected_factor_set_ids", label),
        f"{label}.selected_factor_set_ids",
    )
    _require_string_list(
        _require_list(payload, "recommendation_summary", label),
        f"{label}.recommendation_summary",
    )


def validate_candidate_recipe_generation_payload(
    payload: Mapping[str, object],
    *,
    source: str | Path | None = None,
) -> None:
    label = (
        str(source)
        if source is not None
        else "candidate_recipe_generation.json"
    )
    _require_non_empty_string(payload, "schema_version", label)
    artifact_type = _require_non_empty_string(payload, "artifact_type", label)
    if artifact_type != _CANDIDATE_RECIPE_GENERATION_TYPE:
        _raise(
            f"{label}.artifact_type",
            f"must be `{_CANDIDATE_RECIPE_GENERATION_TYPE}`",
        )
    _require_non_empty_string(payload, "generated_at_utc", label)
    _require_non_empty_string(payload, "comparison_json_path", label)
    _require_non_empty_string(payload, "default_profile", label)
    _require_object(payload, "source_artifacts", label)
    policy = _require_object(payload, "policy", label)
    _require_non_empty_string(policy, "policy_id", f"{label}.policy")
    _require_non_empty_string(policy, "formula_text", f"{label}.policy")

    recipes = _require_list(payload, "generated_recipes", label)
    for idx, row in enumerate(recipes):
        item_label = f"{label}.generated_recipes[{idx}]"
        if not isinstance(row, Mapping):
            _raise(item_label, "must be an object")
        item = cast(Mapping[str, object], row)
        for key in (
            "recipe_id",
            "recipe_name",
            "source_factor_set_id",
            "construction_variant",
            "weighting_scheme",
            "neutralization_mode",
            "turnover_penalty_mode",
            "benchmark_mode",
        ):
            _require_non_empty_string(item, key, item_label)
        _require_string_list(
            _require_list(item, "source_factor_ids", item_label),
            f"{item_label}.source_factor_ids",
        )
        _require_string_list(
            _require_list(item, "rationale", item_label),
            f"{item_label}.rationale",
        )
        _require_string_list(
            _require_list(item, "assumptions", item_label),
            f"{item_label}.assumptions",
        )
        _require_string_list(
            _require_list(item, "warnings", item_label),
            f"{item_label}.warnings",
        )

    _require_string_list(
        _require_list(payload, "recommendation_summary", label),
        f"{label}.recommendation_summary",
    )


def validate_winner_selection_payload(
    payload: Mapping[str, object],
    *,
    source: str | Path | None = None,
) -> None:
    label = str(source) if source is not None else "winner_selection.json"
    _require_non_empty_string(payload, "schema_version", label)
    artifact_type = _require_non_empty_string(payload, "artifact_type", label)
    if artifact_type != _WINNER_SELECTION_TYPE:
        _raise(f"{label}.artifact_type", f"must be `{_WINNER_SELECTION_TYPE}`")
    _require_non_empty_string(payload, "generated_at_utc", label)
    _require_non_empty_string(payload, "comparison_json_path", label)
    _require_non_empty_string(payload, "default_profile", label)
    _require_object(payload, "source_artifacts", label)
    policy = _require_object(payload, "decision_policy", label)
    _require_non_empty_string(policy, "decision_policy_id", f"{label}.decision_policy")
    _require_non_empty_string(policy, "policy_formula_text", f"{label}.decision_policy")
    _require_list(policy, "component_weights", f"{label}.decision_policy")

    _require_string(payload, "winner_recipe_id", label)
    for key in (
        "challenger_recipe_ids",
        "watchlist_recipe_ids",
        "rejected_recipe_ids",
        "decision_reasons",
        "challenger_reasons",
        "rejection_reasons",
        "next_actions",
        "missing_data_caveats",
    ):
        _require_string_list(
            _require_list(payload, key, label),
            f"{label}.{key}",
        )
    for key in (
        "decision_reasons_zh",
        "challenger_reasons_zh",
        "rejection_reasons_zh",
        "next_actions_zh",
    ):
        if key in payload and payload.get(key) is not None:
            _require_string_list(
                _require_list(payload, key, label),
                f"{label}.{key}",
            )

    score_rows = _require_list(payload, "score_table", label)
    for idx, row in enumerate(score_rows):
        item_label = f"{label}.score_table[{idx}]"
        if not isinstance(row, Mapping):
            _raise(item_label, "must be an object")
        item = cast(Mapping[str, object], row)
        _require_non_empty_string(item, "recipe_id", item_label)
        _validate_finite_number_or_none_if_present(
            item,
            "composite_score",
            item_label,
        )


def validate_next_step_recommendations_payload(
    payload: Mapping[str, object],
    *,
    source: str | Path | None = None,
) -> None:
    label = (
        str(source)
        if source is not None
        else "next_step_recommendations.json"
    )
    _require_non_empty_string(payload, "schema_version", label)
    artifact_type = _require_non_empty_string(payload, "artifact_type", label)
    if artifact_type != _NEXT_STEP_RECOMMENDATIONS_TYPE:
        _raise(
            f"{label}.artifact_type",
            f"must be `{_NEXT_STEP_RECOMMENDATIONS_TYPE}`",
        )
    _require_non_empty_string(payload, "generated_at_utc", label)
    _require_non_empty_string(payload, "comparison_json_path", label)
    _require_non_empty_string(payload, "default_profile", label)
    _require_object(payload, "source_artifacts", label)
    policy = _require_object(payload, "policy", label)
    _require_non_empty_string(policy, "policy_id", f"{label}.policy")
    _require_non_empty_string(policy, "policy_formula_text", f"{label}.policy")

    recommendations = _require_list(payload, "recommendations", label)
    for idx, row in enumerate(recommendations):
        item_label = f"{label}.recommendations[{idx}]"
        if not isinstance(row, Mapping):
            _raise(item_label, "must be an object")
        item = cast(Mapping[str, object], row)
        for key in (
            "recommendation_id",
            "category",
            "priority",
            "action",
            "rationale",
        ):
            _require_non_empty_string(item, key, item_label)
        for key in ("label_zh", "action_text_zh", "rationale_zh"):
            if key in item and item.get(key) is not None:
                _require_non_empty_string(item, key, item_label)
        _require_string_list(
            _require_list(item, "triggered_by", item_label),
            f"{item_label}.triggered_by",
        )
        _require_string_list(
            _require_list(item, "supporting_evidence", item_label),
            f"{item_label}.supporting_evidence",
        )

    _require_string_list(
        _require_list(payload, "summary", label),
        f"{label}.summary",
    )
    if "summary_zh" in payload and payload.get("summary_zh") is not None:
        _require_string_list(
            _require_list(payload, "summary_zh", label),
            f"{label}.summary_zh",
        )


def validate_artifact_load_diagnostics_payload(
    payload: Mapping[str, object],
    *,
    source: str | Path | None = None,
) -> None:
    label = (
        str(source)
        if source is not None
        else "artifact_load_diagnostics.json"
    )
    _require_non_empty_string(payload, "schema_version", label)
    artifact_type = _require_non_empty_string(payload, "artifact_type", label)
    if artifact_type != _ARTIFACT_LOAD_DIAGNOSTICS_TYPE:
        _raise(
            f"{label}.artifact_type",
            f"must be `{_ARTIFACT_LOAD_DIAGNOSTICS_TYPE}`",
        )
    _require_non_empty_string(payload, "generated_at_utc", label)
    _require_non_empty_string(payload, "comparison_json_path", label)
    _require_non_empty_string(payload, "default_profile", label)
    mode = _require_non_empty_string(payload, "artifact_load_mode", label)
    if mode not in {"permissive", "strict"}:
        _raise(
            f"{label}.artifact_load_mode",
            "must be `permissive` or `strict`",
        )
    _require_string_list(
        _require_list(payload, "artifact_load_policy_summary", label),
        f"{label}.artifact_load_policy_summary",
    )
    _require_object(payload, "source_artifacts", label)
    diagnostics = _require_list(payload, "diagnostics", label)
    for idx, row in enumerate(diagnostics):
        item_label = f"{label}.diagnostics[{idx}]"
        if not isinstance(row, Mapping):
            _raise(item_label, "must be an object")
        item = cast(Mapping[str, object], row)
        _require_non_empty_string(item, "code", item_label)
        severity = _require_non_empty_string(item, "severity", item_label)
        if severity not in {"warning", "error"}:
            _raise(
                f"{item_label}.severity",
                "must be `warning` or `error`",
            )
        _require_non_empty_string(item, "artifact_type", item_label)
        _require_non_empty_string(item, "object_scope", item_label)
        _require_non_empty_string(item, "message", item_label)
        diagnostic_mode = _require_non_empty_string(item, "mode", item_label)
        if diagnostic_mode not in {"permissive", "strict"}:
            _raise(
                f"{item_label}.mode",
                "must be `permissive` or `strict`",
            )
        _require_bool(item, "fallback_used", item_label)
        for key in ("path", "case_name", "profile_name", "remediation_hint"):
            value = item.get(key)
            if value is None:
                continue
            if not _is_non_empty_string(value):
                _raise(
                    f"{item_label}.{key}",
                    "must be a non-empty string or null",
                )


def validate_research_artifact_manifest_payload(
    payload: Mapping[str, object],
    *,
    source: str | Path | None = None,
) -> None:
    label = (
        str(source)
        if source is not None
        else "research_artifact_manifest.json"
    )
    _require_non_empty_string(payload, "schema_version", label)
    artifact_type = _require_non_empty_string(payload, "artifact_type", label)
    if artifact_type != _RESEARCH_ARTIFACT_MANIFEST_TYPE:
        _raise(
            f"{label}.artifact_type",
            f"must be `{_RESEARCH_ARTIFACT_MANIFEST_TYPE}`",
        )
    _require_non_empty_string(payload, "generated_at_utc", label)
    _require_non_empty_string(payload, "comparison_json_path", label)
    _require_non_empty_string(payload, "default_profile", label)

    entries = _require_list(payload, "artifact_entries", label)
    for idx, row in enumerate(entries):
        item_label = f"{label}.artifact_entries[{idx}]"
        if not isinstance(row, Mapping):
            _raise(item_label, "must be an object")
        item = cast(Mapping[str, object], row)
        _require_non_empty_string(item, "artifact_name", item_label)
        _require_non_empty_string(item, "artifact_type", item_label)
        artifact_layer = _require_non_empty_string(item, "artifact_layer", item_label)
        if artifact_layer not in {"canonical", "workflow", "governance"}:
            _raise(
                f"{item_label}.artifact_layer",
                "must be one of `canonical`, `workflow`, `governance`",
            )
        scope = _require_non_empty_string(item, "scope", item_label)
        if scope not in {"campaign", "profile", "case", "comparison"}:
            _raise(
                f"{item_label}.scope",
                "must be one of `campaign`, `profile`, `case`, `comparison`",
            )

        _require_non_empty_string(item, "producer_hint", item_label)
        _require_bool(item, "required_in_strict_mode", item_label)

        validation_status = _require_non_empty_string(
            item,
            "validation_status",
            item_label,
        )
        if validation_status not in {
            "valid",
            "invalid",
            "missing",
            "unresolved",
            "unchecked",
        }:
            _raise(
                f"{item_label}.validation_status",
                (
                    "must be one of `valid`, `invalid`, `missing`, "
                    "`unresolved`, `unchecked`"
                ),
            )

        for key in ("path", "case_name", "profile_name", "lineage_role"):
            value = item.get(key)
            if value is None:
                continue
            if not _is_non_empty_string(value):
                _raise(
                    f"{item_label}.{key}",
                    "must be a non-empty string or null",
                )

    summary = payload.get("summary")
    if summary is not None:
        if not isinstance(summary, Mapping):
            _raise(f"{label}.summary", "must be an object")
        summary_obj = cast(Mapping[str, object], summary)
        _require_int(summary_obj, "total_entries", f"{label}.summary")
        for key in ("by_layer", "by_artifact_type", "by_validation_status"):
            obj = _require_object(summary_obj, key, f"{label}.summary")
            for name, value in obj.items():
                if not _is_non_empty_string(name):
                    _raise(
                        f"{label}.summary.{key}",
                        "keys must be non-empty strings",
                    )
                if isinstance(value, bool) or not isinstance(value, int):
                    _raise(
                        f"{label}.summary.{key}[{name!r}]",
                        "must be an integer",
                    )


def _require_object(
    payload: Mapping[str, object],
    key: str,
    label: str,
) -> Mapping[str, object]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        _raise(f"{label}.{key}", "must be an object")
    return cast(Mapping[str, object], value)


def _require_list(payload: Mapping[str, object], key: str, label: str) -> list[object]:
    value = payload.get(key)
    if not isinstance(value, list):
        _raise(f"{label}.{key}", "must be a list")
    return cast(list[object], value)


def _require_non_empty_string(payload: Mapping[str, object], key: str, label: str) -> str:
    value = payload.get(key)
    if not _is_non_empty_string(value):
        _raise(f"{label}.{key}", "must be a non-empty string")
    return str(value).strip()


def _require_string(payload: Mapping[str, object], key: str, label: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        _raise(f"{label}.{key}", "must be a string")
    return value


def _require_int(payload: Mapping[str, object], key: str, label: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        _raise(f"{label}.{key}", "must be an integer")
    return cast(int, value)


def _require_bool(payload: Mapping[str, object], key: str, label: str) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        _raise(f"{label}.{key}", "must be a boolean")
    return cast(bool, value)


def _require_string_list(
    values: list[object],
    label: str,
    *,
    allow_empty: bool = True,
) -> None:
    if not allow_empty and not values:
        _raise(label, "must be a non-empty list of strings")
    for idx, item in enumerate(values):
        if not _is_non_empty_string(item):
            _raise(f"{label}[{idx}]", "must be a non-empty string")


def _validate_object_if_present(payload: Mapping[str, object], key: str, label: str) -> None:
    if key not in payload:
        return
    value = payload.get(key)
    if not isinstance(value, Mapping):
        _raise(f"{label}.{key}", "must be an object when provided")


def _validate_string_list_if_present(payload: Mapping[str, object], key: str, label: str) -> None:
    if key not in payload:
        return
    value = payload.get(key)
    if not isinstance(value, list):
        _raise(f"{label}.{key}", "must be a list of strings when provided")
    _require_string_list(cast(list[object], value), f"{label}.{key}")


def _validate_string_list_or_none_if_present(
    payload: Mapping[str, object],
    key: str,
    label: str,
) -> None:
    if key not in payload:
        return
    value = payload.get(key)
    if value is None:
        return
    if not isinstance(value, list):
        _raise(f"{label}.{key}", "must be a list of strings or null when provided")
    _require_string_list(cast(list[object], value), f"{label}.{key}")


def _validate_portfolio_robustness_summary_if_present(
    payload: Mapping[str, object],
    *,
    label: str,
) -> None:
    if "portfolio_robustness_summary" not in payload:
        return
    robustness = _require_object(payload, "portfolio_robustness_summary", label)
    _require_non_empty_string(robustness, "taxonomy_label", f"{label}.portfolio_robustness_summary")
    _validate_string_list_if_present(
        robustness,
        "support_reasons",
        f"{label}.portfolio_robustness_summary",
    )
    _validate_string_list_if_present(
        robustness,
        "fragility_reasons",
        f"{label}.portfolio_robustness_summary",
    )
    _validate_string_list_if_present(
        robustness,
        "scenario_sensitivity_notes",
        f"{label}.portfolio_robustness_summary",
    )
    for key in (
        "benchmark_relative_support_note",
        "cost_sensitivity_note",
        "concentration_turnover_risk_note",
    ):
        if key in robustness:
            _require_non_empty_string(
                robustness,
                key,
                f"{label}.portfolio_robustness_summary",
            )


def _validate_bool_or_none_if_present(payload: Mapping[str, object], key: str, label: str) -> None:
    if key not in payload:
        return
    value = payload.get(key)
    if value is not None and not isinstance(value, bool):
        _raise(f"{label}.{key}", "must be a boolean or null when provided")


def _validate_transition_reason_stat_list(value: object, label: str) -> None:
    if not isinstance(value, list):
        _raise(label, "must be a list")
    rows = cast(list[object], value)
    for idx, row in enumerate(rows):
        item_label = f"{label}[{idx}]"
        if not isinstance(row, Mapping):
            _raise(item_label, "must be an object")
        row_obj = cast(Mapping[str, object], row)
        _require_non_empty_string(row_obj, "reason", item_label)
        _require_int(row_obj, "count", item_label)
        _require_int(row_obj, "n_cases_with_label", item_label)
        prop = row_obj.get("proportion_of_label_cases")
        if not isinstance(prop, (int, float)) or isinstance(prop, bool):
            _raise(f"{item_label}.proportion_of_label_cases", "must be a number")


def _validate_transition_reason_delta_list(value: object, label: str) -> None:
    if not isinstance(value, list):
        _raise(label, "must be a list")
    rows = cast(list[object], value)
    for idx, row in enumerate(rows):
        item_label = f"{label}[{idx}]"
        if not isinstance(row, Mapping):
            _raise(item_label, "must be an object")
        row_obj = cast(Mapping[str, object], row)
        _require_non_empty_string(row_obj, "reason", item_label)
        for field in (
            "from_count",
            "from_n_cases_with_label",
            "to_count",
            "to_n_cases_with_label",
            "delta_count",
        ):
            _require_int(row_obj, field, item_label)
        for field in (
            "from_proportion_of_label_cases",
            "to_proportion_of_label_cases",
            "delta_proportion_of_label_cases",
        ):
            field_value = row_obj.get(field)
            if not isinstance(field_value, (int, float)) or isinstance(field_value, bool):
                _raise(f"{item_label}.{field}", "must be a number")


def _validate_int_or_none_if_present(payload: Mapping[str, object], key: str, label: str) -> None:
    if key not in payload:
        return
    value = payload.get(key)
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int):
        _raise(f"{label}.{key}", "must be an integer or null when provided")


def _validate_finite_number_or_none_if_present(
    payload: Mapping[str, object],
    key: str,
    label: str,
) -> None:
    if key not in payload:
        return
    value = payload.get(key)
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _raise(f"{label}.{key}", "must be a number or null when provided")


def _validate_time_value_rows(value: object, label: str) -> None:
    if not isinstance(value, list):
        _raise(label, "must be a list")
    rows = cast(list[object], value)
    for idx, row in enumerate(rows):
        item_label = f"{label}[{idx}]"
        if not isinstance(row, list):
            _raise(item_label, "must be a [timestamp, value] list")
        pair = cast(list[object], row)
        if len(pair) != 2:
            _raise(item_label, "must contain exactly two items")
        timestamp = pair[0]
        number = pair[1]
        if not _is_non_empty_string(timestamp):
            _raise(f"{item_label}[0]", "must be a non-empty timestamp string")
        if isinstance(number, bool) or not isinstance(number, (int, float)):
            _raise(f"{item_label}[1]", "must be a number")


def _is_non_empty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _raise(field: str, message: str) -> NoReturn:
    raise AlphaLabDataError(f"{field} {message}")
