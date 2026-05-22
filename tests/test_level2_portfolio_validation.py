from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import alpha_lab.reporting.level2_portfolio_validation as level2_portfolio_validation_module
from alpha_lab.experiment import run_factor_experiment
from alpha_lab.factors.momentum import momentum
from alpha_lab.reporting import summarise_experiment_result
from alpha_lab.reporting.level2_portfolio_validation import (
    LEVEL2_PORTFOLIO_ROBUSTNESS_TAXONOMY,
    LEVEL2_PORTFOLIO_VALIDATION_RECOMMENDATION_TAXONOMY,
    build_level2_portfolio_validation_bundle,
)
from alpha_lab.research_evaluation_config import (
    Level2PortfolioValidationConfig,
    get_research_evaluation_config,
)


def _make_prices(n_assets: int = 8, n_days: int = 80, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    assets = [f"A{i}" for i in range(n_assets)]
    rows: list[dict[str, object]] = []
    for asset in assets:
        price = 100.0
        for date in dates:
            price *= 1.0 + rng.normal(0.0, 0.012)
            rows.append({"date": date, "asset": asset, "close": price})
    return pd.DataFrame(rows)


def _momentum_fn(prices: pd.DataFrame) -> pd.DataFrame:
    return momentum(prices, window=5)


def _base_metrics() -> tuple[dict[str, object], object]:
    result = run_factor_experiment(
        _make_prices(), _momentum_fn, n_quantiles=5, horizon=5, allow_full_sample_evaluation=True
    )
    summary = summarise_experiment_result(result).iloc[0].to_dict()
    metrics = dict(summary)
    metrics["case_name"] = "demo_case"
    metrics["rebalance_frequency"] = "W"
    metrics["promotion_decision"] = "Promote to Level 2"
    metrics["promotion_reasons"] = ["evidence satisfies Level 2 promotion gate"]
    metrics["promotion_blockers"] = []
    metrics["neutralization_comparison_flags"] = ["neutralization preserves most evidence"]
    return metrics, result


def test_level2_portfolio_validation_bundle_builds_completed_payload_for_promoted_case() -> None:
    metrics, result = _base_metrics()
    bundle = build_level2_portfolio_validation_bundle(
        key_metrics=metrics,
        promotion_decision={
            "verdict": "Promote to Level 2",
            "reasons": metrics["promotion_reasons"],
            "blockers": metrics["promotion_blockers"],
            "source": "level2_promotion_gate",
        },
        case_context={
            "case_name": "demo_case",
            "package_type": "single_factor",
            "rebalance_frequency": "W",
        },
        experiment_result=result,
    )

    assert bundle.summary["validation_status"] == "completed"
    assert bundle.summary["recommendation"] in LEVEL2_PORTFOLIO_VALIDATION_RECOMMENDATION_TAXONOMY
    scenario_metrics = bundle.metrics.get("scenario_metrics")
    assert isinstance(scenario_metrics, list)
    assert len(scenario_metrics) > 0
    package = bundle.package
    assert package["package_type"] == "alpha_lab_level2_portfolio_validation_package"
    assert "key_portfolio_results" in package


def test_level2_portfolio_validation_skips_non_promoted_case_by_default() -> None:
    metrics, result = _base_metrics()
    metrics["promotion_decision"] = "Hold for refinement"
    metrics["promotion_blockers"] = ["blocked by fragile subperiod evidence"]

    bundle = build_level2_portfolio_validation_bundle(
        key_metrics=metrics,
        promotion_decision={
            "verdict": "Hold for refinement",
            "reasons": ["additional robustness evidence is required before Level 2"],
            "blockers": metrics["promotion_blockers"],
            "source": "level2_promotion_gate",
        },
        case_context={
            "case_name": "demo_case",
            "package_type": "single_factor",
            "rebalance_frequency": "W",
        },
        experiment_result=result,
    )

    assert bundle.summary["validation_status"] == "skipped_not_promoted"
    assert bundle.summary["recommendation"] == "Not evaluated (not promoted)"
    robustness = bundle.summary.get("portfolio_robustness_summary")
    assert isinstance(robustness, dict)
    assert robustness.get("taxonomy_label") == LEVEL2_PORTFOLIO_ROBUSTNESS_TAXONOMY[3]
    scenario_metrics = bundle.metrics.get("scenario_metrics")
    assert isinstance(scenario_metrics, list)
    assert not scenario_metrics


def test_level2_portfolio_validation_generates_benchmark_relative_metrics_when_available() -> None:
    metrics, result = _base_metrics()
    metrics["benchmark_name"] = "CSI300"
    metrics["benchmark_excess_return"] = 0.0004
    metrics["benchmark_information_ratio"] = 0.45
    metrics["benchmark_tracking_error"] = 0.020
    metrics["benchmark_long_short_excess_return"] = 0.0005

    bundle = build_level2_portfolio_validation_bundle(
        key_metrics=metrics,
        promotion_decision={
            "verdict": "Promote to Level 2",
            "reasons": metrics["promotion_reasons"],
            "blockers": metrics["promotion_blockers"],
            "source": "level2_promotion_gate",
        },
        case_context={
            "case_name": "demo_case",
            "package_type": "single_factor",
            "rebalance_frequency": "W",
        },
        experiment_result=result,
    )

    benchmark = bundle.metrics.get("benchmark_relative_evaluation")
    assert isinstance(benchmark, dict)
    assert benchmark.get("status") == "available"
    assert benchmark.get("assessment") == "supports_standalone_strength"
    assert benchmark.get("benchmark_excess_return") == 0.0004
    assert benchmark.get("benchmark_active_return") == 0.0004
    assert benchmark.get("benchmark_tracking_error") == 0.020
    assert benchmark.get("risk_flags") == []
    assert benchmark.get("interpretation") == "remains credible relative to benchmark"
    assert bundle.summary.get("benchmark_relative_status") == "available"
    assert bundle.summary.get("benchmark_relative_risks") == []


def test_level2_portfolio_validation_integrates_benchmark_relative_risks() -> None:
    metrics, result = _base_metrics()
    metrics["benchmark_name"] = "CSI300"
    metrics["benchmark_excess_return"] = -0.0002
    metrics["benchmark_information_ratio"] = -0.10
    metrics["benchmark_tracking_error"] = 0.08
    metrics["benchmark_relative_max_drawdown"] = 0.02

    bundle = build_level2_portfolio_validation_bundle(
        key_metrics=metrics,
        promotion_decision={
            "verdict": "Promote to Level 2",
            "reasons": metrics["promotion_reasons"],
            "blockers": metrics["promotion_blockers"],
            "source": "level2_promotion_gate",
        },
        case_context={
            "case_name": "demo_case",
            "package_type": "single_factor",
            "rebalance_frequency": "W",
        },
        experiment_result=result,
    )

    benchmark = bundle.metrics.get("benchmark_relative_evaluation")
    assert isinstance(benchmark, dict)
    assert benchmark.get("status") == "available"
    benchmark_risks = benchmark.get("risk_flags")
    assert isinstance(benchmark_risks, list)
    assert "excess return is weak relative to benchmark" in benchmark_risks
    assert "benchmark-relative risk is elevated" in benchmark_risks
    assert bundle.summary.get("recommendation") == "Needs portfolio refinement"
    major_risks = bundle.summary.get("major_risks")
    assert isinstance(major_risks, list)
    assert "excess return is weak relative to benchmark" in major_risks


def test_level2_portfolio_validation_marks_benchmark_relative_unavailable_when_missing() -> None:
    metrics, result = _base_metrics()
    bundle = build_level2_portfolio_validation_bundle(
        key_metrics=metrics,
        promotion_decision={
            "verdict": "Promote to Level 2",
            "reasons": metrics["promotion_reasons"],
            "blockers": metrics["promotion_blockers"],
            "source": "level2_promotion_gate",
        },
        case_context={
            "case_name": "demo_case",
            "package_type": "single_factor",
            "rebalance_frequency": "W",
        },
        experiment_result=result,
    )

    benchmark = bundle.metrics.get("benchmark_relative_evaluation")
    assert isinstance(benchmark, dict)
    assert benchmark.get("status") == "not_available"
    assert "not present in case evidence" in str(benchmark.get("note"))
    assert bundle.summary.get("benchmark_relative_status") == "not_available"


def test_level2_portfolio_validation_robustness_summary_classifies_robust_scenarios(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metrics, result = _base_metrics()
    metrics["benchmark_name"] = "CSI300"
    metrics["benchmark_excess_return"] = 0.0004
    metrics["benchmark_information_ratio"] = 0.35
    metrics["benchmark_tracking_error"] = 0.02
    monkeypatch.setattr(
        level2_portfolio_validation_module,
        "_run_scenarios",
        lambda **_: _scenario_rows(mode="robust"),
    )

    bundle = build_level2_portfolio_validation_bundle(
        key_metrics=metrics,
        promotion_decision={
            "verdict": "Promote to Level 2",
            "reasons": metrics["promotion_reasons"],
            "blockers": metrics["promotion_blockers"],
            "source": "level2_promotion_gate",
        },
        case_context={"case_name": "demo_case", "rebalance_frequency": "W"},
        experiment_result=result,
    )
    robustness = bundle.summary.get("portfolio_robustness_summary")
    assert isinstance(robustness, dict)
    assert robustness.get("taxonomy_label") == LEVEL2_PORTFOLIO_ROBUSTNESS_TAXONOMY[0]
    assert "baseline portfolio return is positive" in "; ".join(
        list(robustness.get("support_reasons") or [])
    )
    assert list(robustness.get("fragility_reasons") or []) == []


def test_level2_portfolio_validation_robustness_summary_classifies_sensitive_scenarios(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metrics, result = _base_metrics()
    monkeypatch.setattr(
        level2_portfolio_validation_module,
        "_run_scenarios",
        lambda **_: _scenario_rows(mode="sensitive"),
    )

    bundle = build_level2_portfolio_validation_bundle(
        key_metrics=metrics,
        promotion_decision={
            "verdict": "Promote to Level 2",
            "reasons": metrics["promotion_reasons"],
            "blockers": metrics["promotion_blockers"],
            "source": "level2_promotion_gate",
        },
        case_context={"case_name": "demo_case", "rebalance_frequency": "W"},
        experiment_result=result,
    )
    robustness = bundle.summary.get("portfolio_robustness_summary")
    assert isinstance(robustness, dict)
    assert robustness.get("taxonomy_label") == LEVEL2_PORTFOLIO_ROBUSTNESS_TAXONOMY[1]
    notes = "; ".join(list(robustness.get("scenario_sensitivity_notes") or []))
    assert "material" in notes


def test_level2_portfolio_validation_threshold_override_changes_robustness_taxonomy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metrics, result = _base_metrics()
    monkeypatch.setattr(
        level2_portfolio_validation_module,
        "_run_scenarios",
        lambda **_: _scenario_rows(mode="sensitive"),
    )

    baseline_bundle = build_level2_portfolio_validation_bundle(
        key_metrics=metrics,
        promotion_decision={
            "verdict": "Promote to Level 2",
            "reasons": metrics["promotion_reasons"],
            "blockers": metrics["promotion_blockers"],
            "source": "level2_promotion_gate",
        },
        case_context={"case_name": "demo_case", "rebalance_frequency": "W"},
        experiment_result=result,
        config=Level2PortfolioValidationConfig(default_holding_period=5),
    )
    tuned_bundle = build_level2_portfolio_validation_bundle(
        key_metrics=metrics,
        promotion_decision={
            "verdict": "Promote to Level 2",
            "reasons": metrics["promotion_reasons"],
            "blockers": metrics["promotion_blockers"],
            "source": "level2_promotion_gate",
        },
        case_context={"case_name": "demo_case", "rebalance_frequency": "W"},
        experiment_result=result,
        config=Level2PortfolioValidationConfig(
            default_holding_period=5,
            robustness_sensitive_min_material_signal_count=3,
        ),
    )

    baseline_robustness = baseline_bundle.summary.get("portfolio_robustness_summary")
    assert isinstance(baseline_robustness, dict)
    assert baseline_robustness.get("taxonomy_label") == LEVEL2_PORTFOLIO_ROBUSTNESS_TAXONOMY[1]
    tuned_robustness = tuned_bundle.summary.get("portfolio_robustness_summary")
    assert isinstance(tuned_robustness, dict)
    assert tuned_robustness.get("taxonomy_label") == LEVEL2_PORTFOLIO_ROBUSTNESS_TAXONOMY[0]


def test_level2_portfolio_validation_profile_guardrails_remain_coherent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metrics, result = _base_metrics()
    rows = _scenario_rows(mode="robust")
    for row in rows:
        row["mean_turnover"] = 0.72
    monkeypatch.setattr(
        level2_portfolio_validation_module,
        "_run_scenarios",
        lambda **_: rows,
    )

    default_cfg = get_research_evaluation_config("default_research").level2_portfolio_validation
    stricter_cfg = get_research_evaluation_config("stricter_research").level2_portfolio_validation

    default_bundle = build_level2_portfolio_validation_bundle(
        key_metrics=metrics,
        promotion_decision={
            "verdict": "Promote to Level 2",
            "reasons": metrics["promotion_reasons"],
            "blockers": metrics["promotion_blockers"],
            "source": "level2_promotion_gate",
        },
        case_context={"case_name": "demo_case", "rebalance_frequency": "W"},
        experiment_result=result,
        config=default_cfg,
    )
    stricter_bundle = build_level2_portfolio_validation_bundle(
        key_metrics=metrics,
        promotion_decision={
            "verdict": "Promote to Level 2",
            "reasons": metrics["promotion_reasons"],
            "blockers": metrics["promotion_blockers"],
            "source": "level2_promotion_gate",
        },
        case_context={"case_name": "demo_case", "rebalance_frequency": "W"},
        experiment_result=result,
        config=stricter_cfg,
    )

    assert default_bundle.summary.get("recommendation") == "Credible at portfolio level"
    assert stricter_bundle.summary.get("recommendation") == "Needs portfolio refinement"
    default_risks = list(default_bundle.summary.get("major_risks") or [])
    stricter_risks = list(stricter_bundle.summary.get("major_risks") or [])
    assert "high portfolio turnover under baseline assumptions" not in default_risks
    assert "high portfolio turnover under baseline assumptions" in stricter_risks


def test_level2_portfolio_validation_robustness_summary_classifies_fragile_scenarios(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metrics, result = _base_metrics()
    metrics["benchmark_name"] = "CSI300"
    metrics["benchmark_excess_return"] = -0.0002
    metrics["benchmark_information_ratio"] = -0.1
    metrics["benchmark_tracking_error"] = 0.08
    monkeypatch.setattr(
        level2_portfolio_validation_module,
        "_run_scenarios",
        lambda **_: _scenario_rows(mode="fragile"),
    )

    bundle = build_level2_portfolio_validation_bundle(
        key_metrics=metrics,
        promotion_decision={
            "verdict": "Promote to Level 2",
            "reasons": metrics["promotion_reasons"],
            "blockers": metrics["promotion_blockers"],
            "source": "level2_promotion_gate",
        },
        case_context={"case_name": "demo_case", "rebalance_frequency": "W"},
        experiment_result=result,
    )
    robustness = bundle.summary.get("portfolio_robustness_summary")
    assert isinstance(robustness, dict)
    assert robustness.get("taxonomy_label") == LEVEL2_PORTFOLIO_ROBUSTNESS_TAXONOMY[2]
    fragility = "; ".join(list(robustness.get("fragility_reasons") or []))
    assert "benchmark" in fragility
    assert "transaction-cost" in fragility


def test_run_scenarios_reuses_method_level_intermediates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counts = {
        "weights": 0,
        "turnover": 0,
        "active_turnover": 0,
        "concentration": 0,
        "simulate": 0,
        "cost_adjusted": 0,
    }

    def _fake_weights(*args, **kwargs) -> pd.DataFrame:
        del args, kwargs
        counts["weights"] += 1
        return pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
                "asset": ["A0", "A1"],
                "weight": [0.5, -0.5],
            }
        )

    def _fake_turnover(weights: pd.DataFrame) -> pd.DataFrame:
        del weights
        counts["turnover"] += 1
        return pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
                "portfolio_turnover": [np.nan, 0.25],
            }
        )

    def _fake_active_turnover(turnover: pd.DataFrame, *, rebalance_step: int) -> pd.DataFrame:
        del rebalance_step
        counts["active_turnover"] += 1
        return turnover

    def _fake_concentration(weights: pd.DataFrame) -> dict[str, float | None]:
        del weights
        counts["concentration"] += 1
        return {
            "max_abs_weight_mean": 0.5,
            "top5_abs_weight_share_mean": 1.0,
            "effective_names_mean": 2.0,
            "gross_exposure_mean": 1.0,
            "net_exposure_mean": 0.0,
        }

    def _fake_simulate(
        weights: pd.DataFrame,
        returns: pd.DataFrame,
        *,
        holding_period: int,
        rebalance_frequency: int,
    ) -> pd.DataFrame:
        del weights, returns, holding_period, rebalance_frequency
        counts["simulate"] += 1
        return pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
                "portfolio_return": [0.01, 0.02],
            }
        )

    def _fake_cost_adjusted(
        portfolio_return: pd.DataFrame,
        active_turnover: pd.DataFrame,
        *,
        cost_rate: float,
    ) -> pd.DataFrame:
        del portfolio_return, active_turnover, cost_rate
        counts["cost_adjusted"] += 1
        return pd.DataFrame({"adjusted_return": [0.01, 0.015]})

    monkeypatch.setattr(level2_portfolio_validation_module, "portfolio_weights", _fake_weights)
    monkeypatch.setattr(level2_portfolio_validation_module, "portfolio_turnover", _fake_turnover)
    monkeypatch.setattr(
        level2_portfolio_validation_module, "_active_turnover", _fake_active_turnover
    )
    monkeypatch.setattr(
        level2_portfolio_validation_module,
        "_concentration_metrics",
        _fake_concentration,
    )
    monkeypatch.setattr(
        level2_portfolio_validation_module,
        "simulate_portfolio_returns",
        _fake_simulate,
    )
    monkeypatch.setattr(
        level2_portfolio_validation_module,
        "portfolio_cost_adjusted_returns",
        _fake_cost_adjusted,
    )

    rows = level2_portfolio_validation_module._run_scenarios(
        eval_factor=pd.DataFrame(),
        eval_returns=pd.DataFrame(),
        methods=("equal", "rank"),
        holding_grid=(1, 3),
        cost_grid=(0.0, 0.001),
        rebalance_step=5,
        leg_k=3,
    )

    assert len(rows) == 4
    assert counts["weights"] == 2
    assert counts["turnover"] == 2
    assert counts["active_turnover"] == 2
    assert counts["concentration"] == 2
    assert counts["simulate"] == 4
    assert counts["cost_adjusted"] == 8


def _scenario_rows(mode: str) -> list[dict[str, object]]:
    rows = [
        {
            "weighting_method": "rank",
            "holding_period": 5,
            "rebalance_step": 5,
            "leg_size_k": 5,
            "n_return_dates": 32,
            "mean_portfolio_return": 0.0012,
            "portfolio_ir": 0.6,
            "portfolio_hit_rate": 0.62,
            "mean_turnover": 0.24,
            "max_abs_weight_mean": 0.08,
            "top5_abs_weight_share_mean": 0.35,
            "effective_names_mean": 22.0,
            "gross_exposure_mean": 2.0,
            "net_exposure_mean": 0.0,
            "mean_cost_adjusted_return_by_cost_rate": {
                "0.0000": 0.0012,
                "0.0010": 0.0010,
                "0.0020": 0.0008,
            },
        },
        {
            "weighting_method": "rank",
            "holding_period": 1,
            "rebalance_step": 5,
            "leg_size_k": 5,
            "n_return_dates": 32,
            "mean_portfolio_return": 0.0011,
            "portfolio_ir": 0.55,
            "portfolio_hit_rate": 0.61,
            "mean_turnover": 0.25,
            "max_abs_weight_mean": 0.08,
            "top5_abs_weight_share_mean": 0.35,
            "effective_names_mean": 22.0,
            "gross_exposure_mean": 2.0,
            "net_exposure_mean": 0.0,
            "mean_cost_adjusted_return_by_cost_rate": {
                "0.0000": 0.0011,
                "0.0010": 0.0009,
                "0.0020": 0.0007,
            },
        },
        {
            "weighting_method": "equal",
            "holding_period": 5,
            "rebalance_step": 5,
            "leg_size_k": 5,
            "n_return_dates": 32,
            "mean_portfolio_return": 0.00115,
            "portfolio_ir": 0.56,
            "portfolio_hit_rate": 0.61,
            "mean_turnover": 0.24,
            "max_abs_weight_mean": 0.08,
            "top5_abs_weight_share_mean": 0.35,
            "effective_names_mean": 21.0,
            "gross_exposure_mean": 2.0,
            "net_exposure_mean": 0.0,
            "mean_cost_adjusted_return_by_cost_rate": {
                "0.0000": 0.00115,
                "0.0010": 0.00095,
                "0.0020": 0.00075,
            },
        },
    ]
    if mode == "sensitive":
        rows[1]["mean_portfolio_return"] = 0.0001
        rows[1]["mean_cost_adjusted_return_by_cost_rate"] = {
            "0.0000": 0.0001,
            "0.0010": 0.00005,
            "0.0020": 0.00001,
        }
        rows[2]["mean_portfolio_return"] = 0.0002
        rows[2]["mean_cost_adjusted_return_by_cost_rate"] = {
            "0.0000": 0.0002,
            "0.0010": 0.0001,
            "0.0020": 0.00002,
        }
        return rows
    if mode == "fragile":
        rows[0]["mean_turnover"] = 1.05
        rows[0]["max_abs_weight_mean"] = 0.33
        rows[0]["effective_names_mean"] = 4.0
        rows[0]["mean_cost_adjusted_return_by_cost_rate"] = {
            "0.0000": 0.0008,
            "0.0010": -0.0002,
            "0.0020": -0.0006,
        }
        rows[1]["mean_portfolio_return"] = -0.0002
        rows[1]["mean_cost_adjusted_return_by_cost_rate"] = {
            "0.0000": -0.0002,
            "0.0010": -0.0004,
            "0.0020": -0.0007,
        }
        rows[2]["mean_portfolio_return"] = -0.0001
        return rows
    return rows
