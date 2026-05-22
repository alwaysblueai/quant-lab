from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_lab.experiment import run_factor_experiment
from alpha_lab.factors.momentum import momentum
from alpha_lab.reporting.campaign_triage import build_campaign_triage
from alpha_lab.reporting.factor_verdict import build_factor_verdict
from alpha_lab.reporting.level2_promotion import build_level2_promotion
from alpha_lab.reporting.neutralization_comparison import (
    MODERATE_WEAKENING_FLAG,
    PRESERVES_EVIDENCE_FLAG,
    build_raw_vs_neutralized_comparison,
)
from alpha_lab.reporting.uncertainty import compute_core_uncertainty
from alpha_lab.research_evaluation_config import (
    AVAILABLE_RESEARCH_EVALUATION_PROFILES,
    DEFAULT_RESEARCH_EVALUATION_CONFIG,
    CampaignTriageConfig,
    FactorVerdictConfig,
    Level2PortfolioValidationConfig,
    Level2PromotionConfig,
    NeutralizationComparisonConfig,
    RollingStabilityConfig,
    UncertaintyConfig,
    get_research_evaluation_config,
    get_research_evaluation_profile_intent,
)


def test_default_research_profile_preserves_legacy_thresholds() -> None:
    cfg = DEFAULT_RESEARCH_EVALUATION_CONFIG
    assert cfg.profile_name == "default_research"
    assert cfg.factor_verdict.min_eval_dates_basic == 20
    assert cfg.uncertainty.method == "normal"
    assert cfg.uncertainty.confidence_level == pytest.approx(0.95)
    assert cfg.uncertainty.normal_small_sample_use_t is True
    assert cfg.uncertainty.normal_small_sample_threshold == 30
    assert cfg.rolling_stability.rolling_window_size == 20
    assert cfg.neutralization_comparison.material_max_retention == pytest.approx(0.35)
    assert cfg.factor_verdict.ic_decay_warn_rebalance_ratio == pytest.approx(1.0)
    assert cfg.factor_verdict.ic_decay_block_rebalance_ratio == pytest.approx(2.0)
    assert cfg.campaign_triage.min_rolling_positive_share_stable == pytest.approx(0.60)
    assert cfg.campaign_triage.ic_decay_warn_rebalance_ratio == pytest.approx(1.0)
    assert cfg.campaign_triage.ic_decay_block_rebalance_ratio == pytest.approx(2.0)
    assert cfg.level2_promotion.min_rolling_positive_share_promote == pytest.approx(0.60)
    assert cfg.level2_portfolio_validation.default_weighting_method == "rank"
    assert cfg.level2_portfolio_validation.review_cost_rate == pytest.approx(0.0010)
    assert cfg.level2_portfolio_validation.sensitivity_material_spread_ratio_warn == pytest.approx(
        0.75
    )
    assert cfg.level2_portfolio_validation.sensitivity_stable_spread_ratio_max == pytest.approx(
        0.25
    )
    assert cfg.level2_portfolio_validation.robustness_fragile_min_severe_signal_count == 2
    assert cfg.level2_portfolio_validation.robustness_sensitive_min_severe_signal_count == 1
    assert cfg.level2_portfolio_validation.robustness_sensitive_min_material_signal_count == 1


def test_unknown_research_profile_raises() -> None:
    with pytest.raises(ValueError):
        get_research_evaluation_config("unknown_profile")


def test_research_evaluation_snapshot_includes_level2_robustness_cutoffs() -> None:
    snapshot = DEFAULT_RESEARCH_EVALUATION_CONFIG.to_audit_snapshot()
    uncertainty = snapshot["uncertainty"]
    assert uncertainty["normal_small_sample_use_t"] is True
    assert uncertainty["normal_small_sample_threshold"] == 30
    level2 = snapshot["level2_portfolio_validation"]
    assert level2["min_effective_names_warn"] == pytest.approx(8.0)
    assert level2["min_cost_adjusted_return_warn"] == pytest.approx(0.0)
    assert level2["sensitivity_sign_flip_pivot_return"] == pytest.approx(0.0)
    assert level2["sensitivity_material_spread_ratio_warn"] == pytest.approx(0.75)
    assert level2["sensitivity_stable_spread_ratio_max"] == pytest.approx(0.25)
    assert level2["robustness_fragile_min_severe_signal_count"] == 2
    assert level2["robustness_sensitive_min_severe_signal_count"] == 1
    assert level2["robustness_sensitive_min_material_signal_count"] == 1
    assert level2["robustness_needs_refinement_implies_sensitive"] is True
    factor_verdict = snapshot["factor_verdict"]
    campaign_triage = snapshot["campaign_triage"]
    assert factor_verdict["ic_decay_warn_rebalance_ratio"] == pytest.approx(1.0)
    assert factor_verdict["ic_decay_block_rebalance_ratio"] == pytest.approx(2.0)
    assert campaign_triage["ic_decay_warn_rebalance_ratio"] == pytest.approx(1.0)
    assert campaign_triage["ic_decay_block_rebalance_ratio"] == pytest.approx(2.0)


def test_profiles_are_registered_and_selectable() -> None:
    assert AVAILABLE_RESEARCH_EVALUATION_PROFILES == (
        "default_research",
        "exploratory_screening",
        "stricter_research",
    )
    for profile_name in AVAILABLE_RESEARCH_EVALUATION_PROFILES:
        cfg = get_research_evaluation_config(profile_name)
        assert cfg.profile_name == profile_name


def test_profile_intent_guidance_is_available() -> None:
    assert "candidate discovery" in get_research_evaluation_profile_intent("exploratory_screening")
    assert "conservative" in get_research_evaluation_profile_intent("stricter_research")
    assert "baseline" in get_research_evaluation_profile_intent("")


def test_profile_threshold_deltas_cover_triage_promotion_and_level2_guardrails() -> None:
    default_cfg = get_research_evaluation_config("default_research")
    exploratory_cfg = get_research_evaluation_config("exploratory_screening")
    stricter_cfg = get_research_evaluation_config("stricter_research")

    assert (
        exploratory_cfg.factor_verdict.min_eval_dates_basic
        < default_cfg.factor_verdict.min_eval_dates_basic
    )
    assert (
        stricter_cfg.factor_verdict.min_eval_dates_basic
        > default_cfg.factor_verdict.min_eval_dates_basic
    )
    assert (
        exploratory_cfg.level2_promotion.min_valid_ratio_promote
        < default_cfg.level2_promotion.min_valid_ratio_promote
    )
    assert exploratory_cfg.level2_promotion.require_strong_verdict_for_promote is False
    assert (
        stricter_cfg.level2_promotion.min_valid_ratio_promote
        > default_cfg.level2_promotion.min_valid_ratio_promote
    )
    assert exploratory_cfg.level2_portfolio_validation.run_for_non_promoted_cases is False
    assert default_cfg.level2_portfolio_validation.run_for_non_promoted_cases is False
    assert stricter_cfg.level2_portfolio_validation.max_mean_turnover_warn < (
        default_cfg.level2_portfolio_validation.max_mean_turnover_warn
    )


def test_profile_variants_change_factor_verdict_on_borderline_case() -> None:
    metrics = _borderline_profile_metrics()
    metrics["n_dates_used"] = 16

    exploratory = build_factor_verdict(
        metrics,
        thresholds=get_research_evaluation_config("exploratory_screening").factor_verdict,
    )
    default = build_factor_verdict(
        metrics,
        thresholds=get_research_evaluation_config("default_research").factor_verdict,
    )
    stricter = build_factor_verdict(
        metrics,
        thresholds=get_research_evaluation_config("stricter_research").factor_verdict,
    )

    assert exploratory.label == "Promising but fragile"
    assert default.label == "Fails basic robustness"
    assert stricter.label == "Fails basic robustness"


def test_exploratory_profile_streamlines_optional_single_factor_diagnostics() -> None:
    exploratory = get_research_evaluation_config("exploratory_screening")
    default = get_research_evaluation_config("default_research")

    assert exploratory.single_factor_diagnostics.compute_ic_decay is True
    assert exploratory.single_factor_diagnostics.run_param_sensitivity is False
    assert exploratory.single_factor_diagnostics.run_baseline_comparison is False
    assert exploratory.single_factor_diagnostics.run_lag_sensitivity is False
    assert exploratory.single_factor_diagnostics.diagnostic_max_dates == 500
    assert exploratory.single_factor_diagnostics.diagnostic_sample_mode == "latest"
    assert default.single_factor_diagnostics.compute_ic_decay is True
    assert default.single_factor_diagnostics.run_param_sensitivity is True
    assert default.single_factor_diagnostics.diagnostic_max_dates is None


def test_profile_variants_change_triage_and_promotion_on_borderline_case() -> None:
    metrics = _borderline_profile_metrics()

    exploratory_cfg = get_research_evaluation_config("exploratory_screening")
    default_cfg = get_research_evaluation_config("default_research")
    stricter_cfg = get_research_evaluation_config("stricter_research")

    triage_exploratory = build_campaign_triage(
        metrics,
        thresholds=exploratory_cfg.campaign_triage,
    )
    triage_default = build_campaign_triage(
        metrics,
        thresholds=default_cfg.campaign_triage,
    )
    triage_stricter = build_campaign_triage(
        metrics,
        thresholds=stricter_cfg.campaign_triage,
    )
    assert triage_exploratory.label == "Advance to Level 2"
    assert triage_default.label == "Needs refinement"
    assert triage_stricter.label == "Fragile / monitor"

    promotion_exploratory = build_level2_promotion(
        metrics,
        thresholds=exploratory_cfg.level2_promotion,
    )
    promotion_default = build_level2_promotion(
        metrics,
        thresholds=default_cfg.level2_promotion,
    )
    promotion_stricter = build_level2_promotion(
        metrics,
        thresholds=stricter_cfg.level2_promotion,
    )
    assert promotion_exploratory.label == "Promote to Level 2"
    assert promotion_default.label == "Hold for refinement"
    assert promotion_stricter.label == "Blocked from Level 2"


def test_factor_verdict_threshold_override_changes_classification() -> None:
    metrics = {
        "mean_ic": 0.03,
        "mean_rank_ic": 0.04,
        "ic_positive_rate": 0.62,
        "rank_ic_positive_rate": 0.64,
        "ic_valid_ratio": 0.90,
        "rank_ic_valid_ratio": 0.90,
        "mean_long_short_return": 0.003,
        "long_short_ir": 0.70,
        "long_short_return_per_turnover": 0.01,
        "mean_long_short_turnover": 0.20,
        "subperiod_ic_positive_share": 1.0,
        "subperiod_long_short_positive_share": 1.0,
        "eval_coverage_ratio_mean": 0.80,
        "eval_coverage_ratio_min": 0.70,
        "n_dates_used": 24,
        "instability_flags": [],
    }
    baseline = build_factor_verdict(metrics)
    stricter = build_factor_verdict(
        metrics,
        thresholds=FactorVerdictConfig(min_eval_dates_basic=30),
    )
    assert baseline.label != "Fails basic robustness"
    assert stricter.label == "Fails basic robustness"


def test_uncertainty_threshold_override_changes_flags() -> None:
    values = [0.01, 0.011, 0.012, 0.013, 0.014]
    baseline = compute_core_uncertainty(
        ic_values=values,
        rank_ic_values=values,
        long_short_values=values,
    )
    stricter = compute_core_uncertainty(
        ic_values=values,
        rank_ic_values=values,
        long_short_values=values,
        thresholds=UncertaintyConfig(relative_half_width_warn=0.05),
    )
    assert "ic_ci_wide" not in baseline.uncertainty_flags
    assert "ic_ci_wide" in stricter.uncertainty_flags


def test_uncertainty_config_bootstrap_roundtrip() -> None:
    cfg = UncertaintyConfig(
        method="bootstrap",
        bootstrap_resamples=180,
        bootstrap_confidence_level=0.90,
        bootstrap_random_seed=5,
    )
    assert cfg.method == "bootstrap"
    assert cfg.bootstrap_resamples == 180
    assert cfg.bootstrap_confidence_level == pytest.approx(0.90)
    assert cfg.bootstrap_random_seed == 5
    assert cfg.block_bootstrap_block_length == 5


def test_uncertainty_config_block_bootstrap_roundtrip() -> None:
    cfg = UncertaintyConfig(
        method="block_bootstrap",
        bootstrap_resamples=210,
        bootstrap_confidence_level=0.92,
        bootstrap_random_seed=23,
        block_bootstrap_block_length=7,
    )
    assert cfg.method == "block_bootstrap"
    assert cfg.bootstrap_resamples == 210
    assert cfg.bootstrap_confidence_level == pytest.approx(0.92)
    assert cfg.bootstrap_random_seed == 23
    assert cfg.block_bootstrap_block_length == 7


def test_neutralization_threshold_override_changes_interpretation() -> None:
    raw_metrics = _neutralization_metrics(0.040, 0.050, 0.0030, 0.80)
    neutralized_metrics = _neutralization_metrics(0.038, 0.048, 0.0028, 0.75)
    baseline = build_raw_vs_neutralized_comparison(raw_metrics, neutralized_metrics)
    stricter = build_raw_vs_neutralized_comparison(
        raw_metrics,
        neutralized_metrics,
        thresholds=NeutralizationComparisonConfig(preserve_min_retention=0.99),
    )
    assert PRESERVES_EVIDENCE_FLAG in baseline.interpretation_flags
    assert MODERATE_WEAKENING_FLAG in stricter.interpretation_flags


def test_campaign_triage_threshold_override_changes_label() -> None:
    metrics = {
        "factor_verdict": "Strong candidate",
        "mean_ic_ci_lower": 0.010,
        "mean_ic_ci_upper": 0.030,
        "mean_rank_ic_ci_lower": 0.012,
        "mean_rank_ic_ci_upper": 0.035,
        "mean_long_short_return_ci_lower": 0.001,
        "mean_long_short_return_ci_upper": 0.004,
        "rolling_ic_positive_share": 0.72,
        "rolling_rank_ic_positive_share": 0.70,
        "rolling_long_short_positive_share": 0.68,
        "rolling_ic_min_mean": 0.004,
        "rolling_rank_ic_min_mean": 0.003,
        "rolling_long_short_min_mean": 0.0003,
        "subperiod_ic_positive_share": 0.75,
        "subperiod_long_short_positive_share": 0.75,
        "coverage_mean": 0.82,
        "coverage_min": 0.70,
        "ic_valid_ratio": 0.90,
        "rank_ic_valid_ratio": 0.90,
        "mean_long_short_turnover": 0.35,
        "long_short_return_per_turnover": 0.009,
        "uncertainty_flags": [],
        "rolling_instability_flags": [],
        "neutralization_comparison_flags": ["neutralization preserves most evidence"],
    }
    baseline = build_campaign_triage(metrics)
    stricter = build_campaign_triage(
        metrics,
        thresholds=CampaignTriageConfig(supportive_ci_min_count=4),
    )
    assert baseline.label == "Advance to Level 2"
    assert stricter.label != "Advance to Level 2"


def test_rolling_stability_threshold_override_changes_instability_flags() -> None:
    result_default = run_factor_experiment(
        _make_prices(n_assets=6, n_days=28),
        _momentum_fn,
        allow_full_sample_evaluation=True,
    )
    result_override = run_factor_experiment(
        _make_prices(n_assets=6, n_days=28),
        _momentum_fn,
        rolling_stability_thresholds=RollingStabilityConfig(instability_short_eval_window_dates=5),
        allow_full_sample_evaluation=True,
    )
    assert "short_eval_window" in result_default.summary.instability_flags
    assert "short_eval_window" not in result_override.summary.instability_flags


def test_level2_promotion_threshold_override_changes_label() -> None:
    metrics = {
        "factor_verdict": "Strong candidate",
        "campaign_triage": "Advance to Level 2",
        "mean_ic_ci_lower": 0.010,
        "mean_ic_ci_upper": 0.030,
        "mean_rank_ic_ci_lower": 0.012,
        "mean_rank_ic_ci_upper": 0.035,
        "mean_long_short_return_ci_lower": 0.001,
        "mean_long_short_return_ci_upper": 0.004,
        "rolling_ic_positive_share": 0.72,
        "rolling_rank_ic_positive_share": 0.70,
        "rolling_long_short_positive_share": 0.68,
        "rolling_ic_min_mean": 0.004,
        "rolling_rank_ic_min_mean": 0.003,
        "rolling_long_short_min_mean": 0.0003,
        "subperiod_ic_positive_share": 0.75,
        "subperiod_long_short_positive_share": 0.75,
        "coverage_mean": 0.82,
        "coverage_min": 0.70,
        "ic_valid_ratio": 0.90,
        "rank_ic_valid_ratio": 0.90,
        "mean_long_short_turnover": 0.35,
        "long_short_return_per_turnover": 0.009,
        "neutralization_comparison_flags": ["neutralization preserves most evidence"],
    }
    baseline = build_level2_promotion(metrics)
    stricter = build_level2_promotion(
        metrics,
        thresholds=Level2PromotionConfig(min_supportive_ci_count_promote=4),
    )
    assert baseline.label == "Promote to Level 2"
    assert stricter.label != "Promote to Level 2"


def test_level2_portfolio_validation_config_override_roundtrip() -> None:
    cfg = Level2PortfolioValidationConfig(
        default_weighting_method="equal",
        holding_period_grid=(1, 2),
        transaction_cost_grid=(0.0, 0.0015),
        review_cost_rate=0.0015,
        min_benchmark_excess_return_warn=-0.0001,
        max_benchmark_tracking_error_warn=0.08,
        sensitivity_sign_flip_pivot_return=-0.0002,
        sensitivity_material_spread_ratio_warn=0.9,
        sensitivity_stable_spread_ratio_max=0.2,
        robustness_fragile_min_severe_signal_count=3,
        robustness_sensitive_min_severe_signal_count=2,
        robustness_sensitive_min_material_signal_count=2,
        robustness_needs_refinement_implies_sensitive=False,
    )
    assert cfg.default_weighting_method == "equal"
    assert cfg.holding_period_grid == (1, 2)
    assert cfg.transaction_cost_grid == (0.0, 0.0015)
    assert cfg.review_cost_rate == pytest.approx(0.0015)
    assert cfg.min_benchmark_excess_return_warn == pytest.approx(-0.0001)
    assert cfg.max_benchmark_tracking_error_warn == pytest.approx(0.08)
    assert cfg.sensitivity_sign_flip_pivot_return == pytest.approx(-0.0002)
    assert cfg.sensitivity_material_spread_ratio_warn == pytest.approx(0.9)
    assert cfg.sensitivity_stable_spread_ratio_max == pytest.approx(0.2)
    assert cfg.robustness_fragile_min_severe_signal_count == 3
    assert cfg.robustness_sensitive_min_severe_signal_count == 2
    assert cfg.robustness_sensitive_min_material_signal_count == 2
    assert cfg.robustness_needs_refinement_implies_sensitive is False


def _neutralization_metrics(
    mean_ic: float,
    mean_rank_ic: float,
    mean_ls: float,
    ic_ir: float,
) -> dict[str, object]:
    return {
        "mean_ic": mean_ic,
        "mean_rank_ic": mean_rank_ic,
        "mean_long_short_return": mean_ls,
        "ic_ir": ic_ir,
        "ic_valid_ratio": 0.9,
        "rank_ic_valid_ratio": 0.9,
        "eval_coverage_ratio_mean": 0.8,
        "eval_coverage_ratio_min": 0.7,
        "rolling_ic_positive_share": 0.7,
        "rolling_rank_ic_positive_share": 0.7,
        "rolling_long_short_positive_share": 0.7,
        "rolling_ic_min_mean": 0.002,
        "rolling_rank_ic_min_mean": 0.002,
        "rolling_long_short_min_mean": 0.0002,
        "uncertainty_flags": [],
        "rolling_instability_flags": [],
    }


def _borderline_profile_metrics() -> dict[str, object]:
    return {
        "factor_verdict": "Strong candidate",
        "mean_ic": 0.03,
        "mean_rank_ic": 0.03,
        "ic_positive_rate": 0.53,
        "rank_ic_positive_rate": 0.53,
        "ic_valid_ratio": 0.72,
        "rank_ic_valid_ratio": 0.72,
        "mean_long_short_return": 0.0020,
        "long_short_ir": 0.40,
        "mean_long_short_turnover": 0.70,
        "long_short_return_per_turnover": 0.0016,
        "subperiod_ic_positive_share": 0.62,
        "subperiod_long_short_positive_share": 0.62,
        "rolling_ic_positive_share": 0.58,
        "rolling_rank_ic_positive_share": 0.57,
        "rolling_long_short_positive_share": 0.56,
        "rolling_ic_min_mean": 0.0006,
        "rolling_rank_ic_min_mean": 0.0005,
        "rolling_long_short_min_mean": 0.0001,
        "coverage_mean": 0.62,
        "coverage_min": 0.42,
        "eval_coverage_ratio_mean": 0.62,
        "eval_coverage_ratio_min": 0.42,
        "mean_ic_ci_lower": 0.001,
        "mean_ic_ci_upper": 0.010,
        "mean_rank_ic_ci_lower": 0.001,
        "mean_rank_ic_ci_upper": 0.010,
        "mean_long_short_return_ci_lower": 0.0002,
        "mean_long_short_return_ci_upper": 0.0010,
        "uncertainty_flags": [],
        "instability_flags": [],
        "rolling_instability_flags": [],
        "neutralization_comparison_flags": ["neutralization preserves most evidence"],
        "campaign_triage": "Strong Level 1 candidate",
    }


def _make_prices(n_assets: int = 6, n_days: int = 30, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    assets = [f"A{i}" for i in range(n_assets)]
    rows: list[dict[str, object]] = []
    for asset in assets:
        price = 100.0
        for date in dates:
            price *= 1.0 + rng.normal(0.0, 0.01)
            rows.append({"date": date, "asset": asset, "close": price})
    return pd.DataFrame(rows)


def _momentum_fn(prices: pd.DataFrame) -> pd.DataFrame:
    return momentum(prices, window=5)
