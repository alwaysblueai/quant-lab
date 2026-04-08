"""Unified Level 1/2 threshold governance for research evaluation heuristics."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import TypedDict

from alpha_lab.exceptions import AlphaLabConfigError

_PROFILE_INTENTS: dict[str, str] = {
    "default_research": (
        "Balanced Level 1/2 baseline for routine research, triage, promotion, "
        "and portfolio-validation checks."
    ),
    "exploratory_screening": (
        "More permissive settings for broad candidate discovery; accepts noisier "
        "early signals and runs lighter Level 2 guardrails."
    ),
    "stricter_research": (
        "More conservative settings for stronger evidence standards and "
        "Level 2-readiness decisions."
    ),
}


class FactorVerdictSnapshot(TypedDict):
    min_eval_dates_basic: int
    min_valid_ratio_fail: float
    min_subperiod_share_fail: float
    min_rolling_positive_share_regime_warning: float


class UncertaintySnapshot(TypedDict):
    method: str
    confidence_level: float
    relative_half_width_warn: float
    bootstrap_resamples: int
    bootstrap_confidence_level: float | None
    bootstrap_random_seed: int | None
    block_bootstrap_block_length: int


class RollingStabilitySnapshot(TypedDict):
    rolling_window_size: int
    rolling_regime_min_positive_share: float
    rolling_regime_sign_flip_threshold: float


class NeutralizationComparisonSnapshot(TypedDict):
    preserve_min_retention: float
    material_max_retention: float
    exposure_corr_reduction_threshold: float


class CampaignTriageSnapshot(TypedDict):
    min_subperiod_positive_share_fail: float
    min_rolling_positive_share_stable: float
    min_coverage_mean_fail: float


class Level2PromotionSnapshot(TypedDict):
    min_subperiod_positive_share_block: float
    min_rolling_positive_share_promote: float
    min_coverage_mean_block: float
    min_valid_ratio_block: float


class Level2PortfolioValidationSnapshot(TypedDict):
    default_weighting_method: str
    holding_period_grid: list[int]
    transaction_cost_grid: list[float]
    review_cost_rate: float
    max_mean_turnover_warn: float
    min_cost_adjusted_return_warn: float
    max_single_name_weight_warn: float
    min_effective_names_warn: float
    min_benchmark_excess_return_warn: float
    min_benchmark_information_ratio_warn: float
    max_benchmark_tracking_error_warn: float
    max_benchmark_relative_drawdown_warn: float
    sensitivity_sign_flip_pivot_return: float
    sensitivity_material_spread_ratio_warn: float
    sensitivity_stable_spread_ratio_max: float
    robustness_fragile_min_severe_signal_count: int
    robustness_sensitive_min_severe_signal_count: int
    robustness_sensitive_min_material_signal_count: int
    robustness_needs_refinement_implies_sensitive: bool


class ResearchEvaluationAuditSnapshot(TypedDict):
    profile_name: str
    profile_intent: str
    factor_verdict: FactorVerdictSnapshot
    uncertainty: UncertaintySnapshot
    rolling_stability: RollingStabilitySnapshot
    neutralization_comparison: NeutralizationComparisonSnapshot
    campaign_triage: CampaignTriageSnapshot
    level2_promotion: Level2PromotionSnapshot
    level2_portfolio_validation: Level2PortfolioValidationSnapshot


def get_research_evaluation_profile_intent(profile_name: str) -> str:
    """Return one-line operator guidance for an evaluation profile."""
    normalized = profile_name.strip().lower()
    if not normalized:
        normalized = "default_research"
    return _PROFILE_INTENTS.get(
        normalized,
        (
            "Custom profile guidance unavailable; inspect audit snapshot to review "
            "active thresholds."
        ),
    )


@dataclass(frozen=True)
class FactorVerdictConfig:
    """Thresholds for factor verdict classification."""

    min_eval_dates_basic: int = 20
    min_eval_dates_preferred: int = 30
    min_valid_ratio_strong: float = 0.80
    min_valid_ratio_fail: float = 0.60
    min_sign_positive_rate: float = 0.55
    weak_sign_positive_rate: float = 0.50
    min_subperiod_share_strong: float = 2.0 / 3.0
    min_subperiod_share_fail: float = 0.50
    min_coverage_mean_strong: float = 0.70
    min_coverage_mean_warn: float = 0.60
    min_coverage_mean_fail: float = 0.50
    min_coverage_min_strong: float = 0.50
    min_coverage_min_fail: float = 0.30
    high_turnover: float = 0.80
    min_return_per_turnover: float = 0.0
    high_turnover_low_efficiency_rpt: float = 0.002
    neutralization_material_corr_reduction: float = 0.20
    uncertainty_overlap_zero_fail_count: int = 2
    min_rolling_positive_share_persistent: float = 0.60
    min_rolling_positive_share_regime_warning: float = 0.50


@dataclass(frozen=True)
class UncertaintyConfig:
    """Thresholds for uncertainty warning flags."""

    method: str = "normal"
    confidence_level: float = 0.95
    relative_half_width_warn: float = 1.0
    min_abs_mean_for_relative_width: float = 1e-6
    bootstrap_resamples: int = 400
    bootstrap_confidence_level: float | None = None
    bootstrap_random_seed: int | None = 7
    block_bootstrap_block_length: int = 5


@dataclass(frozen=True)
class RollingStabilityConfig:
    """Thresholds for rolling stability diagnostics and instability flags."""

    rolling_window_size: int = 20
    rolling_regime_min_positive_share: float = 0.60
    rolling_regime_sign_flip_threshold: float = 0.45
    rolling_regime_min_windows_for_sign_flip: int = 6
    instability_short_eval_window_dates: int = 30
    instability_ic_valid_ratio_min: float = 0.80
    instability_rank_ic_valid_ratio_min: float = 0.80
    instability_ic_positive_rate_min: float = 0.50
    instability_subperiod_positive_share_min: float = 2.0 / 3.0
    instability_eval_coverage_ratio_mean_min: float = 0.60
    instability_high_turnover: float = 0.80
    instability_high_turnover_negative_spread_max_return: float = 0.0
    instability_long_short_ir_min: float = 0.0


@dataclass(frozen=True)
class NeutralizationComparisonConfig:
    """Thresholds for raw-vs-neutralized interpretation heuristics."""

    preserve_mean_ic_loss_max: float = 0.005
    preserve_mean_rank_ic_loss_max: float = 0.005
    preserve_long_short_loss_max: float = 0.0005
    preserve_min_retention: float = 0.75
    material_mean_ic_loss: float = 0.015
    material_mean_rank_ic_loss: float = 0.015
    material_long_short_loss: float = 0.0015
    material_ic_ir_loss: float = 0.25
    material_max_retention: float = 0.35
    material_loss_hit_count: int = 2
    material_loss_hit_count_with_core_shift: int = 1
    exposure_raw_positive_core_min: int = 2
    exposure_neutralized_positive_core_max: int = 1
    exposure_corr_reduction_threshold: float = 0.20
    stability_share_gain_min: float = 0.05
    stability_worst_mean_gain_min: float = 0.0


@dataclass(frozen=True)
class CampaignTriageConfig:
    """Thresholds for campaign triage and campaign ranking metadata."""

    min_subperiod_positive_share_fail: float = 0.50
    min_subperiod_positive_share_stable: float = 2.0 / 3.0
    min_rolling_positive_share_stable: float = 0.60
    min_rolling_positive_share_fragile: float = 0.50
    min_coverage_mean_fail: float = 0.50
    min_coverage_min_fail: float = 0.30
    min_coverage_mean_warn: float = 0.65
    min_coverage_min_warn: float = 0.45
    min_valid_ratio_fail: float = 0.60
    min_return_per_turnover: float = 0.0
    high_turnover: float = 0.80
    high_turnover_low_efficiency_rpt: float = 0.002
    supportive_ci_min_count: int = 2
    uncertainty_overlap_fragile_min_count: int = 1
    rolling_worst_mean_positive_min: float = 0.0
    fragile_signal_count_for_strong_candidate_max: int = 1
    fragile_signal_count_for_fragile_min: int = 2


@dataclass(frozen=True)
class Level2PromotionConfig:
    """Thresholds for Level 2 promotion gate classification."""

    min_subperiod_positive_share_block: float = 0.50
    min_subperiod_positive_share_promote: float = 2.0 / 3.0
    min_rolling_positive_share_block: float = 0.50
    min_rolling_positive_share_promote: float = 0.60
    rolling_worst_mean_block_max: float = 0.0
    rolling_worst_mean_promote_min: float = 0.0
    min_coverage_mean_block: float = 0.50
    min_coverage_min_block: float = 0.30
    min_coverage_mean_promote: float = 0.65
    min_coverage_min_promote: float = 0.45
    min_valid_ratio_block: float = 0.60
    min_valid_ratio_promote: float = 0.75
    min_supportive_ci_count_promote: int = 2
    uncertainty_overlap_block_min_count: int = 2
    high_turnover: float = 0.80
    min_return_per_turnover: float = 0.0
    high_turnover_low_efficiency_rpt: float = 0.002
    require_strong_verdict_for_promote: bool = True
    require_neutralization_support_for_promote: bool = True
    rolling_instability_is_blocker: bool = True


@dataclass(frozen=True)
class Level2PortfolioValidationConfig:
    """Protocol settings and guardrails for Level 2 portfolio validation."""

    run_for_non_promoted_cases: bool = False
    default_weighting_method: str = "rank"
    weighting_methods: tuple[str, ...] = ("equal", "rank", "score")
    default_holding_period: int = 1
    holding_period_grid: tuple[int, ...] = (1, 3, 5)
    transaction_cost_grid: tuple[float, ...] = (0.0, 0.0005, 0.0010, 0.0020)
    review_cost_rate: float = 0.0010
    max_mean_turnover_warn: float = 0.80
    min_cost_adjusted_return_warn: float = 0.0
    max_single_name_weight_warn: float = 0.20
    min_effective_names_warn: float = 8.0
    min_benchmark_excess_return_warn: float = 0.0
    min_benchmark_information_ratio_warn: float = 0.0
    max_benchmark_tracking_error_warn: float = 0.05
    max_benchmark_relative_drawdown_warn: float = 0.0
    sensitivity_sign_flip_pivot_return: float = 0.0
    sensitivity_material_spread_ratio_warn: float = 0.75
    sensitivity_stable_spread_ratio_max: float = 0.25
    robustness_fragile_min_severe_signal_count: int = 2
    robustness_sensitive_min_severe_signal_count: int = 1
    robustness_sensitive_min_material_signal_count: int = 1
    robustness_needs_refinement_implies_sensitive: bool = True


@dataclass(frozen=True)
class ResearchEvaluationConfig:
    """Unified research-evaluation governance profile for Level 1/2 workflows."""

    profile_name: str = "default_research"
    factor_verdict: FactorVerdictConfig = field(default_factory=FactorVerdictConfig)
    uncertainty: UncertaintyConfig = field(default_factory=UncertaintyConfig)
    rolling_stability: RollingStabilityConfig = field(default_factory=RollingStabilityConfig)
    neutralization_comparison: NeutralizationComparisonConfig = field(
        default_factory=NeutralizationComparisonConfig
    )
    campaign_triage: CampaignTriageConfig = field(default_factory=CampaignTriageConfig)
    level2_promotion: Level2PromotionConfig = field(default_factory=Level2PromotionConfig)
    level2_portfolio_validation: Level2PortfolioValidationConfig = field(
        default_factory=Level2PortfolioValidationConfig
    )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def to_audit_snapshot(self) -> ResearchEvaluationAuditSnapshot:
        """Concise snapshot for reports and package metadata."""
        return {
            "profile_name": self.profile_name,
            "profile_intent": get_research_evaluation_profile_intent(self.profile_name),
            "factor_verdict": {
                "min_eval_dates_basic": self.factor_verdict.min_eval_dates_basic,
                "min_valid_ratio_fail": self.factor_verdict.min_valid_ratio_fail,
                "min_subperiod_share_fail": self.factor_verdict.min_subperiod_share_fail,
                "min_rolling_positive_share_regime_warning": (
                    self.factor_verdict.min_rolling_positive_share_regime_warning
                ),
            },
            "uncertainty": {
                "method": self.uncertainty.method,
                "confidence_level": self.uncertainty.confidence_level,
                "relative_half_width_warn": self.uncertainty.relative_half_width_warn,
                "bootstrap_resamples": self.uncertainty.bootstrap_resamples,
                "bootstrap_confidence_level": self.uncertainty.bootstrap_confidence_level,
                "bootstrap_random_seed": self.uncertainty.bootstrap_random_seed,
                "block_bootstrap_block_length": (
                    self.uncertainty.block_bootstrap_block_length
                ),
            },
            "rolling_stability": {
                "rolling_window_size": self.rolling_stability.rolling_window_size,
                "rolling_regime_min_positive_share": (
                    self.rolling_stability.rolling_regime_min_positive_share
                ),
                "rolling_regime_sign_flip_threshold": (
                    self.rolling_stability.rolling_regime_sign_flip_threshold
                ),
            },
            "neutralization_comparison": {
                "preserve_min_retention": (
                    self.neutralization_comparison.preserve_min_retention
                ),
                "material_max_retention": (
                    self.neutralization_comparison.material_max_retention
                ),
                "exposure_corr_reduction_threshold": (
                    self.neutralization_comparison.exposure_corr_reduction_threshold
                ),
            },
            "campaign_triage": {
                "min_subperiod_positive_share_fail": (
                    self.campaign_triage.min_subperiod_positive_share_fail
                ),
                "min_rolling_positive_share_stable": (
                    self.campaign_triage.min_rolling_positive_share_stable
                ),
                "min_coverage_mean_fail": self.campaign_triage.min_coverage_mean_fail,
            },
            "level2_promotion": {
                "min_subperiod_positive_share_block": (
                    self.level2_promotion.min_subperiod_positive_share_block
                ),
                "min_rolling_positive_share_promote": (
                    self.level2_promotion.min_rolling_positive_share_promote
                ),
                "min_coverage_mean_block": self.level2_promotion.min_coverage_mean_block,
                "min_valid_ratio_block": self.level2_promotion.min_valid_ratio_block,
            },
            "level2_portfolio_validation": {
                "default_weighting_method": (
                    self.level2_portfolio_validation.default_weighting_method
                ),
                "holding_period_grid": list(self.level2_portfolio_validation.holding_period_grid),
                "transaction_cost_grid": list(
                    self.level2_portfolio_validation.transaction_cost_grid
                ),
                "review_cost_rate": self.level2_portfolio_validation.review_cost_rate,
                "max_mean_turnover_warn": (
                    self.level2_portfolio_validation.max_mean_turnover_warn
                ),
                "min_cost_adjusted_return_warn": (
                    self.level2_portfolio_validation.min_cost_adjusted_return_warn
                ),
                "max_single_name_weight_warn": (
                    self.level2_portfolio_validation.max_single_name_weight_warn
                ),
                "min_effective_names_warn": (
                    self.level2_portfolio_validation.min_effective_names_warn
                ),
                "min_benchmark_excess_return_warn": (
                    self.level2_portfolio_validation.min_benchmark_excess_return_warn
                ),
                "min_benchmark_information_ratio_warn": (
                    self.level2_portfolio_validation.min_benchmark_information_ratio_warn
                ),
                "max_benchmark_tracking_error_warn": (
                    self.level2_portfolio_validation.max_benchmark_tracking_error_warn
                ),
                "max_benchmark_relative_drawdown_warn": (
                    self.level2_portfolio_validation.max_benchmark_relative_drawdown_warn
                ),
                "sensitivity_sign_flip_pivot_return": (
                    self.level2_portfolio_validation.sensitivity_sign_flip_pivot_return
                ),
                "sensitivity_material_spread_ratio_warn": (
                    self.level2_portfolio_validation.sensitivity_material_spread_ratio_warn
                ),
                "sensitivity_stable_spread_ratio_max": (
                    self.level2_portfolio_validation.sensitivity_stable_spread_ratio_max
                ),
                "robustness_fragile_min_severe_signal_count": (
                    self.level2_portfolio_validation.robustness_fragile_min_severe_signal_count
                ),
                "robustness_sensitive_min_severe_signal_count": (
                    self.level2_portfolio_validation.robustness_sensitive_min_severe_signal_count
                ),
                "robustness_sensitive_min_material_signal_count": (
                    self.level2_portfolio_validation.robustness_sensitive_min_material_signal_count
                ),
                "robustness_needs_refinement_implies_sensitive": (
                    self.level2_portfolio_validation.robustness_needs_refinement_implies_sensitive
                ),
            },
        }


def _build_default_research_profile() -> ResearchEvaluationConfig:
    return ResearchEvaluationConfig(profile_name="default_research")


def _build_exploratory_screening_profile() -> ResearchEvaluationConfig:
    return ResearchEvaluationConfig(
        profile_name="exploratory_screening",
        factor_verdict=FactorVerdictConfig(
            min_eval_dates_basic=15,
            min_eval_dates_preferred=24,
            min_valid_ratio_strong=0.75,
            min_valid_ratio_fail=0.50,
            min_sign_positive_rate=0.52,
            weak_sign_positive_rate=0.48,
            min_subperiod_share_strong=0.60,
            min_subperiod_share_fail=0.45,
            min_coverage_mean_strong=0.65,
            min_coverage_mean_warn=0.55,
            min_coverage_mean_fail=0.45,
            min_coverage_min_strong=0.45,
            min_coverage_min_fail=0.25,
            high_turnover=0.90,
            min_return_per_turnover=-0.0005,
            high_turnover_low_efficiency_rpt=0.0015,
            neutralization_material_corr_reduction=0.30,
            uncertainty_overlap_zero_fail_count=3,
            min_rolling_positive_share_persistent=0.55,
            min_rolling_positive_share_regime_warning=0.45,
        ),
        uncertainty=UncertaintyConfig(
            confidence_level=0.90,
            relative_half_width_warn=1.25,
        ),
        rolling_stability=RollingStabilityConfig(
            rolling_regime_min_positive_share=0.55,
            rolling_regime_sign_flip_threshold=0.55,
            rolling_regime_min_windows_for_sign_flip=8,
            instability_short_eval_window_dates=20,
            instability_ic_valid_ratio_min=0.70,
            instability_rank_ic_valid_ratio_min=0.70,
            instability_ic_positive_rate_min=0.45,
            instability_subperiod_positive_share_min=0.55,
            instability_eval_coverage_ratio_mean_min=0.50,
            instability_high_turnover=0.90,
            instability_high_turnover_negative_spread_max_return=-0.0005,
            instability_long_short_ir_min=-0.05,
        ),
        neutralization_comparison=NeutralizationComparisonConfig(
            preserve_mean_ic_loss_max=0.007,
            preserve_mean_rank_ic_loss_max=0.007,
            preserve_long_short_loss_max=0.0008,
            preserve_min_retention=0.65,
            material_mean_ic_loss=0.020,
            material_mean_rank_ic_loss=0.020,
            material_long_short_loss=0.0020,
            material_ic_ir_loss=0.35,
            material_max_retention=0.25,
            material_loss_hit_count=3,
            material_loss_hit_count_with_core_shift=2,
            exposure_corr_reduction_threshold=0.30,
            stability_share_gain_min=0.03,
            stability_worst_mean_gain_min=-0.0002,
        ),
        campaign_triage=CampaignTriageConfig(
            min_subperiod_positive_share_fail=0.45,
            min_subperiod_positive_share_stable=0.60,
            min_rolling_positive_share_stable=0.55,
            min_rolling_positive_share_fragile=0.45,
            min_coverage_mean_fail=0.45,
            min_coverage_min_fail=0.25,
            min_coverage_mean_warn=0.60,
            min_coverage_min_warn=0.40,
            min_valid_ratio_fail=0.50,
            min_return_per_turnover=-0.0005,
            high_turnover=0.90,
            high_turnover_low_efficiency_rpt=0.0015,
            supportive_ci_min_count=1,
            uncertainty_overlap_fragile_min_count=2,
            rolling_worst_mean_positive_min=-0.0002,
            fragile_signal_count_for_strong_candidate_max=2,
            fragile_signal_count_for_fragile_min=3,
        ),
        level2_promotion=Level2PromotionConfig(
            min_subperiod_positive_share_block=0.45,
            min_subperiod_positive_share_promote=0.60,
            min_rolling_positive_share_block=0.45,
            min_rolling_positive_share_promote=0.55,
            rolling_worst_mean_block_max=-0.0005,
            rolling_worst_mean_promote_min=-0.0002,
            min_coverage_mean_block=0.45,
            min_coverage_min_block=0.25,
            min_coverage_mean_promote=0.60,
            min_coverage_min_promote=0.40,
            min_valid_ratio_block=0.50,
            min_valid_ratio_promote=0.70,
            min_supportive_ci_count_promote=1,
            uncertainty_overlap_block_min_count=3,
            high_turnover=0.90,
            min_return_per_turnover=-0.0005,
            high_turnover_low_efficiency_rpt=0.0015,
            require_neutralization_support_for_promote=False,
            rolling_instability_is_blocker=False,
        ),
        level2_portfolio_validation=Level2PortfolioValidationConfig(
            run_for_non_promoted_cases=True,
            holding_period_grid=(1, 3),
            transaction_cost_grid=(0.0, 0.0005, 0.0010),
            review_cost_rate=0.0005,
            max_mean_turnover_warn=0.95,
            min_cost_adjusted_return_warn=-0.0005,
            max_single_name_weight_warn=0.25,
            min_effective_names_warn=6.0,
        ),
    )


def _build_stricter_research_profile() -> ResearchEvaluationConfig:
    return ResearchEvaluationConfig(
        profile_name="stricter_research",
        factor_verdict=FactorVerdictConfig(
            min_eval_dates_basic=30,
            min_eval_dates_preferred=45,
            min_valid_ratio_strong=0.85,
            min_valid_ratio_fail=0.70,
            min_sign_positive_rate=0.60,
            weak_sign_positive_rate=0.55,
            min_subperiod_share_strong=0.75,
            min_subperiod_share_fail=0.60,
            min_coverage_mean_strong=0.75,
            min_coverage_mean_warn=0.65,
            min_coverage_mean_fail=0.60,
            min_coverage_min_strong=0.55,
            min_coverage_min_fail=0.40,
            high_turnover=0.70,
            min_return_per_turnover=0.0005,
            high_turnover_low_efficiency_rpt=0.0030,
            neutralization_material_corr_reduction=0.15,
            uncertainty_overlap_zero_fail_count=1,
            min_rolling_positive_share_persistent=0.65,
            min_rolling_positive_share_regime_warning=0.55,
        ),
        uncertainty=UncertaintyConfig(
            confidence_level=0.99,
            relative_half_width_warn=0.75,
        ),
        rolling_stability=RollingStabilityConfig(
            rolling_regime_min_positive_share=0.65,
            rolling_regime_sign_flip_threshold=0.35,
            rolling_regime_min_windows_for_sign_flip=5,
            instability_short_eval_window_dates=40,
            instability_ic_valid_ratio_min=0.85,
            instability_rank_ic_valid_ratio_min=0.85,
            instability_ic_positive_rate_min=0.55,
            instability_subperiod_positive_share_min=0.75,
            instability_eval_coverage_ratio_mean_min=0.65,
            instability_high_turnover=0.70,
            instability_high_turnover_negative_spread_max_return=0.0005,
            instability_long_short_ir_min=0.05,
        ),
        neutralization_comparison=NeutralizationComparisonConfig(
            preserve_mean_ic_loss_max=0.003,
            preserve_mean_rank_ic_loss_max=0.003,
            preserve_long_short_loss_max=0.0003,
            preserve_min_retention=0.85,
            material_mean_ic_loss=0.010,
            material_mean_rank_ic_loss=0.010,
            material_long_short_loss=0.0010,
            material_ic_ir_loss=0.20,
            material_max_retention=0.45,
            material_loss_hit_count=1,
            material_loss_hit_count_with_core_shift=1,
            exposure_corr_reduction_threshold=0.15,
            stability_share_gain_min=0.08,
            stability_worst_mean_gain_min=0.0002,
        ),
        campaign_triage=CampaignTriageConfig(
            min_subperiod_positive_share_fail=0.55,
            min_subperiod_positive_share_stable=0.75,
            min_rolling_positive_share_stable=0.65,
            min_rolling_positive_share_fragile=0.55,
            min_coverage_mean_fail=0.55,
            min_coverage_min_fail=0.35,
            min_coverage_mean_warn=0.70,
            min_coverage_min_warn=0.50,
            min_valid_ratio_fail=0.70,
            min_return_per_turnover=0.0005,
            high_turnover=0.70,
            high_turnover_low_efficiency_rpt=0.0030,
            supportive_ci_min_count=3,
            uncertainty_overlap_fragile_min_count=1,
            rolling_worst_mean_positive_min=0.0002,
            fragile_signal_count_for_strong_candidate_max=0,
            fragile_signal_count_for_fragile_min=1,
        ),
        level2_promotion=Level2PromotionConfig(
            min_subperiod_positive_share_block=0.55,
            min_subperiod_positive_share_promote=0.75,
            min_rolling_positive_share_block=0.55,
            min_rolling_positive_share_promote=0.65,
            rolling_worst_mean_block_max=0.0002,
            rolling_worst_mean_promote_min=0.0002,
            min_coverage_mean_block=0.55,
            min_coverage_min_block=0.35,
            min_coverage_mean_promote=0.72,
            min_coverage_min_promote=0.50,
            min_valid_ratio_block=0.70,
            min_valid_ratio_promote=0.85,
            min_supportive_ci_count_promote=3,
            uncertainty_overlap_block_min_count=1,
            high_turnover=0.70,
            min_return_per_turnover=0.0005,
            high_turnover_low_efficiency_rpt=0.0030,
        ),
        level2_portfolio_validation=Level2PortfolioValidationConfig(
            transaction_cost_grid=(0.0, 0.0005, 0.0010, 0.0020, 0.0030),
            review_cost_rate=0.0015,
            max_mean_turnover_warn=0.65,
            min_cost_adjusted_return_warn=0.0005,
            max_single_name_weight_warn=0.15,
            min_effective_names_warn=10.0,
        ),
    )


_PROFILE_BUILDERS: dict[str, Callable[[], ResearchEvaluationConfig]] = {
    "default_research": _build_default_research_profile,
    "exploratory_screening": _build_exploratory_screening_profile,
    "stricter_research": _build_stricter_research_profile,
}

AVAILABLE_RESEARCH_EVALUATION_PROFILES: tuple[str, ...] = tuple(
    sorted(_PROFILE_BUILDERS)
)


def get_research_evaluation_config(
    profile_name: str = "default_research",
) -> ResearchEvaluationConfig:
    """Return a config profile by name."""
    normalized = profile_name.strip().lower()
    if not normalized:
        normalized = "default_research"
    builder = _PROFILE_BUILDERS.get(normalized)
    if builder is None:
        raise AlphaLabConfigError(
            "unknown research evaluation profile: "
            f"{profile_name!r}; available profiles={list(AVAILABLE_RESEARCH_EVALUATION_PROFILES)}"
        )
    return builder()


DEFAULT_RESEARCH_EVALUATION_CONFIG = get_research_evaluation_config("default_research")


def research_evaluation_audit_snapshot(
    config: ResearchEvaluationConfig = DEFAULT_RESEARCH_EVALUATION_CONFIG,
) -> ResearchEvaluationAuditSnapshot:
    """Return concise evaluation-governance metadata for artifacts."""
    return config.to_audit_snapshot()
