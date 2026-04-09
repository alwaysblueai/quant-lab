from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class FactorSummary:
    factor_id: str
    factor_name: str
    short_description: str
    factor_family: str
    mathematical_definition: str
    display_name_zh: str | None = None
    short_description_zh: str | None = None
    required_input_fields: tuple[str, ...] = field(default_factory=tuple)
    frequency: str = "N/A"
    lookback_parameters: tuple[str, ...] = field(default_factory=tuple)
    lag_delay_rule: str = "N/A"
    expected_sign: str = "N/A"
    economic_intuition: str = "N/A"
    coverage_ratio: float | None = None
    missingness_summary: str = "N/A"
    last_updated_time: str | None = None
    research_status: str = "draft"
    signal_quality_score: float | None = None


@dataclass(frozen=True)
class ValidationSummary:
    ic_mean: float | None = None
    rank_ic_mean: float | None = None
    icir: float | None = None
    t_stat_proxy: float | None = None
    hit_rate: float | None = None
    positive_ic_frequency: float | None = None
    decay_profile: tuple[str, ...] = field(default_factory=tuple)
    horizon_analysis: tuple[str, ...] = field(default_factory=tuple)
    quantile_return_spread: float | None = None
    long_short_performance_summary: str = "N/A"
    monotonicity_diagnostics: str = "N/A"
    regime_breakdown: str = "N/A"
    industry_neutral_comparison: str = "N/A"
    size_neutral_comparison: str = "N/A"
    split_summary: str = "N/A"
    oos_stability_comparison: str = "N/A"


@dataclass(frozen=True)
class FactorDetail:
    summary: FactorSummary
    formal_definition: str
    implementation_notes: str
    pit_anti_lookahead_notes: str
    data_dependencies: tuple[str, ...] = field(default_factory=tuple)
    parameter_settings: tuple[str, ...] = field(default_factory=tuple)
    intended_holding_horizon: str = "N/A"
    coverage_over_time: str = "N/A"
    cross_sectional_coverage: str = "N/A"
    missingness: str = "N/A"
    winsorization_clipping_summary: str = "N/A"
    standardization_neutralization_summary: str = "N/A"
    distribution_snapshots: tuple[str, ...] = field(default_factory=tuple)
    turnover_of_factor_values: str = "N/A"
    stability_over_time: str = "N/A"
    validation: ValidationSummary = field(default_factory=ValidationSummary)
    concise_verdict: str = "N/A"
    concise_verdict_zh: str | None = None
    strengths: tuple[str, ...] = field(default_factory=tuple)
    weaknesses: tuple[str, ...] = field(default_factory=tuple)
    likely_failure_modes: tuple[str, ...] = field(default_factory=tuple)
    proceed_to_portfolio_layer: bool = False
    related_artifacts: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class FactorComparisonRow:
    factor_id: str
    factor_name: str
    factor_family: str
    ic_mean: float | None = None
    rank_ic_mean: float | None = None
    icir: float | None = None
    turnover: float | None = None
    coverage: float | None = None
    long_short_return: float | None = None
    monotonicity_share: float | None = None
    oos_stability_share: float | None = None
    monotonicity: str = "N/A"
    oos_stability: str = "N/A"


@dataclass(frozen=True)
class FactorShortlistConfig:
    formula: str = (
        "score = weighted_average(clip01(IC), clip01(RankIC), clip01(ICIR), "
        "clip01(monotonicity), clip01(1-turnover), clip01(OOS stability))"
    )
    component_weights: tuple[tuple[str, float], ...] = (
        ("ic_mean", 0.22),
        ("rank_ic_mean", 0.18),
        ("icir", 0.28),
        ("monotonicity_share", 0.12),
        ("turnover_efficiency", 0.10),
        ("oos_stability_share", 0.10),
    )
    keep_score_min: float = 0.65
    watchlist_score_min: float = 0.45
    min_ic_mean: float = 0.015
    min_rank_ic_mean: float = 0.020
    min_icir: float = 0.25
    min_monotonicity_share: float = 0.55
    max_turnover: float = 0.80
    min_oos_stability_share: float = 0.55
    redundancy_correlation_max: float = 0.70


@dataclass(frozen=True)
class FactorShortlistEntry:
    rank: int
    factor_id: str
    factor_name: str
    factor_family: str
    composite_score: float | None = None
    recommendation: str = "watchlist"
    ic_mean: float | None = None
    rank_ic_mean: float | None = None
    icir: float | None = None
    turnover: float | None = None
    monotonicity_share: float | None = None
    oos_stability_share: float | None = None
    max_correlation_to_selected: float | None = None
    redundancy_with: str | None = None
    rationale: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class FactorShortlistResult:
    config: FactorShortlistConfig = field(default_factory=FactorShortlistConfig)
    selected_factor_ids: tuple[str, ...] = field(default_factory=tuple)
    entries: tuple[FactorShortlistEntry, ...] = field(default_factory=tuple)
    recommendation_summary: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class FactorSetConstructionConfig:
    policy_id: str = "factor_set_construction_v1"
    formula_text: str = (
        "selected_set=top_keep_factors_by_score; "
        "candidate_set=diversified_keep_watchlist_mix; "
        "watchlist_set=top_per_family_watchlist; "
        "rejected_set=drop_or_failed_quality_guardrails"
    )
    selected_set_size: int = 3
    candidate_set_size: int = 4
    watchlist_set_size: int = 3
    redundancy_correlation_max: float = 0.70
    turnover_max: float = 0.80
    oos_stability_min: float = 0.55
    min_selected_score: float = 0.65
    min_candidate_score: float = 0.45


@dataclass(frozen=True)
class FactorSetScoreSummary:
    mean_shortlist_score: float | None = None
    mean_icir: float | None = None
    mean_turnover: float | None = None
    mean_oos_stability_share: float | None = None
    max_pair_correlation: float | None = None
    family_balance_ratio: float | None = None


@dataclass(frozen=True)
class FactorSetDefinition:
    factor_set_id: str
    label_zh: str | None = None
    factor_ids: tuple[str, ...] = field(default_factory=tuple)
    factor_names: tuple[str, ...] = field(default_factory=tuple)
    source_shortlist_entries: tuple[str, ...] = field(default_factory=tuple)
    construction_rule: str = "N/A"
    status: str = "candidate"
    rationale: tuple[str, ...] = field(default_factory=tuple)
    rationale_zh: tuple[str, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)
    score_summary: FactorSetScoreSummary = field(default_factory=FactorSetScoreSummary)


@dataclass(frozen=True)
class FactorSetConstructionResult:
    config: FactorSetConstructionConfig = field(default_factory=FactorSetConstructionConfig)
    factor_sets: tuple[FactorSetDefinition, ...] = field(default_factory=tuple)
    selected_factor_set_ids: tuple[str, ...] = field(default_factory=tuple)
    recommendation_summary: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class PortfolioRecipeSummary:
    recipe_id: str
    recipe_name: str
    selected_factors: tuple[str, ...] = field(default_factory=tuple)
    weighting_scheme: str = "N/A"
    neutralization_constraints: str = "N/A"
    benchmark_mode: str = "N/A"
    industry_constraints: str = "N/A"
    style_constraints: str = "N/A"
    turnover_penalty_settings: str = "N/A"
    rebalance_frequency: str = "N/A"
    transaction_cost_assumptions: str = "N/A"
    universe_definition: str = "N/A"
    position_limits: str = "N/A"
    factor_contributions: tuple[str, ...] = field(default_factory=tuple)
    expected_risk_summary: str = "N/A"
    expected_return_proxy: str = "N/A"
    optimizer_diagnostics: str = "N/A"
    infeasible_configuration_warnings: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class PortfolioBacktestSummary:
    recipe_id: str
    factor_id: str
    annualized_return: float | None = None
    annualized_volatility: float | None = None
    sharpe: float | None = None
    sortino: float | None = None
    max_drawdown: float | None = None
    calmar: float | None = None
    win_rate: float | None = None
    turnover: float | None = None
    information_ratio: float | None = None
    excess_return_vs_benchmark: float | None = None
    tracking_error: float | None = None
    pre_cost_return: float | None = None
    post_cost_return: float | None = None
    nav_points: tuple[tuple[str, float], ...] = field(default_factory=tuple)
    monthly_return_table: tuple[tuple[str, float], ...] = field(default_factory=tuple)
    drawdown_table: tuple[tuple[str, float], ...] = field(default_factory=tuple)
    period_by_period_attribution: str = "N/A"
    subperiod_analysis: str = "N/A"
    regime_analysis: str = "N/A"
    rolling_sharpe: float | None = None
    rolling_drawdown: float | None = None
    portfolio_composition_snapshot: str = "N/A"
    trade_statistics: str = "N/A"
    capacity_implementability_notes: str = "N/A"


@dataclass(frozen=True)
class RecipeComparisonRow:
    recipe_id: str
    recipe_name: str
    selected_factors: tuple[str, ...] = field(default_factory=tuple)
    factor_family_mix: tuple[str, ...] = field(default_factory=tuple)
    objective_tag: str = "N/A"
    construction_style: str = "N/A"
    weighting_scheme: str = "N/A"
    neutralization_constraints: str = "N/A"
    turnover_penalty_settings: str = "N/A"
    transaction_cost_assumptions: str = "N/A"
    benchmark_mode: str = "N/A"
    position_limits: str = "N/A"
    expected_return_proxy: str = "N/A"
    expected_risk_summary: str = "N/A"
    sharpe: float | None = None
    annualized_return: float | None = None
    max_drawdown: float | None = None
    information_ratio: float | None = None
    post_cost_return: float | None = None


@dataclass(frozen=True)
class RecipeLeaderboardEntry:
    objective: str
    rank: int
    recipe_id: str
    recipe_name: str
    metric_value: float | None = None


@dataclass(frozen=True)
class RecipeHeadToHeadInsight:
    objective: str
    winner_recipe_id: str
    loser_recipe_id: str
    summary: str
    reasons: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class RecipeComparisonView:
    rows: tuple[RecipeComparisonRow, ...] = field(default_factory=tuple)
    leaderboards: tuple[RecipeLeaderboardEntry, ...] = field(default_factory=tuple)
    head_to_head: tuple[RecipeHeadToHeadInsight, ...] = field(default_factory=tuple)
    grouping_summary: tuple[tuple[str, int], ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class CandidateRecipeGenerationConfig:
    policy_id: str = "candidate_recipe_generation_v1"
    formula_text: str = (
        "for each selected/candidate factor_set, emit deterministic variants "
        "across weighting, neutralization, turnover_penalty, and benchmark_mode"
    )
    max_recipes_per_factor_set: int = 3
    weighting_schemes: tuple[str, ...] = ("rank", "equal_weight")
    neutralization_modes: tuple[str, ...] = (
        "neutralization_on",
        "neutralization_off",
    )
    turnover_penalty_modes: tuple[str, ...] = ("strict", "balanced")
    benchmark_modes: tuple[str, ...] = ("benchmark_relative", "absolute")


@dataclass(frozen=True)
class CandidateRecipe:
    recipe_id: str
    recipe_name: str
    source_factor_set_id: str
    source_factor_ids: tuple[str, ...] = field(default_factory=tuple)
    construction_variant: str = "N/A"
    weighting_scheme: str = "N/A"
    neutralization_mode: str = "N/A"
    turnover_penalty_mode: str = "N/A"
    benchmark_mode: str = "N/A"
    rationale: tuple[str, ...] = field(default_factory=tuple)
    assumptions: tuple[str, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class CandidateRecipeGenerationResult:
    config: CandidateRecipeGenerationConfig = field(default_factory=CandidateRecipeGenerationConfig)
    generated_recipes: tuple[CandidateRecipe, ...] = field(default_factory=tuple)
    recommendation_summary: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class WinnerSelectionPolicy:
    decision_policy_id: str = "winner_selection_policy_v1"
    formula_text: str = (
        "composite = weighted_average(sharpe, post_cost_return, annualized_return, "
        "drawdown_quality, composition_quality, robustness_quality); "
        "heuristic fallback applies when metrics are missing"
    )
    component_weights: tuple[tuple[str, float], ...] = (
        ("sharpe", 0.30),
        ("post_cost_return", 0.25),
        ("annualized_return", 0.15),
        ("drawdown_quality", 0.15),
        ("composition_quality", 0.10),
        ("robustness_quality", 0.05),
    )
    min_sharpe_for_winner: float = 0.60
    min_post_cost_return_for_winner: float = 0.00
    max_drawdown_floor: float = -0.35
    challenger_count: int = 2
    watchlist_score_min: float = 0.40
    reject_score_max: float = 0.20


@dataclass(frozen=True)
class WinnerSelectionResult:
    decision_policy_id: str = ""
    winner_recipe_id: str = ""
    challenger_recipe_ids: tuple[str, ...] = field(default_factory=tuple)
    watchlist_recipe_ids: tuple[str, ...] = field(default_factory=tuple)
    rejected_recipe_ids: tuple[str, ...] = field(default_factory=tuple)
    policy_formula_text: str = "N/A"
    decision_reasons: tuple[str, ...] = field(default_factory=tuple)
    decision_reasons_zh: tuple[str, ...] = field(default_factory=tuple)
    challenger_reasons: tuple[str, ...] = field(default_factory=tuple)
    challenger_reasons_zh: tuple[str, ...] = field(default_factory=tuple)
    rejection_reasons: tuple[str, ...] = field(default_factory=tuple)
    rejection_reasons_zh: tuple[str, ...] = field(default_factory=tuple)
    next_actions: tuple[str, ...] = field(default_factory=tuple)
    next_actions_zh: tuple[str, ...] = field(default_factory=tuple)
    score_table: tuple[tuple[str, float | None], ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class NextStepRecommendation:
    recommendation_id: str
    action: str
    rationale: str
    action_text_zh: str | None = None
    rationale_zh: str | None = None
    label_zh: str | None = None
    category: str = "research_action"
    priority: str = "P2"
    trigger_objects: tuple[str, ...] = field(default_factory=tuple)
    supporting_evidence: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class NextStepRecommendationResult:
    policy_id: str = "next_step_policy_v1"
    policy_formula_text: str = (
        "emit deterministic recommendations from shortlist/factor_set/"
        "candidate_recipe/winner_selection signals"
    )
    recommendations: tuple[NextStepRecommendation, ...] = field(default_factory=tuple)
    summary: tuple[str, ...] = field(default_factory=tuple)
    summary_zh: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class RobustnessSummary:
    factor_id: str
    parameter_sensitivity: str = "N/A"
    lookback_sensitivity: str = "N/A"
    universe_sensitivity: str = "N/A"
    rebalance_sensitivity: str = "N/A"
    transaction_cost_sensitivity: str = "N/A"
    profile_sensitivity: str = "N/A"
    leakage_checks: str = "N/A"
    survivorship_pit_checks: str = "N/A"
    implementation_warnings: tuple[str, ...] = field(default_factory=tuple)
    robustness_verdict: str = "N/A"


@dataclass(frozen=True)
class ExperimentRegistryEntry:
    case_name: str
    profile_name: str
    run_id: str
    run_timestamp_utc: str | None = None
    factor_id: str = ""
    recipe_id: str = ""
    backtest_id: str = ""
    output_dir: str = ""
    run_manifest_path: str = ""
    factor_definition_path: str = ""
    signal_validation_path: str = ""
    portfolio_recipe_path: str = ""
    backtest_result_path: str = ""
    provenance_links: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ResearchLineageLink:
    from_object: str
    relation: str
    to_object: str


@dataclass(frozen=True)
class ResearchLineageRegistry:
    entries: tuple[ExperimentRegistryEntry, ...] = field(default_factory=tuple)
    links: tuple[ResearchLineageLink, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class DashboardOverview:
    total_candidate_factors: int
    validated_factors: int
    active_portfolio_recipes: int
    completed_backtests: int
    top_factors_by_signal_quality: tuple[str, ...] = field(default_factory=tuple)
    top_portfolios_by_objective: tuple[str, ...] = field(default_factory=tuple)
    recent_research_runs: tuple[str, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ArtifactLoadDiagnostic:
    code: str
    severity: Literal["warning", "error"]
    artifact_type: str
    object_scope: str
    message: str
    path: str | None = None
    case_name: str | None = None
    profile_name: str | None = None
    mode: str = "permissive"
    fallback_used: bool = False
    remediation_hint: str | None = None


@dataclass(frozen=True)
class ResearchDashboardData:
    overview: DashboardOverview
    factor_summaries: tuple[FactorSummary, ...] = field(default_factory=tuple)
    factor_details: tuple[FactorDetail, ...] = field(default_factory=tuple)
    comparison_rows: tuple[FactorComparisonRow, ...] = field(default_factory=tuple)
    factor_shortlist: FactorShortlistResult = field(default_factory=FactorShortlistResult)
    factor_sets: FactorSetConstructionResult = field(default_factory=FactorSetConstructionResult)
    candidate_recipe_generation: CandidateRecipeGenerationResult = field(
        default_factory=CandidateRecipeGenerationResult
    )
    portfolio_recipes: tuple[PortfolioRecipeSummary, ...] = field(default_factory=tuple)
    recipe_comparison: RecipeComparisonView = field(default_factory=RecipeComparisonView)
    winner_selection: WinnerSelectionResult = field(default_factory=WinnerSelectionResult)
    next_step_recommendations: NextStepRecommendationResult = field(
        default_factory=NextStepRecommendationResult
    )
    backtests: tuple[PortfolioBacktestSummary, ...] = field(default_factory=tuple)
    lineage_registry: ResearchLineageRegistry = field(default_factory=ResearchLineageRegistry)
    robustness_summaries: tuple[RobustnessSummary, ...] = field(default_factory=tuple)
    factor_correlation_matrix: tuple[tuple[str, tuple[tuple[str, float | None], ...]], ...] = (
        field(default_factory=tuple)
    )
    factor_family_summary: tuple[tuple[str, int], ...] = field(default_factory=tuple)
    shortlist_recommendations: tuple[str, ...] = field(default_factory=tuple)
    generated_at_utc: str | None = None
    default_profile: str = "N/A"
    source_json_path: str = ""
    artifact_load_mode: str = "permissive"
    artifact_load_policy_summary: tuple[str, ...] = field(default_factory=tuple)
    artifact_load_warnings: tuple[str, ...] = field(default_factory=tuple)
    artifact_load_diagnostics: tuple[ArtifactLoadDiagnostic, ...] = field(default_factory=tuple)
