from __future__ import annotations

import math

import pytest

from alpha_lab.research_bridge.scoring import (
    ALPHA_MODE_WEIGHTS,
    DAILY_DATA_INVENTORY,
    FORBIDDEN_FACTOR_LABELS,
    INTRADAY_DATA_INVENTORY,
    MECHANISM_DISCOVERY,
    MODEL_MODE_WEIGHTS,
    SIGNAL_MAPPING,
    VALIDATION_KILL_TESTS,
    WORKFLOW_STAGES,
    CardMetadata,
    QueryAnchor,
    ScoreComponents,
    ScoreWeights,
    aggregate_score,
    data_dependency_score,
    derive_failure_keywords,
    failure_proximity_score,
    infer_available_data_from_frequency,
    mechanism_proximity_score,
    metadata_compat_score,
    normalize_data_set,
    normalize_workflow_stage,
    recommend_next_stage,
    score_card,
)


def test_metadata_compat_returns_zero_when_anchor_empty() -> None:
    anchor = QueryAnchor()
    card = CardMetadata(name="X", domain="price_action", market="a_share")
    assert metadata_compat_score(anchor, card) == 0.0


def test_metadata_compat_averages_matched_fields() -> None:
    anchor = QueryAnchor(domain="price_action", market="a_share")
    same = CardMetadata(name="X", domain="price_action", market="a_share")
    half = CardMetadata(name="Y", domain="price_action", market="hk_share")
    miss = CardMetadata(name="Z", domain="fundamental", market="hk_share")
    assert metadata_compat_score(anchor, same) == 1.0
    assert metadata_compat_score(anchor, half) == 0.5
    assert metadata_compat_score(anchor, miss) == 0.0


def test_mechanism_proximity_prefers_mechanism_over_family() -> None:
    anchor = QueryAnchor(mechanism="behavioral", factor_family="momentum")
    same_mech = CardMetadata(name="A", mechanism="behavioral", factor_family="value")
    same_family = CardMetadata(name="B", mechanism="risk", factor_family="momentum")
    different = CardMetadata(name="C", mechanism="risk", factor_family="value")
    assert mechanism_proximity_score(anchor, same_mech) == 1.0
    assert mechanism_proximity_score(anchor, same_family) == pytest.approx(0.6)
    assert mechanism_proximity_score(anchor, different) == 0.0


def test_data_dependency_neutral_when_no_inventory() -> None:
    card = CardMetadata(name="A", uses_data=frozenset({"close"}))
    score, satisfied, reason = data_dependency_score(card, available_data=None)
    assert satisfied is True
    assert reason == ""
    assert score == 0.5


def test_data_dependency_full_coverage_keeps_card() -> None:
    card = CardMetadata(name="A", uses_data=frozenset({"close", "volume"}))
    score, satisfied, _ = data_dependency_score(
        card, available_data=frozenset({"close", "volume", "open"})
    )
    assert satisfied is True
    assert score == 1.0


def test_data_dependency_partial_coverage_drops_card() -> None:
    card = CardMetadata(name="A", uses_data=frozenset({"close", "intraday_volume"}))
    score, satisfied, reason = data_dependency_score(
        card, available_data=frozenset({"close"})
    )
    assert satisfied is False
    assert math.isclose(score, 0.5)
    assert "intraday_volume" in reason


def test_data_dependency_no_uses_keeps_card_with_full_score() -> None:
    card = CardMetadata(name="A", uses_data=frozenset())
    score, satisfied, _ = data_dependency_score(
        card, available_data=frozenset({"close"})
    )
    assert satisfied is True
    assert score == 1.0


def test_failure_proximity_direct_and_token_overlap() -> None:
    keywords = derive_failure_keywords(
        [
            {
                "title": "Behavioral overcrowding in momentum",
                "failure_class": "saturation",
                "failure_statement": "Momentum trades become crowded",
                "prevention_rule": "avoid behavioral momentum baseline",
            }
        ]
    )
    direct = CardMetadata(
        name="A", mechanism="behavioral", factor_family="momentum"
    )
    related = CardMetadata(name="B", mechanism="risk", factor_family="momentum")
    unrelated = CardMetadata(name="C", mechanism="value", factor_family="dividend")
    assert failure_proximity_score(direct, keywords) == 1.0
    assert failure_proximity_score(related, keywords) == 1.0
    assert failure_proximity_score(unrelated, keywords) == 0.0


def test_aggregate_score_uses_weighted_sum() -> None:
    components = ScoreComponents(
        semantic=0.8,
        metadata=0.5,
        mechanism=1.0,
        dependency=0.5,
        failure=1.0,
        llm_relevance=0.0,
    )
    weights = ScoreWeights(
        semantic=1.0,
        metadata=0.0,
        mechanism=0.0,
        dependency=0.0,
        failure=-0.5,
        llm_relevance=0.0,
    )
    assert aggregate_score(components, weights) == pytest.approx(0.8 - 0.5)


def test_aggregate_includes_llm_relevance() -> None:
    components = ScoreComponents(semantic=0.8, llm_relevance=1.0)
    weights = ScoreWeights(semantic=1.0, llm_relevance=0.6)
    assert aggregate_score(components, weights) == pytest.approx(1.4)


def test_score_card_drops_when_dependency_unsatisfied() -> None:
    anchor = QueryAnchor(domain="price_action", mechanism="behavioral")
    card = CardMetadata(
        name="X",
        domain="price_action",
        mechanism="behavioral",
        uses_data=frozenset({"close", "intraday_volume"}),
    )
    components, kept, reason = score_card(
        anchor=anchor,
        card=card,
        semantic_score=0.9,
        available_data=frozenset({"close"}),
        failure_keywords=frozenset(),
    )
    assert kept is False
    assert "intraday_volume" in reason
    # components are still returned for observability
    assert components.semantic == 0.9


def test_score_card_keeps_when_no_inventory_provided() -> None:
    anchor = QueryAnchor(domain="price_action")
    card = CardMetadata(
        name="X", domain="price_action", uses_data=frozenset({"close"})
    )
    _, kept, reason = score_card(
        anchor=anchor,
        card=card,
        semantic_score=0.5,
        available_data=None,
        failure_keywords=frozenset(),
    )
    assert kept is True
    assert reason == ""


def test_alpha_mode_weights_have_three_modes_and_distinct_profiles() -> None:
    assert set(ALPHA_MODE_WEIGHTS.keys()) == {"start", "free", "constrained"}
    # failure proximity is diagnostic-only in Stage 1; it must not rank or kill.
    assert ALPHA_MODE_WEIGHTS["start"].failure == 0
    assert ALPHA_MODE_WEIGHTS["constrained"].failure == 0
    # constrained should up-weight dependency satisfaction
    assert ALPHA_MODE_WEIGHTS["constrained"].dependency > ALPHA_MODE_WEIGHTS["start"].dependency
    assert ALPHA_MODE_WEIGHTS["free"].failure == 0


def test_model_mode_weights_have_three_modes_and_share_shape() -> None:
    assert set(MODEL_MODE_WEIGHTS.keys()) == {"start", "explore", "constrained"}
    assert MODEL_MODE_WEIGHTS["start"].failure == 0
    assert MODEL_MODE_WEIGHTS["constrained"].failure == 0
    assert MODEL_MODE_WEIGHTS["constrained"].dependency > MODEL_MODE_WEIGHTS["explore"].dependency


def test_mode_weights_have_llm_relevance() -> None:
    assert ALPHA_MODE_WEIGHTS["start"].llm_relevance == pytest.approx(1.0)
    assert ALPHA_MODE_WEIGHTS["free"].llm_relevance == pytest.approx(0.6)
    assert ALPHA_MODE_WEIGHTS["constrained"].llm_relevance == pytest.approx(0.3)

    assert MODEL_MODE_WEIGHTS["start"].llm_relevance == pytest.approx(1.0)
    assert MODEL_MODE_WEIGHTS["explore"].llm_relevance == pytest.approx(0.6)
    assert MODEL_MODE_WEIGHTS["constrained"].llm_relevance == pytest.approx(0.3)


def test_normalize_data_set_strips_and_lowercases() -> None:
    assert normalize_data_set([" Close ", "VOLUME", ""]) == frozenset(
        {"close", "volume"}
    )
    assert normalize_data_set(None) == frozenset()


# ---------------------------------------------------------------------------
# Workflow stage axis (P2)
# ---------------------------------------------------------------------------


def test_workflow_stages_are_stage1_substeps_only() -> None:
    assert WORKFLOW_STAGES == frozenset(
        {MECHANISM_DISCOVERY, SIGNAL_MAPPING}
    )


def test_normalize_workflow_stage_defaults_to_mechanism_discovery() -> None:
    assert normalize_workflow_stage(None) == MECHANISM_DISCOVERY
    assert normalize_workflow_stage("") == MECHANISM_DISCOVERY
    assert normalize_workflow_stage("garbage") == MECHANISM_DISCOVERY
    assert normalize_workflow_stage("MECHANISM_DISCOVERY") == MECHANISM_DISCOVERY
    assert normalize_workflow_stage(SIGNAL_MAPPING) == SIGNAL_MAPPING


def test_recommend_next_stage_progresses_then_stops() -> None:
    assert recommend_next_stage(MECHANISM_DISCOVERY) == SIGNAL_MAPPING
    assert recommend_next_stage(SIGNAL_MAPPING) is None
    # Legacy Stage 3 helper is retained for direct prompt/lint use, but it
    # is no longer part of the idea-explorer progression.
    assert recommend_next_stage(VALIDATION_KILL_TESTS) is None
    # Unknown stage normalizes to mechanism_discovery, which then progresses.
    assert recommend_next_stage("garbage") == SIGNAL_MAPPING


def test_infer_available_data_for_daily_and_intraday() -> None:
    assert infer_available_data_from_frequency("daily") == DAILY_DATA_INVENTORY
    assert infer_available_data_from_frequency("Daily") == DAILY_DATA_INVENTORY
    assert infer_available_data_from_frequency("weekly") == DAILY_DATA_INVENTORY
    assert infer_available_data_from_frequency("intraday") == INTRADAY_DATA_INVENTORY
    assert infer_available_data_from_frequency("1min") == INTRADAY_DATA_INVENTORY
    assert infer_available_data_from_frequency("hft") == INTRADAY_DATA_INVENTORY


def test_infer_available_data_unknown_returns_none() -> None:
    assert infer_available_data_from_frequency(None) is None
    assert infer_available_data_from_frequency("") is None
    assert infer_available_data_from_frequency("custom_freq") is None


def test_daily_inventory_excludes_intraday_only_fields() -> None:
    intraday_only = INTRADAY_DATA_INVENTORY - DAILY_DATA_INVENTORY
    assert "tick" in intraday_only
    assert "bid_ask" in intraday_only
    assert "intraday_trades" in intraday_only
    assert "intraday_tick_volume" in intraday_only
    # Daily-tier projects must never auto-pull intraday-only fields.
    assert intraday_only.isdisjoint(DAILY_DATA_INVENTORY)


def test_forbidden_factor_labels_includes_canonical_set() -> None:
    expected_subset = {
        "reversal",
        "momentum",
        "value",
        "size",
        "skewness",
        "liquidity",
        "sentiment",
        "crowding",
        "反转",
        "动量",
    }
    assert expected_subset.issubset(set(FORBIDDEN_FACTOR_LABELS))
