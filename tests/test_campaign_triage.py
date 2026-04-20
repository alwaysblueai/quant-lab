from __future__ import annotations

from alpha_lab.reporting.campaign_triage import (
    CAMPAIGN_TRIAGE_TAXONOMY,
    build_campaign_triage,
    campaign_rank_sort_key,
)


def _base_metrics() -> dict[str, object]:
    return {
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
        "instability_flags": [],
        "rolling_instability_flags": [],
        "neutralization_comparison_flags": [
            "neutralization preserves most evidence",
        ],
        "ic_ir": 0.82,
        "mean_long_short_return": 0.0032,
        "rebalance_step_dates": 5,
        "ic_half_life_horizon": 10.0,
        "ic_half_life_status": "estimated",
        "ic_half_life_not_reached": False,
        "ic_decay_rebalance_ratio": 0.5,
    }


def test_campaign_triage_taxonomy_is_compact_and_stable() -> None:
    assert CAMPAIGN_TRIAGE_TAXONOMY == (
        "Advance to Level 2",
        "Strong Level 1 candidate",
        "Needs refinement",
        "Fragile / monitor",
        "Drop for now",
    )


def test_campaign_triage_classifies_advance_to_level2() -> None:
    triage = build_campaign_triage(_base_metrics())
    assert triage.label == "Advance to Level 2"
    reason_text = "; ".join(triage.reasons)
    assert "strong raw and neutralized evidence" in reason_text
    assert "stable across rolling windows" in reason_text
    assert "confidence intervals remain supportive" in reason_text


def test_campaign_triage_classifies_needs_refinement_when_neutralization_material() -> None:
    metrics = _base_metrics()
    metrics["neutralization_comparison_flags"] = [
        "neutralization materially reduces independent evidence"
    ]
    triage = build_campaign_triage(metrics)
    assert triage.label == "Needs refinement"
    assert "evidence weakens materially after neutralization" in "; ".join(triage.reasons)


def test_campaign_triage_classifies_fragile_monitor_for_instability() -> None:
    metrics = _base_metrics()
    metrics["rolling_instability_flags"] = ["rolling_ic_sign_flip_instability"]
    metrics["subperiod_ic_positive_share"] = 0.60
    triage = build_campaign_triage(metrics)
    assert triage.label == "Fragile / monitor"
    reason_text = "; ".join(triage.reasons)
    assert "fragile across rolling windows" in reason_text
    assert "fragile across subperiods" in reason_text


def test_campaign_triage_classifies_drop_for_thin_coverage() -> None:
    metrics = _base_metrics()
    metrics["coverage_mean"] = 0.42
    triage = build_campaign_triage(metrics)
    assert triage.label == "Drop for now"
    assert "coverage too thin" in "; ".join(triage.reasons)


def test_campaign_triage_auto_computes_dsr_when_multi_trial_context_provided() -> None:
    metrics = _base_metrics()
    metrics["n_trials"] = 20
    metrics["n_dates"] = 120
    metrics["long_short_ir"] = 3.0
    triage = build_campaign_triage(metrics)
    assert "multi-trial deflated sharpe remains supportive" in "; ".join(triage.reasons)


def test_campaign_rank_sort_key_orders_by_triage_then_explicit_metrics() -> None:
    top = _base_metrics()
    top["neutralization_comparison_flags"] = []
    top["ic_ir"] = 0.95
    top["mean_long_short_return"] = 0.004

    lower = _base_metrics()
    lower["neutralization_comparison_flags"] = []
    lower["ic_ir"] = 0.45
    lower["mean_long_short_return"] = 0.0015

    keys = [
        (
            "top_case",
            campaign_rank_sort_key(
                "top_case",
                status="success",
                metrics=top,
            ),
        ),
        (
            "lower_case",
            campaign_rank_sort_key(
                "lower_case",
                status="success",
                metrics=lower,
            ),
        ),
    ]
    ordered = [name for name, _ in sorted(keys, key=lambda row: row[1])]
    assert ordered == ["top_case", "lower_case"]


def test_campaign_triage_degrades_when_rebalance_slower_than_decay() -> None:
    metrics = _base_metrics()
    metrics["ic_decay_rebalance_ratio"] = 1.3
    triage = build_campaign_triage(metrics)
    assert triage.label == "Needs refinement"
    assert "rebalance cadence is slower than IC decay" in "; ".join(triage.reasons)


def test_campaign_triage_blocks_when_rebalance_materially_exceeds_half_life() -> None:
    metrics = _base_metrics()
    metrics["ic_decay_rebalance_ratio"] = 2.4
    triage = build_campaign_triage(metrics)
    assert triage.label == "Drop for now"
    assert "rebalance cadence materially exceeds IC half-life" in "; ".join(triage.reasons)
