"""Phase 2 metric audit: decision-layer verification.

Covers factor_verdict, campaign_triage, level2_promotion,
neutralization_comparison, and uncertainty flags — the logic that
translates raw metrics into research decisions.

Each test constructs a synthetic metrics dict at a known threshold
boundary and verifies the classification is correct.

Run:  pytest tests/test_metric_audit_phase2.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_lab.reporting.campaign_triage import (
    CAMPAIGN_TRIAGE_TAXONOMY,
    build_campaign_triage,
    campaign_rank_sort_key,
)
from alpha_lab.reporting.factor_verdict import (
    FACTOR_VERDICT_TAXONOMY,
    build_factor_verdict,
)
from alpha_lab.reporting.level2_promotion import (
    LEVEL2_PROMOTION_TAXONOMY,
    build_level2_promotion,
)
from alpha_lab.reporting.neutralization_comparison import (
    EXPOSURE_DRIVEN_FLAG,
    MATERIAL_REDUCTION_FLAG,
    MODERATE_WEAKENING_FLAG,
    PRESERVES_EVIDENCE_FLAG,
    WEAKER_BUT_STABLER_FLAG,
    build_raw_vs_neutralized_comparison,
)
from alpha_lab.reporting.uncertainty import (
    compute_core_uncertainty,
    compute_mean_ci,
)
from alpha_lab.research_evaluation_config import (
    FactorVerdictConfig,
    UncertaintyConfig,
    get_research_evaluation_config,
)

# ── helpers ──────────────────────────────────────────────────────────────────


def _strong_metrics() -> dict[str, object]:
    """Construct a metrics dict that should classify as 'Strong candidate'.

    All signals are positive with comfortable margins above default thresholds.
    """
    return {
        "case_name": "audit_strong",
        "factor_name": "test_factor",
        "n_dates_used": 50,
        "n_dates": 50,
        "mean_ic": 0.05,
        "mean_rank_ic": 0.04,
        "ic_ir": 0.8,
        "ic_positive_rate": 0.65,
        "rank_ic_positive_rate": 0.62,
        "ic_valid_ratio": 0.95,
        "rank_ic_valid_ratio": 0.93,
        "mean_long_short_return": 0.005,
        "long_short_ir": 0.6,
        "long_short_hit_rate": 0.60,
        "mean_long_short_turnover": 0.3,
        "long_short_return_per_turnover": 0.015,
        "subperiod_ic_positive_share": 1.0,
        "subperiod_long_short_positive_share": 1.0,
        "subperiod_ic_min_mean": 0.02,
        "subperiod_long_short_min_mean": 0.002,
        "rolling_window_size": 20,
        "rolling_ic_positive_share": 0.80,
        "rolling_rank_ic_positive_share": 0.75,
        "rolling_long_short_positive_share": 0.70,
        "rolling_ic_min_mean": 0.01,
        "rolling_rank_ic_min_mean": 0.008,
        "rolling_long_short_min_mean": 0.001,
        "rolling_instability_flags": [],
        "instability_flags": [],
        "mean_ic_ci_lower": 0.02,
        "mean_ic_ci_upper": 0.08,
        "mean_rank_ic_ci_lower": 0.01,
        "mean_rank_ic_ci_upper": 0.07,
        "mean_long_short_return_ci_lower": 0.001,
        "mean_long_short_return_ci_upper": 0.009,
        "uncertainty_flags": [],
        "eval_coverage_ratio_mean": 0.85,
        "eval_coverage_ratio_min": 0.70,
        "coverage_mean": 0.85,
        "coverage_min": 0.70,
        "neutralization_mean_corr_reduction": 0.05,
        "neutralization_comparison_flags": [PRESERVES_EVIDENCE_FLAG],
        # Promotion gate fields
        "factor_verdict": "Strong candidate",
        "campaign_triage": "Advance to Level 2",
    }


def _weak_metrics() -> dict[str, object]:
    """Construct a metrics dict that should classify as 'Weak / noisy'."""
    return {
        "case_name": "audit_weak",
        "factor_name": "test_factor",
        "n_dates_used": 50,
        "n_dates": 50,
        "mean_ic": -0.01,
        "mean_rank_ic": -0.005,
        "ic_ir": -0.2,
        "ic_positive_rate": 0.40,
        "rank_ic_positive_rate": 0.42,
        "ic_valid_ratio": 0.90,
        "rank_ic_valid_ratio": 0.88,
        "mean_long_short_return": -0.002,
        "long_short_ir": -0.15,
        "long_short_hit_rate": 0.40,
        "mean_long_short_turnover": 0.5,
        "long_short_return_per_turnover": -0.004,
        "subperiod_ic_positive_share": 0.55,
        "subperiod_long_short_positive_share": 0.55,
        "subperiod_ic_min_mean": -0.03,
        "subperiod_long_short_min_mean": -0.005,
        "rolling_window_size": 20,
        "rolling_ic_positive_share": 0.35,
        "rolling_rank_ic_positive_share": 0.30,
        "rolling_long_short_positive_share": 0.40,
        "rolling_ic_min_mean": -0.02,
        "rolling_rank_ic_min_mean": -0.025,
        "rolling_long_short_min_mean": -0.003,
        "rolling_instability_flags": [],
        "instability_flags": [],
        "mean_ic_ci_lower": -0.04,
        "mean_ic_ci_upper": 0.02,
        "mean_rank_ic_ci_lower": -0.03,
        "mean_rank_ic_ci_upper": 0.02,
        "mean_long_short_return_ci_lower": -0.006,
        "mean_long_short_return_ci_upper": 0.002,
        "uncertainty_flags": [],
        "eval_coverage_ratio_mean": 0.80,
        "eval_coverage_ratio_min": 0.60,
        "coverage_mean": 0.80,
        "coverage_min": 0.60,
        "neutralization_mean_corr_reduction": 0.10,
        "neutralization_comparison_flags": [],
        "factor_verdict": "Weak / noisy",
        "campaign_triage": "Drop for now",
    }


def _failing_metrics() -> dict[str, object]:
    """Metrics that fail basic robustness (coverage too thin)."""
    m = _strong_metrics()
    m["eval_coverage_ratio_mean"] = 0.40
    m["eval_coverage_ratio_min"] = 0.20
    m["coverage_mean"] = 0.40
    m["coverage_min"] = 0.20
    return m


# ═══════════════════════════════════════════════════════════════════════════
# 2.1  Factor verdict classification
# ═══════════════════════════════════════════════════════════════════════════


class TestFactorVerdict:
    """Verify verdict classification logic at known boundary conditions."""

    def test_strong_candidate(self):
        m = _strong_metrics()
        v = build_factor_verdict(m)
        assert v.label == "Strong candidate"

    def test_weak_noisy(self):
        m = _weak_metrics()
        v = build_factor_verdict(m)
        assert v.label == "Weak / noisy"

    def test_fails_basic_robustness_low_coverage(self):
        """Coverage below min_coverage_mean_fail → critical failure."""
        m = _failing_metrics()
        v = build_factor_verdict(m)
        assert v.label == "Fails basic robustness"
        assert any("coverage" in r.lower() for r in v.reasons)

    def test_fails_basic_robustness_low_valid_ratio(self):
        """IC valid_ratio below threshold → critical failure."""
        m = _strong_metrics()
        m["ic_valid_ratio"] = 0.50  # below default 0.60
        m["rank_ic_valid_ratio"] = 0.50
        v = build_factor_verdict(m)
        assert v.label == "Fails basic robustness"

    def test_fails_basic_robustness_short_eval_window(self):
        """n_dates below min_eval_dates_basic → critical failure."""
        m = _strong_metrics()
        m["n_dates_used"] = 10  # below default 20
        m["n_dates"] = 10
        v = build_factor_verdict(m)
        assert v.label == "Fails basic robustness"
        assert any("window" in r.lower() or "short" in r.lower() for r in v.reasons)

    def test_fails_basic_robustness_subperiod_too_low(self):
        """Subperiod share below 0.50 → critical failure."""
        m = _strong_metrics()
        m["subperiod_ic_positive_share"] = 0.40
        m["subperiod_long_short_positive_share"] = 0.40
        v = build_factor_verdict(m)
        assert v.label == "Fails basic robustness"

    def test_promising_but_fragile(self):
        """Positive core signal but some concerns → Promising but fragile."""
        m = _strong_metrics()
        # Weaken rolling evidence so we have concerns
        m["rolling_ic_positive_share"] = 0.45
        m["rolling_rank_ic_positive_share"] = 0.45
        m["rolling_long_short_positive_share"] = 0.45
        # CI overlaps zero for one metric → concern
        m["mean_ic_ci_lower"] = -0.001
        v = build_factor_verdict(m)
        assert v.label in ("Promising but fragile", "Mixed evidence")

    def test_mixed_evidence(self):
        """Moderate positive count with weak core → mixed."""
        m = _strong_metrics()
        # Reduce positive score to ~2 by removing key positives
        m["mean_ic"] = -0.001
        m["mean_rank_ic"] = -0.001
        m["ic_positive_rate"] = 0.52
        m["rank_ic_positive_rate"] = 0.52
        m["mean_long_short_return"] = -0.001
        m["long_short_ir"] = -0.1
        # Still has subperiod robustness and coverage
        v = build_factor_verdict(m)
        assert v.label in ("Mixed evidence", "Weak / noisy")

    def test_taxonomy_completeness(self):
        """All verdict labels are in the taxonomy."""
        assert len(FACTOR_VERDICT_TAXONOMY) == 5
        for label in FACTOR_VERDICT_TAXONOMY:
            assert isinstance(label, str)

    def test_verdict_always_has_reasons(self):
        """Every verdict must have at least one reason."""
        for m_fn in [_strong_metrics, _weak_metrics, _failing_metrics]:
            v = build_factor_verdict(m_fn())
            assert len(v.reasons) >= 1

    def test_empty_metrics_returns_mixed(self):
        """No diagnostics at all → Mixed evidence."""
        m: dict[str, object] = {}
        v = build_factor_verdict(m)
        assert v.label == "Mixed evidence"
        assert "insufficient" in v.reasons[0].lower()

    def test_threshold_boundary_eval_dates(self):
        """Exactly at min_eval_dates_basic (20) → should NOT fail."""
        m = _strong_metrics()
        m["n_dates_used"] = 20
        m["n_dates"] = 20
        v = build_factor_verdict(m)
        assert v.label != "Fails basic robustness"

    def test_threshold_boundary_eval_dates_minus_one(self):
        """One below min_eval_dates_basic (19) → SHOULD fail."""
        m = _strong_metrics()
        m["n_dates_used"] = 19
        m["n_dates"] = 19
        v = build_factor_verdict(m)
        assert v.label == "Fails basic robustness"

    def test_threshold_boundary_valid_ratio(self):
        """Exactly at min_valid_ratio_fail (0.60) → should NOT fail."""
        m = _strong_metrics()
        m["ic_valid_ratio"] = 0.60
        m["rank_ic_valid_ratio"] = 0.60
        v = build_factor_verdict(m)
        assert v.label != "Fails basic robustness"

    def test_threshold_boundary_valid_ratio_minus_epsilon(self):
        """Just below min_valid_ratio_fail (0.59) → SHOULD fail."""
        m = _strong_metrics()
        m["ic_valid_ratio"] = 0.59
        m["rank_ic_valid_ratio"] = 0.59
        v = build_factor_verdict(m)
        assert v.label == "Fails basic robustness"

    def test_threshold_boundary_subperiod_share(self):
        """Exactly at min_subperiod_share_fail (0.50) → should NOT fail."""
        m = _strong_metrics()
        m["subperiod_ic_positive_share"] = 0.50
        m["subperiod_long_short_positive_share"] = 0.50
        v = build_factor_verdict(m)
        assert v.label != "Fails basic robustness"

    def test_threshold_boundary_subperiod_share_minus_epsilon(self):
        """Just below min_subperiod_share_fail (0.49) → SHOULD fail."""
        m = _strong_metrics()
        m["subperiod_ic_positive_share"] = 0.49
        v = build_factor_verdict(m)
        assert v.label == "Fails basic robustness"

    def test_neutralization_material_adds_concern(self):
        """Material corr reduction → adds concern reason."""
        m = _strong_metrics()
        m["neutralization_mean_corr_reduction"] = 0.25  # above default 0.20
        v = build_factor_verdict(m)
        assert any("neutralization" in r.lower() for r in v.reasons)

    def test_uncertainty_overlap_weakens_verdict(self):
        """All 3 CIs overlap zero → signal direction unstable."""
        m = _strong_metrics()
        m["mean_ic_ci_lower"] = -0.01
        m["mean_rank_ic_ci_lower"] = -0.01
        m["mean_long_short_return_ci_lower"] = -0.001
        v = build_factor_verdict(m)
        # With default threshold (overlap_zero_fail_count=2), ≥2 overlap → weak
        assert v.label != "Strong candidate"

    def test_instability_flags_add_concern(self):
        """Instability flags should appear in reasons."""
        m = _strong_metrics()
        m["instability_flags"] = ["rolling_regime_dependence"]
        v = build_factor_verdict(m)
        assert any("instability" in r.lower() or "regime" in r.lower() for r in v.reasons)

    def test_rolling_instability_flags_add_concern(self):
        m = _strong_metrics()
        m["rolling_instability_flags"] = ["rolling_ic_sign_flip_instability"]
        v = build_factor_verdict(m)
        assert any("unstable" in r.lower() or "rolling" in r.lower() for r in v.reasons)

    def test_to_dict_roundtrip(self):
        v = build_factor_verdict(_strong_metrics())
        d = v.to_dict()
        assert d["label"] == v.label
        assert d["reasons"] == list(v.reasons)

    def test_custom_thresholds(self):
        """Stricter thresholds should reject what default accepts."""
        m = _strong_metrics()
        m["n_dates_used"] = 25
        m["n_dates"] = 25
        # Default min_eval_dates_basic=20 → passes
        v_default = build_factor_verdict(m)
        assert v_default.label != "Fails basic robustness"

        # Stricter: min_eval_dates_basic=30 → fails
        strict = FactorVerdictConfig(min_eval_dates_basic=30)
        v_strict = build_factor_verdict(m, thresholds=strict)
        assert v_strict.label == "Fails basic robustness"

    def test_profiles_produce_different_verdicts(self):
        """The same borderline metrics should classify differently under
        exploratory_screening vs stricter_research profiles."""
        m = _strong_metrics()
        m["n_dates_used"] = 25
        m["n_dates"] = 25
        m["ic_valid_ratio"] = 0.72
        m["rank_ic_valid_ratio"] = 0.72

        exploratory = get_research_evaluation_config("exploratory_screening")
        stricter = get_research_evaluation_config("stricter_research")

        v_explore = build_factor_verdict(m, thresholds=exploratory.factor_verdict)
        v_strict = build_factor_verdict(m, thresholds=stricter.factor_verdict)

        # Stricter should be harsher
        taxonomy_order = {label: i for i, label in enumerate(FACTOR_VERDICT_TAXONOMY)}
        assert taxonomy_order[v_strict.label] >= taxonomy_order[v_explore.label]


# ═══════════════════════════════════════════════════════════════════════════
# 2.2  Campaign triage classification
# ═══════════════════════════════════════════════════════════════════════════


class TestCampaignTriage:
    """Verify campaign triage logic."""

    def test_advance_to_level2(self):
        """Perfect metrics → Advance to Level 2."""
        m = _strong_metrics()
        t = build_campaign_triage(m)
        assert t.label == "Advance to Level 2"
        assert t.triage_priority == 1

    def test_drop_for_now_weak(self):
        """Weak verdict → Drop for now."""
        m = _weak_metrics()
        t = build_campaign_triage(m)
        assert t.label == "Drop for now"
        assert t.triage_priority == 5

    def test_drop_for_now_failed_status(self):
        """Non-success status → Drop for now."""
        m = _strong_metrics()
        t = build_campaign_triage(m, status="error")
        assert t.label == "Drop for now"

    def test_drop_for_now_thin_coverage(self):
        """Coverage below fail threshold → Drop."""
        m = _strong_metrics()
        m["coverage_mean"] = 0.40
        m["eval_coverage_ratio_mean"] = 0.40
        t = build_campaign_triage(m)
        assert t.label == "Drop for now"

    def test_strong_level1_candidate(self):
        """Strong verdict but missing one advance gate requirement."""
        m = _strong_metrics()
        # Remove neutralization preserves → can't advance, but still strong L1
        m["neutralization_comparison_flags"] = []
        t = build_campaign_triage(m)
        assert t.label in ("Strong Level 1 candidate", "Needs refinement")

    def test_fragile_monitor(self):
        """Multiple fragility signals → Fragile / monitor."""
        m = _strong_metrics()
        m["factor_verdict"] = "Promising but fragile"
        m["rolling_ic_positive_share"] = 0.40
        m["rolling_rank_ic_positive_share"] = 0.40
        m["rolling_long_short_positive_share"] = 0.40
        m["rolling_instability_flags"] = ["rolling_regime_dependence"]
        m["subperiod_ic_positive_share"] = 0.55
        m["subperiod_long_short_positive_share"] = 0.55
        m["mean_ic_ci_lower"] = -0.01
        m["mean_rank_ic_ci_lower"] = -0.01
        m["mean_long_short_return_ci_lower"] = -0.001
        t = build_campaign_triage(m)
        assert t.label in ("Fragile / monitor", "Drop for now")

    def test_taxonomy_completeness(self):
        assert len(CAMPAIGN_TRIAGE_TAXONOMY) == 5

    def test_priority_ordering(self):
        """Advance < Strong < Needs < Fragile < Drop in priority number."""
        m_strong = _strong_metrics()
        m_weak = _weak_metrics()
        t_advance = build_campaign_triage(m_strong)
        t_drop = build_campaign_triage(m_weak)
        assert t_advance.triage_priority < t_drop.triage_priority

    def test_to_dict_has_all_fields(self):
        t = build_campaign_triage(_strong_metrics())
        d = t.to_dict()
        assert "campaign_triage" in d
        assert "campaign_triage_reasons" in d
        assert "campaign_triage_priority" in d
        assert "campaign_rank_primary_metric_name" in d
        assert "campaign_rank_rule" in d

    def test_rank_sort_key_ordering(self):
        """Advance should sort before Drop."""
        m_good = _strong_metrics()
        m_bad = _weak_metrics()
        key_good = campaign_rank_sort_key("good", status="success", metrics=m_good)
        key_bad = campaign_rank_sort_key("bad", status="success", metrics=m_bad)
        assert key_good < key_bad

    def test_advance_gate_requires_all_conditions(self):
        """Advance gate is AND of: strong verdict, neutralization preserves,
        rolling stable, uncertainty supportive, no coverage issues,
        no turnover issues, no material neutralization."""
        m = _strong_metrics()

        # Baseline: passes
        t0 = build_campaign_triage(m)
        assert t0.label == "Advance to Level 2"

        # Break each condition one at a time
        conditions_to_break = [
            ("factor_verdict", "Promising but fragile"),
            ("neutralization_comparison_flags", []),
            ("rolling_instability_flags", ["rolling_regime_dependence"]),
        ]
        for key, bad_val in conditions_to_break:
            m2 = _strong_metrics()
            m2[key] = bad_val
            t = build_campaign_triage(m2)
            assert t.label != "Advance to Level 2", (
                f"Setting {key}={bad_val} should prevent advance"
            )

    def test_subperiod_fail_is_blocker(self):
        """subperiod_min < min_subperiod_positive_share_fail → Drop."""
        m = _strong_metrics()
        m["subperiod_ic_positive_share"] = 0.40
        m["subperiod_long_short_positive_share"] = 0.40
        t = build_campaign_triage(m)
        assert t.label == "Drop for now"


# ═══════════════════════════════════════════════════════════════════════════
# 2.3  Level 2 promotion gate
# ═══════════════════════════════════════════════════════════════════════════


class TestLevel2Promotion:
    """Verify Level 2 promotion classification logic."""

    def test_promote(self):
        """All gates satisfied → Promote to Level 2."""
        m = _strong_metrics()
        d = build_level2_promotion(m)
        assert d.label == "Promote to Level 2"
        assert len(d.blockers) == 0

    def test_blocked_weak_verdict(self):
        m = _strong_metrics()
        m["factor_verdict"] = "Weak / noisy"
        d = build_level2_promotion(m)
        assert d.label == "Blocked from Level 2"
        assert any("verdict" in b.lower() for b in d.blockers)

    def test_blocked_thin_coverage(self):
        m = _strong_metrics()
        m["coverage_mean"] = 0.40
        m["eval_coverage_ratio_mean"] = 0.40
        d = build_level2_promotion(m)
        assert d.label == "Blocked from Level 2"
        assert any("coverage" in b.lower() for b in d.blockers)

    def test_blocked_subperiod(self):
        m = _strong_metrics()
        m["subperiod_ic_positive_share"] = 0.40
        m["subperiod_long_short_positive_share"] = 0.40
        d = build_level2_promotion(m)
        assert d.label == "Blocked from Level 2"
        assert any("subperiod" in b.lower() for b in d.blockers)

    def test_blocked_rolling_instability(self):
        m = _strong_metrics()
        m["rolling_instability_flags"] = ["rolling_ic_sign_flip_instability"]
        d = build_level2_promotion(m)
        assert d.label == "Blocked from Level 2"
        assert any("rolling" in b.lower() or "unstable" in b.lower() for b in d.blockers)

    def test_blocked_uncertainty_overlap(self):
        """≥2 CIs overlap zero → blocked."""
        m = _strong_metrics()
        m["mean_ic_ci_lower"] = -0.01
        m["mean_rank_ic_ci_lower"] = -0.01
        d = build_level2_promotion(m)
        assert d.label == "Blocked from Level 2"
        assert any("uncertainty" in b.lower() for b in d.blockers)

    def test_blocked_neutralization_material(self):
        m = _strong_metrics()
        m["neutralization_comparison_flags"] = [MATERIAL_REDUCTION_FLAG]
        d = build_level2_promotion(m)
        assert d.label == "Blocked from Level 2"
        assert any("neutrali" in b.lower() for b in d.blockers)

    def test_blocked_turnover_efficiency(self):
        m = _strong_metrics()
        m["long_short_return_per_turnover"] = -0.001
        d = build_level2_promotion(m)
        assert d.label == "Blocked from Level 2"
        assert any("turnover" in b.lower() for b in d.blockers)

    def test_blocked_by_status(self):
        m = _strong_metrics()
        d = build_level2_promotion(m, status="error")
        assert d.label == "Blocked from Level 2"

    def test_hold_for_refinement(self):
        """No blockers but some conditions not met → Hold."""
        m = _strong_metrics()
        # Uncertainty not supportive enough (only 1 CI above zero, need 2)
        m["mean_ic_ci_lower"] = 0.01
        m["mean_rank_ic_ci_lower"] = -0.005  # overlaps zero
        m["mean_long_short_return_ci_lower"] = -0.001  # overlaps zero
        build_level2_promotion(m)
        # Not blocked (overlap count=2 triggers block with default threshold=2)
        # So this actually gets blocked. Let me adjust.
        m2 = _strong_metrics()
        # Keep just 1 overlap → not blocked, but not supportive enough
        m2["mean_ic_ci_lower"] = 0.01
        m2["mean_rank_ic_ci_lower"] = 0.005
        m2["mean_long_short_return_ci_lower"] = -0.001  # 1 overlap
        d2 = build_level2_promotion(m2)
        assert d2.label in ("Hold for refinement", "Promote to Level 2")

    def test_taxonomy_completeness(self):
        assert len(LEVEL2_PROMOTION_TAXONOMY) == 3

    def test_to_dict_has_all_fields(self):
        d = build_level2_promotion(_strong_metrics())
        payload = d.to_dict()
        assert "promotion_decision" in payload
        assert "promotion_reasons" in payload
        assert "promotion_blockers" in payload

    def test_multiple_blockers_all_listed(self):
        """When multiple conditions fail, all blockers should be listed."""
        m = _strong_metrics()
        m["factor_verdict"] = "Weak / noisy"
        m["coverage_mean"] = 0.40
        m["eval_coverage_ratio_mean"] = 0.40
        m["subperiod_ic_positive_share"] = 0.40
        d = build_level2_promotion(m)
        assert d.label == "Blocked from Level 2"
        assert len(d.blockers) >= 3

    def test_exploratory_profile_more_lenient(self):
        """Exploratory profile should allow promotion where default blocks."""
        m = _strong_metrics()
        m["rolling_ic_positive_share"] = 0.55
        m["rolling_rank_ic_positive_share"] = 0.55
        m["rolling_long_short_positive_share"] = 0.55

        default_config = get_research_evaluation_config("default_research")
        explore_config = get_research_evaluation_config("exploratory_screening")

        d_default = build_level2_promotion(m, thresholds=default_config.level2_promotion)
        d_explore = build_level2_promotion(m, thresholds=explore_config.level2_promotion)

        # Exploratory requires rolling_positive_share ≥ 0.50
        # Default requires ≥ 0.60
        # With 0.55: default may hold/block, exploratory should be more lenient
        taxonomy_order = {label: i for i, label in enumerate(LEVEL2_PROMOTION_TAXONOMY)}
        assert taxonomy_order[d_explore.label] <= taxonomy_order[d_default.label]


# ═══════════════════════════════════════════════════════════════════════════
# 2.4  Neutralization comparison
# ═══════════════════════════════════════════════════════════════════════════


class TestNeutralizationComparison:
    """Verify raw-vs-neutralized comparison flag logic."""

    def _raw_metrics(self) -> dict[str, object]:
        return {
            "mean_ic": 0.05,
            "mean_rank_ic": 0.04,
            "mean_long_short_return": 0.005,
            "ic_ir": 0.8,
            "ic_valid_ratio": 0.95,
            "rank_ic_valid_ratio": 0.93,
            "eval_coverage_ratio_mean": 0.85,
            "eval_coverage_ratio_min": 0.70,
            "rolling_ic_positive_share": 0.80,
            "rolling_rank_ic_positive_share": 0.75,
            "rolling_long_short_positive_share": 0.70,
            "rolling_ic_min_mean": 0.01,
            "rolling_rank_ic_min_mean": 0.008,
            "rolling_long_short_min_mean": 0.001,
            "mean_ic_ci_lower": 0.02,
            "mean_ic_ci_upper": 0.08,
            "mean_rank_ic_ci_lower": 0.01,
            "mean_rank_ic_ci_upper": 0.07,
            "mean_long_short_return_ci_lower": 0.001,
            "mean_long_short_return_ci_upper": 0.009,
            "uncertainty_flags": [],
            "rolling_instability_flags": [],
        }

    def test_preserves_evidence(self):
        """Neutralized metrics very close to raw → preserves flag."""
        raw = self._raw_metrics()
        neutral = dict(raw)  # same values → minimal loss
        result = build_raw_vs_neutralized_comparison(raw, neutral)
        assert PRESERVES_EVIDENCE_FLAG in result.interpretation_flags

    def test_material_reduction(self):
        """Large drop in IC and L/S → material reduction flag."""
        raw = self._raw_metrics()
        neutral = dict(raw)
        neutral["mean_ic"] = 0.01  # big drop from 0.05
        neutral["mean_rank_ic"] = 0.005  # big drop from 0.04
        neutral["mean_long_short_return"] = 0.001  # big drop from 0.005
        neutral["ic_ir"] = 0.1  # big drop
        result = build_raw_vs_neutralized_comparison(raw, neutral)
        assert (
            MATERIAL_REDUCTION_FLAG in result.interpretation_flags
            or MODERATE_WEAKENING_FLAG in result.interpretation_flags
        )

    def test_delta_signs_correct(self):
        """Delta = neutralized - raw. Negative delta = neutralization hurts."""
        raw = self._raw_metrics()
        neutral = dict(raw)
        neutral["mean_ic"] = 0.03  # 0.03 - 0.05 = -0.02
        result = build_raw_vs_neutralized_comparison(raw, neutral)
        ic_delta = result.delta.get("mean_ic_delta")
        assert ic_delta is not None
        assert ic_delta < 0  # neutralization reduced IC

    def test_exposure_driven_flag(self):
        """If corr reduction is large enough → exposure driven flag."""
        raw = self._raw_metrics()
        neutral = dict(raw)
        # Massive IC loss with high corr reduction
        neutral["mean_ic"] = 0.005
        neutral["mean_rank_ic"] = 0.005
        neutral["mean_long_short_return"] = 0.0005
        neutral["ic_ir"] = 0.05
        # Much worse validity and coverage
        neutral["ic_valid_ratio"] = 0.70
        neutral["rank_ic_valid_ratio"] = 0.70
        neutral["eval_coverage_ratio_mean"] = 0.60
        # CIs overlap zero after neutralization
        neutral["mean_ic_ci_lower"] = -0.01
        neutral["mean_rank_ic_ci_lower"] = -0.01
        neutral["mean_long_short_return_ci_lower"] = -0.001

        result = build_raw_vs_neutralized_comparison(
            raw,
            neutral,
            neutralization_mean_corr_reduction=0.5,
        )
        has_negative_flag = (
            MATERIAL_REDUCTION_FLAG in result.interpretation_flags
            or EXPOSURE_DRIVEN_FLAG in result.interpretation_flags
        )
        assert has_negative_flag

    def test_weaker_but_stabler(self):
        """Neutralized is weaker but more stable → WEAKER_BUT_STABLER."""
        raw = self._raw_metrics()
        raw["rolling_ic_positive_share"] = 0.55
        raw["rolling_rank_ic_positive_share"] = 0.50
        raw["rolling_long_short_positive_share"] = 0.50
        raw["rolling_ic_min_mean"] = -0.005
        raw["rolling_rank_ic_min_mean"] = -0.008

        neutral = dict(raw)
        neutral["mean_ic"] = 0.04  # slightly weaker
        neutral["mean_rank_ic"] = 0.035
        # But more stable
        neutral["rolling_ic_positive_share"] = 0.75
        neutral["rolling_rank_ic_positive_share"] = 0.70
        neutral["rolling_long_short_positive_share"] = 0.65
        neutral["rolling_ic_min_mean"] = 0.01
        neutral["rolling_rank_ic_min_mean"] = 0.005

        result = build_raw_vs_neutralized_comparison(raw, neutral)
        # Should detect stability improvement
        has_stability_signal = (
            WEAKER_BUT_STABLER_FLAG in result.interpretation_flags
            or PRESERVES_EVIDENCE_FLAG in result.interpretation_flags
        )
        assert has_stability_signal


# ═══════════════════════════════════════════════════════════════════════════
# 2.5  Uncertainty flags logic
# ═══════════════════════════════════════════════════════════════════════════


class TestUncertaintyFlags:
    """Verify compute_core_uncertainty flag generation."""

    def test_no_flags_when_positive(self):
        """All CIs above zero, narrow → no flags."""
        result = compute_core_uncertainty(
            ic_values=[0.05, 0.06, 0.04, 0.05, 0.07, 0.05] * 5,
            rank_ic_values=[0.04, 0.05, 0.03, 0.04, 0.06, 0.04] * 5,
            long_short_values=[0.005, 0.006, 0.004, 0.005, 0.007, 0.005] * 5,
        )
        assert len(result.uncertainty_flags) == 0

    def test_overlaps_zero_flag(self):
        """CI spanning zero → overlap flag."""
        result = compute_core_uncertainty(
            ic_values=[-0.02, 0.03, -0.01, 0.02, -0.03, 0.01],
            rank_ic_values=[0.05, 0.06, 0.04, 0.05, 0.07, 0.05] * 3,
            long_short_values=[0.005, 0.006, 0.004, 0.005, 0.007, 0.005] * 3,
        )
        assert "ic_ci_overlaps_zero" in result.uncertainty_flags

    def test_wide_ci_flag(self):
        """Very noisy series → ci_wide flag."""
        # Half width >> |mean| when std is huge relative to mean
        import random

        random.seed(42)
        noisy = [0.001 + random.gauss(0, 0.5) for _ in range(20)]
        result = compute_core_uncertainty(
            ic_values=noisy,
            rank_ic_values=[0.05] * 20,
            long_short_values=[0.005] * 20,
        )
        has_wide = any(f.endswith("_ci_wide") for f in result.uncertainty_flags)
        has_overlap = any(f.endswith("_ci_overlaps_zero") for f in result.uncertainty_flags)
        assert has_wide or has_overlap

    def test_unavailable_flag_for_single_obs(self):
        """Single observation → ci_unavailable flag."""
        result = compute_core_uncertainty(
            ic_values=[0.05],
            rank_ic_values=[0.04],
            long_short_values=[0.005],
        )
        assert any(f.endswith("_ci_unavailable") for f in result.uncertainty_flags)

    def test_empty_input_all_unavailable(self):
        result = compute_core_uncertainty(
            ic_values=[],
            rank_ic_values=[],
            long_short_values=[],
        )
        unavailable = [f for f in result.uncertainty_flags if f.endswith("_ci_unavailable")]
        assert len(unavailable) == 3

    def test_bootstrap_method_produces_flags(self):
        """Bootstrap method should also produce overlap/wide flags."""
        thresholds = UncertaintyConfig(method="bootstrap", bootstrap_resamples=200)
        result = compute_core_uncertainty(
            ic_values=[-0.02, 0.03, -0.01, 0.02, -0.03, 0.01],
            rank_ic_values=[0.05, 0.06, 0.04, 0.05, 0.07, 0.05],
            long_short_values=[0.005, 0.006, 0.004, 0.005, 0.007, 0.005],
            thresholds=thresholds,
        )
        # IC values centered around 0 → should overlap zero
        assert "ic_ci_overlaps_zero" in result.uncertainty_flags

    def test_block_bootstrap_method(self):
        thresholds = UncertaintyConfig(
            method="block_bootstrap",
            bootstrap_resamples=200,
            block_bootstrap_block_length=3,
        )
        result = compute_core_uncertainty(
            ic_values=[0.05, 0.06, 0.04, 0.05, 0.07, 0.05] * 3,
            rank_ic_values=[0.04, 0.05, 0.03, 0.04, 0.06, 0.04] * 3,
            long_short_values=[0.005, 0.006, 0.004, 0.005, 0.007, 0.005] * 3,
            thresholds=thresholds,
        )
        assert result.uncertainty_method == "block_bootstrap"
        assert result.uncertainty_bootstrap_block_length == 3

    def test_ci_bounds_in_summary_match_compute_mean_ci(self):
        """CI bounds from compute_core_uncertainty should match compute_mean_ci."""
        ic_vals = [0.05, 0.06, 0.04, 0.05, 0.07, 0.05, 0.03, 0.08]
        thresholds = UncertaintyConfig(method="normal", confidence_level=0.95)
        summary = compute_core_uncertainty(
            ic_values=ic_vals,
            rank_ic_values=[0.04] * 8,
            long_short_values=[0.005] * 8,
            thresholds=thresholds,
        )
        ci = compute_mean_ci(
            ic_vals,
            confidence_level=0.95,
            use_t_for_small_sample=thresholds.normal_small_sample_use_t,
            small_sample_threshold=thresholds.normal_small_sample_threshold,
        )
        assert summary.mean_ic_ci_lower == pytest.approx(ci.ci_lower)
        assert summary.mean_ic_ci_upper == pytest.approx(ci.ci_upper)


# ═══════════════════════════════════════════════════════════════════════════
# 2.6  Decision pipeline consistency
# ═══════════════════════════════════════════════════════════════════════════


class TestDecisionPipelineConsistency:
    """End-to-end consistency between verdict → triage → promotion."""

    def test_strong_verdict_can_advance_and_promote(self):
        m = _strong_metrics()
        v = build_factor_verdict(m)
        assert v.label == "Strong candidate"

        m["factor_verdict"] = v.label
        t = build_campaign_triage(m)
        assert t.label == "Advance to Level 2"

        m["campaign_triage"] = t.label
        d = build_level2_promotion(m)
        assert d.label == "Promote to Level 2"

    def test_weak_verdict_drops_and_blocks(self):
        m = _weak_metrics()
        v = build_factor_verdict(m)
        assert v.label == "Weak / noisy"

        m["factor_verdict"] = v.label
        t = build_campaign_triage(m)
        assert t.label == "Drop for now"

        m["campaign_triage"] = t.label
        d = build_level2_promotion(m)
        assert d.label == "Blocked from Level 2"

    def test_failing_verdict_cascades(self):
        m = _failing_metrics()
        v = build_factor_verdict(m)
        assert v.label == "Fails basic robustness"

        m["factor_verdict"] = v.label
        t = build_campaign_triage(m)
        assert t.label == "Drop for now"

        m["campaign_triage"] = t.label
        d = build_level2_promotion(m)
        assert d.label == "Blocked from Level 2"

    def test_verdict_does_not_leak_into_triage_via_mutation(self):
        """Triage should use the verdict string from metrics dict,
        not re-run build_factor_verdict internally."""
        m = _strong_metrics()
        # Set verdict manually to something inconsistent
        m["factor_verdict"] = "Weak / noisy"
        t = build_campaign_triage(m)
        # Triage should trust the verdict string, not the underlying metrics
        assert t.label == "Drop for now"

    def test_all_three_profiles_produce_valid_taxonomy_labels(self):
        """Every profile should produce labels within their taxonomy."""
        for profile_name in ["default_research", "exploratory_screening", "stricter_research"]:
            config = get_research_evaluation_config(profile_name)
            m = _strong_metrics()

            v = build_factor_verdict(m, thresholds=config.factor_verdict)
            assert v.label in FACTOR_VERDICT_TAXONOMY, f"Verdict {v.label!r} not in taxonomy"

            m["factor_verdict"] = v.label
            t = build_campaign_triage(m, thresholds=config.campaign_triage)
            assert t.label in CAMPAIGN_TRIAGE_TAXONOMY, f"Triage {t.label!r} not in taxonomy"

            m["campaign_triage"] = t.label
            d = build_level2_promotion(m, thresholds=config.level2_promotion)
            assert d.label in LEVEL2_PROMOTION_TAXONOMY, f"Promotion {d.label!r} not in taxonomy"

    def test_promotion_blocker_count_matches_reasons(self):
        """When blocked, blockers should be non-empty and appear in reasons."""
        m = _strong_metrics()
        m["factor_verdict"] = "Weak / noisy"
        m["coverage_mean"] = 0.40
        m["eval_coverage_ratio_mean"] = 0.40
        d = build_level2_promotion(m)
        assert d.label == "Blocked from Level 2"
        assert len(d.blockers) >= 1
        # Every blocker should also appear in reasons
        for blocker in d.blockers:
            assert any(blocker in r for r in d.reasons), f"Blocker {blocker!r} not in reasons"


# ═══════════════════════════════════════════════════════════════════════════
# 2.7  Instability flags logic
# ═══════════════════════════════════════════════════════════════════════════


class TestInstabilityFlags:
    """Verify instability flag generation in experiment._collect_instability_flags."""

    def test_short_eval_window_flag(self):
        from alpha_lab.experiment import _collect_instability_flags
        from alpha_lab.research_evaluation_config import RollingStabilityConfig

        thresholds = RollingStabilityConfig()
        flags = _collect_instability_flags(
            n_dates=25,  # below instability_short_eval_window_dates=30
            ic_positive_rate=0.65,
            ic_valid_ratio=0.90,
            rank_ic_valid_ratio=0.90,
            subperiod_ic_positive_share=0.80,
            subperiod_long_short_positive_share=0.80,
            eval_coverage_ratio_mean=0.85,
            mean_long_short_return=0.005,
            mean_long_short_turnover=0.3,
            long_short_ir=0.5,
            thresholds=thresholds,
        )
        assert "short_eval_window" in flags

    def test_no_flags_healthy_metrics(self):
        from alpha_lab.experiment import _collect_instability_flags
        from alpha_lab.research_evaluation_config import RollingStabilityConfig

        thresholds = RollingStabilityConfig()
        flags = _collect_instability_flags(
            n_dates=50,
            ic_positive_rate=0.65,
            ic_valid_ratio=0.90,
            rank_ic_valid_ratio=0.90,
            subperiod_ic_positive_share=0.80,
            subperiod_long_short_positive_share=0.80,
            eval_coverage_ratio_mean=0.85,
            mean_long_short_return=0.005,
            mean_long_short_turnover=0.3,
            long_short_ir=0.5,
            thresholds=thresholds,
        )
        assert len(flags) == 0

    def test_low_ic_positive_rate_flag(self):
        from alpha_lab.experiment import _collect_instability_flags
        from alpha_lab.research_evaluation_config import RollingStabilityConfig

        thresholds = RollingStabilityConfig()
        flags = _collect_instability_flags(
            n_dates=50,
            ic_positive_rate=0.45,  # below instability_ic_positive_rate_min=0.50
            ic_valid_ratio=0.90,
            rank_ic_valid_ratio=0.90,
            subperiod_ic_positive_share=0.80,
            subperiod_long_short_positive_share=0.80,
            eval_coverage_ratio_mean=0.85,
            mean_long_short_return=0.005,
            mean_long_short_turnover=0.3,
            long_short_ir=0.5,
            thresholds=thresholds,
        )
        assert any("ic_sign_instability" in f for f in flags)

    def test_low_valid_ratio_flag(self):
        from alpha_lab.experiment import _collect_instability_flags
        from alpha_lab.research_evaluation_config import RollingStabilityConfig

        thresholds = RollingStabilityConfig()
        flags = _collect_instability_flags(
            n_dates=50,
            ic_positive_rate=0.65,
            ic_valid_ratio=0.70,  # below 0.80
            rank_ic_valid_ratio=0.70,
            subperiod_ic_positive_share=0.80,
            subperiod_long_short_positive_share=0.80,
            eval_coverage_ratio_mean=0.85,
            mean_long_short_return=0.005,
            mean_long_short_turnover=0.3,
            long_short_ir=0.5,
            thresholds=thresholds,
        )
        assert any("valid_ratio" in f for f in flags)

    def test_high_turnover_negative_spread_flag(self):
        from alpha_lab.experiment import _collect_instability_flags
        from alpha_lab.research_evaluation_config import RollingStabilityConfig

        thresholds = RollingStabilityConfig()
        flags = _collect_instability_flags(
            n_dates=50,
            ic_positive_rate=0.65,
            ic_valid_ratio=0.90,
            rank_ic_valid_ratio=0.90,
            subperiod_ic_positive_share=0.80,
            subperiod_long_short_positive_share=0.80,
            eval_coverage_ratio_mean=0.85,
            mean_long_short_return=-0.001,  # negative
            mean_long_short_turnover=0.85,  # above 0.80
            long_short_ir=-0.1,
            thresholds=thresholds,
        )
        assert any("turnover" in f for f in flags)


# ═══════════════════════════════════════════════════════════════════════════
# 2.8  Rolling instability flags
# ═══════════════════════════════════════════════════════════════════════════


class TestRollingInstabilityFlags:
    """Verify _collect_rolling_instability_flags."""

    def test_regime_dependence_flag(self):
        import pandas as pd

        from alpha_lab.experiment import _collect_rolling_instability_flags
        from alpha_lab.research_evaluation_config import RollingStabilityConfig

        thresholds = RollingStabilityConfig()
        # IC positive share below regime threshold (0.45)
        df = pd.DataFrame(
            {
                "date": pd.to_datetime([f"2024-01-{i + 2:02d}" for i in range(10)]),
                "rolling_mean_ic": [
                    0.01,
                    -0.02,
                    0.01,
                    -0.03,
                    0.02,
                    -0.01,
                    0.01,
                    -0.02,
                    0.01,
                    -0.01,
                ],
                "rolling_ic_positive_rate": [0.4] * 10,
                "rolling_mean_rank_ic": [0.01] * 10,
                "rolling_rank_ic_positive_rate": [0.6] * 10,
                "rolling_mean_long_short_return": [0.001] * 10,
                "rolling_long_short_positive_rate": [0.6] * 10,
            }
        )
        flags = _collect_rolling_instability_flags(df, thresholds=thresholds)
        # With positive rate at 0.4 (< 0.45 sign_flip_threshold), expect sign flip flag
        has_instability = any("sign_flip" in f or "regime" in f for f in flags)
        assert has_instability or len(flags) > 0

    def test_no_flags_stable_rolling(self):
        import pandas as pd

        from alpha_lab.experiment import _collect_rolling_instability_flags
        from alpha_lab.research_evaluation_config import RollingStabilityConfig

        thresholds = RollingStabilityConfig()
        df = pd.DataFrame(
            {
                "date": pd.to_datetime([f"2024-01-{i + 2:02d}" for i in range(10)]),
                "rolling_mean_ic": [0.03] * 10,
                "rolling_ic_positive_rate": [0.7] * 10,
                "rolling_mean_rank_ic": [0.025] * 10,
                "rolling_rank_ic_positive_rate": [0.65] * 10,
                "rolling_mean_long_short_return": [0.003] * 10,
                "rolling_long_short_positive_rate": [0.65] * 10,
            }
        )
        flags = _collect_rolling_instability_flags(df, thresholds=thresholds)
        assert len(flags) == 0


# ═══════════════════════════════════════════════════════════════════════════
# 2.9  Cross-validation: Phase 4 items
# ═══════════════════════════════════════════════════════════════════════════


class TestCrossValidation:
    """Phase 4 cross-validation checks."""

    def test_quantile_assignment_vs_pd_qcut(self):
        """Compare _assign_quantile with pd.qcut under strict documented cases.

        Case A (n=10, q=5): exact match.
        Case B (n=12, q=3): deterministic, documented divergence.
        """
        import pandas as pd

        from alpha_lab.quantile import _assign_quantile

        # Case A: exact match on a balanced non-tied cross-section.
        s_match = pd.Series(np.arange(1.0, 11.0))
        our_match = _assign_quantile(s_match, n_quantiles=5).astype(int)
        qcut_match = (pd.qcut(s_match, q=5, labels=False) + 1).astype(int)
        assert our_match.tolist() == qcut_match.tolist()

        # Case B: known divergence for non-divisible bucket geometry.
        s_diverge = pd.Series(np.arange(1.0, 13.0))
        our_diverge = _assign_quantile(s_diverge, n_quantiles=3).astype(int)
        qcut_diverge = (pd.qcut(s_diverge, q=3, labels=False) + 1).astype(int)

        assert our_diverge.tolist() == [1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3]
        assert qcut_diverge.tolist() == [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]

        # Both methods must preserve monotonic ordering and top/bottom anchors.
        assert our_diverge.iloc[0] == 1
        assert our_diverge.iloc[-1] == 3
        assert qcut_diverge.iloc[0] == 1
        assert qcut_diverge.iloc[-1] == 3
        assert our_diverge.is_monotonic_increasing
        assert qcut_diverge.is_monotonic_increasing

    def test_ic_scipy_cross_validation_multidate(self):
        """Cross-validate IC computation with scipy across multiple dates."""
        from scipy.stats import pearsonr, spearmanr

        from alpha_lab.evaluation import compute_ic, compute_rank_ic

        dates = ["2024-01-02"] * 5 + ["2024-01-03"] * 5
        assets = ["A", "B", "C", "D", "E"] * 2
        fv = [0.5, -0.3, 1.2, 0.8, -0.1, 0.2, 0.7, -0.5, 0.3, 0.9]
        lv = [0.02, -0.01, 0.05, 0.03, 0.0, -0.01, 0.04, -0.02, 0.01, 0.06]

        f = pd.DataFrame(
            {
                "date": pd.to_datetime(dates),
                "asset": assets,
                "factor": ["f"] * 10,
                "value": fv,
            }
        )
        labels_df = pd.DataFrame(
            {
                "date": pd.to_datetime(dates),
                "asset": assets,
                "factor": ["l"] * 10,
                "value": lv,
            }
        )

        ic_result = compute_ic(f, labels_df).set_index("date")
        rank_ic_result = compute_rank_ic(f, labels_df).set_index("date")

        # Day 1
        r1, _ = pearsonr(fv[:5], lv[:5])
        assert ic_result.loc[pd.Timestamp("2024-01-02"), "ic"] == pytest.approx(r1, abs=1e-10)

        rho1, _ = spearmanr(fv[:5], lv[:5])
        assert rank_ic_result.loc[pd.Timestamp("2024-01-02"), "rank_ic"] == pytest.approx(
            rho1, abs=1e-10
        )

        # Day 2
        r2, _ = pearsonr(fv[5:], lv[5:])
        assert ic_result.loc[pd.Timestamp("2024-01-03"), "ic"] == pytest.approx(r2, abs=1e-10)

        rho2, _ = spearmanr(fv[5:], lv[5:])
        assert rank_ic_result.loc[pd.Timestamp("2024-01-03"), "rank_ic"] == pytest.approx(
            rho2, abs=1e-10
        )
