from __future__ import annotations

import pandas as pd

from alpha_lab.rebalance_recommendation import (
    recommend_rebalance_cadence,
    recommend_rebalance_from_decay_profile,
)


def test_recommend_rebalance_cadence_fast_signal() -> None:
    rec = recommend_rebalance_cadence(half_life_periods=4.0, turnover_context=1.0)
    assert rec.cadence_periods <= 3
    assert rec.cadence_label in {"daily_or_2day", "weekly"}


def test_recommend_rebalance_cadence_slow_signal() -> None:
    rec = recommend_rebalance_cadence(half_life_periods=40.0, turnover_context=0.2)
    assert rec.cadence_periods >= 15


def test_recommend_rebalance_cadence_fallback_when_missing_half_life() -> None:
    rec = recommend_rebalance_cadence(half_life_periods=None)
    assert rec.cadence_periods == 5


def test_recommend_rebalance_from_decay_profile() -> None:
    decay = pd.DataFrame(
        {
            "horizon": [1, 2, 5, 10, 20],
            "mean_rank_ic": [0.08, 0.06, 0.04, 0.02, 0.01],
        }
    )
    rec = recommend_rebalance_from_decay_profile(decay, turnover_context=0.5)
    assert rec.cadence_periods >= 1

