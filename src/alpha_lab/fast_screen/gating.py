"""PASS / WARN / FAIL gate rules for the fast-screen page.

The rules are deliberately few and conservative. A factor failing any FAIL
rule should not be pushed to Tier-2 without an override. WARN rules flag
caveats that the researcher should acknowledge, not hard-stop.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeGuard

from .contracts import FastScreenResult, MetricStatus, Verdict


@dataclass(frozen=True)
class GateThresholds:
    """Numeric thresholds for FAIL / WARN gates.

    Defaults chosen for daily-frequency A-share universe. Adjust per project.
    """

    fail_abs_mean_rank_ic: float = 0.015
    fail_rank_ic_ir: float = 0.2
    fail_ls_sharpe_net: float = 0.3
    fail_effective_days: int = 250

    warn_ic_positive_ratio: float = 0.52
    warn_half_life_min: float = 1.0
    warn_turnover_high: float = 0.6
    warn_sharpe_when_turnover_high: float = 0.8
    warn_monotonicity_tau: float = 0.3


DEFAULT_GATES = GateThresholds()


def _finite(value: float | None) -> TypeGuard[float]:
    return value is not None and math.isfinite(value)


def evaluate_gates(
    result: FastScreenResult,
    *,
    thresholds: GateThresholds = DEFAULT_GATES,
    integrity_passed: bool = True,
) -> Verdict:
    """Apply gating rules to a Tier-1 result and produce the verdict.

    ``integrity_passed`` is an explicit signal from the Tier-1 integrity_quick
    check; we keep it out of the MetricCard list so the integrity status is
    never mistaken for a scalar metric.
    """
    by_key = {m.key: m for m in result.metrics}
    fails: list[str] = []
    warns: list[str] = []

    def _val(key: str) -> float | None:
        card = by_key.get(key)
        if card is None:
            return None
        if card.status is not MetricStatus.COMPUTED and card.status is not MetricStatus.PARTIAL:
            return None
        return card.value

    ic = _val("mean_rank_ic")
    if _finite(ic) and abs(ic) < thresholds.fail_abs_mean_rank_ic:
        fails.append(f"|mean_rank_ic| < {thresholds.fail_abs_mean_rank_ic:.3f}")

    ir = _val("rank_ic_ir")
    if _finite(ir) and ir < thresholds.fail_rank_ic_ir:
        fails.append(f"rank_ic_ir < {thresholds.fail_rank_ic_ir:.2f}")

    sharpe = _val("ls_sharpe_net")
    if _finite(sharpe) and sharpe < thresholds.fail_ls_sharpe_net:
        fails.append(f"ls_sharpe_net < {thresholds.fail_ls_sharpe_net:.2f}")

    if not integrity_passed:
        fails.append("integrity check failed")

    coverage_card = by_key.get("coverage")
    if coverage_card is not None:
        eff_days = coverage_card.secondary.get("effective_days")
        try:
            eff_days_int = int(eff_days) if eff_days is not None else None
        except (TypeError, ValueError):
            eff_days_int = None
        if (
            coverage_card.status is MetricStatus.PARTIAL
            and eff_days_int is not None
            and eff_days_int < thresholds.fail_effective_days
        ):
            fails.append(f"coverage partial & effective_days < {thresholds.fail_effective_days}")

    pos = _val("ic_positive_ratio")
    if _finite(pos) and pos < thresholds.warn_ic_positive_ratio:
        warns.append(f"ic_positive_ratio < {thresholds.warn_ic_positive_ratio:.2f}")

    half_life = _val("ic_half_life")
    if _finite(half_life) and half_life < thresholds.warn_half_life_min:
        warns.append("half-life < 1 (signal decays too fast)")

    turnover = _val("turnover")
    if (
        _finite(turnover)
        and turnover > thresholds.warn_turnover_high
        and _finite(sharpe)
        and sharpe < thresholds.warn_sharpe_when_turnover_high
    ):
        warns.append("high turnover with weak net Sharpe")

    mono_card = by_key.get("group_monotonicity")
    if mono_card is not None and mono_card.status is MetricStatus.COMPUTED:
        tau = mono_card.secondary.get("kendall_tau")
        try:
            tau_f = float(tau) if tau is not None else None
        except (TypeError, ValueError):
            tau_f = None
        if tau_f is not None and abs(tau_f) < thresholds.warn_monotonicity_tau:
            warns.append(f"|monotonicity tau| < {thresholds.warn_monotonicity_tau:.2f}")

    for card in result.metrics:
        if card.status is MetricStatus.MISSING_INPUT:
            warns.append(f"{card.key}: missing input")

    if fails:
        return Verdict(
            status="fail",
            triggered_rules=fails[:3],
            next_step="Do not promote. Investigate data or drop factor.",
        )
    if warns:
        return Verdict(
            status="warn",
            triggered_rules=warns[:3],
            next_step="Review flagged caveats before running deep dive.",
        )
    return Verdict(
        status="pass",
        triggered_rules=[],
        next_step="Eligible for Tier-2 deep dive.",
    )


_CALLABLES: dict[str, Callable[..., object]] = {"evaluate_gates": evaluate_gates}
