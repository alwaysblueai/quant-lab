"""Walk-forward verdict overlay.

Given a base factor verdict derived from the full-sample evaluation and a
:class:`~alpha_lab.walk_forward.WalkForwardAggregate` from a matching
walk-forward run, this module produces an adjusted verdict that integrates
out-of-sample evidence.

The base single-factor pipeline does not run walk-forward; walk-forward is
an opt-in, higher-cost path.  Callers that produce both artifacts can apply
this overlay to obtain a verdict reflecting the additional OOS signal.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from alpha_lab.reporting.factor_verdict import FACTOR_VERDICT_TAXONOMY
from alpha_lab.walk_forward import WalkForwardAggregate

# Ordered worst -> best; one step right is a downgrade.
_VERDICT_ORDER: tuple[str, ...] = (
    "Fails basic robustness",
    "Weak / noisy",
    "Mixed evidence",
    "Promising but fragile",
    "Strong candidate",
)

# IC-IR ratio (WF / IS) below this fraction triggers a downgrade.
_IC_IR_DEGRADATION_RATIO = 0.5
# Absolute floor below which WF IC-IR is considered not supportive.
_IC_IR_WEAK_FLOOR = 0.0
# DSR p-value above this implies OOS Sharpe is not distinguishable.
_DSR_WEAK_PVALUE = 0.5


@dataclass(frozen=True)
class WalkForwardOverlayResult:
    """Result of applying walk-forward evidence to a base verdict."""

    base_label: str
    adjusted_label: str
    downgrade_steps: int
    reasons: tuple[str, ...]
    walk_forward_ic_ir: float
    walk_forward_pooled_ic_mean: float
    walk_forward_dsr_pvalue: float
    walk_forward_n_folds: int

    def to_dict(self) -> dict[str, object]:
        return {
            "walk_forward_overlay_applied": True,
            "walk_forward_base_verdict": self.base_label,
            "walk_forward_adjusted_verdict": self.adjusted_label,
            "walk_forward_downgrade_steps": int(self.downgrade_steps),
            "walk_forward_overlay_reasons": list(self.reasons),
            "walk_forward_ic_ir": self.walk_forward_ic_ir,
            "walk_forward_pooled_ic_mean": self.walk_forward_pooled_ic_mean,
            "walk_forward_dsr_pvalue": self.walk_forward_dsr_pvalue,
            "walk_forward_n_folds": int(self.walk_forward_n_folds),
        }


def build_walk_forward_verdict_overlay(
    base_metrics: Mapping[str, object],
    wf_aggregate: WalkForwardAggregate,
) -> WalkForwardOverlayResult:
    """Adjust ``factor_verdict`` using walk-forward aggregate evidence.

    Downgrade rules (each firing moves the verdict one step toward the
    "Weak / noisy" end of :data:`FACTOR_VERDICT_TAXONOMY`):

    - Pooled OOS IC-IR is materially below in-sample IC-IR
      (``pooled_ic_ir / ic_ir < 0.5``) or falls at/below zero.
    - Pooled OOS mean IC has the opposite sign of the in-sample mean IC.
    - Deflated-Sharpe p-value on pooled OOS returns is not below 0.5.

    The adjusted label never moves below ``"Weak / noisy"``; escalation to
    ``"Fails basic robustness"`` remains reserved for hard-threshold failures
    in the base verdict pipeline.
    """
    base_label = str(base_metrics.get("factor_verdict") or "").strip()
    if base_label not in FACTOR_VERDICT_TAXONOMY:
        base_label = "Mixed evidence"

    is_mean_ic = _coerce_float(base_metrics.get("mean_ic"))
    is_ic_ir = _coerce_float(base_metrics.get("ic_ir"))

    wf_ic_ir = wf_aggregate.pooled_ic_ir
    wf_ic_mean = wf_aggregate.pooled_ic_mean
    wf_dsr = wf_aggregate.dsr_pvalue

    reasons: list[str] = []
    downgrade = 0

    if math.isfinite(wf_ic_ir) and math.isfinite(is_ic_ir) and is_ic_ir > 0:
        ratio = wf_ic_ir / is_ic_ir
        if wf_ic_ir <= _IC_IR_WEAK_FLOOR:
            downgrade += 1
            reasons.append("walk-forward pooled IC-IR is non-positive")
        elif ratio < _IC_IR_DEGRADATION_RATIO:
            downgrade += 1
            reasons.append("walk-forward pooled IC-IR materially below in-sample level")
    elif math.isfinite(wf_ic_ir) and wf_ic_ir <= _IC_IR_WEAK_FLOOR:
        downgrade += 1
        reasons.append("walk-forward pooled IC-IR is non-positive")

    if (
        math.isfinite(wf_ic_mean)
        and math.isfinite(is_mean_ic)
        and is_mean_ic != 0
        and (wf_ic_mean * is_mean_ic) < 0
    ):
        downgrade += 1
        reasons.append("walk-forward pooled IC has opposite sign to in-sample")

    if math.isfinite(wf_dsr) and wf_dsr >= _DSR_WEAK_PVALUE:
        downgrade += 1
        reasons.append("walk-forward deflated-Sharpe p-value is not significant")

    adjusted_label = _downgrade_label(base_label, downgrade)

    return WalkForwardOverlayResult(
        base_label=base_label,
        adjusted_label=adjusted_label,
        downgrade_steps=downgrade,
        reasons=tuple(reasons),
        walk_forward_ic_ir=wf_ic_ir,
        walk_forward_pooled_ic_mean=wf_ic_mean,
        walk_forward_dsr_pvalue=wf_dsr,
        walk_forward_n_folds=wf_aggregate.n_folds,
    )


def apply_walk_forward_overlay(
    metrics: dict[str, object],
    wf_aggregate: WalkForwardAggregate,
) -> WalkForwardOverlayResult:
    """In-place variant: apply overlay to a metrics dict and return the result.

    Mutates ``metrics`` by writing the overlay fields and replacing
    ``factor_verdict`` / ``factor_verdict_reasons`` with the adjusted label and
    merged reasons.  The original base verdict is preserved under
    ``walk_forward_base_verdict``.
    """
    overlay = build_walk_forward_verdict_overlay(metrics, wf_aggregate)
    metrics.update(overlay.to_dict())
    if overlay.downgrade_steps > 0:
        existing_value = metrics.get("factor_verdict_reasons")
        existing = (
            [str(item) for item in existing_value]
            if isinstance(existing_value, (list, tuple))
            else []
        )
        metrics["factor_verdict"] = overlay.adjusted_label
        metrics["factor_verdict_reasons"] = [*existing, *overlay.reasons]
    return overlay


def _downgrade_label(label: str, steps: int) -> str:
    if steps <= 0:
        return label
    try:
        idx = _VERDICT_ORDER.index(label)
    except ValueError:
        return label
    # Don't drop below "Weak / noisy" (index 1); "Fails basic robustness"
    # is reserved for hard failures surfaced by the base pipeline.
    min_idx = 1
    new_idx = max(min_idx, idx - steps)
    return _VERDICT_ORDER[new_idx]


def _coerce_float(value: object) -> float:
    if value is None or isinstance(value, bool):
        return math.nan
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return math.nan
        try:
            return float(text)
        except ValueError:
            return math.nan
    return math.nan
