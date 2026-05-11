"""Single-factor evaluation package.

Public API is re-exported here so existing callers using
``from alpha_lab.real_cases.single_factor.evaluate import ...`` keep working.
The implementation is split across topic modules for maintainability.
"""

from __future__ import annotations

from ._utils import _evaluate_variant_lightweight
from .comparisons import (
    _merge_baseline_factor_comparison_metrics,
    _merge_param_sensitivity_metrics,
)
from .core import (
    SingleFactorEvaluationResult,
    evaluate_single_factor_case,
    run_factor_experiment,
)

# Internal helpers re-exported because tests import them directly.
from .coverage import (
    _annotate_coverage_warmup,
    _count_coverage_break_days,
    _coverage_decision_frame,
    _summarise_effective_coverage,
    _with_split_phase,
)
from .pnl_attribution import (
    _merge_daily_pnl_attribution_metrics,
    _merge_signal_lag_sensitivity_metrics,
)

__all__ = [
    "SingleFactorEvaluationResult",
    "evaluate_single_factor_case",
    "run_factor_experiment",
    "_annotate_coverage_warmup",
    "_coverage_decision_frame",
    "_count_coverage_break_days",
    "_summarise_effective_coverage",
    "_with_split_phase",
    "_merge_daily_pnl_attribution_metrics",
    "_merge_param_sensitivity_metrics",
    "_merge_baseline_factor_comparison_metrics",
    "_merge_signal_lag_sensitivity_metrics",
    "_evaluate_variant_lightweight",
]
