"""Real-case single-factor research-validation package (v1)."""

from .pipeline import SingleFactorCaseRunResult, run_single_factor_case
from .spec import (
    NeutralizationSpec,
    OutputSpec,
    PreprocessSpec,
    SingleFactorCaseSpec,
    TargetSpec,
    TransactionCostSpec,
    UniverseSpec,
    load_single_factor_case_spec,
    single_factor_case_spec_from_mapping,
)

__all__ = [
    "NeutralizationSpec",
    "OutputSpec",
    "PreprocessSpec",
    "SingleFactorCaseRunResult",
    "SingleFactorCaseSpec",
    "TargetSpec",
    "TransactionCostSpec",
    "UniverseSpec",
    "load_single_factor_case_spec",
    "run_single_factor_case",
    "single_factor_case_spec_from_mapping",
]
