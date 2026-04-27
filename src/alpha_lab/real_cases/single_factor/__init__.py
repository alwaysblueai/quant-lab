"""Real-case single-factor research-validation package (v1)."""

from .pipeline import (
    SingleFactorBaseFeatureCache,
    SingleFactorBatchDefinition,
    SingleFactorBatchParallelConfig,
    SingleFactorCaseRunResult,
    SingleFactorInputBundle,
    load_standard_inputs,
    prepare_base_features,
    run_single_factor_batch,
    run_single_factor_case,
    run_single_factor_cases,
)
from .spec import (
    FactorInputSpec,
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
    "FactorInputSpec",
    "NeutralizationSpec",
    "OutputSpec",
    "PreprocessSpec",
    "SingleFactorBatchDefinition",
    "SingleFactorBatchParallelConfig",
    "SingleFactorBaseFeatureCache",
    "SingleFactorCaseRunResult",
    "SingleFactorInputBundle",
    "SingleFactorCaseSpec",
    "TargetSpec",
    "TransactionCostSpec",
    "UniverseSpec",
    "load_standard_inputs",
    "prepare_base_features",
    "load_single_factor_case_spec",
    "run_single_factor_batch",
    "run_single_factor_case",
    "run_single_factor_cases",
    "single_factor_case_spec_from_mapping",
]
