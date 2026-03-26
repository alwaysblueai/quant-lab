"""Real-case composite research package (v1)."""

from .pipeline import CompositeCaseRunResult, run_composite_case
from .spec import (
    ComponentSpec,
    CompositeCaseSpec,
    NeutralizationSpec,
    OutputSpec,
    PreprocessSpec,
    TargetSpec,
    TransactionCostSpec,
    UniverseSpec,
    composite_case_spec_from_mapping,
    load_composite_case_spec,
)

__all__ = [
    "ComponentSpec",
    "CompositeCaseRunResult",
    "CompositeCaseSpec",
    "NeutralizationSpec",
    "OutputSpec",
    "PreprocessSpec",
    "TargetSpec",
    "TransactionCostSpec",
    "UniverseSpec",
    "composite_case_spec_from_mapping",
    "load_composite_case_spec",
    "run_composite_case",
]
