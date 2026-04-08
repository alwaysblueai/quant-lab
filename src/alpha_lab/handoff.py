"""Deprecated: import from ``alpha_lab.experimental_level3.handoff`` instead.

This module is a compatibility shim.  All symbols are re-exported via
``__getattr__`` so the import is deferred until first attribute access,
avoiding the circular-import that a module-level import would cause.
"""

from __future__ import annotations

import importlib
import warnings

_EXPORTED_NAMES: frozenset[str] = frozenset(
    {
        "EXECUTION_ASSUMPTIONS_SCHEMA_VERSION",
        "HANDOFF_SCHEMA_VERSION",
        "HANDOFF_TIMING_SEMANTIC_KEYS",
        "PORTFOLIO_CONSTRUCTION_SCHEMA_VERSION",
        "ExecutionAssumptionsSpec",
        "PortfolioConstructionSpec",
        "summarize_handoff_timing_semantics",
        "validate_handoff_artifact",
    }
)

_DEPRECATION_MSG = (
    "alpha_lab.handoff is deprecated and will be removed in a future release. "
    "Import from alpha_lab.experimental_level3.handoff instead."
)


def __getattr__(name: str) -> object:
    if name in _EXPORTED_NAMES:
        warnings.warn(_DEPRECATION_MSG, DeprecationWarning, stacklevel=2)
        mod = importlib.import_module("alpha_lab.experimental_level3.handoff")
        return getattr(mod, name)
    raise AttributeError(f"module 'alpha_lab.handoff' has no attribute {name!r}")
