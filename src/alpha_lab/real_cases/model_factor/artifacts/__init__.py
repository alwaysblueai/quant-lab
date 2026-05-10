"""Model-factor artifacts package.

Public API is re-exported here so existing callers using
``from alpha_lab.real_cases.model_factor.artifacts import ...`` keep working.
The implementation is split across topic modules for maintainability.
"""

from ._utils import ModelFactorArtifactPaths
from .core import export_artifact_bundle
from .diagnostics import write_diagnostics_artifact

__all__ = [
    "export_artifact_bundle",
    "write_diagnostics_artifact",
    "ModelFactorArtifactPaths",
]
