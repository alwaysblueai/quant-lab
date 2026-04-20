"""Fast-screen: Tier-1 (10 metrics + 4 charts + verdict) and Tier-2 (on-demand deep dive).

See CLAUDE.md / design notes. Tier-1 MUST NOT import Tier-2 heavy modules.
"""

from .artifacts import (
    FastScreenArtifactPaths,
    load_tier1_result,
    load_tier2_index,
    save_tier1_result,
    save_tier2_module,
    tier1_dir,
    tier2_module_dir,
)
from .contracts import (
    CORE_CHART_KEYS,
    CORE_METRIC_KEYS,
    ChartSeries,
    FastScreenResult,
    MetricCard,
    MetricStatus,
    Tier2ModuleStatus,
    Verdict,
)
from .gating import evaluate_gates
from .tier1 import Tier1Inputs, run_tier1
from .tier2 import TIER2_MODULES, run_tier2_modules

__all__ = [
    "CORE_CHART_KEYS",
    "CORE_METRIC_KEYS",
    "ChartSeries",
    "FastScreenArtifactPaths",
    "FastScreenResult",
    "MetricCard",
    "MetricStatus",
    "Tier1Inputs",
    "TIER2_MODULES",
    "Tier2ModuleStatus",
    "Verdict",
    "evaluate_gates",
    "load_tier1_result",
    "load_tier2_index",
    "run_tier1",
    "run_tier2_modules",
    "save_tier1_result",
    "save_tier2_module",
    "tier1_dir",
    "tier2_module_dir",
]
