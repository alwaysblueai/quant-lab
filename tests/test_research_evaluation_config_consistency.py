"""Cross-check that ``ResearchEvaluationConfig`` keeps its three views aligned.

Every profile section is described in three places:

1. A frozen dataclass (``FactorVerdictConfig``, ``UncertaintyConfig``, ...).
2. A ``TypedDict`` snapshot type (``FactorVerdictSnapshot``, ...).
3. The dictionary literal produced inside
   ``ResearchEvaluationConfig.to_audit_snapshot()``.

When new thresholds are added, all three must stay in sync. These tests pin
the invariants:

* Every key declared by a snapshot ``TypedDict`` is present in the produced
  snapshot dict (no missing keys at runtime).
* Every key present in the produced snapshot dict is declared by the
  ``TypedDict`` (no stray runtime keys / stale declarations).
* Every snapshot key resolves to a real attribute on the corresponding
  dataclass (so ``to_audit_snapshot`` cannot drift to a typo'd attr name).
* The top-level ``ResearchEvaluationAuditSnapshot`` covers the same section
  set as ``ResearchEvaluationConfig``.

This is OPT-P2-5 (Phase 2, batch 2): a pure assertion test — no production
code changes. If any check fails, ``to_audit_snapshot()`` or the snapshot
TypedDict has drifted from the dataclass and must be reconciled manually.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any, get_type_hints

import pytest

from alpha_lab.research_evaluation_config import (
    AVAILABLE_RESEARCH_EVALUATION_PROFILES,
    CampaignTriageSnapshot,
    FactorVerdictSnapshot,
    Level2PortfolioValidationSnapshot,
    Level2PromotionSnapshot,
    ModelFactorOverridesSnapshot,
    NeutralizationComparisonSnapshot,
    ResearchEvaluationAuditSnapshot,
    ResearchEvaluationConfig,
    RollingStabilitySnapshot,
    SingleFactorDiagnosticsSnapshot,
    UncertaintySnapshot,
    get_research_evaluation_config,
)

# Map snapshot section name (top-level snapshot key) -> (
#     TypedDict describing the section,
#     attribute on ResearchEvaluationConfig holding the matching dataclass,
# )
_SECTIONS: dict[str, tuple[type, str]] = {
    "factor_verdict": (FactorVerdictSnapshot, "factor_verdict"),
    "uncertainty": (UncertaintySnapshot, "uncertainty"),
    "rolling_stability": (RollingStabilitySnapshot, "rolling_stability"),
    "neutralization_comparison": (
        NeutralizationComparisonSnapshot,
        "neutralization_comparison",
    ),
    "campaign_triage": (CampaignTriageSnapshot, "campaign_triage"),
    "level2_promotion": (Level2PromotionSnapshot, "level2_promotion"),
    "level2_portfolio_validation": (
        Level2PortfolioValidationSnapshot,
        "level2_portfolio_validation",
    ),
    "single_factor_diagnostics": (
        SingleFactorDiagnosticsSnapshot,
        "single_factor_diagnostics",
    ),
    "model_factor_overrides": (
        ModelFactorOverridesSnapshot,
        "model_factor_overrides",
    ),
}


def _typeddict_keys(td: type) -> set[str]:
    """Return the set of keys declared by a TypedDict at runtime."""
    # ``get_type_hints`` works on TypedDict subclasses and includes inherited
    # keys. ``__annotations__`` alone would miss base-class keys.
    return set(get_type_hints(td).keys())


def _dataclass_field_names(config_attr: object) -> set[str]:
    assert is_dataclass(config_attr), f"expected dataclass instance, got {type(config_attr)!r}"
    return {f.name for f in fields(config_attr)}


@pytest.mark.parametrize("profile_name", AVAILABLE_RESEARCH_EVALUATION_PROFILES)
def test_audit_snapshot_top_level_keys_match_typed_dict(profile_name: str) -> None:
    config = get_research_evaluation_config(profile_name)
    snapshot = config.to_audit_snapshot()

    runtime_keys = set(snapshot.keys())
    declared_keys = _typeddict_keys(ResearchEvaluationAuditSnapshot)

    assert runtime_keys == declared_keys, (
        f"top-level snapshot keys diverged for profile {profile_name!r}:\n"
        f"  only in runtime:  {sorted(runtime_keys - declared_keys)}\n"
        f"  only in TypedDict: {sorted(declared_keys - runtime_keys)}"
    )


@pytest.mark.parametrize("profile_name", AVAILABLE_RESEARCH_EVALUATION_PROFILES)
@pytest.mark.parametrize("section_name", list(_SECTIONS.keys()))
def test_audit_snapshot_section_keys_match_typed_dict(
    profile_name: str, section_name: str
) -> None:
    config = get_research_evaluation_config(profile_name)
    snapshot = config.to_audit_snapshot()
    snapshot_typeddict, _ = _SECTIONS[section_name]

    section_payload: Any = snapshot[section_name]  # type: ignore[literal-required]
    assert isinstance(section_payload, dict), (
        f"{section_name} should map to a dict, got {type(section_payload).__name__}"
    )

    runtime_keys = set(section_payload.keys())
    declared_keys = _typeddict_keys(snapshot_typeddict)

    assert runtime_keys == declared_keys, (
        f"snapshot section {section_name!r} keys diverged for profile {profile_name!r}:\n"
        f"  only in runtime:  {sorted(runtime_keys - declared_keys)}\n"
        f"  only in TypedDict: {sorted(declared_keys - runtime_keys)}"
    )


@pytest.mark.parametrize("section_name", list(_SECTIONS.keys()))
def test_snapshot_section_keys_resolve_to_real_dataclass_fields(section_name: str) -> None:
    config = ResearchEvaluationConfig()  # defaults — any profile would share the layout
    snapshot = config.to_audit_snapshot()
    snapshot_typeddict, config_attr_name = _SECTIONS[section_name]

    section_payload: Any = snapshot[section_name]  # type: ignore[literal-required]
    runtime_keys = set(section_payload.keys())

    dataclass_attr = getattr(config, config_attr_name)
    dataclass_field_names = _dataclass_field_names(dataclass_attr)

    unknown_keys = runtime_keys - dataclass_field_names
    assert not unknown_keys, (
        f"snapshot section {section_name!r} contains keys that do not exist on "
        f"the underlying dataclass {type(dataclass_attr).__name__}: {sorted(unknown_keys)}"
    )

    # Sanity: the TypedDict's declared keys should also be a subset of the
    # dataclass fields. (TypedDict is the projection contract; it must not
    # promise keys the dataclass cannot supply.)
    declared_keys = _typeddict_keys(snapshot_typeddict)
    unknown_declared = declared_keys - dataclass_field_names
    assert not unknown_declared, (
        f"TypedDict {snapshot_typeddict.__name__} declares keys not present on "
        f"{type(dataclass_attr).__name__}: {sorted(unknown_declared)}"
    )


def test_top_level_snapshot_covers_all_sections_with_dataclass_backing() -> None:
    """The 9 known sections must each map to a real dataclass attribute."""
    config = ResearchEvaluationConfig()
    for section_name, (_, config_attr_name) in _SECTIONS.items():
        attr = getattr(config, config_attr_name, None)
        assert attr is not None, f"section {section_name!r} has no dataclass attr"
        assert is_dataclass(attr), (
            f"section {section_name!r} attr {config_attr_name!r} is not a dataclass"
        )
