# System Simplification — Round 1 Completed Checkpoint

## 1. Overview

### What Round 1 was trying to achieve

Alpha-lab accumulated a set of duplicated resolution paths, silent failure modes, and
redundant control surfaces across its three layers (quant-knowledge, alpha-lab, research
bridge). Round 1 targeted these specific accumulations without restructuring the system.

The goal was not architectural redesign. It was removal of redundancy, clarification of
authority, and hardening of boundaries that were already implied by the system's intent
but not enforced in code or documentation.

### Why the system did NOT need a redesign

The core architecture — factor discovery (Level 1), portfolio validation (Level 2),
experimental replay layer (Level 3), vault export as the sole authorized write path —
was already correct. The problem was drift between the documented intent and the actual
enforcement. Round 1 enforced what was already intended.

### What kinds of changes were intentionally prioritized

- Removing duplicated code paths rather than consolidating into new abstractions
- Converting silent failure modes into explicit errors
- Enforcing write boundaries that were documented but not guarded in code
- Making authority single-source (one resolution function, one read sequence)
- Demoting experimental surface off the default import path


## 2. Completed Changes

### Duplication removal

| Item | Files affected | Change |
|---|---|---|
| `validate_canonical_table()` | `experiment.py` and all callsites | Fully removed; no replacement abstraction added |
| Vault path resolution | `vault_export.py`, `reporting/__init__.py` | All callers now go through `resolve_vault_root()`; `config.OBSIDIAN_VAULT_PATH` constant removed |
| Shared PIT gate | `research_integrity/asof.py`, `experiment.py`, `walk_forward.py` | `pit_check()` extracted and used at both targeted callsites; redundant `raise_on_hard_failures` call removed from `walk_forward.py` |

### Boundary hardening

| Item | Files affected | Change |
|---|---|---|
| Vault write boundary | `obsidian.py`, `reporting/__init__.py` | `write_obsidian_note()` gained `restricted_root` parameter; both vault-targeting callers pass `restricted_root=vault/50_experiments` |
| Lifecycle evidence gate | `obsidian.py` | Cards with `lifecycle: validated-backtest` or higher must contain `[[50_experiments/...]]` wikilink or `LifecyclePromotionError` is raised |
| Vault write exception types | `exceptions.py` | `VaultWriteError` and `LifecyclePromotionError` added |

### Explicit failure semantics

| Item | Files affected | Change |
|---|---|---|
| Missing vault path | `vault_export.py`, `test_vault_export.py` | Unset `OBSIDIAN_VAULT_PATH` with non-skip mode now returns `success=False, status="failed"`; previously returned silent `success=True, status="skipped"` |
| Suppressed L2 validation | `reporting/level2_portfolio_validation.py` | `run_for_non_promoted_cases=False` now emits `warnings.warn()` instead of silently skipping |

### Documentation authority consolidation

| Item | Files affected | Change |
|---|---|---|
| Startup read sequence | `CLAUDE-QUICKREF.md`, `CLAUDE.md` | Single-source in `CLAUDE-QUICKREF.md §1`; `CLAUDE.md` defers to it |
| One-line intake shortcut | `CLAUDE-QUICKREF.md §3` | Merged inline; `Protocol - One-Line Source Intake Entry.md` deleted |
| Promotion rules | `00_protocols/Protocol - Card Schema.md` | Lifecycle progression and evidence requirement stated once; duplicate removed |
| Stale docstring | `reporting/__init__.py` | Removed reference to removed `config.OBSIDIAN_VAULT_PATH`; replaced with accurate description of `resolve_vault_root()` and `OBSIDIAN_VAULT_PATH` env var |

### Experimental surface isolation

| Item | Files affected | Change |
|---|---|---|
| `handoff.py` demotion | `handoff.py`, `experimental_level3/handoff.py`, `experimental_level3/__init__.py` | Canonical location moved to `experimental_level3/handoff.py`; old path replaced with `__getattr__`-based lazy shim that emits `DeprecationWarning` on attribute access |

### PIT consistency improvements

| Item | Files affected | Change |
|---|---|---|
| `pit_check()` extraction | `research_integrity/asof.py` | Shared function wraps `check_no_future_dates_in_input` + immediate `raise_on_hard_failures`; single-point hard-abort semantics |
| `experiment.py` PIT callers | `experiment.py` | 6 calls to `check_no_future_dates_in_input` replaced with `pit_check()` |
| `walk_forward.py` PIT callers | `walk_forward.py` | 2 calls replaced with `pit_check()`; now-redundant `raise_on_hard_failures` call and its import removed |
| Novelty sidecar freshness | `generate_inbox_novelty_sidecar.py` | Sidecar hook now calls `_rebuild_card_index()` before reading CARD-INDEX.tsv; self-contained regardless of hook ordering |
| Experiment provenance | `experiment.py` | `ExperimentProvenance` gained `git_dirty: bool | None` field |


## 3. Verified Outcomes

Post-implementation verification confirmed:

- `validate_canonical_table()`: zero grep matches across `src/`
- `config.OBSIDIAN_VAULT_PATH`: removed from `config.py`; all vault path resolution goes through `resolve_vault_root()`
- All `write_obsidian_note()` callers audited: two vault-targeting callers pass `restricted_root`; two local-path callers correctly pass `None`
- Lifecycle backlink gate fires on `restricted_root`-gated writes for any lifecycle value outside `{"", "draft", "active", "theoretical"}`
- `pit_check()` confirmed at all six `experiment.py` sites and both `walk_forward.py` sites
- Handoff shim confirmed: `DeprecationWarning` emitted on attribute access; canonical path clean

**Test state:**
- 17 tests pass
- 1 pre-existing test failure remains unchanged: `test_obsidian_markdown_has_required_sections` fails because section headers in `to_obsidian_markdown()` are Chinese (`## 解释`, `## 下一步`) while the test expects English. This failure existed before Round 1 and is not caused by any Round 1 change.


## 4. Non-Negotiable Invariants

The following architectural and research-integrity invariants were preserved without
exception throughout Round 1:

- **PIT hard gate**: `pit_check()` raises immediately on any future-date violation; no soft-pass path exists at the targeted callsites
- **Splits precede labels**: temporal ordering of training/test splits relative to label computation was not touched
- **`export_to_vault()` is the sole authorized vault write path**: direct writes to the vault outside this function are blocked by `restricted_root`; the gate raises `VaultWriteError` on boundary violation
- **Completed experiments must export or fail visibly**: missing vault path is now a hard failure, not a silent skip
- **Lifecycle promotion requires evidence backlink**: `validated-backtest` or higher lifecycle requires a `[[50_experiments/...]]` wikilink or promotion is blocked at write time
- **Preflight blocks before compute**: no changes were made to preflight validation ordering


## 5. What Was Deliberately Deferred

### Out-of-scope PIT callers

The following files call `check_no_future_dates_in_input` directly (not through
`pit_check()`) using a deferred-raise pattern. They were identified during the Round 1
audit and explicitly left out of scope:

- `real_cases/single_factor/pipeline.py` — 3 direct calls
- `real_cases/composite/pipeline.py` — 3 direct calls
- `real_cases/model_factor/pipeline.py` — 3+ direct calls
- `model_factor/core.py` — 2 direct calls

These callers use `_record_integrity()` plus a deferred `raise_on_hard_failures` call
at the end of their pipeline. This is a valid pattern but differs from the immediate-
abort semantics of `pit_check()`. Consolidating them was deferred because it would
require auditing pipeline-level raise semantics — a larger change than Round 1's
removal-focused scope.

### Other deliberately excluded changes

- No changes to `real_cases/*/pipeline.py` behavior
- No changes to the `model_factor` pipeline
- No changes to `walk_forward.py` fold logic beyond removing the now-redundant call
- No changes to the pre-existing failing test (`test_obsidian_markdown_has_required_sections`)
- No refactoring of the `reporting/` render layer
- No vault card modifications beyond the auto-gen marker addition to the Registry board


## 6. Recommended Next Wave

**Round 2 scope: narrow PIT consolidation only.**

Round 2 should consolidate the four deferred callsite groups listed in Section 5 into
`pit_check()` (or an equivalent shared gate compatible with the deferred-raise pattern).

Round 2 should NOT:
- broaden into another architecture sweep
- touch the vault layer
- modify the experimental Level 3 surface
- change any reporting or dashboard code
- introduce new abstractions beyond what the PIT consolidation strictly requires

The deferred callers share a pattern: they collect `IntegrityCheckResult` objects and
raise at the end of a function. The Round 2 task is to decide whether `pit_check()`
(immediate-raise) or a separate `pit_check_collect()` (collect-only, no raise) is the
right shared gate for that pattern, then apply it uniformly to the four files.


## 7. Final Status

Round 1 is complete and verified.

The system is cleaner, boundaries are enforced, authority is single-source, and the
experimental surface is isolated. No behavior was changed that was not directly required
by the removals and hardenings listed above.

This document serves as the checkpoint record for Round 1. Round 2 has not been started.
