# Checkpoint: post-control-surface-reduction-round1

**Date:** 2026-04-08

## Summary

Round 1 of the alpha-lab control-surface reduction plan is complete and verified. The
round removed duplicated resolution paths (vault path, PIT gate), converted silent
failure modes into explicit errors, enforced write boundaries in code rather than only
in documentation, consolidated documentation authority to single-source files, demoted
the experimental handoff surface off the default import path, migrated utility-layer
exception raises to the AlphaLabError hierarchy, and wired research integrity checks
(PIT, leakage, rolling stability) into experiment.py and walk_forward.py. No
architectural redesign was performed; all changes targeted enforcement of intent that
was already present in the system's design. New test files cover the research_integrity
package and experiment/walk-forward integrity integration. 1 pre-existing test failure
is unchanged and was not introduced by this round.

## Reference

Full details in: `docs/system_reduction_round1.md`

## Git tagging

No automated tagging workflow is defined in this repository. If a checkpoint tag is
desired, create it manually:

```bash
git tag -a post-control-surface-reduction-round1 -m "Round 1 control-surface reduction complete"
```

Do not ask an assistant to execute this unless the repository has a documented
assistant-tagging workflow.
