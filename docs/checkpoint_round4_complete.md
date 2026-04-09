# Checkpoint: round4-complete

**Date:** 2026-04-09

## Summary

Round 4 of the alpha-lab commit recovery is complete and verified.  The round
added the volatility factor (amplitude + downside_volatility), updated
`factors/__init__` and `factors/momentum` (new `skip_recent` parameter), added
`factor_recipe.py` for factor composition and dispatch, backfilled Round 3
reporting and campaign test coverage (replacing one obsolete test file), and
committed the config and entrypoint layer (`config.py`, `cli.py`,
`walk_forward_cli.py`, `research_package.py`) together with new tests for
config path resolution and unified CLI routing.  No architectural redesign was
performed.  Three CLI tests that require `experimental_level3` are
intentionally skipped with an explicit reason; `experimental_level3` remains
untracked and outside Round 4 scope.

Verification result: **complete with three intentional skips.**

## Reference

Full details in: `docs/round4_summary.md`

## Git tagging

No automated tagging workflow is defined in this repository.  If a checkpoint
tag is desired, create it manually:

```bash
git tag -a round4-complete -m "Round 4 factor/recipe line, test backfill, and entrypoint layer complete"
```
