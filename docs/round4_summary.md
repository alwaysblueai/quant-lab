# Round 4 Summary: Factor/Recipe Line, Test Backfill, and Entrypoint Layer

**Branch:** round1-recovery
**Commits:** 0c81975 → d034403 (3 commits)
**Verification result:** Complete with one intentional skip (see R4-3 below)

---

## What Round 4 committed

### R4-1 — Volatility factor line and factor recipe (0c81975)

| File | Type |
|---|---|
| `src/alpha_lab/factors/volatility.py` | new — amplitude + downside_volatility factors |
| `src/alpha_lab/factors/__init__.py` | updated — exports volatility |
| `src/alpha_lab/factors/low_volatility.py` | updated — exception migration to AlphaLabError hierarchy |
| `src/alpha_lab/factors/momentum.py` | updated — new `skip_recent` parameter |
| `src/alpha_lab/factor_recipe.py` | new — factor recipe composition and dispatch |
| `tests/test_momentum.py` | updated — covers `skip_recent` behavior |
| `tests/test_volatility.py` | new — amplitude + downside_volatility unit tests |

### R4-2 — Round 3 test backfill (001831a)

| File | Type |
|---|---|
| `tests/test_campaign_report_renderer.py` | updated — expanded coverage |
| `tests/test_campaign_summary_generation.py` | updated — expanded coverage |
| `tests/test_case_report_renderer.py` | updated — expanded coverage |
| `tests/test_reporting.py` | updated — expanded coverage |
| `tests/test_research_campaign_1.py` | updated — expanded coverage |
| `tests/test_real_case_single_factor_package.py` | deleted — superseded by expanded tests above |

### R4-3 — Config and entrypoint layer (d034403)

| File | Type |
|---|---|
| `src/alpha_lab/config.py` | updated — env-var override + DATA_ROOT + integrity check |
| `src/alpha_lab/cli.py` | updated — unified top-level router; adds real-case, campaign, bridge, experimental, profiles, web, data commands |
| `src/alpha_lab/walk_forward_cli.py` | updated — minor alignment with updated cli.py helpers |
| `src/alpha_lab/research_package.py` | updated — replay/execution-impact package assembly |
| `tests/test_config.py` | new — config path resolution + env-var override tests |
| `tests/test_smoke.py` | updated — aligns smoke assertions to current contracts |
| `tests/test_unified_cli.py` | new — routing tests for all committed CLI subcommands |

#### Intentional skips in test_unified_cli.py

Three tests are decorated `@pytest.mark.skip`:

- `test_unified_cli_routes_experimental_single_factor_package`
- `test_unified_cli_routes_experimental_execution_realism_package`
- `test_unified_cli_routes_experimental_factor_health_monitor`

These tests call `monkeypatch.setattr` with string paths into
`alpha_lab.experimental_level3.*`, which triggers actual module imports at
test execution time.  `experimental_level3/` is untracked and explicitly
outside Round 4 scope.  The skip reason on each test reads:

> `alpha_lab.experimental_level3 is not yet committed; remains out of Round 4 scope`

The matching lazy imports in `cli.py` itself are safe: they execute only
inside `if args.top_command == "experimental":` branches and are never
reached during normal test runs.

---

## Intentionally deferred / out of scope

The following areas were not committed in Round 4 and carry no R4 dependency:

| File / Area | Reason |
|---|---|
| `backtest_adapter/` (8 files) | Independent subsystem; no R4 pipeline dependency |
| `pyproject.toml`, `uv.lock` | Dependency manifest; no R4 source file requires the new entries |
| `scripts/run_experiment.py` and siblings (4 files) | Entry-point scripts; no runtime dependency from committed code |
| `docs/` narrative updates (architecture, developer_guide, system_manual, etc.) | Documentation refresh; deferred |
| `AGENTS.md`, `CLAUDE.md`, `README.md` | Top-level docs; deferred |
| `experimental_level3/` | Explicitly excluded; guarded by lazy imports in `cli.py` |
| `configs/real_cases/` (untracked) | Case config files; no code dependency |
| `model_factor/`, `data_adapters/`, `data_store/`, `research_bridge/`, `web_*` | Out of recovery scope |
| Advanced reporting tooling (`workflow_artifact_service`, `research_artifact_manifest`, `campaign_profile_dashboard`, `factor_decomposition`) | No R4 pipeline dependency; lazy-loaded or unreferenced by committed code |
| `campaigns/profile_comparison.py` | Depends on untracked `examples/` package |
