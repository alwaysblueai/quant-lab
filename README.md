# Alpha Lab

Research-grade quant lab for Level 1/2 workflows:

- Level 1: factor discovery
- Level 2: portfolio construction validation

## Scope

Alpha Lab is a research system for Level 1 factor discovery and Level 2
portfolio construction validation, focused on correctness, robustness, and
reproducibility.

`alpha-lab` is explicitly scoped as a research system for correctness,
robustness, and reproducibility. It supports:

- reusable factor code under `src/alpha_lab`
- point-in-time and anti-leakage integrity checks
- factor evaluation diagnostics (IC/RankIC, quantile spread, turnover, subperiod robustness, instability flags)
- research-level portfolio validation (weight mapping, constraints, turnover, simple costs)
- reproducible reporting and experiment export to quant-knowledge

`alpha-lab` is **not**:

- an execution replay engine
- an implementability-grade platform
- an order-fill simulator
- a microstructure-aware execution simulator
- a real execution-semantics validator

## Architecture

The repository is structured around core Level 1/2 layers only.

```
Layer A — Research Integrity Layer
  alpha_lab.research_integrity.*   — PIT/as-of discipline, anti-leakage checks,
                                      cross-timeframe correctness, bar-completeness checks

Layer B — Factor Research Layer
  alpha_lab.factors.*          — factor computation (e.g. momentum)
  alpha_lab.labels             — forward-return label generation
  alpha_lab.evaluation         — IC / Rank-IC computation
  alpha_lab.quantile           — quantile bucket returns and long-short
  alpha_lab.turnover           — quantile / long-short turnover
  alpha_lab.preprocess         — winsorize, z-score

Layer C — Portfolio Research Layer
  alpha_lab.strategy.StrategySpec   — explicit portfolio construction spec:
                                      long_top_k, short_bottom_k,
                                      weighting_method, holding_period,
                                      rebalance_frequency
                                      (n_quantiles is a factor-evaluation
                                       param, not part of StrategySpec)

  alpha_lab.portfolio_research — portfolio_weights, simulate_portfolio_returns,
                                 portfolio_turnover,
                                 portfolio_cost_adjusted_returns

Layer D — Reporting / Registry / Knowledge Export
  alpha_lab.reporting          — summaries, markdown, experiment-card export
  alpha_lab.reporting.research_validation_package
                               — reproducible Level 1/2 research package export
  alpha_lab.registry           — append-only research registry
  alpha_lab.comparison         — legacy side-by-side research comparison helper

Orchestration (Level 1/2)
  alpha_lab.experiment         — run_factor_experiment (single split)
  alpha_lab.walk_forward       — run_walk_forward_experiment (rolling OOS)
```

`StrategySpec` is the explicit boundary between the factor research layer and
the portfolio research layer.  It answers: which assets to include in each leg,
how to weight them, and how often to rebalance.  Passing a `StrategySpec` to
`run_factor_experiment` or `run_walk_forward_experiment` makes all portfolio
construction intent visible in one place rather than spread across call sites.

**Core boundary**: portfolio outputs are research-level approximations for Level
2 validation. They are not fill-level execution simulations.

## Setup

Requirements:

- Python 3.12
- `uv`

Install the environment:

```bash
uv sync --all-extras
```

Large vendor-backed datasets should live outside the repository. By default the
external data root resolves to `~/.local/share/alpha-lab/data`; override it
when needed:

```bash
export ALPHA_LAB_DATA_ROOT=/path/to/alpha-lab-data
```

Run the local checks:

```bash
make check
```

If `uv` cache permissions are restricted in WSL or a sandboxed environment, set a
writable cache directory:

```bash
UV_CACHE_DIR=/tmp/uv-cache make check
```

## Project Structure

- `src/alpha_lab`: reusable research code
- `tests`: unit and regression tests
- `docs`: repository conventions and lightweight documentation
- `data/raw`: raw immutable datasets
- `data/processed`: derived intermediate datasets
- `scripts`: one-off scripts
- `notebooks`: exploratory work

## Workflow

1. Add or modify reusable logic under `src/alpha_lab`.
2. Add tests for every reusable function and every known leakage or alignment risk.
3. Use `alpha-lab data ...` to ingest/update vendor data and export small case
   slices as `prices.csv` / `universe.csv` inputs, with factor CSVs exported only
   when explicitly requested.
4. Use `alpha-lab bridge ...` to manage one factor-project pack under
   `quant-knowledge/55_projects/`, generate one round context bundle for
   ChatGPT Projects, scaffold `alpha-lab` case drafts, and stage reviewed
   writeback drafts before applying them to `50_experiments/`.
5. Run:

```bash
make lint
make typecheck
make test
```

Or:

```bash
make check
```

For read-only inspection of the external canonical store, use:

```bash
alpha-lab data query --sql "select count(*) as n_rows from daily_bars"
```

If you already have a local nested ZIP of per-asset A-share daily CSV files,
you can ingest it directly into the canonical store. The importer writes
`daily_bars` / `adj_factor` / `asset_status` as Parquet datasets partitioned
by `asset=<ts_code>` so the warehouse stays efficient for symbol-scoped
inspection while preserving the existing Level 1/2 slice/export flow:

```bash
alpha-lab data ingest local-zip ashare-daily \
  --zip-path "/path/to/股票日线.zip"
```

For daily price-volume research, export raw or前复权 price slices directly:

```bash
alpha-lab data export-case-inputs \
  --output-dir data/processed/real_case_inputs/ashare_qfq \
  --slice-preset standard
```

Built-in slice presets for A-share daily price-volume research:

- `pilot`: recent 3 years, `top_liquid_300`, `qfq`
- `standard`: recent 5 years, `listed_90d`, `qfq`
- `robust`: recent 8 years, `listed_90d`, `qfq`
- `institutional`: recent 8 years, `institutional_ashare`, `qfq`

Explicit `--start-date` / `--end-date` / `--universe` / `--adjustment` still override the preset.

Supported liquidity universes now include:

- `institutional_ashare`
- `top_liquid_300`
- `top_liquid_500`
- `top_liquid_800`

`institutional_ashare` keeps A-share rows that are listed for at least 180 days,
are not ST or suspended on the date, and pass a trailing 20-sample liquidity
screen: average `amount` at least `20000` (Tushare daily `amount` is thousand
CNY, so roughly RMB 20 million) and outside the bottom 20% by date. The
`top_liquid_*` universes are ranked by trailing 60-trading-day average `amount`,
not same-day amount.

The legacy Web UI auto-data-source flow also defaults to `standard`, and exposes
`pilot / standard / robust / institutional` directly in the form. New interactive
research workflows should use `alpha-lab web unified`.

For A-share daily price-volume research, exported `prices.csv` now includes the
standard research columns needed by most daily factor recipes:

- `open`, `high`, `low`, `close`, `pre_close`
- `volume`, `amount`, `vwap`, `turnover_rate`
- `up_limit`, `down_limit`, `is_limit_up`, `is_limit_down`
- `is_suspended`, `is_st`
- `is_hs300`, `is_zz500`, `is_zz1000`, `is_sz50`

When the corresponding canonical tables are available, `export-case-inputs`
also writes:

- `asset_status.csv`
- `index_membership.csv`

Coverage rules for the external A-share daily store:

- minimum requirement: latest available date must have at least recent 3 years of history
- robust target: latest available date should have recent 8 years of history
- `export-case-inputs` rejects windows outside the available `daily_bars` coverage, while treating non-trading start/end dates via the open-day calendar boundary

Long Tushare ingest/update runs are chunked automatically by default:

- default ingest/update chunk size: `6` months
- progress is printed per chunk and per ingest stage
- pass `--chunk-months 0` to disable chunking for a single long request

For daily price-volume research only, prefer:

```bash
alpha-lab data ingest tushare core \
  --start-date 2023-04-01 \
  --end-date 2026-04-01 \
  --asset-limit 1000 \
  --daily-research-only
```

This skips slow ROE financial-indicator fetches while retaining the daily
research tables needed for OHLCV, turnover, limit prices, suspension/ST, and
index-membership flags.

## Research Bridge

`alpha-lab bridge` is the opt-in project layer for the hybrid workflow:

- `quant-knowledge`: long-term knowledge and project state
- `alpha-lab`: Level 1/2 experiment execution
- `ChatGPT Projects`: discussion, synthesis, and lightweight web search

The bridge keeps the formal memory in your own vault rather than in platform
memory alone. First create one project:

```bash
alpha-lab bridge init-project \
  --slug momentum-factor \
  --title-zh 动量因子项目 \
  --category factor_family \
  --owner yukun \
  --market ashare \
  --frequency daily \
  --chatgpt-project-name "Momentum Factor"
```

This creates `quant-knowledge/55_projects/momentum-factor/` with:

- `project.yaml`
- `01_project_brief.md`
- `02_project_rules.md`
- `03_card_map.md`
- `10_active_state.md`
- `20_decision_log.md`
- `30_rounds/`
- `40_specs/`
- `50_writeback_drafts/`

Then use the round workflow:

```bash
alpha-lab bridge start-round \
  --project momentum-factor \
  --topic "三个月成交额加权动量"

alpha-lab bridge scaffold-case \
  --project momentum-factor \
  --round <round_id> \
  --case-name mom_amt_60

alpha-lab bridge summarize-run \
  --project momentum-factor \
  --round <round_id> \
  --run-root dist/bridge_runs/momentum-factor/mom_amt_60/<run_dir>
```

`summarize-run` writes:

- `30_rounds/<round_id>/latest_experiment_feedback.md`
- `50_writeback_drafts/*__writeback_draft.md`
- `50_writeback_drafts/*__state_update_patch.md`

Only after manual review should you edit the draft frontmatter to
`review_status: approved` and apply it:

```bash
alpha-lab bridge apply-writeback \
  --project momentum-factor \
  --draft /path/to/*__writeback_draft.md
```

That copies experiment artifacts into `50_experiments/`, updates
`10_active_state.md`, and appends a project-level `20_decision_log.md` entry.

## Canonical Factor Output Schema

All reusable factors must return long-form output with exactly these columns:

- `date`: observation timestamp for the factor value
- `asset`: asset identifier
- `factor`: factor name
- `value`: numeric factor value

Rules:

- one row per `(date, asset, factor)`
- features at `date=t` may only use information available at or before `t`
- labels and forward returns belong in separate tables
- merges must be explicit on `("date", "asset")`, and include `factor` when stacking factors

Example:

| date | asset | factor | value |
|------|-------|--------|-------|
| 2024-01-02 | AAPL | momentum_20d | 0.031 |
| 2024-01-02 | MSFT | momentum_20d | -0.008 |

## Documentation

Full index: [docs/README.md](docs/README.md). Quick links:

- [docs/architecture.md](docs/architecture.md) — layer contracts, data flow, path/config
- [docs/system_manual.md](docs/system_manual.md) — API reference and usage patterns
- [docs/developer_guide.md](docs/developer_guide.md) — how to extend the codebase
- [docs/data_conventions.md](docs/data_conventions.md) — canonical timestamp, merge, and storage rules
- [docs/module_classification.md](docs/module_classification.md) — core module map

## Current Reusable Components

- `alpha_lab.strategy.StrategySpec`
- `alpha_lab.strategy.portfolio_weights_from_strategy`
- `alpha_lab.factors.momentum.momentum`
- `alpha_lab.labels.forward_return`
- `alpha_lab.evaluation.compute_ic`
- `alpha_lab.evaluation.compute_rank_ic`
- `alpha_lab.quantile.quantile_returns`
- `alpha_lab.quantile.long_short_return`
- `alpha_lab.splits.time_split`
- `alpha_lab.splits.walk_forward_split`
- `alpha_lab.experiment.run_factor_experiment`
- `alpha_lab.reporting.summarise_experiment_result`
- `alpha_lab.reporting.export_summary_csv`
- `alpha_lab.reporting.to_obsidian_markdown`
- `alpha_lab.quantile.quantile_assignments`
- `alpha_lab.turnover.quantile_turnover`
- `alpha_lab.turnover.long_short_turnover`
- `alpha_lab.costs.apply_linear_cost`
- `alpha_lab.costs.cost_adjusted_long_short`
- `alpha_lab.preprocess.winsorize_series`
- `alpha_lab.preprocess.zscore_series`
- `alpha_lab.interfaces.validate_factor_output`
- `alpha_lab.comparison.compare_experiments` (legacy helper)
- `alpha_lab.comparison.rank_experiments` (legacy helper)
- `alpha_lab.registry.register_experiment`
- `alpha_lab.registry.load_registry`
- `alpha_lab.registry.append_to_registry`

## Strategy Construction Intent

Use `StrategySpec` to make portfolio construction intent explicit before
passing it to the experiment runner:

```python
from alpha_lab.strategy import StrategySpec

# Long-only: top 10 assets, rank-weighted, rebalance every date, hold 1 period
spec = StrategySpec(
    long_top_k=10,
    weighting_method="rank",
    holding_period=1,
    rebalance_frequency=1,
)

# Long-short: top 5 long / bottom 5 short, equal-weighted
ls_spec = StrategySpec(
    long_top_k=5,
    short_bottom_k=5,
    weighting_method="equal",
    holding_period=2,
    rebalance_frequency=1,
)

result = run_factor_experiment(
    prices,
    lambda p: momentum(p, window=20),
    horizon=5,
    strategy=spec,
    portfolio_cost_rate=0.001,
)
print(result.portfolio_summary)
```

When `strategy` is provided it overrides `holding_period`, `rebalance_frequency`,
and `weighting_method` (a `UserWarning` is raised if those are also passed
explicitly).  `n_quantiles` governs the factor-evaluation path (IC, quantile
returns) and is **not** part of `StrategySpec` — pass it directly to
`run_factor_experiment`.  `portfolio_cost_rate` is intentionally not part of
`StrategySpec` — it is a cost assumption, not a construction decision.

## Running an Experiment

`run_factor_experiment` connects all evaluation modules into a single call:

```python
import pandas as pd
from alpha_lab.experiment import run_factor_experiment
from alpha_lab.factors.momentum import momentum

prices: pd.DataFrame  # long-form [date, asset, close]

result = run_factor_experiment(
    prices,
    lambda p: momentum(p, window=20),
    horizon=5,          # forward-return look-ahead in per-asset rows
    n_quantiles=5,
    train_end="2022-12-31",
    test_start="2023-01-01",
)

print(result.summary)
# ExperimentSummary(mean_ic=..., mean_rank_ic=..., ic_ir=...,
#                   ic_positive_rate=..., long_short_ir=...,
#                   subperiod_ic_positive_share=...,
#                   eval_coverage_ratio_mean=..., instability_flags=(...))
```

`result.factor_df` and `result.label_df` always cover the full sample.
`result.ic_df`, `result.rank_ic_df`, `result.quantile_returns_df`, and
`result.long_short_df` are restricted to the evaluation period.

The split is date-based: every row sharing a test-period date enters evaluation,
while train-period rows are excluded. Labels at test date `t` still use strictly
future prices (`close[t+horizon]/close[t]-1`) — that is by construction, not
lookahead, because the label value is stored at `t` for alignment with factor
values observed at `t`.

## Reporting

Turn any `ExperimentResult` into a summary record, CSV, or Obsidian note:

```python
from alpha_lab.reporting import (
    export_summary_csv,
    summarise_experiment_result,
    to_obsidian_markdown,
)

# One-row summary DataFrame (stackable across experiments)
# n_quantiles, train_end, and test_start are carried on result automatically
summary = summarise_experiment_result(result)

# Export to CSV (parent directories created automatically)
export_summary_csv(summary, "output/reports/momentum_5d.csv")

# Obsidian-friendly markdown note
md = to_obsidian_markdown(result, title="Momentum 5d — OOS", notes="Needs decay analysis.")
```

Experiment card export to quant-knowledge:

```python
import os
from alpha_lab.reporting import export_experiment_card

# Option A: set environment variable once
os.environ["OBSIDIAN_VAULT_PATH"] = "/path/to/quant-knowledge"
path = export_experiment_card(result, name="momentum-5d-Ashare")

# Option B: pass vault_path explicitly
path = export_experiment_card(
    result,
    name="momentum-5d-Ashare",
    vault_path="/path/to/quant-knowledge",
)
```

If neither `vault_path` nor `OBSIDIAN_VAULT_PATH` is provided, export raises
`ValueError`.

## Turnover and Cost Estimation

`ExperimentResult` now includes portfolio turnover outputs computed alongside
the IC and quantile-return metrics:

```python
# Turnover is already computed inside run_factor_experiment
result.quantile_turnover_df      # (date, factor, quantile, turnover)
result.long_short_turnover_df    # (date, factor, long_short_turnover)
result.summary.mean_long_short_turnover

# Apply a cost rate manually
from alpha_lab.costs import cost_adjusted_long_short
adj = cost_adjusted_long_short(
    result.long_short_df,
    result.long_short_turnover_df,
    cost_rate=0.001,  # 10 bps one-way
)

# Or include in the summary / markdown report
summary = summarise_experiment_result(result, cost_rate=0.001)
md = to_obsidian_markdown(result, cost_rate=0.001)
```

**Important:** This is a minimal research friction estimate only.  Turnover
uses a one-way entry-rate definition on calendar-rebalance portfolios.  The
cost model is `adjusted_return = return - cost_rate × turnover` with a
user-supplied flat one-way rate.  It does not model market impact, bid-ask
spread variation, short-borrow fees, or execution timing.

## CLI

A thin command-line wrapper over the existing pipeline lives at
`scripts/run_experiment.py`.  It does not redesign the pipeline — it parses
arguments and delegates to the same modules used in notebook workflows.

Top-level CLI routing is also available via:

- `alpha-lab run ...` (legacy single-experiment route)
- `alpha-lab real-case ...` (Level 1/2 research-validation workflows)
- `alpha-lab campaign ...` (Level 1/2 campaign workflows)
- `alpha-lab profiles` (evaluation-profile discovery)
- `alpha-lab web unified` (maintained local research frontend)
- `alpha-lab web ui` (deprecated legacy single-factor UI)

Default real-case/campaign workflow stages are:

1. Level 1 factor evaluation (IC, rank-IC, long-short, stability, uncertainty)
2. Campaign triage
3. Level 2 promotion gate
4. Level 2 portfolio validation (for promoted cases by default)

Evaluation thresholds are selected by `--evaluation-profile`
(currently `default_research`; future profiles can be added centrally).
Real-case inputs (`prices_path`, `universe.path`, and single-factor `factor_path`)
support both CSV (`.csv`) and Parquet (`.parquet` / `.pq`).

```bash
# See all available Level 1/2 evaluation profiles.
alpha-lab profiles

# Single-factor real-case workflow with explicit evaluation profile
alpha-lab real-case single-factor run configs/real_cases/single_factor/bp.yaml \
    --evaluation-profile default_research

# Composite real-case workflow with explicit evaluation profile
alpha-lab real-case composite run configs/real_cases/composite/value_quality_lowvol.yaml \
    --evaluation-profile default_research

# Campaign workflow with explicit evaluation profile
alpha-lab campaign run research_campaign_1 --evaluation-profile default_research

# Profile-aware campaign comparison (built-in lightweight deterministic example)
alpha-lab campaign compare-profiles \
    --source example \
    --output-root-dir dist/examples/profile_aware_campaign_level12 \
    --profiles exploratory_screening default_research stricter_research

# Profile-aware campaign comparison for an existing campaign definition
alpha-lab campaign compare-profiles \
    --source campaign \
    --campaign-config configs/campaigns/research_campaign_1/campaign.yaml \
    --output-root-dir dist/campaign_profile_comparisons/research_campaign_1 \
    --profiles exploratory_screening default_research stricter_research

# Render a Chinese-first local HTML dashboard from comparison artifacts
alpha-lab campaign render-dashboard \
    --comparison-json dist/examples/profile_aware_campaign_level12/campaign_profile_comparison.json

# Start the maintained local research frontend
alpha-lab web unified --host 127.0.0.1 --port 8766
```

The deprecated `alpha-lab web ui` command remains available during the
compatibility window for older single-factor upload flows.

Each run prints the selected evaluation profile, triage label, promotion
decision, portfolio-validation status, and artifact paths.

### End-to-End Level 1/2 Walkthrough

This is the default repository workflow from one case to campaign review.

```bash
# 1) Discover available governance profiles.
alpha-lab profiles

# 2) Run a single-factor case (Level 1 evaluation -> triage -> promotion -> Level 2 validation).
alpha-lab real-case single-factor run configs/real_cases/single_factor/bp.yaml \
    --evaluation-profile default_research \
    --render-report

# 3) Run a composite case with the same profile.
alpha-lab real-case composite run configs/real_cases/composite/value_quality_lowvol.yaml \
    --evaluation-profile default_research \
    --render-report

# 4) Run the campaign to rank cases and aggregate promotion/validation outcomes.
alpha-lab campaign run research_campaign_1 --evaluation-profile default_research --render-report
```

After each case run, inspect:

- `run_manifest.json` for canonical input/output pointers and integrity summary
- `metrics.json` for Level 1 verdict, campaign triage, promotion, and Level 2 validation fields
- `level2_portfolio_validation/` for standardized Level 2 portfolio-validation exports
- `*_workflow_summary.json` (package scripts) and `research_validation_package.json` for auditable export payloads
- Core JSON artifact contracts are now validated at write/read boundaries for:
  `run_manifest.json`, `metrics.json`, `campaign_manifest.json`,
  `campaign_results.json`, `research_validation_package.json`,
  `portfolio_validation_summary.json`, `portfolio_validation_metrics.json`,
  `portfolio_validation_package.json`, and `campaign_profile_comparison.json`.

### Profile-Aware Compact Example

For a reproducible profile comparison on one lightweight case, run:

```bash
uv run --no-sync --frozen python scripts/run_profile_aware_level12_example.py \
    --output-root-dir dist/examples/profile_aware_level12 \
    --profiles exploratory_screening default_research
```

Then inspect:

- `dist/examples/profile_aware_level12/profile_comparison.md`
- `dist/examples/profile_aware_level12/runs/<profile>/profile_aware_bp_single_factor/metrics.json`

Detailed walkthrough: `docs/profile_aware_level12_example.md`.

### Profile-Aware Campaign Comparison (First-Class CLI)

Use the main CLI to compare campaign outcomes across multiple evaluation
profiles and export standardized campaign-level comparison artifacts:

```bash
alpha-lab campaign compare-profiles \
    --source example \
    --output-root-dir dist/examples/profile_aware_campaign_level12 \
    --profiles exploratory_screening default_research stricter_research
```

Or compare an existing campaign definition:

```bash
alpha-lab campaign compare-profiles \
    --source campaign \
    --campaign-config configs/campaigns/research_campaign_1/campaign.yaml \
    --output-root-dir dist/campaign_profile_comparisons/research_campaign_1 \
    --profiles exploratory_screening default_research stricter_research
```

Primary artifacts:

- `campaign_profile_comparison.md`
- `campaign_profile_comparison.json`
- `campaign_profile_case_matrix.csv`
- `campaign_profile_dashboard_zh.html` (factor-first local research workbench dashboard)

Generate the local dashboard explicitly:

```bash
alpha-lab campaign render-dashboard \
    --comparison-json dist/examples/profile_aware_campaign_level12/campaign_profile_comparison.json \
    --output-html dist/examples/profile_aware_campaign_level12/campaign_profile_dashboard_zh.html \
    --overwrite
```

Legacy script compatibility is retained:

```bash
uv run --no-sync --frozen python scripts/run_profile_aware_campaign_level12_example.py \
    --output-root-dir dist/examples/profile_aware_campaign_level12 \
    --profiles exploratory_screening default_research stricter_research
```

**Input CSV** must contain at least the columns `date`, `asset`, and `close`.
Extra columns are ignored.

```bash
# Minimal run — writes a summary CSV to output/
uv run python scripts/run_experiment.py \
    --input-path data/raw/prices.csv \
    --factor momentum \
    --label-horizon 5 \
    --quantiles 5

# Full run: split, cost rate, Obsidian note, registry entry
uv run python scripts/run_experiment.py \
    --input-path data/raw/prices.csv \
    --factor momentum \
    --momentum-window 20 \
    --label-horizon 5 \
    --quantiles 5 \
    --train-end 2022-12-31 \
    --test-start 2023-01-01 \
    --cost-rate 0.001 \
    --experiment-name momentum_20d_5q_oos_2023 \
    --output-dir output/reports \
    --obsidian-markdown-path notes/momentum_20d_5q_oos_2023.md \
    --append-registry

# Write the note into a directory — filename is auto-generated as
# YYYY-MM-DD_{experiment_name}.md
uv run python scripts/run_experiment.py \
    --input-path data/raw/prices.csv \
    --factor momentum \
    --label-horizon 5 \
    --quantiles 5 \
    --obsidian-markdown-path notes/
```

`--experiment-name` must contain only letters, digits, hyphens, underscores,
and dots — path separators are rejected to prevent accidental file writes
outside `--output-dir`.

## Comparison and Registry

`alpha_lab.comparison` is a legacy low-level helper for side-by-side summaries.
For new Level 1/2 comparison workflows, prefer campaign profile comparison
artifacts such as `campaign_profile_comparison.json` and
`campaign_profile_case_matrix.csv`. The helper remains available for older
notebooks that compare one-row experiment summaries:

```python
from alpha_lab.comparison import compare_experiments, rank_experiments
from alpha_lab.registry import load_registry, register_experiment
from alpha_lab.reporting import summarise_experiment_result

# --- 1. Run experiments and summarise ---
result_a = run_factor_experiment(prices, lambda p: momentum(p, window=20), horizon=5)
result_b = run_factor_experiment(prices, lambda p: momentum(p, window=5), horizon=5)

summary_a = summarise_experiment_result(result_a)
summary_b = summarise_experiment_result(result_b)

# --- 2. Compare side-by-side ---
comparison = compare_experiments([summary_a, summary_b])
ranked = rank_experiments(comparison, metric="ic_ir")

# --- 3. Register to the CSV log ---
register_experiment("momentum_20d_5h", summary_a)
register_experiment("momentum_5d_5h",  summary_b)

# --- 4. Reload the registry ---
registry = load_registry()
```

The registry is stored at `data/processed/experiment_registry.csv` by default.
Each call to `register_experiment` appends one row; the file is created on
first use.  The registry is an append-only log — duplicate experiment names
are permitted.  Schema consistency is checked on every append and load.

## Walk-Forward Evaluation and Portfolio Research

### Walk-Forward Evaluation

A single train/test split can overfit to the test period: the researcher may
consciously or unconsciously choose factor parameters that happen to look good
on that one window.  Walk-forward evaluation forces every evaluation date to
be strictly out-of-sample by rolling the train and test windows forward
through time.

`run_walk_forward_experiment` wraps `run_factor_experiment` over all folds
produced by `walk_forward_split`.  Each fold receives only the prices visible
up to its own test-end date — no future data can leak into the factor
computation.  Evaluation metrics (IC, L/S return, turnover) are computed on
the test window only.

```python
from alpha_lab.walk_forward import run_walk_forward_experiment
from alpha_lab.factors.momentum import momentum

wf = run_walk_forward_experiment(
    prices,
    lambda p: momentum(p, window=20),
    train_size=252,   # 1-year training window (trading days)
    test_size=63,     # 1-quarter test window
    step=63,          # advance by one quarter between folds
    horizon=5,
    n_quantiles=5,
    cost_rate=0.001,
)

print(wf.aggregate_summary)
# WalkForwardAggregate(n_folds=4, mean_ic=..., std_ic=...,
#   pooled_ic_mean=..., pooled_ic_ir=...,
#   mean_portfolio_return=..., pooled_portfolio_return_mean=...,
#   pooled_cost_adjusted_return_mean=..., ...)

# Per-fold breakdown
print(wf.fold_summary_df[["fold_id", "start_date", "end_date", "mean_ic", "ic_ir"]])

# Each fold's full ExperimentResult is also available
first_fold = wf.per_fold_results[0]

# Pooled OOS observation DataFrames (all folds concatenated, test window only)
wf.pooled_ic_df                              # [fold_id, date, ic]
wf.pooled_portfolio_return_df               # [fold_id, date, portfolio_return]
wf.pooled_portfolio_turnover_df             # [fold_id, date, portfolio_turnover]
wf.pooled_cost_adjusted_portfolio_return_df # [fold_id, date, portfolio_return, adjusted_return]
```

**Why walk-forward reduces overfitting risk**: the aggregate `mean_ic` and
`std_ic` reflect the factor's consistency across multiple independent test
windows.  A high `std_ic` relative to `mean_ic` signals that performance is
unstable and may not generalise.  The `best_fold` / `worst_fold` fields
identify the most and least favourable periods for deeper investigation.

**Single split vs. walk-forward**: `run_factor_experiment` with a single
`train_end` / `test_start` evaluates on one contiguous test window.
Walk-forward provides multiple independent windows of the same total span,
giving a distribution of outcomes rather than a point estimate.

### Portfolio Research Layer

`alpha_lab.portfolio_research` provides research-level portfolio construction
and simulation tools.  These are designed for signal evaluation, not live
execution.

#### Computing weights

```python
from alpha_lab.portfolio_research import portfolio_weights

# Long-only: top 20 assets, weights proportional to factor rank
weights = portfolio_weights(
    factor_df,
    method="rank",   # or "equal", "score"
    top_k=20,
)

# Long-short: top 10 long (+weight sums to 1),
#             bottom 10 short (weight sums to -1, net = 0)
ls_weights = portfolio_weights(
    factor_df,
    method="equal",
    top_k=10,
    bottom_k=10,
)
```

Weight methods:
- `"equal"`: uniform weight across selected assets.
- `"rank"`: weight proportional to cross-sectional factor rank.
- `"score"`: weight proportional to `value − min(value)` across the selection.

#### Simulating returns with overlapping holdings

```python
from alpha_lab.portfolio_research import simulate_portfolio_returns
from alpha_lab.labels import forward_return

# 1-period returns (pass result.label_df or compute fresh)
labels = forward_return(prices, horizon=1)

port_returns = simulate_portfolio_returns(
    weights,
    labels,
    holding_period=5,       # hold each position for 5 rebalance periods
    rebalance_frequency=1,  # rebalance at every available date
)
# Returns: DataFrame[date, portfolio_return]
```

When `holding_period > rebalance_frequency`, multiple overlapping positions
are active simultaneously.  The portfolio return on each date is the mean
across all currently active positions — the standard staggered-portfolio
model used in academic factor research.

#### Portfolio turnover

```python
from alpha_lab.portfolio_research import portfolio_turnover

to = portfolio_turnover(weights)
# Returns: DataFrame[date, portfolio_turnover]
# turnover(t) = 0.5 × Σ|w_new_i − w_old_i|  (two-way, fraction traded)
# First date is always NaN (no prior state).
```

#### Integrating into run_factor_experiment

Pass `holding_period` and `rebalance_frequency` to attach portfolio
outputs to the standard `ExperimentResult`:

```python
result = run_factor_experiment(
    prices,
    lambda p: momentum(p, window=20),
    horizon=5,
    n_quantiles=5,
    holding_period=1,
    rebalance_frequency=1,
    weighting_method="rank",
)

result.portfolio_weights_df   # DataFrame[date, asset, weight]
result.portfolio_return_df    # DataFrame[date, portfolio_return]
```

**Research disclaimer**: this is a minimal friction estimate for signal
evaluation only.  It does not model market impact, intraday slippage,
short-borrow costs, execution timing, or partial fills.

## Current Limitations

- no full backtesting engine or realistic execution simulation
- no transaction-cost model beyond linear flat-rate research approximation
- no database, deployed multi-user dashboard, or experiment tracking framework
- Execution/implementability code is intentionally not part of this release
