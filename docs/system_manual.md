# System Manual

Reference guide for working with Alpha Lab.  For higher-level architectural
context see [architecture.md](architecture.md).  For contributor guidance see
[developer_guide.md](developer_guide.md).

---

## Scope

`alpha-lab` is a Level 1/2 research system:

- Level 1: factor discovery
- Level 2: portfolio construction validation

It is not an execution replay engine, implementability-grade platform,
order-fill simulator, or microstructure execution simulator. Execution realism
is intentionally out of scope.

---

## Setup

**Requirements**: Python 3.12, `uv`.

```bash
uv sync --all-extras
```

**Path behaviour**: the package resolves the project root via
`src/alpha_lab/config.py`.  For editable installs (the standard workflow)
this works automatically.  For non-editable installs, set:

```bash
export ALPHA_LAB_PROJECT_ROOT=/path/to/alpha-lab
```

**External data root**: vendor-backed research datasets should live outside the
repository.  By default the external store resolves to
`~/.local/share/alpha-lab/data`. Override it when needed:

```bash
export ALPHA_LAB_DATA_ROOT=/path/to/alpha-lab-data
```

A `RuntimeError` is raised at import time if the resolved root does not contain
`pyproject.toml`, preventing silent artifact misplacement.

Run all checks:

```bash
make check   # lint + typecheck + test
```

In WSL or sandboxed environments with restricted cache directories:

```bash
UV_CACHE_DIR=/tmp/uv-cache make check
```

---

## CLI Workflow Map

`alpha-lab --help` shows the unified Level 1/2 router:

- `alpha-lab run ...` (legacy single-experiment route)
- `alpha-lab real-case ...` (single-factor/model-factor/composite case workflows)
- `alpha-lab campaign ...` (campaign orchestration and ranking)
- `alpha-lab bridge ...` (project-pack / round-context / writeback drafting)
- `alpha-lab data ...` (external dataset ingest / update / export)
- `alpha-lab profiles` (evaluation-profile discovery)

Profile discovery:

```bash
alpha-lab profiles
```

Read-only data inspection example:

```bash
alpha-lab data query \
  --sql "select date, asset, close from daily_bars order by date desc, asset limit 5"
```

Evaluation profile guidance:

| Profile | Primary use | Decision tendency |
|---|---|---|
| `default_research` | Routine Level 1/2 research runs | Balanced baseline standards |
| `exploratory_screening` | Broad candidate discovery and early funneling | More permissive: easier to retain borderline candidates |
| `stricter_research` | Higher-confidence evidence and Level 2-readiness review | More conservative: stricter robustness/promotion/validation checks |

Profile choice affects all default Level 1/2 decision stages:

1. factor verdict classification
2. campaign triage label and ranking metadata
3. Level 2 promotion gate outcomes
4. Level 2 portfolio-validation guardrails and recommendations

Uncertainty mode defaults to normal-approximation CI (`uncertainty.method=normal`).
Bootstrap CI is opt-in through evaluation config/profile
(`uncertainty.method=bootstrap` or `uncertainty.method=block_bootstrap`) and will
propagate method metadata into outputs.

Campaign flag forwarding example:

```bash
alpha-lab campaign run research_campaign_1 --evaluation-profile default_research
```

Bridge quickstart:

```bash
alpha-lab bridge init-project \
  --slug momentum-factor \
  --title-zh 动量因子项目 \
  --category factor_family \
  --owner yukun \
  --market ashare \
  --frequency daily \
  --chatgpt-project-name "Momentum Factor"

alpha-lab bridge start-round \
  --project momentum-factor \
  --topic "三个月成交额加权动量"
```

---

## End-to-End Level 1/2 Walkthrough

```bash
# 1) Inspect available profile(s)
alpha-lab profiles

# 2) Run single-factor case workflow
alpha-lab real-case single-factor run configs/real_cases/single_factor/bp.yaml \
  --evaluation-profile default_research \
  --render-report

# 3) Run model-factor case workflow
alpha-lab real-case model-factor run configs/real_cases/model_factor/demo.yaml \
  --evaluation-profile default_research \
  --render-report

# 4) Run composite case workflow
alpha-lab real-case composite run configs/real_cases/composite/value_quality_lowvol.yaml \
  --evaluation-profile default_research \
  --render-report

# 5) Run campaign workflow
alpha-lab campaign run research_campaign_1 \
  --evaluation-profile default_research \
  --render-report
```

Expected Level 1/2 flow for each case:

1. Level 1 evaluation
2. Campaign triage
3. Level 2 promotion gate
4. Level 2 portfolio validation

For `real-case model-factor`, the case-specific front half is:

1. load wide research feature table
2. train rolling/expanding cross-sectional model on past data only
3. emit canonical factor output `[date, asset, factor, value]`
4. continue through the standard Level 1/2 flow above

Primary artifacts to audit:

- `run_manifest.json`
- `metrics.json`
- `summary.md`
- `experiment_card.md`
- `level2_portfolio_validation/portfolio_validation_summary.json`
- `level2_portfolio_validation/portfolio_validation_metrics.json`
- `level2_portfolio_validation/portfolio_validation_package.json`

Core Level 1/2 JSON contract validation is enforced for:
`run_manifest.json`, `metrics.json`, `campaign_manifest.json`,
`campaign_results.json`, `research_validation_package.json`,
`portfolio_validation_summary.json`, `portfolio_validation_metrics.json`,
`portfolio_validation_package.json`, and `campaign_profile_comparison.json`.

### Tushare Pro Real-Data Quickstart

Use a small pilot first to control point usage, then scale up. The recommended
flow is: ingest raw snapshots into the external Parquet-backed data store, then
export a compact case slice as CSV inputs for the existing Level 1/2 pipeline.

```bash
# 0) Set token (replace with your real token)
export TUSHARE_TOKEN="your_tushare_token"

# 1) Initialize the external data root once
alpha-lab data init

# 2) Ingest a pilot Tushare core dataset into the external Parquet store
alpha-lab data ingest tushare core \
  --start-date 2024-01-01 \
  --end-date 2024-03-31 \
  --asset-limit 100

# 3) Export the requested Level 1/2 case slice as canonical CSV inputs
alpha-lab data export-case-inputs \
  --output-dir data/processed/real_case_inputs/tushare_v1 \
  --slice-preset standard

# Optional: use a custom ts_code list instead of full-market fetch
# --assets-file path/to/ts_codes.txt

# Optional: control long-window chunking
# --chunk-months 6   # default
# --chunk-months 0   # disable chunking

# Optional: for daily price-volume research only, skip slow ROE fetches
# --daily-research-only

# Optional: explicitly export bp / roe_ttm when needed
# --factors bp roe_ttm

# Optional: fully override the preset window or export mode
# --start-date 2021-01-01 --end-date 2024-03-31 --universe all_ashare --adjustment raw

# 4) Run single-factor real-case pipelines on generated inputs
alpha-lab real-case single-factor run configs/real_cases/bp_tushare_v1.yaml \
  --evaluation-profile default_research \
  --render-report

alpha-lab real-case single-factor run configs/real_cases/roe_tushare_v1.yaml \
  --evaluation-profile default_research \
  --render-report
```

If you see `您的token不对，请确认。`, your current token is invalid or expired.
Re-check the token value and reset `TUSHARE_TOKEN`.

内置的日频量价切片预设如下：

- `pilot`: 最近 3 年，`top_liquid_300`，`qfq`
- `standard`: 最近 5 年，`listed_90d`，`qfq`
- `robust`: 最近 8 年，`listed_90d`，`qfq`
- `institutional`: 最近 8 年，`institutional_ashare`，`qfq`

目前支持的流动性 universe 额外包括：

- `institutional_ashare`
- `top_liquid_300`
- `top_liquid_500`
- `top_liquid_800`

`institutional_ashare` 会保留上市满 `180` 天、当日非 ST/非停牌，且通过过去
`20` 个样本日流动性筛选的 A 股：平均 `amount >= 20000`（Tushare 日频
`amount` 单位为千元，约等于人民币 2000 万元），并剔除当日流动性后 `20%`。
`top_liquid_*` 则按过去 `60` 个交易日平均成交额 `amount` 排序，而不是按当日成交额排序。

针对 A 股日频量价因子研究，`alpha-lab data export-case-inputs` 现在默认导出更完整的
研究列：

- `open / high / low / close / pre_close`
- `volume / amount / vwap / turnover_rate`
- `up_limit / down_limit / is_limit_up / is_limit_down`
- `is_suspended / is_st`
- `is_hs300 / is_zz500 / is_zz1000 / is_sz50`

当 canonical 主库中存在对应表时，还会额外导出：

- `asset_status.csv`
- `index_membership.csv`

覆盖约束如下：

- 最低要求：最新可用日期向前至少覆盖最近 3 年，否则 `validate` 会报错
- 稳健目标：最新可用日期向前覆盖最近 8 年；如果不足 8 年，`validate` 会给 warning
- `export-case-inputs` 会拒绝超出 `daily_bars` 可用范围的窗口，但会用交易日历把非交易日的开始/结束日期映射到最近的开市边界

长窗口的 Tushare ingest / update 默认会自动分块：

- 默认按 `6` 个月分块
- CLI 会打印每个 chunk 和每个主要抓取阶段的进度
- 如果你明确想关掉分块，可以传 `--chunk-months 0`

如果你当前只做日频量价研究，推荐在 ingest / update 时加上
`--daily-research-only`。这样会跳过最慢的 ROE / `financial_indicator`
抓取，但仍保留以下研究必需数据：

- `daily_bars` 中的 `open/high/low/close/pre_close/volume/amount/vwap/turnover_rate`
- `up_limit/down_limit/is_limit_up/is_limit_down`
- `asset_status` 中的 `is_suspended/is_st`
- `index_membership` 及 `prices.csv` 里的指数成分标记列

### BaoStock Real-Data Quickstart

BaoStock does not require Tushare points and can be used as an alternative
A-share source.

```bash
# 1) Generate canonical CSV inputs from BaoStock (pilot mode)
UV_CACHE_DIR=/tmp/uv-cache uv run --with baostock python \
  scripts/generate_baostock_real_case_inputs.py \
  --start-date 2024-01-01 \
  --end-date 2024-03-31 \
  --asset-limit 100 \
  --output-dir data/processed/real_case_inputs/baostock_v1

# Optional: use a custom stock list
# --assets-file path/to/ashare_codes.txt

# 2) Run single-factor real-case pipeline on canonical Tushare inputs
alpha-lab real-case single-factor run configs/real_cases/single_factor/mom20_ex5_reversal_5d_tushare_qfq_listed90.yaml \
  --evaluation-profile default_research \
  --render-report
```

For a stable first-pass demo, use a price-only recipe config like
`mom20_ex5_reversal_5d_tushare_qfq_listed90` before moving to wider data bundles.

### Research Bridge + ChatGPT Projects

`alpha-lab bridge` 是 `quant-knowledge + alpha-lab + ChatGPT Projects` 混合工作流的
项目层。它不会改变默认的 Level 1/2 研究主路径，只负责：

- 在 `quant-knowledge/55_projects/<slug>/` 下维护项目状态
- 生成适合上传到 ChatGPT Project 的稳定项目包与单轮上下文包
- 把讨论结果脚手架成 `alpha-lab` case 草案
- 把实验结果先沉淀为人工审核草稿，再正式写回 `50_experiments/`

标准命令如下：

```bash
alpha-lab bridge refresh-project-pack --project momentum-factor
alpha-lab bridge start-round --project momentum-factor --topic "三个月成交额加权动量"
alpha-lab bridge scaffold-case --project momentum-factor --round <round_id> --case-name mom_amt_60
alpha-lab bridge summarize-run --project momentum-factor --round <round_id> --run-root <run_dir>
alpha-lab bridge apply-writeback --project momentum-factor --draft <draft_path>
```

生成物分层：

- 稳定项目包：
  - `01_project_brief.md`
  - `02_project_rules.md`
  - `03_card_map.md`
  - `10_active_state.md`
- 单轮包：
  - `30_rounds/<round_id>/round_context_digest.md`
  - `30_rounds/<round_id>/round_prompt.md`
  - `30_rounds/<round_id>/web_search_tasks.md`
  - `30_rounds/<round_id>/discussion_capture.md`
- 回写草稿：
  - `50_writeback_drafts/*__writeback_draft.md`
  - `50_writeback_drafts/*__state_update_patch.md`

`apply-writeback` 默认要求 draft frontmatter 里已经人工改成
`review_status: approved`。这一步完成后，bridge 会：

- 把 `experiment_card.md / summary.md / run_manifest.json` 写入 `50_experiments/`
- 更新 `55_projects/<slug>/10_active_state.md`
- 向 `55_projects/<slug>/20_decision_log.md` 追加一条项目级结论

### Profile-Aware Compact Workflow

Run one deterministic case under two profiles and export side-by-side
comparisons:

```bash
uv run --no-sync --frozen python scripts/run_profile_aware_level12_example.py \
  --output-root-dir dist/examples/profile_aware_level12 \
  --profiles exploratory_screening default_research
```

Inspect:

- `dist/examples/profile_aware_level12/profile_comparison.md`
- `dist/examples/profile_aware_level12/profile_comparison.json`
- `dist/examples/profile_aware_level12/runs/<profile>/profile_aware_bp_single_factor/metrics.json`

See `docs/profile_aware_level12_example.md` for the full walkthrough.

### Profile-Aware Campaign Comparison Workflow

Run a compact deterministic multi-case campaign under multiple profiles and
export campaign-level profile sensitivity:

```bash
alpha-lab campaign compare-profiles \
  --source example \
  --output-root-dir dist/examples/profile_aware_campaign_level12 \
  --pair-mode adjacent \
  --artifact-hint-path-mode relative \
  --profiles exploratory_screening default_research stricter_research
```

Inspect:

- `dist/examples/profile_aware_campaign_level12/campaign_profile_comparison.md`
- `dist/examples/profile_aware_campaign_level12/campaign_profile_comparison.json`
- `dist/examples/profile_aware_campaign_level12/campaign_profile_case_matrix.csv`
- `dist/examples/profile_aware_campaign_level12/campaign_profile_dashboard_zh.html` (factor-first dashboard)

For an existing campaign definition:

```bash
alpha-lab campaign compare-profiles \
  --source campaign \
  --campaign-config configs/campaigns/research_campaign_1/campaign.yaml \
  --output-root-dir dist/campaign_profile_comparisons/research_campaign_1 \
  --pair-mode adjacent \
  --artifact-hint-path-mode relative \
  --profiles exploratory_screening default_research stricter_research
```

By default, comparison artifact hints are rendered relative to `--output-root-dir`
for portability. Use `--artifact-hint-path-mode absolute` to keep absolute paths.
Use `--pair-mode all_pairs` to include non-adjacent profile pairs such as
`exploratory_screening -> stricter_research`.

Render a Chinese-first local HTML dashboard from comparison outputs:

```bash
alpha-lab campaign render-dashboard \
  --comparison-json dist/examples/profile_aware_campaign_level12/campaign_profile_comparison.json \
  --output-html dist/examples/profile_aware_campaign_level12/campaign_profile_dashboard_zh.html \
  --overwrite
```

Legacy script route remains available for backward compatibility:

```bash
uv run --no-sync --frozen python scripts/run_profile_aware_campaign_level12_example.py \
  --output-root-dir dist/examples/profile_aware_campaign_level12 \
  --profiles exploratory_screening default_research stricter_research
```

See `docs/profile_aware_campaign_level12_example.md` for the full walkthrough.

---

## Raw Input Validation

`alpha_lab.data_validation.validate_price_panel(df)` is called automatically at
`run_factor_experiment()` and at the CLI entry point.  It raises `ValueError`
(or `SystemExit` at the CLI) on the first violation:

| Check | Error trigger |
|---|---|
| Required columns | `date`, `asset`, `close` missing |
| Non-empty | zero rows |
| Valid dates | NaT or unparseable values in `date` |
| Non-null asset | null or empty-string in `asset` |
| No duplicates | duplicate `(date, asset)` pairs |
| NaN close | any NaN in `close` |
| Positive close | any `close <= 0` |

You can also call it directly before passing data to any pipeline function:

```python
from alpha_lab.data_validation import validate_price_panel
validate_price_panel(your_prices_df)  # raises ValueError on violation
```

---

## Canonical Data Contract

All factor outputs must conform to the long-form schema:

| Column | Type | Description |
|--------|------|-------------|
| `date` | datetime | Observation timestamp |
| `asset` | str | Asset identifier |
| `factor` | str | Factor name |
| `value` | float | Numeric factor value |

Rules:
- At most one row per `(date, asset, factor)`.
- Factor values at `date=t` may only use information available at or before `t`.
- Labels and forward returns must be stored in **separate** tables.

---

## Core API

### run_factor_experiment

Connects all evaluation modules (IC, quantile returns, long-short, turnover,
and optionally portfolio simulation) into a single call.

```python
from alpha_lab.experiment import run_factor_experiment
from alpha_lab.factors.momentum import momentum

result = run_factor_experiment(
    prices,                      # long-form [date, asset, close]
    lambda p: momentum(p, window=20),
    horizon=5,                   # forward-return look-ahead in rows
    n_quantiles=5,               # quantile buckets for IC/quantile eval
    train_end="2022-12-31",      # omit for full-sample evaluation
    test_start="2023-01-01",
)

# Core outputs (always present)
result.summary               # ExperimentSummary scalar metrics
result.ic_df                 # [date, factor, ic]
result.rank_ic_df            # [date, factor, rank_ic]
result.quantile_returns_df   # [date, factor, quantile, quantile_return]
result.long_short_df         # [date, factor, long_short_return]
result.factor_df             # full-sample factor values
result.label_df              # full-sample forward-return labels
```

**Portfolio simulation** (optional):

```python
result = run_factor_experiment(
    prices,
    lambda p: momentum(p, window=20),
    horizon=5,
    holding_period=1,
    rebalance_frequency=1,
    weighting_method="rank",    # "equal", "rank", or "score"
    portfolio_cost_rate=0.001,  # 10 bps one-way; omit for no cost adjustment
)

result.portfolio_weights_df             # [date, asset, weight]
result.portfolio_return_df              # [date, portfolio_return]
result.portfolio_turnover_df            # [date, portfolio_turnover] (active rebalance dates)
result.portfolio_cost_adjusted_return_df  # [date, portfolio_return, adjusted_return]
result.portfolio_summary                # PortfolioSummary scalars
```

**Using StrategySpec** (makes construction intent explicit):

```python
from alpha_lab.strategy import StrategySpec

spec = StrategySpec(
    long_top_k=10,
    weighting_method="rank",
    holding_period=1,
    rebalance_frequency=1,
)

result = run_factor_experiment(
    prices,
    lambda p: momentum(p, window=20),
    horizon=5,
    n_quantiles=5,         # factor-eval param; not part of StrategySpec
    strategy=spec,
    portfolio_cost_rate=0.001,
)
```

When `strategy` is provided, `holding_period`, `rebalance_frequency`, and
`weighting_method` are taken from the spec.  Passing them explicitly alongside
`strategy` raises a `UserWarning` (spec values win).

---

### run_walk_forward_experiment

Rolls `run_factor_experiment` across non-overlapping test windows.  Every
evaluation date is strictly out-of-sample.

```python
from alpha_lab.walk_forward import run_walk_forward_experiment

wf = run_walk_forward_experiment(
    prices,
    lambda p: momentum(p, window=20),
    train_size=252,    # unique dates in each training window
    test_size=63,      # unique dates in each test window
    step=63,           # advance by this many dates between folds
    horizon=5,
    n_quantiles=5,
    cost_rate=0.001,   # long-short cost-adjusted return (separate from portfolio path)
)

# Fold-level summary
wf.fold_summary_df          # one row per fold
wf.per_fold_results         # list of ExperimentResult, one per fold

# Aggregate statistics
agg = wf.aggregate_summary  # WalkForwardAggregate
agg.n_folds
agg.pooled_ic_mean          # mean IC across all OOS observations
agg.pooled_ic_ir            # IC-IR from pooled series
agg.best_fold               # fold_id with highest mean_ic

# Pooled OOS DataFrames
wf.pooled_ic_df                              # [fold_id, date, ic]
wf.pooled_portfolio_return_df               # [fold_id, date, portfolio_return]
wf.pooled_portfolio_turnover_df             # [fold_id, date, portfolio_turnover]
wf.pooled_cost_adjusted_portfolio_return_df # [fold_id, date, portfolio_return, adjusted_return]
```

**`val_size`** is a fold-construction parameter only.  It reserves trailing
training-window dates as a gap between training and test windows.  No
validation-period outputs are produced — the validation dates are excluded from
both training and test evaluation.

---

### StrategySpec

Frozen dataclass that is the explicit boundary between the factor research
layer and the portfolio research layer.

```python
from alpha_lab.strategy import StrategySpec

# Long-only
spec = StrategySpec(
    long_top_k=10,            # None = all assets
    weighting_method="rank",  # "equal", "rank", or "score"
    holding_period=1,
    rebalance_frequency=1,
)

# Long-short (net-zero)
ls_spec = StrategySpec(
    long_top_k=5,
    short_bottom_k=5,
    weighting_method="equal",
    holding_period=2,
    rebalance_frequency=1,
)

spec.is_long_short   # True when short_bottom_k is not None
```

`n_quantiles` is **not** a field of `StrategySpec`.  It governs the
factor-evaluation path (IC, quantile returns) and is passed directly to the
experiment runner.

---

### Reporting

```python
from alpha_lab.reporting import (
    summarise_experiment_result,
    export_summary_csv,
    to_obsidian_markdown,
    export_experiment_card,
)

summary = summarise_experiment_result(result, cost_rate=0.001)
export_summary_csv(summary, "output/reports/momentum_5d.csv")
md = to_obsidian_markdown(result, title="Momentum 5d OOS", cost_rate=0.001)
```

#### export_experiment_card

Writes a structured experiment note to `{vault}/50_experiments/Exp - YYYYMM - {name}.md`
using the quant-knowledge frontmatter schema.

```python
path = export_experiment_card(result, name="momentum-5d-Ashare")
# requires OBSIDIAN_VAULT_PATH to be set, unless vault_path is passed explicitly
# returns the resolved Path of the written file

# Explicit vault and overwrite:
path = export_experiment_card(
    result,
    name="momentum-5d-Ashare",
    vault_path="/path/to/quant-knowledge",
    overwrite=True,
)
```

**Behaviour:**
- The vault root must already exist (`FileNotFoundError` if not).
- The `50_experiments/` subdir is created automatically if absent.
- Default is safe: raises `FileExistsError` if the card already exists.
  Pass `overwrite=True` to replace an existing card intentionally.
- `name` must be non-empty and must not contain path separators.
- Returns the resolved `Path` of the written file.

**Generated vs manual sections:**
Setup, Results, and YAML frontmatter are auto-generated from the
`ExperimentResult` and must not be edited manually.  The note includes a
visible notice to that effect.  Interpretation, Next Steps, Open Questions,
and Notes are placeholders for researcher completion.

Generated research cards should be Chinese-first in their human-readable
sections. Keep the body and headings in Chinese by default, and preserve
English only for necessary technical terms, proper nouns, formulas, code
symbols, file paths, and quoted source titles.

**Vault path resolution order:**
`vault_path` argument → `OBSIDIAN_VAULT_PATH` env var.  An empty or whitespace
env var is treated as "not configured".

---

### Registry

Append-only CSV log of experiment results.  Default location:
`<project_root>/data/processed/experiment_registry.csv`
(anchored by `alpha_lab.config.PROCESSED_DATA_DIR`, not the current working
directory).

```python
from alpha_lab.registry import register_experiment, load_registry

register_experiment("momentum_20d_5q_oos_2023", summary)
registry = load_registry()
```

Schema is validated on every append and load; mismatches raise `ValueError`.

---

### Comparison

`alpha_lab.comparison` 是旧版低层 helper，保留给已有 notebook 和脚本兼容。
新的 Level 1/2 对比流程优先使用 `campaign_profile_comparison.json`、
`campaign_profile_case_matrix.csv` 等 campaign profile comparison artifacts。

```python
from alpha_lab.comparison import compare_experiments, rank_experiments

comparison = compare_experiments([summary_a, summary_b])
ranked = rank_experiments(comparison, metric="ic_ir")
```

---

## Timestamp Discipline

- `forward_return(prices, horizon=h)` stores `close[t+h]/close[t]-1` **at
  row `t`** so it can be merged with factor values observed at `t` without
  lookahead.  The label value uses strictly future prices.
- Portfolio simulation inside `_run_portfolio_block` always uses
  `forward_return(prices, horizon=1)` (one-period step returns), **not** the
  H-period evaluation labels, to avoid compounding mismatch in the
  staggered-portfolio model.

---

## Cost Model

```
adjusted_return(t) = portfolio_return(t) − cost_rate × turnover(t)
```

- Applied only on **active rebalance dates** (every `rebalance_frequency`-th
  weight date).
- First active rebalance date is always `NaN` (no prior portfolio state).
- Non-rebalance evaluation dates receive `adjusted_return = portfolio_return`
  (zero incremental cost).
- `cost_rate` is one-way, flat-rate.  It does not model market impact,
  bid-ask spread variation, short-borrow fees, or execution timing.

---

## Provenance and Diagnostics

Every `ExperimentResult` carries a `provenance` field (`ExperimentProvenance`)
and three diagnostic counts:

```python
result.provenance.factor_name        # e.g. "momentum_5d"
result.provenance.horizon            # forward-return horizon used
result.provenance.n_quantiles        # quantile buckets used
result.provenance.run_timestamp_utc  # ISO-8601 UTC run time
result.provenance.git_commit         # short commit hash or None
result.provenance.portfolio_cost_rate
result.provenance.strategy_repr      # repr(spec) or None

result.n_eval_dates       # distinct dates in the evaluation period
result.n_eval_assets      # distinct assets in the evaluation period
result.n_label_nan_dates  # eval dates with no valid forward return label
                          # (= horizon for full-sample runs)
```

`n_label_nan_dates` tells you how many trailing dates were excluded from IC and
quantile-return computation because the forward-return horizon extended beyond
the available price history.

---

## Parameter Misuse Warnings

Two `UserWarning`s are raised for clearly no-op parameter combinations:

1. **`portfolio_cost_rate` without portfolio mode** — if `portfolio_cost_rate`
   is supplied but neither `holding_period`/`rebalance_frequency` nor a
   `StrategySpec` is provided, the rate would be silently dropped.  A warning
   is raised in both `run_factor_experiment` and `run_walk_forward_experiment`.

2. **`holding_period`/`rebalance_frequency` alongside `strategy`** — the spec
   values override; the explicit arguments are warned and ignored.

---

## Scope Limitations

- No full backtesting engine or realistic execution simulation.
- No position accounting or broker integration.
- No market impact or intraday slippage model.
- No database, deployed multi-user dashboard, or streaming experiment tracking.
- Cost model is a minimal research friction estimate only.
