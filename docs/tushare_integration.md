# Tushare Integration

Alpha Lab includes a disciplined Tushare ingestion layer for real A-share
research inputs. The design goal is to keep vendor extraction, internal
standardization, and research workflows separate.

## Scope

This integration supports the minimum real-data surface needed for two
canonical research cases:

- single-factor: liquidity-screened short-term reversal
- composite: value + quality + momentum

It does not turn workflows into online API jobs. Tushare is only used in the
ingestion layer.

## Layer Boundaries

The data path is:

```text
Tushare Pro API
  -> data/raw/tushare/<snapshot_name>/
  -> data/processed/tushare_standardized/<snapshot_name>/
  -> data/processed/tushare_research_inputs/<snapshot_name>/
  -> existing workflow configs / handoff / replay / research package
```

Responsibilities are explicit:

- `tushare_client.py`: live client wrapper, lazy import, no workflow coupling
- `tushare_extractors.py`: endpoint extraction and snapshot manifests
- `tushare_cache.py`: deterministic raw snapshot persistence
- `tushare_standardize.py`: vendor-schema to research-table mapping
- `tushare_bundle_builder.py`: workflow-compatible bundle and case configs

## Raw Snapshots

Raw snapshots are stored under:

```text
data/raw/tushare/<snapshot_name>/
```

Each endpoint writes:

- `<endpoint>.csv`: vendor columns preserved as returned
- `<endpoint>.meta.json`: extraction timestamp, params, row count, columns
- `manifest.json`: batch-level endpoint list, unavailable optional endpoints, notes

Properties:

- deterministic sorting on known vendor keys
- extraction parameters preserved
- reproducible reload through `load_raw_snapshot(...)`
- optional endpoints recorded explicitly when unavailable
- per-endpoint extraction provenance is recorded in `manifest.json` under
  `endpoint_extractions`

### Extraction Modes

`fetch_tushare_raw_snapshots(...)` uses explicit endpoint extraction modes:

- `range_query`: endpoints that support `start_date/end_date` windows
  (for example `daily`, `daily_basic`, `trade_cal`, `suspend_d`, `stk_limit`)
- `static_snapshot`: endpoint-specific snapshot queries (for example `stock_basic`)
- `financial_quarterly`: endpoint-specific logic for financial indicators

Financial indicators are intentionally handled differently from market endpoints:

- `fina_indicator` requires `ts_code`; it is not a market-wide range API
- extraction first tries `fina_indicator_vip(period=...)` over quarter-end periods
  that cover the requested window
- if VIP is unavailable, extraction falls back to per-stock
  `fina_indicator(ts_code=..., start_date=..., end_date=...)`
- ingestion never calls `fina_indicator` with only `start_date/end_date`

## Standardized Research Tables

`build_standardized_tushare_tables(...)` maps raw vendor files into explicit
internal tables:

- `prices.csv`
- `asset_metadata.csv`
- `trade_calendar.csv`
- `market_state.csv`
- `daily_fundamentals.csv`
- `financial_indicators.csv`
- `manifest.json`

### Schema Notes

`prices.csv`
- canonical keys: `date`, `asset`
- uses raw `daily.close`
- `volume` = `daily.vol * 100`
- `dollar_volume_yuan` = `daily.amount * 1000`
- carries daily `pb`, `pe_ttm`, `total_mv_yuan`, `circ_mv_yuan` when available

`asset_metadata.csv`
- normalized asset identifier from `ts_code`
- listing / delisting dates parsed explicitly
- `is_st` inferred conservatively from stock name containing `ST`
- industry, market, exchange, listing status preserved when available

`market_state.csv`
- `is_halted` from `suspend_d` when available, otherwise conservative fallback
- `is_limit_locked` from `stk_limit` when available
- `is_st` joined from asset metadata

`financial_indicators.csv`
- announcement date and report period normalized
- quality fields restricted to explicit columns used by v1 bundle builder

## PIT Safety

The ingestion layer is PIT-aware where the source surface permits it.

Implemented PIT-safe behavior:

- workflows consume stored snapshots, not live API calls
- trading dates and identifiers are normalized before research use
- raw close is preserved; no future-adjusted close is injected into workflows
- quality proxy uses `fina_indicator.ann_date` with backward-asof alignment

Current limits:

- daily valuation fields from `daily_basic` are treated as same-day vendor
  observations; they are suitable for the v1 value proxy but still inherit the
  vendor publication assumptions
- if financial indicators are unavailable (`fina_indicator_vip` + fallback
  `fina_indicator`), quality inputs degrade gracefully and this is recorded
  explicitly in manifests
- if `suspend_d` or `stk_limit` are unavailable, tradability flags fall back
  conservatively and the missing endpoint is recorded

## Workflow-Compatible Research Inputs

`build_tushare_research_inputs(...)` writes workflow-facing tables under:

```text
data/processed/tushare_research_inputs/<snapshot_name>/
```

Generated files:

- `prices.csv`
- `asset_metadata.csv`
- `market_state.csv`
- `neutralization_exposures.csv`
- `candidate_signals_vqm.csv`
- `manifest.json`
- `tushare_case_single_reversal.json`
- `tushare_case_composite_vqm.json`
- `tushare_case_data_manifest.json`

### v1 Derived Inputs

Neutralization exposures:

- `size_exposure`: `log(total_mv_yuan)`
- `beta_exposure`: rolling 60-day beta to equal-weight market return
- `industry`: from standardized asset metadata

Composite candidate signals:

- `momentum_63d`: existing momentum factor over standardized price table
- `value_book_to_price_proxy`: inverse `pb`
- `quality_profitability_proxy`: PIT-aligned `roe_dt`, fallback to `roe`, then `roa`

The single-factor reversal case does not call Tushare directly. It uses the
generated research tables with the existing reversal factor workflow.

## CLI / Script Usage

Fetch raw snapshots:

```bash
uv run python scripts/tushare_pipeline.py fetch-snapshots \
  --snapshot-name ashare_202401_202412 \
  --start-date 20240101 \
  --end-date 20241231
```

Build standardized tables:

```bash
uv run python scripts/tushare_pipeline.py build-standardized \
  --snapshot-dir data/raw/tushare/ashare_202401_202412
```

Build workflow-compatible case inputs and configs:

```bash
uv run python scripts/tushare_pipeline.py build-cases \
  --standardized-dir data/processed/tushare_standardized/ashare_202401_202412
```

Then run the existing workflows:

```bash
uv run python scripts/run_research_workflow.py run-single-factor \
  --config-path data/processed/tushare_research_inputs/ashare_202401_202412/tushare_case_single_reversal.json \
  --output-dir data/processed/tushare_research_inputs/ashare_202401_202412/single_case_output

uv run python scripts/run_research_workflow.py run-composite \
  --config-path data/processed/tushare_research_inputs/ashare_202401_202412/tushare_case_composite_vqm.json \
  --output-dir data/processed/tushare_research_inputs/ashare_202401_202412/composite_case_output
```

## Permissions and Fallbacks

Expected v1 endpoints:

- required: `trade_cal`, `stock_basic`, `daily`, `daily_basic`
- optional: `adj_factor`, `suspend_d`, `stk_limit`, `fina_indicator`

Fallback behavior is explicit:

- missing optional endpoint does not break extraction
- manifest records unavailable endpoints and per-endpoint extraction metadata
- downstream standardized/build steps keep sections present and populate
  missing values conservatively where possible

Financial endpoint fallback behavior:

- preferred path: `fina_indicator_vip` by quarter `period`
- fallback path: per-stock `fina_indicator` with explicit `ts_code`
- manifest records requested endpoint, actual endpoint used, extraction mode,
  date window/periods, query counts, fallback status, and degradation status
- if both VIP and fallback fail, `fina_indicator` is marked unavailable while
  the rest of the optional pipeline proceeds (single-factor reversal remains runnable)

## Testing

The test suite uses mocks and deterministic fixtures only. No live Tushare
access is required for:

- raw snapshot determinism
- schema mapping and normalization
- missing optional field handling
- bundle-building compatibility with existing research contracts
