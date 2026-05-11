# Intraday-augmented factor workflow

This document is the canonical recipe for running a single-factor backtest on
this machine when the factor consumes any combination of:

- **daily price-volume columns** (canonical: ``open / high / low / close /
  pre_close / volume / amount / vwap / adj_factor / up_limit / down_limit /
  is_limit_up / is_limit_down / is_suspended / is_st``);
- **intraday-derived daily columns** (the 60+ features written by
  ``scripts/etl/build_intraday_features.py`` — ``signed_jump``, ``rv_5m``,
  ``vwap_close_dev``, ``amount_hhi``, status flags, etc.; full list in
  ``docs/intraday_etl_contract.md``).

It exists because the joined dataset under
``data/processed/real_case_inputs/ashare_institutional_intraday_v1/`` is
**101 cols × 7.5M rows × 3.1 GB on disk**. Loading it whole into pandas peaks
at 12–18 GB — well above the WSL2 user-cgroup memory budget on this laptop —
and ``--evaluation-profile default_research`` adds 9 diagnostics
(``run_param_sensitivity``, ``run_lag_sensitivity``,
``run_execution_price_sensitivity``, ``compute_capacity_estimation``,
``compute_factor_autocorrelation``, ``compute_conditional_ic``,
``run_marginal_contribution``, ``run_tradability_checks``,
``run_neutralization_raw_comparison``) that each duplicate parts of the
working set.

## TL;DR

```bash
# 1. one-shot helper: builds a slim slice + precomputed factor parquet
python scripts/etl/build_factor_run_inputs.py \
    --factor signed_jump_neg_5d \
    --start-date 2020-01-01

# 2. wire the printed paths into the case YAML (factor_input.mode: file)
# 3. run
alpha-lab real-case single-factor run \
    configs/real_cases/single_factor/signed_jump_neg_5d_v1.yaml \
    --evaluation-profile default_research \
    --render-report
```

## Why ``mode: file`` instead of ``mode: recipe``

``factor_recipe`` runs ``build_factor`` inside the pipeline process, after
``_normalize_prices`` strips the prices frame. With the joined dataset that
strip step alone forces all 101 cols into RAM. ``mode: file`` precomputes the
factor in a one-off process (which can use a slim slice), then ships a
``(date, asset, factor, value)`` parquet to the pipeline — typically <100 MB.

The pipeline still loads ``prices_path``, but if you point that at a slim
slice it stays ~3 GB peak instead of ~15 GB.

The trade-off: every parameter change (window, min_periods, etc.) requires
re-running the precompute. For iteration on parameters, keep ``mode: recipe``
on a *small* prices slice; switch to ``mode: file`` once parameters are
locked and you want ``default_research``.

## Layered optimisation cheatsheet

| Pressure point | Lever | Expected pandas peak |
|---|---|---|
| Full joined dataset (101 cols, 7.5M rows) | none | 12–18 GB → OOM |
| Slim slice (17 base cols + factor's intraday cols only) | ``build_factor_run_inputs.py`` | ~3–4 GB |
| Slim slice + ``factor_input.mode: file`` | precompute factor, drop recipe layer | ~2–3 GB |
| Above + 4-year window (e.g. 2020-01-01 to 2026-04-15) | ``--start-date`` | ~1.5–2 GB |
| Above + ``default_research`` diagnostics | (no lever; budget for +5–8 GB peak) | ~6–10 GB |

For ``default_research``: aim for the bottom half of the table.
``exploratory_screening`` is fine on the third row.

## What the helper does

``scripts/etl/build_factor_run_inputs.py`` reads
``custom_factors/research/<factor>/factor.json``, then:

1. Diffs ``required_columns`` against ``BASE_PRICE_COLUMNS`` to find the
   intraday-derived columns the factor actually needs.
2. Reads the joined dataset with ``pyarrow`` column projection — never
   loads the unused 60+ intraday cols into pandas.
3. Optionally clips by date.
4. Writes ``data/processed/real_case_inputs/<dataset_name>/prices.parquet``
   plus a slice manifest, and symlinks the universe file.
5. Compiles ``build_factor`` from the same factor.json, runs it on the slim
   slice, and writes ``custom_factors/research/<factor>/factor_<factor>_<suffix>.parquet``.

The printed paths are ready to paste into a case YAML.

## Recommended WSL2 budget

Default WSL2 memory on a 32 GB host is half the host (~16 GB), and systemd's
PSI-based killer trips well before that. Edit
``C:\Users\<you>\.wslconfig`` to:

```
[wsl2]
memory=24GB
processors=8
swap=16GB
```

Then ``wsl --shutdown`` from PowerShell. With 24 GB + 16 GB swap, the slim
slice + ``default_research`` profile combo has comfortable headroom for the
diagnostics that duplicate (date × asset) matrices.

## When ``mode: recipe`` is still OK

If the factor only consumes the 17 base cols (no intraday-derived input),
``mode: recipe`` on the joined dataset is fine — the recipe normaliser
already drops the wide intraday columns before the builder runs, so the
peak is governed by the base-cols slice anyway.

For intraday-augmented factors, always go through this helper.
