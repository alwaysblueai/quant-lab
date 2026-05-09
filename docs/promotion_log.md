# Promotion Log

记录 Tier 2 诊断、helper、notebook 结论是否升级到 Tier 1。

## Candidate Queue

| Date | Candidate | Current Tier | Decision | Notes |
| --- | --- | --- | --- | --- |
| 2026-05-01 | `conditional_ic_by_trailing_return.csv` | Tier 2 | deferred | 先在 `asym_vol_reversal` deepdive 验证判别力；稳定 builder 已有，尚未接 `evaluate.py`。 |
| 2026-05-01 | layout 整体重整 | planning | deferred | 短期采用 option (c)：`configs/real_cases/*` 与 `outputs/real_cases/*` 是 canonical path；撤掉空的 `configs/single_factor/*`，等 `model_factor` 进入 deepdive 后再一次性做并列布局。 |
| 2026-05-01 | `asym_vol_reversal_v1` output path cleanup | path hygiene | done | 唯一写入 `outputs/single_factor/` 的 single-factor case 已迁回 `outputs/real_cases/asym_vol_reversal_v1/`，YAML 与 notebook 已同步。 |
| 2026-05-01 | model_factor 共享 cache 架构修复 | model_factor roadmap | deferred | `dataset_cache.py` 已是 Qlib 风格 content-addressable，但 `web_unified.py:1432` 把 `_root_dir` 包成 `_web_runs/<run_id>/`，让 cache_key 跨 run 失效（每个 web run 复制 ~4.5GB feature 矩阵）。短期靠 `scripts/gc_web_runs.py` 止血；长期把 `_root_dir` 拆成 `outputs/_model_factor_shared_cache/` 共享层 + `_web_runs/<run_id>/` 仅放 run 专属产物。等 model_factor roadmap 5a-(6) 之后统筹。 |

## Promoted

| Date | Item | From | To | Evidence |
| --- | --- | --- | --- | --- |
| 2026-05-01 | `conditional_ic_by_bucket` rank IC + min-assets gate | notebook-local helper | `src/alpha_lab/grouped_evaluation.py` | 所有因子 deepdive 都需要 comparable IC / RankIC bucket stats。 |

## Rejected / Parked

| Date | Item | Reason |
| --- | --- | --- |
