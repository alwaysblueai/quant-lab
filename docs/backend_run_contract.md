# 后端实验运行契约

本文档定义 `backend_run_contract_v1`。它是 Codex GUI 后端实验的共同合同，
覆盖 single-factor 与 model-factor 两条 Level 1/2 流程。目标是让每轮研究从
draft 到 artifact audit 都有固定路径、固定命令、固定产物和固定失败条件。

本契约不引入 Level 3 语义；不讨论回放、成交、适配器一致性或线上执行。

## 适用范围

- single-factor draft：`custom_factors/research/<factor>/factor.json`
- model-factor draft：`custom_models/research/<candidate>/model_candidate.json`
- case YAML：`configs/real_cases/single_factor/*.yaml` 或
  `configs/real_cases/model_factor/*.yaml`
- 标准运行入口：
  - `alpha-lab real-case single-factor run <case.yaml>`
  - `alpha-lab real-case model-factor run <case.yaml> --draft-model-candidate <candidate.json>`

## 标准生命周期

每次后端实验必须按以下顺序执行：

1. `draft/source payload`：只接受合同化的 `factor_json_payload` 或
   `model_candidate_payload`。
2. `validate`：先运行 `validate-draft-factor` 或 `validate-draft-model`。
3. `materialize case`：case YAML 必须落在对应 `configs/real_cases/...` 目录。
   model 侧可从 `case_spec_payload` 自动 materialize；factor 侧必须显式提供
   case YAML，因为 factor payload 不包含价格、股票池、目标和输出目录合同。
4. `standard run`：只调用标准 `real-case` pipeline，不写临时脚本。
5. `artifact audit`：检查稳定 artifact、source hash、manifest outputs 与报告产物。
6. `archive/compare`：每轮本地输出 `experiment_card.md` 与
   `comparison_summary.json`；正式写入 vault 仍走 `alpha-lab archive ...`。
7. `receipt`：写入 `backend_run_receipt.json`，记录本轮 validator、case、run、audit。

## 稳定产物

共同必有产物：

- `run_manifest.json`
- `metrics.json`
- `summary.md`
- `integrity_report.json`
- `integrity_report.md`
- `signal_validation.json`
- `portfolio_recipe.json`
- `backtest_result.json`
- `research_tearsheet.json`
- `research_tearsheet.pdf`
- `experiment_card.md`
- `case_report.md`
- `level2_portfolio_validation/portfolio_validation_summary.json`
- `level2_portfolio_validation/portfolio_validation_metrics.json`
- `level2_portfolio_validation/portfolio_validation_package.json`
- `level2_portfolio_validation/portfolio_validation_package.md`
- `comparison_summary.json`
- `backend_run_receipt.json`

`run_manifest.json.artifact_tiers` 对重型明细产物使用结构化对象，至少说明
`tier`、`is_complete`、`row_count`、`source_row_count`、`omitted_row_count`、
`sampling_policy` 和 `reason`。前端必须据此区分完整产物与
`exploratory_screening` 下的采样产物，不能把 sampled holdings/weights 当作
完整 holdings/weights 展示。

### `artifact_tiers` vs `model_diagnostic_artifacts`（两条线不共用同一机制）

这两个 manifest 块解决的是不同问题，**不要混用**，前端必须按 `workflow` 选对块读：

| 维度 | `artifact_tiers`（single-factor 线） | `model_diagnostic_artifacts`（model-factor 线） |
| --- | --- | --- |
| 解决的问题 | 重型明细产物被**行采样** | profile 是否**跳过**某诊断产物的计算 |
| 典型产物 | holdings / weights（`sampled_extreme_quantiles` / `sampled_nonzero_weights`） | `feature_oos_ic`（`exploratory_screening` 下 `compute_feature_oos_ic=False`） |
| 条目语义 | `tier` / `is_complete` / `row_count` / `sampling_policy` | `contract_status`（`emitted_v1` / `not_emitted_v1`）/ `emitted` / `evaluation_profile` / `reason` |
| 前端误读风险 | 把 sampled 行当完整行展示 | 把 profile 抑制的空文件当作"零 IC / 零信号" |
| 由谁写 | `real_cases/single_factor/artifacts.py` | `artifact_contracts.build_model_diagnostic_artifact_status`（`real_cases/model_factor/artifacts/core.py`） |

判定规则：

- **model-factor 默认不产 `artifact_tiers`**。model 线的评估明细（IC、turnover、group returns、coverage 等）都是**完整、非行采样**产物，没有 holdings/weights 行采样语义，所以 manifest 不会出现 `artifact_tiers` 块——这是符合契约的缺失，不是 bug。model 线统一用 `model_diagnostic_artifacts` 表达"哪些诊断被 profile 抑制"。
- **single-factor 才用 `artifact_tiers`**，因为它会在 `exploratory_screening` 下对 holdings/weights 做行采样。
- 只有当 model 线将来真的产出**被行采样的重型明细产物**（例如某条 model 候选触发了会采样 holdings/weights 的 Level 2 组合验证）时，才需要为 model 线补 `artifact_tiers`；在此之前，`model_diagnostic_artifacts` 即可承载全部需求。
- 前端读取建议：`workflow=single_factor` → 读 `artifact_tiers`；`workflow=model_factor` → 读 `model_diagnostic_artifacts`，并对 `contract_status=not_emitted_v1` 的诊断显示"该 profile 未计算"，而非数值 0。

single-factor 额外必有：

- `factor_definition.json`
- `factor_definition.yaml`

若运行使用 research custom factor，`run_manifest.json` 与
`factor_definition.json` 必须包含 `custom_factor_source`，至少包括：

- `path`
- `code_sha256`
- `factor_json_sha256`
- `required_columns`
- `frequency`
- `unavailable_data_policy`
- `pit_assumption`

model-factor 额外必有：

- `factor_definition.json`
- `model_factor_definition.yaml`
- `model_definition.json`
- `feature_manifest.json`
- `model_selection.json`
- `diagnostics.json`
- `training_log.csv`
- `training_metrics.csv`
- `feature_importance.csv`
- `feature_importance_ledger.csv`
- `feature_oos_ic.csv`

若运行使用 research draft model，`run_manifest.json`、`model_definition.json`
与 `feature_manifest.json` 必须包含 `draft_model_source`，至少包括：

- `path`
- `candidate_json_sha256`
- `case_spec_sha256`
- `feature_contract_sha256`
- `contract_version`
- `implementation_status`
- `implementation_type`
- `feature_columns`
- `model_family`

## 契约执行点

契约的强制点**不是**一条独立 CLI 子命令，而是现有 `real-case` pipeline 的
收尾阶段。`real-case single-factor run` / `real-case model-factor run`
完成后，按以下顺序判断与执行：

1. 调用 `detect_research_draft_run(output_dir, workflow=...)`。判定依据是
   `run_manifest.json` 中是否带有完整的 `custom_factor_source` /
   `draft_model_source` 块，且 `path` 落在 `custom_factors/research/` /
   `custom_models/research/`。
2. 命中 → 调用 `finalize_backend_contract(...)`，它**在内部**严格按顺序：
   1. `write_comparison_summary(...)` → 写 `comparison_summary.json`
   2. `audit_backend_run_artifacts(...)` → 校验稳定 artifact、Level 1/2 contract、
      `custom_factor_source` / `draft_model_source` 在所有应出现位置的 hash 一致
   3. `build_backend_run_receipt(...)` 内嵌 audit 结果
   4. `write_backend_run_receipt(...)` → 写 `backend_run_receipt.json`
   5. `attach_backend_contract_to_manifest(...)` → 把两个 sidecar 路径与
      `backend_run_contract.{contract_version,status,issue_count}` 回挂到
      `run_manifest.json`
3. 未命中（非 research draft run）→ 不写 sidecar，不改 manifest。契约对老调用
   完全透明，不会对非 draft run 强加新产物。

audit 失败时，`comparison_summary.json` 与 `backend_run_receipt.json` 仍会
写出（`status=failed` + 完整 `issues`），并且 `real-case` 命令以非零退出码
返回。意图：Codex GUI 与 Web 两边都能"看到 sidecar → 看 issues → 改
draft"，而不是 silently 跑过。

## 失败条件

以下任一情况本轮视为失败：

- draft validator 返回错误。
- case YAML 不在对应 `configs/real_cases/...` 目录。
- 标准 pipeline 运行失败。
- 必有 artifact 缺失。
- source hash audit 缺少路径或 hash 字段。
- factor 的 `custom_factor_source` 与 `factor.json` hash 不一致。
- model 的 `draft_model_source` 与 `model_candidate.json` hash 不一致。
- `case_report.md`、`comparison_summary.json` 或 `backend_run_receipt.json` 缺失。

## 允许的非失败提示

以下 warning 不代表契约失败：

- 非 promoted draft 因子默认不跑完整 Level 2 portfolio validation；只要稳定
  Level 2 artifact 仍按合同写出，contract audit 可以继续通过。
- 当本地或 CI 没有 Playwright/Chromium 时，`research_tearsheet.pdf` 会使用
  matplotlib fallback 生成。它满足 artifact contract；需要高保真 PDF 时再安装
  Playwright/Chromium。

## 内存预算与 resource_usage.json

`real-case` pipeline 内置一个**软**内存 guard（`RunMemoryMonitor`）：在阶段边界
（`run_start` / `load_inputs` / `evaluate` / `artifacts_exported`）采样进程 RSS，
当 `ALPHA_LAB_MAX_RSS_MB` 设了预算且 peak 超预算时抛
`AlphaLabMemoryError`，让 run 以可审计、可归因的方式失败，而不是被 OS OOM-killer
静默杀掉。

- **成功 run**：在 `artifacts_exported` 阶段写出 `resource_usage.json`
  （`peak_rss_mb`、按阶段的 `stage_rss_mb`、`max_rss_mb_budget`）。
- **预算失败 run**：run 在导出阶段前中止，但 CLI 的失败收尾仍会写出标准 sidecar
  （`backend_run_receipt.json` status=`failed` + `validation.errors` 含
  `memory_budget_exceeded`、`comparison_summary.json`、`run_manifest.json` 的
  `backend_run_contract.status=failed`），并以非零退出码返回。此时同样写出
  `resource_usage.json`（快照从 `AlphaLabMemoryError` 中恢复），使前端无论 run
  成功还是因预算失败，都能在同一个文件里读取 peak/stage RSS。
- `resource_usage.json` 是**遥测**产物：RSS 读数非确定，不属于必有/golden 比较
  artifact，audit 不因其缺失而失败。

**软 guard 边界**：采样发生在阶段边界，单个阶段内的瞬时大分配仍可能在下一次采样
前冲破 OS 硬限并被 OOM-killer 杀掉——这种情况下不会有 receipt。因此预算应设在
宿主硬限之下（例如 ~18-19GB 的 WSL 机器设 14000-15000），并对 intraday 宽表
采用更保守的预算。诊断硬 OOM 见 `docs/runtime_stability_runbook.md`。

## 与前端关系

前端不承担探索期试错。前端只展示已完成标准后端运行、通过 artifact audit、
并且值得比较或归档的候选。Web 后端不重新实现 validator、case parser 或
pipeline——它走的就是同一条 `real-case` pipeline，因此自动继承契约执行点。

Web 后端的标准行为：

- model-lab draft candidate 通过 `save -> validate -> materialize-spec -> standard run`
  提交到 run store。run 成功后，因走的是同一份 `real-case` 收尾，契约会自动
  在 run 目录里产出 `comparison_summary.json` 与 `backend_run_receipt.json`，
  并登记到 `run_manifest.json.outputs`。
- single-factor Web run 同理：如果 manifest 带有完整的 `custom_factor_source`
  且 source path 指向 `custom_factors/research/<factor>/factor.json`，
  契约自动触发；否则保持普通 Level 1/2 run 行为。
- Web 端 `runs/compare` 直接读取 manifest 中的 `backend_run_contract` 块，
  在 metric row 中暴露 `backend_contract_status`、`backend_artifact_audit_ok`、
  `backend_contract_issue_count`，用于候选版本比较时先看 audit 状态。
- Web 端 experiment-card 导出会将 `backend_run_receipt.json` 与
  `comparison_summary.json` 作为可选 sidecar 一并写入 vault 的 `50_experiments/<case>/`。
