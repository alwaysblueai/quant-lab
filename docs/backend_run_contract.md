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
