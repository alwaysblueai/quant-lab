# 后端草稿模型流程（Model-Lab Stage 3，v1：spec 变体）

本文定义从网页版 GPT model-lab Stage2 输出到 alpha-lab 后端快速验证之间的标准
流程。目标是让研究态草稿模型可以快速迭代，同时保持可审计、可复现、可对比，
并且不污染前端注册、core API 与 promoted 候选库。

## 上游来源（Stage 0 + Stage 1 + Stage 2）

本流程接的是 `ideas/<idea_id>/stage2_payload.json`（或同等内容）。完整链路：

1. **Stage 0** — `alpha-lab model-idea distribute --idea ... --output-dir ideas/<id>/`
   产出 retrieval pack（vault 卡 + 代码库 custom_models 索引 +
   ModelFactorCaseSpec schema 摘录 + validator 硬约束清单）+
   两份对称 engine prompts：`prompt_claude.md` 和 `prompt_codex.md`。
2. **Stage 1** — Claude Code 与 Codex GUI 各自输出
   `stage1_claude.md` / `stage1_codex.md`，两份都包含 generator + reviewer。
3. **Stage 2** — 网页版 GPT 接 `stage2_input.md` + reconcile + candidate 模板，输出唯一
   `model_candidate_payload`，含 `provenance.idea_id` /
   `provenance.stage2_payload_sha256` / `provenance.audience_chain`。

详见 `docs/research_workflow.md`。

网页版 GPT 项目中建议固定加入以下来源文件，确保输出从 Stage1 到 Stage3 都是
“人可读 + 机器可提取”的固定合同：

- `docs/templates/model_lab_source_pack.md`
- `docs/templates/model_lab_stage1_reconcile_contract.md`（已加 `code_feasibility_review` 输入槽位）
- `docs/templates/model_lab_stage2_candidate_contract.md`

当用户提供 Claude Code / Codex GUI 两份 Stage1 讨论结果时，网页版 GPT 必须先
输出 `contract_version=model_stage1_reconcile_v1` 的
`model_stage1_reconcile_payload`。只输出 candidate 名字、中文名、slug 或自然语言
建议，均视为不合格。

当用户要求进入可执行草稿时，网页版 GPT 必须输出
`contract_version=model_stage2_candidate_output_v1`，其中包含唯一的
`model_candidate_payload` 和完整 `case_spec_payload`。Stage3 只接受这个
`model_candidate_payload` 作为机器事实。

v1 只支持**spec 变体型草稿模型**：通过现有 `ModelFactorCaseSpec` 字段
（`feature_columns` / `feature_preprocess` / `model.family` / `training` /
`feature_availability` 等）做参数变体；不允许引入自定义 feature builder code
或自定义 estimator code。

## 核心边界

后端草稿模型不是正式模型，只能存在于：

- `custom_models/research/<candidate>/model_candidate.json`
- `custom_models/research/<candidate>/research_log.md`
- `configs/real_cases/model_factor/<candidate>_vN.yaml`
- 标准 pipeline 输出目录

不得为了某个候选新增一次性脚本。所有实验都必须通过：

```bash
alpha-lab real-case model-factor run <case.yaml>
```

运行 case 之前必须先通过草稿模型门禁：

```bash
alpha-lab validate-draft-model custom_models/research/<candidate>/model_candidate.json
```

## 输入契约

网页版 GPT 的 Stage2 输出必须包含可机器提取的 `model_candidate_payload`，
其中关键字段是完整可写入的 `case_spec_payload`。

```yaml
contract_version: "stage2_model_candidate_v1"
candidate_name: "<snake_case>"
implementation_status: "draft_for_stage3"
implementation_type: "spec_variant"
source_mechanisms: []
base_case_spec_path: ""
expected_horizon: "t_plus_1_or_later"
data_contract: { ... }
risk_controls: { ... }
run_controls:
  evaluation_profile: "exploratory_screening"
  screening_retrain_every_n_dates: 40
  vault_export_mode: "skip"
case_spec_payload: { ... 完整 ModelFactorCaseSpec 字段 ... }
stage3_validation_focus: []
provenance:
  idea_id: "20260511T143000Z__turnover-conditioned-pv"
  stage2_payload_sha256: "<sha256 of canonical-JSON Stage2 payload>"
  audience_chain: ["claude", "codex", "web_gpt_stage2"]
```

`case_spec_payload` 不允许只给 patch；网页 GPT 必须输出完整 case spec，避免
Codex GUI 自行猜测如何 merge。

`provenance` 块由 Stage 2 网页 GPT 填写，Stage 3 不得改写或删除。
validator 会做形态校验，artifact 审计块会把 `provenance.idea_id` 复制进
`draft_model_source.provenance`。

### `stage2_payload_sha256` 物化约定（Stage 2 placeholder → Stage 3 真实 sha）

Stage 2 网页 GPT 无法自引用它正在书写的 payload 的哈希，所以
`provenance.stage2_payload_sha256` 在 Stage2 产物里是 **placeholder**
（例如 `PENDING_STAGE3_MATERIALIZE`）。Stage 3 materialize 时**必须**按以下唯一
约定把它替换成真实 64 位 hex sha，否则 `validate-draft-model` 会以
`provenance_payload_sha256`（非 64 位 hex）报错：

1. 取 `model_candidate_payload`，把 `provenance.stage2_payload_sha256` 置为空串 `""`。
2. `json.dumps(payload, ensure_ascii=False, sort_keys=True)` 得到 canonical JSON。
3. 对其 UTF-8 字节取 `sha256` 的 hexdigest，写回 `provenance.stage2_payload_sha256`。

该约定的可执行规范见
`tests/test_draft_model_validation.py::test_stage2_payload_sha_materialization_is_deterministic_and_hex`
与 `::test_materialized_candidate_passes_validator_but_placeholder_fails`。所有
执行端（Codex GUI / Web Model-Lab / Claude Code）必须用同一约定，保证同一 payload
产出同一 sha。

## Codex GUI 执行流

1. 使用 `docs/templates/codex_gui_model_stage3_execution_envelope.md` 作为任务入口。
2. 提取 Stage2 输出中的 `model_candidate_payload`。
3. 写入 `custom_models/research/<candidate>/model_candidate.json`。
4. 创建或更新 `configs/real_cases/model_factor/<candidate>_vN.yaml`，字段必须与 `case_spec_payload` 完全一致。
5. 运行 `alpha-lab validate-draft-model ...`。
6. validator 通过后运行 `alpha-lab real-case model-factor run ... --draft-model-candidate ...`。
7. 读取 artifact 并确认 hash 审计字段。
8. 用中文总结初筛结果和下一轮 case_spec 字段调整方向。

## Web UI 执行流

Model Lab Web UI 的定位是**成熟候选的完整报告工作台**，不是第一轮 idea
exploration 或快速试错入口。模糊 idea、草稿模型初筛、失败归因和下一轮
`case_spec_payload` 调整，默认由 Codex GUI 在后端完成；用户再把后端实验报告
交给网页版 GPT 产出下一版方案。

前端只在候选经过若干轮后端迭代、已经值得生成完整可视化报告时使用。入口是
`/model-lab` 页面中的 `Draft Candidates` 面板。UI 只做编排，不重新实现
validator、case parser 或 model-factor pipeline。

标准操作顺序：

1. 在 `model_candidate_payload` 文本框粘贴 Stage2 输出。
2. 点击主按钮 `导入并运行完整报告`，UI 固定执行
   save -> validate -> materialize -> run；run 默认使用 `default_research`、
   `vault_export_mode=skip`，并把 `--draft-model-candidate` 传入标准 CLI。
   快速初筛仍应在 Codex GUI 后端显式使用 `exploratory_screening` 命令完成。
3. `保存 Candidate`、`Validate`、`生成 Case YAML` 保留在
   `Advanced Candidate Actions` 中，仅用于调试或分步排错。
4. Run 队列中会显示 `draft:<candidate>` badge；artifact viewer 会从
   `run_manifest.json`、`model_definition.json`、`feature_manifest.json` 中展示
   `candidate_json_sha256`、`case_spec_sha256`、`feature_contract_sha256` 的短 hash。

Web UI 不提供 idea explorer 主入口、promoted、前端正式注册、自定义 feature
code 或自定义 estimator code 入口。若 artifact 缺少 `draft_model_source`，本轮
Stage3 仍视为流程失败。

## Validator 门禁

`validate-draft-model` 至少检查：

- `model_candidate.json` 是合法 JSON object。
- `contract_version=stage2_model_candidate_v1`、
  `implementation_status=draft_for_stage3`、`implementation_type=spec_variant`。
- `candidate_name` 是英文 snake_case（3-64）。
- 文件位于 `custom_models/research/<candidate>/model_candidate.json`。
- `case_spec_payload` 通过 `model_factor_case_spec_from_mapping` 解析无错（覆盖 `model.family`、`feature_columns` 不重名/不覆盖保留列、`feature_availability` PIT 合同、`training` 与 `model_selection` 字段一致性等）。
- features 文件可读，且 `feature_columns` 全部出现在文件表头中。
- 疑似基本面特征不允许 `feature_availability.mode='required_timestamp'` 且
  `column=null`（需要 known_at 时间戳或 `safety_lag_days >= 1`）。
- 整个 payload 不出现 Level 3 / execution_replay / fill_simulation /
  portfolio_construction / live_trading 关键词。
- 输出 `candidate_json_sha256`、`case_spec_sha256`、`feature_contract_sha256`，
  便于后续 artifact 对照。

validator 不写 case YAML，不跑训练，不修改 features/prices 文件，仅做只读审计。

## 标准运行命令

快速初筛（推荐用于每一轮）：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync --frozen alpha-lab validate-draft-model \
  custom_models/research/<candidate>/model_candidate.json
UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync --frozen alpha-lab real-case model-factor run \
  configs/real_cases/model_factor/<candidate>_v1.yaml \
  --evaluation-profile exploratory_screening \
  --screening-retrain-every-n-dates 40 \
  --render-report \
  --vault-export-mode skip \
  --draft-model-candidate custom_models/research/<candidate>/model_candidate.json
```

更完整验证（在初筛通过后才跑）：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync --frozen alpha-lab real-case model-factor run \
  configs/real_cases/model_factor/<candidate>_v1.yaml \
  --evaluation-profile default_research \
  --render-report \
  --vault-export-mode skip \
  --draft-model-candidate custom_models/research/<candidate>/model_candidate.json
```

## Artifact 审计

运行完成后，必须确认以下文件均包含 `draft_model_source` 审计块：

- `run_manifest.json`
- `model_definition.json`
- `feature_manifest.json`

最小审计字段：

```json
{
  "draft_model_source": {
    "name": "<candidate>",
    "scope": "research",
    "path": "custom_models/research/<candidate>/model_candidate.json",
    "candidate_json_sha256": "...",
    "case_spec_sha256": "...",
    "feature_contract_sha256": "...",
    "contract_version": "stage2_model_candidate_v1",
    "implementation_status": "draft_for_stage3",
    "implementation_type": "spec_variant",
    "factor_name": "...",
    "feature_columns": ["..."],
    "feature_availability": { "mode": "...", "column": "..." },
    "model_family": "...",
    "provenance": {
      "idea_id": "20260511T143000Z__turnover-conditioned-pv",
      "stage2_payload_sha256": "...",
      "audience_chain": ["claude", "codex", "web_gpt_stage2"]
    }
  }
}
```

缺少 `candidate_json_sha256`、`case_spec_sha256`、`feature_contract_sha256` 或
source path 时，本轮 Stage3 视为失败。`provenance` 缺失仅产生 warning（兼容旧
candidate），但所有走 `alpha-lab model-idea distribute` 流的候选必须有 provenance。

## 后端契约自动化

`alpha-lab real-case model-factor run` 在收尾时会自动检测本轮是否走的是
`custom_models/research/<candidate>/model_candidate.json`，若是则按
`docs/backend_run_contract.md` 中 `backend_run_contract_v1` 跑完整收尾：

- 写 `comparison_summary.json`、`backend_run_receipt.json`
- 把 `backend_run_contract.{contract_version,status,issue_count}` 与 sidecar
  路径回挂到 `run_manifest.json`
- audit 同时校验 `draft_model_source` 在 `run_manifest.json`、
  `model_definition.json`、`feature_manifest.json` 中的 hash 一致性
- audit 失败时仍写出两份 sidecar（`status=failed` + 完整 `issues`），并
  以非零退出码返回；`run-batch` 模式下 `worst_rc` 表示整批是否全部通过

非 research draft run（普通 case YAML、promoted candidate）不会触发契约收尾，
行为与历史一致。

## 结果分析

每轮运行后，Codex GUI 必须读取并总结：

- `metrics.json`
- `summary.md`
- `integrity_report.md`
- `model_definition.json`
- `feature_manifest.json`
- `run_manifest.json`

分析重点：

- 是否有 PIT / temporal alignment / split contract 硬失败。
- coverage、IC、rank IC、decay、turnover、cost-aware 初筛是否支持继续。
- 训练样本量、训练通过率、`model_selection` 是否启用、是否过拟合。
- feature 重要性是否落在合理少数列上，还是被某个 noise feature 主导。
- 下一轮只允许在 `case_spec_payload` 字段层面做调整：`feature_columns`、
  `feature_preprocess`、`model.family`、`training.window_type` /
  `train_window_n_dates` / `retrain_every_n_dates`、`target.horizon`、
  `feature_availability` 等。

## Stage0 → Stage3 验收清单

一轮 model-lab 后端草稿实验**通过**当且仅当下列各项全部满足（任一不满足即本轮失败，
按对应阶段停下并报告，不在合同外改写）：

**Stage 0（idea 分发）**
- [ ] `alpha-lab model-idea distribute` 产出 `ideas/<idea_id>/` 下 5 个文件：
      `manifest.json`、`retrieval_pack.md`、`prompt_claude.md`、`prompt_codex.md`、`stage2_input.md`。
- [ ] `manifest.json::idea_id` 与 `retrieval_pack.md` 中 idea_id 一致。

**Stage 1（双引擎对称输出）**
- [ ] `stage1_claude.md` / `stage1_codex.md` 各含 4 行 header + Part A 机制候选 + Part B 可执行性评审。
- [ ] 每条机制有 `implementation_status` 与 `required_columns_present/missing`、`validator_blockers`。
- [ ] 未注册 feature/column 不作为 v1 required input；`needs_extension` 仅作上下文。

**Stage 2（reconcile + candidate）**
- [ ] `stage1_reconcile.yaml` = `model_stage1_reconcile_v1`；`stage2_payload_v1.yaml` = `model_stage2_candidate_output_v1`。
- [ ] `model_candidate_payload` 内层 `stage2_model_candidate_v1` + `draft_for_stage3` + `spec_variant` + 完整 `case_spec_payload`。
- [ ] 三处 `provenance.idea_id`（stage1 / stage2 / candidate）与 `manifest.json::idea_id` 一致。
- [ ] `save_model_factor_stage2_intake` 返回 `ok=True`（placeholder sha 仅 warning）。

**Stage 3（materialize + validate + run）**
- [ ] 只写 `custom_models/research/<cand>/model_candidate.json`、`research_log.md`、`configs/real_cases/model_factor/<cand>_v1.yaml`。
- [ ] `stage2_payload_sha256` 已按上文约定物化成真实 64 位 hex。
- [ ] `validate-draft-model` 通过，输出 `candidate_json_sha256` / `case_spec_sha256` / `feature_contract_sha256`。
- [ ] `real-case model-factor run … --draft-model-candidate …` 退出码 0（或预算失败时写出标准 failed receipt）。

**收尾审计**
- [ ] `backend_run_receipt.json` status=`success`、`run_manifest.json.backend_run_contract.status=passed`、`comparison_summary.json` 写出。
- [ ] `draft_model_source` 在 `run_manifest.json` / `model_definition.json` / `feature_manifest.json` 三处齐全，且三处 hash 与 validator 输出一致。
- [ ] `resource_usage.json` 含 `peak_rss_mb` / `stage_rss_mb` / `max_rss_mb_budget`。
- [ ] model-factor 全部额外必有 artifact 写出（见 `docs/backend_run_contract.md`）；model 线默认**不**产 `artifact_tiers`，由 `model_diagnostic_artifacts` 承载。

该清单的端到端固化测试是
`tests/test_backend_run_contract.py::test_model_factor_draft_backend_contract_e2e`
与 `tests/test_draft_model_validation.py` 中的 Stage2→Stage3 seam 测试。

## 晋升边界

草稿模型通过 `default_research` 后仍然不自动晋升。晋升必须另走流程：

1. 满足 promotion thresholds（与 single-factor 等价的标准评价）。
2. 复制到 `custom_models/promoted/<candidate>/`。
3. 写 promotion 卡片。
4. 更新 promotion 日志。
5. 再通过前端跑完整图文报告归档。

前端只处理已通过后端标准评价的候选，不承担探索期试错。v1 不会启用任何
自定义 feature builder code 或 estimator code 的晋升路径。
