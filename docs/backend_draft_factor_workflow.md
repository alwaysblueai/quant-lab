# 后端草稿因子流程

本文定义从网页版 GPT Stage 2 输出到 alpha-lab 后端快速验证之间的标准流程。
目标是让研究态因子可以快速迭代，同时保持可审计、可复现、可对比，并且不污染前端注册与 core API。

## 上游来源（Stage 0 + Stage 1 + Stage 2）

本流程接的是 `ideas/<idea_id>/stage2_payload.json`（或同等内容），上游链路：

1. **Stage 0** — `alpha-lab idea distribute --idea ... --output-dir ideas/<id>/`
   产出 retrieval pack + 两份对称 engine prompts：`prompt_claude.md`
   和 `prompt_codex.md`。
2. **Stage 1** — Claude Code 与 Codex GUI 各自输出
   `stage1_claude.md` / `stage1_codex.md`，两份都包含 generator + reviewer。
3. **Stage 2** — 网页版 GPT 接 Stage 2 输入模板（`stage2_input.md`）、
   reconcile 合同（`docs/templates/single_factor_stage1_reconcile_contract.md`）
   + Stage 2 candidate 模板（`docs/templates/single_factor_stage2_candidate_contract.md`），
   输出唯一 `factor_json_payload`，含 `provenance.idea_id` /
   `provenance.stage2_payload_sha256` / `provenance.audience_chain`。

详见 `docs/research_workflow.md`。

## 核心边界

后端草稿因子不是正式因子，只能存在于：

- `custom_factors/research/<factor>/factor.json`（含 `provenance` 块）
- `custom_factors/research/<factor>/research_log.md`
- `configs/real_cases/single_factor/<factor>_vN.yaml`
- 标准 pipeline 输出目录
- `ideas/<idea_id>/stage3_runs/`（可选：写入 artifact 软链或摘要，便于跨轮迭代追溯）

不得为了某个因子新增一次性脚本。所有实验都必须通过：

```bash
alpha-lab real-case single-factor run <case.yaml>
```

运行 case 之前必须先通过草稿因子门禁：

```bash
alpha-lab validate-draft-factor custom_factors/research/<factor>/factor.json
```

## 输入契约

网页版 GPT 的 Stage2 输出必须包含可机器提取的 `factor_json_payload`。
Codex GUI 只把这个 JSON 作为落盘事实来源。

```json
{
  "name": "example_factor",
  "description": "中文机制说明",
  "required_columns": ["close", "volume"],
  "optional_columns": ["amount"],
  "frequency": "daily",
  "unavailable_data_policy": "return_nan",
  "pit_assumption": "所有 rolling 特征只使用当前及过去 bar，不使用未来收益标签。",
  "code": "def build_factor(frame): ...",
  "provenance": {
    "idea_id": "20260511T143000Z__signed-jump-reversal",
    "stage2_payload_sha256": "<sha256 of canonical-JSON Stage2 payload>",
    "audience_chain": ["claude", "codex", "web_gpt_stage2"]
  }
}
```

`provenance` 块由 Stage 2 网页 GPT 填写，Stage 3 不得改写或删除。
validator 会做形态校验，artifact 审计块会把 `provenance.idea_id` 复制进
`custom_factor_source.provenance`。

推荐接口是：

```python
def build_factor(frame):
    ...
    return value
```

其中 `value` 必须是与输入 index 对齐的 `pd.Series`。旧的
`builder(prices, ...) -> DataFrame(date, asset, value)` 接口仍可兼容，但新草稿优先使用
`build_factor(frame)`。

## Codex GUI 执行流

1. 使用 `docs/templates/codex_gui_stage3_execution_envelope.md` 作为任务入口。
2. 提取 Stage2 输出中的 `factor_json_payload`。
3. 写入 `custom_factors/research/<factor>/factor.json`。
4. 更新或创建 `configs/real_cases/single_factor/<factor>_vN.yaml`，使用 recipe base method 指向该 factor。
5. 在 case YAML 根字段写入 `project_slug`、`archive_identity`、`evaluation_profile`；`project_slug` 只能来自明确项目上下文，不能从 factor 名、路径或文件名猜测。
6. 覆盖已有 case YAML 前先读取旧 YAML，并保留已有 `project_slug` / `archive_identity` / `evaluation_profile`，除非用户明确要求修改。
7. 运行 `alpha-lab validate-draft-factor ...`。
8. validator 通过后运行 `alpha-lab real-case single-factor run ...`。
9. 读取 artifact 并确认 hash 审计字段。
10. 用中文总结初筛结果和下一轮改进方向。

## Validator 门禁

`validate-draft-factor` 至少检查：

- `factor.json` 是合法 JSON object。
- `name` 是英文 snake_case，且不覆盖内置因子名。
- 文件位于 `custom_factors/research/<factor>/factor.json`。
- 必填字段存在：`name`、`description`、`required_columns`、`optional_columns`、`frequency`、`unavailable_data_policy`、`pit_assumption`、`code`。
- `frequency` 为 `daily`。
- `code` 定义 `build_factor(frame)` 或兼容旧接口 `builder(prices, ...)`。
- 代码不包含文件读写、网络访问、`subprocess`、`eval`、`exec`。
- 明显 future leakage 模式被拦截，例如 `shift(-1)`、负向 `pct_change`、feature 中出现 future label 命名。
- rolling / expanding 逻辑必须能看到按 `asset` 分组的实现线索。
- toy frame 上可以运行，并返回标准输出。
- 输出 `code_sha256` 与 `factor_json_sha256`，便于后续 artifact 对照。

## 标准运行命令

快速初筛：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync --frozen alpha-lab validate-draft-factor custom_factors/research/<factor>/factor.json
UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync --frozen alpha-lab real-case single-factor run configs/real_cases/single_factor/<factor>_v1.yaml --evaluation-profile exploratory_screening --render-report --vault-export-mode skip
```

需要更完整验证时：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync --frozen alpha-lab real-case single-factor run configs/real_cases/single_factor/<factor>_v1.yaml --evaluation-profile default_research --render-report --vault-export-mode skip
```

## Artifact 审计

后端 pipeline 会自动加载 `custom_factors/{promoted,research}/<factor>/factor.json`
到 `factor_registry`。运行完成后，必须确认以下文件包含 `custom_factor_source`：

- `run_manifest.json`
- `factor_definition.json`

最小审计字段：

```json
{
  "custom_factor_source": {
    "name": "<factor>",
    "scope": "research",
    "path": "custom_factors/research/<factor>/factor.json",
    "code_sha256": "...",
    "factor_json_sha256": "...",
    "required_columns": ["..."],
    "optional_columns": ["..."],
    "frequency": "daily",
    "unavailable_data_policy": "...",
    "pit_assumption": "...",
    "provenance": {
      "idea_id": "20260511T143000Z__signed-jump-reversal",
      "stage2_payload_sha256": "...",
      "audience_chain": ["claude", "codex", "web_gpt_stage2"]
    }
  }
}
```

缺少 `code_sha256`、`factor_json_sha256` 或 source path 时，本轮 Stage3 视为失败。
`provenance` 缺失时仅产生 warning（兼容旧的 pre-protocol 因子），但所有新走
`alpha-lab idea distribute` 流的因子必须有 provenance。

## 结果分析

每轮运行后，Codex GUI 必须读取并总结：

- `metrics.json`
- `summary.md`
- `integrity_report.md`
- `factor_definition.json`
- `run_manifest.json`

分析重点：

- 是否有 PIT / temporal alignment / cross-section scope 硬失败。
- coverage、IC、rank IC、decay、turnover、cost-aware 初筛是否支持继续。
- 因子是否只是动量、反转、低波、振幅、成交活跃或流动性变量的别名。
- 下一轮只允许在机制 proxy、字段、参数、方向、过滤条件或中性化层面做具体调整。

## 晋升边界

草稿因子通过 `default_research` 后仍然不自动晋升。晋升必须遵守
`docs/factor_promotion_checklist.md`：

1. 满足 promotion thresholds。
2. 复制到 `custom_factors/promoted/<factor>/`。
3. 写 `promotion_card.md`。
4. 更新 promotion log。
5. 再通过前端跑完整图文报告归档。

前端只处理已通过后端标准评价的候选，不承担探索期试错。
