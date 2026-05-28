# Codex GUI Stage 3 后端草稿因子执行合同

本文件约束 Codex GUI 在 alpha-lab 本地仓库中执行网页版 GPT 输出的
Stage2 `factor_json_payload`。当前阶段只做后端草稿因子的标准验证，不做前端注册，不做晋升。

## 机器事实来源

只把以下内容作为机器事实：

- Stage2 输出中的 `factor_json_payload`
- Stage2 输出中的 `machine_contract`
- 用户明确指定的本地路径或命令

如果正文、`human_summary`、`stage3_notes` 与 `factor_json_payload` 冲突：

1. 以 `factor_json_payload` 为准。
2. 在 `custom_factors/research/<factor_name>/research_log.md` 记录冲突。
3. 不要根据自由文本自行改写代码。

## 允许写入

只能写入或更新：

- `custom_factors/research/<factor_name>/factor.json`
- `custom_factors/research/<factor_name>/research_log.md`
- `configs/real_cases/single_factor/<factor_name>_vN.yaml`

禁止写入：

- 临时 Python 脚本
- notebook
- 散落的 `.py` 文件
- `custom_factors/promoted`
- `src/alpha_lab/factors`
- 前端正式注册文件
- execution / replay / fill simulation 相关目录

## factor.json 合同

`factor_json_payload` 必须可以直接写入：

```text
custom_factors/research/<factor_name>/factor.json
```

最小结构：

```json
{
  "name": "",
  "description": "",
  "required_columns": [],
  "optional_columns": [],
  "frequency": "daily",
  "unavailable_data_policy": "return_nan",
  "pit_assumption": "",
  "code": ""
}
```

`code` 推荐定义：

```python
def build_factor(frame):
    ...
    return value
```

兼容旧接口：

```python
def builder(prices, *, window=20, skip_recent=0, min_periods=None, **kwargs):
    ...
    return frame
```

约束：

- `build_factor(frame)` 返回与输入 index 对齐的 `pd.Series`。
- `builder(prices, ...)` 返回包含 `date / asset / value` 的 DataFrame。
- 必须显式检查 required columns。
- 不得读写文件。
- 不得访问网络。
- 不得使用隐藏全局状态。
- 不得使用未来收益、label 或全样本统计构造 feature。

## 标准执行步骤

1. 从 Stage2 输出中提取唯一的 `factor_json_payload`。
2. 写入 `custom_factors/research/<factor_name>/factor.json`。
3. 更新或创建对应 single-factor case YAML。
   - 新 YAML 根字段应包含 `project_slug`、`archive_identity`、`evaluation_profile`。
   - `project_slug` 只能来自用户或执行信封明确给出的项目上下文；不能从 factor 名、路径、文件名猜。
   - 覆盖已有 YAML 前，先读取旧 YAML；若已有 `project_slug` / `archive_identity` / `evaluation_profile`，默认保留，除非用户明确要求修改。
4. 先运行 validator：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync --frozen alpha-lab validate-draft-factor custom_factors/research/<factor_name>/factor.json
```

5. validator 通过后，只运行标准 single-factor pipeline。

快速初筛：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync --frozen alpha-lab real-case single-factor run configs/real_cases/single_factor/<factor_name>_vN.yaml --evaluation-profile exploratory_screening --render-report --vault-export-mode skip
```

需要更完整验证时：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync --frozen alpha-lab real-case single-factor run configs/real_cases/single_factor/<factor_name>_vN.yaml --evaluation-profile default_research --render-report --vault-export-mode skip
```

## Preflight 必查

- `factor_json_payload` 是合法 JSON。
- `name` 是英文 snake_case。
- `factor.json` 位于 `custom_factors/research/<factor_name>/factor.json`。
- `required_columns` 在可用字段列表中。
- `code` 定义 `build_factor(frame)` 或旧接口 `builder(prices, ...)`。
- 代码没有 `open()`、路径写入、网络访问、`subprocess`、`eval`、`exec`。
- rolling / expanding 逻辑按 `asset` 分组。
- 没有 `shift(-n)`、负向 `pct_change` 或 future label 进入 feature。
- 没有全样本均值、全样本标准差作为 feature 标准化。

## Artifact 审计

运行后必须读取：

- `run_manifest.json`
- `factor_definition.json`
- `metrics.json`
- `summary.md`
- `integrity_report.md`
- `case_report.md`，如果已渲染

必须确认 `run_manifest.json` 和 `factor_definition.json` 包含：

```json
{
  "custom_factor_source": {
    "name": "",
    "scope": "research",
    "path": "",
    "code_sha256": "",
    "factor_json_sha256": "",
    "required_columns": [],
    "optional_columns": [],
    "frequency": "daily",
    "unavailable_data_policy": "",
    "pit_assumption": ""
  }
}
```

如果缺少 `code_sha256`、`factor_json_sha256` 或 source path，本次 Stage3 视为失败。

## 输出要求

Stage3 的最终交付物是**一份固定 schema 的迭代反馈包**，供网页版 GPT 在同一个
因子项目里生成下一版 Stage2 payload。它有两个落点：

**(a) durable trail —— 追加一行到 `research_log.md`**（≤ 80 字符，遵守 promotion
checklist 对 research_log 的"只写流水、不写决策文档"约束）：

```text
2026-05-26  v1 baseline   case=tail_drop_volume_knot_v1_v1.yaml  art=outputs/.../tail_drop_volume_knot_v1_v1  RankIC=0.008 IR=0.13  verdict=fragile
```

**(b) feedback pack —— 作为本轮 Stage3 的最终结构化输出（贴进网页版 GPT 的就是
这一块）**。固定 schema 如下；prose 字段用中文，专有名词/代码符号保留英文：

```yaml
stage3_feedback_version: single_factor_stage3_feedback_v1
idea_id: <从 factor.json provenance 原样带出>
factor_name: <factor_name>
iteration: <本次评估的 payload 版本，如 v1>
factor_thesis: <一句话经济假设，逐字保留、除非用户显式重审>
run_output_dir: <artifact 输出目录>

contract:                       # 工程契约一行带过，不展开审计细节
  validator: passed             # validate-draft-factor 结果
  source_hash_audit: ok         # code_sha256 / factor_json_sha256 / source path
  artifact_audit: ok            # run_manifest + factor_definition custom_factor_source
  provenance_idea_id: present

verdict:
  campaign_triage: <如 "Drop for now" / "Keep iterating">
  promotion_tier_L: <pass / blocked>   # 对照 Tier L（library 准入）
  one_line: <一句话定性，如 "工程契约全过；方向对但覆盖稀疏、rolling 不稳">

scorecard:                      # 指标 vs Tier L 准入门槛；只放决定性的几项
  rank_ic_mean:   {value: <num>, gate: ">=0.01",  pass: <bool>}
  rank_ic_ir:     {value: <num>, gate: ">=0.15",  pass: <bool>}
  cost_ir_5bps:   {value: <num>, gate: ">=0.05",  pass: <bool>}
  coverage_mean:  {value: <num>, gate: ">=0.65",  pass: <bool>}
  regime_sign:    {value: <x>/4, gate: ">=2/4",   pass: <bool>}
  core_gap: <一句话：距 Tier C 核心档还差什么；不到 Tier L 就写 "未到准入档">

blockers:                       # 真正卡住准入的 1-3 条现象（不是动作）
  - <如 coverage 稀疏/不均>
  - <如 rolling IC 不稳>

codex_assessment:               # Codex 自主研判：advisory，网页版 GPT 可推翻
  read: |                       # 自由文本，≤ ~150 字：本轮结果说明了什么、为何强/弱
    <失败模式 / 有效之处 / 数据层面观察；不复述 scorecard 数字>
  recommended_directions:       # ranked，最多 4 条，每条一句理由
    - {move: <方向>, rationale: <为何>, confidence: high}
  lead_pick: <如果只能改一处先改哪个 + 一句为什么>

iteration_request:
  preserve: [idea_id, factor_thesis, 已注册数据列, PIT discipline, "high=long"]
  do_not: [新增未注册字段, needs_extension 机制, Level3/execution 语义, 换数据/绕 validator]
  try_next:                     # codex_assessment 的 lead_pick + top directions 蒸馏成 payload 改动，≤3 条
    - <如 "软化 4-way AND：late_drop + close_below_vwap 为核心，其余转 soft 权重">

resource: <一行；如 "oom: not_triggered, headroom ok"；触发或逼近预算才展开>

history:                        # 读取 research_log.md 重建，保证贴最新块也不丢轨迹
  - {v: v1, change: <一句话>, rank_ic: <num>, verdict: <如 fragile>}
```

规则：

- scorecard 的 `gate` 值必须与 `docs/factor_promotion_checklist.md` 的 **Tier L**
  保持一致；改了 checklist 就同步改这里。
- 填不出可靠值的字段**直接省略**，不要用 `<value>` / `<未知>` 占位凑格式。
- `idea_id` 与 `factor_thesis` 必须逐字延续，这是网页版 GPT 识别"在迭代同一个
  因子"的对齐锚。
- `codex_assessment` 是 advisory，网页版 GPT 可推翻；它**不改** `preserve` /
  `do_not` 硬约束，且 `recommended_directions` 必须全部落在 `do_not` 之外（不准
  建议新增未注册字段 / `needs_extension` / Level3 / 换数据绕 validator）。
- `try_next` 必须是 `codex_assessment` 的蒸馏，二者不得矛盾；探索阶段不 KILL，
  最重只能 "drop for now"。

禁止输出：

- 因子已经有效
- 可以实盘
- 买入 / 卖出建议
- Level 3 execution / fill simulation / replay 建议
