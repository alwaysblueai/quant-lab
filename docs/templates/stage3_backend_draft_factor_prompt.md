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

最终用中文输出：

1. 写入了哪些文件。
2. 实际运行的 validator 命令和实验命令。
3. 实验成功或失败，以及 artifact 输出路径。
4. 是否确认 `code_sha256`、`factor_json_sha256`、source path、required columns、PIT 假设。
5. 初筛结果摘要：coverage、IC / rank IC、decay、turnover、extreme values、cost-aware 初筛。
6. 主要失败点或脆弱点。
7. 下一轮 1-3 个具体修改方向。

禁止输出：

- 因子已经有效
- 可以实盘
- 买入 / 卖出建议
- Level 3 execution / fill simulation / replay 建议
