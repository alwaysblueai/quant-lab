# Research Playbook

本文件定义 alpha-lab 的单因子研究分层。原则是：

> 能解释一个因子的，放 Tier 2；能比较一批因子的，才进 Tier 1。

## Tier 1: 单因子 pipeline + dashboard

Tier 1 是稳定评估层，用来回答“这个因子是否值得继续看”。

Tier 1 代码和契约包括：

- `src/alpha_lab/real_cases/single_factor/`
- `src/alpha_lab/grouped_evaluation.py`
- `src/alpha_lab/bucket_builders.py`
- pipeline artifact contracts / manifest / dashboard types

Tier 1 允许加入的诊断必须满足至少一个条件：

- 可用于横向比较一批因子
- 会影响 continue / stop verdict
- schema 稳定，能写测试，能被 dashboard 长期消费

新增 Tier 1 artifact 时必须同步考虑：

- artifact 文件名和 schema
- manifest / contract
- tests
- dashboard 是否真的需要展示

当前 canonical single-factor path 是：

- config: `configs/real_cases/single_factor/`
- output: `outputs/real_cases/<case>/`

不要新增 `configs/single_factor/{tier1,sweeps}/` 这类并行配置入口。路径重整应等
`single_factor`、`model_factor`、`multi_factor` 可以一起迁移时单独执行，避免只迁一类 case
导致 `real_cases/` 半拆分。

新 pipeline canonical output 以 `outputs/real_cases/<case>/` 为准，直到整体 layout
refactor 完成。

## Tier 2: 因子专属深度研究

Tier 2 是研究工作台，用来解释机制、做专项诊断、形成可晋升候选。

Tier 2 文件放在：

- `notebooks/deepdives/<factor>/`
- `src/alpha_lab/research/`
- `<case_dir>/deepdive/`
- `docs/experiment_cards/`

Tier 2 可以包含：

- 特定 lookback / regime / 二维 bucket
- 因子机制解释
- 临时图表和表格
- residual 对照、holding sweep、强信号子集分析

Tier 2 不应直接修改：

- pipeline manifest schema
- dashboard panel
- Tier 1 artifact contract

## 共享 helper 归属

- `src/alpha_lab/bucket_builders.py`: 稳定、可 Tier 1 使用的 builder
- `src/alpha_lab/research/bucket_builders.py`: 实验性或因子专属 builder
- `src/alpha_lab/research/deepdive_io.py`: notebook artifact IO
- `notebooks/deepdives/<factor>/`: glue code、图表编排、结论判断

如果 helper 服务一批因子，抽到 `src/alpha_lab/`；如果只解释一个因子，先留在 Tier 2。

## 任务标签

- `[Tier1 Pipeline]`: pipeline、artifact、manifest、contract、稳定测试
- `[Tier1 Dashboard]`: frontend 消费稳定 Tier 1 artifact
- `[Tier2 Research]`: notebook、deepdive output、factor-specific analysis
- `[Shared Research Lib]`: research helper 或可晋升的共享函数

## Promotion Checklist

从 Tier 2 升到 Tier 1 前逐项确认：

- 是否能比较一批因子？
- 是否影响 continue / stop verdict？
- schema 是否稳定？
- 是否已有至少两个以上 deepdive 证明它有复用价值？
- 是否能写 contract / tests？
- dashboard 展示是否必要？

不满足时，继续留在 Tier 2，并记录到 `docs/promotion_log.md`。

## Deferred Tier 1 Candidates

- `conditional_ic_by_trailing_return.csv`: 当前保留为 Tier 2/deepdive 诊断。稳定 builder 已存在，但尚未接入 `evaluate.py`。
- layout 整体重整：短期不迁 single-factor 专用路径；等 `model_factor` 也进入 deepdive
  阶段后，再把 `real_cases/` 拆为 `single_factor/`、`model_factor/`、`multi_factor/`
  并列布局。

## 与 research_workflow.md 的关系

`docs/research_workflow.md` 描述研究阶段和方法论；本文件描述代码、artifact、notebook 的工程边界。两者正交：研究流程决定“问什么问题”，本 playbook 决定“代码和结果放哪里”。
