# Single-Factor 快速验证 Pipeline（长期复用）

本文档定义当前 `real_cases/single_factor` 的长期可复用快速验证框架，目标是让新增因子时只关注三件事：

1. 输入字段（`date, asset, close` + 可选字段）
2. 因子定义（`factor_path` 或 `factor_input.recipe`）
3. 是否复用基础缓存（`reuse_input_bundle`）

---

## 1. 分层主流程（稳定边界）

实现位置：`src/alpha_lab/real_cases/single_factor/pipeline.py`

1. **Input Loading**
   - `load_standard_inputs(...)`
   - 读取 `prices + universe`，并标准化为 `SingleFactorInputBundle`

2. **Base Feature Preparation**
   - `prepare_base_features(...)`
   - 在标准化 panel 上预计算常用 trailing return 与 forward labels
   - 缓存到 `SingleFactorBaseFeatureCache`

3. **Factor Construction**
   - 文件模式：`_default_factor_loader(...)` / `factor_path`
   - 配方模式：`_load_factor_from_recipe(...)` / `factor_input.recipe`
   - 预处理：`_prepare_factor(...)`（winsorize/standardize/direction/min_coverage）
   - 可选中性化：`_maybe_neutralize_factor(...)`

4. **Evaluation Kernel**
   - `evaluate_single_factor_case(...)`（位于 `evaluate.py`）
   - 使用共享输入与预计算 labels 做 Level 1/2 评估

5. **Batch Execution**
   - `run_single_factor_batch(...)`
   - `run_single_factor_cases(...)`
   - 并行配置：`SingleFactorBatchParallelConfig(mode='serial'|'thread'|'process', ...)`

---

## 2. 统一 API（新增因子建议入口）

### 2.1 单因子 sanity check

CLI：

```bash
alpha-lab real-case single-factor run configs/real_cases/single_factor/bp.yaml \
  --evaluation-profile exploratory_screening
```

Python：

```python
from alpha_lab.real_cases.single_factor import run_single_factor_case

result = run_single_factor_case(
    "configs/real_cases/single_factor/bp.yaml",
    evaluation_profile="exploratory_screening",
)
```

### 2.2 批量跑多个因子（推荐）

```python
from alpha_lab.real_cases.single_factor import (
    SingleFactorBatchParallelConfig,
    run_single_factor_batch,
)

runs = run_single_factor_batch(
    "configs/real_cases/single_factor/bp.yaml",  # base spec
    [
        {"factor_name": "bp", "case_name": "batch_bp"},
        {"factor_name": "neg_bp", "factor_path": "data/neg_bp.csv", "case_name": "batch_neg_bp"},
        {
            "factor_name": "mom20_recipe",
            "case_name": "batch_mom20_recipe",
            "factor_input": {
                "mode": "recipe",
                "recipe": {
                    "steps": [{"op": "returns", "window": 20}],
                },
            },
        },
    ],
    evaluation_profile="exploratory_screening",
    reuse_input_bundle=True,
    batch_parallel_config=SingleFactorBatchParallelConfig(
        mode="process",
        max_workers=4,
        factors_per_worker=1,
    ),
)
```

---

## 3. 缓存/加速路径与 fallback

### 3.1 当前支持的缓存/加速路径

1. **输入缓存（核心）**
   - `reuse_input_bundle=True`（默认）
   - `prices/universe/base features/forward labels` 在 batch 内复用

2. **因子级并行**
   - `thread`：共享内存、避免对象复制，适合内存受限环境
   - `process`：真正 CPU 并行，适合中大批量因子吞吐

3. **按批分配 worker**
   - `factors_per_worker` 支持 “一个 worker 一个因子” 或 “一个 worker 一批因子”

### 3.2 fallback 路径（稳定保底）

1. `mode='serial'`：完全串行，最易调试
2. `reuse_input_bundle=False`：每个 case 独立加载输入（便于隔离排障）
3. `process` 模式下若 `max_workers<=1`，自动回退串行路径

### 3.3 已知边界

1. 当前未做真正共享内存 dataframe；`process` 仍有多进程内存副本成本
2. `process` 模式不支持自定义 `factor_loader`
3. 从 `stdin/REPL` 临时脚本触发 `spawn` 多进程有入口限制，建议用 `.py` 文件/CLI

---

## 4. 新增因子的推荐最小改动

1. 尽量只改 `factor_path` 或 `factor_input.recipe`
2. 保持 `prices_path` / `universe` 不变以命中输入缓存
3. 批量验证先用 `reuse_input_bundle=True` + `serial/thread` 做 correctness
4. correctness 稳定后再切 `process` 做吞吐测试

