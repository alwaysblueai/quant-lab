# downshock_cushion_absorb — research log

idea_id: `20260511T164925Z__idea-f92b851b`
stage2_payload_sha256: `2c612c23cf11472776b360e19ec0bf4b04bca63b7c205f0b13f8ef8a48ffd648`
audience_chain: claude → codex → web_gpt_stage2
primary_mechanism: M1 (shock-gated off-low cushion)

## v1 (2026-05-12)

- **Source**: stage2 `factor_json_payload` 原样落盘，未改 code / required_columns / pit_assumption。
- **Data**: `data/processed/real_case_inputs/ashare_institutional_20160418_20260415/prices.parquet` (7.5M 行)
- **Case**: `configs/real_cases/single_factor/downshock_cushion_absorb_v1.yaml`
- **Direction**: long
- **Horizon**: 5d forward return
- **Profile**: exploratory_screening

**Notes — Codex first execution（弃用）**：

Codex GUI 第一轮在 `\\wsl.localhost\.codex\worktrees\b7fc\alpha-lab` 这个孤儿 worktree
执行，工作树不带最新 governance docs，且本地无 TUSHARE_TOKEN / BaoStock 超时，
退化到合成 OHLC 面板做 Tier1 smoke。verdict（`Promising but fragile / Drop / Blocked`）
是 thin-coverage 默认门槛而非机制证据，**作废**。该 worktree 已清空。

**主仓库 v1 实跑结果（2026-05-12，exploratory_screening profile）**：

- 输出：`outputs/real_cases/downshock_cushion_absorb_v1/`
- **Mean RankIC = -0.0206**（IS）/ -0.0275（OOS）—— 负且 OOS 比 IS 更负
- **ICIR = -0.164**（IS）/ -0.206（OOS）—— 负方向稳定
- **Long-Short Return = -0.0082** —— 负
- **Mean MI = 0.039** —— 有非线性信息含量但线性方向反了
- **Verdict: Fails basic robustness / Drop for now / Blocked from Level 2**
- Mean Turnover = 0.58 / Coverage 0.59（IS）— 激活相对充分
- Coverage OOS 0.78 高于 IS，但 IC 反而更负，不是 thin coverage 问题

**机制解读**：

负 ICIR + 负 long-short 不是"信号无效"，是 **"方向反了"**——把 `(close-low)/(high-low)`
高解释为 alpha 在 A 股日频上不成立。这印证了 5/7 那次跑 `downshock_failed_cushion_risk_v1`
得到的"该信号应当 risk 看而非 alpha 看"结论。但既然用户已经把 risk 翻译版本判为"失败品"，
那合起来的结论是：**这条机制的日频 OHLC proxy 不论方向都不可靠**——真正的 absorption
信号需要日内 intraday 列把"卖压被买盘吸收"的微结构事实直接测出来，而不是用 OHLC 边角
当代理。

## v2 方向（基于 v1 实跑结果）

v1 失败模式 = 单日 OHLC 表面承接不能区分"真承接（流动性回补）"和"伪承接（机械尾盘拉起）"。
v2 必须用 intraday-enhanced 列直接测微结构层面的承接证据。可选输入（已注册在
`data/processed/real_case_inputs/ashare_institutional_intraday_v1/prices.parquet`，
101 列，2016-10 → 2025-12-31）：

强 candidate（直接对应"承接"语义）：
- `signed_amount_imbalance` —— 分钟级买卖方向加权成交额不平衡（DPIN 的日频降维）
- `amount_share_close30` —— 尾盘 30 min 成交额占比（"卖压被尾盘吸收"的直接测量）
- `minutes_at_low_count` —— 价格停留在低点的分钟数（小 = 真没贴在低点）
- `vwap_close_dev` —— close 相对盘中 vwap 偏离（后段定价高于均价 = 后段被承接）

辅助（区分 shock 类型）：
- `rv_neg_5m` —— 下行 realized variance（精确度量下行冲击）
- `signed_jump` —— 有方向的跳跃分量（事件型 vs 漂移型下行）
- `time_at_extremes_share` —— 价格停在极端位的总占比（governance 信号）

建议组合：
- 替代 `(close-low)/(high-low)` → `vwap_close_dev` 或 `-minutes_at_low_count_normalized`
- 替代 `(open-low)/open` shock gate → `rv_neg_5m` 当日 rolling 70 分位
- **新增** `signed_amount_imbalance` 作 directional weight：只在 `signed_amount_imbalance > 0`（净买盘）
  时记 absorption 正分；负不平衡的"承接"是被动反弹，置零

direction 决策：v1 long 失败 + risk 反向版也"失败品" → v2 应**先以 RankIC 绝对值是否进入
0.02+ 区间为目标**，方向由数据决定，**不预设**。

v2 改了 code → 视作新 candidate；理论上应回 Stage 2.2 重出 payload + 更新 sha。如果工程
允许 in-place 演进，需在 provenance 块加 `derived_from: stage2_payload_sha256` 字段做
lineage 追溯。

## 配套工程修复（2026-05-12 同日）

- 主仓库 `service.distribute_idea`: workspace_root 解析时若当前路径无
  `custom_factors/` 则向上找祖先（fix C1）。上一轮 Stage 0 distribute 跑出
  `codebase_snapshot.factors.research = []` 就是因为 web 前端默认 `--workspace-root .`
  resolve 到了 `/mnt/c/Users/yukun zhao/`，那里没有 custom_factors，导致 Stage 1
  两引擎 reviewer 看不到 `downshock_failed_cushion_risk_v1` 已有先例。修复后下一次
  distribute 会正确暴露同名机制的失败先例给 reviewer。
- 1 GB 孤儿 codex worktree 内容已删除。

## v2 计划（待 v1 结果）

用户明确希望 "下行冲击 + 收离低点 + 成交活跃" 三件套，并已确认日内分钟特征加强
的日频数据 (`ashare_institutional_intraday_v1`, 101 列, 2016-10 → 2025-12-31)
可用。Stage 1/2 当时把 volume 和 intraday 列保守标 M2/M6（v1 不引入），现在
约束被解除，v2 应回收：

- **加 M2 volume 确认**：`volume` z-score positive only（rolling 20d, asset-grouped）
  乘进 raw_t。或者改用 `turnover_rate`（已归一化，免 corporate-action 复权噪声）。
- **加 1 个 intraday 列做 cushion 精化**（候选）：
  - `time_at_extremes_share`：低值 = 价格没有长时间停在极端位 → 真实流动性承接
  - `amount_share_close30`：尾盘 30 分钟成交额占比 → 直接测量"尾盘承接活跃度"
  - `vwap_close_dev`：close vs vwap 偏离 → 后段定价相对盘中均价的位置
- **dataset 切到** `ashare_institutional_intraday_v1`（注意 cutoff 2025-12-31）

v2 改了 code → 视作新 candidate，理论上应回 Stage 2.2 重出 payload + 更新
stage2_payload_sha256。如果工程允许 in-place 演进，需在 provenance 块加
`derived_from: stage2_payload_sha256` 字段做 lineage 追溯。

## 历史先例（注意区分）

`custom_factors/research/downshock_failed_cushion_risk_v1/`（2026-05-07）使用
同源 OHLC 原料但解释为"失败承接 / 延续风险"（direction=short），且加了 governance
filter / 残差化 / winsorized z-score。已被用户标为"失败品，没有参考价值"。
本 v1 主动选择不复制其复杂度，先看 M1 裸跑信号强度。
