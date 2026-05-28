# Retrieval Pack — 用价格、换手率与流动性相关的安全技术特征，对未来短期（约一周）横截面收益排序做轻量预测；偏好低复杂度线性族、成交活跃但波动不过热的股票更可信

idea_id: `20260525T115846Z__idea-f3ca571d`
lab: `model_factor`
mode: `start`

## Engines (Stage 1, symmetric task)
Both engines run the same generator + reviewer task and produce one
markdown each. Web GPT综合两份输出取长补短 in Stage 2.

- `claude` → `prompt_claude.md`
- `codex` → `prompt_codex.md`

## Vault cards (shared retrieval; identical between engines)
- vault_root: `/mnt/c/quant/vault/quant-knowledge`

### K1: Avramov He 2026 Cross-Asset Spillover SDF
- path: `40_papers/Paper - Avramov He 2026 Cross-Asset Spillover SDF.md`
- summary: 本文提出一套线性随机贴现因子（SDF）框架，同时学习“哪些公司层信号重要”以及“预测信息如何在资产之间溢出传播”。它的关键增量不是再做一版单资产排序，而是把跨资产信息传导显式纳入 SDF 构造，并以样本外 Sharpe 最大化作为估计目标。

### K2: Haitong Factor Series 24 Goodness-of-Fit Volatility-Adjusted Factor Premium
- path: `40_papers/Paper - Haitong Factor Series 24 Goodness-of-Fit Volatility-Adjusted Factor Premium.md`
- summary: 这篇来源的核心贡献不是新 factor，而是对 Fama-MacBeth 因子溢价估计链做了两个方法改进：用截面回归拟合优度驱动自适应 EWMA，再对因子溢价做波动率标准化，从而在风格切换阶段更快更新 expected premium。

### K3: Conditional Mean-Variance Multifactor Portfolio with Volatility Management
- path: `20_methods/optimization/Method - Conditional Mean-Variance Multifactor Portfolio with Volatility Management.md`
- summary: 条件均值-方差多因子波动率管理把各因子权重写成市场波动率倒数的仿射函数，并把交易成本与跨因子轧差内生进优化器，用于在高波动期系统性压低风险暴露而不牺牲多因子净收益。
- transferable_moves:
  - id: volatility_state_factor_weight_rule
    one_line: 把每个因子 sleeve 权重写成市场波动率状态的仿射函数
    why_it_works: 正文 L40-L54 定义 theta_k,t = a_k + b_k / sigma_t，并说明不同因子允许有不同 b_k，高波动时差异化降权。
    transfer_caveats: 只适用于统一状态变量对多个 sleeve 有解释力且响应异质性可估的多因子账簿；总组合 inverse-vol targeting 不等价。; 条件变量必须 PIT，不能事后用更优 proxy 重写历史状态。
    seen_in: 20_methods/optimization/Method - Conditional Mean-Variance Multifactor Portfolio with Volatility Management.md
  - id: stock_level_crossing_cost_netting
    one_line: 先在股票层把多个因子 sleeve 的交易净额轧差，再计算交易成本
    why_it_works: 正文 L71-L79 明确 TC_net 用跨因子对冲后的净股票交易向量 Delta w 计算，方法有效性依赖真实 crossing。
    transfer_caveats: 只有多个 sleeve 在同一执行账簿内真实可 crossing 时才适用；ETF、子账户或多 manager 封装结构可能不能共享净额。; 不能把单 sleeve 成本简单相加后声称用了 crossing。
    seen_in: 20_methods/optimization/Method - Conditional Mean-Variance Multifactor Portfolio with Volatility Management.md
  - id: cost_function_optimization_embedding
    one_line: 将线性成本和冲击成本函数内生写入组合优化目标，使调仓量由净 alpha 决定
    why_it_works: 正文 L56-L69 把 TC(eta) 写进扩展因子空间优化目标，L122-L124 要同时比较不计成本、计成本不 crossing、计成本且 crossing。
    transfer_caveats: 成本函数必须进入目标或约束；事后报告成本不构成本 move。
    seen_in: 20_methods/optimization/Method - Conditional Mean-Variance Multifactor Portfolio with Volatility Management.md
  - id: gross_to_net_friction_haircut
    one_line: 将 gross alpha 按成本、冲击、融资、借券、操作失败和容量压力逐项扣成 deployable alpha
    why_it_works: 正文 L56-L79 将交易成本内生进优化器，L122-L124 明确报告不计成本、计成本但不 crossing、计成本且 crossing 三种净收益口径。
    transfer_caveats: 必须显式 haircut gross alpha 或 gross return；普通 net-return 报告不应自动贴这个 id。
    seen_in: 20_methods/optimization/Method - Conditional Mean-Variance Multifactor Portfolio with Volatility Management.md

### K4: Factor Research Operating Manual
- path: `80_pipelines/alpha-research/Pipeline - Factor Research Operating Manual.md`
- summary: 因子研究操作手册是一条 repository-grounded 的研究流水线: 从想法、假设、因子设计、验证、实验、晋升和监控, 规定每一步应产生什么工件以及何时停止。
- transferable_moves:
  - id: broker_factor_research_stage_gate
    one_line: 将多因子研究拆成数据与因子筛选、收益预测、风险预测、组合优化四道闸门
    why_it_works: 华泰证券 2016 把多因子体系组织为数据标准化与有效因子识别、预期收益估计、风险预测和组合优化四个阶段，能把研究问题从“找因子”扩展到“能否部署”。
    transfer_caveats: 这是流程闸门，不是单个模型；每道闸门都要产生可追踪工件，而不是只写结论。; 收益预测与风险预测的输入时点必须独立冻结，避免在组合优化前引入未来信息。
    seen_in: 80_pipelines/alpha-research/Pipeline - Factor Research Operating Manual.md
  - id: preprocess_order_universe_data_quality_gate
    one_line: 先固定股票池和原始数据可得性，再按异常值、缺失值、标准化的顺序处理因子输入
    why_it_works: 民生证券 2020 强调财报滞后、重组可比性、行业分类不完整和流动性股票池都会改变因子结论；异常值应在缺失填充和标准化之前处理，因为后两者会使用横截面均值或分布信息。
    transfer_caveats: 股票池、停牌/ST/涨跌停、上市天数和财报 PIT 规则必须写入 validation artifact。; z-score 保留距离但受极端值影响，rank 标准化稳健但会丢失距离信息；选择应与因子机制一起记录。
    seen_in: 80_pipelines/alpha-research/Pipeline - Factor Research Operating Manual.md
  - id: mechanism_direction_prior_before_validation
    one_line: 在验证前先给候选信号记录机制家族和方向先验
    why_it_works: High-Frequency Factor Taxonomy 卡要求先判家族再判方向，并把 reversal、momentum 或 mixed 作为验证前假设。
    transfer_caveats: 方向先验不是最终方向 gate；实证验证后仍需更新或推翻。
    seen_in: 20_methods/signal/Method - A-share High-Frequency Factor Taxonomy (Five-Family Framework).md; 20_methods/signal/Method - Abnormality Radar Event Cluster Stock Selection (A-share).md; 20_methods/signal/Method - HF Fact...
    adapted_how: Design Gate 要求在因子设计时明确经济机制、方向、horizon 与预期失效条件
  - id: universe_fixed_before_sort
    one_line: 先固定可交易 universe 与流动性边界，再执行排序或分组
    seen_in: 30_factors/cross-asset/Factor - Currency Carry.md; 20_methods/ml/Method - Asset Pricing Trees for Basis Asset Construction.md; 20_methods/ml/Method - Learning-to-Rank Cross-Sectional Asset Pricing.md; 20_methods/ml/Me...
    adapted_how: 数据与 universe contract 在信号构造前固定，避免事后样本筛选

### K5: Tie-Handling in Cross-Sectional Ranking
- path: `10_concepts/portfolio/Concept - Tie-Handling in Cross-Sectional Ranking.md`
- summary: 许多量化 Alpha 依赖于通过 `rank()` 算子将原始 Cross-sectional 数据转化为均匀 Rank 值（$[0, 1]$ 或 $[-1, 1]$）。当底层数据包含大量相同值时（如零成交量、涨跌停价格、相同的类别评分），对这些\"并列\"值的处理方式会显著影响所得 Alpha 的统计特性与中性化效果。
- transferable_moves:
  - id: differentiable_rank_weight_factor_construction
    one_line: 用可微 rank-weight activation 把特征排序转成 long-short factor weights
    seen_in: 20_methods/ml/Method - Deep Characteristics-Sorted Factor Model.md; 10_concepts/portfolio/Concept - Tie-Handling in Cross-Sectional Ranking.md
    adapted_how: 卡片讨论 rank 算子如何把原始截面值映射成可排序信号，tie 处理会直接改变 rank-weight 输出
  - id: operator_semantics_catalog_with_domain_signature
    one_line: 把每个算子的输入域、输出域、arity、窗口和边界条件登记成语义目录
    why_it_works: WorldQuant Operator Library 卡 L34-L40 与 L115-L122 将 operator_semantics_catalog、causal_window_spec 和 cross_section_rule_spec 列为稳定输出
    transfer_caveats: 只在方法 owner 是算子语义治理时复用；候选表达式搜索或 alpha lifecycle 应使用更高层 move; 同名算子的 ties、NaN、warm-up 或 group 规则若未冻结，目录不可比较
    seen_in: 20_methods/signal/Method - WorldQuant Alpha Operator Library.md; 10_concepts/ml-quant/Concept - Operator-Graph Complexity & Overfitting.md; 10_concepts/portfolio/Concept - Tie-Handling in Cross-Sectional Ranking.md
    adapted_how: 卡片强调 average、min、max、dense、ordinal 等 tie-breaking 规则必须作为 rank 算子语义的一部分冻结
  - id: style_industry_neutralization_gate
    one_line: 对因子做市值、行业、常见风格暴露控制后再判断增量
    transfer_caveats: 只有正文明确要求市值/行业/风格中性化、特定风格相关性复核，或说明某类暴露会伪装成 alpha 时才复用；普通稳健性检验不能自动贴这个 id。
    seen_in: 30_factors/equity/Factor - High-Volatility Risk Compensation (勇攀高峰因子).md; 30_factors/equity/Factor - Against-the-Current Buying Intensity Factor (激流勇进因子).md; 30_factors/equity/Factor - Analyst Forecast Revision Inerti...
    adapted_how: 卡片指出 tie-handling 会改变截面均值、暴露对称性和中性化效果，因此排序后仍需检查行业、风格或规模暴露

### K6: Volatility-Stratified Single-Name Deviation Constraints for Index Enhancement (A-share)
- path: `20_methods/optimization/Method - Volatility-Stratified Single-Name Deviation Constraints for Index Enhancement (A-share).md`
- summary: 波动率分层个股偏离约束先按个股波动率分高低组，再对高波组放宽、低波组收紧单名偏离上限，用来修正指数增强组合系统性低配高波股票时的超额与回撤失衡。
- transferable_moves:
  - id: active_weight_benchmark_relative_optimization
    one_line: 把优化变量改成相对基准的主动权重，并用主动风险预算控制偏离
    why_it_works: 见 L32-L40，卡片先把组合权重改写为 a_i=w_i-b_i，说明作用对象是 benchmark-relative active weight。
    transfer_caveats: 只有明确存在 benchmark 与 active-risk 口径时复用；普通 long-only 权重约束不自动复用。
    seen_in: 20_methods/optimization/Method - Volatility-Stratified Single-Name Deviation Constraints for Index Enhancement (A-share).md
  - id: volatility_bucket_deviation_budget_schedule
    one_line: 按个股波动率分层分配单名主动偏离上限
    why_it_works: 见 L42-L74，高波组和低波组分别使用 |a_i|<=delta_H 与 |a_i|<=delta_L，且通常要求 delta_H 大于 delta_L。
    transfer_caveats: 只用于 benchmark-relative 单名偏离预算重分配，不是一般波动率中性或低波因子。; 若核心是整体压缩高波 regime 下的 factor sleeve risk，应使用相邻风险管理 move。
    seen_in: 20_methods/optimization/Method - Volatility-Stratified Single-Name Deviation Constraints for Index Enhancement (A-share).md
  - id: constraint_feasibility_binding_report
    one_line: 先检查约束集可行域和松弛顺序，再记录最优解上真正绑定的约束
    why_it_works: 见 L124-L129，验证纪律要求报告约束绑定位置迁移，确认改善来自 budget reallocation。
    transfer_caveats: 只有约束可行域、松弛顺序或 binding constraint 诊断是正文输出时复用。
    seen_in: 20_methods/optimization/Method - Volatility-Stratified Single-Name Deviation Constraints for Index Enhancement (A-share).md
  - id: constraint_set_semantics_version_gate
    one_line: 将约束集合、边界数值、线性化口径和松弛顺序一起版本化，防止策略语义悄悄漂移
    why_it_works: 见 L112-L122，constraint_spec、分层阈值 q 与 delta_H/delta_L 必须联立版本化，避免约束语义漂移。
    transfer_caveats: 它治理约束语义版本，不替代具体的波动率分层偏离预算 move。
    seen_in: 20_methods/optimization/Method - Volatility-Stratified Single-Name Deviation Constraints for Index Enhancement (A-share).md

### K7: Z-Score Normalization and Aggregation
- path: `20_methods/signal/Method - Z-Score Normalization and Aggregation.md`
- summary: Z-score标准化与聚合先把量纲、波动范围和方向不同的信号映射到可比尺度，再通过线性加权形成综合评分，是构建透明多因子基线组合最常见的信号压缩方法。
- transferable_moves:
  - id: cross_section_winsor_zscore_direction_align
    one_line: 对信号先截尾并按横截面口径做 z-score，再按预测方向统一符号
    why_it_works: Z-Score Normalization 卡 L19-L24 与 L52-L66 要求把不同量纲信号统一尺度，并记录 winsor、zscore scope 与方向翻转
    transfer_caveats: 只适用于横截面可比对象；时间序列状态变量不应机械套用; 方向翻转本身是研究假设，必须记录而不能隐藏在标准化代码里
    seen_in: 20_methods/signal/Method - Z-Score Normalization and Aggregation.md
  - id: low_freedom_linear_score_blend
    one_line: 用等权或少量分组权重把标准化信号低自由度线性压缩成综合评分
    why_it_works: Z-Score Normalization 卡 L65-L80 将聚合权重列为线性压缩规则，并建议等权或少量分组权重作为低自由度基线
    transfer_caveats: 只适合作为透明基线；存在强非线性、门槛或状态切换时不应替代结构模型; 权重若频繁调参，本 move 会从基线聚合退化成样本内优化
    seen_in: 20_methods/signal/Method - Z-Score Normalization and Aggregation.md
  - id: descriptor_cleaning_standardization_exposure_map
    one_line: 将原始描述符经过缺失治理、截尾、标准化和聚合映射成发布暴露矩阵
    why_it_works: Descriptor Standardization 卡把原始 descriptor 清洗为 z_ij,t，再按 family weights 映射成 factor_exposure_matrix_X。
    transfer_caveats: 这是风险暴露工厂，不是 alpha 因子构造；收益预测信号应另建 owner。; 缺失值处理、winsorize、标准化和聚合权重必须分开记录，不能混成一个黑箱转换。
    seen_in: 20_methods/risk-model/Method - Descriptor Standardization and Exposure Mapping.md; 20_methods/risk-model/Method - Fundamental Factor Risk Model Construction.md; 20_methods/signal/Method - Z-Score Normalization and Agg...
    adapted_how: 缺失治理、截尾、标准化和聚合配置与风险模型 descriptor cleaning 共享可审计处理链
  - id: equal_weight_component_composite
    one_line: 把同一 owner 下的多条 component 等权合成为单一 score
    why_it_works: 卡内对若干 sub-component 不做 orthogonalization 也不做差异化加权，直接 1/N 等权汇总，靠 owner 一致性而非去相关来保证可解释性
    transfer_caveats: 区别于 orthogonalized_equal_weight_component_composite：本 move 不要求 component 之间事先正交化/去相关/残差化；若正文有 orthogonalization / decorrelation / residualization 步骤，应改用前者; component 数量≥3 且 owner 一致时才适用；2 个 component 直接平均不构成 family co...
    seen_in: 30_factors/equity/Factor - Hotspot Reaction Factor (热点反应因子).md; 20_methods/macro-cross-asset/Method - A-share Equity Timing Four-Dimension Scoring System.md; 20_methods/macro-cross-asset/Method - A-share Five-Dimensio...
    adapted_how: 等权或少量分组权重把多个标准化 component 低自由度压缩成综合 score

### K8: Caitong Q-Factor A-Share Empirical Improvement 2020
- path: `40_papers/Paper - Caitong Q-Factor A-Share Empirical Improvement 2020.md`
- summary: 这篇财通报告用 2016–2020 年 A 股数据检验 Hou、Xue、Zhang（2015）的 q-factor 模型，并以 Barra CNE6 风格因子收益作为测试资产。核心结论是：投资因子 $r_{I/A}$ 在 A 股上并不显著，原始 q-model 对 Value 和 Yield 的解释明显不足，用 value 因子替换 investment 因子后，整体解释力会更好。

## Cross-card synthesis
- 本轮未生成跨卡合成摘要。

## Codebase snapshot
- single-factor promoted: 0
- single-factor research: 5
- custom models promoted: 0
- custom models research: 3
- single-factor cases registered: 1
- model-factor cases registered: 24
