# 分钟数据 ETL 契约 v1

研究侧用法：因子要在前端跑回测时同时消费日频 + 日内派生列，请走 `docs/intraday_factor_workflow.md` 与 `scripts/etl/build_factor_run_inputs.py` 的 slim slice + `mode=file` 流程，不要直接把 case YAML 指向完整 joined dataset，否则会 OOM。

本文档是 `ashare_institutional_20160418_20260415_supplemented` 这批分钟数据接入的单一真理源。范围只覆盖：

- `data/processed/real_case_inputs/ashare_institutional_20160418_20260415_supplemented/prices.parquet`
- `data/processed/real_case_inputs/ashare_institutional_20160418_20260415_supplemented/universe_mask.parquet`
- 上述两张表过滤出的 5416 只资产

## 物理边界

Stage B 保留 lookback，Stage C 保持正式研究边界：

```text
Stage B: data/processed/minute_panel/year=YYYY/part-0.parquet
  date range: 2016-04-18 -> 2026-04-15
  asset set : universe_mask.asset 去重后的 5416 只
  content   : raw minute bars，按 asset, datetime 排序

Stage C: data/processed/intraday_features/year=YYYY/part-0.parquet
  date range: 2016-10-17 -> 2026-04-15
  asset set : prices.parquet 中存在的 (date, asset)
  content   : Group A daily PV + status flags + 已 promote 的 intraday feature batches
```

`universe_mask.parquet` 是稀疏长表，只存 `in_universe=1` 的行。ETL 不按 daily universe mask 删除分钟行；研究侧使用 mask 时必须 reindex 到完整 `(date, asset)` 网格，并把缺失解释为 `False`。

Stage C 不预先 join `in_universe`。它是日内特征表，不是 universe 表。

## 复权契约

Stage B 永远存 raw 分钟价格，不做复权；分钟 CSV 的成交量/成交额保持源口径，即 `股/元`。研究侧禁止各自发明复权口径，统一使用：

```text
同日内派生: raw price
跨日价格关联: adj_price_t = raw_price_t * adj_factor_t
隔夜收益: open_t * adj_factor_t / (close_{t-1} * adj_factor_{t-1}) - 1
日内收益: close_t / open_t - 1
```

Stage C 的 daily PV 由 raw 分钟聚合而来，并转换成 `prices.parquet` 的日频单位：`volume=股/100`，`amount=元/1000`。验证时必须对比 `prices.parquet` 的 raw 列：

```text
open   -> raw_open
high   -> raw_high
low    -> raw_low
close  -> raw_close
volume -> raw_vol
amount -> raw_amount
```

`prices.parquet` 没有 `raw_vwap`。Stage C 的 `vwap` 是 raw minute 聚合口径：`amount * 10 / volume`，只在 `volume > 0` 时计算。

## CSV 读取契约

读取层必须：

- 支持 zip 内 CSV 和目录内 CSV，不把 300GB raw CSV 解压成持久文件。
- sniff `utf-8` / `gb18030` / `gbk`，并 strip `\ufeff` BOM。
- 使用 asset whitelist，只读取白名单资产，其他 CSV 跳过。
- 剔除 09:25 集合竞价 bar；manifest 保留 `has_0925_bar`。
- 不把 09:25 并入 09:30。
- `datetime` 存 naive timestamp；时区语义固定为 Asia/Shanghai。

Stage B manifest 必带：

```text
asset, year, rows,
first_datetime, last_datetime,
n_unique_dates, n_minutes_per_day_mode,
has_0925_bar, n_dup_datetime,
n_dup_datetime_raw, n_deduped_rows,
n_invalid_ohlc_raw, n_dropped_all_zero_ohlc,
n_zero_volume, n_zero_amount,
n_invalid_ohlc,
source_csv_sha256, source_path,
schema_version
```

`n_minutes_per_day_mode != 240` 只做记录，不作为 gate。历史半日、特殊交易日、北交所差异都不能被静默误杀。

若源 CSV 存在完全重复的 `(asset, datetime, OHLCV...)` 行，Stage B 去掉完全重复行，避免日聚合时 volume/amount 翻倍。`n_dup_datetime_raw` 记录源文件中的重复 datetime 数，`n_deduped_rows` 记录被移除的完全重复行数，`n_dup_datetime` 记录清洗后仍然存在的 datetime 冲突数。gate 使用清洗后的 `n_dup_datetime == 0`，但报告仍要暴露 raw duplicate 分布。

若源 CSV 存在 `open=high=low=close=volume=amount=0` 的全零占位分钟，Stage B 丢弃这些行。它们是 vendor 停牌/缺行情占位，不是一字封板；一字封板应有有效价格但无成交。`n_invalid_ohlc_raw` 记录清洗前 OHLC 异常数，`n_dropped_all_zero_ohlc` 记录移除的全零占位行数，`n_invalid_ohlc` 记录清洗后仍存在的 OHLC 异常。

## 缺失与状态旗

Stage C 输出以下状态列：

```text
is_session_active   # 当日有分钟行，且不是 pre-listing
is_actively_traded  # 当日存在 volume > 0 或 amount > 0 的分钟
is_pre_listing      # Stage C 基于 prices keys 输出，固定为 0
is_likely_suspended # 同日 >=95% 资产有分钟行，但本资产没有
is_panel_missing    # 同日覆盖率 <95% 且本资产没有分钟行
stale_days          # 距上一次 is_actively_traded=1 的交易日间隔
vol_unreliable      # 分钟聚合 volume 与 prices.raw_vol 偏差 >10%
amt_unreliable      # 分钟聚合 amount 与 prices.raw_amount 偏差 >10%
```

一字封板零成交日应表达为：

```text
is_session_active=1
is_actively_traded=0
```

缺失特征保持 null，不 forward fill。任何 forward fill 都必须发生在模型或回测层，并显式记录。

`vol_unreliable=1` 或 `amt_unreliable=1` 的 `(date, asset)` 不禁用价格类因子，但研究侧必须屏蔽依赖成交量/成交额的因子，包括 turnover、Amihud、amount/volume profile、signed amount imbalance、HHI、VWAP amount weighting 等。该旗标用于隔离少数 vendor 成交数据异常，不作为 universe 删除条件。

## Batch 1 Intraday Feature Columns

第一批扩展由 `src/alpha_lab/intraday/features.py` 的纯函数计算，ETL 脚本只负责读 Stage B、按 `(date, asset)` 分组、合并已验证的 Group A Stage C。默认输出到 `data/processed/intraday_features_batch1/`，验证通过后再显式 promote，避免覆盖 gate 已闭环的数据。

### Group B: 时段收益分解

全部使用 raw 价，同日内不复权：

```text
ret_intraday  = close / open - 1
ret_morning   = close_1130 / open - 1
ret_afternoon = close / open_1300 - 1
ret_open5     = close_0935 / open - 1
ret_close5    = close / close_1455 - 1
ret_first30   = close_1000 / open - 1
ret_last30    = close / close_1430 - 1
ret_mid       = close_1300 / close_1000 - 1
```

时间点缺失时，`close_HHMM` 使用该时间点之前或等于该时间点的最后一根 close；`open_1300` 使用 13:00 之后或等于 13:00 的第一根 open。分母缺失、非有限或接近 0 时输出 `NaN`。

### Group C: Realized Volatility

收益使用分钟 close 的 log return；5m/15m 使用按成交 bar index 的非重叠采样 close-to-close log return，并在采样尾部显式包含当日最后 close：

```text
rv_1m     = sum(r_1m^2)
rv_5m     = sum(r_5m^2)
rv_15m    = sum(r_15m^2)
bv_5m     = (pi / 2) * sum(|r_t| * |r_{t-1}|)
jump_5m   = max(rv_5m - bv_5m, 0)
rv_pos_5m = sum(r_5m^2 * 1{r_5m > 0})
rv_neg_5m = sum(r_5m^2 * 1{r_5m < 0})
signed_jump = (rv_pos_5m - rv_neg_5m) / rv_5m
rv_morning   = sum(r_1m^2) within rows <= 11:30
rv_afternoon = sum(r_1m^2) within rows >= 13:00
```

`signed_jump` 在 `rv_5m <= 1e-12` 时输出 `NaN`。上午/下午 RV 在各自 session 内重新计算 log diff，不跨午休。

### Group D: 高阶矩

```text
intraday_skew_1m = skew(r_1m)
intraday_kurt_1m = Fisher kurtosis(r_1m)
intraday_skew_5m = skew(r_5m)
intraday_kurt_5m = Fisher kurtosis(r_5m)
```

1m 至少需要 30 个非零 return，5m 至少需要 6 个非零 return，否则输出 `NaN`。所有 `inf/-inf` 在公式层转为 `NaN`。

Batch 1 验证使用 `scripts/verify/intraday_batch1_summary.py`：

```text
Group A hash: 每年 base columns hash 必须完全一致
NaN reliable rows gate:
  ret / rv / signed_jump 等列 < 0.5%
  intraday_skew_* / intraday_kurt_* < 1.0%
ret_intraday p1/p99: within +/-11%
rv_5m p99: < 0.10^2
rv_5m == rv_pos_5m + rv_neg_5m: abs diff <= 1e-12
jump_5m == max(rv_5m - bv_5m, 0): abs diff <= 1e-12
intraday_kurt_1m >= -2
file size per year: 150-400 MB
```

`bv_5m <= rv_5m` 不是理论恒等式；平滑连续收益路径下 bipower variation 可以高于 realized variance。因此报告保留该项为 diagnostic，不作为 blocking gate。

`ret_morning + ret_afternoon ~= ret_intraday` 也不是当前列定义的恒等式：Batch 1 存的是 simple return，且 morning/afternoon 两段不包含独立午休 gap 项。报告保留该项为 diagnostic，不作为 blocking gate。

## Batch 2 Intraday Feature Columns

第二批扩展仍由 `src/alpha_lab/intraday/features.py` 的纯函数计算，默认在 Batch 1 promoted root 上追加 Group E/F，不重算或覆盖已有列。推荐输出到 `data/processed/intraday_features_batch2/`，通过 `scripts/verify/intraday_batch2_summary.py` 后再显式 promote。

### Group E: 成交时段分布与集中度

分钟 bar 的 `datetime` 解释为该分钟开始时间。Batch 2 锁定以下时段：

```text
OPEN30       = 09:30 <= t < 10:00
PRE_LUNCH30 = 11:00 <= t <= 11:30
POST_LUNCH30= 13:00 <= t <= 13:30
CLOSE30      = 14:30 <= t <= 15:00
MORNING      = 09:30 <= t <= 11:30
AFTERNOON    = 13:00 <= t <= 15:00
```

`CLOSE30` 显式包含 15:00 bar，因为 15:00 bar 承载 14:57-15:00 收盘集合竞价成交，语义上属于尾盘窗口。
实际 Stage B 面板常规日包含 11:30 bar，且部分个股该 bar 成交额可能很大；因此 `MORNING` 和 `PRE_LUNCH30` 显式包含 11:30，使 `amount_share_morning + amount_share_afternoon == 1` 成为完整交易日恒等式。常规日午后通常从 13:01 开始，`POST_LUNCH30` 包含 13:30 以覆盖午后前 30 根附近的成交。

```text
amount_share_open30        = amount(OPEN30) / total_amount
amount_share_pre_lunch30   = amount(PRE_LUNCH30) / total_amount
amount_share_post_lunch30  = amount(POST_LUNCH30) / total_amount
amount_share_close30       = amount(CLOSE30) / total_amount
amount_share_morning       = amount(MORNING) / total_amount
amount_share_afternoon     = amount(AFTERNOON) / total_amount
amount_hhi                 = sum((amount_i / total_amount)^2)
amount_top10_share         = sum(top 10 minute amounts) / total_amount
volume_kurt_1m             = Fisher kurtosis(volume_i), only volume_i > 0
minutes_to_50pct_amount    = first chronological minute position where cumulative amount >= 50%
```

所有 `amount_share_*`、`amount_hhi`、`amount_top10_share`、`minutes_to_50pct_amount` 在 `total_amount <= 1e-12` 时输出 `NaN`。`volume_kurt_1m` 只使用 `volume > 0` 的分钟，至少需要 30 个有效分钟，否则输出 `NaN`。

`minutes_to_50pct_amount` 按时间顺序累计，不按金额排序；它表达成交重心偏早还是偏晚，不能退化成 HHI/top-k 的近似列。

### Group F: VWAP 偏离

日内 VWAP 使用分钟 raw amount / raw volume 计算，与 Group A 的 `vwap` 同口径：

```text
day_vwap = sum(amount) / sum(volume)
vwap_close_dev = (close - day_vwap) / day_vwap
vwap_open_dev  = (open  - day_vwap) / day_vwap
vwap_high_dev  = (high  - day_vwap) / day_vwap
vwap_low_dev   = (low   - day_vwap) / day_vwap
vwap_minute_dispersion = std(close_1m - day_vwap, ddof=0) / day_vwap
```

`vwap_minute_dispersion` 使用 minute close 的 unweighted population std (`ddof=0`)，再除以 `day_vwap` 消除量纲。若 `sum(volume) <= 1e-12` 或 `day_vwap <= 1e-12`，Group F 全部输出 `NaN`。

Batch 2 验证使用 `scripts/verify/intraday_batch2_summary.py`：

```text
Group A + Batch 1 hash: 每年 base columns hash 必须完全一致
NaN reliable rows gate:
  amount/VWAP share/deviation columns < 0.5%
  volume_kurt_1m < 1.0%
amount_share_morning + amount_share_afternoon == 1: abs diff <= 1e-9
amount_share_open30 + pre_lunch30 + post_lunch30 + close30 <= 1
amount_hhi: 1 / n_minutes_traded <= amount_hhi <= 1
amount_top10_share: 0 <= value <= 1
minutes_to_50pct_amount: 1 <= value <= n_minutes_traded + n_minutes_zero_volume
vwap_minute_dispersion >= 0
file size per year: 250-700 MB
```

上述恒等校验只在 `is_actively_traded=1 AND vol_unreliable=0 AND amt_unreliable=0` 的 reliable rows 上执行。若全市场历史上存在合法的半日或无午后交易日，验证脚本应把该日单独列为 diagnostic，而不是改公式口径。

## Batch 3 Intraday Feature Columns

### Group G: 价量协同 (PV correlation)

```text
corr_ret_volume_1m       = pearson(r_1m, vol_1m), 仅 vol>0 的分钟, min_count=30
corr_absret_volume_1m    = pearson(|r_1m|, vol_1m), 同样 vol>0 + min_count=30
                         #  Kyle-λ 代理：高换手分钟 vs 收益绝对值的关联
signed_amount_imbalance  = (Σ amount·1{r>0} − Σ amount·1{r<0}) / total_amt
pos_amount_share         = Σ amount·1{r>0} / total_amt
neg_amount_share         = Σ amount·1{r<0} / total_amt
zero_ret_amount_share    = Σ amount·1{r==0 OR ret is NaN} / total_amt
amihud_intraday          = mean(|r_1m| / amount_1m), 仅 amount>0 的分钟
```

零分母策略：`total_amt <= eps` 时 `signed_amount_imbalance / *_share` 全 NaN。
`amihud_intraday` 在 valid (amount>0 且 ret 非 NaN) 行数 = 0 时 NaN。
`corr_*` 在样本不足 30 或 std==0 时 NaN（不抛异常）。

### Group I: 微频时序 (Microfrequency timeseries)

```text
ret_autocorr_1m_lag1      = pearson(r_1m_t, r_1m_{t-1}), min_count=30
amount_autocorr_1m_lag1   = pearson(amount_t, amount_{t-1}), min_count=30
avg_gap_between_trades    = 相邻 vol>0 分钟的平均行内距离（分钟数）
time_at_extremes_share    = #{|close - day_high|/day_range < 1% OR
                              |close - day_low|/day_range < 1%} / N_minutes
acceleration_max          = max(|2·close_t - close_{t-1} - close_{t+1}|) / day_vwap
```

`avg_gap_between_trades`：vol>0 分钟数 < 2 时 NaN。
`time_at_extremes_share`：day_range == 0（一字封板）时 NaN。
`acceleration_max`：N_minutes < 3 或 day_vwap = 0 时 NaN。

## Batch 4 Intraday Feature Columns

### Group H: 微观结构 (Microstructure / extremes)

需要 join `prices.parquet` 的 `up_limit / down_limit / raw_pre_close`。daily_meta 在 ETL 阶段以 `(date, asset, up_limit, down_limit, prev_close)` 静态行注入。

```text
limit_up_touch_count    = #{|close_1m - up_limit| <= 0.005}
limit_up_open_count     = up_limit 状态从 1 → 0 的转移次数（单日内打开次数）
limit_down_touch_count  = #{|close_1m - down_limit| <= 0.005}
limit_down_open_count   = down_limit 状态 1 → 0 的转移次数
minutes_at_high_count   = #{|close_1m - max(close)| <= 0.005}     # 用 max(close) 而非 max(high)
minutes_at_low_count    = #{|close_1m - min(close)| <= 0.005}
sign_flip_count         = #{r_1m 符号反转, 仅 |r| > 1e-4}
                        # 零收益 / 噪声不计入翻转
max_abs_return_zscore   = max|r_1m| / std(r_1m), min_count=30
roll_spread_proxy       = 2·sqrt(max(0, -cov(r_1m_t, r_1m_{t-1})))
                        # Roll's implied spread; cov<0 时才有效，否则 0.0
gap_fill_ratio          = clip((open - close)/(open - prev_close), -3, 3)
                        # |open - prev_close| < 0.005 时 NaN（无 gap）
```

零分母 / 边界策略：
- up_limit / down_limit 缺失（prices 未提供）→ 对应 4 个 limit_* 列 NaN
- `minutes_at_*` 用 close-based 极值（不用 high/low），避免单 bar 高低污染
- `sign_flip_count`：仅有 0 或 1 个非零方向时 = 0
- `max_abs_return_zscore`、`roll_spread_proxy`：valid returns < 30 时 NaN
- `gap_fill_ratio`：prev_close 缺失或 |gap| < tick → NaN，结果钳到 [-3, 3]

### 恒等校验

```text
limit_up_touch_count >= limit_up_open_count
limit_down_touch_count >= limit_down_open_count
minutes_at_high_count, minutes_at_low_count >= 0
0 <= time_at_extremes_share <= 1
acceleration_max >= 0
roll_spread_proxy >= 0
sign_flip_count >= 0
```

## Known Source Discrepancies

- `prices.parquet` 的 daily open/high/low 可能包含 09:15-09:25 开盘集合竞价和 14:57-15:00 收盘集合竞价的价格；Stage B 分钟面板从 09:30 连续竞价开始，不包含这些集合竞价 tick。
- 因此分钟聚合的 open/high/low 与 `prices.raw_open/raw_high/raw_low` 在除权日、上市首日、涨跌停日和北交所样本上预期存在结构性差异。这类差异不单独判为 ETL 错误。
- 北交所部分股票的 `prices.raw_open` 来自集合竞价成交价，可能不同于 09:30 第一根分钟 bar 的 open。BJ open mismatch 单独归类为集合竞价口径差异，不作为硬 gate；非北交所 open 仍保留宽松硬 gate，用来捕捉全局错位。
- 少数资产存在成交量/成交额 vendor 异常，例如分钟值本身相对 daily truth 偏离约 2 倍。ETL 不对单资产做 hard-coded 修复，用 `vol_unreliable` / `amt_unreliable` 屏蔽依赖成交数据的因子。

## Known Feature Limitations

- `amount_share_close30`（Group E，尾盘 30min 成交占比）：在 2024-2025 子样本上对 1 日前向收益有显著横截面 IC（约 +0.014, t=+3.8），但**全 10 年样本（2016-2025）IC 趋零（约 -0.001, t=-0.74）**，呈现"近期管用、长期失效"特征。原因推测是 2021 年量化普及后机构尾盘策略竞争加剧。
  - 列本身的物理量与公式都正确（value_distribution 分布正常、`amount_share_morning + amount_share_afternoon == 1` 恒等通过），不是 ETL 或公式 bug
  - 研究侧不应单独把它当因子直接用；可作为多因子模型输入或与其他时段 share 列做交互
  - readiness review 里 sanity_ic 会触发该列的 "flat" 条件，属于已知现象，不阻断 v1.0 release

## Residual Amount / Volume Tolerance

`vol_unreliable=1` 和 `amt_unreliable=1` 的行已经不参与依赖成交量/成交额的研究特征，因此 verify gate 也只在 reliable rows 上计算成交量/成交额 pass rate：

```text
amount gate denominator: amt_unreliable == 0
volume gate denominator: vol_unreliable == 0
```

过滤 unreliable rows 后，仍可能存在少量 bounded residual：SH/SZ 普通日约 0.3% 行、BJ 普通日约 1% 行会与 `prices.parquet` 的成交额/成交量存在小偏差，典型 rel diff 中位数小于 2%。该差异来自 vendor-side 的成交额聚合/舍入/口径残差，不能安全判定哪一侧绝对正确。

这些 residual：

- 被 10% unreliable threshold 上界约束。
- 不影响价格类因子。
- 对 ranking-based 的成交量/成交额派生因子影响低于噪声下限。
- 不作为 universe 删除条件。

BJ 与 SH/SZ 分开 gate，避免 BJ 的 vendor 口径残差拖低 SH/SZ 的数据质量判断。

2024-2025 smoke 中，BJ event rows 只有约 800 行，失败集中在 limit day，且 reliable rows 的 rel diff 仍被 10% unreliable threshold 约束。BJ event 使用单独 hard gate：amount >= 95%，volume >= 97.5%。

## 验证 gate

`scripts/verify/daily_pv_diff.py` 生成 markdown 报告。硬 gate 只覆盖 `prices.parquet` 中存在的正式 `(date, asset)` 行；lookback 区间只做 Stage B manifest/schema/OHLC 异常检查。

容差：

```text
open, close: abs <= 0.01 或 rel <= 1e-4
high, low  : abs <= 0.01 或 rel <= 1e-4
volume     : abs <= 10 手 或 rel <= 1e-3
amount     : rel <= 1e-3
```

报告必须包含：

- assets only in prices / only in intraday
- 每个 metric 的 pass_rate、p50/p95/max abs diff、p50/p95/max rel diff
- mismatch rate by year、exchange、board
- mismatch rate by event class
- top-100 worst `(date, asset)`，附带 status flags

第一版 smoke 使用 2024-2025。放行门槛：

```text
manifest: n_dup_datetime == 0
manifest: n_invalid_ohlc == 0 after dropping all-zero vendor placeholders
close: all event_class pass_rate >= 99.99%
open: non-BJ pass_rate >= 99.5%; BJ open 作为 auction known limitation 单独报告
amount reliable rows:
  ordinary_SHSZ >= 99.5%; ordinary_BJ >= 98.5%
  event_SHSZ >= 99%; event_BJ >= 95%
volume reliable rows:
  ordinary_SHSZ >= 99.5%; ordinary_BJ >= 98%
  event_SHSZ >= 98%; event_BJ >= 97.5%
high/low: ordinary pass_rate >= 95%; event pass_rate >= 80% (known limitation)
vol_unreliable / amt_unreliable: all rows < 1% (informational, not universe gate)
```

任一未达标时，不静默放行；要么回炉，要么在报告和本文档后续版本中写清楚放行原因。
报告中的 `known_limitation` 和 `informational` 行用于暴露口径差异和质量旗分布，不计入 blocking hard gate；只有 `gate_type=hard` 且 `blocking=True` 才阻断后续 60 列特征放行。
