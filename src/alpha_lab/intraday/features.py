"""Pure intraday feature formulas for one asset-date minute panel."""

from __future__ import annotations

from datetime import time

import numpy as np
import pandas as pd

from alpha_lab.intraday._formulas import (
    EPS,
    bipower_variation,
    first_value_at_or_after,
    kurt_with_min_nonzero,
    last_value_at_or_before,
    log_returns,
    prepare_minute_day,
    realized_variance,
    safe_autocorr_lag1,
    safe_corr,
    safe_return,
    sampled_log_returns,
    segment_log_returns,
    signed_realized_variance,
    skew_with_min_nonzero,
    time_values,
)

RETURN_DECOMPOSITION_COLUMNS = [
    "ret_intraday",
    "ret_morning",
    "ret_afternoon",
    "ret_open5",
    "ret_close5",
    "ret_first30",
    "ret_last30",
    "ret_mid",
]

REALIZED_VOLATILITY_COLUMNS = [
    "rv_1m",
    "rv_5m",
    "rv_15m",
    "bv_5m",
    "jump_5m",
    "rv_pos_5m",
    "rv_neg_5m",
    "signed_jump",
    "rv_morning",
    "rv_afternoon",
]

INTRADAY_MOMENT_COLUMNS = [
    "intraday_skew_1m",
    "intraday_kurt_1m",
    "intraday_skew_5m",
    "intraday_kurt_5m",
]

BATCH1_FEATURE_COLUMNS = (
    RETURN_DECOMPOSITION_COLUMNS + REALIZED_VOLATILITY_COLUMNS + INTRADAY_MOMENT_COLUMNS
)

VOLUME_TIMING_COLUMNS = [
    "amount_share_open30",
    "amount_share_pre_lunch30",
    "amount_share_post_lunch30",
    "amount_share_close30",
    "amount_share_morning",
    "amount_share_afternoon",
    "amount_hhi",
    "amount_top10_share",
    "volume_kurt_1m",
    "minutes_to_50pct_amount",
]

VWAP_DEVIATION_COLUMNS = [
    "vwap_close_dev",
    "vwap_open_dev",
    "vwap_high_dev",
    "vwap_low_dev",
    "vwap_minute_dispersion",
]

BATCH2_FEATURE_COLUMNS = VOLUME_TIMING_COLUMNS + VWAP_DEVIATION_COLUMNS
BATCH12_FEATURE_COLUMNS = BATCH1_FEATURE_COLUMNS + BATCH2_FEATURE_COLUMNS

PV_CORRELATION_COLUMNS = [
    "corr_ret_volume_1m",
    "corr_absret_volume_1m",
    "signed_amount_imbalance",
    "pos_amount_share",
    "neg_amount_share",
    "zero_ret_amount_share",
    "amihud_intraday",
]

MICROFREQ_COLUMNS = [
    "ret_autocorr_1m_lag1",
    "amount_autocorr_1m_lag1",
    "avg_gap_between_trades",
    "time_at_extremes_share",
    "acceleration_max",
]

BATCH3_FEATURE_COLUMNS = PV_CORRELATION_COLUMNS + MICROFREQ_COLUMNS

MICROSTRUCTURE_COLUMNS = [
    "limit_up_touch_count",
    "limit_up_open_count",
    "limit_down_touch_count",
    "limit_down_open_count",
    "minutes_at_high_count",
    "minutes_at_low_count",
    "sign_flip_count",
    "max_abs_return_zscore",
    "roll_spread_proxy",
    "gap_fill_ratio",
]

BATCH4_FEATURE_COLUMNS = MICROSTRUCTURE_COLUMNS

BATCH123_FEATURE_COLUMNS = (
    BATCH1_FEATURE_COLUMNS + BATCH2_FEATURE_COLUMNS + BATCH3_FEATURE_COLUMNS
)
BATCH1234_FEATURE_COLUMNS = BATCH123_FEATURE_COLUMNS + BATCH4_FEATURE_COLUMNS

# Sign-flip threshold: a 1bp move discriminates real direction reversals from
# rounding noise. Tied with `tick_threshold` referenced in the contract.
SIGN_FLIP_TICK = 1e-4
# A minute is "at extremes" when its close is within EXTREME_RATIO of the daily
# range vs day_high or day_low.
EXTREME_RATIO = 0.01
# Two consecutive same-cents prices vs `up_limit`/`down_limit` that are within
# this absolute tolerance are treated as touching the limit price.
LIMIT_TOUCH_TOL = 0.005


def compute_return_decomposition(day: pd.DataFrame) -> dict[str, float]:
    """Compute Group B raw same-day return decomposition."""

    frame = prepare_minute_day(day)
    if frame.empty:
        return {column: np.nan for column in RETURN_DECOMPOSITION_COLUMNS}

    day_open = float(frame["open"].iloc[0])
    day_close = float(frame["close"].iloc[-1])

    close_0935 = last_value_at_or_before(frame, "close", time(9, 35))
    close_1000 = last_value_at_or_before(frame, "close", time(10, 0))
    close_1130 = last_value_at_or_before(frame, "close", time(11, 30))
    close_1300 = last_value_at_or_before(frame, "close", time(13, 0))
    open_1300 = first_value_at_or_after(frame, "open", time(13, 0))
    close_1430 = last_value_at_or_before(frame, "close", time(14, 30))
    close_1455 = last_value_at_or_before(frame, "close", time(14, 55))

    return {
        "ret_intraday": safe_return(day_close, day_open),
        "ret_morning": safe_return(close_1130, day_open),
        "ret_afternoon": safe_return(day_close, open_1300),
        "ret_open5": safe_return(close_0935, day_open),
        "ret_close5": safe_return(day_close, close_1455),
        "ret_first30": safe_return(close_1000, day_open),
        "ret_last30": safe_return(day_close, close_1430),
        "ret_mid": safe_return(close_1300, close_1000),
    }


def compute_realized_volatility(day: pd.DataFrame) -> dict[str, float]:
    """Compute Group C realized volatility features."""

    frame = prepare_minute_day(day)
    if frame.empty:
        return {column: np.nan for column in REALIZED_VOLATILITY_COLUMNS}

    returns_1m = log_returns(frame["close"]).dropna()
    returns_5m = sampled_log_returns(frame, 5)
    returns_15m = sampled_log_returns(frame, 15)
    morning_returns = segment_log_returns(frame, start=None, end=time(11, 30))
    afternoon_returns = segment_log_returns(frame, start=time(13, 0), end=None)

    rv_5m = realized_variance(returns_5m)
    bv_5m = bipower_variation(returns_5m)
    rv_pos_5m = signed_realized_variance(returns_5m, positive=True)
    rv_neg_5m = signed_realized_variance(returns_5m, positive=False)

    if np.isfinite(rv_5m) and np.isfinite(bv_5m):
        jump_5m = max(rv_5m - bv_5m, 0.0)
    else:
        jump_5m = np.nan

    if np.isfinite(rv_5m) and rv_5m > EPS:
        signed_jump = (rv_pos_5m - rv_neg_5m) / rv_5m
    else:
        signed_jump = np.nan

    return {
        "rv_1m": realized_variance(returns_1m),
        "rv_5m": rv_5m,
        "rv_15m": realized_variance(returns_15m),
        "bv_5m": bv_5m,
        "jump_5m": jump_5m,
        "rv_pos_5m": rv_pos_5m,
        "rv_neg_5m": rv_neg_5m,
        "signed_jump": signed_jump,
        "rv_morning": realized_variance(morning_returns),
        "rv_afternoon": realized_variance(afternoon_returns),
    }


def compute_intraday_moments(
    day: pd.DataFrame,
    *,
    min_nonzero_1m: int = 30,
    min_nonzero_5m: int = 6,
) -> dict[str, float]:
    """Compute Group D intraday skew/kurtosis features.

    Pandas' Fisher kurtosis convention is used, so a normal distribution has
    kurtosis near 0.
    """

    frame = prepare_minute_day(day)
    if frame.empty:
        return {column: np.nan for column in INTRADAY_MOMENT_COLUMNS}

    returns_1m = log_returns(frame["close"]).dropna()
    returns_5m = sampled_log_returns(frame, 5)

    return {
        "intraday_skew_1m": skew_with_min_nonzero(returns_1m, min_nonzero=min_nonzero_1m),
        "intraday_kurt_1m": kurt_with_min_nonzero(returns_1m, min_nonzero=min_nonzero_1m),
        "intraday_skew_5m": skew_with_min_nonzero(returns_5m, min_nonzero=min_nonzero_5m),
        "intraday_kurt_5m": kurt_with_min_nonzero(returns_5m, min_nonzero=min_nonzero_5m),
    }


def compute_batch1_features(day: pd.DataFrame) -> dict[str, float]:
    """Compute first intraday expansion batch: Groups B, C, and D."""

    frame = prepare_minute_day(day)
    output: dict[str, float] = {}
    output.update(compute_return_decomposition(frame))
    output.update(compute_realized_volatility(frame))
    output.update(compute_intraday_moments(frame))
    return output


def compute_volume_timing(
    day: pd.DataFrame,
    *,
    min_volume_kurt_count: int = 30,
) -> dict[str, float]:
    """Compute Group E amount timing and concentration features."""

    frame = prepare_minute_day(day)
    if frame.empty:
        return {column: np.nan for column in VOLUME_TIMING_COLUMNS}

    amount = frame["amount"].fillna(0.0)
    total_amount = float(amount.sum())
    times = time_values(frame)

    def _amount_share(start: time, end: time, *, end_inclusive: bool) -> float:
        if total_amount <= EPS:
            return np.nan
        if end_inclusive:
            mask = (times >= start) & (times <= end)
        else:
            mask = (times >= start) & (times < end)
        return float(amount.loc[mask].sum() / total_amount)

    if total_amount > EPS:
        weights = amount / total_amount
        amount_hhi = float(np.square(weights).sum())
        amount_top10_share = float(amount.nlargest(10).sum() / total_amount)
        cumsum = amount.cumsum()
        cross = cumsum >= 0.5 * total_amount
        minutes_to_50pct = float(np.flatnonzero(cross.to_numpy())[0] + 1) if cross.any() else np.nan
    else:
        amount_hhi = np.nan
        amount_top10_share = np.nan
        minutes_to_50pct = np.nan

    positive_volume = frame.loc[frame["volume"] > 0, "volume"]
    volume_kurt = (
        float(positive_volume.kurt())
        if len(positive_volume.dropna()) >= min_volume_kurt_count
        else np.nan
    )

    return {
        "amount_share_open30": _amount_share(time(9, 30), time(10, 0), end_inclusive=False),
        "amount_share_pre_lunch30": _amount_share(time(11, 0), time(11, 30), end_inclusive=True),
        "amount_share_post_lunch30": _amount_share(time(13, 0), time(13, 30), end_inclusive=True),
        "amount_share_close30": _amount_share(time(14, 30), time(15, 0), end_inclusive=True),
        "amount_share_morning": _amount_share(time(9, 30), time(11, 30), end_inclusive=True),
        "amount_share_afternoon": _amount_share(time(13, 0), time(15, 0), end_inclusive=True),
        "amount_hhi": amount_hhi,
        "amount_top10_share": amount_top10_share,
        "volume_kurt_1m": volume_kurt,
        "minutes_to_50pct_amount": minutes_to_50pct,
    }


def compute_vwap_deviation(day: pd.DataFrame) -> dict[str, float]:
    """Compute Group F daily VWAP deviation features."""

    frame = prepare_minute_day(day)
    if frame.empty:
        return {column: np.nan for column in VWAP_DEVIATION_COLUMNS}

    total_volume = float(frame["volume"].fillna(0.0).sum())
    total_amount = float(frame["amount"].fillna(0.0).sum())
    day_vwap = total_amount / total_volume if total_volume > EPS else np.nan
    if not np.isfinite(day_vwap) or day_vwap <= EPS:
        return {column: np.nan for column in VWAP_DEVIATION_COLUMNS}

    day_open = float(frame["open"].iloc[0])
    day_close = float(frame["close"].iloc[-1])
    day_high = float(frame["high"].max())
    day_low = float(frame["low"].min())
    close_dev = (pd.to_numeric(frame["close"], errors="coerce") - day_vwap).std(ddof=0) / day_vwap

    return {
        "vwap_close_dev": (day_close - day_vwap) / day_vwap,
        "vwap_open_dev": (day_open - day_vwap) / day_vwap,
        "vwap_high_dev": (day_high - day_vwap) / day_vwap,
        "vwap_low_dev": (day_low - day_vwap) / day_vwap,
        "vwap_minute_dispersion": float(close_dev),
    }


def compute_batch2_features(day: pd.DataFrame) -> dict[str, float]:
    """Compute second intraday expansion batch: Groups E and F."""

    frame = prepare_minute_day(day)
    output: dict[str, float] = {}
    output.update(compute_volume_timing(frame))
    output.update(compute_vwap_deviation(frame))
    return output


def compute_pv_correlation(
    day: pd.DataFrame,
    *,
    min_count: int = 30,
) -> dict[str, float]:
    """Compute Group G price-volume correlation features."""

    frame = prepare_minute_day(day)
    if frame.empty:
        return {column: np.nan for column in PV_CORRELATION_COLUMNS}

    returns_1m = log_returns(frame["close"])
    aligned = pd.DataFrame(
        {
            "ret": returns_1m,
            "vol": frame["volume"],
            "amt": frame["amount"],
        }
    )
    # Keep first minute (NaN ret) so denominator covers full session and
    # pos+neg+zero shares sum to 1 by construction.
    valid_amt = aligned[(aligned["amt"] > 0) & aligned["ret"].notna()]
    valid_vol = aligned[(aligned["vol"] > 0) & aligned["ret"].notna()]

    total_amount_signed = float(aligned["amt"].fillna(0.0).sum())
    if total_amount_signed <= EPS:
        signed_imb = np.nan
        pos_share = np.nan
        neg_share = np.nan
        zero_share = np.nan
    else:
        pos_amt = float(aligned.loc[aligned["ret"] > 0, "amt"].sum())
        neg_amt = float(aligned.loc[aligned["ret"] < 0, "amt"].sum())
        zero_amt = float(
            aligned.loc[(aligned["ret"] == 0) | aligned["ret"].isna(), "amt"].sum()
        )
        signed_imb = (pos_amt - neg_amt) / total_amount_signed
        pos_share = pos_amt / total_amount_signed
        neg_share = neg_amt / total_amount_signed
        zero_share = zero_amt / total_amount_signed

    if not valid_amt.empty:
        amihud = float((valid_amt["ret"].abs() / valid_amt["amt"]).mean())
    else:
        amihud = np.nan

    return {
        "corr_ret_volume_1m": safe_corr(
            valid_vol["ret"], valid_vol["vol"], min_count=min_count
        ),
        "corr_absret_volume_1m": safe_corr(
            valid_vol["ret"].abs(), valid_vol["vol"], min_count=min_count
        ),
        "signed_amount_imbalance": signed_imb,
        "pos_amount_share": pos_share,
        "neg_amount_share": neg_share,
        "zero_ret_amount_share": zero_share,
        "amihud_intraday": amihud,
    }


def compute_microfreq_timeseries(
    day: pd.DataFrame,
    *,
    min_count: int = 30,
    extreme_ratio: float = EXTREME_RATIO,
) -> dict[str, float]:
    """Compute Group I microfrequency timeseries features."""

    frame = prepare_minute_day(day)
    if frame.empty:
        return {column: np.nan for column in MICROFREQ_COLUMNS}

    returns_1m = log_returns(frame["close"])
    amount_series = frame["amount"]

    ret_ac = safe_autocorr_lag1(returns_1m.dropna(), min_count=min_count)
    amt_ac = safe_autocorr_lag1(amount_series.dropna(), min_count=min_count)

    # avg_gap_between_trades: mean minutes between consecutive vol>0 minutes.
    positive = frame.loc[frame["volume"] > 0]
    if len(positive) < 2:
        avg_gap = np.nan
    else:
        positions = positive.index.to_numpy()
        gaps = np.diff(positions)
        avg_gap = float(gaps.mean()) if gaps.size else np.nan

    # time_at_extremes_share: minutes whose close is within extreme_ratio
    # of the daily range to either day_high or day_low.
    high = float(frame["high"].max())
    low = float(frame["low"].min())
    rng = high - low
    if rng <= EPS or len(frame) == 0:
        extreme_share = np.nan
    else:
        close = pd.to_numeric(frame["close"], errors="coerce")
        near_high = (high - close).abs() / rng < extreme_ratio
        near_low = (close - low).abs() / rng < extreme_ratio
        extreme_share = float((near_high | near_low).mean())

    # acceleration_max: max absolute second-difference of close, normalized by
    # day vwap so it is comparable across price levels.
    total_amount = float(frame["amount"].fillna(0.0).sum())
    total_volume = float(frame["volume"].fillna(0.0).sum())
    day_vwap = total_amount / total_volume if total_volume > EPS else np.nan
    if not np.isfinite(day_vwap) or day_vwap <= EPS or len(frame) < 3:
        accel_max = np.nan
    else:
        close = pd.to_numeric(frame["close"], errors="coerce").to_numpy()
        accel = np.abs(2.0 * close[1:-1] - close[:-2] - close[2:])
        finite = accel[np.isfinite(accel)]
        accel_max = float(finite.max() / day_vwap) if finite.size else np.nan

    return {
        "ret_autocorr_1m_lag1": ret_ac,
        "amount_autocorr_1m_lag1": amt_ac,
        "avg_gap_between_trades": avg_gap,
        "time_at_extremes_share": extreme_share,
        "acceleration_max": accel_max,
    }


def compute_batch3_features(day: pd.DataFrame) -> dict[str, float]:
    """Compute third intraday expansion batch: Groups G and I."""

    frame = prepare_minute_day(day)
    output: dict[str, float] = {}
    output.update(compute_pv_correlation(frame))
    output.update(compute_microfreq_timeseries(frame))
    return output


def compute_microstructure(
    day: pd.DataFrame,
    *,
    up_limit: float | None = None,
    down_limit: float | None = None,
    prev_close: float | None = None,
    min_count: int = 30,
    sign_flip_tick: float = SIGN_FLIP_TICK,
    limit_touch_tol: float = LIMIT_TOUCH_TOL,
) -> dict[str, float]:
    """Compute Group H microstructure features.

    `up_limit` / `down_limit` / `prev_close` are static daily values from the
    canonical prices.parquet supplement. They may be passed directly (per-day
    function) or via constant columns on the input frame (vectorized path
    accepts them via `up_limit`/`down_limit`/`prev_close` columns).
    """

    frame = prepare_minute_day(day)
    if frame.empty:
        return {column: np.nan for column in MICROSTRUCTURE_COLUMNS}

    if up_limit is None and "up_limit" in frame.columns:
        up_limit = (
            float(frame["up_limit"].dropna().iloc[0])
            if frame["up_limit"].notna().any()
            else None
        )
    if down_limit is None and "down_limit" in frame.columns:
        down_limit = (
            float(frame["down_limit"].dropna().iloc[0])
            if frame["down_limit"].notna().any()
            else None
        )
    if prev_close is None and "prev_close" in frame.columns:
        prev_close = (
            float(frame["prev_close"].dropna().iloc[0])
            if frame["prev_close"].notna().any()
            else None
        )

    close = pd.to_numeric(frame["close"], errors="coerce")
    # Use close-based daily extremes for "minutes at extreme" so that intraday
    # spikes confined to a single bar's high/low do not invalidate the count.
    day_high = float(close.max()) if close.notna().any() else np.nan
    day_low = float(close.min()) if close.notna().any() else np.nan

    # Limit touch / open counts.
    limit_up_touch: float
    limit_up_open: float
    limit_dn_touch: float
    limit_dn_open: float
    minutes_at_high: float
    minutes_at_low: float
    if up_limit is not None and np.isfinite(up_limit) and up_limit > 0:
        at_up = (close - up_limit).abs() <= limit_touch_tol
        limit_up_touch = int(at_up.sum())
        # An "open" is a transition from at-limit to not-at-limit within the day.
        flipped = at_up.to_numpy().astype(int)
        opens = int(((flipped[:-1] == 1) & (flipped[1:] == 0)).sum()) if flipped.size > 1 else 0
        limit_up_open = opens
    else:
        limit_up_touch = np.nan
        limit_up_open = np.nan

    if down_limit is not None and np.isfinite(down_limit) and down_limit > 0:
        at_dn = (close - down_limit).abs() <= limit_touch_tol
        limit_dn_touch = int(at_dn.sum())
        flipped_dn = at_dn.to_numpy().astype(int)
        limit_dn_open = (
            int(((flipped_dn[:-1] == 1) & (flipped_dn[1:] == 0)).sum())
            if flipped_dn.size > 1
            else 0
        )
    else:
        limit_dn_touch = np.nan
        limit_dn_open = np.nan

    # Minutes at daily extreme.
    if np.isfinite(day_high):
        minutes_at_high = int(((close - day_high).abs() <= limit_touch_tol).sum())
    else:
        minutes_at_high = np.nan
    if np.isfinite(day_low):
        minutes_at_low = int(((close - day_low).abs() <= limit_touch_tol).sum())
    else:
        minutes_at_low = np.nan

    # Sign flip count over 1m log returns.
    returns_1m = log_returns(frame["close"]).dropna().to_numpy()
    if returns_1m.size < 2:
        sign_flips = np.nan
    else:
        signed = np.where(
            np.abs(returns_1m) <= sign_flip_tick,
            0,
            np.sign(returns_1m),
        )
        nonzero = signed[signed != 0]
        if nonzero.size < 2:
            sign_flips = 0
        else:
            sign_flips = int((np.diff(nonzero) != 0).sum())

    # Max-abs-return z-score (within day).
    valid_returns = log_returns(frame["close"]).dropna()
    if len(valid_returns) < min_count:
        max_abs_z = np.nan
    else:
        std = float(valid_returns.std(ddof=0))
        max_abs_z = (
            float(valid_returns.abs().max() / std)
            if std > EPS
            else np.nan
        )

    # Roll's implied spread proxy.
    if len(valid_returns) < min_count + 1:
        roll_spread = np.nan
    else:
        diffs = valid_returns.diff().dropna()
        # cov(r_t, r_{t-1}) ≈ -1/4 * spread^2 if Roll's model holds.
        cov = float(np.cov(valid_returns.iloc[1:], valid_returns.iloc[:-1], ddof=0)[0, 1])
        roll_spread = float(2.0 * np.sqrt(max(0.0, -cov)))
        del diffs  # keep linter happy

    # Gap fill ratio: (open - close) / (open - prev_close).
    day_open = float(frame["open"].iloc[0])
    day_close = float(close.iloc[-1])
    if (
        prev_close is None
        or not np.isfinite(prev_close)
        or prev_close <= 0
        or abs(day_open - prev_close) < limit_touch_tol
    ):
        gap_fill = np.nan
    else:
        ratio = (day_open - day_close) / (day_open - prev_close)
        gap_fill = float(np.clip(ratio, -3.0, 3.0))

    return {
        "limit_up_touch_count": limit_up_touch,
        "limit_up_open_count": limit_up_open,
        "limit_down_touch_count": limit_dn_touch,
        "limit_down_open_count": limit_dn_open,
        "minutes_at_high_count": minutes_at_high,
        "minutes_at_low_count": minutes_at_low,
        "sign_flip_count": sign_flips,
        "max_abs_return_zscore": max_abs_z,
        "roll_spread_proxy": roll_spread,
        "gap_fill_ratio": gap_fill,
    }


def compute_batch4_features(
    day: pd.DataFrame,
    *,
    up_limit: float | None = None,
    down_limit: float | None = None,
    prev_close: float | None = None,
    min_count: int = 30,
) -> dict[str, float]:
    """Compute fourth intraday expansion batch: Group H microstructure."""

    frame = prepare_minute_day(day)
    return compute_microstructure(
        frame,
        up_limit=up_limit,
        down_limit=down_limit,
        prev_close=prev_close,
        min_count=min_count,
    )


def compute_batch1_feature_frame(
    minutes: pd.DataFrame,
    *,
    min_nonzero_1m: int = 30,
    min_nonzero_5m: int = 6,
) -> pd.DataFrame:
    """Vectorized pure batch1 computation for many asset-date groups.

    This is still a formula function: no parquet IO, no partition assumptions.
    It exists so production ETL does not need to call the per-day golden-test
    function millions of times.
    """

    if minutes.empty:
        return pd.DataFrame(columns=["date", "asset", *BATCH1_FEATURE_COLUMNS])

    frame = prepare_minute_day(minutes)
    frame = frame.sort_values(["asset", "date", "datetime"], kind="mergesort").reset_index(
        drop=True
    )
    keys = ["date", "asset"]
    grouped = frame.groupby(keys, sort=False)
    index = pd.MultiIndex.from_frame(frame[keys].drop_duplicates())
    out = pd.DataFrame(index=index)

    day_open = grouped["open"].first()
    day_close = grouped["close"].last()

    def _safe_return_series(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
        ratio = numerator / denominator.where(denominator.abs() > EPS)
        return (ratio - 1.0).replace([np.inf, -np.inf], np.nan)

    times = time_values(frame)

    def _last_close_at_or_before(boundary: time) -> pd.Series:
        return frame.loc[times <= boundary].groupby(keys, sort=False)["close"].last()

    def _first_open_at_or_after(boundary: time) -> pd.Series:
        return frame.loc[times >= boundary].groupby(keys, sort=False)["open"].first()

    close_0935 = _last_close_at_or_before(time(9, 35))
    close_1000 = _last_close_at_or_before(time(10, 0))
    close_1130 = _last_close_at_or_before(time(11, 30))
    close_1300 = _last_close_at_or_before(time(13, 0))
    open_1300 = _first_open_at_or_after(time(13, 0))
    close_1430 = _last_close_at_or_before(time(14, 30))
    close_1455 = _last_close_at_or_before(time(14, 55))

    out["ret_intraday"] = _safe_return_series(day_close, day_open)
    out["ret_morning"] = _safe_return_series(close_1130, day_open)
    out["ret_afternoon"] = _safe_return_series(day_close, open_1300)
    out["ret_open5"] = _safe_return_series(close_0935, day_open)
    out["ret_close5"] = _safe_return_series(day_close, close_1455)
    out["ret_first30"] = _safe_return_series(close_1000, day_open)
    out["ret_last30"] = _safe_return_series(day_close, close_1430)
    out["ret_mid"] = _safe_return_series(close_1300, close_1000)

    log_close = np.log(frame["close"].where(frame["close"] > 0))
    frame["_r1m"] = (
        log_close - log_close.groupby([frame["date"], frame["asset"]]).shift(1)
    ).replace([np.inf, -np.inf], np.nan)

    r1 = frame["_r1m"]
    out["rv_1m"] = r1.pow(2).groupby([frame["date"], frame["asset"]], sort=False).sum(min_count=1)

    def _sampled_returns(step: int) -> pd.DataFrame:
        pos = grouped.cumcount()
        last_pos = pos.groupby([frame["date"], frame["asset"]], sort=False).transform("max")
        sampled = frame.loc[(pos % step == 0) | (pos == last_pos), keys + ["close"]].copy()
        sampled["_log_close"] = np.log(sampled["close"].where(sampled["close"] > 0))
        sampled["_r"] = (
            sampled["_log_close"]
            - sampled["_log_close"].groupby([sampled["date"], sampled["asset"]]).shift(1)
        ).replace([np.inf, -np.inf], np.nan)
        return sampled

    returns_5m = _sampled_returns(5)
    returns_15m = _sampled_returns(15)

    r5 = returns_5m["_r"]
    r5_group = [returns_5m["date"], returns_5m["asset"]]
    rv_5m = r5.pow(2).groupby(r5_group, sort=False).sum(min_count=1)
    bv_5m = (
        (r5.abs() * r5.abs().groupby(r5_group, sort=False).shift(1))
        .groupby(r5_group, sort=False)
        .sum(min_count=1)
        * (np.pi / 2.0)
    )
    rv_pos_5m = r5.pow(2).where(r5 > 0, 0.0).groupby(r5_group, sort=False).sum(min_count=1)
    rv_neg_5m = r5.pow(2).where(r5 < 0, 0.0).groupby(r5_group, sort=False).sum(min_count=1)

    out["rv_5m"] = rv_5m
    out["rv_15m"] = (
        returns_15m["_r"]
        .pow(2)
        .groupby([returns_15m["date"], returns_15m["asset"]], sort=False)
        .sum(min_count=1)
    )
    out["bv_5m"] = bv_5m
    out["jump_5m"] = (rv_5m - bv_5m).clip(lower=0.0)
    out["rv_pos_5m"] = rv_pos_5m
    out["rv_neg_5m"] = rv_neg_5m
    out["signed_jump"] = (rv_pos_5m - rv_neg_5m) / rv_5m.where(rv_5m > EPS)

    def _segment_rv(start: time | None, end: time | None) -> pd.Series:
        mask = pd.Series(True, index=frame.index)
        if start is not None:
            mask &= times >= start
        if end is not None:
            mask &= times <= end
        segment = frame.loc[mask, keys + ["close"]].copy()
        segment["_log_close"] = np.log(segment["close"].where(segment["close"] > 0))
        segment["_r"] = (
            segment["_log_close"]
            - segment["_log_close"].groupby([segment["date"], segment["asset"]]).shift(1)
        ).replace([np.inf, -np.inf], np.nan)
        return (
            segment["_r"]
            .pow(2)
            .groupby([segment["date"], segment["asset"]], sort=False)
            .sum(min_count=1)
        )

    out["rv_morning"] = _segment_rv(None, time(11, 30))
    out["rv_afternoon"] = _segment_rv(time(13, 0), None)

    def _moments(
        returns: pd.Series,
        group_keys: list[pd.Series],
        min_nonzero: int,
    ) -> tuple[pd.Series, pd.Series]:
        nonzero_count = (
            returns.abs().gt(EPS).groupby(group_keys, sort=False).sum(min_count=1)
        )
        skew = returns.groupby(group_keys, sort=False).skew()
        kurt = returns.groupby(group_keys, sort=False).apply(lambda x: x.kurt())
        return skew.where(nonzero_count >= min_nonzero), kurt.where(nonzero_count >= min_nonzero)

    skew_1m, kurt_1m = _moments(r1, [frame["date"], frame["asset"]], min_nonzero_1m)
    skew_5m, kurt_5m = _moments(r5, r5_group, min_nonzero_5m)
    out["intraday_skew_1m"] = skew_1m
    out["intraday_kurt_1m"] = kurt_1m
    out["intraday_skew_5m"] = skew_5m
    out["intraday_kurt_5m"] = kurt_5m

    result = out.reindex(columns=BATCH1_FEATURE_COLUMNS).reset_index()
    return result.replace([np.inf, -np.inf], np.nan)


def compute_batch2_feature_frame(
    minutes: pd.DataFrame,
    *,
    min_volume_kurt_count: int = 30,
) -> pd.DataFrame:
    """Vectorized pure Batch 2 computation for many asset-date groups."""

    if minutes.empty:
        return pd.DataFrame(columns=["date", "asset", *BATCH2_FEATURE_COLUMNS])

    frame = prepare_minute_day(minutes)
    frame = frame.sort_values(["asset", "date", "datetime"], kind="mergesort").reset_index(
        drop=True
    )
    keys = ["date", "asset"]
    grouped = frame.groupby(keys, sort=False)
    index = pd.MultiIndex.from_frame(frame[keys].drop_duplicates())
    out = pd.DataFrame(index=index)

    amount = frame["amount"].fillna(0.0)
    volume = frame["volume"].fillna(0.0)
    total_amount = amount.groupby([frame["date"], frame["asset"]], sort=False).sum()
    total_volume = volume.groupby([frame["date"], frame["asset"]], sort=False).sum()
    times = time_values(frame)

    def _amount_share(start: time, end: time, *, end_inclusive: bool) -> pd.Series:
        if end_inclusive:
            mask = (times >= start) & (times <= end)
        else:
            mask = (times >= start) & (times < end)
        numerator = (
            amount.loc[mask]
            .groupby([frame.loc[mask, "date"], frame.loc[mask, "asset"]], sort=False)
            .sum()
            .reindex(total_amount.index, fill_value=0.0)
        )
        return numerator / total_amount.where(total_amount > EPS)

    out["amount_share_open30"] = _amount_share(time(9, 30), time(10, 0), end_inclusive=False)
    out["amount_share_pre_lunch30"] = _amount_share(
        time(11, 0), time(11, 30), end_inclusive=True
    )
    out["amount_share_post_lunch30"] = _amount_share(
        time(13, 0), time(13, 30), end_inclusive=True
    )
    out["amount_share_close30"] = _amount_share(time(14, 30), time(15, 0), end_inclusive=True)
    out["amount_share_morning"] = _amount_share(time(9, 30), time(11, 30), end_inclusive=True)
    out["amount_share_afternoon"] = _amount_share(time(13, 0), time(15, 0), end_inclusive=True)

    amount_weight = amount / total_amount.reindex(pd.MultiIndex.from_frame(frame[keys])).to_numpy()
    amount_weight = amount_weight.where(np.isfinite(amount_weight), np.nan)
    out["amount_hhi"] = (
        amount_weight.pow(2).groupby([frame["date"], frame["asset"]], sort=False).sum(min_count=1)
    )
    out["amount_hhi"] = out["amount_hhi"].where(total_amount > EPS)

    top10 = (
        frame.assign(_amount=amount)
        .sort_values(["asset", "date", "_amount"], ascending=[True, True, False], kind="mergesort")
        .groupby(keys, sort=False)
        .head(10)
        .groupby(keys, sort=False)["_amount"]
        .sum()
    )
    out["amount_top10_share"] = top10 / total_amount.where(total_amount > EPS)

    frame["_pos"] = grouped.cumcount() + 1
    frame["_cum_amount"] = amount.groupby([frame["date"], frame["asset"]], sort=False).cumsum()
    frame["_half_amount"] = (
        total_amount.reindex(pd.MultiIndex.from_frame(frame[keys])).to_numpy() * 0.5
    )
    cross = frame["_cum_amount"] >= frame["_half_amount"]
    out["minutes_to_50pct_amount"] = (
        frame.loc[cross].groupby(keys, sort=False)["_pos"].first().astype(float)
    )
    out["minutes_to_50pct_amount"] = out["minutes_to_50pct_amount"].where(total_amount > EPS)

    positive_volume = frame[frame["volume"] > 0].copy()
    vol_count = positive_volume.groupby(keys, sort=False)["volume"].count()
    vol_kurt = positive_volume.groupby(keys, sort=False)["volume"].apply(lambda x: x.kurt())
    out["volume_kurt_1m"] = vol_kurt.where(vol_count >= min_volume_kurt_count)

    day_vwap = total_amount / total_volume.where(total_volume > EPS)
    day_open = grouped["open"].first()
    day_close = grouped["close"].last()
    day_high = grouped["high"].max()
    day_low = grouped["low"].min()
    out["vwap_close_dev"] = (day_close - day_vwap) / day_vwap.where(day_vwap > EPS)
    out["vwap_open_dev"] = (day_open - day_vwap) / day_vwap.where(day_vwap > EPS)
    out["vwap_high_dev"] = (day_high - day_vwap) / day_vwap.where(day_vwap > EPS)
    out["vwap_low_dev"] = (day_low - day_vwap) / day_vwap.where(day_vwap > EPS)

    frame["_day_vwap"] = day_vwap.reindex(pd.MultiIndex.from_frame(frame[keys])).to_numpy()
    frame["_close_dev"] = frame["close"] - frame["_day_vwap"]
    dispersion = frame.groupby(keys, sort=False)["_close_dev"].std(ddof=0) / day_vwap.where(
        day_vwap > EPS
    )
    out["vwap_minute_dispersion"] = dispersion

    result = out.reindex(columns=BATCH2_FEATURE_COLUMNS).reset_index()
    return result.replace([np.inf, -np.inf], np.nan)


def compute_batch12_feature_frame(minutes: pd.DataFrame) -> pd.DataFrame:
    """Compute Batch 1 and Batch 2 features for many asset-date groups."""

    batch1 = compute_batch1_feature_frame(minutes)
    batch2 = compute_batch2_feature_frame(minutes)
    return batch1.merge(batch2, on=["date", "asset"], how="outer", validate="one_to_one")[
        ["date", "asset", *BATCH12_FEATURE_COLUMNS]
    ]


def _grouped_pearson(
    x: pd.Series,
    y: pd.Series,
    keys: list[pd.Series],
    *,
    min_count: int,
) -> pd.Series:
    """Vectorized Pearson correlation per group via aggregated moments."""

    valid = (~x.isna()) & (~y.isna())
    xv = x.where(valid)
    yv = y.where(valid)
    g = pd.concat(keys, axis=1)
    grouper = [g[c] for c in g.columns]

    n = valid.groupby(grouper, sort=False).sum().astype(float)
    sx = xv.groupby(grouper, sort=False).sum(min_count=1)
    sy = yv.groupby(grouper, sort=False).sum(min_count=1)
    sxx = (xv * xv).groupby(grouper, sort=False).sum(min_count=1)
    syy = (yv * yv).groupby(grouper, sort=False).sum(min_count=1)
    sxy = (xv * yv).groupby(grouper, sort=False).sum(min_count=1)

    n_safe = n.where(n > 0)
    mx = sx / n_safe
    my = sy / n_safe
    var_x = sxx / n_safe - mx * mx
    var_y = syy / n_safe - my * my
    cov = sxy / n_safe - mx * my
    denom = np.sqrt(var_x.where(var_x > EPS) * var_y.where(var_y > EPS))
    corr = (cov / denom).where(n >= min_count)
    return corr.replace([np.inf, -np.inf], np.nan)


def compute_batch3_feature_frame(
    minutes: pd.DataFrame,
    *,
    min_count: int = 30,
    extreme_ratio: float = EXTREME_RATIO,
) -> pd.DataFrame:
    """Vectorized Batch 3 (Group G + I) computation for many asset-date groups."""

    if minutes.empty:
        return pd.DataFrame(columns=["date", "asset", *BATCH3_FEATURE_COLUMNS])

    frame = prepare_minute_day(minutes)
    frame = frame.sort_values(["asset", "date", "datetime"], kind="mergesort").reset_index(
        drop=True
    )

    keys = ["date", "asset"]
    grouper_cols = [frame["date"], frame["asset"]]
    grouped = frame.groupby(keys, sort=False)
    index = pd.MultiIndex.from_frame(frame[keys].drop_duplicates())
    out = pd.DataFrame(index=index)

    log_close = np.log(frame["close"].where(frame["close"] > 0))
    r1 = (log_close - log_close.groupby(grouper_cols).shift(1)).replace(
        [np.inf, -np.inf], np.nan
    )
    amount = frame["amount"].fillna(0.0)
    volume = frame["volume"].fillna(0.0)
    ret_valid = r1.notna()

    # ---- Group G ----
    total_amount = amount.groupby(grouper_cols, sort=False).sum()
    pos_amt = (amount.where(r1 > 0, 0.0)).groupby(grouper_cols, sort=False).sum()
    neg_amt = (amount.where(r1 < 0, 0.0)).groupby(grouper_cols, sort=False).sum()
    zero_amt = (amount.where((r1 == 0) | r1.isna(), 0.0)).groupby(grouper_cols, sort=False).sum()

    out["corr_ret_volume_1m"] = _grouped_pearson(
        r1.where(volume > 0), volume.where(volume > 0), grouper_cols, min_count=min_count
    )
    out["corr_absret_volume_1m"] = _grouped_pearson(
        r1.abs().where(volume > 0),
        volume.where(volume > 0),
        grouper_cols,
        min_count=min_count,
    )

    total_safe = total_amount.where(total_amount > EPS)
    out["signed_amount_imbalance"] = (pos_amt - neg_amt) / total_safe
    out["pos_amount_share"] = pos_amt / total_safe
    out["neg_amount_share"] = neg_amt / total_safe
    out["zero_ret_amount_share"] = zero_amt / total_safe

    valid_amount = (amount > 0) & ret_valid
    abs_over_amt = (r1.abs() / amount.where(amount > 0)).where(valid_amount)
    valid_count = valid_amount.groupby(grouper_cols, sort=False).sum()
    valid_count_safe = valid_count.where(valid_count > 0)
    amihud_sum = abs_over_amt.groupby(grouper_cols, sort=False).sum(min_count=1)
    out["amihud_intraday"] = amihud_sum / valid_count_safe

    # ---- Group I ----
    r1_lag = r1.groupby(grouper_cols).shift(1)
    out["ret_autocorr_1m_lag1"] = _grouped_pearson(
        r1, r1_lag, grouper_cols, min_count=min_count
    )
    amount_lag = amount.groupby(grouper_cols).shift(1)
    out["amount_autocorr_1m_lag1"] = _grouped_pearson(
        amount, amount_lag, grouper_cols, min_count=min_count
    )

    # avg_gap_between_trades: mean of consecutive position diffs for vol>0 rows
    pos = grouped.cumcount()
    pos_active = pos.where(volume > 0)
    pos_active_lag = pos_active.groupby(grouper_cols).shift(1)
    gaps = (pos_active - pos_active_lag).where(volume > 0)
    gap_count = gaps.notna().groupby(grouper_cols, sort=False).sum().astype(float)
    gap_sum = gaps.groupby(grouper_cols, sort=False).sum(min_count=1)
    out["avg_gap_between_trades"] = gap_sum / gap_count.where(gap_count > 0)

    # time_at_extremes_share: minutes near day_high (max of high column)
    # or day_low (min of low column). Matches per-day semantics.
    close = frame["close"]
    day_high = frame["high"].groupby(grouper_cols, sort=False).transform("max")
    day_low = frame["low"].groupby(grouper_cols, sort=False).transform("min")
    day_range = day_high - day_low
    near_high = (day_high - close).abs() / day_range.where(day_range > EPS) < extreme_ratio
    near_low = (close - day_low).abs() / day_range.where(day_range > EPS) < extreme_ratio
    near_extreme = (near_high | near_low).astype(float)
    near_extreme = near_extreme.where(day_range > EPS)
    extreme_share = near_extreme.groupby(grouper_cols, sort=False).mean()
    out["time_at_extremes_share"] = extreme_share

    # acceleration_max: max(|2 c_t - c_{t-1} - c_{t+1}|) normalized by day vwap
    c_lag = close.groupby(grouper_cols).shift(1)
    c_lead = close.groupby(grouper_cols).shift(-1)
    accel = (2.0 * close - c_lag - c_lead).abs()
    accel_max = accel.groupby(grouper_cols, sort=False).max()
    total_volume = volume.groupby(grouper_cols, sort=False).sum()
    day_vwap = total_amount / total_volume.where(total_volume > EPS)
    out["acceleration_max"] = accel_max / day_vwap.where(day_vwap > EPS)

    result = out.reindex(columns=BATCH3_FEATURE_COLUMNS).reset_index()
    return result.replace([np.inf, -np.inf], np.nan)


def compute_batch4_feature_frame(
    minutes: pd.DataFrame,
    *,
    daily_meta: pd.DataFrame | None = None,
    min_count: int = 30,
    sign_flip_tick: float = SIGN_FLIP_TICK,
    limit_touch_tol: float = LIMIT_TOUCH_TOL,
) -> pd.DataFrame:
    """Vectorized Batch 4 (Group H) microstructure features.

    `daily_meta` must include columns ``date``, ``asset``, ``up_limit``,
    ``down_limit``, ``prev_close``. When omitted, the function falls back to
    ``minutes`` having those columns (constant per group). The contract calls
    for prices.parquet to be the source of truth.
    """

    if minutes.empty:
        return pd.DataFrame(columns=["date", "asset", *BATCH4_FEATURE_COLUMNS])

    frame = prepare_minute_day(minutes)
    frame = frame.sort_values(["asset", "date", "datetime"], kind="mergesort").reset_index(
        drop=True
    )

    keys = ["date", "asset"]
    grouper_cols = [frame["date"], frame["asset"]]
    index = pd.MultiIndex.from_frame(frame[keys].drop_duplicates())
    out = pd.DataFrame(index=index)

    # Resolve daily meta (broadcast to each minute).
    if daily_meta is None:
        for column in ("up_limit", "down_limit", "prev_close"):
            if column not in frame.columns:
                frame[column] = np.nan
        meta_frame = (
            frame.groupby(keys, sort=False)[["up_limit", "down_limit", "prev_close"]]
            .first()
        )
    else:
        m = daily_meta[["date", "asset", "up_limit", "down_limit", "prev_close"]].copy()
        m["date"] = m["date"].astype(str)
        m["asset"] = m["asset"].astype(str)
        meta_frame = m.set_index(keys)

    meta_aligned = meta_frame.reindex(index)
    up_limit = meta_aligned["up_limit"]
    down_limit = meta_aligned["down_limit"]
    prev_close = meta_aligned["prev_close"]

    close = pd.to_numeric(frame["close"], errors="coerce")
    log_close = np.log(close.where(close > 0))
    r1 = (log_close - log_close.groupby(grouper_cols).shift(1)).replace(
        [np.inf, -np.inf], np.nan
    )

    # Broadcast static daily values to each minute via map.
    up_per_minute = pd.Series(
        meta_aligned["up_limit"].reindex(pd.MultiIndex.from_frame(frame[keys])).to_numpy(),
        index=frame.index,
    )
    down_per_minute = pd.Series(
        meta_aligned["down_limit"].reindex(pd.MultiIndex.from_frame(frame[keys])).to_numpy(),
        index=frame.index,
    )

    at_up = (close - up_per_minute).abs() <= limit_touch_tol
    at_dn = (close - down_per_minute).abs() <= limit_touch_tol
    out["limit_up_touch_count"] = (
        at_up.where(up_per_minute.notna()).groupby(grouper_cols, sort=False).sum(min_count=1)
    ).astype("Float64")
    out["limit_up_touch_count"] = out["limit_up_touch_count"].where(up_limit.notna())
    out["limit_down_touch_count"] = (
        at_dn.where(down_per_minute.notna()).groupby(grouper_cols, sort=False).sum(min_count=1)
    ).astype("Float64")
    out["limit_down_touch_count"] = out["limit_down_touch_count"].where(down_limit.notna())

    # Limit open transitions: at_t == 1 AND at_{t+1} == 0
    at_up_int = at_up.astype("Int64")
    at_up_lead = at_up_int.groupby(grouper_cols).shift(-1)
    open_up = ((at_up_int == 1) & (at_up_lead == 0)).astype("Float64")
    out["limit_up_open_count"] = (
        open_up.where(up_per_minute.notna()).groupby(grouper_cols, sort=False).sum(min_count=1)
    ).where(up_limit.notna())

    at_dn_int = at_dn.astype("Int64")
    at_dn_lead = at_dn_int.groupby(grouper_cols).shift(-1)
    open_dn = ((at_dn_int == 1) & (at_dn_lead == 0)).astype("Float64")
    out["limit_down_open_count"] = (
        open_dn.where(down_per_minute.notna()).groupby(grouper_cols, sort=False).sum(min_count=1)
    ).where(down_limit.notna())

    # Minutes at close-based day_high / day_low.
    day_high = close.groupby(grouper_cols, sort=False).transform("max")
    day_low = close.groupby(grouper_cols, sort=False).transform("min")
    out["minutes_at_high_count"] = (
        ((close - day_high).abs() <= limit_touch_tol)
        .astype("Float64")
        .groupby(grouper_cols, sort=False)
        .sum(min_count=1)
    )
    out["minutes_at_low_count"] = (
        ((close - day_low).abs() <= limit_touch_tol)
        .astype("Float64")
        .groupby(grouper_cols, sort=False)
        .sum(min_count=1)
    )

    # Sign flip count: count direction transitions on |ret| > tick.
    sign = np.where(np.abs(r1) <= sign_flip_tick, 0, np.sign(r1))
    sign_series = pd.Series(sign, index=frame.index)
    sign_lag = sign_series.groupby(grouper_cols).shift(1)
    flips = ((sign_series != 0) & (sign_lag != 0) & (sign_series != sign_lag)).astype("Float64")
    out["sign_flip_count"] = flips.groupby(grouper_cols, sort=False).sum(min_count=1)

    # max_abs_return_zscore: per-group max(|r|) / std(r), gated by min_count.
    abs_r = r1.abs()
    n_valid = r1.notna().groupby(grouper_cols, sort=False).sum().astype(float)
    max_abs = abs_r.groupby(grouper_cols, sort=False).max()
    sum_r = r1.groupby(grouper_cols, sort=False).sum(min_count=1)
    sum_r2 = r1.pow(2).groupby(grouper_cols, sort=False).sum(min_count=1)
    n_safe = n_valid.where(n_valid > 0)
    var_r = sum_r2 / n_safe - (sum_r / n_safe).pow(2)
    std_r = np.sqrt(var_r.where(var_r > EPS))
    out["max_abs_return_zscore"] = (max_abs / std_r).where(n_valid >= min_count)

    # roll_spread_proxy: 2 * sqrt(max(0, -cov(r_t, r_{t-1}))).
    r_lag = r1.groupby(grouper_cols).shift(1)
    valid_pair = r1.notna() & r_lag.notna()
    prod = (r1 * r_lag).where(valid_pair)
    n_pair = valid_pair.groupby(grouper_cols, sort=False).sum().astype(float)
    sum_prod = prod.groupby(grouper_cols, sort=False).sum(min_count=1)
    # population covariance
    sum_a = r1.where(valid_pair).groupby(grouper_cols, sort=False).sum(min_count=1)
    sum_b = r_lag.where(valid_pair).groupby(grouper_cols, sort=False).sum(min_count=1)
    n_pair_safe = n_pair.where(n_pair > 0)
    cov = sum_prod / n_pair_safe - (sum_a / n_pair_safe) * (sum_b / n_pair_safe)
    roll = 2.0 * np.sqrt((-cov).clip(lower=0.0))
    out["roll_spread_proxy"] = roll.where(n_pair >= min_count)

    # gap_fill_ratio: (open - close) / (open - prev_close); NaN when no gap.
    day_open = frame.groupby(keys, sort=False)["open"].first()
    day_close = frame.groupby(keys, sort=False)["close"].last()
    gap_denom = day_open - prev_close
    no_gap = gap_denom.abs() < limit_touch_tol
    ratio = (day_open - day_close) / gap_denom.where(~no_gap)
    out["gap_fill_ratio"] = ratio.clip(-3.0, 3.0).where(prev_close.notna() & (~no_gap))

    result = out.reindex(columns=BATCH4_FEATURE_COLUMNS).reset_index()
    return result.replace([np.inf, -np.inf], np.nan)


def compute_batch1234_feature_frame(
    minutes: pd.DataFrame,
    *,
    daily_meta: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute Batch 1, 2, 3, and 4 feature columns for many asset-date groups."""

    batch12 = compute_batch12_feature_frame(minutes)
    batch3 = compute_batch3_feature_frame(minutes)
    batch4 = compute_batch4_feature_frame(minutes, daily_meta=daily_meta)
    out = (
        batch12.merge(batch3, on=["date", "asset"], how="outer", validate="one_to_one")
        .merge(batch4, on=["date", "asset"], how="outer", validate="one_to_one")
    )
    return out[["date", "asset", *BATCH1234_FEATURE_COLUMNS]]
