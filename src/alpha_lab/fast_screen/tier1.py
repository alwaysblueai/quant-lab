"""Tier-1 fast-screen pipeline.

Computes *only* the data needed for the 10 metric cards + 4 charts + verdict,
plus a lightweight integrity check. Deliberately does NOT import Tier-2
modules (random-null, kfold, conditional-ic, capacity, level2, ...).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from alpha_lab.costs import cost_adjusted_long_short
from alpha_lab.decay import compute_ic_decay, estimate_ic_half_life
from alpha_lab.experiment import ExperimentSummary, run_factor_experiment

from .contracts import (
    CORE_METRIC_KEYS,
    ChartSeries,
    FastScreenResult,
    MetricCard,
    MetricStatus,
    Verdict,
    metric_card,
)
from .gating import evaluate_gates

DEFAULT_DECAY_HORIZONS: tuple[int, ...] = (1, 2, 3, 5, 10, 20)
DEFAULT_ANNUALIZATION: int = 252
DEFAULT_COVERAGE_FULL_THRESHOLD: int = 250  # eval days above which coverage is "full"


@dataclass(frozen=True)
class Tier1Inputs:
    """Minimal, fully-prepared inputs for Tier-1.

    ``factor_df`` is the canonical long-form ``[date, asset, factor, value]``
    table, already neutralised / coverage-gated upstream. ``prices`` is the
    long-form ``[date, asset, close]`` panel scoped to the same universe.
    ``horizon`` is the forward-return holding horizon in rows.
    """

    factor_name: str
    factor_df: pd.DataFrame
    prices: pd.DataFrame
    horizon: int = 1
    n_quantiles: int = 5
    cost_rate: float = 0.0
    universe: str = "default"
    frequency: str = "daily"
    annualization: int = DEFAULT_ANNUALIZATION
    integrity_passed: bool = True


def run_tier1(inputs: Tier1Inputs, *, run_id: str | None = None) -> FastScreenResult:
    """Execute the Tier-1 pipeline and produce a ``FastScreenResult``.

    Failure mode: data issues surfacing as empty frames produce metric cards
    with status ``MISSING_INPUT`` or ``PARTIAL`` rather than raising; hard
    errors (bad schema, wrong dtypes) are propagated from the underlying
    primitives.
    """
    result = run_factor_experiment(
        inputs.prices,
        lambda _p: inputs.factor_df.copy(),
        horizon=inputs.horizon,
        n_quantiles=inputs.n_quantiles,
    )
    summary = result.summary

    decay_df = compute_ic_decay(
        factor_df=inputs.factor_df,
        prices_df=inputs.prices,
        horizons=DEFAULT_DECAY_HORIZONS,
    )
    half_life_payload = estimate_ic_half_life(decay_df)

    ls_net_df = _cost_adjusted(
        result.long_short_df, result.long_short_turnover_df, inputs.cost_rate
    )
    ls_sharpe_net, ls_mdd, ls_cum = _ls_stats(ls_net_df, inputs.annualization)

    monotonicity = _group_monotonicity(result.quantile_returns_df, inputs.n_quantiles)

    metrics = _build_metrics(
        summary=summary,
        rank_ic_df=result.rank_ic_df,
        half_life=half_life_payload,
        horizon=inputs.horizon,
        ls_sharpe_net=ls_sharpe_net,
        ls_mdd=ls_mdd,
        monotonicity=monotonicity,
        n_eval_dates=result.n_eval_dates,
        mean_eval_assets=summary.mean_eval_assets_per_date,
        coverage_full_threshold=DEFAULT_COVERAGE_FULL_THRESHOLD,
    )

    charts = _build_charts(
        rolling_rank_ic=result.rank_ic_df,
        decay_df=decay_df,
        quantile_returns=result.quantile_returns_df,
        ls_cum_net=ls_cum,
        n_quantiles=inputs.n_quantiles,
    )

    window = _date_window(inputs.factor_df)
    inputs_hash = _hash_inputs(inputs)

    stub = FastScreenResult(
        factor_name=inputs.factor_name,
        run_id=run_id or inputs_hash,
        universe=inputs.universe,
        frequency=inputs.frequency,
        window=window,
        metrics=metrics,
        charts=charts,
        verdict=Verdict(status="pass", triggered_rules=[], next_step=""),
        inputs_hash=inputs_hash,
        generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    verdict = evaluate_gates(stub, integrity_passed=inputs.integrity_passed)
    return FastScreenResult(
        factor_name=stub.factor_name,
        run_id=stub.run_id,
        universe=stub.universe,
        frequency=stub.frequency,
        window=stub.window,
        metrics=stub.metrics,
        charts=stub.charts,
        verdict=verdict,
        inputs_hash=stub.inputs_hash,
        generated_at=stub.generated_at,
    )


def _cost_adjusted(
    long_short_df: pd.DataFrame,
    turnover_df: pd.DataFrame,
    cost_rate: float,
) -> pd.DataFrame:
    if long_short_df.empty:
        return pd.DataFrame(
            columns=["date", "factor", "long_short_return", "turnover", "adjusted_return"]
        )
    if cost_rate <= 0:
        out = long_short_df.copy()
        out["adjusted_return"] = out["long_short_return"]
        out["turnover"] = float("nan")
        return out[["date", "factor", "long_short_return", "turnover", "adjusted_return"]]
    return cost_adjusted_long_short(long_short_df, turnover_df, cost_rate=cost_rate)


def _ls_stats(
    ls_net_df: pd.DataFrame,
    annualization: int,
) -> tuple[float, float, pd.DataFrame]:
    """Return (net Sharpe, max drawdown, cumulative-NAV dataframe)."""
    if ls_net_df.empty or "adjusted_return" not in ls_net_df.columns:
        empty = pd.DataFrame(columns=["date", "nav"])
        return float("nan"), float("nan"), empty

    rets = pd.to_numeric(ls_net_df["adjusted_return"], errors="coerce").dropna()
    if rets.empty:
        empty = pd.DataFrame(columns=["date", "nav"])
        return float("nan"), float("nan"), empty

    mean = float(rets.mean())
    std = float(rets.std(ddof=1)) if len(rets) > 1 else float("nan")
    sharpe = (
        float(mean / std * np.sqrt(annualization))
        if (std is not None and std > 0)
        else float("nan")
    )

    nav_df = ls_net_df[["date", "adjusted_return"]].copy().sort_values("date", kind="mergesort")
    nav_df["adjusted_return"] = nav_df["adjusted_return"].fillna(0.0)
    nav_df["nav"] = (1.0 + nav_df["adjusted_return"]).cumprod()
    peak = nav_df["nav"].cummax()
    dd = nav_df["nav"] / peak - 1.0
    mdd = float(dd.min()) if not dd.empty else float("nan")
    return sharpe, mdd, nav_df[["date", "nav"]]


def _group_monotonicity(quantile_returns_df: pd.DataFrame, n_quantiles: int) -> dict[str, float]:
    if quantile_returns_df.empty:
        return {"q_top_minus_bottom": float("nan"), "kendall_tau": float("nan")}
    by_q = quantile_returns_df.groupby("quantile")["mean_return"].mean().sort_index()
    if by_q.empty:
        return {"q_top_minus_bottom": float("nan"), "kendall_tau": float("nan")}
    top = float(by_q.iloc[-1])
    bottom = float(by_q.iloc[0])
    tau = _kendall_tau(list(range(len(by_q))), list(by_q.values))
    return {"q_top_minus_bottom": top - bottom, "kendall_tau": tau}


def _kendall_tau(x: list[float], y: list[float]) -> float:
    """Tie-free Kendall tau-a over small samples (n<=10)."""
    n = len(x)
    if n < 2:
        return float("nan")
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[j] - x[i]
            dy = y[j] - y[i]
            if dx == 0 or dy == 0:
                continue
            if (dx > 0) == (dy > 0):
                concordant += 1
            else:
                discordant += 1
    denom = 0.5 * n * (n - 1)
    if denom == 0:
        return float("nan")
    return (concordant - discordant) / denom


def _build_metrics(
    *,
    summary: ExperimentSummary,
    rank_ic_df: pd.DataFrame,
    half_life: dict[str, object],
    horizon: int,
    ls_sharpe_net: float,
    ls_mdd: float,
    monotonicity: dict[str, float],
    n_eval_dates: int,
    mean_eval_assets: float,
    coverage_full_threshold: int,
) -> list[MetricCard]:
    cards: list[MetricCard] = []

    mean_rank_ic = float(summary.mean_rank_ic)
    cards.append(
        metric_card(
            "mean_rank_ic",
            "Mean RankIC",
            None if _nan(mean_rank_ic) else mean_rank_ic,
            MetricStatus.COMPUTED if not _nan(mean_rank_ic) else MetricStatus.PARTIAL,
        )
    )

    rank_ic_ir = _rank_ic_ir(rank_ic_df)
    cards.append(
        metric_card(
            "rank_ic_ir",
            "RankIC IR",
            None if _nan(rank_ic_ir) else rank_ic_ir,
            MetricStatus.COMPUTED if not _nan(rank_ic_ir) else MetricStatus.PARTIAL,
        )
    )

    pos_ratio = float(summary.rank_ic_positive_rate)
    cards.append(
        metric_card(
            "ic_positive_ratio",
            "RankIC > 0 %",
            None if _nan(pos_ratio) else pos_ratio,
            MetricStatus.COMPUTED if not _nan(pos_ratio) else MetricStatus.PARTIAL,
            unit="ratio",
        )
    )

    tau = monotonicity.get("kendall_tau", float("nan"))
    q_diff = monotonicity.get("q_top_minus_bottom", float("nan"))
    mono_secondary: dict[str, float | int | str] = {}
    tau_value = _f_or_none(tau)
    if tau_value is not None:
        mono_secondary["kendall_tau"] = tau_value
    cards.append(
        metric_card(
            "group_monotonicity",
            "Q5-Q1 Spread",
            None if _nan(q_diff) else float(q_diff),
            MetricStatus.COMPUTED if not _nan(q_diff) else MetricStatus.PARTIAL,
            secondary=mono_secondary,
        )
    )

    hl_status = str(half_life.get("ic_half_life_status", "unavailable"))
    hl_value = half_life.get("ic_half_life_horizon", float("nan"))
    hl_value_f = _f_or_none(hl_value)
    hl_ms = _half_life_status(hl_status, hl_value_f, horizon)
    cards.append(
        metric_card(
            "ic_half_life",
            "IC Half-Life",
            hl_value_f,
            hl_ms,
            unit="d",
            note=_half_life_note(hl_status),
        )
    )

    turnover = float(summary.mean_long_short_turnover)
    cards.append(
        metric_card(
            "turnover",
            "Turnover (1-way)",
            None if _nan(turnover) else turnover,
            MetricStatus.COMPUTED if not _nan(turnover) else MetricStatus.PARTIAL,
        )
    )

    cov_status = (
        MetricStatus.COMPUTED if n_eval_dates >= coverage_full_threshold else MetricStatus.PARTIAL
    )
    coverage_secondary: dict[str, float | int | str] = {
        "effective_days": int(n_eval_dates),
    }
    mean_assets = _f_or_none(mean_eval_assets)
    if mean_assets is not None:
        coverage_secondary["avg_n_assets"] = mean_assets
    cards.append(
        metric_card(
            "coverage",
            "Coverage",
            float(n_eval_dates),
            cov_status,
            unit="days",
            secondary=coverage_secondary,
        )
    )

    cards.append(
        metric_card(
            "ls_sharpe_net",
            "LS Sharpe (net)",
            None if _nan(ls_sharpe_net) else ls_sharpe_net,
            MetricStatus.COMPUTED if not _nan(ls_sharpe_net) else MetricStatus.PARTIAL,
        )
    )

    ic_t_stat = float(getattr(summary, "ic_t_stat", float("nan")))
    cards.append(
        metric_card(
            "ic_t_stat",
            "IC t-stat",
            None if _nan(ic_t_stat) else ic_t_stat,
            MetricStatus.COMPUTED if not _nan(ic_t_stat) else MetricStatus.PARTIAL,
        )
    )

    cards.append(
        metric_card(
            "max_drawdown",
            "Max Drawdown",
            None if _nan(ls_mdd) else ls_mdd,
            MetricStatus.COMPUTED if not _nan(ls_mdd) else MetricStatus.PARTIAL,
        )
    )

    assert [c.key for c in cards] == list(CORE_METRIC_KEYS), "metric order drift"
    return cards


def _build_charts(
    *,
    rolling_rank_ic: pd.DataFrame,
    decay_df: pd.DataFrame,
    quantile_returns: pd.DataFrame,
    ls_cum_net: pd.DataFrame,
    n_quantiles: int,
) -> list[ChartSeries]:
    charts: list[ChartSeries] = []

    if not rolling_rank_ic.empty:
        sorted_ric = rolling_rank_ic.sort_values("date", kind="mergesort")
        window = min(63, max(5, len(sorted_ric) // 4 or 5))
        rolling = sorted_ric["rank_ic"].rolling(window, min_periods=max(3, window // 3)).mean()
        charts.append(
            ChartSeries(
                key="rolling_rank_ic",
                label=f"Rolling RankIC ({window}d)",
                kind="line",
                x=[_iso(d) for d in sorted_ric["date"]],
                y=[_f_or_none(v) for v in rolling.tolist()],
                status=MetricStatus.COMPUTED,
                extras={"window": window},
            )
        )
    else:
        charts.append(
            ChartSeries(
                key="rolling_rank_ic",
                label="Rolling RankIC",
                kind="line",
                x=[],
                y=[],
                status=MetricStatus.PARTIAL,
                note="no IC data",
            )
        )

    if not decay_df.empty:
        charts.append(
            ChartSeries(
                key="ic_decay",
                label="IC Decay",
                kind="line",
                x=[int(h) for h in decay_df["horizon"].tolist()],
                y=[_f_or_none(v) for v in decay_df["mean_ic"].tolist()],
                status=MetricStatus.COMPUTED,
                extras={
                    "mean_rank_ic": [_f_or_none(v) for v in decay_df["mean_rank_ic"].tolist()],
                },
            )
        )
    else:
        charts.append(
            ChartSeries(
                key="ic_decay",
                label="IC Decay",
                kind="line",
                x=[],
                y=[],
                status=MetricStatus.PARTIAL,
                note="no decay data",
            )
        )

    if not quantile_returns.empty:
        by_q = quantile_returns.groupby("quantile")["mean_return"].mean().sort_index()
        charts.append(
            ChartSeries(
                key="group_mean_return",
                label=f"Group Mean Return (Q1..Q{n_quantiles})",
                kind="bar",
                x=[int(q) for q in by_q.index.tolist()],
                y=[_f_or_none(v) for v in by_q.values.tolist()],
                status=MetricStatus.COMPUTED,
            )
        )
    else:
        charts.append(
            ChartSeries(
                key="group_mean_return",
                label="Group Mean Return",
                kind="bar",
                x=[],
                y=[],
                status=MetricStatus.PARTIAL,
                note="no quantile data",
            )
        )

    if not ls_cum_net.empty:
        charts.append(
            ChartSeries(
                key="ls_cum_nav_net",
                label="LS Cumulative NAV (net)",
                kind="line",
                x=[_iso(d) for d in ls_cum_net["date"]],
                y=[_f_or_none(v) for v in ls_cum_net["nav"].tolist()],
                status=MetricStatus.COMPUTED,
            )
        )
    else:
        charts.append(
            ChartSeries(
                key="ls_cum_nav_net",
                label="LS Cumulative NAV (net)",
                kind="line",
                x=[],
                y=[],
                status=MetricStatus.PARTIAL,
                note="no LS data",
            )
        )

    return charts


def _rank_ic_ir(rank_ic_df: pd.DataFrame) -> float:
    if rank_ic_df.empty or "rank_ic" not in rank_ic_df.columns:
        return float("nan")
    vals = pd.to_numeric(rank_ic_df["rank_ic"], errors="coerce").dropna()
    if len(vals) < 2:
        return float("nan")
    std = float(vals.std(ddof=1))
    if std <= 0:
        return float("nan")
    return float(vals.mean() / std)


def _half_life_status(raw_status: str, value: float | None, horizon: int) -> MetricStatus:
    if raw_status == "estimated":
        return MetricStatus.COMPUTED
    if raw_status == "not_reached":
        return MetricStatus.PARTIAL
    if value is None and horizon >= 20:
        return MetricStatus.NOT_APPLICABLE
    return MetricStatus.PARTIAL


def _half_life_note(raw_status: str) -> str:
    if raw_status == "not_reached":
        return "baseline never halved within measured horizons"
    if raw_status == "unavailable":
        return "insufficient decay observations"
    return ""


def _date_window(factor_df: pd.DataFrame) -> dict[str, str]:
    if factor_df.empty:
        return {"start": "", "end": ""}
    dates = pd.to_datetime(factor_df["date"], errors="coerce").dropna()
    if dates.empty:
        return {"start": "", "end": ""}
    return {
        "start": dates.min().date().isoformat(),
        "end": dates.max().date().isoformat(),
    }


def _hash_inputs(inputs: Tier1Inputs) -> str:
    parts: list[str] = [
        inputs.factor_name,
        inputs.universe,
        str(inputs.horizon),
        str(inputs.n_quantiles),
        str(round(float(inputs.cost_rate), 8)),
    ]
    if not inputs.factor_df.empty:
        dates = pd.to_datetime(inputs.factor_df["date"], errors="coerce").dropna()
        if not dates.empty:
            parts.append(dates.min().date().isoformat())
            parts.append(dates.max().date().isoformat())
        parts.append(str(len(inputs.factor_df)))
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]


def _iso(value: object) -> str:
    ts = pd.Timestamp(value)
    if pd.isna(ts):
        return ""
    return str(ts.strftime("%Y-%m-%d"))


def _f_or_none(value: object) -> float | None:
    if isinstance(value, bool):
        f = float(value)
    elif isinstance(value, (int, float)):
        f = float(value)
    elif isinstance(value, str):
        token = value.strip()
        if not token:
            return None
        try:
            f = float(token)
        except ValueError:
            return None
    else:
        return None
    if f != f:  # NaN
        return None
    if f in (float("inf"), float("-inf")):
        return None
    return f


def _nan(value: object) -> bool:
    parsed = _f_or_none(value)
    if parsed is None:
        return True
    f = parsed
    return f != f
