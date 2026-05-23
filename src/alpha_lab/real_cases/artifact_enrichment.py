from __future__ import annotations

import math
from collections.abc import Mapping

import pandas as pd

from alpha_lab.reporting._shared import (
    annualized_from_series as _annualized_from_series,
)
from alpha_lab.reporting._shared import (
    periods_per_year as _periods_per_year,
)

_BACKTEST_DERIVED_FIELDS: tuple[str, ...] = (
    "annualized_return",
    "annualized_volatility",
    "sortino",
    "max_drawdown",
    "calmar",
    "rolling_sharpe",
    "rolling_drawdown",
    "nav_points",
    "monthly_return_table",
    "drawdown_table",
    "subperiod_analysis",
    "regime_analysis",
)


def build_portfolio_recipe_controls(
    *,
    metrics_for_payload: Mapping[str, object],
    portfolio_validation_payload: Mapping[str, object],
) -> dict[str, str]:
    """Build canonical portfolio recipe control fields from write-time artifacts."""

    portfolio_validation_metrics = _as_object(
        portfolio_validation_payload.get("portfolio_validation_metrics")
    )
    protocol_settings = _as_object(portfolio_validation_metrics.get("protocol_settings"))
    concentration = _as_object(
        portfolio_validation_metrics.get("concentration_exposure_diagnostics")
    )

    snapshot = _as_object(metrics_for_payload.get("research_evaluation_snapshot"))
    thresholds = _as_object(snapshot.get("level2_portfolio_validation"))
    turnover_warn = _safe_float(thresholds.get("max_mean_turnover_warn"))
    if turnover_warn is None:
        turnover_penalty_settings = "N/A"
    else:
        turnover_penalty_settings = f"warn if mean turnover > {turnover_warn:.2f}"

    transaction_cost_rate = _safe_float(metrics_for_payload.get("transaction_cost_one_way_rate"))
    cost_grid_payload = protocol_settings.get("transaction_cost_sensitivity")
    cost_grid: list[str] = []
    if isinstance(cost_grid_payload, list):
        for item in cost_grid_payload:
            if isinstance(item, bool):
                continue
            text = str(item).strip()
            if text:
                cost_grid.append(text)
    one_way_text = _fmt_number(transaction_cost_rate)
    if cost_grid:
        transaction_cost_assumptions = f"one-way={one_way_text}; grid={','.join(cost_grid)}"
    else:
        transaction_cost_assumptions = f"one-way={one_way_text}"

    max_abs_weight = _safe_float(concentration.get("max_abs_weight_mean"))
    effective_names = _safe_float(concentration.get("effective_names_mean"))
    if max_abs_weight is None and effective_names is None:
        position_limits = "N/A"
    else:
        position_limits = (
            f"max|w|~{_fmt_number(max_abs_weight)}; effective names~{_fmt_number(effective_names)}"
        )

    return {
        "turnover_penalty_settings": turnover_penalty_settings,
        "transaction_cost_assumptions": transaction_cost_assumptions,
        "position_limits": position_limits,
    }


def build_backtest_summary_payload(
    *,
    group_returns_df: pd.DataFrame,
    rebalance_frequency: str,
    metrics_for_payload: Mapping[str, object],
    label_horizon: int = 1,
) -> tuple[dict[str, object], list[str]]:
    """Build canonical backtest summary fields from write-time group-returns data.

    ``label_horizon`` is the forward-return label horizon in trading days
    (e.g., a 5-day forward return uses ``label_horizon=5``).  When labels
    overlap (horizon > 1) the long-short series cannot be daily-compounded
    without over-counting; both the statistics and the chart NAV are sampled
    at ``max(rebalance_step, label_horizon)`` so windows are non-overlapping.
    """

    rebalance_step = _rebalance_step(rebalance_frequency)
    safe_label_horizon = max(1, int(label_horizon)) if label_horizon else 1
    effective_step = max(rebalance_step, safe_label_horizon)
    long_short_series = _long_short_series(group_returns_df)
    stats_series = _sample_rebalance_series(
        long_short_series,
        step=effective_step,
    )
    if effective_step <= 1:
        effective_periods_per_year = _periods_per_year(rebalance_frequency)
    else:
        effective_periods_per_year = max(1, round(252 / effective_step))
    stats = _return_stats(
        stats_series,
        periods_per_year=effective_periods_per_year,
    )
    nav_points = _nav_points(stats_series)
    contribution_diagnostics = build_group_contribution_diagnostics(
        group_returns_df,
        rebalance_frequency=rebalance_frequency,
        label_horizon=safe_label_horizon,
        metrics_for_payload=metrics_for_payload,
    )

    summary: dict[str, object] = {
        "annualized_return": _safe_float(stats.get("annualized_return")),
        "annualized_volatility": _safe_float(stats.get("annualized_volatility")),
        "sharpe": _safe_float(stats.get("sharpe")),
        "sortino": _safe_float(stats.get("sortino")),
        "max_drawdown": _safe_float(stats.get("max_drawdown")),
        "calmar": _safe_float(stats.get("calmar")),
        "win_rate": _coalesce_float(
            _safe_float(metrics_for_payload.get("long_short_hit_rate_full")),
            _safe_float(metrics_for_payload.get("long_short_hit_rate")),
            _safe_float(stats.get("win_rate")),
        ),
        "turnover": _coalesce_float(
            _safe_float(metrics_for_payload.get("mean_long_short_turnover_full")),
            _safe_float(metrics_for_payload.get("mean_long_short_turnover")),
        ),
        "information_ratio": _safe_float(
            metrics_for_payload.get("portfolio_validation_benchmark_information_ratio")
        ),
        "excess_return_vs_benchmark": _safe_float(
            metrics_for_payload.get("portfolio_validation_benchmark_excess_return")
        ),
        "tracking_error": _safe_float(
            metrics_for_payload.get("portfolio_validation_benchmark_tracking_error")
        ),
        "pre_cost_return": _coalesce_float(
            _safe_float(metrics_for_payload.get("mean_long_short_return_full")),
            _safe_float(metrics_for_payload.get("mean_long_short_return")),
        ),
        "post_cost_return": _coalesce_float(
            _safe_float(metrics_for_payload.get("mean_cost_adjusted_long_short_return_full")),
            _safe_float(metrics_for_payload.get("mean_cost_adjusted_long_short_return")),
        ),
        "max_drawdown_oos": _coalesce_float(
            _safe_float(metrics_for_payload.get("max_drawdown_oos")),
            _safe_float(metrics_for_payload.get("ls_max_drawdown_oos")),
        ),
        "pre_cost_return_oos": _safe_float(
            metrics_for_payload.get("mean_long_short_return_oos")
        ),
        "post_cost_return_oos": _safe_float(
            metrics_for_payload.get("mean_cost_adjusted_long_short_return_oos")
        ),
        "turnover_oos": _safe_float(metrics_for_payload.get("mean_long_short_turnover_oos")),
        "rolling_sharpe": _safe_float(stats.get("rolling_sharpe")),
        "rolling_drawdown": _safe_float(stats.get("rolling_drawdown")),
        "subperiod_analysis": _safe_text(stats.get("subperiod_analysis")) or "N/A",
        "regime_analysis": _safe_text(stats.get("regime_analysis")) or "N/A",
        "nav_points": _rows_to_json(nav_points),
        "monthly_return_table": _rows_to_json(stats.get("monthly_returns")),
        "drawdown_table": _rows_to_json(stats.get("drawdown_table")),
        "nav_source": "quantile_long_short_from_group_returns",
        "nav_series_policy": (
            "non_overlapping_forward_return_path_for_chart"
            if effective_step > 1
            else "daily_available_forward_return_path_for_chart"
        ),
        "nav_point_interval": (
            f"{effective_step}D_non_overlapping" if effective_step > 1 else "1D_available"
        ),
        "nav_rebalance_step": effective_step,
        "label_horizon": safe_label_horizon,
        "statistics_series_policy": "rebalance_sampled_non_overlapping_forward_returns",
        "statistics_rebalance_step": effective_step,
        "statistics_periods_per_year": effective_periods_per_year,
        "contribution_diagnostics": contribution_diagnostics,
    }

    fallback_derived_fields = [
        field
        for field in _BACKTEST_DERIVED_FIELDS
        if _is_unresolved_backtest_field(field, summary.get(field))
    ]
    return summary, fallback_derived_fields


def build_group_nav_table(
    group_returns_df: pd.DataFrame,
    *,
    rebalance_frequency: str,
    label_horizon: int = 1,
) -> pd.DataFrame:
    """Build a canonical, non-overlapping NAV table from a daily group_returns frame.

    ``group_returns.csv`` stores per-(date, group) **H-day forward returns** on
    the daily evaluation grid.  Naively compounding consecutive rows multiplies
    each return ~H times.  This helper samples every ``max(rebalance_step,
    label_horizon)`` trading dates so windows are non-overlapping, then runs
    cumprod within each group.  The resulting frame is the single artifact
    downstream consumers (notebooks, dashboards, exports) should compound.

    Returned columns: ``date``, ``group``, ``period_return``, ``nav``,
    ``sample_step``, ``rebalance_step``, ``label_horizon``.  Empty input or
    missing required columns yield an empty frame with the same schema.
    """

    columns = [
        "date",
        "group",
        "period_return",
        "nav",
        "sample_step",
        "rebalance_step",
        "label_horizon",
    ]
    required = {"date", "group", "group_return"}
    if not required.issubset(set(group_returns_df.columns)):
        return pd.DataFrame(columns=columns)

    frame = group_returns_df.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["group_return"] = pd.to_numeric(frame["group_return"], errors="coerce")
    frame = frame.dropna(subset=["date", "group", "group_return"])
    if frame.empty:
        return pd.DataFrame(columns=columns)

    rebalance_step = max(1, _rebalance_step(rebalance_frequency))
    safe_label_horizon = max(1, int(label_horizon)) if label_horizon else 1
    sample_step = max(rebalance_step, safe_label_horizon)

    rows: list[dict[str, object]] = []
    for group, block in frame.groupby("group", sort=True):
        per_group = (
            block[["date", "group_return"]]
            .groupby("date", as_index=False, sort=True)["group_return"]
            .mean()
            .sort_values("date", kind="mergesort")
            .reset_index(drop=True)
        )
        if per_group.empty:
            continue
        sampled = per_group.iloc[::sample_step].reset_index(drop=True)
        if sampled.empty:
            continue
        nav = (1.0 + sampled["group_return"]).cumprod()
        for date_value, period_return, nav_value in zip(
            sampled["date"], sampled["group_return"], nav, strict=True
        ):
            rows.append(
                {
                    "date": pd.Timestamp(date_value).strftime("%Y-%m-%d"),
                    "group": group,
                    "period_return": float(period_return),
                    "nav": float(nav_value),
                    "sample_step": sample_step,
                    "rebalance_step": rebalance_step,
                    "label_horizon": safe_label_horizon,
                }
            )

    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns)


def build_group_contribution_diagnostics(
    group_returns_df: pd.DataFrame,
    *,
    rebalance_frequency: str,
    label_horizon: int = 1,
    metrics_for_payload: Mapping[str, object] | None = None,
    top_n_dates: int = 10,
) -> dict[str, object]:
    """Summarize which years and dates drive quantile NAV paths.

    The diagnostic uses the same non-overlapping sampling step as
    ``build_group_nav_table`` so H-day labels are not over-compounded.
    """

    required = {"date", "group", "group_return"}
    if not required.issubset(set(group_returns_df.columns)):
        return {}

    frame = group_returns_df.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["group"] = pd.to_numeric(frame["group"], errors="coerce")
    frame["group_return"] = pd.to_numeric(frame["group_return"], errors="coerce")
    frame = frame.dropna(subset=["date", "group", "group_return"])
    if frame.empty:
        return {}

    per_bucket = (
        frame.groupby(["date", "group"], sort=True, as_index=False)["group_return"]
        .mean()
        .sort_values(["date", "group"], kind="mergesort")
    )
    rows: list[tuple[pd.Timestamp, float, float, float]] = []
    for date, block in per_bucket.groupby("date", sort=True):
        if block["group"].nunique() < 2:
            continue
        top = float(block.iloc[-1]["group_return"])
        bottom = float(block.iloc[0]["group_return"])
        group_avg = float(block["group_return"].mean())
        rows.append((pd.Timestamp(date), top, top - group_avg, top - bottom))
    if not rows:
        return {}

    index = pd.DatetimeIndex([row[0] for row in rows])
    series_by_name = {
        "top_absolute": pd.Series([row[1] for row in rows], index=index, dtype=float),
        "top_minus_group_average": pd.Series([row[2] for row in rows], index=index, dtype=float),
        "top_minus_bottom": pd.Series([row[3] for row in rows], index=index, dtype=float),
    }

    rebalance_step = max(1, _rebalance_step(rebalance_frequency))
    safe_label_horizon = max(1, int(label_horizon)) if label_horizon else 1
    sample_step = max(rebalance_step, safe_label_horizon)
    oos_start = _split_oos_start(metrics_for_payload or {})
    top_n = max(1, int(top_n_dates))

    series_summaries: list[dict[str, object]] = []
    annual_contribution: list[dict[str, object]] = []
    top_positive_dates: list[dict[str, object]] = []
    top_negative_dates: list[dict[str, object]] = []

    for name, raw_series in series_by_name.items():
        sampled = _sample_rebalance_series(raw_series, step=sample_step)
        if sampled.empty:
            continue
        series_summaries.append(_contribution_series_summary(name, sampled, oos_start=oos_start))
        annual_contribution.extend(_annual_contribution_rows(name, sampled))
        top_positive_dates.extend(
            _top_contribution_date_rows(name, sampled, ascending=False, limit=top_n)
        )
        top_negative_dates.extend(
            _top_contribution_date_rows(name, sampled, ascending=True, limit=top_n)
        )

    if not series_summaries:
        return {}

    return {
        "source": "group_returns_non_overlapping",
        "sample_step": sample_step,
        "rebalance_step": rebalance_step,
        "label_horizon": safe_label_horizon,
        "oos_start": oos_start,
        "series": series_summaries,
        "annual_contribution": annual_contribution,
        "top_positive_dates": top_positive_dates,
        "top_negative_dates": top_negative_dates,
    }


def _long_short_series(group_returns_df: pd.DataFrame) -> pd.Series:
    required = {"date", "group", "group_return"}
    if not required.issubset(set(group_returns_df.columns)):
        return pd.Series(dtype=float)

    frame = group_returns_df.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["group"] = pd.to_numeric(frame["group"], errors="coerce")
    frame["group_return"] = pd.to_numeric(frame["group_return"], errors="coerce")
    frame = frame.dropna(subset=["date", "group", "group_return"])
    if frame.empty:
        return pd.Series(dtype=float)

    per_bucket = (
        frame.groupby(["date", "group"], sort=True, as_index=False)["group_return"]
        .mean()
        .sort_values(["date", "group"], kind="mergesort")
    )
    rows: list[tuple[pd.Timestamp, float]] = []
    for date, block in per_bucket.groupby("date", sort=True):
        if block["group"].nunique() < 2:
            continue
        bottom_row = block.iloc[0]
        top_row = block.iloc[-1]
        rows.append(
            (
                pd.Timestamp(date),
                float(top_row["group_return"]) - float(bottom_row["group_return"]),
            )
        )
    if not rows:
        return pd.Series(dtype=float)
    long_short = pd.Series(
        [value for _, value in rows],
        index=pd.DatetimeIndex([date for date, _ in rows]),
        dtype=float,
    ).sort_index()
    if long_short.empty:
        return pd.Series(dtype=float)
    return long_short


def _return_stats(series: pd.Series, periods_per_year: int) -> dict[str, object]:
    clean = pd.to_numeric(series, errors="coerce").dropna().sort_index()
    if len(clean) < 2:
        return {
            "annualized_return": None,
            "annualized_volatility": None,
            "sharpe": None,
            "sortino": None,
            "max_drawdown": None,
            "calmar": None,
            "win_rate": None,
            "rolling_sharpe": None,
            "rolling_drawdown": None,
            "nav_points": [],
            "monthly_returns": [],
            "drawdown_table": [],
            "subperiod_analysis": "N/A",
            "regime_analysis": "N/A",
        }

    nav = (1.0 + clean).cumprod()
    total_return = float(nav.iloc[-1] - 1.0)
    annualized_return = float((1.0 + total_return) ** (periods_per_year / len(clean)) - 1.0)
    annualized_volatility = float(clean.std(ddof=1) * math.sqrt(periods_per_year))

    sharpe = None
    if annualized_volatility > 0:
        sharpe = annualized_return / annualized_volatility

    downside = clean[clean < 0]
    sortino = None
    if len(downside) >= 2:
        downside_vol = float(downside.std(ddof=1) * math.sqrt(periods_per_year))
        if downside_vol > 0:
            sortino = annualized_return / downside_vol

    drawdown = nav / nav.cummax() - 1.0
    max_drawdown = float(drawdown.min())
    calmar = None
    if max_drawdown < 0:
        calmar = annualized_return / abs(max_drawdown)

    win_rate = float((clean > 0).mean())

    window = min(20, len(clean))
    rolling_sharpe = None
    if window >= 5:
        rolling_mean = clean.rolling(window).mean()
        rolling_std = clean.rolling(window).std(ddof=1)
        rolling_ratio = (rolling_mean / rolling_std).replace([math.inf, -math.inf], pd.NA)
        rolling_ratio = rolling_ratio.dropna()
        if not rolling_ratio.empty:
            rolling_sharpe = float(rolling_ratio.iloc[-1] * math.sqrt(periods_per_year))

    rolling_drawdown = float(drawdown.iloc[-1])

    monthly = (1.0 + clean).resample("ME").prod() - 1.0
    monthly_rows = [[idx.strftime("%Y-%m"), float(value)] for idx, value in monthly.items()]

    worst_drawdowns = drawdown.nsmallest(8)
    drawdown_rows = [
        [idx.strftime("%Y-%m-%d"), float(value)] for idx, value in worst_drawdowns.items()
    ]

    split = len(clean) // 2
    first_half = clean.iloc[:split]
    second_half = clean.iloc[split:]
    first_ann = _annualized_from_series(first_half, periods_per_year)
    second_ann = _annualized_from_series(second_half, periods_per_year)
    subperiod_analysis = (
        f"first_half_ann={_fmt_pct(first_ann)}; second_half_ann={_fmt_pct(second_ann)}"
    )

    volatility_cut = clean.abs().median()
    high_vol = clean[clean.abs() >= volatility_cut]
    low_vol = clean[clean.abs() < volatility_cut]
    regime_analysis = (
        f"high-vol mean={_fmt_number(high_vol.mean() if len(high_vol) > 0 else None)}; "
        f"low-vol mean={_fmt_number(low_vol.mean() if len(low_vol) > 0 else None)}"
    )

    nav_points = [[idx.strftime("%Y-%m-%d"), float(value)] for idx, value in nav.items()]

    return {
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_drawdown,
        "calmar": calmar,
        "win_rate": win_rate,
        "rolling_sharpe": rolling_sharpe,
        "rolling_drawdown": rolling_drawdown,
        "nav_points": nav_points,
        "monthly_returns": monthly_rows,
        "drawdown_table": drawdown_rows,
        "subperiod_analysis": subperiod_analysis,
        "regime_analysis": regime_analysis,
    }


def _nav_points(series: pd.Series) -> list[list[object]]:
    clean = pd.to_numeric(series, errors="coerce").dropna().sort_index()
    if len(clean) < 2:
        return []
    nav = (1.0 + clean).cumprod()
    return [[idx.strftime("%Y-%m-%d"), float(value)] for idx, value in nav.items()]


def _split_oos_start(metrics: Mapping[str, object]) -> str | None:
    split_contract = metrics.get("split_contract")
    if isinstance(split_contract, Mapping):
        text = _safe_text(split_contract.get("oos_start"))
        if text:
            return text
    for key in ("oos_start", "split_oos_start"):
        text = _safe_text(metrics.get(key))
        if text:
            return text
    return None


def _contribution_series_summary(
    name: str,
    series: pd.Series,
    *,
    oos_start: str | None,
) -> dict[str, object]:
    clean = pd.to_numeric(series, errors="coerce").dropna().sort_index()
    if clean.empty:
        return {
            "name": name,
            "n_periods": 0,
            "full_total_return": None,
            "is_total_return": None,
            "oos_reset_total_return": None,
            "full_nav_terminal": None,
            "is_nav_terminal": None,
            "oos_reset_nav_terminal": None,
        }
    is_series = clean
    oos_series = pd.Series(dtype=float)
    if oos_start:
        start = pd.Timestamp(oos_start)
        is_series = clean[clean.index < start]
        oos_series = clean[clean.index >= start]
    return {
        "name": name,
        "n_periods": int(len(clean)),
        "full_total_return": _series_total_return(clean),
        "is_total_return": _series_total_return(is_series) if oos_start else None,
        "oos_reset_total_return": _series_total_return(oos_series) if oos_start else None,
        "full_nav_terminal": _series_nav_terminal(clean),
        "is_nav_terminal": _series_nav_terminal(is_series) if oos_start else None,
        "oos_reset_nav_terminal": _series_nav_terminal(oos_series) if oos_start else None,
    }


def _annual_contribution_rows(name: str, series: pd.Series) -> list[dict[str, object]]:
    clean = pd.to_numeric(series, errors="coerce").dropna().sort_index()
    if clean.empty:
        return []
    rows: list[dict[str, object]] = []
    for year, block in clean.groupby(clean.index.year, sort=True):
        rows.append(
            {
                "series": name,
                "year": str(int(year)),
                "period_count": int(len(block)),
                "total_return": _series_total_return(block),
                "log_return": _series_log_return(block),
                "mean_period_return": _safe_float(float(block.mean())),
            }
        )
    return rows


def _top_contribution_date_rows(
    name: str,
    series: pd.Series,
    *,
    ascending: bool,
    limit: int,
) -> list[dict[str, object]]:
    clean = pd.to_numeric(series, errors="coerce").dropna().sort_index()
    clean = clean[clean > -1.0]
    if clean.empty:
        return []
    ranked = clean.sort_values(ascending=ascending, kind="mergesort").head(limit)
    return [
        {
            "series": name,
            "date": idx.strftime("%Y-%m-%d"),
            "period_return": _safe_float(float(value)),
            "log_return": _safe_float(float(math.log1p(float(value)))),
        }
        for idx, value in ranked.items()
    ]


def _series_nav_terminal(series: pd.Series) -> float | None:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    clean = clean[clean > -1.0]
    if clean.empty:
        return None
    return _safe_float(float((1.0 + clean).prod()))


def _series_total_return(series: pd.Series) -> float | None:
    nav_terminal = _series_nav_terminal(series)
    if nav_terminal is None:
        return None
    return _safe_float(float(nav_terminal - 1.0))


def _series_log_return(series: pd.Series) -> float | None:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    clean = clean[clean > -1.0]
    if clean.empty:
        return None
    return _safe_float(float(clean.map(math.log1p).sum()))




def _rebalance_step(rebalance_frequency: str) -> int:
    freq = (rebalance_frequency or "").strip().upper()
    if not freq:
        return 1
    try:
        explicit = int(freq)
    except ValueError:
        explicit = 0
    if explicit > 0:
        return explicit
    if freq in {"D", "DAILY"}:
        return 1
    if freq in {"W", "WEEKLY"}:
        return 5
    if freq in {"M", "MONTHLY"}:
        return 21
    if freq in {"Q", "QUARTERLY"}:
        return 63
    return 1


def _sample_rebalance_series(series: pd.Series, *, step: int) -> pd.Series:
    clean = pd.to_numeric(series, errors="coerce").dropna().sort_index()
    if clean.empty:
        return clean
    safe_step = max(1, int(step))
    if safe_step == 1:
        return clean
    return clean.iloc[::safe_step]


def _rows_to_json(value: object) -> list[list[object]]:
    if not isinstance(value, list):
        return []
    rows = []
    for row in value:
        if not isinstance(row, list) or len(row) != 2:
            continue
        timestamp = str(row[0]).strip()
        number = _safe_float(row[1])
        if not timestamp or number is None:
            continue
        rows.append([timestamp, number])
    return rows


def _is_unresolved_backtest_field(field: str, value: object) -> bool:
    if field in {"nav_points", "monthly_return_table", "drawdown_table"}:
        return not isinstance(value, list) or len(value) == 0
    if field in {"subperiod_analysis", "regime_analysis"}:
        return _safe_text(value) is None
    return value is None


def _coalesce_float(*values: float | None) -> float | None:
    for value in values:
        if value is not None:
            return value
    return None


def _safe_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


def _safe_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _fmt_number(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.4f}"


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.2%}"


def _as_object(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}
