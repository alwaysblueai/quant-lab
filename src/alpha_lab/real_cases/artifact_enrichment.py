from __future__ import annotations

import math
from collections.abc import Mapping

import pandas as pd

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
            f"max|w|~{_fmt_number(max_abs_weight)}; "
            f"effective names~{_fmt_number(effective_names)}"
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
) -> tuple[dict[str, object], list[str]]:
    """Build canonical backtest summary fields from write-time group-returns data."""

    long_short_series = _long_short_series(group_returns_df)
    stats = _return_stats(
        long_short_series,
        periods_per_year=_periods_per_year(rebalance_frequency),
    )

    summary: dict[str, object] = {
        "annualized_return": _safe_float(stats.get("annualized_return")),
        "annualized_volatility": _safe_float(stats.get("annualized_volatility")),
        # Preserve legacy interpretation of this field while allowing canonical fallback.
        "sharpe": _coalesce_float(
            _safe_float(metrics_for_payload.get("long_short_ir")),
            _safe_float(stats.get("sharpe")),
        ),
        "sortino": _safe_float(stats.get("sortino")),
        "max_drawdown": _safe_float(stats.get("max_drawdown")),
        "calmar": _safe_float(stats.get("calmar")),
        "win_rate": _coalesce_float(
            _safe_float(metrics_for_payload.get("long_short_hit_rate")),
            _safe_float(stats.get("win_rate")),
        ),
        "turnover": _safe_float(metrics_for_payload.get("mean_long_short_turnover")),
        "information_ratio": _safe_float(
            metrics_for_payload.get("portfolio_validation_benchmark_information_ratio")
        ),
        "excess_return_vs_benchmark": _safe_float(
            metrics_for_payload.get("portfolio_validation_benchmark_excess_return")
        ),
        "tracking_error": _safe_float(
            metrics_for_payload.get("portfolio_validation_benchmark_tracking_error")
        ),
        "pre_cost_return": _safe_float(metrics_for_payload.get("mean_long_short_return")),
        "post_cost_return": _safe_float(
            metrics_for_payload.get("mean_cost_adjusted_long_short_return")
        ),
        "rolling_sharpe": _safe_float(stats.get("rolling_sharpe")),
        "rolling_drawdown": _safe_float(stats.get("rolling_drawdown")),
        "subperiod_analysis": _safe_text(stats.get("subperiod_analysis")) or "N/A",
        "regime_analysis": _safe_text(stats.get("regime_analysis")) or "N/A",
        "nav_points": _rows_to_json(stats.get("nav_points")),
        "monthly_return_table": _rows_to_json(stats.get("monthly_returns")),
        "drawdown_table": _rows_to_json(stats.get("drawdown_table")),
    }

    fallback_derived_fields = [
        field
        for field in _BACKTEST_DERIVED_FIELDS
        if _is_unresolved_backtest_field(field, summary.get(field))
    ]
    return summary, fallback_derived_fields


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

    pivot = frame.pivot_table(index="date", columns="group", values="group_return", aggfunc="mean")
    if pivot.shape[1] < 2:
        return pd.Series(dtype=float)

    bottom = pivot.columns.min()
    top = pivot.columns.max()
    long_short = (pivot[top] - pivot[bottom]).sort_index().dropna()
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

    monthly = clean.resample("ME").apply(lambda values: float((1.0 + values).prod() - 1.0))
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


def _annualized_from_series(series: pd.Series, periods_per_year: int) -> float | None:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return None
    nav = (1.0 + clean).cumprod()
    total_return = float(nav.iloc[-1] - 1.0)
    return float((1.0 + total_return) ** (periods_per_year / len(clean)) - 1.0)


def _periods_per_year(rebalance_frequency: str) -> int:
    freq = (rebalance_frequency or "").strip().upper()
    if freq.startswith("D"):
        return 252
    if freq.startswith("W"):
        return 52
    if freq.startswith("M"):
        return 12
    return 252


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


def _coalesce_float(primary: float | None, secondary: float | None) -> float | None:
    if primary is not None:
        return primary
    return secondary


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
