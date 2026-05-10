from __future__ import annotations

import pandas as pd

from alpha_lab.labels import LabelCache
from alpha_lab.research_evaluation_config import (
    ResearchEvaluationConfig,
)

# Cross-module imports (auto-added)
from ._utils import _date_text, _finite_or_none, _jsonable_scalar, _model_factor_decay_horizons
from .feature_manifest import _unique_columns


def _model_factor_price_read_columns(
    evaluation_config: ResearchEvaluationConfig,
    *,
    target_price_column: str = "close",
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    target_column = str(target_price_column or "close").strip() or "close"
    required = _unique_columns(["date", "asset", "close", target_column])
    diagnostics_cfg = evaluation_config.single_factor_diagnostics
    optional: list[str] = [
        # Preserve default dividend back-adjustment and data-quality summaries
        # while still avoiding unrelated wide price columns.
        "dividend_per_share",
        "volume",
    ]
    if diagnostics_cfg.run_tradability_checks:
        optional.extend(["open", "high", "low", "volume"])
    if diagnostics_cfg.run_execution_price_sensitivity:
        optional.append("open")
    if diagnostics_cfg.compute_capacity_estimation:
        optional.extend(["amount", "market_cap", "circ_mv", "total_mv", "value"])
    if diagnostics_cfg.run_baseline_comparison:
        optional.extend(["ret_5d", "ret_20d"])
    return required, _unique_columns(optional, exclude=set(required))


def _build_forward_label_cache(
    *,
    prices: pd.DataFrame,
    target_horizon: int,
    target_label_df: pd.DataFrame,
    target_price_column: str = "close",
    max_abs_forward_return: float | None = None,
    evaluation_config: ResearchEvaluationConfig,
) -> dict[int, pd.DataFrame]:
    horizons = {int(target_horizon)}
    if evaluation_config.single_factor_diagnostics.compute_ic_decay:
        horizons.update(_model_factor_decay_horizons(target_horizon))

    label_prices = _prices_for_forward_labels(prices, price_column=target_price_column)
    label_cache = LabelCache(label_prices)
    cache: dict[int, pd.DataFrame] = {}
    for horizon in sorted(horizons):
        if horizon == int(target_horizon):
            cache[horizon] = target_label_df.copy()
        else:
            labels = label_cache.forward_return(horizon)
            cache[horizon] = _filter_forward_label_frame(
                labels,
                max_abs_forward_return=max_abs_forward_return,
            )
    return cache


def _prices_for_forward_labels(prices: pd.DataFrame, *, price_column: str) -> pd.DataFrame:
    column = str(price_column or "close").strip() or "close"
    if column not in prices.columns:
        raise ValueError(f"target price column {column!r} is missing from prices")
    frame = prices.loc[:, ["date", "asset", column]].copy()
    if column != "close":
        frame = frame.rename(columns={column: "close"})
    return frame.loc[:, ["date", "asset", "close"]]


def _filter_forward_label_frame(
    labels: pd.DataFrame,
    *,
    max_abs_forward_return: float | None,
) -> pd.DataFrame:
    if max_abs_forward_return is None or labels.empty:
        return labels
    out = labels.copy()
    values = pd.to_numeric(out["value"], errors="coerce")
    out.loc[values.abs() > float(max_abs_forward_return), "value"] = pd.NA
    return out


def _group_return_extreme_rows(
    group_returns: pd.DataFrame,
    *,
    threshold: float = 0.30,
    limit: int = 20,
) -> list[dict[str, object]]:
    if group_returns.empty or "group_return" not in group_returns.columns:
        return []
    frame = group_returns.copy()
    values = pd.to_numeric(frame["group_return"], errors="coerce")
    frame = frame[values.abs() > float(threshold)].copy()
    if frame.empty:
        return []
    frame["_abs"] = values.loc[frame.index].abs()
    frame = frame.sort_values("_abs", ascending=False, kind="mergesort").head(int(limit))
    rows: list[dict[str, object]] = []
    for row in frame.itertuples(index=False):
        rows.append(
            {
                "date": _date_text(getattr(row, "date", None)),
                "group": _jsonable_scalar(getattr(row, "group", None)),
                "group_return": _finite_or_none(getattr(row, "group_return", None)),
            }
        )
    return rows


def _enrich_label_extreme_samples(
    samples: object,
    *,
    assignments: pd.DataFrame,
    group_returns: pd.DataFrame,
) -> list[dict[str, object]]:
    if not isinstance(samples, list) or not samples:
        return []
    rows = pd.DataFrame(samples)
    if rows.empty or not {"date", "asset"}.issubset(rows.columns):
        return []
    rows["date"] = pd.to_datetime(rows["date"], errors="coerce")
    assign = assignments.copy()
    if not assign.empty and {"date", "asset", "quantile"}.issubset(assign.columns):
        assign["date"] = pd.to_datetime(assign["date"], errors="coerce")
        rows = rows.merge(
            assign[["date", "asset", "quantile"]],
            on=["date", "asset"],
            how="left",
            validate="many_to_one",
        )
    if "quantile" in rows.columns:
        group_frame = group_returns.copy()
        required_group_columns = {"date", "group", "group_return"}
        if not group_frame.empty and required_group_columns.issubset(group_frame.columns):
            group_frame["date"] = pd.to_datetime(group_frame["date"], errors="coerce")
            group_frame["_quantile"] = pd.to_numeric(group_frame["group"], errors="coerce")
            rows["_quantile"] = pd.to_numeric(rows["quantile"], errors="coerce")
            rows = rows.merge(
                group_frame[["date", "_quantile", "group_return"]],
                on=["date", "_quantile"],
                how="left",
                validate="many_to_one",
            )
    out: list[dict[str, object]] = []
    for row in rows.itertuples(index=False):
        out.append(
            {
                "date": _date_text(getattr(row, "date", None)),
                "group": _jsonable_scalar(getattr(row, "quantile", None)),
                "group_return": _finite_or_none(getattr(row, "group_return", None)),
                "asset": _jsonable_scalar(getattr(row, "asset", None)),
                "raw_return": _finite_or_none(getattr(row, "raw_return", None)),
                "entry_price": _finite_or_none(getattr(row, "entry_price", None)),
                "exit_price": _finite_or_none(getattr(row, "exit_price", None)),
            }
        )
    return out
