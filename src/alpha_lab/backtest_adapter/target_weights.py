from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from alpha_lab.backtest_adapter.schema import (
    AdapterWarning,
    BacktestInputBundle,
    PortfolioIntentFrame,
)
from alpha_lab.backtest_adapter.validators import validate_backtest_input_bundle
from alpha_lab.handoff import PortfolioConstructionSpec

_INTENT_COLUMNS: tuple[str, ...] = (
    "date",
    "asset",
    "signal_name",
    "signal_value",
    "in_universe",
    "is_tradable",
    "is_executable",
    "target_weight",
)


def build_target_weights(bundle: BacktestInputBundle) -> PortfolioIntentFrame:
    """Translate handoff contracts into deterministic per-date target weights."""

    validate_backtest_input_bundle(bundle)
    merged = _merge_bundle_tables(bundle)
    pc = bundle.portfolio_construction
    intent_method = _resolve_intent_method(pc)

    warning_records: list[AdapterWarning] = []
    if (
        bundle.execution_assumptions.trade_when_not_tradable
        and not merged["is_tradable"].all()
    ):
        warning_records.append(
            AdapterWarning(
                code="trade_when_not_tradable_enabled",
                message=(
                    "execution_assumptions.trade_when_not_tradable=True allows targets on "
                    "assets marked non-tradable in tradability_mask.csv"
                ),
            )
        )

    rows: list[pd.DataFrame] = []
    diagnostics: list[dict[str, object]] = []
    for date, group in merged.groupby("date", sort=True):
        one, one_diag, one_warnings = _build_weights_for_date(
            date=date,
            rows=group.copy(),
            intent_method=intent_method,
            bundle=bundle,
        )
        rows.append(one)
        diagnostics.append(one_diag)
        warning_records.extend(one_warnings)

    target_weights_df = (
        pd.concat(rows, ignore_index=True)
        .sort_values(["date", "asset"])
        .reset_index(drop=True)
    )
    diagnostics_df = pd.DataFrame(diagnostics).sort_values("date").reset_index(drop=True)

    return PortfolioIntentFrame(
        intent_method=intent_method,
        target_weights_df=target_weights_df.loc[:, _INTENT_COLUMNS],
        diagnostics_df=diagnostics_df,
        warnings=tuple(_dedupe_warnings(warning_records)),
    )


def _merge_bundle_tables(bundle: BacktestInputBundle) -> pd.DataFrame:
    merged = bundle.signal_snapshot_df.merge(
        bundle.universe_mask_df,
        on=["date", "asset"],
        how="inner",
        validate="one_to_one",
    )
    merged = merged.merge(
        bundle.tradability_mask_df,
        on=["date", "asset"],
        how="inner",
        validate="one_to_one",
    )
    merged["in_universe"] = merged["in_universe"].astype(bool)
    merged["is_tradable"] = merged["is_tradable"].astype(bool)
    merged["signal_value"] = pd.to_numeric(merged["signal_value"], errors="coerce")
    trade_not_tradable = bundle.execution_assumptions.trade_when_not_tradable
    merged["is_executable"] = (
        merged["in_universe"]
        & (merged["is_tradable"] | bool(trade_not_tradable))
        & merged["signal_value"].notna()
    )
    return merged.sort_values(["date", "asset"]).reset_index(drop=True)


def _resolve_intent_method(portfolio_construction: PortfolioConstructionSpec) -> str:
    construction_method = portfolio_construction.construction_method
    weight_method = portfolio_construction.weight_method
    long_short = bool(portfolio_construction.long_short)
    if construction_method == "top_bottom_k" and weight_method == "equal":
        return "rank_topbottom_equal" if long_short else "rank_topk_equal"
    if weight_method in {"rank", "score"}:
        return "zscore_proportional"
    if construction_method == "full_universe" and weight_method == "equal" and not long_short:
        return "rank_topk_equal"
    raise ValueError(
        "unsupported portfolio construction mode for v1 adapter: "
        f"construction_method={construction_method!r}, weight_method={weight_method!r}, "
        f"long_short={long_short!r}"
    )


def _build_weights_for_date(
    *,
    date: pd.Timestamp,
    rows: pd.DataFrame,
    intent_method: str,
    bundle: BacktestInputBundle,
) -> tuple[pd.DataFrame, dict[str, object], list[AdapterWarning]]:
    warnings: list[AdapterWarning] = []
    out = rows.copy()
    out["target_weight"] = 0.0
    pc = bundle.portfolio_construction

    executable = out[out["is_executable"]].copy()
    if executable.empty:
        warnings.append(
            AdapterWarning(
                code="no_executable_assets",
                message=f"{date.date().isoformat()}: no executable assets after masking",
            )
        )
        return out, _diagnostic_row(date, out), warnings

    raw_weights = _raw_weights(
        executable,
        intent_method=intent_method,
        bundle=bundle,
        warnings=warnings,
    )
    constrained = _apply_constraints(raw_weights, bundle=bundle, warnings=warnings)
    if not executable.index.equals(constrained.index):
        raise ValueError("internal adapter error: index mismatch while assigning target weights")
    out.loc[executable.index, "target_weight"] = constrained.to_numpy(dtype=float)
    if not pc.long_short:
        out["target_weight"] = out["target_weight"].clip(lower=0.0)
    return out, _diagnostic_row(date, out), warnings


def _raw_weights(
    executable: pd.DataFrame,
    *,
    intent_method: str,
    bundle: BacktestInputBundle,
    warnings: list[AdapterWarning],
) -> pd.Series:
    pc = bundle.portfolio_construction
    if intent_method == "rank_topk_equal":
        if pc.construction_method == "full_universe":
            top_k = int(len(executable))
        else:
            if pc.top_k is None:
                raise ValueError("top_k is required for rank_topk_equal")
            top_k = int(pc.top_k)
        return _rank_topk_equal(executable, top_k=top_k)

    if intent_method == "rank_topbottom_equal":
        if pc.top_k is None or pc.bottom_k is None:
            raise ValueError("top_k and bottom_k are required for rank_topbottom_equal")
        return _rank_topbottom_equal(
            executable,
            top_k=int(pc.top_k),
            bottom_k=int(pc.bottom_k),
            warnings=warnings,
        )

    if intent_method == "zscore_proportional":
        return _zscore_proportional(executable, bundle=bundle, warnings=warnings)

    raise ValueError(f"unsupported intent method {intent_method!r}")


def _rank_topk_equal(executable: pd.DataFrame, *, top_k: int) -> pd.Series:
    if top_k <= 0:
        raise ValueError("top_k must be > 0")
    ordered = executable.sort_values(
        ["signal_value", "asset"],
        ascending=[False, True],
        kind="mergesort",
    )
    k = min(top_k, len(ordered))
    weights = pd.Series(0.0, index=executable.index, dtype=float)
    if k == 0:
        return weights
    selected = ordered.iloc[:k].index
    weights.loc[selected] = 1.0 / float(k)
    return weights


def _rank_topbottom_equal(
    executable: pd.DataFrame,
    *,
    top_k: int,
    bottom_k: int,
    warnings: list[AdapterWarning],
) -> pd.Series:
    if top_k <= 0 or bottom_k <= 0:
        raise ValueError("top_k and bottom_k must be > 0 for long-short ranking")
    ordered_desc = executable.sort_values(
        ["signal_value", "asset"],
        ascending=[False, True],
        kind="mergesort",
    )
    ordered_asc = executable.sort_values(
        ["signal_value", "asset"],
        ascending=[True, True],
        kind="mergesort",
    )
    long_count = min(top_k, len(ordered_desc))
    long_idx = ordered_desc.iloc[:long_count].index
    remaining_for_short = ordered_asc.loc[~ordered_asc.index.isin(long_idx)]
    short_count = min(bottom_k, len(remaining_for_short))
    if short_count < bottom_k:
        warnings.append(
            AdapterWarning(
                code="insufficient_short_candidates",
                message=(
                    "insufficient distinct assets for requested bottom_k; "
                    f"requested={bottom_k}, assigned={short_count}"
                ),
            )
        )

    weights = pd.Series(0.0, index=executable.index, dtype=float)
    if long_count > 0:
        weights.loc[long_idx] = 1.0 / float(long_count)
    if short_count > 0:
        short_idx = remaining_for_short.iloc[:short_count].index
        weights.loc[short_idx] = -1.0 / float(short_count)
    return weights


def _zscore_proportional(
    executable: pd.DataFrame,
    *,
    bundle: BacktestInputBundle,
    warnings: list[AdapterWarning],
) -> pd.Series:
    pc = bundle.portfolio_construction
    signal = executable["signal_value"].astype(float)
    std = float(signal.std(ddof=0))
    if not np.isfinite(std) or std <= 0.0:
        warnings.append(
            AdapterWarning(
                code="zero_signal_variance",
                message="signal variance is zero for one date; target weights set to 0",
            )
        )
        return pd.Series(0.0, index=executable.index, dtype=float)

    z = (signal - float(signal.mean())) / std
    weights = pd.Series(z.to_numpy(dtype=float), index=executable.index, dtype=float)
    if pc.construction_method == "top_bottom_k":
        if pc.top_k is None:
            raise ValueError("top_k is required when construction_method='top_bottom_k'")
        top_idx = executable.sort_values(
            ["signal_value", "asset"],
            ascending=[False, True],
            kind="mergesort",
        ).iloc[: min(int(pc.top_k), len(executable))].index
        selected_idx = set(top_idx)
        if pc.long_short:
            if pc.bottom_k is None:
                raise ValueError(
                    "bottom_k is required when long_short=True and "
                    "construction_method='top_bottom_k'"
                )
            bottom_ordered = executable.sort_values(
                ["signal_value", "asset"],
                ascending=[True, True],
                kind="mergesort",
            )
            bottom_ordered = bottom_ordered.loc[~bottom_ordered.index.isin(top_idx)]
            bottom_idx = bottom_ordered.iloc[
                : min(int(pc.bottom_k), len(bottom_ordered))
            ].index
            selected_idx.update(bottom_idx.tolist())
        weights = weights.where(weights.index.to_series().isin(selected_idx), other=0.0)

    if not pc.long_short:
        weights = weights.clip(lower=0.0)
    if pc.weight_clip is not None:
        clip = float(pc.weight_clip)
        if pc.long_short:
            weights = weights.clip(lower=-clip, upper=clip)
        else:
            weights = weights.clip(lower=0.0, upper=clip)
    return weights


def _apply_constraints(
    raw_weights: pd.Series,
    *,
    bundle: BacktestInputBundle,
    warnings: list[AdapterWarning],
) -> pd.Series:
    pc = bundle.portfolio_construction
    w = raw_weights.copy()
    if not pc.long_short:
        w = w.clip(lower=0.0)
    if pc.min_weight is not None:
        mask = (w != 0.0) & (w.abs() < float(pc.min_weight))
        w.loc[mask] = 0.0

    gross_target = float(pc.gross_limit) * (1.0 - float(pc.cash_buffer))
    if gross_target <= 0.0:
        warnings.append(
            AdapterWarning(
                code="non_positive_investable_gross",
                message="gross_limit × (1 - cash_buffer) <= 0; target weights set to 0",
            )
        )
        return pd.Series(0.0, index=w.index, dtype=float)
    if float(w.abs().sum()) == 0.0:
        return w

    if pc.long_short:
        w = _normalize_long_short(
            w,
            gross_target=gross_target,
            net_limit=float(pc.net_limit),
            warnings=warnings,
        )
    else:
        w = _normalize_long_only(
            w,
            gross_target=gross_target,
            net_limit=float(pc.net_limit),
            warnings=warnings,
        )

    max_weight = float(pc.max_weight)
    over_max = w.abs() > max_weight
    if over_max.any():
        warnings.append(
            AdapterWarning(
                code="max_weight_clipped",
                message=(
                    "target weights exceeded max_weight and were clipped; "
                    "portfolio may run below gross_limit"
                ),
            )
        )
    w = w.clip(lower=-max_weight, upper=max_weight)
    if not pc.long_short:
        w = w.clip(lower=0.0)
    return w


def _normalize_long_short(
    weights: pd.Series,
    *,
    gross_target: float,
    net_limit: float,
    warnings: list[AdapterWarning],
) -> pd.Series:
    pos = weights.clip(lower=0.0)
    neg_abs = (-weights.clip(upper=0.0))
    pos_sum = float(pos.sum())
    neg_sum = float(neg_abs.sum())
    if pos_sum == 0.0 or neg_sum == 0.0:
        warnings.append(
            AdapterWarning(
                code="long_short_side_missing",
                message=(
                    "long-short intent has an empty long or short side on one date; "
                    "falling back to one-sided normalization"
                ),
            )
        )
        one_sided = pos if pos_sum > 0.0 else (-neg_abs)
        gross = float(one_sided.abs().sum())
        if gross == 0.0:
            return pd.Series(0.0, index=weights.index, dtype=float)
        return one_sided * (gross_target / gross)

    raw_net = pos_sum - neg_sum
    target_net = float(np.clip(raw_net, -net_limit, net_limit))
    long_budget = (gross_target + target_net) / 2.0
    short_budget = (gross_target - target_net) / 2.0
    if long_budget < 0.0 or short_budget < 0.0:
        raise ValueError("invalid gross/net constraints: implied side budget is negative")
    scaled_pos = pos * (long_budget / pos_sum)
    scaled_neg = neg_abs * (short_budget / neg_sum)
    return scaled_pos - scaled_neg


def _normalize_long_only(
    weights: pd.Series,
    *,
    gross_target: float,
    net_limit: float,
    warnings: list[AdapterWarning],
) -> pd.Series:
    long_only = weights.clip(lower=0.0)
    gross = float(long_only.sum())
    if gross == 0.0:
        return pd.Series(0.0, index=weights.index, dtype=float)
    scaled = long_only * (gross_target / gross)
    if net_limit > 0.0 and float(scaled.sum()) > net_limit:
        scaled = scaled * (net_limit / float(scaled.sum()))
    elif net_limit == 0.0:
        warnings.append(
            AdapterWarning(
                code="long_only_net_limit_zero_ignored",
                message=(
                    "long-only construction has net_limit=0; treated as unconstrained in v1 "
                    "adapter to avoid degenerate zero portfolio"
                ),
            )
        )
    return scaled


def _diagnostic_row(date: pd.Timestamp, weighted_rows: pd.DataFrame) -> dict[str, object]:
    weights = weighted_rows["target_weight"].astype(float)
    return {
        "date": date,
        "gross_exposure": float(weights.abs().sum()),
        "net_exposure": float(weights.sum()),
        "n_positions": int((weights.abs() > 0.0).sum()),
        "n_executable_assets": int(weighted_rows["is_executable"].sum()),
        "n_universe_assets": int(weighted_rows["in_universe"].sum()),
    }


def _dedupe_warnings(warnings: list[AdapterWarning]) -> list[AdapterWarning]:
    seen: set[tuple[str, str]] = set()
    out: list[AdapterWarning] = []
    for warning in warnings:
        key = (warning.code, warning.message)
        if key in seen:
            continue
        seen.add(key)
        out.append(replace(warning))
    return out
