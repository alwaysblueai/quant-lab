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

    warning_records: list[AdapterWarning] = []
    if bundle.execution_assumptions.trade_when_not_tradable:
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
    for _, group in merged.groupby("date", sort=True):
        one, one_diag, one_warnings = _build_weights_for_date(group.copy(), bundle)
        rows.append(one)
        diagnostics.append(one_diag)
        warning_records.extend(one_warnings)

    target_weights_df = pd.concat(rows, ignore_index=True)
    target_weights_df = target_weights_df.loc[:, list(_INTENT_COLUMNS)]
    target_weights_df = target_weights_df.sort_values(
        ["date", "asset"], kind="mergesort"
    ).reset_index(drop=True)

    diagnostics_df = (
        pd.DataFrame(diagnostics).sort_values(["date"], kind="mergesort").reset_index(drop=True)
    )

    return PortfolioIntentFrame(
        intent_method=_resolve_intent_method(bundle),
        target_weights_df=target_weights_df,
        diagnostics_df=diagnostics_df,
        warnings=_dedupe_warnings(warning_records),
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
    merged = merged.dropna(subset=["signal_value"])

    trade_not_tradable = bundle.execution_assumptions.trade_when_not_tradable
    merged["is_executable"] = merged["in_universe"] & (merged["is_tradable"] | trade_not_tradable)
    return merged.sort_values(["date", "asset"], kind="mergesort").reset_index(drop=True)


def _resolve_intent_method(bundle: BacktestInputBundle) -> str:
    construction_method = bundle.portfolio_construction.construction_method
    weight_method = bundle.portfolio_construction.weight_method
    long_short = bundle.portfolio_construction.long_short

    if construction_method == "top_bottom_k":
        if long_short and weight_method in {"rank", "equal"}:
            return f"{weight_method}_topbottom"
        if not long_short and weight_method in {"rank", "equal"}:
            return f"{weight_method}_topk"
    if construction_method == "full_universe" and weight_method == "score":
        return "zscore_proportional"

    raise ValueError(
        "unsupported portfolio construction mode for v1 adapter: "
        f"construction_method={construction_method!r}, "
        f"weight_method={weight_method!r}, long_short={long_short!r}"
    )


def _build_weights_for_date(
    group: pd.DataFrame,
    bundle: BacktestInputBundle,
) -> tuple[pd.DataFrame, dict[str, object], list[AdapterWarning]]:
    warnings: list[AdapterWarning] = []
    group["target_weight"] = 0.0

    executable = group[group["is_executable"]].copy()
    if executable.empty:
        warnings.append(
            AdapterWarning(
                code="no_executable_assets",
                message="one date has no executable assets after masking",
            )
        )
        return group, _diagnostic_row(group), warnings

    pc = bundle.portfolio_construction
    if pc.construction_method == "top_bottom_k":
        raw_weights = _top_bottom_weights(executable, pc, warnings)
    elif pc.construction_method == "full_universe":
        raw_weights = _zscore_proportional(executable, warnings)
    else:
        raise ValueError(f"unsupported construction_method: {pc.construction_method}")

    constrained = _apply_constraints(raw_weights, pc, warnings)
    if not constrained.index.equals(executable.index):
        raise ValueError("internal adapter error: index mismatch while assigning target weights")
    group.loc[constrained.index, "target_weight"] = constrained.to_numpy(dtype=float)
    return group, _diagnostic_row(group), warnings


def _top_bottom_weights(
    executable: pd.DataFrame,
    pc,
    warnings: list[AdapterWarning],
) -> pd.Series:
    top_k = pc.top_k
    if top_k is None:
        raise ValueError("top_k is required when construction_method='top_bottom_k'")
    if top_k <= 0:
        raise ValueError("top_k must be > 0")

    ordered_desc = executable.sort_values(
        ["signal_value", "asset"], ascending=[False, True], kind="mergesort"
    )
    ordered_asc = executable.sort_values(
        ["signal_value", "asset"], ascending=[True, True], kind="mergesort"
    )

    long_idx = ordered_desc.index[: min(top_k, len(ordered_desc))]
    weights = pd.Series(0.0, index=executable.index, dtype=float)

    if pc.weight_method == "equal":
        long_side = pd.Series(1.0, index=long_idx)
    else:  # rank
        rank = np.arange(len(long_idx), 0, -1, dtype=float)
        long_side = pd.Series(rank, index=long_idx)
    weights.loc[long_side.index] = long_side

    if pc.long_short:
        bottom_k = pc.bottom_k
        if bottom_k is None or bottom_k <= 0:
            raise ValueError("bottom_k must be > 0 when long_short=True")
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
        short_idx = remaining_for_short.index[:short_count]
        if pc.weight_method == "equal":
            short_side = pd.Series(-1.0, index=short_idx)
        else:
            rank = np.arange(len(short_idx), 0, -1, dtype=float)
            short_side = pd.Series(-rank, index=short_idx)
        weights.loc[short_side.index] = short_side

    return weights


def _zscore_proportional(
    executable: pd.DataFrame,
    warnings: list[AdapterWarning],
) -> pd.Series:
    signal = executable["signal_value"].astype(float)
    std = float(signal.std(ddof=0))
    if not np.isfinite(std) or std == 0.0:
        warnings.append(
            AdapterWarning(
                code="zero_signal_variance",
                message="signal variance is zero for one date; target weights set to 0",
            )
        )
        return pd.Series(0.0, index=executable.index, dtype=float)
    z = (signal - float(signal.mean())) / std
    return pd.Series(z.to_numpy(dtype=float), index=executable.index, dtype=float)


def _apply_constraints(
    raw_weights: pd.Series,
    pc,
    warnings: list[AdapterWarning],
) -> pd.Series:
    out = raw_weights.copy()
    gross_target = pc.gross_limit * (1.0 - pc.cash_buffer)
    if gross_target <= 0:
        warnings.append(
            AdapterWarning(
                code="non_positive_investable_gross",
                message=(
                    f"gross_limit {pc.gross_limit} × (1 - cash_buffer) <= 0; "
                    "target weights set to 0"
                ),
            )
        )
        return pd.Series(0.0, index=out.index, dtype=float)

    if pc.long_short:
        out = _normalize_long_short(out, gross_target, pc.net_limit, warnings)
    else:
        out = _normalize_long_only(out, gross_target, pc.net_limit, warnings)

    if pc.max_weight is not None:
        mask = out.abs() > pc.max_weight
        if mask.any():
            out = out.clip(lower=-pc.max_weight, upper=pc.max_weight)
            warnings.append(
                AdapterWarning(
                    code="max_weight_clipped",
                    message=(
                        "target weights exceeded max_weight and were clipped; "
                        "portfolio may run below gross_limit"
                    ),
                )
            )
    if pc.min_weight is not None:
        nonzero = out.abs() > 0
        out.loc[nonzero & (out.abs() < pc.min_weight)] = (
            np.sign(out.loc[nonzero & (out.abs() < pc.min_weight)]) * pc.min_weight
        )

    gross = float(out.abs().sum())
    if gross > gross_target and gross > 0:
        out = out * (gross_target / gross)
    return out


def _normalize_long_short(
    raw: pd.Series,
    gross_target: float,
    net_limit: float,
    warnings: list[AdapterWarning],
) -> pd.Series:
    pos = raw.clip(lower=0.0)
    neg_abs = -raw.clip(upper=0.0)
    pos_sum = float(pos.sum())
    neg_sum = float(neg_abs.sum())
    if pos_sum <= 0 or neg_sum <= 0:
        warnings.append(
            AdapterWarning(
                code="long_short_side_missing",
                message=(
                    "long-short intent has an empty long or short side on one date; "
                    "falling back to one-sided normalization"
                ),
            )
        )
        return _normalize_long_only(raw, gross_target, net_limit, warnings)

    long_budget = (gross_target + net_limit) / 2.0
    short_budget = (gross_target - net_limit) / 2.0
    if long_budget < 0 or short_budget < 0:
        raise ValueError("invalid gross/net constraints: implied side budget is negative")

    scaled_pos = pos * (long_budget / pos_sum)
    scaled_neg = -neg_abs * (short_budget / neg_sum)
    return scaled_pos + scaled_neg


def _normalize_long_only(
    raw: pd.Series,
    gross_target: float,
    net_limit: float,
    warnings: list[AdapterWarning],
) -> pd.Series:
    pos = raw.clip(lower=0.0)
    if net_limit == 0:
        warnings.append(
            AdapterWarning(
                code="long_only_net_limit_zero_ignored",
                message=(
                    "long-only construction has net_limit=0; treated as unconstrained "
                    "in v1 adapter to avoid degenerate zero portfolio"
                ),
            )
        )
    pos_sum = float(pos.sum())
    if pos_sum <= 0:
        return pd.Series(0.0, index=raw.index, dtype=float)
    return pos * (gross_target / pos_sum)


def _diagnostic_row(weighted_rows: pd.DataFrame) -> dict[str, object]:
    return {
        "date": pd.Timestamp(weighted_rows["date"].iloc[0]).isoformat(),
        "gross_exposure": float(weighted_rows["target_weight"].abs().sum()),
        "net_exposure": float(weighted_rows["target_weight"].sum()),
        "n_positions": int((weighted_rows["target_weight"] != 0).sum()),
        "n_executable_assets": int(weighted_rows["is_executable"].sum()),
        "n_universe_assets": int(weighted_rows["in_universe"].sum()),
    }


def _dedupe_warnings(warnings: list[AdapterWarning]) -> tuple[AdapterWarning, ...]:
    seen: set[tuple[str, str]] = set()
    deduped: list[AdapterWarning] = []
    for warning in warnings:
        key = (warning.code, warning.message)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(replace(warning))
    return tuple(deduped)
