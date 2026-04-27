"""Detect untradable days for A-share style price-volume factors.

A day is considered **untradable** when an order placed at that day's open
cannot be executed at any reasonable price.  Typical causes:

- **一字涨停板 (one-word limit-up):** open == high == low == close, and the
  close is at or above the limit-up reference (prior close × (1 + limit_pct)).
  No seller steps in, so a buy order cannot fill.
- **一字跌停板 (one-word limit-down):** symmetric — no buyer steps in, so a
  sell order cannot fill.
- **Suspended / zero-volume days:** no trading at all.

For short-horizon price-volume factors these are the single largest silent
source of backtest leakage: the factor "signals" on day ``t`` and the
vectorised backtest marks the return using prices the strategy could never
have touched.

This module provides pure detection helpers.  Callers wire the mask into
their evaluation (``apply_untradable_mask_to_labels``) when they want the
filter applied.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_lab.exceptions import AlphaLabDataError

# A-share default daily price-move limit (±10%).  Practical detection uses
# a slightly tighter threshold so rounding and reference-price noise do not
# generate false negatives.
_DEFAULT_LIMIT_PCT = 0.099


def detect_limit_moves(
    prices_df: pd.DataFrame,
    *,
    limit_pct: float = _DEFAULT_LIMIT_PCT,
) -> pd.DataFrame:
    """Flag rows where the open cannot be transacted.

    Returns a copy of ``prices_df`` sorted by ``(asset, date)`` with added
    boolean columns:

    - ``is_limit_up_board``: one-word limit-up board; buy orders cannot fill.
    - ``is_limit_down_board``: one-word limit-down board; sell orders cannot fill.
    - ``is_untradable``: the union of the above plus zero/NaN volume.

    Detection requires at least ``[date, asset, close]``.  Uses ``open``,
    ``high``, ``low`` when available to confirm one-word boards; otherwise
    falls back to a pct-change-based proxy (any move ≥ ``limit_pct`` on a day
    with zero intraday range).  ``volume`` is used for the suspended check
    when present.

    Parameters
    ----------
    prices_df:
        Long-form price panel.
    limit_pct:
        Daily price move threshold (default 9.9%).  A.k.a. the "price
        limit" — ST stocks use 5%, which callers should pass explicitly.
    """
    if limit_pct <= 0.0 or limit_pct >= 1.0:
        raise ValueError("limit_pct must be in (0, 1)")
    _require_columns(prices_df, ("date", "asset", "close"))

    out = prices_df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out = out.sort_values(["asset", "date"], kind="mergesort").reset_index(drop=True)

    prev_close = out.groupby("asset", sort=False)["close"].shift(1)
    pct_change = out["close"].div(prev_close).sub(1.0)

    has_ohl = {"open", "high", "low"}.issubset(out.columns)
    if has_ohl:
        open_ = pd.to_numeric(out["open"], errors="coerce")
        high = pd.to_numeric(out["high"], errors="coerce")
        low = pd.to_numeric(out["low"], errors="coerce")
        # One-word board: open == high == low == close (up to floating noise)
        # and the move is ≥ limit_pct in the corresponding direction.
        range_zero = (high.sub(low).abs() <= 1e-9) & (open_.sub(out["close"]).abs() <= 1e-9)
        up_board = range_zero & (pct_change >= limit_pct)
        down_board = range_zero & (pct_change <= -limit_pct)
    else:
        # Fallback: we cannot verify "one-word" without intraday range.
        # Treat any day whose move meets the limit as potentially untradable —
        # this overcounts, but under-counting is the worse failure mode for
        # research integrity.
        up_board = pct_change >= limit_pct
        down_board = pct_change <= -limit_pct

    up_board = up_board.fillna(False).astype(bool)
    down_board = down_board.fillna(False).astype(bool)

    if "volume" in out.columns:
        vol = pd.to_numeric(out["volume"], errors="coerce")
        suspended = (vol <= 0.0) | vol.isna()
    else:
        suspended = pd.Series(False, index=out.index)

    out["is_limit_up_board"] = up_board
    out["is_limit_down_board"] = down_board
    out["is_untradable"] = (up_board | down_board | suspended).astype(bool)
    return out


def apply_untradable_mask_to_labels(
    label_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    *,
    direction: str = "both",
    limit_pct: float = _DEFAULT_LIMIT_PCT,
) -> pd.DataFrame:
    """Blank labels whose next-day execution price is unreachable.

    A factor value at ``t`` is intended to inform a trade executed at ``t+1``.
    If day ``t+1`` is a one-word board in the trading direction, the trade
    cannot happen; the label at ``t`` is therefore not realisable and should
    be treated as missing.

    Parameters
    ----------
    label_df:
        Long-form ``[date, asset, factor, value]`` forward-return labels.
    prices_df:
        Long-form price panel used to detect untradable days.
    direction:
        ``"long"`` blanks only when next-day is limit-up (buy side).
        ``"short"`` blanks only when next-day is limit-down (sell side).
        ``"both"`` (default) blanks either — appropriate for a long-short
        factor evaluation where the same label feeds both legs.
    limit_pct:
        See :func:`detect_limit_moves`.

    Returns
    -------
    pd.DataFrame
        Copy of ``label_df`` with ``value`` set to NaN on masked rows.
        A boolean column ``is_next_day_untradable`` is added.
    """
    direction = direction.strip().lower()
    if direction not in {"long", "short", "both"}:
        raise ValueError("direction must be 'long', 'short', or 'both'")
    _require_columns(label_df, ("date", "asset", "value"))

    flagged = detect_limit_moves(prices_df, limit_pct=limit_pct)
    if direction == "long":
        mask_col = flagged["is_limit_up_board"]
    elif direction == "short":
        mask_col = flagged["is_limit_down_board"]
    else:
        mask_col = flagged["is_untradable"]

    # The *execution* day is t+1 from the factor timestamp.  Shift the
    # untradable mask one step back per asset so each label at t inherits
    # the execution-day tradability of t+1.
    flagged = flagged.assign(
        _exec_untradable=mask_col.astype(bool),
    )
    flagged["_label_untradable"] = flagged.groupby("asset", sort=False)["_exec_untradable"].shift(
        -1
    )
    key = flagged[["date", "asset", "_label_untradable"]].copy()
    key["date"] = pd.to_datetime(key["date"], errors="coerce")

    out = label_df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    merged = out.merge(key, on=["date", "asset"], how="left")
    untradable = merged["_label_untradable"].fillna(False).astype(bool)

    merged["is_next_day_untradable"] = untradable
    merged.loc[untradable, "value"] = np.nan
    merged = merged.drop(columns=["_label_untradable"])
    return merged


def summarise_tradability(flagged: pd.DataFrame) -> dict[str, float]:
    """Compact counts / rates from a frame returned by :func:`detect_limit_moves`."""
    required = {"is_limit_up_board", "is_limit_down_board", "is_untradable"}
    if not required.issubset(flagged.columns):
        raise AlphaLabDataError(
            f"flagged frame missing tradability columns: {sorted(required - set(flagged.columns))}"
        )
    total = int(len(flagged))
    up = int(flagged["is_limit_up_board"].fillna(False).astype(bool).sum())
    down = int(flagged["is_limit_down_board"].fillna(False).astype(bool).sum())
    any_ = int(flagged["is_untradable"].fillna(False).astype(bool).sum())
    rate = float(any_) / float(total) if total > 0 else float("nan")
    return {
        "tradability_total_rows": total,
        "tradability_limit_up_rows": up,
        "tradability_limit_down_rows": down,
        "tradability_untradable_rows": any_,
        "tradability_untradable_rate": rate,
    }


def _require_columns(frame: pd.DataFrame, cols: tuple[str, ...]) -> None:
    missing = set(cols) - set(frame.columns)
    if missing:
        raise AlphaLabDataError(f"missing required columns: {sorted(missing)}")
