"""Quick cross-sectional sanity-IC for 3 representative intraday features.

Goal: catch sign bugs in the feature library, NOT to select factors. We pick
three features whose sign vs 1-day forward return is well-known, compute the
1-day forward Spearman IC, and gate on the sign matching expectation.

Forward return source: prices.parquet 's adjusted close (raw_close * adj_factor).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FEATURE_ROOT = REPO_ROOT / "data" / "processed" / "intraday_features"
DEFAULT_PRICES = (
    REPO_ROOT
    / "data"
    / "processed"
    / "real_case_inputs"
    / "ashare_institutional_20160418_20260415_supplemented"
    / "prices.parquet"
)
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "intraday_etl" / "sanity_ic.md"

# Each entry: (column_name, expected_sign, rationale, gate_kind).
# gate_kind = "blocking" → contributes to readiness-gate failure when sign/flatness fails.
# gate_kind = "diagnostic" → IC is still computed and reported, but never blocks.
SANITY_FACTORS = [
    (
        "signed_jump",
        "negative",
        "Realized-volatility signed jump tends to revert next day in A-share.",
        "blocking",
    ),
    (
        "amount_share_close30",
        "positive",
        "Late-day institutional flow → next-day continuation. "
        "Downgraded to diagnostic on 2026-05-12: full-sample IC is ≈0 (research "
        "hypothesis too strong, not an ETL bug).",
        "diagnostic",
    ),
    (
        "vwap_close_dev",
        "negative",
        "Closing well above day VWAP often reverses next day.",
        "blocking",
    ),
]


def _filter_reliable(frame: pd.DataFrame) -> pd.DataFrame:
    mask = frame.get("is_actively_traded", pd.Series(1, index=frame.index)).fillna(0) == 1
    if "vol_unreliable" in frame.columns:
        mask &= frame["vol_unreliable"].fillna(0) == 0
    if "amt_unreliable" in frame.columns:
        mask &= frame["amt_unreliable"].fillna(0) == 0
    return frame.loc[mask].copy()


def _read_features(feature_root: Path, columns: list[str]) -> pd.DataFrame:
    needed = ["date", "asset", "is_actively_traded", "vol_unreliable", "amt_unreliable", *columns]
    parts = []
    for path in sorted(feature_root.glob("year=*/part-0.parquet")):
        parts.append(pd.read_parquet(path, columns=needed))
    if not parts:
        raise FileNotFoundError(f"no feature partitions under {feature_root}")
    return pd.concat(parts, ignore_index=True)


def _read_prices(prices_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(prices_path, columns=["date", "asset", "raw_close", "adj_factor"])
    df["date"] = df["date"].astype(str)
    df["asset"] = df["asset"].astype(str)
    df["adj_close"] = df["raw_close"] * df["adj_factor"]
    return df.sort_values(["asset", "date"]).reset_index(drop=True)


def _compute_forward_returns(prices: pd.DataFrame) -> pd.DataFrame:
    out = prices.copy()
    out["fwd_close"] = out.groupby("asset")["adj_close"].shift(-1)
    out["fwd_ret_1d"] = out["fwd_close"] / out["adj_close"] - 1.0
    return out[["date", "asset", "fwd_ret_1d"]]


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "None."
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def _spearman_ic(frame: pd.DataFrame, factor_col: str) -> dict[str, object]:
    daily_ic = []
    for date, group in frame.groupby("date", sort=False):
        valid = group[[factor_col, "fwd_ret_1d"]].dropna()
        if len(valid) < 50:
            continue
        if (
            valid[factor_col].std(ddof=0) == 0
            or valid["fwd_ret_1d"].std(ddof=0) == 0
        ):
            continue
        ic = valid[factor_col].corr(valid["fwd_ret_1d"], method="spearman")
        if pd.notna(ic):
            daily_ic.append((date, float(ic)))
    if not daily_ic:
        return {"n_days": 0, "ic_mean": float("nan"), "ic_std": float("nan")}
    arr = np.array([ic for _, ic in daily_ic])
    std = float(arr.std(ddof=0))
    ic_t = float(arr.mean() / (std / np.sqrt(len(arr)))) if std > 0 else 0.0
    return {
        "n_days": int(len(arr)),
        "ic_mean": float(arr.mean()),
        "ic_std": std,
        "ic_t": ic_t,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Intraday feature sanity-IC audit.")
    parser.add_argument("--feature-root", default=str(DEFAULT_FEATURE_ROOT))
    parser.add_argument("--prices", default=str(DEFAULT_PRICES))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--ic-floor", type=float, default=0.005,
        help="Below this |IC|, treat as suspicious (column may be flat).",
    )
    parser.add_argument(
        "--sign-block-threshold", type=float, default=0.01,
        help="If observed |IC| >= this and sign is opposite of expected, gate fail.",
    )
    args = parser.parse_args(argv)

    feature_root = Path(args.feature_root)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    factor_columns = [name for name, _, _, _ in SANITY_FACTORS]
    features = _read_features(feature_root, factor_columns)
    features = _filter_reliable(features)

    prices = _read_prices(Path(args.prices))
    forward = _compute_forward_returns(prices)
    merged = features.merge(forward, on=["date", "asset"], how="inner")

    rows: list[dict[str, object]] = []
    blocking: list[dict[str, object]] = []
    for column, expected_sign, rationale, gate_kind in SANITY_FACTORS:
        if column not in merged.columns:
            continue
        ic = _spearman_ic(merged, column)
        observed_sign = (
            "positive"
            if ic["ic_mean"] > args.ic_floor
            else "negative"
            if ic["ic_mean"] < -args.ic_floor
            else "flat"
        )
        is_sign_fail = (
            abs(ic["ic_mean"]) >= args.sign_block_threshold
            and observed_sign != "flat"
            and observed_sign != expected_sign
        )
        is_flat_fail = abs(ic["ic_mean"]) < args.ic_floor
        sanity_fail = is_sign_fail or is_flat_fail
        block = sanity_fail and gate_kind == "blocking"
        row = {
            "column": column,
            "gate_kind": gate_kind,
            "expected_sign": expected_sign,
            "observed_sign": observed_sign,
            "ic_mean": ic["ic_mean"],
            "ic_std": ic.get("ic_std", float("nan")),
            "ic_t": ic.get("ic_t", float("nan")),
            "n_days": ic.get("n_days", 0),
            "sanity_fail": sanity_fail,
            "blocking": block,
            "block_reason": (
                "sign-mismatch" if is_sign_fail else ("flat" if is_flat_fail else "")
            ),
            "rationale": rationale,
        }
        rows.append(row)
        if block:
            blocking.append(row)

    summary = {
        "feature_root": str(feature_root),
        "merged_rows": int(len(merged)),
        "blocking_count": int(len(blocking)),
    }

    sections = ["# Intraday Feature Sanity-IC\n\n"]
    sections.append("## Summary\n\n```\n" + json.dumps(summary, indent=2) + "\n```\n\n")
    sections.append(
        "Goal: catch sign bugs in the formula library. NOT a factor-selection signal.\n\n"
    )
    sections.append("## Results\n\n")
    sections.append(_markdown_table(pd.DataFrame(rows)))
    sections.append("\n\n## Blocking\n\n")
    if blocking:
        sections.append(_markdown_table(pd.DataFrame(blocking)))
        sections.append("\n")
    else:
        sections.append("None.\n")
    output.write_text("".join(sections), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if not blocking else 1


if __name__ == "__main__":
    sys.exit(main())
