"""Cross-feature redundancy audit for the promoted intraday feature library.

Computes the Spearman correlation matrix on a sampled subset of reliable rows.
For column pairs whose absolute correlation exceeds a threshold, classifies
them against a known-baseline catalogue and emits the unexpected pairs for
human review. Writes a Markdown report and an optional PNG heatmap.

This script is informational; it does not gate anything.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FEATURE_ROOT = REPO_ROOT / "data" / "processed" / "intraday_features"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "intraday_etl" / "redundancy_audit.md"

# Pairs we already understand. Keys are sorted tuples of (col_a, col_b).
KNOWN_HIGH_CORR: dict[tuple[str, str], str] = {
    tuple(sorted(("rv_5m", "rv_pos_5m"))): "partial-sum identity",
    tuple(sorted(("rv_5m", "rv_neg_5m"))): "partial-sum identity",
    tuple(sorted(("rv_5m", "rv_15m"))): "cross-window aggregation",
    tuple(sorted(("rv_5m", "rv_1m"))): "cross-window aggregation",
    tuple(sorted(("rv_morning", "rv_5m"))): "subset",
    tuple(sorted(("rv_afternoon", "rv_5m"))): "subset",
    tuple(sorted(("amount_share_morning", "amount_share_afternoon"))): "sums to 1",
    tuple(sorted(("vwap_close_dev", "intraday_ret"))): "monotone proxy",
    tuple(sorted(("vwap_high_dev", "vwap_low_dev"))): "co-movement",
    tuple(sorted(("intraday_skew_5m", "signed_jump"))): "same-sign asymmetry",
    tuple(sorted(("pos_amount_share", "signed_amount_imbalance"))): "linear proxy",
    tuple(sorted(("neg_amount_share", "signed_amount_imbalance"))): "linear proxy",
    tuple(sorted(("pos_amount_share", "neg_amount_share"))): "sum-to-1 minus zero",
    tuple(sorted(("amount_share_close30", "amount_share_afternoon"))): "subset",
    tuple(sorted(("amount_share_open30", "amount_share_morning"))): "subset",
    tuple(sorted(("limit_up_touch_count", "minutes_at_high_count"))): "limit hit ⇒ at high",
    tuple(sorted(("limit_down_touch_count", "minutes_at_low_count"))): "limit hit ⇒ at low",
    tuple(sorted(("intraday_skew_1m", "intraday_skew_5m"))): "same metric, different scale",
    tuple(sorted(("intraday_kurt_1m", "intraday_kurt_5m"))): "same metric, different scale",
}

FEATURE_GROUP_HINT_FILE = REPO_ROOT / "src" / "alpha_lab" / "intraday" / "features.py"


def _load_feature_columns(feature_root: Path) -> list[str]:
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from alpha_lab.intraday.features import BATCH1234_FEATURE_COLUMNS

    candidate = list(BATCH1234_FEATURE_COLUMNS)
    sample = next(iter(feature_root.glob("year=*/part-0.parquet")), None)
    if sample is None:
        return candidate

    import pyarrow.parquet as pq

    schema = pq.ParquetFile(sample).schema_arrow
    have = {field.name for field in schema}
    return [column for column in candidate if column in have]


def _read_features(feature_root: Path, columns: list[str]) -> pd.DataFrame:
    needed = ["date", "asset", "is_actively_traded", "vol_unreliable", "amt_unreliable"]
    needed.extend(columns)
    parts = []
    for path in sorted(feature_root.glob("year=*/part-0.parquet")):
        df = pd.read_parquet(path, columns=[c for c in needed if c is not None])
        parts.append(df)
    if not parts:
        raise FileNotFoundError(f"no feature partitions under {feature_root}")
    return pd.concat(parts, ignore_index=True)


def _filter_reliable(frame: pd.DataFrame) -> pd.DataFrame:
    mask = frame.get("is_actively_traded", pd.Series(1, index=frame.index)).fillna(0) == 1
    if "vol_unreliable" in frame.columns:
        mask &= frame["vol_unreliable"].fillna(0) == 0
    if "amt_unreliable" in frame.columns:
        mask &= frame["amt_unreliable"].fillna(0) == 0
    return frame.loc[mask].copy()


def _spearman_corr(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    available = [c for c in columns if c in frame.columns]
    return frame[available].corr(method="spearman")


def _high_corr_pairs(corr: pd.DataFrame, threshold: float) -> list[dict[str, object]]:
    pairs: list[dict[str, object]] = []
    cols = list(corr.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            value = corr.loc[a, b]
            if pd.isna(value):
                continue
            if abs(value) >= threshold:
                key = tuple(sorted((a, b)))
                pairs.append(
                    {
                        "col_a": a,
                        "col_b": b,
                        "abs_corr": float(abs(value)),
                        "corr": float(value),
                        "baseline": KNOWN_HIGH_CORR.get(key, "UNEXPECTED"),
                    }
                )
    pairs.sort(key=lambda row: row["abs_corr"], reverse=True)
    return pairs


def _format_pairs_markdown(pairs: list[dict[str, object]]) -> str:
    if not pairs:
        return "No pairs found above threshold.\n"
    lines = ["| col_a | col_b | corr | abs | baseline |", "| --- | --- | --- | --- | --- |"]
    for row in pairs:
        lines.append(
            "| {col_a} | {col_b} | {corr:.4f} | {abs_corr:.4f} | {baseline} |".format(**row)
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Intraday feature redundancy audit.")
    parser.add_argument("--feature-root", default=str(DEFAULT_FEATURE_ROOT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--sample-rows", type=int, default=200_000)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=20260509)
    args = parser.parse_args(argv)

    feature_root = Path(args.feature_root)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    columns = _load_feature_columns(feature_root)
    frame = _read_features(feature_root, columns)
    reliable = _filter_reliable(frame)
    if len(reliable) > args.sample_rows:
        reliable = reliable.sample(n=args.sample_rows, random_state=args.seed)

    corr = _spearman_corr(reliable, columns)
    pairs = _high_corr_pairs(corr, args.threshold)
    unexpected = [pair for pair in pairs if pair["baseline"] == "UNEXPECTED"]

    summary = {
        "feature_root": str(feature_root),
        "total_rows": int(len(frame)),
        "reliable_rows": int(len(reliable)),
        "n_columns": int(len(columns)),
        "threshold": float(args.threshold),
        "n_pairs_above_threshold": int(len(pairs)),
        "n_unexpected_pairs": int(len(unexpected)),
    }

    sections = ["# Intraday Feature Redundancy Audit\n\n"]
    sections.append("## Summary\n\n")
    sections.append("```\n" + json.dumps(summary, indent=2) + "\n```\n\n")
    sections.append(f"## High-correlation pairs (|rho| >= {args.threshold})\n\n")
    sections.append(_format_pairs_markdown(pairs))
    sections.append("\n## Unexpected high-correlation pairs (review needed)\n\n")
    sections.append(_format_pairs_markdown(unexpected))
    output.write_text("".join(sections), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
