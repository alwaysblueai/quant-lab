"""Per-column value distribution and outlier audit for the promoted feature library.

Computes percentile snapshots and flags non-finite values. Hard-fails on inf
or sign violations; emits recommended winsorize ranges as guidance for
downstream research code (advisory only).
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
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "intraday_etl" / "value_distribution.md"

# Columns whose median should be non-negative (shares, counts, vols).
NON_NEGATIVE_COLUMNS = {
    "rv_1m",
    "rv_5m",
    "rv_15m",
    "bv_5m",
    "jump_5m",
    "rv_pos_5m",
    "rv_neg_5m",
    "rv_morning",
    "rv_afternoon",
    "amount_share_open30",
    "amount_share_pre_lunch30",
    "amount_share_post_lunch30",
    "amount_share_close30",
    "amount_share_morning",
    "amount_share_afternoon",
    "amount_hhi",
    "amount_top10_share",
    "minutes_to_50pct_amount",
    "pos_amount_share",
    "neg_amount_share",
    "zero_ret_amount_share",
    "amihud_intraday",
    "avg_gap_between_trades",
    "time_at_extremes_share",
    "acceleration_max",
    "limit_up_touch_count",
    "limit_up_open_count",
    "limit_down_touch_count",
    "limit_down_open_count",
    "minutes_at_high_count",
    "minutes_at_low_count",
    "sign_flip_count",
    "max_abs_return_zscore",
    "roll_spread_proxy",
}


def _load_feature_columns(feature_root: Path) -> list[str]:
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from alpha_lab.intraday.features import BATCH1234_FEATURE_COLUMNS

    candidate = list(BATCH1234_FEATURE_COLUMNS)
    sample = next(iter(feature_root.glob("year=*/part-0.parquet")), None)
    if sample is None:
        return candidate
    import pyarrow.parquet as pq

    schema = pq.ParquetFile(sample).schema_arrow
    have = {f.name for f in schema}
    return [c for c in candidate if c in have]


def _load_features(feature_root: Path, columns: list[str]) -> pd.DataFrame:
    needed = ["date", "asset", "is_actively_traded", "vol_unreliable", "amt_unreliable"]
    needed += list(columns)
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


def _column_stats(values: pd.Series) -> dict[str, float]:
    numeric = pd.to_numeric(values, errors="coerce")
    finite = numeric[np.isfinite(numeric)]
    inf_count = int(((numeric.notna()) & (~np.isfinite(numeric))).sum())
    if finite.empty:
        return {
            "n": int(numeric.notna().sum()),
            "n_inf": inf_count,
            "mean": float("nan"),
            "std": float("nan"),
            "p1": float("nan"),
            "p5": float("nan"),
            "p25": float("nan"),
            "p50": float("nan"),
            "p75": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
        }
    return {
        "n": int(len(finite)),
        "n_inf": inf_count,
        "mean": float(finite.mean()),
        "std": float(finite.std(ddof=0)),
        "p1": float(finite.quantile(0.01)),
        "p5": float(finite.quantile(0.05)),
        "p25": float(finite.quantile(0.25)),
        "p50": float(finite.quantile(0.50)),
        "p75": float(finite.quantile(0.75)),
        "p95": float(finite.quantile(0.95)),
        "p99": float(finite.quantile(0.99)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Intraday feature value distribution audit.")
    parser.add_argument("--feature-root", default=str(DEFAULT_FEATURE_ROOT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args(argv)

    feature_root = Path(args.feature_root)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    columns = _load_feature_columns(feature_root)
    frame = _load_features(feature_root, columns)
    reliable = _filter_reliable(frame)

    rows: list[dict[str, object]] = []
    blocking: list[dict[str, object]] = []
    for col in columns:
        if col not in reliable.columns:
            continue
        stats = _column_stats(reliable[col])
        is_inf_block = stats["n_inf"] > 0
        is_sign_block = (
            col in NON_NEGATIVE_COLUMNS
            and np.isfinite(stats.get("p50", float("nan")))
            and stats["p50"] < 0
        )
        block = is_inf_block or is_sign_block
        row = {
            "column": col,
            **stats,
            "blocking": block,
            "block_reason": (
                "inf"
                if is_inf_block
                else ("sign-violation" if is_sign_block else "")
            ),
        }
        rows.append(row)
        if block:
            blocking.append(row)

    df = pd.DataFrame(rows)
    summary = {
        "feature_root": str(feature_root),
        "reliable_rows": int(len(reliable)),
        "n_columns": int(len(columns)),
        "blocking_count": int(len(blocking)),
    }

    sections = ["# Intraday Feature Value Distribution\n\n"]
    sections.append("## Summary\n\n```\n" + json.dumps(summary, indent=2) + "\n```\n\n")
    sections.append("## Blocking columns\n\n")
    if blocking:
        sections.append(_markdown_table(pd.DataFrame(blocking)))
        sections.append("\n\n")
    else:
        sections.append("None.\n\n")
    sections.append("## Per-column distribution (reliable rows)\n\n")
    sections.append(_markdown_table(df))
    sections.append("\n\n## Suggested winsorize bounds (advisory)\n\n")
    advisory_cols = [
        c
        for c in columns
        if c in df["column"].values
        and (
            c.endswith("_dev")
            or c.startswith("ret_")
            or c == "signed_jump"
            or c == "signed_amount_imbalance"
            or c == "amihud_intraday"
            or c == "max_abs_return_zscore"
            or c == "acceleration_max"
        )
    ]
    advisory = (
        df[df["column"].isin(advisory_cols)][["column", "p1", "p99"]]
        .rename(columns={"p1": "lower", "p99": "upper"})
    )
    sections.append(_markdown_table(advisory))
    sections.append("\n")
    output.write_text("".join(sections), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if not blocking else 1


if __name__ == "__main__":
    sys.exit(main())
