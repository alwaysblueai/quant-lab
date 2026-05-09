"""NaN profile of the promoted intraday feature library.

Computes per-column NaN rates broken out by exchange and board on the reliable-rows
subset (is_actively_traded=1 AND vol_unreliable=0 AND amt_unreliable=0). Acts as a
hard gate: any column whose ordinary-day NaN rate exceeds threshold is flagged.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FEATURE_ROOT = REPO_ROOT / "data" / "processed" / "intraday_features"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "intraday_etl" / "nan_profile.md"

# Columns whose NaN rate is structurally higher (small-N requirements). Allowed
# higher gate so we do not block on legitimate data sparseness.
HIGH_NAN_TOLERANCE_COLUMNS = {
    "intraday_skew_1m",
    "intraday_kurt_1m",
    "intraday_skew_5m",
    "intraday_kurt_5m",
    "signed_jump",
    "volume_kurt_1m",
    "ret_autocorr_1m_lag1",
    "amount_autocorr_1m_lag1",
    "max_abs_return_zscore",
    "roll_spread_proxy",
    "limit_up_touch_count",
    "limit_up_open_count",
    "limit_down_touch_count",
    "limit_down_open_count",
    "gap_fill_ratio",
}


def _load_feature_columns(feature_root: Path) -> list[str]:
    """Detect feature columns from the actual parquet schema, falling back to
    BATCH1234_FEATURE_COLUMNS minus base/status columns. This lets the audit
    auto-adapt to whichever batches have been promoted."""

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


def _exchange_of(asset: str) -> str:
    if asset.endswith(".SH"):
        return "SH"
    if asset.endswith(".SZ"):
        return "SZ"
    if asset.endswith(".BJ"):
        return "BJ"
    return "OTHER"


def _board_of(asset: str) -> str:
    code = asset.split(".", 1)[0]
    if asset.endswith(".SH"):
        return "STAR" if code.startswith("688") else "SH_MAIN"
    if asset.endswith(".SZ"):
        if code.startswith("300"):
            return "CHINEXT"
        if code.startswith("000") or code.startswith("001"):
            return "SZ_MAIN"
        return "OTHER"
    if asset.endswith(".BJ"):
        return "BJ"
    return "OTHER"


def _load_features(feature_root: Path, columns: list[str]) -> pd.DataFrame:
    needed = ["date", "asset", "is_actively_traded", "vol_unreliable", "amt_unreliable"]
    needed += list(columns)
    parts = []
    for path in sorted(feature_root.glob("year=*/part-0.parquet")):
        df = pd.read_parquet(path, columns=[c for c in needed if c is not None])
        parts.append(df)
    if not parts:
        raise FileNotFoundError(f"no feature partitions under {feature_root}")
    out = pd.concat(parts, ignore_index=True)
    out["exchange"] = out["asset"].map(_exchange_of)
    out["board"] = out["asset"].map(_board_of)
    return out


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Intraday feature NaN profile.")
    parser.add_argument("--feature-root", default=str(DEFAULT_FEATURE_ROOT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--gate-default", type=float, default=0.05,
        help="Max NaN rate for a default-gated column on reliable rows.",
    )
    parser.add_argument(
        "--gate-high-tolerance", type=float, default=0.20,
        help="Max NaN rate for HIGH_NAN_TOLERANCE_COLUMNS on reliable rows.",
    )
    args = parser.parse_args(argv)

    feature_root = Path(args.feature_root)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    columns = _load_feature_columns(feature_root)
    frame = _load_features(feature_root, columns)
    reliable = _filter_reliable(frame)

    overall_rows: list[dict[str, object]] = []
    by_exchange: list[dict[str, object]] = []
    blocking: list[dict[str, object]] = []
    n_total = max(len(reliable), 1)

    for col in columns:
        if col not in reliable.columns:
            continue
        nan_rate = float(reliable[col].isna().mean())
        gate = (
            args.gate_high_tolerance
            if col in HIGH_NAN_TOLERANCE_COLUMNS
            else args.gate_default
        )
        is_block = nan_rate > gate
        overall_rows.append(
            {
                "column": col,
                "n_reliable": int(n_total),
                "nan_rate": nan_rate,
                "gate": gate,
                "blocking": is_block,
            }
        )
        if is_block:
            blocking.append(overall_rows[-1])

        for exchange, group in reliable.groupby("exchange"):
            if len(group) == 0:
                continue
            by_exchange.append(
                {
                    "column": col,
                    "exchange": exchange,
                    "n_reliable": int(len(group)),
                    "nan_rate": float(group[col].isna().mean()),
                }
            )

    overall_df = pd.DataFrame(overall_rows).sort_values("nan_rate", ascending=False)
    by_ex_df = pd.DataFrame(by_exchange).sort_values(["column", "exchange"])

    summary = {
        "feature_root": str(feature_root),
        "total_rows": int(len(frame)),
        "reliable_rows": int(n_total),
        "n_columns": int(len(columns)),
        "gate_default": float(args.gate_default),
        "gate_high_tolerance": float(args.gate_high_tolerance),
        "blocking_count": int(len(blocking)),
    }

    sections = ["# Intraday Feature NaN Profile\n\n"]
    sections.append("## Summary\n\n```\n" + json.dumps(summary, indent=2) + "\n```\n\n")
    sections.append("## Blocking columns\n\n")
    if blocking:
        sections.append(_markdown_table(overall_df[overall_df["blocking"]]))
        sections.append("\n\n")
    else:
        sections.append("None.\n\n")
    sections.append("## Per-column NaN rate (reliable rows)\n\n")
    sections.append(_markdown_table(overall_df))
    sections.append("\n\n## Per-column / exchange NaN rate\n\n")
    sections.append(_markdown_table(by_ex_df))
    sections.append("\n")
    output.write_text("".join(sections), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if not blocking else 1


if __name__ == "__main__":
    sys.exit(main())
