"""Verify Batch 2 intraday feature expansion output."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from alpha_lab.intraday.features import BATCH1_FEATURE_COLUMNS, BATCH2_FEATURE_COLUMNS

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FEATURE_ROOT = REPO_ROOT / "data" / "processed" / "intraday_features_batch2"
DEFAULT_BASE_ROOT = REPO_ROOT / "data" / "processed" / "intraday_features"
DEFAULT_REPORT = REPO_ROOT / "outputs" / "intraday_etl" / "batch2_summary_2024_2025.md"
DEFAULT_SNAPSHOT = REPO_ROOT / "tests" / "intraday" / "golden" / "full_run_snapshot.json"

SNAPSHOT_ASSETS = [
    "600519.SH",
    "601318.SH",
    "600036.SH",
    "600004.SH",
    "920077.BJ",
    "920575.BJ",
    "688375.SH",
]
SNAPSHOT_DATES = ["2024-10-21", "2025-01-02"]
SNAPSHOT_COLUMNS = BATCH1_FEATURE_COLUMNS + BATCH2_FEATURE_COLUMNS


def _parse_years(raw: str | None) -> list[int] | None:
    if not raw:
        return None
    return sorted({int(token.strip()) for token in raw.split(",") if token.strip()})


def _discover_years(root: Path) -> list[int]:
    return sorted(
        int(path.name.split("=")[1])
        for path in root.glob("year=*")
        if path.is_dir() and path.name.split("=")[1].isdigit()
    )


def _read_dataset(root: Path, years: list[int]) -> pd.DataFrame:
    parts = []
    for year in years:
        path = root / f"year={year}" / "part-0.parquet"
        if not path.exists():
            raise FileNotFoundError(f"missing parquet partition: {path}")
        parts.append(pd.read_parquet(path))
    frame = pd.concat(parts, ignore_index=True)
    frame["date"] = frame["date"].astype(str)
    frame["asset"] = frame["asset"].astype(str)
    return frame.sort_values(["date", "asset"], kind="mergesort").reset_index(drop=True)


def _exchange(asset: pd.Series) -> pd.Series:
    return asset.str[-2:]


def _frame_hash(frame: pd.DataFrame) -> str:
    normalized = frame.copy()
    for column in normalized.columns:
        if pd.api.types.is_datetime64_any_dtype(normalized[column]):
            normalized[column] = normalized[column].astype("datetime64[ns]").astype(str)
    row_hashes = pd.util.hash_pandas_object(normalized, index=False).to_numpy(dtype="uint64")
    return hashlib.sha256(row_hashes.tobytes()).hexdigest()


def _format_value(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, float):
        if abs(value) >= 1000:
            return f"{value:,.2f}"
        if abs(value) >= 1:
            return f"{value:.6f}"
        return f"{value:.8f}"
    return str(value)


def _markdown_table(frame: pd.DataFrame, *, max_rows: int | None = None) -> str:
    if max_rows is not None:
        frame = frame.head(max_rows)
    if frame.empty:
        return "_无数据_\n"
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(_format_value(row[column]) for column in columns) + " |")
    return "\n".join(lines) + "\n"


def _reliable_mask(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["is_actively_traded"].fillna(0).astype(int).eq(1)
        & frame["vol_unreliable"].fillna(0).astype(int).eq(0)
        & frame["amt_unreliable"].fillna(0).astype(int).eq(0)
    )


def _nan_by_exchange(feature: pd.DataFrame) -> pd.DataFrame:
    frame = feature.copy()
    frame["exchange"] = _exchange(frame["asset"])
    reliable = _reliable_mask(frame)
    rows = []
    for column in BATCH2_FEATURE_COLUMNS:
        for exchange, block in frame.groupby("exchange", sort=True):
            reliable_block = block.loc[reliable.loc[block.index]]
            rows.append(
                {
                    "new_col": column,
                    "exchange": exchange,
                    "rows": len(block),
                    "nan_rate": block[column].isna().mean(),
                    "reliable_rows": len(reliable_block),
                    "nan_rate_reliable": (
                        reliable_block[column].isna().mean() if len(reliable_block) else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def _feature_nan_gates(nan_by_exchange: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        nan_by_exchange.groupby("new_col", as_index=False)
        .apply(
            lambda block: pd.Series(
                {
                    "reliable_rows": int(block["reliable_rows"].sum()),
                    "nan_rows_reliable": int(
                        round((block["nan_rate_reliable"] * block["reliable_rows"]).sum())
                    ),
                }
            ),
            include_groups=False,
        )
        .reset_index(drop=True)
    )
    grouped["observed_rate"] = grouped["nan_rows_reliable"] / grouped["reliable_rows"].clip(
        lower=1
    )
    grouped["threshold"] = np.where(grouped["new_col"].eq("volume_kurt_1m"), 0.01, 0.005)
    grouped["passed"] = grouped["observed_rate"] < grouped["threshold"]
    grouped["gate_type"] = "hard"
    grouped["blocking"] = ~grouped["passed"]
    return grouped[
        [
            "new_col",
            "reliable_rows",
            "nan_rows_reliable",
            "observed_rate",
            "threshold",
            "passed",
            "gate_type",
            "blocking",
        ]
    ]


def _sanity_checks(feature: pd.DataFrame) -> pd.DataFrame:
    reliable = feature[_reliable_mask(feature)].copy()
    rows = []

    morning_afternoon = (
        reliable["amount_share_morning"] + reliable["amount_share_afternoon"] - 1.0
    ).abs()
    rows.append(
        {
            "check": "amount_share_morning + amount_share_afternoon == 1",
            "observed": float(morning_afternoon.max()),
            "passed": bool(morning_afternoon.fillna(0).max() <= 1e-9),
            "gate_type": "hard",
        }
    )

    window_sum = (
        reliable["amount_share_open30"]
        + reliable["amount_share_pre_lunch30"]
        + reliable["amount_share_post_lunch30"]
        + reliable["amount_share_close30"]
    )
    rows.append(
        {
            "check": "non-overlap amount window shares <= 1",
            "observed": float(window_sum.max()),
            "passed": bool(window_sum.fillna(0).max() <= 1.0 + 1e-12),
            "gate_type": "hard",
        }
    )

    active_minutes = (
        reliable["n_minutes_traded"] + reliable["n_minutes_zero_volume"]
    ).clip(lower=1)
    hhi = pd.to_numeric(reliable["amount_hhi"], errors="coerce")
    hhi_low_diff = (1.0 / active_minutes - hhi).clip(lower=0.0)
    hhi_high_diff = (hhi - 1.0).clip(lower=0.0)
    rows.append(
        {
            "check": "amount_hhi within [1 / active_minutes, 1]",
            "observed": f"low_max={hhi_low_diff.max():.6g}, high_max={hhi_high_diff.max():.6g}",
            "passed": bool(hhi_low_diff.fillna(0).max() <= 1e-12)
            and bool(hhi_high_diff.fillna(0).max() <= 1e-12),
            "gate_type": "hard",
        }
    )

    top10 = pd.to_numeric(reliable["amount_top10_share"], errors="coerce")
    top10_bad = top10.lt(-1e-12) | top10.gt(1.0 + 1e-12)
    rows.append(
        {
            "check": "amount_top10_share within [0, 1]",
            "observed": int(top10_bad.sum()),
            "passed": bool(top10_bad.sum() == 0),
            "gate_type": "hard",
        }
    )

    minutes_50 = pd.to_numeric(reliable["minutes_to_50pct_amount"], errors="coerce")
    minutes_bad = minutes_50.lt(1.0) | minutes_50.gt(active_minutes)
    rows.append(
        {
            "check": "minutes_to_50pct_amount within active session bounds",
            "observed": int(minutes_bad.sum()),
            "passed": bool(minutes_bad.sum() == 0),
            "gate_type": "hard",
        }
    )

    dispersion = pd.to_numeric(reliable["vwap_minute_dispersion"], errors="coerce")
    dispersion_bad = dispersion.lt(-1e-12)
    rows.append(
        {
            "check": "vwap_minute_dispersion >= 0",
            "observed": int(dispersion_bad.sum()),
            "passed": bool(dispersion_bad.sum() == 0),
            "gate_type": "hard",
        }
    )

    vwap_dev_cols = [
        "vwap_close_dev",
        "vwap_open_dev",
        "vwap_high_dev",
        "vwap_low_dev",
        "vwap_minute_dispersion",
    ]
    finite_values = reliable[vwap_dev_cols].apply(pd.to_numeric, errors="coerce")
    finite_bad = ~np.isfinite(finite_values.to_numpy())
    rows.append(
        {
            "check": "VWAP deviation values finite on reliable rows",
            "observed": int(finite_bad.sum()),
            "passed": bool(finite_bad.sum() == 0),
            "gate_type": "hard",
        }
    )

    out = pd.DataFrame(rows)
    out["blocking"] = out["gate_type"].eq("hard") & ~out["passed"]
    return out


def _file_size_gates(root: Path, years: list[int]) -> pd.DataFrame:
    rows = []
    for year in years:
        path = root / f"year={year}" / "part-0.parquet"
        size_mb = path.stat().st_size / 1024 / 1024
        passed = 250 <= size_mb <= 700
        rows.append(
            {
                "year": year,
                "size_mb": size_mb,
                "threshold": "250 <= size_mb <= 700",
                "passed": bool(passed),
                "gate_type": "hard",
                "blocking": not bool(passed),
            }
        )
    return pd.DataFrame(rows)


def _base_hash_gates(base: pd.DataFrame, feature: pd.DataFrame) -> pd.DataFrame:
    rows = []
    missing_base_cols = [column for column in base.columns if column not in feature.columns]
    base_cols = [column for column in base.columns if column in feature.columns]
    for year, base_year in base.groupby(base["date"].str[:4].astype(int), sort=True):
        feature_year = feature[feature["date"].str[:4].astype(int) == year]
        before = _frame_hash(
            base_year[base_cols].sort_values(["date", "asset"]).reset_index(drop=True)
        )
        after = _frame_hash(
            feature_year[base_cols].sort_values(["date", "asset"]).reset_index(drop=True)
        )
        passed = before == after and len(base_year) == len(feature_year) and not missing_base_cols
        rows.append(
            {
                "year": int(year),
                "rows_base": len(base_year),
                "rows_feature": len(feature_year),
                "missing_base_cols": ",".join(missing_base_cols),
                "hash_before": before,
                "hash_after": after,
                "passed": passed,
                "gate_type": "hard",
                "blocking": not passed,
            }
        )
    return pd.DataFrame(rows)


def _write_snapshot(feature: pd.DataFrame, path: Path) -> dict[str, object]:
    mask = feature["asset"].isin(SNAPSHOT_ASSETS) & feature["date"].isin(SNAPSHOT_DATES)
    snapshot = feature.loc[mask, ["date", "asset", *SNAPSHOT_COLUMNS]].sort_values(
        ["date", "asset"]
    )

    one_line = feature[
        feature["is_session_active"].fillna(0).astype(int).eq(1)
        & feature["is_actively_traded"].fillna(0).astype(int).eq(0)
    ].sort_values(["date", "asset"])
    if not one_line.empty:
        extra = one_line.iloc[[0]][["date", "asset", *SNAPSHOT_COLUMNS]]
        snapshot = pd.concat([snapshot, extra], ignore_index=True).drop_duplicates(
            ["date", "asset"], keep="first"
        )

    fallback_pairs = []
    existing_pairs = set(zip(snapshot["date"], snapshot["asset"], strict=False))
    requested_pairs = {(date, asset) for date in SNAPSHOT_DATES for asset in SNAPSHOT_ASSETS}
    missing = sorted(requested_pairs.difference(existing_pairs))
    fallback_rows = []
    for requested_date, asset in missing:
        candidates = feature[feature["asset"].eq(asset)].copy()
        if candidates.empty:
            continue
        candidates["_distance"] = (
            pd.to_datetime(candidates["date"]) - pd.Timestamp(requested_date)
        ).abs()
        row = candidates.sort_values(["_distance", "date"]).iloc[0]
        fallback_rows.append(row[["date", "asset", *SNAPSHOT_COLUMNS]])
        fallback_pairs.append(
            {
                "requested_date": requested_date,
                "asset": asset,
                "fallback_date": row["date"],
            }
        )

    if fallback_rows:
        snapshot = pd.concat(
            [snapshot, pd.DataFrame(fallback_rows)],
            ignore_index=True,
        ).drop_duplicates(["date", "asset"], keep="first")
        existing_pairs = set(zip(snapshot["date"], snapshot["asset"], strict=False))
        missing = sorted(requested_pairs.difference(existing_pairs))

    payload = {
        "source": "intraday_features_batch2",
        "snapshot_dates": SNAPSHOT_DATES,
        "snapshot_assets": SNAPSHOT_ASSETS,
        "missing_requested_pairs": [{"date": date, "asset": asset} for date, asset in missing],
        "fallback_pairs": fallback_pairs,
        "rows": json.loads(snapshot.replace([np.inf, -np.inf], np.nan).to_json(orient="records")),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize and gate Batch 2 intraday feature output.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--feature-root", default=str(DEFAULT_FEATURE_ROOT))
    parser.add_argument("--base-root", default=str(DEFAULT_BASE_ROOT))
    parser.add_argument("--years", default="2024,2025")
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--snapshot", default=str(DEFAULT_SNAPSHOT))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    feature_root = Path(args.feature_root)
    base_root = Path(args.base_root)
    years = _parse_years(args.years) or _discover_years(feature_root)

    feature = _read_dataset(feature_root, years)
    base = _read_dataset(base_root, years)

    nan_exchange = _nan_by_exchange(feature)
    nan_gates = _feature_nan_gates(nan_exchange)
    sanity = _sanity_checks(feature)
    file_sizes = _file_size_gates(feature_root, years)
    base_hash = _base_hash_gates(base, feature)
    snapshot_payload = _write_snapshot(feature, Path(args.snapshot))

    hard_tables = [nan_gates, sanity, file_sizes, base_hash]
    blocking_count = int(sum(table["blocking"].sum() for table in hard_tables))

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = [
        "# Intraday Batch 2 Summary",
        "",
        f"- feature root: `{args.feature_root}`",
        f"- base root: `{args.base_root}`",
        f"- years: `{years}`",
        f"- rows: `{len(feature)}`",
        f"- feature columns: `{len(BATCH2_FEATURE_COLUMNS)}`",
        f"- total snapshot columns: `{len(SNAPSHOT_COLUMNS)}`",
        f"- blocking hard gate count: `{blocking_count}`",
        f"- snapshot: `{args.snapshot}`",
        "",
        "## Notes",
        "",
        "- Gate denominators use reliable rows only: `is_actively_traded=1`, "
        "`vol_unreliable=0`, and `amt_unreliable=0`.",
        "- `amount_share_close30` includes the 15:00 bar by contract.",
        "- `vwap_minute_dispersion` is unweighted minute-close population std "
        "(`ddof=0`) divided by day VWAP.",
        "",
        "## Reliable NaN Gates",
        "",
        _markdown_table(nan_gates),
        "",
        "## NaN By Exchange",
        "",
        _markdown_table(nan_exchange, max_rows=300),
        "",
        "## Sanity Checks",
        "",
        _markdown_table(sanity),
        "",
        "## File Size Gates",
        "",
        _markdown_table(file_sizes),
        "",
        "## Base Hash Gates",
        "",
        _markdown_table(base_hash),
        "",
        "## Snapshot Coverage",
        "",
        "### Missing Requested Pairs",
        "",
        _markdown_table(pd.DataFrame(snapshot_payload["missing_requested_pairs"])),
        "",
        "### Fallback Pairs",
        "",
        _markdown_table(pd.DataFrame(snapshot_payload["fallback_pairs"])),
    ]
    report_path.write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote report: {report_path}")
    print(f"Wrote snapshot: {args.snapshot}")
    print(f"blocking_hard_gate_count={blocking_count}")
    return 0 if blocking_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
