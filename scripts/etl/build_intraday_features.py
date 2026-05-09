"""Build wider intraday features from Stage B minute panels.

This script is the Stage C expansion path. It keeps the already verified Group A
daily PV table as the base and left-joins pure formula outputs computed from
Stage B. By default it writes to a separate output root so the verified base
dataset is not overwritten accidentally.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from collections.abc import Iterable
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from alpha_lab.intraday.features import (
    BATCH1_FEATURE_COLUMNS,
    BATCH2_FEATURE_COLUMNS,
    BATCH3_FEATURE_COLUMNS,
    BATCH4_FEATURE_COLUMNS,
    BATCH1234_FEATURE_COLUMNS,
    compute_batch1_feature_frame,
    compute_batch2_feature_frame,
    compute_batch3_feature_frame,
    compute_batch4_feature_frame,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PANEL_ROOT = REPO_ROOT / "data" / "processed" / "minute_panel"
DEFAULT_BASE_ROOT = REPO_ROOT / "data" / "processed" / "intraday_features"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "processed" / "intraday_features_batch1"
DEFAULT_PRICES_PATH = (
    REPO_ROOT
    / "data"
    / "processed"
    / "real_case_inputs"
    / "ashare_institutional_20160418_20260415_supplemented"
    / "prices.parquet"
)


def _sql_path(path: Path | str) -> str:
    return str(path).replace("\\", "/").replace("'", "''")


def _parse_years(raw: str | None) -> list[int] | None:
    if not raw:
        return None
    return sorted({int(token.strip()) for token in raw.split(",") if token.strip()})


def _parse_assets(raw: str | None) -> set[str] | None:
    if not raw:
        return None
    assets = {token.strip().upper() for token in raw.split(",") if token.strip()}
    return assets or None


def _chunked(values: list[str], size: int) -> Iterable[list[str]]:
    if size <= 0:
        raise ValueError("chunk size must be positive")
    for idx in range(0, len(values), size):
        yield values[idx : idx + size]


def _prepare_output(output_root: Path, years: list[int] | None, overwrite: bool) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    existing_year_dirs = [
        path
        for path in output_root.glob("year=*")
        if path.is_dir() and (years is None or int(path.name.split("=")[1]) in years)
    ]
    if existing_year_dirs and not overwrite:
        existing = ", ".join(str(path) for path in existing_year_dirs[:5])
        raise FileExistsError(f"Output year partitions already exist: {existing}")
    if overwrite:
        for path in existing_year_dirs:
            shutil.rmtree(path)

    tmp_root = output_root / "_in_progress" / time.strftime("%Y%m%d_%H%M%S")
    tmp_root.mkdir(parents=True, exist_ok=False)
    return tmp_root


def _finalize_output(tmp_root: Path, output_root: Path) -> None:
    for tmp_year_dir in sorted(tmp_root.glob("year=*")):
        final_year_dir = output_root / tmp_year_dir.name
        if final_year_dir.exists():
            raise FileExistsError(f"Final partition exists before finalize: {final_year_dir}")
        os.replace(tmp_year_dir, final_year_dir)
    summary_path = tmp_root / "_feature_summary.json"
    if summary_path.exists():
        os.replace(summary_path, output_root / "_feature_summary.json")
    try:
        tmp_root.rmdir()
        tmp_root.parent.rmdir()
    except OSError:
        pass


def _discover_base_years(base_root: Path) -> list[int]:
    return sorted(
        int(path.name.split("=")[1])
        for path in base_root.glob("year=*")
        if path.is_dir() and path.name.split("=")[1].isdigit()
    )


def _read_base_year(base_root: Path, year: int) -> pd.DataFrame:
    path = base_root / f"year={year}" / "part-0.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Base Stage C partition not found: {path}")
    return pd.read_parquet(path)


def _asset_filter_sql(assets: list[str]) -> str:
    quoted = ", ".join("'" + asset.replace("'", "''") + "'" for asset in assets)
    return f"asset IN ({quoted})"


def _load_minute_chunk(
    con: duckdb.DuckDBPyConnection,
    panel_file: Path,
    assets: list[str],
) -> pd.DataFrame:
    return con.execute(
        f"""
        SELECT
            date,
            asset,
            datetime,
            open,
            high,
            low,
            close,
            volume,
            amount
        FROM read_parquet('{_sql_path(panel_file)}')
        WHERE {_asset_filter_sql(assets)}
        ORDER BY asset, date, datetime
        """
    ).fetchdf()


def _parse_batches(raw: str) -> tuple[int, ...]:
    aliases = {
        "1": (1,),
        "2": (2,),
        "3": (3,),
        "4": (4,),
        "1,2": (1, 2),
        "2,1": (1, 2),
        "3,4": (3, 4),
        "4,3": (3, 4),
        "1,2,3": (1, 2, 3),
        "1,2,3,4": (1, 2, 3, 4),
        "all": (1, 2, 3, 4),
    }
    key = ",".join(token.strip() for token in raw.lower().split(",") if token.strip())
    if key not in aliases:
        raise ValueError(
            "--batches must be one of: 1, 2, 3, 4, 1,2, 3,4, 1,2,3, 1,2,3,4, all"
        )
    return aliases[key]


def _feature_columns_for_batches(batches: tuple[int, ...]) -> list[str]:
    columns: list[str] = []
    if 1 in batches:
        columns += BATCH1_FEATURE_COLUMNS
    if 2 in batches:
        columns += BATCH2_FEATURE_COLUMNS
    if 3 in batches:
        columns += BATCH3_FEATURE_COLUMNS
    if 4 in batches:
        columns += BATCH4_FEATURE_COLUMNS
    return columns


def _compute_chunk_features(
    minute_chunk: pd.DataFrame,
    batches: tuple[int, ...],
    daily_meta: pd.DataFrame | None,
) -> pd.DataFrame:
    columns = _feature_columns_for_batches(batches)
    if minute_chunk.empty:
        return pd.DataFrame(columns=["date", "asset", *columns])

    parts: list[pd.DataFrame] = []
    if 1 in batches:
        parts.append(compute_batch1_feature_frame(minute_chunk))
    if 2 in batches:
        parts.append(compute_batch2_feature_frame(minute_chunk))
    if 3 in batches:
        parts.append(compute_batch3_feature_frame(minute_chunk))
    if 4 in batches:
        parts.append(compute_batch4_feature_frame(minute_chunk, daily_meta=daily_meta))

    out = parts[0]
    for part in parts[1:]:
        out = out.merge(part, on=["date", "asset"], how="outer", validate="one_to_one")
    return out[["date", "asset", *columns]]


def _load_daily_meta(
    con: duckdb.DuckDBPyConnection,
    prices_path: Path,
    assets: list[str],
    year: int,
) -> pd.DataFrame:
    """Load up_limit / down_limit / prev_close from prices.parquet for Batch 4."""

    return con.execute(
        f"""
        SELECT
            CAST(date AS VARCHAR) AS date,
            asset,
            CAST(up_limit AS DOUBLE) AS up_limit,
            CAST(down_limit AS DOUBLE) AS down_limit,
            CAST(raw_pre_close AS DOUBLE) AS prev_close
        FROM read_parquet('{_sql_path(prices_path)}')
        WHERE EXTRACT(year FROM CAST(date AS DATE)) = {int(year)}
          AND {_asset_filter_sql(assets)}
        """
    ).fetchdf()


def _feature_quality_summary(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    valid_mask = pd.Series(True, index=frame.index)
    for column in ["is_session_active", "is_actively_traded"]:
        if column in frame.columns:
            valid_mask &= frame[column].fillna(0).astype(int) == 1
    for column in ["is_likely_suspended", "is_panel_missing"]:
        if column in frame.columns:
            valid_mask &= frame[column].fillna(0).astype(int) == 0

    feature_columns = [column for column in BATCH1234_FEATURE_COLUMNS if column in frame.columns]
    valid = frame.loc[valid_mask, feature_columns]
    out: dict[str, dict[str, float]] = {}
    denominator = max(len(valid), 1)
    for column in feature_columns:
        values = pd.to_numeric(valid[column], errors="coerce")
        finite = np.isfinite(values)
        out[column] = {
            "nan_rate": float(values.isna().sum() / denominator),
            "inf_rate": float((~finite & values.notna()).sum() / denominator),
        }
    return out


def _frame_hash(frame: pd.DataFrame) -> str:
    normalized = frame.copy()
    for column in normalized.columns:
        if pd.api.types.is_datetime64_any_dtype(normalized[column]):
            normalized[column] = normalized[column].astype("datetime64[ns]").astype(str)
    row_hashes = pd.util.hash_pandas_object(normalized, index=False).to_numpy(dtype="uint64")
    return hashlib.sha256(row_hashes.tobytes()).hexdigest()


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build intraday feature expansion batches from Stage B minute panels.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--minute-panel-root", default=str(DEFAULT_PANEL_ROOT))
    parser.add_argument("--base-root", default=str(DEFAULT_BASE_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--prices-path",
        default=str(DEFAULT_PRICES_PATH),
        help="Path to prices.parquet supplying up_limit/down_limit/prev_close for Batch 4.",
    )
    parser.add_argument(
        "--batches",
        "--batch",
        default="1,2",
        help="Feature batches to compute: 1, 2, 3, 4, 1,2, 3,4, 1,2,3, 1,2,3,4, all.",
    )
    parser.add_argument("--years", default=None, help="Comma-separated years, e.g. 2024,2025.")
    parser.add_argument("--assets", default=None, help="Optional comma-separated asset subset.")
    parser.add_argument("--max-assets", type=int, default=0)
    parser.add_argument("--asset-chunk-size", type=int, default=32)
    parser.add_argument("--row-group-size", type=int, default=2_000_000)
    parser.add_argument("--compression", default="zstd")
    parser.add_argument("--duckdb-threads", type=int, default=0)
    parser.add_argument("--duckdb-memory-limit", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    panel_root = Path(args.minute_panel_root)
    base_root = Path(args.base_root)
    output_root = Path(args.output_root)
    prices_path = Path(args.prices_path)
    batches = _parse_batches(args.batches)
    feature_columns = _feature_columns_for_batches(batches)
    needs_meta = 4 in batches

    if base_root.resolve() == output_root.resolve():
        raise ValueError(
            "output-root must differ from base-root for this expansion script; "
            "write to a new root, then promote explicitly after validation."
        )

    years = _parse_years(args.years) or _discover_base_years(base_root)
    asset_subset = _parse_assets(args.assets)
    tmp_root = _prepare_output(output_root, years, args.overwrite)

    con = duckdb.connect()
    if args.duckdb_threads > 0:
        con.execute(f"PRAGMA threads={int(args.duckdb_threads)}")
    if args.duckdb_memory_limit:
        con.execute(f"PRAGMA memory_limit='{args.duckdb_memory_limit}'")

    summary: dict[str, object] = {
        "years": years,
        "minute_panel_root": str(panel_root),
        "base_root": str(base_root),
        "output_root": str(output_root),
        "batches": batches,
        "feature_columns": feature_columns,
        "year_summaries": [],
    }

    try:
        for year in years:
            base = _read_base_year(base_root, year)
            base["date"] = base["date"].astype(str)
            base["asset"] = base["asset"].astype(str)
            assets = sorted(base["asset"].dropna().unique().tolist())
            if asset_subset is not None:
                assets = [asset for asset in assets if asset in asset_subset]
                base = base[base["asset"].isin(assets)].copy()
            if args.max_assets and args.max_assets > 0:
                assets = assets[: args.max_assets]
                base = base[base["asset"].isin(assets)].copy()
            base = base.sort_values(["date", "asset"], kind="mergesort").reset_index(drop=True)
            base_columns = list(base.columns)
            base_hash_before = _frame_hash(base[base_columns])
            base_for_merge = base.drop(
                columns=[column for column in feature_columns if column in base.columns]
            )

            panel_file = panel_root / f"year={year}" / "part-0.parquet"
            if not panel_file.exists():
                raise FileNotFoundError(f"Minute panel partition not found: {panel_file}")

            year_meta: pd.DataFrame | None = None
            if needs_meta:
                if not prices_path.exists():
                    raise FileNotFoundError(
                        f"prices.parquet not found for Batch 4 daily meta: {prices_path}"
                    )
                year_meta = _load_daily_meta(con, prices_path, assets, year)

            feature_parts: list[pd.DataFrame] = []
            for chunk_index, asset_chunk in enumerate(
                _chunked(assets, int(args.asset_chunk_size)),
                start=1,
            ):
                minute_chunk = _load_minute_chunk(con, panel_file, asset_chunk)
                if needs_meta and year_meta is not None:
                    chunk_meta = year_meta[year_meta["asset"].isin(asset_chunk)]
                else:
                    chunk_meta = None
                features = _compute_chunk_features(minute_chunk, batches, chunk_meta)
                feature_parts.append(features)
                print(
                    f"year={year} chunk={chunk_index} assets={len(asset_chunk)} "
                    f"minute_rows={len(minute_chunk)} feature_rows={len(features)}",
                    flush=True,
                )

            feature_frame = (
                pd.concat(feature_parts, ignore_index=True)
                if feature_parts
                else pd.DataFrame(columns=["date", "asset", *feature_columns])
            )
            feature_frame["date"] = feature_frame["date"].astype(str)
            feature_frame["asset"] = feature_frame["asset"].astype(str)

            merged = base_for_merge.merge(
                feature_frame,
                on=["date", "asset"],
                how="left",
                validate="one_to_one",
            )
            merged = merged.sort_values(["date", "asset"], kind="mergesort").reset_index(drop=True)
            base_hash_after = _frame_hash(merged[base_columns])

            tmp_year_dir = tmp_root / f"year={year}"
            tmp_year_dir.mkdir(parents=True, exist_ok=True)
            output_file = tmp_year_dir / "part-0.parquet"
            merged.to_parquet(
                output_file,
                index=False,
                compression=args.compression,
                row_group_size=int(args.row_group_size),
            )

            year_summary = {
                "year": year,
                "rows": int(len(merged)),
                "assets": int(len(assets)),
                "feature_rows": int(len(feature_frame)),
                "base_columns_hash_before": base_hash_before,
                "base_columns_hash_after": base_hash_after,
                "base_columns_hash_match": base_hash_before == base_hash_after,
                "feature_quality": _feature_quality_summary(merged),
            }
            summary["year_summaries"].append(year_summary)

        (tmp_root / "_feature_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        _finalize_output(tmp_root, output_root)
    except Exception:
        shutil.rmtree(tmp_root, ignore_errors=True)
        raise

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
