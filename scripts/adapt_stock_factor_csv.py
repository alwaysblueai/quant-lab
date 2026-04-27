#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

SAFE_BASELINE_FEATURE_COLUMNS: tuple[str, ...] = (
    "turnover_rate",
    "turnover_rate_f",
    "volume_ratio",
    "pe_ttm",
    "pb",
    "ps_ttm",
    "dv_ttm",
    "total_mv",
    "circ_mv",
    "free_share",
    "atr_bfq",
    "bias1_bfq",
    "bias2_bfq",
    "bias3_bfq",
    "cci_bfq",
    "macd_bfq",
    "macd_dea_bfq",
    "macd_dif_bfq",
    "mtm_bfq",
    "obv_bfq",
    "rsi_bfq_6",
    "rsi_bfq_12",
    "rsi_bfq_24",
    "vr_bfq",
    "wr_bfq",
)

RAW_PRICE_OUTPUT_COLUMNS: tuple[str, ...] = (
    "date",
    "asset",
    "open",
    "high",
    "low",
    "close",
    "pre_close",
    "volume",
    "amount",
)

VENDOR_QFQ_PRICE_OUTPUT_COLUMNS: tuple[str, ...] = (
    "date",
    "asset",
    "open",
    "high",
    "low",
    "close",
    "pre_close",
    "volume",
    "amount",
)

SOURCE_REQUIRED_COLUMNS: tuple[str, ...] = (
    "ts_code",
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "pre_close",
    "vol",
    "amount",
    "open_qfq",
    "high_qfq",
    "low_qfq",
    "close_qfq",
)


@dataclass(frozen=True)
class YearSummary:
    source_file: str
    rows: int
    unique_assets: int
    min_date: str
    max_date: str
    output_master_parquet: str


def _print(message: str) -> None:
    print(message, flush=True)


def _discover_source_files(source_dir: Path) -> list[Path]:
    files = sorted(source_dir.glob("stock_factor_*.csv"))
    if not files:
        raise FileNotFoundError(f"no stock_factor_*.csv files found under {source_dir}")
    return files


def _read_header(path: Path) -> list[str]:
    return list(pd.read_csv(path, nrows=0).columns)


def _validate_source_schema(files: list[Path]) -> list[str]:
    base_header = _read_header(files[0])
    missing = [column for column in SOURCE_REQUIRED_COLUMNS if column not in base_header]
    if missing:
        raise ValueError(
            f"source file {files[0]} is missing required columns: {sorted(missing)}"
        )
    for path in files[1:]:
        header = _read_header(path)
        if header != base_header:
            raise ValueError(f"schema mismatch detected for {path}")
    return base_header


def _dtype_map(columns: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for column in columns:
        if column == "ts_code":
            out[column] = "string"
        elif column == "trade_date":
            out[column] = "string"
        else:
            out[column] = "float64"
    return out


def _normalize_master_chunk(chunk: pd.DataFrame, *, source_year: int) -> pd.DataFrame:
    out = chunk.rename(columns={"ts_code": "asset", "trade_date": "date"}).copy()
    out["asset"] = out["asset"].astype("string").astype(str)
    out["date"] = pd.to_datetime(out["date"], format="%Y%m%d", errors="coerce")
    if out["date"].isna().any():
        raise ValueError("source chunk contains invalid trade_date values")
    out["known_at"] = out["date"]
    out["source_year"] = source_year
    return out


def _prepare_raw_prices(chunk: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(chunk["trade_date"], format="%Y%m%d", errors="coerce"),
            "asset": chunk["ts_code"].astype("string").astype(str),
            "open": chunk["open"],
            "high": chunk["high"],
            "low": chunk["low"],
            "close": chunk["close"],
            "pre_close": chunk["pre_close"],
            "volume": chunk["vol"],
            "amount": chunk["amount"],
        }
    )
    if out["date"].isna().any():
        raise ValueError("raw prices chunk contains invalid trade_date values")
    return out.loc[:, list(RAW_PRICE_OUTPUT_COLUMNS)]


def _prepare_vendor_qfq_prices(chunk: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(chunk["trade_date"], format="%Y%m%d", errors="coerce"),
            "asset": chunk["ts_code"].astype("string").astype(str),
            "open": chunk["open_qfq"],
            "high": chunk["high_qfq"],
            "low": chunk["low_qfq"],
            "close": chunk["close_qfq"],
            "pre_close": chunk["pre_close"],
            "volume": chunk["vol"],
            "amount": chunk["amount"],
        }
    )
    if out["date"].isna().any():
        raise ValueError("vendor qfq prices chunk contains invalid trade_date values")
    return out.loc[:, list(VENDOR_QFQ_PRICE_OUTPUT_COLUMNS)]


def _prepare_safe_features(
    chunk: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...],
) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(chunk["trade_date"], format="%Y%m%d", errors="coerce"),
            "asset": chunk["ts_code"].astype("string").astype(str),
            "known_at": pd.to_datetime(chunk["trade_date"], format="%Y%m%d", errors="coerce"),
        }
    )
    if out["date"].isna().any():
        raise ValueError("feature chunk contains invalid trade_date values")
    for column in feature_columns:
        out[column] = chunk[column]
    return out


def _write_parquet_chunk(
    frame: pd.DataFrame,
    *,
    writer: pq.ParquetWriter | None,
    path: Path,
) -> pq.ParquetWriter:
    table = pa.Table.from_pandas(frame, preserve_index=False)
    if writer is None:
        path.parent.mkdir(parents=True, exist_ok=True)
        writer = pq.ParquetWriter(path, table.schema, compression="zstd")
    writer.write_table(table)
    return writer


def _append_csv_chunk(frame: pd.DataFrame, *, path: Path, header: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    csv_frame = frame.copy()
    for column in ("date", "known_at"):
        if column in csv_frame.columns:
            csv_frame[column] = pd.to_datetime(csv_frame[column], errors="coerce").dt.strftime(
                "%Y-%m-%d"
            )
    csv_frame.to_csv(
        path,
        mode="w" if header else "a",
        index=False,
        header=header,
        float_format="%.10g",
    )


def _relative(path: Path, *, base_dir: Path) -> str:
    return str(path.resolve().relative_to(base_dir.resolve()))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize yearly vendor stock factor CSV files into a parquet master layer "
            "plus conservative baseline inputs for alpha-lab."
        )
    )
    parser.add_argument(
        "--source-dir",
        default=(
            "/mnt/c/Users/yukun zhao/OneDrive/Desktop/Quant/stock_factor4model/2016-2016"
        ),
        help="Directory containing stock_factor_YYYY.csv source files.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/stock_factor4model_2016_2026",
        help="Output directory for processed artifacts.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50000,
        help="CSV chunk size for streaming conversion.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()

    source_dir = Path(args.source_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    master_dir = output_dir / "master_dataset"
    baseline_dir = output_dir / "baseline_inputs"
    metadata_dir = output_dir / "metadata"

    source_files = _discover_source_files(source_dir)
    schema = _validate_source_schema(source_files)
    dtype_map = _dtype_map(schema)

    existing_feature_columns = tuple(
        column for column in SAFE_BASELINE_FEATURE_COLUMNS if column in schema
    )
    if not existing_feature_columns:
        raise ValueError("none of the configured safe baseline feature columns exist in source")

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    if (source_dir / "字段说明.txt").exists():
        shutil.copy2(source_dir / "字段说明.txt", metadata_dir / "字段说明.txt")

    baseline_features_csv = baseline_dir / "features_safe.csv"
    baseline_features_parquet = baseline_dir / "features_safe.parquet"
    baseline_prices_raw_parquet = baseline_dir / "prices_raw.parquet"
    baseline_prices_vendor_qfq_parquet = baseline_dir / "prices_vendor_qfq_unaudited.parquet"

    if baseline_features_csv.exists():
        baseline_features_csv.unlink()

    master_year_summaries: list[YearSummary] = []
    overall_assets: set[str] = set()
    overall_rows = 0
    overall_min_date: pd.Timestamp | None = None
    overall_max_date: pd.Timestamp | None = None

    features_csv_header_written = False
    features_parquet_writer: pq.ParquetWriter | None = None
    prices_raw_writer: pq.ParquetWriter | None = None
    prices_vendor_qfq_writer: pq.ParquetWriter | None = None

    try:
        for path in source_files:
            year_text = path.stem.rsplit("_", 1)[-1]
            source_year = int(year_text)
            year_master_path = master_dir / f"stock_factor_{source_year}.parquet"
            if year_master_path.exists():
                year_master_path.unlink()

            _print(f"[year {source_year}] reading {path.name}")
            year_writer: pq.ParquetWriter | None = None
            year_rows = 0
            year_assets: set[str] = set()
            year_min_date: pd.Timestamp | None = None
            year_max_date: pd.Timestamp | None = None

            try:
                for chunk_idx, chunk in enumerate(
                    pd.read_csv(
                        path,
                        dtype=dtype_map,
                        chunksize=args.chunk_size,
                        low_memory=False,
                    ),
                    start=1,
                ):
                    year_rows += len(chunk)
                    overall_rows += len(chunk)

                    dates = pd.to_datetime(chunk["trade_date"], format="%Y%m%d", errors="coerce")
                    chunk_min = pd.Timestamp(dates.min())
                    chunk_max = pd.Timestamp(dates.max())
                    year_min_date = (
                        chunk_min if year_min_date is None else min(year_min_date, chunk_min)
                    )
                    year_max_date = (
                        chunk_max if year_max_date is None else max(year_max_date, chunk_max)
                    )
                    overall_min_date = (
                        chunk_min if overall_min_date is None else min(overall_min_date, chunk_min)
                    )
                    overall_max_date = (
                        chunk_max if overall_max_date is None else max(overall_max_date, chunk_max)
                    )

                    assets = set(chunk["ts_code"].dropna().astype(str).tolist())
                    year_assets.update(assets)
                    overall_assets.update(assets)

                    master_chunk = _normalize_master_chunk(chunk, source_year=source_year)
                    year_writer = _write_parquet_chunk(
                        master_chunk,
                        writer=year_writer,
                        path=year_master_path,
                    )

                    raw_prices_chunk = _prepare_raw_prices(chunk)
                    prices_raw_writer = _write_parquet_chunk(
                        raw_prices_chunk,
                        writer=prices_raw_writer,
                        path=baseline_prices_raw_parquet,
                    )

                    vendor_qfq_chunk = _prepare_vendor_qfq_prices(chunk)
                    prices_vendor_qfq_writer = _write_parquet_chunk(
                        vendor_qfq_chunk,
                        writer=prices_vendor_qfq_writer,
                        path=baseline_prices_vendor_qfq_parquet,
                    )

                    features_chunk = _prepare_safe_features(
                        chunk,
                        feature_columns=existing_feature_columns,
                    )
                    _append_csv_chunk(
                        features_chunk,
                        path=baseline_features_csv,
                        header=not features_csv_header_written,
                    )
                    features_csv_header_written = True
                    features_parquet_writer = _write_parquet_chunk(
                        features_chunk,
                        writer=features_parquet_writer,
                        path=baseline_features_parquet,
                    )

                    if chunk_idx % 10 == 0:
                        _print(
                            f"[year {source_year}] chunks={chunk_idx} rows={year_rows:,}"
                        )
            finally:
                if year_writer is not None:
                    year_writer.close()

            master_year_summaries.append(
                YearSummary(
                    source_file=path.name,
                    rows=year_rows,
                    unique_assets=len(year_assets),
                    min_date=year_min_date.strftime("%Y-%m-%d"),
                    max_date=year_max_date.strftime("%Y-%m-%d"),
                    output_master_parquet=_relative(year_master_path, base_dir=output_dir),
                )
            )
            _print(
                f"[year {source_year}] done rows={year_rows:,} assets={len(year_assets):,} "
                f"range={year_min_date.strftime('%Y-%m-%d')}..{year_max_date.strftime('%Y-%m-%d')}"
            )
    finally:
        for writer in (
            features_parquet_writer,
            prices_raw_writer,
            prices_vendor_qfq_writer,
        ):
            if writer is not None:
                writer.close()

    feature_columns_path = metadata_dir / "safe_feature_columns.txt"
    feature_columns_path.write_text(
        "\n".join(existing_feature_columns) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "source_dir": str(source_dir),
        "source_files": [path.name for path in source_files],
        "source_schema_column_count": len(schema),
        "source_schema_columns": schema,
        "processing_notes": {
            "raw_source_files_preserved": True,
            "master_dataset_format": "parquet_per_year",
            "baseline_feature_policy": "no_suffix + selected_bfq_only",
            "vendor_qfq_status": "preserved_for_audit_but_not_used_in_safe_features",
            "vendor_hfq_status": "excluded_from_baseline_outputs",
        },
        "baseline_feature_columns": list(existing_feature_columns),
        "outputs": {
            "master_dataset_dir": _relative(master_dir, base_dir=output_dir),
            "baseline_features_csv": _relative(baseline_features_csv, base_dir=output_dir),
            "baseline_features_parquet": _relative(
                baseline_features_parquet,
                base_dir=output_dir,
            ),
            "baseline_prices_raw_parquet": _relative(
                baseline_prices_raw_parquet,
                base_dir=output_dir,
            ),
            "baseline_prices_vendor_qfq_unaudited_parquet": _relative(
                baseline_prices_vendor_qfq_parquet,
                base_dir=output_dir,
            ),
            "safe_feature_columns": _relative(feature_columns_path, base_dir=output_dir),
            "field_description_copy": (
                _relative(metadata_dir / "字段说明.txt", base_dir=output_dir)
                if (metadata_dir / "字段说明.txt").exists()
                else None
            ),
        },
        "overall_summary": {
            "rows": overall_rows,
            "unique_assets": len(overall_assets),
            "min_date": overall_min_date.strftime("%Y-%m-%d"),
            "max_date": overall_max_date.strftime("%Y-%m-%d"),
        },
        "per_year_summary": [asdict(item) for item in master_year_summaries],
    }
    manifest_path = output_dir / "processing_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    _print("")
    _print(f"processed output: {output_dir}")
    _print(f"manifest        : {manifest_path}")
    _print(f"master dataset  : {master_dir}")
    _print(f"safe features   : {baseline_features_csv}")
    _print(f"raw prices      : {baseline_prices_raw_parquet}")
    _print(f"vendor qfq      : {baseline_prices_vendor_qfq_parquet}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
