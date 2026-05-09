"""Build the raw A-share minute panel from zipped or directory CSV inputs.

The script reads one asset CSV at a time, keeps prices unadjusted, removes
09:25 auction bars, and appends rows to per-year parquet writers under a
temporary `_in_progress/` run directory. Completed year partitions are moved
into place only after all writers close cleanly.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import shutil
import sys
import time
import zipfile
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CASE_DIR = (
    REPO_ROOT
    / "data"
    / "processed"
    / "real_case_inputs"
    / "ashare_institutional_20160418_20260415_supplemented"
)
DEFAULT_UNIVERSE_MASK = DEFAULT_CASE_DIR / "universe_mask.parquet"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "processed" / "minute_panel"

PANEL_SCHEMA = pa.schema(
    [
        pa.field("date", pa.string()),
        pa.field("asset", pa.string()),
        pa.field("datetime", pa.timestamp("ns")),
        pa.field("open", pa.float64()),
        pa.field("high", pa.float64()),
        pa.field("low", pa.float64()),
        pa.field("close", pa.float64()),
        pa.field("volume", pa.float64()),
        pa.field("amount", pa.float64()),
        pa.field("vwap", pa.float64()),
        pa.field("turnover_rate", pa.float64()),
        pa.field("float_shares", pa.float64()),
        pa.field("total_shares", pa.float64()),
    ]
)

REQUIRED_FIELDS = ("open", "high", "low", "close", "volume", "amount")
OPTIONAL_FIELDS = ("vwap", "turnover_rate", "float_shares", "total_shares")

COLUMN_ALIASES = {
    "datetime": ("datetime", "date_time", "trade_time", "日期时间", "交易时间"),
    "date": ("date", "trade_date", "日期", "交易日期"),
    "time": ("time", "minute", "bar_time", "时间", "成交时间"),
    "open": ("open", "开盘", "开盘价"),
    "high": ("high", "最高", "最高价"),
    "low": ("low", "最低", "最低价"),
    "close": ("close", "收盘", "收盘价", "现价"),
    "volume": ("volume", "vol", "成交量", "成交量(手)", "成交量(股)"),
    "amount": ("amount", "amt", "成交额", "成交额(元)", "成交金额", "成交金额(元)"),
    "vwap": ("vwap", "均价", "成交均价"),
    "turnover_rate": ("turnover_rate", "turnover", "换手率", "换手率(%)", "换手"),
    "float_shares": ("float_shares", "float_share", "流通股本", "流通股本(股)", "流通股"),
    "total_shares": ("total_shares", "total_share", "总股本", "总股本(股)", "总股"),
}


@dataclass(frozen=True)
class CsvPayload:
    asset: str
    source_path: str
    data: bytes


@dataclass
class ManifestRow:
    asset: str
    year: int
    rows: int
    first_datetime: str | None
    last_datetime: str | None
    n_unique_dates: int
    n_minutes_per_day_mode: int | None
    has_0925_bar: bool
    n_dup_datetime_raw: int
    n_deduped_rows: int
    n_dup_datetime: int
    n_invalid_ohlc_raw: int
    n_dropped_all_zero_ohlc: int
    n_zero_volume: int
    n_zero_amount: int
    n_invalid_ohlc: int
    source_csv_sha256: str
    source_path: str
    schema_version: str


def _clean_column_name(value: object) -> str:
    text = str(value).lstrip("\ufeff").strip()
    return re.sub(r"\s+", "", text)


def _alias_key(value: str) -> str:
    return re.sub(r"[\s_（）()]+", "", value.lstrip("\ufeff").strip().lower())


ALIAS_LOOKUP = {
    _alias_key(alias): field for field, aliases in COLUMN_ALIASES.items() for alias in aliases
}


def _canonical_asset(raw: str, code_to_asset: dict[str, str]) -> str | None:
    name = Path(raw).stem.upper()
    compact = re.sub(r"[^A-Z0-9]", "", name)

    prefixed = re.search(r"(SH|SZ|BJ)(\d{6})", compact)
    if prefixed:
        exchange, code = prefixed.group(1), prefixed.group(2)
        return code_to_asset.get(code, f"{code}.{exchange}")

    suffixed = re.search(r"(\d{6})(SH|SZ|BJ)", compact)
    if suffixed:
        code, exchange = suffixed.group(1), suffixed.group(2)
        return code_to_asset.get(code, f"{code}.{exchange}")

    dotted = re.fullmatch(r"(\d{6})\.(SH|SZ|BJ)", name)
    if dotted:
        code, exchange = dotted.group(1), dotted.group(2)
        return code_to_asset.get(code, f"{code}.{exchange}")

    digits = re.search(r"(\d{6})", compact)
    if digits:
        code = digits.group(1)
        return code_to_asset.get(code)
    return None


def _normalize_asset_value(value: object) -> str:
    text = str(value).strip().upper()
    dotted = re.fullmatch(r"(\d{6})\.(SH|SZ|BJ)", text)
    if dotted:
        return f"{dotted.group(1)}.{dotted.group(2)}"
    prefixed = re.fullmatch(r"(SH|SZ|BJ)(\d{6})", re.sub(r"[^A-Z0-9]", "", text))
    if prefixed:
        return f"{prefixed.group(2)}.{prefixed.group(1)}"
    raise ValueError(f"Unsupported asset code in whitelist: {value!r}")


def _load_asset_whitelist(path: Path) -> tuple[set[str], dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Asset whitelist not found: {path}")

    if path.suffix.lower() == ".parquet":
        assets = pd.read_parquet(path, columns=["asset"])["asset"]
    elif path.suffix.lower() in {".csv", ".txt"}:
        frame = pd.read_csv(path)
        if "asset" in frame.columns:
            assets = frame["asset"]
        else:
            assets = frame.iloc[:, 0]
    else:
        raise ValueError(f"Unsupported whitelist format: {path}")

    normalized = sorted({_normalize_asset_value(asset) for asset in assets.dropna().unique()})
    code_to_asset: dict[str, str] = {}
    for asset in normalized:
        code, _exchange = asset.split(".")
        if code in code_to_asset and code_to_asset[code] != asset:
            raise ValueError(f"Ambiguous six-digit code in whitelist: {code}")
        code_to_asset[code] = asset
    return set(normalized), code_to_asset


def _detect_encoding(data: bytes) -> str:
    sample = data[: 256 * 1024]
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            sample.decode(encoding)
            return encoding
        except UnicodeDecodeError:
            continue
    return "utf-8"


def _pyarrow_encoding(encoding: str) -> str:
    if encoding in {"utf-8", "utf-8-sig"}:
        return "utf8"
    return encoding


def _detect_delimiter(data: bytes, encoding: str) -> str:
    text = data[: 64 * 1024].decode(encoding, errors="ignore")
    try:
        return csv.Sniffer().sniff(text, delimiters=",\t;|").delimiter
    except csv.Error:
        first_lines = "\n".join(text.splitlines()[:10])
        counts = {delimiter: first_lines.count(delimiter) for delimiter in (",", "\t", ";", "|")}
        return max(counts, key=counts.get) if max(counts.values()) > 0 else ","


def _prefilter_csv_bytes(
    data: bytes,
    *,
    years: set[int] | None,
) -> bytes:
    """Fast path for year-limited smoke runs on sorted minute CSV files."""
    if years is None:
        return data
    allowed = {str(year).encode("ascii") for year in years}
    lines = data.splitlines(keepends=True)
    if len(lines) <= 1:
        return data
    kept = [lines[0]]
    kept.extend(line for line in lines[1:] if line[:4] in allowed)
    return b"".join(kept)


def _read_csv_payload(payload: CsvPayload, *, years: set[int] | None) -> pd.DataFrame:
    data = _prefilter_csv_bytes(payload.data, years=years)
    encoding = _detect_encoding(data)
    delimiter = _detect_delimiter(data, encoding)
    read_options = pacsv.ReadOptions(encoding=_pyarrow_encoding(encoding), block_size=1 << 24)
    parse_options = pacsv.ParseOptions(delimiter=delimiter)
    convert_options = pacsv.ConvertOptions(strings_can_be_null=True)
    try:
        table = pacsv.read_csv(
            pa.BufferReader(data),
            read_options=read_options,
            parse_options=parse_options,
            convert_options=convert_options,
        )
        frame = table.to_pandas()
    except Exception:  # noqa: BLE001 - fallback handles GBK/BOM edge cases
        frame = pd.read_csv(io.BytesIO(data), encoding=encoding, sep=delimiter)

    frame.columns = [_clean_column_name(column) for column in frame.columns]
    return frame


def _resolve_columns(frame: pd.DataFrame) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for column in frame.columns:
        key = _alias_key(column)
        if key in ALIAS_LOOKUP and ALIAS_LOOKUP[key] not in resolved:
            resolved[ALIAS_LOOKUP[key]] = column

    missing_required = [field for field in REQUIRED_FIELDS if field not in resolved]
    has_datetime = "datetime" in resolved or "date" in resolved
    if not has_datetime:
        missing_required.append("datetime or date+time")
    if missing_required:
        raise ValueError(f"CSV missing required columns: {', '.join(missing_required)}")
    return resolved


def _format_time_series(values: pd.Series) -> pd.Series:
    text = values.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    has_colon = text.str.contains(":", na=False)
    out = text.copy()

    digits = text[~has_colon].str.replace(r"\D", "", regex=True)
    short = digits.str.len() <= 4
    digits.loc[short] = digits.loc[short].str.zfill(4)
    digits.loc[~short] = digits.loc[~short].str.zfill(6)

    formatted = pd.Series(pd.NA, index=digits.index, dtype="string")
    hhmm = digits.str.len() <= 4
    formatted.loc[hhmm] = (
        digits.loc[hhmm].str.slice(0, 2) + ":" + digits.loc[hhmm].str.slice(2, 4) + ":00"
    )
    formatted.loc[~hhmm] = (
        digits.loc[~hhmm].str.slice(0, 2)
        + ":"
        + digits.loc[~hhmm].str.slice(2, 4)
        + ":"
        + digits.loc[~hhmm].str.slice(4, 6)
    )
    out.loc[~has_colon] = formatted
    return out


def _coerce_datetime(frame: pd.DataFrame, columns: dict[str, str]) -> pd.Series:
    if "datetime" in columns:
        return pd.to_datetime(frame[columns["datetime"]], errors="coerce")
    if "time" not in columns:
        return pd.to_datetime(frame[columns["date"]], errors="coerce")

    date_part = pd.to_datetime(frame[columns["date"]], errors="coerce").dt.strftime("%Y-%m-%d")
    time_part = _format_time_series(frame[columns["time"]])
    return pd.to_datetime(date_part.astype("string") + " " + time_part, errors="coerce")


def _numeric_column(frame: pd.DataFrame, columns: dict[str, str], field: str) -> pd.Series:
    if field not in columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[columns[field]], errors="coerce").astype("float64")


def _transform_payload(
    payload: CsvPayload,
    *,
    date_from: str,
    date_to: str,
    years: set[int] | None,
) -> tuple[pd.DataFrame, set[int], dict[int, tuple[int, int, int]], dict[int, tuple[int, int]]]:
    raw = _read_csv_payload(payload, years=years)
    columns = _resolve_columns(raw)
    dt = _coerce_datetime(raw, columns)

    frame = pd.DataFrame(
        {
            "date": dt.dt.strftime("%Y-%m-%d"),
            "asset": payload.asset,
            "datetime": dt,
            "open": _numeric_column(raw, columns, "open"),
            "high": _numeric_column(raw, columns, "high"),
            "low": _numeric_column(raw, columns, "low"),
            "close": _numeric_column(raw, columns, "close"),
            "volume": _numeric_column(raw, columns, "volume"),
            "amount": _numeric_column(raw, columns, "amount"),
            "vwap": _numeric_column(raw, columns, "vwap"),
            "turnover_rate": _numeric_column(raw, columns, "turnover_rate"),
            "float_shares": _numeric_column(raw, columns, "float_shares"),
            "total_shares": _numeric_column(raw, columns, "total_shares"),
        }
    )

    frame = frame[frame["datetime"].notna()]
    frame = frame[(frame["date"] >= date_from) & (frame["date"] <= date_to)]
    if years is not None:
        frame = frame[frame["datetime"].dt.year.isin(years)]

    minute_hhmm = frame["datetime"].dt.strftime("%H:%M")
    has_0925_years = set(frame.loc[minute_hhmm == "09:25", "datetime"].dt.year.astype(int))
    frame = frame[minute_hhmm != "09:25"].copy()

    year_values = frame["datetime"].dt.year.astype(int)
    raw_dup_by_year = frame.groupby(year_values)["datetime"].apply(
        lambda item: item.duplicated().sum()
    )
    exact_dup_mask = frame.duplicated(subset=list(PANEL_SCHEMA.names), keep="first")
    deduped_by_year = exact_dup_mask.groupby(year_values).sum()
    if bool(exact_dup_mask.any()):
        frame = frame[~exact_dup_mask].copy()
        year_values = frame["datetime"].dt.year.astype(int)
    post_dup_by_year = frame.groupby(year_values)["datetime"].apply(
        lambda item: item.duplicated().sum()
    )
    duplicate_stats: dict[int, tuple[int, int, int]] = {}
    all_years = (
        set(raw_dup_by_year.index) | set(deduped_by_year.index) | set(post_dup_by_year.index)
    )
    for year in all_years:
        duplicate_stats[int(year)] = (
            int(raw_dup_by_year.get(year, 0)),
            int(deduped_by_year.get(year, 0)),
            int(post_dup_by_year.get(year, 0)),
        )

    year_values = frame["datetime"].dt.year.astype(int)
    raw_invalid_mask = _invalid_ohlc_mask(frame)
    all_zero_ohlc_mask = (
        (frame[["open", "high", "low", "close"]].fillna(0.0) == 0.0).all(axis=1)
        & (frame["volume"].fillna(0.0) == 0.0)
        & (frame["amount"].fillna(0.0) == 0.0)
    )
    raw_invalid_by_year = raw_invalid_mask.groupby(year_values).sum()
    dropped_zero_by_year = all_zero_ohlc_mask.groupby(year_values).sum()
    if bool(all_zero_ohlc_mask.any()):
        frame = frame[~all_zero_ohlc_mask].copy()
    invalid_drop_stats: dict[int, tuple[int, int]] = {}
    all_invalid_years = set(raw_invalid_by_year.index) | set(dropped_zero_by_year.index)
    for year in all_invalid_years:
        invalid_drop_stats[int(year)] = (
            int(raw_invalid_by_year.get(year, 0)),
            int(dropped_zero_by_year.get(year, 0)),
        )

    frame = frame.sort_values(["asset", "datetime"], kind="mergesort")
    return frame, has_0925_years, duplicate_stats, invalid_drop_stats


def _invalid_ohlc_mask(frame: pd.DataFrame) -> pd.Series:
    max_open_close = frame[["open", "close"]].max(axis=1)
    min_open_close = frame[["open", "close"]].min(axis=1)
    invalid_ohlc = (
        (frame[["open", "high", "low", "close"]] <= 0.0).any(axis=1)
        | (frame["high"] < max_open_close)
        | (frame["low"] > min_open_close)
        | (frame["high"] < frame["low"])
    )
    return invalid_ohlc.fillna(True)


def _quality_counts(frame: pd.DataFrame) -> tuple[int, int, int, int]:
    n_dup_datetime = int(frame["datetime"].duplicated().sum())
    n_zero_volume = int((frame["volume"].fillna(0.0) == 0.0).sum())
    n_zero_amount = int((frame["amount"].fillna(0.0) == 0.0).sum())
    n_invalid_ohlc = int(_invalid_ohlc_mask(frame).sum())
    return n_dup_datetime, n_zero_volume, n_zero_amount, n_invalid_ohlc


def _mode_minutes_per_day(frame: pd.DataFrame) -> int | None:
    if frame.empty:
        return None
    counts = frame.groupby("date", observed=True).size()
    if counts.empty:
        return None
    modes = counts.mode()
    return int(modes.iloc[0]) if not modes.empty else None


def _iter_csv_payloads(
    sources: list[Path],
    whitelist: set[str],
    code_to_asset: dict[str, str],
) -> Iterator[CsvPayload]:
    for source in sources:
        if not source.exists():
            raise FileNotFoundError(f"Minute source not found: {source}")
        if source.is_dir():
            for csv_path in sorted(source.rglob("*.csv")):
                asset = _canonical_asset(str(csv_path), code_to_asset)
                if asset in whitelist:
                    yield CsvPayload(
                        asset=asset,
                        source_path=str(csv_path),
                        data=csv_path.read_bytes(),
                    )
        elif source.suffix.lower() == ".zip":
            with zipfile.ZipFile(source) as archive:
                for info in sorted(archive.infolist(), key=lambda item: item.filename):
                    if info.is_dir() or not info.filename.lower().endswith(".csv"):
                        continue
                    asset = _canonical_asset(info.filename, code_to_asset)
                    if asset not in whitelist:
                        continue
                    with archive.open(info) as handle:
                        data = _read_all(handle)
                    yield (
                        CsvPayload(
                            asset=asset,
                            source_path=f"{source}::{info.filename}",
                            data=data,
                        )
                    )
        elif source.suffix.lower() == ".csv":
            asset = _canonical_asset(str(source), code_to_asset)
            if asset in whitelist:
                yield CsvPayload(asset=asset, source_path=str(source), data=source.read_bytes())
        else:
            raise ValueError(f"Unsupported minute source: {source}")


def _read_all(handle: BinaryIO) -> bytes:
    chunks: list[bytes] = []
    while True:
        chunk = handle.read(16 * 1024 * 1024)
        if not chunk:
            break
        chunks.append(chunk)
    return b"".join(chunks)


def _prepare_output(output_root: Path, years: set[int] | None, overwrite: bool) -> Path:
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

    run_id = time.strftime("%Y%m%d_%H%M%S")
    tmp_root = output_root / "_in_progress" / run_id
    tmp_root.mkdir(parents=True, exist_ok=False)
    return tmp_root


def _finalize_output(tmp_root: Path, output_root: Path) -> None:
    for tmp_year_dir in sorted(tmp_root.glob("year=*")):
        final_year_dir = output_root / tmp_year_dir.name
        if final_year_dir.exists():
            raise FileExistsError(f"Final partition exists before finalize: {final_year_dir}")
        os.replace(tmp_year_dir, final_year_dir)

    for name in ("_manifest.parquet", "_manifest_summary.json"):
        tmp_file = tmp_root / name
        if tmp_file.exists():
            os.replace(tmp_file, output_root / name)
    try:
        tmp_root.rmdir()
        tmp_root.parent.rmdir()
    except OSError:
        pass


def _parse_years(raw: str | None) -> set[int] | None:
    if not raw:
        return None
    return {int(token.strip()) for token in raw.split(",") if token.strip()}


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert raw minute CSV/ZIP files into per-year parquet minute panels.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--minute-source",
        action="append",
        required=True,
        help="Raw minute source path. May be repeated. Supports .zip, .csv, or a directory.",
    )
    parser.add_argument("--asset-whitelist", default=str(DEFAULT_UNIVERSE_MASK))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--date-from", default="2016-04-18")
    parser.add_argument("--date-to", default="2026-04-15")
    parser.add_argument(
        "--years",
        default=None,
        help="Comma-separated year filter, e.g. 2024,2025.",
    )
    parser.add_argument("--schema-version", default="minute_panel_v1")
    parser.add_argument("--row-group-size", type=int, default=2_000_000)
    parser.add_argument("--compression", default="zstd")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--allow-duplicate-source",
        action="store_true",
        help="Allow multiple CSV payloads for the same asset. Default is to fail fast.",
    )
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    years = _parse_years(args.years)
    output_root = Path(args.output_root)
    tmp_root = _prepare_output(output_root, years, args.overwrite)

    whitelist, code_to_asset = _load_asset_whitelist(Path(args.asset_whitelist))
    sources = [Path(path) for path in args.minute_source]
    seen_assets: set[str] = set()

    writers: dict[int, pq.ParquetWriter] = {}
    manifest: list[ManifestRow] = []
    processed_assets: set[str] = set()
    total_rows = 0
    payload_count = 0

    try:
        for index, payload in enumerate(
            _iter_csv_payloads(sources, whitelist, code_to_asset),
            start=1,
        ):
            payload_count = index
            if payload.asset in seen_assets and not args.allow_duplicate_source:
                raise ValueError(f"Duplicate CSV source for asset: {payload.asset}")
            seen_assets.add(payload.asset)
            processed_assets.add(payload.asset)
            source_hash = hashlib.sha256(payload.data).hexdigest()
            frame, has_0925_years, duplicate_stats, invalid_drop_stats = _transform_payload(
                payload,
                date_from=args.date_from,
                date_to=args.date_to,
                years=years,
            )
            if frame.empty:
                continue

            for year, year_frame in frame.groupby(frame["datetime"].dt.year, sort=True):
                year = int(year)
                tmp_year_dir = tmp_root / f"year={year}"
                tmp_year_dir.mkdir(parents=True, exist_ok=True)
                writer = writers.get(year)
                if writer is None:
                    writer = pq.ParquetWriter(
                        tmp_year_dir / "part-0.parquet",
                        PANEL_SCHEMA,
                        compression=args.compression,
                        use_dictionary=["date", "asset"],
                    )
                    writers[year] = writer

                year_frame = year_frame[list(PANEL_SCHEMA.names)]
                table = pa.Table.from_pandas(year_frame, schema=PANEL_SCHEMA, preserve_index=False)
                writer.write_table(table, row_group_size=args.row_group_size)

                rows = len(year_frame)
                total_rows += rows
                n_dup, n_zero_vol, n_zero_amt, n_invalid = _quality_counts(year_frame)
                n_dup_raw, n_deduped, n_dup_post = duplicate_stats.get(year, (0, 0, n_dup))
                n_invalid_raw, n_dropped_zero = invalid_drop_stats.get(year, (n_invalid, 0))
                manifest.append(
                    ManifestRow(
                        asset=payload.asset,
                        year=year,
                        rows=rows,
                        first_datetime=str(year_frame["datetime"].min()),
                        last_datetime=str(year_frame["datetime"].max()),
                        n_unique_dates=int(year_frame["date"].nunique()),
                        n_minutes_per_day_mode=_mode_minutes_per_day(year_frame),
                        has_0925_bar=year in has_0925_years,
                        n_dup_datetime_raw=n_dup_raw,
                        n_deduped_rows=n_deduped,
                        n_dup_datetime=n_dup_post,
                        n_invalid_ohlc_raw=n_invalid_raw,
                        n_dropped_all_zero_ohlc=n_dropped_zero,
                        n_zero_volume=n_zero_vol,
                        n_zero_amount=n_zero_amt,
                        n_invalid_ohlc=n_invalid,
                        source_csv_sha256=source_hash,
                        source_path=payload.source_path,
                        schema_version=args.schema_version,
                    )
                )

            if index % args.progress_every == 0:
                print(
                    f"processed_csv={index} assets={len(processed_assets)} rows={total_rows}",
                    flush=True,
                )
    finally:
        for writer in writers.values():
            writer.close()

    manifest_frame = pd.DataFrame([asdict(row) for row in manifest])
    manifest_frame.to_parquet(tmp_root / "_manifest.parquet", index=False)

    missing_assets = sorted(whitelist - processed_assets)
    summary = {
        "schema_version": args.schema_version,
        "date_from": args.date_from,
        "date_to": args.date_to,
        "years": sorted(years) if years is not None else None,
        "asset_whitelist_count": len(whitelist),
        "processed_asset_count": len(processed_assets),
        "missing_asset_count": len(missing_assets),
        "missing_assets": missing_assets,
        "payload_count": payload_count,
        "total_rows": int(total_rows),
        "output_root": str(output_root),
    }
    (tmp_root / "_manifest_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    _finalize_output(tmp_root, output_root)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
