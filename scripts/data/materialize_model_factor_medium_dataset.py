from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

PRICE_SOURCE_COLUMNS: tuple[str, ...] = (
    "date",
    "asset",
    "open",
    "high",
    "low",
    "close",
    "close_qfq",
    "pre_close",
    "vol",
    "amount",
)
PRICE_OUTPUT_RENAMES: dict[str, str] = {"vol": "volume"}


@dataclass(frozen=True)
class MaterializedTable:
    source: Path
    output: Path
    rows: int
    cutoff_date: str


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize a medium-size model-factor dataset by streaming the latest "
            "calendar window from full Parquet inputs."
        )
    )
    parser.add_argument(
        "--features",
        default=(
            "data/processed/stock_factor4model_2016_2026/"
            "baseline_inputs/features_safe_bfq_35.parquet"
        ),
        help="Full feature Parquet input.",
    )
    parser.add_argument(
        "--prices",
        default=(
            "data/processed/stock_factor4model_2016_2026/"
            "master_dataset"
        ),
        help="Full prices Parquet input or dataset directory.",
    )
    parser.add_argument(
        "--required-price-column",
        action="append",
        default=["close", "close_qfq"],
        help=(
            "Price column that must be present in the materialized price source. "
            "May be repeated; defaults require raw close and qfq close."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/stock_factor4model_2016_2026/medium_inputs",
        help="Directory for medium Parquet outputs.",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=365,
        help="Latest calendar days to keep.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=262_144,
        help="Arrow scanner batch size.",
    )
    args = parser.parse_args()

    if args.days <= 0:
        raise ValueError("--days must be > 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")

    features_path = Path(args.features).resolve()
    prices_path = Path(args.prices).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cutoff = _latest_shared_cutoff(
        features_path=features_path,
        prices_path=prices_path,
        days=int(args.days),
    )
    outputs = [
        _write_filtered_parquet(
            source=features_path,
            output=output_dir / "features_safe_bfq_35_recent1y.parquet",
            cutoff=cutoff,
            batch_size=int(args.batch_size),
        ),
        _write_filtered_parquet(
            source=prices_path,
            output=output_dir / "prices_raw_recent1y.parquet",
            cutoff=cutoff,
            batch_size=int(args.batch_size),
            required_columns=tuple(args.required_price_column),
            source_columns=PRICE_SOURCE_COLUMNS,
            rename_columns=PRICE_OUTPUT_RENAMES,
        ),
    ]

    print("")
    print("  Workflow : materialize-model-factor-medium-dataset")
    print("  Status   : success")
    print(f"  Cutoff   : {cutoff.date().isoformat()}")
    for item in outputs:
        print(f"  - {item.output}: {item.rows} rows")
    return 0


def _latest_shared_cutoff(*, features_path: Path, prices_path: Path, days: int) -> pd.Timestamp:
    latest_feature_date = _max_date(features_path)
    latest_price_date = _max_date(prices_path)
    latest_shared_date = min(latest_feature_date, latest_price_date)
    return pd.Timestamp(latest_shared_date - timedelta(days=days))


def _max_date(path: Path) -> pd.Timestamp:
    table = pq.read_table(path, columns=["date"])
    if table.num_rows == 0:
        raise ValueError(f"{path} contains no rows")
    values = table.column("date").to_pandas()
    latest = pd.to_datetime(values, errors="coerce").max()
    if pd.isna(latest):
        raise ValueError(f"{path} date column contains no valid timestamps")
    return pd.Timestamp(latest)


def _write_filtered_parquet(
    *,
    source: Path,
    output: Path,
    cutoff: pd.Timestamp,
    batch_size: int,
    required_columns: tuple[str, ...] = (),
    source_columns: tuple[str, ...] | None = None,
    rename_columns: dict[str, str] | None = None,
) -> MaterializedTable:
    dataset = ds.dataset(source, format="parquet")
    source_names = set(dataset.schema.names)
    missing_projection_columns = sorted(set(source_columns or ()) - source_names)
    if missing_projection_columns:
        raise ValueError(
            f"{source} is missing requested source columns: {missing_projection_columns}"
        )
    missing_columns = sorted(set(required_columns) - source_names)
    if missing_columns:
        raise ValueError(
            f"{source} is missing required price columns: {missing_columns}. "
            "Use a source that includes adjusted close columns before materializing."
        )
    cutoff_scalar = pa.scalar(cutoff.to_pydatetime(), type=pa.timestamp("us"))
    scanner = dataset.scanner(
        columns=list(source_columns) if source_columns is not None else None,
        filter=ds.field("date") >= cutoff_scalar,
        batch_size=batch_size,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()

    rows = 0
    writer: pq.ParquetWriter | None = None
    try:
        for batch in scanner.to_batches():
            if batch.num_rows == 0:
                continue
            batch = _rename_batch_columns(batch, rename_columns or {})
            if writer is None:
                writer = pq.ParquetWriter(output, batch.schema, compression="zstd")
            writer.write_batch(batch)
            rows += int(batch.num_rows)
    finally:
        if writer is not None:
            writer.close()

    if rows == 0:
        pq.write_table(
            pa.Table.from_batches(
                [],
                schema=_project_schema(
                    dataset.schema,
                    source_columns=source_columns,
                    rename_columns=rename_columns or {},
                ),
            ),
            output,
            compression="zstd",
        )
    return MaterializedTable(
        source=source,
        output=output,
        rows=rows,
        cutoff_date=cutoff.date().isoformat(),
    )


def _rename_batch_columns(batch: pa.RecordBatch, rename_columns: dict[str, str]) -> pa.RecordBatch:
    if not rename_columns:
        return batch
    names = [rename_columns.get(name, name) for name in batch.schema.names]
    return pa.RecordBatch.from_arrays(
        [batch.column(index) for index in range(batch.num_columns)],
        names=names,
    )


def _project_schema(
    schema: pa.Schema,
    *,
    source_columns: tuple[str, ...] | None,
    rename_columns: dict[str, str],
) -> pa.Schema:
    fields = [schema.field(name) for name in source_columns] if source_columns else list(schema)
    if not rename_columns:
        return pa.schema(fields)
    return pa.schema(
        [
            field.with_name(rename_columns.get(field.name, field.name))
            for field in fields
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
