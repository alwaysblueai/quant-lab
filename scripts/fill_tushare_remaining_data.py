#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import pandas as pd

from alpha_lab.data_adapters import tushare_adapter as tushare
from alpha_lab.data_store.catalog import DataCatalog
from alpha_lab.data_store.tushare import (
    TushareIngestor,
    _canonicalize_daily_basic,
    _canonicalize_index_membership,
    _canonicalize_industry_classification,
    _canonicalize_industry_membership,
    _canonicalize_moneyflow,
)


def _print(message: str) -> None:
    print(message, flush=True)


def _month_end(ts: pd.Timestamp) -> pd.Timestamp:
    return (ts + pd.offsets.MonthEnd(0)).normalize()


def _iter_month_chunks(start_date: str, end_date: str, *, months: int) -> list[tuple[str, str]]:
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    chunks: list[tuple[str, str]] = []
    current = start
    while current <= end:
        chunk_end = min(_month_end(current + pd.DateOffset(months=months - 1)), end)
        chunks.append((current.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")))
        current = chunk_end + pd.Timedelta(days=1)
    return chunks


def _load_state(path: Path) -> dict[str, object]:
    if not path.exists():
        return {
            "financial_batches_done": [],
            "daily_basic_chunks_done": [],
            "moneyflow_chunks_done": [],
            "index_chunks_done": [],
            "industry_done": False,
            "liquidity_chunks_done": [],
        }
    return json.loads(path.read_text(encoding="utf-8"))


def _save_state(path: Path, state: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )


def _dataset_note_version(
    catalog: DataCatalog,
    *,
    table_names: tuple[str, ...],
    notes: dict[str, object],
) -> None:
    version = catalog.write_dataset_version(
        dataset_name=DataCatalog.CORE_DATASET_NAME,
        table_names=table_names,
        raw_snapshot_id=None,
        notes=notes,
    )
    _print(f"[version] {version.version_id} tables={','.join(table_names)}")


def _load_all_assets(catalog: DataCatalog) -> list[str]:
    instruments = catalog.load_table("instruments")
    if instruments.empty:
        raise RuntimeError("instruments is empty; cannot resolve asset universe.")
    return sorted(instruments["asset"].astype(str).dropna().unique().tolist())


def _run_financial_batches(
    *,
    catalog: DataCatalog,
    token: str,
    financial_start_date: str,
    end_date: str,
    batch_size: int,
    state: dict[str, object],
) -> None:
    assets = _load_all_assets(catalog)
    done = {int(item) for item in state.get("financial_batches_done", [])}
    ingestor = TushareIngestor(catalog)
    total_batches = (len(assets) + batch_size - 1) // batch_size
    for batch_idx in range(total_batches):
        if batch_idx in done:
            continue
        start = batch_idx * batch_size
        stop = min(start + batch_size, len(assets))
        batch_assets = tuple(assets[start:stop])
        _print(
            f"[financial] batch {batch_idx + 1}/{total_batches} "
            f"assets={len(batch_assets)} range={batch_assets[0]}..{batch_assets[-1]}"
        )
        result = ingestor.ingest_core_chunked(
            start_date=financial_start_date,
            end_date=end_date,
            token=token,
            assets=batch_assets,
            chunk_months=0,
            mode="fundamental",
            include_reference_data=False,
            progress_callback=lambda msg, batch=batch_idx + 1, total=total_batches: _print(
                f"[financial {batch}/{total}] {msg}"
            ),
        )
        _print(
            f"[financial] snapshot={result.snapshot_manifest.snapshot_id} "
            f"version={result.dataset_version.version_id}"
        )
        _print(f"[financial] rows={result.row_counts}")
        done.add(batch_idx)
        state["financial_batches_done"] = sorted(done)
        _save_state(_state_path(catalog), state)


def _run_daily_sidecar_chunks(
    *,
    catalog: DataCatalog,
    token: str,
    start_date: str,
    end_date: str,
    chunk_months: int,
    state: dict[str, object],
) -> None:
    assets = tuple(_load_all_assets(catalog))
    pro = tushare._build_tushare_client(token)
    done_daily_basic = set(state.get("daily_basic_chunks_done", []))
    done_moneyflow = set(state.get("moneyflow_chunks_done", []))
    done_index = set(state.get("index_chunks_done", []))

    for chunk_start, chunk_end in _iter_month_chunks(start_date, end_date, months=chunk_months):
        chunk_key = f"{chunk_start}_{chunk_end}"
        _print(f"[daily] chunk {chunk_key}")

        if chunk_key not in done_daily_basic:
            fundamentals = tushare.fetch_fundamentals(
                pro,
                assets=assets,
                start_date=chunk_start,
                end_date=chunk_end,
                include_daily_basic=True,
                include_roe=False,
                token=token,
            )
            daily_basic = _canonicalize_daily_basic(fundamentals.pb_raw)
            if not daily_basic.empty:
                catalog.upsert_table(
                    "daily_basic",
                    daily_basic,
                    key_cols=("date", "asset"),
                    partition_column="date",
                )
                _dataset_note_version(
                    catalog,
                    table_names=("daily_basic",),
                    notes={
                        "source_vendor": "tushare",
                        "operation": "fill_remaining_data",
                        "stage": "daily_basic",
                        "chunk_start": chunk_start,
                        "chunk_end": chunk_end,
                    },
                )
                _print(f"[daily_basic] rows={len(daily_basic)}")
            done_daily_basic.add(chunk_key)
            state["daily_basic_chunks_done"] = sorted(done_daily_basic)
            _save_state(_state_path(catalog), state)

        if chunk_key not in done_moneyflow:
            moneyflow_raw, _ = tushare.fetch_moneyflow(
                pro,
                start_date=chunk_start,
                end_date=chunk_end,
                assets=assets,
            )
            moneyflow = _canonicalize_moneyflow(moneyflow_raw)
            if not moneyflow.empty:
                catalog.upsert_table(
                    "moneyflow",
                    moneyflow,
                    key_cols=("date", "asset"),
                    partition_column="date",
                )
                _dataset_note_version(
                    catalog,
                    table_names=("moneyflow",),
                    notes={
                        "source_vendor": "tushare",
                        "operation": "fill_remaining_data",
                        "stage": "moneyflow",
                        "chunk_start": chunk_start,
                        "chunk_end": chunk_end,
                    },
                )
                _print(f"[moneyflow] rows={len(moneyflow)}")
            done_moneyflow.add(chunk_key)
            state["moneyflow_chunks_done"] = sorted(done_moneyflow)
            _save_state(_state_path(catalog), state)

        if chunk_key not in done_index:
            index_raw, _ = tushare.fetch_index_membership(
                pro,
                start_date=chunk_start,
                end_date=chunk_end,
                assets=assets,
            )
            index_membership = _canonicalize_index_membership(index_raw)
            if not index_membership.empty:
                catalog.upsert_table(
                    "index_membership",
                    index_membership,
                    key_cols=("date", "index_code", "asset"),
                    partition_column="date",
                )
                _dataset_note_version(
                    catalog,
                    table_names=("index_membership",),
                    notes={
                        "source_vendor": "tushare",
                        "operation": "fill_remaining_data",
                        "stage": "index_membership",
                        "chunk_start": chunk_start,
                        "chunk_end": chunk_end,
                    },
                )
                _print(f"[index_membership] rows={len(index_membership)}")
            done_index.add(chunk_key)
            state["index_chunks_done"] = sorted(done_index)
            _save_state(_state_path(catalog), state)


def _run_industry_fill(
    *,
    catalog: DataCatalog,
    token: str,
    snapshot_date: str,
    state: dict[str, object],
) -> None:
    if bool(state.get("industry_done")):
        return
    pro = tushare._build_tushare_client(token)
    _print(f"[industry] fetching classification snapshot={snapshot_date}")
    classification_raw, _ = tushare.fetch_industry_classification(pro, snapshot_date=snapshot_date)
    classification = _canonicalize_industry_classification(classification_raw)
    if not classification.empty:
        catalog.upsert_table(
            "industry_classification",
            classification,
            key_cols=("snapshot_date", "industry_standard", "index_code"),
            partition_column="snapshot_date",
        )
    _print("[industry] fetching membership history")
    membership_raw, _ = tushare.fetch_industry_membership(pro, classification=classification_raw)
    membership = _canonicalize_industry_membership(membership_raw)
    if not membership.empty:
        catalog.upsert_table(
            "industry_membership",
            membership,
            key_cols=("industry_standard", "asset", "l3_code", "in_date", "out_date"),
            partition_column="in_date",
        )
    if not classification.empty or not membership.empty:
        _dataset_note_version(
            catalog,
            table_names=("industry_classification", "industry_membership"),
            notes={
                "source_vendor": "tushare",
                "operation": "fill_remaining_data",
                "stage": "industry",
                "snapshot_date": snapshot_date,
            },
        )
    _print(
        f"[industry] classification_rows={len(classification)} membership_rows={len(membership)}"
    )
    state["industry_done"] = True
    _save_state(_state_path(catalog), state)


def _run_liquidity_fill(
    *,
    catalog: DataCatalog,
    data_root: Path,
    start_date: str,
    end_date: str,
    chunk_months: int,
    state: dict[str, object],
) -> None:
    done = set(state.get("liquidity_chunks_done", []))
    canonical_root = data_root / "canonical"
    bars_glob = str(canonical_root / "daily_bars" / "year=*" / "month=*" / "*.parquet")
    status_glob = str(canonical_root / "asset_status" / "year=*" / "month=*" / "*.parquet")

    for chunk_start, chunk_end in _iter_month_chunks(start_date, end_date, months=chunk_months):
        chunk_key = f"{chunk_start}_{chunk_end}"
        if chunk_key in done:
            continue
        overlap_start = (pd.Timestamp(chunk_start) - pd.Timedelta(days=60)).strftime("%Y-%m-%d")
        _print(f"[liquidity] chunk {chunk_key} overlap_start={overlap_start}")
        con = duckdb.connect()
        try:
            frame = con.execute(
                f"""
                WITH bars AS (
                    SELECT
                        CAST(date AS DATE) AS date,
                        asset,
                        amount,
                        COALESCE(is_limit_up, 0) AS is_limit_up,
                        COALESCE(is_limit_down, 0) AS is_limit_down
                    FROM read_parquet('{bars_glob}', hive_partitioning=1)
                    WHERE CAST(date AS DATE) BETWEEN DATE '{overlap_start}' AND DATE '{chunk_end}'
                ),
                status AS (
                    SELECT
                        CAST(date AS DATE) AS date,
                        asset,
                        COALESCE(is_suspended, 0) AS is_suspended,
                        COALESCE(is_st, 0) AS is_st
                    FROM read_parquet('{status_glob}', hive_partitioning=1)
                    WHERE CAST(date AS DATE) BETWEEN DATE '{chunk_start}' AND DATE '{chunk_end}'
                ),
                rolling AS (
                    SELECT
                        b.date,
                        b.asset,
                        b.amount,
                        AVG(b.amount) OVER (
                            PARTITION BY b.asset
                            ORDER BY b.date
                            ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
                        ) AS avg_amount_20d,
                        b.is_limit_up,
                        b.is_limit_down,
                        COALESCE(s.is_suspended, 0) AS is_suspended,
                        COALESCE(s.is_st, 0) AS is_st
                    FROM bars b
                    LEFT JOIN status s
                    USING (date, asset)
                ),
                ranked AS (
                    SELECT
                        *,
                        ROW_NUMBER() OVER (
                            PARTITION BY date
                            ORDER BY avg_amount_20d DESC NULLS LAST, asset
                        ) AS amount_rank,
                        CAST(
                            ROW_NUMBER() OVER (
                                PARTITION BY date
                                ORDER BY avg_amount_20d DESC NULLS LAST, asset
                            ) AS DOUBLE
                        ) / NULLIF(COUNT(*) OVER (PARTITION BY date), 0) AS amount_percentile
                    FROM rolling
                    WHERE date BETWEEN DATE '{chunk_start}' AND DATE '{chunk_end}'
                )
                SELECT
                    strftime(date, '%Y-%m-%d') AS date,
                    asset,
                    amount,
                    avg_amount_20d,
                    amount_rank,
                    amount_percentile,
                    LEAST(
                        5,
                        GREATEST(1, CAST(FLOOR(amount_percentile * 5 - 1e-12) + 1 AS BIGINT))
                    ) AS liquidity_tier,
                    CASE
                        WHEN is_suspended = 0 AND amount IS NOT NULL AND amount > 0 THEN 1
                        ELSE 0
                    END AS is_tradable,
                    CASE
                        WHEN is_suspended = 0 AND is_st = 0
                             AND amount IS NOT NULL AND amount > 0
                             AND is_limit_up = 0 THEN 1
                        ELSE 0
                    END AS can_buy,
                    CASE
                        WHEN is_suspended = 0 AND is_st = 0
                             AND amount IS NOT NULL AND amount > 0
                             AND is_limit_down = 0 THEN 1
                        ELSE 0
                    END AS can_sell
                FROM ranked
                ORDER BY date, asset
                """
            ).fetchdf()
        finally:
            con.close()
        if not frame.empty:
            catalog.upsert_table(
                "liquidity_profile",
                frame,
                key_cols=("date", "asset"),
                partition_column="date",
            )
            _dataset_note_version(
                catalog,
                table_names=("liquidity_profile",),
                notes={
                    "source_vendor": "derived",
                    "operation": "fill_remaining_data",
                    "stage": "liquidity_profile",
                    "chunk_start": chunk_start,
                    "chunk_end": chunk_end,
                },
            )
            _print(f"[liquidity] rows={len(frame)}")
        done.add(chunk_key)
        state["liquidity_chunks_done"] = sorted(done)
        _save_state(_state_path(catalog), state)


def _state_path(catalog: DataCatalog) -> Path:
    return catalog.metadata_root / "ops" / "tushare_remaining_fill_state.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="补全 Tushare 剩余研究数据并按块落盘。")
    parser.add_argument("--token", required=True)
    parser.add_argument(
        "--data-root", default=str(Path.home() / ".local" / "share" / "alpha-lab" / "data")
    )
    parser.add_argument("--daily-start-date", default="2016-04-18")
    parser.add_argument("--financial-start-date", default="2015-01-01")
    parser.add_argument("--end-date", default="2026-04-15")
    parser.add_argument("--financial-batch-size", type=int, default=100)
    parser.add_argument("--daily-chunk-months", type=int, default=12)
    parser.add_argument(
        "--steps",
        default="financial,daily,industry,liquidity,validate",
        help="Comma-separated steps: financial,daily,industry,liquidity,validate",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    catalog = DataCatalog(root=data_root)
    catalog.ensure_layout()
    state = _load_state(_state_path(catalog))
    steps = {item.strip() for item in args.steps.split(",") if item.strip()}

    if "financial" in steps:
        _run_financial_batches(
            catalog=catalog,
            token=args.token,
            financial_start_date=args.financial_start_date,
            end_date=args.end_date,
            batch_size=args.financial_batch_size,
            state=state,
        )
    if "daily" in steps:
        _run_daily_sidecar_chunks(
            catalog=catalog,
            token=args.token,
            start_date=args.daily_start_date,
            end_date=args.end_date,
            chunk_months=args.daily_chunk_months,
            state=state,
        )
    if "industry" in steps:
        _run_industry_fill(
            catalog=catalog,
            token=args.token,
            snapshot_date=args.end_date,
            state=state,
        )
    if "liquidity" in steps:
        _run_liquidity_fill(
            catalog=catalog,
            data_root=data_root,
            start_date=args.daily_start_date,
            end_date=args.end_date,
            chunk_months=args.daily_chunk_months,
            state=state,
        )
    if "validate" in steps:
        report = catalog.validate_core_dataset()
        _print(f"[validate] status={report.status} report={report.report_path}")


if __name__ == "__main__":
    main()
