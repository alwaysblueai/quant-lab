from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from alpha_lab.data_adapters.tushare_adapter import FundamentalFetchResult
from alpha_lab.exceptions import AlphaLabConfigError, AlphaLabDataError

from .catalog import (
    CaseInputExportResult,
    DataCatalog,
    DatasetVersion,
    SliceSpec,
    SnapshotManifest,
)


@dataclass(frozen=True)
class TushareCoreIngestResult:
    snapshot_manifest: SnapshotManifest
    dataset_version: DatasetVersion
    written_tables: tuple[str, ...]
    row_counts: dict[str, int]
    quality_notes: dict[str, object]
    chunk_count: int = 1


type _IngestTableResult = tuple[pd.DataFrame, int]
type _IngestTaskResult = _IngestTableResult | FundamentalFetchResult | pd.DataFrame


class TushareIngestor:
    """Ingest raw Tushare data into the external Parquet-backed data store."""

    def __init__(self, catalog: DataCatalog | None = None) -> None:
        self.catalog = catalog or DataCatalog()

    def ingest_core(
        self,
        *,
        start_date: str,
        end_date: str,
        token: str | None = None,
        assets: tuple[str, ...] | None = None,
        asset_limit: int | None = None,
        mode: str = "full",
        include_reference_data: bool = True,
        daily_research_only: bool = False,
        progress_callback: Callable[[str], None] | None = None,
    ) -> TushareCoreIngestResult:
        from alpha_lab.data_adapters import tushare_adapter as tushare

        resolved_mode = _normalize_ingest_mode(
            mode=mode,
            daily_research_only=daily_research_only,
            default_mode="full",
        )
        self.catalog.ensure_layout()
        resolved_token = tushare._require_tushare_token(token)
        pro = tushare._build_tushare_client(resolved_token)
        explicit_assets = tuple(sorted(set(assets))) if assets is not None else None
        if explicit_assets is not None and asset_limit is not None:
            explicit_assets = explicit_assets[:asset_limit]

        prices = pd.DataFrame(
            columns=[
                "date",
                "asset",
                "open",
                "high",
                "low",
                "close",
                "pre_close",
                "volume",
                "amount",
            ]
        )
        prices_dup_count = 0
        trade_calendar = pd.DataFrame(columns=["date", "exchange", "is_open"])
        selected_assets: tuple[str, ...]
        if resolved_mode in {"daily", "full"}:
            if progress_callback is not None:
                progress_callback(f"fetching prices for {start_date} -> {end_date}")
            prices, prices_dup_count = tushare.fetch_prices(
                pro,
                start_date=start_date,
                end_date=end_date,
                assets=explicit_assets,
            )
            if explicit_assets is None and asset_limit is not None:
                selected_asset_window = tuple(
                    sorted(prices["asset"].astype(str).unique().tolist())[:asset_limit]
                )
                prices = prices[prices["asset"].isin(selected_asset_window)].copy()
            selected_assets = tuple(sorted(prices["asset"].astype(str).unique().tolist()))
            trade_calendar = _build_trade_calendar(prices)
        else:
            selected_assets = explicit_assets or _resolve_assets_from_instruments(
                pro,
                start_date=start_date,
                end_date=end_date,
                asset_limit=asset_limit,
            )
        if not selected_assets:
            raise AlphaLabDataError("No assets available after applying filters.")
        if progress_callback is not None:
            progress_callback(f"resolved {len(selected_assets)} assets")

        tasks: dict[str, tuple[Callable[..., _IngestTaskResult], dict[str, object]]] = {}
        if resolved_mode in {"daily", "full"}:
            tasks.update(
                {
                    "adj_factor": (
                        tushare.fetch_adj_factor,
                        {
                            "start_date": start_date,
                            "end_date": end_date,
                            "assets": selected_assets,
                        },
                    ),
                    "stk_limit": (
                        tushare.fetch_stk_limit,
                        {
                            "start_date": start_date,
                            "end_date": end_date,
                            "assets": selected_assets,
                        },
                    ),
                    "suspend_status": (
                        tushare.fetch_suspend_status,
                        {
                            "start_date": start_date,
                            "end_date": end_date,
                            "assets": selected_assets,
                        },
                    ),
                    "st_name_events": (
                        tushare.fetch_st_name_events,
                        {
                            "start_date": start_date,
                            "end_date": end_date,
                            "assets": selected_assets,
                        },
                    ),
                    "index_membership": (
                        tushare.fetch_index_membership,
                        {
                            "start_date": start_date,
                            "end_date": end_date,
                            "assets": selected_assets,
                        },
                    ),
                    "moneyflow": (
                        tushare.fetch_moneyflow,
                        {
                            "start_date": start_date,
                            "end_date": end_date,
                            "assets": selected_assets,
                        },
                    ),
                }
            )
        tasks["fundamentals"] = (
            tushare.fetch_fundamentals,
            {
                "assets": selected_assets,
                "start_date": start_date,
                "end_date": end_date,
                "include_daily_basic": resolved_mode in {"daily", "full"},
                "include_roe": resolved_mode in {"fundamental", "full"} and not daily_research_only,
                "token": resolved_token,
            },
        )
        if include_reference_data:
            tasks["instruments"] = (
                _fetch_instruments_optional,
                {
                    "assets": selected_assets,
                },
            )
            if resolved_mode in {"fundamental", "full"}:
                tasks["industry_classification"] = (
                    tushare.fetch_industry_classification,
                    {
                        "snapshot_date": end_date,
                    },
                )
        task_results: dict[str, _IngestTaskResult] = {}
        max_workers = min(len(tasks), 6)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {}
            for name, (fn, kwargs) in tasks.items():
                if progress_callback is not None:
                    progress_callback(f"fetching {name}")
                future = pool.submit(_run_tushare_task, resolved_token, fn, kwargs)
                futures[future] = name
            for future in as_completed(futures):
                name = futures[future]
                task_results[name] = future.result()
                if progress_callback is not None:
                    progress_callback(f"completed {name}")

        adj_factor_raw, adj_factor_dup_count = _extract_table_result(
            task_results.get("adj_factor"),
            default_columns=["date", "asset", "adj_factor"],
        )
        stk_limit_raw, stk_limit_dup_count = _extract_table_result(
            task_results.get("stk_limit"),
            default_columns=["date", "asset", "up_limit", "down_limit"],
        )
        suspend_status_raw, suspend_dup_count = _extract_table_result(
            task_results.get("suspend_status"),
            default_columns=["date", "asset", "is_suspended"],
        )
        st_name_events_raw, st_name_dup_count = _extract_table_result(
            task_results.get("st_name_events"),
            default_columns=["asset", "start_date", "end_date", "is_st"],
        )
        index_membership_raw, index_membership_dup_count = _extract_table_result(
            task_results.get("index_membership"),
            default_columns=["date", "index_code", "index_name", "asset", "weight"],
        )
        moneyflow_raw, moneyflow_dup_count = _extract_table_result(
            task_results.get("moneyflow"),
            default_columns=["date", "asset", "net_mf_amount"],
        )
        fundamentals_result = task_results.get("fundamentals")
        if not isinstance(fundamentals_result, FundamentalFetchResult):
            raise AlphaLabDataError("fundamentals fetch did not return FundamentalFetchResult.")
        fundamentals = fundamentals_result
        instruments = _extract_frame_result(
            task_results.get("instruments"),
            default_columns=[
                "asset",
                "symbol",
                "name",
                "area",
                "industry",
                "market",
                "list_date",
                "delist_date",
            ],
        )
        industry_classification_raw, industry_classification_dup_count = _extract_table_result(
            task_results.get("industry_classification"),
            default_columns=["snapshot_date", "industry_standard", "index_code", "industry_name"],
        )
        industry_membership_raw = pd.DataFrame(
            columns=["industry_standard", "asset", "in_date", "out_date"]
        )
        industry_membership_dup_count = 0
        if (
            include_reference_data
            and resolved_mode in {"fundamental", "full"}
            and not industry_classification_raw.empty
        ):
            industry_membership_raw, industry_membership_dup_count = _extract_table_result(
                _run_tushare_task(
                    resolved_token,
                    tushare.fetch_industry_membership,
                    {
                        "classification": industry_classification_raw,
                    },
                ),
                default_columns=["industry_standard", "asset", "in_date", "out_date"],
            )
        financial_indicator, dropped_missing_ann_date = _canonicalize_financial_indicator(
            fundamentals.roe_raw,
            roe_source_column=fundamentals.roe_source_column,
        )
        balance_sheet = _canonicalize_accounting_statement(
            fundamentals.balance_sheet_raw,
            value_columns=(
                "goodwill_balance",
                "short_term_borrow",
                "long_term_borrow",
                "bonds_payable",
                "monetary_capital",
                "tradable_fin_assets",
                "invest_real_estate",
                "derivative_fin_assets",
                "dividend_receivable",
                "interest_receivable",
                "fin_assets_avail_for_sale",
                "held_to_mty_invest",
                "other_debt_investment",
                "other_equity_investment",
                "debt_investment",
            ),
        )
        income_statement = _canonicalize_accounting_statement(
            fundamentals.income_statement_raw,
            value_columns=(
                "operating_revenue_ttm",
                "operating_cost_ttm",
                "rd_expense",
                "selling_expense",
                "admin_expense",
            ),
        )
        cash_flow_statement = _canonicalize_accounting_statement(
            fundamentals.cash_flow_statement_raw,
            value_columns=("operating_cash_flow_ttm",),
        )
        daily_bars = (
            _canonicalize_daily_bars(
                prices,
                daily_basic_raw=fundamentals.pb_raw,
                stk_limit_raw=stk_limit_raw,
            )
            if resolved_mode in {"daily", "full"}
            else pd.DataFrame(
                columns=[
                    "date",
                    "asset",
                    "open",
                    "high",
                    "low",
                    "close",
                    "pre_close",
                    "volume",
                    "amount",
                    "vwap",
                    "turnover_rate",
                    "up_limit",
                    "down_limit",
                    "is_limit_up",
                    "is_limit_down",
                ]
            )
        )
        adj_factor = (
            _canonicalize_adj_factor(adj_factor_raw)
            if resolved_mode in {"daily", "full"}
            else pd.DataFrame(columns=["date", "asset", "adj_factor"])
        )
        daily_basic = _canonicalize_daily_basic(fundamentals.pb_raw)
        asset_status = (
            _canonicalize_asset_status(
                trade_calendar=trade_calendar,
                prices=prices,
                suspend_status_raw=suspend_status_raw,
                st_name_events_raw=st_name_events_raw,
                instruments=instruments,
            )
            if resolved_mode in {"daily", "full"}
            else pd.DataFrame(columns=["date", "asset", "is_suspended", "is_st"])
        )
        index_membership = (
            _canonicalize_index_membership(index_membership_raw)
            if resolved_mode in {"daily", "full"}
            else pd.DataFrame(columns=["date", "index_code", "index_name", "asset", "weight"])
        )
        moneyflow = (
            _canonicalize_moneyflow(moneyflow_raw)
            if resolved_mode in {"daily", "full"}
            else pd.DataFrame(columns=["date", "asset", "net_mf_amount"])
        )
        liquidity_profile = (
            _build_liquidity_profile(
                daily_bars=daily_bars,
                asset_status=asset_status,
            )
            if resolved_mode in {"daily", "full"}
            else pd.DataFrame(
                columns=[
                    "date",
                    "asset",
                    "avg_amount_20d",
                    "liquidity_tier",
                    "is_tradable",
                    "can_buy",
                    "can_sell",
                ]
            )
        )
        industry_classification = (
            _canonicalize_industry_classification(industry_classification_raw)
            if include_reference_data and resolved_mode in {"fundamental", "full"}
            else pd.DataFrame(
                columns=[
                    "snapshot_date",
                    "industry_standard",
                    "index_code",
                    "industry_name",
                    "parent_code",
                    "level",
                    "industry_code",
                    "is_published",
                ]
            )
        )
        industry_membership = (
            _canonicalize_industry_membership(industry_membership_raw)
            if include_reference_data and resolved_mode in {"fundamental", "full"}
            else pd.DataFrame(
                columns=[
                    "industry_standard",
                    "asset",
                    "in_date",
                    "out_date",
                    "is_new",
                    "l1_code",
                    "l1_name",
                    "l2_code",
                    "l2_name",
                    "l3_code",
                    "l3_name",
                ]
            )
        )
        if progress_callback is not None:
            progress_callback("writing raw snapshot")

        raw_tables: dict[str, pd.DataFrame] = {
            "pb_raw": fundamentals.pb_raw.reset_index(drop=True),
            "roe_raw": fundamentals.roe_raw.reset_index(drop=True),
            "balance_sheet_raw": fundamentals.balance_sheet_raw.reset_index(drop=True),
            "income_statement_raw": fundamentals.income_statement_raw.reset_index(drop=True),
            "cash_flow_statement_raw": fundamentals.cash_flow_statement_raw.reset_index(drop=True),
            "instruments": instruments.reset_index(drop=True),
        }
        if resolved_mode in {"daily", "full"}:
            raw_tables.update(
                {
                    "prices_raw": prices.reset_index(drop=True),
                    "adj_factor_raw": adj_factor_raw.reset_index(drop=True),
                    "stk_limit_raw": stk_limit_raw.reset_index(drop=True),
                    "suspend_status_raw": suspend_status_raw.reset_index(drop=True),
                    "st_name_events_raw": st_name_events_raw.reset_index(drop=True),
                    "index_membership_raw": index_membership_raw.reset_index(drop=True),
                    "moneyflow_raw": moneyflow_raw.reset_index(drop=True),
                    "trade_calendar": trade_calendar.reset_index(drop=True),
                }
            )
        if include_reference_data and resolved_mode in {"fundamental", "full"}:
            raw_tables.update(
                {
                    "industry_classification_raw": industry_classification_raw.reset_index(
                        drop=True
                    ),
                    "industry_membership_raw": industry_membership_raw.reset_index(drop=True),
                }
            )

        snapshot_manifest = self.catalog.write_raw_snapshot(
            vendor="tushare",
            dataset_name="core",
            tables=raw_tables,
            request_params={
                "start_date": start_date,
                "end_date": end_date,
                "assets": list(selected_assets),
                "asset_limit": asset_limit,
                "mode": resolved_mode,
            },
            time_range={"start_date": start_date, "end_date": end_date},
            notes={
                "prices_raw_duplicate_rows": prices_dup_count,
                "adj_factor_raw_duplicate_rows": adj_factor_dup_count,
                "stk_limit_raw_duplicate_rows": stk_limit_dup_count,
                "suspend_status_raw_duplicate_rows": suspend_dup_count,
                "st_name_events_raw_duplicate_rows": st_name_dup_count,
                "index_membership_raw_duplicate_rows": index_membership_dup_count,
                "moneyflow_raw_duplicate_rows": moneyflow_dup_count,
                "industry_classification_raw_duplicate_rows": industry_classification_dup_count,
                "industry_membership_raw_duplicate_rows": industry_membership_dup_count,
                **fundamentals.dedup_counts,
                "roe_source_column": fundamentals.roe_source_column,
                "financial_indicator_rows_dropped_missing_ann_date": dropped_missing_ann_date,
                "daily_research_only": daily_research_only,
                "mode": resolved_mode,
            },
        )
        raw_validation_report = self.catalog.validate_raw_snapshot(snapshot_manifest.snapshot_id)

        written_tables: list[str] = []
        if progress_callback is not None:
            progress_callback("upserting canonical tables")
        if resolved_mode in {"daily", "full"} and not trade_calendar.empty:
            self.catalog.upsert_table(
                "trade_calendar",
                trade_calendar,
                key_cols=("date",),
                partition_column="date",
            )
            written_tables.append("trade_calendar")
        if not instruments.empty:
            self.catalog.upsert_table(
                "instruments",
                instruments,
                key_cols=("asset",),
                partition_column="list_date",
            )
            written_tables.append("instruments")
        if resolved_mode in {"daily", "full"}:
            self.catalog.upsert_table(
                "daily_bars",
                daily_bars,
                key_cols=("date", "asset"),
                partition_column="date",
            )
            self.catalog.upsert_table(
                "adj_factor",
                adj_factor,
                key_cols=("date", "asset"),
                partition_column="date",
            )
            self.catalog.upsert_table(
                "daily_basic",
                daily_basic,
                key_cols=("date", "asset"),
                partition_column="date",
            )
            self.catalog.upsert_table(
                "asset_status",
                asset_status,
                key_cols=("date", "asset"),
                partition_column="date",
            )
            self.catalog.upsert_table(
                "index_membership",
                index_membership,
                key_cols=("date", "index_code", "asset"),
                partition_column="date",
            )
            self.catalog.upsert_table(
                "moneyflow",
                moneyflow,
                key_cols=("date", "asset"),
                partition_column="date",
            )
            self.catalog.upsert_table(
                "liquidity_profile",
                liquidity_profile,
                key_cols=("date", "asset"),
                partition_column="date",
            )
            written_tables.extend(
                [
                    "daily_bars",
                    "adj_factor",
                    "daily_basic",
                    "asset_status",
                    "index_membership",
                    "moneyflow",
                    "liquidity_profile",
                ]
            )
        if resolved_mode in {"fundamental", "full"}:
            self.catalog.upsert_table(
                "financial_indicator",
                financial_indicator,
                key_cols=("asset", "ann_date", "end_date"),
                partition_column="ann_date",
            )
            self.catalog.upsert_table(
                "balance_sheet",
                balance_sheet,
                key_cols=("asset", "ann_date", "end_date"),
                partition_column="ann_date",
            )
            self.catalog.upsert_table(
                "income_statement",
                income_statement,
                key_cols=("asset", "ann_date", "end_date"),
                partition_column="ann_date",
            )
            self.catalog.upsert_table(
                "cash_flow_statement",
                cash_flow_statement,
                key_cols=("asset", "ann_date", "end_date"),
                partition_column="ann_date",
            )
            if not industry_classification.empty:
                self.catalog.upsert_table(
                    "industry_classification",
                    industry_classification,
                    key_cols=("snapshot_date", "industry_standard", "index_code"),
                    partition_column="snapshot_date",
                )
                written_tables.append("industry_classification")
            if not industry_membership.empty:
                self.catalog.upsert_table(
                    "industry_membership",
                    industry_membership,
                    key_cols=("industry_standard", "asset", "l3_code", "in_date", "out_date"),
                    partition_column="in_date",
                )
                written_tables.append("industry_membership")
            written_tables.extend(
                [
                    "financial_indicator",
                    "balance_sheet",
                    "income_statement",
                    "cash_flow_statement",
                ]
            )

        dataset_version = self.catalog.write_dataset_version(
            dataset_name=DataCatalog.CORE_DATASET_NAME,
            table_names=tuple(written_tables),
            raw_snapshot_id=snapshot_manifest.snapshot_id,
            notes={
                "start_date": start_date,
                "end_date": end_date,
                "n_assets": len(selected_assets),
                "roe_source_column": fundamentals.roe_source_column,
                "financial_indicator_rows_dropped_missing_ann_date": dropped_missing_ann_date,
                "daily_research_only": daily_research_only,
                "mode": resolved_mode,
            },
        )
        if progress_callback is not None:
            progress_callback(f"completed {start_date} -> {end_date}")
        row_counts: dict[str, int] = {"instruments": int(len(instruments))}
        if resolved_mode in {"daily", "full"}:
            row_counts.update(
                {
                    "daily_bars": int(len(daily_bars)),
                    "adj_factor": int(len(adj_factor)),
                    "daily_basic": int(len(daily_basic)),
                    "asset_status": int(len(asset_status)),
                    "index_membership": int(len(index_membership)),
                    "moneyflow": int(len(moneyflow)),
                    "liquidity_profile": int(len(liquidity_profile)),
                    "trade_calendar": int(len(trade_calendar)),
                }
            )
        if resolved_mode in {"fundamental", "full"}:
            row_counts.update(
                {
                    "financial_indicator": int(len(financial_indicator)),
                    "balance_sheet": int(len(balance_sheet)),
                    "income_statement": int(len(income_statement)),
                    "cash_flow_statement": int(len(cash_flow_statement)),
                    "industry_classification": int(len(industry_classification)),
                    "industry_membership": int(len(industry_membership)),
                }
            )
        return TushareCoreIngestResult(
            snapshot_manifest=snapshot_manifest,
            dataset_version=dataset_version,
            written_tables=tuple(written_tables),
            row_counts=row_counts,
            quality_notes={
                "roe_source_column": fundamentals.roe_source_column,
                "financial_indicator_rows_dropped_missing_ann_date": dropped_missing_ann_date,
                "daily_research_only": daily_research_only,
                "mode": resolved_mode,
                "raw_validation_report_path": str(raw_validation_report.report_path),
                "raw_validation_ok": raw_validation_report.ok,
            },
            chunk_count=1,
        )

    def ingest_daily_core(self, **kwargs) -> TushareCoreIngestResult:  # type: ignore[no-untyped-def]
        return self.ingest_core(mode="daily", **kwargs)

    def ingest_fundamental_core(self, **kwargs) -> TushareCoreIngestResult:  # type: ignore[no-untyped-def]
        return self.ingest_core(mode="fundamental", **kwargs)

    def ingest_core_full(self, **kwargs) -> TushareCoreIngestResult:  # type: ignore[no-untyped-def]
        return self.ingest_core(mode="full", **kwargs)

    def ingest_core_chunked(
        self,
        *,
        start_date: str,
        end_date: str,
        token: str | None = None,
        assets: tuple[str, ...] | None = None,
        asset_limit: int | None = None,
        chunk_months: int = 6,
        mode: str = "full",
        include_reference_data: bool = True,
        daily_research_only: bool = False,
        progress_callback: Callable[[str], None] | None = None,
    ) -> TushareCoreIngestResult:
        from alpha_lab.data_adapters import tushare_adapter as tushare

        resolved_mode = _normalize_ingest_mode(
            mode=mode,
            daily_research_only=daily_research_only,
            default_mode="full",
        )
        chunk_ranges = _split_date_range_into_chunks(
            start_date=start_date,
            end_date=end_date,
            chunk_months=chunk_months,
        )
        resolved_assets: tuple[str, ...] | None = (
            tuple(sorted(set(assets))) if assets is not None else None
        )
        resolved_token = tushare._require_tushare_token(token)

        if resolved_assets is None and (asset_limit is not None or resolved_mode == "fundamental"):
            if progress_callback is not None:
                progress_callback("resolving stable asset universe before chunked ingest")
            pro = tushare._build_tushare_client(resolved_token)
            resolved_assets = _resolve_assets_from_instruments(
                pro,
                start_date=start_date,
                end_date=end_date,
                asset_limit=asset_limit,
            )
            if progress_callback is not None:
                progress_callback(f"stable asset universe resolved: {len(resolved_assets)} assets")
            asset_limit = None
        elif resolved_assets is not None and asset_limit is not None:
            resolved_assets = resolved_assets[:asset_limit]
            asset_limit = None

        last_result: TushareCoreIngestResult | None = None
        aggregate_counts: dict[str, int] = {}
        combined_notes: dict[str, object] = {
            "chunk_ranges": [[start, end] for start, end in chunk_ranges],
        }
        written_tables: set[str] = set()

        for index, (chunk_start, chunk_end) in enumerate(chunk_ranges, start=1):
            chunk_prefix = f"[chunk {index}/{len(chunk_ranges)} {chunk_start} -> {chunk_end}]"
            chunk_progress = _prefixed_progress_callback(progress_callback, prefix=chunk_prefix)
            if chunk_progress is not None:
                chunk_progress("starting")
            result = self.ingest_core(
                start_date=chunk_start,
                end_date=chunk_end,
                token=resolved_token,
                assets=resolved_assets,
                asset_limit=asset_limit,
                mode=resolved_mode,
                include_reference_data=include_reference_data and index == 1,
                daily_research_only=daily_research_only,
                progress_callback=chunk_progress,
            )
            last_result = result
            written_tables.update(result.written_tables)
            for name, count in result.row_counts.items():
                aggregate_counts[name] = int(aggregate_counts.get(name, 0)) + int(count)
            combined_notes.update(result.quality_notes)

        if last_result is None:
            raise AlphaLabDataError("No chunks were executed for the requested ingest window.")

        return TushareCoreIngestResult(
            snapshot_manifest=last_result.snapshot_manifest,
            dataset_version=last_result.dataset_version,
            written_tables=tuple(sorted(written_tables)),
            row_counts=aggregate_counts,
            quality_notes=combined_notes,
            chunk_count=len(chunk_ranges),
        )

    def update_core(
        self,
        *,
        end_date: str,
        token: str | None = None,
        assets: tuple[str, ...] | None = None,
        asset_limit: int | None = None,
        chunk_months: int = 6,
        mode: str = "daily",
        daily_research_only: bool = False,
        progress_callback: Callable[[str], None] | None = None,
    ) -> TushareCoreIngestResult:
        resolved_mode = _normalize_ingest_mode(
            mode=mode,
            daily_research_only=daily_research_only,
            default_mode="daily",
        )
        if resolved_mode == "fundamental":
            latest_date = self.catalog.latest_date("financial_indicator", date_field="ann_date")
            if latest_date is None:
                latest_date = self.catalog.latest_date("daily_bars")
        else:
            latest_date = self.catalog.latest_date("daily_bars")
        if latest_date is None:
            raise AlphaLabConfigError("No existing dataset found for update; run ingest first.")
        next_start = (pd.Timestamp(latest_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        if pd.Timestamp(next_start) > pd.Timestamp(end_date):
            version = self.catalog.get_current_dataset_version(DataCatalog.CORE_DATASET_NAME)
            if version is None:
                raise AlphaLabConfigError(
                    "No dataset version found even though canonical data exists."
                )
            return TushareCoreIngestResult(
                snapshot_manifest=SnapshotManifest(
                    snapshot_id="noop",
                    vendor="tushare",
                    dataset_name="core",
                    requested_at_utc=datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                    request_params={
                        "start_date": next_start,
                        "end_date": end_date,
                        "mode": resolved_mode,
                    },
                    row_counts={},
                    file_hashes={},
                    time_range={"start_date": next_start, "end_date": end_date},
                    notes={"status": "no_update_needed", "mode": resolved_mode},
                ),
                dataset_version=version,
                written_tables=tuple(),
                row_counts={},
                quality_notes={"status": "no_update_needed", "mode": resolved_mode},
                chunk_count=0,
            )
        return self.ingest_core_chunked(
            start_date=next_start,
            end_date=end_date,
            token=token,
            assets=assets,
            asset_limit=asset_limit,
            chunk_months=chunk_months,
            mode=resolved_mode,
            daily_research_only=daily_research_only,
            progress_callback=progress_callback,
        )

    def export_case_inputs(
        self,
        *,
        start_date: str,
        end_date: str,
        output_dir: str | Path,
        assets: tuple[str, ...] | None = None,
        asset_limit: int | None = None,
        factors: tuple[str, ...] = (),
        adjustment: str = "raw",
        universe_name: str = "all_ashare",
        formats: tuple[str, ...] | None = None,
        prefer_cache: bool = True,
    ) -> CaseInputExportResult:
        return self.catalog.export_case_inputs(
            slice_spec=SliceSpec(
                start_date=start_date,
                end_date=end_date,
                universe_name=universe_name,
                assets=assets,
                asset_limit=asset_limit,
                factors=factors,
                adjustment=adjustment,
            ),
            output_dir=output_dir,
            formats=formats,
            prefer_cache=prefer_cache,
        )


def _canonicalize_daily_bars(
    prices: pd.DataFrame,
    *,
    daily_basic_raw: pd.DataFrame,
    stk_limit_raw: pd.DataFrame,
) -> pd.DataFrame:
    frame = prices.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    for column in ("open", "high", "low", "pre_close", "volume", "amount"):
        if column not in frame.columns:
            frame[column] = pd.NA
    for column in ("open", "high", "low", "close", "pre_close", "volume", "amount"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    if not daily_basic_raw.empty:
        basics = daily_basic_raw.copy()
        basics["date"] = pd.to_datetime(basics["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        basics["turnover_rate"] = pd.to_numeric(basics.get("turnover_rate"), errors="coerce")
        frame = frame.merge(
            basics[["date", "asset", "turnover_rate"]].drop_duplicates(
                subset=["date", "asset"], keep="last"
            ),
            on=["date", "asset"],
            how="left",
            validate="one_to_one",
        )
    else:
        frame["turnover_rate"] = pd.NA

    if not stk_limit_raw.empty:
        limits = stk_limit_raw.copy()
        limits["date"] = pd.to_datetime(limits["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        limits["up_limit"] = pd.to_numeric(limits.get("up_limit"), errors="coerce")
        limits["down_limit"] = pd.to_numeric(limits.get("down_limit"), errors="coerce")
        frame = frame.merge(
            limits[["date", "asset", "up_limit", "down_limit"]].drop_duplicates(
                subset=["date", "asset"], keep="last"
            ),
            on=["date", "asset"],
            how="left",
            validate="one_to_one",
        )
    else:
        frame["up_limit"] = pd.NA
        frame["down_limit"] = pd.NA

    # Tushare A-share `daily` uses volume in lots and amount in thousand CNY.
    volume = pd.to_numeric(frame["volume"], errors="coerce")
    amount = pd.to_numeric(frame["amount"], errors="coerce")
    frame["vwap"] = pd.NA
    valid_vwap = volume.notna() & (volume > 0) & amount.notna()
    frame.loc[valid_vwap, "vwap"] = (amount[valid_vwap] * 10.0) / volume[valid_vwap]

    close = pd.to_numeric(frame["close"], errors="coerce")
    up_limit = pd.to_numeric(frame["up_limit"], errors="coerce")
    down_limit = pd.to_numeric(frame["down_limit"], errors="coerce")
    frame["is_limit_up"] = (
        (up_limit.notna()) & (close.notna()) & (close >= up_limit - 0.01)
    ).astype(int)
    frame["is_limit_down"] = (
        (down_limit.notna()) & (close.notna()) & (close <= down_limit + 0.01)
    ).astype(int)

    columns = [
        "date",
        "asset",
        "open",
        "high",
        "low",
        "close",
        "pre_close",
        "volume",
        "amount",
        "vwap",
        "turnover_rate",
        "up_limit",
        "down_limit",
        "is_limit_up",
        "is_limit_down",
    ]
    return frame[columns].copy()


def _resolve_assets_from_instruments(
    pro: object,
    *,
    start_date: str,
    end_date: str,
    asset_limit: int | None = None,
) -> tuple[str, ...]:
    instruments = _fetch_instruments_optional(pro, assets=tuple())
    if instruments.empty:
        raise AlphaLabDataError(
            "Unable to resolve assets for chunked ingest because instruments is empty."
        )

    window_start = pd.Timestamp(start_date)
    window_end = pd.Timestamp(end_date)
    list_date = pd.to_datetime(instruments["list_date"], errors="coerce")
    delist_date = pd.to_datetime(instruments["delist_date"], errors="coerce")
    keep = list_date.notna() & (list_date <= window_end)
    keep &= delist_date.isna() | (delist_date >= window_start)
    filtered = instruments.loc[keep, "asset"].astype(str).str.strip()
    filtered = filtered[filtered != ""]
    deduped = sorted(filtered.drop_duplicates().tolist())
    selected = tuple(deduped[:asset_limit] if asset_limit is not None else deduped)
    if not selected:
        raise AlphaLabDataError("Resolved empty asset universe for chunked ingest.")
    return selected


def _run_tushare_task(
    token: str,
    fn: Callable[..., _IngestTaskResult],
    kwargs: dict[str, object],
) -> _IngestTaskResult:
    from alpha_lab.data_adapters import tushare_adapter as tushare

    pro = tushare._build_tushare_client(token)
    return fn(pro, **kwargs)


def _extract_table_result(
    result: _IngestTaskResult | None,
    *,
    default_columns: list[str],
) -> _IngestTableResult:
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], pd.DataFrame):
        return result[0], _coerce_int_value(result[1])
    return pd.DataFrame(columns=default_columns), 0


def _extract_frame_result(
    result: _IngestTaskResult | None,
    *,
    default_columns: list[str],
) -> pd.DataFrame:
    if isinstance(result, pd.DataFrame):
        return result
    return pd.DataFrame(columns=default_columns)


def _coerce_int_value(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if pd.isna(value):
            return 0
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return 0
        try:
            return int(float(text))
        except ValueError:
            return 0
    return 0


def _normalize_ingest_mode(
    *,
    mode: str,
    daily_research_only: bool,
    default_mode: str,
) -> str:
    normalized = str(mode or default_mode).strip().lower()
    if normalized not in {"daily", "fundamental", "full"}:
        raise AlphaLabConfigError("mode must be one of ['daily', 'fundamental', 'full']")
    if daily_research_only and normalized == "full":
        return "daily"
    return normalized


def _split_date_range_into_chunks(
    *,
    start_date: str,
    end_date: str,
    chunk_months: int,
) -> list[tuple[str, str]]:
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    if end_ts < start_ts:
        raise AlphaLabConfigError("end_date must be >= start_date")
    if chunk_months <= 0:
        return [(start_ts.strftime("%Y-%m-%d"), end_ts.strftime("%Y-%m-%d"))]

    ranges: list[tuple[str, str]] = []
    cursor = start_ts
    while cursor <= end_ts:
        chunk_end = min(cursor + pd.DateOffset(months=chunk_months) - pd.Timedelta(days=1), end_ts)
        ranges.append((cursor.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")))
        cursor = chunk_end + pd.Timedelta(days=1)
    return ranges


def _prefixed_progress_callback(
    callback: Callable[[str], None] | None,
    *,
    prefix: str,
) -> Callable[[str], None] | None:
    if callback is None:
        return None

    def _wrapped(message: str) -> None:
        callback(f"{prefix} {message}")

    return _wrapped


def _canonicalize_adj_factor(adj_factor_raw: pd.DataFrame) -> pd.DataFrame:
    frame = adj_factor_raw.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    frame["adj_factor"] = pd.to_numeric(frame["adj_factor"], errors="coerce")
    frame = frame.dropna(subset=["date", "asset", "adj_factor"]).copy()
    frame = frame[frame["adj_factor"] > 0].copy()
    return frame[["date", "asset", "adj_factor"]].copy()


def _canonicalize_daily_basic(pb_raw: pd.DataFrame) -> pd.DataFrame:
    frame = pb_raw.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    numeric_columns = (
        "pb",
        "pe",
        "pe_ttm",
        "ps",
        "ps_ttm",
        "dv_ttm",
        "total_mv",
        "circ_mv",
        "turnover_rate",
        "turnover_rate_f",
        "volume_ratio",
    )
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame.get(column), errors="coerce")
    return frame[["date", "asset", *numeric_columns]].copy()


def _canonicalize_moneyflow(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["date", "asset", "net_mf_amount"])
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    numeric_columns = [column for column in out.columns if column not in {"date", "asset"}]
    for column in numeric_columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.dropna(subset=["date", "asset"]).copy()
    out = out[out["asset"].astype(str).str.strip() != ""].copy()
    out = out.drop_duplicates(subset=["date", "asset"], keep="last")
    out = out.sort_values(["date", "asset"], kind="mergesort").reset_index(drop=True)
    return out


def _canonicalize_financial_indicator(
    roe_raw: pd.DataFrame,
    *,
    roe_source_column: str,
) -> tuple[pd.DataFrame, int]:
    frame = roe_raw.copy()
    ann_date = pd.to_datetime(frame["ann_date"], errors="coerce")
    end_date = pd.to_datetime(frame["end_date"], errors="coerce")
    roe_value = pd.to_numeric(frame["roe_value"], errors="coerce")
    dropped_missing_ann_date = int((ann_date.isna() & roe_value.notna()).sum())

    out = pd.DataFrame(
        {
            "asset": frame["asset"].astype(str).str.strip(),
            "ann_date": ann_date,
            "end_date": end_date,
            "roe_value": roe_value,
            "roe_source_column": roe_source_column,
        }
    )
    out = out.dropna(subset=["asset", "ann_date", "roe_value"]).copy()
    out = out[out["asset"] != ""].copy()
    out["ann_date"] = out["ann_date"].dt.strftime("%Y-%m-%d")
    out["end_date"] = out["end_date"].dt.strftime("%Y-%m-%d")
    out = out.drop_duplicates(subset=["asset", "ann_date", "end_date"], keep="last")
    out = out.sort_values(["asset", "ann_date", "end_date"], kind="mergesort").reset_index(
        drop=True
    )
    return out, dropped_missing_ann_date


def _canonicalize_accounting_statement(
    frame: pd.DataFrame,
    *,
    value_columns: tuple[str, ...],
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["asset", "ann_date", "end_date", *value_columns])

    out = frame.copy()
    out["asset"] = out["asset"].astype(str).str.strip()
    out["ann_date"] = pd.to_datetime(out["ann_date"], errors="coerce")
    out["end_date"] = pd.to_datetime(out["end_date"], errors="coerce")
    for column in value_columns:
        if column not in out.columns:
            out[column] = pd.NA
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.dropna(subset=["asset", "ann_date"]).copy()
    out = out[out["asset"] != ""].copy()
    out["ann_date"] = out["ann_date"].dt.strftime("%Y-%m-%d")
    out["end_date"] = out["end_date"].dt.strftime("%Y-%m-%d")
    out = out.drop_duplicates(subset=["asset", "ann_date", "end_date"], keep="last")
    out = out.sort_values(["asset", "ann_date", "end_date"], kind="mergesort").reset_index(
        drop=True
    )
    return out[["asset", "ann_date", "end_date", *value_columns]].copy()


def _build_trade_calendar(prices: pd.DataFrame) -> pd.DataFrame:
    calendar = prices[["date"]].drop_duplicates().copy()
    calendar["date"] = pd.to_datetime(calendar["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    calendar["exchange"] = "SSE"
    calendar["is_open"] = 1
    return calendar[["date", "exchange", "is_open"]].sort_values("date", kind="mergesort")


def _canonicalize_asset_status(
    *,
    trade_calendar: pd.DataFrame,
    prices: pd.DataFrame,
    suspend_status_raw: pd.DataFrame,
    st_name_events_raw: pd.DataFrame,
    instruments: pd.DataFrame,
) -> pd.DataFrame:
    price_keys = prices[["date", "asset"]].drop_duplicates().copy()
    if not suspend_status_raw.empty:
        suspend_keys = suspend_status_raw[["date", "asset"]].drop_duplicates().copy()
        status_index = pd.concat([price_keys, suspend_keys], ignore_index=True)
        status_index = status_index.drop_duplicates(subset=["date", "asset"], keep="last")
    else:
        status_index = price_keys
    if status_index.empty:
        return pd.DataFrame(columns=["date", "asset", "is_suspended", "is_st"])

    status = status_index.copy()
    status["date_ts"] = pd.to_datetime(status["date"], errors="coerce")
    status["is_suspended"] = 0
    status["is_st"] = 0

    if not suspend_status_raw.empty:
        suspend = suspend_status_raw.copy()
        suspend["date"] = pd.to_datetime(suspend["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        suspend["is_suspended"] = (
            pd.to_numeric(suspend.get("is_suspended"), errors="coerce").fillna(0).astype(int)
        )
        status = status.merge(
            suspend[["date", "asset", "is_suspended"]].drop_duplicates(
                subset=["date", "asset"], keep="last"
            ),
            on=["date", "asset"],
            how="left",
            validate="one_to_one",
            suffixes=("", "_raw"),
        )
        status["is_suspended"] = (
            status["is_suspended_raw"].fillna(status["is_suspended"]).astype(int)
        )
        status = status.drop(columns=["is_suspended_raw"])

    if not st_name_events_raw.empty:
        st_events = st_name_events_raw.copy()
        st_events = st_events[
            pd.to_numeric(st_events.get("is_st"), errors="coerce").fillna(0) > 0
        ].copy()
        if not st_events.empty:
            st_events["start_date"] = pd.to_datetime(st_events["start_date"], errors="coerce")
            st_events["end_date"] = pd.to_datetime(st_events["end_date"], errors="coerce")
            st_events["end_date"] = st_events["end_date"].fillna(pd.Timestamp.max.normalize())
            for asset, events in st_events.groupby("asset", sort=False):
                mask = status["asset"] == asset
                if not mask.any():
                    continue
                asset_dates = status.loc[mask, "date_ts"]
                is_st = pd.Series(False, index=asset_dates.index)
                for event in events.itertuples(index=False):
                    is_st |= (asset_dates >= event.start_date) & (asset_dates <= event.end_date)
                status.loc[mask, "is_st"] = is_st.astype(int)

    if not instruments.empty:
        inst = instruments[["asset", "list_date", "delist_date"]].copy()
        inst["list_date"] = pd.to_datetime(inst["list_date"], errors="coerce")
        inst["delist_date"] = pd.to_datetime(inst["delist_date"], errors="coerce")
        status = status.merge(inst, on="asset", how="left", validate="many_to_one")
        live_mask = status["list_date"].isna() | (status["date_ts"] >= status["list_date"])
        if "delist_date" in status:
            live_mask &= status["delist_date"].isna() | (status["date_ts"] <= status["delist_date"])
        status = status.loc[live_mask].copy()
        status = status.drop(columns=["list_date", "delist_date"])

    status["date"] = status["date_ts"].dt.strftime("%Y-%m-%d")
    status = status.drop(columns=["date_ts"])
    status = status.sort_values(["date", "asset"], kind="mergesort").reset_index(drop=True)
    return status[["date", "asset", "is_suspended", "is_st"]]


def _canonicalize_index_membership(index_membership_raw: pd.DataFrame) -> pd.DataFrame:
    if index_membership_raw.empty:
        return pd.DataFrame(columns=["date", "index_code", "index_name", "asset", "weight"])
    frame = index_membership_raw.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    frame["weight"] = pd.to_numeric(frame.get("weight"), errors="coerce")
    frame = frame.dropna(subset=["date", "index_code", "asset"]).copy()
    frame = frame.drop_duplicates(subset=["date", "index_code", "asset"], keep="last")
    frame = frame.sort_values(["date", "index_code", "asset"], kind="mergesort").reset_index(
        drop=True
    )
    return frame[["date", "index_code", "index_name", "asset", "weight"]]


def _canonicalize_industry_classification(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "snapshot_date",
                "industry_standard",
                "index_code",
                "industry_name",
                "parent_code",
                "level",
                "industry_code",
                "is_published",
            ]
        )
    out = frame.copy()
    out["snapshot_date"] = pd.to_datetime(out["snapshot_date"], errors="coerce").dt.strftime(
        "%Y-%m-%d"
    )
    out["is_published"] = pd.to_numeric(out.get("is_published"), errors="coerce")
    out = out.dropna(subset=["snapshot_date", "industry_standard", "index_code"]).copy()
    out = out.drop_duplicates(
        subset=["snapshot_date", "industry_standard", "index_code"], keep="last"
    )
    out = out.sort_values(
        ["snapshot_date", "industry_standard", "level", "index_code"], kind="mergesort"
    ).reset_index(drop=True)
    return out[
        [
            "snapshot_date",
            "industry_standard",
            "index_code",
            "industry_name",
            "parent_code",
            "level",
            "industry_code",
            "is_published",
        ]
    ].copy()


def _canonicalize_industry_membership(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "industry_standard",
                "asset",
                "in_date",
                "out_date",
                "is_new",
                "l1_code",
                "l1_name",
                "l2_code",
                "l2_name",
                "l3_code",
                "l3_name",
            ]
        )
    out = frame.copy()
    out["in_date"] = pd.to_datetime(out["in_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out["out_date"] = pd.to_datetime(out["out_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out["is_new"] = pd.to_numeric(out.get("is_new"), errors="coerce")
    out = out.dropna(subset=["industry_standard", "asset", "in_date", "l3_code"]).copy()
    out = out.drop_duplicates(
        subset=["industry_standard", "asset", "l3_code", "in_date", "out_date"],
        keep="last",
    )
    out = out.sort_values(
        ["asset", "in_date", "l1_code", "l2_code", "l3_code"], kind="mergesort"
    ).reset_index(drop=True)
    return out[
        [
            "industry_standard",
            "asset",
            "in_date",
            "out_date",
            "is_new",
            "l1_code",
            "l1_name",
            "l2_code",
            "l2_name",
            "l3_code",
            "l3_name",
        ]
    ].copy()


def _build_liquidity_profile(
    *,
    daily_bars: pd.DataFrame,
    asset_status: pd.DataFrame,
) -> pd.DataFrame:
    if daily_bars.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "asset",
                "amount",
                "avg_amount_20d",
                "amount_rank",
                "amount_percentile",
                "liquidity_tier",
                "is_tradable",
                "can_buy",
                "can_sell",
            ]
        )
    frame = daily_bars.copy()
    frame["date_ts"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["amount"] = pd.to_numeric(frame["amount"], errors="coerce")
    frame = frame.sort_values(["asset", "date_ts", "date"], kind="mergesort").reset_index(drop=True)
    frame["avg_amount_20d"] = (
        frame.groupby("asset", sort=False)["amount"]
        .rolling(window=20, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    frame["amount_rank"] = frame.groupby("date", sort=False)["avg_amount_20d"].rank(
        method="first", ascending=False, na_option="bottom"
    )
    frame["amount_percentile"] = frame.groupby("date", sort=False)["avg_amount_20d"].rank(
        method="first", pct=True, na_option="bottom"
    )
    frame["liquidity_tier"] = (frame["amount_percentile"].fillna(0) * 5).apply(
        lambda value: max(1, min(5, int(value - 1e-12) + 1))
    )
    status = asset_status.copy()
    if status.empty:
        frame["is_suspended"] = 0
        frame["is_st"] = 0
    else:
        status["is_suspended"] = (
            pd.to_numeric(status.get("is_suspended"), errors="coerce").fillna(0).astype(int)
        )
        status["is_st"] = pd.to_numeric(status.get("is_st"), errors="coerce").fillna(0).astype(int)
        frame = frame.merge(
            status[["date", "asset", "is_suspended", "is_st"]],
            on=["date", "asset"],
            how="left",
            validate="one_to_one",
        )
        frame["is_suspended"] = frame["is_suspended"].fillna(0).astype(int)
        frame["is_st"] = frame["is_st"].fillna(0).astype(int)
    frame["is_limit_up"] = (
        pd.to_numeric(frame.get("is_limit_up"), errors="coerce").fillna(0).astype(int)
    )
    frame["is_limit_down"] = (
        pd.to_numeric(frame.get("is_limit_down"), errors="coerce").fillna(0).astype(int)
    )
    frame["is_tradable"] = (
        (frame["is_suspended"] == 0) & frame["amount"].notna() & (frame["amount"] > 0)
    ).astype(int)
    frame["can_buy"] = (
        (frame["is_tradable"] == 1) & (frame["is_st"] == 0) & (frame["is_limit_up"] == 0)
    ).astype(int)
    frame["can_sell"] = (
        (frame["is_tradable"] == 1) & (frame["is_st"] == 0) & (frame["is_limit_down"] == 0)
    ).astype(int)
    frame["date"] = frame["date_ts"].dt.strftime("%Y-%m-%d")
    frame = frame.sort_values(["date", "asset"], kind="mergesort").reset_index(drop=True)
    return frame[
        [
            "date",
            "asset",
            "amount",
            "avg_amount_20d",
            "amount_rank",
            "amount_percentile",
            "liquidity_tier",
            "is_tradable",
            "can_buy",
            "can_sell",
        ]
    ].copy()


def _fetch_instruments_optional(pro: Any, *, assets: tuple[str, ...]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    statuses = ("L", "D", "P")
    for status in statuses:
        try:
            frame = pro.stock_basic(
                exchange="",
                list_status=status,
                fields="ts_code,symbol,name,area,industry,market,list_date,delist_date",
            )
        except (RuntimeError, OSError, ValueError, KeyError, AttributeError) as exc:
            import logging

            logging.getLogger(__name__).warning("stock_basic API call failed: %s", exc)
            return pd.DataFrame(
                columns=[
                    "asset",
                    "symbol",
                    "name",
                    "area",
                    "industry",
                    "market",
                    "list_date",
                    "delist_date",
                ]
            )
        if frame is None or frame.empty:
            continue
        frames.append(frame)
    if not frames:
        return pd.DataFrame(
            columns=[
                "asset",
                "symbol",
                "name",
                "area",
                "industry",
                "market",
                "list_date",
                "delist_date",
            ]
        )

    merged = pd.concat(frames, ignore_index=True)
    if assets:
        merged = merged[merged["ts_code"].astype(str).isin(set(assets))].copy()
    if merged.empty:
        return pd.DataFrame(
            columns=[
                "asset",
                "symbol",
                "name",
                "area",
                "industry",
                "market",
                "list_date",
                "delist_date",
            ]
        )
    merged = merged.rename(columns={"ts_code": "asset"})
    merged["list_date"] = pd.to_datetime(merged["list_date"], errors="coerce")
    merged["delist_date"] = pd.to_datetime(merged["delist_date"], errors="coerce")
    merged = merged[merged["list_date"].notna()].copy()
    merged["list_date"] = merged["list_date"].dt.strftime("%Y-%m-%d")
    merged["delist_date"] = merged["delist_date"].dt.strftime("%Y-%m-%d")
    merged = merged.drop_duplicates(subset=["asset"], keep="last")
    merged = merged.sort_values("asset", kind="mergesort").reset_index(drop=True)
    return merged[
        ["asset", "symbol", "name", "area", "industry", "market", "list_date", "delist_date"]
    ]
