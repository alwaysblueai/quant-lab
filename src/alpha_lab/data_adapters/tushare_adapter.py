from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from alpha_lab.exceptions import (
    AlphaLabConfigError,
    AlphaLabDataError,
    AlphaLabExperimentError,
)

logger = logging.getLogger(__name__)

_PER_ASSET_FETCH_MAX_WORKERS = 2
_PER_ASSET_FETCH_RETRIES = 5
_PER_ASSET_RETRY_BACKOFF_SECONDS = 1.0
_TUSHARE_RATE_LIMIT_WAIT_SECONDS = 65.0

DEFAULT_INDEX_CODES: dict[str, str] = {
    "hs300": "000300.SH",
    "zz500": "000905.SH",
    "zz1000": "000852.SH",
    "sz50": "000016.SH",
}


@dataclass(frozen=True)
class FundamentalFetchResult:
    """Raw fundamentals fetched from Tushare before factor construction."""

    pb_raw: pd.DataFrame
    roe_raw: pd.DataFrame
    balance_sheet_raw: pd.DataFrame
    income_statement_raw: pd.DataFrame
    cash_flow_statement_raw: pd.DataFrame
    dedup_counts: dict[str, int]
    roe_source_column: str


_DAILY_BASIC_FIELD_MAP: dict[str, str] = {
    "pb": "pb",
    "pe": "pe",
    "pe_ttm": "pe_ttm",
    "ps": "ps",
    "ps_ttm": "ps_ttm",
    "dv_ttm": "dv_ttm",
    "total_mv": "total_mv",
    "circ_mv": "circ_mv",
    "turnover_rate": "turnover_rate",
    "turnover_rate_f": "turnover_rate_f",
    "volume_ratio": "volume_ratio",
}


_BALANCE_SHEET_FIELD_MAP: dict[str, str] = {
    "goodwill": "goodwill_balance",
    "st_borr": "short_term_borrow",
    "lt_borr": "long_term_borrow",
    "bond_payable": "bonds_payable",
    "money_cap": "monetary_capital",
    "trad_asset": "tradable_fin_assets",
    "invest_real_estate": "invest_real_estate",
    "deriv_assets": "derivative_fin_assets",
    "div_receiv": "dividend_receivable",
    "int_receiv": "interest_receivable",
    "fa_avail_for_sale": "fin_assets_avail_for_sale",
    "htm_invest": "held_to_mty_invest",
    "oth_debt_invest": "other_debt_investment",
    "oth_eqt_invest": "other_equity_investment",
    "debt_invest": "debt_investment",
}

_INCOME_STATEMENT_FIELD_MAP: dict[str, str] = {
    "total_revenue": "operating_revenue_ttm",
    "oper_cost": "operating_cost_ttm",
    "rd_exp": "rd_expense",
    "sell_exp": "selling_expense",
    "admin_exp": "admin_expense",
}

_CASH_FLOW_STATEMENT_FIELD_MAP: dict[str, str] = {
    "n_cashflow_act": "operating_cash_flow_ttm",
}


@dataclass(frozen=True)
class GeneratedRealCaseInputs:
    """Output summary for generated canonical real-case CSV inputs."""

    output_dir: Path
    output_paths: dict[str, Path]
    row_counts: dict[str, int]
    dedup_counts: dict[str, int]
    roe_rows_using_end_date_fallback: int
    roe_source_column: str
    dataset_version_id: str | None = None
    data_root: Path | None = None


def fetch_prices(
    pro: Any,
    *,
    start_date: str,
    end_date: str,
    assets: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, int]:
    """Fetch and normalize daily OHLCV prices to canonical schema.

    Returns `(prices_df, duplicate_raw_row_count)`.
    """
    start_ymd = _to_tushare_date(start_date)
    end_ymd = _to_tushare_date(end_date)
    trading_dates = _list_trading_dates(pro, start_ymd=start_ymd, end_ymd=end_ymd)

    frames: list[pd.DataFrame] = []
    for trade_date in trading_dates:
        daily = pro.daily(
            trade_date=trade_date,
            fields="ts_code,trade_date,open,high,low,close,pre_close,vol,amount",
        )
        if daily is None or daily.empty:
            continue
        frames.append(daily)

    if not frames:
        raise AlphaLabDataError(
            f"Tushare daily returned no price rows for {start_date} to {end_date}"
        )

    raw = pd.concat(frames, ignore_index=True)
    if assets is not None:
        allowed = set(assets)
        raw = raw[raw["ts_code"].isin(allowed)].copy()
        if raw.empty:
            raise AlphaLabDataError("No price rows left after applying asset filter.")

    prices = pd.DataFrame(
        {
            "date": _parse_tushare_dates(raw["trade_date"]),
            "asset": raw["ts_code"].astype(str).str.strip(),
            "open": pd.to_numeric(raw.get("open"), errors="coerce"),
            "high": pd.to_numeric(raw.get("high"), errors="coerce"),
            "low": pd.to_numeric(raw.get("low"), errors="coerce"),
            "close": pd.to_numeric(raw["close"], errors="coerce"),
            "pre_close": pd.to_numeric(raw.get("pre_close"), errors="coerce"),
            "volume": pd.to_numeric(raw.get("vol"), errors="coerce"),
            "amount": pd.to_numeric(raw.get("amount"), errors="coerce"),
        }
    )
    prices = prices.dropna(subset=["date", "asset", "close"]).copy()
    prices = prices[prices["asset"] != ""].copy()

    prices, n_dup = _deduplicate_rows(
        prices,
        key_cols=["date", "asset"],
        table_name="prices",
    )
    prices = prices[prices["close"] > 0].copy()
    prices["date"] = prices["date"].dt.strftime("%Y-%m-%d")
    prices = prices[
        ["date", "asset", "open", "high", "low", "close", "pre_close", "volume", "amount"]
    ]
    prices = prices.sort_values(["date", "asset"], kind="mergesort").reset_index(drop=True)

    if prices.empty:
        raise AlphaLabDataError("prices became empty after normalization.")
    return prices, n_dup


def fetch_fundamentals(
    pro: Any,
    *,
    assets: Sequence[str],
    start_date: str,
    end_date: str,
    include_daily_basic: bool = True,
    include_roe: bool = True,
    token: str | None = None,
) -> FundamentalFetchResult:
    """Fetch raw daily_basic and/or accounting fundamentals from Tushare."""
    if not assets:
        raise AlphaLabDataError("assets must be non-empty for fundamentals fetch.")

    start_ymd = _to_tushare_date(start_date)
    end_ymd = _to_tushare_date(end_date)
    allowed_assets = set(assets)

    if include_daily_basic:
        trading_dates = _list_trading_dates(pro, start_ymd=start_ymd, end_ymd=end_ymd)
        pb_frames: list[pd.DataFrame] = []
        for trade_date in trading_dates:
            daily_basic = pro.daily_basic(
                trade_date=trade_date,
                fields="ts_code,trade_date," + ",".join(_DAILY_BASIC_FIELD_MAP.keys()),
            )
            if daily_basic is None or daily_basic.empty:
                continue
            daily_basic = daily_basic[daily_basic["ts_code"].isin(allowed_assets)].copy()
            if daily_basic.empty:
                continue
            pb_frames.append(daily_basic)

        if not pb_frames:
            raise AlphaLabDataError("No PB rows returned from Tushare daily_basic.")

        pb_raw = pd.concat(pb_frames, ignore_index=True)
        payload: dict[str, pd.Series] = {
            "date": _parse_tushare_dates(pb_raw["trade_date"]),
            "asset": pb_raw["ts_code"].astype(str).str.strip(),
        }
        for source_col, target_col in _DAILY_BASIC_FIELD_MAP.items():
            payload[target_col] = pd.to_numeric(pb_raw.get(source_col), errors="coerce")
        pb_raw = pd.DataFrame(payload)
        pb_raw = pb_raw.dropna(subset=["date", "asset"]).copy()
        pb_raw = pb_raw[pb_raw["asset"] != ""].copy()
        pb_raw, pb_dup_count = _deduplicate_rows(
            pb_raw,
            key_cols=["date", "asset"],
            table_name="pb_raw",
        )
        pb_raw["date"] = pb_raw["date"].dt.strftime("%Y-%m-%d")
        pb_raw = pb_raw[["date", "asset", *_DAILY_BASIC_FIELD_MAP.values()]]
        pb_raw = pb_raw.sort_values(["date", "asset"], kind="mergesort").reset_index(drop=True)
    else:
        pb_dup_count = 0
        pb_raw = pd.DataFrame(columns=["date", "asset", *_DAILY_BASIC_FIELD_MAP.values()])

    if not include_roe:
        return FundamentalFetchResult(
            pb_raw=pb_raw,
            roe_raw=pd.DataFrame(columns=["asset", "ann_date", "end_date", "roe_value"]),
            balance_sheet_raw=pd.DataFrame(columns=["asset", "ann_date", "end_date"]),
            income_statement_raw=pd.DataFrame(columns=["asset", "ann_date", "end_date"]),
            cash_flow_statement_raw=pd.DataFrame(columns=["asset", "ann_date", "end_date"]),
            dedup_counts={"pb_raw": pb_dup_count, "roe_raw": 0},
            roe_source_column="skipped_daily_research_only",
        )

    sample_asset = next(iter(sorted(allowed_assets)))
    roe_source_col = _resolve_roe_source_column(
        pro,
        sample_asset=sample_asset,
        start_ymd=start_ymd,
        end_ymd=end_ymd,
    )
    roe_fields = f"ts_code,ann_date,end_date,{roe_source_col}"

    roe_frames, failed_assets = _fetch_per_asset_endpoint_frames(
        pro,
        api_name="fina_indicator",
        assets=sorted(allowed_assets),
        request_builder=lambda asset: {
            "ts_code": asset,
            "start_date": start_ymd,
            "end_date": end_ymd,
            "fields": roe_fields,
        },
        token=token,
    )

    if failed_assets:
        logger.warning(
            "ROE fetch failed for %d assets; proceeding with available rows.",
            failed_assets,
        )
    if not roe_frames:
        raise AlphaLabDataError("No ROE rows returned from Tushare fina_indicator.")

    roe_raw = pd.concat(roe_frames, ignore_index=True)
    ann_date = _parse_tushare_dates(roe_raw["ann_date"])
    end_date_series = _parse_tushare_dates(roe_raw["end_date"])
    roe_raw = pd.DataFrame(
        {
            "asset": roe_raw["ts_code"].astype(str).str.strip(),
            "ann_date": ann_date,
            "end_date": end_date_series,
            "roe_value": pd.to_numeric(roe_raw[roe_source_col], errors="coerce"),
        }
    )
    roe_raw = roe_raw.dropna(subset=["asset"]).copy()
    roe_raw = roe_raw[roe_raw["asset"] != ""].copy()
    roe_raw, roe_dup_count = _deduplicate_rows(
        roe_raw,
        key_cols=["asset", "ann_date", "end_date"],
        table_name="roe_raw",
    )
    roe_raw["ann_date"] = roe_raw["ann_date"].dt.strftime("%Y-%m-%d")
    roe_raw["end_date"] = roe_raw["end_date"].dt.strftime("%Y-%m-%d")
    roe_raw = roe_raw[["asset", "ann_date", "end_date", "roe_value"]]
    roe_raw = roe_raw.sort_values(["asset", "ann_date", "end_date"], kind="mergesort")
    roe_raw = roe_raw.reset_index(drop=True)

    balance_sheet_raw, balance_sheet_dup_count = _fetch_accounting_statement(
        pro,
        assets=sorted(allowed_assets),
        start_ymd=start_ymd,
        end_ymd=end_ymd,
        api_name="balancesheet",
        field_map=_BALANCE_SHEET_FIELD_MAP,
        token=token,
    )
    income_statement_raw, income_statement_dup_count = _fetch_accounting_statement(
        pro,
        assets=sorted(allowed_assets),
        start_ymd=start_ymd,
        end_ymd=end_ymd,
        api_name="income",
        field_map=_INCOME_STATEMENT_FIELD_MAP,
        token=token,
    )
    cash_flow_statement_raw, cash_flow_statement_dup_count = _fetch_accounting_statement(
        pro,
        assets=sorted(allowed_assets),
        start_ymd=start_ymd,
        end_ymd=end_ymd,
        api_name="cashflow",
        field_map=_CASH_FLOW_STATEMENT_FIELD_MAP,
        token=token,
    )

    return FundamentalFetchResult(
        pb_raw=pb_raw,
        roe_raw=roe_raw,
        balance_sheet_raw=balance_sheet_raw,
        income_statement_raw=income_statement_raw,
        cash_flow_statement_raw=cash_flow_statement_raw,
        dedup_counts={
            "pb_raw": pb_dup_count,
            "roe_raw": roe_dup_count,
            "balance_sheet_raw": balance_sheet_dup_count,
            "income_statement_raw": income_statement_dup_count,
            "cash_flow_statement_raw": cash_flow_statement_dup_count,
        },
        roe_source_column=roe_source_col,
    )


def _fetch_accounting_statement(
    pro: Any,
    *,
    assets: Sequence[str],
    start_ymd: str,
    end_ymd: str,
    api_name: str,
    field_map: dict[str, str],
    token: str | None = None,
) -> tuple[pd.DataFrame, int]:
    if not assets:
        return pd.DataFrame(columns=["asset", "ann_date", "end_date"]), 0

    endpoint = getattr(pro, api_name, None)
    if endpoint is None:
        logger.warning("Skipping %s fetch because endpoint is unavailable on client.", api_name)
        return pd.DataFrame(columns=["asset", "ann_date", "end_date"]), 0

    fields = ",".join(["ts_code", "ann_date", "end_date", *field_map.keys()])
    frames, failed_assets = _fetch_per_asset_endpoint_frames(
        pro,
        api_name=api_name,
        assets=assets,
        request_builder=lambda asset: {
            "ts_code": asset,
            "start_date": start_ymd,
            "end_date": end_ymd,
            "fields": fields,
        },
        token=token,
    )

    if failed_assets:
        logger.warning(
            "%s fetch failed for %d assets; proceeding with available rows.",
            api_name,
            failed_assets,
        )
    if not frames:
        return pd.DataFrame(columns=["asset", "ann_date", "end_date", *field_map.values()]), 0

    raw = pd.concat(frames, ignore_index=True)
    payload: dict[str, pd.Series] = {
        "asset": raw["ts_code"].astype(str).str.strip(),
        "ann_date": _parse_tushare_dates(raw["ann_date"]),
        "end_date": _parse_tushare_dates(raw["end_date"]),
    }
    for source_col, target_col in field_map.items():
        payload[target_col] = pd.to_numeric(raw.get(source_col), errors="coerce")
    out = pd.DataFrame(payload)
    out = out.dropna(subset=["asset"]).copy()
    out = out[out["asset"] != ""].copy()
    out, dup_count = _deduplicate_rows(
        out,
        key_cols=["asset", "ann_date", "end_date"],
        table_name=f"{api_name}_raw",
    )
    out["ann_date"] = pd.to_datetime(out["ann_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out["end_date"] = pd.to_datetime(out["end_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out = out.sort_values(["asset", "ann_date", "end_date"], kind="mergesort").reset_index(
        drop=True
    )
    return out, dup_count


def _fetch_per_asset_endpoint_frames(
    pro: Any,
    *,
    api_name: str,
    assets: Sequence[str],
    request_builder: Callable[[str], dict[str, object]],
    token: str | None = None,
    max_workers: int = _PER_ASSET_FETCH_MAX_WORKERS,
    retries: int = _PER_ASSET_FETCH_RETRIES,
) -> tuple[list[pd.DataFrame], int]:
    if not assets:
        return [], 0

    def _call_endpoint(client: Any, asset: str) -> pd.DataFrame | None:
        endpoint = getattr(client, api_name, None)
        if endpoint is None:
            raise AttributeError(f"Tushare client has no endpoint {api_name!r}")
        last_exc: Exception | None = None
        for attempt in range(retries):
            try:
                return endpoint(**request_builder(asset))
            except Exception as exc:  # pragma: no cover - API-dependent path
                last_exc = exc
                if _is_tushare_rate_limit_error(exc):
                    time.sleep(_TUSHARE_RATE_LIMIT_WAIT_SECONDS)
                    continue
                if attempt + 1 >= retries:
                    raise
                time.sleep(_PER_ASSET_RETRY_BACKOFF_SECONDS * (attempt + 1))
        if last_exc is not None:  # pragma: no cover - defensive
            raise last_exc
        return None

    frames: list[pd.DataFrame] = []
    failed_assets = 0
    ordered_assets = list(assets)

    if token is None or len(ordered_assets) == 1:
        for asset in ordered_assets:
            try:
                frame = _call_endpoint(pro, asset)
            except Exception as exc:  # pragma: no cover - API-dependent path
                failed_assets += 1
                logger.warning(
                    "Skipping %s fetch for %s due to Tushare error: %s", api_name, asset, exc
                )
                continue
            if frame is None or frame.empty:
                continue
            frames.append(frame)
        return frames, failed_assets

    resolved_token = _require_tushare_token(token)

    def _worker(asset: str) -> tuple[str, pd.DataFrame | None, Exception | None]:
        try:
            client = _build_tushare_client(resolved_token)
            frame = _call_endpoint(client, asset)
            return asset, frame, None
        except Exception as exc:  # pragma: no cover - API-dependent path
            return asset, None, exc

    with ThreadPoolExecutor(max_workers=min(max_workers, len(ordered_assets))) as pool:
        futures = {pool.submit(_worker, asset): asset for asset in ordered_assets}
        for future in as_completed(futures):
            asset, frame, error = future.result()
            if error is not None:
                failed_assets += 1
                logger.warning(
                    "Skipping %s fetch for %s due to Tushare error: %s", api_name, asset, error
                )
                continue
            if frame is None or frame.empty:
                continue
            frames.append(frame)
    return frames, failed_assets


def _is_tushare_rate_limit_error(exc: Exception) -> bool:
    message = str(exc)
    return "每分钟最多访问该接口" in message


def fetch_adj_factor(
    pro: Any,
    *,
    start_date: str,
    end_date: str,
    assets: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, int]:
    """Fetch and normalize Tushare复权因子 to canonical schema."""
    start_ymd = _to_tushare_date(start_date)
    end_ymd = _to_tushare_date(end_date)
    trading_dates = _list_trading_dates(pro, start_ymd=start_ymd, end_ymd=end_ymd)

    frames: list[pd.DataFrame] = []
    for trade_date in trading_dates:
        frame = pro.adj_factor(
            trade_date=trade_date,
            fields="ts_code,trade_date,adj_factor",
        )
        if frame is None or frame.empty:
            continue
        frames.append(frame)

    if not frames:
        raise AlphaLabDataError(
            f"Tushare adj_factor returned no rows for {start_date} to {end_date}"
        )

    raw = pd.concat(frames, ignore_index=True)
    if assets is not None:
        allowed = set(assets)
        raw = raw[raw["ts_code"].isin(allowed)].copy()
        if raw.empty:
            raise AlphaLabDataError("No adj_factor rows left after applying asset filter.")

    adj_factor = pd.DataFrame(
        {
            "date": _parse_tushare_dates(raw["trade_date"]),
            "asset": raw["ts_code"].astype(str).str.strip(),
            "adj_factor": pd.to_numeric(raw["adj_factor"], errors="coerce"),
        }
    )
    adj_factor = adj_factor.dropna(subset=["date", "asset", "adj_factor"]).copy()
    adj_factor = adj_factor[adj_factor["asset"] != ""].copy()
    adj_factor, n_dup = _deduplicate_rows(
        adj_factor,
        key_cols=["date", "asset"],
        table_name="adj_factor_raw",
    )
    adj_factor = adj_factor[adj_factor["adj_factor"] > 0].copy()
    adj_factor["date"] = adj_factor["date"].dt.strftime("%Y-%m-%d")
    adj_factor = adj_factor[["date", "asset", "adj_factor"]]
    adj_factor = adj_factor.sort_values(["date", "asset"], kind="mergesort").reset_index(drop=True)

    if adj_factor.empty:
        raise AlphaLabDataError("adj_factor became empty after normalization.")
    return adj_factor, n_dup


def fetch_stk_limit(
    pro: Any,
    *,
    start_date: str,
    end_date: str,
    assets: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, int]:
    """Fetch daily涨跌停价格 to canonical schema."""
    start_ymd = _to_tushare_date(start_date)
    end_ymd = _to_tushare_date(end_date)
    trading_dates = _list_trading_dates(pro, start_ymd=start_ymd, end_ymd=end_ymd)

    frames: list[pd.DataFrame] = []
    for trade_date in trading_dates:
        frame = pro.stk_limit(
            trade_date=trade_date,
            fields="ts_code,trade_date,up_limit,down_limit",
        )
        if frame is None or frame.empty:
            continue
        frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=["date", "asset", "up_limit", "down_limit"]), 0

    raw = pd.concat(frames, ignore_index=True)
    if assets is not None:
        allowed = set(assets)
        raw = raw[raw["ts_code"].isin(allowed)].copy()
    if raw.empty:
        return pd.DataFrame(columns=["date", "asset", "up_limit", "down_limit"]), 0

    limits = pd.DataFrame(
        {
            "date": _parse_tushare_dates(raw["trade_date"]),
            "asset": raw["ts_code"].astype(str).str.strip(),
            "up_limit": pd.to_numeric(raw.get("up_limit"), errors="coerce"),
            "down_limit": pd.to_numeric(raw.get("down_limit"), errors="coerce"),
        }
    )
    limits = limits.dropna(subset=["date", "asset"]).copy()
    limits = limits[limits["asset"] != ""].copy()
    limits, n_dup = _deduplicate_rows(
        limits,
        key_cols=["date", "asset"],
        table_name="stk_limit_raw",
    )
    limits["date"] = limits["date"].dt.strftime("%Y-%m-%d")
    limits = limits[["date", "asset", "up_limit", "down_limit"]]
    limits = limits.sort_values(["date", "asset"], kind="mergesort").reset_index(drop=True)
    return limits, n_dup


def fetch_suspend_status(
    pro: Any,
    *,
    start_date: str,
    end_date: str,
    assets: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, int]:
    """Fetch daily suspension flags from `suspend_d`.

    The official endpoint returns one row for suspension/resume events on a
    trade date. We keep only suspension rows and later left-join them to the
    research panel.
    """
    start_ymd = _to_tushare_date(start_date)
    end_ymd = _to_tushare_date(end_date)
    frame = pro.suspend_d(
        start_date=start_ymd,
        end_date=end_ymd,
        suspend_type="S",
        fields="ts_code,trade_date,suspend_type",
    )
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["date", "asset", "is_suspended"]), 0
    if assets is not None:
        allowed = set(assets)
        frame = frame[frame["ts_code"].isin(allowed)].copy()
    if frame.empty:
        return pd.DataFrame(columns=["date", "asset", "is_suspended"]), 0

    status = pd.DataFrame(
        {
            "date": _parse_tushare_dates(frame["trade_date"]),
            "asset": frame["ts_code"].astype(str).str.strip(),
            "is_suspended": 1,
        }
    )
    status = status.dropna(subset=["date", "asset"]).copy()
    status = status[status["asset"] != ""].copy()
    status, n_dup = _deduplicate_rows(
        status,
        key_cols=["date", "asset"],
        table_name="suspend_raw",
    )
    status["date"] = status["date"].dt.strftime("%Y-%m-%d")
    status = status[["date", "asset", "is_suspended"]]
    status = status.sort_values(["date", "asset"], kind="mergesort").reset_index(drop=True)
    return status, n_dup


def fetch_st_name_events(
    pro: Any,
    *,
    start_date: str,
    end_date: str,
    assets: Sequence[str],
) -> tuple[pd.DataFrame, int]:
    """Fetch historical ST name intervals from `namechange`."""
    if not assets:
        return pd.DataFrame(columns=["asset", "start_date", "end_date", "name", "is_st"]), 0
    start_ymd = _to_tushare_date(start_date)
    end_ymd = _to_tushare_date(end_date)
    frames: list[pd.DataFrame] = []
    failed_assets = 0
    for asset in sorted(set(assets)):
        try:
            frame = pro.namechange(
                ts_code=asset,
                start_date=start_ymd,
                end_date=end_ymd,
                fields="ts_code,name,start_date,end_date,ann_date,change_reason",
            )
        except Exception as exc:  # pragma: no cover - API-dependent path
            failed_assets += 1
            logger.warning("Skipping namechange fetch for %s due to Tushare error: %s", asset, exc)
            continue
        if frame is None or frame.empty:
            continue
        frames.append(frame)
    if failed_assets:
        logger.warning(
            "ST namechange fetch failed for %d assets; proceeding with available rows.",
            failed_assets,
        )
    if not frames:
        return pd.DataFrame(columns=["asset", "start_date", "end_date", "name", "is_st"]), 0

    raw = pd.concat(frames, ignore_index=True)
    events = pd.DataFrame(
        {
            "asset": raw["ts_code"].astype(str).str.strip(),
            "start_date": _parse_tushare_dates(raw["start_date"]),
            "end_date": _parse_tushare_dates(raw["end_date"]),
            "name": raw.get("name", pd.Series(index=raw.index, dtype="object"))
            .astype(str)
            .str.strip(),
        }
    )
    events["is_st"] = events["name"].str.upper().str.contains("ST", na=False).astype(int)
    events = events.dropna(subset=["asset", "start_date"]).copy()
    events = events[events["asset"] != ""].copy()
    events, n_dup = _deduplicate_rows(
        events,
        key_cols=["asset", "start_date", "end_date", "name"],
        table_name="st_namechange_raw",
    )
    events["start_date"] = events["start_date"].dt.strftime("%Y-%m-%d")
    events["end_date"] = events["end_date"].dt.strftime("%Y-%m-%d")
    events = events[["asset", "start_date", "end_date", "name", "is_st"]]
    events = events.sort_values(["asset", "start_date", "end_date"], kind="mergesort").reset_index(
        drop=True
    )
    return events, n_dup


def fetch_index_membership(
    pro: Any,
    *,
    start_date: str,
    end_date: str,
    assets: Sequence[str] | None = None,
    index_codes: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, int]:
    """Fetch monthly index constituent snapshots from `index_weight`."""
    start_ymd = _to_tushare_date(start_date)
    end_ymd = _to_tushare_date(end_date)
    code_map = dict(index_codes or DEFAULT_INDEX_CODES)
    frames: list[pd.DataFrame] = []

    for index_name, index_code in code_map.items():
        try:
            frame = pro.index_weight(
                index_code=index_code,
                start_date=start_ymd,
                end_date=end_ymd,
                fields="index_code,con_code,trade_date,weight",
            )
        except Exception as exc:  # pragma: no cover - API-dependent path
            logger.warning(
                "Skipping index_weight fetch for %s due to Tushare error: %s", index_code, exc
            )
            continue
        if frame is None or frame.empty:
            continue
        frame = frame.copy()
        frame["index_name"] = index_name
        frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=["date", "index_code", "index_name", "asset", "weight"]), 0

    raw = pd.concat(frames, ignore_index=True)
    if assets is not None:
        allowed = set(assets)
        raw = raw[raw["con_code"].isin(allowed)].copy()
    if raw.empty:
        return pd.DataFrame(columns=["date", "index_code", "index_name", "asset", "weight"]), 0

    members = pd.DataFrame(
        {
            "date": _parse_tushare_dates(raw["trade_date"]),
            "index_code": raw["index_code"].astype(str).str.strip(),
            "index_name": raw["index_name"].astype(str).str.strip(),
            "asset": raw["con_code"].astype(str).str.strip(),
            "weight": pd.to_numeric(raw.get("weight"), errors="coerce"),
        }
    )
    members = members.dropna(subset=["date", "index_code", "asset"]).copy()
    members = members[(members["index_code"] != "") & (members["asset"] != "")].copy()
    members, n_dup = _deduplicate_rows(
        members,
        key_cols=["date", "index_code", "asset"],
        table_name="index_membership_raw",
    )
    members["date"] = members["date"].dt.strftime("%Y-%m-%d")
    members = members[["date", "index_code", "index_name", "asset", "weight"]]
    members = members.sort_values(["date", "index_code", "asset"], kind="mergesort").reset_index(
        drop=True
    )
    return members, n_dup


def fetch_moneyflow(
    pro: Any,
    *,
    start_date: str,
    end_date: str,
    assets: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, int]:
    """Fetch daily stock moneyflow rows to canonical schema."""
    start_ymd = _to_tushare_date(start_date)
    end_ymd = _to_tushare_date(end_date)
    trading_dates = _list_trading_dates(pro, start_ymd=start_ymd, end_ymd=end_ymd)

    frames: list[pd.DataFrame] = []
    fields = (
        "ts_code,trade_date,"
        "buy_sm_vol,buy_sm_amount,sell_sm_vol,sell_sm_amount,"
        "buy_md_vol,buy_md_amount,sell_md_vol,sell_md_amount,"
        "buy_lg_vol,buy_lg_amount,sell_lg_vol,sell_lg_amount,"
        "buy_elg_vol,buy_elg_amount,sell_elg_vol,sell_elg_amount,"
        "net_mf_vol,net_mf_amount"
    )
    for trade_date in trading_dates:
        frame = pro.moneyflow(trade_date=trade_date, fields=fields)
        if frame is None or frame.empty:
            continue
        frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=["date", "asset", "net_mf_amount"]), 0

    raw = pd.concat(frames, ignore_index=True)
    if assets is not None:
        allowed = set(assets)
        raw = raw[raw["ts_code"].isin(allowed)].copy()
    if raw.empty:
        return pd.DataFrame(columns=["date", "asset", "net_mf_amount"]), 0

    moneyflow = pd.DataFrame(
        {
            "date": _parse_tushare_dates(raw["trade_date"]),
            "asset": raw["ts_code"].astype(str).str.strip(),
            "buy_sm_vol": pd.to_numeric(raw.get("buy_sm_vol"), errors="coerce"),
            "buy_sm_amount": pd.to_numeric(raw.get("buy_sm_amount"), errors="coerce"),
            "sell_sm_vol": pd.to_numeric(raw.get("sell_sm_vol"), errors="coerce"),
            "sell_sm_amount": pd.to_numeric(raw.get("sell_sm_amount"), errors="coerce"),
            "buy_md_vol": pd.to_numeric(raw.get("buy_md_vol"), errors="coerce"),
            "buy_md_amount": pd.to_numeric(raw.get("buy_md_amount"), errors="coerce"),
            "sell_md_vol": pd.to_numeric(raw.get("sell_md_vol"), errors="coerce"),
            "sell_md_amount": pd.to_numeric(raw.get("sell_md_amount"), errors="coerce"),
            "buy_lg_vol": pd.to_numeric(raw.get("buy_lg_vol"), errors="coerce"),
            "buy_lg_amount": pd.to_numeric(raw.get("buy_lg_amount"), errors="coerce"),
            "sell_lg_vol": pd.to_numeric(raw.get("sell_lg_vol"), errors="coerce"),
            "sell_lg_amount": pd.to_numeric(raw.get("sell_lg_amount"), errors="coerce"),
            "buy_elg_vol": pd.to_numeric(raw.get("buy_elg_vol"), errors="coerce"),
            "buy_elg_amount": pd.to_numeric(raw.get("buy_elg_amount"), errors="coerce"),
            "sell_elg_vol": pd.to_numeric(raw.get("sell_elg_vol"), errors="coerce"),
            "sell_elg_amount": pd.to_numeric(raw.get("sell_elg_amount"), errors="coerce"),
            "net_mf_vol": pd.to_numeric(raw.get("net_mf_vol"), errors="coerce"),
            "net_mf_amount": pd.to_numeric(raw.get("net_mf_amount"), errors="coerce"),
        }
    )
    moneyflow = moneyflow.dropna(subset=["date", "asset"]).copy()
    moneyflow = moneyflow[moneyflow["asset"] != ""].copy()
    moneyflow, n_dup = _deduplicate_rows(
        moneyflow,
        key_cols=["date", "asset"],
        table_name="moneyflow_raw",
    )
    moneyflow["date"] = moneyflow["date"].dt.strftime("%Y-%m-%d")
    moneyflow = moneyflow.sort_values(["date", "asset"], kind="mergesort").reset_index(drop=True)
    return moneyflow, n_dup


def fetch_industry_classification(
    pro: Any,
    *,
    snapshot_date: str,
    src: str = "SW2021",
) -> tuple[pd.DataFrame, int]:
    """Fetch Shenwan industry classification dimensions."""
    frames: list[pd.DataFrame] = []
    for level in ("L1", "L2", "L3"):
        frame = pro.index_classify(
            level=level,
            src=src,
            fields="index_code,industry_name,parent_code,level,industry_code,is_pub,src",
        )
        if frame is None or frame.empty:
            continue
        frames.append(frame)
    if not frames:
        return pd.DataFrame(
            columns=["snapshot_date", "industry_standard", "index_code", "industry_name"]
        ), 0

    raw = pd.concat(frames, ignore_index=True)
    classification = pd.DataFrame(
        {
            "snapshot_date": pd.to_datetime(snapshot_date, errors="coerce"),
            "industry_standard": raw.get("src", pd.Series(index=raw.index, dtype="object"))
            .astype(str)
            .str.strip(),
            "index_code": raw["index_code"].astype(str).str.strip(),
            "industry_name": raw.get("industry_name", pd.Series(index=raw.index, dtype="object"))
            .astype(str)
            .str.strip(),
            "parent_code": raw.get("parent_code", pd.Series(index=raw.index, dtype="object"))
            .astype(str)
            .str.strip(),
            "level": raw.get("level", pd.Series(index=raw.index, dtype="object"))
            .astype(str)
            .str.strip(),
            "industry_code": raw.get("industry_code", pd.Series(index=raw.index, dtype="object"))
            .astype(str)
            .str.strip(),
            "is_published": pd.to_numeric(raw.get("is_pub"), errors="coerce"),
        }
    )
    classification = classification.dropna(
        subset=["snapshot_date", "industry_standard", "index_code"]
    ).copy()
    classification = classification[
        (classification["industry_standard"] != "") & (classification["index_code"] != "")
    ].copy()
    classification, n_dup = _deduplicate_rows(
        classification,
        key_cols=["snapshot_date", "industry_standard", "index_code"],
        table_name="industry_classification_raw",
    )
    classification["snapshot_date"] = classification["snapshot_date"].dt.strftime("%Y-%m-%d")
    classification = classification.sort_values(
        ["snapshot_date", "industry_standard", "level", "index_code"],
        kind="mergesort",
    ).reset_index(drop=True)
    return classification, n_dup


def fetch_industry_membership(
    pro: Any,
    *,
    classification: pd.DataFrame,
    src: str = "SW2021",
) -> tuple[pd.DataFrame, int]:
    """Fetch Shenwan industry membership history from level-3 industry codes."""
    if classification.empty:
        return pd.DataFrame(columns=["industry_standard", "asset", "in_date", "out_date"]), 0
    l3_codes = sorted(
        classification.loc[classification["level"].astype(str) == "L3", "index_code"]
        .astype(str)
        .dropna()
        .unique()
        .tolist()
    )
    if not l3_codes:
        return pd.DataFrame(columns=["industry_standard", "asset", "in_date", "out_date"]), 0

    class_map = classification.copy()
    class_map = class_map.sort_values(["snapshot_date", "level", "index_code"], kind="mergesort")
    l1_map = class_map[class_map["level"] == "L1"].set_index("index_code")
    l2_map = class_map[class_map["level"] == "L2"].set_index("index_code")
    l3_map = class_map[class_map["level"] == "L3"].set_index("index_code")

    frames: list[pd.DataFrame] = []
    for l3_code in l3_codes:
        try:
            frame = pro.index_member_all(l3_code=l3_code)
        except Exception as exc:  # pragma: no cover - API-dependent path
            logger.warning(
                "Skipping index_member_all fetch for %s due to Tushare error: %s", l3_code, exc
            )
            continue
        if frame is None or frame.empty:
            continue
        frame = frame.copy()
        frame["l3_code"] = l3_code
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["industry_standard", "asset", "in_date", "out_date"]), 0

    raw = pd.concat(frames, ignore_index=True)
    out = pd.DataFrame(
        {
            "industry_standard": src,
            "asset": raw["con_code"].astype(str).str.strip(),
            "in_date": _parse_tushare_dates(
                raw.get("in_date", pd.Series(index=raw.index, dtype="object"))
            ),
            "out_date": _parse_tushare_dates(
                raw.get("out_date", pd.Series(index=raw.index, dtype="object"))
            ),
            "is_new": pd.to_numeric(raw.get("is_new"), errors="coerce"),
            "l3_code": raw["l3_code"].astype(str).str.strip(),
        }
    )
    out["l2_code"] = out["l3_code"].map(l3_map["parent_code"].to_dict())
    out["l1_code"] = out["l2_code"].map(l2_map["parent_code"].to_dict())
    out["l3_name"] = out["l3_code"].map(l3_map["industry_name"].to_dict())
    out["l2_name"] = out["l2_code"].map(l2_map["industry_name"].to_dict())
    out["l1_name"] = out["l1_code"].map(l1_map["industry_name"].to_dict())
    out = out.dropna(subset=["asset", "in_date", "l3_code"]).copy()
    out = out[out["asset"] != ""].copy()
    out, n_dup = _deduplicate_rows(
        out,
        key_cols=["industry_standard", "asset", "l3_code", "in_date", "out_date"],
        table_name="industry_membership_raw",
    )
    out["in_date"] = pd.to_datetime(out["in_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out["out_date"] = pd.to_datetime(out["out_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out = out.sort_values(
        ["asset", "in_date", "l1_code", "l2_code", "l3_code"], kind="mergesort"
    ).reset_index(drop=True)
    return out, n_dup


def build_bp_factor(
    pb_raw: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """Build canonical BP factor from PB raw rows and align to price keys."""
    required = {"date", "asset", "pb"}
    missing = required - set(pb_raw.columns)
    if missing:
        raise AlphaLabDataError(f"pb_raw is missing required columns: {sorted(missing)}")

    price_keys = prices[["date", "asset"]].drop_duplicates().copy()
    frame = pb_raw.copy()
    frame["pb"] = pd.to_numeric(frame["pb"], errors="coerce")
    frame = frame[frame["pb"] > 0].copy()
    frame["value"] = 1.0 / frame["pb"]
    frame["factor"] = "bp"
    factor = frame[["date", "asset", "factor", "value"]].copy()
    factor = factor.merge(
        price_keys,
        on=["date", "asset"],
        how="inner",
        validate="many_to_one",
    )
    factor, _ = _deduplicate_rows(
        factor,
        key_cols=["date", "asset", "factor"],
        table_name="bp_factor",
    )
    factor = factor.sort_values(["date", "asset"], kind="mergesort").reset_index(drop=True)
    if factor.empty:
        raise AlphaLabDataError("bp factor became empty after filtering/alignment.")
    return factor


def build_roe_factor(
    roe_raw: pd.DataFrame,
    prices: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    """Build canonical ROE factor with minimal daily forward-fill alignment.

    v1 limitation:
    - Not a fully PIT-safe fundamentals pipeline.
    - Uses `ann_date` as the effective timestamp and falls back to `end_date`
      when `ann_date` is missing.
    - Does not model restatements or vendor revision history.

    Returns `(factor_df, rows_using_end_date_fallback)`.
    """
    required = {"asset", "ann_date", "end_date", "roe_value"}
    missing = required - set(roe_raw.columns)
    if missing:
        raise AlphaLabDataError(f"roe_raw is missing required columns: {sorted(missing)}")

    events = roe_raw.copy()
    events["ann_date"] = pd.to_datetime(events["ann_date"], errors="coerce")
    events["end_date"] = pd.to_datetime(events["end_date"], errors="coerce")
    events["roe_value"] = pd.to_numeric(events["roe_value"], errors="coerce")
    events["event_date"] = events["ann_date"].where(events["ann_date"].notna(), events["end_date"])
    fallback_count = int((events["ann_date"].isna() & events["end_date"].notna()).sum())
    events = events.dropna(subset=["asset", "event_date", "roe_value"]).copy()

    events = events.sort_values(["asset", "event_date", "end_date"], kind="mergesort")
    events, _ = _deduplicate_rows(
        events,
        key_cols=["asset", "event_date"],
        table_name="roe_events",
    )

    price_keys = prices[["date", "asset"]].drop_duplicates().copy()
    price_keys["date"] = pd.to_datetime(price_keys["date"], errors="coerce")
    price_keys = price_keys.dropna(subset=["date", "asset"]).copy()

    aligned_parts: list[pd.DataFrame] = []
    for asset, px in price_keys.groupby("asset"):
        asset_events = events[events["asset"] == asset][["event_date", "roe_value"]].copy()
        if asset_events.empty:
            continue
        asset_events = asset_events.sort_values("event_date", kind="mergesort")
        px_sorted = px[["date"]].sort_values("date", kind="mergesort")
        aligned = pd.merge_asof(
            px_sorted,
            asset_events,
            left_on="date",
            right_on="event_date",
            direction="backward",
            allow_exact_matches=False,
        )
        aligned["asset"] = asset
        aligned_parts.append(aligned[["date", "asset", "roe_value"]])

    if not aligned_parts:
        raise AlphaLabDataError("No ROE rows could be aligned to price dates.")

    factor = pd.concat(aligned_parts, ignore_index=True)
    factor = factor.dropna(subset=["roe_value"]).copy()
    factor["date"] = factor["date"].dt.strftime("%Y-%m-%d")
    factor["factor"] = "roe_ttm"
    factor = factor.rename(columns={"roe_value": "value"})
    factor = factor[["date", "asset", "factor", "value"]]
    factor, _ = _deduplicate_rows(
        factor,
        key_cols=["date", "asset", "factor"],
        table_name="roe_factor",
    )
    factor = factor.sort_values(["date", "asset"], kind="mergesort").reset_index(drop=True)
    if factor.empty:
        raise AlphaLabDataError("roe_ttm factor became empty after daily alignment.")
    return factor, fallback_count


def build_universe(prices: pd.DataFrame) -> pd.DataFrame:
    """Build a minimal universe mask directly from price keys."""
    required = {"date", "asset"}
    missing = required - set(prices.columns)
    if missing:
        raise AlphaLabDataError(
            f"prices is missing required columns for universe build: {sorted(missing)}"
        )

    universe = prices[["date", "asset"]].drop_duplicates().copy()
    universe["in_universe"] = 1
    universe = universe[["date", "asset", "in_universe"]]
    universe = universe.sort_values(["date", "asset"], kind="mergesort").reset_index(drop=True)
    if universe.empty:
        raise AlphaLabDataError("universe became empty.")
    return universe


def generate_real_case_inputs(
    output_dir: str | Path,
    start_date: str = "2024-01-01",
    end_date: str = "2024-12-31",
    *,
    token: str | None = None,
    assets: Sequence[str] | None = None,
    asset_limit: int | None = None,
) -> GeneratedRealCaseInputs:
    """Generate canonical real-case CSV inputs from the Tushare-backed data store.

    The v2 path ingests raw Tushare responses into the external Parquet-backed
    data root first, then exports a small case slice as CSV so existing Level
    1/2 workflows remain unchanged.
    """
    if asset_limit is not None and asset_limit <= 0:
        raise AlphaLabConfigError("asset_limit must be positive when provided.")
    from alpha_lab.data_store.tushare import TushareIngestor

    ingestor = TushareIngestor()
    ingest_result = ingestor.ingest_core(
        start_date=start_date,
        end_date=end_date,
        token=token,
        assets=tuple(assets) if assets is not None else None,
        asset_limit=asset_limit,
        mode="full",
    )
    export_result = ingestor.export_case_inputs(
        start_date=start_date,
        end_date=end_date,
        output_dir=output_dir,
        assets=tuple(assets) if assets is not None else None,
        asset_limit=asset_limit,
        factors=("bp", "roe_ttm"),
    )

    return GeneratedRealCaseInputs(
        output_dir=export_result.output_dir,
        output_paths=export_result.output_paths,
        row_counts=export_result.row_counts,
        dedup_counts={
            "prices_raw": _coerce_int_note(
                ingest_result.snapshot_manifest.notes.get("prices_raw_duplicate_rows", 0)
            ),
            "pb_raw": _coerce_int_note(ingest_result.snapshot_manifest.notes.get("pb_raw", 0)),
            "roe_raw": _coerce_int_note(ingest_result.snapshot_manifest.notes.get("roe_raw", 0)),
        },
        roe_rows_using_end_date_fallback=0,
        roe_source_column=str(ingest_result.quality_notes.get("roe_source_column") or "roe"),
        dataset_version_id=ingest_result.dataset_version.version_id,
        data_root=ingestor.catalog.root,
    )


def _build_tushare_client(token: str) -> Any:
    try:
        import tushare as ts
    except ImportError as exc:  # pragma: no cover - environment-dependent import
        raise AlphaLabExperimentError(
            "tushare package is required for this adapter. Install it in your environment first."
        ) from exc
    return ts.pro_api(token)


def _require_tushare_token(token: str | None) -> str:
    resolved = (token or os.environ.get("TUSHARE_TOKEN") or "").strip()
    if not resolved:
        raise AlphaLabExperimentError(
            "Missing Tushare token. Set TUSHARE_TOKEN or pass token=... explicitly."
        )
    return resolved


def _to_tushare_date(date_str: str) -> str:
    dt = pd.to_datetime(date_str, errors="coerce")
    if pd.isna(dt):
        raise AlphaLabConfigError(f"Invalid date value: {date_str!r}")
    return str(dt.strftime("%Y%m%d"))


def _parse_tushare_dates(series: pd.Series) -> pd.Series:
    values = series.astype(str).str.strip()
    parsed = pd.to_datetime(values, format="%Y%m%d", errors="coerce")
    if parsed.isna().all():
        parsed = pd.to_datetime(values, errors="coerce")
    return parsed


def _list_trading_dates(
    pro: Any,
    *,
    start_ymd: str,
    end_ymd: str,
) -> list[str]:
    calendar = pro.trade_cal(
        exchange="SSE",
        start_date=start_ymd,
        end_date=end_ymd,
        is_open="1",
        fields="cal_date,is_open",
    )
    if calendar is None or calendar.empty:
        raise AlphaLabDataError(
            f"Tushare trade_cal returned no open days for {start_ymd} to {end_ymd}"
        )
    calendar["cal_date"] = calendar["cal_date"].astype(str).str.strip()
    trading_dates = sorted(d for d in calendar["cal_date"].tolist() if d)
    if not trading_dates:
        raise AlphaLabDataError(
            f"No trading dates parsed from Tushare trade_cal for {start_ymd} to {end_ymd}"
        )
    return trading_dates


def _resolve_roe_source_column(
    pro: Any,
    *,
    sample_asset: str,
    start_ymd: str,
    end_ymd: str,
) -> str:
    fields_pref = "ts_code,ann_date,end_date,roe_ttm"
    try:
        probe = pro.fina_indicator(
            ts_code=sample_asset,
            start_date=start_ymd,
            end_date=end_ymd,
            fields=fields_pref,
        )
    except (RuntimeError, OSError, ValueError, KeyError):
        logger.warning(
            "fina_indicator does not support `roe_ttm` in this environment; using `roe`."
        )
        return "roe"

    if probe is not None and not probe.empty and "roe_ttm" in probe.columns:
        return "roe_ttm"
    return "roe"


def _deduplicate_rows(
    frame: pd.DataFrame,
    *,
    key_cols: list[str],
    table_name: str,
) -> tuple[pd.DataFrame, int]:
    duplicate_rows = int(frame.duplicated(subset=key_cols, keep=False).sum())
    if duplicate_rows > 0:
        logger.warning(
            "%s contains %d duplicate raw rows on %s; keeping last deterministically.",
            table_name,
            duplicate_rows,
            key_cols,
        )
        frame = frame.sort_values(key_cols, kind="mergesort").drop_duplicates(
            subset=key_cols,
            keep="last",
        )
    return frame.reset_index(drop=True), duplicate_rows


def _coerce_int_note(value: object, *, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if pd.isna(value):
            return default
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return default
        try:
            return int(float(text))
        except ValueError:
            return default
    return default
