from __future__ import annotations

import logging
import os
import re
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from alpha_lab.exceptions import (
    AlphaLabConfigError,
    AlphaLabDataError,
    AlphaLabExperimentError,
)

from .tushare_adapter import (
    GeneratedRealCaseInputs,
    build_bp_factor,
    build_roe_factor,
    build_universe,
)

logger = logging.getLogger(__name__)
_ASSET_RE = re.compile(r"^(?P<code>\d{6})\.(?P<exchange>SH|SZ)$")
_BAOSTOCK_ASSET_RE = re.compile(r"^(?P<exchange>sh|sz)\.(?P<code>\d{6})$")
_ASSET_RESOLUTION_LOOKBACK_DAYS = 20
_MAX_BAOSTOCK_FETCH_WORKERS = 8
_BAOSTOCK_PARALLEL_ENV = "ALPHA_LAB_BAOSTOCK_PARALLEL_WORKERS"


@dataclass(frozen=True)
class _BaostockRawInputs:
    prices: pd.DataFrame
    pb_raw: pd.DataFrame
    roe_raw: pd.DataFrame
    dedup_counts: dict[str, int]
    roe_source_column: str


def generate_real_case_inputs(
    output_dir: str | Path,
    start_date: str = "2024-01-01",
    end_date: str = "2024-12-31",
    *,
    assets: Sequence[str] | None = None,
    asset_limit: int | None = None,
    include_roe: bool = True,
) -> GeneratedRealCaseInputs:
    """Generate canonical real-case CSV inputs from BaoStock for A-share research."""
    if asset_limit is not None and asset_limit <= 0:
        raise AlphaLabConfigError("asset_limit must be positive when provided.")

    raw_inputs = _run_with_baostock_session(
        lambda client: _fetch_raw_inputs(
            client,
            start_date=start_date,
            end_date=end_date,
            assets=assets,
            asset_limit=asset_limit,
            include_roe=include_roe,
        )
    )
    prices, pb_raw, roe_raw = _apply_asset_filters(
        prices=raw_inputs.prices,
        pb_raw=raw_inputs.pb_raw,
        roe_raw=raw_inputs.roe_raw,
        assets=assets,
        asset_limit=asset_limit,
    )
    if prices.empty:
        raise AlphaLabDataError("No price rows left after BaoStock filtering.")
    if pb_raw.empty:
        raise AlphaLabDataError("No PB rows returned from BaoStock after normalization.")

    bp_factor = build_bp_factor(pb_raw, prices)
    if roe_raw.empty:
        logger.warning(
            "BaoStock returned no ROE rows after normalization; writing empty roe_ttm.csv."
        )
        roe_factor = pd.DataFrame(columns=["date", "asset", "factor", "value"])
        roe_fallback_rows = 0
    else:
        roe_factor, roe_fallback_rows = build_roe_factor(roe_raw, prices)
    universe = build_universe(prices)

    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "prices": out_dir / "prices.csv",
        "bp": out_dir / "bp.csv",
        "roe_ttm": out_dir / "roe_ttm.csv",
        "universe": out_dir / "universe.csv",
    }
    prices.to_csv(output_paths["prices"], index=False)
    bp_factor.to_csv(output_paths["bp"], index=False)
    roe_factor.to_csv(output_paths["roe_ttm"], index=False)
    universe.to_csv(output_paths["universe"], index=False)

    return GeneratedRealCaseInputs(
        output_dir=out_dir,
        output_paths=output_paths,
        row_counts={
            "prices": len(prices),
            "bp": len(bp_factor),
            "roe_ttm": len(roe_factor),
            "universe": len(universe),
        },
        dedup_counts=dict(raw_inputs.dedup_counts),
        roe_rows_using_end_date_fallback=roe_fallback_rows,
        roe_source_column=raw_inputs.roe_source_column,
    )


def _run_with_baostock_session[T](fn: Callable[[Any], T]) -> T:
    bs = _import_baostock()
    login_result = bs.login()
    if str(getattr(login_result, "error_code", "")) != "0":
        msg = str(getattr(login_result, "error_msg", "")).strip() or "unknown login error"
        raise AlphaLabExperimentError(f"BaoStock login failed: {msg}")
    try:
        return fn(bs)
    finally:
        try:
            bs.logout()
        except (OSError, RuntimeError):
            pass


def _import_baostock() -> Any:
    try:
        import baostock as bs
    except ImportError as exc:  # pragma: no cover - environment-dependent import
        raise AlphaLabExperimentError(
            "baostock package is required for this adapter. "
            "Install it first (for uv one-off run: "
            "`UV_CACHE_DIR=/tmp/uv-cache uv run --with baostock ...`; "
            "for persistent project env: `uv add baostock`)."
        ) from exc
    return bs


def _fetch_raw_inputs(
    client: Any,
    *,
    start_date: str,
    end_date: str,
    assets: Sequence[str] | None,
    asset_limit: int | None,
    include_roe: bool,
) -> _BaostockRawInputs:
    baostock_assets = _resolve_assets(
        client,
        end_date=end_date,
        assets=assets,
        asset_limit=asset_limit,
    )
    if not baostock_assets:
        raise AlphaLabDataError("No A-share assets resolved for BaoStock fetch.")

    price_frames, pb_frames = _fetch_price_and_pb_frames(
        client=client,
        assets=baostock_assets,
        start_date=start_date,
        end_date=end_date,
    )
    if not price_frames:
        raise AlphaLabDataError("BaoStock returned no daily price rows.")

    prices = pd.concat(price_frames, ignore_index=True)
    pb_raw = pd.concat(pb_frames, ignore_index=True) if pb_frames else pd.DataFrame()
    prices, prices_dup_count = _deduplicate_rows(
        prices,
        key_cols=["date", "asset"],
        table_name="prices_raw",
    )
    pb_raw, pb_dup_count = _deduplicate_rows(
        pb_raw,
        key_cols=["date", "asset"],
        table_name="pb_raw",
    )
    prices = prices.sort_values(["date", "asset"], kind="mergesort").reset_index(drop=True)
    pb_raw = pb_raw.sort_values(["date", "asset"], kind="mergesort").reset_index(drop=True)

    if include_roe:
        roe_raw = _fetch_roe_raw(
            client,
            assets=baostock_assets,
            start_date=start_date,
            end_date=end_date,
        )
        roe_raw, roe_dup_count = _deduplicate_rows(
            roe_raw,
            key_cols=["asset", "ann_date", "end_date"],
            table_name="roe_raw",
        )
        roe_raw = roe_raw.sort_values(["asset", "ann_date", "end_date"], kind="mergesort")
        roe_raw = roe_raw.reset_index(drop=True)
    else:
        logger.info("Skipping BaoStock ROE fetch because include_roe=False.")
        roe_raw = pd.DataFrame(columns=["asset", "ann_date", "end_date", "roe_value"])
        roe_dup_count = 0

    return _BaostockRawInputs(
        prices=prices,
        pb_raw=pb_raw,
        roe_raw=roe_raw,
        dedup_counts={
            "prices_raw": prices_dup_count,
            "pb_raw": pb_dup_count,
            "roe_raw": roe_dup_count,
        },
        roe_source_column="roeAvg",
    )


def _resolve_assets(
    client: Any,
    *,
    end_date: str,
    assets: Sequence[str] | None,
    asset_limit: int | None,
) -> list[str]:
    if assets is not None:
        normalized = sorted({_to_baostock_asset(asset) for asset in assets})
    else:
        normalized = _resolve_assets_via_query_all_stock(client, end_date=end_date)
    if asset_limit is not None:
        normalized = normalized[:asset_limit]
    return normalized


def _resolve_parallel_worker_count(item_count: int) -> int:
    if item_count <= 1:
        return 1
    # BaoStock sessions are not reliably thread-safe in practice. Keep serial by default,
    # and only allow explicit opt-in parallelism via env override.
    raw = os.environ.get(_BAOSTOCK_PARALLEL_ENV, "").strip()
    if not raw:
        return 1
    try:
        configured = int(raw)
    except ValueError:
        return 1
    if configured <= 1:
        return 1
    cpu_cap = os.cpu_count() or 4
    return max(1, min(_MAX_BAOSTOCK_FETCH_WORKERS, cpu_cap, item_count, configured))


def _chunk_assets(assets: Sequence[str], worker_count: int) -> list[list[str]]:
    if not assets:
        return []
    chunk_size = max(1, (len(assets) + worker_count - 1) // worker_count)
    return [list(assets[i : i + chunk_size]) for i in range(0, len(assets), chunk_size)]


def _fetch_price_and_pb_frames(
    *,
    client: Any,
    assets: Sequence[str],
    start_date: str,
    end_date: str,
) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    worker_count = _resolve_parallel_worker_count(len(assets))
    chunks = _chunk_assets(assets, worker_count)
    if len(chunks) <= 1:
        return _fetch_price_and_pb_frames_for_chunk(
            client=client,
            assets=list(assets),
            start_date=start_date,
            end_date=end_date,
        )

    def _worker(chunk: list[str]) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
        return _run_with_baostock_session(
            lambda chunk_client: _fetch_price_and_pb_frames_for_chunk(
                client=chunk_client,
                assets=chunk,
                start_date=start_date,
                end_date=end_date,
            )
        )

    with ThreadPoolExecutor(max_workers=len(chunks)) as pool:
        results = list(pool.map(_worker, chunks))
    price_frames: list[pd.DataFrame] = []
    pb_frames: list[pd.DataFrame] = []
    for price_chunk, pb_chunk in results:
        price_frames.extend(price_chunk)
        pb_frames.extend(pb_chunk)
    return price_frames, pb_frames


def _fetch_price_and_pb_frames_for_chunk(
    *,
    client: Any,
    assets: Sequence[str],
    start_date: str,
    end_date: str,
) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    price_frames: list[pd.DataFrame] = []
    pb_frames: list[pd.DataFrame] = []
    for asset in assets:
        frame = _query_history_price_and_pb(
            client,
            asset=asset,
            start_date=start_date,
            end_date=end_date,
        )
        if frame.empty:
            continue
        price_frames.append(frame[["date", "asset", "close", "volume", "amount"]])
        pb_frames.append(frame[["date", "asset", "pb"]])
    return price_frames, pb_frames


def _resolve_assets_via_query_all_stock(client: Any, *, end_date: str) -> list[str]:
    anchor = pd.to_datetime(end_date, errors="coerce")
    if pd.isna(anchor):
        candidate_days = [end_date]
    else:
        candidate_days = [
            (anchor - pd.Timedelta(days=delta)).strftime("%Y-%m-%d")
            for delta in range(_ASSET_RESOLUTION_LOOKBACK_DAYS + 1)
        ]
    first_day = candidate_days[0]
    for day in candidate_days:
        resolved = _query_all_stock_ashare_assets(client, day=day)
        if resolved:
            if day != first_day:
                logger.info(
                    "BaoStock query_all_stock returned no A-share rows on %s; "
                    "fell back to %s with %d assets.",
                    first_day,
                    day,
                    len(resolved),
                )
            return resolved
    return []


def _query_all_stock_ashare_assets(client: Any, *, day: str) -> list[str]:
    query = client.query_all_stock(day=day)
    frame = _result_to_frame(query, query_name="query_all_stock")
    code_col = _pick_column(frame, candidates=["code"])
    if code_col is None:
        raise AlphaLabDataError("BaoStock query_all_stock did not return `code` column.")
    return sorted(
        {
            _to_baostock_asset(str(code))
            for code in frame[code_col].astype(str).tolist()
            if _is_ashare_baostock_asset(str(code))
        }
    )


def _query_history_price_and_pb(
    client: Any,
    *,
    asset: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    query = client.query_history_k_data_plus(
        asset,
        "date,code,close,volume,amount,pbMRQ",
        start_date=start_date,
        end_date=end_date,
        frequency="d",
        adjustflag="2",
    )
    frame = _result_to_frame(query, query_name="query_history_k_data_plus")
    if frame.empty:
        return pd.DataFrame(columns=["date", "asset", "close", "volume", "amount", "pb"])
    date_col = _pick_column(frame, candidates=["date"])
    code_col = _pick_column(frame, candidates=["code"])
    close_col = _pick_column(frame, candidates=["close"])
    volume_col = _pick_column(frame, candidates=["volume", "vol"])
    amount_col = _pick_column(frame, candidates=["amount"])
    pb_col = _pick_column(frame, candidates=["pbMRQ", "pb_mrq", "pb"])
    if date_col is None or code_col is None or close_col is None or pb_col is None:
        return pd.DataFrame(columns=["date", "asset", "close", "volume", "amount", "pb"])

    out = pd.DataFrame(
        {
            "date": pd.to_datetime(frame[date_col], errors="coerce"),
            "asset": frame[code_col].astype(str).map(_to_tushare_asset),
            "close": pd.to_numeric(frame[close_col], errors="coerce"),
            "volume": (
                pd.to_numeric(frame[volume_col], errors="coerce")
                if volume_col is not None
                else np.nan
            ),
            "amount": (
                pd.to_numeric(frame[amount_col], errors="coerce")
                if amount_col is not None
                else np.nan
            ),
            "pb": pd.to_numeric(frame[pb_col], errors="coerce"),
        }
    )
    out = out.dropna(subset=["date", "asset", "close"]).copy()
    out = out[out["close"] > 0].copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    return out[["date", "asset", "close", "volume", "amount", "pb"]]


def _fetch_roe_raw(
    client: Any,
    *,
    assets: Sequence[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    year_quarters = _iter_profit_query_year_quarters(start_date=start_date, end_date=end_date)
    if not year_quarters:
        return pd.DataFrame(columns=["asset", "ann_date", "end_date", "roe_value"])

    worker_count = _resolve_parallel_worker_count(len(assets))
    chunks = _chunk_assets(assets, worker_count)
    if len(chunks) <= 1:
        frames = _fetch_roe_frames_for_chunk(
            client=client,
            assets=list(assets),
            year_quarters=year_quarters,
        )
    else:

        def _worker(chunk: list[str]) -> list[pd.DataFrame]:
            return _run_with_baostock_session(
                lambda chunk_client: _fetch_roe_frames_for_chunk(
                    client=chunk_client,
                    assets=chunk,
                    year_quarters=year_quarters,
                )
            )

        with ThreadPoolExecutor(max_workers=len(chunks)) as pool:
            nested_frames = list(pool.map(_worker, chunks))
        frames = [frame for chunk_frames in nested_frames for frame in chunk_frames]

    if not frames:
        return pd.DataFrame(columns=["asset", "ann_date", "end_date", "roe_value"])

    raw = pd.concat(frames, ignore_index=True)
    code_col = _pick_column(raw, candidates=["code"])
    ann_col = _pick_column(raw, candidates=["pubDate", "pubdate", "ann_date"])
    end_col = _pick_column(raw, candidates=["statDate", "statdate", "end_date"])
    roe_col = _pick_column(raw, candidates=["roeAvg", "roeavg", "roe"])
    if code_col is None or roe_col is None:
        return pd.DataFrame(columns=["asset", "ann_date", "end_date", "roe_value"])

    ann = pd.to_datetime(raw[ann_col], errors="coerce") if ann_col is not None else pd.NaT
    stat = pd.to_datetime(raw[end_col], errors="coerce") if end_col is not None else pd.NaT
    normalized = pd.DataFrame(
        {
            "asset": raw[code_col].astype(str).map(_to_tushare_asset),
            "ann_date": ann,
            "end_date": stat,
            "roe_value": pd.to_numeric(raw[roe_col], errors="coerce"),
        }
    )
    normalized = normalized.dropna(subset=["asset", "roe_value"], how="any").copy()
    if normalized.empty:
        return pd.DataFrame(columns=["asset", "ann_date", "end_date", "roe_value"])
    normalized = normalized[normalized["asset"] != ""].copy()
    normalized["ann_date"] = normalized["ann_date"].dt.strftime("%Y-%m-%d")
    normalized["end_date"] = normalized["end_date"].dt.strftime("%Y-%m-%d")
    return normalized[["asset", "ann_date", "end_date", "roe_value"]]


def _fetch_roe_frames_for_chunk(
    *,
    client: Any,
    assets: Sequence[str],
    year_quarters: Sequence[tuple[int, int]],
) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for asset in assets:
        for year, quarter in year_quarters:
            query = client.query_profit_data(code=asset, year=year, quarter=quarter)
            frame = _result_to_frame(
                query,
                query_name="query_profit_data",
                allow_error=True,
            )
            if frame.empty:
                continue
            frames.append(frame)
    return frames


def _iter_profit_query_year_quarters(*, start_date: str, end_date: str) -> list[tuple[int, int]]:
    start_year = int(pd.to_datetime(start_date).year)
    end_year = int(pd.to_datetime(end_date).year)
    return [
        (year, quarter) for year in range(start_year - 1, end_year + 1) for quarter in (1, 2, 3, 4)
    ]


def _apply_asset_filters(
    *,
    prices: pd.DataFrame,
    pb_raw: pd.DataFrame,
    roe_raw: pd.DataFrame,
    assets: Sequence[str] | None,
    asset_limit: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected = sorted(prices["asset"].astype(str).unique().tolist())
    if assets is not None:
        requested = {_to_tushare_asset(asset) for asset in assets}
        selected = [asset for asset in selected if asset in requested]
    if asset_limit is not None:
        selected = selected[:asset_limit]
    selected_set = set(selected)
    return (
        prices[prices["asset"].isin(selected_set)].copy(),
        pb_raw[pb_raw["asset"].isin(selected_set)].copy(),
        roe_raw[roe_raw["asset"].isin(selected_set)].copy(),
    )


def _result_to_frame(
    result: Any,
    *,
    query_name: str,
    allow_error: bool = False,
) -> pd.DataFrame:
    if str(getattr(result, "error_code", "")) != "0":
        msg = str(getattr(result, "error_msg", "")).strip() or "unknown error"
        if allow_error:
            logger.warning("BaoStock %s failed: %s", query_name, msg)
            return pd.DataFrame()
        raise AlphaLabExperimentError(f"BaoStock {query_name} failed: {msg}")

    fields_obj = getattr(result, "fields", [])
    fields = [str(col) for col in fields_obj] if fields_obj is not None else []
    rows: list[list[str]] = []
    while bool(result.next()):
        row = result.get_row_data()
        rows.append(list(row))
    if not rows:
        return pd.DataFrame(columns=fields)
    if fields:
        return pd.DataFrame(rows, columns=fields)
    return pd.DataFrame(rows)


def _pick_column(frame: pd.DataFrame, *, candidates: Sequence[str]) -> str | None:
    col_map = {str(col).lower(): str(col) for col in frame.columns}
    for candidate in candidates:
        match = col_map.get(candidate.lower())
        if match is not None:
            return match
    return None


def _to_baostock_asset(asset: str) -> str:
    token = str(asset).strip()
    bs_match = _BAOSTOCK_ASSET_RE.match(token.lower())
    if bs_match is not None:
        exchange = bs_match.group("exchange")
        code = bs_match.group("code")
        return f"{exchange}.{code}"
    normalized = token.upper()
    ts_match = _ASSET_RE.match(normalized)
    if ts_match is None:
        raise AlphaLabConfigError(f"Unsupported asset format: {asset!r}")
    code = ts_match.group("code")
    exchange = ts_match.group("exchange").lower()
    return f"{exchange}.{code}"


def _to_tushare_asset(asset: str) -> str:
    token = str(asset).strip()
    match = _BAOSTOCK_ASSET_RE.match(token.lower())
    if match is not None:
        return f"{match.group('code')}.{match.group('exchange').upper()}"
    match = _ASSET_RE.match(token.upper())
    if match is not None:
        return f"{match.group('code')}.{match.group('exchange').upper()}"
    return ""


def _is_ashare_baostock_asset(asset: str) -> bool:
    match = _BAOSTOCK_ASSET_RE.match(str(asset).strip().lower())
    if match is None:
        return False
    exchange = match.group("exchange")
    code = match.group("code")
    if exchange == "sh":
        return code.startswith("6")
    return code.startswith(("0", "3"))


def _deduplicate_rows(
    frame: pd.DataFrame,
    *,
    key_cols: list[str],
    table_name: str,
) -> tuple[pd.DataFrame, int]:
    if frame.empty:
        return frame.copy(), 0
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
