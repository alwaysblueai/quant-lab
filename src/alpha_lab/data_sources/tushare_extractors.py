from __future__ import annotations

import datetime
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import pandas as pd

from alpha_lab.data_sources.tushare_cache import (
    RawSnapshotManifest,
    TushareRawSnapshot,
    default_raw_snapshot_dir,
    write_raw_snapshot,
    write_raw_snapshot_manifest,
)
from alpha_lab.data_sources.tushare_client import TushareClientProtocol
from alpha_lab.data_sources.tushare_schemas import (
    ENDPOINT_STATUS_DEGRADED,
    ENDPOINT_STATUS_FAILED,
    ENDPOINT_STATUS_SUCCESS,
    RAW_SNAPSHOT_SORT_KEYS,
)


class ExtractionMode(StrEnum):
    RANGE_QUERY = "range_query"
    STATIC_SNAPSHOT = "static_snapshot"
    TRADE_DATE_WINDOW = "trade_date_window"
    FINANCIAL_QUARTERLY = "financial_quarterly"


@dataclass(frozen=True)
class TushareSnapshotRequest:
    """One raw extraction request against a Tushare endpoint."""

    endpoint: str
    params: Mapping[str, object]
    fields: str | None = None
    extraction_mode: ExtractionMode = ExtractionMode.RANGE_QUERY
    required: bool = True
    note_on_failure: str | None = None


@dataclass(frozen=True)
class TushareExtractionArtifacts:
    """Collection of raw snapshot files and their manifest."""

    manifest: RawSnapshotManifest
    snapshots: tuple[TushareRawSnapshot, ...]


class RequiredEndpointExtractionError(RuntimeError):
    """Raised when a required endpoint fails during fetch-snapshots."""

    def __init__(
        self,
        *,
        endpoint: str,
        extraction_mode: str,
        params: Mapping[str, object],
        raw_error: str,
        manifest_path: Path,
    ) -> None:
        self.endpoint = endpoint
        self.extraction_mode = extraction_mode
        self.params = dict(params)
        self.raw_error = raw_error
        self.manifest_path = manifest_path
        message = (
            f"fetch-snapshots failed on endpoint={endpoint} mode={extraction_mode} "
            f"params={_format_params(params)} error={raw_error}. "
            f"manifest={manifest_path}"
        )
        super().__init__(message)


@dataclass(frozen=True)
class _EndpointExtractionResult:
    payload: pd.DataFrame
    actual_endpoint: str
    metadata: dict[str, object]


class _EndpointExtractionError(RuntimeError):
    def __init__(self, message: str, *, metadata: Mapping[str, object] | None = None) -> None:
        super().__init__(message)
        self.metadata = dict(metadata or {})


def fetch_tushare_raw_snapshots(
    client: TushareClientProtocol,
    *,
    snapshot_name: str,
    start_date: str,
    end_date: str,
    raw_root: str | Path | None = None,
    extraction_utc: str | None = None,
    page_size: int = 5000,
    endpoint_diagnostic_hook: Callable[[Mapping[str, object]], None] | None = None,
) -> TushareExtractionArtifacts:
    """Fetch the v1 raw data surface needed by the canonical A-share cases."""

    if not snapshot_name.strip():
        raise ValueError("snapshot_name must be non-empty")
    snapshot_dir = (
        Path(raw_root).resolve() / snapshot_name
        if raw_root is not None
        else default_raw_snapshot_dir(snapshot_name)
    )
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    requests = _default_snapshot_requests(start_date=start_date, end_date=end_date)
    written: list[TushareRawSnapshot] = []
    unavailable: list[str] = []
    notes: list[str] = []
    endpoint_extractions: dict[str, object] = {}
    endpoint_status_table: list[dict[str, object]] = []
    degraded_endpoints: set[str] = set()
    payload_cache: dict[str, pd.DataFrame] = {}
    fatal_required_failure: (
        tuple[TushareSnapshotRequest, Exception, dict[str, object]] | None
    ) = None

    for request in requests:
        started_at = _utc_now()
        _emit_endpoint_diagnostic(
            endpoint_diagnostic_hook,
            {
                "event": "started",
                **_base_endpoint_metadata(request=request, started_at=started_at),
            },
        )
        try:
            extracted = _extract_one_request(
                client,
                request=request,
                page_size=page_size,
                stock_basic_snapshot=payload_cache.get("stock_basic"),
                trade_calendar_snapshot=payload_cache.get("trade_cal"),
            )
        except Exception as exc:
            failure_meta = _build_failure_metadata(
                request=request,
                exc=exc,
                started_at=started_at,
                finished_at=_utc_now(),
            )
            endpoint_extractions[request.endpoint] = failure_meta
            endpoint_status_table.append(dict(failure_meta))
            if bool(failure_meta.get("degradation_used", False)):
                degraded_endpoints.add(request.endpoint)
            _emit_endpoint_diagnostic(
                endpoint_diagnostic_hook,
                {"event": "finished", **failure_meta},
            )
            if request.required:
                fatal_required_failure = (request, exc, failure_meta)
                break
            unavailable.append(request.endpoint)
            if request.note_on_failure is not None:
                notes.append(request.note_on_failure)
            continue
        endpoint_meta = _build_success_metadata(
            request=request,
            extracted=extracted,
            started_at=started_at,
            finished_at=_utc_now(),
        )
        endpoint_extractions[request.endpoint] = endpoint_meta
        endpoint_status_table.append(dict(endpoint_meta))
        if bool(endpoint_meta.get("degradation_used", False)):
            degraded_endpoints.add(request.endpoint)
        _emit_endpoint_diagnostic(
            endpoint_diagnostic_hook,
            {"event": "finished", **endpoint_meta},
        )
        written.append(
            write_raw_snapshot(
                extracted.payload,
                endpoint=request.endpoint,
                snapshot_dir=snapshot_dir,
                params=dict(request.params),
                extraction_utc=extraction_utc,
                sort_keys=RAW_SNAPSHOT_SORT_KEYS.get(request.endpoint),
                metadata_extras={"extraction": endpoint_meta},
            )
        )
        payload_cache[request.endpoint] = extracted.payload.copy()

    manifest = write_raw_snapshot_manifest(
        snapshot_name=snapshot_name,
        snapshot_dir=snapshot_dir,
        extraction_utc=extraction_utc,
        endpoints=[item.endpoint for item in written],
        unavailable_endpoints=unavailable,
        notes=notes,
        endpoint_extractions=endpoint_extractions,
        endpoint_status_table=endpoint_status_table,
        degraded_endpoints=sorted(degraded_endpoints),
    )
    if fatal_required_failure is not None:
        failed_request, raw_exc, failure_meta = fatal_required_failure
        raise RequiredEndpointExtractionError(
            endpoint=failed_request.endpoint,
            extraction_mode=failed_request.extraction_mode.value,
            params=failed_request.params,
            raw_error=str(failure_meta.get("error_message", _error_text(raw_exc))),
            manifest_path=manifest.manifest_path,
        ) from raw_exc
    return TushareExtractionArtifacts(
        manifest=manifest,
        snapshots=tuple(written),
    )


def _extract_one_request(
    client: TushareClientProtocol,
    *,
    request: TushareSnapshotRequest,
    page_size: int,
    stock_basic_snapshot: pd.DataFrame | None,
    trade_calendar_snapshot: pd.DataFrame | None,
) -> _EndpointExtractionResult:
    if request.extraction_mode in {ExtractionMode.RANGE_QUERY, ExtractionMode.STATIC_SNAPSHOT}:
        payload = _query_all_pages(
            client,
            endpoint=request.endpoint,
            params=dict(request.params),
            fields=request.fields,
            page_size=page_size,
        )
        metadata = {
            "endpoint_requested": request.endpoint,
            "endpoint_used": request.endpoint,
            "extraction_mode": request.extraction_mode.value,
            "date_window": _extract_date_window(request.params),
            "fallback_occurred": False,
            "degraded": False,
            "row_count": int(len(payload)),
        }
        return _EndpointExtractionResult(
            payload=payload,
            actual_endpoint=request.endpoint,
            metadata=metadata,
        )
    if request.extraction_mode is ExtractionMode.TRADE_DATE_WINDOW:
        return _extract_trade_date_window(
            client,
            request=request,
            page_size=page_size,
            trade_calendar_snapshot=trade_calendar_snapshot,
        )
    if request.extraction_mode is ExtractionMode.FINANCIAL_QUARTERLY:
        return _extract_financial_quarterly(
            client,
            request=request,
            page_size=page_size,
            stock_basic_snapshot=stock_basic_snapshot,
        )
    raise ValueError(f"unsupported extraction mode: {request.extraction_mode!r}")


def _default_snapshot_requests(
    *,
    start_date: str,
    end_date: str,
) -> tuple[TushareSnapshotRequest, ...]:
    common_range: dict[str, object] = {"start_date": start_date, "end_date": end_date}
    return (
        TushareSnapshotRequest(
            endpoint="trade_cal",
            params={
                "exchange": "SSE",
                "start_date": start_date,
                "end_date": end_date,
            },
            extraction_mode=ExtractionMode.RANGE_QUERY,
            required=True,
        ),
        TushareSnapshotRequest(
            endpoint="stock_basic",
            params={"exchange": "", "list_status": "L"},
            fields=(
                "ts_code,symbol,name,area,industry,market,exchange,list_status,"
                "list_date,delist_date,is_hs"
            ),
            extraction_mode=ExtractionMode.STATIC_SNAPSHOT,
            required=True,
        ),
        TushareSnapshotRequest(
            endpoint="daily",
            params=common_range,
            extraction_mode=ExtractionMode.TRADE_DATE_WINDOW,
            required=True,
        ),
        TushareSnapshotRequest(
            endpoint="daily_basic",
            params=common_range,
            extraction_mode=ExtractionMode.TRADE_DATE_WINDOW,
            required=True,
        ),
        TushareSnapshotRequest(
            endpoint="adj_factor",
            params=common_range,
            extraction_mode=ExtractionMode.RANGE_QUERY,
            required=False,
            note_on_failure=(
                "adj_factor unavailable; standardized prices will retain raw close only"
            ),
        ),
        TushareSnapshotRequest(
            endpoint="suspend_d",
            params=common_range,
            extraction_mode=ExtractionMode.RANGE_QUERY,
            required=False,
            note_on_failure="suspend_d unavailable; halt flags will use conservative fallback",
        ),
        TushareSnapshotRequest(
            endpoint="stk_limit",
            params=common_range,
            extraction_mode=ExtractionMode.RANGE_QUERY,
            required=False,
            note_on_failure=(
                "stk_limit unavailable; limit-locked flags will use conservative fallback"
            ),
        ),
        TushareSnapshotRequest(
            endpoint="fina_indicator",
            params=common_range,
            extraction_mode=ExtractionMode.FINANCIAL_QUARTERLY,
            required=False,
            note_on_failure=(
                "fina_indicator unavailable; quality proxy will be unavailable for composite case"
            ),
        ),
    )


def _extract_trade_date_window(
    client: TushareClientProtocol,
    *,
    request: TushareSnapshotRequest,
    page_size: int,
    trade_calendar_snapshot: pd.DataFrame | None,
) -> _EndpointExtractionResult:
    start_date = _required_param(request.params, "start_date")
    end_date = _required_param(request.params, "end_date")
    trade_dates = _resolve_trade_dates(
        client,
        start_date=start_date,
        end_date=end_date,
        page_size=page_size,
        trade_calendar_snapshot=trade_calendar_snapshot,
    )
    base_params = {
        key: value
        for key, value in dict(request.params).items()
        if key not in {"start_date", "end_date"}
    }
    frames: list[pd.DataFrame] = []
    for trade_date in trade_dates:
        frame = _query_no_pagination(
            client,
            endpoint=request.endpoint,
            params={**base_params, "trade_date": trade_date},
            fields=request.fields,
        )
        if not frame.empty and "trade_date" in frame.columns:
            resolved_trade_date = frame["trade_date"].astype(str).str.strip()
            frame = frame.loc[resolved_trade_date == trade_date].reset_index(drop=True)
        if not frame.empty:
            frames.append(frame.copy())
    payload = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return _EndpointExtractionResult(
        payload=payload,
        actual_endpoint=request.endpoint,
        metadata={
            "endpoint_requested": request.endpoint,
            "endpoint_used": request.endpoint,
            "extraction_mode": request.extraction_mode.value,
            "date_window": {"start_date": start_date, "end_date": end_date},
            "trade_dates_requested_count": len(trade_dates),
            "trade_dates_sample": trade_dates[:20],
            "fallback_occurred": False,
            "degraded": False,
            "row_count": int(len(payload)),
        },
    )


def _extract_financial_quarterly(
    client: TushareClientProtocol,
    *,
    request: TushareSnapshotRequest,
    page_size: int,
    stock_basic_snapshot: pd.DataFrame | None,
) -> _EndpointExtractionResult:
    start_date = _required_param(request.params, "start_date")
    end_date = _required_param(request.params, "end_date")
    periods = _quarter_end_periods(start_date=start_date, end_date=end_date)

    vip_error: Exception | None = None
    try:
        payload = _fetch_fina_indicator_vip(
            client,
            periods=periods,
            fields=request.fields,
            page_size=page_size,
        )
        return _EndpointExtractionResult(
            payload=payload,
            actual_endpoint="fina_indicator_vip",
            metadata={
                "endpoint_requested": request.endpoint,
                "endpoint_used": "fina_indicator_vip",
                "extraction_mode": request.extraction_mode.value,
                "date_window": {"start_date": start_date, "end_date": end_date},
                "periods_requested": periods,
                "period_query_count": len(periods),
                "stocks_queried_count": 0,
                "fallback_occurred": False,
                "degraded": False,
                "row_count": int(len(payload)),
            },
        )
    except Exception as exc:
        vip_error = exc

    stock_codes = _resolve_stock_universe(
        client,
        stock_basic_snapshot=stock_basic_snapshot,
        page_size=page_size,
    )
    fallback_payload, failed_codes = _fetch_fina_indicator_per_stock(
        client,
        stock_codes=stock_codes,
        start_date=start_date,
        end_date=end_date,
        fields=request.fields,
        page_size=page_size,
    )
    vip_error_text = _error_text(vip_error)
    vip_permission = _is_permission_or_capability_error(vip_error)
    if not stock_codes:
        raise _EndpointExtractionError(
            "fina_indicator fallback failed because stock universe is empty",
            metadata={
                "endpoint_requested": request.endpoint,
                "endpoint_used": "fina_indicator",
                "extraction_mode": request.extraction_mode.value,
                "date_window": {"start_date": start_date, "end_date": end_date},
                "periods_requested": periods,
                "period_query_count": len(periods),
                "stocks_queried_count": 0,
                "stocks_failed_count": 0,
                "fallback_occurred": True,
                "degraded": True,
                "degradation_reason": "empty_stock_universe",
                "vip_error": vip_error_text,
                "vip_permission_or_capability_issue": vip_permission,
            },
        )

    if fallback_payload.empty and failed_codes:
        raise _EndpointExtractionError(
            "fina_indicator fallback failed for all queried stocks",
            metadata={
                "endpoint_requested": request.endpoint,
                "endpoint_used": "fina_indicator",
                "extraction_mode": request.extraction_mode.value,
                "date_window": {"start_date": start_date, "end_date": end_date},
                "periods_requested": periods,
                "period_query_count": len(periods),
                "stocks_queried_count": len(stock_codes),
                "stocks_failed_count": len(failed_codes),
                "failed_stock_codes_sample": failed_codes[:20],
                "fallback_occurred": True,
                "degraded": True,
                "degradation_reason": "vip_failed_and_all_stock_queries_failed",
                "vip_error": vip_error_text,
                "vip_permission_or_capability_issue": vip_permission,
            },
        )

    degraded = bool(failed_codes) or vip_permission
    degradation_reason = []
    if vip_permission:
        degradation_reason.append("vip_permission_or_capability_issue")
    elif vip_error_text:
        degradation_reason.append("vip_error_fallback_used")
    if failed_codes:
        degradation_reason.append("partial_stock_query_failures")
    return _EndpointExtractionResult(
        payload=fallback_payload,
        actual_endpoint="fina_indicator",
        metadata={
            "endpoint_requested": request.endpoint,
            "endpoint_used": "fina_indicator",
            "extraction_mode": request.extraction_mode.value,
            "date_window": {"start_date": start_date, "end_date": end_date},
            "periods_requested": periods,
            "period_query_count": len(periods),
            "stocks_queried_count": len(stock_codes),
            "stocks_failed_count": len(failed_codes),
            "failed_stock_codes_sample": failed_codes[:20],
            "fallback_occurred": True,
            "degraded": degraded,
            "degradation_reason": ",".join(degradation_reason) if degradation_reason else None,
            "vip_error": vip_error_text,
            "vip_permission_or_capability_issue": vip_permission,
            "row_count": int(len(fallback_payload)),
        },
    )


def _fetch_fina_indicator_vip(
    client: TushareClientProtocol,
    *,
    periods: list[str],
    fields: str | None,
    page_size: int,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for period in periods:
        frame = _query_all_pages(
            client,
            endpoint="fina_indicator_vip",
            params={"period": period},
            fields=fields,
            page_size=page_size,
        )
        if not frame.empty:
            frames.append(frame.copy())
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _fetch_fina_indicator_per_stock(
    client: TushareClientProtocol,
    *,
    stock_codes: list[str],
    start_date: str,
    end_date: str,
    fields: str | None,
    page_size: int,
) -> tuple[pd.DataFrame, list[str]]:
    frames: list[pd.DataFrame] = []
    failed_codes: list[str] = []
    for ts_code in stock_codes:
        try:
            frame = _query_all_pages(
                client,
                endpoint="fina_indicator",
                params={
                    "ts_code": ts_code,
                    "start_date": start_date,
                    "end_date": end_date,
                },
                fields=fields,
                page_size=page_size,
            )
        except Exception:
            failed_codes.append(ts_code)
            continue
        if not frame.empty:
            frames.append(frame.copy())
    if not frames:
        return pd.DataFrame(), failed_codes
    return pd.concat(frames, ignore_index=True), failed_codes


def _resolve_stock_universe(
    client: TushareClientProtocol,
    *,
    stock_basic_snapshot: pd.DataFrame | None,
    page_size: int,
) -> list[str]:
    stock_basic = stock_basic_snapshot
    if stock_basic is None or stock_basic.empty or "ts_code" not in stock_basic.columns:
        stock_basic = _query_all_pages(
            client,
            endpoint="stock_basic",
            params={"exchange": "", "list_status": "L"},
            fields="ts_code",
            page_size=page_size,
        )
    if stock_basic.empty or "ts_code" not in stock_basic.columns:
        return []
    ts_code_series = stock_basic["ts_code"].astype(str).str.strip()
    ts_code_series = ts_code_series[ts_code_series != ""]
    return sorted(ts_code_series.unique().tolist())


def _resolve_trade_dates(
    client: TushareClientProtocol,
    *,
    start_date: str,
    end_date: str,
    page_size: int,
    trade_calendar_snapshot: pd.DataFrame | None,
) -> list[str]:
    trade_calendar = trade_calendar_snapshot
    if trade_calendar is None or trade_calendar.empty or "cal_date" not in trade_calendar.columns:
        trade_calendar = _query_all_pages(
            client,
            endpoint="trade_cal",
            params={"exchange": "SSE", "start_date": start_date, "end_date": end_date},
            fields=None,
            page_size=page_size,
        )
    if trade_calendar.empty or "cal_date" not in trade_calendar.columns:
        raise _EndpointExtractionError(
            "trade_cal did not return cal_date; cannot resolve trade_date window",
            metadata={
                "endpoint_used": "trade_cal",
                "degradation_reason": "trade_calendar_missing",
            },
        )
    calendar = trade_calendar.copy()
    calendar["cal_date"] = calendar["cal_date"].astype(str).str.strip()
    in_window = (calendar["cal_date"] >= start_date) & (calendar["cal_date"] <= end_date)
    if "is_open" in calendar.columns:
        open_mask = _is_open_calendar_flag(calendar["is_open"])
    else:
        open_mask = pd.Series(True, index=calendar.index)
    selected = calendar.loc[in_window & open_mask, "cal_date"]
    dates = sorted({item for item in selected.tolist() if item})
    if not dates:
        raise _EndpointExtractionError(
            "trade_cal returned no open trading dates for requested window",
            metadata={
                "endpoint_used": "trade_cal",
                "date_window": {"start_date": start_date, "end_date": end_date},
                "degradation_reason": "empty_trade_date_window",
            },
        )
    return dates


def _is_open_calendar_flag(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if not numeric.isna().all():
        return numeric.fillna(0).astype(int) == 1
    text = series.astype(str).str.strip().str.lower()
    return text.isin({"1", "true", "t", "y", "yes"})


def _quarter_end_periods(*, start_date: str, end_date: str) -> list[str]:
    start = pd.to_datetime(start_date, format="%Y%m%d", errors="raise")
    end = pd.to_datetime(end_date, format="%Y%m%d", errors="raise")
    if start > end:
        raise ValueError("start_date must be <= end_date")
    periods = pd.period_range(start=start.to_period("Q"), end=end.to_period("Q"), freq="Q")
    return [period.end_time.strftime("%Y%m%d") for period in periods]


def _required_param(params: Mapping[str, object], key: str) -> str:
    value = str(params.get(key, "")).strip()
    if not value:
        raise ValueError(f"{key} is required for financial quarterly extraction")
    return value


def _extract_date_window(params: Mapping[str, object]) -> dict[str, str] | None:
    start = str(params.get("start_date", "")).strip()
    end = str(params.get("end_date", "")).strip()
    if not start and not end:
        return None
    return {"start_date": start, "end_date": end}


def _is_permission_or_capability_error(exc: Exception | None) -> bool:
    if exc is None:
        return False
    text = _error_text(exc).lower()
    keywords = (
        "permission",
        "forbidden",
        "denied",
        "insufficient",
        "vip",
        "积分",
        "权限",
        "必填参数",
        "unknown api",
        "not support",
    )
    return any(token in text for token in keywords)


def _error_text(exc: Exception | None) -> str:
    if exc is None:
        return ""
    return str(exc).strip() or exc.__class__.__name__


def _build_failure_metadata(
    *,
    request: TushareSnapshotRequest,
    exc: Exception,
    started_at: str,
    finished_at: str,
) -> dict[str, object]:
    metadata = _base_endpoint_metadata(request=request, started_at=started_at)
    metadata["finished_at"] = finished_at
    if isinstance(exc, _EndpointExtractionError):
        metadata.update(exc.metadata)
    fallback_used = bool(metadata.get("fallback_occurred", False))
    degradation_used = bool(metadata.get("degraded", False))
    if not request.required:
        degradation_used = True
    metadata["status"] = ENDPOINT_STATUS_FAILED
    metadata["fallback_used"] = fallback_used
    metadata["degradation_used"] = degradation_used
    metadata["fallback_occurred"] = fallback_used
    metadata["degraded"] = degradation_used
    metadata["row_count"] = None
    metadata["error_message"] = _error_text(exc)
    metadata["error"] = metadata["error_message"]
    metadata["required"] = request.required
    metadata["required_or_optional"] = "required" if request.required else "optional"
    metadata["params"] = dict(request.params)
    metadata.setdefault("endpoint_used", request.endpoint)
    metadata.setdefault("endpoint_requested", request.endpoint)
    metadata.setdefault("extraction_mode", request.extraction_mode.value)
    return metadata


def _build_success_metadata(
    *,
    request: TushareSnapshotRequest,
    extracted: _EndpointExtractionResult,
    started_at: str,
    finished_at: str,
) -> dict[str, object]:
    metadata = _base_endpoint_metadata(request=request, started_at=started_at)
    metadata.update(dict(extracted.metadata))
    metadata["finished_at"] = finished_at
    metadata["params"] = dict(request.params)
    metadata["required"] = request.required
    metadata["required_or_optional"] = "required" if request.required else "optional"
    metadata["endpoint_requested"] = request.endpoint
    metadata["endpoint_used"] = extracted.actual_endpoint
    fallback_used = bool(metadata.get("fallback_occurred", False))
    degradation_used = bool(metadata.get("degraded", False))
    metadata["fallback_used"] = fallback_used
    metadata["degradation_used"] = degradation_used
    metadata["fallback_occurred"] = fallback_used
    metadata["degraded"] = degradation_used
    metadata["status"] = ENDPOINT_STATUS_DEGRADED if degradation_used else ENDPOINT_STATUS_SUCCESS
    row_count = metadata.get("row_count")
    metadata["row_count"] = int(row_count) if row_count is not None else int(len(extracted.payload))
    metadata["error_message"] = None
    metadata["error"] = None
    return metadata


def _base_endpoint_metadata(
    *,
    request: TushareSnapshotRequest,
    started_at: str,
) -> dict[str, object]:
    return {
        "endpoint_requested": request.endpoint,
        "endpoint_used": request.endpoint,
        "extraction_mode": request.extraction_mode.value,
        "params": dict(request.params),
        "required": request.required,
        "required_or_optional": "required" if request.required else "optional",
        "started_at": started_at,
        "finished_at": None,
        "status": None,
        "row_count": None,
        "fallback_used": False,
        "degradation_used": False,
        "fallback_occurred": False,
        "degraded": False,
        "error_message": None,
        "error": None,
    }


def _emit_endpoint_diagnostic(
    hook: Callable[[Mapping[str, object]], None] | None,
    diagnostic: Mapping[str, object],
) -> None:
    if hook is None:
        return
    hook(dict(diagnostic))


def _query_no_pagination(
    client: TushareClientProtocol,
    *,
    endpoint: str,
    params: dict[str, object],
    fields: str | None,
) -> pd.DataFrame:
    frame = client.query(
        endpoint,
        fields=fields,
        **params,
    )
    return frame.copy()


def _query_all_pages(
    client: TushareClientProtocol,
    *,
    endpoint: str,
    params: dict[str, object],
    fields: str | None,
    page_size: int,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    empty_columns: list[str] | None = None
    offset = 0
    while True:
        frame = client.query(
            endpoint,
            fields=fields,
            limit=page_size,
            offset=offset,
            **params,
        )
        if frame.empty:
            if empty_columns is None and len(frame.columns) > 0:
                empty_columns = list(frame.columns)
            break
        frames.append(frame.copy())
        if len(frame) < page_size:
            break
        offset += page_size
    if not frames:
        if empty_columns is not None:
            return pd.DataFrame(columns=empty_columns)
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _format_params(params: Mapping[str, object]) -> str:
    try:
        return json.dumps(dict(params), sort_keys=True, ensure_ascii=False, default=str)
    except TypeError:
        return repr(dict(params))


def _utc_now() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")
