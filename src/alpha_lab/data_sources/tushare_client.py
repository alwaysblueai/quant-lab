from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, cast

import pandas as pd


class TushareClientProtocol(Protocol):
    """Minimal protocol so extractors can be fully mocked in tests."""

    def query(
        self,
        api_name: str,
        *,
        fields: str | None = None,
        limit: int | None = None,
        offset: int | None = None,
        **params: object,
    ) -> pd.DataFrame: ...


class _TushareProProtocol(Protocol):
    def query(self, api_name: str, **params: object) -> pd.DataFrame: ...


class TushareQueryError(RuntimeError):
    """Raised when a live Tushare query fails."""

    def __init__(
        self,
        *,
        endpoint: str,
        params: Mapping[str, object],
        raw_error: Exception,
    ) -> None:
        self.endpoint = endpoint
        self.params = dict(params)
        self.raw_error = raw_error
        message = (
            f"Tushare query failed endpoint={endpoint} "
            f"params={_format_params(self.params)} "
            f"error={_error_text(raw_error)}"
        )
        super().__init__(message)


@dataclass(frozen=True)
class TushareProClient:
    """Thin wrapper around Tushare Pro query API.

    Import remains lazy so tests and offline environments do not require the
    package unless a live client is actually instantiated.
    """

    token: str

    def __post_init__(self) -> None:
        if not self.token.strip():
            raise ValueError("Tushare token must be non-empty")

    def query(
        self,
        api_name: str,
        *,
        fields: str | None = None,
        limit: int | None = None,
        offset: int | None = None,
        **params: object,
    ) -> pd.DataFrame:
        pro = _build_live_pro(self.token)
        query_params: dict[str, object] = dict(params)
        if fields is not None:
            query_params["fields"] = fields
        if limit is not None:
            query_params["limit"] = int(limit)
        if offset is not None:
            query_params["offset"] = int(offset)
        try:
            result = pro.query(api_name, **query_params)
        except Exception as exc:
            raise TushareQueryError(
                endpoint=api_name,
                params=query_params,
                raw_error=exc,
            ) from exc
        if not isinstance(result, pd.DataFrame):
            raise ValueError(f"Tushare API {api_name!r} did not return a DataFrame")
        return result.copy()


def build_tushare_pro_client(
    *,
    token: str | None = None,
    token_env_var: str = "TUSHARE_TOKEN",
) -> TushareProClient:
    """Construct a live Tushare client from an explicit token or env var."""

    resolved_token = token if token is not None else os.environ.get(token_env_var, "")
    resolved_token = resolved_token.strip()
    if not resolved_token:
        raise ValueError(
            "Tushare token was not provided and environment variable "
            f"{token_env_var!r} is empty"
        )
    return TushareProClient(token=resolved_token)


def _build_live_pro(token: str) -> _TushareProProtocol:
    try:
        import tushare as ts  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "tushare is not installed. Install it in the runtime environment "
            "to use the live ingestion path."
        ) from exc
    return cast(_TushareProProtocol, ts.pro_api(token))


def _format_params(params: Mapping[str, object]) -> str:
    try:
        return json.dumps(dict(params), sort_keys=True, ensure_ascii=False, default=str)
    except TypeError:
        return repr(dict(params))


def _error_text(exc: Exception) -> str:
    return str(exc).strip() or exc.__class__.__name__
