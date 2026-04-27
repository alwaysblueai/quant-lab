"""Data ingestion adapters for external data vendors."""

from .baostock_adapter import generate_real_case_inputs as generate_baostock_real_case_inputs
from .tushare_adapter import (
    GeneratedRealCaseInputs,
    build_bp_factor,
    build_roe_factor,
    build_universe,
    fetch_fundamentals,
    fetch_prices,
    generate_real_case_inputs,
)

__all__ = [
    "GeneratedRealCaseInputs",
    "build_bp_factor",
    "build_roe_factor",
    "build_universe",
    "fetch_fundamentals",
    "fetch_prices",
    "generate_baostock_real_case_inputs",
    "generate_real_case_inputs",
]
