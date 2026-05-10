from __future__ import annotations

import math
from typing import Any, cast

import pandas as pd


def _date_text(value: object) -> str | None:
    if value is None:
        return None
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return str(value)
    return str(pd.Timestamp(timestamp).date().isoformat())


def _finite_or_none(value: object) -> float | None:
    if value is None:
        return None
    try:
        number = float(cast(Any, value))
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _jsonable_scalar(value: object) -> object:
    if value is None:
        return None
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _model_factor_decay_horizons(target_horizon: int) -> tuple[int, ...]:
    horizons = {1, 2, 3, 5, 10, 20}
    if target_horizon > 0:
        horizons.add(int(target_horizon))
    return tuple(sorted(horizons))
