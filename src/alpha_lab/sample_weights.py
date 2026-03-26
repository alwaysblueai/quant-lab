from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SampleWeightResult:
    """Container for sample-weight components and final combined weights."""

    weights: pd.DataFrame
    metadata: dict[str, object]


def concurrency_by_date(
    events: pd.DataFrame,
    *,
    start_col: str = "event_start",
    end_col: str = "event_end",
    calendar: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """Compute event concurrency count for each date on a calendar."""
    frame = _prepare_events(events, start_col=start_col, end_col=end_col)
    cal = _resolve_calendar(frame, calendar=calendar)
    indicator = _event_indicator_matrix(frame, cal)
    conc = indicator.sum(axis=0).astype(float)
    return pd.DataFrame({"date": cal, "concurrency": conc})


def uniqueness_weights(
    events: pd.DataFrame,
    *,
    sample_id_col: str = "sample_id",
    start_col: str = "event_start",
    end_col: str = "event_end",
    calendar: pd.DatetimeIndex | None = None,
) -> pd.Series:
    """Average uniqueness per event: mean(1 / concurrency) over active dates."""
    frame = _prepare_events(
        events,
        start_col=start_col,
        end_col=end_col,
        sample_id_col=sample_id_col,
    )
    cal = _resolve_calendar(frame, calendar=calendar)
    indicator = _event_indicator_matrix(frame, cal)
    concurrency = indicator.sum(axis=0).astype(float)
    inv_conc = np.divide(1.0, concurrency, out=np.zeros_like(concurrency), where=concurrency > 0)

    weights = np.zeros(len(frame), dtype=float)
    for i in range(len(frame)):
        active = indicator[i]
        if active.any():
            weights[i] = float(inv_conc[active].mean())
        else:
            weights[i] = 0.0
    out = pd.Series(weights, index=frame[sample_id_col], dtype=float, name="uniqueness_weight")
    return _normalize_nonnegative(out)


def return_magnitude_weights(
    values: pd.Series,
    *,
    clip_quantile: float = 0.99,
) -> pd.Series:
    """Absolute-return style weights with robust clipping."""
    if clip_quantile <= 0 or clip_quantile > 1:
        raise ValueError("clip_quantile must be in (0, 1]")
    s = pd.to_numeric(values, errors="coerce").abs()
    cap = float(s.quantile(clip_quantile)) if s.notna().any() else 0.0
    if np.isfinite(cap) and cap > 0:
        s = s.clip(upper=cap)
    s = s.fillna(0.0)
    s.name = "return_magnitude_weight"
    return _normalize_nonnegative(s)


def time_decay_weights(
    decision_dates: pd.Series,
    *,
    half_life_periods: float,
    reference_date: pd.Timestamp | None = None,
) -> pd.Series:
    """Exponential time-decay weights on the decision-date axis."""
    if half_life_periods <= 0:
        raise ValueError("half_life_periods must be > 0")
    dates = pd.to_datetime(decision_dates, errors="coerce")
    if dates.isna().any():
        raise ValueError("decision_dates contains invalid timestamps")

    ordered = pd.Index(dates.drop_duplicates().sort_values())
    rank = dates.map({d: i for i, d in enumerate(ordered)})
    if reference_date is None:
        max_rank = int(rank.max())
    else:
        ref = pd.Timestamp(reference_date)
        max_rank = int(ordered.searchsorted(ref, side="right") - 1)
        max_rank = max(max_rank, 0)
    age = (max_rank - rank).clip(lower=0).astype(float)
    decay = np.exp(-np.log(2.0) * age / float(half_life_periods))
    out = pd.Series(decay, index=decision_dates.index, dtype=float, name="time_decay_weight")
    return _normalize_nonnegative(out)


def confidence_weights(
    confidence: pd.Series,
    *,
    min_weight: float = 0.1,
    max_weight: float = 1.0,
) -> pd.Series:
    """Map confidence-like values to bounded non-negative weights."""
    if min_weight < 0:
        raise ValueError("min_weight must be >= 0")
    if max_weight <= min_weight:
        raise ValueError("max_weight must be > min_weight")

    s = pd.to_numeric(confidence, errors="coerce")
    lo = float(s.min()) if s.notna().any() else 0.0
    hi = float(s.max()) if s.notna().any() else 0.0
    if np.isclose(hi, lo):
        scaled = pd.Series(1.0, index=s.index, dtype=float)
    else:
        scaled = (s - lo) / (hi - lo)
    scaled = scaled.fillna(0.0)
    bounded = min_weight + (max_weight - min_weight) * scaled
    bounded.name = "confidence_weight"
    return _normalize_nonnegative(bounded)


def combine_weight_components(
    components: dict[str, pd.Series],
    *,
    normalize: bool = True,
) -> pd.Series:
    """Combine weight components multiplicatively with aligned indices."""
    if not components:
        raise ValueError("components must be non-empty")
    aligned = pd.concat(components, axis=1)
    if aligned.empty:
        raise ValueError("components have no aligned rows")
    aligned = aligned.fillna(1.0)
    if (aligned < 0).any().any():
        raise ValueError("all components must be non-negative")

    combined = aligned.prod(axis=1)
    combined.name = "sample_weight"
    if normalize:
        return _normalize_nonnegative(combined)
    return combined


def build_sample_weights(
    events: pd.DataFrame,
    *,
    sample_id_col: str = "sample_id",
    decision_col: str = "date",
    start_col: str = "event_start",
    end_col: str = "event_end",
    return_col: str | None = None,
    confidence_col: str | None = None,
    half_life_periods: float | None = None,
    calendar: pd.DatetimeIndex | None = None,
) -> SampleWeightResult:
    """Build a canonical sample-weight table from event metadata."""
    frame = _prepare_events(
        events,
        start_col=start_col,
        end_col=end_col,
        sample_id_col=sample_id_col,
        decision_col=decision_col,
    )

    comp: dict[str, pd.Series] = {}
    comp["uniqueness"] = uniqueness_weights(
        frame,
        sample_id_col=sample_id_col,
        start_col="event_start",
        end_col="event_end",
        calendar=calendar,
    )
    if return_col is not None:
        if return_col not in frame.columns:
            raise ValueError(f"events missing return_col {return_col!r}")
        comp["return_magnitude"] = return_magnitude_weights(
            frame.set_index(sample_id_col)[return_col]
        )
    if confidence_col is not None:
        if confidence_col not in frame.columns:
            raise ValueError(f"events missing confidence_col {confidence_col!r}")
        comp["confidence"] = confidence_weights(
            frame.set_index(sample_id_col)[confidence_col]
        )
    if half_life_periods is not None:
        comp["time_decay"] = time_decay_weights(
            frame.set_index(sample_id_col)["decision_date"],
            half_life_periods=half_life_periods,
        )

    combined = combine_weight_components(comp)
    out = frame[[sample_id_col, "decision_date", "event_start", "event_end"]].copy()
    out = out.set_index(sample_id_col)
    for name, series in comp.items():
        out[f"weight_{name}"] = series.reindex(out.index)
    out["sample_weight"] = combined.reindex(out.index)
    out = out.reset_index().sort_values(sample_id_col, kind="mergesort").reset_index(drop=True)

    return SampleWeightResult(
        weights=out,
        metadata={
            "schema_version": "1.0.0",
            "sample_id_col": sample_id_col,
            "return_col": return_col,
            "confidence_col": confidence_col,
            "half_life_periods": half_life_periods,
        },
    )


def _prepare_events(
    events: pd.DataFrame,
    *,
    start_col: str,
    end_col: str,
    sample_id_col: str = "sample_id",
    decision_col: str = "date",
) -> pd.DataFrame:
    frame = events.copy()
    if sample_id_col not in frame.columns:
        frame[sample_id_col] = np.arange(len(frame), dtype=int)
    if frame.duplicated(subset=[sample_id_col]).any():
        raise ValueError(f"events has duplicate {sample_id_col!r} values")

    if decision_col in frame.columns:
        frame["decision_date"] = pd.to_datetime(frame[decision_col], errors="coerce")
    else:
        frame["decision_date"] = pd.NaT
    if start_col in frame.columns:
        frame["event_start"] = pd.to_datetime(frame[start_col], errors="coerce")
    else:
        frame["event_start"] = frame["decision_date"]
    if end_col in frame.columns:
        frame["event_end"] = pd.to_datetime(frame[end_col], errors="coerce")
    else:
        frame["event_end"] = frame["decision_date"]
    frame["decision_date"] = frame["decision_date"].fillna(frame["event_start"])

    if frame["event_start"].isna().any() or frame["event_end"].isna().any():
        raise ValueError("events contains invalid event_start/event_end timestamps")
    if frame["decision_date"].isna().any():
        raise ValueError("events contains invalid decision dates")
    if (frame["event_end"] < frame["event_start"]).any():
        raise ValueError("events has event_end < event_start")

    return frame.sort_values(
        ["decision_date", "event_start", "event_end", sample_id_col],
        kind="mergesort",
    ).reset_index(drop=True)


def _resolve_calendar(
    frame: pd.DataFrame,
    *,
    calendar: pd.DatetimeIndex | None,
) -> pd.DatetimeIndex:
    if calendar is not None:
        cal = pd.DatetimeIndex(pd.to_datetime(pd.Index(calendar), errors="coerce")).dropna()
        if len(cal) == 0:
            raise ValueError("calendar is empty after datetime parsing")
        return cal.sort_values().unique()

    start = pd.Timestamp(frame["event_start"].min())
    end = pd.Timestamp(frame["event_end"].max())
    return pd.date_range(start, end, freq="B")


def _event_indicator_matrix(frame: pd.DataFrame, calendar: pd.DatetimeIndex) -> np.ndarray:
    starts = frame["event_start"].to_numpy(dtype="datetime64[ns]")
    ends = frame["event_end"].to_numpy(dtype="datetime64[ns]")
    cal = calendar.to_numpy(dtype="datetime64[ns]")
    out = (starts[:, None] <= cal[None, :]) & (ends[:, None] >= cal[None, :])
    return np.asarray(out, dtype=bool)


def _normalize_nonnegative(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").fillna(0.0).clip(lower=0.0)
    total = float(s.sum())
    if total <= 0:
        if len(s) == 0:
            return s.astype(float)
        return pd.Series(np.full(len(s), 1.0 / len(s)), index=s.index, dtype=float, name=s.name)
    return (s / total).astype(float)
