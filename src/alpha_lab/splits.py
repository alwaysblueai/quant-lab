from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from alpha_lab.exceptions import AlphaLabConfigError, AlphaLabDataError
from alpha_lab.research_integrity.contracts import IntegrityCheckResult


@dataclass(frozen=True)
class TimeSeriesSplitContract:
    """Auditable IS/OOS split contract for one chronological experiment."""

    is_start: pd.Timestamp
    is_end: pd.Timestamp
    oos_start: pd.Timestamp
    oos_end: pd.Timestamp
    embargo_days: int
    min_oos_dates: int
    min_is_dates: int
    policy: str
    source: str
    n_dates: int
    n_is_dates: int
    n_oos_dates: int
    target_horizon: int
    rebalance_step: int

    def __post_init__(self) -> None:
        is_start = pd.Timestamp(self.is_start)
        is_end = pd.Timestamp(self.is_end)
        oos_start = pd.Timestamp(self.oos_start)
        oos_end = pd.Timestamp(self.oos_end)
        if is_start > is_end:
            raise AlphaLabConfigError("split contract requires is_start <= is_end")
        if is_end >= oos_start:
            raise AlphaLabConfigError("split contract requires is_end < oos_start")
        if oos_start > oos_end:
            raise AlphaLabConfigError("split contract requires oos_start <= oos_end")
        for name in ("embargo_days", "min_oos_dates", "min_is_dates", "n_dates"):
            if int(getattr(self, name)) < 0:
                raise AlphaLabConfigError(f"split contract field {name} must be non-negative")
        for name in ("target_horizon", "rebalance_step"):
            if int(getattr(self, name)) < 1:
                raise AlphaLabConfigError(f"split contract field {name} must be >= 1")

        object.__setattr__(self, "is_start", is_start)
        object.__setattr__(self, "is_end", is_end)
        object.__setattr__(self, "oos_start", oos_start)
        object.__setattr__(self, "oos_end", oos_end)
        object.__setattr__(self, "embargo_days", int(self.embargo_days))
        object.__setattr__(self, "min_oos_dates", int(self.min_oos_dates))
        object.__setattr__(self, "min_is_dates", int(self.min_is_dates))
        object.__setattr__(self, "n_dates", int(self.n_dates))
        object.__setattr__(self, "n_is_dates", int(self.n_is_dates))
        object.__setattr__(self, "n_oos_dates", int(self.n_oos_dates))
        object.__setattr__(self, "target_horizon", int(self.target_horizon))
        object.__setattr__(self, "rebalance_step", int(self.rebalance_step))

    @property
    def train_end(self) -> pd.Timestamp:
        """Alias used by the existing experiment API."""
        return self.is_end

    @property
    def test_start(self) -> pd.Timestamp:
        """Alias used by the existing experiment API."""
        return self.oos_start

    def to_metadata(self) -> dict[str, object]:
        """Return a JSON-friendly contract snapshot."""
        return {
            "policy": self.policy,
            "source": self.source,
            "is_start": self.is_start.date().isoformat(),
            "is_end": self.is_end.date().isoformat(),
            "oos_start": self.oos_start.date().isoformat(),
            "oos_end": self.oos_end.date().isoformat(),
            "embargo_days": self.embargo_days,
            "min_oos_dates": self.min_oos_dates,
            "min_is_dates": self.min_is_dates,
            "n_dates": self.n_dates,
            "n_is_dates": self.n_is_dates,
            "n_oos_dates": self.n_oos_dates,
            "target_horizon": self.target_horizon,
            "rebalance_step": self.rebalance_step,
        }


def build_strict_split_contract_check_result(
    contract: TimeSeriesSplitContract,
    *,
    object_name: str,
    module_name: str,
    usage_phrase: str = "evaluation",
) -> IntegrityCheckResult:
    """Build the canonical ``strict_time_series_split_contract`` integrity result.

    Both real-case pipelines (single-factor and model-factor) emit this same
    pass-status result to document that a chronological IS/OOS split contract
    was resolved before the downstream step runs. The audit message phrasing
    is line-specific — ``"evaluation"`` for the single-factor backtest, and
    ``"model training"`` for the model-factor training step — so the
    ``usage_phrase`` kwarg preserves that wording without scattering identical
    builder copies across the codebase.

    The function is intentionally pure: it always returns ``status="pass"``
    and ``severity="info"``; callers fold the result into their integrity
    report via ``_record_integrity`` so the resolved IS/OOS window shows up
    in the run audit trail.
    """
    metadata = contract.to_metadata()
    return IntegrityCheckResult(
        check_name="strict_time_series_split_contract",
        status="pass",
        severity="info",
        object_name=object_name,
        module_name=module_name,
        message=(
            f"Strict chronological IS/OOS split resolved before {usage_phrase}: "
            f"IS {metadata['is_start']}..{metadata['is_end']}, "
            f"OOS {metadata['oos_start']}..{metadata['oos_end']}, "
            f"embargo={metadata['embargo_days']}."
        ),
        metrics=metadata,
    )


def rebalance_frequency_to_step(rebalance_frequency: str | int | None) -> int:
    """Resolve a common rebalance label into approximate trading-day steps."""
    if rebalance_frequency is None:
        return 1
    if isinstance(rebalance_frequency, int):
        if rebalance_frequency <= 0:
            raise AlphaLabConfigError("rebalance_frequency integer must be positive")
        return int(rebalance_frequency)

    text = str(rebalance_frequency).strip().upper()
    if not text:
        return 1
    try:
        explicit = int(text)
    except ValueError:
        explicit = 0
    if explicit > 0:
        return explicit

    mapping = {
        "D": 1,
        "DAILY": 1,
        "W": 5,
        "WEEKLY": 5,
        "M": 21,
        "MONTHLY": 21,
        "Q": 63,
        "QUARTERLY": 63,
        "Y": 252,
        "A": 252,
        "YEARLY": 252,
        "ANNUAL": 252,
    }
    if text in mapping:
        return mapping[text]

    unit = text[-1]
    prefix = text[:-1]
    if prefix.isdigit() and unit in {"D", "W", "M", "Q", "Y", "A"}:
        multiplier = {"D": 1, "W": 5, "M": 21, "Q": 63, "Y": 252, "A": 252}[unit]
        return max(1, int(prefix) * multiplier)

    raise AlphaLabConfigError(
        f"unsupported rebalance_frequency {rebalance_frequency!r}; "
        "expected D/W/M/Q/Y, an integer step, or labels like 5D/2W."
    )


def infer_default_time_series_split_contract(
    dates: pd.Series | np.ndarray,
    *,
    target_horizon: int = 1,
    rebalance_frequency: str | int | None = None,
    rebalance_step: int | None = None,
    min_oos_dates: int = 60,
    min_is_dates: int = 60,
    policy: str = "auto_holdout_v1",
    source: str = "auto",
) -> TimeSeriesSplitContract:
    """Infer the default chronological IS/OOS contract for Level 1/2 runs.

    The OOS window is roughly the final 20% of the date axis.  For long
    histories (five years or more) it is bounded to about 1.5-2 trading years.
    An embargo gap of ``max(target_horizon, rebalance_step)`` unique dates is
    inserted between IS and OOS.  The raw OOS window includes an extra
    ``target_horizon`` dates so at least ``min_oos_dates`` dates can carry
    finite forward-return labels.
    """
    horizon = int(target_horizon)
    if horizon < 1:
        raise AlphaLabConfigError("target_horizon must be >= 1")
    if rebalance_step is None:
        step = rebalance_frequency_to_step(rebalance_frequency)
    else:
        step = int(rebalance_step)
        if step < 1:
            raise AlphaLabConfigError("rebalance_step must be >= 1")
    min_oos = int(min_oos_dates)
    min_is = int(min_is_dates)
    if min_oos < 1:
        raise AlphaLabConfigError("min_oos_dates must be >= 1")
    if min_is < 1:
        raise AlphaLabConfigError("min_is_dates must be >= 1")

    unique_dates = _unique_sorted_dates(dates)
    n_dates = int(len(unique_dates))
    embargo = max(horizon, step)
    min_oos_window = min_oos + horizon
    required = min_is + embargo + min_oos_window
    if n_dates < required:
        raise AlphaLabConfigError(
            "not enough unique dates for strict IS/OOS split: "
            f"got {n_dates}, need at least {required} "
            f"(min_is_dates={min_is}, embargo_days={embargo}, "
            f"min_oos_dates={min_oos}, target_horizon={horizon})."
        )

    desired_oos = _default_oos_window_size(n_dates, min_oos_dates=min_oos)
    max_oos = n_dates - min_is - embargo
    oos_len = min(max(desired_oos, min_oos_window), max_oos)
    if oos_len < min_oos_window:
        raise AlphaLabConfigError(
            "not enough unique dates after embargo for strict OOS window: "
            f"got max_oos={max_oos}, min_oos_dates={min_oos}, "
            f"target_horizon={horizon}."
        )

    oos_start_idx = n_dates - oos_len
    is_end_idx = oos_start_idx - embargo - 1
    if is_end_idx < min_is - 1:
        raise AlphaLabConfigError("strict split inference produced too few IS dates")

    return TimeSeriesSplitContract(
        is_start=pd.Timestamp(unique_dates[0]),
        is_end=pd.Timestamp(unique_dates[is_end_idx]),
        oos_start=pd.Timestamp(unique_dates[oos_start_idx]),
        oos_end=pd.Timestamp(unique_dates[-1]),
        embargo_days=embargo,
        min_oos_dates=min_oos,
        min_is_dates=min_is,
        policy=policy,
        source=source,
        n_dates=n_dates,
        n_is_dates=is_end_idx + 1,
        n_oos_dates=oos_len,
        target_horizon=horizon,
        rebalance_step=step,
    )


def preflight_split_contract(
    dates: pd.Series | np.ndarray,
    *,
    target_horizon: int = 1,
    rebalance_frequency: str | int | None = None,
    rebalance_step: int | None = None,
    min_oos_dates: int = 60,
    min_is_dates: int = 60,
    policy: str = "auto_holdout_v1",
    source: str = "preflight",
) -> tuple[TimeSeriesSplitContract | None, tuple[str, ...]]:
    """Return a strict split contract or actionable remediation messages."""
    try:
        return (
            infer_default_time_series_split_contract(
                dates,
                target_horizon=target_horizon,
                rebalance_frequency=rebalance_frequency,
                rebalance_step=rebalance_step,
                min_oos_dates=min_oos_dates,
                min_is_dates=min_is_dates,
                policy=policy,
                source=source,
            ),
            (),
        )
    except (AlphaLabConfigError, AlphaLabDataError, ValueError) as exc:
        return None, _split_preflight_remediations(
            str(exc),
            target_horizon=target_horizon,
            rebalance_frequency=rebalance_frequency,
            rebalance_step=rebalance_step,
            min_oos_dates=min_oos_dates,
            min_is_dates=min_is_dates,
        )


def _split_preflight_remediations(
    error: str,
    *,
    target_horizon: int,
    rebalance_frequency: str | int | None,
    rebalance_step: int | None,
    min_oos_dates: int,
    min_is_dates: int,
) -> tuple[str, ...]:
    try:
        step_value = (
            rebalance_step
            if rebalance_step is not None
            else rebalance_frequency_to_step(rebalance_frequency)
        )
        step_text = str(step_value)
    except AlphaLabConfigError:
        step_text = str(rebalance_frequency)
    return (
        f"strict IS/OOS split preflight failed: {error}",
        "扩充 prices 数据时间范围，确保按股票池过滤后的唯一交易日足够覆盖 IS、embargo 和 OOS。",
        (
            "如这是短窗口调试，请降低 target.horizon 或 rebalance_frequency；"
            f"当前 target_horizon={int(target_horizon)}, rebalance_step={step_text}。"
        ),
        (
            "完整研究默认至少需要 "
            f"min_is_dates={int(min_is_dates)}、min_oos_dates={int(min_oos_dates)}，"
            "并保留 max(horizon, rebalance_step) 个交易日 embargo。"
        ),
    )


def _unique_sorted_dates(dates: pd.Series | np.ndarray) -> pd.DatetimeIndex:
    dates_ts = pd.DatetimeIndex(pd.to_datetime(pd.Index(dates)))
    if pd.isna(dates_ts).any():
        raise AlphaLabDataError("dates contains NaT values; all dates must be valid timestamps")
    unique_dates = pd.DatetimeIndex(dates_ts.unique()).sort_values()
    if unique_dates.empty:
        raise AlphaLabDataError("dates must contain at least one timestamp")
    return unique_dates


def _default_oos_window_size(n_dates: int, *, min_oos_dates: int) -> int:
    if n_dates >= 252 * 5:
        return min(max(round(n_dates * 0.2), 378, min_oos_dates), 504)
    return max(round(n_dates * 0.2), min_oos_dates)


def time_split(
    dates: pd.Series | np.ndarray,
    train_end: str | pd.Timestamp,
    test_start: str | pd.Timestamp,
    val_start: str | pd.Timestamp | None = None,
) -> dict[str, np.ndarray]:
    """Split dates into train / (optional) validation / test boolean masks.

    All splits are strictly time-ordered with no overlap.  The function
    returns boolean masks rather than modified data so the caller controls
    how the masks are applied.

    Parameters
    ----------
    dates:
        1-D array-like of dates.  Need not be sorted.
    train_end:
        Last date (inclusive) of the training set.
    test_start:
        First date (inclusive) of the test set.  Must be strictly after
        ``train_end``.
    val_start:
        First date (inclusive) of the validation set.  Must satisfy
        ``train_end < val_start <= test_start``.  Dates in
        ``[val_start, test_start)`` are validation; dates in
        ``(train_end, val_start)`` are in neither split (gap).
        If ``None`` no validation mask is returned.

    Returns
    -------
    dict
        Maps ``'train'`` and ``'test'`` (and ``'val'`` when requested) to
        boolean numpy arrays of shape ``(len(dates),)``.  The masks are
        mutually exclusive: at most one mask is True at any index position.
    """
    # pd.to_datetime(dates) returns a Series (if dates is Series) or
    # DatetimeIndex (if array-like) — both support boolean comparison and
    # .to_numpy().  Avoid np.asarray() here; it produces a raw ndarray that
    # loses the .to_numpy() method in pandas 2.x.
    dates_ts = pd.to_datetime(dates)
    if pd.isna(dates_ts).any():
        raise AlphaLabDataError("dates contains NaT values; all dates must be valid timestamps")
    train_end_ts = pd.Timestamp(train_end)
    test_start_ts = pd.Timestamp(test_start)

    if train_end_ts >= test_start_ts:
        raise AlphaLabConfigError(
            f"train_end ({train_end_ts.date()}) must be strictly before "
            f"test_start ({test_start_ts.date()})"
        )

    train_mask: np.ndarray = (dates_ts <= train_end_ts).to_numpy(dtype=bool)
    test_mask: np.ndarray = (dates_ts >= test_start_ts).to_numpy(dtype=bool)

    if val_start is not None:
        val_start_ts = pd.Timestamp(val_start)
        if val_start_ts <= train_end_ts:
            raise AlphaLabConfigError(
                f"val_start ({val_start_ts.date()}) must be strictly after "
                f"train_end ({train_end_ts.date()})"
            )
        if val_start_ts > test_start_ts:
            raise AlphaLabConfigError(
                f"val_start ({val_start_ts.date()}) must be at or before "
                f"test_start ({test_start_ts.date()})"
            )
        val_mask: np.ndarray = ((dates_ts >= val_start_ts) & (dates_ts < test_start_ts)).to_numpy(
            dtype=bool
        )
        return {"train": train_mask, "val": val_mask, "test": test_mask}

    return {"train": train_mask, "test": test_mask}


def walk_forward_split(
    dates: pd.Series | np.ndarray,
    train_size: int,
    test_size: int,
    step: int,
    val_size: int = 0,
) -> list[dict[str, np.ndarray]]:
    """Generate rolling walk-forward train / (optional) val / test masks.

    Each fold is strictly time-ordered: the test window immediately follows
    the training window with no overlap.  Folds advance by ``step`` rows
    between iterations.

    Parameters
    ----------
    dates:
        1-D array of dates in chronological order.  **Must be sorted.**
    train_size:
        Total number of rows in the training window, including the optional
        validation tail.
    test_size:
        Number of rows in the test window.
    step:
        Number of rows to advance the window start between folds.
    val_size:
        Number of rows at the *end* of the training window to reserve for
        validation.  These rows are excluded from the ``'train'`` mask and
        placed in the ``'val'`` mask.  0 (default) means no validation.

    Returns
    -------
    list of dict
        Each element maps ``'train'``, ``'test'``, and optionally ``'val'``
        to boolean numpy arrays of shape ``(len(dates),)``.
        Folds where the test window would exceed the available data are not
        generated.
    """
    if train_size <= 0 or test_size <= 0 or step <= 0:
        raise AlphaLabConfigError("train_size, test_size, and step must be positive integers")
    if val_size < 0:
        raise AlphaLabConfigError("val_size must be non-negative")
    if val_size >= train_size:
        raise AlphaLabConfigError("val_size must be less than train_size")

    dates_arr = np.asarray(pd.to_datetime(np.asarray(dates)))
    n = len(dates_arr)

    if pd.isnull(dates_arr).any():
        raise AlphaLabDataError("dates contains NaT values; all dates must be valid timestamps")

    if n > 1 and not np.all(dates_arr[:-1] <= dates_arr[1:]):
        raise AlphaLabDataError("dates must be sorted in chronological order")

    # Repeated dates indicate panel data (many assets per date).  Row-based
    # slicing would split a shared timestamp across folds, creating leakage.
    # Pass unique dates instead: e.g. df["date"].drop_duplicates().sort_values()
    if len(dates_arr) != len(np.unique(dates_arr)):
        raise AlphaLabDataError(
            "dates contains repeated values. walk_forward_split() operates on "
            "unique dates. For panel data pass the unique sorted date axis: "
            "e.g. df['date'].drop_duplicates().sort_values().reset_index(drop=True)"
        )

    splits: list[dict[str, np.ndarray]] = []
    train_start = 0

    while True:
        train_end = train_start + train_size  # exclusive index
        test_end = train_end + test_size  # exclusive index

        if test_end > n:
            break

        train_mask = np.zeros(n, dtype=bool)
        train_mask[train_start : train_end - val_size] = True

        test_mask = np.zeros(n, dtype=bool)
        test_mask[train_end:test_end] = True

        fold: dict[str, np.ndarray] = {"train": train_mask, "test": test_mask}

        if val_size > 0:
            val_mask = np.zeros(n, dtype=bool)
            val_mask[train_end - val_size : train_end] = True
            fold["val"] = val_mask

        splits.append(fold)
        train_start += step

    return splits
