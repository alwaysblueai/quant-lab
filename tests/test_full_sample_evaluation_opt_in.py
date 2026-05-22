"""OPT-P0-2 — ``allow_full_sample_evaluation`` opt-in matrix.

Pins the three-state contract for ``run_factor_experiment``:

* ``None`` (default, transitional) — preserves legacy full-sample behaviour
  but emits ``FullSampleEvaluationWithoutOptInWarning`` when no split is
  provided.
* ``True`` — explicit opt-in; full-sample evaluation runs without a
  warning. This is what the 6 legal production callers (fast_screen
  tier1/tier2, composite eval main + raw, single-factor dual-scope full
  + IS) now pass.
* ``False`` — explicit opt-out; no-split with this flag raises.
* Any value (``None`` / ``True`` / ``False``) is ignored when a split
  (``train_end`` + ``test_start`` or ``split_contract``) is provided.

Also verifies ``alpha-lab run`` fails fast when ``--train-end`` /
``--test-start`` are missing, before ``run_factor_experiment`` is ever
reached.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from alpha_lab.exceptions import AlphaLabConfigError
from alpha_lab.experiment import (
    FullSampleEvaluationWithoutOptInWarning,
    run_factor_experiment,
)
from alpha_lab.factors.momentum import momentum


def _make_prices(n_assets: int = 6, n_days: int = 30, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    assets = [f"A{i}" for i in range(n_assets)]
    rows = []
    for asset in assets:
        price = 100.0
        for date in dates:
            price *= 1.0 + rng.normal(0.0, 0.01)
            rows.append({"date": date, "asset": asset, "close": price})
    return pd.DataFrame(rows)


def _momentum_fn(prices: pd.DataFrame) -> pd.DataFrame:
    return momentum(prices, window=5)


def _capture_opt_in_warnings(fn):  # type: ignore[no-untyped-def]
    """Run ``fn()`` and return any ``FullSampleEvaluationWithoutOptInWarning``
    instances that were emitted. ``pytest.warns(None)`` was removed in pytest 8,
    so this helper centralizes the catch-warnings boilerplate.
    """
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        result = fn()
    matched = [
        w for w in captured if isinstance(w.message, FullSampleEvaluationWithoutOptInWarning)
    ]
    return result, matched


# ---------------------------------------------------------------------------
# Three-state matrix on the no-split branch
# ---------------------------------------------------------------------------


def test_no_split_omitted_kwarg_emits_deprecation_warning() -> None:
    with pytest.warns(FullSampleEvaluationWithoutOptInWarning):
        result = run_factor_experiment(_make_prices(), _momentum_fn)
    # Legacy behaviour preserved — still produces a result.
    assert result.summary is not None


def test_no_split_explicit_true_is_silent() -> None:
    result, matched = _capture_opt_in_warnings(
        lambda: run_factor_experiment(
            _make_prices(),
            _momentum_fn,
            allow_full_sample_evaluation=True,
        )
    )
    assert matched == [], (
        "allow_full_sample_evaluation=True must not emit "
        f"FullSampleEvaluationWithoutOptInWarning. Got: {[str(w.message) for w in matched]}"
    )
    assert result.summary is not None


def test_no_split_explicit_false_raises() -> None:
    with pytest.raises(AlphaLabConfigError, match="allow_full_sample_evaluation=False"):
        run_factor_experiment(
            _make_prices(),
            _momentum_fn,
            allow_full_sample_evaluation=False,
        )


# ---------------------------------------------------------------------------
# Explicit split: kwarg is ignored across all three states
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("opt_in", [None, True, False])
def test_explicit_split_kwarg_is_a_no_op(opt_in: bool | None) -> None:
    prices = _make_prices()
    # Pick a split that lies inside the synthetic panel.
    train_end = "2024-01-15"
    test_start = "2024-01-22"
    result, matched = _capture_opt_in_warnings(
        lambda: run_factor_experiment(
            prices,
            _momentum_fn,
            train_end=train_end,
            test_start=test_start,
            allow_full_sample_evaluation=opt_in,
        )
    )
    assert matched == [], (
        "An explicit split must suppress the OPT-P0-2 warning regardless of "
        f"allow_full_sample_evaluation; got: {[str(w.message) for w in matched]}"
    )
    assert result.summary is not None


# ---------------------------------------------------------------------------
# CLI ``alpha-lab run`` fail-fast
# ---------------------------------------------------------------------------


def _write_minimal_price_csv(path) -> None:  # type: ignore[no-untyped-def]
    df = _make_prices()
    df.to_csv(path, index=False)


def test_alpha_lab_run_requires_train_and_test(
    tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``alpha-lab run`` without --train-end/--test-start must exit non-zero
    and explain how to get either split-aware or full-sample workflows."""
    from alpha_lab.cli import main

    csv_path = tmp_path / "prices.csv"
    _write_minimal_price_csv(csv_path)

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "run",
                "--input-path",
                str(csv_path),
                "--factor",
                "momentum",
                "--label-horizon",
                "1",
                "--quantiles",
                "5",
            ]
        )
    assert exc_info.value.code != 0
    captured = capsys.readouterr()
    err = captured.err + captured.out
    assert "alpha-lab run requires both --train-end and --test-start" in err
    assert "alpha-lab fast-screen" in err


def test_alpha_lab_run_with_split_does_not_warn(
    tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    """When the split is supplied, ``alpha-lab run`` must not surface the
    OPT-P0-2 deprecation warning."""
    from alpha_lab.cli import main

    csv_path = tmp_path / "prices.csv"
    _write_minimal_price_csv(csv_path)

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    rc, matched = _capture_opt_in_warnings(
        lambda: main(
            [
                "run",
                "--input-path",
                str(csv_path),
                "--factor",
                "momentum",
                "--label-horizon",
                "1",
                "--quantiles",
                "5",
                "--train-end",
                "2024-01-12",
                "--test-start",
                "2024-01-22",
                "--output-dir",
                str(out_dir),
            ]
        )
    )
    assert matched == [], (
        "alpha-lab run with explicit split must not trigger the OPT-P0-2 "
        f"warning. Got: {[str(w.message) for w in matched]}"
    )
    assert rc == 0


# ---------------------------------------------------------------------------
# Production callers that pass ``allow_full_sample_evaluation=True`` must
# not surface the warning. Smoke-test the two pure-Python entry points
# (Tier-1 fast screen and composite evaluate) to lock that contract.
# ---------------------------------------------------------------------------


def test_tier1_fast_screen_does_not_emit_opt_in_warning() -> None:
    from alpha_lab.fast_screen.tier1 import Tier1Inputs, run_tier1

    prices = _make_prices()
    factor_df = _momentum_fn(prices)
    inputs = Tier1Inputs(
        factor_name="momentum_smoke",
        factor_df=factor_df,
        prices=prices,
        horizon=1,
        n_quantiles=5,
    )

    _, matched = _capture_opt_in_warnings(lambda: run_tier1(inputs))
    assert matched == [], (
        "Tier-1 fast screen must pass allow_full_sample_evaluation=True. "
        f"Got: {[str(w.message) for w in matched]}"
    )
