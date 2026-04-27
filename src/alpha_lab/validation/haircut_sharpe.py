"""Haircut Sharpe ratio — multiple-testing adjustment for observed Sharpe.

Complements :func:`alpha_lab.validation.deflated_sharpe_ratio`, which reports
a p-value.  The haircut expresses the same information as an adjusted Sharpe:
the portion of the observed Sharpe that is defensible given how many trials
were implicitly or explicitly searched.

This is not a replacement for DSR — it is a researcher-friendly presentation
of the same quantity.  Use the haircut when explaining the result to a human
reader; use the p-value when applying a threshold in an automated gate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from alpha_lab.validation.deflated_sharpe import expected_max_sharpe


@dataclass(frozen=True)
class HaircutSharpeResult:
    """Output of :func:`haircut_sharpe_ratio`."""

    observed_sharpe: float
    """The raw, unadjusted Sharpe ratio."""

    expected_max_sharpe: float
    """Bailey–López de Prado expected maximum Sharpe under the null,
    given ``n_trials`` and ``n_obs``."""

    haircut_sharpe: float
    """``observed_sharpe - expected_max_sharpe``, floored at zero.  The
    portion of the observed Sharpe that exceeds what random chance across
    ``n_trials`` trials would produce."""

    haircut_ratio: float
    """``haircut_sharpe / observed_sharpe`` when ``observed_sharpe > 0``;
    NaN otherwise.  1.0 means no haircut needed, 0.0 means the observation
    is fully explained by multiple testing."""

    n_trials: int
    """Number of trials assumed in the expected-max computation."""

    n_obs: int
    """Number of observations underlying the Sharpe estimate."""


def haircut_sharpe_ratio(
    observed_sharpe: float,
    *,
    n_trials: int,
    n_obs: int,
) -> HaircutSharpeResult:
    """Return the Sharpe ratio adjusted for multiple-testing effects.

    Parameters
    ----------
    observed_sharpe:
        The un-annualised Sharpe computed from the realised long-short (or
        equivalent) return series.  If you pass an annualised Sharpe, make
        sure ``n_trials`` is chosen consistently — the null benchmark
        depends on trial count, not on annualisation.
    n_trials:
        The number of strategy variants effectively tried.  For a single
        factor with a single parameter setting, ``n_trials=1`` reproduces
        ``observed_sharpe`` exactly.  For parameter-sweep research, set this
        to the grid size.  Community priors (e.g. "everyone has tried
        momentum") suggest at least 10–100 for well-known factor families.
    n_obs:
        Number of observations (typically date count) underlying the
        Sharpe.  Passed through to
        :func:`alpha_lab.validation.expected_max_sharpe` so the null benchmark
        reflects the estimation uncertainty in Sharpe itself.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    if n_obs < 2:
        raise ValueError("n_obs must be >= 2")

    if not math.isfinite(observed_sharpe):
        return HaircutSharpeResult(
            observed_sharpe=float("nan"),
            expected_max_sharpe=float("nan"),
            haircut_sharpe=float("nan"),
            haircut_ratio=float("nan"),
            n_trials=int(n_trials),
            n_obs=int(n_obs),
        )

    benchmark = expected_max_sharpe(n_trials=n_trials, n_obs=n_obs)
    adjusted = max(float(observed_sharpe) - benchmark, 0.0)
    ratio = adjusted / float(observed_sharpe) if float(observed_sharpe) > 0.0 else float("nan")
    return HaircutSharpeResult(
        observed_sharpe=float(observed_sharpe),
        expected_max_sharpe=float(benchmark),
        haircut_sharpe=float(adjusted),
        haircut_ratio=float(ratio),
        n_trials=int(n_trials),
        n_obs=int(n_obs),
    )
