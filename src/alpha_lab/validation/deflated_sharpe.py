from __future__ import annotations

import math

from scipy import stats as scipy_stats

_EULER_GAMMA = 0.5772156649015329


def expected_max_sharpe(n_trials: int, n_obs: int) -> float:
    """Expected maximum Sharpe under the null across ``n_trials`` trials.

    Uses the Bailey & Lopez de Prado approximation:
    ``E[max] ≈ sqrt(2 log n) - (gamma + log(log n)) / (2 sqrt(2 log n))``.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    if n_obs < 2:
        raise ValueError("n_obs must be >= 2")
    if n_trials == 1:
        return 0.0

    log_trials = math.log(float(n_trials))
    if log_trials <= 0.0:
        return 0.0

    lead = math.sqrt(2.0 * log_trials)
    correction = (_EULER_GAMMA + math.log(log_trials)) / (2.0 * lead)
    return float(lead - correction)


def deflated_sharpe_ratio(
    observed_sr: float,
    n_trials: int,
    n_obs: int,
    skewness: float = 0.0,
    excess_kurtosis: float = 0.0,
) -> float:
    """Return the DSR p-value for an observed Sharpe ratio.

    The returned value is the probability that the best of ``n_trials`` zero-alpha
    strategies would achieve a Sharpe at least as large as ``observed_sr`` by chance.
    Larger values imply weaker evidence against backtest overfitting.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    if n_obs < 2:
        raise ValueError("n_obs must be >= 2")
    if not math.isfinite(observed_sr):
        return float("nan")

    benchmark_sr = expected_max_sharpe(n_trials=n_trials, n_obs=n_obs)

    # Lo (2002)-style finite-sample Sharpe variance correction with
    # non-normality terms.  excess_kurtosis -> kurtosis minus one is (excess + 2).
    variance_term = (
        1.0
        - float(skewness) * float(observed_sr)
        + 0.25 * (float(excess_kurtosis) + 2.0) * (float(observed_sr) ** 2)
    )
    if not math.isfinite(variance_term) or variance_term <= 0.0:
        return float("nan")

    sr_std = math.sqrt(variance_term / float(n_obs - 1))
    if not math.isfinite(sr_std) or sr_std <= 0.0:
        return float("nan")

    z_score = (float(observed_sr) - benchmark_sr) / sr_std
    p_value = float(scipy_stats.norm.sf(z_score))
    return min(max(p_value, 0.0), 1.0)
