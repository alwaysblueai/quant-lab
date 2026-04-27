"""Lightweight uncertainty estimates for Level 1 factor diagnostics."""

from __future__ import annotations

import math
import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from alpha_lab.research_evaluation_config import (
    DEFAULT_RESEARCH_EVALUATION_CONFIG,
    UncertaintyConfig,
)

CoreUncertaintyThresholds = UncertaintyConfig
DEFAULT_CORE_UNCERTAINTY_THRESHOLDS = DEFAULT_RESEARCH_EVALUATION_CONFIG.uncertainty

# Below this many resamples, percentile-bootstrap quantiles are noisy enough
# that downstream CIs should be treated as indicative only. The threshold is
# a rule-of-thumb floor (e.g., Efron & Tibshirani 1993 §13.3) — calls below
# it still succeed, but emit a UserWarning so reviewers notice.
_BOOTSTRAP_RESAMPLE_WARN_THRESHOLD = 200


@dataclass(frozen=True)
class MeanCI:
    mean: float
    stderr: float
    ci_lower: float
    ci_upper: float
    n_obs: int

    @property
    def has_finite_ci(self) -> bool:
        return math.isfinite(self.ci_lower) and math.isfinite(self.ci_upper)

    @property
    def overlaps_zero(self) -> bool:
        return self.has_finite_ci and self.ci_lower <= 0.0 <= self.ci_upper

    @property
    def half_width(self) -> float:
        if not self.has_finite_ci:
            return float("nan")
        return 0.5 * (self.ci_upper - self.ci_lower)


@dataclass(frozen=True)
class CoreUncertaintySummary:
    mean_ic_ci_lower: float
    mean_ic_ci_upper: float
    mean_rank_ic_ci_lower: float
    mean_rank_ic_ci_upper: float
    mean_long_short_return_ci_lower: float
    mean_long_short_return_ci_upper: float
    uncertainty_flags: tuple[str, ...]
    uncertainty_method: str
    uncertainty_confidence_level: float
    uncertainty_bootstrap_resamples: int | None
    uncertainty_bootstrap_block_length: int | None

    def to_dict(self) -> dict[str, object]:
        return {
            "mean_ic_ci_lower": self.mean_ic_ci_lower,
            "mean_ic_ci_upper": self.mean_ic_ci_upper,
            "mean_rank_ic_ci_lower": self.mean_rank_ic_ci_lower,
            "mean_rank_ic_ci_upper": self.mean_rank_ic_ci_upper,
            "mean_long_short_return_ci_lower": self.mean_long_short_return_ci_lower,
            "mean_long_short_return_ci_upper": self.mean_long_short_return_ci_upper,
            "uncertainty_flags": self.uncertainty_flags,
            "uncertainty_method": self.uncertainty_method,
            "uncertainty_confidence_level": self.uncertainty_confidence_level,
            "uncertainty_bootstrap_resamples": self.uncertainty_bootstrap_resamples,
            "uncertainty_bootstrap_block_length": self.uncertainty_bootstrap_block_length,
        }


def compute_mean_ci(
    values: Iterable[float] | Sequence[float] | pd.Series,
    *,
    confidence_level: float = 0.95,
    use_t_for_small_sample: bool = False,
    small_sample_threshold: int = 30,
) -> MeanCI:
    """Compute mean CI from finite observations.

    Uses a sample standard error (`ddof=1`) and normal quantile by default.
    Optionally, when ``use_t_for_small_sample=True`` and ``n_obs`` is below
    ``small_sample_threshold``, uses Student-t critical values.
    For fewer than 2 observations, CI and stderr are NaN.
    """

    cleaned = _finite_values(values)
    n_obs = len(cleaned)
    if n_obs == 0:
        return MeanCI(
            mean=float("nan"),
            stderr=float("nan"),
            ci_lower=float("nan"),
            ci_upper=float("nan"),
            n_obs=0,
        )

    mean = float(cleaned.mean())
    if n_obs < 2:
        return MeanCI(
            mean=mean,
            stderr=float("nan"),
            ci_lower=float("nan"),
            ci_upper=float("nan"),
            n_obs=n_obs,
        )

    std = float(cleaned.std(ddof=1))
    stderr = std / math.sqrt(float(n_obs))
    if not math.isfinite(stderr):
        return MeanCI(
            mean=mean,
            stderr=float("nan"),
            ci_lower=float("nan"),
            ci_upper=float("nan"),
            n_obs=n_obs,
        )

    critical_value = _critical_value(
        confidence_level=confidence_level,
        n_obs=n_obs,
        use_t_for_small_sample=use_t_for_small_sample,
        small_sample_threshold=small_sample_threshold,
    )
    half_width = critical_value * stderr
    return MeanCI(
        mean=mean,
        stderr=stderr,
        ci_lower=mean - half_width,
        ci_upper=mean + half_width,
        n_obs=n_obs,
    )


def compute_bootstrap_mean_ci(
    values: Iterable[float] | Sequence[float] | pd.Series,
    *,
    confidence_level: float = 0.95,
    n_resamples: int = 400,
    random_seed: int | None = 7,
) -> MeanCI:
    """Compute percentile-bootstrap CI for the mean from finite observations."""

    cleaned = _finite_values(values)
    n_obs = len(cleaned)
    if n_obs == 0:
        return MeanCI(
            mean=float("nan"),
            stderr=float("nan"),
            ci_lower=float("nan"),
            ci_upper=float("nan"),
            n_obs=0,
        )

    mean = float(cleaned.mean())
    if n_obs < 2:
        return MeanCI(
            mean=mean,
            stderr=float("nan"),
            ci_lower=float("nan"),
            ci_upper=float("nan"),
            n_obs=n_obs,
        )

    _validate_confidence_level(confidence_level)
    if n_resamples < 2:
        raise ValueError("bootstrap n_resamples must be >= 2")
    _maybe_warn_small_resamples(n_resamples, kind="bootstrap")

    arr = cleaned.to_numpy(dtype=float)
    rng = np.random.default_rng(random_seed)
    draws = rng.choice(arr, size=(n_resamples, n_obs), replace=True)
    resampled_means = draws.mean(axis=1)
    finite_means = resampled_means[np.isfinite(resampled_means)]
    if len(finite_means) < 2:
        return MeanCI(
            mean=mean,
            stderr=float("nan"),
            ci_lower=float("nan"),
            ci_upper=float("nan"),
            n_obs=n_obs,
        )

    alpha = 0.5 * (1.0 - confidence_level)
    ci_lower = float(np.quantile(finite_means, alpha))
    ci_upper = float(np.quantile(finite_means, 1.0 - alpha))
    stderr = float(np.std(finite_means, ddof=1))
    if not math.isfinite(stderr):
        stderr = float("nan")
    return MeanCI(
        mean=mean,
        stderr=stderr,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        n_obs=n_obs,
    )


def compute_block_bootstrap_mean_ci(
    values: Iterable[float] | Sequence[float] | pd.Series,
    *,
    confidence_level: float = 0.95,
    n_resamples: int = 400,
    block_length: int = 5,
    random_seed: int | None = 7,
) -> MeanCI:
    """Compute contiguous-block bootstrap CI for the mean from finite observations."""

    cleaned = _finite_values(values)
    n_obs = len(cleaned)
    if n_obs == 0:
        return MeanCI(
            mean=float("nan"),
            stderr=float("nan"),
            ci_lower=float("nan"),
            ci_upper=float("nan"),
            n_obs=0,
        )

    mean = float(cleaned.mean())
    if n_obs < 2:
        return MeanCI(
            mean=mean,
            stderr=float("nan"),
            ci_lower=float("nan"),
            ci_upper=float("nan"),
            n_obs=n_obs,
        )

    _validate_confidence_level(confidence_level)
    if n_resamples < 2:
        raise ValueError("block bootstrap n_resamples must be >= 2")
    if block_length < 1:
        raise ValueError("block bootstrap block_length must be >= 1")
    _maybe_warn_small_resamples(n_resamples, kind="block_bootstrap")

    arr = cleaned.to_numpy(dtype=float)
    effective_block_length = min(int(block_length), n_obs)
    n_blocks = int(math.ceil(n_obs / float(effective_block_length)))
    max_start = n_obs - effective_block_length

    rng = np.random.default_rng(random_seed)
    starts = rng.integers(0, max_start + 1, size=(n_resamples, n_blocks))
    offsets = np.arange(effective_block_length, dtype=int)
    draw_indices = (starts[..., None] + offsets).reshape(
        n_resamples, n_blocks * effective_block_length
    )[:, :n_obs]

    resampled_means = arr[draw_indices].mean(axis=1)
    finite_means = resampled_means[np.isfinite(resampled_means)]
    if len(finite_means) < 2:
        return MeanCI(
            mean=mean,
            stderr=float("nan"),
            ci_lower=float("nan"),
            ci_upper=float("nan"),
            n_obs=n_obs,
        )

    alpha = 0.5 * (1.0 - confidence_level)
    ci_lower = float(np.quantile(finite_means, alpha))
    ci_upper = float(np.quantile(finite_means, 1.0 - alpha))
    stderr = float(np.std(finite_means, ddof=1))
    if not math.isfinite(stderr):
        stderr = float("nan")
    return MeanCI(
        mean=mean,
        stderr=stderr,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        n_obs=n_obs,
    )


def compute_core_uncertainty(
    *,
    ic_values: Iterable[float] | Sequence[float] | pd.Series,
    rank_ic_values: Iterable[float] | Sequence[float] | pd.Series,
    long_short_values: Iterable[float] | Sequence[float] | pd.Series,
    thresholds: CoreUncertaintyThresholds = DEFAULT_CORE_UNCERTAINTY_THRESHOLDS,
) -> CoreUncertaintySummary:
    """Compute core uncertainty outputs and warning flags."""

    method = _resolve_uncertainty_method(thresholds.method)
    if method in {"bootstrap", "block_bootstrap"}:
        confidence_level = _resolve_bootstrap_confidence_level(thresholds=thresholds)
        n_resamples = int(thresholds.bootstrap_resamples)
        if n_resamples < 2:
            raise ValueError("bootstrap_resamples must be >= 2")
        seed = thresholds.bootstrap_random_seed

        block_length_out: int | None = None
        if method == "bootstrap":
            ic_ci = compute_bootstrap_mean_ci(
                ic_values,
                confidence_level=confidence_level,
                n_resamples=n_resamples,
                random_seed=_seed_for_metric(seed, 0),
            )
            rank_ic_ci = compute_bootstrap_mean_ci(
                rank_ic_values,
                confidence_level=confidence_level,
                n_resamples=n_resamples,
                random_seed=_seed_for_metric(seed, 1),
            )
            long_short_ci = compute_bootstrap_mean_ci(
                long_short_values,
                confidence_level=confidence_level,
                n_resamples=n_resamples,
                random_seed=_seed_for_metric(seed, 2),
            )
        else:
            block_length = int(thresholds.block_bootstrap_block_length)
            if block_length < 1:
                raise ValueError("block_bootstrap_block_length must be >= 1")
            ic_ci = compute_block_bootstrap_mean_ci(
                ic_values,
                confidence_level=confidence_level,
                n_resamples=n_resamples,
                block_length=block_length,
                random_seed=_seed_for_metric(seed, 0),
            )
            rank_ic_ci = compute_block_bootstrap_mean_ci(
                rank_ic_values,
                confidence_level=confidence_level,
                n_resamples=n_resamples,
                block_length=block_length,
                random_seed=_seed_for_metric(seed, 1),
            )
            long_short_ci = compute_block_bootstrap_mean_ci(
                long_short_values,
                confidence_level=confidence_level,
                n_resamples=n_resamples,
                block_length=block_length,
                random_seed=_seed_for_metric(seed, 2),
            )
            block_length_out = block_length

        bootstrap_resamples_out: int | None = n_resamples
    else:
        confidence_level = thresholds.confidence_level
        _validate_confidence_level(confidence_level)
        use_t_for_small_sample = bool(thresholds.normal_small_sample_use_t)
        small_sample_threshold = int(thresholds.normal_small_sample_threshold)
        if small_sample_threshold < 2:
            raise ValueError("normal_small_sample_threshold must be >= 2")
        ic_ci = compute_mean_ci(
            ic_values,
            confidence_level=confidence_level,
            use_t_for_small_sample=use_t_for_small_sample,
            small_sample_threshold=small_sample_threshold,
        )
        rank_ic_ci = compute_mean_ci(
            rank_ic_values,
            confidence_level=confidence_level,
            use_t_for_small_sample=use_t_for_small_sample,
            small_sample_threshold=small_sample_threshold,
        )
        long_short_ci = compute_mean_ci(
            long_short_values,
            confidence_level=confidence_level,
            use_t_for_small_sample=use_t_for_small_sample,
            small_sample_threshold=small_sample_threshold,
        )
        bootstrap_resamples_out = None
        block_length_out = None

    flags: list[str] = []
    flags.extend(_flags_for_metric("ic", ic_ci, thresholds=thresholds))
    flags.extend(_flags_for_metric("rank_ic", rank_ic_ci, thresholds=thresholds))
    flags.extend(_flags_for_metric("long_short", long_short_ci, thresholds=thresholds))

    return CoreUncertaintySummary(
        mean_ic_ci_lower=ic_ci.ci_lower,
        mean_ic_ci_upper=ic_ci.ci_upper,
        mean_rank_ic_ci_lower=rank_ic_ci.ci_lower,
        mean_rank_ic_ci_upper=rank_ic_ci.ci_upper,
        mean_long_short_return_ci_lower=long_short_ci.ci_lower,
        mean_long_short_return_ci_upper=long_short_ci.ci_upper,
        uncertainty_flags=tuple(flags),
        uncertainty_method=method,
        uncertainty_confidence_level=confidence_level,
        uncertainty_bootstrap_resamples=bootstrap_resamples_out,
        uncertainty_bootstrap_block_length=block_length_out,
    )


def _flags_for_metric(
    prefix: str,
    ci: MeanCI,
    *,
    thresholds: CoreUncertaintyThresholds,
) -> list[str]:
    out: list[str] = []
    if not ci.has_finite_ci:
        out.append(f"{prefix}_ci_unavailable")
        return out

    if ci.overlaps_zero:
        out.append(f"{prefix}_ci_overlaps_zero")

    mean_abs = abs(ci.mean)
    if mean_abs >= thresholds.min_abs_mean_for_relative_width:
        if ci.half_width > thresholds.relative_half_width_warn * mean_abs:
            out.append(f"{prefix}_ci_wide")
    return out


def _z_value(confidence_level: float) -> float:
    _validate_confidence_level(confidence_level)
    tail = 0.5 + 0.5 * confidence_level
    return float(NormalDist().inv_cdf(tail))


def _t_value(confidence_level: float, *, dof: int) -> float:
    _validate_confidence_level(confidence_level)
    if dof < 1:
        raise ValueError("degrees of freedom for Student-t must be >= 1")
    tail = 0.5 + 0.5 * confidence_level
    return float(student_t.ppf(tail, df=dof))


def _critical_value(
    *,
    confidence_level: float,
    n_obs: int,
    use_t_for_small_sample: bool,
    small_sample_threshold: int,
) -> float:
    if use_t_for_small_sample:
        if small_sample_threshold < 2:
            raise ValueError("small_sample_threshold must be >= 2")
        if n_obs < small_sample_threshold:
            return _t_value(confidence_level, dof=n_obs - 1)
    return _z_value(confidence_level)


def _resolve_uncertainty_method(method: str) -> str:
    normalized = str(method).strip().lower()
    if not normalized:
        normalized = "normal"
    if normalized not in {"normal", "bootstrap", "block_bootstrap"}:
        raise ValueError(f"unknown uncertainty method: {method!r}")
    return normalized


def _resolve_bootstrap_confidence_level(*, thresholds: CoreUncertaintyThresholds) -> float:
    if thresholds.bootstrap_confidence_level is None:
        return thresholds.confidence_level
    return float(thresholds.bootstrap_confidence_level)


def _seed_for_metric(base_seed: int | None, offset: int) -> int | None:
    if base_seed is None:
        return None
    return int(base_seed) + int(offset)


def _maybe_warn_small_resamples(n_resamples: int, *, kind: str) -> None:
    if n_resamples < _BOOTSTRAP_RESAMPLE_WARN_THRESHOLD:
        warnings.warn(
            (
                f"{kind} CI computed with n_resamples={n_resamples} "
                f"(<{_BOOTSTRAP_RESAMPLE_WARN_THRESHOLD}); percentile quantiles "
                "may be unstable — treat the CI as indicative only."
            ),
            UserWarning,
            stacklevel=3,
        )


def _validate_confidence_level(confidence_level: float) -> None:
    if confidence_level <= 0.0 or confidence_level >= 1.0:
        raise ValueError("confidence_level must be in (0, 1)")


def _finite_values(values: Iterable[float] | Sequence[float] | pd.Series) -> pd.Series:
    arr = np.asarray(list(values) if not isinstance(values, pd.Series) else values)
    numeric = pd.to_numeric(pd.Series(arr), errors="coerce").dropna()
    return numeric[numeric.map(math.isfinite)].astype(float)
