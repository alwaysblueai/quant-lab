from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from alpha_lab.exceptions import AlphaLabConfigError, AlphaLabExperimentError


@dataclass(frozen=True)
class PortfolioConstraints:
    max_weight: float = 0.10
    min_weight: float = -0.10
    max_gross: float = 1.0
    max_net: float | None = None
    max_turnover: float | None = None
    sector_limits: dict[str, tuple[float, float]] | None = None
    style_limits: dict[str, tuple[float, float]] | None = None


def optimize_portfolio(
    alpha: np.ndarray,
    cov: np.ndarray,
    constraints: PortfolioConstraints,
    risk_aversion: float = 1.0,
    prev_weights: np.ndarray | None = None,
    sector_exposures: np.ndarray | None = None,
    style_exposures: np.ndarray | None = None,
    benchmark_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Solve mean-variance portfolio optimization under linear constraints."""
    if risk_aversion < 0.0:
        raise AlphaLabConfigError("risk_aversion must be >= 0")

    try:
        import cvxpy as cp
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "cvxpy is required for optimize_portfolio. Install dependency: cvxpy>=1.6."
        ) from exc

    alpha_vec = np.asarray(alpha, dtype=float).reshape(-1)
    n_assets = len(alpha_vec)
    if n_assets == 0:
        raise AlphaLabConfigError("alpha must be non-empty")

    cov_mat = np.asarray(cov, dtype=float)
    if cov_mat.shape != (n_assets, n_assets):
        raise AlphaLabConfigError(
            f"cov must have shape ({n_assets}, {n_assets}), got {cov_mat.shape}"
        )

    cov_psd = _project_to_psd(cov_mat)

    h_vec = cp.Variable(n_assets)
    objective = cp.Maximize(alpha_vec @ h_vec - float(risk_aversion) * cp.quad_form(h_vec, cov_psd))

    constraints_list: list[cp.Constraint] = [
        h_vec >= float(constraints.min_weight),
        h_vec <= float(constraints.max_weight),
        cp.norm1(h_vec) <= float(constraints.max_gross),
    ]
    if constraints.max_net is not None:
        constraints_list.append(cp.abs(cp.sum(h_vec)) <= float(constraints.max_net))
    if constraints.max_turnover is not None:
        if prev_weights is None:
            constraints_list.append(cp.norm1(h_vec) <= float(constraints.max_turnover))
        else:
            prev = np.asarray(prev_weights, dtype=float).reshape(-1)
            if prev.shape != (n_assets,):
                raise AlphaLabConfigError(
                    f"prev_weights must have shape ({n_assets},), got {prev.shape}"
                )
            constraints_list.append(cp.norm1(h_vec - prev) <= float(constraints.max_turnover))

    anchor = (
        np.asarray(benchmark_weights, dtype=float).reshape(-1)
        if benchmark_weights is not None
        else np.zeros(n_assets, dtype=float)
    )
    if anchor.shape != (n_assets,):
        raise AlphaLabConfigError(
            f"benchmark_weights must have shape ({n_assets},), got {anchor.shape}"
        )

    _add_exposure_constraints(
        constraints_list=constraints_list,
        h_vec=h_vec,
        exposure_matrix=sector_exposures,
        limits=constraints.sector_limits,
        anchor=anchor,
        label="sector",
        n_assets=n_assets,
    )
    _add_exposure_constraints(
        constraints_list=constraints_list,
        h_vec=h_vec,
        exposure_matrix=style_exposures,
        limits=constraints.style_limits,
        anchor=np.zeros(n_assets, dtype=float),
        label="style",
        n_assets=n_assets,
    )

    problem = cp.Problem(objective, constraints_list)
    solved = False
    for solver in ("OSQP", "ECOS", "SCS"):
        try:
            problem.solve(solver=solver, warm_start=True, verbose=False)
        except Exception:
            continue
        if problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} and h_vec.value is not None:
            solved = True
            break

    if not solved or h_vec.value is None:
        raise AlphaLabExperimentError(
            f"mean-variance optimization failed, status={problem.status!r}"
        )

    return np.asarray(h_vec.value, dtype=float).reshape(-1)


def _project_to_psd(matrix: np.ndarray) -> np.ndarray:
    symmetric = 0.5 * (matrix + matrix.T)
    eigvals, eigvecs = np.linalg.eigh(symmetric)
    eigvals = np.clip(eigvals, a_min=0.0, a_max=None)
    projected = eigvecs @ np.diag(eigvals) @ eigvecs.T
    return cast(np.ndarray, 0.5 * (projected + projected.T))


def _add_exposure_constraints(
    *,
    constraints_list: list[Any],
    h_vec: Any,
    exposure_matrix: np.ndarray | None,
    limits: dict[str, tuple[float, float]] | None,
    anchor: np.ndarray,
    label: str,
    n_assets: int,
) -> None:
    if not limits:
        return
    if exposure_matrix is None:
        raise AlphaLabConfigError(f"{label}_limits provided but {label}_exposures is None")

    matrix = np.asarray(exposure_matrix, dtype=float)
    if matrix.shape[0] != n_assets:
        raise AlphaLabConfigError(
            f"{label}_exposures first dimension must be {n_assets}, got {matrix.shape}"
        )
    if matrix.shape[1] != len(limits):
        raise AlphaLabConfigError(
            f"{label}_limits size {len(limits)} does not match "
            f"{label}_exposures width {matrix.shape[1]}"
        )

    active = matrix.T @ (h_vec - anchor)
    for idx, (_, (lower, upper)) in enumerate(limits.items()):
        constraints_list.append(active[idx] >= float(lower))
        constraints_list.append(active[idx] <= float(upper))
