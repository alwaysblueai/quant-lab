from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

_VALID_WEIGHT_METHODS: frozenset[str] = frozenset({"equal", "rank", "score"})


@dataclass(frozen=True)
class StrategySpec:
    """Research-level portfolio construction specification.

    Separates the *intent* of how to translate factor signals into portfolio
    weights from the mechanics of IC evaluation and quantile analysis.

    **Layer contract**

    ``StrategySpec`` is the explicit boundary between the factor research
    layer (factor values, IC, quantile returns) and the portfolio research
    layer (:mod:`alpha_lab.portfolio_research`).  It answers:

    - which assets to include in each leg (``long_top_k``, ``short_bottom_k``)
    - how to size them (``weighting_method``)
    - how often to rebalance and how long to hold (``rebalance_frequency``,
      ``holding_period``)

    ``n_quantiles`` is deliberately **not** part of this spec.  It governs
    the factor-evaluation path (IC, quantile returns) rather than portfolio
    construction, and belongs to the experiment runner as a separate parameter.

    **This is not an execution specification.**  It does not model order
    routing, market impact, position accounting, or broker constraints.
    It is a research-level description of portfolio construction intent.

    **Long-only vs long-short**

    When ``short_bottom_k`` is ``None`` the portfolio is long-only.
    Long-short portfolios are fully supported: long-leg weights sum to +1,
    short-leg weights sum to −1, and the net portfolio weight is 0.

    The quantile long-short evaluation path
    (:func:`~alpha_lab.quantile.long_short_return`) always compares the top
    and bottom quantile buckets and is not controlled by ``short_bottom_k``.
    ``short_bottom_k`` governs only the weight-based portfolio path in
    :func:`~alpha_lab.portfolio_research.portfolio_weights`.
    """

    long_top_k: int | None = None
    """Number of highest-ranked assets to include in the long leg.
    ``None`` includes all assets in the long leg."""

    short_bottom_k: int | None = None
    """Number of lowest-ranked assets to include in the short leg.
    ``None`` means no short leg (long-only portfolio)."""

    weighting_method: str = "equal"
    """Weight allocation method: ``"equal"``, ``"rank"``, or ``"score"``."""

    holding_period: int = 1
    """Number of rebalance periods to hold each position.  Must be >= 1."""

    rebalance_frequency: int = 1
    """Rebalance every N dates within the weight date grid.  Must be >= 1."""

    def __post_init__(self) -> None:
        if self.long_top_k is not None and self.long_top_k <= 0:
            raise ValueError(
                f"long_top_k must be a positive integer, got {self.long_top_k}"
            )
        if self.short_bottom_k is not None and self.short_bottom_k <= 0:
            raise ValueError(
                f"short_bottom_k must be a positive integer, got {self.short_bottom_k}"
            )
        if self.weighting_method not in _VALID_WEIGHT_METHODS:
            raise ValueError(
                f"weighting_method must be one of {sorted(_VALID_WEIGHT_METHODS)}, "
                f"got {self.weighting_method!r}"
            )
        if self.holding_period < 1:
            raise ValueError(
                f"holding_period must be >= 1, got {self.holding_period}"
            )
        if self.rebalance_frequency < 1:
            raise ValueError(
                f"rebalance_frequency must be >= 1, got {self.rebalance_frequency}"
            )

    @property
    def is_long_short(self) -> bool:
        """``True`` when the strategy specifies an explicit short leg."""
        return self.short_bottom_k is not None


def portfolio_weights_from_strategy(
    factor_df: pd.DataFrame,
    spec: StrategySpec,
) -> pd.DataFrame:
    """Compute portfolio weights according to a :class:`StrategySpec`.

    A thin factory that passes ``spec.long_top_k``, ``spec.short_bottom_k``,
    and ``spec.weighting_method`` to
    :func:`~alpha_lab.portfolio_research.portfolio_weights`, making
    construction intent explicit rather than implied by caller-side defaults.

    Parameters
    ----------
    factor_df:
        Canonical long-form factor DataFrame ``[date, asset, factor, value]``.
        Must contain exactly one factor name.
    spec:
        Strategy specification that governs portfolio construction.

    Returns
    -------
    pd.DataFrame
        Columns: ``[date, asset, weight]``.
    """
    from alpha_lab.portfolio_research import portfolio_weights

    return portfolio_weights(
        factor_df,
        method=spec.weighting_method,
        top_k=spec.long_top_k,
        bottom_k=spec.short_bottom_k,
    )
