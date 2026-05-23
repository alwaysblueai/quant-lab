"""Shared pytest fixtures for alpha-lab tests.

Fixtures are additive: existing tests that inline `_make_prices`, `_build_vault`,
etc. continue to work unchanged. New tests should prefer these fixtures.

Coverage:
- ``synthetic_prices``: deterministic long-form price panel (`date, asset, close`).
- ``synthetic_price_panel_factory``: parametrizable factory variant.
- ``minimal_factor_frame``: deterministic long-form factor frame.
- ``minimal_factor_frame_factory``: parametrizable variant.
- ``empty_vault``: scratch quant-knowledge vault under ``tmp_path`` with bare dirs.
- ``populated_vault``: vault with CARD-INDEX.tsv + a Factor + a Concept card.
- ``long_universe_mask``: long-form universe-mask sheet for the synthetic panel.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Price panels
# ---------------------------------------------------------------------------


def _build_synthetic_prices(
    *,
    n_assets: int = 6,
    n_days: int = 30,
    seed: int = 42,
    start: str = "2024-01-02",
) -> pd.DataFrame:
    """Return a long-form `(date, asset, close)` panel with positive prices."""

    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=n_days)
    assets = [f"A{i:04d}" for i in range(n_assets)]

    log_steps = rng.normal(loc=0.0, scale=0.01, size=(n_days, n_assets))
    log_levels = np.cumsum(log_steps, axis=0)
    closes = 10.0 * np.exp(log_levels)

    rows: list[dict[str, object]] = []
    for di, date in enumerate(dates):
        for ai, asset in enumerate(assets):
            rows.append(
                {
                    "date": date.normalize(),
                    "asset": asset,
                    "close": float(closes[di, ai]),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_prices() -> pd.DataFrame:
    """Default-shape synthetic prices (6 assets × 30 business days, seed=42)."""

    return _build_synthetic_prices()


@pytest.fixture
def synthetic_price_panel_factory() -> Callable[..., pd.DataFrame]:
    """Factory for synthetic price panels; pass overrides as kwargs."""

    return _build_synthetic_prices


# ---------------------------------------------------------------------------
# Factor frames
# ---------------------------------------------------------------------------


def _build_minimal_factor_frame(
    prices: pd.DataFrame,
    *,
    factor_name: str = "smoke_factor",
    seed: int = 11,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = prices[["date", "asset"]].copy()
    base["factor"] = factor_name
    base["value"] = rng.standard_normal(len(base))
    return base


@pytest.fixture
def minimal_factor_frame(synthetic_prices: pd.DataFrame) -> pd.DataFrame:
    """Long-form `(date, asset, factor, value)` frame aligned to ``synthetic_prices``."""

    return _build_minimal_factor_frame(synthetic_prices)


@pytest.fixture
def minimal_factor_frame_factory(
    synthetic_prices: pd.DataFrame,
) -> Callable[..., pd.DataFrame]:
    """Factory variant of ``minimal_factor_frame`` — pass overrides as kwargs."""

    def factory(
        prices: pd.DataFrame | None = None,
        **kwargs: object,
    ) -> pd.DataFrame:
        return _build_minimal_factor_frame(
            prices if prices is not None else synthetic_prices,
            **kwargs,  # type: ignore[arg-type]
        )

    return factory


# ---------------------------------------------------------------------------
# Universe mask
# ---------------------------------------------------------------------------


@pytest.fixture
def long_universe_mask(synthetic_prices: pd.DataFrame) -> pd.DataFrame:
    """Long-form universe-mask sheet — all rows ``in_universe=1`` over ``synthetic_prices``.

    The factor pipeline expects ``in_universe`` as int (0/1). Tests that need
    a sparse "drop some rows" mask should ``.iloc`` / ``.query`` this base.
    """

    base = synthetic_prices[["date", "asset"]].copy()
    base["in_universe"] = 1
    return base


# ---------------------------------------------------------------------------
# Vault scaffolding
# ---------------------------------------------------------------------------


_CANONICAL_VAULT_DIRS = (
    "00_inbox",
    "_sources",
    "10_concepts",
    "20_methods",
    "30_factors",
    "50_experiments",
    "55_projects",
    "90_computed",
    "90_moc",
)


@pytest.fixture
def empty_vault(tmp_path: Path) -> Path:
    """Empty quant-knowledge vault root with the canonical subdirs created."""

    vault = tmp_path / "vault"
    for sub in _CANONICAL_VAULT_DIRS:
        (vault / sub).mkdir(parents=True, exist_ok=True)
    return vault


@pytest.fixture
def populated_vault(tmp_path: Path) -> Path:
    """Vault with a CARD-INDEX.tsv + one Factor card + one Concept card.

    Matches the shape used by ``research_bridge`` retrieval tests and
    ``web_unified`` smoke tests so they can ``rmtree`` and rebuild from a
    known baseline rather than re-rolling the structure inline.
    """

    vault = tmp_path / "quant-knowledge"
    for sub in _CANONICAL_VAULT_DIRS:
        (vault / sub).mkdir(parents=True, exist_ok=True)
    (vault / "90_moc" / "CARD-INDEX.tsv").write_text(
        "path\ttype\tname\tdomain\tlifecycle\ttags\tparent_moc\n"
        "30_factors/Factor - Momentum Base.md\tfactor\tMomentum Base\talpha_research\t"
        "theoretical\tmomentum,factor\tMOC - Factors\n"
        "10_concepts/Concept - IC.md\tconcept\tIC\talpha_research\t"
        "stable\tic,evaluation\tMOC - Concepts\n",
        encoding="utf-8",
    )
    (vault / "30_factors" / "Factor - Momentum Base.md").write_text(
        "---\ntype: factor\n---\n# 动量基类\n\n用于测试。\n",
        encoding="utf-8",
    )
    (vault / "10_concepts" / "Concept - IC.md").write_text(
        "---\ntype: concept\n---\n# Information Coefficient\n\n占位定义。\n",
        encoding="utf-8",
    )
    return vault
