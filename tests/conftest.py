"""Shared pytest fixtures for alpha-lab tests.

Fixtures are additive: existing tests that inline `_make_prices`, `_build_vault`,
etc. continue to work unchanged. New tests should prefer these fixtures.

Coverage:
- ``synthetic_prices``: deterministic long-form price panel (`date, asset, close`).
- ``synthetic_price_panel_factory``: parametrizable factory variant.
- ``minimal_factor_frame``: deterministic long-form factor frame.
- ``empty_vault``: scratch quant-knowledge vault under ``tmp_path``.
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


@pytest.fixture
def minimal_factor_frame(synthetic_prices: pd.DataFrame) -> pd.DataFrame:
    """Long-form `(date, asset, factor, value)` frame aligned to ``synthetic_prices``."""

    rng = np.random.default_rng(11)
    base = synthetic_prices[["date", "asset"]].copy()
    base["factor"] = "smoke_factor"
    base["value"] = rng.standard_normal(len(base))
    return base


# ---------------------------------------------------------------------------
# Vault scaffolding
# ---------------------------------------------------------------------------


@pytest.fixture
def empty_vault(tmp_path: Path) -> Path:
    """Empty quant-knowledge vault root with the canonical subdirs created."""

    vault = tmp_path / "vault"
    for sub in ("10_concepts", "20_methods", "30_factors", "50_experiments", "55_projects"):
        (vault / sub).mkdir(parents=True, exist_ok=True)
    return vault
