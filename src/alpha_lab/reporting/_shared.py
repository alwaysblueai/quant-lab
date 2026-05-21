"""Shared helpers used across reporting submodules.

Consolidates byte-identical duplicates that were copy-pasted between
``factor_verdict``, ``campaign_triage``, ``level2_promotion``,
``research_artifact_manifest``, ``workflow_artifact_service``,
``renderers/campaign_profile_dashboard``, and
``real_cases/artifact_enrichment``.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import pandas as pd

from alpha_lab.reporting.display_helpers import safe_text


def finalize_reasons(*groups: Sequence[str], max_items: int) -> tuple[str, ...]:
    """Merge multiple ordered reason groups into a deduped, capped tuple.

    Empty strings and duplicates (after stripping whitespace) are
    dropped while preserving first-occurrence order.
    """
    merged: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for reason in group:
            token = reason.strip()
            if not token or token in seen:
                continue
            merged.append(token)
            seen.add(token)
            if len(merged) >= max_items:
                return tuple(merged)
    return tuple(merged)


def resolve_artifact_path(value: object, *, base_dir: Path) -> Path | None:
    """Resolve a manifest-recorded artifact path to an absolute ``Path``.

    Returns ``None`` when the value is empty/unparseable. Relative
    paths are anchored at ``base_dir``.
    """
    text = safe_text(value)
    if not text:
        return None
    candidate = Path(text)
    if not candidate.is_absolute():
        candidate = base_dir / candidate
    return candidate.resolve()


def periods_per_year(rebalance_frequency: str) -> int:
    """Map a free-form rebalance frequency string to periods per year."""
    freq = (rebalance_frequency or "").strip().upper()
    if freq.startswith("D"):
        return 252
    if freq.startswith("W"):
        return 52
    if freq.startswith("M"):
        return 12
    return 252


def load_required_json(path: Path) -> dict[str, object]:
    """Load a required JSON artifact, raising if missing or non-object."""
    if not path.exists():
        raise FileNotFoundError(f"required artifact not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return cast(dict[str, object], payload)


def annualized_from_series(series: pd.Series, periods_per_year_: int) -> float | None:
    """Annualize a finite period-return series into a single rate.

    Returns ``None`` when the cleaned series is empty.
    """
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return None
    nav = (1.0 + clean).cumprod()
    total_return = float(nav.iloc[-1] - 1.0)
    return float((1.0 + total_return) ** (periods_per_year_ / len(clean)) - 1.0)
