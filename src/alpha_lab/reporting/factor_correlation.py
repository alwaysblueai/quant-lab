from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class CorrelationEntry:
    factor_name: str
    pearson: float
    rank: float


@dataclass(frozen=True, slots=True)
class FactorCorrelationReport:
    candidate_name: str
    correlations: list[CorrelationEntry]
    max_abs_correlation: float
    likely_redundant: bool
    r_squared: float | None


def compute_factor_correlation(
    candidate_ic: pd.Series,
    existing_ic: dict[str, pd.Series],
    *,
    candidate_name: str = "candidate",
    redundancy_threshold: float = 0.7,
) -> FactorCorrelationReport:
    """Compare one candidate IC series against existing IC series."""
    if not isinstance(candidate_ic, pd.Series) or candidate_ic.empty:
        return FactorCorrelationReport(
            candidate_name=candidate_name,
            correlations=[],
            max_abs_correlation=0.0,
            likely_redundant=False,
            r_squared=None,
        )

    entries: list[CorrelationEntry] = []
    for name, series in existing_ic.items():
        if not isinstance(series, pd.Series):
            continue
        aligned = pd.concat([candidate_ic, series], axis=1, join="inner").dropna()
        if aligned.shape[0] < 5:
            continue
        x = aligned.iloc[:, 0].astype(float)
        y = aligned.iloc[:, 1].astype(float)
        pearson = float(x.corr(y))
        rank = float(x.rank().corr(y.rank()))
        entries.append(CorrelationEntry(factor_name=name, pearson=pearson, rank=rank))

    entries.sort(key=lambda item: abs(item.pearson), reverse=True)
    max_abs = max((abs(item.pearson) for item in entries), default=0.0)
    r_squared = _ols_r_squared(candidate_ic, existing_ic)

    return FactorCorrelationReport(
        candidate_name=candidate_name,
        correlations=entries,
        max_abs_correlation=max_abs,
        likely_redundant=max_abs >= redundancy_threshold,
        r_squared=r_squared,
    )


@dataclass(frozen=True, slots=True)
class FactorCorrelationMatch:
    name: str
    score: float | None
    source: str
    redundant: bool

    def to_payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "score": self.score,
            "source": self.source,
            "redundant": self.redundant,
        }


@dataclass(frozen=True, slots=True)
class FactorCorrelationSignal:
    redundant: bool
    top_match: str
    max_abs_correlation: float
    evidence_path: str
    reason: str


def inspect_run_factor_correlation(
    run_root: str | Path,
    *,
    threshold: float = 0.7,
) -> FactorCorrelationSignal | None:
    resolved = Path(run_root).expanduser().resolve()
    for candidate in _iter_artifacts(resolved):
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            continue
        signal = _parse_signal(payload, evidence_path=candidate, threshold=threshold)
        if signal is not None:
            return signal
    return None


def collect_run_factor_correlation_summary(
    run_root: str | Path,
    *,
    limit: int = 3,
    threshold: float = 0.7,
) -> list[FactorCorrelationMatch]:
    resolved = Path(run_root).expanduser().resolve()
    matches: list[FactorCorrelationMatch] = []
    seen: set[str] = set()

    signal = inspect_run_factor_correlation(resolved, threshold=threshold)
    if signal is not None and signal.top_match:
        score = signal.max_abs_correlation if signal.max_abs_correlation > 0 else None
        matches.append(
            FactorCorrelationMatch(
                name=signal.top_match,
                score=score,
                source=Path(signal.evidence_path).stem or "decomposition",
                redundant=signal.redundant,
            )
        )
        seen.add(signal.top_match)

    for candidate in _iter_artifacts(resolved):
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            continue
        for row in _extract_ranked_rows(payload):
            name = str(row.get("name") or "").strip()
            if not name or name in seen:
                continue
            score = _safe_float_or_none(row.get("score"))
            matches.append(
                FactorCorrelationMatch(
                    name=name,
                    score=score,
                    source=candidate.stem,
                    redundant=bool(score is not None and score >= threshold),
                )
            )
            seen.add(name)
            if len(matches) >= limit:
                return matches[:limit]
    return matches[:limit]


def _iter_artifacts(run_root: Path) -> list[Path]:
    return [run_root / "factor_correlation.json", run_root / "decomposition.json"]


def _parse_signal(
    payload: object,
    *,
    evidence_path: Path,
    threshold: float,
) -> FactorCorrelationSignal | None:
    if isinstance(payload, dict):
        top_match = str(
            payload.get("top_match")
            or payload.get("best_match")
            or payload.get("similar_factor")
            or ""
        ).strip()
        max_corr = _safe_float(
            payload.get("max_abs_correlation")
            or payload.get("max_correlation")
            or payload.get("correlation")
            or payload.get("r_squared")
            or 0.0
        )
        if top_match or max_corr > 0.0:
            return FactorCorrelationSignal(
                redundant=max_corr >= threshold,
                top_match=top_match,
                max_abs_correlation=max_corr,
                evidence_path=str(evidence_path),
                reason=f"run artifact indicates strongest overlap with {top_match or 'unknown'}",
            )
        ranked = payload.get("ranked_matches") or payload.get("correlations")
        if isinstance(ranked, list):
            return _parse_ranked_signal(ranked, evidence_path=evidence_path, threshold=threshold)
    if isinstance(payload, list):
        return _parse_ranked_signal(payload, evidence_path=evidence_path, threshold=threshold)
    return None


def _parse_ranked_signal(
    ranked: list[object],
    *,
    evidence_path: Path,
    threshold: float,
) -> FactorCorrelationSignal | None:
    best_name = ""
    best_score = 0.0
    for item in ranked:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or item.get("factor") or "").strip()
        score = _safe_float(
            item.get("max_abs_correlation")
            or item.get("max_correlation")
            or item.get("correlation")
            or item.get("score")
            or item.get("r_squared")
            or 0.0
        )
        if score > best_score:
            best_name = name
            best_score = score
    if not best_name and best_score <= 0.0:
        return None
    return FactorCorrelationSignal(
        redundant=best_score >= threshold,
        top_match=best_name,
        max_abs_correlation=best_score,
        evidence_path=str(evidence_path),
        reason=f"ranked overlap artifact points to {best_name or 'unknown'}",
    )


def _extract_ranked_rows(payload: object) -> list[dict[str, object]]:
    if isinstance(payload, dict):
        ranked = payload.get("ranked_matches") or payload.get("correlations")
        if isinstance(ranked, list):
            return [_normalize_ranked_row(item) for item in ranked if isinstance(item, dict)]
        if isinstance(ranked, dict):
            return [{"name": str(name), "score": score} for name, score in ranked.items()]
        top_match = str(
            payload.get("top_match")
            or payload.get("best_match")
            or payload.get("similar_factor")
            or ""
        ).strip()
        if top_match:
            return [
                {
                    "name": top_match,
                    "score": (
                        payload.get("max_abs_correlation")
                        or payload.get("max_correlation")
                        or payload.get("correlation")
                        or payload.get("r_squared")
                    ),
                }
            ]
        return []
    if isinstance(payload, list):
        return [_normalize_ranked_row(item) for item in payload if isinstance(item, dict)]
    return []


def _ols_r_squared(
    candidate: pd.Series,
    existing: dict[str, pd.Series],
) -> float | None:
    """Regress candidate on all existing factors and return R-squared."""
    if not existing:
        return None

    df = pd.DataFrame(existing)
    aligned = pd.concat([candidate.rename("_candidate"), df], axis=1, join="inner").dropna()
    if aligned.empty:
        return None

    y = aligned["_candidate"].astype(float).to_numpy()
    x = aligned.drop(columns=["_candidate"]).astype(float).to_numpy()
    if x.shape[0] < max(10, x.shape[1] + 2):
        return None
    x = np.column_stack([x, np.ones(x.shape[0])])

    try:
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    except Exception:
        return None
    y_hat = x @ beta
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot <= 0.0:
        return None
    return 1.0 - (ss_res / ss_tot)


def _normalize_ranked_row(item: dict[str, object]) -> dict[str, object]:
    return {
        "name": str(item.get("name") or item.get("factor") or "").strip(),
        "score": (
            item.get("max_abs_correlation")
            or item.get("max_correlation")
            or item.get("correlation")
            or item.get("score")
            or item.get("r_squared")
        ),
    }


def _safe_float(value: object) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return 0.0
        try:
            return float(text)
        except ValueError:
            return 0.0
    return 0.0


def _safe_float_or_none(value: object) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None
