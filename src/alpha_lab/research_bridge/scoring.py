"""Typed hybrid scoring for explore-idea card retrieval (P0 + P1).

Splits "card relevance" into base orthogonal components and lets each
exploration mode weight them differently. Also exposes a hard filter for
unsatisfied data dependencies (P1: ``depends_on`` / ``uses_data`` are
graph relations, not similarity terms).

This module has no I/O — it consumes pre-extracted ``CardMetadata`` and
returns scalar scores. Callers source metadata from VaultGraph,
CARD-INDEX.tsv, and frontmatter.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+|[一-鿿]+")


@dataclass(frozen=True, slots=True)
class CardMetadata:
    """Structured metadata for one card. Sets are normalized to lowercase."""

    name: str
    type: str = ""
    domain: str = ""
    lifecycle: str = ""
    market: str = ""
    mechanism: str = ""
    factor_family: str = ""
    uses_data: frozenset[str] = field(default_factory=frozenset)
    depends_on: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True, slots=True)
class QueryAnchor:
    """Inferred research direction used as the comparison target.

    Built from project hints + the top-1 semantic match's metadata. We
    deliberately keep this a flat record rather than a CardMetadata so it
    can never carry a ``uses_data`` requirement of its own.
    """

    domain: str = ""
    market: str = ""
    mechanism: str = ""
    factor_family: str = ""


@dataclass(frozen=True, slots=True)
class ScoreComponents:
    semantic: float = 0.0
    metadata: float = 0.0
    mechanism: float = 0.0
    dependency: float = 0.0
    failure: float = 0.0
    llm_relevance: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "semantic": self.semantic,
            "metadata": self.metadata,
            "mechanism": self.mechanism,
            "dependency": self.dependency,
            "failure": self.failure,
            "llm_relevance": self.llm_relevance,
        }


@dataclass(frozen=True, slots=True)
class ScoreWeights:
    semantic: float = 1.0
    metadata: float = 0.0
    mechanism: float = 0.0
    dependency: float = 0.0
    failure: float = 0.0
    llm_relevance: float = 0.0


# Mode-conditioned weights for the alpha-lab factor-recipe explorer.
# The vault is a source-material library, not a precedent court. Failure
# proximity remains a diagnostic component surfaced to prompts, but it no
# longer contributes to ranking or static kill behavior in Stage 1. Only
# available_data can hard-filter a card before generation.
ALPHA_MODE_WEIGHTS: dict[str, ScoreWeights] = {
    "start": ScoreWeights(
        semantic=1.0,
        metadata=0.2,
        mechanism=0.2,
        dependency=0.0,
        failure=0.0,
        llm_relevance=1.0,
    ),
    "free": ScoreWeights(
        semantic=0.6,
        metadata=0.7,
        mechanism=0.7,
        dependency=0.3,
        failure=0.0,
        llm_relevance=0.6,
    ),
    "constrained": ScoreWeights(
        semantic=0.4,
        metadata=0.6,
        mechanism=0.5,
        dependency=0.8,
        failure=0.0,
        llm_relevance=0.3,
    ),
}

# Mode-conditioned weights for the model-lab idea explorer. Same shape
# as the alpha set, with the middle band keyed on ``explore`` instead of
# ``free`` to match model-lab's mode names.
MODEL_MODE_WEIGHTS: dict[str, ScoreWeights] = {
    "start": ScoreWeights(
        semantic=1.0,
        metadata=0.2,
        mechanism=0.2,
        dependency=0.0,
        failure=0.0,
        llm_relevance=1.0,
    ),
    "explore": ScoreWeights(
        semantic=0.6,
        metadata=0.7,
        mechanism=0.6,
        dependency=0.3,
        failure=0.0,
        llm_relevance=0.6,
    ),
    "constrained": ScoreWeights(
        semantic=0.4,
        metadata=0.6,
        mechanism=0.5,
        dependency=0.8,
        failure=0.0,
        llm_relevance=0.3,
    ),
}


def aggregate_score(components: ScoreComponents, weights: ScoreWeights) -> float:
    return (
        components.semantic * weights.semantic
        + components.metadata * weights.metadata
        + components.mechanism * weights.mechanism
        + components.dependency * weights.dependency
        + components.failure * weights.failure
        + components.llm_relevance * weights.llm_relevance
    )


def metadata_compat_score(anchor: QueryAnchor, card: CardMetadata) -> float:
    """Average exact-match indicator over (domain, market).

    Returns 0.0 when neither side has any of the fields populated.
    """
    parts: list[float] = []
    for left, right in (
        (anchor.domain, card.domain),
        (anchor.market, card.market),
    ):
        left_norm = (left or "").strip().lower()
        right_norm = (right or "").strip().lower()
        if not left_norm or not right_norm:
            continue
        parts.append(1.0 if left_norm == right_norm else 0.0)
    if not parts:
        return 0.0
    return sum(parts) / len(parts)


def mechanism_proximity_score(anchor: QueryAnchor, card: CardMetadata) -> float:
    """1.0 for same mechanism, 0.6 for same factor_family, else 0.0."""
    anchor_mechanism = (anchor.mechanism or "").strip().lower()
    card_mechanism = (card.mechanism or "").strip().lower()
    if anchor_mechanism and card_mechanism and anchor_mechanism == card_mechanism:
        return 1.0
    anchor_family = (anchor.factor_family or "").strip().lower()
    card_family = (card.factor_family or "").strip().lower()
    if anchor_family and card_family and anchor_family == card_family:
        return 0.6
    return 0.0


def data_dependency_score(
    card: CardMetadata,
    available_data: frozenset[str] | None,
) -> tuple[float, bool, str]:
    """Return ``(score, satisfied, reason)``.

    * ``score`` — fraction of ``card.uses_data`` covered by ``available_data``.
      0..1. When the card has no data dependencies declared, we return 1.0.
    * ``satisfied`` — False only when ``available_data`` is explicitly
      provided AND coverage is below 1.0. Used by the hard filter.
    * ``reason`` — human-readable explanation when ``satisfied`` is False.

    When ``available_data`` is None the caller hasn't supplied an
    inventory, so we treat dependency as a soft 0.5 signal and never
    block.
    """
    if not card.uses_data:
        return 1.0, True, ""
    if available_data is None:
        return 0.5, True, ""
    overlap = card.uses_data & available_data
    coverage = len(overlap) / len(card.uses_data)
    if coverage < 1.0:
        missing = sorted(card.uses_data - available_data)
        return coverage, False, "missing_data: " + ", ".join(missing)
    return 1.0, True, ""


def failure_proximity_score(
    card: CardMetadata,
    failure_keywords: frozenset[str],
) -> float:
    """How close a card sits to known failure registry entries.

    1.0 — direct mechanism / factor_family overlap with the failure index.
    0.5 — token-level overlap on those fields.
    0.0 — no signal or no failure context.
    """
    if not failure_keywords:
        return 0.0
    targets = {
        (card.mechanism or "").strip().lower(),
        (card.factor_family or "").strip().lower(),
    }
    targets.discard("")
    if targets & failure_keywords:
        return 1.0
    target_tokens: set[str] = set()
    for value in targets:
        for token in _TOKEN_RE.findall(value):
            target_tokens.add(token.lower())
    if target_tokens & failure_keywords:
        return 0.5
    return 0.0


def derive_failure_keywords(
    failure_refs: Iterable[Mapping[str, object]],
) -> frozenset[str]:
    keywords: set[str] = set()
    for item in failure_refs:
        for field_name in (
            "title",
            "failure_class",
            "failure_statement",
            "prevention_rule",
        ):
            text = str(item.get(field_name) or "")
            for token in _TOKEN_RE.findall(text):
                if len(token) > 1:
                    keywords.add(token.lower())
    return frozenset(keywords)


def score_card(
    *,
    anchor: QueryAnchor,
    card: CardMetadata,
    semantic_score: float,
    available_data: frozenset[str] | None,
    failure_keywords: frozenset[str],
) -> tuple[ScoreComponents, bool, str]:
    """Compute the five-component score plus the hard-filter verdict.

    Returns ``(components, kept, drop_reason)``. ``kept`` is False when a
    hard filter rejects the card; ``drop_reason`` is empty when kept.
    """
    metadata = metadata_compat_score(anchor, card)
    mechanism = mechanism_proximity_score(anchor, card)
    dependency, satisfied, missing_reason = data_dependency_score(
        card, available_data
    )
    failure = failure_proximity_score(card, failure_keywords)
    components = ScoreComponents(
        semantic=float(semantic_score),
        metadata=metadata,
        mechanism=mechanism,
        dependency=dependency,
        failure=failure,
    )
    if not satisfied:
        return components, False, missing_reason
    return components, True, ""


def normalize_data_set(values: Iterable[str] | None) -> frozenset[str]:
    """Lowercase + strip a data-identifier set, dropping empties."""
    if values is None:
        return frozenset()
    return frozenset(
        item.strip().lower() for item in values if item and str(item).strip()
    )


# ---------------------------------------------------------------------------
# Workflow stages (P2)
#
# These are the Stage 1 prompt substeps from docs/research_workflow.md:
# mechanism discovery followed by signal mapping. Stage 3 data-kill still
# has a legacy LLM helper constant below, but it is no longer part of the
# public idea-explorer progression.
# ---------------------------------------------------------------------------

MECHANISM_DISCOVERY = "mechanism_discovery"
SIGNAL_MAPPING = "signal_mapping"
VALIDATION_KILL_TESTS = "validation_kill_tests"

WORKFLOW_STAGES: frozenset[str] = frozenset(
    {MECHANISM_DISCOVERY, SIGNAL_MAPPING}
)
STAGE3_LLM_HELPER_STAGES: frozenset[str] = frozenset({VALIDATION_KILL_TESTS})

_WORKFLOW_STAGE_PROGRESSION: dict[str, str | None] = {
    MECHANISM_DISCOVERY: SIGNAL_MAPPING,
    SIGNAL_MAPPING: None,
}


def normalize_workflow_stage(
    stage: str | None, *, allow_stage3_helper: bool = False
) -> str:
    text = (stage or "").strip().lower()
    if text in WORKFLOW_STAGES:
        return text
    if allow_stage3_helper and text in STAGE3_LLM_HELPER_STAGES:
        return text
    return MECHANISM_DISCOVERY


def recommend_next_stage(current: str | None) -> str | None:
    if (current or "").strip().lower() in STAGE3_LLM_HELPER_STAGES:
        return None
    return _WORKFLOW_STAGE_PROGRESSION.get(normalize_workflow_stage(current))


def is_stage3_llm_helper_stage(stage: str | None) -> bool:
    return (stage or "").strip().lower() in STAGE3_LLM_HELPER_STAGES


# ---------------------------------------------------------------------------
# Default data inventories by frequency tier.
#
# These are coarse priors used to auto-populate ``available_data`` when the
# caller hasn't passed an explicit set. Daily-tier projects should never
# auto-pull HFT cards into their context, so the daily inventory is the
# strict subset and must not include intraday-only fields.
# ---------------------------------------------------------------------------

DAILY_DATA_INVENTORY: frozenset[str] = frozenset(
    {
        "open",
        "close",
        "high",
        "low",
        "volume",
        "vwap",
        "amount",
        "turnover",
        "shares_outstanding",
        "adj_close",
        "industry",
        "market_cap",
        "free_float",
        "pre_close",
        "limit_up",
        "limit_down",
    }
)

INTRADAY_DATA_INVENTORY: frozenset[str] = DAILY_DATA_INVENTORY | frozenset(
    {
        "tick",
        "tick_volume",
        "tick_price",
        "bid_ask",
        "bid",
        "ask",
        "spread",
        "depth",
        "order_book",
        "intraday_trades",
        "intraday_tick_volume",
        "minute_bar",
    }
)

_DAILY_FREQUENCY_TOKENS: frozenset[str] = frozenset(
    {"daily", "weekly", "monthly", "d", "w", "m", "ashare", "a_share"}
)
_INTRADAY_FREQUENCY_TOKENS: frozenset[str] = frozenset(
    {
        "intraday",
        "tick",
        "minute",
        "minutes",
        "1min",
        "5min",
        "15min",
        "30min",
        "60min",
        "hourly",
        "second",
        "hf",
        "hft",
    }
)


def infer_available_data_from_frequency(
    frequency: str | None,
) -> frozenset[str] | None:
    """Map ``project.frequency`` to a default data inventory.

    Returns None when the frequency is unrecognized — the caller should
    treat that as "no inventory provided" so the dependency hard filter
    stays inactive (better silent than false-negative).
    """
    text = (frequency or "").strip().lower()
    if not text:
        return None
    if text in _DAILY_FREQUENCY_TOKENS:
        return DAILY_DATA_INVENTORY
    if text in _INTRADAY_FREQUENCY_TOKENS:
        return INTRADAY_DATA_INVENTORY
    return None


# ---------------------------------------------------------------------------
# Forbidden labels (mechanism_discovery stage).
#
# In mechanism_discovery the LLM must not pre-name a candidate using one of
# the established factor labels — naming = premature classification, which
# collapses the hypothesis space before signal_mapping has a chance to test
# alternatives. The list is intentionally a fixed vocabulary so it is easy
# to lint against; it is not a regex over substrings.
# ---------------------------------------------------------------------------

FORBIDDEN_FACTOR_LABELS: tuple[str, ...] = (
    "reversal",
    "mean reversion",
    "mean-reversion",
    "momentum",
    "value",
    "growth",
    "quality",
    "size",
    "low-vol",
    "low volatility",
    "low-volatility",
    "skewness",
    "kurtosis",
    "liquidity",
    "sentiment",
    "crowding",
    "carry",
    "defensive",
    "low risk",
    "low-risk",
    # CN labels
    "反转",
    "动量",
    "价值",
    "成长",
    "规模",
    "质量",
    "情绪",
    "拥挤",
    "低波",
    "防御",
    "低风险",
)


# ---------------------------------------------------------------------------
# Validation & Kill-Tests vocabulary (validation_kill_tests stage).
#
# In mechanism_discovery, these labels are guardrails (avoid premature
# naming). Here, the *same* labels return as **audit targets** — for each
# one, the LLM must explicitly answer "is this candidate just an alias of
# the named factor?". Same words, opposite role.
# ---------------------------------------------------------------------------

VALIDATION_ALIAS_TARGETS: tuple[tuple[str, str], ...] = (
    ("reversal", "短期均值回复 / 反转 / over-reaction reversal"),
    (
        "volatility",
        "realized vol（含上下行分解）/ idiosyncratic vol / GARCH / "
        "implied vol",
    ),
    (
        "skewness / downside risk",
        "co-skewness / negative coskew / VaR / max-drawdown 类",
    ),
    (
        "liquidity / turnover",
        "Amihud illiquidity / 换手率 / bid-ask spread / Roll spread",
    ),
    (
        "size / industry / price level",
        "市值暴露 / 行业暴露 / 价格水平偏低或偏高 / 流通盘暴露",
    ),
)

# ---------------------------------------------------------------------------
# Signal-mapping confound controls (signal_mapping stage).
#
# Same five families as VALIDATION_ALIAS_TARGETS but used in a different
# role. In validation_kill_tests they're audit targets ("is the factor
# secretly an X-clone?"). In signal_mapping they're confounds the *new*
# signal version must explicitly handle: include, residualize, hard-
# control, or knowingly leave alone with stated risk. The volatility
# entry is reframed as "total" volatility to make the contrast with
# directional/asymmetric signals explicit.
# ---------------------------------------------------------------------------

SIGNAL_MAPPING_CONFOUND_CONTROLS: tuple[tuple[str, str], ...] = (
    ("reversal", "短期均值回复 / 1-5 日反转效应"),
    ("total volatility", "不分上下行的总波动 / realized vol / idiosyncratic vol"),
    ("skewness / downside risk", "co-skewness / negative coskew / VaR / max-drawdown 类"),
    ("liquidity / turnover", "Amihud illiquidity / 换手率 / bid-ask spread"),
    ("size / industry / price level", "市值暴露 / 行业暴露 / 价格水平 / 流通盘"),
)
