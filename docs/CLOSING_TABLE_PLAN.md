# Closing Table Execution Plan

> Scope: complete the 4 remaining items from the Graph-Enhanced Research Agent System closing table.
> All items are non-blocking enhancements — the main explore-idea pipeline is already working.
> Target: Codex direct execution.

---

## Pre-read: key files

Before starting any task, read these files to understand existing patterns:

- `src/alpha_lab/web_unified.py` — unified HTTP server, all routes
- `src/alpha_lab/vault_export_graph_feedback.py` — graph feedback after writeback
- `src/alpha_lab/reporting/factor_decomposition.py` — existing decomposition reader
- `src/alpha_lab/research_bridge/service.py` — service layer (explore_idea, apply_writeback, etc.)
- `src/alpha_lab/research_bridge/embeddings.py` — VaultEmbeddings class
- `src/alpha_lab/research_bridge/graph_view.py` — VaultGraph class
- `src/alpha_lab/cli.py` — top-level CLI dispatcher
- `tests/test_web_unified_http.py` — existing HTTP live tests (15 test functions)
- `tests/test_research_bridge_service.py` — existing service tests

---

## Task 1 — HTTP Live Integration Tests (Priority 1)

### Goal

`tests/test_web_unified_http.py` already has 15 tests covering core routes. Add tests for every POST route and remaining GET route that lacks coverage.

### What exists

- `tests/test_web_unified_http.py` with:
  - Fixture `_build_vault(tmp_path)` — creates minimal vault with CARD-INDEX.tsv, one factor card, inbox note
  - Fixture `live_server` — real `ThreadingHTTPServer` on random port, yields `(base_url, svc)`
  - Fixture `seeded_server` — extends `live_server` with project + round + case + draft
  - Helpers `_get(base_url, path)` and `_post(base_url, path, payload)` returning `(status, data)`
  - Tests for: root HTML, vault/stats, vault/inbox, vault/card/{name}, evaluation-profiles, explore-idea (4 variants), project cases, round artifacts, draft read, 404

### Step 1: discover untested routes

Search `web_unified.py` for all `parsed.path ==` or `parts ==` in both `do_GET` and `do_POST`. List every route. Compare against existing test coverage. The untested routes are the ones to add.

Expected untested routes (verify by reading `web_unified.py`):

**POST routes:**
- `/api/projects` — create project
- `/api/projects/{slug}/rounds` — create round
- `/api/projects/{slug}/cases` — create case
- `/api/projects/{slug}/drafts/patch` — patch draft frontmatter
- `/api/vault/preflight` — run preflight checks

**GET routes:**
- `/api/projects` — list projects
- `/api/categories` — list category profiles (if exists)

### Step 2: add tests

For each untested route, add one test function following the existing pattern in the file. Use existing fixtures. Example pattern:

```python
def test_create_project_route(live_server: tuple[str, _UnifiedService]) -> None:
    base_url, svc = live_server
    payload = {
        "slug": "http-test-proj",
        "title_zh": "HTTP 测试项目",
        "category": "factor_recipe",
        "owner": "test",
        "market": "ashare",
        "frequency": "daily",
        "chatgpt_project_name": "HTTP Test",
        "origin_cards": [],
    }
    status, data = _post(base_url, "/api/projects", payload)
    assert status == 200
    assert isinstance(data, dict)
```

Adapt payload and assertions to each route's actual schema (read `do_POST` dispatch code).

### Step 3: verify the exact route paths

The actual route paths might differ from the above guesses. Read `do_POST` carefully:
- Look for `parsed.path ==` string literals
- Look for path-pattern matching via `_path_parts(parsed.path)`
- Match each to the correct test

### Verification

```bash
python -m pytest tests/test_web_unified_http.py -v
```

All tests must pass. No mocking of HTTP — use real server.

---

## Task 2 — Feedback Enhancement: `similar_to_suggestions` + `correlation_summary` (Priority 2)

### Goal

Enhance `apply_graph_feedback()` to return embedding-based `similar_to` suggestions and a human-readable `correlation_summary`.

### What exists

- `src/alpha_lab/vault_export_graph_feedback.py`:
  - `GraphFeedbackResult` dataclass with fields: `tested_in_updates`, `exploration_updated`, `failure_entry_id`, `suggested_similar_to`
  - `apply_graph_feedback()` already computes `suggested_similar_to` from decomposition only (lines ~91-94)
- `src/alpha_lab/research_bridge/embeddings.py`: `VaultEmbeddings` with `suggest_similar(name, threshold)` method

### Changes

#### 2a. Add `correlation_summary: str` field to `GraphFeedbackResult`

```python
@dataclass(frozen=True, slots=True)
class GraphFeedbackResult:
    tested_in_updates: list[str]
    exploration_updated: bool
    failure_entry_id: str
    suggested_similar_to: list[str]
    correlation_summary: str          # NEW — empty string when no data
```

#### 2b. Add embedding-based suggestions in `apply_graph_feedback`

After existing decomposition logic (around line 94), before the `_run_rebuild_script` calls:

```python
# Embedding-based similar_to suggestions
embeddings = _load_embeddings_optional(vault_root)
if embeddings is not None:
    for raw_card in project.origin_cards:
        card_name = Path(raw_card).stem.split(" - ", 1)[-1].strip()
        try:
            candidates = embeddings.suggest_similar(card_name, threshold=0.82)
            for c in candidates:
                if c not in suggested_similar_to:
                    suggested_similar_to.append(c)
        except Exception:
            pass
```

Add helper (follow existing `_load_*` pattern in the file):

```python
def _load_embeddings_optional(vault_root: Path):
    try:
        from alpha_lab.research_bridge.embeddings import VaultEmbeddings
        emb = VaultEmbeddings.from_vault_root(vault_root)
        emb.build(vault_root=vault_root)
        return emb
    except Exception:
        return None
```

#### 2c. Build `correlation_summary`

After finalizing `suggested_similar_to`:

```python
correlation_summary = ""
if suggested_similar_to:
    parts = []
    for name in suggested_similar_to[:5]:
        source = "decomposition" if (decomposition and name == decomposition.top_match) else "embedding"
        parts.append(f"- {name} (via {source})")
    correlation_summary = "Similar factor suggestions:\n" + "\n".join(parts)
```

Pass `correlation_summary` to the return `GraphFeedbackResult(...)`.

#### 2d. Update all callers

Search for `GraphFeedbackResult` usage. If anything destructures or accesses this dataclass, it needs to handle the new field. Since it's frozen with slots, adding a field is a breaking change — verify all call sites.

#### 2e. Tests

Create `tests/test_vault_export_graph_feedback.py` (or add to existing). Minimal test:

```python
def test_graph_feedback_returns_correlation_summary(tmp_path):
    # Set up minimal vault + project + draft frontmatter + export result
    # Call apply_graph_feedback(...)
    # Assert result.correlation_summary is a str (possibly empty)
    # Assert result.suggested_similar_to is a list
```

Follow existing test patterns from `tests/test_research_bridge_service.py` for vault/project setup.

### Verification

```bash
python -m pytest tests/ -k "graph_feedback" -v
```

---

## Task 3 — Unified `vault` CLI Namespace (Priority 3)

### Goal

Wrap vault rebuild scripts under `alpha-lab vault <subcommand>`.

### What exists

- `src/alpha_lab/cli.py` — top-level commands: `run`, `real-case`, `campaign`, `bridge`, `experimental`, `profiles`, `web`, `data`
- Vault scripts at `/mnt/c/quant/vault/quant-knowledge/00_protocols/`:
  - `rebuild-graph.py`
  - `rebuild-embeddings.py`
  - `rebuild-exploration-map.py`
  - `suggest-edges.py`
- These scripts accept vault_root as a positional CLI argument

### Changes

#### 3a. Create `src/alpha_lab/vault_cli.py`

```python
"""CLI subcommands for vault computed-layer management."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from alpha_lab.vault_export import resolve_vault_root

_SCRIPT_NAMES = (
    "rebuild-graph",
    "rebuild-embeddings",
    "rebuild-exploration-map",
    "suggest-edges",
)


def build_vault_parser(parser: argparse.ArgumentParser) -> None:
    parser.description = "Manage quant-knowledge vault computed layer."
    commands = parser.add_subparsers(dest="vault_action", required=True)

    for name in _SCRIPT_NAMES:
        cmd = commands.add_parser(
            name,
            help=f"Run {name} against the vault.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        cmd.add_argument("--vault-root", default=None, help="Vault root. Defaults to OBSIDIAN_VAULT_PATH.")

    all_cmd = commands.add_parser(
        "rebuild-all",
        help="Run rebuild-graph, rebuild-embeddings, rebuild-exploration-map in sequence.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    all_cmd.add_argument("--vault-root", default=None, help="Vault root. Defaults to OBSIDIAN_VAULT_PATH.")


def run_vault_command(args: argparse.Namespace) -> int:
    vault_root = resolve_vault_root(getattr(args, "vault_root", None))
    if vault_root is None:
        print("Error: vault root unresolved; pass --vault-root or set OBSIDIAN_VAULT_PATH", file=sys.stderr)
        return 1

    if args.vault_action == "rebuild-all":
        scripts = ["rebuild-graph.py", "rebuild-embeddings.py", "rebuild-exploration-map.py"]
    else:
        scripts = [f"{args.vault_action}.py"]

    exit_code = 0
    for script_name in scripts:
        script_path = vault_root / "00_protocols" / script_name
        if not script_path.exists():
            print(f"Warning: {script_path} not found, skipping.", file=sys.stderr)
            continue
        print(f"  Running {script_name} ...")
        result = subprocess.run(
            [sys.executable, str(script_path), str(vault_root)],
            capture_output=False,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            exit_code = 1
    return exit_code
```

#### 3b. Register in `cli.py`

1. Add `"vault"` to `_UNIFIED_TOP_LEVEL_COMMANDS` set (line ~43).

2. After the `web` subparser block (around line 663-680), add:

```python
vault_parser = top.add_parser(
    "vault",
    help="Manage quant-knowledge vault computed layer.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
from alpha_lab.vault_cli import build_vault_parser  # noqa: PLC0415
build_vault_parser(vault_parser)
```

3. In the dispatch section (around line 920-968), add before the final return:

```python
if args.top_command == "vault":
    from alpha_lab.vault_cli import run_vault_command  # noqa: PLC0415
    return run_vault_command(args)
```

#### 3c. Tests — `tests/test_vault_cli.py`

```python
"""Tests for vault CLI subcommands."""
from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import patch

from alpha_lab.vault_cli import build_vault_parser, run_vault_command


def test_parser_accepts_all_subcommands():
    parser = argparse.ArgumentParser()
    build_vault_parser(parser)
    for name in ("rebuild-graph", "rebuild-embeddings", "rebuild-exploration-map", "suggest-edges", "rebuild-all"):
        args = parser.parse_args([name, "--vault-root", "/tmp/fake"])
        assert args.vault_action == name


def test_missing_script_warns_but_succeeds(tmp_path: Path):
    args = argparse.Namespace(vault_action="rebuild-graph", vault_root=str(tmp_path))
    with patch("alpha_lab.vault_cli.resolve_vault_root", return_value=tmp_path):
        code = run_vault_command(args)
    assert isinstance(code, int)


def test_rebuild_all_runs_scripts(tmp_path: Path):
    protocols = tmp_path / "00_protocols"
    protocols.mkdir()
    for name in ("rebuild-graph.py", "rebuild-embeddings.py", "rebuild-exploration-map.py"):
        (protocols / name).write_text("import sys; print('ok')", encoding="utf-8")
    args = argparse.Namespace(vault_action="rebuild-all", vault_root=str(tmp_path))
    with patch("alpha_lab.vault_cli.resolve_vault_root", return_value=tmp_path):
        code = run_vault_command(args)
    assert code == 0


def test_unresolved_vault_returns_error():
    args = argparse.Namespace(vault_action="rebuild-graph", vault_root=None)
    with patch("alpha_lab.vault_cli.resolve_vault_root", return_value=None):
        code = run_vault_command(args)
    assert code == 1
```

### Verification

```bash
python -m pytest tests/test_vault_cli.py -v
```

---

## Task 4 — Independent `factor_correlation.py` Module (Priority 3)

### Goal

Create `src/alpha_lab/reporting/factor_correlation.py` — computes IC-series correlation matrix between a candidate factor and existing factors, with OLS R-squared for redundancy detection.

### What exists

- `src/alpha_lab/reporting/factor_decomposition.py` — reads pre-computed JSON from run output
- The design doc (§6) specifies: `candidate IC ~ Sigma(w_i * existing_i)`, flag redundant if R^2 > 0.7

### Create `src/alpha_lab/reporting/factor_correlation.py`

```python
"""Factor correlation analysis — compare candidate IC series against known factors."""
from __future__ import annotations

from dataclasses import dataclass

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
    """Compare candidate IC series against existing factor IC series.

    Parameters
    ----------
    candidate_ic:
        Series indexed by date with IC values.
    existing_ic:
        Mapping factor_name -> IC series.
    candidate_name:
        Label for the candidate factor.
    redundancy_threshold:
        |correlation| above this flags likely_redundant.
    """
    if not existing_ic:
        return FactorCorrelationReport(
            candidate_name=candidate_name,
            correlations=[],
            max_abs_correlation=0.0,
            likely_redundant=False,
            r_squared=None,
        )

    entries: list[CorrelationEntry] = []
    for name, series in existing_ic.items():
        aligned = pd.concat([candidate_ic, series], axis=1, join="inner").dropna()
        if len(aligned) < 5:
            continue
        pearson = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))
        rank = float(aligned.iloc[:, 0].rank().corr(aligned.iloc[:, 1].rank()))
        entries.append(CorrelationEntry(factor_name=name, pearson=pearson, rank=rank))

    entries.sort(key=lambda e: abs(e.pearson), reverse=True)
    max_abs = max((abs(e.pearson) for e in entries), default=0.0)
    r_squared = _ols_r_squared(candidate_ic, existing_ic)

    return FactorCorrelationReport(
        candidate_name=candidate_name,
        correlations=entries,
        max_abs_correlation=max_abs,
        likely_redundant=max_abs >= redundancy_threshold,
        r_squared=r_squared,
    )


def _ols_r_squared(
    candidate: pd.Series,
    existing: dict[str, pd.Series],
) -> float | None:
    """Regress candidate on all existing factors; return R-squared."""
    try:
        X = pd.DataFrame(existing)
        combined = pd.concat([candidate.rename("_y"), X], axis=1, join="inner").dropna()
        if len(combined) < max(10, len(existing) + 2):
            return None
        y = combined["_y"].values.astype(float)
        X_mat = combined.drop(columns=["_y"]).values.astype(float)
        X_mat = np.column_stack([X_mat, np.ones(len(X_mat))])
        beta = np.linalg.lstsq(X_mat, y, rcond=None)[0]
        y_hat = X_mat @ beta
        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        if ss_tot == 0:
            return None
        return 1.0 - ss_res / ss_tot
    except Exception:
        return None
```

### Create `tests/test_factor_correlation.py`

```python
"""Tests for factor_correlation module."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_lab.reporting.factor_correlation import compute_factor_correlation


def _make_ic(values: list[float]) -> pd.Series:
    dates = pd.date_range("2020-01-01", periods=len(values), freq="D")
    return pd.Series(values, index=dates, name="ic")


def test_empty_existing():
    result = compute_factor_correlation(_make_ic([0.1, 0.2, 0.3, 0.4, 0.5]), {})
    assert result.max_abs_correlation == 0.0
    assert result.likely_redundant is False
    assert result.r_squared is None


def test_perfect_correlation():
    vals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    result = compute_factor_correlation(_make_ic(vals), {"clone": _make_ic(vals)})
    assert result.max_abs_correlation == pytest.approx(1.0)
    assert result.likely_redundant is True


def test_low_correlation_not_redundant():
    candidate = _make_ic([0.1, -0.1, 0.1, -0.1, 0.1, -0.1, 0.1, -0.1, 0.1, -0.1])
    existing = {"flat": _make_ic([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])}
    result = compute_factor_correlation(candidate, existing)
    assert result.likely_redundant is False


def test_r_squared_multivariate():
    np.random.seed(42)
    n = 50
    x1, x2 = np.random.randn(n), np.random.randn(n)
    y = 0.7 * x1 + 0.3 * x2 + 0.05 * np.random.randn(n)
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    result = compute_factor_correlation(
        pd.Series(y, index=dates),
        {"f1": pd.Series(x1, index=dates), "f2": pd.Series(x2, index=dates)},
        candidate_name="composite",
    )
    assert result.r_squared is not None
    assert result.r_squared > 0.9


def test_too_few_observations():
    result = compute_factor_correlation(
        _make_ic([0.1, 0.2]),
        {"short": _make_ic([0.3, 0.4])},
    )
    # Only 2 points — below the 5-point minimum for correlation
    assert result.correlations == []
```

### Verification

```bash
python -m pytest tests/test_factor_correlation.py -v
```

---

## Execution Order

Tasks 1, 3, 4 are independent — can run in parallel.
Task 2 should run after Task 4 (may reference correlation logic).

```
    ┌── Task 1 (HTTP live tests) ──┐
    │                               │
────┼── Task 3 (vault CLI)  ───────┼──→ Task 2 (feedback enhancement) ──→ Done
    │                               │
    └── Task 4 (factor_correlation) ┘
```

## Final verification (all tasks)

```bash
cd /home/yukun_zhao/quant/projects/alpha-lab
python -m pytest tests/test_web_unified_http.py tests/test_factor_correlation.py tests/test_vault_cli.py -v
python -m pytest tests/ -k "graph_feedback" -v
```
