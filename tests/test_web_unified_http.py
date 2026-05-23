"""HTTP handler smoke tests for web_unified.py.

Spins up a real ThreadingHTTPServer on a random port, makes urllib requests,
and validates status codes + JSON structure. Exercises actual route dispatch
rather than mocking the handler.
"""

from __future__ import annotations

import json
import socket
import threading
import urllib.error
import urllib.request
from collections.abc import Generator
from datetime import date, timedelta
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

from alpha_lab.web_unified import _RunRecord, _UnifiedRequestHandler, _UnifiedService

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _build_vault(tmp_path: Path) -> Path:
    vault = tmp_path / "quant-knowledge"
    for rel in [
        "00_inbox",
        "_sources",
        "10_concepts",
        "30_factors",
        "50_experiments",
        "55_projects",
        "90_moc",
    ]:
        (vault / rel).mkdir(parents=True, exist_ok=True)

    (vault / "90_moc" / "CARD-INDEX.tsv").write_text(
        "path\ttype\tname\tdomain\tlifecycle\ttags\tparent_moc\n"
        "30_factors/Factor - Momentum Base.md\tfactor\tMomentum Base\talpha_research\t"
        "theoretical\tmomentum\tMOC - Factors\n",
        encoding="utf-8",
    )
    (vault / "30_factors" / "Factor - Momentum Base.md").write_text(
        "---\ntype: factor\n---\n# 动量基类\n",
        encoding="utf-8",
    )
    (vault / "00_inbox" / "note.md").write_text("inbox note", encoding="utf-8")
    return vault


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _business_date_strings(start: str = "2024-01-02", n: int = 160) -> list[str]:
    current = date.fromisoformat(start)
    dates: list[str] = []
    while len(dates) < n:
        if current.weekday() < 5:
            dates.append(current.isoformat())
        current += timedelta(days=1)
    return dates


def _write_single_factor_fixture_inputs(base_dir: Path) -> dict[str, str]:
    inputs_dir = base_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    dates = _business_date_strings()
    assets = [f"A{i:03d}" for i in range(8)]

    price_lines = ["date,asset,close"]
    factor_lines = ["date,asset,factor,value"]
    universe_lines = ["date,asset,in_universe"]
    for asset_idx, asset in enumerate(assets):
        price = 20.0 + asset_idx
        for t, date_text in enumerate(dates):
            ret = 0.001 + 0.0004 * (((t + asset_idx) % 7) - 3)
            price = max(1.0, price * (1.0 + ret))
            signal = 0.2 * asset_idx + 0.05 * ((t % 11) - 5)
            price_lines.append(f"{date_text},{asset},{price:.6f}")
            factor_lines.append(f"{date_text},{asset},mom_5d,{signal:.6f}")
            universe_lines.append(f"{date_text},{asset},true")

    prices_path = inputs_dir / "prices.csv"
    factor_path = inputs_dir / "factor.csv"
    universe_path = inputs_dir / "universe.csv"
    prices_path.write_text("\n".join(price_lines) + "\n", encoding="utf-8")
    factor_path.write_text("\n".join(factor_lines) + "\n", encoding="utf-8")
    universe_path.write_text("\n".join(universe_lines) + "\n", encoding="utf-8")
    return {
        "prices_path": "./inputs/prices.csv",
        "factor_path": "./inputs/factor.csv",
        "universe_path": "./inputs/universe.csv",
    }


def _write_model_factor_fixture_inputs(base_dir: Path) -> None:
    base_dir.mkdir(parents=True, exist_ok=True)
    dates = _business_date_strings()
    assets = [f"{i:06d}.SZ" for i in range(1, 9)]
    feature_lines = ["date,asset,feature_a"]
    price_lines = ["date,asset,close"]
    for asset_idx, asset in enumerate(assets):
        price = 10.0 + asset_idx
        for t, date_text in enumerate(dates):
            feature = 0.1 * asset_idx + 0.03 * ((t % 13) - 6)
            ret = 0.001 + 0.0003 * (((t + asset_idx) % 9) - 4)
            price = max(1.0, price * (1.0 + ret))
            feature_lines.append(f"{date_text},{asset},{feature:.6f}")
            price_lines.append(f"{date_text},{asset},{price:.6f}")
    (base_dir / "features.csv").write_text("\n".join(feature_lines) + "\n", encoding="utf-8")
    (base_dir / "prices.csv").write_text("\n".join(price_lines) + "\n", encoding="utf-8")


def _build_model_candidate_payload(
    base_dir: Path,
    *,
    candidate_name: str = "http_model_candidate",
    feature_columns: list[str] | None = None,
) -> dict[str, object]:
    _write_model_factor_fixture_inputs(base_dir)
    columns = feature_columns or ["feature_a"]
    return {
        "contract_version": "stage2_model_candidate_v1",
        "candidate_name": candidate_name,
        "implementation_status": "draft_for_stage3",
        "implementation_type": "spec_variant",
        "source_mechanisms": ["http_test"],
        "base_case_spec_path": "",
        "expected_horizon": "t_plus_1_or_later",
        "data_contract": {
            "prices_required_columns": ["date", "asset", "close"],
            "feature_required_columns": columns,
            "feature_optional_columns": [],
            "feature_availability": {
                "mode": "safety_lag",
                "column": None,
                "safety_lag_days": 1,
            },
        },
        "risk_controls": {
            "feature_availability_pit": "safety lag",
            "label_leakage": "validator",
            "overfit_complexity": "simple ridge",
            "turnover_cost": "screening",
            "feature_instability": "review",
            "split_regime_fragility": "review",
        },
        "run_controls": {
            "evaluation_profile": "exploratory_screening",
            "screening_retrain_every_n_dates": 40,
            "vault_export_mode": "skip",
        },
        "case_spec_payload": {
            "name": f"{candidate_name}_case",
            "factor_name": candidate_name,
            "features_path": str(base_dir / "features.csv"),
            "feature_columns": columns,
            "prices_path": str(base_dir / "prices.csv"),
            "rebalance_frequency": "W",
            "n_quantiles": 5,
            "direction": "long",
            "universe": {"name": "default"},
            "target": {"kind": "forward_return", "horizon": 5},
            "feature_availability": {
                "mode": "safety_lag",
                "safety_lag_days": 1,
            },
            "feature_preprocess": {
                "missing_policy": "median_impute",
                "scale_features": "auto",
            },
            "model": {"family": "ridge", "params": {"alpha": 1.0}},
            "model_selection": {"enabled": False},
            "training": {
                "window_type": "rolling",
                "train_window_n_dates": 60,
                "min_train_dates": 40,
                "min_train_rows": 200,
                "retrain_every_n_dates": 5,
                "min_score_assets": 5,
            },
            "neutralization": {"enabled": False},
            "transaction_cost": {"one_way_rate": 0.001},
            "output": {"root_dir": str(base_dir / "outputs")},
        },
        "stage3_validation_focus": ["feature columns", "PIT", "artifact audit"],
    }


@pytest.fixture()
def live_server(tmp_path: Path) -> Generator[tuple[str, _UnifiedService], None, None]:
    """Start a real ThreadingHTTPServer; yield (base_url, service)."""
    vault = _build_vault(tmp_path)
    svc = _UnifiedService(vault_root=vault, workspace_root=tmp_path)

    class _Handler(_UnifiedRequestHandler):
        pass

    _Handler.svc = svc  # type: ignore[attr-defined]

    try:
        port = _free_port()
    except PermissionError:
        pytest.skip("socket creation denied in current environment")
    server = ThreadingHTTPServer(("127.0.0.1", port), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}", svc
    finally:
        server.shutdown()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get(base_url: str, path: str) -> tuple[int, dict | str]:
    """GET {base_url}{path}. Returns (status_code, parsed_json_or_body_text)."""
    req = urllib.request.Request(f"{base_url}{path}")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            body = resp.read()
            ct = resp.headers.get("Content-Type", "")
            if "json" in ct:
                return resp.status, json.loads(body)
            return resp.status, body.decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        body = e.read()
        try:
            return e.code, json.loads(body)
        except Exception:
            return e.code, body.decode("utf-8", errors="replace")


def _post(base_url: str, path: str, payload: dict) -> tuple[int, dict | str]:
    """POST JSON to {base_url}{path}."""
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{base_url}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            body = resp.read()
            ct = resp.headers.get("Content-Type", "")
            if "json" in ct:
                return resp.status, json.loads(body)
            return resp.status, body.decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        body = e.read()
        try:
            return e.code, json.loads(body)
        except Exception:
            return e.code, body.decode("utf-8", errors="replace")


def _delete(base_url: str, path: str) -> tuple[int, dict | str]:
    """DELETE {base_url}{path}."""
    req = urllib.request.Request(f"{base_url}{path}", method="DELETE")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            body = resp.read()
            ct = resp.headers.get("Content-Type", "")
            if "json" in ct:
                return resp.status, json.loads(body)
            return resp.status, body.decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        body = e.read()
        try:
            return e.code, json.loads(body)
        except Exception:
            return e.code, body.decode("utf-8", errors="replace")


def _patch(base_url: str, path: str, payload: dict) -> tuple[int, dict | str]:
    """PATCH JSON to {base_url}{path}."""
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{base_url}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="PATCH",
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            body = resp.read()
            ct = resp.headers.get("Content-Type", "")
            if "json" in ct:
                return resp.status, json.loads(body)
            return resp.status, body.decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        body = e.read()
        try:
            return e.code, json.loads(body)
        except Exception:
            return e.code, body.decode("utf-8", errors="replace")


def _put(base_url: str, path: str, payload: dict) -> tuple[int, dict | str]:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{base_url}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="PUT",
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            body = resp.read()
            ct = resp.headers.get("Content-Type", "")
            if "json" in ct:
                return resp.status, json.loads(body)
            return resp.status, body.decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        body = e.read()
        try:
            return e.code, json.loads(body)
        except Exception:
            return e.code, body.decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Root page
# ---------------------------------------------------------------------------


def test_root_returns_html(live_server: tuple[str, _UnifiedService]) -> None:
    base_url, _ = live_server
    status, body = _get(base_url, "/")
    assert status == 200
    assert isinstance(body, str)
    assert "<html" in body.lower()
    assert "Alpha Lab" in body


def test_model_lab_page_returns_html(live_server: tuple[str, _UnifiedService]) -> None:
    base_url, _ = live_server
    status, body = _get(base_url, "/model-lab")
    assert status == 200
    assert isinstance(body, str)
    assert "Model Lab" in body
    assert 'id="btnIdeaRecordResponse"' not in body
    assert 'id="explorerResponse"' not in body
    assert 'id="explorerIdea"' not in body
    assert 'id="btnIdeaExplore"' not in body
    assert 'id="explorerAgentPromptCards"' not in body
    assert "后端迭代优先" in body
    assert "Draft Candidates" in body
    assert "candidatePayloadInput" in body
    assert "/api/model-lab/candidates" in body
    assert "btnCandidateRun" in body
    assert "导入并运行完整报告" in body
    assert "Advanced Candidate Actions" in body


# ---------------------------------------------------------------------------
# /api/vault/stats
# ---------------------------------------------------------------------------


def test_vault_stats_route(live_server: tuple[str, _UnifiedService]) -> None:
    base_url, _ = live_server
    status, data = _get(base_url, "/api/vault/stats")
    assert status == 200
    assert isinstance(data, dict)
    assert data["total_cards"] == 1
    assert data["inbox_count"] == 1
    assert "by_type" in data
    assert "by_lifecycle" in data


# ---------------------------------------------------------------------------
# /api/vault/inbox
# ---------------------------------------------------------------------------


def test_vault_inbox_route(live_server: tuple[str, _UnifiedService]) -> None:
    base_url, _ = live_server
    status, data = _get(base_url, "/api/vault/inbox")
    assert status == 200
    assert isinstance(data, dict)
    assert data["count"] == 1
    assert any(item["name"] == "note.md" for item in data["items"])


# ---------------------------------------------------------------------------
# /api/vault/card/{name}
# ---------------------------------------------------------------------------


def test_read_card_route_found_bare_name(live_server: tuple[str, _UnifiedService]) -> None:
    base_url, _ = live_server
    status, data = _get(base_url, "/api/vault/card/Factor%20-%20Momentum%20Base.md")
    assert status == 200
    assert isinstance(data, dict)
    assert "动量基类" in data["content"]
    assert data["truncated"] is False


def test_read_card_route_found_vault_relative_path(
    live_server: tuple[str, _UnifiedService],
) -> None:
    # Vault-relative path as stored in CARD-INDEX.tsv — the main real-world case
    base_url, svc = live_server
    # Create a nested card
    nested_dir = svc.vault_root / "10_concepts" / "behavioral"
    nested_dir.mkdir(parents=True, exist_ok=True)
    (nested_dir / "Concept - Habit Formation.md").write_text(
        "# Habit Formation\n\ntest content\n", encoding="utf-8"
    )
    status, data = _get(
        base_url,
        "/api/vault/card/10_concepts%2Fbehavioral%2FConcept%20-%20Habit%20Formation.md",
    )
    assert status == 200
    assert isinstance(data, dict)
    assert "Habit Formation" in data["content"]


def test_read_card_route_not_found(live_server: tuple[str, _UnifiedService]) -> None:
    base_url, _ = live_server
    status, data = _get(base_url, "/api/vault/card/Factor%20-%20Nonexistent.md")
    assert status in (404, 500)
    assert isinstance(data, dict)
    assert data.get("ok") is False
    assert "error" in data


def test_read_card_route_traversal_rejected(live_server: tuple[str, _UnifiedService]) -> None:
    base_url, _ = live_server
    status, data = _get(base_url, "/api/vault/card/..%2F..%2Fetc%2Fpasswd")
    assert status in (400, 403, 404, 500)
    assert isinstance(data, dict)
    assert data.get("ok") is False


# ---------------------------------------------------------------------------
# /api/evaluation-profiles
# ---------------------------------------------------------------------------


def test_evaluation_profiles_route(live_server: tuple[str, _UnifiedService]) -> None:
    base_url, _ = live_server
    status, data = _get(base_url, "/api/evaluation-profiles")
    assert status == 200
    assert isinstance(data, dict)
    assert "profiles" in data
    assert "default_research" in data["profiles"]
    assert "default_profile" in data


# ---------------------------------------------------------------------------
# Removed legacy idea explorer routes
# ---------------------------------------------------------------------------


def test_legacy_vault_idea_explorer_routes_are_removed(
    live_server: tuple[str, _UnifiedService],
) -> None:
    base_url, _ = live_server

    for path, method in (
        ("/api/vault/explore-idea", "POST"),
        ("/api/vault/record-explore-response", "POST"),
        ("/api/vault/explore-sessions?limit=10", "GET"),
        ("/api/vault/explore-sessions/example", "GET"),
        ("/api/vault/explore-sessions/example", "DELETE"),
    ):
        if method == "GET":
            status, data = _get(base_url, path)
        elif method == "DELETE":
            status, data = _delete(base_url, path)
        else:
            status, data = _post(base_url, path, {"idea": "legacy"})
        assert status == 404
        assert isinstance(data, dict)
        assert data.get("ok") is False
# ---------------------------------------------------------------------------
# Project-scoped routes (need a project first)
# ---------------------------------------------------------------------------


@pytest.fixture()
def seeded_server(live_server: tuple[str, _UnifiedService]) -> tuple[str, _UnifiedService, str]:
    """Live server with one project + case seeded."""
    base_url, svc = live_server
    slug = "test-momentum"
    svc.create_project(
        {
            "slug": slug,
            "title_zh": "动量测试",
            "category": "factor_family",
            "owner": "test",
            "market": "ashare",
            "frequency": "daily",
            "chatgpt_project_name": "Test",
            "origin_cards": [],
        }
    )
    project_inputs = _write_single_factor_fixture_inputs(
        svc.vault_root / "55_projects" / slug
    )
    svc.create_case(
        slug,
        {
            "case_name": "mom_5d",
            "factor_name": "mom_5d",
            "base_method": "momentum",
            "lookback": 5,
            "skip_recent": 0,
            "target_horizon": 5,
            **project_inputs,
        },
    )
    return base_url, svc, slug


def _seed_succeeded_run(
    *,
    svc: _UnifiedService,
    tmp_path: Path,
    project_slug: str,
    run_id: str,
) -> _RunRecord:
    output_dir = tmp_path / f"run-{run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    spec_path = tmp_path / f"{run_id}.yaml"
    spec_path.write_text("name: mom_5d\nfactor_name: mom_5d\n", encoding="utf-8")

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps({"metrics": {"factor_name": "mom_5d", "mean_rank_ic": 0.03, "ic_t_stat": 2.2}}),
        encoding="utf-8",
    )
    (output_dir / "backtest_result.json").write_text(
        json.dumps({"stats": {"annualized_return": 0.11}}),
        encoding="utf-8",
    )
    (output_dir / "ic_timeseries.csv").write_text(
        "date,rank_ic\n2026-01-02,0.03\n2026-01-03,0.01\n",
        encoding="utf-8",
    )
    (output_dir / "rolling_stability.csv").write_text(
        "date,rolling_rank_ic\n2026-01-02,0.02\n2026-01-03,0.01\n",
        encoding="utf-8",
    )
    (output_dir / "ic_decay.csv").write_text(
        "horizon,mean_rank_ic\n1,0.03\n5,0.02\n",
        encoding="utf-8",
    )
    (output_dir / "quantile_returns.csv").write_text(
        "date,group,mean_return\n2026-01-02,Q1,0.01\n2026-01-02,Q5,-0.01\n",
        encoding="utf-8",
    )
    (output_dir / "factor_autocorrelation.csv").write_text(
        "date,autocorr\n2026-01-02,0.25\n",
        encoding="utf-8",
    )
    (output_dir / "turnover.csv").write_text(
        "date,turnover\n2026-01-02,0.18\n",
        encoding="utf-8",
    )

    record = _RunRecord(
        run_id=run_id,
        project_slug=project_slug,
        case_name="mom_5d",
        round_id=None,
        spec_path=str(spec_path),
        submitted_at_utc="2026-04-20T00:00:00Z",
        evaluation_profile="default_research",
        output_root_dir=None,
        render_report=True,
        status="succeeded",
        output_dir=str(output_dir),
        progress_percent=100,
        progress_message="运行完成",
        progress_events=[
            {"ts": "2026-04-20T00:00:01Z", "message": "step-1", "percent": 10},
            {"ts": "2026-04-20T00:00:02Z", "message": "step-2", "percent": 40},
            {"ts": "2026-04-20T00:00:03Z", "message": "step-3", "percent": 70},
            {"ts": "2026-04-20T00:00:04Z", "message": "step-4", "percent": 100},
        ],
        artifact_paths={
            "metrics": str(metrics_path),
            "backtest_result_json": str(output_dir / "backtest_result.json"),
            "ic_timeseries": str(output_dir / "ic_timeseries.csv"),
            "rolling_stability": str(output_dir / "rolling_stability.csv"),
            "ic_decay": str(output_dir / "ic_decay.csv"),
            "quantile_returns": str(output_dir / "quantile_returns.csv"),
            "factor_autocorrelation": str(output_dir / "factor_autocorrelation.csv"),
            "turnover": str(output_dir / "turnover.csv"),
        },
        summary={
            "factor_name": "mom_5d",
            "mean_rank_ic": 0.03,
            "mean_rank_ic_full": 0.031,
            "rank_ic_ir_full": 0.42,
            "long_short_ir_full": 0.7,
            "max_drawdown_full": -0.12,
            "ic_t_stat": 2.2,
            "extra_metric": "should_not_appear_in_compact",
        },
    )
    with svc.run_store._lock:  # noqa: SLF001 - tests intentionally seed in-memory store
        svc.run_store._records[run_id] = record  # noqa: SLF001
    return record


def _seed_model_lab_compare_run(
    *,
    svc: _UnifiedService,
    tmp_path: Path,
    run_id: str,
    factor_name: str,
    case_name: str,
    model_family: str,
    metrics: dict[str, object],
    top_features: list[str],
    rank_ic_values: list[float],
    integrity_summary: dict[str, object],
    note: str | None = None,
) -> None:
    output_dir = tmp_path / f"run-{run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    spec_path = tmp_path / f"{run_id}.yaml"
    spec_path.write_text(f"name: {run_id}\nfactor_name: {factor_name}\n", encoding="utf-8")

    ic_path = output_dir / "ic_timeseries.csv"
    ic_lines = ["date,ic,rank_ic"]
    for idx, value in enumerate(rank_ic_values, start=1):
        ic_lines.append(f"2026-01-{idx:02d},{value:.6f},{value:.6f}")
    ic_path.write_text("\n".join(ic_lines) + "\n", encoding="utf-8")

    turnover_path = output_dir / "turnover.csv"
    turnover_lines = ["date,turnover", "2026-01-01,0.61", "2026-01-02,0.58"]
    turnover_path.write_text("\n".join(turnover_lines) + "\n", encoding="utf-8")

    metrics_payload = {
        "metrics": {
            "factor_name": factor_name,
            **{str(key): value for key, value in metrics.items()},
            "model_family": model_family,
        },
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics_payload), encoding="utf-8")

    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "case_name": case_name,
                "integrity_summary": integrity_summary,
            }
        ),
        encoding="utf-8",
    )

    feature_importance_path = output_dir / "feature_importance.csv"
    feature_importance_path.write_text(
        "feature,mean_abs_importance,latest_importance\n"
        + "\n".join(
            f"{name},{idx + 1:.6f},{0.8 * idx:.6f}" for idx, name in enumerate(top_features)
        )
        + "\n",
        encoding="utf-8",
    )

    record = _RunRecord(
        run_id=run_id,
        project_slug="__model_lab__",
        case_name=case_name,
        round_id=None,
        spec_path=str(spec_path),
        submitted_at_utc="2026-01-01T00:00:00Z",
        evaluation_profile="default_research",
        output_root_dir=None,
        render_report=True,
        status="succeeded",
        output_dir=str(output_dir),
        progress_percent=100,
        progress_message="completed",
        summary={
            "factor_name": factor_name,
            "model_family": model_family,
            **{str(key): value for key, value in metrics.items()},
        },
        note=note,
        artifact_paths={
            "metrics": str(metrics_path),
            "run_manifest": str(manifest_path),
            "ic_timeseries": str(ic_path),
            "turnover": str(turnover_path),
            "feature_importance": str(feature_importance_path),
        },
        progress_events=[],
        workflow="model_factor",
    )
    with svc.run_store._lock:  # noqa: SLF001 - tests intentionally seed in-memory store
        svc.run_store._records[run_id] = record  # noqa: SLF001


def _seed_model_lab_overview_run(
    *,
    svc: _UnifiedService,
    tmp_path: Path,
    run_id: str,
    factor_name: str,
    case_name: str,
    model_family: str,
    notes: dict[str, object],
    integrity_checks: list[dict[str, object]],
) -> None:
    _seed_model_lab_compare_run(
        svc=svc,
        tmp_path=tmp_path,
        run_id=run_id,
        factor_name=factor_name,
        case_name=case_name,
        model_family=model_family,
        metrics=notes,
        top_features=["feat1", "feat2", "feat3", "feat4", "feat5"],
        rank_ic_values=[0.012, 0.013, 0.011],
        integrity_summary={
            "n_checks": len(integrity_checks),
            "n_pass": 1,
            "n_warn": 1,
            "n_fail": 1,
            "highest_severity": "warn",
        },
        note="overview test",
    )
    output_dir = tmp_path / f"run-{run_id}"
    decay_path = output_dir / "ic_decay.csv"
    decay_path.write_text(
        "\n".join(
            [
                "date,horizon,mean_ic,mean_rank_ic",
                "2026-01-01,1,0.015,0.010",
                "2026-01-01,3,0.012,0.009",
                "2026-01-01,5,0.010,0.007",
                "2026-01-01,10,0.007,0.005",
                "2026-01-01,20,0.004,0.003",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    group_path = output_dir / "group_returns.csv"
    group_path.write_text(
        "\n".join(
            [
                "date,group,group_return",
                "2026-01-01,Q1,0.012",
                "2026-01-01,Q5,-0.008",
                "2026-01-02,Q1,0.009",
                "2026-01-02,Q5,-0.010",
                "2026-01-03,Q1,-0.006",
                "2026-01-03,Q5,0.001",
                "2026-01-04,Q1,0.014",
                "2026-01-04,Q5,-0.004",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    backtest_path = output_dir / "backtest_result.json"
    backtest_path.write_text(
        json.dumps(
            {
                "summary": {
                    "nav_points": [
                        ["2026-01-01", 1.0],
                        ["2026-01-02", 1.011],
                        ["2026-01-03", 1.023],
                        ["2026-01-04", 1.036],
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    integrity_path = output_dir / "integrity_report.json"
    integrity_path.write_text(
        json.dumps(
            {
                "summary": {
                    "n_checks": len(integrity_checks),
                    "n_pass": 1,
                    "n_warn": 1,
                    "n_fail": 1,
                    "highest_severity": "warn",
                },
                "checks": integrity_checks,
            }
        ),
        encoding="utf-8",
    )
    with svc.run_store._lock:  # noqa: SLF001 - tests intentionally seed in-memory store
        record = svc.run_store._records[run_id]
        replacement = record.clone()
        replacement.artifact_paths = dict(
            replacement.artifact_paths,
            **{
                "ic_decay": str(decay_path),
                "group_returns": str(group_path),
                "backtest_result_json": str(backtest_path),
                "integrity_report_json": str(integrity_path),
            },
        )
        svc.run_store._records[run_id] = replacement  # noqa: SLF001


def _seed_draft(
    *,
    svc: _UnifiedService,
    slug: str,
    name: str,
) -> Path:
    drafts_dir = svc.vault_root / "55_projects" / slug / "50_writeback_drafts"
    drafts_dir.mkdir(parents=True, exist_ok=True)
    path = drafts_dir / name
    path.write_text(
        f"""---
project: {slug}
case_name: mom_5d
round_id: r01
review_status: pending
reviewed_by: ""
reviewed_at: ""
one_sentence_verdict: ""
status_lifecycle: ""
current_hypothesis: ""
current_focus: ""
next_action: ""
vault_export_mode: skip
---

# 导出草案（测试）

placeholder
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return path


def test_list_cases_route(seeded_server: tuple[str, _UnifiedService, str]) -> None:
    base_url, _, slug = seeded_server
    status, data = _get(base_url, f"/api/projects/{slug}/cases")
    assert status == 200
    assert isinstance(data, dict)
    assert data["project_slug"] == slug
    cases = data["cases"]
    assert len(cases) == 1
    assert cases[0]["case_name"] == "mom_5d"
    assert cases[0]["spec_exists"] is True


def test_list_rounds_route_empty(seeded_server: tuple[str, _UnifiedService, str]) -> None:
    base_url, _, slug = seeded_server
    status, data = _get(base_url, f"/api/projects/{slug}/rounds")
    assert status == 200
    assert isinstance(data, dict)
    assert data["project_slug"] == slug
    assert data["rounds"] == []


def test_create_round_route(seeded_server: tuple[str, _UnifiedService, str]) -> None:
    base_url, _, slug = seeded_server
    payload = {
        "topic": "三个月成交额加权动量",
        "mode": "standard",
    }
    status, data = _post(base_url, f"/api/projects/{slug}/rounds", payload)
    assert status == 201
    assert isinstance(data, dict)
    assert data["project"] == slug
    assert isinstance(data["round_id"], str)
    assert data["round_id"]

    status2, data2 = _get(base_url, f"/api/projects/{slug}/rounds")
    assert status2 == 200
    assert isinstance(data2, dict)
    assert data2["project_slug"] == slug
    assert isinstance(data2["rounds"], list)
    assert any(
        item["round_id"] == data["round_id"] and item["has_discussion_capture"]
        for item in data2["rounds"]
    )


def test_create_round_route_requires_topic(seeded_server: tuple[str, _UnifiedService, str]) -> None:
    base_url, _, slug = seeded_server
    status, data = _post(base_url, f"/api/projects/{slug}/rounds", {"mode": "standard"})
    assert status in (400, 422, 500)
    assert isinstance(data, dict)
    assert data.get("ok") is False


# ---------------------------------------------------------------------------
# 404 for unknown routes
# ---------------------------------------------------------------------------


def test_unknown_route_returns_404(live_server: tuple[str, _UnifiedService]) -> None:
    base_url, _ = live_server
    status, data = _get(base_url, "/api/definitely/not/a/real/route")
    assert status == 404
    assert isinstance(data, dict)
    assert data.get("ok") is False


# ---------------------------------------------------------------------------
# GET /api/dashboard
# ---------------------------------------------------------------------------


def test_dashboard_route(live_server: tuple[str, _UnifiedService], tmp_path: Path) -> None:
    base_url, svc = live_server
    _seed_succeeded_run(
        svc=svc,
        tmp_path=tmp_path,
        project_slug="dashboard-project",
        run_id="dashboard-run",
    )
    status, data = _get(base_url, "/api/dashboard")
    assert status == 200
    assert isinstance(data, dict)
    assert "project_count" in data
    assert "run_status_counts" in data
    assert "vault_card_count" in data
    assert data["run_status_counts"]["succeeded"] == 1
    assert isinstance(data["recent_runs"], list)
    assert len(data["recent_runs"]) == 1
    recent = data["recent_runs"][0]
    assert recent["_compact"] is True
    assert recent["summary"]["factor_name"] == "mom_5d"
    assert "extra_metric" not in recent["summary"]
    assert recent["artifact_paths"]["metrics"] is True


# ---------------------------------------------------------------------------
# GET/POST /api/settings/llm
# ---------------------------------------------------------------------------


def test_llm_settings_routes(
    live_server: tuple[str, _UnifiedService],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_url, _ = live_server
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.delenv("ALPHA_LAB_RESEARCH_BRIDGE_V2", raising=False)

    status, data = _get(base_url, "/api/settings/llm")
    assert status == 200
    assert isinstance(data, dict)
    assert data["anthropic_api_key_configured"] is False
    assert data["anthropic_api_key_source"] == "none"
    assert data["anthropic_base_url"] == ""
    assert data["anthropic_base_url_source"] == "default"

    status, data = _post(
        base_url,
        "/api/settings/llm",
        {
            "anthropic_api_key": "sk-ant-http-test",
            "anthropic_base_url": "https://anthropic-proxy.example/v1",
            "research_bridge_v2_enabled": True,
        },
    )

    assert status == 200
    assert isinstance(data, dict)
    assert data["anthropic_api_key_configured"] is True
    assert data["anthropic_api_key_source"] == "saved"
    assert data["anthropic_base_url"] == "https://anthropic-proxy.example/v1"
    assert data["anthropic_base_url_source"] == "saved"
    assert data["research_bridge_v2_enabled"] is True
    assert "anthropic_api_key" not in data
    settings_path = Path(str(data["settings_path"]))
    saved = json.loads(settings_path.read_text(encoding="utf-8"))
    assert saved["anthropic_api_key"] == "sk-ant-http-test"
    assert saved["anthropic_base_url"] == "https://anthropic-proxy.example/v1"

    status, data = _post(
        base_url,
        "/api/settings/llm",
        {
            "clear_anthropic_api_key": True,
            "clear_anthropic_base_url": True,
            "research_bridge_v2_enabled": False,
        },
    )
    assert status == 200
    assert isinstance(data, dict)
    assert data["anthropic_api_key_configured"] is False
    assert data["anthropic_base_url"] == ""
    assert data["research_bridge_v2_enabled"] is False


# ---------------------------------------------------------------------------
# GET /api/vault/graph/coverage
# ---------------------------------------------------------------------------


def test_graph_coverage_route(live_server: tuple[str, _UnifiedService]) -> None:
    base_url, _ = live_server
    status, data = _get(base_url, "/api/vault/graph/coverage")
    assert status == 200
    assert isinstance(data, dict)
    # May return ok=False if no graph.json, but should not crash
    assert "matrix" in data or "ok" in data


# ---------------------------------------------------------------------------
# GET /api/cards/search
# ---------------------------------------------------------------------------


def test_search_cards_route(live_server: tuple[str, _UnifiedService]) -> None:
    base_url, _ = live_server
    status, data = _get(base_url, "/api/cards/search?q=Momentum")
    assert status == 200
    assert isinstance(data, dict)
    assert isinstance(data["cards"], list)
    assert len(data["cards"]) >= 1
    assert data["query"] == "Momentum"


def test_search_cards_empty_query(live_server: tuple[str, _UnifiedService]) -> None:
    base_url, _ = live_server
    status, data = _get(base_url, "/api/cards/search?q=")
    assert status == 200
    assert isinstance(data, dict)
    assert isinstance(data["cards"], list)


# ---------------------------------------------------------------------------
# GET /api/categories
# ---------------------------------------------------------------------------


def test_categories_route(live_server: tuple[str, _UnifiedService]) -> None:
    base_url, _ = live_server
    status, data = _get(base_url, "/api/categories")
    assert status == 200
    assert isinstance(data, dict)
    assert isinstance(data["categories"], list)


# ---------------------------------------------------------------------------
# GET /api/projects/{slug}  (single project detail)
# ---------------------------------------------------------------------------


def test_get_project_detail_route(seeded_server: tuple[str, _UnifiedService, str]) -> None:
    base_url, _, slug = seeded_server
    status, data = _get(base_url, f"/api/projects/{slug}")
    assert status == 200
    assert isinstance(data, dict)
    assert data["project"]["slug"] == slug
    assert "cases" in data


def test_get_project_detail_not_found(live_server: tuple[str, _UnifiedService]) -> None:
    base_url, _ = live_server
    status, data = _get(base_url, "/api/projects/nonexistent-slug")
    assert status in (404, 500)
    assert isinstance(data, dict)
    assert data.get("ok") is False


# ---------------------------------------------------------------------------
# GET /api/projects/{slug}/runs  (list runs)
# ---------------------------------------------------------------------------


def test_list_runs_route(seeded_server: tuple[str, _UnifiedService, str]) -> None:
    base_url, _, slug = seeded_server
    status, data = _get(base_url, f"/api/projects/{slug}/runs")
    assert status == 200
    assert isinstance(data, dict)
    assert data["project_slug"] == slug
    assert isinstance(data["runs"], list)


def test_list_runs_compact_route(
    seeded_server: tuple[str, _UnifiedService, str], tmp_path: Path
) -> None:
    base_url, svc, slug = seeded_server
    seeded = _seed_succeeded_run(
        svc=svc,
        tmp_path=tmp_path,
        project_slug=slug,
        run_id="run-compact-1",
    )

    status, data = _get(base_url, f"/api/projects/{slug}/runs?compact=1")
    assert status == 200
    assert isinstance(data, dict)
    assert isinstance(data["runs"], list)

    compact = next((item for item in data["runs"] if item["run_id"] == seeded.run_id), None)
    assert compact is not None
    assert compact["_compact"] is True
    assert compact["summary"]["factor_name"] == "mom_5d"
    assert compact["summary"]["mean_rank_ic_full"] == 0.031
    assert compact["summary"]["rank_ic_ir_full"] == 0.42
    assert compact["summary"]["long_short_ir_full"] == 0.7
    assert compact["summary"]["max_drawdown_full"] == -0.12
    assert "extra_metric" not in compact["summary"]
    assert compact["artifact_paths"]["metrics"] is True
    assert len(compact["progress_events"]) == 2
    assert compact["progress_events"][-1]["message"] == "step-4"

    status_full, data_full = _get(base_url, f"/api/projects/{slug}/runs")
    assert status_full == 200
    full = next((item for item in data_full["runs"] if item["run_id"] == seeded.run_id), None)
    assert full is not None
    assert isinstance(full["artifact_paths"]["metrics"], str)


def test_create_run_route(seeded_server: tuple[str, _UnifiedService, str]) -> None:
    base_url, _, slug = seeded_server
    status, data = _post(base_url, f"/api/projects/{slug}/runs", {"case_name": "mom_5d"})
    assert status == 201
    assert isinstance(data, dict)
    assert data["project_slug"] == slug
    assert data["case_name"] == "mom_5d"
    assert isinstance(data["run_id"], str)
    run_id = data["run_id"]

    status2, data2 = _get(base_url, f"/api/projects/{slug}/runs/{run_id}")
    assert status2 == 200
    assert isinstance(data2, dict)
    assert data2["run_id"] == run_id
    assert data2["case_name"] == "mom_5d"


def test_summarize_run_route(
    seeded_server: tuple[str, _UnifiedService, str],
    tmp_path: Path,
) -> None:
    base_url, svc, slug = seeded_server
    run = _seed_succeeded_run(
        svc=svc,
        tmp_path=tmp_path,
        project_slug=slug,
        run_id="run-summary-1",
    )
    output_dir = Path(run.output_dir or "")
    (output_dir / "run_manifest.json").write_text(
        json.dumps({"run_id": run.run_id, "case_name": run.case_name}),
        encoding="utf-8",
    )

    status, data = _post(base_url, f"/api/projects/{slug}/runs/{run.run_id}/summarize", {})
    assert status == 200
    assert isinstance(data, dict)
    assert data["project"] == slug
    assert isinstance(data["summary_path"], str)
    assert isinstance(data["latest_path"], str)
    assert isinstance(data["decision_log_path"], str)
    assert data["summary_path"]
    assert data["latest_path"]
    assert isinstance(data["graph_feedback"], dict)
    assert "suggested_similar_to" in data["graph_feedback"]
    assert "correlation_summary" in data["graph_feedback"]

    status2, data2 = _get(base_url, f"/api/projects/{slug}/runs/{run.run_id}")
    assert status2 == 200
    assert isinstance(data2, dict)
    assert data2["summarize_feedback_path"]
    assert data2["summarize_draft_path"]
    assert data2["summarize_state_patch_path"]


def test_list_projects_route(live_server: tuple[str, _UnifiedService]) -> None:
    base_url, svc = live_server
    svc.create_project(
        {
            "slug": "http-list-proj",
            "title_zh": "HTTP 列表测试",
            "category": "factor_recipe",
            "owner": "test",
            "market": "ashare",
            "frequency": "daily",
            "chatgpt_project_name": "HTTP List",
            "origin_cards": [],
        }
    )
    status, data = _get(base_url, "/api/projects")
    assert status == 200
    assert isinstance(data, dict)
    assert "projects" in data
    assert any(item["slug"] == "http-list-proj" for item in data["projects"])


def test_model_lab_spec_routes(live_server: tuple[str, _UnifiedService], tmp_path: Path) -> None:
    base_url, svc = live_server
    specs_dir = svc.workspace_root / "configs" / "real_cases" / "model_factor"
    specs_dir.mkdir(parents=True, exist_ok=True)
    spec_path = specs_dir / "http_model_lab.yaml"
    spec_path.write_text(
        "\n".join(
            [
                "name: http_model_lab_case",
                "factor_name: http_model_lab",
                "features_path: ./features.csv",
                "feature_columns: [feature_a]",
                "prices_path: ./prices.csv",
                "rebalance_frequency: W",
                "n_quantiles: 5",
                "direction: long",
                "universe: {name: default}",
                "target: {kind: forward_return, horizon: 5}",
                "feature_preprocess: {missing_policy: median_impute, scale_features: auto}",
                "model: {family: ridge, params: {alpha: 1.0}}",
                "training:",
                "  window_type: rolling",
                "  train_window_n_dates: 60",
                "  min_train_dates: 40",
                "  min_train_rows: 200",
                "  retrain_every_n_dates: 5",
                "  min_score_assets: 5",
                "neutralization: {enabled: false}",
                "transaction_cost: {one_way_rate: 0.001}",
                "output: {root_dir: ./outputs}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    status, data = _get(base_url, "/api/model-lab/specs")
    assert status == 200
    assert isinstance(data, dict)
    assert any(item["name"] == "http_model_lab.yaml" for item in data["specs"])

    status2, data2 = _get(base_url, "/api/model-lab/specs/http_model_lab.yaml")
    assert status2 == 200
    assert isinstance(data2, dict)
    assert data2["name"] == "http_model_lab.yaml"
    assert "factor_name: http_model_lab" in data2["content"]

    updated = str(data2["content"]).replace("alpha: 1.0", "alpha: 3.0")
    status3, data3 = _put(
        base_url,
        "/api/model-lab/specs/http_model_lab.yaml",
        {"content": updated},
    )
    assert status3 == 200
    assert isinstance(data3, dict)
    assert data3["ok"] is True
    assert "alpha: 3.0" in spec_path.read_text(encoding="utf-8")



def test_legacy_model_lab_idea_explorer_routes_are_removed(
    live_server: tuple[str, _UnifiedService],
) -> None:
    base_url, _ = live_server

    for path, method in (
        ("/api/model-lab/idea-explorer/explore", "POST"),
        ("/api/model-lab/idea-explorer/record-response", "POST"),
        ("/api/model-lab/idea-explorer/apply-patch-hint", "POST"),
        ("/api/model-lab/idea-explorer/sessions?limit=10", "GET"),
        ("/api/model-lab/idea-explorer/sessions/example", "GET"),
        ("/api/model-lab/idea-explorer/sessions/example", "DELETE"),
    ):
        if method == "GET":
            status, data = _get(base_url, path)
        elif method == "DELETE":
            status, data = _delete(base_url, path)
        else:
            status, data = _post(base_url, path, {"idea": "legacy"})
        assert status == 404
        assert isinstance(data, dict)
        assert data.get("ok") is False

def test_model_lab_run_compare_route(live_server: tuple[str, _UnifiedService]) -> None:
    base_url, svc = live_server

    _seed_model_lab_compare_run(
        svc=svc,
        tmp_path=svc.workspace_root,
        run_id="mrun-http-a",
        factor_name="factor_a",
        case_name="case_a",
        model_family="ridge",
        metrics={
            "mean_ic": 0.01,
            "ic_ir": 0.20,
            "mean_rank_ic": 0.012,
            "rank_ic_ir": 0.18,
            "mean_long_short_turnover": 0.6,
            "long_short_ir": 0.32,
        },
        top_features=["feat1", "feat2", "feat3", "feat4", "feat5"],
        rank_ic_values=[0.01, 0.02],
        integrity_summary={
            "n_checks": 2,
            "n_pass": 1,
            "n_warn": 1,
            "n_fail": 0,
            "highest_severity": "warn",
        },
    )
    _seed_model_lab_compare_run(
        svc=svc,
        tmp_path=svc.workspace_root,
        run_id="mrun-http-b",
        factor_name="factor_b",
        case_name="case_b",
        model_family="gbdt",
        metrics={
            "mean_ic": 0.02,
            "ic_ir": 0.31,
            "mean_rank_ic": 0.015,
            "rank_ic_ir": 0.23,
            "mean_long_short_turnover": 0.58,
            "long_short_ir": 0.28,
        },
        top_features=["feat3", "feat4", "feat9", "feat2", "feat11"],
        rank_ic_values=[0.03, 0.01],
        integrity_summary={
            "n_checks": 1,
            "n_pass": 1,
            "n_warn": 0,
            "n_fail": 0,
            "highest_severity": "pass",
        },
    )

    status, data = _post(
        base_url,
        "/api/model-lab/runs/compare",
        {"run_ids": ["mrun-http-a", "mrun-http-b"], "top_k_features": 4},
    )
    assert status == 200
    assert isinstance(data, dict)
    assert data["ok"] is True
    assert data["run_count"] == 2
    assert data["case_names"] == ["case_a", "case_b"]
    assert data["case_name_by_run_id"] == {
        "mrun-http-a": "case_a",
        "mrun-http-b": "case_b",
    }
    assert len(data["metric_rows"]) == 2
    assert isinstance(data["run_failures"], list)
    assert len(data["run_failures"]) == 2
    assert {item["case_name"] for item in data["run_failures"]} == {"case_a", "case_b"}
    assert data["feature_stability"]["pair_count"] == 1
    assert isinstance(data["ic_series"], list)
    assert isinstance(data["turnover_series"], list)
    assert len(data["turnover_series"]) > 0
    assert isinstance(data["leakage"]["runs"], list)
    assert data["leakage"]["severity_by_run"] == {
        "mrun-http-a": "warn",
        "mrun-http-b": "pass",
    }


def test_model_lab_run_overview_route_partition_rows(
    live_server: tuple[str, _UnifiedService],
    tmp_path: Path,
) -> None:
    base_url, svc = live_server
    run_id = "mrun-http-overview-partition"
    _seed_model_lab_overview_run(
        svc=svc,
        tmp_path=tmp_path,
        run_id=run_id,
        factor_name="factor_overview2",
        case_name="case_overview2",
        model_family="ridge",
        notes={
            "mean_ic": 0.02,
            "ic_ir": 0.21,
            "mean_rank_ic": 0.018,
            "rank_ic_ir": 0.20,
            "mean_long_short_turnover": 0.53,
            "long_short_ir": 0.30,
            "factor_name": "factor_overview2",
            "model_family": "ridge",
        },
        integrity_checks=[
            {
                "check_name": "lag_check",
                "status": "pass",
                "severity": "pass",
                "module_name": "demo",
                "object_name": "o1",
            }
        ],
    )
    status, data = _get(base_url, f"/api/model-lab/runs/{run_id}/overview")
    assert status == 200
    snapshot = data["snapshot"]
    assert isinstance(snapshot["industryRows"], list)
    assert isinstance(snapshot["sizeRows"], list)
    assert isinstance(snapshot["regimeRows"], list)
    assert isinstance(snapshot["backtest"], dict)


def test_model_lab_runs_route_supports_query_filtering(
    live_server: tuple[str, _UnifiedService],
) -> None:
    base_url, svc = live_server
    _seed_model_lab_compare_run(
        svc=svc,
        tmp_path=svc.workspace_root,
        run_id="mrun-http-filter-a",
        factor_name="factor_a",
        case_name="case_momentum",
        model_family="ridge",
        metrics={"mean_ic": 0.01},
        top_features=["feat1", "feat2"],
        rank_ic_values=[0.01, 0.02],
        integrity_summary={
            "n_checks": 1,
            "n_pass": 1,
            "n_warn": 0,
            "n_fail": 0,
            "highest_severity": "pass",
        },
        note="note alpha",
    )
    _seed_model_lab_compare_run(
        svc=svc,
        tmp_path=svc.workspace_root,
        run_id="mrun-http-filter-b",
        factor_name="factor_b",
        case_name="case_value",
        model_family="gbdt",
        metrics={"mean_ic": 0.02},
        top_features=["feat3", "feat4"],
        rank_ic_values=[0.02, -0.01],
        integrity_summary={
            "n_checks": 1,
            "n_pass": 1,
            "n_warn": 0,
            "n_fail": 0,
            "highest_severity": "pass",
        },
        note="beta note",
    )

    with svc.run_store._lock:  # noqa: SLF001
        record = svc.run_store._records["mrun-http-filter-b"]
        failed = record.clone()
        failed.status = "failed"
        svc.run_store._records["mrun-http-filter-b"] = failed

    status, filtered = _get(base_url, "/api/model-lab/runs?compact=1&status=failed")
    assert status == 200
    assert isinstance(filtered, dict)
    assert isinstance(filtered["runs"], list)
    assert len(filtered["runs"]) == 1
    assert filtered["runs"][0]["run_id"] == "mrun-http-filter-b"
    assert filtered["runs"][0]["factor_name"] == "factor_b"
    assert filtered["runs"][0]["evaluation_title"] == "failed"

    status_case, filtered_case = _get(base_url, "/api/model-lab/runs?compact=1&case=momentum")
    assert status_case == 200
    assert isinstance(filtered_case, dict)
    assert [item["run_id"] for item in filtered_case["runs"]] == ["mrun-http-filter-a"]
    compact_summary = filtered_case["runs"][0]["summary"]
    assert compact_summary["model_family"] == "ridge"
    assert compact_summary["mean_ic"] == 0.01
    assert filtered_case["runs"][0]["factor_name"] == "factor_a"
    assert filtered_case["runs"][0]["evaluation_title"] == "completed"

    status_note, filtered_note = _get(base_url, "/api/model-lab/runs?compact=1&note=beta")
    assert status_note == 200
    assert isinstance(filtered_note, dict)
    assert len(filtered_note["runs"]) == 1
    assert filtered_note["runs"][0]["run_id"] == "mrun-http-filter-b"


def test_model_lab_run_export_card_route(
    live_server: tuple[str, _UnifiedService],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_url, svc = live_server
    _seed_model_lab_compare_run(
        svc=svc,
        tmp_path=svc.workspace_root,
        run_id="mrun-http-export",
        factor_name="factor_export",
        case_name="case_export",
        model_family="ridge",
        metrics={"mean_ic": 0.02},
        top_features=["feat1", "feat2"],
        rank_ic_values=[0.01, 0.02],
        integrity_summary={
            "n_checks": 1,
            "n_pass": 1,
            "n_warn": 0,
            "n_fail": 0,
            "highest_severity": "pass",
        },
        note="export",
    )
    output_dir = svc.workspace_root / "run-mrun-http-export"
    (output_dir / "experiment_card.md").write_text("# experiment card", encoding="utf-8")
    (output_dir / "summary.md").write_text("summary", encoding="utf-8")
    (output_dir / "run_manifest.json").write_text('{"case_name":"case_export"}', encoding="utf-8")

    observed: dict[str, object] = {}

    def _fake_export_to_vault(
        *,
        source_paths: dict[str, Path],
        case_name: str,
        vault_root: Path,
        mode: str = "versioned",
    ) -> SimpleNamespace:
        observed["source_paths"] = source_paths
        observed["case_name"] = case_name
        observed["mode"] = mode
        observed["vault_root"] = vault_root
        return SimpleNamespace(
            status="success",
            success=True,
            target_paths=(str(Path("/tmp") / "experiment-card.md"),),
            mode_used=mode,
            error=None,
        )

    monkeypatch.setattr("alpha_lab.web_unified._service.export_to_vault", _fake_export_to_vault)

    status, payload = _post(
        base_url,
        "/api/model-lab/runs/mrun-http-export/export-card",
        {"mode": "versioned"},
    )
    assert status == 200
    assert isinstance(payload, dict)
    assert payload["ok"] is True
    assert payload["success"] is True
    assert payload["run_id"] == "mrun-http-export"
    assert payload["mode_used"] == "versioned"
    assert payload["target_paths"] == ["/tmp/experiment-card.md"]
    assert observed["case_name"] == "case_export"


def test_model_lab_run_overview_route(
    live_server: tuple[str, _UnifiedService],
    tmp_path: Path,
) -> None:
    base_url, svc = live_server
    run_id = "mrun-http-overview"
    _seed_model_lab_overview_run(
        svc=svc,
        tmp_path=tmp_path,
        run_id=run_id,
        factor_name="factor_overview",
        case_name="case_overview",
        model_family="ridge",
        notes={
            "mean_ic": 0.018,
            "ic_ir": 0.25,
            "mean_rank_ic": 0.021,
            "rank_ic_ir": 0.19,
            "mean_long_short_turnover": 0.54,
            "long_short_ir": 0.31,
            "factor_name": "factor_overview",
            "model_family": "ridge",
        },
        integrity_checks=[
            {
                "check_name": "lag_check",
                "status": "warn",
                "severity": "warn",
                "module_name": "demo",
                "object_name": "o1",
            },
            {
                "check_name": "lookahead",
                "status": "pass",
                "severity": "pass",
                "module_name": "demo",
                "object_name": "o2",
            },
        ],
    )
    status, data = _get(base_url, f"/api/model-lab/runs/{run_id}/overview")
    assert status == 200
    assert isinstance(data, dict)
    assert data["ok"] is True
    assert data["run_id"] == run_id
    summary = data["summary"]
    assert summary["factor_name"] == "factor_overview"
    assert summary["model_family"] == "ridge"
    snapshot = data["snapshot"]
    assert isinstance(snapshot["backtest"], dict)
    assert isinstance(snapshot["icRows"], list)
    assert isinstance(snapshot["decayRows"], list)
    assert isinstance(snapshot["groupRows"], list)
    assert isinstance(snapshot["turnoverRows"], list)
    assert isinstance(snapshot["integrity"], dict)
    integrity = snapshot["integrity"]
    assert integrity["integrity_summary"]["n_checks"] == 2
    assert isinstance(integrity["integrity_checks"], list)
    assert integrity["integrity_checks"][0]["check_name"] == "lag_check"


def test_model_lab_run_overview_route_not_found(live_server: tuple[str, _UnifiedService]) -> None:
    base_url, _ = live_server
    status, data = _get(base_url, "/api/model-lab/runs/not-exists/overview")
    assert status == 404
    assert isinstance(data, dict)
    assert data.get("ok") is False


def test_model_lab_dev_overview_fixture_route(
    live_server: tuple[str, _UnifiedService],
) -> None:
    base_url, _ = live_server

    status, html = _get(
        base_url,
        "/dev/model-lab/overview-fixture?case=strong_skipped_extreme_nav",
    )
    assert status == 200
    assert isinstance(html, str)
    assert "overviewFixtureIdFromLocation" in html

    status, listing = _get(base_url, "/api/dev/model-lab/overview-fixtures")
    assert status == 200
    assert isinstance(listing, dict)
    assert listing["ok"] is True
    assert any(item["id"] == "strong_skipped_extreme_nav" for item in listing["fixtures"])

    status, data = _get(
        base_url,
        "/api/dev/model-lab/overview-fixtures/strong_skipped_extreme_nav",
    )
    assert status == 200
    assert isinstance(data, dict)
    assert data["ok"] is True
    assert data["fixture_id"] == "strong_skipped_extreme_nav"
    assert data["summary"]["portfolio_validation_status"] == "skipped_not_promoted"
    snapshot = data["snapshot"]
    assert snapshot["portfolioValidation"]["summary"]["validation_status"] == "skipped_not_promoted"
    assert len(snapshot["coverageRows"]) >= 10
    assert snapshot["industryRows"] == []
    assert snapshot["sizeRows"] == []
    assert snapshot["regimeRows"] == []


def test_alpha_lab_dev_overview_fixture_routes(
    live_server: tuple[str, _UnifiedService],
) -> None:
    base_url, _ = live_server

    status, html = _get(
        base_url,
        "/dev/alpha-lab/overview-fixture?case=quick_screening",
    )
    assert status == 200
    assert isinstance(html, str)
    assert "alphaLabOverviewFixtureIdFromLocation" in html

    status, alias_html = _get(
        base_url,
        "/dev/alpha-lab1/overview-fixture?case=full_evaluation",
    )
    assert status == 200
    assert isinstance(alias_html, str)
    assert "loadAlphaLabOverviewFixture" in alias_html

    status, listing = _get(base_url, "/api/dev/alpha-lab/overview-fixtures")
    assert status == 200
    assert isinstance(listing, dict)
    assert listing["ok"] is True
    fixture_ids = {item["id"] for item in listing["fixtures"]}
    assert {"quick_screening", "full_evaluation"}.issubset(fixture_ids)

    status, quick = _get(
        base_url,
        "/api/dev/alpha-lab/overview-fixtures/quick_screening",
    )
    assert status == 200
    assert isinstance(quick, dict)
    assert quick["ok"] is True
    assert quick["fixture_id"] == "quick_screening"
    assert quick["summary"]["research_evaluation_profile"] == "quick_screening"
    assert len(quick["snapshot"]["coverageRows"]) >= 8

    status, full = _get(
        base_url,
        "/api/dev/alpha-lab1/overview-fixtures/full_evaluation",
    )
    assert status == 200
    assert isinstance(full, dict)
    assert full["ok"] is True
    assert full["fixture_id"] == "full_evaluation"
    assert full["summary"]["research_evaluation_profile"] == "stricter_research"
    assert full["snapshot"]["portfolioValidation"]["summary"]["validation_status"] == "completed"


def test_model_lab_dev_artifact_fixture_routes(
    live_server: tuple[str, _UnifiedService],
) -> None:
    base_url, _ = live_server

    status, html = _get(
        base_url,
        "/dev/model-lab/artifact-fixture?case=strong_skipped_extreme_nav&artifact=training_log",
    )
    assert status == 200
    assert isinstance(html, str)
    assert "artifactFixtureParamsFromLocation" in html

    status, listing = _get(base_url, "/api/dev/model-lab/artifact-fixtures")
    assert status == 200
    assert isinstance(listing, dict)
    assert listing["ok"] is True
    assert any(item["id"] == "strong_skipped_extreme_nav" for item in listing["fixtures"])

    status, metrics = _get(
        base_url,
        "/api/dev/model-lab/artifact-fixtures/strong_skipped_extreme_nav/artifact/metrics",
    )
    assert status == 200
    assert isinstance(metrics, dict)
    assert metrics["metrics"]["portfolio_validation_status"] == "skipped_not_promoted"

    status, training_log = _get(
        base_url,
        "/api/dev/model-lab/artifact-fixtures/strong_skipped_extreme_nav/artifact/training_log",
    )
    assert status == 200
    assert isinstance(training_log, str)
    assert "score_date,status,model_version" in training_log
    assert "fit_scored" in training_log

    status, feature_importance = _get(
        base_url,
        "/api/dev/model-lab/artifact-fixtures/strong_skipped_extreme_nav/artifact/feature_importance",
    )
    assert status == 200
    assert isinstance(feature_importance, str)
    assert "turnover_rate_f" in feature_importance

    status, diag_html = _get(
        base_url,
        "/dev/model-lab/diagnostics-fixture?case=strong_skipped_extreme_nav",
    )
    assert status == 200
    assert isinstance(diag_html, str)
    assert "diagnosticsFixtureIdFromLocation" in diag_html

    status, diagnostics = _get(
        base_url,
        "/api/dev/model-lab/artifact-fixtures/strong_skipped_extreme_nav/artifact/diagnostics",
    )
    assert status == 200
    assert isinstance(diagnostics, dict)
    assert diagnostics["run_summary"]["highest_severity"] == "warn"


def test_model_lab_candidate_routes(
    live_server: tuple[str, _UnifiedService],
) -> None:
    base_url, svc = live_server

    status_empty, empty = _get(base_url, "/api/model-lab/candidates")
    assert status_empty == 200
    assert isinstance(empty, dict)
    assert empty["candidates"] == []

    payload = _build_model_candidate_payload(
        svc.workspace_root / "candidate_http_inputs",
        candidate_name="http_model_candidate",
    )
    status_post, saved = _post(
        base_url,
        "/api/model-lab/candidates",
        {"model_candidate_payload": payload},
    )
    assert status_post == 201
    assert isinstance(saved, dict)
    assert saved["name"] == "http_model_candidate"
    candidate_path = (
        svc.workspace_root
        / "custom_models"
        / "research"
        / "http_model_candidate"
        / "model_candidate.json"
    )
    assert candidate_path.exists()
    assert candidate_path.with_name("research_log.md").exists()

    status_validate, validation = _post(
        base_url,
        "/api/model-lab/candidates/http_model_candidate/validate",
        {},
    )
    assert status_validate == 200
    assert isinstance(validation, dict)
    assert validation["ok"] is True
    assert validation["candidate_json_sha256"]
    assert validation["case_spec_sha256"]
    assert validation["feature_contract_sha256"]

    status_mat_1, mat_1 = _post(
        base_url,
        "/api/model-lab/candidates/http_model_candidate/materialize-spec",
        {},
    )
    assert status_mat_1 == 200
    assert isinstance(mat_1, dict)
    assert mat_1["ok"] is True
    assert mat_1["name"] == "http_model_candidate_v1.yaml"
    assert (svc.model_lab_specs_root / "http_model_candidate_v1.yaml").exists()

    status_mat_2, mat_2 = _post(
        base_url,
        "/api/model-lab/candidates/http_model_candidate/materialize-spec",
        {},
    )
    assert status_mat_2 == 200
    assert isinstance(mat_2, dict)
    assert mat_2["name"] == "http_model_candidate_v2.yaml"

    bad_payload = _build_model_candidate_payload(
        svc.workspace_root / "candidate_http_bad_inputs",
        candidate_name="http_model_bad",
        feature_columns=["feature_missing"],
    )
    status_bad_post, _ = _post(
        base_url,
        "/api/model-lab/candidates",
        {"model_candidate_payload": bad_payload},
    )
    assert status_bad_post == 201
    status_bad_validate, bad_validation = _post(
        base_url,
        "/api/model-lab/candidates/http_model_bad/validate",
        {},
    )
    assert status_bad_validate == 200
    assert isinstance(bad_validation, dict)
    assert bad_validation["ok"] is False
    assert any(
        issue["code"] == "feature_columns_not_in_features"
        for issue in bad_validation["errors"]
    )

    status_run, run_payload = _post(
        base_url,
        "/api/model-lab/candidates/http_model_candidate/run",
        {},
    )
    assert status_run == 200
    assert isinstance(run_payload, dict)
    assert run_payload["ok"] is True
    submitted = run_payload["run"]
    assert isinstance(submitted, dict)
    assert submitted["draft_model_candidate_name"] == "http_model_candidate"
    assert submitted["draft_model_candidate_path"] == str(candidate_path)
    record = svc.run_store.get(str(submitted["run_id"]))
    assert record is not None
    assert record.draft_model_candidate_path == str(candidate_path)
    assert record.evaluation_profile == "default_research"


def test_model_lab_run_routes_run_note_and_duplicate_spec_diff(
    live_server: tuple[str, _UnifiedService],
    tmp_path: Path,
) -> None:
    base_url, svc = live_server
    specs_dir = svc.workspace_root / "configs" / "real_cases" / "model_factor"
    specs_dir.mkdir(parents=True, exist_ok=True)
    _write_model_factor_fixture_inputs(specs_dir)
    spec_path = specs_dir / "http_model_lab_route.yaml"
    spec_path.write_text(
        "\n".join(
            [
                "name: http_model_lab_case",
                "factor_name: http_model_lab",
                "features_path: ./features.csv",
                "feature_columns: [feature_a]",
                "prices_path: ./prices.csv",
                "rebalance_frequency: W",
                "n_quantiles: 5",
                "direction: long",
                "universe: {name: default}",
                "target: {kind: forward_return, horizon: 5}",
                "feature_preprocess: {missing_policy: median_impute, scale_features: auto}",
                "model: {family: ridge, params: {alpha: 1.0}}",
                "training:",
                "  window_type: rolling",
                "  train_window_n_dates: 60",
                "  min_train_dates: 40",
                "  min_train_rows: 200",
                "  retrain_every_n_dates: 5",
                "  min_score_assets: 5",
                "neutralization: {enabled: false}",
                "transaction_cost: {one_way_rate: 0.001}",
                "output: {root_dir: ./outputs}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    status_post, data_post = _post(
        base_url,
        "/api/model-lab/runs",
        {"spec_name": "http_model_lab_route.yaml", "note": "route test"},
    )
    assert status_post == 200
    assert isinstance(data_post, dict)
    assert data_post["note"] == "route test" if isinstance(data_post, dict) else True
    assert data_post["status"] in {"queued", "running", "succeeded", "failed"}

    status_dup, data_dup = _put(
        base_url,
        "/api/model-lab/specs/http_model_lab_route.yaml/duplicate",
        {"target_name": "route_copy.yaml"},
    )
    assert status_dup == 200
    assert isinstance(data_dup, dict)
    assert data_dup["ok"] is True
    assert data_dup["source"] == "http_model_lab_route.yaml"

    status_diff, data_diff = _post(
        base_url,
        "/api/model-lab/specs/diff",
        {"left": "http_model_lab_route.yaml", "right": data_dup["name"]},
    )
    assert status_diff == 200
    assert isinstance(data_diff, dict)
    assert data_diff["left"] == "http_model_lab_route.yaml"
    assert data_diff["right"] == data_dup["name"]
    assert data_diff["has_difference"] is True


def test_model_lab_source_routes(live_server: tuple[str, _UnifiedService]) -> None:
    base_url, _ = live_server

    status, data = _get(base_url, "/api/model-lab/sources")
    assert status == 200
    assert isinstance(data, dict)
    assert any(item["key"] == "core" for item in data["sources"])

    status2, data2 = _get(base_url, "/api/model-lab/sources/core")
    assert status2 == 200
    assert isinstance(data2, dict)
    assert data2["key"] == "core"
    assert "build_model_factor" in data2["content"]


def test_list_drafts_route(seeded_server: tuple[str, _UnifiedService, str]) -> None:
    base_url, _, slug = seeded_server
    status, data = _get(base_url, f"/api/projects/{slug}/drafts")
    assert status == 200
    assert isinstance(data, dict)
    assert data["project_slug"] == slug
    assert isinstance(data["drafts"], list)


def test_patch_draft_route(seeded_server: tuple[str, _UnifiedService, str]) -> None:
    base_url, svc, slug = seeded_server
    draft_path = _seed_draft(svc=svc, slug=slug, name="http-draft-review.md")
    payload = {
        "review_status": "approved",
        "reviewed_by": "alpha-tester",
        "reviewed_at": "now",
        "one_sentence_verdict": "继续推进。",
    }
    status, data = _patch(base_url, f"/api/projects/{slug}/drafts/{draft_path.name}", payload)
    assert status == 200
    assert isinstance(data, dict)
    assert data["name"] == draft_path.name
    assert data["review_status"] == "approved"
    assert data["reviewed_by"] == "alpha-tester"
    assert data["reviewed_at"] not in {"", None}

    status2, data2 = _get(base_url, f"/api/projects/{slug}/drafts/{draft_path.name}")
    assert status2 == 200
    fm = data2["frontmatter"]
    assert fm["review_status"] == "approved"
    assert fm["reviewed_by"] == "alpha-tester"


def test_patch_draft_legacy_route(seeded_server: tuple[str, _UnifiedService, str]) -> None:
    base_url, svc, slug = seeded_server
    draft_path = _seed_draft(svc=svc, slug=slug, name="http-draft-legacy.md")
    payload = {
        "draft_name": draft_path.name,
        "review_status": "approved",
        "reviewed_by": "legacy-tester",
        "reviewed_at": "now",
        "one_sentence_verdict": "兼容性覆盖。",
    }
    status, data = _post(base_url, f"/api/projects/{slug}/drafts/patch", payload)
    assert status == 200
    assert isinstance(data, dict)
    assert data["name"] == draft_path.name
    assert data["review_status"] == "approved"
    assert data["reviewed_by"] == "legacy-tester"


def test_patch_draft_legacy_route_missing_name(
    seeded_server: tuple[str, _UnifiedService, str],
) -> None:
    base_url, _, slug = seeded_server
    status, data = _post(
        base_url, f"/api/projects/{slug}/drafts/patch", {"review_status": "approved"}
    )
    assert status in (400, 422, 500)
    assert isinstance(data, dict)
    assert data.get("ok") is False


def test_apply_draft_route_invokes_writeback(
    seeded_server: tuple[str, _UnifiedService, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_url, svc, slug = seeded_server
    draft_path = _seed_draft(svc=svc, slug=slug, name="http-draft-apply.md")
    observed: dict[str, object] = {}

    def _fake_apply_writeback(*, vault_root, project_slug, draft_path, mode=None):
        del vault_root
        observed["project_slug"] = project_slug
        observed["draft_path"] = str(draft_path)
        observed["mode"] = mode
        assert project_slug == slug
        return SimpleNamespace(
            project=SimpleNamespace(slug=slug),
            draft_path=Path(draft_path),
            export_result=SimpleNamespace(
                status="success",
                success=True,
                target_paths=["/tmp/export1", "/tmp/export2"],
                mode_used=mode or "versioned",
                error=None,
            ),
        )

    monkeypatch.setattr("alpha_lab.web_unified._service.apply_writeback", _fake_apply_writeback)
    status, data = _post(base_url, f"/api/projects/{slug}/drafts/{draft_path.name}/apply", {})
    assert status == 200
    assert isinstance(data, dict)
    assert observed["project_slug"] == slug
    assert observed["draft_path"] == str(draft_path)
    assert observed["mode"] is None
    assert data["project"] == slug
    assert data["success"] is True
    assert data["mode_used"] == "versioned"
    assert data["target_paths"] == ["/tmp/export1", "/tmp/export2"]


def test_delete_run_route(seeded_server: tuple[str, _UnifiedService, str], tmp_path: Path) -> None:
    base_url, svc, slug = seeded_server
    seeded = _seed_succeeded_run(
        svc=svc,
        tmp_path=tmp_path,
        project_slug=slug,
        run_id="run-delete-1",
    )

    status, data = _delete(base_url, f"/api/projects/{slug}/runs/{seeded.run_id}")
    assert status == 200
    assert isinstance(data, dict)
    assert data["ok"] is True
    assert data["run_id"] == seeded.run_id

    status2, data2 = _get(base_url, f"/api/projects/{slug}/runs/{seeded.run_id}")
    assert status2 == 404
    assert isinstance(data2, dict)
    assert data2.get("ok") is False


def test_get_run_artifact_pdf_supports_inline_and_download(
    seeded_server: tuple[str, _UnifiedService, str],
    tmp_path: Path,
) -> None:
    base_url, svc, slug = seeded_server
    seeded = _seed_succeeded_run(
        svc=svc,
        tmp_path=tmp_path,
        project_slug=slug,
        run_id="run-pdf-1",
    )
    output_dir = Path(seeded.output_dir or "")
    pdf_path = output_dir / "research_tearsheet.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%alpha-lab\n")
    with svc.run_store._lock:  # noqa: SLF001 - tests intentionally seed in-memory store
        svc.run_store._records[seeded.run_id].artifact_paths["research_tearsheet_pdf"] = str(  # noqa: SLF001
            pdf_path
        )

    inline_req = urllib.request.Request(
        f"{base_url}/api/projects/{slug}/runs/{seeded.run_id}/artifact/research_tearsheet_pdf"
    )
    with urllib.request.urlopen(inline_req, timeout=5) as resp:
        assert resp.status == 200
        assert (resp.headers.get("Content-Type") or "").startswith("application/pdf")
        assert (resp.headers.get("Content-Disposition") or "").startswith("inline;")
        assert resp.read(5).startswith(b"%PDF-")

    download_req = urllib.request.Request(
        f"{base_url}/api/projects/{slug}/runs/{seeded.run_id}/artifact/research_tearsheet_pdf?download=1"
    )
    with urllib.request.urlopen(download_req, timeout=5) as resp:
        assert resp.status == 200
        assert (resp.headers.get("Content-Type") or "").startswith("application/pdf")
        assert (resp.headers.get("Content-Disposition") or "").startswith("attachment;")


def test_get_run_overview_route_uses_quantile_fallback(
    seeded_server: tuple[str, _UnifiedService, str],
    tmp_path: Path,
) -> None:
    base_url, svc, slug = seeded_server
    seeded = _seed_succeeded_run(
        svc=svc,
        tmp_path=tmp_path,
        project_slug=slug,
        run_id="run-overview-1",
    )

    status, data = _get(base_url, f"/api/projects/{slug}/runs/{seeded.run_id}/overview")
    assert status == 200
    assert isinstance(data, dict)
    assert data["ok"] is True
    assert data["project_slug"] == slug
    assert data["run_id"] == seeded.run_id
    assert data["summary"]["factor_name"] == "mom_5d"

    snapshot = data["snapshot"]
    assert isinstance(snapshot["backtest"], dict)
    assert isinstance(snapshot["icRows"], list)
    assert isinstance(snapshot["rollingRows"], list)
    assert isinstance(snapshot["decayRows"], list)
    assert isinstance(snapshot["groupRows"], list)
    assert snapshot["groupRows"][0]["group"] == "Q1"
    assert isinstance(snapshot["autocorrRows"], list)
    assert isinstance(snapshot["turnoverRows"], list)


def test_get_run_overview_route_not_found(seeded_server: tuple[str, _UnifiedService, str]) -> None:
    base_url, _, slug = seeded_server
    status, data = _get(base_url, f"/api/projects/{slug}/runs/not-exists/overview")
    assert status == 404
    assert isinstance(data, dict)
    assert data.get("ok") is False


def test_factor_correlation_diagnostics_route(
    seeded_server: tuple[str, _UnifiedService, str],
) -> None:
    base_url, _, slug = seeded_server
    status, data = _get(base_url, f"/api/projects/{slug}/diagnostics/factor-correlation")
    assert status == 200
    assert isinstance(data, dict)
    assert "ok" in data
    assert "labels" in data
    assert "matrix" in data
    assert "redundancy_pairs" in data
    assert "dsr_summary" in data
    assert "dsr_by_factor" in data


# ---------------------------------------------------------------------------
# POST /api/projects  (create project)
# ---------------------------------------------------------------------------


def test_create_project_route(live_server: tuple[str, _UnifiedService]) -> None:
    base_url, _ = live_server
    payload = {
        "slug": "http-create-test",
        "title_zh": "HTTP 创建测试",
        "category": "factor_recipe",
        "owner": "test",
        "market": "ashare",
        "frequency": "daily",
        "chatgpt_project_name": "HTTP Create Test",
        "origin_cards": [],
    }
    status, data = _post(base_url, "/api/projects", payload)
    assert status == 201
    assert isinstance(data, dict)
    assert data.get("slug") == "http-create-test"


def test_create_project_missing_fields(live_server: tuple[str, _UnifiedService]) -> None:
    base_url, _ = live_server
    status, data = _post(base_url, "/api/projects", {"slug": "bad"})
    assert status in (400, 422, 500)
    assert isinstance(data, dict)
    assert data.get("ok") is False


# ---------------------------------------------------------------------------
# POST /api/projects/{slug}/refresh
# ---------------------------------------------------------------------------


def test_refresh_project_route(seeded_server: tuple[str, _UnifiedService, str]) -> None:
    base_url, _, slug = seeded_server
    status, data = _post(base_url, f"/api/projects/{slug}/refresh", {})
    assert status == 200
    assert isinstance(data, dict)
    assert data.get("slug") == slug


# ---------------------------------------------------------------------------
# POST /api/projects/{slug}/cases  (create case)
# ---------------------------------------------------------------------------


def test_create_case_route(seeded_server: tuple[str, _UnifiedService, str]) -> None:
    base_url, _, slug = seeded_server
    payload = {
        "case_name": "http_case_new",
        "factor_name": "http_factor",
        "base_method": "momentum",
        "lookback": 10,
        "skip_recent": 0,
        "target_horizon": 5,
    }
    status, data = _post(base_url, f"/api/projects/{slug}/cases", payload)
    assert status == 201
    assert isinstance(data, dict)


def test_create_case_missing_fields(seeded_server: tuple[str, _UnifiedService, str]) -> None:
    base_url, _, slug = seeded_server
    status, data = _post(base_url, f"/api/projects/{slug}/cases", {})
    assert status in (400, 422, 500)
    assert isinstance(data, dict)
    assert data.get("ok") is False


def test_claim_backend_case_route(seeded_server: tuple[str, _UnifiedService, str]) -> None:
    base_url, svc, slug = seeded_server
    specs_root = svc.workspace_root / "configs" / "real_cases" / "single_factor"
    specs_root.mkdir(parents=True, exist_ok=True)
    spec_path = specs_root / "http_claim_case.yaml"
    spec_path.write_text(
        "\n".join(
            [
                "name: http_claim_case",
                "factor_name: http_claim_factor",
                "factor_path: ./factor.csv",
                "prices_path: ./prices.csv",
                "rebalance_frequency: W",
                "n_quantiles: 5",
                "direction: long",
                "universe: {name: demo, path: ./universe.csv, in_universe_column: in_universe}",
                "target: {kind: forward_return, horizon: 5}",
                "transaction_cost: {one_way_rate: 0.001}",
                "output: {root_dir: ./outputs}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    status, data = _post(
        base_url,
        f"/api/projects/{slug}/cases/claim",
        {"spec_path": str(spec_path)},
    )
    assert status == 200
    assert data["status"] == "claimed"

    status, listed = _get(base_url, f"/api/projects/{slug}/cases")
    assert status == 200
    row = next(item for item in listed["cases"] if item["case_name"] == "http_claim_case")
    assert row["claim_status"] == "claimed_by_current_project"
    assert row["requires_explicit_selection"] is False
    assert row["is_recommended"] is True


# ---------------------------------------------------------------------------
# POST /api/vault/preflight
# ---------------------------------------------------------------------------


def test_preflight_route(live_server: tuple[str, _UnifiedService]) -> None:
    base_url, _ = live_server
    payload = {
        "candidate_name": "test_factor",
        "candidate_family": "momentum",
        "candidate_mechanism": "behavioral",
        "candidate_similar": [],
        "candidate_uses_data": [],
    }
    status, data = _post(base_url, "/api/vault/preflight", payload)
    # 200 when graph exists, 400 when vault has no graph.json
    assert status in (200, 400)
    assert isinstance(data, dict)


# ---------------------------------------------------------------------------
# PATCH /api/projects/{slug}  (update project status)
# ---------------------------------------------------------------------------


def test_update_project_status_route(seeded_server: tuple[str, _UnifiedService, str]) -> None:
    base_url, _, slug = seeded_server
    status, data = _patch(
        base_url,
        f"/api/projects/{slug}",
        {"lifecycle": "paused", "current_focus": "暂停中"},
    )
    assert status == 200
    assert isinstance(data, dict)


# ---------------------------------------------------------------------------
# POST unknown route returns 404
# ---------------------------------------------------------------------------


def test_post_unknown_route_returns_404(live_server: tuple[str, _UnifiedService]) -> None:
    base_url, _ = live_server
    status, data = _post(base_url, "/api/not/a/real/post/route", {})
    assert status == 404
    assert isinstance(data, dict)
    assert data.get("ok") is False


# ---------------------------------------------------------------------------
# PATCH unknown route returns 404
# ---------------------------------------------------------------------------


def test_patch_unknown_route_returns_404(live_server: tuple[str, _UnifiedService]) -> None:
    base_url, _ = live_server
    status, data = _patch(base_url, "/api/not/a/real/patch/route", {})
    assert status == 404
    assert isinstance(data, dict)
    assert data.get("ok") is False


# ---------------------------------------------------------------------------
# Custom Factor Workshop HTTP tests
# ---------------------------------------------------------------------------

_VALID_FACTOR_CODE = """
def builder(prices, *, window=20, **kwargs):
    import pandas as pd
    frame = prices.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values(["asset", "date"]).reset_index(drop=True)
    ret = frame.groupby("asset", sort=False)["close"].pct_change(fill_method=None)
    result = frame[["date", "asset"]].copy()
    result["factor"] = "http_test"
    result["value"] = -ret.rolling(window).std()
    return result
""".strip()


def test_list_custom_factors_route(live_server: tuple[str, _UnifiedService]) -> None:
    base_url, _ = live_server
    status, data = _get(base_url, "/api/custom-factors")
    assert status == 200
    assert "factors" in data
    names = [f["name"] for f in data["factors"]]
    assert "momentum" in names
    baseline_names = [f["name"] for f in data["baseline_factor_suite"]]
    assert "mom_20d" in baseline_names
    assert "rev_5d" in baseline_names


def test_register_custom_factor_route(live_server: tuple[str, _UnifiedService]) -> None:
    base_url, _ = live_server
    status, data = _post(
        base_url,
        "/api/custom-factors",
        {
            "name": "http_test_factor",
            "code": _VALID_FACTOR_CODE,
            "description": "HTTP test factor",
        },
    )
    assert status == 201
    assert data["registered"] is True

    # Verify it shows in the list
    status2, data2 = _get(base_url, "/api/custom-factors")
    names = [f["name"] for f in data2["factors"]]
    assert "http_test_factor" in names

    # Clean up
    from alpha_lab.factor_recipe import factor_registry

    factor_registry._builders.pop("http_test_factor", None)


def test_register_custom_factor_invalid_returns_400(
    live_server: tuple[str, _UnifiedService],
) -> None:
    base_url, _ = live_server
    status, data = _post(
        base_url,
        "/api/custom-factors",
        {
            "name": "bad_factor",
            "code": "x = 42",
        },
    )
    assert status == 400
    assert "error" in data


def test_delete_custom_factor_route(live_server: tuple[str, _UnifiedService]) -> None:
    base_url, _ = live_server
    # Register first
    _post(
        base_url,
        "/api/custom-factors",
        {
            "name": "del_test",
            "code": _VALID_FACTOR_CODE,
        },
    )
    # Delete
    status, data = _delete(base_url, "/api/custom-factors/del_test")
    assert status == 200
    assert data["deleted"] is True


def test_delete_builtin_factor_returns_400(live_server: tuple[str, _UnifiedService]) -> None:
    base_url, _ = live_server
    status, data = _delete(base_url, "/api/custom-factors/momentum")
    assert status == 400
    assert "error" in data


def test_get_custom_factor_code_route(live_server: tuple[str, _UnifiedService]) -> None:
    base_url, _ = live_server
    _post(
        base_url,
        "/api/custom-factors",
        {
            "name": "view_http_test",
            "code": _VALID_FACTOR_CODE,
            "description": "view test",
        },
    )
    status, data = _get(base_url, "/api/custom-factors/view_http_test")
    assert status == 200
    assert "def builder" in data["code"]
    assert data["description"] == "view test"

    # Clean up
    from alpha_lab.factor_recipe import factor_registry

    factor_registry._builders.pop("view_http_test", None)
