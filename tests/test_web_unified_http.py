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
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from alpha_lab.web_unified import _UnifiedRequestHandler, _UnifiedService


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


@pytest.fixture()
def live_server(tmp_path: Path) -> Generator[tuple[str, _UnifiedService], None, None]:
    """Start a real ThreadingHTTPServer; yield (base_url, service)."""
    vault = _build_vault(tmp_path)
    svc = _UnifiedService(vault_root=vault, workspace_root=tmp_path)

    class _Handler(_UnifiedRequestHandler):
        pass

    _Handler.svc = svc  # type: ignore[attr-defined]

    port = _free_port()
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
        f"{base_url}{path}", data=data,
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
        f"{base_url}{path}", data=data,
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


def test_read_card_route_found_vault_relative_path(live_server: tuple[str, _UnifiedService]) -> None:
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
# /api/vault/explore-idea
# ---------------------------------------------------------------------------


def test_explore_idea_free_route(live_server: tuple[str, _UnifiedService]) -> None:
    base_url, _ = live_server
    status, data = _post(base_url, "/api/vault/explore-idea", {"idea": "momentum 动量", "mode": "free"})
    assert status == 200
    assert isinstance(data, dict)
    assert data["mode"] == "free"
    assert isinstance(data["related_cards"], list)
    assert isinstance(data["gpt_prompt"], str)
    assert "你的想法" in data["gpt_prompt"]


def test_explore_idea_constrained_route(live_server: tuple[str, _UnifiedService]) -> None:
    base_url, _ = live_server
    status, data = _post(base_url, "/api/vault/explore-idea", {"idea": "动量", "mode": "constrained"})
    assert status == 200
    assert isinstance(data, dict)
    assert data["mode"] == "constrained"
    cr = data["constraint_report"]
    assert isinstance(cr, dict)
    assert "crowding_warning" in cr


def test_explore_idea_route_accepts_project_slug(live_server: tuple[str, _UnifiedService]) -> None:
    base_url, svc = live_server
    svc.create_project(
        {
            "slug": "http-momentum",
            "title_zh": "HTTP 动量项目",
            "category": "factor_recipe",
            "owner": "test",
            "market": "ashare",
            "frequency": "daily",
            "chatgpt_project_name": "HTTP Momentum",
            "origin_cards": ["30_factors/Factor - Momentum Base.md"],
        }
    )

    status, data = _post(
        base_url,
        "/api/vault/explore-idea",
        {"idea": "momentum 动量", "mode": "free", "project_slug": "http-momentum"},
    )
    assert status == 200
    assert isinstance(data, dict)
    assert data["mode"] == "free"


def test_explore_idea_empty_body_returns_error(live_server: tuple[str, _UnifiedService]) -> None:
    base_url, _ = live_server
    status, data = _post(base_url, "/api/vault/explore-idea", {"idea": "", "mode": "free"})
    assert status in (400, 422, 500)
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
    svc.create_case(
        slug,
        {
            "case_name": "mom_5d",
            "factor_name": "mom_5d",
            "base_method": "momentum",
            "lookback": 5,
            "skip_recent": 0,
            "target_horizon": 5,
        },
    )
    return base_url, svc, slug


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


def test_dashboard_route(live_server: tuple[str, _UnifiedService]) -> None:
    base_url, _ = live_server
    status, data = _get(base_url, "/api/dashboard")
    assert status == 200
    assert isinstance(data, dict)
    assert "project_count" in data
    assert "run_status_counts" in data
    assert "vault_card_count" in data


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


def test_register_custom_factor_route(live_server: tuple[str, _UnifiedService]) -> None:
    base_url, _ = live_server
    status, data = _post(base_url, "/api/custom-factors", {
        "name": "http_test_factor",
        "code": _VALID_FACTOR_CODE,
        "description": "HTTP test factor",
    })
    assert status == 201
    assert data["registered"] is True

    # Verify it shows in the list
    status2, data2 = _get(base_url, "/api/custom-factors")
    names = [f["name"] for f in data2["factors"]]
    assert "http_test_factor" in names

    # Clean up
    from alpha_lab.factor_recipe import factor_registry
    factor_registry._builders.pop("http_test_factor", None)


def test_register_custom_factor_invalid_returns_400(live_server: tuple[str, _UnifiedService]) -> None:
    base_url, _ = live_server
    status, data = _post(base_url, "/api/custom-factors", {
        "name": "bad_factor",
        "code": "x = 42",
    })
    assert status == 400
    assert "error" in data


def test_delete_custom_factor_route(live_server: tuple[str, _UnifiedService]) -> None:
    base_url, _ = live_server
    # Register first
    _post(base_url, "/api/custom-factors", {
        "name": "del_test",
        "code": _VALID_FACTOR_CODE,
    })
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
    _post(base_url, "/api/custom-factors", {
        "name": "view_http_test",
        "code": _VALID_FACTOR_CODE,
        "description": "view test",
    })
    status, data = _get(base_url, "/api/custom-factors/view_http_test")
    assert status == 200
    assert "def builder" in data["code"]
    assert data["description"] == "view test"

    # Clean up
    from alpha_lab.factor_recipe import factor_registry
    factor_registry._builders.pop("view_http_test", None)
