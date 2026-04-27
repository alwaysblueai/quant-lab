"""HTTP smoke tests for the fast-screen surface exposed by web_unified."""

from __future__ import annotations

import json
import socket
import threading
import urllib.error
import urllib.request
from collections.abc import Generator
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from alpha_lab.factors.momentum import momentum
from alpha_lab.fast_screen import Tier1Inputs, run_tier1, save_tier1_result
from alpha_lab.web_unified import _UnifiedRequestHandler, _UnifiedService


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _prices() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    dates = pd.date_range("2024-01-02", periods=60, freq="B")
    rows = []
    for a in [f"A{i}" for i in range(6)]:
        price = 100.0
        for d in dates:
            price *= 1.0 + rng.normal(0, 0.01)
            rows.append({"date": d, "asset": a, "close": price})
    return pd.DataFrame(rows)


@pytest.fixture()
def live_server(tmp_path: Path) -> Generator[tuple[str, _UnifiedService], None, None]:
    vault = tmp_path / "vault"
    vault.mkdir()
    svc = _UnifiedService(vault_root=vault, workspace_root=tmp_path)
    if not hasattr(svc, "fast_screen_artifact_root"):
        pytest.skip("fast-screen HTTP surface has moved out of web_unified")

    # Seed one fast-screen run on disk.
    prices = _prices()
    factor_df = momentum(prices, window=5)
    inputs = Tier1Inputs(
        factor_name="momentum",
        factor_df=factor_df,
        prices=prices,
        horizon=1,
        n_quantiles=5,
        cost_rate=0.0005,
        universe="synth",
        frequency="daily",
    )
    result = run_tier1(inputs, run_id="run_abc")
    save_tier1_result(svc.fast_screen_artifact_root, result)

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


def _get(base_url: str, path: str) -> tuple[int, dict | str]:
    try:
        with urllib.request.urlopen(f"{base_url}{path}", timeout=5) as resp:
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


def test_fast_screen_page_is_html(live_server):
    base, _ = live_server
    status, body = _get(base, "/fast-screen")
    assert status == 200
    assert isinstance(body, str)
    assert "Fast Screen" in body


def test_list_runs_returns_seeded_run(live_server):
    base, _ = live_server
    status, data = _get(base, "/api/fast-screen/runs")
    assert status == 200
    assert isinstance(data, dict)
    runs = data["runs"]
    assert len(runs) == 1
    assert runs[0]["factor_name"] == "momentum"
    assert runs[0]["run_id"] == "run_abc"
    assert runs[0]["verdict"] in {"pass", "warn", "fail"}


def test_get_run_returns_tier1_bundle(live_server):
    base, _ = live_server
    status, data = _get(base, "/api/fast-screen/runs/momentum/run_abc")
    assert status == 200
    assert "tier1" in data and "tier2_index" in data and "available_modules" in data
    t1 = data["tier1"]
    assert len(t1["metrics"]) == 10
    assert len(t1["charts"]) == 4
    assert t1["verdict"]["status"] in {"pass", "warn", "fail"}


def test_get_unknown_run_returns_404(live_server):
    base, _ = live_server
    status, data = _get(base, "/api/fast-screen/runs/momentum/does_not_exist")
    assert status == 404
    assert data["ok"] is False
