"""Browser-level E2E tests for web_unified.py using Playwright."""

from __future__ import annotations

import socket
import threading
from collections.abc import Generator
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from alpha_lab.web_unified import _UnifiedRequestHandler, _UnifiedService

playwright_sync = pytest.importorskip("playwright.sync_api")
sync_playwright = playwright_sync.sync_playwright


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

    # Seed a few reversal cards so constrained mode can produce meaningful report.
    (vault / "90_moc" / "CARD-INDEX.tsv").write_text(
        (
            "path\ttype\tname\tdomain\tlifecycle\ttags\tparent_moc\n"
            "30_factors/Factor - Short-term Reversal A.md\tfactor\tShort-term Reversal A\t"
            "alpha_research\tvalidated\treversal,overreaction,behavioral\tMOC - Factors\n"
            "30_factors/Factor - Short-term Reversal B.md\tfactor\tShort-term Reversal B\t"
            "alpha_research\tvalidated\treversal,overreaction,behavioral\tMOC - Factors\n"
            "30_factors/Factor - Short-term Reversal C.md\tfactor\tShort-term Reversal C\t"
            "alpha_research\tvalidated\treversal,overreaction,behavioral\tMOC - Factors\n"
        ),
        encoding="utf-8",
    )

    for suffix in ("A", "B", "C"):
        (vault / "30_factors" / f"Factor - Short-term Reversal {suffix}.md").write_text(
            "---\n"
            "type: factor\n"
            "factor_family: reversal\n"
            "mechanism: behavioral\n"
            f"summary: 短期反转因子{suffix}\n"
            "---\n"
            f"# Short-term Reversal {suffix}\n\n"
            "用于测试 E2E。\n",
            encoding="utf-8",
        )
    return vault


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture()
def live_server(tmp_path: Path) -> Generator[str, None, None]:
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
        yield f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()


def _launch_chromium_or_skip(playwright_obj: object):
    try:
        return playwright_obj.chromium.launch(headless=True)
    except Exception as exc:  # pragma: no cover - env-dependent fallback
        msg = str(exc)
        if "error while loading shared libraries" in msg or "libasound.so.2" in msg:
            pytest.skip(f"Playwright Chromium missing runtime libs: {msg.splitlines()[0]}")
        pytest.skip(f"Playwright Chromium launch failed: {msg.splitlines()[0]}")


def test_bridge_idea_explorer_end_to_end(live_server: str) -> None:
    with sync_playwright() as p:
        browser = _launch_chromium_or_skip(p)
        page = browser.new_page()
        page.goto(f"{live_server}/", wait_until="domcontentloaded")

        # Enter bridge workspace and create a project.
        page.click('button[data-view="bridge"]')
        page.fill("#createSlug", "e2e-reversal")
        page.fill("#createTitle", "E2E 反转项目")
        page.fill("#createOwner", "e2e")
        page.fill("#createMarket", "ashare")
        page.fill("#createFrequency", "daily")
        page.fill("#createChatgptName", "E2E Reversal")
        page.click("#btnCreateProject")
        page.wait_for_timeout(400)

        # Run constrained exploration.
        page.fill("#exploreIdea", "短期反转 overreaction 均值回归")
        page.check('input[name="exploreMode"][value="constrained"]')
        page.click("#btnExploreIdea")

        # Assertions on visible UI outputs.
        page.wait_for_selector("#exploreResults", state="visible", timeout=10000)
        page.wait_for_selector("#exploreCardList >> text=Short-term Reversal", timeout=10000)
        page.wait_for_function(
            "document.getElementById('explorePromptBox').textContent.trim().length > 0"
        )
        assert (page.locator("#explorePromptBox").text_content() or "").strip() != ""
        assert page.locator("#exploreConstraintBox").is_visible()

        # Click a related card and verify card viewer is populated.
        page.click('#exploreCardList [data-action="selectCard"]')
        page.wait_for_function(
            "document.getElementById('cardViewName').value.includes('Short-term Reversal')"
        )
        page.wait_for_function(
            "document.getElementById('cardContent').textContent.includes('Short-term Reversal')"
        )
        assert "Short-term Reversal" in (page.locator("#cardContent").text_content() or "")

        browser.close()
