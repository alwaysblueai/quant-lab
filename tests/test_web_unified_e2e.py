"""Browser-level E2E tests for web_unified.py using Playwright."""

from __future__ import annotations

import socket
import threading
from collections.abc import Generator
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from alpha_lab.web_unified import _UnifiedRequestHandler, _UnifiedService

try:
    from playwright.sync_api import sync_playwright
except ModuleNotFoundError:  # pragma: no cover - optional local browser dependency
    sync_playwright = None  # type: ignore[assignment]


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


def _sync_playwright_or_skip():
    if sync_playwright is None:
        pytest.skip("Playwright is not installed")
    return sync_playwright()


def test_bridge_idea_explorer_end_to_end(live_server: str) -> None:
    with _sync_playwright_or_skip() as p:
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

        # Run exploration with automatic stage/mode selection.
        page.fill("#exploreIdea", "短期反转 overreaction 均值回归")
        page.click("#btnExploreIdea")

        # Assertions on visible UI outputs.
        page.wait_for_selector("#exploreResults", state="visible", timeout=10000)
        page.wait_for_selector("#exploreCardList >> text=Short-term Reversal", timeout=10000)
        page.wait_for_function(
            "document.getElementById('explorePromptBox').textContent.trim().length > 0"
        )
        assert (page.locator("#explorePromptBox").text_content() or "").strip() != ""

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


def test_model_lab_overview_fixture_screenshot_smoke(
    live_server: str,
    tmp_path: Path,
) -> None:
    with _sync_playwright_or_skip() as p:
        browser = _launch_chromium_or_skip(p)
        page = browser.new_page(viewport={"width": 1440, "height": 1100})
        try:
            page.goto(
                f"{live_server}/dev/model-lab/overview-fixture?case=strong_skipped_extreme_nav",
                wait_until="domcontentloaded",
            )
            page.wait_for_selector(".overview-executive-verdict", timeout=10000)
            page.wait_for_selector(".overview-grouped-metrics", timeout=10000)
            page.wait_for_selector(".coverage-break-strip", timeout=10000)
            verdict_reason = page.locator(
                ".overview-verdict-reason"
            ).text_content() or ""
            assert "Strong signal metrics" in verdict_reason
            assert page.get_by_text("Extreme NAV growth detected").first.is_visible()
            assert page.get_by_text(
                "Default 10 bps estimated cost adjustment is available"
            ).first.is_visible()

            full_page = tmp_path / "model_lab_overview_fixture_full.png"
            page.screenshot(path=str(full_page), full_page=True)
            assert full_page.stat().st_size > 5000

            coverage_strip = page.locator(".coverage-break-strip").first
            coverage_strip.scroll_into_view_if_needed()
            coverage_path = tmp_path / "model_lab_overview_fixture_coverage.png"
            coverage_strip.screenshot(path=str(coverage_path))
            assert coverage_path.stat().st_size > 1000

            page.click("text=Optional Diagnostics / Missing Diagnostics")
            page.get_by_text("Required artifact").first.wait_for(timeout=5000)
            missing_path = tmp_path / "model_lab_overview_fixture_missing_diagnostics.png"
            page.locator(".missing-diagnostics-panel").first.screenshot(path=str(missing_path))
            assert missing_path.stat().st_size > 1000
        finally:
            browser.close()


def test_alpha_lab_overview_fixture_modes_screenshot_smoke(
    live_server: str,
    tmp_path: Path,
) -> None:
    cases = [
        ("quick_screening", "快速筛选模式", "Factor Snapshot", "Quick Decision Metrics"),
        ("full_evaluation", "全面评价模式", "Executive Verdict", "Core Decision Charts"),
    ]
    with _sync_playwright_or_skip() as p:
        browser = _launch_chromium_or_skip(p)
        page = browser.new_page(viewport={"width": 1440, "height": 1100})
        try:
            for fixture_id, mode_text, primary_text, secondary_text in cases:
                page.goto(
                    f"{live_server}/dev/alpha-lab/overview-fixture?case={fixture_id}",
                    wait_until="domcontentloaded",
                )
                page.wait_for_selector(".artifact-overview-shell", timeout=10000)
                page.get_by_text(mode_text).first.wait_for(timeout=10000)
                page.get_by_text(primary_text).first.wait_for(timeout=10000)
                page.get_by_text(secondary_text).first.wait_for(timeout=10000)
                path = tmp_path / f"alpha_lab_overview_fixture_{fixture_id}.png"
                page.screenshot(path=str(path), full_page=True)
                assert path.stat().st_size > 4000
        finally:
            browser.close()


def test_model_lab_artifact_fixture_screenshot_smoke(
    live_server: str,
    tmp_path: Path,
) -> None:
    artifacts = [
        ("metrics", "case_name"),
        ("training_log", "Training Health Summary"),
        ("feature_importance", "Feature Importance Summary"),
        ("model_definition_json", "feature_columns"),
        ("run_manifest", "run_timestamp_utc"),
        ("summary", "实验摘要"),
    ]
    with _sync_playwright_or_skip() as p:
        browser = _launch_chromium_or_skip(p)
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        try:
            for artifact, expected_text in artifacts:
                page.goto(
                    f"{live_server}/dev/model-lab/artifact-fixture"
                    f"?case=strong_skipped_extreme_nav&artifact={artifact}",
                    wait_until="domcontentloaded",
                )
                page.wait_for_selector("#viewer", timeout=10000)
                page.get_by_text(expected_text).first.wait_for(timeout=10000)
                path = tmp_path / f"model_lab_artifact_fixture_{artifact}.png"
                page.screenshot(path=str(path), full_page=True)
                assert path.stat().st_size > 3000
        finally:
            browser.close()


def test_model_lab_diagnostics_fixture_screenshot_smoke(
    live_server: str,
    tmp_path: Path,
) -> None:
    with _sync_playwright_or_skip() as p:
        browser = _launch_chromium_or_skip(p)
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        try:
            page.goto(
                f"{live_server}/dev/model-lab/diagnostics-fixture?case=strong_skipped_extreme_nav",
                wait_until="domcontentloaded",
            )
            page.wait_for_selector("#diagContent", timeout=10000)
            page.get_by_text("Training Health").first.wait_for(timeout=10000)
            path = tmp_path / "model_lab_diagnostics_fixture.png"
            page.screenshot(path=str(path), full_page=True)
            assert path.stat().st_size > 3000
        finally:
            browser.close()
