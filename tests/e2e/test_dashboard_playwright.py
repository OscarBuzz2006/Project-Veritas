"""Playwright end-to-end functional and responsive layout tests for docs/ dashboard."""

from __future__ import annotations

import functools
import http.server
import os
import socketserver
import threading
from collections.abc import Generator
from pathlib import Path

import pytest
from playwright.sync_api import ConsoleMessage, sync_playwright

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = REPO_ROOT / "docs"
pytestmark = pytest.mark.e2e


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        return


@pytest.fixture(scope="module")
def docs_server_url() -> Generator[str, None, None]:
    handler = functools.partial(_QuietHandler, directory=str(DOCS_DIR))
    with socketserver.TCPServer(("127.0.0.1", 0), handler) as httpd:
        port = httpd.server_address[1]
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{port}/index.html"
        finally:
            httpd.shutdown()


def test_dashboard_interactive_controls_and_responsive_viewports(
    docs_server_url: str, tmp_path: Path
) -> None:
    if not (DOCS_DIR / "index.html").exists():
        pytest.skip("docs/index.html not generated yet")

    save_repo_shots = os.environ.get("SAVE_DASHBOARD_SCREENSHOTS") == "1"
    shot_dir = DOCS_DIR / "screenshots" if save_repo_shots else tmp_path / "screenshots"
    shot_dir.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)

        # Desktop 1440x900
        desktop_ctx = browser.new_context(viewport={"width": 1440, "height": 900})
        page = desktop_ctx.new_page()
        console_errors: list[str] = []
        page_errors: list[str] = []

        def _on_console(msg: ConsoleMessage) -> None:
            if msg.type == "error":
                console_errors.append(msg.text)

        page.on("console", _on_console)
        page.on("pageerror", lambda err: page_errors.append(str(err)))

        page.goto(docs_server_url, wait_until="networkidle")
        body_text = page.locator("body").inner_text()
        assert "Epistemic Prism" in body_text
        assert "NaN" not in body_text
        assert "undefined" not in body_text

        for section in (
            "masthead-banner",
            "headline-findings",
            "epistemic-quadrant",
            "decoding-dissociation",
            "round-trajectory-explorer",
            "condition-score-table",
        ):
            assert page.locator(f'[data-section="{section}"]').is_visible()

        # Check horizontal overflow at 1440px and ensure desktop table inspect button is unclipped
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
        box = page.locator('[data-inspect-condition="cross_national_k4"]').bounding_box()
        assert box is not None
        assert box["x"] + box["width"] <= 1440

        # Save desktop screenshot before mutating tab/drawer state
        page.screenshot(path=str(shot_dir / "desktop-1440.png"), full_page=True)

        # Filter interaction
        assert page.locator("#score-table-body tr").count() == 11
        page.locator('[data-filter-group="nationality"]').click()
        assert page.locator("#score-table-body tr").count() == 4
        page.locator('[data-filter-group="all"]').click()
        assert page.locator("#score-table-body tr").count() == 11

        # Track tab switcher
        page.locator("#tab-numeric").click()
        assert page.locator("#numeric-core-view").is_visible()
        assert "C2 vs C1" in page.locator("#numeric-contrasts-host").inner_text()
        page.locator("#tab-track4").click()
        assert page.locator("#track4-dissoc-view").is_visible()

        # Round scrubber interaction
        initial_round = page.locator("#round-label").inner_text()
        assert initial_round == "R5"
        page.locator("#round-prev-btn").click()
        assert page.locator("#round-label").inner_text() == "R4"
        page.locator("#traj-metric-select").select_option("goldman_effective_sources")
        assert page.locator("#trajectory-svg-host svg").is_visible()

        # Trace drawer inspect and close interaction
        bubble = page.locator('[data-condition-bubble="cross_national_k4"]')
        assert bubble.get_attribute("tabindex") == "0"
        assert bubble.get_attribute("role") == "button"
        page.locator('#score-table th[data-sort="ground_truth_accuracy"]').click()
        assert (
            page.locator('#score-table th[data-sort="ground_truth_accuracy"]').get_attribute(
                "aria-sort"
            )
            == "descending"
        )
        page.locator('[data-inspect-condition="cross_national_k4"]').click()
        assert page.locator("#trace-drawer").is_visible()
        assert "Assistant:" in page.locator("#drawer-content").inner_text()
        page.locator("#drawer-close-btn").click()
        assert not page.locator("#trace-drawer").is_visible()

        assert console_errors == []
        assert page_errors == []
        desktop_ctx.close()

        # Mobile 390x844
        mobile_ctx = browser.new_context(viewport={"width": 390, "height": 844})
        m_page = mobile_ctx.new_page()
        m_errors: list[str] = []
        m_page.on("pageerror", lambda err: m_errors.append(str(err)))
        m_page.goto(docs_server_url, wait_until="networkidle")
        assert m_page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
        m_page.screenshot(path=str(shot_dir / "mobile-390.png"), full_page=True)
        assert m_errors == []
        mobile_ctx.close()
        browser.close()
