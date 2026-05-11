"""HTML/JS template and dev-fixture asset paths for the unified web frontend.

Extracted from the legacy single-file ``web_unified.py`` so the templating
concern lives next to its data and so the ``_UnifiedRequestHandler`` and
``_UnifiedService`` modules can stay focused on routing/business logic.

The asset files (`web_unified_index.html`, `web_model_lab.html`,
`web_unified_md_render.js`, `dev_fixtures/`) still live at the
``src/alpha_lab/`` package root for now; only the helpers that load them
have moved here.
"""

from __future__ import annotations

from pathlib import Path

# ``_templates.py`` is at ``src/alpha_lab/web_unified/_templates.py``;
# ``parent.parent`` walks up to ``src/alpha_lab/`` where the HTML/JS assets live.
_ALPHA_LAB_PKG_DIR = Path(__file__).resolve().parent.parent

_MD_RENDER_JS_PATH = _ALPHA_LAB_PKG_DIR / "web_unified_md_render.js"
_INDEX_HTML_TEMPLATE_PATH = _ALPHA_LAB_PKG_DIR / "web_unified_index.html"
_MODEL_LAB_HTML_TEMPLATE_PATH = _ALPHA_LAB_PKG_DIR / "web_model_lab.html"

_ALPHA_LAB_OVERVIEW_FIXTURE_DIR = _ALPHA_LAB_PKG_DIR / "dev_fixtures" / "alpha_lab_overview"
_MODEL_LAB_OVERVIEW_FIXTURE_DIR = _ALPHA_LAB_PKG_DIR / "dev_fixtures" / "model_lab_overview"

_MD_RENDER_JS: str | None = None
_INDEX_HTML_TEMPLATE: str | None = None
_MODEL_LAB_HTML_TEMPLATE: str | None = None


def _md_render_js() -> str:
    """Load mdRender JS function from file with in-memory caching."""

    global _MD_RENDER_JS
    if _MD_RENDER_JS is None:
        _MD_RENDER_JS = _MD_RENDER_JS_PATH.read_text(encoding="utf-8")
    return _MD_RENDER_JS


def _load_index_html_template() -> str:
    """Load frontend HTML template with in-memory caching."""

    global _INDEX_HTML_TEMPLATE
    if _INDEX_HTML_TEMPLATE is None:
        _INDEX_HTML_TEMPLATE = _INDEX_HTML_TEMPLATE_PATH.read_text(encoding="utf-8")
    return _INDEX_HTML_TEMPLATE


def _index_html_raw() -> str:
    return _load_index_html_template()


def _index_html(*, reload_template: bool = False) -> str:
    template = (
        _INDEX_HTML_TEMPLATE_PATH.read_text(encoding="utf-8")
        if reload_template
        else _index_html_raw()
    )
    return template.replace("@@MD_RENDER_JS@@", _md_render_js())


def _load_model_lab_html_template() -> str:
    global _MODEL_LAB_HTML_TEMPLATE
    if _MODEL_LAB_HTML_TEMPLATE is None:
        _MODEL_LAB_HTML_TEMPLATE = _MODEL_LAB_HTML_TEMPLATE_PATH.read_text(encoding="utf-8")
    return _MODEL_LAB_HTML_TEMPLATE


def _model_lab_html(*, reload_template: bool = False) -> str:
    template = (
        _MODEL_LAB_HTML_TEMPLATE_PATH.read_text(encoding="utf-8")
        if reload_template
        else _load_model_lab_html_template()
    )
    return template.replace("@@MD_RENDER_JS@@", _md_render_js())


__all__ = [
    "_ALPHA_LAB_OVERVIEW_FIXTURE_DIR",
    "_ALPHA_LAB_PKG_DIR",
    "_INDEX_HTML_TEMPLATE_PATH",
    "_MODEL_LAB_HTML_TEMPLATE_PATH",
    "_MODEL_LAB_OVERVIEW_FIXTURE_DIR",
    "_MD_RENDER_JS_PATH",
    "_index_html",
    "_index_html_raw",
    "_load_index_html_template",
    "_load_model_lab_html_template",
    "_md_render_js",
    "_model_lab_html",
]
