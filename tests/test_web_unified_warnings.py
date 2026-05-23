"""OPT-P1-1a: ``WebUnifiedConfigLoadWarning`` observability.

The web frontend keeps its "skip a single broken artifact, keep going" semantics
so a corrupt case spec / project config / run manifest cannot blank out the
whole UI. The warning surfaces the skip without changing the documented
fallback behavior.

These tests pin:

* the helper ``_warn_web_config_load`` emits exactly one
  ``WebUnifiedConfigLoadWarning`` per call;
* ``_resolve_run_factor_label`` warns when the referenced case spec yaml is
  unparseable but still returns the documented fallback (``run.case_name``);
* ``_UnifiedService.list_projects`` warns when a project yaml is corrupt and
  skips it rather than raising;
* ``_UnifiedService._restore_completed_web_runs`` warns when a run manifest
  cannot be reassembled into a record.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from alpha_lab.web_unified._service import _UnifiedService
from alpha_lab.web_unified._utils import (
    WebUnifiedConfigLoadWarning,
    _warn_web_config_load,
)

# ---------------------------------------------------------------------------
# Helper smoke
# ---------------------------------------------------------------------------


def test_warn_web_config_load_emits_warning_class() -> None:
    with pytest.warns(WebUnifiedConfigLoadWarning) as records:
        _warn_web_config_load(
            source=Path("/tmp/example.yaml"),
            action="probe",
            exc=ValueError("invalid payload"),
        )
    assert len(records) == 1
    message = str(records[0].message)
    assert "probe skipped for" in message
    assert "ValueError: invalid payload" in message


# ---------------------------------------------------------------------------
# _resolve_run_factor_label warns on unparseable spec yaml
# ---------------------------------------------------------------------------


def test_resolve_run_factor_label_warns_on_corrupt_spec_yaml(
    tmp_path: Path,
) -> None:
    """When the spec yaml referenced by a run record is unreadable, the
    helper must warn and fall back to ``run.case_name``."""
    from alpha_lab.web_unified import _resolve_run_factor_label
    from alpha_lab.web_unified._run_store import _RunRecord

    spec_path = tmp_path / "broken_spec.yaml"
    spec_path.write_text("not: valid: yaml: :::", encoding="utf-8")

    record = _RunRecord(
        run_id="run_id_for_warning_test",
        project_slug="project_slug_for_warning_test",
        case_name="case_name_for_warning_test",
        round_id=None,
        spec_path=str(spec_path),
        submitted_at_utc="2026-05-22T00:00:00Z",
        evaluation_profile="default_research",
        output_root_dir=None,
        render_report=False,
        status="succeeded",
    )
    # summary is empty -> falls through to load_yaml_document(spec_path) ->
    # the parser raises and the warning fires.
    record.summary = {}

    with pytest.warns(WebUnifiedConfigLoadWarning, match=str(spec_path)):
        resolved = _resolve_run_factor_label(record)

    # Documented fallback: when the spec cannot be loaded, return case_name.
    assert resolved == "case_name_for_warning_test"


# ---------------------------------------------------------------------------
# list_projects warns when a project yaml is corrupt and skips it
# ---------------------------------------------------------------------------


def test_list_projects_warns_and_skips_corrupt_project_yaml(
    tmp_path: Path,
) -> None:
    from alpha_lab.research_bridge.service import PROJECTS_DIRNAME

    vault = tmp_path / "vault"
    projects_root = vault / PROJECTS_DIRNAME
    broken_project = projects_root / "broken_project"
    broken_project.mkdir(parents=True)
    (broken_project / "project.yaml").write_text(
        "this: is: : invalid :::", encoding="utf-8"
    )

    service = _UnifiedService(vault_root=vault, workspace_root=tmp_path)

    with pytest.warns(WebUnifiedConfigLoadWarning, match="project.yaml"):
        rows = service.list_projects()

    # The corrupt project must be skipped rather than crashing the route.
    assert all(row.get("slug") != "broken_project" for row in rows)


# ---------------------------------------------------------------------------
# _restore_completed_web_runs warns when a run manifest cannot be reassembled
# ---------------------------------------------------------------------------


def test_restore_completed_web_runs_warns_on_corrupt_manifest(
    tmp_path: Path,
) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    # Build a malformed run_manifest under the standard web-runs layout.
    web_runs_root = tmp_path / "outputs" / "real_cases" / "_web_runs"
    bad_run_dir = web_runs_root / "case_for_warning_test" / "run_for_warning_test"
    bad_run_dir.mkdir(parents=True)
    bad_manifest = bad_run_dir / "run_manifest.json"
    bad_manifest.write_text("not json", encoding="utf-8")

    # The constructor calls _restore_completed_web_runs at the end, but it
    # swallows the warning quietly during init. Re-trigger by calling the
    # private method directly under pytest.warns so we can assert it.
    service = _UnifiedService(vault_root=vault, workspace_root=tmp_path)

    with pytest.warns(WebUnifiedConfigLoadWarning, match="run_manifest.json"):
        service._restore_completed_web_runs()

    # The restore loop must NOT raise; the bad manifest is simply skipped.
    # Confirm by re-reading json (sanity) so the file state is unchanged.
    assert bad_manifest.read_text(encoding="utf-8") == "not json"
