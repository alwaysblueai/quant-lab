"""Tests for spec-level evaluation_profile parsing and CLI fallback resolution.

The spec stays profile-agnostic (campaign comparison runs one spec under several
profiles); the spec's optional ``evaluation_profile`` is only a default that a
bare CLI run falls back to when ``--evaluation-profile`` is not supplied.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

import alpha_lab.real_cases.single_factor.cli as cli
from alpha_lab.exceptions import AlphaLabConfigError
from alpha_lab.real_cases.single_factor.spec import (
    load_single_factor_case_spec,
    single_factor_case_spec_from_mapping,
)
from tests.single_factor_case_helpers import write_demo_single_factor_case


def _spec_payload(spec_path: Path) -> dict:
    return yaml.safe_load(spec_path.read_text(encoding="utf-8"))


def _set_profile(spec_path: Path, profile: str | None) -> None:
    payload = _spec_payload(spec_path)
    if profile is None:
        payload.pop("evaluation_profile", None)
    else:
        payload["evaluation_profile"] = profile
    spec_path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def test_spec_parses_optional_evaluation_profile(tmp_path: Path) -> None:
    spec_path = write_demo_single_factor_case(tmp_path, factor_name="bp")

    # Absent -> None (spec stays profile-agnostic).
    assert load_single_factor_case_spec(spec_path).evaluation_profile is None

    _set_profile(spec_path, "exploratory_screening")
    assert load_single_factor_case_spec(spec_path).evaluation_profile == "exploratory_screening"


def test_spec_rejects_unknown_evaluation_profile(tmp_path: Path) -> None:
    payload = _spec_payload(write_demo_single_factor_case(tmp_path, factor_name="bp"))
    payload["evaluation_profile"] = "not_a_profile"
    with pytest.raises(AlphaLabConfigError, match="evaluation_profile must be one of"):
        single_factor_case_spec_from_mapping(payload)


def _run_cli_profile(spec_path: Path, out_dir: Path, flag: str | None) -> str:
    name = _spec_payload(spec_path)["name"]
    argv = ["run", str(spec_path), "--output-root-dir", str(out_dir), "--vault-export-mode", "skip"]
    if flag is not None:
        argv += ["--evaluation-profile", flag]
    rc = cli.main(argv)
    assert rc == 0
    metrics = json.loads((out_dir / name / "metrics.json").read_text(encoding="utf-8"))["metrics"]
    return str(metrics["research_evaluation_profile"])


def test_cli_uses_spec_profile_when_flag_absent(tmp_path: Path) -> None:
    spec_path = write_demo_single_factor_case(tmp_path, factor_name="bp")
    _set_profile(spec_path, "exploratory_screening")
    assert _run_cli_profile(spec_path, tmp_path / "a", flag=None) == "exploratory_screening"


def test_cli_flag_overrides_spec_profile(tmp_path: Path) -> None:
    spec_path = write_demo_single_factor_case(tmp_path, factor_name="bp")
    _set_profile(spec_path, "exploratory_screening")
    assert (
        _run_cli_profile(spec_path, tmp_path / "b", flag="default_research") == "default_research"
    )


def test_cli_defaults_when_neither_flag_nor_spec(tmp_path: Path) -> None:
    spec_path = write_demo_single_factor_case(tmp_path, factor_name="bp")
    _set_profile(spec_path, None)
    assert _run_cli_profile(spec_path, tmp_path / "c", flag=None) == "default_research"
