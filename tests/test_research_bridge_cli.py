from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from alpha_lab.research_bridge.cli import main


def test_bridge_cli_init_project_routes_to_service(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: dict[str, Any] = {}

    def _fake_init_project(**kwargs: Any):
        captured.update(kwargs)
        return type(
            "_Result",
            (),
            {
                "project": type("_Project", (), {"slug": "momentum-factor"})(),
                "paths": type(
                    "_Paths",
                    (),
                    {
                        "project_dir": tmp_path / "55_projects" / "momentum-factor",
                    },
                )(),
            },
        )()

    monkeypatch.setattr("alpha_lab.research_bridge.cli.init_project", _fake_init_project)

    rc = main(
        [
            "init-project",
            "--slug",
            "momentum-factor",
            "--title-zh",
            "动量因子项目",
            "--category",
            "factor_family",
            "--owner",
            "yukun",
            "--market",
            "ashare",
            "--frequency",
            "daily",
            "--chatgpt-project-name",
            "Momentum Factor",
            "--origin-card",
            "30_factors/Factor - Momentum Base.md",
            "--vault-root",
            str(tmp_path),
        ]
    )

    assert rc == 0
    assert captured["slug"] == "momentum-factor"
    assert captured["origin_cards"] == ["30_factors/Factor - Momentum Base.md"]
    out = capsys.readouterr().out
    assert "bridge-init-project" in out
