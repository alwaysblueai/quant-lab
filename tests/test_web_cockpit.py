from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import alpha_lab.web_cockpit as web_cockpit
from alpha_lab.web_cockpit import (
    _build_frontend_batch_parallel_config,
    _CockpitRunStore,
    _CockpitService,
    _RunTask,
)


def _build_vault(tmp_path: Path) -> Path:
    vault = tmp_path / "quant-knowledge"
    for rel in [
        "10_concepts",
        "20_methods",
        "30_factors",
        "50_experiments",
        "90_moc",
    ]:
        (vault / rel).mkdir(parents=True, exist_ok=True)
    (vault / "30_factors" / "Factor - Momentum Base.md").write_text(
        "# 动量基类\n\n用于测试。\n",
        encoding="utf-8",
    )
    (vault / "90_moc" / "CARD-INDEX.tsv").write_text(
        "path\ttype\tname\tdomain\tlifecycle\ttags\tparent_moc\n"
        "30_factors/Factor - Momentum Base.md\tfactor\tMomentum Base\talpha_research\t"
        "theoretical\tmomentum,factor\tMOC - Factors\n",
        encoding="utf-8",
    )
    return vault


def _wait_run_status(
    service: _CockpitService,
    *,
    project_slug: str,
    run_id: str,
    timeout_s: float = 5.0,
) -> dict[str, Any]:
    start = time.time()
    while time.time() - start < timeout_s:
        run = service.run_store.get(run_id)
        if run is None:
            time.sleep(0.05)
            continue
        payload = run.to_payload()
        if payload["status"] in {"succeeded", "failed"}:
            return payload
        time.sleep(0.05)
    raise AssertionError(f"run {run_id} did not finish in {timeout_s}s for project {project_slug}")


def _wait_store_status(
    store: _CockpitRunStore,
    *,
    run_id: str,
    timeout_s: float = 5.0,
) -> dict[str, Any]:
    start = time.time()
    while time.time() - start < timeout_s:
        run = store.get(run_id)
        if run is None:
            time.sleep(0.05)
            continue
        payload = run.to_payload()
        if payload["status"] in {"succeeded", "failed"}:
            return payload
        time.sleep(0.05)
    raise AssertionError(f"run {run_id} did not finish in {timeout_s}s")


def test_frontend_batch_parallel_config_prefers_process_mode() -> None:
    config = _build_frontend_batch_parallel_config(5)
    assert config.mode == "process"
    assert config.max_workers == 4
    assert config.factors_per_worker == 2


def test_cockpit_service_dashboard_and_card_search(tmp_path: Path) -> None:
    vault = _build_vault(tmp_path)
    service = _CockpitService(vault_root=vault, workspace_root=tmp_path)

    created = service.create_project(
        {
            "slug": "momentum-factor",
            "title_zh": "动量因子项目",
            "category": "factor_family",
            "owner": "yukun",
            "market": "ashare",
            "frequency": "daily",
            "chatgpt_project_name": "Momentum Factor",
            "origin_cards": ["30_factors/Factor - Momentum Base.md"],
        }
    )
    assert created["slug"] == "momentum-factor"

    cards = service.search_cards("momentum", limit=10)
    assert len(cards["cards"]) == 1
    assert cards["cards"][0]["name"] == "Momentum Base"

    dashboard = service.dashboard()
    assert dashboard["project_count"] == 1
    assert dashboard["pending_writebacks"] == 0


def test_cockpit_service_run_summarize_and_apply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vault = _build_vault(tmp_path)
    service = _CockpitService(vault_root=vault, workspace_root=tmp_path)
    service.create_project(
        {
            "slug": "momentum-factor",
            "title_zh": "动量因子项目",
            "category": "factor_family",
            "owner": "yukun",
            "market": "ashare",
            "frequency": "daily",
            "chatgpt_project_name": "Momentum Factor",
            "origin_cards": ["30_factors/Factor - Momentum Base.md"],
        }
    )
    service.create_round(
        "momentum-factor",
        {
            "topic": "三个月成交额加权动量",
            "round_id": "r01",
        },
    )
    service.create_case(
        "momentum-factor",
        {
            "round_id": "r01",
            "case_name": "mom_amt_60",
            "factor_name": "mom_amt_60",
            "base_method": "momentum",
            "lookback": 60,
            "skip_recent": 5,
            "target_horizon": 5,
        },
    )

    def _fake_run_single_factor_case(
        spec_or_path: str | Path,
        *,
        output_root_dir: str | Path | None = None,
        factor_loader: Any = None,
        evaluation_profile: str = "default_research",
        vault_root: str | Path | None = None,
        vault_export_mode: str = "versioned",
    ) -> Any:
        del factor_loader, evaluation_profile, vault_root, vault_export_mode
        spec_path = Path(spec_or_path).resolve()
        root = (
            Path(output_root_dir).resolve()
            if output_root_dir is not None
            else tmp_path / "run_root"
        )
        run_dir = root / "mom_amt_60"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "run_manifest.json").write_text(
            json.dumps({"case_name": "mom_amt_60"}, indent=2),
            encoding="utf-8",
        )
        (run_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "factor_verdict": "promising",
                    "mean_rank_ic": 0.041,
                    "mean_ic": 0.032,
                    "mean_long_short_return": 0.006,
                    "mean_long_short_turnover": 0.19,
                    "promotion_decision": "promote_with_review",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        (run_dir / "summary.md").write_text("# Summary\n\nrun summary\n", encoding="utf-8")
        (run_dir / "experiment_card.md").write_text(
            "# Experiment Card\n\nmachine owned sections\n",
            encoding="utf-8",
        )

        assert spec_path.exists()
        return SimpleNamespace(
            output_dir=run_dir,
            artifact_paths={
                "run_manifest": run_dir / "run_manifest.json",
                "metrics": run_dir / "metrics.json",
                "summary": run_dir / "summary.md",
                "experiment_card": run_dir / "experiment_card.md",
            },
        )

    def _fake_write_case_report(output_dir: Path, *, overwrite: bool = False) -> Path:
        del overwrite
        report = output_dir / "case_report.md"
        report.write_text("# Report\n\nok\n", encoding="utf-8")
        return report

    monkeypatch.setattr(
        "alpha_lab.web_cockpit.run_single_factor_case", _fake_run_single_factor_case
    )
    monkeypatch.setattr("alpha_lab.web_cockpit.write_case_report", _fake_write_case_report)

    run_payload = service.submit_run(
        "momentum-factor",
        {
            "case_name": "mom_amt_60",
            "round_id": "r01",
            "output_root_dir": str(tmp_path / "run_root"),
            "render_report": True,
        },
    )
    finished = _wait_run_status(
        service,
        project_slug="momentum-factor",
        run_id=str(run_payload["run_id"]),
    )
    assert finished["status"] == "succeeded"
    assert "metrics" in finished["artifact_paths"]

    summarized = service.summarize_run(
        "momentum-factor",
        str(run_payload["run_id"]),
        {"round_id": "r01"},
    )
    assert Path(str(summarized["writeback_draft"])).exists()
    drafts = service.list_drafts("momentum-factor")
    assert len(drafts) == 1
    draft_name = str(drafts[0]["name"])

    patched = service.patch_draft(
        "momentum-factor",
        draft_name,
        {
            "review_status": "approved",
            "reviewed_by": "yukun",
            "reviewed_at": "now",
            "one_sentence_verdict": "继续迭代。",
        },
    )
    assert patched["review_status"] == "approved"
    assert patched["reviewed_by"] == "yukun"

    applied = service.apply_draft("momentum-factor", draft_name, {})
    assert applied["success"] is True
    exported_latest = vault / "50_experiments" / "mom_amt_60" / "latest.md"
    assert exported_latest.exists()


def test_cockpit_run_store_isolates_frontend_run_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _CockpitRunStore()
    single_calls: list[str] = []
    single_output_roots: list[Path] = []

    def _fake_single(*args, **kwargs):
        spec_obj = Path(args[0])
        single_calls.append(str(spec_obj))
        single_output_roots.append(Path(str(kwargs["output_root_dir"])))
        run_dir = tmp_path / "cockpit_single_results" / spec_obj.stem
        run_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = run_dir / "metrics.json"
        metrics_path.write_text(
            json.dumps({"metrics": {"factor_verdict": "single", "mean_ic": 0.01}}),
            encoding="utf-8",
        )
        return SimpleNamespace(
            output_dir=run_dir,
            artifact_paths={"metrics": metrics_path},
        )

    monkeypatch.setattr(web_cockpit, "run_single_factor_case", _fake_single)

    spec1 = tmp_path / "demo_1.yaml"
    spec2 = tmp_path / "demo_2.yaml"
    spec1.write_text("name: demo_1\n", encoding="utf-8")
    spec2.write_text("name: demo_2\n", encoding="utf-8")

    store.submit(
        _RunTask(
            run_id="run_1",
            project_slug="demo",
            case_name="demo_1",
            round_id=None,
            spec_path=str(spec1),
            evaluation_profile="exploratory_screening",
            output_root_dir=str(tmp_path / "run_root"),
            render_report=False,
        )
    )
    store.submit(
        _RunTask(
            run_id="run_2",
            project_slug="demo",
            case_name="demo_2",
            round_id=None,
            spec_path=str(spec2),
            evaluation_profile="exploratory_screening",
            output_root_dir=str(tmp_path / "run_root"),
            render_report=False,
        )
    )

    run1 = _wait_store_status(store, run_id="run_1")
    run2 = _wait_store_status(store, run_id="run_2")
    assert run1["status"] == "succeeded"
    assert run2["status"] == "succeeded"
    assert single_calls == [str(spec1), str(spec2)]
    assert single_output_roots == [
        (tmp_path / "run_root" / "_web_runs" / "run_1").resolve(),
        (tmp_path / "run_root" / "_web_runs" / "run_2").resolve(),
    ]
