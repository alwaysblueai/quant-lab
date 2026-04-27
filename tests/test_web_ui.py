from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import alpha_lab.web_ui as web_ui
from alpha_lab.web_ui import (
    _build_frontend_batch_parallel_config,
    _extract_metrics_summary,
    _extract_visualization_payload,
    _parse_run_task,
    _prepare_spec_for_data_source,
    _rewrite_spec_with_source_inputs,
    _RunTask,
    _WebRunRecord,
    _WebRunStore,
)


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _wait_web_run_status(
    store: _WebRunStore,
    *,
    run_id: str,
    timeout_s: float = 5.0,
) -> dict[str, object]:
    import time

    start = time.time()
    while time.time() - start < timeout_s:
        record = store.get(run_id)
        if record is None:
            time.sleep(0.05)
            continue
        payload = record.to_payload()
        if payload["status"] in {"succeeded", "failed"}:
            return payload
        time.sleep(0.05)
    raise AssertionError(f"run {run_id} did not finish in {timeout_s}s")


def test_extract_visualization_payload_builds_series(tmp_path: Path) -> None:
    ic_path = _write(
        tmp_path / "ic_timeseries.csv",
        "date,ic,rank_ic\n2024-01-01,0.10,0.20\n2024-01-02,0.05,0.10\n",
    )
    turnover_path = _write(
        tmp_path / "turnover.csv",
        "date,factor,turnover\n2024-01-01,demo,0.30\n2024-01-02,demo,0.40\n",
    )
    rolling_path = _write(
        tmp_path / "rolling_stability.csv",
        "date,rolling_mean_ic\n2024-01-01,0.02\n2024-01-02,0.03\n",
    )
    group_returns_path = _write(
        tmp_path / "group_returns.csv",
        "date,factor,group,group_return\n"
        "2024-01-01,demo,1,-0.01\n"
        "2024-01-01,demo,5,0.02\n"
        "2024-01-02,demo,1,-0.02\n"
        "2024-01-02,demo,5,0.03\n",
    )

    payload = _extract_visualization_payload(
        {
            "ic_timeseries": str(ic_path),
            "turnover": str(turnover_path),
            "rolling_stability": str(rolling_path),
            "group_returns": str(group_returns_path),
        }
    )

    series = payload["series"]
    assert isinstance(series, dict)

    ic = series["ic"]
    rank_ic = series["rank_ic"]
    turnover = series["turnover"]
    rolling_ic = series["rolling_mean_ic"]
    long_short = series["long_short"]
    cum_long_short = series["cum_long_short"]
    long_short_drawdown = series["long_short_drawdown"]

    assert len(ic) == 2
    assert len(rank_ic) == 2
    assert len(turnover) == 2
    assert len(rolling_ic) == 2
    assert len(long_short) == 2
    assert len(cum_long_short) == 2
    assert len(long_short_drawdown) == 2

    assert long_short[0]["date"] == "2024-01-01"
    assert abs(float(long_short[0]["value"]) - 0.03) < 1e-12
    assert long_short[1]["date"] == "2024-01-02"
    assert abs(float(long_short[1]["value"]) - 0.05) < 1e-12
    assert abs(float(cum_long_short[1]["value"]) - 0.0815) < 1e-12
    assert payload["group_mean_returns"] != []
    assert payload["ic_histogram"] != []
    assert payload["rank_ic_histogram"] != []


def test_extract_visualization_payload_handles_missing_artifacts() -> None:
    payload = _extract_visualization_payload({})
    series = payload["series"]
    assert isinstance(series, dict)
    assert series["ic"] == []
    assert series["rank_ic"] == []
    assert series["turnover"] == []
    assert series["rolling_mean_ic"] == []
    assert series["long_short"] == []
    assert series["cum_long_short"] == []
    assert series["long_short_drawdown"] == []
    assert payload["group_mean_returns"] == []
    assert payload["ic_histogram"] == []
    assert payload["rank_ic_histogram"] == []


def test_extract_metrics_summary_merges_backtest_fields(tmp_path: Path) -> None:
    metrics_path = _write(
        tmp_path / "metrics.json",
        "{\n"
        '  "metrics": {\n'
        '    "factor_verdict": "Pass",\n'
        '    "campaign_triage": "Keep",\n'
        '    "promotion_decision": "Promote",\n'
        '    "mean_ic": 0.12,\n'
        '    "ic_ir": 0.9,\n'
        '    "factor_verdict_reasons": ["stable"]\n'
        "  }\n"
        "}\n",
    )
    backtest_path = _write(
        tmp_path / "backtest_result.json",
        "{\n"
        '  "summary": {\n'
        '    "annualized_return": 0.15,\n'
        '    "sharpe": 1.1,\n'
        '    "max_drawdown": -0.2\n'
        "  }\n"
        "}\n",
    )
    summary = _extract_metrics_summary(metrics_path, backtest_result_path=backtest_path)
    assert summary["factor_verdict"] == "Pass"
    assert summary["campaign_triage"] == "Keep"
    assert float(summary["mean_ic"]) == 0.12
    assert float(summary["annualized_return"]) == 0.15
    assert float(summary["sharpe"]) == 1.1
    assert float(summary["max_drawdown"]) == -0.2
    interview_brief = summary["interview_brief"]
    assert isinstance(interview_brief, dict)
    assert "opening_30s" in interview_brief
    assert "deep_dive_90s" in interview_brief
    assert interview_brief["highlights"] != []
    decision_analysis = summary["decision_analysis"]
    assert isinstance(decision_analysis, dict)
    assert "workflow" in decision_analysis
    assert "nodes" in decision_analysis


def test_parse_run_task_accepts_tushare_token_and_source_controls() -> None:
    task = _parse_run_task(
        {
            "spec_text": "name: demo\nfactor_name: bp\nfactor_path: x\nprices_path: y\n",
            "spec_filename": "demo.yaml",
            "evaluation_profile": "default_research",
            "data_source": "tushare",
            "tushare_token": "token_x",
            "data_start_date": "2024-01-01",
            "data_end_date": "2024-12-31",
            "data_asset_limit": 100,
            "output_root_dir": "dist/web_ui_runs",
            "render_report": True,
        }
    )
    assert task.data_source == "tushare"
    assert task.tushare_token == "token_x"
    assert task.data_start_date == "2024-01-01"
    assert task.data_end_date == "2024-12-31"
    assert task.data_slice_preset == "standard"
    assert task.data_asset_limit == 100


def test_parse_run_task_requires_tushare_token_when_source_is_tushare() -> None:
    with pytest.raises(ValueError, match="tushare_token is required"):
        _parse_run_task(
            {
                "spec_text": "name: demo\nfactor_name: bp\nfactor_path: x\nprices_path: y\n",
                "spec_filename": "demo.yaml",
                "evaluation_profile": "default_research",
                "data_source": "tushare",
            }
        )


def test_parse_run_task_rejects_unknown_slice_preset() -> None:
    with pytest.raises(ValueError, match="data_slice_preset must be one of"):
        _parse_run_task(
            {
                "spec_text": "name: demo\nfactor_name: bp\nfactor_path: x\nprices_path: y\n",
                "spec_filename": "demo.yaml",
                "evaluation_profile": "default_research",
                "data_source": "baostock",
                "data_slice_preset": "weird",
            }
        )


def test_rewrite_spec_with_source_inputs_updates_factor_prices_universe_paths(
    tmp_path: Path,
) -> None:
    spec_path = _write(
        tmp_path / "case.yaml",
        "name: bp_demo\n"
        "factor_name: bp\n"
        "factor_path: ./old/bp.csv\n"
        "prices_path: ./old/prices.csv\n"
        "rebalance_frequency: M\n"
        "n_quantiles: 5\n"
        "direction: long\n"
        "universe:\n"
        "  name: demo\n"
        "  path: ./old/universe.csv\n"
        "  in_universe_column: in_universe\n"
        "target:\n"
        "  kind: forward_return\n"
        "  horizon: 5\n"
        "preprocess:\n"
        "  winsorize: true\n"
        "  winsorize_lower: 0.01\n"
        "  winsorize_upper: 0.99\n"
        "  standardization: zscore\n"
        "  min_group_size: 5\n"
        "  min_coverage: 0.6\n"
        "neutralization:\n"
        "  enabled: false\n"
        "  exposures_path:\n"
        "  size_col:\n"
        "  industry_col:\n"
        "  min_obs: 20\n"
        "  ridge: 1.0e-8\n"
        "transaction_cost:\n"
        "  one_way_rate: 0.001\n"
        "output:\n"
        "  root_dir: ./out\n",
    )
    source_input_dir = tmp_path / "source_inputs"
    source_input_dir.mkdir(parents=True, exist_ok=True)
    _write(source_input_dir / "bp.csv", "date,asset,factor,value\n2024-01-02,000001.SZ,bp,0.5\n")
    _write(source_input_dir / "prices.csv", "date,asset,close\n2024-01-02,000001.SZ,10\n")
    _write(
        source_input_dir / "universe.csv",
        "date,asset,in_universe\n2024-01-02,000001.SZ,1\n",
    )

    rewritten_path = _rewrite_spec_with_source_inputs(
        original_spec_path=spec_path,
        source_input_dir=source_input_dir,
    )
    text = rewritten_path.read_text(encoding="utf-8")
    assert str(source_input_dir / "bp.csv") in text
    assert str(source_input_dir / "prices.csv") in text
    assert str(source_input_dir / "universe.csv") in text


def test_prepare_spec_for_data_source_reuses_cached_inputs(tmp_path: Path, monkeypatch) -> None:
    spec_path = _write(
        tmp_path / "case.yaml",
        "name: bp_demo\n"
        "factor_name: bp\n"
        "factor_path: ./old/bp.csv\n"
        "prices_path: ./old/prices.csv\n"
        "rebalance_frequency: M\n"
        "n_quantiles: 5\n"
        "direction: long\n"
        "universe:\n"
        "  name: demo\n"
        "  path: ./old/universe.csv\n"
        "  in_universe_column: in_universe\n"
        "target:\n"
        "  kind: forward_return\n"
        "  horizon: 5\n"
        "preprocess:\n"
        "  winsorize: true\n"
        "  winsorize_lower: 0.01\n"
        "  winsorize_upper: 0.99\n"
        "  standardization: zscore\n"
        "  min_group_size: 5\n"
        "  min_coverage: 0.6\n"
        "neutralization:\n"
        "  enabled: false\n"
        "  exposures_path:\n"
        "  size_col:\n"
        "  industry_col:\n"
        "  min_obs: 20\n"
        "  ridge: 1.0e-8\n"
        "transaction_cost:\n"
        "  one_way_rate: 0.001\n"
        "output:\n"
        "  root_dir: ./out\n",
    )

    calls: list[dict[str, object]] = []

    def _fake_generate_baostock_inputs(
        output_dir: str | Path,
        start_date: str,
        end_date: str,
        *,
        assets=None,
        asset_limit: int | None = None,
        include_roe: bool = True,
    ):
        calls.append(
            {
                "output_dir": str(output_dir),
                "start_date": start_date,
                "end_date": end_date,
                "asset_limit": asset_limit,
                "include_roe": include_roe,
            }
        )
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        _write(out / "bp.csv", "date,asset,factor,value\n2024-01-02,000001.SZ,bp,0.5\n")
        _write(out / "prices.csv", "date,asset,close\n2024-01-02,000001.SZ,10\n")
        _write(out / "universe.csv", "date,asset,in_universe\n2024-01-02,000001.SZ,1\n")
        _write(out / "roe_ttm.csv", "date,asset,factor,value\n")

    monkeypatch.setattr(
        web_ui,
        "generate_baostock_real_case_inputs",
        _fake_generate_baostock_inputs,
    )

    task = web_ui._RunTask(
        run_id="run_1",
        spec_text=spec_path.read_text(encoding="utf-8"),
        spec_filename=spec_path.name,
        evaluation_profile="default_research",
        data_source="baostock",
        data_start_date="2024-01-01",
        data_end_date="2024-03-31",
        data_asset_limit=10,
        tushare_token=None,
        render_report=False,
        output_root_dir=str(tmp_path / "runs"),
    )

    cache_root = tmp_path / "source_cache"
    rewritten_1 = _prepare_spec_for_data_source(
        task=task,
        original_spec_path=spec_path,
        cache_root=cache_root,
    )
    rewritten_2 = _prepare_spec_for_data_source(
        task=task,
        original_spec_path=spec_path,
        cache_root=cache_root,
    )

    assert rewritten_1.exists()
    assert rewritten_2.exists()
    assert len(calls) == 1
    assert calls[0]["include_roe"] is False


def test_prepare_spec_for_data_source_uses_tushare_slice_preset_defaults(
    tmp_path: Path,
    monkeypatch,
) -> None:
    spec_path = _write(
        tmp_path / "case_tushare.yaml",
        "name: bp_demo\n"
        "factor_name: bp\n"
        "factor_path: ./old/bp.csv\n"
        "prices_path: ./old/prices.csv\n"
        "rebalance_frequency: M\n"
        "n_quantiles: 5\n"
        "direction: long\n"
        "universe:\n"
        "  name: demo\n"
        "  path: ./old/universe.csv\n"
        "  in_universe_column: in_universe\n"
        "target:\n"
        "  kind: forward_return\n"
        "  horizon: 5\n"
        "output:\n"
        "  root_dir: ./out\n",
    )

    calls: dict[str, object] = {}

    class _FakeIngestor:
        def ingest_core(self, **kwargs):  # type: ignore[no-untyped-def]
            calls["ingest"] = kwargs
            return object()

        def export_case_inputs(self, **kwargs):  # type: ignore[no-untyped-def]
            calls["export"] = kwargs
            out = Path(kwargs["output_dir"])
            out.mkdir(parents=True, exist_ok=True)
            _write(out / "prices.csv", "date,asset,close\n2024-01-02,000001.SZ,10\n")
            _write(out / "universe.csv", "date,asset,in_universe\n2024-01-02,000001.SZ,1\n")
            _write(out / "bp.csv", "date,asset,factor,value\n2024-01-02,000001.SZ,bp,0.5\n")
            return object()

    monkeypatch.setattr(web_ui, "TushareIngestor", _FakeIngestor)

    task = web_ui._RunTask(
        run_id="run_tushare_preset",
        spec_text=spec_path.read_text(encoding="utf-8"),
        spec_filename=spec_path.name,
        evaluation_profile="default_research",
        data_source="tushare",
        data_start_date="2024-01-01",
        data_end_date="2024-03-31",
        data_asset_limit=50,
        tushare_token="token_x",
        render_report=False,
        output_root_dir=str(tmp_path / "runs"),
        data_slice_preset="robust",
    )
    rewritten = _prepare_spec_for_data_source(
        task=task,
        original_spec_path=spec_path,
        cache_root=tmp_path / "source_cache",
    )

    assert rewritten.exists()
    assert calls["ingest"] == {
        "start_date": "2024-01-01",
        "end_date": "2024-03-31",
        "token": "token_x",
        "asset_limit": 50,
    }
    export_call = calls["export"]
    assert isinstance(export_call, dict)
    assert export_call["start_date"] == "2024-01-01"
    assert export_call["end_date"] == "2024-03-31"
    assert export_call["asset_limit"] == 50
    assert export_call["factors"] == ("bp",)
    assert export_call["adjustment"] == "qfq"
    assert export_call["universe_name"] == "listed_90d"
    assert Path(str(export_call["output_dir"])).exists()


def test_prepare_spec_for_data_source_keeps_custom_factor_path_for_unknown_factor(
    tmp_path: Path,
    monkeypatch,
) -> None:
    custom_factor_path = "/tmp/custom_factor.csv"
    spec_path = _write(
        tmp_path / "case_custom.yaml",
        "name: custom_demo\n"
        "factor_name: momentum_20d\n"
        f"factor_path: {custom_factor_path}\n"
        "prices_path: ./old/prices.csv\n"
        "rebalance_frequency: M\n"
        "n_quantiles: 5\n"
        "direction: long\n"
        "universe:\n"
        "  name: demo\n"
        "  path: ./old/universe.csv\n"
        "  in_universe_column: in_universe\n"
        "target:\n"
        "  kind: forward_return\n"
        "  horizon: 5\n"
        "preprocess:\n"
        "  winsorize: true\n"
        "  winsorize_lower: 0.01\n"
        "  winsorize_upper: 0.99\n"
        "  standardization: zscore\n"
        "  min_group_size: 5\n"
        "  min_coverage: 0.6\n"
        "neutralization:\n"
        "  enabled: false\n"
        "  exposures_path:\n"
        "  size_col:\n"
        "  industry_col:\n"
        "  min_obs: 20\n"
        "  ridge: 1.0e-8\n"
        "transaction_cost:\n"
        "  one_way_rate: 0.001\n"
        "output:\n"
        "  root_dir: ./out\n",
    )

    calls: list[dict[str, object]] = []

    def _fake_generate_baostock_inputs(
        output_dir: str | Path,
        start_date: str,
        end_date: str,
        *,
        assets=None,
        asset_limit: int | None = None,
        include_roe: bool = True,
    ):
        calls.append({"include_roe": include_roe, "asset_limit": asset_limit})
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        _write(out / "prices.csv", "date,asset,close\n2024-01-02,000001.SZ,10\n")
        _write(out / "universe.csv", "date,asset,in_universe\n2024-01-02,000001.SZ,1\n")

    monkeypatch.setattr(
        web_ui,
        "generate_baostock_real_case_inputs",
        _fake_generate_baostock_inputs,
    )

    task = web_ui._RunTask(
        run_id="run_custom",
        spec_text=spec_path.read_text(encoding="utf-8"),
        spec_filename=spec_path.name,
        evaluation_profile="default_research",
        data_source="baostock",
        data_start_date="2024-01-01",
        data_end_date="2024-03-31",
        data_asset_limit=20,
        tushare_token=None,
        render_report=False,
        output_root_dir="dist/web_ui_runs",
    )
    rewritten = _prepare_spec_for_data_source(
        task=task,
        original_spec_path=spec_path,
        cache_root=tmp_path / "source_cache",
    )
    text = rewritten.read_text(encoding="utf-8")
    assert f"factor_path: {custom_factor_path}" in text
    assert "prices.csv" in text
    assert "universe.csv" in text
    assert len(calls) == 1
    assert calls[0]["include_roe"] is False


def test_prepare_spec_for_data_source_builds_factor_from_recipe(
    tmp_path: Path,
    monkeypatch,
) -> None:
    spec_path = _write(
        tmp_path / "case_recipe.yaml",
        "name: recipe_demo\n"
        "factor_name: mom_recipe\n"
        "factor_path: ./old/mom_recipe.csv\n"
        "prices_path: ./old/prices.csv\n"
        "factor_input:\n"
        "  mode: recipe\n"
        "  disable_pipeline_preprocess: true\n"
        "  recipe:\n"
        "    base:\n"
        "      method: momentum\n"
        "      window: 2\n"
        "    preprocess:\n"
        "      standardization: none\n"
        "rebalance_frequency: M\n"
        "n_quantiles: 5\n"
        "direction: long\n"
        "universe:\n"
        "  name: demo\n"
        "  path: ./old/universe.csv\n"
        "  in_universe_column: in_universe\n"
        "target:\n"
        "  kind: forward_return\n"
        "  horizon: 5\n"
        "preprocess:\n"
        "  winsorize: true\n"
        "  winsorize_lower: 0.01\n"
        "  winsorize_upper: 0.99\n"
        "  standardization: zscore\n"
        "  min_group_size: 5\n"
        "  min_coverage: 0.6\n"
        "neutralization:\n"
        "  enabled: false\n"
        "  exposures_path:\n"
        "  size_col:\n"
        "  industry_col:\n"
        "  min_obs: 20\n"
        "  ridge: 1.0e-8\n"
        "transaction_cost:\n"
        "  one_way_rate: 0.001\n"
        "output:\n"
        "  root_dir: ./out\n",
    )

    calls: list[dict[str, object]] = []

    def _fake_generate_baostock_inputs(
        output_dir: str | Path,
        start_date: str,
        end_date: str,
        *,
        assets=None,
        asset_limit: int | None = None,
        include_roe: bool = True,
    ):
        calls.append({"include_roe": include_roe, "asset_limit": asset_limit})
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        _write(
            out / "prices.csv",
            "date,asset,close\n"
            "2024-01-01,000001.SZ,10\n"
            "2024-01-02,000001.SZ,10.1\n"
            "2024-01-03,000001.SZ,10.4\n"
            "2024-01-01,000002.SZ,20\n"
            "2024-01-02,000002.SZ,20.2\n"
            "2024-01-03,000002.SZ,20.1\n",
        )
        _write(
            out / "universe.csv",
            "date,asset,in_universe\n"
            "2024-01-01,000001.SZ,1\n"
            "2024-01-02,000001.SZ,1\n"
            "2024-01-03,000001.SZ,1\n"
            "2024-01-01,000002.SZ,1\n"
            "2024-01-02,000002.SZ,1\n"
            "2024-01-03,000002.SZ,1\n",
        )

    monkeypatch.setattr(
        web_ui,
        "generate_baostock_real_case_inputs",
        _fake_generate_baostock_inputs,
    )

    task = web_ui._RunTask(
        run_id="run_recipe",
        spec_text=spec_path.read_text(encoding="utf-8"),
        spec_filename=spec_path.name,
        evaluation_profile="default_research",
        data_source="baostock",
        data_start_date="2024-01-01",
        data_end_date="2024-01-10",
        data_asset_limit=20,
        tushare_token=None,
        render_report=False,
        output_root_dir="dist/web_ui_runs",
    )
    rewritten = _prepare_spec_for_data_source(
        task=task,
        original_spec_path=spec_path,
        cache_root=tmp_path / "source_cache",
    )
    text = rewritten.read_text(encoding="utf-8")
    assert "mom_recipe.csv" in text
    assert "winsorize: false" in text
    assert "standardization: none" in text
    generated_candidates = sorted((tmp_path / "source_cache").glob("baostock_*/mom_recipe.csv"))
    assert len(generated_candidates) == 1
    generated_factor_path = generated_candidates[0]
    assert generated_factor_path.exists()
    factor_text = generated_factor_path.read_text(encoding="utf-8")
    assert ",mom_recipe," in factor_text
    assert len(calls) == 1
    assert calls[0]["include_roe"] is False


def test_prepare_spec_for_data_source_rejects_mixed_recipe_schemas(
    tmp_path: Path,
) -> None:
    spec_path = _write(
        tmp_path / "case_mixed_recipe.yaml",
        "name: mixed_recipe_demo\n"
        "factor_name: mom_recipe\n"
        "factor_path: ./old/mom_recipe.csv\n"
        "prices_path: ./old/prices.csv\n"
        "factor_recipe:\n"
        "  base:\n"
        "    method: momentum\n"
        "    window: 2\n"
        "factor_input:\n"
        "  mode: recipe\n"
        "  recipe:\n"
        "    base:\n"
        "      method: momentum\n"
        "      window: 2\n"
        "rebalance_frequency: M\n"
        "n_quantiles: 5\n"
        "direction: long\n"
        "universe:\n"
        "  name: demo\n"
        "  path: ./old/universe.csv\n"
        "  in_universe_column: in_universe\n"
        "target:\n"
        "  kind: forward_return\n"
        "  horizon: 5\n"
        "output:\n"
        "  root_dir: ./out\n",
    )
    task = web_ui._RunTask(
        run_id="run_mixed_recipe",
        spec_text=spec_path.read_text(encoding="utf-8"),
        spec_filename=spec_path.name,
        evaluation_profile="default_research",
        data_source="baostock",
        data_start_date="2024-01-01",
        data_end_date="2024-01-10",
        data_asset_limit=20,
        tushare_token=None,
        render_report=False,
        output_root_dir="dist/web_ui_runs",
    )
    with pytest.raises(ValueError, match="use one schema only"):
        _prepare_spec_for_data_source(
            task=task,
            original_spec_path=spec_path,
            cache_root=tmp_path / "source_cache",
        )


def test_prepare_spec_for_data_source_builds_missing_custom_factor_from_cached_prices(
    tmp_path: Path,
    monkeypatch,
) -> None:
    spec_path = _write(
        tmp_path / "case_cached_recipe.yaml",
        "name: cached_recipe_demo\n"
        "factor_name: vcimom20_5\n"
        "factor_path: ./old/vcimom20_5.csv\n"
        "prices_path: ./old/prices.csv\n"
        "factor_input:\n"
        "  mode: recipe\n"
        "  disable_pipeline_preprocess: true\n"
        "  recipe:\n"
        "    base:\n"
        "      method: vcimom\n"
        "      residual_window: 4\n"
        "      momentum_window: 4\n"
        "      skip_recent: 1\n"
        "      confirm_window: 3\n"
        "      penalty_window: 2\n"
        "      amount_window: 3\n"
        "rebalance_frequency: W\n"
        "n_quantiles: 5\n"
        "direction: long\n"
        "universe:\n"
        "  name: demo\n"
        "  path: ./old/universe.csv\n"
        "  in_universe_column: in_universe\n"
        "target:\n"
        "  kind: forward_return\n"
        "  horizon: 5\n"
        "output:\n"
        "  root_dir: ./out\n",
    )
    task = web_ui._RunTask(
        run_id="run_cached_recipe",
        spec_text=spec_path.read_text(encoding="utf-8"),
        spec_filename=spec_path.name,
        evaluation_profile="default_research",
        data_source="baostock",
        data_start_date="2024-01-01",
        data_end_date="2024-01-10",
        data_asset_limit=20,
        tushare_token=None,
        render_report=False,
        output_root_dir="dist/web_ui_runs",
    )
    cache_key = web_ui._build_data_source_cache_key(
        task=task,
        start_date="2024-01-01",
        end_date="2024-01-10",
        factor_name="vcimom20_5",
        required_factor_csv="",
        factor_recipe_hash=web_ui._stable_hash_mapping(
            {
                "base": {
                    "method": "vcimom",
                    "residual_window": 4,
                    "momentum_window": 4,
                    "skip_recent": 1,
                    "confirm_window": 3,
                    "penalty_window": 2,
                    "amount_window": 3,
                }
            }
        ),
    )
    input_dir = tmp_path / "source_cache" / f"baostock_{cache_key}"
    input_dir.mkdir(parents=True, exist_ok=True)
    _write(
        input_dir / "prices.csv",
        "date,asset,close,volume,amount\n"
        "2024-01-01,000001.SZ,10,1000,10000\n"
        "2024-01-02,000001.SZ,10.1,1100,11110\n"
        "2024-01-03,000001.SZ,10.4,1200,12480\n"
        "2024-01-04,000001.SZ,10.3,1300,13390\n"
        "2024-01-05,000001.SZ,10.6,1400,14840\n"
        "2024-01-06,000001.SZ,10.7,1500,16050\n"
        "2024-01-01,000002.SZ,20,1000,20000\n"
        "2024-01-02,000002.SZ,20.1,1100,22110\n"
        "2024-01-03,000002.SZ,20.2,1200,24240\n"
        "2024-01-04,000002.SZ,20.4,1300,26520\n"
        "2024-01-05,000002.SZ,20.3,1400,28420\n"
        "2024-01-06,000002.SZ,20.6,1500,30900\n",
    )
    _write(
        input_dir / "universe.csv",
        "date,asset,in_universe\n"
        "2024-01-01,000001.SZ,1\n"
        "2024-01-02,000001.SZ,1\n"
        "2024-01-03,000001.SZ,1\n"
        "2024-01-04,000001.SZ,1\n"
        "2024-01-05,000001.SZ,1\n"
        "2024-01-06,000001.SZ,1\n"
        "2024-01-01,000002.SZ,1\n"
        "2024-01-02,000002.SZ,1\n"
        "2024-01-03,000002.SZ,1\n"
        "2024-01-04,000002.SZ,1\n"
        "2024-01-05,000002.SZ,1\n"
        "2024-01-06,000002.SZ,1\n",
    )

    def _fail_generate(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("generate_baostock_real_case_inputs should not be called")

    monkeypatch.setattr(web_ui, "generate_baostock_real_case_inputs", _fail_generate)
    monkeypatch.setattr(
        web_ui,
        "build_factor_from_recipe_mapping",
        lambda prices, recipe, factor_name: pd.DataFrame(
            {
                "date": ["2024-01-06", "2024-01-06"],
                "asset": ["000001.SZ", "000002.SZ"],
                "factor": [factor_name, factor_name],
                "value": [0.1, -0.1],
            }
        ),
    )

    rewritten = _prepare_spec_for_data_source(
        task=task,
        original_spec_path=spec_path,
        cache_root=tmp_path / "source_cache",
    )

    assert rewritten.exists()
    assert (input_dir / "vcimom20_5.csv").exists()


def test_frontend_batch_parallel_config_prefers_process_mode() -> None:
    config = _build_frontend_batch_parallel_config(3)
    assert config.mode == "process"
    assert config.max_workers == 3
    assert config.factors_per_worker == 2


def test_web_run_store_isolates_frontend_run_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _WebRunStore(
        workspace_root=tmp_path,
        upload_root=tmp_path / "uploads",
        default_output_root=tmp_path / "runs",
    )
    single_calls: list[str] = []
    single_output_roots: list[Path] = []

    def _fake_single(*args, **kwargs):
        spec_obj = Path(args[0])
        single_calls.append(str(spec_obj))
        single_output_roots.append(Path(str(kwargs["output_root_dir"])))
        run_dir = tmp_path / "single_results" / spec_obj.stem
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

    monkeypatch.setattr(web_ui, "run_single_factor_case", _fake_single)

    spec1 = _write(
        tmp_path / "uploads" / "run_1" / "demo_1.yaml",
        "name: demo_1\nfactor_name: bp\nfactor_path: x\nprices_path: y\n",
    )
    spec2 = _write(
        tmp_path / "uploads" / "run_2" / "demo_2.yaml",
        "name: demo_2\nfactor_name: mom\nfactor_path: x\nprices_path: y\n",
    )

    task1 = _RunTask(
        run_id="run_1",
        spec_text=spec1.read_text(encoding="utf-8"),
        spec_filename="demo_1.yaml",
        evaluation_profile="exploratory_screening",
        data_source="manual",
        data_start_date=None,
        data_end_date=None,
        data_asset_limit=None,
        tushare_token=None,
        render_report=False,
        output_root_dir=str(tmp_path / "runs"),
    )
    task2 = _RunTask(
        run_id="run_2",
        spec_text=spec2.read_text(encoding="utf-8"),
        spec_filename="demo_2.yaml",
        evaluation_profile="exploratory_screening",
        data_source="manual",
        data_start_date=None,
        data_end_date=None,
        data_asset_limit=None,
        tushare_token=None,
        render_report=False,
        output_root_dir=str(tmp_path / "runs"),
    )

    with store._lock:
        store._records["run_1"] = _WebRunRecord(
            run_id="run_1",
            status="running",
            submitted_at_utc="2026-04-19T00:00:00Z",
            evaluation_profile="exploratory_screening",
            data_source="manual",
            render_report=False,
            output_root_dir=str(tmp_path / "runs"),
            spec_path=str(spec1),
        )
        store._records["run_2"] = _WebRunRecord(
            run_id="run_2",
            status="running",
            submitted_at_utc="2026-04-19T00:00:01Z",
            evaluation_profile="exploratory_screening",
            data_source="manual",
            render_report=False,
            output_root_dir=str(tmp_path / "runs"),
            spec_path=str(spec2),
        )
        store._tasks["run_1"] = task1
        store._tasks["run_2"] = task2

    store._execute_task_group([task1, task2])

    assert store.get("run_1") is not None
    assert store.get("run_2") is not None
    assert single_calls == [str(spec1), str(spec2)]
    assert single_output_roots == [
        (tmp_path / "runs" / "_web_runs" / "run_1").resolve(),
        (tmp_path / "runs" / "_web_runs" / "run_2").resolve(),
    ]
