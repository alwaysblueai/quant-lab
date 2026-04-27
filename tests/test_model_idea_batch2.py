from __future__ import annotations

import json
from pathlib import Path

import yaml

from alpha_lab.research_bridge.model_idea import apply_spec_patch_hint, explore_model_idea
from tests.model_factor_case_helpers import write_demo_model_factor_case


def _build_minimal_vault(tmp_path: Path) -> Path:
    vault = tmp_path / "quant-knowledge"
    for rel in ["20_methods", "30_factors", "90_computed", "90_moc"]:
        (vault / rel).mkdir(parents=True, exist_ok=True)
    (vault / "90_moc" / "CARD-INDEX.tsv").write_text(
        "path\ttype\tname\tdomain\tlifecycle\ttags\tparent_moc\tsummary\n"
        "30_factors/Factor - Finance Tree.md\tfactor\tFinance Tree\tmodel\t"
        "validated\tlightgbm,turnover,industry\tMOC - Models\t"
        "Known_at + winsorize + turnover-aware model selection.\n",
        encoding="utf-8",
    )
    (vault / "30_factors" / "Factor - Finance Tree.md").write_text(
        "---\n"
        "type: factor\n"
        "name: Finance Tree\n"
        "summary: Finance tree handling patterns.\n"
        "mechanism: model\n"
        "factor_family: machine_learning\n"
        "---\n\n"
        "Use known_at and winsorize_zscore. Track turnover and industry grouping.\n",
        encoding="utf-8",
    )
    (vault / "90_computed" / "graph.json").write_text(
        json.dumps(
            {
                "meta": {"node_count": 1, "edge_count": 0},
                "nodes": {
                    "Finance Tree": {
                        "type": "factor",
                        "domain": "model",
                        "lifecycle": "validated",
                        "market": "a_share",
                        "mechanism": "model",
                        "factor_family": "machine_learning",
                        "path": "30_factors/Factor - Finance Tree.md",
                    }
                },
                "edges": [],
                "diagnostics": {"dangling_edges": [], "orphan_nodes": []},
            }
        ),
        encoding="utf-8",
    )
    (vault / "90_computed" / "exploration_map.json").write_text(
        json.dumps(
            {
                "meta": {},
                "explored_regions": [],
                "frontier": [
                    {
                        "direction": "turnover-aware tree model",
                        "factor_family": "machine_learning",
                        "mechanism": "model",
                        "reason": "test fixture",
                        "suggested_by": "test",
                        "priority": "high",
                    }
                ],
                "failure_registry_refs": [
                    {
                        "failure_id": "MF-001",
                        "title": "Tree overfit",
                        "status": "watch",
                        "failure_class": "overfit",
                        "failure_statement": "Overfit when no turnover penalty.",
                        "prevention_rule": "Enable turnover-aware metric.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return vault


def _build_model_vault_with_intraday_dependency(tmp_path: Path) -> Path:
    vault = tmp_path / "quant-knowledge-model-data"
    for rel in ["30_factors", "90_computed", "90_moc"]:
        (vault / rel).mkdir(parents=True, exist_ok=True)
    (vault / "90_moc" / "CARD-INDEX.tsv").write_text(
        "path\ttype\tname\tdomain\tlifecycle\ttags\tparent_moc\tsummary\n"
        "30_factors/Factor - Daily Model Baseline.md\tfactor\tDaily Model Baseline\t"
        "model\tvalidated\tdaily,model\tMOC - Models\t"
        "Daily model baseline using close and volume features.\n"
        "30_factors/Factor - Intraday Model Microstructure.md\tfactor\t"
        "Intraday Model Microstructure\tmodel\ttheoretical\tintraday,tick,model\t"
        "MOC - Models\tIntraday volume model requiring intraday_tick_volume.\n",
        encoding="utf-8",
    )
    (vault / "30_factors" / "Factor - Daily Model Baseline.md").write_text(
        "---\n"
        "type: factor\n"
        "name: Daily Model Baseline\n"
        "summary: Daily model baseline using close and volume features.\n"
        "mechanism: model\n"
        "factor_family: machine_learning\n"
        "---\n",
        encoding="utf-8",
    )
    (vault / "30_factors" / "Factor - Intraday Model Microstructure.md").write_text(
        "---\n"
        "type: factor\n"
        "name: Intraday Model Microstructure\n"
        "summary: Intraday volume model requiring intraday_tick_volume.\n"
        "mechanism: model\n"
        "factor_family: machine_learning\n"
        "---\n",
        encoding="utf-8",
    )
    (vault / "90_computed" / "graph.json").write_text(
        json.dumps(
            {
                "meta": {"node_count": 2, "edge_count": 3},
                "nodes": {
                    "Daily Model Baseline": {
                        "type": "factor",
                        "domain": "model",
                        "lifecycle": "validated",
                        "market": "a_share",
                        "mechanism": "model",
                        "factor_family": "machine_learning",
                        "path": "30_factors/Factor - Daily Model Baseline.md",
                    },
                    "Intraday Model Microstructure": {
                        "type": "factor",
                        "domain": "model",
                        "lifecycle": "theoretical",
                        "market": "a_share",
                        "mechanism": "model",
                        "factor_family": "machine_learning",
                        "path": "30_factors/Factor - Intraday Model Microstructure.md",
                    },
                },
                "edges": [
                    {
                        "source": "Daily Model Baseline",
                        "target": "close",
                        "type": "uses_data",
                        "target_kind": "data_identifier",
                        "derived": False,
                    },
                    {
                        "source": "Daily Model Baseline",
                        "target": "volume",
                        "type": "uses_data",
                        "target_kind": "data_identifier",
                        "derived": False,
                    },
                    {
                        "source": "Intraday Model Microstructure",
                        "target": "intraday_tick_volume",
                        "type": "uses_data",
                        "target_kind": "data_identifier",
                        "derived": False,
                    },
                ],
                "diagnostics": {
                    "dangling_edges": [],
                    "orphan_nodes": [],
                    "malformed_fields": [],
                },
            }
        ),
        encoding="utf-8",
    )
    return vault


def _build_workspace_runs(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    (workspace / "dist").mkdir(parents=True, exist_ok=True)

    run_success = workspace / "dist" / "run_success"
    run_success.mkdir(parents=True, exist_ok=True)
    (run_success / "run_manifest.json").write_text(
        json.dumps(
            {
                "workflow": "real_case_model_factor",
                "run_id": "run_success",
                "case_name": "case_success",
                "status": "succeeded",
            }
        ),
        encoding="utf-8",
    )
    (run_success / "metrics.json").write_text(
        json.dumps(
            {
                "metrics": {
                    "factor_name": "alpha_success",
                    "model_family": "ridge",
                    "mean_rank_ic": 0.061,
                    "mean_ic": 0.029,
                    "factor_verdict": "pass",
                }
            }
        ),
        encoding="utf-8",
    )

    run_failed = workspace / "dist" / "run_failed"
    run_failed.mkdir(parents=True, exist_ok=True)
    (run_failed / "run_manifest.json").write_text(
        json.dumps(
            {
                "workflow": "real_case_model_factor",
                "run_id": "run_failed",
                "case_name": "case_failed",
                "status": "failed",
                "error": "integrity check failure",
            }
        ),
        encoding="utf-8",
    )
    (run_failed / "metrics.json").write_text(
        json.dumps(
            {
                "metrics": {
                    "factor_name": "alpha_failed",
                    "model_family": "gbdt",
                    "mean_rank_ic": -0.012,
                    "mean_ic": -0.004,
                    "factor_verdict": "fail",
                    "highest_severity": "fail",
                }
            }
        ),
        encoding="utf-8",
    )
    return workspace


def test_model_idea_loads_knowledge_context(tmp_path: Path) -> None:
    vault = _build_minimal_vault(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)

    payload = explore_model_idea(
        idea="Need a turnover-aware lightgbm model with industry grouping.",
        mode="constrained",
        vault_root=vault,
        workspace_root=workspace,
        top_k=3,
    )

    extras = payload["constraint_report"]["recommendations"]["extras"]
    assert extras["knowledge_context_status"] in {"loaded", "loaded_no_match"}
    assert isinstance(extras["knowledge_matches"], list)
    assert len(extras["knowledge_matches"]) >= 1
    assert isinstance(extras["frontier_matches"], list)
    assert isinstance(extras["failure_refs"], list)
    assert "[K1]" in payload["gpt_prompt"]


def test_model_idea_spec_frequency_infers_inventory_and_filters_hft(
    tmp_path: Path,
) -> None:
    vault = _build_model_vault_with_intraday_dependency(tmp_path)
    spec_path = write_demo_model_factor_case(tmp_path, factor_name="daily_model_idea")

    payload = explore_model_idea(
        idea="intraday volume model improvement",
        mode="explore",
        spec=str(spec_path),
        vault_root=vault,
        workspace_root=tmp_path,
        top_k=4,
    )

    diag = payload["retrieval_diagnostics"]
    assert diag["available_data_provided"] is True
    assert diag["available_data_source"] == "frequency:W"
    dropped = diag["dropped_cards"]
    assert {
        "name": "Intraday Model Microstructure",
        "reason": "missing_data: intraday_tick_volume",
    } in dropped

    extras = payload["constraint_report"]["recommendations"]["extras"]
    kept_names = {row["name"] for row in extras["knowledge_matches"]}
    assert "Intraday Model Microstructure" not in kept_names
    assert "Daily Model Baseline" in kept_names


def test_model_idea_loads_experiment_context_from_dist(tmp_path: Path) -> None:
    workspace = _build_workspace_runs(tmp_path)

    payload = explore_model_idea(
        idea="Improve ridge baseline and avoid failed gbdt setup.",
        mode="constrained",
        workspace_root=workspace,
        top_k=3,
    )

    report = payload["constraint_report"]
    assert isinstance(report["validated_baselines"], list)
    assert len(report["validated_baselines"]) >= 1
    assert isinstance(report["recent_failures"], list)
    assert len(report["recent_failures"]) >= 1
    assert "[E1]" in payload["gpt_prompt"]
    assert "[F1]" in payload["gpt_prompt"]


def test_model_idea_spec_patch_hint_and_apply_merge(tmp_path: Path) -> None:
    spec_path = write_demo_model_factor_case(tmp_path, factor_name="batch2_model_idea")
    original_spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))

    payload = explore_model_idea(
        idea="Use lightgbm with turnover penalty and industry grouping.",
        mode="constrained",
        spec=str(spec_path),
        workspace_root=tmp_path,
    )
    patch_hint = payload["spec_patch_hint"]
    assert isinstance(patch_hint, dict)
    assert patch_hint["requires_code_change"] is False
    patch_fields = patch_hint["patch_fields"]
    assert patch_fields["model"]["family"] == "lightgbm"
    assert patch_fields["model_selection"]["enabled"] is True
    assert patch_fields["feature_preprocess"]["cross_sectional_group_scope"] == "date_and_industry"

    merged = apply_spec_patch_hint(original_spec, patch_hint)
    assert merged["model"]["family"] == "lightgbm"
    assert merged["model_selection"]["enabled"] is True
    assert merged["feature_preprocess"]["cross_sectional_group_scope"] == "date_and_industry"
    assert merged["name"] == original_spec["name"]
