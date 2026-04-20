# Module Classification

This table defines the current default boundary for `alpha-lab`.

| Layer | Status | Modules | Notes |
|---|---|---|---|
| Layer A — Research Integrity | Core (Level 1/2) | `alpha_lab.research_integrity.asof`, `alpha_lab.research_integrity.leakage_checks`, `alpha_lab.research_integrity.reporting`, `alpha_lab.research_integrity.contracts` | PIT/as-of, anti-leakage, temporal correctness |
| Layer B — Factor Research | Core (Level 1) | `alpha_lab.factors.*`, `alpha_lab.labels`, `alpha_lab.evaluation`, `alpha_lab.quantile`, `alpha_lab.turnover`, `alpha_lab.preprocess`, `alpha_lab.neutralization`, `alpha_lab.signal_transforms` | Factor computation and diagnostics |
| Layer C — Portfolio Research | Core (Level 2) | `alpha_lab.strategy`, `alpha_lab.portfolio_research`, `alpha_lab.experiment`, `alpha_lab.walk_forward`, `alpha_lab.splits`, `alpha_lab.costs` | Research-level portfolio validation only |
| Layer D — Reporting / Registry / Export | Core (Level 1/2) | `alpha_lab.reporting`, `alpha_lab.reporting.renderers.*`, `alpha_lab.reporting.research_validation_package`, `alpha_lab.registry`, `alpha_lab.comparison`, `alpha_lab.vault_export`, `alpha_lab.obsidian` | Reproducible outputs and quant-knowledge export |
Default workflow stages (Level 1/2):

1. Level 1 evaluation
2. Campaign triage
3. Level 2 promotion gate
4. Level 2 portfolio validation

Use `alpha-lab profiles` to discover evaluation-profile names for these stages.
