# Knowledge Bridge

## Purpose

This document defines the minimum handoff between the `quant-knowledge` vault and
`alpha-lab`.

- `quant-knowledge` stores reusable ideas, methods, factor definitions, and routing.
- `alpha-lab` validates those ideas with reproducible Level 1/2 research workflows.

The bridge exists to reduce repeated context-loading. A factor or method should be
portable from the vault into `alpha-lab` without re-reading long source material.

## Default Workflow

Use this loop by default:

1. distill a vault idea into a compact handoff package
2. map the handoff into a runnable `alpha-lab` case spec
3. run the case and inspect diagnostics
4. export a structured experiment card back to the vault
5. update the originating vault card only after evidence exists

Templates for this loop live in:

- [knowledge_handoff_template.md](/home/yukun_zhao/quant/projects/alpha-lab/docs/templates/knowledge_handoff_template.md)
- [single_factor_recipe_case_template.yaml](/home/yukun_zhao/quant/projects/alpha-lab/docs/templates/single_factor_recipe_case_template.yaml)
- [experiment_feedback_template.md](/home/yukun_zhao/quant/projects/alpha-lab/docs/templates/experiment_feedback_template.md)
- [vcimom20_5_handoff.md](/home/yukun_zhao/quant/projects/alpha-lab/docs/examples/vcimom20_5_handoff.md)

Vault-side governance companion:

- `C:\quant\vault\quant-knowledge\60_playbooks\research-governance\Playbook - Knowledge to alpha-lab Validation Loop.md`

## research-bridge Project Layer

When you want a persistent project loop across `quant-knowledge`, `alpha-lab`,
and ChatGPT Projects, use `alpha-lab bridge ...`.

The bridge creates one project root under:

- `quant-knowledge/55_projects/<project_slug>/`

Default layout:

- `project.yaml`
- `01_project_brief.md`
- `02_project_rules.md`
- `03_card_map.md`
- `10_active_state.md`
- `20_decision_log.md`
- `30_rounds/`
- `40_specs/`
- `50_writeback_drafts/`

Typical command loop:

```bash
alpha-lab bridge init-project --slug momentum-factor --title-zh 动量因子项目 ...
alpha-lab bridge refresh-project-pack --project momentum-factor
alpha-lab bridge start-round --project momentum-factor --topic "三个月成交额加权动量"
alpha-lab bridge scaffold-case --project momentum-factor --round <round_id> --case-name mom_amt_60
alpha-lab bridge summarize-run --project momentum-factor --round <round_id> --run-root <run_dir>
alpha-lab bridge apply-writeback --project momentum-factor --draft <draft_path>
```

The bridge is deliberately file-pack driven:

- it does not automate browser uploads into ChatGPT Projects
- it does not auto-approve vault writeback
- it does not auto-upgrade theory cards

Instead it gives you:

- a stable project pack to upload to ChatGPT Projects
- a per-round discussion bundle
- a case spec draft for `alpha-lab`
- a reviewed writeback draft before formal vault export

## Upstream Inputs From quant-knowledge

Useful upstream artifacts are:

- concept cards in `10_concepts/`
- method cards in `20_methods/`
- factor cards in `30_factors/`
- playbooks in `60_playbooks/`
- pipelines in `80_pipelines/`

`Review Card/` and `40_papers/` are evidence layers. They should only be used when
the formal layer is insufficient.

## Minimum Handoff Package

Before starting an `alpha-lab` experiment, the upstream knowledge should be reduced
to a compact package:

1. `hypothesis`
2. `market`
3. `frequency`
4. `data requirements`
5. `factor or method definition`
6. `preprocess / neutralization / signal plan`
7. `target definition`
8. `evaluation plan`

If these eight items are not clear, the idea is not ready for validation.

## Mapping Into alpha-lab

Typical mapping:

- factor definition -> `factor_input.recipe` or factor CSV
- preprocessing plan -> case-level `preprocess`
- direction -> case-level `direction`
- target -> case-level `target`
- universe and dates -> case spec plus data-source configuration
- evaluation plan -> research profile plus follow-up diagnostics

For single-factor work, the preferred path is:

1. define the factor in `quant-knowledge`
2. map it into a case spec or recipe
3. run in `alpha-lab`
4. inspect diagnostics and robustness
5. export the experiment result back to the vault

## Output Back To quant-knowledge

Validation should flow back as an experiment card, not as ad hoc edits scattered
across the vault.

The return package should include:

- source idea or origin cards
- spec path
- market and date range
- verdict
- key diagnostics
- next step

That output belongs in `50_experiments/` via `export_experiment_card()`.

Example:

```python
from alpha_lab.reporting import export_experiment_card

path = export_experiment_card(
    result,
    name="vcimom20_5_a_share_short_window",
    vault_path="/path/to/quant-knowledge",
    overwrite=True,
)
```

Generated sections of the experiment card should stay machine-owned. Manual
interpretation should be added in the placeholders after export.

Human-written research cards in the vault should be Chinese-first. Keep the
main body and section headings in Chinese by default, and retain English only
when it is required for technical terms, proper nouns, established abbreviations,
formulas, code symbols, file paths, or source titles.

## Lifecycle Rule

Formal cards in the vault should follow this progression:

1. `theoretical`
2. `validated-backtest`
3. `production`
4. `retired`

Promotion from `theoretical` to `validated-backtest` requires actual `alpha-lab`
evidence, not just literature support.

