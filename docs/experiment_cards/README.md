# Experiment Cards

本目录是 alpha-lab 仓库内的 experiment-card 索引或本地副本。

正式长期归档仍写入 quant-knowledge，但只归档经人工确认需要保留的真实研究证据：

```python
from alpha_lab.reporting import export_experiment_card
```

目标位置：

```text
<OBSIDIAN_VAULT_PATH>/50_experiments/
```

默认探索、Stage 3、demo、smoke、e2e、UI fixture、ad-hoc batch 都应使用：

```bash
--vault-export-mode skip
```

只有当某次 run 被选中作为长期 evidence，才显式使用 `versioned` 或
`overwrite` 写入 vault。建议每个 deepdive 结束后至少记录：

- factor / case id
- verdict
- 关键诊断图表路径
- promotion candidate
- 是否已 export 到 quant-knowledge

不要手工在 `50_experiments/` 创建 Markdown；预审批总结应先放在
`55_projects/<slug>/50_writeback_drafts/`。
