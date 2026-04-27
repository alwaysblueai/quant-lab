# Alpha Lab Runtime Stability Runbook

这份 runbook 用来区分三类常见中断来源：VPN/代理连通性问题、WSL OOM、以及单个 model-lab 子进程失败。

## 启动方式

推荐用 tmux 启动统一前端，这样 VPN 抖动或终端断开时进程不会跟着退出：

```bash
cd /home/yukun_zhao/quant/projects/alpha-lab
scripts/run_unified_tmux.sh
```

默认行为：

- tmux session: `lab`
- URL: `http://127.0.0.1:8766/`
- vault root: `/mnt/c/quant/vault/quant-knowledge`
- runtime log: `outputs/runtime_logs/unified_lab_<timestamp>.log`

可选覆盖：

```bash
SESSION=lab PORT=8766 VAULT_ROOT=/mnt/c/quant/vault/quant-knowledge scripts/run_unified_tmux.sh
```

如果 session 已存在，脚本不会重复启动，只会输出 attach 命令：

```bash
tmux attach -t lab
```

## 稳定性诊断

基础诊断：

```bash
scripts/diagnose_runtime_stability.sh
```

包含网络探测的诊断：

```bash
scripts/diagnose_runtime_stability.sh --network
```

诊断输出会覆盖：

- `free -h`、swap、RSS 最大进程
- `dmesg` 中最近的 OOM / killed process 证据
- 当前 proxy 环境变量和 `ip route`
- tmux session
- alpha-lab / python / codex / claude 相关进程
- `outputs/runtime_logs` 中最近的 web 日志
- model-lab 子进程 `_web_run_logs/*/status.json`

## Model Lab 默认资源策略

`web unified` 里的 model-factor run 已默认改成隔离子进程执行。父进程只负责调度与 artifact 收尾，子进程失败不会带崩 web 前端。

默认资源控制：

```bash
ALPHA_LAB_MODEL_LAB_MAX_WORKERS=1
ALPHA_LAB_MODEL_LAB_THREADS=1
```

如要手动提高吞吐，先确认 `free -h` 有足够余量，再临时设置：

```bash
ALPHA_LAB_MODEL_LAB_MAX_WORKERS=2 ALPHA_LAB_MODEL_LAB_THREADS=1 scripts/run_unified_tmux.sh
```

不建议在 20GB WSL memory 下同时提高 workers 和 BLAS/OMP threads；很多模型库会内部并行，容易把 RSS 峰值放大。

## 判断根因

如果终端显示 `Killed`，优先检查：

```bash
dmesg -T | grep -Ei 'oom|out of memory|killed process|invoked oom-killer' | tail -n 30
```

如果有 `oom-killer` 或 `Killed process`，根因是内存压力，不是 VPN。处理顺序：

1. 保持 `ALPHA_LAB_MODEL_LAB_MAX_WORKERS=1`。
2. 降低实验数据窗口、特征数量或模型搜索规模。
3. 检查 `outputs/.../_web_run_logs/<run_id>/status.json` 的 `peak_rss_kb`。
4. 仍不够时再提高 WSL memory/swap。

如果没有 OOM，但 Codex/Claude 或 API 请求失败，优先检查：

```bash
env | grep -i proxy
curl -I https://api.openai.com --max-time 10
```

如果 web 前端仍然可访问但单个 run 失败，优先看该 run 的 artifact：

- `subprocess_status`
- `subprocess_stdout`
- `subprocess_stderr`
- `diagnostics`

这类失败通常是 case spec、数据 schema、模型参数或单个子进程 OOM，不应该再让 `alpha-lab web unified` 整体退出。

## 验收标准

- `alpha-lab web unified` 由 tmux session 托管，断开终端后仍可恢复。
- `scripts/diagnose_runtime_stability.sh` 能输出内存、OOM、代理、tmux、进程和 run 日志诊断。
- model-lab run 失败时保留子进程 stdout/stderr/status artifact。
- 默认 model-lab 并发为 1，避免多个重实验同时抢内存。
