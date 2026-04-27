#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NETWORK_CHECK=0
if [[ "${1:-}" == "--network" ]]; then
  NETWORK_CHECK=1
fi

section() {
  printf '\n== %s ==\n' "$1"
}

run_or_note() {
  local label="$1"
  shift
  printf '$ %s\n' "$label"
  "$@" || true
}

section "runtime"
printf 'time: %s\n' "$(date --iso-8601=seconds)"
printf 'root: %s\n' "$ROOT_DIR"
printf 'kernel: %s\n' "$(uname -a)"
printf 'python: %s\n' "$(command -v python || true)"

section "git snapshot"
cd "$ROOT_DIR"
git status --short | sed -n '1,80p' || true

section "memory and swap"
run_or_note "free -h" free -h
run_or_note "swapon --show" swapon --show
run_or_note "top rss processes" bash -lc "ps -eo pid,ppid,rss,pmem,comm,args --sort=-rss | head -n 15"

section "recent oom evidence"
if command -v dmesg >/dev/null 2>&1; then
  dmesg -T 2>/dev/null | grep -Ei 'oom|out of memory|killed process|invoked oom-killer' | tail -n 30 || true
else
  printf 'dmesg not available\n'
fi

section "proxy and network"
env | grep -i proxy || true
run_or_note "ip route" ip route
if [[ "$NETWORK_CHECK" == "1" ]]; then
  if command -v curl >/dev/null 2>&1; then
    printf '$ curl -I https://api.openai.com --max-time 10\n'
    curl -I https://api.openai.com --max-time 10 || true
  else
    printf 'curl not available\n'
  fi
else
  printf 'skip network probe; pass --network to test curl reachability\n'
fi

section "tmux sessions"
if command -v tmux >/dev/null 2>&1; then
  tmux list-sessions 2>/dev/null || printf 'no tmux sessions\n'
else
  printf 'tmux not available\n'
fi

section "alpha-lab processes"
ps -eo pid,ppid,rss,pmem,etime,comm,args --sort=-rss \
  | grep -E 'alpha-lab|alpha_lab|python|codex|claude' \
  | grep -v grep \
  | head -n 30 || true

section "runtime logs"
if [[ -d "$ROOT_DIR/outputs/runtime_logs" ]]; then
  find "$ROOT_DIR/outputs/runtime_logs" -maxdepth 1 -type f -name '*.log' -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr \
    | head -n 5 \
    | while read -r _ path; do
        printf '\n-- %s --\n' "$path"
        tail -n 30 "$path" || true
      done
else
  printf 'no runtime log directory yet: %s\n' "$ROOT_DIR/outputs/runtime_logs"
fi

section "model-lab subprocess statuses"
if [[ -d "$ROOT_DIR/outputs" ]]; then
  find "$ROOT_DIR/outputs" -path '*_web_run_logs*' -name status.json -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr \
    | head -n 10 \
    | while read -r _ path; do
        printf '\n-- %s --\n' "$path"
        python - "$path" <<'PY' || cat "$path"
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
keys = [
    "status",
    "run_id",
    "case_name",
    "pid",
    "returncode",
    "elapsed_seconds",
    "peak_rss_kb",
    "stdout_log",
    "stderr_log",
]
for key in keys:
    if key in payload:
        print(f"{key}: {payload[key]}")
PY
      done
else
  printf 'no outputs directory yet: %s\n' "$ROOT_DIR/outputs"
fi

section "diagnosis hints"
cat <<'EOF'
- If dmesg shows oom-killer/Killed process, the root cause is memory pressure, not VPN.
- If curl fails but no OOM appears, inspect proxy env and WSL gateway/127.0.0.1:7897 reachability.
- If alpha-lab web is alive but a run failed, open the run's subprocess_stderr/subprocess_status artifacts.
- Keep ALPHA_LAB_MODEL_LAB_MAX_WORKERS=1 unless memory headroom is clearly sufficient.
EOF
