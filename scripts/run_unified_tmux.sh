#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION="${SESSION:-lab}"
VAULT_ROOT="${VAULT_ROOT:-/mnt/c/quant/vault/quant-knowledge}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8766}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/outputs/runtime_logs}"
ALPHA_LAB_BIN="${ALPHA_LAB_BIN:-}"

section() {
  printf '\n== %s ==\n' "$1"
}

resolve_alpha_lab_bin() {
  if [[ -n "$ALPHA_LAB_BIN" ]]; then
    printf '%s\n' "$ALPHA_LAB_BIN"
    return
  fi
  if [[ -x "$ROOT_DIR/.venv/bin/alpha-lab" ]]; then
    printf '%s\n' "$ROOT_DIR/.venv/bin/alpha-lab"
    return
  fi
  if command -v alpha-lab >/dev/null 2>&1; then
    command -v alpha-lab
    return
  fi
  printf 'alpha-lab executable not found. Set ALPHA_LAB_BIN or create .venv first.\n' >&2
  exit 1
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    printf 'required command not found: %s\n' "$1" >&2
    exit 1
  fi
}

require_cmd tmux
ALPHA_LAB_BIN_RESOLVED="$(resolve_alpha_lab_bin)"
mkdir -p "$LOG_DIR"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/unified_${SESSION}_${STAMP}.log"
RUNNER="$LOG_DIR/unified_${SESSION}_runner.sh"

section "preflight"
printf 'root: %s\n' "$ROOT_DIR"
printf 'session: %s\n' "$SESSION"
printf 'vault_root: %s\n' "$VAULT_ROOT"
printf 'url: http://%s:%s/\n' "$HOST" "$PORT"
printf 'alpha_lab_bin: %s\n' "$ALPHA_LAB_BIN_RESOLVED"
printf 'log_file: %s\n' "$LOG_FILE"
printf 'memory:\n'
free -h || true
printf 'proxy env:\n'
env | grep -i proxy || true

if tmux has-session -t "$SESSION" 2>/dev/null; then
  section "existing session"
  tmux list-sessions | grep "^${SESSION}:" || true
  printf 'Attach with: tmux attach -t %s\n' "$SESSION"
  exit 0
fi

cat >"$RUNNER" <<EOF
#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(printf '%q' "$ROOT_DIR")"
mkdir -p "$(printf '%q' "$LOG_DIR")"
{
  printf 'started_at=%s\n' "\$(date --iso-8601=seconds)"
  printf 'cwd=%s\n' "\$(pwd)"
  printf 'command=%s web unified --vault-root %s --host %s --port %s\n' "$(printf '%q' "$ALPHA_LAB_BIN_RESOLVED")" "$(printf '%q' "$VAULT_ROOT")" "$(printf '%q' "$HOST")" "$(printf '%q' "$PORT")"
  printf 'memory_before:\n'
  free -h || true
  printf 'proxy_env:\n'
  env | grep -i proxy || true
} | tee -a "$(printf '%q' "$LOG_FILE")"
exec "$(printf '%q' "$ALPHA_LAB_BIN_RESOLVED")" web unified --vault-root "$(printf '%q' "$VAULT_ROOT")" --host "$(printf '%q' "$HOST")" --port "$(printf '%q' "$PORT")" 2>&1 | tee -a "$(printf '%q' "$LOG_FILE")"
EOF
chmod +x "$RUNNER"

tmux new-session -d -s "$SESSION" -c "$ROOT_DIR" "$RUNNER"

section "started"
tmux list-sessions | grep "^${SESSION}:" || true
printf 'Attach with: tmux attach -t %s\n' "$SESSION"
printf 'Tail logs: tail -f %s\n' "$LOG_FILE"
