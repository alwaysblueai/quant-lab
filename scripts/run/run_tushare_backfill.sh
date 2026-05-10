#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/yukun_zhao/quant/projects/alpha-lab"
LOG_PREFIX="[alpha-lab-backfill]"

cd "$ROOT"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"

if [[ -z "${TUSHARE_TOKEN:-}" ]]; then
  echo "$LOG_PREFIX missing TUSHARE_TOKEN" >&2
  exit 1
fi

echo "$LOG_PREFIX start $(date -Is)"

uv run --no-sync --frozen python -m alpha_lab.cli data ingest tushare core \
  --start-date 2020-04-01 \
  --end-date 2026-04-14 \
  --mode fundamental \
  --chunk-months 0 \
  --token "$TUSHARE_TOKEN"

uv run --no-sync --frozen python -m alpha_lab.cli data ingest tushare core \
  --start-date 2020-04-01 \
  --end-date 2026-04-14 \
  --mode daily \
  --chunk-months 12 \
  --token "$TUSHARE_TOKEN"

uv run --no-sync --frozen python -m alpha_lab.cli data validate --level all

echo "$LOG_PREFIX done $(date -Is)"
