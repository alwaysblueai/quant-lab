#!/usr/bin/env bash
set -euo pipefail

# Guarded launcher for full/medium model-factor benchmarks.
#
# Usage:
#   scripts/run_model_factor_benchmark_guarded.sh configs/real_cases/model_factor/stock_ridge_medium_bfq.yaml \
#     --output-root-dir /tmp/alpha-lab-medium/cases \
#     --benchmark-output-dir /tmp/alpha-lab-medium/records \
#     --evaluation-profile exploratory_screening \
#     --screening-retrain-every-n-dates 60

MEMORY_LIMIT_GB="${ALPHA_LAB_BENCHMARK_MEMORY_LIMIT_GB:-24}"
MEMORY_LIMIT_KB="$((MEMORY_LIMIT_GB * 1024 * 1024))"

ulimit -v "${MEMORY_LIMIT_KB}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
export PYTHONMALLOC="${PYTHONMALLOC:-malloc}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"

exec uv run --no-sync --frozen python -m alpha_lab.real_cases.model_factor.cli benchmark \
  "$@" \
  --memory-limit-gb "${MEMORY_LIMIT_GB}"
