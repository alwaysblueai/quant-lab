#!/usr/bin/env bash
# Run the 4 readiness audits in sequence. Exit non-zero on any blocking failure.
set -euo pipefail
cd "$(dirname "$0")/../.."

echo "=== 1/4 redundancy_audit ==="
python scripts/intraday/redundancy_audit.py
echo "=== 2/4 nan_profile ==="
python scripts/intraday/nan_profile.py
echo "=== 3/4 value_distribution ==="
python scripts/intraday/value_distribution.py
echo "=== 4/4 quick_sanity_ic ==="
python scripts/intraday/quick_sanity_ic.py

echo "=== readiness review pass ==="
ls -la outputs/intraday_etl/
