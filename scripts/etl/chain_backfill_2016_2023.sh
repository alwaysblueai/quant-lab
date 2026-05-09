#!/usr/bin/env bash
# Chain runner: waits for minute_to_panel.py 2016-2023 to finish, then runs
# aggregate_daily_pv.py + build_intraday_features.py for those years.
#
# Usage: bash scripts/etl/chain_backfill_2016_2023.sh <minute_to_panel_pid>
# Logs to /tmp/backfill_chain.log

set -uo pipefail
cd "$(dirname "$0")/../.."

LOG=/tmp/backfill_chain.log
M2P_PID="${1:-}"

log() {
    echo "[$(date '+%F %T')] $*" | tee -a "$LOG"
}

log "=== chain_backfill_2016_2023 starting (m2p_pid=$M2P_PID) ==="

# Phase 1: wait for minute_to_panel.py to finish (if PID given)
if [[ -n "$M2P_PID" ]]; then
    log "waiting for minute_to_panel PID=$M2P_PID"
    while kill -0 "$M2P_PID" 2>/dev/null; do
        sleep 60
    done
    log "minute_to_panel exited"
fi

# Sanity: verify Stage B for 2016-2023 exists
for year in 2016 2017 2018 2019 2020 2021 2022 2023; do
    if [[ ! -f "data/processed/minute_panel/year=${year}/part-0.parquet" ]]; then
        log "FATAL: Stage B for year=${year} missing"
        exit 1
    fi
done
log "Stage B 2016-2023 verified"

# Phase 2: aggregate_daily_pv.py for 2016-2023 → temporary base directory.
# Avoid mixing 19-col base partitions into the 78-col promoted intraday_features/.
BASE_TMP=data/processed/intraday_features_base_2016_2023
log "=== Phase 2: aggregate_daily_pv 2016-2023 → $BASE_TMP ==="
python scripts/etl/aggregate_daily_pv.py \
    --years 2016,2017,2018,2019,2020,2021,2022,2023 \
    --output-root "$BASE_TMP" \
    --overwrite \
    >> "$LOG" 2>&1
status=$?
log "aggregate_daily_pv exit=$status"
if [[ $status -ne 0 ]]; then exit $status; fi

# Phase 3: build_intraday_features.py for 2016-2023, batches all.
# Reads from $BASE_TMP, writes 78-col output to intraday_features_backfill.
log "=== Phase 3: build_intraday_features 2016-2023 (batches all) ==="
python scripts/etl/build_intraday_features.py \
    --batches 1,2,3,4 \
    --years 2016,2017,2018,2019,2020,2021,2022,2023 \
    --base-root "$BASE_TMP" \
    --output-root data/processed/intraday_features_backfill \
    --asset-chunk-size 32 \
    --duckdb-threads 4 \
    --overwrite \
    >> "$LOG" 2>&1
status=$?
log "build_intraday_features exit=$status"
if [[ $status -ne 0 ]]; then exit $status; fi

log "=== chain_backfill_2016_2023 PASS ==="
log "next manual step: merge intraday_features_backfill into intraday_features/, then rm intraday_features_base_2016_2023"
