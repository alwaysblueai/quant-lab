#!/usr/bin/env bash
# Promote intraday_features_backfill (year=2016..2023, 78 cols) into the
# canonical intraday_features/ directory after the 2016-2023 backfill chain
# finishes and any audits look reasonable.
#
# Safe-by-default: refuses to overwrite years already in intraday_features.
# Pass --force to override (only after manual review).

set -euo pipefail
cd "$(dirname "$0")/../.."

BACKFILL=data/processed/intraday_features_backfill
TARGET=data/processed/intraday_features
BASE_TMP=data/processed/intraday_features_base_2016_2023
FORCE=${1:-}

if [[ ! -d "$BACKFILL" ]]; then
    echo "FATAL: $BACKFILL not found"
    exit 1
fi

for partition in "$BACKFILL"/year=*; do
    year=$(basename "$partition" | cut -d= -f2)
    target="$TARGET/year=${year}"
    if [[ -d "$target" && "$FORCE" != "--force" ]]; then
        echo "SKIP year=${year}: $target exists (pass --force to overwrite)"
        continue
    fi
    if [[ -d "$target" && "$FORCE" == "--force" ]]; then
        echo "FORCE: removing existing $target"
        rm -rf "$target"
    fi
    echo "promote year=${year}: mv $partition -> $target"
    mv "$partition" "$target"
done

echo "remaining backfill artifacts:"
ls -la "$BACKFILL" 2>/dev/null || echo "  (gone)"

echo "post-promote intraday_features partitions:"
ls "$TARGET"

echo "Stage C base 2016-2023 staging dir is now safe to remove:"
echo "  rm -rf $BASE_TMP"
