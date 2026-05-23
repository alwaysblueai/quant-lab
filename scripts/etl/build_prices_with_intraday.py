"""Build a joined prices file that combines daily PV truth with intraday features.

Input
-----
- data/processed/real_case_inputs/ashare_institutional_20160418_20260415_supplemented/
    prices.parquet            (31 cols: qfq + raw_* + adj_factor + up/down_limit + index flags)
    universe_mask.parquet
- data/processed/intraday_features/year=*/part-0.parquet
    (78 cols: 14 base + 5 status + 22 batch1 + 15 batch2 + 12 batch3 + 10 batch4)

Output
------
- data/processed/real_case_inputs/ashare_institutional_intraday_v1/
    prices.parquet            (31 + 70 ≈ 99 unique cols, left-joined on (date, asset))
    universe_mask.parquet     (symlink to source)
    slice_manifest.json       (provenance)

Design notes
------------
- Drops the 7 OHLCV columns from intraday_features (open/high/low/close/volume/amount/vwap)
  because prices.parquet already has both qfq and raw_* versions; the intraday OHLCV
  is raw and would clash with the supplement layer's verified accounting.
- Keeps all 60 derived intraday feature columns plus 6 status flags + 2 quality flags +
  2 minute counts.
- Left join: every prices row is preserved. Rows with no minute panel coverage (e.g.,
  prices range extends into 2026 but intraday_features stops at 2025-12-31) get NaN
  intraday columns. Status flags then default to NaN; downstream factor code is expected
  to mask via the existing reliable-rows convention.
- Output uses zstd compression and 500k row groups (good random access; ~3 GB total).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[2]
PRICES_PATH = (
    REPO_ROOT
    / "data"
    / "processed"
    / "real_case_inputs"
    / "ashare_institutional_20160418_20260415_supplemented"
    / "prices.parquet"
)
UNIVERSE_PATH = PRICES_PATH.parent / "universe_mask.parquet"
INTRADAY_ROOT = REPO_ROOT / "data" / "processed" / "intraday_features"
OUTPUT_DIR = (
    REPO_ROOT
    / "data"
    / "processed"
    / "real_case_inputs"
    / "ashare_institutional_intraday_v1"
)

# Columns in intraday_features that overlap with prices.parquet's daily OHLCV.
# We drop them to avoid schema conflict; prices.parquet's qfq + raw_* remain canonical.
OVERLAP_DROP = ["open", "high", "low", "close", "volume", "amount", "vwap"]


def _read_intraday_features(root: Path) -> pd.DataFrame:
    parts = sorted(root.glob("year=*/part-0.parquet"))
    if not parts:
        raise FileNotFoundError(f"no intraday partitions under {root}")
    frames = [pq.read_table(p).to_pandas() for p in parts]
    out = pd.concat(frames, ignore_index=True)
    out["date"] = out["date"].astype(str)
    out["asset"] = out["asset"].astype(str)
    return out.drop(columns=[c for c in OVERLAP_DROP if c in out.columns])


def main() -> int:
    if not PRICES_PATH.exists():
        raise FileNotFoundError(f"prices.parquet missing: {PRICES_PATH}")
    if not INTRADAY_ROOT.exists():
        raise FileNotFoundError(f"intraday_features missing: {INTRADAY_ROOT}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("loading prices.parquet ...", flush=True)
    prices = pd.read_parquet(PRICES_PATH)
    prices["date"] = prices["date"].astype(str)
    prices["asset"] = prices["asset"].astype(str)
    print(f"  prices: {len(prices):,} rows × {len(prices.columns)} cols", flush=True)

    print("loading intraday_features (10 years) ...", flush=True)
    intra = _read_intraday_features(INTRADAY_ROOT)
    print(f"  intraday: {len(intra):,} rows × {len(intra.columns)} cols", flush=True)

    # Defensive uniqueness check; both panels should be one-to-one on (date, asset).
    for name, frame in (("prices", prices), ("intraday", intra)):
        dups = frame.duplicated(subset=["date", "asset"]).sum()
        if dups:
            raise ValueError(f"{name} has {dups} duplicate (date, asset) rows")

    print("merging on (date, asset) left ...", flush=True)
    joined = prices.merge(
        intra,
        on=["date", "asset"],
        how="left",
        validate="one_to_one",
    )
    if "is_session_active" in joined.columns:
        matched = int(joined["is_session_active"].notna().sum())
    else:
        matched = 0
    unmatched = len(joined) - matched
    print(
        f"  joined: {len(joined):,} rows × {len(joined.columns)} cols",
        flush=True,
    )
    print(f"  intraday-matched: {matched:,}  unmatched (NaN intra cols): {unmatched:,}",
          flush=True)

    out_prices = OUTPUT_DIR / "prices.parquet"
    print(f"writing {out_prices} ...", flush=True)
    joined.to_parquet(
        out_prices,
        index=False,
        compression="zstd",
        row_group_size=500_000,
    )

    out_universe = OUTPUT_DIR / "universe_mask.parquet"
    if out_universe.exists() or out_universe.is_symlink():
        out_universe.unlink()
    rel = os.path.relpath(UNIVERSE_PATH, out_universe.parent)
    os.symlink(rel, out_universe)
    print(f"symlinked {out_universe} -> {rel}", flush=True)

    manifest = {
        "name": OUTPUT_DIR.name,
        "source_prices": str(PRICES_PATH.relative_to(REPO_ROOT)),
        "source_intraday_root": str(INTRADAY_ROOT.relative_to(REPO_ROOT)),
        "source_universe": str(UNIVERSE_PATH.relative_to(REPO_ROOT)),
        "row_count": int(len(joined)),
        "column_count": int(len(joined.columns)),
        "intraday_matched_rows": int(matched),
        "intraday_unmatched_rows": int(unmatched),
        "overlap_dropped_from_intraday": OVERLAP_DROP,
        "joined_at": pd.Timestamp.utcnow().isoformat(),
    }
    (OUTPUT_DIR / "slice_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
