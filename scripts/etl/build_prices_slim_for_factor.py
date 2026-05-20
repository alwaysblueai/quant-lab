"""Build a per-factor slim prices.parquet by keeping only columns a factor needs.

Single-factor pipeline currently does not project columns at parquet-read time
(see ``load_prices`` call in ``src/alpha_lab/real_cases/single_factor/pipeline.py``).
Loading the full 101-column joined dataset (~3.1 GB on disk, ~15 GB in pandas)
exhausts WSL2 memory. This script writes a slim variant that retains the 31
canonical price columns plus a user-specified subset of intraday features.

Inputs
------
- Source joined dataset: ``data/processed/real_case_inputs/ashare_institutional_intraday_v1/``
- Source 31-col prices:
  ``data/processed/real_case_inputs/ashare_institutional_20160418_20260415_supplemented/prices.parquet``
  (used as the column whitelist; we keep every column it has)

Output
------
- ``data/processed/real_case_inputs/<output_name>/``
    prices.parquet            (31 base + user-listed intraday cols)
    universe_mask.parquet     (symlink to source)
    slice_manifest.json       (provenance + kept columns)

Usage
-----
    python scripts/etl/build_prices_slim_for_factor.py \\
        --intraday-cols signed_jump \\
        --output-name ashare_institutional_intraday_signed_jump_v1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_JOINED = (
    REPO_ROOT
    / "data"
    / "processed"
    / "real_case_inputs"
    / "ashare_institutional_intraday_v1"
    / "prices.parquet"
)
BASE_PRICES = (
    REPO_ROOT
    / "data"
    / "processed"
    / "real_case_inputs"
    / "ashare_institutional_20160418_20260415_supplemented"
    / "prices.parquet"
)
UNIVERSE_PATH = SOURCE_JOINED.parent / "universe_mask.parquet"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--intraday-cols",
        nargs="+",
        required=True,
        help="Intraday feature columns to keep (e.g. signed_jump rv_5m).",
    )
    p.add_argument(
        "--output-name",
        required=True,
        help="Output dir name under data/processed/real_case_inputs/.",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    if not SOURCE_JOINED.exists():
        raise FileNotFoundError(f"joined prices missing: {SOURCE_JOINED}")
    if not BASE_PRICES.exists():
        raise FileNotFoundError(f"base prices missing: {BASE_PRICES}")

    base_cols = list(pq.read_schema(BASE_PRICES).names)
    joined_cols = set(pq.read_schema(SOURCE_JOINED).names)

    missing = [c for c in args.intraday_cols if c not in joined_cols]
    if missing:
        raise ValueError(f"intraday cols not found in source: {missing}")

    keep = [c for c in base_cols if c in joined_cols] + [
        c for c in args.intraday_cols if c not in base_cols
    ]
    print(f"keeping {len(keep)} cols: {keep}", flush=True)

    out_dir = REPO_ROOT / "data" / "processed" / "real_case_inputs" / args.output_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_prices = out_dir / "prices.parquet"

    print(f"reading {SOURCE_JOINED} (projected to {len(keep)} cols) ...", flush=True)
    df = pd.read_parquet(SOURCE_JOINED, columns=keep)
    print(f"  loaded: {len(df):,} rows × {len(df.columns)} cols", flush=True)

    print(f"writing {out_prices} ...", flush=True)
    df.to_parquet(out_prices, index=False, compression="zstd", row_group_size=500_000)

    out_universe = out_dir / "universe_mask.parquet"
    if out_universe.exists() or out_universe.is_symlink():
        out_universe.unlink()
    rel = os.path.relpath(UNIVERSE_PATH, out_universe.parent)
    os.symlink(rel, out_universe)
    print(f"symlinked {out_universe} -> {rel}", flush=True)

    manifest = {
        "name": args.output_name,
        "source_joined": str(SOURCE_JOINED.relative_to(REPO_ROOT)),
        "source_universe": str(UNIVERSE_PATH.relative_to(REPO_ROOT)),
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "kept_columns": keep,
        "extra_intraday_columns": list(args.intraday_cols),
        "built_at": pd.Timestamp.utcnow().isoformat(),
    }
    (out_dir / "slice_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
