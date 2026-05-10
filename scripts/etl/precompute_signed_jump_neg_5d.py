"""Pre-compute signed_jump_neg_5d to a small factor parquet (date, asset, value).

Why a precompute step exists
----------------------------
The single-factor pipeline's `mode: recipe` path loads the full prices frame
into memory and lets `factor_recipe` build the signal inline. With the joined
prices+intraday dataset (~7.5M rows × 32 cols slim, ~101 cols full), peak RSS
exceeds the WSL2 user-cgroup memory budget (~12GB) and OOM-kills the run.

Switching the case to `mode: file` lets the pipeline ship a slim 3-column
factor file (~50MB) and use a minimal prices slice for label/cost computation.

Reads
-----
- data/processed/real_case_inputs/ashare_institutional_intraday_signed_jump_v1/prices.parquet
  (32 cols including signed_jump)

Writes
------
- custom_factors/research/signed_jump_neg_5d/factor_signed_jump_neg_5d_v1.parquet
  (date, asset, value)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE = (
    REPO_ROOT
    / "data"
    / "processed"
    / "real_case_inputs"
    / "ashare_institutional_intraday_signed_jump_v1"
    / "prices.parquet"
)
OUT_DIR = REPO_ROOT / "custom_factors" / "research" / "signed_jump_neg_5d"
OUT = OUT_DIR / "factor_signed_jump_neg_5d_v1.parquet"

WINDOW = 5
MIN_PERIODS = 3


def main() -> int:
    if not SOURCE.exists():
        raise FileNotFoundError(f"slim prices missing: {SOURCE}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"reading {SOURCE} (date, asset, signed_jump only) ...", flush=True)
    df = pd.read_parquet(SOURCE, columns=["date", "asset", "signed_jump"])
    print(f"  loaded: {len(df):,} rows", flush=True)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "asset"]).copy()
    df["asset"] = df["asset"].astype(str).str.strip()
    df = df[df["asset"] != ""].copy()
    df = df.sort_values(["asset", "date"], kind="mergesort").reset_index(drop=True)

    print(f"computing rolling({WINDOW}, min_periods={MIN_PERIODS}) per asset ...", flush=True)
    rolled = (
        df.groupby("asset", sort=False)["signed_jump"]
        .rolling(window=WINDOW, min_periods=MIN_PERIODS)
        .mean()
        .reset_index(level=0, drop=True)
    )
    df["value"] = -pd.to_numeric(rolled, errors="coerce").replace([np.inf, -np.inf], np.nan)
    out = df[["date", "asset", "value"]].dropna(subset=["value"]).copy()
    out["factor"] = "signed_jump_neg_5d"
    out = out[["date", "asset", "factor", "value"]]
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")

    print(f"writing {OUT} ({len(out):,} rows) ...", flush=True)
    out.to_parquet(OUT, index=False, compression="zstd")
    print(f"  on-disk size: {OUT.stat().st_size / 1024 / 1024:.1f} MB", flush=True)
    print(f"  date range: {out['date'].min()} .. {out['date'].max()}", flush=True)
    print(f"  unique assets: {out['asset'].nunique():,}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
