"""Build slim prices slice + precomputed factor parquet for a single intraday-augmented factor.

Why this exists
---------------
``alpha-lab real-case single-factor run`` loads the entire ``prices_path`` into
pandas memory and then layers diagnostics on top. The joined prices+intraday
dataset (101 cols × 7.5M rows) blows past the WSL2 user-cgroup budget under
``default_research`` because diagnostics like ``run_param_sensitivity`` and
``run_lag_sensitivity`` re-run the backtest on perturbed inputs and duplicate
the working set several times.

This helper standardises the per-factor optimisation:

1. **Slim prices slice**: read the joined dataset projecting only
   ``BASE_PRICE_COLUMNS`` (17 OHLCV/limit/index cols) plus the intraday-derived
   columns the factor's ``required_columns`` declares. Optionally clip the
   date range to shrink the row count further.
2. **Precomputed factor parquet**: compile ``build_factor`` from the factor.json
   draft and run it once on the slim slice; write a ``(date, asset, factor,
   value)`` parquet so the case YAML can use ``factor_input.mode: file`` and
   bypass the heavy recipe path entirely.

Outputs land at deterministic paths so case YAMLs can be wired to them
without per-factor scripting:

- ``data/processed/real_case_inputs/<dataset_name>/prices.parquet``
  (where ``dataset_name`` defaults to ``ashare_institutional_<factor>_v1``)
- ``custom_factors/research/<factor>/factor_<factor>_v1.parquet``

Usage
-----
::

    python scripts/etl/build_factor_run_inputs.py \\
        --factor signed_jump_neg_5d \\
        --start-date 2020-01-01

Optional flags:
    ``--end-date YYYY-MM-DD``  trim the upper end of the slice
    ``--dataset-name NAME``    override the slim-slice dir name
    ``--output-suffix v2``     bump the factor-parquet suffix
    ``--skip-precompute``      only build the slim slice (use for recipe mode)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np  # noqa: F401  (available to compiled build_factor)
import pandas as pd
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[2]
JOINED_PRICES = (
    REPO_ROOT
    / "data"
    / "processed"
    / "real_case_inputs"
    / "ashare_institutional_intraday_v1"
    / "prices.parquet"
)
UNIVERSE_SOURCE = JOINED_PRICES.parent / "universe_mask.parquet"

# 17 cols the single_factor pipeline uses for label, cost, universe filtering,
# capacity (when amount/turnover present), and limit-touch tradability checks.
BASE_PRICE_COLUMNS: tuple[str, ...] = (
    "date",
    "asset",
    "open",
    "high",
    "low",
    "close",
    "pre_close",
    "volume",
    "amount",
    "vwap",
    "adj_factor",
    "up_limit",
    "down_limit",
    "is_limit_up",
    "is_limit_down",
    "is_suspended",
    "is_st",
)

CUSTOM_FACTORS_ROOT = REPO_ROOT / "custom_factors" / "research"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--factor", required=True, help="Factor name under custom_factors/research/")
    p.add_argument("--start-date", default=None, help="Inclusive lower bound (YYYY-MM-DD).")
    p.add_argument("--end-date", default=None, help="Inclusive upper bound (YYYY-MM-DD).")
    p.add_argument("--dataset-name", default=None, help="Override slim slice dir name.")
    p.add_argument("--output-suffix", default="v1", help="Suffix for the precomputed factor parquet.")
    p.add_argument(
        "--skip-precompute",
        action="store_true",
        help="Only build the slim slice; useful when the factor uses recipe mode for iteration.",
    )
    return p.parse_args()


def _load_factor_meta(factor: str) -> dict[str, object]:
    meta_path = CUSTOM_FACTORS_ROOT / factor / "factor.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"factor.json missing for {factor!r}: {meta_path}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _required_intraday_cols(meta: dict[str, object]) -> list[str]:
    required = list(meta.get("required_columns") or [])
    optional = list(meta.get("optional_columns") or [])
    base = set(BASE_PRICE_COLUMNS)
    extras: list[str] = []
    seen: set[str] = set()
    for col in [*required, *optional]:
        if col in base or col in {"date", "asset"} or col in seen:
            continue
        extras.append(col)
        seen.add(col)
    return extras


def _build_slim_slice(
    *,
    extras: list[str],
    start_date: str | None,
    end_date: str | None,
    dataset_name: str,
) -> Path:
    if not JOINED_PRICES.exists():
        raise FileNotFoundError(f"joined dataset missing: {JOINED_PRICES}")
    available = set(pq.read_schema(JOINED_PRICES).names)
    keep = [c for c in BASE_PRICE_COLUMNS if c in available] + [c for c in extras if c in available]
    missing_extras = [c for c in extras if c not in available]
    if missing_extras:
        raise ValueError(
            f"intraday cols not available in joined dataset: {missing_extras}. "
            f"Available examples: {sorted(c for c in available if c not in BASE_PRICE_COLUMNS)[:8]}"
        )

    out_dir = REPO_ROOT / "data" / "processed" / "real_case_inputs" / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_prices = out_dir / "prices.parquet"

    print(
        f"[slim] reading {JOINED_PRICES.name} cols={len(keep)} extras={extras} ...",
        flush=True,
    )
    df = pd.read_parquet(JOINED_PRICES, columns=keep)
    if start_date is not None or end_date is not None:
        df["date"] = df["date"].astype(str)
        if start_date:
            df = df[df["date"] >= start_date]
        if end_date:
            df = df[df["date"] <= end_date]
    print(f"[slim]   rows={len(df):,} cols={len(df.columns)}", flush=True)

    df.to_parquet(out_prices, index=False, compression="zstd", row_group_size=500_000)
    print(
        f"[slim] wrote {out_prices} ({out_prices.stat().st_size / 1024 / 1024:.0f} MB)",
        flush=True,
    )

    universe_dst = out_dir / "universe_mask.parquet"
    if universe_dst.exists() or universe_dst.is_symlink():
        universe_dst.unlink()
    rel = os.path.relpath(UNIVERSE_SOURCE, universe_dst.parent)
    os.symlink(rel, universe_dst)
    print(f"[slim] symlinked universe -> {rel}", flush=True)

    manifest = {
        "name": dataset_name,
        "source_joined": str(JOINED_PRICES.relative_to(REPO_ROOT)),
        "kept_columns": keep,
        "extra_intraday_columns": extras,
        "start_date": start_date,
        "end_date": end_date,
        "row_count": int(len(df)),
        "built_at": pd.Timestamp.utcnow().isoformat(),
    }
    (out_dir / "slice_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return out_prices


def _precompute_factor(
    *,
    factor: str,
    meta: dict[str, object],
    slim_prices: Path,
    suffix: str,
) -> Path:
    code = str(meta.get("code") or "")
    if not code.strip():
        raise ValueError(f"factor.json for {factor!r} has empty 'code' field")

    namespace: dict[str, object] = {"np": np, "pd": pd}
    compiled = compile(code, f"<custom_factor:{factor}>", "exec")
    exec(compiled, namespace)  # noqa: S102 -- code is the user's own draft factor
    builder = namespace.get("build_factor") or namespace.get("builder")
    if not callable(builder):
        raise ValueError(f"factor.json for {factor!r} defines neither build_factor nor builder")

    required = list(meta.get("required_columns") or [])
    keep_cols = sorted({"date", "asset", *required, "close"})
    available = set(pq.read_schema(slim_prices).names)
    keep_cols = [c for c in keep_cols if c in available]
    print(f"[precompute] reading slim slice cols={keep_cols} ...", flush=True)
    frame = pd.read_parquet(slim_prices, columns=keep_cols)
    print(f"[precompute]   rows={len(frame):,}", flush=True)

    print("[precompute] running build_factor ...", flush=True)
    raw = builder(frame)
    if not isinstance(raw, pd.DataFrame):
        raise ValueError("build_factor must return a DataFrame")
    if not {"date", "asset", "value"}.issubset(raw.columns):
        raise ValueError("build_factor output must contain date, asset, value columns")
    out = raw[["date", "asset", "value"]].copy()
    out["value"] = pd.to_numeric(out["value"], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    out = out.dropna(subset=["value"]).reset_index(drop=True)
    out["factor"] = factor
    out = out[["date", "asset", "factor", "value"]]
    if not pd.api.types.is_string_dtype(out["date"]):
        out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    out_path = CUSTOM_FACTORS_ROOT / factor / f"factor_{factor}_{suffix}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False, compression="zstd")
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(
        f"[precompute] wrote {out_path} rows={len(out):,} size={size_mb:.1f} MB",
        flush=True,
    )
    return out_path


def main() -> int:
    args = _parse_args()
    factor = args.factor.strip()
    meta = _load_factor_meta(factor)
    extras = _required_intraday_cols(meta)

    dataset_name = args.dataset_name or f"ashare_institutional_{factor}_{args.output_suffix}"
    slim = _build_slim_slice(
        extras=extras,
        start_date=args.start_date,
        end_date=args.end_date,
        dataset_name=dataset_name,
    )

    factor_path: Path | None = None
    if not args.skip_precompute:
        factor_path = _precompute_factor(
            factor=factor,
            meta=meta,
            slim_prices=slim,
            suffix=args.output_suffix,
        )

    print()
    print("DONE — wire into case YAML:")
    print(f"  prices_path: ../../../data/processed/real_case_inputs/{dataset_name}/prices.parquet")
    print(f"  universe.path: ../../../data/processed/real_case_inputs/{dataset_name}/universe_mask.parquet")
    if factor_path is not None:
        rel = factor_path.relative_to(REPO_ROOT)
        print("  factor_input:")
        print("    mode: file")
        print("    disable_pipeline_preprocess: true")
        print(f"  factor_path: ../../../{rel}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
