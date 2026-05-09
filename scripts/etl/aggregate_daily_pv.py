"""Aggregate Stage B minute panels into Stage C daily PV features.

This first-stage aggregator intentionally produces only Group A daily PV plus
status flags. Wider intraday feature families should be built after the raw
daily PV gate passes.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import duckdb
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CASE_DIR = (
    REPO_ROOT
    / "data"
    / "processed"
    / "real_case_inputs"
    / "ashare_institutional_20160418_20260415_supplemented"
)
DEFAULT_PRICES = DEFAULT_CASE_DIR / "prices.parquet"
DEFAULT_UNIVERSE_MASK = DEFAULT_CASE_DIR / "universe_mask.parquet"
DEFAULT_PANEL_ROOT = REPO_ROOT / "data" / "processed" / "minute_panel"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "processed" / "intraday_features"

OUTPUT_COLUMNS = [
    "date",
    "asset",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "vwap",
    "vol_unreliable",
    "amt_unreliable",
    "n_minutes_traded",
    "n_minutes_zero_volume",
    "is_session_active",
    "is_actively_traded",
    "is_pre_listing",
    "is_likely_suspended",
    "is_panel_missing",
    "stale_days",
]


def _sql_path(path: Path | str) -> str:
    return str(path).replace("\\", "/").replace("'", "''")


def _load_asset_whitelist(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        assets = pd.read_parquet(path, columns=["asset"])["asset"].drop_duplicates()
    else:
        frame = pd.read_csv(path)
        assets = frame["asset"] if "asset" in frame.columns else frame.iloc[:, 0]
        assets = assets.drop_duplicates()
    return pd.DataFrame({"asset": sorted(str(asset).upper() for asset in assets.dropna())})


def _parse_years(raw: str | None) -> list[int] | None:
    if not raw:
        return None
    return sorted({int(token.strip()) for token in raw.split(",") if token.strip()})


def _year_filter_sql(years: list[int] | None, table_alias: str = "") -> str:
    if not years:
        return "1=1"
    prefix = f"{table_alias}." if table_alias else ""
    return f"EXTRACT(year FROM CAST({prefix}date AS DATE)) IN ({','.join(map(str, years))})"


def _prepare_output(output_root: Path, years: list[int] | None, overwrite: bool) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    existing_year_dirs = [
        path
        for path in output_root.glob("year=*")
        if path.is_dir() and (years is None or int(path.name.split("=")[1]) in years)
    ]
    if existing_year_dirs and not overwrite:
        existing = ", ".join(str(path) for path in existing_year_dirs[:5])
        raise FileExistsError(f"Output year partitions already exist: {existing}")
    if overwrite:
        for path in existing_year_dirs:
            shutil.rmtree(path)

    tmp_root = output_root / "_in_progress" / time.strftime("%Y%m%d_%H%M%S")
    tmp_root.mkdir(parents=True, exist_ok=False)
    return tmp_root


def _finalize_output(tmp_root: Path, output_root: Path) -> None:
    for tmp_year_dir in sorted(tmp_root.glob("year=*")):
        final_year_dir = output_root / tmp_year_dir.name
        if final_year_dir.exists():
            raise FileExistsError(f"Final partition exists before finalize: {final_year_dir}")
        os.replace(tmp_year_dir, final_year_dir)
    summary_path = tmp_root / "_aggregation_summary.json"
    if summary_path.exists():
        os.replace(summary_path, output_root / "_aggregation_summary.json")
    try:
        tmp_root.rmdir()
        tmp_root.parent.rmdir()
    except OSError:
        pass


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate minute panel Stage B into daily PV Stage C.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--minute-panel-root", default=str(DEFAULT_PANEL_ROOT))
    parser.add_argument("--prices", default=str(DEFAULT_PRICES))
    parser.add_argument("--asset-whitelist", default=str(DEFAULT_UNIVERSE_MASK))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--date-from", default="2016-10-17")
    parser.add_argument("--date-to", default="2026-04-15")
    parser.add_argument(
        "--years",
        default=None,
        help="Comma-separated year filter, e.g. 2024,2025.",
    )
    parser.add_argument("--vwap-scale", type=float, default=10.0)
    parser.add_argument(
        "--volume-scale",
        type=float,
        default=0.01,
        help=(
            "Scale summed minute volume into prices.raw_vol units. "
            "CSV is shares; prices uses hands."
        ),
    )
    parser.add_argument(
        "--amount-scale",
        type=float,
        default=0.001,
        help=(
            "Scale summed minute amount into prices.raw_amount units. "
            "CSV is yuan; prices uses k-yuan."
        ),
    )
    parser.add_argument("--coverage-threshold", type=float, default=0.95)
    parser.add_argument(
        "--quality-ratio-threshold",
        type=float,
        default=0.10,
        help="Flag volume/amount rows whose minute-vs-daily ratio differs by more than this.",
    )
    parser.add_argument("--row-group-size", type=int, default=2_000_000)
    parser.add_argument("--compression", default="zstd")
    parser.add_argument("--duckdb-threads", type=int, default=0)
    parser.add_argument("--duckdb-memory-limit", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    years = _parse_years(args.years)
    panel_glob = Path(args.minute_panel_root) / "year=*" / "*.parquet"
    output_root = Path(args.output_root)
    tmp_root = _prepare_output(output_root, years, args.overwrite)

    con = duckdb.connect()
    if args.duckdb_threads > 0:
        con.execute(f"PRAGMA threads={int(args.duckdb_threads)}")
    if args.duckdb_memory_limit:
        con.execute(f"PRAGMA memory_limit='{args.duckdb_memory_limit}'")

    whitelist = _load_asset_whitelist(Path(args.asset_whitelist))
    con.register("asset_whitelist", whitelist)

    price_year_filter = _year_filter_sql(years, "p")
    minute_year_filter = _year_filter_sql(years)
    panel_sql = _sql_path(panel_glob)
    prices_sql = _sql_path(Path(args.prices))

    con.execute(
        f"""
        CREATE TEMP TABLE daily_minute AS
        SELECT
            date,
            asset,
            arg_min(open, datetime) AS open,
            max(high) AS high,
            min(low) AS low,
            arg_max(close, datetime) AS close,
            sum(volume) * {float(args.volume_scale)} AS volume,
            sum(amount) * {float(args.amount_scale)} AS amount,
            CASE
                WHEN sum(volume) > 0 THEN
                    sum(amount)
                    * {float(args.amount_scale)}
                    * {float(args.vwap_scale)}
                    / (sum(volume) * {float(args.volume_scale)})
                ELSE NULL
            END AS vwap,
            CAST(
                sum(
                    CASE
                        WHEN coalesce(volume, 0) > 0 OR coalesce(amount, 0) > 0
                        THEN 1 ELSE 0
                    END
                ) AS BIGINT
            ) AS n_minutes_traded,
            CAST(
                sum(CASE WHEN coalesce(volume, 0) = 0 THEN 1 ELSE 0 END) AS BIGINT
            ) AS n_minutes_zero_volume
        FROM read_parquet('{panel_sql}', hive_partitioning = 1)
        WHERE date BETWEEN '{args.date_from}' AND '{args.date_to}'
          AND {minute_year_filter}
        GROUP BY date, asset
        """
    )

    con.execute(
        f"""
        CREATE TEMP TABLE intraday_features_all AS
        WITH price_base AS (
            SELECT p.date, p.asset, p.raw_vol, p.raw_amount
            FROM read_parquet('{prices_sql}') AS p
            JOIN asset_whitelist AS w ON p.asset = w.asset
            WHERE p.date BETWEEN '{args.date_from}' AND '{args.date_to}'
              AND {price_year_filter}
        ),
        joined AS (
            SELECT
                p.date,
                p.asset,
                d.open,
                d.high,
                d.low,
                d.close,
                d.volume,
                d.amount,
                d.vwap,
                CASE
                    WHEN d.asset IS NULL
                        OR d.volume IS NULL
                        OR p.raw_vol IS NULL
                        OR p.raw_vol <= 0
                    THEN 0
                    WHEN abs(d.volume / p.raw_vol - 1) > {float(args.quality_ratio_threshold)}
                    THEN 1 ELSE 0
                END AS vol_unreliable,
                CASE
                    WHEN d.asset IS NULL
                        OR d.amount IS NULL
                        OR p.raw_amount IS NULL
                        OR p.raw_amount <= 0
                    THEN 0
                    WHEN abs(d.amount / p.raw_amount - 1) > {float(args.quality_ratio_threshold)}
                    THEN 1 ELSE 0
                END AS amt_unreliable,
                d.n_minutes_traded,
                d.n_minutes_zero_volume,
                CASE WHEN d.asset IS NOT NULL THEN 1 ELSE 0 END AS is_session_active,
                CASE
                    WHEN coalesce(d.n_minutes_traded, 0) > 0 THEN 1 ELSE 0
                END AS is_actively_traded,
                0 AS is_pre_listing
            FROM price_base AS p
            LEFT JOIN daily_minute AS d
              ON p.date = d.date AND p.asset = d.asset
        ),
        coverage AS (
            SELECT date, avg(CAST(is_session_active AS DOUBLE)) AS active_ratio
            FROM joined
            GROUP BY date
        ),
        flagged AS (
            SELECT
                j.*,
                CASE
                    WHEN j.is_session_active = 0
                        AND c.active_ratio >= {float(args.coverage_threshold)}
                    THEN 1 ELSE 0
                END AS is_likely_suspended,
                CASE
                    WHEN j.is_session_active = 0
                        AND c.active_ratio < {float(args.coverage_threshold)}
                    THEN 1 ELSE 0
                END AS is_panel_missing
            FROM joined AS j
            JOIN coverage AS c ON j.date = c.date
        ),
        numbered AS (
            SELECT
                *,
                row_number() OVER (PARTITION BY asset ORDER BY CAST(date AS DATE)) AS rn
            FROM flagged
        ),
        stale AS (
            SELECT
                *,
                max(CASE WHEN is_actively_traded = 1 THEN rn ELSE NULL END)
                    OVER (
                        PARTITION BY asset
                        ORDER BY CAST(date AS DATE)
                        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                    )
                    AS last_active_rn
            FROM numbered
        )
        SELECT
            date,
            asset,
            open,
            high,
            low,
            close,
            volume,
            amount,
            vwap,
            CAST(vol_unreliable AS BIGINT) AS vol_unreliable,
            CAST(amt_unreliable AS BIGINT) AS amt_unreliable,
            n_minutes_traded,
            n_minutes_zero_volume,
            CAST(is_session_active AS BIGINT) AS is_session_active,
            CAST(is_actively_traded AS BIGINT) AS is_actively_traded,
            CAST(is_pre_listing AS BIGINT) AS is_pre_listing,
            CAST(is_likely_suspended AS BIGINT) AS is_likely_suspended,
            CAST(is_panel_missing AS BIGINT) AS is_panel_missing,
            CAST(
                CASE WHEN last_active_rn IS NULL THEN NULL ELSE rn - last_active_rn END
                AS BIGINT
            ) AS stale_days,
            EXTRACT(year FROM CAST(date AS DATE)) AS output_year
        FROM stale
        """
    )

    output_years = [
        int(row[0])
        for row in con.execute(
            "SELECT DISTINCT output_year FROM intraday_features_all ORDER BY output_year"
        ).fetchall()
    ]
    for year in output_years:
        tmp_year_dir = tmp_root / f"year={year}"
        tmp_year_dir.mkdir(parents=True, exist_ok=True)
        output_file = tmp_year_dir / "part-0.parquet"
        select_columns = ", ".join(OUTPUT_COLUMNS)
        con.execute(
            f"""
            COPY (
                SELECT {select_columns}
                FROM intraday_features_all
                WHERE output_year = {year}
                ORDER BY date, asset
            )
            TO '{_sql_path(output_file)}'
            (
                FORMAT PARQUET,
                COMPRESSION {args.compression.upper()},
                ROW_GROUP_SIZE {int(args.row_group_size)}
            )
            """
        )

    summary = con.execute(
        """
        SELECT
            count(*) AS rows,
            count(DISTINCT asset) AS assets,
            min(date) AS first_date,
            max(date) AS last_date,
            sum(is_session_active) AS session_active_rows,
            sum(is_actively_traded) AS actively_traded_rows,
            sum(is_likely_suspended) AS likely_suspended_rows,
            sum(is_panel_missing) AS panel_missing_rows,
            sum(vol_unreliable) AS vol_unreliable_rows,
            sum(amt_unreliable) AS amt_unreliable_rows
        FROM intraday_features_all
        """
    ).fetchone()
    summary_dict = {
        "rows": int(summary[0]),
        "assets": int(summary[1]),
        "first_date": summary[2],
        "last_date": summary[3],
        "session_active_rows": int(summary[4] or 0),
        "actively_traded_rows": int(summary[5] or 0),
        "likely_suspended_rows": int(summary[6] or 0),
        "panel_missing_rows": int(summary[7] or 0),
        "vol_unreliable_rows": int(summary[8] or 0),
        "amt_unreliable_rows": int(summary[9] or 0),
        "years": output_years,
        "minute_panel_root": str(args.minute_panel_root),
        "prices": str(args.prices),
        "output_root": str(output_root),
        "volume_scale": float(args.volume_scale),
        "amount_scale": float(args.amount_scale),
        "vwap_scale": float(args.vwap_scale),
        "quality_ratio_threshold": float(args.quality_ratio_threshold),
    }
    (tmp_root / "_aggregation_summary.json").write_text(
        json.dumps(summary_dict, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    _finalize_output(tmp_root, output_root)
    print(json.dumps(summary_dict, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
