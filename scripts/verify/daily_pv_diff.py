"""Verify Stage C daily PV against raw daily columns in prices.parquet."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
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
DEFAULT_INTRADAY_ROOT = REPO_ROOT / "data" / "processed" / "intraday_features"
DEFAULT_REPORT = REPO_ROOT / "outputs" / "verify" / "daily_pv_diff.md"


@dataclass(frozen=True)
class MetricSpec:
    metric: str
    feature_col: str
    price_col: str
    abs_tol: float | None
    rel_tol: float | None


METRICS = [
    MetricSpec("open", "open", "raw_open", 0.01, 1e-4),
    MetricSpec("close", "close", "raw_close", 0.01, 1e-4),
    MetricSpec("high", "high", "raw_high", 0.01, 1e-4),
    MetricSpec("low", "low", "raw_low", 0.01, 1e-4),
    MetricSpec("volume", "volume", "raw_vol", 10.0, 1e-3),
    MetricSpec("amount", "amount", "raw_amount", None, 1e-3),
]


def _sql_path(path: Path | str) -> str:
    return str(path).replace("\\", "/").replace("'", "''")


def _parquet_input(path: Path) -> str:
    if path.is_file():
        return _sql_path(path)
    return _sql_path(path / "year=*" / "*.parquet")


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


def _year_filter_sql(years: list[int] | None, alias: str = "") -> str:
    if not years:
        return "1=1"
    prefix = f"{alias}." if alias else ""
    return f"EXTRACT(year FROM CAST({prefix}date AS DATE)) IN ({','.join(map(str, years))})"


def _rel_expr(feature_col: str, price_col: str) -> str:
    return f"""
        CASE
            WHEN {feature_col} IS NULL OR {price_col} IS NULL THEN NULL
            WHEN abs({price_col}) = 0 THEN
                CASE WHEN abs({feature_col} - {price_col}) = 0 THEN 0 ELSE NULL END
            ELSE abs({feature_col} - {price_col}) / abs({price_col})
        END
    """


def _metric_select(spec: MetricSpec) -> str:
    abs_expr = f"abs({spec.feature_col} - {spec.price_col})"
    rel_expr = _rel_expr(spec.feature_col, spec.price_col)
    pass_terms: list[str] = []
    if spec.abs_tol is not None:
        pass_terms.append(f"{abs_expr} <= {spec.abs_tol}")
    if spec.rel_tol is not None:
        pass_terms.append(f"({rel_expr}) <= {spec.rel_tol}")
    pass_expr = " OR ".join(pass_terms)
    return f"""
        SELECT
            date,
            asset,
            year,
            exchange,
            board,
            event_class,
            '{spec.metric}' AS metric,
            {abs_expr} AS abs_diff,
            {rel_expr} AS rel_diff,
            CASE WHEN {pass_expr} THEN 1 ELSE 0 END AS pass,
            is_session_active,
            is_actively_traded,
            is_likely_suspended,
            is_panel_missing,
            vol_unreliable,
            amt_unreliable,
            stale_days
        FROM joined
    """


def _format_value(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, float):
        if abs(value) >= 1000:
            return f"{value:,.2f}"
        if abs(value) >= 1:
            return f"{value:.6f}"
        return f"{value:.8f}"
    return str(value)


def _markdown_table(frame: pd.DataFrame, *, max_rows: int | None = None) -> str:
    if max_rows is not None:
        frame = frame.head(max_rows)
    if frame.empty:
        return "_无数据_\n"
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(_format_value(row[column]) for column in columns) + " |")
    return "\n".join(lines) + "\n"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare intraday daily PV output against prices.parquet raw columns.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--intraday-root", default=str(DEFAULT_INTRADAY_ROOT))
    parser.add_argument("--prices", default=str(DEFAULT_PRICES))
    parser.add_argument("--asset-whitelist", default=str(DEFAULT_UNIVERSE_MASK))
    parser.add_argument("--date-from", default="2016-10-17")
    parser.add_argument("--date-to", default="2026-04-15")
    parser.add_argument(
        "--years",
        default=None,
        help="Comma-separated year filter, e.g. 2024,2025.",
    )
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--top-n", type=int, default=100)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    years = _parse_years(args.years)
    prices_sql = _sql_path(Path(args.prices))
    intraday_sql = _parquet_input(Path(args.intraday_root))
    year_filter_p = _year_filter_sql(years, "p")
    year_filter_f = _year_filter_sql(years, "f")

    con = duckdb.connect()
    con.register("asset_whitelist", _load_asset_whitelist(Path(args.asset_whitelist)))

    con.execute(
        f"""
        CREATE TEMP TABLE price_keys AS
        SELECT p.date, p.asset
        FROM read_parquet('{prices_sql}') AS p
        JOIN asset_whitelist AS w ON p.asset = w.asset
        WHERE p.date BETWEEN '{args.date_from}' AND '{args.date_to}'
          AND {year_filter_p}
        """
    )
    con.execute(
        f"""
        CREATE TEMP TABLE feature_keys AS
        SELECT f.date, f.asset
        FROM read_parquet('{intraday_sql}', hive_partitioning = 1) AS f
        WHERE f.date BETWEEN '{args.date_from}' AND '{args.date_to}'
          AND {year_filter_f}
        """
    )

    con.execute(
        f"""
        CREATE TEMP TABLE joined AS
        WITH price_enriched AS (
            SELECT
                p.*,
                lag(p.adj_factor) OVER (
                    PARTITION BY p.asset ORDER BY CAST(p.date AS DATE)
                ) AS prev_adj_factor,
                row_number() OVER (
                    PARTITION BY p.asset ORDER BY CAST(p.date AS DATE)
                ) AS asset_day_number
            FROM read_parquet('{prices_sql}') AS p
            JOIN asset_whitelist AS w ON p.asset = w.asset
            WHERE p.date BETWEEN '{args.date_from}' AND '{args.date_to}'
              AND {year_filter_p}
        ),
        feature_enriched AS (
            SELECT *
            FROM read_parquet('{intraday_sql}', hive_partitioning = 1) AS f
            WHERE f.date BETWEEN '{args.date_from}' AND '{args.date_to}'
              AND {year_filter_f}
        ),
        base AS (
            SELECT
                p.date,
                p.asset,
                EXTRACT(year FROM CAST(p.date AS DATE)) AS year,
                right(p.asset, 2) AS exchange,
                CASE
                    WHEN right(p.asset, 2) = 'BJ' THEN 'BJ'
                    WHEN starts_with(p.asset, '688') OR starts_with(p.asset, '689') THEN 'STAR'
                    WHEN starts_with(p.asset, '300') THEN 'CHINEXT'
                    WHEN starts_with(p.asset, '60') THEN 'SH_MAIN'
                    WHEN starts_with(p.asset, '00') THEN 'SZ_MAIN'
                    ELSE 'OTHER'
                END AS board,
                p.raw_open,
                p.raw_high,
                p.raw_low,
                p.raw_close,
                p.raw_vol,
                p.raw_amount,
                p.adj_factor,
                p.prev_adj_factor,
                p.asset_day_number,
                p.is_limit_up,
                p.is_limit_down,
                f.open,
                f.high,
                f.low,
                f.close,
                f.volume,
                f.amount,
                coalesce(f.vol_unreliable, 0) AS vol_unreliable,
                coalesce(f.amt_unreliable, 0) AS amt_unreliable,
                f.is_session_active,
                f.is_actively_traded,
                f.is_likely_suspended,
                f.is_panel_missing,
                f.stale_days
            FROM price_enriched AS p
            LEFT JOIN feature_enriched AS f
              ON p.date = f.date AND p.asset = f.asset
        ),
        with_lags AS (
            SELECT
                *,
                lag(stale_days) OVER (
                    PARTITION BY asset ORDER BY CAST(date AS DATE)
                ) AS prev_stale_days
            FROM base
        )
        SELECT
            *,
            CASE
                WHEN asset_day_number = 1 THEN 'first_price_day'
                WHEN prev_adj_factor IS NOT NULL
                    AND abs(adj_factor / prev_adj_factor - 1) > 0.05
                THEN 'ex_right_day'
                WHEN is_limit_up = 1 OR is_limit_down = 1 THEN 'limit_day'
                WHEN prev_stale_days >= 5 AND is_actively_traded = 1 THEN 'resume_day'
                WHEN is_session_active = 1 AND is_actively_traded = 0
                THEN 'one_line_limit_or_no_trade'
                ELSE 'ordinary'
            END AS event_class
        FROM with_lags
        """
    )

    metric_sql = "\nUNION ALL\n".join(_metric_select(spec) for spec in METRICS)
    con.execute(f"CREATE TEMP TABLE metric_diff AS {metric_sql}")

    gate_summary = con.execute(
        """
        WITH close_gate AS (
            SELECT
                metric,
                'all' AS event_scope,
                count(*) AS rows,
                avg(CAST(pass AS DOUBLE)) AS observed_rate,
                0.9999 AS threshold,
                avg(CAST(pass AS DOUBLE)) >= 0.9999 AS passed,
                'hard' AS gate_type
            FROM metric_diff
            WHERE metric = 'close'
            GROUP BY metric
        ),
        open_non_bj_gate AS (
            SELECT
                metric,
                'non_bj' AS event_scope,
                count(*) AS rows,
                avg(CAST(pass AS DOUBLE)) AS observed_rate,
                0.995 AS threshold,
                avg(CAST(pass AS DOUBLE)) >= 0.995 AS passed,
                'hard' AS gate_type
            FROM metric_diff
            WHERE metric = 'open'
              AND exchange <> 'BJ'
            GROUP BY metric
        ),
        open_bj_gate AS (
            SELECT
                metric,
                'bj_auction' AS event_scope,
                count(*) AS rows,
                avg(CAST(pass AS DOUBLE)) AS observed_rate,
                0.80 AS threshold,
                avg(CAST(pass AS DOUBLE)) >= 0.80 AS passed,
                'known_limitation' AS gate_type
            FROM metric_diff
            WHERE metric = 'open'
              AND exchange = 'BJ'
            GROUP BY metric
        ),
        reliable_pv_rows AS (
            SELECT
                metric,
                pass,
                CASE
                    WHEN event_class = 'ordinary' THEN 'ordinary'
                    ELSE 'event'
                END AS event_bucket,
                CASE WHEN exchange = 'BJ' THEN 'BJ' ELSE 'SHSZ' END AS market_group
            FROM metric_diff
            WHERE (
                metric = 'amount'
                AND coalesce(amt_unreliable, 0) = 0
            )
               OR (
                metric = 'volume'
                AND coalesce(vol_unreliable, 0) = 0
            )
        ),
        reliable_pv_gate_base AS (
            SELECT
                metric,
                event_bucket || '_' || market_group || '_reliable' AS event_scope,
                count(*) AS rows,
                avg(CAST(pass AS DOUBLE)) AS observed_rate,
                CASE
                    WHEN metric = 'amount'
                        AND event_bucket = 'ordinary'
                        AND market_group = 'BJ'
                    THEN 0.985
                    WHEN metric = 'amount'
                        AND event_bucket = 'ordinary'
                    THEN 0.995
                    WHEN metric = 'amount'
                        AND event_bucket = 'event'
                        AND market_group = 'BJ'
                    THEN 0.95
                    WHEN metric = 'amount'
                        AND event_bucket = 'event'
                    THEN 0.99
                    WHEN metric = 'volume'
                        AND event_bucket = 'ordinary'
                        AND market_group = 'BJ'
                    THEN 0.98
                    WHEN metric = 'volume'
                        AND event_bucket = 'ordinary'
                    THEN 0.995
                    WHEN metric = 'volume'
                        AND event_bucket = 'event'
                        AND market_group = 'BJ'
                    THEN 0.975
                    WHEN metric = 'volume'
                        AND event_bucket = 'event'
                    THEN 0.98
                END AS threshold,
                'hard' AS gate_type
            FROM reliable_pv_rows
            GROUP BY metric, event_bucket, market_group
        ),
        reliable_pv_gate AS (
            SELECT
                metric,
                event_scope,
                rows,
                observed_rate,
                threshold,
                observed_rate >= threshold AS passed,
                gate_type
            FROM reliable_pv_gate_base
        ),
        high_low_rows AS (
            SELECT
                metric,
                pass,
                CASE
                    WHEN event_class = 'ordinary' THEN 'ordinary'
                    ELSE 'event'
                END AS event_scope
            FROM metric_diff
            WHERE metric IN ('high', 'low')
        ),
        high_low_gate AS (
            SELECT
                metric,
                event_scope,
                count(*) AS rows,
                avg(CAST(pass AS DOUBLE)) AS observed_rate,
                CASE WHEN event_scope = 'ordinary' THEN 0.95 ELSE 0.80 END AS threshold,
                avg(CAST(pass AS DOUBLE))
                    >= CASE WHEN event_scope = 'ordinary' THEN 0.95 ELSE 0.80 END AS passed,
                'known_limitation' AS gate_type
            FROM high_low_rows
            GROUP BY metric, event_scope
        ),
        metric_gates AS (
            SELECT * FROM close_gate
            UNION ALL
            SELECT * FROM open_non_bj_gate
            UNION ALL
            SELECT * FROM open_bj_gate
            UNION ALL
            SELECT * FROM reliable_pv_gate
            UNION ALL
            SELECT * FROM high_low_gate
        ),
        flag_gates AS (
            SELECT
                'vol_unreliable' AS metric,
                'all' AS event_scope,
                count(*) AS rows,
                avg(CAST(vol_unreliable AS DOUBLE)) AS observed_rate,
                0.01 AS threshold,
                avg(CAST(vol_unreliable AS DOUBLE)) < 0.01 AS passed,
                'informational' AS gate_type
            FROM joined
            UNION ALL
            SELECT
                'amt_unreliable' AS metric,
                'all' AS event_scope,
                count(*) AS rows,
                avg(CAST(amt_unreliable AS DOUBLE)) AS observed_rate,
                0.01 AS threshold,
                avg(CAST(amt_unreliable AS DOUBLE)) < 0.01 AS passed,
                'informational' AS gate_type
            FROM joined
        )
        SELECT
            *,
            gate_type = 'hard' AND passed = false AS blocking
        FROM (
            SELECT *
            FROM metric_gates
            UNION ALL
            SELECT *
            FROM flag_gates
        )
        ORDER BY gate_type, metric, event_scope
        """
    ).fetchdf()

    asset_gap = con.execute(
        """
        SELECT
            (
                SELECT count(*)
                FROM (
                    SELECT DISTINCT asset FROM price_keys
                    EXCEPT
                    SELECT DISTINCT asset FROM feature_keys
                )
            )
                AS assets_only_in_prices,
            (
                SELECT count(*)
                FROM (
                    SELECT DISTINCT asset FROM feature_keys
                    EXCEPT
                    SELECT DISTINCT asset FROM price_keys
                )
            )
                AS assets_only_in_intraday,
            (
                SELECT count(*)
                FROM (
                    SELECT date, asset FROM price_keys
                    EXCEPT
                    SELECT date, asset FROM feature_keys
                )
            )
                AS rows_only_in_prices,
            (
                SELECT count(*)
                FROM (
                    SELECT date, asset FROM feature_keys
                    EXCEPT
                    SELECT date, asset FROM price_keys
                )
            )
                AS rows_only_in_intraday
        """
    ).fetchdf()

    metric_summary = con.execute(
        """
        SELECT
            metric,
            count(*) AS rows,
            avg(CAST(pass AS DOUBLE)) AS pass_rate,
            quantile_cont(abs_diff, 0.50) AS abs_p50,
            quantile_cont(abs_diff, 0.95) AS abs_p95,
            max(abs_diff) AS abs_max,
            quantile_cont(rel_diff, 0.50) AS rel_p50,
            quantile_cont(rel_diff, 0.95) AS rel_p95,
            max(rel_diff) AS rel_max
        FROM metric_diff
        GROUP BY metric
        ORDER BY metric
        """
    ).fetchdf()

    by_year_exchange_board = con.execute(
        """
        SELECT
            metric,
            year,
            exchange,
            board,
            count(*) AS rows,
            1 - avg(CAST(pass AS DOUBLE)) AS mismatch_rate
        FROM metric_diff
        GROUP BY metric, year, exchange, board
        ORDER BY metric, year, exchange, board
        """
    ).fetchdf()

    by_event = con.execute(
        """
        SELECT
            metric,
            event_class,
            count(*) AS rows,
            avg(CAST(pass AS DOUBLE)) AS pass_rate,
            1 - avg(CAST(pass AS DOUBLE)) AS mismatch_rate
        FROM metric_diff
        GROUP BY metric, event_class
        ORDER BY metric, event_class
        """
    ).fetchdf()

    top_worst = con.execute(
        f"""
        WITH row_scores AS (
            SELECT
                date,
                asset,
                exchange,
                board,
                event_class,
                max(
                    CASE metric
                        WHEN 'open' THEN coalesce(abs_diff / 0.01, 0)
                        WHEN 'close' THEN coalesce(abs_diff / 0.01, 0)
                        WHEN 'high' THEN coalesce(abs_diff / 0.01, 0)
                        WHEN 'low' THEN coalesce(abs_diff / 0.01, 0)
                        WHEN 'volume' THEN coalesce(rel_diff / 1e-3, 0)
                        WHEN 'amount' THEN coalesce(rel_diff / 1e-3, 0)
                        ELSE 0
                    END
                ) AS worst_score,
                sum(CASE WHEN pass = 0 THEN 1 ELSE 0 END) AS failed_metrics,
                max(is_session_active) AS is_session_active,
                max(is_actively_traded) AS is_actively_traded,
                max(is_likely_suspended) AS is_likely_suspended,
                max(is_panel_missing) AS is_panel_missing,
                max(vol_unreliable) AS vol_unreliable,
                max(amt_unreliable) AS amt_unreliable,
                max(stale_days) AS stale_days
            FROM metric_diff
            GROUP BY date, asset, exchange, board, event_class
        )
        SELECT *
        FROM row_scores
        WHERE failed_metrics > 0
        ORDER BY worst_score DESC, failed_metrics DESC, date, asset
        LIMIT {int(args.top_n)}
        """
    ).fetchdf()

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = [
        "# Daily PV Diff Report",
        "",
        f"- prices: `{args.prices}`",
        f"- intraday: `{args.intraday_root}`",
        f"- date range: `{args.date_from}` -> `{args.date_to}`",
        f"- years: `{years if years is not None else 'all'}`",
        "",
        "## Key Coverage",
        "",
        _markdown_table(asset_gap),
        "",
        "## Gate Summary",
        "",
        _markdown_table(gate_summary),
        "",
        "## Metric Summary",
        "",
        _markdown_table(metric_summary),
        "",
        "## Mismatch By Year / Exchange / Board",
        "",
        _markdown_table(by_year_exchange_board, max_rows=300),
        "",
        "## Mismatch By Event Class",
        "",
        _markdown_table(by_event),
        "",
        "## Top Worst Rows",
        "",
        _markdown_table(top_worst, max_rows=args.top_n),
    ]
    report_path.write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote report: {report_path}")
    print(_markdown_table(metric_summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
