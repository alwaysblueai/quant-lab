from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from alpha_lab.data_store.catalog import DatasetVersion, SnapshotManifest
from alpha_lab.data_store.cli import _resolve_export_slice_spec, main
from alpha_lab.data_store.tushare import _split_date_range_into_chunks


class _StubCatalog:
    def __init__(self, latest: str | None) -> None:
        self._latest = latest

    def latest_date(self, table_name: str) -> str | None:
        assert table_name == "daily_bars"
        return self._latest


def test_resolve_export_slice_spec_uses_standard_defaults() -> None:
    args = argparse.Namespace(
        slice_preset="standard",
        start_date=None,
        end_date=None,
        output_dir="/tmp/out",
        universe=None,
        factors=[],
        adjustment=None,
        assets_file=None,
        asset_limit=None,
    )

    slice_spec, preset = _resolve_export_slice_spec(args, _StubCatalog("2024-12-31"))

    assert preset.name == "standard"
    assert slice_spec.slice_preset == "standard"
    assert slice_spec.start_date == "2020-01-01"
    assert slice_spec.end_date == "2024-12-31"
    assert slice_spec.universe_name == "listed_90d"
    assert slice_spec.adjustment == "qfq"
    assert slice_spec.factors == ()


def test_resolve_export_slice_spec_allows_explicit_overrides() -> None:
    args = argparse.Namespace(
        slice_preset="pilot",
        start_date="2023-01-01",
        end_date="2024-01-31",
        output_dir="/tmp/out",
        universe="all_ashare",
        factors=["bp"],
        adjustment="raw",
        assets_file=None,
        asset_limit=50,
    )

    slice_spec, preset = _resolve_export_slice_spec(args, _StubCatalog("2024-12-31"))

    assert preset.name == "pilot"
    assert slice_spec.slice_preset == "pilot"
    assert slice_spec.start_date == "2023-01-01"
    assert slice_spec.end_date == "2024-01-31"
    assert slice_spec.universe_name == "all_ashare"
    assert slice_spec.adjustment == "raw"
    assert slice_spec.asset_limit == 50
    assert slice_spec.factors == ("bp",)


def test_resolve_export_slice_spec_accepts_top_liquid_500_override() -> None:
    args = argparse.Namespace(
        slice_preset="standard",
        start_date=None,
        end_date=None,
        output_dir="/tmp/out",
        universe="top_liquid_500",
        factors=[],
        adjustment=None,
        assets_file=None,
        asset_limit=None,
    )

    slice_spec, preset = _resolve_export_slice_spec(args, _StubCatalog("2026-04-01"))

    assert preset.name == "standard"
    assert slice_spec.universe_name == "top_liquid_500"


def test_resolve_export_slice_spec_requires_end_date_when_catalog_empty() -> None:
    args = argparse.Namespace(
        slice_preset="standard",
        start_date=None,
        end_date=None,
        output_dir="/tmp/out",
        universe=None,
        factors=[],
        adjustment=None,
        assets_file=None,
        asset_limit=None,
    )

    with pytest.raises(ValueError, match="end_date is required"):
        _resolve_export_slice_spec(args, _StubCatalog(None))


def test_split_date_range_into_chunks_breaks_long_window() -> None:
    chunks = _split_date_range_into_chunks(
        start_date="2023-04-01",
        end_date="2024-03-31",
        chunk_months=6,
    )

    assert chunks == [
        ("2023-04-01", "2023-09-30"),
        ("2023-10-01", "2024-03-31"),
    ]


def test_data_cli_ingest_uses_chunked_path_and_prints_progress(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    class _FakeIngestor:
        def __init__(self, catalog) -> None:
            self.catalog = catalog

        def ingest_core_chunked(self, **kwargs):
            captured.update(kwargs)
            progress = kwargs["progress_callback"]
            progress("starting")
            return type(
                "_Result",
                (),
                {
                    "snapshot_manifest": SnapshotManifest(
                        snapshot_id="snap_x",
                        vendor="tushare",
                        dataset_name="core",
                        requested_at_utc="2026-04-02T00:00:00Z",
                        request_params={},
                        row_counts={},
                        file_hashes={},
                        time_range={"start_date": "2023-04-01", "end_date": "2023-09-30"},
                        notes={},
                    ),
                    "dataset_version": DatasetVersion(
                        dataset_name="tushare_core",
                        version_id="ver_x",
                        created_at_utc="2026-04-02T00:00:00Z",
                        raw_snapshot_id="snap_x",
                        table_hashes={},
                        notes={},
                    ),
                    "row_counts": {"daily_bars": 12},
                    "chunk_count": 2,
                },
            )()

    monkeypatch.setattr("alpha_lab.data_store.cli.TushareIngestor", _FakeIngestor)

    rc = main(
        [
            "ingest",
            "tushare",
            "core",
            "--start-date",
            "2023-04-01",
            "--end-date",
            "2024-03-31",
            "--chunk-months",
            "6",
            "--daily-research-only",
            "--data-root",
            str(tmp_path / "warehouse"),
        ]
    )

    assert rc == 0
    assert captured["chunk_months"] == 6
    assert captured["mode"] == "full"
    assert captured["daily_research_only"] is True
    assert callable(captured["progress_callback"])
    out = capsys.readouterr().out
    assert "Progress" in out
    assert "Chunks            : 2" in out


def test_data_cli_export_forwards_format_and_cache_flags(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    class _FakeCatalog:
        CORE_DATASET_NAME = "tushare_core"

        def __init__(self, root=None) -> None:
            self.root = Path(root) if root is not None else tmp_path / "warehouse"

        def latest_date(self, table_name: str) -> str | None:
            assert table_name == "daily_bars"
            return "2024-12-31"

        def export_case_inputs(self, **kwargs):
            captured.update(kwargs)
            output_dir = Path(kwargs["output_dir"])
            output_dir.mkdir(parents=True, exist_ok=True)
            return type(
                "_Result",
                (),
                {
                    "output_dir": output_dir,
                    "output_paths": {"prices": output_dir / "prices.csv"},
                    "format_output_paths": {
                        "prices": {
                            "csv": output_dir / "prices.csv",
                            "parquet": output_dir / "prices.parquet",
                        }
                    },
                    "row_counts": {"prices": 2},
                    "dataset_version_id": "ver_x",
                    "cache_key": "cache_x",
                    "cache_dir": tmp_path / "warehouse" / "cache" / "slices" / "cache_x",
                    "formats_written": ("csv", "parquet"),
                },
            )()

    monkeypatch.setattr("alpha_lab.data_store.cli.DataCatalog", _FakeCatalog)

    rc = main(
        [
            "export-case-inputs",
            "--output-dir",
            str(tmp_path / "slice"),
            "--format",
            "both",
            "--prefer-cache",
        ]
    )

    assert rc == 0
    assert captured["formats"] == ("csv", "parquet")
    assert captured["prefer_cache"] is True
    out = capsys.readouterr().out
    assert "Cache dir" in out


def test_data_cli_local_zip_ingest_prints_summary(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    class _FakeIngestor:
        def __init__(self, catalog) -> None:
            self.catalog = catalog

        def ingest_daily_zip(self, **kwargs):
            progress = kwargs["progress_callback"]
            progress("opening archive")
            return type(
                "_Result",
                (),
                {
                    "dataset_version": DatasetVersion(
                        dataset_name="tushare_core",
                        version_id="ver_local_zip",
                        created_at_utc="2026-04-16T00:00:00Z",
                        raw_snapshot_id=None,
                        table_hashes={},
                        notes={"source_vendor": "local_zip"},
                    ),
                    "written_tables": (
                        "daily_bars",
                        "adj_factor",
                        "asset_status",
                        "trade_calendar",
                        "instruments",
                    ),
                    "row_counts": {
                        "daily_bars": 4,
                        "adj_factor": 4,
                        "asset_status": 4,
                        "trade_calendar": 2,
                        "instruments": 2,
                    },
                    "asset_count": 2,
                    "date_range": {"start_date": "2024-01-02", "end_date": "2024-01-03"},
                    "notes": {"source_vendor": "local_zip"},
                },
            )()

    monkeypatch.setattr(
        "alpha_lab.data_store.cli.LocalZipAshareDailyIngestor",
        _FakeIngestor,
    )

    rc = main(
        [
            "ingest",
            "local-zip",
            "ashare-daily",
            "--zip-path",
            str(tmp_path / "stocks.zip"),
            "--data-root",
            str(tmp_path / "warehouse"),
        ]
    )

    assert rc == 0
    out = capsys.readouterr().out
    assert "data-ingest-local-zip-ashare-daily" in out
    assert "Assets" in out
    assert "Progress" in out


def test_data_cli_organize_local_zip_prints_summary(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    class _FakeIngestor:
        def __init__(self, catalog) -> None:
            self.catalog = catalog

        def organize_daily_storage(self, **kwargs):
            progress = kwargs["progress_callback"]
            progress("rewriting daily_bars")
            return type(
                "_Result",
                (),
                {
                    "dataset_version": DatasetVersion(
                        dataset_name="tushare_core",
                        version_id="ver_local_zip_reorg",
                        created_at_utc="2026-04-16T00:00:00Z",
                        raw_snapshot_id=None,
                        table_hashes={},
                        notes={"source_vendor": "local_zip"},
                    ),
                    "rewritten_tables": (
                        "daily_bars",
                        "adj_factor",
                        "asset_status",
                        "trade_calendar",
                        "instruments",
                    ),
                    "row_counts": {
                        "daily_bars": 4,
                        "adj_factor": 4,
                        "asset_status": 4,
                        "trade_calendar": 2,
                        "instruments": 2,
                    },
                    "notes": {"storage_layout": "year_month"},
                },
            )()

    monkeypatch.setattr(
        "alpha_lab.data_store.cli.LocalZipAshareDailyIngestor",
        _FakeIngestor,
    )

    rc = main(
        [
            "organize-local-zip",
            "--data-root",
            str(tmp_path / "warehouse"),
        ]
    )

    assert rc == 0
    out = capsys.readouterr().out
    assert "data-organize-local-zip" in out
    assert "Dataset version" in out
    assert "Progress" in out
