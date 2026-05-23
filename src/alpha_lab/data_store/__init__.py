"""``alpha_lab.data_store`` — Parquet-backed external dataset store.

Public names are re-exported lazily via ``__getattr__`` so that importing
``alpha_lab.data_store.cli`` (e.g. from the unified-CLI parser-registration
path) does not transitively pull in ``catalog`` → ``pyarrow.dataset`` /
pandas just to register subcommands.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .catalog import (
        CaseInputExportResult,
        CaseSliceBundle,
        DataCatalog,
        DatasetVersion,
        QualityIssue,
        QualityReport,
        RawSnapshotQualityReport,
        SliceSpec,
        SnapshotManifest,
    )
    from .local_zip import (
        LocalZipAshareDailyIngestor,
        LocalZipAshareDailyIngestResult,
        LocalZipAshareDailyOrganizeResult,
    )
    from .tushare import TushareCoreIngestResult, TushareIngestor


__all__ = [
    "CaseInputExportResult",
    "CaseSliceBundle",
    "DataCatalog",
    "DatasetVersion",
    "LocalZipAshareDailyIngestResult",
    "LocalZipAshareDailyIngestor",
    "LocalZipAshareDailyOrganizeResult",
    "QualityIssue",
    "QualityReport",
    "RawSnapshotQualityReport",
    "SliceSpec",
    "SnapshotManifest",
    "TushareCoreIngestResult",
    "TushareIngestor",
]


_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "CaseInputExportResult": (".catalog", "CaseInputExportResult"),
    "CaseSliceBundle": (".catalog", "CaseSliceBundle"),
    "DataCatalog": (".catalog", "DataCatalog"),
    "DatasetVersion": (".catalog", "DatasetVersion"),
    "QualityIssue": (".catalog", "QualityIssue"),
    "QualityReport": (".catalog", "QualityReport"),
    "RawSnapshotQualityReport": (".catalog", "RawSnapshotQualityReport"),
    "SliceSpec": (".catalog", "SliceSpec"),
    "SnapshotManifest": (".catalog", "SnapshotManifest"),
    "LocalZipAshareDailyIngestor": (".local_zip", "LocalZipAshareDailyIngestor"),
    "LocalZipAshareDailyIngestResult": (".local_zip", "LocalZipAshareDailyIngestResult"),
    "LocalZipAshareDailyOrganizeResult": (".local_zip", "LocalZipAshareDailyOrganizeResult"),
    "TushareCoreIngestResult": (".tushare", "TushareCoreIngestResult"),
    "TushareIngestor": (".tushare", "TushareIngestor"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(
            f"module 'alpha_lab.data_store' has no attribute {name!r}"
        )
    from importlib import import_module

    submodule_path, attr = target
    return getattr(import_module(submodule_path, package=__name__), attr)
