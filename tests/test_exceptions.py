"""Smoke coverage for :mod:`alpha_lab.exceptions`.

Asserts the inheritance contract callers depend on:
- everything inherits from ``AlphaLabError``
- data/config errors are catchable as ``ValueError`` for backward compat
- I/O errors are catchable as ``OSError``
- experiment errors are catchable as ``RuntimeError``
"""

from __future__ import annotations

import pytest

from alpha_lab.exceptions import (
    AlphaLabConfigError,
    AlphaLabDataError,
    AlphaLabError,
    AlphaLabExperimentError,
    AlphaLabIOError,
)


@pytest.mark.parametrize(
    "exc_cls",
    [AlphaLabConfigError, AlphaLabDataError, AlphaLabIOError, AlphaLabExperimentError],
)
def test_all_specific_errors_inherit_alpha_lab_error(exc_cls: type[Exception]) -> None:
    assert issubclass(exc_cls, AlphaLabError)


def test_data_error_is_value_error() -> None:
    with pytest.raises(ValueError):
        raise AlphaLabDataError("invalid")


def test_config_error_is_value_error() -> None:
    with pytest.raises(ValueError):
        raise AlphaLabConfigError("bad spec")


def test_io_error_is_os_error() -> None:
    with pytest.raises(OSError):
        raise AlphaLabIOError("missing file")


def test_experiment_error_is_runtime_error() -> None:
    with pytest.raises(RuntimeError):
        raise AlphaLabExperimentError("no folds")
