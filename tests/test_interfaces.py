import pandas as pd
import pytest

from alpha_lab.interfaces import FACTOR_OUTPUT_COLUMNS, validate_factor_output


def test_validate_factor_output_accepts_canonical_schema():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "asset": ["A", "A"],
            "factor": ["momentum_20d", "momentum_20d"],
            "value": [0.1, 0.2],
        }
    )

    validate_factor_output(df)
    assert tuple(df.columns) == FACTOR_OUTPUT_COLUMNS


def test_validate_factor_output_rejects_missing_column():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02"]),
            "asset": ["A"],
            "value": [0.1],
        }
    )

    with pytest.raises(ValueError, match="Missing required columns"):
        validate_factor_output(df)


def test_validate_factor_output_rejects_empty():
    df = pd.DataFrame(columns=FACTOR_OUTPUT_COLUMNS)

    with pytest.raises(ValueError, match="empty"):
        validate_factor_output(df)


def test_validate_factor_output_rejects_all_nan_values():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02"]),
            "asset": ["A"],
            "factor": ["momentum_20d"],
            "value": [float("nan")],
        }
    )

    with pytest.raises(ValueError, match="all NaN"):
        validate_factor_output(df)


def test_validate_factor_output_rejects_duplicates():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-02"]),
            "asset": ["A", "A"],
            "factor": ["momentum_20d", "momentum_20d"],
            "value": [0.1, 0.2],
        }
    )

    with pytest.raises(ValueError, match="duplicate"):
        validate_factor_output(df)
