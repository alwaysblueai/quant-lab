import pandas as pd
import pytest

from alpha_lab.interfaces import FACTOR_OUTPUT_COLUMNS, validate_factor_output


def _canonical() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "asset": ["A", "A"],
            "factor": ["momentum_20d", "momentum_20d"],
            "value": [0.1, 0.2],
        }
    )


def test_validate_factor_output_accepts_canonical_schema():
    df = _canonical()
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


# ---------------------------------------------------------------------------
# New: NaT dates, null/empty asset, null/empty factor
# ---------------------------------------------------------------------------


def test_validate_rejects_nat_date() -> None:
    df = _canonical()
    df.iloc[0, df.columns.get_loc("date")] = pd.NaT
    with pytest.raises(ValueError, match="NaT"):
        validate_factor_output(df)


def test_validate_rejects_null_asset() -> None:
    df = _canonical()
    df.iloc[0, df.columns.get_loc("asset")] = None
    with pytest.raises(ValueError, match="null"):
        validate_factor_output(df)


def test_validate_rejects_empty_string_asset() -> None:
    df = _canonical()
    df.iloc[0, df.columns.get_loc("asset")] = "   "
    with pytest.raises(ValueError, match="empty string"):
        validate_factor_output(df)


def test_validate_rejects_null_factor_name() -> None:
    df = _canonical()
    df.iloc[0, df.columns.get_loc("factor")] = None
    with pytest.raises(ValueError, match="null"):
        validate_factor_output(df)


def test_validate_rejects_empty_string_factor_name() -> None:
    df = _canonical()
    df.iloc[0, df.columns.get_loc("factor")] = ""
    with pytest.raises(ValueError, match="empty string"):
        validate_factor_output(df)
