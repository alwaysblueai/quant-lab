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


# ---------------------------------------------------------------------------
# Fingerprint cache hardening (OPT-P0-1)
#
# The previous fingerprint only sampled ``(len, value.iat[0])`` and could be
# bypassed by a caller mutating any row past the first. These tests pin the
# stronger fingerprint behavior: any meaningful in-place mutation must force
# re-validation.
# ---------------------------------------------------------------------------


def _larger_canonical() -> pd.DataFrame:
    """A multi-row frame so mid/last-row mutations are testable."""
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
            ),
            "asset": ["A", "A", "B", "B"],
            "factor": ["momentum_20d"] * 4,
            "value": [0.1, 0.2, 0.3, 0.4],
        }
    )


def test_validate_factor_output_fingerprint_detects_mid_row_value_mutation() -> None:
    df = _larger_canonical()
    validate_factor_output(df)  # primes the fingerprint cache

    # Replace a middle row's value with NaN; len + value.iat[0] are unchanged,
    # but n_value_nan changes, so the fingerprint must invalidate the cache.
    df.iloc[2, df.columns.get_loc("value")] = float("nan")

    # Validation must run again; the frame still has at least one non-NaN value
    # so it should pass cleanly. Importantly: it must NOT silently short-circuit.
    validate_factor_output(df)

    # Now turn every value NaN to make the validator's all-NaN check fire,
    # proving the cache is being rebuilt rather than reused.
    df["value"] = float("nan")
    with pytest.raises(ValueError, match="all NaN"):
        validate_factor_output(df)


def test_validate_factor_output_fingerprint_detects_last_row_replacement() -> None:
    df = _larger_canonical()
    validate_factor_output(df)

    # Mutate the boundary row's date to NaT. Old fingerprint only sampled
    # value.iat[0], which is unchanged; new fingerprint includes date.iat[-1].
    df.iloc[-1, df.columns.get_loc("date")] = pd.NaT
    with pytest.raises(ValueError, match="NaT"):
        validate_factor_output(df)


def test_validate_factor_output_fingerprint_detects_duplicate_insertion() -> None:
    df = _larger_canonical()
    validate_factor_output(df)

    # Force a duplicate (date, asset, factor) by overwriting the last row with
    # the first row's coordinates. value.iat[0] is unchanged; new fingerprint
    # changes via date.iat[-1] and asset.iat[-1].
    df.iloc[-1, df.columns.get_loc("date")] = df["date"].iat[0]
    df.iloc[-1, df.columns.get_loc("asset")] = df["asset"].iat[0]
    with pytest.raises(ValueError, match="duplicate"):
        validate_factor_output(df)
