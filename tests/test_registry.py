from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

from alpha_lab.registry import (
    REGISTRY_COLUMNS,
    append_to_registry,
    load_registry,
    register_experiment,
)
from alpha_lab.reporting import SUMMARY_COLUMNS

# ---------------------------------------------------------------------------
# Shared fixture builder
# ---------------------------------------------------------------------------


def _make_summary(
    factor_name: str = "momentum_20d",
    mean_ic: float = 0.05,
    cost_rate: float = float("nan"),
    mean_cost_adjusted_long_short_return: float = float("nan"),
) -> pd.DataFrame:
    row = {
        "factor_name": factor_name,
        "label_name": "forward_return_5",
        "n_quantiles": 5,
        "split_description": "full_sample",
        "mean_ic": mean_ic,
        "mean_rank_ic": 0.06,
        "ic_ir": 1.2,
        "mean_long_short_return": 0.003,
        "long_short_hit_rate": 0.55,
        "n_dates_used": 100,
        "mean_long_short_turnover": 0.4,
        "cost_rate": cost_rate,
        "mean_cost_adjusted_long_short_return": mean_cost_adjusted_long_short_return,
    }
    return pd.DataFrame([row], columns=list(SUMMARY_COLUMNS))


@pytest.fixture
def rpath(tmp_path: Path) -> Path:
    return tmp_path / "registry.csv"


# ---------------------------------------------------------------------------
# 1. load_registry — file does not exist
# ---------------------------------------------------------------------------


def test_load_registry_missing_file_returns_empty_df(rpath: Path):
    df = load_registry(rpath)
    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_load_registry_missing_file_has_registry_columns(rpath: Path):
    df = load_registry(rpath)
    assert set(REGISTRY_COLUMNS).issubset(df.columns)


# ---------------------------------------------------------------------------
# 2. register_experiment — basic correctness
# ---------------------------------------------------------------------------


def test_register_experiment_creates_file(rpath: Path):
    assert not rpath.exists()
    register_experiment("exp1", _make_summary(), rpath)
    assert rpath.exists()


def test_register_experiment_one_row_written(rpath: Path):
    register_experiment("exp1", _make_summary(), rpath)
    df = load_registry(rpath)
    assert len(df) == 1


def test_register_experiment_name_stored(rpath: Path):
    register_experiment("my_run_1", _make_summary(), rpath)
    df = load_registry(rpath)
    assert df["experiment_name"].iloc[0] == "my_run_1"


def test_register_experiment_factor_name_stored(rpath: Path):
    register_experiment("exp1", _make_summary(factor_name="rev_20d"), rpath)
    df = load_registry(rpath)
    assert df["factor_name"].iloc[0] == "rev_20d"


def test_register_experiment_label_factor_mapped_from_label_name(rpath: Path):
    register_experiment("exp1", _make_summary(), rpath)
    df = load_registry(rpath)
    assert df["label_factor"].iloc[0] == "forward_return_5"


def test_register_experiment_quantiles_mapped_from_n_quantiles(rpath: Path):
    register_experiment("exp1", _make_summary(), rpath)
    df = load_registry(rpath)
    assert int(df["quantiles"].iloc[0]) == 5


def test_register_experiment_mean_ic_preserved(rpath: Path):
    register_experiment("exp1", _make_summary(mean_ic=0.042), rpath)
    df = load_registry(rpath)
    assert float(df["mean_ic"].iloc[0]) == pytest.approx(0.042)


def test_register_experiment_timestamp_is_string(rpath: Path):
    register_experiment("exp1", _make_summary(), rpath)
    df = load_registry(rpath)
    ts = df["timestamp"].iloc[0]
    assert isinstance(ts, str)
    assert len(ts) > 0


def test_register_experiment_timestamp_is_iso_format(rpath: Path):
    register_experiment("exp1", _make_summary(), rpath)
    df = load_registry(rpath)
    ts = str(df["timestamp"].iloc[0])
    # ISO format: YYYY-MM-DDTHH:MM:SS — basic structural check
    assert "T" in ts
    assert len(ts) >= 19


def test_register_experiment_obsidian_path_stored(rpath: Path):
    register_experiment("exp1", _make_summary(), rpath, obsidian_path="notes/exp1.md")
    df = load_registry(rpath)
    assert df["obsidian_path"].iloc[0] == "notes/exp1.md"


def test_register_experiment_obsidian_path_empty_string_when_none(rpath: Path):
    register_experiment("exp1", _make_summary(), rpath)
    df = load_registry(rpath)
    val = df["obsidian_path"].iloc[0]
    assert val == "" or (isinstance(val, float) and math.isnan(val))


def test_register_experiment_nan_cost_rate_round_trips(rpath: Path):
    register_experiment("exp1", _make_summary(cost_rate=float("nan")), rpath)
    df = load_registry(rpath)
    assert math.isnan(float(df["cost_rate"].iloc[0]))


def test_register_experiment_finite_cost_rate_round_trips(rpath: Path):
    register_experiment("exp1", _make_summary(cost_rate=0.001), rpath)
    df = load_registry(rpath)
    assert float(df["cost_rate"].iloc[0]) == pytest.approx(0.001)


# ---------------------------------------------------------------------------
# 3. append behavior — multiple experiments
# ---------------------------------------------------------------------------


def test_append_two_experiments(rpath: Path):
    register_experiment("exp1", _make_summary(factor_name="f1"), rpath)
    register_experiment("exp2", _make_summary(factor_name="f2"), rpath)
    df = load_registry(rpath)
    assert len(df) == 2


def test_append_preserves_insertion_order(rpath: Path):
    for i in range(4):
        register_experiment(f"exp{i}", _make_summary(factor_name=f"f{i}"), rpath)
    df = load_registry(rpath)
    assert list(df["experiment_name"]) == ["exp0", "exp1", "exp2", "exp3"]


def test_append_five_experiments(rpath: Path):
    for i in range(5):
        register_experiment(f"run{i}", _make_summary(factor_name=f"f{i}"), rpath)
    df = load_registry(rpath)
    assert len(df) == 5
    assert list(df["factor_name"]) == [f"f{i}" for i in range(5)]


def test_append_allows_duplicate_names(rpath: Path):
    """Registry is an append-only log; duplicate experiment names are allowed."""
    register_experiment("same_name", _make_summary(mean_ic=0.01), rpath)
    register_experiment("same_name", _make_summary(mean_ic=0.02), rpath)
    df = load_registry(rpath)
    assert len(df) == 2
    assert list(df["mean_ic"].astype(float)) == pytest.approx([0.01, 0.02])


# ---------------------------------------------------------------------------
# 4. load_registry — after writes
# ---------------------------------------------------------------------------


def test_load_registry_returns_dataframe(rpath: Path):
    register_experiment("exp1", _make_summary(), rpath)
    df = load_registry(rpath)
    assert isinstance(df, pd.DataFrame)


def test_load_registry_has_all_registry_columns(rpath: Path):
    register_experiment("exp1", _make_summary(), rpath)
    df = load_registry(rpath)
    for col in REGISTRY_COLUMNS:
        assert col in df.columns


def test_load_registry_index_reset(rpath: Path):
    for i in range(3):
        register_experiment(f"e{i}", _make_summary(), rpath)
    df = load_registry(rpath)
    assert list(df.index) == [0, 1, 2]


# ---------------------------------------------------------------------------
# 5. Input validation — register_experiment
# ---------------------------------------------------------------------------


def test_register_experiment_rejects_empty_summary(rpath: Path):
    empty = pd.DataFrame(columns=list(SUMMARY_COLUMNS))
    with pytest.raises(ValueError, match="empty"):
        register_experiment("exp1", empty, rpath)


def test_register_experiment_rejects_non_dataframe(rpath: Path):
    with pytest.raises(TypeError):
        register_experiment("exp1", {"factor_name": "f"}, rpath)  # type: ignore[arg-type]


def test_register_experiment_rejects_missing_columns(rpath: Path):
    bad = pd.DataFrame([{"factor_name": "f"}])
    with pytest.raises(ValueError, match="missing required columns"):
        register_experiment("exp1", bad, rpath)


def test_register_experiment_rejects_extra_columns(rpath: Path):
    """Summary with columns beyond SUMMARY_COLUMNS must be rejected."""
    s = _make_summary()
    s_extra = s.copy()
    s_extra["bonus_col"] = 42
    with pytest.raises(ValueError, match="unexpected columns"):
        register_experiment("exp1", s_extra, rpath)


def test_register_experiment_rejects_multi_row_summary(rpath: Path):
    """A summary with more than one row must be rejected; only 1-row summaries
    are valid so that no rows are silently dropped."""
    multi = pd.concat([_make_summary(), _make_summary()], ignore_index=True)
    with pytest.raises(ValueError, match="exactly one row"):
        register_experiment("exp1", multi, rpath)


# ---------------------------------------------------------------------------
# 6. Input validation — append_to_registry
# ---------------------------------------------------------------------------


def test_append_to_registry_rejects_missing_registry_columns(rpath: Path):
    bad_row = pd.DataFrame([{"experiment_name": "e1", "factor_name": "f"}])
    with pytest.raises(ValueError, match="missing required registry columns"):
        append_to_registry(bad_row, rpath)


# ---------------------------------------------------------------------------
# 7. Schema drift protection
# ---------------------------------------------------------------------------


def test_load_registry_raises_on_corrupt_schema(rpath: Path):
    """A registry file missing expected columns raises ValueError on load."""
    corrupt = pd.DataFrame([{"experiment_name": "e1", "factor_name": "f"}])
    corrupt.to_csv(rpath, index=False)
    with pytest.raises(ValueError, match="incompatible schema"):
        load_registry(rpath)


def test_load_registry_raises_on_extra_columns(rpath: Path):
    """A registry file with extra (unexpected) columns must also raise —
    extra columns are treated as schema drift."""
    register_experiment("exp1", _make_summary(), rpath)
    df = pd.read_csv(rpath)
    df["extra_col"] = "oops"
    df.to_csv(rpath, index=False)
    with pytest.raises(ValueError, match="incompatible schema"):
        load_registry(rpath)


def test_load_registry_raises_on_reordered_columns(rpath: Path):
    """A registry file with the correct columns in a different order must raise.
    CSV append is positional: a reordered header would silently corrupt data.
    """
    register_experiment("exp1", _make_summary(), rpath)
    df = pd.read_csv(rpath)
    # Reverse the column order — same set, wrong sequence
    df = df[list(reversed(df.columns))]
    df.to_csv(rpath, index=False)
    with pytest.raises(ValueError, match="incompatible schema"):
        load_registry(rpath)


def test_append_to_registry_raises_on_corrupt_existing_file(rpath: Path):
    """Appending to a file with missing columns raises ValueError."""
    corrupt = pd.DataFrame([{"experiment_name": "e1", "factor_name": "f"}])
    corrupt.to_csv(rpath, index=False)
    register_summary_row = pd.DataFrame(
        [
            {
                "experiment_name": "e2",
                "factor_name": "f2",
                "label_factor": "fwd_1",
                "quantiles": 5,
                "split_description": "full_sample",
                "cost_rate": float("nan"),
                "mean_ic": 0.05,
                "ic_ir": 1.2,
                "mean_long_short_return": 0.003,
                "mean_cost_adjusted_long_short_return": float("nan"),
                "timestamp": "2024-01-01T00:00:00",
                "obsidian_path": "",
            }
        ],
        columns=list(REGISTRY_COLUMNS),
    )
    with pytest.raises(ValueError, match="incompatible schema"):
        append_to_registry(register_summary_row, rpath)


def test_append_to_registry_raises_on_reordered_existing_file(rpath: Path):
    """Appending to a file whose columns are correct but reordered must raise —
    a positional CSV append against a reordered header would corrupt data."""
    register_experiment("exp1", _make_summary(), rpath)
    df = pd.read_csv(rpath)
    df = df[list(reversed(df.columns))]
    df.to_csv(rpath, index=False)
    good_row = pd.DataFrame(
        [
            {
                "experiment_name": "exp2",
                "factor_name": "f2",
                "label_factor": "fwd_1",
                "quantiles": 5,
                "split_description": "full_sample",
                "cost_rate": float("nan"),
                "mean_ic": 0.05,
                "ic_ir": 1.2,
                "mean_long_short_return": 0.003,
                "mean_cost_adjusted_long_short_return": float("nan"),
                "timestamp": "2024-01-01T00:00:00",
                "obsidian_path": "",
            }
        ],
        columns=list(REGISTRY_COLUMNS),
    )
    with pytest.raises(ValueError, match="incompatible schema"):
        append_to_registry(good_row, rpath)


def test_append_to_registry_raises_on_extra_columns_in_existing_file(rpath: Path):
    """Appending to a file with extra (unexpected) columns raises ValueError."""
    register_experiment("exp1", _make_summary(), rpath)
    df = pd.read_csv(rpath)
    df["surprise_col"] = 99
    df.to_csv(rpath, index=False)
    good_row = pd.DataFrame(
        [
            {
                "experiment_name": "exp2",
                "factor_name": "f2",
                "label_factor": "fwd_1",
                "quantiles": 5,
                "split_description": "full_sample",
                "cost_rate": float("nan"),
                "mean_ic": 0.05,
                "ic_ir": 1.2,
                "mean_long_short_return": 0.003,
                "mean_cost_adjusted_long_short_return": float("nan"),
                "timestamp": "2024-01-01T00:00:00",
                "obsidian_path": "",
            }
        ],
        columns=list(REGISTRY_COLUMNS),
    )
    with pytest.raises(ValueError, match="incompatible schema"):
        append_to_registry(good_row, rpath)


def test_append_to_registry_rejects_non_dataframe(rpath: Path):
    with pytest.raises(TypeError):
        append_to_registry({"experiment_name": "e1"}, rpath)  # type: ignore[arg-type]


def test_append_to_registry_rejects_extra_columns_in_row(rpath: Path):
    """A row with extra columns beyond REGISTRY_COLUMNS must be rejected
    rather than silently truncated."""
    good_row = pd.DataFrame(
        [
            {
                "experiment_name": "exp1",
                "factor_name": "f1",
                "label_factor": "fwd_5",
                "quantiles": 5,
                "split_description": "full_sample",
                "cost_rate": float("nan"),
                "mean_ic": 0.05,
                "ic_ir": 1.2,
                "mean_long_short_return": 0.003,
                "mean_cost_adjusted_long_short_return": float("nan"),
                "timestamp": "2024-01-01T00:00:00",
                "obsidian_path": "",
                "surprise_extra_col": "should_not_pass",
            }
        ]
    )
    with pytest.raises(ValueError, match="unexpected columns"):
        append_to_registry(good_row, rpath)


# ---------------------------------------------------------------------------
# 8. Parent directory creation
# ---------------------------------------------------------------------------


def test_register_experiment_creates_parent_dirs(tmp_path: Path):
    nested = tmp_path / "a" / "b" / "c" / "registry.csv"
    assert not nested.parent.exists()
    register_experiment("exp1", _make_summary(), nested)
    assert nested.exists()
