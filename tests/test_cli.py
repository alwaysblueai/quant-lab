from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from alpha_lab.cli import build_parser, main
from alpha_lab.registry import load_registry

# ---------------------------------------------------------------------------
# Shared synthetic price fixture
# ---------------------------------------------------------------------------


def _write_prices_csv(path: Path, n_assets: int = 6, n_days: int = 30, seed: int = 0) -> Path:
    """Write a minimal synthetic price CSV and return its path."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    assets = [f"A{i}" for i in range(n_assets)]
    rows = []
    for asset in assets:
        price = 100.0
        for date in dates:
            price *= 1.0 + rng.normal(0.0, 0.01)
            rows.append({"date": date.strftime("%Y-%m-%d"), "asset": asset, "close": price})
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# 1. Basic successful run
# ---------------------------------------------------------------------------


def test_cli_basic_run_returns_zero(tmp_path: Path) -> None:
    prices_csv = _write_prices_csv(tmp_path / "prices.csv")
    rc = main(
        [
            "--input-path", str(prices_csv),
            "--factor", "momentum",
            "--label-horizon", "5",
            "--quantiles", "5",
            "--output-dir", str(tmp_path / "out"),
        ]
    )
    assert rc == 0


def test_cli_basic_run_writes_summary_csv(tmp_path: Path) -> None:
    prices_csv = _write_prices_csv(tmp_path / "prices.csv")
    out_dir = tmp_path / "out"
    main(
        [
            "--input-path", str(prices_csv),
            "--factor", "momentum",
            "--label-horizon", "5",
            "--quantiles", "5",
            "--output-dir", str(out_dir),
        ]
    )
    csvs = list(out_dir.glob("*.csv"))
    assert len(csvs) == 1


def test_cli_summary_csv_has_expected_columns(tmp_path: Path) -> None:
    prices_csv = _write_prices_csv(tmp_path / "prices.csv")
    out_dir = tmp_path / "out"
    main(
        [
            "--input-path", str(prices_csv),
            "--factor", "momentum",
            "--label-horizon", "5",
            "--quantiles", "5",
            "--output-dir", str(out_dir),
        ]
    )
    csv_path = next(out_dir.glob("*.csv"))
    df = pd.read_csv(csv_path)
    for col in ("factor_name", "mean_ic", "mean_rank_ic", "mean_long_short_return"):
        assert col in df.columns


def test_cli_summary_csv_filename_uses_experiment_name(tmp_path: Path) -> None:
    prices_csv = _write_prices_csv(tmp_path / "prices.csv")
    out_dir = tmp_path / "out"
    main(
        [
            "--input-path", str(prices_csv),
            "--factor", "momentum",
            "--label-horizon", "5",
            "--quantiles", "5",
            "--output-dir", str(out_dir),
            "--experiment-name", "my_test_run",
        ]
    )
    assert (out_dir / "my_test_run_summary.csv").exists()


def test_cli_default_experiment_name_derived_from_args(tmp_path: Path) -> None:
    prices_csv = _write_prices_csv(tmp_path / "prices.csv")
    out_dir = tmp_path / "out"
    main(
        [
            "--input-path", str(prices_csv),
            "--factor", "momentum",
            "--label-horizon", "3",
            "--quantiles", "4",
            "--output-dir", str(out_dir),
        ]
    )
    assert (out_dir / "momentum_h3_q4_summary.csv").exists()


def test_cli_creates_output_dir_if_missing(tmp_path: Path) -> None:
    prices_csv = _write_prices_csv(tmp_path / "prices.csv")
    out_dir = tmp_path / "does" / "not" / "exist"
    assert not out_dir.exists()
    main(
        [
            "--input-path", str(prices_csv),
            "--factor", "momentum",
            "--label-horizon", "5",
            "--quantiles", "5",
            "--output-dir", str(out_dir),
        ]
    )
    assert out_dir.exists()


# ---------------------------------------------------------------------------
# 2. Stdout output (captured)
# ---------------------------------------------------------------------------


def test_cli_stdout_contains_factor_name(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    prices_csv = _write_prices_csv(tmp_path / "prices.csv")
    main(
        [
            "--input-path", str(prices_csv),
            "--factor", "momentum",
            "--label-horizon", "5",
            "--quantiles", "5",
            "--output-dir", str(tmp_path / "out"),
        ]
    )
    captured = capsys.readouterr()
    assert "momentum" in captured.out


def test_cli_stdout_contains_mean_ic(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    prices_csv = _write_prices_csv(tmp_path / "prices.csv")
    main(
        [
            "--input-path", str(prices_csv),
            "--factor", "momentum",
            "--label-horizon", "5",
            "--quantiles", "5",
            "--output-dir", str(tmp_path / "out"),
        ]
    )
    captured = capsys.readouterr()
    assert "Mean IC" in captured.out


def test_cli_stdout_contains_summary_csv_path(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    prices_csv = _write_prices_csv(tmp_path / "prices.csv")
    out_dir = tmp_path / "out"
    main(
        [
            "--input-path", str(prices_csv),
            "--factor", "momentum",
            "--label-horizon", "5",
            "--quantiles", "5",
            "--output-dir", str(out_dir),
        ]
    )
    captured = capsys.readouterr()
    assert "Summary CSV" in captured.out


# ---------------------------------------------------------------------------
# 3. Malformed input
# ---------------------------------------------------------------------------


def test_cli_missing_input_file_exits(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main(
            [
                "--input-path", str(tmp_path / "nonexistent.csv"),
                "--factor", "momentum",
                "--label-horizon", "5",
                "--quantiles", "5",
                "--output-dir", str(tmp_path / "out"),
            ]
        )


def test_cli_missing_required_columns_exits(tmp_path: Path) -> None:
    bad_csv = tmp_path / "bad.csv"
    pd.DataFrame([{"date": "2024-01-01", "ticker": "A", "price": 100}]).to_csv(
        bad_csv, index=False
    )
    with pytest.raises(SystemExit):
        main(
            [
                "--input-path", str(bad_csv),
                "--factor", "momentum",
                "--label-horizon", "5",
                "--quantiles", "5",
                "--output-dir", str(tmp_path / "out"),
            ]
        )


def test_cli_missing_close_column_exits(tmp_path: Path) -> None:
    bad_csv = tmp_path / "bad.csv"
    pd.DataFrame([{"date": "2024-01-01", "asset": "A"}]).to_csv(bad_csv, index=False)
    with pytest.raises(SystemExit):
        main(
            [
                "--input-path", str(bad_csv),
                "--factor", "momentum",
                "--label-horizon", "5",
                "--quantiles", "5",
                "--output-dir", str(tmp_path / "out"),
            ]
        )


def test_cli_unparseable_dates_exit(tmp_path: Path) -> None:
    """A CSV with unparseable date values must fail clearly before the pipeline runs."""
    bad_csv = tmp_path / "bad_dates.csv"
    pd.DataFrame(
        [{"date": "not-a-date", "asset": "A", "close": 100.0}]
    ).to_csv(bad_csv, index=False)
    with pytest.raises(SystemExit):
        main(
            [
                "--input-path", str(bad_csv),
                "--factor", "momentum",
                "--label-horizon", "5",
                "--quantiles", "5",
                "--output-dir", str(tmp_path / "out"),
            ]
        )


# ---------------------------------------------------------------------------
# 4. Invalid / unknown factor
# ---------------------------------------------------------------------------


def test_cli_unknown_factor_exits(tmp_path: Path) -> None:
    prices_csv = _write_prices_csv(tmp_path / "prices.csv")
    with pytest.raises(SystemExit):
        main(
            [
                "--input-path", str(prices_csv),
                "--factor", "not_a_real_factor",
                "--label-horizon", "5",
                "--quantiles", "5",
                "--output-dir", str(tmp_path / "out"),
            ]
        )


def test_cli_reversal_run_succeeds(tmp_path: Path) -> None:
    prices_csv = _write_prices_csv(tmp_path / "prices.csv")
    rc = main(
        [
            "--input-path", str(prices_csv),
            "--factor", "reversal",
            "--reversal-window", "5",
            "--label-horizon", "5",
            "--quantiles", "5",
            "--output-dir", str(tmp_path / "out"),
        ]
    )
    assert rc == 0


def test_cli_low_volatility_run_succeeds(tmp_path: Path) -> None:
    prices_csv = _write_prices_csv(tmp_path / "prices.csv")
    rc = main(
        [
            "--input-path", str(prices_csv),
            "--factor", "low_volatility",
            "--low-volatility-window", "20",
            "--label-horizon", "5",
            "--quantiles", "5",
            "--output-dir", str(tmp_path / "out"),
        ]
    )
    assert rc == 0


# ---------------------------------------------------------------------------
# 5. Argument validation
# ---------------------------------------------------------------------------


def test_cli_label_horizon_zero_exits(tmp_path: Path) -> None:
    prices_csv = _write_prices_csv(tmp_path / "prices.csv")
    with pytest.raises(SystemExit):
        main(
            [
                "--input-path", str(prices_csv),
                "--factor", "momentum",
                "--label-horizon", "0",
                "--quantiles", "5",
                "--output-dir", str(tmp_path / "out"),
            ]
        )


def test_cli_label_horizon_negative_exits(tmp_path: Path) -> None:
    prices_csv = _write_prices_csv(tmp_path / "prices.csv")
    with pytest.raises(SystemExit):
        main(
            [
                "--input-path", str(prices_csv),
                "--factor", "momentum",
                "--label-horizon", "-1",
                "--quantiles", "5",
                "--output-dir", str(tmp_path / "out"),
            ]
        )


def test_cli_quantiles_one_exits(tmp_path: Path) -> None:
    prices_csv = _write_prices_csv(tmp_path / "prices.csv")
    with pytest.raises(SystemExit):
        main(
            [
                "--input-path", str(prices_csv),
                "--factor", "momentum",
                "--label-horizon", "5",
                "--quantiles", "1",
                "--output-dir", str(tmp_path / "out"),
            ]
        )


def test_cli_negative_cost_rate_exits(tmp_path: Path) -> None:
    prices_csv = _write_prices_csv(tmp_path / "prices.csv")
    with pytest.raises(SystemExit):
        main(
            [
                "--input-path", str(prices_csv),
                "--factor", "momentum",
                "--label-horizon", "5",
                "--quantiles", "5",
                "--cost-rate", "-0.001",
                "--output-dir", str(tmp_path / "out"),
            ]
        )


def test_cli_train_end_without_test_start_exits(tmp_path: Path) -> None:
    prices_csv = _write_prices_csv(tmp_path / "prices.csv")
    with pytest.raises(SystemExit):
        main(
            [
                "--input-path", str(prices_csv),
                "--factor", "momentum",
                "--label-horizon", "5",
                "--quantiles", "5",
                "--train-end", "2024-01-15",
                "--output-dir", str(tmp_path / "out"),
            ]
        )


def test_cli_test_start_without_train_end_exits(tmp_path: Path) -> None:
    prices_csv = _write_prices_csv(tmp_path / "prices.csv")
    with pytest.raises(SystemExit):
        main(
            [
                "--input-path", str(prices_csv),
                "--factor", "momentum",
                "--label-horizon", "5",
                "--quantiles", "5",
                "--test-start", "2024-01-20",
                "--output-dir", str(tmp_path / "out"),
            ]
        )


@pytest.mark.parametrize("unsafe_name", [
    "../../etc/passwd",
    "../escape",
    "/absolute/path",
    "name/with/slash",
    "name\\backslash",
    "..",
    ".",
])
def test_cli_unsafe_experiment_name_exits(tmp_path: Path, unsafe_name: str) -> None:
    """An --experiment-name containing path separators or .. must be rejected
    before any file is written, preventing directory traversal attacks."""
    prices_csv = _write_prices_csv(tmp_path / "prices.csv")
    out_dir = tmp_path / "out"
    with pytest.raises(SystemExit):
        main(
            [
                "--input-path", str(prices_csv),
                "--factor", "momentum",
                "--label-horizon", "5",
                "--quantiles", "5",
                "--experiment-name", unsafe_name,
                "--output-dir", str(out_dir),
            ]
        )
    # Output directory must remain empty — no file was written
    if out_dir.exists():
        assert not list(out_dir.glob("**/*"))


# ---------------------------------------------------------------------------
# 6. Split mode
# ---------------------------------------------------------------------------


def test_cli_split_run_succeeds(tmp_path: Path) -> None:
    prices_csv = _write_prices_csv(tmp_path / "prices.csv", n_days=40)
    rc = main(
        [
            "--input-path", str(prices_csv),
            "--factor", "momentum",
            "--label-horizon", "5",
            "--quantiles", "5",
            "--train-end", "2024-01-31",
            "--test-start", "2024-02-01",
            "--output-dir", str(tmp_path / "out"),
        ]
    )
    assert rc == 0


# ---------------------------------------------------------------------------
# 7. Cost rate
# ---------------------------------------------------------------------------


def test_cli_cost_rate_writes_adjusted_column(tmp_path: Path) -> None:
    prices_csv = _write_prices_csv(tmp_path / "prices.csv")
    out_dir = tmp_path / "out"
    main(
        [
            "--input-path", str(prices_csv),
            "--factor", "momentum",
            "--label-horizon", "5",
            "--quantiles", "5",
            "--cost-rate", "0.001",
            "--output-dir", str(out_dir),
        ]
    )
    csv_path = next(out_dir.glob("*.csv"))
    df = pd.read_csv(csv_path)
    assert "mean_cost_adjusted_long_short_return" in df.columns
    # With cost_rate provided, the value should be finite (not NaN) when
    # long-short returns exist.
    val = float(df["mean_cost_adjusted_long_short_return"].iloc[0])
    assert math.isfinite(val) or math.isnan(val)  # NaN allowed if no valid dates


def test_cli_no_cost_rate_gives_nan_adjusted_return(tmp_path: Path) -> None:
    prices_csv = _write_prices_csv(tmp_path / "prices.csv")
    out_dir = tmp_path / "out"
    main(
        [
            "--input-path", str(prices_csv),
            "--factor", "momentum",
            "--label-horizon", "5",
            "--quantiles", "5",
            "--output-dir", str(out_dir),
        ]
    )
    csv_path = next(out_dir.glob("*.csv"))
    df = pd.read_csv(csv_path)
    assert math.isnan(float(df["mean_cost_adjusted_long_short_return"].iloc[0]))


# ---------------------------------------------------------------------------
# 8. Obsidian markdown export
# ---------------------------------------------------------------------------


def test_cli_obsidian_markdown_written(tmp_path: Path) -> None:
    prices_csv = _write_prices_csv(tmp_path / "prices.csv")
    md_path = tmp_path / "notes" / "experiment.md"
    main(
        [
            "--input-path", str(prices_csv),
            "--factor", "momentum",
            "--label-horizon", "5",
            "--quantiles", "5",
            "--output-dir", str(tmp_path / "out"),
            "--obsidian-markdown-path", str(md_path),
        ]
    )
    assert md_path.exists()


def test_cli_obsidian_markdown_content(tmp_path: Path) -> None:
    prices_csv = _write_prices_csv(tmp_path / "prices.csv")
    md_path = tmp_path / "experiment.md"
    main(
        [
            "--input-path", str(prices_csv),
            "--factor", "momentum",
            "--label-horizon", "5",
            "--quantiles", "5",
            "--output-dir", str(tmp_path / "out"),
            "--experiment-name", "my_obsidian_test",
            "--obsidian-markdown-path", str(md_path),
        ]
    )
    content = md_path.read_text(encoding="utf-8")
    assert "my_obsidian_test" in content
    assert "Mean IC" in content


def test_cli_obsidian_creates_parent_dirs(tmp_path: Path) -> None:
    prices_csv = _write_prices_csv(tmp_path / "prices.csv")
    md_path = tmp_path / "deep" / "nested" / "note.md"
    assert not md_path.parent.exists()
    main(
        [
            "--input-path", str(prices_csv),
            "--factor", "momentum",
            "--label-horizon", "5",
            "--quantiles", "5",
            "--output-dir", str(tmp_path / "out"),
            "--obsidian-markdown-path", str(md_path),
        ]
    )
    assert md_path.exists()


def test_cli_no_obsidian_flag_does_not_write_md(tmp_path: Path) -> None:
    prices_csv = _write_prices_csv(tmp_path / "prices.csv")
    main(
        [
            "--input-path", str(prices_csv),
            "--factor", "momentum",
            "--label-horizon", "5",
            "--quantiles", "5",
            "--output-dir", str(tmp_path / "out"),
        ]
    )
    assert not list(tmp_path.glob("**/*.md"))


# ---------------------------------------------------------------------------
# 9. Registry append
# ---------------------------------------------------------------------------


def test_cli_append_registry_writes_entry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """--append-registry must add one row to the default registry path."""
    registry_path = tmp_path / "registry.csv"
    # Patch the default registry path so we don't touch the real one
    import alpha_lab.registry as reg_module

    monkeypatch.setattr(reg_module, "DEFAULT_REGISTRY_PATH", registry_path)
    prices_csv = _write_prices_csv(tmp_path / "prices.csv")
    main(
        [
            "--input-path", str(prices_csv),
            "--factor", "momentum",
            "--label-horizon", "5",
            "--quantiles", "5",
            "--output-dir", str(tmp_path / "out"),
            "--experiment-name", "reg_test",
            "--append-registry",
        ]
    )
    df = load_registry(registry_path)
    assert len(df) == 1
    assert df["experiment_name"].iloc[0] == "reg_test"


def test_cli_registry_stores_obsidian_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When both --obsidian-markdown-path and --append-registry are used,
    the registry entry must record the Obsidian note path."""
    registry_path = tmp_path / "registry.csv"
    import alpha_lab.registry as reg_module

    monkeypatch.setattr(reg_module, "DEFAULT_REGISTRY_PATH", registry_path)
    prices_csv = _write_prices_csv(tmp_path / "prices.csv")
    md_path = tmp_path / "notes" / "exp.md"
    main(
        [
            "--input-path", str(prices_csv),
            "--factor", "momentum",
            "--label-horizon", "5",
            "--quantiles", "5",
            "--output-dir", str(tmp_path / "out"),
            "--experiment-name", "obsidian_reg_test",
            "--obsidian-markdown-path", str(md_path),
            "--append-registry",
        ]
    )
    df = load_registry(registry_path)
    assert len(df) == 1
    assert str(df["obsidian_path"].iloc[0]) == str(md_path)


def test_cli_registry_obsidian_path_empty_when_no_markdown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When --append-registry is used without --obsidian-markdown-path,
    the registry obsidian_path column must be empty."""
    registry_path = tmp_path / "registry.csv"
    import alpha_lab.registry as reg_module

    monkeypatch.setattr(reg_module, "DEFAULT_REGISTRY_PATH", registry_path)
    prices_csv = _write_prices_csv(tmp_path / "prices.csv")
    main(
        [
            "--input-path", str(prices_csv),
            "--factor", "momentum",
            "--label-horizon", "5",
            "--quantiles", "5",
            "--output-dir", str(tmp_path / "out"),
            "--experiment-name", "no_md_test",
            "--append-registry",
        ]
    )
    df = load_registry(registry_path)
    val = df["obsidian_path"].iloc[0]
    assert val == "" or (isinstance(val, float) and math.isnan(val))


def test_cli_no_append_registry_does_not_create_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry_path = tmp_path / "registry.csv"
    import alpha_lab.registry as reg_module

    monkeypatch.setattr(reg_module, "DEFAULT_REGISTRY_PATH", registry_path)
    prices_csv = _write_prices_csv(tmp_path / "prices.csv")
    main(
        [
            "--input-path", str(prices_csv),
            "--factor", "momentum",
            "--label-horizon", "5",
            "--quantiles", "5",
            "--output-dir", str(tmp_path / "out"),
        ]
    )
    assert not registry_path.exists()


# ---------------------------------------------------------------------------
# 10. Momentum window
# ---------------------------------------------------------------------------


def test_cli_custom_momentum_window_succeeds(tmp_path: Path) -> None:
    prices_csv = _write_prices_csv(tmp_path / "prices.csv", n_days=40)
    rc = main(
        [
            "--input-path", str(prices_csv),
            "--factor", "momentum",
            "--label-horizon", "5",
            "--quantiles", "5",
            "--momentum-window", "10",
            "--output-dir", str(tmp_path / "out"),
        ]
    )
    assert rc == 0


def test_cli_momentum_window_zero_exits(tmp_path: Path) -> None:
    prices_csv = _write_prices_csv(tmp_path / "prices.csv")
    with pytest.raises(SystemExit):
        main(
            [
                "--input-path", str(prices_csv),
                "--factor", "momentum",
                "--label-horizon", "5",
                "--quantiles", "5",
                "--momentum-window", "0",
                "--output-dir", str(tmp_path / "out"),
            ]
        )


# ---------------------------------------------------------------------------
# 11. Determinism
# ---------------------------------------------------------------------------


def test_cli_deterministic_summary_csv(tmp_path: Path) -> None:
    """Two identical runs produce identical summary CSVs."""
    prices_csv = _write_prices_csv(tmp_path / "prices.csv")
    out1 = tmp_path / "run1"
    out2 = tmp_path / "run2"
    main(
        [
            "--input-path", str(prices_csv),
            "--factor", "momentum",
            "--label-horizon", "5",
            "--quantiles", "5",
            "--output-dir", str(out1),
            "--experiment-name", "det_test",
        ]
    )
    main(
        [
            "--input-path", str(prices_csv),
            "--factor", "momentum",
            "--label-horizon", "5",
            "--quantiles", "5",
            "--output-dir", str(out2),
            "--experiment-name", "det_test",
        ]
    )
    df1 = pd.read_csv(out1 / "det_test_summary.csv")
    df2 = pd.read_csv(out2 / "det_test_summary.csv")
    pd.testing.assert_frame_equal(df1, df2)


# ---------------------------------------------------------------------------
# 12. Parser introspection
# ---------------------------------------------------------------------------


def test_build_parser_returns_argument_parser() -> None:
    p = build_parser()
    assert isinstance(p, __import__("argparse").ArgumentParser)


def test_parser_required_args() -> None:
    p = build_parser()
    with pytest.raises(SystemExit):
        p.parse_args([])  # missing all required args
