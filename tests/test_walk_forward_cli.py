from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from alpha_lab.walk_forward_cli import main


def _write_prices_csv(path: Path, n_assets: int = 6, n_days: int = 80, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    rows = []
    for asset in [f"A{i}" for i in range(n_assets)]:
        price = 100.0
        for date in dates:
            price *= 1.0 + rng.normal(0.0, 0.01)
            rows.append({"date": date.strftime("%Y-%m-%d"), "asset": asset, "close": price})
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_walk_forward_cli_writes_aggregate_and_fold_csvs(tmp_path: Path) -> None:
    prices_csv = _write_prices_csv(tmp_path / "prices.csv")
    out_dir = tmp_path / "out"
    rc = main(
        [
            "--input-path", str(prices_csv),
            "--factor", "reversal",
            "--reversal-window", "5",
            "--label-horizon", "5",
            "--quantiles", "5",
            "--train-size", "30",
            "--test-size", "10",
            "--step", "10",
            "--output-dir", str(out_dir),
        ]
    )
    assert rc == 0
    assert (out_dir / "wf_reversal_h5_q5_aggregate.csv").exists()
    assert (out_dir / "wf_reversal_h5_q5_folds.csv").exists()


def test_walk_forward_cli_writes_obsidian_note(tmp_path: Path) -> None:
    prices_csv = _write_prices_csv(tmp_path / "prices.csv")
    note_path = tmp_path / "wf_note.md"
    main(
        [
            "--input-path", str(prices_csv),
            "--factor", "low_volatility",
            "--low-volatility-window", "20",
            "--label-horizon", "5",
            "--quantiles", "5",
            "--train-size", "30",
            "--test-size", "10",
            "--step", "10",
            "--output-dir", str(tmp_path / "out"),
            "--obsidian-markdown-path", str(note_path),
        ]
    )
    assert note_path.exists()
    assert "## Aggregate Results" in note_path.read_text(encoding="utf-8")
