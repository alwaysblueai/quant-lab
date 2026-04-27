from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _write_summary(
    run_dir: Path,
    *,
    run_id: str,
    total_wall_seconds: float,
    stage_seconds: dict[str, float],
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run_id,
        "size": "medium",
        "total_wall_seconds": total_wall_seconds,
        "stages": [
            {"name": stage, "wall_seconds": seconds}
            for stage, seconds in stage_seconds.items()
        ],
    }
    (run_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _script_path() -> Path:
    return Path(__file__).resolve().parents[1] / "scripts" / "ab_compare.py"


def test_ab_compare_outputs_and_verdict(tmp_path: Path) -> None:
    pre_root = tmp_path / "pre_group"
    post_root = tmp_path / "post_group"

    _write_summary(
        pre_root / "run_a",
        run_id="run_a",
        total_wall_seconds=20.0,
        stage_seconds={"load": 5.0, "train": 2.0},
    )
    _write_summary(
        pre_root / "run_b",
        run_id="run_b",
        total_wall_seconds=19.0,
        stage_seconds={"load": 5.0, "train": 2.0},
    )
    _write_summary(
        post_root / "run_c",
        run_id="run_c",
        total_wall_seconds=18.0,
        stage_seconds={"load": 4.5, "train": 1.0},
    )
    _write_summary(
        post_root / "run_d",
        run_id="run_d",
        total_wall_seconds=17.0,
        stage_seconds={"load": 4.5, "train": 1.0},
    )

    out_root = tmp_path / "out"
    out_dir = out_root / "fixed_dir"
    proc = subprocess.run(
        [
            sys.executable,
            str(_script_path()),
            "--pre",
            str(pre_root),
            "--post",
            str(post_root),
            "--stage",
            "train",
            "--sigma-k",
            "2",
            "--min-runs",
            "2",
            "--output-dir",
            str(out_dir),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr

    compare_json = out_dir / "compare.json"
    payload = json.loads(compare_json.read_text(encoding="utf-8"))
    assert payload["artifact_type"] == "alpha_lab_ab_compare"
    assert payload["min_runs"] == 2
    assert payload["stage_filter"] == "train"
    stage_rows = payload["stages"]
    assert len(stage_rows) == 1
    assert stage_rows[0]["stage"] == "train"
    assert stage_rows[0]["verdict"] == "faster_than_noise"


def test_ab_compare_rejects_when_below_min_runs(tmp_path: Path) -> None:
    pre_root = tmp_path / "pre_group"
    post_root = tmp_path / "post_group"

    _write_summary(
        pre_root / "run_a",
        run_id="run_a",
        total_wall_seconds=20.0,
        stage_seconds={"load": 5.0, "train": 2.0},
    )
    _write_summary(
        post_root / "run_b",
        run_id="run_b",
        total_wall_seconds=18.0,
        stage_seconds={"load": 4.5, "train": 1.5},
    )

    proc = subprocess.run(
        [
            sys.executable,
            str(_script_path()),
            "--pre",
            str(pre_root),
            "--post",
            str(post_root),
            "--stage",
            "train",
            "--min-runs",
            "2",
            "--output-root",
            str(tmp_path / "out"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode != 0
    assert "below min_runs=2" in (proc.stderr + proc.stdout)
