from __future__ import annotations

import pandas as pd

from alpha_lab.research.deepdive_io import CaseArtifacts, save_deepdive


def test_save_deepdive_accepts_case_artifacts(tmp_path) -> None:
    artifacts = CaseArtifacts(
        case_dir=tmp_path / "case_a",
        factor_df=pd.DataFrame(),
        labels_df=pd.DataFrame(),
        prices=pd.DataFrame(),
        ic_df=pd.DataFrame(),
        long_short_df=pd.DataFrame(),
    )
    frame = pd.DataFrame({"bucket": ["Q1"], "ic": [0.1]})

    path = save_deepdive(frame, artifacts, "table.csv")

    assert path == artifacts.deepdive_dir / "table.csv"
    assert path.exists()
