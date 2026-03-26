from __future__ import annotations

import pandas as pd

from alpha_lab.exposure_audit import run_exposure_audit


def _weights_exposures() -> tuple[pd.DataFrame, pd.DataFrame]:
    weights = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-02")] * 3,
            "asset": ["A", "B", "C"],
            "weight": [0.4, -0.2, -0.2],
        }
    )
    exposures = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-02")] * 3,
            "asset": ["A", "B", "C"],
            "industry": ["Tech", "Tech", "Fin"],
            "size_exposure": [1.0, -1.0, 0.5],
            "beta_exposure": [1.2, 0.8, 1.0],
        }
    )
    return weights, exposures


def test_run_exposure_audit_outputs_expected_tables() -> None:
    weights, exposures = _weights_exposures()
    out = run_exposure_audit(weights, exposures)
    assert {"date", "industry", "net_weight"} == set(out.industry_exposure.columns)
    assert {"date", "exposure_name", "weighted_exposure"} == set(out.style_exposure.columns)
    assert "max_abs_industry_exposure" in out.summary.columns

