from alpha_lab.config import DATA_DIR, DATA_ROOT, PROCESSED_DATA_DIR, PROJECT_ROOT, RAW_DATA_DIR
from alpha_lab.factors import momentum
from alpha_lab.interfaces import FACTOR_OUTPUT_COLUMNS


def test_package_contracts_exist():
    assert PROJECT_ROOT.exists()
    assert DATA_DIR == PROJECT_ROOT / "data"
    assert RAW_DATA_DIR == DATA_DIR / "raw"
    assert PROCESSED_DATA_DIR == DATA_DIR / "processed"
    assert DATA_ROOT.is_absolute()
    assert tuple(FACTOR_OUTPUT_COLUMNS) == ("date", "asset", "factor", "value")
    assert callable(momentum)
