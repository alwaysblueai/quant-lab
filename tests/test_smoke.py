from alpha_lab.config import PROJECT_ROOT, DATA_DIR

def test_paths_exist():
    assert PROJECT_ROOT.exists()
    assert DATA_DIR.exists()
