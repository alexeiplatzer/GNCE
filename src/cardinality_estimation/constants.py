from pathlib import Path

PROJECT_ROOT_PATH = Path(__file__).parent.parent.parent  # should be the parent of src
DATASETS_PATH = PROJECT_ROOT_PATH.parent / "Datasets"
PACKAGE_PATH = PROJECT_ROOT_PATH / "src" / "cardinality_estimation"
GNCE_PATH = PACKAGE_PATH / "GNCE"

DATASETS_PATH.mkdir(parents=True, exist_ok=True)
