from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"

MODELS_DIR = PROJECT_ROOT / "models" / "artifacts"
REPORTS_DIR = PROJECT_ROOT / "reports"

RAW_FILE = RAW_DIR / "readmission.csv"
CLEAN_FILE = PROCESSED_DIR / "clean.csv"

FEATURE_FILE = FEATURES_DIR / "features.parquet"
TRAIN_FILE = FEATURES_DIR / "train.parquet"
TEST_FILE = FEATURES_DIR / "test.parquet"

METRICS_FILE = REPORTS_DIR / "metrics.json"
BEST_MODEL_FILE = REPORTS_DIR / "best_model.json"
