from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = BASE_DIR / "data/raw"
VALIDATED_DATA_DIR = BASE_DIR / "data/validated"
PROCESSED_DATA_DIR = BASE_DIR / "data/processed"

MODEL_PATH = BASE_DIR / "models/churn_pipeline.joblib"
DRIFT_REPORT_PATH = BASE_DIR / "reports/drift_report.json"

TARGET_COL = "Churn"
RANDOM_STATE = 42