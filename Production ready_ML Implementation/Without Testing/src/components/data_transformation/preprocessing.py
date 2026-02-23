import pandas as pd
from datetime import datetime
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from src.config.config import PROCESSED_DATA_DIR

NUM_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
CAT_COLS = ["gender", "Partner", "Dependents", "PhoneService", "InternetService"]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, NUM_COLS),
    ("cat", categorical_pipeline, CAT_COLS)
])

def save_processed(df: pd.DataFrame):
    PROCESSED_DATA_DIR.mkdir(exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = PROCESSED_DATA_DIR / f"processed_{ts}.csv"
    df.to_csv(path, index=False)
    return path