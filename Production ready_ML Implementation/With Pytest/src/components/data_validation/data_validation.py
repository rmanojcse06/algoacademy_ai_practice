import pandas as pd
from pathlib import Path
from src.schema.schema import SCHEMA
from src.config.config import VALIDATED_DATA_DIR, TARGET_COL

def validate_and_save(df: pd.DataFrame, filename: str):
    missing_cols = set(SCHEMA) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    if TARGET_COL not in df.columns:
        raise ValueError("Target column missing")

    VALIDATED_DATA_DIR.mkdir(exist_ok=True)

    output_path = VALIDATED_DATA_DIR / filename
    df.to_csv(output_path, index=False)
    return output_path