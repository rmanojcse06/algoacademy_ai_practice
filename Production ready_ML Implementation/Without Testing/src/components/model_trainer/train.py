import pandas as pd
import joblib
import logging
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.config.config import RAW_DATA_DIR, MODEL_PATH, TARGET_COL, RANDOM_STATE
from src.components.data_validation.data_validation import validate_and_save
from src.components.data_transformation.preprocessing import (
    preprocessor, save_processed
)

logger = logging.getLogger(__name__)

def train():
    raw_file = next(RAW_DATA_DIR.glob("*.csv"))
    df = pd.read_csv(raw_file)

    validated_path = validate_and_save(df, raw_file.name)
    logger.info("Validated data saved to %s", validated_path)

    processed_path = save_processed(df)
    logger.info("Processed data snapshot saved to %s", processed_path)

    df[TARGET_COL] = df[TARGET_COL].map({"Yes": 1, "No": 0})
    X = df.drop(TARGET_COL, axis=1)
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)
    joblib.dump(pipeline, MODEL_PATH)

    return pipeline, X_test, y_test, processed_path